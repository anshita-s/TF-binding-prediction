from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import pandas as pd
from pyfaidx import Fasta
from collections import Counter
import math
import random
import time
import os


# TSV File + sequence fetching

def add_dna_sequence_column(tsv_path, fasta_dir, chrom_name, out_path):
    df = pd.read_csv(tsv_path, sep="\t")

    start_col, end_col = "start", "end"
    dna_col = "DNA sequence"

    genes = Fasta(f"{fasta_dir}/{chrom_name}.fa")

    def fetch_seq(row):
        s = int(row[start_col])
        e = int(row[end_col])
        return str(genes[chrom_name][s:e])

    df[dna_col] = df.apply(fetch_seq, axis=1)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"Added '{dna_col}' for {len(df)} rows — saved to: {out_path}")
    return out_path


def split_sequences_by_tf(dnaseq_tsv_path, tf_name):
    df = pd.read_csv(dnaseq_tsv_path, sep="\t")

    tf_col_map = {"CTCF": "CTCF", "REST": "REST", "EP300": "EP300"}
    dna_col = "DNA sequence"
    tf_col = tf_col_map[tf_name]

    bound, unbound = [], []
    for _, row in df.iterrows():
        if row[tf_col] == "B":
            bound.append(row[dna_col])
        else:
            unbound.append(row[dna_col])

    # normalize strings
    bound = [s.strip().upper() for s in bound if isinstance(s, str)]
    unbound = [s.strip().upper() for s in unbound if isinstance(s, str)]
    if len(bound) == 0 or len(unbound) == 0:
        raise ValueError("Bound or unbound sequence list is empty for the chosen TF.")
    return bound, unbound



# Markov model building and scoring

BASE_NAME = ("A", "C", "G", "T")

def k_fold_split(seq_list, k):
    arr = seq_list.copy()
    random.shuffle(arr)
    n = len(arr)
    sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    folds, idx = [], 0
    for sz in sizes:
        folds.append(arr[idx: idx + sz])
        idx += sz
    return folds


def build_counts(seqs, m):
    counts = {}
    for s in seqs:
        for i in range(m, len(s)):
            ctx = s[i - m:i] if m > 0 else ""
            base = s[i]
            if (len(ctx) != m) or (base not in "ACGT") or any(ch not in "ACGT" for ch in ctx):
                continue
            counts.setdefault(ctx, Counter())[base] += 1
    return counts


def build_probs(counts, pseudocount):
    probs = {}
    for ctx, ctr in counts.items():
        total = sum(ctr.get(b, 0) for b in BASE_NAME) + 4 * pseudocount
        probs[ctx] = {b: (ctr.get(b, 0) + pseudocount) / total for b in BASE_NAME}
    return probs


def score_log_odds(seq, m, probs_b, probs_u):
    s = 0.0
    for i in range(m, len(seq)):
        ctx = seq[i - m:i] if m > 0 else ""
        base = seq[i]
        if (len(ctx) != m) or (base not in "ACGT") or any(ch not in "ACGT" for ch in ctx):
            continue

        pb_ctx = probs_b.get(ctx)
        pu_ctx = probs_u.get(ctx)

        pb = (1.0 / 4.0) if pb_ctx is None else pb_ctx[base]
        pu = (1.0 / 4.0) if pu_ctx is None else pu_ctx[base]

        s += math.log(pb) - math.log(pu)
    return s



# 3) Train + plotting

def run_markov_cv_and_plot(pos, neg, tf="CTCF", k=5, pseudocount=0.5, m=3):
    # enforce feasible k
    max_k = min(len(pos), len(neg))
    if k > max_k:
        print(f"Warning: requested k={k} too large for class sizes; reducing k -> {max_k}")
        k = max_k
    if k < 2:
        raise ValueError("k must be >= 2")

    # split folds
    folds_pos = k_fold_split(pos, k)
    folds_neg = k_fold_split(neg, k)

    # plot figures
    roc_fig, roc_ax = plt.subplots(constrained_layout=True)
    pr_fig, pr_ax = plt.subplots(constrained_layout=True)

    roc_ax.set_title(f"ROC curve — TF={tf}  k={k}  m={m}")
    roc_ax.set_xlabel("False Positive Rate")
    roc_ax.set_ylabel("True Positive Rate")
    roc_ax.grid(True, linewidth=0.6, alpha=0.5)
    roc_ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="random")

    pr_ax.set_title(f"Precision-Recall — TF={tf}  k={k}  m={m}")
    pr_ax.set_xlabel("Recall")
    pr_ax.set_ylabel("Precision")
    pr_ax.grid(True, linewidth=0.6, alpha=0.5)

    fold_times = []
    fold_aucs = []
    fold_aps = []

    print(f"\nTF={tf} | order m={m} | k={k} folds")
    print(f"Positive (bound) count: {len(pos)} | Negative (unbound) count: {len(neg)}\n")

    t0_all = time.perf_counter()

    for fold_idx in range(k):
        t0 = time.perf_counter()

        # train = all folds except fold_idx
        train_pos = [s for i, f in enumerate(folds_pos) if i != fold_idx for s in f]
        train_neg = [s for i, f in enumerate(folds_neg) if i != fold_idx for s in f]
        test_pos  = folds_pos[fold_idx]
        test_neg  = folds_neg[fold_idx]

        # basic length filtering
        train_pos = [s for s in train_pos if isinstance(s, str) and len(s) >= m + 1]
        train_neg = [s for s in train_neg if isinstance(s, str) and len(s) >= m + 1]
        test_pos  = [s for s in test_pos  if isinstance(s, str) and len(s) >= m + 1]
        test_neg  = [s for s in test_neg  if isinstance(s, str) and len(s) >= m + 1]

        # train fold model
        counts_b = build_counts(train_pos, m)
        counts_u = build_counts(train_neg, m)
        probs_b  = build_probs(counts_b, pseudocount)
        probs_u  = build_probs(counts_u, pseudocount)

        # fold scores (continuous) + labels
        y_true, y_score = [], []
        for s in test_pos:
            y_true.append(1)
            y_score.append(score_log_odds(s, m, probs_b, probs_u))
        for s in test_neg:
            y_true.append(0)
            y_score.append(score_log_odds(s, m, probs_b, probs_u))

        # ROC for this fold
        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)

        # PR for this fold
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        pr_auc = auc(recall, precision)  # area under PR curve 

        t1 = time.perf_counter()

        fold_times.append(t1 - t0)
        fold_aucs.append(roc_auc)
        fold_aps.append(pr_auc)

        # plot this fold
        roc_ax.plot(fpr, tpr, lw=1.8, label=f"Fold {fold_idx+1}: AUC={roc_auc:.3f}")
        pr_ax.plot(recall, precision, lw=1.8, label=f"Fold {fold_idx+1}: AP={pr_auc:.3f}")

        print(f"Fold {fold_idx+1}/{k}: ROC AUC = {roc_auc:.4f} | PR AUC = {pr_auc:.4f} | time = {fold_times[-1]:.3f}s")

    # PR baseline (class prevalence)
    baseline = 1.0 * len(pos) / (len(pos) + len(neg))
    pr_ax.hlines(baseline, 0, 1, linestyle="--", color="gray", label=f"baseline={baseline:.3f}")

    roc_ax.legend(loc="lower right", fontsize=9)
    pr_ax.legend(loc="upper right", fontsize=9)

    t_all = time.perf_counter() - t0_all
    print("\nSummary:")
    print(f"  Mean ROC AUC: {sum(fold_aucs)/k:.3f}")
    print(f"  Mean PR  AUC: {sum(fold_aps)/k:.3f}")
    print(f"  Total time:   {t_all:.3f}s | Mean per fold: {sum(fold_times)/k:.3f}s")

    roc_fig.savefig(f"ROC_curve_{tf}_k{k}_m{m}_allfolds.png", dpi=300)
    pr_fig.savefig(f"PR_curve_{tf}_k{k}_m{m}_allfolds.png", dpi=300)

    plt.show()

#Main code

chrom_number = input("Enter chromosome number (except 3,10,17,x,y): ")
tf = input("Choose TF (CTCF, REST, EP300): ")
m = int(input("Markov model order m: "))
k = int(input("k-fold value (>=2): "))

chrom_name = f"chr{chrom_number}"


fasta_dir = f"data/{chrom_name}.fa"
tsv_path = f"data/{chrom_name}.fa/{chrom_name}_200bp_bins.tsv"
dnaseq_out = f"data/{chrom_name}.fa/{chrom_name}_DNA_sequence.tsv"

dnaseq_path = add_dna_sequence_column(
    tsv_path,
    fasta_dir,
    chrom_name,
    dnaseq_out
)

pos, neg = split_sequences_by_tf(dnaseq_path, tf)

run_markov_cv_and_plot(pos, neg, tf=tf, k=k, pseudocount=0.5, m=m)



