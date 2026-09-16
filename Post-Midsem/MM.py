import numpy as np
import pandas as pd
from pyfaidx import Fasta
import matplotlib.pyplot as plt



#  This function loads the data, filters for the specified TF, and extracts sequences and labels


def load_data(tf_name, tsv_file, genome_fasta):

    tf_name = tf_name.upper()
    df = pd.read_csv(tsv_file, sep="\t")

    if tf_name not in df.columns:
        raise ValueError(f"{tf_name} not found in file")

    mask = df[tf_name].isin(["B", "U"])
    df = df[mask]

    labels = np.where(df[tf_name].values == "B", 1, 0)

    genome = Fasta(genome_fasta)

    sequences = []
    valid_labels = []

    for c, s, e, lab in zip(df["chr"], df["start"], df["end"], labels):
        seq = genome[c][s:e].seq.upper()
        if "N" not in seq:
            sequences.append(seq)
            valid_labels.append(lab)

    return np.array(sequences), np.array(valid_labels)



#  this function encodes DNA sequences into integer arrays (A=0, C=1, G=2 and T=3) using numpy, for efficient processing in the Markov model scoring


def encode_sequences(sequences):

    mapping = np.zeros(256, dtype=np.int8)
    mapping[ord('A')] = 0
    mapping[ord('C')] = 1
    mapping[ord('G')] = 2
    mapping[ord('T')] = 3

    seq_array = np.frombuffer(
        ''.join(sequences).encode('ascii'),
        dtype=np.uint8
    )

    seq_length = len(sequences[0])
    encoded = mapping[seq_array].reshape(len(sequences), seq_length)

    return encoded



# this function creates the stratified k-fold splits, ensuring each fold has bound and unbound sequences to train and test the markov model


def create_stratified_folds(labels, k, seed=42):

    rng = np.random.default_rng(seed)

    pos_idx = np.where(labels == 1)[0] #contains B sequences
    neg_idx = np.where(labels == 0)[0] #contains U sequences

    rng.shuffle(pos_idx) #shuffles B sequences
    rng.shuffle(neg_idx) #shuffles U sequences

    pos_folds = np.array_split(pos_idx, k)
    neg_folds = np.array_split(neg_idx, k)

    folds = []

    for i in range(k):
        test_idx = np.concatenate([pos_folds[i], neg_folds[i]])
        train_idx = np.setdiff1d(np.arange(len(labels)), test_idx)
        folds.append((train_idx, test_idx))

    return folds



# this function builds the markov model, taking the order m as an input, and calculates the probabilities of each base given the previous m bases

def build_markov_model(encoded_sequences, m):

    n_sequences, seq_length = encoded_sequences.shape

    if m == 0:
        counts = np.ones((1, 4))
        bases = encoded_sequences.ravel()
        counts[0] += np.bincount(bases, minlength=4)

    else:
        n_contexts = 4 ** m
        counts = np.ones((n_contexts, 4))

        powers = 4 ** np.arange(m-1, -1, -1)

        contexts = np.lib.stride_tricks.sliding_window_view(
            encoded_sequences, window_shape=m, axis=1
        )

        contexts = contexts[:, :-1, :]
        context_indices = np.tensordot(contexts, powers, axes=([2], [0]))

        next_bases = encoded_sequences[:, m:]

        np.add.at(
            counts,
            (context_indices.ravel(), next_bases.ravel()),
            1
        )

    probs = counts / counts.sum(axis=1, keepdims=True)
    return probs



# this function scores the sequences using the log-odds ratio of the probabilities from the 2 previous markov models, summing over those for the entire sequence to generate the final score


def score_sequences(encoded_sequences, probs_B, probs_U, m):

    if m == 0:
        log_ratio = np.log(probs_B[0]) - np.log(probs_U[0])
        return np.sum(log_ratio[encoded_sequences], axis=1)

    powers = 4 ** np.arange(m-1, -1, -1)

    contexts = np.lib.stride_tricks.sliding_window_view(
        encoded_sequences, window_shape=m, axis=1
    )

    contexts = contexts[:, :-1, :]
    context_indices = np.tensordot(contexts, powers, axes=([2], [0]))

    next_bases = encoded_sequences[:, m:]

    log_ratio = np.log(probs_B) - np.log(probs_U)

    scores = log_ratio[context_indices, next_bases]

    return scores.sum(axis=1)



# this function computes the ROC and PR curves


def compute_roc(labels, scores):

    order = np.argsort(-scores)
    labels = labels[order]

    P = np.sum(labels == 1)
    N = np.sum(labels == 0)

    tps = np.cumsum(labels == 1)
    fps = np.cumsum(labels == 0)

    tpr = tps / P
    fpr = fps / N

    return fpr, tpr


def compute_pr(labels, scores):

    order = np.argsort(-scores)
    labels = labels[order]

    P = np.sum(labels == 1)

    tps = np.cumsum(labels == 1)
    fps = np.cumsum(labels == 0)

    precision = tps / (tps + fps)
    recall = tps / P

    return recall, precision


def compute_auc(x, y):
    return np.trapz(y, x)



# this function runs the entire pipeline: loading data, encoding nucleotides, creating k folds, building the markov models, scoring the sequences, computing the ROC and PR curves, and plotting the results with mean AUROC and AUPR values 


def run_pipeline(tf_name, tsv_file, genome_fasta, k, m):

    sequences, labels = load_data(tf_name, tsv_file, genome_fasta)
    encoded = encode_sequences(sequences)

    folds = create_stratified_folds(labels, k)

    roc_curves = []
    pr_curves = []
    auroc_values = []
    aupr_values = []

    for train_idx, test_idx in folds:

        X_train = encoded[train_idx]
        y_train = labels[train_idx]

        X_test = encoded[test_idx]
        y_test = labels[test_idx]

        if np.sum(y_train == 1) == 0 or np.sum(y_train == 0) == 0:
            continue

        probs_B = build_markov_model(X_train[y_train == 1], m)
        probs_U = build_markov_model(X_train[y_train == 0], m)

        scores = score_sequences(X_test, probs_B, probs_U, m)

        fpr, tpr = compute_roc(y_test, scores)
        recall, precision = compute_pr(y_test, scores)

        roc_auc = compute_auc(fpr, tpr)
        pr_auc = compute_auc(recall, precision)

        roc_curves.append((fpr, tpr))
        pr_curves.append((recall, precision))
        auroc_values.append(roc_auc)
        aupr_values.append(pr_auc)

    mean_auroc = np.mean(auroc_values)
    mean_aupr = np.mean(aupr_values)

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    for fpr, tpr in roc_curves:
        plt.plot(fpr, tpr, alpha=0.4)
    plt.title(f"ROC (Mean AUROC = {mean_auroc:.3f})")
    plt.xlabel("FPR")
    plt.ylabel("TPR")

    plt.subplot(1,2,2)
    for recall, precision in pr_curves:
        plt.plot(recall, precision, alpha=0.4)
    plt.title(f"PR (Mean AUPR = {mean_aupr:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.tight_layout()
    plt.show()

    print(f"\nMean AUROC: {mean_auroc:.4f}")
    print(f"Mean AUPR: {mean_aupr:.4f}")

### for assignment question 6. uncomment when m=3 and k=5, to generate a single roc and pr curve.

    # fold_to_plot = 0   # choose any fold from 0 to k-1

    # single_fpr, single_tpr = roc_curves[fold_to_plot]
    # single_recall, single_precision = pr_curves[fold_to_plot]
    # single_auroc = auroc_values[fold_to_plot]
    # single_aupr = aupr_values[fold_to_plot]


    # plt.figure(figsize=(12,5))

    # # ROC (single fold)
    # plt.subplot(1,2,1)
    # plt.plot(single_fpr, single_tpr)
    # plt.title(f"Single Fold ROC (AUROC = {single_auroc:.3f})")
    # plt.xlabel("FPR")
    # plt.ylabel("TPR")

    # # PR (single fold)
    # plt.subplot(1,2,2)
    # plt.plot(single_recall, single_precision)
    # plt.title(f"Single Fold PR (AUPR = {single_aupr:.3f})")
    # plt.xlabel("Recall")
    # plt.ylabel("Precision")

    # plt.tight_layout()
    # plt.show()

    # print(f"\nSingle Fold AUROC: {single_auroc:.4f}")
    # print(f"Single Fold AUPR: {single_aupr:.4f}")





# asks for input from the user and actually runs the entire pipeline


if __name__ == "__main__":

    tf = input("Enter TF name: ")
    chromosome = input("Enter chromosome (e.g. chr8): ")
    k = int(input("Enter number of folds k (3-5): "))
    m = int(input("Enter Markov order m (0-10): "))

    tsv_file = f"{chromosome}_200bp_bins.tsv"
    genome_fasta = "hg38.fa"

    run_pipeline(tf, tsv_file, genome_fasta, k, m)
