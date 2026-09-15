#!/usr/bin/env python3
"""
TF binding prediction pipeline (fast, debuggable, 4-feature SVM)
================================================================

What this version fixes / changes:
1. Fixes pyfaidx extraction bug:
     genome[chrom][start:end].upper()   # correct when as_raw=True
2. Uses separate standardized features for SVM:
     [PWM_score, mean_PhastCons, ATAC_numeric, FIMO_score]
   instead of summing z-scores into a single scalar.
3. Adds runtime diagnostics so you can SEE what is going on:
   - sequence extraction sanity checks
   - FIMO hit counts and hit fractions per TF/chromosome
   - per-feature summaries by class for each TF
   - per-fold AUROC/AUPRC
4. Uses dataset-specific caches so train / chr3 / chr10 / chr17 do not collide.
5. Forces chromosome FASTA regeneration for each dataset tag to avoid stale bad FASTA.
6. Keeps the workflow crash-safe with atomic cache writes.

Required Python packages:
    pip install numpy pandas scikit-learn matplotlib pyBigWig biopython pyfaidx

External tools required in PATH:
    fimo      (MEME suite)

Usage:
    python tf_binding_pipeline_linSVM_4feat_debug.py

Notes:
- JASPAR files are used for PWM scoring.
- MEME files are used for FIMO.
- This script does not silently assume FIMO is healthy. It prints hit fractions and
  warns if they are suspiciously low.
"""

from __future__ import annotations

import os
import re
import sys
import math
import pickle
import logging
import warnings
import subprocess
import shutil
import hashlib
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool
from typing import Dict, List, Tuple, Iterable

import numpy as np
import pandas as pd
import pyBigWig
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq

from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
from sklearn.utils.class_weight import compute_class_weight

warnings.filterwarnings("ignore")

CONFIG = {
    "data_dir": "/home/piyush/urvija_temp/projectData",
    "genome_fa": "/home/piyush/urvija_temp/hg38.fa",
    "phastcons_bw": "/home/piyush/urvija_temp/hg38.phastCons30way.bw",
    "jaspar_dir": "/home/piyush/urvija_temp",
    "meme_dir": "/home/piyush/urvija_temp",
    "cache_dir": "/home/piyush/urvija_temp/linearSVM_out/cache",
    "output_dir": "/home/piyush/urvija_temp/linearSVM_out/predictions",
    "plot_dir": "/home/piyush/urvija_temp/linearSVM_out/plots",
    "predict_chrs": ["chr3", "chr10", "chr17"],
    "predict_chrs": ["chr3", "chr10", "chr17"],
    "n_folds": 5,
    "tfs": ["CTCF", "REST", "EP300"],
    "n_workers": 4,
    "svm_max_iter": 10000,
    "fimo_pval_thresh": 1e-2,
    "fimo_timeout_sec": 3600,
    "atac_value_B": 1.0,
    "atac_value_U": 0.02,
    "force_rebuild_fastas": True,
    "force_recompute_fimo": False,
    "min_fimo_hit_fraction_warn": 0.005,
    "sequence_sample_size": 200,
    "random_seed": 42,
}

for _d in [CONFIG["cache_dir"], CONFIG["output_dir"], CONFIG["plot_dir"]]:
    os.makedirs(_d, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(CONFIG["output_dir"], "pipeline_4feat.log"), mode="w"),
    ],
)
log = logging.getLogger(__name__)
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

BASE_IDX = {"A": 0, "C": 1, "G": 2, "T": 3}
COMPLEMENT_ROW_IDX = np.array([3, 2, 1, 0])  # A,C,G,T -> T,G,C,A
FEATURE_NAMES = ["PWM", "PhastCons", "ATAC", "FIMO"]
SAFE_FASTA_ID_RE = re.compile(r"^RID__(chr[^_\s]+)__(\d+)__(\d+)$")


def stage_banner(title: str) -> None:
    bar = "=" * 80
    log.info("\n%s\n%s\n%s", bar, title, bar)


def atomic_pickle_dump(obj, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def safe_pickle_load(path: str):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


def dataset_signature(df: pd.DataFrame) -> str:
    m = hashlib.md5()
    for chrom, start, end in df[["chr", "start", "end"]].itertuples(index=False):
        m.update(f"{chrom}:{int(start)}-{int(end)}|".encode())
    return m.hexdigest()[:12]


def discover_tsv_files(data_dir: str) -> Dict[str, str]:
    files: Dict[str, str] = {}
    for f in sorted(Path(data_dir).glob("*.tsv")):
        m = re.search(r"(chr\d+|chrX|chrY)", f.name)
        if m:
            files[m.group(1)] = str(f)
    return files


def load_tsv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, sep="\t")
    df.columns = df.columns.str.strip()
    rename = {}
    for c in df.columns:
        cl = c.lower()
        if cl in ("chr", "chrom", "chromosome"):
            rename[c] = "chr"
        elif cl == "start":
            rename[c] = "start"
        elif cl == "end":
            rename[c] = "end"
        elif cl == "atac":
            rename[c] = "ATAC"
        elif c.upper() in {"CTCF", "REST", "EP300"}:
            rename[c] = c.upper()
    df = df.rename(columns=rename)
    required = {"chr", "start", "end", "ATAC"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {filepath}: {sorted(missing)}")
    df["start"] = df["start"].astype(int)
    df["end"] = df["end"].astype(int)
    return df


def load_all_training_data(all_files: Dict[str, str], predict_chrs: List[str]) -> pd.DataFrame:
    dfs = []
    for chrom, fp in all_files.items():
        if chrom not in predict_chrs:
            dfs.append(load_tsv(fp))
    if not dfs:
        raise RuntimeError("No training chromosome TSVs found.")
    out = pd.concat(dfs, ignore_index=True)
    log.info(f"Training data: {len(out):,} regions across {len(dfs)} chromosomes.")
    return out


def get_sequences_for_chrom(args):
    genome_fa, chrom, rows, cache_key = args
    cached = safe_pickle_load(cache_key)
    if cached is not None:
        return chrom, cached

    seqs = {}
    try:
        from pyfaidx import Fasta
        genome = Fasta(genome_fa, as_raw=True, sequence_always_upper=True)
        for start, end in rows:
            try:
                # CORRECT with as_raw=True: slice returns a string, not an object with .seq
                seq = genome[chrom][start:end].upper()
            except Exception:
                seq = "N" * max(0, end - start)
            seqs[(chrom, int(start), int(end))] = seq
    except ImportError:
        log.warning("pyfaidx not installed, falling back to BioPython (slower).")
        record = None
        for rec in SeqIO.parse(genome_fa, "fasta"):
            if rec.id == chrom:
                record = rec
                break
        for start, end in rows:
            seq = str(record.seq[start:end]).upper() if record is not None else "N" * max(0, end - start)
            seqs[(chrom, int(start), int(end))] = seq

    atomic_pickle_dump(seqs, cache_key)
    return chrom, seqs


def extract_all_sequences(genome_fa: str, df: pd.DataFrame, n_workers: int, dataset_tag: str) -> Dict[Tuple[str, int, int], str]:
    grouped = df.groupby("chr", sort=True)
    args = []
    for chrom, grp in grouped:
        cache_key = os.path.join(CONFIG["cache_dir"], f"seq_{dataset_tag}_{chrom}.pkl")
        args.append((genome_fa, chrom, list(zip(grp["start"], grp["end"])), cache_key))

    all_seqs = {}
    with Pool(min(n_workers, len(args))) as pool:
        for chrom, seqs in pool.imap_unordered(get_sequences_for_chrom, args):
            all_seqs.update(seqs)
            log.info(f"Sequences ready: {chrom} ({len(seqs):,} regions)")
    return all_seqs


def summarize_sequences(seqs: Dict[Tuple[str, int, int], str], sample_size: int = 200) -> None:
    if not seqs:
        log.warning("No sequences available to summarize.")
        return
    keys = list(seqs.keys())
    sample = keys[: min(sample_size, len(keys))]
    lengths = []
    non_n_fracs = []
    bad = 0
    for k in sample:
        seq = seqs[k]
        lengths.append(len(seq))
        if len(seq) == 0:
            bad += 1
            continue
        non_n = sum(1 for b in seq if b in BASE_IDX)
        non_n_fracs.append(non_n / len(seq))
        if non_n == 0:
            bad += 1
    log.info(
        "Sequence sanity: sampled %d regions | mean_len=%.2f | mean_nonN_frac=%.4f | allN_regions=%d",
        len(sample),
        float(np.mean(lengths)) if lengths else 0.0,
        float(np.mean(non_n_fracs)) if non_n_fracs else 0.0,
        bad,
    )
    if non_n_fracs and float(np.mean(non_n_fracs)) < 0.90:
        log.warning("Mean non-N fraction is unexpectedly low. Re-check genome path / coordinates / cache.")


def parse_jaspar(filepath: str) -> np.ndarray:
    counts = {}
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(">"):
                continue
            m = re.match(r"([ACGT])\s*\[([^\]]+)\]", line)
            if m:
                counts[m.group(1)] = np.array(list(map(float, m.group(2).split())), dtype=float)
    if len(counts) != 4:
        raise ValueError(f"Could not parse full JASPAR matrix from {filepath}")
    C = np.array([counts["A"], counts["C"], counts["G"], counts["T"]], dtype=float)
    C += 0.1
    freq = C / C.sum(axis=0, keepdims=True)
    bg = np.array([0.25, 0.25, 0.25, 0.25], dtype=float)[:, None]
    pwm = np.log2(freq / bg)
    return pwm


def score_sequence_pwm(seq: str, pwm: np.ndarray) -> Tuple[float, int, int]:
    L = pwm.shape[1]
    n = len(seq)
    if n < L:
        return 0.0, 0, n

    best_score = -np.inf
    best_start = 0

    pwm_rc = pwm[COMPLEMENT_ROW_IDX][:, ::-1]

    for p in (pwm, pwm_rc):
        for i in range(n - L + 1):
            subseq = seq[i:i + L]
            score = 0.0
            valid = True
            for j, b in enumerate(subseq):
                idx = BASE_IDX.get(b)
                if idx is None:
                    valid = False
                    break
                score += p[idx, j]
            if valid and score > best_score:
                best_score = score
                best_start = i

    if not np.isfinite(best_score):
        return 0.0, 0, min(L, n)
    return float(best_score), int(best_start), int(best_start + L)


def batch_score_pwm(args):
    tf, pwm, region_seqs = args
    out = {}
    for key, seq in region_seqs:
        out[key] = score_sequence_pwm(seq, pwm)
    return tf, out


def compute_pwm_scores_all(df: pd.DataFrame, seqs: Dict[Tuple[str, int, int], str], pwms: Dict[str, np.ndarray], n_workers: int, dataset_tag: str):
    cache_key = os.path.join(CONFIG["cache_dir"], f"pwm_scores_{dataset_tag}.pkl")
    cached = safe_pickle_load(cache_key)
    if cached is not None:
        log.info("Loaded cached PWM scores.")
        return cached

    region_seqs = [
        ((row["chr"], int(row["start"]), int(row["end"])), seqs[(row["chr"], int(row["start"]), int(row["end"]))])
        for _, row in df.iterrows()
    ]
    chunk_size = max(1, len(region_seqs) // max(1, n_workers * 4))
    chunks = [region_seqs[i:i + chunk_size] for i in range(0, len(region_seqs), chunk_size)]

    pwm_scores = {tf: {} for tf in pwms}
    for tf, pwm in pwms.items():
        log.info(f"PWM scoring for {tf} ...")
        tasks = [(tf, pwm, chunk) for chunk in chunks]
        with Pool(min(n_workers, len(tasks))) as pool:
            for tf_name, partial in pool.imap_unordered(batch_score_pwm, tasks):
                pwm_scores[tf_name].update(partial)
        vals = [v[0] for v in pwm_scores[tf].values()]
        log.info("PWM summary %s: mean=%.4f median=%.4f max=%.4f", tf, float(np.mean(vals)), float(np.median(vals)), float(np.max(vals)))

    atomic_pickle_dump(pwm_scores, cache_key)
    return pwm_scores


def compute_phastcons_all(df: pd.DataFrame, pwm_scores: dict, tfs: List[str], bw_path: str, dataset_tag: str) -> dict:
    cache_key = os.path.join(CONFIG["cache_dir"], f"phastcons_{dataset_tag}.pkl")
    cached = safe_pickle_load(cache_key)
    if cached is not None:
        log.info("Loaded cached PhastCons scores.")
        return cached

    pc_scores = {tf: {} for tf in tfs}
    bw = pyBigWig.open(bw_path)
    chrom_sizes = bw.chroms()

    try:
        for tf in tfs:
            for _, row in df.iterrows():
                key = (row["chr"], int(row["start"]), int(row["end"]))
                chrom = row["chr"]
                _, ms, me = pwm_scores[tf].get(key, (0.0, 0, min(20, int(row["end"]) - int(row["start"])) ))
                chrom_size = chrom_sizes.get(chrom, 0)
                abs_start = max(0, int(row["start"]) + int(ms))
                abs_end = min(chrom_size, int(row["start"]) + int(me))
                if abs_start >= abs_end or chrom_size == 0:
                    pc_scores[tf][key] = 0.0
                    continue
                try:
                    vals = bw.stats(chrom, abs_start, abs_end, type="mean")
                    pc_scores[tf][key] = float(vals[0]) if vals and vals[0] is not None else 0.0
                except Exception:
                    pc_scores[tf][key] = 0.0
            vals = list(pc_scores[tf].values())
            log.info("PhastCons summary %s: mean=%.4f median=%.4f max=%.4f", tf, float(np.mean(vals)), float(np.median(vals)), float(np.max(vals)))
    finally:
        bw.close()

    atomic_pickle_dump(pc_scores, cache_key)
    return pc_scores


def make_safe_region_id(chrom: str, start: int, end: int) -> str:
    return f"RID__{chrom}__{int(start)}__{int(end)}"


def parse_safe_region_id(seq_name: str):
    m = SAFE_FASTA_ID_RE.match(seq_name.strip())
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def write_fasta_for_chrom(df_chrom: pd.DataFrame, seqs: dict, fasta_path: str) -> None:
    records = []
    for _, row in df_chrom.iterrows():
        key = (row["chr"], int(row["start"]), int(row["end"]))
        seq = seqs[key]
        region_id = make_safe_region_id(row['chr'], int(row['start']), int(row['end']))
        records.append(SeqRecord(Seq(seq), id=region_id, description=""))
    with open(fasta_path, "w") as f:
        SeqIO.write(records, f, "fasta")


def run_fimo_for_chrom(args):
    tf, chrom, motif_path, fasta_path, fimo_pval, cache_key = args

    if (not CONFIG["force_recompute_fimo"]):
        cached = safe_pickle_load(cache_key)
        if cached is not None:
            return tf, chrom, cached

    out_dir = cache_key + "_out"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir, exist_ok=True)

    cmd = [
        "fimo",
        "--thresh", str(fimo_pval),
        "--no-qvalue",
        "--oc", out_dir,
        motif_path,
        fasta_path,
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=CONFIG["fimo_timeout_sec"])
    except FileNotFoundError as e:
        raise RuntimeError("FIMO executable not found in PATH.") from e
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"FIMO timed out for {tf} {chrom}") from e

    if proc.returncode != 0:
        raise RuntimeError(f"FIMO failed for {tf} {chrom}:\nSTDERR:\n{proc.stderr[:2000]}")

    fimo_tsv = os.path.join(out_dir, "fimo.tsv")
    if not os.path.exists(fimo_tsv):
        raise RuntimeError(f"FIMO finished but {fimo_tsv} was not produced for {tf} {chrom}")

    best_scores = defaultdict(float)
    raw_hit_lines = 0
    parsed_hit_lines = 0
    unparsable_examples = []
    with open(fimo_tsv) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("motif_id"):
                continue
            raw_hit_lines += 1
            parts = line.split("\t")
            if len(parts) < 8:
                continue
            seq_name = parts[2].strip()
            parsed = parse_safe_region_id(seq_name)
            if parsed is None:
                if len(unparsable_examples) < 5:
                    unparsable_examples.append(seq_name)
                continue
            parsed_hit_lines += 1
            try:
                pval = float(parts[7])
            except ValueError:
                continue
            if pval <= 0:
                pval = 1e-320
            score = -math.log10(pval)
            key = parsed
            if score > best_scores[key]:
                best_scores[key] = score

    if raw_hit_lines == 0:
        log.warning(f"FIMO raw output had 0 hit lines for {tf} {chrom}.")
    else:
        log.info(
            f"FIMO raw lines for {tf} {chrom}: raw_hit_lines={raw_hit_lines:,} | parsed_hit_lines={parsed_hit_lines:,} | unique_regions={len(best_scores):,}"
        )
    if unparsable_examples:
        log.warning(f"Example unparsable FIMO seq_names for {tf} {chrom}: {unparsable_examples}")

    result = dict(best_scores)
    atomic_pickle_dump(result, cache_key)
    shutil.rmtree(out_dir, ignore_errors=True)
    return tf, chrom, result


def run_all_fimo(df: pd.DataFrame, seqs: dict, tfs: List[str], meme_dir: str, dataset_tag: str, fimo_pval: float, n_workers: int):
    fasta_dir = os.path.join(CONFIG["cache_dir"], f"fastas_{dataset_tag}")
    os.makedirs(fasta_dir, exist_ok=True)

    chrom_fastas = {}
    for chrom, grp in df.groupby("chr", sort=True):
        fasta_path = os.path.join(fasta_dir, f"{chrom}.fa")
        if CONFIG["force_rebuild_fastas"] and os.path.exists(fasta_path):
            os.remove(fasta_path)
        write_fasta_for_chrom(grp, seqs, fasta_path)
        chrom_fastas[chrom] = fasta_path

    tasks = []
    for tf in tfs:
        motif_path = os.path.join(meme_dir, f"{tf}.meme")
        if not os.path.exists(motif_path):
            raise FileNotFoundError(f"Missing MEME motif file: {motif_path}")
        for chrom, fasta_path in chrom_fastas.items():
            cache_key = os.path.join(CONFIG["cache_dir"], f"fimo_{dataset_tag}_{tf}_{chrom}.pkl")
            tasks.append((tf, chrom, motif_path, fasta_path, fimo_pval, cache_key))

    fimo_results: Dict[str, Dict[str, Dict[Tuple[str, int, int], float]]] = {}
    with Pool(min(n_workers, len(tasks))) as pool:
        for tf, chrom, result in pool.imap_unordered(run_fimo_for_chrom, tasks):
            fimo_results.setdefault(tf, {})[chrom] = result
            log.info("FIMO done: %s %s -> %d hit-regions", tf, chrom, len(result))

    log_fimo_summary(df, fimo_results)
    return fimo_results


def log_fimo_summary(df: pd.DataFrame, fimo_results: dict) -> None:
    region_counts = df.groupby("chr").size().to_dict()
    for tf, chrom_dict in fimo_results.items():
        tf_total_regions = 0
        tf_total_hit_regions = 0
        for chrom, hits in sorted(chrom_dict.items()):
            n_regions = int(region_counts.get(chrom, 0))
            n_hits = len(hits)
            frac = (n_hits / n_regions) if n_regions else 0.0
            tf_total_regions += n_regions
            tf_total_hit_regions += n_hits
            log.info("FIMO summary %s %s: hit_regions=%d / %d (%.4f)", tf, chrom, n_hits, n_regions, frac)
            if n_regions and frac < CONFIG["min_fimo_hit_fraction_warn"]:
                log.warning("Very low FIMO hit fraction for %s %s (%.4f). Check motif file / threshold / FASTA content.", tf, chrom, frac)
        overall_frac = (tf_total_hit_regions / tf_total_regions) if tf_total_regions else 0.0
        log.info("FIMO overall %s: hit_regions=%d / %d (%.4f)", tf, tf_total_hit_regions, tf_total_regions, overall_frac)


def build_feature_matrix(df: pd.DataFrame, tf: str, pwm_scores: dict, pc_scores: dict, fimo_results: dict):
    rows = []
    labels = []
    keys = []

    for _, row in df.iterrows():
        chrom = row["chr"]
        start = int(row["start"])
        end = int(row["end"])
        key = (chrom, start, end)

        pwm_val = float(pwm_scores[tf].get(key, (0.0, 0, 0))[0])
        phastcons_val = float(pc_scores[tf].get(key, 0.0))
        atac_val = CONFIG["atac_value_B"] if str(row.get("ATAC", "U")).strip() == "B" else CONFIG["atac_value_U"]
        fimo_val = float(fimo_results.get(tf, {}).get(chrom, {}).get(key, 0.0))

        rows.append([pwm_val, phastcons_val, atac_val, fimo_val])
        keys.append(key)
        if tf in df.columns:
            labels.append(1 if str(row[tf]).strip() == "B" else 0)
        else:
            labels.append(None)

    X = np.asarray(rows, dtype=np.float32)
    y = None if not labels or labels[0] is None else np.asarray(labels, dtype=np.int8)
    return X, y, keys


def summarize_features_by_class(tf: str, X: np.ndarray, y: np.ndarray) -> None:
    pos = y == 1
    neg = y == 0
    log.info("Feature summary for %s:", tf)
    for j, name in enumerate(FEATURE_NAMES):
        pos_mean = float(np.mean(X[pos, j])) if np.any(pos) else float("nan")
        neg_mean = float(np.mean(X[neg, j])) if np.any(neg) else float("nan")
        pos_med = float(np.median(X[pos, j])) if np.any(pos) else float("nan")
        neg_med = float(np.median(X[neg, j])) if np.any(neg) else float("nan")
        log.info("  %s | pos_mean=%.4f neg_mean=%.4f | pos_median=%.4f neg_median=%.4f", name, pos_mean, neg_mean, pos_med, neg_med)


def zscore_features(X_train: np.ndarray, X_val: np.ndarray | None = None):
    scaler = StandardScaler()
    X_train_z = scaler.fit_transform(X_train)
    X_val_z = scaler.transform(X_val) if X_val is not None else None
    return X_train_z, X_val_z, scaler


def apply_zscore_transform(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    return scaler.transform(X)


def make_chrom_folds(train_chroms: List[str], n_folds: int) -> List[Tuple[List[str], List[str]]]:
    chroms = sorted(train_chroms)
    rng = np.random.default_rng(CONFIG["random_seed"])
    chroms = rng.permutation(chroms).tolist()
    fold_size = len(chroms) // n_folds
    remainder = len(chroms) % n_folds

    fold_groups = []
    idx = 0
    for i in range(n_folds):
        size = fold_size + (1 if i < remainder else 0)
        fold_groups.append(chroms[idx:idx + size])
        idx += size

    return [
        ([c for j, grp in enumerate(fold_groups) if j != i for c in grp], fold_groups[i])
        for i in range(n_folds)
    ]


def train_svm(X_train: np.ndarray, y_train: np.ndarray) -> CalibratedClassifierCV:
    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    cw = dict(zip(classes, weights))
    base = LinearSVC(
        C=1.0,
        class_weight=cw,
        max_iter=CONFIG["svm_max_iter"],
        dual="auto",
        random_state=CONFIG["random_seed"],
    )
    clf = CalibratedClassifierCV(base, cv=3, method="sigmoid")
    clf.fit(X_train, y_train)
    return clf


def evaluate_fold(clf, X_val: np.ndarray, y_val: np.ndarray) -> dict:
    proba = clf.predict_proba(X_val)[:, 1]
    auroc = roc_auc_score(y_val, proba)
    auprc = average_precision_score(y_val, proba)
    fpr, tpr, _ = roc_curve(y_val, proba)
    precision, recall, _ = precision_recall_curve(y_val, proba)
    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "proba": proba,
        "y_val": y_val,
    }


def log_model_weights(tf: str, final_clf: CalibratedClassifierCV) -> None:
    try:
        est = final_clf.calibrated_classifiers_[0].estimator
        coef = np.ravel(est.coef_)
        if coef.size == len(FEATURE_NAMES):
            pairs = ", ".join(f"{n}={w:.4f}" for n, w in zip(FEATURE_NAMES, coef))
            log.info("Approximate linear weights for %s: %s", tf, pairs)
    except Exception as e:
        log.warning("Could not read calibrated SVM coefficients for %s: %s", tf, e)


def run_kfold_for_tf(tf: str, df_train: pd.DataFrame, pwm_scores: dict, pc_scores: dict, fimo_results: dict, folds: List[Tuple[List[str], List[str]]]) -> dict:
    log.info("\n%s\nCross-validation for %s\n%s", "=" * 60, tf, "=" * 60)
    X_all, y_all, keys_all = build_feature_matrix(df_train, tf, pwm_scores, pc_scores, fimo_results)
    summarize_features_by_class(tf, X_all, y_all)

    key_to_chrom = {k: k[0] for k in keys_all}
    fold_results = []

    for i, (train_chroms, val_chroms) in enumerate(folds, start=1):
        train_mask = np.array([key_to_chrom[k] in train_chroms for k in keys_all], dtype=bool)
        val_mask = np.array([key_to_chrom[k] in val_chroms for k in keys_all], dtype=bool)
        if train_mask.sum() == 0 or val_mask.sum() == 0:
            log.warning("Skipping fold %d for %s due to empty split.", i, tf)
            continue

        X_tr, y_tr = X_all[train_mask], y_all[train_mask]
        X_vl, y_vl = X_all[val_mask], y_all[val_mask]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_vl)) < 2:
            log.warning("Skipping fold %d for %s due to single-class split.", i, tf)
            continue

        X_tr_z, X_vl_z, _ = zscore_features(X_tr, X_vl)
        clf = train_svm(X_tr_z, y_tr)
        metrics = evaluate_fold(clf, X_vl_z, y_vl)
        fold_results.append(metrics)
        log.info("Fold %d | val_chroms=%s | AUROC=%.4f | AUPRC=%.4f", i, val_chroms, metrics["auroc"], metrics["auprc"])

    X_all_z, _, final_scaler = zscore_features(X_all)
    final_clf = train_svm(X_all_z, y_all)
    log_model_weights(tf, final_clf)

    if fold_results:
        log.info("%s mean CV | AUROC=%.4f | AUPRC=%.4f", tf, float(np.mean([r["auroc"] for r in fold_results])), float(np.mean([r["auprc"] for r in fold_results])))
    else:
        log.warning("No valid CV folds for %s.", tf)

    return {
        "fold_results": fold_results,
        "final_clf": final_clf,
        "final_scaler": final_scaler,
        "X_all": X_all,
        "y_all": y_all,
    }


def plot_curves(tf: str, fold_results: List[dict], plot_dir: str) -> None:
    if not fold_results:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 1, len(fold_results)))

    ax = axes[0]
    for i, res in enumerate(fold_results, start=1):
        ax.plot(res["fpr"], res["tpr"], color=colors[i - 1], label=f"Fold {i} (AUC={res['auroc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{tf} ROC | mean AUROC={np.mean([r['auroc'] for r in fold_results]):.4f}")
    ax.legend(fontsize=8)

    ax = axes[1]
    for i, res in enumerate(fold_results, start=1):
        ax.plot(res["recall"], res["precision"], color=colors[i - 1], label=f"Fold {i} (AUPRC={res['auprc']:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{tf} PR | mean AUPRC={np.mean([r['auprc'] for r in fold_results]):.4f}")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(plot_dir, f"{tf}_curves_4feat.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    log.info(f"Saved curve plot: {out_path}")


def predict_unknown(chrom: str, df_unk: pd.DataFrame, pwm_scores_unk: dict, pc_scores_unk: dict, fimo_results_unk: dict, tf_models: dict) -> pd.DataFrame:
    result_df = df_unk[["chr", "start", "end", "ATAC"]].copy()
    for tf, model_info in tf_models.items():
        feature_rows = []
        for _, row in df_unk.iterrows():
            key = (row["chr"], int(row["start"]), int(row["end"]))
            pwm_val = float(pwm_scores_unk[tf].get(key, (0.0, 0, 0))[0])
            phastcons_val = float(pc_scores_unk[tf].get(key, 0.0))
            atac_val = CONFIG["atac_value_B"] if str(row.get("ATAC", "U")).strip() == "B" else CONFIG["atac_value_U"]
            fimo_val = float(fimo_results_unk.get(tf, {}).get(chrom, {}).get(key, 0.0))
            feature_rows.append([pwm_val, phastcons_val, atac_val, fimo_val])
        X_unk = np.asarray(feature_rows, dtype=np.float32)
        X_unk_z = apply_zscore_transform(model_info["final_scaler"], X_unk)
        result_df[tf] = np.round(model_info["final_clf"].predict_proba(X_unk_z)[:, 1], 6)
    return result_df


def print_prereq_notes() -> None:
    log.info("Python deps: numpy pandas scikit-learn matplotlib pyBigWig biopython pyfaidx")
    log.info("External deps: fimo (MEME suite) in PATH")
    log.info("This run uses 4 separate z-scored features for SVM: %s", ", ".join(FEATURE_NAMES))
    log.info("For live console logs, run with: python -u %s", os.path.basename(__file__))


def main() -> None:
    print_prereq_notes()
    C = CONFIG
    all_files = discover_tsv_files(C["data_dir"])
    if not all_files:
        raise RuntimeError(f"No TSV files found in {C['data_dir']}")

    train_chroms = [c for c in all_files if c not in C["predict_chrs"]]
    log.info(f"Training chromosomes ({len(train_chroms)}): {sorted(train_chroms)}")
    log.info(f"Prediction chromosomes: {C['predict_chrs']}")

    df_train = load_all_training_data(all_files, C["predict_chrs"])
    train_tag = f"train_{dataset_signature(df_train)}"
    log.info(f"Training dataset tag: {train_tag}")

    pwms = {}
    for tf in C["tfs"]:
        jp = os.path.join(C["jaspar_dir"], f"{tf}.jaspar")
        if not os.path.exists(jp):
            raise FileNotFoundError(f"Missing JASPAR file: {jp}")
        pwms[tf] = parse_jaspar(jp)
        log.info(f"Parsed JASPAR PWM for {tf} | motif_length={pwms[tf].shape[1]}")

    stage_banner("EXTRACTING TRAINING SEQUENCES")
    seqs_train = extract_all_sequences(C["genome_fa"], df_train, C["n_workers"], train_tag)
    summarize_sequences(seqs_train, C["sequence_sample_size"])

    stage_banner("RUNNING FIMO ON TRAINING DATA")
    fimo_train = run_all_fimo(df_train, seqs_train, C["tfs"], C["meme_dir"], train_tag, C["fimo_pval_thresh"], C["n_workers"])

    stage_banner("COMPUTING PWM SCORES")
    pwm_scores_train = compute_pwm_scores_all(df_train, seqs_train, pwms, C["n_workers"], train_tag)

    stage_banner("COMPUTING PHASTCONS SCORES")
    pc_scores_train = compute_phastcons_all(df_train, pwm_scores_train, C["tfs"], C["phastcons_bw"], train_tag)

    folds = make_chrom_folds(train_chroms, C["n_folds"])
    for i, (tr, vl) in enumerate(folds, start=1):
        log.info(f"Fold {i}: train={tr} | val={vl}")

    stage_banner("TRAINING AND CROSS-VALIDATING PER TF")
    tf_models = {}
    for tf in C["tfs"]:
        results = run_kfold_for_tf(tf, df_train, pwm_scores_train, pc_scores_train, fimo_train, folds)
        tf_models[tf] = results
        plot_curves(tf, results["fold_results"], C["plot_dir"])

    log.info("\nPredicting on unknown chromosomes...")
    for pred_chrom in C["predict_chrs"]:
        if pred_chrom not in all_files:
            log.warning(f"Missing TSV for {pred_chrom}; skipping.")
            continue
        df_unk = load_tsv(all_files[pred_chrom])
        unk_tag = f"{pred_chrom}_{dataset_signature(df_unk)}"
        log.info(f"{pred_chrom}: {len(df_unk):,} regions | dataset tag {unk_tag}")

        seqs_unk = extract_all_sequences(C["genome_fa"], df_unk, C["n_workers"], unk_tag)
        summarize_sequences(seqs_unk, C["sequence_sample_size"])
        fimo_unk = run_all_fimo(df_unk, seqs_unk, C["tfs"], C["meme_dir"], unk_tag, C["fimo_pval_thresh"], C["n_workers"])
        pwm_unk = compute_pwm_scores_all(df_unk, seqs_unk, pwms, C["n_workers"], unk_tag)
        pc_unk = compute_phastcons_all(df_unk, pwm_unk, C["tfs"], C["phastcons_bw"], unk_tag)

        pred_df = predict_unknown(pred_chrom, df_unk, pwm_unk, pc_unk, fimo_unk, tf_models)
        out_path = os.path.join(C["output_dir"], f"{pred_chrom}_predictions.tsv.gz")
        pred_df.to_csv(out_path, sep="\t", index=False, compression="gzip")
        log.info(f"Saved predictions: {out_path}")

    log.info("Pipeline complete.")


if __name__ == "__main__":
    main()
