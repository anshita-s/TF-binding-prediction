# Final pipelines: TF binding prediction (CTCF, REST)

Part of the TF binding prediction project. See top-level README for context.
The earlier Markov baseline is in `../midsem/`.

Predicts in vivo binding potential for CTCF and REST in K562 cells across
200 bp bins, using sequence, chromatin accessibility, motif, and
conservation features. Final submission targets chromosomes 3, 10, and 17.

An EP300 pipeline was also developed and evaluated; it is documented in
`../report.pdf` §4.3 but its code is not preserved here.

## Results (held-out chr 3, 10, 17)

| TF    | auROC | auPRC |
|-------|-------|-------|
| CTCF  | 0.896 | 0.400 |
| REST  | 0.905 | 0.228 |
| EP300 | 0.922 | 0.126 |

Baseline prevalence is ~5% bound, so a random classifier gives auPRC
near 0.05.

## Files

- `tf_binding_pipeline_linSVM_hardened.py` — **CTCF pipeline.** Linear SVM
  (calibrated) over four features: JASPAR PWM score, PhastCons30way
  conservation over the best PWM hit, binary ATAC, and FIMO score from
  the CTCF MEME motif. Runs 5-fold chromosome-level CV, plots ROC/PR
  curves, trains a final model, and writes predictions for chr3/10/17.
- `MM_kmer_LR_balanced-1.ipynb` — **REST pipeline.** Logistic regression
  over strand-invariant 4-mer frequencies (136 canonical k-mers), Markov
  log-odds (m=6), and binary ATAC. Writes predictions back into the
  unknown TSVs.
- `MM.py` — shared Markov model module. Provides `encode_sequences`,
  `build_markov_model`, and `score_sequences`, imported by the REST
  notebook. Also contains its own standalone `run_pipeline` for the
  midterm Markov classifier.

## Requirements

Python 3.12+. Libraries: `numpy pandas scikit-learn matplotlib pyBigWig biopython pyfaidx`

External tools (CTCF pipeline only):

- **FIMO** from the [MEME Suite](https://meme-suite.org/meme/tools/fimo),
  available on `PATH`.

## Input data

Not included in this repository. You will need:

- Per-chromosome 200 bp bin TSVs from the course (chr1–chr22, excluding
  3, 10, 17), with columns `chr, start, end, ATAC, CTCF, REST, EP300`.
- `hg38.fa` reference genome.
- JASPAR PWM files: `CTCF.jaspar`, `REST.jaspar`.
- `hg38.phastCons30way.bw` from UCSC (CTCF pipeline only).
- MEME motif file `CTCF.meme` (CTCF pipeline only).

See `../report.pdf` §2 for full sources.

## Before running

Both pipelines currently use **hard-coded absolute paths** from the
machines they were developed on:

- `tf_binding_pipeline_linSVM_hardened.py` — edit the `CONFIG` dict at the
  top of the file to point to your local copies of the data, genome,
  JASPAR, MEME, and cache/output directories.
- `MM_kmer_LR_balanced-1.ipynb` — edit the `DATA_DIR` and `BIN_DIR`
  variables in Cell 2.

The REST notebook also expects each per-chromosome FASTA to be a bare
sequence file (no header); it will prepend one automatically if needed.

## How to run

### CTCF
``
python tf_binding_pipeline_linSVM_hardened.py
``

Runs end to end: extracts sequences, runs FIMO, computes PWM and
PhastCons scores, cross-validates, plots curves, and writes
```
<output_dir>/chr3_predictions.tsv.gz
<output_dir>/chr10_predictions.tsv.gz
<output_dir>/chr17_predictions.tsv.gz
```

Intermediate results are cached; delete the cache or set
`force_recompute_fimo = True` in `CONFIG` to force recomputation.

### REST

Open the notebook and run all cells:
```jupyter notebook MM_kmer_LR_balanced-1.ipynb```

Ensure `MM.py` is in the same directory so the import succeeds. The
notebook writes predictions in place, into the `REST` column of the
unknown TSVs for chr3, chr10, and chr17.

## Output format

Each prediction TSV has the columns:
```
chr start end ATAC CTCF REST EP300
chr3 12200 12400 U 0.2 0.3 0.0
...
```

Higher scores mean higher predicted binding potential.

## Notes on the implementations

- **CTCF** uses four separately z-scored features (not summed) so that
  each feature's contribution is preserved. The SVM is wrapped in
  `CalibratedClassifierCV` to produce probabilities.
- **REST** uses strand-invariant k-mers — each 4-mer is mapped to the
  lexicographically smaller of itself and its reverse complement — and
  a 6th-order Markov log-odds score computed on the training fold only.
  Features are z-scored before logistic regression with
  `class_weight='balanced'`.
- Bins containing `N` are dropped during sequence extraction in both
  pipelines.
- Full methodological details, including the EP300 pipeline and
  approaches that were tried and discarded, are in `../report.pdf` §3
  and §4.
