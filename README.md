# Predicting TF Binding Potential of DNA Sequences

Course project, **Computational Functional Genomics** (Jan–Apr 2026).
By- Anshita Sharma and Urvija Agrawal

Predicts in vivo binding of transcription factors from 200 bp DNA bins in
K562 cells, integrating sequence, chromatin accessibility, motif, and
conservation features. The project had two milestones: a Markov-model
baseline for the midterm, and per-TF pipelines for the final submission.

## Contents
```
.
├── README.md ← you are here
├── Project_description.pdf ← original project brief
├── Report.pdf ← full writeup: methods, results, and what didn't work
├── Pre-Midsem/ ← Markov model classifier (midterm milestone)
│ └── README.md ← setup and run instructions for this part
│ └── MarkovCrossValidation.py ← Markov binary classifier code
└── Post-Midsem/ ← final per-TF pipelines (CTCF, REST)
│ └── tf_binding_pipeline_linSVM_hardened.py ← CTCF pipeline
│ └── MM_kmer_LR_balanced-1.ipynb ← REST pipeline
│ └── MM.py ← Markov classifier code
└── README.md ← setup and run instructions for this part
```

## Results

**Final pipelines — held-out chromosomes 3, 10, and 17:**

| TF    | auROC | auPRC |
|-------|-------|-------|
| CTCF  | 0.896 | 0.400 |
| REST  | 0.905 | 0.228 |
| EP300 | 0.922 | 0.126 |

Baseline prevalence is ~5% bound, so a random classifier gives auPRC
near 0.05. EP300's pipeline is documented in `report.pdf` §4.3 but its
code is not preserved in this repository.

**Midterm Markov baseline:** auROC 0.74–0.78, auPRC 0.02–0.05 across the
three TFs. Full breakdown in `report.pdf` §3.1.

## Where to start

- **Curious about the science?** Read `report.pdf`. It covers the problem,
  data, methods, results, and — importantly — approaches that were tried
  and discarded.
- **Want to run something?** See `midsem/README.md` or
  `finalsem/README.md`. Each has its own requirements, data setup, and
  run instructions.
- **Want the problem statement?** `description.pdf` is the brief as
  originally given.

## What's in each milestone

**`midsem/`** — a Markov model classifier. Trains bound and unbound
Markov models of order `m` on k-fold splits of a single chromosome,
scores held-out bins by log-odds ratio, and plots ROC and PR curves.
Supports `m = 0..10` and `k = 3..5`.

**`finalsem/`** — two pipelines:
- **CTCF:** linear SVM over JASPAR PWM score, PhastCons conservation,
  binary ATAC, and FIMO score.
- **REST:** logistic regression over strand-invariant k-mer frequencies,
  Markov log-odds (m=6), and binary ATAC.

## Data

Not included in this repository. The training TSVs are course-provided
and derive from ENCODE ChIP-seq peaks on the hg38 genome. Reference
genome, JASPAR PFMs, PhastCons bigWig, and MEME motif files are
downloaded separately. See the per-milestone READMEs for the exact
expected layout, or `report.pdf` §2 for full sources.

## Notes

- AI tools were used for debugging and report-writing assistance; all
  code is understood and owned by the team.
- Collaborators and acknowledgements are in `report.pdf` §8.
- Full methodological details, including approaches that were tried and
  discarded, are in `report.pdf` §3 and §4.
  
