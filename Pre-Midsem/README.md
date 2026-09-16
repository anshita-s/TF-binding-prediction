# Mid-semester Goal: Markov model classifier

Part of the TF binding prediction project. See top-level README for context.

## Requirements

- Python 3.12.4
- Libraries: numpy, pandas, matplotlib, scikit-learn, pyfaidx

## Files

- `MarkovCrossValidation.py` — main classifier. Trains bound and unbound
  Markov models of order `m` on the training folds, scores each held-out
  bin by the log-odds ratio of the two models, and produces per-fold ROC
  and precision-recall curves along with mean auROC and auPRC.

## How to run

From inside the `midsem/` (working directory):
` python MarkovCrossValidation.py`

The script will prompt for:
```
Enter chromosome number (except 3,10,17,x,y):
Choose TF (CTCF, REST, EP300):
Markov model order m:
k-fold value (>=2):
```

Example session:
```
Enter chromosome number (except 3,10,17,x,y): 4
Choose TF (CTCF, REST, EP300): REST
Markov model order m: 3
k-fold value (>=2): 5

```

## Input data

The data is not included in this repository. The script expects the
following layout, where each chromosome has its own folder containing
both the FASTA file and the 200 bp bins TSV:
```
data/
├── chr1.fa/
│ ├── chr1.fa
│ └── chr1_200bp_bins.tsv
├── chr2.fa/
│ ├── chr2.fa
│ └── chr2_200bp_bins.tsv
└── ...
```


Chromosomes 3, 10, 17, X, and Y are excluded from training. Data was
provided as part of the course and derives from ENCODE ChIP-seq peaks on
the hg38 genome.
**Update the file path, once you [Download data.zip from Google Drive](https://drive.google.com/file/d/1JUuClufcE9jrF9ilLhNqedvsy1lxvzsT/view?usp=sharing)**

## Output

For a run with TF `T`, order `m`, and `k` folds, the script writes two
files to the current working directory:

- `ROC_curve_<T>_k<k>_m<m>_allfolds.png`
- `PR_curve_<T>_k<k>_m<m>_allfolds.png`

Each figure overlays the `k` fold curves and reports per-fold metrics in
the legend. Mean auROC and mean auPRC are printed to stdout at the end of
the run.

## Notes

- The script supports Markov orders `m = 0` to `10` and fold counts
  `k = 3` to `5`. If `k` exceeds the size of the smaller class, it is
  automatically reduced and a warning is printed.
- A pseudocount of `0.5` is applied when estimating base probabilities
  from the training folds.
- Only the single chromosome specified at the prompt is used. Folds are
  created by partitioning the bins of that chromosome, as required for
  the mid-semester milestone.
- The ATAC column and the other two TFs are ignored in this milestone, as
  specified in the project brief.
