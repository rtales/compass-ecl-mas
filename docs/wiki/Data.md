# Data

Compass-ECL-MAS supports two modes:
1) **Simulated education data** (default)
2) **Real data (UCI Student Performance)** (optional)

---

## 1) Simulated education dataset
The simulated dataset is designed for controlled experiments:
- known group attribute (used for auditing only)
- configurable feature sets and noise
- multi-seed runs for robustness

This is appropriate for a *systems paper proof-of-concept* because it allows:
- controlled ground-truth behavior
- stress-testing constraints
- repeatable experiments without IRB/legal constraints

---

## 2) Real dataset: UCI Student Performance

### Source
- UCI Machine Learning Repository — Student Performance dataset
  - URL: https://archive.ics.uci.edu/dataset/320/student%2Bperformance
  - DOI (UCI): 10.24432/C5TG7T

### How we use it
We do **not** treat this dataset as a definitive public-sector deployment case study.
We use it as a *reality check*: does the pipeline behave sensibly on a real, widely-used educational dataset?

### Transformation into the project schema
We provide a script which:
- fetches the dataset using `ucimlrepo` (or direct download)
- selects and renames columns into the pipeline’s expected schema
- derives the classification target (configurable threshold on final grade)
- creates an **audit-only** sensitive attribute (e.g., SES proxy) where applicable
- outputs a single CSV used by the pipeline

Script:
```bash
python scripts/fetch_uci_student_performance.py --out data/uci_student/uci_student_processed.csv
```

### Notes
- The UCI page states “Has Missing Values? No”, but downstream conversions may introduce NaNs if you coerce types; the script handles this carefully.
- Always cite the dataset DOI above in the paper’s Data Statement if you report UCI results.

