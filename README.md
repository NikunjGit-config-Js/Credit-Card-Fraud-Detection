# Credit Card Fraud Detection

Binary classification of card transactions as fraud or not fraud on the Kaggle
[Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) dataset
(European cardholders, September 2013): **284,807 transactions, 492 frauds (0.17%)**. Features are
`Time`, `Amount` and 28 anonymised PCA components `V1`–`V28`; the target is `Class`.

| File | What it is |
|---|---|
| `fraud_detection.py` | **Corrected workflow** (current results, `results/`) |
| `Credit_Card_Fraud_Detection.ipynb` | Original notebook, kept unchanged as a **historical** artifact |

## Corrected workflow (`fraud_detection.py`)

- Split first: stratified train / validation / test with the original sizes (240,000 / 22,807 / 22,000
  rows; 415 / 39 / 38 frauds), `random_state=1`.
- `Amount` (`RobustScaler`) and `Time` (min-max) are scaled with statistics from the training split only.
- Same model families as the notebook: logistic regression, the shallow Keras network (2 units; also 1
  unit in the balanced experiment), random forest (`max_depth=2`), gradient boosting, linear SVM
  (`class_weight='balanced'`). Two deliberate changes: logistic regression uses `max_iter=1000` so it
  converges, and the random models have fixed seeds.
- Two training sets: the full imbalanced training split, and a balanced set made by undersampling
  **the training split only** (415 frauds + 415 non-frauds).
- Every model is scored on the **untouched, naturally imbalanced validation split**. The model with the
  best validation PR-AUC is then scored **once** on the untouched test split.
- Metrics for the fraud class: precision, recall, F1 (default 0.5 threshold), PR-AUC (average
  precision), ROC-AUC and the confusion matrix. Accuracy is not reported: predicting "not fraud" for
  everything already scores 99.8%.

### Validation results (22,807 rows, 39 frauds; `results/validation_metrics.csv`)

| Training set | Model | Precision | Recall | F1 | PR-AUC | ROC-AUC |
|---|---|---|---|---|---|---|
| Imbalanced | Logistic regression | 0.903 | 0.718 | 0.800 | **0.830** | 0.974 |
| Imbalanced | Shallow network (2 units) | 0.868 | 0.846 | 0.857 | 0.795 | 0.983 |
| Imbalanced | Random forest (depth 2) | 0.893 | 0.641 | 0.746 | 0.818 | 0.954 |
| Imbalanced | Gradient boosting | 0.821 | 0.590 | 0.687 | 0.485 | 0.795 |
| Imbalanced | Linear SVM (balanced weights) | 0.082 | 0.974 | 0.151 | 0.796 | 0.993 |
| Undersampled | Logistic regression | 0.044 | 0.974 | 0.084 | 0.731 | 0.993 |
| Undersampled | Shallow network (2 units) | 0.050 | 0.923 | 0.095 | 0.280 | 0.982 |
| Undersampled | Shallow network (1 unit) | 0.052 | 0.923 | 0.098 | 0.433 | 0.985 |
| Undersampled | Random forest (depth 2) | 0.171 | 0.923 | 0.288 | 0.766 | 0.997 |
| Undersampled | Gradient boosting | 0.029 | 0.923 | 0.057 | 0.177 | 0.989 |
| Undersampled | Linear SVM (balanced weights) | 0.046 | 0.974 | 0.089 | 0.777 | 0.994 |

Models trained on undersampled data catch more frauds but, at the real 0.17% fraud rate, most of their
alerts are false (precision 3–17% at the default threshold). A tuned decision threshold would trade
recall for precision; thresholds were not tuned here.

### Test result of the selected model (22,000 rows, 38 frauds; `results/test_metrics.json`)

Selected on validation PR-AUC: **logistic regression trained on the imbalanced data**.

| Precision | Recall | F1 | PR-AUC | ROC-AUC | Confusion matrix (TN, FP, FN, TP) |
|---|---|---|---|---|---|
| 0.960 | 0.632 | 0.762 | 0.816 | 0.976 | 21,961 / 1 / 14 / 24 |

With only 38 test frauds these estimates are uncertain (each fraud is 2.6 points of recall).

### Reproduce

1. Download `creditcard.csv` from the Kaggle page above into `data/` (git-ignored). The run in
   `results/` used the identical file published at
   `https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv` (284,807 rows, 492
   frauds; SHA-256 in `results/run_info.json`).
2. `pip install pandas scikit-learn tensorflow`
3. `python fraud_detection.py --data data/creditcard.csv` (a few minutes on a laptop CPU).

`results/run_info.json` records the split sizes and the Python, scikit-learn, pandas and TensorFlow
versions used.

## Historical notebook (`Credit_Card_Fraud_Detection.ipynb`)

Kept unchanged for reference. Its recorded numbers are **not valid estimates** for the reasons below,
which the corrected workflow fixes.

- **Imbalanced experiment** (validation, 36 frauds): fraud-class F1 0.67 (logistic regression), 0.68
  (shallow network), 0.60 (random forest), 0.67 (gradient boosting), 0.14 (linear SVM).
- **Balanced experiment:** the shallow network scored fraud precision 0.97, recall 0.88 and accuracy
  0.93 on a 142-row test set that was 50% fraud.

Known issues in the notebook:

1. **Wrong models in three balanced reports.** The cells that train `rf_b`, `gbc_b` and `svc_b` print
   reports for `rf`, `gbc` and `svc` (the imbalanced-data models).
2. **Balanced test set.** Precision measured at 50% fraud does not describe precision at the real 0.17%.
   On the real imbalanced validation split, undersampled models reach only 3–17% precision (table above).
3. **Scaling before the split.** `RobustScaler` and the `Time` scaling were fitted on the full dataset.
4. **The imbalanced 22,000-row test split was never evaluated.**
5. **Not reproducible top to bottom.** Execution counts start at 2 with gaps, and
   `pd.read_csv('creditcard.csv')` does not use the path returned by `kagglehub.dataset_download`.
