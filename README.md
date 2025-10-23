# SOFE3370 — Linear Regression Milestone (Preprocessing + Training)

This submission contains my preprocessing output and a Linear Regression pipeline to predict **Pack_SOH** (pack-level state of health). We also implemented the threshold-based “healthy/problem” classification, where the threshold is user-configurable.

---

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install -U pip
# install from the requirements section below (or save it as requirements.txt first)
python -m pip install -r requirements.txt
```

## How to run

We ran two evaluations to match the assignment:

- Run A — rubric threshold 0.6

- Run B — scale-appropriate threshold 3.8 (Pack_SOH ≈ 3.5–4.0), to yield a non-trivial confusion matrix

```bash
# Run A — rubric threshold (0.6)
python train_linear.py --data final_project_preprocessed_data.csv --target Pack_SOH --threshold 0.6

# Run B — practical threshold on this scale (3.8)
python train_linear.py --data final_project_preprocessed_data.csv --target Pack_SOH --threshold 3.8
```

You can change the decision rule by adjusting --threshold <value> without modifying the code.

## What the code does (short)

Loads the preprocessed CSV and prevents leakage:

- Drops the target from features.

- When predicting Pack_SOH, it also drops U1..U21 (since Pack_SOH was computed from those during preprocessing).

Preprocessing via ColumnTransformer:

- StandardScaler on numeric (No., Qn, Q, Pt, SOC, SOE)

- OneHotEncoder on categorical (Mat, ID)

- Train/test split (stratified by thresholded labels when possible).

Trains LinearRegression and evaluates:

- Regression: R², MSE, RMSE, MAE, and 5-fold CV R² (mean ± std)

- Classification (threshold rule): accuracy + confusion matrix

- Saves metrics and the plot for easy marking.

## Artifacts (for marking)

- `train_linear.py` — training script (pipeline + metrics)

- `final_project_preprocessed_data.csv` — preprocessed data (includes Pack_SOH)

- `figs/pred_vs_actual.png` — Predicted vs Actual scatter with 45° line

- `models/test_metrics.json` — all numbers (regression + classification) and the exact feature lists used

## Results

### Run A — Threshold = **0.6** (as per rubric)
*(On this scale, 0.6 makes all samples “healthy,” so classification is trivial; regression metrics still valid.)*

**Target:** `Pack_SOH`

**Regression metrics**
| R² | MSE | RMSE | MAE | 5-fold CV R² (mean ± std) |
|---:|---:|---:|---:|:---------------------------|
| 0.9955008017572972 | 7.569849604207866e-05 | 0.008700488264579101 | 0.006860438851389045 | 0.9955681761312537 ± 0.00039375949931226583 |

**Classification (≥ 0.6 = healthy)**
| Accuracy | Confusion [tn, fp; fn, tp] | Test class counts (healthy / problem) |
|---:|:-------------------------------:|:-------------------------------------:|
| 1.0 | [0, 0; 0, 134] | 134 / 0 |


### Run B — Threshold = **3.8** (meaningful split on ~3.5–4.0 scale)

**Target:** `Pack_SOH`

**Regression metrics**
| R² | MSE | RMSE | MAE | 5-fold CV R² (mean ± std) |
|---:|---:|---:|---:|:---------------------------|
| 0.9951298763523511 | 7.758755632286618e-05 | 0.008808379892061092 | 0.006817631771606307 | 0.9955606537573507 ± 0.0004702138555643102 |

**Classification (≥ 3.8 = healthy)**
| Accuracy | Confusion [tn, fp; fn, tp] | Test class counts (healthy / problem) |
|---:|:-------------------------------:|:-------------------------------------:|
| 0.9850746268656716 | [94, 0; 2, 38] | 40 / 94 |

## Notes for the TA

The saved `models/test_metrics.json` also includes numeric_columns and categorical_columns to verify no leakage (neither the target nor U1..U21 are in the model inputs when predicting Pack_SOH).

You can re-run with a different --threshold to see another confusion matrix without changing any code.

