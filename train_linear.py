# ----------------------------
# LINEAR REGRESSION FOR BATTERY SOH / PACK_SOH
# - Prevents leakage (drops target from X; if target=Pack_SOH also drops U1..U21)
# - Scales numeric + one-hot encodes categoricals
# - Stratified split by threshold; safe confusion matrix
# - Saves model, metrics JSON, and Pred vs Actual plot
# ----------------------------

# import all the tools I need for arguments, saving files, data handling, modeling, and plotting
import argparse, json, os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# import scikit-learn utilities for splitting, pipelines, preprocessing, and metrics
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, confusion_matrix


# define command-line options so I can reuse this script without changing code
parser = argparse.ArgumentParser()
parser.add_argument("--data", default="final_project_preprocessed_data.csv",
                    help="Path to preprocessed CSV")
parser.add_argument("--target", default="Pack_SOH",
                    help="Target column name to predict: 'Pack_SOH' or 'SOH'")
parser.add_argument("--threshold", type=float, default=0.6,
                    help="SOH threshold for healthy (>=thr) vs problem (<thr)")
parser.add_argument("--test_size", type=float, default=0.2,
                    help="Fraction of data used for testing")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")
args = parser.parse_args()


# load my dataset from CSV
df = pd.read_csv(args.data)

# normalize column names (strip spaces) so matching by name is reliable
df.columns = df.columns.str.strip()

# find the exact target column name in a case-insensitive way
target_norm = args.target.strip().lower()
cols_lower = df.columns.str.strip().str.lower()
# validate that the target column exists
if target_norm not in set(cols_lower):
    raise ValueError(f"Target column '{args.target}' not found in CSV columns: {list(df.columns)}")
# get the exact column name as in the dataframe
target_col = df.columns[cols_lower == target_norm][0]

# set y to the chosen target and start X as "all other columns"
y = df[target_col].astype(float)
X = df.drop(columns=[target_col]).copy()

# remove any obvious label variants from X to avoid label leakage
# (e.g., if target is 'SOH', drop any 'Pack_SOH' column from features, and vice versa)
for leak in ["soh", "pack_soh"]:
    if leak != target_norm:
        X = X.drop(columns=[c for c in X.columns if c.strip().lower() == leak], errors="ignore")

# If predicting Pack_SOH, also drop U1..U21 because Pack_SOH = mean(U1..U21) (would cause trivial R^2=1)
if target_norm == "pack_soh":
    u_cols = [f"U{i}" for i in range(1, 22) if f"U{i}" in X.columns]
    X = X.drop(columns=u_cols, errors="ignore")

# build a binary label from y using the threshold (used only for stratification and classification metrics)
thr = args.threshold
y_cls_all = (y >= thr).astype(int)

# split the data into train/test, trying to stratify by the class so both classes appear in each split
try:
    X_train, X_test, y_train, y_test, y_train_cls, y_test_cls = train_test_split(
        X, y, y_cls_all, test_size=args.test_size, random_state=args.seed, stratify=y_cls_all
    )
except ValueError:
    # If the whole dataset is single-class at this threshold, fall back to a normal split
    X_train, X_test, y_train, y_test, y_train_cls, y_test_cls = train_test_split(
        X, y, y_cls_all, test_size=args.test_size, random_state=args.seed
    )

# figure out which columns are numeric vs categorical to preprocess correctly
num_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
cat_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

# print them once to verify what the model sees
print("Numeric columns:", num_cols)
print("Categorical columns:", cat_cols)

# create a OneHotEncoder that works across sklearn versions
try:
    # Newer sklearn (>=1.2)
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
except TypeError:
    # Older sklearn
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

# build a column transformer: scale numerics and one-hot encode categoricals
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(with_mean=True, with_std=True), num_cols),
        ("cat", ohe, cat_cols),
    ],
    remainder="drop"
)

# assemble a pipeline that preprocesses then runs Linear Regression
pipe = Pipeline([
    ("prep", preprocessor),
    ("lr", LinearRegression())
])

# train (fit) the pipeline on the training data
pipe.fit(X_train, y_train)

# predict on the test data (preprocessing is applied automatically)
y_pred = pipe.predict(X_test)

# compute regression metrics to evaluate the model quality
r2  = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

# also run 5-fold cross-validation on the training set to see performance stability
cv = KFold(n_splits=5, shuffle=True, random_state=args.seed)
cv_r2 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="r2")

# convert regression predictions into healthy/problem using the threshold
y_pred_cls = (y_pred >= thr).astype(int)

# compute a safe 2x2 confusion matrix (force labels=[0,1] so shape is predictable)
cm = confusion_matrix(y_test_cls, y_pred_cls, labels=[0, 1])

# derive tn, fp, fn, tp robustly (handling single-class corner cases)
if cm.shape == (2, 2):
    tn, fp, fn, tp = cm.ravel()
elif cm.shape == (1, 1):
    # Only one class present in both y_true and y_pred
    if int(y_test_cls.iloc[0]) == 0:
        tn, fp, fn, tp = cm[0, 0], 0, 0, 0
    else:
        tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
elif cm.shape == (1, 2):   # only one true class present
    tn, fp = cm[0, 0], cm[0, 1]
    fn, tp = 0, 0
elif cm.shape == (2, 1):   # only one predicted class present
    tn, fn = cm[0, 0], cm[1, 0]
    fp, tp = 0, 0
else:
    tn, fp, fn, tp = 0, 0, 0, 0

# compute classification accuracy from the confusion matrix values
total = tn + fp + fn + tp
clf_acc = (tn + tp) / total if total > 0 else 0.0

# print results clearly so I can paste them into my report
print("\n=== Linear Regression (Battery Pack SOH) ===")
print(f"Target: {target_col}")
print(f"Test R^2 : {r2:.4f}")
print(f"Test MSE : {mse:.6f}")
print(f"Test RMSE: {rmse:.6f}")
print(f"Test MAE : {mae:.6f}")
print(f"5-fold CV R^2 (train): mean={cv_r2.mean():.4f}, std={cv_r2.std():.4f}")
print(f"\nThreshold = {thr:.2f} (>= healthy)")
print(f"Class distribution in TEST (true): healthy={int((y_test_cls==1).sum())}, problem={int((y_test_cls==0).sum())}")
print(f"Classification accuracy (on test): {clf_acc:.4f}")
print(f"Confusion matrix [tn, fp; fn, tp]: [{tn}, {fp}; {fn}, {tp}]")

# ensure folders exist for saving artifacts
os.makedirs("models", exist_ok=True)
os.makedirs("figs", exist_ok=True)

# save the trained pipeline so I can reuse it later (same preprocessing at inference)
joblib.dump(pipe, "models/linreg_soh.pkl")

# save all key metrics (and the feature lists) to JSON for easy inspection
with open("models/test_metrics.json", "w") as f:
    json.dump({
        "target": target_col,
        "r2": float(r2),
        "mse": float(mse),
        "rmse": float(rmse),
        "mae": float(mae),
        "cv_r2_mean": float(cv_r2.mean()),
        "cv_r2_std": float(cv_r2.std()),
        "threshold": float(thr),
        "clf_accuracy": float(clf_acc),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "numeric_columns": num_cols,
        "categorical_columns": cat_cols,
        "test_class_counts": {
            "healthy": int((y_test_cls==1).sum()),
            "problem": int((y_test_cls==0).sum())
        }
    }, f, indent=2)

# plot Predicted vs Actual to visually see how close we are to the ideal 45° line
plt.figure()
plt.scatter(y_test, y_pred, alpha=0.6)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, linestyle="--")
plt.xlabel(f"Actual {target_col}")
plt.ylabel(f"Predicted {target_col}")
plt.title("Predicted vs Actual SOH (Linear Regression)")
plt.tight_layout()
plt.savefig("figs/pred_vs_actual.png", dpi=200)
# plt.show()  # keep this off so the script runs headless

print("\nSaved model -> models/linreg_soh.pkl")
print("Saved metrics -> models/test_metrics.json")
print("Saved plot -> figs/pred_vs_actual.png")
