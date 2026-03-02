import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

VAL_PATH  = DATA_DIR / "goals_poisson_val_predictions.csv"
TEST_PATH = DATA_DIR / "goals_poisson_test_predictions.csv"

val = pd.read_csv(VAL_PATH)
test = pd.read_csv(TEST_PATH)

# --- True labels ---
val_y = (val["total_goals"] >= 3).astype(int).values
test_y = (test["total_goals"] >= 3).astype(int).values

# --- Predicted probabilities ---
eps = 1e-6
val_p = val["P_over_2_5"].clip(eps, 1-eps).values
test_p = test["P_over_2_5"].clip(eps, 1-eps).values

# --- Logit transform ---
def logit(p):
    return np.log(p / (1 - p))

X_val = logit(val_p).reshape(-1, 1)
X_test = logit(test_p).reshape(-1, 1)

def print_metrics(y, p, name):
    print(f"\n{name}")
    print(f"Brier:   {brier_score_loss(y, p):.6f}")
    print(f"LogLoss: {log_loss(y, p):.6f}")

# --- Before calibration ---
print_metrics(val_y, val_p, "VAL (before)")
print_metrics(test_y, test_p, "TEST (before)")

# --- Fit Platt calibrator ---
platt = LogisticRegression(solver="lbfgs")
platt.fit(X_val, val_y)

# --- Apply calibration ---
val_p_cal = platt.predict_proba(X_val)[:, 1]
test_p_cal = platt.predict_proba(X_test)[:, 1]

# --- After calibration ---
print_metrics(val_y, val_p_cal, "VAL (after Platt)")
print_metrics(test_y, test_p_cal, "TEST (after Platt)")

# --- Save calibrated TEST ---
test_out = test.copy()
test_out["P_over_2_5_cal"] = test_p_cal
test_out["P_under_2_5_cal"] = 1.0 - test_out["P_over_2_5_cal"]

out_path = DATA_DIR / "goals_poisson_test_predictions_calibrated_over25_platt.csv"
test_out.to_csv(out_path, index=False)
print("\nSaved calibrated test predictions to:", out_path)

# --- Save calibrator ---
cal_path = DATA_DIR / "calibrator_over25_platt.joblib"
joblib.dump(platt, cal_path)
print("Saved calibrator to:", cal_path)
