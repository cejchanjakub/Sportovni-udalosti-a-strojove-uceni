import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

VAL_PATH  = DATA_DIR / "goals_poisson_val_predictions.csv"
TEST_PATH = DATA_DIR / "goals_poisson_test_predictions.csv"

LINES = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5]
EPS = 1e-6

val = pd.read_csv(VAL_PATH)
test = pd.read_csv(TEST_PATH)

results = []

def logit(p):
    return np.log(p / (1 - p))

for L in LINES:
    col = f"P_over_{str(L).replace('.', '_')}"

    # pokud nemáš předpočítané, dopočti z lambda
    if col not in val.columns:
        from scipy.stats import poisson
        k = int(np.floor(L))
        val[col] = 1.0 - poisson.cdf(k, val["lambda_goals"])
        test[col] = 1.0 - poisson.cdf(k, test["lambda_goals"])

    # binární target
    threshold = int(np.floor(L)) + 1
    val_y = (val["total_goals"] >= threshold).astype(int).values
    test_y = (test["total_goals"] >= threshold).astype(int).values

    val_p = val[col].clip(EPS, 1-EPS).values
    test_p = test[col].clip(EPS, 1-EPS).values

    # --- before ---
    res = {
        "line": L,
        "val_brier_before": brier_score_loss(val_y, val_p),
        "test_brier_before": brier_score_loss(test_y, test_p),
        "val_logloss_before": log_loss(val_y, val_p),
        "test_logloss_before": log_loss(test_y, test_p),
    }

    # --- Platt calibration ---
    X_val = logit(val_p).reshape(-1, 1)
    X_test = logit(test_p).reshape(-1, 1)

    platt = LogisticRegression(solver="lbfgs")
    platt.fit(X_val, val_y)

    val_p_cal = platt.predict_proba(X_val)[:, 1]
    test_p_cal = platt.predict_proba(X_test)[:, 1]

    res.update({
        "val_brier_after": brier_score_loss(val_y, val_p_cal),
        "test_brier_after": brier_score_loss(test_y, test_p_cal),
        "val_logloss_after": log_loss(val_y, val_p_cal),
        "test_logloss_after": log_loss(test_y, test_p_cal),
    })

    results.append(res)

df_res = pd.DataFrame(results)
out_path = DATA_DIR / "calibration_summary_multiline.csv"
df_res.to_csv(out_path, index=False)

print("Saved calibration summary to:", out_path)
print(df_res)
