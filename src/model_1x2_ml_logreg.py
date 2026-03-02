import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, accuracy_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

train = pd.read_csv(DATA_DIR / "train.csv")
val   = pd.read_csv(DATA_DIR / "val.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")

# --- Create 1X2 target: 0 home, 1 draw, 2 away ---
def make_y(df):
    return np.where(
        df["FTHG"] > df["FTAG"], 0,
        np.where(df["FTHG"] == df["FTAG"], 1, 2)
    )

y_train = make_y(train)
y_val   = make_y(val)
y_test  = make_y(test)

# --- Feature selection (start small, stable) ---
NUM_FEATURES = [
    "elo_diff",
    "gd_last5_home",
    "gd_last5_away",
    "goals_expected_total_proxy_last10",
    "days_rest_home",
    "days_rest_away",
    "is_midweek",
]

CAT_FEATURES = ["HomeTeam", "AwayTeam"]

# fill missing numeric
for c in NUM_FEATURES:
    for df in (train, val, test):
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

preprocess = ColumnTransformer(
    transformers=[
        ("num", Pipeline(steps=[("scaler", StandardScaler())]), NUM_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ]
)

clf = LogisticRegression(
    solver="lbfgs",
    max_iter=2000,
    C=1.0
)

pipe = Pipeline(steps=[("prep", preprocess), ("clf", clf)])
pipe.fit(train, y_train)

# predict probabilities
p_val  = pipe.predict_proba(val)
p_test = pipe.predict_proba(test)

# metrics
def multiclass_brier(y_true, probs):
    y_onehot = np.zeros_like(probs)
    y_onehot[np.arange(len(y_true)), y_true] = 1.0
    return np.mean(np.sum((probs - y_onehot) ** 2, axis=1))

val_ll = log_loss(y_val, p_val, labels=[0,1,2])
test_ll = log_loss(y_test, p_test, labels=[0,1,2])

val_brier = multiclass_brier(y_val, p_val)
test_brier = multiclass_brier(y_test, p_test)

val_acc = accuracy_score(y_val, np.argmax(p_val, axis=1))
test_acc = accuracy_score(y_test, np.argmax(p_test, axis=1))

print("ML 1X2 Logistic Regression")
print(f"VAL  LogLoss: {val_ll:.4f} | Brier: {val_brier:.4f} | Acc: {val_acc:.4f}")
print(f"TEST LogLoss: {test_ll:.4f} | Brier: {test_brier:.4f} | Acc: {test_acc:.4f}")

# save predictions
out = test[["match_id","season","HomeTeam","AwayTeam","FTHG","FTAG"]].copy()
out["p_home_win_ml"] = p_test[:,0]
out["p_draw_ml"] = p_test[:,1]
out["p_away_win_ml"] = p_test[:,2]

out_path = DATA_DIR / "match_odds_1x2_ml_logreg_test.csv"
out.to_csv(out_path, index=False)
print("Saved:", out_path)
