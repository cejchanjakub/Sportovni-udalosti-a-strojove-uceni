import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "EPL_2024-25_features.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["kickoff"])
df = df.sort_values("kickoff").reset_index(drop=True)

# --- features + target ---
feature_cols = [
    "home_goals_scored_avg", "home_goals_conceded_avg",
    "away_goals_scored_avg", "away_goals_conceded_avg",
    "goals_scored_diff", "goals_conceded_diff",
]
X = df[feature_cols]
y = df["over_2_5"].astype(int)

# --- time split 70/30 ---
split_idx = int(len(df) * 0.7)
X_train = X.iloc[:split_idx].copy()
X_test = X.iloc[split_idx:].copy()
y_train = y.iloc[:split_idx].copy()
y_test = y.iloc[split_idx:].copy()

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])
print(
    "Train period:",
    df.loc[:split_idx - 1, "kickoff"].min().date(),
    "→",
    df.loc[:split_idx - 1, "kickoff"].max().date(),
)
print(
    "Test period:",
    df.loc[split_idx:, "kickoff"].min().date(),
    "→",
    df.loc[split_idx:, "kickoff"].max().date(),
)

# --- baseline ---
p_base = float(y_train.mean())

p_base_test = np.full(shape=len(y_test), fill_value=p_base, dtype=float)
p_base_test = np.clip(p_base_test, 1e-15, 1 - 1e-15)

base_brier = brier_score_loss(y_test, p_base_test)
base_logloss = log_loss(y_test, p_base_test)

print("\nBaseline P(Over2.5) from train:", round(p_base, 4))
print("Baseline Brier:", round(base_brier, 4))
print("Baseline LogLoss:", round(base_logloss, 4))

# --- model pipeline ---
model = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=200, solver="lbfgs"))
])

model.fit(X_train, y_train)
p_pred = model.predict_proba(X_test)[:, 1]
p_pred = np.clip(p_pred, 1e-15, 1 - 1e-15)

ml_brier = brier_score_loss(y_test, p_pred)
ml_logloss = log_loss(y_test, p_pred)

print("\nLogReg Brier:", round(ml_brier, 4))
print("LogReg LogLoss:", round(ml_logloss, 4))

# --- save predictions for calibration step ---
out = df.iloc[split_idx:].copy()
out["p_base"] = p_base
out["p_logreg"] = p_pred

OUT_PATH = PROJECT_ROOT / "data" / "processed" / "over25_predictions.csv"
out[["match_id", "kickoff", "HomeTeam", "AwayTeam", "over_2_5", "p_base", "p_logreg"]].to_csv(OUT_PATH, index=False)

print("\nSaved predictions to:", OUT_PATH)
