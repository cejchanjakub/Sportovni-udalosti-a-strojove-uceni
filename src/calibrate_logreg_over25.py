import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "EPL_2024-25_features.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["kickoff"])
df = df.sort_values("kickoff").reset_index(drop=True)

feature_cols = [
    "home_goals_scored_avg", "home_goals_conceded_avg",
    "away_goals_scored_avg", "away_goals_conceded_avg",
    "goals_scored_diff", "goals_conceded_diff",
]

X = df[feature_cols]
y = df["over_2_5"].astype(int)

# --- time split ---
split_idx = int(len(df) * 0.7)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# --- base model ---
base_model = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=200))
])

base_model.fit(X_train, y_train)
p_uncal = base_model.predict_proba(X_test)[:, 1]

# --- Platt scaling ---
platt = CalibratedClassifierCV(
    estimator=base_model,
    method="sigmoid",
    cv=3
)
platt.fit(X_train, y_train)
p_platt = platt.predict_proba(X_test)[:, 1]

# --- Isotonic regression ---
iso = CalibratedClassifierCV(
    estimator=base_model,
    method="isotonic",
    cv=3
)
iso.fit(X_train, y_train)
p_iso = iso.predict_proba(X_test)[:, 1]

# --- metrics ---
def report(name, y_true, p):
    p = np.clip(p, 1e-15, 1 - 1e-15)
    print(f"{name:12s} | Brier: {brier_score_loss(y_true, p):.4f} | LogLoss: {log_loss(y_true, p):.4f}")

print("\n=== Calibration comparison (Over 2.5) ===")
report("Uncalibrated", y_test, p_uncal)
report("Platt", y_test, p_platt)
report("Isotonic", y_test, p_iso)
