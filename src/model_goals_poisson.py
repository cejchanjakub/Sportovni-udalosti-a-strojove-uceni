import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_poisson_deviance
import statsmodels.api as sm
from scipy.stats import poisson

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"
PRED_DIR = PROJECT_ROOT / "data" / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(FEATURES_DIR / "train_features.csv", low_memory=False)
val   = pd.read_csv(FEATURES_DIR / "val_features.csv", low_memory=False)
test  = pd.read_csv(FEATURES_DIR / "test_features.csv", low_memory=False)

TARGET = "total_goals"

# --- Explicitní blacklist leakage sloupců (match stats / výsledky / targety) ---
LEAKAGE_COLS = {
    TARGET,
    # výsledky a góly
    "FTHG", "FTAG", "FT_Goals", "over_2_5",
    "home_win", "draw", "away_win",
    "FTR",
    # poločas (leakage)
    "HTHG", "HTAG", "HTR",
    # match stats (leakage)
    "HS", "AS", "HST", "AST", "HF", "AF", "HC", "AC", "HY", "AY", "HR", "AR",
    # agregace match stats (leakage)
    "total_corners", "total_shots", "total_fouls", "total_cards",
    "Shots", "ShotsOT", "Fouls", "Yellow", "Corners",
    # identifikátory / stringy
    "match_id", "season", "kickoff", "kickoff_dt", "Date", "Time",
    "HomeTeam", "AwayTeam", "HomeTeam_std", "AwayTeam_std",
    "HomeCoach", "AwayCoach",
}

# povolené pre-match odds (pokud existují)
ODDS_ALLOW = {"AvgH", "AvgD", "AvgA", "MaxH", "MaxD", "MaxA"}

# regex pro rolling/diff featury vytvořené v build_features_all.py
ROLL_RE = re.compile(r"^(home|away)_(goals|shots|shotsot|fouls|yellow|corners)_(for|against)_roll(3|5|10)$")
DIFF_RE = re.compile(r"^diff_(goals|shots|shotsot|fouls|yellow|corners)_(for|against)_roll(3|5|10)$")

# coach featury (prematch)
COACH_ALLOW = {
    "HomeCoachTenureDays", "AwayCoachTenureDays",
    "HomeCoachTenure_log1p", "AwayCoachTenure_log1p",
    "CoachTenureDiff",
    "NewHomeCoach_30", "NewHomeCoach_60", "NewHomeCoach_90",
    "NewAwayCoach_30", "NewAwayCoach_60", "NewAwayCoach_90",
}

def pick_features(df: pd.DataFrame) -> list[str]:
    feats = []
    for c in df.columns:
        if c in LEAKAGE_COLS:
            continue
        if c in COACH_ALLOW or c in ODDS_ALLOW:
            feats.append(c)
            continue
        if ROLL_RE.match(c) or DIFF_RE.match(c):
            feats.append(c)
            continue
    # nechceme nic nenumerického
    feats = [c for c in feats if pd.api.types.is_numeric_dtype(df[c])]
    # vyhoď all-NA
    feats = [c for c in feats if not df[c].isna().all()]
    return feats

FEATURES = pick_features(train)

if TARGET not in train.columns:
    raise ValueError(f"Target '{TARGET}' není v train_features.csv. Zkontroluj build_processed_all.py / build_features_all.py")

if len(FEATURES) == 0:
    raise ValueError("Nenašel jsem žádné použitelné pre-match numerické featury. Zkontroluj build_features_all.py výstup.")

def prepare(train_df: pd.DataFrame, df: pd.DataFrame, features: list[str]):
    # median imputace pouze z train (bez leakage)
    med = train_df[features].median(numeric_only=True)

    X = df[features].copy()
    X = X.fillna(med).fillna(0)

    X = sm.add_constant(X, has_constant="add")

    y = pd.to_numeric(df[TARGET], errors="coerce").values
    return X, y

X_train, y_train = prepare(train, train, FEATURES)
X_val, y_val     = prepare(train, val, FEATURES)
X_test, y_test   = prepare(train, test, FEATURES)

# --- Fit Poisson GLM ---
model = sm.GLM(y_train, X_train, family=sm.families.Poisson())
res = model.fit(maxiter=200)

print(res.summary())
print("\nUsed features:", len(FEATURES))

# --- Predict lambda ---
val_lambda  = res.predict(X_val)
test_lambda = res.predict(X_test)

# Safety: clip λ do rozumného rozsahu (stabilita metrik)
# pro EPL total goals je typicky ~0–8, výjimečně víc; 20 je už hodně “safe upper”
val_lambda  = np.clip(val_lambda, 1e-6, 20.0)
test_lambda = np.clip(test_lambda, 1e-6, 20.0)

# --- Metrics ---
val_dev  = mean_poisson_deviance(y_val, val_lambda)
test_dev = mean_poisson_deviance(y_test, test_lambda)

print(f"\nPoisson deviance:")
print(f"Validation: {val_dev:.4f}")
print(f"Test:       {test_dev:.4f}")

# --- Over/Under probabilities ---
def over_prob(lmbda, line: float) -> float:
    k = int(np.floor(line))
    return float(1.0 - poisson.cdf(k, lmbda))

def add_predictions(df: pd.DataFrame, lambdas: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["lambda_goals"] = lambdas
    out["P_over_2_5"] = pd.Series(lambdas).apply(lambda l: over_prob(l, 2.5)).values
    out["P_under_2_5"] = 1.0 - out["P_over_2_5"]
    return out

def save_preds(df_pred: pd.DataFrame, fname: str):
    cols = [
        "match_id", "season",
        "HomeTeam", "AwayTeam",
        "HomeTeam_std", "AwayTeam_std",
        "total_goals", "lambda_goals",
        "P_over_2_5", "P_under_2_5",
    ]
    cols = [c for c in cols if c in df_pred.columns]
    out_path = PRED_DIR / fname
    df_pred[cols].to_csv(out_path, index=False, encoding="utf-8")
    print("Saved predictions to:", out_path)

val_out = add_predictions(val, val_lambda)
test_out = add_predictions(test, test_lambda)

save_preds(val_out, "goals_poisson_val_predictions.csv")
save_preds(test_out, "goals_poisson_test_predictions.csv")
