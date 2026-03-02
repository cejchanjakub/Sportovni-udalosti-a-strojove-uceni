import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

train = pd.read_csv(DATA_DIR / "train.csv")
val   = pd.read_csv(DATA_DIR / "val.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")

# --- Targets ---
HOME_TARGET = "FTHG"
AWAY_TARGET = "FTAG"

# --- Features ---
NUM_FEATURES = [
    "elo_diff",
    "is_midweek",
    "days_rest_home",
    "days_rest_away",
]

TEAM_FEATURES = ["HomeTeam", "AwayTeam"]

def build_pipeline(home_model=True):
    """
    home_model=True  -> modeluje FTHG
    home_model=False -> modeluje FTAG
    """

    if home_model:
        # Home goals: HomeTeam = attack, AwayTeam = defence
        team_features = ["HomeTeam", "AwayTeam"]
    else:
        # Away goals: AwayTeam = attack, HomeTeam = defence
        team_features = ["AwayTeam", "HomeTeam"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUM_FEATURES),
            ("team", OneHotEncoder(handle_unknown="ignore"), team_features),
        ]
    )

    model = PoissonRegressor(
        alpha=1.0,      # L2 regularizace (klíčové!)
        max_iter=1000
    )

    pipe = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("model", model),
        ]
    )

    return pipe

# ------------------------
# HOME GOALS MODEL
# ------------------------
home_pipe = build_pipeline(home_model=True)
home_pipe.fit(train, train[HOME_TARGET])

val_home_mu  = home_pipe.predict(val)
test_home_mu = home_pipe.predict(test)

home_val_dev  = mean_poisson_deviance(val[HOME_TARGET], val_home_mu)
home_test_dev = mean_poisson_deviance(test[HOME_TARGET], test_home_mu)

print("\nHOME GOALS (team-aware)")
print(f"Deviance VAL:  {home_val_dev:.4f}")
print(f"Deviance TEST: {home_test_dev:.4f}")

# ------------------------
# AWAY GOALS MODEL
# ------------------------
away_pipe = build_pipeline(home_model=False)
away_pipe.fit(train, train[AWAY_TARGET])

val_away_mu  = away_pipe.predict(val)
test_away_mu = away_pipe.predict(test)

away_val_dev  = mean_poisson_deviance(val[AWAY_TARGET], val_away_mu)
away_test_dev = mean_poisson_deviance(test[AWAY_TARGET], test_away_mu)

print("\nAWAY GOALS (team-aware)")
print(f"Deviance VAL:  {away_val_dev:.4f}")
print(f"Deviance TEST: {away_test_dev:.4f}")

# ------------------------
# SAVE TEST PREDICTIONS
# ------------------------
out = test.copy()
out["lambda_home"] = test_home_mu
out["lambda_away"] = test_away_mu

out_path = DATA_DIR / "team_aware_poisson_test_predictions.csv"
out.to_csv(out_path, index=False)

print("\nSaved:", out_path)
