import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "EPL_2024-25_processed.csv"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "EPL_2024-25_features.csv"

df = pd.read_csv(DATA_PATH, parse_dates=["kickoff"])
df = df.sort_values("kickoff").reset_index(drop=True)

WINDOW = 5

def rolling_mean_prev_matches(series: pd.Series, window: int) -> pd.Series:
    # posun o 1 => nepoužijeme aktuální zápas (žádný leakage)
    return series.shift(1).rolling(window=window, min_periods=window).mean()

# goals scored / conceded (pre-match rolling means)
df["home_goals_scored_avg"] = df.groupby("HomeTeam")["FTHG"].transform(
    lambda s: rolling_mean_prev_matches(s, WINDOW)
)
df["home_goals_conceded_avg"] = df.groupby("HomeTeam")["FTAG"].transform(
    lambda s: rolling_mean_prev_matches(s, WINDOW)
)

df["away_goals_scored_avg"] = df.groupby("AwayTeam")["FTAG"].transform(
    lambda s: rolling_mean_prev_matches(s, WINDOW)
)
df["away_goals_conceded_avg"] = df.groupby("AwayTeam")["FTHG"].transform(
    lambda s: rolling_mean_prev_matches(s, WINDOW)
)

# differences
df["goals_scored_diff"] = df["home_goals_scored_avg"] - df["away_goals_scored_avg"]
df["goals_conceded_diff"] = df["away_goals_conceded_avg"] - df["home_goals_conceded_avg"]

# drop early rows without enough history
feature_cols = [
    "home_goals_scored_avg", "home_goals_conceded_avg",
    "away_goals_scored_avg", "away_goals_conceded_avg",
    "goals_scored_diff", "goals_conceded_diff"
]

before = df.shape[0]
df_features = df.dropna(subset=feature_cols).copy()
after = df_features.shape[0]

print("Rows before:", before)
print("Rows after feature engineering:", after)
print("Dropped due to insufficient history:", before - after)

df_features.to_csv(OUT_PATH, index=False)
print(f"Saved features dataset to: {OUT_PATH}")
