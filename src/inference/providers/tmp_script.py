import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE_FEATURES = PROJECT_ROOT / "data" / "features" / "live_features.csv"

def main():
    df = pd.read_csv(LIVE_FEATURES)

    home = set(df["HomeTeam"].dropna().astype(str))
    away = set(df["AwayTeam"].dropna().astype(str))

    teams = sorted(home.union(away))
    print("Teams in live_features.csv:")
    for t in teams:
        print(t)

if __name__ == "__main__":
    main()