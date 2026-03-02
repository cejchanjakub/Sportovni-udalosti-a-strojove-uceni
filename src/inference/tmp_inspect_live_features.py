import pandas as pd
from pathlib import Path

# vždy kořen projektu: .../dp_sazeni_ml
PROJECT_ROOT = Path(__file__).resolve().parents[2]
LIVE_FEATURES = PROJECT_ROOT / "data" / "features" / "live_features.csv"

def main():
    print("PROJECT_ROOT:", PROJECT_ROOT)
    print("LIVE_FEATURES:", LIVE_FEATURES)

    if not LIVE_FEATURES.exists():
        raise FileNotFoundError(f"Soubor neexistuje: {LIVE_FEATURES}")

    df = pd.read_csv(LIVE_FEATURES)

    print("\nColumns (count={}):".format(len(df.columns)))
    print(df.columns.tolist())

    cand = [c for c in df.columns if any(k in c.lower() for k in ["date", "team", "home", "away", "match", "id"])]
    print("\nCandidate ID columns:", cand)

    if cand:
        print("\nFirst 3 rows (candidate cols):")
        print(df[cand].head(3).to_string(index=False))
    else:
        print("\nFirst 3 rows:")
        print(df.head(3).to_string(index=False))

if __name__ == "__main__":
    main()