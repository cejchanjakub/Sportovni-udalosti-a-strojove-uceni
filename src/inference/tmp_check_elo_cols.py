import json
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_FEATURES = PROJECT_ROOT / "artifacts" / "v1_model_freeze" / "1x2" / "features.json"
LIVE_FEATURES = PROJECT_ROOT / "data" / "features" / "live_features.csv"

def main():
    with open(ARTIFACT_FEATURES, "r", encoding="utf-8") as f:
        needed = json.load(f)

    df = pd.read_csv(LIVE_FEATURES, nrows=5)
    have = set(df.columns)

    missing = [c for c in needed if c not in have]
    print("Live features file:", LIVE_FEATURES)
    print("Needed feature count:", len(needed))
    print("Missing in live_features.csv:", missing)

if __name__ == "__main__":
    main()