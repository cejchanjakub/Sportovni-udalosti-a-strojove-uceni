import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "features" / "train_features.csv"

print("Loading:", DATA_PATH)

df = pd.read_csv(DATA_PATH)

print("\nELO columns:")
print([c for c in df.columns if "elo" in c.lower()])

print("\nCoach columns:")
print([c for c in df.columns if "coach" in c.lower()])

print("\nTable columns:")
print([c for c in df.columns if "table" in c.lower()])
