import pandas as pd
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"

SPLITS = ["train", "val", "test", "live"]

IN_SUFFIX = "_with_coaches.csv"
OUT_SUFFIX = "_with_coach_features.csv"


def add_manager_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # Tenure columns musí existovat (z add_coaches_to_matches.py)
    if "HomeCoachTenureDays" not in out.columns or "AwayCoachTenureDays" not in out.columns:
        raise ValueError("Chybí HomeCoachTenureDays/AwayCoachTenureDays. Nejprve spusť add_coaches_to_matches.py")

    # ensure numeric
    out["HomeCoachTenureDays"] = pd.to_numeric(out["HomeCoachTenureDays"], errors="coerce")
    out["AwayCoachTenureDays"] = pd.to_numeric(out["AwayCoachTenureDays"], errors="coerce")

    # log transform (robustní na NaN)
    out["HomeCoachTenure_log1p"] = np.log1p(out["HomeCoachTenureDays"])
    out["AwayCoachTenure_log1p"] = np.log1p(out["AwayCoachTenureDays"])

    # diff
    out["CoachTenureDiff"] = out["HomeCoachTenureDays"] - out["AwayCoachTenureDays"]

    # new coach flags
    for d in [30, 60, 90]:
        out[f"NewHomeCoach_{d}"] = (out["HomeCoachTenureDays"] <= d).astype("Int64")
        out[f"NewAwayCoach_{d}"] = (out["AwayCoachTenureDays"] <= d).astype("Int64")

        # pokud tenure chybí, flag má být NA (ne 0)
        out.loc[out["HomeCoachTenureDays"].isna(), f"NewHomeCoach_{d}"] = pd.NA
        out.loc[out["AwayCoachTenureDays"].isna(), f"NewAwayCoach_{d}"] = pd.NA

    return out


def main():
    saved = []

    for split in SPLITS:
        in_path = DATA_DIR / f"{split}{IN_SUFFIX}"
        if not in_path.exists():
            print(f"⚠️ Přeskakuju {split}, neexistuje: {in_path}")
            continue

        df = pd.read_csv(in_path, low_memory=False)

        out = add_manager_features(df)

        out_path = DATA_DIR / f"{split}{OUT_SUFFIX}"
        out.to_csv(out_path, index=False, encoding="utf-8")
        saved.append(out_path)

        # rychlý sanity log
        miss_h = out["HomeCoachTenureDays"].isna().mean() * 100
        miss_a = out["AwayCoachTenureDays"].isna().mean() * 100
        print(f"[{split}] Saved: {out_path}")
        print(f"[{split}] Missing HomeCoachTenureDays: {miss_h:.2f}% | Away: {miss_a:.2f}%")

        preview_cols = [
            "HomeTeam_std", "AwayTeam_std", "kickoff_dt",
            "HomeCoach", "AwayCoach",
            "HomeCoachTenureDays", "AwayCoachTenureDays",
            "NewHomeCoach_30", "NewAwayCoach_30",
            "CoachTenureDiff"
        ]
        preview_cols = [c for c in preview_cols if c in out.columns]
        print(out[preview_cols].head(3).to_string(index=False))
        print()

    print("Done. Files:")
    for p in saved:
        print(" -", p)


if __name__ == "__main__":
    main()
