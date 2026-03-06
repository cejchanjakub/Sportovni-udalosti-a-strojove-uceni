# src/model_1X2.py

from pathlib import Path
import argparse
import json
import datetime as dt
import hashlib
import joblib

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import CalibratedClassifierCV


# ============================================================
# CONFIG
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = PROJECT_ROOT / "data" / "features"


# ============================================================
# UTIL
# ============================================================

def _hash_file(path: Path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return h.hexdigest()


def pick_features(df: pd.DataFrame) -> list[str]:
    """
    Výběr featur pro 1X2 model.
    Zahrnuje: ELO, formu (rolling win rate obě varianty),
    tabulkovou pozici, days rest a coach features.
    Nezahrnuje: match stats (leakage), bookmaker odds.
    """
    feats = []

    # ELO – nejsilnější single feature pro 1X2
    for c in ["elo_home", "elo_away", "elo_diff"]:
        if c in df.columns:
            feats.append(c)

    # Forma – rolling win rate (varianta A: všechny zápasy)
    for c in df.columns:
        if c.startswith("home_points_roll") or c.startswith("away_points_roll") or c.startswith("diff_points_roll"):
            feats.append(c)

    # Forma – rolling win rate (varianta B: home/away split)
    for c in df.columns:
        if c.startswith("home_form_home_roll") or c.startswith("away_form_away_roll") or c.startswith("diff_form_ha_roll"):
            feats.append(c)

    # Tabulková pozice
    for c in ["home_table_pos", "away_table_pos", "table_pos_diff",
              "home_table_points", "away_table_points", "table_points_diff"]:
        if c in df.columns:
            feats.append(c)

    # Days rest
    for c in ["home_days_rest", "away_days_rest", "days_rest_diff", "is_midweek"]:
        if c in df.columns:
            feats.append(c)

    # Coach features
    for c in ["HomeCoachTenureDays", "AwayCoachTenureDays", "CoachTenureDiff",
              "NewHomeCoach_30", "NewAwayCoach_30",
              "HomeCoachTenure_log1p", "AwayCoachTenure_log1p"]:
        if c in df.columns:
            feats.append(c)

    # Goals rolling (jako proxy formy – sekundární signal)
    for c in df.columns:
        if ("goals_for_roll" in c or "goals_against_roll" in c or "diff_goals_" in c):
            feats.append(c)

    # Deduplikace při zachování pořadí
    seen = set()
    out = []
    for c in feats:
        if c not in seen and pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
            seen.add(c)

    return out


def build_target(df: pd.DataFrame) -> np.ndarray:
    """
    0 = výhra domácích (H)
    1 = remíza (D)
    2 = výhra hostů (A)
    """
    conditions = [
        df["FTHG"] > df["FTAG"],
        df["FTHG"] == df["FTAG"],
    ]
    return np.select(conditions, [0, 1], default=2).astype(int)


def prepare_xy(
    df: pd.DataFrame,
    feats: list[str],
    train_median: pd.Series | None,
    fit_median: bool,
) -> tuple[np.ndarray, pd.Series]:
    """
    Median imputace pouze z train setu (no leakage).
    """
    X = df[feats].copy()
    X = X.replace([np.inf, -np.inf], np.nan)

    if fit_median:
        train_median = X.median()

    X = X.fillna(train_median).fillna(0)
    return X, train_median


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_artifacts", action="store_true")
    ap.add_argument("--version", default="v1_model_freeze")
    ap.add_argument("--calibrate", action="store_true", help="Kalibrovat pravděpodobnosti (isotonic)")
    ap.add_argument("--C", type=float, default=1.0, help="Regularizační parametr logistické regrese")
    args = ap.parse_args()

    train = pd.read_csv(FEATURES_DIR / "train_features.csv", low_memory=False)
    val   = pd.read_csv(FEATURES_DIR / "val_features.csv",   low_memory=False)
    test  = pd.read_csv(FEATURES_DIR / "test_features.csv",  low_memory=False)

    feats = pick_features(train)

    # Ověř že featury existují i v val/test
    feats = [c for c in feats if c in val.columns and c in test.columns]

    if len(feats) == 0:
        raise RuntimeError("Nenašel jsem žádné použitelné featury. Zkontroluj build_features_all.py.")

    print(f"\nPoužité featury ({len(feats)}):")
    for c in feats:
        print(f"  {c}")

    # Median imputace z train setu
    X_train, train_median = prepare_xy(train, feats, None, fit_median=True)
    X_val,   _            = prepare_xy(val,   feats, train_median, fit_median=False)
    X_test,  _            = prepare_xy(test,  feats, train_median, fit_median=False)

    y_train = build_target(train)
    y_val   = build_target(val)
    y_test  = build_target(test)

    # Standardizace (fit pouze na train)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    # Model
    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        C=args.C,
    )
    model.fit(X_train_sc, y_train)

    # Kalibrace
    if args.calibrate:
        calibrator = CalibratedClassifierCV(
            estimator=model,
            method="isotonic",
            cv=5,
        )
        calibrator.fit(X_train_sc, y_train)
        model = calibrator

    # Evaluace
    def evaluate(X, y, name):
        probs = model.predict_proba(X)
        ll = log_loss(y, probs)
        brier = float(np.mean([
            brier_score_loss((y == i).astype(int), probs[:, i])
            for i in range(3)
        ]))
        acc = float(np.mean(model.predict(X) == y))
        print(f"{name}: LogLoss={ll:.4f} | Brier={brier:.4f} | Accuracy={acc:.4f}")
        return {"log_loss": ll, "brier": brier, "accuracy": acc}

    print("\n--- Evaluace ---")
    m_train = evaluate(X_train_sc, y_train, "TRAIN")
    m_val   = evaluate(X_val_sc,   y_val,   "VAL  ")
    m_test  = evaluate(X_test_sc,  y_test,  "TEST ")

    # Distribuce predikcí
    print("\n--- Distribuce predikcí (TEST) ---")
    probs_test = model.predict_proba(X_test_sc)
    print(f"  Průměrná P(H)={probs_test[:,0].mean():.3f} | P(D)={probs_test[:,1].mean():.3f} | P(A)={probs_test[:,2].mean():.3f}")
    print(f"  Skutečné rozložení: H={np.mean(y_test==0):.3f} | D={np.mean(y_test==1):.3f} | A={np.mean(y_test==2):.3f}")

    # ============================================================
    # ULOŽENÍ ARTEFAKTŮ
    # ============================================================
    if args.save_artifacts:
        artifact_dir = PROJECT_ROOT / "artifacts" / args.version / "1x2"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(model,  artifact_dir / "model.joblib")
        joblib.dump(scaler, artifact_dir / "scaler.joblib")

        with open(artifact_dir / "features.json", "w", encoding="utf-8") as f:
            json.dump(feats, f, indent=2, ensure_ascii=False)

        meta = {
            "saved_at": dt.datetime.now().isoformat(timespec="seconds"),
            "target": "1X2",
            "model": "LogisticRegression (multinomial)",
            "C": args.C,
            "calibrated": args.calibrate,
            "n_features": len(feats),
            "features": feats,
            "metrics": {"train": m_train, "val": m_val, "test": m_test},
            "data_hash_train": _hash_file(FEATURES_DIR / "train_features.csv"),
            "data_hash_val":   _hash_file(FEATURES_DIR / "val_features.csv"),
            "data_hash_test":  _hash_file(FEATURES_DIR / "test_features.csv"),
        }

        with open(artifact_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        print(f"\n[ARTIFACTS SAVED] {artifact_dir}")
        print(f"  - model.joblib")
        print(f"  - scaler.joblib")
        print(f"  - features.json ({len(feats)} featur)")
        print(f"  - meta.json")


if __name__ == "__main__":
    main()