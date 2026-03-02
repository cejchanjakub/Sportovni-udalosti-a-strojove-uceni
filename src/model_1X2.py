# src/model_1x2.py

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


def pick_features(df: pd.DataFrame):

    feats = []

    # goals rolling
    feats += [c for c in df.columns if "goals_for_roll" in c]
    feats += [c for c in df.columns if "goals_against_roll" in c]
    feats += [c for c in df.columns if "diff_goals_" in c]

    # elo
    for c in ["elo_home", "elo_away", "elo_diff"]:
        if c in df.columns:
            feats.append(c)

    feats = sorted(set(feats))
    return feats


def build_target(df: pd.DataFrame):

    y = []
    for _, row in df.iterrows():
        if row["FTHG"] > row["FTAG"]:
            y.append(0)  # Home
        elif row["FTHG"] == row["FTAG"]:
            y.append(1)  # Draw
        else:
            y.append(2)  # Away

    return np.array(y)


# ============================================================
# MAIN
# ============================================================

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--save_artifacts", action="store_true")
    ap.add_argument("--version", default="v1_model_freeze")
    ap.add_argument("--calibrate", action="store_true")
    args = ap.parse_args()

    train = pd.read_csv(FEATURES_DIR / "train_features.csv")
    val = pd.read_csv(FEATURES_DIR / "val_features.csv")
    test = pd.read_csv(FEATURES_DIR / "test_features.csv")

    feats = pick_features(train)

    X_train = train[feats].fillna(0)
    X_val = val[feats].fillna(0)
    X_test = test[feats].fillna(0)

    y_train = build_target(train)
    y_val = build_target(val)
    y_test = build_target(test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs"
    )

    model.fit(X_train, y_train)

    if args.calibrate:
        calibrator = CalibratedClassifierCV(
            estimator=model,
            method="isotonic",
            cv=5
        )

        calibrator.fit(X_train, y_train)
        model = calibrator

    def evaluate(X, y, name):
        probs = model.predict_proba(X)
        ll = log_loss(y, probs)
        brier = np.mean([
            brier_score_loss((y == i).astype(int), probs[:, i])
            for i in range(3)
        ])
        print(f"{name}: LogLoss={ll:.4f} | Brier={brier:.4f}")

    print("\n--- Evaluation ---")
    evaluate(X_train, y_train, "TRAIN")
    evaluate(X_val, y_val, "VAL")
    evaluate(X_test, y_test, "TEST")

    # ========================================================
    # SAVE
    # ========================================================

    if args.save_artifacts:

        artifact_dir = PROJECT_ROOT / "artifacts" / args.version / "1x2"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(model, artifact_dir / "model.joblib")
        joblib.dump(scaler, artifact_dir / "scaler.joblib")

        with open(artifact_dir / "features.json", "w") as f:
            json.dump(feats, f, indent=2)

        meta = {
            "saved_at": dt.datetime.now().isoformat(),
            "features": feats,
            "n_features": len(feats),
            "calibrated": args.calibrate,
            "data_hash_train": _hash_file(FEATURES_DIR / "train_features.csv"),
        }

        with open(artifact_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"\nArtifacts saved to: {artifact_dir}")


if __name__ == "__main__":
    main()
