# src/model_count_glm.py
# ------------------------------------------------------------
# Count-model (Poisson / NegBin) pro různé betting trhy (goals, corners, shots, fouls, yellow)
#
# MODEL FREEZE úpravy:
#  - NEPOUŽÍVAT bookmaker odds (AvgH/AvgD/AvgA/MaxH/MaxD/MaxA)
#  - Referee featury pouze pro fouls + yellow/cards (dle --prefix, ne dle --target)
#  - Standardizace: fit scaler pouze na TRAIN, následně transform VAL/TEST
#  - Trénujeme na data/features/*_features.csv (ne na data/processed)
#  - Ukládání artefaktů: --save_artifacts (model + scaler + features.json + meta.json)
# ------------------------------------------------------------

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import List, Tuple, Optional

import joblib
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


# ----------------------------
# Helpers
# ----------------------------

def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_split(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Chybí soubor: {csv_path}")
    return pd.read_csv(csv_path)


def safe_median_impute(df: pd.DataFrame) -> pd.DataFrame:
    out = df.replace([np.inf, -np.inf], np.nan).copy()
    med = out.median(numeric_only=True)
    out = out.fillna(med).fillna(0)
    return out


def pick_feature_columns(df: pd.DataFrame, prefix: str) -> List[str]:
    """
    Výběr featur podle prefixu (freeze-friendly).
    - Primárně bereme rolling a diff rolling featury pro daný trh.
    - Přidáme globální featury, pokud existují (ELO, rest days, coach tenure atd.).
    - Nepřidáváme bookmaker odds (AvgH/AvgD/AvgA/MaxH/MaxD/MaxA).
    - Referee featury přidáme jinde dle prefixu.
    """
    cols = df.columns.tolist()

    # 1) core rolling/diff featury pro prefix
    core = [
        c for c in cols
        if (
            (f"_{prefix}_" in c.lower() or c.lower().startswith(f"{prefix}_") or c.lower().endswith(f"_{prefix}")
             or c.lower().startswith(f"home_{prefix}_") or c.lower().startswith(f"away_{prefix}_")
             or f"diff_{prefix}_" in c.lower()
             )
            and ("roll" in c.lower() or "diff_" in c.lower() or "_diff" in c.lower())
        )
    ]

    # 2) globální featury (pokud existují) – safe pro všechny trhy
    global_candidates = [
        # ELO
        "elo_home", "elo_away", "elo_diff",
        # Days rest
        "home_days_rest", "away_days_rest", "days_rest_diff", "is_midweek",
        # Coach
        "HomeCoachTenureDays", "AwayCoachTenureDays", "CoachTenureDiff",
        "NewHomeCoach_30", "NewAwayCoach_30",
        "HomeCoachTenure_log1p", "AwayCoachTenure_log1p",
        # Tabulka
        "home_table_pos", "away_table_pos", "table_pos_diff",
        "home_table_points", "away_table_points", "table_points_diff",
        # Forma (rolling win rate – varianta A: všechny zápasy)
        "home_points_roll3", "away_points_roll3", "diff_points_roll3",
        "home_points_roll5", "away_points_roll5", "diff_points_roll5",
        "home_points_roll10", "away_points_roll10", "diff_points_roll10",
        # Forma (rolling win rate – varianta B: home/away split)
        "home_form_home_roll5", "away_form_away_roll5", "diff_form_ha_roll5",
        "home_form_home_roll10", "away_form_away_roll10", "diff_form_ha_roll10",
    ]
    global_feats = [c for c in global_candidates if c in cols]

    # 3) explicitně vyhazujeme bookmaker odds (freeze)
    bookmaker_odds = {"AvgH", "AvgD", "AvgA", "MaxH", "MaxD", "MaxA"}
    feats = [c for c in (core + global_feats) if c not in bookmaker_odds]

    # deduplikace při zachování pořadí
    seen = set()
    ordered = []
    for c in feats:
        if c not in seen:
            ordered.append(c)
            seen.add(c)

    return ordered


def add_referee_features(df: pd.DataFrame, prefix: str, current_feats: List[str]) -> List[str]:
    """
    Referee featury pouze pro:
    - fouls  -> ref_fouls_avg_last20 + ref_matches_count_last20 + ref_unknown
    - yellow/cards -> ref_cards_avg_last20 + ref_matches_count_last20 + ref_unknown
    """
    cols = set(df.columns.tolist())
    feats = list(current_feats)

    shared = ["ref_matches_count_last20", "ref_unknown"]

    if prefix == "fouls":
        ref_cols = ["ref_fouls_avg_last20"] + shared
        feats += [c for c in ref_cols if c in cols]

    if prefix in ("yellow", "cards"):
        ref_cols = ["ref_cards_avg_last20"] + shared
        feats += [c for c in ref_cols if c in cols]

    seen = set()
    out = []
    for c in feats:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out


def prepare_xy(
    df: pd.DataFrame,
    target: str,
    feature_cols: List[str],
    standardize: bool,
    scaler: Optional[StandardScaler],
    fit_scaler: bool
) -> Tuple[np.ndarray, np.ndarray, Optional[StandardScaler]]:
    if target not in df.columns:
        raise KeyError(f"Target '{target}' není ve sloupcích datasetu.")

    X_df = df[feature_cols].copy()
    X_df = safe_median_impute(X_df)

    if standardize:
        if scaler is None:
            scaler = StandardScaler()
        if fit_scaler:
            X_scaled = scaler.fit_transform(X_df.values)
        else:
            X_scaled = scaler.transform(X_df.values)
        X = pd.DataFrame(X_scaled, columns=feature_cols, index=X_df.index)
    else:
        X = X_df

    X = sm.add_constant(X, has_constant="add")
    y = df[target].to_numpy(dtype=float)
    return X.values, y, scaler


def clip_target(y: np.ndarray, clip_value: Optional[float]) -> np.ndarray:
    if clip_value is None:
        return y
    return np.clip(y, 0, clip_value)


def eval_basic(y_true: np.ndarray, y_pred_mean: np.ndarray) -> dict:
    mae = float(np.mean(np.abs(y_true - y_pred_mean)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred_mean) ** 2)))
    return {"mae": mae, "rmse": rmse}


def _hash_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in s)


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--family", default="poisson", choices=["poisson", "negbin"])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--clip", type=float, default=None)
    ap.add_argument("--standardize", action="store_true")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--save_artifacts", action="store_true")
    ap.add_argument("--version", default="v1_model_freeze")
    ap.add_argument("--artifact_root", default=None)
    ap.add_argument("--tag", default=None)

    args = ap.parse_args()

    TARGET = args.target
    PREFIX = args.prefix.lower().strip()

    root = project_root()
    data_dir = Path(args.data_dir) if args.data_dir else (root / "data" / "features")

    train_path = data_dir / "train_features.csv"
    val_path   = data_dir / "val_features.csv"
    test_path  = data_dir / "test_features.csv"

    train = load_split(train_path)
    val   = load_split(val_path)
    test  = load_split(test_path)

    feats = pick_feature_columns(train, PREFIX)
    feats = add_referee_features(train, PREFIX, feats)

    common  = [c for c in feats if c in val.columns and c in test.columns]
    dropped = [c for c in feats if c not in common]
    feats   = common

    if len(feats) == 0:
        raise RuntimeError(
            f"Nenašel jsem žádné featury pro prefix='{PREFIX}'. "
            f"Zkontroluj názvy sloupců v train_features.csv nebo rozšiř pick_feature_columns()."
        )

    print(f"\nTARGET: {TARGET}")
    print(f"PREFIX: {PREFIX}")
    print(f"FAMILY: {args.family}")
    print(f"Standardize: {args.standardize}")
    if dropped:
        print(f"Pozn.: Vyřazeno {len(dropped)} featur: {dropped[:15]}{'...' if len(dropped)>15 else ''}")

    print(f"\nPoužité featury ({len(feats)}):")
    for c in feats:
        print(f"  {c}")

    scaler: Optional[StandardScaler] = None
    X_train, y_train, scaler = prepare_xy(train, TARGET, feats, args.standardize, scaler, fit_scaler=True)
    X_val,   y_val,   _      = prepare_xy(val,   TARGET, feats, args.standardize, scaler, fit_scaler=False)
    X_test,  y_test,  _      = prepare_xy(test,  TARGET, feats, args.standardize, scaler, fit_scaler=False)

    y_train_c = clip_target(y_train, args.clip)
    y_val_c   = clip_target(y_val,   args.clip)
    y_test_c  = clip_target(y_test,  args.clip)

    if args.family == "poisson":
        fam = sm.families.Poisson()
    else:
        fam = sm.families.NegativeBinomial(alpha=float(args.alpha))

    model = sm.GLM(y_train_c, X_train, family=fam)
    res   = model.fit()

    pred_train = res.predict(X_train)
    pred_val   = res.predict(X_val)
    pred_test  = res.predict(X_test)

    m_train = eval_basic(y_train_c, pred_train)
    m_val   = eval_basic(y_val_c,   pred_val)
    m_test  = eval_basic(y_test_c,  pred_test)

    print("\n--- Výsledky (MAE/RMSE na mean predikci) ---")
    print(f"TRAIN: MAE={m_train['mae']:.4f} | RMSE={m_train['rmse']:.4f}")
    print(f"VAL:   MAE={m_val['mae']:.4f} | RMSE={m_val['rmse']:.4f}")
    print(f"TEST:  MAE={m_test['mae']:.4f} | RMSE={m_test['rmse']:.4f}")

    print("\n--- GLM summary (zkráceně) ---")
    print(f"nobs(train): {int(res.nobs)}")
    try:
        print(f"AIC: {float(res.aic):.3f}")
    except Exception:
        pass
    try:
        print(f"Deviance: {float(res.deviance):.3f}")
    except Exception:
        pass

    meta = {
        "target": TARGET,
        "prefix": PREFIX,
        "family": args.family,
        "alpha": args.alpha if args.family == "negbin" else None,
        "clip": args.clip,
        "standardize": bool(args.standardize),
        "n_features": len(feats),
        "features": feats,
        "metrics": {"train": m_train, "val": m_val, "test": m_test},
    }
    print("\n--- META (pro kontrolu) ---")
    print(json.dumps(meta, ensure_ascii=False, indent=2))

    if args.save_artifacts:
        artifact_root = Path(args.artifact_root) if args.artifact_root else (root / "artifacts")
        run_dir = artifact_root / args.version / _safe_name(f"{PREFIX}__{TARGET}")
        run_dir.mkdir(parents=True, exist_ok=True)

        model_path = run_dir / "model.sm"
        res.save(str(model_path))

        if args.standardize:
            joblib.dump(scaler, run_dir / "scaler.joblib")

        with open(run_dir / "features.json", "w", encoding="utf-8") as f:
            json.dump(feats, f, ensure_ascii=False, indent=2)

        stamp = dt.datetime.now().isoformat(timespec="seconds")
        meta["saved_at"] = stamp
        meta["tag"] = args.tag
        meta["data_files"] = {
            "train": str(train_path),
            "val":   str(val_path),
            "test":  str(test_path),
        }
        try:
            meta["data_hash_train"] = _hash_file(train_path)
            meta["data_hash_val"]   = _hash_file(val_path)
            meta["data_hash_test"]  = _hash_file(test_path)
        except Exception as e:
            meta["data_hash_error"] = str(e)

        with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print(f"\n[ARTIFACTS SAVED] {run_dir}")
        print(f"  - model.sm")
        if args.standardize:
            print("  - scaler.joblib")
        print("  - features.json")
        print("  - meta.json")


if __name__ == "__main__":
    main()