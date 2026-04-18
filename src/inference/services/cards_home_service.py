# src/inference/services/cards_home_service.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm
from src.inference.Team_mapper import map_team
from src.inference.predict_1x2_from_live_features import (
    LIVE_FEATURES, _pick_row, _feature_baseline_means,
)
from src.inference.model_loader import load_cards_home_model
from src.line_generator import generate_ou_lines_around_mean, pick_main_line, to_dicts


@dataclass
class CardsHomeResult:
    mapped: Dict[str, str]
    swapped: bool
    mean: float
    lines: Dict[str, Any]
    matched_row: Dict[str, Any]
    referee: Optional[str] = None


class CardsHomeService:
    _model = None
    _feat_order = None
    _alpha = None
    _OFFSETS = [-1.5, -0.5, 0.5, 1.5]
    _MIN_LINE = 0.5
    _MAX_LINE = 6.5
    _MAX_LINES = 4
    _DIST = "negbin"

    def __init__(self) -> None:
        if self.__class__._model is None:
            model, feat_order, alpha = load_cards_home_model()
            self.__class__._model = model
            self.__class__._feat_order = feat_order
            self.__class__._alpha = alpha

    def predict_from_match(
        self,
        utc_date: str,
        home_team: str,
        away_team: str,
        *,
        referee: Optional[str] = None,
        margin: float = 0.0,
    ) -> CardsHomeResult:
        home_std = map_team(home_team)
        away_std = map_team(away_team)
        df_live = pd.read_csv(LIVE_FEATURES)
        row, swapped = _pick_row(df_live, home_std, away_std, utc_date)
        row = row.copy()

        # Přepis referee featur pokud je znám rozhodčí
        if referee:
            row = _inject_referee_features(row, referee)

        X = pd.DataFrame([row.to_dict()])[self.__class__._feat_order]
        means = _feature_baseline_means(self.__class__._feat_order)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(means).replace([np.inf, -np.inf], 0.0).fillna(0.0)
        X_const = sm.add_constant(X, has_constant="add")
        mean_pred = float(self.__class__._model.predict(X_const)[0])
        alpha_arg = {"alpha": self.__class__._alpha} if self.__class__._alpha is not None else {}
        lines_rows = generate_ou_lines_around_mean(
            mean_pred, dist=self._DIST,
            offsets=self._OFFSETS,
            min_line=self._MIN_LINE,
            max_line=self._MAX_LINE,
            max_lines=self._MAX_LINES,
            margin=margin,
            **alpha_arg
        )
        main = pick_main_line(lines_rows, target_p_over=0.5)
        return CardsHomeResult(
            mapped={"home": home_std, "away": away_std},
            swapped=swapped,
            mean=mean_pred,
            lines={"main_line": float(main.line), "ou": to_dicts(lines_rows)},
            matched_row={"HomeTeam": row.get("HomeTeam"), "AwayTeam": row.get("AwayTeam"), "Date": row.get("Date")},
            referee=referee,
        )


# ---------------------------------------------------------------------------
# Pomocná funkce – přepis referee featur v row
# ---------------------------------------------------------------------------

_referee_cache: Dict[str, Dict[str, float]] = {}

def _inject_referee_features(row: pd.Series, referee_name: str) -> pd.Series:
    """
    Načte statistiky rozhodčího z train_features.csv a přepíše
    ref_fouls_avg_last20, ref_cards_avg_last20, ref_matches_count_last20, ref_unknown v row.
    """
    global _referee_cache

    if not _referee_cache:
        _load_referee_cache()

    stats = _referee_cache.get(referee_name)
    if not stats:
        return row  # neznámý rozhodčí → ponech průměr

    row = row.copy()
    for col, val in stats.items():
        if col in row.index:
            row[col] = val

    row["ref_unknown"] = 0
    return row


def _load_referee_cache():
    """Načte průměrné referee statistiky z train_features.csv."""
    global _referee_cache
    from src.inference.predict_1x2_from_live_features import TRAIN_FEATURES

    if not TRAIN_FEATURES.exists():
        return

    df = pd.read_csv(TRAIN_FEATURES)
    ref_stat_cols = [c for c in df.columns if c.startswith("ref_") and c != "ref_unknown"]

    if "Referee" not in df.columns or not ref_stat_cols:
        return

    agg = df[df["Referee"].notna()].groupby("Referee")[ref_stat_cols].mean()
    for ref_name, stats_row in agg.iterrows():
        _referee_cache[ref_name] = stats_row.to_dict()