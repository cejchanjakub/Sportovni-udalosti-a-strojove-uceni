# src/inference/services/one_x_two_service.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.inference.Team_mapper import map_team
from src.inference.predict_1x2_from_live_features import (
    LIVE_FEATURES,
    _pick_row,
    _load_artifacts,
    _feature_baseline_means,
)


@dataclass
class OneXTwoResult:
    mapped: Dict[str, str]
    swapped: bool
    p_home: float
    p_draw: float
    p_away: float
    odds_home: float | None
    odds_draw: float | None
    odds_away: float | None
    matched_row: Dict[str, Any]


class OneXTwoService:
    """
    Service vrstva pro 1X2 inference.
    Používá sklearn CalibratedClassifierCV + scaler (trénován se standardizací).
    """

    _model = None
    _scaler = None
    _feat_order = None

    def __init__(self) -> None:
        if self.__class__._model is None:
            model, scaler, feat_order = _load_artifacts()
            self.__class__._model = model
            self.__class__._scaler = scaler
            self.__class__._feat_order = feat_order

    def predict_from_match(self, utc_date: str, home_team: str, away_team: str) -> OneXTwoResult:
        home_std = map_team(home_team)
        away_std = map_team(away_team)

        df_live = pd.read_csv(LIVE_FEATURES)
        row, swapped = _pick_row(df_live, home_std, away_std, utc_date)

        X = pd.DataFrame([row.to_dict()])
        X = X[self.__class__._feat_order]

        means = _feature_baseline_means(self.__class__._feat_order)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(means)
        X = X.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        Xs = self.__class__._scaler.transform(X)
        probs = self.__class__._model.predict_proba(Xs)[0]
        p_home, p_draw, p_away = map(float, probs)

        if swapped:
            p_home, p_away = p_away, p_home

        return OneXTwoResult(
            mapped={"home": home_std, "away": away_std},
            swapped=swapped,
            p_home=p_home,
            p_draw=p_draw,
            p_away=p_away,
            odds_home=(1 / p_home) if p_home > 0 else None,
            odds_draw=(1 / p_draw) if p_draw > 0 else None,
            odds_away=(1 / p_away) if p_away > 0 else None,
            matched_row={
                "HomeTeam": row.get("HomeTeam"),
                "AwayTeam": row.get("AwayTeam"),
                "Date": row.get("Date"),
            },
        )