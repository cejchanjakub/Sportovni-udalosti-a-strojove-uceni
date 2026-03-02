# src/inference/services/corners_total_service.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd
import statsmodels.api as sm

from src.inference.Team_mapper import map_team
from src.inference.predict_1x2_from_live_features import (
    LIVE_FEATURES,
    _pick_row,
    _feature_baseline_means,
)
from src.inference.model_loader import load_corners_total_model
from src.line_generator import generate_ou_lines_around_mean, pick_main_line, to_dicts


@dataclass
class CornersTotalResult:
    mapped: Dict[str, str]
    swapped: bool
    mean: float  # očekávaný počet rohů celkem (lambda)
    lines: Dict[str, Any]  # O/U linky
    matched_row: Dict[str, Any]


class CornersTotalService:
    """
    Service pro total_corners – Poisson GLM.

    - načte GLM artefakty (corners__total_corners)
    - najde odpovídající live row
    - připraví feature vektor
    - doplní NaN baseline průměry
    - standardizuje (scaler)
    - přidá const (intercept) pro statsmodels
    - vrátí mean (lambda) + dynamické O/U linky kolem mean (round(mean) ± offsets)
    """

    _model = None
    _scaler = None
    _feat_order = None

    # konfigurace dynamických linek pro corners_total
    _OFFSETS = [-1.5, -0.5, 0.5, 1.5]
    _MIN_LINE = 0.5
    _MAX_LINE = 19.5
    _MAX_LINES = 4

    def __init__(self) -> None:
        if self.__class__._model is None:
            model, scaler, feat_order = load_corners_total_model()
            self.__class__._model = model
            self.__class__._scaler = scaler
            self.__class__._feat_order = feat_order

    def predict_from_match(
        self,
        utc_date: str,
        home_team: str,
        away_team: str,
        *,
        margin: float = 0.0,
    ) -> CornersTotalResult:
        home_std = map_team(home_team)
        away_std = map_team(away_team)

        df_live = pd.read_csv(LIVE_FEATURES)
        row, swapped = _pick_row(df_live, home_std, away_std, utc_date)

        X = pd.DataFrame([row.to_dict()])
        X = X[self.__class__._feat_order]

        means = _feature_baseline_means(self.__class__._feat_order)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(means)

        Xs = self.__class__._scaler.transform(X)
        Xs_const = sm.add_constant(Xs, has_constant="add")

        mean_pred = float(self.__class__._model.predict(Xs_const)[0])

        lines_rows = generate_ou_lines_around_mean(
            mean_pred,
            dist="poisson",
            offsets=self._OFFSETS,
            min_line=self._MIN_LINE,
            max_line=self._MAX_LINE,
            max_lines=self._MAX_LINES,
            margin=margin,
        )
        main = pick_main_line(lines_rows, target_p_over=0.5)

        return CornersTotalResult(
            mapped={"home": home_std, "away": away_std},
            swapped=swapped,
            mean=mean_pred,
            lines={
                "main_line": float(main.line),
                "ou": to_dicts(lines_rows),
            },
            matched_row={
                "HomeTeam": row.get("HomeTeam"),
                "AwayTeam": row.get("AwayTeam"),
                "Date": row.get("Date"),
            },
        )