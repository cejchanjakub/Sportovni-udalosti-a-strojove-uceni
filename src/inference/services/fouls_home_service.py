# src/inference/services/fouls_home_service.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict
import math

import pandas as pd
import statsmodels.api as sm

from src.inference.Team_mapper import map_team
from src.inference.predict_1x2_from_live_features import (
    LIVE_FEATURES,
    _pick_row,
    _feature_baseline_means,
)
from src.inference.model_loader import load_fouls_home_model, load_fouls_away_model
from src.line_generator import (
    generate_ou_lines_around_mean,
    generate_ou_lines,
    pick_main_line,
    to_dicts,
)


@dataclass
class FoulsHomeResult:
    mapped: Dict[str, str]
    swapped: bool
    mean: float  # očekávaný počet faulů REQUEST-HOME
    lines: Dict[str, Any]
    matched_row: Dict[str, Any]


class FoulsHomeService:
    """
    Service pro fauly domácího týmu (request-home).

    Logika swapped:
    - pokud swapped=True, request-home je v row jako AWAY -> použij away model
    - jinak použij home model
    """

    _model_home = None
    _scaler_home = None
    _feat_home = None

    _model_away = None
    _scaler_away = None
    _feat_away = None

    _OFFSETS = [-2.5, -1.5, 1.5, 2.5]
    _MIN_LINE = 0.5
    _MAX_LINE = 35.5
    _MAX_LINES = 4

    def __init__(self) -> None:
        if self.__class__._model_home is None:
            m, s, f = load_fouls_home_model()
            self.__class__._model_home = m
            self.__class__._scaler_home = s
            self.__class__._feat_home = f

        if self.__class__._model_away is None:
            m, s, f = load_fouls_away_model()
            self.__class__._model_away = m
            self.__class__._scaler_away = s
            self.__class__._feat_away = f

    def predict_from_match(
        self,
        utc_date: str,
        home_team: str,
        away_team: str,
        *,
        margin: float = 0.0,
    ) -> FoulsHomeResult:
        home_std = map_team(home_team)
        away_std = map_team(away_team)

        df_live = pd.read_csv(LIVE_FEATURES)
        row, swapped = _pick_row(df_live, home_std, away_std, utc_date)

        # swapped => request-home je v row jako AWAY
        if swapped:
            model = self.__class__._model_away
            scaler = self.__class__._scaler_away
            feat_order = self.__class__._feat_away
        else:
            model = self.__class__._model_home
            scaler = self.__class__._scaler_home
            feat_order = self.__class__._feat_home

        X = pd.DataFrame([row.to_dict()])
        X = X[feat_order]

        means = _feature_baseline_means(feat_order)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(means)

        Xs = scaler.transform(X)
        Xs_const = sm.add_constant(Xs, has_constant="add")

        mean_pred = float(model.predict(Xs_const)[0])

        if not math.isfinite(mean_pred) or mean_pred < 0:
            raise ValueError(f"FoulsHomeService: invalid mean_pred={mean_pred}")

        lines_rows = generate_ou_lines_around_mean(
            mean_pred,
            dist="poisson",
            offsets=self._OFFSETS,
            min_line=self._MIN_LINE,
            max_line=self._MAX_LINE,
            max_lines=self._MAX_LINES,
            margin=margin,
        )
        if not lines_rows:
            lines_rows = generate_ou_lines(
                mean_pred,
                dist="poisson",
                margin=margin,
            )

        main = pick_main_line(lines_rows, target_p_over=0.5)

        return FoulsHomeResult(
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