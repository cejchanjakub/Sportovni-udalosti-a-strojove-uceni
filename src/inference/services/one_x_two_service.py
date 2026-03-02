from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd

from src.inference.Team_mapper import map_team
from src.inference.predict_1x2_from_live_features import (
    LIVE_FEATURES,
    _pick_row,
    _load_artifacts,
    _feature_baseline_means,
)
from pathlib import Path

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

    - přijme match input (utc_date, home_team, away_team)
    - namapuje názvy týmů
    - najde odpovídající řádek v LIVE_FEATURES
    - připraví feature vektor v přesném pořadí modelu
    - doplní NaN baseline průměry
    - provede predikci (model + scaler)
    - ošetří swapped (když je v live_features řádek opačně)
    """

    # Cache artefaktů (načtou se jednou na proces)
    _model = None
    _scaler = None
    _feat_order = None

    def __init__(self) -> None:
        # načti artefakty jen jednou
        if self.__class__._model is None:
            model, scaler, feat_order = _load_artifacts()
            self.__class__._model = model
            self.__class__._scaler = scaler
            self.__class__._feat_order = feat_order

    def predict_from_match(self, utc_date: str, home_team: str, away_team: str) -> OneXTwoResult:
        # 1) map team names do standardu
        home_std = map_team(home_team)
        away_std = map_team(away_team)

        # 2) najdi řádek v live features
        df_live = pd.read_csv(LIVE_FEATURES)
        row, swapped = _pick_row(df_live, home_std, away_std, utc_date)

        # 3) připrav X v pořadí featur
        X = pd.DataFrame([row.to_dict()])
        X = X[self.__class__._feat_order]

        # 4) doplň NaN baseline průměry
        means = _feature_baseline_means(self.__class__._feat_order)
        X = X.apply(pd.to_numeric, errors="coerce").fillna(means)

        # 5) predikce
        Xs = self.__class__._scaler.transform(X)
        probs = self.__class__._model.predict_proba(Xs)[0]
        p_home, p_draw, p_away = map(float, probs)

        # 6) ošetři swapped (když je řádek v live_features opačně)
        if swapped:
            p_home, p_away = p_away, p_home

        odds_home = (1 / p_home) if p_home > 0 else None
        odds_draw = (1 / p_draw) if p_draw > 0 else None
        odds_away = (1 / p_away) if p_away > 0 else None

        live_path = Path(LIVE_FEATURES)
        if not live_path.exists():
            raise FileNotFoundError(f"LIVE_FEATURES not found: {live_path}")

        df_live = pd.read_csv(live_path)

        row, swapped = _pick_row(df_live, home_std, away_std, utc_date)
        if row is None or len(row) == 0:
            raise LookupError(f"No matching row in live_features for {home_std} vs {away_std} at {utc_date}")

        return OneXTwoResult(
            mapped={"home": home_std, "away": away_std},
            swapped=swapped,
            p_home=p_home,
            p_draw=p_draw,
            p_away=p_away,
            odds_home=odds_home,
            odds_draw=odds_draw,
            odds_away=odds_away,
            matched_row={
                "HomeTeam": row.get("HomeTeam"),
                "AwayTeam": row.get("AwayTeam"),
                "Date": row.get("Date"),
            },
        )