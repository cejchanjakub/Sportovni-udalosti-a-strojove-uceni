# src/inference/registry.py
from __future__ import annotations

from typing import Dict

from src.inference.services.one_x_two_service import OneXTwoService
from src.inference.services.goals_total_service import GoalsTotalService
from src.inference.services.goals_home_service import GoalsHomeService
from src.inference.services.goals_away_service import GoalsAwayService
from src.inference.services.corners_total_service import CornersTotalService
from src.inference.services.corners_home_service import CornersHomeService
from src.inference.services.corners_away_service import CornersAwayService
from src.inference.services.fouls_total_service import FoulsTotalService
from src.inference.services.fouls_home_service import FoulsHomeService
from src.inference.services.fouls_away_service import FoulsAwayService
from src.inference.services.cards_total_service import CardsTotalService
from src.inference.services.cards_home_service import CardsHomeService
from src.inference.services.cards_away_service import CardsAwayService
from src.inference.services.shots_total_service import ShotsTotalService
from src.inference.services.shots_home_service import ShotsHomeService
from src.inference.services.shots_away_service import ShotsAwayService


class MarketRegistry:
    """
    Registry pro mapování market name -> service instance.
    """

    def __init__(self) -> None:
        self._services: Dict[str, object] = {}

    def register(self, market: str, service: object) -> None:
        self._services[market] = service

    def register_defaults(self) -> None:
        self.register("1x2",           OneXTwoService())
        self.register("goals_total",   GoalsTotalService())
        self.register("goals_home",    GoalsHomeService())
        self.register("goals_away",    GoalsAwayService())
        self.register("corners_total", CornersTotalService())
        self.register("corners_home",  CornersHomeService())
        self.register("corners_away",  CornersAwayService())
        self.register("fouls_total",   FoulsTotalService())
        self.register("fouls_home",    FoulsHomeService())
        self.register("fouls_away",    FoulsAwayService())
        self.register("cards_total",   CardsTotalService())
        self.register("cards_home",    CardsHomeService())
        self.register("cards_away",    CardsAwayService())
        self.register("shots_total",   ShotsTotalService())
        self.register("shots_home",    ShotsHomeService())
        self.register("shots_away",    ShotsAwayService())

    def get(self, market: str) -> object:
        if market not in self._services:
            raise KeyError(f"Unknown market: {market}")
        return self._services[market]

    def list_markets(self) -> list[str]:
        return sorted(self._services.keys())