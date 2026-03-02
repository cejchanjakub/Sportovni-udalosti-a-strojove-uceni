from __future__ import annotations

from typing import Dict

from src.inference.services.one_x_two_service import OneXTwoService


class MarketRegistry:
    """
    Registry pro mapování market name -> service instance.
    """

    def __init__(self) -> None:
        self._services: Dict[str, object] = {}

    def register(self, market: str, service: object) -> None:
        self._services[market] = service

    def register_defaults(self) -> None:
        # fallback pokud by se registry použil samostatně
        self.register("1x2", OneXTwoService())

    def get(self, market: str) -> object:
        if market not in self._services:
            raise KeyError(f"Unknown market: {market}")
        return self._services[market]

    def list_markets(self) -> list[str]:
        return sorted(self._services.keys())