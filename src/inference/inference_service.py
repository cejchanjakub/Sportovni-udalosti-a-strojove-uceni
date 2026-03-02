# src/inference/inference_service.py
from __future__ import annotations

import inspect
from typing import Any, Dict, List


class InferenceService:
    """
    Orchestrátor pro multi-market predikce.

    Nové chování:
    - žádný hardcoded whitelist (1x2/goals_total/...)
    - pro každý market si vezme service z registry a zavolá predict_from_match
    - automaticky předá jen ty parametry, které daná metoda přijímá (bezpečné pro budoucí rozšíření match payloadu)
    """

    def __init__(self, registry):
        self._registry = registry

    @staticmethod
    def _call_with_accepted_kwargs(func, kwargs: Dict[str, Any]):
        sig = inspect.signature(func)
        accepted = {}
        for name, param in sig.parameters.items():
            # ignoruj self
            if name == "self":
                continue
            if name in kwargs:
                accepted[name] = kwargs[name]
        return func(**accepted)

    @staticmethod
    def _to_plain(obj: Any) -> Any:
        # service může vracet dict nebo dataclass-like objekt
        if isinstance(obj, dict):
            return obj
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        return obj

    def predict(self, match: Dict[str, Any], markets: List[str]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        for market in markets:
            service = self._registry.get(market)
            if service is None:
                raise KeyError(f"Unknown market: {market}")

            if not hasattr(service, "predict_from_match"):
                raise NotImplementedError(f"Service for market '{market}' has no predict_from_match()")

            res = self._call_with_accepted_kwargs(service.predict_from_match, match)
            out[market] = self._to_plain(res)

        return out