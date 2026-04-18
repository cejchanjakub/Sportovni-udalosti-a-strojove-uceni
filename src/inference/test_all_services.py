# src/inference/test_all_services.py
"""
Rychlý test všech inference services na prvním nadcházejícím zápase z API.
Spustit: python -m src.inference.test_all_services
"""
from __future__ import annotations

from src.inference.providers.football_data_provider import FootballDataProvider
from src.inference.Team_mapper import map_team

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


SERVICES = {
    "1x2":           OneXTwoService(),
    "goals_total":   GoalsTotalService(),
    "goals_home":    GoalsHomeService(),
    "goals_away":    GoalsAwayService(),
    "corners_total": CornersTotalService(),
    "corners_home":  CornersHomeService(),
    "corners_away":  CornersAwayService(),
    "fouls_total":   FoulsTotalService(),
    "fouls_home":    FoulsHomeService(),
    "fouls_away":    FoulsAwayService(),
    "cards_total":   CardsTotalService(),
    "cards_home":    CardsHomeService(),
    "cards_away":    CardsAwayService(),
    "shots_total":   ShotsTotalService(),
    "shots_home":    ShotsHomeService(),
    "shots_away":    ShotsAwayService(),
}


def main():
    provider = FootballDataProvider()
    fixtures = provider.get_upcoming_matches(days_ahead=14)
    if not fixtures:
        print("Žádné nadcházející zápasy z API.")
        return

    fx = fixtures[0]
    home = fx["home_team"]
    away = fx["away_team"]
    utc  = fx["utc_date"]

    print(f"Testovací zápas: {home} vs {away} | {utc}")
    print(f"Mapped: {map_team(home)} vs {map_team(away)}")
    print("=" * 60)

    passed, failed = [], []

    for market, service in SERVICES.items():
        try:
            if market == "1x2":
                result = service.predict_from_match(utc, home, away)
                print(f"  ✅ {market:15s} | P(H)={result.p_home:.3f} P(D)={result.p_draw:.3f} P(A)={result.p_away:.3f}")
            else:
                result = service.predict_from_match(utc, home, away)
                print(f"  ✅ {market:15s} | mean={result.mean:.2f} | main_line={result.lines['main_line']:.1f}")
            passed.append(market)
        except Exception as e:
            print(f"  ❌ {market:15s} | {type(e).__name__}: {e}")
            failed.append(market)

    print("=" * 60)
    print(f"Výsledek: {len(passed)}/{len(SERVICES)} OK")
    if failed:
        print(f"Selhalo: {failed}")


if __name__ == "__main__":
    main()