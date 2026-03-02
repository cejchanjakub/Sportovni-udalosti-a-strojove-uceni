import os
from datetime import datetime, timedelta, date
import requests

BASE_URL = "https://api.football-data.org/v4"
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")


class FootballDataProvider:

    def __init__(self):
        if API_KEY is None:
            raise ValueError("FOOTBALL_DATA_API_KEY není nastavený v prostředí.")

        self.headers = {"X-Auth-Token": API_KEY}

    import requests
    from datetime import date, timedelta

    # ... (zbytek souboru nech beze změny)

    def get_upcoming_matches(self, days_ahead=14):
        """
        Vrátí nadcházející zápasy Premier League v intervalu [dnes, dnes+days_ahead].
        Používá endpoint soutěže, aby se vyhnul 400 chybám na /matches.
        """
        today = date.today()
        to_day = today + timedelta(days=days_ahead)

        url = f"{BASE_URL}/competitions/PL/matches"
        params = {
            "dateFrom": today.isoformat(),
            "dateTo": to_day.isoformat(),
        }

        response = requests.get(url, headers=self.headers, params=params)

        # Debug při chybě: vypiš server message
        if response.status_code >= 400:
            print("[FootballDataProvider] ERROR", response.status_code)
            try:
                print(response.json())
            except Exception:
                print(response.text)
            response.raise_for_status()

        data = response.json()

        matches = data.get("matches", [])
        print(
            f"[FootballDataProvider] competitions/PL/matches returned {len(matches)} matches for {params['dateFrom']}..{params['dateTo']}")

        fixtures = []
        for match in matches:
            # některé zápasy můžou být FINISHED / POSTPONED – my chceme jen budoucí
            status = match.get("status")
            if status not in ("SCHEDULED", "TIMED"):
                continue

            fixtures.append({
                "match_id": match["id"],
                "utc_date": match["utcDate"],
                "home_team": match["homeTeam"]["name"],
                "away_team": match["awayTeam"]["name"],
                "venue": match.get("venue"),
                "referee": self._extract_referee(match)
            })

        return fixtures

    def _extract_referee(self, match):
        referees = match.get("referees", [])
        for ref in referees:
            if ref.get("type") == "REFEREE":
                return ref.get("name")
        return None

    def get_match_detail(self, match_id: int):
        url = f"{BASE_URL}/matches/{match_id}"
        r = requests.get(url, headers=self.headers)
        r.raise_for_status()
        return r.json()

    def get_referee_for_match(self, match_id: int):
        detail = self.get_match_detail(match_id)
        match = detail.get("match", {})
        referees = match.get("referees", []) or []
        for ref in referees:
            # football-data někdy používá "type": "REFEREE"
            if ref.get("type") == "REFEREE":
                return ref.get("name")
        return None

