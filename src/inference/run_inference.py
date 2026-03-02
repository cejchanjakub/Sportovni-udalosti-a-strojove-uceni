from src.inference.providers.football_data_provider import FootballDataProvider
from src.inference.Team_mapper import map_team

from src.inference.Predict_1X2 import predict_1x2

# TODO: zatím dummy featury (jen test)
dummy_features = {
    "AvgH": 1.5,
    "AvgD": 3.5,
    "AvgA": 2.8,
    # sem musí přijít všechny featury podle features.json
}

result = predict_1x2(dummy_features)
print(result)


def main():
    provider = FootballDataProvider()
    matches = provider.get_upcoming_matches(days_ahead=7)

    match = matches[0]
    match_id = match["match_id"]

    # zkus dočíst rozhodčího z detailu
    ref = provider.get_referee_for_match(match_id)
    match["referee"] = ref

    print("RAW API:")
    print(match)

    home_std = map_team(match["home_team"])
    away_std = map_team(match["away_team"])

    print("\nPO MAPOVÁNÍ:")
    print("Home:", home_std)
    print("Away:", away_std)
    print("Ref:", ref)

if __name__ == "__main__":
    main()
