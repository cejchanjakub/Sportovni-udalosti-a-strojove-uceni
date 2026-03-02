import re

ALIASES = {
    # Wolves
    "wolverhampton wanderers": "Wolves",
    "wolverhampton wanderers fc": "Wolves",
    "wolves": "Wolves",

    # Leeds
    "leeds united": "Leeds",
    "leeds united fc": "Leeds",
    "leeds": "Leeds",

    # Standardní PL zkratky / běžné zápisy
    "manchester united": "Man United",
    "manchester united fc": "Man United",
    "man united": "Man United",

    "manchester city": "Man City",
    "manchester city fc": "Man City",
    "man city": "Man City",

    "tottenham hotspur": "Tottenham",
    "tottenham hotspur fc": "Tottenham",
    "spurs": "Tottenham",
    "tottenham": "Tottenham",

    "nottingham forest": "Nott'm Forest",
    "nottingham forest fc": "Nott'm Forest",
    "nottingham": "Nott'm Forest",

    "west ham united": "West Ham",
    "west ham united fc": "West Ham",
    "west ham": "West Ham",

    "brighton and hove albion": "Brighton",
    "brighton and hove albion fc": "Brighton",
    "brighton": "Brighton",

    "newcastle united": "Newcastle",
    "newcastle united fc": "Newcastle",
    "newcastle": "Newcastle",

    # ostatní: Arsenal, Chelsea, Liverpool, Everton, Fulham, Brentford, Bournemouth,
    # Crystal Palace, Aston Villa, Burnley, Sunderland ... obvykle sedí už tak
}


def normalize(name: str) -> str:
    s = str(name).lower()
    s = s.replace("&", "and")
    s = re.sub(r"\bfc\b", "", s)
    s = re.sub(r"\bafc\b", "", s)
    s = re.sub(r"[^a-z0-9\s']", "", s)  # necháme apostrof kvůli Nott'm
    s = re.sub(r"\s+", " ", s).strip()
    return s


def map_team(name: str) -> str:
    key = normalize(name)

    # pokus o přímý alias
    if key in ALIASES:
        return ALIASES[key]

    # drobné normalizace (např. "wolverhampton wanderers" bez fc)
    key2 = key.replace(" wanderers", "").strip()
    if key2 in ALIASES:
        return ALIASES[key2]

    # fallback: Title Case (většinou pro "Arsenal FC" -> "Arsenal")
    return " ".join([w.capitalize() for w in key.split()])