"""Stadium metadata + name normalization, owned by bigleague.

All 30 active parks are bundled inline so we work offline. Dimensions are
LF / CF / RF distances to the wall in feet, used to label the diagrams drawn
by `plots.spray_chart` and `plots.stadium_outline`. The parks' shapes come from
the outline data in `bigleague.stadium_paths`.

`STADIUM_ALIASES` covers franchise renames where Statcast still uses the
old key (e.g. CLE = "indians" even after the 2022 rename to Guardians).
"""

from __future__ import annotations

STADIUM_ALIASES: dict[str, str] = {
    "guardians": "indians",  # franchise rename 2022 — Statcast still uses CLE
}

# stadium_key -> Statcast home_team code. Used to filter Statcast batted
# balls by the home park where they were hit.
STADIUM_TO_TEAM_CODE: dict[str, str] = {
    "angels": "LAA",
    "astros": "HOU",
    "athletics": "ATH",  # Savant's 2025+ code (was OAK; see LEGACY_TEAM_CODES)
    "blue_jays": "TOR",
    "braves": "ATL",
    "brewers": "MIL",
    "cardinals": "STL",
    "cubs": "CHC",
    "diamondbacks": "ARI",
    "dodgers": "LAD",
    "giants": "SF",
    "indians": "CLE",
    "guardians": "CLE",
    "mariners": "SEA",
    "marlins": "MIA",
    "mets": "NYM",
    "nationals": "WSH",
    "orioles": "BAL",
    "padres": "SD",
    "phillies": "PHI",
    "pirates": "PIT",
    "rangers": "TEX",
    "rays": "TB",
    "red_sox": "BOS",
    "reds": "CIN",
    "rockies": "COL",
    "royals": "KC",
    "tigers": "DET",
    "twins": "MIN",
    "white_sox": "CWS",
    "yankees": "NYY",
}

# Codes that changed over time: current code -> codes older Statcast frames
# carry in home_team (OAK→ATH in 2025). Filters that match on a stadium's
# team code should accept any era's code.
LEGACY_TEAM_CODES: dict[str, tuple[str, ...]] = {
    "ATH": ("OAK",),
}

# Bundled stadium metadata — name, location, and outfield wall distances
# (LF/CF/RF) in feet. Numbers are the published park-factor distances;
# CF-Triangle quirks at Fenway are absorbed into the CF figure (420ft).
STADIUM_DATA: dict[str, dict[str, str | int]] = {
    "angels": {"name": "Angel Stadium", "location": "Anaheim, CA", "lf": 330, "cf": 396, "rf": 330},
    "astros": {"name": "Daikin Park", "location": "Houston, TX", "lf": 315, "cf": 409, "rf": 326},
    "athletics": {
        "name": "Oakland Coliseum",
        "location": "Oakland, CA",
        "lf": 330,
        "cf": 400,
        "rf": 330,
    },
    "blue_jays": {
        "name": "Rogers Centre",
        "location": "Toronto, ON",
        "lf": 328,
        "cf": 400,
        "rf": 328,
    },
    "braves": {
        "name": "Truist Park",
        "location": "Cumberland, GA",
        "lf": 335,
        "cf": 400,
        "rf": 325,
    },
    "brewers": {
        "name": "American Family Field",
        "location": "Milwaukee, WI",
        "lf": 344,
        "cf": 400,
        "rf": 345,
    },
    "cardinals": {
        "name": "Busch Stadium",
        "location": "St. Louis, MO",
        "lf": 336,
        "cf": 400,
        "rf": 335,
    },
    "cubs": {"name": "Wrigley Field", "location": "Chicago, IL", "lf": 355, "cf": 400, "rf": 353},
    "diamondbacks": {
        "name": "Chase Field",
        "location": "Phoenix, AZ",
        "lf": 330,
        "cf": 407,
        "rf": 335,
    },
    "dodgers": {
        "name": "Dodger Stadium",
        "location": "Los Angeles, CA",
        "lf": 330,
        "cf": 395,
        "rf": 330,
    },
    "giants": {
        "name": "Oracle Park",
        "location": "San Francisco, CA",
        "lf": 339,
        "cf": 399,
        "rf": 309,
    },
    "indians": {
        "name": "Progressive Field",
        "location": "Cleveland, OH",
        "lf": 325,
        "cf": 410,
        "rf": 325,
    },
    "mariners": {
        "name": "T-Mobile Park",
        "location": "Seattle, WA",
        "lf": 331,
        "cf": 401,
        "rf": 326,
    },
    "marlins": {"name": "loanDepot Park", "location": "Miami, FL", "lf": 344, "cf": 407, "rf": 335},
    "mets": {"name": "Citi Field", "location": "Queens, NY", "lf": 335, "cf": 408, "rf": 330},
    "nationals": {
        "name": "Nationals Park",
        "location": "Washington, DC",
        "lf": 336,
        "cf": 402,
        "rf": 335,
    },
    "orioles": {
        "name": "Oriole Park at Camden Yards",
        "location": "Baltimore, MD",
        "lf": 333,
        "cf": 410,
        "rf": 318,
    },
    "padres": {"name": "Petco Park", "location": "San Diego, CA", "lf": 336, "cf": 396, "rf": 322},
    "phillies": {
        "name": "Citizens Bank Park",
        "location": "Philadelphia, PA",
        "lf": 329,
        "cf": 401,
        "rf": 330,
    },
    "pirates": {"name": "PNC Park", "location": "Pittsburgh, PA", "lf": 325, "cf": 399, "rf": 320},
    "rangers": {
        "name": "Globe Life Field",
        "location": "Arlington, TX",
        "lf": 329,
        "cf": 407,
        "rf": 326,
    },
    "rays": {
        "name": "Tropicana Field",
        "location": "St. Petersburg, FL",
        "lf": 315,
        "cf": 404,
        "rf": 322,
    },
    "red_sox": {"name": "Fenway Park", "location": "Boston, MA", "lf": 310, "cf": 420, "rf": 302},
    "reds": {
        "name": "Great American Ball Park",
        "location": "Cincinnati, OH",
        "lf": 328,
        "cf": 404,
        "rf": 325,
    },
    "rockies": {"name": "Coors Field", "location": "Denver, CO", "lf": 347, "cf": 415, "rf": 350},
    "royals": {
        "name": "Kauffman Stadium",
        "location": "Kansas City, MO",
        "lf": 330,
        "cf": 410,
        "rf": 330,
    },
    "tigers": {"name": "Comerica Park", "location": "Detroit, MI", "lf": 345, "cf": 420, "rf": 330},
    "twins": {
        "name": "Target Field",
        "location": "Minneapolis, MN",
        "lf": 339,
        "cf": 404,
        "rf": 328,
    },
    "white_sox": {"name": "Rate Field", "location": "Chicago, IL", "lf": 330, "cf": 400, "rf": 335},
    "yankees": {"name": "Yankee Stadium", "location": "Bronx, NY", "lf": 318, "cf": 408, "rf": 314},
    "generic": {"name": "Generic Ballpark", "location": "—", "lf": 330, "cf": 400, "rf": 330},
}

_KNOWN: frozenset[str] = frozenset(STADIUM_DATA.keys())


def normalize(name: str) -> str:
    """Map a user-supplied team name to a stadium key.

    Steps:
      1. Lowercase, replace spaces/hyphens with underscores.
      2. Apply franchise rename aliases (guardians -> indians).
      3. Fall back to "generic" if still unknown.
    """
    key = name.lower().replace(" ", "_").replace("-", "_")
    key = STADIUM_ALIASES.get(key, key)
    return key if key in _KNOWN else "generic"


def team_code(stadium_key: str) -> str | None:
    """Return the Statcast `home_team` code for a stadium key, if any."""
    return STADIUM_TO_TEAM_CODE.get(stadium_key)


def metadata(stadium_key: str) -> dict[str, str | int]:
    """Return the full metadata dict for a stadium key.

    Raises ValueError for unknown keys (use `normalize()` first if the
    input is user-supplied).
    """
    if stadium_key not in STADIUM_DATA:
        raise ValueError(f"Unknown stadium key: {stadium_key!r}")
    return STADIUM_DATA[stadium_key]


def known_stadiums() -> frozenset[str]:
    return _KNOWN
