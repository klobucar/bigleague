from __future__ import annotations

import pytest
from bigleague import stadiums


def test_all_thirty_teams_normalize() -> None:
    expected = {
        "Yankees",
        "Red Sox",
        "Blue Jays",
        "Orioles",
        "Rays",
        "White Sox",
        "Guardians",
        "Tigers",
        "Royals",
        "Twins",
        "Astros",
        "Angels",
        "Mariners",
        "Rangers",
        "Athletics",
        "Mets",
        "Phillies",
        "Nationals",
        "Marlins",
        "Braves",
        "Cubs",
        "Brewers",
        "Reds",
        "Cardinals",
        "Pirates",
        "Dodgers",
        "Padres",
        "Giants",
        "Diamondbacks",
        "Rockies",
    }
    for name in expected:
        key = stadiums.normalize(name)
        assert key != "generic", f"{name!r} normalized to 'generic'"
        assert key in stadiums.known_stadiums()


def test_guardians_alias_maps_to_indians() -> None:
    assert stadiums.normalize("Guardians") == "indians"


def test_unknown_team_falls_back_to_generic() -> None:
    assert stadiums.normalize("Wizards") == "generic"


def test_team_codes_all_resolve_to_known_stadiums() -> None:
    """Every key in STADIUM_TO_TEAM_CODE must be a known stadium OR an alias
    that normalizes to one."""
    for stadium_key in stadiums.STADIUM_TO_TEAM_CODE:
        assert stadiums.normalize(stadium_key) in stadiums.known_stadiums()


def test_athletics_uses_current_savant_code_with_legacy_alias() -> None:
    """The library normalizes OAK→ATH (Savant's 2025+ code); the stadium map
    must agree or athletics filters match zero rows. Pre-2025 frames still
    carry OAK, kept reachable via LEGACY_TEAM_CODES."""
    assert stadiums.STADIUM_TO_TEAM_CODE["athletics"] == "ATH"
    assert stadiums.LEGACY_TEAM_CODES["ATH"] == ("OAK",)


def test_metadata_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown stadium"):
        stadiums.metadata("not_a_real_park")


def test_metadata_returns_required_keys() -> None:
    meta = stadiums.metadata("yankees")
    for key in ("name", "location", "lf", "cf", "rf"):
        assert key in meta
