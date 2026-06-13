from __future__ import annotations

import httpx
import pytest
import respx
from bigleague import mlb

_BASE = "https://statsapi.mlb.com/api/v1"


def test_all_canonical_abbreviations_resolve() -> None:
    for ab in mlb._CANONICAL_ABBRS:
        assert ab in mlb.TEAM_IDS


def test_unknown_team_raises_with_valid_list() -> None:
    with pytest.raises(ValueError, match="Unknown team abbreviation 'XYZ'"):
        mlb._team_id("XYZ")


@respx.mock
async def test_roster_returns_active_players() -> None:
    respx.get(f"{_BASE}/teams/147/roster").mock(
        return_value=httpx.Response(
            200,
            json={
                "roster": [
                    {
                        "person": {"fullName": "Aaron Judge"},
                        "position": {"abbreviation": "RF"},
                        "status": {"description": "Active"},
                    }
                ]
            },
        )
    )
    rows = await mlb.roster("NYY")
    assert rows[0]["name"] == "Aaron Judge"
    assert rows[0]["pos"] == "RF"


@respx.mock
async def test_injuries_filters_d_status_codes() -> None:
    respx.get(f"{_BASE}/teams/147/roster").mock(
        return_value=httpx.Response(
            200,
            json={
                "roster": [
                    {
                        "person": {"fullName": "Healthy Hank"},
                        "position": {"abbreviation": "1B"},
                        "status": {"code": "A", "description": "Active"},
                    },
                    {
                        "person": {"fullName": "Hurt Harry"},
                        "position": {"abbreviation": "P"},
                        "status": {"code": "D15", "description": "Injured 15-Day"},
                        "note": "Left elbow strain",
                    },
                ]
            },
        )
    )
    rows = await mlb.injuries("NYY")
    assert len(rows) == 1
    assert rows[0]["name"] == "Hurt Harry"
    assert "15-Day IL" in rows[0]["note"]
    assert "elbow" in rows[0]["note"]


@respx.mock
async def test_recent_results_keeps_postponed_and_sorts_oldest_first() -> None:
    respx.get(f"{_BASE}/schedule").mock(
        return_value=httpx.Response(
            200,
            json={
                "dates": [
                    {
                        "date": "2024-04-02",
                        "games": [
                            {
                                "status": {"abstractGameState": "Final", "detailedState": "Final"},
                                "teams": {
                                    "home": {
                                        "team": {"id": 147, "abbreviation": "NYY"},
                                        "score": 5,
                                        "isWinner": True,
                                    },
                                    "away": {
                                        "team": {"id": 111, "abbreviation": "BOS"},
                                        "score": 3,
                                    },
                                },
                            }
                        ],
                    },
                    {
                        "date": "2024-04-01",
                        "games": [
                            {
                                "status": {
                                    "abstractGameState": "Preview",
                                    "detailedState": "Postponed",
                                },
                                "teams": {
                                    "home": {"team": {"id": 147, "abbreviation": "NYY"}},
                                    "away": {"team": {"id": 110, "abbreviation": "BAL"}},
                                },
                            }
                        ],
                    },
                    {
                        "date": "2024-04-03",
                        "games": [
                            {
                                "status": {
                                    "abstractGameState": "Live",
                                    "detailedState": "In Progress",
                                },
                                "teams": {
                                    "home": {"team": {"id": 147, "abbreviation": "NYY"}},
                                    "away": {"team": {"id": 111, "abbreviation": "BOS"}},
                                },
                            }
                        ],
                    },
                ]
            },
        )
    )
    out = await mlb.recent_results("NYY")
    assert [g["date"] for g in out] == ["2024-04-01", "2024-04-02"]
    assert out[0]["result"] == "P"
    assert out[1]["result"] == "W"


@respx.mock
async def test_standings_returns_six_divisions() -> None:
    respx.get(f"{_BASE}/standings").mock(
        return_value=httpx.Response(
            200,
            json={
                "records": [
                    {
                        "division": {"nameShort": f"Div {i}"},
                        "teamRecords": [
                            {
                                "team": {"name": f"Team {i}"},
                                "wins": 50,
                                "losses": 40,
                                "gamesBack": "-",
                                "winningPercentage": ".556",
                            }
                        ],
                    }
                    for i in range(1, 7)
                ]
            },
        )
    )
    out = await mlb.standings(2024)
    assert len(out) == 6


@respx.mock
async def test_players_prefers_use_name_over_legal_first_name() -> None:
    """statsapi always populates firstName with the LEGAL name; useName is
    what the player goes by (Jazz, not Jasrado) and must win when present."""
    respx.get(f"{_BASE}/sports/1/players").mock(
        return_value=httpx.Response(
            200,
            json={
                "people": [
                    {
                        "id": 665862,
                        "firstName": "Jasrado",
                        "useName": "Jazz",
                        "lastName": "Chisholm Jr.",
                        "primaryPosition": {"abbreviation": "3B"},
                    },
                    {
                        "id": 1,
                        "firstName": "Plain",
                        "lastName": "Fallback",
                        "primaryPosition": {"abbreviation": "P"},
                    },
                ]
            },
        )
    )
    out = await mlb.players(2025)
    assert out[0]["first"] == "Jazz"
    assert out[1]["first"] == "Plain"  # no useName -> legal first name


@respx.mock
async def test_mlb_get_retries_5xx() -> None:
    route = respx.get(f"{_BASE}/teams/147/roster").mock(
        side_effect=[
            httpx.Response(503),
            httpx.Response(
                200,
                json={
                    "roster": [
                        {
                            "person": {"fullName": "X"},
                            "position": {"abbreviation": "P"},
                            "status": {"description": "Active"},
                        }
                    ]
                },
            ),
        ]
    )
    out = await mlb.roster("NYY")
    assert route.call_count == 2
    assert out[0]["name"] == "X"
