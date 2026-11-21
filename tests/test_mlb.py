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


def _sched_game(
    pk: int,
    home_id: int,
    away_id: int,
    *,
    state: str = "Final",
    detailed: str | None = None,
    home_score: int = 0,
    away_score: int = 0,
    winner: str | None = None,
    time: str = "2024-04-01T23:05:00Z",
) -> dict:
    home = {"team": {"id": home_id, "name": f"T{home_id}"}, "score": home_score}
    away = {"team": {"id": away_id, "name": f"T{away_id}"}, "score": away_score}
    if winner == "home":
        home["isWinner"] = True
    elif winner == "away":
        away["isWinner"] = True
    return {
        "gamePk": pk,
        "gameDate": time,
        "status": {"abstractGameState": state, "detailedState": detailed or state},
        "teams": {"home": home, "away": away},
    }


@respx.mock
async def test_schedule_builds_running_record_in_play_order() -> None:
    route = respx.get(f"{_BASE}/schedule").mock(
        return_value=httpx.Response(
            200,
            json={
                "dates": [
                    # Deliberately out of order: the API order must not matter.
                    {
                        "date": "2024-04-03",
                        "games": [_sched_game(4, 147, 111, state="Preview", detailed="Scheduled")],
                    },
                    {
                        "date": "2024-04-01",
                        "games": [
                            # Doubleheader: game 2 listed first, earlier game must sort first.
                            _sched_game(
                                2,
                                111,
                                147,
                                home_score=2,
                                away_score=6,
                                winner="away",
                                time="2024-04-01T23:05:00Z",
                            ),
                            _sched_game(
                                1,
                                111,
                                147,
                                home_score=4,
                                away_score=3,
                                winner="home",
                                time="2024-04-01T17:05:00Z",
                            ),
                        ],
                    },
                    {
                        "date": "2024-04-02",
                        "games": [_sched_game(3, 147, 110, state="Preview", detailed="Postponed")],
                    },
                ]
            },
        )
    )
    rows = await mlb.schedule("NYY", 2024)

    assert [r["game_pk"] for r in rows] == [1, 2, 3, 4]
    assert [r["status"] for r in rows] == ["L", "W", "P", "S"]
    assert [r["record"] for r in rows] == ["0-1", "1-1", "1-1", "1-1"]
    assert rows[0]["score"] == "3-4"  # NYY's runs first, even when away
    assert rows[1]["score"] == "6-2"
    assert rows[0]["is_home"] is False
    assert rows[0]["opponent"] == "BOS"  # canonical abbr from the team id
    assert rows[2]["opponent"] == "BAL"
    assert rows[2]["score"] == "" == rows[3]["score"]
    params = route.calls.last.request.url.params
    assert params["season"] == "2024"
    assert params["teamId"] == "147"
    assert params["gameType"] == "R"


@respx.mock
async def test_schedule_infers_winner_when_is_winner_missing() -> None:
    respx.get(f"{_BASE}/schedule").mock(
        return_value=httpx.Response(
            200,
            json={
                "dates": [
                    {
                        "date": "2024-04-01",
                        "games": [_sched_game(1, 147, 111, home_score=5, away_score=1)],
                    }
                ]
            },
        )
    )
    rows = await mlb.schedule("NYY", 2024)
    assert rows[0]["status"] == "W"
    assert rows[0]["record"] == "1-0"


@respx.mock
async def test_schedule_falls_back_to_team_name_for_unknown_opponent() -> None:
    respx.get(f"{_BASE}/schedule").mock(
        return_value=httpx.Response(
            200,
            json={
                "dates": [
                    {
                        "date": "2024-03-01",
                        "games": [_sched_game(1, 147, 9999, home_score=1, away_score=0)],
                    }
                ]
            },
        )
    )
    rows = await mlb.schedule("NYY", 2024, game_type="S")
    assert rows[0]["opponent"] == "T9999"


@respx.mock
async def test_schedule_empty_raises() -> None:
    respx.get(f"{_BASE}/schedule").mock(return_value=httpx.Response(200, json={"dates": []}))
    with pytest.raises(ValueError, match="No 'R' schedule for NYY in 2099"):
        await mlb.schedule("NYY", 2099)


async def test_schedule_rejects_unknown_team() -> None:
    with pytest.raises(ValueError, match="Unknown team"):
        await mlb.schedule("XXX", 2024)
