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


# ---------------------------------------------------------------------------
# Injuries note formatting
# ---------------------------------------------------------------------------


@respx.mock
async def test_injuries_note_falls_back_when_stint_or_injury_missing() -> None:
    def _p(name: str, code: str, desc: str, note: str | None) -> dict:
        row = {
            "person": {"fullName": name},
            "position": {"abbreviation": "P"},
            "status": {"code": code, "description": desc},
        }
        if note is not None:
            row["note"] = note
        return row

    respx.get(f"{_BASE}/teams/147/roster").mock(
        return_value=httpx.Response(
            200,
            json={
                "roster": [
                    _p("Stint Only", "D60", "Injured 60-Day", None),
                    _p("Injury Only", "D10", "", "Hamstring"),
                    _p("Neither", "D7", "", "  "),
                ]
            },
        )
    )
    notes = {r["name"]: r["note"] for r in await mlb.injuries("NYY")}
    assert notes == {"Stint Only": "60-Day IL", "Injury Only": "Hamstring", "Neither": "—"}


@respx.mock
async def test_injuries_tolerates_null_status() -> None:
    respx.get(f"{_BASE}/teams/147/roster").mock(
        return_value=httpx.Response(200, json={"roster": [{"person": {}, "status": None}]})
    )
    assert await mlb.injuries("NYY") == []


# ---------------------------------------------------------------------------
# Transactions
# ---------------------------------------------------------------------------


@respx.mock
async def test_transactions_newest_first_with_date_trimmed() -> None:
    route = respx.get(f"{_BASE}/transactions").mock(
        return_value=httpx.Response(
            200,
            json={
                "transactions": [
                    {"date": "2024-05-01T00:00:00", "description": "older"},
                    {"date": "2024-05-03T00:00:00", "description": "newest"},
                    {"date": "2024-05-02T00:00:00"},
                ]
            },
        )
    )
    rows = await mlb.transactions("nyy", days=3)
    assert [r["desc"] for r in rows] == ["newest", "—", "older"]
    assert rows[0]["date"] == "2024-05-03"
    params = route.calls.last.request.url.params
    assert params["teamId"] == "147"


@respx.mock
async def test_transactions_empty_raises() -> None:
    respx.get(f"{_BASE}/transactions").mock(return_value=httpx.Response(200, json={}))
    with pytest.raises(ValueError, match="No transactions for NYY"):
        await mlb.transactions("NYY")


# ---------------------------------------------------------------------------
# Live scores + next game
# ---------------------------------------------------------------------------


def _live_game(state: str, linescore: dict | None = None) -> dict:
    game: dict = {
        "gamePk": 1,
        "gameDate": "2024-05-01T23:05:00Z",
        "status": {"detailedState": state},
        "teams": {
            "home": {"team": {"name": "Yankees"}, "score": 3},
            "away": {"team": {"name": "Red Sox"}},
        },
    }
    if linescore is not None:
        game["linescore"] = linescore
    return game


@respx.mock
async def test_live_scores_formats_in_progress_and_pregame() -> None:
    respx.get(f"{_BASE}/schedule").mock(
        return_value=httpx.Response(
            200,
            json={
                "dates": [
                    {
                        "games": [
                            _live_game(
                                "In Progress",
                                {"currentInningOrdinal": "7th", "inningHalf": "Top", "outs": 2},
                            ),
                            _live_game("Pre-Game"),
                        ]
                    }
                ]
            },
        )
    )
    live, pre = await mlb.live_scores("NYY")
    assert live["inning"] == "Top 7th"
    assert live["outs"] == 2
    assert live["home_score"] == 3
    assert live["away_score"] == 0  # absent score defaults to 0
    assert pre["inning"] == "Pre-Game"  # no linescore: status stands in


@respx.mock
async def test_live_scores_off_day_raises() -> None:
    respx.get(f"{_BASE}/schedule").mock(return_value=httpx.Response(200, json={"dates": []}))
    with pytest.raises(ValueError, match="NYY has no game today"):
        await mlb.live_scores("nyy")


@respx.mock
async def test_next_game_skips_finished_games_and_formats_time() -> None:
    def _g(state: str, game_date: str, home_pp: str | None = None) -> dict:
        home: dict = {"team": {"name": "Yankees"}}
        if home_pp:
            home["probablePitcher"] = {"fullName": home_pp}
        return {
            "gameDate": game_date,
            "status": {"abstractGameState": state},
            "teams": {"home": home, "away": {"team": {"name": "Red Sox"}}},
        }

    respx.get(f"{_BASE}/schedule").mock(
        return_value=httpx.Response(
            200,
            json={
                "dates": [
                    {"date": "2024-05-01", "games": [_g("Final", "2024-05-01T23:05:00Z")]},
                    {
                        "date": "2024-05-02",
                        "games": [_g("Preview", "2024-05-02T17:35:00Z", home_pp="Gerrit Cole")],
                    },
                ]
            },
        )
    )
    game = await mlb.next_game("NYY")
    assert game["date"] == "2024-05-02"
    assert game["time"] == "17:35 UTC"
    assert game["home_probable"] == "Gerrit Cole"
    assert game["away_probable"] == "TBD"


@respx.mock
async def test_next_game_time_tbd_when_unset() -> None:
    respx.get(f"{_BASE}/schedule").mock(
        return_value=httpx.Response(
            200,
            json={
                "dates": [
                    {
                        "date": "2024-05-02",
                        "games": [
                            {
                                "status": {"abstractGameState": "Preview"},
                                "teams": {
                                    "home": {"team": {"name": "H"}},
                                    "away": {"team": {"name": "A"}},
                                },
                            }
                        ],
                    }
                ]
            },
        )
    )
    assert (await mlb.next_game("NYY"))["time"] == "TBD"


@respx.mock
async def test_next_game_none_scheduled_raises() -> None:
    respx.get(f"{_BASE}/schedule").mock(return_value=httpx.Response(200, json={"dates": []}))
    with pytest.raises(ValueError, match="next 14 days"):
        await mlb.next_game("NYY")


# ---------------------------------------------------------------------------
# players_multi_season + standings
# ---------------------------------------------------------------------------


async def test_players_multi_season_fetches_concurrently_and_keeps_order(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    import asyncio

    in_flight = 0
    peak = 0

    async def _fake_players(season: int) -> list[dict]:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0.01)
        in_flight -= 1
        if season == 2022:
            raise RuntimeError("upstream 500")
        return [{"id": season}]

    monkeypatch.setattr(mlb, "players", _fake_players)
    with caplog.at_level("WARNING", logger="bigleague"):
        out = await mlb.players_multi_season([2024, 2022, 2023])

    assert [p["id"] for p in out] == [2024, 2023]  # input order, failure skipped
    assert peak == 3  # all seasons in flight together, not one at a time
    assert "players(2022) failed: upstream 500" in caplog.text


async def test_players_multi_season_propagates_cancellation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio

    async def _cancelled(_season: int) -> list[dict]:
        raise asyncio.CancelledError

    monkeypatch.setattr(mlb, "players", _cancelled)
    with pytest.raises(asyncio.CancelledError):
        await mlb.players_multi_season([2024])


@respx.mock
async def test_players_maps_team_ids_to_canonical_abbr() -> None:
    respx.get(f"{_BASE}/sports/1/players").mock(
        return_value=httpx.Response(
            200,
            json={
                "people": [
                    {"id": 1, "firstName": "A", "lastName": "B", "currentTeam": {"id": 135}},
                    {"id": 2, "firstName": "C", "lastName": "D", "currentTeam": None},
                ]
            },
        )
    )
    out = await mlb.players(2024)
    assert [p["team"] for p in out] == ["SD", ""]


@respx.mock
async def test_standings_empty_raises() -> None:
    respx.get(f"{_BASE}/standings").mock(return_value=httpx.Response(200, json={"records": []}))
    with pytest.raises(ValueError, match="No standings data for 2024"):
        await mlb.standings(2024)
