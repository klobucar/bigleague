from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import polars as pl
import pytest
from bigleague import fangraphs


def _fake_resp(status: int = 200, payload: dict[str, Any] | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload or {"data": []}
    resp.raise_for_status.return_value = None
    return resp


def _fake_html_resp(html: str, status: int = 200) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.text = html
    resp.raise_for_status.return_value = None
    return resp


def _next_data_html(queries: list[dict[str, Any]]) -> str:
    payload = {"props": {"pageProps": {"dehydratedState": {"queries": queries}}}}
    return (
        '<html><body><script id="__NEXT_DATA__" type="application/json">'
        f"{json.dumps(payload)}</script></body></html>"
    )


def _patch_cc(monkeypatch: pytest.MonkeyPatch, mock_get: Any) -> None:
    """Replace curl_cffi.requests.get with our mock."""
    import curl_cffi.requests as cc

    monkeypatch.setattr(cc, "get", mock_get)


async def test_403_raises_fangraphs_blocked_error(monkeypatch: pytest.MonkeyPatch) -> None:
    mock = MagicMock(return_value=_fake_resp(403))
    _patch_cc(monkeypatch, mock)

    with pytest.raises(fangraphs.FangraphsBlockedError):
        await fangraphs.leaderboard(2024, "bat")
    assert mock.call_count == 1  # 403 doesn't retry


async def test_5xx_retries_three_times(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [_fake_resp(503), _fake_resp(503), _fake_resp(200, {"data": []})]
    mock = MagicMock(side_effect=responses)
    _patch_cc(monkeypatch, mock)
    monkeypatch.setattr(fangraphs.time, "sleep", lambda *_a: None)

    df = await fangraphs.leaderboard(2024, "bat")
    assert df.is_empty()
    assert mock.call_count == 3


async def test_html_anchor_tags_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        "data": [
            {
                "Name": '<a href="/players/judge">Aaron Judge</a>',
                "Team": '<a href="/teams/nyy">NYY</a>',
                "WAR": 8.5,
                "PA": 700,
            },
        ]
    }
    _patch_cc(monkeypatch, MagicMock(return_value=_fake_resp(200, payload)))

    df = await fangraphs.leaderboard(2024, "bat")
    assert df["Name"][0] == "Aaron Judge"
    assert df["Team"][0] == "NYY"


async def test_leaderboard_mixed_int_then_float_builds_float64(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression: a column that's int in early rows but float in a later row
    (past the default inference window) must build as Float64, not raise
    ComputeError or truncate the float to 0."""
    # 150 int rows then one float row — float lands outside the 100-row window.
    data: list[dict[str, Any]] = [{"Name": f"P{i}", "Team": "NYY", "WAR": i} for i in range(150)]
    data.append({"Name": "Late", "Team": "BOS", "WAR": 0.000838})
    _patch_cc(monkeypatch, MagicMock(return_value=_fake_resp(200, {"data": data})))

    df = await fangraphs.leaderboard(2024, "bat")
    assert df["WAR"].dtype == pl.Float64
    assert df.height == 151
    # The float survives rather than being coerced to 0.
    assert df.filter(pl.col("Name") == "Late")["WAR"][0] == 0.000838


async def test_batting_stats_sends_start_and_end_season(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: end_season was dropped — season/season1 both got start_season.
    FanGraphs semantics: season1 = start year, season = end year."""
    seen: list[dict[str, Any]] = []

    def _get(_url, params, **_kw):
        seen.append(params)
        return _fake_resp(200, {"data": []})

    _patch_cc(monkeypatch, MagicMock(side_effect=_get))

    await fangraphs.batting_stats(2021, 2023)
    assert seen[0]["season1"] == 2021
    assert seen[0]["season"] == 2023


async def test_pitching_stats_sends_start_and_end_season(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[dict[str, Any]] = []

    def _get(_url, params, **_kw):
        seen.append(params)
        return _fake_resp(200, {"data": []})

    _patch_cc(monkeypatch, MagicMock(side_effect=_get))

    await fangraphs.pitching_stats(2021, 2023)
    assert seen[0]["season1"] == 2021
    assert seen[0]["season"] == 2023


async def test_fielding_stats_sends_start_and_end_season(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: fielding sent season=start, season1=end — inverted.
    FanGraphs semantics: season1 = start year, season = end year."""
    seen: list[dict[str, Any]] = []

    def _get(_url, params, **_kw):
        seen.append(params)
        return _fake_resp(200, {"data": []})

    _patch_cc(monkeypatch, MagicMock(side_effect=_get))

    await fangraphs.fielding_stats(2021, 2023)
    assert seen[0]["season1"] == 2021
    assert seen[0]["season"] == 2023


@pytest.mark.parametrize("fn_name", ["team_batting", "team_pitching"])
async def test_team_stats_send_start_and_end_season(
    monkeypatch: pytest.MonkeyPatch, fn_name: str
) -> None:
    """Regression: team endpoints sent season=start, season1=end — inverted."""
    seen: list[dict[str, Any]] = []

    def _get(_url, params, **_kw):
        seen.append(params)
        return _fake_resp(200, {"data": []})

    _patch_cc(monkeypatch, MagicMock(side_effect=_get))

    await getattr(fangraphs, fn_name)(2021, 2023)
    assert seen[0]["season1"] == 2021
    assert seen[0]["season"] == 2023


async def test_stats_range_sends_start_and_end_year(monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: date-range mode also had season/season1 inverted."""
    from datetime import date

    seen: list[dict[str, Any]] = []

    def _get(_url, params, **_kw):
        seen.append(params)
        return _fake_resp(200, {"data": []})

    _patch_cc(monkeypatch, MagicMock(side_effect=_get))

    await fangraphs.batting_stats_range(date(2021, 4, 1), date(2023, 9, 1))
    assert seen[0]["season1"] == 2021
    assert seen[0]["season"] == 2023
    assert seen[0]["startdate"] == "2021-04-01"
    assert seen[0]["enddate"] == "2023-09-01"


async def test_leaderboard_single_year_sends_equal_seasons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[dict[str, Any]] = []

    def _get(_url, params, **_kw):
        seen.append(params)
        return _fake_resp(200, {"data": []})

    _patch_cc(monkeypatch, MagicMock(side_effect=_get))

    await fangraphs.leaderboard(2024, "bat")
    assert seen[0]["season"] == 2024
    assert seen[0]["season1"] == 2024


async def test_player_stats_auto_detects_pitcher(monkeypatch: pytest.MonkeyPatch) -> None:
    pitch_payload = {"data": [{"Name": "Gerrit Cole", "Team": "NYY", "ERA": 3.5, "WAR": 5.0}]}
    bat_payload = {"data": []}

    seen: list[dict[str, Any]] = []

    def _get(_url, params, **_kw):
        seen.append(params)
        if params["stats"] == "pit":
            return _fake_resp(200, pitch_payload)
        return _fake_resp(200, bat_payload)

    _patch_cc(monkeypatch, MagicMock(side_effect=_get))

    out = await fangraphs.player_stats("Gerrit", "Cole", 2024)
    assert out["type"] == "pitcher"
    assert out["team"] == "NYY"


async def test_player_stats_falls_back_to_batter(monkeypatch: pytest.MonkeyPatch) -> None:
    pitch_payload: dict[str, Any] = {"data": []}
    bat_payload = {"data": [{"Name": "Aaron Judge", "Team": "NYY", "AVG": 0.310, "WAR": 8.0}]}

    def _get(_url, params, **_kw):
        return _fake_resp(200, pitch_payload if params["stats"] == "pit" else bat_payload)

    _patch_cc(monkeypatch, MagicMock(side_effect=_get))

    out = await fangraphs.player_stats("Aaron", "Judge", 2024)
    assert out["type"] == "batter"


async def test_player_stats_not_found_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    empty: dict[str, Any] = {"data": []}
    _patch_cc(monkeypatch, MagicMock(return_value=_fake_resp(200, empty)))

    with pytest.raises(ValueError, match="No FanGraphs stats"):
        await fangraphs.player_stats("Nobody", "Athome", 2024)


async def test_top_leaderboard_sorts_descending(monkeypatch: pytest.MonkeyPatch) -> None:
    bat_payload = {
        "data": [
            {"Name": "B", "Team": "BOS", "HR": 30},
            {"Name": "A", "Team": "NYY", "HR": 50},
            {"Name": "C", "Team": "TB", "HR": 40},
        ]
    }
    pitch_payload: dict[str, Any] = {"data": []}

    def _get(_url, params, **_kw):
        return _fake_resp(200, pitch_payload if params["stats"] == "pit" else bat_payload)

    _patch_cc(monkeypatch, MagicMock(side_effect=_get))

    out = await fangraphs.top_leaderboard("HR", 2024, n=3, player_type="batter")
    assert [r["name"] for r in out] == ["A", "C", "B"]
    assert out[0]["rank"] == 1


async def test_top_leaderboard_sorts_ascending_for_era(monkeypatch: pytest.MonkeyPatch) -> None:
    """Lower-is-better stats (ERA, FIP, WHIP, ...) must sort ascending —
    'top 10 ERA' means the best pitchers, not the worst."""
    pitch_payload = {
        "data": [
            {"Name": "B", "Team": "BOS", "ERA": 4.50},
            {"Name": "A", "Team": "NYY", "ERA": 2.10},
            {"Name": "C", "Team": "TB", "ERA": 3.20},
        ]
    }

    def _get(_url, params, **_kw):
        return _fake_resp(200, pitch_payload if params["stats"] == "pit" else {"data": []})

    _patch_cc(monkeypatch, MagicMock(side_effect=_get))

    out = await fangraphs.top_leaderboard("ERA", 2024, n=3, player_type="pitcher")
    assert [r["name"] for r in out] == ["A", "C", "B"]
    assert out[0]["value"] == "2.100"


# --- career stats (probe + gather) ------------------------------------------


def _patch_mlb_years(monkeypatch: pytest.MonkeyPatch, debut: int, last: int) -> None:
    async def _fake_years(_first: str, _last: str) -> tuple[int, int]:
        return (debut, last)

    import bigleague.players as players_mod

    monkeypatch.setattr(players_mod, "mlb_years", _fake_years)


async def test_career_stats_probes_pitcher_side_then_gathers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pitcher's career must never fetch the batter leaderboard: the first
    year is probed, then the remaining years fetch only the matched side."""
    _patch_mlb_years(monkeypatch, 2020, 2022)
    era = {2020: 3.00, 2021: 6.00, 2022: 4.50}
    seen: list[dict[str, Any]] = []

    def _get(_url, params, **_kw):
        seen.append(dict(params))
        if params["stats"] != "pit":
            return _fake_resp(200, {"data": []})
        row = {
            "Name": "Test Player",
            "Team": "NYY",
            "IP": 100.0,
            "ERA": era[params["season"]],
            "WAR": 2.0,
        }
        return _fake_resp(200, {"data": [row]})

    _patch_cc(monkeypatch, MagicMock(side_effect=_get))

    out = await fangraphs.career_stats("Test", "Player")
    assert out["type"] == "pitcher"
    stats_block = out["stats"]
    assert isinstance(stats_block, dict)
    assert stats_block["Seasons"] == "3"
    assert stats_block["IP"] == "300.0"
    # Equal-IP seasons: ERA averages to (3 + 6 + 4.5) / 3 = 4.5; WAR sums.
    assert stats_block["ERA"] == "4.500"
    assert stats_block["WAR"] == "6.000"
    # Exactly one fetch per career year, all pitcher-side.
    assert all(p["stats"] == "pit" for p in seen)
    assert sorted(p["season"] for p in seen) == [2020, 2021, 2022]


async def test_career_stats_falls_back_to_batter_side(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_mlb_years(monkeypatch, 2020, 2021)
    seen: list[dict[str, Any]] = []

    def _get(_url, params, **_kw):
        seen.append(dict(params))
        if params["stats"] == "pit":
            return _fake_resp(200, {"data": []})
        row = {"Name": "Aaron Judge", "Team": "NYY", "PA": 600.0, "AVG": 0.300, "WAR": 8.0}
        return _fake_resp(200, {"data": [row]})

    _patch_cc(monkeypatch, MagicMock(side_effect=_get))

    out = await fangraphs.career_stats("Aaron", "Judge")
    assert out["type"] == "batter"
    stats_block = out["stats"]
    assert isinstance(stats_block, dict)
    assert stats_block["Seasons"] == "2"
    # Pitcher side hit only for the probe year; the rest go straight to bat.
    assert [p["stats"] for p in seen].count("pit") == 1


async def test_career_stats_blocked_error_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_mlb_years(monkeypatch, 2020, 2022)
    _patch_cc(monkeypatch, MagicMock(return_value=_fake_resp(403)))

    with pytest.raises(fangraphs.FangraphsBlockedError):
        await fangraphs.career_stats("Test", "Player")


async def test_career_stats_ambiguous_probe_sweeps_remaining_years(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No data in the debut year (probe miss) must not lose the career —
    the remaining years are swept per side."""
    _patch_mlb_years(monkeypatch, 2020, 2022)

    def _get(_url, params, **_kw):
        if params["stats"] == "pit" and params["season"] in (2021, 2022):
            row = {"Name": "Test Player", "Team": "NYY", "IP": 100.0, "ERA": 3.00, "W": 10}
            return _fake_resp(200, {"data": [row]})
        return _fake_resp(200, {"data": []})

    _patch_cc(monkeypatch, MagicMock(side_effect=_get))

    out = await fangraphs.career_stats("Test", "Player")
    assert out["type"] == "pitcher"
    stats_block = out["stats"]
    assert isinstance(stats_block, dict)
    assert stats_block["Seasons"] == "2"
    assert stats_block["W"] == "20.0"


# --- playoff odds ----------------------------------------------------------


def _odds_query(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"queryHash": json.dumps(["playoff-odds", "div", "fg"]), "state": {"data": rows}}


async def test_playoff_odds_parses_next_data(monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [
        {
            "abbName": "NYY",
            "shortName": "Yankees",
            "W": 50,
            "L": 30,
            "Wpct": 0.625,
            "GB": 0.0,
            "endData": {
                "ExpW": 95,
                "ExpL": 67,
                "rosW": 0.50,
                "poffTitle": 0.99,
                "divTitle": 0.80,
                "wcTitle": 0.19,
                "wsWin": 0.15,
            },
        },
        {
            "abbName": "BOS",
            "shortName": "Red Sox",
            "W": 45,
            "L": 35,
            "Wpct": 0.563,
            "GB": 5.0,
            "endData": {
                "ExpW": 88,
                "ExpL": 74,
                "rosW": 0.42,
                "poffTitle": 0.65,
                "divTitle": 0.18,
                "wcTitle": 0.47,
                "wsWin": 0.06,
            },
        },
    ]
    # A decoy query without "playoff-odds" must be skipped.
    html = _next_data_html(
        [
            {"queryHash": json.dumps(["standings"]), "state": {"data": []}},
            _odds_query(rows),
        ]
    )
    _patch_cc(monkeypatch, MagicMock(return_value=_fake_html_resp(html)))

    df = await fangraphs.playoff_odds()
    assert df.height == 2
    assert set(df.columns) == {
        "abbName", "shortName", "W", "L", "Wpct", "GB", "exp_w", "exp_l",
        "ros_pct", "playoff_pct", "division_pct", "wildcard_pct", "ws_pct",
    }  # fmt: skip
    nyy = df.filter(pl.col("abbName") == "NYY").row(0, named=True)
    assert nyy["shortName"] == "Yankees"
    assert nyy["exp_w"] == 95
    # Probabilities stay as raw [0, 1] fractions — caller scales to %.
    assert nyy["playoff_pct"] == 0.99
    assert nyy["ws_pct"] == 0.15
    assert nyy["division_pct"] == 0.80
    assert nyy["wildcard_pct"] == 0.19


async def test_playoff_odds_layout_changed_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    # __NEXT_DATA__ present, but no query mentions playoff-odds.
    html = _next_data_html([{"queryHash": json.dumps(["standings"]), "state": {"data": []}}])
    _patch_cc(monkeypatch, MagicMock(return_value=_fake_html_resp(html)))

    with pytest.raises(ValueError, match="layout may have changed"):
        await fangraphs.playoff_odds()


async def test_playoff_odds_403_raises_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_cc(monkeypatch, MagicMock(return_value=_fake_resp(403)))
    with pytest.raises(fangraphs.FangraphsBlockedError):
        await fangraphs.playoff_odds()


# ── single-player lookups filter the league before building a frame ───────────

_LEAGUE = [
    {"Name": '<a href="statss.aspx?playerid=1">Aaron Judge</a>', "Team": "NYY", "WAR": 8.0},
    {"Name": '<a href="statss.aspx?playerid=2">Mike Trout</a>', "Team": "LAA", "WAR": 6.0},
    {"Name": "Juan Soto", "Team": "NYM", "WAR": 7.0},
    {"Name": None, "Team": "—", "WAR": 0.0},
]


def test_name_filter_matches_name_match_semantics() -> None:
    """The record-level filter must agree with `_name_match` on a built frame,
    including anchor tags, case, surrounding whitespace and null names."""
    records = [*_LEAGUE, {"Name": "  <b>AARON JUDGE</b> "}, {"Name": "Aaron Judgement"}]
    keep = fangraphs._name_filter("aaron", "JUDGE")
    kept = [r for r in records if keep(r)]

    via_frame = fangraphs._name_match(fangraphs._rows_to_df(records), "aaron", "JUDGE")
    assert len(kept) == via_frame.height == 2
    assert fangraphs._name_match(fangraphs._rows_to_df(kept), "aaron", "judge").height == 2


async def test_player_lookup_builds_a_frame_of_only_the_matching_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The point of the filter: the whole league is never turned into a frame
    (the endpoint returns ~7 MB of JSON and 500+ columns per row)."""
    _patch_cc(monkeypatch, MagicMock(return_value=_fake_resp(200, {"data": _LEAGUE})))
    built: list[int] = []
    real = fangraphs._rows_to_df

    def _spy(rows: list[dict[str, Any]]) -> pl.DataFrame:
        built.append(len(rows))
        return real(rows)

    monkeypatch.setattr(fangraphs, "_rows_to_df", _spy)

    df = await fangraphs.player_stats_for_year("Aaron", "Judge", 2024, "batter")

    assert built == [1]  # one row, not the league's four
    assert df is not None
    assert df["Team"].to_list() == ["NYY"]
    assert df["_year"].to_list() == [2024]


async def test_player_stats_filters_both_boards(monkeypatch: pytest.MonkeyPatch) -> None:
    pit_league = [
        {"Name": "Gerrit Cole", "Team": "NYY", "ERA": 3.5},
        {"Name": "Paul Skenes", "Team": "PIT", "ERA": 1.9},
    ]

    def _get(_url, params, **_kw):
        return _fake_resp(200, {"data": pit_league if params["stats"] == "pit" else _LEAGUE})

    _patch_cc(monkeypatch, MagicMock(side_effect=_get))
    built: list[int] = []
    real = fangraphs._rows_to_df
    monkeypatch.setattr(
        fangraphs, "_rows_to_df", lambda rows: (built.append(len(rows)), real(rows))[1]
    )

    out = await fangraphs.player_stats("Aaron", "Judge", 2024)  # not a pitcher: tries both

    assert out["type"] == "batter"
    assert out["team"] == "NYY"
    assert built == [1]  # pitcher board had no match, so no frame; batter board: 1 row


async def test_public_leaderboards_still_return_the_whole_league(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only single-player lookups filter; `leaderboard` is unchanged."""
    _patch_cc(monkeypatch, MagicMock(return_value=_fake_resp(200, {"data": _LEAGUE})))
    df = await fangraphs.leaderboard(2024, "bat")
    assert df.height == len(_LEAGUE)
