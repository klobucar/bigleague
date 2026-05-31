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
    data = [{"Name": f"P{i}", "Team": "NYY", "WAR": i} for i in range(150)]
    data.append({"Name": "Late", "Team": "BOS", "WAR": 0.000838})
    _patch_cc(monkeypatch, MagicMock(return_value=_fake_resp(200, {"data": data})))

    df = await fangraphs.leaderboard(2024, "bat")
    assert df["WAR"].dtype == pl.Float64
    assert df.height == 151
    # The float survives rather than being coerced to 0.
    assert df.filter(pl.col("Name") == "Late")["WAR"][0] == 0.000838


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
