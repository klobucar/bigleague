from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from bigleague import fangraphs


def _fake_resp(status: int = 200, payload: dict[str, Any] | None = None) -> MagicMock:
    resp = MagicMock()
    resp.status_code = status
    resp.json.return_value = payload or {"data": []}
    resp.raise_for_status.return_value = None
    return resp


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
