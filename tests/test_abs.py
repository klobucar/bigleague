from __future__ import annotations

import httpx
import pytest
import respx
from bigleague import abs

_ABS_URL = "https://baseballsavant.mlb.com/leaderboard/abs-challenges"

# Columns Savant ships on the player-role ABS feeds (subset we rely on).
_PLAYER_COLS = [
    "entity_name",
    "team_abbr",
    "n_challenges",
    "n_overturns",
    "n_confirms",
    "n_strikeouts_flip",
    "n_walks_flip",
    "rate_overturns",
]


def _csv(headers: list[str], rows: list[list[str]], *, bom: bool = False) -> bytes:
    body = "\n".join([",".join(headers), *[",".join(r) for r in rows]]).encode("utf-8")
    return b"\xef\xbb\xbf" + body if bom else body


def _route_by_role(feeds: dict[str, bytes]) -> None:
    """Mock the ABS endpoint, dispatching on the challengeType query param."""

    def _handler(request: httpx.Request) -> httpx.Response:
        role = request.url.params["challengeType"]
        return httpx.Response(200, content=feeds.get(role, _csv(_PLAYER_COLS, [])))

    respx.get(_ABS_URL).mock(side_effect=_handler)


# --- parsing quirks --------------------------------------------------------


def test_parse_strips_bom_and_blanks_are_null() -> None:
    body = _csv(
        ["entity_name", "team_abbr", "n_overturns", "n_confirms"],
        [["Aaron Judge", "NYY", "0", ""]],
        bom=True,
    )
    df = abs._parse_csv(body)
    # BOM must not leak into the first column name.
    assert df.columns[0] == "entity_name"
    row = df.row(0, named=True)
    assert row["n_overturns"] == 0  # an explicit 0 survives
    assert row["n_confirms"] is None  # a blank cell is null, not "" or 0


def test_normalize_team_maps_to_savant_abbrevs() -> None:
    assert abs.normalize_team("ARI") == "AZ"
    assert abs.normalize_team("oak") == "ATH"
    assert abs.normalize_team("WSN") == "WSH"
    assert abs.normalize_team("WAS") == "WSH"
    # An already-Savant abbrev passes through untouched.
    assert abs.normalize_team("nyy") == "NYY"


# --- high-level helpers ----------------------------------------------------


@respx.mock
async def test_team_summary_normalizes_and_returns_row() -> None:
    feed = _csv(
        ["team_abbr", "n_challenges", "n_overturns"],
        [["AZ", "40", "18"], ["NYY", "35", "15"]],
    )
    _route_by_role({"team-summary": feed})

    row = await abs.team_summary("ARI", 2026)
    assert row["team_abbr"] == "AZ"
    assert row["n_overturns"] == 18


@respx.mock
async def test_team_summary_missing_raises() -> None:
    _route_by_role({"team-summary": _csv(["team_abbr", "n_challenges"], [["NYY", "35"]])})
    with pytest.raises(ValueError, match="No ABS data"):
        await abs.team_summary("BOS", 2026)


@respx.mock
async def test_team_top_challengers_dedupes_keeping_max() -> None:
    # A two-way player shows up in both the batter and pitcher feeds; we want
    # one entry carrying the larger challenge count.
    batter = _csv(
        _PLAYER_COLS,
        [["Shohei Ohtani", "LAD", "5", "3", "2", "1", "0", "0.6"]],
    )
    pitcher = _csv(
        _PLAYER_COLS,
        [["Shohei Ohtani", "LAD", "2", "1", "1", "0", "0", "0.5"]],
    )
    catcher = _csv(
        _PLAYER_COLS,
        [["Will Smith", "LAD", "4", "1", "3", "0", "0", "0.25"]],
    )
    _route_by_role({"batter": batter, "pitcher": pitcher, "catcher": catcher})

    top = await abs.team_top_challengers("LAD", 2026, limit=5)
    names = [r["entity_name"] for r in top]
    assert names.count("Shohei Ohtani") == 1
    ohtani = next(r for r in top if r["entity_name"] == "Shohei Ohtani")
    assert ohtani["n_challenges"] == 5  # the batter row won
    assert ohtani["role"] == "batter"
    # Sorted by challenge count desc.
    assert names == ["Shohei Ohtani", "Will Smith"]


@respx.mock
async def test_league_leaders_floor_and_sort_order() -> None:
    batter = _csv(
        _PLAYER_COLS,
        [
            # 1 challenge, perfect rate — must be filtered out by the floor.
            ["Rookie One", "SF", "1", "1", "0", "0", "0", "1.0"],
            ["High Rate", "SD", "5", "4", "1", "0", "0", "0.8"],
        ],
    )
    pitcher = _csv(
        _PLAYER_COLS,
        [
            ["Mid Rate", "TB", "10", "6", "4", "0", "0", "0.6"],
            # Same rate as High Rate, but more challenges → ranks ahead on tiebreak.
            ["Tie Rate", "TEX", "20", "16", "4", "0", "0", "0.8"],
        ],
    )
    _route_by_role({"batter": batter, "pitcher": pitcher, "catcher": _csv(_PLAYER_COLS, [])})

    leaders = await abs.league_leaders(2026, limit=10, min_challenges=3)
    names = [r["entity_name"] for r in leaders]
    assert "Rookie One" not in names  # below the floor
    # rate desc, then n_challenges desc as tiebreak.
    assert names == ["Tie Rate", "High Rate", "Mid Rate"]
    assert leaders[0]["role"] == "pitcher"


@respx.mock
async def test_player_merges_feeds_and_sums_totals() -> None:
    batter = _csv(_PLAYER_COLS, [["Shohei Ohtani", "LAD", "5", "3", "2", "1", "0", "0.6"]])
    pitcher = _csv(_PLAYER_COLS, [["Shohei Ohtani", "LAD", "2", "1", "1", "0", "1", "0.5"]])
    _route_by_role({"batter": batter, "pitcher": pitcher, "catcher": _csv(_PLAYER_COLS, [])})

    out = await abs.player("Shohei", "Ohtani", 2026)
    assert out["name"] == "Shohei Ohtani"
    assert out["team_abbr"] == "LAD"
    assert out["totals"]["n_challenges"] == 7
    assert out["totals"]["n_overturns"] == 4
    assert out["totals"]["n_walks_flip"] == 1
    assert len(out["roles"]) == 2


@respx.mock
async def test_player_not_found_raises() -> None:
    empty = _csv(_PLAYER_COLS, [])
    _route_by_role({"batter": empty, "pitcher": empty, "catcher": empty})
    with pytest.raises(ValueError, match="No ABS challenge data"):
        await abs.player("Nobody", "Athome", 2026)
