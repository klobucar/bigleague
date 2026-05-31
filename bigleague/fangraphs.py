"""FanGraphs JSON API helpers — sync internals + async public surface.

`curl_cffi` with Chrome TLS fingerprinting is load-bearing here:
fangraphs.com sits behind Cloudflare since mid-2025 and any plain-TLS
client (httpx, requests) eats a 403. The Chrome impersonation works
because Cloudflare allows the JA3 fingerprint of a real browser.

`FangraphsBlockedError` distinguishes the 403 case from generic 4xx so
callers can present a "try again in a few minutes" message without
silently treating it as "no data found".

Public functions wrap the sync calls in `asyncio.to_thread()` so they
fit cleanly in async caller code.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import date
from typing import Any, Literal

import polars as pl

from bigleague.stats import _fmt, aggregate_career

log = logging.getLogger("bigleague")

_PITCHING_KEYS: list[str] = [
    "W", "L", "ERA", "FIP", "xFIP", "WHIP", "K/9", "BB/9", "HR/9", "WAR", "IP",
]  # fmt: skip
_BATTING_KEYS: list[str] = [
    "AVG", "OBP", "SLG", "OPS", "wOBA", "wRC+", "HR", "RBI", "SB", "WAR", "PA",
]  # fmt: skip

_FG_API_URL = "https://www.fangraphs.com/api/leaders/major-league/data"
_FG_ODDS_URL = "https://www.fangraphs.com/standings/playoff-odds/fg/div"
_FG_HTML_TAG_RE = re.compile(r"<[^>]+>")
_NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.DOTALL)

_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 0.75

_CF_BLOCKED_MSG = (
    "FanGraphs is blocking the request right now (probably Cloudflare). Try again in a few minutes."
)


class FangraphsBlockedError(ValueError):
    """Raised when FanGraphs returns 403 — Cloudflare is filtering us.

    Subclasses ValueError so existing `except ValueError` paths in
    callers surface the friendly message via str(exc).
    """


# ---------------------------------------------------------------------------
# Sync internals
# ---------------------------------------------------------------------------


def _name_match(df: pl.DataFrame, first: str, last: str) -> pl.DataFrame:
    if "Name" not in df.columns or df.is_empty():
        return df.head(0)
    target = f"{first.lower()} {last.lower()}"
    return df.filter(pl.col("Name").str.to_lowercase() == target)


def _strip_html(df: pl.DataFrame) -> pl.DataFrame:
    """FanGraphs wraps Name/Team in <a> tags; strip them in place."""
    for col in ("Name", "Team"):
        if col in df.columns:
            df = df.with_columns(
                pl.col(col)
                .cast(pl.String)
                .str.replace_all(_FG_HTML_TAG_RE.pattern, "")
                .str.strip_chars()
            )
    return df


def _rows_to_df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Build a frame from FanGraphs JSON records, stripping anchor tags.

    `infer_schema_length=None` scans every row before picking column dtypes:
    FanGraphs ships a column as a bare int in early rows and a float later
    (e.g. a rate that's 0 up top, 0.000838 further down), and the default
    100-row inference window would lock in Int64 and then fail to append the
    float. Inferring across all rows types it Float64 and keeps the value
    (strict=False would instead truncate the float to 0)."""
    return _strip_html(pl.DataFrame(rows, infer_schema_length=None))


def _cc_get(url: str, params: dict[str, Any] | None = None) -> Any:
    """GET a FanGraphs URL through curl_cffi + Chrome impersonation.

    Shared by the JSON API and the Next.js-rendered odds page: 403 becomes a
    FangraphsBlockedError (Cloudflare gate), 5xx/connection errors retry with
    exponential backoff, everything else returns the raw curl_cffi response.
    """
    from curl_cffi import requests as cc_requests

    resp = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            resp = cc_requests.get(url, params=params, impersonate="chrome", timeout=20)
        except Exception as exc:
            if attempt < _RETRY_ATTEMPTS - 1:
                log.warning(
                    "FanGraphs %s (attempt %d/%d), retrying...",
                    type(exc).__name__,
                    attempt + 1,
                    _RETRY_ATTEMPTS,
                )
                time.sleep(_RETRY_BASE_DELAY * (2**attempt))
                continue
            raise

        status = getattr(resp, "status_code", 200)
        if status == 403:
            log.warning("FanGraphs returned 403 — likely Cloudflare gate.")
            raise FangraphsBlockedError(_CF_BLOCKED_MSG)
        if 500 <= status < 600 and attempt < _RETRY_ATTEMPTS - 1:
            log.warning(
                "FanGraphs returned %d (attempt %d/%d), retrying...",
                status,
                attempt + 1,
                _RETRY_ATTEMPTS,
            )
            time.sleep(_RETRY_BASE_DELAY * (2**attempt))
            continue
        break

    if resp is None:
        raise RuntimeError("unreachable")
    resp.raise_for_status()
    return resp


def _fg_get(params: dict[str, Any]) -> dict[str, Any]:
    """Synchronous FanGraphs JSON-API GET."""
    payload = _cc_get(_FG_API_URL, params).json()
    return payload if isinstance(payload, dict) else {"data": payload}


def _fetch_leaderboard_sync(
    year: int,
    kind: Literal["bat", "pit"],
    qual: int | str = 1,
) -> pl.DataFrame:
    if kind not in ("bat", "pit"):
        raise ValueError(f"kind must be 'bat' or 'pit', got {kind!r}")
    params = {
        "age": "",
        "pos": "all",
        "stats": kind,
        "lg": "all",
        "qual": qual,
        "season": year,
        "season1": year,
        "ind": 0,
        "team": 0,
        "month": 0,
        "pageitems": 2000,
    }
    rows = _fg_get(params).get("data") or []
    if not rows:
        return pl.DataFrame()
    return _rows_to_df(rows)


def _fetch_leaderboard_range_sync(
    start: date,
    end: date,
    kind: Literal["bat", "pit"],
) -> pl.DataFrame:
    """FanGraphs supports a startdate/enddate parameter combination on the
    same endpoint; `month=1000` triggers the date-range mode."""
    params = {
        "age": "",
        "pos": "all",
        "stats": kind,
        "lg": "all",
        "qual": 0,
        "season": start.year,
        "season1": end.year,
        "ind": 0,
        "team": 0,
        "month": 1000,
        "startdate": start.isoformat(),
        "enddate": end.isoformat(),
        "pageitems": 2000,
    }
    rows = _fg_get(params).get("data") or []
    if not rows:
        return pl.DataFrame()
    return _rows_to_df(rows)


def _fetch_fielding_sync(
    start_season: int,
    end_season: int,
    qual: int | str = 1,
) -> pl.DataFrame:
    params = {
        "age": "",
        "pos": "all",
        "stats": "fld",
        "lg": "all",
        "qual": qual,
        "season": start_season,
        "season1": end_season,
        "ind": 0,
        "team": 0,
        "month": 0,
        "pageitems": 2000,
    }
    rows = _fg_get(params).get("data") or []
    if not rows:
        return pl.DataFrame()
    return _rows_to_df(rows)


def _fetch_team_sync(
    start_season: int,
    end_season: int,
    kind: Literal["bat", "pit"],
) -> pl.DataFrame:
    """Team-aggregated stats. `team=0&aggregateby=team` sums across players."""
    params = {
        "age": "",
        "pos": "all",
        "stats": kind,
        "lg": "all",
        "qual": 0,
        "season": start_season,
        "season1": end_season,
        "ind": 0,
        "team": "0,ts",  # 'ts' = team summary
        "month": 0,
        "pageitems": 2000,
    }
    rows = _fg_get(params).get("data") or []
    if not rows:
        return pl.DataFrame()
    return _rows_to_df(rows)


# ---------------------------------------------------------------------------
# Public async API
# ---------------------------------------------------------------------------


async def leaderboard(
    year: int,
    kind: Literal["bat", "pit"],
    qual: int | str = 1,
) -> pl.DataFrame:
    return await asyncio.to_thread(_fetch_leaderboard_sync, year, kind, qual)


async def batting_stats(start_season: int, end_season: int, qual: int | str = 1) -> pl.DataFrame:
    return await asyncio.to_thread(_fetch_leaderboard_sync, start_season, "bat", qual)
    # NOTE: harry-bot's pybaseball variant aggregated multi-season; the JSON
    # endpoint with season=A&season1=B returns the multi-year leaderboard.
    # If we need true multi-season aggregation we revisit here.


async def pitching_stats(start_season: int, end_season: int, qual: int | str = 1) -> pl.DataFrame:
    return await asyncio.to_thread(_fetch_leaderboard_sync, start_season, "pit", qual)


async def batting_stats_range(start: date, end: date) -> pl.DataFrame:
    return await asyncio.to_thread(_fetch_leaderboard_range_sync, start, end, "bat")


async def pitching_stats_range(start: date, end: date) -> pl.DataFrame:
    return await asyncio.to_thread(_fetch_leaderboard_range_sync, start, end, "pit")


async def fielding_stats(start_season: int, end_season: int) -> pl.DataFrame:
    return await asyncio.to_thread(_fetch_fielding_sync, start_season, end_season)


async def team_batting(start_season: int, end_season: int) -> pl.DataFrame:
    return await asyncio.to_thread(_fetch_team_sync, start_season, end_season, "bat")


async def team_pitching(start_season: int, end_season: int) -> pl.DataFrame:
    return await asyncio.to_thread(_fetch_team_sync, start_season, end_season, "pit")


# ---------------------------------------------------------------------------
# Player-centric helpers
# ---------------------------------------------------------------------------


def _player_stats_from_row(r: dict[str, Any], keys: list[str]) -> dict[str, str]:
    return {k: _fmt(r.get(k)) for k in keys if k in r}


async def player_stats_for_year(
    first: str,
    last: str,
    year: int,
    player_type: Literal["pitcher", "batter"],
) -> pl.DataFrame | None:
    """One year of FanGraphs stats for a single player; tagged with `_year`.

    Returns None on no-data or non-blocking error so callers can keep
    aggregating other years. `FangraphsBlockedError` always propagates.
    """
    try:
        kind: Literal["bat", "pit"] = "pit" if player_type == "pitcher" else "bat"
        df = await leaderboard(year, kind)
        if df.is_empty():
            return None
        res = _name_match(df, first, last)
        if res.is_empty():
            return None
        return res.with_columns(pl.lit(year).alias("_year"))
    except FangraphsBlockedError:
        raise
    except Exception as exc:
        log.exception("FanGraphs Error [%s %d]: %s", player_type, year, exc)
        return None


async def player_stats(first: str, last: str, year: int) -> dict[str, Any]:
    """FanGraphs season stats for a player; auto-detects pitcher vs batter."""
    pitch_df = await leaderboard(year, "pit")
    if not pitch_df.is_empty():
        row = _name_match(pitch_df, first, last)
        if not row.is_empty():
            r = row.row(0, named=True)
            return {
                "type": "pitcher",
                "team": r.get("Team", "N/A"),
                "stats": _player_stats_from_row(r, _PITCHING_KEYS),
            }

    bat_df = await leaderboard(year, "bat")
    if not bat_df.is_empty():
        row = _name_match(bat_df, first, last)
        if not row.is_empty():
            r = row.row(0, named=True)
            return {
                "type": "batter",
                "team": r.get("Team", "N/A"),
                "stats": _player_stats_from_row(r, _BATTING_KEYS),
            }

    raise ValueError(f"No FanGraphs stats found for {first} {last} in {year}.")


async def career_stats(first: str, last: str) -> dict[str, Any]:
    """Aggregate a player's career across all years they appear in the
    register's mlb_played_first..last range.

    Tries pitcher-side first; falls back to batter-side.
    """
    from bigleague.players import mlb_years

    debut, last_yr = await mlb_years(first, last)
    years = list(range(debut, last_yr + 1))

    pitch_frames: list[pl.DataFrame] = []
    bat_frames: list[pl.DataFrame] = []
    for yr in years:
        p = await player_stats_for_year(first, last, yr, "pitcher")
        if p is not None:
            pitch_frames.append(p)
        b = await player_stats_for_year(first, last, yr, "batter")
        if b is not None:
            bat_frames.append(b)

    if pitch_frames:
        return aggregate_career(pitch_frames, first, last, years, "pitcher")
    if bat_frames:
        return aggregate_career(bat_frames, first, last, years, "batter")
    raise ValueError(f"No FanGraphs career data found for {first} {last}.")


async def top_leaderboard(
    stat: str,
    year: int,
    n: int = 10,
    player_type: Literal["pitcher", "batter", "auto"] = "auto",
) -> list[dict[str, Any]]:
    """Top N players by `stat`. `auto` tries both leaderboards."""

    def _from_df(df: pl.DataFrame) -> list[dict[str, Any]]:
        if df.is_empty():
            return []
        col_map = {c.lower(): c for c in df.columns}
        col = col_map.get(stat.lower())
        if col is None:
            return []
        top = df.sort(col, descending=True).head(n)
        out: list[dict[str, Any]] = []
        for i, row in enumerate(top.iter_rows(named=True)):
            out.append(
                {
                    "rank": i + 1,
                    "name": row.get("Name", "?"),
                    "team": row.get("Team", "—"),
                    "value": _fmt(row.get(col)),
                }
            )
        return out

    boards: list[list[dict[str, Any]]] = []
    if player_type in ("pitcher", "auto"):
        boards.append(_from_df(await leaderboard(year, "pit")))
    if player_type in ("batter", "auto"):
        boards.append(_from_df(await leaderboard(year, "bat")))

    for board in boards:
        if board:
            return board

    raise ValueError(
        f"Stat '{stat}' not found in FanGraphs leaderboards for {year}. "
        "Try a column name like ERA, WAR, HR, K%, OPS, wRC+, FIP, WHIP."
    )


# ---------------------------------------------------------------------------
# Playoff odds (Next.js-rendered page)
# ---------------------------------------------------------------------------


def _parse_playoff_odds(html: str) -> list[dict[str, Any]]:
    """Pull the playoff-odds team rows out of the embedded __NEXT_DATA__ blob.

    The page dehydrates a react-query cache into <script id="__NEXT_DATA__">.
    We walk props → pageProps → dehydratedState → queries and pick the query
    whose queryHash (a JSON-encoded array) mentions "playoff-odds"; its
    state.data is the list of team rows. Raises ValueError if no such query /
    rows are present (the layout changed out from under us).
    """
    match = _NEXT_DATA_RE.search(html)
    if not match:
        raise ValueError(
            "FanGraphs odds page had no __NEXT_DATA__ blob — site layout may have changed."
        )
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        raise ValueError("FanGraphs odds __NEXT_DATA__ blob was not valid JSON.") from exc

    queries = (
        payload.get("props", {}).get("pageProps", {}).get("dehydratedState", {}).get("queries", [])
    )
    for q in queries:
        qhash = q.get("queryHash") or ""
        # queryHash is a JSON-encoded array; a plain substring search works
        # whether it's still a string or already decoded into a list.
        flat = qhash if isinstance(qhash, str) else json.dumps(qhash)
        if "playoff-odds" not in flat:
            continue
        data = q.get("state", {}).get("data")
        if isinstance(data, list) and data and isinstance(data[0], dict) and "abbName" in data[0]:
            return data

    raise ValueError(
        "FanGraphs returned the playoff odds page but no team rows were found — "
        "site layout may have changed."
    )


def _playoff_odds_sync() -> pl.DataFrame:
    resp = _cc_get(_FG_ODDS_URL)
    html = getattr(resp, "text", None)
    if html is None:  # curl_cffi may only expose bytes
        html = resp.content.decode("utf-8", errors="replace")
    records: list[dict[str, Any]] = []
    for row in _parse_playoff_odds(html):
        end = row.get("endData") or {}
        records.append(
            {
                "abbName": row.get("abbName"),
                "shortName": row.get("shortName"),
                "W": row.get("W"),
                "L": row.get("L"),
                "Wpct": row.get("Wpct"),
                "GB": row.get("GB"),
                "exp_w": end.get("ExpW"),
                "exp_l": end.get("ExpL"),
                "ros_pct": end.get("rosW"),
                "playoff_pct": end.get("poffTitle"),
                "division_pct": end.get("divTitle"),
                "wildcard_pct": end.get("wcTitle"),
                "ws_pct": end.get("wsWin"),
            }
        )
    # strict=False so a mixed-type column (e.g. GB rendered as "-" for a
    # division leader alongside numeric gaps) doesn't blow up construction.
    return pl.DataFrame(records, strict=False)


async def playoff_odds() -> pl.DataFrame:
    """FanGraphs playoff odds — one row per team.

    Probability fields stay as raw [0, 1] fractions (the caller scales to %).
    Columns: abbName, shortName, W, L, Wpct, GB, exp_w (ExpW), exp_l (ExpL),
    ros_pct (rosW), playoff_pct (poffTitle), division_pct (divTitle),
    wildcard_pct (wcTitle), ws_pct (wsWin).

    Raises FangraphsBlockedError on a Cloudflare 403, and ValueError if the
    __NEXT_DATA__ blob is present but no playoff-odds rows are found.
    """
    return await asyncio.to_thread(_playoff_odds_sync)
