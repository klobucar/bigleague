"""FanGraphs JSON API helpers — sync internals + async public surface.

`curl_cffi` with Chrome TLS fingerprinting is load-bearing here:
fangraphs.com sits behind Cloudflare since mid-2025 and any plain-TLS
client (httpx, requests) eats a 403. The Chrome impersonation works
because Cloudflare allows the JA3 fingerprint of a real browser.

`FangraphsBlockedError` distinguishes the 403 case from generic 4xx so
callers can present a "try again in a few minutes" message without
silently treating it as "no data found".

Public functions run the sync calls in worker threads so they fit cleanly
in async caller code. Those calls go through a per-event-loop limiter: at
most `_FG_MAX_CONCURRENCY` FanGraphs requests are in flight at once and the
rest queue in asyncio (without occupying executor threads), so a career
lookup fanning out over 20 seasons stays polite to Cloudflare.

All requests share one lazily created `curl_cffi` Session, so TLS
connections are reused instead of re-handshaking on every call (a career
lookup makes one request per season) and Cloudflare's cookies persist
between requests. The Session is thread-safe: it gives each worker thread
its own curl handle.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
import weakref
from collections.abc import Callable
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

# Stats where a smaller value is better — `top_leaderboard` must sort these
# ascending or "top 10 ERA" returns the worst pitchers. Lowercased for
# case-insensitive matching against user-typed stat names.
_LOWER_IS_BETTER_STATS: frozenset[str] = frozenset(
    {"era", "fip", "xfip", "whip", "xera", "siera", "bb/9", "hr/9"}
)

_FG_API_URL = "https://www.fangraphs.com/api/leaders/major-league/data"
_FG_ODDS_URL = "https://www.fangraphs.com/standings/playoff-odds/fg/div"
_FG_HTML_TAG_RE = re.compile(r"<[^>]+>")
_NEXT_DATA_RE = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.DOTALL)

_RETRY_ATTEMPTS = 3
_RETRY_BASE_DELAY = 0.75

# Cap on concurrent FanGraphs requests per event loop. Each request is a
# multi-MB leaderboard behind Cloudflare; more than a few at once risks a 403.
_FG_MAX_CONCURRENCY = 3

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


def _name_filter(first: str, last: str) -> Callable[[dict[str, Any]], bool]:
    """A row predicate with the same meaning as `_name_match`, for raw records.

    `_name_match` compares the lowercased, tag-stripped `Name` to "first last".
    This does the same on a decoded JSON record, so a lookup can drop the rest
    of the league before building a frame.
    """
    target = f"{first.lower()} {last.lower()}"

    def keep(record: dict[str, Any]) -> bool:
        name = _FG_HTML_TAG_RE.sub("", str(record.get("Name", ""))).strip()
        return name.lower() == target

    return keep


# One semaphore per event loop: an asyncio.Semaphore binds to the loop it is
# first contended on, so a single module-level one would break the second
# `asyncio.run(...)` in a process. Weak keys let a closed loop's entry go.
_limiters: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
    weakref.WeakKeyDictionary()
)


def _limiter() -> asyncio.Semaphore:
    loop = asyncio.get_running_loop()
    sem = _limiters.get(loop)
    if sem is None:
        sem = _limiters[loop] = asyncio.Semaphore(_FG_MAX_CONCURRENCY)
    return sem


async def _run_limited[**P, R](fn: Callable[P, R], *args: P.args, **kwargs: P.kwargs) -> R:
    """Run a blocking FanGraphs call in a thread, under the request cap.

    The permit is taken before the thread is, so queued calls wait in
    asyncio rather than parking executor threads. Retry backoff happens
    inside `fn`, so a struggling FanGraphs also slows the queue behind it.
    """
    async with _limiter():
        return await asyncio.to_thread(fn, *args, **kwargs)


async def _player_board(
    year: int, kind: Literal["bat", "pit"], first: str, last: str
) -> pl.DataFrame:
    """One season's leaderboard, reduced to the named player's rows."""
    return await _run_limited(
        _fetch_leaderboard_sync, year, year, kind, 1, _name_filter(first, last)
    )


def _strip_html(df: pl.DataFrame) -> pl.DataFrame:
    """FanGraphs wraps Name/Team in <a> tags; strip them in place."""
    cols = [c for c in ("Name", "Team") if c in df.columns]
    if not cols:
        return df
    return df.with_columns(
        pl.col(cols).cast(pl.String).str.replace_all(_FG_HTML_TAG_RE.pattern, "").str.strip_chars()
    )


def _rows_to_df(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Build a frame from FanGraphs JSON records, stripping anchor tags.

    `infer_schema_length=None` scans every row before picking column dtypes:
    FanGraphs ships a column as a bare int in early rows and a float later
    (e.g. a rate that's 0 up top, 0.000838 further down), and the default
    100-row inference window would lock in Int64 and then fail to append the
    float. Inferring across all rows types it Float64 and keeps the value
    (strict=False would instead truncate the float to 0)."""
    return _strip_html(pl.DataFrame(rows, infer_schema_length=None))


_session: Any = None
_session_lock = threading.Lock()


def _get_session() -> Any:
    """The process-wide curl_cffi Session, created on first use.

    Created lazily so importing this module doesn't import curl_cffi, and
    under a lock so concurrent first calls from worker threads share one.
    """
    global _session
    if _session is None:
        with _session_lock:
            if _session is None:
                from curl_cffi import requests as cc_requests

                _session = cc_requests.Session(impersonate="chrome", timeout=20)
    return _session


def _reset_session() -> None:
    """Close and forget the shared Session. Used by tests."""
    global _session
    with _session_lock:
        if _session is not None:
            _session.close()
        _session = None


def _cc_get(url: str, params: dict[str, Any] | None = None) -> Any:
    """GET a FanGraphs URL through curl_cffi + Chrome impersonation.

    Shared by the JSON API and the Next.js-rendered odds page: 403 becomes a
    FangraphsBlockedError (Cloudflare gate), 5xx/connection errors retry with
    exponential backoff, everything else returns the raw curl_cffi response.
    """
    session = _get_session()
    resp = None
    for attempt in range(_RETRY_ATTEMPTS):
        try:
            resp = session.get(url, params=params, impersonate="chrome", timeout=20)
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
    start_season: int,
    end_season: int,
    kind: Literal["bat", "pit"],
    qual: int | str = 1,
    keep: Callable[[dict[str, Any]], bool] | None = None,
) -> pl.DataFrame:
    """Fetch a season leaderboard as a frame.

    `keep`, if given, is applied to the decoded JSON records *before* the frame
    is built. The endpoint always returns the whole league (about 7 MB of JSON,
    500+ columns per row), so a lookup of one player that built the full frame
    and then filtered it held two copies of the league at peak. Filtering the
    records first leaves only the matching rows to turn into a frame.
    """
    if kind not in ("bat", "pit"):
        raise ValueError(f"kind must be 'bat' or 'pit', got {kind!r}")
    # FanGraphs leaders API: `season1` is the START year and `season` is the
    # END year; season=A&season1=B returns the multi-year leaderboard.
    params = {
        "age": "",
        "pos": "all",
        "stats": kind,
        "lg": "all",
        "qual": qual,
        "season": end_season,
        "season1": start_season,
        "ind": 0,
        "team": 0,
        "month": 0,
        "pageitems": 2000,
    }
    rows = _fg_get(params).get("data") or []
    if keep is not None:
        rows = [r for r in rows if keep(r)]
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
        # `season1` is the START year and `season` is the END year (same
        # semantics as _fetch_leaderboard_sync) even in date-range mode.
        "season": end.year,
        "season1": start.year,
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
        # `season1` = START year, `season` = END year (FanGraphs semantics).
        "season": end_season,
        "season1": start_season,
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
        # `season1` = START year, `season` = END year (FanGraphs semantics).
        "season": end_season,
        "season1": start_season,
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
    return await _run_limited(_fetch_leaderboard_sync, year, year, kind, qual)


async def batting_stats(start_season: int, end_season: int, qual: int | str = 1) -> pl.DataFrame:
    return await _run_limited(_fetch_leaderboard_sync, start_season, end_season, "bat", qual)


async def pitching_stats(start_season: int, end_season: int, qual: int | str = 1) -> pl.DataFrame:
    return await _run_limited(_fetch_leaderboard_sync, start_season, end_season, "pit", qual)


async def batting_stats_range(start: date, end: date) -> pl.DataFrame:
    return await _run_limited(_fetch_leaderboard_range_sync, start, end, "bat")


async def pitching_stats_range(start: date, end: date) -> pl.DataFrame:
    return await _run_limited(_fetch_leaderboard_range_sync, start, end, "pit")


async def fielding_stats(start_season: int, end_season: int) -> pl.DataFrame:
    return await _run_limited(_fetch_fielding_sync, start_season, end_season)


async def team_batting(start_season: int, end_season: int) -> pl.DataFrame:
    return await _run_limited(_fetch_team_sync, start_season, end_season, "bat")


async def team_pitching(start_season: int, end_season: int) -> pl.DataFrame:
    return await _run_limited(_fetch_team_sync, start_season, end_season, "pit")


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
        df = await _player_board(year, kind, first, last)
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
    """FanGraphs season stats for a player; auto-detects pitcher vs batter.

    Both boards are fetched concurrently; a pitcher-board match wins. An
    error on one board only surfaces if the answer depended on it: a
    pitcher match is returned even if the batter board failed.
    """
    pit_res, bat_res = await asyncio.gather(
        _player_board(year, "pit", first, last),
        _player_board(year, "bat", first, last),
        return_exceptions=True,
    )
    sides = (
        ("pitcher", pit_res, _PITCHING_KEYS),
        ("batter", bat_res, _BATTING_KEYS),
    )
    for kind, res, keys in sides:
        if isinstance(res, BaseException):
            raise res
        row = _name_match(res, first, last)
        if not row.is_empty():
            r = row.row(0, named=True)
            return {
                "type": kind,
                "team": r.get("Team", "N/A"),
                "stats": _player_stats_from_row(r, keys),
            }

    raise ValueError(f"No FanGraphs stats found for {first} {last} in {year}.")


async def _career_frames_for_side(
    first: str,
    last: str,
    years: list[int],
    side: Literal["pitcher", "batter"],
) -> list[pl.DataFrame]:
    """Fetch one side's frames for all `years` concurrently, dropping misses."""
    frames = await asyncio.gather(*(player_stats_for_year(first, last, yr, side) for yr in years))
    return [f for f in frames if f is not None]


async def career_stats(first: str, last: str) -> dict[str, Any]:
    """Aggregate a player's career across all years they appear in the
    register's mlb_played_first..last range.

    Probes both boards for the debut year at once to learn whether the
    player lives on the pitcher or batter leaderboard (pitcher-side
    preferred, matching `player_stats`), then fetches the remaining years
    for that side only. If the probe year has the player on neither side
    (injury year, register noise), both sides are swept across the remaining
    years. All fetches share the FanGraphs request cap.
    """
    from bigleague.players import mlb_years

    debut, last_yr = await mlb_years(first, last)
    years = list(range(debut, last_yr + 1))

    probe_year, rest_years = years[0], years[1:]
    pit_probe, bat_probe = await asyncio.gather(
        player_stats_for_year(first, last, probe_year, "pitcher"),
        player_stats_for_year(first, last, probe_year, "batter"),
    )
    side: Literal["pitcher", "batter"] = "pitcher" if pit_probe is not None else "batter"
    probe = pit_probe if pit_probe is not None else bat_probe

    if probe is not None:
        frames = [probe, *await _career_frames_for_side(first, last, rest_years, side)]
        return aggregate_career(frames, first, last, years, side)

    # Ambiguous probe: the player has no data in the debut year. Sweep both
    # sides together; the pitcher side still wins if both have data.
    pitch_frames, bat_frames = await asyncio.gather(
        _career_frames_for_side(first, last, rest_years, "pitcher"),
        _career_frames_for_side(first, last, rest_years, "batter"),
    )
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
        descending = stat.lower() not in _LOWER_IS_BETTER_STATS
        top = df.sort(col, descending=descending, nulls_last=True).head(n)
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

    # Each board is a ~7 MB download, so stop at the first one that has the
    # stat rather than always pulling both in `auto` mode.
    kinds: list[Literal["bat", "pit"]] = []
    if player_type in ("pitcher", "auto"):
        kinds.append("pit")
    if player_type in ("batter", "auto"):
        kinds.append("bat")
    for kind in kinds:
        board = _from_df(await leaderboard(year, kind))
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
    return await _run_limited(_playoff_odds_sync)
