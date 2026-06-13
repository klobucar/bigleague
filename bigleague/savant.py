"""Async Baseball Savant fetchers — Statcast pitch data + leaderboards.

Pitch-level fetchers split the request into 7-day chunks, run them
concurrently under a 5-permit semaphore (Savant's effective rate
limit), and cache each chunk to parquet. Chunks whose end_date >= today
are NOT cached — they would be served stale until the chunk fully
ages into the past.

Leaderboard endpoints are JSON-ish CSV one-shots; no chunking, no cache.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import date, timedelta
from io import BytesIO
from typing import Literal

import polars as pl

from bigleague.cache import cache
from bigleague.http import savant_get
from bigleague.schema import STATCAST_SCHEMA

_SAVANT_BASE = "https://baseballsavant.mlb.com"
_STATCAST_CSV = f"{_SAVANT_BASE}/statcast_search/csv"
_CHUNK_DAYS = 7

# Map common user-typed abbreviations onto the codes Savant actually uses.
# Savant's team filters are case-insensitive but NOT alias-tolerant for these:
# querying "CHW"/"KCR"/"SDP"/"SFG"/"TBR"/"WSN"/"WAS" returns an empty frame,
# so they must be rewritten to Savant's form. Shared with bigleague.abs, which
# matches the same scheme on its leaderboard team_abbr column.
_TEAM_TO_SAVANT: dict[str, str] = {
    "ARI": "AZ",
    "OAK": "ATH",
    "KCR": "KC",
    "SDP": "SD",
    "SFG": "SF",
    "TBR": "TB",
    "CHW": "CWS",
    "WSN": "WSH",
    "WAS": "WSH",
}


# Shape of a plausible team code. Deliberately NOT a closed allowlist of
# current teams (codes get added/renamed — OAK→ATH), but tight enough that
# the result is safe to embed in a cache filename: unknown values used to
# pass through verbatim, so team="../../tmp/pwn" became a path that escaped
# the cache dir.
_TEAM_CODE_RE = re.compile(r"[A-Z]{2,5}")


def normalize_team(abbr: str) -> str:
    """Normalize a user-typed team abbreviation to the form Savant uses.

    Raises ValueError if the result doesn't look like a team code
    (2-5 ASCII uppercase letters).
    """
    up = abbr.strip().upper()
    code = _TEAM_TO_SAVANT.get(up, up)
    if not _TEAM_CODE_RE.fullmatch(code):
        msg = f"invalid team abbreviation: {abbr!r} (expected 2-5 letters, e.g. 'NYY')"
        raise ValueError(msg)
    return code


# ---------------------------------------------------------------------------
# Pitch-level (chunked + cached + concurrent)
# ---------------------------------------------------------------------------


def _chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Split [start, end] into inclusive 7-day windows."""
    out: list[tuple[date, date]] = []
    cur = start
    while cur <= end:
        chunk_end = min(cur + timedelta(days=_CHUNK_DAYS - 1), end)
        out.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    return out


def _player_chunks(start: date, end: date) -> list[tuple[date, date]]:
    """Adaptive chunks for single-player queries.

    The historical portion splits on calendar-year boundaries (Jan 1 -
    Dec 31, clipped to [start, end]): a full player-season is ~3k rows,
    so each year-chunk stays comfortably under Savant's ~25k row cap.
    One fat multi-year request would be silently truncated at the cap —
    and then cached, making the data loss sticky. Year boundaries are
    also stable, so historical chunks stay cache-friendly. The right
    edge — anything from today onward — splits off as its own chunk so
    the existing right-edge cache guard keeps it from being persisted.
    """
    today = date.today()
    out: list[tuple[date, date]] = []
    hist_end = min(end, today - timedelta(days=1))
    cur = start
    while cur <= hist_end:
        chunk_end = min(date(cur.year, 12, 31), hist_end)
        out.append((cur, chunk_end))
        cur = chunk_end + timedelta(days=1)
    if end >= today:
        out.append((max(start, today), end))
    return out


# Schema fingerprint baked into every cache key. Constraint: schema
# evolution must never mix old and new cached frames — parquet written
# under an older StatcastPitch schema breaks `pl.concat(...,
# how="vertical_relaxed")` with freshly fetched frames when columns are
# added (or silently serves frames missing the new columns). A schema
# change changes the fingerprint, so old cache files simply go stale-unused.
_SCHEMA_FINGERPRINT = hashlib.sha256("|".join(STATCAST_SCHEMA).encode()).hexdigest()[:8]


def _cache_key(prefix: str, scope: str, start: date, end: date) -> str:
    return f"{prefix}_{scope}_{start.isoformat()}_{end.isoformat()}_{_SCHEMA_FINGERPRINT}"


def _parse_csv(body: bytes) -> pl.DataFrame:
    """Parse a Savant CSV into the canonical STATCAST_SCHEMA frame.

    Empty bodies (no pitches in the window) return an empty frame with
    the full schema applied so concat downstream stays type-stable.
    """
    if not body or body.strip() in (b"", b'""'):
        return pl.DataFrame(schema=STATCAST_SCHEMA)
    try:
        return pl.read_csv(
            BytesIO(body),
            schema_overrides=STATCAST_SCHEMA,
            null_values=["", "null", "NA"],
            ignore_errors=True,
            try_parse_dates=True,
        )
    except pl.exceptions.NoDataError:
        return pl.DataFrame(schema=STATCAST_SCHEMA)


async def _fetch_chunk(params: dict[str, str], cache_key: str) -> pl.DataFrame:
    """Fetch one chunk with cache check + opportunistic write-through."""
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    resp = await savant_get(_STATCAST_CSV, params)
    df = _parse_csv(resp.content)

    # Right-edge guard: only cache chunks that are fully historical.
    # `params` carries the chunk's end as `game_date_lt`.
    end_str = params.get("game_date_lt", "")
    try:
        chunk_end = date.fromisoformat(end_str)
    except ValueError:
        chunk_end = date.today()  # be conservative
    if chunk_end < date.today():
        cache.set(cache_key, df)
    return df


async def _fetch_many(
    chunks: list[tuple[date, date]],
    base_params: dict[str, str],
    key_prefix: str,
    key_scope: str,
) -> pl.DataFrame:
    coros = []
    for start, end in chunks:
        params = {
            **base_params,
            "game_date_gt": start.isoformat(),
            "game_date_lt": end.isoformat(),
        }
        coros.append(_fetch_chunk(params, _cache_key(key_prefix, key_scope, start, end)))
    frames = await asyncio.gather(*coros)
    non_empty = [f for f in frames if f.height > 0]
    if not non_empty:
        return pl.DataFrame(schema=STATCAST_SCHEMA)
    return pl.concat(non_empty, how="vertical_relaxed")


_BASE_PARAMS: dict[str, str] = {"all": "true", "type": "details"}


async def statcast(
    start: date,
    end: date,
    team: str | None = None,
) -> pl.DataFrame:
    """League-wide pitch data for a date range. Optional team filter."""
    params = dict(_BASE_PARAMS)
    if team:
        # Savant rejects several common aliases (CHW, KCR, SDP, ...) with an
        # empty frame; normalize to its codes so statcast(..., team="CHW")
        # works. Also dedupes the cache scope across "nyy"/"NYY".
        team = normalize_team(team)
        params["team"] = team
    scope = team or "all"
    return await _fetch_many(_chunks(start, end), params, "statcast", scope)


async def statcast_batter(
    player_id: int,
    start: date,
    end: date,
) -> pl.DataFrame:
    # Savant's CSV endpoint silently ignores `player_id`; the actual batter
    # filter is `batters_lookup[]`. Sending the wrong key returns league-wide
    # data capped at ~25k rows per chunk.
    params = {
        **_BASE_PARAMS,
        "batters_lookup[]": str(player_id),
        "player_type": "batter",
    }
    return await _fetch_many(_player_chunks(start, end), params, "batter", str(player_id))


async def statcast_pitcher(
    player_id: int,
    start: date,
    end: date,
) -> pl.DataFrame:
    params = {
        **_BASE_PARAMS,
        "pitchers_lookup[]": str(player_id),
        "player_type": "pitcher",
    }
    return await _fetch_many(_player_chunks(start, end), params, "pitcher", str(player_id))


async def statcast_season(
    player_id: int,
    year: int,
    player_type: Literal["batter", "pitcher"],
) -> pl.DataFrame:
    """Convenience: full season Mar 1 - Nov 30."""
    start = date(year, 3, 1)
    end = date(year, 11, 30)
    if player_type == "pitcher":
        return await statcast_pitcher(player_id, start, end)
    return await statcast_batter(player_id, start, end)


# ---------------------------------------------------------------------------
# Leaderboard endpoints (CSV one-shots, no chunking, no cache)
# ---------------------------------------------------------------------------


async def _csv_leaderboard(path: str, params: dict[str, str]) -> pl.DataFrame:
    resp = await savant_get(f"{_SAVANT_BASE}{path}", params)
    if not resp.content.strip():
        return pl.DataFrame()
    return pl.read_csv(BytesIO(resp.content), infer_schema_length=2000, ignore_errors=True)


def _filter_by_player_id(df: pl.DataFrame, player_id: int) -> pl.DataFrame:
    """Filter a leaderboard frame to one player when the id column exists."""
    if df.is_empty() or "player_id" not in df.columns:
        return df
    return df.filter(pl.col("player_id") == player_id)


async def batter_percentile_ranks(player_id: int, year: int) -> pl.DataFrame:
    """Percentile ranks for a single batter.

    Savant only exposes the full leaderboard CSV; we fetch all batters and
    filter to `player_id` (empty frame if the player isn't on the board).
    """
    df = await _csv_leaderboard(
        "/leaderboard/percentile-rankings",
        {"type": "batter", "year": str(year), "csv": "true"},
    )
    return _filter_by_player_id(df, player_id)


async def pitcher_percentile_ranks(player_id: int, year: int) -> pl.DataFrame:
    """Percentile ranks for a single pitcher (filtered from leaderboard)."""
    df = await _csv_leaderboard(
        "/leaderboard/percentile-rankings",
        {"type": "pitcher", "year": str(year), "csv": "true"},
    )
    return _filter_by_player_id(df, player_id)


async def pitcher_pitch_arsenal(player_id: int, year: int) -> pl.DataFrame:
    """Pitch arsenal for one pitcher (filtered from leaderboard)."""
    df = await _csv_leaderboard(
        "/leaderboard/pitch-arsenal-stats",
        {"type": "pitcher", "year": str(year), "min": "100", "csv": "true"},
    )
    return _filter_by_player_id(df, player_id)


async def batter_pitch_arsenal(year: int, min_pa: int = 25) -> pl.DataFrame:
    return await _csv_leaderboard(
        "/leaderboard/pitch-arsenal-stats",
        {"type": "batter", "year": str(year), "min": str(min_pa), "csv": "true"},
    )


async def batter_exit_velo(year: int, min_bbe: int = 25) -> pl.DataFrame:
    return await _csv_leaderboard(
        "/leaderboard/statcast",
        {"type": "batter", "year": str(year), "minBBE": str(min_bbe), "csv": "true"},
    )


async def pitcher_exit_velo(year: int, min_bbe: int = 25) -> pl.DataFrame:
    return await _csv_leaderboard(
        "/leaderboard/statcast",
        {"type": "pitcher", "year": str(year), "minBBE": str(min_bbe), "csv": "true"},
    )


async def batter_expected_stats(year: int, min_pa: int = 0) -> pl.DataFrame:
    return await _csv_leaderboard(
        "/leaderboard/expected_statistics",
        {"type": "batter", "year": str(year), "min": str(min_pa), "csv": "true"},
    )


async def pitcher_expected_stats(year: int, min_ip: int = 0) -> pl.DataFrame:
    return await _csv_leaderboard(
        "/leaderboard/expected_statistics",
        {"type": "pitcher", "year": str(year), "min": str(min_ip), "csv": "true"},
    )


async def sprint_speed(year: int, min_opp: int = 10) -> pl.DataFrame:
    """Sprint-speed leaderboard."""
    return await _csv_leaderboard(
        "/leaderboard/sprint_speed",
        {"year": str(year), "min_opp": str(min_opp), "csv": "true"},
    )


running_splits = sprint_speed  # deprecated alias — this always fetched sprint speed


async def pitcher_pitch_movement(year: int, min_p: int = 100) -> pl.DataFrame:
    """Horizontal/vertical movement leaderboard. Default pitch_type=FF."""
    return await _csv_leaderboard(
        "/leaderboard/pitch-movement",
        {
            "year": str(year),
            "team": "",
            "min": str(min_p),
            "pitch_type": "FF",
            "csv": "true",
        },
    )
