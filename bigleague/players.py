"""Player ID resolution against the Chadwick Bureau register.

The register is a 2 MB CSV that changes infrequently — we cache it as a
parquet under the same cache tree the Statcast cache uses (resolved via
`cache.resolve_cache_dir`, so `BIGLEAGUE_CACHE_DIR` / `BIGLEAGUE_NO_CACHE`
configure it identically). The first call downloads; subsequent calls read
parquet from disk unless caching is disabled.
"""

from __future__ import annotations

import asyncio
import datetime as _dt
import os
from io import BytesIO
from pathlib import Path

import polars as pl

from bigleague.cache import resolve_cache_dir
from bigleague.http import get_client

# Chadwick no longer ships a single people.csv — the register is split into
# 16 shards (people-0.csv … people-f.csv) keyed by the first hex digit of the
# person UUID. We fetch them concurrently and concat into one frame.
#
# Hit raw.githubusercontent.com directly: the github.com/.../raw/... path 302s
# here, and our shared httpx client doesn't follow redirects.
#
# Pin to a specific commit rather than `master` for supply-chain integrity:
# a git commit SHA is content-addressed, so the bytes we fetch can't change
# under us (or be swapped by a compromised branch). Bump deliberately to pick
# up newly registered players.
_REGISTER_COMMIT = "6e9db4bfeed5de99cb338a248c00030f53e98344"  # 2026-06-01
_REGISTER_BASE = (
    f"https://raw.githubusercontent.com/chadwickbureau/register/{_REGISTER_COMMIT}/data"
)
_REGISTER_SHARDS = tuple(f"people-{c}.csv" for c in "0123456789abcdef")
_REGISTER_FILENAME = "chadwick_people.parquet"

_REGISTER_COLS = (
    "name_last",
    "name_first",
    "key_mlbam",
    "key_fangraphs",
    "key_bbref",
    "mlb_played_first",
    "mlb_played_last",
)


def _register_path() -> Path:
    """Where the cached register parquet lives, resolved at call time."""
    return resolve_cache_dir() / _REGISTER_FILENAME


async def _fetch_shard(shard: str) -> pl.DataFrame:
    client = await get_client()
    resp = await client.get(f"{_REGISTER_BASE}/{shard}")
    resp.raise_for_status()
    return pl.read_csv(
        BytesIO(resp.content),
        columns=list(_REGISTER_COLS),
        infer_schema_length=10_000,
    )


async def _fetch_register() -> pl.DataFrame:
    """Download all register shards concurrently and concat them.

    `vertical_relaxed` reconciles per-shard dtype inference (e.g. a shard
    where every key_mlbam is null infers a narrower type than its peers).
    """
    frames = await asyncio.gather(*(_fetch_shard(s) for s in _REGISTER_SHARDS))
    return pl.concat(frames, how="vertical_relaxed")


def _write_register_sync(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Temp-file + atomic rename so a crash mid-write can't leave a
    # truncated parquet that breaks every subsequent load.
    tmp = path.with_suffix(".parquet.tmp")
    df.write_parquet(tmp)
    tmp.replace(path)


async def _load_register() -> pl.DataFrame:
    # Honor BIGLEAGUE_NO_CACHE like StatcastCache: never touch disk, always
    # fetch fresh. Parquet reads/writes are blocking multi-MB I/O — run
    # them in a thread so they don't stall the event loop.
    no_cache = bool(os.environ.get("BIGLEAGUE_NO_CACHE"))
    path = _register_path()
    if not no_cache and path.exists():
        return await asyncio.to_thread(pl.read_parquet, path)

    df = await _fetch_register()
    if not no_cache:
        await asyncio.to_thread(_write_register_sync, df, path)
    return df


async def lookup(
    last: str,
    first: str | None = None,
    fuzzy: bool = True,
) -> pl.DataFrame:
    """Look up players by last (and optional first) name.

    Case-insensitive. With `fuzzy=True`, falls back to a startswith match
    on first name if the exact match returns nothing.
    """
    df = await _load_register()
    last_lc = last.lower()
    df = df.filter(pl.col("name_last").str.to_lowercase() == last_lc)
    if first is None:
        return df

    first_lc = first.lower()
    exact = df.filter(pl.col("name_first").str.to_lowercase() == first_lc)
    if not exact.is_empty() or not fuzzy:
        return exact
    return df.filter(pl.col("name_first").str.to_lowercase().str.starts_with(first_lc))


async def resolve_id(first: str, last: str) -> int | None:
    """Return MLBAM ID for a player or None if unknown / no MLBAM mapping."""
    res = await lookup(last, first, fuzzy=True)
    res = res.filter(pl.col("key_mlbam").is_not_null())
    if res.is_empty():
        return None
    return int(res["key_mlbam"][0])


async def reverse_lookup(player_ids: list[int]) -> pl.DataFrame:
    """Return register rows for the given MLBAM IDs."""
    df = await _load_register()
    return df.filter(pl.col("key_mlbam").is_in(player_ids))


async def mlb_years(first: str, last: str) -> tuple[int, int]:
    """Return (debut_year, last_active_year) for a player.

    Falls back to (current_year - 3, current_year) if the register has no
    MLB-played dates for this name. For active players whose
    `mlb_played_last` lags the current calendar year, we extend the
    upper bound to today's year so callers don't miss this season.
    """
    cy = _dt.date.today().year
    res = await lookup(last, first, fuzzy=True)
    if res.is_empty():
        return (cy - 3, cy)

    row = res.row(0, named=True)
    debut = row.get("mlb_played_first")
    last_yr = row.get("mlb_played_last")

    debut_int = int(debut) if debut is not None else (cy - 3)
    last_int = int(last_yr) if last_yr is not None else cy
    return (debut_int, max(last_int, cy))
