"""Player ID resolution against the Chadwick Bureau register.

The register ships as 16 CSV shards (about 516,000 people, roughly 22 MB once
loaded). Almost all of those rows are name-only entries that carry no id and
no MLB debut year, so no function here could use them. We keep only the rows
that have at least one of `key_mlbam`, `key_fangraphs`, `key_bbref` or an MLB
played year: about 128,000 rows, roughly 6 MB.

The pruned register is loaded once per process and shared by every lookup, so
`resolve_id`, `lookup`, `reverse_lookup` and `mlb_years` don't re-read or
re-fetch anything. Concurrent first calls share a single load.

On disk the pruned register is cached as a parquet under the same cache tree
the Statcast cache uses (resolved via `cache.resolve_cache_dir`, so
`BIGLEAGUE_CACHE_DIR` configures it identically). The first call in a fresh
cache downloads the shards one at a time, streaming each and pruning it in
small blocks so a whole shard is never held in memory, then writes the
parquet; later processes read it back. A cache written by bigleague 0.1.0
(the full, unpruned register under another filename) is pruned and rewritten
in the new format, with no download.

`BIGLEAGUE_NO_CACHE` means nothing is read from or written to disk. The
register is still held in memory for the life of the process: it is pinned to
a commit, so refetching it per call would only cost memory and time.
"""

from __future__ import annotations

import asyncio
import contextlib
import datetime as _dt
import os
from io import BytesIO
from pathlib import Path

import polars as pl

from bigleague.cache import atomic_write_parquet, resolve_cache_dir
from bigleague.http import get_client

# Chadwick no longer ships a single people.csv — the register is split into
# 16 shards (people-0.csv … people-f.csv) keyed by the first hex digit of the
# person UUID, about 4 MB of CSV each. We fetch them one at a time, streaming
# the response and parsing it in small blocks, so memory never holds a whole
# raw shard (let alone all 16).
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

# Size of the newline-aligned blocks a shard is parsed in. Measured: parsing a
# shard whole peaks about 85 MB higher than parsing it in blocks, and block
# sizes from 64 KB to 1 MB perform the same.
_BLOCK_BYTES = 256 * 1024

# The pruned register. A new filename (not the 0.1.0 one) so an old, unpruned
# parquet can never be mistaken for a pruned one.
_REGISTER_FILENAME = "chadwick_people.v2.parquet"
# What bigleague 0.1.0 wrote: the full register, all 516k rows.
_LEGACY_REGISTER_FILENAME = "chadwick_people.parquet"

_REGISTER_COLS = (
    "name_last",
    "name_first",
    "key_mlbam",
    "key_fangraphs",
    "key_bbref",
    "mlb_played_first",
    "mlb_played_last",
)

# Columns that identify a person to the outside world. A row with none of
# these is a bare name: `lookup` would return it with every other field null,
# and nothing else in bigleague can use it, so it is dropped.
_USEFUL_COLS = (
    "key_mlbam",
    "key_fangraphs",
    "key_bbref",
    "mlb_played_first",
    "mlb_played_last",
)

# Explicit dtypes so a header-only or all-null shard doesn't infer a narrower
# type than its peers, and `lookup` returns stable dtypes.
_REGISTER_SCHEMA_OVERRIDES: dict[str, pl.DataType | type[pl.DataType]] = {
    "key_mlbam": pl.Int64,
    "key_fangraphs": pl.Int64,
    "mlb_played_first": pl.Int64,
    "mlb_played_last": pl.Int64,
}

# Process-wide state: the loaded register, and the in-flight load (with the
# event loop it belongs to) so concurrent first calls share one load.
_register: pl.DataFrame | None = None
_register_inflight: tuple[asyncio.AbstractEventLoop, asyncio.Task[pl.DataFrame]] | None = None


def _register_path() -> Path:
    """Where the cached (pruned) register parquet lives, resolved at call time."""
    return resolve_cache_dir() / _REGISTER_FILENAME


def _legacy_register_path() -> Path:
    """Where bigleague 0.1.0 cached the full, unpruned register."""
    return resolve_cache_dir() / _LEGACY_REGISTER_FILENAME


def _keep_expr() -> pl.Expr:
    """True for rows worth keeping: at least one id or MLB-played year."""
    return pl.any_horizontal(*(pl.col(c).is_not_null() for c in _USEFUL_COLS))


def _parse_block(header: bytes, body: bytes) -> pl.DataFrame:
    """Parse a block of CSV rows (under the shard's header) and prune it."""
    df = pl.read_csv(
        BytesIO(header + body),
        columns=list(_REGISTER_COLS),
        schema_overrides=_REGISTER_SCHEMA_OVERRIDES,
        infer_schema_length=10_000,
    )
    return df.filter(_keep_expr())


def _cut_point(buf: bytes | bytearray) -> int:
    """Index just past the last newline that ends a complete CSV record.

    A newline inside a quoted field doesn't end a record, so a cut is only
    valid where everything before it has an even number of quote characters
    (an escaped quote is a doubled one, which keeps the parity). Returns 0 if
    there is no such newline in `buf` yet.
    """
    end = len(buf)
    while (i := buf.rfind(b"\n", 0, end)) >= 0:
        if buf.count(b'"', 0, i) % 2 == 0:
            return i + 1
        end = i
    return 0


async def _fetch_shard(shard: str) -> pl.DataFrame:
    """Stream one shard and return only its kept rows.

    The response is parsed in `_BLOCK_BYTES` blocks as it arrives, so neither
    the whole raw shard nor its parse buffers are ever alive at once. Parsing
    is blocking CPU work, so each block goes to a worker thread.
    """
    client = await get_client()
    header: bytes | None = None
    buf = bytearray()
    frames: list[pl.DataFrame] = []
    async with client.stream("GET", f"{_REGISTER_BASE}/{shard}") as resp:
        resp.raise_for_status()
        async for chunk in resp.aiter_bytes():
            buf += chunk
            if header is None:
                nl = buf.find(b"\n")
                if nl < 0:
                    continue
                header = bytes(buf[: nl + 1])
                del buf[: nl + 1]
            if len(buf) >= _BLOCK_BYTES and (cut := _cut_point(buf)):
                body = bytes(buf[:cut])
                del buf[:cut]
                frames.append(await asyncio.to_thread(_parse_block, header, body))
    if header is None:
        raise ValueError(f"Chadwick register shard {shard} was empty.")
    if buf.strip() or not frames:
        # The tail, or (for a header-only shard) an empty block so the frame
        # still carries the right schema.
        frames.append(await asyncio.to_thread(_parse_block, header, bytes(buf)))
    return pl.concat(frames, how="vertical_relaxed")


async def _fetch_register() -> pl.DataFrame:
    """Download the register shards one at a time, pruning as we go.

    Shard order is preserved, so rows keep the order they had in the full
    register. `vertical_relaxed` reconciles any residual per-shard dtype
    differences.
    """
    frames = [await _fetch_shard(s) for s in _REGISTER_SHARDS]
    return pl.concat(frames, how="vertical_relaxed").rechunk()


def _write_register_sync(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_parquet(df, path)


def _read_cached_register() -> pl.DataFrame | None:
    """The register from disk, or None if there is no usable cache.

    Reads the pruned parquet when present. Otherwise, if a bigleague 0.1.0
    cache (the full register) exists, prunes it with a filtered scan, so only
    the kept rows are ever materialized, and rewrites it in the new format.
    An unreadable file is treated as a miss rather than an error, like the
    Statcast cache.
    """
    path = _register_path()
    if path.exists():
        try:
            return pl.read_parquet(path)
        except (pl.exceptions.PolarsError, OSError):
            path.unlink(missing_ok=True)
            return None

    legacy = _legacy_register_path()
    if legacy.exists():
        try:
            df = (
                pl.scan_parquet(legacy)
                .select(list(_REGISTER_COLS))
                .filter(_keep_expr())
                .collect()
                .rechunk()
            )
        except (pl.exceptions.PolarsError, OSError):
            return None
        # A read-only cache dir is fine: we still have the data in memory.
        with contextlib.suppress(OSError):
            _write_register_sync(df, path)
        return df
    return None


async def _build_register(*, no_cache: bool) -> pl.DataFrame:
    """Produce the register (disk cache or network) and memoize it."""
    global _register
    path = _register_path()
    df: pl.DataFrame | None = None
    if not no_cache:
        # Parquet reads/writes are blocking multi-MB I/O: run them in a
        # thread so they don't stall the event loop.
        df = await asyncio.to_thread(_read_cached_register)
    if df is None:
        df = await _fetch_register()
        if not no_cache:
            await asyncio.to_thread(_write_register_sync, df, path)
    _register = df
    return df


def _on_register_done(task: asyncio.Task[pl.DataFrame]) -> None:
    """Forget a finished load so a failed one can be retried by the next call."""
    global _register_inflight
    if _register_inflight is not None and _register_inflight[1] is task:
        _register_inflight = None
    if not task.cancelled():
        task.exception()  # mark retrieved; awaiting callers still see it


async def _load_register() -> pl.DataFrame:
    """The pruned register, loaded at most once per process.

    Concurrent first calls await the same task, so the register is read or
    fetched once, not once per caller. The load runs shielded: one caller
    being cancelled doesn't abort it for the others.
    """
    global _register_inflight
    if _register is not None:
        return _register

    loop = asyncio.get_running_loop()
    inflight = _register_inflight
    if inflight is None or inflight[0] is not loop:
        # No load in progress, or one that belongs to a different (likely
        # closed) event loop: start a fresh one on this loop. BIGLEAGUE_NO_CACHE
        # is read here, per load, not frozen at import.
        no_cache = bool(os.environ.get("BIGLEAGUE_NO_CACHE"))
        task = loop.create_task(_build_register(no_cache=no_cache))
        task.add_done_callback(_on_register_done)
        _register_inflight = (loop, task)
    else:
        task = inflight[1]
    return await asyncio.shield(task)


def _forget_register() -> None:
    """Drop the in-memory register (and any in-flight load). Used by tests."""
    global _register, _register_inflight
    _register = None
    _register_inflight = None


async def lookup(
    last: str,
    first: str | None = None,
    fuzzy: bool = True,
) -> pl.DataFrame:
    """Look up players by last (and optional first) name.

    Case-insensitive. With `fuzzy=True`, falls back to a startswith match
    on first name if the exact match returns nothing. Searches the pruned
    register, so a name with no id and no MLB years is not returned.
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
