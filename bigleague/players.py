"""Player ID resolution against the Chadwick Bureau register.

The register is a 2 MB CSV that changes infrequently — we cache it as a
parquet under the same `~/.bigleague/cache` tree the Statcast cache uses.
The first call downloads; subsequent calls read parquet from disk.
"""

from __future__ import annotations

import datetime as _dt
from io import BytesIO
from pathlib import Path

import polars as pl

from bigleague.http import get_client

_REGISTER_URL = (
    "https://github.com/chadwickbureau/register/raw/master/data/people.csv"
)
_REGISTER_PATH = Path("~/.bigleague/cache/chadwick_people.parquet").expanduser()

_REGISTER_COLS = (
    "name_last",
    "name_first",
    "key_mlbam",
    "key_fangraphs",
    "key_bbref",
    "mlb_played_first",
    "mlb_played_last",
)


async def _load_register() -> pl.DataFrame:
    if _REGISTER_PATH.exists():
        return pl.read_parquet(_REGISTER_PATH)

    client = await get_client()
    resp = await client.get(_REGISTER_URL)
    resp.raise_for_status()

    df = pl.read_csv(
        BytesIO(resp.content),
        columns=list(_REGISTER_COLS),
        infer_schema_length=10_000,
    )
    _REGISTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(_REGISTER_PATH)
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
