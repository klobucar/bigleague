"""Async ABS-challenge leaderboards from Baseball Savant.

The Automated Ball-Strike challenge system gives each team a fixed number
of challenges per game; the batter, pitcher, or catcher can request a
Hawk-Eye review of the umpire's ball/strike call. Savant publishes per-role
leaderboards (batter / pitcher / catcher) plus a team-summary feed.

Source: https://baseballsavant.mlb.com/leaderboard/abs-challenges
The CSV backing the leaderboard is public — pass `csv=true` and
`challengeType=<role>`. We fetch it through `bigleague.http.savant_get`
(shared client, 5-permit semaphore, retries, browser UA).

Quirks preserved here: Savant prepends a UTF-8 BOM to the CSV, and blank
cells must read as null rather than empty string so callers can tell 0
from missing. Schema is inferred from the CSV; `entity_name`/`team_abbr`
stay strings, the count/rate columns come back numeric.
"""

from __future__ import annotations

import asyncio
from io import BytesIO
from typing import Any

import polars as pl

from bigleague.http import savant_get

# The user-abbrev → Savant-code normalizer lives in bigleague.savant (both
# modules hit Baseball Savant and share the same code scheme); re-exported
# here so bigleague.abs.normalize_team stays a stable part of this surface.
from bigleague.savant import normalize_team

_SAVANT_BASE = "https://baseballsavant.mlb.com"
_ABS_PATH = "/leaderboard/abs-challenges"

# Roles accepted by the leaderboard's challengeType param.
ROLE_BATTER = "batter"
ROLE_PITCHER = "pitcher"
ROLE_CATCHER = "catcher"
ROLE_TEAM_SUMMARY = "team-summary"
PLAYER_ROLES: tuple[str, ...] = (ROLE_BATTER, ROLE_PITCHER, ROLE_CATCHER)


def _parse_csv(body: bytes) -> pl.DataFrame:
    """Parse an ABS leaderboard CSV into a polars frame.

    Strips Savant's leading UTF-8 BOM and lets polars infer the schema from
    the whole file. Empty cells parse as null (polars' default for unquoted
    missing fields), so callers can distinguish 0 from missing.
    """
    if body[:3] == b"\xef\xbb\xbf":
        body = body[3:]
    if not body.strip():
        return pl.DataFrame()
    return pl.read_csv(BytesIO(body), infer_schema_length=None)


async def challenges(
    year: int,
    role: str,
    *,
    game_type: str = "regular",
    level: str = "mlb",
) -> pl.DataFrame:
    """Raw ABS leaderboard frame for one (year, role).

    `role` is a challengeType value: "batter", "pitcher", "catcher", or
    "team-summary". Schema is inferred from the CSV.
    """
    params = {
        "year": str(year),
        "challengeType": role,
        "gameType": game_type,
        "level": level,
        "csv": "true",
    }
    resp = await savant_get(f"{_SAVANT_BASE}{_ABS_PATH}", params)
    return _parse_csv(resp.content)


async def _player_feeds(year: int) -> list[dict[str, Any]]:
    """Union the three player-role feeds, fetched concurrently.

    Each row is tagged with a "role" key. The three feeds go out together
    under the Savant semaphore rather than serially.
    """
    frames = await asyncio.gather(*(challenges(year, role) for role in PLAYER_ROLES))
    pool: list[dict[str, Any]] = []
    for role, frame in zip(PLAYER_ROLES, frames, strict=True):
        pool.extend({**row, "role": role} for row in frame.to_dicts())
    return pool


async def team_summary(team: str, year: int) -> dict[str, Any]:
    """The team-summary row for one team. Raises ValueError if absent."""
    sav = normalize_team(team)
    df = await challenges(year, ROLE_TEAM_SUMMARY)
    for row in df.to_dicts():
        if row.get("team_abbr") == sav:
            return row
    raise ValueError(f"No ABS data for {team!r} in {year}.")


async def team_top_challengers(team: str, year: int, limit: int = 3) -> list[dict[str, Any]]:
    """A team's top individual challengers by raw challenge count.

    Unions the batter/pitcher/catcher feeds, filters to the team, dedupes by
    `entity_name` (keeping the row with the most `n_challenges`, so a two-way
    player who challenged from both sides appears once), attaches the "role"
    label, sorts by `n_challenges` descending, and takes the top `limit`.
    """
    sav = normalize_team(team)
    by_name: dict[str, dict[str, Any]] = {}
    for r in await _player_feeds(year):
        if r.get("team_abbr") != sav:
            continue
        name = r.get("entity_name")
        if not isinstance(name, str):
            continue
        existing = by_name.get(name)
        if existing is None or (r.get("n_challenges") or 0) > (existing.get("n_challenges") or 0):
            by_name[name] = r

    ranked = sorted(by_name.values(), key=lambda r: r.get("n_challenges") or 0, reverse=True)
    return ranked[:limit]


async def player(first: str, last: str, year: int) -> dict[str, Any]:
    """Merged ABS challenge stats for one player across all three role feeds.

    Matches case-insensitively on `entity_name`. Sums the count columns into
    a `totals` dict and keeps the matched per-role rows under `roles`.
    Raises ValueError if the player is in none of the feeds.
    """
    target = f"{first.strip()} {last.strip()}".lower()
    matches = [
        r
        for r in await _player_feeds(year)
        if isinstance(r.get("entity_name"), str) and r["entity_name"].lower() == target
    ]
    if not matches:
        raise ValueError(f"No ABS challenge data for {first} {last} in {year}.")

    totals = {
        "n_challenges": sum((m.get("n_challenges") or 0) for m in matches),
        "n_overturns": sum((m.get("n_overturns") or 0) for m in matches),
        "n_confirms": sum((m.get("n_confirms") or 0) for m in matches),
        "n_strikeouts_flip": sum((m.get("n_strikeouts_flip") or 0) for m in matches),
        "n_walks_flip": sum((m.get("n_walks_flip") or 0) for m in matches),
    }
    return {
        "name": matches[0].get("entity_name") or f"{first} {last}",
        "team_abbr": matches[0].get("team_abbr") or "",
        "totals": totals,
        "roles": matches,
    }


async def league_leaders(
    year: int,
    limit: int = 10,
    min_challenges: int = 3,
) -> list[dict[str, Any]]:
    """League-wide leaders by overturn rate, with a min-challenge floor.

    Unions the three player feeds, drops rows with fewer than
    `min_challenges` (otherwise a 1-for-1 player sits at a 100% rate), sorts
    by (`rate_overturns`, `n_challenges`) descending, and takes the top
    `limit`. Each row carries its "role".
    """
    pool = [r for r in await _player_feeds(year) if (r.get("n_challenges") or 0) >= min_challenges]
    pool.sort(
        key=lambda r: (r.get("rate_overturns") or 0, r.get("n_challenges") or 0),
        reverse=True,
    )
    return pool[:limit]
