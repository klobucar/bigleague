# bigleague

Async, polars-native baseball data and plotting library: Statcast, FanGraphs,
the MLB Stats API and Baseball Savant leaderboards, plus matplotlib renderers
for strike zones, hot zones, spray charts and stadium outlines. No pybaseball
and no pandas.

It powers [harry-bot](https://github.com/klobucar/harry-bot), a Discord
baseball bot.

## Why

- **Polars everywhere.** No pandas. Smaller memory footprint, faster joins,
  cleaner null semantics.
- **Async-first.** httpx + asyncio, with a 5-permit semaphore on Baseball
  Savant fetches (their effective rate limit).
- **No pybaseball.** Strike zones, hot-zone heatmaps, and stadium outlines
  are drawn from scratch with matplotlib patches. No CSV scraping.
- **curl_cffi for FanGraphs.** Cloudflare has gated fangraphs.com to plain
  TLS clients since mid-2025; Chrome impersonation is load-bearing.
- **Parquet cache, optional bloom accelerator.** Parquet on disk. With the
  optional `bloom` extra installed, a bloom filter adds a sub-microsecond
  negative-lookup shortcut; without it the cache works the same, just without
  that shortcut. The right edge of the cache (chunks containing today's date)
  is intentionally NOT cached to avoid serving stale partial data.

## Benchmarks

Same Ohtani (660271) Statcast pulls against real Savant, each library in
its own subprocess for clean RSS:

| lib            | scenario   |  rows |  frame |  peak heap |  peak RSS |  wall |
| -------------- | ---------- | ----: | -----: | ---------: | --------: | ----: |
| bigleague      | month cold |   426 | 194 KB |      52 MB |    161 MB | 1.35s |
| pybaseball     | month cold |   426 | 433 KB |      85 MB |    260 MB | 2.69s |
| pandas+pyarrow | month cold |   426 | 375 KB |      48 MB |    167 MB | 1.27s |
| bigleague      | month warm |   426 | 193 KB |      48 MB |    151 MB | 1.20s |
| pybaseball     | month warm |   426 | 433 KB |      85 MB |    260 MB | 2.60s |
| pandas+pyarrow | month warm |   426 | 375 KB |      48 MB |    166 MB | 1.27s |
| bigleague      | season     |  2859 | 1.3 MB |      55 MB |    168 MB | 1.40s |
| pybaseball     | season     |  2850 | 2.8 MB |      96 MB |    281 MB | 2.71s |
| pandas+pyarrow | season     |  2859 | 2.5 MB |      51 MB |    176 MB | 1.29s |
| bigleague      | hotzones   |   426 | 193 KB |      48 MB |    158 MB | 1.23s |
| pybaseball     | hotzones   |   426 | 433 KB |      99 MB |    290 MB | 2.93s |
| pandas+pyarrow | hotzones   |   426 | 375 KB |      80 MB |    250 MB | 2.21s |

Takeaways:

- **In-memory frames are ~2× smaller** than pybaseball and ~1.5–2× smaller
  than pandas-with-pyarrow-backend across every scenario — the polars-arrow
  layout win.
- **Peak heap and RSS** track frame size: bigleague and pandas+pyarrow are
  comparable (Arrow under both); pybaseball is ~1.7× heavier on heap and
  ~70% heavier on RSS thanks to numpy-backed pandas.
- **Warm reads** drop from 1.35s to 1.20s on the same window via the
  parquet cache. pybaseball and pandas+pyarrow have no cache, so their
  "warm" timing is just Savant CDN noise.
- **Player-scoped queries** issue one request, not one-per-7-days
  (`_player_chunks`), so cold season pulls take ~1.4s instead of ~6s.
  League-wide `statcast()` keeps 7-day chunking — one day league-wide
  blows past Savant's ~25k row CSV cap.

Reproduce: `python bench/compare_pybaseball.py`. Network required; not run
in CI. Install bench deps: `uv pip install pybaseball pandas pyarrow`.

## Install

```bash
pip install bigleague
# or
uv add bigleague
```

Requires Python ≥ 3.12. Tested on 3.14.

To work on bigleague itself:

```bash
uv sync --extra dev
```

The `bloom` extra (the `bouquet` filter) is an **optional** runtime
accelerator, not a dev requirement — the suite skips bloom-only tests when
it's absent. To exercise that path locally:

```bash
uv sync --extra dev --extra bloom
```

## Quick start

```python
import asyncio
from datetime import date

import bigleague as bl


async def main() -> None:
    df = await bl.statcast_batter(660271, date(2025, 4, 1), date(2025, 4, 30))
    png = bl.strike_zone(df, "Shohei Ohtani — April 2025")
    stats = bl.hot_cold_batter(df)
    print(stats)
    # {pa: 87, ab: 76, h: 22, avg: 0.289, obp: 0.356, slg: 0.526, ops: 0.882}


asyncio.run(main())
```

### FanGraphs

```python
stats = await bl.player_stats("Shohei", "Ohtani", 2025)
leaders = await bl.top_leaderboard("WAR", 2025, n=10)
try:
    stats = await bl.player_stats("Gerrit", "Cole", 2025)
except bl.FangraphsBlockedError:
    pass  # transient Cloudflare block — retry later
```

### MLB API

```python
games = await bl.live_scores("NYY")
game = await bl.next_game("NYY")
# game = {date, time, home_team, away_team, home_probable, away_probable}

season = await bl.schedule("NYY", 2025)
# one row per game, in play order:
# {date, start_time, game_pk, is_home, opponent, status, score, record}
# status is "W" / "L" / "P" (postponed) / "S" (scheduled); record is the
# running "W-L" after that game.
last5 = [g for g in season if g["status"] in ("W", "L")][-5:]
```

## Cache

| Aspect | Behavior |
| --- | --- |
| Backend | parquet on disk; optional bloom-filter accelerator (`bloom` extra) |
| Default location | `~/.bigleague/cache` |
| Override | `BIGLEAGUE_CACHE_DIR=/path` |
| Disable | `BIGLEAGUE_NO_CACHE=1` |
| Right-edge guard | chunks where `end_date >= today` are NOT cached |
| Eviction | none. Use `bl.cache.clear()` and `bl.cache.size()` to manage manually. |

The right-edge guard means a fresh request for today's data always re-fetches
until the chunk fully ages into the past. If you want to override this for a
backfill, set `BIGLEAGUE_NO_CACHE=1` for the run.

## Plots

All renderers return PNG `bytes`. No file I/O. `plt.close(fig)` and
`gc.collect()` run in a `finally` block on every render to keep matplotlib
from leaking C-side bitmaps under sustained load.

```python
png = bl.strike_zone(df, title)
png = bl.batter_zone(df, title)
png = bl.matchup_zone(df, title)
png = bl.hot_zones(df, title)
png = bl.spray_chart(df, title, stadium_key="yankees")
png = bl.stadium_outline("yankees")
```

## Stadium normalization

```python
bl.normalize_stadium("Guardians")  # -> "indians"  (Statcast key)
bl.normalize_stadium("Wizards")  # -> "generic"
bl.stadium_metadata("yankees")  # -> {"name": "Yankee Stadium", ...}
```

## ABS challenge data

`bigleague.abs` wraps Baseball Savant's Automated Ball-Strike challenge
leaderboards (batter / pitcher / catcher roles plus a team summary feed).
Public functions: `challenges(year, role)` returns the raw leaderboard
frame, and `team_summary`, `team_top_challengers`, `player`, and
`league_leaders` answer the common questions on top of it.

```python
row = await bl.abs.team_summary("NYY", 2025)
leaders = await bl.abs.league_leaders(2025, limit=10)
```

## Tests

```bash
uv run pytest tests/
```

`BIGLEAGUE_NO_CACHE=1` is set automatically by `tests/conftest.py`. All
HTTP is mocked via `respx` (httpx) or monkey-patched (`curl_cffi`).

A VCR layer (`pytest-recording`) is wired up but no cassettes are checked in.
To record one against a live endpoint:

```bash
uv run pytest tests/test_savant.py --record-mode=once
```

Cassettes go under `tests/cassettes/`.

## Lint + type-check

```bash
uv run ruff check bigleague/ tests/
uv run ty check bigleague/ tests/
```

## Architecture

```
bigleague/
├── models.py     # StatcastPitch dataclass — schema source of truth
├── schema.py     # polars dtype map derived from models
├── http.py       # shared async client, semaphore, retries
├── cache.py      # parquet (+ optional bloom)
├── stadiums.py   # stadium data + normalization
├── stats.py      # pure polars computation (no I/O)
├── players.py    # Chadwick Bureau lookup
├── mlb.py        # MLB Stats API
├── savant.py     # Baseball Savant — pitch chunks + leaderboards
├── abs.py        # Baseball Savant — ABS challenge leaderboards
├── fangraphs.py  # FanGraphs JSON API (curl_cffi)
└── plots.py      # matplotlib renderers
```
