# Changelog

All notable changes to this project are documented here. This project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html); pre-1.0, minor
versions may break.

## [Unreleased]

## [0.2.0] — 2026-11-22

### Fixed

- **`spray_chart` and `stadium_outline` drew every park as the same generic
  shape.** They drew a smooth arc scaled to a park's three wall distances (LF,
  CF, RF), so Yankee Stadium, Fenway and Oracle Park all looked alike. They now
  draw each park's real outline: stands, wall, foul ground, infield dirt and
  grass, and foul lines. Batted balls are plotted at their raw `hc_x` / `hc_y`
  coordinates on the same grid, instead of being rescaled to feet. The LF, CF
  and RF distance labels are kept.

### Added

- `bigleague.stadium_paths`: the park outline data (`paths`, `known_parks`).
  It is Ben Dilday's GeomMLBStadiums data (MIT), as redistributed in
  pybaseball, bundled as a 96 KB file with its license notice. An unknown
  stadium key draws the generic park.

### Changed

- The `bloom` extra now requires `bouquet-bloom>=0.2.0` (was `>=0.1.0`).
  bigleague uses only `AtomicFilter.from_false_pos`, `insert`, `insert_many`,
  `in` and `clear`, which 0.2.0 keeps unchanged, so there are no code changes.
  0.2.0 adds batch, union and serialization methods and makes inserts and
  lookups faster. A cache lookup spends about 45 ns in the filter and about
  33 µs overall, so the speedup is not visible in bigleague.

## [0.1.1] — 2026-11-21

### Fixed

- **Player lookups no longer re-read the whole Chadwick register on every
  call.** `resolve_id`, `lookup`, `reverse_lookup` and `mlb_years` loaded all
  516,106 rows (about 22 MB) for each call and kept nothing, and a cold start
  fetched and parsed all 16 shards at once. A process doing only lookups
  reached 480 MB cold and 285 MB warm, and kept growing. The register is now
  loaded once per process (concurrent first calls share one load) and
  pruned to the 127,934 rows that carry an id or MLB years. A cold start
  streams one shard at a time in small blocks. Measured on macOS with Python
  3.13 (`bench/register_memory.py`, import baseline 94 MB): 480 → 197 MB cold,
  285 → 136 MB warm, 580 → 178 MB with `BIGLEAGUE_NO_CACHE=1`, flat across
  repeated lookups.
- **`pitcher_pitch_arsenal` returned the wrong Savant leaderboard.** It read
  the run-value board, which has no `avg_speed` or `avg_spin`, so
  `parse_pitch_arsenal(await pitcher_pitch_arsenal(...))` was always empty,
  and its `min=100` filter hid low-usage pitches. It now reads the
  `pitch-arsenals` board with a minimum of 10 pitches.

### Changed

- **Breaking:** `pitcher_pitch_arsenal(player_id, year, min_pitches=10)` now
  returns one row per pitch (`pitch_type`, `pitch_name`, `avg_speed`,
  `avg_spin`, `pitch_usage`) instead of the run-value rows. The run-value
  board is available as `pitcher_pitch_arsenal_stats`, with the same request
  and `min=100` default as before.
- The register cache is now `chadwick_people.v2.parquet`, holding only the
  pruned rows (about 1.4 MB instead of 4.1 MB). A cache written by 0.1.0 is
  converted on first use without downloading anything, and the old file is
  left in place. A corrupt cache is discarded and refetched.
- `BIGLEAGUE_NO_CACHE=1` still never reads or writes the register on disk,
  but the register is now kept in memory for the process instead of being
  downloaded again on every call.
- `lookup` no longer returns bare-name register rows that have no id and no
  MLB years. Namesake order is unchanged. People with only a FanGraphs or
  Baseball Reference id, or only MLB years, are still returned.

- **`import bigleague` no longer imports matplotlib or numpy.** They cost about
  40 MB of resident memory (94 MB → 55 MB after import, same on every run),
  which a process that never plots shouldn't pay. They are imported on the
  first plot. Plots are now built on `matplotlib.figure.Figure` instead of
  pyplot, so bigleague no longer calls `matplotlib.use("Agg")` at import
  (your backend is left alone) and there is no global figure registry. The
  PNG output is unchanged. The `gc.collect()` after each render stays: without
  it, 40 renders ended 39 MB higher.
- `player_stats`, `player_stats_for_year` and `career_stats` filter the
  FanGraphs league board down to the requested player before building a frame,
  instead of building a frame of every player and then filtering it. The
  endpoint returns the whole league (about 7 MB of JSON, 500+ columns per
  row). One `player_stats` call peaked 114 MB above its starting memory before
  and 72 MB after. `leaderboard`, `batting_stats` and the other public
  leaderboards still return the full frame.

### Added

- `pitcher_pitch_arsenal_stats`, the Savant run-value board for one pitcher.
- `bench/register_memory.py`, a resident-memory benchmark for player lookups.

## [0.1.0] — 2026-11-21

Initial public release.

- **Statcast.** Pitch-level data for a player, a team or the league
  (`statcast`, `statcast_batter`, `statcast_pitcher`, `statcast_season`),
  returned as polars DataFrames with a typed schema (`StatcastPitch`,
  `STATCAST_SCHEMA`). Requests run concurrently with a cap of 5 in flight per
  Baseball Savant.
- **Savant leaderboards.** Percentile ranks, pitch arsenal, exit velocity,
  expected stats, sprint speed and pitch movement, plus the Automated
  Ball-Strike challenge leaderboards (`bigleague.abs`).
- **FanGraphs.** Batting, pitching, fielding and team leaderboards, season and
  date-range queries, player and career stats, and playoff odds. Requests use
  `curl_cffi` Chrome impersonation; a Cloudflare block raises
  `FangraphsBlockedError`.
- **MLB Stats API.** `schedule` (full-season schedule with a running record),
  `live_scores`, `next_game`, `recent_results`, `standings`, `roster`,
  `injuries`, `transactions`.
- **Player lookup** against the Chadwick Bureau register, fetched from a pinned
  commit (`lookup`, `resolve_id`, `reverse_lookup`, `mlb_years`).
- **Plots.** `strike_zone`, `batter_zone`, `matchup_zone`, `hot_zones`,
  `spray_chart` and `stadium_outline`, each returning PNG bytes.
- **Parquet cache.** Statcast chunks are cached on disk, and chunks that include
  today's date are never cached. Configure it with `BIGLEAGUE_CACHE_DIR` and
  `BIGLEAGUE_NO_CACHE`. The optional `bloom` extra
  ([`bouquet-bloom`](https://pypi.org/project/bouquet-bloom/)) adds a
  thread-safe bloom filter in front of it.
