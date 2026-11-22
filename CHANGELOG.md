# Changelog

All notable changes to this project are documented here. This project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html); pre-1.0, minor
versions may break.

## [Unreleased]

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
