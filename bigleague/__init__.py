"""bigleague — async polars-native baseball data and plotting.

Public API surface. Submodules remain importable directly
(`bigleague.savant`, etc.) for code that prefers namespacing.

`mlb.players()` and `mlb.players_multi_season()` are intentionally NOT
re-exported at the top level — they would collide with the
`bigleague.players` submodule. Use `bigleague.mlb.players(...)` instead.
"""

from __future__ import annotations

from bigleague import abs
from bigleague.cache import StatcastCache, cache
from bigleague.fangraphs import (
    FangraphsBlockedError,
    batting_stats,
    batting_stats_range,
    career_stats,
    fielding_stats,
    leaderboard,
    pitching_stats,
    pitching_stats_range,
    player_stats,
    player_stats_for_year,
    playoff_odds,
    team_batting,
    team_pitching,
    top_leaderboard,
)
from bigleague.mlb import (
    POS_NAMES,
    TEAM_IDS,
    injuries,
    live_scores,
    next_game,
    recent_results,
    roster,
    standings,
    transactions,
)
from bigleague.models import StatcastPitch
from bigleague.players import lookup, mlb_years, resolve_id, reverse_lookup
from bigleague.plots import (
    batter_zone,
    hot_zones,
    matchup_zone,
    spray_chart,
    stadium_outline,
    strike_zone,
)
from bigleague.savant import (
    batter_exit_velo,
    batter_expected_stats,
    batter_percentile_ranks,
    batter_pitch_arsenal,
    pitcher_exit_velo,
    pitcher_expected_stats,
    pitcher_percentile_ranks,
    pitcher_pitch_arsenal,
    pitcher_pitch_movement,
    running_splits,
    statcast,
    statcast_batter,
    statcast_pitcher,
    statcast_season,
)
from bigleague.schema import STATCAST_SCHEMA
from bigleague.stadiums import (
    STADIUM_ALIASES,
    STADIUM_DATA,
    STADIUM_TO_TEAM_CODE,
    known_stadiums,
)
from bigleague.stadiums import metadata as stadium_metadata
from bigleague.stadiums import normalize as normalize_stadium
from bigleague.stadiums import team_code as stadium_team_code
from bigleague.stats import (
    AB_EVENTS,
    HIT_EVENTS,
    aggregate_career,
    hot_cold_batter,
    hot_cold_pitcher,
    hot_zones_grid,
    matchup_stats,
    parse_exit_velo,
    parse_expected_stats,
    parse_percentile_ranks,
    parse_pitch_arsenal,
    season_range,
)

__all__ = [
    "AB_EVENTS",
    "HIT_EVENTS",
    "POS_NAMES",
    "STADIUM_ALIASES",
    "STADIUM_DATA",
    "STADIUM_TO_TEAM_CODE",
    "STATCAST_SCHEMA",
    "TEAM_IDS",
    "FangraphsBlockedError",
    "StatcastCache",
    "StatcastPitch",
    "abs",
    "aggregate_career",
    "batter_exit_velo",
    "batter_expected_stats",
    "batter_percentile_ranks",
    "batter_pitch_arsenal",
    "batter_zone",
    "batting_stats",
    "batting_stats_range",
    "cache",
    "career_stats",
    "fielding_stats",
    "hot_cold_batter",
    "hot_cold_pitcher",
    "hot_zones",
    "hot_zones_grid",
    "injuries",
    "known_stadiums",
    "leaderboard",
    "live_scores",
    "lookup",
    "matchup_stats",
    "matchup_zone",
    "mlb_years",
    "next_game",
    "normalize_stadium",
    "parse_exit_velo",
    "parse_expected_stats",
    "parse_percentile_ranks",
    "parse_pitch_arsenal",
    "pitcher_exit_velo",
    "pitcher_expected_stats",
    "pitcher_percentile_ranks",
    "pitcher_pitch_arsenal",
    "pitcher_pitch_movement",
    "pitching_stats",
    "pitching_stats_range",
    "player_stats",
    "player_stats_for_year",
    "playoff_odds",
    "recent_results",
    "resolve_id",
    "reverse_lookup",
    "roster",
    "running_splits",
    "season_range",
    "spray_chart",
    "stadium_metadata",
    "stadium_outline",
    "stadium_team_code",
    "standings",
    "statcast",
    "statcast_batter",
    "statcast_pitcher",
    "statcast_season",
    "strike_zone",
    "team_batting",
    "team_pitching",
    "top_leaderboard",
    "transactions",
]
