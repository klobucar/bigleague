"""StatcastPitch dataclass — column source of truth for the Statcast CSV schema.

Field list and types derived from a real Baseball Savant CSV sample
(2024-04-01..02). 118 columns total. Types are conservative: anything
that has been observed null in the wild is `T | None`. The schema in
`bigleague.schema` is generated mechanically from these annotations, so
this is the only place column metadata lives.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import date
from typing import Any


@dataclass(slots=True)
class StatcastPitch:
    pitch_type: str | None
    game_date: date
    release_speed: float | None
    release_pos_x: float | None
    release_pos_z: float | None
    player_name: str | None
    batter: int
    pitcher: int
    events: str | None
    description: str | None
    spin_dir: str | None
    spin_rate_deprecated: str | None
    break_angle_deprecated: str | None
    break_length_deprecated: str | None
    zone: int | None
    des: str | None
    game_type: str | None
    stand: str | None
    p_throws: str | None
    home_team: str
    away_team: str
    type: str | None
    hit_location: int | None
    bb_type: str | None
    balls: int
    strikes: int
    game_year: int
    pfx_x: float | None
    pfx_z: float | None
    plate_x: float | None
    plate_z: float | None
    on_3b: int | None
    on_2b: int | None
    on_1b: int | None
    outs_when_up: int
    inning: int
    inning_topbot: str | None
    hc_x: float | None
    hc_y: float | None
    tfs_deprecated: str | None
    tfs_zulu_deprecated: str | None
    umpire: str | None
    sv_id: str | None
    vx0: float | None
    vy0: float | None
    vz0: float | None
    ax: float | None
    ay: float | None
    az: float | None
    sz_top: float | None
    sz_bot: float | None
    hit_distance_sc: float | None
    launch_speed: float | None
    launch_angle: float | None
    effective_speed: float | None
    release_spin_rate: float | None
    release_extension: float | None
    game_pk: int
    fielder_2: int | None
    fielder_3: int | None
    fielder_4: int | None
    fielder_5: int | None
    fielder_6: int | None
    fielder_7: int | None
    fielder_8: int | None
    fielder_9: int | None
    release_pos_y: float | None
    estimated_ba_using_speedangle: float | None
    estimated_woba_using_speedangle: float | None
    woba_value: float | None
    woba_denom: float | None
    babip_value: float | None
    iso_value: float | None
    launch_speed_angle: int | None
    at_bat_number: int
    pitch_number: int
    pitch_name: str | None
    home_score: int
    away_score: int
    bat_score: int
    fld_score: int
    post_away_score: int
    post_home_score: int
    post_bat_score: int
    post_fld_score: int
    if_fielding_alignment: str | None
    of_fielding_alignment: str | None
    spin_axis: int | None
    delta_home_win_exp: float | None
    delta_run_exp: float | None
    bat_speed: float | None
    swing_length: float | None
    estimated_slg_using_speedangle: float | None
    delta_pitcher_run_exp: float | None
    hyper_speed: float | None
    home_score_diff: int | None
    bat_score_diff: int | None
    home_win_exp: float | None
    bat_win_exp: float | None
    age_pit_legacy: int | None
    age_bat_legacy: int | None
    age_pit: int | None
    age_bat: int | None
    n_thruorder_pitcher: int | None
    n_priorpa_thisgame_player_at_bat: int | None
    pitcher_days_since_prev_game: int | None
    batter_days_since_prev_game: int | None
    pitcher_days_until_next_game: int | None
    batter_days_until_next_game: int | None
    api_break_z_with_gravity: float | None
    api_break_x_arm: float | None
    api_break_x_batter_in: float | None
    arm_angle: float | None
    attack_angle: float | None
    attack_direction: float | None
    swing_path_tilt: float | None
    intercept_ball_minus_batter_pos_x_inches: float | None
    intercept_ball_minus_batter_pos_y_inches: float | None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> StatcastPitch:
        kwargs: dict[str, Any] = {f.name: row.get(f.name) for f in fields(cls)}
        return cls(**kwargs)
