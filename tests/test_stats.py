from __future__ import annotations

from datetime import date

import polars as pl
import pytest
from bigleague import stats


def _make_batter_df(events: list[str | None], pitchers: list[int] | None = None) -> pl.DataFrame:
    n = len(events)
    return pl.DataFrame(
        {
            "events": events,
            "pitcher": pitchers if pitchers is not None else [1] * n,
        }
    )


def test_season_range_returns_mar1_nov30() -> None:
    start, end = stats.season_range(2024)
    assert start == date(2024, 3, 1)
    assert end == date(2024, 11, 30)


def test_ab_events_contain_hit_events() -> None:
    assert stats.HIT_EVENTS <= stats.AB_EVENTS


def test_matchup_stats_filters_by_pitcher() -> None:
    df = _make_batter_df(
        ["single", "field_out", "home_run", "strikeout"],
        pitchers=[10, 10, 20, 10],
    )
    out = stats.matchup_stats(df, 10)
    assert out["pa"] == 3
    assert out["ab"] == 3
    assert out["hits"] == 1
    assert out["strikeouts"] == 1
    assert out["batting_avg"] == pytest.approx(1 / 3)


def test_matchup_stats_raises_for_missing_pitcher() -> None:
    df = _make_batter_df(["single"], [10])
    with pytest.raises(ValueError, match="No matchup rows"):
        stats.matchup_stats(df, 99)


def test_hot_cold_batter_basic() -> None:
    df = _make_batter_df(
        ["single", "double", "home_run", "field_out", "strikeout", "walk", "walk"]
    )
    out = stats.hot_cold_batter(df)
    # 5 ABs (single, double, hr, out, k), 3 H, 2 BB, PA=7
    assert out["ab"] == 5
    assert out["h"] == 3
    assert out["bb"] == 2
    assert out["pa"] == 7
    # AVG = 3/5 = .600, OBP = 5/7 ≈ .714, SLG = (1+2+4)/5 = 1.400
    assert out["avg"] == pytest.approx(0.600)
    assert out["slg"] == pytest.approx(1.400)


def test_hot_cold_pitcher_counts_events() -> None:
    df = _make_batter_df(
        ["strikeout", "strikeout_double_play", "walk", "home_run", "single", None]
    )
    out = stats.hot_cold_pitcher(df)
    assert out["pitches"] == 6
    assert out["k"] == 2
    assert out["bb"] == 1
    assert out["hr_allowed"] == 1
    assert out["pa_faced"] == 5  # 5 non-null events


def test_hot_zones_grid_returns_keys_1_through_9() -> None:
    df = pl.DataFrame(
        {
            "zone": [1, 1, 5, 5, 5, 9, None, 12],
            "events": [
                "single", "field_out",
                "home_run", "double", "field_out",
                "strikeout", "single", "single",
            ],
        }
    )
    grid = stats.hot_zones_grid(df)
    assert set(grid.keys()) == set(range(1, 10))
    # Zone 1: 1 hit / 2 abs = .500
    assert grid[1] == pytest.approx(0.5)
    # Zone 5: 2 hits / 3 abs ≈ .667
    assert grid[5] == pytest.approx(2 / 3)
    # Zone 9: 0 hits / 1 ab = 0.0
    assert grid[9] == 0.0
    # Zones with no data
    assert grid[2] == 0.0


def test_parse_pitch_arsenal_sorts_by_usage() -> None:
    df = pl.DataFrame(
        {
            "pitch_type": ["FF", "SL", "CH"],
            "avg_speed": [95.0, 84.0, 86.0],
            "avg_spin": [2400, 2600, 1800],
            "pitches": [600, 300, 100],
        }
    )
    out = stats.parse_pitch_arsenal(df)
    assert [p["pitch"] for p in out] == ["FF", "SL", "CH"]
    usages = [float(p["usage_pct"]) for p in out]  # type: ignore[arg-type]
    assert usages[0] > usages[1] > usages[2]
    assert out[0]["mph"] == 95.0


def test_parse_exit_velo_filters_by_player() -> None:
    df = pl.DataFrame(
        {
            "player_id": [1, 2],
            "avg_hit_speed": [92.5, 88.0],
            "max_hit_speed": [115.0, 110.0],
            "avg_hit_angle": [12.5, 10.0],
            "brl_percent": [10.0, 5.0],
            "anglesweetspotpercent": [40.0, 30.0],
            "brl_pa": [25, 12],
            "attempts": [400, 300],
        }
    )
    out = stats.parse_exit_velo(df, 1)
    assert "92.5" in out["Avg Exit Velo"]
    assert out["BBE"] == "400"


def test_parse_percentile_ranks_pitcher() -> None:
    df = pl.DataFrame(
        {
            "player_id": [1],
            "k_percent": [78],
            "bb_percent": [55],
            "xwoba": [82],
        }
    )
    out = stats.parse_percentile_ranks(df, 1, "pitcher")
    assert out["K%"] == "78 th percentile"
    assert out["xwOBA"] == "82 th percentile"


def test_aggregate_career_pitcher_ip_weights_rate_stats() -> None:
    f1 = pl.DataFrame({
        "Name": ["Test Player"], "_year": [2020],
        "IP": [200.0], "ERA": [3.00], "W": [15],
    })
    f2 = pl.DataFrame({
        "Name": ["Test Player"], "_year": [2021],
        "IP": [100.0], "ERA": [6.00], "W": [5],
    })
    out = stats.aggregate_career([f1, f2], "Test", "Player", [2020, 2021], "pitcher")
    assert out["type"] == "pitcher"
    stats_block = out["stats"]
    assert isinstance(stats_block, dict)
    # IP-weighted ERA = (3*200 + 6*100) / 300 = 4.00
    assert stats_block["ERA"] == "4.000"
    # Sum of W = 20
    assert stats_block["W"] == "20.0"
