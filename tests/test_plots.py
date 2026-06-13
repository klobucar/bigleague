from __future__ import annotations

from unittest.mock import patch

import polars as pl
import pytest
from bigleague import plots

_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _zone_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "plate_x": [-0.5, 0.0, 0.5, 0.2, -0.3],
            "plate_z": [2.0, 2.5, 3.0, 1.8, 3.2],
            "pitch_type": ["FF", "SL", "CH", "FF", "CU"],
            "zone": [4, 5, 6, 7, 2],
            "events": ["single", "field_out", "home_run", "strikeout", "double"],
        }
    )


def _spray_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "type": ["X", "X", "X", "X", "S"],
            "hc_x": [100.0, 125.0, 150.0, 110.0, None],
            "hc_y": [80.0, 90.0, 100.0, 70.0, None],
            "events": ["single", "field_out", "home_run", "double", "strikeout"],
            "home_team": ["NYY", "NYY", "NYY", "NYY", "NYY"],
        }
    )


def test_strike_zone_returns_png() -> None:
    out = plots.strike_zone(_zone_df(), "Test")
    assert out.startswith(_PNG_MAGIC)


def test_batter_zone_returns_png() -> None:
    out = plots.batter_zone(_zone_df(), "Test")
    assert out.startswith(_PNG_MAGIC)


def test_matchup_zone_returns_png() -> None:
    out = plots.matchup_zone(_zone_df(), "Test")
    assert out.startswith(_PNG_MAGIC)


def test_hot_zones_returns_png_and_calls_grid() -> None:
    df = _zone_df()
    with patch("bigleague.plots.hot_zones_grid", wraps=plots.hot_zones_grid) as m:
        out = plots.hot_zones(df, "Test")
    assert out.startswith(_PNG_MAGIC)
    m.assert_called_once_with(df)


def test_savant_to_feet_scales_pixel_grid() -> None:
    """Regression: coordinates were translated but never scaled to feet."""
    # Home plate maps to the origin.
    assert plots._savant_to_feet(125.42, 198.27) == (0.0, 0.0)
    # A known point: 100 px straight up from home plate is ~249.5 ft to CF.
    x, y = plots._savant_to_feet(125.42, 98.27)
    assert x == pytest.approx(0.0)
    assert y == pytest.approx(249.5)
    # And a pulled ball lands left of center, scaled by the same factor.
    x, y = plots._savant_to_feet(100.0, 100.0)
    assert x == pytest.approx((100.0 - 125.42) * 2.495)
    assert y == pytest.approx((198.27 - 100.0) * 2.495)


def test_spray_chart_filters_to_in_play() -> None:
    out = plots.spray_chart(_spray_df(), "Test", stadium_key="yankees")
    assert out.startswith(_PNG_MAGIC)


def test_stadium_outline_returns_png() -> None:
    out = plots.stadium_outline("yankees")
    assert out.startswith(_PNG_MAGIC)


def test_strike_zone_empty_raises() -> None:
    df = pl.DataFrame({"plate_x": [], "plate_z": [], "pitch_type": []})
    with pytest.raises(ValueError, match="No plottable pitches"):
        plots.strike_zone(df, "Test")


def test_spray_chart_empty_raises() -> None:
    df = pl.DataFrame({"type": [], "hc_x": [], "hc_y": [], "events": [], "home_team": []})
    with pytest.raises(ValueError, match="No batted-ball events"):
        plots.spray_chart(df, "Test")


def test_stadium_outline_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown stadium"):
        plots.stadium_outline("not_a_park")


def test_close_fig_called_in_finally() -> None:
    """Even if savefig fails, plt.close() must run (no leaked figures)."""
    from unittest.mock import MagicMock

    df = _zone_df()
    fake_fig, fake_ax = MagicMock(), MagicMock()
    with (
        patch("bigleague.plots.plt") as mock_plt,
        patch("bigleague.plots._save_png", side_effect=RuntimeError("boom")),
    ):
        mock_plt.subplots.return_value = (fake_fig, fake_ax)
        with pytest.raises(RuntimeError, match="boom"):
            plots.strike_zone(df, "Test")
        mock_plt.close.assert_called_once_with(fake_fig)
