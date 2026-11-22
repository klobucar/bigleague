from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
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


def test_hot_zones_cold_slump_renders_but_truly_empty_raises() -> None:
    # 0-for-everything: every zone average is 0.0 but the at-bats are real —
    # this is a cold slump, not missing data, and must render.
    cold = pl.DataFrame(
        {
            "zone": [1, 5, 9],
            "events": ["field_out", "strikeout", "field_out"],
        }
    )
    out = plots.hot_zones(cold, "Cold")
    assert out.startswith(_PNG_MAGIC)

    # No AB-resolving events at all -> genuinely nothing to chart.
    empty = pl.DataFrame({"zone": [1, 5], "events": ["ball", "foul"]})
    with pytest.raises(ValueError, match="No at-bats"):
        plots.hot_zones(empty, "Empty")


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


@pytest.mark.parametrize("home_code", ["ATH", "OAK"])
def test_spray_chart_athletics_matches_both_era_codes(home_code: str) -> None:
    """Savant uses ATH from 2025 on; older frames carry OAK. The athletics
    park filter must keep rows from either era."""
    df = pl.DataFrame(
        {
            "type": ["X", "X"],
            "hc_x": [100.0, 150.0],
            "hc_y": [80.0, 100.0],
            "events": ["single", "home_run"],
            "home_team": [home_code, home_code],
        }
    )
    out = plots.spray_chart(df, "Test", stadium_key="athletics")
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


def test_gc_runs_even_when_saving_fails() -> None:
    """A figure is a web of reference cycles holding a bitmap: the collection
    in `finally` must run even if rendering blows up."""
    df = _zone_df()
    with (
        patch("bigleague.plots._save_png", side_effect=RuntimeError("boom")),
        patch("bigleague.plots.gc.collect") as mock_gc,
        pytest.raises(RuntimeError, match="boom"),
    ):
        plots.strike_zone(df, "Test")
    mock_gc.assert_called_once()


_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    """Run `code` in a fresh interpreter, so module state from other tests
    can't leak into what these tests assert about imports."""
    env = {**os.environ, "PYTHONPATH": _PROJECT_ROOT, "BIGLEAGUE_NO_CACHE": "1"}
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )


def test_importing_bigleague_does_not_import_matplotlib() -> None:
    """matplotlib costs ~40 MB resident; processes that never plot shouldn't pay."""
    out = _run_python(
        "import sys, bigleague\n"
        "assert 'matplotlib' not in sys.modules, 'matplotlib was imported'\n"
        "assert 'matplotlib.pyplot' not in sys.modules\n"
    )
    assert out.returncode == 0, out.stderr


def test_rendering_works_without_pyplot_and_leaves_the_backend_alone() -> None:
    out = _run_python(
        "import sys, matplotlib, polars as pl, bigleague as bl\n"
        "matplotlib.use('svg')  # the caller's choice\n"
        "df = pl.DataFrame({'plate_x': [0.0, 0.5], 'plate_z': [2.5, 3.0],\n"
        "                   'pitch_type': ['FF', 'SL']})\n"
        "png = bl.strike_zone(df, 't')\n"
        "assert png.startswith(b'\\x89PNG'), png[:8]\n"
        "assert 'matplotlib.pyplot' not in sys.modules, 'pyplot was imported'\n"
        "assert matplotlib.get_backend() == 'svg', matplotlib.get_backend()\n"
    )
    assert out.returncode == 0, out.stderr
