from __future__ import annotations

import typing
from dataclasses import fields

import pytest
from bigleague.models import StatcastPitch
from bigleague.schema import STATCAST_SCHEMA


def test_schema_covers_every_field() -> None:
    field_names = {f.name for f in fields(StatcastPitch)}
    assert set(STATCAST_SCHEMA.keys()) == field_names


def test_every_field_has_annotation() -> None:
    hints = typing.get_type_hints(StatcastPitch)
    for f in fields(StatcastPitch):
        assert f.name in hints, f"{f.name} is missing a type annotation"


def test_unknown_field_type_raises_on_schema_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """Adding a field with an unsupported type must blow up loudly."""
    from bigleague import schema as schema_mod

    class _Bogus:
        pass

    fake_hints = {**typing.get_type_hints(StatcastPitch), "fake_col": _Bogus}
    monkeypatch.setattr(schema_mod.typing, "get_type_hints", lambda *_a, **_k: fake_hints)

    fake_fields = (*fields(StatcastPitch),)

    class _Field:
        def __init__(self, name: str) -> None:
            self.name = name

    monkeypatch.setattr(
        schema_mod, "fields", lambda *_a, **_k: (*fake_fields, _Field("fake_col"))
    )
    with pytest.raises(TypeError, match="unsupported type"):
        schema_mod._build_schema()


def test_from_row_accepts_partial_dict() -> None:
    row = {f.name: None for f in fields(StatcastPitch)}
    row["batter"] = 1
    row["pitcher"] = 2
    row["balls"] = 0
    row["strikes"] = 0
    row["game_year"] = 2024
    row["outs_when_up"] = 0
    row["inning"] = 1
    row["at_bat_number"] = 1
    row["pitch_number"] = 1
    row["home_score"] = 0
    row["away_score"] = 0
    row["bat_score"] = 0
    row["fld_score"] = 0
    row["post_away_score"] = 0
    row["post_home_score"] = 0
    row["post_bat_score"] = 0
    row["post_fld_score"] = 0
    row["game_pk"] = 1
    row["home_team"] = "NYY"
    row["away_team"] = "BOS"
    pitch = StatcastPitch.from_row(row)
    assert pitch.batter == 1
    assert pitch.home_team == "NYY"
