"""Polars schema mechanically derived from `StatcastPitch` annotations.

Importing `STATCAST_SCHEMA` is the canonical way to get the column ->
polars dtype map for parsing Savant CSVs. Any new field added to
`StatcastPitch` shows up here automatically; missing or unannotated
fields raise loudly at import time.
"""

from __future__ import annotations

import typing
from dataclasses import fields
from datetime import date

import polars as pl
from bigleague.models import StatcastPitch

PY_TO_POLARS: dict[type, pl.DataType] = {
    float: pl.Float32(),
    int: pl.Int32(),
    str: pl.String(),
    date: pl.Date(),
    bool: pl.Boolean(),
}


def _unwrap(hint: object) -> object:
    """Return the non-None type from `T | None`, or `hint` itself."""
    origin = typing.get_origin(hint)
    if origin is typing.Union or str(origin) == "types.UnionType":
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return hint


def _build_schema() -> dict[str, pl.DataType]:
    type_hints = typing.get_type_hints(StatcastPitch)
    schema: dict[str, pl.DataType] = {}
    for f in fields(StatcastPitch):
        hint = type_hints.get(f.name)
        if hint is None:
            raise TypeError(f"StatcastPitch.{f.name} has no type annotation")
        inner = _unwrap(hint)
        if not isinstance(inner, type) or inner not in PY_TO_POLARS:
            raise TypeError(
                f"StatcastPitch.{f.name}: unsupported type {inner!r} "
                f"(expected one of {sorted(t.__name__ for t in PY_TO_POLARS)})"
            )
        schema[f.name] = PY_TO_POLARS[inner]
    return schema


STATCAST_SCHEMA: dict[str, pl.DataType] = _build_schema()
