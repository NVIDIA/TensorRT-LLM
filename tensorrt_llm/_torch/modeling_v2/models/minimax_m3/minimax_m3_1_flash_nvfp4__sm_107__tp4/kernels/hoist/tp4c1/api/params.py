# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Params: a variant's frozen, typed knob set, built from a preset table (closed schema)."""

from __future__ import annotations

import dataclasses
import types
from typing import Any, Literal, Union, get_args, get_origin, get_type_hints


class ParamsError(ValueError):
    pass


def params(cls):
    """Class decorator: frozen, keyword-only dataclass that is a Params."""
    if not issubclass(cls, Params):
        raise TypeError(f"{cls.__name__} must subclass tp4c1.api.Params")
    return dataclasses.dataclass(frozen=True, kw_only=True)(cls)


class Params:
    """Base for every variant's Params. Subclass and decorate with @params; every field needs a type and a default
    equal to the landed value, so an empty preset table reproduces the landing."""

    @classmethod
    def fields(cls) -> dict[str, Any]:
        hints = get_type_hints(cls)
        return {f.name: hints[f.name] for f in dataclasses.fields(cls)}

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None, where: str = "") -> "Params":
        d = dict(d or {})
        for old, new in getattr(
            cls, "RENAMED", {}
        ).items():  # field names before a rename stay accepted
            if old in d:
                if new in d:
                    raise ParamsError(
                        f"{where or cls.__name__}: both {old!r} (old name) and {new!r} given"
                    )
                d[new] = d.pop(old)
        fl = cls.fields()
        unknown = sorted(set(d) - set(fl))
        if unknown:
            raise ParamsError(
                f"{where or cls.__name__}: unknown key(s) {unknown}; allowed {sorted(fl)}"
            )
        kw = {k: _coerce(v, fl[k], f"{where or cls.__name__}.{k}") for k, v in d.items()}
        return cls(**kw)

    def to_dict(self) -> dict[str, Any]:
        return {f.name: _plain(getattr(self, f.name)) for f in dataclasses.fields(self)}


def _plain(v):
    if isinstance(v, tuple):
        return [_plain(x) for x in v]
    return v


def _coerce(v, tp, where):
    origin = get_origin(tp)
    if tp is Any:
        return v
    if origin is Literal:
        if v not in get_args(tp):
            raise ParamsError(f"{where}: {v!r} not in {get_args(tp)}")
        return v
    if origin in (Union, getattr(types, "UnionType", Union)):
        errs = []
        for a in get_args(tp):
            if a is type(None):
                if v is None:
                    return None
                continue
            try:
                return _coerce(v, a, where)
            except ParamsError as e:
                errs.append(str(e))
        raise ParamsError(f"{where}: {v!r} matches none of {tp}: {errs}")
    if origin is tuple:
        if not isinstance(v, (list, tuple)):
            raise ParamsError(f"{where}: expected a list for {tp}, got {type(v).__name__}")
        args = get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(x, args[0], f"{where}[{i}]") for i, x in enumerate(v))
        if len(args) != len(v):
            raise ParamsError(f"{where}: expected {len(args)} items, got {len(v)}")
        return tuple(_coerce(x, a, f"{where}[{i}]") for i, (x, a) in enumerate(zip(v, args)))
    if tp is bool:
        if not isinstance(v, bool):
            raise ParamsError(f"{where}: expected bool, got {v!r}")
        return v
    if tp is int:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ParamsError(f"{where}: expected int, got {v!r}")
        return v
    if tp is float:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ParamsError(f"{where}: expected float, got {v!r}")
        return float(v)
    if tp is str:
        if not isinstance(v, str):
            raise ParamsError(f"{where}: expected str, got {v!r}")
        return v
    raise ParamsError(
        f"{where}: unsupported Params field type {tp!r} (use bool/int/float/str/Literal/tuple/Optional)"
    )


@params
class NoParams(Params):
    pass


__all__ = ["Params", "params", "ParamsError", "NoParams"]
