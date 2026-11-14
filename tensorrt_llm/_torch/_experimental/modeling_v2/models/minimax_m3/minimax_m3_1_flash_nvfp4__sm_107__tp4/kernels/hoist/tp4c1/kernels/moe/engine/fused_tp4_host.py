# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The moe.sht4-n48 entry point: the fused engine at TP4 (48 shared CTAs, the TP4 engine keys over the shipped
config)."""

import contextlib
import json

from ....runtime.knobs import KNOBS as _K
from ..knobs import knob


def _over():
    return json.loads(knob("fused_tp4_engine_cfg", '{"b2img": 1}') or "{}")


@contextlib.contextmanager
def _patched():  # compat: comm's TP4 per-token factory still wraps hoist._c1_build with this
    from . import router_hoist_host as hoist

    orig = hoist._c1_build
    over = _over()

    def b(blk, reg, cfg, **route):
        return orig(blk, reg, {**cfg, **over}, **route)

    hoist._c1_build = b
    try:
        yield
    finally:
        hoist._c1_build = orig


@contextlib.contextmanager
def _nsh(n):  # compat: as _patched
    old = _K.get("H4_NSH4")
    _K["H4_NSH4"] = str(n)  # h4/k3.py reads it at build (shared CTAs at <= 4 rows; TP4 default 24)
    try:
        yield
    finally:
        if old is None:
            _K.pop("H4_NSH4", None)
        else:
            _K["H4_NSH4"] = old


def _sht_n(blk, n):
    """The TP4 fused engine: n shared CTAs and the TP4 engine keys (TP4_MOE_OVER) over the shipped config. The moe
    family's sht4-n48 make() calls it with its nsh param."""
    from . import fused_shared_trigger_host as sht

    return sht.make_x4_sht(blk, engine_cfg=_over(), shared_ctas_le4=n)


def make_x4_sht4_n48(blk):
    return _sht_n(blk, 48)
