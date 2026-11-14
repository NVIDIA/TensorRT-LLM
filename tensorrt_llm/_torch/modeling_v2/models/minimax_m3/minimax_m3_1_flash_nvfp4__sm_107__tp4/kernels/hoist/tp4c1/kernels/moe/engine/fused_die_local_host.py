# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Loads the fused engine package (fused_engine_dir) and builds its shipped config with die-local roles on the
uGPU expert shards."""

import importlib
import importlib.util
import os
import sys

from ....runtime.knobs import KNOBS as _K
from ..knobs import knob, require

DL_PORT = require("fused_engine_dir")
_st = {}


def _op_dl():
    if "op" not in _st:
        pkg = DL_PORT
        spec = importlib.util.spec_from_file_location(
            "m3moe_dl", os.path.join(pkg, "__init__.py"), submodule_search_locations=[pkg]
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules["m3moe_dl"] = mod
        spec.loader.exec_module(mod)
        _st["op"] = importlib.import_module("m3moe_dl.host")
    return _st["op"]


def _build(blk, over, reg, shared_ctas_le4=None):
    from . import fused_host as k3

    _K.setdefault("M3_G1_ITP", str(blk.mw.inter))  # read at op import (INTER: TP2 constants)
    op = _op_dl()
    return _dbg_topk(
        blk, k3.make_k3(blk, over, reg=reg, op=op, sharded=True, shared_ctas_le4=shared_ctas_le4)
    )


# DEV (opt-in, M3LL_DEV_MOE_B2LAG=1, or "b2buf,stages,lag"): the 5-8 row engine with a double-buffered FC2 B image
# and the kernel's FC2 six phases behind FC1 (b2buf 2, stages 6, lag 6) instead of b2buf 1, stages 7, lag 2; the
# host's phase plan stays at lag 2 (M3_MOE_LAG untouched: a plan built for lag 6 measured slower at D >= 20).
# Scheduling only (same MMAs, same reduction order): outputs bitwise equal to the shipped config (a microbenchmark,
# rows 5..8, D 4..32). With one B2 buffer the FC2 B loader of group g waits until the epilogue has drained FC2 of
# group g - 1, so every FC2 phase exposes the image wait + bulk copy + TMEM copy and the weight ring backs up behind
# it. Standalone (cold L2, 8 rows): -2.4 / -2.7 / -3.2 / -2.7 us per layer at D 16 / 20 / 24 / 32 distinct experts,
# 0 at D 4..8. Same smem footprint as the shipped config (stages 7 with b2buf 2 adds 18 KB
# and lost 0.5 us at D 4).
_B2LAG_ENV = "M3LL_DEV_MOE_B2LAG"


def _dev_b2lag():
    import os

    v = os.environ.get(_B2LAG_ENV, "").strip()
    if v in ("", "0"):
        return None
    b, s, lag = (2, 6, 6) if v == "1" else (int(x) for x in v.split(","))
    cfg = {"b2buf": b, "stages": s, "lag": lag}
    if not _st.get("b2lag_logged"):
        _st["b2lag_logged"] = True
        print(
            f"tp4c1: {_B2LAG_ENV}={v} engaged: m3_moe 5-8 rows b2buf={b} stages={s} lag={lag}",
            file=sys.stderr,
            flush=True,
        )
    return cfg


def _c2(blk):
    if blk.t > 4:
        return dict({"sf2_w6": 1, "b2buf": 1, "stages": 7}, **(_dev_b2lag() or {}))
    return {"sf2_w6": 1}


def build_shipped(blk, reg, cfg, shared_ctas_le4=None):
    """The shipped fused engine config (K128 FC1, die-local roles, bulk FC1 scales, early trigger, 4 shared stages)
    with cfg on top."""
    return _build(
        blk,
        dict(_c2(blk), early_trig=1, dl=1, sfa_bulk=1, k128=1, sh_nst=4, sh_pf=0, **cfg),
        reg,
        shared_ctas_le4,
    )


def _dbg_topk(blk, fn):
    """M3MINI_DBG_TOPK=1 (mini.py's untimed --route-check build): blk.dbg_topk_ids [t, 4] int32 = the top-4 expert ids
    the engine consumed (its route_dbg[:, :4]), refreshed every call. Unset: fn unchanged (no extra op)."""
    if knob("debug_topk_ids") != "1":
        return fn
    import torch

    rd = blk.h4_state.route_dbg
    blk.dbg_topk_ids = torch.zeros(blk.t, 4, dtype=torch.int32, device=rd.device)

    def fn_dbg():
        fn()
        blk.dbg_topk_ids.copy_(rd[:, :4])

    return fn_dbg
