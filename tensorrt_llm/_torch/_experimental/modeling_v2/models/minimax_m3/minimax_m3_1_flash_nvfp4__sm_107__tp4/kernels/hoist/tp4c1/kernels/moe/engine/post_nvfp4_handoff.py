# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The x4 hand-off: the post writes the engine's NVFP4 activation (lbh.x4_bufs) and the engine built with b1_x4
reads it after its grid wait; _ext_x4 builds the engine and checks the pairing."""

from __future__ import annotations

import contextlib

from ....runtime.knobs import KNOBS as _K
from ...comm.allreduce_norm_quant import router_exchange as lbh
from . import router_hoist_host as hoist


@contextlib.contextmanager
def _env(key, val):
    old = _K.get(key)
    _K[key] = val
    try:
        yield
    finally:
        if old is None:
            _K.pop(key, None)
        else:
            _K[key] = old


def _post_x4(factory, blk):
    with _env("H9_RH_X4", "1"):  # lbh reads it at factory time (lbh._kw)
        fn = factory(blk)
    b = getattr(blk, "h9_x4", None)
    if b is None or not b.post:
        raise RuntimeError(
            f"h9 b1x4 L{blk.layer_idx}: {factory.__name__} did not build an X4 post (lbh without x4?)"
        )
    return fn


def make_post_rh_x4(blk):
    """hoist.make_post_rh (full trio) that also writes x4 (pairs with make_oproj_rh and an _x4 engine)"""
    return _post_x4(hoist.make_post_rh, blk)


def _x4_build(blk, build, seen):
    """hoist build adapter: the ext_route=1 build gets b1_x4=1 and blk's x4 buffers as route kwargs (op.layer for r0,
    op16.run for x0, as hoist binds route_lines / rctr); its state's run is observed during the factory's compile
    call."""
    b = lbh.x4_bufs(blk)

    def build_x4(blk_, reg, cfg, **route):
        if int(cfg.get("ext_route", 0)) != 1:
            return build(blk_, reg, cfg, **route)
        cfg = dict(cfg, b1_x4=1)
        on_layer = route.pop("on_layer", None)
        patched = []

        def observe(st):
            run = type(st).run

            def run_obs(layer, hidden, *a, **kw):
                lay = getattr(
                    layer, "_lay", layer
                )  # hoist._LayerProxy (x0) or the port's layer (r0)
                pick = lambda k: kw[k] if kw.get(k) is not None else getattr(lay, k, None)  # noqa: E731
                seen.append(
                    dict(state=st, x4=pick("x4"), x4sf=pick("x4sf"), sfsc=lay.fc31_input_scale)
                )
                return run(st, layer, hidden, *a, **kw)

            st.run = run_obs
            patched.append(st)
            if on_layer is not None:
                on_layer(st)

        try:
            return build(blk_, reg, cfg, x4=b.q, x4sf=b.sf, on_layer=observe, **route)
        finally:
            for st in patched:
                st.__dict__.pop(
                    "run", None
                )  # back to the class method (no per-call wrapper in the chain)

    return build_x4


def _ext_x4(blk, arm, build, state_of):
    seen = []
    fn = hoist._ext(blk, arm, _x4_build(blk, build, seen), state_of)
    st = state_of(blk)
    if not seen or not getattr(st, "b1_x4", 0):
        raise RuntimeError(
            f"h9 b1x4 {arm} L{blk.layer_idx}: the ext engine was not built with b1_x4 "
            f"(observed compile calls {len(seen)}, state b1_x4 {getattr(st, 'b1_x4', None)}): wrong port?"
        )
    s = seen[-1]
    if s["x4"] is None or s["x4sf"] is None:
        raise RuntimeError(
            f"h9 b1x4 {arm} L{blk.layer_idx}: the engine's compile call carried no x4 / x4sf"
        )
    lbh.check_x4_pair(blk, True, s["x4"], s["x4sf"], s["sfsc"])
    return fn
