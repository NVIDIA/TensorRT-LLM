# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The moe.sht-x4 entry point: the fused engine on the post's routes and x4 input, with the shared CTAs triggering
PDL right after their own wait (sh_trig)."""

from __future__ import annotations

import atexit
import json
import os

from ....runtime.knobs import KNOBS as _K
from ..knobs import knob
from . import post_nvfp4_handoff as b1x4
from . import router_hoist_host as hoist


def _cfg():
    cfg = {"sh_trig": 1}
    cfg.update(json.loads(knob("fused_shared_trigger_cfg", "{}") or "{}"))
    bad = [k for k in cfg if k != "sh_trig"]
    if bad:
        raise ValueError(f"h9 shtrig: H9_SHT_CFG takes sh_trig only, got {bad}")
    return cfg


def _sht_build(ext_only, engine_cfg=None, shared_ctas_le4=None):
    over = _cfg()

    def b(blk, reg, cfg, **route):
        if not ext_only or int(cfg.get("ext_route", 0)):
            cfg = dict(cfg, **over)
        if engine_cfg:
            cfg = {**cfg, **engine_cfg}
        return hoist._c1_build(blk, reg, cfg, shared_ctas_le4=shared_ctas_le4, **route)

    return b


def _check(blk, arm):
    mod = getattr(hoist._state_c1(blk), "mod", None)
    if not hasattr(mod, "SH_TRIG") or int(mod.SH_TRIG) != int(_cfg()["sh_trig"]):
        raise RuntimeError(
            f"h9 shtrig {arm} L{blk.layer_idx}: engine module SH_TRIG={getattr(mod, 'SH_TRIG', None)}: "
            "H4_DL_PORT is not w6/shtrig/port_h9r0?"
        )


def make_x4_sht(blk, engine_cfg=None, shared_ctas_le4=None):
    """The fused engine on the post's routes and x4 with sh_trig; engine_cfg keys go on top of every engine config,
    shared_ctas_le4 sets the shared CTAs at <= 4 rows (default: the H4_NSH4 knob)."""
    fn = b1x4._ext_x4(
        blk, "extx4sht", _sht_build(True, engine_cfg, shared_ctas_le4), hoist._state_c1
    )
    _check(blk, "extx4sht")
    return fn


def _q(v):
    v = sorted(v)
    return (
        [round(v[len(v) // 2] / 1e3, 2), round(v[0] / 1e3, 2), round(v[-1] / 1e3, 2)] if v else None
    )


def _dump():
    out = _K.get("MINI_OUT")
    if not out or not _K.get("M3_TIMING") or not hoist._TRACK:
        return
    recs = []
    for arm, li, t, st, _ in hoist._TRACK:
        rec = {
            "arm": arm,
            "layer": li,
            "t": t,
            "sh_trig": int(getattr(getattr(st, "mod", None), "SH_TRIG", -1)),
        }
        try:
            td = st.tdbg.cpu()
            n = int(st.num_ctas)
            ent = td[:n, 0][td[:n, 0] > 0]
            if ent.numel() == 0 or td.shape[0] <= n:
                continue
            t0 = int(ent.min())
            col = lambda rows, c: [int(td[r, c]) - t0 for r in rows if int(td[r, c]) > 0]  # noqa: E731
            rt, sh = range(n), range(n, td.shape[0])
            rec["routed"] = {
                s: _q(col(rt, int(s))) for s in ("1", "2", "7")
            }  # [median, min, max] us from t0
            rec["shared"] = {s: _q(col(sh, int(s))) for s in ("1", "51", "52")}
            e7, s52 = rec["routed"]["7"], rec["shared"]["52"]
            if e7 and s52:
                rec["last_shared_52_minus_last_routed_7"] = round(s52[2] - e7[2], 2)
                rec["median_routed_7_minus_last_routed_7"] = round(e7[0] - e7[2], 2)
        except Exception as err:  # noqa: BLE001 (a dump must not mask the run's own exit status)
            rec["error"] = repr(err)
        recs.append(rec)
    with open(os.path.join(out, f"sht-tdbg-r{_K.get('RANK', '0')}.json"), "w") as f:
        json.dump({"layers": recs}, f, indent=1)


atexit.register(_dump)
