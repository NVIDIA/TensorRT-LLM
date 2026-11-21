# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""moe.sht4-n48-q: moe.sht4-n48 on the mode-Q engine (cfg pt5 = 1, the bytes of moe.sht-x4-q); pairs with
router.lr4-prr4-q only. After its grid wait the engine scales each token's FC1 sums by s_t from row 16 of the post's x4
scale allocation, before the SwiGLU epilogue; FC2 uses the fixed scale, applied as weight prep."""

from __future__ import annotations

from ...api import CuteModule, Variant
from . import _common as C
from . import sht4_n48 as B

NAME = "sht4-n48-q"
MODULES = (CuteModule("moe", NAME, "engine", "kernels/moe/engine/fused"),)


def runtime_knobs(p: B.Sht4N48Params, llk) -> dict:
    return {**B.runtime_knobs(p, llk), "H4_DL_PORT": C.engine_dir("fused")}


def make(blk, p: B.Sht4N48Params):
    if not p.sh_trig or not p.b2img:
        raise ValueError(f"{VARIANT.id}: the mode-Q engine needs sh_trig 1 and b2img 1")
    C.bind(VARIANT.id, runtime_knobs(p, blk.llk))
    # Imported after the bind: the host module reads its knobs at import.
    from ...kernels.comm.allreduce_norm_quant import route_warp_tp4_fc1_per_token_host as pt4

    pt4.prep_mw(blk.mw)
    fn = pt4.make_moe_pt(blk)
    return C.stage(VARIANT, fn, reads=B.READS, writes=B.WRITES, pdl=B.PDL, footprint=C.UNMEASURED)


VARIANT = Variant(
    family="moe",
    name=NAME,
    stages=B.VARIANT.stages,
    params=B.Sht4N48Params,
    make=make,
    pdl=B.PDL,
    footprint=C.UNMEASURED,
    modules=MODULES,
    tp=(4,),
    arch=("sm_107a",),
)

WAIVERS = tuple({**w, "stage": w["stage"].replace(B.VARIANT.id, VARIANT.id)} for w in B.WAIVERS)
