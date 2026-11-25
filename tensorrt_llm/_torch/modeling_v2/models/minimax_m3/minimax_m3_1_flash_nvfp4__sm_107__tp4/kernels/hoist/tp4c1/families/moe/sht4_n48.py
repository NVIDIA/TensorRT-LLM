# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""moe.sht4-n48: the TP4 C1 MoE engine (the TP2 C1 hoist engine at I = 768, 48 shared CTAs).
Pairs with router.lr4-prr4.
"""

from __future__ import annotations

from typing import Literal

from ...api import CuteModule, Params, Variant, params
from . import _common as C
from .sht_x4 import engine_pdl

NAME = "sht4-n48"


@params
class Sht4N48Params(Params):
    sh_trig: Literal[0, 1] = 1  # H9_SHT_CFG
    b2img: Literal[0, 1] = (
        1  # TP4_MOE_OVER {"b2img": 1}: the TP2 4-token FC2-B image path, required by sf2_w6
    )
    nsh: Literal[48] = (
        48  # shared CTAs (moe4._sht_n sets H4_NSH4 around the build); only 48 is landed
    )


PDL = engine_pdl(
    what="TP4 C1 (co-resident with the 32-CTA prr4 post: 221,760 + 1 KB + post smem <= 233,472 B)"
)
KERNEL_PDL = (("moe.m3_moe", PDL),)
SMID_ROLE_MAP = {
    "kind": "arithmetic",
    "site": "_dl_role in m3_moe_kernel.py",
    "grid": "NCTA + NSH == SMs",
    "ncta": 164,
    "nsh": 48,
    "sms": C.VR200_SMS,
    "asserted": "hoist.py checks the grid size only",
    "missing": "a probe that the %smid set is every SM, and a per-call role claim",
}

MODULES = (CuteModule("moe", NAME, "engine", "kernels/moe/engine/fused"),)


def runtime_knobs(p: Sht4N48Params, llk) -> dict:
    return {
        **C.base_knobs(llk),
        **C.g3_knobs(llk),
        "H4_DL_PORT": C.engine_dir("fused"),
        "H9_SHT_CFG": C.jdump({"sh_trig": p.sh_trig}, compact=True),
        "TP4_MOE_OVER": C.jdump({"b2img": p.b2img}, compact=False),
    }


READS = (
    "moe_in",
    "moe_in_q",
    "route",
    "mw.experts",
    "mw.shared_gu",
    "mw.shared_dn",
    "mw.router",
    "mw.bias",
)
WRITES = ("ffn_out",)


def make(blk, p: Sht4N48Params):
    C.bind(VARIANT.id, runtime_knobs(p, blk.llk))
    # Imported after the bind: the host module reads its knobs at import.
    from ...kernels.moe.engine import fused_tp4_host

    fn = fused_tp4_host._sht_n(blk, p.nsh)
    return C.stage(VARIANT, fn, reads=READS, writes=WRITES, pdl=PDL, footprint=C.UNMEASURED)


VARIANT = Variant(
    family="moe",
    name=NAME,
    stages=("moe_experts", "shared", "combine"),
    params=Sht4N48Params,
    make=make,
    pdl=PDL,
    footprint=C.UNMEASURED,
    modules=MODULES,
    tp=(4,),
    arch=("sm_107a",),
)

WAIVERS = (
    C.waiver(
        VARIANT.id,
        "6",
        "Same engine bytes and trigger as moe.sht-x4 (after its wait, before the push). "
        + C.NO_TP4
        + ". Argument only: "
        "every engine poll is safe on a single stream; the 32-CTA prr4 post co-resides with the engine (221,760 + 1 KB "
        "+ post smem <= 233,472 B), and front(L+1) (232 KB) launches only after all 212 engine CTAs triggered.",
    ),
)
