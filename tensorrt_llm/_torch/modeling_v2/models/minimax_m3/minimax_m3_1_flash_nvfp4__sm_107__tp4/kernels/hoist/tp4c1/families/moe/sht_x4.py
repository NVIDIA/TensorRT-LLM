# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""moe.sht-x4: the TP2 C1 MoE engine (one kernel: router-hoisted routed + shared + combine + push).
Pairs with router.lr3c-prr on the same layer.
"""

from __future__ import annotations

from typing import Literal

from ...api import CuteModule, Params, PdlSpec, Spin, Variant, params
from . import _common as C

NAME = "sht-x4"


@params
class ShtX4Params(Params):
    sh_trig: Literal[0, 1] = (
        1  # H9_SHT_CFG: shared CTAs trigger PDL right after their own wait (0 = they do not)
    )


def engine_pdl(grid: int = C.VR200_SMS, what: str = "C1") -> PdlSpec:
    return PdlSpec(
        wait="after_poll",
        trigger="after_wait",
        push=True,
        smid_roles=True,
        grid=grid,
        ctas_per_sm=1,
        tmem_before_wait=True,
        spins=(
            Spin("N-1", "PDL", C.HANG_CAP_NS, on="post route words, polled before the wait"),
            Spin(
                "same-grid",
                "OCC",
                C.HANG_CAP_NS,
                on="engine roles: hx/B2IMG/FC1->FC2 records/partials/y_sh "
                "across the %smid role bijection",
            ),
        ),
        lamport_buckets=("g3.b8:p", "route:c"),
        why=f"{what} m3_moe engine. Triggers after its own wait (routed CTAs at the route barrier, shared CTAs via "
        "sh_trig), before the finalize push into front(L+1). Allocates TMEM before the wait, which is safe because "
        "it runs 1 CTA/SM, smem-exclusive with every co-resident kernel. The pre-wait route poll falls back "
        "silently to experts 0..3 with weight 0 after HANG_CAP. Roles come from %smid with grid == SMs, a "
        "bijection only while no co-tenant stream holds an SM across the launch.",
    )


PDL = engine_pdl()
KERNEL_PDL = (("moe.m3_moe", PDL),)
# %smid role map: requires the %smid set == {0..211}, grid == SMs, 1 CTA/SM and no co-tenant SM holder
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


def runtime_knobs(p: ShtX4Params, llk) -> dict:
    return {
        **C.base_knobs(llk),
        **C.g3_knobs(llk),
        "H4_DL_PORT": C.engine_dir("fused"),  # hoist checks the basename (port_h9r0)
        "H9_SHT_CFG": C.jdump({"sh_trig": p.sh_trig}, compact=True),
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


def make(blk, p: ShtX4Params):
    C.bind(VARIANT.id, runtime_knobs(p, blk.llk))
    # Imported after the bind: the host module reads its knobs at import.
    from ...kernels.moe.engine.fused_shared_trigger_host import make_x4_sht

    fn = make_x4_sht(blk)
    return C.stage(VARIANT, fn, reads=READS, writes=WRITES, pdl=PDL, footprint=C.UNMEASURED)


VARIANT = Variant(
    family="moe",
    name=NAME,
    stages=("moe_experts", "shared", "combine"),
    params=ShtX4Params,
    make=make,
    pdl=PDL,
    footprint=C.UNMEASURED,
    modules=MODULES,
    tp=(2,),
    arch=("sm_107a",),
)

WAIVERS = (
    C.waiver(
        VARIANT.id,
        "6",
        "Push producer that triggers after its own wait and before its push into front(L+1), by design. The consumer "
        "front(L+1) polls the push before its own wait under PDL residency: it launches only after all 212 engine CTAs "
        "triggered, so they are resident and the push lands. Rule 6's invariant (post launched => o_proj passed its "
        "wait) is on the o_proj->post edge and holds there. "
        + C.FS_C1
        + ", which models the trigger before the push. "
        + C.FS_CAVEAT
        + ".",
    ),
)
