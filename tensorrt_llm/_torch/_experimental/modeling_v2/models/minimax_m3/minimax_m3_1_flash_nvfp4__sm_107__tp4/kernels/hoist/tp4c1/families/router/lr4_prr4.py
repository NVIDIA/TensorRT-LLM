# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""router.lr4-prr4: the TP4 C1 early router, two kernels in one Stage (o_proj, then post).
Covers oproj, boundary_post, router and moe_quant; pairs with moe.sht4-n48.
"""

from __future__ import annotations

from typing import Literal

from ...api import CuteModule, NvccModule, Params, PdlSpec, Spin, Variant, params
from ..moe import _common as C

NAME = "lr4-prr4"
F_OPROJ = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3."
    "minimax_m3_1_flash_nvfp4__sm_107__tp4.kernels.hoist.tp4c1.kernels.comm.allreduce_norm_quant.route_warp_tp4_host:make_oproj_ra4_p"
)
F_POST = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3."
    "minimax_m3_1_flash_nvfp4__sm_107__tp4.kernels.hoist.tp4c1.kernels.comm.allreduce_norm_quant.route_warp_tp4_host:make_post_rh_x4_p"
)
HDR_HASH = "10d611694e7b"  # src_hash() of the host module that builds this post


@params
class Lr4Prr4Params(Params):
    rx2: Literal[0, 1] = 1  # PRR_RX2


OPROJ_PDL = PdlSpec(
    wait="entry",
    trigger="exit",
    push=True,
    grid=112,
    ctas_per_sm=1,
    lamport_buckets=("g3.b8:p", "rx:p"),
    why="lr4: router CTAs wait after the pre-wait residual load and W'/W'' smem burst, then exit; GEMV warp 1 waits "
    "before the epilogue. There is no explicit trigger (mode 4 + ptrig 0 -> trigger_early=False), so dependents "
    "launch at exit, after the push: rule 6 holds, stricter than after_push. A future ptrig/tune must keep "
    "trigger_early False.",
)
POST_PDL = PdlSpec(
    wait="none",
    trigger="entry",
    grid=32,
    ctas_per_sm=1,
    lamport_buckets=("g3.b8:c", "rx:c", "route:p"),
    spins=(
        Spin("N-1", "PDL", C.HANG_CAP_NS, on="o_proj(L) g3.b8 partials, local, complete"),
        Spin("peer", "PEER", C.HANG_CAP_NS, on="3 peers' o_proj(L) g3.b8 partials"),
        Spin(
            "peer",
            "PEER",
            C.RX_DEADLINE_NS,
            on="peers' lr4 router rx words (2 s deadline, then __trap)",
        ),
    ),
    why="prr4 lbv3rp: no wait anywhere, and a trigger at entry before both polls: conditional on the peer, so a real "
    "rule-2/3 site that needs a waiver. Boot-time trap risk: the 2 s rx deadline against the lr4 router kernel's "
    "lazy compile on the first eager call.",
)
# DEV (opt-in, not this spec): M3LL_DEV_OPROJ_PTRIG=1 runs the o_proj at ptrig 1 (GEMV warp 1 and the router CTAs
# trigger right after their wait, before the push) and the post with PW 1 (griddepcontrol.wait at the end of every
# CTA). Shipped source note: in push mode 4 the GEMV CTAs already trigger after their push (kernel_tp4.py, tx 128
# after bar.sync 1, ar_push >= 2) and the router CTAs exit right after theirs, so the shipped edge is after_push.
KERNEL_PDL = (("oproj.lr4", OPROJ_PDL), ("post.prr4", POST_PDL))
PDL = PdlSpec(
    wait=OPROJ_PDL.wait,
    trigger=POST_PDL.trigger,
    spins=POST_PDL.spins,
    grid=112 + 32,
    ctas_per_sm=1,
    lamport_buckets=OPROJ_PDL.lamport_buckets + POST_PDL.lamport_buckets,
    why="envelope of KERNEL_PDL (o_proj lr4 then post prr4); see each kernel's why. "
    + POST_PDL.why,
)
SMID_ROLE_MAP = None

_NV = ("-O3", "-std=c++20", "-gencode=arch=compute_107a,code=sm_107a")
MODULES = (
    CuteModule("router", NAME, "oproj_lr4", "kernels/moe/router/fused_o_proj"),
    NvccModule(
        "router",
        NAME,
        "prr4",
        sources=("comm/allreduce_norm_quant/route_warp_tp4_ext.cu",),
        include_dirs=("comm/allreduce_norm_quant",),
        flags=_NV + (f"-DH9_HDR_HASH=0x{HDR_HASH}",),
        cflags=("-O3", "-std=c++20"),
    ),
)


def runtime_knobs(p: Lr4Prr4Params, llk) -> dict:
    return {**C.base_knobs(llk), **C.g3_knobs(llk), "H9_RH_ROLL": "0", "PRR_RX2": str(p.rx2)}


READS = (
    "chain.out",
    "attn_partial",
    "residual_mid",
    "aw.o",
    "aw.post_norm",
    "mw.router",
    "mw.bias",
)
WRITES = ("attn_partial", "residual_out", "moe_in", "moe_in_q", "route")


def make(blk, p: Lr4Prr4Params):
    C.bind(VARIANT.id, runtime_knobs(p, blk.llk))
    f_oproj = C.legacy_attr(F_OPROJ)(blk)
    f_post = C.legacy_attr(F_POST)(blk)
    return C.stage(
        VARIANT,
        C.chain(f_oproj, f_post),
        reads=READS,
        writes=WRITES,
        pdl=PDL,
        footprint=C.UNMEASURED,
        kernels=KERNEL_PDL,
    )


VARIANT = Variant(
    family="router",
    name=NAME,
    stages=("oproj", "boundary_post", "router", "moe_quant"),
    params=Lr4Prr4Params,
    make=make,
    pdl=PDL,
    kernels=KERNEL_PDL,
    footprint=C.UNMEASURED,
    modules=MODULES,
    tp=(4,),
    arch=("sm_107a",),
)

WAIVERS = (
    C.waiver(
        f"{VARIANT.id}/post.prr4",
        "3",
        "prr4 lbv3rp has no wait and triggers at entry before its peer polls: conditional on the peer. "
        + C.NO_TP4
        + ". "
        "Argument only: same structure as the proved C1 chain - one never-waiting grid of 32 CTAs (~0 smem), no side "
        "stream, no separate routing grid, o_proj triggers at exit. Open: the 2 s rx deadline against the lr4 router "
        "kernel's lazy compile can trap at boot.",
    ),
)
