# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""router.lr4-prr4-q: router.lr4-prr4 with the mode-Q per-token post at world 4; pairs with moe.sht4-n48-q only.
The post quantizes each row with the QAT row quantizer and writes fp32 s_t[t] into row 16 of the x4 scale allocation;
the o_proj is the parent's lr4 kernel, run on this post's extension."""

from __future__ import annotations

from ...api import CuteModule, NvccModule, Variant
from ..moe import _common as C
from . import lr4_prr4 as B

NAME = "lr4-prr4-q"
HDR_HASH = "40c1042cbf7d"  # src_hash() of the host module that builds this post

MODULES = (
    CuteModule("router", B.NAME, "oproj_lr4", "kernels/moe/router/fused_o_proj"),
    NvccModule(
        "router",
        NAME,
        "pt4post",
        sources=("comm/allreduce_norm_quant/route_warp_tp4_fc1_per_token_ext.cu",),
        include_dirs=("comm/allreduce_norm_quant",),
        flags=B._NV + (f"-DH9_HDR_HASH=0x{HDR_HASH}",),
        cflags=("-O3", "-std=c++20"),
    ),
)


def runtime_knobs(p: B.Lr4Prr4Params, llk) -> dict:
    return B.runtime_knobs(p, llk)


def make(blk, p: B.Lr4Prr4Params):
    C.bind(VARIANT.id, runtime_knobs(p, blk.llk))
    # Imported after the bind: the host module reads its knobs at import.
    from ...kernels.comm.allreduce_norm_quant import route_warp_tp4_fc1_per_token_host as pt4

    f_oproj = pt4.make_oproj(blk)
    f_post = pt4.make_post_pt(blk)
    return C.stage(
        VARIANT,
        C.chain(f_oproj, f_post),
        reads=B.READS,
        writes=B.WRITES,
        pdl=B.PDL,
        footprint=C.UNMEASURED,
        kernels=B.KERNEL_PDL,
    )


VARIANT = Variant(
    family="router",
    name=NAME,
    stages=B.VARIANT.stages,
    params=B.Lr4Prr4Params,
    make=make,
    pdl=B.PDL,
    kernels=B.KERNEL_PDL,
    footprint=C.UNMEASURED,
    modules=MODULES,
    tp=(4,),
    arch=("sm_107a",),
)

WAIVERS = tuple({**w, "stage": w["stage"].replace(B.VARIANT.id, VARIANT.id)} for w in B.WAIVERS)
