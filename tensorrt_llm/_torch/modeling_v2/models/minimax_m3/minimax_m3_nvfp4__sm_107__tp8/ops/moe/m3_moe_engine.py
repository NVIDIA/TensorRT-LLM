# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3's MoE layer for 1..8 decode tokens in one CuTe DSL launch (``m3_moe``, the sm_107 TP2 build: routed
and shared expert width 1536 per rank): the router, the routed NVFP4 experts and the shared MXFP8 expert, on
caller-owned state (an ``M3MoeState`` and one ``M3MoeLayer`` per MoE layer)."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.logger import logger

from ....ops.moe.reference import _E2M1 as _E2M1
from ....ops.moe.reference import _F32_MAX as _F32_MAX
from ....ops.moe.reference import _LOG2E as _LOG2E
from ....ops.moe.reference import _RCP6 as _RCP6
from ....ops.moe.reference import ROUTED_SCALING as ROUTED_SCALING
from ....ops.moe.reference import SHARED_ALPHA as SHARED_ALPHA
from ....ops.moe.reference import SHARED_LIMIT as SHARED_LIMIT
from ....ops.moe.reference import _bf16 as _bf16
from ....ops.moe.reference import _e2m1_decode as _e2m1_decode
from ....ops.moe.reference import _e2m1_round as _e2m1_round
from ....ops.moe.reference import _e4m3_round as _e4m3_round
from ....ops.moe.reference import _f32 as _f32
from ....ops.moe.reference import _fc1_epilogue as _fc1_epilogue
from ....ops.moe.reference import _gemm as _gemm
from ....ops.moe.reference import _nvfp4_act as _nvfp4_act
from ....ops.moe.reference import _nvfp4_expert as _nvfp4_expert
from ....ops.moe.reference import _row_perm as _row_perm
from ....ops.moe.reference import _shared_expert as _shared_expert_reference
from ....ops.moe.reference import _ue8m0_up as _ue8m0_up
from ....ops.moe.reference import moe_reference as _moe_reference
from ....ops.moe.reference import moe_route as moe_route
from ....ops.moe.reference import mxfp8_dequant as mxfp8_dequant
from ....ops.moe.reference import mxfp8_quant_dequant as mxfp8_quant_dequant
from ....ops.moe.reference import unswizzle_sf as unswizzle_sf
from ..m3_moe.op import HIDDEN as HIDDEN
from ..m3_moe.op import INTER, SHARED_INTER, M3MoeLayer, M3MoeState, ar_push_desc, is_supported
from ..m3_moe.op import TOP_K as TOP_K

__all__ = [
    "M3MoeLayer",
    "M3MoeState",
    "ar_push_desc",
    "is_supported",
    "m3_moe_engine",
    "moe_reference",
    "moe_route",
    "mxfp8_dequant",
    "mxfp8_quant_dequant",
    "unswizzle_sf",
]


def _shared_expert(
    x: torch.Tensor,
    shared_mxfp8: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    return _shared_expert_reference(x, shared_mxfp8, shared_inter=SHARED_INTER)


def moe_reference(hidden: torch.Tensor, layer: M3MoeLayer, shared: bool) -> torch.Tensor:
    return _moe_reference(hidden, layer, shared, inter=INTER, shared_inter=SHARED_INTER)


_ENGAGED: set = set()


class M3MoeEngine(OpWrapper):
    """One MiniMax-M3 MoE layer for M (1..8) decode tokens: ``layer.state.run(layer, hidden, out, arp)``.

    ``hidden`` bf16 [M, 6144] contiguous, M = the state's token count. ``layer`` (from ``M3MoeState.layer``) carries the
    TRTLLM-Gen NVFP4 expert buffers read in place, the per-expert scalars, the router weight / bias, the activation's
    NVFP4 global scale and, on a ``shared`` state, the shared expert's MXFP8 weights; its state carries the compiled
    kernel and the workspace every layer on it shares. Returns bf16 [M, 6144]: bf16(shared + routed) on a ``shared``
    state, else the routed partial. ``out``: write there instead of a new tensor. ``arp`` (``ar_push_desc``): also
    push the rows into a multicast all-reduce exchange.

    The routed part is TRTLLM-Gen's NVFP4 MoE on the fp32 router logits: sigmoid (tanh form) + correction bias top-4
    of 128 (ties to the lower id), weights sigmoid / sum * 2.0 rounded to bf16; the activation quantized to NVFP4 with
    the layer's global scale; per expert the raw FC1 sums, SwiGLU-OAI (alpha / beta / clamp), the NVFP4 requant with
    fc31_scale_c, FC2, bf16(fc2_alpha x sum); the fma chain over the top-4 in order. The shared expert is the target's
    MXFP8 SwiGLU-OAI MLP.
    """

    ARCHS = frozenset({Arch.SM_107})

    # One cell per token count: m_max is a trace-time constant, so each is its own build, and the target builds all
    # eight (decode batches of 1..8 requests, spec-decode verify rows up to 8). The whys name what the build differs in.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="1 token: FC1 -> FC2 records replicated 32x, logits 32x, last phase without trailing FC2",
            spec=dict(tokens=1),
        ),
        Cell(
            why="2 tokens: records 32x, logits 32x, the one build without AR_PRE",
            spec=dict(tokens=2),
        ),
        Cell(why="3 tokens: records 8x, logits 32x, 4 FC2-B builder warps", spec=dict(tokens=3)),
        Cell(
            why="4 tokens: one request verifying 3 drafts (the TP2 b1 point): 6-stage ring, B2IMG hand-off, "
            "the 4 two-block FC2 CTAs stream no FC1",
            spec=dict(tokens=4),
        ),
        Cell(
            why="5 tokens: single records, a routing warp takes two tokens, logits 32x",
            spec=dict(tokens=5),
        ),
        Cell(
            why="6 tokens: single records, routing warps 8.. take tokens 4..5", spec=dict(tokens=6)
        ),
        Cell(why="7 tokens: the B2IMG global FC1 -> FC2 hand-off", spec=dict(tokens=7)),
        Cell(
            why="8 tokens: two requests verifying 3 drafts (the TP2 b2 point): 5-stage ring, x read from global "
            "(no XSTAGE), no early group 0, the 4 two-block FC2 CTAs stream no FC1",
            spec=dict(tokens=8),
        ),
    )

    note = """
    The kernel's elementwise math uses the PTX approximations (rcp.approx, ex2.approx, tanh) where `reference` uses
    torch's correctly rounded ones, and its GEMMs sum in tensor-core order where `reference` sums in fp64: a
    quantization boundary (NVFP4 activation / intermediate code, routing near-tie) can fall on the other side, so the
    two agree within the band the test derives, not bit for bit. The routed part is bit for bit the stock TRTLLM-Gen
    NVFP4 MoE (fp4_quantize + fp4_block_scale_moe_runner) on the same routing; the test checks that too.

    Not covered: the all-reduce push (`arp`), which needs a multicast exchange (the decode forward's boundary all-reduce
    exercises it); the non-shared state (`shared=False`), which this target never builds. Every layer on one state
    must run in one stream order: the workspace is shared and tagged per call, never reset.

    No `is_valid`: the compiled call checks every buffer's dtype, shape and alignment against its signature and
    raises on a mismatch (a misaligned `hidden`, an `out` that is not bf16 [M, 6144], a malformed `arp`).
    """

    def __call__(
        self,
        hidden: torch.Tensor,
        layer: M3MoeLayer,
        out: Optional[torch.Tensor] = None,
        arp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        state = layer.state
        if (state.device, state.m) not in _ENGAGED:
            # The engagement line (once per device and token count, on every rank: the first call, eager or captured).
            _ENGAGED.add((state.device, state.m))
            logger.info(
                f"[M3 sm_107 tp2] MoE engine m3_moe engaged: {state.m} tokens, {state.num_ctas} routed + "
                f"{state.mod.NSH} shared CTAs, {state.mod.STAGES}-stage ring, on {state.device}"
            )
        return state.run(layer, hidden, out, arp=arp)

    def reference(
        self,
        hidden: torch.Tensor,
        layer: M3MoeLayer,
        out: Optional[torch.Tensor] = None,
        arp: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Plain torch over the layer's buffers (see the class docstring); `out` and `arp` are ignored."""
        return moe_reference(hidden, layer, shared=layer.state.shared)


m3_moe_engine = M3MoeEngine()
