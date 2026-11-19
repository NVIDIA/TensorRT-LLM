# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 MoE head above the MoE front's rows: router logits and routed latent in one launch."""

from typing import Optional, Tuple

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

HIDDEN = 7168
ROUTER_ROWS = 896
LATENT_ROWS = 3584
# K tiles (128 wide) of the router's first K half: cuBLAS splits the FP32 router GEMM into two equal K halves here.
K_SPLIT_TILES = HIDDEN // 128 // 2
# Rows the kernel serves and its (token tile, A/B stages, latent in two K halves) per row range: where cuBLAS
# splits the router GEMM's K in two halves, and the latent down projection's K in the same two halves (65-96 rows) or
# not at all (97-176 rows). GB300, cold weights, us per layer against the two-stream router | latent cuBLAS head:
# 65-96 rows 17.9-18.3 -> 13.4-14.0; 97-128 17.6-18.0 -> 13.4-13.7; 129-160 18.1-18.9 -> 14.2-14.3; 161-176 18.4-18.8
# -> 14.4.
HEAD_CONFIGS: Tuple[Tuple[int, Tuple[int, int, bool]], ...] = (
    (96, (64, 9, True)),
    (128, (64, 9, False)),
    (160, (80, 8, False)),
    (176, (96, 8, False)),
)
MIN_ROWS = 65
MAX_ROWS = HEAD_CONFIGS[-1][0]


def head_config(rows: int) -> Optional[Tuple[int, int, bool]]:
    """The kernel config serving ``rows`` tokens, or None outside ``[MIN_ROWS, MAX_ROWS]``."""
    if not MIN_ROWS <= rows <= MAX_ROWS:
        return None
    return next(config for bound, config in HEAD_CONFIGS if rows <= bound)


class K31MoeHeadGemm(OpWrapper):
    """``(logits, latent)`` of ``hidden_states [M, 7168]`` BF16 against the stacked head weight ``[896 + 3584, 7168]``
    BF16 (router rows first): ``logits = hidden_states @ head_weight[:896].T`` FP32 ``[M, 896]`` and ``latent =
    hidden_states @ head_weight[896:].T`` BF16 ``[M, 3584]``, one launch of
    ``kimi_k31_pro_mxfp4__sm_103__tep/kernels/moe/moe_head_gemm``.

    The logits are the sum of the two K halves' FP32 products; the latent is the whole K accumulated in order, or at
    65-96 rows the sum of the same two K halves, rounded once: at ``MIN_ROWS``-``MAX_ROWS`` rows these are the bits
    of the router (``dsv3_router_gemm_op``, cuBLAS with FP32 output) and of the latent down projection
    (``k31_fused_a_gemm``, cuBLAS) that the head runs otherwise.
    """

    ARCHS = frozenset({Arch.SM_103})
    CELLS: Tuple[Cell, ...] = tuple(
        Cell(
            why=f"TEP16 MoE head above the front's rows, {rows} verify rows ({why})",
            spec=dict(rows=rows),
        )
        for rows, why in (
            (65, "the first row served, latent in two K halves"),
            (85, "B17 x 5"),
            (96, "the last row with the latent in two K halves"),
            (97, "the first row with the latent's whole K"),
            (100, "B20 x 5"),
            (128, "the last 64-token tile row"),
            (131, "surge B26 x 5"),
            (145, "surge B29 x 5"),
            (160, "the last 80-token tile row"),
            (176, "the last row served"),
        )
    )
    note = """
    Serves ``head_config(rows)`` rows only (None elsewhere: the caller keeps its two-GEMM head). ``hidden_states``
    and ``head_weight`` contiguous and 32-byte aligned. The kernel JIT-compiles per config on its first call, which
    must come before CUDA-graph capture (the engine's warmup forward). ``pdl`` launches with programmatic dependent
    launch (the token loads wait on the previous grid).
    """

    def __call__(
        self, hidden_states: torch.Tensor, head_weight: torch.Tensor, pdl: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        from ...kernels.moe.moe_head_gemm import run_moe_head

        rows = hidden_states.shape[0]
        config = head_config(rows)
        if config is None:
            raise ValueError(f"the MoE head kernel serves {MIN_ROWS}-{MAX_ROWS} rows, got {rows}")
        logits = torch.empty(rows, ROUTER_ROWS, dtype=torch.float32, device=hidden_states.device)
        latent = torch.empty(rows, LATENT_ROWS, dtype=torch.bfloat16, device=hidden_states.device)
        run_moe_head(hidden_states, head_weight, logits, latent, config, pdl, K_SPLIT_TILES)
        return logits, latent

    def reference(
        self, hidden_states: torch.Tensor, head_weight: torch.Tensor, pdl: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The two-GEMM head: router logits through ``dsv3_router_gemm_op``, latent through ``k31_fused_a_gemm``."""
        from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.gemm.fused_a_gemm import (
            k31_fused_a_gemm,
        )
        from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.dsv3_router_gemm_op import (
            dsv3_router_gemm_op,
        )

        logits = dsv3_router_gemm_op(hidden_states, head_weight[:ROUTER_ROWS].t())
        return logits, k31_fused_a_gemm(hidden_states, head_weight[ROUTER_ROWS:].t())

    def is_valid(
        self, hidden_states: torch.Tensor, head_weight: torch.Tensor, pdl: bool = True
    ) -> None:
        assert head_config(hidden_states.shape[0]) is not None, "rows outside the kernel's range"
        assert hidden_states.dtype == torch.bfloat16 and hidden_states.shape[1] == HIDDEN
        assert head_weight.dtype == torch.bfloat16 and tuple(head_weight.shape) == (
            ROUTER_ROWS + LATENT_ROWS,
            HIDDEN,
        )
        for t in (hidden_states, head_weight):
            assert t.is_contiguous() and t.data_ptr() % 32 == 0, (
                "operands must be contiguous, 32-byte aligned"
            )


k31_moe_head_gemm = K31MoeHeadGemm()
