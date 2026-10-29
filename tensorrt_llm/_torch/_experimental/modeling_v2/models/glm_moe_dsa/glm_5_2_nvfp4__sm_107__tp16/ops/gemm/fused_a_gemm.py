# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GLM-5 fused-A GEMM (kv_a_proj_with_mqa, optionally with the DSA indexer's wk | weights_proj
rows) for up to 8 tokens, with an L2 prefetch of later weights, via the trtllm glm_fused_a_gemm
op."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)

HIDDEN_SIZE = 6144
# kv_a_proj_with_mqa: q_a (2048) | compressed KV (512) | k_pe (64).
OUT_FEATURES = 2624
# The same rows, then the indexer's wk (128) and weights_proj (32) rows.
INDEXER_OUT_FEATURES = 2784
ROWS_PER_CTA = 16
MAX_TOKENS = 8

# What a c1 decode step lists behind fused-A, at TP16 per-rank shapes.
_INDEXER_WQ_B = ((4096, 2048), torch.bfloat16)  # indexer wq_b, q_a norm folded in (not TP-split)
_Q_B_ABSORBED = ((2304, 2048), torch.bfloat16)  # q_b with k_b absorbed: 4 heads x (512 + 64)
_V_B = ((4, 256, 512), torch.bfloat16)  # v_b_proj, 4 heads
_O_PROJ = ((6144, 1024), torch.bfloat16)  # o_proj shard
_LEADER_TOPK = ((8, 2048), torch.int32)  # the indexer layer's top-k that a shared layer reuses


class GlmFusedAGemm(OpWrapper):
    """`hidden_states @ weight.T`: BF16 operands, FP32 accumulation, BF16 out, `[tokens, rows]`.

    `weight` is the kv_a_proj_with_mqa weight as stored, `[2624, 6144]` row-major (q_a, compressed
    KV, k_pe), or one tensor with the DSA indexer's wk and weights_proj rows appended,
    `[2784, 6144]`. Output columns then read `[0, 2048)` q_a, `[2048, 2560)` compressed KV,
    `[2560, 2624)` k_pe, `[2624, 2752)` indexer K, `[2752, 2784)` indexer weights; columns
    `[0, 2624)` are bitwise those of the 2624-row call.

    One CTA per 16 weight rows (164 or 174 CTAs, one per SM) with all eight warps issuing the
    HMMAs. Before its grid-dependency wait the kernel stages the CTA's weight rows (tensor copies
    into shared memory, two thirds of them on into registers) and then prefetches the
    `l2_prefetch` tensors into L2; it reads `hidden_states` only after the wait and triggers its
    dependents right after the wait.
    """

    ARCHS = frozenset({Arch.SM_107})

    # The c1 decode step runs the op 85 times: 78 verify layers at 8 rows plus 7 MTP draft steps.
    # 22 calls (21 indexer layers and draft step 0) take the 2784-row weight.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="indexer layers' verify step and MTP draft step 0: 2784 rows, 8 tokens (3D "
            "activation copies), the list indexer wq_b, v_b, o_proj",
            spec=dict(tokens=8, out_features=2784, l2_prefetch=(_INDEXER_WQ_B, _V_B, _O_PROJ)),
        ),
        Cell(
            why="shared-top-k layers' verify step (57 of 78): 2624 rows, 8 tokens, a full list "
            "of four ending in the indexer layer's int32 top-k",
            spec=dict(
                tokens=8,
                out_features=2624,
                l2_prefetch=(_Q_B_ABSORBED, _V_B, _O_PROJ, _LEADER_TOPK),
            ),
        ),
        Cell(
            why="MTP draft steps 1-6: 2624 rows at one token (2D activation boxes of one row)",
            spec=dict(tokens=1, out_features=2624, l2_prefetch=(_Q_B_ABSORBED, _V_B, _O_PROJ)),
        ),
        Cell(
            why="4 tokens (MTP with 3 draft tokens): 2D boxes between 1 and 8 rows, 2784 rows",
            spec=dict(tokens=4, out_features=2784, l2_prefetch=(_INDEXER_WQ_B, _V_B, _O_PROJ)),
        ),
        Cell(
            why="empty list: the instance without L2 prefetches, for more than 32 rows per step",
            spec=dict(tokens=8, out_features=2624, l2_prefetch=()),
        ),
    )

    note = """
    `l2_prefetch` is a cache hint: up to four contiguous, 16-byte aligned CUDA tensors on the
    GEMM's device, issued in 16 KB bulk prefetches spread over the CTAs once each CTA's weight
    rows are staged. It never enters the output, and the kernel writes none of it.

    PDL (`TRTLLM_ENABLE_PDL`): before its grid-dependency wait the kernel reads `weight`, so no
    kernel launched earlier in the step may write it (load-time weights only). Its dependents
    launch once it passes the wait and must wait on their own grid dependency before reading the
    output or writing `hidden_states`.

    Preconditions, enforced loudly by the op: CUDA BF16 2-D `hidden_states` and `weight` on one
    device, contiguous and 16-byte aligned; hidden 6144; 2624 or 2784 rows; 1 to 8 tokens; at most
    four `l2_prefetch` tensors, each contiguous, aligned and on the device; SM90 or later.

    Numerics: the products of a plain GEMM, summed in FP32 in the kernel's order (two
    accumulators per warp, the eight warp partials in warp order) and rounded to BF16 once;
    deterministic. The shared `dsv3_fused_a_gemm_op` sums the same products in another order:
    nearly all outputs agree bitwise, the rest by one BF16 rounding step.
    """

    def __call__(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        l2_prefetch: list[torch.Tensor],
    ) -> torch.Tensor:
        return torch.ops.trtllm.glm_fused_a_gemm(hidden_states, weight, l2_prefetch)

    def reference(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        l2_prefetch: list[torch.Tensor],
    ) -> torch.Tensor:
        """`hidden_states @ weight.T` in FP32 (TF32 off), rounded to BF16 once.

        `l2_prefetch` does not enter the result; it keeps its name to mirror `__call__`.
        """
        with true_fp32_matmul():
            return (hidden_states.float() @ weight.float().t()).to(hidden_states.dtype)

    def is_valid(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        l2_prefetch: list[torch.Tensor],
    ) -> None:
        # The shape gates of the kernel, which a target checks to route anything else to the
        # shared dsv3_fused_a_gemm_op. The op raises on all but the last of them as well.
        # gemm_k 6144: 24 k stages of 256, three per MMA warp.
        assert hidden_states.dim() == 2 and hidden_states.shape[1] == HIDDEN_SIZE, (
            f"hidden_states must be [tokens, {HIDDEN_SIZE}]"
        )
        # Tiles of 16 weight rows, one CTA each; instances exist for the 2624-row weight and the
        # 2784-row weight with the indexer rows.
        assert weight.dim() == 2 and weight.shape[1] == HIDDEN_SIZE, (
            f"weight must be [rows, {HIDDEN_SIZE}]"
        )
        assert weight.shape[0] in (OUT_FEATURES, INDEXER_OUT_FEATURES), (
            f"weight must have {OUT_FEATURES} or {INDEXER_OUT_FEATURES} rows, got {weight.shape[0]}"
        )
        # One activation tile of 8 token rows per k stage.
        assert 1 <= hidden_states.shape[0] <= MAX_TOKENS, (
            f"the kernel takes 1 to {MAX_TOKENS} tokens, got {hidden_states.shape[0]}"
        )
        # One CTA per SM in a single wave: the kernel stages its weights before the
        # grid-dependency wait, and a second wave of CTAs would start only after the first ends.
        # The output stays right; on fewer SMs the caller takes the shared dsv3_fused_a_gemm_op
        # instead (32-row tiles, half the CTAs, for the 2624-row weight).
        sms = torch.cuda.get_device_properties(hidden_states.device).multi_processor_count
        ctas = weight.shape[0] // ROWS_PER_CTA
        assert sms >= ctas, f"{ctas} CTAs need as many SMs for one wave, the device has {sms} SMs"


fused_a_gemm = GlmFusedAGemm()
