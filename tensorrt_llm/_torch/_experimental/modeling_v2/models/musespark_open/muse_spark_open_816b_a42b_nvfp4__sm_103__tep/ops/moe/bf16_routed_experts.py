# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The BF16 experts of a pre-routed decode step with unpacked routing (``bf16_routed_experts``).

The two MoE layers the checkpoint leaves unquantized (0 and 61) run
FlashInfer's TRTLLM-gen BF16 MoE. TRT-LLM's FlashInfer leaf
(``FlashinferOpBackend.run_bf16_moe``) hands it pre-routed ids as one packed
int32 word per (token, slot), ``(id << 16) | bits(bf16 weight)``, which takes
three elementwise launches (the shift, the int16 -> int32 copy of the weight
bits, the or) between the expert placement and the routing-indices kernel.
FlashInfer also takes the ids and the BF16 weights as two tensors
(``RoutingInputMode.UnpackedPrecomputed``, the ``(ids, weights)`` form of
``trtllm_bf16_routed_moe``), which the routing-indices kernel reads directly.
This entry is that call: the same routing-indices, FC1 and FC2 kernels at the
same tactic (FlashInfer's autotuner fallback, i.e. the launcher's default
``[tile_N, config]``, as the built-in call gets), without the three launches.

Output: the unfinalized triple the MuseSpark MoE finalize reads,
``[gemm2_output, expert_weights, expanded_idx_to_permuted_idx]``; the expert
weights are ``topk_weights`` itself (the model's finalize reads the FP32
weights from its routing memo, never these). Every routed FC2 row, read
through ``expanded_idx_to_permuted_idx``, is bitwise the packed call's.

Launch order and PDL: the entry launches the routing-indices kernel, FC1 and
FC2, nothing else, and writes neither ``topk_ids`` nor ``topk_weights``.
Without the packing launches in front of it, the routing-indices kernel follows
the pre-expert work on the stream directly and can be resident before that
work has finished (a programmatic edge). It reads ``topk_ids`` /
``topk_weights`` (both written by ``ring_buddy_assign``) and every other global
input only after its grid-dependency wait, and its trigger comes after that
wait, so FC1 / FC2, which launch behind it, also read ``hidden_states`` only
after the predecessor has completed.
"""

from __future__ import annotations

from typing import List

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Model geometry of the BF16 expert layers (after the pool projection).
HIDDEN = 4096
INTER = 4096
EXPERTS = 256
TOP_K = 8
#: The window the cells use: the tep8 ring-buddy stack ``[own block | predecessor's
#: block]``, which is also a tep4 rank's own block (the owner-only window). The op
#: itself takes any window (``local_num_experts``).
WINDOW = 64
#: BlockMajorK's K block: 128 bytes of BF16.
BLOCK_K = 64


class Bf16RoutedExperts(OpWrapper):
    """FlashInfer's TRTLLM-gen BF16 MoE on pre-routed, unpacked ids and BF16 weights.

    ``hidden_states`` ``[M, 4096]`` BF16; ``gemm1_weights`` / ``gemm2_weights``
    the window's experts in TRTLLM-gen's shuffled BlockMajorK layout
    (``BF16TRTLLMGenFusedMoEMethod``); ``topk_ids`` ``[M, 8]`` int32 in the
    window's id space (an id outside ``[local_expert_offset, local_expert_offset
    + local_num_experts)`` is not run here); ``topk_weights`` ``[M, 8]`` BF16.
    ``routing_method_type`` and ``activation_type`` are FlashInfer's enum
    values (``FlashinferOpBackend.cvt_routing_method_type`` /
    ``cvt_activation_type`` of the leaf's TRT-LLM values).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one decode row, one local expert: the smallest routing table",
            spec=dict(rows=1, active=1),
        ),
        Cell(
            why="DFlash with 1 draft token, 3 local experts",
            spec=dict(rows=2, active=3),
        ),
        Cell(
            why="the K = 7 verify width at the layer-61 critical-rank mode (4 experts)",
            spec=dict(rows=8, active=4),
        ),
        Cell(
            why="the K = 7 verify width at the layer-0 critical-rank mode (6 experts, "
            "2.5 FC1 waves)",
            spec=dict(rows=8, active=6),
        ),
        Cell(
            why="the K = 7 verify width with no local expert: the routing table is empty",
            spec=dict(rows=8, active=0),
        ),
        Cell(
            why="the stacked path's row limit: 16 rows, 9 local experts",
            spec=dict(rows=16, active=9),
        ),
    )

    note = """
    Not bitwise for FP32 ``topk_weights``: FlashInfer accepts them, but its
    routing then produces FP32 expert weights where the packed word carries
    BF16; this entry takes BF16 only (``is_valid``). The tactic is FlashInfer's
    autotuner fallback, i.e. the launcher's default ``[tile_N, config]`` for
    the shape, on every rank, as for the built-in call.
    """

    def __call__(
        self,
        hidden_states: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm2_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routing_method_type: int,
        activation_type: int,
        use_shuffled_weight: bool,
        weight_layout: int,
        num_experts: int = EXPERTS,
        top_k: int = TOP_K,
    ) -> List[torch.Tensor]:
        from flashinfer.fused_moe import trtllm_bf16_routed_moe

        return trtllm_bf16_routed_moe(
            (topk_ids, topk_weights),
            hidden_states,
            gemm1_weights,
            gemm2_weights,
            num_experts,
            top_k,
            None,
            None,
            intermediate_size,
            local_expert_offset,
            local_num_experts,
            routed_scaling_factor=None,
            routing_method_type=routing_method_type,
            use_shuffled_weight=use_shuffled_weight,
            weight_layout=weight_layout,
            do_finalize=False,
            enable_pdl=None,
            activation_type=activation_type,
        )

    def reference(
        self,
        hidden_states: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm2_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routing_method_type: int,
        activation_type: int,
        use_shuffled_weight: bool,
        weight_layout: int,
        num_experts: int = EXPERTS,
        top_k: int = TOP_K,
    ) -> torch.Tensor:
        """Every local slot's expert output, ``[M * top_k, hidden]`` (other rows zero), in FP32.

        Undoes the BlockMajorK blocking and TRTLLM-gen's row shuffle of the
        weights (``BF16TRTLLMGenFusedMoEMethod``), then
        ``(silu(x W1^T) * (x W3^T)) W2^T`` per local slot, in FP32 with the
        SwiGLU output rounded to BF16 as the FC2 input (the op feeds FC2 a BF16
        intermediate). The rows are indexed ``t * top_k + k``, i.e. already
        through ``expanded_idx_to_permuted_idx``. Ignores ``topk_weights``
        (the unfinalized rows are unweighted), ``routing_method_type``,
        ``num_experts`` and ``use_shuffled_weight`` / ``weight_layout`` beyond
        the layout this target stores (shuffled BlockMajorK).
        """
        w13, w2 = _unshuffled(gemm1_weights, gemm2_weights)
        x = hidden_states.float()
        rows = torch.zeros(
            (hidden_states.shape[0] * top_k, w2.shape[1]),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        for t in range(hidden_states.shape[0]):
            for k in range(top_k):
                e = int(topk_ids[t, k]) - local_expert_offset
                if not 0 <= e < local_num_experts:
                    continue
                h = w13[e].float() @ x[t]
                up, gate = h[:intermediate_size], h[intermediate_size:]
                act = (gate * torch.sigmoid(gate) * up).to(torch.bfloat16).float()
                rows[t * top_k + k] = w2[e].float() @ act
        return rows

    def is_valid(
        self,
        hidden_states: torch.Tensor,
        gemm1_weights: torch.Tensor,
        gemm2_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        intermediate_size: int,
        local_expert_offset: int,
        local_num_experts: int,
        routing_method_type: int,
        activation_type: int,
        use_shuffled_weight: bool,
        weight_layout: int,
        num_experts: int = EXPERTS,
        top_k: int = TOP_K,
    ) -> None:
        """Inputs the op takes and answers with other bits than the packed call."""
        assert topk_weights.dtype == torch.bfloat16, (
            "topk_weights must be BF16: FP32 weights change the routing's weight output and "
            "are not the packed call's"
        )
        assert topk_ids.dtype == torch.int32 and topk_ids.is_contiguous(), (
            "topk_ids must be contiguous int32"
        )
        assert topk_weights.is_contiguous() and topk_weights.shape == topk_ids.shape, (
            "topk_weights must be contiguous and shaped like topk_ids"
        )
        assert topk_ids.shape == (hidden_states.shape[0], top_k), (
            f"topk_ids must be [M, {top_k}], got {tuple(topk_ids.shape)}"
        )
        window_end = local_expert_offset + local_num_experts
        assert 0 <= local_expert_offset and window_end <= num_experts, (
            "the local window must lie inside the expert range"
        )


def _unshuffled(gemm1_weights: torch.Tensor, gemm2_weights: torch.Tensor):
    """The ``[E, 2I, H]`` / ``[E, H, I]`` weights from TRTLLM-gen's shuffled BlockMajorK layout."""
    from tensorrt_llm._torch.moe.fused_moe.quantization import BF16TRTLLMGenFusedMoEMethod

    method = BF16TRTLLMGenFusedMoEMethod()

    def unblock(w):
        # [E, K / 64, R, 64] -> [E, R, K]
        e, kb, r, bk = w.shape
        return w.permute(0, 2, 1, 3).reshape(e, r, kb * bk)

    w13 = unblock(gemm1_weights)
    w2 = unblock(gemm2_weights)
    # The permutations depend only on the shapes (the method caches them by shape).
    perm13 = method._get_w3_w1_permute_indices(
        torch.empty(w13.shape[1:], dtype=w13.dtype, device=w13.device)
    ).to(w13.device)
    perm2 = method._get_w2_permute_indices(
        torch.empty(w2.shape[1:], dtype=w2.dtype, device=w2.device)
    ).to(w2.device)
    return w13[:, torch.argsort(perm13)], w2[:, torch.argsort(perm2)]


bf16_routed_experts = Bf16RoutedExperts()
