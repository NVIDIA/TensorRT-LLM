# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 MoE layer entered with router logits, on TRT-LLM's CuTe DSL Blackwell
MoE kernels: Renormalize routing (`renorm_moe_routing_op`), the token sort,
a gathered grouped FC1 GEMM with SwiGLU and NVFP4 requantization in its
epilogue, and a grouped FC2 GEMM that scatter-adds the routing-weighted rows
into the output -- the layer `fp4_block_scale_moe_renorm_routed` computes, for
prefill-sized row counts, from a second copy of the experts in this kernel
family's layout."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from .fp4_block_scale_moe_renorm_routed import CERTIFIED_PAIRS, _ckpt
from .fp4_block_scale_moe_renorm_routed import fp4_block_scale_moe_renorm_routed as _routed
from .renorm_moe_routing_op import renorm_moe_routing_op

#: Token tile of the grouped GEMMs (the kernels' M tile).
TILE_TOKENS = 128
_FP4 = torch.float4_e2m1fn_x2


class Fp4MoeRenormCuteDsl(OpWrapper):
    """One NVFP4 MoE layer whose routing happens inside the call, on the CuTe
    DSL MoE kernels. Same function as `fp4_block_scale_moe_renorm_routed`:
    per token the `top_k` largest logits (ties to the LOWER expert index),
    weighted by an fp32 softmax over just those `top_k` rounded to bf16; the
    experts' SwiGLU FC1, the NVFP4 requantization of its output, FC2, and the
    weighted sum. Returns the combined `[T, H]` bf16 rows.

    Operands, in this kernel family's layout (see `weights.py`): `fc1_w`
    `[E, 2I, H/2]` uint8, `[up ; gate]` rows interleaved in 64-row groups;
    `fc1_s` `[E, 2I, H/16]` e4m3 bytes in that row order, 128x4-swizzled;
    `fc2_w` `[E, H, I/2]` as stored; `fc2_s` `[E, H, I/16]` swizzled.
    `hidden_states` / `hidden_states_scale` are the NVFP4 activation with
    **linear** block scales, quantized with the layer's FC1 activation global
    scale. Scalars: `fc1_alpha` `[E]` dequantizes the FC1 accumulator (the
    routed entry's `output1_scale_gate_scalar`); `fc2_input_scale` `[1]` is the
    FC2 activation's quantization multiplier (the reciprocal of its dequant
    scale), one number for every expert; `fc2_alpha` `[E]` dequantizes the FC2
    accumulator (the routed entry's `output2_scale_scalar`).
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="4096 rows: where the target switches to this entry; the sort's cluster kernel",
            spec=_ckpt(4096),
        ),
        Cell(
            why="16377 rows: a con=128 mixed step (16256 context + 121 generation)",
            spec=_ckpt(16377),
        ),
        Cell(
            why="16384 rows: max_num_tokens of the tuned config; the sort's cooperative kernel",
            spec=_ckpt(16384),
        ),
    )

    note = """
    Semantics against `fp4_block_scale_moe_renorm_routed` (trtllm-gen) on the
    same experts: the same routing (both are the `renorm_moe_routing_op`
    function, weights rounded to bf16 once) and the same NVFP4 arithmetic per
    expert (e2m1 x e4m3-block x fp32-alpha products accumulated in fp32, SwiGLU
    on the FC1 accumulator, the FC2 input requantized to NVFP4 with 16-element
    blocks), but not bitwise: the GEMMs' accumulation order differs, and the
    weighted sum over a token's experts is a scatter-add with fp32 atomics, so
    its order -- and the output's last bit -- varies between launches. Both sit
    inside the routed entry's band against its fp32 reference.

    One FC2 activation scale for all experts: the routed entry takes it per
    expert (folded into `output1_scale_scalar`); a checkpoint whose experts
    disagree on `down_proj.input_scale` cannot use this entry as is.

    Kernels (all TRT-LLM's): `trtllm::moe_sort` (tile metadata),
    `trtllm::moe_output_memset_inplace` (zeroes the rows the scatter-add
    touches), `trtllm::cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell`,
    `trtllm::cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell`; the two
    GEMMs pick their tactics through the AutoTuner. Not CUDA-graph certified
    (prefill steps run eagerly).

    Hard limits
      T            certified 4096..16384 (the target calls it from
                   `_CUTE_DSL_MOE_MIN_ROWS` = 4096 rows)
      num_experts  512, top_k 10 (the routed entry's CERTIFIED_PAIRS)
    """

    def __call__(
        self,
        routing_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        fc1_w: torch.Tensor,
        fc1_s: torch.Tensor,
        fc1_alpha: torch.Tensor,
        fc2_w: torch.Tensor,
        fc2_s: torch.Tensor,
        fc2_alpha: torch.Tensor,
        fc2_input_scale: torch.Tensor,
        num_experts: int,
        top_k: int,
    ) -> torch.Tensor:
        tokens = hidden_states.shape[0]
        hidden = hidden_states.shape[1] * 2
        ids, weights = renorm_moe_routing_op(routing_logits, top_k, torch.bfloat16)
        scales = weights.float()
        t2e, t2lim, e2p, p2e, _, live = torch.ops.trtllm.moe_sort(
            token_selected_experts=ids,
            token_final_scales=scales,
            num_experts=num_experts,
            top_k=top_k,
            local_expert_offset=0,
            local_num_experts=num_experts,
            tile_tokens_dim=TILE_TOKENS,
        )
        out = torch.empty((tokens, hidden), dtype=torch.bfloat16, device=hidden_states.device)
        torch.ops.trtllm.moe_output_memset_inplace(
            input=out,
            tile_idx_to_mn_limit=t2lim,
            expanded_idx_to_permuted_idx=e2p,
            permuted_idx_to_expanded_idx=p2e,
            num_non_exiting_tiles=live,
            tile_tokens_dim=TILE_TOKENS,
            top_k=top_k,
            ep_size=1,
            enable_alltoall=False,
        )
        h, h_sf = torch.ops.trtllm.cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell(
            input=hidden_states.view(_FP4),
            weight=fc1_w.view(_FP4),
            input_scale=hidden_states_scale.view(torch.uint8).view(tokens, hidden // 16),
            weight_scale=fc1_s.view(torch.uint8),
            alpha=fc1_alpha,
            tile_idx_to_group_idx=t2e,
            tile_idx_to_mn_limit=t2lim,
            permuted_idx_to_expanded_idx=p2e,
            num_non_exiting_tiles=live,
            global_sf=fc2_input_scale,
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_experts,
            local_expert_offset=0,
            tile_size=TILE_TOKENS,
        )
        torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell(
            input=h.view(_FP4),
            weight=fc2_w.view(_FP4),
            input_scale=h_sf.view(torch.uint8),
            weight_scale=fc2_s.view(torch.uint8),
            alpha=fc2_alpha,
            output=out,
            tile_idx_to_group_idx=t2e,
            tile_idx_to_mn_limit=t2lim,
            permuted_idx_to_expanded_idx=p2e,
            num_non_exiting_tiles=live,
            token_final_scales=scales,
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_experts,
            local_expert_offset=0,
            tile_size=TILE_TOKENS,
            output_dtype=torch.bfloat16,
        )
        return out

    def reference(
        self,
        routing_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        top_k: int,
        up: tuple,
        gate: tuple,
        down: tuple,
        fc2_global_scale: float,
    ) -> torch.Tensor:
        """The routed entry's fp32 reference (routing, then the layer over the
        dequantized `(codes, scales)` per-expert operands); fp32 `[T, H]`."""
        return _routed.reference(
            routing_logits, hidden_states, top_k, up, gate, down, fc2_global_scale
        )

    def is_valid(
        self,
        routing_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        hidden_states_scale: torch.Tensor,
        fc1_w: torch.Tensor,
        fc1_s: torch.Tensor,
        fc1_alpha: torch.Tensor,
        fc2_w: torch.Tensor,
        fc2_s: torch.Tensor,
        fc2_alpha: torch.Tensor,
        fc2_input_scale: torch.Tensor,
        num_experts: int,
        top_k: int,
    ) -> None:
        for name, tensor in (
            ("routing_logits", routing_logits),
            ("hidden_states", hidden_states),
            ("hidden_states_scale", hidden_states_scale),
            ("fc1_w", fc1_w),
            ("fc1_s", fc1_s),
            ("fc2_w", fc2_w),
            ("fc2_s", fc2_s),
        ):
            assert tensor.is_contiguous(), f"{name} must be contiguous"
        assert (num_experts, top_k) in CERTIFIED_PAIRS, (
            f"(num_experts, top_k) = ({num_experts}, {top_k}) is not certified"
        )
        tokens, half_hidden = hidden_states.shape
        assert hidden_states_scale.numel() == tokens * half_hidden * 2 // 16, (
            "hidden_states_scale must hold one linear e4m3 scale a 16-element block"
        )
        assert fc1_alpha.shape == (num_experts,) and fc2_alpha.shape == (num_experts,), (
            "per-expert alphas"
        )
        assert fc2_input_scale.numel() == 1, "one FC2 activation scale for every expert"


fp4_moe_renorm_cute_dsl = Fp4MoeRenormCuteDsl()
