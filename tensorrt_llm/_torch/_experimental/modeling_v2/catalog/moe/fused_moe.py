# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fused mixture-of-experts layer: expert permutation + grouped FC1/FC2 GEMMs
with a gated activation between them + routing-weighted combine, in one call."""

from typing import List, Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*


def fused_moe(
    input: torch.Tensor,
    token_selected_experts: torch.Tensor,
    token_final_scales: Optional[torch.Tensor],
    fc1_expert_weights: torch.Tensor,
    fc1_expert_biases: Optional[torch.Tensor],
    fc2_expert_weights: torch.Tensor,
    fc2_expert_biases: Optional[torch.Tensor],
    output_dtype: torch.dtype,
    quant_scales: List[torch.Tensor],
    input_sf: Optional[torch.Tensor] = None,
    swizzled_input_sf: bool = True,
    swiglu_alpha: Optional[torch.Tensor] = None,
    swiglu_beta: Optional[torch.Tensor] = None,
    swiglu_limit: Optional[torch.Tensor] = None,
    tp_size: int = 1,
    tp_rank: int = 0,
    ep_size: int = 1,
    ep_rank: int = 0,
    cluster_size: int = 1,
    cluster_rank: int = 0,
    enable_alltoall: bool = False,
    use_deepseek_fp8_block_scale: bool = False,
    use_w4_group_scaling: bool = False,
    use_int8_woq_per_channel: bool = False,
    use_mxfp8_act_scaling: bool = False,
    min_latency_mode: bool = False,
    use_fused_finalize: bool = True,
    tune_max_num_tokens: int = 8192,
    tuner_num_tokens: Optional[int] = None,
    tuner_top_k: Optional[int] = None,
    activation_type: int = 5,  # ActivationType.Swiglu
    unpadded_hidden_size: Optional[int] = None,
    out_tensor: Optional[torch.Tensor] = None,
    use_dynamic_fc2_scale: bool = False,
    use_mxfp8_weight_scaling: bool = False,
    fc1_lora_ranks: Optional[torch.Tensor] = None,
    fc1_lora_weight_ptrs: Optional[torch.Tensor] = None,
    fc2_lora_ranks: Optional[torch.Tensor] = None,
    fc2_lora_weight_ptrs: Optional[torch.Tensor] = None,
    gated_lora_ranks: Optional[torch.Tensor] = None,
    gated_lora_weight_ptrs: Optional[torch.Tensor] = None,
    host_request_types: Optional[torch.Tensor] = None,
    host_context_lengths: Optional[torch.Tensor] = None,
    lora_max_low_rank: int = 0,
    fc1_slot_lora_ranks: Optional[torch.Tensor] = None,
    fc1_slot_lora_weight_ptrs: Optional[torch.Tensor] = None,
    fc2_slot_lora_ranks: Optional[torch.Tensor] = None,
    fc2_slot_lora_weight_ptrs: Optional[torch.Tensor] = None,
    gated_slot_lora_ranks: Optional[torch.Tensor] = None,
    gated_slot_lora_weight_ptrs: Optional[torch.Tensor] = None,
    token_to_slot: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """Run one MoE layer over pre-routed tokens.

    Returns `[out]` with `out` a fresh `[num_tokens, hidden_size]` tensor in
    `output_dtype`, or `[]` when `out_tensor` is given (written in place).
    """
    # Pure-metadata guard: on the unquantized high-precision path the store
    # is done in the activation dtype while the output tensor is allocated
    # with output_dtype, so a mismatch reinterprets the written bits — observed
    # on this machine to return plausible-looking wrong values, never to raise.
    if input.dtype in (torch.bfloat16, torch.float16) and (fc1_expert_weights.dtype == input.dtype):
        assert output_dtype == input.dtype, (
            f"output_dtype ({output_dtype}) must equal input.dtype ({input.dtype}) "
            "on the unquantized path; a mismatch is written as raw activation-dtype "
            "bits into an output_dtype buffer and silently gives wrong values"
        )
    return torch.ops.trtllm.fused_moe(
        input,
        token_selected_experts,
        token_final_scales,
        fc1_expert_weights,
        fc1_expert_biases,
        fc2_expert_weights,
        fc2_expert_biases,
        output_dtype,
        quant_scales,
        input_sf=input_sf,
        swizzled_input_sf=swizzled_input_sf,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        cluster_size=cluster_size,
        cluster_rank=cluster_rank,
        enable_alltoall=enable_alltoall,
        use_deepseek_fp8_block_scale=use_deepseek_fp8_block_scale,
        use_w4_group_scaling=use_w4_group_scaling,
        use_int8_woq_per_channel=use_int8_woq_per_channel,
        use_mxfp8_act_scaling=use_mxfp8_act_scaling,
        min_latency_mode=min_latency_mode,
        use_fused_finalize=use_fused_finalize,
        tune_max_num_tokens=tune_max_num_tokens,
        tuner_num_tokens=tuner_num_tokens,
        tuner_top_k=tuner_top_k,
        activation_type=activation_type,
        unpadded_hidden_size=unpadded_hidden_size,
        out_tensor=out_tensor,
        use_dynamic_fc2_scale=use_dynamic_fc2_scale,
        use_mxfp8_weight_scaling=use_mxfp8_weight_scaling,
        fc1_lora_ranks=fc1_lora_ranks,
        fc1_lora_weight_ptrs=fc1_lora_weight_ptrs,
        fc2_lora_ranks=fc2_lora_ranks,
        fc2_lora_weight_ptrs=fc2_lora_weight_ptrs,
        gated_lora_ranks=gated_lora_ranks,
        gated_lora_weight_ptrs=gated_lora_weight_ptrs,
        host_request_types=host_request_types,
        host_context_lengths=host_context_lengths,
        lora_max_low_rank=lora_max_low_rank,
        fc1_slot_lora_ranks=fc1_slot_lora_ranks,
        fc1_slot_lora_weight_ptrs=fc1_slot_lora_weight_ptrs,
        fc2_slot_lora_ranks=fc2_slot_lora_ranks,
        fc2_slot_lora_weight_ptrs=fc2_slot_lora_weight_ptrs,
        gated_slot_lora_ranks=gated_slot_lora_ranks,
        gated_slot_lora_weight_ptrs=gated_slot_lora_weight_ptrs,
        token_to_slot=token_to_slot,
    )
