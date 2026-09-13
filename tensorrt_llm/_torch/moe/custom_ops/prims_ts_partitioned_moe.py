# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PrimsTS BF16 and localized MoE launches with explicit weight ownership."""

from dataclasses import replace
from functools import lru_cache

import torch

from ...autotuner import AutoTuner
from ...custom_ops.trtllm_gen_custom_ops import FP4BlockScaleMoERunner, prepare_dummy_topk_and_hook
from ...locality_domain.runtime import LocalityDomainRuntime
from .prims_ts_moe import PrimsTSMoERunner, _with_routing_profile_hook, _workspace_rows


@lru_cache(maxsize=None)
def _runtime(device_index: int) -> LocalityDomainRuntime:
    # Runtime resources are already keyed by device; retaining this lightweight
    # wrapper gives tuning and inference the same topology identity.
    return LocalityDomainRuntime(2)


@torch.library.custom_op("trtllm::prims_ts_partitioned_moe", mutates_args=("output",))
def prims_ts_partitioned_moe(
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor | None,
    gemm1_weights: list[torch.Tensor],
    gemm1_weights_scale: list[torch.Tensor],
    gemm2_weights: list[torch.Tensor],
    gemm2_weights_scale: list[torch.Tensor],
    gemm1_alpha: torch.Tensor | None,
    gemm1_beta: torch.Tensor | None,
    gemm1_clamp_limit: torch.Tensor | None,
    output1_scale_scalar: torch.Tensor | None,
    output1_scale_gate_scalar: torch.Tensor | None,
    output2_scale_scalar: torch.Tensor | None,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    local_expert_offset: int,
    activation_type: int,
    do_finalize: bool,
    output: torch.Tensor,
    tune_max_num_tokens: int = 8192,
    use_dp: bool = False,
    routing_method_type: int = 0,
    n_group: int | None = None,
    topk_group: int | None = None,
    routed_scaling_factor: float | None = None,
    routing_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_partitions = len(gemm1_weights)
    if num_partitions not in (1, 2) or len(gemm2_weights) != num_partitions:
        raise ValueError("PrimsTS requires one or two matching weight partitions")
    is_bf16 = hidden_states.dtype == torch.bfloat16
    if not is_bf16 and hidden_states.dtype != torch.uint8:
        raise ValueError("Partitioned PrimsTS supports BF16 or NVFP4")
    if not is_bf16 and (
        len(gemm1_weights_scale) != num_partitions or len(gemm2_weights_scale) != num_partitions
    ):
        raise ValueError("NVFP4 requires scales for every weight partition")
    if is_bf16 and (gemm1_weights_scale or gemm2_weights_scale):
        raise ValueError("BF16 weights do not use block scales")
    if hidden_states.shape[0] == 0:
        return (
            hidden_states.new_empty((0, output.shape[1]), dtype=torch.bfloat16),
            topk_ids.new_empty((0, topk_ids.shape[1]), dtype=torch.int32),
            topk_weights.new_empty((0,)),
        )

    local_experts = gemm1_weights[0].shape[0]
    top_k = topk_ids.shape[1]
    tuning_config = FP4BlockScaleMoERunner.get_tuning_config(
        num_experts // local_experts, tune_max_num_tokens, use_dp
    )
    if is_bf16:
        tuning_config = replace(
            tuning_config,
            constraint_specs=tuple(s for s in tuning_config.constraint_specs if s.input_idx != 3),
        )
    if num_partitions > 1:
        # Cold-L2 profiling clones tensors into ordinary memory, destroying
        # the localized placement whose concurrent performance we must tune.
        tuning_config = replace(tuning_config, use_cold_l2_cache=False)
    tune_logits, tune_weights, tune_ids, tuning_config = prepare_dummy_topk_and_hook(
        topk_weights,
        topk_ids,
        hidden_states,
        None,
        routing_method_type,
        tuning_config,
        top_k,
        num_experts,
        local_experts,
        n_group,
        topk_group,
        routed_scaling_factor,
        local_expert_offset=local_expert_offset,
        use_dp=use_dp,
        routing_bias=routing_bias,
    )
    tuning_config = _with_routing_profile_hook(tuning_config, tune_weights, tune_ids)
    sf1 = [None] * num_partitions if is_bf16 else gemm1_weights_scale
    sf2 = [None] * num_partitions if is_bf16 else gemm2_weights_scale
    inputs = [
        tune_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights[0],
        sf1[0],
        None,
        gemm1_alpha,
        gemm1_beta,
        gemm1_clamp_limit,
        gemm2_weights[0],
        sf2[0],
        None,
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        tune_weights,
        tune_ids,
    ]
    runner = PrimsTSMoERunner(
        num_experts,
        local_expert_offset,
        activation_type,
        do_finalize,
        False,
        use_dp,
        top_k,
        routing_method_type,
        n_group,
        topk_group,
        routed_scaling_factor,
    )
    if num_partitions > 1:
        runner.locality_runtime = _runtime(hidden_states.device.index)
        runner.locality_weights = tuple(zip(gemm1_weights, sf1, gemm2_weights, sf2))
    runner, tactic = AutoTuner.get().choose_one(
        "trtllm::prims_ts_partitioned_moe", [runner], tuning_config, inputs
    )
    inputs[-2:] = [topk_weights, topk_ids]
    return runner(inputs, tactic=tactic, output=output)


@prims_ts_partitioned_moe.register_fake
def _prims_ts_partitioned_moe_fake(
    hidden_states,
    hidden_states_scale,
    gemm1_weights,
    gemm1_weights_scale,
    gemm2_weights,
    gemm2_weights_scale,
    gemm1_alpha,
    gemm1_beta,
    gemm1_clamp_limit,
    output1_scale_scalar,
    output1_scale_gate_scalar,
    output2_scale_scalar,
    topk_ids,
    topk_weights,
    num_experts,
    local_expert_offset,
    activation_type,
    do_finalize,
    output,
    tune_max_num_tokens=8192,
    use_dp=False,
    routing_method_type=0,
    n_group=None,
    topk_group=None,
    routed_scaling_factor=None,
    routing_bias=None,
):
    num_tokens, hidden_size = output.shape
    top_k = topk_ids.shape[1]
    rows = 0
    if num_tokens:
        _, rows = _workspace_rows(num_tokens, top_k, gemm1_weights[0].shape[0], 256, hidden_size)
    return (
        hidden_states.new_empty((rows, hidden_size), dtype=torch.bfloat16),
        topk_ids.new_empty((num_tokens, top_k), dtype=torch.int32),
        topk_weights.new_empty((0,)),
    )
