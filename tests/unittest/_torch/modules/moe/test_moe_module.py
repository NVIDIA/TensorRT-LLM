# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
MoE Module Unit Tests

This module provides a unified test framework for testing MoE modules through the
high-level create_moe() + forward() interface, rather than the backend-level interfaces.

Design Goals:
1. Test MoE module via: create_moe -> load_weights -> forward
2. Cover key quantization + backend combinations
3. Support EPLB (Expert Load Balancing) testing
4. Support autotune and tactic capture testing
"""

import copy
import logging
import os
import pickle
import sys
from contextlib import nullcontext
from itertools import product
from typing import List, Optional

import cloudpickle
import pytest
import torch
from _torch.modules.moe.moe_test_utils import (
    MoeBackendType,
    MoeModelConfig,
    create_test_param,
    get_quick_skip_reason,
    iter_base_test_configs,
    module_timer,  # noqa: F401 - imported for pytest fixture registration
    replay_tactics_and_check,
    should_skip_cutedsl,
    should_skip_deepgemm,
    should_skip_multi_gpu,
    should_skip_trtllm,
    supports_autotuner_capture,
)
from _torch.modules.moe.quantize_utils import get_test_quant_params
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from transformers.configuration_utils import PretrainedConfig

import tensorrt_llm.bindings.internal.runtime as _tbr
from tensorrt_llm._torch.autotuner import AutoTuner, autotune
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import (
    DeepSeekV3MoeRoutingMethod,
    DefaultMoeRoutingMethod,
    Llama4RenormalizeMoeRoutingMethod,
    MiniMaxM2MoeRoutingMethod,
    RenormalizeMoeRoutingMethod,
    RenormalizeNaiveMoeRoutingMethod,
    create_moe,
)
from tensorrt_llm._torch.modules.fused_moe.moe_load_balancer import (
    MoeLoadBalancer,
    MoeLoadBalancerIterContext,
)
from tensorrt_llm._torch.modules.fused_moe.quantization import (
    DeepSeekFP8BlockScalesFusedMoEMethod,
    FP8QDQFusedMoEMethod,
    INT8WoqPerChannelFusedMoEMethod,
    NVFP4CutlassFusedMoEMethod,
    NVFP4TRTLLMGenFusedMoEMethod,
    UnquantizedFusedMoEMethod,
    W4A8MXFP4FP8CutlassFusedMoEMethod,
    W4A8MXFP4FP8TRTLLMGenFusedMoEMethod,
    W4A8MXFP4MXFP8CutlassFusedMoEMethod,
    W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod,
    W4A8NVFP4FP8TRTLLMGenFusedMoEMethod,
    W4A16MXFP4TRTLLMGenFusedMoEMethod,
    WFP4A16FusedMoEMethod,
    WInt4AFP8FusedMoEMethod,
)
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.llmapi.llm_args import MoeLoadBalancerConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo

logger = logging.getLogger(__name__)

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


def _create_mapping_for_parallel_mode(world_size, parallel_mode):
    """Create Mapping for different parallelism strategies.

    Args:
        world_size: Total number of GPUs
        parallel_mode: One of "DEP", "TEP", "DTP", "TTP"
            - DEP: Attention uses DP, MoE uses EP
            - TEP: Attention uses TP, MoE uses EP
            - DTP: Attention uses DP, MoE uses TP
            - TTP: Attention uses TP, MoE uses TP

    Returns:
        Mapping object configured for the specified parallel mode
    """
    configs = {
        "DEP": {  # Attention DP, MoE EP
            "moe_ep_size": world_size,
            "moe_tp_size": 1,
            "enable_attention_dp": True,
        },
        "TEP": {  # Attention TP, MoE EP
            "moe_ep_size": world_size,
            "moe_tp_size": 1,
            "enable_attention_dp": False,
        },
        "DTP": {  # Attention DP, MoE TP
            "moe_ep_size": 1,
            "moe_tp_size": world_size,
            "enable_attention_dp": True,
        },
        "TTP": {  # Attention TP, MoE TP
            "moe_ep_size": 1,
            "moe_tp_size": world_size,
            "enable_attention_dp": False,
        },
    }
    if parallel_mode not in configs:
        raise ValueError(
            f"Unknown parallel_mode: {parallel_mode}. Must be one of {list(configs.keys())}"
        )

    cfg = configs[parallel_mode]
    return Mapping(
        world_size=world_size,
        tp_size=world_size,
        moe_ep_size=cfg["moe_ep_size"],
        moe_tp_size=cfg["moe_tp_size"],
        enable_attention_dp=cfg["enable_attention_dp"],
    )


def _create_moe_load_balancer(model_cfg, enable_eplb):
    """Create MoeLoadBalancer if EPLB is enabled, otherwise return nullcontext."""
    if not enable_eplb:
        return nullcontext()

    ep_rank = model_cfg.mapping.moe_ep_rank
    ep_size = model_cfg.mapping.moe_ep_size
    model_cfg.moe_load_balancer.setup(ep_rank=ep_rank, ep_size=ep_size)
    return MoeLoadBalancer(
        ep_rank=ep_rank,
        ep_size=ep_size,
        layer_updates_per_iter=model_cfg.moe_load_balancer.layer_updates_per_iter,
    )


def _setup_autotuner_for_test(mapping):
    """Configure AutoTuner for faster unit test profiling."""
    AutoTuner.get().setup_distributed_state(mapping)
    AutoTuner.get().clear_cache()
    autotuner = AutoTuner.get()
    autotuner.warmup = 0  # default: 2
    autotuner.repeat = 1  # default: 10
    autotuner.stream_delay_micro_secs = 10  # default: 1000


def _create_model_config(
    num_experts,
    hidden_size,
    intermediate_size,
    dtype,
    mapping,
    quant_config,
    moe_backend,
    enable_eplb=False,
    num_slots=-1,
    layer_updates_per_iter=-1,
):
    """Create PretrainedConfig and ModelConfig for MoE testing."""
    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = num_experts
    pretrained_config.hidden_size = hidden_size
    pretrained_config.intermediate_size = intermediate_size
    pretrained_config.torch_dtype = dtype

    moe_load_balancer_config = (
        MoeLoadBalancerConfig(
            num_slots=num_slots,
            layer_updates_per_iter=layer_updates_per_iter,
        )
        if enable_eplb
        else None
    )

    return ModelConfig(
        pretrained_config=pretrained_config,
        mapping=mapping,
        quant_config=quant_config,
        moe_backend=moe_backend,
        moe_disable_finalize_fusion=False,
        moe_load_balancer=moe_load_balancer_config,
    )


def _run_autotune_test(
    run_forward_fn, ref_fused_moe, ref_output, backend_type, quant_algo, run_all_tactics=False
):
    """Run autotune phase and tactic replay test.

    Args:
        run_forward_fn: Forward function to run
        ref_fused_moe: Reference MoE module for accuracy check
        ref_output: Reference output for comparison
        backend_type: MoE backend type
        quant_algo: Quantization algorithm
        run_all_tactics: If False, skip full tactic replay and only run simple accuracy check
    """
    # Autotune phase
    with torch.inference_mode(), autotune(cache_path="/tmp/moe_module_autotuner_cache.json"):
        _ = run_forward_fn()

    # Check if we should run full tactic replay
    if not run_all_tactics or not supports_autotuner_capture(backend_type, quant_algo):
        # Simple accuracy check for unsupported backends or when run_all_tactics is False
        with torch.inference_mode():
            output = run_forward_fn()
            ref_fused_moe.check_accuracy(output, ref_output)
        return

    # Capture phase: record which tactics are used
    with AutoTuner.get().capture() as all_tactics, torch.inference_mode():
        _ = run_forward_fn()

    # Replay phase: test each tactic for correctness
    replay_tactics_and_check(
        all_tactics=all_tactics,
        run_moe_fn=run_forward_fn,
        check_accuracy_fn=ref_fused_moe.check_accuracy,
        ref_output=ref_output,
        backend_type=backend_type,
        quant_algo=quant_algo,
        fail_fast=False,
    )


def _run_eplb_test(
    run_forward_fn, ref_fused_moe, ref_output, moe_load_balancer, initial_expert_ids
):
    """Run EPLB multi-iteration test.

    Args:
        run_forward_fn: Forward function to run
        ref_fused_moe: Reference MoE module for accuracy check
        ref_output: Reference output for comparison
        moe_load_balancer: MoeLoadBalancer instance
        initial_expert_ids: Expert IDs recorded immediately after MoE initialization (before any forward)
    """
    assert isinstance(moe_load_balancer, MoeLoadBalancer), (
        "Moe load balancer should be created when eplb is enabled"
    )
    assert initial_expert_ids is not None, (
        "initial_expert_ids should be recorded before any forward pass"
    )

    extra_steps = 1
    for _ in range(extra_steps):
        output = run_forward_fn()
        ref_fused_moe.check_accuracy(output, ref_output)

    current_expert_ids = copy.deepcopy(
        moe_load_balancer.single_layer_load_balancers[0].get_old_rank_expert_ids()
    )

    # EPLB should have updated expert_ids from initial state
    assert initial_expert_ids != current_expert_ids, (
        f"Expert ids after eplb update should be different from the initial loaded ones. "
        f"Initial: {initial_expert_ids}, Current: {current_expert_ids}"
    )


def _create_routing_method(routing_method_cls, top_k, num_experts, dtype):
    """
    Create a routing method instance with appropriate parameters for each routing method type.

    Args:
        routing_method_cls: The routing method class to instantiate
        top_k: Number of experts to select per token
        num_experts: Total number of experts
        dtype: Data type for tensors

    Returns:
        An instance of the routing method
    """
    # Routing methods with force_enable_pytorch_op support
    if routing_method_cls in (RenormalizeMoeRoutingMethod, DefaultMoeRoutingMethod):
        return routing_method_cls(top_k=top_k, force_enable_pytorch_op=True)

    # Simple routing methods (only top_k)
    if routing_method_cls in (RenormalizeNaiveMoeRoutingMethod, Llama4RenormalizeMoeRoutingMethod):
        return routing_method_cls(top_k=top_k)

    # DeepSeekV3 routing method requires special parameters
    if routing_method_cls == DeepSeekV3MoeRoutingMethod:
        # DeepSeek-V3 routing: groups experts, selects top groups, then selects top_k from those
        # The routing logic does topk(k=2) within each group, so each group must have >= 2 experts
        # Calculate n_group such that each group has at least 2 experts
        experts_per_group = 2
        n_group = max(1, num_experts // experts_per_group)
        # topk_group should be <= n_group and reasonable for the selection
        topk_group = min(n_group, max(1, n_group // 2))
        routed_scaling_factor = 1.0
        # Create e_score_correction_bias as a zero tensor (no bias correction in test)
        e_score_correction_bias = torch.zeros(num_experts, dtype=dtype, device="cuda")
        return routing_method_cls(
            top_k=top_k,
            n_group=n_group,
            topk_group=topk_group,
            routed_scaling_factor=routed_scaling_factor,
            callable_e_score_correction_bias=lambda: e_score_correction_bias,
            is_fused=False,  # Use PyTorch implementation for testing
        )

    # MiniMaxM2 routing method requires special parameters
    if routing_method_cls == MiniMaxM2MoeRoutingMethod:
        # Create e_score_correction_bias as a zero tensor (no bias correction in test)
        e_score_correction_bias = torch.zeros(num_experts, dtype=dtype, device="cuda")
        return routing_method_cls(
            top_k=top_k,
            num_experts=num_experts,
            callable_e_score_correction_bias=lambda: e_score_correction_bias,
        )

    # Fallback: try with just top_k
    return routing_method_cls(top_k=top_k)


def _test_moe_worker(
    moe_backend,
    dtype,
    quant_algo,
    mapping=None,
    enable_eplb=False,
    layer_updates_per_iter=-1,
    num_slots=-1,
    model_config: Optional[MoeModelConfig] = None,
    seq_len: int = 4,
    enable_autotune: bool = False,
    routing_method_cls=RenormalizeMoeRoutingMethod,
    dtype_routing_logits=None,
    swiglu_alpha: float = 1,
    swiglu_beta: float = 0,
    swiglu_limit: float = float("inf"),
):
    """
    Test MoE module worker function.

    This test verifies:
    1. MoE module forward pass produces correct results
    2. EPLB (Expert Load Balancing) works correctly when enabled
    3. Autotune works correctly with the module when enabled
    4. All tactics are captured and replayed properly when autotune is enabled

    Args:
        routing_method_cls: Routing method class to use (default: RenormalizeMoeRoutingMethod)
        dtype_routing_logits: Data type for routing logits (default: same as dtype).
                              DeepSeekV3 routing requires torch.float32.
        swiglu_alpha: SwiGLU alpha parameter (default=1, non-gptoss)
        swiglu_beta: SwiGLU beta parameter (default=0, non-gptoss)
        swiglu_limit: SwiGLU limit parameter (default=inf, non-gptoss)
    """
    import traceback

    try:
        _test_moe_worker_impl(
            moe_backend=moe_backend,
            dtype=dtype,
            quant_algo=quant_algo,
            mapping=mapping,
            enable_eplb=enable_eplb,
            layer_updates_per_iter=layer_updates_per_iter,
            num_slots=num_slots,
            model_config=model_config,
            seq_len=seq_len,
            enable_autotune=enable_autotune,
            routing_method_cls=routing_method_cls,
            dtype_routing_logits=dtype_routing_logits,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
        )
    except Exception:
        traceback.print_exc()
        raise


def _test_moe_worker_impl(
    moe_backend,
    dtype,
    quant_algo,
    mapping=None,
    enable_eplb=False,
    layer_updates_per_iter=-1,
    num_slots=-1,
    model_config: Optional[MoeModelConfig] = None,
    seq_len: int = 4,
    enable_autotune: bool = False,
    routing_method_cls=RenormalizeMoeRoutingMethod,
    dtype_routing_logits=None,
    swiglu_alpha: float = 1,
    swiglu_beta: float = 0,
    swiglu_limit: float = float("inf"),
):
    """Actual implementation of _test_moe_worker."""
    # Default routing logits dtype to model dtype if not specified
    if dtype_routing_logits is None:
        dtype_routing_logits = dtype
    # Parse model config
    if model_config is not None:
        num_experts = model_config.num_experts
        top_k = model_config.top_k
        hidden_size = model_config.hidden_size
        intermediate_size = model_config.intermediate_size
    else:
        num_experts, top_k, hidden_size, intermediate_size = 8, 2, 512, 512

    # Setup mapping
    mapping = mapping or Mapping()
    mapping.rank = mpi_rank()
    all_rank_num_tokens = [seq_len] * mapping.world_size
    torch.cuda.set_device(mapping.rank)

    with torch.device(f"cuda:{mapping.rank}"):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)

        # Create routing method and input tensors
        routing_method = _create_routing_method(
            routing_method_cls, top_k=top_k, num_experts=num_experts, dtype=dtype
        )
        x = torch.randn((seq_len, hidden_size), dtype=dtype, device="cuda")
        if enable_eplb:
            # Same router_logits for all tokens to force the eplb update weights
            router_logits = torch.randn(
                (1, num_experts), dtype=dtype_routing_logits, device="cuda"
            ).repeat(seq_len, 1)
        else:
            router_logits = torch.randn(
                (seq_len, num_experts), dtype=dtype_routing_logits, device="cuda"
            )

        # Determine swiglu_gptoss_style
        swiglu_gptoss_style = swiglu_alpha != 1 or swiglu_beta != 0 or swiglu_limit != float("inf")

        # In EP mode, swiglu tensors must be sized per local experts
        # (C++ kernels check: swiglu_alpha.size(0) == num_experts_on_rank)
        num_local_experts = num_experts // mapping.moe_ep_size

        # Setup quantization
        backend_type = MoeBackendType(moe_backend)
        quantize_util_cls, quant_config, quant_kwargs = get_test_quant_params(
            quant_algo, x, backend_type
        )
        quantize_util = quantize_util_cls(
            num_experts=num_experts,
            dtype=dtype,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            quant_config=quant_config,
            bias=swiglu_gptoss_style,
            swiglu_gptoss_style=swiglu_gptoss_style,
            swiglu_alpha=swiglu_alpha if swiglu_gptoss_style else None,
            swiglu_beta=swiglu_beta if swiglu_gptoss_style else None,
            swiglu_limit=swiglu_limit if swiglu_gptoss_style else None,
            num_local_experts=num_local_experts,
        )
        weights = quantize_util.create_weights(**quant_kwargs)

        # For EPLB, keep weights on CPU
        if enable_eplb:
            for key in weights:
                if isinstance(weights[key], torch.Tensor):
                    weights[key] = weights[key].to("cpu")
        ref_weights = copy.deepcopy(weights) if enable_eplb else weights

        # Create configs
        model_cfg = _create_model_config(
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            mapping=mapping,
            quant_config=quant_config,
            moe_backend=moe_backend,
            enable_eplb=enable_eplb,
            num_slots=num_slots,
            layer_updates_per_iter=layer_updates_per_iter,
        )

        # Create MoE load balancer
        moe_load_balancer = _create_moe_load_balancer(model_cfg, enable_eplb)

        # Get swiglu tensors if swiglu_gptoss_style is enabled
        swiglu_tensors = quantize_util.get_swiglu_tensors()

        with moe_load_balancer:
            # Create and setup fused MoE module
            fused_moe = create_moe(
                routing_method=routing_method,
                reduce_results=True,
                model_config=model_cfg,
                bias=swiglu_gptoss_style,
                swiglu_alpha=swiglu_tensors["swiglu_alpha"] if swiglu_tensors else None,
                swiglu_beta=swiglu_tensors["swiglu_beta"] if swiglu_tensors else None,
                swiglu_limit=swiglu_tensors["swiglu_limit"] if swiglu_tensors else None,
            )
            fused_moe.load_weights([weights])
            fused_moe.post_load_weights()
            fused_moe.cuda(f"cuda:{mapping.rank}")

            # Record initial expert_ids before any forward pass (for EPLB test)
            initial_expert_ids = None
            if isinstance(moe_load_balancer, MoeLoadBalancer):
                moe_load_balancer.register_weight_slots_after_to_cuda()
                moe_load_balancer.finalize_model()
                moe_load_balancer.set_iter_info(enable_statistic=True, enable_update_weights=True)
                # Record initial expert_ids immediately after initialization
                # Use deepcopy to avoid reference issues if the list is modified in-place
                initial_expert_ids = copy.deepcopy(
                    moe_load_balancer.single_layer_load_balancers[0].get_old_rank_expert_ids()
                )
                logger.info(f"[EPLB Debug] Initial expert_ids (after init): {initial_expert_ids}")

            # Create reference module
            ref_fused_moe = quantize_util.create_ref_module(routing_method)
            ref_fused_moe.load_weights([ref_weights])
            ref_fused_moe.cuda(f"cuda:{mapping.rank}")

            # Define forward function
            def run_forward():
                with torch.inference_mode():
                    if isinstance(moe_load_balancer, MoeLoadBalancer):
                        with MoeLoadBalancerIterContext(moe_load_balancer):
                            output = fused_moe.forward(
                                x, router_logits, all_rank_num_tokens=all_rank_num_tokens
                            )
                    else:
                        output = fused_moe.forward(
                            x, router_logits, all_rank_num_tokens=all_rank_num_tokens
                        )
                torch.cuda.synchronize()
                return output

            # Get reference output
            with torch.inference_mode():
                ref_output = ref_fused_moe.forward(x, router_logits)

            # Run tests
            if enable_autotune:
                _setup_autotuner_for_test(mapping)
                _run_autotune_test(run_forward, ref_fused_moe, ref_output, backend_type, quant_algo)
            else:
                output = run_forward()
                ref_fused_moe.check_accuracy(output, ref_output)

            if enable_eplb:
                _run_eplb_test(
                    run_forward, ref_fused_moe, ref_output, moe_load_balancer, initial_expert_ids
                )


def _test_moe_multi_gpu(
    comm_method_type,
    moe_backend,
    quant_algo,
    dtype,
    world_size,
    parallel_mode="DEP",
    enable_eplb=False,
    layer_updates_per_iter=-1,
    num_slots=-1,
    model_config: Optional[MoeModelConfig] = None,
    seq_len: int = 4,
    enable_autotune: bool = False,
    routing_method_cls=RenormalizeMoeRoutingMethod,
    dtype_routing_logits=None,
    swiglu_alpha: float = 1,
    swiglu_beta: float = 0,
    swiglu_limit: float = float("inf"),
):
    """
    Test MoE module with multi-GPU support.

    Args:
        comm_method_type: Communication method type
        moe_backend: Backend type string
        quant_algo: Quantization algorithm
        dtype: Activation data type
        world_size: Total world size
        parallel_mode: Parallelism strategy ("DEP", "TEP", "DTP", "TTP")
        enable_eplb: Enable Expert Load Balancing
        layer_updates_per_iter: EPLB layer updates per iteration
        num_slots: EPLB number of slots
        model_config: MoE model configuration
        seq_len: Sequence length for test input
        enable_autotune: Enable autotune and tactic capture/replay testing
        routing_method_cls: Routing method class to use
        dtype_routing_logits: Data type for routing logits (default: same as dtype)
        swiglu_alpha: SwiGLU alpha parameter (default=1, non-gptoss)
        swiglu_beta: SwiGLU beta parameter (default=0, non-gptoss)
        swiglu_limit: SwiGLU limit parameter (default=inf, non-gptoss)
    """

    def init_worker(custom_paths, comm_method_type):
        # Update the sys.path to align with main process for submodule import
        for custom_path in custom_paths:
            if custom_path.endswith("tests/unittest") and custom_path not in sys.path:
                sys.path.append(custom_path)

        # Set comm method
        os.environ["TRTLLM_FORCE_COMM_METHOD"] = comm_method_type

    mapping = _create_mapping_for_parallel_mode(world_size, parallel_mode)

    with MPIPoolExecutor(
        initializer=init_worker, initargs=(sys.path, comm_method_type), max_workers=world_size
    ) as executor:
        results = executor.map(
            _test_moe_worker,
            *zip(
                *[
                    (
                        moe_backend,
                        dtype,
                        quant_algo,
                        mapping,
                        enable_eplb,
                        layer_updates_per_iter,
                        num_slots,
                        model_config,
                        seq_len,
                        enable_autotune,
                        routing_method_cls,
                        dtype_routing_logits,
                        swiglu_alpha,
                        swiglu_beta,
                        swiglu_limit,
                    )
                ]
                * world_size
            ),
        )
        for r in results:
            assert r is None


# ============================================================================
# Test Parameters Configuration
# ============================================================================

# Quantization algorithms to test
QUANT_ALGOS = [
    None,  # Unquantized
    QuantAlgo.FP8,
    QuantAlgo.NVFP4,
    QuantAlgo.FP8_BLOCK_SCALES,
    QuantAlgo.W4A8_NVFP4_FP8,
    QuantAlgo.W4A16_MXFP4,
    QuantAlgo.W4A8_MXFP4_MXFP8,
    QuantAlgo.W8A16,
    QuantAlgo.W4A8_AWQ,
]

# Backend types to test
BACKEND_TYPES = [
    MoeBackendType.CUTLASS,
    MoeBackendType.TRTLLM,
    MoeBackendType.CUTEDSL,
    MoeBackendType.DEEPGEMM,
]

# Data types to test
DTYPES = [
    torch.float16,
    torch.bfloat16,
]

# Model configurations for testing
# (num_experts, top_k, hidden_size, intermediate_size)
#
# Default runs the full local config matrix (TRTLLM_TEST_MOE_CI=0).
# Set TRTLLM_TEST_MOE_CI=1 in CI to run only the smaller subset for speed.
CI_MOE_MODEL_CONFIGS = [
    MoeModelConfig(60, 4, 2048, 1408),  # Qwen1.5-MoE-A2.7B
    MoeModelConfig(32, 8, 7168, 2048),  # DeepSeek-V3 (reduced from 256 experts to accelerate test)
    MoeModelConfig(128, 4, 2880, 2880),  # GPT-OSS-120B
    MoeModelConfig(8, 1, 512, 512),  # boundary: top_k=1, single expert activated
]

LOCAL_MOE_MODEL_CONFIGS = CI_MOE_MODEL_CONFIGS + [
    MoeModelConfig(64, 6, 2048, 1408),  # DeepSeek-MoE-16B / DeepSeek-V2-Lite
    MoeModelConfig(384, 8, 7168, 2048),  # Kimi-K2
    # === Boundary Tests: num_experts / top_k ===
    MoeModelConfig(4, 4, 512, 512),  # top_k=num_experts, all experts activated
    MoeModelConfig(7, 2, 256, 512),  # prime num_experts
    MoeModelConfig(13, 3, 256, 512),  # prime num_experts, odd top_k
    # === Boundary Tests: small sizes ===
    MoeModelConfig(4, 2, 64, 128),  # very small hidden_size
    MoeModelConfig(4, 2, 128, 64),  # intermediate < hidden
]

MOE_MODEL_CONFIGS = (
    CI_MOE_MODEL_CONFIGS
    if os.environ.get("TRTLLM_TEST_MOE_CI", "0") == "1"
    else LOCAL_MOE_MODEL_CONFIGS
)

# Sequence lengths to test
SEQ_LENS = [1, 8]

# Routing methods to test
ROUTING_METHODS = [
    RenormalizeMoeRoutingMethod,  # TopK -> Softmax (Mixtral, etc.)
    DefaultMoeRoutingMethod,  # Softmax -> TopK
    RenormalizeNaiveMoeRoutingMethod,  # Softmax -> TopK -> Renormalize (Qwen3)
    Llama4RenormalizeMoeRoutingMethod,  # Top1 -> Sigmoid (Llama4)
    DeepSeekV3MoeRoutingMethod,  # Sigmoid -> BiasAdd -> Group TopK (DeepSeek-V3)
    MiniMaxM2MoeRoutingMethod,  # Sigmoid -> BiasAdd -> TopK -> Renormalize (MiniMax-M2)
]


MULTI_GPU_ROUTING_METHODS = [
    RenormalizeMoeRoutingMethod,  # TopK -> Softmax (Mixtral, etc.)
    DeepSeekV3MoeRoutingMethod,  # Sigmoid -> BiasAdd -> Group TopK (DeepSeek-V3)
]


# ============================================================================
# Multi-GPU Test Configuration
# ============================================================================
# Parallel modes to test
PARALLEL_MODES = [
    "DEP",  # Attention DP, MoE EP
    "TEP",  # Attention TP, MoE EP
    "DTP",  # Attention DP, MoE TP
    "TTP",  # Attention TP, MoE TP
]

# Communication methods to test
COMM_METHODS = [
    "NVLINK_ONE_SIDED",
    "NVLINK_TWO_SIDED",
    "DEEPEP",
    "DEEPEPLOWLATENCY",
]

# SwiGLU parameters for swiglu_gptoss_style testing
SWIGLU_ALPHAS = [1, 1.702]  # default, GPT-OSS (modeling_gpt_oss.py)
SWIGLU_BETAS = [0, 1.0]  # default, GPT-OSS
SWIGLU_LIMITS = [float("inf"), 7.0]  # default, GPT-OSS

# Single-GPU: full product of all SwiGLU combos
SWIGLU_COMBOS = list(product(SWIGLU_ALPHAS, SWIGLU_BETAS, SWIGLU_LIMITS))

# Multi-GPU: only non-gptoss (default) and one gptoss combo
MULTI_GPU_SWIGLU_COMBOS = [
    (1, 0, float("inf")),  # non-gptoss (default SwiGLU)
    (1.702, 1.0, 7.0),  # gptoss style (GPT-OSS real values)
]


def _get_comm_method_skip_reason(
    comm_method: str,
    model_config: "MoeModelConfig",
) -> Optional[str]:
    """
    Check if a communication method is compatible with the given model config.

    Returns a skip reason string if incompatible, None otherwise.
    """
    from tensorrt_llm._torch.modules.fused_moe.communication.deep_ep_low_latency import (
        DeepEPLowLatency,
    )

    if comm_method == "DEEPEPLOWLATENCY":
        if model_config.hidden_size not in DeepEPLowLatency.SUPPORTED_HIDDEN_SIZES:
            return (
                f"DeepEPLowLatency does not support hidden_size={model_config.hidden_size}, "
                f"requires one of {sorted(DeepEPLowLatency.SUPPORTED_HIDDEN_SIZES)}"
            )
    return None


def generate_multi_gpu_test_params(
    parallel_modes,
    comm_methods,
    swiglu_combos,
    model_configs,
    seq_lens,
    dtypes,
    backend_types,
    quant_algos,
    routing_methods,
) -> List:
    """
    Generate test parameter combinations for multi-GPU tests.

    Args:
        parallel_modes: List of parallel modes
        comm_methods: List of communication methods
        swiglu_combos: List of (swiglu_alpha, swiglu_beta, swiglu_limit) tuples
        model_configs: List of MoeModelConfig
        seq_lens: List of sequence lengths
        dtypes: List of data types
        backend_types: List of backend types
        quant_algos: List of quantization algorithms
        routing_methods: List of routing method classes

    Returns:
        List of pytest.param objects with appropriate skip marks
    """
    params: List = []
    for parallel_mode, comm_method in product(parallel_modes, comm_methods):
        for (
            swiglu_alpha,
            swiglu_beta,
            swiglu_limit,
            model_config,
            seq_len,
            dtype,
            backend_type,
            quant_algo,
            routing_method_cls,
            skip_reason,
            base_test_id,
        ) in iter_base_test_configs(
            swiglu_combos,
            model_configs,
            seq_lens,
            dtypes,
            backend_types,
            quant_algos,
            routing_methods,
        ):
            # Check multi-GPU specific skip conditions
            if not skip_reason:
                skip_reason = _get_comm_method_skip_reason(comm_method, model_config)
            if not skip_reason:
                skip_reason = should_skip_trtllm(
                    backend_type, quant_algo, model_config, comm_method=comm_method
                )
            if not skip_reason:
                skip_reason = should_skip_cutedsl(
                    backend_type, quant_algo, model_config, comm_method
                )
            if not skip_reason:
                skip_reason = should_skip_deepgemm(
                    backend_type, comm_method, quant_algo=quant_algo, model_config=model_config
                )
            if not skip_reason:
                skip_reason = should_skip_multi_gpu(parallel_mode, model_config, world_size=4)

            test_id = f"parallel={parallel_mode}-comm={comm_method}-{base_test_id}"
            param_values = (
                parallel_mode,
                comm_method,
                dtype,
                backend_type.value,
                quant_algo,
                seq_len,
                model_config,
                routing_method_cls,
                swiglu_alpha,
                swiglu_beta,
                swiglu_limit,
            )
            params.append(create_test_param(param_values, test_id, skip_reason))

    return params


def generate_base_test_params(
    swiglu_combos, model_configs, seq_lens, dtypes, backend_types, quant_algos, routing_methods
) -> List:
    """
    Generate test parameter combinations for base tests.

    Args:
        swiglu_combos: List of (swiglu_alpha, swiglu_beta, swiglu_limit) tuples
        model_configs: List of MoeModelConfig
        seq_lens: List of sequence lengths
        dtypes: List of data types
        backend_types: List of backend types
        quant_algos: List of quantization algorithms
        routing_methods: List of routing method classes

    Returns:
        List of pytest.param objects with appropriate skip marks
    """
    params: List = []
    for (
        swiglu_alpha,
        swiglu_beta,
        swiglu_limit,
        model_config,
        seq_len,
        dtype,
        backend_type,
        quant_algo,
        routing_method_cls,
        skip_reason,
        base_test_id,
    ) in iter_base_test_configs(
        swiglu_combos, model_configs, seq_lens, dtypes, backend_types, quant_algos, routing_methods
    ):
        param_values = (
            dtype,
            backend_type.value,
            quant_algo,
            seq_len,
            model_config,
            routing_method_cls,
            swiglu_alpha,
            swiglu_beta,
            swiglu_limit,
        )
        params.append(create_test_param(param_values, base_test_id, skip_reason))

    return params


# ============================================================================
# MoE Single GPU Tests
# ============================================================================
# Pre-generate test parameters at module load time
BASE_TEST_PARAMS = generate_base_test_params(
    swiglu_combos=SWIGLU_COMBOS,
    model_configs=MOE_MODEL_CONFIGS,
    seq_lens=SEQ_LENS,
    dtypes=DTYPES,
    backend_types=BACKEND_TYPES,
    quant_algos=QUANT_ALGOS,
    routing_methods=ROUTING_METHODS,
)


@pytest.mark.skip(reason="Temporarily skipped due to the long time to run the test")
@pytest.mark.parametrize(
    "dtype,moe_backend,quant_algo,seq_len,model_config,routing_method_cls,"
    "swiglu_alpha,swiglu_beta,swiglu_limit",
    BASE_TEST_PARAMS,
)
def test_ConfigurableMoE_single_gpu(
    dtype: torch.dtype,
    moe_backend: str,
    quant_algo: Optional[QuantAlgo],
    seq_len: int,
    model_config: MoeModelConfig,
    routing_method_cls,
    swiglu_alpha: float,
    swiglu_beta: float,
    swiglu_limit: float,
):
    """
    Single-GPU test for ConfigurableMoE module.

    This test verifies:
    1. MoE create_moe -> load_weights -> forward produces correct results
    2. Various backend + quantization combinations work correctly
    3. Autotune captures and replays all tactics properly
    4. swiglu_gptoss_style (SwiGLU with custom parameters) works correctly
    """
    # DeepSeekV3 routing requires float32 routing_logits for TRTLLM backend
    # See: cpp/tensorrt_llm/thop/fp4BlockScaleMoe.cpp:70-72
    dtype_routing_logits = None
    if (
        moe_backend == MoeBackendType.TRTLLM.value
        and routing_method_cls == DeepSeekV3MoeRoutingMethod
    ):
        dtype_routing_logits = torch.float32

    _test_moe_worker(
        moe_backend=moe_backend,
        dtype=dtype,
        quant_algo=quant_algo,
        model_config=model_config,
        seq_len=seq_len,
        enable_autotune=True,
        routing_method_cls=routing_method_cls,
        dtype_routing_logits=dtype_routing_logits,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
    )


# ============================================================================
# MoE Multi-GPU Tests
# ============================================================================
# Pre-generate multi-GPU test parameters at module load time
MULTI_GPU_TEST_PARAMS = generate_multi_gpu_test_params(
    parallel_modes=PARALLEL_MODES,
    comm_methods=COMM_METHODS,
    swiglu_combos=MULTI_GPU_SWIGLU_COMBOS,
    model_configs=MOE_MODEL_CONFIGS,
    seq_lens=SEQ_LENS,
    dtypes=DTYPES,
    backend_types=BACKEND_TYPES,
    quant_algos=QUANT_ALGOS,
    routing_methods=MULTI_GPU_ROUTING_METHODS,
)


@pytest.mark.skip(reason="Temporarily skipped due to the long time to run the test")
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs to run this test")
@pytest.mark.parametrize(
    "parallel_mode,comm_method_type,dtype,moe_backend,quant_algo,seq_len,model_config,"
    "routing_method_cls,swiglu_alpha,swiglu_beta,swiglu_limit",
    MULTI_GPU_TEST_PARAMS,
)
def test_ConfigurableMoE_multi_gpu(
    parallel_mode,
    comm_method_type,
    dtype,
    moe_backend,
    quant_algo,
    seq_len,
    model_config,
    routing_method_cls,
    swiglu_alpha,
    swiglu_beta,
    swiglu_limit,
):
    # DeepSeekV3 routing requires float32 routing_logits for TRTLLM backend
    # See: cpp/tensorrt_llm/thop/fp4BlockScaleMoe.cpp:70-72
    dtype_routing_logits = None
    if (
        moe_backend == MoeBackendType.TRTLLM.value
        and routing_method_cls == DeepSeekV3MoeRoutingMethod
    ):
        dtype_routing_logits = torch.float32

    world_size = 4
    _test_moe_multi_gpu(
        comm_method_type,
        moe_backend,
        quant_algo,
        dtype=dtype,
        world_size=world_size,
        parallel_mode=parallel_mode,
        model_config=model_config,
        seq_len=seq_len,
        routing_method_cls=routing_method_cls,
        dtype_routing_logits=dtype_routing_logits,
        swiglu_alpha=swiglu_alpha,
        swiglu_beta=swiglu_beta,
        swiglu_limit=swiglu_limit,
    )


# ============================================================================
# MoE Multi-GPU EPLB Tests
# ============================================================================
# EPLB-specific configuration
EPLB_PARALLEL_MODES = ["DEP"]  # EPLB only works with DEP mode (use_dp=True)
EPLB_COMM_METHODS = [
    "NVLINK_ONE_SIDED",
    "NVLINK_TWO_SIDED",
]  # Communication methods for EPLB
EPLB_ROUTING_METHODS = [RenormalizeMoeRoutingMethod]  # Common routing methods
EPLB_MODEL_CONFIGS = [MoeModelConfig(8, 2, 512, 512)]  # Model configs for EPLB
EPLB_NUM_SLOTS_LIST = [16]  # Must be > num_experts (8) to be effective


def _get_fused_moe_method_class(quant_algo, backend_type):
    """
    Get the FusedMoEMethod class based on quant_algo and backend_type.

    This mirrors the logic in each backend's _get_quant_method() method.

    Returns:
        FusedMoEMethod class or None if not found
    """
    backend_str = backend_type.value if hasattr(backend_type, "value") else str(backend_type)

    if quant_algo is None:
        # Unquantized - only CUTLASS supports it
        if backend_str == "CUTLASS":
            return UnquantizedFusedMoEMethod
        return None

    # CUTLASS backend
    # Mapping based on CutlassFusedMoE._get_quant_method() logic
    if backend_str == "CUTLASS":
        method_map = {
            QuantAlgo.FP8: FP8QDQFusedMoEMethod,
            QuantAlgo.FP8_BLOCK_SCALES: DeepSeekFP8BlockScalesFusedMoEMethod,
            QuantAlgo.NVFP4: NVFP4CutlassFusedMoEMethod,
            # W4A8_AWQ uses is_int4_weight_only_per_group() -> WInt4AFP8FusedMoEMethod
            QuantAlgo.W4A8_AWQ: WInt4AFP8FusedMoEMethod,
            QuantAlgo.W8A16: INT8WoqPerChannelFusedMoEMethod,
            QuantAlgo.W4A16_MXFP4: WFP4A16FusedMoEMethod,
            QuantAlgo.W4A8_MXFP4_FP8: W4A8MXFP4FP8CutlassFusedMoEMethod,
            QuantAlgo.W4A8_MXFP4_MXFP8: W4A8MXFP4MXFP8CutlassFusedMoEMethod,
            # Note: W4A8_NVFP4_FP8 is NOT supported by CUTLASS backend
        }
        return method_map.get(quant_algo)

    # TRTLLM backend
    if backend_str == "TRTLLM":
        method_map = {
            QuantAlgo.FP8_BLOCK_SCALES: DeepSeekFP8BlockScalesFusedMoEMethod,
            QuantAlgo.NVFP4: NVFP4TRTLLMGenFusedMoEMethod,
            QuantAlgo.W4A16_MXFP4: W4A16MXFP4TRTLLMGenFusedMoEMethod,
            QuantAlgo.W4A8_NVFP4_FP8: W4A8NVFP4FP8TRTLLMGenFusedMoEMethod,
            QuantAlgo.W4A8_MXFP4_FP8: W4A8MXFP4FP8TRTLLMGenFusedMoEMethod,
            QuantAlgo.W4A8_MXFP4_MXFP8: W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod,
        }
        return method_map.get(quant_algo)

    # CUTEDSL backend uses same methods as CUTLASS for quantization
    if backend_str == "CUTEDSL":
        method_map = {
            QuantAlgo.NVFP4: NVFP4CutlassFusedMoEMethod,
        }
        return method_map.get(quant_algo)

    # DEEPGEMM backend
    if backend_str == "DEEPGEMM":
        method_map = {
            QuantAlgo.FP8_BLOCK_SCALES: DeepSeekFP8BlockScalesFusedMoEMethod,
        }
        return method_map.get(quant_algo)

    return None


def _should_skip_EPLB(quant_algo, backend_type, num_slots, num_experts):
    """
    Check if EPLB test should be skipped based on quant_algo, backend_type, and slot configuration.

    Returns:
        str or None: Skip reason if should skip, None otherwise
    """
    # Check num_slots > num_experts requirement
    if num_slots <= num_experts:
        return f"EPLB requires num_slots ({num_slots}) > num_experts ({num_experts})"

    # Get the FusedMoEMethod class for this quant_algo + backend combination
    method_class = _get_fused_moe_method_class(quant_algo, backend_type)

    if method_class is None:
        # Cannot determine the method class, skip the test
        return (
            f"Cannot determine FusedMoEMethod for quant_algo={quant_algo}, backend={backend_type}"
        )

    # Query the method class directly for EPLB support
    if not method_class.supports_online_eplb():
        return f"EPLB not supported for {method_class.__name__} (supports_online_eplb=False)"

    return None


def generate_eplb_test_params(
    parallel_modes,
    comm_methods,
    model_configs,
    num_slots_list,
    dtypes,
    backend_types,
    quant_algos,
    routing_methods,
) -> List:
    """
    Generate test parameter combinations for EPLB tests.

    EPLB requires num_slots > num_experts to be effective.

    Args:
        parallel_modes: List of parallel modes (only EP modes: DEP, TEP)
        comm_methods: List of communication methods
        model_configs: List of MoeModelConfig
        num_slots_list: List of EPLB slots (must be > num_experts)
        dtypes: List of data types
        backend_types: List of backend types
        quant_algos: List of quantization algorithms
        routing_methods: List of routing method classes

    Returns:
        List of pytest.param objects with appropriate skip marks
    """
    params: List = []

    for (
        parallel_mode,
        comm_method,
        model_config,
        num_slots,
        dtype,
        backend_type,
        quant_algo,
        routing_method_cls,
    ) in product(
        parallel_modes,
        comm_methods,
        model_configs,
        num_slots_list,
        dtypes,
        backend_types,
        quant_algos,
        routing_methods,
    ):
        # Get skip reason using existing logic
        skip_reason = get_quick_skip_reason(
            backend_type, quant_algo, dtype, model_config, routing_method_cls
        )

        # Check EPLB-specific skip conditions
        if not skip_reason:
            skip_reason = _should_skip_EPLB(
                quant_algo, backend_type, num_slots, model_config.num_experts
            )

        routing_name = routing_method_cls.__name__.replace("MoeRoutingMethod", "")
        test_id = (
            f"parallel={parallel_mode}-comm={comm_method}-{model_config}-slots={num_slots}-"
            f"dtype={dtype}-backend={backend_type.value}-quant={quant_algo}-routing={routing_name}"
        )

        param_values = (
            parallel_mode,
            comm_method,
            dtype,
            backend_type.value,
            quant_algo,
            model_config,
            num_slots,
            routing_method_cls,
        )
        params.append(create_test_param(param_values, test_id, skip_reason))

    return params


# Pre-generate EPLB test parameters at module load time
EPLB_TEST_PARAMS = generate_eplb_test_params(
    parallel_modes=EPLB_PARALLEL_MODES,
    comm_methods=EPLB_COMM_METHODS,
    model_configs=EPLB_MODEL_CONFIGS,
    num_slots_list=EPLB_NUM_SLOTS_LIST,
    dtypes=DTYPES,
    backend_types=BACKEND_TYPES,
    quant_algos=QUANT_ALGOS,
    routing_methods=EPLB_ROUTING_METHODS,
)


@pytest.mark.skip(reason="Temporarily skipped due to the long time to run the test")
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs to run this test")
@pytest.mark.skipif(
    not _tbr.is_host_accessible_device_memory_supported(),
    reason="needs support of host accessible device memory",
)
@pytest.mark.parametrize(
    "parallel_mode,comm_method_type,dtype,moe_backend,quant_algo,model_config,num_slots,routing_method_cls",
    EPLB_TEST_PARAMS,
)
def test_ConfigurableMoE_multi_gpu_eplb(
    parallel_mode,
    comm_method_type,
    dtype,
    moe_backend,
    quant_algo,
    model_config,
    num_slots,
    routing_method_cls,
):
    world_size = 4
    _test_moe_multi_gpu(
        comm_method_type,
        moe_backend,
        quant_algo,
        dtype=dtype,
        world_size=world_size,
        parallel_mode=parallel_mode,
        enable_eplb=True,
        layer_updates_per_iter=1,
        num_slots=num_slots,
        model_config=model_config,
        routing_method_cls=routing_method_cls,
    )
