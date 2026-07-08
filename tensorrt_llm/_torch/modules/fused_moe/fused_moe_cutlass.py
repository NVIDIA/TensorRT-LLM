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

import inspect
import os
from functools import cached_property
from typing import Dict, List, Optional, Tuple, Union

import torch

from tensorrt_llm._mnnvl_utils import MnnvlMemory, MnnvlMoe
from tensorrt_llm._torch.distributed.moe_alltoall import MoeAlltoAll
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantAlgo
from tensorrt_llm.tools.layer_wise_benchmarks import get_calibrator

from ...distributed import allgather
from ...expert_statistic import ExpertStatistic
from ...model_config import ModelConfig
from ...peft.lora.layer import (MOE_LORA_MODULE_NAMES,
                                MOE_LORA_MODULE_TO_KERNEL_SLOT, LoraModuleType,
                                MoeLoraLayer)
from ...peft.lora.validation import has_moe_lora_targets
from ...utils import (ActivationType, AuxStreamType, EventType,
                      Fp4QuantizedTensor)
from .interface import AlltoallMethodType, MoE
from .quantization import UnquantizedFusedMoEMethod
from .wide_ep_ft import get_wide_ep_ft_options

# isort: off
from .quantization import (
    DeepSeekFP8BlockScalesFusedMoEMethod, FP8QDQFusedMoEMethod,
    MoEWeightLoadingMode, MXFP8CutlassFusedMoEMethod,
    NVFP4CutlassFusedMoEMethod, INT8WoqPerChannelFusedMoEMethod,
    W4A16NVFP4CutlassFusedMoEMethod, W4A8MXFP4FP8CutlassFusedMoEMethod,
    W4A8MXFP4MXFP8CutlassFusedMoEMethod, WFP4A16FusedMoEMethod,
    WInt4AFP8FusedMoEMethod)
# isort: on
from .routing import BaseMoeRoutingMethod


def raise_moe_lora_multichunk_unsupported(num_chunks: int) -> None:
    """Reject multi-chunk execution for routed-expert MoE LoRA.

    Routed-expert MoE LoRA passes per-request/slot adapter metadata that is not
    re-sliced per token-chunk, so multi-chunk execution would mismatch the
    kernel's per-token expansion. Shared by CutlassFusedMoE.forward_impl and the
    MoEScheduler so the message stays in one place.
    """
    raise NotImplementedError(
        f"Routed-expert MoE LoRA does not support multi-chunk execution "
        f"(num_chunks={num_chunks}). Reduce the per-forward token count or "
        f"increase `moe_max_num_tokens` so the MoE runs in a single chunk.")


class CutlassFusedMoE(MoE):
    """
    Fused Mixture of Experts (MoE) Layer with performance tuning.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of top experts to select for each input token.
        hidden_size (int): Size of the hidden state.
        intermediate_size (int): Size of the intermediate state.
        aux_stream_dict (Optional[Dict[AuxStreamType, torch.cuda.Stream]]): Auxiliary CUDA streams for overlapping.
        dtype (Optional[torch.dtype]): Data type for the weights.
        reduce_results (bool): Whether to reduce the results across devices.
        model_config (ModelConfig): Configuration object for the model.

    MoE torch custom op:
        In max-throughput mode:
        Quant:
            fp8 block scales (SM90 Hopper only):
                FusedMoE Op: dynamic quant + scatter + gemm1 + swiglu + gemm2 + finalizeMoeRoute (return one tensor)
            p8 qdq, nvfp4:
                FusedMoE Op: scatter + gemm1 + swiglu + gemm2 + finalizeMoeRoute (return one tensor)

    FusedMoE module:
        max-throughput mode:
            routing(topK, etc.) [+ dynamic quant for fp8 qdq and nvfp4 ] [+ fp4_allgather] + FusedMoe Op[no allreduce] + reducescatter, with AttentionDP on
            equals to: dynamic quant + routing(topK, etc.) [+ fp4_allgather] + scatter + gemm1 + swiglu + gemm2 + finalizeMoeRoute [no allreduce] + reducescatter
    """

    # Quantization algorithm support table for can_implement()
    # Format: quant_algo -> {sm_constraint, dtypes}
    # sm_constraint types:
    #   - ("min", N): SM >= N
    #   - ("exact", N): SM == N
    #   - ("in", {N1, N2, ...}): SM in set
    _QUANT_SUPPORT_TABLE = {
        # Unquantized (FP16/BF16): SM >= 80
        None: {
            "sm_constraint": ("min", 80),
            "dtypes": {torch.float16, torch.bfloat16},
        },
        # FP8 per-tensor (QDQ): SM >= 89
        QuantAlgo.FP8: {
            "sm_constraint": ("min", 89),
            "dtypes": {torch.float16, torch.bfloat16, torch.float32},
        },
        # FP8_BLOCK_SCALES: SM in {90, 120}
        QuantAlgo.FP8_BLOCK_SCALES: {
            "sm_constraint": ("in", {90, 120}),
            "dtypes": {torch.bfloat16},
        },
        # NVFP4: SM in {100, 103, 120, 121}
        QuantAlgo.NVFP4: {
            "sm_constraint": ("in", {100, 103, 120, 121}),
            "dtypes": {torch.float16, torch.bfloat16, torch.float8_e4m3fn},
        },
        # W4A8_AWQ: SM in {89, 90} only
        QuantAlgo.W4A8_AWQ: {
            "sm_constraint": ("in", {89, 90}),
            "dtypes": {torch.float16, torch.bfloat16},
        },
        # W8A16: SM >= 80
        QuantAlgo.W8A16: {
            "sm_constraint": ("min", 80),
            "dtypes": {torch.float16, torch.bfloat16},
        },
        # W4A16_MXFP4: SM == 90 only
        QuantAlgo.W4A16_MXFP4: {
            "sm_constraint": ("exact", 90),
            "dtypes": {torch.float16, torch.bfloat16},
        },
        # W4A8_MXFP4_FP8: SM in {100, 103}
        QuantAlgo.W4A8_MXFP4_FP8: {
            "sm_constraint": ("in", {100, 103}),
            "dtypes": {torch.float16, torch.bfloat16, torch.float32},
        },
        # W4A8_MXFP4_MXFP8: SM in {100, 103, 120, 121}
        QuantAlgo.W4A8_MXFP4_MXFP8: {
            "sm_constraint": ("in", {100, 103, 120, 121}),
            "dtypes": {torch.float16, torch.bfloat16},
        },
        # MXFP8 (W8A8 e4m3xe4m3 with UE8M0 1x32 block scales): SM in {100, 103}.
        # M3.1 enables construction/load; the fused kernel is M3.2.
        QuantAlgo.MXFP8: {
            "sm_constraint": ("in", {100, 103}),
            "dtypes": {torch.float16, torch.bfloat16},
        },
    }

    _GPTOSS_SUPPORTED_ALGOS = {QuantAlgo.W4A8_MXFP4_MXFP8}
    """set[QuantAlgo]: Quantization algorithms that support swiglu_gptoss_style."""

    @classmethod
    def can_implement(
        cls,
        quant_algo: Optional[QuantAlgo],
        dtype_activation: torch.dtype = torch.bfloat16,
        swiglu_gptoss_style: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if CutlassFusedMoE can implement the given quantization algorithm.

        CutlassFusedMoE supports:
        - Unquantized (FP16/BF16): SM >= 80
        - FP8 per-tensor (QDQ): SM >= 89
        - FP8_BLOCK_SCALES: SM in {90, 120}
        - NVFP4: SM in {100, 103, 120, 121}
        - W4A8_AWQ: SM in {89, 90} only
        - W8A16: SM >= 80
        - W4A16_MXFP4: SM == 90 only
        - W4A8_MXFP4_FP8: SM in {100, 103}
        - W4A8_MXFP4_MXFP8: SM in {100, 103, 120, 121}

        Args:
            quant_algo: The quantization algorithm to check (None for unquantized)
            dtype_activation: The activation input data type (before quantization).
                Supported dtypes vary by quantization mode:
                - Unquantized: float16, bfloat16
                - FP8/FP8_BLOCK_SCALES/W4A8_MXFP4_FP8: float16, bfloat16, float32
                - NVFP4: float16, bfloat16, float8_e4m3fn
                - W4A16_MXFP4/W4A8_AWQ/W8A16/W4A8_MXFP4_MXFP8: float16, bfloat16
            swiglu_gptoss_style: Whether swiglu_gptoss_style (bias/swiglu with custom alpha/beta/limit) is enabled.
                CutlassFusedMoE only supports swiglu_gptoss_style for W4A8_MXFP4_MXFP8 quantization.

        Returns:
            Tuple[bool, Optional[str]]: (can_implement, skip_reason)
        """
        from .interface import _warn_and_return

        sm_version = get_sm_version()

        # Check minimum SM version for Cutlass backend
        if sm_version < 80:
            return _warn_and_return(
                f"CutlassFusedMoE requires SM >= 80, got SM{sm_version}")

        # Check swiglu_gptoss_style support
        if swiglu_gptoss_style and quant_algo not in cls._GPTOSS_SUPPORTED_ALGOS:
            return _warn_and_return(
                f"CutlassFusedMoE swiglu_gptoss_style only supports W4A8_MXFP4_MXFP8 "
                f"(got quant_algo={quant_algo})")

        # Check if quant_algo is supported
        if quant_algo not in cls._QUANT_SUPPORT_TABLE:
            return _warn_and_return(
                f"CutlassFusedMoE does not support quant_algo={quant_algo}")

        support_info = cls._QUANT_SUPPORT_TABLE[quant_algo]

        # Check SM version constraint
        constraint_type, constraint_value = support_info["sm_constraint"]
        algo_name = "unquantized" if quant_algo is None else quant_algo.name

        if constraint_type == "min":
            if sm_version < constraint_value:
                return _warn_and_return(
                    f"CutlassFusedMoE {algo_name} requires SM >= {constraint_value}, "
                    f"got SM{sm_version}")
        elif constraint_type == "exact":
            if sm_version != constraint_value:
                return _warn_and_return(
                    f"CutlassFusedMoE {algo_name} only supports SM{constraint_value}, "
                    f"got SM{sm_version}")
        elif constraint_type == "in":
            if sm_version not in constraint_value:
                sm_list = "/".join(f"SM{v}" for v in sorted(constraint_value))
                return _warn_and_return(
                    f"CutlassFusedMoE {algo_name} only supports {sm_list}, "
                    f"got SM{sm_version}")

        # Check dtype_activation
        supported_dtypes = support_info["dtypes"]
        if dtype_activation not in supported_dtypes:
            dtype_list = ", ".join(str(d) for d in supported_dtypes)
            return _warn_and_return(
                f"CutlassFusedMoE {algo_name} requires {dtype_list}, "
                f"got {dtype_activation}")

        return True, None

    def __init__(
        self,
        *,
        routing_method: BaseMoeRoutingMethod,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        model_config: ModelConfig = ModelConfig(),
        aux_stream_dict: Optional[Dict[AuxStreamType,
                                       torch.cuda.Stream]] = None,
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.
        VANILLA,
        bias: bool = False,
        apply_router_weight_on_input: bool = False,
        layer_idx: Optional[int] = None,
        swiglu_alpha: Optional[torch.Tensor] = None,
        swiglu_beta: Optional[torch.Tensor] = None,
        swiglu_limit: Optional[torch.Tensor] = None,
        swiglu_limit_scalar: Optional[float] = None,
        init_load_balancer: bool = True,
        without_comm: bool = False,
        activation_type: ActivationType = ActivationType.Swiglu,
    ):

        super().__init__(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            weight_loading_mode=weight_loading_mode,
            bias=bias,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            swiglu_limit_scalar=swiglu_limit_scalar,
            layer_idx=layer_idx,
            init_load_balancer=init_load_balancer,
            activation_type=activation_type,
        )

        # Store original hidden size before any potential padding
        self.unpadded_hidden_size = self.hidden_size

        if model_config.quant_config and model_config.quant_config.layer_quant_mode.has_w4a16_mxfp4(
        ):
            self.hidden_size = ((self.hidden_size + 127) // 128) * 128
            self.intermediate_size_per_partition = (
                (self.intermediate_size_per_partition + 127) // 128) * 128

        # Note: num_slots, expert_size_per_partition, initial_global_assignments,
        # slot_start, slot_end, initial_local_expert_ids are all initialized by
        # base class's _init_load_balancer() method

        # moe_max_num_tokens is set in ModelConfig.__post_init__ if not specified
        # The default value is max_num_tokens * dp_size
        self.moe_max_num_tokens = model_config.moe_max_num_tokens
        # The auxiliary CUDA stream and CUDA events are only used when MoE chunking is applied
        default_moe_max_num_tokens = model_config.max_num_tokens * model_config.mapping.dp_size
        if self.moe_max_num_tokens < default_moe_max_num_tokens:
            self.aux_stream = aux_stream_dict[
                AuxStreamType.
                MoeChunkingOverlap] if aux_stream_dict is not None else torch.cuda.Stream(
                )
            self.event_dict = {
                key: torch.cuda.Event()
                for key in [EventType.Main, EventType.MoeChunkingOverlap]
            }
        else:
            self.aux_stream = None
            self.event_dict = None

        # The profiler converges on the same best tactic when the number of tokens is large enough.
        # To avoid long profiling time, the max number of tokens used in the profiling is capped to
        # around 16k tokens per expert, which is well into the compute bound domain.
        self.tune_max_num_tokens = min(
            self.moe_max_num_tokens,
            16384 * self.num_slots // routing_method.get_experts_per_token(),
        )
        self.has_been_profiled = False
        self.has_been_profiled_min_latency = False

        # When without_comm=True, skip communication initialization (ConfigurableMoE will handle it)
        if not without_comm:
            self.alltoall_method_type = self.select_alltoall_method_type()
            logger.info_once(
                f"{self.__class__.__name__} selects alltoall_method_type {self.alltoall_method_type!r}",
                key="alltoall_method_type")
            self.alltoall_workspace = None
            self.alltoall_prepare_workspace = None
            self.use_low_precision_combine = False
            if self.enable_alltoall:
                self.use_low_precision_combine = model_config.use_low_precision_moe_combine

                if self.alltoall_method_type == AlltoallMethodType.NVLinkTwoSided:
                    MnnvlMemory.initialize()
                    self.alltoall_workspace = MnnvlMoe.get_moe_workspaces(
                        model_config.mapping)
                    self.alltoall_prepare_workspace = MnnvlMoe.get_moe_prepare_workspace(
                        model_config.mapping)
                elif self.alltoall_method_type == AlltoallMethodType.NVLinkOneSided:
                    # Calculate required workspace size
                    ep_size = self.mapping.moe_ep_size
                    max_num_tokens = model_config.max_num_tokens
                    hidden_size = self.hidden_size
                    dtype = self.dtype or torch.float16

                    workspace_size = MoeAlltoAll.calculate_required_workspace_size(
                        ep_size,
                        self.routing_method.experts_per_token,
                        max_num_tokens,
                        hidden_size,
                        dtype,
                        self.num_experts if self.layer_load_balancer else None,
                    )
                    ep_group_health, watchdog_timeout_s, watchdog_poll_interval_s = (
                        get_wide_ep_ft_options(model_config))

                    self.moe_a2a = MoeAlltoAll(
                        mapping=self.mapping,
                        max_num_tokens=model_config.max_num_tokens,
                        top_k=self.routing_method.experts_per_token,
                        num_slots=self.num_slots,
                        workspace_size_per_rank=workspace_size,
                        num_experts=self.num_experts
                        if self.layer_load_balancer else None,
                        ep_group_health=ep_group_health,
                        alltoall_watchdog_timeout_s=watchdog_timeout_s,
                        alltoall_watchdog_poll_interval_s=
                        watchdog_poll_interval_s,
                    )
                elif self.alltoall_method_type == AlltoallMethodType.DeepEP or self.alltoall_method_type == AlltoallMethodType.DeepEPLowLatency:
                    raise NotImplementedError(
                        "DeepEP and DeepEPLowLatency are not supported for CutlassFusedMoE yet"
                    )
                else:
                    raise NotImplementedError(
                        f"Unsupported alltoall method type: {self.alltoall_method_type!r}"
                    )
        else:
            # When without_comm=True, set minimal attributes
            # Communication will be handled by parent wrapper (e.g., ConfigurableMoE)
            self.alltoall_method_type = AlltoallMethodType.NotEnabled
            self.alltoall_workspace = None
            self.alltoall_prepare_workspace = None
            self.use_low_precision_combine = False
            self.moe_a2a = None

        # If True, the router weight will be multiplied on the input rather than at the end of FC2
        self.apply_router_weight_on_input = apply_router_weight_on_input

        # Finalize fusion should be disabled if Lora is used.
        self.use_fused_finalize = not model_config.moe_disable_finalize_fusion and model_config.lora_config is None

        # Routed-expert LoRA is fused inside torch.ops.trtllm.fused_moe. This
        # flag records whether the layer was configured with MoE LoRA targets,
        # so forward_impl can reject stray lora_params instead of ignoring them.
        self._moe_lora_enabled = self._has_moe_lora_targets(model_config)

        # Discovery-only marker submodule. The actual LoRA GEMMs are fused into
        # torch.ops.trtllm.fused_moe; MoeLoraLayer exists purely so that
        # CudaGraphLoraManager and the target-module validator can find this MoE
        # layer via isinstance(child, LoraLayer) traversal and read its
        # lora_module_types / output_hidden_sizes when building slot tables.
        self.lora = self._maybe_make_lora_marker(model_config)

        self._weights_created = False
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    # ---- Routed-expert LoRA helpers ----

    def _has_moe_lora_targets(self, model_config: ModelConfig) -> bool:
        """Return True iff this MoE layer is in the routed-expert LoRA
        target-module set. The LoRA application itself is fused into
        `torch.ops.trtllm.fused_moe`; no submodule is registered.
        """
        return has_moe_lora_targets(getattr(model_config, "lora_config", None))

    def _maybe_make_lora_marker(
            self, model_config: ModelConfig) -> Optional[MoeLoraLayer]:
        """Construct a MoeLoraLayer marker iff this MoE layer is in the LoRA
        target-module set. The marker is a discovery-only submodule; the actual
        LoRA application is fused into torch.ops.trtllm.fused_moe.

        The output_hidden_sizes recorded here are the per-token outputs of the
        LoRA-side GEMM (not per-expert weight shapes): MOE_H_TO_4H / MOE_GATE
        produce intermediate_size, MOE_4H_TO_H produces hidden_size.
        """
        lora_config = getattr(model_config, "lora_config", None)
        if lora_config is None:
            return None
        # Normalize to lowercase to match has_moe_lora_targets (which lowercases
        # before comparing), so a mixed-case config marks the layer and builds
        # the discovery marker consistently.
        targets = {
            name.lower()
            for name in (getattr(lora_config, "lora_target_modules", []) or [])
        }
        active_modules: List[LoraModuleType] = []
        active_out_sizes: List[int] = []
        for name in MOE_LORA_MODULE_NAMES:
            if name not in targets:
                continue
            module_type = LoraModuleType.from_string(name)
            if name == "moe_4h_to_h":
                active_out_sizes.append(self.hidden_size)
            else:
                active_out_sizes.append(self.intermediate_size)
            active_modules.append(module_type)
        if not active_modules:
            return None
        return MoeLoraLayer(active_modules, active_out_sizes)

    def reserve_moe_lora_cuda_graph_workspace(self, max_num_tokens: int,
                                              max_lora_rank: int,
                                              max_lora_size: int) -> None:
        """Pre-size the C++ FusedMoeRunner's MoE-LoRA scratch to the engine's
        worst case so no (re)allocation happens during CUDA graph capture or
        replay (which would dangle addresses baked into earlier graphs).

        No-op for layers without MoE LoRA targets and for quantized layers (MoE
        LoRA requires unquantized fp16/bf16); idempotent and grow-only. Call
        during warmup, before any capture that exercises MoE LoRA;
        CudaGraphLoraManager does this automatically.

        Args:
            max_num_tokens: Worst-case tokens in a captured forward
                (max_batch_size * max_tokens_per_seq).
            max_lora_rank: Largest LoRA rank across adapters.
            max_lora_size: Adapter-slot pool size for the slot-indexed device tables.
        """
        if not self._moe_lora_enabled or max_num_tokens <= 0:
            return
        # MoE LoRA only runs on the unquantized fp16/bf16 path (the C++ op
        # rejects quantized weights), so a quantized layer can never reach the
        # LoRA scratch; skip and let the (impossible) runtime path error loudly.
        if getattr(self, "has_any_quant", False):
            return
        # Weights must exist to read the runner's weight dtype. If they have not
        # been created yet, skip; the lazy sizing + in-capture guard still
        # protect correctness.
        if getattr(self, "w3_w1_weight", None) is None:
            return

        # The reservation must cover the engine's worst case, otherwise the first
        # capture hits a lazy allocation that the C++ in-capture guard rejects.
        assert max_lora_rank > 0, (
            "reserve_moe_lora_cuda_graph_workspace requires max_lora_rank > 0 "
            f"(got {max_lora_rank}); set lora_config.max_lora_rank.")
        assert max_lora_size > 0, (
            "reserve_moe_lora_cuda_graph_workspace requires max_lora_size > 0 "
            f"(got {max_lora_size}).")
        # MoE LoRA only runs on the unquantized fp16/bf16 path, so the MoERunner
        # instance key below (all-False quant flags, x/weight/out == self.dtype)
        # must match the key the runtime fused_moe op uses on the same layer;
        # otherwise the reservation lands on a different cached C++ runner.
        assert self.dtype in (torch.float16, torch.bfloat16), (
            "MoE LoRA requires fp16/bf16 activations to reserve a deterministic "
            f"FusedMoeRunner key; got {self.dtype}.")

        from ...custom_ops.torch_custom_ops import MoERunner

        # Build the MoERunner with the same instance key the functional
        # torch.ops.trtllm.fused_moe op uses, so we reserve on the *same* cached
        # C++ FusedMoeRunner that capture will use. For the unquantized LoRA
        # path x/weight/output dtypes all equal self.dtype and every quant flag
        # is False. If a runtime call ever uses a different key, the C++
        # capture guard surfaces a clear error rather than corrupting replay.
        weight_dtype = self.w3_w1_weight.dtype
        runner = MoERunner(
            x_dtype=self.dtype,
            weight_dtype=weight_dtype,
            output_dtype=self.dtype,
            top_k=self.routing_method.experts_per_token,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            cluster_size=self.cluster_size,
            cluster_rank=self.cluster_rank,
            use_deepseek_fp8_block_scale=False,
            use_w4_group_scaling=False,
            use_int8_woq_per_channel=False,
            use_mxfp8_act_scaling=False,
            min_latency_mode=False,
            use_fused_finalize=self.use_fused_finalize,
            activation_type=self.activation_type,
        )
        runner.fused_moe_runner.reserve_lora_host_buffers(
            int(max_num_tokens),
            int(self.routing_method.experts_per_token),
            int(max_lora_rank),
            int(max_lora_size),
            bool(self.is_gated_activation),
        )

    def _moe_lora_active(self, lora_params: Optional[Dict]) -> bool:
        """Return True when lora_params carries routed-expert MoE LoRA tensors
        for this layer, meaning run_moe would fuse a LoRA delta.
        """
        if not lora_params or self.layer_idx is None:
            return False
        # CUDA-graph slot-indexed mode carries MoE LoRA in cuda_graph_params
        # rather than a per-layer eager dict (mirrors _extract_moe_lora_tensors),
        # so consult the graph layer map to keep the stray-param and multi-chunk
        # guards effective during capture/replay.
        if lora_params.get("use_cuda_graph_mode", False):
            cuda_graph_params = lora_params.get("cuda_graph_params")
            if cuda_graph_params is None:
                return False
            layer_module2key = getattr(cuda_graph_params, "layer_module2key",
                                       {})
            return any(
                (self.layer_idx,
                 int(LoraModuleType.from_string(name))) in layer_module2key
                for name in MOE_LORA_MODULE_NAMES)
        layer_params = lora_params.get(self.layer_idx, {})
        if not layer_params:
            return False
        return any(
            int(LoraModuleType.from_string(name)) in layer_params
            for name in MOE_LORA_MODULE_NAMES)

    @staticmethod
    def _empty_kernel_slot_dict() -> Dict[str, Optional[torch.Tensor]]:
        return {"fc1": None, "fc2": None, "gated": None}

    def _gather_moe_lora_slots(self, source):
        """Gather per-kernel-slot (ranks, weight_ptrs) tensors.

        `source(module_type)` returns the (ranks, weight_ptrs) pair for an MoE
        LoRA module, or None if absent. Returns (ranks_by_slot, ptrs_by_slot)
        dicts keyed by the kernel slot ("fc1"/"gated"/"fc2"); see
        MOE_LORA_MODULE_TO_KERNEL_SLOT for the module->slot convention. Shared by
        the eager (per-request) and CUDA-graph (slot-indexed) extraction paths.
        """
        ranks = self._empty_kernel_slot_dict()
        ptrs = self._empty_kernel_slot_dict()
        for module_type, slot in MOE_LORA_MODULE_TO_KERNEL_SLOT.items():
            got = source(module_type)
            if got is None:
                continue
            ranks[slot], ptrs[slot] = got
        return ranks, ptrs

    @staticmethod
    def _require_fc1_fc2(ranks: Dict[str, Optional[torch.Tensor]]) -> None:
        """The kernel always dereferences the fc1 and fc2 rank/pointer arrays
        (see setupLoraWorkspace in moe_kernels.cu), so moe_h_to_4h (fc1/gate) and
        moe_4h_to_h (fc2/down) must both be present when MoE LoRA is active. The
        gated slot (moe_gate) is only read for gated activations.
        """
        if ranks["fc1"] is None or ranks["fc2"] is None:
            raise ValueError(
                "MoE LoRA requires both `moe_h_to_4h` (gate/SiLU) and "
                "`moe_4h_to_h` (down) in lora_target_modules.")

    def _extract_moe_lora_tensors(
            self, lora_params: Optional[Dict]) -> Optional[Dict[str, object]]:
        """Pick the MoE-side LoRA tensors out of the global `lora_params` dict
        for this layer. Returns a dict with the kwargs expected by
        `torch.ops.trtllm.fused_moe`, or None when no MoE LoRA applies.

        Each entry is a CPU tensor:
            *_lora_ranks         : int32  [num_seqs]
            *_lora_weight_ptrs   : int64  [num_seqs, 3]   (A, B, DoRA_unused)
            host_request_types   : int32  [num_seqs]      (0=CTX, 1=GEN)
            host_context_lengths : int32  [num_seqs]
            lora_max_low_rank    : int (max rank across the active modules)
        """
        if not lora_params:
            return None
        # Slot-indexed (CUDA-graph decode) path: the per-token expansion is
        # driven inside the op by token_to_slot indexed into stable slot
        # tables owned by CudaGraphLoraParams (see _extract_moe_lora_tensors_cuda_graph).
        if lora_params.get("use_cuda_graph_mode", False):
            return self._extract_moe_lora_tensors_cuda_graph(lora_params)
        layer_params = lora_params.get(
            self.layer_idx, {}) if self.layer_idx is not None else {}
        if not layer_params:
            return None

        # Gather (ranks, weight_ptrs) per kernel slot. weight_pointers is built
        # flat ([num_seqs * 3], row-major (A, B, DoRA) per seq) in
        # PyTorchModelEngine._build_lora_params; the op expects [num_seqs, 3].
        active_max_rank = 0

        def _source(module_type: LoraModuleType):
            nonlocal active_max_rank
            entry = layer_params.get(int(module_type))
            if entry is None:
                return None
            rank_t = entry["adapter_size"]
            if rank_t.numel() > 0:
                active_max_rank = max(active_max_rank, int(rank_t.max().item()))
            return rank_t, entry["weight_pointers"].reshape(-1, 3)

        ranks, ptrs = self._gather_moe_lora_slots(_source)
        if all(v is None for v in ranks.values()):
            return None
        self._require_fc1_fc2(ranks)

        num_seqs = lora_params["num_seqs"]

        def _slice(t):
            return t[:num_seqs].contiguous() if t is not None else None

        return {
            "fc1_lora_ranks": _slice(ranks["fc1"]),
            "fc1_lora_weight_ptrs": _slice(ptrs["fc1"]),
            "fc2_lora_ranks": _slice(ranks["fc2"]),
            "fc2_lora_weight_ptrs": _slice(ptrs["fc2"]),
            "gated_lora_ranks": _slice(ranks["gated"]),
            "gated_lora_weight_ptrs": _slice(ptrs["gated"]),
            "host_request_types": _slice(lora_params["host_request_types"]),
            "host_context_lengths": _slice(lora_params["prompt_lens_cpu"]),
            "lora_max_low_rank": active_max_rank,
        }

    def _extract_moe_lora_tensors_cuda_graph(
            self, lora_params: Dict) -> Optional[Dict[str, object]]:
        """CUDA-graph slot-indexed extraction for routed-expert MoE LoRA.

        Pulls per-module slot tables and token_to_slot out of
        CudaGraphLoraParams and returns the slot-indexed kwargs accepted by
        torch.ops.trtllm.fused_moe. Returns None when this layer does not
        carry any MoE LoRA modules in the graph layer map.

        Returned tensor addresses are stable across captures and replays: they
        come from persistent pinned host buffers owned by CudaGraphLoraParams
        and the per-module packed pointer cache. Uses the same module->kernel
        slot convention as the per-request path (moe_h_to_4h -> fc1,
        moe_gate -> gated, moe_4h_to_h -> fc2).
        """
        if self.layer_idx is None:
            return None
        cuda_graph_params = lora_params.get("cuda_graph_params")
        if cuda_graph_params is None:
            return None

        def _source(module_type: LoraModuleType):
            return cuda_graph_params.get_moe_slot_inputs(
                self.layer_idx, int(module_type))

        slot_ranks, slot_ptrs = self._gather_moe_lora_slots(_source)
        if slot_ranks["fc1"] is None or slot_ranks["fc2"] is None:
            return None

        num_seqs = lora_params["num_seqs"]
        tokens_per_seq = getattr(cuda_graph_params, "max_tokens_per_seq", 1)
        num_tokens = num_seqs * max(int(tokens_per_seq), 1)
        token_to_slot = cuda_graph_params.token_to_slot_host[:
                                                             num_tokens].contiguous(
                                                             )

        # Pass the global max LoRA rank, not the per-step active max: the device
        # path uses it only to size the low-rank workspace strides baked into the
        # captured graph, so the global max keeps them valid for any per-slot
        # rank across replays. The actual per-token rank is read on-device from
        # the slot table, so a smaller rank just runs a smaller GEMM.
        max_rank = int(getattr(cuda_graph_params, "max_rank", 0))
        if max_rank <= 0:
            return None

        return {
            "fc1_slot_lora_ranks":
            slot_ranks["fc1"].contiguous(),
            "fc1_slot_lora_weight_ptrs":
            slot_ptrs["fc1"].contiguous(),
            "fc2_slot_lora_ranks":
            slot_ranks["fc2"].contiguous(),
            "fc2_slot_lora_weight_ptrs":
            slot_ptrs["fc2"].contiguous(),
            "gated_slot_lora_ranks":
            (slot_ranks["gated"].contiguous()
             if slot_ranks["gated"] is not None else None),
            "gated_slot_lora_weight_ptrs":
            (slot_ptrs["gated"].contiguous()
             if slot_ptrs["gated"] is not None else None),
            "token_to_slot":
            token_to_slot,
            "lora_max_low_rank":
            max_rank,
        }

    def _check_configs(self):
        assert self._weights_created

        if self.apply_router_weight_on_input:
            assert self.routing_method.top_k == 1, "Current walkaround only supports top-1 routing"

        if self.quant_config and self.quant_config.quant_mode.has_any_quant(
                exclude_kv_cache=True):
            if not (self.quant_config.quant_mode.has_nvfp4()
                    | self.quant_config.quant_mode.has_fp8_block_scales()
                    | self.quant_config.quant_mode.has_fp8_qdq()
                    | self.quant_config.quant_mode.is_weight_only()
                    | self.quant_config.quant_mode.has_w4a8_mxfp4_fp8()
                    | self.quant_config.quant_mode.has_w4a16_mxfp4()
                    | self.quant_config.quant_mode.has_w4a8_mxfp4_mxfp8()
                    | self.quant_config.quant_mode.has_mxfp8()):
                raise ValueError(
                    f"unsupported quantization mode: {self.quant_config.quant_mode}"
                )

    @property
    def has_w4afp8(self):
        assert self._weights_created
        return self.quant_config and self.quant_config.quant_mode.is_int4_weight_only_per_group(
        )

    @property
    def has_int8_woq_per_channel(self):
        return self.quant_config and self.quant_config.layer_quant_mode.is_int8_weight_only(
        ) and not self.quant_config.layer_quant_mode.has_per_group_scaling()

    def select_alltoall_method_type(self) -> AlltoallMethodType:
        # If no attention DP, no need to use AlltoAll.
        if self.mapping.dp_size == 1:
            return AlltoallMethodType.NotEnabled

        # AlltoAll cannot support MoE TP.
        if self.mapping.moe_tp_size != 1:
            return AlltoallMethodType.NotEnabled

        if not MnnvlMemory.supports_mnnvl():
            return AlltoallMethodType.NotEnabled

        all2all_method_type = os.environ.get("TRTLLM_FORCE_ALLTOALL_METHOD")
        if all2all_method_type is not None:
            if AlltoallMethodType[all2all_method_type] in [
                    AlltoallMethodType.DeepEP,
                    AlltoallMethodType.DeepEPLowLatency
            ]:
                raise NotImplementedError(
                    "DeepEP and DeepEPLowLatency are not supported for CutlassFusedMoE yet"
                )
            return AlltoallMethodType[all2all_method_type]

        # TODO: We found that NVLinkOneSided performs better than NCCL AllGather/ReduceScatter,
        # regardless of the relationship between EP size and topK. We favor NVLinkOneSided for now.
        # if not self.mapping.moe_ep_size > self.routing_method.experts_per_token:
        #     return AlltoallMethodType.NotEnabled
        return AlltoallMethodType.NVLinkOneSided

    @cached_property
    def enable_alltoall(self):
        """ enable_alltoall (bool): whether to enable alltoall instead of allgather/reducescatter
        """
        return self.alltoall_method_type != AlltoallMethodType.NotEnabled

    def quantize_input(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        post_quant_comm: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Quantize input tensor - CutlassFusedMoE implementation

        Handles all quantization cases for Cutlass backend.

        Args:
            x: Input tensor to quantize
            post_quant_comm: Whether this is for post-quantization communication
                           (allgather or alltoall). If True, x_sf will be reshaped to 2D.

        Returns:
            Tuple of (quantized_x, x_sf)
        """
        x_sf = None
        if self.has_any_quant:
            # W4A16 NVFP4 path keeps activations hp; skip FP4 quant below.
            if isinstance(self.quant_method, W4A16NVFP4CutlassFusedMoEMethod):
                return x, None
            if self.has_fp8_qdq or self.has_w4a8_mxfp4_fp8:
                x, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                    x, self.fc31_input_dequant)
            elif self.has_deepseek_fp8_block_scales:
                # No quantization needed here, handled in kernel
                pass
            elif self.has_w4afp8:
                # No quantization needed here, handled in kernel
                pass
            elif self.has_w4a16_mxfp4:
                # Padding deferred to run_moe so that dispatch sends
                # unpadded tensors (avoids NVLink workspace overallocation).
                pass
            elif self.has_int8_woq_per_channel:
                # No quantization needed here, handled in kernel
                pass
            elif self.has_nvfp4:
                if hasattr(
                        self,
                        'fc31_act_scale') and self.fc31_act_scale is not None:
                    assert not isinstance(
                        x, Fp4QuantizedTensor
                    ), "Fp4QuantizedTensor is not expected for AWQ quantization."
                    x = x * self.fc31_act_scale

                # Dynamic quantization: compute input_scale from current input
                # and update alpha in-place (same tensor addresses for CUDA graph).
                if self.force_dynamic_quantization and hasattr(
                        self, 'fc31_weight_scale_2'):
                    FP8_MAX, E2M1_MAX = 448.0, 6.0
                    amax_input = torch.amax(torch.abs(x)).float()
                    dyn_input_scale = FP8_MAX * E2M1_MAX / amax_input

                    # fc31_alpha[e] = weight_scale_2[e] / dyn_input_scale
                    self.fc31_alpha.data.copy_(self.fc31_weight_scale_2.data /
                                               dyn_input_scale)
                    self.fc31_input_scale.data.copy_(dyn_input_scale)

                # Quantize based on communication scenario
                if post_quant_comm:
                    if isinstance(x, Fp4QuantizedTensor):
                        assert not x.is_sf_swizzled, "Fp4QuantizedTensor should not be swizzled before communication"
                        x, x_sf = x.fp4_tensor, x.scaling_factor
                        x_row = x.shape[0]
                    else:
                        x_row = x.shape[0]
                        x, x_sf = torch.ops.trtllm.fp4_quantize(
                            x, self.fc31_input_scale, self.scaling_vector_size,
                            False, False)
                    # Reshape x_sf to 2D for post-quant communication
                    if x_sf is not None:
                        x_sf = x_sf.view((x_row, -1))
                else:
                    if not isinstance(x, Fp4QuantizedTensor):
                        x, x_sf = torch.ops.trtllm.fp4_quantize(
                            x, self.fc31_input_scale, self.scaling_vector_size,
                            False, True)
            elif self.has_w4a8_mxfp4_mxfp8 or self.has_mxfp8:
                # MXFP8 dynamic activation quantize. The MXFP8xMXFP8 path reuses
                # the same activation quant kernel as W4A8 MXFP4xMXFP8 -- only
                # the weight side differs (B element widens from fp4 to fp8).
                if post_quant_comm:
                    x, x_sf = torch.ops.trtllm.mxfp8_quantize(
                        x, False, alignment=self.quant_method.weight_alignment)
                    # Reshape x_sf to 2D for post-quant communication
                    # x.shape[0] is padded
                    if x_sf is not None:
                        x_sf = x_sf.view((x.shape[0], -1))
                else:
                    x, x_sf = torch.ops.trtllm.mxfp8_quantize(
                        x, True, alignment=self.quant_method.weight_alignment)
            else:
                raise ValueError(
                    f"unsupported quantization mode: {self.quant_config.quant_mode}"
                )

        return x, x_sf

    def _supports_load_balancer(self) -> bool:
        """CutlassFusedMoE supports load balancer."""
        return True

    def _get_quant_method(self):
        if self.quant_config is not None and self.quant_config.layer_quant_mode.has_any_quant(
                exclude_kv_cache=True):
            if self.quant_config.layer_quant_mode.has_fp8_qdq():
                return FP8QDQFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_fp8_block_scales():
                return DeepSeekFP8BlockScalesFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_nvfp4():
                return NVFP4CutlassFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.is_int4_weight_only_per_group(
            ):
                return WInt4AFP8FusedMoEMethod()
            elif self.has_int8_woq_per_channel:
                return INT8WoqPerChannelFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_w4a8_mxfp4_fp8():
                return W4A8MXFP4FP8CutlassFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_w4a16_mxfp4():
                return WFP4A16FusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_w4a8_mxfp4_mxfp8():
                return W4A8MXFP4MXFP8CutlassFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_mxfp8():
                return MXFP8CutlassFusedMoEMethod()
            else:
                raise ValueError(
                    f"Unsupported quantization mode: {self.quant_config.quant_mode}"
                )
        else:
            return UnquantizedFusedMoEMethod()

    def create_weights(self):
        if self._weights_created:
            return

        self.quant_method = self._get_quant_method()
        self.quant_method.create_weights(self)

        self._weights_created = True
        self._check_configs()

    def supports_moe_output_in_alltoall_workspace(self):
        return True

    def run_moe(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        x_sf: Optional[torch.Tensor] = None,
        is_sf_swizzled: bool = True,
        output_dtype: Optional[torch.dtype] = None,
        tuner_num_tokens: Optional[int] = None,
        tuner_top_k: Optional[int] = None,
        moe_output: Optional[torch.Tensor] = None,
        enable_alltoall: Optional[bool] = None,
        lora_params: Optional[Dict] = None,
    ) -> torch.Tensor:
        """
        Run MoE computation with Cutlass backend.

        This method encapsulates the core MoE computation logic, handling different
        quantization schemes.

        Args:
            x: Input hidden states (may be pre-quantized)
            token_selected_experts: Expert IDs or expert slots [num_tokens, top_k]
                                   If EPLB is enabled, represents expert slots; otherwise expert IDs
            token_final_scales: Final scaling factors for each token
            x_sf: Input scale factors (optional, for certain quantization schemes)
            is_sf_swizzled: Whether scaling factors are swizzled
            output_dtype: Output data type (optional)
            tuner_num_tokens: Number of tokens for profiling tuner (optional)
            tuner_top_k: Top-k value for profiling tuner (optional)
            moe_output: Pre-allocated output buffer (optional)
            enable_alltoall: Whether alltoall communication is enabled (optional). If None, defaults to self.enable_alltoall.

        Returns:
            final_hidden_states: Output tensor from MoE computation
        """
        # W4A16 NVFP4 fallback (SM<100).
        if isinstance(self.quant_method, W4A16NVFP4CutlassFusedMoEMethod):
            return self._run_moe_w4a16_nvfp4(
                x,
                token_selected_experts,
                token_final_scales,
                output_dtype=output_dtype,
                tuner_num_tokens=tuner_num_tokens,
                tuner_top_k=tuner_top_k,
                moe_output=moe_output,
                enable_alltoall=enable_alltoall,
            )

        # SM120 + FP8 block scales: use Triton kernel (CUTLASS TMA fails on SM120
        # for large token counts due to cuTensorMapEncodeTiled limitations).
        if self.has_deepseek_fp8_block_scales and get_sm_version() == 120:
            from .fused_moe_triton_fp8_block_scale import \
                run_triton_fp8_block_scale_moe
            _use_alltoall = (enable_alltoall if enable_alltoall is not None else
                             self.enable_alltoall)
            # forward_chunk sets token_final_scales=None when
            # apply_router_weight_on_input=True (weights already folded into x);
            # substitute ones so the Triton kernel's per-token scaling is a no-op.
            if token_final_scales is None:
                token_final_scales = torch.ones_like(token_selected_experts,
                                                     dtype=torch.float32)
            # token_selected_experts contains GLOBAL expert IDs in the non-alltoall
            # path (slot_start .. slot_end-1 for this rank's local experts, plus
            # IDs for other ranks).  The Triton kernel operates on LOCAL IDs
            # (0 .. expert_size_per_partition-1), so remap and zero-scale any
            # non-local token-expert pairs to suppress their contribution.
            local_n = self.expert_size_per_partition
            if _use_alltoall:
                # After alltoall dispatch, IDs are already local; padding = local_n
                local_ids = token_selected_experts.clamp(0, local_n - 1)
                is_local = token_selected_experts < local_n
            else:
                slot_start = self.slot_start
                local_ids = (token_selected_experts - slot_start).clamp(
                    0, local_n - 1)
                is_local = ((token_selected_experts >= slot_start)
                            & (token_selected_experts < slot_start + local_n))
            local_scales = token_final_scales * is_local.to(
                token_final_scales.dtype)
            result = run_triton_fp8_block_scale_moe(
                x,
                local_ids,
                local_scales,
                self.w3_w1_weight,
                self.quant_scales.fc_weight_scales,
                self.w2_weight,
                self.quant_scales.proj_weight_scales,
                activation_type=self.activation_type,
                output_dtype=output_dtype,
            )
            return result

        # Pad input for mxfp4 alignment (128-aligned hidden_size).
        # Done here rather than in quantize_input so that dispatch sends
        # unpadded tensors and avoids NVLink workspace overallocation.
        if self.has_w4a16_mxfp4:
            pad_size = self.hidden_size - x.shape[-1]
            if pad_size > 0:
                x = torch.nn.functional.pad(x, (0, pad_size))

        # Determine weight dtype based on quantization mode
        weight_dtype = self.w3_w1_weight.dtype
        if self.has_any_quant:
            if self.has_w4afp8:
                weight_dtype = torch.quint4x2
            elif self.has_w4a16_mxfp4:
                weight_dtype = torch.uint8

        if enable_alltoall is None:
            enable_alltoall = self.enable_alltoall

        use_dynamic_fc2_scale = (self.has_nvfp4 and getattr(
            self, 'force_dynamic_quantization', False)
                                 and hasattr(self, 'fc2_weight_scale_2'))

        lora_kwargs = self._extract_moe_lora_tensors(lora_params)
        if lora_kwargs is None:
            lora_kwargs = {}

        result = torch.ops.trtllm.fused_moe(
            x,
            token_selected_experts,
            token_final_scales,
            self.w3_w1_weight.view(weight_dtype),
            self.w3_w1_bias,
            self.w2_weight.view(weight_dtype),
            self.w2_bias,
            output_dtype,
            quant_scales=list(self.quant_scales) +
            ([self.fc2_weight_scale_2] if use_dynamic_fc2_scale else []),
            input_sf=x_sf,
            swizzled_input_sf=is_sf_swizzled,
            swiglu_alpha=self.swiglu_alpha,
            swiglu_beta=self.swiglu_beta,
            swiglu_limit=self.swiglu_limit,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            cluster_size=self.cluster_size,
            cluster_rank=self.cluster_rank,
            enable_alltoall=enable_alltoall,
            use_deepseek_fp8_block_scale=self.has_deepseek_fp8_block_scales,
            use_w4_group_scaling=self.has_w4afp8 or self.has_w4a16_mxfp4,
            use_int8_woq_per_channel=self.has_int8_woq_per_channel,
            # use_mxfp8_act_scaling drives dynamic MXFP8 activation quantization
            # before the GEMM; required for both W4A8 MXFP4xMXFP8 and W8A8
            # MXFP8xMXFP8 paths.
            use_mxfp8_act_scaling=self.has_w4a8_mxfp4_mxfp8 or self.has_mxfp8,
            min_latency_mode=False,
            use_fused_finalize=self.use_fused_finalize,
            tune_max_num_tokens=self.tune_max_num_tokens,
            tuner_num_tokens=tuner_num_tokens,
            tuner_top_k=tuner_top_k,
            activation_type=self.activation_type,
            unpadded_hidden_size=self.unpadded_hidden_size,
            out_tensor=moe_output,
            use_dynamic_fc2_scale=use_dynamic_fc2_scale,
            # use_mxfp8_weight_scaling selects the MXFP8xMXFP8 block-scaled
            # kernel path within the <e4m3, e4m3> CutlassMoeFCRunner template
            # (per-tensor FP8 otherwise).
            use_mxfp8_weight_scaling=self.has_mxfp8,
            **lora_kwargs,
        )
        # When moe_output is provided, the result is written in-place and
        # fused_moe returns empty list to avoid aliasing constraint violation.
        # Otherwise, unpack the single tensor from the returned list.
        if moe_output is not None:
            final_hidden_states = moe_output
        else:
            final_hidden_states = result[0]

        return final_hidden_states

    def _run_moe_w4a16_nvfp4(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        output_dtype: Optional[torch.dtype] = None,
        tuner_num_tokens: Optional[int] = None,
        tuner_top_k: Optional[int] = None,
        moe_output: Optional[torch.Tensor] = None,
        enable_alltoall: Optional[bool] = None,
    ) -> torch.Tensor:
        """W4A16 fallback for NVFP4 MoE on SM<100. Active-mask dequant into
        a static [E_total, N, K] bf16 workspace, then bf16 fused_moe with the
        original (global) token_selected_experts. CUDA-graph capturable.
        """
        assert isinstance(self.quant_method, W4A16NVFP4CutlassFusedMoEMethod)

        if enable_alltoall is None:
            enable_alltoall = self.enable_alltoall
        if output_dtype is None:
            output_dtype = x.dtype

        # Same EP id convention as the FP8 path above: global ids (or
        # ``local_n``-padded under alltoall). Clamp to local range so the
        # active-mask scatter is in-bounds; non-local tokens collapse onto a
        # boundary expert (1 extra dequant/rank). ``trtllm.fused_moe`` below
        # still gets the original global ids -- it does its own remap.
        local_n = self.expert_size_per_partition
        if enable_alltoall:
            local_ids = token_selected_experts.clamp(0, local_n - 1)
        else:
            local_ids = (token_selected_experts - self.slot_start).clamp(
                0, local_n - 1)

        w3_w1_hp, w2_hp = self.quant_method.dequant_active_experts_to_hp(
            self, local_ids, output_dtype)

        # bf16 fused_moe with empty quant_scales (matches unquantized path).
        result = torch.ops.trtllm.fused_moe(
            x,
            token_selected_experts,
            token_final_scales,
            w3_w1_hp,
            self.w3_w1_bias,
            w2_hp,
            self.w2_bias,
            output_dtype,
            quant_scales=[],
            input_sf=None,
            swizzled_input_sf=False,
            swiglu_alpha=self.swiglu_alpha,
            swiglu_beta=self.swiglu_beta,
            swiglu_limit=self.swiglu_limit,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            cluster_size=self.cluster_size,
            cluster_rank=self.cluster_rank,
            enable_alltoall=enable_alltoall,
            use_deepseek_fp8_block_scale=False,
            use_w4_group_scaling=False,
            use_int8_woq_per_channel=False,
            use_mxfp8_act_scaling=False,
            min_latency_mode=False,
            use_fused_finalize=self.use_fused_finalize,
            tune_max_num_tokens=self.tune_max_num_tokens,
            tuner_num_tokens=tuner_num_tokens,
            tuner_top_k=tuner_top_k,
            activation_type=self.activation_type,
            unpadded_hidden_size=self.unpadded_hidden_size,
            out_tensor=moe_output,
            use_dynamic_fc2_scale=False,
        )
        if moe_output is not None:
            return moe_output
        return result[0]

    def forward_chunk(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        input_ids: Optional[torch.IntTensor],
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        repeating_info: tuple = (True, True),
        lora_params: Optional[Dict] = None,
    ) -> torch.Tensor:
        if isinstance(x, Fp4QuantizedTensor):
            assert output_dtype is not None
        else:
            output_dtype = x.dtype

        is_first_call, is_last_call = repeating_info

        self._load_balancer_start_wait_gpu_stage(is_first_call)

        # apply routing
        token_selected_experts, token_final_scales = self.routing_method.apply(
            router_logits, input_ids)
        assert token_selected_experts.shape[
            1] == self.routing_method.experts_per_token
        assert token_selected_experts.shape == token_final_scales.shape
        assert token_selected_experts.shape[0] == router_logits.shape[0]
        assert token_final_scales.dtype == torch.float32
        assert token_selected_experts.dtype == torch.int32

        if self.layer_load_balancer:
            self._load_balancer_done_wait_gpu_stage(is_first_call)
            ignore_allreduce = self.enable_alltoall and self.alltoall_method_type in (
                AlltoallMethodType.NVLinkTwoSided,
                AlltoallMethodType.NVLinkOneSided,
            )
            self._load_balancer_update_statistic(
                token_selected_experts,
                is_first_call,
                is_last_call,
                ignore_allreduce=ignore_allreduce)
            token_selected_slots = self._load_balancer_route(
                token_selected_experts, self.use_dp)
        else:
            token_selected_slots = token_selected_experts

        # If load balancer is disabled, the statistics are collected from expert IDs.
        # If load balancer is enabled, the statistics are collected from expert slot IDs.
        ExpertStatistic.set_layer(self.layer_idx)
        ExpertStatistic.maybe_add_info(self.num_slots, token_selected_slots)
        token_selected_slots = get_calibrator().maybe_collect_or_replay_slots(
            self.num_slots, token_selected_slots)

        if self.apply_router_weight_on_input:
            assert x.dtype != torch.float8_e4m3fn, "Current workaround for apply_router_weight_on_input does not support fp8 input"
            x = x * token_final_scales.to(x.dtype)
            # TODO: remove this once we have correct fusedmoe kernel ready
            token_final_scales = None

        run_post_quant_allgather = self.use_dp and self.parallel_size > 1

        # Quantize inputs using extracted method
        # For post_quant_comm scenarios, x_sf will be reshaped to 2D inside quantize_input
        post_quant_comm = run_post_quant_allgather or self.enable_alltoall
        x, x_sf = self.quantize_input(x, post_quant_comm=post_quant_comm)

        # Prepare additional information for profiling in case padding is applied when using alltoall.
        # Only the non-alltoall case is considered for profiling in the warmup phase.
        # Therefore, to get the correct tactics during the actual inference, the inputs to the tuner should be the same as when not using alltoall.
        if self.enable_alltoall:
            if all_rank_num_tokens is not None:
                tuner_num_tokens = sum(all_rank_num_tokens)
            else:
                tuner_num_tokens = x.shape[0] * self.mapping.tp_size
            tuner_top_k = token_selected_slots.shape[1]
        else:
            tuner_num_tokens = None
            tuner_top_k = None

        # Alltoall or allgather for attention DP
        token_count = x.shape[0]
        alltoall_info = None  # Store for later combine
        is_sf_swizzled = True  # In case of post-quant communication, scaling factors will not be swizzled before communication, and swizzling after communication is merged into MoE.
        if self.enable_alltoall:
            assert all_rank_num_tokens is not None, "all_rank_num_tokens required for alltoall"
            # Prepare alltoall indices
            top_k = self.routing_method.experts_per_token
            runtime_max_tokens_per_rank = max(
                all_rank_num_tokens) if all_rank_num_tokens else token_count

            # Handle case where token_final_scales might be None (when apply_router_weight_on_input=True)
            if token_final_scales is None:
                token_final_scales = torch.ones_like(token_selected_slots,
                                                     dtype=torch.float32)

            if self.alltoall_method_type == AlltoallMethodType.NVLinkTwoSided:
                assert self.alltoall_prepare_workspace is not None, "alltoall_prepare_workspace should be initialized"
                if is_last_call:
                    loadbalancer_local_statistic_info = self._load_balancer_get_local_statistic_tensor(
                    )
                else:
                    loadbalancer_local_statistic_info = None
                alltoall_info, gathered_loadbalancer_local_statistic_info = MnnvlMoe.mnnvl_moe_alltoallv_prepare_without_allgather(
                    token_selected_slots, loadbalancer_local_statistic_info,
                    self.alltoall_prepare_workspace,
                    runtime_max_tokens_per_rank, self.ep_rank, self.ep_size,
                    self.num_experts, self.num_slots, top_k)
                if gathered_loadbalancer_local_statistic_info is not None:
                    gathered_loadbalancer_local_statistic_info = gathered_loadbalancer_local_statistic_info.view(
                        (self.mapping.moe_ep_size, self.num_experts))
                    self._load_balancer_update_statistic_with_gathered_statistic(
                        gathered_loadbalancer_local_statistic_info)

                # Dispatch x, x_sf, token_selected_slots, token_final_scales in one alltoall kernel
                x, x_sf, token_selected_slots, token_final_scales = MnnvlMoe.mnnvl_moe_alltoallv(
                    [x, x_sf, token_selected_slots, token_final_scales],
                    alltoall_info, self.alltoall_workspace, self.ep_rank,
                    self.ep_size)

                torch.ops.trtllm.memset_expert_ids(
                    token_selected_slots, alltoall_info.recv_rank_count_cumsum,
                    runtime_max_tokens_per_rank, top_k, self.num_slots,
                    self.ep_size)
            elif self.alltoall_method_type == AlltoallMethodType.NVLinkOneSided:
                # Python MoeAlltoAll path

                payloads = []
                payloads.append(x)
                if x_sf is not None:
                    payloads.append(x_sf)
                    expert_id_payload_index = 2
                else:
                    expert_id_payload_index = 1
                payloads.append(token_selected_slots)
                payloads.append(token_final_scales)

                loadbalancer_local_statistic_info = None
                if self.layer_load_balancer and is_last_call:
                    loadbalancer_local_statistic_info = self._load_balancer_get_local_statistic_tensor(
                    )
                if loadbalancer_local_statistic_info is not None:
                    recv_tensors = self.moe_a2a.dispatch(
                        token_selected_slots,
                        payloads,
                        runtime_max_tokens_per_rank,
                        invalid_token_expert_id=self.
                        num_slots,  # Caution: Cutlass MoE uses num_slots as invalid token expert id
                        expert_id_payload_index=expert_id_payload_index,
                        eplb_local_stats=loadbalancer_local_statistic_info,
                    )
                    gathered_stats = self.moe_a2a._state.eplb_gathered_stats
                    self._load_balancer_update_statistic_with_gathered_statistic(
                        gathered_stats)
                else:
                    recv_tensors = self.moe_a2a.dispatch(
                        token_selected_slots,
                        payloads,
                        runtime_max_tokens_per_rank,
                        invalid_token_expert_id=self.
                        num_slots,  # Caution: Cutlass MoE uses num_slots as invalid token expert id
                        expert_id_payload_index=expert_id_payload_index,
                    )

                if x_sf is not None:
                    x_recv, x_sf_recv, token_selected_slots_recv, token_final_scales_recv = recv_tensors
                    x_sf = x_sf_recv.view(-1, x_sf_recv.shape[-1])
                else:
                    x_recv, token_selected_slots_recv, token_final_scales_recv = recv_tensors
                x = x_recv.view(-1, x_recv.shape[-1])
                token_selected_slots = token_selected_slots_recv.view(
                    -1, token_selected_slots_recv.shape[-1])
                token_final_scales = token_final_scales_recv.view(
                    -1, token_final_scales_recv.shape[-1])
            else:
                raise ValueError(
                    f"Unsupported moe alltoall method type: {self.alltoall_method_type}"
                )

        elif run_post_quant_allgather:
            # Original allgather logic
            # x_sf is already 2D after quantize_input with post_quant_comm=True

            x, x_sf, token_selected_slots, token_final_scales = allgather(
                [x, x_sf, token_selected_slots, token_final_scales],
                self.mapping,
                dim=0,
                sizes=None if use_dp_padding else all_rank_num_tokens)

        # Optionally provide an output tensor to fused_moe so it writes directly to our buffer
        moe_output: Optional[torch.Tensor] = None
        if self.alltoall_method_type == AlltoallMethodType.NVLinkOneSided:
            # Retrieve a workspace-backed output tensor sized by runtime tokens
            runtime_max_tokens_per_rank = max(
                all_rank_num_tokens) if all_rank_num_tokens else x.shape[0]
            moe_output = self.moe_a2a.get_combine_payload_tensor_in_workspace(
                runtime_max_tokens_per_rank, self.unpadded_hidden_size,
                output_dtype)

        # Call extracted run_moe method
        final_hidden_states = self.run_moe(
            x=x,
            token_selected_experts=token_selected_slots,
            token_final_scales=token_final_scales,
            x_sf=x_sf,
            is_sf_swizzled=not post_quant_comm,
            output_dtype=output_dtype,
            tuner_num_tokens=tuner_num_tokens,
            tuner_top_k=tuner_top_k,
            moe_output=moe_output,
            lora_params=lora_params,
        )

        self._load_balancer_start_set_cpu_stage(is_last_call)

        # Combine results if using alltoall
        if self.enable_alltoall:
            if self.alltoall_method_type == AlltoallMethodType.NVLinkTwoSided:
                if alltoall_info is not None:
                    top_k = self.routing_method.experts_per_token
                    final_hidden_states = MnnvlMoe.mnnvl_moe_alltoallv_combine(
                        final_hidden_states,
                        alltoall_info,
                        self.alltoall_workspace,
                        ep_rank=self.ep_rank,
                        ep_size=self.ep_size,
                        top_k=top_k,
                        use_low_precision_combine=self.
                        use_low_precision_combine,
                        token_count=token_count)
            elif self.alltoall_method_type == AlltoallMethodType.NVLinkOneSided:
                output_hidden_size = final_hidden_states.shape[-1]
                runtime_max_tokens_per_rank = max(
                    all_rank_num_tokens) if all_rank_num_tokens else token_count
                final_hidden_states = self.moe_a2a.combine(
                    final_hidden_states.view(self.ep_size,
                                             runtime_max_tokens_per_rank,
                                             output_hidden_size),
                    runtime_max_tokens_per_rank,
                    payload_in_workspace=True)
            else:
                raise ValueError(
                    f"Unsupported moe alltoall method type: {self.alltoall_method_type}"
                )

        self._load_balancer_done_set_cpu_stage(is_last_call)

        return final_hidden_states

    def split_chunk(self, split_token_num: int, split_num_chunks: int):
        val_div = split_token_num // split_num_chunks
        val_mod = split_token_num % split_num_chunks
        split_chunk_size_list = [val_div + 1] * val_mod + [val_div] * (
            split_num_chunks - val_mod)
        return split_chunk_size_list

    def forward_impl(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        *,
        input_ids: Optional[torch.IntTensor] = None,
        do_finalize: bool = True,  # used by other MoE backends
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        lora_params: Optional[Dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert do_finalize, "CutlassFusedMoE does not support do_finalize=False"
        if not self._moe_lora_enabled and self._moe_lora_active(lora_params):
            # Caller passed MoE LoRA tensors but this layer was not configured
            # for it. Surface a clear error rather than silently ignoring.
            raise RuntimeError(
                "Received MoE LoRA params for a CutlassFusedMoE layer that was "
                "not configured with LoRA target modules. Ensure "
                "`lora_config.lora_target_modules` includes the desired MoE modules."
            )
        if self.use_dp and self.parallel_size > 1:
            assert all_rank_num_tokens is not None
            assert use_dp_padding is not None
            num_rows = sum(all_rank_num_tokens)
        else:
            num_rows = x.shape[0]

        if use_dp_padding:
            all_rank_num_tokens_padded = [max(all_rank_num_tokens)
                                          ] * len(all_rank_num_tokens)
            num_rows = sum(all_rank_num_tokens_padded)
        else:
            all_rank_num_tokens_padded = all_rank_num_tokens

        # in case of num_rows is larger than max_chunk_size, we need to split the input into multiple chunks
        num_chunks = (num_rows + self.moe_max_num_tokens -
                      1) // self.moe_max_num_tokens

        if num_chunks > 1 and self._moe_lora_active(lora_params):
            raise_moe_lora_multichunk_unsupported(num_chunks)

        if num_chunks == 1:
            is_first_call = self.repeat_idx == 0
            is_last_call = self.repeat_idx == self.repeat_count - 1
            outputs = self.forward_chunk(
                x,
                router_logits,
                input_ids=input_ids,
                output_dtype=output_dtype,
                all_rank_num_tokens=all_rank_num_tokens_padded,
                use_dp_padding=use_dp_padding,
                repeating_info=(is_first_call, is_last_call),
                lora_params=lora_params)
            outputs = self.reducescatter_or_allreduce(
                outputs,
                all_rank_num_tokens=all_rank_num_tokens_padded,
                use_dp_padding=use_dp_padding)
        else:
            if self.use_dp:
                all_rank_chunk_size_list = [
                    self.split_chunk(val, num_chunks)
                    for val in all_rank_num_tokens_padded
                ]
                all_rank_num_tokens_list = [[
                    val[idx_chunk] for val in all_rank_chunk_size_list
                ] for idx_chunk in range(num_chunks)]
                chunk_size_list = all_rank_chunk_size_list[self.parallel_rank]
            else:
                all_rank_num_tokens_list = [None] * num_chunks
                chunk_size_list = self.split_chunk(x.shape[0], num_chunks)

            x_list = x.split(chunk_size_list)
            router_logits_list = router_logits.split(chunk_size_list)
            input_ids_list = input_ids.split(
                chunk_size_list) if input_ids is not None else [None
                                                                ] * num_chunks

            self.event_dict[EventType.Main].record()
            with torch.cuda.stream(self.aux_stream):
                self.event_dict[EventType.Main].wait()

            def _forward_chunk(x_, router_logits_, input_ids_, idx):
                is_first_call = idx == 0 and self.repeat_idx == 0
                is_last_call = idx == num_chunks - 1 and self.repeat_idx == self.repeat_count - 1
                return self.forward_chunk(
                    x_,
                    router_logits_,
                    input_ids=input_ids_,
                    all_rank_num_tokens=all_rank_num_tokens_list[idx]
                    if self.use_dp else None,
                    use_dp_padding=use_dp_padding,
                    repeating_info=(is_first_call, is_last_call),
                    lora_params=lora_params)

            def _reducescatter_or_allreduce(x_, idx):
                return self.reducescatter_or_allreduce(
                    x_,
                    all_rank_num_tokens=all_rank_num_tokens_list[idx],
                    use_dp_padding=use_dp_padding)

            outputs_list = []
            # Postpone reduce-scatter/all-reduce to the next iteration to achieve better overlap
            for idx_chunk, (x, router_logits, input_ids) in enumerate(
                    zip(x_list, router_logits_list, input_ids_list)):
                if not (self.alltoall_method_type
                        == AlltoallMethodType.NVLinkOneSided
                        or self.alltoall_method_type
                        == AlltoallMethodType.NVLinkTwoSided):
                    if idx_chunk % 2 == 0:
                        with torch.cuda.stream(self.aux_stream):
                            outputs = _forward_chunk(x, router_logits,
                                                     input_ids, idx_chunk)
                        if idx_chunk > 0:
                            outputs_list[-1] = _reducescatter_or_allreduce(
                                outputs_list[-1], idx_chunk - 1)
                    else:
                        outputs = _forward_chunk(x, router_logits, input_ids,
                                                 idx_chunk)
                        with torch.cuda.stream(self.aux_stream):
                            outputs_list[-1] = _reducescatter_or_allreduce(
                                outputs_list[-1], idx_chunk - 1)
                else:
                    outputs = _forward_chunk(x, router_logits, input_ids,
                                             idx_chunk)

                outputs_list.append(outputs)

            if not (self.alltoall_method_type
                    == AlltoallMethodType.NVLinkOneSided
                    or self.alltoall_method_type
                    == AlltoallMethodType.NVLinkTwoSided):
                if num_chunks % 2 == 0:
                    outputs_list[-1] = _reducescatter_or_allreduce(
                        outputs_list[-1], -1)
                else:
                    with torch.cuda.stream(self.aux_stream):
                        outputs_list[-1] = _reducescatter_or_allreduce(
                            outputs_list[-1], -1)
                with torch.cuda.stream(self.aux_stream):
                    self.event_dict[EventType.MoeChunkingOverlap].record()
                self.event_dict[EventType.MoeChunkingOverlap].wait()

            outputs = torch.cat(outputs_list)

        if self.use_dp and self.parallel_size > 1:
            rank = self.parallel_rank
            outputs = outputs[:all_rank_num_tokens[rank]]
        self.repeat_idx = 0 if self.repeat_idx == self.repeat_count - 1 else self.repeat_idx + 1
        return outputs

    def forward_fake(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        *,
        do_finalize: bool = True,
        output_dtype: Optional[torch.dtype] = None,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        **kwargs,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        return super().forward_fake(
            x,
            router_logits,
            do_finalize=do_finalize,
            output_dtype=output_dtype,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=use_dp_padding,
            **kwargs,
        )

    def load_weights(self,
                     weights: List[Dict],
                     allow_partial_loading: bool = False):
        assert self._weights_created
        assert len(weights) == 1
        weights = weights[0]

        kargs = {}
        if "allow_partial_loading" in inspect.getfullargspec(
                self.quant_method.load_weights).args:
            kargs["allow_partial_loading"] = allow_partial_loading
        self.quant_method.load_weights(self, weights, self.weight_loading_mode,
                                       **kargs)
