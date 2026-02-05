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
from torch import nn

from tensorrt_llm._mnnvl_utils import MnnvlMemory, MnnvlMoe
from tensorrt_llm._torch.distributed.moe_alltoall import MoeAlltoAll
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ...custom_ops.trtllm_gen_custom_ops import \
    fp4_block_scale_fake_output_without_finalize
from ...distributed import allgather
from ...expert_statistic import ExpertStatistic
from ...model_config import ModelConfig
from ...utils import ActivationType, AuxStreamType, Fp4QuantizedTensor
from .interface import AlltoallMethodType, MoE, MoEWeightLoadingMode

# isort: off
from .quantization import (
    DeepSeekFP8BlockScalesFusedMoEMethod, NVFP4TRTLLMGenFusedMoEBaseMethod,
    NVFP4TRTLLMGenFusedMoEMethod, W4A8MXFP4FP8TRTLLMGenFusedMoEMethod,
    W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod, W4A8NVFP4FP8TRTLLMGenFusedMoEMethod,
    W4A16MXFP4TRTLLMGenFusedMoEMethod)
# isort: on
from .routing import BaseMoeRoutingMethod, DeepSeekV3MoeRoutingMethod


class TRTLLMGenFusedMoE(MoE):
    """
    Fused Mixture of Experts (MoE) Layer with performance tuning.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of top experts to select for each input token.
        hidden_size (int): Size of the hidden state.
        intermediate_size (int): Size of the intermediate state.
        dtype (Optional[torch.dtype]): Data type for the weights.
        reduce_results (bool): Whether to reduce the results across devices.
        model_config (ModelConfig): Configuration object for the model.
        aux_stream_dict (Optional[Dict[AuxStreamType, torch.cuda.Stream]]): Auxiliary CUDA streams for overlapping.

    MoE torch custom op:
        Only support min-latency mode now (SM100 Blackwell only).
        Quant: fp8 block scales quant and nvfp4 quant and w4a16_mxfp4 quant
            FusedMoE Op: routing(topK, etc.) + scatter + gemm1 + swiglu + gemm2 + finalize MoeRoute

    FusedMoE module:
        min-latency mode:
            dynamic quant + FusedMoe Op
            equals to: dynamic quant + routing(topK, etc.) + scatter + gemm1 + swiglu + gemm2 + finalize MoeRoute

    In min-latency mode, setting `reduce_results=False` disables the AllReduce in the FusedMoE module, so any necessary AllReduce operations must be added explicitly in the model definition.
    AttentionDP should be turned off for min-latency mode.

    When we have redundant expert, we have more weight slots than `num_experts`, in that case, we separate the concepts of expert and slot.
    Expert is the concept from model's perspective while slot is the concept from model engine's perspective.
    There should be at lease `num_experts` slots in the model engine. More than that is OK, in that case, some experts may have multiple replicas.
    """

    # Supported quantization algorithms for TRTLLMGenFusedMoE
    _SUPPORTED_QUANT_ALGOS = {
        QuantAlgo.NVFP4,
        QuantAlgo.FP8_BLOCK_SCALES,
        QuantAlgo.W4A8_NVFP4_FP8,
        QuantAlgo.W4A16_MXFP4,
        QuantAlgo.W4A8_MXFP4_FP8,
        QuantAlgo.W4A8_MXFP4_MXFP8,
    }

    # Quantization algorithms that support gptoss_style
    _GPTOSS_SUPPORTED_ALGOS = {
        QuantAlgo.NVFP4,
        QuantAlgo.W4A16_MXFP4,
        QuantAlgo.W4A8_MXFP4_FP8,
        QuantAlgo.W4A8_MXFP4_MXFP8,
    }

    @classmethod
    def can_implement(
        cls,
        quant_algo: Optional[QuantAlgo],
        dtype_activation: torch.dtype = torch.bfloat16,
        gptoss_style: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if TRTLLMGenFusedMoE can implement the given quantization algorithm.

        TRTLLMGenFusedMoE only supports SM in {100, 103} and the following quantizations:
        - NVFP4
        - FP8_BLOCK_SCALES
        - W4A8_NVFP4_FP8
        - W4A16_MXFP4
        - W4A8_MXFP4_FP8
        - W4A8_MXFP4_MXFP8

        Does NOT support unquantized mode. Output dtype is hardcoded to bfloat16.

        Args:
            quant_algo: The quantization algorithm to check (None for unquantized)
            dtype_activation: The activation input data type. Only bfloat16 is supported.
                See: forward_impl() assert x.dtype == torch.bfloat16 (line 722).
            gptoss_style: Whether gptoss_style (bias/swiglu with custom alpha/beta/limit) is enabled.
                Only supported for nvfp4 and mxfp4 variants.

        Returns:
            Tuple[bool, Optional[str]]: (can_implement, skip_reason)
        """
        from .interface import _warn_and_return

        sm_version = get_sm_version()

        # TRTLLMGenFusedMoE requires SM in {100, 103}
        if sm_version not in {100, 103}:
            return _warn_and_return(
                f"TRTLLMGenFusedMoE requires SM100 or SM103, got SM{sm_version}"
            )

        # Check dtype_activation: only bfloat16 is supported
        if dtype_activation != torch.bfloat16:
            return _warn_and_return(
                f"TRTLLMGenFusedMoE only supports bfloat16 activation, got {dtype_activation}"
            )

        # TRTLLMGenFusedMoE does NOT support unquantized mode
        if quant_algo is None:
            return _warn_and_return(
                "TRTLLMGenFusedMoE does not support unquantized mode")

        # Check if quant_algo is supported
        if quant_algo not in cls._SUPPORTED_QUANT_ALGOS:
            return _warn_and_return(
                f"TRTLLMGenFusedMoE does not support quant_algo={quant_algo}")

        # Check gptoss_style support: only supported for nvfp4 and mxfp4 variants
        if gptoss_style and quant_algo not in cls._GPTOSS_SUPPORTED_ALGOS:
            return _warn_and_return(
                f"TRTLLMGenFusedMoE supports gptoss_style (bias/swiglu) only for nvfp4 and mxfp4 variants, "
                f"got quant_algo={quant_algo}")

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
        layer_idx: Optional[int] = None,
        bias: bool = False,
        swiglu_alpha: Optional[torch.Tensor] = None,
        swiglu_beta: Optional[torch.Tensor] = None,
        swiglu_limit: Optional[torch.Tensor] = None,
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
            aux_stream_dict=aux_stream_dict,
            weight_loading_mode=weight_loading_mode,
            bias=bias,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            swiglu_limit=swiglu_limit,
            layer_idx=layer_idx,
            init_load_balancer=init_load_balancer,
            activation_type=activation_type,
        )

        sm_version = get_sm_version()
        if sm_version >= 120:
            raise NotImplementedError(
                "TRTLLMGenFusedMoE does not support SM120 and above.")

        assert not self.smart_router, "Smart router is not supported in TRTLLMGenFusedMoE."

        # Note: Load balancer initialization is handled by base class _init_load_balancer()
        # If no load balancer is available, the base class will set:
        # - self.num_slots = self.num_experts
        # - self.expert_size_per_partition = self.num_experts // self.ep_size
        # - self.initial_global_assignments, self.slot_start, self.slot_end, etc.

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
                    dtype = self.dtype or torch.bfloat16

                    workspace_size = MoeAlltoAll.calculate_required_workspace_size(
                        ep_size,
                        self.routing_method.experts_per_token,
                        max_num_tokens,
                        hidden_size,
                        dtype,
                        self.num_experts if self.layer_load_balancer else None,
                    )

                    self.moe_a2a = MoeAlltoAll(
                        mapping=self.mapping,
                        max_num_tokens=model_config.max_num_tokens,
                        top_k=self.routing_method.experts_per_token,
                        num_slots=self.num_slots,
                        workspace_size_per_rank=workspace_size,
                        num_experts=self.num_experts
                        if self.layer_load_balancer else None)
                elif self.alltoall_method_type == AlltoallMethodType.DeepEP or self.alltoall_method_type == AlltoallMethodType.DeepEPLowLatency:
                    raise NotImplementedError(
                        "DeepEP and DeepEPLowLatency are not supported for TRTLLMGenFusedMoE yet"
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

        self._weights_created = False
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    def _to_trtllm_gen_activation_type(self,
                                       activation_type: ActivationType) -> int:
        if activation_type == ActivationType.Swiglu:
            return 0
        elif activation_type == ActivationType.Relu2:
            return 1
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

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

        # We found that NVLinkOneSided performs better than NCCL AllGather/ReduceScatter,
        # regardless of the relationship between EP size and topK. We favor NVLinkOneSided for now.
        # if not self.mapping.moe_ep_size > self.routing_method.experts_per_token:
        #     return AlltoallMethodType.NotEnabled
        return AlltoallMethodType.NVLinkOneSided

    def _supports_load_balancer(self) -> bool:
        """TRTLLMGenFusedMoE supports load balancer."""
        return self.use_dp and self.parallel_size > 1

    @cached_property
    def enable_alltoall(self):
        """ enable_alltoall (bool): whether to enable alltoall instead of allgather/reducescatter
        """
        return self.alltoall_method_type != AlltoallMethodType.NotEnabled

    def _check_configs(self):
        assert self.has_deepseek_fp8_block_scales \
            or self.has_nvfp4 or self.has_w4a16_mxfp4 or self.has_w4a8_nvfp4_fp8 \
            or self.has_w4a8_mxfp4_fp8 or self.has_w4a8_mxfp4_mxfp8, "TRTLLMGenFusedMoE only supports fp8_block_scaling, nvfp4, w4a16_mxfp4, w4a8_mxfp4_fp8 and w4a8_mxfp4_mxfp8 dtypes."

        if self.bias or self.swiglu_alpha is not None or self.swiglu_beta is not None or self.swiglu_limit is not None:
            assert self.has_nvfp4 or self.has_w4a16_mxfp4 or self.has_w4a8_mxfp4_fp8 or self.has_w4a8_mxfp4_mxfp8, "TRTLLMGenFusedMoE supports bias/swiglu only for nvfp4 and mxfp4 variants."

    def _get_quant_method(self):
        if self.quant_config is not None and self.quant_config.layer_quant_mode.has_any_quant(
                exclude_kv_cache=True):
            if self.quant_config.layer_quant_mode.has_fp8_block_scales():
                return DeepSeekFP8BlockScalesFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_nvfp4():
                return NVFP4TRTLLMGenFusedMoEMethod(
                ) if self.swiglu_alpha is not None or self.activation_type == ActivationType.Relu2 else NVFP4TRTLLMGenFusedMoEBaseMethod(
                )
            elif self.quant_config.layer_quant_mode.has_w4a16_mxfp4():
                return W4A16MXFP4TRTLLMGenFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_w4a8_nvfp4_fp8():
                return W4A8NVFP4FP8TRTLLMGenFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_w4a8_mxfp4_fp8():
                return W4A8MXFP4FP8TRTLLMGenFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_w4a8_mxfp4_mxfp8():
                return W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod()
            else:
                raise NotImplementedError(
                    f"Unsupported quantization method by TRTLLMGenFusedMoE: {self.quant_config.quant_mode}"
                )
        else:
            raise NotImplementedError(
                "TRTLLMGenFusedMoE doesn't support fp16/bf16/fp32 MoE.")

    def create_weights(self):
        if self._weights_created:
            return

        self.quant_method = self._get_quant_method()
        self.quant_method.create_weights(self)

        self._weights_created = True
        self._check_configs()

        if (self.has_w4a16_mxfp4 or self.has_w4a8_nvfp4_fp8
                or self.has_w4a8_mxfp4_fp8
                or self.has_w4a8_mxfp4_mxfp8) and not self.bias:
            self.w3_w1_bias = nn.Parameter(torch.zeros(
                (self.w3_w1_weight.shape[0], self.w3_w1_weight.shape[1]),
                dtype=torch.float32),
                                           requires_grad=False)
            self.register_parameter("w3_w1_bias", self.w3_w1_bias)
            self.w2_bias = nn.Parameter(torch.zeros(
                (self.w2_weight.shape[0], self.w2_weight.shape[1]),
                dtype=torch.float32),
                                        requires_grad=False)
            self.register_parameter("w2_bias", self.w2_bias)

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

    def post_load_weights(self):
        self.quant_method.post_load_weights(self)

    def quantize_input(self, x, post_quant_comm: bool = True):
        """Quantize inputs prior to post-communication (alltoall/allgather) or before MoE computation.

        Args:
            x: Input tensor to quantize
            post_quant_comm:
                If True, quantize for post-quant communication path.
                If False, quantize for non-communication path

        Returns: (x, x_sf) where x_sf is already reshaped to 2D if needed

        For quantization methods that produce scaling factors:
        - x_sf is reshaped from 1D to 2D: [num_elements] -> [batch_size, ceil_div(hidden_size, scaling_vector_size)]
        - The 2D shape is required for proper handling in alltoall/allgather operations
        - scaling_vector_size is typically the group size for block-wise quantization
        """
        x_sf = None
        if self.has_w4a8_mxfp4_fp8:
            pad_size = self.w3_w1_weight.shape[-1] * 2 - x.shape[-1]
            x = torch.nn.functional.pad(x, (0, pad_size))
            if post_quant_comm:
                x, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                    x, self.fc31_input_dequant[0])
            else:
                x, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                    x, self.fc31_input_gate_dequant[0])
        elif self.has_nvfp4:
            if isinstance(x, Fp4QuantizedTensor):
                assert not x.is_sf_swizzled, "Fp4QuantizedTensor should not be swizzled before communication"
                x_row = x.shape[0]
                x, x_sf = x.fp4_tensor, x.scaling_factor
            else:
                # Apply pre_quant_scale if it exists (for NVFP4_AWQ)
                # fc31_act_scale shape: (1, hidden_size)
                # x shape: (num_tokens, hidden_size)
                if hasattr(
                        self,
                        'fc31_act_scale') and self.fc31_act_scale is not None:
                    x = x * self.fc31_act_scale

                pad_size = self.w3_w1_weight.shape[-1] * 2 - x.shape[-1]
                if pad_size > 0:
                    x = torch.nn.functional.pad(x, (0, pad_size))

                x_row = x.shape[0]
                x, x_sf = torch.ops.trtllm.fp4_quantize(
                    x, self.fc31_input_scale, self.scaling_vector_size, False,
                    False)
        elif self.has_w4a8_mxfp4_mxfp8:
            x, x_sf = torch.ops.trtllm.mxfp8_quantize(
                x, False, alignment=self.quant_method.input_hidden_alignment)
            x_row, x_col = x.shape[0], x.shape[1]
        elif self.has_deepseek_fp8_block_scales:
            # For SM100+, fp8_quantize_1x128 returns x_sf with shape (blocked_n, num_tokens),
            # but moe_a2a_dispatch requires all payloads to have first dim = num_tokens.
            # Transpose x_sf before dispatch and transpose back after receive, but this may
            # introduce perf regression. So we don't supports post_quant_comm for fp8_block_scales.
            # TODO: Consider remove the constraint of the OneSided AlltoAll
            pass
        elif self.has_w4a16_mxfp4:
            pad_size = self.w3_w1_weight.shape[-1] * 2 - x.shape[-1]
            x = torch.nn.functional.pad(x, (0, pad_size))
        elif self.has_w4a8_nvfp4_fp8:
            x, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
                x, 1.0 / self.fc31_input_scale)
        else:
            raise ValueError(
                f"unsupported quantization mode for post communication: {self.quant_config.quant_mode}"
            )

        if x_sf is not None:
            x_sf = x_sf.view(x_row, -1)

        return x, x_sf

    def supports_moe_output_in_alltoall_workspace(self):
        return self.has_w4a8_mxfp4_mxfp8

    def run_moe(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        x_sf: Optional[torch.Tensor] = None,
        router_logits: Optional[torch.Tensor] = None,
        do_finalize: bool = True,
        moe_output: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, tuple]:
        """
        Run MoE computation with TRTLLMGen backend.

        This method encapsulates the core MoE computation logic, handling different
        quantization schemes (fp8_block_scales, nvfp4, w4a16_mxfp4, w4a8_nvfp4_fp8,
        w4a8_mxfp4_fp8, w4a8_mxfp4_mxfp8).

        Args:
            # Standard MoE interface parameters:
            x: Input hidden states (may be pre-quantized)
            token_selected_experts: Expert IDs [num_tokens, top_k]. If EPLB is enabled,
                                    this represents expert slots [num_tokens, top_k] instead.
            token_final_scales: Final scaling factors for each token
            x_sf: Input scale factors (optional, for certain quantization schemes)

            # TRTLLMGen-specific additional parameters:
            router_logits: Router logits for integrated routing in some kernels.
                          Should be None if routing has already been done (e.g., post_quant_comm).
            do_finalize: Whether to finalize the output. If False, returns intermediate
                        results (tuple) for nvfp4 and w4a8_nvfp4_fp8 schemes.
            moe_output: Pre-allocated output buffer from workspace (optional).
                       Used for mnnvlthroughput alltoall backend to avoid extra copies.

        Returns:
            If do_finalize=True: final_hidden_states tensor
            If do_finalize=False: tuple of intermediate outputs (for nvfp4 and w4a8_nvfp4_fp8)
        """
        # Extract routing parameters from routing_method
        if isinstance(self.routing_method, DeepSeekV3MoeRoutingMethod):
            top_k = self.routing_method.routing_impl.top_k
            routing_bias = self.routing_method.e_score_correction_bias
            n_group = self.routing_method.routing_impl.n_group
            topk_group = self.routing_method.routing_impl.topk_group
            routed_scaling_factor = self.routing_method.routing_impl.routed_scaling_factor
        else:
            top_k = self.routing_method.top_k
            routing_bias = None
            n_group = None
            topk_group = None
            routed_scaling_factor = None

        routing_bias = routing_bias if router_logits is not None else None

        # Ensure x_sf is 2D before flattening
        if x_sf is not None:
            assert len(
                x_sf.shape
            ) == 2, f"x_sf should be 2D tensor, got shape {x_sf.shape}"
            x_sf = x_sf.flatten()

        if self.has_deepseek_fp8_block_scales:
            assert do_finalize, "fp8_block_scale_moe_runner does not support do_finalize=False"
            # fp8_block_scale_moe_runner needs 2D shape for x_sf and only support SM100+
            if x_sf is None:
                x, x_sf = torch.ops.trtllm.fp8_quantize_1x128(x)

            final_hidden_states = torch.ops.trtllm.fp8_block_scale_moe_runner(
                router_logits,
                routing_bias,
                x,
                x_sf,
                self.w3_w1_weight,
                self.w3_w1_weight_scaling_factor,
                self.w2_weight,
                self.w2_weight_scaling_factor,
                self.num_slots,
                top_k,
                n_group,
                topk_group,
                self.intermediate_size_per_partition,
                self.slot_start,
                self.expert_size_per_partition,
                routed_scaling_factor,
                self.routing_method.routing_method_type,
                topk_weights=token_final_scales,
                topk_ids=token_selected_experts,
            )
        elif self.has_nvfp4:
            factor = 1 if self.activation_type == ActivationType.Relu2 else 2
            intermediate_size_per_partition_padded = self.w3_w1_weight.shape[
                -2] // factor
            act_type = self._to_trtllm_gen_activation_type(self.activation_type)
            outputs = torch.ops.trtllm.fp4_block_scale_moe_runner(
                router_logits,
                routing_bias,
                x,
                x_sf.view(torch.float8_e4m3fn),
                self.w3_w1_weight,
                self.w3_w1_weight_scale.view(torch.float8_e4m3fn),
                self.w3_w1_bias if self.bias else None,
                self.swiglu_alpha,
                self.swiglu_beta,
                self.swiglu_limit,
                self.w2_weight,
                self.w2_weight_scale.view(torch.float8_e4m3fn),
                self.w2_bias if self.bias else None,
                self.fc31_scale_c.data,
                self.fc31_alpha.data,
                self.fc2_alpha.data,
                self.num_slots,
                top_k,
                n_group,
                topk_group,
                intermediate_size_per_partition_padded,
                self.slot_start,
                self.expert_size_per_partition,
                routed_scaling_factor,
                self.routing_method.routing_method_type,
                do_finalize=do_finalize,
                act_type=act_type,
                topk_weights=token_final_scales,
                topk_ids=token_selected_experts,
            )

            if not do_finalize:
                assert not self.reduce_results, "reduce_results must be False when do_finalize is False"
                return outputs
            else:
                final_hidden_states = outputs[0]
                # Slice output if it was padded
                if final_hidden_states.shape[1] > self.hidden_size:
                    final_hidden_states = final_hidden_states[:, :self.
                                                              hidden_size].contiguous(
                                                              )
        elif self.has_w4a16_mxfp4:
            assert x.dtype == torch.bfloat16

            intermediate_size_per_partition_padded = self.w3_w1_weight.shape[
                -2] // 2
            final_hidden_states = torch.ops.trtllm.bf16_mxe2m1_block_scale_moe_runner(
                router_logits,
                routing_bias,
                x,
                self.w3_w1_weight,
                self.w3_w1_weight_scale,
                self.w3_w1_bias,
                self.swiglu_alpha,
                self.swiglu_beta,
                self.swiglu_limit,
                self.w2_weight,
                self.w2_weight_scale,
                self.w2_bias,
                self.num_slots,
                top_k,
                n_group,
                topk_group,
                intermediate_size_per_partition_padded,
                self.hidden_size,
                self.quant_method.intermediate_size_per_partition_lean,
                self.slot_start,
                self.expert_size_per_partition,
                routed_scaling_factor,
                self.routing_method.routing_method_type,
                0,  # act_type
                token_final_scales,
                token_selected_experts,
            )
            final_hidden_states = final_hidden_states[:, :self.
                                                      hidden_size].contiguous()
        elif self.has_w4a8_nvfp4_fp8:

            outputs = torch.ops.trtllm.fp8_fp4_block_scale_moe_runner(
                router_logits,
                routing_bias,
                x,
                self.w3_w1_weight,
                self.w3_w1_weight_scale.view(torch.float8_e4m3fn),
                self.w2_weight,
                self.w2_weight_scale.view(torch.float8_e4m3fn),
                self.fc31_scale_c.data,
                self.fc31_alpha.data,
                self.fc2_alpha.data,
                self.num_slots,
                top_k,
                n_group,
                topk_group,
                self.intermediate_size_per_partition,
                self.slot_start,
                self.expert_size_per_partition,
                routed_scaling_factor,
                self.routing_method.routing_method_type,
                do_finalize=do_finalize,
                act_type=0,
                topk_ids=token_selected_experts,
                topk_weights=token_final_scales,
            )

            if not do_finalize:
                assert not self.reduce_results, "reduce_results must be False when do_finalize is False"
                return outputs
            else:
                final_hidden_states = outputs[0]
        elif self.has_w4a8_mxfp4_fp8:

            intermediate_size_per_partition_padded = self.w3_w1_weight.shape[
                -2] // 2

            final_hidden_states = torch.ops.trtllm.e4m3_mxe2m1_block_scale_moe_runner(
                router_logits,
                routing_bias,
                x,
                self.w3_w1_weight,
                self.w3_w1_weight_scale,
                self.w3_w1_bias,
                self.swiglu_alpha,
                self.swiglu_beta,
                self.swiglu_limit,
                self.w2_weight,
                self.w2_weight_scale,
                self.w2_bias,
                self.fc31_input_dequant,
                self.fc31_input_gate_dequant,
                self.fc2_input_dequant,
                self.num_slots,
                top_k,
                n_group,
                topk_group,
                intermediate_size_per_partition_padded,
                self.hidden_size,
                self.quant_method.intermediate_size_per_partition_lean,
                self.slot_start,
                self.expert_size_per_partition,
                routed_scaling_factor,
                self.routing_method.routing_method_type,
                0,  # act_type
                token_final_scales,
                token_selected_experts,
            )
            final_hidden_states = final_hidden_states[:, :self.
                                                      hidden_size].contiguous()
        elif self.has_w4a8_mxfp4_mxfp8:

            mxfp8_x, sf = x, x_sf

            intermediate_size_per_partition_padded = self.w3_w1_weight.shape[
                -2] // 2

            result = torch.ops.trtllm.mxe4m3_mxe2m1_block_scale_moe_runner(
                router_logits,
                routing_bias,
                x,
                x_sf,
                self.w3_w1_weight,
                self.w3_w1_weight_scale,
                self.w3_w1_bias,
                self.swiglu_alpha,
                self.swiglu_beta,
                self.swiglu_limit,
                self.w2_weight,
                self.w2_weight_scale,
                self.w2_bias,
                self.num_slots,
                top_k,
                n_group,
                topk_group,
                intermediate_size_per_partition_padded,
                self.hidden_size,
                self.quant_method.intermediate_size_per_partition_lean,
                self.slot_start,
                self.expert_size_per_partition,
                routed_scaling_factor,
                self.routing_method.routing_method_type,
                0,  # act_type
                token_final_scales,
                token_selected_experts,
                output=moe_output,
            )

            # When output is provided, use it directly as the result
            # (custom op returns empty tensor to avoid PyTorch aliasing constraints)
            final_hidden_states = moe_output if moe_output is not None else result
        else:
            raise NotImplementedError(
                "TRTLLMGenFusedMoE only supports fp8_block_scaling, nvfp4, w4a16_mxfp4, w4a8_mxfp4_mxfp8 and w4a8_mxfp4_fp8 dtypes."
            )

        return final_hidden_states

    def forward_impl(
        self,
        x: Union[torch.Tensor, Fp4QuantizedTensor],
        router_logits: torch.Tensor,
        *,
        do_finalize: bool = True,
        all_rank_num_tokens: Optional[List[int]] = None,
        use_dp_padding: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        assert x.dtype == torch.bfloat16

        # Get top_k for routing (other routing parameters are extracted inside run_moe)
        if isinstance(self.routing_method, DeepSeekV3MoeRoutingMethod):
            top_k = self.routing_method.routing_impl.top_k
        else:
            top_k = self.routing_method.top_k

        run_post_quant_allgather = (self.use_dp and self.parallel_size > 1
                                    and not self.enable_alltoall)
        post_quant_comm = run_post_quant_allgather or self.enable_alltoall

        x_sf = None
        token_selected_experts = None
        token_final_scales = None
        token_count = x.shape[0]
        alltoall_info = None
        # Determine if this is first/last call (TRTLLMGenFusedMoE doesn't use chunking)
        is_first_call = self.repeat_idx == 0
        is_last_call = self.repeat_idx == self.repeat_count - 1

        if post_quant_comm:
            self._load_balancer_start_wait_gpu_stage(is_first_call)

            token_selected_experts, token_final_scales = self.routing_method.apply(
                router_logits)
            token_selected_experts = token_selected_experts.to(torch.int32)
            if token_final_scales is not None:
                token_final_scales = token_final_scales.to(torch.bfloat16)

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

            # Route tokens to slots
            token_selected_slots = self._load_balancer_route(
                token_selected_experts, self.use_dp)

            # Update expert statistics
            ExpertStatistic.set_layer(self.layer_idx)
            ExpertStatistic.maybe_add_info(self.num_slots, token_selected_slots)

            # Use routed slots for subsequent processing
            token_selected_experts = token_selected_slots

            x, x_sf = self.quantize_input(x)

        if self.enable_alltoall:
            assert all_rank_num_tokens is not None, "all_rank_num_tokens required for alltoall"

            runtime_max_tokens_per_rank = max(
                all_rank_num_tokens) if all_rank_num_tokens else token_count

            if token_final_scales is None:
                token_final_scales = torch.ones_like(token_selected_experts,
                                                     dtype=torch.float32)
            else:
                token_final_scales = token_final_scales.to(torch.float32)

            if self.alltoall_method_type == AlltoallMethodType.NVLinkTwoSided:
                assert self.alltoall_prepare_workspace is not None, "alltoall_prepare_workspace should be initialized"
                if is_last_call:
                    loadbalancer_local_statistic_info = self._load_balancer_get_local_statistic_tensor(
                    )
                else:
                    loadbalancer_local_statistic_info = None
                alltoall_info, gathered_loadbalancer_local_statistic_info = MnnvlMoe.mnnvl_moe_alltoallv_prepare_without_allgather(
                    token_selected_experts,
                    loadbalancer_local_statistic_info,
                    self.alltoall_prepare_workspace,
                    runtime_max_tokens_per_rank,
                    self.ep_rank,
                    self.ep_size,
                    self.num_experts,
                    self.num_slots,
                    top_k,
                )
                if gathered_loadbalancer_local_statistic_info is not None:
                    gathered_loadbalancer_local_statistic_info = gathered_loadbalancer_local_statistic_info.view(
                        (self.mapping.moe_ep_size, self.num_experts))
                    self._load_balancer_update_statistic_with_gathered_statistic(
                        gathered_loadbalancer_local_statistic_info)

        if self.enable_alltoall:
            if self.alltoall_method_type == AlltoallMethodType.NVLinkTwoSided:
                x, x_sf, token_selected_experts, token_final_scales = MnnvlMoe.mnnvl_moe_alltoallv(
                    [x, x_sf, token_selected_experts, token_final_scales],
                    alltoall_info,
                    self.alltoall_workspace,
                    self.ep_rank,
                    self.ep_size,
                )

                torch.ops.trtllm.memset_expert_ids(
                    token_selected_experts,
                    alltoall_info.recv_rank_count_cumsum,
                    runtime_max_tokens_per_rank,
                    top_k,
                    -1,  # Caution: TRTLLM-Gen uses -1 as invalid token expert id
                    self.ep_size,
                )

                if token_final_scales is not None:
                    token_final_scales = token_final_scales.to(torch.bfloat16)
            elif self.alltoall_method_type == AlltoallMethodType.NVLinkOneSided:
                payloads = []
                payloads.append(x)
                if x_sf is not None:
                    payloads.append(x_sf)
                    expert_id_payload_index = 2
                else:
                    expert_id_payload_index = 1
                payloads.append(token_selected_experts)
                payloads.append(token_final_scales)

                loadbalancer_local_statistic_info = None
                if self.layer_load_balancer and is_last_call:
                    loadbalancer_local_statistic_info = self._load_balancer_get_local_statistic_tensor(
                    )
                if loadbalancer_local_statistic_info is not None:
                    recv_tensors = self.moe_a2a.dispatch(
                        token_selected_experts,
                        payloads,
                        runtime_max_tokens_per_rank,
                        invalid_token_expert_id=
                        -1,  # Caution: TRTLLM-Gen uses -1 as invalid token expert id
                        expert_id_payload_index=expert_id_payload_index,
                        eplb_local_stats=loadbalancer_local_statistic_info,
                    )
                    gathered_stats = self.moe_a2a._state.eplb_gathered_stats
                    self._load_balancer_update_statistic_with_gathered_statistic(
                        gathered_stats)
                else:
                    recv_tensors = self.moe_a2a.dispatch(
                        token_selected_experts,
                        payloads,
                        runtime_max_tokens_per_rank,
                        invalid_token_expert_id=
                        -1,  # Caution: TRTLLM-Gen uses -1 as invalid token expert id
                        expert_id_payload_index=expert_id_payload_index,
                    )

                if x_sf is not None:
                    x_recv, x_sf_recv, token_selected_experts_recv, token_final_scales_recv = recv_tensors
                    x_sf = x_sf_recv.view(-1, x_sf_recv.shape[-1])
                else:
                    x_recv, token_selected_experts_recv, token_final_scales_recv = recv_tensors
                x = x_recv.view(-1, x_recv.shape[-1])
                token_selected_experts = token_selected_experts_recv.view(
                    -1, token_selected_experts_recv.shape[-1])
                token_final_scales = token_final_scales_recv.view(
                    -1, token_final_scales_recv.shape[-1])

                if token_final_scales is not None:
                    token_final_scales = token_final_scales.to(torch.bfloat16)
            else:
                raise ValueError(
                    f"Unsupported moe alltoall method type: {self.alltoall_method_type}"
                )

        elif run_post_quant_allgather:
            if x_sf is not None:
                assert len(
                    x_sf.shape
                ) == 2, "The hidden states scaling factor should be 2D tensor before allgather"
            x, x_sf, token_selected_experts, token_final_scales = allgather(
                [x, x_sf, token_selected_experts, token_final_scales],
                self.mapping,
                dim=0,
                sizes=None if use_dp_padding else all_rank_num_tokens)
        else:
            # No communication path: use non-post-quant-comm quantization
            x, x_sf = self.quantize_input(x, post_quant_comm=False)

        moe_output: Optional[torch.Tensor] = None
        use_workspace_output = False
        # TODO: use_workspace_output only supports w4a8_mxfp4_mxfp8 (gpt-oss) for now
        if self.alltoall_method_type == AlltoallMethodType.NVLinkOneSided and self.has_w4a8_mxfp4_mxfp8:
            moe_output = self.moe_a2a.get_combine_payload_tensor_in_workspace(
                runtime_max_tokens_per_rank, self.hidden_size, torch.bfloat16)
            use_workspace_output = True

        # Call the extracted run_moe interface
        # Determine router_logits based on post_quant_comm
        router_logits_arg = None if post_quant_comm else router_logits

        final_hidden_states = self.run_moe(
            x=x,
            token_selected_experts=token_selected_experts,
            token_final_scales=token_final_scales,
            x_sf=x_sf,
            # TRTLLMGenFusedMoE extra parameters
            router_logits=router_logits_arg,
            do_finalize=do_finalize,
            moe_output=moe_output,
        )

        self._load_balancer_start_set_cpu_stage(is_last_call)

        # Combine results if using alltoall
        if self.enable_alltoall:
            if self.alltoall_method_type == AlltoallMethodType.NVLinkTwoSided:
                if alltoall_info is not None:
                    final_hidden_states = MnnvlMoe.mnnvl_moe_alltoallv_combine(
                        final_hidden_states,
                        alltoall_info,
                        self.alltoall_workspace,
                        ep_rank=self.ep_rank,
                        ep_size=self.ep_size,
                        top_k=top_k,
                        use_low_precision_combine=self.
                        use_low_precision_combine,
                        token_count=token_count,
                    )
            elif self.alltoall_method_type == AlltoallMethodType.NVLinkOneSided:
                # If use_workspace_output=True, the MoE result is already in workspace
                # Otherwise, we need to reshape and pass it
                if use_workspace_output:
                    # Workspace payload is returned as 2D [ep_size * max_tokens, hidden]; reshape to 3D.
                    hidden = final_hidden_states.shape[-1]
                    payload = moe_output.view(self.ep_size,
                                              runtime_max_tokens_per_rank,
                                              hidden)
                    final_hidden_states = self.moe_a2a.combine(
                        payload,
                        runtime_max_tokens_per_rank,
                        payload_in_workspace=True)
                else:
                    hidden = final_hidden_states.shape[-1]
                    payload = final_hidden_states.view(
                        self.ep_size, runtime_max_tokens_per_rank, hidden)
                    final_hidden_states = self.moe_a2a.combine(
                        payload,
                        runtime_max_tokens_per_rank,
                        payload_in_workspace=False)
            else:
                raise ValueError(
                    f"Unsupported moe alltoall method type: {self.alltoall_method_type}"
                )

        final_hidden_states = self.reducescatter_or_allreduce(
            final_hidden_states,
            all_rank_num_tokens=all_rank_num_tokens,
            use_dp_padding=use_dp_padding,
        )

        self._load_balancer_done_set_cpu_stage(is_last_call)

        if use_dp_padding:
            rank = self.parallel_rank
            final_hidden_states = final_hidden_states[:
                                                      all_rank_num_tokens[rank]]

        # Update repeat index for load balancer
        if self.layer_load_balancer:
            self.repeat_idx = 0 if self.repeat_idx == self.repeat_count - 1 else self.repeat_idx + 1

        return final_hidden_states

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
        if do_finalize:
            # TRTLLMGenFusedMoE only supports bfloat16 output
            return super().forward_fake(x,
                                        router_logits,
                                        do_finalize=do_finalize,
                                        output_dtype=torch.bfloat16,
                                        all_rank_num_tokens=all_rank_num_tokens,
                                        use_dp_padding=use_dp_padding,
                                        **kwargs)
        else:
            is_deepseek_v3_routing = isinstance(self.routing_method,
                                                DeepSeekV3MoeRoutingMethod)
            top_k = self.routing_method.routing_impl.top_k if is_deepseek_v3_routing else self.routing_method.top_k
            routing_bias = self.routing_method.e_score_correction_bias if is_deepseek_v3_routing else None
            return fp4_block_scale_fake_output_without_finalize(
                x,
                self.num_experts,
                top_k,
                routing_bias,
            )
