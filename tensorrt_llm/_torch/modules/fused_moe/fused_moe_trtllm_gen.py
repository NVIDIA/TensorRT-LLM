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
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from torch import nn

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ...custom_ops.trtllm_gen_custom_ops import \
    fp4_block_scale_fake_output_without_finalize
from ...model_config import ModelConfig
from ...utils import (ActivationType, ActType_TrtllmGen, AuxStreamType,
                      Fp4QuantizedTensor, MxFp8QuantizedTensor)
from ..gated_mlp import GatedMLP
from .impl_base import MoEImplBase, apply_moe_impl_construction_state
from .impl_contract import (MoEDeployment, MoEEligibility, MoEInputRequirement,
                            MoEProblem, MoERejectReason, MoERunContext,
                            require_comm_plan)
from .impl_environment import MoEDep
from .interface import FORCE_SEPARATED_ROUTING, MoEWeightLoadingMode, _reject
from .moe_op_backend import MoEOpBackend, TRTLLMOpBackend, get_op_backend

# isort: off
from .quantization import (
    BF16TRTLLMGenFusedMoEMethod, DeepSeekFP8BlockScalesFusedMoEMethod,
    NVFP4TRTLLMGenFusedMoEBaseMethod, NVFP4TRTLLMGenFusedMoEMethod,
    W4A8MXFP4FP8TRTLLMGenFusedMoEMethod, W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod,
    W4A8NVFP4FP8TRTLLMGenFusedMoEMethod, W4A16MXFP4TRTLLMGenFusedMoEMethod)
# isort: on
from .routing import (BaseMoeRoutingMethod, DeepSeekV3MoeRoutingMethod,
                      DeepSeekV4MoeRoutingMethod, DefaultMoeRoutingMethod,
                      MiniMaxM2MoeRoutingMethod, MiniMaxM3MoeRoutingMethod)


@dataclass
class RoutingParams:
    top_k: int
    routing_bias: Optional[torch.Tensor]
    n_group: Optional[int]
    topk_group: Optional[int]
    routed_scaling_factor: Optional[float]


class TRTLLMGenFusedMoE(MoEImplBase):
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

    # bfloat16 routing scales are what these kernels read, and the DeepEP
    # dispatch has to mark unfilled rows before they reach them.
    input_requirement = MoEInputRequirement(
        routing_scales_dtype=torch.bfloat16,
        requires_sanitized_expert_ids=True,
        # The combine reduction runs in bf16 regardless of the model's output
        # dtype, so the NVLink one-sided payload buffer must be bf16 too.
        onesided_workspace_dtype=torch.bfloat16,
    )

    # Supported quantization algorithms for TRTLLMGenFusedMoE
    _SUPPORTED_QUANT_ALGOS = {
        QuantAlgo.NVFP4,
        QuantAlgo.FP8_BLOCK_SCALES,
        QuantAlgo.W4A8_NVFP4_FP8,
        QuantAlgo.W4A16_MXFP4,
        QuantAlgo.W4A8_MXFP4_FP8,
        QuantAlgo.W4A8_MXFP4_MXFP8,
    }

    # Quantization algorithms that support full swiglu_gptoss_style.
    # FP8_BLOCK_SCALES supports the DSV4-style uniform swiglu_limit_scalar
    # through the DeepSeek FP8 separate-activation path, but not bias or
    # swiglu_alpha/beta.
    _GPTOSS_SUPPORTED_ALGOS = {
        QuantAlgo.NVFP4,
        QuantAlgo.W4A16_MXFP4,
        QuantAlgo.W4A8_MXFP4_FP8,
        QuantAlgo.W4A8_MXFP4_MXFP8,
    }

    # Activations supported by the FlashInfer BF16 kernels: Swiglu and Relu2.
    _BF16_SUPPORTED_ACTIVATIONS = {
        ActivationType.Swiglu,
        ActivationType.Relu2,
    }

    @classmethod
    def can_implement(cls, p: MoEProblem, d: MoEDeployment) -> MoEEligibility:
        """TRTLLM-Gen kernels: SM100/SM103, bfloat16 activations.

        Quantized coverage is ``_SUPPORTED_QUANT_ALGOS``. The unquantized BF16
        path is served by a FlashInfer kernel, so it is available only where
        that wheel exposes ``trtllm_bf16_moe``.
        """
        sm_version = d.env.sm
        quant_algo = p.quant_algo

        # TRTLLMGenFusedMoE requires SM in {100, 103}
        if sm_version not in {100, 103}:
            return _reject(
                MoERejectReason.SM_UNSUPPORTED,
                f"TRTLLMGenFusedMoE requires SM100 or SM103, got SM{sm_version}"
            )

        # forward_impl asserts x.dtype == torch.bfloat16
        if p.dtype_act != torch.bfloat16:
            return _reject(
                MoERejectReason.DTYPE_UNSUPPORTED,
                f"TRTLLMGenFusedMoE only supports bfloat16 activation, got {p.dtype_act}"
            )

        if d.smart_router:
            return _reject(
                MoERejectReason.TOPOLOGY_UNSUPPORTED,
                f"TRTLLMGenFusedMoE has no smart-router path (moe_cluster_size="
                f"{d.cluster_size})")

        if quant_algo is None:
            if p.swiglu_gptoss_style:
                return _reject(
                    MoERejectReason.ACTIVATION_UNSUPPORTED,
                    "TRTLLMGenFusedMoE BF16 path does not support bias/swiglu custom parameters."
                )
            # Same set _check_configs asserts on, so the verdict and the
            # constructor agree instead of failing later at create_weights.
            if p.activation_type not in cls._BF16_SUPPORTED_ACTIVATIONS:
                supported = ", ".join(
                    sorted(activation.name
                           for activation in cls._BF16_SUPPORTED_ACTIVATIONS))
                return _reject(
                    MoERejectReason.ACTIVATION_UNSUPPORTED,
                    f"TRTLLMGenFusedMoE BF16 path only supports {supported} "
                    f"activations, got {p.activation}")
            if not d.env.has_dep(MoEDep.FLASHINFER_BF16_MOE):
                return _reject(
                    MoERejectReason.DEP_MISSING,
                    "TRTLLMGenFusedMoE unquantized BF16 path requires FlashInfer fused MoE "
                    "with trtllm_bf16_moe support.")
            # FlashInfer BF16 kernels require the per-rank intermediate size
            # to be a multiple of 128.
            if p.intermediate_size is not None:
                inter = p.intermediate_size
                if d.tp_size > 1:
                    if inter % d.tp_size != 0:
                        return _reject(
                            MoERejectReason.SHAPE_UNALIGNED,
                            "TRTLLMGenFusedMoE BF16 FlashInfer path requires "
                            f"intermediate_size ({inter}) divisible by "
                            f"moe_tp_size ({d.tp_size})")
                    inter = inter // d.tp_size
                if inter % 128 != 0:
                    return _reject(
                        MoERejectReason.SHAPE_UNALIGNED,
                        "TRTLLMGenFusedMoE BF16 FlashInfer path requires "
                        "intermediate_size_per_partition % 128 == 0; "
                        f"got {inter} "
                        f"(full intermediate_size={p.intermediate_size}, "
                        f"moe_tp_size={d.tp_size})")
            return MoEEligibility.ok()

        # Check if quant_algo is supported
        if quant_algo not in cls._SUPPORTED_QUANT_ALGOS:
            return _reject(
                MoERejectReason.QUANT_UNSUPPORTED,
                f"TRTLLMGenFusedMoE does not support quant_algo={quant_algo}")

        # swiglu_gptoss_style is only supported for nvfp4 and mxfp4 variants
        if p.swiglu_gptoss_style and quant_algo not in cls._GPTOSS_SUPPORTED_ALGOS:
            return _reject(
                MoERejectReason.ACTIVATION_UNSUPPORTED,
                f"TRTLLMGenFusedMoE supports swiglu_gptoss_style (bias/swiglu) only for nvfp4 and mxfp4 variants, "
                f"got quant_algo={quant_algo}")

        return MoEEligibility.ok()

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
        swiglu_limit_scalar: Optional[float] = None,
        init_load_balancer: bool = True,
        activation_type: ActivationType = ActivationType.Swiglu,
        trtllm_gen_activation_type: Optional[ActType_TrtllmGen] = None,
        trtllm_gen_activation_alpha: Optional[float] = None,
        trtllm_gen_activation_beta: Optional[float] = None,
    ):
        super().__init__(eplb=None)
        apply_moe_impl_construction_state(
            self,
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
            swiglu_limit_scalar=swiglu_limit_scalar,
            layer_idx=layer_idx,
            init_load_balancer=init_load_balancer,
            activation_type=activation_type,
        )

        self.trtllm_gen_activation_type = (
            ActType_TrtllmGen(trtllm_gen_activation_type)
            if trtllm_gen_activation_type is not None else None)
        self.trtllm_gen_activation_alpha = trtllm_gen_activation_alpha
        self.trtllm_gen_activation_beta = trtllm_gen_activation_beta
        self._validate_backend_local_activation()

        # Cached for autotune profile sizing (forward path passes
        # tune_max_num_tokens to the MoE op).
        self.max_num_tokens = model_config.max_num_tokens

        # Eligibility (SM / smart_router / BF16 FlashInfer dep) is owned by
        # ``can_implement``. Keep only the provider selection for the op.
        self.use_flashinfer = self._check_flashinfer_backend_support()
        backend_name = "flashinfer" if self.use_flashinfer else "trtllm"
        self.op_backend: MoEOpBackend = get_op_backend(backend_name)

        self._weights_created = False
        self.num_fused_shared_expert = 0

        # Fusing the shared experts into the routed-expert grouped GEMM is opt-in:
        # set TLLM_MOE_ENABLE_SHARED_EXPERT_FUSION=1 to enable it. The benefit is
        # workload-dependent (small decode batches gain, large prefill chunks lose the
        # aux-stream overlap of the unfused path), and the fused path additionally
        # restricts tactics to tileN>=32 to avoid a small-tile dynB kernel defect.
        fusion_enabled = os.environ.get("TLLM_MOE_ENABLE_SHARED_EXPERT_FUSION",
                                        "0") == "1"
        # Only the trtllm op backend implements fused shared experts
        on_trtllm_backend = isinstance(self.op_backend, TRTLLMOpBackend)
        # Expert parallelism (moe_ep_size > 1) is not supported by the fused path yet
        # (the routing kernel's shared-expert append assumes the full expert set is
        # local); gate it out here so EP configs fall back to the unfused path instead
        # of tripping the runtime EP check in the TRTLLM-Gen runner.
        fusion_supported = (
            fusion_enabled and on_trtllm_backend
            and model_config.mapping.dp_size == 1
            and model_config.mapping.moe_ep_size == 1
            and self.quant_config is not None
            and self.quant_config.layer_quant_mode.has_fp8_block_scales())
        if fusion_supported:
            # Not all models that use this backend define shared experts (e.g. non-DeepSeek
            # MoEs), so fall back to 0 when the config has no `n_shared_experts`.
            self.num_fused_shared_expert = getattr(
                model_config.pretrained_config, "n_shared_experts", 0) or 0
            if self.num_fused_shared_expert > 0:
                logger.info_once(
                    f"Shared-expert fusion enabled: folding "
                    f"{self.num_fused_shared_expert} shared expert(s) into the "
                    f"routed-expert grouped GEMM.",
                    key="trtllm_gen_shared_expert_fusion")

        # create_weights must see the final fused-expert count so the fused shared
        # slots are allocated when fusion is enabled.
        if not model_config.skip_create_weights_in_init:
            self.create_weights()
        self.layer_idx = layer_idx

    def _to_trtllm_gen_activation_type(self,
                                       activation_type: ActivationType) -> int:
        if self.trtllm_gen_activation_type is not None:
            return int(self.trtllm_gen_activation_type)
        if activation_type == ActivationType.Swiglu:
            return 0
        elif activation_type == ActivationType.SwigluBias:
            # SwigluBias uses the same SwiGlu kernel path (ActType::SwiGlu == 0);
            # the per-expert alpha/beta/clamp_limit are passed as separate tensors.
            return 0
        elif activation_type == ActivationType.Relu2:
            return 1
        elif activation_type == ActivationType.Silu:
            return 2
        else:
            raise ValueError(f"Unsupported activation type: {activation_type}")

    @property
    def is_situ_activation(self) -> bool:
        return self.trtllm_gen_activation_type == ActType_TrtllmGen.SiTu

    def _validate_backend_local_activation(self) -> None:
        # Runs from __init__, before create_weights, so the swiglu_* attributes
        # checked below are still the constructor-provided values. For SiTu,
        # create_weights later reuses the swiglu_alpha/swiglu_beta storage for
        # the backend-local activation parameters (SiTu and SwiGLU are mutually
        # exclusive and feed the same gemm1_alpha/gemm1_beta op slots).
        if self.trtllm_gen_activation_type is None:
            if (self.trtllm_gen_activation_alpha is not None
                    or self.trtllm_gen_activation_beta is not None):
                raise ValueError(
                    "TRTLLM-Gen backend-local activation alpha/beta require "
                    "trtllm_gen_activation_type.")
            return

        if not self.is_situ_activation:
            raise ValueError(
                "Only the SiTu TRTLLM-Gen backend-local activation is "
                f"supported, got {self.trtllm_gen_activation_type.name}.")
        if self.dtype != torch.bfloat16:
            raise ValueError(
                "TRTLLM-Gen SiTu requires bfloat16 activations, got "
                f"{self.dtype}.")
        if get_sm_version() not in {100, 103}:
            raise ValueError("TRTLLM-Gen SiTu requires SM100 or SM103, got "
                             f"SM{get_sm_version()}.")
        if (self.quant_config is None
                or self.quant_config.quant_algo != QuantAlgo.W4A8_MXFP4_MXFP8):
            quant_algo = (None if self.quant_config is None else
                          self.quant_config.quant_algo)
            raise ValueError(
                "TRTLLM-Gen SiTu requires W4A8_MXFP4_MXFP8 quantization, "
                f"got {quant_algo}.")
        if self.tp_size > 1:
            # Intra-expert MoE TP: w1/w3 column-shard and w2 row-shard along
            # the intermediate dim (the stock MXFP4 quant-method loaders slice
            # the group-32 packed bytes and scales per rank). Require the
            # per-rank shard to stay a whole multiple of the quant method's
            # weight alignment so per-shard scale groups and the padded
            # weight buffers line up without fractional groups.
            alignment = W4A8MXFP4MXFP8TRTLLMGenFusedMoEMethod.weight_alignment
            if (self.intermediate_size % self.tp_size != 0
                    or self.intermediate_size_per_partition % alignment != 0):
                raise ValueError(
                    "TRTLLM-Gen SiTu MoE TP requires intermediate_size "
                    f"({self.intermediate_size}) divisible by moe_tp_size "
                    f"({self.tp_size}) with the per-rank shard a multiple of "
                    f"{alignment}, got "
                    f"{self.intermediate_size_per_partition}.")
        if self.activation_type != ActivationType.Swiglu:
            raise ValueError(
                "TRTLLM-Gen SiTu must use generic SwiGLU geometry so FC1 "
                "contains gate and up projections.")
        if self.bias or any(
                value is not None
                for value in (self.swiglu_alpha, self.swiglu_beta,
                              self.swiglu_limit, self.swiglu_limit_scalar)):
            raise ValueError(
                "TRTLLM-Gen SiTu does not support bias or SwiGLU-specific "
                "alpha/beta/limit parameters.")
        if (self.trtllm_gen_activation_alpha is None
                or self.trtllm_gen_activation_beta is None):
            raise ValueError(
                "TRTLLM-Gen SiTu requires both backend-local activation "
                "alpha and beta.")
        if (self.trtllm_gen_activation_alpha <= 0.0
                or self.trtllm_gen_activation_beta <= 0.0):
            raise ValueError(
                "TRTLLM-Gen SiTu activation alpha/beta must be positive, got "
                f"{self.trtllm_gen_activation_alpha} and "
                f"{self.trtllm_gen_activation_beta}.")

    @staticmethod
    def _is_flashinfer_fused_moe_available() -> bool:
        try:
            from flashinfer.fused_moe import core as _core
        except (ImportError, ModuleNotFoundError):
            return False
        return (hasattr(_core, "trtllm_bf16_moe")
                and hasattr(_core, "trtllm_bf16_routed_moe"))

    def _is_unquantized_path(self) -> bool:
        return self.quant_config is None or not self.quant_config.layer_quant_mode.has_any_quant(
            exclude_kv_cache=True)

    def _requires_separated_routing(self) -> bool:
        """BF16 FlashInfer uses separated routing, except DeepSeekV3 which uses
        the fused kernel (its separated variant has accuracy issues)."""
        if not (self.use_flashinfer and self._is_unquantized_path()):
            return False
        return not isinstance(self.routing_method, DeepSeekV3MoeRoutingMethod)

    def _check_flashinfer_backend_support(self) -> bool:
        # SiTu is provided by the native TRTLLM-Gen cubin and is not part of
        # FlashInfer's activation enum.
        if self.is_situ_activation:
            return False

        # For BF16 (unquantized) path, we will use FlashInfer regardless whether
        # env TRTLLM_GEN_FUSED_MOE_USE_FLASHINFER=1 is set or not as it's the only way.
        if self._is_unquantized_path():
            if not self._is_flashinfer_fused_moe_available():
                return False
            if self.activation_type not in self._BF16_SUPPORTED_ACTIVATIONS:
                return False
            return True

        use_flashinfer = os.environ.get("TRTLLM_GEN_FUSED_MOE_USE_FLASHINFER",
                                        "0")
        if use_flashinfer != "1":
            return False

        # Unsupported activation type or routing method
        if self.activation_type == ActivationType.Relu2:
            return False
        if isinstance(self.routing_method,
                      (DeepSeekV3MoeRoutingMethod, DefaultMoeRoutingMethod)):
            return False

        quant_method = self._get_quant_method()

        # NVFP4 base method is always supported
        if type(quant_method) is NVFP4TRTLLMGenFusedMoEBaseMethod:
            return True

        mode = self.quant_config.layer_quant_mode

        # These quant modes are never supported via op backend
        if mode.has_w4a8_nvfp4_fp8() or mode.has_w4a8_mxfp4_fp8():
            return False

        # These quant modes require alignment and no bias
        if mode.has_nvfp4() or mode.has_w4a16_mxfp4(
        ) or mode.has_w4a8_mxfp4_mxfp8():
            if self.bias:
                return False
            if self.intermediate_size_per_partition % quant_method.weight_alignment != 0:
                return False
            if self.hidden_size % quant_method.input_hidden_alignment != 0:
                return False

        return True

    def _get_data_or_none(self, attr_name: str) -> Optional[torch.Tensor]:
        attr = getattr(self, attr_name, None)
        return attr.data if attr is not None else None

    def _supports_load_balancer(self) -> bool:
        """Whether separated routing (top-k outside the kernel) is used.

        ConfigurableMoE uses this flag to decide whether routing is separated
        (top-k ids/scales computed outside backend) or fused inside the kernel.
        BF16 FlashInfer path always requires separated routing.
        """
        if self._requires_separated_routing():
            return True
        return self.use_dp and self.parallel_size > 1

    def _routes_outside_the_kernel(self) -> bool:
        """Whether top-k is precomputed, so the kernel must not route again.

        Three independent triggers, none of which subsumes the others: a
        kernel or parallel layout that forces it (both folded into
        ``_supports_load_balancer``), a routing algorithm no C++ kernel
        implements, and the host-routing override.
        """
        return (self._supports_load_balancer()
                or self.routing_method.requires_separated_routing
                or FORCE_SEPARATED_ROUTING)

    def _check_configs(self):
        assert not self.has_any_quant \
            or self.has_deepseek_fp8_block_scales \
            or self.has_nvfp4 or self.has_w4a16_mxfp4 or self.has_w4a8_nvfp4_fp8 \
            or self.has_w4a8_mxfp4_fp8 or self.has_w4a8_mxfp4_mxfp8, \
            "TRTLLMGenFusedMoE only supports bf16 (FlashInfer), fp8_block_scaling, nvfp4, w4a16_mxfp4, w4a8_mxfp4_fp8 and w4a8_mxfp4_mxfp8 dtypes."

        if not self.has_any_quant:
            assert self.activation_type in self._BF16_SUPPORTED_ACTIVATIONS, \
                ("TRTLLMGenFusedMoE BF16 path only supports "
                 f"{[a.name for a in self._BF16_SUPPORTED_ACTIVATIONS]} activations, "
                 f"got {self.activation_type.name}.")
            assert not self.bias and self.swiglu_alpha is None and self.swiglu_beta is None and self.swiglu_limit is None, \
                "TRTLLMGenFusedMoE BF16 path does not support bias/swiglu custom parameters."

        if self.bias or self.swiglu_alpha is not None or self.swiglu_beta is not None:
            assert self.has_nvfp4 or self.has_w4a16_mxfp4 or self.has_w4a8_mxfp4_fp8 or self.has_w4a8_mxfp4_mxfp8, \
                "TRTLLMGenFusedMoE supports bias/swiglu_alpha/swiglu_beta only for nvfp4 and mxfp4 variants."
        if self.swiglu_limit is not None or self.swiglu_limit_scalar is not None:
            # swiglu_limit additionally goes through the DeepSeek FP8
            # separate-activation path
            # (DevKernel.cu::activationDeepSeekKernel) when
            # has_deepseek_fp8_block_scales. The FP8 path consumes the scalar
            # variant (uniform across experts); NVFP4/MXFP4 fused-activation
            # cubins consume the per-expert tensor.
            assert self.has_nvfp4 or self.has_w4a16_mxfp4 or self.has_w4a8_mxfp4_fp8 \
                or self.has_w4a8_mxfp4_mxfp8 or self.has_deepseek_fp8_block_scales, \
                "TRTLLMGenFusedMoE supports swiglu_limit only for nvfp4, mxfp4, and fp8_block_scale variants."
            # The FP8 block-scale separate-activation kernel only consumes the
            # uniform scalar (swiglu_limit_scalar); a per-expert swiglu_limit
            # tensor would be silently ignored, so reject it explicitly.
            if self.has_deepseek_fp8_block_scales:
                assert self.swiglu_limit is None, \
                    "TRTLLMGenFusedMoE FP8 block-scale path only supports the uniform " \
                    "swiglu_limit_scalar, not a per-expert swiglu_limit tensor."

        if self.is_situ_activation:
            if not isinstance(self.op_backend, TRTLLMOpBackend):
                raise ValueError(
                    "TRTLLM-Gen SiTu requires the native TRTLLM op backend.")
            if not self.has_w4a8_mxfp4_mxfp8:
                raise ValueError(
                    "TRTLLM-Gen SiTu requires the W4A8_MXFP4_MXFP8 path.")
            if self.scaling_vector_size != 32:
                raise ValueError(
                    "TRTLLM-Gen SiTu requires MXFP4 scaling vector size 32, "
                    f"got {self.scaling_vector_size}.")
            # For SiTu these hold the backend-local activation parameters
            # (populated by create_weights, which runs before this check).
            for name in ("swiglu_alpha", "swiglu_beta"):
                value = getattr(self, name)
                if (value.dtype != torch.float32
                        or value.shape != (self.expert_size_per_partition, )
                        or not value.is_contiguous()):
                    raise ValueError(
                        f"{name} must be a contiguous float32 tensor with "
                        "one value per local expert/slot.")

    def _get_quant_method(self):
        if self.quant_config is not None and self.quant_config.layer_quant_mode.has_any_quant(
                exclude_kv_cache=True):
            if self.quant_config.layer_quant_mode.has_fp8_block_scales():
                return DeepSeekFP8BlockScalesFusedMoEMethod()
            elif self.quant_config.layer_quant_mode.has_nvfp4():
                return NVFP4TRTLLMGenFusedMoEMethod(
                ) if self.swiglu_alpha is not None or self.activation_type in [
                    ActivationType.Relu2, ActivationType.Silu
                ] else NVFP4TRTLLMGenFusedMoEBaseMethod()
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
            return BF16TRTLLMGenFusedMoEMethod()

    def create_weights(self):
        if self._weights_created:
            return

        self.quant_method = self._get_quant_method()
        if self.quant_config is not None and self.quant_config.layer_quant_mode.has_fp8_block_scales(
        ):
            self.quant_method.create_weights(self, self.num_fused_shared_expert)
        else:
            self.quant_method.create_weights(self)

        # SiTu reuses the swiglu_alpha/swiglu_beta storage: SiTu and SwiGLU are
        # mutually exclusive (constructor-provided SwiGLU parameters are
        # rejected by _validate_backend_local_activation) and feed the same
        # gemm1_alpha/gemm1_beta op slots. Safe with respect to the
        # `swiglu_alpha is not None` gates: create_moe.py checks the
        # constructor kwargs (None for SiTu); _get_quant_method consults
        # swiglu_alpha only on the nvfp4 branch (SiTu requires
        # W4A8_MXFP4_MXFP8) and has already run above; _check_configs runs
        # after this point and its swiglu gate admits w4a8_mxfp4_mxfp8.
        if self.is_situ_activation:
            self.swiglu_alpha = nn.Parameter(torch.full(
                (self.expert_size_per_partition, ),
                float(self.trtllm_gen_activation_alpha),
                dtype=torch.float32),
                                             requires_grad=False)
            self.swiglu_beta = nn.Parameter(torch.full(
                (self.expert_size_per_partition, ),
                float(self.trtllm_gen_activation_beta),
                dtype=torch.float32),
                                            requires_grad=False)

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

    def cache_derived_state(self) -> None:
        super().cache_derived_state()
        if self.is_situ_activation:
            # Reinitialize constants after meta-device materialization. These
            # are backend configuration, not checkpoint weights.
            self.swiglu_alpha.data.fill_(float(
                self.trtllm_gen_activation_alpha))
            self.swiglu_beta.data.fill_(float(self.trtllm_gen_activation_beta))

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

    def try_fused_kimi_route_quant(
        self,
        x: Union[torch.Tensor, MxFp8QuantizedTensor],
        router_logits: torch.Tensor,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                        torch.Tensor]]:
        """Fuse Kimi K3 no-aux routing and MXFP8 input quantization.

        This launch-overhead optimization is deliberately specialized to the
        K3 decode shape. Returning ``None`` keeps every other model, shape,
        architecture, and op backend on the existing unfused path.
        """
        if (os.environ.get("TLLM_K3_DISABLE_FUSED_ROUTE_QUANT", "0") == "1"
                or isinstance(x, MxFp8QuantizedTensor)):
            return None

        sm_version = get_sm_version()
        if (not 100 <= sm_version < 110 or not self.has_w4a8_mxfp4_mxfp8
                or not isinstance(self.op_backend, TRTLLMOpBackend)
                or not isinstance(self.routing_method,
                                  DeepSeekV3MoeRoutingMethod)):
            return None

        routing = self.routing_method.routing_impl
        bias = self.routing_method.e_score_correction_bias
        if (not routing.is_fused or routing.n_group != 1
                or routing.topk_group != 1 or routing.top_k != 16
                or router_logits.ndim != 2 or router_logits.shape[1] != 896
                or router_logits.dtype != torch.float32
                or not router_logits.is_contiguous()
                or bias.dtype != torch.float32 or not bias.is_contiguous()
                or x.ndim != 2 or x.shape != (router_logits.shape[0], 3584)
                or not 0 < x.shape[0] <= 64 or x.dtype != torch.bfloat16
                or not x.is_contiguous()):
            return None

        return torch.ops.trtllm.kimi_k3_noaux_tc_mxfp8_quant(
            router_logits, bias, x, routing.routed_scaling_factor)

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
        if not self.has_any_quant:
            return x, x_sf
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
            elif isinstance(x, MxFp8QuantizedTensor):
                assert not x.is_sf_swizzled, "MxFp8QuantizedTensor should not be swizzled before communication"
                x_row = x.shape[0]
                x, x_sf = x.fp8_tensor, x.scaling_factor
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
                x, x_sf = self.op_backend.fp4_quantize(x, self.fc31_input_scale,
                                                       self.scaling_vector_size,
                                                       False, False)
        elif self.has_w4a8_mxfp4_mxfp8:
            x, x_sf = self.op_backend.mxfp8_quantize(
                x, False, alignment=self.quant_method.input_hidden_alignment)
            x_row = x.shape[0]
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
        return self.has_any_quant and not self.use_flashinfer

    def _extract_routing_params(self) -> RoutingParams:
        if isinstance(self.routing_method, DeepSeekV3MoeRoutingMethod):
            return RoutingParams(
                top_k=self.routing_method.routing_impl.top_k,
                routing_bias=self.routing_method.e_score_correction_bias,
                n_group=self.routing_method.routing_impl.n_group,
                topk_group=self.routing_method.routing_impl.topk_group,
                routed_scaling_factor=self.routing_method.routing_impl.
                routed_scaling_factor,
            )
        elif isinstance(self.routing_method, MiniMaxM3MoeRoutingMethod):
            return RoutingParams(
                top_k=self.routing_method.top_k,
                routing_bias=self.routing_method.e_score_correction_bias,
                n_group=None,
                topk_group=None,
                routed_scaling_factor=self.routing_method.routed_scaling_factor,
            )
        elif isinstance(self.routing_method, MiniMaxM2MoeRoutingMethod):
            return RoutingParams(
                top_k=self.routing_method.top_k,
                routing_bias=self.routing_method.e_score_correction_bias,
                n_group=None,
                topk_group=None,
                routed_scaling_factor=None,
            )
        elif isinstance(self.routing_method, DeepSeekV4MoeRoutingMethod):
            return RoutingParams(
                top_k=self.routing_method.top_k,
                routing_bias=self.routing_method.e_score_correction_bias,
                n_group=self.routing_method.n_group,
                topk_group=self.routing_method.topk_group,
                routed_scaling_factor=self.routing_method.routed_scaling_factor,
            )
        else:
            return RoutingParams(
                top_k=self.routing_method.top_k,
                routing_bias=None,
                n_group=None,
                topk_group=None,
                routed_scaling_factor=None,
            )

    def fuse_shared_expert(self, shared_experts: GatedMLP):
        assert self._weights_created
        self.quant_method.fuse_shared_expert(self, shared_experts,
                                             self.num_fused_shared_expert)

    def run_moe(
        self,
        ctx: MoERunContext,
        *,
        workspace: Optional[dict] = None,
    ) -> Union[torch.Tensor, tuple]:
        """
        Run MoE computation with TRTLLMGen backend.

        This method encapsulates the core MoE computation logic, handling different
        quantization schemes (bf16, fp8_block_scales, nvfp4, w4a16_mxfp4,
        w4a8_nvfp4_fp8, w4a8_mxfp4_fp8, w4a8_mxfp4_mxfp8).

        Returns:
            If ``ctx.do_finalize``: final_hidden_states tensor
            Otherwise: tuple of intermediate outputs (for nvfp4 and w4a8_nvfp4_fp8)
        """
        del workspace  # TRTLLMGen kernels allocate their own intermediates.
        plan = require_comm_plan(self, ctx)
        x = ctx.x
        token_selected_experts = ctx.token_selected_experts
        token_final_scales = ctx.token_final_scales
        x_sf = ctx.x_sf
        do_finalize = ctx.do_finalize
        moe_output = plan.moe_output
        # The caller used to apply this filter before handing over the kwargs.
        if self._routes_outside_the_kernel():
            if ctx.router_logits is not None and token_selected_experts is None:
                raise ValueError(
                    f"{type(self).__name__} requires separated routing for this "
                    "config, so ctx.router_logits is ignored, but "
                    "ctx.token_selected_experts is None -- there is nothing left "
                    "to route with. Supply precomputed top-k ids and scales.")
            router_logits = None
        else:
            router_logits = ctx.router_logits

        routing_params = self._extract_routing_params()
        top_k = routing_params.top_k
        routing_bias = routing_params.routing_bias if router_logits is not None else None
        n_group = routing_params.n_group
        topk_group = routing_params.topk_group
        routed_scaling_factor = routing_params.routed_scaling_factor

        if token_selected_experts is not None:
            # for cases like deepep low latency where fake top_k=1 might be used
            top_k = token_selected_experts.shape[-1]

        # Ensure x_sf is 2D before flattening
        if x_sf is not None:
            assert len(
                x_sf.shape
            ) == 2, f"x_sf should be 2D tensor, got shape {x_sf.shape}"
            x_sf = x_sf.flatten()

        if not self.has_any_quant:
            result = self.op_backend.run_bf16_moe(
                router_logits=router_logits,
                routing_bias=routing_bias,
                hidden_states=x,
                gemm1_weights=self.w3_w1_weight,
                gemm2_weights=self.w2_weight,
                num_experts=self.num_slots,
                top_k=top_k,
                n_group=n_group,
                topk_group=topk_group,
                intermediate_size=self.intermediate_size_per_partition,
                local_expert_offset=self.slot_start,
                local_num_experts=self.expert_size_per_partition,
                routed_scaling_factor=routed_scaling_factor,
                routing_method_type=self.routing_method.routing_method_type,
                topk_weights=token_final_scales,
                topk_ids=token_selected_experts,
                gated_act_type=self._to_trtllm_gen_activation_type(
                    self.activation_type),
                output=moe_output,
                use_shuffled_weight=getattr(self.quant_method,
                                            "use_shuffled_weight", False),
                weight_layout=getattr(self.quant_method, "weight_layout", 0),
                do_finalize=do_finalize,
            )
            if not do_finalize:
                assert not self.reduce_results, "reduce_results must be False when do_finalize is False"
                return result
            final_hidden_states = result
        elif self.has_deepseek_fp8_block_scales:
            assert do_finalize, "fp8_block_scale_moe_runner does not support do_finalize=False"
            # fp8_quantize_1x128 returns 2D x_sf on SM100+, 1D on SM90
            if x_sf is None:
                x, x_sf = torch.ops.trtllm.fp8_quantize_1x128(x)
            result = self.op_backend.run_fp8_block_scale_moe(
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
                self.num_fused_shared_expert,
                n_group,
                topk_group,
                self.intermediate_size_per_partition,
                self.slot_start,
                self.expert_size_per_partition,
                routed_scaling_factor,
                self.routing_method.routing_method_type,
                topk_weights=token_final_scales,
                topk_ids=token_selected_experts,
                gemm1_clamp_limit=self.swiglu_limit_scalar,
                output=moe_output,
                tune_max_num_tokens=self.max_num_tokens,
                use_dp=self.use_dp,
            )
            # When output is provided, use it directly as the result
            final_hidden_states = moe_output if moe_output is not None else result
        elif self.has_nvfp4 or self.has_w4a16_mxfp4 or self.has_w4a8_mxfp4_mxfp8:
            act_type = self._to_trtllm_gen_activation_type(self.activation_type)
            factor = 1 if act_type in [
                ActType_TrtllmGen.Relu2, ActType_TrtllmGen.Silu
            ] else 2
            intermediate_size_per_partition_padded = self.w3_w1_weight.shape[
                -2] // factor
            # Holds SwiGLU's per-expert alpha/beta, or SiTu's backend-local
            # activation parameters (which reuse this storage; see
            # create_weights).
            gemm1_alpha, gemm1_beta = self.swiglu_alpha, self.swiglu_beta

            output1_scale_scalar = self._get_data_or_none("fc31_scale_c")
            output1_scale_gate_scalar = self._get_data_or_none("fc31_alpha")
            output2_scale_scalar = self._get_data_or_none("fc2_alpha")

            outputs = self.op_backend.run_fp4_block_scale_moe(
                router_logits,
                routing_bias,
                x,
                x_sf,
                self.w3_w1_weight,
                self.w3_w1_weight_scale,
                self.w3_w1_bias if self.bias else None,
                gemm1_alpha,
                gemm1_beta,
                self.swiglu_limit,
                self.w2_weight,
                self.w2_weight_scale,
                self.w2_bias if self.bias else None,
                output1_scale_scalar,
                output1_scale_gate_scalar,
                output2_scale_scalar,
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
                topk_weights=token_final_scales,
                topk_ids=token_selected_experts,
                valid_hidden_size=self.hidden_size,
                valid_intermediate_size=getattr(
                    self.quant_method, 'intermediate_size_per_partition_lean',
                    None),
                gated_act_type=act_type,
                output=moe_output,
                # Pass that to the autotuner so the top bucket profiles per-expert load at runtime scale.
                tune_max_num_tokens=self.max_num_tokens,
                use_dp=self.use_dp,
            )

            if not do_finalize:
                assert not self.reduce_results, "reduce_results must be False when do_finalize is False"
                return outputs
            else:
                # When output is provided, use it directly as the result
                final_hidden_states = moe_output if moe_output is not None else outputs
                # Slice output if it was padded (only needed when moe_output is not provided)
                if moe_output is None and final_hidden_states.shape[
                        1] > self.hidden_size:
                    final_hidden_states = final_hidden_states[:, :self.
                                                              hidden_size].contiguous(
                                                              )
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
                topk_weights=token_final_scales,
                topk_ids=token_selected_experts,
                output=moe_output,
                tune_max_num_tokens=self.max_num_tokens,
                use_dp=self.use_dp,
            )

            if not do_finalize:
                assert not self.reduce_results, "reduce_results must be False when do_finalize is False"
                return outputs
            else:
                # When output is provided, use it directly as the result
                final_hidden_states = moe_output if moe_output is not None else outputs[
                    0]
        elif self.has_w4a8_mxfp4_fp8:

            intermediate_size_per_partition_padded = self.w3_w1_weight.shape[
                -2] // 2

            result = torch.ops.trtllm.e4m3_mxe2m1_block_scale_moe_runner(
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
                output=moe_output,
                tune_max_num_tokens=self.max_num_tokens,
                use_dp=self.use_dp,
            )
            # When output is provided, use it directly as the result
            final_hidden_states = moe_output if moe_output is not None else result
            if moe_output is None:
                final_hidden_states = final_hidden_states[:, :self.
                                                          hidden_size].contiguous(
                                                          )
        else:
            raise NotImplementedError(
                "TRTLLMGenFusedMoE only supports fp8_block_scaling, nvfp4, w4a16_mxfp4, w4a8_mxfp4_mxfp8 and w4a8_mxfp4_fp8 dtypes."
            )

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
            is_minimax_routing = isinstance(self.routing_method,
                                            MiniMaxM2MoeRoutingMethod)
            top_k = self.routing_method.routing_impl.top_k if is_deepseek_v3_routing else self.routing_method.top_k
            routing_bias = self.routing_method.e_score_correction_bias if (
                is_deepseek_v3_routing or is_minimax_routing) else None
            return fp4_block_scale_fake_output_without_finalize(
                x,
                self.num_experts,
                top_k,
                routing_bias,
            )
