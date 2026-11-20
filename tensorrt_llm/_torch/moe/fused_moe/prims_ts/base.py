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
"""``PrimsTsFusedMoEBase``, the abstract root of the PrimsTS leaves."""

import torch

from ....flashinfer_utils import get_env_enable_pdl
from ....model_config import ModelConfig
from ....utils import ActivationType, AuxStreamType
from ...custom_ops.prims_ts_moe import prims_ts_moe
from ..activation import (
    DEFAULT_MOE_ACTIVATION,
    ActivationParamShape,
    MoEActivation,
    MoEActivationSupport,
    materialize_activation_params,
    resolve_activation_support,
)
from ..cutlass.eligibility import BF16_ONLY, SmSupport
from ..impl_base import MoEImplBase, apply_moe_impl_construction_state
from ..impl_contract import MoERunContext, require_comm_plan
from ..quantization import MoEWeightLoadingMode
from ..routing import BaseMoeRoutingMethod
from ..trtllm_gen.kernel_inputs import extract_routing_params, get_data_or_none


class PrimsTsFusedMoEBase(MoEImplBase):
    """Abstract root of the ``flashinfer.prims_ts.batched_gemm.*`` implementations.

    The FC1 (with its fused activation) and FC2 batched GEMMs are FlashInfer's
    PrimsTS kernels, vendored under ``tensorrt_llm/_torch/moe/kernels/prims_ts``
    and launched by ``trtllm::prims_ts_moe``. Everything around them is
    TRT-LLM's: the host top-k (``_supports_load_balancer`` forces separated
    routing), ``moe_sort`` and ``moe_unpermute_inplace``, and the weights, which
    the TRTLLM-Gen quantization methods lay out in the shuffled MajorK /
    BlockMajorK form the kernels read. A leaf supplies only its identity, its
    weight method, its input quantization and the shape alignment that keeps
    the weight method from padding the hidden size.

    Abstract and unregistered: a descriptor is exactly one identity, and this
    class stands for several.

    Args:
        routing_method (BaseMoeRoutingMethod): Routing method; its top-k is the
            number of experts each token selects.
        num_experts (int): Number of experts in the MoE layer.
        hidden_size (int): Size of the hidden state.
        intermediate_size (int): Size of the intermediate state.
        dtype (torch.dtype | None): Data type for the weights.
        reduce_results (bool): Whether to reduce the results across devices.
        model_config (ModelConfig): Configuration object for the model.
        aux_stream_dict (dict[AuxStreamType, torch.cuda.Stream] | None): Auxiliary
            CUDA streams for overlapping.
        weight_loading_mode (MoEWeightLoadingMode): How checkpoint weights map
            onto this backend's parameters.
        layer_idx (int | None): Index of the layer this backend serves.
        activation (MoEActivation): The layer's activation kind and its constants.
        init_load_balancer (bool): Must stay False: an execution unit is always a
            ``ConfigurableMoE`` backend, never a standalone layer.
    """

    sm_support = SmSupport(allowed=frozenset({100, 103, 107}))
    # The kernels write a bfloat16 output.
    supported_dtypes = BF16_ONLY

    #: Multiple the hidden size must be, so that the weight method stores FC1's
    #: input and FC2's output at the layer's own width.
    hidden_size_alignment: int = 128
    #: Multiple each rank's intermediate shard must be (whole scale groups for
    #: the quantized formats); set by each leaf.
    intermediate_shard_alignment: int

    # The fused FC1 epilogues read alpha/beta/clamp per expert. SiTU's two soft
    # caps ride in the alpha/beta slots, as for TRTLLM-Gen.
    activation_support = MoEActivationSupport(
        kinds=frozenset({ActivationType.Swiglu, ActivationType.SiTu}),
        alpha_beta=ActivationParamShape.PER_EXPERT_TENSOR,
        limit=ActivationParamShape.PER_EXPERT_TENSOR,
    )

    def __init__(
        self,
        *,
        routing_method: BaseMoeRoutingMethod,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: torch.dtype | None = None,
        reduce_results: bool = False,
        model_config: ModelConfig = ModelConfig(),
        aux_stream_dict: dict[AuxStreamType, torch.cuda.Stream] | None = None,
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.VANILLA,
        layer_idx: int | None = None,
        activation: MoEActivation = DEFAULT_MOE_ACTIVATION,
        init_load_balancer: bool = False,
    ) -> None:
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
            layer_idx=layer_idx,
            activation=activation,
            init_load_balancer=init_load_balancer,
        )
        # The autotuner profiles token counts up to this bound.
        self.max_num_tokens = model_config.max_num_tokens

        self._weights_created = False
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    def create_weights(self) -> None:
        if self._weights_created:
            return
        self._check_quant_config_is_my_format()
        super().create_weights()

    def _check_configs(self) -> None:
        """FC2 must write rows of the layer's own width, which the finalize relies on.

        Selection declines a hidden size the weight method would pad, but it
        abstains when the shapes are unknown; this catches what it let through.
        """
        w2 = self.w2_weight
        fc2_width = w2.shape[2] if w2.ndim == 4 else w2.shape[1]
        if fc2_width != self.hidden_size:
            raise ValueError(
                f"{type(self).__name__} requires FC2 weights of width hidden_size="
                f"{self.hidden_size}, but the weight layout padded it to {fc2_width}"
            )

    @property
    def is_situ_activation(self) -> bool:
        """Read by the NVFP4 weight method: SiTU's ``fc31_scale_c`` omits the FC1 dequant.

        The SiTU epilogue dequantizes both FC1 halves through the gate scale
        before its soft caps, so ``fc31_scale_c`` must not apply it again.
        """
        return self.activation_type is ActivationType.SiTu

    def cache_derived_state(self) -> None:
        """Rebuild SiTU's soft caps, which the NVFP4 weight method rescales.

        ``NVFP4TRTLLMGenFusedMoEBaseMethod`` divides ``act_beta`` by
        ``fc31_alpha``, the accumulator units SwiGLU's additive beta needs; the
        SiTU epilogue applies both caps to the dequantized halves instead.
        """
        super().cache_derived_state()
        if not self.is_situ_activation:
            return
        params = materialize_activation_params(
            self.activation,
            resolve_activation_support(self),
            num_local_experts=self.expert_size_per_partition,
            device=self.act_beta.device,
            owner=type(self).__name__,
        )
        self.act_alpha.data.copy_(params.alpha)
        self.act_beta.data.copy_(params.beta)

    def _supports_load_balancer(self) -> bool:
        """Route on the host: ``run_moe`` sorts the scheduler's top-k with ``moe_sort``.

        This is what makes the scheduler separate routing; EPLB itself is still
        declined by ``can_implement``.
        """
        return True

    def supports_moe_output_in_alltoall_workspace(self) -> bool:
        """``run_moe`` finalizes into a caller-supplied output buffer."""
        return True

    def _kernel_activation_type(self) -> int:
        """The ``flashinfer.tllm_enums.ActivationType`` of the fused FC1 epilogue."""
        from flashinfer.tllm_enums import ActivationType as KernelActivationType

        kernel_activation = {
            ActivationType.Swiglu: KernelActivationType.Swiglu,
            ActivationType.SiTu: KernelActivationType.Situ,
        }
        return int(kernel_activation[self.activation_type])

    def run_moe(self, ctx: MoERunContext, *, workspace: dict | None = None) -> torch.Tensor:
        """Run FC1, the activation and FC2 over the routed tokens; returns the finalized output."""
        del workspace  # The op allocates its own intermediates.
        plan = require_comm_plan(self, ctx)
        if not ctx.do_finalize:
            raise ValueError(f"{type(self).__name__} only produces finalized output")
        x = ctx.x
        output = plan.moe_output
        if output is None:
            output = x.new_empty((x.shape[0], self.hidden_size), dtype=torch.bfloat16)
        routing = extract_routing_params(self.routing_method)
        prims_ts_moe(
            x,
            ctx.x_sf,
            self.w3_w1_weight,
            getattr(self, "w3_w1_weight_scale", None),
            self.w2_weight,
            getattr(self, "w2_weight_scale", None),
            self.act_alpha,
            self.act_beta,
            self.act_clamp,
            get_data_or_none(self, "fc31_scale_c"),
            get_data_or_none(self, "fc31_alpha"),
            get_data_or_none(self, "fc2_alpha"),
            ctx.token_selected_experts,
            ctx.token_final_scales,
            self.num_slots,
            self.slot_start,
            self._kernel_activation_type(),
            output,
            int(self.routing_method.routing_method_type),
            routing.n_group,
            routing.topk_group,
            routing.routed_scaling_factor,
            enable_pdl=get_env_enable_pdl(),
            tune_max_num_tokens=self.max_num_tokens,
            use_dp=self.use_dp,
        )
        return output
