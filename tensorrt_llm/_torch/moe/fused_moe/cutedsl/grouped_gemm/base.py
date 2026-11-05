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
"""``CuteDslFusedMoEBase``, the abstract root of the grouped-GEMM leaves."""

import torch

from .....model_config import ModelConfig
from .....utils import AuxStreamType
from ...activation import DEFAULT_MOE_ACTIVATION, MoEActivation
from ...cutlass.eligibility import BF16_ONLY, SmSupport
from ...impl_base import MoEImplBase, apply_moe_impl_construction_state
from ...quantization import MoEWeightLoadingMode
from ...routing import BaseMoeRoutingMethod
from ..locality_domain import CuteDslLocalityDomainMixin


class CuteDslFusedMoEBase(CuteDslLocalityDomainMixin, MoEImplBase):
    """Abstract root of the CuteDSL grouped-GEMM implementations.

    Holds what its leaves share: construction, and through
    :class:`CuteDslLocalityDomainMixin` the output-memset overlap stream, the
    locality-domain plan with its weight shards, and the weight lifecycle.
    Each format owns the rest -- its descriptor, eligibility, input
    quantization and ``run_moe`` -- in
    :class:`TrtllmCutedslGroupedGemmNvfp4Impl` (NVFP4),
    :class:`TrtllmCutedslGroupedGemmNvfp4PlusImpl` (NVFP4_PLUS) and
    :class:`TrtllmCutedslGroupedGemmBf16Impl` (unquantized BF16). Abstract and
    unregistered: a descriptor is exactly one identity, and this class stands
    for several.

    A leaf's format is fixed: :meth:`create_weights` refuses a layer whose
    final ``quant_config`` is not the one the leaf was resolved for.

    Args:
        routing_method (BaseMoeRoutingMethod): Routing method; its top-k is the
            number of experts each token selects.
        num_experts (int): Number of experts in the MoE layer.
        hidden_size (int): Size of the hidden state.
        intermediate_size (int): Size of the intermediate state.
        aux_stream_dict (dict[AuxStreamType, torch.cuda.Stream] | None): Auxiliary CUDA
            streams for overlapping.
        dtype (torch.dtype | None): Data type for the weights.
        reduce_results (bool): Whether to reduce the results across devices.
        model_config (ModelConfig): Configuration object for the model.
        weight_loading_mode (MoEWeightLoadingMode): How checkpoint weights
            map onto this backend's parameters.
        apply_router_weight_on_input (bool): Pre-scale the input by the
            routing weight instead of scaling the output. Requires top-1
            routing, which ``_check_configs`` asserts.
        layer_idx (int | None): Index of the layer this backend serves.
        activation (MoEActivation): The layer's activation kind and its
            constants.
        init_load_balancer (bool): Must stay False: an execution unit is
            always a ``ConfigurableMoE`` backend, never a standalone layer.
    """

    # CuteDSL dispatch/combine path exercises the ceil/floor partition
    # (NVLinkOneSided alltoall with kernel-level remainder handling), so EP
    # need not divide the expert count.
    _supports_non_divisible_ep: bool = True

    # No default: a wrong SM set silently admits a layer whose kernel does not
    # exist.
    sm_support: SmSupport
    # Both leaves run bfloat16-only kernels: NVFP4 writes a bfloat16 output,
    # and the BF16 ops read and write bfloat16.
    supported_dtypes = BF16_ONLY

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
        apply_router_weight_on_input: bool = False,
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
        self.apply_router_weight_on_input = apply_router_weight_on_input

        # Whether FC2 fuses finalize; the fused epilogue leaves no seam for a
        # LoRA GEMM.
        self.use_fused_finalize = (
            not model_config.moe_disable_finalize_fusion and model_config.lora_config is None
        )

        self._init_moe_output_memset()
        self._init_format_state()
        self._init_locality_domain(model_config)

        self._weights_created = False
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    def create_weights(self) -> None:
        if self._weights_created:
            return
        self._check_quant_config_is_my_format()
        super().create_weights()

    def _supports_load_balancer(self) -> bool:
        return True

    def _check_configs(self) -> None:
        assert self._weights_created
        if self.apply_router_weight_on_input:
            assert self.routing_method.top_k == 1, "Current walkaround only supports top-1 routing"

    def supports_moe_output_in_alltoall_workspace(self) -> bool:
        return True

    def _init_format_state(self) -> None:
        """Set up construction state only one format needs. No-op by default."""
