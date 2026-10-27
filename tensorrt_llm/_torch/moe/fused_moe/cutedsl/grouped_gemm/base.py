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

from typing import Any

import torch

from .....locality_domain.policy import LocalityDomainExecutionPlanner
from .....locality_domain.runtime import LocalityDomainRuntime
from .....locality_domain_utils import _copy_to_new_cuda_allocation, get_reserved_remainder_stream
from .....model_config import ModelConfig
from .....utils import ActivationType, AuxStreamType, EventType
from ...activation import DEFAULT_MOE_ACTIVATION, MoEActivation
from ...cutlass.eligibility import BF16_ONLY, SmSupport
from ...impl_base import MoEImplBase, apply_moe_impl_construction_state
from ...quantization import MoEWeightLoadingMode
from ...routing import BaseMoeRoutingMethod


class CuteDslFusedMoEBase(MoEImplBase):
    """Abstract root of the CuteDSL grouped-GEMM implementations.

    Holds what both leaves share: construction, the output-memset overlap
    stream, the locality-domain plan with its weight shards, and the weight
    lifecycle. Each format owns the rest -- its descriptor, eligibility, input
    quantization and ``run_moe`` -- in
    :class:`TrtllmCutedslGroupedGemmNvfp4Impl` (NVFP4) and
    :class:`TrtllmCutedslGroupedGemmBf16Impl` (unquantized BF16). Abstract and
    unregistered: a descriptor is exactly one identity, and this class stands
    for two.

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

    #: Full-size scale parameters that ``_locality_domain_scale_shard``
    #: replaces with per-partition shards, released after the split.
    _locality_domain_full_scale_params: tuple[str, ...] = ()

    def _has_moe_output_memset_aux_stream(self) -> bool:
        event_dict = getattr(self, "event_dict", None)
        aux_stream_dict = getattr(self, "aux_stream_dict", None)
        return (
            event_dict is not None
            and aux_stream_dict is not None
            and EventType.Main in event_dict
            and EventType.MoeOutputMemset in event_dict
            and AuxStreamType.MoeOutputMemset in aux_stream_dict
        )

    def _get_reserved_moe_output_memset_stream(self) -> torch.cuda.Stream | None:
        """Resolve and cache the strict split's remainder stream before capture."""
        if hasattr(self, "_cached_reserved_moe_output_memset_stream"):
            return self._cached_reserved_moe_output_memset_stream

        stream = None
        if self._locality_domain_runtime is not None:
            stream = get_reserved_remainder_stream()
        self._cached_reserved_moe_output_memset_stream = stream
        return stream

    def _moe_output_memset_run_stream(self) -> torch.cuda.Stream:
        """Select the remainder stream when present, otherwise the aux stream."""
        remainder_stream = self._get_reserved_moe_output_memset_stream()
        if remainder_stream is not None:
            return remainder_stream
        return self.aux_stream_dict[AuxStreamType.MoeOutputMemset]

    def _locality_domain_workload_identity(self, input_dtype: torch.dtype) -> tuple[Any, ...]:
        """Describe the sharded MoE workload and its compute topology."""
        if self._locality_domain_runtime is None or self._locality_domain_weight_shards is None:
            raise RuntimeError("locality domain workload identity requires initialized shards")
        shard_identity = tuple(
            (
                tuple(shard["w3_w1_weight"].shape),
                str(shard["w3_w1_weight"].dtype),
                tuple(shard["w2_weight"].shape),
                str(shard["w2_weight"].dtype),
            )
            for shard in self._locality_domain_weight_shards
        )
        return (
            self._locality_domain_plan.num_partitions,
            self._locality_domain_runtime.topology_identity(),
            str(input_dtype),
            shard_identity,
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

        # Output-memset overlap is independent of MoE chunking, so ensure its
        # stream and events exist even if the parent creates no chunking event.
        if self.aux_stream_dict is None:
            self.aux_stream_dict = {}
        if AuxStreamType.MoeOutputMemset not in self.aux_stream_dict:
            self.aux_stream_dict[AuxStreamType.MoeOutputMemset] = torch.cuda.Stream()
        self.event_dict = {
            key: torch.cuda.Event() for key in [EventType.Main, EventType.MoeOutputMemset]
        }
        self._init_format_state()

        self._weights_created = False

        # locality domain: fork/join with _locality_domain kernel variants + shared output buffers.
        # Weight splitting happens in transform_weights after normal loading.
        self._locality_domain_runtime = None
        self._locality_domain_weight_shards = None  # set in transform_weights
        planner = LocalityDomainExecutionPlanner(model_config.locality_domain_policy)
        self._locality_domain_plan = planner.plan_moe(
            self.quant_config,
            moe_backend=model_config.moe_backend,
            use_fused_finalize=self.use_fused_finalize,
            dtype_activation=self.dtype,
            activation=ActivationType(self.activation_type).name,
        )
        if self._locality_domain_plan.enabled:
            self._locality_domain_runtime = LocalityDomainRuntime(
                self._locality_domain_plan.num_partitions
            )
        if not model_config.skip_create_weights_in_init:
            self.create_weights()

    def create_weights(self) -> None:
        if self._weights_created:
            return
        self._check_quant_config_is_my_format()
        super().create_weights()

    @property
    def uses_locality_domain(self) -> bool:
        return self._locality_domain_plan.enabled

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

    def _locality_domain_scale_shard(
        self, partition_id: int, half_n1: int, half_n2: int
    ) -> dict[str, torch.Tensor]:
        """Scale entries of one partition's weight shard; none by default.

        Runs inside the partition's weight context, so whatever it copies lands
        on that partition's localized memory. ``half_n1`` / ``half_n2`` are the
        per-partition FC1 / FC2 widths along N.
        """
        del partition_id, half_n1, half_n2
        return {}

    def load_weights(self, weights: list[dict], allow_partial_loading: bool = False) -> None:
        if self._locality_domain_weight_shards is not None:
            self.pre_reload_weights()
        super().load_weights(weights, allow_partial_loading=allow_partial_loading)
        # Keep DWDP registration after base weight loading. This preserves
        # loaded tensors for collector setup and remains compatible with the
        # later locality domain transform_weights splitting flow.
        dwdp_handle_collector = getattr(self, "dwdp_handle_collector", None)
        if dwdp_handle_collector is not None:
            dwdp_handle_collector.register_weights(self)

    def transform_weights(self) -> None:
        if getattr(self, "_weights_transformed", False):
            return
        super().transform_weights()
        # Split full weights into per-partition halves on localized memory
        if self._locality_domain_runtime is not None:
            # Refresh views of finalized scales before copying their shards.
            self.quant_method.setup_quant_scales(self)
            self._locality_domain_weight_shards = self._split_weights_for_locality_domain()
            self._release_full_weights_after_locality_domain_split()

    def cache_derived_state(self) -> None:
        super().cache_derived_state()
        if self._locality_domain_runtime is not None:
            # Staged loads (e.g. GMS read-only, MX receiver) skip transform_weights,
            # which builds the shards, and do not carry them over from the source.
            if self._locality_domain_weight_shards is None:
                raise NotImplementedError(
                    "Locality domain CuteDslFusedMoE requires transform_weights() "
                    "to build its weight shards; staged loads that skip it are "
                    "not supported"
                )
            # Weight splitting initializes the process-lifetime locality domain resource.
            # Resolve the borrowed remainder stream now, never during capture.
            self._get_reserved_moe_output_memset_stream()

    def pre_reload_weights(self) -> None:
        # The quant method rebuilds the original full-weight schemas. Drop
        # shards so the next transform uses the newly loaded checkpoint.
        super().pre_reload_weights()
        self._locality_domain_weight_shards = None
        self._weights_transformed = False

    def _release_full_weights_after_locality_domain_split(self) -> None:
        """Release full tensors that are replaced by localized locality domain shards."""
        for param_name in ("w3_w1_weight", "w2_weight", *self._locality_domain_full_scale_params):
            param = getattr(self, param_name)
            metadata = self.rebuild_tensor_metadata.get(param_name)
            meta_tensor = param.to("meta") if metadata is None else metadata["meta"]
            # Retain the reload schema without a reference to the full weight.
            self.rebuild_tensor_metadata[param_name] = {"meta": meta_tensor}
            setattr(
                self,
                param_name,
                torch.nn.Parameter(param.new_empty(0), requires_grad=False),
            )
        self.quant_method.setup_quant_scales(self)

    def _split_weights_for_locality_domain(self) -> list[dict[str, torch.Tensor]]:
        """Split full N-dimension weights into per-partition halves.

        After normal load_weights + the base transform_weights, the full weights
        are on self. Split them along dim=1 (N) and allocate halves on
        each locality domain partition's localized memory.
        """
        num_p = self._locality_domain_plan.num_partitions
        shards = []
        for pid in range(num_p):
            with self._locality_domain_runtime.partition_weight_context(pid):
                n1 = self.w3_w1_weight.size(1)
                n2 = self.w2_weight.size(1)
                half_n1 = n1 // num_p
                half_n2 = n2 // num_p
                w3_w1_slice = self.w3_w1_weight[:, pid * half_n1 : (pid + 1) * half_n1]
                w2_slice = self.w2_weight[:, pid * half_n2 : (pid + 1) * half_n2]
                shard = {
                    "w3_w1_weight": _copy_to_new_cuda_allocation(w3_w1_slice),
                    "w2_weight": _copy_to_new_cuda_allocation(w2_slice),
                }
                shard.update(self._locality_domain_scale_shard(pid, half_n1, half_n2))
                shards.append(shard)
        return shards
