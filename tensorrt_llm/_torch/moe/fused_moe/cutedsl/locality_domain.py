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
"""The output-memset stream and locality-domain lifecycle CuteDSL leaves share.

A mixin rather than a kernel root, so a leaf of another kernel can take it
without inheriting from the grouped-GEMM root.
"""

from typing import Any

import torch

from tensorrt_llm.logger import logger

from ....locality_domain.policy import LocalityDomainExecutionPlanner, PartitionPlan
from ....locality_domain.runtime import LocalityDomainRuntime
from ....locality_domain_utils import _copy_to_new_cuda_allocation, get_reserved_remainder_stream
from ....model_config import ModelConfig
from ....utils import ActivationType, AuxStreamType, EventType

#: Rows along N each locality-domain shard must be a multiple of: the Rubin
#: locality-domain kernels tile N in 128-row blocks and the shard scales are
#: cut on 128-row swizzle atoms.
_LOCALITY_DOMAIN_SHARD_N_ALIGNMENT = 128


class CuteDslLocalityDomainMixin:
    """The finalize output memset stream and the locality-domain weight shards.

    The host calls :meth:`_init_moe_output_memset` and then
    :meth:`_init_locality_domain` (or :meth:`_init_output_memset_and_locality_domain`
    for both) from its constructor, after ``apply_moe_impl_construction_state``
    and before it creates weights. Besides that state, the planner reads
    ``self.use_fused_finalize``, which the host sets first. The memset init
    sets ``self.event_dict`` to the two events it uses, replacing any earlier
    value. The lifecycle overrides chain to ``super()``, so the mixin sits
    before ``MoEImplBase`` in the bases.
    """

    #: Full-size scale parameters that ``_locality_domain_scale_shard``
    #: replaces with per-partition shards, released after the split.
    _locality_domain_full_scale_params: tuple[str, ...] = ()

    def _init_output_memset_and_locality_domain(self, model_config: ModelConfig) -> None:
        """Run :meth:`_init_moe_output_memset`, then :meth:`_init_locality_domain`."""
        self._init_moe_output_memset()
        self._init_locality_domain(model_config)

    def _init_moe_output_memset(self) -> None:
        """Create the output-memset aux stream and the events that order it."""
        # Output-memset overlap is independent of MoE chunking, so ensure its
        # stream and events exist even if the parent creates no chunking event.
        if self.aux_stream_dict is None:
            self.aux_stream_dict = {}
        if AuxStreamType.MoeOutputMemset not in self.aux_stream_dict:
            self.aux_stream_dict[AuxStreamType.MoeOutputMemset] = torch.cuda.Stream()
        self.event_dict = {
            key: torch.cuda.Event() for key in [EventType.Main, EventType.MoeOutputMemset]
        }

    def _init_locality_domain(self, model_config: ModelConfig) -> None:
        """Plan the locality-domain split and, when enabled, create its runtime."""
        # locality domain: fork/join with _locality_domain kernel variants + shared output buffers.
        # Weight splitting happens in transform_weights after normal loading.
        self._locality_domain_runtime = None
        self._locality_domain_weight_shards = None  # set in transform_weights
        planner = LocalityDomainExecutionPlanner(model_config.locality_domain_policy)
        self._locality_domain_plan = self._adjust_locality_domain_plan(
            planner.plan_moe(
                self.quant_config,
                moe_backend=model_config.moe_backend,
                use_fused_finalize=self.use_fused_finalize,
                dtype_activation=self.dtype,
                activation=ActivationType(self.activation_type).name,
            )
        )
        if self._locality_domain_plan.enabled:
            self._locality_domain_runtime = LocalityDomainRuntime(
                self._locality_domain_plan.num_partitions
            )

    def _adjust_locality_domain_plan(self, plan: PartitionPlan) -> PartitionPlan:
        """Veto a plan whose shards the locality-domain kernels cannot run.

        The default split (:meth:`_split_weights_for_locality_domain`) cuts
        FC1's N (``2 * intermediate_size_per_partition``; the planner admits
        gated activations only) and FC2's N (``hidden_size``) into equal
        halves, each of which must be a multiple of
        ``_LOCALITY_DOMAIN_SHARD_N_ALIGNMENT``; otherwise the layer runs the
        full-GPU kernels. A leaf with another split overrides this.
        """
        if not plan.enabled:
            return plan
        granule = _LOCALITY_DOMAIN_SHARD_N_ALIGNMENT * plan.num_partitions
        fc1_n = 2 * self.intermediate_size_per_partition
        if fc1_n % granule == 0 and self.hidden_size % granule == 0:
            return plan
        reason = (
            f"locality domain MoE splits FC1 N ({fc1_n}) and FC2 N "
            f"({self.hidden_size}) into {plan.num_partitions} shards that must "
            f"each be a multiple of {_LOCALITY_DOMAIN_SHARD_N_ALIGNMENT}; "
            "running the full-GPU kernels"
        )
        logger.warning(f"{type(self).__name__}: {reason}")
        return PartitionPlan(enabled=False, reason_if_disabled=reason)

    def _has_moe_output_memset_aux_stream(self) -> bool:
        """Whether :meth:`_init_moe_output_memset` set up the stream and both events."""
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

    @property
    def uses_locality_domain(self) -> bool:
        """Whether this layer runs the locality-domain split."""
        return self._locality_domain_plan.enabled

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
        """Drop existing shards before a reload, then load the full weights."""
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
        """Run the base transform, then split the weights into localized shards."""
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
        """Require built shards and resolve the remainder stream before capture."""
        super().cache_derived_state()
        if self._locality_domain_runtime is not None:
            # Staged loads (e.g. GMS read-only, MX receiver) skip transform_weights,
            # which builds the shards, and do not carry them over from the source.
            if self._locality_domain_weight_shards is None:
                raise NotImplementedError(
                    f"Locality domain {type(self).__name__} requires transform_weights() "
                    "to build its weight shards; staged loads that skip it are "
                    "not supported"
                )
            # Weight splitting initializes the process-lifetime locality domain resource.
            # Resolve the borrowed remainder stream now, never during capture.
            self._get_reserved_moe_output_memset_stream()

    def pre_reload_weights(self) -> None:
        """Drop the shards so the next transform splits the reloaded weights."""
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

        Raises:
            ValueError: If a half is not a multiple of
                ``_LOCALITY_DOMAIN_SHARD_N_ALIGNMENT``, e.g. when the stored
                weights are padded past the widths
                :meth:`_adjust_locality_domain_plan` checked.
        """
        num_p = self._locality_domain_plan.num_partitions
        n1 = self.w3_w1_weight.size(1)
        n2 = self.w2_weight.size(1)
        granule = _LOCALITY_DOMAIN_SHARD_N_ALIGNMENT * num_p
        if n1 % granule != 0 or n2 % granule != 0:
            raise ValueError(
                f"{type(self).__name__}: locality domain shards of FC1 N ({n1}) and "
                f"FC2 N ({n2}) over {num_p} partitions must each be a multiple of "
                f"{_LOCALITY_DOMAIN_SHARD_N_ALIGNMENT}"
            )
        half_n1 = n1 // num_p
        half_n2 = n2 // num_p
        shards = []
        for pid in range(num_p):
            with self._locality_domain_runtime.partition_weight_context(pid):
                w3_w1_slice = self.w3_w1_weight[:, pid * half_n1 : (pid + 1) * half_n1]
                w2_slice = self.w2_weight[:, pid * half_n2 : (pid + 1) * half_n2]
                shard = {
                    "w3_w1_weight": _copy_to_new_cuda_allocation(w3_w1_slice),
                    "w2_weight": _copy_to_new_cuda_allocation(w2_slice),
                }
                shard.update(self._locality_domain_scale_shard(pid, half_n1, half_n2))
                shards.append(shard)
        return shards
