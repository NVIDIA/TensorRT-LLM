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
"""The concrete blocks that belong to whoever owns MoE expert weights.

Both :class:`~.impl_base.MoEImplBase` (an execution unit) and the legacy ``MoE``
layer include these. They are stated once, here, because a class that allocates
and loads expert weights needs them regardless of whether it also happens to be
a complete layer.

Only *implementations* live here. Each base states its own abstract contract,
because the two contracts differ on purpose -- ``MoEImplBase.run_moe`` is
narrower than ``MoE.run_moe``, and only ``MoEImplBase`` enforces completeness at
construction time. Sharing the contract would force those two into one shape and
would need ``ABCMeta`` to see markers through a plain mixin, which it does not.

The blocks deliberately exclude everything a *complete layer* owns -- no
``forward``, no ``_register_layer``, no routing, no reduce/allreduce, and none
of the EPLB forward-time orchestration (``_load_balancer_*``, ``repeat_idx``),
which is driven from the wrapper and stays on ``MoE``.
"""

from typing import Dict, List

import torch


class MoEWeightOwnerMixin:
    """Everything that has to sit on the module holding the expert weights.

    Three groups live here -- the staged operations over those weights, the
    quantization format they were created in, and the partition shape -- and
    what they share is ownership, not any one of the three. The operations
    delegate to ``self.quant_method`` (a ``quantization.FusedMoEMethodBase``),
    which writes into this same module; the format predicates read
    ``self.quant_config`` and assert ``self._weights_created``. Neither works
    from anywhere but the owner, which is what makes ownership the criterion
    and the name.
    """

    # ---- weight lifecycle -------------------------------------------------
    def transform_weights(self) -> None:
        if getattr(self, "_weights_transformed", False):
            return
        self.quant_method.transform_weights(self)
        self._weights_transformed = True

    def cache_derived_state(self) -> None:
        self.quant_method.cache_derived_state(self)

    def post_load_weights(self) -> None:
        self.transform_weights()
        self.cache_derived_state()

    def process_weights_after_loading(self):
        """
        Apply quantization processing to loaded weights.

        When allow_partial_loading=True is used in load_weights(), this method
        must be called separately to complete the loading setup.
        """
        if hasattr(self.quant_method, "process_weights_after_loading"):
            self.quant_method.process_weights_after_loading(self)

    def pre_reload_weights(self):
        """
        Prepare tensors for weight reloading by reverting them to their original creation shape.
        """
        assert hasattr(self.quant_method, "pre_reload_weights"), (
            "pre_reload_weights is not supported for this quant method"
        )
        if self._using_load_balancer():
            raise NotImplementedError(
                "Weight reloading is not compatible with Expert Parallel Load Balancer (EPLB). "
            )
        self.quant_method.pre_reload_weights(self)

    # ---- quantization state of the weights this module holds ---------------
    @property
    def has_any_quant(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_any_quant(
            exclude_kv_cache=True
        )

    # The following three properties are common enough to warrant inclusion in the interface.
    @property
    def has_fp8_qdq(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_fp8_qdq()

    @property
    def has_deepseek_fp8_block_scales(self):
        assert self._weights_created
        return (
            self.quant_config is not None
            and self.quant_config.layer_quant_mode.has_fp8_block_scales()
        )

    @property
    def has_nvfp4(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_nvfp4()

    @property
    def has_nvfp4_activation_quantization(self):
        assert self._weights_created
        return self.quant_method.quantizes_nvfp4_activations

    @property
    def has_w4a8_nvfp4_fp8(self):
        assert self._weights_created
        return (
            self.quant_config is not None
            and self.quant_config.layer_quant_mode.has_w4a8_nvfp4_fp8()
        )

    @property
    def has_w4a8_mxfp4_fp8(self):
        assert self._weights_created
        return (
            self.quant_config is not None
            and self.quant_config.layer_quant_mode.has_w4a8_mxfp4_fp8()
        )

    @property
    def has_w4a8_mxfp4_mxfp8(self):
        assert self._weights_created
        return (
            self.quant_config is not None
            and self.quant_config.layer_quant_mode.has_w4a8_mxfp4_mxfp8()
        )

    @property
    def has_w4a16_mxfp4(self):
        assert self._weights_created
        return (
            self.quant_config is not None and self.quant_config.layer_quant_mode.has_w4a16_mxfp4()
        )

    @property
    def has_mxfp8(self):
        assert self._weights_created
        return self.quant_config is not None and self.quant_config.layer_quant_mode.has_mxfp8()

    # ---- shape / partition -------------------------------------------------
    @property
    def expand_intermediate_size_per_partition(self):
        return self.intermediate_size_per_partition * self.intermediate_size_expand_ratio


class MoEEplbWeightLayoutMixin:
    """The EPLB half that follows the weights, not the forward pass.

    "Weight" here means weight-side, not static-only: four of these members
    return early unless ``_using_dynamic_load_balancer()``, so the block serves
    the dynamic balancer at least as much as the static one. Naming it after
    static layout would send a reader looking for the dynamic path elsewhere.

    ``create_weights`` / ``load_weights`` delegate to
    ``quantization.FusedMoEMethodBase``, which reads the layout straight off
    the weight owner: ``expert_size_per_partition``,
    ``initial_local_expert_ids``, ``layer_load_balancer``, ``num_slots``,
    ``layer_idx``, ``initial_global_assignments``. Weight *shapes* depend on
    those values, so they must be settled before ``create_weights`` runs --
    which is what makes them the weight owner's state rather than the
    wrapper's.

    The complementary half -- ``_init_load_balancer``, the ``_load_balancer_*``
    methods and ``repeat_idx`` / ``repeat_count`` -- is forward-time
    orchestration driven from the wrapper and stays on ``MoE``.
    """

    def _add_raw_shared_weights_for_unmap(self, weight_tensors: List[torch.Tensor]):
        if self._using_dynamic_load_balancer():
            self.layer_load_balancer._add_raw_host_weight_for_unmap(weight_tensors)

    def _supports_load_balancer(self) -> bool:
        """Check if this MoE implementation supports load balancer.

        Subclasses can override this to indicate load balancer support.
        """
        return False

    def _using_load_balancer(self) -> bool:
        """Check if this MoE is using load balancer."""
        return self.layer_load_balancer is not None

    def _using_dynamic_load_balancer(self) -> bool:
        """Check if this MoE is using dynamic load balancer."""
        if self.layer_load_balancer:
            return self.layer_load_balancer.is_dynamic_routing()
        return False

    def register_parameter_weight_slot_fn(self, weight_name: str, local_slot_id: int):
        """Register parameter weight slot function for load balancer."""
        if not self._using_dynamic_load_balancer():
            return

        assert hasattr(self, weight_name), f"MoE doesn't have weight attr: {weight_name}"
        weight_tensor = getattr(self, weight_name).data[local_slot_id]
        self.layer_load_balancer.register_weight_slot(local_slot_id, weight_name, weight_tensor)

    def register_to_fix_weight_fn(self, weight_name: str):
        """Register weight fixing function for load balancer."""
        if not self._using_dynamic_load_balancer():
            return

        assert hasattr(self, weight_name), f"MoE doesn't have weight attr: {weight_name}"
        param = getattr(self, weight_name)
        weight_tensor = param.detach()
        assert isinstance(weight_tensor, torch.Tensor), f"weight {weight_name} should be a tensor"
        assert weight_tensor.is_contiguous(), (
            f"weight {weight_name} should be contiguous, "
            f"shape={weight_tensor.shape}, strides={weight_tensor.stride()}"
        )
        assert (
            weight_tensor.numel() * weight_tensor.element_size()
            == weight_tensor.untyped_storage().size()
        ), (
            f"weight {weight_name} shape={weight_tensor.shape} "
            f"storage_size = {weight_tensor.untyped_storage().size()}, "
            f"numel={weight_tensor.numel()}, eltsize={weight_tensor.element_size()}, "
            f"dtype={weight_tensor.dtype}"
        )
        self.layer_load_balancer.make_tensor_host_accessible(weight_tensor)
        param.data = weight_tensor

    def register_all_parameter_slot_and_to_fix_weight_fns(
        self, weight_and_tensor_dict: Dict[str, torch.Tensor]
    ):
        """Register all parameter slot and weight fixing functions for load balancer."""
        if not self._using_dynamic_load_balancer():
            return

        # Register weight functions for each local slot
        for local_slot_id, expert_id in enumerate(self.initial_local_expert_ids):
            for weight_name in weight_and_tensor_dict:
                self.layer_load_balancer.add_register_weight_fn(
                    self.register_parameter_weight_slot_fn, (weight_name, local_slot_id)
                )

        # Register weight migration functions
        for weight_name in weight_and_tensor_dict:
            self.layer_load_balancer.add_to_migrate_weight_fn(
                self.register_to_fix_weight_fn, (weight_name,)
            )

        # Setup host tensor sharing
        local_shared_load_expert_ids = self.layer_load_balancer.get_load_expert_ids()
        for expert_id in range(self.num_experts):
            for weight_name, weight_tensor in weight_and_tensor_dict.items():
                if expert_id in local_shared_load_expert_ids:
                    local_slot_id = local_shared_load_expert_ids.index(expert_id)
                    self.layer_load_balancer.host_tensor_sharer.share_host_tensor_with_shape(
                        expert_id, weight_name, weight_tensor[local_slot_id]
                    )
                else:
                    self.layer_load_balancer.host_tensor_sharer.pre_register_host_tensor_with_shape(
                        expert_id, weight_name, weight_tensor.dtype, weight_tensor[0].shape
                    )
