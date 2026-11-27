# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bind Shortcut's expert banks to the executor's standard expert load balancer."""

from tensorrt_llm._torch.moe.fused_moe.moe_load_balancer import get_moe_load_balancer


class ExpertPlacement:
    def __init__(self, config, manager, layer_idx, num_experts, topk):
        placement = config.get_layer_initial_global_assignments(layer_idx)
        if placement is None:
            resident = num_experts // config.ep_size
            placement = [
                (rank * resident + slot) % num_experts
                for rank in range(config.ep_size)
                for slot in range(config.num_local_slots)
            ]
        if set(placement) != set(range(num_experts)):
            raise ValueError(
                f"Expert placement for layer {layer_idx} must cover every logical expert"
            )
        self.initial_local_expert_ids = placement[config.slot_start : config.slot_end]
        self.layer_load_balancer = manager.add_layer(num_experts, topk, config.num_local_slots)
        self.layer_load_balancer.set_initial_weight_assignments(placement)

    def bind_weights(self, parameters, prefix, weights_for_host):
        layer = self.layer_load_balancer
        for name, bank in weights_for_host.items():
            key = f"{prefix}experts_{name}"
            layer.register_local_host_weights(name, bank, parameters[key])
            layer.add_to_migrate_weight_fn(self._bind_device_slots, (name, parameters, key))

    def _bind_device_slots(self, name, parameters, key):
        # The loader can replace a parameter when it moves the model to CUDA.
        parameter = parameters[key]
        slots = parameter.detach()
        if not slots.is_contiguous() or slots.nbytes != slots.untyped_storage().nbytes():
            raise ValueError(f"Expert slots must own contiguous storage: {key}")
        layer = self.layer_load_balancer
        layer.make_tensor_host_accessible(slots)
        parameter.data = slots
        for slot_id, weight in enumerate(slots.unbind()):
            layer.register_weight_slot(slot_id, name, weight)

    def route(self, ids):
        layer = self.layer_load_balancer
        if layer.is_dynamic_routing():
            layer.start_wait_gpu_stage()
            layer.done_wait_gpu_stage()
            # The decoder already gathered the logical IDs from all attention-DP ranks.
            layer.update_statistic_with_global_ids(ids, is_first_stage=True, is_last_stage=True)
        return layer.route(ids, offset_by_ep_rank=False)

    def finish(self):
        layer = self.layer_load_balancer
        if layer.is_dynamic_routing():
            layer.start_set_cpu_stage()
            layer.done_set_cpu_stage()


def create_placements(model_config, kinds, moe_kinds, num_experts, topk):
    config = model_config.moe_load_balancer
    manager = get_moe_load_balancer()
    if manager is None:
        raise ValueError(
            "Shortcut expert placement requires an active expert-parallel load balancer"
        )
    if config.mode != "standard" or config.warmup_placement is not None:
        raise ValueError("Shortcut supports standard static or online expert placement")
    if config.num_slots < num_experts:
        raise ValueError("The number of physical expert slots must cover every logical expert")
    return {
        f"l{i}_": ExpertPlacement(config, manager, i, num_experts, topk)
        for i, kind in enumerate(kinds)
        if kind in moe_kinds
    }
