# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-level owner of the per-layer dynamic EPLB layers and their GPU resources."""

from __future__ import annotations

from typing import Any, Optional

import torch

from ..moe_load_balancer import BaseMoeLoadBalancer
from .layer import PerLayerDynamicMoeLayerLoadBalancer, _append_note


class PerLayerDynamicMoeLoadBalancer(BaseMoeLoadBalancer):
    """Model-level owner for per-layer dynamic EPLB layers and GPU resources."""

    def __init__(
        self,
        *,
        ep_rank: int,
        ep_size: int,
        config: Any,
        mapping: Optional[Any] = None,
    ) -> None:
        # Per-layer dynamic EPLB has no C++ standard-EPLB object and creates no
        # MPI split; its EP communicator is resolved by the first MoE layer.
        if ep_size <= 1 or not 0 <= ep_rank < ep_size:
            raise ValueError(f"invalid EPLB EP rank/size: rank={ep_rank}, size={ep_size}")
        num_slots = int(config.num_slots)
        if num_slots <= 0 or num_slots % ep_size != 0:
            raise ValueError(
                f"num_slots must be positive and divisible by EP size; got {num_slots}"
            )

        self.is_shutdown = False
        self.ep_rank = int(ep_rank)
        self.ep_size = int(ep_size)
        self.config = config
        self.mapping = mapping
        self.num_slots = num_slots
        self.num_local_slots = num_slots // ep_size
        self.auxiliary_sms = int(getattr(config, "auxiliary_sms", 4))
        if self.auxiliary_sms <= 0:
            raise ValueError("auxiliary_sms must be positive")
        helper_slot_mode = getattr(config, "helper_slot_mode", "per_layer_resident")
        if helper_slot_mode not in ("per_layer_resident", "shared_banks"):
            raise ValueError(f"unknown helper_slot_mode {helper_slot_mode!r}")
        # Resident helper experts need helper rows that only their layer writes.
        self.helper_residency = helper_slot_mode == "per_layer_resident"

        # State read by the shared lifecycle and by the engine.
        self._previous_balancer = None
        self.single_layer_load_balancers: list[PerLayerDynamicMoeLayerLoadBalancer] = []
        self.next_layer_repeated_count: Optional[int] = None
        self.iter_id = 0
        self.in_iter = False
        self.enable_statistic = False
        self.enable_update_weights = False
        self._copy_stream: Optional[torch.cuda.Stream] = None
        self._copy_stream_device: Optional[int] = None
        self._ep_comm: Optional[Any] = None
        self._scheduler_spin_bound: Optional[Any] = None

        # Providers use this manager as their scope owner. Resource references
        # stay reachable here, never through Mapping or a process-global registry.
        self._arena_providers: list[Any] = []
        self._rebalance_shared_slot_pools: dict[tuple, Any] = {}
        self._rebalance_shared_slot_pool_order: list[Any] = []

    def __del__(self) -> None:
        # Teardown is collective and must be driven explicitly by the worker;
        # a Python finalizer cannot safely enter EP collectives.
        pass

    def __repr__(self) -> str:
        return (
            f"PerLayerDynamicMoeLoadBalancer(ep_rank={self.ep_rank}, "
            f"ep_size={self.ep_size}, num_slots={self.num_slots}, "
            f"auxiliary_sms={self.auxiliary_sms})"
        )

    def is_static_routing(self) -> bool:
        return False

    def set_repeated_for_next_layer(self, repeated_count: int) -> None:
        if repeated_count <= 0:
            raise ValueError("repeat count must be positive")
        self.next_layer_repeated_count = int(repeated_count)

    def add_layer(
        self,
        expert_count: int,
        top_k: int,
        slot_count_per_rank: int,
        aux_stream: Optional[torch.cuda.Stream] = None,
    ) -> PerLayerDynamicMoeLayerLoadBalancer:
        del aux_stream  # The scheduler owns its high-priority copy stream.
        self.config.validate_expert_capacity(expert_count)
        if int(slot_count_per_rank) != self.num_local_slots:
            raise ValueError(
                "per-layer dynamic EPLB layer capacity must match the configured "
                f"local slot count; got {slot_count_per_rank}, expected "
                f"{self.num_local_slots}"
            )
        repeated_count = self.next_layer_repeated_count or 1
        self.next_layer_repeated_count = None
        layer = PerLayerDynamicMoeLayerLoadBalancer(
            self,
            layer_idx=len(self.single_layer_load_balancers),
            expert_count=expert_count,
            top_k=top_k,
            compute_slots_per_rank=slot_count_per_rank,
            repeated_count=repeated_count,
        )
        self.single_layer_load_balancers.append(layer)
        return layer

    def get_copy_stream(self, device: int) -> torch.cuda.Stream:
        """Return this model's single highest-priority auxiliary stream."""
        device = int(device)
        if self._copy_stream is None:
            total_sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
            if self.auxiliary_sms >= total_sms:
                raise ValueError(
                    "auxiliary_sms must leave at least one SM for the compute stream; "
                    f"got {self.auxiliary_sms} of {total_sms}"
                )
            _, priority = torch.cuda.Stream.priority_range()
            self._copy_stream = torch.cuda.Stream(device=device, priority=priority)
            self._copy_stream_device = device
        elif self._copy_stream_device != device:
            raise RuntimeError("one per-layer dynamic EPLB manager cannot span CUDA devices")
        return self._copy_stream

    def get_ep_comm(self, process_group: Any) -> Any:
        """Return this model's adapter for its resolved EP ProcessGroup."""
        from .arena import _EpComm

        if self._ep_comm is None:
            self._ep_comm = _EpComm(process_group)
        elif self._ep_comm.process_group is not process_group:
            raise RuntimeError("one per-layer dynamic EPLB manager cannot span EP ProcessGroups")
        self._ep_comm.validate_geometry(expected_rank=self.ep_rank, expected_world=self.ep_size)
        return self._ep_comm

    def get_scheduler_spin_bound(self) -> Any:
        """Return the HALO-Q spin bound shared by this model's layers.

        It is registered with the MoE communication timeout guard, which
        lengthens it while the engine warms up.
        """
        if self._scheduler_spin_bound is None:
            from ....autotuner import AutoTuner
            from ..moe_comm_timeout_guard import (
                get_moe_comm_timeout_budgets,
                register_moe_comm_timeout_proxy,
            )
            from .planner import SchedulerSpinBound

            spin_bound = SchedulerSpinBound(
                warmup_seconds=get_moe_comm_timeout_budgets().warmup_seconds,
                is_tuning=lambda: AutoTuner.get().is_tuning_mode,
            )
            register_moe_comm_timeout_proxy(spin_bound)
            self._scheduler_spin_bound = spin_bound
        return self._scheduler_spin_bound

    def create_arena_provider(self) -> Any:
        """Create a per-layer provider scoped to this model manager."""
        from .arena import SharedSlotArenaProvider

        provider = SharedSlotArenaProvider(self, per_layer_helper_sets=self.helper_residency)
        self._arena_providers.append(provider)
        return provider

    def adopt_arena_provider(self, provider: Any) -> None:
        """Retain a transitional provider already created by the allocator."""
        if provider not in self._arena_providers:
            self._arena_providers.append(provider)

    def _register_runtime(self, layer: PerLayerDynamicMoeLayerLoadBalancer) -> None:
        if layer not in self.single_layer_load_balancers:
            raise ValueError("cannot bind a layer owned by another EPLB manager")
        assert layer._arena is not None
        self.adopt_arena_provider(layer._arena.provider)

    def start_iter(self) -> None:
        if self.in_iter:
            raise RuntimeError("already in a MoE load-balancer iteration")
        self.in_iter = True

    def end_iter(self) -> None:
        if not self.in_iter:
            raise RuntimeError("not in a MoE load-balancer iteration")
        unfinished = [
            layer.layer_idx
            for layer in self.single_layer_load_balancers
            if layer.replica_plan is not None
        ]
        if unfinished:
            raise RuntimeError(f"unfinished replica plans at iteration end: {unfinished}")
        self.in_iter = False
        self.iter_id += 1

    def abort_iter(self, error: BaseException) -> None:
        """End a failed iteration after releasing the plans it left unfinished.

        A forward that raises between a layer's route and its post-``run_moe``
        hook leaves that layer's plan pending.
        """
        try:
            for layer in self.single_layer_load_balancers:
                if layer.replica_plan is None:
                    continue
                try:
                    layer.finish_replica_plan()
                except BaseException as cleanup_error:  # noqa: BLE001
                    _append_note(
                        error,
                        "per-layer dynamic EPLB could not release the plan of layer "
                        f"{layer.layer_idx}: {cleanup_error!r}",
                    )
        finally:
            self.in_iter = False
            self.iter_id += 1

    def register_weight_slots_after_to_cuda(self) -> None:
        # Live aliases are registered by the per-layer dynamic arena binder.
        pass

    def finalize_model(self) -> None:
        # Scheduler groups are built when each layer's final aliases exist.
        pass

    def set_warm_up_iter_count(self, iter_count: int) -> None:
        del iter_count

    def shutdown(self, local_safe: bool = True) -> bool:
        """Collectively close this model's pools in deterministic layer order."""
        if self.is_shutdown:
            return True

        # Agree on local readiness before any rank starts destructive teardown;
        # a rank is unsafe while any plan is pending. Every MoE layer resolves
        # the EP communicator in create_weights, and a rank whose model load
        # failed never owns an engine that reaches this method, so the
        # communicator exists on every EP rank or on none.
        local_safe = bool(
            local_safe
            and not self.in_iter
            and all(layer.replica_plan is None for layer in self.single_layer_load_balancers)
        )
        comm = self._ep_comm
        if comm is not None:
            states = comm.allgather(
                {
                    "local_safe": local_safe,
                    "pool_count": len(self._rebalance_shared_slot_pool_order),
                }
            )
            pool_counts = {state["pool_count"] for state in states}
            if len(pool_counts) != 1:
                raise RuntimeError(
                    "per-layer dynamic EPLB pool counts differ across the EP communicator"
                )
            local_safe = all(state["local_safe"] for state in states)
        if not local_safe:
            return False

        # Snapshot creation order because each successful close removes itself.
        for pool in list(self._rebalance_shared_slot_pool_order):
            if not pool.close_collectively(local_safe=True):
                return False

        # Pool teardown closes its registered producers. Calling close again is
        # deliberately allowed and also covers a transitional group that has
        # not yet been registered with a pool.
        for layer in self.single_layer_load_balancers:
            group = layer._scheduler_group
            if group is not None:
                group.close()
            layer._clear_runtime()
        self._arena_providers.clear()
        self._ep_comm = None
        if self._scheduler_spin_bound is not None:
            from ..moe_comm_timeout_guard import unregister_moe_comm_timeout_proxy

            unregister_moe_comm_timeout_proxy(self._scheduler_spin_bound)
            self._scheduler_spin_bound = None
        self._rebalance_shared_slot_pools.clear()
        self._rebalance_shared_slot_pool_order.clear()
        self._copy_stream = None
        self._copy_stream_device = None
        self.is_shutdown = True
        return True
