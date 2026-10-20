# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Model-owned lifecycle for per-layer dynamic expert load balancing.

The standard EPLB implementation couples route updates to host weight
migration. Per-layer dynamic EPLB instead produces one device-side replica plan
for every MoE invocation and copies helper weights on a CUDA stream. These
classes preserve the existing load-balancer seams while keeping the scheduler,
arena, and their teardown under one model-level owner.
"""

from __future__ import annotations

import weakref
from typing import Any, Callable, Optional, Sequence

import torch

from .impl_contract import MoEReplicaPlan
from .moe_load_balancer import BaseMoeLoadBalancer


class PerLayerDynamicMoeLayerLoadBalancer:
    """Adapter between the common MoE routing seam and one replica planner."""

    # ``route`` publishes a ``MoEReplicaPlan`` that the caller must finish.
    produces_replica_plan = True

    def __init__(
        self,
        manager: "PerLayerDynamicMoeLoadBalancer",
        *,
        layer_idx: int,
        expert_count: int,
        top_k: int,
        compute_slots_per_rank: int,
        repeated_count: int = 1,
    ) -> None:
        if expert_count <= 0 or expert_count % manager.ep_size != 0:
            raise ValueError(
                "Per-layer dynamic EPLB requires the expert count to be positive "
                f"and divisible by EP size; got E={expert_count}, EP={manager.ep_size}."
            )
        resident = expert_count // manager.ep_size
        compute = int(compute_slots_per_rank)
        if compute <= resident:
            raise ValueError(
                "Per-layer dynamic EPLB requires at least one helper slot per rank; "
                f"got H={resident}, M={compute}."
            )
        if top_k <= 0 or repeated_count <= 0:
            raise ValueError("top_k and repeated_count must be positive")

        self.manager = manager
        self.layer_idx = int(layer_idx)
        self.expert_count = int(expert_count)
        self.top_k = int(top_k)
        self.resident_experts_per_rank = resident
        self.compute_slots_per_rank = compute
        self.helper_slots_per_rank = compute - resident
        self.repeated_count = int(repeated_count)

        self._arena: Optional[Any] = None
        self._scheduler_group: Optional[Any] = None
        self._bound_backend_ref: Optional[weakref.ReferenceType[Any]] = None
        self._pending_replica_plan: Optional[Any] = None

    def get_layer_idx(self) -> int:
        return self.layer_idx

    def get_repeat_count(self) -> int:
        return self.repeated_count

    def get_load_expert_ids(self) -> list[int]:
        start = self.manager.ep_rank * self.resident_experts_per_rank
        return list(range(start, start + self.resident_experts_per_rank))

    def is_static_routing(self) -> bool:
        return False

    def is_dynamic_routing(self) -> bool:
        """Keep the legacy host-migration path disabled.

        The common route seam still calls :meth:`route` whenever a layer
        adapter exists. The legacy meaning of ``is_dynamic_routing`` is
        broader: it enables CPU statistics and host-weight migration, neither
        of which belongs to the per-layer dynamic CUDA planner.
        """
        return False

    def need_load_shared_weights(self) -> bool:
        return False

    def set_initial_weight_assignments(
        self, initial_weight_assignments: Optional[Sequence[int]]
    ) -> None:
        if initial_weight_assignments is None:
            return
        assignments = tuple(int(expert) for expert in initial_weight_assignments)
        canonical = tuple(range(self.expert_count))
        if assignments != canonical:
            raise ValueError(
                "Per-layer dynamic EPLB requires the canonical initial assignment "
                "range(num_experts); custom initial_global_assignments are only "
                "valid in standard mode."
            )

    # These hooks are part of the standard EPLB layer interface. Routing and
    # statistics are produced by HALO-Q, so the corresponding CPU stages are
    # intentionally empty here.
    def start_wait_gpu_stage(self) -> None:
        pass

    def done_wait_gpu_stage(self) -> None:
        pass

    def start_set_cpu_stage(self) -> None:
        pass

    def done_set_cpu_stage(self) -> None:
        pass

    def update_local_statistic(self, *args, **kwargs) -> None:
        pass

    def update_statistic_with_local_ids(self, *args, **kwargs) -> None:
        pass

    def update_statistic_with_global_ids(self, *args, **kwargs) -> None:
        pass

    def update_statistic_with_gathered_statistic(self, *args, **kwargs) -> None:
        pass

    def get_local_statistic_tensor(self) -> None:
        return None

    def bind_runtime(self, *, arena: Any, scheduler_group: Any) -> None:
        """Attach resources built after final weight storage is available."""
        if arena is None or scheduler_group is None:
            raise ValueError("arena and scheduler_group are both required")
        if self._arena is not None:
            if arena is self._arena and scheduler_group is self._scheduler_group:
                return
            raise RuntimeError(f"per-layer dynamic EPLB layer {self.layer_idx} is already bound")

        arena_home = getattr(arena, "home_experts", self.resident_experts_per_rank)
        arena_helpers = getattr(arena, "helper_slots", self.helper_slots_per_rank)
        group_home = getattr(scheduler_group, "home_experts", self.resident_experts_per_rank)
        group_helpers = getattr(scheduler_group, "helper_slots", self.helper_slots_per_rank)
        if (int(arena_home), int(arena_helpers)) != (
            self.resident_experts_per_rank,
            self.helper_slots_per_rank,
        ):
            raise ValueError("arena H/S geometry does not match the EPLB layer")
        if (int(group_home), int(group_helpers)) != (
            self.resident_experts_per_rank,
            self.helper_slots_per_rank,
        ):
            raise ValueError("scheduler H/S geometry does not match the EPLB layer")

        self._arena = arena
        self._scheduler_group = scheduler_group
        self.manager._register_runtime(self)

    def allocate_arena(
        self,
        *,
        bundle: Any,
        ep_process_group: Any,
        mapping: Any,
        device: int,
        layer_idx: Optional[int],
    ) -> Any:
        """Allocate this layer's H + S live-weight arena from model-owned resources.

        Collective across the EP group: every rank calls it from the same point
        of the backend's ``create_weights``, after the symmetric-buffer
        rendezvous and before the kernel-plane Parameters are registered.
        ``bundle`` is the quant method's live-weight plane spec.
        """
        from .mega_moe.rebalance_live_arena import allocate_rebalance_arena

        return allocate_rebalance_arena(
            home_experts=self.resident_experts_per_rank,
            helper_slots=self.helper_slots_per_rank,
            bundle=bundle,
            ep_comm=self.manager.get_ep_comm(ep_process_group),
            mapping=mapping,
            provider=self.manager.create_arena_provider(),
            device=int(device),
            layer_idx=layer_idx,
        )

    def bind_backend(self, backend: Any) -> None:
        """Build and own the producer after the backend's live aliases exist."""
        arena = backend.live_weight_arena
        if arena is None:
            raise RuntimeError("per-layer dynamic EPLB backend has no live arena")
        if self._arena is not None or self._scheduler_group is not None:
            bound_backend = (
                self._bound_backend_ref() if self._bound_backend_ref is not None else None
            )
            if bound_backend is not backend:
                raise RuntimeError(
                    f"per-layer dynamic EPLB layer {self.layer_idx} is already "
                    "bound to a different backend"
                )
            if self._arena is not arena:
                raise RuntimeError(
                    f"per-layer dynamic EPLB layer {self.layer_idx} backend "
                    "changed its rebalance arena after binding"
                )
            if self._scheduler_group is None:
                raise RuntimeError(
                    f"per-layer dynamic EPLB layer {self.layer_idx} has an incomplete runtime binding"
                )
            return

        backend_ref = weakref.ref(backend)
        from .mega_moe.rebalance_slot_scheduler import build_rebalance_slot_scheduler_group

        if bool(getattr(backend, "tactic_autotune", False)):
            # Synthetic profiling may visit every helper row before a real
            # plan selects it. Seed those rows from resident weights once;
            # serving generations still replace active rows and publish READY.
            home = int(arena.home_experts)
            with torch.no_grad():
                for plane in arena.tekit_alias_views:
                    for helper in range(int(arena.helper_slots)):
                        plane[home + helper].copy_(plane[helper % home], non_blocking=True)

        # ``layer_idx`` is the backend's model layer index: shared-slot pools
        # compare producers across ranks by it.
        scheduler_group = build_rebalance_slot_scheduler_group(
            arena=arena,
            manager=self.manager,
            mapping=backend.mapping,
            ep_process_group=backend.ep_process_group,
            home_experts=self.resident_experts_per_rank,
            helper_slots=self.helper_slots_per_rank,
            topk=self.top_k,
            max_tokens_per_rank=backend.max_tokens_per_rank_capacity,
            layer_idx=backend.layer_idx,
        )
        self.bind_runtime(arena=arena, scheduler_group=scheduler_group)
        self._bound_backend_ref = backend_ref

    @property
    def replica_plan(self) -> Optional[Any]:
        return self._pending_replica_plan

    def route(
        self,
        token_selected_experts: torch.Tensor,
        offset_by_ep_rank: bool = False,
        *,
        overlap_fn: Optional[Callable[[], None]] = None,
    ) -> torch.Tensor:
        """Submit HALO-Q/TMA and return its physical routes without a host wait.

        ``overlap_fn`` runs after the copy stream is submitted but before the
        physical routes are lent to the caller, so independent main-stream work
        such as shared experts overlaps planning and the helper-weight copy.
        """
        del offset_by_ep_rank  # HALO-Q consumes global logical expert IDs.
        if overlap_fn is not None and not callable(overlap_fn):
            raise TypeError("overlap_fn must be callable")
        if self._scheduler_group is None or self._arena is None:
            raise RuntimeError(
                f"per-layer dynamic EPLB layer {self.layer_idx} has no bound runtime"
            )
        if self._pending_replica_plan is not None:
            raise RuntimeError(
                f"per-layer dynamic EPLB layer {self.layer_idx} has an unfinished plan"
            )

        if overlap_fn is None:
            physical, generation = self._scheduler_group.plan(
                token_selected_experts, defer_wait=True
            )
        else:
            part = self._scheduler_group.plan_schedule(token_selected_experts)
            try:
                overlap_fn()
            except BaseException as error:  # noqa: BLE001
                try:
                    self._scheduler_group.discard_plan(part)
                    cleanup_note = "the unused rebalance generation was safely released."
                except BaseException as cleanup_error:  # noqa: BLE001
                    cleanup_note = (
                        "rebalance generation cleanup failed and the group "
                        f"requires teardown: {cleanup_error!r}"
                    )
                notes = getattr(error, "__notes__", None)
                if type(notes) is not list:
                    notes = []
                    error.__notes__ = notes
                notes.append("overlap_fn failed after HALO-Q/TMA enqueue; " + cleanup_note)
                raise
            physical, generation = self._scheduler_group.plan_finish(part, defer_wait=True)
        generation = int(generation)
        self._arena.publish_ready_generation(generation)

        plan = MoEReplicaPlan(
            resident_experts_per_rank=self.resident_experts_per_rank,
            compute_slots_per_rank=self.compute_slots_per_rank,
            ready_flags=self._arena.terminal_flags_tensor(),
            ready_generation=generation,
            reserved_sms=self.manager.auxiliary_sms,
            _wait_for_routes=self._scheduler_group.wait_for_routes,
        )
        self._pending_replica_plan = plan
        return physical

    def require_replica_plan(self) -> Any:
        if self._pending_replica_plan is None:
            raise RuntimeError(
                f"per-layer dynamic EPLB layer {self.layer_idx} has not produced a plan"
            )
        return self._pending_replica_plan

    def finish_replica_plan(
        self,
        plan: Optional[Any] = None,
        *,
        completion_event: Optional[torch.cuda.Event] = None,
    ) -> None:
        pending = self._pending_replica_plan
        if pending is None:
            raise RuntimeError(
                f"per-layer dynamic EPLB layer {self.layer_idx} has no plan to finish"
            )
        if plan is not None and plan is not pending:
            raise RuntimeError("attempted to finish a different replica plan")
        assert self._scheduler_group is not None
        # Also makes cleanup safe when input preparation failed before the
        # backend reached its first route consumer.
        pending.wait_for_routes()
        self._scheduler_group.finish(completion_event=completion_event)
        self._pending_replica_plan = None

    @property
    def has_submission_owner(self) -> bool:
        group = self._scheduler_group
        return bool(group is not None and group.has_submission_owner)

    def handoff_submission_owner(self, execution_stream_handle: int) -> None:
        """Release this layer's warmup submitter after a caller device drain."""
        group = self._scheduler_group
        if group is not None and group.has_submission_owner:
            group.release_warmup_owner_after_sync(int(execution_stream_handle))

    def _clear_runtime(self) -> None:
        self._pending_replica_plan = None
        self._scheduler_group = None
        self._arena = None
        self._bound_backend_ref = None


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
        self.auxiliary_sms = int(getattr(config, "auxiliary_sms", 8))
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
        from .mega_moe.rebalance_live_arena import _EpComm

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
            from ...autotuner import AutoTuner
            from .mega_moe.rebalance_slot_scheduler import SchedulerSpinBound
            from .moe_comm_timeout_guard import (
                get_moe_comm_timeout_budgets,
                register_moe_comm_timeout_proxy,
            )

            spin_bound = SchedulerSpinBound(
                warmup_seconds=get_moe_comm_timeout_budgets().warmup_seconds,
                is_tuning=lambda: AutoTuner.get().is_tuning_mode,
            )
            register_moe_comm_timeout_proxy(spin_bound)
            self._scheduler_spin_bound = spin_bound
        return self._scheduler_spin_bound

    def create_arena_provider(self) -> Any:
        """Create a per-layer provider scoped to this model manager."""
        from .mega_moe.rebalance_live_arena import SharedSlotArenaProvider

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

    def register_weight_slots_after_to_cuda(self) -> None:
        # Live aliases are registered by the per-layer dynamic arena binder.
        pass

    def finalize_model(self) -> None:
        # Scheduler groups are built when each layer's final aliases exist.
        pass

    def set_warm_up_iter_count(self, iter_count: int) -> None:
        del iter_count

    @property
    def has_submission_owner(self) -> bool:
        return any(layer.has_submission_owner for layer in self.single_layer_load_balancers)

    def handoff_submission_owner(self, execution_stream_handle: int) -> None:
        """Transfer all warmup-owned submitters after one caller device drain."""
        for layer in self.single_layer_load_balancers:
            layer.handoff_submission_owner(execution_stream_handle)

    def release_quiesced_submission_owner(self, execution_stream_handle: int) -> None:
        """Release executor ownership after the caller drains the device."""
        for layer in self.single_layer_load_balancers:
            group = layer._scheduler_group
            if group is not None:
                group.release_quiesced_submission_owner(int(execution_stream_handle))

    def shutdown(self, local_safe: bool = True) -> bool:
        """Collectively close this model's pools in deterministic layer order."""
        if self.is_shutdown:
            return True

        # Agree on local readiness before any rank starts destructive teardown.
        # Every MoE layer resolves the EP communicator in create_weights, and a
        # rank whose model load failed never owns an engine that reaches this
        # method, so the communicator exists on every EP rank or on none.
        local_safe = bool(local_safe and not self.in_iter)
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
            from .moe_comm_timeout_guard import unregister_moe_comm_timeout_proxy

            unregister_moe_comm_timeout_proxy(self._scheduler_spin_bound)
            self._scheduler_spin_bound = None
        self._rebalance_shared_slot_pools.clear()
        self._rebalance_shared_slot_pool_order.clear()
        self._copy_stream = None
        self._copy_stream_device = None
        self.is_shutdown = True
        return True


__all__ = ["PerLayerDynamicMoeLayerLoadBalancer", "PerLayerDynamicMoeLoadBalancer"]
