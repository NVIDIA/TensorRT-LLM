# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-layer handle of per-layer dynamic EPLB: it plans each route and releases the plan."""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import torch

from ..impl_contract import MoEReplicaPlan

if TYPE_CHECKING:
    from .manager import PerLayerDynamicMoeLoadBalancer


def _append_note(error: BaseException, note: str) -> None:
    """Attach ``note`` to ``error`` like ``BaseException.add_note``, which Python 3.10 lacks."""
    notes = getattr(error, "__notes__", None)
    if type(notes) is not list:
        notes = []
        error.__notes__ = notes
    notes.append(note)


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
        # The arena's READY terminal view, resolved once the runtime is bound.
        self._ready_flags: Optional[torch.Tensor] = None
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

    def start_set_cpu_stage(self) -> None:
        """Release the plan of the chunk whose MoE kernel was just enqueued."""
        self.finish_replica_plan()

    # The other hooks of the standard EPLB layer interface are intentionally
    # empty: HALO-Q produces the routing and statistics on the GPU, and no CPU
    # stage migrates weights.
    def start_wait_gpu_stage(self) -> None:
        pass

    def done_wait_gpu_stage(self) -> None:
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
        if self._scheduler_group is not None:
            if arena is self._arena and scheduler_group is self._scheduler_group:
                return
            raise RuntimeError(f"per-layer dynamic EPLB layer {self.layer_idx} is already bound")
        if self._arena is not None and arena is not self._arena:
            raise RuntimeError(
                f"per-layer dynamic EPLB layer {self.layer_idx} cannot bind an arena "
                "it did not allocate"
            )

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
        self._ready_flags = arena.terminal_flags_tensor()
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

        Collective across the EP group: the quant method calls it from its
        ``create_weights`` on every rank, before it registers the kernel-plane
        Parameters. ``bundle`` is the quant method's live-weight plane spec.
        A later call returns the same arena.
        """
        if self._arena is not None:
            if bundle != self._arena.bundle:
                raise RuntimeError(
                    f"per-layer dynamic EPLB layer {self.layer_idx} already allocated "
                    "its arena for a different live-weight plane spec"
                )
            return self._arena
        from .arena import allocate_rebalance_arena

        self._arena = allocate_rebalance_arena(
            home_experts=self.resident_experts_per_rank,
            helper_slots=self.helper_slots_per_rank,
            bundle=bundle,
            ep_comm=self.manager.get_ep_comm(ep_process_group),
            mapping=mapping,
            provider=self.manager.create_arena_provider(),
            device=int(device),
            layer_idx=layer_idx,
        )
        return self._arena

    @property
    def live_weight_arena(self) -> Optional[Any]:
        """The H + S arena that the live kernel planes alias, or ``None``."""
        return self._arena

    def bind_backend(self, backend: Any) -> None:
        """Build and own the producer after the backend's live aliases exist."""
        arena = self._arena
        if arena is None:
            raise RuntimeError(
                f"per-layer dynamic EPLB layer {self.layer_idx} has no live-weight arena"
            )
        if self._scheduler_group is not None:
            bound_backend = (
                self._bound_backend_ref() if self._bound_backend_ref is not None else None
            )
            if bound_backend is not backend:
                raise RuntimeError(
                    f"per-layer dynamic EPLB layer {self.layer_idx} is already "
                    "bound to a different backend"
                )
            return

        backend_ref = weakref.ref(backend)
        from .planner import build_rebalance_slot_scheduler_group

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
                _append_note(error, "overlap_fn failed after HALO-Q/TMA enqueue; " + cleanup_note)
                raise
            physical, generation = self._scheduler_group.plan_finish(part, defer_wait=True)
        generation = int(generation)
        self._arena.publish_ready_generation(generation)

        plan = MoEReplicaPlan(
            resident_experts_per_rank=self.resident_experts_per_rank,
            compute_slots_per_rank=self.compute_slots_per_rank,
            ready_flags=self._ready_flags,
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
        # backend reached its first route consumer. This is the plan's route
        # wait, called on the group directly.
        self._scheduler_group.wait_for_routes()
        self._scheduler_group.finish(completion_event=completion_event)
        self._pending_replica_plan = None

    def _clear_runtime(self) -> None:
        self._pending_replica_plan = None
        self._scheduler_group = None
        self._arena = None
        self._ready_flags = None
        self._bound_backend_ref = None
