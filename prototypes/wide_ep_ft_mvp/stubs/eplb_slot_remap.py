# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""1b.1-3 (stub): EPLB ``reconfigure_mask_only`` for 1-2 layers.

Production PRs 1b.1-3 propagate the failed-rank mask through ``MoeLoadBalancer``,
rewrite ``MoePlacementInfo`` for **all 58 layers** of DeepSeek-V3 within a
< 10 ms budget, and coordinate with the EPLB worker thread's pause/resume.
This stub does the absolute minimum:

  * Reconfigures only the layers explicitly registered with
    ``register_layer_placement``.
  * Zeros out the dead-rank slot in each registered layer's placement table
    in-place. No replication-aware slot reassignment, no weight migration,
    no rebalance — the assumption is replication ≥ 2 so every expert has a
    surviving copy somewhere else and EPLB doesn't need to migrate anything
    at recovery time.
  * Synchronous: caller blocks until all layers' placements have been
    rewritten. No EPLB-worker pause/resume coordination (the prototype just
    avoids running EPLB-worker stride during a kill, controlled from the
    test driver).

Stub contract — exercises these seams:

  * **Iteration-boundary hook → reconfigure_mask_only(failed_ranks)** — the
    contract that 1c.4 calls into.
  * **Per-layer placement update mechanics** — same operation shape as
    production, just on N layers instead of 58.
  * **Idempotency** — calling twice with the same failed-rank set is a no-op
    on the second call. Real production hook will rely on this so
    iteration-boundary checks can be conservative without thrash.

Stub deliberately omits, vs. production:

  * All 58 layers / < 10 ms budget validation.
  * Replication-aware slot reassignment / weight migration.
  * Coordination with EPLB worker thread (pause/resume).
  * Thread-safety beyond the per-layer lock.
  * Telemetry (time-to-reconfigure, per-layer breakdown).
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LayerPlacement:
    """Per-layer placement info: which (rank, slot) holds each expert.

    The production ``MoePlacementInfo`` is much richer (slot counts, expert
    weights, routing metadata). This stub keeps just the rank-of-each-slot
    table because that's the only field ``reconfigure_mask_only`` actually
    rewrites.

    ``slot_to_rank[i] == -1`` means "this slot is dead, route around it".
    """

    layer_index: int
    slot_count: int
    slot_to_rank: list[int]  # length == slot_count
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class EplbSlotRemapStub:
    """In-process registry of layer placements + the ``reconfigure_mask_only`` op.

    Args:
        ep_size: Number of ranks in the EP group.
    """

    def __init__(self, ep_size: int) -> None:
        if ep_size <= 0:
            raise ValueError(f"ep_size must be > 0, got {ep_size}")
        self._ep_size = ep_size
        self._layers: dict[int, LayerPlacement] = {}
        self._registry_lock = threading.Lock()
        self._last_reconfigure_failed_ranks: frozenset[int] = frozenset()
        self._last_reconfigure_at: Optional[float] = None
        self._reconfigure_count = 0

    def register_layer_placement(self, placement: LayerPlacement) -> None:
        """Register a layer's placement table. Call once per layer at startup."""
        if not 0 <= placement.layer_index:
            raise ValueError(f"layer_index must be >= 0, got {placement.layer_index}")
        if placement.slot_count != len(placement.slot_to_rank):
            raise ValueError(
                f"slot_count {placement.slot_count} != len(slot_to_rank) "
                f"{len(placement.slot_to_rank)}"
            )
        for r in placement.slot_to_rank:
            if not -1 <= r < self._ep_size:
                raise ValueError(f"slot rank {r} not in [-1, {self._ep_size})")
        with self._registry_lock:
            if placement.layer_index in self._layers:
                raise ValueError(
                    f"layer {placement.layer_index} already registered; "
                    f"the prototype assumes one registration per layer at startup"
                )
            self._layers[placement.layer_index] = placement

    def reconfigure_mask_only(self, failed_ranks: Iterable[int]) -> ReconfigureResult:
        """Zero out the dead-rank slots in every registered layer.

        Idempotent: if ``failed_ranks`` matches the previous call, this is a
        no-op (the production version uses the EPGroupHealth generation counter
        for the same purpose; the stub just compares the set directly).

        Returns:
            ``ReconfigureResult`` with timing + per-layer slot-affected counts
            for the timeline JSON.
        """
        failed = frozenset(failed_ranks)
        for r in failed:
            if not 0 <= r < self._ep_size:
                raise ValueError(f"failed rank {r} not in [0, {self._ep_size})")

        if failed == self._last_reconfigure_failed_ranks:
            return ReconfigureResult(
                duration_sec=0.0,
                slots_zeroed_per_layer={},
                layers_touched=0,
                was_no_op=True,
            )

        t0 = time.monotonic()
        slots_zeroed_per_layer: dict[int, int] = {}
        with self._registry_lock:
            layer_items = list(self._layers.items())
        for layer_index, placement in layer_items:
            n = 0
            with placement._lock:
                for slot in range(placement.slot_count):
                    if placement.slot_to_rank[slot] in failed:
                        if placement.slot_to_rank[slot] != -1:
                            placement.slot_to_rank[slot] = -1
                            n += 1
            if n > 0:
                slots_zeroed_per_layer[layer_index] = n
        t1 = time.monotonic()

        self._last_reconfigure_failed_ranks = failed
        self._last_reconfigure_at = t1
        self._reconfigure_count += 1
        return ReconfigureResult(
            duration_sec=t1 - t0,
            slots_zeroed_per_layer=slots_zeroed_per_layer,
            layers_touched=len(slots_zeroed_per_layer),
            was_no_op=False,
        )

    def get_layer_placement(self, layer_index: int) -> LayerPlacement:
        with self._registry_lock:
            return self._layers[layer_index]

    def reconfigure_count(self) -> int:
        """How many *effective* (non-no-op) reconfigures have run."""
        return self._reconfigure_count


@dataclass(frozen=True)
class ReconfigureResult:
    """Returned by ``reconfigure_mask_only`` for the timeline JSON."""

    duration_sec: float
    slots_zeroed_per_layer: dict[int, int]
    layers_touched: int
    was_no_op: bool
