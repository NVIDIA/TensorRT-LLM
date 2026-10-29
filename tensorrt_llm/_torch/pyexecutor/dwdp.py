# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""DwdpManager facade for VA-based DWDP.

This module owns four concerns and delegates the rest:
  * Lifecycle / global singleton (``__enter__`` / ``__exit__`` +
    ``set_global_dwdp_manager`` / ``get_global_dwdp_manager``)
  * DWDP group creation (``_create_dwdp_comm``): MPI sub-communicator or
    store rendezvous
  * Layer index registration (SSOT: ``add_layer`` → ``_registered_layers``)
    plus runtime entry points that forward to ``DWDPWeightManager``
    (``prefetch_first_layers``, ``wait_and_bind``,
    ``record_compute_and_prefetch_next``)
  * Dense DWDP ownership: the ``DenseDWDP`` runtime that ``setup(model)``
    commits, exposed as ``dense`` to the model's layer hooks, primed together
    with the expert prefetch (``prefetch_first_layers``, ``ensure_primed``)
    and released before the expert teardown

The heavy lifting — MNNVL handle allocation, composite VA layout,
double-buffer scheduling — lives in ``tensorrt_llm._torch.modules.dwdp``
and is wired up by ``setup_dwdp()`` and ``setup_dense_dwdp()`` during
``setup(model)``.
"""

from __future__ import annotations

import traceback
import weakref
from typing import List, Optional

import torch
import torch.nn as nn
from mpi4py.MPI import COMM_WORLD

from tensorrt_llm._torch.distributed import MPIDist
from tensorrt_llm._torch.modules.dwdp import (
    DenseDWDP,
    DWDPWeightManager,
    StoreDwdpGroup,
    dense_dwdp_requested,
    dwdp_global_rank,
    setup_dense_dwdp,
    setup_dwdp,
    store_rendezvous_enabled,
    teardown_dwdp,
)
from tensorrt_llm._torch.modules.dwdp.dense.setup import (
    dense_dwdp_active,
    dense_roots_available,
    refuse_dense_dwdp_without_roots,
)
from tensorrt_llm._utils import get_env_flag
from tensorrt_llm.llmapi.llm_args import DwdpConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

#: A/B switch: ``TRTLLM_DWDP_LEGACY_PREFETCH=1`` restores the previous one-layer expert
#: prefetch protocol (``wait_and_bind`` releases the other
#: slot, each layer prefetches the next one). Expert DWDP only; dense DWDP refuses it.
LEGACY_PREFETCH_ENV = "TRTLLM_DWDP_LEGACY_PREFETCH"


def legacy_prefetch_requested() -> bool:
    """Whether ``TRTLLM_DWDP_LEGACY_PREFETCH`` asks for the one-layer expert prefetch protocol."""
    return bool(get_env_flag(LEGACY_PREFETCH_ENV, False))


#: ``TRTLLM_DWDP_WRAP_PREFETCH=1`` turns on the cross-forward expert prefetch (off by default):
#: the last two MoE layers of a forward prefetch the first two MoE layers of the next forward
#: into the slots they release, so those gathers run in the forward's tail and the host gap
#: instead of at the start of the next forward. Two-layer expert protocol only (no dense, no
#: legacy).
WRAP_PREFETCH_ENV = "TRTLLM_DWDP_WRAP_PREFETCH"


def wrap_prefetch_requested() -> bool:
    """Whether ``TRTLLM_DWDP_WRAP_PREFETCH`` (default off) asks for the cross-forward prefetch."""
    return bool(get_env_flag(WRAP_PREFETCH_ENV, False))


_global_dwdp_manager: Optional["DwdpManager"] = None


def set_global_dwdp_manager(manager: Optional["DwdpManager"]) -> None:
    global _global_dwdp_manager
    _global_dwdp_manager = manager


def get_global_dwdp_manager() -> Optional["DwdpManager"]:
    return _global_dwdp_manager


class DwdpManager:
    """Lifecycle facade that plumbs the VA-based DWDP pipeline.

    Construction creates the DWDP MPI sub-communicator. ``add_layer`` is the
    Single Source of Truth for MoE layer indices — ``configurable_moe``
    registers each MoE layer during model construction, and ``setup(model)``
    passes that list to ``setup_dwdp()`` so ``collect_moe_params`` never has
    to re-discover layers by walking the model tree.

    After ``setup()`` returns, the runtime methods (``prefetch_first_layers``,
    ``wait_and_bind``, ``record_compute_and_prefetch_next``) forward to the
    ``DWDPWeightManager`` instance produced by ``setup_dwdp``.

    When dense DWDP is committed, ``dense`` returns its runtime; the model's
    decoder-layer hooks drive it and the model's forward calls
    ``ensure_primed``. Each forward then starts from
    ``L(first), E(first), L(second), E(second)`` in that copy-stream order,
    where ``L`` is a layer's dense gather and ``E`` its expert prefetch.
    """

    def __init__(
        self,
        config: DwdpConfig,
        dist: object,
        mapping: Mapping,
    ) -> None:
        if get_global_dwdp_manager() is not None:
            raise RuntimeError("DwdpManager already registered globally")
        self._legacy_prefetch = legacy_prefetch_requested()
        if self._legacy_prefetch and dense_dwdp_requested():
            # Before any collective, so every rank of a uniform launch refuses alike.
            raise RuntimeError(
                f"{LEGACY_PREFETCH_ENV}=1 (the one-layer expert prefetch) cannot run with "
                "dense DWDP (TRTLLM_DWDP_DENSE=1), which orders its copies around the "
                "two-layer protocol; unset one of them"
            )
        if self._legacy_prefetch:
            logger.info(
                f"[DwdpManager] {LEGACY_PREFETCH_ENV}=1: one-layer expert prefetch "
                "(wait_and_bind releases the other slot, each layer prefetches the next)"
            )
        # Dense DWDP primes its own slots per forward, so the wrap is also off once
        # ``setup`` commits a dense runtime (checked per call in ``_wrap_target``).
        self._wrap_prefetch = wrap_prefetch_requested() and not self._legacy_prefetch
        logger.info(
            f"[DwdpManager] {WRAP_PREFETCH_ENV}: cross-forward expert prefetch "
            f"{'on' if self._wrap_prefetch else 'off'}"
        )
        # Expert-only DWDP: True from a forward's first ``wait_and_bind`` until its last MoE
        # layer's ``record_compute_and_prefetch_next``. Still True at the next priming means the
        # forward stopped in between (see ``prefetch_first_layers``). Dense DWDP tracks its own.
        self._expert_forward_open = False
        # Plain `trtllm-serve` workers have size-1 MPI worlds, so the group
        # rendezvous over a TCPStore instead -- see modules.dwdp.rendezvous.
        self._store_rendezvous = store_rendezvous_enabled()
        if not self._store_rendezvous and not isinstance(dist, MPIDist):
            raise RuntimeError(
                "DWDP requires the MPI backend (MPIDist), or a store "
                "rendezvous: set TRTLLM_DWDP_RANK and "
                "TRTLLM_DWDP_MASTER_ADDR when launching the context "
                "workers with plain trtllm-serve"
            )
        if not mapping.dwdp_enabled:
            raise RuntimeError(
                f"DwdpManager requires mapping.dwdp_enabled (dwdp_size > 1); "
                f"got dwdp_size={mapping.dwdp_size}"
            )

        # Validate DwdpConfig topology against the actual MPI launch.
        # `num_groups` declares the user's intended topology — that the launch
        # contains `num_groups × dwdp_size` CTX worker ranks, partitioned into
        # `num_groups` independent DWDP groups of `dwdp_size` ranks each.
        # Without this validation `num_groups` is a dead schema field, which
        # is how it was treated up through commit `1fbc0d49`.
        if config.num_groups <= 0:
            raise ValueError(f"DwdpConfig.num_groups must be positive, got {config.num_groups}")
        self.rank = dwdp_global_rank()
        group_id = self.rank // mapping.dwdp_size
        if group_id >= config.num_groups:
            raise ValueError(
                f"DwdpManager rank {self.rank} computes group_id={group_id} "
                f"(rank // dwdp_size={mapping.dwdp_size}) which is >= "
                f"DwdpConfig.num_groups ({config.num_groups}). The launch "
                f"started more CTX workers than fit in {config.num_groups} × "
                f"{mapping.dwdp_size} = {config.num_groups * mapping.dwdp_size} "
                f"ranks, or num_groups in the YAML is set too small."
            )
        expected_min_world = config.num_groups * mapping.dwdp_size
        # A store-rendezvous worker's COMM_WORLD is size 1 and says nothing about
        # the launch; a short launch surfaces as the first allgather timing out.
        actual_world = 0 if self._store_rendezvous else COMM_WORLD.Get_size()
        if actual_world and actual_world < expected_min_world:
            raise ValueError(
                f"DwdpConfig declares {config.num_groups} groups × dwdp_size "
                f"{mapping.dwdp_size} = {expected_min_world} CTX workers, but "
                f"MPI world size is only {actual_world}. Either the launch's "
                f"--ntasks is too small, or num_groups is set too high."
            )

        self.config = config
        self.dist = dist
        self._mapping = mapping
        self.dwdp_size = mapping.dwdp_size
        self.dwdp_rank = mapping.dwdp_rank
        self.num_experts_per_worker = config.num_experts_per_worker
        self.num_groups = config.num_groups
        self.num_prefetch_experts = config.num_prefetch_experts

        # Per-rank expert range. ``num_prefetch_experts`` is the rank-to-rank
        # stride and ``num_experts_per_worker`` is the (uniform across ranks)
        # storage chunk size. When the user requests redundancy
        # (``num_prefetch_experts < num_experts_per_worker``) consecutive
        # ranks' ranges overlap; when the partition is non-uniform (e.g.
        # dwdp_size = 3 over 256 experts) the tail rank's
        # ``[start, end)`` runs past ``num_experts`` and is later capped at
        # weight-load time. ``_init_dwdp_expert_layout`` reads
        # ``start_expert_id`` to install the correct ``slot_start`` /
        # ``slot_end`` on every fused MoE backend before ``create_weights``.
        self.start_expert_id = self.dwdp_rank * self.num_prefetch_experts
        self.end_expert_id = self.start_expert_id + self.num_experts_per_worker

        self.dwdp_group = self._create_dwdp_comm()

        # SSOT for MoE layer indices; populated by configurable_moe.add_layer()
        self._registered_layers: List[int] = []

        # Set by setup(model); None until then
        self._weight_manager: Optional[DWDPWeightManager] = None
        self._model_ref: Optional[weakref.ReferenceType[nn.Module]] = None
        # The committed dense DWDP runtime; None when dense DWDP is off, before
        # setup(model) and after cleanup()
        self._dense: DenseDWDP | None = None

    # ------------------------------------------------------------------
    # DWDP MPI group
    # ------------------------------------------------------------------

    def _create_dwdp_comm(self):
        """Create the group handle scoped to this rank's DWDP group.

        With num_groups=2, dwdp_size=4:
            Group 0: ranks [0, 1, 2, 3]
            Group 1: ranks [4, 5, 6, 7]

        Returns an MPI sub-communicator or a ``StoreDwdpGroup``; both expose
        ``allgather`` / ``Barrier`` / ``Free``.
        """
        if self._store_rendezvous:
            return StoreDwdpGroup.from_env(global_rank=self.rank, dwdp_size=self.dwdp_size)

        group_id = self.rank // self.dwdp_size
        group_start = group_id * self.dwdp_size
        ranks = list(range(group_start, group_start + self.dwdp_size))
        return COMM_WORLD.Create_group(COMM_WORLD.group.Incl(ranks))

    # ------------------------------------------------------------------
    # Global singleton lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> "DwdpManager":
        if get_global_dwdp_manager() is not None:
            raise RuntimeError("DwdpManager already registered globally")
        set_global_dwdp_manager(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        try:
            self.cleanup()
        finally:
            set_global_dwdp_manager(None)
        return False

    def is_enabled(self) -> bool:
        return self.dwdp_size > 1

    @property
    def dense(self) -> DenseDWDP | None:
        """The committed dense DWDP runtime, or None when dense DWDP is not active.

        None when dense DWDP is off or was abandoned, before ``setup(model)``
        commits it and after ``cleanup()``. The model's layer hooks run only
        while it is not None.
        """
        return self._dense

    def cleanup(self) -> None:
        """Release dense DWDP, tear down the weight manager, free the DWDP group.

        Dense DWDP goes first: its release synchronizes the device, repoints the
        model's dense weights to zero-size tensors and frees the staging slots
        and the row store. The model keeps its ``_dense_dwdp_active`` mark, so
        weight reloads stay refused. If the model was garbage collected first,
        the bindings of its modules count as detached, since nothing can read
        them any more. Every step runs even when an earlier one fails; the
        first error is raised at the end and later ones are logged. Idempotent.
        """
        cleanup_error = None
        cleanup_traceback = None
        try:
            model = self._model_ref() if self._model_ref is not None else None
            dense, self._dense = self._dense, None
            if dense is not None:
                try:
                    dense.release()
                except BaseException as exc:  # the expert teardown below must still run
                    cleanup_error, cleanup_traceback = exc, exc.__traceback__
                    logger.error(
                        "[DwdpManager] Dense DWDP release failed; tearing down expert "
                        f"DWDP anyway\n{traceback.format_exc()}"
                    )
            if model is not None and self._weight_manager is not None:
                teardown_dwdp(model, expected_manager=self._weight_manager)
            elif self._weight_manager is not None:
                self._weight_manager.release()
        except BaseException as exc:
            if cleanup_error is None:
                cleanup_error, cleanup_traceback = exc, exc.__traceback__
            else:
                logger.error(
                    "[DwdpManager] Expert DWDP teardown failed after the dense DWDP "
                    f"release error\n{traceback.format_exc()}"
                )
        finally:
            self._weight_manager = None
            self._model_ref = None
            if self.dwdp_group is not None:
                try:
                    self.dwdp_group.Free()
                except BaseException:
                    if cleanup_error is None:
                        raise
                    logger.error(
                        "[DwdpManager] Failed to free DWDP group after runtime cleanup "
                        f"error\n{traceback.format_exc()}"
                    )
                finally:
                    self.dwdp_group = None

        if cleanup_error is not None:
            raise cleanup_error.with_traceback(cleanup_traceback)

    # ------------------------------------------------------------------
    # Layer registration (SSOT) + setup
    # ------------------------------------------------------------------

    def add_layer(self, layer_idx: int) -> None:
        """Register an MoE layer index. Called from configurable_moe.__init__."""
        if layer_idx in self._registered_layers:
            logger.warning(f"[DwdpManager] Layer {layer_idx} already registered; ignoring")
            return
        self._registered_layers.append(layer_idx)

    def setup(self, model: nn.Module) -> Optional[DWDPWeightManager]:
        """Run the VA initialization pipeline on ``model``.

        Delegates to ``setup_dwdp`` (transport + weight buffer + weight
        manager + MoE backend fixup). Stores the returned ``DWDPWeightManager``
        for subsequent runtime calls.

        Then, if the model's decoder layers provide dense roots, runs
        ``setup_dense_dwdp`` on every rank, also on a rank that did not ask
        for dense DWDP, so the first dense collective gives every rank of the
        group the same verdict when ranks disagree on ``TRTLLM_DWDP_DENSE``.
        The committed runtime, if any, becomes ``dense``; dense setup errors
        propagate. Whether a model has roots depends only on its code, so on
        a model without them no rank makes a dense collective: a rank that
        asks for dense DWDP there decides alone, at once
        (``refuse_dense_dwdp_without_roots``: it raises unless its
        ``TRTLLM_DWDP_DENSE_REQUIRED`` is off).

        Called once from ``py_executor_creator`` after weight loading. Once
        dense DWDP is committed, a later call for the same model returns the
        weight manager and keeps the runtime, making no collective; every
        rank committed together, so every rank returns early alike.

        Raises:
            RuntimeError: Called after ``cleanup()``; dense DWDP is committed
                for another model; or the model's dense weights were moved
                by a dense DWDP runtime this manager does not hold (it was
                released, so the modules hold zero-size tensors). Each is
                raised before any expert or dense setup work.
            DenseSetupError: Dense DWDP setup failed, or was requested on a
                model without dense roots, with ``TRTLLM_DWDP_DENSE_REQUIRED``
                on.
        """
        if self.dwdp_group is None:
            raise RuntimeError("DwdpManager.setup() called after cleanup()")
        if self._dense is not None:
            if self._model_ref is None or self._model_ref() is not model:
                raise RuntimeError(
                    "DwdpManager.setup(): dense DWDP is already committed for another "
                    "model; a manager serves one model"
                )
            return self._weight_manager
        if dense_dwdp_active(model):
            raise RuntimeError(
                "DwdpManager.setup(): dense DWDP already moved this model's dense "
                "weights and its runtime was released, so the modules hold zero-size "
                "tensors; load the model again to set it up"
            )
        layer_indices = sorted(self._registered_layers)
        device_id = torch.cuda.current_device()
        self._weight_manager = setup_dwdp(
            model=model,
            mapping=self._mapping,
            device_id=device_id,
            comm=self.dwdp_group,
            layer_indices=layer_indices,
            num_experts_per_worker=self.num_experts_per_worker,
            num_prefetch_experts=self.num_prefetch_experts,
            contention_opt=self.config.contention_opt,
        )
        self._model_ref = weakref.ref(model)
        if dense_roots_available(model):
            self._dense = setup_dense_dwdp(
                model,
                self._weight_manager,
                self.dwdp_group,
                dwdp_rank=self.dwdp_rank,
                dwdp_size=self.dwdp_size,
                device_id=device_id,
                moe_layers=layer_indices,
            )
        elif dense_dwdp_requested():
            refuse_dense_dwdp_without_roots(model)
        return self._weight_manager

    # ------------------------------------------------------------------
    # Runtime entry points (forwarded to DWDPWeightManager)
    # ------------------------------------------------------------------

    def prefetch_first_layers(self) -> None:
        """Executor-loop priming of the first two MoE layers before a forward.

        Without dense DWDP, prefetches the experts of the first two MoE layers.
        If the previous forward stopped between its first and last MoE layer, a
        kernel it enqueued may still read a slot whose consume event it never
        recorded; both slots' consume events are then recorded on the current
        (execution) stream first, so these copies wait for every read that
        forward enqueued. With dense DWDP, enqueues ``L(first), E(first),
        L(second), E(second)`` unless the slots are already primed, or a forward
        is still open, which means the previous forward aborted: the model's
        ``ensure_primed`` then recovers on the execution stream.
        """
        if self._weight_manager is None:
            raise RuntimeError("DwdpManager.setup() has not been called yet")
        dense = self._dense
        if dense is None:
            weight_manager = self._weight_manager
            first = weight_manager.first_moe_layer()
            second = weight_manager.next_moe_layer(first)
            if self._expert_forward_open:
                logger.warning(
                    "[DwdpManager] the previous forward stopped before its last MoE layer; "
                    "recording a WAR barrier on both expert slots before priming"
                )
                weight_manager.record_compute(first)
                if second is not None:
                    weight_manager.record_compute(second)
                self._expert_forward_open = False
            weight_manager.prefetch_layer(first)
            if second is not None:
                weight_manager.prefetch_layer(second)
            return
        if dense.forward_open or dense.primed:
            return
        self._prime_all(dense, self._weight_manager)

    def ensure_primed(self) -> None:
        """Model-side priming (H0): called at the model's forward entry when ``dense`` is set.

        Runs on the execution stream, so it also covers the forwards the
        executor loop does not prime (warmup, autotuning, KV-cache estimation).
        It refuses a CUDA graph capture and multi-stream execution
        (``DenseDWDP.require_eager``) before it records or enqueues anything,
        so a refused forward leaves the runtime as it found it. Then, if the
        previous forward aborted between its first and last layer, it records
        a WAR barrier on this stream: every dense consume event, then the
        consume events of both expert slots, so the next copies wait for every
        read the aborted forward enqueued. Then it enqueues
        ``L(first), E(first), L(second), E(second)`` unless already primed.

        Raises:
            RuntimeError: Dense DWDP is not active.
            DenseProtocolError: The current stream is capturing a CUDA graph,
                or multi-stream execution is on.
        """
        dense = self._dense
        weight_manager = self._weight_manager
        if dense is None or weight_manager is None:
            raise RuntimeError("DwdpManager.ensure_primed() needs an active dense DWDP runtime")
        dense.require_eager("DwdpManager.ensure_primed (H0)")
        if dense.forward_open:
            dense.war_barrier()
            first = weight_manager.first_moe_layer()
            weight_manager.record_compute(first)
            second = weight_manager.next_moe_layer(first)
            if second is not None:
                weight_manager.record_compute(second)
        if not dense.primed:
            self._prime_all(dense, weight_manager)

    @staticmethod
    def _prime_all(dense: DenseDWDP, weight_manager: DWDPWeightManager) -> None:
        """Enqueue the first two layers in deadline order, then mark the dense slots primed.

        ``L(l)`` goes before ``E(l)``: the copy stream runs them in enqueue
        order, and layer ``l`` reads ``L(l)`` (attention) before ``E(l)``
        (experts).
        """
        first = weight_manager.first_moe_layer()
        for layer in (first, weight_manager.next_moe_layer(first)):
            if layer is None:
                break
            dense.prefetch(layer)
            weight_manager.prefetch_layer(layer)
        dense.mark_primed()

    def wait_and_bind(self, backend_module: nn.Module, layer_idx: int) -> None:
        """Wait for prefetch and bind full-tensor views onto backend weights."""
        if self._weight_manager is None:
            raise RuntimeError("DwdpManager.setup() has not been called yet")
        self._weight_manager.wait_and_bind(backend_module, layer_idx)
        if self._dense is None:
            self._expert_forward_open = True
        if self._legacy_prefetch:
            self._weight_manager.record_other_slot_consumed(layer_idx)

    def record_compute_and_prefetch_next(self, layer_idx: int) -> None:
        """Record compute and maintain the two-layer prefetch window.

        ``prefetch_first_layers`` warms the first two MoE layers. After each
        layer's kernel has been enqueued, record consumption of its current
        buffer slot and prefetch the layer two positions ahead. This avoids
        duplicate prefetch of the second layer and prevents reuse of a slot
        before the current kernel has finished reading it.

        With ``TRTLLM_DWDP_LEGACY_PREFETCH=1`` it prefetches the next MoE layer
        instead, into the slot ``wait_and_bind`` released, as before #18644. The
        last MoE layer still records its own slot, so the next forward's priming
        copies wait for its kernel; the old protocol left that slot unsignalled.
        The weight manager skips a layer that its slot already holds, so the
        legacy protocol no longer gathers the second MoE layer twice.

        The last two MoE layers have no layer two positions ahead. With
        ``TRTLLM_DWDP_WRAP_PREFETCH=1`` (and no dense DWDP), they prefetch the
        first two MoE layers of the next forward instead (see ``_wrap_target``);
        the next ``prefetch_first_layers`` then finds them resident and enqueues
        nothing.
        """
        if self._weight_manager is None:
            raise RuntimeError("DwdpManager.setup() has not been called yet")
        if self._legacy_prefetch:
            next_idx = self._weight_manager.next_moe_layer(layer_idx)
            if next_idx is not None:
                self._weight_manager.prefetch_layer(next_idx)
            else:
                self._weight_manager.record_compute(layer_idx)
                self._expert_forward_open = False
            return
        self._weight_manager.record_compute(layer_idx)
        next_idx = self._weight_manager.next_moe_layer(layer_idx)
        if next_idx is None:
            self._expert_forward_open = False
        next_next_idx = None
        if next_idx is not None:
            next_next_idx = self._weight_manager.next_moe_layer(next_idx)
        if next_next_idx is None:
            next_next_idx = self._wrap_target(layer_idx, next_idx)
        if next_next_idx is not None:
            self._weight_manager.prefetch_layer(next_next_idx)

    def _wrap_target(self, layer_idx: int, next_idx: Optional[int]) -> Optional[int]:
        """Next forward's MoE layer for the slot that ``layer_idx`` just released, or None.

        Two positions ahead in cyclic MoE order: the second-to-last MoE layer
        maps to the first, the last to the second. The target must use the same
        slot as ``layer_idx`` (whose consume event was just recorded): with an
        odd number of MoE layers it does not, and the other slot still holds a
        layer this forward has not consumed, so nothing is prefetched.
        """
        if not self._wrap_prefetch or self._dense is not None:
            return None
        weight_manager = self._weight_manager
        first = weight_manager.first_moe_layer()
        target = first if next_idx is not None else weight_manager.next_moe_layer(first)
        if target is None or target == layer_idx:
            return None
        if weight_manager.slot_index(target) != weight_manager.slot_index(layer_idx):
            return None
        return target
