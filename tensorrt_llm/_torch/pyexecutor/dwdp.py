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
  * DWDP group creation (``_create_dwdp_comm``): an MPI sub-communicator, or
    a TCPStore rendezvous for independently launched workers
  * Layer index registration (SSOT: ``add_layer`` → ``_registered_layers``)
    plus runtime entry points that forward to ``DWDPWeightManager``
    (``start_forward``, ``wait_and_bind``,
    ``finish_layer``, ``invalidate_prefetch``)
  * Attached dense DWDP: the ``DenseDwdpManager`` created and registered
    with this manager, through which the model's layer hooks reach the
    ``DenseDWDP`` runtime. ``setup(model)`` commits the runtime over the DWDP
    group before the expert setup allocates its prefetch ring, so the
    replicated dense weights are freed first, then hands it to the dense
    manager together with the weight manager, which runs on the same copy
    stream; each forward starts the dense prefetch ahead of the expert
    prefetch (``start_forward``, ``DenseDwdpManager.ensure_primed``), and
    ``cleanup`` releases dense DWDP before the expert teardown

The heavy lifting — MNNVL handle allocation, composite VA layout,
prefetch scheduling — lives in ``tensorrt_llm._torch.modules.dwdp``
and is wired up by ``setup_dwdp()`` and ``setup_dense_dwdp()`` during
``setup(model)``.
"""

from __future__ import annotations

import traceback
import weakref
from types import TracebackType
from typing import List, Optional

import torch
import torch.nn as nn
from mpi4py.MPI import COMM_WORLD, Intracomm

from tensorrt_llm._torch.cute_dsl_utils import set_sm_reserve
from tensorrt_llm._torch.distributed import MPIDist
from tensorrt_llm._torch.modules.dwdp import (
    DenseDWDP,
    DWDPWeightManager,
    StoreDwdpGroup,
    dwdp_global_rank,
    fabric_gather,
    setup_dense_dwdp,
    setup_dwdp,
    store_rendezvous_enabled,
    teardown_dwdp,
)
from tensorrt_llm._torch.modules.dwdp.dense.config import DenseConfig
from tensorrt_llm._torch.modules.dwdp.dense.manager import (
    DenseDwdpManager,
    get_global_dense_dwdp_manager,
)
from tensorrt_llm._torch.modules.dwdp.dense.setup import (
    dense_dwdp_active,
    dense_roots_available,
    refuse_dense_dwdp_without_roots,
)
from tensorrt_llm._torch.modules.dwdp.dense.types import DenseMode
from tensorrt_llm._torch.modules.dwdp.setup import _release_during_rollback
from tensorrt_llm._torch.modules.dwdp.weight_manager import expert_transfer
from tensorrt_llm.llmapi.llm_args import DwdpConfig
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping

_global_dwdp_manager: Optional["DwdpManager"] = None


def set_global_dwdp_manager(manager: Optional["DwdpManager"]) -> None:
    global _global_dwdp_manager
    _global_dwdp_manager = manager


def get_global_dwdp_manager() -> Optional["DwdpManager"]:
    return _global_dwdp_manager


def new_expert_copy_stream(device_id: int) -> torch.cuda.Stream:
    """The expert DWDP copy stream on ``cuda:<device_id>``, made as ``DWDPWeightManager`` makes it.

    ``DwdpManager.setup`` creates it for attached dense DWDP, which runs before the weight manager
    exists, and hands it to ``setup_dwdp``.  The test seam: tests monkeypatch it.
    """
    return torch.cuda.Stream(device=torch.device("cuda", device_id))


class _AttachedTransfer:
    """The transfer of attached dense DWDP (``DenseTransfer``), before the weight manager exists.

    Dense DWDP commits before ``setup_dwdp`` allocates the expert prefetch ring, so it cannot take
    the expert weight manager as its transfer.  This stands in with what the weight manager will
    use:

    * ``copy_stream``: made by ``new_expert_copy_stream`` at the first read, which dense setup
      makes in P2, inside the phase's local work (a failure is this rank's failed P2 report);
      ``DwdpManager.setup`` then hands it to ``setup_dwdp``.
    * ``use_gather`` and ``use_batched_prefetch``: computed when read, as
      ``DWDPWeightManager.__init__`` computes them (``expert_transfer`` of ``copy_mode`` and
      ``contention_opt``).

    Construction touches no CUDA, probes nothing and logs nothing.
    """

    __slots__ = ("_device_id", "_contention_opt", "_copy_mode", "_stream")

    def __init__(
        self, device_id: int, *, contention_opt: bool, copy_mode: Optional[str] = None
    ) -> None:
        self._device_id = device_id
        self._contention_opt = contention_opt
        self._copy_mode = copy_mode
        self._stream: Optional[torch.cuda.Stream] = None

    @property
    def copy_stream(self) -> torch.cuda.Stream:
        """The copy stream: created on the first read, the same object on every later one."""
        if self._stream is None:
            self._stream = new_expert_copy_stream(self._device_id)
        return self._stream

    @property
    def created_stream(self) -> Optional[torch.cuda.Stream]:
        """The copy stream if a read created it, else None."""
        return self._stream

    @property
    def use_gather(self) -> bool:
        """Whether the expert prefetch uses the SM fabric gather."""
        return expert_transfer(self._copy_mode, self._contention_opt)[0]

    @property
    def use_batched_prefetch(self) -> bool:
        """Whether the expert prefetch uses one batched copy per layer."""
        return expert_transfer(self._copy_mode, self._contention_opt)[1]


def _check_shared_transfer(
    transfer: _AttachedTransfer | DWDPWeightManager, weight_manager: DWDPWeightManager
) -> None:
    """The weight manager must run on the copy stream and transfer dense DWDP was set up with.

    Dense and expert copies share one copy stream, which runs them in the order they are enqueued,
    and the dense copies follow the expert transfer.  On a repeated setup dense DWDP ran on the
    weight manager itself, which the expert setup must then have returned.  Nothing to check when
    dense setup never read the attached transfer's stream.

    Raises:
        RuntimeError: The weight manager runs on another stream or with other transfer flags.
    """
    if transfer is weight_manager:
        return
    if isinstance(transfer, _AttachedTransfer):
        stream = transfer.created_stream
        if stream is None:
            return
    else:
        stream = transfer.copy_stream
    dense_flags = (transfer.use_gather, transfer.use_batched_prefetch)
    expert_flags = (weight_manager.use_gather, weight_manager.use_batched_prefetch)
    if weight_manager.copy_stream is not stream or expert_flags != dense_flags:
        raise RuntimeError(
            "DwdpManager.setup(): the expert weight manager does not run on the copy stream and "
            "transfer dense DWDP was set up with; the dense and the expert copies must share one "
            f"copy stream (expert: {weight_manager.copy_stream}, (use_gather, "
            f"use_batched_prefetch)={expert_flags}; dense: {stream}, {dense_flags})"
        )


def _dense_slot_storages(dense: DenseDWDP | None) -> frozenset[int]:
    """The storages of a committed dense runtime's slots, which the expert transport must not take.

    After the commit the modules' dense tensors are views of these slots.  Empty without a
    runtime.
    """
    if dense is None:
        return frozenset()
    return frozenset(
        slot.storage.untyped_storage().data_ptr() for ring in dense.slots.values() for slot in ring
    )


class DwdpManager:
    """Lifecycle facade that plumbs the VA-based DWDP pipeline.

    Construction creates the DWDP group: an MPI sub-communicator, or a
    ``StoreDwdpGroup`` when the context workers are launched independently
    (see ``modules.dwdp.rendezvous``), and the attached dense DWDP manager
    (``dense_manager``); ``__enter__`` registers both globally.
    ``add_layer`` is the Single Source of Truth for MoE layer indices —
    ``configurable_moe`` registers each MoE layer during model construction,
    and ``setup(model)`` passes that list to ``setup_dwdp()`` so
    ``collect_moe_params`` never has to re-discover layers by walking the
    model tree.

    After ``setup()`` returns, the runtime methods (``start_forward``,
    ``wait_and_bind``, ``finish_layer``) forward to the
    ``DWDPWeightManager`` instance produced by ``setup_dwdp``.

    When dense DWDP is committed, the dense manager owns its runtime
    (``dense``) and holds the weight manager: the model's decoder-layer hooks
    drive the runtime and the model's forward calls
    ``DenseDwdpManager.ensure_primed``. ``L`` is a layer's dense gather and
    ``E`` its expert prefetch, both on the weight manager's copy stream. A
    completed forward leaves ``L`` and ``E`` of the first two layers enqueued
    for the next one; otherwise ``start_forward`` or ``ensure_primed``
    enqueues ``L(first), L(second)``, then the expert prefetch its
    ``E(first), E(second)``. Without a committed runtime the dense manager
    does nothing. ``setup`` commits dense DWDP before the expert setup
    allocates its prefetch ring, on the copy stream the weight manager then
    adopts.
    """

    def __init__(
        self,
        config: DwdpConfig,
        dist: object,
        mapping: Mapping,
    ) -> None:
        if get_global_dwdp_manager() is not None:
            raise RuntimeError("DwdpManager already registered globally")
        # __enter__ registers the attached dense DWDP manager as well; refuse a
        # registered one here, before the DWDP group exists.
        if get_global_dense_dwdp_manager() is not None:
            raise RuntimeError("DenseDwdpManager already registered globally")
        # Independently launched workers each have their own MPI world, so the
        # group rendezvous over a TCPStore instead of COMM_WORLD.
        self._store_rendezvous = store_rendezvous_enabled()
        if not self._store_rendezvous and not isinstance(dist, MPIDist):
            raise RuntimeError(
                "DWDP requires the MPI backend (MPIDist), or a store "
                "rendezvous: set TRTLLM_DWDP_RANK and TRTLLM_DWDP_MASTER_ADDR "
                "when the context workers are launched independently"
            )
        if not mapping.dwdp_enabled:
            raise RuntimeError(
                f"DwdpManager requires mapping.dwdp_enabled (dwdp_size > 1); "
                f"got dwdp_size={mapping.dwdp_size}"
            )

        # Validate DwdpConfig topology against the actual launch.
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
        # Under a store rendezvous COMM_WORLD spans only this worker and says
        # nothing about the launch; the group_id check above bounds the rank,
        # and a missing peer surfaces as the first store allgather timing out.
        actual_world = None if self._store_rendezvous else COMM_WORLD.Get_size()
        if actual_world is not None and actual_world < expected_min_world:
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

        # The expert transfer: an explicit copy_mode overrides contention_opt and
        # TRTLLM_DWDP_PREFETCH_GATHER (a copy_mode gather without its op raises
        # here, before the DWDP group exists).
        use_gather, _ = expert_transfer(config.copy_mode, config.contention_opt)
        gather_env = fabric_gather.env_request()
        if config.copy_mode is not None and gather_env is not None and gather_env != use_gather:
            logger.warning(
                f"[DwdpManager] dwdp_config.copy_mode={config.copy_mode!r} overrides "
                f"TRTLLM_DWDP_PREFETCH_GATHER, which asks for the gather "
                f"{'on' if gather_env else 'off'}"
            )
        # SMs the expert gather holds for the whole step; persistent grids leave
        # them out (``set_sm_reserve`` on ``__enter__``, before the model is built).
        self.resident_sms = fabric_gather.resident_sms(gather=use_gather)

        self.dwdp_group = self._create_dwdp_comm()

        # SSOT for MoE layer indices; populated by configurable_moe.add_layer()
        self._registered_layers: List[int] = []
        # Layers that run several times in a row per pass; see add_repeated_layer()
        self._repeated_layers: List[int] = []

        # Set by setup(model); None until then
        self._weight_manager: Optional[DWDPWeightManager] = None
        self._model_ref: Optional[weakref.ReferenceType[nn.Module]] = None
        # Owns the committed dense DWDP runtime. Created whether or not
        # dwdp_config.enable_dense is set: the model's layers capture it when
        # they are built, and without a runtime it does nothing.
        self._dense_manager = DenseDwdpManager(DenseMode.ATTACHED)

    # ------------------------------------------------------------------
    # DWDP group
    # ------------------------------------------------------------------

    def _create_dwdp_comm(self) -> Intracomm | StoreDwdpGroup:
        """Create the group handle scoped to this rank's DWDP group.

        With num_groups=2, dwdp_size=4:
            Group 0: ranks [0, 1, 2, 3]
            Group 1: ranks [4, 5, 6, 7]

        Returns an MPI sub-communicator, or a ``StoreDwdpGroup`` under a store
        rendezvous; both expose ``allgather`` / ``Barrier`` / ``Free``.
        """
        if self._store_rendezvous:
            return StoreDwdpGroup.from_env(
                global_rank=self.rank,
                dwdp_size=self.dwdp_size,
                num_groups=self.config.num_groups,
            )

        group_id = self.rank // self.dwdp_size
        group_start = group_id * self.dwdp_size
        ranks = list(range(group_start, group_start + self.dwdp_size))
        return COMM_WORLD.Create_group(COMM_WORLD.group.Incl(ranks))

    # ------------------------------------------------------------------
    # Global singleton lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> "DwdpManager":
        """Register this manager and its dense DWDP manager globally (both or neither), then
        keep the gather's resident SMs out of the persistent grids."""
        if get_global_dwdp_manager() is not None:
            raise RuntimeError("DwdpManager already registered globally")
        set_global_dwdp_manager(self)
        try:
            self._dense_manager.__enter__()
            set_sm_reserve(self.resident_sms)
        except BaseException:
            # Not entered, so no __exit__ follows: unregister both and free the DWDP group
            # __init__ created. Nothing else exists before setup().
            try:
                self.cleanup()
            except BaseException:
                logger.error(
                    "[DwdpManager] Failed to free the DWDP group after a failed registration\n"
                    f"{traceback.format_exc()}"
                )
            finally:
                self._dense_manager.__exit__(None, None, None)
                set_global_dwdp_manager(None)
            raise
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        try:
            self.cleanup()
        finally:
            set_sm_reserve(0)
            set_global_dwdp_manager(None)
            self._dense_manager.__exit__(None, None, None)
        return False

    def is_enabled(self) -> bool:
        return self.dwdp_size > 1

    @property
    def dense_manager(self) -> DenseDwdpManager:
        """The attached dense DWDP manager, registered globally while this manager is entered."""
        return self._dense_manager

    @property
    def dense(self) -> DenseDWDP | None:
        """The committed dense DWDP runtime, or None when dense DWDP is not active.

        The dense manager's runtime: None when dense DWDP is off, before
        ``setup(model)`` commits it and after ``cleanup()``. The model's layer
        hooks run only while it is not None.
        """
        return self._dense_manager.dense

    def cleanup(self) -> None:
        """Release dense DWDP, tear down the weight manager, free the DWDP group.

        Dense DWDP goes first (``DenseDwdpManager.cleanup``, which also drops
        its reference to the weight manager): the release synchronizes the
        device, repoints the model's dense weights to zero-size tensors and
        frees the staging slots and the row store. The model keeps its
        ``_dense_dwdp_active`` mark, so weight reloads stay refused. If the
        model was garbage collected first, the bindings of its modules count
        as detached, since nothing can read them any more. Every step runs even
        when an earlier one fails; the first error is raised at the end and
        later ones are logged. Idempotent.
        """
        cleanup_error = None
        cleanup_traceback = None
        try:
            model = self._model_ref() if self._model_ref is not None else None
            try:
                self._dense_manager.cleanup()  # a no-op without a committed runtime
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

    def add_repeated_layer(self, layer_idx: int) -> None:
        """Declare that layer ``layer_idx`` runs several times in a row per forward pass.

        Called by models that reuse a decoder layer, e.g. the one-model MTP-Eagle layer that runs
        once per draft step. Its repeats bind the expert copy still in the prefetch slot instead
        of gathering the layer again. Layers whose MoE does not use DWDP are ignored.
        """
        if layer_idx not in self._repeated_layers:
            self._repeated_layers.append(layer_idx)

    def setup(self, model: nn.Module) -> Optional[DWDPWeightManager]:
        """Set dense DWDP, then the VA pipeline of expert DWDP, up on ``model``.

        First, if the model's decoder layers provide dense roots, runs
        ``setup_dense_dwdp`` on every rank, also on a rank that did not ask
        for dense DWDP, so the first dense collective gives every rank of the
        group the same verdict when ranks disagree on
        ``dwdp_config.enable_dense``. Dense setup errors propagate, raised on
        every rank together, before any expert setup work. A commit
        returns the replicated dense weights to the driver before the expert
        setup allocates its prefetch ring, so the setup peak is the larger of
        the replicated dense weights with the dense store and slots, and the
        dense shard with the ring, not their sum. Dense setup runs on an
        attached transfer: a copy stream that the transfer creates when dense
        setup first reads it, and the transfer flags computed as the weight
        manager computes them.

        Then ``setup_dwdp`` (transport + weight buffer + weight manager + MoE
        backend fixup) runs with that copy stream, if dense setup created
        it, so the dense and the expert copies share one copy stream, and the
        returned ``DWDPWeightManager`` is stored for subsequent runtime calls.
        It gets the committed runtime's slots as reserved storages, so that
        an expert weight dense DWDP moved is refused before the transport
        would free a slot. The committed runtime, if any, goes to the dense
        manager together with the weight manager (``DenseDwdpManager.attach``)
        and becomes ``dense``. If anything fails after the commit and before
        the dense manager took the runtime over, the runtime is released and
        the error propagates. Whether a model has roots depends only on its
        code, so on a model without them no rank makes a dense collective: a
        rank that asks for dense DWDP there raises alone, at once, after the
        expert setup (``refuse_dense_dwdp_without_roots``).

        Called once from ``py_executor_creator`` after weight loading. Once
        dense DWDP is committed, a later call for the same model returns the
        weight manager and keeps the runtime, making no collective; every
        rank committed together, so every rank returns early alike. A later
        call for the same model after dense DWDP stayed off runs dense setup
        on the weight manager, whose stream and flags it then uses; the
        expert setup must return that same weight manager.

        Raises:
            RuntimeError: Called after ``cleanup()``; dense DWDP is committed
                for another model; or the model's dense weights were moved
                by a dense DWDP runtime this manager does not hold (it was
                released, so the modules hold zero-size tensors). Each is
                raised before any expert or dense setup work. Also raised
                when the weight manager does not run on the copy stream and
                transfer dense DWDP was set up with, and when dense DWDP
                moved an expert weight the transport takes over.
            DenseSetupError: Dense DWDP setup failed, or was requested on a
                model without dense roots.
            PrefetchRingMemoryError: The expert prefetch ring does not fit in
                the free device memory.
        """
        if self.dwdp_group is None:
            raise RuntimeError("DwdpManager.setup() called after cleanup()")
        if self._dense_manager.dense is not None:
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
        roots = dense_roots_available(model)
        # A repeated call for the model whose expert setup is done (dense DWDP stayed off): the
        # expert setup returns its cached weight manager, so dense setup runs on that one.
        expert_done = (
            self._weight_manager is not None
            and self._model_ref is not None
            and self._model_ref() is model
        )
        attached = (
            None
            if expert_done
            else _AttachedTransfer(
                device_id,
                contention_opt=self.config.contention_opt,
                copy_mode=self.config.copy_mode,
            )
        )
        transfer = self._weight_manager if attached is None else attached
        dense = None
        if roots:  # on every rank, requested or not: collective #1 gives one verdict
            # Collective #1 is the group's first collective of the setup, where a rank waits for
            # peers still loading: one line, so that a rank waiting there shows where it waits.
            logger.info(
                f"[DwdpManager] DWDP rank {self.dwdp_rank} of {self.dwdp_size}: dense DWDP setup "
                f"first (its collective #1 waits for every rank of the DWDP group), then expert "
                f"DWDP setup"
            )
            dense = setup_dense_dwdp(
                model,
                transfer,
                self.dwdp_group,
                dwdp_rank=self.dwdp_rank,
                dwdp_size=self.dwdp_size,
                device_id=device_id,
                moe_layers=layer_indices,
                config=DenseConfig.from_dwdp_config(self.config),
            )
        try:
            self._weight_manager = setup_dwdp(
                model=model,
                mapping=self._mapping,
                device_id=device_id,
                comm=self.dwdp_group,
                layer_indices=layer_indices,
                num_experts_per_worker=self.num_experts_per_worker,
                num_prefetch_experts=self.num_prefetch_experts,
                contention_opt=self.config.contention_opt,
                copy_mode=self.config.copy_mode,
                num_prefetch_slots=self.config.num_prefetch_slots,
                repeated_layer_indices=[
                    layer for layer in self._repeated_layers if layer in self._registered_layers
                ],
                copy_stream=None if attached is None else attached.created_stream,
                reserved_storages=_dense_slot_storages(dense),
            )
            self._model_ref = weakref.ref(model)
            if dense is not None:
                _check_shared_transfer(transfer, self._weight_manager)
                self._dense_manager.attach(dense, expert=self._weight_manager)
        except BaseException:
            if dense is not None and self._dense_manager.dense is not dense:
                _release_during_rollback(dense, "dense DWDP runtime")
            raise
        if not roots and self.config.enable_dense:
            refuse_dense_dwdp_without_roots(model)
        if self.dense is not None and self.config.num_prefetch_slots > 2:
            logger.info(
                f"[DwdpManager] dense DWDP with {self.config.num_prefetch_slots} expert prefetch "
                f"slots: each expert copy runs ahead of its layer's dense copy on the shared copy "
                f"stream"
            )
        if self._store_rendezvous:
            # Lets DWDP rank 0, which serves the store, release it once every group is set up.
            self.dwdp_group.mark_setup_done()
        return self._weight_manager

    # ------------------------------------------------------------------
    # Runtime entry points (forwarded to DWDPWeightManager)
    # ------------------------------------------------------------------

    def start_forward(self) -> None:
        """Start a forward pass: prefetch its first layers.

        Prefetches the experts of the first MoE layers unless the previous
        forward pass already did (its last MoE layers gather the first layers
        of the next pass). With dense DWDP, the dense manager first enqueues
        the dense weights of the first two layers, unless they are already
        enqueued, so the copies this enqueues put each layer's dense copy
        ahead of its expert copy on the shared copy stream. While a forward is
        still open, which means the previous forward aborted inside the covered
        layers, this enqueues nothing: the model's forward entry
        (``DenseDwdpManager.ensure_primed``) recovers on the execution stream,
        dense first.
        """
        if self._weight_manager is None:
            raise RuntimeError("DwdpManager.setup() has not been called yet")
        dense = self._dense_manager.dense
        if dense is not None and dense.forward_open:
            return
        self._dense_manager.prefetch_first_layers()  # a no-op without a committed runtime
        self._weight_manager.start_forward()

    def ensure_primed(self) -> None:
        """Model-side priming (H0) through ``DenseDwdpManager.ensure_primed``.

        The model reaches the dense manager directly; this delegate serves
        callers that hold the ``DwdpManager``. The dense manager refuses a CUDA
        graph capture and multi-stream execution before it records or enqueues
        anything; after an aborted forward it records the dense WAR barrier on
        the current stream; it enqueues ``L(first), L(second)`` unless already
        primed, then starts the expert prefetch (``start_forward`` of the
        weight manager).

        Raises:
            RuntimeError: Dense DWDP is not active.
            DenseProtocolError: The current stream is capturing a CUDA graph,
                or multi-stream execution is on.
        """
        if self._dense_manager.dense is None or self._weight_manager is None:
            raise RuntimeError("DwdpManager.ensure_primed() needs an active dense DWDP runtime")
        self._dense_manager.ensure_primed()

    def wait_and_bind(self, backend_module: nn.Module, layer_idx: int) -> None:
        """Wait for prefetch and bind full-tensor views onto backend weights."""
        if self._weight_manager is None:
            raise RuntimeError("DwdpManager.setup() has not been called yet")
        self._weight_manager.wait_and_bind(backend_module, layer_idx)

    def finish_layer(self, layer_idx: int) -> None:
        """Release the layer's buffer slot and prefetch the layers it unblocks.

        Called after the layer's MoE kernel has been enqueued, so the slot's
        consume event covers that kernel.
        """
        if self._weight_manager is None:
            raise RuntimeError("DwdpManager.setup() has not been called yet")
        self._weight_manager.finish_layer(layer_idx)

    def invalidate_prefetch(self) -> None:
        """Drop the prefetches issued ahead of the next forward pass.

        Call after the expert weights may have changed (e.g. a weight update);
        the next forward pass gathers its first layers again.
        """
        if self._weight_manager is not None:
            self._weight_manager.invalidate_prefetch()
