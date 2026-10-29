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
"""Dense DWDP setup: phases P1-P6 over five fixed collectives.

Setup runs once per context worker, from ``DwdpManager.setup`` right after the expert DWDP setup
returned, so the model is fully loaded and post-processed and the expert transport, buffers and
backend fixups are done.  Each phase's local work is wrapped so that a failure becomes
``ok=False`` in the next collective instead of an exception that would skip it; every rank makes
the same collectives in the same order and reaches the same verdict from the same gathered
payloads.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, NamedTuple, Protocol

import torch
from torch import nn

from tensorrt_llm.logger import logger

from ..setup import _get_decoder_model
from .config import DENSE_ENV, DENSE_REQUIRED_ENV, DenseConfig, dense_dwdp_requested
from .layout import collect_dense_layout, compute_checksums
from .plan import DenseCopyLauncher, build_copy_plans, select_copy_mode
from .repoint import BindingState, DenseBindings
from .runtime import DenseDWDP
from .store import DenseShardStore, FabricShardStore, StorePayload
from .types import (
    LOG_PREFIX,
    Checksum,
    DenseLayout,
    DenseLayoutError,
    DenseSetupError,
    DeviceOps,
    LayerCopyPlan,
    LayerLauncher,
    SegmentKind,
    SlotBuffer,
    TorchCudaOps,
    allocate_slot_ring,
)
from .watchdog import SlotWatchdog

if TYPE_CHECKING:
    from ..weight_manager import DWDPWeightManager

#: Free device memory P1 requires on top of the store and the slots.
SETUP_HEADROOM_BYTES = 2 << 30

#: The phase each collective concludes, by collective number.
_COLLECTIVES = {
    1: "P1 collection",
    2: "P2 store and slots",
    3: "P3 peer mapping and copy plans",
    4: "P4 Phase V",
    5: "P5 repoint self-check",
}

#: Most paths or findings one message lists.
_MAX_LISTED = 8


class GroupComm(Protocol):
    """The DWDP group calls setup makes: an mpi4py sub-communicator or ``StoreDwdpGroup``."""

    def allgather(self, obj: object) -> list[object]:
        """Gather one picklable object per rank, ordered by rank."""
        ...

    def Get_rank(self) -> int:
        """This rank within the group."""
        ...

    def Get_size(self) -> int:
        """Ranks in the group."""
        ...


class PhaseReport(NamedTuple):
    """What one rank contributes to a setup collective.

    Attributes:
        ok: The phase's local work succeeded.
        error: ``str`` of the local failure; empty when ``ok``.
        data: Phase payload (picklable); see :func:`setup_dense_dwdp` for each collective.
    """

    ok: bool
    error: str = ""
    data: object = None


class _CollectData(NamedTuple):
    """The data of one rank in collective #1."""

    requested: bool
    """``TRTLLM_DWDP_DENSE`` on this rank."""

    required: bool
    """``TRTLLM_DWDP_DENSE_REQUIRED`` on this rank; True if its switches could not be read."""

    digest: str
    """Layout digest; empty if collection failed."""

    checksums: tuple[Checksum, ...]
    """Per-segment content checksums in store order."""

    store_bytes: int
    """Store bytes this rank would allocate."""

    slot_bytes: int
    """Slot bytes this rank would allocate."""

    free_bytes: int
    """Free device memory before setup allocated anything, with the allocator cache emptied."""


#: Collective #1 data of a rank that did not ask for dense DWDP.
_NOT_REQUESTED = _CollectData(False, True, "", (), 0, 0, 0)

#: Builds the row store for ``(layout, rank, device)``; tests pass the loopback store.
StoreFactory = Callable[[DenseLayout, int, torch.device], DenseShardStore]

#: Builds the copy launcher for ``(plans, weight_manager, device)``; setup calls it with the
#: expert DWDP copy stream current.
LauncherFactory = Callable[
    [Mapping[int, LayerCopyPlan], "DWDPWeightManager", torch.device], LayerLauncher
]

#: ``(free, total)`` bytes of the dense device, as ``torch.cuda.mem_get_info`` returns them.
MemGetInfo = Callable[[], tuple[int, int]]


def dense_roots_available(model: nn.Module) -> bool:
    """Whether the model's decoder layers provide ``dense_dwdp_roots``.

    ``DwdpManager.setup`` calls :func:`setup_dense_dwdp` for such models whether or not dense
    DWDP is requested on this rank, so collective #1 always runs and catches ranks whose
    ``TRTLLM_DWDP_DENSE`` disagrees.  The answer depends only on the model code, so it is the
    same on every rank: for a model without roots no rank makes a dense collective, and a rank
    that asks for dense DWDP calls :func:`refuse_dense_dwdp_without_roots` instead.  The decoder
    is found as the expert DWDP setup finds it; a model without one has no roots.
    """
    try:
        decoder = _get_decoder_model(model)
    except RuntimeError:
        return False
    return any(callable(getattr(layer, "dense_dwdp_roots", None)) for layer in decoder.layers)


def refuse_dense_dwdp_without_roots(model: nn.Module) -> None:
    """Turn down a request for dense DWDP on a model without ``dense_dwdp_roots``.

    ``DwdpManager.setup`` calls it instead of :func:`setup_dense_dwdp` on a rank whose
    ``TRTLLM_DWDP_DENSE`` is on when :func:`dense_roots_available` is False.  No rank makes a
    dense collective for such a model (whether it has roots is the same on every rank), so this
    rank decides alone and at once rather than wait in collective #1 for peers whose switch is
    off: it raises if its ``TRTLLM_DWDP_DENSE_REQUIRED`` is on (also when its switches cannot be
    read), else logs an error and returns, and dense DWDP stays off with the dense weights
    replicated.  A rank whose ``TRTLLM_DWDP_DENSE`` is off does not call it and carries on.

    Args:
        model: The loaded model, named in the message.

    Raises:
        DenseSetupError: ``TRTLLM_DWDP_DENSE_REQUIRED`` is on or unreadable.
    """
    try:
        required = DenseConfig.from_env().required
    except ValueError:
        required = True
    message = (
        f"{DENSE_ENV} asks for dense DWDP, but the decoder layers of {type(model).__name__} do not "
        f"provide dense_dwdp_roots, so it has no dense weights to shard"
    )
    if required:
        raise DenseSetupError(f"{message}; unset {DENSE_ENV} for this model")
    logger.error(
        f"{LOG_PREFIX} {message}; {DENSE_REQUIRED_ENV}=0, so dense DWDP is off on this rank and "
        f"the dense weights stay replicated"
    )


def dense_dwdp_active(model: nn.Module) -> bool:
    """Whether dense DWDP moved the dense weights of ``model``.

    True once setup P6 (or a release after the commit point) set ``_dense_dwdp_active = True``
    on the model or its decoder.  The mark stays after the runtime is released, when the modules
    hold zero-size tensors, so such a model can be neither reloaded nor set up again.  The
    decoder is found as the expert DWDP setup finds it.
    """
    if getattr(model, "_dense_dwdp_active", False) is True:
        return True
    try:
        decoder = _get_decoder_model(model)
    except RuntimeError:
        return False
    return getattr(decoder, "_dense_dwdp_active", False) is True


def setup_dense_dwdp(
    model: nn.Module,
    weight_manager: DWDPWeightManager,
    comm: GroupComm,
    *,
    dwdp_rank: int,
    dwdp_size: int,
    device_id: int,
    moe_layers: list[int],
    config: DenseConfig | None = None,
    store_factory: StoreFactory | None = None,
    ops: DeviceOps | None = None,
    device: torch.device | None = None,
    launcher_factory: LauncherFactory | None = None,
    mem_get_info: MemGetInfo | None = None,
) -> DenseDWDP | None:
    """Shard, verify and repoint the dense weights of ``model``.

    Each collective carries a :class:`PhaseReport` per rank; its local work and verdict:

    1. P1: read the config, ``layout.collect_dense_layout`` over the decoder layers with the
       ``dense_dwdp_attr_allow`` / ``dense_dwdp_exclude`` tables the layers carry, check that the
       covered layers are ``moe_layers`` and that ``free >= store + slots +
       SETUP_HEADROOM_BYTES`` (``free`` read after ``torch.cuda.empty_cache()``), then
       ``layout.compute_checksums``; no copies.  A rank that did not ask for dense DWDP does none
       of this.  Data ``(requested, required, digest, checksums, store_bytes, slot_bytes,
       free_bytes)``.  No rank requested: return None and make no further collective.  Some did:
       every rank raises a configuration error, whatever ``required`` says.  All ok with equal
       digests and checksums: continue; else abandon.
    2. P2: allocate the store and the slots, write this rank's rows, ``synchronize``, construct
       the runtime (it records the consume events; with ``config.slot_watchdog`` it gets the
       slot checksum watchdog over the P1 checksums), export.  Data: the exported handles.
    3. P3: import and map the peers, build the plans and, with the copy stream current, the
       launcher; install it, ``synchronize``.  Doubles as the barrier after every write and
       mapping.
    4. P4, Phase V: per layer ``DenseDWDP.fetch_blocking``, then ``torch.equal`` of every
       anchor's slot bytes with its ``source``.  Data: the mismatching anchor paths.
    5. P5: ``DenseBindings.repoint`` (originals alive), then per layer ``fetch_blocking`` and
       ``DenseBindings.verify_layer``.  Data: the mismatching binding paths.  The commit point:
       unless every rank is ok, every rank rolls back and abandons.

    After collectives 2 to 4, any rank not ok means every rank abandons.

    P6 (no collective, host only): ``DenseBindings.commit`` (drop originals, release check),
    ``torch.cuda.empty_cache()``, ``DenseDWDP.commit`` with ``config.slot_fill_byte``, set
    ``_dense_dwdp_active`` on the model and its decoder, log the commit line with the moved,
    store, slot and freed bytes and the leak count.

    Every phase's local work catches ``Exception`` into ``ok=False``; ``BaseException`` propagates.
    Abandoning releases everything this rank allocated: the bindings are rolled back, then the
    runtime is released (launcher, slots, store) or, if no runtime was built, the store and the
    slots; then the launcher.  Then every rank raises :class:`~.types.DenseSetupError` if any
    rank's ``config.required`` is set (an unreadable config counts as set), else logs an error
    and returns None, with the model's original tensors in place (expert DWDP and the replicated
    dense weights keep working).  A rank whose rollback fails raises regardless.  Any exception
    that escapes (a broken group, ``BaseException``, a P6 failure) releases this rank's dense
    state the same way first; bindings already committed are detached instead of rolled back.

    Args:
        model: The loaded model (the VL wrapper or the causal LM); the decoder is found like the
            expert setup finds it.
        weight_manager: The expert DWDP weight manager: copy stream and transfer mode.
        comm: The DWDP group.
        dwdp_rank: This rank in the group.
        dwdp_size: Group size ``N``.
        device_id: CUDA device of this rank.
        moe_layers: Expert DWDP MoE layers; the covered dense layers must equal them.
        config: Switches; ``DenseConfig.from_env()`` (inside P1) if None.  ``config.enabled`` (or
            ``TRTLLM_DWDP_DENSE``) says whether this rank asks for dense DWDP.
        store_factory: Row store constructor; ``FabricShardStore`` if None.
        ops: CUDA seam for the runtime; ``TorchCudaOps(device_id)`` if None.
        device: The dense device; ``cuda:<device_id>`` if None (unit tests pass the CPU).
        launcher_factory: Builds the copy launcher from the plans; if None,
            ``DenseCopyLauncher(plans, select_copy_mode(use_gather=weight_manager.use_gather,
            use_batched=weight_manager.use_batched_prefetch), device=device)``.
        mem_get_info: ``(free, total)`` device memory; ``torch.cuda.mem_get_info(device_id)`` if
            None.

    Returns:
        The committed runtime, or None when dense DWDP is off on every rank or was abandoned
        with ``required`` off on every rank.

    Raises:
        DenseSetupError: Abandoned with ``required`` on, ranks disagree on
            ``TRTLLM_DWDP_DENSE``, or this rank could not restore its original dense weights.
        DenseRepointError: ``config.strict`` and an original storage survived the commit (this
            rank only; its dense state is released first).
    """
    if config is not None:
        requested = config.enabled
    else:
        requested = dense_dwdp_requested()
    run = _DenseSetup(
        model,
        weight_manager,
        comm,
        rank=dwdp_rank,
        size=dwdp_size,
        device_id=device_id,
        device=device,
        moe_layers=moe_layers,
        config=config,
        requested=requested,
        store_factory=FabricShardStore if store_factory is None else store_factory,
        ops=TorchCudaOps(device_id) if ops is None else ops,
        launcher_factory=_copy_launcher if launcher_factory is None else launcher_factory,
        mem_get_info=_device_mem_get_info(device_id) if mem_get_info is None else mem_get_info,
    )
    return run.run()


class _DenseSetup:
    """One rank's run of :func:`setup_dense_dwdp`: its phases, collectives and resources.

    It holds what this rank allocates until the runtime owns it: the store and the slots until
    the runtime constructor returns, the bindings until ``DenseDWDP.commit``, the launcher until
    setup ends.  Every rank calls :meth:`_exchange` exactly when every other rank does, because
    whether collective ``k + 1`` happens depends only on the reports collective ``k`` gathered.
    """

    def __init__(
        self,
        model: nn.Module,
        weight_manager: DWDPWeightManager,
        comm: GroupComm,
        *,
        rank: int,
        size: int,
        device_id: int,
        device: torch.device | None,
        moe_layers: Sequence[int],
        config: DenseConfig | None,
        requested: bool,
        store_factory: StoreFactory,
        ops: DeviceOps,
        launcher_factory: LauncherFactory,
        mem_get_info: MemGetInfo,
    ) -> None:
        self._model = model
        self._weight_manager = weight_manager
        self._comm = comm
        self._rank = rank
        self._size = size
        self._device_id = device_id
        self._device_override = device
        self._moe_layers = moe_layers
        self._config = config
        self._requested = requested
        self._store_factory = store_factory
        self._ops = ops
        self._launcher_factory = launcher_factory
        self._mem_get_info = mem_get_info
        self._started = time.monotonic()
        # Decided from collective #1; raising is the safe default until then.
        self._required = True
        self._device: torch.device | None = None
        self._decoder: nn.Module | None = None
        self._free_before = 0
        self._allocated_before = 0
        self._layout: DenseLayout | None = None
        self._store: DenseShardStore | None = None
        self._slots: dict[SegmentKind, tuple[SlotBuffer, ...]] | None = None
        self._runtime: DenseDWDP | None = None
        self._launcher: LayerLauncher | None = None
        self._bindings: DenseBindings | None = None

    def run(self) -> DenseDWDP | None:
        """Run the phases; whatever escapes releases this rank's dense state first."""
        try:
            return self._run()
        except BaseException:
            self._release()
            raise

    def _run(self) -> DenseDWDP | None:
        reports = self._exchange(1, self._collect())
        data = [report.data for report in reports]
        if not any(item.requested for item in data):
            logger.debug(f"{LOG_PREFIX} TRTLLM_DWDP_DENSE is off on every rank; dense DWDP is off")
            return None
        if not all(item.requested for item in data):
            on = [rank for rank, item in enumerate(data) if item.requested]
            off = [rank for rank, item in enumerate(data) if not item.requested]
            raise DenseSetupError(
                f"TRTLLM_DWDP_DENSE differs across the DWDP group: on for {_ranks(on)}, off for "
                f"{_ranks(off)}; set it the same on every context worker of the group"
            )
        self._required = any(item.required for item in data)
        reason = _failures(reports) or _disagreement(self._layout, data)
        if reason:
            return self._abandon(1, reason)
        self._log_agreement(data)

        reports = self._exchange(2, self._allocate())
        if reason := _failures(reports):
            return self._abandon(2, reason)
        reports = self._exchange(3, self._map([report.data for report in reports]))
        if reason := _failures(reports):
            return self._abandon(3, reason)
        reports = self._exchange(4, self._compare_blobs())
        if reason := _failures(reports):
            return self._abandon(4, reason)
        reports = self._exchange(5, self._repoint())
        if reason := _failures(reports):
            return self._abandon(5, reason)
        return self._commit()

    # ------------------------------------------------------------------
    # Collectives and verdicts
    # ------------------------------------------------------------------

    def _exchange(self, number: int, report: PhaseReport) -> list[PhaseReport]:
        """Collective ``number``: every rank's report in rank order; malformed ones count as failed."""
        reports = [_checked_report(number, item) for item in self._comm.allgather(report)]
        logger.debug(
            f"{LOG_PREFIX} collective #{number} ({_COLLECTIVES[number]}): "
            f"{sum(report.ok for report in reports)}/{len(reports)} ranks ok"
        )
        return reports

    def _abandon(self, number: int, reason: str) -> None:
        """Release this rank's dense state, then raise or return as the group's verdict says."""
        intact = self._release()
        torch.cuda.empty_cache()
        message = (
            f"dense DWDP setup abandoned at collective #{number} ({_COLLECTIVES[number]}): {reason}"
        )
        if not intact:
            raise DenseSetupError(
                f"{message}; this rank could not restore its original dense weights (see the "
                f"errors logged above)"
            )
        if self._required:
            raise DenseSetupError(message)
        logger.error(
            f"{LOG_PREFIX} {message}; TRTLLM_DWDP_DENSE_REQUIRED=0 on every rank, so dense DWDP "
            f"is off and the dense weights stay replicated"
        )
        return None

    def _log_agreement(self, data: Sequence[_CollectData]) -> None:
        layout = self._layout
        lowest = min(range(len(data)), key=lambda rank: data[rank].free_bytes)
        logger.info(
            f"{LOG_PREFIX} layout agreed by {len(data)} ranks: {len(layout.layers)} layers "
            f"{layout.layers[0]}..{layout.layers[-1]}, segments {_segment_names(layout)}, "
            f"moved={_gb(layout.moved_bytes)}GB store={_gb(layout.store_bytes)}GB "
            f"slots={_gb(layout.slot_bytes)}GB, digest {layout.digest[:16]}; lowest free memory "
            f"{_gb(data[lowest].free_bytes)}GB on rank {lowest}"
        )

    # ------------------------------------------------------------------
    # Phases (local work; every failure becomes a report)
    # ------------------------------------------------------------------

    def _collect(self) -> PhaseReport:
        """P1, no copies: switches, collection, coverage, memory precheck, checksums."""
        if not self._requested:
            return PhaseReport(True, data=_NOT_REQUESTED)
        required = True if self._config is None else self._config.required
        free_bytes = 0
        try:
            config = DenseConfig.from_env() if self._config is None else self._config
            self._config = config
            required = config.required
            self._check_group()
            if self._device_override is None:
                self._device = torch.device("cuda", self._device_id)
            else:
                self._device = torch.device(self._device_override)
            # Blocks the caching allocator holds count as free: the store needs driver memory,
            # and P6 measures freed(mem_get_info) after its own empty_cache.
            torch.cuda.empty_cache()
            free_bytes, _ = self._mem_get_info()
            self._allocated_before = _allocated_bytes(self._device)
            self._decoder = _get_decoder_model(self._model)
            layers = list(enumerate(self._decoder.layers))
            attr_allow, exclude = _model_tables(layers)
            layout = collect_dense_layout(
                layers,
                config,
                num_ranks=self._size,
                device=self._device,
                attr_allow=attr_allow,
                exclude=exclude,
            )
            _check_coverage(layout, self._moe_layers)
            _check_free_memory(layout, free_bytes)
            compute_checksums(layout)
            data = _CollectData(
                requested=True,
                required=required,
                digest=layout.digest,
                checksums=layout.checksums(),
                store_bytes=layout.store_bytes,
                slot_bytes=layout.slot_bytes,
                free_bytes=free_bytes,
            )
        except Exception as exc:  # a local failure must reach collective #1, never skip it
            return PhaseReport(
                False, _describe(exc), _CollectData(True, required, "", (), 0, 0, free_bytes)
            )
        self._layout = layout
        self._free_before = free_bytes
        return PhaseReport(True, data=data)

    def _allocate(self) -> PhaseReport:
        """P2: store, slots, this rank's rows, synchronize, runtime and its watchdog, export."""
        layout = self._layout
        try:
            store = self._store_factory(layout, self._rank, self._device)
            self._store = store
            store.allocate()
            self._slots = {
                kind: allocate_slot_ring(layout.slot_specs[kind], self._device)
                for kind in layout.segment_kinds
            }
            store.write_local_rows()
            self._ops.synchronize()
            watchdog = None
            if self._config.slot_watchdog:
                watchdog = SlotWatchdog.from_layout(layout, self._device)
            self._runtime = DenseDWDP(
                layout.layers,
                layout.segment_kinds,
                self._slots,
                self._weight_manager.copy_stream,
                ops=self._ops,
                strict=self._config.strict,
                store=store,
                watchdog=watchdog,
            )
            payload = store.export()
        except Exception as exc:  # a local failure must reach collective #2, never skip it
            return PhaseReport(False, _describe(exc))
        return PhaseReport(True, data=payload)

    def _map(self, payloads: Sequence[StorePayload]) -> PhaseReport:
        """P3: map the peers' rows, build the plans and the launcher, synchronize."""
        try:
            store = self._store
            store.import_peers(payloads)
            slot_bases = {kind: [slot.base for slot in ring] for kind, ring in self._slots.items()}
            plans = build_copy_plans(
                self._layout,
                rank=self._rank,
                chunk_bases=store.chunk_bases(),
                slot_bases=slot_bases,
            )
            with self._ops.stream(self._weight_manager.copy_stream):
                self._launcher = self._launcher_factory(plans, self._weight_manager, self._device)
            self._runtime.set_launcher(self._launcher)
            self._ops.synchronize()
        except Exception as exc:  # a local failure must reach collective #3, never skip it
            return PhaseReport(False, _describe(exc))
        return PhaseReport(True)

    def _compare_blobs(self) -> PhaseReport:
        """P4, Phase V: gather every layer and compare each anchor's slot bytes with its source."""
        layout = self._layout
        mismatches: list[str] = []
        try:
            for layer in layout.layers:
                self._runtime.fetch_blocking(layer)
                for segment in layout.layer_segments(layer):
                    slot = self._slots[segment.segment][layout.slot_index(segment.segment, layer)]
                    blob = slot.blob_view(segment.blob_bytes)
                    for anchor in segment.anchors:
                        gathered = blob[anchor.blob_offset : anchor.blob_end]
                        if not torch.equal(gathered, anchor.source):
                            mismatches.append(anchor.path)
        except Exception as exc:  # a local failure must reach collective #4, never skip it
            return PhaseReport(False, _describe(exc), mismatches)
        if mismatches:
            return PhaseReport(
                False,
                f"Phase V: {len(mismatches)} gathered anchors differ from their originals: "
                f"{_listing(mismatches)}",
                mismatches,
            )
        return PhaseReport(True, data=mismatches)

    def _repoint(self) -> PhaseReport:
        """P5: repoint every binding, then gather every layer and self-check it through the modules."""
        mismatches: list[str] = []
        try:
            bindings = DenseBindings(self._layout, self._slots)
            self._bindings = bindings
            bindings.repoint()
            for layer in self._layout.layers:
                self._runtime.fetch_blocking(layer)
                mismatches.extend(bindings.verify_layer(layer))
        except Exception as exc:  # a local failure must reach collective #5, never skip it
            return PhaseReport(False, _describe(exc), mismatches)
        if mismatches:
            return PhaseReport(
                False,
                f"P5 self-check: {len(mismatches)} module tensors are not their slot views or "
                f"differ from their P1 checksums after repointing: {_listing(mismatches)}",
                mismatches,
            )
        return PhaseReport(True, data=mismatches)

    def _commit(self) -> DenseDWDP:
        """P6, past the commit point, host only: drop the originals and arm the runtime."""
        config, layout = self._config, self._layout
        bindings, runtime = self._bindings, self._runtime
        leaks = bindings.commit(strict=config.strict, describe_referrers=config.debug >= 1)
        torch.cuda.empty_cache()
        free_after, _ = self._mem_get_info()
        allocated_after = _allocated_bytes(self._device)
        runtime.commit(bindings, fill_byte=config.slot_fill_byte)
        # The runtime owns the bindings, the slots, the store and the launcher from here on.
        self._bindings = self._runtime = self._store = self._slots = self._launcher = None
        self._mark_active()
        logger.info(
            f"{LOG_PREFIX} committed: moved={_gb(layout.moved_bytes)}GB "
            f"store={_gb(layout.store_bytes)}GB slots={_gb(layout.slot_bytes)}GB "
            f"freed(mem_get_info)={_gb(free_after - self._free_before)}GB "
            f"freed(memory_allocated)={_gb(self._allocated_before - allocated_after)}GB "
            f"leaks={len(leaks)} layers={len(layout.layers)} segments={_segment_names(layout)} "
            f"setup={time.monotonic() - self._started:.1f}s"
        )
        return runtime

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_group(self) -> None:
        """The group must be the one the caller describes; the rows are indexed by its ranks."""
        group = (self._comm.Get_rank(), self._comm.Get_size())
        if group != (self._rank, self._size):
            raise DenseSetupError(
                f"the DWDP group reports rank {group[0]} of {group[1]}, but dense setup runs as "
                f"rank {self._rank} of {self._size}"
            )

    def _mark_active(self) -> None:
        """Flag the model and its decoder: their dense tensors are no longer their own weights.

        The weight reload, update and load guards read this flag.
        """
        self._model._dense_dwdp_active = True
        if self._decoder is not None:
            self._decoder._dense_dwdp_active = True

    def _release(self) -> bool:
        """Undo this rank's setup work; every step runs and failures are logged, never raised.

        Order: the bindings go back to the originals (to zero-size tensors once committed); then
        the runtime is released, which synchronizes, drops the launcher and the slots and
        releases the store, or, if no runtime was built, the store is released and the slots
        dropped; then the launcher's own ``release`` runs.

        Returns:
            Whether the modules hold their original tensors.
        """
        intact = True
        bindings, self._bindings = self._bindings, None
        if bindings is not None:
            committed = bindings.state in (BindingState.COMMITTED, BindingState.DETACHED)
            if committed:
                intact = False
                self._mark_active()
            try:
                if committed:
                    bindings.detach()
                else:
                    bindings.rollback()
            except Exception as exc:  # every release step runs; the failure is logged
                intact = False
                logger.error(f"{LOG_PREFIX} restoring the module tensors failed: {exc!r}")
        runtime, self._runtime = self._runtime, None
        store, self._store = self._store, None
        self._slots = None
        try:
            if runtime is not None:
                runtime.release()
            elif store is not None:
                store.release()
        except Exception as exc:  # every release step runs; the failure is logged
            logger.error(f"{LOG_PREFIX} releasing the dense store and slots failed: {exc!r}")
        launcher, self._launcher = self._launcher, None
        release = getattr(launcher, "release", None)
        if callable(release):
            try:
                release()
            except Exception as exc:  # every release step runs; the failure is logged
                logger.error(f"{LOG_PREFIX} releasing the dense copy launcher failed: {exc!r}")
        self._layout = None
        return intact


def _copy_launcher(
    plans: Mapping[int, LayerCopyPlan], weight_manager: DWDPWeightManager, device: torch.device
) -> LayerLauncher:
    """The production launcher: the transfer the expert DWDP prefetch uses."""
    mode = select_copy_mode(
        use_gather=weight_manager.use_gather, use_batched=weight_manager.use_batched_prefetch
    )
    return DenseCopyLauncher(plans, mode, device=device)


def _device_mem_get_info(device_id: int) -> MemGetInfo:
    def mem_get_info() -> tuple[int, int]:
        return torch.cuda.mem_get_info(device_id)

    return mem_get_info


def _allocated_bytes(device: torch.device) -> int:
    """Bytes the caching allocator hands out on ``device``; 0 for a device that is not CUDA."""
    return torch.cuda.memory_allocated(device) if device.type == "cuda" else 0


def _model_tables(layers: Sequence[tuple[int, nn.Module]]) -> tuple[frozenset[str], frozenset[str]]:
    """The model's allow and exclude tables, read from the decoder layers.

    Raises:
        DenseLayoutError: Decoder layers carry different tables; the collector applies one pair
            to every layer.
    """
    tables: dict[tuple[frozenset[str], frozenset[str]], list[int]] = {}
    for index, layer in layers:
        pair = (
            frozenset(getattr(layer, "dense_dwdp_attr_allow", frozenset())),
            frozenset(getattr(layer, "dense_dwdp_exclude", frozenset())),
        )
        tables.setdefault(pair, []).append(index)
    if len(tables) > 1:
        groups = "; ".join(f"layers {indices[:_MAX_LISTED]}" for indices in tables.values())
        raise DenseLayoutError(
            f"the decoder layers carry {len(tables)} different dense_dwdp_attr_allow / "
            f"dense_dwdp_exclude tables ({groups}); dense DWDP applies one pair to every layer"
        )
    if not tables:
        return frozenset(), frozenset()
    (pair,) = tables
    return pair


def _check_coverage(layout: DenseLayout, moe_layers: Sequence[int]) -> None:
    """The covered layers must be the expert DWDP MoE layers: priming walks those for both.

    Raises:
        DenseLayoutError: They differ, or ``moe_layers`` repeats a layer.
    """
    expected = sorted(moe_layers)
    if len(set(expected)) != len(expected):
        raise DenseLayoutError(f"the expert DWDP MoE layers repeat a layer: {expected}")
    if list(layout.layers) != expected:
        dense_only = sorted(set(layout.layers) - set(expected))
        expert_only = sorted(set(expected) - set(layout.layers))
        raise DenseLayoutError(
            f"dense DWDP covers {len(layout.layers)} layers and expert DWDP manages "
            f"{len(expected)}; they must be the same layers: only dense "
            f"{dense_only[:_MAX_LISTED]}, only expert {expert_only[:_MAX_LISTED]}"
        )


def _check_free_memory(layout: DenseLayout, free_bytes: int) -> None:
    """P1 precheck: ``free >= store + slots + SETUP_HEADROOM_BYTES``.

    Raises:
        DenseSetupError: Too little free device memory.
    """
    needed = layout.store_bytes + layout.slot_bytes + SETUP_HEADROOM_BYTES
    if free_bytes < needed:
        raise DenseSetupError(
            f"{_gb(free_bytes)} GB of device memory is free, but the store "
            f"({_gb(layout.store_bytes)} GB), the slots ({_gb(layout.slot_bytes)} GB) and the "
            f"setup headroom ({_gb(SETUP_HEADROOM_BYTES)} GB) need {_gb(needed)} GB"
        )


def _checked_report(number: int, item: object) -> PhaseReport:
    """``item`` as a report of collective ``number``; anything malformed is a failed report."""
    if not isinstance(item, PhaseReport):
        item = PhaseReport(False, f"sent a {type(item).__name__} instead of a PhaseReport")
    if number == 1 and not isinstance(item.data, _CollectData):
        return PhaseReport(
            False,
            item.error or "sent no collective #1 data",
            _CollectData(True, True, "", (), 0, 0, 0),
        )
    return item


def _failures(reports: Sequence[PhaseReport]) -> str:
    """The failed ranks grouped by error; empty if every rank is ok."""
    by_error: dict[str, list[int]] = {}
    for rank, report in enumerate(reports):
        if not report.ok:
            by_error.setdefault(report.error or "failed without a message", []).append(rank)
    return "; ".join(f"{_ranks(ranks)}: {error}" for error, ranks in by_error.items())


def _disagreement(layout: DenseLayout | None, data: Sequence[_CollectData]) -> str:
    """Why ok ranks cannot share rows: different layouts, or the same layout with other bytes."""
    digests = _group_ranks(item.digest for item in data)
    if len(digests) > 1:
        listed = "; ".join(f"{digest[:16]} on {_ranks(ranks)}" for digest, ranks in digests.items())
        return (
            f"the dense layouts differ across ranks (digest {listed}): the ranks' models or "
            f"TRTLLM_DWDP_DENSE_* switches differ"
        )
    findings = []
    for index, (layer, kind) in enumerate(layout.segments):
        holders = _group_ranks(item.checksums[index] for item in data)
        if len(holders) > 1:
            findings.append(
                f"layers.{layer} {kind.value}: "
                + " vs ".join(_ranks(ranks) for ranks in holders.values())
            )
    if findings:
        return f"the dense weights differ across ranks: {_listing(findings)}"
    return ""


def _group_ranks(values: Iterable[object]) -> dict[object, list[int]]:
    groups: dict[object, list[int]] = {}
    for rank, value in enumerate(values):
        groups.setdefault(value, []).append(rank)
    return groups


def _ranks(ranks: Sequence[int]) -> str:
    return f"rank {ranks[0]}" if len(ranks) == 1 else f"ranks {list(ranks)}"


def _describe(exc: Exception) -> str:
    return f"{type(exc).__name__}: {exc}"


def _listing(findings: Sequence[str]) -> str:
    listed = "; ".join(findings[:_MAX_LISTED])
    more = len(findings) - _MAX_LISTED
    return listed + (f"; and {more} more" if more > 0 else "")


def _segment_names(layout: DenseLayout) -> str:
    return ",".join(kind.value for kind in layout.segment_kinds)


def _gb(nbytes: int) -> str:
    return f"{nbytes / 1e9:.2f}"
