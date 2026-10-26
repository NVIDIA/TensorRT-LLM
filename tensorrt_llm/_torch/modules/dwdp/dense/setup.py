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
"""Dense DWDP setup: phases P1-P6 over six collectives.

Setup runs once per context worker, after the model is fully loaded and post-processed: from
``DwdpManager.setup`` before the expert DWDP setup (with expert DWDP, so the commit returns the
replicated dense weights to the driver before the expert prefetch ring is allocated), or from
``manager.DenseDwdpManager.setup`` (standalone, over the attention-DP group).  Each phase's local
work is wrapped so that a failure becomes ``ok=False`` in the next collective instead of an
exception that would skip it; every rank makes the same collectives in the same order, reaches the
same verdict from the same gathered payloads, and raises or returns together with the others.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import NamedTuple, NoReturn, Protocol

import torch
from torch import nn

from tensorrt_llm.logger import logger

from ..rendezvous import GroupComm
from ..setup import _get_decoder_model
from .config import DenseConfig
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
    DenseRepointError,
    DenseSetupError,
    DeviceOps,
    LayerCopyPlan,
    LayerLauncher,
    SegmentKind,
    SlotBuffer,
    StreamLike,
    TorchCudaOps,
    allocate_slot_ring,
)

#: Free device memory P1 requires on top of the store and the slots.
SETUP_HEADROOM_BYTES = 2 << 30

#: The phase each collective concludes, by collective number.
_COLLECTIVES = {
    1: "P1 collection",
    2: "P2 store and slots",
    3: "P3 peer mapping and copy plans",
    4: "P4 Phase V",
    5: "P5 repoint self-check",
    6: "P6 commit",
}

#: Most paths or findings one message lists.
_MAX_LISTED = 8


class DenseTransfer(Protocol):
    """Where and how the dense copies run.

    ``DwdpManager``'s attached transfer (with expert DWDP: the copy stream the expert DWDP weight
    manager then runs on, and the expert transfer; on a repeated setup the ``DWDPWeightManager``
    itself) and ``manager.StandaloneTransfer`` (standalone: the dense manager's own stream and
    ``DwdpConfig.copy_mode``) implement it.  Setup first reads ``copy_stream`` in P2, inside the
    phase's local work: an implementation may create the stream on that first read, and a failure
    then becomes this rank's failed P2 report.
    """

    @property
    def copy_stream(self) -> StreamLike:
        """The stream every dense copy runs on; the same object on every read."""
        ...

    @property
    def use_gather(self) -> bool:
        """Whether the copies use the SM fabric gather (``CopyMode.GATHER``)."""
        ...

    @property
    def use_batched_prefetch(self) -> bool:
        """Whether the copies use one batched copy-engine copy per layer (``CopyMode.BATCHED``)."""
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
    """Whether this rank asks for dense DWDP (``DenseConfig.enabled``)."""

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
_NOT_REQUESTED = _CollectData(False, "", (), 0, 0, 0)


class _CommitData(NamedTuple):
    """The data of one rank in collective #6."""

    leaks: int
    """Original storages still alive after the commit."""

    leaked_paths: tuple[str, ...]
    """The first leaked anchor paths, at most ``_MAX_LISTED``."""

    strict: bool
    """Whether this rank runs in strict mode (``DenseConfig.strict``), where a leak fails setup."""


#: Builds the row store for ``(layout, rank, device)``; tests pass the loopback store.
StoreFactory = Callable[[DenseLayout, int, torch.device], DenseShardStore]

#: Builds the copy launcher for ``(plans, transfer, device)``; setup calls it with the transfer's
#: copy stream current.
LauncherFactory = Callable[
    [Mapping[int, LayerCopyPlan], DenseTransfer, torch.device], LayerLauncher
]

#: ``(free, total)`` bytes of the dense device, as ``torch.cuda.mem_get_info`` returns them.
MemGetInfo = Callable[[], tuple[int, int]]


def dense_roots_available(model: nn.Module) -> bool:
    """Whether the model's decoder layers provide ``dense_dwdp_roots``.

    With expert DWDP, ``DwdpManager.setup`` calls :func:`setup_dense_dwdp` for such models whether
    or not dense DWDP is requested on this rank, so collective #1 always runs and catches ranks
    whose ``DwdpConfig.enable_dense`` disagrees.  Standalone, ``DenseDwdpManager.setup`` checks it
    before it creates the dense group.  The answer depends only on the model code, so it is the
    same on every rank: for a model without roots no rank makes a dense collective (standalone,
    no rank builds the group either), and a rank that asks for dense DWDP calls
    :func:`refuse_dense_dwdp_without_roots` instead.  The decoder is found as the expert DWDP
    setup finds it; a model without one has no roots.
    """
    try:
        decoder = _get_decoder_model(model)
    except RuntimeError:
        return False
    return any(callable(getattr(layer, "dense_dwdp_roots", None)) for layer in decoder.layers)


def refuse_dense_dwdp_without_roots(model: nn.Module) -> NoReturn:
    """Turn down a request for dense DWDP on a model without ``dense_dwdp_roots``.

    ``DwdpManager.setup`` (with expert DWDP) calls it instead of :func:`setup_dense_dwdp` on a
    rank that asks for dense DWDP (``DwdpConfig.enable_dense``) when :func:`dense_roots_available`
    is False, and so does ``DenseDwdpManager.setup`` (standalone, where every rank asks for dense
    DWDP) instead of building the dense group.  No rank makes a dense collective for such a model
    (whether it has roots is the same on every rank), so this rank raises alone and at once rather
    than wait in collective #1 for peers that do not ask for dense DWDP.  A rank that does not ask
    for it does not call it and carries on.

    Args:
        model: The loaded model, named in the message.

    Raises:
        DenseSetupError: Always.
    """
    raise DenseSetupError(
        f"dwdp_config.enable_dense asks for dense DWDP, but the decoder layers of "
        f"{type(model).__name__} do not provide dense_dwdp_roots, so it has no dense weights to "
        f"shard; set dwdp_config.enable_dense=False for this model"
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
    transfer: DenseTransfer,
    comm: GroupComm,
    *,
    dwdp_rank: int,
    dwdp_size: int,
    device_id: int,
    moe_layers: Sequence[int] | None = None,
    config: DenseConfig,
    store_factory: StoreFactory | None = None,
    ops: DeviceOps | None = None,
    device: torch.device | None = None,
    launcher_factory: LauncherFactory | None = None,
    mem_get_info: MemGetInfo | None = None,
) -> DenseDWDP | None:
    """Shard, verify and repoint the dense weights of ``model``.

    Each collective carries a :class:`PhaseReport` per rank; its local work and verdict:

    1. P1: ``layout.collect_dense_layout`` over the decoder layers with the
       ``dense_dwdp_attr_allow`` / ``dense_dwdp_exclude`` tables the layers carry, check that the
       covered layers are ``moe_layers`` (if given) and that ``free >= store + slots +
       SETUP_HEADROOM_BYTES`` (``free`` read after ``torch.cuda.empty_cache()``), then
       ``layout.compute_checksums``; no copies.  A rank that did not ask for dense DWDP does none
       of this.  Data ``(requested, digest, checksums, store_bytes, slot_bytes, free_bytes)``.
       No rank requested: return None and make no further collective.  Some did: every rank
       raises a configuration error.  All ok with equal digests and checksums: continue; else
       abandon.  The checksums catch a dense tensor that has the same shape on every rank but
       other bytes, e.g. a tensor-parallel shard.
    2. P2: allocate the store and the slots, write this rank's rows, ``synchronize``, construct
       the runtime on ``transfer.copy_stream`` (the first read of it; the runtime records the
       consume events), export.  Data: the exported handles.
    3. P3: import and map the peers, build the plans and, with the copy stream current, the
       launcher; install it, ``synchronize``.  Doubles as the barrier after every write and
       mapping.
    4. P4, Phase V: per layer ``DenseDWDP.fetch_blocking``, then ``torch.equal`` of every
       anchor's slot bytes with its ``source``.  Data: the mismatching anchor paths.
    5. P5: ``DenseBindings.repoint`` (originals alive), then per layer ``fetch_blocking`` and
       ``DenseBindings.verify_layer``.  Data: the mismatching binding paths.  The commit point:
       unless every rank is ok, every rank rolls back and abandons.
    6. P6, past the commit point: ``DenseBindings.commit`` (drop the originals; the storages
       still alive are reported, never raised), ``torch.cuda.empty_cache()``, ``DenseDWDP.commit``
       (slots filled with zeros).  Data ``(leaks, leaked_paths, strict)``.  Any rank not ok:
       every rank releases its dense state and raises :class:`~.types.DenseSetupError`; a leak
       on any rank with ``config.strict`` on any rank: every rank releases and raises
       :class:`~.types.DenseRepointError` naming the ranks and paths.  Otherwise every rank sets
       ``_dense_dwdp_active`` on the model and its decoder, logs the commit line with the moved,
       store, slot and freed bytes and the leak count, and returns the runtime.

    After collectives 2 to 4, any rank not ok means every rank abandons.

    Every phase's local work catches ``Exception`` into ``ok=False``; ``BaseException`` propagates.
    Abandoning releases everything this rank allocated: the bindings are rolled back, then the
    runtime is released (launcher, slots, store) or, if no runtime was built, the store and the
    slots; then the launcher.  Then every rank raises :class:`~.types.DenseSetupError`, with the
    model's original tensors in place, unless this rank could not restore them.  Releasing after
    the commit point detaches the committed bindings (zero-size tensors) instead and marks the
    model, which can then be neither reloaded nor set up again.  Any exception that escapes (a
    broken group, ``BaseException``) releases this rank's dense state the same way first.

    Args:
        model: The loaded model (the VL wrapper or the causal LM); the decoder is found like the
            expert setup finds it.
        transfer: Copy stream and transfer mode: with expert DWDP, ``DwdpManager``'s attached
            transfer, whose copy stream the expert DWDP weight manager then runs on; a
            ``manager.StandaloneTransfer`` standalone.
        comm: The dense group: the DWDP group, or the attention-DP group standalone.
        dwdp_rank: This rank in the dense group (its DWDP rank, or its tp rank standalone).
        dwdp_size: Dense group size ``N``.
        device_id: CUDA device of this rank.
        moe_layers: Expert DWDP MoE layers; the covered dense layers must equal them, and the
            runtime prefetches the next forward's first layers at the end of a forward
            (``DenseDWDP``'s ``prefetch_next_pass``).  None (standalone): no such check; the
            covered layers are the decoder layers whose dense roots are not empty, ascending, and
            the dense manager primes every forward.
        config: The settings; ``config.enabled`` says whether this rank asks for dense DWDP.
        store_factory: Row store constructor; ``FabricShardStore`` if None.
        ops: CUDA seam for the runtime; ``TorchCudaOps(device_id)`` if None.
        device: The dense device; ``cuda:<device_id>`` if None (unit tests pass the CPU).
        launcher_factory: Builds the copy launcher from the plans; if None,
            ``DenseCopyLauncher(plans, select_copy_mode(use_gather=transfer.use_gather,
            use_batched=transfer.use_batched_prefetch), device=device)``.
        mem_get_info: ``(free, total)`` device memory; ``torch.cuda.mem_get_info(device_id)`` if
            None.

    Returns:
        The committed runtime, or None when dense DWDP is off on every rank.

    Raises:
        DenseSetupError: Setup failed on any rank, or the ranks disagree on whether they ask for
            dense DWDP; raised on every rank of the group.
        DenseRepointError: Strict mode and an original storage survived the commit on some rank;
            raised on every rank of the group, after its dense state is released.
    """
    run = _DenseSetup(
        model,
        transfer,
        comm,
        rank=dwdp_rank,
        size=dwdp_size,
        device_id=device_id,
        device=device,
        moe_layers=moe_layers,
        config=config,
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
        transfer: DenseTransfer,
        comm: GroupComm,
        *,
        rank: int,
        size: int,
        device_id: int,
        device: torch.device | None,
        moe_layers: Sequence[int] | None,
        config: DenseConfig,
        store_factory: StoreFactory,
        ops: DeviceOps,
        launcher_factory: LauncherFactory,
        mem_get_info: MemGetInfo,
    ) -> None:
        self._model = model
        self._transfer = transfer
        self._comm = comm
        self._rank = rank
        self._size = size
        self._device_id = device_id
        self._device_override = device
        self._moe_layers = moe_layers
        self._config = config
        self._store_factory = store_factory
        self._ops = ops
        self._launcher_factory = launcher_factory
        self._mem_get_info = mem_get_info
        self._started = time.monotonic()
        self._device: torch.device | None = None
        self._decoder: nn.Module | None = None
        self._free_before = 0
        self._allocated_before = 0
        self._free_after = 0
        self._allocated_after = 0
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
        """The six collectives and their verdicts; see :func:`setup_dense_dwdp`."""
        reports = self._exchange(1, self._collect())
        data = [report.data for report in reports]
        if not any(item.requested for item in data):
            logger.debug(f"{LOG_PREFIX} dense DWDP is off on every rank")
            return None
        if not all(item.requested for item in data):
            on = [rank for rank, item in enumerate(data) if item.requested]
            off = [rank for rank, item in enumerate(data) if not item.requested]
            raise DenseSetupError(
                f"dense DWDP is enabled on {_ranks(on)} and off on {_ranks(off)} of the DWDP "
                f"group; set dwdp_config.enable_dense the same on every server of the group"
            )
        reason = _failures(reports) or _disagreement(self._layout, data)
        if reason:
            self._abandon(1, reason)
        self._log_agreement(data)

        reports = self._exchange(2, self._allocate())
        if reason := _failures(reports):
            self._abandon(2, reason)
        reports = self._exchange(3, self._map([report.data for report in reports]))
        if reason := _failures(reports):
            self._abandon(3, reason)
        reports = self._exchange(4, self._compare_blobs())
        if reason := _failures(reports):
            self._abandon(4, reason)
        reports = self._exchange(5, self._repoint())
        if reason := _failures(reports):
            self._abandon(5, reason)
        reports = self._exchange(6, self._commit())
        return self._conclude(reports)

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

    def _abandon(self, number: int, reason: str) -> NoReturn:
        """Release this rank's dense state and raise, as every rank does after collective
        ``number``."""
        intact = self._release()
        torch.cuda.empty_cache()
        message = (
            f"dense DWDP setup abandoned at collective #{number} ({_COLLECTIVES[number]}): {reason}"
        )
        if number == 6:
            message += (
                "; past the commit point the original dense weights are dropped, so the model "
                "must be loaded again"
            )
        elif not intact:
            message += (
                "; this rank could not restore its original dense weights (see the errors "
                "logged above)"
            )
        raise DenseSetupError(message)

    def _conclude(self, reports: Sequence[PhaseReport]) -> DenseDWDP:
        """The verdict of collective #6, the same on every rank.

        Raises:
            DenseSetupError: The commit failed on some rank.
            DenseRepointError: An original storage survived the commit on some rank, and some
                rank runs in strict mode.
        """
        if reason := _failures(reports):
            self._abandon(6, reason)
        commits = [report.data for report in reports]
        leakers = [rank for rank, commit in enumerate(commits) if commit.leaks]
        if leakers and any(commit.strict for commit in commits):
            self._release()
            torch.cuda.empty_cache()
            paths = [path for rank in leakers for path in commits[rank].leaked_paths]
            raise DenseRepointError(
                f"dense DWDP setup failed after collective #6 ({_COLLECTIVES[6]}): moved "
                f"storages are still alive after commit on {_ranks(leakers)}, which strict mode "
                f"forbids: {_listing(paths)}"
            )
        layout, runtime = self._layout, self._runtime
        # The runtime owns the bindings, the slots, the store and the launcher from here on.
        self._bindings = self._runtime = self._store = self._slots = self._launcher = None
        self._mark_active()
        logger.info(
            f"{LOG_PREFIX} committed: moved={_gb(layout.moved_bytes)}GB "
            f"store={_gb(layout.store_bytes)}GB slots={_gb(layout.slot_bytes)}GB "
            f"freed(mem_get_info)={_gb(self._free_after - self._free_before)}GB "
            f"freed(memory_allocated)={_gb(self._allocated_before - self._allocated_after)}GB "
            f"leaks={commits[self._rank].leaks} layers={len(layout.layers)} "
            f"segments={_segment_names(layout)} setup={time.monotonic() - self._started:.1f}s"
        )
        return runtime

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
        """P1, no copies: collection, coverage, memory precheck, checksums."""
        config = self._config
        if not config.enabled:
            return PhaseReport(True, data=_NOT_REQUESTED)
        free_bytes = 0
        try:
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
            if self._moe_layers is not None:
                _check_coverage(layout, self._moe_layers)
            _check_free_memory(layout, free_bytes)
            compute_checksums(layout)
            data = _CollectData(
                requested=True,
                digest=layout.digest,
                checksums=layout.checksums(),
                store_bytes=layout.store_bytes,
                slot_bytes=layout.slot_bytes,
                free_bytes=free_bytes,
            )
        except Exception as exc:  # a local failure must reach collective #1, never skip it
            return PhaseReport(False, _describe(exc), _CollectData(True, "", (), 0, 0, free_bytes))
        self._layout = layout
        self._free_before = free_bytes
        return PhaseReport(True, data=data)

    def _allocate(self) -> PhaseReport:
        """P2: store, slots, this rank's rows, synchronize, runtime, export."""
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
            # The first read of the copy stream, which the transfer may create here: a failure
            # is this rank's P2 report.
            self._runtime = DenseDWDP(
                layout.layers,
                layout.segment_kinds,
                self._slots,
                self._transfer.copy_stream,
                # With expert DWDP the next forward's first layers are gathered at the end of a
                # forward, beside the expert prefetch; standalone, the dense manager primes them.
                prefetch_next_pass=self._moe_layers is not None,
                ops=self._ops,
                strict=self._config.strict,
                store=store,
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
            with self._ops.stream(self._transfer.copy_stream):
                self._launcher = self._launcher_factory(plans, self._transfer, self._device)
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

    def _commit(self) -> PhaseReport:
        """P6, past the commit point, host only: drop the originals and arm the runtime.

        A leak is reported in collective #6, not raised here, so that every rank decides on it
        alike; this rank keeps owning its dense state until that verdict.
        """
        try:
            leaks = self._bindings.commit()
            torch.cuda.empty_cache()
            self._free_after, _ = self._mem_get_info()
            self._allocated_after = _allocated_bytes(self._device)
            self._runtime.commit(self._bindings)
        except Exception as exc:  # a local failure must reach collective #6, never skip it
            return PhaseReport(False, _describe(exc))
        data = _CommitData(
            leaks=len(leaks), leaked_paths=tuple(leaks[:_MAX_LISTED]), strict=self._config.strict
        )
        return PhaseReport(True, data=data)

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
    plans: Mapping[int, LayerCopyPlan], transfer: DenseTransfer, device: torch.device
) -> LayerLauncher:
    """The production launcher in the transfer's mode.

    With expert DWDP that is the transfer the expert DWDP prefetch uses; standalone, the one
    ``DwdpConfig.copy_mode`` picks.
    """
    mode = select_copy_mode(
        use_gather=transfer.use_gather, use_batched=transfer.use_batched_prefetch
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
    """The covered layers must be the expert DWDP MoE layers.

    The dense priming and the expert prefetch start with the same first two layers, and each
    layer's dense copy ``L(l)`` pairs with its expert copy ``E(l)`` on the shared copy stream.

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
            False, item.error or "sent no collective #1 data", _CollectData(True, "", (), 0, 0, 0)
        )
    if number == 6 and item.ok and not isinstance(item.data, _CommitData):
        return PhaseReport(False, "sent no collective #6 data")
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
            f"dense DWDP settings differ"
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
