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

"""DWDPWeightManager: Runtime P2P prefetch and weight binding orchestrator.

The WeightManager is the runtime scheduler for DWDP expert weight prefetch.
It runs INDEPENDENTLY per instance (no cross-instance sync at runtime) and
manages:
    - Async P2P copies from peer MNNVL handles into WeightBuffer's remote slices,
      in the order of a ``PrefetchSchedule`` over consecutive forward passes
    - Per-slot event sync (prefetch_events, consume_events) with zero CPU blocking
    - param.data binding for MoE forward pass

Design invariants:
    - Never calls torch.cuda.synchronize() in steady state — all sync via
      stream.wait_event(); only a forward pass abandoned inside a MoE layer is
      recovered with a device synchronization
    - NEVER allocates/frees memory during inference — all buffers pre-allocated
    - Only copies remote experts — local experts are zero-copy MNNVL
    - Per-slot prefetch/consume events prevent RAW and WAR hazards

Timeline visualization (two slots, MoE layers L3, L4, L5, L6):
    compute_stream: --wait(prefetch[0])--> [MoE L3] rec(consume[0]) [attention]
                    --wait(prefetch[1])--> [MoE L4] rec(consume[1]) [attention] ...
    copy_stream:    --wait(consume[0])--> [P2P L5] --wait(consume[1])--> [P2P L6]
"""

from __future__ import annotations

import logging
import traceback
from typing import Dict, List, Optional, Sequence, Tuple

import torch

# Try to use tensorrt_llm logger if available, otherwise use standard logging
try:
    from tensorrt_llm.logger import logger
except ImportError:
    logger = logging.getLogger(__name__)

from . import fabric_gather
from .prefetch_schedule import PrefetchSchedule, ScheduleStep
from .specs import PeerRanges, lookup_owner
from .weight_buffer import WeightBuffer

# Slice granularity for the batched prefetch path: 2 MiB.  Each peer's
# contribution to a layer is split into chunks of this size and the chunks
# are interleaved across peers in a single ``cudaMemcpyBatchAsync`` call
# so that any moment in flight touches multiple NVLink links, limiting
# single-link contention.  Ported from the IPC-era DWDP contention
# optimization (PR #12974) to the VA-based pipeline.
MEMCPY_BATCH_SLICE_BYTES = 1 << 21

# One ``cudaMemcpyBatchAsync`` batch: parallel destination-pointer,
# source-pointer and size lists, always of equal length.
_CopyBatch = Tuple[List[int], List[int], List[int]]

# Launch arguments for one layer's ``trtllm::dwdp_fabric_gather``: device-resident
# destination, source, 16-byte-line-count and group-boundary tensors, plus the
# number of blocks each group gets.
_GatherPlan = Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]


class DWDPWeightManager:
    """Runtime P2P copy orchestrator for DWDP expert weight prefetch.

    Manages async P2P copies from peer MNNVL handles into WeightBuffer's
    remote slices.  A ``PrefetchSchedule`` decides when each copy is issued;
    every slot has a prefetch (RAW) and a consume (WAR) event.  All
    synchronization is done via CUDA events and stream.wait_event() — the CPU
    is never blocked.

    Runtime protocol: ``start_forward`` before a forward pass, then per MoE
    layer ``wait_and_bind`` before its MoE and ``finish_layer`` right after
    the MoE is enqueued; ``invalidate_prefetch`` after the source expert
    weights changed.

    Attributes:
        weight_buffer: WeightBuffer with composite VA layout for all MoE layers.
        dwdp_rank: This instance's DWDP rank (0..dwdp_size-1).
        dwdp_size: Total number of DWDP ranks.
    """

    __slots__ = (
        "_weight_buffer",
        "_peer_views",
        "_peer_ranges",
        "_moe_layer_indices",
        "_moe_layer_set",
        "_weight_names",
        "_dwdp_rank",
        "_dwdp_size",
        "_experts_per_rank",
        "_copy_stream",
        "_prefetch_events",
        "_consume_events",
        "_transport",
        "_use_batched_prefetch",
        "_batched_copy_plans",
        "_use_gather",
        "_gather_plans",
        "_layer_order",
        "_schedule",
        "_reported_restarts",
        "_released",
    )

    def __init__(
        self,
        weight_buffer: WeightBuffer,
        peer_views: Dict[Tuple[int, int, str], torch.Tensor],
        peer_ranges: PeerRanges,
        moe_layer_indices: List[int],
        weight_names: List[str],
        dwdp_rank: int,
        dwdp_size: int,
        contention_opt: bool = False,
        repeated_layer_indices: Sequence[int] = (),
    ) -> None:
        """Initialize the DWDPWeightManager.

        Creates a dedicated copy stream and pre-allocates the CUDA events of
        every buffer slot. The consume_events are recorded immediately so the
        first prefetch does not stall waiting for a never-recorded event.

        Args:
            weight_buffer: WeightBuffer with composite VA layout for all MoE layers.
            peer_views: Mapping of (peer_rank, layer_idx, weight_name) to tensor
                views into peer's MNNVL handle. These are IMMUTABLE — the source
                GPU never writes to them during inference.
            peer_ranges: Per-peer ``(local_start, local_end_capped)`` tuples
                indexed by DWDP rank.  ``prefetch_layer`` resolves the owner of
                a remote expert id with ``lookup_owner(expert_id, peer_ranges)``,
                which handles non-uniform partition (tail-rank padding) and
                redundancy (overlapping ranges) uniformly.
            moe_layer_indices: Sorted list of decoder layer indices that are MoE
                layers (e.g., layers 3..60 in a typical MoE model where the first
                few layers are dense).
            weight_names: Weight parameter names to prefetch
                (e.g., ["gate_up_proj", "down_proj"]).
            dwdp_rank: This instance's DWDP rank (0..dwdp_size-1).
            dwdp_size: Total number of DWDP ranks.
            contention_opt: If True, ``prefetch_layer`` issues a single
                ``cudaMemcpyBatchAsync`` per layer with sub-slices interleaved
                across peers in ``MEMCPY_BATCH_SLICE_BYTES`` (2 MiB) chunks
                to reduce single-NVLink-link contention.  Plans are cached
                per layer since peer ``data_ptr()`` and destination
                ``data_ptr()`` are stable after initialization.
            repeated_layer_indices: MoE layers that run several times in a row
                within a forward pass; their repeats bind the copy still in
                the slot (see ``PrefetchSchedule``).

        Raises:
            ValueError: If dwdp_rank is out of range, dwdp_size is invalid,
                there are no MoE layers or a repeated layer is not one of them.
        """
        if dwdp_size <= 0:
            raise ValueError(f"dwdp_size must be positive, got {dwdp_size}")
        if dwdp_rank < 0 or dwdp_rank >= dwdp_size:
            raise ValueError(f"dwdp_rank must be in [0, {dwdp_size}), got {dwdp_rank}")

        self._weight_buffer = weight_buffer
        self._peer_views = peer_views
        self._peer_ranges = peer_ranges
        self._moe_layer_indices = sorted(moe_layer_indices)
        self._moe_layer_set = set(self._moe_layer_indices)
        self._weight_names = list(weight_names)
        self._dwdp_rank = dwdp_rank
        self._dwdp_size = dwdp_size

        # Storage chunk size of every peer's MNNVL handle.  Uniform across
        # ranks (Phase 1 made ``num_experts_per_worker`` the storage size for
        # every rank) — the variability lives in the *valid* sub-range
        # tracked by ``_peer_ranges``.
        self._experts_per_rank = weight_buffer.local_end - weight_buffer.local_start

        # Dedicated CUDA stream for P2P copy operations.
        # Using a separate stream allows copy/compute overlap.
        device = torch.device("cuda", weight_buffer.device_id)
        self._copy_stream = torch.cuda.Stream(device=device)

        num_slots = weight_buffer.num_slots
        self._layer_order = {layer: order for order, layer in enumerate(self._moe_layer_indices)}
        # The weight buffer owns the slot mapping; the schedule and the event
        # indices below both read it from there.
        unknown = set(repeated_layer_indices) - self._moe_layer_set
        if unknown:
            raise ValueError(f"repeated layers {sorted(unknown)} are not DWDP MoE layers")
        self._schedule = PrefetchSchedule(
            [weight_buffer.buffer_index_for_layer(layer) for layer in self._moe_layer_indices],
            repeated_orders=[self._layer_order[layer] for layer in repeated_layer_indices],
        )
        self._reported_restarts = 0

        # Buffer-slot events: one pair per slot.
        # prefetch_events[i]: recorded on copy_stream after P2P copy into slot i.
        #   Waited on by compute_stream (RAW — must finish copy before read).
        # consume_events[i]: recorded on compute_stream after the MoE using slot i.
        #   Waited on by copy_stream (WAR — must finish read before overwrite).
        self._prefetch_events: List[torch.cuda.Event] = [
            torch.cuda.Event() for _ in range(num_slots)
        ]
        self._consume_events: List[torch.cuda.Event] = [
            torch.cuda.Event() for _ in range(num_slots)
        ]
        self._transport = None

        # Initialize consume_events as "already signaled" so the first prefetch
        # does not stall. We record them on the current (default) stream, which
        # establishes them as completed from the copy_stream's perspective.
        current_stream = torch.cuda.current_stream(device)
        for event in self._consume_events:
            event.record(current_stream)

        # Batched contention-optimized prefetch.  Plans are stable after
        # initialization (peer data pointers + destination slice pointers
        # do not move during inference) so we cache one ``(dst_ptrs,
        # src_ptrs, sizes)`` tuple per layer to avoid host-side rebuild on
        # every prefetch.
        self._use_batched_prefetch = contention_opt
        self._batched_copy_plans: Dict[int, _CopyBatch] = {}
        self._released = False

        # SM gather: the same per-peer plan, pulled by a kernel instead of the copy
        # engine (see ``fabric_gather``).  Opt-in; off if the op is not registered.
        # Layers whose descriptors are not 16-byte aligned cache None and fall back
        # to the copy engine.
        self._use_gather = fabric_gather.available()
        if self._use_gather:
            # Pin the dense GEMM to the CLC scheduler and turn off cooperative MoE
            # routing unless the user set them explicitly.  Runs during setup, before
            # the first forward.  The persistent grids' SM reserve is set by DwdpManager.
            fabric_gather.apply_compute_side_defaults()
        if fabric_gather.is_enabled() and not self._use_gather:
            logger.warning(
                "[DWDPWeightManager] Fabric gather requested but trtllm::dwdp_fabric_gather is "
                "not registered; falling back to the copy-engine prefetch"
            )
        self._gather_plans: Dict[int, Optional[_GatherPlan]] = {}

        logger.info(
            f"[DWDPWeightManager] Initialized: rank={dwdp_rank}/{dwdp_size}, "
            f"moe_layers={len(self._moe_layer_indices)}, "
            f"weights={self._weight_names}, "
            f"experts_per_rank={self._experts_per_rank}, "
            f"contention_opt={contention_opt}, "
            f"gather={self._use_gather}"
        )

    @property
    def weight_buffer(self) -> WeightBuffer:
        """The underlying WeightBuffer."""
        return self._weight_buffer

    @property
    def dwdp_rank(self) -> int:
        """This instance's DWDP rank."""
        return self._dwdp_rank

    @property
    def dwdp_size(self) -> int:
        """Total number of DWDP ranks."""
        return self._dwdp_size

    def is_moe_layer(self, layer_idx: int) -> bool:
        """Check if a given layer index is a MoE layer.

        Args:
            layer_idx: Decoder layer index to check.

        Returns:
            True if layer_idx is a MoE layer managed by this instance.
        """
        return layer_idx in self._moe_layer_set

    def prefetch_layer(self, layer_idx: int) -> None:
        """Kick off async P2P copies for all remote expert slices of a layer.

        Called for the copies the prefetch schedule issues. Switches to the
        copy_stream, waits for the consume_event on the target buffer slot
        (WAR hazard — ensures compute has finished reading the slot),
        then issues P2P copies from peer MNNVL views into the WeightBuffer's
        remote slices. Finally, records a prefetch_event so the compute stream
        can wait for the copy to complete.

        Three transports share the per-peer plan: the default per-slice path
        (one ``torch.Tensor.copy_`` per chunk), ``DwdpConfig.contention_opt``
        (one ``cudaMemcpyBatchAsync`` per layer, 2 MiB sub-slices interleaved
        across peers to limit single-link contention) and
        ``TRTLLM_DWDP_PREFETCH_GATHER`` (one SM kernel per layer; see ``fabric_gather``).

        This method returns immediately — all GPU work is enqueued on
        copy_stream and does not block the CPU.

        Args:
            layer_idx: MoE layer index to prefetch. Must be in moe_layer_indices.

        Raises:
            KeyError: If layer_idx is not a valid MoE layer.
        """
        if layer_idx not in self._moe_layer_set:
            raise KeyError(
                f"Layer {layer_idx} is not a MoE layer. Valid layers: {self._moe_layer_indices}"
            )

        buf_idx = self._schedule.slot(self._order_of(layer_idx))

        with torch.cuda.stream(self._copy_stream):
            # WAR hazard: wait until compute finishes reading this buffer slot
            # before overwriting it with new P2P data.
            self._copy_stream.wait_event(self._consume_events[buf_idx])

            if self._use_gather:
                self._prefetch_layer_gather(layer_idx)
            elif self._use_batched_prefetch:
                self._prefetch_layer_batched(layer_idx)
            else:
                self._prefetch_layer_per_slice(layer_idx)

            # RAW signal: record event so compute_stream knows copy is done.
            self._prefetch_events[buf_idx].record(self._copy_stream)

        logger.debug(
            f"[DWDPWeightManager] Prefetch enqueued: layer={layer_idx}, buf_slot={buf_idx}"
        )

    def _prefetch_layer_per_slice(self, layer_idx: int) -> None:
        """Default prefetch: one ``torch.Tensor.copy_`` per peer chunk."""
        for name in self._weight_names:
            remote_slices = self._weight_buffer.get_remote_slices(layer_idx, name)
            for dst_slice, expert_start, expert_end in remote_slices:
                # A remote slice may span multiple peer ranks.  Walk
                # the destination range and dispatch each contiguous
                # sub-chunk to the peer that owns it.  ``lookup_owner``
                # picks the lowest-rank owner (matters under redundancy
                # where multiple peers cover the same expert id).
                cursor = expert_start
                dst_offset = 0
                while cursor < expert_end:
                    peer_rank = lookup_owner(cursor, self._peer_ranges)
                    peer_local_start, peer_local_end = self._peer_ranges[peer_rank]
                    local_offset = cursor - peer_local_start
                    # Stop at the first of: end of the destination
                    # remote slice, or end of this peer's valid range.
                    # Crossing the latter means the next expert is
                    # owned by a different peer (or by *us*, which
                    # would be a bug since this is the *remote* slice).
                    chunk_end = min(expert_end, peer_local_end)
                    n = chunk_end - cursor

                    peer_key = (peer_rank, layer_idx, name)
                    src_tensor = self._peer_views[peer_key]
                    dst_slice[dst_offset : dst_offset + n].copy_(
                        src_tensor[local_offset : local_offset + n]
                    )
                    dst_offset += n
                    cursor = chunk_end

    def _build_peer_descriptor_groups(self, layer_idx: int) -> List[List[Tuple[int, int, int]]]:
        """Build a per-layer copy plan, grouped by peer.

        Returns:
            Coalesced ``(dst_ptr, src_ptr, size_bytes)`` descriptors per non-empty peer, ascending.
        """
        # Step 1: collect per-peer ``(dst_ptr, src_ptr, total_size)`` triples.
        peer_triples: Dict[int, List[Tuple[int, int, int]]] = {}
        for name in self._weight_names:
            remote_slices = self._weight_buffer.get_remote_slices(layer_idx, name)
            for dst_slice, expert_start, expert_end in remote_slices:
                cursor = expert_start
                dst_offset = 0
                while cursor < expert_end:
                    peer_rank = lookup_owner(cursor, self._peer_ranges)
                    peer_local_start, peer_local_end = self._peer_ranges[peer_rank]
                    local_offset = cursor - peer_local_start
                    chunk_end = min(expert_end, peer_local_end)
                    n = chunk_end - cursor

                    peer_key = (peer_rank, layer_idx, name)
                    src_tensor = self._peer_views[peer_key]
                    dst_sub = dst_slice[dst_offset : dst_offset + n]
                    src_sub = src_tensor[local_offset : local_offset + n]
                    size_bytes = dst_sub.numel() * dst_sub.element_size()

                    peer_triples.setdefault(peer_rank, []).append(
                        (dst_sub.data_ptr(), src_sub.data_ptr(), size_bytes)
                    )

                    dst_offset += n
                    cursor = chunk_end

        # Step 2: merge triples adjacent in both address spaces (only fragmented
        # remote slices produce more than one triple per peer and weight).
        coalesced: Dict[int, List[Tuple[int, int, int]]] = {}
        for peer, triples in peer_triples.items():
            merged: List[Tuple[int, int, int]] = []
            for dst_ptr, src_ptr, size in sorted(triples, key=lambda t: (t[0], t[1])):
                if merged:
                    prev_dst, prev_src, prev_size = merged[-1]
                    if prev_dst + prev_size == dst_ptr and prev_src + prev_size == src_ptr:
                        merged[-1] = (prev_dst, prev_src, prev_size + size)
                        continue
                merged.append((dst_ptr, src_ptr, size))
            coalesced[peer] = merged

        # Step 3: one descriptor group per peer, peers ascending.
        return [coalesced[peer] for peer in sorted(coalesced) if coalesced[peer]]

    def _build_batched_prefetch_copy_plan(self, layer_idx: int) -> _CopyBatch:
        """Build a per-layer ``cudaMemcpyBatchAsync`` plan.

        Strategy: take the per-peer descriptor groups, split each peer's
        contiguous chunks into ``MEMCPY_BATCH_SLICE_BYTES`` (2 MiB) sub-slices
        and interleave them across peers in round-robin order.  Interleaving
        is what limits single-NVLink-link contention — at any moment the
        in-flight memcpys touch every peer link rather than saturating one.

        Returns:
            Three parallel lists ``(dst_ptrs, src_ptrs, sizes)`` of the
            same length, suitable for ``cudaMemcpyBatchAsync``.
        """
        # Step 1: split each peer's descriptors into 2 MiB sub-slices.
        peer_subslices: List[List[Tuple[int, int, int]]] = []
        for group in self._build_peer_descriptor_groups(layer_idx):
            sub_list: List[Tuple[int, int, int]] = []
            for dst_ptr, src_ptr, total in group:
                offset = 0
                while offset < total:
                    sz = min(MEMCPY_BATCH_SLICE_BYTES, total - offset)
                    sub_list.append((dst_ptr + offset, src_ptr + offset, sz))
                    offset += sz
            peer_subslices.append(sub_list)

        # Step 2: round-robin across peers.  At each round, take the next
        # sub-slice from every peer that still has one, then advance.
        # Peers naturally drop out as their sub-slice lists exhaust.
        dst_ptrs: List[int] = []
        src_ptrs: List[int] = []
        sizes: List[int] = []
        cursors = {peer: 0 for peer in range(len(peer_subslices))}
        while cursors:
            for peer in list(cursors):
                idx = cursors[peer]
                if idx < len(peer_subslices[peer]):
                    d, s, sz = peer_subslices[peer][idx]
                    dst_ptrs.append(d)
                    src_ptrs.append(s)
                    sizes.append(sz)
                    cursors[peer] = idx + 1
                else:
                    del cursors[peer]

        return dst_ptrs, src_ptrs, sizes

    def _prefetch_layer_batched(self, layer_idx: int) -> None:
        """Batched prefetch: one ``cudaMemcpyBatchAsync`` per layer."""
        plan = self._batched_copy_plans.get(layer_idx)
        if plan is None:
            plan = self._build_batched_prefetch_copy_plan(layer_idx)
            self._batched_copy_plans[layer_idx] = plan

        dst_ptrs, src_ptrs, sizes = plan
        if not dst_ptrs:
            return

        # Imported lazily because ``cuda.bindings.runtime`` is a heavy
        # binding and the default per-slice path does not need it.
        from cuda.bindings import runtime as cudart

        attr = cudart.cudaMemcpyAttributes()
        attr.srcAccessOrder = cudart.cudaMemcpySrcAccessOrder.cudaMemcpySrcAccessOrderStream
        (err,) = cudart.cudaMemcpyBatchAsync(
            dst_ptrs,
            src_ptrs,
            sizes,
            len(dst_ptrs),
            [attr],
            [0],
            1,
            self._copy_stream.cuda_stream,
        )
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(
                f"cudaMemcpyBatchAsync failed for DWDP prefetch (layer="
                f"{layer_idx}, copies={len(dst_ptrs)}): {err}"
            )

    def _build_gather_plan(self, layer_idx: int) -> Optional[_GatherPlan]:
        """Build the gather launch arguments, or None if the layer is empty or not 16-byte aligned."""
        groups = self._build_peer_descriptor_groups(layer_idx)
        if not groups:
            return None
        return fabric_gather.build_plan(groups, self._copy_stream.device)

    def _prefetch_layer_gather(self, layer_idx: int) -> None:
        """SM-gather prefetch: one kernel per layer; misaligned layers fall back to the copy engine."""
        if layer_idx in self._gather_plans:
            plan = self._gather_plans[layer_idx]
        else:
            plan = self._build_gather_plan(layer_idx)
            self._gather_plans[layer_idx] = plan
            if plan is None:
                logger.warning(
                    f"[DWDPWeightManager] Layer {layer_idx} is not 16-byte aligned; "
                    f"using the copy-engine prefetch for it"
                )

        if plan is None:
            self._prefetch_layer_batched(layer_idx)
            return

        dst_ptrs, src_ptrs, lines, row_starts, blocks = plan
        fabric_gather.gather_op()(
            dst_ptrs, src_ptrs, lines, row_starts, blocks, fabric_gather.THREADS
        )

    def start_forward(self) -> None:
        """Prefetch the first MoE layers of the coming forward pass.

        Enqueues nothing when the previous forward pass already prefetched
        them.
        """
        self._apply(self._schedule.start_forward())

    def wait_and_bind(self, backend_module: torch.nn.Module, layer_idx: int) -> None:
        """Wait for prefetch completion and bind weight tensors to the module.

        Issues the layer's copy first if nothing did (a forward pass without
        ``start_forward``, or one that restarts the prefetch sequence). Then,
        on the compute stream:
            1. Wait for prefetch_events[buf_idx] (RAW — copy must finish before read).
            2. Bind each weight's full tensor (local zero-copy + remote P2P data)
               as the param.data of the backend module's corresponding attribute.

        The matching WAR signal is deliberately recorded by ``finish_layer``
        after the MoE kernel has been enqueued. Recording it here would place the
        event before the kernel on the compute stream and could let the copy
        stream overwrite a slot while that kernel is still reading it.

        Args:
            backend_module: The MoE backend module whose weight parameters will
                be rebound. Must have attributes matching self._weight_names
                (e.g., backend_module.gate_up_proj, backend_module.down_proj).
            layer_idx: MoE layer index whose weights to bind.

        Raises:
            KeyError: If layer_idx is not a valid MoE layer.
            AttributeError: If backend_module lacks a required weight attribute.
        """
        order = self._order_of(layer_idx)
        self._apply(self._schedule.bind(order))

        buf_idx = self._schedule.slot(order)
        compute_stream = torch.cuda.current_stream(
            torch.device("cuda", self._weight_buffer.device_id)
        )

        # RAW hazard: wait until P2P copy into this slot is complete
        # before the forward pass reads the data.  The schedule issues no
        # other copy into the slot until this layer finishes, so the last
        # recorded prefetch event of the slot is this layer's.
        compute_stream.wait_event(self._prefetch_events[buf_idx])

        # Bind full [num_experts, ...] tensors to the backend module's parameters.
        # The tensor is a composite view: local region is zero-copy MNNVL,
        # remote regions contain the freshly P2P-copied expert weights.
        for name in self._weight_names:
            full_tensor = self._weight_buffer.get_full_tensor(layer_idx, name)
            param = getattr(backend_module, name)
            param.data = full_tensor

        logger.debug(f"[DWDPWeightManager] Bound weights: layer={layer_idx}, buf_slot={buf_idx}")

    def finish_layer(self, layer_idx: int) -> None:
        """Release the layer's buffer slot and issue the copies this unblocks.

        This method must be called after the layer's MoE kernel is enqueued on
        the current compute stream. CUDA stream ordering then guarantees that
        the consume event cannot complete until the kernel has stopped reading
        the slot. The copy stream waits on this event before reusing the slot;
        the last MoE layers of a forward pass issue the copies of the first
        layers of the next one.

        Args:
            layer_idx: MoE layer whose compute was just enqueued.

        Raises:
            KeyError: If layer_idx is not a valid MoE layer.
            RuntimeError: If layer_idx is not the bound layer.
        """
        self._apply(self._schedule.finish(self._order_of(layer_idx)))

    def invalidate_prefetch(self) -> None:
        """Drop the copies issued ahead of the next forward pass.

        Call after the source expert weights changed; the next forward pass
        gathers its first layers again.
        """
        self._apply(self._schedule.invalidate())

    def _order_of(self, layer_idx: int) -> int:
        order = self._layer_order.get(layer_idx)
        if order is None:
            raise KeyError(
                f"Layer {layer_idx} is not a MoE layer. Valid layers: {self._moe_layer_indices}"
            )
        return order

    def _apply(self, step: ScheduleStep) -> None:
        """Enqueue one schedule step: device sync if needed, slot release, then copies."""
        if step.synchronize:
            logger.warning(
                "[DWDPWeightManager] Prefetch sequence restarted after an unfinished forward "
                "pass; synchronizing the device before reusing its buffer slots"
            )
            torch.cuda.synchronize(self._weight_buffer.device_id)
        if step.release is not None:
            compute_stream = torch.cuda.current_stream(
                torch.device("cuda", self._weight_buffer.device_id)
            )
            self._consume_events[step.release].record(compute_stream)
        restarts = self._schedule.num_restarts
        if restarts != self._reported_restarts:
            self._reported_restarts = restarts
            # Power-of-two counts only: a model that keeps breaking the sequence still shows up.
            if restarts & (restarts - 1) == 0:
                logger.warning(
                    f"[DWDPWeightManager] Prefetch sequence restarted ({restarts} restart(s) so "
                    f"far): a forward pass did not run the MoE layers in order"
                )
        for order in step.prefetch:
            self.prefetch_layer(self._moe_layer_indices[order])

    def release(self) -> None:
        """Synchronize outstanding work and release all owned VMM resources."""
        if self._released:
            return

        # Teardown is outside the inference hot path. A device synchronization
        # makes both compute and copy-stream users of the composite VA quiescent
        # before unmapping it.
        torch.cuda.synchronize(self._weight_buffer.device_id)
        self._peer_views.clear()
        try:
            self._weight_buffer.release()
        finally:
            # The transport owns MNNVL handles that outlive the process, so free
            # them even when the buffer release raises; the buffer error stays in
            # flight. Mark released either way so __del__ does not retry a call
            # that just failed.
            if self._transport is not None:
                try:
                    self._transport.release()
                except Exception:  # noqa: BLE001 - must not mask the buffer error
                    logger.error(
                        f"[DWDPWeightManager] Transport release failed\n{traceback.format_exc()}"
                    )
                self._transport = None
            self._batched_copy_plans.clear()
            # Gather plans hold raw pointers into the VA just unmapped.
            self._gather_plans.clear()
            self._released = True

    def __del__(self) -> None:
        """Best-effort fallback for callers that miss explicit teardown."""
        try:
            self.release()
        except Exception as exc:
            logger.debug(f"[DWDPWeightManager] __del__ release failed: {exc!r}")
