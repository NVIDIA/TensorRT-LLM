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
    - Async P2P copies from peer MNNVL handles into WeightBuffer's remote slices
    - Bidirectional event sync (prefetch_events, consume_events) with zero CPU blocking
    - param.data binding for MoE forward pass

Design invariants:
    - NEVER calls torch.cuda.synchronize() — all sync via stream.wait_event()
    - NEVER allocates/frees memory during inference — all buffers pre-allocated
    - Only copies remote experts — local experts are zero-copy MNNVL
    - Double buffer with bidirectional events prevents RAW and WAR hazards

Timeline visualization:
    compute_stream: [forward L3] --wait(prefetch[0])--> [forward L4] --wait(prefetch[1])--> [forward L5]
    copy_stream:    --wait(consume[0])--> [P2P L4] --wait(consume[1])--> [P2P L5]
"""

from __future__ import annotations

import bisect
import logging
from typing import Dict, List, Optional, Tuple

import torch

# Try to use tensorrt_llm logger if available, otherwise use standard logging
try:
    from tensorrt_llm.logger import logger
except ImportError:
    logger = logging.getLogger(__name__)

from .specs import PeerRanges, lookup_owner
from .weight_buffer import WeightBuffer


class DWDPWeightManager:
    """Runtime P2P copy orchestrator for DWDP expert weight prefetch.

    Manages async P2P copies from peer MNNVL handles into WeightBuffer's
    remote slices using a double-buffered event protocol. All synchronization
    is done via CUDA events and stream.wait_event() — the CPU is never blocked.

    Attributes:
        weight_buffer: WeightBuffer with composite VAs for all MoE layers.
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
    ) -> None:
        """Initialize the DWDPWeightManager.

        Creates a dedicated copy stream and pre-allocates all CUDA events for
        the double-buffer protocol. Both consume_events are recorded immediately
        so the first prefetch does not stall waiting for a never-recorded event.

        Args:
            weight_buffer: WeightBuffer with composite VAs for all MoE layers.
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

        Raises:
            ValueError: If dwdp_rank is out of range or dwdp_size is invalid.
        """
        if dwdp_size <= 0:
            raise ValueError(f"dwdp_size must be positive, got {dwdp_size}")
        if dwdp_rank < 0 or dwdp_rank >= dwdp_size:
            raise ValueError(
                f"dwdp_rank must be in [0, {dwdp_size}), got {dwdp_rank}"
            )

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

        # Double-buffer events: one per buffer slot (0 and 1).
        # prefetch_events[i]: recorded on copy_stream after P2P copy into slot i.
        #   Waited on by compute_stream (RAW — must finish copy before read).
        # consume_events[i]: recorded on compute_stream after forward using slot i.
        #   Waited on by copy_stream (WAR — must finish read before overwrite).
        self._prefetch_events: List[torch.cuda.Event] = [
            torch.cuda.Event() for _ in range(2)
        ]
        self._consume_events: List[torch.cuda.Event] = [
            torch.cuda.Event() for _ in range(2)
        ]

        # Initialize consume_events as "already signaled" so the first prefetch
        # does not stall. We record them on the current (default) stream, which
        # establishes them as completed from the copy_stream's perspective.
        current_stream = torch.cuda.current_stream(device)
        for event in self._consume_events:
            event.record(current_stream)

        logger.info(
            f"[DWDPWeightManager] Initialized: rank={dwdp_rank}/{dwdp_size}, "
            f"moe_layers={len(self._moe_layer_indices)}, "
            f"weights={self._weight_names}, "
            f"experts_per_rank={self._experts_per_rank}"
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

    def next_moe_layer(self, layer_idx: int) -> Optional[int]:
        """Return the next MoE layer index after layer_idx.

        Uses binary search for O(log n) lookup.

        Args:
            layer_idx: Current layer index.

        Returns:
            The next MoE layer index strictly greater than layer_idx,
            or None if layer_idx is at or past the last MoE layer.
        """
        if not self._moe_layer_indices:
            return None
        pos = bisect.bisect_right(self._moe_layer_indices, layer_idx)
        if pos < len(self._moe_layer_indices):
            return self._moe_layer_indices[pos]
        return None

    def first_moe_layer(self) -> int:
        """Return the first MoE layer index.

        Returns:
            The smallest MoE layer index.

        Raises:
            IndexError: If there are no MoE layers.
        """
        if not self._moe_layer_indices:
            raise IndexError("No MoE layers configured")
        return self._moe_layer_indices[0]

    def prefetch_layer(self, layer_idx: int) -> None:
        """Kick off async P2P copies for all remote expert slices of a layer.

        Switches to the copy_stream, waits for the consume_event on the target
        buffer slot (WAR hazard — ensures compute has finished reading the slot),
        then issues P2P copies from peer MNNVL views into the WeightBuffer's
        remote slices. Finally, records a prefetch_event so the compute stream
        can wait for the copy to complete.

        This method returns immediately — all GPU work is enqueued on
        copy_stream and does not block the CPU.

        Args:
            layer_idx: MoE layer index to prefetch. Must be in moe_layer_indices.

        Raises:
            KeyError: If layer_idx is not a valid MoE layer.
        """
        if layer_idx not in self._moe_layer_set:
            raise KeyError(
                f"Layer {layer_idx} is not a MoE layer. "
                f"Valid layers: {self._moe_layer_indices}"
            )

        buf_idx = self._weight_buffer.buffer_index_for_layer(layer_idx)

        with torch.cuda.stream(self._copy_stream):
            # WAR hazard: wait until compute finishes reading this buffer slot
            # before overwriting it with new P2P data.
            self._copy_stream.wait_event(self._consume_events[buf_idx])

            for name in self._weight_names:
                remote_slices = self._weight_buffer.get_remote_slices(
                    layer_idx, name
                )
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
                        dst_slice[dst_offset:dst_offset + n].copy_(
                            src_tensor[local_offset:local_offset + n]
                        )
                        dst_offset += n
                        cursor = chunk_end

            # RAW signal: record event so compute_stream knows copy is done.
            self._prefetch_events[buf_idx].record(self._copy_stream)

        logger.debug(
            f"[DWDPWeightManager] Prefetch enqueued: layer={layer_idx}, "
            f"buf_slot={buf_idx}"
        )

    def wait_and_bind(
        self, backend_module: torch.nn.Module, layer_idx: int
    ) -> None:
        """Wait for prefetch completion and bind weight tensors to the module.

        On the compute stream:
            1. Wait for prefetch_events[buf_idx] (RAW — copy must finish before read).
            2. Bind each weight's full tensor (local zero-copy + remote P2P data)
               as the param.data of the backend module's corresponding attribute.
            3. Record consume_events[other_buf] (WAR signal for the OTHER buffer
               slot, telling copy_stream it is safe to overwrite that slot).

        The WAR event is recorded for the OTHER buffer slot because by the time
        this forward pass completes, the copy_stream may already be writing to
        the alternate slot for the next-next layer's prefetch.

        Design rationale — implicit compute-done signal via stream in-order semantics:
            The IPC-era predecessor of this code carried per-layer compute_events
            (O(N_moe), ~116 events for DSv3 with 58 MoE layers and ping-pong slots)
            to signal "kernel(L) finished, slot is reusable". This implementation
            replaces them with per-slot consume_events recorded inside this very
            method (4 events total, independent of N_moe).

            The simplification relies on CUDA stream in-order semantics: an event
            recorded on a stream fires only after every prior enqueue on that
            stream has completed. Because step 3 above (record consume_events[
            other_buf]) is enqueued on compute_stream AFTER step 2's binding —
            which is in turn ordered before the next kernel(L) launch on the same
            stream — the consume event for slot S cannot fire until every kernel
            that read slot S has finished. This gives the same WAR ordering as
            an explicit per-layer compute_event would, with O(1) bookkeeping.

            Implication for future work: if num_buffers ever grows beyond 2 (e.g.
            to support cross-node prefetch with deeper pipelines), per-slot events
            may need to become per-slot event LISTS, and the "next kernel is
            enqueued on the same stream" invariant must be preserved.

        Args:
            backend_module: The MoE backend module whose weight parameters will
                be rebound. Must have attributes matching self._weight_names
                (e.g., backend_module.gate_up_proj, backend_module.down_proj).
            layer_idx: MoE layer index whose weights to bind.

        Raises:
            KeyError: If layer_idx is not a valid MoE layer.
            AttributeError: If backend_module lacks a required weight attribute.
        """
        if layer_idx not in self._moe_layer_set:
            raise KeyError(
                f"Layer {layer_idx} is not a MoE layer. "
                f"Valid layers: {self._moe_layer_indices}"
            )

        buf_idx = self._weight_buffer.buffer_index_for_layer(layer_idx)
        other_buf = 1 - buf_idx

        compute_stream = torch.cuda.current_stream(
            torch.device("cuda", self._weight_buffer.device_id)
        )

        # RAW hazard: wait until P2P copy into this slot is complete
        # before the forward pass reads the data.
        compute_stream.wait_event(self._prefetch_events[buf_idx])

        # Bind full [num_experts, ...] tensors to the backend module's parameters.
        # The tensor is a composite view: local region is zero-copy MNNVL,
        # remote regions contain the freshly P2P-copied expert weights.
        for name in self._weight_names:
            full_tensor = self._weight_buffer.get_full_tensor(layer_idx, name)
            param = getattr(backend_module, name)
            param.data = full_tensor

        # WAR signal for the OTHER buffer slot: after this forward pass
        # finishes consuming the current slot, the copy_stream is free to
        # overwrite the other slot for the next-next layer's prefetch.
        self._consume_events[other_buf].record(compute_stream)

        logger.debug(
            f"[DWDPWeightManager] Bound weights: layer={layer_idx}, "
            f"buf_slot={buf_idx}, signal_consume_slot={other_buf}"
        )
