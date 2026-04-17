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

"""WeightBuffer: Composite VA layout with zero-copy local + page-pool-backed remote double buffer.

The WeightBuffer is the central component of DWDP weight management. It creates
a composite virtual address space that seamlessly combines:
    - Local region: Zero-copy mapping of the MNNVL handle (no D2D copy)
    - Remote regions (pre/post): Page pool backed buffers for P2P copy targets

Key invariant: The local expert region is NEVER the target of a memcpy. It is
always a direct cuMemMap to the MNNVL physical handle.

NOT a singleton - one instance per DWDP group. However, within a DWDP group
there is exactly one WeightBuffer managing all MoE layers for that group.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import logging

import torch

# Try to use tensorrt_llm logger if available, otherwise use standard logging
try:
    from tensorrt_llm.logger import logger
except ImportError:
    logger = logging.getLogger(__name__)

from .page_pool import PagePool, compute_slot_sizes
from .specs import EdgeInfo, LayerWeightSpecs, MnnvlHandleSet, PageAlignedLayout, WeightSpec
from .vmm import (
    align_up,
    check_cu_result,
    free_va,
    get_allocation_granularity,
    map_handle,
    reserve_va,
    set_access,
    tensor_from_ptr,
    unmap_va,
)


@dataclass
class LayerBufferState:
    """Internal state for a single layer's composite buffer.

    Attributes:
        layer_idx: Layer index.
        va_base: Base virtual address for the first weight's composite buffer
                 (kept for legacy compatibility; may be 0 if unused).
        va_size: Total size of the first weight's reserved VA region.
        layouts: Per-weight PageAlignedLayout.
        tensors: Per-weight full tensor views.
        remote_slices: Per-weight remote slice info.
        mappings: List of (va, size) for all sub-region mappings (for unmap cleanup).
        va_regions: List of (va_base, va_size) for all per-weight VA regions
                    (for free_va cleanup, one entry per weight name).
    """

    layer_idx: int
    va_base: int
    va_size: int
    layouts: Dict[str, PageAlignedLayout]
    tensors: Dict[str, torch.Tensor]
    remote_slices: Dict[str, List[Tuple[torch.Tensor, int, int]]]
    mappings: List[Tuple[int, int]]
    va_regions: List[Tuple[int, int]]


class WeightBuffer:
    """Composite VA layout with zero-copy local + page-pool-backed remote double buffer.

    The WeightBuffer does NOT copy any data. It only sets up VA mappings:
        - Local region: cuMemMap to MNNVL handle (zero-copy, no D2D)
        - Remote region: cuMemMap to page pool (P2P copy target, written by WeightManager)

    Attributes:
        local_start: First local expert index (inclusive).
        local_end: Last local expert index (exclusive).
        dwdp_size: Number of DWDP ranks.
        device_id: CUDA device ordinal.
        granularity: VMM page granularity in bytes.
    """

    __slots__ = (
        "_layer_weight_specs",
        "_handles",
        "_local_start",
        "_local_end",
        "_dwdp_size",
        "_device_id",
        "_granularity",
        "_pool_page_size",
        "_page_pool",
        "_layer_states",
        "_moe_layer_indices",
        "_released",
    )

    def __init__(
        self,
        layer_weight_specs: LayerWeightSpecs,
        handles: MnnvlHandleSet,
        local_start: int,
        local_end: int,
        dwdp_size: int,
        device_id: int,
        granularity: Optional[int] = None,
        pool_page_size: Optional[int] = None,
    ):
        """Internal constructor. Use WeightBuffer.create() factory method.

        Args:
            layer_weight_specs: Per-layer weight specifications.
            handles: MNNVL handles from Transport.
            local_start: First local expert index (inclusive).
            local_end: Last local expert index (exclusive).
            dwdp_size: Number of DWDP ranks.
            device_id: CUDA device ordinal.
            granularity: VMM page granularity. If None, queries device.
            pool_page_size: Pool page handle size in bytes. If None,
                defaults to ``PagePool.DEFAULT_PAGE_SIZE_MULTIPLIER *
                granularity`` (16MB on GB200). Must be a multiple of
                granularity.
        """
        self._layer_weight_specs = layer_weight_specs
        self._handles = handles
        self._local_start = local_start
        self._local_end = local_end
        self._dwdp_size = dwdp_size
        self._device_id = device_id

        if granularity is None:
            self._granularity = get_allocation_granularity(device_id)
        else:
            self._granularity = granularity

        if pool_page_size is None:
            self._pool_page_size = (
                PagePool.DEFAULT_PAGE_SIZE_MULTIPLIER * self._granularity
            )
        else:
            self._pool_page_size = pool_page_size

        self._page_pool: Optional[PagePool] = None
        self._layer_states: Dict[int, LayerBufferState] = {}
        self._moe_layer_indices: List[int] = sorted(layer_weight_specs.keys())
        self._released = False

    @classmethod
    def create(
        cls,
        layer_weight_specs: LayerWeightSpecs,
        handles: MnnvlHandleSet,
        local_start: int,
        local_end: int,
        dwdp_size: int,
        device_id: int,
    ) -> WeightBuffer:
        """Create all VA mappings and remote page pool.

        Steps:
            1. Compute page-aligned layout for each (layer, weight) pair
               (num_experts comes from WeightSpec.num_experts per layer)
            2. Allocate page pool for remote double buffer
            3. Create per-layer composite VA mappings
            4. Create tensor views

        Args:
            layer_weight_specs: Per-layer weight specifications
                (includes num_experts per layer).
            handles: MNNVL handles from Transport (local expert data lives here).
            local_start: First local expert index (inclusive).
            local_end: Last local expert index (exclusive).
            dwdp_size: Number of DWDP ranks.
            device_id: CUDA device ordinal.

        Returns:
            Ready-to-use WeightBuffer with all VA mappings established.

        Raises:
            ValueError: If parameters are inconsistent.
        """
        # Validate inputs
        cls._validate_inputs(layer_weight_specs, handles, local_start, local_end, dwdp_size)

        # Create instance
        buffer = cls(
            layer_weight_specs=layer_weight_specs,
            handles=handles,
            local_start=local_start,
            local_end=local_end,
            dwdp_size=dwdp_size,
            device_id=device_id,
        )

        try:
            # Step 1: Compute layouts (uses pool_page_size as pool_granularity
            # so that pre/post sizes are aligned to the pool page boundary).
            layouts = buffer._compute_all_layouts()

            # Step 2: Compute slot sizes and create page pool.
            # Because layouts already have pre/post sizes aligned to
            # pool_page_size, the slot sizes are automatically aligned too.
            buffer_slot_assignments = {
                layer_idx: buffer.buffer_index_for_layer(layer_idx)
                for layer_idx in layer_weight_specs.keys()
            }
            slot_sizes = compute_slot_sizes(layouts, buffer_slot_assignments)

            buffer._page_pool = PagePool.create(
                slot_sizes, device_id, page_size=buffer._pool_page_size
            )

            # Step 3 & 4: Create composite VA mappings and tensor views
            for layer_idx in buffer._moe_layer_indices:
                buffer._setup_layer(layer_idx, layouts[layer_idx])

            logger.info(
                f"[WeightBuffer] Created for {len(buffer._moe_layer_indices)} layers, "
                f"local experts [{local_start}, {local_end}), DWDP size {dwdp_size}"
            )

            return buffer

        except Exception:
            buffer.release()
            raise

    @staticmethod
    def _validate_inputs(
        layer_weight_specs: LayerWeightSpecs,
        handles: MnnvlHandleSet,
        local_start: int,
        local_end: int,
        dwdp_size: int,
    ) -> None:
        """Validate constructor inputs."""
        if local_start < 0:
            raise ValueError(f"local_start must be non-negative, got {local_start}")
        if local_end <= local_start:
            raise ValueError(
                f"local_end must be greater than local_start, got {local_end} <= {local_start}"
            )
        if dwdp_size <= 0:
            raise ValueError(f"dwdp_size must be positive, got {dwdp_size}")

        # Verify that every (layer_idx, name) in specs has a matching handle
        for layer_idx, weight_specs in layer_weight_specs.items():
            for name in weight_specs.keys():
                key = (layer_idx, name)
                if key not in handles.handles:
                    raise ValueError(f"Missing handle for {key}")

    def _compute_all_layouts(self) -> Dict[int, Dict[str, PageAlignedLayout]]:
        """Compute page-aligned layouts for all (layer, weight) pairs.

        Uses ``_pool_page_size`` as the ``pool_granularity`` so that pre/post
        region sizes are aligned to the pool page boundary.  This ensures
        each pool-backed region is an exact number of pool pages, eliminating
        cuMemMap overlap between pool handles and the MNNVL handle.
        """
        layouts: Dict[int, Dict[str, PageAlignedLayout]] = {}

        for layer_idx, weight_specs in self._layer_weight_specs.items():
            layouts[layer_idx] = {}

            for name, spec in weight_specs.items():
                handle_size = self._handles.get_size(layer_idx, name)

                layout = PageAlignedLayout.compute(
                    expert_bytes=spec.expert_bytes,
                    num_experts=spec.num_experts,
                    local_start=self._local_start,
                    local_end=self._local_end,
                    granularity=self._granularity,
                    handle_phys_size=handle_size,
                    pool_granularity=self._pool_page_size,
                )
                layouts[layer_idx][name] = layout

        return layouts

    def _setup_layer(
        self,
        layer_idx: int,
        weight_layouts: Dict[str, PageAlignedLayout],
    ) -> None:
        """Set up composite VA tensor views for a single layer.

        Composite VA layout per weight:

            [ pre_region | mnnvl_region | post_region ]
              (page pool)  (MNNVL handle)  (page pool)

        - pre_region  : page-pool fabric pages for remote experts [0, local_start)
        - mnnvl_region: zero-copy mapping of the MNNVL fabric handle
                        (contains local experts [local_start, local_end))
        - post_region : page-pool fabric pages for remote experts [local_end, N)

        Routing table budget (GB200):
            ~928 total entries
            - 696 used by Transport (3 peers × 4 weights × 58 layers, persistent)
            - 232 used here (1 entry per weight per layer, persistent for lifetime)
            = exactly 928 = 0 remaining.

        We achieve 1 routing entry per weight by calling set_access ONCE on the
        full composite VA (va_base, total_size).  Calling it three times
        (once per sub-region) would consume 696 entries here, exhausting the table.
        """
        weight_specs = self._layer_weight_specs[layer_idx]
        buf_slot = self.buffer_index_for_layer(layer_idx)

        layer_state = LayerBufferState(
            layer_idx=layer_idx,
            va_base=0,
            va_size=0,
            layouts=weight_layouts,
            tensors={},
            remote_slices={},
            mappings=[],
            va_regions=[],
        )

        # Track page pool offsets within the slot — each weight name allocates
        # its own pages sequentially within the slot.
        page_pool_offset = 0

        for name, layout in weight_layouts.items():
            spec = weight_specs[name]
            handle = self._handles.get_handle(layer_idx, name)

            # Reserve one contiguous VA for this weight's full composite buffer.
            va_base = reserve_va(layout.total_size, self._granularity)
            all_mappings: List[Tuple[int, int]] = []

            try:
                # --- Map pre-region (page pool pages for remote experts before local) ---
                if layout.pre_size > 0:
                    pre_mappings = self._page_pool.map_pages(
                        slot=buf_slot,
                        va_start=va_base,
                        size=layout.pre_size,
                        page_offset=page_pool_offset,
                    )
                    all_mappings.extend(pre_mappings)
                    page_pool_offset += layout.pre_pages

                # --- Map MNNVL region (zero-copy local expert data) ---
                mnnvl_va = va_base + layout.pre_size
                map_handle(mnnvl_va, layout.mnnvl_size, handle, offset=0)
                all_mappings.append((mnnvl_va, layout.mnnvl_size))

                # --- Map post-region (page pool pages for remote experts after local) ---
                if layout.post_size > 0:
                    post_va = mnnvl_va + layout.mnnvl_size
                    post_mappings = self._page_pool.map_pages(
                        slot=buf_slot,
                        va_start=post_va,
                        size=layout.post_size,
                        page_offset=page_pool_offset,
                    )
                    all_mappings.extend(post_mappings)
                    page_pool_offset += layout.post_pages

                # --- Single set_access for the ENTIRE composite VA ---
                # One cuMemSetAccess call = 1 routing table entry consumed.
                # This is critical: calling it per sub-region would consume 3
                # entries per weight per layer (696 total) and exhaust the
                # routing table on top of Transport's 696 persistent entries.
                set_access(va_base, layout.total_size, self._device_id)

                # --- Create full [num_experts, ...] tensor view ---
                # The VA layout is:
                #   va_base                            -> start of pre-region (pool)
                #   va_base + pre_size                 -> start of mnnvl (local data)
                #   va_base + pre_size + mnnvl_size    -> start of post-region (pool)
                #
                # When pool_granularity > granularity, pre_size may be larger
                # than page_start due to rounding.  The difference is
                # pre_padding bytes of unused pool-backed space at the end of
                # the pre-region.  The tensor view must start at
                # va_base + pre_padding so that expert index 0 maps to the
                # correct physical memory:
                #   expert[0]           @ va_base + pre_padding
                #   expert[local_start] @ va_base + pre_size + data_offset
                #                       = va_base + pre_padding + page_start + data_offset
                #                       = va_base + pre_padding + local_start_bytes
                tensor_start = va_base + layout.pre_padding
                full_tensor = tensor_from_ptr(
                    ptr=tensor_start,
                    shape=spec.full_shape,
                    dtype=spec.dtype,
                    device_id=self._device_id,
                )

            except Exception:
                # Best-effort cleanup on error
                for va, size in all_mappings:
                    try:
                        unmap_va(va, size)
                    except Exception:
                        pass
                try:
                    free_va(va_base, layout.total_size)
                except Exception:
                    pass
                raise

            # Accumulate into layer_state for release() cleanup:
            # - mappings: all (va, size) sub-region mappings (for unmap_va)
            # - va_regions: all (va_base, total_size) per-weight VA (for free_va)
            layer_state.mappings.extend(all_mappings)
            layer_state.va_regions.append((va_base, layout.total_size))

            # Store the full tensor and compute remote slices.
            layer_state.tensors[name] = full_tensor
            remote_slices = self._compute_remote_slices(
                full_tensor, layout, spec.num_experts
            )
            layer_state.remote_slices[name] = remote_slices

        self._layer_states[layer_idx] = layer_state

    def _create_tensor_view(
        self,
        va: int,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Create a tensor view over a VA region."""
        return tensor_from_ptr(va, shape, dtype, self._device_id)

    def _compute_remote_slices(
        self,
        full_tensor: torch.Tensor,
        layout: PageAlignedLayout,
        num_experts: int,
    ) -> List[Tuple[torch.Tensor, int, int]]:
        """Compute remote slices for P2P copy destinations.

        Returns:
            List of (slice_tensor, expert_start, expert_end) for each remote region.
        """
        slices = []

        # Pre-remote region: experts [0, local_start)
        if self._local_start > 0:
            pre_slice = full_tensor[: self._local_start]
            slices.append((pre_slice, 0, self._local_start))

        # Post-remote region: experts [local_end, num_experts)
        if self._local_end < num_experts:
            post_slice = full_tensor[self._local_end :]
            slices.append((post_slice, self._local_end, num_experts))

        return slices

    # --- Tensor access (used by WeightManager at runtime) ---

    def get_full_tensor(self, layer_idx: int, name: str) -> torch.Tensor:
        """Full [num_experts, ...] tensor view over composite VA.

        Local region [local_start:local_end] is zero-copy MNNVL.
        Remote regions are backed by page pool (P2P copy target).

        Args:
            layer_idx: Layer index.
            name: Weight name.

        Returns:
            Full tensor view.

        Raises:
            KeyError: If layer/weight not found.
            RuntimeError: If buffer has been released.
        """
        if self._released:
            raise RuntimeError("WeightBuffer has been released")
        if layer_idx not in self._layer_states:
            raise KeyError(f"Layer {layer_idx} not found")
        if name not in self._layer_states[layer_idx].tensors:
            raise KeyError(f"Weight '{name}' not found in layer {layer_idx}")
        return self._layer_states[layer_idx].tensors[name]

    def get_remote_slices(
        self, layer_idx: int, name: str
    ) -> List[Tuple[torch.Tensor, int, int]]:
        """Get (slice_tensor, expert_start, expert_end) for each remote region.

        These are P2P copy destinations. WeightManager copies peer data here.

        Args:
            layer_idx: Layer index.
            name: Weight name.

        Returns:
            List of (tensor_slice, start_idx, end_idx) for remote regions.

        Raises:
            KeyError: If layer/weight not found.
            RuntimeError: If buffer has been released.
        """
        if self._released:
            raise RuntimeError("WeightBuffer has been released")
        if layer_idx not in self._layer_states:
            raise KeyError(f"Layer {layer_idx} not found")
        if name not in self._layer_states[layer_idx].remote_slices:
            raise KeyError(f"Weight '{name}' not found in layer {layer_idx}")
        return self._layer_states[layer_idx].remote_slices[name]

    # --- Edge info (used by orchestrator for one-time setup) ---

    def get_edge_info(self, layer_idx: int, name: str) -> EdgeInfo:
        """Page-alignment edge info for setup-time peer data fill.

        Args:
            layer_idx: Layer index.
            name: Weight name.

        Returns:
            EdgeInfo with data_offset, leading_edge, trailing_edge.

        Raises:
            KeyError: If layer/weight not found.
        """
        return self.get_layout(layer_idx, name).get_edge_info()

    def get_data_offset(self, layer_idx: int, name: str) -> int:
        """Byte offset where local data starts within MNNVL handle.

        Args:
            layer_idx: Layer index.
            name: Weight name.

        Returns:
            Data offset in bytes.
        """
        return self.get_layout(layer_idx, name).data_offset

    def compute_peer_data_offset(
        self, layer_idx: int, name: str, peer_local_start: int
    ) -> int:
        """Compute data_offset for a peer rank given their local_start.

        layer_idx is required because different layers may have
        different expert_bytes, affecting page alignment.

        Args:
            layer_idx: Layer index.
            name: Weight name.
            peer_local_start: The peer rank's local_start expert index.

        Returns:
            Data offset in bytes for the peer.
        """
        layout = self.get_layout(layer_idx, name)
        peer_start_bytes = peer_local_start * layout.expert_bytes
        peer_page_start = (peer_start_bytes // self._granularity) * self._granularity
        return peer_start_bytes - peer_page_start

    # --- Layout queries (diagnostics, tests) ---

    def get_layout(self, layer_idx: int, name: str) -> PageAlignedLayout:
        """Page-aligned layout for a specific (layer, weight) pair.

        Args:
            layer_idx: Layer index.
            name: Weight name.

        Returns:
            PageAlignedLayout for this weight.

        Raises:
            KeyError: If layer/weight not found.
        """
        if layer_idx not in self._layer_states:
            raise KeyError(f"Layer {layer_idx} not found")
        if name not in self._layer_states[layer_idx].layouts:
            raise KeyError(f"Weight '{name}' not found in layer {layer_idx}")
        return self._layer_states[layer_idx].layouts[name]

    @property
    def granularity(self) -> int:
        """CUDA VMM page granularity in bytes (typically 2MB on GB200)."""
        return self._granularity

    @property
    def pool_page_size(self) -> int:
        """Pool page handle size in bytes (typically 16MB on GB200)."""
        return self._pool_page_size

    @property
    def layer_indices(self) -> List[int]:
        """MoE layer indices managed by this buffer."""
        return self._moe_layer_indices.copy()

    @property
    def local_start(self) -> int:
        """First local expert index (inclusive)."""
        return self._local_start

    @property
    def local_end(self) -> int:
        """Last local expert index (exclusive)."""
        return self._local_end

    @property
    def dwdp_size(self) -> int:
        """Number of DWDP ranks."""
        return self._dwdp_size

    @property
    def device_id(self) -> int:
        """CUDA device ordinal."""
        return self._device_id

    def buffer_index_for_layer(self, layer_idx: int) -> int:
        """Double buffer slot: 0 or 1 based on MoE order (odd/even).

        Args:
            layer_idx: Layer index.

        Returns:
            Buffer slot index (0 or 1).
        """
        # Find position of this layer in the MoE layer sequence
        if layer_idx in self._moe_layer_indices:
            moe_order = self._moe_layer_indices.index(layer_idx)
        else:
            moe_order = layer_idx

        return moe_order % 2

    def weight_names(self, layer_idx: int) -> List[str]:
        """Get weight names for a specific layer.

        Args:
            layer_idx: Layer index.

        Returns:
            List of weight names.
        """
        if layer_idx not in self._layer_weight_specs:
            raise KeyError(f"Layer {layer_idx} not found")
        return list(self._layer_weight_specs[layer_idx].keys())

    # --- Lifecycle ---

    def release(self) -> None:
        """Release all CUDA VMM resources. Idempotent. Safe from __del__."""
        if self._released:
            return

        self._released = True

        # Unmap and free all VA regions
        for layer_idx, state in self._layer_states.items():
            # First unmap all sub-region mappings (must precede free_va)
            for va, size in state.mappings:
                try:
                    unmap_va(va, size)
                except Exception as e:
                    logger.warning(f"[WeightBuffer] Failed to unmap layer {layer_idx}: {e}")

            # Free each per-weight VA region
            for va_base, va_size in state.va_regions:
                try:
                    free_va(va_base, va_size)
                except Exception as e:
                    logger.warning(
                        f"[WeightBuffer] Failed to free VA {va_base:#x} for layer {layer_idx}: {e}"
                    )

            # Legacy fallback: free va_base/va_size if va_regions is empty
            # (handles old-style single-VA LayerBufferState from older code paths)
            if not state.va_regions and state.va_base != 0:
                try:
                    free_va(state.va_base, state.va_size)
                except Exception as e:
                    logger.warning(
                        f"[WeightBuffer] Failed to free VA for layer {layer_idx}: {e}"
                    )

        self._layer_states.clear()

        # Release page pool
        if self._page_pool is not None:
            self._page_pool.release()
            self._page_pool = None

        logger.debug("[WeightBuffer] Released all resources")

    def __del__(self):
        """Clean up on destruction."""
        try:
            self.release()
        except Exception:
            pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.release()
        return False

    # --- Debug utilities ---

    def debug_info(self) -> Dict:
        """Get debug information about the buffer state.

        Returns:
            Dictionary with debug information.
        """
        info = {
            "local_range": (self._local_start, self._local_end),
            "dwdp_size": self._dwdp_size,
            "device_id": self._device_id,
            "granularity": self._granularity,
            "pool_page_size": self._pool_page_size,
            "num_layers": len(self._moe_layer_indices),
            "layer_indices": self._moe_layer_indices,
            "released": self._released,
        }

        if self._page_pool is not None:
            info["page_pool"] = {
                "page_size": self._page_pool.page_size,
                "slot0_pages": self._page_pool.num_pages(0),
                "slot1_pages": self._page_pool.num_pages(1),
                "slot0_size": self._page_pool.slot_size(0),
                "slot1_size": self._page_pool.slot_size(1),
            }

        return info
