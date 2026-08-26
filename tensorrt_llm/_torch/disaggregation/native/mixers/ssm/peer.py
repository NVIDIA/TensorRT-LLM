# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Dict, List, Optional, Tuple

import numpy as np

from tensorrt_llm._torch.disaggregation.base.region import (
    MemRegionGroup,
    RegionMapperBase,
    SpecRegion,
    SpecRegionPair,
)
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.page import (
    KVCachePageTable,
    MambaLayerGroup,
    PhysicalPool,
)
from tensorrt_llm._utils import nvtx_range


class MambaHeadMatchMapper(RegionMapperBase):
    """
    ---- mapper_mamba_head_match ----

    Mapper for Mamba states (SSM/Conv) when head counts match or when we only need
    to select overlapping layers.

    Input: src_regions and dst_regions each contain num_local_layers addresses
           (one address per layer, for a single slot)
    Output: transfer_layers addresses (only the overlapping layers)

    The mapper selects a subset of layers based on src_layer_offset and dst_layer_offset.
    """

    def __init__(
        self,
        transfer_layers: int,
        src_layer_off: int,
        dst_layer_off: int,
        block_bytes_per_layer: int,
    ):
        self._transfer_layers = transfer_layers
        self._src_layer_off = src_layer_off
        self._dst_layer_off = dst_layer_off
        self._block_bytes = block_bytes_per_layer

    @nvtx_range("MambaHeadMatchMapper.map")
    def map(self, src_regions: SpecRegion, dst_regions: SpecRegion) -> SpecRegionPair:
        src_group = src_regions.memory
        dst_group = dst_regions.memory

        # Select overlapping layers
        src_ptrs = src_group.ptrs[self._src_layer_off : self._src_layer_off + self._transfer_layers]
        dst_ptrs = dst_group.ptrs[self._dst_layer_off : self._dst_layer_off + self._transfer_layers]

        assert src_ptrs.size == dst_ptrs.size, (
            f"Number of regions of src({src_ptrs.size}) and dst({dst_ptrs.size}) must match"
        )

        new_src = MemRegionGroup(ptrs=src_ptrs, bytes_per_region=self._block_bytes)
        new_dst = MemRegionGroup(ptrs=dst_ptrs, bytes_per_region=self._block_bytes)
        return SpecRegionPair(
            src=SpecRegion(memory=new_src, spec=src_regions.spec),
            dst=SpecRegion(memory=new_dst, spec=dst_regions.spec),
        )


class MambaHeadMismatchMapper(RegionMapperBase):
    """
    ---- mapper_mamba_head_mismatch ----

    Mapper for Mamba SSM states when head counts differ due to different TP sizes.
    headNum and tp_size are inversely proportional.

    For SSM State: shape per layer = (nheads, head_dim, d_state)
        - bytes_per_head = head_dim * d_state * element_bytes
        - Split by head, transfer min(self_nheads, peer_nheads) contiguous heads

    NOTE: For Conv State with sectioned layout (e.g., [Q|K|V] in Qwen3Next),
    use ConvStateMismatchMapper instead.

    Input: src_regions and dst_regions each contain num_local_layers addresses
    Output: Expanded addresses with head-level granularity
    """

    def __init__(
        self,
        transfer_layers: int,
        src_layer_off: int,
        dst_layer_off: int,
        bytes_per_head: int,
        self_nheads: int,
        peer_nheads: int,
        self_tp_per_dp: int,
        peer_tp_per_dp: int,
        self_tp_rank: int,
        peer_tp_rank: int,
    ):
        self._transfer_layers = transfer_layers
        self._src_layer_off = src_layer_off
        self._dst_layer_off = dst_layer_off
        self._bytes_per_head = bytes_per_head

        # Compute contiguous heads to transfer
        self._cont_heads = min(self_nheads, peer_nheads)
        self._bytes_cont_heads = self._cont_heads * bytes_per_head

        # Compute head offsets based on TP ratio
        self._src_head_off, self._dst_head_off = _compute_tp_offsets(
            self_tp_per_dp,
            peer_tp_per_dp,
            self_tp_rank,
            peer_tp_rank,
            self._bytes_cont_heads,
        )

    @nvtx_range("MambaHeadMismatchMapper.map")
    def map(self, src_regions: SpecRegion, dst_regions: SpecRegion) -> SpecRegionPair:
        src_group = src_regions.memory
        dst_group = dst_regions.memory

        # Select overlapping layers
        src_layer_ptrs = src_group.ptrs[
            self._src_layer_off : self._src_layer_off + self._transfer_layers
        ]
        dst_layer_ptrs = dst_group.ptrs[
            self._dst_layer_off : self._dst_layer_off + self._transfer_layers
        ]
        if src_layer_ptrs.size != dst_layer_ptrs.size:
            raise ValueError(
                f"Number of layer ptrs mismatch: src={src_layer_ptrs.size}, dst={dst_layer_ptrs.size}"
            )

        # Apply head offset to each layer's address (vectorized)
        new_src_ptrs = src_layer_ptrs + self._src_head_off
        new_dst_ptrs = dst_layer_ptrs + self._dst_head_off

        new_src = MemRegionGroup(ptrs=new_src_ptrs, bytes_per_region=self._bytes_cont_heads)
        new_dst = MemRegionGroup(ptrs=new_dst_ptrs, bytes_per_region=self._bytes_cont_heads)
        return SpecRegionPair(
            src=SpecRegion(memory=new_src, spec=src_regions.spec),
            dst=SpecRegion(memory=new_dst, spec=dst_regions.spec),
        )


class ConvStateMismatchMapper(RegionMapperBase):
    """
    ---- mapper_conv_state_mismatch ----

    Mapper for conv_state when TP sizes differ.
    Handles internal sectioned structure: [Section0 | Section1 | Section2]
    where each section is independently sharded by TP.

    For example:
        Qwen3Next: [Q(ng*ds/tp) | K(ng*ds/tp) | V(d_inner/tp)]
        Mamba2:    [x(d_inner/tp) | B(ng*ds/tp) | C(ng*ds/tp)]

    When TP sizes differ (e.g., self TP=2, peer TP=4), each section
    must be mapped independently with its own byte offset and transfer size.

    Returns List[SpecRegionPair], one per section.
    Each SpecRegionPair has transfer_layers ptrs with a fixed bytes_per_region.
    """

    def __init__(
        self,
        transfer_layers: int,
        src_layer_off: int,
        dst_layer_off: int,
        self_section_bytes: List[int],
        peer_section_bytes: List[int],
        self_tp_per_dp: int,
        peer_tp_per_dp: int,
        self_tp_rank: int,
        peer_tp_rank: int,
    ):
        assert len(self_section_bytes) == len(peer_section_bytes), (
            f"Section count mismatch: self={len(self_section_bytes)}, "
            f"peer={len(peer_section_bytes)}"
        )
        self._transfer_layers = transfer_layers
        self._src_layer_off = src_layer_off
        self._dst_layer_off = dst_layer_off

        # Pre-compute per-section mapping plans:
        # [(src_offset_in_buffer, dst_offset_in_buffer, transfer_bytes), ...]
        self._section_plans = self._compute_section_plans(
            self_section_bytes,
            peer_section_bytes,
            self_tp_per_dp,
            peer_tp_per_dp,
            self_tp_rank,
            peer_tp_rank,
        )
        # Pre-compute offset arrays for vectorized map()
        self._section_src_offs = np.array([p[0] for p in self._section_plans], dtype=np.int64)
        self._section_dst_offs = np.array([p[1] for p in self._section_plans], dtype=np.int64)

    @nvtx_range("ConvStateMismatchMapper.map")
    def map(
        self,
        src_regions: SpecRegion,
        dst_regions: SpecRegion,
    ) -> List[SpecRegionPair]:
        """Returns List[SpecRegionPair], one per section."""
        src_group = src_regions.memory
        dst_group = dst_regions.memory

        # Select overlapping layers
        src_layer_ptrs = src_group.ptrs[
            self._src_layer_off : self._src_layer_off + self._transfer_layers
        ]
        dst_layer_ptrs = dst_group.ptrs[
            self._dst_layer_off : self._dst_layer_off + self._transfer_layers
        ]

        assert src_layer_ptrs.size == dst_layer_ptrs.size, (
            f"Number of layer ptrs mismatch: src={src_layer_ptrs.size}, dst={dst_layer_ptrs.size}"
        )

        # Vectorized: broadcast all section offsets at once
        # _section_offsets shape: (num_sections, 3) with (src_off, dst_off, transfer_bytes)
        src_offs = self._section_src_offs  # (num_sections,)
        dst_offs = self._section_dst_offs  # (num_sections,)

        # (num_sections, num_layers) = (num_sections, 1) + (1, num_layers)
        all_src_ptrs = src_offs[:, None] + src_layer_ptrs[None, :]
        all_dst_ptrs = dst_offs[:, None] + dst_layer_ptrs[None, :]

        results: List[SpecRegionPair] = []
        for i, (_, _, transfer_bytes) in enumerate(self._section_plans):
            results.append(
                SpecRegionPair(
                    src=SpecRegion(
                        memory=MemRegionGroup(
                            ptrs=all_src_ptrs[i],
                            bytes_per_region=transfer_bytes,
                        ),
                        spec=src_regions.spec,
                    ),
                    dst=SpecRegion(
                        memory=MemRegionGroup(
                            ptrs=all_dst_ptrs[i],
                            bytes_per_region=transfer_bytes,
                        ),
                        spec=dst_regions.spec,
                    ),
                )
            )
        return results

    @staticmethod
    def _compute_section_plans(
        self_section_bytes: List[int],
        peer_section_bytes: List[int],
        self_tp_per_dp: int,
        peer_tp_per_dp: int,
        self_tp_rank: int,
        peer_tp_rank: int,
    ) -> List[tuple]:
        """Compute (src_offset, dst_offset, transfer_bytes) for each section.

        Within each section, the mapping logic is analogous to head mismatch:
        - transfer_bytes = min(self_sec_bytes, peer_sec_bytes)
        - The rank with fewer TP ranks owns a larger chunk; the rank with
          more TP ranks selects a sub-chunk based on the TP ratio.
        """
        plans = []
        src_section_start = 0
        dst_section_start = 0

        for self_sec, peer_sec in zip(self_section_bytes, peer_section_bytes):
            transfer_bytes = min(self_sec, peer_sec)
            src_inner_off, dst_inner_off = _compute_tp_offsets(
                self_tp_per_dp,
                peer_tp_per_dp,
                self_tp_rank,
                peer_tp_rank,
                transfer_bytes,
            )
            plans.append(
                (
                    src_section_start + src_inner_off,
                    dst_section_start + dst_inner_off,
                    transfer_bytes,
                )
            )
            src_section_start += self_sec
            dst_section_start += peer_sec

        return plans


def _compute_tp_offsets(
    self_tp_per_dp: int,
    peer_tp_per_dp: int,
    self_tp_rank: int,
    peer_tp_rank: int,
    transfer_bytes: int,
) -> tuple:
    """Shared TP offset logic: the rank with more TP shards is at finer
    granularity and needs an intra-rank offset."""
    if self_tp_per_dp == peer_tp_per_dp:
        return 0, 0
    larger = max(self_tp_per_dp, peer_tp_per_dp)
    smaller = min(self_tp_per_dp, peer_tp_per_dp)
    assert larger % smaller == 0, (
        f"TP sizes must be divisible: self_tp_per_dp={self_tp_per_dp}, "
        f"peer_tp_per_dp={peer_tp_per_dp}"
    )
    ratio = larger // smaller
    if self_tp_per_dp < peer_tp_per_dp:
        # self has fewer ranks -> larger chunk; peer selects sub-chunk
        return (peer_tp_rank % ratio) * transfer_bytes, 0
    else:
        # peer has fewer ranks -> larger chunk; self selects sub-chunk
        return 0, (self_tp_rank % ratio) * transfer_bytes


class MambaPolicy:
    """
    Dispatch mappers and build frags for Mamba state transfer.
    """

    @staticmethod
    def _find_mamba_layer_group(
        page_table: Optional[KVCachePageTable],
    ) -> Optional[MambaLayerGroup]:
        if page_table is None:
            return None
        return next(
            (lg for lg in page_table.layer_groups if isinstance(lg, MambaLayerGroup)),
            None,
        )

    @staticmethod
    def validate_peer_compatible(
        self_ri: RankInfo,
        peer_ri: RankInfo,
        self_page_table: Optional[KVCachePageTable],
        peer_page_table: Optional[KVCachePageTable],
    ) -> None:
        """Validate recurrent-state (Mamba/KDA) layout compatibility with a peer.

        Analogue of the C++ ``rnnCacheFormatter inquireSupport`` gate for the
        V2 path: reject at peer-registration time instead of corrupting
        memory at transfer time. Raises ``ValueError`` naming the mismatched
        field.

        The core invariant checked is *global* (TP-aggregated) state size:
        for a TP-sharded state, ``per_rank_bytes * mamba_tp`` is
        TP-invariant, so it must match between peers even when their TP
        sizes differ. Kimi K3 KDA satisfies it: with attention-DP off the
        cache manager head-shards the state across tp_size (matching the
        model's head-sharded KDA compute), so heterogeneous ctx/gen TP
        passes; under attention-DP the state is replicated and ``_mamba_tp``
        is 1 on that side. A model that kept a replicated full-size per-rank
        state while reporting ``mamba_tp > 1`` would violate the invariant
        under heterogeneous TP — exactly the configuration where the
        TP-mismatch mappers would compute shard offsets past the end of the
        slot, silently corrupting the state — and is rejected here.
        """
        self_mlg = MambaPolicy._find_mamba_layer_group(self_page_table)
        peer_mlg = MambaPolicy._find_mamba_layer_group(peer_page_table)
        if self_mlg is None or peer_mlg is None:
            # Under pipeline parallelism each rank publishes only its own
            # stage's layers, so a hybrid model can pair a rank holding
            # recurrent layers with a peer stage holding none. The transfer
            # path (build_mamba_frags) intersects the two layer sets and
            # moves nothing when either side has no recurrent layers, so
            # there is nothing to validate for this pair.
            return

        # Layer sets are NOT required to match: with pipeline parallelism the
        # two sides may partition layers differently, and the transfer covers
        # exactly the intersection (empty intersection moves no state). The
        # invariants below are per-layer-slot quantities, uniform across a
        # model's recurrent layers, so they apply regardless of which layers
        # overlap.
        if (
            self_mlg.ssm_bytes_per_head is not None
            and peer_mlg.ssm_bytes_per_head is not None
            and self_mlg.ssm_bytes_per_head != peer_mlg.ssm_bytes_per_head
        ):
            # TP-invariant: head_dim * d_state * element_size. A mismatch
            # means different state shape or SSM cache dtype.
            raise ValueError(
                "MambaPolicy.validate_peer_compatible: ssm_bytes_per_head differs "
                f"(local={self_mlg.ssm_bytes_per_head}, peer={peer_mlg.ssm_bytes_per_head}); "
                "check head_dim / d_state / mamba_ssm_cache_dtype"
            )

        self_tp, _ = MambaPolicy._mamba_tp(self_ri)
        peer_tp, _ = MambaPolicy._mamba_tp(peer_ri)

        def _check_global(field: str, self_bytes: int, peer_bytes: int) -> None:
            if self_bytes * self_tp != peer_bytes * peer_tp:
                raise ValueError(
                    f"MambaPolicy.validate_peer_compatible: global (TP-aggregated) {field} "
                    f"differs: local {self_bytes} bytes/rank x mamba_tp={self_tp} vs "
                    f"peer {peer_bytes} bytes/rank x mamba_tp={peer_tp}. Per-rank state "
                    "sizes are inconsistent with a TP-sharded layout across the two "
                    "sides; either the state shape/dtype differs, or the model keeps a "
                    "replicated (non-TP-sharded) per-rank recurrent state while "
                    "reporting mamba_tp > 1, which supports heterogeneous ctx/gen TP "
                    "only with attention-DP enabled on both sides."
                )

        _check_global(
            "ssm slot_bytes", self_mlg.ssm_states.slot_bytes, peer_mlg.ssm_states.slot_bytes
        )
        _check_global(
            "conv slot_bytes", self_mlg.conv_states.slot_bytes, peer_mlg.conv_states.slot_bytes
        )

        if self_mlg.conv_section_bytes is not None and peer_mlg.conv_section_bytes is not None:
            if len(self_mlg.conv_section_bytes) != len(peer_mlg.conv_section_bytes):
                raise ValueError(
                    "MambaPolicy.validate_peer_compatible: conv section count differs "
                    f"(local={len(self_mlg.conv_section_bytes)}, "
                    f"peer={len(peer_mlg.conv_section_bytes)})"
                )
            for i, (s, p) in enumerate(
                zip(self_mlg.conv_section_bytes, peer_mlg.conv_section_bytes)
            ):
                _check_global(f"conv_section_bytes[{i}]", s, p)

        overlapping_layers = set(self_mlg.mamba_layer_offsets) & set(peer_mlg.mamba_layer_offsets)
        for role in sorted(set(self_mlg.side_states) | set(peer_mlg.side_states)):
            self_state = self_mlg.side_states.get(role)
            peer_state = peer_mlg.side_states.get(role)
            self_layers = set(self_state.layer_offsets) if self_state is not None else set()
            peer_layers = set(peer_state.layer_offsets) if peer_state is not None else set()
            missing_layers = overlapping_layers & (self_layers ^ peer_layers)
            if missing_layers:
                raise ValueError(
                    "MambaPolicy.validate_peer_compatible: recurrent "
                    f"side-state role {role!r} differs for overlapping layers "
                    f"{sorted(missing_layers)}"
                )
            if not (overlapping_layers & self_layers & peer_layers):
                continue
            if self_state.pool.slot_bytes != peer_state.pool.slot_bytes:
                raise ValueError(
                    "MambaPolicy.validate_peer_compatible: replicated recurrent "
                    f"side-state role {role!r} slot_bytes differs "
                    f"(local={self_state.pool.slot_bytes}, "
                    f"peer={peer_state.pool.slot_bytes})"
                )

    @staticmethod
    def _mamba_tp(ri: RankInfo) -> Tuple[int, int]:
        """Return (mamba_effective_tp_size, mamba_effective_tp_rank).

        When attention_dp is enabled, mamba is not TP-sharded.
        """
        if ri.attention and ri.attention.enable_attention_dp:
            return 1, 0
        if ri.cp_size > 1:
            # Helix repurposes CP ranks as TP for the mamba layers, so the
            # shard grid is tp*cp with CP-minor flat rank (matches Mapping).
            return ri.tp_size * ri.cp_size, ri.tp_rank * ri.cp_size + ri.cp_rank
        return ri.tp_size, ri.tp_rank

    @staticmethod
    def _build_layer_ptrs(
        pool: PhysicalPool,
        layer_offsets: Dict[int, int],
        overlapping_layers: List[int],
        slot: int,
    ) -> np.ndarray:
        """Build per-layer pointers from a pool's affine layer/slot layout."""
        slot_stride_bytes = pool.slot_stride_bytes
        layer_stride_bytes = pool.layer_stride_bytes
        assert slot_stride_bytes is not None
        assert layer_stride_bytes is not None
        ptrs = [
            pool.base_address
            + layer_offsets[global_layer_id] * layer_stride_bytes
            + slot * slot_stride_bytes
            for global_layer_id in overlapping_layers
        ]
        return np.array(ptrs, dtype=np.int64)

    @staticmethod
    def _select_mapper(
        *,
        is_conv: bool,
        tp_match: bool,
        transfer_layers: int,
        self_mlg: MambaLayerGroup,
        peer_mlg: MambaLayerGroup,
        self_pool: PhysicalPool,
        peer_pool: PhysicalPool,
        self_mamba_tp: int,
        peer_mamba_tp: int,
        self_mamba_tp_rank: int,
        peer_mamba_tp_rank: int,
    ) -> RegionMapperBase:
        """Select the appropriate mapper for a conv/ssm pool pair."""
        if tp_match:
            return MambaHeadMatchMapper(
                transfer_layers=transfer_layers,
                src_layer_off=0,
                dst_layer_off=0,
                block_bytes_per_layer=self_pool.slot_bytes,
            )
        if is_conv:
            return ConvStateMismatchMapper(
                transfer_layers=transfer_layers,
                src_layer_off=0,
                dst_layer_off=0,
                self_section_bytes=self_mlg.conv_section_bytes,
                peer_section_bytes=peer_mlg.conv_section_bytes,
                self_tp_per_dp=self_mamba_tp,
                peer_tp_per_dp=peer_mamba_tp,
                self_tp_rank=self_mamba_tp_rank,
                peer_tp_rank=peer_mamba_tp_rank,
            )
        # SSM state: head-level granularity
        self_nheads = self_pool.slot_bytes // self_mlg.ssm_bytes_per_head
        peer_nheads = peer_pool.slot_bytes // peer_mlg.ssm_bytes_per_head
        return MambaHeadMismatchMapper(
            transfer_layers=transfer_layers,
            src_layer_off=0,
            dst_layer_off=0,
            bytes_per_head=self_mlg.ssm_bytes_per_head,
            self_nheads=self_nheads,
            peer_nheads=peer_nheads,
            self_tp_per_dp=self_mamba_tp,
            peer_tp_per_dp=peer_mamba_tp,
            self_tp_rank=self_mamba_tp_rank,
            peer_tp_rank=peer_mamba_tp_rank,
        )

    @staticmethod
    def build_mamba_frags(
        self_mlg: MambaLayerGroup,
        peer_mlg: MambaLayerGroup,
        src_slot: int,
        dst_slot: int,
        self_ri: RankInfo,
        peer_ri: RankInfo,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Build (src_frags, dst_frags, kv_sizes) for mamba state transfer.

        Returns empty lists if there are no overlapping layers.
        """
        overlapping_layers = sorted(
            set(self_mlg.mamba_layer_offsets.keys()) & set(peer_mlg.mamba_layer_offsets.keys())
        )
        transfer_layers = len(overlapping_layers)
        if transfer_layers == 0:
            return [], [], []

        self_mamba_tp, self_mamba_tp_rank = MambaPolicy._mamba_tp(self_ri)
        peer_mamba_tp, peer_mamba_tp_rank = MambaPolicy._mamba_tp(peer_ri)
        tp_match = self_mamba_tp == peer_mamba_tp

        # Overlap targets are attention-domain (helix: every ctx rank
        # targets every gen rank) but the mappers below assume corresponding
        # mamba shards; unpaired senders must send nothing, or they
        # last-writer-win the receiver's slot with the wrong head range.
        if self_mamba_tp <= peer_mamba_tp:
            ratio = peer_mamba_tp // self_mamba_tp
            paired = (peer_mamba_tp_rank // ratio) == self_mamba_tp_rank
        else:
            ratio = self_mamba_tp // peer_mamba_tp
            paired = (self_mamba_tp_rank // ratio) == peer_mamba_tp_rank
        if not paired:
            return [], [], []

        src_frags: List[int] = []
        dst_frags: List[int] = []
        kv_sizes: List[int] = []

        for self_pool, peer_pool, is_conv in [
            (self_mlg.conv_states, peer_mlg.conv_states, True),
            (self_mlg.ssm_states, peer_mlg.ssm_states, False),
        ]:
            src_ptrs = MambaPolicy._build_layer_ptrs(
                self_pool,
                self_mlg.mamba_layer_offsets,
                overlapping_layers,
                src_slot,
            )
            dst_ptrs = MambaPolicy._build_layer_ptrs(
                peer_pool,
                peer_mlg.mamba_layer_offsets,
                overlapping_layers,
                dst_slot,
            )

            src_region = SpecRegion(
                memory=MemRegionGroup(ptrs=src_ptrs, bytes_per_region=self_pool.slot_bytes)
            )
            dst_region = SpecRegion(
                memory=MemRegionGroup(ptrs=dst_ptrs, bytes_per_region=peer_pool.slot_bytes)
            )

            mapper = MambaPolicy._select_mapper(
                is_conv=is_conv,
                tp_match=tp_match,
                transfer_layers=transfer_layers,
                self_mlg=self_mlg,
                peer_mlg=peer_mlg,
                self_pool=self_pool,
                peer_pool=peer_pool,
                self_mamba_tp=self_mamba_tp,
                peer_mamba_tp=peer_mamba_tp,
                self_mamba_tp_rank=self_mamba_tp_rank,
                peer_mamba_tp_rank=peer_mamba_tp_rank,
            )

            region_pair = mapper.map(src_region, dst_region)
            region_pairs = region_pair if isinstance(region_pair, list) else [region_pair]
            for rp in region_pairs:
                src_frags.extend(rp.src.memory.ptrs)
                dst_frags.extend(rp.dst.memory.ptrs)
                frag_size = rp.src.memory.bytes_per_region
                kv_sizes.extend([frag_size] * len(rp.src.memory.ptrs))

        for role in sorted(set(self_mlg.side_states) & set(peer_mlg.side_states)):
            self_state = self_mlg.side_states[role]
            peer_state = peer_mlg.side_states[role]
            side_layers = sorted(set(self_state.layer_offsets) & set(peer_state.layer_offsets))
            if not side_layers:
                continue
            if self_state.pool.slot_bytes != peer_state.pool.slot_bytes:
                raise ValueError(
                    f"Recurrent side-state role {role!r} slot_bytes differs: "
                    f"local={self_state.pool.slot_bytes}, "
                    f"peer={peer_state.pool.slot_bytes}"
                )
            src_ptrs = MambaPolicy._build_layer_ptrs(
                self_state.pool,
                self_state.layer_offsets,
                side_layers,
                src_slot,
            )
            dst_ptrs = MambaPolicy._build_layer_ptrs(
                peer_state.pool,
                peer_state.layer_offsets,
                side_layers,
                dst_slot,
            )
            src_frags.extend(src_ptrs)
            dst_frags.extend(dst_ptrs)
            kv_sizes.extend([self_state.pool.slot_bytes] * len(side_layers))

        return src_frags, dst_frags, kv_sizes

    @staticmethod
    def receiver_payload_bytes(
        sender_page_table: KVCachePageTable,
        receiver_page_table: KVCachePageTable,
        dst_slot: Optional[int],
    ) -> int:
        """Recurrent-state bytes that will land in the receiver's slot.

        Receiver-local invariant: regardless of the sender-side shard
        pairing, the slot receives exactly the receiver's own per-layer
        slot bytes over the overlapping mamba layers. A sender-side
        simulation is not usable here: RankInfoServer runs on ctx rank 0
        only, so receivers paired with any other sender would compute 0
        and under-reserve the bounce slot.
        """
        if dst_slot is None:
            return 0
        self_mlg = MambaPolicy._find_mamba_layer_group(sender_page_table)
        peer_mlg = MambaPolicy._find_mamba_layer_group(receiver_page_table)
        if self_mlg is None or peer_mlg is None:
            return 0
        overlap = set(self_mlg.mamba_layer_offsets.keys()) & set(
            peer_mlg.mamba_layer_offsets.keys()
        )
        per_layer = peer_mlg.conv_states.slot_bytes + peer_mlg.ssm_states.slot_bytes
        total = len(overlap) * int(per_layer)
        for role in set(self_mlg.side_states) & set(peer_mlg.side_states):
            peer_state = peer_mlg.side_states[role]
            side_overlap = (
                overlap
                & set(self_mlg.side_states[role].layer_offsets)
                & set(peer_state.layer_offsets)
            )
            total += len(side_overlap) * int(peer_state.pool.slot_bytes)
        return total

    @staticmethod
    def payload_bytes(
        sender_page_table: KVCachePageTable,
        receiver_page_table: KVCachePageTable,
        dst_slot: Optional[int],
        sender_ri: RankInfo,
        receiver_ri: RankInfo,
    ) -> int:
        """Total recurrent-state bytes one sender appends to the KV write for a request.

        Mirrors the sender's ``collect_frags`` call in ``_build_kv_write_meta`` with the same
        argument roles (sender as self/src), so the receiver can size a bounce region for the
        exact bytes the coalesced write will carry. Slot indices only shift pointers, never
        fragment sizes, so a dummy ``src_slot`` stands in for the sender's state index (which
        the receiver does not know at reserve time). Returns 0 when either side has no mamba
        layer group or the receiver has no state slot for the request.
        """
        if dst_slot is None:
            return 0
        _, _, sizes = MambaPolicy.collect_frags(
            self_page_table=sender_page_table,
            peer_page_table=receiver_page_table,
            src_slot=0,
            dst_slot=dst_slot,
            self_ri=sender_ri,
            peer_ri=receiver_ri,
        )
        return int(sum(sizes))

    @staticmethod
    def collect_frags(
        self_page_table: KVCachePageTable,
        peer_page_table: KVCachePageTable,
        src_slot: Optional[int],
        dst_slot: Optional[int],
        self_ri: RankInfo,
        peer_ri: RankInfo,
    ) -> Tuple[List[int], List[int], List[int]]:
        """Find mamba layer groups from page tables and build transfer frags.

        Returns (src_frags, dst_frags, kv_sizes) — all empty if not applicable.
        """
        self_mlg = next(
            (lg for lg in self_page_table.layer_groups if isinstance(lg, MambaLayerGroup)),
            None,
        )
        peer_mlg = next(
            (lg for lg in peer_page_table.layer_groups if isinstance(lg, MambaLayerGroup)),
            None,
        )
        if self_mlg is None or peer_mlg is None or src_slot is None or dst_slot is None:
            return [], [], []
        return MambaPolicy.build_mamba_frags(
            self_mlg=self_mlg,
            peer_mlg=peer_mlg,
            src_slot=src_slot,
            dst_slot=dst_slot,
            self_ri=self_ri,
            peer_ri=peer_ri,
        )
