from typing import List

import numpy as np

from tensorrt_llm._torch.disaggregation.base.region import (
    MemRegionGroup,
    RegionMapperBase,
    SpecRegion,
    SpecRegionPair,
)
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo


class IdentityMapper(RegionMapperBase):
    """
    ---- mapper_identity ----

    Pass-through mapping. Do not change pointers or sizes.

    src_ptrs: [ S0 ] [ S1 ] [ S2 ] ...
                |      |      |
                v      v      v
    dst_ptrs: [ D0 ] [ D1 ] [ D2 ] ...
    """

    def map(self, src_regions: SpecRegion, dst_regions: SpecRegion) -> SpecRegionPair:
        src_group = src_regions.memory
        dst_group = dst_regions.memory
        assert len(src_group.ptrs) == len(dst_group.ptrs), (
            f"Number of regions of src({len(src_group.ptrs)}) and dst({len(dst_group.ptrs)}) must match"
        )
        new_src = MemRegionGroup(
            ptrs=list(src_group.ptrs), bytes_per_region=src_group.bytes_per_region
        )
        new_dst = MemRegionGroup(
            ptrs=list(dst_group.ptrs), bytes_per_region=dst_group.bytes_per_region
        )
        return SpecRegionPair(
            src=SpecRegion(memory=new_src, spec=src_regions.spec),
            dst=SpecRegion(memory=new_dst, spec=dst_regions.spec),
        )


class HeadMatchMapper(RegionMapperBase):
    """
    ---- mapper_head_match ----

    Move/copy entire contiguous block(s) (multi-layer fragment) as a single chunk.
    Align by whole fragment size (frag_size) and apply a constant source/destination block offset.

    src_ptrs:  [ S0 ]         [ S1 ]          ...
                 |              |
              + src_off      + src_off
                 |              |
          [ S0 + src_off ] [ S1 + src_off ]   ->  (each points to a frag of size frag_size)
                   copy whole frag
                 |              |
                 v              v
          [ D0 + dst_off ] [ D1 + dst_off ]   ->  (destination frags)
    """

    def __init__(
        self,
        transfer_layers: int,
        src_layer_off: int,
        dst_layer_off: int,
        self_ri: RankInfo,
        peer_ri: RankInfo,
    ):
        self._kv_factor = self_ri.kv_factor
        self._frag_size = self._block_size(transfer_layers, self_ri)
        self._src_block_off = self._block_size(src_layer_off, self_ri)
        self._dst_block_off = self._block_size(dst_layer_off, peer_ri)

    def map(self, src_regions: SpecRegion, dst_regions: SpecRegion) -> SpecRegionPair:
        src_group = src_regions.memory
        dst_group = dst_regions.memory
        assert len(src_group.ptrs) == len(dst_group.ptrs), (
            f"Number of regions of src({len(src_group.ptrs)}) and dst({len(dst_group.ptrs)}) must match"
        )
        new_src_ptrs = [src_ptr + self._src_block_off for src_ptr in src_group.ptrs]
        new_dst_ptrs = [dst_ptr + self._dst_block_off for dst_ptr in dst_group.ptrs]
        new_src = MemRegionGroup(ptrs=new_src_ptrs, bytes_per_region=self._frag_size)
        new_dst = MemRegionGroup(ptrs=new_dst_ptrs, bytes_per_region=self._frag_size)
        return SpecRegionPair(
            src=SpecRegion(memory=new_src, spec=src_regions.spec),
            dst=SpecRegion(memory=new_dst, spec=dst_regions.spec),
        )

    def _block_size(self, layer_num: int, ri: RankInfo) -> int:
        return (
            layer_num
            * ri.kv_factor
            * ri.kv_heads_per_rank
            * ri.tokens_per_block
            * ri.dims_per_head
            * ri.element_bytes
        )


class IndexerKCacheHeadMatchMapper(RegionMapperBase):
    """
    ---- mapper_head_match ----

    Move/copy entire contiguous block(s) (multi-layer fragment) as a single chunk.
    Align by whole fragment size (frag_size) and apply a constant source/destination block offset.

    src_ptrs:  [ S0 ]         [ S1 ]          ...
                 |              |
              + src_off      + src_off
                 |              |
          [ S0 + src_off ] [ S1 + src_off ]   ->  (each points to a frag of size frag_size)
                   copy whole frag
                 |              |
                 v              v
          [ D0 + dst_off ] [ D1 + dst_off ]   ->  (destination frags)
    """

    def __init__(
        self,
        transfer_layers: int,
        src_layer_off: int,
        dst_layer_off: int,
        self_ri: RankInfo,
        peer_ri: RankInfo,
        block_size_per_layer: int,
    ):
        self._frag_size = block_size_per_layer * transfer_layers
        self._src_block_off = block_size_per_layer * src_layer_off
        self._dst_block_off = block_size_per_layer * dst_layer_off

    def map(self, src_regions: SpecRegion, dst_regions: SpecRegion) -> SpecRegionPair:
        src_group = src_regions.memory
        dst_group = dst_regions.memory
        assert len(src_group.ptrs) == len(dst_group.ptrs), (
            f"Number of regions of src({len(src_group.ptrs)}) and dst({len(dst_group.ptrs)}) must match"
        )
        new_src_ptrs = [src_ptr + self._src_block_off for src_ptr in src_group.ptrs]
        new_dst_ptrs = [dst_ptr + self._dst_block_off for dst_ptr in dst_group.ptrs]
        new_src = MemRegionGroup(ptrs=new_src_ptrs, bytes_per_region=self._frag_size)
        new_dst = MemRegionGroup(ptrs=new_dst_ptrs, bytes_per_region=self._frag_size)
        return SpecRegionPair(
            src=SpecRegion(memory=new_src, spec=src_regions.spec),
            dst=SpecRegion(memory=new_dst, spec=dst_regions.spec),
        )


class HeadMismatchMapper(RegionMapperBase):
    """
    ---- mapper_head_mismatch ----

    Fine-grained mapping when head counts or TP/DP partitioning differ.
    Split layers into per-head (or contiguous-heads) fragments and map them individually.
    Handles kv_factor (e.g., key+value duplication) and TP/DP head offsets.

    Source (layers x heads):
    L0: [S00 S01] [S02 S03] ...
    L1: [S10 S11] [S12 S13] ...

    Destination (layers x heads, different layout possible):
    L0': [D00] [D01] [D02] ...
    L1': [D10] [D11] ...

    Mapping (each arrow = copy cont_heads_frag):
    [S00 S01] -> [D00]
    [S02 S03] -> [D01]
    [S10 S11] -> [D02]
    """

    def __init__(
        self,
        transfer_layers: int,
        src_layer_off: int,
        peer_layer_off: int,
        self_ri: RankInfo,
        peer_ri: RankInfo,
    ):
        self._ri = self_ri
        self._peer_ri = peer_ri
        self._src_layer_off = src_layer_off

        kv_factor = self_ri.kv_factor
        self_tp_per_dp = self_ri.tp_size // self_ri.dp_size
        peer_tp_per_dp = peer_ri.tp_size // peer_ri.dp_size
        self_tp_rank = self_ri.tp_rank
        peer_tp_rank = peer_ri.tp_rank

        bytes_per_head = self._ri.tokens_per_block * self._ri.dims_per_head * self._ri.element_bytes
        self._bytes_cont_heads = (
            min(self._ri.kv_heads_per_rank, peer_ri.kv_heads_per_rank) * bytes_per_head
        )

        self._src_head_off, self._dst_head_off = self._compute_head_offsets(
            self_tp_per_dp,
            peer_tp_per_dp,
            self_tp_rank,
            peer_tp_rank,
            self._bytes_cont_heads,
        )
        self._layer_indices = np.arange(transfer_layers, dtype=np.int64)
        self._kv_indices = np.arange(kv_factor, dtype=np.int64)
        self._peer_layer_off = peer_layer_off

    def map(self, src_regions: SpecRegion, dst_regions: SpecRegion) -> SpecRegionPair:
        src_group = src_regions.memory
        dst_group = dst_regions.memory
        assert len(src_group.ptrs) == len(dst_group.ptrs), (
            f"Number of regions of src({len(src_group.ptrs)}) and dst({len(dst_group.ptrs)}) must match"
        )
        src_bases = np.array(src_group.ptrs, dtype=np.int64)
        dst_bases = np.array(dst_group.ptrs, dtype=np.int64)
        src_frags = self._get_frags(
            bases=src_bases,
            layer_indices=self._src_layer_off + self._layer_indices,
            layer_kv_num=self._get_layer_kv_num(self._ri),
            kv_indices=self._kv_indices,
            head_off=self._src_head_off,
            kv_factor=self._kv_indices.size,
        )
        dst_frags = self._get_frags(
            bases=dst_bases,
            layer_indices=self._peer_layer_off + self._layer_indices,
            layer_kv_num=self._get_layer_kv_num(self._peer_ri),
            kv_indices=self._kv_indices,
            head_off=self._dst_head_off,
            kv_factor=self._kv_indices.size,
        )
        all_src_ptrs = [int(x) for x in src_frags.flatten()]
        all_dst_ptrs = [int(x) for x in dst_frags.flatten()]
        new_src = MemRegionGroup(ptrs=all_src_ptrs, bytes_per_region=self._bytes_cont_heads)
        new_dst = MemRegionGroup(ptrs=all_dst_ptrs, bytes_per_region=self._bytes_cont_heads)
        return SpecRegionPair(
            src=SpecRegion(memory=new_src, spec=src_regions.spec),
            dst=SpecRegion(memory=new_dst, spec=dst_regions.spec),
        )

    @staticmethod
    def _compute_head_offsets(
        self_tp_per_dp: int,
        peer_tp_per_dp: int,
        self_tp_rank: int,
        peer_tp_rank: int,
        bytes_cont_heads: int,
    ) -> tuple[int, int]:
        if self_tp_per_dp == peer_tp_per_dp:
            return 0, 0
        ratio = max(self_tp_per_dp, peer_tp_per_dp) // min(self_tp_per_dp, peer_tp_per_dp)
        if self_tp_per_dp < peer_tp_per_dp:
            return (peer_tp_rank % ratio) * bytes_cont_heads, 0
        else:
            return 0, (self_tp_rank % ratio) * bytes_cont_heads

    @staticmethod
    def _get_layer_kv_num(ri: RankInfo) -> int:
        return ri.kv_heads_per_rank * ri.tokens_per_block * ri.dims_per_head * ri.element_bytes

    @staticmethod
    def _get_frags(bases, layer_indices, layer_kv_num, kv_indices, head_off, kv_factor):
        layer_num = layer_kv_num * kv_factor
        return (
            bases[:, None, None]
            + layer_num * layer_indices[None, :, None]
            + layer_kv_num * kv_indices[None, None, :]
            + head_off
        )


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

    def map(self, src_regions: SpecRegion, dst_regions: SpecRegion) -> SpecRegionPair:
        src_group = src_regions.memory
        dst_group = dst_regions.memory

        # Select overlapping layers
        src_ptrs = src_group.ptrs[self._src_layer_off : self._src_layer_off + self._transfer_layers]
        dst_ptrs = dst_group.ptrs[self._dst_layer_off : self._dst_layer_off + self._transfer_layers]

        assert len(src_ptrs) == len(dst_ptrs), (
            f"Number of regions of src({len(src_ptrs)}) and dst({len(dst_ptrs)}) must match"
        )

        new_src = MemRegionGroup(ptrs=list(src_ptrs), bytes_per_region=self._block_bytes)
        new_dst = MemRegionGroup(ptrs=list(dst_ptrs), bytes_per_region=self._block_bytes)
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

        # Apply head offset to each layer's address
        new_src_ptrs = [ptr + self._src_head_off for ptr in src_layer_ptrs]
        new_dst_ptrs = [ptr + self._dst_head_off for ptr in dst_layer_ptrs]

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

        assert len(src_layer_ptrs) == len(dst_layer_ptrs), (
            f"Number of layer ptrs mismatch: src={len(src_layer_ptrs)}, dst={len(dst_layer_ptrs)}"
        )

        results: List[SpecRegionPair] = []
        for src_off, dst_off, transfer_bytes in self._section_plans:
            sec_src_ptrs = [ptr + src_off for ptr in src_layer_ptrs]
            sec_dst_ptrs = [ptr + dst_off for ptr in dst_layer_ptrs]
            results.append(
                SpecRegionPair(
                    src=SpecRegion(
                        memory=MemRegionGroup(
                            ptrs=sec_src_ptrs,
                            bytes_per_region=transfer_bytes,
                        ),
                        spec=src_regions.spec,
                    ),
                    dst=SpecRegion(
                        memory=MemRegionGroup(
                            ptrs=sec_dst_ptrs,
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
    ratio = max(self_tp_per_dp, peer_tp_per_dp) // min(self_tp_per_dp, peer_tp_per_dp)
    if self_tp_per_dp < peer_tp_per_dp:
        # self has fewer ranks -> larger chunk; peer selects sub-chunk
        return (peer_tp_rank % ratio) * transfer_bytes, 0
    else:
        # peer has fewer ranks -> larger chunk; self selects sub-chunk
        return 0, (self_tp_rank % ratio) * transfer_bytes
