import numpy as np

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.base.region import (
    MemRegionGroup,
    RegionMapperBase,
    SpecRegion,
    SpecRegionPair,
)
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.utils import PoolRole
from tensorrt_llm._utils import nvtx_range


class IdentityMapper(RegionMapperBase):
    """
    ---- mapper_identity ----

    Pass-through mapping. Do not change pointers or sizes.

    src_ptrs: [ S0 ] [ S1 ] [ S2 ] ...
                |      |      |
                v      v      v
    dst_ptrs: [ D0 ] [ D1 ] [ D2 ] ...
    """

    @nvtx_range("IdentityMapper.map")
    def map(self, src_regions: SpecRegion, dst_regions: SpecRegion) -> SpecRegionPair:
        src_group = src_regions.memory
        dst_group = dst_regions.memory
        assert src_group.ptrs.size == dst_group.ptrs.size, (
            f"Number of regions of src({src_group.ptrs.size}) and dst({dst_group.ptrs.size}) must match"
        )
        return SpecRegionPair(
            src=SpecRegion(memory=src_group, spec=src_regions.spec),
            dst=SpecRegion(memory=dst_group, spec=dst_regions.spec),
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

    Contiguous-layer assumption:
        This mapper assumes that ``transfer_layers`` consecutive layers
        starting at ``src_layer_off`` (and ``dst_layer_off``) are laid out
        contiguously within each slot.  This holds because
        ``buffer_attributes()`` in the storage config assigns buffer
        offsets sequentially from 0 for each layer_group (life cycle),
        and each PoolDescriptor only contains layers belonging to a single
        layer_group.  Even when multiple layer_groups share the same
        physical storage pool_group, each layer_group independently
        occupies the full slot (offsets start from 0), so the contiguous
        layout is preserved.
    """

    def __init__(
        self,
        transfer_layers: int,
        src_layer_off: int,
        dst_layer_off: int,
        self_ri: RankInfo,
        peer_ri: RankInfo,
        slot_size_per_layer: int,
    ):
        if not isinstance(slot_size_per_layer, int):
            raise TypeError(
                f"slot_size_per_layer must be int, got {type(slot_size_per_layer).__name__} "
                f"(value={slot_size_per_layer}). Use // instead of / for integer division."
            )
        self._kv_factor = self_ri.attention.kv_factor
        self._frag_size = self._block_size(transfer_layers, slot_size_per_layer=slot_size_per_layer)
        self._src_block_off = self._block_size(
            src_layer_off, slot_size_per_layer=slot_size_per_layer
        )
        self._dst_block_off = self._block_size(
            dst_layer_off, slot_size_per_layer=slot_size_per_layer
        )

    @nvtx_range("HeadMatchMapper.map")
    def map(self, src_regions: SpecRegion, dst_regions: SpecRegion) -> SpecRegionPair:
        src_group = src_regions.memory
        dst_group = dst_regions.memory
        assert src_group.ptrs.size == dst_group.ptrs.size, (
            f"Number of regions of src({src_group.ptrs.size}) and dst({dst_group.ptrs.size}) must match"
        )
        new_src_ptrs = src_group.ptrs + self._src_block_off
        new_dst_ptrs = dst_group.ptrs + self._dst_block_off
        new_src = MemRegionGroup(ptrs=new_src_ptrs, bytes_per_region=self._frag_size)
        new_dst = MemRegionGroup(ptrs=new_dst_ptrs, bytes_per_region=self._frag_size)
        return SpecRegionPair(
            src=SpecRegion(memory=new_src, spec=src_regions.spec),
            dst=SpecRegion(memory=new_dst, spec=dst_regions.spec),
        )

    def _block_size(self, layer_num: int, slot_size_per_layer: int) -> int:
        return layer_num * slot_size_per_layer


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

        kv_factor = self_ri.attention.kv_factor
        self_tp_per_dp = self_ri.tp_size_per_dp_group
        peer_tp_per_dp = peer_ri.tp_size_per_dp_group
        self_tp_rank = self_ri.tp_rank
        peer_tp_rank = peer_ri.tp_rank

        bytes_per_head = (
            self._ri.attention.tokens_per_block
            * self._ri.attention.dims_per_head
            * self._ri.attention.element_bytes
        )
        self._bytes_cont_heads = (
            min(self._ri.attention.kv_heads_per_rank, peer_ri.attention.kv_heads_per_rank)
            * bytes_per_head
        )

        self._src_head_off, self._dst_head_off = self._compute_head_offsets(
            self_tp_per_dp,
            peer_tp_per_dp,
            self_tp_rank,
            peer_tp_rank,
            self._bytes_cont_heads,
        )
        self._peer_layer_off = peer_layer_off

        # --- Pre-compute flat 1D offset arrays ---
        #
        # Each KV cache block (slot) is laid out as:
        #
        #   block_base ──► [layer_0 kv_0] [layer_0 kv_1] [layer_1 kv_0] [layer_1 kv_1] ...
        #                   ◄─ layer_kv ─► ◄─ layer_kv ─►
        #                   ◄────── layer_num (= layer_kv * kv_factor) ──────►
        #
        # To address fragment (layer=j, kv=k) within a block at base_ptr:
        #
        #   frag_ptr = base_ptr
        #            + layer_num * (layer_off + j)    # skip to the right layer
        #            + layer_kv  * k                  # skip to key or value
        #            + head_off                       # head offset for TP mismatch
        #
        # The original code computed this as a 3D broadcast in map():
        #   bases[:, None, None] + layer_offsets[None, :, None]
        #                        + kv_offsets[None, None, :] + head_off
        # producing shape (n_blocks, transfer_layers, kv_factor) then .ravel().
        #
        # Optimization: since the (layer, kv) offsets are independent of the
        # per-call block base pointers, we pre-compute them here as a flat 1D
        # array of length (transfer_layers * kv_factor).  At map() time we only
        # need np.add.outer(bases, flat_offsets).ravel(), which produces the
        # same result in the same C-order traversal (blocks outer, offsets inner)
        # but with fewer intermediate allocations.
        layer_indices = np.arange(transfer_layers, dtype=np.int64)
        kv_indices = np.arange(kv_factor, dtype=np.int64)

        src_layer_kv_num = self._get_layer_kv_num(self._ri)
        src_layer_num = src_layer_kv_num * kv_factor
        # Shape (transfer_layers, kv_factor) → ravel to 1D
        self._src_flat_offsets = (
            src_layer_num * (src_layer_off + layer_indices)[:, None]
            + src_layer_kv_num * kv_indices[None, :]
            + self._src_head_off
        ).ravel()

        dst_layer_kv_num = self._get_layer_kv_num(self._peer_ri)
        dst_layer_num = dst_layer_kv_num * kv_factor
        self._dst_flat_offsets = (
            dst_layer_num * (peer_layer_off + layer_indices)[:, None]
            + dst_layer_kv_num * kv_indices[None, :]
            + self._dst_head_off
        ).ravel()

    @nvtx_range("HeadMismatchMapper.map")
    def map(self, src_regions: SpecRegion, dst_regions: SpecRegion) -> SpecRegionPair:
        src_group = src_regions.memory
        dst_group = dst_regions.memory
        assert src_group.ptrs.size == dst_group.ptrs.size, (
            f"Number of regions of src({src_group.ptrs.size}) and dst({dst_group.ptrs.size}) must match"
        )
        # np.add.outer(ptrs, offsets) produces every (base + offset) combination:
        #   shape (n_blocks, transfer_layers * kv_factor)
        # .ravel() flattens in C-order: for each block, emit all layer×kv fragments.
        # This is equivalent to the original 3D broadcast + ravel, but the per-(layer,kv)
        # offsets are pre-computed in __init__ so map() does a single vectorized add.
        all_src_ptrs = np.add.outer(src_group.ptrs, self._src_flat_offsets).ravel()
        all_dst_ptrs = np.add.outer(dst_group.ptrs, self._dst_flat_offsets).ravel()
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
        return (
            ri.attention.kv_heads_per_rank
            * ri.attention.tokens_per_block
            * ri.attention.dims_per_head
            * ri.attention.element_bytes
        )


class IndexerKCacheHeadMatchMapper(RegionMapperBase):
    """
    Mapper for indexer K cache when head counts match.

    Moves contiguous block(s) as a single chunk, aligned by block_size_per_layer,
    with constant source/destination block offsets.
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
        if not isinstance(block_size_per_layer, int):
            raise TypeError(
                f"block_size_per_layer must be int, got {type(block_size_per_layer).__name__} "
                f"(value={block_size_per_layer}). Use // instead of / for integer division."
            )
        self._frag_size = block_size_per_layer * transfer_layers
        self._src_block_off = block_size_per_layer * src_layer_off
        self._dst_block_off = block_size_per_layer * dst_layer_off

    @nvtx_range("IndexerKCacheHeadMatchMapper.map")
    def map(self, src_regions: SpecRegion, dst_regions: SpecRegion) -> SpecRegionPair:
        src_group = src_regions.memory
        dst_group = dst_regions.memory
        assert src_group.ptrs.size == dst_group.ptrs.size, (
            f"Number of regions of src({src_group.ptrs.size}) and dst({dst_group.ptrs.size}) must match"
        )
        new_src_ptrs = src_group.ptrs + self._src_block_off
        new_dst_ptrs = dst_group.ptrs + self._dst_block_off
        new_src = MemRegionGroup(ptrs=new_src_ptrs, bytes_per_region=self._frag_size)
        new_dst = MemRegionGroup(ptrs=new_dst_ptrs, bytes_per_region=self._frag_size)
        return SpecRegionPair(
            src=SpecRegion(memory=new_src, spec=src_regions.spec),
            dst=SpecRegion(memory=new_dst, spec=dst_regions.spec),
        )


class AttentionPolicy:
    def __init__(self, self_rank_info: RankInfo):
        self._ri = self_rank_info

    def _tp_per_dp(self, ri: RankInfo) -> int:
        if getattr(ri.attention, "enable_attention_dp", False):
            return ri.tp_size // ri.dp_size
        return ri.tp_size

    def _fail_if(self, cond: bool, reason: str, **kv) -> bool:
        if not cond:
            return False
        details = ", ".join(f"{k}={v!r}" for k, v in kv.items())
        msg = f"AttentionPolicy: incompatible: {reason}" + (f"; {details}" if details else "")
        logger.warning("%s", msg)
        return True

    def _mismatch(self, field: str, local, peer) -> bool:
        return self._fail_if(
            local != peer, f"{field} mismatch", field=field, local=local, peer=peer
        )

    def check_peer_compatible(self, peer_ri: RankInfo) -> bool:
        a = self._ri.attention
        b = peer_ri.attention

        return not (
            self._mismatch("is_mla", a.is_mla, b.is_mla)
            or self._fail_if(
                self._ri.cp_size != 1 or peer_ri.cp_size != 1,
                "cp_size must be 1 for both ranks",
                local=self._ri.cp_size,
                peer=peer_ri.cp_size,
            )
            or self._mismatch("element_bytes", a.element_bytes, b.element_bytes)
            or self._mismatch("tokens_per_block", a.tokens_per_block, b.tokens_per_block)
            or self._mismatch("dims_per_head", a.dims_per_head, b.dims_per_head)
            or self._fail_if(
                a.is_mla and (a.kv_heads_per_rank != 1 or b.kv_heads_per_rank != 1),
                "MLA requires kv_heads_per_rank == 1 for both ranks",
                local=a.kv_heads_per_rank,
                peer=b.kv_heads_per_rank,
            )
        )

    def _head_factors(self, peer_ri: RankInfo) -> tuple[int, int]:
        self_tp = self._tp_per_dp(self._ri)
        peer_tp = self._tp_per_dp(peer_ri)
        a = self._ri.attention
        b = peer_ri.attention
        return a.kv_heads_per_rank * self_tp, b.kv_heads_per_rank * peer_tp

    def head_match(self, peer_ri: RankInfo) -> tuple[bool, bool]:
        factor_self, factor_peer = self._head_factors(peer_ri)
        is_dup_head = factor_self != factor_peer
        head_match = (
            is_dup_head
            or self._ri.attention.is_mla
            or (self._tp_per_dp(self._ri) == self._tp_per_dp(peer_ri))
        )
        return head_match, is_dup_head

    def duplicate_head_factors(self, peer_ri: RankInfo) -> tuple[int, int]:
        factor_self, factor_peer = self._head_factors(peer_ri)
        dup_head = max(1, factor_self // factor_peer)
        peer_dup_head = max(1, factor_peer // factor_self)
        return dup_head, peer_dup_head

    def build_kv_mapper(
        self,
        *,
        peer_ri: RankInfo,
        pool_role: PoolRole,
        transfer_layers: int,
        self_layer_offset: int,
        peer_layer_offset: int,
        self_pool_num_layers: int,
        peer_pool_num_layers: int,
        self_pool_slot_bytes: int,
        peer_pool_slot_bytes: int,
    ) -> RegionMapperBase:
        head_match, _ = self.head_match(peer_ri)

        if head_match and transfer_layers == self_pool_num_layers == peer_pool_num_layers:
            return IdentityMapper()

        if head_match:
            if pool_role == PoolRole.INDEXER:
                block_size_per_layer = self_pool_slot_bytes // self_pool_num_layers
                return IndexerKCacheHeadMatchMapper(
                    transfer_layers=transfer_layers,
                    src_layer_off=self_layer_offset,
                    dst_layer_off=peer_layer_offset,
                    self_ri=self._ri,
                    peer_ri=peer_ri,
                    block_size_per_layer=block_size_per_layer,
                )

            slot_size_per_layer = self_pool_slot_bytes // self_pool_num_layers
            peer_size_per_layer = peer_pool_slot_bytes // peer_pool_num_layers
            assert slot_size_per_layer == peer_size_per_layer, (
                f"slot_size_per_layer mismatch between self ({slot_size_per_layer}) "
                f"and peer ({peer_size_per_layer}) for HeadMatchMapper"
            )
            return HeadMatchMapper(
                transfer_layers=transfer_layers,
                src_layer_off=self_layer_offset,
                dst_layer_off=peer_layer_offset,
                self_ri=self._ri,
                peer_ri=peer_ri,
                slot_size_per_layer=slot_size_per_layer,
            )

        if pool_role == PoolRole.INDEXER:
            raise ValueError("IndexerKCacheHeadMatchMapper is not supported for head mismatch case")

        return HeadMismatchMapper(
            transfer_layers=transfer_layers,
            src_layer_off=self_layer_offset,
            peer_layer_off=peer_layer_offset,
            self_ri=self._ri,
            peer_ri=peer_ri,
        )
