from dataclasses import dataclass

import numpy as np

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.base.region import (
    MemRegionGroup,
    RegionMapperBase,
    SpecRegion,
    SpecRegionPair,
)
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.page import MapperKind
from tensorrt_llm._torch.disaggregation.resource.utils import get_pool_view_mapper_kinds
from tensorrt_llm._utils import nvtx_range


@dataclass(frozen=True)
class PoolBufferMapping:
    """One logical buffer mapped between two physical pool slots."""

    src_offset: int
    dst_offset: int
    src_bytes: int
    dst_bytes: int
    mapper_kind: MapperKind


class IdentityMapper(RegionMapperBase):
    """
    ---- mapper_identity ----

    Pass-through mapping. Do not change pointers or sizes.

    src_ptrs: [ S0 ] [ S1 ] [ S2 ] ...
                |      |      |
                v      v      v
    dst_ptrs: [ D0 ] [ D1 ] [ D2 ] ...
    """

    def __init__(
        self,
        self_region_bytes: int | None = None,
        peer_region_bytes: int | None = None,
        *,
        mapper_name: str = "Identity",
    ) -> None:
        if self_region_bytes is not None and peer_region_bytes is not None:
            if self_region_bytes != peer_region_bytes:
                raise ValueError(
                    f"{mapper_name} cache region size mismatch: "
                    f"local={self_region_bytes}, peer={peer_region_bytes}"
                )

    @nvtx_range("IdentityMapper.map")
    def map(self, src_regions: SpecRegion, dst_regions: SpecRegion) -> SpecRegionPair:
        src_group = src_regions.memory
        dst_group = dst_regions.memory
        if src_group.ptrs.size != dst_group.ptrs.size:
            raise ValueError(
                f"Number of regions of src({src_group.ptrs.size}) and "
                f"dst({dst_group.ptrs.size}) must match"
            )
        return SpecRegionPair(
            src=SpecRegion(memory=src_group, spec=src_regions.spec),
            dst=SpecRegion(memory=dst_group, spec=dst_regions.spec),
        )


class ReplicatedMapper(IdentityMapper):
    """Copy a replicated cache region without KV-head remapping.

    Every TP rank holds identical bytes (for example, MiniMax M3 index-key),
    so mapping is an identity copy plus a strict source/destination size check.
    """

    def __init__(self, self_region_bytes: int, peer_region_bytes: int) -> None:
        super().__init__(self_region_bytes, peer_region_bytes, mapper_name="Replicated")


class PoolBufferMapper(RegionMapperBase):
    """Map heterogeneous logical buffers inside one physical pool view.

    Matching physical layouts with equal heads retain the whole-pool identity
    path. Otherwise, per-buffer offsets select NHD head slices, replicated
    buffers, or legacy HND buffers without expanding the page table per layer.
    """

    def __init__(
        self,
        *,
        mappings: list[PoolBufferMapping],
        self_ri: RankInfo,
        peer_ri: RankInfo,
        self_region_bytes: int,
        peer_region_bytes: int,
        full_region_identity: bool,
        include_sharded: bool,
        include_replicated: bool,
    ) -> None:
        head_match = (
            self_ri.attention.is_mla
            or self_ri.attention.kv_heads_per_rank == peer_ri.attention.kv_heads_per_rank
        )
        all_mappings_included = all(
            include_replicated if mapping.mapper_kind == MapperKind.REPLICATED else include_sharded
            for mapping in mappings
        )
        self._identity = (
            IdentityMapper(self_region_bytes, peer_region_bytes, mapper_name="PoolBuffer")
            if full_region_identity and head_match and all_mappings_included
            else None
        )
        self._plans = self._build_plans(
            mappings=mappings,
            self_ri=self_ri,
            peer_ri=peer_ri,
            include_sharded=include_sharded,
            include_replicated=include_replicated,
        )

    @staticmethod
    def _exact_div(numerator: int, denominator: int, *, what: str) -> int:
        if denominator <= 0 or numerator % denominator != 0:
            raise ValueError(
                f"{what} is not evenly divisible: numerator={numerator}, denominator={denominator}"
            )
        return numerator // denominator

    @classmethod
    def _build_plans(
        cls,
        *,
        mappings: list[PoolBufferMapping],
        self_ri: RankInfo,
        peer_ri: RankInfo,
        include_sharded: bool,
        include_replicated: bool,
    ) -> list[tuple[np.ndarray, np.ndarray, int]]:
        head_match = (
            self_ri.attention.is_mla
            or self_ri.attention.kv_heads_per_rank == peer_ri.attention.kv_heads_per_rank
        )
        self_heads = self_ri.attention.kv_heads_per_rank
        peer_heads = peer_ri.attention.kv_heads_per_rank
        self_tpb = self_ri.attention.tokens_per_block
        peer_tpb = peer_ri.attention.tokens_per_block
        offsets_by_size: dict[int, tuple[list[int], list[int]]] = {}

        def append_offsets(size: int, src_offsets, dst_offsets) -> None:
            src_list, dst_list = offsets_by_size.setdefault(size, ([], []))
            src_list.extend(int(offset) for offset in src_offsets)
            dst_list.extend(int(offset) for offset in dst_offsets)

        for mapping in mappings:
            kind = mapping.mapper_kind
            if kind == MapperKind.REPLICATED:
                if not include_replicated:
                    continue
            elif not include_sharded:
                continue

            if head_match or kind == MapperKind.REPLICATED:
                if mapping.src_bytes != mapping.dst_bytes:
                    raise ValueError(
                        "Pool buffer size mismatch: "
                        f"local={mapping.src_bytes}, peer={mapping.dst_bytes}, "
                        f"mapper={kind.name}"
                    )
                append_offsets(
                    mapping.src_bytes,
                    [mapping.src_offset],
                    [mapping.dst_offset],
                )
                continue

            if kind == MapperKind.NHD:
                if self_tpb != peer_tpb:
                    raise ValueError(
                        "NHD pool buffer mapping requires equal tokens_per_block: "
                        f"local={self_tpb}, peer={peer_tpb}"
                    )
                self_bytes_per_token_head = cls._exact_div(
                    mapping.src_bytes,
                    self_tpb * self_heads,
                    what="local NHD buffer bytes",
                )
                peer_bytes_per_token_head = cls._exact_div(
                    mapping.dst_bytes,
                    peer_tpb * peer_heads,
                    what="peer NHD buffer bytes",
                )
                if self_bytes_per_token_head != peer_bytes_per_token_head:
                    raise ValueError(
                        "NHD bytes per token/head mismatch: "
                        f"local={self_bytes_per_token_head}, "
                        f"peer={peer_bytes_per_token_head}"
                    )
                src_head_off, dst_head_off = HeadMismatchMapper._compute_head_offsets(
                    self_ri.tp_size_per_dp_group,
                    peer_ri.tp_size_per_dp_group,
                    self_ri.tp_rank,
                    peer_ri.tp_rank,
                    self_kv_heads=self_heads,
                    peer_kv_heads=peer_heads,
                    bytes_per_head=self_bytes_per_token_head,
                )
                transfer_bytes = min(self_heads, peer_heads) * self_bytes_per_token_head
                token_indices = np.arange(self_tpb, dtype=np.int64)
                append_offsets(
                    transfer_bytes,
                    mapping.src_offset
                    + token_indices * self_heads * self_bytes_per_token_head
                    + src_head_off,
                    mapping.dst_offset
                    + token_indices * peer_heads * peer_bytes_per_token_head
                    + dst_head_off,
                )
                continue

            if kind == MapperKind.INDEXED:
                self_bytes_per_head = cls._exact_div(
                    mapping.src_bytes,
                    self_heads,
                    what="local HND buffer bytes",
                )
                peer_bytes_per_head = cls._exact_div(
                    mapping.dst_bytes,
                    peer_heads,
                    what="peer HND buffer bytes",
                )
                if self_bytes_per_head != peer_bytes_per_head:
                    raise ValueError(
                        "HND bytes per head mismatch: "
                        f"local={self_bytes_per_head}, peer={peer_bytes_per_head}"
                    )
                src_head_off, dst_head_off = HeadMismatchMapper._compute_head_offsets(
                    self_ri.tp_size_per_dp_group,
                    peer_ri.tp_size_per_dp_group,
                    self_ri.tp_rank,
                    peer_ri.tp_rank,
                    self_kv_heads=self_heads,
                    peer_kv_heads=peer_heads,
                    bytes_per_head=self_bytes_per_head,
                )
                transfer_bytes = min(self_heads, peer_heads) * self_bytes_per_head
                append_offsets(
                    transfer_bytes,
                    [mapping.src_offset + src_head_off],
                    [mapping.dst_offset + dst_head_off],
                )
                continue

            raise ValueError(f"Unsupported per-buffer mapper kind: {kind.name}")

        return [
            (
                np.asarray(src_offsets, dtype=np.int64),
                np.asarray(dst_offsets, dtype=np.int64),
                size,
            )
            for size, (src_offsets, dst_offsets) in offsets_by_size.items()
        ]

    @nvtx_range("PoolBufferMapper.map")
    def map(
        self,
        src_regions: SpecRegion,
        dst_regions: SpecRegion,
    ) -> SpecRegionPair | list[SpecRegionPair]:
        if self._identity is not None:
            return self._identity.map(src_regions, dst_regions)

        src_group = src_regions.memory
        dst_group = dst_regions.memory
        if src_group.ptrs.size != dst_group.ptrs.size:
            raise ValueError(
                f"Number of regions of src({src_group.ptrs.size}) and "
                f"dst({dst_group.ptrs.size}) must match"
            )

        results = []
        for src_offsets, dst_offsets, transfer_bytes in self._plans:
            src_ptrs = np.add.outer(src_group.ptrs, src_offsets).ravel()
            dst_ptrs = np.add.outer(dst_group.ptrs, dst_offsets).ravel()
            results.append(
                SpecRegionPair(
                    src=SpecRegion(
                        memory=MemRegionGroup(src_ptrs, transfer_bytes),
                        spec=src_regions.spec,
                    ),
                    dst=SpecRegion(
                        memory=MemRegionGroup(dst_ptrs, transfer_bytes),
                        spec=dst_regions.spec,
                    ),
                )
            )
        return results


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
        if src_group.ptrs.size != dst_group.ptrs.size:
            raise ValueError(
                f"Number of regions of src({src_group.ptrs.size}) and "
                f"dst({dst_group.ptrs.size}) must match"
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
            self_kv_heads=self._ri.attention.kv_heads_per_rank,
            peer_kv_heads=peer_ri.attention.kv_heads_per_rank,
            bytes_per_head=bytes_per_head,
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
        if src_group.ptrs.size != dst_group.ptrs.size:
            raise ValueError(
                f"Number of regions of src({src_group.ptrs.size}) and "
                f"dst({dst_group.ptrs.size}) must match"
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
        self_kv_heads: int,
        peer_kv_heads: int,
        bytes_per_head: int,
    ) -> tuple[int, int]:
        if self_tp_per_dp == peer_tp_per_dp:
            return 0, 0
        # total_unique = total KV heads in model; head_idx via integer division
        # handles head duplication (total_kv_heads < tp_per_dp) correctly.
        if self_tp_per_dp < peer_tp_per_dp:
            total_unique = self_kv_heads * self_tp_per_dp
            src_head_idx = (peer_tp_rank * total_unique) // peer_tp_per_dp % self_kv_heads
            return src_head_idx * bytes_per_head, 0
        else:
            total_unique = peer_kv_heads * peer_tp_per_dp
            dst_head_idx = (self_tp_rank * total_unique) // self_tp_per_dp % peer_kv_heads
            return 0, dst_head_idx * bytes_per_head

    @staticmethod
    def _get_layer_kv_num(ri: RankInfo) -> int:
        return (
            ri.attention.kv_heads_per_rank
            * ri.attention.tokens_per_block
            * ri.attention.dims_per_head
            * ri.attention.element_bytes
        )


class NHDHeadMismatchMapper(HeadMismatchMapper):
    """Map heterogeneous KV heads stored token-major as ``[N, H, D]``.

    ``HeadMismatchMapper`` selects one contiguous head range per K/V buffer,
    which is correct for HND storage. In NHD storage, the selected head range
    is contiguous only within one token, so this mapper emits one fragment per
    ``(layer, K/V, token)``. Only offset precomputation differs from the
    parent; the inherited :meth:`map` consumes ``_src_flat_offsets``,
    ``_dst_flat_offsets``, and ``_bytes_cont_heads``.
    """

    def __init__(
        self,
        transfer_layers: int,
        src_layer_off: int,
        peer_layer_off: int,
        self_ri: RankInfo,
        peer_ri: RankInfo,
        self_region_bytes: int,
        peer_region_bytes: int,
        self_pool_num_layers: int,
        peer_pool_num_layers: int,
        self_buffers_per_layer: int,
        peer_buffers_per_layer: int,
    ) -> None:
        # Deliberately do not call HeadMismatchMapper.__init__: its offsets
        # assume HND-contiguous heads. Initialize the three attributes consumed
        # by the inherited map() with NHD token-granular geometry instead.
        self_tpb = self_ri.attention.tokens_per_block
        peer_tpb = peer_ri.attention.tokens_per_block
        if self_tpb != peer_tpb:
            raise ValueError(
                "NHDHeadMismatchMapper requires equal tokens_per_block; "
                f"local={self_tpb}, peer={peer_tpb}"
            )

        self_heads = self_ri.attention.kv_heads_per_rank
        peer_heads = peer_ri.attention.kv_heads_per_rank
        if self_buffers_per_layer != peer_buffers_per_layer:
            raise ValueError(
                "NHD buffer count per layer mismatch: "
                f"local={self_buffers_per_layer}, peer={peer_buffers_per_layer}"
            )

        self_bytes_per_token_head = self._bytes_per_token_head(
            region_bytes=self_region_bytes,
            num_layers=self_pool_num_layers,
            buffers_per_layer=self_buffers_per_layer,
            tokens_per_block=self_tpb,
            heads=self_heads,
        )
        peer_bytes_per_token_head = self._bytes_per_token_head(
            region_bytes=peer_region_bytes,
            num_layers=peer_pool_num_layers,
            buffers_per_layer=peer_buffers_per_layer,
            tokens_per_block=peer_tpb,
            heads=peer_heads,
        )
        if self_bytes_per_token_head != peer_bytes_per_token_head:
            raise ValueError(
                "NHD bytes per token/head mismatch: "
                f"local={self_bytes_per_token_head}, peer={peer_bytes_per_token_head}"
            )
        self._bytes_cont_heads = min(self_heads, peer_heads) * self_bytes_per_token_head

        src_head_off, dst_head_off = HeadMismatchMapper._compute_head_offsets(
            self_ri.tp_size_per_dp_group,
            peer_ri.tp_size_per_dp_group,
            self_ri.tp_rank,
            peer_ri.tp_rank,
            self_kv_heads=self_heads,
            peer_kv_heads=peer_heads,
            bytes_per_head=self_bytes_per_token_head,
        )

        self._src_flat_offsets = self._build_flat_offsets(
            transfer_layers=transfer_layers,
            layer_offset=src_layer_off,
            buffers_per_layer=self_buffers_per_layer,
            tokens_per_block=self_tpb,
            heads=self_heads,
            bytes_per_token_head=self_bytes_per_token_head,
            head_offset=src_head_off,
        )
        self._dst_flat_offsets = self._build_flat_offsets(
            transfer_layers=transfer_layers,
            layer_offset=peer_layer_off,
            buffers_per_layer=peer_buffers_per_layer,
            tokens_per_block=peer_tpb,
            heads=peer_heads,
            bytes_per_token_head=peer_bytes_per_token_head,
            head_offset=dst_head_off,
        )

    @staticmethod
    def _bytes_per_token_head(
        *,
        region_bytes: int,
        num_layers: int,
        buffers_per_layer: int,
        tokens_per_block: int,
        heads: int,
    ) -> int:
        denominator = num_layers * buffers_per_layer * tokens_per_block * heads
        if denominator <= 0 or region_bytes % denominator != 0:
            raise ValueError(
                "NHD region geometry is not evenly divisible: "
                f"region_bytes={region_bytes}, layers={num_layers}, "
                f"buffers_per_layer={buffers_per_layer}, "
                f"tokens_per_block={tokens_per_block}, kv_heads={heads}, "
                f"denominator={denominator}"
            )
        return region_bytes // denominator

    @staticmethod
    def _build_flat_offsets(
        *,
        transfer_layers: int,
        layer_offset: int,
        buffers_per_layer: int,
        tokens_per_block: int,
        heads: int,
        bytes_per_token_head: int,
        head_offset: int,
    ) -> np.ndarray:
        layer_indices = np.arange(transfer_layers, dtype=np.int64)
        buffer_indices = np.arange(buffers_per_layer, dtype=np.int64)
        token_indices = np.arange(tokens_per_block, dtype=np.int64)
        token_bytes = heads * bytes_per_token_head
        buffer_bytes = tokens_per_block * token_bytes
        layer_bytes = buffers_per_layer * buffer_bytes
        return (
            layer_bytes * (layer_offset + layer_indices)[:, None, None]
            + buffer_bytes * buffer_indices[None, :, None]
            + token_bytes * token_indices[None, None, :]
            + head_offset
        ).ravel()


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
        if src_group.ptrs.size != dst_group.ptrs.size:
            raise ValueError(
                f"Number of regions of src({src_group.ptrs.size}) and "
                f"dst({dst_group.ptrs.size}) must match"
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

    @staticmethod
    def _uses_exact_tpb_mapper(ri: RankInfo) -> bool:
        if ri.page_table is None:
            return False
        return any(
            bool(get_pool_view_mapper_kinds(pool_view) & {MapperKind.NHD, MapperKind.REPLICATED})
            for layer_group in ri.page_table.layer_groups
            for pool_view in getattr(layer_group, "pool_views", ())
        )

    @staticmethod
    def _uses_legacy_subbyte_mapper(ri: RankInfo) -> bool:
        element_bytes = ri.attention.element_bytes
        if float(element_bytes).is_integer():
            return False
        if ri.page_table is None:
            return True
        return any(
            bool(get_pool_view_mapper_kinds(pool_view) - {MapperKind.NHD, MapperKind.REPLICATED})
            for layer_group in ri.page_table.layer_groups
            for pool_view in getattr(layer_group, "pool_views", ())
        )

    def _tpb_check(self, local: int, peer: int, peer_ri: RankInfo) -> bool:
        if local == peer:
            return False
        if self._uses_exact_tpb_mapper(self._ri) or self._uses_exact_tpb_mapper(peer_ri):
            logger.warning(
                "AttentionPolicy: incompatible: tokens_per_block mismatch for "
                "NHD/replicated logical pools; local=%d peer=%d",
                local,
                peer,
            )
            return True
        larger, smaller = max(local, peer), min(local, peer)
        if larger % smaller != 0:
            logger.warning(
                "AttentionPolicy: incompatible: tokens_per_block not divisible; local=%d peer=%d",
                local,
                peer,
            )
            return True
        logger.warning(
            "AttentionPolicy: tokens_per_block mismatch (local=%d, peer=%d); "
            "KV transfer proceeds — ensure block boundaries align during transfer.",
            local,
            peer,
        )
        return False

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
            or self._fail_if(
                not self.head_match(peer_ri)[0]
                and (
                    self._uses_legacy_subbyte_mapper(self._ri)
                    or self._uses_legacy_subbyte_mapper(peer_ri)
                ),
                "sub-byte cache dtype requires geometry-aware NHD/replicated pools",
                local_element_bytes=a.element_bytes,
                peer_element_bytes=b.element_bytes,
            )
            or self._tpb_check(a.tokens_per_block, b.tokens_per_block, peer_ri)
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
        # Slot layout is compatible if kv_heads_per_rank matches (same slot size).
        head_match = (
            self._ri.attention.is_mla
            or self._ri.attention.kv_heads_per_rank == peer_ri.attention.kv_heads_per_rank
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
        mapper_kind: MapperKind,
        transfer_layers: int,
        self_layer_offset: int,
        peer_layer_offset: int,
        self_pool_num_layers: int,
        peer_pool_num_layers: int,
        self_buffers_per_layer: int,
        peer_buffers_per_layer: int,
        self_pool_slot_bytes: int,
        peer_pool_slot_bytes: int,
    ) -> RegionMapperBase:
        if mapper_kind == MapperKind.REPLICATED:
            return ReplicatedMapper(self_pool_slot_bytes, peer_pool_slot_bytes)

        head_match, _ = self.head_match(peer_ri)

        if head_match and transfer_layers == self_pool_num_layers == peer_pool_num_layers:
            return IdentityMapper(self_pool_slot_bytes, peer_pool_slot_bytes)

        if head_match:
            if mapper_kind == MapperKind.FLAT:
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
            if slot_size_per_layer != peer_size_per_layer:
                raise ValueError(
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

        if mapper_kind == MapperKind.FLAT:
            raise ValueError("IndexerKCacheHeadMatchMapper is not supported for head mismatch case")

        if mapper_kind == MapperKind.NHD:
            return NHDHeadMismatchMapper(
                transfer_layers=transfer_layers,
                src_layer_off=self_layer_offset,
                peer_layer_off=peer_layer_offset,
                self_ri=self._ri,
                peer_ri=peer_ri,
                self_region_bytes=self_pool_slot_bytes,
                peer_region_bytes=peer_pool_slot_bytes,
                self_pool_num_layers=self_pool_num_layers,
                peer_pool_num_layers=peer_pool_num_layers,
                self_buffers_per_layer=self_buffers_per_layer,
                peer_buffers_per_layer=peer_buffers_per_layer,
            )

        return HeadMismatchMapper(
            transfer_layers=transfer_layers,
            src_layer_off=self_layer_offset,
            peer_layer_off=peer_layer_offset,
            self_ri=self._ri,
            peer_ri=peer_ri,
        )
