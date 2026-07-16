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

from collections.abc import Sequence

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
from tensorrt_llm._utils import nvtx_range


class IntactMapper(RegionMapperBase):
    """
    ---- mapper_intact ----

    Copy the selected layers' class regions between slots. Consumes slot-base
    pointers from the extractor and expands them with slot-relative per-layer
    byte offsets taken from the logical view's buffer entries, so it handles
    non-uniform layer strides (a slot may interleave other role classes
    between this class's layers).

    src slot: [ base ]--+off(L2)--> [L2 region] --+off(L3)--> [L3 region] ...
    dst slot: [ base ]--+off'(L2)-> [L2 region] --+off'(L3)-> [L3 region] ...

    Layers whose regions are contiguous on BOTH sides are merged into one
    fragment, so a fully contiguous class (e.g. K/V in a dedicated pool)
    degrades to a single whole-region copy per block.
    """

    def __init__(
        self,
        src_layer_offsets: Sequence[int] | np.ndarray,
        dst_layer_offsets: Sequence[int] | np.ndarray,
        self_bytes_per_layer: int,
        peer_bytes_per_layer: int,
        *,
        mapper_name: str = "Intact",
    ) -> None:
        if self_bytes_per_layer != peer_bytes_per_layer:
            raise ValueError(
                f"{mapper_name} cache region size mismatch: "
                f"local={self_bytes_per_layer}, peer={peer_bytes_per_layer}"
            )
        src = np.asarray(src_layer_offsets, dtype=np.int64)
        dst = np.asarray(dst_layer_offsets, dtype=np.int64)
        if src.size == 0 or src.size != dst.size:
            raise ValueError(
                f"{mapper_name} layer offsets must be non-empty and equal-length: "
                f"src={src.size}, dst={dst.size}"
            )
        self._runs = self._merge_contiguous(src, dst, self_bytes_per_layer)

    @staticmethod
    def _merge_contiguous(
        src: np.ndarray, dst: np.ndarray, bytes_per_layer: int
    ) -> list[tuple[int, int, int]]:
        runs: list[tuple[int, int, int]] = []
        run_start = 0
        for i in range(1, src.size + 1):
            if (
                i == src.size
                or src[i] != src[i - 1] + bytes_per_layer
                or dst[i] != dst[i - 1] + bytes_per_layer
            ):
                run_layers = i - run_start
                runs.append(
                    (int(src[run_start]), int(dst[run_start]), run_layers * bytes_per_layer)
                )
                run_start = i
        return runs

    @nvtx_range("IntactMapper.map")
    def map(self, src_regions: SpecRegion, dst_regions: SpecRegion):
        src_group = src_regions.memory
        dst_group = dst_regions.memory
        if src_group.ptrs.size != dst_group.ptrs.size:
            raise ValueError(
                f"Number of regions of src({src_group.ptrs.size}) and "
                f"dst({dst_group.ptrs.size}) must match"
            )
        pairs = [
            SpecRegionPair(
                src=SpecRegion(
                    memory=MemRegionGroup(
                        ptrs=src_group.ptrs + src_off, bytes_per_region=run_bytes
                    ),
                    spec=src_regions.spec,
                ),
                dst=SpecRegion(
                    memory=MemRegionGroup(
                        ptrs=dst_group.ptrs + dst_off, bytes_per_region=run_bytes
                    ),
                    spec=dst_regions.spec,
                ),
            )
            for src_off, dst_off, run_bytes in self._runs
        ]
        return pairs[0] if len(pairs) == 1 else pairs


class ReplicatedMapper(IntactMapper):
    """Copy TP-replicated per-layer regions without KV-head remapping.

    Every TP rank holds identical bytes per layer (MiniMax M3 index-key, DSA
    indexer K), so no head slicing applies. Layer selection under partial PP
    overlap happens through the per-layer offsets, and fan-in routing (one
    owning sender per destination) is decided upstream by
    ``PeerRegistrar.should_send_pool``.
    """

    def __init__(
        self,
        src_layer_offsets: Sequence[int] | np.ndarray,
        dst_layer_offsets: Sequence[int] | np.ndarray,
        self_bytes_per_layer: int,
        peer_bytes_per_layer: int,
    ) -> None:
        super().__init__(
            src_layer_offsets,
            dst_layer_offsets,
            self_bytes_per_layer,
            peer_bytes_per_layer,
            mapper_name="Replicated",
        )


class HNDHeadMismatchMapper(RegionMapperBase):
    """
    ---- mapper_hnd_head_mismatch ----

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
        *,
        src_layer_offsets: "Sequence[int] | np.ndarray",
        dst_layer_offsets: "Sequence[int] | np.ndarray",
        self_ri: RankInfo,
        peer_ri: RankInfo,
        self_bytes_per_layer: int,
        peer_bytes_per_layer: int,
        self_buffers_per_layer: int,
        peer_buffers_per_layer: int,
    ):
        self._ri = self_ri
        self._peer_ri = peer_ri

        self_tp_per_dp = self_ri.tp_size_per_dp_group
        peer_tp_per_dp = peer_ri.tp_size_per_dp_group
        self_tp_rank = self_ri.tp_rank
        peer_tp_rank = peer_ri.tp_rank

        if self_buffers_per_layer != peer_buffers_per_layer:
            raise ValueError(
                "HND buffer count per layer mismatch: "
                f"local={self_buffers_per_layer}, peer={peer_buffers_per_layer}"
            )
        src_offsets = np.asarray(src_layer_offsets, dtype=np.int64)
        dst_offsets = np.asarray(dst_layer_offsets, dtype=np.int64)
        if src_offsets.size == 0 or src_offsets.size != dst_offsets.size:
            raise ValueError(
                "HND layer offsets must be non-empty and equal-length: "
                f"src={src_offsets.size}, dst={dst_offsets.size}"
            )

        # Byte geometry is derived from the per-layer region size registered
        # by storage (always whole bytes) rather than element_bytes x dims
        # arithmetic, so sub-byte dtypes (e.g. NVFP4) stay exact-integer.
        src_buffer_bytes = self._bytes_per_buffer(
            bytes_per_layer=self_bytes_per_layer,
            buffers_per_layer=self_buffers_per_layer,
            side="local",
        )
        dst_buffer_bytes = self._bytes_per_buffer(
            bytes_per_layer=peer_bytes_per_layer,
            buffers_per_layer=peer_buffers_per_layer,
            side="peer",
        )
        bytes_per_head = self._bytes_per_head(
            src_buffer_bytes, self._ri.attention.kv_heads_per_rank, side="local"
        )
        peer_bytes_per_head = self._bytes_per_head(
            dst_buffer_bytes, peer_ri.attention.kv_heads_per_rank, side="peer"
        )
        if bytes_per_head != peer_bytes_per_head:
            raise ValueError(
                f"HND bytes per head mismatch: local={bytes_per_head}, peer={peer_bytes_per_head}"
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

        # Pre-compute flat 1D offset arrays: one fragment per (layer, buffer)
        # where buffers are the layer's K/V (or scale) buffers laid out
        # back-to-back within the layer's region. Layer starts come from the
        # view's buffer entries, so interleaved role classes (non-uniform
        # layer strides) are handled the same way as everywhere else. At
        # map() time a single np.add.outer(bases, flat_offsets) expands the
        # per-block base pointers.
        self._src_flat_offsets = self._build_flat_offsets(
            layer_offsets=src_offsets,
            buffers_per_layer=self_buffers_per_layer,
            buffer_bytes=src_buffer_bytes,
            head_offset=self._src_head_off,
        )
        self._dst_flat_offsets = self._build_flat_offsets(
            layer_offsets=dst_offsets,
            buffers_per_layer=peer_buffers_per_layer,
            buffer_bytes=dst_buffer_bytes,
            head_offset=self._dst_head_off,
        )

    @staticmethod
    def _build_flat_offsets(
        *,
        layer_offsets: np.ndarray,
        buffers_per_layer: int,
        buffer_bytes: int,
        head_offset: int,
    ) -> np.ndarray:
        buffer_indices = np.arange(buffers_per_layer, dtype=np.int64)
        return (
            layer_offsets[:, None] + buffer_bytes * buffer_indices[None, :] + head_offset
        ).ravel()

    @nvtx_range("HNDHeadMismatchMapper.map")
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
    def _bytes_per_buffer(*, bytes_per_layer: int, buffers_per_layer: int, side: str) -> int:
        """Bytes of one buffer (K or V) within a layer's region."""
        if buffers_per_layer <= 0 or bytes_per_layer % buffers_per_layer != 0:
            raise ValueError(
                f"HND layer geometry is not evenly divisible ({side}): "
                f"bytes_per_layer={bytes_per_layer}, buffers_per_layer={buffers_per_layer}"
            )
        return bytes_per_layer // buffers_per_layer

    @staticmethod
    def _bytes_per_head(layer_kv_bytes: int, heads: int, *, side: str) -> int:
        """Bytes of one head's rows within a K/V buffer.

        Byte-granular head slicing is only valid when a head lands on a byte
        boundary; sub-byte dtypes (e.g. NVFP4) satisfy this whenever the
        per-head element count covers whole bytes, which this divisibility
        check enforces without any fractional arithmetic.
        """
        if heads <= 0 or layer_kv_bytes % heads != 0:
            raise ValueError(
                f"HND head slicing is not byte-aligned ({side}): "
                f"layer_kv_bytes={layer_kv_bytes}, kv_heads={heads}"
            )
        return layer_kv_bytes // heads


class NHDHeadMismatchMapper(HNDHeadMismatchMapper):
    """Map heterogeneous KV heads stored token-major as ``[N, H, D]``.

    ``HNDHeadMismatchMapper`` selects one contiguous head range per K/V buffer,
    which is correct for HND storage. In NHD storage, the selected head range
    is contiguous only within one token, so this mapper emits one fragment per
    ``(layer, K/V, token)``. Only offset precomputation differs from the
    parent; the inherited :meth:`map` consumes ``_src_flat_offsets``,
    ``_dst_flat_offsets``, and ``_bytes_cont_heads``.
    """

    def __init__(
        self,
        *,
        src_layer_offsets: Sequence[int] | np.ndarray,
        dst_layer_offsets: Sequence[int] | np.ndarray,
        self_ri: RankInfo,
        peer_ri: RankInfo,
        self_bytes_per_layer: int,
        peer_bytes_per_layer: int,
        self_buffers_per_layer: int,
        peer_buffers_per_layer: int,
    ) -> None:
        # Deliberately do not call HNDHeadMismatchMapper.__init__: its offsets
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
        src_offsets = np.asarray(src_layer_offsets, dtype=np.int64)
        dst_offsets = np.asarray(dst_layer_offsets, dtype=np.int64)
        if src_offsets.size == 0 or src_offsets.size != dst_offsets.size:
            raise ValueError(
                "NHD layer offsets must be non-empty and equal-length: "
                f"src={src_offsets.size}, dst={dst_offsets.size}"
            )

        self_bytes_per_token_head = self._bytes_per_token_head(
            bytes_per_layer=self_bytes_per_layer,
            buffers_per_layer=self_buffers_per_layer,
            tokens_per_block=self_tpb,
            heads=self_heads,
        )
        peer_bytes_per_token_head = self._bytes_per_token_head(
            bytes_per_layer=peer_bytes_per_layer,
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

        src_head_off, dst_head_off = HNDHeadMismatchMapper._compute_head_offsets(
            self_ri.tp_size_per_dp_group,
            peer_ri.tp_size_per_dp_group,
            self_ri.tp_rank,
            peer_ri.tp_rank,
            self_kv_heads=self_heads,
            peer_kv_heads=peer_heads,
            bytes_per_head=self_bytes_per_token_head,
        )

        self._src_flat_offsets = self._build_flat_offsets(
            layer_offsets=src_offsets,
            buffers_per_layer=self_buffers_per_layer,
            tokens_per_block=self_tpb,
            heads=self_heads,
            bytes_per_token_head=self_bytes_per_token_head,
            head_offset=src_head_off,
        )
        self._dst_flat_offsets = self._build_flat_offsets(
            layer_offsets=dst_offsets,
            buffers_per_layer=peer_buffers_per_layer,
            tokens_per_block=peer_tpb,
            heads=peer_heads,
            bytes_per_token_head=peer_bytes_per_token_head,
            head_offset=dst_head_off,
        )

    @staticmethod
    def _bytes_per_token_head(
        *,
        bytes_per_layer: int,
        buffers_per_layer: int,
        tokens_per_block: int,
        heads: int,
    ) -> int:
        denominator = buffers_per_layer * tokens_per_block * heads
        if denominator <= 0 or bytes_per_layer % denominator != 0:
            raise ValueError(
                "NHD region geometry is not evenly divisible: "
                f"bytes_per_layer={bytes_per_layer}, "
                f"buffers_per_layer={buffers_per_layer}, "
                f"tokens_per_block={tokens_per_block}, kv_heads={heads}, "
                f"denominator={denominator}"
            )
        return bytes_per_layer // denominator

    @staticmethod
    def _build_flat_offsets(
        *,
        layer_offsets: np.ndarray,
        buffers_per_layer: int,
        tokens_per_block: int,
        heads: int,
        bytes_per_token_head: int,
        head_offset: int,
    ) -> np.ndarray:
        buffer_indices = np.arange(buffers_per_layer, dtype=np.int64)
        token_indices = np.arange(tokens_per_block, dtype=np.int64)
        token_bytes = heads * bytes_per_token_head
        buffer_bytes = tokens_per_block * token_bytes
        return (
            layer_offsets[:, None, None]
            + buffer_bytes * buffer_indices[None, :, None]
            + token_bytes * token_indices[None, None, :]
            + head_offset
        ).ravel()


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
        """NHD / replicated pools address bytes inside a block, so their
        geometry only lines up when both sides use the same tokens_per_block."""
        if ri.page_table is None:
            return False
        return any(
            pool_view.mapper_kind in (MapperKind.NHD, MapperKind.REPLICATED)
            for layer_group in ri.page_table.layer_groups
            for pool_view in getattr(layer_group, "pool_views", ())
        )

    def _tpb_check(self, local: int, peer: int, peer_ri: RankInfo) -> bool:
        if local == peer:
            return False
        if self._uses_exact_tpb_mapper(self._ri) or self._uses_exact_tpb_mapper(peer_ri):
            logger.warning(
                "AttentionPolicy: incompatible: tokens_per_block mismatch for "
                "NHD/replicated pools; local=%d peer=%d",
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
                and not float(a.tokens_per_block * a.dims_per_head * a.element_bytes).is_integer(),
                "sub-byte head slicing is not byte-aligned",
                tokens_per_block=a.tokens_per_block,
                dims_per_head=a.dims_per_head,
                element_bytes=a.element_bytes,
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
        self_layer_offsets: "Sequence[int] | np.ndarray",
        peer_layer_offsets: "Sequence[int] | np.ndarray",
        self_bytes_per_layer: int,
        peer_bytes_per_layer: int,
        self_buffers_per_layer: int = 1,
        peer_buffers_per_layer: int = 1,
    ) -> RegionMapperBase:
        """Pick the mapper for one view pair.

        Every view is entries-driven: layer selection always uses explicit
        slot-relative per-layer byte offsets, never positional arithmetic
        (other role classes may interleave between this class's layers).
        The kind only decides the two irreducible semantic differences:

        - REPLICATED skips head matching entirely (bytes are identical on
          every TP rank; fan-in ownership is decided upstream).
        - Under head mismatch, HND (INDEXED) slices one contiguous head
          range per K/V buffer, while NHD must slice inside every token.

        Head-matched views of any kind collapse into IntactMapper,
        whose run merging degrades to a single whole-region copy per block
        when the selected layers are contiguous on both sides.
        """
        if mapper_kind == MapperKind.REPLICATED:
            return ReplicatedMapper(
                self_layer_offsets,
                peer_layer_offsets,
                self_bytes_per_layer,
                peer_bytes_per_layer,
            )

        head_match, _ = self.head_match(peer_ri)
        if head_match:
            return IntactMapper(
                self_layer_offsets,
                peer_layer_offsets,
                self_bytes_per_layer,
                peer_bytes_per_layer,
                mapper_name=mapper_kind.name,
            )

        if mapper_kind == MapperKind.NHD:
            return NHDHeadMismatchMapper(
                src_layer_offsets=self_layer_offsets,
                dst_layer_offsets=peer_layer_offsets,
                self_ri=self._ri,
                peer_ri=peer_ri,
                self_bytes_per_layer=self_bytes_per_layer,
                peer_bytes_per_layer=peer_bytes_per_layer,
                self_buffers_per_layer=self_buffers_per_layer,
                peer_buffers_per_layer=peer_buffers_per_layer,
            )

        return HNDHeadMismatchMapper(
            src_layer_offsets=self_layer_offsets,
            dst_layer_offsets=peer_layer_offsets,
            self_ri=self._ri,
            peer_ri=peer_ri,
            self_bytes_per_layer=self_bytes_per_layer,
            peer_bytes_per_layer=peer_bytes_per_layer,
            self_buffers_per_layer=self_buffers_per_layer,
            peer_buffers_per_layer=peer_buffers_per_layer,
        )
