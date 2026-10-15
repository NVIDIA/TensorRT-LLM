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
"""The transfer planner every model gets unless it supplies its own.

Attention (PAGED) and recurrent-state (STATE) layer groups are planned by
``AttentionPolicy`` and ``MambaPolicy``; pools are matched by role and layer
overlap; TP/PP/CP overlap decides which peer ranks receive what. The rows each
side holds come in from the cache describer; this class only pairs them. All of
it is behind the :class:`TransferPlanner` protocol, so ``native/transfer.py``
only ever sees ranks, pointer arrays and byte counts.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.base.backend import RecvReqInfo, TokenRange
from tensorrt_llm._torch.disaggregation.base.planner import (
    LayerGroupRows,
    PeerIncompatibleError,
    PeerTargets,
    TransferPlan,
    TransferPlanner,
)
from tensorrt_llm._torch.disaggregation.base.region import (
    MemRegionGroup,
    RegionMapperBase,
    SpecRegion,
)
from tensorrt_llm._torch.disaggregation.native.mixers.attention.peer import AttentionPolicy
from tensorrt_llm._torch.disaggregation.native.mixers.ssm.peer import (
    MambaPolicy,
    mamba_receiver_payload_bytes,
)
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.page import (
    CacheKind,
    KVCachePageTable,
    MapperKind,
    PoolView,
)
from tensorrt_llm._torch.disaggregation.resource.utils import (
    get_layer_byte_ranges,
    get_layer_to_layer_group,
    get_pool_view_global_layer_ids,
)

# Type alias for (lg_idx, pool_idx) pair
LGPoolKey = Tuple[int, int]


@dataclass
class PeerOverlap:
    overlap_pp_size: int = 0
    overlap_tp_size: int = 0
    overlap_cp_size: int = 0
    duplicate_head_factor: int = 1
    peer_duplicate_head_factor: int = 1
    ranks: List[int] = field(default_factory=list)


class DefaultTransferPlanner(TransferPlanner):
    """Policy-driven planning over the page tables both peers publish.

    Per-peer state (pool mapping, mappers, peer extractor) is cached by peer
    key and dropped in :meth:`forget_peer`; the TP/PP/CP overlap is cached per
    peer *instance* and DP group because every rank of a DP group shares it.
    """

    planner_id = "trtllm.default"

    # Registry: CacheKind -> PolicyClass. Add new layer types here. The
    # policy owns every mapper kind of its life cycle and picks the mapper
    # per view in build_mapper.
    _POLICY_CLASSES = {
        CacheKind.PAGED: AttentionPolicy,
        CacheKind.STATE: MambaPolicy,
    }

    def __init__(self, self_rank_info: RankInfo, self_extractor: KVRegionExtractorV1):
        self._ri = self_rank_info
        self._self_ext_cache = self_extractor
        self._policies: Dict[CacheKind, Union[AttentionPolicy, MambaPolicy]] = {}
        self._kv_map_cache: Dict[
            tuple, RegionMapperBase
        ] = {}  # key: (peer_key, self_lg_pool_key, peer_lg_pool_key)
        self._overlap_cache: Dict[str, PeerOverlap] = {}
        self._targets_cache: Dict[str, PeerTargets] = {}
        self._lg_pool_mapping_cache: Dict[
            str, Dict[LGPoolKey, LGPoolKey]
        ] = {}  # peer_key -> {(self_lg, self_pi) -> (peer_lg, peer_pi)}
        # (peer_key or "", layer group, view) -> per-layer offsets within a STATE slot
        self._state_layout_cache: Dict[Tuple[str, int, int], Tuple[np.ndarray, int]] = {}

    @staticmethod
    def _unique_key(name: str, rank: int) -> str:
        return name + str(rank)

    # ---- TransferPlanner protocol -------------------------------------------------

    def check_peer(self, self_ri: RankInfo, peer_ri: RankInfo) -> None:
        """Vet a peer. Every layout or geometry mismatch surfaces as PeerIncompatibleError
        so the caller fails only that peer's requests and caches the verdict."""
        try:
            if self_ri.bounded_replay_on_generation != peer_ri.bounded_replay_on_generation:
                raise ValueError(
                    "bounded_replay_on_generation must match between peers "
                    f"(local={self_ri.bounded_replay_on_generation}, "
                    f"peer={peer_ri.bounded_replay_on_generation})"
                )
            if not self._get_policy(CacheKind.PAGED).check_peer_compatible(peer_ri):
                raise ValueError(
                    "attention geometry differs from the peer "
                    f"(peer={peer_ri.instance_name}, rank={peer_ri.instance_rank})"
                )
            # Recurrent-state (Mamba/KDA) layout gate, with a field-level diagnostic.
            self_page_table = self._self_ext_cache.page_table
            MambaPolicy.validate_peer_compatible(
                self._ri,
                peer_ri,
                self_page_table,
                peer_ri.page_table,
            )
            AttentionPolicy.validate_peer_compatible(self_page_table, peer_ri.page_table)

            self_layers = sum(self._ri.layer_num_per_pp)
            peer_layers = sum(peer_ri.layer_num_per_pp)
            if self_layers != peer_layers:
                # Allow mismatch when one side has speculative (e.g. MTP) layers
                # that the other side doesn't. The pool_mapping logic will only
                # transfer layers that exist on both sides.
                logger.warning(
                    "DefaultTransferPlanner: layer count differs "
                    f"(local={self_layers}, peer={peer_layers}), "
                    "allowing partial layer transfer."
                )
            # Pool matching runs here too; a peer that groups layers differently
            # raises from it and must be refused like any other mismatch.
            self._warn_nhd_head_mismatch(peer_ri)
        except PeerIncompatibleError:
            raise
        except ValueError as e:
            raise PeerIncompatibleError(f"DefaultTransferPlanner.check_peer: {e}") from e

    def _warn_nhd_head_mismatch(self, peer_ri: RankInfo) -> None:
        head_match, _ = self._get_policy(CacheKind.PAGED).head_match(peer_ri)
        if not head_match:
            self_page_table = self._self_ext_cache.page_table
            nhd_fragments_per_token = sum(
                len(
                    self_page_table.layer_groups[layer_group_id].pool_views[pool_idx].buffer_entries
                )
                for layer_group_id, pool_idx in self.get_pool_mapping(peer_ri)
                if self_page_table.layer_groups[layer_group_id].pool_views[pool_idx].mapper_kind
                == MapperKind.NHD
            )
            if nhd_fragments_per_token:
                local_heads = self._ri.attention.kv_heads_per_rank
                peer_heads = peer_ri.attention.kv_heads_per_rank
                logger.warning_once(
                    "NHD head-mismatched disaggregated KV transfer has no "
                    "contiguous staging path and will emit approximately "
                    f"{nhd_fragments_per_token} NIXL descriptors per transferred "
                    "token per peer, excluding block-level replicated pools "
                    f"(local_kv_heads={local_heads}, peer_kv_heads={peer_heads}). "
                    "Long-context TEP/DEP transfers may have high latency.",
                    key=(
                        "native-nhd-head-mismatch-"
                        f"{local_heads}-{peer_heads}-{nhd_fragments_per_token}"
                    ),
                )

    def targets(self, self_ri: RankInfo, peer_ri: RankInfo, peer_dp_rank: int) -> PeerTargets:
        # The sender asks on every request; the fan-in gate walks both page
        # tables, so the answer is kept per (peer instance, DP group) like the
        # overlap it derives from.
        key = self._unique_key(peer_ri.instance_name, peer_dp_rank)
        cached = self._targets_cache.get(key)
        if cached is not None:
            return cached
        overlap = self.get_peer_overlap(peer_ri, peer_dp_rank)
        # Helix writers contribute unequal shares (one KV owner, one state
        # sender, the rest empty), so CP on either side rules out the equal
        # fan-in split as well.
        uniform = (
            self_ri.cp_size == 1
            and peer_ri.cp_size == 1
            and self._fanin_bounce_safe(overlap, peer_ri, self._self_ext_cache.page_table)
        )
        targets = PeerTargets(ranks=list(overlap.ranks), uniform_writers=uniform)
        self._targets_cache[key] = targets
        return targets

    def pair(
        self,
        self_ri: RankInfo,
        peer_ri: RankInfo,
        self_rows: Sequence[LayerGroupRows],
        peer_rows: Sequence[LayerGroupRows],
        *,
        token_range: TokenRange,
        prompt_len: int,
        dst_start_token: Optional[int],
    ) -> TransferPlan:
        # Only a piece short of the whole prompt runs the chunk projection below, and that is the
        # part with no answer under CP; a sole piece covering everything takes the untouched path.
        is_partial = token_range.start > 0 or token_range.end < prompt_len
        if is_partial and (self_ri.cp_size > 1 or peer_ri.cp_size > 1):
            raise ValueError(
                "enable_pipelined_transfer is not supported with context parallelism "
                f"(sender cp_size={self_ri.cp_size}, receiver cp_size={peer_ri.cp_size})"
            )
        targets = self.get_peer_overlap(peer_ri, peer_ri.dp_rank)
        self_by_view = {(r.layer_group_index, r.view_index): r for r in self_rows}
        peer_by_view = {(r.layer_group_index, r.view_index): r for r in peer_rows}
        tpb = self._self_ext_cache.page_table.tokens_per_block
        dst_start_block = dst_start_token // tpb if dst_start_token is not None else 0
        # Helix: the receiver owns global blocks [cp_rank::cp_size] (same protocol as
        # partition_context_for_helix), so its ordinals are the strided subset of ours;
        # block reuse is rejected under helix.
        helix = (
            (peer_ri.cp_rank, peer_ri.cp_size)
            if peer_ri.cp_size > 1 and self_ri.cp_size == 1
            else None
        )

        # Aggregate fragment pointers from all matching pool pairs.
        # Each pool pair produces one or more region pairs (rp), each containing
        # a numpy array of src/dst pointers and a uniform bytes_per_region.
        src_frag_parts: list[np.ndarray] = []
        dst_frag_parts: list[np.ndarray] = []
        # Instead of calling np.full() per region pair to build a size array and
        # then np.concatenate() all of them, we record (count, bytes_per_region)
        # tuples and construct the final sizes array with a single np.repeat().
        # For 48k+ items this avoids many small allocations in the hot loop.
        size_specs: list[tuple[int, int]] = []
        pool_mapping = self.get_pool_mapping(peer_ri)

        # Aggregate fragments from all matching pools using numpy concatenation.
        # Send ownership is per pool: replicated pools elect one fan-in
        # owner, sharded pools keep head-duplication routing.
        for (self_lg, self_pi), (peer_lg, peer_pi) in pool_mapping.items():
            src = self_by_view.get((self_lg, self_pi))
            if src is None:  # a view the describer left out for this piece
                continue
            dst = peer_by_view[(peer_lg, peer_pi)]
            if not self.should_send_pool(targets, peer_ri, self_lg, self_pi):
                continue

            # Build regions: STATE = per-layer ptrs of the one slot, PAGED = per-block ptrs
            if src.kind == CacheKind.STATE:
                if len(src) == 0 or len(dst) == 0:
                    continue
                if not MambaPolicy.is_paired(self_ri, peer_ri):
                    continue
                src_region = self._state_region("", self_ri.page_table, src)
                dst_region = self._state_region(
                    self._unique_key(peer_ri.instance_name, peer_ri.instance_rank),
                    peer_ri.page_table,
                    dst,
                )
            else:
                src_sel, dst_sel = self._pair_rows(
                    src, dst, dst_start_block=dst_start_block, helix=helix
                )
                if src_sel.size == 0:
                    continue
                src_region = SpecRegion(MemRegionGroup(src.addresses[src_sel], src.row_bytes))
                dst_region = SpecRegion(MemRegionGroup(dst.addresses[dst_sel], dst.row_bytes))

            # Map and collect fragments (same for all layer types)
            mapper = self.get_kv_map(peer_ri, (self_lg, self_pi), (peer_lg, peer_pi))
            region_pair = mapper.map(src_region, dst_region)
            region_pairs = region_pair if isinstance(region_pair, list) else [region_pair]
            for rp in region_pairs:
                src_frag_parts.append(rp.src.memory.ptrs)
                dst_frag_parts.append(rp.dst.memory.ptrs)
                size_specs.append((rp.src.memory.ptrs.size, rp.src.memory.bytes_per_region))

        if not src_frag_parts:
            return TransferPlan.empty()
        # Build the sizes array in one shot: np.repeat expands each
        # bytes_per_region value by its count, e.g.:
        #   values=[4096, 8192], counts=[100, 200]
        #   -> [4096]*100 ++ [8192]*200
        counts, values = zip(*size_specs)
        return TransferPlan(
            src_ptrs=np.concatenate(src_frag_parts),
            dst_ptrs=np.concatenate(dst_frag_parts),
            sizes=np.repeat(np.array(values, dtype=np.int64), counts),
        )

    def _state_region(
        self, peer_key: str, page_table: KVCachePageTable, rows: LayerGroupRows
    ) -> SpecRegion:
        """Per-layer pointers of a STATE row: ``slot address + view offset of each layer``.

        View offsets are explicit because a recurrent role may exist on only a
        subset of layers or be interleaved with another role in a V2 pool.
        """
        key = (peer_key, rows.layer_group_index, rows.view_index)
        layout = self._state_layout_cache.get(key)
        if layout is None:
            pv = page_table.layer_groups[rows.layer_group_index].pool_views[rows.view_index]
            layer_offsets, bytes_per_layer = get_layer_byte_ranges(pv)
            layout = (np.array(sorted(layer_offsets.values()), dtype=np.int64), bytes_per_layer)
            self._state_layout_cache[key] = layout
        ordered_offsets, bytes_per_layer = layout
        return SpecRegion(MemRegionGroup(ordered_offsets + int(rows.addresses[0]), bytes_per_layer))

    def receiver_payload_bytes(
        self, self_ri: RankInfo, peer_ri: RankInfo, req_info: RecvReqInfo
    ) -> Optional[int]:
        receiver_pt = self._self_ext_cache.page_table
        dst_slot = self._get_mamba_slot(receiver_pt, req_info.block_ids_per_layer_groups)
        if dst_slot is None:
            return 0
        if peer_ri.page_table is None:
            return None  # cannot size the recurrent-state payload
        return mamba_receiver_payload_bytes(
            sender_page_table=peer_ri.page_table,
            receiver_page_table=receiver_pt,
            dst_slot=dst_slot,
        )

    def forget_peer(self, peer_ri: RankInfo) -> None:
        key = self._unique_key(peer_ri.instance_name, peer_ri.instance_rank)
        self._lg_pool_mapping_cache.pop(key, None)
        for k in [k for k in self._kv_map_cache if k[0] == key]:
            del self._kv_map_cache[k]
        for k in [k for k in self._state_layout_cache if k[0] == key]:
            del self._state_layout_cache[k]

    # ---- helpers moved from the transfer framework ---------------------------------

    @staticmethod
    def _get_mamba_slot(
        page_table: Optional[KVCachePageTable], dst_block_ids_per_lg: List[np.ndarray]
    ) -> Optional[int]:
        """The receiver slot of the first STATE layer group, or None when there is none."""
        if page_table is None:
            return None
        for lg_idx, lg in enumerate(page_table.layer_groups):
            if lg.kind == CacheKind.STATE:
                block_ids = dst_block_ids_per_lg[lg_idx]
                if block_ids.size > 0:
                    return int(block_ids[0])
        return None

    @staticmethod
    def _pair_rows(
        src: LayerGroupRows,
        dst: LayerGroupRows,
        *,
        dst_start_block: int = 0,
        helix: Optional[Tuple[int, int]] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pair the rows both sides hold, ordinal by ordinal; returns row indices into each side.

        Rows are keyed by block ordinal, and a side lists only the ordinals it
        holds, so a block moves exactly where both sides hold it: a sender that
        has not evicted yet, a receiver that already reused a prefix, a receiver
        with a wider speculative window, or a sender chunk that blanks everything
        outside its range all fall out of the same intersection without any
        per-case arithmetic. Ordinals below ``dst_start_block`` are excluded as
        well. Under ``helix=(cp_rank, cp_size)`` the receiver holds every
        ``cp_size``-th global ordinal from ``cp_rank``, numbered from 0.
        """
        src_ordinals = src.block_ordinals
        src_index = np.arange(src_ordinals.size)
        src_span = src.span_blocks
        if helix is not None:
            cp_rank, cp_size = helix
            keep = (src_ordinals >= cp_rank) & ((src_ordinals - cp_rank) % cp_size == 0)
            src_index = src_index[keep]
            src_ordinals = (src_ordinals[keep] - cp_rank) // cp_size
            src_span = len(range(cp_rank, src_span, cp_size))
        if src_span != dst.span_blocks:
            raise ValueError(
                f"positional block tables differ in length: src={src_span} "
                f"dst={dst.span_blocks}; both sides must span the same prompt"
            )
        common, src_pos, dst_pos = np.intersect1d(
            src_ordinals, dst.block_ordinals, assume_unique=True, return_indices=True
        )
        accepted = common >= dst_start_block
        return src_index[src_pos[accepted]], dst_pos[accepted]

    @staticmethod
    def _fanin_bounce_safe(
        overlap: PeerOverlap,
        peer_ri: RankInfo,
        receiver_page_table: Optional[KVCachePageTable],
    ) -> bool:
        """Whether multi-writer bounce's equal total//num_writers split is valid for this overlap.
        The split assumes every writer contributes the same size, which holds when:
          * duplicate_head_factor == 1 -- else some ranks don't send KV yet still count in
            expected_transfers, so the live writers overflow their slots;
          * the PP layer split is even -- a single PP stage (overlap_pp_size <= 1) is trivially fine;
            for PP fan-in, every overlapping stage must hold the same number of layers
            (peer_ri.layer_num_per_pp all-equal) or per-writer sizes differ. If that full per-stage
            list isn't available (shorter than the fan-in degree), be conservative and fall back.
        Otherwise fall back to the per-fragment path (correct, just not coalesced).
        Equal layer count means equal bytes only when the per-block sizes match; reserve() rejects
        the mismatched case, so this only needs the count to split evenly."""
        if overlap.duplicate_head_factor != 1:
            return False
        if overlap.overlap_pp_size > 1:
            lpp = getattr(peer_ri, "layer_num_per_pp", None)
            if not lpp or len(lpp) < overlap.overlap_pp_size or len(set(lpp)) != 1:
                return False
        # Replicated pools (e.g. MiniMax M3 index-key) are sent by one elected
        # fan-in owner only, so with multiple writers their contributions
        # differ in size and the equal split is invalid. Inspect both endpoints:
        # a masked PP stage may advertise no replicated view even though another
        # stage owns one that is visible in the receiver's page table.
        if len(overlap.ranks) > 1:
            for page_table in (peer_ri.page_table, receiver_page_table):
                if page_table is None:
                    continue
                for layer_group in page_table.layer_groups:
                    for pool_view in getattr(layer_group, "pool_views", ()):
                        if pool_view.mapper_kind == MapperKind.REPLICATED:
                            return False
        return True

    # ---- pool matching, mappers and topology overlap ------------------------------

    def get_pool_mapping(self, peer_ri: RankInfo) -> Dict[LGPoolKey, LGPoolKey]:
        """Get mapping from (self_lg_idx, self_pool_idx) -> (peer_lg_idx, peer_pool_idx).

        Two-step matching:
        1. Find peer layer_group via layer_to_layer_group (global_layer_id -> lg_idx).
        2. Within the matched peer layer_group, find the unique peer pool whose
           ``PoolView.pool_role`` equals self's and whose global_layer_ids
           overlap.

        Layer-overlap is required: a peer pool with the same pool_role but
        zero layer overlap with self is *not* a match — the two pools cover
        disjoint layers and have nothing to transfer.

        A self layer group never matches multiple peer layer groups, so the
        result is one peer pool per self pool. Layer groups partition each
        rank's layers by attention/life-cycle class, which both sides derive
        from the same model config; PP only changes which layers overlap (the
        fan-out across peer PP ranks is handled by calling this method once
        per peer rank). Step 1 raises if this invariant is ever violated.
        """
        key = self._unique_key(peer_ri.instance_name, peer_ri.instance_rank)
        if key in self._lg_pool_mapping_cache:
            return self._lg_pool_mapping_cache[key]

        mapping: Dict[LGPoolKey, LGPoolKey] = {}
        self_pt = self._self_ext_cache.page_table
        peer_pt = peer_ri.page_table

        if self_pt is None or peer_pt is None:
            self._lg_pool_mapping_cache[key] = mapping
            return mapping
        if not self_pt.layer_groups or not peer_pt.layer_groups:
            self._lg_pool_mapping_cache[key] = mapping
            return mapping

        # Build kind-specific peer layer-to-group mappings lazily. In hybrid
        # models (e.g. Qwen3Next) a global_layer_id may appear in both a
        # PAGED and a STATE layer group, so the lookup must be scoped to
        # the same CacheKind as self's layer group.
        _peer_l2g_cache: Dict[CacheKind, Dict[int, int]] = {}

        def _peer_l2g(kind: CacheKind) -> Dict[int, int]:
            if kind not in _peer_l2g_cache:
                _peer_l2g_cache[kind] = get_layer_to_layer_group(peer_pt, kind)
            return _peer_l2g_cache[kind]

        for self_lg_idx, self_lg in enumerate(self_pt.layer_groups):
            if self_lg.kind not in (CacheKind.PAGED, CacheKind.STATE):
                continue
            for self_pi, self_pv in enumerate(self_lg.pool_views):
                # Every view carries buffer_entries, so a view's exact layer
                # set always comes from its entries (a view may cover a
                # subset of the LG when V2 splits an LG into multiple pools
                # by buffer-size class, or when a role class exists only on
                # some layers, e.g. sparse-layer index-K).
                pv_global_ids = get_pool_view_global_layer_ids(self_pv, self_lg)
                if not pv_global_ids:
                    continue

                # Step 1: find the peer layer_group via overlapping global_layer_ids.
                # A self layer group (hence each of its pool views) never matches
                # multiple peer layer groups: layer groups partition a rank's
                # layers by attention/life-cycle class, both sides derive that
                # class from the same model config, and global ids are
                # PP-invariant. So PP only changes WHICH layers overlap — layers
                # the peer doesn't hold are a legal skip (PP slices, one-sided
                # MTP layers) — never how many peer LGs they land in; the PP
                # fan-out is handled by per-peer-rank calls of this method. A
                # multi-LG hit therefore means the two peers group layers
                # differently (unsupported topology), and we fail loudly instead
                # of silently transferring only the first LG's overlap.
                peer_layer_to_group = _peer_l2g(self_lg.kind)
                peer_lg_indices = {
                    peer_layer_to_group[g] for g in pv_global_ids if g in peer_layer_to_group
                }
                if not peer_lg_indices:
                    continue
                if len(peer_lg_indices) > 1:
                    raise ValueError(
                        "DefaultTransferPlanner.get_pool_mapping: pool view "
                        f"(lg={self_lg_idx}, pool={self_pi}) spans multiple peer "
                        f"layer groups {sorted(peer_lg_indices)}; mismatched layer "
                        "grouping between peers is not supported"
                    )
                peer_lg_idx = next(iter(peer_lg_indices))
                peer_lg = peer_pt.layer_groups[peer_lg_idx]

                # Step 2: pick the first peer pool with the same pool_role
                # whose layers overlap self's (zero-overlap pools cover
                # disjoint layers — nothing to transfer).
                #
                # Uniqueness assumption: at most one peer pool can match on
                # both ``pool_role`` (frozenset equality) and layer overlap.
                # We do *not* assume ``pool_role`` is unique within a peer LG
                # — V2 may split an LG into multiple same-role pools by
                # buffer-size class (e.g. VSWA). What we rely on is that both
                # peers run the same pool-grouping logic, so for every self_pv
                # there is exactly one peer pool with the same role *and* an
                # overlapping layer set; other same-role peer pools cover
                # disjoint layers and fall out via the overlap filter.
                self_layer_set = set(pv_global_ids)
                matched_peer_pi = None
                for peer_pi, peer_pv in enumerate(peer_lg.pool_views):
                    if peer_pv.pool_role != self_pv.pool_role:
                        continue
                    peer_global_ids = get_pool_view_global_layer_ids(peer_pv, peer_lg)
                    if not set(peer_global_ids) & self_layer_set:
                        continue
                    if peer_pv.mapper_kind != self_pv.mapper_kind:
                        raise ValueError(
                            "DefaultTransferPlanner.get_pool_mapping: incompatible mapper "
                            f"kinds for pool role {sorted(self_pv.pool_role)} "
                            f"(local={self_pv.mapper_kind.name}, "
                            f"peer={peer_pv.mapper_kind.name}, peer_pool={peer_pi})"
                        )
                    matched_peer_pi = peer_pi
                    break

                if matched_peer_pi is not None:
                    mapping[(self_lg_idx, self_pi)] = (peer_lg_idx, matched_peer_pi)

        self._lg_pool_mapping_cache[key] = mapping
        return mapping

    def _get_policy(self, kind: CacheKind) -> Union[AttentionPolicy, MambaPolicy]:
        """Return the policy for a CacheKind (lazily instantiated)."""
        if kind not in self._policies:
            self._policies[kind] = self._POLICY_CLASSES[kind](self._ri)
        return self._policies[kind]

    def get_kv_map(
        self,
        peer_ri: RankInfo,
        self_pool_key: LGPoolKey,
        peer_pool_key: LGPoolKey,
    ) -> RegionMapperBase:
        """Get mapper for a specific pool pair.

        Args:
            peer_ri: Peer rank info.
            self_pool_key: (self_lg_idx, self_pool_idx).
            peer_pool_key: (peer_lg_idx, peer_pool_idx).
        """
        peer_key = self._unique_key(peer_ri.instance_name, peer_ri.instance_rank)
        cache_key = (peer_key, self_pool_key, peer_pool_key)
        if cache_key in self._kv_map_cache:
            return self._kv_map_cache[cache_key]

        self_pt = self._self_ext_cache.page_table
        peer_pt = peer_ri.page_table
        assert self_pt is not None
        assert peer_pt is not None
        self_lg_idx, self_pi = self_pool_key
        peer_lg_idx, peer_pi = peer_pool_key
        self_lg = self_pt.layer_groups[self_lg_idx]
        peer_lg = peer_pt.layer_groups[peer_lg_idx]
        self_pv = self_lg.pool_views[self_pi]
        peer_pv = peer_lg.pool_views[peer_pi]

        if self_pv.mapper_kind != peer_pv.mapper_kind:
            raise ValueError(
                "DefaultTransferPlanner.get_kv_map: incompatible mapper kinds "
                f"(local={self_pv.mapper_kind.name}, peer={peer_pv.mapper_kind.name})"
            )

        # Every view is entries-driven: resolve the overlap layers to
        # slot-relative byte offsets on each side from the views' buffer
        # entries. Layer selection is explicit, so mappers never assume a
        # uniform layer stride (other role classes may interleave), and no
        # convention about global-id/byte-offset ordering is needed.
        self_global_ids = get_pool_view_global_layer_ids(self_pv, self_lg)
        peer_global_ids = get_pool_view_global_layer_ids(peer_pv, peer_lg)
        # Iterate the overlap in self's physical slot order (not sorted by
        # global id) so that layers whose regions are contiguous on both
        # sides stay adjacent in the offset arrays and the mappers can merge
        # them into one fragment even when global-id order diverges from the
        # physical layout. Order only affects run merging (a perf property):
        # each layer's byte offset is looked up explicitly below, so any
        # iteration order transfers correct bytes.
        overlap = set(self_global_ids) & set(peer_global_ids)
        overlapping_layers = [gid for gid in self_global_ids if gid in overlap]

        self_starts, self_bytes_per_layer = get_layer_byte_ranges(self_pv)
        peer_starts, peer_bytes_per_layer = get_layer_byte_ranges(peer_pv)
        self_g2l = {ll.global_layer_id: ll.local_layer_id for ll in self_lg.local_layers}
        peer_g2l = {ll.global_layer_id: ll.local_layer_id for ll in peer_lg.local_layers}
        self_layer_offsets = np.array(
            [self_starts[self_g2l[gid]] for gid in overlapping_layers], dtype=np.int64
        )
        peer_layer_offsets = np.array(
            [peer_starts[peer_g2l[gid]] for gid in overlapping_layers], dtype=np.int64
        )
        # Per-layer buffer count (K and V are separate buffers within a
        # layer's region); head-mismatch mappers slice heads inside each.
        self_buffers_per_layer = self._get_buffers_per_layer(
            self_pv,
            layer_group_id=self_lg_idx,
            pool_idx=self_pi,
        )
        peer_buffers_per_layer = self._get_buffers_per_layer(
            peer_pv,
            layer_group_id=peer_lg_idx,
            pool_idx=peer_pi,
        )

        # Polymorphic dispatch: attention vs mamba policy
        policy = self._get_policy(self_lg.kind)

        # For mamba, compute ptr-array layer offsets for partial PP overlap.
        # _state_region returns ptrs for ALL local_layer_ids in sorted order;
        # the mapper must slice only the overlapping subset.
        extra_kwargs = {}
        if self_lg.kind == CacheKind.STATE:
            # Positions index the ptr array ``_state_region`` builds, which is
            # ordered by view offset.
            self_gid_to_pos = {gid: i for i, gid in enumerate(self_global_ids)}
            peer_gid_to_pos = {gid: i for i, gid in enumerate(peer_global_ids)}
            self_overlap_positions = [self_gid_to_pos[gid] for gid in overlapping_layers]
            peer_overlap_positions = [peer_gid_to_pos[gid] for gid in overlapping_layers]
            # Under contiguous PP partitioning, overlapping layers form a
            # contiguous block in the ptrs array.
            extra_kwargs["src_layer_off"] = (
                self_overlap_positions[0] if self_overlap_positions else 0
            )
            extra_kwargs["dst_layer_off"] = (
                peer_overlap_positions[0] if peer_overlap_positions else 0
            )

        mapper = policy.build_mapper(
            peer_ri=peer_ri,
            mapper_kind=self_pv.mapper_kind,
            self_layer_offsets=self_layer_offsets,
            peer_layer_offsets=peer_layer_offsets,
            self_bytes_per_layer=self_bytes_per_layer,
            peer_bytes_per_layer=peer_bytes_per_layer,
            self_buffers_per_layer=self_buffers_per_layer,
            peer_buffers_per_layer=peer_buffers_per_layer,
            self_lg=self_lg,
            peer_lg=peer_lg,
            self_pv=self_pv,
            peer_pv=peer_pv,
            **extra_kwargs,
        )

        self._kv_map_cache[cache_key] = mapper
        return mapper

    @staticmethod
    def _get_buffers_per_layer(
        pool_view: PoolView,
        *,
        layer_group_id: int,
        pool_idx: int,
    ) -> int:
        """Per-layer buffer count of a view (e.g. K+V -> 2, key-only -> 1).

        Views are bucketed per (layer group, pool, mapper kind) at page-table
        build time, so every layer in a view carries the same role set and
        hence the same entry count — a skewed distribution should never occur.
        Still verify it per layer rather than via total-count divisibility:
        e.g. 1 + 3 entries over two layers passes ``total % layers == 0`` yet
        would make head-slicing mappers split every layer at wrong offsets.
        """
        entries = pool_view.buffer_entries
        if len(entries) == 0:
            return 1
        counts = Counter(int(e["local_layer_id"]) for e in entries)
        distinct = set(counts.values())
        if len(distinct) != 1:
            raise ValueError(
                "PoolView buffer entries are not evenly distributed across layers: "
                f"layer_group={layer_group_id}, pool={pool_idx}, "
                f"per-layer entry counts={sorted(counts.items())}"
            )
        return distinct.pop()

    @staticmethod
    def _find_overlap(self_val, peer_val, self_rank, peer_rank=None):
        if self_val <= peer_val:
            overlap = peer_val // self_val
            start = self_rank * overlap + (peer_rank * peer_val if peer_rank is not None else 0)
            end = start + overlap
        else:
            ratio = self_val // peer_val
            start = (self_rank // ratio) + (peer_rank * peer_val if peer_rank is not None else 0)
            overlap = 1
            end = start + overlap

        return overlap, start, end

    def get_peer_overlap(self, peer_rank_info: RankInfo, peer_dp_rank: int) -> PeerOverlap:
        peer_ri = peer_rank_info
        key = self._unique_key(peer_ri.instance_name, peer_dp_rank)
        if key in self._overlap_cache:
            return self._overlap_cache[key]

        # compute pp overlap and target layers
        self_start_layer = sum(self._ri.layer_num_per_pp[: self._ri.pp_rank])
        self_end_layer = self_start_layer + self._ri.layer_num_per_pp[self._ri.pp_rank]

        pre = 0
        tgt_pp_ranks: List[int] = []
        for p in range(peer_ri.pp_size):
            peer_start_layer = pre
            peer_end_layer = peer_start_layer + peer_ri.layer_num_per_pp[p]
            if self_start_layer < peer_end_layer and self_end_layer > peer_start_layer:
                tgt_pp_ranks.append(p)
            pre += peer_ri.layer_num_per_pp[p]

        if tgt_pp_ranks == []:
            targets = PeerOverlap()
            self._overlap_cache[key] = targets
            return targets

        peer_start_pp = tgt_pp_ranks[0]
        overlap_pp_size = len(tgt_pp_ranks)
        peer_end_pp = peer_start_pp + overlap_pp_size

        self_tp_per_dp = self._ri.tp_size_per_dp_group
        peer_tp_per_dp = peer_ri.tp_size_per_dp_group
        self_tp_rank_in_dp = self._ri.tp_rank % self_tp_per_dp

        overlap_tp_size, peer_start_tp, peer_end_tp = self._find_overlap(
            self_tp_per_dp, peer_tp_per_dp, self_tp_rank_in_dp, peer_dp_rank
        )
        overlap_cp_size, peer_start_cp, peer_end_cp = self._find_overlap(
            self._ri.cp_size, peer_ri.cp_size, self._ri.cp_rank
        )

        ranks: List[int] = []
        for pp in range(peer_start_pp, peer_end_pp):
            for tp in range(peer_start_tp, peer_end_tp):
                for cp in range(peer_start_cp, peer_end_cp):
                    # CP-minor flat rank (ppRank*(TP*CP) + tpRank*CP +
                    # cpRank), matching Mapping and the C++ transceiver; a
                    # TP-minor formula mis-routes when tp > 1 and cp > 1.
                    # Loop nesting mirrors the layout so ranks come out in
                    # ascending order.
                    ranks.append(pp * peer_ri.tp_size * peer_ri.cp_size + tp * peer_ri.cp_size + cp)

        dup_head, peer_dup_head = self._get_policy(CacheKind.PAGED).duplicate_head_factors(peer_ri)

        targets = PeerOverlap(
            overlap_pp_size=overlap_pp_size,
            overlap_tp_size=overlap_tp_size,
            overlap_cp_size=overlap_cp_size,
            duplicate_head_factor=dup_head,
            peer_duplicate_head_factor=peer_dup_head,
            ranks=ranks,
        )
        self._overlap_cache[key] = targets
        return targets

    def _owns_tp_fan_in(self, peer_rank_info: RankInfo) -> bool:
        """Elect one owner when replicated bytes fan in across TP ranks.

        A peer with fewer TP shards receives identical replicated data from
        several local ranks. Rotate the elected owner by the destination's
        DP rank (mirroring ``should_send_kv``'s head-duplication pairing) so
        that with a multi-DP-group generation side the extra replicated
        traffic spreads across local ranks instead of always landing on the
        first rank of each fan-in group.
        """
        ratio = max(
            1,
            self._ri.tp_size_per_dp_group // peer_rank_info.tp_size_per_dp_group,
        )
        self_tp_rank = self._ri.tp_rank % self._ri.tp_size_per_dp_group
        return self_tp_rank % ratio == peer_rank_info.dp_rank % ratio

    def should_send_pool(
        self,
        peer_overlap: PeerOverlap,
        peer_rank_info: RankInfo,
        layer_group_id: int,
        pool_idx: int,
    ) -> bool:
        """Return whether this rank owns the transfer of one view pair.

        ``pool_idx`` indexes the layer group's ``pool_views`` list (one view
        per role class; several views may share a physical pool). Each view
        is kind-homogeneous, so ownership is a single per-view decision:
        replicated views use one sender per fan-in group (the policy returns
        None for them), sharded views retain head-duplication routing.

        For mamba layer groups, ownership is based on mamba's own TP routing
        (independent of attention's duplicate-head logic):
        - If mamba_tp == 1 (attention_dp enabled): fan-in election (one sender)
        - Otherwise: always send (each rank owns unique TP-sharded state)
        """
        layer_group = self._self_ext_cache.page_table.layer_groups[layer_group_id]
        pool_view = layer_group.pool_views[pool_idx]
        # Delegate to the policy's should_send. None means fan-in election.
        policy = self._get_policy(layer_group.kind)
        result = policy.should_send(peer_overlap, peer_rank_info, mapper_kind=pool_view.mapper_kind)
        return self._owns_tp_fan_in(peer_rank_info) if result is None else result
