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

import dataclasses
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.native.auxiliary import AuxTransferLayout
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.native.runs import RunsMapper
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.page import CacheKind, KVCachePageTable
from tensorrt_llm._torch.disaggregation.resource.utils import (
    get_layer_to_layer_group,
    get_pool_view_global_layer_ids,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.layout_mapping import (
    BufferRuns,
    check_compatible,
    fan,
    group_kinds,
    map_buffers,
    merge_runs,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.page_layout import PageLayout

# Type alias for (lg_idx, pool_idx) pair
LGPoolKey = Tuple[int, int]


@dataclass
class PeerOverlap:
    overlap_pp_size: int = 0
    overlap_tp_size: int = 0
    overlap_cp_size: int = 0
    ranks: List[int] = field(default_factory=list)


class PeerRegistrar:
    def __init__(self, self_rank_info: RankInfo, self_extractor: KVRegionExtractorV1):
        self._ri = self_rank_info
        self._peer_ri_cache: Dict[str, RankInfo] = {}
        self._kv_map_cache: Dict[
            tuple, RunsMapper
        ] = {}  # key: (peer_key, self_lg_pool_key, peer_lg_pool_key)
        # peer_key -> what this rank sends that peer rank, buffer by buffer.
        self._buffer_runs_cache: Dict[str, List[BufferRuns]] = {}
        self._self_ext_cache = self_extractor
        self._peer_ext_cache: Dict[str, KVRegionExtractorV1] = {}
        self._overlap_cache: Dict[str, PeerOverlap] = {}
        # (peer instance, writer ranks) -> whether those writers send alike.
        self._writers_alike_cache: Dict[Tuple[str, Tuple[int, ...]], bool] = {}
        self._aux_transfer_layout_cache: Dict[str, AuxTransferLayout] = {}
        self._lg_pool_mapping_cache: Dict[
            str, Dict[LGPoolKey, LGPoolKey]
        ] = {}  # peer_key -> {(self_lg, self_pi) -> (peer_lg, peer_pi)}

    def register(self, peer_name: str, peer_rank: int, peer_ri: RankInfo):
        assert self._self_ext_cache is not None
        self.check_peer_compatible(peer_ri)
        key = self._unique_key(peer_name, peer_rank)
        self._peer_ri_cache[key] = peer_ri
        self._aux_transfer_layout_cache.pop(key, None)
        peer_ri = self.get_peer_rank_info(peer_name, peer_rank)
        extractor = KVRegionExtractorV1(peer_ri.page_table)
        self._peer_ext_cache[key] = extractor
        self._warn_fragmented_pages(peer_ri)

    def _warn_fragmented_pages(self, peer_ri: RankInfo) -> None:
        """Warn when the paged buffers this rank sends the peer move in more fragments than there
        are buffers: the two sides order a buffer's bytes differently, such as token-major KV over
        different head counts, and every page then costs that many transfer descriptors."""
        own = self._self_ext_cache.page_table
        buffers = fragments = 0
        for (lg, pi), peer_key in self.get_pool_mapping(peer_ri).items():
            if own.layer_groups[lg].kind != CacheKind.PAGED:
                continue
            mapper = self.get_kv_map(peer_ri, (lg, pi), peer_key)
            if not mapper.empty:
                buffers += len(own.layer_groups[lg].pool_views[pi].buffer_entries)
                fragments += mapper.runs_per_slot
        if fragments > buffers:
            logger.warning_once(
                f"Disaggregated KV transfer to {peer_ri.instance_name} moves each page in "
                f"{fragments} fragments for {buffers} buffers, as the two sides order the buffers' "
                "bytes differently (such as token-major KV over different head counts); every page "
                "costs that many NIXL descriptors per peer, so long-context transfers may have high "
                "latency.",
                key=f"native-fragmented-pages-{fragments}-{buffers}",
            )

    def peer_extractor(self, peer_name: str, peer_rank: int) -> KVRegionExtractorV1:
        return self._peer_ext_cache[self._unique_key(peer_name, peer_rank)]

    @property
    def self_extractor(self) -> KVRegionExtractorV1:
        assert self._self_ext_cache is not None
        return self._self_ext_cache

    def unregister(self, peer_name: str, peer_rank: int):
        key = self._unique_key(peer_name, peer_rank)
        if key in self._peer_ri_cache:
            del self._peer_ri_cache[key]
        if key in self._peer_ext_cache:
            del self._peer_ext_cache[key]
        self._aux_transfer_layout_cache.pop(key, None)
        # Clean up kv_map_cache entries for this peer
        keys_to_remove = [k for k in self._kv_map_cache if k[0] == key]
        for k in keys_to_remove:
            del self._kv_map_cache[k]
        self._buffer_runs_cache.pop(key, None)
        if key in self._lg_pool_mapping_cache:
            del self._lg_pool_mapping_cache[key]

    def get_peer_rank_info(self, peer_name: str, peer_rank: int):
        return self._peer_ri_cache[self._unique_key(peer_name, peer_rank)]

    def get_aux_transfer_layout(
        self, peer_name: str, peer_rank: int
    ) -> Optional[AuxTransferLayout]:
        return self._aux_transfer_layout_cache.get(self._unique_key(peer_name, peer_rank))

    def cache_aux_transfer_layout(
        self, peer_name: str, peer_rank: int, layout: AuxTransferLayout
    ) -> None:
        self._aux_transfer_layout_cache[self._unique_key(peer_name, peer_rank)] = layout

    @property
    def self_rank_info(self) -> RankInfo:
        return self._ri

    def _unique_key(self, name: str, rank: int) -> str:
        return name + str(rank)

    def validate_bounded_replay_compatible(self, peer_ri: RankInfo) -> None:
        """Require the same generation replay mode on both peers."""
        if self._ri.bounded_replay_on_generation != peer_ri.bounded_replay_on_generation:
            raise ValueError(
                "PeerRegistrar: bounded_replay_on_generation must match between peers "
                f"(local={self._ri.bounded_replay_on_generation}, "
                f"peer={peer_ri.bounded_replay_on_generation})."
            )

    def check_peer_compatible(self, peer_ri: RankInfo) -> None:
        """Check that this rank and the peer rank can exchange pages: the same generation replay
        mode, context parallelism on at most one side (Helix pairs a context instance without it
        with a generation instance using it), and page layouts that correspond
        (``layout_mapping.check_compatible``). Ranks holding different layers are allowed: only
        the layers both hold move.

        Raises:
            ValueError: Naming what differs.
        """
        self.validate_bounded_replay_compatible(peer_ri)
        if self._ri.cp_size > 1 and peer_ri.cp_size > 1:
            raise ValueError(
                "PeerRegistrar: context parallelism on both sides is unsupported "
                f"(local cp_size={self._ri.cp_size}, peer cp_size={peer_ri.cp_size})"
            )
        check_compatible(
            _layout_of(self._self_ext_cache.page_table), _layout_of(peer_ri.page_table)
        )

        self_layers = sum(self._ri.layer_num_per_pp)
        peer_layers = sum(peer_ri.layer_num_per_pp)
        if self_layers != peer_layers:
            # Allow mismatch when one side has speculative (e.g. MTP) layers
            # that the other side doesn't. The pool_mapping logic will only
            # transfer layers that exist on both sides.
            logger.warning(
                "PeerRegistrar: layer count differs "
                f"(local={self_layers}, peer={peer_layers}), "
                "allowing partial layer transfer."
            )

    def writers_send_alike(self, peer_ri: RankInfo, overlap: PeerOverlap) -> bool:
        """Whether the peer ranks writing this rank's pages (``overlap.ranks``) each send it the
        same bytes per page of its attention groups, so that one region split evenly among them
        holds each writer's share.

        Only the writers on the stage whose page layout the peer rank info carries are computed,
        each from that layout with the writer's place put in, as the ranks of one stage lay out
        pages alike apart from where they stand. Writers on other stages may hold other layers,
        so with several stages any attention buffer every rank holds whole, of which one writer
        sends all, counts as unequal.

        Raises:
            ValueError: If a writer's pages do not correspond to this rank's (``map_buffers``).
        """
        cache_key = (peer_ri.instance_name, tuple(overlap.ranks))
        if cache_key in self._writers_alike_cache:
            return self._writers_alike_cache[cache_key]
        own = _layout_of(self._self_ext_cache.page_table)
        template = _layout_of(peer_ri.page_table)
        kinds = group_kinds(own)
        place = template.topology
        per_stage = peer_ri.tp_size * peer_ri.cp_size
        sent = set()
        for rank in overlap.ranks:
            pp, rest = divmod(rank, per_stage)
            if pp != place.pp_rank:
                continue
            tp, cp = divmod(rest, peer_ri.cp_size)
            writer = dataclasses.replace(
                place, tp_rank=tp, cp_rank=cp, dp_rank=tp if place.attention_dp else place.dp_rank
            )
            sent.add(
                sum(
                    int(runs.sizes.sum())
                    for runs in map_buffers(
                        dataclasses.replace(template, topology=writer),
                        own,
                        key=own.topology.dp_rank,
                    )
                    if kinds.get((runs.dst.pool_group, runs.dst.layer)) == "attention"
                )
            )
        alike = len(sent) <= 1
        if alike and overlap.overlap_pp_size > 1 and len(overlap.ranks) > 1:
            alike = not any(
                b.transferred and not b.split and kinds.get((b.pool_group, b.layer)) == "attention"
                for b in own.buffers
            )
        self._writers_alike_cache[cache_key] = alike
        return alike

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
                        "PeerRegistrar.get_pool_mapping: pool view "
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
                            "PeerRegistrar.get_pool_mapping: incompatible mapper "
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

    def get_kv_map(
        self,
        peer_ri: RankInfo,
        self_pool_key: LGPoolKey,
        peer_pool_key: LGPoolKey,
        excluded: frozenset = frozenset(),
    ) -> RunsMapper:
        """What this rank sends the peer rank of one view pair: the bytes of the views' buffers
        that ``layout_mapping`` elects this rank to send, relative to each side's pool slot. An
        ``empty`` mapper means this rank sends nothing of the pair.

        Args:
            peer_ri: Peer rank info.
            self_pool_key: (self_lg_idx, self_pool_idx).
            peer_pool_key: (peer_lg_idx, peer_pool_idx).
            excluded: Buffers, as (global layer id, role), this transfer leaves out.
        """
        peer_key = self._unique_key(peer_ri.instance_name, peer_ri.instance_rank)
        cache_key = (peer_key, self_pool_key, peer_pool_key, excluded)
        if cache_key in self._kv_map_cache:
            return self._kv_map_cache[cache_key]

        self_pt = self._self_ext_cache.page_table
        peer_pt = peer_ri.page_table
        runs = self._buffer_runs_cache.get(peer_key)
        if runs is None:
            # Ties between ranks holding the same bytes rotate by the destination's DP rank.
            runs = map_buffers(_layout_of(self_pt), _layout_of(peer_pt), key=peer_ri.dp_rank)
            self._buffer_runs_cache[peer_key] = runs
        self_view = _view_buffers(self_pt, *self_pool_key)
        peer_view = _view_buffers(peer_pt, *peer_pool_key)
        parts = [
            r
            for r in runs
            if (r.src.layer, r.src.pool_index, r.src.slot_offset) in self_view
            and (r.dst.layer, r.dst.pool_index, r.dst.slot_offset) in peer_view
            and (r.src.layer, r.src.role) not in excluded
        ]
        if parts:
            mapper = RunsMapper(
                *merge_runs(
                    np.concatenate([r.src_offsets for r in parts]),
                    np.concatenate([r.dst_offsets for r in parts]),
                    np.concatenate([r.sizes for r in parts]),
                )
            )
        else:
            mapper = RunsMapper(*(np.zeros(0, dtype=np.int64),) * 3)
        self._kv_map_cache[cache_key] = mapper
        return mapper

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

        # The peer ranks whose share of each dimension overlaps this rank's (layout_mapping.fan),
        # the same pairing map_buffers elects senders within; tensor ranks count within the
        # peer's data-parallel group peer_dp_rank.
        self_tp_per_dp = self._ri.tp_size_per_dp_group
        peer_tp_per_dp = peer_ri.tp_size_per_dp_group
        tp_fan = fan(peer_tp_per_dp, self_tp_per_dp, self._ri.tp_rank % self_tp_per_dp)
        cp_fan = fan(peer_ri.cp_size, self._ri.cp_size, self._ri.cp_rank)
        tp_offset = peer_dp_rank * peer_tp_per_dp

        ranks: List[int] = []
        for pp in range(peer_start_pp, peer_end_pp):
            for tp in range(tp_offset + tp_fan.start, tp_offset + tp_fan.stop):
                for cp in cp_fan:
                    # CP-minor flat rank (ppRank*(TP*CP) + tpRank*CP +
                    # cpRank), matching Mapping and the C++ transceiver; a
                    # TP-minor formula mis-routes when tp > 1 and cp > 1.
                    # Loop nesting mirrors the layout so ranks come out in
                    # ascending order.
                    ranks.append(pp * peer_ri.tp_size * peer_ri.cp_size + tp * peer_ri.cp_size + cp)

        targets = PeerOverlap(
            overlap_pp_size=overlap_pp_size,
            overlap_tp_size=len(tp_fan),
            overlap_cp_size=len(cp_fan),
            ranks=ranks,
        )
        self._overlap_cache[key] = targets
        return targets

    def should_send_aux(self, peer_rank_info: RankInfo) -> bool:
        # to ensure the transfer aux is not duplicated

        # TP: only the first rank in each peer-TP-sized group sends aux
        ratio = max(1, self._ri.tp_size_per_dp_group // peer_rank_info.tp_size_per_dp_group)
        self_tp_rank_in_dp_group = self._ri.tp_rank % self._ri.tp_size_per_dp_group
        should_send_in_tp = self_tp_rank_in_dp_group % ratio == 0

        # PP: only the first self-PP rank whose layers overlap with the peer's PP rank sends aux.
        # All tp/pp ranks have the same aux data, so pick the first overlapping one to avoid duplication.
        peer_start_layer = sum(peer_rank_info.layer_num_per_pp[: peer_rank_info.pp_rank])
        peer_end_layer = peer_start_layer + peer_rank_info.layer_num_per_pp[peer_rank_info.pp_rank]
        offset = 0
        for p, n in enumerate(self._ri.layer_num_per_pp):
            if offset < peer_end_layer and offset + n > peer_start_layer:
                return should_send_in_tp and p == self._ri.pp_rank
            offset += n
        return False


def _view_buffers(
    page_table: KVCachePageTable, layer_group_id: int, pool_idx: int
) -> set[tuple[int, int, int]]:
    """(global layer, pool, slot offset) of each buffer of one view."""
    group = page_table.layer_groups[layer_group_id]
    view = group.pool_views[pool_idx]
    global_of = {ll.local_layer_id: ll.global_layer_id for ll in group.local_layers}
    return {
        (global_of[int(e["local_layer_id"])], view.pool_idx, int(e["offset"]))
        for e in view.buffer_entries
    }


def _layout_of(page_table: Optional[KVCachePageTable]) -> PageLayout:
    """The page layout a page table carries."""
    if page_table is None or page_table.layout is None:
        raise ValueError(
            "PeerRegistrar: a page table carries no page layout; build it from the manager's page "
            "table (build_page_table_from_manager)"
        )
    return page_table.layout
