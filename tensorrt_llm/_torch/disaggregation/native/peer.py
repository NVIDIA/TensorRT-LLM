from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.base.region import RegionMapperBase
from tensorrt_llm._torch.disaggregation.native.mixers.attention.peer import (
    AttentionPolicy,
    PoolBufferMapper,
    PoolBufferMapping,
)
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
from tensorrt_llm._torch.disaggregation.resource.page import (
    AttentionLayerGroup,
    MapperKind,
    PoolView,
)
from tensorrt_llm._torch.disaggregation.resource.utils import (
    get_buffer_mapper_kinds,
    get_global_layer_ids,
    get_layer_group_num_layers,
    get_layer_to_layer_group,
    get_num_buffer_entries,
    get_physical_pool,
    get_pool_view_global_layer_ids,
    get_pool_view_num_layers,
)

# Type alias for (lg_idx, pool_idx) pair
LGPoolKey = Tuple[int, int]
PoolPair = Tuple[LGPoolKey, LGPoolKey]


@dataclass
class PeerOverlap:
    overlap_pp_size: int = 0
    overlap_tp_size: int = 0
    overlap_cp_size: int = 0
    duplicate_head_factor: int = 1
    peer_duplicate_head_factor: int = 1
    ranks: List[int] = field(default_factory=list)


class PeerRegistrar:
    def __init__(self, self_rank_info: RankInfo, self_extractor: KVRegionExtractorV1):
        self._ri = self_rank_info
        self._attention_policy = AttentionPolicy(self_rank_info)
        self._peer_ri_cache: Dict[str, RankInfo] = {}
        self._kv_map_cache: Dict[
            tuple, RegionMapperBase
        ] = {}  # key: (peer_key, self_lg_pool_key, peer_lg_pool_key)
        self._self_ext_cache = self_extractor
        self._peer_ext_cache: Dict[str, KVRegionExtractorV1] = {}
        self._overlap_cache: Dict[str, PeerOverlap] = {}
        self._lg_pool_mapping_cache: Dict[str, List[PoolPair]] = {}

    def register(self, peer_name: str, peer_rank: int, peer_ri: RankInfo):
        assert self._self_ext_cache is not None
        if not self._check_peer_compatible(peer_ri):
            raise ValueError(
                f"PeerRegistrar.register: peer {peer_name} (rank={peer_rank}) is incompatible with local rank."
            )
        key = self._unique_key(peer_name, peer_rank)
        self._peer_ri_cache[key] = peer_ri
        peer_ri = self.get_peer_rank_info(peer_name, peer_rank)
        extractor = KVRegionExtractorV1(peer_ri.page_table)
        self._peer_ext_cache[key] = extractor

        head_match, _ = self._attention_policy.head_match(peer_ri)
        if not head_match:
            self_page_table = self._self_ext_cache.page_table
            mapped_self_pools = {
                self_pool_key for self_pool_key, _ in self.get_pool_mapping(peer_ri)
            }
            nhd_fragments_per_token = sum(
                sum(
                    kind == MapperKind.NHD
                    for kind in get_buffer_mapper_kinds(
                        self_page_table.layer_groups[layer_group_id].pool_views[pool_idx]
                    )
                )
                for layer_group_id, pool_idx in mapped_self_pools
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
        # Clean up kv_map_cache entries for this peer
        keys_to_remove = [k for k in self._kv_map_cache if k[0] == key]
        for k in keys_to_remove:
            del self._kv_map_cache[k]
        if key in self._lg_pool_mapping_cache:
            del self._lg_pool_mapping_cache[key]

    def get_peer_rank_info(self, peer_name: str, peer_rank: int):
        return self._peer_ri_cache[self._unique_key(peer_name, peer_rank)]

    @property
    def self_rank_info(self) -> RankInfo:
        return self._ri

    def _unique_key(self, name: str, rank: int) -> str:
        return name + str(rank)

    def _check_peer_compatible(self, peer_ri: RankInfo) -> bool:
        if not self._attention_policy.check_peer_compatible(peer_ri):
            return False

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

        return True

    @staticmethod
    def _buffer_records(
        pool_view: PoolView,
        layer_group: AttentionLayerGroup,
    ) -> dict[tuple[int, str], tuple[int, int, MapperKind]]:
        """Map ``(global_layer_id, role)`` to ``(offset, size, kind)``."""
        if not pool_view.buffer_roles:
            return {}
        local_to_global = {
            layer.local_layer_id: layer.global_layer_id for layer in layer_group.local_layers
        }
        mapper_kinds = get_buffer_mapper_kinds(pool_view)
        records = {}
        for entry, role, mapper_kind in zip(
            pool_view.buffer_entries,
            pool_view.buffer_roles,
            mapper_kinds,
        ):
            local_layer_id = int(entry["local_layer_id"])
            if local_layer_id not in local_to_global:
                raise ValueError(
                    "PoolView references a layer outside its layer group: "
                    f"local_layer_id={local_layer_id}"
                )
            buffer_key = (local_to_global[local_layer_id], role)
            if buffer_key in records:
                raise ValueError(f"Duplicate PoolView buffer metadata for {buffer_key!r}")
            records[buffer_key] = (
                int(entry["offset"]),
                int(entry["size"]),
                mapper_kind,
            )
        return records

    def get_pool_mapping(self, peer_ri: RankInfo) -> List[PoolPair]:
        """Return physical pool pairs that share logical layer/role buffers.

        New V2 page tables carry per-buffer roles, so a coalesced local pool
        may map to multiple peer pools when topology changes pool grouping.
        Legacy page tables retain the original one-to-one role-set matching.
        """
        key = self._unique_key(peer_ri.instance_name, peer_ri.instance_rank)
        if key in self._lg_pool_mapping_cache:
            return self._lg_pool_mapping_cache[key]

        mapping: List[PoolPair] = []
        self_pt = self._self_ext_cache.page_table
        peer_pt = peer_ri.page_table

        if self_pt is None or peer_pt is None:
            self._lg_pool_mapping_cache[key] = mapping
            return mapping
        if not self_pt.layer_groups or not peer_pt.layer_groups:
            self._lg_pool_mapping_cache[key] = mapping
            return mapping

        peer_layer_to_group = get_layer_to_layer_group(peer_pt)

        for self_lg_idx, self_lg in enumerate(self_pt.layer_groups):
            if not isinstance(self_lg, AttentionLayerGroup):
                continue
            for self_pi, self_pv in enumerate(self_lg.pool_views):
                # The only place mapper_kind affects pool matching:
                #   INDEXED → pool may cover a subset of the LG; read
                #             buffer_entries to find the exact layer set.
                #   FLAT    → pool covers the entire LG by convention;
                #             use the LG's layer ids directly.
                self_is_flat = self_pv.mapper_kind == MapperKind.FLAT
                pv_global_ids = (
                    get_global_layer_ids(self_lg)
                    if self_is_flat
                    else get_pool_view_global_layer_ids(self_pv, self_lg)
                )
                if not pv_global_ids:
                    continue

                # One physical pool may span layers that land in different
                # peer PP layer groups. Preserve their order while visiting
                # every overlapping group.
                peer_lg_indices = list(
                    dict.fromkeys(
                        peer_layer_to_group[g] for g in pv_global_ids if g in peer_layer_to_group
                    )
                )
                if not peer_lg_indices:
                    continue
                self_records = self._buffer_records(self_pv, self_lg)
                self_layer_set = set(pv_global_ids)
                for peer_lg_idx in peer_lg_indices:
                    peer_lg = peer_pt.layer_groups[peer_lg_idx]
                    for peer_pi, peer_pv in enumerate(peer_lg.pool_views):
                        peer_global_ids = (
                            get_global_layer_ids(peer_lg)
                            if peer_pv.mapper_kind == MapperKind.FLAT
                            else get_pool_view_global_layer_ids(peer_pv, peer_lg)
                        )
                        if not set(peer_global_ids) & self_layer_set:
                            continue

                        peer_records = self._buffer_records(peer_pv, peer_lg)
                        if self_records and peer_records:
                            overlapping_keys = self_records.keys() & peer_records.keys()
                            if not overlapping_keys:
                                continue
                            for buffer_key in overlapping_keys:
                                self_kind = self_records[buffer_key][2]
                                peer_kind = peer_records[buffer_key][2]
                                if self_kind != peer_kind:
                                    raise ValueError(
                                        "PeerRegistrar.get_pool_mapping: incompatible mapper "
                                        f"kinds for buffer {buffer_key!r} "
                                        f"(local={self_kind.name}, peer={peer_kind.name})"
                                    )
                            mapping.append(((self_lg_idx, self_pi), (peer_lg_idx, peer_pi)))
                            continue

                        if peer_pv.pool_role != self_pv.pool_role:
                            continue
                        if peer_pv.mapper_kind != self_pv.mapper_kind:
                            raise ValueError(
                                "PeerRegistrar.get_pool_mapping: incompatible mapper "
                                f"kinds for pool role {sorted(self_pv.pool_role)} "
                                f"(local={self_pv.mapper_kind.name}, "
                                f"peer={peer_pv.mapper_kind.name}, peer_pool={peer_pi})"
                            )
                        mapping.append(((self_lg_idx, self_pi), (peer_lg_idx, peer_pi)))
                        break

        self._lg_pool_mapping_cache[key] = mapping
        return mapping

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
        self_phys = get_physical_pool(self_pt, self_lg_idx, self_pv.pool_idx)
        peer_phys = get_physical_pool(peer_pt, peer_lg_idx, peer_pv.pool_idx)

        assert self._ri.attention is not None
        assert isinstance(self_lg, AttentionLayerGroup)
        assert isinstance(peer_lg, AttentionLayerGroup)
        self_records = self._buffer_records(self_pv, self_lg)
        peer_records = self._buffer_records(peer_pv, peer_lg)
        if self_records and peer_records:
            overlapping_keys = [key for key in self_records if key in peer_records]
            mappings = []
            for buffer_key in overlapping_keys:
                self_offset, self_size, self_kind = self_records[buffer_key]
                peer_offset, peer_size, peer_kind = peer_records[buffer_key]
                if self_kind != peer_kind:
                    raise ValueError(
                        "PeerRegistrar.get_kv_map: incompatible mapper kinds "
                        f"for buffer {buffer_key!r} "
                        f"(local={self_kind.name}, peer={peer_kind.name})"
                    )
                mappings.append(
                    PoolBufferMapping(
                        src_offset=self_offset,
                        dst_offset=peer_offset,
                        src_bytes=self_size,
                        dst_bytes=peer_size,
                        mapper_kind=self_kind,
                    )
                )

            full_region_identity = len(overlapping_keys) == len(self_records) == len(
                peer_records
            ) and all(
                self_records[buffer_key] == peer_records[buffer_key]
                for buffer_key in overlapping_keys
            )
            targets = self.get_peer_overlap(peer_ri, peer_ri.dp_rank)
            mapper = PoolBufferMapper(
                mappings=mappings,
                self_ri=self._ri,
                peer_ri=peer_ri,
                self_region_bytes=self_phys.slot_bytes,
                peer_region_bytes=peer_phys.slot_bytes,
                full_region_identity=full_region_identity,
                include_sharded=self.should_send_kv(targets, peer_ri),
                include_replicated=self._owns_tp_fan_in(peer_ri),
            )
            self._kv_map_cache[cache_key] = mapper
            return mapper

        if self_pv.mapper_kind != peer_pv.mapper_kind:
            raise ValueError(
                "PeerRegistrar.get_kv_map: incompatible mapper kinds "
                f"(local={self_pv.mapper_kind.name}, peer={peer_pv.mapper_kind.name})"
            )

        # Order both layer-id lists by physical slot position so that a layer's
        # index in the list *is* its slot offset. The KV transfer maps layers
        # positionally (byte offset = index * per-layer stride) and the mappers
        # copy one contiguous ``[offset, offset + transfer_layers)`` fragment, so
        # the order must reflect the actual physical layout. We derive it from
        # the KV-cache manager's layout rather than assuming ``global_layer_id``
        # is monotonic with the layer's byte offset in the slot:
        #   INDEXED: ``get_pool_view_global_layer_ids`` orders layers by their
        #            ``buffer_entries`` offsets (the V2 pool-view layout).
        #   FLAT:    the pool has no per-buffer layer info; it packs the whole
        #            layer group equal-sized in ``local_layers`` order, so that
        #            order already *is* the physical order.
        if self_pv.mapper_kind == MapperKind.FLAT:
            self_global_ids = get_global_layer_ids(self_lg)
            peer_global_ids = get_global_layer_ids(peer_lg)
            self_num_layers = get_layer_group_num_layers(self_lg)
            peer_num_layers = get_layer_group_num_layers(peer_lg)
        elif self_pv.mapper_kind == MapperKind.INDEXED:
            self_global_ids = get_pool_view_global_layer_ids(self_pv, self_lg)
            peer_global_ids = get_pool_view_global_layer_ids(peer_pv, peer_lg)
            self_num_layers = get_pool_view_num_layers(self_pv)
            peer_num_layers = get_pool_view_num_layers(peer_pv)
        else:
            raise ValueError(
                f"PeerRegistrar.get_kv_map: unexpected mapper kind {self_pv.mapper_kind!r}"
            )

        overlap = set(self_global_ids) & set(peer_global_ids)
        transfer_layers = len(overlap)

        if transfer_layers > 0:
            # Anchor on the overlap layer that comes first in self's physical
            # order and locate the *same* global layer in peer's physical order.
            # Since the mapper copies a single contiguous fragment, the shared
            # layers must occupy an aligned, contiguous run of slots on both
            # peers. Validate that here instead of relying on a global-layer-id
            # ordering convention and silently transferring the wrong bytes.
            first_overlap_layer = next(g for g in self_global_ids if g in overlap)
            self_layer_offset = self_global_ids.index(first_overlap_layer)
            peer_layer_offset = peer_global_ids.index(first_overlap_layer)
            self_run = self_global_ids[self_layer_offset : self_layer_offset + transfer_layers]
            peer_run = peer_global_ids[peer_layer_offset : peer_layer_offset + transfer_layers]
            if set(self_run) != overlap or self_run != peer_run:
                raise ValueError(
                    "PeerRegistrar.get_kv_map: shared layers do not form an "
                    "aligned contiguous run of physical slots on both peers "
                    f"(self={self_global_ids}, peer={peer_global_ids}, "
                    f"overlap={sorted(overlap)}); the KV transfer requires shared "
                    "layers to occupy matching contiguous slot ranges."
                )
        else:
            self_layer_offset = 0
            peer_layer_offset = 0

        self_region_bytes = self_pv.bytes_per_region or self_phys.slot_bytes
        peer_region_bytes = peer_pv.bytes_per_region or peer_phys.slot_bytes
        if self_pv.mapper_kind == MapperKind.NHD:
            self_buffers_per_layer = self._get_buffers_per_layer(
                self_pv,
                self_num_layers,
                layer_group_id=self_lg_idx,
                pool_idx=self_pi,
            )
            peer_buffers_per_layer = self._get_buffers_per_layer(
                peer_pv,
                peer_num_layers,
                layer_group_id=peer_lg_idx,
                pool_idx=peer_pi,
            )
        else:
            self_buffers_per_layer = peer_buffers_per_layer = 1
        mapper = self._attention_policy.build_kv_mapper(
            peer_ri=peer_ri,
            mapper_kind=self_pv.mapper_kind,
            transfer_layers=transfer_layers,
            self_layer_offset=self_layer_offset,
            peer_layer_offset=peer_layer_offset,
            self_pool_num_layers=self_num_layers,
            peer_pool_num_layers=peer_num_layers,
            self_buffers_per_layer=self_buffers_per_layer,
            peer_buffers_per_layer=peer_buffers_per_layer,
            self_pool_slot_bytes=self_region_bytes,
            peer_pool_slot_bytes=peer_region_bytes,
        )

        self._kv_map_cache[cache_key] = mapper
        return mapper

    @staticmethod
    def _get_buffers_per_layer(
        pool_view: PoolView,
        num_layers: int,
        *,
        layer_group_id: int,
        pool_idx: int,
    ) -> int:
        num_entries = get_num_buffer_entries(pool_view)
        if num_entries == 0:
            return 1
        if num_layers <= 0 or num_entries % num_layers != 0:
            raise ValueError(
                "PoolView buffer entries are not evenly distributed across layers: "
                f"layer_group={layer_group_id}, pool={pool_idx}, "
                f"entries={num_entries}, layers={num_layers}"
            )
        return num_entries // num_layers

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
            for cp in range(peer_start_cp, peer_end_cp):
                for tp in range(peer_start_tp, peer_end_tp):
                    ranks.append(pp * peer_ri.tp_size * peer_ri.cp_size + cp * peer_ri.tp_size + tp)

        dup_head, peer_dup_head = self._attention_policy.duplicate_head_factors(peer_ri)

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

    def should_send_kv(self, peer_overlap: PeerOverlap, peer_rank_info: RankInfo) -> bool:
        dup_head_factor = peer_overlap.duplicate_head_factor
        if dup_head_factor <= 1:
            return True
        self_tp_rank_in_dp_group = self._ri.tp_rank % self._ri.tp_size_per_dp_group
        return (peer_rank_info.dp_rank % dup_head_factor) == (
            self_tp_rank_in_dp_group % dup_head_factor
        )

    def _owns_tp_fan_in(self, peer_rank_info: RankInfo) -> bool:
        """Elect one owner when replicated bytes fan in across TP ranks.

        A peer with fewer TP shards receives identical replicated data from
        several local ranks. Elect the first local rank in each peer-sized
        fan-in group so exactly one copy reaches each destination.
        """
        ratio = max(
            1,
            self._ri.tp_size_per_dp_group // peer_rank_info.tp_size_per_dp_group,
        )
        self_tp_rank = self._ri.tp_rank % self._ri.tp_size_per_dp_group
        return self_tp_rank % ratio == 0

    def should_send_pool(
        self,
        peer_overlap: PeerOverlap,
        peer_rank_info: RankInfo,
        layer_group_id: int,
        pool_idx: int,
        peer_layer_group_id: int,
        peer_pool_idx: int,
    ) -> bool:
        """Return whether this rank owns any logical buffer in a pool pair.

        Normal KV buffers retain head-duplication routing. Replicated side
        caches use one sender per fan-in group. A mixed physical pool is sent
        when this rank owns at least one of its overlapping logical buffers;
        PoolBufferMapper filters the remaining entries.
        """
        layer_group = self._self_ext_cache.page_table.layer_groups[layer_group_id]
        pool_view = layer_group.pool_views[pool_idx]
        peer_layer_group = peer_rank_info.page_table.layer_groups[peer_layer_group_id]
        peer_pool_view = peer_layer_group.pool_views[peer_pool_idx]
        self_records = self._buffer_records(pool_view, layer_group)
        peer_records = self._buffer_records(peer_pool_view, peer_layer_group)
        if self_records and peer_records:
            kinds = {
                self_records[buffer_key][2]
                for buffer_key in self_records.keys() & peer_records.keys()
            }
        else:
            kinds = {pool_view.mapper_kind}

        owns_sharded = any(kind != MapperKind.REPLICATED for kind in kinds) and self.should_send_kv(
            peer_overlap, peer_rank_info
        )
        owns_replicated = MapperKind.REPLICATED in kinds and self._owns_tp_fan_in(peer_rank_info)
        return owns_sharded or owns_replicated

    def should_send_aux(self, peer_rank_info: RankInfo) -> bool:
        # to ensure the transfer aux is not duplicated

        # TP: only the first rank in each peer-TP-sized group sends aux
        should_send_in_tp = self._owns_tp_fan_in(peer_rank_info)

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
