from dataclasses import dataclass, field
from typing import Dict, List

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.native.region.block import (
    ConvStateMismatchMapper,
    HeadMatchMapper,
    HeadMismatchMapper,
    IdentityMapper,
    IndexerKCacheHeadMatchMapper,
    MambaHeadMatchMapper,
    MambaHeadMismatchMapper,
    RegionMapperBase,
)
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1, PoolRole


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
        self._peer_ri_cache: Dict[str, RankInfo] = {}
        self._kv_map_cache: Dict[
            tuple, RegionMapperBase
        ] = {}  # key: (peer_key, self_group_id, peer_group_id)
        self._self_ext_cache = self_extractor
        self._peer_ext_cache: Dict[str, KVRegionExtractorV1] = {}
        self._overlap_cache: Dict[str, PeerOverlap] = {}
        self._group_id_mapping_cache: Dict[
            str, Dict[int, int]
        ] = {}  # peer_key -> {self_group_id -> peer_group_id}

    def _block_size(self, layer_num: int, ri: RankInfo) -> int:
        return (
            layer_num
            * ri.kv_factor
            * ri.kv_heads_per_rank
            * ri.tokens_per_block
            * ri.dims_per_head
            * ri.element_bytes
        )

    def register(self, peer_name: str, peer_rank: int, peer_ri: RankInfo):
        # TODO: check if peer is valid for registration
        assert self._self_ext_cache is not None
        if not self._check_peer_compatible(peer_ri):
            raise ValueError(
                f"PeerRegistrar.register: peer {peer_name} (rank={peer_rank}) is incompatible with local rank."
            )
        key = self._unique_key(peer_name, peer_rank)
        self._peer_ri_cache[key] = peer_ri
        peer_ri = self.get_peer_rank_info(peer_name, peer_rank)
        # Use kv_pool_attrs from RankInfo (already serialized V2 format)
        extractor = KVRegionExtractorV1(peer_ri.kv_pool_attrs)
        self._peer_ext_cache[key] = extractor

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
        if key in self._kv_map_cache:
            del self._kv_map_cache[key]

    def get_peer_rank_info(self, peer_name: str, peer_rank: int):
        return self._peer_ri_cache[self._unique_key(peer_name, peer_rank)]

    @property
    def self_rank_info(self) -> RankInfo:
        return self._ri

    def _unique_key(self, name: str, rank: int) -> str:
        return name + str(rank)

    def _check_peer_compatible(self, peer_ri: RankInfo) -> bool:
        if self._ri.is_mla != peer_ri.is_mla:
            logger.warning(
                "PeerRegistrar: compatibility check failed: 'is_mla' differs "
                f"(local={self._ri.is_mla}, peer={peer_ri.is_mla})."
            )
            return False
        if self._ri.cp_size != 1 or peer_ri.cp_size != 1:
            logger.warning(
                "PeerRegistrar: unsupported configuration: context parallelism (cp_size) "
                f"must be 1 for both local and peer ranks (local={self._ri.cp_size}, peer={peer_ri.cp_size})."
            )
            return False
        if self._ri.element_bytes != peer_ri.element_bytes:
            logger.warning(
                "PeerRegistrar: element size mismatch "
                f"(local={self._ri.element_bytes} bytes, peer={peer_ri.element_bytes} bytes)."
            )
            return False
        if self._ri.tokens_per_block != peer_ri.tokens_per_block:
            logger.warning(
                "PeerRegistrar: tokens_per_block mismatch "
                f"(local={self._ri.tokens_per_block}, peer={peer_ri.tokens_per_block})."
            )
            return False
        if self._ri.dims_per_head != peer_ri.dims_per_head:
            logger.warning(
                "PeerRegistrar: dims_per_head mismatch "
                f"(local={self._ri.dims_per_head}, peer={peer_ri.dims_per_head})."
            )
            return False

        self_layers = sum(self._ri.layer_num_per_pp)
        peer_layers = sum(peer_ri.layer_num_per_pp)
        if self_layers != peer_layers:
            logger.warning(
                "PeerRegistrar: total layer count mismatch "
                f"(local={self_layers}, peer={peer_layers})."
            )
            return False

        if self._ri.is_mla:
            if peer_ri.kv_heads_per_rank != 1 or self._ri.kv_heads_per_rank != 1:
                logger.warning(
                    "PeerRegistrar: MLA mode requires exactly 1 KV head per rank for both local and peer."
                    f" (local={self._ri.kv_heads_per_rank}, peer={peer_ri.kv_heads_per_rank})"
                )
                return False

        # Check Mamba compatibility
        if not self._check_mamba_compatible(peer_ri):
            return False

        return True

    def _check_mamba_compatible(self, peer_ri: RankInfo) -> bool:
        """Check Mamba-specific compatibility between self and peer."""
        self_attrs = self._self_ext_cache.kv_pool_attrs if self._self_ext_cache else None
        peer_attrs = peer_ri.kv_pool_attrs

        if self_attrs is None or peer_attrs is None:
            return True  # No attrs to check

        # Find Mamba layer groups
        def find_mamba_group(attrs):
            for group in attrs.layer_group_attrs_list:
                if (
                    PoolRole.SSM_STATE in group.roles_to_pool_idx
                    or PoolRole.CONV_STATE in group.roles_to_pool_idx
                ):
                    return group
            return None

        self_mamba_group = find_mamba_group(self_attrs)
        peer_mamba_group = find_mamba_group(peer_attrs)

        # If one has Mamba and the other doesn't, they're incompatible
        if (self_mamba_group is None) != (peer_mamba_group is None):
            logger.warning(
                "PeerRegistrar: Mamba compatibility check failed: "
                f"local has_mamba={self_mamba_group is not None}, "
                f"peer has_mamba={peer_mamba_group is not None}"
            )
            return False

        if self_mamba_group is None:
            return True  # Neither has Mamba, compatible

        # Both have Mamba, check that global layer counts match
        self_mamba_layers = set(self_mamba_group.global_layer_ids)
        peer_mamba_layers = set(peer_mamba_group.global_layer_ids)
        all_mamba_layers = self_mamba_layers | peer_mamba_layers

        if len(all_mamba_layers) != len(self_mamba_layers) or len(all_mamba_layers) != len(
            peer_mamba_layers
        ):
            # Different total mamba layer counts - check if there's any overlap
            overlapping = self_mamba_layers & peer_mamba_layers
            if not overlapping:
                logger.warning(
                    "PeerRegistrar: Mamba layer compatibility check failed: "
                    f"no overlapping mamba layers between local and peer. "
                    f"local={sorted(self_mamba_layers)}, peer={sorted(peer_mamba_layers)}"
                )

        return True

    def _tp_per_dp(self, info: RankInfo) -> int:
        return (
            info.tp_size // info.dp_size
            if getattr(info, "enable_attention_dp", False)
            else info.tp_size
        )

    def get_group_id_mapping(self, peer_ri: RankInfo) -> Dict[int, int]:
        """
        Get mapping from self_group_id to peer_group_id based on global_layer_id matching.

        Returns:
            Dict mapping self_group_id -> peer_group_id for overlapping layer groups.
        """
        key = self._unique_key(peer_ri.instance_name, peer_ri.instance_rank)
        if key in self._group_id_mapping_cache:
            return self._group_id_mapping_cache[key]

        mapping: Dict[int, int] = {}
        self_attrs = self._self_ext_cache.kv_pool_attrs
        peer_attrs = peer_ri.kv_pool_attrs

        for self_group in self_attrs.layer_group_attrs_list:
            for self_global_layer_id in self_group.global_layer_ids:
                if self_global_layer_id in peer_attrs.layer_to_group_id:
                    peer_group_id = peer_attrs.layer_to_group_id[self_global_layer_id]
                    mapping[self_group.group_id] = peer_group_id
                    break  # Found one match, that determines the group mapping

        self._group_id_mapping_cache[key] = mapping
        return mapping

    def get_kv_map(
        self,
        peer_ri: RankInfo,
        self_layer_group_id: int = 0,
        peer_layer_group_id: int = 0,
        pool_role: PoolRole = PoolRole.KV_CACHE,
    ) -> RegionMapperBase:
        """
        Get mapper for a specific layer group pair.

        Args:
            peer_ri: Peer's RankInfo
            self_layer_group_id: Local layer group ID
            peer_layer_group_id: Peer's layer group ID
            pool_role: Pool role (KV_CACHE, INDEXER, SSM_STATE, CONV_STATE)

        Returns:
            RegionMapperBase for the specified layer group pair.
        """
        peer_key = self._unique_key(peer_ri.instance_name, peer_ri.instance_rank)
        key = (peer_key, self_layer_group_id, peer_layer_group_id, pool_role)
        if key in self._kv_map_cache:
            return self._kv_map_cache[key]

        self_tp_per_dp = self._tp_per_dp(self._ri)
        peer_tp_per_dp = self._tp_per_dp(peer_ri)

        # Get layer group attributes
        self_attrs = self._self_ext_cache.kv_pool_attrs
        peer_attrs = peer_ri.kv_pool_attrs
        self_group_attrs = self_attrs.layer_group_attrs_list[self_layer_group_id]
        peer_group_attrs = peer_attrs.layer_group_attrs_list[peer_layer_group_id]

        # Find overlapping global_layer_ids between the two groups
        self_layers = set(self_group_attrs.global_layer_ids)
        peer_layers = set(peer_group_attrs.global_layer_ids)
        overlapping_layers = sorted(self_layers & peer_layers)

        transfer_layers = len(overlapping_layers)

        # Compute layer offsets within each group
        # self_layer_offset: index of first overlapping layer in self's group
        # peer_layer_offset: index of first overlapping layer in peer's group
        if transfer_layers > 0:
            first_overlap_layer = overlapping_layers[0]
            self_layer_offset = self_group_attrs.global_layer_ids.index(first_overlap_layer)
            peer_layer_offset = peer_group_attrs.global_layer_ids.index(first_overlap_layer)
        else:
            self_layer_offset = 0
            peer_layer_offset = 0

        # Handle Mamba states (SSM_STATE, CONV_STATE)
        if pool_role in (PoolRole.SSM_STATE, PoolRole.CONV_STATE):
            mapper = self._get_mamba_mapper(
                pool_role=pool_role,
                transfer_layers=transfer_layers,
                self_layer_offset=self_layer_offset,
                peer_layer_offset=peer_layer_offset,
                self_group_attrs=self_group_attrs,
                peer_group_attrs=peer_group_attrs,
                self_tp_per_dp=self_tp_per_dp,
                peer_tp_per_dp=peer_tp_per_dp,
                peer_tp_rank=peer_ri.tp_rank,
            )
            self._kv_map_cache[key] = mapper
            return mapper

        # KV cache handling
        is_dup_head = (
            self._ri.kv_heads_per_rank * self_tp_per_dp
            != peer_ri.kv_heads_per_rank * peer_tp_per_dp
        )
        head_match = is_dup_head or self._ri.is_mla or self_tp_per_dp == peer_tp_per_dp

        logger.debug(
            "PeerRegistrar.get_kv_map: "
            "head_match=%s, is_dup_head=%s, self_is_mla=%s, "
            "self_tp_per_dp=%s, peer_tp_per_dp=%s, "
            "self_group_id=%s, peer_group_id=%s, transfer_layers=%s",
            head_match,
            is_dup_head,
            self._ri.is_mla,
            self_tp_per_dp,
            peer_tp_per_dp,
            self_layer_group_id,
            peer_layer_group_id,
            transfer_layers,
        )

        # fast identity when write_all and same structure
        self_group_layer_count = len(self_group_attrs.global_layer_ids)
        peer_group_layer_count = len(peer_group_attrs.global_layer_ids)
        if head_match and transfer_layers == self_group_layer_count == peer_group_layer_count:
            mapper = IdentityMapper()
            self._kv_map_cache[key] = mapper
            return mapper

        if head_match:
            if pool_role == PoolRole.INDEXER:
                pool_idx = self_group_attrs.roles_to_pool_idx[pool_role]
                block_size_per_layer = self_group_attrs.block_bytes_per_pool[pool_idx] / (
                    len(self_group_attrs.global_layer_ids)
                )
                mapper = IndexerKCacheHeadMatchMapper(
                    transfer_layers=transfer_layers,
                    src_layer_off=self_layer_offset,
                    dst_layer_off=peer_layer_offset,
                    self_ri=self._ri,
                    peer_ri=peer_ri,
                    block_size_per_layer=block_size_per_layer,
                )
            else:
                mapper = HeadMatchMapper(
                    transfer_layers=transfer_layers,
                    src_layer_off=self_layer_offset,
                    dst_layer_off=peer_layer_offset,
                    self_ri=self._ri,
                    peer_ri=peer_ri,
                )
            self._kv_map_cache[key] = mapper
            return mapper

        if pool_role == PoolRole.INDEXER:
            raise ValueError("IndexerKCacheHeadMatchMapper is not supported for head mismatch case")

        # head mismatch case
        mapper = HeadMismatchMapper(
            transfer_layers=transfer_layers,
            src_layer_off=self_layer_offset,
            peer_layer_off=peer_layer_offset,
            self_ri=self._ri,
            peer_ri=peer_ri,
        )
        self._kv_map_cache[key] = mapper
        return mapper

    def _get_mamba_mapper(
        self,
        pool_role: PoolRole,
        transfer_layers: int,
        self_layer_offset: int,
        peer_layer_offset: int,
        self_group_attrs,
        peer_group_attrs,
        self_tp_per_dp: int,
        peer_tp_per_dp: int,
        peer_tp_rank: int,
    ) -> RegionMapperBase:
        """
        Get mapper for Mamba states (SSM_STATE or CONV_STATE).

        For Mamba, headNum and tp_size are inversely proportional.
        For CONV_STATE with sectioned layout info (MambaLayerGroupAttrs),
        uses ConvStateMismatchMapper to handle each section independently.
        """
        pool_idx = self_group_attrs.roles_to_pool_idx[pool_role]
        block_bytes_per_layer = self_group_attrs.block_bytes_per_pool[pool_idx]

        # Get head counts from layer group attrs
        self_nheads = self_group_attrs.kv_head_num_per_rank
        peer_nheads = peer_group_attrs.kv_head_num_per_rank

        # Check if head counts match (considering TP relationship)
        # headNum * tp_size should be constant, so if tp_per_dp differs, heads differ
        head_match = self_tp_per_dp == peer_tp_per_dp

        self_group_layer_count = len(self_group_attrs.global_layer_ids)
        peer_group_layer_count = len(peer_group_attrs.global_layer_ids)

        # Fast identity when all layers match and head counts match
        if head_match and transfer_layers == self_group_layer_count == peer_group_layer_count:
            return IdentityMapper()

        if head_match:
            return MambaHeadMatchMapper(
                transfer_layers=transfer_layers,
                src_layer_off=self_layer_offset,
                dst_layer_off=peer_layer_offset,
                block_bytes_per_layer=block_bytes_per_layer,
            )

        # Head mismatch case: for CONV_STATE, prefer ConvStateMismatchMapper
        # if section information is available (from MambaLayerGroupAttrs)
        if pool_role == PoolRole.CONV_STATE:
            self_sec = getattr(self_group_attrs, "conv_section_bytes_per_rank", None)
            peer_sec = getattr(peer_group_attrs, "conv_section_bytes_per_rank", None)
            if self_sec and peer_sec:
                return ConvStateMismatchMapper(
                    transfer_layers=transfer_layers,
                    src_layer_off=self_layer_offset,
                    dst_layer_off=peer_layer_offset,
                    self_section_bytes=self_sec,
                    peer_section_bytes=peer_sec,
                    self_tp_per_dp=self_tp_per_dp,
                    peer_tp_per_dp=peer_tp_per_dp,
                    self_tp_rank=self._ri.tp_rank,
                    peer_tp_rank=peer_tp_rank,
                )

        # SSM_STATE or CONV_STATE without section info: use head-based mapper
        # bytes_per_head = block_bytes_per_layer / nheads
        bytes_per_head = block_bytes_per_layer // self_nheads

        return MambaHeadMismatchMapper(
            transfer_layers=transfer_layers,
            src_layer_off=self_layer_offset,
            dst_layer_off=peer_layer_offset,
            bytes_per_head=bytes_per_head,
            self_nheads=self_nheads,
            peer_nheads=peer_nheads,
            self_tp_per_dp=self_tp_per_dp,
            peer_tp_per_dp=peer_tp_per_dp,
            self_tp_rank=self._ri.tp_rank,
            peer_tp_rank=peer_tp_rank,
        )

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
            # no overlap found
            targets = PeerOverlap()
            self._overlap_cache[key] = targets
            return targets

        peer_start_pp = tgt_pp_ranks[0]
        overlap_pp_size = len(tgt_pp_ranks)
        peer_end_pp = peer_start_pp + overlap_pp_size

        # tp per dp-group
        self_tp_per_dp = self._tp_per_dp(self._ri)
        peer_tp_per_dp = self._tp_per_dp(peer_ri)
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

        factor_self = self._ri.kv_heads_per_rank * self_tp_per_dp
        factor_peer = peer_ri.kv_heads_per_rank * peer_tp_per_dp
        dup_head = max(1, factor_self // factor_peer)
        peer_dup_head = max(1, factor_peer // factor_self)

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
