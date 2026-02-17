from dataclasses import dataclass, field
from typing import Dict, List

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo
from tensorrt_llm._torch.disaggregation.native.region.block import (
    HeadMatchMapper,
    HeadMismatchMapper,
    IdentityMapper,
    RegionMapperBase,
)
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import (
    KVPoolAttrs,
    KVRegionExtractorV1,
)


@dataclass
class PeerOverlap:
    overlap_pp_size: int = 0
    overlap_tp_size: int = 0
    overlap_cp_size: int = 0
    duplicate_head_factor: int = 1
    peer_duplicate_head_factor: int = 1
    target_peer_pp_layer_num: List[int] = field(default_factory=list)
    ranks: List[int] = field(default_factory=list)


class PeerRegistrar:
    def __init__(self, self_rank_info: RankInfo, self_extractor: KVRegionExtractorV1):
        self._ri = self_rank_info
        self._peer_ri_cache: Dict[str, RankInfo] = {}
        self._kv_map_cache: Dict[str, RegionMapperBase] = {}
        self._self_ext_cache = self_extractor
        self._peer_ext_cache: Dict[str, KVRegionExtractorV1] = {}
        self._overlap_cache: Dict[str, PeerOverlap] = {}

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
        layer_num = peer_ri.layer_num_per_pp[peer_ri.pp_rank]
        block_size = self._block_size(layer_num, peer_ri)
        extractor = KVRegionExtractorV1(
            KVPoolAttrs(pool_ptrs=peer_ri.kv_ptrs, block_bytes=[block_size])
        )
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
        return True

    def _tp_per_dp(self, info: RankInfo) -> int:
        return (
            info.tp_size // info.dp_size
            if getattr(info, "enable_attention_dp", False)
            else info.tp_size
        )

    def get_kv_map(self, peer_ri: RankInfo):
        key = self._unique_key(peer_ri.instance_name, peer_ri.instance_rank)
        if key in self._kv_map_cache:
            return self._kv_map_cache[key]

        self_tp_per_dp = self._tp_per_dp(self._ri)
        peer_tp_per_dp = self._tp_per_dp(peer_ri)

        is_dup_head = (
            self._ri.kv_heads_per_rank * self_tp_per_dp
            != peer_ri.kv_heads_per_rank * peer_tp_per_dp
        )
        head_match = is_dup_head or self._ri.is_mla or self_tp_per_dp == peer_tp_per_dp
        logger.debug(
            "KVMapperFactory.get_kv_map: "
            f"head_match={head_match}, is_dup_head={is_dup_head}, self_is_mla={self._ri.is_mla}, "
            f"self_tp_per_dp={self_tp_per_dp}, peer_tp_per_dp={peer_tp_per_dp}"
        )
        # fast identity when write_all and same pp_size
        if head_match and self._ri.pp_size == peer_ri.pp_size:
            mapper = IdentityMapper()
            self._kv_map_cache[key] = mapper
            return mapper

        # compute overlapping layers
        self_start_layer = sum(self._ri.layer_num_per_pp[: self._ri.pp_rank])
        self_end_layer = self_start_layer + self._ri.layer_num_per_pp[self._ri.pp_rank]
        peer_start_layer = sum(peer_ri.layer_num_per_pp[: peer_ri.pp_rank])
        peer_end_layer = peer_start_layer + peer_ri.layer_num_per_pp[peer_ri.pp_rank]
        start = max(self_start_layer, peer_start_layer)
        end = min(self_end_layer, peer_end_layer)
        transfer_layers = end - start
        self_layer_offset = start - self_start_layer
        peer_layer_offset = start - peer_start_layer

        if head_match:
            mapper = HeadMatchMapper(
                transfer_layers=transfer_layers,
                src_layer_off=self_layer_offset,  # local layer offset
                dst_layer_off=peer_layer_offset,  # peer layer offset
                self_ri=self._ri,
                peer_ri=peer_ri,
            )
            self._kv_map_cache[key] = mapper
            return mapper

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
        tgt_pp_layer_num: List[int] = []
        for p in range(peer_ri.pp_size):
            peer_start_layer = pre
            peer_end_layer = peer_start_layer + peer_ri.layer_num_per_pp[p]
            if self_start_layer < peer_end_layer and self_end_layer > peer_start_layer:
                tgt_pp_ranks.append(p)
                tgt_pp_layer_num.append(
                    min(peer_end_layer, self_end_layer) - max(peer_start_layer, self_start_layer)
                )
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
            target_peer_pp_layer_num=tgt_pp_layer_num,
            ranks=ranks,
        )
        self._overlap_cache[key] = targets
        return targets
