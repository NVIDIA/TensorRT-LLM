from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, List, Optional, Union

import msgpack
import numpy as np

from tensorrt_llm import logger
from tensorrt_llm._torch.disaggregation.native.aux_buffer import AuxBufferMeta
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import get_size_in_bytes, nvtx_range


@dataclass
class PeerOverlapTargets:
    overlap_pp_size: int = 0
    overlap_tp_size: int = 0
    overlap_cp_size: int = 0
    duplicate_head_factor: int = 1
    peer_duplicate_head_factor: int = 1
    target_peer_pp_layer_num: List[int] = field(default_factory=list)
    ranks: List[int] = field(default_factory=list)


@dataclass
class RankInfo:
    instance_name: str
    instance_rank: int
    tp_size: int
    tp_rank: int
    pp_size: int
    pp_rank: int
    dp_size: int
    dp_rank: int
    cp_size: int
    cp_rank: int
    device_id: int
    kv_heads_per_rank: int
    # [numLayers, kv_factor, heads, tokens, dims_per_head]
    tokens_per_block: int
    dims_per_head: int
    element_bytes: int
    enable_attention_dp: bool
    is_mla: bool
    layer_num_per_pp: List[int]
    kv_ptrs: List[int]
    aux_ptrs: List[int]
    server_endpoint: str
    self_endpoint: str
    transfer_engine_info: bytes
    aux_meta: Optional[AuxBufferMeta]

    def to_bytes(self) -> bytes:
        data = asdict(self)
        data["aux_meta"] = self.aux_meta.to_dict() if self.aux_meta is not None else None
        return msgpack.packb(data)

    @classmethod
    def from_bytes(cls, data: bytes) -> "RankInfo":
        unpacked = msgpack.unpackb(data)
        if unpacked.get("aux_meta") is not None:
            unpacked["aux_meta"] = AuxBufferMeta.from_dict(unpacked["aux_meta"])
        return cls(**unpacked)


@dataclass
class InstanceInfo:
    instance_name: str
    tp_size: int
    pp_size: int
    dp_size: int
    cp_size: int
    kv_heads_per_rank: int
    tokens_per_block: int
    dims_per_head: int
    element_bytes: int
    enable_attention_dp: bool
    is_mla: bool
    layer_num_per_pp: List[int]
    sender_endpoints: List[str]

    def to_bytes(self) -> bytes:
        return msgpack.packb(asdict(self))

    @classmethod
    def from_bytes(cls, data: bytes) -> "InstanceInfo":
        return cls(**msgpack.unpackb(data))


@dataclass
class KVPoolAttrs:
    pool_ptrs: List[int]
    block_bytes: List[int]


class KVPtrExtractor:
    def __init__(self, kv_pool_attrs: Union[KVPoolAttrs, KVCacheManager]):
        if self._is_kv_cache_manager(kv_pool_attrs):
            self._kv_pool_attrs = self._attrs_from_manager(kv_pool_attrs)
        else:
            self._kv_pool_attrs = kv_pool_attrs

    def block_ptrs(self, kv_block_ids: List[int], pool_idx: int = 0) -> List[int]:
        ptrs = self._kv_pool_attrs.pool_ptrs
        sizes = self._kv_pool_attrs.block_bytes

        if pool_idx < 0 or pool_idx >= len(ptrs):
            raise IndexError(
                f"KVPtrExtractor.block_ptrs: pool_idx {pool_idx} out of range "
                f"(available pools = {len(ptrs)})"
            )

        base_ptr = int(ptrs[pool_idx])
        block_size = int(sizes[pool_idx])
        return [base_ptr + block_size * int(bid) for bid in kv_block_ids]

    @property
    def kv_pool_attrs(self) -> KVPoolAttrs:
        return self._kv_pool_attrs

    # ---------------- internal helpers ----------------
    @staticmethod
    def _is_kv_cache_manager(obj) -> bool:
        # A light-weight duck-typing check to decide whether obj behaves like KVCacheManager.
        # We expect KVCacheManager to provide get_unique_primary_pool() and dtype attribute.
        return hasattr(obj, "get_unique_primary_pool") and hasattr(obj, "dtype")

    @staticmethod
    def _attrs_from_manager(manager: KVCacheManager) -> KVPoolAttrs:
        """
        Convert a KVCacheManager into KVPoolAttrs (ptr list and block sizes).
        """
        try:
            pools = manager.get_unique_primary_pool()
        except Exception as ex:
            raise ValueError(
                "KVPtrExtractor: failed to get pool(s) from KVCacheManager via "
                "get_unique_primary_pool(): " + str(ex)
            )

        # Normalize to list for uniform handling
        if isinstance(pools, (list, tuple)):
            pool_list = list(pools)
        else:
            pool_list = [pools]

        element_bytes = get_size_in_bytes(1, manager.dtype)

        ptrs: List[int] = []
        block_sizes: List[int] = []
        for p in pool_list:
            # pointer extraction
            if hasattr(p, "data_ptr") and callable(getattr(p, "data_ptr")):
                try:
                    ptr = int(p.data_ptr())
                except Exception as ex:
                    raise ValueError(
                        f"KVPtrExtractor: failed to call data_ptr() on pool object: {ex}"
                    )
            elif isinstance(p, int):
                ptr = int(p)
            else:
                raise ValueError(
                    "KVPtrExtractor: pool object does not expose data_ptr() and is not an int: "
                    + repr(p)
                )
            ptrs.append(ptr)

            # compute block size: try to infer using first sub-tensor or tensor properties
            try:
                # if p is indexable and has a first element representing a block
                if (
                    hasattr(p, "__getitem__")
                    and hasattr(p[0], "numel")
                    and callable(getattr(p[0], "numel"))
                ):
                    n = int(p[0].numel())
                # fall back to tensor.numel()
                elif hasattr(p, "numel") and callable(getattr(p, "numel")):
                    n = int(p.numel())
                # or a numpy-like field
                else:
                    raise RuntimeError("cannot determine number of elements for pool object")
            except Exception as ex:
                raise ValueError(
                    "KVPtrExtractor: failed to determine block size from pool object: "
                    + repr(p)
                    + " -> "
                    + str(ex)
                )

            block_sizes.append(int(n) * int(element_bytes))

        return KVPoolAttrs(pool_ptrs=ptrs, block_bytes=block_sizes)


@dataclass
class CopyArgs:
    src_ptrs: List[int]
    dst_ptrs: List[int]
    size: int


class PeerRegistrar:
    def __init__(
        self,
        rank_info: RankInfo,
        instance_info: InstanceInfo,
    ):
        self._ri = rank_info
        self._ii = instance_info

        self._peer_ri_cache: Dict[str, RankInfo] = {}

    # ---------------- public simple APIs ----------------
    def register(self, peer_name: str, peer_rank: int, peer_ri: RankInfo):
        # TODO: check if peer is valid for registration
        if not self._check_peer_compatible(peer_ri):
            raise ValueError(
                f"PeerRegistrar.register: peer {peer_name} (rank={peer_rank}) is incompatible with local rank."
            )
        self._peer_ri_cache[self._unique_key(peer_name, peer_rank)] = peer_ri

    def unregister(self, peer_name: str, peer_rank: int):
        key = self._unique_key(peer_name, peer_rank)
        if key in self._peer_ri_cache:
            del self._peer_ri_cache[key]

    def get_peer_rank_info(self, peer_name: str, peer_rank: int):
        return self._peer_ri_cache[self._unique_key(peer_name, peer_rank)]

    @property
    def instance_info(self) -> InstanceInfo:
        return self._ii

    @property
    def rank_info(self) -> RankInfo:
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


class KVMapperFactoryBase(ABC):
    PtrMapper = Callable[[List[int], int, List[int], int], CopyArgs]

    def __init__(
        self,
        registrar: PeerRegistrar,
        kv_cache_manager: KVCacheManager,
    ):
        self._registrar = registrar
        self._kv_cache_manager = kv_cache_manager

    @abstractmethod
    def get_peer_overlap_targets(
        self, peer_instance_info: InstanceInfo, peer_dp_rank: int
    ) -> PeerOverlapTargets: ...

    @abstractmethod
    def get_kv_map(self, peer_ri: RankInfo) -> PtrMapper: ...


class KVMapperFactory(KVMapperFactoryBase):
    def __init__(
        self,
        registrar: PeerRegistrar,
        kv_cache_manager: KVCacheManager,
    ):
        super().__init__(registrar, kv_cache_manager)
        self._overlap_cache: Dict[str, PeerOverlapTargets] = {}
        self._kv_map_cache: Dict[str, callable] = {}
        self._peer_ext_cache: Dict[str, KVPtrExtractor] = {}
        self._self_ext = KVPtrExtractor(self._kv_cache_manager)

        # cache self info
        self._ri = self._registrar.rank_info
        self._ii = self._registrar.instance_info

    @property
    def self_extractor(self) -> KVPtrExtractor:
        return self._self_ext

    # ---------------- kv pool extractor ----------------
    def peer_extractor(self, peer_name: str, peer_rank: int) -> KVPtrExtractor:
        key = self._unique_key(peer_name, peer_rank)
        if key not in self._peer_ext_cache:
            peer_ri = self._registrar.get_peer_rank_info(peer_name, peer_rank)
            kv_factor = 1 if peer_ri.is_mla else 2
            layer_num = peer_ri.layer_num_per_pp[peer_ri.pp_rank]
            block_size = self._block_size(layer_num, kv_factor, peer_ri)
            extractor = KVPtrExtractor(
                kv_pool_attrs=KVPoolAttrs(pool_ptrs=peer_ri.kv_ptrs, block_bytes=[block_size])
            )
            self._peer_ext_cache[key] = extractor
        return self._peer_ext_cache[key]

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

    # ---------------- peer overlap targets ----------------
    def get_peer_overlap_targets(
        self, peer_instance_info: InstanceInfo, peer_dp_rank: int
    ) -> PeerOverlapTargets:
        peer_ii = peer_instance_info
        key = self._unique_key(peer_ii.instance_name, peer_dp_rank)
        if key in self._overlap_cache:
            return self._overlap_cache[key]

        # compute pp overlap and target layers
        self_start_layer = sum(self._ri.layer_num_per_pp[: self._ri.pp_rank])
        self_end_layer = self_start_layer + self._ri.layer_num_per_pp[self._ri.pp_rank]

        pre = 0
        tgt_pp_ranks: List[int] = []
        tgt_pp_layer_num: List[int] = []
        for p in range(peer_ii.pp_size):
            peer_start_layer = pre
            peer_end_layer = peer_start_layer + peer_ii.layer_num_per_pp[p]
            if self_start_layer < peer_end_layer and self_end_layer > peer_start_layer:
                tgt_pp_ranks.append(p)
                tgt_pp_layer_num.append(
                    min(peer_end_layer, self_end_layer) - max(peer_start_layer, self_start_layer)
                )
            pre += peer_ii.layer_num_per_pp[p]

        if tgt_pp_ranks == []:
            # no overlap found
            targets = PeerOverlapTargets()
            self._overlap_cache[key] = targets
            return targets

        peer_start_pp = tgt_pp_ranks[0]
        overlap_pp_size = len(tgt_pp_ranks)
        peer_end_pp = peer_start_pp + overlap_pp_size

        # tp per dp-group
        self_tp_per_dp = self._tp_per_dp(self._ri)
        peer_tp_per_dp = self._tp_per_dp(peer_ii)
        self_tp_rank_in_dp = self._ri.tp_rank % self_tp_per_dp

        overlap_tp_size, peer_start_tp, peer_end_tp = self._find_overlap(
            self_tp_per_dp, peer_tp_per_dp, self_tp_rank_in_dp, peer_dp_rank
        )
        overlap_cp_size, peer_start_cp, peer_end_cp = self._find_overlap(
            self._ri.cp_size, peer_ii.cp_size, self._ri.cp_rank
        )

        ranks: List[int] = []
        for pp in range(peer_start_pp, peer_end_pp):
            for cp in range(peer_start_cp, peer_end_cp):
                for tp in range(peer_start_tp, peer_end_tp):
                    ranks.append(pp * peer_ii.tp_size * peer_ii.cp_size + cp * peer_ii.tp_size + tp)

        factor_self = self._ri.kv_heads_per_rank * self_tp_per_dp
        factor_peer = peer_ii.kv_heads_per_rank * peer_tp_per_dp
        dup_head = max(1, factor_self // factor_peer)
        peer_dup_head = max(1, factor_peer // factor_self)

        targets = PeerOverlapTargets(
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

    @property
    def peer_registrar(self) -> PeerRegistrar:
        return self._registrar

    # ---------------- kv block ptrs mapper ----------------
    def get_kv_map(self, peer_ri: RankInfo) -> KVMapperFactoryBase.PtrMapper:
        key = self._unique_key(peer_ri.instance_name, peer_ri.instance_rank)
        if key in self._kv_map_cache:
            return self._kv_map_cache[key]

        kv_factor = 1 if self._ri.is_mla else 2
        self_tp_per_dp = self._tp_per_dp(self._ri)
        peer_tp_per_dp = self._tp_per_dp(peer_ri)
        self_tp_rank = self._ri.tp_rank % self_tp_per_dp
        peer_tp_rank = peer_ri.tp_rank % peer_tp_per_dp

        # head_num_per_rank = 1 when is_dup_head
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
            mapper = self._kv_map_identity()
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
            mapper = self._kv_map_head_match(
                transfer_layers, kv_factor, self_layer_offset, peer_layer_offset, peer_ri
            )
            self._kv_map_cache[key] = mapper
            return mapper

        # head mismatch case
        mapper = self._kv_map_head_mismatch(
            transfer_layers,
            kv_factor,
            self_tp_per_dp,
            peer_tp_per_dp,
            self_tp_rank,
            peer_tp_rank,
            self_layer_offset,
            peer_layer_offset,
            peer_ri,
        )
        self._kv_map_cache[key] = mapper
        return mapper

    # ---------------- private helpers ----------------
    def _unique_key(self, name: str, rank: int) -> str:
        return name + str(rank)

    def _tp_per_dp(self, info: RankInfo) -> int:
        return (
            info.tp_size // info.dp_size
            if getattr(info, "enable_attention_dp", False)
            else info.tp_size
        )

    def _block_size(self, layer_num: int, kv_factor: int, ri: RankInfo) -> int:
        return (
            layer_num
            * kv_factor
            * ri.kv_heads_per_rank
            * ri.tokens_per_block
            * ri.dims_per_head
            * ri.element_bytes
        )

    """
    ---- mapper_identity ----

    Pass-through mapping. Do not change pointers or sizes.

    src_ptrs: [ S0 ] [ S1 ] [ S2 ] ...
                |      |      |
                v      v      v
    dst_ptrs: [ D0 ] [ D1 ] [ D2 ] ...
    """

    def _kv_map_identity(self) -> KVMapperFactoryBase.PtrMapper:
        def mapper(
            src_blocks: List[int], src_size: int, dst_blocks: List[int], dst_size: int
        ) -> CopyArgs:
            return src_blocks, dst_blocks, dst_size

        return mapper

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

    def _kv_map_head_match(
        self,
        transfer_layers: int,
        kv_factor: int,
        src_layer_off: int,
        dst_layer_off: int,
        peer_ri: RankInfo,
    ) -> KVMapperFactoryBase.PtrMapper:
        frag_size = self._block_size(transfer_layers, kv_factor, self._ri)
        src_block_off = self._block_size(src_layer_off, kv_factor, self._ri)
        dst_block_off = self._block_size(dst_layer_off, kv_factor, peer_ri)

        @nvtx_range("mapper_head_match")
        def mapper(
            src_blocks: List[int], src_size: int, dst_blocks: List[int], dst_size: int
        ) -> CopyArgs:
            src_frags = [p + src_block_off for p in src_blocks]
            dst_frags = [p + dst_block_off for p in dst_blocks]
            return src_frags, dst_frags, frag_size

        return mapper

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

    def _kv_map_head_mismatch(
        self,
        transfer_layers: int,
        kv_factor: int,
        self_tp_per_dp: int,
        peer_tp_per_dp: int,
        self_tp_rank: int,
        peer_tp_rank: int,
        src_layer_off: int,
        peer_layer_off: int,
        peer_ri: RankInfo,
    ) -> KVMapperFactoryBase.PtrMapper:
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

        def _get_layer_kv_num(ri: RankInfo) -> int:
            return ri.kv_heads_per_rank * ri.tokens_per_block * ri.dims_per_head * ri.element_bytes

        # Each pool has shape [max_blocks, num_layers, kv_factor, num_kv_heads, tokens_per_block,
        # dims_per_head, element_bytes].
        bytes_per_head = self._ri.tokens_per_block * self._ri.dims_per_head * self._ri.element_bytes
        bytes_cont_heads = (
            min(self._ri.kv_heads_per_rank, peer_ri.kv_heads_per_rank) * bytes_per_head
        )

        src_head_off, dst_head_off = _compute_head_offsets(
            self_tp_per_dp,
            peer_tp_per_dp,
            self_tp_rank,
            peer_tp_rank,
            bytes_cont_heads,
        )

        layer_indices = np.arange(transfer_layers, dtype=np.int64)
        kv_indices = np.arange(kv_factor, dtype=np.int64)

        def _get_frags(bases, layer_indices, layer_kv_num, kv_indices, head_off):
            layer_num = layer_kv_num * kv_factor
            return (
                bases[:, None, None]
                + layer_num * layer_indices[None, :, None]
                + layer_kv_num * kv_indices[None, None, :]
                + head_off
            )

        @nvtx_range("mapper_head_mismatch")
        def mapper(
            src_blocks: List[int], src_size: int, dst_blocks: List[int], dst_size: int
        ) -> CopyArgs:
            src_frags = _get_frags(
                bases=np.array(src_blocks, dtype=np.int64),
                layer_indices=src_layer_off + layer_indices,
                layer_kv_num=_get_layer_kv_num(self._ri),
                kv_indices=kv_indices,
                head_off=src_head_off,
            )

            dst_frags = _get_frags(
                bases=np.array(dst_blocks, dtype=np.int64),
                layer_indices=peer_layer_off + layer_indices,
                layer_kv_num=_get_layer_kv_num(peer_ri),
                kv_indices=kv_indices,
                head_off=dst_head_off,
            )

            return src_frags.ravel().tolist(), dst_frags.ravel().tolist(), bytes_cont_heads

        return mapper
