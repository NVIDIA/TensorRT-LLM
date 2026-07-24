from dataclasses import asdict, dataclass
from typing import List, Optional

import msgpack

from tensorrt_llm._torch.disaggregation.native.auxiliary import AuxBufferMeta
from tensorrt_llm._torch.disaggregation.native.mixers.attention.spec import AttentionInfo
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import build_page_table_from_manager
from tensorrt_llm._torch.disaggregation.resource.page import KVCachePageTable
from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
from tensorrt_llm._utils import get_size_in_bytes


@dataclass
class RankInfo:
    instance_name: str
    instance_rank: int
    tp_size: int
    tp_rank: int
    pp_size: int
    pp_rank: int
    layer_num_per_pp: List[int]
    sender_endpoints: List[str]
    server_endpoint: str
    self_endpoint: str
    transfer_engine_info: bytes

    dp_size: int = 1
    dp_rank: int = 0
    cp_size: int = 1
    cp_rank: int = 0
    device_id: int = 0

    attention: Optional[AttentionInfo] = None
    aux_meta: Optional[AuxBufferMeta] = None
    page_table: Optional[KVCachePageTable] = None

    @property
    def tp_size_per_dp_group(self) -> int:
        if self.attention is None:
            return self.tp_size
        return self.tp_size // self.dp_size if self.attention.enable_attention_dp else self.tp_size

    def to_bytes(self) -> bytes:
        data = asdict(self)
        data["attention"] = self.attention.to_dict() if self.attention is not None else None
        data["aux_meta"] = self.aux_meta.to_dict() if self.aux_meta is not None else None
        data["page_table"] = self.page_table.to_dict() if self.page_table is not None else None
        return msgpack.packb(data)

    @classmethod
    def from_kv_cache_manager(
        cls,
        instance_name: str,
        kv_cache_manager: KVCacheManager,
        device_id: int,
        aux_buffer_meta: Optional[AuxBufferMeta] = None,
    ) -> "RankInfo":
        m = kv_cache_manager.mapping
        kvm = kv_cache_manager
        enable_attention_dp = m.enable_attention_dp
        return cls(
            instance_name=instance_name,
            instance_rank=m.rank,
            tp_size=m.tp_size,
            tp_rank=m.tp_rank,
            pp_size=m.pp_size,
            pp_rank=m.pp_rank,
            dp_size=m.tp_size if enable_attention_dp else m.dp_size,
            dp_rank=m.tp_rank if enable_attention_dp else 0,
            cp_size=m.cp_size,
            cp_rank=m.cp_rank,
            device_id=device_id,
            layer_num_per_pp=[len(kvm.pp_layers)],
            sender_endpoints=[],
            server_endpoint="",
            self_endpoint="",
            transfer_engine_info=bytes(),
            attention=AttentionInfo(
                kv_heads_per_rank=kvm.num_kv_heads_per_layer[0],
                tokens_per_block=kvm.tokens_per_block,
                dims_per_head=kvm.head_dim,
                element_bytes=get_size_in_bytes(1, kvm.dtype),
                enable_attention_dp=enable_attention_dp,
                is_mla=kvm.kv_factor == 1,
            ),
            aux_meta=aux_buffer_meta,
            page_table=build_page_table_from_manager(kvm),
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "RankInfo":
        unpacked = msgpack.unpackb(data, strict_map_key=False)
        if unpacked.get("attention") is not None:
            unpacked["attention"] = AttentionInfo.from_dict(unpacked["attention"])
        if unpacked.get("page_table") is not None:
            unpacked["page_table"] = KVCachePageTable.from_dict(unpacked["page_table"])
        if unpacked.get("aux_meta") is not None:
            unpacked["aux_meta"] = AuxBufferMeta.from_dict(unpacked["aux_meta"])
        return cls(**unpacked)
