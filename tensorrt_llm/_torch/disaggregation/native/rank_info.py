from dataclasses import asdict, dataclass
from typing import List, Optional

import msgpack

from tensorrt_llm._torch.disaggregation.native.auxiliary import AuxBufferMeta
from tensorrt_llm._torch.disaggregation.native.mixers.attention.spec import AttentionInfo
from tensorrt_llm._torch.disaggregation.resource.page import KVCachePageTable


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
    def from_bytes(cls, data: bytes) -> "RankInfo":
        unpacked = msgpack.unpackb(data, strict_map_key=False)
        if unpacked.get("attention") is not None:
            unpacked["attention"] = AttentionInfo.from_dict(unpacked["attention"])
        if unpacked.get("page_table") is not None:
            unpacked["page_table"] = KVCachePageTable.from_dict(unpacked["page_table"])
        if unpacked.get("aux_meta") is not None:
            unpacked["aux_meta"] = AuxBufferMeta.from_dict(unpacked["aux_meta"])
        return cls(**unpacked)
