from dataclasses import asdict, dataclass
from typing import List, Optional

import msgpack

from tensorrt_llm._torch.disaggregation.native.region.aux import AuxBufferMeta
from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVPoolAttrs


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
        return cls(**msgpack.unpackb(data, strict_map_key=False))


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
    server_endpoint: str
    self_endpoint: str
    transfer_engine_info: bytes
    aux_meta: Optional[AuxBufferMeta] = None
    kv_pool_attrs: Optional[KVPoolAttrs] = None

    @property
    def kv_factor(self) -> int:
        return 2 if not self.is_mla else 1

    def to_bytes(self) -> bytes:
        data = asdict(self)
        data["aux_meta"] = self.aux_meta.to_dict() if self.aux_meta is not None else None
        data["kv_pool_attrs"] = (
            self.kv_pool_attrs.to_dict() if self.kv_pool_attrs is not None else None
        )
        return msgpack.packb(data)

    @classmethod
    def from_bytes(cls, data: bytes) -> "RankInfo":
        unpacked = msgpack.unpackb(data, strict_map_key=False)
        if unpacked.get("aux_meta") is not None:
            unpacked["aux_meta"] = AuxBufferMeta.from_dict(unpacked["aux_meta"])
        if unpacked.get("kv_pool_attrs") is not None:
            unpacked["kv_pool_attrs"] = KVPoolAttrs.from_dict(unpacked["kv_pool_attrs"])
        return cls(**unpacked)
