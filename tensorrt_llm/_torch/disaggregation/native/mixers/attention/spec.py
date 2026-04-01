from dataclasses import asdict, dataclass


# [numLayers, kv_factor, heads, tokens, dims_per_head]
@dataclass
class AttentionInfo:
    kv_heads_per_rank: int
    tokens_per_block: int
    dims_per_head: int
    element_bytes: int
    enable_attention_dp: bool
    is_mla: bool

    @property
    def kv_factor(self) -> int:
        return 2 if not self.is_mla else 1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AttentionInfo":
        return cls(**data)
