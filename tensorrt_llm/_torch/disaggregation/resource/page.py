from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

BUFFER_ENTRY_DTYPE = np.dtype(
    [
        ("local_layer_id", np.uint32),
        ("role", np.uint32),
        ("offset", np.uint32),
        ("size", np.uint32),
    ]
)


@dataclass
class PhysicalPool:
    base_address: int  # uint64
    slot_bytes: int
    num_slots: int

    def to_dict(self) -> dict:
        return {
            "base_address": int(self.base_address),
            "slot_bytes": int(self.slot_bytes),
            "num_slots": int(self.num_slots),
        }

    @staticmethod
    def from_dict(data: dict) -> "PhysicalPool":
        return PhysicalPool(
            base_address=int(data["base_address"]),
            slot_bytes=int(data["slot_bytes"]),
            num_slots=int(data["num_slots"]),
        )


@dataclass
class PhysicalPoolGroup:
    pools: List[PhysicalPool]

    def to_dict(self) -> dict:
        return {"pools": [p.to_dict() for p in self.pools]}

    @classmethod
    def from_dict(cls, data: dict) -> "PhysicalPoolGroup":
        return cls(pools=[PhysicalPool.from_dict(p) for p in data.get("pools", [])])


@dataclass(frozen=True)
class LocalLayer:
    """Mapping between a local/internal layer id and a collision-free global layer id."""

    local_layer_id: int
    global_layer_id: int

    def to_dict(self) -> dict:
        return {
            "local_layer_id": int(self.local_layer_id),
            "global_layer_id": int(self.global_layer_id),
        }

    @staticmethod
    def from_dict(data: dict) -> "LocalLayer":
        return LocalLayer(
            local_layer_id=int(data["local_layer_id"]),
            global_layer_id=int(data["global_layer_id"]),
        )


@dataclass
class PoolView:
    """
    Per-layer-group view of a physical pool (slot layout for this life cycle).
    """

    pool_idx: int
    buffer_entries: np.ndarray  # dtype=BUFFER_ENTRY_DTYPE

    def to_dict(self) -> dict:
        return {
            "pool_idx": int(self.pool_idx),
            "buffer_entries": self.buffer_entries.tolist(),
        }

    @staticmethod
    def from_dict(data: dict) -> "PoolView":
        raw = data.get("buffer_entries", [])
        # msgpack deserializes tuples as lists; np.array requires tuples for
        # structured dtypes (enforced in numpy >=2.0), so convert explicitly.
        return PoolView(
            pool_idx=int(data["pool_idx"]),
            buffer_entries=np.array(
                [tuple(row) for row in raw],
                dtype=BUFFER_ENTRY_DTYPE,
            ),
        )


@dataclass
class LayerGroup:
    """
    Base class for one life cycle / layer-group.
    """

    pool_group_idx: int

    def to_dict(self) -> dict:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, data: dict) -> "LayerGroup":
        if data.get("kv_head_num_per_rank") is not None:
            return AttentionLayerGroup.from_dict(data)
        elif data.get("mamba_layer_offsets") is not None:
            return MambaLayerGroup.from_dict(data)
        raise ValueError("Invalid layer group")


@dataclass
class AttentionLayerGroup(LayerGroup):
    """Layer group for attention KV cache."""

    kv_head_num_per_rank: int = 0
    sliding_window_size: Optional[int] = None
    local_layers: List[LocalLayer] = field(default_factory=list)
    pool_views: List[PoolView] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "pool_group_idx": int(self.pool_group_idx),
            "kv_head_num_per_rank": int(self.kv_head_num_per_rank),
            "sliding_window_size": self.sliding_window_size,
            "local_layers": [ll.to_dict() for ll in self.local_layers],
            "pool_views": [pv.to_dict() for pv in self.pool_views],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AttentionLayerGroup":
        return cls(
            pool_group_idx=int(data["pool_group_idx"]),
            kv_head_num_per_rank=int(data["kv_head_num_per_rank"]),
            sliding_window_size=data.get("sliding_window_size"),
            local_layers=[LocalLayer.from_dict(x) for x in data.get("local_layers", [])],
            pool_views=[PoolView.from_dict(pv) for pv in data.get("pool_views", [])],
        )


@dataclass
class MambaLayerGroup(LayerGroup):
    """Layer group for Mamba SSM states."""

    mamba_layer_offsets: Dict[int, int] = field(default_factory=dict)
    conv_states: Optional[PhysicalPool] = None
    ssm_states: Optional[PhysicalPool] = None
    conv_section_bytes: Optional[List[int]] = None
    ssm_bytes_per_head: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "pool_group_idx": int(self.pool_group_idx),
            "mamba_layer_offsets": {int(k): int(v) for k, v in self.mamba_layer_offsets.items()},
            "conv_states": self.conv_states.to_dict(),
            "ssm_states": self.ssm_states.to_dict(),
            "conv_section_bytes": self.conv_section_bytes,
            "ssm_bytes_per_head": self.ssm_bytes_per_head,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MambaLayerGroup":
        conv_section_bytes = data.get("conv_section_bytes")
        ssm_bytes_per_head = data.get("ssm_bytes_per_head")
        return cls(
            pool_group_idx=int(data["pool_group_idx"]),
            mamba_layer_offsets={int(k): int(v) for k, v in data["mamba_layer_offsets"].items()},
            conv_states=PhysicalPool.from_dict(data["conv_states"]),
            ssm_states=PhysicalPool.from_dict(data["ssm_states"]),
            conv_section_bytes=[int(x) for x in conv_section_bytes]
            if conv_section_bytes is not None
            else None,
            ssm_bytes_per_head=int(ssm_bytes_per_head) if ssm_bytes_per_head is not None else None,
        )


@dataclass
class KVCachePageTable:
    tokens_per_block: int
    layer_groups: List[LayerGroup]
    pool_groups: List[PhysicalPoolGroup]  # indexed by LayerGroup.pool_group_idx

    def to_dict(self) -> dict:
        return {
            "tokens_per_block": int(self.tokens_per_block),
            "layer_groups": [lg.to_dict() for lg in self.layer_groups],
            "pool_groups": [pg.to_dict() for pg in self.pool_groups],
        }

    @staticmethod
    def from_dict(data: dict) -> "KVCachePageTable":
        return KVCachePageTable(
            tokens_per_block=int(data["tokens_per_block"]),
            layer_groups=[LayerGroup.from_dict(lg) for lg in data.get("layer_groups", [])],
            pool_groups=[PhysicalPoolGroup.from_dict(pg) for pg in data.get("pool_groups", [])],
        )
