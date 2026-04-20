# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Self-contained distributed configuration for AutoDeploy sharding.

``DistConfig`` replaces the dependency on ``tensorrt_llm.mapping.Mapping``
within AutoDeploy.  It carries the minimal set of parallelism parameters
needed by the sharding transforms and custom ops, plus serialization
support for graph-level metadata (e.g., MoE all-to-all dispatch).
"""

import json
from typing import Any

from pydantic import BaseModel, Field, model_validator


class DistConfig(BaseModel):
    """Distributed parallelism configuration for AutoDeploy."""

    model_config = {"extra": "allow"}

    world_size: int = Field(default=1, ge=1)
    rank: int = Field(default=0, ge=0)
    tp_size: int = Field(default=1, ge=1)
    pp_size: int = Field(default=1, ge=1)
    moe_tp_size: int = Field(default=1, ge=1)
    moe_ep_size: int = Field(default=1, ge=1)
    moe_cluster_size: int = Field(default=1, ge=1)
    enable_attention_dp: bool = Field(default=False)
    allreduce_strategy: str = Field(default="NCCL")

    @model_validator(mode="after")
    def _validate_grid(self) -> "DistConfig":
        if self.rank >= self.world_size:
            raise ValueError(f"rank ({self.rank}) must be < world_size ({self.world_size})")
        if self.tp_size > self.world_size:
            raise ValueError(f"tp_size ({self.tp_size}) must be <= world_size ({self.world_size})")
        moe_grid = self.moe_tp_size * self.moe_ep_size * self.moe_cluster_size
        if moe_grid != self.tp_size:
            raise ValueError(
                f"moe_tp_size * moe_ep_size * moe_cluster_size ({moe_grid}) "
                f"must equal tp_size ({self.tp_size})"
            )
        return self

    @property
    def tp_rank(self) -> int:
        """Local rank within tensor parallelism (0 .. tp_size - 1)."""
        return self.rank % self.tp_size

    @property
    def pp_rank(self) -> int:
        """Pipeline-parallel stage index for this process."""
        return self.rank // self.tp_size

    @property
    def moe_tp_rank(self) -> int:
        """MoE tensor-parallel rank within the MoE TP subgroup."""
        return self.tp_rank // (self.moe_ep_size * self.moe_cluster_size)

    @property
    def moe_ep_rank(self) -> int:
        """Expert-parallel rank derived from the tensor-parallel rank."""
        return self.tp_rank % self.moe_ep_size

    @property
    def moe_cluster_rank(self) -> int:
        """MoE cluster index derived from the tensor-parallel rank."""
        return self.tp_rank % self.moe_cluster_size

    def to_dict(self) -> dict:
        """Return a plain dict of serializable DistConfig fields."""
        return {
            "world_size": self.world_size,
            "rank": self.rank,
            "tp_size": self.tp_size,
            "pp_size": self.pp_size,
            "moe_tp_size": self.moe_tp_size,
            "moe_ep_size": self.moe_ep_size,
            "moe_cluster_size": self.moe_cluster_size,
            "enable_attention_dp": self.enable_attention_dp,
            "allreduce_strategy": self.allreduce_strategy,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DistConfig":
        """Construct from a dict, ignoring keys that are not DistConfig fields."""
        known = {f for f in cls.model_fields}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    def serialize(self) -> str:
        """JSON string for this config (via ``to_dict``)."""
        return json.dumps(self.to_dict())

    @classmethod
    def deserialize(cls, s: str) -> "DistConfig":
        """Parse a JSON string into a ``DistConfig``."""
        return cls.from_dict(json.loads(s))

    @staticmethod
    def from_mapping(mapping: Any) -> "DistConfig":
        """Construct from a ``tensorrt_llm.mapping.Mapping`` instance."""
        return DistConfig(
            world_size=mapping.world_size,
            rank=mapping.rank,
            tp_size=mapping.tp_size,
            pp_size=mapping.pp_size,
            moe_tp_size=mapping.moe_tp_size,
            moe_ep_size=mapping.moe_ep_size,
            moe_cluster_size=mapping.moe_cluster_size,
            enable_attention_dp=mapping.enable_attention_dp,
        )

    def to_mapping(self) -> Any:
        """Convert back to a ``tensorrt_llm.mapping.Mapping`` for C++ op interop."""
        from tensorrt_llm.mapping import Mapping  # will be deprecated by DistConfig

        return Mapping(
            world_size=self.world_size,
            rank=self.rank,
            tp_size=self.tp_size,
            pp_size=self.pp_size,
            moe_tp_size=self.moe_tp_size,
            moe_ep_size=self.moe_ep_size,
            moe_cluster_size=self.moe_cluster_size,
            enable_attention_dp=self.enable_attention_dp,
        )

    def print_grid(self) -> str:
        """Human-readable summary of the TP / MoE parallelism grid."""
        return (
            f"process grid: [TP, MoE_TP, MoE_EP] = "
            f"[{self.tp_size}, {self.moe_tp_size}, {self.moe_ep_size}]"
        )

    def print_rank(self) -> str:
        """Human-readable summary of this process's rank assignments."""
        return f"rank: [{self.rank}, {self.moe_tp_rank}, {self.moe_ep_rank}]"
