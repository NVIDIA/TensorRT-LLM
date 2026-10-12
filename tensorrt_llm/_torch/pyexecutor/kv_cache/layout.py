# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction

from tensorrt_llm.bindings import DataType
from tensorrt_llm.runtime.kv_cache_manager_v2 import (
    AttentionLayerConfig,
    AttentionReusePolicy,
    BufferConfig,
    DataRole,
    LayerId,
)


class LayerCacheOwner(str, Enum):
    """Model component that reads and writes a cache layer."""

    TARGET = "target"
    DRAFT = "draft"


class LayerCacheTransferPolicy(str, Enum):
    """How a cache layer participates in request-level KV transfer."""

    REQUIRED = "required"
    RECOMPUTABLE = "recomputable"
    LOCAL_ONLY = "local_only"


@dataclass(frozen=True, slots=True)
class LayerBufferSpec:
    """Exact physical buffer allocated for one cache layer.

    ``size`` is the byte size of one physical page, matching
    :class:`BufferConfig`. Keeping the exact size in the pre-construction plan
    lets budgeting and runtime allocation consume the same value.
    """

    role: DataRole
    size: int
    tokens_per_block_override: int | None = None
    is_sparse: bool = False
    transfer_policy: LayerCacheTransferPolicy = LayerCacheTransferPolicy.REQUIRED

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError(f"layer buffer size must be positive, got {self.size}")
        if self.tokens_per_block_override is not None and self.tokens_per_block_override <= 0:
            raise ValueError(
                "tokens_per_block_override must be positive when set, got "
                f"{self.tokens_per_block_override}"
            )

    def bytes_per_token(self, default_tokens_per_block: int) -> Fraction:
        """Return the exact average storage bytes per represented token."""
        tokens_per_block = self.tokens_per_block_override or default_tokens_per_block
        return Fraction(self.size, tokens_per_block)

    def to_buffer_config(self) -> BufferConfig:
        """Convert the semantic plan entry to the bound runtime config."""
        return BufferConfig(
            role=self.role,
            size=self.size,
            tokens_per_block_override=self.tokens_per_block_override,
            is_sparse=self.is_sparse,
        )


@dataclass(frozen=True, slots=True)
class LayerCacheSpec:
    """Semantic and physical cache layout for one rank-local layer."""

    local_layer_id: int
    global_layer_id: int
    owner: LayerCacheOwner
    dtype: DataType
    num_kv_heads: int
    head_dim: int
    buffers: tuple[LayerBufferSpec, ...]
    sliding_window_size: int | None = None
    num_sink_tokens: int | None = None
    reuse_policy: AttentionReusePolicy = AttentionReusePolicy.REQUIRED
    transfer_policy: LayerCacheTransferPolicy = LayerCacheTransferPolicy.REQUIRED

    def __post_init__(self) -> None:
        if self.local_layer_id < 0:
            raise ValueError(f"local_layer_id must be non-negative, got {self.local_layer_id}")
        if self.global_layer_id < 0:
            raise ValueError(f"global_layer_id must be non-negative, got {self.global_layer_id}")
        if self.num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {self.num_kv_heads}")
        if self.head_dim <= 0:
            raise ValueError(f"head_dim must be positive, got {self.head_dim}")
        if not self.buffers:
            raise ValueError("a cache layer must contain at least one buffer")
        roles = [buffer.role for buffer in self.buffers]
        if len(set(roles)) != len(roles):
            raise ValueError(
                f"layer {self.global_layer_id} contains duplicate buffer roles: {roles}"
            )
        sparse_values = {buffer.is_sparse for buffer in self.buffers}
        if len(sparse_values) != 1:
            raise ValueError(f"layer {self.global_layer_id} mixes sparse and dense buffers")
        if self.sliding_window_size is not None and self.sliding_window_size <= 0:
            raise ValueError(
                f"sliding_window_size must be positive when set, got {self.sliding_window_size}"
            )
        if self.num_sink_tokens is not None and self.num_sink_tokens < 0:
            raise ValueError(
                f"num_sink_tokens must be non-negative when set, got {self.num_sink_tokens}"
            )

    @property
    def buffer_roles(self) -> tuple[DataRole, ...]:
        return tuple(buffer.role for buffer in self.buffers)

    @property
    def is_sparse(self) -> bool:
        return self.buffers[0].is_sparse

    def bytes_per_token(self, default_tokens_per_block: int) -> Fraction:
        """Return the exact aggregate bytes per represented token."""
        return sum(
            (buffer.bytes_per_token(default_tokens_per_block) for buffer in self.buffers),
            start=Fraction(),
        )

    def to_attention_layer_config(self) -> AttentionLayerConfig:
        """Build the existing C++-facing configuration for this layer."""
        return AttentionLayerConfig(
            layer_id=LayerId(self.local_layer_id),
            buffers=[buffer.to_buffer_config() for buffer in self.buffers],
            sliding_window_size=self.sliding_window_size,
            num_sink_tokens=self.num_sink_tokens,
            reuse_policy=self.reuse_policy,
        )


@dataclass(frozen=True, slots=True)
class KVCacheLayoutPlan:
    """Immutable rank-local layout consumed by budgeting and allocation."""

    tokens_per_block: int
    layers: tuple[LayerCacheSpec, ...]

    def __post_init__(self) -> None:
        if self.tokens_per_block <= 0:
            raise ValueError(f"tokens_per_block must be positive, got {self.tokens_per_block}")
        if not self.layers:
            raise ValueError("a KV cache layout plan must contain at least one local layer")
        local_layer_ids = tuple(layer.local_layer_id for layer in self.layers)
        expected_local_ids = tuple(range(len(self.layers)))
        if local_layer_ids != expected_local_ids:
            raise ValueError(
                "layout-plan local layer IDs must be dense and ordered: "
                f"expected {expected_local_ids}, got {local_layer_ids}"
            )
        global_layer_ids = [layer.global_layer_id for layer in self.layers]
        if len(set(global_layer_ids)) != len(global_layer_ids):
            raise ValueError(f"layout plan contains duplicate global layer IDs: {global_layer_ids}")

    @property
    def target_layer_ids(self) -> tuple[int, ...]:
        return tuple(
            layer.global_layer_id for layer in self.layers if layer.owner is LayerCacheOwner.TARGET
        )

    @property
    def draft_layer_ids(self) -> tuple[int, ...]:
        return tuple(
            layer.global_layer_id for layer in self.layers if layer.owner is LayerCacheOwner.DRAFT
        )

    @property
    def has_mixed_dtype(self) -> bool:
        return len({layer.dtype for layer in self.layers}) > 1

    @property
    def uniform_dtype(self) -> DataType | None:
        dtypes = {layer.dtype for layer in self.layers}
        return next(iter(dtypes)) if len(dtypes) == 1 else None

    def get_layer(self, global_layer_id: int) -> LayerCacheSpec:
        for layer in self.layers:
            if layer.global_layer_id == global_layer_id:
                return layer
        raise KeyError(f"global layer {global_layer_id} is not in this rank's KV cache layout")

    def required_transfer_layers(self) -> tuple[LayerCacheSpec, ...]:
        return tuple(
            layer
            for layer in self.layers
            if layer.transfer_policy is LayerCacheTransferPolicy.REQUIRED
            and any(
                buffer.transfer_policy is LayerCacheTransferPolicy.REQUIRED
                for buffer in layer.buffers
            )
        )

    def cache_size_layer_components(self) -> tuple[list[int], list[int | None]]:
        """Return per-layer bytes/token and retention windows for budgeting."""
        layer_sizes = []
        for layer in self.layers:
            size = layer.bytes_per_token(self.tokens_per_block)
            if size.denominator != 1:
                raise ValueError(
                    f"layer {layer.global_layer_id} has fractional aggregate bytes/token: {size}"
                )
            layer_sizes.append(int(size))
        return layer_sizes, [layer.sliding_window_size for layer in self.layers]

    def transfer_compatibility_errors(self) -> tuple[str, ...]:
        """Return current native-disaggregation limitations for this plan."""
        layers = self.required_transfer_layers()
        if not layers:
            return ()
        errors = []
        if len({layer.dtype for layer in layers}) > 1:
            errors.append("required transfer layers use different dtypes")
        if len({layer.num_kv_heads for layer in layers}) > 1:
            errors.append("required transfer layers use different KV-head counts")
        if len({layer.head_dim for layer in layers}) > 1:
            errors.append("required transfer layers use different head dimensions")
        kv_roles = {
            tuple(
                buffer.role
                for buffer in layer.buffers
                if buffer.role in _KV_DATA_ROLES
                and buffer.transfer_policy is LayerCacheTransferPolicy.REQUIRED
            )
            for layer in layers
        }
        if len(kv_roles) > 1:
            errors.append("required transfer layers use different K/V role layouts")
        return tuple(errors)

    def to_attention_layer_configs(self) -> list[AttentionLayerConfig]:
        """Build the existing per-layer runtime configurations."""
        return [layer.to_attention_layer_config() for layer in self.layers]


_KV_DATA_ROLES = frozenset({DataRole("key"), DataRole("value")})


__all__ = [
    "KVCacheLayoutPlan",
    "LayerBufferSpec",
    "LayerCacheOwner",
    "LayerCacheSpec",
    "LayerCacheTransferPolicy",
]
