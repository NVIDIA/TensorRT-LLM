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

# Currently, our nvfp4 kernels require that KV data and its corresponding KV block scale use the same
# block index, but different base address.
# As the ratio between KV data size and KV block scale size is fixed, we can simply use a pool with
# smaller block size and the same number of blocks for block scale.
import os
from dataclasses import dataclass, field
from typing import NewType, Protocol

from ._common import CacheTier, LayerId

# The data role of a buffer inside one layer.
# Must be unique for each buffer inside a layer.
# Examples: "key", "value", "key_block_quant", "value_block_quant".
DataRole = NewType("DataRole", str)


class CacheTierConfig(Protocol):
    """Protocol for cache tier configuration."""

    quota: int  # in bytes

    @property
    def tier(self) -> CacheTier: ...

    def assert_valid(self) -> None: ...


@dataclass(slots=True)
class GpuCacheTierConfig:
    quota: int  # in bytes

    @property
    def tier(self) -> CacheTier:
        return CacheTier.GPU_MEM

    def assert_valid(self) -> None:
        assert self.quota > 0, "Quota must be positive"


@dataclass(slots=True)
class HostCacheTierConfig:
    quota: int  # in bytes

    @property
    def tier(self) -> CacheTier:
        return CacheTier.HOST_MEM

    def assert_valid(self) -> None:
        assert self.quota > 0, "Quota must be positive"


@dataclass(slots=True)
class DiskCacheTierConfig:
    quota: int  # in bytes
    path: str  # a folder where we will store data as files

    @property
    def tier(self) -> CacheTier:
        return CacheTier.DISK

    def assert_valid(self) -> None:
        assert self.quota > 0, "Quota must be positive"
        assert os.path.isdir(self.path), (
            f"Disk path {self.path} does not exist or is not a directory"
        )


@dataclass(slots=True)
class BufferConfig:
    role: DataRole
    size: int


@dataclass(slots=True)
class AttentionLayerConfig:
    layer_id: LayerId
    # Each page can have multiple sub-pages, e.g. separate K and V data, block quantization scales for K and/or V, etc.
    # KV cache manager will automatically group sub-pages of the same size, and redirect pages of different sizes to
    # different memory pools

    # BufferConfig.role should not duplicate
    buffers: list[BufferConfig]
    # Note that we use None to represent "no sliding window". Sink tokens are excluded.
    sliding_window_size: int | None = None
    num_sink_tokens: int | None = None

    @property
    def window_size(self) -> int | None:
        return self.sliding_window_size

    def __post_init__(self) -> None:
        assert len(set(buffer.role for buffer in self.buffers)) == len(self.buffers), (
            "duplicate buffer role"
        )


@dataclass(slots=True)
class HelixConfig:
    helix_group_size: int
    helix_gpu_rank: int
    # number of tokens in one helix shard
    helix_shard_size: int
    # must be the same for all ranks in the same helix group and different for different helix groups.
    shared_comm_port: int


@dataclass(slots=True)
class KVCacheManagerConfig:
    """
    Configuration for the KV cache manager.
    """

    tokens_per_block: int
    # if you have p-tuning tokens, include them. Only needed for multi-modal.
    vocab_size: int
    # cache tiers are sorted from warm to cold. The first one must be GPU memory.
    cache_tiers: list[CacheTierConfig]

    # AttentionLayerConfig.layer_id should not duplicate
    layers: list[AttentionLayerConfig]

    # When memory utilization is above this threshold, KV cache resuming will fail. This helps
    # reserving some memory for KVCache growth and avoids frequent suspend/resume for dynamic batch size.
    max_util_for_resume: float = field(default=0.9)

    # unsupported yet
    helix_config: HelixConfig | None = field(default=None)

    def __post_init__(self) -> None:
        assert self.cache_tiers and self.cache_tiers[0].tier == CacheTier.GPU_MEM
        assert len(set(layer.layer_id for layer in self.layers)) == len(self.layers), (
            "duplicate layer id"
        )
