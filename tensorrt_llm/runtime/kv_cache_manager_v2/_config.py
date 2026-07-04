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
from enum import IntEnum
from typing import ClassVar, NewType, Protocol

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

    tokens_per_block_override: int | None = None
    """
    If not None, overrides the tokens_per_block in KVCacheManagerConfig. Must be a factor of tokens_per_block in
    KVCacheManagerConfig and size should be based on tokens_per_block_override.
    """


class LayerType(IntEnum):
    ATTENTION = 0
    SSM = 1


@dataclass(slots=True)
class AttentionLayerConfig:
    type: ClassVar[LayerType] = LayerType.ATTENTION

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
class SsmLayerConfig:
    type: ClassVar[LayerType] = LayerType.SSM

    layer_id: LayerId

    buffers: list[BufferConfig]

    def __post_init__(self) -> None:
        assert len(set(buffer.role for buffer in self.buffers)) == len(self.buffers), (
            "duplicate buffer role"
        )
        assert all(buf.tokens_per_block_override is None for buf in self.buffers)


LayerConfig = AttentionLayerConfig | SsmLayerConfig


@dataclass(slots=True, frozen=True)
class KVCacheDesc:
    capacity: int
    history_length: int

    def __post_init__(self) -> None:
        assert 0 <= self.history_length <= self.capacity


# A batch of requests, working as a use case the KVCacheManager must always support.
@dataclass(slots=True, frozen=True)
class BatchDesc:
    kv_caches: list[KVCacheDesc]
    # Tokens shared by all requests. Set to 0 if no kv cache reuse.
    system_prompt_length: int = 0

    def __post_init__(self) -> None:
        assert self.system_prompt_length >= 0


@dataclass(slots=True)
class HelixConfig:
    helix_group_size: int
    helix_gpu_rank: int
    # number of tokens in one helix shard
    helix_shard_size: int
    # must be the same for all ranks in the same helix group and different for different helix groups.
    shared_comm_port: int


@dataclass(slots=True)
class SwaScratchReuseConfig:
    """
    Configuration for SWA scratch reuse.

    Args:
        max_rewind_len: Maximum number of tail tokens that can be rewound after
            scratch-enabled allocation. Scratch reuse will not cover blocks that
            may be needed to preserve those tokens.
    """

    max_rewind_len: int = 0

    def __post_init__(self) -> None:
        assert self.max_rewind_len >= 0, "max_rewind_len must be non-negative"


class WriteThroughPolicy(IntEnum):
    """When active (GPU) blocks are copied out to the host (stale) tier.

    See ``contiguous_primary_kvcache/DESIGN.md`` §4.3.
    """

    # Copy a committed block to host only when its arena pages are freed
    # (sequence completion / preemption). Simplest; bursts D2H at free time.
    ON_FREE = 0
    # Copy a committed block to host opportunistically the moment it commits
    # (blocks are immutable once committed), making the common free path
    # copy-free. Must be rate-limited (DESIGN.md §4.3 risk #4).
    ON_COMMIT = 1


@dataclass(slots=True)
class ContiguousArenaConfig:
    """Enables the contiguous-in-VA active KV cache (per-sequence arenas backed
    by CUDA VMM demand paging).

    This is a **prototype** feature; see ``contiguous_primary_kvcache/DESIGN.md``
    for the full design. When this config is ``None`` on
    :class:`KVCacheManagerConfig`, behavior is unchanged (classic paged v2).

    Args:
        phys_page_size: Physical mapping granularity in bytes ("super-page").
            Must be a multiple of the VMM allocation granularity (2 MiB). Larger
            pages amortize the fixed per-mapping ``cuMemSetAccess`` cost (~8x
            cheaper per byte at 16 MiB vs 2 MiB) at the cost of more tail waste
            (~page_size/2 per sequence per pool group). See DESIGN.md §4.8.
        map_ahead_pages: Number of physical pages to keep mapped ahead of a
            growing sequence's write frontier, to hide driver-call latency and
            absorb speculative-decoding bursts (DESIGN.md §4.2).
        write_through: See :class:`WriteThroughPolicy`.
        lazy_gpu_retention: If True, keep a freed sequence's pages mapped
            (LRU-ordered) until the shared pool needs them, so a reuse hit on a
            still-resident block is a D2D copy instead of H2D (DESIGN.md §4.4,
            phase 1). P0 default is False (stale is host-only).
        max_va_bytes_per_pool: Hard cap on the VA reservation per (pool group,
            pool) arena. 0 means derive from per-request maxima at startup.
    """

    phys_page_size: int = 2 << 20
    map_ahead_pages: int = 1
    write_through: WriteThroughPolicy = WriteThroughPolicy.ON_FREE
    lazy_gpu_retention: bool = False
    max_va_bytes_per_pool: int = 0
    # Defer growth maps and execute them in one batched pass per iteration
    # (§4.2). The budget is still charged at resize time, so admission
    # control is unchanged -- but the OWNER of the manager must call
    # KVCacheManager.flush_gpu_mappings() after scheduling and before any
    # GPU work touches the newly grown blocks (the executor adapter does).
    # Off by default so direct users keep synchronous mapping semantics.
    batched_map_sweep: bool = False

    def __post_init__(self) -> None:
        _MIN_GRANULARITY = 2 << 20
        assert self.phys_page_size > 0 and self.phys_page_size % _MIN_GRANULARITY == 0, (
            f"phys_page_size ({self.phys_page_size}) must be a positive multiple of "
            f"the 2 MiB VMM granularity"
        )
        assert self.map_ahead_pages >= 0, "map_ahead_pages must be non-negative"
        assert self.max_va_bytes_per_pool >= 0, "max_va_bytes_per_pool must be non-negative"


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
    layers: list[LayerConfig]

    # When memory utilization is above this threshold, KV cache resuming will fail. This helps
    # reserving some memory for KVCache growth and avoids frequent suspend/resume for dynamic batch size.
    max_util_for_resume: float = 0.97

    enable_partial_reuse: bool = True
    """
    If True, we will try to reuse tokens from partially matched blocks.
    """

    constraints: list[BatchDesc] = field(default_factory=list)
    """
    A list of step configurations that must always be supported.
    """

    typical_step: BatchDesc | None = None
    """
    A typical step configuration used to decide initial memory partitioning between
    layer groups.
    """

    ssm_reuse_interval: int = 512
    """
    Interval (in tokens) at which SSM state is snapshotted for prefix reuse.
    Must be a positive multiple of tokens_per_block. Only takes effect when SSM layers are present.
    """

    swa_scratch_reuse: SwaScratchReuseConfig | None = None
    """
    When set, SWA layers reuse physical pages for out-of-window blocks during prefill.
    Scratch blocks share coalesced slot sub-pages across blocks for the currently executing
    layer, reducing peak memory. Trade-off: KV cache reuse is degraded because scratch blocks
    have no preserved data after the step.

    If max_rewind_len is non-zero, the rewindable tail is excluded from scratch reuse so
    draft/target shared KV cache can preserve tokens that may survive speculative rewind.

    Most useful for disaggregated prefill servers handling long prompts or long prompt chunks,
    where the number of out-of-window blocks dominates memory usage.
    """

    enable_stats: bool = True
    """
    Collect V2 KV cache allocation, reuse, and transfer statistics.
    """

    contiguous_arena: ContiguousArenaConfig | None = None
    """
    When set, enables the prototype contiguous-in-VA active KV cache: each
    sequence's active blocks are laid out contiguously in a per-sequence virtual
    address range backed by CUDA VMM demand paging, and stale (reusable) blocks
    live in the host tier. When None (default), behavior is unchanged.
    See ``contiguous_primary_kvcache/DESIGN.md``.
    """

    # unsupported yet
    helix_config: HelixConfig | None = None

    @property
    def enable_swa_scratch_reuse(self) -> bool:
        return self.swa_scratch_reuse is not None

    def __post_init__(self) -> None:
        assert self.cache_tiers and self.cache_tiers[0].tier == CacheTier.GPU_MEM
        assert len(set(layer.layer_id for layer in self.layers)) == len(self.layers), (
            "duplicate layer id"
        )
        assert all(
            buffer.tokens_per_block_override is None
            or self.tokens_per_block % buffer.tokens_per_block_override == 0
            for layer in self.layers
            for buffer in layer.buffers
        )
        if any(layer.type == LayerType.SSM for layer in self.layers):
            assert self.ssm_reuse_interval > 0, "ssm_reuse_interval must be positive"
            assert self.ssm_reuse_interval % self.tokens_per_block == 0, (
                f"ssm_reuse_interval ({self.ssm_reuse_interval}) must be a multiple of "
                f"tokens_per_block ({self.tokens_per_block})"
            )
            assert not self.enable_partial_reuse, (
                "enable_partial_reuse must be False when SSM layers are present"
            )
        if self.contiguous_arena is not None:
            # Prototype feature; a few combinations are not wired up yet (P0 scope,
            # DESIGN.md §7). Fail loudly rather than silently misbehave.
            assert self.helix_config is None, (
                "contiguous_arena is not yet supported together with helix_config"
            )
            assert all(layer.type == LayerType.ATTENTION for layer in self.layers), (
                "contiguous_arena does not support SSM layers yet (DESIGN.md §4.7)"
            )
            assert all(layer.sliding_window_size is None for layer in self.layers), (
                "contiguous_arena does not support sliding-window layers yet (DESIGN.md §4.7)"
            )
            assert self.swa_scratch_reuse is None, (
                "contiguous_arena does not support SWA scratch reuse (scattered scratch slots)"
            )
