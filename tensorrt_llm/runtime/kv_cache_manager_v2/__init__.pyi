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

import array
import enum
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    ClassVar,
    Final,
    Iterable,
    Iterator,
    NamedTuple,
    NewType,
    Protocol,
    Sequence,
    Type,
    TypeAlias,
    Union,
    overload,
)

from tensorrt_llm.bindings.executor import KvCacheRetentionConfig

# From _common.py
NDEBUG: Final[int]
DEFAULT_BEAM_INDEX: Final[BeamIndex]
DEFAULT_COLD_TIER_OFFLOAD_MIN_PRIORITY: Final[int]
BAD_PAGE_INDEX: Final[int]
GPU_LEVEL: Final[CacheLevel]
CACHE_LEVEL1: Final[CacheLevel]

class CorruptedError(Exception):
    """Raised by every public entry point once a broken invariant has been recorded."""

class CuError(Exception):
    """A CUDA driver call failed; carries the driver's own status code."""

    error_code: Any

class OutOfMemoryError(Exception): ...

class HostOOMError(OutOfMemoryError):
    """The host tier could not be backed by physical memory."""

class DiskOOMError(OutOfMemoryError):
    """The disk tier's backing file could not be grown."""

class CuOOMError(OutOfMemoryError):
    """A CUDA allocation failed for want of device memory."""

class LogicError(Exception):
    """A caller broke a precondition the API states, such as seating a cache twice."""

class OutOfPagesError(Exception): ...

def poison_reason() -> str | None:
    """First recorded invariant violation, or None. Never clears, so it is safe to poll."""

def take_poison() -> str | None:
    """Report the recorded violation and clear it, but only once no manager is alive."""

def num_live_managers() -> int:
    """Number of constructed, not-yet-destroyed managers."""

class CacheTier(enum.IntEnum):
    GPU_MEM = 0
    HOST_MEM = 1
    DISK = 2

class PageStatus(enum.Enum):
    LOCKED = enum.auto()
    HELD = enum.auto()
    DROPPABLE = enum.auto()

class PageIndexMode(enum.IntEnum):
    SHARED = 0
    PER_LAYER = 1

LifeCycleId = NewType("LifeCycleId", int)
CacheGroupId: TypeAlias = LifeCycleId
LayerGroupId: TypeAlias = LifeCycleId

class AttnLifeCycle:
    """The attention life cycle, keyed by window, sink tokens, and reuse policy."""

    def __init__(
        self,
        window_size: int | None,
        num_sink_blocks: int,
        reuse_policy: ReusePolicy = ...,
    ) -> None: ...
    @staticmethod
    def make(
        window_size: int | None,
        num_sink_tokens: int | None,
        tokens_per_block: int,
        reuse_policy: ReusePolicy = ...,
    ) -> "AttnLifeCycle": ...
    @property
    def window_size(self) -> int | None: ...
    @property
    def num_sink_blocks(self) -> int: ...
    @property
    def reuse_policy(self) -> ReusePolicy: ...
    def get_stale_range(self, history_length: int, tokens_per_block: int) -> HalfOpenRange: ...

CacheLevel = NewType("CacheLevel", int)
TokenId = NewType("TokenId", int)

class MmItemContext:
    def __init__(self, digest: bytes, uuid: str | None = None) -> None: ...
    @property
    def digest(self) -> bytes: ...
    @property
    def uuid(self) -> str | None: ...

TokenIdExt = Union[TokenId, bytes, MmItemContext]

class PlannedDropHandle:
    def drop(self) -> None: ...

class ReuseScope:
    """Reuse identity; salt is also reported in stored-block events."""

    def __init__(
        self,
        lora_id: int | None = None,
        salt: str | None = None,
    ) -> None: ...
    @property
    def lora_id(self) -> int | None: ...
    @property
    def salt(self) -> str | None: ...
    def __len__(self) -> int: ...
    def __getitem__(self, index: int) -> int | str | None: ...
    def __iter__(self) -> Iterator[int | str | None]: ...
    def __hash__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...

SlidingWindowSize: TypeAlias = int | None
LayerId = NewType("LayerId", int)
CudaStream = NewType("CudaStream", int)
BeamIndex = NewType("BeamIndex", int)
MemAddress = NewType("MemAddress", int)
Priority = NewType("Priority", int)
PoolGroupIndex = NewType("PoolGroupIndex", int)
PoolIndex = NewType("PoolIndex", int)
BatchRow = NewType("BatchRow", int)

# From _storage_manager.py
class StorageStatistics:
    """Independent per-pool storage counts returned by the manager."""

    @property
    def slot_sizes(self) -> list[int]: ...
    @property
    def total(self) -> int: ...
    @property
    def free(self) -> int: ...
    @property
    def evictable(self) -> int: ...
    @property
    def available(self) -> int: ...
    @property
    def unavailable(self) -> int: ...

# From _stats.py
@dataclass(slots=True)
class KVCacheStatsDelta:
    alloc_total_blocks: int = 0
    alloc_new_blocks: int = 0
    reused_blocks: int = 0
    missed_blocks: int = 0

@dataclass(slots=True)
class KVCacheIterationStatsDelta:
    iter_alloc_total_blocks: int = 0
    iter_alloc_new_blocks: int = 0
    iter_reused_blocks: int = 0
    iter_full_reused_blocks: int = 0
    iter_partial_reused_blocks: int = 0
    iter_missed_blocks: int = 0
    iter_gen_alloc_blocks: int = 0
    iter_onboard_blocks: int = 0
    iter_onboard_bytes: int = 0
    iter_offload_blocks: int = 0
    iter_offload_bytes: int = 0
    iter_intra_device_copy_blocks: int = 0
    iter_intra_device_copy_bytes: int = 0
    iter_host_dropped_blocks: int = 0
    iter_host_dropped_bytes: int = 0

@dataclass(slots=True)
class ReusedBlocksByLevel:
    """Reuse block counts split by the cache level the reused pages were resident on.

    Indices are CacheLevel values, so entry i is the i-th configured tier.
    """

    full: list[int]
    partial: list[int]

class SsmSnapshotIterationStatsDelta:
    iter_snapshot_lookups: int = 0
    iter_snapshot_hits: int = 0
    iter_snapshot_misses: int = 0
    iter_reused_tokens: int = 0
    iter_unreused_tokens: int = 0
    iter_aligned_snapshot_hits: int = 0
    iter_unaligned_snapshot_hits: int = 0
    @property
    def iter_snapshot_hit_rate(self) -> float: ...

@dataclass(slots=True, frozen=True)
class PoolGroupPeakBlockStats:
    available: int
    unavailable: int
    evictable: int

class DeviceArray(Protocol):
    """A dense array in GPU memory, borrowed for the duration of a call.

    Structural, not nominal: anything exporting the DLPack protocol over a CUDA buffer
    satisfies it, so a ``torch.Tensor`` is accepted without this package depending on
    torch. Callers may equally pass a CuPy, JAX or numba device array.

    Shape and dtype are stated per parameter rather than in the type, and are checked at
    the boundary. Contents are read or written on the caller's stream; the array must
    stay alive and unmodified until that work completes.
    """

    def __dlpack__(self, *, stream: int | None = ...) -> Any: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...

# From _config.py
DataRole = NewType("DataRole", str)

class CacheTierConfig(Protocol):
    quota: int
    @property
    def tier(self) -> CacheTier: ...
    def validate(self) -> None: ...

@dataclass(slots=True)
class GpuCacheTierConfig:
    quota: int
    @property
    def tier(self) -> CacheTier: ...
    def validate(self) -> None: ...

@dataclass(slots=True)
class HostCacheTierConfig:
    quota: int
    max_quota: int | None = None
    allow_remote_numa_fallback: bool = True
    @property
    def tier(self) -> CacheTier: ...
    def validate(self) -> None: ...

@dataclass(slots=True)
class DiskCacheTierConfig:
    quota: int
    path: str
    @property
    def tier(self) -> CacheTier: ...
    def validate(self) -> None: ...

@dataclass(slots=True, frozen=True)
class BufferConfig:
    """One buffer of a layer's KV cache.

    ``reuse_policy`` of ``None`` uses the layer's policy. Attention and SSM
    buffers can override the layer's policy; the buffer then belongs to the
    layer group for its effective policy and lifecycle properties while keeping
    its layer ID, so one layer can span several layer groups.

    ``is_sparse`` marks a buffer whose history is too large for device memory and is read
    through a per-request device cache rather than directly (see
    :meth:`KVCacheManager.is_sparse`).

    It changes the *medium* its pages are made of, not where they lock. Every buffer locks
    its pages at the hot level; a sparse buffer's hot pool is host-backed, device-mapped so
    a kernel can read it. Its pages are otherwise ordinary hot pages -- coalesced,
    addressed by page index, visible to the transceiver -- and the page-index table stays
    uniform. A sparse buffer places no requirement on the cold
    tiers, which may be host, disk, or absent.

    Backing is decided per pool, and a pool is uniformly sparse or uniformly dense because
    buffers group on ``(size, is_sparse)``, so one layer's buffers may land in both. A group
    holds whichever its buffers grouped into, and reports GPU memory either way. A pool
    group must hold at least one non-sparse pool, since slot counts are sized from device
    bytes; a wholly sparse pool group is refused by the ``KVCacheManager`` constructor.
    A wholly sparse layer is allowed when its group also contains dense buffers from
    another layer.

    The layer's :attr:`AttentionLayerConfig.sparse_cache` configures its fetch path,
    including selection width, cache capacity and unit size. Without it, sparse storage
    and locking still work, but the buffer has no GPU cache to read through.

    Device residency is a lease on the life cycle's arena owned by a :class:`Batch`.
    Each Batch row has one lease, held by its seated request. Both the unit being written
    and the selected history occupy cells in it. The selection is below the watermark;
    the single unit in flight is above it and cannot be evicted.

    A lease holds contiguous runs of buffer-sized cells, addressed by arithmetic from
    the buffer and entry index. :meth:`Batch.fetch_sparse_pages` fills selected units.
    ``sparse_unit_chunks``, the layer's ``sparse_cache.unit_tokens`` and the buffer's
    size determine where each unit's bytes are in a densely packed page.

    :meth:`Batch.flush_sparse_units` writes completed units back to their host pages
    before :meth:`_KVCache.resize` advances history or the request leaves its Batch.
    The flush leaves the cells cached; advancing history makes them evictable.

    Fixed at construction: the storage layout derived from it is built once.
    """

    role: DataRole
    size: int
    tokens_per_block_override: int | None = None
    is_sparse: bool = False
    coalescing_group: int = 0

    sparse_unit_chunks: int = 1
    """Pieces a selection unit is gathered from within *this buffer's* page.

    Per buffer, because it is this buffer's own dimension order:

    - **1** for a sequence-major page (``[token][head][dim]``), where a unit's tokens carry
      their heads with them and the whole unit is one contiguous range. Also 1 for any
      single-head page -- a compressed or pooled latent -- where the orders coincide.
    - **numHeads** for a head-major page (``[head][token][dim]``), where each head
      contributes its own run and a unit is that many strided pieces.

    Every sparse model in this repository today is one of the first two cases, so 1 is
    almost always right. The rest of the layout follows from this, the layer's
    ``sparse_cache.unit_tokens`` and the page being densely packed, so a padded page is not
    expressible.
    """
    reuse_policy: ReusePolicy | None = None

@dataclass(slots=True)
class SparseCacheConfig:
    """One layer's sparse fetch path.

    Every sparse buffer of a layer shares one resolve, one resident set and one recency
    order. Within each Batch row's lease, metadata is laid out ``[selection][entry]``.
    Layers with the same ``selection_id`` share that metadata while keeping separate
    data cells; otherwise each layer has its own selection.

    Its presence is what configures the fetch path. A sparse buffer in a layer without it
    still gets sparse storage and locking; it simply has no GPU cache to read through.

    Layers sharing a life cycle must agree on width, capacity and unit size, since they
    share one arena; their ``selection_id`` values may differ. Batch construction checks
    the rounded cache capacity and selection width against the resolve kernel's available
    shared memory on the current device.
    """

    num_selected_blocks: int
    """Selection width: units a sparse-attention step selects for this layer."""

    max_cached_units: int
    """Cells one buffer gets in a request's GPU cache, counting both populations: the
    history it caches and every unit still in flight.

    Two to four times ``num_selected_blocks`` is the usual starting point: a cache only as
    wide as the selection retains nothing between steps, and retention of units not
    selected this step is where the hit rate comes from. It must exceed
    ``num_selected_blocks``, since equality would leave no cell for the unit being written.
    """

    unit_tokens: int
    """Original tokens one selection unit spans.

    A whole page is ``tokens_per_block``, which a caller wanting that states rather than
    omits -- there is no unit size that suits a layer which did not name one.

    Must divide ``tokens_per_block``; the quotient is how many units a page holds.
    Distinct from ``BufferConfig.tokens_per_block_override``, which reinterprets a buffer's
    page for *storage*: this names the sparse cache's granularity and nothing else, so
    pages, page indices and index spaces all keep the manager's ``tokens_per_block``.
    Setting both on one buffer is rejected.
    """

    selection_id: int | None
    """Layers whose selection is one selection, named so they can say so.

    A model running one indexer across several layers hands them all the same units every
    step. Naming that lets those layers share one resident set and one resolve, where
    otherwise each repeats the other's answer against its own copy of it.

    ``None`` means the layer selects for itself, and is read as the layer's own id -- so a
    layer that names nothing never shares, and an explicit id that names such a layer is
    rejected. Layers that do share must agree on ``num_selected_blocks``,
    ``max_cached_units`` and ``unit_tokens``.
    """

class ReusePolicy(enum.IntEnum):
    REQUIRED = 0
    PRIVATE = 1
    OPTIONAL = 2

@dataclass(slots=True)
class AttentionLayerConfig:
    """One attention layer's buffers and its history rules.

    A sparse layer may also declare a sliding window, though the combination is unusual:
    sparsity already bounds how much history a step reads. Where both apply, the window
    bounds what is *selectable* as well as what is retained -- the indexer producing the
    selection applies the same window, so it never names a block the window has made stale.
    """

    layer_id: LayerId
    buffers: list[BufferConfig]
    sliding_window_size: int | None = None
    num_sink_tokens: int | None = None
    reuse_policy: ReusePolicy = ReusePolicy.REQUIRED

    sparse_cache: SparseCacheConfig | None = None
    """The sparse fetch path for this layer's sparse buffers, if it has one.

    Setting it on a layer with no sparse buffer to serve is rejected.
    """
    @property
    def window_size(self) -> int | None: ...

@dataclass(slots=True)
class SsmLayerConfig:
    """Recurrent buffers with a layer-level default reuse policy.

    A buffer's ``reuse_policy`` overrides the default for that buffer.
    OPTIONAL state is restored only when selected on first resume at the exact
    matched endpoint. The caller initializes or reconstructs all fresh state.
    """

    layer_id: LayerId
    buffers: list[BufferConfig]
    reuse_policy: ReusePolicy = ReusePolicy.REQUIRED

LayerConfig = AttentionLayerConfig | SsmLayerConfig

@dataclass(slots=True)
class KVCacheDesc:
    capacity: int
    history_length: int
    beam_width: int = 1
    prompt_length: int = 0

@dataclass(slots=True)
class BatchDesc:
    kv_caches: list[KVCacheDesc]
    system_prompt_length: int = 0

@dataclass(slots=True)
class SwaScratchReuseConfig:
    max_rewind_len: int = 0

@dataclass(slots=True)
class PoolRebalanceConfig:
    min_sampled_kv_caches: int = 2000
    cooldown_secs: float = 120.0
    target_ratio_update_interval: int = 100
    ratio_tolerance: float = 0.25
    moving_average_decay: float = 0.9999

@dataclass(slots=True)
class KVCacheManagerConfig:
    tokens_per_block: int
    cache_tiers: list[CacheTierConfig]
    layers: list[LayerConfig]
    max_util_for_resume: float = ...
    enable_partial_reuse: bool = True
    # Tokens trimmed off the tail of every prefix match; nonzero only for a pool
    # that also holds state reading that many tokens ahead of each position.
    reuse_match_backoff: int = 0
    constraints: list[BatchDesc] = ...
    typical_step: BatchDesc | None = None
    # One positive, normalized hot-tier byte-quota weight per layer group. Cold initialization preserves the implied
    # layer-group slot-count proportions.
    initial_pool_ratio: list[float] | None = None
    swa_scratch_reuse: SwaScratchReuseConfig | None = None
    commit_min_snapshot: bool = False
    enable_stats: bool = True
    text_only: bool = False
    enable_partial_commit: bool = True
    pool_rebalance: PoolRebalanceConfig = ...
    cold_tier_offload_min_priority: int = DEFAULT_COLD_TIER_OFFLOAD_MIN_PRIORITY
    # Host bytes the GPU tier's host-backed pools may take, or None for unbounded. A sparse
    # buffer's pages are host memory, so the device quota does not pay for them and nothing
    # else bounds them: a slot spans every pool of its group, so their row count follows from
    # the device-backed pools the quota does size. This only lowers a slot count, and pool
    # groups holding such pools share it by the ratio that divides the device quota.
    #
    # Kept exactly, as is the feasibility floor ``constraints`` implies; the pool ratio is what
    # gives way between them. The two can only conflict if the floors alone exceed this
    # budget, which construction rejects.
    host_sparse_pool_quota: int | None = None
    @property
    def enable_swa_scratch_reuse(self) -> bool: ...

# From _event_manager.py
EventBlockHash: TypeAlias = int | str
BlockHashLike: TypeAlias = bytes | EventBlockHash
BlockHashesLike: TypeAlias = BlockHashLike | Iterable[BlockHashLike]
EventTokenId: TypeAlias = int | str
MmKey: TypeAlias = (
    tuple[bytes, int] | tuple[bytes, int, str | None] | tuple[bytes, int, str | None, bool]
)
AttentionDpGatherFn: TypeAlias = Callable[[list["KVCacheEvent"]], list[list["KVCacheEvent"]]]

@dataclass(slots=True, frozen=True)
class UniqueToken:
    token_id: EventTokenId
    token_extra_id: int = ...

@dataclass(slots=True, frozen=True)
class KVCacheCreatedData:
    num_blocks_per_cache_level: list[int]

@dataclass(slots=True, frozen=True)
class KVCacheStoredBlockData:
    block_hash: EventBlockHash
    tokens: list[UniqueToken]
    cache_level: int
    priority: int
    mm_keys: list[MmKey] = ...
    cache_salt: str | None = ...

@dataclass(slots=True, frozen=True)
class KVCacheStoredData:
    parent_hash: EventBlockHash | None
    blocks: list[KVCacheStoredBlockData]

@dataclass(slots=True, frozen=True)
class KVCacheRemovedData:
    block_hashes: list[EventBlockHash]

@dataclass(slots=True, frozen=True)
class KVCacheEventDiff:
    old_value: int
    new_value: int

@dataclass(slots=True, frozen=True)
class KVCacheUpdatedData:
    block_hash: EventBlockHash
    cache_level: KVCacheEventDiff | None
    priority: KVCacheEventDiff | None

@dataclass(slots=True, frozen=True)
class KVCacheEvent:
    event_id: int
    data: KVCacheCreatedData | KVCacheStoredData | KVCacheRemovedData | KVCacheUpdatedData
    window_size: int
    hash_algo: str | None = None
    attention_dp_rank: int | None = None
    layer_group_id: int | None = None

class StreamingBlockStoredData:
    @property
    def lora_id(self) -> int | None: ...
    @property
    def block_hashes(self) -> list[int]: ...
    @property
    def parent_block_hash(self) -> int | None: ...
    @property
    def token_ids(self) -> list[EventTokenId]: ...
    @property
    def mm_keys(self) -> list[list[MmKey]]: ...

class StreamingBlockRemovedData:
    @property
    def block_hashes(self) -> list[int]: ...

class StreamingAllBlocksClearedData: ...

class StreamingEventStats:
    @property
    def stored_blocks(self) -> int: ...
    @property
    def removed_blocks(self) -> int: ...
    @property
    def partial_blocks_suppressed(self) -> int: ...
    @property
    def non_target_life_cycles_ignored(self) -> int: ...
    @property
    def dropped_events(self) -> int: ...

class StreamingEventSink:
    def __init__(
        self,
        max_entries: int = ...,
        mm_token_id_offset: int | None = None,
    ) -> None: ...
    def set_target_life_cycle(self, life_cycle_id: int) -> None: ...
    def drain_iteration_events(
        self,
    ) -> list[
        StreamingBlockStoredData | StreamingBlockRemovedData | StreamingAllBlocksClearedData
    ]: ...
    @property
    def stats(self) -> StreamingEventStats: ...

class KVCacheEventManager:
    @staticmethod
    def _hash_cache_salt(salt: str) -> int: ...
    def __init__(
        self,
        max_kv_event_entries: int,
        *,
        window_size: int = ...,
        attention_dp_rank: int | None = None,
        attention_dp_gather: AttentionDpGatherFn | None = None,
        hash_algo: str = ...,
        window_size_by_layer_group: dict[int, int] | None = None,
        mm_token_id_offset: int | None = None,
    ) -> None: ...
    def add_created_event(
        self,
        num_blocks_per_cache_level: Sequence[int],
        layer_group_ids: Sequence[int] | None = None,
    ) -> None: ...
    def set_layer_group_window_sizes(self, window_sizes: dict[int, int]) -> None: ...
    def add_stored_event(
        self,
        parent_hash: EventBlockHash | None,
        blocks: Sequence[KVCacheStoredBlockData],
        layer_group_id: int | None = None,
    ) -> None: ...
    def add_stored_block_event_from_block(self, block: Any) -> None: ...
    def add_stored_life_cycle_event_from_block(self, block: Any, life_cycle_id: int) -> None: ...
    def add_removed_event(self, block_hashes: BlockHashesLike) -> None: ...
    def add_removed_life_cycle_event(self, block_hash: bytes, life_cycle_id: int) -> None: ...
    def add_updated_event(
        self,
        block_hash: BlockHashLike,
        *,
        cache_level: KVCacheEventDiff | None = None,
        priority: KVCacheEventDiff | None = None,
        layer_group_id: int | None = None,
    ) -> None: ...
    def flush_iteration_events(self) -> None: ...
    def get_latest_events(self, timeout_ms: float | None = None) -> list[KVCacheEvent]: ...

# Native key builders, shared with the radix tree so routing hashes match the engine's.
def gen_multimodal_cache_key_tokens(
    id_offset: int,
    multi_modal_data_digest: bytes,
    num_tokens: int,
    token_offset: int = 0,
    uuid: str | None = None,
) -> list[TokenIdExt]: ...
def sequence_to_blockchain_keys(
    tokens_per_block: int,
    reuse_scope: ReuseScope,
    tokens: Sequence[TokenIdExt],
) -> Iterator[tuple[list[TokenIdExt], bytes]]: ...

# From _core/_kv_cache.py
class _Status(enum.Enum):
    ACTIVE = enum.auto()
    SUSPENDED = enum.auto()
    CLOSED = enum.auto()

KvCacheStatus: TypeAlias = _Status

IndexSeq = array.array[int] | memoryview[int]

class ReuseGroupStatus:
    """Group availability at the initial prefix claim, not subsequent request progress."""

    # Lifecycle group from get_layer_group_id(layer_id), not a physical pool index.
    @property
    def group_id(self) -> LayerGroupId: ...
    # REQUIRED constrains the match; OPTIONAL is selected at resume; PRIVATE is never reused.
    @property
    def policy(self) -> ReusePolicy: ...
    # Exclusive token endpoint claimed by REQUIRED groups, shared by all groups.
    @property
    def endpoint(self) -> int: ...
    # All needed state is available, not necessarily on GPU or selected for reuse.
    # False for an empty prefix, PRIVATE groups, or missing/discarded OPTIONAL candidates.
    @property
    def complete(self) -> bool: ...
    # Retained [begin, end) ranges, including block-rounded SWA windows and sinks.
    # SSM uses (endpoint, endpoint) for its exact state; empty when complete is false.
    @property
    def coverage(self) -> Sequence[tuple[int, int]]: ...

class _KVCache:
    Status: ClassVar[Type[_Status]]
    id: Any
    def __init__(
        self,
        manager: "KVCacheManager",
        reuse_scope: ReuseScope,
        reuse_match: Any | None,
        id: Any,
        custom_priority_callback: Callable[[int, Any], Priority],
        expected_prompt_length: int | None = None,
        text_only: bool | None = None,
        enable_request_stats: bool = False,
    ) -> None: ...
    def set_base_page_index_buf(
        self, beam_idx: BeamIndex, layer_group_id: LayerGroupId, buf: memoryview | None
    ) -> None:
        """Write page indices through ``buf`` instead of this cache's own storage.

        Raises while this cache belongs to a :class:`Batch`: membership already binds every
        layer group to a row of the batch's table, and a second buffer would take the writes
        with it, leaving that row holding whatever it held when the cache joined. Remove it
        from the batch first, or give the batch the buffer worth writing to.

        This holds for the batch too -- it ends the membership before handing the indices
        back, so nothing binds a seated member's buffer.
        """
    @property
    def manager(self) -> "KVCacheManager": ...
    @property
    def cuda_stream(self) -> CudaStream: ...
    @cuda_stream.setter
    def cuda_stream(self, cuda_stream: CudaStream) -> None: ...
    @property
    def num_blocks(self) -> int: ...
    @property
    def batch_row(self) -> BatchRow | None:
        """Row this cache occupies in its :class:`Batch`, or ``None`` when it belongs to none.

        Also the answer to who owns its page index buffers: a member's are the batch's
        rows, so :meth:`set_base_page_index_buf` refuses and ``suspend()`` leaves them
        bound rather than copying them back out.
        """
    def commit_pending_stats(self) -> KVCacheStatsDelta: ...
    def discard_pending_stats(self) -> None: ...
    def record_connector_reuse(self, token_begin: int, token_end: int) -> None: ...
    def close(self) -> None: ...
    @property
    def beam_width(self) -> BeamIndex: ...
    @beam_width.setter
    def beam_width(self, beam_width: BeamIndex) -> None:
        """Expand before the first generation step, never during generation (C++ only).

        First resume the cache and materialize prompt storage (or prepare synthetic
        warmup state). Full prompt blocks are shared; the writable tail, including
        preallocated blocks, is copied using the boundary set by
        ``expected_prompt_length`` at cache creation.
        """
        ...
    def get_base_page_indices(
        self, layer_group_id: LayerGroupId, beam_id: BeamIndex = DEFAULT_BEAM_INDEX
    ) -> IndexSeq: ...
    def get_aggregated_page_indices(
        self,
        layer_group_id: LayerGroupId,
        beam_id: BeamIndex = DEFAULT_BEAM_INDEX,
        valid_only: bool = False,
    ) -> Iterator[int]: ...
    def resize(self, capacity: int | None, history_length: int | None = None) -> bool: ...
    @property
    def max_input_length(self) -> int | None:
        """Bound on ``capacity - history_length``: how far capacity may outrun the watermark.

        ``None`` unless a sparse buffer's fetch path is configured, since a bound on
        in-flight blocks describes nothing without a cache to hold them. :meth:`resize`
        rejects a gap wider than this.

        **One token**, and read-only, because it is a consequence of the supported shape
        rather than of how wide a cache is. A sparse request contributes exactly one query
        token per step: :meth:`Batch.fetch_sparse_pages` resolves one selection
        per request, while an attention kernel indexes its selection per query token, and
        the two agree only at one token each. Prefill and speculative decoding give a
        request several query tokens per step and are out of scope; this bound is the guard
        that keeps them out.

        It follows that at most one unit is ever in flight, so essentially all of
        :attr:`SparseCacheConfig.max_cached_units` is available to the selection rather than
        being reserved as headroom.
        """

    @property
    def capacity(self) -> int: ...
    @capacity.setter
    def capacity(self, capacity: int) -> None: ...
    @property
    def history_length(self) -> int: ...
    @history_length.setter
    def history_length(self, history_length: int) -> None: ...
    def commit(
        self,
        accepted_input_tokens: Sequence[TokenIdExt],
        beam_search_indices: Sequence[int] | None = None,
        is_end: bool = False,
    ) -> None: ...
    @property
    def num_committed_tokens(self) -> int: ...
    @property
    def cached_tokens_by_level(self) -> list[int]: ...
    def _get_last_cached_token_level(self) -> int | None: ...
    @property
    def reuse_status(self) -> Sequence[ReuseGroupStatus]: ...
    @property
    def committed_tokens(self) -> list[TokenIdExt]: ...
    @property
    def reuse_scope(self) -> ReuseScope: ...
    def plan_committed_block_drop(self) -> PlannedDropHandle | None: ...
    def stop_committing(self) -> None: ...
    def suspend(self) -> None:
        """Unlock this cache's pages and detach from its stream.

        Raises while it belongs to a :class:`Batch`: the unlocks write ``BAD_PAGE_INDEX``
        through the bound buffer, which is the batch's row, so doing it while seated would
        write a row a publish may still be reading. Remove it from the batch first.
        """

    def resume(
        self,
        cuda_stream: CudaStream | None = None,
        is_decoding: bool | None = None,
        optional_reuse_groups: Sequence[LayerGroupId] | None = None,
    ) -> bool:
        """Lock this cache's pages at their required levels.

        Returns ``False`` when utilisation is too high or pages cannot be had. Raises while
        it belongs to a :class:`Batch`, for the same reason as :meth:`suspend` -- locking
        writes page indices through the batch's row. A member is resumed before it joins.

        Takes no sparse cache arena lease: that follows membership, so a request receiving
        its history costs the GPU cache nothing until it is seated.

        ``is_decoding`` is accepted and ignored. Resuming no longer distinguishes a decoding
        request; the parameter is kept so callers that still pass it keep working, and goes
        with the full sparse attention support.
        """
    def prefetch(self, target: CacheLevel) -> bool: ...
    def get_swa_scratch_range(self, layer_group_id: LayerGroupId) -> HalfOpenRange:
        """Block ordinal range this life cycle serves from scratch slots.

        Empty when none of its blocks do. Pulled per step rather than mirrored: it is two
        integers, and keeping it pulled is what stops a range and its slots from being
        updated at different moments.
        """

    def get_scratch_slots(self, layer_group_id: LayerGroupId) -> list[int]:
        """Page indices of this life cycle's scratch slots, in slot order."""
    @property
    def has_scratch_slots(self) -> bool: ...
    @property
    def enable_swa_scratch_reuse(self) -> bool: ...
    @enable_swa_scratch_reuse.setter
    def enable_swa_scratch_reuse(self, enable: bool) -> None: ...
    @property
    def text_only(self) -> bool: ...
    @text_only.setter
    def text_only(self, text_only: bool) -> None: ...
    # Per life cycle, which is where the scratch slots live. The one-argument form
    # collapses them, so one life cycle's scratch makes it answer False for every other.
    @overload
    def supports_index_mode(self, mode: PageIndexMode) -> bool: ...
    @overload
    def supports_index_mode(self, mode: PageIndexMode, layer_group_id: LayerGroupId) -> bool: ...
    @property
    def status(self) -> _Status: ...
    @property
    def is_active(self) -> bool: ...
    @property
    def tokens_per_block(self) -> int: ...

@dataclass(slots=True, frozen=True)
class PoolDesc:
    pool_index: PoolIndex
    base_address: MemAddress
    slot_bytes: int

    storage_medium: CacheTier
    """What this pool's pages actually are.

    The same as the owning group's tier, except for a sparse pool of the GPU tier: those
    are host-backed, and the group still reports ``GPU_MEM`` because its other pools are
    not. Ask per pool rather than per group wherever the medium of an address decides
    something -- registering it for a disaggregated transfer, or choosing a copy
    implementation -- since a group is not uniform in this.
    """

class BufferId(NamedTuple):
    layer_id: LayerId
    role: DataRole

@dataclass(slots=True, frozen=True)
class ExpandedBuffer:
    id: BufferId
    expansion: int  # expansion factor of page due to heterogeneous tokens_per_block

@dataclass(slots=True, frozen=True)
class AggregatedPageDesc:
    """The data you need would be in the following byte ranges.

    (base + stride * i + Range(0, size) for i in aggregated_page_indices)
    """

    base: MemAddress
    size: int
    stride: int
    layer_group_id: LayerGroupId
    buffers: Sequence[ExpandedBuffer]

@dataclass(slots=True, frozen=True)
class CoalescedBuffer:
    single_buffer_size: int
    buffer_ids: Sequence[BufferId]
    @property
    def size(self) -> int: ...
    @property
    def num_buffers(self) -> int: ...

@dataclass(slots=True, frozen=True)
class SlotDescVariant:
    coalesced_buffers: Sequence[CoalescedBuffer]
    @property
    def layer_group_id(self) -> LayerGroupId: ...
    @property
    def slot_size_list(self) -> Sequence[int]: ...

@dataclass(slots=True, frozen=True)
class SlotDesc:
    variants: Sequence[SlotDescVariant]
    @property
    def slot_size_list(self) -> Sequence[int]: ...

@dataclass(slots=True, frozen=True)
class PoolGroupDesc:
    pool_group_index: PoolGroupIndex
    num_slots: int
    slot_desc: SlotDesc
    pools: Sequence[PoolDesc]

# From _core/_kv_cache_manager.py
class HalfOpenRange:
    def __init__(self, beg: int, end: int) -> None: ...
    @property
    def beg(self) -> int: ...
    @property
    def end(self) -> int: ...
    def __bool__(self) -> bool: ...
    def __len__(self) -> int: ...
    def __eq__(self, other: object) -> bool: ...

class ScratchDesc:
    range: HalfOpenRange
    slot_ids: Sequence[int]
    def __bool__(self) -> bool: ...

@dataclass(slots=True, frozen=True)
class PageIndexConverter:
    scale: int
    expansion: int
    layer_offset: int
    scratch_pages_per_block: int

    def __call__(
        self,
        base_indices: Sequence[int],
        index_mode: PageIndexMode | None = None,
        scratch_range: HalfOpenRange | ScratchDesc | None = None,
        scratch_slots: Sequence[int] = (),
    ) -> list[int]: ...

class IKvCacheColdPageCodec: ...

def create_default_kv_cache_cold_page_codec() -> IKvCacheColdPageCodec:
    """Create the default lossless cold-page codec.

    Passing ``cold_page_codec=None`` to ``KVCacheManager`` already selects this codec, so normal users do not need to
    call this factory. It is primarily provided to demonstrate how a native codec factory exposes an owning
    ``IKvCacheColdPageCodec`` object for transfer into ``KVCacheManager``. Any ``KVCacheManager`` construction attempt
    consumes an explicitly supplied codec, including an attempt that fails.
    """

class IndexTableSpec:
    """Buffers a consumer must address from one table, under one base pointer.

    States a kernel's requirement, not a preference: some attention ops take a single block
    table covering K and V rather than one each. Order defines the table's buffer dimension,
    so a kernel reads buffer ``i`` of this spec at entry ``i``.

    Every buffer must share a life cycle and a page-index scale, since no one table can
    serve buffers whose converted indices differ. A spec spanning either is refused when
    the ``Batch`` is constructed.
    """

    def __init__(self, buffers: Sequence[BufferId]) -> None: ...
    @property
    def buffers(self) -> list[BufferId]: ...

class Batch:
    """The ``_KVCache`` objects stepped together, and their device-side mirror.

    Owns the batch row order and every per-row device array the sparse path reads: a page
    table per layer group, the eligible-block counts, and the staging buffers behind
    them. Every such array is indexed by row, and each must agree on which row a request
    occupies, so the order lives here rather than being re-established per call.

    Rows are slots rather than positions in a compacted list: removing a member leaves a
    hole for reuse instead of shifting later rows, keeping array shapes and addresses
    stable across steps, which is what CUDA graph capture requires.

    Membership is exclusive -- a ``_KVCache`` belongs to at most one ``Batch``.

    :meth:`publish` restages every row. Mutating a member directly therefore costs a
    republish rather than producing a stale table; refreshing only the rows whose locks
    changed is a planned refinement, and a cost rather than a correctness question.
    """

    def __init__(
        self,
        manager: KVCacheManager,
        stream: CudaStream,
        index_tables: Sequence[IndexTableSpec],
        max_batch_size: int | None = None,
        max_blocks_per_seq: int | None = None,
        mode: PageIndexMode | None = None,
        max_total_input_tokens: int | None = None,
    ) -> None:
        """Bound the device arrays, whose shape must not change between steps.

        ``index_tables`` names what a consumer needs addressable together, and is
        required: ``[]`` is how a caller that converts no indices says so, which is a
        choice about what the consumer reads rather than an absence the batch could fill
        in.

        ``max_total_input_tokens`` is the tokens one step may ingest across every row. It
        sizes the write-back's work list: a row advancing ``t`` tokens crosses at most
        ``t / unit_tokens + 1`` units, so the batch crosses at most
        ``max_total_input_tokens / unit_tokens + max_batch_size`` of them. Left unset it
        assumes a decode step, where each row advances one token and so crosses at most
        one unit boundary.

        ``stream`` is the stream every member runs on. A batch owns it rather than taking
        one per call: a member's own work and the batch's transfers have to be ordered
        against each other, and two streams would leave that to the caller. Joining binds
        the member to it, and a member cannot be moved off it.

        Neither can be grown later, so neither is inferred: left unset each derives from
        the manager's ``constraints``, which fails when none were declared. A pool-sizing
        hint is a default, never a substitute for a batch bound.

        The batch stages the host page indices itself; :meth:`coalesced_page_table` exposes
        them for a consumer that needs the same numbers. One buffer with one writer is what
        keeps a member's binding alive for the whole decode, and which side allocates it
        never mattered to that.

        ``mode`` applies to every life cycle. Left unset, each takes the smaller table:
        ``SHARED`` where :meth:`KVCacheManager.supports_index_mode` reports it always
        available, ``PER_LAYER`` otherwise.

        Set, it **overrides** that answer rather than being checked against it, so
        ``SHARED`` may be named where the manager returns ``None``. The manager answers for
        every request a life cycle could carry, so a sliding window leaves it undecided;
        naming ``SHARED`` is the narrower claim that no member of *this* batch will hold
        scratch slots there. A decode-only batch is the case it exists for: scratch covers
        the blocks a prefill chunk adds, and a one-token request adds none. That is what
        lets prefill and decode run as two batches over one manager with SWA scratch
        enabled, the decode one still taking ``SHARED`` for the sliding life cycles -- where
        the saving is worth having, since that batch holds the most rows.

        The claim is validated rather than trusted. Every :meth:`publish` checks each member
        against its own :meth:`_KVCache.supports_index_mode` and raises if one holds scratch
        after all -- mid-step, with the batch already built around the smaller table. Seat
        only members that keep the claim.

        It is fixed here because a converted table's shape is baked into a captured graph,
        so it cannot follow a member's scratch state, which varies per step.
        """

    def add_kv_cache(self, kv_cache: _KVCache, row: BatchRow | None = None) -> BatchRow:
        """Seat a cache in ``row``, or in the lowest free row when it is ``None``.

        Returns the row. Raises if the row is taken or out of range, if the batch is full,
        if the cache already belongs to a batch, if it has not been resumed, or if its beam
        width exceeds one -- only beam 0 is mirrored, so a wider request would have every
        beam read beam 0's pages.

        Takes the sparse cache arena lease, because membership is what being stepped means:
        a resumed request that is not seated cannot fetch, and a disaggregated one receives
        its whole history into host pages without touching the cache. The arena has one
        lease per Batch row, so a free row always has a free lease and this cannot fail
        for want of one.

        So seat a request only once it is ready to be stepped: one still receiving its
        disaggregated history, or sitting a step out, costs the arena nothing while it stays
        out, which is why the arena is sized for the rows rather than for every request the
        scheduler admits. Nothing checks it -- an early seat shows up as arena pressure.
        """

    def remove_kv_cache(self, kv_cache: _KVCache) -> _KVCache:
        """Free the cache's row, leaving a hole for reuse, and return the cache.

        The batch held a reference while the cache was seated; this hands it back, so a
        caller that kept only the batch still gets the cache. Raises when it is not a
        member.

        Returns the arena lease, which **discards whatever the GPU cache held**: a lease
        comes back with every entry free, so a unit whose only copy was in a cell can no
        longer be addressed. Call :meth:`flush_sparse_units` first for anything worth
        keeping. A batch is meant to persist across steps -- churning membership per step
        cold-starts every request's cache.
        """

    def remove_row(self, row: BatchRow) -> _KVCache | None:
        """The same, by the row :meth:`add_kv_cache` returned.

        ``None`` when nothing is seated there, so a caller sweeping rows need not ask twice
        about each one.
        """

    def move(self, dst_row: BatchRow, src_row: BatchRow) -> None:
        """Move the member at ``src_row`` to the empty ``dst_row``, leaving ``src_row`` free.

        Membership does not lapse: the sparse cache lease, its contents and the stream
        binding all survive, which is what separates this from :meth:`remove_row` followed
        by :meth:`add_kv_cache` -- that pair hands the lease back with every entry free. It
        also writes each life cycle's index-table row once instead of twice.

        Raises when either row is out of range, ``src_row`` holds nothing, or ``dst_row`` is
        occupied. Waits for the previous staging upload. Device tables keep their addresses
        and reach the new row on the next ``publish``, so a step may resize, move and publish
        once rather than publishing per operation.
        """

    @property
    def num_index_tables(self) -> int: ...
    def index_table(self, table_id: int) -> list[IndexTableSpec]:
        """Every request this table satisfies.

        A request's buffer order is its own mapping into the buffer dimension, so the first
        buffer of each sits at entry zero. Requests whose converted indices coincide are
        merged onto one table, which is how a quantised layer's data and block scales come
        to share one.
        """

    def index_table_id(self, spec: IndexTableSpec) -> int:
        """Which table serves this spec, which must be one the ``Batch`` was built with.

        The query to prefer: a caller that asked for a table by naming its buffers gets it
        back without going through a buffer, and equivalent specs resolve to the same id --
        a layer's data and its block scales name one table between them. The spec's buffers
        occupy entries ``[0, len(spec.buffers))`` in its own order, so an entry needs no
        separate lookup.

        Specs may not overlap, so a buffer belongs to exactly one. Read a subset of a table
        by slicing it rather than by declaring the subset as a spec of its own.
        """

    def pool_spec_for(self, spec: IndexTableSpec) -> tuple[MemAddress, int]:
        """``(base, page_stride)`` for this spec's converted indices.

        A page index ``p`` of this spec addresses ``base + p * page_stride``.

        Keyed by the spec because the base is. Specs whose converted entries coincide are
        merged onto one table, and reading that one set of entries through different bases
        is what keeps a quantised layer's data and its block scales apart -- so a table has
        as many bases as it serves specs, and only the spec picks one. The spec's buffers
        share a page stride, which is what lets one base address them all.
        """

    def converted_index_table(self, table_id: int) -> DeviceArray:
        """``int32[max_batch, entries, max_blocks_per_seq]`` converted indices for one table.

        A row carries every entry before the next row starts, which is the shape an
        attention kernel takes a key and value page list in. One entry's plane is therefore
        strided rather than contiguous: ``table[:, entry, :]``.

        A DLPack array over memory the ``Batch`` owns; it keeps the ``Batch`` alive for as
        long as the array exists. A buffer reads the entry its position in the spec gives
        it: a spec's buffers occupy entries ``[0, len(spec.buffers))`` in the spec's own
        order, so a caller holding the spec needs no further lookup.

        The address is stable for the ``Batch``'s lifetime, so a captured graph may bake it.
        """

    def index_mode(self, layer_group_id: LayerGroupId) -> PageIndexMode:
        """Page index mode of one life cycle, as chosen at construction.

        Left unset at construction, each life cycle takes the smaller table: ``SHARED``
        where the manager reports it always available, ``PER_LAYER`` otherwise. Fixed here
        because a converted table's shape is baked into a captured graph, so it cannot
        follow a member's scratch state.
        """

    def row_of(self, kv_cache: _KVCache) -> BatchRow | None:
        """Row of a member, or ``None`` when it is not seated."""

    def __getitem__(self, row: BatchRow) -> _KVCache | None:
        """The member seated in ``row``, or ``None`` when that row is free.

        The inverse of :meth:`row_of`. Raises ``IndexError`` when ``row`` is beyond
        ``max_batch_size``, so ``None`` always means an empty row rather than a bad index.
        """

    @property
    def num_kv_caches(self) -> int:
        """Members seated, which is the rows occupied.

        Also the leases held: seating takes one from every life cycle's arena and leaving
        returns it, so the free leases are the arena's capacity less this.
        """

    def peek_hit_miss_counter(self) -> tuple[int, int]:
        """Cumulative ``(hits, misses)`` in selection units, since build or the last reset.

        A hit is a selected unit the GPU cache already held; a miss is one the fetch had to
        copy from its host page. Padded selection entries and in-flight units are neither,
        so they appear in neither figure -- which is why the pair comes from the resolve
        rather than from a count the caller could keep itself.

        Reads on a temporary stream and synchronises only that, so the pair may be a few
        launches stale. That does not move a ratio; synchronise the device first if an exact
        pair matters.
        """

    def reset_hit_miss_counter(self) -> None:
        """Zero the cumulative pair, enqueued on the Batch's stream."""

    @property
    def sparse_arena_bytes(self) -> int:
        """Device bytes every life cycle's sparse arena holds, ``0`` without sparse offload.

        The arenas sit outside the cache tiers' quota, so a caller budgeting device memory
        accounts for them separately and otherwise has to predict the layout to do so. This
        reports what was actually allocated, which makes that prediction checkable rather
        than something only a profiler disagrees with.
        """

    @property
    def is_contiguous(self) -> bool:
        """Whether the occupied rows are exactly ``[0, num_kv_caches)``.

        A batch does not require it: removing a member leaves its row free and the next
        join takes the lowest free one, so a hole is legal and transient. An engine that
        maps rows onto its own request slots usually does require it, and asking here beats
        keeping a mirror of the membership to check against.
        """

    @property
    def cuda_stream(self) -> CudaStream:
        """The stream this batch and every member run on.

        Assigning moves the batch and every member onto the new stream, which first waits
        on what the old one has queued, so no explicit synchronization is needed. A member
        cannot be moved on its own: its own setter refuses while it is in a batch.
        """

    @cuda_stream.setter
    def cuda_stream(self, cuda_stream: CudaStream) -> None: ...
    @property
    def max_batch_size(self) -> int: ...
    def get_sparse_cache_base_address(self, layer_id: LayerId, data_role: DataRole) -> MemAddress:
        """Base of the sparse cache arena backing this buffer. Raises if it has no fetch path.

        The arena is this Batch's: it is sized by :attr:`max_batch_size`, one lease per
        row, so the address is meaningful only for caches seated here.

        The indices from :meth:`fetch_sparse_pages` and :meth:`prepare_sparse_write_indices`
        are relative to this base. The arena holds the GPU-resident units that model kernels
        read and write.

        Distinct from :meth:`KVCacheManager.get_mem_pool_base_address`, which names this
        buffer's *pages* for every buffer alike -- what the transceiver and the KV transfer
        path address.
        That meaning does not change when a buffer is sparse; the arena is a second
        allocation, so it gets a second accessor rather than overloading the first.

        Role-dependent: the arena mirrors the life cycle's hot pool group, so buffers of
        different sizes -- data and its block scales, say -- sit in regions of their own, and
        an index is relative to the region of the buffer it was fetched for. Buffers sharing
        a pool share a region, and the index already carries which of them it belongs to.
        """
    @property
    def max_blocks_per_seq(self) -> int: ...
    def resize(
        self,
        capacities: Sequence[int | None],
        history_lengths: Sequence[int | None],
    ) -> list[int]:
        """Resize every member, then publish once.

        One result per row, in row order, ``0`` for a free row: a member rolls back on its
        own, so a batch can partially fail. A table published after a partial failure
        describes the post-rollback state.

        Rejects, before touching any member, a history length that falls or that outruns
        the capacity its row already had -- capacity granted by this call makes room for
        tokens written *after* it, so history can only cover tokens whose space was already
        there. A request arriving with a whole prompt as history is the one shape that
        excludes, so seed it before it joins.

        A row whose growth fails still takes its history advance, which needs no slots and
        is what lets the units below it be written back; its result stays ``0``, since the
        row is not at the state that was asked for.
        """

    def flush_sparse_units(self, new_history_lengths: Sequence[int]) -> None:
        """Write back the units each row's watermark is about to pass.

        One entry per row, giving the history you are about to set; a row may repeat its
        current history, which asks for nothing. **Call this before advancing history, not
        after**: a block must be in its host page before it can be selected, and
        :meth:`_KVCache.resize` refuses a history this has not covered.

        Bounded by the capacity each row started from, not by how far history travels, so
        a resumed request moving history across a whole prompt writes back nothing -- its
        units reached their host pages directly rather than through a cell. A move wider
        than the in-flight headroom is refused.

        Reads the index table as of the last :meth:`publish`, which covers the blocks in
        range because they were allocated by an earlier step.
        """

    def prepare_sparse_write_indices(
        self,
        buffers: Sequence[BufferId],
        out: DeviceArray,  # int32[batch, len(buffers)]
        stream: CudaStream | None = None,
    ) -> None:
        """Give each request's in-flight unit a cell, and report where to write it.

        The unit between the two watermarks is the one the model is about to produce. It
        has no host copy yet -- a unit reaches its page only when :meth:`_KVCache.resize`
        carries the watermark past it -- so nothing is fetched here. The call assigns a
        cell and leaves its bytes to the model; reading one before the model writes it
        yields whatever the cell last held.

        Separate from :meth:`fetch_sparse_pages` because the orders differ. A fetch needs
        the selection, which the indexer produces from the previous layer's output; a write
        needs only the watermarks. So this can run first, which is the order the model
        requires, and it needs no top-k.

        Idempotent while the watermarks and the reservation hold. The resolve probes
        residency before taking a victim, so a later call -- including the one inside
        :meth:`fetch_sparse_pages` -- finds the entry and reuses it rather than assigning a
        second. The entry is pinned for as long as its unit is in flight, so no fetch can
        evict it, and a fetch never copies over it: a unit at or above the watermark has no
        written-back host page, so there is nothing it could be refilled from.

        The Batch's published rows own the arena leases; each seated cache uses its row's
        lease until it leaves the Batch.

        Args:
            buffers: exactly one layer's sparse buffers, in the cache's order -- the same
                rule and the same order as :meth:`fetch_sparse_pages`.
            out: int32 ``[batch.max_batch_size, len(buffers)]``. ``out[b][s]`` is the index
                of the in-flight unit's cell, and ``BAD_PAGE_INDEX`` where the row holds no
                reservation or has nothing in flight. ``s`` is the buffer's output slice
                from :meth:`sparse_out_slots`, the same assignment a fetch uses: buffers
                that share a slice are written once, and unused slices are left untouched.
                One entry per request, not per query token: :meth:`_KVCache.resize`
                bounds ``capacity - history_length`` to a single token, so at most one unit
                is ever in flight.

                ``BAD_PAGE_INDEX`` is an ordinary outcome, not an error. A unit is in flight
                only once it is whole: with ``SparseCacheConfig.unit_tokens`` of 4, that is one
                step in four, which is also the only step on which a ratio-4 compressor has
                produced anything to write. With one token per unit it is every step.
            stream: the CUDA stream to enqueue on.

        Indices are in the same expanded index space :meth:`fetch_sparse_pages` produces
        and relative to the same base -- :meth:`get_sparse_cache_base_address` for that
        buffer, **not** :meth:`get_mem_pool_base_address`. The latter names the buffer's
        host-backed *pages*; the cells live in a separate GPU arena.

        Call it after publication and before the model's KV writes, ordering the two
        through the stream. Regenerate the output whenever a watermark or a reservation
        changes, and keep it unmodified until its consumers are done. Complete the writes
        before :meth:`flush_sparse_units` writes a finished unit back to its host page,
        then advance history through :meth:`_KVCache.resize`.

        Device-driven and stream-ordered, with static shapes, so a CUDA graph can capture
        it.


        Order it after batch publication and before consumers, on the same stream or
        through events across streams. The call reads no device values on the host, and
        with fixed batch and output shapes it is CUDA-graph-capturable.

        The entry is shared by all of the layer's sparse buffers, but each buffer's index
        is relative to its own :meth:`get_sparse_cache_base_address`, including when data
        and scales sit in different pools. The indices carry request and buffer placement;
        they are neither request-local cache-entry numbers nor byte addresses.
        """
    def fetch_sparse_pages(
        self,
        buffers: Sequence[BufferId],
        selected: DeviceArray,  # int32[batch, topk]
        out: DeviceArray,  # int32[batch, len(buffers), topk]
        stream: CudaStream | None = None,
    ) -> None:
        """Resolve a batch's selections against each buffer's per-request history cache.

        Device-driven and stream-ordered throughout: the selection is read from device
        memory, so the host neither inspects it nor decides anything, and the call
        enqueues onto ``stream`` and returns. Order it against the selection kernel by
        sharing a stream, or across streams with an event. Every shape is static across
        steps, so the call is capturable in a CUDA graph.

        ``stream`` defaults to the batch's own, which is the only correct choice while
        layers are fetched in order, since a layer's selection comes from the one before
        it. Name a stream only to run ahead of that -- a model that fetches several layers
        early can do so on its own -- and then order it against the batch's stream
        yourself: the resolve's scratch and counters are batch state the step clears on
        that stream.

        ``buffers`` must be exactly one layer's sparse buffers, in the order
        :meth:`get_mem_pool_base_address` numbers them. One layer, not one layer group:
        a layer's selection is normally computed from the previous layer's output, so a
        step's selections are not all known at its start, and giving each layer its own
        call lets the first layer's copies start as early as possible.

        The whole layer, not a subset, because the resolve marks entries resident on
        behalf of every buffer at once: filling only some would leave the rest stale while
        the metadata claimed otherwise.

        **Only for a layer whose selection is its own.** A model running one indexer
        across several layers hands them all the same units, and those layers name one
        ``selection_id`` and share a resident set. Their selection is known for all of
        them at once, so they resolve it once -- with
        :meth:`resolve_sparse_selection` -- and then fetch a layer at a time with
        :meth:`fetch_resolved_pages`. Calling this per layer instead would have the
        second resolve read the first's admissions as hits, issue no copies, and leave
        the second layer's cells unfilled; it raises rather than doing that.

        ``selected`` holds selection-unit ordinals, ``topk`` uniform across requests and ``-1``
        past a request's valid count. The page table and the per-request count of
        eligible blocks come from ``batch``, whose row order ``selected`` and ``out``
        both inherit.

        Eligible means below the history watermark: a block at or above it holds input
        tokens whose host page has not been written back yet, so selecting it would read
        stale bytes. Not a memory-safety problem -- the page table is uniform, so the
        ordinal resolves to a real page of the same request -- but a wrong answer. The call
        cannot detect an ineligible ordinal, so naming only eligible ones is the selector's
        responsibility.

        Each selected unit is looked up in the owning request's cache for that layer.
        A hit is reused in place; a miss is copied from the locked host page into a slot
        taken from the free list, evicting least-recently-used when it is empty.
        The Batch row's lease guarantees room for a full selection and the in-flight
        unit. Eviction is local to this selection's resident set; it cannot evict another
        selection's entries. Layers sharing a ``selection_id`` use the split resolve/fetch
        calls described above and keep their cells in matching slots.

        Copies leave the host page authoritative rather than moving it, so evicting a
        fetched cell moves no data. The copies are a snapshot, which is sound only because a
        block below the watermark is already written back and immutable. A cell holding an
        in-flight block is the one exception, and is never an eviction candidate.

        The in-flight unit is reserved internally but is not appended to ``out``. Use
        :meth:`prepare_sparse_write_indices` before the model's KV writes to obtain its
        per-buffer physical write indices independently of the selection.

        ``out`` is **compact**: entry ``[b][s][k]`` is the cache index holding
        ``selected[b][k]``, and ``BAD_PAGE_INDEX`` where that slot carries no selection.
        Padding stays where the selection put it rather than being packed out, so slot
        ``k`` of ``out`` and slot ``k`` of ``selected`` describe the same unit.

        ``s`` is the buffer's *output slice*, from :meth:`sparse_out_slots` -- not its
        position in ``buffers``. Buffers whose arena indices coincide resolve to one
        slice, so a slice may serve several buffers and the slices no buffer maps to are
        left untouched. Read a buffer's row at its slice; ``out``'s second extent stays
        ``len(buffers)`` so the shape is independent of how the buffers happen to merge.

        Compact, in the selection's own order, rather than a table indexed by unit
        ordinal, because that is the shape sparse attention kernels take: a token-sparse
        kernel is handed the index list in place of its page table and reads one unit per
        slot, and a block-sparse kernel takes the list beside a page table it still uses.
        None indexes a table by ordinal, so producing one would only oblige the caller to
        compact it again.

        The buffer dimension is a dimension rather than a separate array per buffer
        because attention kernels commonly want one tensor covering K and V. It cannot be
        dropped: a layer's buffers share the resolve, but not their cells. A lease gives
        each buffer its own contiguous run of cells, so one selection resolves to a
        different index per buffer, and each needs its own row.

        Indices are in the expanded index space :class:`PageIndexConverter` produces, and
        are relative to :meth:`get_sparse_cache_base_address` for the buffer they were
        fetched for -- **not** to :meth:`get_mem_pool_base_address`, and not to a cache
        level. The pool names the buffer's pages, which for a sparse life cycle are host
        memory; the cells a kernel reads live in a separate GPU arena, with a region per
        pool. Pairing these indices with the pool base reads the wrong allocation.

        The selection is per layer, not per KV head: every head of a layer shares one
        resident set. Algorithms that let different KV heads select different units are
        therefore not expressible against this cache.

        **Decode only.** ``selected`` carries one row per batch row, so a request
        contributes a single selection per call. Sparse attention kernels index their
        selection by *query token*, which coincides with the request only while a request
        contributes one token -- that is, during generation.

        The restriction is not just the row count. The resolve makes the selected units
        resident in the request's cache, which holds ``max_cached_units``; residency is a
        property of the request, not of a query token. A prefill chunk of N query tokens
        each choosing their own units has a union of up to N times the selection width, so
        the reservation could no longer cover a call, and a unit evicted for a later query
        token would invalidate an index already returned for an earlier one. Widening this
        to one row per token would make that reachable rather than fix it.

        Serving prefill from a host-resident history is an open design question; see
        SPARSE_ATTENTION_DESIGN.md.
        """

    def resolve_sparse_selection(
        self,
        buffers: Sequence[BufferId],
        selected: DeviceArray,  # int32[batch, topk]
        stream: CudaStream | None = None,
    ) -> None:
        """Resolve one selection, admitting what it names and listing what to copy.

        The first half of :meth:`fetch_sparse_pages`, split out for the layers that share
        a ``selection_id``: they resolve once between them and then fetch a layer at a
        time with :meth:`fetch_resolved_pages`.

        ``buffers`` names any one member layer in full. The residency and the miss list
        belong to the selection rather than to the layer that happened to name it, so
        which member is passed makes no difference -- but it must still be a whole
        layer's sparse buffers, as :meth:`fetch_sparse_pages` requires.

        The miss list this leaves behind is read by every :meth:`fetch_resolved_pages`
        that follows and is overwritten by the next resolve of the same selection, so
        fetch each member layer before resolving the selection again. Both run on
        ``stream``, so that ordering is the caller's only across streams.
        """

    def fetch_resolved_pages(
        self,
        buffers: Sequence[BufferId],
        selected: DeviceArray,  # int32[batch, topk]
        out: DeviceArray,  # int32[batch, len(buffers), topk]
        stream: CudaStream | None = None,
    ) -> None:
        """Copy one layer's share of a resolved selection, and write its indices.

        The second half of :meth:`fetch_sparse_pages`. ``selected`` must be the array the
        matching :meth:`resolve_sparse_selection` read, since the indices written are
        those units'; ``out`` is exactly as :meth:`fetch_sparse_pages` describes it, for
        this layer's buffers alone.

        Every layer of the selection copies the same units, each into its own cells: the
        entry index is shared, the bytes behind it are not.
        """

    def sparse_out_slots(self, buffers: Sequence[BufferId]) -> list[int]:
        """Which slice of ``out`` each buffer's indices land in.

        ``buffers`` is one layer's sparse buffers, as :meth:`fetch_sparse_pages` takes
        them, and the result is parallel to it: entry ``i`` is the second-dimension index
        to read ``buffers[i]`` at in the ``out`` of :meth:`fetch_sparse_pages`, and in the
        ``out`` of :meth:`prepare_sparse_write_indices`. Both use the same assignment.

        Buffers that resolve a selection to the same arena index share a slice, so the
        copy runs once and the kernel reads one row for both. Mirrored pools do this --
        equal width gives equal indices, as for a K and a V of the same shape, or a key
        and a value scale -- and the slots then repeat, e.g. ``[0, 1, 0, 1]``. Buffers of
        unequal width keep slices of their own, so a scale shared by K and V together
        yields ``[0, 1, 2]``.

        Slices are numbered by first use, so the distinct slices are ``0`` upwards and
        ``max(slots) + 1`` of the ``len(buffers)`` rows carry data. The rest are left
        untouched rather than packed out, keeping ``out``'s shape independent of the
        merge.

        The grouping follows the index table spec the batch was built with, so two
        buffers merge only where the caller asked for one table over both. It is fixed
        for the batch's lifetime: ask once and keep the answer.
        """

    def flushed_history(self, row: BatchRow) -> int:
        """The history a row has been written back to.

        A row that has never needed a flush reads its own history, so a caller may always
        compare against it.
        """

    def publish(self) -> None:
        """Copy the staged rows to the device, enqueuing on the batch's stream.

        The staged page indices are slot ids as stored -- unscaled, with
        ``BAD_PAGE_INDEX`` preserved rather than folded to a safe slot, since only the
        sentinel separates an absent block from slot zero. Publishing also converts them
        into the index tables declared at construction, which is what a sparse fetch and
        write-back read: there the buffer's own offset within its page is already in the
        value, so an entry counts whole buffers from its table's base.

        The page table is copied, not read in place: the snapshot is the whole point, since
        the host has to be free to write the next step's rows long before the kernels
        reading this step's have retired.

        Two stages. The host-to-device fill runs on an upload stream of the batch's own,
        so it overlaps the previous step -- the host enqueues this while that step's
        forward is still executing. The batch's stream then carries one linear
        device-to-device copy into the tables the fetch reads, and the conversion. Everything
        expensive is in the first stage, and only the second waits for anything.

        Raises if a member holds scratch slots in a life cycle this batch gave ``SHARED``:
        its layers no longer share one set of converted indices, so the single table the
        batch was built around cannot serve them. Where ``mode`` was derived that means the
        phases were mixed after construction fixed it; where ``SHARED`` was named, this is
        the check that holds the caller to the claim.

        Raises if the previous publish has not been waited on. This call stages the
        watermarks and views itself, so it would be writing into the same hazard it is
        meant to close. It refuses rather than waiting because a wait here covers only its
        own write: a member mutated before the call has already raced by then, and waiting
        would hide that. :meth:`resize` publishes, so republishing after one needs
        :meth:`wait_published` in between.

        **The second stage must be ordered after the kernels that read the previous
        publish.** Those live in a captured graph, so the tables cannot be double-buffered
        out of the hazard -- the capture baked their addresses, which is why the staging
        sits upstream of them instead. Issuing this on the stream the graph replays on is
        what provides that ordering today. A torn publish is not a stale answer: the cache views
        carry pointers, so it can pair one request's residency with another's lease.
        """

    def wait_published(self) -> None:
        """Block until the last :meth:`publish` has finished reading the staging rows.

        The copy is asynchronous and reads pinned memory, so it is still in flight when
        :meth:`publish` returns. Every later host write to a member's page indices -- a
        resize, a commit, a beam fork, an SWA recycle -- must be separated from it by this
        call, or that write races the copy and publishes a torn table.

        :meth:`resize` already does it, which is why driving a batched step through that
        rather than through the members is the sanctioned path. For the operations with no
        ``Batch`` equivalent the simple rule is to call this once at the top of a step: it
        is idempotent and free once the copy has landed.
        """

    def coalesced_page_table(self, layer_group_id: LayerGroupId) -> DeviceArray:
        """``int32[max_batch, max_blocks_per_seq]`` coalesced page indices, on the host.

        A read-only DLPack array over the pinned staging the members write into, which a
        consumer needing these numbers reads instead of keeping a buffer of its own. It keeps
        the ``Batch`` alive for as long as the array exists.

        Read-only because the entries belong to the members, written through their page
        locks: a write here would race a publish already reading the same memory, and would
        be lost anyway, since ``resize`` rewrites only the entries it changes.

        An entry names a coalesced slot, which may hold several buffers; a converted index
        names one buffer inside one. See :meth:`converted_index_table`.
        """

    def history_lengths(self) -> DeviceArray:
        """``int32[max_batch]`` per-row history length, in tokens.

        Published by the same transfer as the page tables, because it bounds which entries
        of a row are valid: pairing one step's table with another step's lengths would read
        past the end of a row that may since have been reused.

        The counts a consumer wants are all derived from this and :meth:`capacities`:
        ``num_blocks`` is ``history_length // tokens_per_block``, while the sparse resolve
        works in units and divides by ``unit_tokens`` instead. Publishing the source rather
        than the three derived counts keeps one truth.

        A free row reads zero -- a request with nothing eligible.
        """

    def capacities(self) -> DeviceArray:
        """``int32[max_batch]`` per-row capacity, in tokens. See :meth:`history_lengths`."""

class KVCacheManager:
    def __init__(
        self,
        config: KVCacheManagerConfig,
        event_manager: KVCacheEventManager | StreamingEventSink | None = None,
        cold_page_codec: IKvCacheColdPageCodec | None = None,
    ) -> None: ...
    def __del__(self) -> None: ...
    def shutdown(self) -> None: ...
    def clear_reusable_blocks(self) -> None: ...
    def get_mem_pool_base_address(
        self, layer_id: LayerId, data_role: DataRole, index_mode: PageIndexMode | None = None
    ) -> MemAddress: ...
    def get_sparse_cache_max_units(self, layer_id: LayerId) -> int:
        """Cells a request's cache holds per buffer of this layer, in selection units.

        Counts both populations -- the history it caches and everything in flight -- so the
        lease follows from this alone. Only one unit is ever in flight, so the surplus above
        the selection width is retention rather than headroom. Zero when the layer has no
        sparse fetch path.
        """

    def get_sparse_cache_units_per_page(self, layer_id: LayerId) -> int:
        """Selection units one page of this layer's buffers holds; one when a unit is a page.

        Distinct from :attr:`PageIndexConverter.expansion`, which is how much a buffer's page
        *index space* is stretched by ``tokens_per_block_override``. A sparse buffer does not
        stretch its index space -- that stays 1 -- while this counts the sparse cache's own
        granularity, set by the layer's ``sparse_cache.unit_tokens``.
        """

    def get_num_bytes(self, level: int, medium: CacheTier) -> int:
        """Bytes this level holds in pools whose pages are ``medium``.

        A level's quota pays only for its own tier, so bytes of a second medium are drawn
        from no quota and appear in no other total a caller can read -- :meth:`get_quota`
        excludes them. Report them separately when accounting for what a deployment pins.

        Pair this with :meth:`sparse_backing_medium` to count what sparse buffers hold. A
        sparse pool's row count follows from the device-backed pools of its group, since a
        slot spans every pool, so that figure moves only when the hot level is resized.
        """

    def sparse_backing_medium(self) -> CacheTier | None:
        """The medium a sparse buffer's pages are, or ``None`` when none is sparse.

        The hot level's second medium: host memory the GPU reads through a device mapping,
        which is what sparse loading pulls into the per-request GPU cache.
        """

    def get_sparse_cache_buffer_count(self, layer_id: LayerId) -> int:
        """Sparse buffers of this layer's life cycle, one per ``(layer, role)``.

        Each has its own cells, so a lease is this many runs of
        :meth:`get_sparse_cache_max_units` cells.
        """

    def is_sparse(self, layer_id: LayerId, data_role: DataRole) -> bool:
        """Whether this buffer was declared ``BufferConfig.is_sparse``.

        Every buffer locks its pages at the hot level, sparse or not, so
        :meth:`_KVCache.get_base_page_indices` reports one level's numbering throughout.
        What a sparse buffer changes is the *medium*: its pages are host memory,
        device-mapped, in a pool of their own beside the layer's device-backed ones.

        That level is not what a caller addresses to read a unit resolved by
        :meth:`fetch_sparse_pages`. Those live in the request's lease on the life cycle's
        cache arena, which is a separate allocation reached through
        :meth:`get_sparse_cache_base_address`.
        """

    def get_page_stride(self, layer_id: LayerId, data_role: DataRole) -> int: ...
    def sparse_index_table_specs(self) -> list[IndexTableSpec]:
        """The index table specs every sparse life cycle needs, one per (layer, pool) run.

        A :class:`Batch` still has to be given these: it refuses a sparse buffer it was told
        nothing about, because a fetch with no converted indices would read the wrong pages
        rather than fail. Their shape is not a choice -- a copy walks one run per launch
        from one base pointer, so the run is the request -- so this saves re-deriving the
        pool layout to restate it.
        """
    def get_page_index_upper_bound(self, layer_id: LayerId, data_role: DataRole) -> int: ...
    def get_page_index_scale(self, layer_id: LayerId, data_role: DataRole) -> int: ...
    def get_page_index_converter(
        self, layer_id: LayerId, data_role: DataRole
    ) -> PageIndexConverter: ...
    def create_kv_cache(
        self,
        reuse_scope: ReuseScope | None = None,
        input_tokens: Sequence[TokenIdExt] | None = None,
        id: Any = None,
        custom_priority_callback: Callable[[int, Any], Priority] = ...,
        expected_prompt_length: int | None = None,
        text_only: bool | None = None,
        enable_request_stats: bool = False,
        kv_cache_retention_config: KvCacheRetentionConfig | None = None,
    ) -> _KVCache:
        """Create a suspended cache with a prefill-to-generation boundary.

        On the C++ backend, ``expected_prompt_length`` also determines which full
        prompt blocks are shared by beams. Pass the actual full prompt length for
        beam search if ``input_tokens`` is absent or shortened for reuse matching.
        It defaults to the non-empty input length; without either value, the beam
        sharing boundary is zero and the statistics boundary is unset. When set,
        it also marks generation-phase allocation stats.
        """
        ...
    def refresh_retention(self) -> None: ...
    def prefetch_reuse(
        self, reuse_scope: ReuseScope, input_tokens: Sequence[TokenIdExt], target: CacheLevel
    ) -> bool:
        """Prefetch matched storage while preserving request retention policies and deadlines."""
    def probe_reuse(
        self,
        reuse_scope: ReuseScope | None = None,
        input_tokens: Sequence[TokenIdExt] | None = None,
    ) -> int: ...
    def probe_first_new_block_key(
        self,
        reuse_scope: ReuseScope | None = None,
        input_tokens: Sequence[TokenIdExt] | None = None,
    ) -> bytes | None: ...
    def resize(self, cache_level: CacheLevel, quota: int, best_efforts: bool = False) -> bool: ...
    def get_quota(self, cache_level: CacheLevel) -> int: ...
    def get_storage_statistics(self, cache_level: CacheLevel = ...) -> list[StorageStatistics]: ...
    def get_life_cycle_pool_group_indices(
        self, cache_level: CacheLevel = ...
    ) -> list[PoolGroupIndex]: ...
    def get_committed_stats(self) -> KVCacheStatsDelta: ...
    def get_and_reset_iteration_stats(self) -> dict[LifeCycleId, KVCacheIterationStatsDelta]: ...
    def get_and_reset_ssm_snapshot_iteration_stats(
        self,
    ) -> dict[LifeCycleId, SsmSnapshotIterationStatsDelta]: ...
    def record_request_suspended(self) -> None: ...
    def record_request_resumed(self) -> None: ...
    def get_and_reset_iteration_suspend_resume_stats(self) -> tuple[int, int]: ...
    def get_and_reset_iteration_disk_prefetch_blocks(self) -> int: ...
    def get_and_reset_iteration_cached_tokens_by_level(self) -> list[int]: ...
    def get_and_reset_iteration_connector_cached_tokens(self) -> int: ...
    def get_and_reset_iteration_reused_blocks_by_level(
        self,
    ) -> dict[LifeCycleId, ReusedBlocksByLevel]: ...
    def get_and_reset_iteration_connector_reused_blocks(self) -> dict[LifeCycleId, int]: ...
    def get_and_reset_iteration_peak_block_stats(
        self, cache_level: CacheLevel
    ) -> Sequence[PoolGroupPeakBlockStats]: ...
    def get_and_reset_iteration_peak_block_stats_by_level(
        self,
    ) -> Sequence[Sequence[PoolGroupPeakBlockStats]]: ...
    def mark_stats_dirty(self, kv_cache_id: int | None) -> None: ...
    def clear_stats_dirty(self, kv_cache_id: int | None) -> None: ...
    def get_dirty_stats_kv_cache_ids(self) -> set[int]: ...
    def mark_stats_excluded(self, kv_cache_id: int | None) -> None: ...
    def clear_stats_excluded(self, kv_cache_id: int | None) -> None: ...
    def is_stats_excluded(self, kv_cache_id: int | None) -> bool: ...
    @property
    def cache_tier_list(self) -> Sequence[CacheTier]: ...
    @property
    def tokens_per_block(self) -> int: ...
    @property
    def event_manager(self) -> Any | None: ...
    @property
    def init_config(self) -> KVCacheManagerConfig: ...
    @property
    def allow_seq_rebasing(self) -> bool: ...
    @property
    def enable_partial_match(self) -> bool: ...
    @property
    def enable_partial_commit(self) -> bool: ...
    def supports_index_mode(
        self, mode: PageIndexMode, layer_group_id: LayerGroupId | None = None
    ) -> bool | None:
        """``None`` means it depends on the request.

        Given a layer group the answer is definitive for every life cycle that cannot hold
        scratch slots -- anything without a sliding window -- which the collapsing form
        cannot express. Configuration only, so it never varies with what a request is doing.
        """

    def max_scratch_slots(self, max_blocks_per_seq: int) -> int:
        """The most scratch slots one request can need, over every life cycle and buffer.

        Given the widest page table a row will get; zero where nothing can hold scratch.

        Sizes scratch staging. A caller that stages scratch slots itself asks this rather
        than deriving it: which blocks can be stale turns on the window, the sinks and the
        rewind margin, and a second copy of that reasoning is a second thing to keep right.
        """

    @property
    def num_layers(self) -> int: ...
    @property
    def layer_ids(self) -> Iterator[LayerId]: ...
    # Deprecated: use get_buffer_group_id. Raises RuntimeError for a split layer.
    def get_layer_group_id(self, layer_id: LayerId) -> LayerGroupId: ...
    def get_buffer_group_id(self, layer_id: LayerId, role: DataRole) -> CacheGroupId: ...
    # Deprecated: use buffer_grouping. Raises RuntimeError if any layer is split.
    @property
    def layer_grouping(self) -> Sequence[Sequence[LayerId]]: ...
    @property
    def buffer_grouping(self) -> Sequence[Sequence[BufferId]]:
        """Buffer membership indexed by CacheGroupId."""
        ...
    @property
    def all_buffer_ids(self) -> Iterator[BufferId]: ...
    def get_aggregated_pages(self, buffers: Iterable[BufferId]) -> Iterator[AggregatedPageDesc]: ...
    @property
    def pool_group_descs(self) -> Sequence[PoolGroupDesc]: ...
    def clamp_max_seq_len_for_mem(self, batch_size: int, token_num_upper_bound: int) -> int: ...
    def adjust(self) -> None: ...
    @property
    def need_adjustment(self) -> bool: ...
    @property
    def commit_min_snapshot(self) -> bool: ...

def exact_div(x: int, y: int) -> int: ...
def typed_range(*args: int) -> range: ...
