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
"""Real KV cache managers, requests and byte oracles for the lender tests. The oracles bypass the
code under test: device pages through ``impl.pool_group_descs``, staging bytes through raw host
addresses, block keys through ``sequence_to_blockchain_keys``. Nothing here touches CUDA at import."""

from __future__ import annotations

import ctypes
import gc
import hashlib
import inspect
import threading
import time
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple
from unittest.mock import patch

import numpy as np
import pytest
import torch

TPB = 32
WINDOW = 64
MAX_SEQ_LEN = 256
SENTINEL = 0xEE
# The smallest pool the harness manager accepts: few enough pages that freed ones get reused.
POOL_TOKENS = 256

_SHARING = "tensorrt_llm._torch.pyexecutor.kv_cache.sharing"


class _RanksAgree:
    """The collectives of a TP manager built alone in this process: every rank agrees with it."""

    local_world_size = 1

    @staticmethod
    def allreduce(value, op=None):
        return value


@contextmanager
def _collectives_for(mapping):
    """Stand-in collectives while a multi-rank manager is built in a single process."""
    if mapping is None or mapping.world_size == 1:
        yield
        return
    from tensorrt_llm._torch.distributed.communicator import Distributed

    with patch.object(Distributed, "get", return_value=_RanksAgree()):
        yield


def make_manager(
    *,
    windows: Optional[List[int]] = None,
    max_tokens: int = 2048,
    tokens_per_block: int = TPB,
    num_layers: int = 2,
    num_kv_heads: int = 4,
    head_dim: int = 64,
    mapping=None,
    kv_cache_type=None,
    dtype=None,
    enable_block_reuse: bool = True,
    swa_scratch_reuse: bool = False,
    max_batch_size: int = 4,
    **kv_cache_config,
):
    """A small real ``KVCacheManagerV2`` on its own execution stream, FP16 unless ``dtype``; further
    keywords go to its ``KvCacheConfig``."""
    import tensorrt_llm.bindings
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    batch_manager = tensorrt_llm.bindings.internal.batch_manager
    config = dict(max_tokens=max_tokens, enable_block_reuse=enable_block_reuse)
    if swa_scratch_reuse:
        config["enable_swa_scratch_reuse"] = True
    if windows is not None:
        config["max_attention_window"] = windows
    config.update(kv_cache_config)
    mapping = mapping or Mapping(world_size=1, tp_size=1, rank=0)
    with _collectives_for(mapping):
        return KVCacheManagerV2(
            kv_cache_config=KvCacheConfig(**config),
            kv_cache_type=kv_cache_type or batch_manager.CacheType.SELF,
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            tokens_per_block=tokens_per_block,
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=max_batch_size,
            mapping=mapping,
            dtype=dtype or tensorrt_llm.bindings.DataType.HALF,
            vocab_size=32000,
            execution_stream=torch.cuda.Stream(),
        )


def _make_deepseek_v4_manager(mapping=None):
    """A four-layer DeepSeek-V4 manager: one layer of each compression kind, FP8, one KV head."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import DeepseekV4CacheManager
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig, KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    compress_ratios = [1, 4, 128, 4]
    mapping = mapping or Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    with _collectives_for(mapping):
        return DeepseekV4CacheManager(
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=True, max_tokens=4096, event_buffer_max_size=0
            ),
            kv_cache_type=CacheType.SELFKONLY,
            num_layers=len(compress_ratios),
            num_kv_heads=1,
            head_dim=512,
            tokens_per_block=128,
            max_seq_len=1024,
            max_batch_size=2,
            max_input_len=1024,
            mapping=mapping,
            dtype=DataType.FP8,
            compressor_dtype=DataType.FLOAT,
            vocab_size=129280,
            max_num_tokens=2 * 1025,
            sparse_attn_config=DeepSeekV4SparseAttentionConfig(
                index_head_dim=128, window_size=128, compress_ratios=compress_ratios
            ),
            execution_stream=torch.cuda.Stream(),
        )


def _make_hybrid_manager():
    """Nemotron-H shaped, few layers: Mamba2 state at layers 0, 2 and 4, attention at layer 3."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import (
        MambaHybridCacheManagerV2,
    )
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig, MambaStateConfig
    from tensorrt_llm.mapping import Mapping

    pattern = "M-M*M-"
    mamba_mask = [c == "M" for c in pattern]
    attn_mask = [c == "*" for c in pattern]
    with _collectives_for(Mapping(world_size=1, rank=0, tp_size=1)):
        return MambaHybridCacheManagerV2(
            mamba_d_state=128,
            mamba_d_conv=4,
            mamba_num_heads=128,
            mamba_n_groups=8,
            mamba_head_dim=80,
            mamba_num_layers=sum(mamba_mask),
            mamba_layer_mask=mamba_mask,
            mamba_cache_dtype=torch.bfloat16,
            mamba_ssm_cache_dtype=torch.float32,
            kv_cache_config=KvCacheConfig(
                max_tokens=2048,
                enable_block_reuse=True,
                mamba_state_config=MambaStateConfig(periodic_snapshot_interval=256),
            ),
            kv_cache_type=CacheType.SELF,
            num_layers=sum(attn_mask),
            num_kv_heads=8,
            head_dim=128,
            tokens_per_block=32,
            max_seq_len=1024,
            max_batch_size=2,
            mapping=Mapping(world_size=1, rank=0, tp_size=1),
            layer_mask=attn_mask,
            vocab_size=1024,
            dtype=DataType.BF16,
        )


@contextmanager
def _managed(factory: Callable[[], object]) -> Iterator[object]:
    """A manager that is shut down, with every request's cache freed, when the block exits."""
    torch.cuda.init()
    gc.collect()
    torch.cuda.empty_cache()
    mgr = factory()
    try:
        yield mgr
    finally:
        stream = getattr(mgr, "_stream", None)
        if stream is not None:
            stream.synchronize()
        mgr.shutdown()
        del mgr
        gc.collect()
        torch.cuda.empty_cache()


@pytest.fixture
def real_manager():
    """``with real_manager(windows=None, max_tokens=2048, ...) as mgr:`` a small ``KVCacheManagerV2``
    with block reuse on and its own execution stream, shut down when the block exits."""

    def factory(**kwargs):
        return _managed(lambda: make_manager(**kwargs))

    return factory


@contextmanager
def _host_tiered(factory: Callable[[], object]) -> Iterator[object]:
    """``_managed``, checked to have enough host slots to spill its GPU pool."""
    with _managed(factory) as mgr:
        tiers = [str(tier) for tier in mgr.impl.cache_tier_list]
        assert len(tiers) == 2 and "HOST" in tiers[1], f"no host tier below the GPU pool: {tiers}"
        gpu_slots = sum(int(stats.total) for stats in mgr.impl.get_storage_statistics(0))
        host_slots = sum(int(stats.total) for stats in mgr.impl.get_storage_statistics(1))
        assert host_slots >= gpu_slots, (
            f"host tier has {host_slots} slots for a GPU pool of {gpu_slots} slots"
        )
        yield mgr


@pytest.fixture
def host_tier_manager():
    """``with host_tier_manager(**kwargs) as mgr:`` like ``real_manager`` on a pool of
    ``POOL_TOKENS``, with an explicitly sized host tier. Index slots for many one-block
    requests and resume allowed up to a full pool let other requests push pages to host and back."""

    def factory(**kwargs):
        # Keep spill capacity independent of the process's memlock-limited automatic quota.
        config = dict(
            max_tokens=POOL_TOKENS,
            max_batch_size=64,
            max_util_for_resume=1.0,
            host_cache_size=16 << 20,
        )
        config.update(kwargs)
        return _host_tiered(lambda: make_manager(**config))

    return factory


@pytest.fixture
def deepseek_v4_manager():
    """``with deepseek_v4_manager(mapping=None) as mgr:`` a small ``DeepseekV4CacheManager``;
    callers skip before Blackwell."""

    def factory(mapping=None):
        return _managed(lambda: _make_deepseek_v4_manager(mapping))

    return factory


@pytest.fixture
def hybrid_manager():
    """``with hybrid_manager() as mgr:`` a manager whose layer groups include recurrent state."""

    def factory():
        return _managed(_make_hybrid_manager)

    return factory


# -- requests ---------------------------------------------------------------------------------


def make_request(request_id: int, tokens: Sequence[int], **kwargs):
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig

    return LlmRequest(
        request_id=request_id,
        max_new_tokens=4,
        input_tokens=list(tokens),
        sampling_config=SamplingConfig(1),
        is_streaming=False,
        **kwargs,
    )


def kv(mgr, request):
    """The request's runtime cache, or ``None``: read straight from the manager's map."""
    return mgr.kv_cache_map.get(request.py_request_id)


def closed(kv_cache) -> bool:
    return kv_cache.status == kv_cache.Status.CLOSED


def chain_keys(kv_cache, tokens: Sequence) -> List[bytes]:
    """The radix keys of every whole block of ``tokens``, computed independently of the lender."""
    from tensorrt_llm.runtime.kv_cache_manager_v2 import sequence_to_blockchain_keys

    tpb = kv_cache.tokens_per_block
    keys = []
    for i, (block, key) in enumerate(
        sequence_to_blockchain_keys(tpb, kv_cache.reuse_scope, list(tokens))
    ):
        if i and len(block) == tpb:
            keys.append(bytes(key))
    return keys


def fake_page(lg: int, key: bytes, nbytes: int) -> bytes:
    """Deterministic page content named by (layer group, block key): equal tokens, equal bytes."""
    seed = int.from_bytes(hashlib.sha256(bytes([lg]) + key).digest()[:8], "little")
    return np.random.default_rng(seed).integers(0, 256, nbytes, dtype=np.uint8).tobytes()


def pages(kv_cache, lg: int) -> List[int]:
    return [int(p) for p in kv_cache.get_aggregated_page_indices(lg, valid_only=False)]


def num_layer_groups(mgr) -> int:
    return len(mgr.impl.layer_grouping)


def pool_group_of(mgr) -> List[int]:
    """Device pool group of each layer group."""
    return [int(g) for g in mgr.impl.get_life_cycle_pool_group_indices()]


def pool_group_ids(mgr) -> List[int]:
    """Device pool groups in index order: the order of a staging lender's parts."""
    return sorted(int(pg.pool_group_index) for pg in mgr.impl.pool_group_descs)


def windows(mgr) -> Tuple[Optional[int], ...]:
    """Each layer group's sliding window in tokens, ``None`` for full attention."""
    out = []
    for lc in mgr._life_cycle_by_layer_group():
        window = getattr(lc, "window_size", None)
        out.append(None if window is None or window >= MAX_SEQ_LEN else int(window))
    return tuple(out)


def stale_blocks(mgr, lg: int, history: int) -> Tuple[int, int]:
    beg, end = mgr._stale_block_range(lg, history)
    return int(beg), int(end)


class DevicePages:
    """Device pages addressed by (layer group, slot), straight from ``impl.pool_group_descs``; all
    work runs on the manager's stream."""

    def __init__(self, mgr):
        from tensorrt_llm._utils import TensorWrapper, convert_to_torch_tensor

        self._mgr = mgr
        self._group_of = pool_group_of(mgr)
        self._pools: Dict[int, List[torch.Tensor]] = {}
        for pg in mgr.impl.pool_group_descs:
            ordered = sorted(pg.pools, key=lambda p: int(p.pool_index))
            self._pools[int(pg.pool_group_index)] = [
                convert_to_torch_tensor(
                    TensorWrapper(
                        int(p.base_address),
                        torch.uint8,
                        shape=(int(pg.num_slots), int(p.slot_bytes)),
                    )
                )
                for p in ordered
            ]
        # Loading the fill kernel waits for the whole device, so load it before any stream is held.
        torch.empty(1, dtype=torch.uint8, device="cuda").fill_(0)
        torch.cuda.synchronize()

    def group_page_bytes(self, group: int) -> int:
        return sum(t.shape[1] for t in self._pools[group])

    def page_bytes(self, lg: int) -> int:
        return self.group_page_bytes(self._group_of[lg])

    def read(self, lg: int, slot: int) -> bytes:
        self._mgr._stream.synchronize()
        tensors = self._pools[self._group_of[lg]]
        return b"".join(t[slot].cpu().numpy().tobytes() for t in tensors)

    def write(self, lg: int, slot: int, data: bytes) -> None:
        stream = self._mgr._stream
        src = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        offset = 0
        with torch.cuda.stream(stream):
            for t in self._pools[self._group_of[lg]]:
                width = t.shape[1]
                t[slot].copy_(src[offset : offset + width].to(t.device))
                offset += width
        stream.synchronize()

    def fill_async(self, lg: int, slot: int, byte: int) -> None:
        """Fill a page on the manager's stream without waiting: ordered after work already queued."""
        with torch.cuda.stream(self._mgr._stream):
            for t in self._pools[self._group_of[lg]]:
                t[slot].fill_(byte)


def page(mgr, request, lg: int, ordinal: int) -> bytes:
    return DevicePages(mgr).read(lg, pages(kv(mgr, request), lg)[ordinal])


def write_fake_kv(mgr, request, first_block: int = 0) -> None:
    """Fill every page the request holds from ``first_block`` on, as a forward pass would."""
    kv_cache = kv(mgr, request)
    dev = DevicePages(mgr)
    keys = chain_keys(kv_cache, request.get_tokens(0))
    for lg in range(num_layer_groups(mgr)):
        for ordinal, slot in enumerate(pages(kv_cache, lg)):
            if ordinal < first_block or slot < 0 or ordinal >= len(keys):
                continue
            dev.write(lg, slot, fake_page(lg, keys[ordinal], dev.page_bytes(lg)))


def fill_sentinel(mgr, request, first_block: int = 0) -> None:
    """Mark the request's pages from ``first_block`` on so a missing copy shows."""
    kv_cache = kv(mgr, request)
    dev = DevicePages(mgr)
    for lg in range(num_layer_groups(mgr)):
        for ordinal, slot in enumerate(pages(kv_cache, lg)):
            if ordinal >= first_block and slot >= 0:
                dev.write(lg, slot, bytes([SENTINEL]) * dev.page_bytes(lg))


def prefill(mgr, request) -> None:
    """Run a whole prompt the way the executor does: allocate, compute (fake KV), advance, then
    ``update_context_resources``, which commits the blocks."""
    from tensorrt_llm._torch.pyexecutor.scheduler import ScheduledRequests

    assert mgr.prepare_context(request)
    kv_cache = kv(mgr, request)
    # Real pages for every block written below; scratch slots would be overwritten.
    kv_cache.enable_swa_scratch_reuse = False
    first = request.context_current_position // mgr.tokens_per_block
    assert mgr.resize_context(request, request.context_remaining_length)
    write_fake_kv(mgr, request, first)
    request.move_to_next_context_chunk()
    batch = ScheduledRequests()
    batch.append_context_request(request)
    mgr.update_context_resources(batch)
    assert request.context_remaining_length == 0
    assert kv_cache.num_committed_tokens == request.prompt_len


def published(mgr, request_id: int, tokens: Sequence[int], **kwargs):
    """A request that computed ``tokens``: its blocks are committed."""
    request = make_request(request_id, tokens, **kwargs)
    prefill(mgr, request)
    return request


def admitted(mgr, request_id: int, tokens: Sequence[int]):
    """A request admitted to fetch its prompt: a cache holding its local match and nothing more,
    scratch reuse off as a fetch target needs. The lender grows it."""
    request = make_request(request_id, tokens)
    assert mgr.prepare_context(request)
    kv(mgr, request).enable_swa_scratch_reuse = False
    return request


def host_bytes(address: int, length: int) -> bytes:
    return ctypes.string_at(address, length)


def digest(value):
    """``value`` with every byte string replaced by its length and hash, or its one repeated byte:
    a failed comparison then names the entries that differ without diffing page contents."""
    if isinstance(value, (bytes, bytearray)):
        if value and value.count(value[:1]) == len(value):
            return f"{len(value)} x {value[0]:#04x}"
        return f"{len(value)} B sha256 {hashlib.sha256(value).hexdigest()[:16]}"
    if isinstance(value, dict):
        return {key: digest(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [digest(item) for item in value]
    return value


def stage(lender, view, byte: int) -> None:
    """Write ``byte`` over every slot of a staging view, as a backend's transfer would."""
    for run in view.runs:
        length = lender.parts[run.part].slot_bytes
        for address in run.addresses.tolist():
            ctypes.memset(address, byte, length)


def relay(src_lender, src_view, dst_lender, dst_view, deliver=None) -> Tuple[np.ndarray, ...]:
    """A backend between two staging lenders: each destination row takes the slot of the source row
    of the same name. ``deliver(run_index, row)`` picks the rows (default all); returns the masks."""
    by_name = {}
    for run in src_view.runs:
        length = src_lender.parts[run.part].slot_bytes
        for name, address in zip(run.names, run.addresses.tolist()):
            by_name[name.tobytes()] = (address, length)
    masks = []
    for i, run in enumerate(dst_view.runs):
        mask = np.zeros(len(run), dtype=bool)
        for row, (name, address) in enumerate(zip(run.names, run.addresses.tolist())):
            if deliver is not None and not deliver(i, row):
                continue
            src, length = by_name[name.tobytes()]
            assert length == dst_lender.parts[run.part].slot_bytes
            ctypes.memmove(address, src, length)
            mask[row] = True
        masks.append(mask)
    return tuple(masks)


# -- streams ----------------------------------------------------------------------------------


class _Gate:
    def __init__(self, flag: torch.Tensor):
        self._flag = flag
        self.opened = False
        self.tripped_by_watchdog = False

    def open(self) -> None:
        self._flag.numpy()[0] = 1
        self.opened = True


def _wait_on_flag(stream: torch.cuda.Stream) -> _Gate:
    from cuda.bindings import driver

    flag = torch.zeros(1, dtype=torch.int32, pin_memory=True)
    (err,) = driver.cuStreamWaitValue32(
        driver.CUstream(stream.cuda_stream),
        driver.CUdeviceptr(flag.data_ptr()),
        1,
        driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_GEQ,
    )
    if err == driver.CUresult.CUDA_ERROR_NOT_SUPPORTED:
        pytest.skip("this device cannot hold a stream on a host flag")
    assert err == driver.CUresult.CUDA_SUCCESS, err
    return _Gate(flag)


@contextmanager
def held_stream(stream: torch.cuda.Stream, strict: bool = True, watchdog_seconds: float = 120.0):
    """Hold ``stream`` on a host flag until ``gate.open()``: work queued meanwhile does not start.
    ``strict`` makes a torch host sync raise; a watchdog opens the gate if the test never does, and
    the block then fails, so a host wait on the stream cannot pass unnoticed."""
    gate = _wait_on_flag(stream)

    def trip():
        gate.tripped_by_watchdog = True
        gate.open()

    watchdog = threading.Timer(watchdog_seconds, trip)
    watchdog.start()
    previous = torch.cuda.get_sync_debug_mode()
    if strict:
        torch.cuda.set_sync_debug_mode("error")
    try:
        yield gate
    finally:
        torch.cuda.set_sync_debug_mode(previous)
        gate.open()
        watchdog.cancel()
        watchdog.join()
        stream.synchronize()
    assert not gate.tripped_by_watchdog, "the stream stayed held: something waited on it"


@contextmanager
def gated_stream(stream: torch.cuda.Stream, open_after: float):
    """Hold ``stream`` until a timer opens it after ``open_after`` seconds; a host wait on the
    stream then simply lasts that long. ``gate.opened`` tells whether it has."""
    gate = _wait_on_flag(stream)
    timer = threading.Timer(open_after, gate.open)
    timer.start()
    try:
        yield gate
    finally:
        timer.cancel()
        gate.open()
        timer.join()
        stream.synchronize()


@contextmanager
def events_report_done():
    """Every CUDA event reports its work done while the block runs, finished or not."""
    original = torch.cuda.Event.query
    torch.cuda.Event.query = lambda self: True
    try:
        yield
    finally:
        torch.cuda.Event.query = original


# -- pool pressure ----------------------------------------------------------------------------


def pool_pages(mgr) -> int:
    (group,) = mgr.impl.pool_group_descs
    return int(group.num_slots)


def warm_tier_moves(mgr) -> None:
    """Move one committed block to host and back. The first such move in a process loads the kernels
    that copy between tiers, which waits for the whole device; a held stream would stall it."""
    tokens = list(range(40_000, 40_000 + TPB + 1))
    mgr.free_resources(published(mgr, 900, tokens))
    others = Requests(mgr)
    try:
        while others.allocate(1, chunk=1):  # until the pool is full: the block went to host
            pass
    finally:
        others.free()
    back = make_request(901, tokens)
    assert mgr.prepare_context(back)
    assert kv(mgr, back).num_committed_tokens == TPB, "the block did not come back from host"
    mgr.free_resources(back)
    torch.cuda.synchronize()


def tier_used(mgr, level: int) -> int:
    """Slots in use at cache tier ``level`` (0 the GPU, 1 the host) over every pool group, from
    the manager's storage statistics."""
    return sum(int(s.total) - int(s.free) for s in mgr.impl.get_storage_statistics(level))


class Requests:
    """Other requests' allocations: private pages taken straight from the pool, sentinel-filled,
    a few blocks per request (a request holds at most ``MAX_SEQ_LEN`` tokens)."""

    def __init__(self, mgr, dev: Optional[DevicePages] = None):
        # ``dev`` built ahead lets ``fill=False`` run while the stream is held: building the page
        # views waits on the stream.
        self._mgr = mgr
        self._dev = dev
        self._next = 100
        self.held = []

    def allocate(self, blocks: int, fill: bool = True, chunk: Optional[int] = None) -> bool:
        """Take ``blocks`` pages, ``chunk`` blocks per request. ``False`` if the pool could not give
        them all; what it gave stays held. ``fill=False`` fills on the stream without waiting."""
        mgr = self._mgr
        most = chunk or MAX_SEQ_LEN // TPB - 1
        while blocks:
            chunk = min(blocks, most)
            rid = self._next
            self._next += 1
            first = 50_000 + 1_000 * rid  # never matches another request's blocks
            request = make_request(rid, list(range(first, first + chunk * TPB)))
            if not (
                mgr.prepare_context(request)
                and mgr.resize_context(request, request.context_remaining_length)
            ):
                mgr.free_resources(request)
                return False
            if fill:
                fill_sentinel(mgr, request)
            else:
                dev = self._dev or DevicePages(mgr)
                for lg in range(num_layer_groups(mgr)):
                    for slot in pages(kv(mgr, request), lg):
                        if slot >= 0:
                            dev.fill_async(lg, slot, SENTINEL)
            self.held.append(request)
            blocks -= chunk
        return True

    def pages(self) -> set:
        return {p for r in self.held for p in pages(kv(self._mgr, r), 0) if p >= 0}

    def free_all_but(self, keep: set) -> None:
        """Free the requests holding none of the pages in ``keep``."""
        kept = []
        for request in self.held:
            if keep & {p for p in pages(kv(self._mgr, request), 0) if p >= 0}:
                kept.append(request)
            else:
                self._mgr.free_resources(request)
        self.held = kept

    def free(self) -> None:
        for request in self.held:
            self._mgr.free_resources(request)
        self.held = []


def overwrite_free_pages(mgr) -> int:
    """Other requests take every free page one block at a time, fill it with ``SENTINEL`` and free
    it again, so whatever a freed page held is gone. Returns how many pages they took."""
    others = Requests(mgr)
    try:
        while others.allocate(1, chunk=1):
            pass
        return len(others.held)
    finally:
        others.free()


def taken_by_others(mgr, lent: set) -> set:
    """Other requests ask for every page the pool has free besides ``lent``, then for one more.
    Returns the pages of ``lent`` they got; they are freed again before returning."""
    others = Requests(mgr)
    try:
        assert others.allocate(pool_pages(mgr) - len(lent))
        others.allocate(1)
        return others.pages() & lent
    finally:
        others.free()


def whole_pool_goes_to_others(mgr, lent: set) -> bool:
    """Whether other requests can take every page of the pool, ``lent`` included."""
    others = Requests(mgr)
    try:
        return others.allocate(pool_pages(mgr)) and lent <= others.pages()
    finally:
        others.free()


class ShutdownSpy:
    """Stands in for ``mgr.impl``, recording whether the manager shut it down."""

    def __init__(self, impl):
        self._impl = impl
        self.shut_down = False

    def shutdown(self):
        self.shut_down = True
        self._impl.shutdown()

    def __getattr__(self, name):
        return getattr(self._impl, name)


class StatsSpy:
    """Stands in for ``mgr.impl``, recording each clear of a stats exclusion and whether the cache
    was closed by then."""

    def __init__(self, impl, kv_cache):
        self._impl = impl
        self._kv = kv_cache
        self.cleared = []

    def clear_stats_excluded(self, request_id):
        self.cleared.append((request_id, closed(self._kv)))
        self._impl.clear_stats_excluded(request_id)

    def __getattr__(self, name):
        return getattr(self._impl, name)


# -- memory kept until exit -------------------------------------------------------------------


def retained() -> Tuple[object, ...]:
    """What the lender keeps until the process exits, through its test hook."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    return _lender._retained()


def staging_memory(parts) -> List[object]:
    """The kept host allocations holding any of ``parts``, compared by address."""
    from tensorrt_llm._torch.pyexecutor.kv_cache.sharing import _lender

    found = []
    for obj in retained():
        if not isinstance(obj, _lender._HostMemory):
            continue
        begin, end = obj.address, obj.address + obj.nbytes
        if any(begin <= part.address < end for part in parts):
            found.append(obj)
    return found


# -- the manager's host page-index buffer ------------------------------------------------------

CANARY = 0x5A5A5A5A


class IndexRow(SimpleNamespace):
    """Where a request's cache writes its base page indices of one pool in the manager's host
    page-index buffer: the buffer's address, size and pinning, and the row's int32 offset in it."""


def index_row(mgr, request, pool: int = 0, beam: int = 0) -> IndexRow:
    buf = mgr.host_kv_cache_block_offsets
    index = mgr.index_mapper.get_index(int(request.py_request_id))
    row = buf[pool, index * mgr.max_beam_width + beam, 0]
    return IndexRow(
        address=buf.data_ptr(),
        nbytes=buf.numel() * buf.element_size(),
        pinned=buf.is_pinned(),
        offset=(row.data_ptr() - buf.data_ptr()) // 4,
        length=row.numel(),
        values=row.tolist(),
    )


def reclaim(row: IndexRow, tries: int = 64) -> Optional[torch.Tensor]:
    """A new host int32 tensor at a freed buffer's address, every element ``CANARY``; ``None`` if
    the allocator hands that address to none of ``tries`` same-size allocations."""
    held = []
    for _ in range(tries):
        t = torch.empty(row.nbytes // 4, dtype=torch.int32, pin_memory=row.pinned)
        if t.data_ptr() == row.address:
            t.fill_(CANARY)
            return t
        held.append(t)
    return None


def canary_written(canary: Optional[torch.Tensor], row: IndexRow) -> List[Tuple[int, int]]:
    """``(cell, value)`` of each cell of ``row`` in ``canary`` that is no longer ``CANARY``."""
    if canary is None:
        return []
    cells = canary[row.offset : row.offset + row.length].numpy()
    return [(int(i), int(cells[i])) for i in np.nonzero(cells != np.int32(CANARY))[0]]


def staging_kept(parts) -> bool:
    """Whether every part lies in memory the lender keeps."""
    kept = staging_memory(parts)
    return all(
        any(m.address <= p.address and p.address + p.nbytes <= m.address + m.nbytes for m in kept)
        for p in parts
    )


def pinned_range(address: int) -> Optional[Tuple[int, int]]:
    """``(start, nbytes)`` of the page-locked host allocation the driver records at ``address``;
    ``None`` when there is none, as after it was freed."""
    from cuda.bindings import driver

    attribute = driver.CUpointer_attribute
    err, kind = driver.cuPointerGetAttribute(attribute.CU_POINTER_ATTRIBUTE_MEMORY_TYPE, address)
    if err != driver.CUresult.CUDA_SUCCESS:
        return None
    assert kind == driver.CUmemorytype.CU_MEMORYTYPE_HOST, kind
    err, start = driver.cuPointerGetAttribute(
        attribute.CU_POINTER_ATTRIBUTE_RANGE_START_ADDR, address
    )
    assert err == driver.CUresult.CUDA_SUCCESS, err
    err, nbytes = driver.cuPointerGetAttribute(attribute.CU_POINTER_ATTRIBUTE_RANGE_SIZE, address)
    assert err == driver.CUresult.CUDA_SUCCESS, err
    return int(start), int(nbytes)


def lender_warnings(monkeypatch) -> List[str]:
    """Collects the warnings the lender module logs from now on."""
    from tensorrt_llm.logger import logger

    seen: List[str] = []

    def spy(original):
        def warn(*args, **kwargs):
            caller = inspect.currentframe().f_back
            if caller is not None and caller.f_globals.get("__name__", "").startswith(_SHARING):
                seen.append(" ".join(str(a) for a in args))
            return original(*args, **kwargs)

        return warn

    for name in ("warning", "warning_once"):
        monkeypatch.setattr(logger, name, spy(getattr(logger, name)))
    return seen


# -- threads ----------------------------------------------------------------------------------


def on_thread(call: Callable[[], object], name: str = "not-the-caller") -> dict:
    """Run ``call`` on a fresh thread, joined before returning; what it returned or raised."""
    box: dict = {}

    def run():
        try:
            box["value"] = call()
        except Exception as error:  # handed back to the test
            box["error"] = error

    thread = threading.Thread(target=run, name=name)
    thread.start()
    thread.join(120)
    assert not thread.is_alive()
    box["thread"] = thread.ident
    return box


def wait_until(predicate: Callable[[], bool], seconds: float = 30.0) -> bool:
    deadline = time.monotonic() + seconds
    while not predicate():
        if time.monotonic() > deadline:
            return False
        time.sleep(0.01)
    return True


_KIT = SimpleNamespace(
    TPB=TPB,
    WINDOW=WINDOW,
    MAX_SEQ_LEN=MAX_SEQ_LEN,
    SENTINEL=SENTINEL,
    POOL_TOKENS=POOL_TOKENS,
    CANARY=CANARY,
    DevicePages=DevicePages,
    Requests=Requests,
    ShutdownSpy=ShutdownSpy,
    StatsSpy=StatsSpy,
    admitted=admitted,
    canary_written=canary_written,
    chain_keys=chain_keys,
    closed=closed,
    digest=digest,
    events_report_done=events_report_done,
    fill_sentinel=fill_sentinel,
    gated_stream=gated_stream,
    held_stream=held_stream,
    host_bytes=host_bytes,
    index_row=index_row,
    kv=kv,
    lender_warnings=lender_warnings,
    make_manager=make_manager,
    make_request=make_request,
    num_layer_groups=num_layer_groups,
    on_thread=on_thread,
    overwrite_free_pages=overwrite_free_pages,
    page=page,
    pages=pages,
    pinned_range=pinned_range,
    pool_group_ids=pool_group_ids,
    pool_group_of=pool_group_of,
    pool_pages=pool_pages,
    prefill=prefill,
    published=published,
    reclaim=reclaim,
    relay=relay,
    retained=retained,
    stage=stage,
    stale_blocks=stale_blocks,
    staging_kept=staging_kept,
    staging_memory=staging_memory,
    taken_by_others=taken_by_others,
    tier_used=tier_used,
    warm_tier_moves=warm_tier_moves,
    wait_until=wait_until,
    whole_pool_goes_to_others=whole_pool_goes_to_others,
    windows=windows,
)


@pytest.fixture
def kit():
    """The helpers and oracles above, for test modules that cannot import this file by name."""
    return _KIT
