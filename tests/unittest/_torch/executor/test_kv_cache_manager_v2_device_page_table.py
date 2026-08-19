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
"""Tests for the device-resident KVCM V2 page-table path.

``copyBatchBlockOffsetsToDeviceKernel`` dereferences the host page table from
inside the kernel, which requires that allocation to be mapped into the GPU's
address space. Confidential Compute does not provide such a mapping, so on a CC
system the kernel faults on an address it can never read (NVBug 6248648).
``_BasePageTableMaterializer`` snapshots the stable base-page rows, moves the
same bytes as an ordinary H2D copy, and invokes the same expansion kernel from
device-resident data instead.
"""

from itertools import product
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor import kv_cache_manager_v2 as kvcm_v2
from tensorrt_llm._torch.pyexecutor.kv_cache_manager_v2 import (
    _DEVICE_PAGE_TABLE_ENV,
    KVCacheManagerV2,
    _BasePageTableMaterializer,
    _check_page_table_is_gpu_addressable,
    _use_device_page_table,
)
from tensorrt_llm._utils import prefer_pinned
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX
from tensorrt_llm.runtime.kv_cache_manager_v2._utils import init_cuda_once

_NUM_POOLS = 2
_CAPACITY = 5
# The native kernel requires the block count to be a multiple of its packed
# access width, so keep this a multiple of 4.
_MAX_BLOCKS_PER_SEQ = 8
_UNTOUCHED = -12345

_TOKENS_PER_BLOCK = 4
_MANAGER_MAX_SEQ_LEN = 16
_MANAGER_BATCH = 3
_TOKENS_PER_REQUEST = 12  # three allocated blocks per sequence

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA device")
# The native kernel loads host memory from the GPU, which is precisely what
# Confidential Compute forbids -- running it there is the crash this path exists
# to avoid, so there is nothing to compare against.
requires_native_page_table_kernel = pytest.mark.skipif(
    not torch.cuda.is_available() or not prefer_pinned(),
    reason="the native page-table kernel cannot run without GPU-addressable host memory",
)


def _make_host_table(pin: bool = False) -> torch.Tensor:
    """A V2 host table with distinct, partly-invalid plane-0 rows."""
    table = torch.zeros(
        (_NUM_POOLS, _CAPACITY, 2, _MAX_BLOCKS_PER_SEQ),
        dtype=torch.int32,
        pin_memory=pin,
    )
    for pool in range(_NUM_POOLS):
        for row in range(_CAPACITY):
            base = 100 * (pool + 1) + 10 * row
            table[pool, row, 0] = torch.arange(base, base + _MAX_BLOCKS_PER_SEQ, dtype=torch.int32)
    # Tail blocks of two rows are unallocated.
    table[:, 1, 0, 5:] = BAD_PAGE_INDEX
    table[:, 3, 0, 2:] = BAD_PAGE_INDEX
    # Plane 1 is part of the established owner ABI but is not a materialization source.
    table[:, :, 1, :] = 777
    return table


def _reference_offsets(
    host_table: torch.Tensor,
    copy_idx: torch.Tensor,
    index_scales: torch.Tensor,
    kv_offset: torch.Tensor,
) -> torch.Tensor:
    """Straight transcription of the C++ kernel's per-element arithmetic."""
    num_pools, num_seqs = index_scales.shape[0], copy_idx.shape[0]
    expected = torch.empty(
        (num_pools, num_seqs, 2, host_table.shape[3]),
        dtype=torch.int32,
    )
    for pool in range(num_pools):
        for seq, row in enumerate(copy_idx.tolist()):
            for block in range(host_table.shape[3]):
                page = int(host_table[pool, row, 0, block])
                if page == BAD_PAGE_INDEX:
                    key = value = 0
                else:
                    key = int(index_scales[pool]) * page
                    value = key + int(kv_offset[pool])
                expected[pool, seq, 0, block] = key
                expected[pool, seq, 1, block] = value
    return expected


def _make_materializer(host_table: torch.Tensor, index_scales, kv_offset):
    return _BasePageTableMaterializer(
        host_table,
        torch.cuda.current_stream(),
        index_scales,
        kv_offset,
    )


def _stub_materializer(host_table: torch.Tensor) -> _BasePageTableMaterializer:
    """Build the CPU-only materializer surface used by host-gather tests."""
    materializer = object.__new__(_BasePageTableMaterializer)
    materializer._host_block_offsets = host_table
    materializer._host_base_page_indices = host_table[:, :, 0, :]
    materializer._num_pools = host_table.shape[0]
    materializer._row_capacity = host_table.shape[1]
    materializer._max_blocks_per_seq = host_table.shape[3]
    materializer._use_device_staging = True
    materializer._use_device_expansion = True
    return materializer


@pytest.fixture
def scales():
    return (
        torch.tensor([2, 3][:_NUM_POOLS], dtype=torch.int32),
        torch.tensor([100, 200][:_NUM_POOLS], dtype=torch.int32),
    )


def test_materializer_gathers_only_canonical_base_rows():
    host_table = _make_host_table()
    copy_idx = torch.tensor([3, 0, 1], dtype=torch.int32)
    materializer = _stub_materializer(host_table)

    rows = materializer._gather_host_rows(
        copy_idx,
        out=torch.empty(
            _NUM_POOLS,
            copy_idx.shape[0],
            2,
            _MAX_BLOCKS_PER_SEQ,
            dtype=torch.int32,
        ),
    )

    assert rows.shape == (_NUM_POOLS, copy_idx.shape[0], 2, _MAX_BLOCKS_PER_SEQ)
    assert torch.equal(rows[:, :, 0], host_table[:, copy_idx.long(), 0, :])


@requires_cuda
def test_index_mapper_snapshot_gather_does_not_mutate_shared_copy_index():
    from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import IndexMapper

    index_mapper = IndexMapper(max_batch_size=2, max_beam_width=1)
    index_mapper.add_new_sequence(11)
    index_mapper.add_new_sequence(22)
    pending_copy_idx = index_mapper.get_copy_index([22, 11], 2, 1)
    expected_copy_idx = pending_copy_idx.clone()
    destination = torch.empty(
        (_NUM_POOLS, 2, 2, _MAX_BLOCKS_PER_SEQ),
        dtype=torch.int32,
    )

    index_mapper.gather_k_block_offsets(
        _make_host_table(),
        destination,
        [11, 22],
        _MAX_BLOCKS_PER_SEQ,
    )

    assert torch.equal(pending_copy_idx, expected_copy_idx)


@requires_cuda
def test_pitched_copy_moves_only_active_base_page_rows():
    from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
        copy_base_page_rows_to_device,
    )

    num_rows = 3
    source = _make_host_table()
    destination = torch.full(
        (_NUM_POOLS, _CAPACITY + 2, 2, _MAX_BLOCKS_PER_SEQ),
        _UNTOUCHED,
        dtype=torch.int32,
        device="cuda",
    )

    copy_base_page_rows_to_device(
        source,
        destination,
        num_rows,
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()

    assert torch.equal(destination[:, :num_rows, 0].cpu(), source[:, :num_rows, 0])
    assert torch.all(destination[:, :num_rows, 1] == _UNTOUCHED)
    assert torch.all(destination[:, num_rows:] == _UNTOUCHED)


def test_materializer_resizes_and_populates_reusable_host_staging():
    host_table = _make_host_table()
    copy_idx = torch.tensor([3, 0, 3], dtype=torch.int32)
    num_blocks = _MAX_BLOCKS_PER_SEQ // 2
    materializer = _stub_materializer(host_table)
    staging = torch.empty(
        _NUM_POOLS,
        _CAPACITY,
        2,
        num_blocks,
        dtype=torch.int32,
    )
    data_ptr = staging.data_ptr()
    was_pinned = staging.is_pinned()

    rows = materializer._gather_host_rows(copy_idx, num_blocks=num_blocks, out=staging)

    assert rows is staging
    assert staging.data_ptr() == data_ptr
    assert staging.is_pinned() == was_pinned
    assert staging.shape == (_NUM_POOLS, copy_idx.shape[0], 2, num_blocks)
    assert torch.equal(staging[:, :, 0], host_table[:, copy_idx.long(), 0, :num_blocks])

    # Re-grow after shrinking, still within the originally allocated storage.
    copy_idx = torch.tensor([4, 2, 0, 3], dtype=torch.int32)
    rows = materializer._gather_host_rows(copy_idx, num_blocks=num_blocks, out=staging)

    assert rows is staging
    assert staging.data_ptr() == data_ptr
    assert staging.is_pinned() == was_pinned
    assert staging.shape == (_NUM_POOLS, copy_idx.shape[0], 2, num_blocks)
    assert torch.equal(staging[:, :, 0], host_table[:, copy_idx.long(), 0, :num_blocks])


def test_materializer_rejects_host_staging_that_would_need_reallocation():
    host_table = _make_host_table()
    copy_idx = torch.tensor([3, 0, 1], dtype=torch.int32)
    materializer = _stub_materializer(host_table)
    staging = torch.empty(
        _NUM_POOLS,
        copy_idx.shape[0] - 1,
        2,
        _MAX_BLOCKS_PER_SEQ,
        dtype=torch.int32,
    )

    with pytest.raises(ValueError, match="requires"):
        materializer._gather_host_rows(copy_idx, out=staging)


@requires_native_page_table_kernel
def test_materializer_regrows_pinned_staging_without_reallocation():
    host_table = _make_host_table(pin=True)
    materializer = _stub_materializer(host_table)
    staging = torch.empty(
        _NUM_POOLS,
        _CAPACITY,
        2,
        _MAX_BLOCKS_PER_SEQ,
        dtype=torch.int32,
        pin_memory=True,
    )
    data_ptr = staging.data_ptr()

    materializer._gather_host_rows(torch.tensor([1], dtype=torch.int32), out=staging)
    materializer._gather_host_rows(
        torch.tensor([4, 2, 0, 3], dtype=torch.int32),
        out=staging,
    )

    assert staging.data_ptr() == data_ptr
    assert staging.is_pinned()


def test_materializer_rejects_noncontiguous_host_staging():
    host_table = _make_host_table()
    copy_idx = torch.tensor([3, 0, 1], dtype=torch.int32)
    materializer = _stub_materializer(host_table)
    staging = torch.empty(
        _NUM_POOLS,
        copy_idx.shape[0],
        2,
        _MAX_BLOCKS_PER_SEQ * 2,
        dtype=torch.int32,
    )[:, :, :, ::2]

    with pytest.raises(AssertionError, match="contiguous, base-aligned staging tensor"):
        materializer._gather_host_rows(copy_idx, out=staging)


def test_native_materializer_rejects_noncontiguous_source():
    from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
        copy_batch_block_offsets_to_device,
    )

    source = torch.empty(
        _NUM_POOLS,
        _CAPACITY,
        2,
        _MAX_BLOCKS_PER_SEQ * 2,
        dtype=torch.int32,
    )[:, :, :, ::2]

    with pytest.raises(RuntimeError, match="input must be contiguous"):
        copy_batch_block_offsets_to_device(
            source,
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            0,
        )


def test_native_materializer_rejects_non_cuda_destination():
    from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
        copy_batch_block_offsets_to_device,
    )

    with pytest.raises(RuntimeError, match="output must be a CUDA tensor"):
        copy_batch_block_offsets_to_device(
            _make_host_table(),
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            torch.empty(0, dtype=torch.int32),
            0,
        )


@requires_cuda
@pytest.mark.parametrize(
    "cuda_sources",
    list(product((False, True), repeat=4)),
    ids=lambda values: "".join("d" if value else "h" for value in values),
)
def test_native_materializer_accepts_each_gpu_readable_source_independently(cuda_sources):
    from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
        copy_batch_block_offsets_to_device,
    )

    requires_host_mapping = not all(cuda_sources)
    if requires_host_mapping and not prefer_pinned():
        pytest.skip("host source tensors are not GPU-addressable")

    host_table = _make_host_table(pin=requires_host_mapping)
    copy_idx = torch.tensor([3, 0, 1], dtype=torch.int32, pin_memory=requires_host_mapping)
    index_scales = torch.tensor([2, 3], dtype=torch.int32, pin_memory=requires_host_mapping)
    kv_offset = torch.tensor([100, 200], dtype=torch.int32, pin_memory=requires_host_mapping)
    host_sources = (host_table, copy_idx, index_scales, kv_offset)
    sources = [
        source.cuda() if on_cuda else source for source, on_cuda in zip(host_sources, cuda_sources)
    ]
    output = torch.empty(
        (_NUM_POOLS, copy_idx.shape[0], 2, _MAX_BLOCKS_PER_SEQ),
        dtype=torch.int32,
        device="cuda",
    )

    copy_batch_block_offsets_to_device(
        sources[0],
        output,
        sources[1],
        sources[2],
        sources[3],
        torch.cuda.current_stream().cuda_stream,
    )
    torch.cuda.synchronize()

    assert torch.equal(
        output.cpu(),
        _reference_offsets(host_table, copy_idx, index_scales, kv_offset),
    )


@requires_cuda
@pytest.mark.parametrize("pageable_source", ("input", "copy_index", "index_scales", "kv_offset"))
def test_native_materializer_validates_pageable_cpu_sources_before_launch(pageable_source):
    """Pageable memory may be readable on coherent systems; otherwise fail before launch."""
    from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
        copy_batch_block_offsets_to_device,
    )

    host_table = _make_host_table()
    copy_idx = torch.tensor([3, 0, 1], dtype=torch.int32)
    index_scales = torch.tensor([2, 3], dtype=torch.int32)
    kv_offset = torch.tensor([100, 200], dtype=torch.int32)
    host_sources = {
        "input": host_table,
        "copy_index": copy_idx,
        "index_scales": index_scales,
        "kv_offset": kv_offset,
    }
    sources = {name: source.cuda() for name, source in host_sources.items()}
    sources[pageable_source] = host_sources[pageable_source]
    output = torch.empty(
        (_NUM_POOLS, copy_idx.shape[0], 2, _MAX_BLOCKS_PER_SEQ),
        dtype=torch.int32,
        device="cuda",
    )

    try:
        copy_batch_block_offsets_to_device(
            sources["input"],
            output,
            sources["copy_index"],
            sources["index_scales"],
            sources["kv_offset"],
            torch.cuda.current_stream().cuda_stream,
        )
    except RuntimeError as error:
        assert pageable_source in str(error)
        assert "not readable by output device" in str(error)
    else:
        torch.cuda.synchronize()
        assert torch.equal(
            output.cpu(),
            _reference_offsets(host_table, copy_idx, index_scales, kv_offset),
        )


class _StubTable:
    """Stands in for the page table: only its pinned-ness is consulted."""

    def __init__(self, pinned: bool):
        self._pinned = pinned

    def is_pinned(self) -> bool:
        return self._pinned


class TestUseDevicePageTable:
    """``auto`` must follow the kernel's contract, not a hardcoded default."""

    def test_manager_predicate_reports_staging_without_expansion(self):
        manager = object.__new__(KVCacheManagerV2)
        manager._page_table_materializer = SimpleNamespace(
            uses_device_staging=True,
            uses_device_expansion=False,
        )

        assert manager.uses_device_page_table

    def test_snapshot_rejects_staging_without_expansion(self):
        manager = object.__new__(KVCacheManagerV2)
        manager._page_table_materializer = SimpleNamespace(
            uses_device_staging=True,
            uses_device_expansion=False,
        )

        with pytest.raises(RuntimeError, match="requires device expansion metadata"):
            manager.materialize_block_offsets_snapshot(torch.empty(0), [])

    @pytest.mark.parametrize("pinned", [True, False])
    @pytest.mark.parametrize("policy_pins", [True, False])
    def test_auto_follows_the_pinning_policy(self, monkeypatch, pinned, policy_pins):
        monkeypatch.delenv(_DEVICE_PAGE_TABLE_ENV, raising=False)
        monkeypatch.setattr(kvcm_v2, "prefer_pinned", lambda: policy_pins)

        # The native kernel is only safe when the policy pins *and* this table
        # really is pinned; CC turns prefer_pinned() off for exactly that reason.
        assert _use_device_page_table(_StubTable(pinned)) is not (policy_pins and pinned)

    @pytest.mark.parametrize(
        "setting, expected",
        [("1", True), ("true", True), ("ON", True), ("0", False), ("no", False), (" Yes ", True)],
    )
    def test_explicit_override(self, monkeypatch, setting, expected):
        monkeypatch.setenv(_DEVICE_PAGE_TABLE_ENV, setting)
        # An explicit setting wins regardless of how the table was allocated.
        assert _use_device_page_table(_StubTable(pinned=True)) is expected

    def test_unparseable_setting_is_rejected(self, monkeypatch):
        monkeypatch.setenv(_DEVICE_PAGE_TABLE_ENV, "maybe")
        with pytest.raises(ValueError, match=_DEVICE_PAGE_TABLE_ENV):
            _use_device_page_table(_StubTable(pinned=True))


@pytest.mark.parametrize(
    "name",
    ["host_kv_cache_block_offsets", "copy_idx", "index_scales", "kv_offset"],
)
def test_pageable_inputs_to_the_native_kernel_fail_fast(name):
    """Every host input the GPU dereferences must be checked explicitly."""
    pageable = torch.zeros(4, dtype=torch.int32)
    inputs = {
        "host_kv_cache_block_offsets": _StubTable(pinned=True),
        "copy_idx": _StubTable(pinned=True),
        "index_scales": _StubTable(pinned=True),
        "kv_offset": _StubTable(pinned=True),
    }
    inputs[name] = pageable
    with pytest.raises(RuntimeError, match=rf"{name}.*pageable"):
        _check_page_table_is_gpu_addressable(**inputs)


@requires_native_page_table_kernel
def test_materializer_matches_the_native_kernel(monkeypatch, scales):
    """The device path must be a drop-in for ``copy_batch_block_offsets_to_device``."""
    from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
        copy_batch_block_offsets_to_device,
    )

    monkeypatch.setenv(_DEVICE_PAGE_TABLE_ENV, "1")
    index_scales, kv_offset = (t.pin_memory() for t in scales)
    host_table = _make_host_table(pin=True)
    copy_idx = torch.tensor([3, 0, 1], dtype=torch.int32).pin_memory()
    num_seqs = copy_idx.shape[0]
    # More destination rows than sequences, so "rows past the batch are left
    # alone" is observable.
    dst_shape = (_NUM_POOLS, num_seqs + 2, 2, _MAX_BLOCKS_PER_SEQ)

    native = torch.full(dst_shape, _UNTOUCHED, dtype=torch.int32, device="cuda")
    copy_batch_block_offsets_to_device(
        host_table,
        native,
        copy_idx,
        index_scales,
        kv_offset,
        torch.cuda.current_stream().cuda_stream,
    )

    uploaded = torch.full(dst_shape, _UNTOUCHED, dtype=torch.int32, device="cuda")
    _make_materializer(host_table, index_scales, kv_offset).copy_block_offsets_to(
        uploaded, copy_idx
    )
    torch.cuda.synchronize()

    assert torch.equal(uploaded, native)
    assert torch.equal(
        uploaded[:, :num_seqs].cpu(),
        _reference_offsets(host_table, copy_idx, index_scales, kv_offset),
    )
    assert torch.all(uploaded[:, num_seqs:] == _UNTOUCHED)


@requires_cuda
def test_materializer_rereads_the_table_every_step(monkeypatch, scales):
    """Each step must re-gather; rows are edited in place and slots are reused."""
    monkeypatch.setenv(_DEVICE_PAGE_TABLE_ENV, "1")
    index_scales, kv_offset = scales
    host_table = _make_host_table()
    copy_idx = torch.tensor([2, 4], dtype=torch.int32)
    materializer = _make_materializer(host_table, index_scales, kv_offset)
    output = torch.empty((_NUM_POOLS, 2, 2, _MAX_BLOCKS_PER_SEQ), dtype=torch.int32, device="cuda")
    device_sources = []
    copy_rows = kvcm_v2.copy_base_page_rows_to_device

    def record_device_source(host_rows, device_rows, num_rows, stream):
        device_sources.append(device_rows.data_ptr())
        return copy_rows(host_rows, device_rows, num_rows, stream)

    monkeypatch.setattr(kvcm_v2, "copy_base_page_rows_to_device", record_device_source)

    def gather_and_check():
        materializer.copy_block_offsets_to(output, copy_idx)
        torch.cuda.synchronize()
        assert torch.equal(
            output.cpu(), _reference_offsets(host_table, copy_idx, index_scales, kv_offset)
        )

    gather_and_check()

    # An active row grows by a block, the way a generation step extends a cache.
    host_table[0, 2, 0, 4] = 4242
    gather_and_check()

    # A row is rewritten while it is *not* in the batch, then re-enters it: a
    # slot recycled by a new request. A cached mirror of the table would serve
    # the previous occupant's blocks here.
    host_table[:, 4, 0, :] = 55
    copy_idx = torch.tensor([4, 2], dtype=torch.int32)
    gather_and_check()

    assert set(device_sources) == {materializer._device_base_page_rows.data_ptr()}


@requires_cuda
def test_materializer_snapshots_rows_before_the_table_moves_on(monkeypatch, scales):
    """The H2D source must not be the live table, nor a reused staging buffer.

    The overlap scheduler runs the host a full iteration ahead, so iteration
    N+1 can rewrite the page table before iteration N's copy has drained. This
    stalls the stream long enough for the host to win that race outright. The
    cycle count is deliberately generous so the window stays open on fast GPUs;
    it mirrors the V1 regression test for nvbug 6293536.
    """
    monkeypatch.setenv(_DEVICE_PAGE_TABLE_ENV, "1")
    index_scales, kv_offset = scales
    host_table = _make_host_table()
    copy_idx = torch.tensor([1], dtype=torch.int32)
    materializer = _make_materializer(host_table, index_scales, kv_offset)
    first = torch.empty((_NUM_POOLS, 1, 2, _MAX_BLOCKS_PER_SEQ), dtype=torch.int32, device="cuda")
    second = torch.empty_like(first)

    torch.cuda._sleep(2_000_000_000)
    materializer.copy_block_offsets_to(first, copy_idx)
    expected_first = _reference_offsets(host_table, copy_idx, index_scales, kv_offset)

    host_table[:, 1, 0, :] = 31
    materializer.copy_block_offsets_to(second, copy_idx)
    expected_second = _reference_offsets(host_table, copy_idx, index_scales, kv_offset)
    torch.cuda.synchronize()

    assert torch.equal(first.cpu(), expected_first)
    assert torch.equal(second.cpu(), expected_second)


@requires_cuda
def test_materializer_leaves_the_destination_alone_for_an_empty_batch(monkeypatch, scales):
    """The native launcher early-returns at ``numSeqs == 0``; so must this."""
    monkeypatch.setenv(_DEVICE_PAGE_TABLE_ENV, "1")
    index_scales, kv_offset = scales
    materializer = _make_materializer(_make_host_table(), index_scales, kv_offset)
    output = torch.full(
        (_NUM_POOLS, 2, 2, _MAX_BLOCKS_PER_SEQ), _UNTOUCHED, dtype=torch.int32, device="cuda"
    )

    materializer.copy_block_offsets_to(output, torch.empty(0, dtype=torch.int32))
    torch.cuda.synchronize()

    assert torch.all(output == _UNTOUCHED)


def _build_manager(*, enable_swa_scratch_reuse: bool = False) -> KVCacheManagerV2:
    """A small manager; ``_DEVICE_PAGE_TABLE_ENV`` is read during construction."""
    init_cuda_once()
    return KVCacheManagerV2(
        KvCacheConfig(
            enable_block_reuse=False,
            max_gpu_total_bytes=16 << 20,
            # Two distinct attention windows, so the layers land in separate
            # pools and the per-pool index_scales/kv_offset are exercised.
            max_attention_window=[_MANAGER_MAX_SEQ_LEN, _TOKENS_PER_BLOCK],
            enable_swa_scratch_reuse=enable_swa_scratch_reuse,
        ),
        CacheType.SELF,
        num_layers=2,
        num_kv_heads=2,
        head_dim=64,
        tokens_per_block=_TOKENS_PER_BLOCK,
        max_seq_len=_MANAGER_MAX_SEQ_LEN,
        max_batch_size=_MANAGER_BATCH,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.HALF,
        enable_stats=False,
    )


@requires_native_page_table_kernel
def test_manager_keeps_the_native_path_on_gpu_addressable_hosts(monkeypatch):
    """``auto`` must not switch ordinary (non-CC) deployments onto the new path."""
    monkeypatch.delenv(_DEVICE_PAGE_TABLE_ENV, raising=False)
    manager = _build_manager()
    try:
        assert not manager._page_table_materializer.uses_device_expansion
        assert not manager._page_table_materializer.uses_device_staging
        assert manager._page_table_materializer._device_base_page_rows is None
        assert manager.host_kv_cache_block_offsets.ndim == 4
        assert manager.host_kv_cache_block_offsets.shape[2] == 2

        request_ids = [1]
        assert (
            manager.add_dummy_requests(
                request_ids=request_ids,
                token_nums=[_TOKENS_PER_REQUEST],
                prepare_resource=True,
            )
            is not None
        )

        def fail_if_cc_path_runs(*args, **kwargs):
            pytest.fail("the native path entered CC-only host gathering")

        monkeypatch.setattr(
            manager._page_table_materializer,
            "copy_block_offsets_to",
            fail_if_cc_path_runs,
        )
        output = torch.full(
            (manager.num_pools, 1, 2, manager.max_blocks_per_seq),
            _UNTOUCHED,
            dtype=torch.int32,
            device="cuda",
        )
        manager.copy_batch_block_offsets(output, request_ids, 1, 1, 1)
        torch.cuda.synchronize()
        assert (output != _UNTOUCHED).any()
    finally:
        manager.shutdown()


@requires_native_page_table_kernel
def test_forced_device_page_table_matches_the_native_kernel(monkeypatch):
    """Exercise the path CC will take, on hardware that has no CC.

    CI has no CC-enabled machines, so ``auto`` never selects the device path
    there and it would ship untested. Forcing the override runs the whole
    production dispatch -- manager construction, the index mapper's copy_idx,
    the live page table -- and diffs it against the native kernel on that same
    state, which is the comparison a CC machine cannot make for itself.
    """
    from tensorrt_llm.bindings.internal.batch_manager.kv_cache_manager_v2_utils import (
        copy_batch_block_offsets_to_device,
    )

    monkeypatch.setenv(_DEVICE_PAGE_TABLE_ENV, "1")
    manager = _build_manager()
    try:
        assert manager._page_table_materializer.uses_device_expansion, (
            "the override did not take effect"
        )
        assert manager._page_table_materializer.uses_device_staging
        request_ids = list(range(1, 1 + _MANAGER_BATCH))
        # Returns None if the cache could not seat the batch, which would leave
        # an empty page table and make the comparison below meaningless.
        assert (
            manager.add_dummy_requests(
                request_ids=request_ids,
                token_nums=[_TOKENS_PER_REQUEST] * _MANAGER_BATCH,
                prepare_resource=True,
            )
            is not None
        )

        def empty_offsets():
            return torch.full(
                (manager.num_pools, _MANAGER_BATCH, 2, manager.max_blocks_per_seq),
                _UNTOUCHED,
                dtype=torch.int32,
                device="cuda",
            )

        # The native kernel reads copy_idx when it runs, and get_copy_index
        # hands back a reused buffer -- so drain it before asking for the next
        # one below.
        native = empty_offsets()
        copy_batch_block_offsets_to_device(
            manager.host_kv_cache_block_offsets,
            native,
            manager.index_mapper.get_copy_index(request_ids, _MANAGER_BATCH, 1),
            manager.index_scales,
            manager.kv_offset,
            manager._stream.cuda_stream,
        )
        torch.cuda.synchronize()

        uploaded = empty_offsets()
        manager.copy_batch_block_offsets(uploaded, request_ids, 1, _MANAGER_BATCH, _MANAGER_BATCH)
        torch.cuda.synchronize()

        assert torch.equal(uploaded, native)

        snapshot = empty_offsets()
        snapshot_host = torch.empty(
            (
                manager.num_pools,
                _MANAGER_BATCH,
                2,
                manager.max_blocks_per_seq,
            ),
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        manager.materialize_block_offsets_snapshot(
            snapshot,
            request_ids,
            host_staging=snapshot_host,
            stream=manager._stream,
        )
        torch.cuda.synchronize()

        assert torch.equal(snapshot, native)
        # Guard the setup itself: an all-sentinel or all-zero table would make
        # the comparison above vacuous.
        assert (uploaded != _UNTOUCHED).any() and (uploaded != 0).any()
    finally:
        manager.shutdown()


@requires_native_page_table_kernel
def test_forced_device_page_table_matches_non_cc_swa(monkeypatch):
    """Compare SWA's CC transport against its established non-CC path."""
    request_ids = [1, 2]
    num_seqs = len(request_ids)

    def materialize(setting: str) -> torch.Tensor:
        monkeypatch.setenv(_DEVICE_PAGE_TABLE_ENV, setting)
        manager = _build_manager(enable_swa_scratch_reuse=True)
        try:
            assert manager.enable_swa_scratch_reuse
            assert manager._device_kv_cache_block_offsets_input.ndim == 4
            assert manager._page_table_materializer.uses_device_expansion is (setting == "1")
            if setting == "1":
                # Device staging is allocated lazily on the first copy.
                assert manager._page_table_materializer._device_base_page_rows is None
            assert (
                manager.add_dummy_requests(
                    request_ids=request_ids,
                    token_nums=[_TOKENS_PER_REQUEST] * num_seqs,
                    prepare_resource=True,
                )
                is not None
            )
            output = torch.full(
                (
                    manager.num_attention_op_pools,
                    num_seqs,
                    2,
                    manager.max_blocks_per_seq,
                ),
                _UNTOUCHED,
                dtype=torch.int32,
                device="cuda",
            )
            manager.copy_batch_block_offsets(
                output,
                request_ids,
                beam_width=1,
                num_contexts=num_seqs,
                num_seqs=num_seqs,
            )
            torch.cuda.synchronize()
            if setting == "1":
                assert torch.all(manager._device_kv_cache_block_offsets_input[:, num_seqs:] == 0)
            return output.cpu()
        finally:
            manager.shutdown()

    non_cc = materialize("0")
    forced_cc = materialize("1")

    assert torch.equal(forced_cc, non_cc)
    assert (forced_cc != _UNTOUCHED).any() and (forced_cc != 0).any()


@requires_cuda
def test_materializer_rejects_a_mismatched_destination(scales):
    index_scales, kv_offset = scales
    materializer = _make_materializer(_make_host_table(), index_scales, kv_offset)
    copy_idx = torch.tensor([0], dtype=torch.int32)

    # A wider destination would read beyond the canonical stable rows.
    with pytest.raises(AssertionError, match="blocks per sequence"):
        materializer.copy_block_offsets_to(
            torch.zeros(
                (_NUM_POOLS, 1, 2, _MAX_BLOCKS_PER_SEQ * 2), dtype=torch.int32, device="cuda"
            ),
            copy_idx,
        )
    with pytest.raises(AssertionError, match="pools"):
        materializer.copy_block_offsets_to(
            torch.zeros((1, 1, 2, _MAX_BLOCKS_PER_SEQ), dtype=torch.int32, device="cuda"), copy_idx
        )


@requires_cuda
def test_materializer_rejects_non_cuda_and_unaligned_destinations(scales):
    index_scales, kv_offset = scales
    materializer = _make_materializer(_make_host_table(), index_scales, kv_offset)
    copy_idx = torch.tensor([0], dtype=torch.int32)
    assert materializer.uses_device_expansion

    with pytest.raises(AssertionError):
        materializer.copy_block_offsets_to(
            torch.zeros((_NUM_POOLS, 1, 2, _MAX_BLOCKS_PER_SEQ), dtype=torch.int32),
            copy_idx,
        )
    with pytest.raises(AssertionError, match="multiple of 4"):
        materializer.copy_block_offsets_to(
            torch.zeros((_NUM_POOLS, 1, 2, 2), dtype=torch.int32, device="cuda"),
            copy_idx,
        )
