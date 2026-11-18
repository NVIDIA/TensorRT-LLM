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
"""The step-input stager's block-offset copy over a target/draft ``KVCacheManagerV2``.

A unified cache whose life cycles mix page sizes (FP8 target, BF16 draft) has one
attention-op table per pool group. The stager's copy of the batch's rows, full or
incremental, must leave the device table bit for bit as the manager's own copy leaves
it: the rows' every column, since a drafter in the manager reads a whole row for its
extent, and nothing outside the batch's rows. Each descriptor reader runs the same
descriptors: the per-field read, the 64-byte read with host-split pieces and on the
two-dimensional grid, and the parameter-block launch. With ``pool_group_tail_marks`` the
copy fills only the tail columns a row may not already hold as the bad-page word, and the
table still ends every step as the manager's own copy leaves it.
"""

from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep import (  # noqa: E501
    runtime,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.ops.runtime import (  # noqa: E501
    step_commit,
    step_commit_inline,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheLayerOwner,
    KVCacheLayerSpec,
    KVCacheManagerV2,
    Role,
)
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import BAD_PAGE_INDEX

TOKENS_PER_BLOCK = 4
WINDOW = 2 * TOKENS_PER_BLOCK
MAX_SEQ_LEN = 64
SENTINEL = -7
# (request id, capacity, history length, dummy): one request far past the window, two
# inside it, a CUDA-graph padding dummy, and a mid-length one.
REQUESTS = ((11, 40, 36, False), (12, 8, 0, False), (13, 4, 0, True), (14, 24, 20, False))

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

#: Descriptor readers: (``step_commit.configure``'s ``v2_mode``, ``inline_commit``).
READERS = {
    "per-field": ("", False),
    "64-byte-split": ("1", False),
    "64-byte-grid": ("nosplit", False),
    "parameter-block": ("", True),
}


@pytest.fixture(params=sorted(READERS))
def reader(request):
    """Builds staging blocks whose launches read the descriptors with one reader."""
    v2_mode, inline_commit = READERS[request.param]
    if inline_commit and not step_commit_inline.available():
        pytest.skip(f"no parameter-block launch: {step_commit_inline.unavailable_reason()}")
    step_commit.configure(v2_mode=v2_mode)
    try:
        yield lambda: step_commit.StepStaging(inline_commit=inline_commit)
    finally:
        step_commit.configure(v2_mode="")


def _layer_specs():
    target = KVCacheLayerSpec(
        global_layer_id=0,
        owner=KVCacheLayerOwner.TARGET,
        dtype=DataType.FP8,
        num_kv_heads=2,
        head_dim=16,
        buffer_roles=(Role.KEY, Role.VALUE),
        sliding_window_size=WINDOW,
    )
    specs = [
        replace(target, global_layer_id=i, sliding_window_size=None if i % 3 == 2 else WINDOW)
        for i in range(6)
    ]
    draft = replace(target, owner=KVCacheLayerOwner.DRAFT, dtype=DataType.BF16)
    return specs + [replace(draft, global_layer_id=6 + i) for i in range(2)]


@pytest.fixture
def manager():
    torch.cuda.init()
    manager = KVCacheManagerV2(
        KvCacheConfig(
            max_gpu_total_bytes=16 << 20,
            host_cache_size=0,
            enable_block_reuse=False,
            enable_swa_scratch_reuse=False,
        ),
        CacheType.SELF,
        num_layers=6,
        num_kv_heads=2,
        head_dim=16,
        tokens_per_block=TOKENS_PER_BLOCK,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=4,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.FP8,
        layer_specs=_layer_specs(),
    )
    for request_id, capacity, history, dummy in REQUESTS:
        cache = manager._create_kv_cache(request_id, None, None, is_dummy=dummy)
        assert cache.resume(manager._stream.cuda_stream)
        assert cache.resize(capacity, history)
    # (window, FP8 pages), (window, BF16 pages), (full, FP8 pages).
    assert manager._attention_op_pool_groups is not None
    assert manager.num_attention_op_pools == 3
    try:
        yield manager
    finally:
        manager.shutdown()


def _table(manager):
    return torch.full(
        (manager.num_attention_op_pools, 6, 2, manager.max_blocks_per_seq),
        SENTINEL,
        dtype=torch.int32,
        device="cuda",
    )


def _native(manager, request_ids):
    dst = _table(manager)
    manager.copy_batch_block_offsets(dst, request_ids, 1, 0, len(request_ids))
    torch.cuda.synchronize()
    return dst


def _staged(manager, dst, request_ids, memo, attn_metadata, staging=None, tail_marks=None):
    stager = runtime._PrepareStager(
        staging if staging is not None else step_commit.StepStaging(),
        Counter(),
        SimpleNamespace(),
        attn_metadata,
        None,
        blocks_memo=memo,
        tail_marks=tail_marks,
    )
    assert stager.block_offsets(manager, dst, request_ids, 1, 0, len(request_ids)), stager.declined
    # A step whose rows all match what the memo recorded has nothing to launch.
    stager.staging.launch()
    torch.cuda.synchronize()
    return stager


@pytest.mark.parametrize("incremental", [False, True], ids=["full", "incremental"])
def test_staged_copy_is_the_native_copy(manager, reader, incremental):
    request_ids = [11, 12, 13]
    want = _native(manager, request_ids)
    dst = _table(manager)
    memo = runtime._BlockOffsetsMemo() if incremental else None
    stager = _staged(manager, dst, request_ids, memo, SimpleNamespace(), reader())
    cols = stager.block_cols
    assert cols < manager.max_blocks_per_seq
    # The long request's windowed tables carry BAD_PAGE_INDEX inside the copied columns.
    assert (want[:, 0, :, :cols] == BAD_PAGE_INDEX).any()
    assert torch.equal(dst, want)


def test_incremental_steps_track_the_native_copy(manager):
    memo, attn_metadata = runtime._BlockOffsetsMemo(), SimpleNamespace()
    dst = _table(manager)
    # A batch, the same batch, a narrower batch in the same rows, then a request that grows.
    steps = ([11, 12, 13], [11, 12, 13], [14, 12, 13], [14, 12, 13])
    kept, copied = [], []
    for step, request_ids in enumerate(steps):
        if step == 3:
            assert manager.kv_cache_map[14].resize(32, 28)
        stager = _staged(manager, dst, request_ids, memo, attn_metadata)
        kept.append(stager.counts["block columns kept"])
        copied.append(stager.counts["block columns copied"])
        assert torch.equal(dst, _native(manager, request_ids)), step
    # The repeated batch copies nothing; the narrower one re-copies only what changed.
    assert copied[1] == 0 and kept[1] > 0
    assert kept[2] > 0 and copied[3] > 0


def test_a_new_attention_metadata_copies_every_column_again(manager):
    memo = runtime._BlockOffsetsMemo()
    dst = _table(manager)
    _staged(manager, dst, [11, 12, 13], memo, SimpleNamespace())
    dst.fill_(SENTINEL)
    _staged(manager, dst, [11, 12, 13], memo, SimpleNamespace())
    assert torch.equal(dst, _native(manager, [11, 12, 13]))


class _RowByRowMemo:
    """The memo as it was row by row (``width`` then ``changed_runs`` per device row), the
    reference ``_BlockOffsetsMemo.table_runs`` must reproduce."""

    def __init__(self):
        self._rows: dict = {}
        self._scope = None

    def forget(self):
        self._scope = None
        self._rows.clear()

    def open(self, manager, dst, attn_metadata):
        scope = (manager, attn_metadata, dst.data_ptr(), tuple(dst.shape))
        last = self._scope
        if (
            last is None
            or last[0] is not manager
            or last[1] is not attn_metadata
            or last[2:] != scope[2:]
        ):
            self._rows.clear()
        self._scope = scope

    def width(self, table, seq):
        entry = self._rows.get((table, seq))
        return None if entry is None else entry[3].shape[0]

    def changed_runs(self, table, seq, row, scale, kv_offset, pages):
        n = pages.shape[0]
        entry = self._rows.get((table, seq))
        if entry is None or entry[:3] != (row, scale, kv_offset):
            runs = [(0, n)] if n else []
        else:
            old = entry[3]
            m = min(old.shape[0], n)
            diff = np.flatnonzero(pages[:m] != old[:m])
            runs = []
            if diff.size:
                cuts = np.flatnonzero(np.diff(diff) > 1)
                starts = np.concatenate(([diff[0]], diff[cuts + 1]))
                ends = np.concatenate((diff[cuts], [diff[-1]])) + 1
                runs = [(int(a), int(b)) for a, b in zip(starts, ends)]
            if n > m:
                if runs and runs[-1][1] == m:
                    runs[-1] = (runs[-1][0], n)
                else:
                    runs.append((m, n))
            if len(runs) > runtime._MAX_BLOCK_RUNS:
                runs = [(runs[0][0], runs[-1][1])]
        self._rows[(table, seq)] = (row, scale, kv_offset, pages.copy())
        return runs

    def table_runs(self, keys, pages):
        tables, seqs, _ = pages.shape
        widths = np.full((tables, seqs), -1, dtype=np.int64)
        runs = []
        for table in range(tables):
            runs.append([])
            for seq in range(seqs):
                width = self.width(table, seq)
                widths[table, seq] = -1 if width is None else width
                row, scale, kv_offset = (int(v) for v in keys[table, seq])
                runs[-1].append(
                    self.changed_runs(table, seq, row, scale, kv_offset, pages[table, seq])
                )
        return widths, runs


def _checked_against_row_by_row(memo):
    """Make ``memo`` assert, at every call, the results of the row-by-row memo fed the same calls."""
    ref = _RowByRowMemo()
    forget, open_, table_runs = memo.forget, memo.open, memo.table_runs
    calls = Counter()

    def checked_forget():
        ref.forget()
        forget()

    def checked_open(manager, dst, attn_metadata):
        ref.open(manager, dst, attn_metadata)
        open_(manager, dst, attn_metadata)

    def checked_table_runs(keys, pages):
        want_widths, want_runs = ref.table_runs(keys, pages)
        widths, runs = table_runs(keys, pages)
        assert np.array_equal(widths, want_widths)
        assert runs == want_runs
        calls["rows"] += widths.size
        calls["rows unchanged"] += sum(not r for row in runs for r in row)
        return widths, runs

    memo.forget, memo.open, memo.table_runs = checked_forget, checked_open, checked_table_runs
    return calls


def test_table_runs_is_the_row_by_row_memo():
    """``table_runs`` over random steps (rows admitted, evicted, reordered, re-keyed; widths that
    grow and shrink; pages that change, some bad) gives every row's width and runs as the
    row-by-row memo does, and remembers the same."""
    rng = np.random.default_rng(0)
    tables, host_rows, num_blocks = 3, 40, 256
    steps = 0
    for _ in range(40):
        memo = runtime._BlockOffsetsMemo()
        calls = _checked_against_row_by_row(memo)
        host = rng.integers(-1, 5000, size=(tables, host_rows, num_blocks), dtype=np.int32)
        rows = [int(r) for r in rng.choice(host_rows, size=8, replace=False)]
        cols = int(rng.integers(1, num_blocks))
        for _ in range(60):
            draw = rng.random()
            if draw < 0.15 and len(rows) < 16:
                rows.append(int(rng.choice([r for r in range(host_rows) if r not in rows])))
            elif draw < 0.3 and len(rows) > 1:
                rows.pop(int(rng.integers(len(rows))))
            elif draw < 0.4:
                cols = int(np.clip(cols + rng.integers(-20, 40), 0, num_blocks))
            elif draw < 0.42:
                memo.forget()
            for _ in range(int(rng.integers(0, 6))):
                at = (rng.integers(tables), rng.integers(host_rows), rng.integers(num_blocks))
                host[at] = rng.integers(-1, 5000)
            rekey = 1 if rng.random() < 0.05 else 0
            keys = np.array(
                [[(row, 2 + t + rekey, 7 * t) for row in rows] for t in range(tables)],
                dtype=np.int64,
            ).reshape(tables, len(rows), 3)
            pages = host[:, np.asarray(rows, dtype=np.int64), :cols]
            memo.table_runs(keys, pages)
            steps += 1
        assert calls["rows"] > calls["rows unchanged"] > 0
    assert steps == 40 * 60


def _admit(manager, request_id, capacity, history):
    cache = manager._create_kv_cache(request_id, None, None)
    assert cache is not None and cache.resume(manager._stream.cuda_stream)
    assert cache.resize(capacity, history)


def _evict(manager, request_id):
    manager.free_resources(
        SimpleNamespace(py_request_id=request_id, request_id=request_id, is_dummy=False)
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_memo_tracks_admissions_evictions_and_growth(manager, seed):
    """Steps that repeat, reorder, admit (into a freed host row), evict and grow requests: the
    memoized copy leaves the batch's rows as the manager's own copy and as the copy without a
    memo, and its runs are the row-by-row memo's at every step."""
    rng = np.random.default_rng(seed)
    memo, attn_metadata = runtime._BlockOffsetsMemo(), SimpleNamespace()
    calls = _checked_against_row_by_row(memo)
    dst, full = _table(manager), _table(manager)
    # The dummy keeps its slot; the batch holds up to 3 of the other requests.
    batch = [11, 12, 14]
    sizes = {rid: (capacity, history) for rid, capacity, history, _ in REQUESTS}
    next_id, kept_steps = 20, 0
    for step in range(30):
        draw = rng.random()
        if draw < 0.2 and len(batch) > 1:
            _evict(manager, batch.pop(int(rng.integers(len(batch)))))
        elif draw < 0.4 and len(batch) < 3:
            capacity = int(rng.choice([4, 8, 24, 40]))
            sizes[next_id] = (capacity, int(rng.integers(0, capacity)))
            _admit(manager, next_id, *sizes[next_id])
            batch.insert(int(rng.integers(len(batch) + 1)), next_id)
            next_id += 1
        elif draw < 0.6:
            rid = batch[int(rng.integers(len(batch)))]
            capacity, history = sizes[rid]
            if capacity < MAX_SEQ_LEN:
                capacity = min(capacity + int(rng.choice([1, 4, 9])), MAX_SEQ_LEN)
                sizes[rid] = (capacity, max(history, capacity - 4))
                assert manager.kv_cache_map[rid].resize(*sizes[rid])
        elif draw < 0.7:
            rng.shuffle(batch)
        stager = _staged(manager, dst, batch, memo, attn_metadata)
        _staged(manager, full, batch, None, attn_metadata)
        want = _native(manager, batch)
        rows = len(batch)
        assert torch.equal(dst[:, :rows], want[:, :rows]), step
        assert torch.equal(full[:, :rows], want[:, :rows]), step
        kept_steps += stager.counts["block columns kept"] > 0
    # The memo did skip copies, and both the unchanged and the changed rows were checked.
    assert kept_steps > 0
    assert calls["rows"] > calls["rows unchanged"] > 0


@pytest.mark.parametrize("count", [4, 3 * 1024 + 5])
def test_blocks_descriptor_bad_page_word(reader, count):
    """Every reader writes a BLOCKS descriptor's ``bad_page`` word for a BAD_PAGE_INDEX page,
    on a row of one chunk and on one longer than several chunks (host-split pieces)."""
    pages = torch.arange(count, dtype=torch.int32)
    pages[1::3] = BAD_PAGE_INDEX
    pages = pages.pin_memory()
    for bad_page in (0, BAD_PAGE_INDEX):
        k = torch.full((count,), SENTINEL, dtype=torch.int32, device="cuda")
        v = torch.full((count,), SENTINEL, dtype=torch.int32, device="cuda")
        staging = reader()
        staging.blocks(k.data_ptr(), v.data_ptr(), pages.data_ptr(), count, 2, 1, bad_page)
        assert staging.launch()
        torch.cuda.synchronize()
        bad = pages.cuda() == BAD_PAGE_INDEX
        assert torch.equal(k, torch.where(bad, bad_page, 2 * pages.cuda()))
        assert torch.equal(v, torch.where(bad, bad_page, 2 * pages.cuda() + 1))
        staging.reset()


# --------------------------------------------------------------------------
# gint: the STEP inline commit (step_commit_v2__INLINE_COMMIT) and the split v2 pieces keep the
# BLOCKS bad-page word (field 7), as the pinned-table kernel does.
# --------------------------------------------------------------------------
@pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the inline commit kernel is built for sm107",
)
@pytest.mark.parametrize("bad_page", [0, BAD_PAGE_INDEX])
def test_inline_commit_writes_the_bad_page_word(bad_page):
    """The parameter-block launch (``INLINE_COMMIT``) writes a BLOCKS row's bad-page word as
    the pinned-table kernel does (field 7)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.ops.runtime import (  # noqa: E501
        step_commit_inline,
    )

    assert step_commit_inline.available(), step_commit_inline.unavailable_reason()
    pages = torch.tensor([3, BAD_PAGE_INDEX, 5, BAD_PAGE_INDEX, 0], dtype=torch.int32).pin_memory()
    out = {}
    for inline in (False, True):
        k = torch.full((5,), SENTINEL, dtype=torch.int32, device="cuda")
        v = torch.full((5,), SENTINEL, dtype=torch.int32, device="cuda")
        fill = torch.full((3,), SENTINEL, dtype=torch.int32, device="cuda")
        staging = step_commit.StepStaging(inline_commit=inline)
        staging.blocks(k.data_ptr(), v.data_ptr(), pages.data_ptr(), 5, 2, 1, bad_page)
        staging.fill(fill.data_ptr(), 1, 3, bad_page)
        assert staging.launch()
        torch.cuda.synchronize()
        # The parameter-block launch records no event; the pinned-table launch does.
        assert (staging._event is None) == inline, "the inline launch did not run"
        out[inline] = (k.tolist(), v.tolist(), fill.tolist())
    assert out[False] == out[True]
    assert out[True][0] == [6, bad_page, 10, bad_page, 0]
    assert out[True][1] == [7, bad_page, 11, bad_page, 1]


def test_blocks_descriptor_bad_page_word_in_split_pieces():
    """A BLOCKS descriptor longer than a chunk, cut into pieces (``split``): each keeps the word."""
    step_commit.configure(v2_mode="split")
    try:
        n = 3 * 1024 + 5
        host = torch.arange(n, dtype=torch.int32)
        host[::7] = BAD_PAGE_INDEX
        pages = host.pin_memory()
        k = torch.full((n,), SENTINEL, dtype=torch.int32, device="cuda")
        v = torch.full((n,), SENTINEL, dtype=torch.int32, device="cuda")
        staging = step_commit.StepStaging()
        staging.blocks(k.data_ptr(), v.data_ptr(), pages.data_ptr(), n, 3, 2, BAD_PAGE_INDEX)
        assert staging.launch()
        torch.cuda.synchronize()
        want_k = torch.where(host == BAD_PAGE_INDEX, BAD_PAGE_INDEX, host * 3)
        assert torch.equal(k.cpu(), want_k)
        assert torch.equal(
            v.cpu(), torch.where(host == BAD_PAGE_INDEX, BAD_PAGE_INDEX, host * 3 + 2)
        )
    finally:
        step_commit.configure(v2_mode="")


def test_tail_marked_steps_track_the_native_copy(manager, reader):
    marks, attn_metadata = runtime._BlockTailMarks(), SimpleNamespace()
    dst = _table(manager)
    # A batch, the same batch, a narrower batch in the same rows, a request that grows, and a
    # wider batch again.
    steps = ([11, 12, 13], [11, 12, 13], [14, 12, 13], [14, 12, 13], [11, 12, 13])
    filled = []
    for step, request_ids in enumerate(steps):
        if step == 3:
            assert manager.kv_cache_map[14].resize(32, 28)
        stager = _staged(manager, dst, request_ids, None, attn_metadata, reader(), marks)
        filled.append(stager.counts["block tail columns filled"])
        assert torch.equal(dst, _native(manager, request_ids)), step
    # The first step fills every row's tail; the repeated batch fills nothing.
    assert filled[0] > 0 and filled[1] == 0
