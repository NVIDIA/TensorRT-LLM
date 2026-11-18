# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The drafter page-table decode in one launch (``drafter_block_table_decode``, opt-in).

``block_table.refresh`` must leave the worker's page table and page counts bit for bit as
the built-in ``DFlashWorker._refresh_ctx_block_tables`` leaves them: every column of the
batch's rows, the extent of each row (past a sliding window's evicted placeholders), and
nothing outside the batch's rows. Checked over random page maps at the generation server's
shapes (up to 96 requests, a 31,180-column table), inside a captured CUDA graph, through
the context-K/V fork's worker on a step the fork did not launch, through the side chain's
decode where ``paged_ctx_block_tables`` leaves it, and over a unified target/draft
``KVCacheManagerV2``.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.kernels.context_kv_fork import (  # noqa: E501
    block_table,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.kernels.context_kv_fork import (  # noqa: E501
    fork as context_kv_fork,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    KVCacheLayerOwner,
    KVCacheLayerSpec,
    KVCacheManagerV2,
    Role,
)
from tensorrt_llm._torch.speculative.dflash import DFlashWorker
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import KvCacheConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

BAD_PAGE_INDEX = -1
#: The generation server's drafter page table: its 97 rows (96 requests and the dummy) and
#: its width in pages.
SLOTS, WIDTH = 97, 31180


def _page_map(gen, pools, rows, width, offset, window):
    """``[pools, rows, 2, width]`` encoded offsets: per row a live tail ``[lo, extent)`` of
    random encoded pages behind evicted placeholders, some entries below ``offset`` (a
    negative encoded page, not a placeholder), an empty row, and a row live to its end."""
    src = torch.full((pools, rows, 2, width), BAD_PAGE_INDEX, dtype=torch.int32)
    for row in range(rows):
        extent = int(torch.randint(0, width + 1, (1,), generator=gen))
        if row == 1:
            extent = 0
        elif row == 2:
            extent = width
        lo = max(0, extent - int(torch.randint(1, window + 1, (1,), generator=gen)))
        live = extent - lo
        if live:
            pages = torch.randint(0, 1 << 22, (pools, 2, live), generator=gen, dtype=torch.int32)
            pages[..., torch.randint(0, live, (max(1, live // 16),), generator=gen)] = (
                BAD_PAGE_INDEX
            )
            if offset:
                low = torch.randint(0, live, (max(1, live // 32),), generator=gen)
                pages[..., low] = torch.randint(0, offset, (len(low),), generator=gen).int()
            src[:, row, :, lo:extent] = pages
    return src.cuda()


def _worker(src, *, pool_idx, offset, divisor, unified=True, slots=SLOTS):
    """A DFlash worker bound to the manager's pages, its table and counts holding stale words."""
    width = src.shape[3]
    gen = torch.Generator().manual_seed(7)
    worker = object.__new__(DFlashWorker)
    worker._ctx_unified_cache = unified
    worker._ctx_pool_idx = pool_idx
    worker._ctx_block_offset = offset
    worker._ctx_block_divisor = divisor
    worker._ctx_block_tables = torch.randint(
        -(1 << 30), 1 << 30, (slots, width), generator=gen, dtype=torch.int32
    ).cuda()
    worker._ctx_block_counts = torch.randint(-5, 5, (slots,), generator=gen).cuda()
    worker._ctx_block_cols = torch.arange(1, width + 1, dtype=torch.long, device="cuda")
    return worker


def _twin(worker):
    other = object.__new__(DFlashWorker)
    other.__dict__.update(worker.__dict__)
    other._ctx_block_tables = worker._ctx_block_tables.clone()
    other._ctx_block_counts = worker._ctx_block_counts.clone()
    return other


def _metadata(src, unified=True):
    name = "kv_cache_block_offsets" if unified else "draft_kv_cache_block_offsets"
    return SimpleNamespace(**{name: src})


def _assert_same(got, want):
    assert torch.equal(got._ctx_block_counts, want._ctx_block_counts)
    assert torch.equal(got._ctx_block_tables, want._ctx_block_tables)


@pytest.mark.parametrize(
    "num_seqs,width,offset,divisor,unified",
    [
        (1, 7, 0, 2, True),
        (5, 1000, 3, 10, True),
        (40, 4097, 1, 1, False),
        (96, WIDTH, 0, 2, True),
        (97, WIDTH, 5, 10, True),
    ],
)
def test_the_decode_is_the_builtin_decode(num_seqs, width, offset, divisor, unified):
    gen = torch.Generator().manual_seed(num_seqs)
    src = _page_map(gen, 3, SLOTS, width, offset, window=130)
    want = _worker(src, pool_idx=2, offset=offset, divisor=divisor, unified=unified)
    got = _twin(want)
    metadata = _metadata(src, unified)
    assert DFlashWorker._refresh_ctx_block_tables(want, metadata, num_seqs)
    assert block_table.refresh(got, metadata, num_seqs) is True
    torch.cuda.synchronize()
    _assert_same(got, want)


def test_what_the_builtin_answers_is_left_to_it():
    src = _page_map(torch.Generator().manual_seed(0), 3, 4, 16, 0, window=8)
    worker = _worker(src, pool_idx=2, offset=0, divisor=2, slots=4)
    declined = []
    # No rows, no source: the built-in returns False or raises.
    assert block_table.refresh(worker, _metadata(src), 0, declined.append) is None
    assert block_table.refresh(worker, SimpleNamespace(), 2, declined.append) is None
    unbound = _twin(worker)
    unbound._ctx_block_tables = None
    assert block_table.refresh(unbound, _metadata(src), 2, declined.append) is None
    assert not declined
    # A source whose columns are not contiguous: declined with its reason.
    strided = src.permute(3, 2, 1, 0).contiguous().permute(3, 2, 1, 0)
    assert block_table.refresh(worker, _metadata(strided), 2, declined.append) is None
    assert declined == ["the attention-op table is not a CUDA int32 [pools, rows, 2, blocks] table"]


def test_the_decode_replays_in_a_cuda_graph():
    gen = torch.Generator().manual_seed(1)
    src = _page_map(gen, 3, SLOTS, WIDTH, 2, window=130)
    want = _worker(src, pool_idx=1, offset=2, divisor=4)
    got = _twin(want)
    metadata = _metadata(src)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        assert block_table.refresh(got, metadata, 64)  # compiled before the capture
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert block_table.refresh(got, metadata, 64)
    for step in range(3):
        src.copy_(_page_map(gen, 3, SLOTS, WIDTH, 2, window=130))
        graph.replay()
        DFlashWorker._refresh_ctx_block_tables(want, metadata, 64)
        torch.cuda.synchronize()
        _assert_same(got, want)


@pytest.mark.parametrize("lever", [False, True], ids=["off", "on"])
def test_the_fork_worker_decodes_with_the_lever(lever, monkeypatch):
    gen = torch.Generator().manual_seed(2)
    src = _page_map(gen, 3, SLOTS, WIDTH, 0, window=130)
    want = _worker(src, pool_idx=2, offset=0, divisor=2)
    got = _twin(want)
    got.__class__ = context_kv_fork.ContextKvForkDFlashWorker
    fork = object.__new__(context_kv_fork.ContextKvFork)
    fork.worker, fork._launched, fork.block_table_on = got, None, lever
    declines = []
    fork.declined = lambda *args: declines.append(args)
    got._kv_fork = fork
    launches = []
    decode = block_table.decode
    monkeypatch.setattr(
        block_table, "decode", lambda *args: (launches.append(args[2]), decode(*args))[1]
    )
    metadata = _metadata(src)
    assert got._refresh_ctx_block_tables(metadata, 96)
    assert DFlashWorker._refresh_ctx_block_tables(want, metadata, 96)
    torch.cuda.synchronize()
    _assert_same(got, want)
    assert launches == ([96] if lever else [])
    assert not declines


PAGE = 16
REQUEST_IDS = range(31, 37)
CAPACITIES, HISTORIES = (200, 40, 16, 64, 120, 32), (180, 24, 0, 40, 100, 9)


def _unified_manager(num_draft, nkv, head_dim):
    """Six FP8 target layers and BF16 draft ones in one cache, all windowed."""
    target = KVCacheLayerSpec(
        global_layer_id=0,
        owner=KVCacheLayerOwner.TARGET,
        dtype=DataType.FP8,
        num_kv_heads=nkv,
        head_dim=head_dim,
        buffer_roles=(Role.KEY, Role.VALUE),
        sliding_window_size=4 * PAGE,
    )
    specs = [replace(target, global_layer_id=i) for i in range(6)]
    draft = replace(target, owner=KVCacheLayerOwner.DRAFT, dtype=DataType.BF16)
    specs += [replace(draft, global_layer_id=6 + i) for i in range(num_draft)]
    manager = KVCacheManagerV2(
        KvCacheConfig(max_gpu_total_bytes=64 << 20, host_cache_size=0, enable_block_reuse=False),
        CacheType.SELF,
        num_layers=6,
        num_kv_heads=nkv,
        head_dim=head_dim,
        tokens_per_block=PAGE,
        max_seq_len=256,
        max_batch_size=len(REQUEST_IDS),
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.FP8,
        layer_specs=specs,
    )
    for request_id, capacity, history in zip(REQUEST_IDS, CAPACITIES, HISTORIES):
        cache = manager._create_kv_cache(request_id, None, None, is_dummy=request_id == 33)
        assert cache.resume(manager._stream.cuda_stream)
        assert cache.resize(capacity, history)
    return manager


def test_the_decode_over_a_unified_kv_cache_is_the_builtin_decode():
    layers, nkv, head_dim = 2, 2, 64
    manager = _unified_manager(layers, nkv, head_dim)
    try:
        num_seqs = len(REQUEST_IDS)
        shape = (manager.num_attention_op_pools, num_seqs, 2, manager.max_blocks_per_seq)
        offsets = torch.full(shape, -7, dtype=torch.int32, device="cuda")
        manager.copy_batch_block_offsets(offsets, list(REQUEST_IDS), 1, 0, num_seqs)
        want = object.__new__(DFlashWorker)
        want._ctx_unified_cache = True
        pool = want._managed_ctx_pool(manager, layers, nkv, head_dim, torch.bfloat16, 2)
        assert want._init_ctx_block_tables(manager, pool, num_seqs, layers, nkv, head_dim, PAGE)
        got = _twin(want)
        metadata = SimpleNamespace(kv_cache_block_offsets=offsets)
        assert DFlashWorker._refresh_ctx_block_tables(want, metadata, num_seqs)
        assert block_table.refresh(got, metadata, num_seqs) is True
        torch.cuda.synchronize()
        _assert_same(got, want)
        assert int(want._ctx_block_counts[0]) > 0
    finally:
        manager.shutdown()


@pytest.mark.parametrize("paged_tables", [False, True], ids=["tables-off", "tables-on"])
def test_the_side_chain_decode_takes_the_lever_where_the_paged_tables_leave_it(
    paged_tables, monkeypatch
):
    """Up to 64 requests ``paged_ctx_block_tables`` keeps the side chain's decode; above it, or
    with that path off, the lever's one launch takes it; the words are the built-in's."""
    gen = torch.Generator().manual_seed(3)
    src = _page_map(gen, 3, SLOTS, WIDTH, 0, window=130)
    for num_seqs in (64, 96):
        want = _worker(src, pool_idx=2, offset=0, divisor=2)
        got = _twin(want)
        fork = object.__new__(context_kv_fork.ContextKvFork)
        fork.worker, fork._launched = got, None
        fork.paged_tables_on, fork.block_table_on = paged_tables, True
        declines = []
        fork.declined = lambda *args: declines.append(args)
        launches = []
        decode = block_table.decode
        monkeypatch.setattr(
            block_table, "decode", lambda *args: (launches.append(args[2]), decode(*args))[1]
        )
        metadata = _metadata(src)
        fork._refresh_block_tables(metadata, num_seqs)
        assert DFlashWorker._refresh_ctx_block_tables(want, metadata, num_seqs)
        torch.cuda.synchronize()
        _assert_same(got, want)
        taken_by_tables = paged_tables and num_seqs <= context_kv_fork.TABLES_MAX_REQUESTS
        assert launches == ([] if taken_by_tables else [num_seqs]), num_seqs
        assert not declines
