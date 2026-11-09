# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ``ctx_block_tables_paged`` (fast path ``paged_ctx_block_tables``).

The decode of a paged drafter context's page table must leave the worker's
``_ctx_block_tables`` and ``_ctx_block_counts`` bit for bit as
``DFlashWorker._refresh_ctx_block_tables`` does -- placeholders before a sliding
window's live tail and past the allocation, entries below the pool's block offset,
the block divisor, rows past the batch untouched -- and the context-K/V fork's side
chain must leave every case the op does not take to the worker's method.
"""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.kernels.context_kv_fork import (  # noqa: E501
    fork as fork_mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.ops.spec import (  # noqa: E501
    ctx_block_tables_paged as mod,
)
from tensorrt_llm._torch.speculative.dflash import DFlashWorker

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

#: Untouched rows keep these.
TABLE_FILL = 12345
COUNT_FILL = 777


class _Worker:
    """The worker state ``_refresh_ctx_block_tables`` reads and writes."""

    def __init__(self, slots, cols, pool_idx, offset, divisor, unified):
        self._ctx_block_tables = torch.full(
            (slots, cols), TABLE_FILL, dtype=torch.int32, device="cuda"
        )
        self._ctx_block_counts = torch.full((slots,), COUNT_FILL, dtype=torch.long, device="cuda")
        self._ctx_block_cols = torch.arange(1, cols + 1, dtype=torch.long, device="cuda")
        self._ctx_pool_idx = pool_idx
        self._ctx_block_offset = offset
        self._ctx_block_divisor = divisor
        self._ctx_unified_cache = unified

    def clone(self):
        other = object.__new__(_Worker)
        other.__dict__.update(self.__dict__)
        other._ctx_block_tables = self._ctx_block_tables.clone()
        other._ctx_block_counts = self._ctx_block_counts.clone()
        return other


def _offsets(pools, batch, cols, offset, divisor, seed):
    """Block offsets ``[pools, batch, 2, cols]``: pages of each pool's blocks, with placeholders.

    Row patterns: a live tail after evicted (placeholder) columns, as a sliding-window
    layer's per-layer table holds; an allocation that ends early (placeholders after it);
    nothing allocated; every column allocated; and entries below the block offset, which
    decode as placeholders too.
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = torch.randint(0, 1 << 18, (pools, batch, 2, cols), generator=g, device="cuda")
    src = (
        offset
        + pages * divisor
        + torch.randint(0, divisor, pages.shape, generator=g, device="cuda")
    )
    src = src.to(torch.int32)
    for b in range(batch):
        kind = b % 5
        if kind == 0:
            src[:, b, :, : cols // 3] = -1
        elif kind == 1:
            src[:, b, :, (2 * cols) // 3 :] = -1
        elif kind == 2:
            src[:, b] = -1
        elif kind == 4 and offset > 0:
            below = torch.rand(cols, generator=g, device="cuda") < 0.3
            src[:, b, :, below] = offset - 1
    return src


def _metadata(src, unified):
    name = "kv_cache_block_offsets" if unified else "draft_kv_cache_block_offsets"
    return types.SimpleNamespace(**{name: src})


def _fork(worker, on=True):
    fork = object.__new__(fork_mod.ContextKvFork)
    fork.worker = worker
    fork.paged_tables_on = on
    return fork


def _assert_same(got, want):
    assert torch.equal(got._ctx_block_tables, want._ctx_block_tables)
    assert torch.equal(got._ctx_block_counts, want._ctx_block_counts)


@pytest.mark.parametrize("cols", [1, 7, 2048, 2049, 32772])
@pytest.mark.parametrize("num_seqs", [1, 5, 48, 64])
@pytest.mark.parametrize(
    "pool_idx,offset,divisor", [(0, 0, 1), (1, 0, 2), (2, 3, 2), (1, 1000, 10)]
)
@pytest.mark.parametrize("unified", [True, False])
def test_bitwise_the_worker(cols, num_seqs, pool_idx, offset, divisor, unified):
    src = _offsets(3, num_seqs + 3, cols, offset, divisor, seed=cols * 131 + num_seqs)
    base = _Worker(num_seqs + 2, cols, pool_idx, offset, divisor, unified)
    want, got = base.clone(), base.clone()
    assert DFlashWorker._refresh_ctx_block_tables(want, _metadata(src, unified), num_seqs)
    fork_mod.ContextKvFork._refresh_block_tables(_fork(got), _metadata(src, unified), num_seqs)
    _assert_same(got, want)
    # Rows past the batch keep what they held.
    assert bool((got._ctx_block_tables[num_seqs:] == TABLE_FILL).all())
    assert bool((got._ctx_block_counts[num_seqs:] == COUNT_FILL).all())


def test_op_matches_its_reference():
    cols, n = 4099, 9
    src = _offsets(2, n, cols, 5, 4, seed=7)[1, :, 0]
    tables = torch.full((n + 1, cols), TABLE_FILL, dtype=torch.int32, device="cuda")
    counts = torch.full((n + 1,), COUNT_FILL, dtype=torch.long, device="cuda")
    ref_tables, ref_counts = tables.clone(), counts.clone()
    mod.ctx_block_tables_paged.is_valid(src, 5, 4, tables, counts)
    mod.ctx_block_tables_paged(src, 5, 4, tables, counts)
    mod.ctx_block_tables_paged.reference(src, 5, 4, ref_tables, ref_counts)
    assert torch.equal(tables, ref_tables)
    assert torch.equal(counts, ref_counts)


def _patched_op(monkeypatch):
    calls = []
    real = fork_mod.ctx_block_tables_paged

    def counting(*args, **kwargs):
        calls.append(args)
        return real(*args, **kwargs)

    monkeypatch.setattr(fork_mod, "ctx_block_tables_paged", counting)
    return calls


@pytest.mark.parametrize("case", ["off", "too_many", "int64_offsets", "strided_table"])
def test_side_chain_leaves_the_rest_to_the_worker(case, monkeypatch):
    cols, n = 64, 4
    src = _offsets(2, 70, cols, 0, 2, seed=11)
    base = _Worker(70, cols, 1, 0, 2, True)
    if case == "too_many":
        n = mod.MAX_REQUESTS + 1
    elif case == "int64_offsets":
        src = src.to(torch.int64)
    want, got = base.clone(), base.clone()
    if case == "strided_table":
        # A column slice of a wider table: not contiguous (``clone`` would make it so).
        for worker in (want, got):
            wide = torch.full((70, cols + 8), TABLE_FILL, dtype=torch.int32, device="cuda")
            worker._ctx_block_tables = wide[:, :cols]
    assert DFlashWorker._refresh_ctx_block_tables(want, _metadata(src, True), n)
    calls = _patched_op(monkeypatch)
    fork_mod.ContextKvFork._refresh_block_tables(_fork(got, case != "off"), _metadata(src, True), n)
    assert not calls
    _assert_same(got, want)


@pytest.mark.parametrize("case", ["no_tables", "no_requests"])
def test_nothing_to_decode(case, monkeypatch):
    src = _offsets(2, 8, 64, 0, 2, seed=13)
    worker = _Worker(8, 64, 1, 0, 2, True)
    n = 4
    if case == "no_tables":
        worker._ctx_block_tables = None
    else:
        n = 0
    counts = worker._ctx_block_counts.clone()
    calls = _patched_op(monkeypatch)
    fork_mod.ContextKvFork._refresh_block_tables(_fork(worker), _metadata(src, True), n)
    assert not calls
    assert torch.equal(worker._ctx_block_counts, counts)


def test_missing_offsets_raise_as_the_worker_does():
    worker = _Worker(4, 16, 0, 0, 1, True)
    metadata = types.SimpleNamespace()
    with pytest.raises(RuntimeError, match="carries no"):
        fork_mod.ContextKvFork._refresh_block_tables(_fork(worker), metadata, 2)
