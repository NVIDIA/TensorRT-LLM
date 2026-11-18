# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ``ctx_kv_store_paged`` (fast path ``paged_ctx_kv_store``).

The store of a step's drafter context K/V into a paged drafter context must leave the
pages bit for bit as ``DFlashWorker._store_context_kv_paged`` (flashinfer's
``append_paged_kv_cache``, one layer at a time) leaves them -- over the KV cache
manager's per-layer views and over the worker's private pool -- and every case the op
does not take must stay the worker's. FP8 E4M3 pages (the FP8 draft KV pool) take BF16
K/V through the saturating cast of ``to_kv_storage``, bit for bit the worker's appends.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.kernels.context_kv_fork import (  # noqa: E501
    fork as fork_mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_107__tep.ops.spec import (  # noqa: E501
    ctx_kv_store_paged as mod,
)
from tensorrt_llm._torch.speculative.dflash import DFlashWorker

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

PAGE = 32
LAYERS = 5
HEAD_DIM = 128
FP8 = torch.float8_e4m3fn


def _bits(pool):
    """The pool's bytes, so FP8 pools (and NaN) compare bit for bit."""
    return pool.view(torch.uint8) if pool.dtype == FP8 else pool


def _state(manager, heads, slots, blocks, seed, page_dtype=torch.bfloat16):
    """The worker attributes ``_store_context_kv_paged`` reads, over a fresh pool."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = slots * blocks + 3
    if manager:
        # The manager's interleaved pool: one view per layer, the same strides for each.
        pool = torch.randn((pages, LAYERS, 2, heads, PAGE, HEAD_DIM), generator=g, device="cuda")
        pool = pool.to(page_dtype)
        caches = [pool[:, layer] for layer in range(LAYERS)]
    else:
        pool = torch.randn((LAYERS, pages, 2, heads, PAGE, HEAD_DIM), generator=g, device="cuda")
        pool = pool.to(page_dtype)
        caches = pool
    # Every (row, block) its own page, so no two entries share a destination.
    table = torch.randperm(pages, generator=g, device="cuda")[: slots * blocks]
    table = table.to(torch.int32).view(slots, blocks).contiguous()
    indptr = torch.arange(0, (slots + 1) * blocks, blocks, dtype=torch.int32, device="cuda")
    state = dict(
        _ctx_kv_buf=caches,
        _ctx_page_size=PAGE,
        _ctx_block_tables=table if manager else None,
        _ctx_block_indptr=indptr if manager else None,
        _ctx_page_table=None if manager else table,
        _ctx_kv_indptr=None if manager else indptr,
        _ctx_kv_last_page_len=torch.full((slots,), PAGE, dtype=torch.int32, device="cuda"),
        _ctx_paged_append=None,
    )
    return state, pool


def _worker(cls, state, fork=None):
    worker = object.__new__(cls)
    worker.__dict__.update(state)
    if fork is not None:
        worker.__dict__["_kv_fork"] = fork
    return worker


def _fork(on):
    fork = object.__new__(fork_mod.ContextKvFork)
    fork.paged_store_on = on
    return fork


def _entries(n, slots, blocks, seed):
    """n distinct (row, position) entries and their K/V ([n, layers, heads, head_dim])."""
    g = torch.Generator(device="cuda").manual_seed(seed + 1)
    flat = torch.randperm(slots * blocks * PAGE, generator=g, device="cuda")[:n]
    rows = (flat // (blocks * PAGE)).contiguous()
    positions = (flat % (blocks * PAGE)).contiguous()
    return rows, positions, g


def _kv(n, heads, g, transposed):
    if transposed:
        # [layers, n, ...] storage viewed as [n, layers, ...], as a permuted projection is.
        k = torch.randn((LAYERS, n, heads, HEAD_DIM), generator=g, device="cuda").transpose(0, 1)
        v = torch.randn((LAYERS, n, heads, HEAD_DIM), generator=g, device="cuda").transpose(0, 1)
    else:
        k = torch.randn((n, LAYERS, heads, HEAD_DIM), generator=g, device="cuda")
        v = torch.randn((n, LAYERS, heads, HEAD_DIM), generator=g, device="cuda")
    return k.to(torch.bfloat16), v.to(torch.bfloat16)


@pytest.mark.parametrize("manager", [True, False])
@pytest.mark.parametrize("heads", [1, 2])
@pytest.mark.parametrize("n", [1, 8, 64, 520])
@pytest.mark.parametrize("transposed", [False, True])
def test_bitwise_the_worker(manager, heads, n, transposed):
    slots, blocks = 70, 12
    state, pool = _state(manager, heads, slots, blocks, seed=n * 7 + heads)
    want_state, want_pool = _state(manager, heads, slots, blocks, seed=n * 7 + heads)
    rows, positions, g = _entries(n, slots, blocks, seed=n)
    k, v = _kv(n, heads, g, transposed)
    DFlashWorker._store_context_kv_paged(_worker(DFlashWorker, want_state), k, v, rows, positions)
    worker = _worker(fork_mod.ContextKvForkDFlashWorker, state, _fork(True))
    assert fork_mod._paged_store_fits(worker, k, v, rows, positions)
    worker._store_context_kv_paged(k, v, rows, positions)
    assert torch.equal(pool, want_pool)


def test_op_matches_its_reference():
    slots, blocks, n, heads = 9, 5, 40, 2
    state, pool = _state(True, heads, slots, blocks, seed=3)
    ref_state, ref_pool = _state(True, heads, slots, blocks, seed=3)
    rows, positions, g = _entries(n, slots, blocks, seed=4)
    k, v = _kv(n, heads, g, False)
    args = (k, v, rows, positions, state["_ctx_block_tables"], PAGE, state["_ctx_kv_buf"])
    mod.ctx_kv_store_paged.is_valid(*args)
    mod.ctx_kv_store_paged(*args)
    mod.ctx_kv_store_paged.reference(
        k, v, rows, positions, ref_state["_ctx_block_tables"], PAGE, ref_state["_ctx_kv_buf"]
    )
    assert torch.equal(pool, ref_pool)


@pytest.mark.parametrize("case", ["off", "int32_rows", "single_latent"])
def test_the_rest_stays_the_workers(case, monkeypatch):
    slots, blocks, n, heads = 16, 6, 24, 1
    state, pool = _state(True, heads, slots, blocks, seed=21)
    want_state, want_pool = _state(True, heads, slots, blocks, seed=21)
    rows, positions, g = _entries(n, slots, blocks, seed=22)
    k, v = _kv(n, heads, g, False)
    if case == "int32_rows":
        rows = rows.to(torch.int32)
    elif case == "single_latent":
        # An MLA drafter's cache: no V, one vector per token per layer (the worker's scatter).
        v = None
    calls = []
    real = fork_mod.ctx_kv_store_paged
    monkeypatch.setattr(
        fork_mod, "ctx_kv_store_paged", lambda *a, **kw: (calls.append(a), real(*a, **kw))
    )
    DFlashWorker._store_context_kv_paged(_worker(DFlashWorker, want_state), k, v, rows, positions)
    worker = _worker(fork_mod.ContextKvForkDFlashWorker, state, _fork(case != "off"))
    worker._store_context_kv_paged(k, v, rows, positions)
    assert not calls
    assert torch.equal(pool, want_pool)


def _kv_wide(n, heads, g):
    """BF16 K/V spanning past E4M3's range, with infinities and NaN, as [n, layers, heads, hd]."""
    k, v = _kv(n, heads, g, False)
    k, v = k * 300, v * 300
    k.view(-1)[:4] = torch.tensor([1e4, -1e4, float("inf"), float("nan")], dtype=k.dtype)
    v.view(-1)[:4] = torch.tensor([-float("inf"), 449.0, -449.0, 1e-9], dtype=v.dtype)
    return k, v


@pytest.mark.parametrize("manager", [True, False])
@pytest.mark.parametrize("n", [1, 64, 520])
def test_fp8_pages_bitwise_the_reference(manager, n):
    slots, blocks, heads = 70, 12, 2
    state, pool = _state(manager, heads, slots, blocks, seed=n + 5, page_dtype=FP8)
    ref_state, ref_pool = _state(manager, heads, slots, blocks, seed=n + 5, page_dtype=FP8)
    rows, positions, g = _entries(n, slots, blocks, seed=n + 6)
    k, v = _kv_wide(n, heads, g)
    table = state["_ctx_block_tables"] if manager else state["_ctx_page_table"]
    ref_table = ref_state["_ctx_block_tables"] if manager else ref_state["_ctx_page_table"]
    args = (k, v, rows, positions, table, PAGE, state["_ctx_kv_buf"])
    mod.ctx_kv_store_paged.is_valid(*args)
    mod.ctx_kv_store_paged(*args)
    mod.ctx_kv_store_paged.reference(
        k, v, rows, positions, ref_table, PAGE, ref_state["_ctx_kv_buf"]
    )
    assert torch.equal(_bits(pool), _bits(ref_pool))


@pytest.mark.parametrize("manager", [True, False])
@pytest.mark.parametrize("n", [1, 8, 520])
def test_fp8_pages_bitwise_the_worker(manager, n):
    slots, blocks, heads = 70, 12, 2
    state, pool = _state(manager, heads, slots, blocks, seed=n + 9, page_dtype=FP8)
    want_state, want_pool = _state(manager, heads, slots, blocks, seed=n + 9, page_dtype=FP8)
    rows, positions, g = _entries(n, slots, blocks, seed=n + 10)
    k, v = _kv(n, heads, g, False)
    DFlashWorker._store_context_kv_paged(_worker(DFlashWorker, want_state), k, v, rows, positions)
    worker = _worker(fork_mod.ContextKvForkDFlashWorker, state, _fork(True))
    assert fork_mod._paged_store_fits(worker, k, v, rows, positions)
    worker._store_context_kv_paged(k, v, rows, positions)
    assert torch.equal(_bits(pool), _bits(want_pool))


@pytest.mark.parametrize("case", ["fp16_kv", "mixed_pages"])
def test_fp8_pages_the_op_does_not_take(case):
    slots, blocks, n, heads = 16, 6, 24, 1
    state, _ = _state(True, heads, slots, blocks, seed=31, page_dtype=FP8)
    rows, positions, g = _entries(n, slots, blocks, seed=32)
    k, v = _kv(n, heads, g, False)
    if case == "fp16_kv":
        k, v = k.to(torch.float16), v.to(torch.float16)
    else:
        bf16_state, _ = _state(True, heads, slots, blocks, seed=31)
        state["_ctx_kv_buf"] = state["_ctx_kv_buf"][:-1] + bf16_state["_ctx_kv_buf"][-1:]
    worker = _worker(fork_mod.ContextKvForkDFlashWorker, state, _fork(True))
    assert not fork_mod._paged_store_fits(worker, k, v, rows, positions)
