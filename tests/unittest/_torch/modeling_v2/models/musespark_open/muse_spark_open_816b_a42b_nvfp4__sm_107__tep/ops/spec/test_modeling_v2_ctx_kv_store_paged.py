# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ``ctx_kv_store_paged`` (fast path ``paged_ctx_kv_store``).

The store of a step's drafter context K/V into a paged drafter context must leave the
pages bit for bit as ``DFlashWorker._store_context_kv_paged`` (flashinfer's
``append_paged_kv_cache``, one layer at a time) leaves them -- over the KV cache
manager's per-layer views and over the worker's private pool -- and every case the op
does not take must stay the worker's.
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


def _state(manager, heads, slots, blocks, seed):
    """The worker attributes ``_store_context_kv_paged`` reads, over a fresh pool."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = slots * blocks + 3
    if manager:
        # The manager's interleaved pool: one view per layer, the same strides for each.
        pool = torch.randn((pages, LAYERS, 2, heads, PAGE, HEAD_DIM), generator=g, device="cuda")
        pool = pool.to(torch.bfloat16)
        caches = [pool[:, layer] for layer in range(LAYERS)]
    else:
        pool = torch.randn((LAYERS, pages, 2, heads, PAGE, HEAD_DIM), generator=g, device="cuda")
        pool = pool.to(torch.bfloat16)
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
