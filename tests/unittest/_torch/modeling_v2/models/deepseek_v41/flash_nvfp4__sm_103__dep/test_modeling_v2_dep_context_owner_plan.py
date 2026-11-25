# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dep attention's compressed groups of a context chunk (`_context_owner_plan`): request `r` of the step's `segs`
has `t1 - t0` rows ending at position `span`, as the engine's prepare uploads its query lengths and KV ends; every
owner's GLOBAL write slots, first positions and (ratio 2) compression batch then equal the engine's PyTorch form
(`CSA2TrtllmMetadata._refresh_owner_slots`) bit for bit, and its batch has the step's rows as output rows."""

import gc
import importlib
import weakref
from types import SimpleNamespace as NS

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="imports the target (CUDA build)"
)
att = importlib.import_module(
    "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__dep.attention"
)
DEV = "cuda"


@pytest.mark.parametrize(
    "rows, ends",
    [((5,), (5,)), ((3, 7, 1, 20), (3, 1008, 65, 57)), ((1, 130, 37), (2048, 259, 37))],
)
def test_context_owner_plan_is_the_engines(rows, ends):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    gen = torch.Generator(device=DEV).manual_seed(sum(rows))
    sources, ratios = (2, 3, 4), {2: 2, 3: 2, 4: 1}
    requests, n, cols = len(rows), sum(rows), 64
    tables = {
        o: torch.randperm(requests * cols, generator=gen, device=DEV)
        .view(requests, cols)
        .int()
        .contiguous()
        for o in sources
    }
    for t in tables.values():
        t[torch.rand(t.shape, generator=gen, device=DEV) < 0.05] = -1
    segs, t0 = [], 0
    for r, (k, e) in enumerate(zip(rows, ends)):
        segs.append((t0, t0 + k, r, e))
        t0 += k
    ctx = NS(
        pos=torch.zeros(n, dtype=torch.int32, device=DEV),
        segs=segs,
        paged=False,
        nreq=requests,
        cache={("t", o, att.GLOBAL): tables[o] for o in sources},
    )
    plan = att._owner_plan(ctx, sources, ratios)
    assert plan is ctx.cache["owners"] and att._owner_plan(ctx, sources, ratios) is plan
    lengths = torch.tensor(rows, dtype=torch.int32, device=DEV)
    e = torch.tensor(ends, dtype=torch.int32, device=DEV)
    assert torch.equal(plan["ends"], e) and torch.equal(plan["lengths"], lengths)
    assert plan["cu_seq"].tolist() == [0, *torch.tensor(rows).cumsum(0).tolist()]
    assert plan["max_outputs"] == max(1, -(-max(rows) // 2))
    starts = e - lengths
    for o in sources:
        r = ratios[o]
        got = plan["owners"][o]
        slots, first = (
            torch.empty(n, dtype=torch.int64, device=DEV),
            torch.empty(n, dtype=torch.int32, device=DEV),
        )
        batch = (
            tuple(
                torch.empty(k, dtype=torch.int32, device=DEV)
                for k in (requests, requests, requests + 1)
            )
            if r == 2
            else (None,) * 3
        )
        CSA2TrtllmMetadata._refresh_owner_slots(
            starts, lengths, torch.zeros_like(starts), r, 128 // r, tables[o], slots, first, *batch
        )
        assert torch.equal(got[1], slots) and torch.equal(got[2], first), o
        assert len(got[5:]) == len(batch), o
        if r == 2:
            assert all(torch.equal(a, b) for a, b in zip(got[5:], batch)), o


class _Manager:
    """Stands in for the engine's CSA2CacheManager: the generation steps' owner plans live on it."""


def _gen_tables(requests, seed):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    tables = {
        o: torch.randperm(requests * 64, generator=gen, device=DEV)
        .view(requests, 64)
        .int()
        .contiguous()
        for o in _SOURCES
    }
    for t in tables.values():
        t[torch.rand(t.shape, generator=gen, device=DEV) < 0.05] = -1
    return tables


_SOURCES, _RATIOS = (2, 3, 4), {2: 2, 3: 2, 4: 1}


def _gen_step(manager, tables, rows, uniform):
    """A paged generation step: `rows[r]` rows of request `r`, the last one at position `500 * (r + 1)`."""
    nreq, n = len(rows), sum(rows)
    ce = torch.tensor(
        [500 * (r + 1) + 1 for r, k in enumerate(rows) for _ in range(k)],
        dtype=torch.int32,
        device=DEV,
    )
    ctx = NS(
        pos=torch.zeros(n, dtype=torch.int32, device=DEV),
        ce=ce,
        nreq=nreq,
        paged=True,
        rows=None if uniform else tuple(rows),
        kvm=NS(kvm=manager),
        cache={("t", o, att.GLOBAL): tables[o] for o in _SOURCES},
    )
    return ctx, att._owner_plan(ctx, _SOURCES, _RATIOS)


def _assert_engines(plan, tables, rows):
    from tensorrt_llm._torch.attention.backends.sparse.csa2.metadata import CSA2TrtllmMetadata

    nreq, n = len(rows), sum(rows)
    lengths = torch.tensor(rows, dtype=torch.int32, device=DEV)
    ends = torch.tensor([500 * (r + 1) + 1 for r in range(nreq)], dtype=torch.int32, device=DEV)
    assert torch.equal(plan["ends"], ends) and torch.equal(plan["lengths"], lengths)
    for o in _SOURCES:
        r, got = _RATIOS[o], plan["owners"][o]
        slots, first = (
            torch.empty(n, dtype=torch.int64, device=DEV),
            torch.empty(n, dtype=torch.int32, device=DEV),
        )
        batch = (
            tuple(torch.empty(k, dtype=torch.int32, device=DEV) for k in (nreq, nreq, nreq + 1))
            if r == 2
            else (None,) * 3
        )
        CSA2TrtllmMetadata._refresh_owner_slots(
            ends - lengths,
            lengths,
            torch.zeros_like(ends),
            r,
            128 // r,
            tables[o],
            slots,
            first,
            *batch,
        )
        assert torch.equal(got[1], slots) and torch.equal(got[2], first), o
        assert len(got[5:]) == len(batch), o
        if r == 2:
            assert all(torch.equal(a, b) for a, b in zip(got[5:], batch)), o


def test_generation_owner_plans_are_bounded():
    """Generation steps: a uniform step's plan (what a CUDA graph captures) keeps its buffers at fixed addresses
    across steps, while eager steps of mixed lengths (one plan per (rows, requests) shape) churning through many shapes
    keep at most `_MIXED_PLANS` of them; every plan still equals the engine's PyTorch form."""
    manager, tables = _Manager(), _gen_tables(4, 11)
    _, uniform = _gen_step(manager, tables, (4, 4, 4, 4), uniform=True)
    addresses = [t.data_ptr() for o in _SOURCES for t in uniform["owners"][o][1:3]]
    _assert_engines(uniform, tables, (4, 4, 4, 4))
    for k in range(3 * att._MIXED_PLANS):  # mixed lengths: draft lengths of 0 to 5 per request
        rows = (1 + k % 6, 1 + (k // 6) % 6, 4, 1 + k)
        _, plan = _gen_step(manager, tables, rows, uniform=False)
        _assert_engines(plan, tables, rows)
    uniform_plans, mixed_plans = getattr(manager, att._OWNER_PLAN)
    assert len(mixed_plans) == att._MIXED_PLANS and len(uniform_plans) == 1
    _, again = _gen_step(manager, tables, (4, 4, 4, 4), uniform=True)
    assert (
        again is uniform
        and [t.data_ptr() for o in _SOURCES for t in again["owners"][o][1:3]] == addresses
    )
    _assert_engines(again, tables, (4, 4, 4, 4))


def test_generation_owner_plans_are_released_with_the_cache_manager():
    """The plans hold views of the manager's page tables: once the manager is gone (the KV-cache estimation pass's
    manager, replaced by the final one), neither its plans nor the tables they view are kept alive."""
    manager, tables = _Manager(), _gen_tables(3, 12)
    alive = [weakref.ref(t) for t in tables.values()]
    ctx, plan = _gen_step(manager, tables, (2, 2, 2), uniform=True)
    mixed_ctx, mixed = _gen_step(manager, tables, (1, 2, 3), uniform=False)
    torch.cuda.synchronize()
    del ctx, plan, mixed_ctx, mixed, manager, tables
    gc.collect()
    assert all(ref() is None for ref in alive)
