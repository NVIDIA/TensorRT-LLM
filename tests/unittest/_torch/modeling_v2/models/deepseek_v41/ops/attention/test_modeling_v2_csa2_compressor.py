# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The `csa2_compressor` entries: the engine's decode compressor of an index owner in the compressed-row layout.

* `csa2_owner_slots` (the engine's `refresh_owner_slots`) against the engine's PyTorch form, bit for bit;
* `csa2_compressor_project` against a group-by-group fp32 reference (latents / index keys in bf16 ulps, state pools
  exactly), on the one-launch path (<= 64 rows) and the reduction + RMSNorm + GEMM path;
* its path choice against the engine's own `CSA2Compressor._fuses`;
* the latents it publishes are those the token-indexed `csa2_compress_decode` publishes at the same pool rows.
"""

import itertools
from types import SimpleNamespace as NS

import pytest
import torch
from _v41_util import DEV, IS_SM103, gen_for

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention import (
    csa2_compressor as cmp,  # noqa: E402
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_compress_decode import (  # noqa: E402,E501
    csa2_compress_decode,
)
from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_compressor import (  # noqa: E402,E501
    csa2_compressor_project,
    csa2_owner_slots,
    owner_slot_descriptors,
)

pytestmark = pytest.mark.skipif(
    not IS_SM103, reason="the engine's compressor kernels are certified on sm_103 only"
)
D, NI = 512, 128
I32 = dict(dtype=torch.int32, device=DEV)


def _lengths(requests, rows):
    """Rows per request: `rows` for every request, or a tuple cycled
    over the requests (an eager step of mixed lengths)."""
    per = (
        [rows] * requests
        if isinstance(rows, int)
        else [rows[i % len(rows)] for i in range(requests)]
    )
    return torch.tensor(per, **I32)


def _step(requests, rows, gen, hist=3000):
    """A generation step: request r has `rows` rows (`_lengths`) from a start drawn around `hist` (odd and even)."""
    starts = torch.randint(
        hist // 2, hist, (requests,), generator=gen, device=DEV, dtype=torch.int32
    )
    lengths = _lengths(requests, rows)
    ends = starts + lengths
    cu_seq = torch.cat([torch.zeros(1, **I32), lengths.cumsum(0).int()])
    return starts, ends, cu_seq


def _owners(requests, rows, gen, cols=64):
    n = int(_lengths(requests, rows).sum())
    owners = []
    for ratio in (2, 2, 2, 1):
        rpp = 128 // ratio
        pages = (
            torch.randperm(requests * cols, generator=gen, device=DEV)[: requests * cols]
            .view(requests, cols)
            .int()
        )
        pages[torch.rand(pages.shape, generator=gen, device=DEV) < 0.03] = -1
        batch = (
            (
                torch.full((requests,), -7, **I32),
                torch.full((requests,), -7, **I32),
                torch.full((requests + 1,), -7, **I32),
            )
            if ratio == 2
            else (None,) * 3
        )
        owners.append(
            (
                pages.contiguous(),
                torch.full((n,), -9, dtype=torch.int64, device=DEV),
                torch.full((n,), -9, **I32),
                ratio,
                rpp,
                *batch,
            )
        )
    return owners


def _clone(owners):
    return [tuple(t.clone() if isinstance(t, torch.Tensor) else t for t in o) for o in owners]


@pytest.mark.parametrize(
    "cell", csa2_owner_slots.CELLS, ids=[c.why[:40] for c in csa2_owner_slots.CELLS]
)
def test_owner_slots_cells(cell, certifying):  # noqa: F811
    certifying(csa2_owner_slots)
    s, gen = cell.spec, gen_for(cell)
    r, rows = s["requests"], s["rows"]
    starts, ends, _ = _step(r, rows, gen)
    lengths, base = _lengths(r, rows), torch.zeros(r, **I32)
    owners = _owners(r, rows, gen)
    ref = _clone(owners)
    csa2_owner_slots(ends, lengths, base, owners, owner_slot_descriptors(lengths, base, owners))
    csa2_owner_slots.reference(ends, lengths, base, ref, None)
    for got, want in zip(owners, ref):
        for a, b in zip(got, want):
            if isinstance(a, torch.Tensor):
                assert torch.equal(a, b)


def _project_inputs(s, gen):
    r, rows, ratio = s["requests"], s["rows"], s["ratio"]
    n = int(_lengths(r, rows).sum())
    starts, ends, cu_seq = _step(r, rows, gen)
    counts = ends // ratio - starts // ratio
    cu_comp = torch.cat([torch.zeros(1, **I32), counts.cumsum(0).int()])
    src = (torch.randn(n, ratio * D, generator=gen, device=DEV)).bfloat16()
    cols = 64
    pool_kv = torch.randn(r * cols + 2, 128, D, generator=gen, device=DEV)
    pool_sc = torch.randn(r * cols + 2, 128, D, generator=gen, device=DEV)
    tab = (torch.randperm(r * cols, generator=gen, device=DEV).view(r, cols) + 1).int().contiguous()
    tab2 = (
        (torch.randperm(r * cols, generator=gen, device=DEV).view(r, cols) + 1).int().contiguous()
    )
    norm_w = (1 + 0.1 * torch.randn(D, generator=gen, device=DEV)).bfloat16()
    wk = (torch.randn(NI, D, generator=gen, device=DEV) * 0.05).bfloat16()
    state = (pool_kv, pool_sc, tab, tab2) if ratio == 2 else (None,) * 4
    most = rows if isinstance(rows, int) else max(rows)
    return (
        src,
        state,
        ends,
        starts,
        cu_seq,
        cu_comp,
        -(-most // ratio),
        norm_w,
        wk,
        ratio,
        int(cu_comp[-1]),
    )


def _bf16_close(got, ref, ulps=4):
    scale = ref.float().abs().amax(-1, keepdim=True).clamp_min(1e-6)
    assert ((got.float() - ref.float()).abs() <= ulps * 2.0**-8 * scale).all()


@pytest.mark.parametrize(
    "cell", csa2_compressor_project.CELLS, ids=[c.why[:40] for c in csa2_compressor_project.CELLS]
)
def test_compressor_project_cells(cell, certifying):  # noqa: F811
    certifying(csa2_compressor_project)
    s, gen = cell.spec, gen_for(cell)
    src, state, ends, starts, cu_seq, cu_comp, mo, norm_w, wk, ratio, groups = _project_inputs(
        s, gen
    )
    rstate = tuple(t.clone() if t is not None else None for t in state)
    lat, k = csa2_compressor_project(
        src, *state, ends, starts, cu_seq, cu_comp, mo, norm_w, 1e-6, wk, ratio
    )
    rlat, rk = csa2_compressor_project.reference(
        src, *rstate, ends, starts, cu_seq, cu_comp, mo, norm_w, 1e-6, wk, ratio
    )
    _bf16_close(lat[:groups], rlat[:groups])
    _bf16_close(k[:groups], rk[:groups], ulps=8)
    assert not lat[groups:].any() and not k[groups:].any(), "rows past the step's groups are zero"
    if ratio == 2:
        assert torch.equal(state[0], rstate[0]) and torch.equal(state[1], rstate[1]), (
            "every row of the step in the state"
        )


@pytest.mark.parametrize("rows, ratio", itertools.product((4, 24, 64, 65, 96, 160, 1280), (1, 2)))
def test_path_is_the_engines(rows, ratio, monkeypatch):
    """The one-launch kernel runs exactly where the engine's `CSA2Compressor._fuses` takes it."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.compressor import CSA2Compressor

    gen = torch.Generator(device=DEV).manual_seed(rows)
    s = dict(
        requests=rows // 4 if rows % 4 == 0 else rows, rows=4 if rows % 4 == 0 else 1, ratio=ratio
    )
    src, state, ends, starts, cu_seq, cu_comp, mo, norm_w, wk, ratio, _ = _project_inputs(s, gen)
    calls = []
    fused = cmp._compress_project
    monkeypatch.setattr(cmp, "_compress_project", lambda *a: (calls.append(1), fused(*a))[1])
    csa2_compressor_project(src, *state, ends, starts, cu_seq, cu_comp, mo, norm_w, 1e-6, wk, ratio)
    me = NS(ratio=ratio, head_dim=D, norm=NS(weight=norm_w))
    batch = (
        None
        if ratio == 1
        else NS(
            output_rows=src.shape[0],
            kv_state=state[0],
            score_state=state[1],
            kv_page_table=state[2],
            score_page_table=state[3],
            kv_lengths=ends,
        )
    )
    assert bool(calls) == CSA2Compressor._fuses(me, src, batch, wk)


@pytest.mark.parametrize("requests, rows, ratio", [(40, 4, 2), (16, 4, 2), (25, 6, 2), (40, 4, 1)])
def test_publishes_the_token_indexed_rows(requests, rows, ratio):
    """Group `g` lands at `slots[g]` with the latent / index key that `csa2_compress_decode` (one row per token)
    publishes at that pool row: the same pairs, the same norm and projection."""
    gen = torch.Generator(device=DEV).manual_seed(requests + rows)
    s = dict(requests=requests, rows=rows, ratio=ratio)
    src, state, ends, starts, cu_seq, cu_comp, mo, norm_w, wk, ratio, groups = _project_inputs(
        s, gen
    )
    cols = 64
    g_table = (
        (torch.randperm(requests * cols, generator=gen, device=DEV).view(requests, cols))
        .int()
        .contiguous()
    )
    lengths, base = torch.full((requests,), rows, **I32), torch.zeros(requests, **I32)
    n = requests * rows
    batch = (
        tuple(torch.empty(k, **I32) for k in (requests, requests, requests + 1))
        if ratio == 2
        else (None,) * 3
    )
    owner = (
        g_table,
        torch.empty(n, dtype=torch.int64, device=DEV),
        torch.empty(n, **I32),
        ratio,
        128 // ratio,
        *batch,
    )
    csa2_owner_slots(ends, lengths, base, [owner], owner_slot_descriptors(lengths, base, [owner]))
    _, slots, first, *_ = owner
    if ratio == 2:
        assert (
            torch.equal(owner[5], starts)
            and torch.equal(owner[6], ends)
            and torch.equal(owner[7], cu_comp)
        )
    old_state = tuple(t.clone() if t is not None else None for t in state)
    lat, k = csa2_compressor_project(
        src, *state, ends, starts, cu_seq, cu_comp, mo, norm_w, 1e-6, wk, ratio
    )
    pos = torch.cat([torch.arange(int(a), int(b), **I32) for a, b in zip(starts, ends)])
    rid = torch.arange(requests, device=DEV).repeat_interleave(rows)
    cs, ce = starts.repeat_interleave(rows), ends.repeat_interleave(rows)
    olat, ok, oslots, ofirst = csa2_compress_decode(
        src, pos, rid, cs, ce, *old_state, norm_w, 1e-6, wk, g_table, ratio
    )
    # the token rows that complete a group publish it (at ratio 2 the odd positions)
    done = (pos % 2 == 1) if ratio == 2 else torch.ones_like(pos, dtype=torch.bool)
    assert int(done.sum()) == groups
    assert torch.equal(slots[:groups], oslots[done]) and torch.equal(first[:groups], ofirst[done])
    _bf16_close(lat[:groups], olat[done], ulps=2)
    _bf16_close(k[:groups], ok[done], ulps=8)


@pytest.mark.parametrize("requests", [1, 2, 3])
def test_one_row_steps_with_a_max_length_request(requests):
    """Spec-off generation steps (one row per request, 1-3 requests per rank) where request 0 sits at the end of a
    max-length cache (the engine's warm-up dummy: position max_seq_len - 2) whose state and GLOBAL tables are mapped
    only at the step's pages: the groups, slots and state writes are the reference's, and nothing is read or written
    through an unmapped (-1) page."""
    gen = torch.Generator(device=DEV).manual_seed(requests)
    max_seq, cols = 996595, 7788
    starts = torch.tensor([max_seq - 2] + [2 * r + 1 for r in range(1, requests)], **I32)
    ends = starts + 1
    cu_seq = torch.arange(0, requests + 1, **I32)
    lengths, base = torch.ones(requests, **I32), torch.zeros(requests, **I32)
    pages = 4 * requests + 2
    pool_kv = torch.randn(pages, 128, D, generator=gen, device=DEV)
    pool_sc = torch.randn(pages, 128, D, generator=gen, device=DEV)
    tab_kv = torch.full((requests, cols), -1, **I32)
    tab_sc = torch.full((requests, cols), -1, **I32)
    g_tab = torch.full((requests, cols), -1, **I32)
    for r in range(requests):
        p = int(starts[r])
        tab_kv[r, p // 128], tab_sc[r, p // 128], g_tab[r, p // 2 // 64] = 2 * r, 2 * r + 1, r
    owner = (
        g_tab,
        torch.empty(requests, dtype=torch.int64, device=DEV),
        torch.empty(requests, **I32),
        2,
        64,
        torch.empty(requests, **I32),
        torch.empty(requests, **I32),
        torch.empty(requests + 1, **I32),
    )
    csa2_owner_slots(ends, lengths, base, [owner], owner_slot_descriptors(lengths, base, [owner]))
    _, slots, first, _, _, b_starts, b_ends, cu_comp = owner
    assert torch.equal(b_starts, starts) and torch.equal(b_ends, ends)
    src = torch.randn(requests, 2 * D, generator=gen, device=DEV).bfloat16()
    norm_w = (1 + 0.1 * torch.randn(D, generator=gen, device=DEV)).bfloat16()
    wk = (torch.randn(NI, D, generator=gen, device=DEV) * 0.05).bfloat16()
    ref_state = (pool_kv.clone(), pool_sc.clone())
    with validating(csa2_compressor_project):
        lat, k = csa2_compressor_project(
            src,
            pool_kv,
            pool_sc,
            tab_kv,
            tab_sc,
            ends,
            starts,
            cu_seq,
            cu_comp,
            1,
            norm_w,
            1e-6,
            wk,
            2,
        )
    torch.cuda.synchronize()
    rlat, rk = csa2_compressor_project.reference(
        src, *ref_state, tab_kv, tab_sc, ends, starts, cu_seq, cu_comp, 1, norm_w, 1e-6, wk, 2
    )
    groups = int(cu_comp[-1])
    assert (
        groups == requests
    )  # every start is odd: each row completes its pair with the cached partner
    _bf16_close(lat[:groups], rlat[:groups])
    _bf16_close(k[:groups], rk[:groups], ulps=8)
    assert torch.equal(pool_kv, ref_state[0]) and torch.equal(pool_sc, ref_state[1])
    assert bool((slots[:groups] >= 0).all()) and torch.equal(first[:groups], starts - 1)


@pytest.mark.parametrize(
    "rows, starts, ratio",
    [
        ((5,), (0,), 2),  # a five-token prompt: the one-launch kernel
        ((3, 7, 1, 20), (0, 1001, 64, 37), 2),  # a mixed chunk of 31 rows, odd and even starts
        ((300,), (0,), 2),  # past 64 rows: the reduction, RMSNorm and GEMM
        ((1, 130, 37), (2047, 129, 0), 2),
        ((5,), (0,), 1),
        ((1, 130, 37), (2047, 129, 0), 1),
        (
            (8996,),
            (0,),
            2,
        ),  # a 9k-token prompt: the index-K GEMM over every row (cuBLAS tiles of large M)
        ((1500, 2600), (0, 777), 2),
        ((8996,), (0,), 1),
    ],
)
def test_context_chunks_are_the_engines_compressor(rows, starts, ratio):
    """A context chunk (any rows per request, the step's capacity as output rows) through the targets' path
    (`csa2_bf16_lowm_proj` of the hidden rows, `csa2_owner_slots`, `csa2_compressor_project`) equals the engine's
    `CSA2Compressor.project` on the same compression batch bit for bit: latents, index keys and the state pools."""
    from tensorrt_llm._torch.attention.backends.sparse.csa2.compressor import (
        CSA2CompressionBatch,
        CSA2Compressor,
    )
    from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.attention.csa2_bf16_lowm_proj import (
        csa2_bf16_lowm_proj,
    )

    gen = torch.Generator(device=DEV).manual_seed(sum(rows) + ratio)
    hidden, requests, n = 5120, len(rows), sum(rows)
    stock = CSA2Compressor(hidden, D, ratio, 1e-6).cuda()
    index_wk = torch.nn.Linear(D, NI, bias=False, dtype=torch.bfloat16, device=DEV)
    with torch.no_grad():
        w = stock.wkv_gate.weight if ratio == 2 else stock.wkv.weight
        w.copy_((torch.randn(w.shape, generator=gen, device=DEV) * 0.02).bfloat16())
        stock.norm.weight.copy_((1 + 0.1 * torch.randn(D, generator=gen, device=DEV)).bfloat16())
        index_wk.weight.copy_((torch.randn(NI, D, generator=gen, device=DEV) * 0.05).bfloat16())
    x = torch.randn(n, hidden, generator=gen, device=DEV).bfloat16()
    lengths = torch.tensor(rows, **I32)
    ends = torch.tensor(starts, **I32) + lengths
    base = torch.zeros(requests, **I32)
    cu_seq = torch.tensor([0, *itertools.accumulate(rows)], **I32)
    cols = 128  # 16384 positions per request
    g_table = (
        torch.randperm(requests * cols, generator=gen, device=DEV)
        .view(requests, cols)
        .int()
        .contiguous()
    )
    batch = (
        tuple(torch.empty(k, **I32) for k in (requests, requests, requests + 1))
        if ratio == 2
        else (None,) * 3
    )
    owner = (
        g_table,
        torch.empty(n, dtype=torch.int64, device=DEV),
        torch.empty(n, **I32),
        ratio,
        128 // ratio,
        *batch,
    )
    csa2_owner_slots(ends, lengths, base, [owner], owner_slot_descriptors(lengths, base, [owner]))
    max_outputs = max(1, -(-max(rows) // 2))
    if ratio == 2:
        pool_kv = torch.randn(requests * cols + 2, 128, D, generator=gen, device=DEV)
        pool_sc = torch.randn(requests * cols + 2, 128, D, generator=gen, device=DEV)
        tab = (
            (torch.randperm(requests * cols, generator=gen, device=DEV).view(requests, cols) + 1)
            .int()
            .contiguous()
        )
        tab2 = (
            (torch.randperm(requests * cols, generator=gen, device=DEV).view(requests, cols) + 1)
            .int()
            .contiguous()
        )
        b_starts, b_ends, cu_comp = owner[5:]
        state = (pool_kv, pool_sc, tab, tab2)
        src = csa2_bf16_lowm_proj(x, stock.wkv_gate.weight)
        args = (b_ends, b_starts, cu_seq, cu_comp)
        engine_batch = CSA2CompressionBatch(
            kv_state=pool_kv.clone(), score_state=pool_sc.clone(), kv_page_table=tab, score_page_table=tab2,
            kv_lengths=b_ends, start_positions=b_starts, cu_seq_lengths=cu_seq, cu_compressed_lengths=cu_comp,
            output_rows=n, page_size=128, max_compressed_per_request=max_outputs,
        )  # fmt: skip
    else:
        state, engine_batch = (None,) * 4, None
        src = csa2_bf16_lowm_proj(x, stock.wkv.weight)
        args = (ends, ends, cu_seq, cu_seq)
    lat, k = csa2_compressor_project(
        src, *state, *args, max_outputs, stock.norm.weight, stock.eps, index_wk.weight, ratio
    )
    with torch.no_grad():
        want_lat, want_k = stock.project(x, engine_batch, index_wk)
    groups = n if ratio == 1 else int(owner[7][-1])
    assert torch.equal(lat[:groups], want_lat[:groups]) and torch.equal(k[:groups], want_k[:groups])
    if ratio == 2:
        assert torch.equal(state[0], engine_batch.kv_state) and torch.equal(
            state[1], engine_batch.score_state
        )
