# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 key-sharded DSA decode Top-K (k31_key_shard_topk), ranks emulated on one GPU.

Every rank of a ``world``-rank group gets its own plain workspace buffer; the peer addresses stand in for the peer
mappings. Each rank scores its key blocks and takes its local Top-L; with the all-keys exchange every rank pushes
it and then every rank merges, with the pruned exchange every rank runs each of its three launches before any rank
runs the next. Every rank's rows, ordered and remapped to pool rows, must equal the replicated path's (the full-row
logits + the deterministic GVR Top-K to pool rows) bit for bit, and on integer data the reference's set exactly.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.attention.dsa_key_shard import (  # noqa: E501
    KEY_SHARD_TOKENS_PER_BLOCK,
    KeyShardTopKMerge,
    key_shard_local_count,
    key_shard_prepare_rows,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.attention.dsa_key_shard_prune import (  # noqa: E501
    KeyShardPrunedExchange,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.attention.dsa_topk_order_remap import (  # noqa: E501
    dsa_topk_order_remap_into,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.cute_dsl_fp8_paged_mqa_logits import (  # noqa: E501
    cute_dsl_fp8_paged_mqa_logits_pdl,
    logits_row_stride,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dsa_key_shard import (
    KEY_SHARD_ENV,
    LAYOUTS,
    MAX_ROWS,
    key_shard_layout,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.dsa_key_shard import (  # noqa: E501
    k31_key_shard_topk as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.indexer_topk_decode_gvr import (  # noqa: E501
    indexer_topk_decode_gvr_paged,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.attention.paged_mqa_schedule import (  # noqa: E501
    paged_mqa_schedule,
)

__extra_import_path__ = ["../../../..", "."]
import _k31_dsa_inputs as inputs  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = inputs.on_sm103

PAGE = KEY_SHARD_TOKENS_PER_BLOCK
H, D = inputs.NUM_INDEX_HEADS, inputs.HEAD_DIM
#: Pool remap of the order step (MLA page table of each request, page index scale, layer).
POOL_SCALE, POOL_LAYER = 24, 7
ROWS_MAX = 128


def _cache(lens, data: str, gen: torch.Generator, world: int, skew: bool):
    """[pages, 64, 1, 132] uint8 index-key cache (each request on random pages) and its [B, pages] table."""
    pages_per_req = [-(-int(c) // PAGE) for c in lens]
    total = sum(pages_per_req)
    num_pages = total + 8
    k, s = inputs.keys(total * PAGE, data, gen)
    if skew:
        # Blocks of rank 0 (block b % world == 0) score far above the rest: most of the Top-K sits on one rank.
        blk = torch.cat([torch.arange(n) for n in pages_per_req])
        s = s.view(total, PAGE)
        s[blk % world == 0] *= 64.0
        s = s.reshape(-1)
    perm = torch.randperm(num_pages, generator=gen)[:total]
    base = torch.randint(0, 256, (num_pages, PAGE * (D + 4)), generator=gen, dtype=torch.uint8)
    base[perm, : PAGE * D] = k.view(torch.uint8).reshape(total, PAGE * D)
    base[perm, PAGE * D :] = s.contiguous().view(torch.uint8).reshape(total, PAGE * 4)
    width = max(pages_per_req)
    table = torch.zeros(len(lens), width, dtype=torch.int32)
    used = 0
    for b, n in enumerate(pages_per_req):
        table[b, :n] = perm[used : used + n].to(torch.int32)
        used += n
    return base.cuda().view(num_pages, PAGE, 1, D + 4), table.cuda()


def _build(spec, seed: int) -> dict:
    gen = torch.Generator().manual_seed(seed)
    batch, next_n, world = spec["batch"], spec["next_n"], spec["world"]
    lo, hi = spec["context_lens"]
    lens = torch.randint(lo, hi + 1, (batch,), generator=gen, dtype=torch.int32)
    data = "int" if spec["data"] == "ties" else "randn"
    kv_cache, table = _cache(lens.tolist(), data, gen, world, spec["data"] == "skewed")
    rows = batch * next_n
    q = inputs.queries(rows, data, gen).view(batch, next_n, H, D)
    if spec["data"] == "zeros":
        # Every score +-0: all keys of a row tie.
        q = torch.zeros_like(q)
    weights = inputs.weights(rows, data, gen)
    pool_table = torch.randint(
        0, 40_000, (batch, table.shape[1]), generator=gen, dtype=torch.int32
    ).cuda()
    return dict(
        spec=spec,
        lens=lens.cuda(),
        kv_cache=kv_cache,
        table=table,
        q=q,
        weights=weights,
        pool_table=pool_table,
    )


def _replicated(a) -> torch.Tensor:
    """The replicated path's pool rows: full-row logits, deterministic GVR Top-K to ordered pool rows."""
    spec = a["spec"]
    batch, next_n, k = spec["batch"], spec["next_n"], spec["topk"]
    max_len = int(a["lens"].max())
    sched = torch.empty(inputs.num_sms() + 1, 2, dtype=torch.int32, device="cuda")
    paged_mqa_schedule(a["lens"].view(-1, 1), 64, inputs.num_sms(), sched)
    logits = torch.empty(batch * next_n, logits_row_stride(max_len), device="cuda")[:, :max_len]
    cute_dsl_fp8_paged_mqa_logits_pdl(
        a["q"], a["kv_cache"], a["weights"], a["lens"], a["table"], sched, max_len, 1, None, logits
    )
    rows = torch.empty(batch * next_n, k, dtype=torch.int32, device="cuda")
    indexer_topk_decode_gvr_paged(
        logits,
        a["lens"],
        rows,
        next_n,
        max_len,
        a["pool_table"],
        PAGE,
        POOL_SCALE * PAGE,
        POOL_LAYER,
    )
    return rows


def _score_and_select(a, v, m, rows, max_local):
    """One rank's local logits into its workspace and its local Top-L."""
    logits = m.local_logits[:rows, :max_local]
    cute_dsl_fp8_paged_mqa_logits_pdl(
        a["q"],
        a["kv_cache"],
        a["weights"],
        v["ctx"],
        v["bt"],
        v["sched"],
        max_local,
        1,
        None,
        logits,
    )
    torch.ops.trtllm.cute_dsl_indexer_topk_decode(
        logits, v["row_lens"], v["local_topk"], m.local_k, 1
    )


def _emulate(a, call_rank=None, state=None):
    """Every rank's merged token rows (rank order). ``call_rank``: a rank whose step runs through the op's
    ``__call__`` (all-keys exchange: driven last, after every other rank has pushed; pruned exchange: on a side
    stream after every other rank's first launch, while the other ranks' later launches run). ``state``: the
    workspaces and views of a previous call to reuse (the call counters continue)."""
    if state is None:
        state = _setup(a)
    views, merges = state
    spec = a["spec"]
    batch, next_n, world, k = spec["batch"], spec["next_n"], spec["world"], spec["topk"]
    rows = batch * next_n
    max_local = views[0]["max_local"]
    outs = [torch.full((rows, k), -7, dtype=torch.int32, device="cuda") for _ in range(world)]
    others = [r for r in range(world) if r != call_rank]

    def call(r):
        v, m = views[r], merges[r]
        op(
            a["q"],
            a["kv_cache"],
            a["weights"],
            v["ctx"],
            v["bt"],
            v["sched"],
            v["row_lens"],
            v["local_topk"],
            max_local,
            m,
            outs[r],
            a["lens"] if spec["pruned"] else None,
        )

    if spec["pruned"]:
        for r in others:
            _score_and_select(a, views[r], merges[r], rows, max_local)
            merges[r].ladder(views[r]["local_topk"], a["lens"], next_n)
        if call_rank is not None:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                call(call_rank)
        for r in others:
            merges[r].push(rows, a["lens"], next_n)
        for r in others:
            merges[r].merge(outs[r], a["lens"], next_n)
    else:
        for r in others:
            _score_and_select(a, views[r], merges[r], rows, max_local)
            merges[r].pack(views[r]["local_topk"], views[r]["row_lens"])
        if call_rank is not None:
            call(call_rank)
        for r in others:
            merges[r].merge(outs[r])
    torch.cuda.synchronize()
    stats = [m.stats() for m in merges]
    return outs, stats, state


def _setup(a):
    """Per emulated rank: a plain workspace buffer, its merge and the step's local views."""
    spec = a["spec"]
    batch, next_n, world, k = spec["batch"], spec["next_n"], spec["world"], spec["topk"]
    rows = batch * next_n
    local_blocks = -(-a["table"].shape[1] // world)
    local_blocks = -(-local_blocks // 4) * 4
    max_local = local_blocks * PAGE
    if spec["pruned"]:
        need = KeyShardPrunedExchange.bytes_needed(ROWS_MAX, world, spec["local_k"], max_local)
        bufs = [torch.zeros(need + 4096, dtype=torch.uint8, device="cuda") for _ in range(world)]
        merges = [
            KeyShardPrunedExchange(bufs, r, ROWS_MAX, k, max_local, spec["local_k"])
            for r in range(world)
        ]
        for m in merges:
            m.rearm()
    else:
        need = KeyShardTopKMerge.bytes_needed(ROWS_MAX, k, world, spec["local_k"], max_local)
        bufs = [torch.zeros(need + 4096, dtype=torch.uint8, device="cuda") for _ in range(world)]
        merges = [
            KeyShardTopKMerge(bufs, r, ROWS_MAX, k, max_local, spec["local_k"])
            for r in range(world)
        ]
    views = []
    for r in range(world):
        bt = torch.full((batch, local_blocks), 2**30, dtype=torch.int32, device="cuda")
        ctx = torch.empty(batch, dtype=torch.int32, device="cuda")
        row_lens = torch.empty(rows, dtype=torch.int32, device="cuda")
        key_shard_prepare_rows(a["lens"], a["table"], bt, ctx, row_lens, r, world, next_n)
        sched = torch.empty(inputs.num_sms() + 1, 2, dtype=torch.int32, device="cuda")
        paged_mqa_schedule(ctx.view(-1, 1), 64, inputs.num_sms(), sched)
        local_topk = torch.empty(rows, spec["local_k"], dtype=torch.int32, device="cuda")
        views.append(
            dict(
                bt=bt,
                ctx=ctx,
                row_lens=row_lens,
                sched=sched,
                local_topk=local_topk,
                max_local=max_local,
            )
        )
    return views, merges


def _pool_rows(a, token_rows: torch.Tensor) -> torch.Tensor:
    out = token_rows.clone()
    dsa_topk_order_remap_into(
        out,
        out,
        None,
        a["pool_table"],
        PAGE,
        POOL_SCALE * PAGE,
        POOL_LAYER,
        rows_per_req=a["spec"]["next_n"],
    )
    return out


def _sorted(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t < 0, 2**31 - 1, t).sort(dim=1).values


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell):
    a = _build(cell.spec, inputs.seed_of(cell))
    expected = _replicated(a)
    outs, stats, (_, merges) = _emulate(a)
    for r, out in enumerate(outs):
        assert torch.equal(_pool_rows(a, out), expected), f"rank {r}"
        assert stats[r][3] == 0, "spin timeouts"
    fallback_rows = stats[0][2]
    if cell.spec["data"] == "skewed":
        assert fallback_rows > 0, "the skewed cell must exercise the exact re-selection"
    if cell.spec["pruned"]:
        # Every exchanged word is empty again after a complete call.
        for r, m in enumerate(merges):
            assert bool((m.lamport == -(2**31)).all()), f"rank {r}: workspace not re-armed"
    # The local Top-L of a rank holding fewer keys than L sends them all: the short-context rows.
    lens = a["lens"].cpu()
    for r in range(cell.spec["world"]):
        counts = key_shard_local_count(lens, r, cell.spec["world"])
        assert int(counts.sum()) > 0


@pytest.mark.parametrize(
    "cell",
    [c for c in op.CELLS if c.spec["data"] == "ties"],
    ids=lambda c: c.why[:48],
)
def test_reference_through_call(cell, certifying):  # noqa: F811
    """On integer data (exact logits whatever the order) the op's rows hold the reference's set; one rank's step
    goes through the op's ``__call__``."""
    certifying(op)
    a = _build(cell.spec, inputs.seed_of(cell))
    outs, _, (views, merges) = _emulate(a, call_rank=1)
    spec = cell.spec
    ref = torch.empty_like(outs[0])
    op.reference(
        a["q"],
        a["kv_cache"],
        a["weights"],
        a["lens"],
        a["table"],
        None,
        views[1]["row_lens"],
        views[1]["local_topk"],
        0,
        merges[1],
        ref,
    )
    for r in range(spec["world"]):
        assert torch.equal(_sorted(outs[r]), _sorted(ref)), f"rank {r}"


def test_repeat_calls_alternate_slots():
    """Three calls in a row on the same workspaces (epochs and receive slots advance) give the same rows."""
    cell = op.CELLS[0]
    spec = dict(cell.spec, batch=4, context_lens=(20_000, 30_000))
    a = _build(spec, 11)
    expected = _replicated(a)
    state = None
    for call in range(3):
        outs, stats, state = _emulate(a, state=state)
        for out in outs:
            assert torch.equal(_pool_rows(a, out), expected)
        assert all(st[0] == call + 1 for st in stats), "one epoch per call"


def test_pruned_back_to_back_calls_rearm():
    """Pruned exchange: calls of 60, 20 and 60 rows (then 60 again) on the same workspaces, back to back. Each gives
    the replicated rows, every exchanged word is empty after each call, and the call counters advance once per
    call; the 20-row call leaves the 60-row call's later rows untouched."""
    base = dict(op.CELLS[0].spec, context_lens=(40_000, 60_000))
    specs = [dict(base, batch=12), dict(base, batch=4), dict(base, batch=12), dict(base, batch=12)]
    builds = [_build(sp, 31 + i) for i, sp in enumerate(specs)]
    merges = _setup(builds[0])[1]
    for call, a in enumerate(builds):
        views = _setup(a)[0]
        expected = _replicated(a)
        outs, stats, _ = _emulate(a, state=(views, merges))
        for r, out in enumerate(outs):
            assert torch.equal(_pool_rows(a, out), expected), f"call {call} rank {r}"
        assert all(st[0] == call + 1 for st in stats), "one call counter step per call"
        for r, m in enumerate(merges):
            assert bool((m.lamport == -(2**31)).all()), (
                f"call {call} rank {r}: workspace not re-armed"
            )


def test_key_shard_layout_is_on_by_default(monkeypatch):
    monkeypatch.delenv(KEY_SHARD_ENV, raising=False)
    for tp in (8, 16, 32):
        assert key_shard_layout(tp) == LAYOUTS[tp]
    layouts = {
        tp: (lay.local_k, lay.min_seqs, lay.max_rows, lay.pruned) for tp, lay in LAYOUTS.items()
    }
    assert layouts == {
        8: (512, 11, MAX_ROWS, False),
        16: (512, 11, MAX_ROWS, True),
        32: (256, 11, MAX_ROWS, True),
    }
    assert MAX_ROWS == 288
    assert key_shard_layout(4) is None
    monkeypatch.setenv(KEY_SHARD_ENV, "0")
    assert key_shard_layout(16) is None
