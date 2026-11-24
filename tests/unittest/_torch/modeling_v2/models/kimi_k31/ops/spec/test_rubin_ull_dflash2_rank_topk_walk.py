# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test (one GPU) for the DFlash2 drafter's one-hop greedy tail.

The pair exchange is emulated on one GPU (``PairExchange.emulated``: three sentinel-filled Lamport
buffers, the publish written by unicast): the other ranks' pairs are written into the current
buffer as their multicast would. Every cell of the three entries:

* ``k31_dflash2_rank_top32_publish``: the words in the rank's columns are bitwise ``top32``'s pairs
  (``-0.0`` as ``+0.0``), nothing else is written, the arrival counters are back at zero;
* ``k31_dflash2_selector_walk_mc``: candidates and unary logits bitwise ``merge_top32`` of the
  gathered rows, the scores within TF32 rounding of torch's TF32 einsum (and the walk on the
  kernel's scores is the engine walk; where the scores are bitwise the einsum's, the tokens are
  the engine path's), the rotation state advanced, the previous call's buffer re-armed and rank
  0's drafts after the rows when published;
* ``k31_draft_agreed_rewind``: the engine's rewind followed by the agreement on rank 0's drafts.

Sixteen emulated ranks with one rank's walk perturbed all end with rank 0's drafts and the engine
agreement's next tokens.

A chain of the three with PDL, captured in a CUDA graph, replays the eager tokens over several
calls (the rotation state cycles through the three buffers).
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.spec import (
    rubin_ull_dflash2_rank_topk_walk as kern,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.spec.dflash2_selector_walk import (  # noqa: E501
    selector_greedy_tokens,
    selector_operands,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_global_top32 import (  # noqa: E501
    k31_dflash2_merge_top32,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.dflash2_step_glue import (  # noqa: E501
    k31_draft_agreed_tokens,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_ull_dflash2_rank_topk_walk import (  # noqa: E501
    SENTINEL,
    PairExchange,
    k31_dflash2_rank_top32_publish,
    k31_dflash2_selector_walk_mc,
    k31_draft_agreed_rewind,
    published_pairs,
    rank_top32_scratch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.rubin_ull_next_new_tokens_rewind import (  # noqa: E501
    draft_next_new_tokens_rewind,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="the Kimi K3.1 DFlash2 kernels are certified on SM10x",
)

SHARD, RANK = 10240, 256
VOCAB = 16 * SHARD
NB = 1 << 20  # bytes per emulated Lamport buffer


def _seed(why: str) -> int:
    return zlib.crc32(why.encode()) % 2**31


def _tf32():
    """The served einsum's precision: torch's FP32 matmul in TF32."""
    m = torch.backends.cuda.matmul
    saved = (m.allow_tf32, getattr(m, "fp32_precision", None))
    m.allow_tf32 = True
    if saved[1] is not None:
        m.fp32_precision = "tf32"
    return saved


def _restore(saved):
    m = torch.backends.cuda.matmul
    m.allow_tf32 = saved[0]
    if saved[1] is not None:
        m.fp32_precision = saved[1]


def _logits(rows, width, values, g):
    x = torch.randn(rows, width, generator=g, device="cuda") * 3
    if values == "ties":
        x = (x * 2).round() / 2
        x[:, ::613] = 9.0
    if values == "special":
        x[:, :6] = torch.tensor([-0.0, float("inf"), float("-inf"), float("nan"), 0.0, 1e30])
    return x.bfloat16()


def _row_words(world):
    return world * 64


def _write_peers(buf, cur, world, rank, pairs):
    """Every other rank's published words into buffer ``cur`` (what their multicast writes)."""
    rw = _row_words(world)
    base = cur * NB // 4
    rows = pairs[0].shape[0]
    view = buf[base : base + rows * rw].view(rows, world, 64)
    for r in range(world):
        if r != rank:
            view[:, r] = pairs[r]


def _site(spec, g):
    world, slots, rank = spec["world"], spec["slots"], spec["rank"]
    full = _logits(slots, world * SHARD, spec["values"], g)
    bank = torch.empty(slots, SHARD + RANK, dtype=torch.bfloat16, device="cuda")
    bank[:, :SHARD] = full[:, rank * SHARD : (rank + 1) * SHARD]
    gate = torch.randn(slots, RANK, generator=g, device="cuda")
    if spec["values"] == "ties":
        gate = gate.round()
    bank[:, SHARD:] = gate.bfloat16()
    pred = (0.2 * torch.randn(world * SHARD, RANK, generator=g, device="cuda")).bfloat16()
    succ = (0.2 * torch.randn(world * SHARD, RANK, generator=g, device="cuda")).bfloat16()
    if spec["values"] == "ties":
        pred, succ = (pred * 4).round() / 4, (succ * 4).round() / 4
    pairs = [published_pairs(full[:, r * SHARD : (r + 1) * SHARD], r * SHARD) for r in range(world)]
    anchor = torch.tensor([777], dtype=torch.int64, device="cuda")
    return bank, pred, succ, pairs, anchor


@pytest.mark.parametrize(
    "cell",
    k31_dflash2_rank_top32_publish.CELLS,
    ids=[c.why[:48] for c in k31_dflash2_rank_top32_publish.CELLS],
)
def test_publish_cells(cell):
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(_seed(cell.why))
    world, rows, rank, pitch = spec["world"], spec["rows"], spec["rank"], spec["pitch"]
    buf_rows = _logits(rows, pitch, spec["values"], g)
    logits = buf_rows[:, : spec["shard"]]
    exchange, buf = PairExchange.emulated(world, rank, "cuda", NB)
    cur = int(exchange.flags[0])
    scratch = rank_top32_scratch(8, spec["shard"], "cuda")
    k31_dflash2_rank_top32_publish.is_valid(exchange, logits, rank * SHARD, scratch)
    k31_dflash2_rank_top32_publish(exchange, logits, rank * SHARD, scratch)
    torch.cuda.synchronize()
    ref = k31_dflash2_rank_top32_publish.reference(exchange, logits, rank * SHARD, scratch)
    rw = _row_words(world)
    view = buf[cur * NB // 4 : cur * NB // 4 + rows * rw].view(rows, world, 64)
    assert torch.equal(view[:, rank], ref)
    others = torch.ones(world, dtype=torch.bool)
    others[rank] = False
    assert bool((view[:, others] == SENTINEL).all())
    assert bool((scratch[3] == 0).all())
    # The Lamport state is untouched: the walk (the call's poller) advances it.
    assert exchange.flags.tolist()[:3] == [0, 2, NB]


def _walk_case(spec, g, cur=0, dirty=2, leftover=200000):
    world, rank = spec["world"], spec["rank"]
    bank, pred, succ, pairs, anchor = _site(spec, g)
    exchange, buf = PairExchange.emulated(world, rank, "cuda", NB)
    exchange.flags.copy_(torch.tensor([cur, dirty, NB, 1, leftover, 0, 0, 0, 0], dtype=torch.int32))
    buf[dirty * NB // 4 : dirty * NB // 4 + leftover // 4] = 5  # the previous call's words
    _write_peers(buf, cur, world, rank, pairs)
    return exchange, buf, bank, pred, succ, pairs, anchor


@pytest.mark.parametrize(
    "cell",
    k31_dflash2_selector_walk_mc.CELLS,
    ids=[c.why[:48] for c in k31_dflash2_selector_walk_mc.CELLS],
)
def test_walk_cells(cell):
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(_seed(cell.why))
    world, slots, rank = spec["world"], spec["slots"], spec["rank"]
    exchange, buf, bank, pred, succ, pairs, anchor = _walk_case(spec, g, cur=1, dirty=0)
    gate = bank[:, SHARD:]
    scratch = rank_top32_scratch(8, SHARD, "cuda")
    k31_dflash2_rank_top32_publish(exchange, bank[:, :SHARD], rank * SHARD, scratch)
    debug = (
        torch.empty(slots, 32, dtype=torch.int32, device="cuda"),
        torch.empty(slots, 32, dtype=torch.int32, device="cuda"),
        torch.empty(slots, 32, 32, dtype=torch.float32, device="cuda"),
    )
    args = (exchange, pred, succ, gate, anchor, spec["block"], spec["publish"])
    k31_dflash2_selector_walk_mc.is_valid(*args)
    tokens = k31_dflash2_selector_walk_mc(*args, debug=debug)
    torch.cuda.synchronize()
    gathered = torch.cat(pairs, dim=1).view(torch.float32).view(1, slots, -1)
    unary, ids = k31_dflash2_merge_top32(gathered.contiguous(), 32)
    cand, unary_bits, scores = debug
    assert torch.equal(cand, ids[0].int()), "candidates"
    assert torch.equal(unary_bits, unary[0].view(torch.int32)), "unary logits"
    saved = _tf32()
    try:
        p, s = selector_operands(pred, succ, ids, gate.view(1, slots, RANK), anchor)
        einsum = torch.einsum("blpr,blcr->blpc", p, s)[0]
        engine = k31_dflash2_selector_walk_mc.reference(gathered[0], *args[1:])
    finally:
        _restore(saved)
    rel = ((scores.double() - einsum.double()).abs().max() / einsum.abs().max()).item()
    assert rel < 1e-3, f"scores against the TF32 einsum: max rel {rel:.3e}"
    walked = selector_greedy_tokens(
        scores.view(1, slots, 32, 32), ids, spec["block"], unary_logits=unary
    )
    assert torch.equal(tokens, walked), "the walk on the kernel's scores is the engine walk"
    if torch.equal(scores, einsum):
        assert torch.equal(tokens, engine), "bitwise scores: the engine path's tokens"
    flags = exchange.flags.tolist()
    rw = _row_words(world)
    assert flags[:5] == [2, 1, NB, 1, slots * rw * 4 + kern.DRAFT_WORDS * 4] and flags[8] == 0
    assert bool((buf[: 200000 // 4] == SENTINEL).all()), "the previous call's buffer re-armed"
    tail = buf[NB // 4 + slots * rw : NB // 4 + slots * rw + kern.DRAFT_WORDS]
    if spec["publish"] and rank == 0:
        assert torch.equal(tail[:slots], tokens[0]) and bool((tail[slots:] == 0).all())
    else:
        assert bool((tail == SENTINEL).all())


@pytest.mark.parametrize(
    "cell", k31_draft_agreed_rewind.CELLS, ids=[c.why[:48] for c in k31_draft_agreed_rewind.CELLS]
)
def test_agreed_rewind_cells(cell):
    spec = cell.spec
    slots, world = spec["slots"], 16
    g = torch.Generator(device="cuda").manual_seed(_seed(cell.why))
    exchange, buf = PairExchange.emulated(world, 3, "cuda", NB)
    # The state after a walk call in buffer 1: the dirty one.
    exchange.flags.copy_(torch.tensor([2, 1, NB, 1, 0, 0, 0, 0, 0], dtype=torch.int32))
    drafts0 = torch.randint(0, VOCAB, (1, slots), generator=g, device="cuda", dtype=torch.int32)
    drafts0[0, 0] = VOCAB - 1
    off = NB // 4 + slots * _row_words(world)
    buf[off : off + slots] = drafts0[0]
    buf[off + slots : off + kern.DRAFT_WORDS] = 0
    acc = torch.randint(0, VOCAB, (1, slots + 1), generator=g, device="cuda").to(spec["acc"])
    nacc = torch.tensor([3], dtype=torch.int32, device="cuda")
    kv = torch.tensor([2 if spec.get("clamp") else 100], dtype=torch.int32, device="cuda")
    rewind = torch.tensor([5], dtype=torch.int32, device="cuda")
    local = torch.randint(0, VOCAB, (1, slots), generator=g, device="cuda", dtype=torch.int32)
    # The engine: the fused rewind on this rank's drafts, then the agreement on rank 0's.
    kv_engine = kv.clone()
    new_local = draft_next_new_tokens_rewind(acc, local, nacc, kv_engine, rewind, 0)
    reduced = torch.zeros(1, 2048, dtype=torch.float32, device="cuda")
    reduced[:, :slots] = drafts0.float()
    agreed, new_engine = k31_draft_agreed_tokens(reduced, slots, torch.int32, new_local)
    args = (exchange, slots, acc, nacc, kv, rewind)
    k31_draft_agreed_rewind.is_valid(*args)
    drafts, new = k31_draft_agreed_rewind(*args)
    torch.cuda.synchronize()
    ref_d, ref_n, ref_kv = k31_draft_agreed_rewind.reference(drafts0, acc, nacc, kv_engine, rewind)
    assert torch.equal(drafts, agreed) and torch.equal(drafts, drafts0)
    assert new.dtype == new_engine.dtype and torch.equal(new, new_engine)
    assert torch.equal(kv, kv_engine)


def test_perturbed_rank_takes_rank0_drafts():
    """The agreement through the walk's publication: 16 ranks, each with its own emulated
    workspace holding every rank's pairs, walk with ``publish_drafts``; one rank's walk is
    perturbed (its gate negated), so its local tokens differ from rank 0's. Rank 0's tail is then
    delivered to every rank's buffer (what its multicast store does), and each rank's agreed
    rewind gives rank 0's drafts and the engine's next tokens and ``kv_lens``: the unfused rewind
    on the rank's own drafts followed by the agreement on rank 0's."""
    world, slots, perturbed = 16, 7, 3
    g = torch.Generator(device="cuda").manual_seed(31)
    full = _logits(slots, world * SHARD, "randn", g)
    gate = torch.randn(slots, RANK, generator=g, device="cuda").bfloat16()
    pred = (0.2 * torch.randn(world * SHARD, RANK, generator=g, device="cuda")).bfloat16()
    succ = (0.2 * torch.randn(world * SHARD, RANK, generator=g, device="cuda")).bfloat16()
    anchor = torch.tensor([777], dtype=torch.int64, device="cuda")
    pairs = [published_pairs(full[:, r * SHARD : (r + 1) * SHARD], r * SHARD) for r in range(world)]
    scratch = rank_top32_scratch(8, SHARD, "cuda")
    rw = _row_words(world)
    tail = slice(slots * rw, slots * rw + kern.DRAFT_WORDS)  # buffer 0: the walk call's
    ranks = []
    for r in range(world):
        exchange, buf = PairExchange.emulated(world, r, "cuda", NB)
        _write_peers(buf, 0, world, r, pairs)
        bank = torch.empty(slots, SHARD + RANK, dtype=torch.bfloat16, device="cuda")
        bank[:, :SHARD] = full[:, r * SHARD : (r + 1) * SHARD]
        bank[:, SHARD:] = -gate if r == perturbed else gate
        k31_dflash2_rank_top32_publish(exchange, bank[:, :SHARD], r * SHARD, scratch)
        tokens = k31_dflash2_selector_walk_mc(
            exchange, pred, succ, bank[:, SHARD:], anchor, torch.bfloat16, True
        )
        torch.cuda.synchronize()
        if r:
            assert bool((buf[tail] == SENTINEL).all()), f"rank {r} published drafts"
        ranks.append((exchange, buf, tokens))
    drafts0 = ranks[0][2]
    assert not torch.equal(ranks[perturbed][2], drafts0), "the perturbation changed no token"
    for _, buf, _ in ranks[1:]:
        buf[tail] = ranks[0][1][tail]
    acc = torch.randint(0, VOCAB, (1, slots + 1), generator=g, device="cuda", dtype=torch.int32)
    nacc = torch.tensor([4], dtype=torch.int32, device="cuda")
    rewind = torch.tensor([3], dtype=torch.int32, device="cuda")
    for r, (exchange, _, tokens) in enumerate(ranks):
        kv = torch.tensor([500], dtype=torch.int32, device="cuda")
        kv_engine = kv.clone()
        new_local = draft_next_new_tokens_rewind(acc, tokens, nacc, kv_engine, rewind, 0)
        reduced = torch.zeros(1, 2048, dtype=torch.float32, device="cuda")
        reduced[:, :slots] = drafts0.float()
        agreed, new_engine = k31_draft_agreed_tokens(reduced, slots, torch.int32, new_local)
        drafts, new = k31_draft_agreed_rewind(exchange, slots, acc, nacc, kv, rewind)
        torch.cuda.synchronize()
        assert torch.equal(drafts, drafts0) and torch.equal(drafts, agreed), f"rank {r} drafts"
        assert torch.equal(new, new_engine), f"rank {r} next tokens"
        assert torch.equal(kv, kv_engine), f"rank {r} kv_lens"


def test_graph_replay_matches_eager():
    """publish -> walk (rank 0 publishing) -> agreed rewind, one rank (the whole exchange local):
    eager over three calls, then a graph of one call replayed three times on the same inputs
    gives the same tokens and next tokens; the state rotates through the three buffers."""
    g = torch.Generator(device="cuda").manual_seed(29)
    slots = 7
    full = _logits(slots, SHARD, "randn", g)
    bank = torch.empty(slots, SHARD + RANK, dtype=torch.bfloat16, device="cuda")
    bank[:, :SHARD] = full
    bank[:, SHARD:] = torch.randn(slots, RANK, generator=g, device="cuda").bfloat16()
    pred = (0.2 * torch.randn(SHARD, RANK, generator=g, device="cuda")).bfloat16()
    succ = (0.2 * torch.randn(SHARD, RANK, generator=g, device="cuda")).bfloat16()
    anchor = torch.tensor([5], dtype=torch.int64, device="cuda")
    exchange, buf = PairExchange.emulated(1, 0, "cuda", NB)
    scratch = rank_top32_scratch(8, SHARD, "cuda")
    acc = torch.arange(slots + 1, dtype=torch.int32, device="cuda").view(1, -1)
    nacc = torch.tensor([2], dtype=torch.int32, device="cuda")
    kv = torch.tensor([1000], dtype=torch.int32, device="cuda")
    rewind = torch.tensor([1], dtype=torch.int32, device="cuda")

    def step():
        k31_dflash2_rank_top32_publish(exchange, bank[:, :SHARD], 0, scratch)
        tok = k31_dflash2_selector_walk_mc(
            exchange, pred, succ, bank[:, SHARD:], anchor, torch.bfloat16, True
        )
        drafts, new = k31_draft_agreed_rewind(exchange, slots, acc, nacc, kv, rewind)
        return tok, drafts, new

    eager = [step() for _ in range(3)]
    for tok, drafts, new in eager:
        assert torch.equal(tok, eager[0][0]) and torch.equal(drafts, tok)
        assert torch.equal(new[:, 1:], tok) and int(new[0, 0]) == 1
    assert int(kv) == 997
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.cuda.graph(graph, stream=stream):
        out = step()
    torch.cuda.current_stream().wait_stream(stream)
    for i in range(3):
        before = int(exchange.flags[0])
        graph.replay()
        torch.cuda.synchronize()
        assert int(exchange.flags[0]) == (before + 1) % 3
        for got, want in zip(out, eager[0]):
            assert torch.equal(got, want), f"replay {i}"
