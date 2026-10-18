# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the draft-head half of the drafter PDL chain (``pdl_head``, ``selector_fork``).

* ``pdl_head``: the draft head's ``_draft_slot_ids`` hands the built-in step a row range
  whose covering ``clamp`` is a ``slice``; the built-in two lines
  (``ids.clamp(max=rows - 1)``, ``hidden_states_out[ids]``) then give the view
  of the very rows the gather copies -- the same bits, the same shape and
  contiguity for the LM head's low-M dispatcher -- and launch nothing.
* ``selector_fork``: the selector's ``hidden_projection`` issued on the side stream
  by the draft head's logits processor is the one ``selector_draft_tokens`` (and the
  built-in selector) then use: the same bits and the same draft tokens,
  eagerly and in a captured graph that forks and joins; a projection that is
  not taken is joined by the next fork. With more than one request (K rows
  each) nothing forks: the projection runs inline, eagerly and in a captured
  graph that stays one chain.

``MODELING_V2_TIMING=1`` also times the draft head from the last hop to the
draft tokens on one rank: a hop stand-in that releases its dependents early,
the draft LM-head GEMM of the TP8 shard (flashinfer split-K with PDL,
default tactic, 7 rows, four distinct 414 MB weights so each call streams from
HBM), then the draft head's real ``selector_draft_tokens`` with an all-reduce
stand-in: base (clamp + gather, projection inline), ``pdl_head`` (view), ``selector_fork``
(projection forked), both; the draft tokens of every arm must be the base's.
"""

from __future__ import annotations

import os
import statistics
import types

import pytest
import torch
import triton
import triton.language as tl
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.draft_head import (  # noqa: E501
    head as draft_head,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.drafter_pdl_chain import (  # noqa: E501
    chain as drafter_pdl,
)

__extra_import_path__ = ["../draft_head"]
from _draft_head_cases import TP, draft_shard_topk, draft_step, selector  # noqa: E402

assert torch.cuda.is_available(), "the draft head requires a CUDA device"

HIDDEN, K, BLOCK = 8192, 7, 8


def _worker():
    worker = object.__new__(draft_head.ShardedDraftHeadDFlashWorker)
    worker.__dict__["_pdl_head"] = True
    return worker


def _builtin_lines(worker, hidden_states_out, num_gens=1, block_size=BLOCK, k=K):
    """The built-in step's draft-row lines (``DFlashWorker._forward_impl``), verbatim."""
    gen_gather_ids = worker._draft_slot_ids(None, num_gens, block_size, k)
    gen_gather_ids = gen_gather_ids.clamp(max=hidden_states_out.shape[0] - 1)
    return hidden_states_out[gen_gather_ids]


@torch.inference_mode()
def test_pdl_head_rows_are_the_gathered_rows_as_a_view():
    from tensorrt_llm._torch.modules.low_m_gemm import LowMGemmDispatcher
    from tensorrt_llm._torch.speculative.dflash import dflash_draft_slot_ids

    worker = _worker()
    hs = torch.randn(BLOCK, HIDDEN, device="cuda").to(torch.bfloat16)
    rows = _builtin_lines(worker, hs)
    ids = dflash_draft_slot_ids(1, BLOCK, K, False).clamp(max=BLOCK - 1)
    want = hs[ids]
    assert torch.equal(rows.view(torch.int16), want.view(torch.int16))
    assert rows.data_ptr() == hs[1].data_ptr() and rows.is_contiguous()
    assert rows.shape == want.shape and rows.stride() == want.stride()
    weight = torch.empty(25256, HIDDEN, dtype=torch.bfloat16, device="cuda")
    assert LowMGemmDispatcher._is_candidate_shape(rows, weight, None)
    assert LowMGemmDispatcher._is_candidate_shape(want, weight, None)
    # Two requests: the built-in gather.
    hs2 = torch.randn(2 * BLOCK, HIDDEN, device="cuda").to(torch.bfloat16)
    ids2 = worker._draft_slot_ids(None, 2, BLOCK, K)
    assert isinstance(ids2, torch.Tensor)
    rows2 = _builtin_lines(worker, hs2, num_gens=2)
    assert torch.equal(rows2, hs2[dflash_draft_slot_ids(2, BLOCK, K, False)])
    # Without the flag: the draft head's cached tensor.
    worker.__dict__["_pdl_head"] = False
    assert isinstance(worker._draft_slot_ids(None, 1, BLOCK, K), torch.Tensor)


@torch.inference_mode()
def test_pdl_head_captures_no_kernel_for_the_draft_rows():
    from cuda.bindings import driver

    worker = _worker()
    hs = torch.randn(BLOCK, HIDDEN, device="cuda").to(torch.bfloat16)
    _builtin_lines(worker, hs)  # fills the draft head's cache and the range outside the capture
    out = torch.empty(K, HIDDEN, dtype=torch.bfloat16, device="cuda")
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        rows = _builtin_lines(worker, hs)
        out.copy_(rows)
    handle = driver.CUgraph(graph.raw_cuda_graph())
    status, _, count = driver.cuGraphGetNodes(handle, 0)
    assert status == driver.CUresult.CUDA_SUCCESS
    assert count == 1, f"{count} nodes: the draft rows launched kernels"
    hs.copy_(torch.randn(BLOCK, HIDDEN, device="cuda").to(torch.bfloat16))
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, hs[1:BLOCK])


def test_pdl_head_first_requested_inside_a_capture_is_the_tensor():
    worker = _worker()
    hs = torch.randn(BLOCK, HIDDEN, device="cuda").to(torch.bfloat16)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ids = worker._draft_slot_ids(None, 1, BLOCK, K)
        assert isinstance(ids, torch.Tensor)
        rows = hs[ids.clamp(max=BLOCK - 1)]
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(rows, hs[1:BLOCK])


# --------------------------------------------------------------------------
# selector_fork
# --------------------------------------------------------------------------
class _FakeAllReduce:
    def __init__(self, shards, rank):
        self.packed = [
            draft_shard_topk(s, rank=r, num_ranks=len(shards)) for r, s in enumerate(shards)
        ]
        self.rank = rank

    def __call__(self, tensor):
        return torch.stack(self.packed).sum(dim=0)


def _forked_selector():
    """A copy of the test selector whose projection is the forked one, and its fork."""
    import copy

    base = selector()
    forked = copy.deepcopy(base)
    forked.hidden_projection.__class__ = drafter_pdl.ForkedSelectorProjection
    fork = drafter_pdl.SelectorFork(forked.hidden_projection)
    forked.hidden_projection._selector_fork = fork
    return base, forked, fork


class _Head:
    """``gathers_column_shards`` admits it; ``apply_linear`` is a bf16 GEMM of the shard."""

    tp_mode = None

    def __init__(self, shard_weight):
        from tensorrt_llm._torch.modules.linear import TensorParallelMode

        self.tp_mode = TensorParallelMode.COLUMN
        self.gather_output = True
        self.tp_size = TP
        self.padding_size = 0
        self.gather_output_sizes = None
        self.lora = None
        self.enable_lm_head_tp_in_adp = False
        self.weight = shard_weight
        self.bias = None

    def apply_linear(self, x, bias):
        return torch.nn.functional.linear(x, self.weight)


@torch.inference_mode()
def test_selector_fork_projection_is_the_inline_bits_and_tokens():
    base, forked, fork = _forked_selector()
    shards, block_hidden, anchor = draft_step(1, K, "gemm", 4141)
    rows = block_hidden.reshape(K, HIDDEN).contiguous()
    block = rows.view(1, K, HIDDEN)
    want_gate = base.hidden_projection(block)
    weight = torch.randn(4096, HIDDEN, device="cuda").to(torch.bfloat16)
    processor = draft_head.ShardDraftLogitsProcessor(nn.Identity())
    processor._selector_fork = fork
    for rank in (0, 3, 7):
        mapping = types.SimpleNamespace(tp_size=TP, tp_rank=rank)
        local = shards[rank].view(1, K, -1)
        want = draft_head.selector_draft_tokens(
            local, block, anchor, base, mapping, _FakeAllReduce(shards, rank)
        )
        processor(rows, _Head(weight), None, True)  # forks the projection
        got = draft_head.selector_draft_tokens(
            local, block, anchor, forked, mapping, _FakeAllReduce(shards, rank)
        )
        assert torch.equal(got, want), f"rank {rank}"
    assert fork.launches == 3 and fork.taken == 3
    fork.launch(rows)
    gate = forked.hidden_projection(block)
    assert torch.equal(gate.view(torch.int16), want_gate.view(torch.int16))
    # A call on other rows computes inline and joins the pending projection.
    fork.launch(rows)
    other = torch.randn_like(block)
    assert torch.equal(forked.hidden_projection(other), base.hidden_projection(other))
    assert fork._pending is None and fork.taken == 4


@torch.inference_mode()
def test_selector_fork_forks_and_joins_inside_a_captured_graph():
    base, forked, fork = _forked_selector()
    rows = torch.randn(K, HIDDEN, device="cuda").to(torch.bfloat16)
    weight = torch.randn(25256, HIDDEN, device="cuda").to(torch.bfloat16) / HIDDEN**0.5
    head = _Head(weight)
    processor = draft_head.ShardDraftLogitsProcessor(nn.Identity())
    processor._selector_fork = fork

    def step():
        logits = processor(rows, head, None, True)
        gate = forked.hidden_projection(rows.view(1, K, HIDDEN))
        return logits, gate

    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits, gate = step()
    for seed in range(5):
        g = torch.Generator(device="cuda").manual_seed(seed)
        rows.copy_(torch.randn(K, HIDDEN, generator=g, device="cuda").to(torch.bfloat16))
        graph.replay()
        torch.cuda.synchronize()
        want = base.hidden_projection(rows.view(1, K, HIDDEN))
        assert torch.equal(gate.view(torch.int16), want.view(torch.int16)), f"replay {seed}"
        assert torch.equal(logits, torch.nn.functional.linear(rows, weight))


@pytest.mark.parametrize("requests", [2, 3])
@torch.inference_mode()
def test_selector_fork_more_than_one_request_forks_nothing(requests):
    """B > 1 (K rows per request): no fork, the projection inline; the base bits and tokens.

    Eagerly and in a captured graph (one stream: the capture would fail on an
    unjoined fork), over replays with new rows.
    """
    from cuda.bindings import driver

    assert drafter_pdl.HEAD_ROWS == K
    base, forked, fork = _forked_selector()
    shards, block_hidden, anchor = draft_step(requests, K, "gemm", 4343 + requests)
    rows = block_hidden.reshape(requests * K, HIDDEN).contiguous()
    block = rows.view(requests, K, HIDDEN)
    weight = torch.randn(4096, HIDDEN, device="cuda").to(torch.bfloat16) / HIDDEN**0.5
    head = _Head(weight)
    processor = draft_head.ShardDraftLogitsProcessor(nn.Identity())
    processor._selector_fork = fork
    for rank in (0, 5):
        mapping = types.SimpleNamespace(tp_size=TP, tp_rank=rank)
        local = shards[rank].view(requests, K, -1)
        want = draft_head.selector_draft_tokens(
            local, block, anchor, base, mapping, _FakeAllReduce(shards, rank)
        )
        logits = processor(rows, head, None, True)
        assert fork._pending is None, "a fork for more than one request"
        assert torch.equal(logits, torch.nn.functional.linear(rows, weight))
        got = draft_head.selector_draft_tokens(
            local, block, anchor, forked, mapping, _FakeAllReduce(shards, rank)
        )
        assert torch.equal(got, want), f"rank {rank}"
    assert fork.launches == 0 and fork.taken == 0

    def step():
        logits = processor(rows, head, None, True)
        return logits, forked.hidden_projection(block)

    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        logits, gate = step()
    assert fork.launches == 0 and fork._pending is None
    handle = driver.CUgraph(graph.raw_cuda_graph())
    status, _, count = driver.cuGraphGetNodes(handle, 0)
    assert status == driver.CUresult.CUDA_SUCCESS
    get_edges = getattr(driver, "cuGraphGetEdges_v2", None) or driver.cuGraphGetEdges
    result = get_edges(handle, 0)
    assert result[0] == driver.CUresult.CUDA_SUCCESS
    edges = int(result[-1])
    assert edges == count - 1, f"{count} nodes, {edges} edges: not one chain"
    for seed in range(5):
        g = torch.Generator(device="cuda").manual_seed(seed)
        rows.copy_(torch.randn(requests * K, HIDDEN, generator=g, device="cuda").to(torch.bfloat16))
        graph.replay()
        torch.cuda.synchronize()
        want = base.hidden_projection(block)
        assert torch.equal(gate.view(torch.int16), want.view(torch.int16)), f"replay {seed}"
        assert torch.equal(logits, torch.nn.functional.linear(rows, weight))


# --------------------------------------------------------------------------
# Head-chain microbenchmark (MODELING_V2_TIMING=1)
# --------------------------------------------------------------------------
@triton.jit
def _globaltimer(dummy):
    return tl.inline_asm_elementwise(
        "mov.u64 $0, %globaltimer;", "=l,r", [dummy], dtype=tl.int64, is_pure=False, pack=1
    )


@triton.jit
def _hop_stand_in_kernel(dst, n, delay_ns, BLOCK_N: tl.constexpr):
    """The last drafter hop: waits, triggers early (after its transport), works, stores."""
    tl.extra.cuda.gdc_wait()
    tl.extra.cuda.gdc_launch_dependents()
    start = _globaltimer(0)
    now = start
    while now - start < delay_ns:
        now = _globaltimer(0)
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < n
    tl.store(dst + offsets, (offsets % 13).to(tl.bfloat16) * 0.125, mask=mask)


@triton.jit
def _all_reduce_stand_in_kernel(src, n, delay_ns, BLOCK_N: tl.constexpr):
    """The key all-reduce's time on 28 CTAs (plain launch); the sum is precomputed."""
    start = _globaltimer(0)
    now = start
    while now - start < delay_ns:
        now = _globaltimer(0)
    offsets = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offsets < n
    values = tl.load(src + offsets, mask=mask)
    tl.store(src + offsets, values, mask=mask)


class _SpinAllReduce:
    """``selector_draft_tokens``' all-reduce: a spin on 28 CTAs, then the 8-rank sum."""

    def __init__(self, shards):
        self.total = _FakeAllReduce(shards, 0)(None)

    def __call__(self, tensor):
        _all_reduce_stand_in_kernel[(28,)](tensor, tensor.numel(), 4_400, BLOCK_N=256, num_warps=4)
        return self.total


@pytest.mark.skipif(os.environ.get("MODELING_V2_TIMING") != "1", reason="set MODELING_V2_TIMING=1")
@torch.inference_mode()
def test_microbenchmark_head_chain():
    """The draft head from the last hop to the draft tokens, one rank.

    Hop stand-in (64 CTAs, triggers early, 6 us) -> draft rows (clamp + gather,
    or pdl_head's view) -> logits processor (selector_fork forks the projection here) ->
    the LM-head GEMM (flashinfer split-K with PDL, default tactic, four
    distinct 414 MB weights so each call streams from HBM) -> the draft head's real
    ``selector_draft_tokens``: shard top-k, the key all-reduce (stand-in:
    a spin on 28 CTAs, then the 8-rank sum of ``draft_step``'s shards),
    the projection (inline, or selector_fork's join), merge, einsum, walk.
    """
    from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import (
        default_tactic,
        run_splitk_dense,
    )

    from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

    assert get_env_enable_pdl()
    vocab_shard = 25256
    weights = [
        (torch.randn(vocab_shard, HIDDEN, device="cuda") / HIDDEN**0.5).to(torch.bfloat16)
        for _ in range(4)
    ]
    hs = torch.zeros(BLOCK, HIDDEN, dtype=torch.bfloat16, device="cuda")
    base, forked, fork = _forked_selector()
    worker = _worker()
    plain = object.__new__(draft_head.ShardedDraftHeadDFlashWorker)
    tactic = default_tactic(K, vocab_shard, HIDDEN)
    outs = [torch.empty(K, vocab_shard, dtype=torch.bfloat16, device="cuda") for _ in weights]
    shards, _, anchor = draft_step(1, K, "gemm", 4242)
    allreduce = _SpinAllReduce(shards)
    mapping = types.SimpleNamespace(tp_size=TP, tp_rank=0)
    head = _Head(weights[0])
    processor = draft_head.ShardDraftLogitsProcessor(nn.Identity())

    def chain(pdl_head_on, selector_fork_on, weight, out):
        _hop_stand_in_kernel[(64,)](
            hs, hs.numel(), 6_000, BLOCK_N=1024, num_warps=4, launch_pdl=True
        )
        rows = _builtin_lines(worker if pdl_head_on else plain, hs)
        processor.__dict__["_selector_fork"] = fork if selector_fork_on else None
        head.weight = weight
        head.apply_linear = lambda x, bias: run_splitk_dense(x, weight.t(), None, out, True, tactic)
        local = processor(rows, head, None, True)
        selector = forked if selector_fork_on else base
        return draft_head.selector_draft_tokens(
            local.view(1, K, -1), rows.view(1, K, HIDDEN), anchor, selector, mapping, allreduce
        )

    arms = {
        "base": (False, False),
        "pdl_head": (True, False),
        "selector_fork": (False, True),
        "hs": (True, True),
    }
    graphs, tokens = {}, {}
    for arm, (h, s) in arms.items():
        for w, o in zip(weights, outs):
            chain(h, s, w, o)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            tokens[arm] = [chain(h, s, w, o) for w, o in zip(weights, outs)]
        graphs[arm] = graph
        for _ in range(10):
            graph.replay()
    torch.cuda.synchronize()
    for arm in arms:
        for got, want in zip(tokens[arm], tokens["base"]):
            assert torch.equal(got, want), f"{arm}: draft tokens differ from the base arm's"
    # Two selector_fork arms, each one eager warm-up pass and one captured pass: every fork taken.
    assert fork.launches == fork.taken == 2 * 2 * len(weights)
    samples = {arm: [] for arm in arms}
    for rnd in range(6):
        order = list(arms) if rnd % 2 == 0 else list(reversed(arms))
        for arm in order:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(50):
                graphs[arm].replay()
            end.record()
            end.synchronize()
            samples[arm].append(start.elapsed_time(end) * 1000.0 / 50 / len(weights))
    print(
        "\ndraft head (hop stand-in -> rows -> LM head -> top-k -> AR stand-in -> projection -> merge"
    )
    print("-> einsum -> walk), us per head:")
    for arm in arms:
        med = statistics.median(samples[arm])
        delta = med - statistics.median(samples["base"])
        rounds = " ".join(f"{x:7.2f}" for x in samples[arm])
        print(
            f"  {arm:5s} median {med:7.2f}  min {min(samples[arm]):7.2f}  delta {delta:+6.2f}"
            f"  rounds {rounds}"
        )
