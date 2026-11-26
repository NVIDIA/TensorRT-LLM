# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S4FE on the tp2 target: the steps S4's chain declines run the in-tree Markov chain with the fused tail
(``ops/spec/m31_markov_epilogue.py``). CPU: the routing (only full-vocab, unsharded steps with a Markov head while S4 is
not serving; the in-tree path otherwise). GPU (SM107, the op built): the certified cells and the
edge cases bitwise the in-tree chain, eager and under graph replay, and the counters left zero. The worker's
padding-row zeroing (``minimax_m31_zero_padding_rows``, one launch) is bitwise the stock
``SpecWorkerBase._zero_padding_rows`` with 0 / 1 / all padding requests at 3 and 4 rows per request."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2 import (  # noqa: E501
        dspark_markov_worker as w1,
    )
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2.ops.spec import (  # noqa: E501
        m31_markov_epilogue as epi,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

VOCAB = 16


class _Base:
    mapping = SimpleNamespace(tp_rank=0)

    def __init__(self, sharded=False):
        self.calls = []
        self.sharded = sharded

    def _refine_block_logits(self, draft_model, gen_logits, inputs, spec_metadata):
        self.calls.append("in-tree")
        return "in-tree"

    def _draft_logits_are_sharded(self, logits, spec_metadata):
        return self.sharded


class _Worker(w1.S4WorkerMixin, _Base):
    pass


def _draft(markov=True, vocab=VOCAB):
    return SimpleNamespace(
        has_markov_head=markov, markov_w1=torch.zeros(vocab, 4), markov_w2=torch.zeros(vocab, 4)
    )


def _route(monkeypatch, wk, draft, logits, result="fused"):
    seen = []
    monkeypatch.setattr(w1, "m31_markov_chain_fused", lambda *a: seen.append(a) or result)
    out = wk._refine_block_logits(
        draft, logits, {"first_prev_tokens": torch.zeros(logits.shape[0])}, None
    )
    return out, seen


def test_a_declined_full_vocab_step_takes_the_fused_chain(monkeypatch):
    wk = _Worker()
    out, seen = _route(monkeypatch, wk, _draft(), torch.zeros(13, 3, VOCAB))
    assert out == "fused" and len(seen) == 1 and wk.calls == []


_GPU = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the S4FE op runs on SM107",
)
_RANK, _VOCAB = 256, 200064


def _inputs(b, k, seed, case="randn"):
    g = torch.Generator(device="cuda").manual_seed(seed)
    ws = {"lo": 2.0**-12, "hi": 2.0**12}.get(case, 1.0)
    base = torch.randn(b, k, _VOCAB, device="cuda", generator=g) * 3 * ws
    if case == "ties":
        base = torch.round(base * 2) / 2
    w1_ = (torch.randn(_VOCAB, _RANK, device="cuda", generator=g) * 0.05 * ws).bfloat16()
    w2_ = (torch.randn(_VOCAB, _RANK, device="cuda", generator=g) * 0.05 * ws).bfloat16()
    prev = torch.randint(0, _VOCAB, (b,), device="cuda", generator=g)
    if case == "special":
        base[0, :, 77] = float("nan")
        base[0, :, 5] = float("nan")
        base[1, :, 1000:150000] = float("-inf")
        base[2] = float("-inf")
        prev[3] = -1
    return base.contiguous(), prev, w1_, w2_


def _same(a, b):
    return torch.equal(a.view(torch.int32), b.view(torch.int32))


@_GPU
@pytest.mark.parametrize(
    "cell", epi.M31MarkovChainFused.CELLS, ids=[c.why[:40] for c in epi.M31MarkovChainFused.CELLS]
)
def test_cells_bitwise_the_in_tree_chain(cell):
    assert epi.markov_epilogue_op() is not None, "this build lacks the Markov epilogue ops"
    op = epi.m31_markov_chain_fused
    for seed in range(2):
        args = _inputs(cell.spec["b"], cell.spec["k"], seed)
        assert _same(op(*args), op.reference(*args))


@_GPU
@pytest.mark.parametrize("case", ["ties", "lo", "hi", "special"])
def test_edge_cases_bitwise_the_in_tree_chain(case):
    assert epi.markov_epilogue_op() is not None, "this build lacks the Markov epilogue ops"
    op = epi.m31_markov_chain_fused
    args = _inputs(16, 3, 11, case)
    assert _same(op(*args), op.reference(*args))


@_GPU
@pytest.mark.parametrize("b,k", [(16, 3), (112, 3), (56, 7)])
def test_graph_replay_matches_eager_and_leaves_the_counters_zero(b, k):
    assert epi.markov_epilogue_op() is not None, "this build lacks the Markov epilogue ops"
    op = epi.m31_markov_chain_fused
    args = _inputs(b, k, 5)
    eager = op(*args)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
        out = op(*args)
    torch.cuda.current_stream().wait_stream(s)
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    assert _same(out, eager) and _same(out, op.reference(*args))
    _, counters = epi.prebuild(args[0].device)
    assert int(counters.abs().sum()) == 0


@_GPU
@pytest.mark.parametrize("rpr", [3, 4])
@pytest.mark.parametrize("npad", [0, 1, 4])
def test_zero_padding_rows_bitwise_the_stock_zeroing(rpr, npad):
    assert epi.markov_epilogue_op() is not None, "this build lacks the Markov epilogue ops"
    dummy, nc, ng, v = 7, 2, 4, 200064
    slots = torch.arange(10, 10 + nc + ng, dtype=torch.int64, device="cuda")
    slots[nc + ng - npad : nc + ng] = dummy
    meta = SimpleNamespace(batch_slot_ids=slots, dummy_slot_row=dummy, is_ragged_verify=False)
    logits = torch.randn(ng * rpr, v, device="cuda")
    logits[0, 5] = float("nan")
    ref = SpecWorkerBase._zero_padding_rows(logits.clone(), meta, nc, nc + ng, rpr)
    got = w1.S4WorkerMixin._zero_padding_rows(logits.clone(), meta, nc, nc + ng, rpr)
    assert torch.equal(ref.view(torch.int32), got.view(torch.int32))
