# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S4FE on the tp2 target: the steps S4's chain declines run the in-tree Markov chain with the fused tail
(``ops/spec/m31_markov_epilogue.py``). CPU: the routing (only full-vocab, unsharded steps with a Markov head while S4 is
not serving; the in-tree path otherwise) and the kill switch. GPU (SM107, the op built): the certified cells and the
edge cases bitwise the in-tree chain, eager and under graph replay, and the counters left zero."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
    dspark_s4 as w1,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.ops.spec import (  # noqa: E501
    m31_markov_epilogue as epi,
)

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


def test_s4fe_is_on_by_default():
    assert w1.S4FE


def test_a_declined_full_vocab_step_takes_the_fused_chain(monkeypatch):
    wk = _Worker()
    out, seen = _route(monkeypatch, wk, _draft(), torch.zeros(13, 3, VOCAB))
    assert out == "fused" and len(seen) == 1 and wk.calls == []


@pytest.mark.parametrize(
    "why, worker, draft, width",
    [
        ("no Markov head", _Worker(), _draft(markov=False), VOCAB),
        ("sharded logits", _Worker(sharded=True), _draft(), VOCAB),
        ("a vocab shard", _Worker(), _draft(), VOCAB // 2),
    ],
)
def test_other_steps_keep_the_in_tree_chain(monkeypatch, why, worker, draft, width):
    out, seen = _route(monkeypatch, worker, draft, torch.zeros(13, 3, width))
    assert out == "in-tree" and seen == [] and worker.calls == ["in-tree"], why


def test_an_op_decline_keeps_the_in_tree_chain(monkeypatch):
    wk = _Worker()
    out, seen = _route(monkeypatch, wk, _draft(), torch.zeros(13, 3, VOCAB), result=None)
    assert out == "in-tree" and len(seen) == 1 and wk.calls == ["in-tree"]


def test_a_served_s4_step_never_reaches_s4fe(monkeypatch):
    wk = _Worker()
    wk._s4_step = SimpleNamespace(
        shard=VOCAB // 2
    )  # an S4 step whose logits are not its shard: S4 falls through
    out, seen = _route(monkeypatch, wk, _draft(), torch.zeros(13, 3, VOCAB))
    assert out == "in-tree" and seen == []


def test_kill_switch():
    code = f"from {w1.__name__} import S4FE\nprint(S4FE)"
    env = {**os.environ, "TRTLLM_M31_DISABLE_S4FE": "1"}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.split()[-1] == "False"


_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 7)
    or epi.markov_epilogue_op() is None,
    reason="the S4FE op runs on SM107 with the op built",
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
    op = epi.m31_markov_chain_fused
    for seed in range(2):
        args = _inputs(cell.spec["b"], cell.spec["k"], seed)
        assert _same(op(*args), op.reference(*args))


@_GPU
@pytest.mark.parametrize("case", ["ties", "lo", "hi", "special"])
def test_edge_cases_bitwise_the_in_tree_chain(case):
    op = epi.m31_markov_chain_fused
    args = _inputs(16, 3, 11, case)
    assert _same(op(*args), op.reference(*args))


@_GPU
@pytest.mark.parametrize("b,k", [(16, 3), (32, 3), (56, 3), (112, 3), (56, 7)])
def test_graph_replay_matches_eager_and_leaves_the_counters_zero(b, k):
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


def test_wider_steps_decline():
    base = torch.zeros(epi.MAX_ROWS + 1, 3, 16)
    assert (
        epi.m31_markov_chain_fused(
            base, torch.zeros(epi.MAX_ROWS + 1), torch.zeros(16, 4), torch.zeros(16, 4)
        )
        is None
    )
