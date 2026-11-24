# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The worker's one-hop greedy tail and its agreement, with a stand-in for ``RankTopKWalk``
(the kernels themselves: ``ops/spec/test_rubin_ull_dflash2_rank_topk_walk.py``):

* ``_bind_rank_topk_walk`` binds nothing without an FP32 MNNVL all-reduce
  over the worker's group;
* ``_rank_topk_walk_tokens`` serves one request's block of the head bank (logits and gate read in
  place) and returns None, for the engine path, on anything else: no runtime, no gate, two
  requests, another row count, FP32 logits, a gate that is not the bank's; it compiles once,
  outside graph capture, and inside a capture with nothing compiled it falls back;
* a successful walk publishes rank 0's drafts;
* ``_agree_draft_tokens`` leaves the outputs alone after an agreed rewind (the flag is consumed,
  the sampled step's proposal rows too), and takes its own path on the next step.
"""

import importlib
from types import SimpleNamespace

import pytest
import torch

_PKG = (
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31."
    "kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"
)
SLOTS, SHARD, RANK = 7, 10240, 256


@pytest.fixture(scope="module")
def mod():
    return importlib.import_module(_PKG + ".modeling")


class _Runtime:
    """Records ``prepare`` / ``tokens``; ``tokens`` returns a fixed row."""

    def __init__(self):
        self.slots, self.shard = SLOTS, SHARD
        self.exchange = SimpleNamespace(rank=0)
        self.ready = set()
        self.prepared, self.calls = [], []
        self.out = torch.arange(SLOTS, dtype=torch.int32).view(1, SLOTS)

    def is_ready(self, *key):
        return key in self.ready

    def prepare(self, *key):
        self.prepared.append(key)
        self.ready.add(key)

    def tokens(self, logits, gate, pred, succ, anchor, block_dtype, publish):
        self.calls.append((logits, gate, anchor, block_dtype, publish))
        return self.out


class _Drafter:
    def __init__(self):
        self.w = {
            "d_sel_pred": torch.zeros(4, RANK, dtype=torch.bfloat16),
            "d_sel_succ": torch.zeros(4, RANK, dtype=torch.bfloat16),
        }

    def _w(self, name):
        return self.w[name]


def _worker(mod, runtime):
    worker = object.__new__(mod.DFlashWorker)
    worker._rank_topk_walk = runtime
    worker._walk_published = False
    worker._agreed_by_rewind = False
    worker._tail_path_logged = set()
    return worker


def _bank(rows=SLOTS, dtype=torch.bfloat16):
    bank = torch.zeros(rows, SHARD + RANK, dtype=dtype)
    return bank[:, :SHARD].view(1, rows, SHARD), bank[:, SHARD:]


def test_bind_requires_an_fp32_mnnvl_group(mod):
    head = SimpleNamespace(col_base=0, shard_width=SHARD)
    worker = _worker(mod, None)
    worker._vocab_parallel = SimpleNamespace(head=head)
    exchange = SimpleNamespace(tp_size=16, tp_rank=0, all_reduce=SimpleNamespace())
    worker._bind_rank_topk_walk(exchange)  # no MNNVL all-reduce
    assert worker._rank_topk_walk is None
    exchange.all_reduce.mnnvl_allreduce = SimpleNamespace(
        dtype=torch.bfloat16, mapping=SimpleNamespace(tp_size=16, tp_rank=0)
    )
    worker._bind_rank_topk_walk(exchange)  # a BF16 one
    assert worker._rank_topk_walk is None
    exchange.all_reduce.mnnvl_allreduce.dtype = torch.float32
    head.col_base = SHARD  # not this rank's shard
    worker._bind_rank_topk_walk(exchange)
    assert worker._rank_topk_walk is None


def test_tail_serves_one_request_block(mod, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    runtime = _Runtime()
    worker, drafter = _worker(mod, runtime), _Drafter()
    logits, gate = _bank()
    anchor = torch.tensor([[11]])
    tokens = worker._rank_topk_walk_tokens(drafter, logits, gate, anchor, 1)
    assert tokens is runtime.out and worker._walk_published is True
    key = (SHARD + RANK, torch.bfloat16, True, torch.bfloat16, anchor.dtype)
    assert runtime.prepared == [key]
    got_logits, got_gate, got_anchor, block_dtype, got_publish = runtime.calls[0]
    assert got_logits.shape == (SLOTS, SHARD) and got_logits.data_ptr() == logits.data_ptr()
    assert got_gate is gate and got_anchor.shape == (1,) and block_dtype == torch.bfloat16
    assert got_publish is True
    worker._rank_topk_walk_tokens(drafter, logits, gate, anchor, 1)
    assert len(runtime.prepared) == 1 and len(runtime.calls) == 2


def test_tail_falls_back(mod, monkeypatch):
    runtime = _Runtime()
    worker, drafter = _worker(mod, runtime), _Drafter()
    logits, gate = _bank()
    anchor = torch.tensor([11])
    assert _worker(mod, None)._rank_topk_walk_tokens(drafter, logits, gate, anchor, 1) is None
    assert worker._rank_topk_walk_tokens(drafter, logits, None, anchor, 1) is None
    two_logits, two_gate = _bank(2 * SLOTS)
    assert worker._rank_topk_walk_tokens(drafter, two_logits, two_gate, anchor, 2) is None
    short_logits, short_gate = _bank(SLOTS - 1)
    assert worker._rank_topk_walk_tokens(drafter, short_logits, short_gate, anchor, 1) is None
    f32_logits, f32_gate = _bank(dtype=torch.float32)
    assert worker._rank_topk_walk_tokens(drafter, f32_logits, gate, anchor, 1) is None
    assert worker._rank_topk_walk_tokens(drafter, logits, gate.contiguous(), anchor, 1) is None
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert worker._rank_topk_walk_tokens(drafter, logits, gate, anchor, 1) is None
    assert not runtime.prepared and not runtime.calls and not worker._walk_published


def test_agreement_skips_after_the_agreed_rewind(mod):
    from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

    worker = _worker(mod, None)
    worker._draft_agreement = None
    worker._rs_agree_rows = "rows"
    worker._agreed_by_rewind = True
    drafts = torch.arange(SLOTS, dtype=torch.int32).view(1, SLOTS)
    outputs = {"next_draft_tokens": drafts, "next_new_tokens": torch.zeros(1, SLOTS + 1)}
    calls = []
    original = SpecWorkerBase._agree_draft_tokens
    try:
        SpecWorkerBase._agree_draft_tokens = lambda self, out: calls.append(out)
        worker._agree_draft_tokens(outputs)
        assert not calls and outputs["next_draft_tokens"] is drafts
        assert worker._agreed_by_rewind is False and worker._rs_agree_rows is None
        worker._agree_draft_tokens(outputs)  # the next step: the engine's agreement again
        assert calls == [outputs]
    finally:
        SpecWorkerBase._agree_draft_tokens = original
