# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A context step in which no attention-DP rank has a row in the decoder half (every rank's chunk ends before its
prompt's final window) still publishes the replay-boundary owner's (layer 20's) compressed GLOBAL / INDEX rows of every
row, as stock's boundary (`prepare_ced_global_kv`) does unconditionally.

The core used to return before the boundary in that case, so the layer-20 rows of such chunks were never written: with
two long prompts prefilled side by side on the two ranks, the generation worker read 64K zero GLOBAL / INDEX rows per
request (ctx15: our dep2 ctx + stock TEP8 gen, the layer-20 GVR top-k several times slower at >= 256K tokens).
Host-only: the real `_run_layers` around stub layers.
"""

import importlib
from types import SimpleNamespace as NS

import pytest
import torch
from torch import nn

modeling = importlib.import_module(
    "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__dep2.modeling"
)

SPLIT, LAYERS, HC, DIM = 2, 4, 2, 8


class _Layer:
    def __init__(self, idx, log):
        self.idx, self.log = idx, log

    def __call__(
        self, x, pre_mix, ctx, attn_metadata, engram_ids, counts, enc_counts, pending, defer=True
    ):
        self.log.append(("run", self.idx, x.shape[0]))
        return x + 1, pre_mix, None

    def publish_global(self, x, pre_mix, ctx):
        self.log.append(("publish", self.idx, x.shape[0], ctx))


def _core(log):
    core = object.__new__(modeling.ModelingV2Core)
    nn.Module.__init__(core)
    core.layers = [_Layer(i, log) for i in range(LAYERS)]
    core.decoder_replay_split, core.hc = SPLIT, HC
    core.engram_hash = NS(lookback=lambda *a: None)
    # the Engram look-back rows (`req_slot` / `hist` read them)
    core.engram_slots = NS(req_slot=torch.zeros(4, dtype=torch.int32), hist=None)
    core.embed_tokens = None
    return core


@pytest.fixture(autouse=True)
def _embedding(monkeypatch):
    monkeypatch.setattr(
        modeling, "token_embedding", lambda table, ids: torch.ones(ids.shape[0], DIM)
    )


def _run(n, counts):
    log = []
    plan = NS(replay_all_rank_num_tokens=counts)
    step = NS(positions=torch.arange(n), n=n, tok_req=torch.zeros(n, dtype=torch.int64))
    ctx = object()
    out = _core(log)._run_layers(
        step,
        ctx,
        NS(decoder_replay_plan=plan, all_rank_num_tokens=[n, 7]),
        torch.zeros(n, dtype=torch.int32),
    )
    return out, log, ctx


def test_boundary_publishes_when_no_rank_replays():
    out, log, ctx = _run(5, [0, 0])
    assert log == [
        ("run", 0, 5),
        ("run", 1, 5),
        ("publish", SPLIT, 5, ctx),
    ]  # the owner's rows, then no decoder layer
    assert out.shape == (5, DIM) and not out.any()


def test_a_rank_without_rows_publishes_nothing():
    out, log, _ = _run(0, [0, 0])
    assert ("publish", SPLIT, 0) not in [e[:3] for e in log] and not any(
        e[0] == "publish" for e in log
    )
    assert out.shape == (0, DIM)
