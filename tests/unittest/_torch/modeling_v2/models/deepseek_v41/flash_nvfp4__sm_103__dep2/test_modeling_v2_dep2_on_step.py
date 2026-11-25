# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The dep2 core's `on_step` gives every request scheduled in a step its own Engram look-back row, as stock's history
does, however many there are: 600 distinct requests in one step each get a row
and their look-back seeds; a released request's row goes to the next request."""

import importlib
from types import SimpleNamespace as NS

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_history_slots import (
    EngramHistorySlots,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="imports the target (CUDA build)"
)
mod = importlib.import_module(
    "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__dep2.modeling"
)
DEV = "cuda"


class _Core(NS):
    hist = property(lambda self: self.engram_slots.hist)
    req_slot = property(lambda self: self.engram_slots.req_slot)


class _Req:
    def __init__(self, rid, tokens):
        self.py_request_id, self.tokens = rid, tokens
        self.is_context_init_state = False

    def get_num_tokens(self, beam):
        return len(self.tokens)

    def get_tokens_range(self, beam, start, end):
        return self.tokens[start:end]


def _core():
    return _Core(
        engram_slots=EngramHistorySlots(mod._RING, -7, mod._MAX_REQS, DEV),
        cfg=NS(engram_max_ngram_size=3), _prev_reqs=0,
        engram_hash=NS(token_map=torch.arange(64, device=DEV, dtype=torch.int64) * 3),
    )  # fmt: skip


def test_600_requests_in_one_step():
    core = _core()
    reqs = [_Req(1000 + i, [(3 * i + k) % 64 for k in range(6)]) for i in range(600)]
    mod.ModelingV2Core.on_step(
        core, NS(context_requests=[], generation_requests=reqs), NS(max_batch_size=640)
    )
    torch.cuda.synchronize()
    rows = [core.engram_slots.row_of(r.py_request_id) for r in reqs]
    assert len(set(rows)) == 600 and max(rows) < core.engram_slots.dummy
    assert core.req_slot[:600].tolist() == rows
    for r, row in zip(reqs, rows):  # both look-back tokens of a 3-gram before the step's position 5
        assert core.hist[row, 3 % mod._RING].item() == r.tokens[3] * 3
        assert core.hist[row, 4 % mod._RING].item() == r.tokens[4] * 3


def test_a_released_row_is_reused():
    core = _core()
    a, b = _Req(1, list(range(10))), _Req(2, list(range(5, 15)))

    def step(reqs):
        return mod.ModelingV2Core.on_step(
            core, NS(context_requests=[], generation_requests=reqs), None
        )

    step([a])
    row_a = core.engram_slots.row_of(1)
    mod.ModelingV2Core.release_request_state(core, 1)
    step([b])
    assert core.engram_slots.row_of(2) == row_a and core.engram_slots.row_of(1) is None
    torch.cuda.synchronize()
    for pos in (
        7,
        8,
    ):  # b's own look-back seeds replaced a's (a.tokens[pos] differs) in the reused row
        assert core.hist[row_a, pos % mod._RING].item() == b.tokens[pos] * 3 != a.tokens[pos] * 3
