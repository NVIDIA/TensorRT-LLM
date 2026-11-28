# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The core's `on_step` seeds a request new to the engine with its engram look-back tokens read through
`get_tokens_range` (only the `max_ngram - 1` tokens before its first position, as the engine's
`prepare_disagg_generation_request` reads them), never the whole history (`get_tokens`, O(prompt) host time for a
~1M-token generation request), and the ring holds those tokens (compressed ids) at their positions. A step with the
same requests in the same rows does nothing (their slots are on the device); a churn step re-places the rows."""

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
    "tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__tep8_moe_tp4_ep2.modeling"
)
DEV = "cuda"


class _Core(NS):
    """The core's on_step state; `hist` / `req_slot` are the slots helper's, as on the core."""

    hist = property(lambda self: self.engram_slots.hist)
    req_slot = property(lambda self: self.engram_slots.req_slot)


class _Req:
    def __init__(self, rid, tokens, context):
        self.py_request_id, self.tokens = rid, tokens
        self.is_context_init_state = context
        self.context_current_position = len(tokens) if context else 0
        self.ranges = []

    def get_num_tokens(self, beam):
        return len(self.tokens)

    def get_tokens_range(self, beam, start, end):
        self.ranges.append((start, end))
        return self.tokens[start:end]

    def get_tokens(self, beam):
        raise AssertionError("on_step must not read a request's whole history")


@pytest.mark.parametrize("history", [0, 1, 2, 3, 1000003])
@pytest.mark.parametrize("context", [False, True])
def test_new_request_seeds_only_the_look_back(history, context):
    vocab = 64
    core = _Core(
        engram_slots=EngramHistorySlots(mod._RING, -7, mod._MAX_REQS, DEV),
        cfg=NS(engram_max_ngram_size=3), _prev_reqs=0, _prev_ids=None, _staging={},
        engram_hash=NS(token_map=torch.arange(vocab, device=DEV, dtype=torch.int64) * 3),
    )  # fmt: skip
    tokens = [(7 * i + 1) % vocab for i in range(history + (0 if context else 1))]
    req = _Req(11, tokens, context)
    mod.ModelingV2Core.on_step(
        core,
        NS(context_requests=[req] if context else [], generation_requests=[] if context else [req]),
        None,
    )
    torch.cuda.synchronize()
    p0 = history if context else len(tokens) - 1
    lo = max(0, p0 - 2)
    assert req.ranges == ([(lo, p0)] if p0 > lo else [])
    slot = core.engram_slots.row_of(11)
    for pos in range(lo, p0):
        assert int(core.hist[slot, pos % mod._RING]) == tokens[pos] * 3, pos
    assert int(core.req_slot[0]) == slot


def _core(vocab=64):
    return _Core(
        engram_slots=EngramHistorySlots(mod._RING, -7, mod._MAX_REQS, DEV),
        cfg=NS(engram_max_ngram_size=3), _prev_reqs=0, _prev_ids=None, _staging={},
        engram_hash=NS(token_map=torch.arange(vocab, device=DEV, dtype=torch.int64) * 3),
    )  # fmt: skip


def test_unchanged_requests_do_nothing_and_churn_replaces_the_rows():
    core = _core()
    a, b, c = (_Req(i, [(5 * i + k) % 64 for k in range(10)], False) for i in (1, 2, 3))

    def step(reqs):
        return mod.ModelingV2Core.on_step(
            core, NS(context_requests=[], generation_requests=reqs), None
        )

    step([a, b])
    torch.cuda.synchronize()
    before = core.req_slot.clone()
    hist = core.hist.clone()
    a.ranges.clear(), b.ranges.clear()
    core.req_slot[:2] = -5  # a step with the same requests must not rewrite the rows
    step([a, b])
    torch.cuda.synchronize()
    assert core.req_slot[:2].tolist() == [-5, -5] and not a.ranges and not b.ranges
    assert torch.equal(core.hist, hist)
    core.req_slot.copy_(before)
    step([b, c])  # a left, c entered: rows re-placed, c seeded, a's slot kept for it
    torch.cuda.synchronize()
    assert core.req_slot[:2].tolist() == [core.engram_slots.row_of(2), core.engram_slots.row_of(3)]
    assert int(core.req_slot[2]) == core.engram_slots.dummy
    assert c.ranges == [(7, 9)] and not b.ranges
    step([b])  # fewer rows: the stale row past the requests is padded
    torch.cuda.synchronize()
    assert core.req_slot[:2].tolist() == [core.engram_slots.row_of(2), core.engram_slots.dummy]


def test_600_requests_in_one_step():
    """Stock's history keeps a row for every request the rank holds: 600 distinct requests scheduled in one step (the
    KV cache manager holds 640) each get their own row and their seeds; none is taken from another live request."""
    core = _core()
    reqs = [_Req(1000 + i, [(3 * i + k) % 64 for k in range(6)], False) for i in range(600)]
    mod.ModelingV2Core.on_step(
        core, NS(context_requests=[], generation_requests=reqs), NS(max_batch_size=640)
    )
    torch.cuda.synchronize()
    rows = [core.engram_slots.row_of(r.py_request_id) for r in reqs]
    assert (
        len(set(rows)) == 600
        and core.engram_slots.dummy >= 640
        and max(rows) < core.engram_slots.dummy
    )
    assert core.req_slot[:600].tolist() == rows
    for r, row in zip(reqs, rows):
        assert (
            core.hist[row, 4 % mod._RING].item() == r.tokens[4] * 3
        )  # its look-back seed, in its own row


def test_a_released_row_is_reused_and_an_unscheduled_request_keeps_its_row():
    core = _core()
    a, b, c = (_Req(i, [(5 * i + k) % 64 for k in range(10)], False) for i in (1, 2, 3))

    def step(reqs):
        return mod.ModelingV2Core.on_step(
            core, NS(context_requests=[], generation_requests=reqs), None
        )

    step([a, b])
    row_a, row_b = core.engram_slots.row_of(1), core.engram_slots.row_of(2)
    step([b])  # a is not scheduled this step: it keeps its row and its history
    assert core.engram_slots.row_of(1) == row_a
    mod.ModelingV2Core.release_request_state(
        core, 1
    )  # the engine released a: its row goes to the next request
    step([b, c])
    torch.cuda.synchronize()
    assert core.engram_slots.row_of(1) is None and core.engram_slots.row_of(3) == row_a
    assert core.engram_slots.row_of(2) == row_b


def test_a_released_request_rescheduled_in_the_same_rows_is_reseeded():
    """`release_request_state` frees the request's row and resets `_prev_ids`: the same ids in the same order with no
    `on_step` in between (a recompute pause readmitted next) are placed again, the request on a fresh row with its
    look-back seeded, instead of keeping the freed row's stale ring."""
    core = _core()
    a = _Req(1, [(5 + k) % 64 for k in range(10)], False)

    def step():
        mod.ModelingV2Core.on_step(core, NS(context_requests=[], generation_requests=[a]), None)

    step()
    torch.cuda.synchronize()
    a.ranges.clear()
    mod.ModelingV2Core.release_request_state(core, 1)
    assert core._prev_ids is None
    step()
    torch.cuda.synchronize()
    assert a.ranges == [
        (len(a.tokens) - 3, len(a.tokens) - 1)
    ]  # seeded again: the look-back of a new request
    assert int(core.req_slot[0]) == core.engram_slots.row_of(1)
