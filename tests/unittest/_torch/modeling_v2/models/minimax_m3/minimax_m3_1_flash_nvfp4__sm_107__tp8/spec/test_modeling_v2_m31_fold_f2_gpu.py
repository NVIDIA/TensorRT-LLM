# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FOLD_F2 on a GPU: the fold tail launch that writes the step's next_new_tokens itself writes exactly what N3 A2's
launch (spec_next_new_tokens) and then the stock C++ tail write: next_new_tokens for every row, the store and the next
step's inputs for the armed rows; it declines, writing nothing, outside its operands."""

import types

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp8.kernels import (  # noqa: E501
    m3_fold_tail,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.dspark import strict_accept as sa

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP8 target runs on SM107 only",
)

VOCAB = 200064
NUM_SLOTS = 9
MAX_ROWS = 8
STALE = -4
I32 = dict(dtype=torch.int32, device="cuda")


def _store(t, g):
    return types.SimpleNamespace(
        new_tokens=torch.randint(0, VOCAB, (t, NUM_SLOTS, 1), generator=g, **I32),
        next_new_tokens=torch.randint(0, VOCAB, (t, NUM_SLOTS, 1), generator=g, **I32),
        next_draft_tokens=torch.randint(0, VOCAB, (NUM_SLOTS, t - 1), generator=g, **I32),
        new_tokens_lens=torch.randint(1, 5, (NUM_SLOTS,), generator=g, **I32),
    )


def _clone(ns):
    return types.SimpleNamespace(**{k: v.clone() for k, v in vars(ns).items()})


def _engine(t):
    return (
        torch.full((MAX_ROWS * t,), STALE, **I32),
        torch.full((MAX_ROWS * (t - 1),), STALE, **I32),
        torch.full((MAX_ROWS * t,), STALE, **I32),
        torch.full((MAX_ROWS,), STALE, **I32),
    )


def _step(rows, t, g, served):
    """The step's outputs and N3 A2's operands: served = the DSpark forward's (accepted IS new_tokens, its counts ARE
    new_tokens_lens, batch rows 0..rows-1); else a wider accepted buffer read through a permuted batch_idx."""
    d = t - 1
    acc_rows = rows if served else rows + 3
    accepted = torch.randint(0, VOCAB, (acc_rows, t), generator=g, **I32)
    nacc = torch.randint(1, t + 1, (rows,), generator=g, **I32)
    if served:
        batch_idx = torch.arange(rows, **I32)
    else:
        batch_idx = torch.randperm(acc_rows, generator=g, device="cuda")[:rows].to(torch.int32)
    draft = torch.randint(0, VOCAB, (rows, d), generator=g, **I32)
    out = {
        "new_tokens": accepted
        if served
        else torch.randint(0, VOCAB, (rows, t), generator=g, **I32),
        "new_tokens_lens": nacc if served else torch.randint(1, t + 1, (rows,), generator=g, **I32),
        "next_draft_tokens": draft,
    }
    return out, (accepted, nacc, batch_idx, draft)


def _tail(store, engine, ctrl, slots_buf, out, next_new, t, rows, next_from=None, **kw):
    return m3_fold_tail.fold_tail(
        ctrl,
        slots_buf,
        rows,
        out["new_tokens"],
        store.new_tokens.squeeze(-1).T,
        next_new,
        store.next_new_tokens.squeeze(-1).T,
        out["next_draft_tokens"],
        store.next_draft_tokens,
        out["new_tokens_lens"],
        store.new_tokens_lens,
        *engine,
        t,
        cpp=kw.get("cpp", True),
        one_warp=kw.get("one_warp", False),
        next_from=next_from,
    )


def _check(ref, got):
    for name in vars(ref[0]):
        assert torch.equal(getattr(ref[0], name), getattr(got[0], name)), name
    for k, (x, y) in enumerate(zip(ref[1], got[1])):
        assert torch.equal(x, y), k


@pytest.mark.parametrize("served", [True, False], ids=["served", "general"])
@pytest.mark.parametrize("t", [8])
@pytest.mark.parametrize("rows", [1, 5, 8])
def test_tail_next_is_n3a2_then_the_tail(rows, t, served):
    for armed in (rows, 0, 1):
        g = torch.Generator(device="cuda").manual_seed(1000 * rows + 10 * t + armed)
        store = _store(t, g)
        out, operands = _step(rows, t, g, served)
        slots_buf = torch.randperm(NUM_SLOTS, generator=g, device="cuda")[:MAX_ROWS].to(torch.long)
        ctrl = torch.tensor([armed], **I32)
        ref_next = sa.spec_next_new_tokens(*operands)
        ref = (_clone(store), _engine(t))
        assert _tail(*ref, ctrl, slots_buf, out, ref_next, t, rows)
        got_next = torch.full((rows, t), STALE, **I32)
        got = (_clone(store), _engine(t))
        assert _tail(*got, ctrl, slots_buf, out, got_next, t, rows, next_from=operands)
        torch.cuda.synchronize()
        assert ref_next.dtype == torch.int32 and torch.equal(got_next, ref_next), armed
        _check(ref, got)


def test_tail_next_declines_without_writing():
    rows, t = 3, 4
    g = torch.Generator(device="cuda").manual_seed(9)
    store = _store(t, g)
    out, (accepted, nacc, batch_idx, draft) = _step(rows, t, g, served=True)
    slots_buf = torch.arange(MAX_ROWS, dtype=torch.long, device="cuda")
    ctrl = torch.tensor([rows], **I32)
    cases = {
        "int64 batch_idx": ((accepted, nacc, batch_idx.long(), draft), {}),
        "fewer grid rows": ((accepted, nacc, batch_idx, draft), {"rows": rows - 1}),
        "strided drafts": (
            (accepted, nacc, batch_idx, torch.zeros(rows, 2 * (t - 1), **I32)[:, ::2]),
            {},
        ),
        "one warp": ((accepted, nacc, batch_idx, draft), {"one_warp": True}),
        "triton": ((accepted, nacc, batch_idx, draft), {"cpp": False}),
    }
    for name, (operands, kw) in cases.items():
        got_next = torch.full((rows, t), STALE, **I32)
        got = (_clone(store), _engine(t))
        n = kw.pop("rows", rows)
        assert not _tail(*got, ctrl, slots_buf, out, got_next, t, n, next_from=operands, **kw), name
        torch.cuda.synchronize()
        assert torch.all(got_next == STALE), name
        _check((store, _engine(t)), got)
