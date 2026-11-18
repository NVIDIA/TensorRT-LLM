# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the forward-tail spec fold (``m3_fold_tail`` and ``M3SpecStepFold`` in the forward) of the
MiniMax-M3.1-Flash sm_107 tp4 target (the family kernels) and tp8 target (its own kernels).

The tail must write exactly what the sampler's store (slot_scatter) and then the engine's next-step gathers
(step_gather, rows = the stored rows in order, first previous row 0) write, for the rows the step armed, and nothing
when the step was not armed.
"""

import importlib
import types

import pytest
import torch

from tensorrt_llm._torch.modeling_v2._native import NoTarget

if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
    pytest.skip("the sm_107 targets run on SM107 only", allow_module_level=True)

_PACKAGE = "tensorrt_llm._torch.modeling_v2.models.minimax_m3"
TP4 = "minimax_m3_1_flash_nvfp4__sm_107__tp4"
TP8 = "minimax_m3_1_flash_nvfp4__sm_107__tp8"
# Target -> the package of the fold-tail kernel it runs, and the M3SpecStepFold arguments it builds. Both targets run
# the family spec-tail kernels.
_KERNELS = {TP4: f"{_PACKAGE}.kernels", TP8: f"{_PACKAGE}.{TP8}.kernels"}
_FOLD_KWARGS = {TP4: {"in_forward": True}, TP8: {}}


def _load(target: str):
    try:
        return types.SimpleNamespace(
            modeling=importlib.import_module(f"{_PACKAGE}.{target}.modeling"),
            m3_fold_tail=importlib.import_module(f"{_KERNELS[target]}.m3_fold_tail"),
            m3_spec_tail=importlib.import_module(f"{_PACKAGE}.kernels.m3_spec_tail"),
        )
    except NoTarget:
        return None


TARGETS = {target: _load(target) for target in _KERNELS}


def _targets(*names):
    return [
        pytest.param(t, id=t.rsplit("__", 1)[1])
        if TARGETS[t] is not None
        else pytest.param(t, marks=pytest.mark.skip(reason=f"{t} is not built"))
        for t in names
    ]


VOCAB = 200064
NUM_SLOTS = 9
MAX_ROWS = 8
STALE = -4


def _store(widths, g):
    wn, wx, wd = widths
    i32 = dict(dtype=torch.int32, device="cuda")
    return types.SimpleNamespace(
        new_tokens=torch.randint(
            0, VOCAB, (wn, NUM_SLOTS, 1), generator=g, device="cuda", dtype=torch.int32
        ),
        next_new_tokens=torch.randint(
            0, VOCAB, (wx, NUM_SLOTS, 1), generator=g, device="cuda", dtype=torch.int32
        ),
        next_draft_tokens=torch.randint(
            0, VOCAB, (NUM_SLOTS, wd), generator=g, device="cuda", dtype=torch.int32
        ),
        new_tokens_lens=torch.randint(1, 5, (NUM_SLOTS,), generator=g, **i32),
    )


def _clone(ns):
    return types.SimpleNamespace(**{k: v.clone() for k, v in vars(ns).items()})


def _outputs(rows, t, g):
    d = t - 1
    return {
        "new_tokens": torch.randint(
            0, VOCAB, (rows, t), generator=g, device="cuda", dtype=torch.int32
        ),
        "next_new_tokens": torch.randint(
            0, VOCAB, (rows, t), generator=g, device="cuda", dtype=torch.int32
        ),
        "next_draft_tokens": torch.randint(
            0, VOCAB, (rows, d), generator=g, device="cuda", dtype=torch.int32
        ),
        "new_tokens_lens": torch.randint(
            1, t + 1, (rows,), generator=g, device="cuda", dtype=torch.int32
        ),
    }


def _engine(t):
    i32 = dict(dtype=torch.int32, device="cuda")
    return (
        torch.full((MAX_ROWS * t,), STALE, **i32),
        torch.full((MAX_ROWS * (t - 1),), STALE, **i32),
        torch.full((MAX_ROWS * t,), STALE, **i32),
        torch.full((MAX_ROWS,), STALE, **i32),
    )


def _reference(k, store, engine, slots, out, t):
    """slot_scatter into the store, then step_gather of the next step's inputs from it."""
    assert k.m3_spec_tail.slot_scatter(
        slots,
        out["new_tokens"],
        store.new_tokens.squeeze(-1).T,
        out["next_new_tokens"],
        store.next_new_tokens.squeeze(-1).T,
        out["new_tokens_lens"],
        store.new_tokens_lens,
        out["next_draft_tokens"],
        store.next_draft_tokens,
    )
    p = slots.numel()
    ids, draft, pos_off, kv_off = engine
    slots32 = slots.to(torch.int32)
    assert k.m3_spec_tail.step_gather(
        store.next_new_tokens,
        store.next_draft_tokens,
        store.new_tokens_lens,
        slots32,
        slots32.repeat_interleave(t),
        p,
        t,
        t - 1,
        ids,
        0,
        draft,
        0,
        pos_off,
        kv_off,
        0,
    )


def _tail(k, store, engine, ctrl, slots_buf, out, t, rows, cpp=False):
    ids, draft, pos_off, kv_off = engine
    assert k.m3_fold_tail.fold_tail(
        ctrl,
        slots_buf,
        rows,
        out["new_tokens"],
        store.new_tokens.squeeze(-1).T,
        out["next_new_tokens"],
        store.next_new_tokens.squeeze(-1).T,
        out["next_draft_tokens"],
        store.next_draft_tokens,
        out["new_tokens_lens"],
        store.new_tokens_lens,
        ids,
        draft,
        pos_off,
        kv_off,
        t,
        cpp=cpp,
    )


@pytest.mark.parametrize("cpp", [False, True], ids=["triton", "cpp"])
@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("widths", [(4, 4, 3), (8, 8, 7)], ids=["store=step", "store wider"])
@pytest.mark.parametrize("target", _targets(TP4, TP8))
def test_tail_matches_scatter_then_gather(target, rows, widths, cpp) -> None:
    """Store bit for bit; engine rows [0, P*T) / [0, P) bit for bit (the forward reads only those rows)."""
    t = 4
    g = torch.Generator(device="cuda").manual_seed(1000 * rows + widths[0])
    store = _store(widths, g)
    out = _outputs(rows, t, g)
    slots = torch.randperm(NUM_SLOTS, generator=g, device="cuda")[:rows]
    ref_store, tail_store = _clone(store), _clone(store)
    ref_engine, tail_engine = _engine(t), _engine(t)
    k = TARGETS[target]
    _reference(k, ref_store, ref_engine, slots, out, t)
    ctrl = torch.tensor([rows], dtype=torch.int32, device="cuda")
    slots_buf = torch.zeros(MAX_ROWS, dtype=torch.long, device="cuda")
    slots_buf[:rows] = slots
    _tail(k, tail_store, tail_engine, ctrl, slots_buf, out, t, rows, cpp)
    torch.cuda.synchronize()
    for name in vars(store):
        assert torch.equal(getattr(ref_store, name), getattr(tail_store, name)), name
    spans = (rows * t, rows * (t - 1), rows * t, rows)
    for k, (ref, got, n) in enumerate(zip(ref_engine, tail_engine, spans)):
        assert torch.equal(ref[:n], got[:n]), k
        assert torch.all(got[n:] == STALE), k


@pytest.mark.parametrize("cpp", [False, True], ids=["triton", "cpp"])
@pytest.mark.parametrize("target", _targets(TP4, TP8))
def test_unarmed_tail_writes_nothing(target, cpp) -> None:
    """ctrl[0] = 0 (the step was not armed): the store and the engine buffers are untouched."""
    t, rows = 4, 2
    g = torch.Generator(device="cuda").manual_seed(7)
    store = _store((4, 4, 3), g)
    out = _outputs(rows, t, g)
    before, engine = _clone(store), _engine(t)
    ctrl = torch.zeros(1, dtype=torch.int32, device="cuda")
    slots_buf = torch.arange(MAX_ROWS, dtype=torch.long, device="cuda")
    _tail(TARGETS[target], store, engine, ctrl, slots_buf, out, t, rows, cpp)
    torch.cuda.synchronize()
    for name in vars(store):
        assert torch.equal(getattr(before, name), getattr(store, name)), name
    for buf in engine:
        assert torch.all(buf == STALE)


@pytest.mark.parametrize("cpp", [False, True], ids=["triton", "cpp"])
@pytest.mark.parametrize("target", _targets(TP8))
def test_strided_slots_are_declined_without_writes(target, cpp) -> None:
    """The kernels read ``slots[i]`` at unit stride: a strided view is declined and nothing is written."""
    t, rows = 4, 2
    g = torch.Generator(device="cuda").manual_seed(17)
    store = _store((4, 4, 3), g)
    out = _outputs(rows, t, g)
    before, engine = _clone(store), _engine(t)
    ctrl = torch.tensor([rows], dtype=torch.int32, device="cuda")
    slots = torch.arange(2 * MAX_ROWS, dtype=torch.long, device="cuda")[::2]
    ids, draft, pos_off, kv_off = engine
    assert not TARGETS[target].m3_fold_tail.fold_tail(
        ctrl,
        slots,
        rows,
        out["new_tokens"],
        store.new_tokens.squeeze(-1).T,
        out["next_new_tokens"],
        store.next_new_tokens.squeeze(-1).T,
        out["next_draft_tokens"],
        store.next_draft_tokens,
        out["new_tokens_lens"],
        store.new_tokens_lens,
        ids,
        draft,
        pos_off,
        kv_off,
        t,
        cpp=cpp,
    )
    torch.cuda.synchronize()
    for name in vars(store):
        assert torch.equal(getattr(before, name), getattr(store, name)), name
    for buf in engine:
        assert torch.all(buf == STALE)


def _fold(target):
    fold = TARGETS[target].modeling.M3SpecStepFold(**_FOLD_KWARGS[target])
    t = 4
    engine = _engine(t)
    fold.bind_engine_buffers(*engine, max_batch_size=MAX_ROWS)
    g = torch.Generator(device="cuda").manual_seed(11)
    fold.bind_sampler_store(_store((4, 4, 3), g))
    return fold, g, t


@pytest.mark.parametrize("target", _targets(TP4, TP8))
def test_fold_handshake_stages_only_confirmed_rows(target) -> None:
    """begin_step arms; the sampler confirms only the armed rows from the tail's outputs; the next step is staged only
    for exactly those rows in that order."""
    fold, g, t = _fold(target)
    assert fold.begin_step([3, 5], 2) is False  # nothing staged before the first forward
    out = _outputs(2, t, g)
    fold.store_in_forward(out)
    assert fold.sampler_rows_stored(out, 0, [3, 5]) is True
    assert fold.begin_step([3, 5], 2) is True
    out = _outputs(2, t, g)
    fold.store_in_forward(out)
    assert (
        fold.sampler_rows_stored(out, 0, [5, 3]) is False
    )  # other order: the sampler stores itself
    assert fold.begin_step([5, 3], 2) is False


@pytest.mark.parametrize("target", _targets(TP4))
def test_a_padded_step_after_an_armed_one_writes_nothing(target) -> None:
    """CUDA-graph padding rows carry no seq slot, so the engine passes rows=None for a padded step.
    Even right after an armed step, that step uploads ctrl 0 and its forward tail leaves the store and the engine
    buffers untouched (a dummy row never writes a live slot)."""
    fold, g, t = _fold(target)
    fold.begin_step([2, 4], 2)
    fold.store_in_forward(_outputs(2, t, g))
    torch.cuda.synchronize()
    assert int(fold._ctrl.item()) == 2

    fold.begin_step(None, 4)
    torch.cuda.synchronize()
    assert int(fold._ctrl.item()) == 0
    store_before = _clone(fold.store)
    engine_before = [buf.clone() for buf in fold.step_inputs]
    fold.store_in_forward(_outputs(4, t, g))
    torch.cuda.synchronize()
    for name in vars(store_before):
        assert torch.equal(getattr(store_before, name), getattr(fold.store, name)), name
    for before, buf in zip(engine_before, fold.step_inputs):
        assert torch.equal(before, buf)
