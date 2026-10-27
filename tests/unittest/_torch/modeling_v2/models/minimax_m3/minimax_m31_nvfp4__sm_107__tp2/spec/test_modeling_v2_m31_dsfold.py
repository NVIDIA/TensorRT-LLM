# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DSFOLD: under the standalone DSpark worker the TP2 target (as TP4 / TP8) folds the engine's previous-batch input
gathers into one ``step_gather`` launch (``M3DSparkStepFold``). Oracle: the decoder runner's own eager gathers
(engine/runners/decoder/inputs.py, the ``previous_batch_len > 0 and not inputs_staged`` branch) on the same inputs;
every destination buffer must match bit for bit, including the rows the gathers do not own."""

import importlib
import inspect
import os
import subprocess
import sys
import zlib

import pytest
import torch

assert torch.cuda.is_available(), "DSFOLD requires a CUDA device"

_TARGETS = ("minimax_m31_nvfp4__sm_107__tp2",)
_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3"

#: Engine buffers sized for the max batch (8 requests x up to 1 + 7 tokens) and the stale fill they start with.
MAX_BATCH = 8
MAX_TOKENS = MAX_BATCH * 8
STALE = -4
VOCAB = 200064


def _modeling(target):
    return importlib.import_module(f"{_PKG}.{target}.modeling")


def _engine_eager(
    new_tokens, next_draft, lens, slots, ppi, P, T, D, ids, ids0, draft, d0, pos_off, kv_off, b0
):
    """The decoder runner's gathers for P previous requests, transcribed from inputs.py (offsets zeroed first)."""
    pos_off.zero_()
    kv_off.zero_()
    s = slots[:P]
    ids[ids0 : ids0 + P * T].copy_(new_tokens.transpose(0, 1)[s, :T].flatten())
    if D > 0:
        draft[d0 : d0 + P * D].copy_(next_draft[s, :D].flatten())
    kv_len_offsets = lens - T
    pos_off[b0 * T : (b0 + P) * T].copy_(lens[ppi[: P * T]])
    kv_off[b0 : b0 + P].copy_(kv_len_offsets[s])


def _case(k, store_k, prev, new_before, dummies, seed):
    """K drafts per step (T = K + 1 tokens) read from a store sized for ``store_k`` drafts; ``prev`` previous requests
    in shuffled slots after ``new_before`` requests without a previous tensor, then ``dummies`` CUDA-graph padding
    rows. Slots include the padding dummy's slot (the store holds MAX_BATCH + 1 slots)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    i32 = dict(dtype=torch.int32, device="cuda")
    T, D, nslots = k + 1, k, MAX_BATCH + 1
    assert new_before + prev + dummies <= MAX_BATCH
    new_tokens = torch.randint(0, VOCAB, (store_k + 1, nslots, 1), generator=g, **i32)
    next_draft = torch.randint(0, VOCAB, (nslots, store_k), generator=g, **i32)
    lens = torch.randint(1, T + 1, (nslots,), generator=g, **i32)
    slots = torch.randperm(nslots, generator=g, device="cuda")[:prev].to(torch.int32)
    ppi = slots.repeat_interleave(T).contiguous()
    ids0, d0 = new_before * T, new_before * D
    bufs = dict(
        ids=torch.full((MAX_TOKENS,), STALE, **i32),
        draft=torch.full((MAX_TOKENS,), STALE, **i32),
        pos_off=torch.full((MAX_TOKENS,), STALE, **i32),
        kv_off=torch.full((MAX_BATCH,), STALE, **i32),
    )
    return new_tokens, next_draft, lens, slots, ppi, prev, T, D, ids0, d0, new_before, bufs


def _run(fold_or_none, case):
    new_tokens, next_draft, lens, slots, ppi, P, T, D, ids0, d0, b0, bufs = case
    b = {n: t.clone() for n, t in bufs.items()}
    if fold_or_none is None:
        _engine_eager(
            new_tokens,
            next_draft,
            lens,
            slots,
            ppi,
            P,
            T,
            D,
            b["ids"],
            ids0,
            b["draft"],
            d0,
            b["pos_off"],
            b["kv_off"],
            b0,
        )
    else:
        fold_or_none.bind_engine_buffers(b["ids"], b["draft"], b["pos_off"], b["kv_off"], MAX_BATCH)
        # The engine's call (inputs.py): first_previous_row = requests without a previous tensor before them.
        assert fold_or_none.gather(new_tokens, next_draft, lens, slots, ppi, P, T, D, ids0, d0, b0)
    torch.cuda.synchronize()
    return b


# (K, store K, previous requests, new requests before them, padding dummies after them)
CELLS = [
    (3, 3, 1, 0, 0),  # c1, K3: one steady request
    (3, 3, 1, 0, 1),  # c1 padded to the batch-2 graph
    (3, 3, 8, 0, 0),  # c8 steady, shuffled slots
    (3, 3, 5, 2, 1),  # mixed: 2 new rows before, 1 dummy after
    (3, 7, 6, 1, 1),  # K3 run from a store sized for K7 (dynamic draft width)
    (7, 7, 1, 0, 0),  # K7 c1
    (7, 7, 8, 0, 0),  # K7 c8
    (7, 7, 4, 3, 1),  # K7 mixed + dummy
]


@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize(
    "cell", CELLS, ids=[f"K{c[0]}_store{c[1]}_p{c[2]}_new{c[3]}_dummy{c[4]}" for c in CELLS]
)
def test_fold_matches_engine_gathers(target, cell):
    case = _case(*cell, seed=zlib.crc32(f"{target}{cell}".encode()) % 2**31)
    ref = _run(None, case)
    fold = _modeling(target).M3DSparkStepFold()
    got = _run(fold, case)
    for name in ("ids", "draft", "pos_off", "kv_off"):
        assert torch.equal(got[name], ref[name]), f"{target} {cell}: {name}"
    assert (fold.taken, fold.declined) == (1, 0)


@pytest.mark.parametrize("target", _TARGETS)
def test_fold_declines_and_engine_copies_run(target):
    """A layout the kernel does not take (int64 slots) is declined with nothing written: the engine then gathers."""
    case = list(_case(3, 3, 4, 0, 0, seed=7))
    case[3] = case[3].long()
    new_tokens, next_draft, lens, slots, ppi, P, T, D, ids0, d0, b0, bufs = case
    fold = _modeling(target).M3DSparkStepFold()
    b = {n: t.clone() for n, t in bufs.items()}
    fold.bind_engine_buffers(b["ids"], b["draft"], b["pos_off"], b["kv_off"], MAX_BATCH)
    assert not fold.gather(new_tokens, next_draft, lens, slots, ppi, P, T, D, ids0, d0, b0)
    assert all(torch.equal(b[n], bufs[n]) for n in bufs)
    assert (fold.taken, fold.declined) == (0, 1)


@pytest.mark.parametrize("target", _TARGETS)
def test_scatter_stays_in_tree(target):
    """The sampler's slot stores are never folded under DSpark (the store path stays in-tree)."""
    fold = _modeling(target).M3DSparkStepFold()
    fold.bind_sampler_store(object())
    assert fold.scatter(None, None, None, None, None) is False
    assert fold.begin_step([0], 1) is False
    assert fold.sampler_rows_stored({}, 0, [0]) is False


@pytest.mark.parametrize("target", _TARGETS)
def test_kill_switch(target):
    """TRTLLM_M31_DISABLE_DSFOLD=1 (read once at import): DSFOLD is off, so a DSpark run keeps spec_fold None and the
    engine's own gathers; the Eagle3 path still builds M3SpecStepFold."""
    probe = f"import {_PKG}.{target}.modeling as m; print(m.DSFOLD)"
    for value, want in (("1", "False"), ("0", "True")):
        env = dict(os.environ, TRTLLM_M31_DISABLE_DSFOLD=value)
        out = subprocess.run(
            [sys.executable, "-c", probe], env=env, capture_output=True, text=True, check=True
        )
        assert out.stdout.strip().splitlines()[-1] == want
    source = inspect.getsource(_modeling(target))
    # The sm_107 targets wire the fold differently (TP8: self._spec_fold; TP4: behind STEP_SPEC_FOLD), so check the
    # DSpark branch rather than one exact text: DSFOLD gates M3DSparkStepFold, the Eagle3 path keeps M3SpecStepFold.
    assert (
        "elif DSFOLD" in source and "M3DSparkStepFold()" in source and "M3SpecStepFold(" in source
    )
