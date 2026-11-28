# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU tests for the DFlash2 step-bookkeeping entries (k31_dflash2_accept_commit, k31_dflash2_accept_record,
k31_dflash2_draft_tail): a sample of each entry's cells bitwise against its torch reference, in-place state
included."""

import types
import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec import dflash2_verify
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.spec.dflash2_step_bookkeeping import (  # noqa: E501
    FORCE_COUNTER_STRIDE,
    FORCE_SLOT_STRIDE,
    k31_dflash2_accept_commit,
    k31_dflash2_accept_record,
    k31_dflash2_draft_tail,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import GREEDY_TEMPERATURE_THRESHOLD as THR
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

__extra_import_path__ = ["../../../..", ".", "../.."]
from _k31_utils import on_sm, sample_cells
from _validating import certifying  # noqa: F401 — needs the path declared above

pytestmark = on_sm(103, reason="the K3.1 DFlash2 bookkeeping entries are certified on sm_103")

V = 163840


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


class _ForceWorker:
    """The engine's forced-acceptance methods and state (what the K3.1 workers inherit)."""

    _apply_force_accepted_tokens = SpecWorkerBase._apply_force_accepted_tokens
    _ensure_force_accept_rng_state = SpecWorkerBase._ensure_force_accept_rng_state

    def __init__(self, forced: float):
        self.force_num_accepted_tokens = forced
        self._force_accept_rng_pool = None
        self._force_accept_rng_counter = None


def _temperatures(kind, n, g):
    if kind == "hot":
        return 0.3 + torch.rand(n, generator=g, device="cuda")
    if kind == "edges":
        t = torch.tensor(THR, dtype=torch.float32)
        values = [
            t,
            torch.nextafter(t, torch.tensor(1.0)),
            torch.tensor(0.0),
            torch.tensor(float("nan")),
        ]
        return torch.stack(values).repeat(n)[:n].cuda()
    temps = torch.rand(n, generator=g, device="cuda")
    temps[::3] = 0.0
    return temps


def _commit_args(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n, K = spec["n"], spec["K"]
    tau = torch.randint(0, K + 1, (n,), generator=g, device="cuda")
    bonus = torch.randint(0, V, (n,), generator=g, device="cuda")
    gtau = torch.randint(0, K + 1, (n,), generator=g, device="cuda")
    gbonus = torch.randint(0, V, (n,), generator=g, device="cuda")
    temps = _temperatures(spec["temps"], n, g)
    drafts = torch.randint(0, V, (n, K), generator=g, device="cuda").to(spec["draft_dtype"])
    worker = _ForceWorker(spec["forced"])
    meta = types.SimpleNamespace(is_cuda_graph=False)
    mode, base, mx, frac, counter, pool = dflash2_verify._force_args(
        worker, K, meta, torch.device("cuda")
    )
    assert mode == spec["force"]
    if counter is None:
        counter = torch.zeros(1, dtype=torch.int64, device="cuda")
        pool = torch.zeros(1, device="cuda")
    else:
        counter.fill_(int(torch.randint(0, 1 << 40, (1,), generator=g, device="cuda")))
    return [
        tau,
        bonus,
        gtau,
        gbonus,
        temps,
        THR,
        drafts,
        mode,
        base,
        mx,
        frac,
        counter,
        pool,
        FORCE_COUNTER_STRIDE,
        FORCE_SLOT_STRIDE,
    ], worker


ACCEPT_COMMIT_CELLS = sample_cells(k31_dflash2_accept_commit.CELLS, size_key="n")[:2]


@pytest.mark.parametrize(
    "cell",
    ACCEPT_COMMIT_CELLS,
    ids=[c.why[:48] for c in ACCEPT_COMMIT_CELLS],
)
def test_accept_commit_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    args, _ = _commit_args(cell.spec, _seed(cell))
    ref_args = list(args)
    ref_args[11] = args[11].clone()
    certifying(k31_dflash2_accept_commit)
    acc, num = k31_dflash2_accept_commit(*args)
    acc_ref, num_ref = k31_dflash2_accept_commit.reference(*ref_args)
    assert torch.equal(acc, acc_ref) and torch.equal(num, num_ref)
    assert torch.equal(args[11], ref_args[11])  # the counter bump


def _record_args(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    n, slots = spec["n"], spec["slots"]
    num = torch.randint(1, 6, (n,), generator=g, device="cuda", dtype=torch.int32)
    if spec["dummies"] == "zero":
        num[::2] = 0
    state = torch.randperm(slots, generator=g, device="cuda")[:n].to(spec["state_dtype"])
    dummy = torch.zeros(n, dtype=torch.bool, device="cuda")
    if spec["dummies"] == "some":
        dummy[1::7] = True
    elif spec["dummies"] == "all":
        dummy[:] = True
    pitch = spec.get("prev_stride", 1)
    # The V2 manager's replay counts: one int32 per slot of a pool at the slot pitch.
    prev = torch.randint(0, 6, (slots, pitch), generator=g, device="cuda", dtype=torch.int32)[:, 0]
    kv = torch.randint(0, 300000, (n,), generator=g, device="cuda", dtype=torch.int32)
    return [num, state, dummy, prev, kv]


ACCEPT_RECORD_CELLS = sample_cells(k31_dflash2_accept_record.CELLS, size_key="n")[:2]


@pytest.mark.parametrize(
    "cell",
    ACCEPT_RECORD_CELLS,
    ids=[c.why[:48] for c in ACCEPT_RECORD_CELLS],
)
def test_accept_record_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    args = _record_args(cell.spec, _seed(cell))
    ref_args = [a.clone() for a in args]
    certifying(k31_dflash2_accept_record)
    rewind = k31_dflash2_accept_record(*args)
    rewind_ref = k31_dflash2_accept_record.reference(*ref_args)
    assert torch.equal(rewind, rewind_ref)
    assert torch.equal(args[3], ref_args[3]) and torch.equal(args[4], ref_args[4])


def _tail_args(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    b, K = spec["b"], spec["K"]
    rows = b + 3
    acc = torch.randint(0, V, (rows, K + 1), generator=g, device="cuda", dtype=torch.int32)
    batch = torch.randperm(rows, generator=g, device="cuda")[:b].to(spec["batch_dtype"])
    num = torch.randint(1, K + 2, (b,), generator=g, device="cuda", dtype=torch.int32)
    kv = torch.randint(0, 300000, (b,), generator=g, device="cuda", dtype=torch.int32)
    rewind = 1 - num
    if spec["counts"] == "edge":
        num[0] = 0
        kv[1] = 0
        rewind[1] = 3
    drafts = torch.randint(0, V, (b, K), generator=g, device="cuda", dtype=torch.int32)
    return [kv, rewind, acc, batch, num, drafts, spec["rewind"]]


DRAFT_TAIL_CELLS = sample_cells(k31_dflash2_draft_tail.CELLS, size_key="b")[:2]


@pytest.mark.parametrize("cell", DRAFT_TAIL_CELLS, ids=[c.why[:48] for c in DRAFT_TAIL_CELLS])
def test_draft_tail_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    args = _tail_args(cell.spec, _seed(cell))
    ref_args = [a.clone() if isinstance(a, torch.Tensor) else a for a in args]
    certifying(k31_dflash2_draft_tail)
    out = k31_dflash2_draft_tail(*args)
    out_ref = k31_dflash2_draft_tail.reference(*ref_args)
    assert out.dtype == out_ref.dtype and torch.equal(out, out_ref)
    assert torch.equal(args[0], ref_args[0])
