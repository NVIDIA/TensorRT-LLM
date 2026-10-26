# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MCAG on the sm_107 TP2 target (world 2; VR port of the sm_103 test).

The rejection sampler's draft-logits gather (SpecWorkerBase.maybe_gather_sharded_draft_logits) takes a target-installed
one-launch gather when it returns a tensor and the stock all-gather otherwise; the target's op declines by dtype /
columns / uncompiled rows. Default OFF, TRTLLM_M31_ENABLE_MCAG=1 turns it on and TRTLLM_M31_DISABLE_MCAG=1 still
wins. CPU only: the gathers are stand-ins."""

import importlib
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
    dspark,
    m31_switches,
)
from tensorrt_llm._torch.distributed import ops as dist_ops
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

STOCK = "stock-gathered"


def _worker(op=None):
    return SimpleNamespace(
        mapping="tp2",
        _draft_logits_are_sharded=lambda logits, md: True,
        _draft_logits_gather_op=op,
    )


def test_the_hook_is_a_class_default():
    """Every stock spec worker reads the class attribute (None), no per-step attribute miss."""
    assert SpecWorkerBase._draft_logits_gather_op is None
    assert "_draft_logits_gather_op" in SpecWorkerBase.__dict__


@pytest.fixture
def stock(monkeypatch):
    calls = []
    monkeypatch.setattr(dist_ops, "allgather", lambda x, mapping, dim: calls.append(dim) or STOCK)
    return calls


def _gather(worker, logits, greedy=False):
    md = SimpleNamespace(is_all_greedy_sample=greedy)
    return SpecWorkerBase.maybe_gather_sharded_draft_logits(worker, logits, md)


def test_the_hook_is_inert_by_default(stock):
    assert _gather(_worker(), torch.zeros(1, 7, 8)) == STOCK
    assert stock == [-1]


def test_the_op_runs_and_a_declined_call_falls_back(stock):
    x = torch.zeros(1, 7, 8)
    assert _gather(_worker(lambda logits: "op-gathered"), x) == "op-gathered"
    assert stock == []
    assert _gather(_worker(lambda logits: None), x) == STOCK
    assert stock == [-1]


def test_greedy_batches_never_gather(stock):
    x = torch.zeros(1, 7, 8)
    assert _gather(_worker(lambda logits: "op"), x, greedy=True) is x
    assert stock == []


def test_the_setter():
    w = SimpleNamespace()
    assert SpecWorkerBase.enable_draft_logits_gather_op(w, None) is False
    op = object()
    assert (
        SpecWorkerBase.enable_draft_logits_gather_op(w, op) is True
        and w._draft_logits_gather_op is op
    )


@pytest.fixture
def mcag_target(monkeypatch):
    calls = []
    compiled = {(7, 100032, 2)}
    monkeypatch.setattr(
        dspark._allgather_cast,
        "is_compiled",
        lambda rows, cols, world, grid=0, device=None, fp32=False: (
            fp32 and (rows, cols, world) in compiled
        ),
    )
    monkeypatch.setattr(
        dspark._allgather_cast,
        "gather_last_dim_f32",
        lambda logits, **ws: calls.append((tuple(logits.shape), ws["world"])) or "mc-gathered",
    )
    target = SimpleNamespace(
        _mcag_cols=100032,
        _a1={"world": 2, "ws_uc": None, "ws_mc": None, "ws_flags": None, "rank": 0},
    )
    return target, calls


def test_the_target_op_takes_compiled_fp32_rows(mcag_target):
    target, calls = mcag_target
    assert dspark.M3DSparkWorker._mcag_gather(target, torch.zeros(1, 7, 100032)) == "mc-gathered"
    assert calls == [((1, 7, 100032), 2)]


@pytest.mark.parametrize(
    "logits",
    [
        torch.zeros(1, 7, 100032, dtype=torch.bfloat16),  # not fp32
        torch.zeros(1, 7, 100000),  # other columns
        torch.zeros(2, 7, 100032),  # rows not compiled
    ],
)
def test_the_target_op_declines(mcag_target, logits):
    target, calls = mcag_target
    assert dspark.M3DSparkWorker._mcag_gather(target, logits) is None
    assert calls == []


@pytest.mark.parametrize(
    "env, expected",
    [
        ({}, False),
        ({"TRTLLM_M31_ENABLE_MCAG": "1"}, True),
        ({"TRTLLM_M31_ENABLE_MCAG": "1", "TRTLLM_M31_DISABLE_MCAG": "1"}, False),
    ],
)
def test_default_off_opt_in_and_kill_switch(monkeypatch: pytest.MonkeyPatch, env, expected) -> None:
    monkeypatch.delenv("TRTLLM_M31_DISABLE_MCAG", raising=False)
    monkeypatch.delenv("TRTLLM_M31_ENABLE_MCAG", raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    importlib.reload(m31_switches)
    try:
        assert importlib.reload(dspark).MCAG is expected
    finally:
        monkeypatch.delenv("TRTLLM_M31_DISABLE_MCAG", raising=False)
        monkeypatch.delenv("TRTLLM_M31_ENABLE_MCAG", raising=False)
        importlib.reload(m31_switches)
        importlib.reload(dspark)
    assert dspark.MCAG is False


def test_c6_mcag_is_unreachable_without_sharded_draft_logits():
    """C6: only the S4-logits path hands the sampler vocab-sharded draft logits; without S4 set up, or with S4-logits
    off, the gather MCAG replaces is never called, and MCAG says why instead of compiling."""
    off = dspark.mcag_unreachable(SimpleNamespace(_s4=None, _s4_logits_on=True))
    assert off is not None and "S4 is not set up" in off
    off = dspark.mcag_unreachable(SimpleNamespace(_s4=object(), _s4_logits_on=False))
    assert off is not None and "S4-logits is off" in off
    assert dspark.mcag_unreachable(SimpleNamespace(_s4=object(), _s4_logits_on=True)) is None
