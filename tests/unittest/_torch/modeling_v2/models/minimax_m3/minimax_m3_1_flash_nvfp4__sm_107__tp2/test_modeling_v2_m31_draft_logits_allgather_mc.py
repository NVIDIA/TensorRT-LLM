# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DRAFT_LOGITS_ALLGATHER_MC on the sm_107 TP2 target (world 2).

The target's draft-logits gather op (``M3DSparkWorker._mcag_gather``) takes compiled fp32 rows of the full shard
width in one multicast launch and declines (None: the stock all-gather runs) by dtype / columns / uncompiled rows.
Off by default; ``TRTLLM_M31_ENABLE_DRAFT_LOGITS_ALLGATHER_MC=1`` opts in. Without vocab-sharded draft logits (no
Markov-chain set-up, or its logits path off) the gather it replaces is never called, and the prebuild says why. CPU
only: the gathers are stand-ins.
"""

import importlib
import os
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2._native import NoTarget

try:
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__tp2 import (  # noqa: E501
        dspark,
        m31_switches,
    )
except NoTarget as exc:
    pytest.skip(str(exc), allow_module_level=True)

pytestmark = pytest.mark.cpu_only

ENV = "TRTLLM_M31_ENABLE_DRAFT_LOGITS_ALLGATHER_MC"


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


def test_off_by_default():
    assert dspark.MCAG is m31_switches.m31_enabled("DRAFT_LOGITS_ALLGATHER_MC")
    if os.environ.get(ENV) is None:
        assert dspark.MCAG is False


@pytest.mark.parametrize("value, expected", [("1", True), ("on", True), ("0", False), ("", False)])
def test_the_opt_in(monkeypatch, value, expected) -> None:
    original = os.environ.get(ENV)
    monkeypatch.setenv(ENV, value)
    try:
        importlib.reload(m31_switches)
        assert importlib.reload(dspark).MCAG is expected
    finally:
        if original is None:
            monkeypatch.delenv(ENV, raising=False)
        else:
            monkeypatch.setenv(ENV, original)
        importlib.reload(m31_switches)
        importlib.reload(dspark)


def test_unreachable_without_sharded_draft_logits():
    """Only the Markov chain's logits path hands the sampler vocab-sharded draft logits; without the chain set up, or
    with its logits path off, the gather this op replaces is never called, and the prebuild says why instead of
    compiling."""
    off = dspark.mcag_unreachable(SimpleNamespace(_s4=None, _s4_logits_on=True))
    assert off is not None and "S4 is not set up" in off
    off = dspark.mcag_unreachable(SimpleNamespace(_s4=object(), _s4_logits_on=False))
    assert off is not None and "S4-logits is off" in off
    assert dspark.mcag_unreachable(SimpleNamespace(_s4=object(), _s4_logits_on=True)) is None
