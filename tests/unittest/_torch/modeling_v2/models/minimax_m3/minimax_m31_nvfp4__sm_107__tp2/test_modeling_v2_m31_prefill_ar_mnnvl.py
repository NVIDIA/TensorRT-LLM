# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PREFILL-MNNVL: the sm_107 TP2 prefill / mixed-step boundaries' built-in all-reduce is MNNVL up to 8192 rows; larger
steps and every generation step keep the target's AUTO all-reduce. TRTLLM_M31_DISABLE_PREFILL_AR_MNNVL=1 restores
AUTO everywhere."""

import importlib
import inspect

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
    m31_switches,
    modeling,
)

AUTO, MNNVL = object(), object()


class _Core:
    def __init__(self, mnnvl) -> None:
        self.allreduce = AUTO
        self.allreduce_mnnvl = mnnvl


@pytest.fixture
def prefill_mnnvl_off(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("TRTLLM_M31_DISABLE_PREFILL_AR_MNNVL", "1")
    importlib.reload(m31_switches)
    try:
        yield importlib.reload(modeling)
    finally:
        monkeypatch.delenv("TRTLLM_M31_DISABLE_PREFILL_AR_MNNVL", raising=False)
        importlib.reload(m31_switches)
        importlib.reload(modeling)


def test_default_on_up_to_8192_rows() -> None:
    assert modeling.PREFILL_AR_MNNVL and modeling.PREFILL_AR_MNNVL_MAX_ROWS == 8192


@pytest.mark.parametrize("rows", [1, 33, 64, 128, 1024, 8192, 8193, 16384])
def test_rows_pick_the_all_reduce(rows: int) -> None:
    want = MNNVL if rows <= 8192 else AUTO
    assert modeling._prefill_allreduce(_Core(MNNVL), rows) is want
    assert modeling._prefill_allreduce(_Core(None), rows) is AUTO


def test_only_the_prefill_target_takes_it() -> None:
    """Every built-in boundary all-reduce of the prefill target goes through ``_prefill_allreduce``; the decode target
    (every generation step, the 33..64-row ones included) never does."""
    prefill = inspect.getsource(modeling.PrefillTarget)
    # The two CTXNORMQ boundaries and the three native fused ones each pass ``_prefill_allreduce``.
    assert prefill.count("_prefill_allreduce(core, ") == 5
    assert prefill.count("_ctxnormq_allreduce_norm(") == 2
    assert prefill.count("_native_allreduce_norm(") == 3
    assert "_prefill_allreduce" not in inspect.getsource(modeling.DecodeTarget)


def test_kill_switch(prefill_mnnvl_off) -> None:
    m = prefill_mnnvl_off
    assert not m.PREFILL_AR_MNNVL
    assert "AllReduceStrategy.MNNVL" in inspect.getsource(m.MiniMaxM3Tp2ModelingV2.__init__)
    assert m._prefill_allreduce(_Core(None), 128) is AUTO
