# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The backend lines of the native SwiGLU-OAI and MoE finalize ops on the tp2 target.

Each wrapper logs, once per (item, site, outcome) per process, "[M3 dispatch] rank N: NATIVE_SWIGLU / NATIVE_FINALIZE
taken (<site>)" when the native op serves, or "... not taken (<site>): <reason>" when the caller's fallback does, so a
serving log shows which backend ran. CPU only: the op is a stand-in.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.kernels import (  # noqa: E501
    m3_report,
)

pytestmark = pytest.mark.cpu_only

CORE = SimpleNamespace(swiglu_limit=7.0, swiglu_alpha=1.702, swiglu_beta=1.0)
SERVED = object()


@pytest.fixture
def lines(monkeypatch):
    """Every line modeling logs, with rank 0, a stand-in native op and fresh once-per-process state."""
    out = []
    monkeypatch.setattr(m3_report, "_info", lambda message: None)
    monkeypatch.setattr(m3_report, "_rank", lambda: 0)
    monkeypatch.setattr(modeling, "mpi_rank", lambda: 0)
    monkeypatch.setattr(modeling, "logger", SimpleNamespace(info=out.append, warning=out.append))
    monkeypatch.setattr(modeling, "_native_op", lambda name: lambda *args: SERVED)
    m3_report.reset_work()
    modeling.reset_logged()
    yield out
    m3_report.reset_work()
    modeling.reset_logged()


def _dispatch(out):
    return [line for line in out if "NATIVE_" in line]


def _gu(rows=8, d=16):
    return torch.zeros(rows, 2 * d, dtype=torch.bfloat16)


def _finalize_args(rows=8, shared=False):
    fc2 = torch.zeros(rows * 2, 16, dtype=torch.bfloat16)
    scales = torch.zeros(rows, 2, dtype=torch.float32)
    perm = torch.zeros(rows, 2, dtype=torch.int32)
    return (
        rows,
        fc2,
        scales,
        perm,
        (torch.zeros(rows, 16, dtype=torch.bfloat16) if shared else None),
    )


@pytest.mark.parametrize("dense,site", [(False, "shared"), (True, "dense")])
def test_swiglu_taken_once_per_site(lines, dense, site):
    for _ in range(3):
        assert modeling._native_swiglu(CORE, _gu(), dense) is SERVED
    assert _dispatch(lines) == [f"[M3 dispatch] rank 0: NATIVE_SWIGLU taken ({site})"]


@pytest.mark.parametrize(
    "patch,gu,why",
    [
        ({"NATIVE_SWIGLU": False}, _gu(), "killed (TRTLLM_M31_DISABLE_NATIVE_SWIGLU=1)"),
        ({"_native_op": lambda name: None}, _gu(), "the op is not registered"),
        ({}, _gu().float(), "input is not a contiguous 2-D bf16 tensor"),
        ({}, _gu().t(), "input is not a contiguous 2-D bf16 tensor"),
    ],
)
def test_swiglu_not_taken_names_the_reason(lines, monkeypatch, patch, gu, why):
    for name, value in patch.items():
        monkeypatch.setattr(modeling, name, value)
    assert modeling._native_swiglu(CORE, gu, False) is None
    assert modeling._native_swiglu(CORE, gu, False) is None
    assert _dispatch(lines) == [f"[M3 dispatch] rank 0: NATIVE_SWIGLU not taken (shared): {why}"]


def test_swiglu_other_constants_are_not_taken(lines):
    core = SimpleNamespace(swiglu_limit=10.0, swiglu_alpha=1.702, swiglu_beta=1.0)
    assert modeling._native_swiglu(core, _gu(), True) is None
    assert _dispatch(lines) == [
        "[M3 dispatch] rank 0: NATIVE_SWIGLU not taken (dense): SwiGLU constants other than (7, 1.702, 1)"
    ]


@pytest.mark.parametrize("shared,site", [(False, "routed"), (True, "shared add")])
def test_finalize_taken_once_per_site(lines, shared, site):
    for _ in range(3):
        assert modeling._native_finalize(*_finalize_args(shared=shared)) is SERVED
    assert _dispatch(lines) == [f"[M3 dispatch] rank 0: NATIVE_FINALIZE taken ({site})"]


def test_finalize_kill_switch_and_missing_op(lines, monkeypatch):
    monkeypatch.setattr(modeling, "NATIVE_FINALIZE", False)
    assert modeling._native_finalize(*_finalize_args()) is None
    monkeypatch.setattr(modeling, "NATIVE_FINALIZE", True)
    monkeypatch.setattr(modeling, "_native_op", lambda name: None)
    assert modeling._native_finalize(*_finalize_args()) is None
    assert _dispatch(lines) == [
        "[M3 dispatch] rank 0: NATIVE_FINALIZE not taken (routed): killed (TRTLLM_M31_DISABLE_NATIVE_FINALIZE=1)",
        "[M3 dispatch] rank 0: NATIVE_FINALIZE not taken (routed): the op is not registered",
    ]


def test_finalize_layout_is_not_taken(lines):
    rows, fc2, scales, perm, shared = _finalize_args()
    assert modeling._native_finalize(rows, fc2.float(), scales, perm, shared) is None
    assert _dispatch(lines) == [
        "[M3 dispatch] rank 0: NATIVE_FINALIZE not taken (routed): operand layout or alignment"
    ]
