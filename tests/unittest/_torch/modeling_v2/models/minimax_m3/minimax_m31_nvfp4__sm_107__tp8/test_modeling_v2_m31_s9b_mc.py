# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""S9B-MC: the sm_107 TP8 target runs S9b's fused drafter all-reduce + norm on its multicast all-reduce + norm (the
drafter-norm form, 8-token bucket) through the DFlash drafter's inert fused-AR-norm hook; other row counts, and every
drafter without the hook filled, keep the Linear's own fused all-reduce. Default off: TRTLLM_M31_ENABLE_S9B_MC=1 opts
in, TRTLLM_M31_DISABLE_S9B_MC=1 still wins."""

import importlib
import inspect
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp8 import (  # noqa: E501
    dspark,
    m31_switches,
)
from tensorrt_llm._torch.models import modeling_dflash
from tensorrt_llm._torch.speculative.dflash import DFlashWorker

STOCK, FROM_OP = ("stock-normed", "stock-res"), ("op-normed", "op-res")


class _Linear:
    def __init__(self) -> None:
        self.calls = 0

    def all_reduce(self, partial, all_reduce_params):
        self.calls += 1
        return STOCK


_NORM = SimpleNamespace(weight="w", variance_epsilon=1e-6)


def test_the_hook_is_inert_by_default() -> None:
    linear = _Linear()
    assert modeling_dflash._dflash_ar_residual_norm(linear, "x", "r", _NORM) == STOCK
    assert modeling_dflash._dflash_ar_residual_norm(linear, "x", "r", _NORM, None) == STOCK
    assert linear.calls == 2


def test_the_op_runs_and_a_declined_call_falls_back() -> None:
    seen = []

    def op(partial, residual, norm_weight, eps):
        seen.append((partial, residual, norm_weight, eps))
        return FROM_OP if partial == "take" else None

    linear = _Linear()
    assert modeling_dflash._dflash_ar_residual_norm(linear, "take", "r", _NORM, op) == FROM_OP
    assert linear.calls == 0
    assert modeling_dflash._dflash_ar_residual_norm(linear, "decline", "r", _NORM, op) == STOCK
    assert linear.calls == 1
    assert seen[0] == ("take", "r", "w", 1e-6)


def test_every_fused_call_site_passes_the_hook() -> None:
    src = inspect.getsource(modeling_dflash.DFlashForCausalLM)
    assert src.count("_dflash_ar_residual_norm(") == 3
    assert (
        src.count("self._dflash_fused_ar_norm_op,") + src.count("self._dflash_fused_ar_norm_op\n")
        >= 3
    )
    assert "self._dflash_fused_ar_norm_op = None" in src


@pytest.mark.parametrize("fused", [True, False])
def test_the_setter_needs_the_fused_ar_norm(fused: bool) -> None:
    model = SimpleNamespace(_dflash_fused_ar_norm=fused, _dflash_fused_ar_norm_op=None)
    op = object()
    assert DFlashWorker.enable_fused_ar_norm_op(None, model, op) is fused
    assert model._dflash_fused_ar_norm_op is (op if fused else None)
    assert DFlashWorker.enable_fused_ar_norm_op(None, SimpleNamespace(), op) is False


def test_default_off_with_the_8_token_bucket() -> None:
    assert dspark.S9B_MC is False and dspark.S9B_MC_MAX_ROWS == 8


@pytest.mark.parametrize("rows", [9, 16, 64])
def test_the_op_declines_past_8_rows_without_a_workspace(monkeypatch, rows: int) -> None:
    def no_workspace(*args, **kwargs):
        raise AssertionError("the op must decline before asking for a workspace")

    monkeypatch.setattr(dspark.m3_mc_allreduce, "get_workspace", no_workspace)
    op = dspark._s9b_mc_op(SimpleNamespace(tp_size=8))
    x = torch.zeros(rows, 16, dtype=torch.bfloat16)
    assert op(x, x, torch.ones(16, dtype=torch.bfloat16), 1e-6) is None


def test_the_op_calls_the_drafter_norm_form(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(dspark.m3_mc_allreduce, "get_workspace", lambda mapping, hidden: "ws")

    def mc(x, residual, w, eps, workspace, use_pdl=False, drafter_norm=False, **kwargs):
        calls.append((tuple(x.shape), workspace, drafter_norm, kwargs))
        return FROM_OP

    monkeypatch.setattr(dspark.m3_mc_allreduce, "m3_mc_allreduce_norm", mc)
    op = dspark._s9b_mc_op(SimpleNamespace(tp_size=8))
    x = torch.zeros(8, 16, dtype=torch.bfloat16)
    assert op(x, x, torch.ones(16, dtype=torch.bfloat16), 1e-6) == FROM_OP
    assert calls == [((8, 16), "ws", True, {})]
    monkeypatch.setattr(dspark.m3_mc_allreduce, "get_workspace", lambda mapping, hidden: None)
    assert op(x, x, torch.ones(16, dtype=torch.bfloat16), 1e-6) is None


@pytest.mark.parametrize(
    "enable, disable, on", [("1", "", True), ("1", "1", False), ("", "1", False)]
)
def test_switches(monkeypatch: pytest.MonkeyPatch, enable: str, disable: str, on: bool) -> None:
    monkeypatch.setenv("TRTLLM_M31_ENABLE_S9B_MC", enable)
    monkeypatch.setenv("TRTLLM_M31_DISABLE_S9B_MC", disable)
    importlib.reload(m31_switches)
    try:
        assert importlib.reload(dspark).S9B_MC is on
    finally:
        monkeypatch.delenv("TRTLLM_M31_ENABLE_S9B_MC", raising=False)
        monkeypatch.delenv("TRTLLM_M31_DISABLE_S9B_MC", raising=False)
        importlib.reload(m31_switches)
        importlib.reload(dspark)
    assert dspark.S9B_MC is False
