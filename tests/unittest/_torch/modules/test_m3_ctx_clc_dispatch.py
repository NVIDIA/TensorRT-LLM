# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU coverage of optional CLC host failures versus actual launch failures."""

from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.modules import m3_ctx_clc_gemm as clc

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def dispatch(monkeypatch):
    runner = Mock(spec_set=["check", "make_args", "run_args"])
    runner.check.return_value = None
    runner.make_args.return_value = ("prepared",)
    monkeypatch.setattr(clc, "_PRECOMPILE", False)
    monkeypatch.setattr(clc, "_failed", {})
    monkeypatch.setattr(clc, "ctx_clc_heads_per_tile", lambda m, n, k: 2)
    monkeypatch.setattr(clc, "_get_gemm", lambda n, hpt: runner)
    warning = Mock(spec_set=clc._skip)
    monkeypatch.setattr(clc, "_skip", warning)
    act = torch.empty((1024, 4096), device="meta", dtype=torch.float8_e4m3fn)
    weight = torch.empty((6144, 4096), device="meta", dtype=torch.float8_e4m3fn)
    scales = torch.empty(1, device="meta", dtype=torch.uint8)
    alpha = torch.empty(1, device="meta")
    return runner, warning, (act, scales, weight, scales, alpha, torch.bfloat16)


@pytest.mark.parametrize("stage", ["check", "make_args"])
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_host_preparation_failure_falls_back_once(dispatch, stage, error_type):
    runner, warning, inputs = dispatch
    getattr(runner, stage).side_effect = error_type("host preparation failure")
    assert clc.try_ctx_clc_mxfp8_gemm(*inputs) is None
    assert clc.try_ctx_clc_mxfp8_gemm(*inputs) is None
    assert runner.check.call_count == 1
    assert runner.make_args.call_count == (stage == "make_args")
    runner.run_args.assert_not_called()
    assert warning.call_count == 1
    assert (6144, 2) in clc._failed


@pytest.mark.parametrize("error_type", [RuntimeError, AssertionError])
def test_kernel_launch_errors_propagate(dispatch, error_type):
    runner, warning, inputs = dispatch
    runner.run_args.side_effect = error_type("kernel launch failure")
    with pytest.raises(error_type, match="kernel launch failure"):
        clc.try_ctx_clc_mxfp8_gemm(*inputs)
    assert not clc._failed
    warning.assert_not_called()


def test_prepared_arguments_reach_the_kernel(dispatch):
    runner, warning, inputs = dispatch
    out = clc.try_ctx_clc_mxfp8_gemm(*inputs)
    assert out.shape == (1024, 6144) and out.dtype == torch.bfloat16
    assert runner.make_args.call_args.args[-1] is out
    runner.run_args.assert_called_once_with(("prepared",))
    warning.assert_not_called()


def test_output_allocation_errors_propagate(dispatch, monkeypatch):
    runner, warning, inputs = dispatch
    allocation = Mock(side_effect=RuntimeError("output allocation failure"))
    monkeypatch.setattr(torch, "empty", allocation)
    with pytest.raises(RuntimeError, match="output allocation failure"):
        clc.try_ctx_clc_mxfp8_gemm(*inputs)
    assert not clc._failed
    runner.make_args.assert_not_called()
    runner.run_args.assert_not_called()
    warning.assert_not_called()


@pytest.mark.parametrize("stage", ["check", "make_args"])
@pytest.mark.parametrize("error_type", [torch.OutOfMemoryError, torch.AcceleratorError])
def test_accelerator_failures_during_preparation_propagate(dispatch, stage, error_type):
    runner, warning, inputs = dispatch
    getattr(runner, stage).side_effect = error_type("accelerator preparation failure")
    with pytest.raises(error_type, match="accelerator preparation failure"):
        clc.try_ctx_clc_mxfp8_gemm(*inputs)
    assert not clc._failed
    runner.run_args.assert_not_called()
    warning.assert_not_called()
