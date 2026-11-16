# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Python boundary tests; runner discovery is covered by native CUDA tests."""

from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.moe.workspace import CutlassWorkspaceReclaimer

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def native_scope(monkeypatch: pytest.MonkeyPatch) -> Mock:
    scope = Mock()
    scope.finish.return_value = True
    monkeypatch.setattr(torch.classes.trtllm, "EagerMoeWorkspaceScope", Mock(return_value=scope))
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr("tensorrt_llm._torch.moe.workspace.do_multi_stream", lambda: False)
    return scope


def test_one_native_boundary_per_model_forward(native_scope: Mock) -> None:
    reclaimer = CutlassWorkspaceReclaimer()
    with reclaimer.forward(warmup=True, num_tokens=4096):
        native_scope.begin.assert_called_once_with(True, 4096)
        native_scope.finish.assert_not_called()
    native_scope.finish.assert_called_once_with(True)
    native_scope.reset_mock()
    with reclaimer.forward(warmup=False, num_tokens=128):
        native_scope.begin.assert_called_once_with(False, 128)
    native_scope.finish.assert_called_once_with(True)


def test_failed_forward_aborts_native_scope(native_scope: Mock) -> None:
    with pytest.raises(ValueError, match="model failed"):
        with CutlassWorkspaceReclaimer().forward(warmup=False):
            raise ValueError("model failed")
    native_scope.finish.assert_called_once_with(False)


def test_empty_warmup_skips_serving_boundaries_but_allows_later_warmup(native_scope: Mock) -> None:
    reclaimer = CutlassWorkspaceReclaimer()
    native_scope.finish.return_value = False
    with reclaimer.forward(warmup=True):
        pass
    native_scope.reset_mock()
    with reclaimer.forward(warmup=False):
        pass
    native_scope.begin.assert_not_called()
    native_scope.finish.assert_not_called()
    native_scope.finish.return_value = True
    with reclaimer.forward(warmup=True):
        pass
    with reclaimer.forward(warmup=False):
        pass
    assert native_scope.begin.call_count == 2
    assert native_scope.finish.call_count == 2


@pytest.mark.parametrize("gate", ["do_multi_stream", "capturing"])
def test_unsupported_execution_bypasses_native_scope(
    native_scope: Mock, monkeypatch: pytest.MonkeyPatch, gate: str
) -> None:
    target = (
        "tensorrt_llm._torch.moe.workspace.do_multi_stream"
        if gate == "do_multi_stream"
        else "torch.cuda.is_current_stream_capturing"
    )
    monkeypatch.setattr(target, lambda: True)
    with CutlassWorkspaceReclaimer().forward(warmup=False):
        pass
    native_scope.begin.assert_not_called()
    native_scope.finish.assert_not_called()
