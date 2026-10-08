# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for grouping native CUTLASS calls into model-forward scopes."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.moe.workspace import CutlassWorkspaceReclaimer, register_cutlass_workspace

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def cuda_scope(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    stream = SimpleNamespace(device=torch.device("cuda:0"), device_index=0, cuda_stream=123)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: stream)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: stream.device.index)
    monkeypatch.setattr(
        torch._C, "_cuda_getCurrentRawStream", lambda device: stream.cuda_stream, raising=False
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr("tensorrt_llm._torch.moe.workspace.do_multi_stream", lambda: False)
    return stream


def test_layers_share_one_native_scope_and_warmup_is_explicit(cuda_scope: SimpleNamespace) -> None:
    reclaimer = CutlassWorkspaceReclaimer()
    runner = Mock()
    runner.begin_workspace_forward.return_value = True
    with reclaimer.forward(warmup=True):
        for _ in range(8):
            register_cutlass_workspace(runner, cuda_scope.device)
    runner.begin_workspace_forward.assert_called_once_with(reclaimer._owner, True, 0)
    runner.finish_workspace_forward.assert_called_once_with(True)
    runner.reset_mock()
    with reclaimer.forward(warmup=False):
        register_cutlass_workspace(runner, cuda_scope.device)
    runner.begin_workspace_forward.assert_called_once_with(reclaimer._owner, False, 0)


def test_failed_forward_finishes_all_owners_and_resets_context(cuda_scope: SimpleNamespace) -> None:
    reclaimer = CutlassWorkspaceReclaimer()
    runners = [Mock(), Mock()]
    with pytest.raises(ValueError, match="forward failed"):
        with reclaimer.forward(warmup=False):
            for runner in runners:
                register_cutlass_workspace(runner, cuda_scope.device)
            raise ValueError("forward failed")
    for runner in runners:
        runner.finish_workspace_forward.assert_called_once_with(False)
        runner.reset_mock()
        register_cutlass_workspace(runner, cuda_scope.device)
        runner.begin_workspace_forward.assert_not_called()


def test_rejected_owner_checked_once_and_not_finished(cuda_scope: SimpleNamespace) -> None:
    runner = Mock()
    runner.begin_workspace_forward.return_value = False
    with CutlassWorkspaceReclaimer().forward(warmup=False):
        for _ in range(8):
            register_cutlass_workspace(runner, cuda_scope.device)
    runner.begin_workspace_forward.assert_called_once()
    runner.finish_workspace_forward.assert_not_called()


def test_nested_scope_rejected_and_cleanup_restores_outer(cuda_scope: SimpleNamespace) -> None:
    outer, inner = CutlassWorkspaceReclaimer(), CutlassWorkspaceReclaimer()
    runner = Mock()
    with outer.forward(warmup=True):
        with pytest.raises(RuntimeError, match="Overlapping"):
            with inner.forward(warmup=False):
                pass
        register_cutlass_workspace(runner, cuda_scope.device)
    runner.begin_workspace_forward.assert_called_once_with(outer._owner, True, 0)


@pytest.mark.parametrize("gate", ["do_multi_stream", "capturing"])
def test_unsupported_execution_does_not_register(
    cuda_scope: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, gate: str
) -> None:
    target = (
        "tensorrt_llm._torch.moe.workspace.do_multi_stream"
        if gate == "do_multi_stream"
        else "torch.cuda.is_current_stream_capturing"
    )
    monkeypatch.setattr(target, lambda: True)
    runner = Mock()
    with CutlassWorkspaceReclaimer().forward(warmup=False):
        register_cutlass_workspace(runner, cuda_scope.device)
    runner.begin_workspace_forward.assert_not_called()


def test_streams_have_independent_native_scopes(
    cuda_scope: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    streams = [
        cuda_scope,
        SimpleNamespace(device=cuda_scope.device, device_index=0, cuda_stream=456),
    ]
    current = streams[0]
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: current)
    monkeypatch.setattr(
        torch._C, "_cuda_getCurrentRawStream", lambda device: current.cuda_stream, raising=False
    )
    restored = []

    def restore_stream(stream: SimpleNamespace) -> nullcontext:
        restored.append(stream)
        return nullcontext()

    monkeypatch.setattr(torch.cuda, "stream", restore_stream)
    runner = Mock()
    with CutlassWorkspaceReclaimer().forward(warmup=True):
        register_cutlass_workspace(runner, current.device)
        current = streams[1]
        register_cutlass_workspace(runner, current.device)
    assert runner.begin_workspace_forward.call_count == 2
    assert runner.finish_workspace_forward.call_count == 2
    assert restored == [streams[0]]


def test_cleanup_error_finishes_other_owners_and_scope_can_be_reused(
    cuda_scope: SimpleNamespace,
) -> None:
    reclaimer = CutlassWorkspaceReclaimer()
    first, failing = Mock(), Mock()
    failing.finish_workspace_forward.side_effect = RuntimeError("cleanup failed")
    with pytest.raises(RuntimeError, match="cleanup failed"):
        with reclaimer.forward(warmup=False):
            register_cutlass_workspace(first, cuda_scope.device)
            register_cutlass_workspace(failing, cuda_scope.device)
    first.finish_workspace_forward.assert_called_once_with(True)
    fresh = Mock()
    register_cutlass_workspace(fresh, cuda_scope.device)
    fresh.begin_workspace_forward.assert_not_called()
    with reclaimer.forward(warmup=True):
        register_cutlass_workspace(fresh, cuda_scope.device)
    fresh.finish_workspace_forward.assert_called_once_with(True)


def test_same_stream_handle_on_different_devices_has_distinct_owners(
    cuda_scope: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    streams = [
        SimpleNamespace(device=torch.device("cuda:0"), device_index=0, cuda_stream=0),
        SimpleNamespace(device=torch.device("cuda:1"), device_index=1, cuda_stream=0),
    ]
    current = streams[0]
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: current)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current.device.index)
    monkeypatch.setattr(
        torch._C, "_cuda_getCurrentRawStream", lambda device: current.cuda_stream, raising=False
    )
    restored = []

    def restore_stream(stream: SimpleNamespace) -> nullcontext:
        restored.append(stream)
        return nullcontext()

    monkeypatch.setattr(torch.cuda, "stream", restore_stream)
    runner = Mock()
    with CutlassWorkspaceReclaimer().forward(warmup=True):
        register_cutlass_workspace(runner, cuda_scope.device)
        current = streams[1]
        register_cutlass_workspace(runner, current.device)
    assert runner.begin_workspace_forward.call_count == 2
    assert restored == [streams[0]]


def test_registration_failure_unwinds_previously_accepted_owners(
    cuda_scope: SimpleNamespace,
) -> None:
    first, failing = Mock(), Mock()
    failing.begin_workspace_forward.side_effect = RuntimeError("begin failed")
    with pytest.raises(RuntimeError, match="begin failed"):
        with CutlassWorkspaceReclaimer().forward(warmup=False):
            register_cutlass_workspace(first, cuda_scope.device)
            register_cutlass_workspace(failing, cuda_scope.device)
    first.finish_workspace_forward.assert_called_once_with(False)
    failing.finish_workspace_forward.assert_not_called()


def test_same_stream_cleanup_avoids_stream_context(
    cuda_scope: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    restore = Mock(side_effect=AssertionError("unnecessary stream switch"))
    monkeypatch.setattr(torch.cuda, "stream", restore)
    runner = Mock()
    with CutlassWorkspaceReclaimer().forward(warmup=True):
        register_cutlass_workspace(runner, cuda_scope.device)
    runner.finish_workspace_forward.assert_called_once_with(True)
    restore.assert_not_called()


def test_repeated_layers_construct_stream_only_once(
    cuda_scope: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    current_stream = Mock(return_value=cuda_scope)
    monkeypatch.setattr(torch.cuda, "current_stream", current_stream)
    runner = Mock()
    with CutlassWorkspaceReclaimer().forward(warmup=True):
        for _ in range(30):
            register_cutlass_workspace(runner, cuda_scope.device)
    current_stream.assert_called_once_with(cuda_scope.device)


def test_known_runner_is_started_before_model_allocates_intermediates(
    cuda_scope: SimpleNamespace,
) -> None:
    reclaimer = CutlassWorkspaceReclaimer()
    runner = Mock()
    with reclaimer.forward(warmup=True, num_tokens=4096):
        register_cutlass_workspace(runner, cuda_scope.device)
    runner.reset_mock()
    with reclaimer.forward(warmup=False, num_tokens=3840):
        runner.begin_workspace_forward.assert_called_once_with(reclaimer._owner, False, 3840)
        register_cutlass_workspace(runner, cuda_scope.device)
        runner.begin_workspace_forward.assert_called_once()
    runner.finish_workspace_forward.assert_called_once_with(True)


def test_early_registration_failure_finishes_already_started_runners(
    cuda_scope: SimpleNamespace,
) -> None:
    reclaimer = CutlassWorkspaceReclaimer()
    first, failing = Mock(), Mock()
    with reclaimer.forward(warmup=True, num_tokens=4096):
        register_cutlass_workspace(first, cuda_scope.device)
        register_cutlass_workspace(failing, cuda_scope.device)
    first.reset_mock()
    failing.reset_mock()
    failing.begin_workspace_forward.side_effect = RuntimeError("reservation failed")
    with pytest.raises(RuntimeError, match="reservation failed"):
        with reclaimer.forward(warmup=False, num_tokens=128):
            pytest.fail("The model must not run after scope setup failed")
    first.finish_workspace_forward.assert_called_once_with(False)
    failing.finish_workspace_forward.assert_not_called()
