# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock

import pytest

from tensorrt_llm._torch.pyexecutor.hang_diagnostics import (
    HANG_DIAGNOSTICS_ENV,
    ExecutorHangDiagnostics,
)


def test_hang_diagnostics_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(HANG_DIAGNOSTICS_ENV, raising=False)

    assert ExecutorHangDiagnostics.from_environment(rank=0) is None


@pytest.mark.parametrize(
    ("event_complete", "expected"),
    [(False, "pending"), (True, "complete")],
)
def test_hang_diagnostics_reports_latest_phase_and_cuda_state(
    monkeypatch: pytest.MonkeyPatch,
    event_complete: bool,
    expected: str,
) -> None:
    monkeypatch.setenv(HANG_DIAGNOSTICS_ENV, "1")
    diagnostics = ExecutorHangDiagnostics.from_environment(rank=3)
    assert diagnostics is not None
    event = Mock()
    event.query.return_value = event_complete

    diagnostics.record(
        "forward_returned",
        iteration=17,
        details="active_request_ids=[42]",
        forward_completion_event=event,
    )
    status = diagnostics.get_status_dump()

    assert "rank=3" in status
    assert "iteration=17" in status
    assert "phase=forward_returned" in status
    assert f"forward_cuda_completion={expected}" in status
    assert "active_request_ids=[42]" in status


def test_hang_diagnostics_tolerates_cuda_query_failure() -> None:
    diagnostics = ExecutorHangDiagnostics(rank=0)
    event = Mock()
    event.query.side_effect = RuntimeError("CUDA context unavailable")
    diagnostics.record("sampling", iteration=2, forward_completion_event=event)

    status = diagnostics.get_status_dump()

    assert "forward_cuda_completion=query-failed(RuntimeError: CUDA context unavailable)" in status


def test_hang_diagnostics_retains_event_until_explicitly_cleared() -> None:
    diagnostics = ExecutorHangDiagnostics(rank=0)
    event = Mock()
    event.query.return_value = False
    diagnostics.record("forward_returned", iteration=2, forward_completion_event=event)

    diagnostics.record("sampling", iteration=2)
    assert "forward_cuda_completion=pending" in diagnostics.get_status_dump()

    diagnostics.record(
        "scheduling",
        iteration=3,
        forward_completion_event=None,
    )
    assert "forward_cuda_completion=not-recorded" in diagnostics.get_status_dump()
