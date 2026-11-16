# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the GPU qualification's passive containment observer."""

import json
from pathlib import Path
from unittest.mock import Mock

import psutil
import pytest
import test_transfer_lifecycle_gpu as qualification

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("selected_rank", [0, 1], ids=["rank0", "rank1"])
@pytest.mark.parametrize(
    "scenario,expected",
    [
        ("absent", "absent"),
        ("gone_during_identity", "absent"),
        ("gone_during_status", "absent"),
        ("zombie", "zombie"),
        ("reused", "reused"),
        ("live", "live"),
        ("denied", "unobservable"),
        ("denied_status", "unobservable"),
        ("exits_later", "absent"),
    ],
)
def test_rank_exit_observation(
    scenario: str,
    expected: str,
    selected_rank: int,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Distinguish original-rank exit from live/unknown state without cleanup.

    Args:
        scenario: Distinct process identity/state transition or observation error.
        expected: Final classification required in the persisted diagnostic.
        selected_rank: Rank exercising the scenario while the other rank is absent.
        tmp_path: Isolated identity and diagnostic directory.
        monkeypatch: Process/clock substitutes; no GPU or real signals are used.
    """
    for rank in range(2):
        (tmp_path / f"gen.{rank}.spawned.json").write_text(
            json.dumps({"pid": 100 + rank, "created": 1.0})
        )
    selected_pid = 100 + selected_rank
    process = Mock(spec=psutil.Process)
    process.create_time.return_value = 2.0 if scenario == "reused" else 1.0
    process.status.return_value = (
        psutil.STATUS_ZOMBIE if scenario == "zombie" else psutil.STATUS_RUNNING
    )
    if scenario == "gone_during_identity":
        process.create_time.side_effect = psutil.NoSuchProcess(selected_pid)
    elif scenario == "gone_during_status":
        process.status.side_effect = psutil.NoSuchProcess(selected_pid)
    elif scenario == "denied_status":
        process.status.side_effect = psutil.AccessDenied(selected_pid)
    for method in (process.terminate, process.kill, process.send_signal):
        method.side_effect = AssertionError("exit observation must not send signals")
    now = [0.0]

    def lookup(pid: int) -> psutil.Process:
        """Return the selected original rank, or its observed disappearance."""
        if (
            pid != selected_pid
            or scenario == "absent"
            or (scenario == "exits_later" and now[0] > 0)
        ):
            raise psutil.NoSuchProcess(pid)
        if scenario == "denied":
            raise psutil.AccessDenied(pid)
        return process

    def advance(delay: float) -> None:
        """Advance only the observer's fake CPU-test clock."""
        now[0] += delay

    monkeypatch.setattr(psutil, "Process", lookup)
    monkeypatch.setattr(qualification.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(qualification.time, "sleep", advance)
    signal_spy = Mock(side_effect=AssertionError("exit observation must not signal or clean up"))
    monkeypatch.setattr(qualification.os, "kill", signal_spy)
    monkeypatch.setattr(qualification, "_stop_owned_jobs", signal_spy)
    if expected in ("live", "unobservable"):
        with pytest.raises(AssertionError, match="did not exit before cleanup"):
            qualification._wait_for_rank_exit(tmp_path, "gen", timeout=0.02)
    else:
        qualification._wait_for_rank_exit(tmp_path, "gen", timeout=0.02)
    evidence = json.loads((tmp_path / "gen.exit_check.json").read_text())
    assert evidence["passed"] == (expected not in ("live", "unobservable"))
    assert evidence["history"][-1]["ranks"][selected_rank]["state"] == expected
    assert evidence["history"][-1]["ranks"][1 - selected_rank]["state"] == "absent"
    assert evidence["elapsed"] <= 0.02
    if scenario == "exits_later":
        assert [item["ranks"][selected_rank]["state"] for item in evidence["history"]] == [
            "live",
            "absent",
        ]
    if expected in ("live", "unobservable"):
        assert evidence["elapsed"] == 0.02
    signal_spy.assert_not_called()
    for method in (process.terminate, process.kill, process.send_signal):
        method.assert_not_called()
