# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serving measurements queue below onboarding and wait for node teardown."""

from __future__ import annotations

import json
import subprocess
import threading
import time
from pathlib import Path

from agent_flow.workflows.trtllm_modeling_v2.infra.compute import spoolctl
from agent_flow.workflows.trtllm_modeling_v2.tune import measure
from agent_flow.workflows.trtllm_modeling_v2.tune.measure import SpoolRunner, settle_prelude


def _spool(root: Path, name: str, *, running=(), pending=()) -> str:
    """A live session spool whose watcher runs ``running`` and holds ``pending`` (dicts)."""
    spool = root / name
    for sub in ("running", "done", "pending"):
        (spool / sub).mkdir(parents=True, exist_ok=True)
    record = {
        "watcher": name,
        "host": name,
        "alive_epoch": time.time(),
        "slurm_job": "1",
        "running": list(running),
        "pending": list(pending),
    }
    (spool / "status.json").write_text(json.dumps(record))
    return str(spool)


def test_a_measurement_takes_an_idle_node_first_then_the_busy_one_with_least_ahead(tmp_path):
    tuner = {"name": "ab-cand", "wall_s": 2700.0, "elapsed_s": 600.0}
    gate = {"name": "regression", "wall_s": 7200.0, "elapsed_s": 300.0}
    onboarding = {"name": "gate-x", "effective_priority": 1}
    background = {"name": "tuner-next", "effective_priority": -2}
    gates = _spool(tmp_path, "gates", running=[gate], pending=[onboarding])
    work = _spool(tmp_path, "work", running=[tuner], pending=[background])
    idle = _spool(tmp_path, "idle")
    runner = SpoolRunner([gates, work, idle])
    assert runner.free_spools() == [idle, work, gates]
    assert "running ab-cand" in runner.not_idle[work]


def test_a_measurement_settles_its_node_and_a_gate_job_does_not(tmp_path, monkeypatch):
    spool = _spool(tmp_path, "work")
    queued = {}
    monkeypatch.setattr(
        spoolctl, "queue_job", lambda sp, name, script: queued.update({name: script})
    )
    runner = SpoolRunner([spool], prefix="tune")
    runner.submit(spool, "e2e-x", "echo measured")
    runner.submit(spool, "gate-x", "echo gated", gpus="4")
    assert "[settle] node idle after" in queued["tune-e2e-x"]
    assert "! -name tune-e2e-x.sh" in queued["tune-e2e-x"]
    assert "[settle]" not in queued["tune-gate-x"]
    fresh = SpoolRunner([spool], prefix="tune")
    Path(spool, "done", "tune-old.rc").write_text("0\n")
    assert not fresh.adopt("old", active_only=True)
    assert fresh.adopt("old")


def _run(prelude: str) -> tuple[float, str]:
    start = time.monotonic()
    out = subprocess.run(["bash", "-c", prelude], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return (time.monotonic() - start, out.stdout)


def test_the_settle_waits_out_the_job_before_and_its_teardown(tmp_path):
    spool = Path(_spool(tmp_path, "work"))
    (spool / "running" / "tune-e2e-x.sh").write_text("")
    took, out = _run(settle_prelude(str(spool), "tune-e2e-x", settle_s=1, wait_s=20))
    assert took < 3 and "node idle after" in out
    other = spool / "running" / "tuner-ab.sh"
    other.write_text("")

    def end_other():
        time.sleep(1.0)
        other.unlink()
        (spool / "done" / "tuner-ab.rc").write_text("0\n")

    threading.Thread(target=end_other).start()
    took, out = _run(settle_prelude(str(spool), "tune-e2e-x", settle_s=2, wait_s=20))
    assert took >= 3.0 and "[settle] node idle after" in out


def test_the_settle_gives_up_after_its_wait(tmp_path):
    spool = Path(_spool(tmp_path, "work"))
    (spool / "running" / "build-cpu.sh").write_text("")
    out = subprocess.run(
        ["bash", "-c", settle_prelude(str(spool), "tune-e2e-x", settle_s=1, wait_s=2)],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert out.returncode == measure.RC_REFUSED
    assert "measurement deferred" in out.stderr and "node idle after" not in out.stdout
    assert measure.SETTLE_WAIT_S >= 60
