# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A GPU job starts only on devices whose memory the last job released.

An engine sizes its KV-cache and recurrent pools from the device memory it
finds free. One started beside memory a cancelled job had not given back
reserved 642 slots where the same flags reserve 1237 everywhere else, and its
gate certified the smaller pool. The device reading is injected, so none of
this needs a GPU.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra.substrate import scheduler, sessions, spoolctl
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate.scheduler import (
    RC_CANCELLED,
    Scheduler,
    SchedulerConfig,
    SpoolPaths,
    build_parser,
    config_from_args,
    read_device_memory_mib,
)

from .test_spool_scheduler import FakeClock, blocking, pump, rc, record, release, running, submit


class Smi:
    """A fake device reading: memory in use (MiB) by device id, and how often it was taken."""

    def __init__(self, used: dict[str, int]) -> None:
        self.used = dict(used)
        self.calls = 0

    def __call__(self) -> dict[str, int]:
        self.calls += 1
        return dict(self.used)


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def spool(tmp_path):
    root = tmp_path / "spool"
    SpoolPaths(root).ensure()
    return root


def _environ() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}


def make(spool, clock, used: dict[str, int], **config) -> tuple[Scheduler, Smi]:
    config.setdefault("devices", ("0", "1"))
    config.setdefault("kill_grace_s", 0.5)
    config.setdefault("leftover_grace_s", 0.0)
    sched = Scheduler(spool, SchedulerConfig(**config), clock=clock, environ=_environ())
    smi = Smi(used)
    sched.device_memory_mib = smi
    return sched, smi


def reason(sched, clock, name: str) -> str:
    return next(p["reason"] for p in sched.snapshot(clock.now)["pending"] if p["name"] == name)


def status_text(spool) -> str:
    return SpoolPaths(spool).status.read_text(encoding="utf-8")


# ------------------------------------------------------------------ waiting


def test_a_gpu_job_waits_until_its_device_memory_is_released(spool, clock):
    sched, smi = make(spool, clock, {"0": 40211}, devices=("0",))
    submit(spool, "boot", "echo started", header="gpus=1")
    sched.tick()
    assert "boot" not in sched.jobs
    assert reason(sched, clock, "boot").startswith(
        "device memory in use: 0 (40211 MiB) > 1024 MiB (waiting 0s of 300s"
    )
    clock.advance(30)
    sched.tick()
    assert "waiting 30s of 300s" in reason(sched, clock, "boot")
    smi.used["0"] = 3
    pump(sched, lambda: rc(spool, "boot") is not None)
    assert rc(spool, "boot") == 0
    assert record(spool, "boot")["devices"] == ["0"]


def test_devices_whose_memory_is_back_are_handed_out_first(spool, clock):
    sched, smi = make(spool, clock, {"0": 40211, "1": 2, "2": 0}, devices=("0", "1", "2"))
    submit(spool, "a", blocking(spool, "a"), header="gpus=1", submitted=clock.now - 20)
    submit(spool, "b", blocking(spool, "b"), header="gpus=1", submitted=clock.now - 10)
    sched.tick()
    assert sched.jobs["a"].devices == ("1",) and sched.jobs["b"].devices == ("2",)
    assert smi.calls == 1, "one reading serves every job a pass starts"
    for name in ("a", "b"):
        release(spool, name)
    pump(sched, lambda: not sched.jobs)


def test_a_job_waiting_for_device_memory_holds_the_queue_behind_it(spool, clock):
    """It waits for devices, which the jobs ranked below it would take."""
    sched, _ = make(spool, clock, {"0": 0, "1": 40211})
    submit(spool, "wide", "echo wide", header="gpus=2", submitted=clock.now - 20)
    submit(spool, "narrow", "echo narrow", header="gpus=1", submitted=clock.now - 10)
    sched.tick()
    assert not sched.jobs
    assert reason(sched, clock, "wide").startswith("device memory in use: 1 (40211 MiB)")
    assert reason(sched, clock, "narrow") == "queued behind wide"


def test_memory_this_sessions_running_jobs_hold_never_holds_a_job(spool, clock):
    sched, smi = make(spool, clock, {"0": 0, "1": 0})
    submit(spool, "engine", blocking(spool, "engine"), header="gpus=1")
    pump(sched, lambda: "engine" in running(sched))
    assert sched.jobs["engine"].devices == ("0",)
    smi.used["0"] = 150000  # the running engine's own pools
    submit(spool, "probe", blocking(spool, "probe"), header="gpus=1")
    sched.tick()
    assert sched.jobs["probe"].devices == ("1",)
    clock.advance(3600)
    sched.tick()
    assert sched.snapshot(clock.now)["quarantined"] == []
    for name in ("engine", "probe"):
        release(spool, name)
    pump(sched, lambda: not sched.jobs)


# --------------------------------------------------------------- quarantine


def test_a_device_that_never_releases_its_memory_is_quarantined(spool, clock):
    sched, smi = make(spool, clock, {"0": 40211, "1": 0})
    submit(spool, "holder", blocking(spool, "holder"), header="gpus=1")
    pump(sched, lambda: "holder" in running(sched))
    assert sched.jobs["holder"].devices == ("1",)
    submit(spool, "boot", "echo started", header="gpus=1")
    sched.tick()
    assert reason(sched, clock, "boot").startswith("device memory in use: 0 (40211 MiB)")
    clock.advance(301)
    sched.tick()
    snapshot = sched.snapshot(clock.now)
    assert snapshot["quarantined"] == ["0"] and snapshot["free"] == []
    assert snapshot["quarantined_memory"]["0"]["used_mib"] == 40211
    assert reason(sched, clock, "boot") == (
        "waiting for 1 GPU(s); 0 free; quarantined, memory not released: 0 (40211 MiB)"
    )
    assert "watcher: device 0 quarantined" in status_text(spool)
    assert sched.counters["device_quarantined"] == 1
    # Placed elsewhere: the device the holder gives back, not the quarantined one.
    release(spool, "holder")
    pump(sched, lambda: rc(spool, "boot") is not None)
    assert record(spool, "boot")["devices"] == ["1"]
    assert sched.snapshot(clock.now)["quarantined"] == ["0"]


def test_a_whole_node_job_takes_a_quarantined_device_back_once_it_is_released(spool, clock):
    sched, smi = make(spool, clock, {"0": 0, "1": 40211})
    submit(spool, "engine", "echo started", header="gpus=all")
    sched.tick()
    assert reason(sched, clock, "engine").startswith("device memory in use: 1 (40211 MiB)")
    clock.advance(301)
    sched.tick()
    assert reason(sched, clock, "engine") == (
        "waiting for the whole node to drain; quarantined, memory not released: 1 (40211 MiB)"
    )
    calls = smi.calls
    sched.tick()
    assert smi.calls == calls + 1, "a job the quarantined device would place reads it each pass"
    smi.used["1"] = 4
    pump(sched, lambda: rc(spool, "engine") is not None)
    assert record(spool, "engine")["devices"] == ["0", "1"]
    assert "watcher: device 1 released its memory (4 MiB in use)" in status_text(spool)


def test_a_quarantined_device_is_read_again_while_nothing_waits(spool, clock):
    sched, smi = make(spool, clock, {"0": 40211}, devices=("0",))
    submit(spool, "boot", "echo started", header="gpus=1")
    sched.tick()
    clock.advance(301)
    sched.tick()
    assert sched.snapshot(clock.now)["quarantined"] == ["0"]
    spoolctl.cancel_job(spool, "boot")
    sched.tick()
    assert rc(spool, "boot") == RC_CANCELLED
    calls = smi.calls
    smi.used["0"] = 0
    clock.advance(scheduler.DEVICE_RECHECK_S / 2)
    sched.tick()
    assert smi.calls == calls
    clock.advance(scheduler.DEVICE_RECHECK_S / 2)
    sched.tick()
    assert smi.calls == calls + 1
    snapshot = sched.snapshot(clock.now)
    assert snapshot["quarantined"] == [] and snapshot["free"] == ["0"]


# --------------------------------------------------------------------- cost


def test_one_reading_a_pass_and_only_when_a_gpu_job_is_about_to_start(spool, clock):
    sched, smi = make(spool, clock, {"0": 0, "1": 0, "2": 0}, devices=("0", "1", "2"))
    submit(spool, "free-notes", "true")
    pump(sched, lambda: rc(spool, "free-notes") is not None)
    assert smi.calls == 0, "a job that holds no device reads none"
    for name in ("a", "b", "c"):
        submit(spool, name, blocking(spool, name), header="gpus=1")
    sched.tick()
    assert running(sched) == {"a", "b", "c"} and smi.calls == 1
    submit(spool, "d", "true", header="gpus=1")
    sched.tick()
    sched.tick()
    assert smi.calls == 1, "a job with no free device to read reads none"
    for name in ("a", "b", "c"):
        release(spool, name)
    pump(sched, lambda: rc(spool, "d") is not None)


def test_the_check_is_off_at_zero(spool, clock):
    sched, smi = make(spool, clock, {"0": 40211}, devices=("0",), device_max_used_mib=0)
    submit(spool, "boot", "echo started", header="gpus=1")
    pump(sched, lambda: rc(spool, "boot") is not None)
    assert rc(spool, "boot") == 0 and smi.calls == 0


def test_without_nvidia_smi_a_job_starts_as_before(spool, clock, monkeypatch, tmp_path):
    sched = Scheduler(
        spool,
        SchedulerConfig(devices=("0",), kill_grace_s=0.5, leftover_grace_s=0.0),
        clock=clock,
        environ=_environ(),
    )
    empty = tmp_path / "no-nvidia-smi"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
    assert read_device_memory_mib() is None
    submit(spool, "boot", "echo started", header="gpus=1")
    pump(sched, lambda: rc(spool, "boot") is not None)
    assert rc(spool, "boot") == 0
    assert sched.snapshot(clock.now)["quarantined"] == []


def _fake_smi(directory: Path, body: str) -> None:
    directory.mkdir(exist_ok=True)
    smi = directory / "nvidia-smi"
    smi.write_text(
        "#!/bin/bash\n"
        '[ "$1" = --query-gpu=index,uuid,memory.used ] && '
        '[ "$2" = --format=csv,noheader,nounits ] || exit 2\n' + body
    )
    smi.chmod(0o755)


def test_the_reading_finds_a_device_by_index_and_by_uuid(monkeypatch, tmp_path):
    """nvidia-smi reports the whole node; a watcher's ids are CUDA_VISIBLE_DEVICES entries."""
    bin_dir = tmp_path / "bin"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    _fake_smi(bin_dir, "echo '0, GPU-aaa, 40211'\necho '1, GPU-bbb, 3'\necho '2, GPU-ccc, [N/A]'\n")
    assert read_device_memory_mib() == {"0": 40211, "GPU-aaa": 40211, "1": 3, "GPU-bbb": 3}
    _fake_smi(bin_dir, "echo 'NVIDIA-SMI has failed because it could not communicate'\nexit 9\n")
    assert read_device_memory_mib() is None


# ---------------------------------------------------------------- operators


def test_spoolctl_status_shows_why_a_job_waits_for_device_memory(spool, clock):
    sched, _ = make(spool, clock, {"0": 40211}, devices=("0",))
    submit(spool, "boot", "echo started", header="gpus=1")
    sched.tick()
    text = spoolctl.format_status(spoolctl.status(spool))
    assert "WAIT boot" in text and "device memory in use: 0 (40211 MiB) > 1024 MiB" in text
    clock.advance(301)
    sched.tick()
    text = spoolctl.format_status(spoolctl.status(spool))
    assert "quarantined 0" in text
    assert "quarantined, memory not released: 0 (40211 MiB)" in text


def test_the_check_is_on_by_default_and_a_session_tunes_it():
    config = config_from_args(build_parser().parse_args(["watch", "/s", "--gpus", "0,1"]), {})
    assert (config.device_max_used_mib, config.device_release_wait_s) == (1024.0, 300.0)
    session = sessions.SessionSpec(
        name="a",
        spool="/s",
        scheduler={"device_max_used_mib": 2048, "device_release_wait": "10m"},
    )
    flags = sessions.scheduler_arguments(session)
    assert flags == ["--device-max-used-mib", "2048", "--device-release-wait", "10m"]
    config = config_from_args(build_parser().parse_args(["watch", "/s", "--gpus", "0", *flags]), {})
    assert (config.device_max_used_mib, config.device_release_wait_s) == (2048.0, 600.0)


@pytest.mark.parametrize(
    "options, fragment",
    [
        ({"device_max_used_mib": 2048, "device_release_wait": "10m"}, None),
        ({"device_max_used_mib": 0}, None),
        ({"device_max_used_mib": "lots"}, "device_max_used_mib"),
        ({"device_max_used_mib": -1}, "device_max_used_mib"),
        ({"device_release_wait": "until it is free"}, "device_release_wait"),
    ],
)
def test_a_session_validates_its_device_memory_options(options, fragment):
    errors: list[str] = []
    block = {"sessions": [{"name": "a", "spool": "/s/a", "scheduler": options}]}
    sessions.validate_sessions(block, errors)
    if fragment is None:
        assert errors == []
    else:
        assert any(fragment in error for error in errors), errors
