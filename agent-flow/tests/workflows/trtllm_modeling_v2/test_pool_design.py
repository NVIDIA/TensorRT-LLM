# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""What an independent reader of the design held, and no other test of these files does.

These cases were written from the design text alone (the watcher's rules, the gang, the
allocation probe). The ones whose decision ``test_pool.py``, ``test_spool_scheduler.py`` or
``test_session_gang.py`` also holds went in the eighth round; these six are held nowhere
else: the exact boundary of Fit, a dead successor that is no reason to retire, a job lost
twice, a gang that waits for an unknown device reading, a probe record of another world's
size, and a native build told apart by its bytes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from agent_flow.workflows.trtllm_modeling_v2.infra.compute import probe, scheduler, spoolctl


@pytest.fixture
def clock():
    return [2000000000.0]


@pytest.fixture
def scheduler_factory(tmp_path, monkeypatch, clock):
    monkeypatch.setattr(scheduler, "read_proc_table", lambda: {})

    def make(name="watcher", nodes=1, end=None, spool=None):
        cfg = scheduler.SchedulerConfig(
            devices=("0", "1", "2", "3"), holder=name, nodes=nodes, ends_at=end
        )
        obj = scheduler.Scheduler(
            spool or tmp_path / "spool",
            cfg,
            clock=lambda: clock[0],
            environ={"PATH": "/usr/bin:/bin"},
            watcher_id=name,
        )
        obj.device_memory_mib = lambda: {str(i): 0 for i in range(4)}
        obj.device_holders = lambda: {}
        obj.announce()
        return obj

    return make


# ---- watcher contract


def queue(sched, name, header):
    script = spoolctl.render_job("true", header=header)
    spoolctl.queue_job(sched.paths.root, name, script, require_watcher=False)


def pump(sched, name):
    for _ in range(100):
        sched.tick()
        if sched.paths.receipt(name).exists():
            return json.loads(sched.paths.record(name).read_text())
        time.sleep(0.01)
    raise AssertionError("CPU fixture did not complete")


def test_exact_wall_margin_fits(scheduler_factory, clock):
    sched = scheduler_factory(end=clock[0] + 30 + scheduler.FIT_MARGIN_S)
    queue(sched, "fits", {"gpus": 1, "wall": 30})
    assert pump(sched, "fits")["rc"] == 0


def test_retirement_requires_a_live_holder_of_the_same_shape(scheduler_factory, clock):
    old = scheduler_factory("old", nodes=2, end=clock[0] + 500)
    scheduler_factory("narrow", nodes=1, end=clock[0] + 1500, spool=old.paths.root)
    assert not old.snapshot(clock[0])["retiring"]
    # Nor for a wider one that ends later: it is there for wider work and is given back when
    # that is over, and the run would then be left with no holder of this width.
    scheduler_factory("wider", nodes=5, end=clock[0] + 1550, spool=old.paths.root)
    assert not old.snapshot(clock[0])["retiring"]
    scheduler_factory("same", nodes=2, end=clock[0] + 1600, spool=old.paths.root)
    assert old.snapshot(clock[0])["retiring"]
    clock[0] += scheduler.WATCHER_STALE_S + 1
    old.heartbeat()
    assert not old.snapshot(clock[0])["retiring"], "dead successor is not retirement"


def orphan(sched, name, reruns):
    text = spoolctl.render_job("true", header={"gpus": 1, "wall": 30})
    if reruns:
        text = scheduler.stamp_rerun(text, reruns)
    (sched.paths.running / f"{name}.sh").write_text(text)
    (sched.paths.running / f"{name}.json").write_text(
        json.dumps(
            {
                "watcher": "gone",
                "host": "other",
                "slurm_job": "OLD",
                "started": "before",
                "devices": ["0"],
            }
        )
    )


def test_lost_job_reruns_only_once_then_no_result(scheduler_factory):
    sched = scheduler_factory()
    sched.device_memory_mib = lambda: None
    orphan(sched, "lost", 0)
    sched.tick()
    pending = sched.paths.pending / "lost.sh"
    assert pending.exists() and scheduler.reruns_of(pending.read_text()) == 1
    os.replace(pending, sched.paths.running / "lost.sh")
    (sched.paths.running / "lost.json").write_text(
        json.dumps(
            {
                "watcher": "gone-again",
                "host": "other",
                "slurm_job": "OLDER",
                "devices": ["0"],
            }
        )
    )
    sched.tick()
    record = json.loads(sched.paths.record("lost").read_text())
    assert record["reason"] == "node_lost" and record["rc"] != 0 and not pending.exists()


# ---- gang and probe contract


@pytest.mark.parametrize("reading", [None, {"0": 3000}])
def test_gang_waits_for_clean_reading_on_every_reserved_node(tmp_path, monkeypatch, clock, reading):
    monkeypatch.setattr(scheduler, "read_proc_table", lambda: {})
    spool = tmp_path / "spool"
    cfg = scheduler.SchedulerConfig(holder="holder", nodes=2, lead=True, ends_at=clock[0] + 1000)
    lead = scheduler.Scheduler(spool, cfg, clock=lambda: clock[0], environ={}, watcher_id="lead")
    lead.announce()
    members = []
    for i in range(2):
        c = scheduler.SchedulerConfig(
            holder="holder", nodes=2, devices=("0",), ends_at=clock[0] + 1000
        )
        w = scheduler.Scheduler(spool, c, clock=lambda: clock[0], environ={}, watcher_id=f"node{i}")
        w.host = f"node{i}"
        w.device_memory_mib = lambda: reading
        w.device_holders = lambda: {}
        w.announce()
        members.append(w)
    spoolctl.queue_job(
        spool,
        "gang",
        spoolctl.render_job("true", header={"nodes": 2, "wall": 30}),
        require_watcher=False,
    )
    lead.tick()
    for w in members:
        w.tick()
    lead.tick()
    started = "gang" in lead.jobs or (spool / "running/gang.sh").exists()
    # Reap harmless CPU child before asserting; no GPUs/process signals involved.
    if "gang" in lead.jobs:
        lead.jobs["gang"].proc.wait(timeout=5)
        lead.tick()
    assert not started, "gang ready marker bypasses Clean for unknown or busy devices"


def test_probe_world_size_must_match_expected(tmp_path):
    (tmp_path / f"{probe.ROLE}.rank0.json").write_text(
        json.dumps({"rank": 0, "size": 8, "host": "h", "problems": []})
    )
    problems, _ = probe.judge(tmp_path, 1)
    assert problems, "expected one world rank, record claims world eight"


def test_native_identity_depends_on_bytes_not_mtime(tmp_path):
    # Drive the real interpreter-start hook, not its private digest implementation.
    hook = Path(scheduler.__file__).parent / "resolvehook"
    package = tmp_path / "tensorrt_llm"
    (package / "libs").mkdir(parents=True)
    (package / "__init__.py").write_text("")
    lib = package / "libs/lib.so"
    builds = []
    for i, content in enumerate((b"AAAA", b"BBBB")):
        lib.write_bytes(content)
        os.utime(lib, ns=(1000000000, 1000000000))
        log = tmp_path / f"origin-{i}.jsonl"
        env = {
            "PATH": "/usr/bin:/bin",
            "TZ": "UTC",
            "LC_ALL": "C.UTF-8",
            "PYTHONPATH": os.pathsep.join((str(hook), str(tmp_path))),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
            "MV2FLOW_RESOLVE_LOG": str(log),
            "MV2FLOW_RESOLVE_MODULES": "tensorrt_llm",
            "MV2FLOW_ROLE": "server",
            "RANK": "0",
        }
        done = subprocess.run(
            [sys.executable, "-s", "-c", "import tensorrt_llm"],
            env=env,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert done.returncode == 0, done.stderr
        builds.append(json.loads(log.read_text().splitlines()[0])["build"])
    assert all(builds) and builds[0] != builds[1]
