# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A check run by hand gets its devices as a run does, and gives them back."""

from __future__ import annotations

import fcntl
import os
import re
import stat
import sys
import threading
import time
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import gate_cli
from agent_flow.workflows.trtllm_modeling_v2.infra.compute import pool as pool_mod
from agent_flow.workflows.trtllm_modeling_v2.infra.compute import slurm as slurm_mod
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.scheduler import STOP_MARKER

FOUR = ("0", "1", "2", "3")


class Cluster:
    """A stand-in Slurm, and the nodes it gives: a submitted holder runs and serves its spool."""

    def __init__(self, fake, spool):
        self.fake, self.spool = fake, spool
        self.stops: list[threading.Event] = []
        self._done = threading.Event()
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self) -> None:
        from .test_pool import _serving

        while not self._done.wait(0.05):
            for job in self.fake.jobs:
                if job["state"] != "PENDING":
                    continue
                asked = re.search(r"--time=(\S+)", open(job["script"]).read()).group(1)
                ends = int(time.time() + slurm_mod.slurm_time_seconds(asked))
                self.fake.change(job["id"], state="RUNNING", start=int(time.time()), end=ends)
                self.stops.append(
                    _serving(self.spool, host="nodeA", devices=FOUR, holder=job["name"], nodes=1)
                )

    def close(self) -> None:
        self._done.set()
        for stop in self.stops:
            stop.set()


@pytest.fixture
def by_hand(task, tmp_path, monkeypatch):
    """A spec in the flow's default form (Slurm, no spool stated), with no run going."""
    from .test_pool import _FAKE, SLURM, FakeSlurm, _mounts

    bindir, root = tmp_path / "slurm-bin", tmp_path / "fake-slurm"
    bindir.mkdir()
    root.mkdir()
    for tool in ("sbatch", "squeue", "sacct", "scancel"):
        script = bindir / tool
        script.write_text(_FAKE.format(python=sys.executable))
        script.chmod(script.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_SLURM", str(root))
    monkeypatch.setattr(pool_mod, "END_WAIT_S", 0.3)
    slurm = {**SLURM, "time": "08:00:00", "lead": "10m", "container_mounts": _mounts(tmp_path)}
    spec = {
        **task,
        "arch": "sm_103",
        "environment": {"python": sys.executable, "slurm": slurm},
        "deployments": [{"name": "core", "options": {}, "timeout": 120}],
    }
    workspace = tmp_path / "ws"
    workspace.mkdir()
    # A tree whose package a rank can import and whose build it can name, for the origin judge.
    libs = Path(task["trtllm_repo_path"]) / "tensorrt_llm" / "libs"
    libs.mkdir()
    (libs / "libth_common.so").write_bytes(b"build-one")

    def gate(**changes):
        slurm.update(changes)
        (workspace / "task.yaml").write_text(yaml.safe_dump(spec))
        return gate_cli.main(
            ["deployment", "--task", str(workspace / "task.yaml")]
            + ["--tree", task["trtllm_repo_path"], "--workdir", str(tmp_path / "work")]
        )

    cluster = Cluster(FakeSlurm(root), workspace / "spool")
    yield gate, cluster, workspace
    cluster.close()


def test_a_check_by_hand_asks_for_one_holder_of_its_jobs_length_and_gives_it_back(by_hand, capsys):
    gate, cluster, workspace = by_hand
    code, said = gate(), capsys.readouterr()
    assert code == 0, said.out + said.err
    assert said.out.startswith("check deployment:core\n") and "[PASS] deployment@tp1" in said.out
    assert "pool: no run keeps this workspace's pool; asking for a holder of 1 node(s)" in said.err
    # One holder, of the job's nodes, for the job's wall and the pool's lead: the session's
    # twelve minutes and the lead's ten. Not the eight hours the spec's holders last.
    (job,) = cluster.fake.jobs
    script = open(job["script"]).read()
    assert len(cluster.fake.calls("sbatch")) == 1 and "--nodes=1" in script
    assert re.search(r"--time=(\S+)", script).group(1) == "00:22:00"
    # Given back when the check was done, and the workspace is free for a run again.
    assert job["state"].startswith("CANCELLED")
    assert (workspace / "spool" / STOP_MARKER).exists()
    with (workspace / gate_cli.LOCK_FILENAME).open("a+") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    # What it asked for is on the workspace's record, where the bound of node-hours counts it.
    assert pool_mod.asked(workspace / "spool").holder_jobs == 1


def test_a_check_by_hand_is_served_by_the_pool_of_a_run_that_is_going(by_hand, capsys):
    from .test_pool import _serving

    gate, cluster, workspace = by_hand
    # A run's controller holds the workspace, and its pool has a node that serves the spool.
    with (workspace / gate_cli.LOCK_FILENAME).open("a+") as held:
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cluster.stops.append(_serving(workspace / "spool", host="nodeA", devices=FOUR))
        assert gate() == 0, capsys.readouterr().err
    assert "a run of this workspace is going; its pool serves this check" in capsys.readouterr().err
    # Nothing was asked of the scheduler, and nothing of the run's pool was ended.
    assert cluster.fake.calls("sbatch") == [] and cluster.fake.calls("scancel") == []
    assert not (workspace / "spool" / STOP_MARKER).exists()


def test_a_holder_that_was_there_before_the_check_serves_it_and_stays(by_hand, capsys):
    from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import load_and_validate_task_yaml

    gate, cluster, workspace = by_hand
    # Somebody started the workspace's pool and left (a holder of the spec's whole time).
    assert gate(max_node_hours=0.1) == 2  # writes the spec; asks for nothing
    spec = load_and_validate_task_yaml(workspace / "task.yaml")
    spec["environment"]["slurm"]["max_node_hours"] = 48
    pool_mod.resolve_spool(spec, workspace)
    pool_mod.build(spec, workspace=workspace).start({1: 600.0})
    deadline = time.time() + 30
    while not cluster.stops and time.time() < deadline:
        time.sleep(0.02)  # the stand-in has started it: it no longer rewrites what is listed
    capsys.readouterr()
    assert gate(max_node_hours=48) == 0, capsys.readouterr().err
    assert "holders that were there before this check serve it and stay" in capsys.readouterr().err
    # It asked for none of its own, and ended none.
    assert len(cluster.fake.calls("sbatch")) == 1 and cluster.fake.calls("scancel") == []
    assert cluster.fake.jobs[0]["state"] == "RUNNING"
    # A scheduler that does not answer is not one with no holder: such a check took the
    # standing holder for its own and cancelled it when it was done. It asks and ends nothing.
    cluster.fake.set("squeue_fail")
    assert gate(max_node_hours=48) == 2
    assert "did not say which holders this workspace has" in capsys.readouterr().err
    assert len(cluster.fake.calls("sbatch")) == 1 and cluster.fake.calls("scancel") == []
    assert cluster.fake.jobs[0]["state"] == "RUNNING"


def test_checks_by_hand_on_one_workspace_keep_its_pool_one_at_a_time(by_hand, capsys):
    gate, cluster, workspace = by_hand
    codes, said = [], ""
    with (workspace / gate_cli.TURN_FILENAME).open("a+") as other:
        fcntl.flock(other, fcntl.LOCK_EX)  # another check by hand keeps the pool
        second = threading.Thread(target=lambda: codes.append(gate()))
        second.start()
        deadline = time.time() + 30
        while "waiting for it" not in said and time.time() < deadline:
            said += capsys.readouterr().err
            time.sleep(0.05)
        # It waits, and asks for nothing of its own meanwhile.
        assert "another check by hand keeps this workspace's pool; waiting for it" in said
        assert second.is_alive() and cluster.fake.calls("sbatch") == []
    second.join(60)
    assert codes == [0] and len(cluster.fake.calls("sbatch")) == 1


def test_a_job_no_holder_of_the_spec_could_start_ends_the_check_before_it_asks(by_hand, capsys):
    gate, cluster, _ = by_hand
    codes: list[int] = []
    # A session of 12 minutes in holders of 15: its wall and the pool's 7 minutes do not fit.
    asking = threading.Thread(target=lambda: codes.append(gate(time="00:15:00")), daemon=True)
    asking.start()
    asking.join(30)
    said = capsys.readouterr().err
    assert codes == [2], said
    assert "states a wall of 12m" in said and "lasts 15m" in said
    assert "it could never start. No holder was asked for" in said
    assert cluster.fake.calls("sbatch") == []


def test_the_bound_of_node_hours_holds_a_check_by_hand_as_it_holds_a_run(by_hand, capsys):
    gate, cluster, _ = by_hand
    # One holder of 22 minutes is 0.4 node-hours: past a bound of 0.1, none is asked for.
    assert gate(max_node_hours=0.1) == 2
    said = capsys.readouterr().err
    assert "error: " in said and "would make 0.4" in said and "max_node_hours" in said
    assert cluster.fake.calls("sbatch") == []
