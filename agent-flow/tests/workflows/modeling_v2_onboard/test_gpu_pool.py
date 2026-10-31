# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The run's GPU pool: its configuration, holder jobs and ``gpu_run``."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from agent_flow.workflows.modeling_v2_onboard import gpu_run
from agent_flow.workflows.modeling_v2_onboard.gpu_pool import GpuPool, PoolConfigError, pool_config
from agent_flow.workflows.modeling_v2_onboard.gpu_run import Waiter, place


def _task(**pool) -> dict:
    return {
        "slurm-environment": {
            "slurm_partition": "batch",
            "docker_image": "/images/trtllm.sqsh",
            "gpu_pool": {"nodes": 2, "account": "acct", **pool},
        }
    }


def test_no_pool_unless_task_yaml_asks_for_one():
    assert pool_config({}) is None
    assert pool_config({"slurm-environment": {"slurm_partition": "batch"}}) is None
    assert pool_config(_task(nodes=0)) is None


def test_pool_config_defaults_to_the_slurm_environment():
    config = pool_config(_task(qos="interactive"))
    assert (config.nodes, config.gpus_per_node, config.reserve) == (2, 4, 2)
    assert (config.partition, config.qos, config.hours) == ("batch", "interactive", 24.0)
    assert config.container_image == "/images/trtllm.sqsh"
    assert config.max_nodes is None
    assert pool_config(_task(max_nodes=6)).max_nodes == 6


def test_pool_config_lists_every_problem():
    with pytest.raises(PoolConfigError) as exc:
        pool_config(_task(nodes=-1, account=None, hours=0, extra=1))
    message = str(exc.value)
    for problem in ("nodes", "account", "hours", "`gpu_pool.extra`"):
        assert problem in message
    with pytest.raises(PoolConfigError) as exc:
        pool_config(_task(max_nodes=1))
    assert "`gpu_pool.max_nodes` must be an integer of at least 2" in str(exc.value)


class _Slurm:
    """Answers squeue with ``jobs`` holder jobs and records every command."""

    def __init__(
        self, jobs: int = 0, squeue_ok: bool = True, sbatch_ok: bool = True, running: int = 0
    ) -> None:
        self.jobs, self.squeue_ok, self.sbatch_ok = jobs, squeue_ok, sbatch_ok
        self.running = running
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        if args[0] == "squeue":
            out = "".join(
                f"{100 + i} {'R' if i < self.running else 'PD'}\n" for i in range(self.jobs)
            )
            return subprocess.CompletedProcess(args, 0 if self.squeue_ok else 1, out, "")
        if args[0] == "sbatch":
            if not self.sbatch_ok:
                return subprocess.CompletedProcess(args, 1, "", "error: QOSMaxSubmitJobPerUser\n")
            self.jobs += 1
            return subprocess.CompletedProcess(
                args, 0, f"Submitted batch job {99 + self.jobs}\n", ""
            )
        return subprocess.CompletedProcess(args, 0, "", "")


def _pool(tmp_path: Path, slurm: _Slurm, **config) -> GpuPool:
    return GpuPool(
        directory=tmp_path / "ws" / "gpu_pool",
        config=pool_config(_task(**config)),
        workspace=tmp_path / "ws",
        mounts=(tmp_path / "ws", tmp_path / "repo"),
        run=slurm,
    )


def test_the_holder_job_holds_one_whole_node(tmp_path):
    pool = _pool(tmp_path, _Slurm(), qos="interactive", hours=1.5, container_mounts=["/data"])
    pool.write()
    script = pool.holder_path.read_text()
    for line in (
        f"#SBATCH --job-name={pool.job_name}",
        "#SBATCH --account=acct",
        "#SBATCH --partition=batch",
        "#SBATCH --qos=interactive",
        "#SBATCH --gres=gpu:4",
        "#SBATCH --time=1:30:00",
        "exec sleep infinity",
    ):
        assert line in script
    data = json.loads(pool.config_path.read_text())
    assert data["job_name"] == pool.job_name
    assert data["container_mounts"] == ["/data", str(tmp_path / "ws"), str(tmp_path / "repo")]
    assert data["priorities"] == str(tmp_path / "ws" / "gpu_priorities.json")
    # Each workspace has its own pool.
    other = GpuPool(
        directory=tmp_path / "b",
        config=pool.config,
        workspace=tmp_path / "b",
        mounts=(),
    )
    assert other.job_name != pool.job_name


def test_the_keeper_submits_holder_jobs_up_to_nodes_and_follows_a_grown_pool(tmp_path):
    slurm = _Slurm()
    pool = _pool(tmp_path, slurm)
    pool.write()
    assert pool.top_up() == "Submitted batch job 100; Submitted batch job 101"
    assert pool.top_up() is None
    pool.nodes = 3
    assert pool.top_up() == "Submitted batch job 102"
    assert [c[0] for c in slurm.calls].count("sbatch") == 3
    pool.release()
    assert slurm.calls[-1][0] == "scancel" and pool.job_name in slurm.calls[-1]


def test_the_keeper_reports_a_refused_submission_and_skips_when_slurm_is_down(tmp_path):
    assert "QOSMaxSubmitJobPerUser" in _pool(tmp_path, _Slurm(sbatch_ok=False)).top_up()
    slurm = _Slurm(squeue_ok=False)
    assert _pool(tmp_path, slurm).top_up() is None
    assert [c[0] for c in slurm.calls] == ["squeue"]


def test_waiters_are_served_by_class_then_by_arrival():
    early_accuracy = Waiter(pid=1, rank=0, since=2.0, gpus=2)
    me = Waiter(pid=2, rank=1, since=1.0, gpus=2)
    # An accuracy waiter that came later is still ahead of a feature command.
    assert place(me, [early_accuracy], {"a": 3}, perf=False, reserve=0) is None
    # Enough GPUs for both: no one waits.
    assert place(me, [early_accuracy], {"a": 4}, perf=False, reserve=0) == "a"
    # Perf work leaves the last `reserve` GPUs alone.
    assert place(me, [], {"a": 3}, perf=True, reserve=2) is None
    assert place(me, [], {"a": 4}, perf=True, reserve=2) == "a"


def test_a_whole_node_waiter_keeps_the_node_it_waits_for():
    node_job = Waiter(pid=1, rank=0, since=1.0, gpus=4)
    one_gpu = Waiter(pid=2, rank=1, since=2.0, gpus=1)
    # Five GPUs free, but no node has four: the 4-GPU command claims the
    # emptiest node, and a 1-GPU command may only use the other one.
    free = {"a": 3, "b": 2}
    assert place(one_gpu, [node_job], free, perf=False, reserve=0) == "b"
    assert place(one_gpu, [node_job], {"a": 3, "b": 0}, perf=False, reserve=0) is None
    # Once a node is whole, the 4-GPU command takes it.
    assert place(node_job, [one_gpu], {"a": 4, "b": 2}, perf=False, reserve=0) == "a"


def test_a_command_goes_to_the_fullest_node_it_fits_and_only_where_time_is_left():
    me = Waiter(pid=1, rank=1, since=1.0, gpus=1)
    free = {"a": 4, "b": 1}
    assert place(me, [], free, perf=False, reserve=0) == "b"
    assert place(me, [], free, perf=False, reserve=0, usable={"a"}) == "a"


def test_the_pool_status_lists_nodes_and_waiters(tmp_path):
    pool = _pool(tmp_path, _Slurm(jobs=2, running=1))
    pool.write()
    ticket = {"pid": os.getpid(), "rank": 0, "since": time.time() - 600, "gpus": 4}
    (pool.directory / "queue" / "1.json").write_text(
        json.dumps({**ticket, "task": "mtp-tp4", "class": "accuracy"})
    )
    status = pool.status()
    assert "2 node(s) of 4 GPUs wanted, 1 running, 1 queued in Slurm" in status
    assert "free GPUs per running node: [4]" in status
    assert "`mtp-tp4` (accuracy) for 4 GPU(s), 10 min" in status


def test_gpu_run_runs_the_command_in_the_worktree_on_locked_gpus(tmp_path, monkeypatch):
    pool = _pool(tmp_path, _Slurm())
    pool.write()
    (tmp_path / "ws" / "gpu_priorities.json").write_text(json.dumps({"op-a": "accuracy"}))
    monkeypatch.setattr(gpu_run, "_holder_jobs", lambda name: [("4242", 10**6)])
    launched: list[list[str]] = []

    class _Proc:
        def __init__(self, args):
            launched.append(args)
            # The command's GPUs stay locked while it runs.
            locks = pool.directory / "locks"
            assert gpu_run.free_gpus(locks, "4242", 4) == 2

        def wait(self):
            return 7

    monkeypatch.setattr(gpu_run.subprocess, "Popen", _Proc)
    monkeypatch.setattr(gpu_run.signal, "signal", lambda *args: None)
    worktree = tmp_path / "ws" / "worktrees" / "op-a"
    worktree.mkdir(parents=True)
    argv = ["--pool", str(pool.directory), "--task", "op-a", "--gpus", "2", str(worktree)]
    assert gpu_run.main([*argv, "pytest", "-x"]) == 7

    (srun,) = launched
    assert srun[:3] == ["srun", "--jobid=4242", "--overlap"]
    assert "--container-image=/images/trtllm.sqsh" in srun
    assert f"--container-workdir={worktree}" in srun
    assert "CUDA_VISIBLE_DEVICES=0,1" in srun
    assert srun[-4:] == ["_", str(worktree), "pytest", "-x"]
    assert gpu_run.free_gpus(pool.directory / "locks", "4242", 4) == 4
