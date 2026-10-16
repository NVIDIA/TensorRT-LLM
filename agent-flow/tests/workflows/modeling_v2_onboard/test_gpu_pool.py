# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The run's GPU pool: its configuration, holder jobs and ``gpu_run``."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from agent_flow.workflows.modeling_v2_onboard import gpu_run
from agent_flow.workflows.modeling_v2_onboard.gpu_pool import GpuPool, PoolConfigError, pool_config
from agent_flow.workflows.modeling_v2_onboard.gpu_run import Waiter, allowed


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


def test_pool_config_lists_every_problem():
    with pytest.raises(PoolConfigError) as exc:
        pool_config(_task(nodes=-1, account=None, hours=0, extra=1))
    message = str(exc.value)
    for problem in ("nodes", "account", "hours", "`gpu_pool.extra`"):
        assert problem in message


class _Slurm:
    """Answers squeue with ``jobs`` holder jobs and records every command."""

    def __init__(self, jobs: int = 0, squeue_ok: bool = True, sbatch_ok: bool = True) -> None:
        self.jobs, self.squeue_ok, self.sbatch_ok = jobs, squeue_ok, sbatch_ok
        self.calls: list[list[str]] = []

    def __call__(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        self.calls.append(args)
        if args[0] == "squeue":
            out = "".join(f"{100 + i}\n" for i in range(self.jobs))
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


def test_the_keeper_submits_one_holder_job_at_a_time_up_to_nodes(tmp_path):
    slurm = _Slurm()
    pool = _pool(tmp_path, slurm)
    pool.write()
    assert pool.top_up() == "Submitted batch job 100"
    assert pool.top_up() == "Submitted batch job 101"
    assert pool.top_up() is None
    assert [c[0] for c in slurm.calls].count("sbatch") == 2
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
    assert not allowed(me, [early_accuracy], 3, perf=False, reserve=0)
    # Enough GPUs for both: no one waits.
    assert allowed(me, [early_accuracy], 4, perf=False, reserve=0)
    # Perf work leaves the last `reserve` GPUs alone.
    assert not allowed(me, [], 3, perf=True, reserve=2)
    assert allowed(me, [], 4, perf=True, reserve=2)


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
            assert gpu_run._free(locks, "4242", 4) == 2

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
    assert gpu_run._free(pool.directory / "locks", "4242", 4) == 4
