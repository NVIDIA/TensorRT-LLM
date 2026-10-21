# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Managed sessions: the run submits a session nobody watches."""

from __future__ import annotations

from agent_flow.workflows.trtllm_modeling_v2.infra.substrate import managed


def _task(tmp_path):
    return {
        "trtllm_repo_path": str(tmp_path / "repo"),
        "environment": {
            "kind": "session_spool",
            "python": "/venv/bin/python",
            "spool": str(tmp_path / "spool-a"),
            "container_image": "/images/trtllm.sqsh",
            "mounts": ["/scratch:/scratch"],
            "sessions": [
                {
                    "name": "a",
                    "spool": str(tmp_path / "spool-a"),
                    "role": "gates",
                    "gpus_per_node": 4,
                    "time_limit": "04:00:00",
                    "partition": "batch",
                }
            ],
        },
    }


class _Slurm:
    """sbatch hands out job ids; squeue lists what it was told is alive."""

    def __init__(self):
        self.next_id = 100
        self.alive: dict[str, str] = {}
        self.calls: list[list[str]] = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        if argv[0] == "sbatch":
            self.next_id += 1
            self.alive[str(self.next_id)] = "PENDING"
            return 0, f"{self.next_id};cluster\n"
        if argv[0] == "squeue":
            return 0, "".join(f"{j} {state}\n" for j, state in self.alive.items())
        return 1, "unknown"


def test_a_session_nobody_watches_is_submitted_once(tmp_path):
    slurm = _Slurm()
    sessions = managed.ManagedSessions(
        _task(tmp_path), tmp_path / "ws", run=slurm, which=lambda tool: f"/usr/bin/{tool}"
    )
    (said,) = sessions.tick()
    assert said == "session a: no watcher and no job in the queue: submitted job 101"
    assert (tmp_path / "spool-a" / "session_entry.sh").is_file()
    assert sessions.submitted()[0]["job_id"] == "101"
    # Its job still waits in the queue: nothing more is submitted.
    assert sessions.tick() == []
    assert [c[0] for c in slurm.calls].count("sbatch") == 1
    # Once it is gone with no watcher, the session is submitted again.
    slurm.alive.clear()
    assert sessions.tick() == ["session a: no watcher and no job in the queue: submitted job 102"]
