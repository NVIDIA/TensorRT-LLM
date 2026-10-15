# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""`--wait-for-spool`: the orchestrator can start while its compute session is still coming up."""

import threading

import yaml

from agent_flow.workflows.trtllm_modeling_v2.cli import _wait_for_spool


def _task(tmp_path, spool):
    task = tmp_path / "task.yaml"
    task.write_text(yaml.safe_dump({"environment": {"kind": "session_spool", "spool": str(spool)}}))
    return task


def test_the_wait_ends_when_the_spool_appears(tmp_path):
    spool = tmp_path / "spool" / "f1"
    timer = threading.Timer(0.2, lambda: spool.mkdir(parents=True))
    timer.start()
    assert _wait_for_spool(_task(tmp_path, spool), timeout_s=5, poll_s=0.05)
    timer.join()


def test_the_wait_gives_up_and_leaves_the_verdict_to_validation(tmp_path):
    assert not _wait_for_spool(_task(tmp_path, tmp_path / "never"), timeout_s=0.2, poll_s=0.05)


def test_other_environments_do_not_wait(tmp_path):
    task = tmp_path / "task.yaml"
    task.write_text(yaml.safe_dump({"environment": {"kind": "slurm"}}))
    assert _wait_for_spool(task, timeout_s=0.1, poll_s=0.05)
