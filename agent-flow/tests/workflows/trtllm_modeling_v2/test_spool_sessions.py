# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Sessions declared in task.yaml: rendered entries, hand-over, and the multi-node agent.

A hand-edited entry script can come up on the wrong spool and read as a dead
session. An allocation that reaches its limit takes the orchestrator and any
in-flight gate with it unless a successor is planned.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2 import cli as cli_mod
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import resolve_environment
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate import (
    SCHEDULER_PATH,
    SCRIPTS_DIR,
    SUBSTRATE_DIR,
    sessions,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.substrate.scheduler import (
    Scheduler,
    SchedulerConfig,
    SpoolPaths,
    allocation_end,
    build_parser,
    write_json_atomic,
)
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import (
    TaskSchemaError,
    load_and_validate_task_yaml,
)

_SCRATCH = "/scratch/mv2flow-run-tmp"


def _environment(tmp_path, **extra):
    return {
        "kind": "session_spool",
        "python": "/opt/venv/bin/python",
        "spool": str(tmp_path / "spool-a"),
        "env": {"TMPDIR": _SCRATCH, "PATH": "/opt/venv/bin"},
        "container_image": "/images/trtllm.sqsh",
        "partition": "batch",
        "account": "acct",
        "mounts": ["/shared:/shared"],
        "sessions": [
            {
                "name": "a",
                "spool": str(tmp_path / "spool-a"),
                "role": "gates",
                "gpus_per_node": 4,
                "time_limit": "24:00:00",
                "handover_lead": "2h",
                "scheduler": {"default_wall": "3h", "pressure_floor_gb": 32},
                "setup": ["echo custom-setup-line"],
            },
            {"name": "b", "spool": str(tmp_path / "spool-b"), "role": "work"},
            {"name": "mn", "spool": str(tmp_path / "spool-mn"), "nodes": 2, "gpus_per_node": 4},
        ],
        **extra,
    }


def _task(tmp_path, **extra):
    return {
        "trtllm_repo_path": str(tmp_path / "repo"),
        "environment": _environment(tmp_path, **extra),
    }


def _errors(block) -> list[str]:
    errors: list[str] = []
    sessions.validate_sessions(block, errors)
    return errors


# ----------------------------------------------------------------- the schema


def test_a_well_formed_session_block_validates(tmp_path):
    assert _errors(_environment(tmp_path)) == []


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda s: s.__setitem__(1, {**s[1], "name": "a"}), "declared twice"),
        (lambda s: s.__setitem__(1, {**s[1], "spool": s[0]["spool"]}), "used by two sessions"),
        (lambda s: s.__setitem__(1, {**s[1], "spool": "relative/spool"}), "absolute path"),
        (lambda s: s.__setitem__(1, {**s[1], "role": "gate"}), "role"),
        (lambda s: s.__setitem__(1, {**s[1], "spoool": "/x"}), "spoool"),
        (lambda s: s.__setitem__(0, {**s[0], "time_limit": "forever"}), "duration"),
        (lambda s: s.__setitem__(0, {**s[0], "scheduler": {"default_wal": "1h"}}), "default_wal"),
        (lambda s: s.__setitem__(2, {**s[2], "nodes": 0}), "nodes"),
    ],
)
def test_a_malformed_session_is_rejected_with_its_reason(tmp_path, mutate, fragment):
    block = _environment(tmp_path)
    mutate(block["sessions"])
    assert any(fragment in error for error in _errors(block)), _errors(block)


def test_the_wait_for_a_lost_session_is_a_duration_with_a_default(tmp_path):
    assert sessions.session_wait_from_task({}) == sessions.DEFAULT_SESSION_WAIT_S
    task = _task(tmp_path, session_wait="90m")
    assert _errors(task["environment"]) == []
    assert sessions.session_wait_from_task(task) == 5400
    assert any(
        "session_wait" in e and "duration" in e
        for e in _errors(_environment(tmp_path, session_wait="until it is back"))
    )


def test_the_task_schema_validates_sessions(tmp_path, repo, checkpoint):  # noqa: F811
    env = _environment(tmp_path)
    (tmp_path / "spool-a").mkdir()
    spec = {
        "checkpoint_path": str(checkpoint),
        "arch": "sm_103",
        "parallel": "tp1",
        "trtllm_repo_path": str(repo),
        "target": {"family": "gpt_oss", "checkpoint_segment": "gpt_oss_120b"},
        "environment": env,
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": "tests/integration/defs/accuracy/references/gsm8k.yaml",
            "model_key": "openai/gpt-oss-120b",
            "tolerance": 5.0,
        },
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    load_and_validate_task_yaml(path)
    env["sessions"][1]["role"] = "nope"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    with pytest.raises(TaskSchemaError, match="sessions\\[1\\].role"):
        load_and_validate_task_yaml(path)


def test_agents_are_told_which_session_is_for_what(tmp_path):
    env = resolve_environment({"environment": _environment(tmp_path)})
    assert [name for name, _, _ in env.sessions] == ["a", "b", "mn"]
    block = env.describe()
    assert "`b` — catalog tests" in block and str(tmp_path / "spool-b") in block


def test_a_bare_spool_is_one_implicit_session(tmp_path):
    task = {"environment": {"kind": "session_spool", "python": "p", "spool": "/s/spool"}}
    (only,) = sessions.sessions_from_task(task)
    assert (only.name, only.spool) == ("default", "/s/spool")


# ------------------------------------------------------------------ rendering


def test_the_entry_is_rendered_from_config_with_no_path_of_its_own(tmp_path):
    text = sessions.render_session_entry(_task(tmp_path), "a")
    assert f"BASE_SPOOL={tmp_path / 'spool-a'}" in text
    assert 'export PATH=/opt/venv/bin:"$PATH"' in text
    assert f"export MV2FLOW_REPO={tmp_path / 'repo'}" in text
    assert "export MV2FLOW_SESSION_SPOOLS=" in text and str(tmp_path / "spool-b") in text
    assert "echo custom-setup-line" in text
    assert str(SCHEDULER_PATH) in text
    for flag in (
        "--gpus 4",
        "--time-limit 86400",
        "--handover-lead 7200",
        "--default-wall 3h",
        "--pressure-floor-gb 32",
    ):
        assert flag in text
    assert "ulimit -c 0" in text
    subprocess.run(["bash", "-n"], input=text, text=True, check=True)


def _run_entry(tmp_path, name, node="0"):
    task = _task(tmp_path)
    entry = sessions.install_session_entry(task, name)
    stub = tmp_path / "stub-python"
    capture = tmp_path / f"captured-{name}-{node}.json"
    stub.write_text(
        "#!/bin/bash\n"
        f'python3 -c \'import json, os, sys; json.dump({{"argv": sys.argv[1:], "env": dict(os.environ)}}, '
        f'open("{capture}", "w"))\' "$@"\n'
    )
    stub.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env.update({"MV2FLOW_SCHEDULER_PYTHON": str(stub), "SLURM_NODEID": node})
    done = subprocess.run(
        ["bash", str(entry)], env=env, capture_output=True, text=True, timeout=120
    )
    assert done.returncode == 0, done.stderr
    return entry, json.loads(capture.read_text())


def test_the_rendered_entry_becomes_the_scheduler(tmp_path):
    entry, seen = _run_entry(tmp_path, "a")
    assert entry == tmp_path / "spool-a" / "session_entry.sh" and os.access(entry, os.X_OK)
    argv = seen["argv"]
    assert argv[:3] == [str(SCHEDULER_PATH), "watch", str(tmp_path / "spool-a")]
    assert argv[argv.index("--entry") + 1] == str(entry)
    assert argv[argv.index("--session") + 1] == "a"
    assert seen["env"]["MV2FLOW_SPOOL"] == str(tmp_path / "spool-a")
    assert seen["env"]["PATH"].startswith("/opt/venv/bin:")
    assert (tmp_path / "spool-a" / "session_entry.log").is_file()


def test_each_node_of_a_multi_node_session_drains_its_own_spool(tmp_path):
    _, seen = _run_entry(tmp_path, "mn", node="1")
    assert seen["argv"][2] == str(tmp_path / "spool-mn-n1")
    assert seen["argv"][seen["argv"].index("--session") + 1] == "mn-n1"
    hosts = list((tmp_path / "spool-mn" / "hosts").iterdir())
    assert [h.read_text().strip() for h in hosts] == [str(tmp_path / "spool-mn-n1")]


def test_the_cli_renders_a_session(tmp_path):
    task_path = tmp_path / "task.yaml"
    task_path.write_text(yaml.safe_dump(_task(tmp_path)), encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        cli_mod.main(["session", "render", "--task", str(task_path), "--name", "b"])
    assert exit_info.value.code == 0
    assert (tmp_path / "spool-b" / "session_entry.sh").is_file()


def test_the_cli_refuses_an_unknown_session(tmp_path, capsys):
    task_path = tmp_path / "task.yaml"
    task_path.write_text(yaml.safe_dump(_task(tmp_path)), encoding="utf-8")
    with pytest.raises(SystemExit) as exit_info:
        cli_mod.main(["session", "render", "--task", str(task_path), "--name", "zz"])
    assert exit_info.value.code == 2
    assert "no session named 'zz'" in capsys.readouterr().err


def test_the_successor_submission_is_rendered_for_the_driver(tmp_path):
    call = sessions.broker_submission(_task(tmp_path), "mn", after_job="500296")
    assert call.startswith("mcp__slurm-broker__slurm_submit(")
    for fragment in (
        'image="/images/trtllm.sqsh"',
        "nodes=2",
        "segment=2",
        "gpus_per_node=4",
        'dependency="afterany:500296"',
        '"/shared:/shared"',
        'job_name="mv2flow-mn"',
        str(tmp_path / "spool-mn" / "session_entry.sh"),
    ):
        assert fragment in call


# ------------------------------------------------------------------ hand-over


def _sched(spool, *, ends_in, watcher_id="w1", lead=3600.0) -> Scheduler:
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    env["SLURM_JOB_ID"] = "1001"
    config = SchedulerConfig(
        devices=("0", "1"),
        session="a",
        ends_at=time.time() + ends_in,
        handover_lead_s=lead,
        poll_s=0.02,
        leftover_grace_s=0,
    )
    return Scheduler(spool, config, environ=env, watcher_id=watcher_id)


def _queue(spool, name, header):
    (SpoolPaths(spool).pending / f"{name}.sh").write_text(f"#!/bin/bash\n#spool: {header}\ntrue\n")


def test_near_its_end_a_session_requests_a_successor_once(tmp_path):
    spool = tmp_path / "spool"
    sched = _sched(spool, ends_in=600)
    sched.tick()
    request = json.loads((spool / "handover" / "request.json").read_text())
    assert request["session"] == "a" and request["slurm_job"] == "1001"
    sched.tick()
    assert (
        json.loads((spool / "handover" / "request.json").read_text())["request_id"]
        == request["request_id"]
    )
    snapshot = json.loads((spool / "status.json").read_text())
    assert (
        snapshot["remaining_s"] < 601
        and snapshot["handover"]["request_id"] == request["request_id"]
    )


def test_a_session_far_from_its_end_requests_nothing(tmp_path):
    spool = tmp_path / "spool"
    _sched(spool, ends_in=86400).tick()
    assert not (spool / "handover" / "request.json").exists()


def test_a_job_that_cannot_finish_in_time_waits_for_the_successor(tmp_path):
    """An allocation's time limit would otherwise kill a long gate partway through."""
    spool = tmp_path / "spool"
    sched = _sched(spool, ends_in=600)
    _queue(spool, "long-gate", "gpus=all wall=2h")
    _queue(spool, "short-test", "gpus=1 wall=60")
    _queue(spool, "free-note", "gpus=0")
    sched.tick()
    assert set(sched.jobs) == {"short-test", "free-note"}
    reasons = {p["name"]: p["reason"] for p in sched.snapshot(time.time())["pending"]}
    assert "waits for the successor" in reasons["long-gate"]
    end = time.monotonic() + 10
    while sched.jobs and time.monotonic() < end:
        sched.tick()
        time.sleep(0.02)

    # The successor comes up on the same spool and takes the job.
    successor = _sched(spool, ends_in=86400, watcher_id="w2")
    successor.tick()
    assert "long-gate" in successor.jobs
    end = time.monotonic() + 10
    while successor.jobs and time.monotonic() < end:
        successor.tick()
        time.sleep(0.02)


def test_once_the_successor_is_up_the_old_session_admits_nothing_and_retires(tmp_path):
    spool = tmp_path / "spool"
    old = _sched(spool, ends_in=600)
    write_json_atomic(
        SpoolPaths(spool).watchers / "w2.json",
        {"watcher": "w2", "alive_epoch": time.time() + 3600, "ends_at_epoch": time.time() + 86400},
    )
    _queue(spool, "free-anything", "gpus=0")
    assert old.serve() == 0
    assert old.handover["successor"] == "w2"
    assert (SpoolPaths(spool).pending / "free-anything.sh").exists()
    assert "handed over to w2" in SpoolPaths(spool).status.read_text()


def test_the_allocation_end_prefers_slurms_own_figure():
    args = build_parser().parse_args(["watch", "/s", "--time-limit", "1h"])
    assert allocation_end(args, {"SLURM_JOB_END_TIME": "2000000000"}) == 2e9
    assert allocation_end(args, {"SLURM_JOB_START_TIME": "1000"}) == 1000 + 3600
    assert allocation_end(build_parser().parse_args(["watch", "/s"]), {}) is None


def test_requests_are_listed_until_acknowledged(tmp_path):
    task = _task(tmp_path)
    _sched(Path(task["environment"]["sessions"][0]["spool"]), ends_in=60).tick()
    (pending,) = sessions.pending_requests(task)
    assert pending["session"] == "a"
    sessions.acknowledge(pending["spool"], job_id="7777", by="test")
    assert sessions.pending_requests(task) == []


def test_a_successor_request_becomes_a_broker_call_for_the_driver(tmp_path):
    task = _task(tmp_path)
    _sched(Path(task["environment"]["sessions"][0]["spool"]), ends_in=60).tick()
    submission = sessions.allocate(task, "a")
    assert 'dependency="afterany:1001"' in submission
    assert sessions.pending_requests(task), "the driver acknowledges it once it has submitted"


def test_session_status_reads_every_spool(tmp_path):
    task = _task(tmp_path)
    _sched(Path(task["environment"]["sessions"][0]["spool"]), ends_in=7200).tick()
    rows = {row["name"]: row for row in sessions.session_status(task)}
    assert rows["a"]["heartbeat_age_s"] is not None and rows["a"]["slurm_job"] == "1001"
    assert rows["b"]["heartbeat_age_s"] is None


# ------------------------------------------------------- the multi-node agent


def test_the_node_agent_runs_the_daemon_through_the_remote_nodes_spool(tmp_path):
    base, remote = tmp_path / "spool-mn", tmp_path / "spool-mn-n1"
    (base / "hosts").mkdir(parents=True)
    (base / "hosts" / "node1").write_text(str(remote))
    env = {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}
    sched = Scheduler(remote, SchedulerConfig(devices=("0",), leftover_grace_s=0), environ=env)
    stop = threading.Event()

    def loop():
        while not stop.is_set():
            sched.tick()
            time.sleep(0.02)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    try:
        done = subprocess.run(
            [
                "bash",
                str(SCRIPTS_DIR / "mn" / "mn_node_agent.sh"),
                "node1.cluster",
                "echo prted-ran; exit 3",
            ],
            env={**env, "MV2FLOW_MN_BASE_SPOOL": str(base)},
            capture_output=True,
            text=True,
            timeout=60,
        )
    finally:
        stop.set()
        thread.join(timeout=5)
    assert done.returncode == 3
    (log,) = (remote / "logs").glob("free-prted-*.log")
    assert "prted-ran" in log.read_text()
    record = json.loads(next((remote / "done").glob("free-prted-*.json")).read_text())
    assert record["gpus"] == "0"


def test_no_substrate_file_names_a_machine_specific_path():
    for path in [*SCRIPTS_DIR.rglob("*"), *SUBSTRATE_DIR.rglob("*.py")]:
        if path.is_file() and "__pycache__" not in path.parts:
            text = path.read_text(encoding="utf-8")
            for forbidden in ("/scratch/", "/lustre/", "/home/"):
                assert forbidden not in text, f"{path} names {forbidden}"
