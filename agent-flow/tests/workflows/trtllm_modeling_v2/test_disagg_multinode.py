# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native multi-node gates require each node's result and one allocation's gang."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time

import pytest

from agent_flow.workflows.trtllm_modeling_v2.checks import disagg
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import (
    CommandResult,
    CommandSpec,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.executor import (
    SpoolCommandExecutor,
    select_command_executor,
)
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.scheduler import SpoolPaths
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import _validate_disagg

from .test_spool_gangs import _node, _pump

NODE = "tests/integration/defs/accuracy/test_disaggregated_multinode.py::TestStep5::test_dep4"
PASS = '<testsuite><testcase name="native_hop"/></testsuite>'


def test_native_command_expands_the_node_report_inside_the_python_launcher(tmp_path):
    repo = tmp_path / "repo with spaces"
    repo.mkdir()
    captured = tmp_path / "arguments.json"
    python = tmp_path / "native python"
    python.write_text(
        "#!/usr/bin/python3\nimport json,sys\njson.dump(sys.argv[1:],open("
        + repr(str(captured))
        + ',"w"))\n'
    )
    python.chmod(0o700)
    environment = RunEnvironment(kind="session_spool", python=str(python), world_size=4)
    spec = disagg.DisaggSpec(test=NODE, devices=4, nodes=2)
    report = tmp_path / "report with spaces.xml"
    command = disagg.disagg_command(environment, repo, spec, report)
    env = {**os.environ, "MV2FLOW_JOB_NODES": "2", "MV2FLOW_JOB_NODE_RANK": "1"}
    subprocess.run(command, shell=True, env=env, check=True)
    args = json.loads(captured.read_text())
    assert f"--junitxml={report.with_suffix('')}.node1.xml" in args
    assert "-m" in args and "pytest" in args
    from agent_flow.workflows.trtllm_modeling_v2.checks.check_runtime import RANK_HOOK_DIR

    assert "MV2FLOW_RANK_FACTS=1" in command
    assert "TRTLLM_MODELING_V2=require" in command
    assert str(RANK_HOOK_DIR) in command
    env["MV2FLOW_JOB_NODES"] = "1"
    assert subprocess.run(command, shell=True, env=env).returncode != 0


@pytest.mark.parametrize(
    "second,code,expected",
    [
        (PASS, 0, True),
        ("<testsuite><testcase><skipped/></testcase></testsuite>", 0, False),
        ("<testsuite><testcase><failure/></testcase></testsuite>", 0, False),
        (None, 0, False),
        ("<broken", 0, False),
        (PASS, 7, False),
    ],
)
def test_every_node_must_have_a_fresh_successful_report(tmp_path, second, code, expected):
    work = tmp_path / "gate"
    work.mkdir()
    paths = [work / f"disagg.disagg.dep4.node{rank}.xml" for rank in range(2)]
    for path in paths:
        path.write_text(PASS)

    def runner(command, env, timeout):
        assert all(not path.exists() for path in paths), "stale reports must be removed"
        paths[0].write_text(PASS)
        if second is not None:
            paths[1].write_text(second)
        return CommandResult(returncode=code)

    result = disagg.run_disagg_gate(
        RunEnvironment(kind="session_spool", python="python3"),
        task_data={"trtllm_repo_path": str(tmp_path)},
        workdir=work,
        runner=runner,
        unit="disagg",
        member="dep4",
        spec=disagg.DisaggSpec(test=NODE, devices=4, nodes=2),
    )
    assert result.passed == expected
    if expected:
        assert "node0" in result.detail and "node1" in result.detail


def test_nonzero_single_node_exit_cannot_pass_from_a_successful_xml(tmp_path):
    def runner(command, env, timeout):
        (tmp_path / "disagg.disagg.dep4.xml").write_text(PASS)
        return CommandResult(returncode=9)

    result = disagg.run_disagg_gate(
        RunEnvironment(kind="local", python="python3"),
        task_data={"trtllm_repo_path": str(tmp_path)},
        workdir=tmp_path,
        runner=runner,
        unit="disagg",
        member="dep4",
        spec=disagg.DisaggSpec(test=NODE),
    )
    assert not result.passed and "exited 9" in result.detail


@pytest.mark.parametrize("nodes", [0, -1, "two", True])
def test_invalid_node_requests_are_rejected(nodes):
    errors = []
    _validate_disagg({"test": NODE, "devices": 4, "nodes": nodes}, errors)
    assert errors


def test_explicit_nodes_reach_both_native_gang_processes(tmp_path):
    spool = tmp_path / "spool"
    SpoolPaths(spool).ensure()
    a, b = _node(spool, time.time, "nodeA"), _node(spool, time.time, "nodeB")
    executor = SpoolCommandExecutor(spool, nodes=2, gpus="4", poll_s=0.01)
    output = tmp_path / "seen"
    command = (
        f'echo "$MV2FLOW_JOB_NODES $MV2FLOW_JOB_NODE_RANK" > "{output}.$MV2FLOW_JOB_NODE_RANK"'
    )
    results = []
    thread = threading.Thread(
        target=lambda: results.append(executor.execute(CommandSpec(command, timeout_s=10)))
    )
    thread.start()
    try:
        _pump([a, b], lambda: bool(results))
    finally:
        thread.join(timeout=5)
    assert not thread.is_alive() and results[0].ok
    assert (tmp_path / "seen.0").read_text() == "2 0\n"
    assert (tmp_path / "seen.1").read_text() == "2 1\n"


def test_a_local_executor_cannot_silently_downgrade_a_multi_node_request():
    with pytest.raises(ValueError, match="gang execution"):
        select_command_executor(RunEnvironment(kind="local", python="python3"), nodes=2)


@pytest.mark.parametrize("available,valid", [(1, False), (2, True)])
def test_preflight_requires_one_allocation_with_enough_nodes(available, valid):
    from agent_flow.workflows.trtllm_modeling_v2.run.startup.preflight import (
        PreflightError,
        check_disagg_placement,
    )

    task = {
        "accuracy": {"disagg": {"test": NODE, "devices": 4, "nodes": 2}},
        "environment": {
            "kind": "session_spool",
            "sessions": [
                {
                    "name": "gates",
                    "role": "gates",
                    "spool": "/spool",
                    "nodes": available,
                    "gpus_per_node": 4,
                }
            ],
        },
    }
    if valid:
        assert "across 2 node(s)" in check_disagg_placement(task)[0]
    else:
        with pytest.raises(PreflightError, match="one session_spool allocation"):
            check_disagg_placement(task)


def test_zero_nodes_cannot_pass_with_an_empty_set_of_reports():
    with pytest.raises(ValueError, match="nodes must be positive"):
        disagg.DisaggSpec(test=NODE, nodes=0)


def test_disagg_observer_reaches_each_native_pytest_process(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.checks.check_runtime import RANK_HOOK_DIR

    captured = tmp_path / "observer.json"
    native = tmp_path / "native.py"
    native.write_text(
        "#!/usr/bin/python3\nimport json,os\njson.dump({k:os.environ.get(k) for k in "
        "['MV2FLOW_RANK_FACTS','TRTLLM_MODELING_V2','TRTLLM_MODELING_V2_VALIDATE','PYTHONPATH']},open("
        + repr(str(captured))
        + ", 'w'))\n"
    )
    native.chmod(0o700)
    env = RunEnvironment(
        kind="session_spool",
        python=str(native),
        world_size=4,
        env={"PYTHONPATH": "/existing/source"},
    )
    spec = disagg.DisaggSpec(test=NODE, devices=4, nodes=2)
    command = disagg.disagg_command(env, tmp_path, spec, tmp_path / "report.xml")
    for rank in [0, 1]:
        subprocess.run(
            command,
            shell=True,
            check=True,
            env={**os.environ, "MV2FLOW_JOB_NODES": "2", "MV2FLOW_JOB_NODE_RANK": str(rank)},
        )
        observed = json.loads(captured.read_text())
        assert observed["MV2FLOW_RANK_FACTS"] == "1"
        assert observed["TRTLLM_MODELING_V2"] == "require"
        assert observed["TRTLLM_MODELING_V2_VALIDATE"] == "1"
        assert observed["PYTHONPATH"].split(os.pathsep) == [str(RANK_HOOK_DIR), "/existing/source"]


@pytest.mark.parametrize("failure", [None, "missing", "skipped", "session"])
def test_split_native_jobs_require_fresh_reports_from_every_node(tmp_path, failure):
    selected = tuple(f"{NODE}[{dtype}]" for dtype in ("auto", "fp8", "nvfp4"))
    spec = disagg.DisaggSpec(test=NODE, tests=selected, devices=4, nodes=2, timeout=17)
    calls = []
    for index in range(3):
        for rank in range(2):
            (tmp_path / f"disagg.disagg.part{index}.dep4.node{rank}.xml").write_text(PASS)

    def runner(command, env, timeout):
        index = len(calls)
        assert timeout == 17
        assert f"[{('auto', 'fp8', 'nvfp4')[index]}]" in command
        paths = [tmp_path / f"disagg.disagg.part{index}.dep4.node{rank}.xml" for rank in range(2)]
        assert all(not path.exists() for path in paths)
        calls.append(command)
        paths[0].write_text(PASS)
        if index == 1 and failure == "session":
            return CommandResult(returncode=-15, infrastructure="allocation disappeared")
        if index == 1 and failure == "skipped":
            paths[1].write_text("<testsuite><testcase><skipped/></testcase></testsuite>")
        elif not (index == 1 and failure == "missing"):
            paths[1].write_text(PASS)
        return CommandResult(returncode=0)

    result = disagg.run_disagg_gate(
        RunEnvironment(kind="session_spool", python="python3"),
        task_data={"trtllm_repo_path": str(tmp_path)},
        workdir=tmp_path,
        runner=runner,
        unit="disagg",
        member="dep4",
        spec=spec,
    )
    assert result.passed is (failure is None)
    assert len(calls) == (3 if failure is None else 2)
    assert result.command == "\n".join(calls)
    assert ("3/3" if failure is None else "2/3") in result.detail
    if failure == "session":
        assert result.infrastructure


@pytest.mark.parametrize(
    "selected,valid",
    [
        ([f"{NODE}[auto]", f"{NODE}[fp8]", f"{NODE}[nvfp4]"], True),
        ([], False),
        (NODE, False),
        ([f"{NODE}[auto]", f"{NODE}[auto]"], False),
        (["tests/other.py::TestOther::test_hop"], False),
        ([False], False),
    ],
)
def test_native_job_splits_are_explicit_unique_refinements(selected, valid):
    block = {"test": NODE, "tests": selected, "devices": 4, "nodes": 2}
    errors = []
    _validate_disagg(block, errors)
    assert (not errors) is valid
    if valid:
        spec = disagg.DisaggSpec.from_task({"accuracy": {"disagg": block}})
        assert spec.tests == tuple(selected)
