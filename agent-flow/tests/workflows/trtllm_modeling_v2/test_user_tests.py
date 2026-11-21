# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The spec's own tests (``verify.tests``): schema, contract, preflight and how they run."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import user_tests
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment
from agent_flow.workflows.trtllm_modeling_v2.run.startup import preflight
from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as contract_mod
from agent_flow.workflows.trtllm_modeling_v2.run.validation.check_runner import CheckRunner
from agent_flow.workflows.trtllm_modeling_v2.spec import task_schema
from agent_flow.workflows.trtllm_modeling_v2.spec import verify as verify_mod

E2E = "tests/x/test_a.py::TestA::test_e2e[{member}]"
ROUTING = "tests/x/test_b.py"
PASS = '<testsuite><testcase classname="TestA" name="test_e2e"/></testsuite>'


def _spec(tmp_path, tests=None, **extra) -> dict:
    checkpoint, repo = tmp_path / "ckpt", tmp_path / "repo"
    checkpoint.mkdir(exist_ok=True)
    repo.mkdir(exist_ok=True)
    spec = {
        "checkpoint_path": str(checkpoint),
        "arch": "sm_103",
        "parallel": "unified",
        "members": ["tep4", "dep4"],
        "trtllm_repo_path": str(repo),
        "environment": {
            "kind": "local",
            "python": "/usr/bin/python3",
            "env": {"TRTLLM_MODELING_V2": "require", "TMPDIR": "/scratch/tmp"},
        },
        "accuracy": {"benchmark": "gsm8k", "reference_yaml": "x.yaml"},
    }
    if tests is not None:
        spec["verify"] = {"tests": tests}
    spec.update(extra)
    return spec


def _load(tmp_path, spec) -> dict:
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return task_schema.load_and_validate_task_yaml(path)


def _tests():
    return {
        "core": {
            "e2e": {
                "purpose": "GSM8K and the reuse panel on one engine",
                "pytest": E2E,
                "members": "all",
                "stage": "delivery",
            },
            "routing": {
                "purpose": "the router picks the reference's experts",
                "pytest": ROUTING,
                "devices": 2,
            },
        }
    }


# ----------------------------------------------------------------------------- the contract


def test_a_spec_test_is_a_check_per_member_at_its_stage(tmp_path):
    data = _load(tmp_path, _spec(tmp_path, _tests()))
    contract = contract_mod.compile_contract(data, world_size=4)
    stages = {c.key: c.stage for c in contract.checks}
    # A whole-model test at every member of its feature, at its stage.
    assert stages["core:test[e2e]@dep4"] == contract_mod.STAGE_DELIVERY
    assert stages["core:test[e2e]@tep4"] == contract_mod.STAGE_DELIVERY
    # A module test once, recorded at its devices, at the default stage.
    assert stages["core:test[routing]@tp2"] == contract_mod.STAGE_LANDING
    # The flow's own checks land; the delivery stage holds the spec's delivery test.
    assert stages["core:accuracy[gsm8k]@tep4"] == contract_mod.STAGE_LANDING
    assert {c.key for c in contract.by_stage(contract_mod.STAGE_DELIVERY)} >= {
        "core:test[e2e]@tep4",
        "core:test[e2e]@dep4",
    }
    assert any("`e2e`" in note and "GSM8K and the reuse panel" in note for note in contract.notes)
    assert "| stage |" in contract.render()
    assert all(c.inputs for c in contract.checks)


def test_a_spec_test_is_owed_whatever_the_profile(tmp_path):
    data = _load(tmp_path, _spec(tmp_path, _tests(), profile="bringup"))
    keys = {c.key for c in contract_mod.compile_contract(data, world_size=4).checks}
    assert {"core:test[e2e]@tep4", "core:test[e2e]@dep4", "core:test[routing]@tp2"} <= keys


def test_named_members_and_a_changed_entry_change_the_check(tmp_path):
    tests = _tests()
    tests["core"]["e2e"]["members"] = ["tep4"]
    data = _load(tmp_path, _spec(tmp_path, tests))
    first = {c.key: c.inputs for c in contract_mod.compile_contract(data, world_size=4).checks}
    assert "core:test[e2e]@tep4" in first and "core:test[e2e]@dep4" not in first
    tests["core"]["e2e"]["purpose"] = "another purpose"
    data = _load(tmp_path, _spec(tmp_path, tests))
    second = {c.key: c.inputs for c in contract_mod.compile_contract(data, world_size=4).checks}
    # A changed entry runs the test again; the flow's own checks keep their verdicts.
    assert first["core:test[e2e]@tep4"] != second["core:test[e2e]@tep4"]
    assert first["core:accuracy[gsm8k]@tep4"] == second["core:accuracy[gsm8k]@tep4"]


def test_the_bring_up_holds_the_first_units_own_checks_only(tmp_path):
    tests = _tests()
    tests["core"]["routing"]["stage"] = "landing"
    data = _load(tmp_path, _spec(tmp_path, tests))
    contract = contract_mod.compile_contract(data, world_size=4)
    gates = {c.gate for c in contract_mod.bringup_checks(contract)}
    assert gates and all(not g.startswith("test[") for g in gates)


# ----------------------------------------------------------------------------- the schema


@pytest.mark.parametrize(
    "tests, says",
    [
        ({"nope": {"t": {"purpose": "p", "pytest": ROUTING}}}, "names no feature"),
        ({"core": {"t": {"pytest": ROUTING}}}, "purpose"),
        ({"core": {"t": {"purpose": "p"}}}, "pytest"),
        ({"core": {"t": {"purpose": "p", "pytest": "/abs/test_x.py"}}}, "repository-relative"),
        ({"core": {"t": {"purpose": "p", "pytest": "tests/x/README.md"}}}, "repository-relative"),
        (
            {"core": {"t": {"purpose": "p", "pytest": ROUTING, "members": ["tp1"], "devices": 1}}},
            "both 'members'",
        ),
        ({"core": {"t": {"purpose": "p", "pytest": ROUTING, "stage": "nightly"}}}, "stage"),
        ({"core": {"t": {"purpose": "p", "pytest": ROUTING, "members": ["xx4"]}}}, "members"),
        ({"core": {"t": {"purpose": "p", "pytest": ROUTING, "members": ["unified"]}}}, "concrete"),
        ({"core": {"t": {"purpose": "p", "pytest": ROUTING, "devices": 0}}}, "devices"),
        ({"core": {"t": {"purpose": "p", "pytest": E2E}}}, "{member}"),
        (
            {"core": {"t": {"purpose": "p", "pytest": ROUTING, "env": {"MV2FLOW_MODEL": "x"}}}},
            "set by the flow",
        ),
        (
            {"core": {"t": {"purpose": "p", "pytest": ROUTING, "run": "pytest"}}},
            "not a key of a test",
        ),
        ({"core": {"t:x": {"purpose": "p", "pytest": ROUTING}}}, "name"),
        (
            {
                "core": {"t": {"purpose": "p", "pytest": ROUTING}},
                "default": {"t": {"purpose": "p", "pytest": ROUTING}},
            },
            "names no feature",
        ),
    ],
)
def test_a_spec_test_the_schema_refuses(tmp_path, tests, says):
    with pytest.raises(task_schema.TaskSchemaError, match=None) as excinfo:
        _load(tmp_path, _spec(tmp_path, tests))
    assert says in str(excinfo.value)


def test_one_name_is_one_test_and_verify_holds_tests_only(tmp_path):
    tests = {"core": {"t": {"purpose": "p", "pytest": ROUTING}}}
    spec = _spec(tmp_path, tests)
    spec["features"] = ["mtp"]
    spec["verify"]["tests"]["mtp"] = {"t": {"purpose": "p", "pytest": ROUTING}}
    spec["verify"]["preset"] = "standard"
    with pytest.raises(task_schema.TaskSchemaError) as excinfo:
        _load(tmp_path, spec)
    assert "also declared under `core`" in str(excinfo.value)
    assert "'verify.preset' is not a key" in str(excinfo.value)


# ----------------------------------------------------------------------------- preflight


def _tree(repo: Path) -> None:
    (repo / "tests" / "x").mkdir(parents=True, exist_ok=True)
    (repo / "tests" / "x" / "test_a.py").write_text(
        "class TestA:\n    def test_e2e(self):\n        pass\n", encoding="utf-8"
    )
    (repo / "tests" / "x" / "test_b.py").write_text("def test_route():\n    pass\n")


def test_preflight_reads_each_spec_test_off_the_trees_sources(tmp_path):
    data = _load(tmp_path, _spec(tmp_path, _tests()))
    _tree(Path(data["trtllm_repo_path"]))
    notes = preflight.check_user_tests(data)
    assert any("`core.e2e`" in note and "delivery" in note for note in notes)
    tests = _tests()
    tests["core"]["e2e"]["pytest"] = "tests/x/test_a.py::TestA::test_gone[{member}]"
    data = _load(tmp_path, _spec(tmp_path, tests))
    with pytest.raises(preflight.SpecInconsistencyError, match="defines no `TestA::test_gone`"):
        preflight.check_user_tests(data)


def test_verify_defined_follows_a_base_class_of_the_tree(tmp_path):
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "base.py").write_text(
        "class Base:\n    def test_x(self):\n        pass\n"
    )
    (tmp_path / "tests" / "test_c.py").write_text(
        "from tests.base import Base\n\nclass TestC(Base):\n    pass\n"
    )
    there, _ = verify_mod.defined(tmp_path, "tests/test_c.py::TestC::test_x[tep4]")
    assert there is True
    there, said = verify_mod.defined(tmp_path, "tests/test_c.py::TestC::test_y")
    assert there is False and "test_y" in said


# ----------------------------------------------------------------------------- running one


def _report(workdir: Path, test: str, unit: str, member: str) -> Path:
    stem = "".join(c if c.isalnum() or c in "._-" else "_" for c in f"test[{test}]")
    return workdir / f"{stem}.{unit}.{member}.xml"


def test_a_whole_model_test_is_handed_its_members_configuration(tmp_path):
    data = _load(tmp_path, _spec(tmp_path, _tests()))
    test = verify_mod.find(data, "e2e")
    work = tmp_path / "gate"
    seen = []

    def runner(command, env, timeout):
        seen.append((command, timeout))
        _report(work, "e2e", "core", "tep4").write_text(PASS)
        return CommandResult(returncode=0)

    result = user_tests.run_user_test(
        RunEnvironment(kind="local", python="python3", world_size=4),
        task_data=data,
        test=test,
        workdir=work,
        runner=runner,
        member="tep4",
        features=[],
    )
    assert result.passed, result.detail
    command, timeout = seen[0]
    assert "test_a.py::TestA::test_e2e[tep4]" in command and timeout == verify_mod.DEFAULT_TIMEOUT_S
    for name in verify_mod.HANDOVER_ENV:
        assert name in command
    options = yaml.safe_load(next(work.glob("test.e2e.core.tep4.llm_api.yaml")).read_text())
    assert options["tensor_parallel_size"] == 4 and options["moe_expert_parallel_size"] == 4
    assert options["modeling_v2"] == "require" and "model" not in options


def test_a_module_test_is_handed_its_feature_only(tmp_path):
    data = _load(tmp_path, _spec(tmp_path, _tests()))
    test = verify_mod.find(data, "routing")
    work = tmp_path / "gate"

    def runner(command, env, timeout):
        assert "MV2FLOW_FEATURE" in command and "MV2FLOW_LLM_OPTIONS" not in command
        _report(work, "routing", "core", "tp2").write_text(PASS)
        return CommandResult(returncode=0)

    result = user_tests.run_user_test(
        RunEnvironment(kind="local", python="python3", world_size=2),
        task_data=data,
        test=test,
        workdir=work,
        runner=runner,
        member="tp2",
    )
    assert result.passed, result.detail


@pytest.mark.parametrize(
    "xml, code, says",
    [
        (
            '<testsuite><testcase name="t"><skipped message="needs 4 GPUs"/></testcase></testsuite>',
            0,
            "skipped",
        ),
        ("<testsuite></testsuite>", 0, "selected no test"),
        (PASS, 3, "exited 3"),
        (None, 0, "no test report"),
    ],
)
def test_a_spec_test_that_tested_nothing_fails(tmp_path, xml, code, says):
    data = _load(tmp_path, _spec(tmp_path, _tests()))
    test = verify_mod.find(data, "routing")
    work = tmp_path / "gate"

    def runner(command, env, timeout):
        if xml is not None:
            _report(work, "routing", "core", "tp2").write_text(xml)
        return CommandResult(returncode=code)

    result = user_tests.run_user_test(
        RunEnvironment(kind="local", python="python3", world_size=2),
        task_data=data,
        test=test,
        workdir=work,
        runner=runner,
        member="tp2",
    )
    assert not result.passed and says in result.detail


def test_every_node_of_a_test_runs_and_each_must_pass(tmp_path):
    tests = _tests()
    tests["core"]["routing"]["pytest"] = [ROUTING, "tests/x/test_c.py"]
    data = _load(tmp_path, _spec(tmp_path, tests))
    test = verify_mod.find(data, "routing")
    work = tmp_path / "gate"
    ran = []

    def runner(command, env, timeout):
        part = len(ran)
        ran.append(command)
        xml = (
            PASS if part == 0 else "<testsuite><testcase name='t'><failure/></testcase></testsuite>"
        )
        _report(work, "routing", f"core.part{part}", "tp2").write_text(xml)
        return CommandResult(returncode=0)

    result = user_tests.run_user_test(
        RunEnvironment(kind="local", python="python3", world_size=2),
        task_data=data,
        test=test,
        workdir=work,
        runner=runner,
        member="tp2",
    )
    assert len(ran) == 2 and not result.passed and "2/2 pytest jobs" in result.detail


def test_the_check_runner_runs_a_spec_test_as_its_gate(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

    data = _load(tmp_path, _spec(tmp_path, _tests()))
    contract = contract_mod.compile_contract(data, world_size=4)
    check = next(c for c in contract.checks if c.key == "core:test[e2e]@tep4")
    workflow = TrtllmModelingV2Workflow(workspace=tmp_path / "ws")
    workflow.environment = RunEnvironment(kind="local", python="python3", world_size=4)
    workflow.task_data, workflow._contract = data, contract
    runner = CheckRunner(workflow)
    gates = workflow.workspace / "gates" / "core"

    def shell(command, env, timeout):
        assert "test_e2e[tep4]" in command and "MV2FLOW_MEMBER" in command
        _report(gates, "e2e", "core", "tep4").write_text(PASS)
        return CommandResult(returncode=0)

    result = runner._gate_call(check)(shell)
    assert result.passed, result.detail


def test_a_spec_tests_file_counts_as_what_its_gate_reads(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.run.workflow import gate_inputs, gate_relevant

    data = _load(tmp_path, _spec(tmp_path, _tests()))
    _tree(Path(data["trtllm_repo_path"]))
    keep = gate_inputs(data, data["trtllm_repo_path"])
    # Under tests/, which no gate reads otherwise: a change to it changes what the test measures.
    assert {"tests/x/test_a.py", "tests/x/test_b.py"} <= keep
    assert gate_relevant("tests/x/test_a.py", keep=keep)
    assert not gate_relevant("tests/x/test_other.py", keep=keep)


def test_the_gate_command_runs_a_spec_test_by_hand(tmp_path, monkeypatch, capsys):
    import re

    from agent_flow.workflows.trtllm_modeling_v2.checks import gate_cli

    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(_spec(tmp_path, _tests())), encoding="utf-8")
    seen = []

    def fake_runner_for(executor, **kwargs):
        def run(command, env=None, timeout=0):
            seen.append(command)
            report = re.search(r"--junitxml=([^\s']+)", command)
            Path(report.group(1)).write_text(PASS, encoding="utf-8")
            return CommandResult(returncode=0)

        return run

    monkeypatch.setattr(gate_cli, "shell_runner_for", fake_runner_for)
    common = [
        "--task",
        str(path),
        "--tree",
        str(tmp_path / "repo"),
        "--workdir",
        str(tmp_path / "w"),
    ]
    assert gate_cli.main(["test", "--test", "e2e", "--member", "tep4", *common]) == 0
    assert "test_e2e[tep4]" in seen[-1] and "MV2FLOW_LLM_OPTIONS" in seen[-1]
    assert gate_cli.main(["test", "--test", "routing", *common]) == 0
    assert "MV2FLOW_LLM_OPTIONS" not in seen[-1]
    assert gate_cli.main(["test", "--test", "nope", *common]) == 2
    assert "declares no test 'nope'" in capsys.readouterr().err
