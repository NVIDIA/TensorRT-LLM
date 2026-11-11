# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 gates behavior and regression checks."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import accuracy as accuracy_checks
from agent_flow.workflows.trtllm_modeling_v2.checks import gate_cli
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import serve_driver, workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.operators import coverage as catalog_paths
from agent_flow.workflows.trtllm_modeling_v2.run.report import test_modules as target_tests
from agent_flow.workflows.trtllm_modeling_v2.spec import task_schema

from ._helpers import attested


def test_the_report_counts_the_targets_test_modules_and_warns_past_twenty(tmp_path):
    """An onboarded target arrived with 35 test modules, past what a reviewer accepts."""
    target = f"{catalog_paths.MODELING_V2_PACKAGE}/models/fam/target"
    tests = tmp_path / catalog_paths.model_ops_tests_dir(target)
    (tests / "attention").mkdir(parents=True)
    (tests / "_helpers.py").write_text("x = 1\n")
    for n in range(20):
        (tests / "attention" / f"test_modeling_v2_op{n}.py").write_text("def test_a():\n    pass\n")
    check = target_tests(tmp_path, target)
    assert check.passed and check.detail.startswith("20 test module(s), 40 lines")
    assert "WARNING" not in check.detail
    (tests / "test_forward.py").write_text("def test_b():\n    pass\n")
    check = target_tests(tmp_path, target)
    assert check.passed, "the budget is reported, never a failure"
    assert check.detail.startswith("21 test module(s)")
    assert "WARNING: more than the 20 a reviewer accepts" in check.detail
    assert target_tests(tmp_path, f"{catalog_paths.MODELING_V2_PACKAGE}/models/fam/other") is None


# ------------------------------------------------------------------ the gates, by deployment

NODE = "tests/integration/defs/accuracy/test_disaggregated_serving.py::TestM::test_auto_dtype"
PASS = '<testsuite><testcase classname="t" name="native_hop"/></testsuite>'


def _spec(tmp_path, **more) -> dict:
    """A spec that states four deployments: one rank, two ranks behind a prefix, a pair, a drafter."""
    for name in ("ckpt", "repo"):
        (tmp_path / name).mkdir(exist_ok=True)
    spec = {
        "checkpoint_path": str(tmp_path / "ckpt"),
        "arch": "sm_103",
        "trtllm_repo_path": str(tmp_path / "repo"),
        "environment": {"python": "/usr/bin/python3", "env": {"TMPDIR": "/scratch/tmp"}},
        "agents": {"defaults": {"backend": "claude-code", "model": "test-model"}},
        "accuracy": {"model_key": "org/toy", "reference_yaml": "r.yaml"},
        "deployments": [
            {"name": "tp1", "options": {"max_seq_len": 5120}},
            {
                "name": "tep2",
                "options": {"tensor_parallel_size": 2, "moe_expert_parallel_size": 2},
                "relative_to": "tp1",
                "env": {"MY_SWITCH": "1"},
            },
            {
                "name": "pd",
                "ctx": {"options": {"tensor_parallel_size": 2}},
                "gen": {"options": {"tensor_parallel_size": 2}},
                "relative_to": "tp1",
                "env": {"LLM_MODELS_ROOT": "/models"},
                "tests": [NODE, NODE.replace("auto", "fp8")],
                "benchmarks": [],
            },
            {
                "name": "remote",
                "options": {"tensor_parallel_size": 8},
                "launch": "srun -N2 --ntasks=8 --mpi=pmix",
                "benchmarks": [{"name": "gsm8k", "env": {"JUDGE": "local"}}],
                "tests": ["tests/unittest/test_accept.py::test_acceptance"],
            },
        ],
    }
    spec.update(more)
    (tmp_path / "repo" / "r.yaml").write_text("org/toy:\n  - accuracy: 90.0\n")
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return task_schema.load_and_validate_task_yaml(path)


def _engine(*said: str, reports: str | None = None, code: int = 0, **changes):
    """A runner whose engines print ``said`` and what their ranks run; it notes each command."""
    seen: list[str] = []

    def run(command, env=None, timeout=0):
        seen.append(command)
        report = re.search(r"--junitxml=([^\s']+)", command)
        if report and reports is not None:
            Path(report.group(1)).write_text(reports, encoding="utf-8")
        text = said[min(len(seen), len(said)) - 1] if said else ""
        return CommandResult(returncode=code, stdout=text + attested(command, **changes))

    run.seen = seen  # type: ignore[attr-defined]
    return run


def _records(*texts):
    return "".join(
        f"{workload_io.RECORD_MARKER} {json.dumps({'index': n, 'text': t, 'finish_reason': 'stop'})}\n"
        for n, t in enumerate(texts)
    )


def test_one_function_runs_a_check_on_the_deployment_it_names(tmp_path):
    data = _spec(tmp_path)
    tree = tmp_path / "worktree"
    tree.mkdir()
    shutil.copy(tmp_path / "repo" / "r.yaml", tree / "r.yaml")
    common = dict(workdir=tmp_path / "w", tree=tree, devices=(4, 5, 6, 7))
    runner = _engine(_records("Paris", "the process"))
    boot = gate_cli.run_check(data, "boot", "tep2", runner=runner, **common)
    assert boot.passed, boot.detail
    # The deployment's ranks, on the first of the run's devices, in the tree that was named.
    [command] = runner.seen
    assert command.startswith(f"cd {tree.resolve()} && export PYTHONPATH={tree.resolve()}")
    assert "CUDA_VISIBLE_DEVICES=4,5 " in command and "mpirun -n 2" in command
    assert "MV2FLOW_RANK_FACTS=1" in command and "TRTLLM_MODELING_V2=require" in command
    assert "MY_SWITCH=1" in command  # the deployment's own environment
    assert boot.summary().startswith("[PASS] tep2:boot: engine cold-started")
    assert boot.path == gate_cli.IN_PROCESS and "[in process]" in boot.summary()
    assert (boot.engine[0]["status"], boot.engine[0]["ranks"]) == ("match", 2)
    # Unnamed, the check is core's: one rank, no launcher.
    runner = _engine(_records("Paris", "the process"))
    core = gate_cli.run_check(data, "boot", runner=runner, **common)
    assert core.deployment == "tp1" and "mpirun" not in runner.seen[0]
    assert "CUDA_VISIBLE_DEVICES=4 " in runner.seen[0]
    # An engine that imported another tree's package did not measure this one.
    runner = _engine(_records("Paris", "the"), origin="/opt/other/tensorrt_llm/__init__.py")
    elsewhere = gate_cli.run_check(data, "boot", runner=runner, **common)
    assert not elsewhere.passed
    assert "imported tensorrt_llm from /opt/other/tensorrt_llm/__init__.py, not from" in (
        elsewhere.detail
    )
    # The boot runs the workload the spec states, the deployment's own entry over the run's.
    boot = {"prompts": ["The capital of France is"], "keywords": ["Paris"]}
    judged = {"model_key": "org/toy", "reference_yaml": "r.yaml"}
    stated = _spec(tmp_path, accuracy={**judged, "boot": {**boot, "tep2": {"keywords": ["Lyon"]}}})
    runner = _engine(_records("Paris"))
    assert gate_cli.run_check(stated, "boot", runner=runner, **common).passed
    lyon = gate_cli.run_check(stated, "boot", "tep2", runner=runner, **common)
    assert not lyon.passed and "missing 1 expected keyword(s): ['Lyon']" in lyon.detail
    # An answer that did not complete (no finish reason) is not an answer.
    cut = f"{workload_io.RECORD_MARKER} {json.dumps({'index': 0, 'text': 'Paris'})}\n"
    unfinished = gate_cli.run_check(stated, "boot", runner=_engine(cut), **common)
    assert (
        not unfinished.passed and "did not complete for b0 (no finish reason)" in unfinished.detail
    )
    with pytest.raises(task_schema.TaskSchemaError, match="names no gate"):
        gate_cli.run_check(data, "perf", **common)
    with pytest.raises(KeyError, match="'dep4' names no deployment of this spec"):
        gate_cli.run_check(data, "boot", "dep4", **common)
    with pytest.raises(task_schema.TaskSchemaError, match="`pd` names no benchmark"):
        gate_cli.run_check(data, "accuracy", "pd", **common)
    with pytest.raises(task_schema.TaskSchemaError, match="`pd` is disaggregated: its boot serves"):
        gate_cli.run_check(data, "serve", "pd", **common)


def test_accuracy_runs_trtllm_eval_behind_the_deployments_own_prefix(tmp_path):
    data = _spec(tmp_path)
    runner = _engine("lm-eval gsm8k average accuracy: 91.0\n")
    result = gate_cli.run_check(
        data, "accuracy[gsm8k]", "remote", workdir=tmp_path / "w", runner=runner
    )
    assert result.passed, result.detail
    assert result.summary().startswith(
        "[PASS] remote:accuracy[gsm8k]: measured 91.0000 vs floor 88.3479"
    )
    assert "anchor: reference" in result.detail and result.path == gate_cli.IN_PROCESS
    [command] = runner.seen
    # The prefix is the site's and places the ranks: the flow adds the LLM API's launcher
    # for the eight its options ask for, and no device of this host.
    assert "; srun -N2 --ntasks=8 --mpi=pmix trtllm-llmapi-launch trtllm-eval " in command
    assert "CUDA_VISIBLE_DEVICES" not in command and "mpirun" not in command
    # The evaluator is the one beside the run's interpreter, and nothing of the
    # orchestrator's scheduler identity goes behind the prefix: not its rank either.
    assert '; PATH=/usr/bin:"$PATH"; srun -N2 ' in command
    assert "SLURM_PROCID" not in command and "${!SLURM@}" in command
    # The evaluator runs offline, on the dataset this host holds, under the benchmark's env.
    assert "HF_DATASETS_OFFLINE=1" in command and "JUDGE=local" in command
    assert " gsm8k --dataset_path=" in command and "TLLM_EVAL_PARTIAL_SCORES_EVERY=50" in command
    options = yaml.safe_load(
        (tmp_path / "w" / "remote" / "eval.remote.gsm8k.yaml").read_text(encoding="utf-8")
    )
    assert options == {"tensor_parallel_size": 8, "modeling_v2": "require"}
    # An engine that ran another configuration fails whatever it scored; one rank silent is unknown.
    other = _engine("lm-eval gsm8k average accuracy: 91.0\n", ranks=7)
    unknown = gate_cli.run_check(data, "accuracy", "remote", workdir=tmp_path / "w", runner=other)
    assert not unknown.passed and "rank 7: it printed no record" in unknown.detail
    # A score in another form is quoted for whoever decides how it is read, then read by pattern.
    printed = "Partial scores after 100/100 responses\nJSON Mode Eval accuracy: 91.50 (100)\n"
    assert accuracy_checks.parse_eval_score(printed) is None
    assert "  JSON Mode Eval accuracy: 91.50 (100)" in accuracy_checks.unparsed_score_detail(
        printed, None
    )
    assert accuracy_checks.parse_eval_score(printed, r"Eval accuracy: ([0-9.]+)") == 91.5


def test_a_dataset_is_looked_for_under_the_deployments_own_environment(tmp_path, monkeypatch):
    root = os.environ["LLM_MODELS_ROOT"]
    for name in ("LLM_MODELS_ROOT", "HF_HUB_CACHE", "HF_DATASETS_CACHE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    data = _spec(tmp_path)
    remote = next(entry for entry in data["deployments"] if entry["name"] == "remote")
    runner = _engine("lm-eval gsm8k average accuracy: 91.0\n")
    common = dict(workdir=tmp_path / "w", runner=runner)
    lost = gate_cli.run_check(data, "accuracy[gsm8k]", "remote", **common)
    assert not lost.passed and "is not on this host" in lost.infrastructure and not runner.seen
    # Where the evaluator runs it has the deployment's models root: the dataset is there.
    remote["env"] = {"LLM_MODELS_ROOT": root}
    found = gate_cli.run_check(data, "accuracy[gsm8k]", "remote", **common)
    assert found.passed and f"--dataset_path={root}/datasets/openai/gsm8k" in runner.seen[0]


def test_a_relative_deployment_is_judged_against_the_other_ones_score_measured_first(
    tmp_path, capsys, monkeypatch
):
    data = _spec(tmp_path)
    # No row carries tep2's tags once its options differ from the row's: judged against tp1.
    for entry in data["deployments"]:
        if entry["name"] == "tep2":
            entry["options"]["kv_cache_config"] = {"dtype": "fp8"}
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    runner = _engine(*(f"lm-eval gsm8k average accuracy: {score}\n" for score in (92, 85)))
    monkeypatch.setattr(gate_cli, "shell_runner_for", lambda executor, **kwargs: runner)
    argv = [
        "accuracy",
        "--task",
        str(path),
        "--deployment",
        "tep2",
        "--workdir",
        str(tmp_path / "w"),
    ]
    assert gate_cli.main(argv) == 1
    out = capsys.readouterr().out
    assert out.startswith("[FAIL] tep2:accuracy[gsm8k]: measured 85.0000 vs floor 89.8872")
    assert "the other deployment's score 92.0000" in out and len(runner.seen) == 2
    # The first run is tp1's, as tp1 runs; the second is the deployment's own.
    assert "MY_SWITCH=1" not in runner.seen[0] and "MY_SWITCH=1" in runner.seen[1]
    assert gate_cli.main(["boot", "--task", str(path), "--deployment", "nope"]) == 2
    assert "error: 'nope' names no deployment of this spec" in capsys.readouterr().err


def test_a_score_nothing_judges_is_a_report_line_and_never_a_failure(tmp_path, capsys, monkeypatch):
    data = _spec(tmp_path)
    for entry in data["deployments"]:
        if entry["name"] == "tep2":
            entry["options"]["kv_cache_config"] = {"dtype": "fp8"}
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    # No row for the model and nothing declared: tp1's score is held to nothing.
    (tmp_path / "repo" / "r.yaml").write_text("org/other:\n  - accuracy: 90.0\n")
    runner = _engine("lm-eval gsm8k average accuracy: 61.5\n")
    monkeypatch.setattr(gate_cli, "shell_runner_for", lambda executor, **kwargs: runner)
    argv = ["accuracy", "--task", str(path), "--workdir", str(tmp_path / "w")]
    assert gate_cli.main(argv) == 0
    out = capsys.readouterr().out
    assert out.startswith("[NO VERDICT] tp1:accuracy[gsm8k]: measured 61.5000; a score with no")
    # What to do about it is said whole: the line is cut at no word.
    assert "(citing where the score came from)" in out and "on the same GPU type.)" in out
    result = gate_cli.run_check(data, "accuracy[gsm8k]", "tp1", workdir=tmp_path / "w")
    assert result.unjudged and result.passed and result.measured == 61.5
    assert result.floor is None and "no declared anchor and no reference row" in result.detail
    # A deployment judged against another's score, when that one measured none: its own
    # score is reported the same way.
    (tmp_path / "repo" / "r.yaml").write_text("org/toy:\n  - accuracy: 90.0\n")
    runner = _engine("no score here\n", "lm-eval gsm8k average accuracy: 85\n")
    monkeypatch.setattr(gate_cli, "shell_runner_for", lambda executor, **kwargs: runner)
    result = gate_cli.run_check(data, "accuracy[gsm8k]", "tep2", workdir=tmp_path / "w")
    assert result.unjudged and result.passed and result.measured == 85.0
    assert "a score with no verdict (`tp1` has no score to hold it to)" in result.detail
    assert result.summary().startswith("[NO VERDICT] tep2:accuracy[gsm8k]: measured 85.0000")


def test_native_tests_run_one_job_each_and_a_skip_or_a_bad_exit_is_not_a_pass(tmp_path):
    data = _spec(tmp_path)
    common = dict(workdir=tmp_path / "w")
    runner = _engine(reports=PASS)
    result = gate_cli.run_check(data, "tests", "pd", runner=runner, **common)
    assert result.passed, result.detail
    assert result.summary().startswith("[PASS] pd:tests: ")
    assert result.detail.count("CLEAN (1 passed, 0 skipped, 0 failing)") == 2
    # One pytest process for each test, under the deployment's environment, on its four ranks.
    assert len(runner.seen) == 2 and all("-m pytest" in c for c in runner.seen)
    assert (
        "LLM_MODELS_ROOT=/models" in runner.seen[0]
        and "TRTLLM_MODELING_V2=require" in runner.seen[0]
    )
    assert "CUDA_VISIBLE_DEVICES=0,1,2,3 " in runner.seen[0]
    assert "test_disaggregated_serving.py::TestM::test_auto_dtype" in runner.seen[0]
    skipped = "<testsuite><testcase classname='t' name='hop'><skipped/></testcase></testsuite>"
    for reports, code, said in (
        (skipped, 0, "t::hop skipped"),
        ("<testsuite><testcase><failure message='x'/></testcase></testsuite>", 0, "FAILING"),
        (None, 0, "no test report"),
        ("<broken", 0, "no test report"),
        (PASS, 9, "pytest exited 9"),
    ):
        runner = _engine(reports=reports, code=code)
        result = gate_cli.run_check(data, "tests", "pd", runner=runner, **common)
        assert not result.passed and said in result.detail, (reports, result.detail)
        assert len(runner.seen) == 1  # the first job that does not pass ends the check
    # An aggregated deployment's native tests run behind its own prefix, as its server does.
    runner = _engine(reports=PASS)
    assert gate_cli.run_check(data, "tests", "remote", runner=runner, **common).passed
    assert (
        "; srun -N2 --ntasks=8 --mpi=pmix trtllm-llmapi-launch /usr/bin/python3 -m pytest"
        in (runner.seen[0])
    )
    assert "tests/unittest/test_accept.py::test_acceptance" in runner.seen[0]
    assert "CUDA_VISIBLE_DEVICES" not in runner.seen[0]
    none = gate_cli.run_check(data, "tests", "tp1", runner=runner, **common)
    assert not none.passed and none.detail == "the deployment names no test"


def test_the_serve_gate_reads_its_driver_and_names_the_deployment(tmp_path):
    data = _spec(tmp_path)
    answer = {"index": 0, "status": 200, "text": " Paris."}
    stdout = (
        f"{workload_io.RECORD_MARKER} {json.dumps(answer)}\n"
        f"{serve_driver.SERVER_MARKER} {json.dumps({'healthy': True, 'boot_s': 3.0})}\n"
    )
    runner = _engine(stdout)
    result = gate_cli.run_check(data, "serve", "tep2", workdir=tmp_path / "w", runner=runner)
    assert result.passed, result.detail
    assert result.summary().startswith("[PASS] tep2:serve: trtllm-serve healthy after 3.0s")
    spec = json.loads((tmp_path / "w" / "tep2" / "serve.tep2.json").read_text())
    assert spec["world_size"] == 2 and spec["serve_args"][-1] == "--num_serve_frontends=1"
