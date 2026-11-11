# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 gates behavior and regression checks."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import accuracy as accuracy_checks
from agent_flow.workflows.trtllm_modeling_v2.checks import delivery, gate_cli
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import serve_driver, workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.operators import coverage as catalog_paths
from agent_flow.workflows.trtllm_modeling_v2.run.certify import (
    target_test_modules_check as target_tests,
)
from agent_flow.workflows.trtllm_modeling_v2.spec import task_schema

from ._helpers import _ENTRY, _REPO_ROOT, TARGET_REL, _build_repo, _verify, _write_entry, attested


def test_normalization_preserves_source_skip_template_order_and_image_requests(tmp_path):
    data = tmp_path / "data.jsonl"
    data.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"q": "skip", "tag": "x"},
                {},
                {"q": "first", "tag": "a"},
                {"q": "second", "tag": "b"},
            ]
        )
    )
    image = {
        "id": "image",
        "prompt": "look",
        "images": ["/local/image.png"],
        "render": "raw",
        "max_tokens": 3,
    }
    spec = {
        "requests": [image],
        "dataset": {
            "source": str(data),
            "text_field": "q",
            "skip": 1,
            "num": 2,
            "template": "{tag}: {text}",
        },
    }
    requests = workload_io.normalize_requests(spec)
    assert requests[0] == image
    assert [r["prompt"] for r in requests[1:]] == ["a: first", "b: second"]
    spec["dataset"]["skip"] = 99
    with pytest.raises(workload_io.DatasetError, match="no usable prompts"):
        workload_io.normalize_requests(spec)


def test_an_uncommitted_target_fails_even_though_the_file_is_there(tmp_path):
    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    (tmp_path / "seed").write_text("s\n", encoding="utf-8")
    git("add", "seed")
    git("-c", "user.email=t@e.st", "-c", "user.name=t", "commit", "-qm", "seed")
    (tmp_path / "modeling.py").write_text("# never committed\n", encoding="utf-8")

    [check] = delivery.check_tree_committed(tmp_path, ("modeling.py",))
    assert not check.passed
    assert "a clone would not have them" in check.detail and "modeling.py" in check.detail


def test_the_accuracy_test_path_follows_the_tree_layout(tmp_path):
    """Each accuracy layout the trees have used maps to its own path, and test-db is read by path."""
    package = tmp_path / delivery.ACCURACY_TESTS_DIR / "modeling_v2"
    oldest = delivery.accuracy_test_path(tmp_path, "gpt_oss", "t")
    assert oldest == "test_modeling_v2_gpt_oss.py"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    assert (
        delivery.accuracy_test_path(tmp_path, "gpt_oss", "t")
        == "modeling_v2/gpt_oss/t/test_accuracy.py"
    )
    (package / "test_gpt_oss.py").write_text("", encoding="utf-8")
    assert delivery.accuracy_test_path(tmp_path, "gpt_oss", "t") == "modeling_v2/test_gpt_oss.py"

    # Per target, every file is test_accuracy.py: registration must match the path, not the stem.
    (package / "test_gpt_oss.py").unlink()
    accuracy_test = "modeling_v2/gpt_oss/t/test_accuracy.py"
    (package / "gpt_oss" / "t").mkdir(parents=True)
    (package / accuracy_test.split("/", 1)[1]).write_text("", encoding="utf-8")
    db = tmp_path / delivery.TEST_DB_DIR
    db.mkdir(parents=True)
    (db / "l0_x.yml").write_text(
        "  - accuracy/modeling_v2/gpt_oss/other/test_accuracy.py::TestOther::test_gsm8k\n",
        encoding="utf-8",
    )
    checks = {c.name: c.passed for c in delivery.check_test_registration(tmp_path, accuracy_test)}
    assert checks == {"tests:accuracy-file": True, "tests:test-db": False}
    (db / "l0_x.yml").write_text(
        f"  - accuracy/{accuracy_test}::TestT::test_gsm8k\n", encoding="utf-8"
    )
    checks = {c.name: c.passed for c in delivery.check_test_registration(tmp_path, accuracy_test)}
    assert checks == {"tests:accuracy-file": True, "tests:test-db": True}
    # An entry may name the target's directory instead of the file; an entry that only shares
    # a string prefix with the path does not count.
    (db / "l0_x.yml").write_text(
        "  - accuracy/modeling_v2/gpt_oss/t TIMEOUT (30)\n", encoding="utf-8"
    )
    checks = {c.name: c.passed for c in delivery.check_test_registration(tmp_path, accuracy_test)}
    assert checks == {"tests:accuracy-file": True, "tests:test-db": True}
    (db / "l0_x.yml").write_text("  - accuracy/modeling_v2/gpt\n", encoding="utf-8")
    checks = {c.name: c.passed for c in delivery.check_test_registration(tmp_path, accuracy_test)}
    assert checks == {"tests:accuracy-file": True, "tests:test-db": False}


@pytest.mark.skipif(
    not (_REPO_ROOT / catalog_paths.MODELING_V2_PACKAGE).is_dir(),
    reason="not running inside a TensorRT-LLM checkout with modeling_v2 present",
)
@pytest.mark.parametrize(
    ("target_rel", "family", "model_key", "accuracy_test"),
    [
        (
            f"{catalog_paths.MODELING_V2_PACKAGE}/models/gpt_oss/gpt_oss_120b__sm_10x__default",
            "gpt_oss",
            "openai/gpt-oss-120b",
            "modeling_v2/gpt_oss/gpt_oss_120b__sm_10x__default/test_accuracy.py",
        ),
        (
            f"{catalog_paths.MODELING_V2_PACKAGE}/models/deepseek_v3/r1_0528_nvfp4__sm_10x__default",
            "deepseek_v3",
            "deepseek-ai/DeepSeek-R1-0528",
            "modeling_v2/deepseek_v3/r1_0528_nvfp4__sm_10x__default/test_accuracy.py",
        ),
    ],
    ids=["gpt_oss", "deepseek_v3"],
)
def test_the_real_merged_target_layout_and_collected_tests(
    target_rel, family, model_key, accuracy_test
):
    """The path conventions must match what the repository actually uses.

    The `deepseek_v3` case is multi-rank, so it also holds the convention for
    the `comm` entries' op-matrix tests.
    """
    report = delivery.verify_delivery(
        repo_root=_REPO_ROOT,
        target_rel=target_rel,
        family=family,
        reference_yaml=f"{delivery.ACCURACY_REFERENCES_DIR}/gsm8k.yaml",
        model_key=model_key,
        accuracy_test=accuracy_test,
    )
    # This existing entry has only a manually launched multi-node rank body on main;
    # delivery must not treat it as a collected pytest test. All other conventions hold.
    assert [check.name for check in report.failures] == ["test:comm/mnnvl_allreduce"], (
        report.render()
    )
    assert "missing:" in report.failures[0].detail


def test_lifecycle_layout_is_a_choice_and_family_sharing_is_allowed(tmp_path):
    target = tmp_path / "models" / "family" / "custom_target"
    sibling = target.parent / "sibling"
    target.mkdir(parents=True)
    sibling.mkdir()
    (sibling / "modeling.py").write_text("class Other: pass\n")
    (target / "modeling.py").write_text("from ..shared import Base\nclass Model(Base): pass\n")
    (target / "weights.py").write_text("from ..weights import load_weights\n")
    assert all(
        check.passed for check in delivery.check_target(tmp_path, "models/family/custom_target")
    )
    (target / "modeling.py").write_text("from ..sibling.modeling import Other\n")
    checks = delivery.check_target(tmp_path, "models/family/custom_target")
    assert [check.name for check in checks if not check.passed] == ["target:isolated"]


def test_inherited_weights_need_no_local_file_but_present_python_is_validated(tmp_path):
    root = _build_repo(tmp_path)
    target = root / TARGET_REL
    (target / "weights.py").unlink()
    assert _verify(root).passed
    (target / "weights.py").write_text("def broken(\n", encoding="utf-8")
    assert any("weights.py" in check.detail for check in _verify(root).failures)


@pytest.mark.parametrize("location", ["helper", "family", "catalog_reexport"])
@pytest.mark.parametrize("defect", ["cells", "archs", "test"])
def test_indirect_entries_must_have_contracts_and_collected_tests(tmp_path, location, defect):
    root = _build_repo(tmp_path, modeling="from .helpers import run as operation\n")
    target = root / TARGET_REL
    helpers = target / "helpers"
    helpers.mkdir()
    (helpers / "__init__.py").write_text("from .nested import run\n", encoding="utf-8")
    nested = helpers / "nested.py"
    if location == "helper":
        nested.write_text("from ..ops.moe.router import Entry as run\n", encoding="utf-8")
        entry = target / "ops/moe/router.py"
        test = (
            root / catalog_paths.model_ops_tests_dir(TARGET_REL) / "moe/test_modeling_v2_router.py"
        )
    elif location == "family":
        nested.write_text("from ...shared import run\n", encoding="utf-8")
        (target.parent / "shared.py").write_text(
            "from .ops.norm.shared import Entry as run\n", encoding="utf-8"
        )
        entry = target.parent / "ops/norm/shared.py"
        test = (
            root
            / catalog_paths.CATALOG_TESTS_DIR
            / "models/gpt_oss/norm/test_modeling_v2_shared.py"
        )
        _write_entry(entry, test)
    else:
        nested.write_text(
            "from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm import exported as run\n",
            encoding="utf-8",
        )
        entry = root / catalog_paths.CATALOG_DIR / "norm/flashinfer_rmsnorm.py"
        (entry.parent / "__init__.py").write_text(
            "from .flashinfer_rmsnorm import Entry as exported\n", encoding="utf-8"
        )
        test = (
            root / catalog_paths.CATALOG_TESTS_DIR / "norm/test_modeling_v2_flashinfer_rmsnorm.py"
        )
    assert _verify(root).passed
    if defect == "test":
        test.unlink()
    else:
        marker = "    CELLS =" if defect == "cells" else "    ARCHS ="
        entry.write_text(
            "\n".join(line for line in _ENTRY.splitlines() if not line.startswith(marker)),
            encoding="utf-8",
        )
    failures = _verify(root).failures
    assert failures
    assert any(
        "missing:" in check.detail if defect == "test" else f"no {defect.upper()}" in check.detail
        for check in failures
    )


@pytest.mark.parametrize("absolute", [False, True])
def test_family_helper_cannot_hide_sibling_import(tmp_path, absolute):
    root = _build_repo(tmp_path, modeling="from ..shared import Other\n")
    target = root / TARGET_REL
    sibling = target.parent / "sibling"
    sibling.mkdir()
    (sibling / "modeling.py").write_text("class Other: pass\n", encoding="utf-8")
    module = (
        "tensorrt_llm._torch._experimental.modeling_v2.models.gpt_oss.sibling.modeling"
        if absolute
        else ".sibling.modeling"
    )
    (target.parent / "shared.py").write_text(f"from {module} import Other\n", encoding="utf-8")
    assert any(check.name == "target:isolated" for check in _verify(root).failures)


@pytest.mark.parametrize("problem", ["missing", "syntax", "missing_submodule"])
def test_reachable_helper_errors_are_reported(tmp_path, problem):
    root = _build_repo(tmp_path, modeling="from .helper import run\n")
    target = root / TARGET_REL
    if problem == "syntax":
        (target / "helper.py").write_text("def run(\n", encoding="utf-8")
    elif problem == "missing_submodule":
        (target / "helper").mkdir()
        (target / "helper/__init__.py").write_text("from . import absent\n", encoding="utf-8")
    assert any(check.name.startswith("dependency:") for check in _verify(root).failures)


def test_target_initializer_and_optional_weights_dependencies_are_checked(tmp_path):
    root = _build_repo(tmp_path, modeling="class Model: pass\n")
    target = root / TARGET_REL
    (target / "__init__.py").write_text("from .bootstrap import start\n", encoding="utf-8")
    (target / "bootstrap.py").write_text(
        "from .ops.moe.router import Entry as start\n", encoding="utf-8"
    )
    (target / "weights.py").write_text("from .loader import load\n", encoding="utf-8")
    (target / "loader.py").write_text(
        "from .ops.moe.router import Entry as load\n", encoding="utf-8"
    )
    test = root / catalog_paths.model_ops_tests_dir(TARGET_REL) / "moe/test_modeling_v2_router.py"
    test.unlink()
    assert any("router" in check.detail for check in _verify(root).failures)


def test_delivery_counts_the_targets_test_modules_and_warns_past_twenty(tmp_path):
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
    assert check.passed, "the budget never blocks delivery"
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
                "launch": "srun -N2 --ntasks=8 --mpi=pmix trtllm-llmapi-launch",
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
        f"{workload_io.RECORD_MARKER} {json.dumps({'index': n, 'text': t})}\n"
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
    # The prefix places the ranks: no device of this host and no launcher of the flow's.
    assert "exec srun -N2 --ntasks=8 --mpi=pmix trtllm-llmapi-launch trtllm-eval " in command
    assert "CUDA_VISIBLE_DEVICES" not in command and "mpirun" not in command
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
    # An aggregated deployment's native tests run the same way.
    runner = _engine(reports=PASS)
    assert gate_cli.run_check(data, "tests", "remote", runner=runner, **common).passed
    assert "tests/unittest/test_accept.py::test_acceptance" in runner.seen[0]
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
