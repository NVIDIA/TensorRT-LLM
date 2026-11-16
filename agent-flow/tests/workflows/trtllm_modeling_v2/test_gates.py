# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 gates behavior and regression checks."""

from __future__ import annotations

import contextlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import acceptance as acceptance_checks
from agent_flow.workflows.trtllm_modeling_v2.checks import accuracy as accuracy_checks
from agent_flow.workflows.trtllm_modeling_v2.checks import (
    anchors,
    delivery,
    gate_cli,
    gate_records,
    native_tests,
    test_results,
)
from agent_flow.workflows.trtllm_modeling_v2.checks import session as session_mod
from agent_flow.workflows.trtllm_modeling_v2.checks import test_results as tc
from agent_flow.workflows.trtllm_modeling_v2.checks import verdict as check_result
from agent_flow.workflows.trtllm_modeling_v2.checks.delivery import Check as DeliveryCheck
from agent_flow.workflows.trtllm_modeling_v2.checks.delivery import DeliveryReport
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import test_driver as drv
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import (
    RunEnvironment,
    resolve_environment,
)
from agent_flow.workflows.trtllm_modeling_v2.operators import coverage as catalog_paths
from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract as contract_mod
from agent_flow.workflows.trtllm_modeling_v2.run.validation import regression
from agent_flow.workflows.trtllm_modeling_v2.run.validation.completion import (
    target_test_modules_check as target_tests,
)
from agent_flow.workflows.trtllm_modeling_v2.run.validation.completion import (
    tests_check as delivered_tests,
)

from ._helpers import (
    _ENTRY,
    _REPO_ROOT,
    TARGET_REL,
    _build_repo,
    _verify,
    _workflow,
    _write_entry,
    passing_shell,
    session_spec,
)

_gates_TASK = {
    "checkpoint_path": "/ckpt/Qwen3-30B-A3B",
    "deployments": [{"name": "core", "options": {}}],
    "trtllm_repo_path": "/repo",
    "accuracy": {
        "benchmark": "gsm8k",
        "reference_yaml": "tests/integration/defs/accuracy/references/gsm8k.yaml",
        "tolerance": 5.0,
    },
}


def _gates_env(world_size: int = 1, devices=(0,)):
    return resolve_environment(
        {"environment": {"kind": "local", "python": "/opt/venv/bin/python", "env": {}}},
        devices=devices,
        world_size=world_size,
    )


def _gates_recorder(result: CommandResult | list[CommandResult]):
    """A runner that returns canned results and records the commands it saw."""
    results = result if isinstance(result, list) else [result]
    seen: list[str] = []

    def run(command, env=None, timeout=0):
        seen.append(command)
        return results[min(len(seen) - 1, len(results) - 1)]

    run.seen = seen  # type: ignore[attr-defined]
    return run


def test_a_cancelled_gate_has_no_verdict_and_is_not_queued_again_at_once(tmp_path):
    """A gate job cancelled through the spool raises: missing, not FAIL, and no re-run here."""
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute.executor import CANCELLED
    from agent_flow.workflows.trtllm_modeling_v2.run.validation import check_runner
    from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

    workflow = TrtllmModelingV2Workflow(workspace=tmp_path / "ws")
    workflow.environment = RunEnvironment(kind="local", python="python3", world_size=1)
    workflow.task_data = dict(_gates_TASK, trtllm_repo_path=str(tmp_path))
    runner = check_runner.CheckRunner(workflow)
    runner._await_session = lambda log: pytest.fail("a cancelled gate waits for no session")
    cancelled = f"{CANCELLED}: cancelled before it started"
    runs = []

    def run():
        runs.append(1)
        return SimpleNamespace(
            name="accuracy[gsm8k]", member="dep4", passed=False, infrastructure=cancelled
        )

    with pytest.raises(check_runner._SessionLost, match="measured nothing"):
        runner._gate_until_verdict(run, None)
    assert runs == [1]


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


def test_the_deliverys_rules_for_one_entry_are_asked_of_the_files_a_task_changed(tmp_path):
    """A whole run met "no OpWrapper subclass" two hours after the file landed: the rule needs no finished target."""
    root = _build_repo(tmp_path)
    ops = f"{TARGET_REL}/{catalog_paths.MODEL_OPS_DIRNAME}"
    bare, helper = f"{ops}/attention/paged_kv_binding.py", f"{ops}/attention/_shared.py"
    for path in (bare, helper):
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text("def bind(x):\n    return x\n", encoding="utf-8")
    changed = [bare, helper, f"{ops}/moe/router.py", f"{TARGET_REL}/weights.py", "docs/x.md"]
    checks = delivery.check_changed_entries(root, TARGET_REL, [*changed, f"{ops}/moe/gone.py"])
    # Two rules for each entry among them; a helper, the model's own files, prose and a
    # file that is gone are no entries.
    assert [c.name.partition(":")[0] for c in checks] == ["entry", "test", "entry", "test"]
    wrong, untested, right, tested = checks
    assert wrong.name == f"entry:{bare}" and not wrong.passed
    assert "no OpWrapper subclass" in wrong.detail
    assert not untested.passed and untested.detail.startswith("missing: ")
    assert untested.detail.endswith("/attention/test_modeling_v2_paged_kv_binding.py")
    assert right.name == f"entry:{ops}/moe/router.py" and right.passed and tested.passed
    # The same lines the whole-target check gives of it at the end, once the model imports it.
    modeling = root / TARGET_REL / "modeling.py"
    modeling.write_text(
        modeling.read_text(encoding="utf-8") + "from .ops.attention import paged_kv_binding\n",
        encoding="utf-8",
    )
    late = {c.name: c for c in _verify(root).checks}
    assert late[wrong.name].detail == wrong.detail and not late[wrong.name].passed


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


_ANCHOR = 92.30


_statistical_floor_TASK = {
    "checkpoint_path": "/ckpt/M",
    "parallel": "tep4",
    "trtllm_repo_path": "/repo",
    "accuracy": {"benchmark": "gsm8k", "reference_yaml": "r.yaml", "tolerance": 5.0},
}


def _verdict(member, measured, reference=_ANCHOR, se=0.734):
    return check_result.GateResult(
        name="accuracy[gsm8k]",
        passed=True,
        detail="",
        member=member,
        measured=measured,
        reference=reference,
        se=se,
    )


def test_delivery_refuses_open_findings_first(tmp_path):
    """Before the delivery report, and as a remediable verdict a relaunch re-enters."""
    from agent_flow.workflows.trtllm_modeling_v2.run.validation import completion, contract

    wf = _workflow(tmp_path)
    wf.task_data = {"accuracy": dict(_statistical_floor_TASK["accuracy"])}
    accuracy_checks.note_systematic_shift(
        gate_records.records_for(wf.workspace),
        [_verdict("tep4", 91.0), _verdict("tp4", 91.1)],
        wf.task_data["accuracy"],
        "dspark",
    )
    closed = []
    wf._close_run = lambda state, log, verdict: closed.append(verdict)
    wf._contract = contract.Contract(checks=())
    wf.environment = SimpleNamespace(world_size=1)
    verdict = contract.evaluate(wf._contract, wf._contract_status(), marker="")
    assert (
        completion.run_deliver_phase(
            wf,
            SimpleNamespace(units_done=["core"]),
            None,
            lambda files: completion.assess(verdict, files=files),
        )
        is False
    )
    assert closed == ["delivery-failed:findings"]
    assert "delivery blocked" in wf.journal.path.read_text(encoding="utf-8")


class _Delivery:
    def __init__(self, tmp_path: Path, head: str = "c0ffee1234567890"):
        self.task_data = {"trtllm_repo_path": str(tmp_path)}
        self.workspace = tmp_path / "ws"
        self._preflight = SimpleNamespace(target_rel="pkg/models/fam/target")
        self._git = lambda repo, *args: head + "\n"


def test_delivery_reads_the_delivered_tests_the_final_tests_ran_and_runs_no_job(
    tmp_path, monkeypatch
):
    from agent_flow.workflows.trtllm_modeling_v2.run.validation import completion

    report = DeliveryReport(
        checks=(DeliveryCheck("test:gemm/x", True, str(tmp_path / "tests/test_modeling_v2_x.py")),)
    )
    test = "tests/test_modeling_v2_x.py"
    asked = {}

    def list_test_files(repo, head, paths):
        asked["paths"] = list(paths)
        return [test]

    results = test_results.Results("c0ffee1234567890", "env", collected={test: ["test_a"]})
    monkeypatch.setattr(test_results, "list_test_files", list_test_files)
    monkeypatch.setattr(test_results, "recorded", lambda cache, commit, env_digest="": results)
    stage = _Delivery(tmp_path)
    # Not run at HEAD yet: nothing to say in the middle of a run, and at delivery a test
    # that did not run has not passed.
    assert delivered_tests(stage, report) is None
    assert asked["paths"][1] == test
    assert completion.delivered_test_paths(stage, report) == asked["paths"]
    unrun = delivered_tests(stage, report, final=True)
    assert not unrun.passed
    # The final tests ran it on the frozen candidate: delivery reads that result at HEAD.
    # Nothing here can start a job: the module holds no runner.
    results.tests[f"{test}::test_a"] = {"outcome": "passed"}
    check = delivered_tests(stage, report, final=True)
    assert check.passed and "as the final tests left them at c0ffee1234" in check.detail
    assert not hasattr(completion, "regression_mod")
    # A delivered test of the engine's accuracy harness was collected and left to CI by
    # the final tests: it has no result at HEAD on purpose, and delivery does not wait
    # for one. A spec that had them run waits for it like any other.
    scored = "tests/test_target_accuracy.py"
    results.collected[scored] = ["TestTarget::test_gsm8k"]
    monkeypatch.setattr(test_results, "list_test_files", lambda repo, head, paths: [test, scored])
    monkeypatch.setattr(
        test_results, "accuracy_tests", lambda repo, commit, files: {scored: "class TestTarget"}
    )
    check = delivered_tests(stage, report, final=True)
    assert (
        check.passed
        and "1 accuracy test file(s) of the engine's harness left to CI" in check.detail
    )
    # One that fails at HEAD fails the cheap read too.
    results.tests[f"{test}::test_a"] = {"outcome": "failed", "message": "wrong"}
    check = delivered_tests(stage, report)
    assert not check.passed and check.detail.startswith("FAILING")


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


_requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not available")


def _git(repo: Path, *args: str, env: dict | None = None) -> str:
    done = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, **(env or {})},
    )
    return done.stdout.strip()


def _commit(repo: Path, base: str, changes: dict[str, str], message: str = "change") -> str:
    """A commit on ``base`` with ``changes``, made without touching the checkout."""
    index = repo / ".git" / f"index-{abs(hash(message)) % 10**8}"
    env = {"GIT_INDEX_FILE": str(index)}
    _git(repo, "read-tree", base, env=env)
    for path, content in changes.items():
        blob = subprocess.run(
            ["git", "-C", str(repo), "hash-object", "-w", "--stdin"],
            input=content,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        _git(repo, "update-index", "--add", "--cacheinfo", f"100644,{blob},{path}", env=env)
    tree = _git(repo, "write-tree", env=env)
    index.unlink(missing_ok=True)
    return _git(repo, "commit-tree", tree, "-p", base, "-m", message)


def _run(command: str) -> tuple[int, str]:
    # pytest-timeout is not a test dependency here.
    done = subprocess.run(
        re.sub(r" --timeout=\d+", "", command), shell=True, capture_output=True, text=True
    )
    return done.returncode, done.stdout + done.stderr


def _regression_repo(root: Path) -> tuple[Path, Path, str]:
    """A repository whose base commit's ``test_a`` and ``test_b`` are recorded, and the workspace."""
    repo, workspace = root / "repo", root / "ws"
    files = {
        "pkg/__init__.py": "",
        "pkg/mod.py": "def f():\n    return 1\n",
        "tests/test_a.py": "from pkg.mod import f\n\n\ndef test_f():\n    assert f() == 1\n",
        "tests/test_b.py": "def test_alone():\n    assert True\n",
        "tests/test_broken.py": "def test_broken():\n    assert False, 'broken before the run'\n",
    }
    for path, content in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(content, encoding="utf-8")
    (repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD")
    cache = workspace / tc.CACHE_DIRNAME
    _run(f"{sys.executable} {drv.__file__} run --cache {cache} --repo {repo} tests/test_a.py "
         "tests/test_b.py -- -q -p no:cacheprovider")  # fmt: skip
    return repo, workspace, base


def _check(repo: Path, workspace: Path, head: str, root: Path, base=None, files=None, **more):
    ran = []

    @contextlib.contextmanager
    def tree_at(commit):
        tree = root / f"tree-{commit[:8]}-{len(ran)}"
        _git(repo, "worktree", "add", "-q", "--detach", str(tree), commit)
        try:
            yield tree
        finally:
            _git(repo, "worktree", "remove", "--force", str(tree))

    def run_job(command, tree, timeout_s=None):
        ran.append(command)
        return _run(command)

    result = regression.run_final_tests(
        workspace=workspace,
        repo=repo,
        head=head,
        base=base,
        files=tc.list_test_files(repo, head, ["tests"]) if files is None else files,
        tree_at=tree_at,
        run_job=run_job,
        python=sys.executable,
        paths=["tests"],
        **more,
    )
    return result, ran


_ACCURACY_TEST = """from pkg.harness import LlmapiAccuracyTestHarness


class TestTarget(LlmapiAccuracyTestHarness):
    def test_gsm8k(self):
        raise AssertionError("ran: a load and 1319 questions")

    def test_gsm8k_fp8(self):
        raise AssertionError("ran: another load")
"""


@_requires_git
def test_an_accuracy_test_of_the_engines_harness_is_collected_and_left_to_ci(tmp_path):
    repo, workspace, base = _regression_repo(tmp_path)
    scored = "ci/test_target_accuracy.py"
    head = _commit(
        repo,
        base,
        {"pkg/harness.py": "class LlmapiAccuracyTestHarness:\n    pass\n", scored: _ACCURACY_TEST},
    )
    files = ["tests/test_a.py", scored]
    # Told by the class it is built on: the plain test is not one.
    assert list(regression.accuracy_tests(repo, head, files)) == [scored]
    measured = regression.measured_by_sessions(_ACCURACY_TEST, {"gsm8k": ["tp1", "kv_fp8"]})
    assert measured == "the session measured `gsm8k` on `tp1`, `kv_fp8`"
    assert regression.measured_by_sessions(_ACCURACY_TEST, {"mmlu": ["tp1"]}) == ""
    result, ran = _check(repo, workspace, head, tmp_path, base, files, ci={scored: measured})
    # Imported and listed in the one test job, before the rest runs; none of its cases ran
    # (each would fail here), and it fails nothing: the session measured that benchmark.
    assert result.verdict == "CLEAN" and result.passed and result.jobs == 1
    assert result.left == [f"{scored} (2 case(s)): left to CI: {measured}"]
    assert "1 accuracy test file(s) collected and left to CI" in result.summary()
    assert len(ran) == 2 and ran[1].startswith("( ") and ran[1].count(" --collect-only") == 1
    assert ran[1].index("--collect-only") < ran[1].index(" ; ")
    # Again on the same candidate: collected already, nothing is asked of a job but the probe.
    again, ran = _check(repo, workspace, head, tmp_path, base, files, ci={scored: measured})
    assert again.left == result.left and len(ran) == 1
    # No session of this run measured a benchmark it names: not measured here, not verified.
    unmeasured, _ = _check(repo, workspace, head, tmp_path, base, files, ci={scored: ""})
    assert unmeasured.verdict == regression.NOT_VERIFIED and unmeasured.left == []
    assert unmeasured.unverified == [
        f"{scored} (2 case(s)): left to CI, not measured here: no session of this run "
        "measured a benchmark it names"
    ]
    # One that does not import fails the final tests, run or not.
    broken = _commit(repo, head, {scored: "import no_such_module\n" + _ACCURACY_TEST}, "broken")
    failed, _ = _check(repo, workspace, broken, tmp_path, base, files, ci={scored: measured})
    assert failed.verdict == "REGRESSED" and failed.left == []
    assert failed.failures[0].startswith(f"{scored} (does not collect at the candidate)")


@_requires_git
def test_a_file_that_ran_where_git_gave_no_answer_counts_in_a_pin_and_is_never_no_result(
    tmp_path, monkeypatch
):
    """Whole runs lost passing files to "no result": git status ran out of time in a pin's tree."""
    repo, workspace, base = _regression_repo(tmp_path)
    # A git that cannot say whether the tree is clean, as one cut at its limit: every
    # other question is the real git's.
    real = shutil.which("git")
    fake = tmp_path / "bin" / "git"
    fake.parent.mkdir()
    fake.write_text(f'#!/bin/sh\ncase " $* " in *" status "*) exit 1;; esac\nexec {real} "$@"\n')
    fake.chmod(0o755)
    ran = []

    def final_tests(pin: bool):
        @contextlib.contextmanager
        def tree_at(commit):
            tree = tmp_path / f"tree-{len(ran)}"
            _git(repo, "worktree", "add", "-q", "--detach", str(tree), commit)
            if pin:  # as the run leaves a pin's tree: its marker names the commit
                (tree / drv.PIN_MARKER).write_text(json.dumps({"commit": commit}))
            try:
                yield tree
            finally:
                _git(repo, "worktree", "remove", "--force", str(tree))

        def run_job(command, tree, timeout_s=None):
            ran.append(command)
            return _run(f"PATH={fake.parent}:$PATH {command}")

        return regression.run_final_tests(
            workspace=workspace, repo=repo, head=base, base=None, files=["tests/test_b.py"],
            tree_at=tree_at, run_job=run_job, python=sys.executable, paths=["tests"],
            name="pin" if pin else "checkout",
        )  # fmt: skip

    # The tree is no pin: what ran there is not a result, and it is not "did not run"
    # either. One test job, not a second one to learn the same.
    for record in (workspace / tc.CACHE_DIRNAME / drv.SESSIONS_DIRNAME).glob("*.json"):
        record.unlink()  # what the fixture recorded at the base, with a git that answers
    unknown = final_tests(pin=False)
    assert unknown.verdict == "INCOMPLETE" and len(ran) == 2, "the probe and one test job"
    (line,) = unknown.failures
    assert line.startswith("tests/test_b.py (ran at this commit (1 passed) and is not counted")
    assert "git gave no answer on whether the tree it ran in was clean" in line
    (record,) = (workspace / tc.CACHE_DIRNAME / drv.SESSIONS_DIRNAME).glob("*.json")
    kept = json.loads(record.read_text())
    assert kept["clean"] is None and kept["pinned"] is False
    # A pin was made from the commit and checked when its job started: its result counts.
    pinned = final_tests(pin=True)
    assert pinned.verdict == "CLEAN" and pinned.held == 1
    # The question itself leaves a file's change time out: a pin's tree is made read-only
    # after its index is written, and git would otherwise read every file again each time.
    asked = []
    monkeypatch.setattr(drv, "_git", lambda repo, *args: asked.append(args) or "")
    drv.tree_state(str(repo))
    assert asked[-1][:3] == ("-c", "core.trustctime=false", "status")
    # A tree git called modified never counts, pin or not.
    assert not tc.Session("c", False, "e", {}, {}, pinned=True).counts
    assert tc.Session("c", None, "e", {}, {}, pinned=True).counts
    assert not tc.Session("c", None, "e", {}, {}).counts


@_requires_git
def test_a_test_job_that_ended_without_a_result_says_what_it_said_last(tmp_path):
    """A run named its workspace `ws`: the driver's job found no list in its tree on the node."""
    repo, workspace, base = _regression_repo(tmp_path)
    head = _commit(repo, base, {"tests/test_new.py": "def test_new():\n    assert True\n"})
    died = (
        'Traceback (most recent call last):\n  File "driver.py", line 1, in <module>\n'
        "FileNotFoundError: [Errno 2] No such file or directory: 'ws/regression/x.collect.txt'\n\n"
    )

    @contextlib.contextmanager
    def tree_at(commit):
        yield repo

    def run_job(command, tree, timeout_s=None):
        # The probe answers; the job that was to run the tests ends in its driver's traceback.
        return _run(command) if "--environment" in command else (1, died)

    result = regression.run_final_tests(
        workspace=workspace, repo=repo, head=head, base=base, files=["tests/test_new.py"],
        tree_at=tree_at, run_job=run_job, python=sys.executable, paths=["tests"],
    )  # fmt: skip
    # No case failed and none passed: the verdict is not a regression, and the result
    # keeps the job's own last line for whoever is asked.
    assert result.verdict == "INCOMPLETE" and result.held == 0
    assert result.job_said == died.strip().splitlines()[-1]
    assert result.job_said.startswith("FileNotFoundError") and not result.passed
    # A result that is not INCOMPLETE quotes no job: there is nothing to look for.
    clean, _ = _check(repo, workspace, base, tmp_path, base, ["tests/test_b.py"])
    assert clean.verdict == "CLEAN" and clean.job_said == ""


@_requires_git
def test_a_case_cut_at_the_drivers_limit_says_whose_limit_it_was(tmp_path):
    repo, workspace, base = _regression_repo(tmp_path)
    # As pytest-timeout fails a case at the limit the driver gives one that states none,
    # and at a limit of the case's own.
    slow = (
        "import pytest\n\n\ndef test_load():\n"
        "    pytest.fail('Timeout (>180.0s) from pytest-timeout.')\n\n\n"
        "def test_own_limit():\n    pytest.fail('Timeout (>600.0s) from pytest-timeout.')\n"
    )
    head = _commit(repo, base, {"tests/test_slow.py": slow})
    result, _ = _check(repo, workspace, head, tmp_path, base, ["tests/test_slow.py"])
    assert result.verdict == "REGRESSED"
    cut = next(line for line in result.failures if "test_load" in line)
    assert "(cut at 180 s, the limit a case gets when it states none of its own)" in cut
    own = next(line for line in result.failures if "test_own_limit" in line)
    assert "cut at 180 s" not in own and "Timeout (>600.0s)" in own
    assert tc.cut_at_default("Failed: Timeout >180.0s") and not tc.cut_at_default("Timeout >1800s")


@_requires_git
def test_environment_change_cannot_reuse_a_pass_and_probe_failure_is_incomplete(
    tmp_path, monkeypatch
):
    repo, workspace, base = _regression_repo(tmp_path)
    monkeypatch.setenv("TRTLLM_TEST_CACHE_ENV", "A")
    both = ["tests/test_a.py", "tests/test_b.py"]
    first, _ = _check(repo, workspace, base, tmp_path, base, both)
    assert first.verdict == "CLEAN"
    a = tc.recorded(workspace / tc.CACHE_DIRNAME, base).env_digest
    monkeypatch.setenv("TRTLLM_TEST_CACHE_ENV", "B")
    second, ran = _check(repo, workspace, base, tmp_path, base, both)
    assert second.verdict == "CLEAN" and len(ran) == 2
    assert tc.recorded(workspace / tc.CACHE_DIRNAME, base).env_digest != a
    results, jobs, code, tail = regression.record_at(
        base,
        ["tests/test_a.py"],
        cache_dir=workspace / tc.CACHE_DIRNAME,
        out=tmp_path / "failed-probe",
        tree_at=lambda commit: contextlib.nullcontext(repo),
        run_job=lambda command, tree, timeout_s=None: (1, "unavailable"),
        python=sys.executable,
    )
    assert not results.env_digest and results.unrun(["tests/test_a.py"])
    assert jobs == 0 and code == 1 and "environment probe failed" in tail


@_requires_git
def test_a_failing_test_that_also_fails_at_the_base_is_reported_so_and_fails_nothing(tmp_path):
    repo, workspace, base = _regression_repo(tmp_path)
    cache = workspace / tc.CACHE_DIRNAME
    (repo / "pkg" / "mod.py").write_text(
        "def f():\n    return 1  # same answer\n", encoding="utf-8"
    )
    _git(repo, "commit", "-qam", "shared change")
    head = _git(repo, "rev-parse", "HEAD")

    result, ran = _check(repo, workspace, head, tmp_path, base)

    # ``test_broken`` fails on the candidate. The base had no result for it, so it ran
    # there too, and it is reported with both results. It fails nothing.
    assert result.verdict == "CLEAN" and result.passed and result.failures == []
    (line,) = result.at_base
    assert line.startswith(
        "tests/test_broken.py::test_broken (fails at the candidate and at the base)"
    )
    assert "broken before the run" in line and "1 fail(s) at the base too" in result.summary()
    assert tc.recorded(cache, head).result("tests/test_broken.py::test_broken").outcome == "failed"
    assert tc.recorded(cache, base).result("tests/test_broken.py::test_broken").outcome == "failed"
    assert len(ran) == 3, "the probe, one test job on the candidate, the failing case at the base"
    # The same candidate again: every result is there, and only the probe runs.
    again, ran = _check(repo, workspace, head, tmp_path, base)
    assert again.verdict == "CLEAN" and again.at_base == result.at_base and len(ran) == 1
    # A later candidate does not pay for it again: a case known to fail at the base cannot
    # regress, so it is not run there, and the report says that it was not.
    (repo / "pkg" / "mod.py").write_text("def f():\n    return 1  # again\n", encoding="utf-8")
    _git(repo, "commit", "-qam", "a repair")
    later = _git(repo, "rev-parse", "HEAD")
    repaired, ran = _check(repo, workspace, later, tmp_path, base)
    assert repaired.verdict == "CLEAN" and len(ran) == 2
    assert tc.recorded(cache, later).result("tests/test_broken.py::test_broken") is None
    assert repaired.at_base == [
        "tests/test_broken.py::test_broken (fails at the base; not run on the candidate: it "
        "cannot regress)"
    ]


@_requires_git
def test_a_test_that_passed_at_the_base_and_fails_on_the_candidate_fails_the_final_tests(tmp_path):
    repo, workspace, base = _regression_repo(tmp_path)
    (repo / "pkg" / "mod.py").write_text("def f():\n    return 2\n", encoding="utf-8")
    (repo / "tests" / "test_added.py").write_text("def test_added():\n    assert 0\n", "utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "a change that breaks a test")
    head = _git(repo, "rev-parse", "HEAD")
    result, _ = _check(repo, workspace, head, tmp_path, base)
    assert result.verdict == "REGRESSED" and not result.passed
    said = "\n".join(result.failures)
    assert "tests/test_a.py::test_f (fails at the candidate; passed at the base)" in said
    assert (
        "tests/test_added.py::test_added (fails at the candidate; the base has no such test)"
        in said
    )
    assert len(result.at_base) == 1 and "test_broken" in result.at_base[0]
    # With no base at all, each failure is the candidate's own.
    alone, _ = _check(repo, workspace, head, tmp_path, None, ["tests/test_a.py"])
    assert alone.verdict == "REGRESSED" and "no base to compare with" in alone.failures[0]


@_requires_git
@pytest.mark.parametrize("with_base", [True, False])
def test_a_selection_of_which_every_test_was_skipped_is_not_verified(tmp_path, with_base):
    """Nothing failed and nothing passed: that certifies nothing, and it is not CLEAN."""
    repo, workspace, base = _regression_repo(tmp_path)
    (repo / "tests" / "test_skipped.py").write_text(
        "import pytest\n\n\n@pytest.mark.skip(reason='needs four devices')\n"
        "def test_wide():\n    assert False\n",
        encoding="utf-8",
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "a test that skips itself")
    head = _git(repo, "rev-parse", "HEAD")
    only = ["tests/test_skipped.py"]
    result, _ = _check(repo, workspace, head, tmp_path, base if with_base else None, only)
    assert result.verdict == regression.NOT_VERIFIED and result.partial and not result.passed
    assert result.failures == []
    assert result.unverified == ["every test that ran was skipped (1 skipped, none passed)"]
    assert "not verified: every test that ran was skipped" in result.summary()
    # One test that passes beside it: the selection is verified.
    mixed, _ = _check(repo, workspace, head, tmp_path, base, [*only, "tests/test_b.py"])
    assert mixed.verdict == "CLEAN" and mixed.unverified == []


@_requires_git
def test_the_final_selection_is_read_off_the_change_and_the_tasks_not_off_imports(tmp_path):
    repo = tmp_path / "repo"
    tests = regression.PACKAGE_TESTS[0]
    own = ("pkg/models/fam/target", f"{tests}/models/fam/target")
    files = {
        "pkg/models/fam/target/modeling.py": "X = 1\n",
        "pkg/shared/norm.py": "N = 1\n",
        "pkg/shared/other.py": "O = 1\n",
        "docs/guide.md": "words\n",
        "tests/integration/test_lists/qa.yml": "- a\n",
        f"{tests}/models/fam/target/test_modeling_v2_target.py": "",
        f"{tests}/shared/test_modeling_v2_norm.py": "",
        f"{tests}/shared/test_modeling_v2_other.py": "",
        f"{tests}/catalog/test_modeling_v2_gemm.py": "",
    }
    for path, content in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(content, encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    base = _git(repo, "rev-parse", "HEAD")
    target = f"{tests}/models/fam/target/test_modeling_v2_target.py"
    norm = f"{tests}/shared/test_modeling_v2_norm.py"
    suite = sorted(p for p in files if p.startswith(tests))
    # The target alone changed, with documentation and a test list: the target's tests.
    head = _commit(repo, base, {"pkg/models/fam/target/modeling.py": "X = 2\n",
                                "docs/guide.md": "more words\n",
                                "tests/integration/test_lists/qa.yml": "- a\n- b\n"})  # fmt: skip
    picked, why = regression.select(repo, base, head, own=own)
    assert picked == [target] and why.startswith("1 test file(s): the target's")
    # A changed integration test is CI's to run (it needs CI's models): the selection says
    # that it was left, and which.
    head = _commit(repo, base, {"tests/integration/defs/test_e2e.py": "def test_e2e(): ...\n"},
                   "an integration test")  # fmt: skip
    picked, why = regression.select(repo, base, head, own=own)
    assert picked == [target]
    assert "1 changed test file(s) under tests/integration/ are CI's to run" in why
    assert "tests/integration/defs/test_e2e.py" in why
    assert regression.missing_declared(repo, head, [target, "tests/x/test_gone.py", "pkg/"]) == [
        "tests/x/test_gone.py"
    ]
    # A shared file a task with tests of its own was given: those tests are its reach.
    head = _commit(repo, base, {"pkg/shared/norm.py": "N = 2\n"}, "norm")
    picked, why = regression.select(
        repo, base, head, own=own, obligations=[norm], covered=["pkg/shared/norm.py"]
    )
    assert picked == [target, norm] and "collected suite" not in why
    # A test file the change itself changed is selected, wherever it is.
    head = _commit(
        repo, base, {f"{tests}/shared/test_modeling_v2_other.py": "# changed\n"}, "a test"
    )
    picked, _ = regression.select(repo, base, head, own=own)
    assert picked == [target, f"{tests}/shared/test_modeling_v2_other.py"]
    # A shared file no such task was given: its reach is unknown, and the package's
    # collected suite runs, once. So it does with no base to read the change from.
    head = _commit(
        repo, base, {"pkg/shared/norm.py": "N = 2\n", "pkg/shared/other.py": "O = 2\n"}, "two"
    )
    picked, why = regression.select(
        repo, base, head, own=own, obligations=[norm], covered=["pkg/shared/norm.py"]
    )
    assert picked == suite
    assert "collected suite, once" in why and "pkg/shared/other.py is unknown" in why
    picked, why = regression.select(repo, None, head, own=own)
    assert picked == suite and "no base" in why


@_requires_git
def test_a_module_that_skips_itself_on_import_has_a_result_not_a_hole(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tests").mkdir(parents=True)
    (repo / "tests" / "test_other_arch.py").write_text(
        "import pytest\n\npytest.skip('certified on sm_100 only', allow_module_level=True)\n\n\n"
        "def test_never():\n    assert False\n",
        encoding="utf-8",
    )
    (repo / ".gitignore").write_text("__pycache__/\n", encoding="utf-8")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "t@example.com")
    _git(repo, "config", "user.name", "t")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-qm", "base")
    head = _git(repo, "rev-parse", "HEAD")
    cache = tmp_path / "ws" / tc.CACHE_DIRNAME
    code, _ = _run(
        f"{sys.executable} {drv.__file__} run --cache {cache} --repo {repo} "
        "tests/test_other_arch.py -- -q -p no:cacheprovider"
    )
    assert code == 0
    results = tc.recorded(cache, head)
    assert results.unrun(["tests/test_other_arch.py"]) == []
    skip = results.result("tests/test_other_arch.py")
    assert skip.outcome == "skipped" and "sm_100 only" in skip.message
    report = tc.compare(
        results, results, ["tests/test_other_arch.py"], ["tests/test_other_arch.py"]
    )
    assert report.verdict == "CLEAN" and report.not_run == []


# ----------------------------------------------------------------------------------------
# One session per deployment: what the separate boot, accuracy, serve, acceptance and
# disaggregated gates held, held by the checks of a session (`checks/session.py`).
# The session's job is a stand-in here (`_helpers.session_stdout`); the servers
# themselves are stood in for in `test_session.py`.

_SESSION_OPTIONS = {"max_seq_len": 64, "max_batch_size": 8, "max_num_tokens": 32}


def _session_task(tmp_path, *deployments, **accuracy) -> dict:
    tree = tmp_path / "tree"
    (tree / "tensorrt_llm").mkdir(parents=True, exist_ok=True)
    (tmp_path / "ckpt").mkdir(exist_ok=True)
    task = {
        "checkpoint_path": str(tmp_path / "ckpt"),
        "trtllm_repo_path": str(tree),
        "environment": {"kind": "local", "python": "/py", "gpus_per_node": 8},
        "deployments": list(deployments) or [{"name": "core", "options": dict(_SESSION_OPTIONS)}],
    }
    if accuracy:
        task["accuracy"] = accuracy
    return task


def _session(tmp_path, task, name=None, *, change=None, score=90.0, **more):
    from agent_flow.workflows.trtllm_modeling_v2.spec import deployments

    seen: list[str] = []
    inner = passing_shell(score=score, change=change)

    def runner(command, env=None, timeout=0):
        seen.append(command)
        return inner(command, env, timeout)

    found = session_mod.run_deployment(
        deployments.named(task, name),
        RunEnvironment(kind="local", python="/py"),
        task_data=task,
        workdir=tmp_path / "gates",
        runner=runner,
        **more,
    )
    return found, seen


def _line(found, name):
    return next(line for line in found.lines if line.name == name)


def test_the_session_sends_the_api_prompts_and_names_the_deployments_shape(tmp_path):
    task = _session_task(tmp_path, {"name": "core", "options": {"tensor_parallel_size": 2}})
    found, seen = _session(tmp_path, task)
    assert found.result.passed, found.render()
    assert found.result.summary().startswith("[PASS] deployment@tp2: 4 check(s) passed")
    steps = {s["id"]: s for s in session_spec(seen[0])["steps"]}
    assert steps["api.chat"]["requests"][0]["endpoint"] == "/v1/chat/completions"
    assert steps["api.stream"]["request"]["endpoint"] == "/v1/completions"
    # The job holds the deployment's devices; the session keeps what it ran beside its files.
    assert session_spec(seen[0])["instances"][0]["ranks"] == 2
    assert (tmp_path / "gates" / "core" / "session.out").read_text().strip()

    def silent(marker, payload):
        if marker == workload_io.RECORD_MARKER and payload["step"] == "api.stream":
            payload["done"] = False
        return payload

    found, _ = _session(tmp_path, task, change=silent)
    assert not _line(found, "api").passed and "never its end marker" in _line(found, "api").detail


NODE = "tests/integration/defs/accuracy/test_disaggregated_serving.py::TestStep5::test_dep4"


PASS = '<testsuite><testcase name="native_hop"/></testsuite>'


@pytest.mark.parametrize(
    "written,code,expected",
    [
        (PASS, 0, True),
        ("<testsuite><testcase><skipped/></testcase></testsuite>", 0, False),
        ("<testsuite><testcase><failure/></testcase></testsuite>", 0, False),
        (None, 0, False),
        ("<broken", 0, False),
        (PASS, 7, False),
    ],
)
def test_native_tests_need_a_fresh_successful_report(tmp_path, written, code, expected):
    work = tmp_path / "gate"
    work.mkdir()
    report = work / "tests.pd3.0.xml"
    report.write_text(PASS)
    seen = []

    def runner(command, env, timeout):
        seen.append(command)
        assert not report.exists(), "a stale report must be removed"
        if written is not None:
            report.write_text(written)
        return CommandResult(returncode=code)

    result = native_tests.run_native_tests(
        RunEnvironment(kind="slurm", python="python3"),
        task_data={"trtllm_repo_path": str(tmp_path)},
        tests=[NODE],
        workdir=work,
        runner=runner,
        member="pd3",
        env={"UCX_TLS": "^ib"},
    )
    assert result.passed == expected and result.name == "tests" and result.member == "pd3"
    # One process from the tree, with the deployment's environment and the switch on.
    assert "UCX_TLS='^ib'" in seen[0] or "UCX_TLS=^ib" in seen[0]
    assert "TRTLLM_MODELING_V2=require" in seen[0] and "-m pytest" in seen[0]
    if code:
        assert "exited 7" in result.detail


def test_a_check_with_no_native_test_has_nothing_to_run_which_is_no_failure(tmp_path):
    result = native_tests.run_native_tests(
        RunEnvironment(kind="local", python="python3"),
        task_data={"trtllm_repo_path": str(tmp_path)},
        tests=[],
        workdir=tmp_path,
        runner=lambda *args: pytest.fail("nothing is run"),
        name="cpu:pd-state",
    )
    assert result.passed and result.summary().startswith("[PASS] cpu:pd-state: [NOTHING TO RUN]")


def test_a_native_test_that_asks_for_ranks_of_its_own_is_not_run_and_not_verified(tmp_path):
    file = "tests/test_transfer.py"
    (tmp_path / "tests").mkdir()
    (tmp_path / file).write_text(
        "import pytest\nfrom other_tests import pool_of_two as borrowed\n\n\n"
        "def test_in_one_process():\n    pass\n\n\n"
        "def test_with_a_fixture_of_another_file(borrowed: MPIPoolExecutor):\n    pass\n\n\n"
        "def test_that_makes_its_own():\n    with MPIPoolExecutor(max_workers=2):\n        pass\n\n\n"
        '@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)\n'
        "def test_across_two_ranks(mpi_pool_executor):\n    pass\n\n\n"
        '@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)\n'
        "class TestWorld:\n    def test_inside(self, mpi_pool_executor):\n        pass\n"
    )
    one, two = f"{file}::test_in_one_process", f"{file}::test_across_two_ranks[2]"
    more = (
        "TestWorld::test_inside",
        "test_with_a_fixture_of_another_file",
        "test_that_makes_its_own",
    )
    asks = [
        native_tests.needs_world(tmp_path, t) for t in (one, two, *(f"{file}::{m}" for m in more))
    ]
    # Read in the test's own source: its arguments and their annotations, its decorators,
    # its class's, its body.
    assert asks == [False, True, True, True, True]
    # A whole file asks where any of its tests does; one that is not there says nothing.
    assert native_tests.needs_world(tmp_path, file)
    assert not native_tests.needs_world(tmp_path, "tests/test_absent.py::test_x")
    seen = []

    def runner(command, env, timeout):
        seen.append(command)
        Path(re.search(r"--junitxml=([^\s']+)", command).group(1)).write_text(PASS)
        return CommandResult(returncode=0)

    def run(tests):
        return native_tests.run_native_tests(
            RunEnvironment(kind="local", python="python3"),
            task_data={"trtllm_repo_path": str(tmp_path)},
            tests=tests,
            workdir=tmp_path / "gate",
            runner=runner,
        )

    # The one that runs in one process is run; the other is not started, where it would
    # wait for its ranks to the check's wall. Nothing failed, and the line says what is
    # not verified.
    result = run([one, two])
    assert len(seen) == 1 and "test_in_one_process" in seen[0] and "two_ranks" not in seen[0]
    assert result.passed and result.unverified.startswith(f"{two}: not run. Its source asks for")
    assert "starts no MPI world" in result.unverified and "not verified" in result.detail
    assert run([two]).unverified and len(seen) == 1


def _cli_task(tmp_path, **changes):
    repo = tmp_path / "repo"
    (repo / "tensorrt_llm").mkdir(parents=True, exist_ok=True)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(exist_ok=True)
    (ckpt / "config.json").write_text(json.dumps({"architectures": ["MForCausalLM"]}))
    moves = {"cache_transceiver_config": {"backend": "DEFAULT"}}
    data = {
        "checkpoint_path": str(ckpt),
        "trtllm_repo_path": str(repo),
        "arch": "sm_100",
        "environment": {"kind": "local", "python": "/usr/bin/python3"},
        "accuracy": {"benchmark": "gsm8k"},
        "deployments": [
            {
                "name": "tep2",
                "options": {"tensor_parallel_size": 2, "moe_expert_parallel_size": 2},
                "benchmarks": [{"name": "gsm8k", "anchor": 92.0, "source": "measured"}],
            },
            {
                "name": "switched",
                "options": {"tensor_parallel_size": 2, "moe_expert_parallel_size": 2},
                "env": {"MY_SWITCH": "1"},
                "relative_to": "tep2",
            },
            {
                "name": "pd",
                "ctx": {"options": moves},
                "gen": {"options": moves},
                "benchmarks": [],
                "tests": [NODE],
            },
        ],
        **changes,
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture
def shell(monkeypatch):
    """Replace the executor-backed runner with a stand-in that answers every job."""
    state = {"seen": [], "junit": None, "score": 90.0}

    def fake_runner_for(executor, **kwargs):
        def run(command, env=None, timeout=0):
            state["seen"].append(command)
            report = re.search(r"--junitxml=([^\s']+)", command)
            if report and state["junit"] is not None:
                Path(report.group(1)).write_text(state["junit"], encoding="utf-8")
            return passing_shell(score=state["score"])(command, env, timeout)

        return run

    monkeypatch.setattr(gate_cli, "shell_runner_for", fake_runner_for)
    return state


def test_a_relative_deployment_is_judged_against_the_others_score_on_record(
    tmp_path, shell, capsys
):
    path = _cli_task(tmp_path)
    work = ["--workdir", str(tmp_path / "w")]
    argv = ["deployment", "--task", str(path), "--deployment", "switched", *work]
    # With no score of `tep2` on record the benchmark is blocked before the load, and said.
    assert gate_cli.main(argv) == 1
    out = capsys.readouterr().out
    assert out.startswith("check deployment:switched\n")
    assert "[BLOCKED] accuracy[gsm8k]@tep2: its anchor is the score of `tep2`" in out
    assert "`gate deployment --deployment tep2`" in out
    assert "[PASS] boot@tep2" in out and "[BLOCKED] deployment@tep2" in out
    # The other deployment's session leaves its line in the run's ledger, on this tree.
    status = contract_mod.ContractStatus.load(tmp_path)
    scored = check_result.GateResult(
        name="accuracy[gsm8k]", member="tep2", passed=True, detail="", measured=92.0
    )
    status.record("tep2", scored)
    shell["score"] = 85.0
    assert gate_cli.main(argv) == 1
    out = capsys.readouterr().out
    assert "[FAIL] accuracy[gsm8k]@tep2: measured 85.0000 vs floor" in out
    assert "the unit's own score 92.0000" in out
    shell["score"] = 91.5
    assert gate_cli.main(argv) == 0
    # Each run is one job: the deployment's own environment goes with its servers.
    spec = session_spec(shell["seen"][-1])
    assert spec["env"] == {"MY_SWITCH": "1"} and len(shell["seen"]) == 3
    # A score in another form is quoted for whoever decides how it is read, then read by pattern.
    printed = "Partial scores after 100/100 responses\nJSON Mode Eval accuracy: 91.50 (100)\n"
    assert accuracy_checks.parse_eval_score(printed) is None
    assert "  JSON Mode Eval accuracy: 91.50 (100)" in accuracy_checks.unparsed_score_detail(
        printed, None
    )
    assert accuracy_checks.parse_eval_score(printed, r"Eval accuracy: ([0-9.]+)") == 91.5


def test_the_front_door_prints_the_verdict_alone_on_standard_output(tmp_path, capsys, monkeypatch):
    """`gate ... --json > file` is JSON: a pool notice said meanwhile goes to standard error."""
    import threading

    path = _cli_task(tmp_path)
    notice = (
        "POOL_DEVICE_HELD device 0 on nodeA holds 3655 MiB that no job of its watcher accounts for"
    )

    def ran(deployment, environment, **kwargs):
        # As the pool's keeper does on its own thread while a check waits, and as any
        # code of the run might: a line on standard output.
        said = threading.Thread(target=lambda: print(notice))
        said.start()
        said.join()
        print(f"waiting: {deployment.name} is queued")
        line = check_result.verdict_factory("clean")(
            name="deployment", member="tep2", passed=True, detail="ok"
        )
        return session_mod.SessionResult((), line)

    monkeypatch.setattr(session_mod, "run_deployment", ran)
    work = ["--workdir", str(tmp_path / "w")]
    assert gate_cli.main(["deployment", "--task", str(path), *work, "--json"]) == 0
    out, err = capsys.readouterr()
    payload = json.loads(out)
    assert payload["id"] == "deployment:tep2" and payload["passed"] is True
    assert notice in err and "waiting: tep2 is queued" in err
    assert "POOL_" not in out and "waiting" not in out
    # Without --json: the id and the lines, and still nothing else.
    assert gate_cli.main(["deployment", "--task", str(path), *work]) == 0
    out, err = capsys.readouterr()
    assert out.splitlines()[0] == "check deployment:tep2" and len(out.splitlines()) == 2
    assert notice in err


def test_the_front_door_routes_a_kernel_row_to_its_native_tests(tmp_path, shell, capsys):
    path = _cli_task(tmp_path, target={"family": "fam", "checkpoint_segment": "m"})
    repo = tmp_path / "repo"
    work = ["--workdir", str(tmp_path / "w")]
    shell["junit"] = PASS
    # kernel: a row of the survey's vocabulary, by its number or its entry, on one device.
    doc = {
        "steps": [],
        "queue": [{"n": 1, "mode": "onboard", "target": "ops/norm/rms"}],
    }
    (tmp_path / "vocabulary.yaml").write_text(yaml.safe_dump(doc))
    kernel = ["kernel", "--task", str(path), *work]
    assert gate_cli.main([*kernel, "--row", "1"]) == 1
    assert "row 1 (ops/norm/rms) has no test in the tree it measures" in capsys.readouterr().out
    target = f"{catalog_paths.MODELING_V2_PACKAGE}/models/fam/m__sm_100__unified"
    test = repo / catalog_paths.model_ops_tests_dir(target) / "norm" / "test_modeling_v2_rms.py"
    test.parent.mkdir(parents=True)
    test.write_text("def test_rms():\n    pass\n")
    before = len(shell["seen"])
    assert gate_cli.main([*kernel, "--row", "ops/norm/rms"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("check kernel:ops/norm/rms\n[PASS] kernel:ops/norm/rms: ")
    assert len(shell["seen"]) == before + 1 and "test_modeling_v2_rms.py" in shell["seen"][-1]
    assert gate_cli.main([*kernel, "--row", "7"]) == 2
    assert "names no row of" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        gate_cli.main(["kernel", "--task", str(path)])


_BASE = [2.6 + 0.5 * math.sin(i) for i in range(32)]


def _drafted(rates, *, steps=99):
    """The answers of an acceptance workload whose prompts accept at ``rates`` per iteration."""
    return [
        {
            "id": f"p{n}",
            "status": 200,
            "error": "",
            "speculative_decoding": {
                "num_spec_steps": steps,
                "total_accepted_draft_tokens": int(round(rate * (steps + 1))) - (steps + 1),
            },
        }
        for n, rate in enumerate(rates)
    ]


def _acceptance_task(**acceptance):
    return {"accuracy": {"acceptance": {"prompts": [f"q{n}" for n in range(32)], **acceptance}}}


def test_a_drafters_reading_is_held_to_its_floor_and_a_refused_workload_is_not_the_targets(
    tmp_path,
):
    from agent_flow.workflows.trtllm_modeling_v2.spec.deployments import Deployment, Server

    mtp = Deployment("mtp", server=Server({}))
    task = _acceptance_task(mtp={"min_absolute": 1.5, "floor_source": "a measured drafter"})
    workload = acceptance_checks.resolve_acceptance_workload(task["accuracy"], "mtp")
    held = acceptance_checks.judge_session(task, mtp, workload, _drafted(_BASE))
    assert held["passed"] and "floor 1.50" in held["detail"] and not held["warnings"]
    # A floor with no measurement behind it is used, and every verdict says so.
    bare = _acceptance_task(mtp=2.0)
    said = acceptance_checks.judge_session(bare, mtp, workload, _drafted(_BASE))
    assert said["passed"] and "not tied to a measured reference" in said["warnings"][0]
    # Too few prompts, or answers that end before drafts are verified, fail as too small.
    few = acceptance_checks.judge_session(task, mtp, workload, _drafted(_BASE[:4]))
    assert not few["passed"] and "workload too small to compare acceptance" in few["detail"]
    short = acceptance_checks.judge_session(task, mtp, workload, _drafted(_BASE, steps=0))
    assert not short["passed"] and "workload too short to measure acceptance" in short["detail"]
    # A prompt the engine refuses for its length fails the workload, not the target.
    error = "RequestError: The prompt length (5000.0) should not exceed max_num_tokens (4096)"
    refused = [{"id": "p0", "status": 400, "error": error}]
    result = acceptance_checks.judge_session(task, mtp, workload, refused)
    assert "refused a prompt of 5000 tokens, longer than max_num_tokens 4096" in result["detail"]
    assert "a property of the workload, not of the target" in result["detail"]
    # Answers that do not say how they were drafted are unknown, which is not a pass.
    mute = [{"id": "p0", "status": 200, "error": ""}]
    unknown = acceptance_checks.judge_session(task, mtp, workload, mute)
    assert not unknown["passed"] and "unknown is not a pass" in unknown["detail"]


def test_a_frozen_anchor_is_changed_only_by_an_explicit_change_and_a_session_is_judged_by_it(
    tmp_path,
):
    scored = [{"name": "gsm8k", "anchor": 10.0, "source": "a placeholder"}]
    task = _session_task(tmp_path, {"name": "core", "options": {}, "benchmarks": scored})
    workspace = tmp_path / "ws"
    anchors.freeze(workspace, task)
    found, _ = _session(tmp_path, task, score=41.0, workspace=workspace)
    first = _line(found, "accuracy[gsm8k]")
    assert first.passed and first.reference == 10.0
    # A run never moves an anchor by itself: the score it measured replaces the
    # placeholder when someone says so, with the reason.
    assert anchors.frozen(workspace, "core", "gsm8k").value == 10.0
    anchors.change(
        workspace,
        "core",
        "gsm8k",
        value=93.3,
        source="the stock path, one node",
        reason="the placeholder is replaced",
    )
    found, _ = _session(tmp_path, task, score=41.0, workspace=workspace)
    second = _line(found, "accuracy[gsm8k]")
    assert not second.passed and second.reference == 93.3
    assert "anchor: declared, the stock path, one node" in second.detail


def test_an_attention_dp_rank_serves_its_share_of_the_global_batch():
    from agent_flow.workflows.trtllm_modeling_v2.spec import derive

    core = {"max_batch_size": 128, "cuda_graph_config": {"enable_padding": True}}
    options = derive.layout_options("dep4", core)
    assert options["max_batch_size"] == 32
    assert options["cuda_graph_config"] == {"enable_padding": True, "max_batch_size": 32}
    assert options["enable_attention_dp"] and options["moe_expert_parallel_size"] == 4
    # Tensor and expert parallel serve the whole batch on every rank.
    assert derive.layout_options("tep4", core)["max_batch_size"] == 128
    assert core == {"max_batch_size": 128, "cuda_graph_config": {"enable_padding": True}}
