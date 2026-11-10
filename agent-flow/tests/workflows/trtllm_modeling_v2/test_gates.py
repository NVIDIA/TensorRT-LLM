# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 gates behavior and regression checks."""

from __future__ import annotations

import contextlib
import io
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import acceptance as acceptance_checks
from agent_flow.workflows.trtllm_modeling_v2.checks import accuracy as accuracy_checks
from agent_flow.workflows.trtllm_modeling_v2.checks import boot as boot_checks
from agent_flow.workflows.trtllm_modeling_v2.checks import (
    check_config,
    delivery,
    disagg,
    divergence,
    gate_cli,
    test_results,
)
from agent_flow.workflows.trtllm_modeling_v2.checks import serve as sg
from agent_flow.workflows.trtllm_modeling_v2.checks import test_results as tc
from agent_flow.workflows.trtllm_modeling_v2.checks import verdict as check_result
from agent_flow.workflows.trtllm_modeling_v2.checks.delivery import Check as DeliveryCheck
from agent_flow.workflows.trtllm_modeling_v2.checks.gate_records import (
    KIND_STOCK_UNAVAILABLE,
    GateRecords,
)
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import (
    engine_driver,
    reuse_probe,
    serve_driver,
    workload_io,
)
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import test_driver as drv
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import (
    RunEnvironment,
    resolve_environment,
)
from agent_flow.workflows.trtllm_modeling_v2.operators import coverage as catalog_paths
from agent_flow.workflows.trtllm_modeling_v2.run import certify
from agent_flow.workflows.trtllm_modeling_v2.run import contract as contract_mod
from agent_flow.workflows.trtllm_modeling_v2.run.certify import (
    target_test_modules_check as target_tests,
)
from agent_flow.workflows.trtllm_modeling_v2.spec import task_schema

from ._helpers import _ENTRY, _REPO_ROOT, TARGET_REL, _build_repo, _verify, _write_entry

_gates_TASK = {
    "checkpoint_path": "/ckpt/Qwen3-30B-A3B",
    "parallel": "tp1",
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


def test_repeat_probe_completes_before_resubmission_and_pins_attention_rank(monkeypatch):
    events = []
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm.scheduling_params",
        SimpleNamespace(SchedulingParams=lambda **kw: kw),
    )

    class LLM:
        tokenizer = SimpleNamespace(encode=lambda *args, **kwargs: [1, 2, 3])

        def get_kv_cache_capacity(self):
            return {"tokensPerBlock": 4, "maxNumTokens": 128}

        def generate_async(self, prompt, params, **kwargs):
            events.append(("submit", kwargs))
            return SimpleNamespace(
                result=lambda: events.append(("complete", {})) or SimpleNamespace(cached_tokens=4)
            )

    result = reuse_probe.probe(LLM(), SimpleNamespace(), {"enable_attention_dp": True})
    assert result["status"] == "OBSERVED" and result["reused"] == 4
    assert [e[0] for e in events] == ["submit", "complete", "submit", "complete"]
    assert events[0][1] == events[2][1]
    assert events[0][1]["scheduling_params"] == {
        "attention_dp_rank": 0,
        "attention_dp_relax": False,
    }
    assert reuse_probe.probe(SimpleNamespace(), SimpleNamespace(), {})["status"] == "UNKNOWN"


def test_the_gate_reads_the_driver_and_names_the_member(tmp_path):
    options = tmp_path / "core.llm_api.yaml"
    options.write_text("kv_cache_config: {free_gpu_memory_fraction: 0.6}\n", encoding="utf-8")
    task = {
        "checkpoint_path": "/ckpt/model",
        "trtllm_repo_path": str(tmp_path / "repo"),
        "parallel": "unified",
        "members": ["tp1", "tp2", "tep2", "dep2", "tp4", "tep4", "dep4"],
        "features": ["mtp", "multimodal"],
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": "x.yaml",
            "llm_api_options": str(options),
            "max_seq_len": 9216,
            "max_batch_size": 128,
        },
    }
    answer = {"index": 0, "status": 200, "text": " Paris."}
    stdout = (
        f"{workload_io.RECORD_MARKER} {json.dumps(answer)}\n"
        f"{serve_driver.SERVER_MARKER} {json.dumps({'healthy': True, 'boot_s': 3.0})}\n"
    )
    block = {"kind": "local", "python": "/venv/bin/python", "env": {"HF_HUB_OFFLINE": "1"}}

    def runner(command, env=None, timeout=0):
        return CommandResult(returncode=0, stdout=stdout)

    result = sg.run_serve_gate(
        resolve_environment({"environment": block}, devices=(), world_size=1),
        task_data=task,
        runner=runner,
        features=[],
        unit="core",
        member="tp1",
        workdir=tmp_path,
    )
    assert result.passed, result.detail
    assert result.summary().startswith("[PASS] serve@tp1: trtllm-serve healthy after 3.0s")
    # One frontend: several need the classic IPC proxy a one-rank engine does not have.
    assert sg.serve_args(task, None, member="tp1", options=options)[:3] == [
        "--config",
        str(options),
        "--num_serve_frontends=1",
    ]


NODE = "tests/integration/defs/accuracy/test_disaggregated_multinode.py::TestStep5::test_dep4"


PASS = '<testsuite><testcase name="native_hop"/></testsuite>'


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
        RunEnvironment(kind="local", python="python3"),
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


def _cli_task(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(exist_ok=True)
    data = {
        "checkpoint_path": str(ckpt),
        "trtllm_repo_path": str(repo),
        "arch": "sm_100",
        "parallel": "unified",
        "members": ["tp2", "tep2"],
        "features": ["mtp", "disagg"],
        "environment": {"kind": "local", "python": "/usr/bin/python3"},
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": "r.yaml",
            "model_key": "M",
            "disagg": {"test": DISAGG_TEST, "env": {"LLM_MODELS_ROOT": "/models"}},
        },
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


DISAGG_TEST = (
    "tests/integration/defs/accuracy/test_disaggregated_serving.py::TestM::test_auto_dtype"
)


def _records(*texts):
    return "".join(
        f"{workload_io.RECORD_MARKER} {json.dumps({'index': n, 'text': t})}\n"
        for n, t in enumerate(texts)
    )


@pytest.fixture
def shell(monkeypatch):
    """Replace the executor-backed runner with a recorder answering `output`."""
    state = {"output": _records("Paris", "the process"), "seen": [], "junit": None}

    def fake_runner_for(executor, **kwargs):
        def run(command, env=None, timeout=0):
            state["seen"].append(command)
            report = re.search(r"--junitxml=([^\s']+)", command)
            if report and state["junit"] is not None:
                Path(report.group(1)).write_text(state["junit"], encoding="utf-8")
            output = state["output"]
            return CommandResult(
                returncode=0, stdout=output.pop(0) if isinstance(output, list) else output
            )

        return run

    monkeypatch.setattr(gate_cli, "shell_runner_for", fake_runner_for)
    return state


def test_a_relative_variant_is_judged_against_its_units_score_measured_first(
    tmp_path, shell, capsys
):
    path = _cli_task(tmp_path)
    data = yaml.safe_load(path.read_text())
    data["variants"] = {"switched": {"env": {"MY_SWITCH": "1"}, "gates": ["accuracy"]}}
    path.write_text(yaml.safe_dump(data))
    shell["output"] = [f"lm-eval gsm8k average accuracy: {score}\n" for score in (92, 85)]
    argv = ["accuracy", "--task", str(path), "--unit", "switched", "--workdir", str(tmp_path / "w")]
    assert gate_cli.main([*argv, "--member", "tep2"]) == 1
    out = capsys.readouterr().out
    assert out.startswith("[FAIL] accuracy[gsm8k]@tep2: measured 85.0000 vs floor 89.8872")
    assert "the unit's own score 92.0000" in out and len(shell["seen"]) == 2
    assert "MY_SWITCH=1" not in shell["seen"][0] and "MY_SWITCH=1" in shell["seen"][1]
    # The benchmark prints its running score, a heartbeat for a long run.
    assert "TLLM_EVAL_PARTIAL_SCORES_EVERY=50" in shell["seen"][0]
    # A score in another form is quoted for whoever decides how it is read, then read by pattern.
    printed = "Partial scores after 100/100 responses\nJSON Mode Eval accuracy: 91.50 (100)\n"
    assert accuracy_checks.parse_eval_score(printed) is None
    assert "  JSON Mode Eval accuracy: 91.50 (100)" in accuracy_checks.unparsed_score_detail(
        printed, None
    )
    assert accuracy_checks.parse_eval_score(printed, r"Eval accuracy: ([0-9.]+)") == 91.5


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


def test_an_op_imported_from_another_target_is_refused_and_copying_it_is_the_fix(tmp_path):
    [check] = delivery.check_model_ops(tmp_path, "models/family/mine", [("theirs", "norm", "rms")])
    assert not check.passed
    assert "another target's ops/ (theirs)" in check.detail
    assert "copy it into this target's ops/norm/" in check.detail


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


_acceptance_gate_TASK = {
    "checkpoint_path": "/ckpt/M",
    "parallel": "tep4",
    "trtllm_repo_path": "/repo",
    "features": ["dspark"],
    "accuracy": {"benchmark": "gsm8k", "reference_yaml": "r.yaml"},
}


def _acceptance_gate_env():
    return resolve_environment(
        {"environment": {"kind": "local", "python": "/py", "env": {}}}, devices=(0,), world_size=1
    )


def _acceptance_gate_driver_output(rates, *, prompts=None, iterations=100):
    """What the engine driver prints for prompts accepting at ``rates``.

    ``iterations`` is every prompt's, or a list with one per prompt.
    """
    lines = []
    for n, rate in enumerate(rates):
        steps = iterations[n] if isinstance(iterations, list) else iterations
        tokens = int(round(rate * steps))
        workload_io_line = {
            "index": n,
            "id": f"d{n}",
            "prompt_sha": (prompts or [f"p{i}" for i in range(len(rates))])[n],
            "tokens": tokens,
            "iterations": steps,
            "token_ids": [],
            "text": "",
        }
        lines.append(f"{workload_io.RECORD_MARKER} {json.dumps(workload_io_line)}")
    return "\n".join(lines) + "\n"


def _acceptance_gate_legs(require, off=None):
    """A runner: the require leg prints ``require``; the off leg ``off`` or cannot load."""
    seen = []

    def run(command, env=None, timeout=0):
        seen.append(command)
        if "TRTLLM_MODELING_V2=off" in command:
            if off is None:
                return CommandResult(returncode=1, stderr="Unknown model architecture: X")
            return CommandResult(returncode=0, stdout=off)
        return CommandResult(returncode=0, stdout=require)

    run.seen = seen
    return run


def _acceptance_gate_task(**acceptance):
    return {
        **_acceptance_gate_TASK,
        "accuracy": {**_acceptance_gate_TASK["accuracy"], "acceptance": acceptance},
    }


_BASE = [2.6 + 0.5 * math.sin(i) for i in range(32)]


@pytest.fixture
def tree(monkeypatch):
    """Pin the tree a gate believes it measured, so a baseline can come from another."""
    current = {"name": "treeA"}
    monkeypatch.setattr(acceptance_checks, "inspect_tree", lambda task_data: current["name"])
    return current


def test_a_significant_drop_against_the_previous_tree_fails(tmp_path, tree):
    records = GateRecords(tmp_path / "r.jsonl")
    common = dict(
        task_data=_acceptance_gate_task(), member="tep4", records=records, workdir=tmp_path / "w"
    )
    acceptance_checks.run_acceptance_gate(
        _acceptance_gate_env(),
        runner=_acceptance_gate_legs(
            _acceptance_gate_driver_output(_BASE), _acceptance_gate_driver_output(_BASE)
        ),
        **common,
    )
    tree["name"] = "treeB"
    dropped = [r - 0.3 for r in _BASE]
    result = acceptance_checks.run_acceptance_gate(
        _acceptance_gate_env(),
        runner=_acceptance_gate_legs(
            _acceptance_gate_driver_output(dropped), _acceptance_gate_driver_output(dropped)
        ),
        **common,
    )
    assert not result.passed
    assert "against tree treeA: REGRESSED" in result.detail
    # A failed verdict is recorded, but is not the next tree's baseline.
    tree["name"] = "treeC"
    again = acceptance_checks.run_acceptance_gate(
        _acceptance_gate_env(),
        runner=_acceptance_gate_legs(
            _acceptance_gate_driver_output(_BASE), _acceptance_gate_driver_output(_BASE)
        ),
        **common,
    )
    assert again.passed and "against tree treeA" in again.detail


_boot_boundaries_TASK = {
    "checkpoint_path": "/ckpt/Qwen3-30B-A3B",
    "parallel": "tp1",
    "trtllm_repo_path": "/repo",
    "runtime": {
        "cuda_graph": True,
        "chunked_prefill": True,
        "extra_flags": ["--max_seq_len=64", "--max_batch_size=8", "--max_num_tokens=32"],
    },
    "accuracy": {
        "benchmark": "gsm8k",
        "reference_yaml": "x.yaml",
        "envelope": {"enabled": True},
    },
}


def _run_boot(tmp_path, monkeypatch, *, mutate=None, task=None, unit=None, member=None):
    events = []
    specs = []
    active = 0

    class Params:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class LLM:
        def __init__(self, **kwargs):
            events.append(("construct", kwargs))
            self.tokenizer = SimpleNamespace(
                apply_chat_template=lambda *args, **kwargs: "chat: " + args[0][0]["content"]
            )

        def generate_async(self, prompt, params, streaming=False):
            nonlocal active
            active += 1
            events.append(("submit", prompt, params, active))
            output = SimpleNamespace(
                prompt_token_ids=(prompt["prompt_token_ids"] if isinstance(prompt, dict) else [1]),
                cached_tokens=0,
                outputs=[
                    SimpleNamespace(
                        token_ids=list(range(params.max_tokens)),
                        text="Paris",
                        finish_reason="length",
                    )
                ],
            )

            def stream():
                nonlocal active
                yield output
                active -= 1
                events.append(("drain", prompt))

            return stream()

        def shutdown(self):
            events.append(("shutdown",))

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm",
        SimpleNamespace(LLM=LLM, SamplingParams=Params, __file__="/fake"),
    )
    monkeypatch.setattr(engine_driver.engine_facts, "emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine_driver.reuse_probe, "probe", lambda *args: events.append(("reuse",)))

    def run(command, env=None, timeout=0):
        words = shlex.split(command)
        path = words[words.index("--spec") + 1]
        from pathlib import Path

        specs.append(json.loads(Path(path).read_text()))
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            assert engine_driver.main(["--spec", path]) == 0
        output = capture.getvalue()
        if mutate:
            lines = []
            for line in output.splitlines():
                if line.startswith(workload_io.RECORD_MARKER):
                    record = json.loads(line[len(workload_io.RECORD_MARKER) :])
                    mutate(record)
                    if record:
                        lines.append(f"{workload_io.RECORD_MARKER} {json.dumps(record)}")
                else:
                    lines.append(line)
            output = "\n".join(lines)
        return CommandResult(returncode=0, stdout=output)

    result = boot_checks.run_boot_gate(
        _gates_env(),
        task_data=task or _boot_boundaries_TASK,
        runner=run,
        workdir=tmp_path,
        unit=unit,
        member=member,
    )
    return result, specs, events


def _parted(gap):
    """Probe0's loaded leg takes another token at step 2, ``gap`` nats behind in both legs."""

    def mutate(record):
        if record.get("id") != "probe0":
            return
        own = 2 if record.get("phase") == "baseline" else 9
        other = 9 if own == 2 else 2
        if record.get("phase") == "loaded":
            record["token_ids"] = [0, 1, 9, 3]
        rows = [[[t, 0.0, 1]] for t in record["token_ids"]]
        rows[2] = [[own, -0.1, 1], [other, -0.1 - gap, 2]]
        record["top_logprobs"] = rows

    return mutate


def test_a_probe_that_parts_at_a_near_tie_passes_and_past_the_margin_fails(tmp_path, monkeypatch):
    result, specs, events = _run_boot(tmp_path, monkeypatch, mutate=_parted(0.05))
    assert result.passed, result.detail
    assert "near-tie(s) (probe0 step 2: baseline 2, loaded 9, 0.050 apart)" in result.detail
    # The probes ask the engine for their top log-probs; the fillers do not.
    probes = [r for p in specs[0]["phases"] for r in p["requests"] if r["id"].startswith("probe")]
    assert probes and all(r["logprobs"] == boot_checks.BOUNDARY_TOP_K for r in probes)
    asked = [e[2] for e in events if e[0] == "submit" and getattr(e[2], "logprobs", 0)]
    assert len(asked) == len(probes)
    result, _, _ = _run_boot(tmp_path, monkeypatch, mutate=_parted(2.0))
    assert not result.passed
    assert "must agree on 4 tokens" in result.detail
    assert "past the near-tie margin 0.150" in result.detail


_SNAPSHOTS = {
    "periodic_snapshot_interval": 2048,
    "additional_snapshot_offsets_from_start": [256, 512, 768],
}


_OPTIONS = {
    "kv_cache_config": {
        "free_gpu_memory_fraction": 0.6,
        "enable_block_reuse": True,
        "mamba_state_config": _SNAPSHOTS,
        "use_kv_cache_manager_v2": True,
    },
    "enable_chunked_prefill": True,
    "max_num_tokens": 8192,
    "cuda_graph_config": {"enable_padding": True, "max_batch_size": 128},
}


_RUNTIME = {
    "cuda_graph": True,
    "chunked_prefill": True,
    "overlap_scheduler": True,
    "kv_cache_reuse": True,
    "extra_flags": [
        "--max_seq_len=9216",
        "--max_batch_size=128",
        "--max_num_tokens=8192",
        "--kv_cache_fraction=0.6",
        "--mamba_ssm_cache_dtype=auto",
    ],
}


_PADDED = [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]


def _engine_attestation_task(tmp_path: Path, **extra) -> dict:
    options = tmp_path / "core.llm_api.yaml"
    options.write_text(yaml.safe_dump(_OPTIONS), encoding="utf-8")
    task = {
        "checkpoint_path": "/ckpt/model",
        "parallel": "unified",
        "trtllm_repo_path": str(tmp_path / "repo"),
        "features": [],
        "runtime": {"default": dict(_RUNTIME)},
        "accuracy": {"benchmark": "gsm8k", "llm_api_options": {"core": str(options)}},
    }
    task.update(extra)
    return task


def _engine_attestation_env() -> RunEnvironment:
    return RunEnvironment(kind="local", python="/py", world_size=1, env={})


def _engine_config(**changes) -> dict:
    """What a correctly built core@tp1 engine reports."""
    config = {
        "modeling_v2": "require",
        "block_reuse": True,
        "mamba_snapshot": {
            **_SNAPSHOTS,
            "additional_snapshot_offsets_from_end": [],
        },
        "kv_cache_dtype": "auto",
        "kv_cache_quant": None,
        "mamba_ssm_cache_dtype": "auto",
        "mamba_ssm_cache_dtype_resolved": "float32",
        "cuda_graph": {"batch_sizes": list(_PADDED), "enable_padding": True},
        "cuda_graph_captured": list(_PADDED),
        "speculative": None,
        "max_num_tokens": 8192,
        "max_batch_size": 128,
        "max_seq_len": 9216,
        "chunked_prefill": True,
        "overlap_scheduler": True,
        "tp": 1,
        "pp": 1,
        "ep": 1,
        "attention_dp": False,
    }
    config.update(changes)
    return config


def _engine_line(config: dict | None = None) -> str:
    """The engine record a worker-side observer logs for ``config``."""
    config = _engine_config() if config is None else config
    record = {
        "argv": [],
        "worker": True,
        "observer": True,
        "config": config,
        "sources": dict.fromkeys(config, "worker"),
        "pools": {"kv_max_tokens": 262144},
        "memory": {
            "source": "nvml",
            "devices": [{"device": "0", "free_mib": 40960.0, "total_mib": 81920.0}],
        },
        "errors": {},
    }
    return f"{workload_io.ENGINE_MARKER} {json.dumps(record)}\n"


def _boot_runner(*outputs: str):
    seen: list[str] = []

    def run(command, env=None, timeout=0):
        seen.append(command)
        return CommandResult(returncode=0, stdout=outputs[min(len(seen), len(outputs)) - 1])

    run.seen = seen  # type: ignore[attr-defined]
    return run


def test_the_acceptance_gate_holds_each_leg_to_its_own_mode(tmp_path):
    task = _engine_attestation_task(tmp_path, features=["mtp"])
    options = {**_OPTIONS, "speculative_config": {"decoding_type": "MTP", "max_draft_len": 3}}
    Path(task["accuracy"]["llm_api_options"]["core"]).write_text(yaml.safe_dump(options))
    mtp = {"algorithm": "MTP", "max_draft_len": 3}
    legs = [
        _engine_line(_engine_config(speculative=mtp)),
        _engine_line(_engine_config(speculative=mtp, modeling_v2="require")),
    ]
    run = _boot_runner(*legs)
    result = acceptance_checks.run_acceptance_gate(
        _engine_attestation_env(),
        task_data=task,
        runner=run,
        unit="core",
        member="tp1",
        workdir=tmp_path,
    )
    assert len(run.seen) == 2 and not result.passed
    assert result.detail.startswith("the engine of the off leg did not run the declared")
    assert "modeling_v2: declared off, engine require" in result.detail
    assert [r["leg"] for r in result.engine] == ["require", "off"]

    # A prompt the engine refuses for its length fails the workload, not the target; a leg
    # waits as long as the spec gives a slow load.
    task["accuracy"]["acceptance"] = {**(task["accuracy"].get("acceptance") or {}), "timeout": 3600}
    waited = []

    def refused(command, env=None, timeout=0):
        waited.append(timeout)
        error = "RequestError: The prompt length (5000.0) should not exceed max_num_tokens (4096)"
        return CommandResult(returncode=1, stdout=error)

    result = acceptance_checks.run_acceptance_gate(
        _engine_attestation_env(), task_data=task, runner=refused, unit="core", member="tp1",
        workdir=tmp_path,
    )  # fmt: skip
    assert "refused a prompt of 5000 tokens, longer than max_num_tokens 4096" in result.detail
    assert "a property of the workload, not of the target" in result.detail and waited == [3600]


_engine_attestation_FAKE_TRTLLM = {
    "tensorrt_llm/__init__.py": "",
    "tensorrt_llm/executor/__init__.py": "",
    "tensorrt_llm/executor/base_worker.py": """
        import types


        class BaseWorker:
            def __init__(self, llm_args):
                self.llm_args = llm_args
                self.rank = 0
                self.engine = None

            def setup_engine(self):
                kv = self.llm_args.kv_cache_config
                state = kv.mamba_state_config
                # What the model loader does for a hybrid model, here without a word.
                if not (state.periodic_snapshot_interval
                        or state.additional_snapshot_offsets_from_start):
                    kv.enable_block_reuse = False
                kv.max_gpu_total_bytes = 1 << 30
                stats = types.SimpleNamespace(max_num_blocks=100, tokens_per_block=32)
                quant = types.SimpleNamespace(
                    kv_cache_quant_algo=None, mamba_ssm_cache_dtype="torch.float32")
                self.engine = types.SimpleNamespace(
                    model_engine=types.SimpleNamespace(
                        model=types.SimpleNamespace(
                            model_config=types.SimpleNamespace(quant_config=quant)),
                        _cuda_graph_batch_sizes=[1, 2],
                    ),
                    kv_cache_manager=types.SimpleNamespace(
                        enable_block_reuse=kv.enable_block_reuse,
                        tokens_per_block=32,
                        get_kv_cache_stats=lambda: stats,
                    ),
                )
    """,
    "pynvml.py": """
        import os
        import types


        class NVMLError(Exception):
            pass


        def nvmlInit():
            pass


        def nvmlShutdown():
            pass


        def nvmlDeviceGetCount():
            return 1


        def nvmlDeviceGetHandleByIndex(index):
            return index


        def nvmlDeviceGetMemoryInfo(handle):
            free = float(os.environ.get("FAKE_FREE_MIB", "40960")) * (1 << 20)
            return types.SimpleNamespace(free=int(free), total=80 << 30)
    """,
}


_FAKE_LLM = """
    import os
    import types

    from tensorrt_llm.executor.base_worker import BaseWorker


    class MambaStateConfig:
        def __init__(self, periodic_snapshot_interval=0,
                     additional_snapshot_offsets_from_start=(),
                     additional_snapshot_offsets_from_end=()):
            self.periodic_snapshot_interval = periodic_snapshot_interval
            self.additional_snapshot_offsets_from_start = list(
                additional_snapshot_offsets_from_start)
            self.additional_snapshot_offsets_from_end = list(
                additional_snapshot_offsets_from_end)


    class KvCacheConfig:
        def __init__(self, **fields):
            self._stated = set(fields)
            self.enable_block_reuse = fields.get("enable_block_reuse", True)
            self.dtype = fields.get("dtype", "auto")
            self.mamba_ssm_cache_dtype = fields.get("mamba_ssm_cache_dtype", "auto")
            self.free_gpu_memory_fraction = fields.get("free_gpu_memory_fraction", 0.9)
            self.tokens_per_block = fields.get("tokens_per_block", 32)
            state = fields.get("mamba_state_config") or {}
            if isinstance(state, dict):
                state = MambaStateConfig(**state)
            self.mamba_state_config = state
            self.max_gpu_total_bytes = 0

        @property
        def model_fields_set(self):
            return self._stated


    class Tokenizer:
        def encode(self, text, add_special_tokens=True):
            return [ord(c) for c in text]


    class RequestOutput:
        # As the LLM API's: the handle `generate_async` returns is the output itself.

        def __init__(self, ids, cached):
            self.prompt_token_ids = ids
            self.cached_tokens = cached
            self.outputs = [types.SimpleNamespace(
                token_ids=[11, 12], text=" Paris", finish_reason="length")]

        def __iter__(self):
            yield self

        def result(self):
            return self


    class SamplingParams:
        def __init__(self, **fields):
            self.__dict__.update(fields)
            self.ignore_eos = False


    class LLM:
        # Resumes a prompt where an earlier one left a recurrent-state snapshot, as a
        # hybrid engine does; FAKE_REUSE=0 never resumes, FAKE_WORKER=0 sets its worker
        # up in another process.

        def __init__(self, model=None, **kw):
            kv = kw.get("kv_cache_config") or KvCacheConfig()
            if isinstance(kv, dict):
                kv = KvCacheConfig(**kv)
            parallel = types.SimpleNamespace(
                tp_size=kw.get("tensor_parallel_size", 1), pp_size=1,
                moe_ep_size=kw.get("moe_expert_parallel_size") or -1, moe_tp_size=-1,
                enable_attention_dp=kw.get("enable_attention_dp", False))
            self.args = types.SimpleNamespace(
                modeling_v2=kw.get("modeling_v2", "off"), kv_cache_config=kv,
                cuda_graph_config=kw.get("cuda_graph_config"), speculative_config=None,
                max_num_tokens=kw.get("max_num_tokens", 8192),
                max_batch_size=kw.get("max_batch_size", 2048),
                max_seq_len=kw.get("max_seq_len"),
                enable_chunked_prefill=kw.get("enable_chunked_prefill", False),
                disable_overlap_scheduler=False, parallel_config=parallel)
            self.tokenizer = Tokenizer()
            self.seen = []
            if os.environ.get("FAKE_WORKER", "1") == "1":
                BaseWorker(self.args).setup_engine()

        def get_kv_cache_capacity(self):
            return {"maxNumBlocks": 100, "tokensPerBlock": 32, "maxNumTokens": 3200}

        def _resumed(self, ids):
            kv = self.args.kv_cache_config
            if os.environ.get("FAKE_REUSE", "1") != "1" or not kv.enable_block_reuse:
                return 0
            shared = 0
            for earlier in self.seen:
                n = 0
                while n < min(len(ids), len(earlier)) and ids[n] == earlier[n]:
                    n += 1
                shared = max(shared, n)
            limit = min(shared, len(ids) - 1)
            state = kv.mamba_state_config
            points = [o for o in state.additional_snapshot_offsets_from_start if o <= limit]
            if state.periodic_snapshot_interval:
                points += list(range(state.periodic_snapshot_interval, limit + 1,
                                     state.periodic_snapshot_interval))
            return max(points, default=0)

        def generate_async(self, inputs, params=None, streaming=False, **kwargs):
            if isinstance(inputs, dict):
                ids = list(inputs["prompt_token_ids"])
            else:
                ids = self.tokenizer.encode(inputs)
            cached = self._resumed(ids)
            self.seen.append(ids)
            return RequestOutput(ids, cached)

        def shutdown(self):
            pass
"""


def _fake_site(tmp_path: Path) -> tuple[Path, Path]:
    """``(site, repo)``: a fake ``tensorrt_llm`` whose engine reuses, and a tree with its quickstart."""
    site = tmp_path / "site"
    files = {**_engine_attestation_FAKE_TRTLLM, "tensorrt_llm/__init__.py": _FAKE_LLM}
    for name, text in files.items():
        path = site / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")
    repo = tmp_path / "repo"
    repo.mkdir()
    return site, repo


def _real_boot(tmp_path: Path, **env) -> check_result.GateResult:
    site, repo = _fake_site(tmp_path)
    environment = RunEnvironment(
        kind="local", python=sys.executable, world_size=1, env={"PYTHONPATH": str(site), **env}
    )
    task = _engine_attestation_task(tmp_path, trtllm_repo_path=str(repo))
    return boot_checks.run_boot_gate(
        environment, task_data=task, unit="core", member="tp1", workdir=tmp_path / "gates"
    )


def test_a_real_boot_fails_on_an_engine_that_reused_nothing(tmp_path):
    for worker in ("1", "0"):
        result = _real_boot(tmp_path / worker, FAKE_WORKER=worker, FAKE_REUSE="0")
        assert not result.passed, result.stdout_tail
        assert "the engine reused nothing: the reuse probe sent one 320-token prompt" in (
            result.detail
        )


PARIS = {11: -0.05, 12: -3.2, 13: -4.0}


def test_a_flip_inside_the_tie_margin_agrees_and_one_outside_it_is_recorded():
    tied = {11: -0.70, 12: -0.75}
    swapped = {11: -0.75, 12: -0.70}
    assert divergence.first_token_agrees(tied, swapped, 0.15)
    found = divergence.assess({"p0": PARIS}, {"p0": {12: -0.05, 11: -3.2, 13: -4.0}},
                              divergence.Bounds())  # fmt: skip
    assert not found.within and found.top1 == 0.0
    assert "first token differs" in found.crossed[0]
    assert 0 < found.max_js <= math.log(2)
    assert "divergence recorded" in found.describe()


def _reference(tmp_path, first_tokens):
    path = tmp_path / "reference.json"
    path.write_text(json.dumps({
        "source": "transformers test", "top_k": 3,
        "prompts": {"p0": "The capital of France is"},
        "first_tokens": {"p0": {str(k): v for k, v in first_tokens.items()}},
    }))  # fmt: skip
    return {
        **_gates_TASK,
        "accuracy": {**_gates_TASK["accuracy"], "reference_logits": {"path": str(path)}},
    }


def _divergence_driver_output(texts, top):
    lines = []
    for n, text in enumerate(texts):
        record = {"index": n, "text": text, "token_ids": [1, 2], "tokens": 2, "iterations": 2}
        lines.append(f"{workload_io.RECORD_MARKER} {json.dumps(record)}")
    record = {"index": len(texts), "text": "x", "token_ids": [1], "tokens": 1, "iterations": 1,
              "top_logprobs": [[[t, lp] for t, lp in top.items()]]}  # fmt: skip
    lines.append(f"{workload_io.RECORD_MARKER} {json.dumps(record)}")
    return "\n".join(lines) + "\n"


def test_the_boot_records_a_divergence_and_still_passes(tmp_path):
    task = _reference(tmp_path, PARIS)
    texts = ("Paris is the capital", "the way plants make food")
    run = _gates_recorder(
        CommandResult(returncode=0, stdout=_divergence_driver_output(texts, PARIS))
    )
    result = boot_checks.run_boot_gate(_gates_env(), task_data=task, runner=run, workdir=tmp_path)
    assert result.passed and not result.warnings
    assert "against the reference, first tokens agree on 100% of 1 prompt(s)" in result.detail
    flipped = {12: -0.05, 11: -3.2, 13: -4.0}
    run = _gates_recorder(
        CommandResult(returncode=0, stdout=_divergence_driver_output(texts, flipped))
    )
    result = boot_checks.run_boot_gate(_gates_env(), task_data=task, runner=run, workdir=tmp_path)
    assert result.passed, result.detail
    assert "divergence from the reference recorded" in result.warnings[0]
    assert "p0 (first token differs" in result.detail


def test_an_unreadable_reference_is_unknown_not_a_failure(tmp_path):
    task = {**_gates_TASK, "accuracy": {**_gates_TASK["accuracy"],
                                  "reference_logits": {"path": str(tmp_path / "absent.json")}}}  # fmt: skip
    texts = ("Paris is the capital", "the way plants make food")
    run = _gates_recorder(
        CommandResult(returncode=0, stdout=_divergence_driver_output(texts, {})[:-1])
    )
    result = boot_checks.run_boot_gate(_gates_env(), task_data=task, runner=run, workdir=tmp_path)
    assert result.passed and "divergence UNKNOWN" in result.detail


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


def _spec(tmp_path) -> dict:
    checkpoint, repo = tmp_path / "ckpt", tmp_path / "repo"
    checkpoint.mkdir(exist_ok=True)
    repo.mkdir(exist_ok=True)
    (tmp_path / "mtp.yaml").write_text(
        yaml.safe_dump({"speculative_config": {"decoding_type": "MTP", "max_draft_len": 3}})
    )
    spec = {
        "checkpoint_path": str(checkpoint),
        "arch": "sm_103",
        "parallel": "unified",
        "members": {"default": ["tp1", "tep2", "tep4"], "mtp": ["tep4"]},
        "trtllm_repo_path": str(repo),
        "features": ["mtp"],
        "environment": {
            "kind": "local",
            "python": "/usr/bin/python3",
            "env": {"TMPDIR": "/scratch/tmp"},
        },
        "runtime": {"cuda_graph": True, "extra_flags": ["--max_seq_len=9216"]},
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": "x.yaml",
            "llm_api_options": {"mtp": str(tmp_path / "mtp.yaml")},
            "discriminators": {"default": {}, "mtp": {"spec_dec_algo": "MTP"}},
        },
        "variants": {
            "kv_fp8": {
                "members": ["tp1"],
                "runtime": {"extra_flags": ["--kv_cache_dtype=fp8"]},
                "llm_api_options": {"kv_cache_config": {"dtype": "fp8"}},
                "accuracy": {"discriminators": {"kv_cache_quant_algo": "FP8"}},
            },
            "mtp7": {
                "unit": "mtp",
                "gates": ["boot", "accuracy", "acceptance"],
                "llm_api_options": {"speculative_config": {"max_draft_len": 7}},
                "env": {"X": "1"},
            },
        },
    }
    return spec


def _load(tmp_path, spec) -> dict:
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return task_schema.load_and_validate_task_yaml(path)


def test_an_attention_dp_rank_serves_its_share_of_the_global_batch():
    options = {
        "max_batch_size": 128,
        "cuda_graph_config": {
            "max_batch_size": 128,
            "batch_sizes": [1, 2, 4, 8, 16, 24, 32, 64, 128],
        },
    }
    check_config.share_batch(options, "dep4")
    assert options["max_batch_size"] == 32
    assert options["cuda_graph_config"] == {
        "max_batch_size": 32,
        "batch_sizes": [1, 2, 4, 8, 16, 24, 32],
    }


def test_the_stock_leg_is_asked_once_per_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(acceptance_checks, "inspect_tree", lambda task_data: "t")
    task = _acceptance_gate_task(dspark={"min_absolute": 1.5, "floor_source": "a measured drafter"})
    rates = [2.6 + 0.5 * math.sin(i) for i in range(32)]
    records = GateRecords(tmp_path / "gate_results.jsonl")
    common = dict(task_data=task, records=records, unit="dspark", workdir=tmp_path / "w")
    first = _acceptance_gate_legs(_acceptance_gate_driver_output(rates))
    ran = acceptance_checks.run_acceptance_gate(_acceptance_gate_env(), runner=first, **common)
    assert ran.passed, ran.detail
    assert any("=off" in cmd for cmd in first.seen)
    assert records.all(KIND_STOCK_UNAVAILABLE)
    again = _acceptance_gate_legs(_acceptance_gate_driver_output(rates))
    result = acceptance_checks.run_acceptance_gate(_acceptance_gate_env(), runner=again, **common)
    assert result.passed and not any("=off" in cmd for cmd in again.seen)
    assert "not run again" in result.detail


def test_the_shared_runner_resolves_each_gate_the_way_a_worker_and_the_certification_ask(
    tmp_path, monkeypatch
):
    """One function runs a check: the contract's gate name, the unit, the member, the tree."""
    spec = _spec(tmp_path)
    spec["features"] = ["mtp", "disagg"]
    spec["accuracy"]["disagg"] = {"test": "tests/accuracy/test_disagg.py::TestToy::test_auto"}
    spec["accuracy"]["serve"] = {"enabled": True}
    data = _load(tmp_path, spec)
    tree = tmp_path / "worktree"
    tree.mkdir()
    seen: dict[str, dict] = {}

    def gate(name):
        def run(environment, **kwargs):
            seen[name] = {"environment": environment, **kwargs}
            return check_result.GateResult(name=name, passed=True, detail="ok", member="tep4")

        return run

    monkeypatch.setattr(boot_checks, "run_boot_gate", gate("boot"))
    monkeypatch.setattr(accuracy_checks, "run_accuracy_gate", gate("accuracy"))
    monkeypatch.setattr(acceptance_checks, "run_acceptance_gate", gate("acceptance"))
    monkeypatch.setattr(sg, "run_serve_gate", gate("serve"))
    monkeypatch.setattr(disagg, "run_disagg_gate", gate("disagg"))
    monkeypatch.setattr(accuracy_checks, "resolve_reference", lambda *a, **k: 85.6)
    runner = _gates_recorder(CommandResult(returncode=0, stdout=""))
    common = dict(workdir=tmp_path / "w", tree=tree, runner=runner, devices=(4, 5, 6, 7))

    # The contract names a benchmark `accuracy[gsm8k]`; a worker types `accuracy`.
    for name in ("boot", "accuracy[gsm8k]", "acceptance", "serve"):
        assert gate_cli.run_check(data, name, "mtp", "tep4", **common).passed
    assert gate_cli.run_check(data, "disagg", "disagg", "tep2", **common).passed
    assert set(seen) == {"boot", "accuracy", "acceptance", "serve", "disagg"}
    for name in ("boot", "accuracy", "acceptance", "serve"):
        call = seen[name]
        assert (call["unit"], call["member"]) == ("mtp", "tep4")
        # The tree measured is the one named, not the spec's checkout; the member's width
        # of the run's own devices, in their order.
        assert call["task_data"]["trtllm_repo_path"] == str(tree.resolve())
        assert call["environment"].devices == (4, 5, 6, 7)
        assert call["environment"].world_size == 4
        assert call["workdir"] == (tmp_path / "w").resolve() / "mtp"
    assert seen["boot"]["features"] == seen["acceptance"]["features"] == ["mtp"]
    assert seen["accuracy"]["reference"] == 85.6
    # Two engines at the member's width for the disaggregated test: four devices, not two.
    assert seen["disagg"]["environment"].devices == (4, 5, 6, 7)
    assert seen["disagg"]["environment"].world_size == 2
    assert (seen["disagg"]["unit"], seen["disagg"]["member"]) == ("disagg", "tep2")
    # A narrower member takes the first of the run's devices.
    gate_cli.run_check(data, "boot", "core", "tp1", **common)
    assert seen["boot"]["environment"].devices == (4,)
    with pytest.raises(task_schema.TaskSchemaError, match="names no gate"):
        gate_cli.run_check(data, "perf", "core", "tp1", **common)


def test_each_member_is_launched_at_its_own_width(tmp_path):
    """One shared runner resolves a check: a wider member's command starts its own ranks."""
    data = _load(tmp_path, _spec(tmp_path))
    seen: dict[str, list[str]] = {}
    for member in ("tp1", "tep2"):
        runner = _gates_recorder(CommandResult(returncode=0, stdout=""))
        result = gate_cli.run_check(
            data, "boot", "core", member, workdir=tmp_path / "w", runner=runner
        )
        assert result.member == member
        seen[member] = runner.seen
    assert any("mpirun -n 2" in command for command in seen["tep2"])
    assert not any("mpirun" in command for command in seen["tp1"])
    # Every command runs from inside the tree it measures, with it first on PYTHONPATH.
    tree = str((tmp_path / "repo").resolve())
    assert all(c.startswith(f"cd {tree} && export PYTHONPATH={tree}") for c in seen["tp1"])
    assert "CUDA_VISIBLE_DEVICES=0,1" in seen["tep2"][0]
    assert "CUDA_VISIBLE_DEVICES=0 " in seen["tp1"][0]


def test_a_check_whose_engine_imported_another_tree_did_not_measure_this_one(tmp_path):
    """The certification measures the checkout: an engine that loaded another tree's package fails."""
    data = _load(tmp_path, _spec(tmp_path))
    tree = (tmp_path / "repo").resolve()

    def engine_from(root):
        said = {"tensorrt_llm": f"{root}/tensorrt_llm/__init__.py", "api": "LLM"}
        return CommandResult(
            returncode=0,
            stdout=f"{workload_io.DRIVER_MARKER} {json.dumps(said)}\n" + _records("Paris", "the"),
        )

    def boot(output):
        return gate_cli.run_check(
            data, "boot", "core", "tp1", workdir=tmp_path / "w", runner=_gates_recorder(output)
        )

    here, elsewhere = boot(engine_from(tree)), boot(engine_from("/opt/other/TensorRT-LLM"))
    assert here.passed and "imported tensorrt_llm from" not in here.detail
    assert not elsewhere.passed
    assert (
        "imported tensorrt_llm from /opt/other/TensorRT-LLM, not from the tree" in elsewhere.detail
    )
    # What it measured is still said, after why it does not count.
    assert here.detail in elsewhere.detail


def test_every_member_under_its_reference_is_a_finding_unless_the_spec_acknowledges_it():
    """Each member can clear its own floor while all of them sit low: the round's verdicts say."""

    def verdict(member, measured, unit="dspark"):
        return contract_mod.Verdict(
            unit, "accuracy[gsm8k]", member, True, measured=measured, reference=_ANCHOR, se=0.734
        )

    low = {v.key: v for v in (verdict("tep4", 91.0), verdict("tp4", 91.1))}
    task = {"accuracy": dict(_statistical_floor_TASK["accuracy"])}
    [finding] = certify.shift_findings(task, low)
    assert finding.startswith("accuracy finding:") and "dspark" in finding
    # One member within an SE of the reference: no systematic shift.
    mixed = {**low, "dspark:accuracy[gsm8k]@tp4": verdict("tp4", _ANCHOR - 0.5)}
    assert certify.shift_findings(task, mixed) == []
    task["accuracy"]["floor"] = {"acknowledge": {"dspark": "the block drafter trades 1 point"}}
    assert certify.shift_findings(task, low) == []


class _Delivery:
    """What the delivered-tests check reads of a workflow."""

    _shell_runner_override = None

    def __init__(self, tmp_path: Path, head: str = "c0ffee1234567890"):
        self.task_data = {"trtllm_repo_path": str(tmp_path)}
        self.workspace = tmp_path / "ws"
        self._preflight = SimpleNamespace(target_rel="pkg/models/fam/target")
        self._git = lambda repo, *args: head + "\n"


def test_delivery_runs_the_delivered_tests_at_the_commit_and_holds_on_their_passing(
    tmp_path, monkeypatch
):
    checks = (DeliveryCheck("test:gemm/x", True, str(tmp_path / "tests/test_modeling_v2_x.py")),)
    test = "tests/test_modeling_v2_x.py"
    asked = {}

    def list_test_files(repo, head, paths):
        asked["paths"] = list(paths)
        return [test]

    results = test_results.Results("c0ffee1234567890", "env", collected={test: ["test_a"]})
    monkeypatch.setattr(test_results, "list_test_files", list_test_files)
    jobs = []

    def run_at(commit, files, **job):
        jobs.append((commit, list(files), job["tree"], job["python"]))
        return results, 1, 0, ""

    monkeypatch.setattr(test_results, "run_at", run_at)
    stage = _Delivery(tmp_path)
    # Nothing passed yet: an unrun test never holds the check.
    check = certify.tests_check(stage, checks, "c0ffee1234567890")
    assert not check.passed and check.detail.startswith("INCOMPLETE")
    assert asked["paths"][1] == test
    assert jobs == [("c0ffee1234567890", [test], tmp_path, "python3")]
    results.tests[f"{test}::test_a"] = {"outcome": "passed"}
    check = certify.tests_check(stage, checks, "c0ffee1234567890")
    assert check.passed and "1 test job(s) ran at c0ffee1234" in check.detail
    results.tests[f"{test}::test_a"] = {"outcome": "failed", "message": "wrong"}
    check = certify.tests_check(stage, checks, "c0ffee1234567890")
    assert not check.passed and check.detail.startswith("FAILING")
    # A probe that could not run leaves no verdict to reuse.
    monkeypatch.setattr(
        test_results, "run_at", lambda *a, **k: (test_results.Results("c0ffee"), 0, 1, "no gpu")
    )
    check = certify.tests_check(stage, checks, "c0ffee1234567890")
    assert not check.passed and check.detail == "INCOMPLETE (exit 1): no gpu"


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
        command.replace(" --timeout=180", ""), shell=True, capture_output=True, text=True
    )
    return done.returncode, done.stdout + done.stderr


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
    found = tc.head_results(results, ["tests/test_other_arch.py"])
    assert found.not_run == [] and found.skipped == ["tests/test_other_arch.py"]

    # The same file through the delivered-tests runner: probe, then one job, then nothing left.
    ran: list[str] = []

    def run_job(command, tree, timeout_s):
        ran.append(command)
        return _run(f"cd {tree} && {command}")

    # A cache of its own, so nothing the direct run above recorded is there to reuse.
    fresh = tmp_path / "ws" / "fresh_cache"
    job = dict(cache_dir=fresh, out=tmp_path / "out", tree=repo, run_job=run_job)
    results, jobs, code, _ = tc.run_at(
        head, ["tests/test_other_arch.py"], python=sys.executable, **job
    )
    assert results.env_digest and jobs == 1 and results.unrun(["tests/test_other_arch.py"]) == []
    again, jobs, _, _ = tc.run_at(head, ["tests/test_other_arch.py"], python=sys.executable, **job)
    assert jobs == 0 and len(ran) == 3  # the probe alone: its results in this environment stand
    failed = tc.run_at(
        head,
        ["tests/test_other_arch.py"],
        python=sys.executable,
        **{**job, "run_job": lambda command, tree, timeout_s: (1, "unavailable")},
    )
    assert not failed[0].env_digest and failed[1:3] == (0, 1)
    assert "environment probe failed" in failed[3]


def test_a_missing_anchor_row_says_how_to_measure_and_write_one(tmp_path):
    references = tmp_path / "gsm8k.yaml"
    references.write_text("org/other:\n  - accuracy: 50.0\n", encoding="utf-8")
    with pytest.raises(KeyError) as raised:
        accuracy_checks.load_reference_score(references, "org/model")
    message = str(raised.value)
    assert "never invented" in message
    assert "`org/model:` then `  - accuracy: <score>`" in message
    assert "trtllm-eval --model <checkpoint> <benchmark>" in message
