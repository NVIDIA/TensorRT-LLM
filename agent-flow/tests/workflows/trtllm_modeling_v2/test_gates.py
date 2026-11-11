# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 gates behavior and regression checks."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import accuracy as accuracy_checks
from agent_flow.workflows.trtllm_modeling_v2.checks import boot as boot_checks
from agent_flow.workflows.trtllm_modeling_v2.checks import delivery, disagg, gate_cli
from agent_flow.workflows.trtllm_modeling_v2.checks import serve as sg
from agent_flow.workflows.trtllm_modeling_v2.checks import verdict as check_result
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import serve_driver, workload_io
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
from agent_flow.workflows.trtllm_modeling_v2.spec import shorthand, task_schema

from ._helpers import _ENTRY, _REPO_ROOT, TARGET_REL, _build_repo, _verify, _write_entry, attested

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
    """The driver's records for ``texts``, then for the prompt a boot repeats where reuse is on."""
    return "".join(
        f"{workload_io.RECORD_MARKER} {json.dumps({'index': n, 'text': t, 'cached_tokens': 64})}\n"
        for n, t in enumerate((*texts, "again", "again"))
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
            said = output.pop(0) if isinstance(output, list) else output
            return CommandResult(returncode=0, stdout=said + attested(command))

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
    shorthand.share_batch(options, "dep4")
    assert options["max_batch_size"] == 32
    assert options["cuda_graph_config"] == {
        "max_batch_size": 32,
        "batch_sizes": [1, 2, 4, 8, 16, 24, 32],
    }


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
    monkeypatch.setattr(sg, "run_serve_gate", gate("serve"))
    monkeypatch.setattr(disagg, "run_disagg_gate", gate("disagg"))
    monkeypatch.setattr(accuracy_checks, "resolve_reference", lambda *a, **k: 85.6)
    runner = _gates_recorder(CommandResult(returncode=0, stdout=""))
    common = dict(workdir=tmp_path / "w", tree=tree, runner=runner, devices=(4, 5, 6, 7))

    # The contract names a benchmark `accuracy[gsm8k]`; a worker types `accuracy`.
    for name in ("boot", "accuracy[gsm8k]", "serve"):
        assert gate_cli.run_check(data, name, "mtp", "tep4", **common).passed
    assert gate_cli.run_check(data, "disagg", "disagg", "tep2", **common).passed
    assert set(seen) == {"boot", "accuracy", "serve", "disagg"}
    for name in ("boot", "accuracy", "serve"):
        call = seen[name]
        assert (call["unit"], call["member"]) == ("mtp", "tep4")
        # The tree measured is the one named, not the spec's checkout; the member's width
        # of the run's own devices, in their order.
        assert call["task_data"]["trtllm_repo_path"] == str(tree.resolve())
        assert call["environment"].devices == (4, 5, 6, 7)
        assert call["environment"].world_size == 4
        assert call["workdir"] == (tmp_path / "w").resolve() / "mtp"
    assert seen["boot"]["features"] == ["mtp"]
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

    def boot(**changes):
        def runner(command, env=None, timeout=0):
            return CommandResult(
                returncode=0, stdout=_records("Paris", "the") + attested(command, **changes)
            )

        return gate_cli.run_check(
            data, "boot", "core", "tp1", workdir=tmp_path / "w", runner=runner
        )

    here = boot()
    elsewhere = boot(origin="/opt/other/TensorRT-LLM/tensorrt_llm/__init__.py")
    assert here.passed and "imported tensorrt_llm from" not in here.detail
    assert not elsewhere.passed
    assert (
        "imported tensorrt_llm from /opt/other/TensorRT-LLM/tensorrt_llm/__init__.py, not from"
        in (elsewhere.detail)
    )


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
