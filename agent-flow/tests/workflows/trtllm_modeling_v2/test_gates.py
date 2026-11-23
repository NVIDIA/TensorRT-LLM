# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Modeling-v2 gates behavior and regression checks."""

from __future__ import annotations

import contextlib
import json
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
from agent_flow.workflows.trtllm_modeling_v2.checks import (
    check_config,
    delivery,
    disagg,
    gate_cli,
    test_results,
)
from agent_flow.workflows.trtllm_modeling_v2.checks import serve as sg
from agent_flow.workflows.trtllm_modeling_v2.checks import test_results as tc
from agent_flow.workflows.trtllm_modeling_v2.checks.delivery import Check as DeliveryCheck
from agent_flow.workflows.trtllm_modeling_v2.checks.delivery import DeliveryReport
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import serve_driver, workload_io
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import test_driver as drv
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import (
    RunEnvironment,
    resolve_environment,
)
from agent_flow.workflows.trtllm_modeling_v2.operators import coverage as catalog_paths
from agent_flow.workflows.trtllm_modeling_v2.run.validation import test_jobs
from agent_flow.workflows.trtllm_modeling_v2.run.validation.check_runner import CheckRunner
from agent_flow.workflows.trtllm_modeling_v2.run.validation.completion import (
    target_test_modules_check as target_tests,
)
from agent_flow.workflows.trtllm_modeling_v2.run.validation.completion import (
    tests_check as delivered_tests,
)

from ._helpers import _ENTRY, _REPO_ROOT, TARGET_REL, _build_repo, _verify, _write_entry

_gates_TASK = {
    "checkpoint_path": "/ckpt/Qwen3-30B-A3B",
    "parallel": "tp1",
    "trtllm_repo_path": "/repo",
    "accuracy": {
        "benchmark": "gsm8k",
        "reference_yaml": "tests/integration/defs/accuracy/references/gsm8k.yaml",
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
    result = sg.run_serve_gate(
        resolve_environment({"environment": block}, devices=(), world_size=1),
        task_data=task,
        runner=lambda command, env=None, timeout=0: CommandResult(returncode=0, stdout=stdout),
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
    paths = [work / f"disagg.disagg.dep4.run.node{rank}.xml" for rank in range(2)]
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
        (tmp_path / "disagg.disagg.dep4.run.xml").write_text(PASS)
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


def test_the_ci_accuracy_test_is_the_generated_e2e_test_and_test_db_is_read_by_path(tmp_path):
    accuracy_test = delivery.accuracy_test_path(tmp_path, "gpt_oss", "pkg/models/gpt_oss/t")
    assert accuracy_test == "modeling_v2/gpt_oss/t/test_e2e.py"
    package = tmp_path / delivery.ACCURACY_TESTS_DIR / "modeling_v2" / "gpt_oss" / "t"
    package.mkdir(parents=True)
    (package / "test_e2e.py").write_text("", encoding="utf-8")
    db = tmp_path / delivery.TEST_DB_DIR
    db.mkdir(parents=True)
    (db / "l0_x.yml").write_text(
        "  - accuracy/modeling_v2/gpt_oss/other/test_e2e.py::TestOther::test_e2e\n",
        encoding="utf-8",
    )
    checks = {c.name: c.passed for c in delivery.check_test_registration(tmp_path, accuracy_test)}
    assert checks == {"tests:accuracy-file": True, "tests:test-db": False}
    (db / "l0_x.yml").write_text(
        f"  - accuracy/{accuracy_test}::TestT::test_e2e\n", encoding="utf-8"
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
            "from tensorrt_llm._torch.modeling_v2.catalog.norm import exported as run\n",
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
        "tensorrt_llm._torch.modeling_v2.models.gpt_oss.sibling.modeling"
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


def _reuse_task(tmp_path: Path, **extra) -> dict:
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


def test_each_member_is_launched_and_scheduled_at_its_own_width(tmp_path):
    from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import RunEnvironment
    from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

    workflow = TrtllmModelingV2Workflow(workspace=tmp_path / "ws")
    workflow.environment = RunEnvironment(
        kind="session_spool", python="python3", world_size=4, spool=str(tmp_path / "spool")
    )
    workflow.task_data = {"trtllm_repo_path": str(tmp_path)}
    runner = CheckRunner(workflow)
    assert runner._member_env("tp1").world_size == 1
    assert "mpirun -n 2" in runner._member_env("tep2").command("python x.py")
    assert "mpirun" not in runner._member_env("tp1").command("python x.py")
    assert runner._member_runner("tp1") is runner._member_runner("tp1")
    runner._member_runner("tep2")
    one = runner._member_runner("tep2", "all")
    two = runner._member_runner("tep2", "all", nodes=2)
    assert one is not two
    assert two is runner._member_runner("tep2", "all", nodes=2)
    # Keep node count separate when the per-node width and queue class match.
    assert sorted(runner._member_runners) == [
        ("1", 0, 1),
        ("2", 0, 1),
        ("all", 0, 1),
        ("all", 0, 2),
    ]


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


class _Delivery:
    def __init__(self, tmp_path: Path, head: str = "c0ffee1234567890"):
        self.task_data = {"trtllm_repo_path": str(tmp_path)}
        self.workspace = tmp_path / "ws"
        self._preflight = SimpleNamespace(target_rel="pkg/models/fam/target")
        self._git = lambda repo, *args: head + "\n"


def test_delivery_runs_the_delivered_tests_at_head_and_holds_on_their_passing(
    tmp_path, monkeypatch
):
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
    # Not run at HEAD yet: nothing to say before delivery runs it.
    assert delivered_tests(stage, report) is None
    assert asked["paths"][1] == test
    jobs = []

    def record_at(commit, files, **job):
        jobs.append((commit, list(files)))
        results.tests[f"{test}::test_a"] = {"outcome": "passed"}
        return results, 1, 0, ""

    monkeypatch.setattr(test_jobs, "job_runner", lambda workflow, name: (None, None, "python"))
    monkeypatch.setattr(test_jobs, "record_at", record_at)
    check = delivered_tests(stage, report, run=True)
    assert check.passed and jobs == [("c0ffee1234567890", [test])]
    heads = iter(["c0ffee1234567890", "a-new-head"])
    stage._git = lambda *args: next(heads)
    changed = delivered_tests(stage, report, run=True)
    assert not changed.passed and "HEAD changed" in changed.detail
    stage._git = lambda *args: "c0ffee1234567890"
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


def _run(command: str) -> tuple[int, str]:
    # pytest-timeout is not a test dependency here.
    done = subprocess.run(
        command.replace(" --timeout=180", ""), shell=True, capture_output=True, text=True
    )
    return done.returncode, done.stdout + done.stderr


def test_a_failed_environment_probe_returns_no_results(tmp_path):
    results, jobs, code, tail = test_jobs.record_at(
        "c0ffee1234567890",
        ["tests/test_a.py"],
        cache_dir=tmp_path / "cache",
        out=tmp_path / "failed-probe",
        tree_at=lambda commit: contextlib.nullcontext(tmp_path),
        run_job=lambda command, tree, timeout_s=None: (1, "unavailable"),
        python=sys.executable,
    )
    assert not results.env_digest and results.unrun(["tests/test_a.py"])
    assert jobs == 0 and code == 1 and "environment probe failed" in tail


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


def test_a_unit_that_drafts_with_an_earlier_units_drafter_is_held_to_its_floor():
    resolve = acceptance_checks.resolve_acceptance_floor_spec
    accuracy = {"acceptance": {"mtp": {"min_absolute": 1.5}}}
    # `multimodal` comes after `mtp` and runs its drafter: mtp's floor holds it.
    assert resolve(accuracy, "multimodal", ["mtp", "multimodal"]).value == 1.5
    assert resolve(accuracy, "multimodal", ["multimodal"]) is None
    accuracy["acceptance"]["multimodal"] = 1.8
    assert resolve(accuracy, "multimodal", ["mtp", "multimodal"]).value == 1.8


_LAYOUT_PACKAGE = "tensorrt_llm/_torch/modeling_v2"
_LAYOUT_TESTS = "tests/unittest/_torch/modeling_v2"
_LAYOUT_TARGET = f"{_LAYOUT_PACKAGE}/models/layout_family/layout_target"
_SHARED_MOE_ENTRIES = (
    "fp4_block_scale_moe_runner",
    "mxe4m3_mxe2m1_block_scale_moe_runner",
    "noaux_tc_op",
)


@pytest.mark.parametrize("layout", ["moe/ops", "moe"])
def test_shared_moe_test_layout_paths_and_delivery(tmp_path: Path, layout: str) -> None:
    name = "fp4_block_scale_moe_runner"
    source = f"{_LAYOUT_PACKAGE}/catalog/moe/{name}.py"
    test = f"{_LAYOUT_TESTS}/{layout}/test_modeling_v2_{name}.py"
    _write_entry(tmp_path / source, tmp_path / test)
    model = tmp_path / _LAYOUT_TARGET / "modeling.py"
    model.parent.mkdir(parents=True)
    model.write_text(
        f"from tensorrt_llm._torch.modeling_v2.catalog.moe.{name} import Entry\n",
        encoding="utf-8",
    )

    assert catalog_paths.shared_entry_paths(f"moe/{name}", tmp_path) == (source, test)
    assert catalog_paths.is_entry_file(test, f"moe/{name}")
    [check] = delivery.check_catalog_tests(tmp_path, [("moe", name)])
    assert check.passed, check.detail
    assert check.detail == str(tmp_path / test)
    checks = delivery._dependency_checks(tmp_path, _LAYOUT_TARGET)
    assert checks and all(c.passed for c in checks), checks
    assert any(c.detail == str(tmp_path / test) for c in checks)


def test_shared_moe_mixed_layout_resolves_each_entry_separately(tmp_path: Path) -> None:
    expected: dict[str, str] = {}
    for index, name in enumerate(_SHARED_MOE_ENTRIES):
        layout = "moe" if index == 1 else "moe/ops"
        test = f"{_LAYOUT_TESTS}/{layout}/test_modeling_v2_{name}.py"
        source = f"{_LAYOUT_PACKAGE}/catalog/moe/{name}.py"
        _write_entry(tmp_path / source, tmp_path / test)
        expected[name] = test
    duplicate = tmp_path / _LAYOUT_TESTS / f"moe/test_modeling_v2_{_SHARED_MOE_ENTRIES[0]}.py"
    duplicate.write_text("def test_without_cells() -> None:\n    pass\n", encoding="utf-8")

    for name, test in expected.items():
        assert catalog_paths.shared_entry_paths(f"moe/{name}", tmp_path)[1] == test
        [check] = delivery.check_catalog_tests(tmp_path, [("moe", name)])
        assert check.passed and check.detail == str(tmp_path / test), check
    other_source = f"{_LAYOUT_PACKAGE}/catalog/gemm/dense.py"
    other_test = f"{_LAYOUT_TESTS}/gemm/test_modeling_v2_dense.py"
    _write_entry(tmp_path / other_source, tmp_path / other_test)
    assert catalog_paths.shared_entry_paths("gemm/dense", tmp_path) == (other_source, other_test)
    assert delivery.check_catalog_tests(tmp_path, [("gemm", "dense")])[0].passed


@pytest.mark.parametrize("owner", ["family", "target"])
@pytest.mark.parametrize("category", ["moe", "comm"])
@pytest.mark.parametrize("mirrored", [False, True], ids=["category_dir", "ops_dir"])
def test_model_op_test_layout_preserves_owner_and_category(
    tmp_path: Path, owner: str, category: str, mirrored: bool
) -> None:
    owner_tail = "models/layout_family" + ("/layout_target" if owner == "target" else "")
    owner_rel = f"{_LAYOUT_PACKAGE}/{owner_tail}"
    source = f"{owner_rel}/ops/{category}/probe.py"
    component = f"ops/{category}" if mirrored else category
    test = f"{_LAYOUT_TESTS}/{owner_tail}/{component}/test_modeling_v2_probe.py"
    _write_entry(tmp_path / source, tmp_path / test)
    model = tmp_path / _LAYOUT_TARGET / "modeling.py"
    model.parent.mkdir(parents=True, exist_ok=True)
    relative_import = "." if owner == "target" else ".."
    model.write_text(f"from {relative_import}ops.{category}.probe import Entry\n", encoding="utf-8")
    item = catalog_paths.QueueItem(n=1, mode="verify", target=f"ops/{category}/probe")

    assert catalog_paths.model_ops_tests_dir(owner_rel) == f"{_LAYOUT_TESTS}/{owner_tail}"
    assert catalog_paths.entry_paths(item, owner_rel, tmp_path) == (source, test)
    assert catalog_paths.is_entry_file(test, item.target)
    [check] = delivery.check_model_op_tests(
        tmp_path, owner_rel, [(Path(owner_rel).name, category, "probe")]
    )
    assert check.passed and check.detail == str(tmp_path / test), check
    checks = delivery._dependency_checks(tmp_path, _LAYOUT_TARGET)
    assert checks and all(c.passed for c in checks), checks
    assert any(c.detail == str(tmp_path / test) for c in checks)


@pytest.mark.parametrize("shared", [False, True], ids=["target", "shared"])
@pytest.mark.parametrize("mirrored", [False, True], ids=["category_dir", "ops_dir"])
def test_op_matrix_layout_retains_collected_launcher_and_rank_body(
    tmp_path: Path, shared: bool, mirrored: bool
) -> None:
    category = "moe" if shared else "comm"
    if shared:
        source = f"{_LAYOUT_PACKAGE}/catalog/moe/probe.py"
        tests = f"{_LAYOUT_TESTS}/moe" + ("/ops" if mirrored else "")
    else:
        source = f"{_LAYOUT_TARGET}/ops/comm/probe.py"
        tests = f"{_LAYOUT_TESTS}/models/layout_family/layout_target/"
        tests += "ops/comm" if mirrored else "comm"
    launcher = f"{tests}/test_modeling_v2_probe_op_matrix.py"
    rank_body = f"{tests}/_probe_op_matrix.py"
    _write_entry(tmp_path / source, tmp_path / rank_body)
    (tmp_path / launcher).write_text("def test_matrix() -> None:\n    pass\n", encoding="utf-8")

    if shared:
        paths = catalog_paths.shared_entry_paths("moe/probe", tmp_path)
        checks = delivery.check_catalog_tests(tmp_path, [(category, "probe")])
    else:
        item = catalog_paths.QueueItem(n=1, mode="verify", target="ops/comm/probe")
        paths = catalog_paths.entry_paths(item, _LAYOUT_TARGET, tmp_path)
        checks = delivery.check_model_op_tests(
            tmp_path, _LAYOUT_TARGET, [("layout_target", category, "probe")]
        )
    assert paths == (source, launcher, rank_body)
    assert len(checks) == 1 and checks[0].passed, checks
    assert checks[0].detail == str(tmp_path / launcher)
    assert catalog_paths.is_entry_file(launcher, f"{'moe' if shared else 'ops/comm'}/probe")
    assert catalog_paths.is_entry_file(rank_body, f"{'moe' if shared else 'ops/comm'}/probe")


@pytest.mark.parametrize("shared", [False, True], ids=["target", "shared"])
@pytest.mark.parametrize("defect", ["rank-body-only", "no-cells"])
def test_new_test_layout_does_not_accept_missing_or_unexercised_tests(
    tmp_path: Path, shared: bool, defect: str
) -> None:
    if shared:
        source = f"{_LAYOUT_PACKAGE}/catalog/moe/probe.py"
        tests = f"{_LAYOUT_TESTS}/moe/ops"
        imported = "tensorrt_llm._torch.modeling_v2.catalog.moe.probe"
    else:
        source = f"{_LAYOUT_TARGET}/ops/moe/probe.py"
        tests = f"{_LAYOUT_TESTS}/models/layout_family/layout_target/ops/moe"
        imported = ".ops.moe.probe"
    filename = "_probe_op_matrix.py" if defect == "rank-body-only" else "test_modeling_v2_probe.py"
    test = tmp_path / tests / filename
    _write_entry(tmp_path / source, test)
    if defect == "no-cells":
        test.write_text("def test_without_cells() -> None:\n    pass\n", encoding="utf-8")
    model = tmp_path / _LAYOUT_TARGET / "modeling.py"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_text(f"from {imported} import Entry\n", encoding="utf-8")

    checks = (
        delivery.check_catalog_tests(tmp_path, [("moe", "probe")])
        if shared
        else delivery.check_model_op_tests(
            tmp_path, _LAYOUT_TARGET, [("layout_target", "moe", "probe")]
        )
    )
    assert len(checks) == 1 and not checks[0].passed, checks
    expected = "missing:" if defect == "rank-body-only" else "never reads the entry's .CELLS"
    assert expected in checks[0].detail
    dependency_failures = [
        check for check in delivery._dependency_checks(tmp_path, _LAYOUT_TARGET) if not check.passed
    ]
    assert len(dependency_failures) == 1, dependency_failures
    assert expected in dependency_failures[0].detail


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        (f"{_LAYOUT_TESTS}/moe/ops/test_modeling_v2_probe.py", True),
        (f"{_LAYOUT_TESTS}/moe/test_modeling_v2_probe.py", True),
        (f"{_LAYOUT_TESTS}/gemm/ops/test_modeling_v2_probe.py", False),
        (
            f"{_LAYOUT_TESTS}/models/layout_family/layout_target/ops/moe/test_modeling_v2_probe.py",
            False,
        ),
        (
            f"{_LAYOUT_TESTS}/models/layout_family/layout_target/moe/ops/test_modeling_v2_probe.py",
            False,
        ),
    ],
)
def test_shared_moe_entry_file_boundary(path: str, expected: bool) -> None:
    assert catalog_paths.is_entry_file(path, "moe/probe") is expected


@pytest.mark.parametrize("component", ["comm", "ops/comm"])
def test_new_model_op_uses_existing_component_directory(tmp_path: Path, component: str) -> None:
    tests = f"{_LAYOUT_TESTS}/models/layout_family/layout_target/{component}"
    existing = tmp_path / tests / "test_modeling_v2_existing_op_matrix.py"
    existing.parent.mkdir(parents=True)
    existing.write_text("def test_existing(): pass\n", encoding="utf-8")
    item = catalog_paths.QueueItem(n=1, mode="verify", target="ops/comm/new_collective")
    expected = f"{tests}/test_modeling_v2_new_collective.py"
    assert catalog_paths.entry_paths(item, _LAYOUT_TARGET, tmp_path)[1] == expected
    [check] = delivery.check_model_op_tests(
        tmp_path, _LAYOUT_TARGET, [("layout_target", "comm", "new_collective")]
    )
    assert not check.passed and check.detail == f"missing: {expected}"
    _write_entry(tmp_path / _LAYOUT_TARGET / "ops/comm/new_collective.py", tmp_path / expected)
    [check] = delivery.check_model_op_tests(
        tmp_path, _LAYOUT_TARGET, [("layout_target", "comm", "new_collective")]
    )
    assert check.passed and check.detail == str(tmp_path / expected)


@pytest.mark.parametrize("owner", ["shared", "family", "target"])
def test_new_moe_entry_uses_contract_directory(tmp_path: Path, owner: str) -> None:
    name = "new_probe"
    owner_rel = ""
    tests_base = _LAYOUT_TESTS
    if owner != "shared":
        owner_tail = "models/layout_family" + ("/layout_target" if owner == "target" else "")
        owner_rel = f"{_LAYOUT_PACKAGE}/{owner_tail}"
        tests_base = f"{_LAYOUT_TESTS}/{owner_tail}"
    old_dir = tmp_path / tests_base / "moe"
    old_dir.mkdir(parents=True)
    (old_dir / "test_modeling_v2_existing.py").write_text(
        "def test_existing(): pass\n", encoding="utf-8"
    )
    tests_dir = f"{tests_base}/ops/moe" if owner_rel else f"{tests_base}/moe/ops"
    expected = f"{tests_dir}/test_modeling_v2_{name}.py"

    if owner_rel:
        source = f"{owner_rel}/ops/moe/{name}.py"
        item = catalog_paths.QueueItem(n=1, mode="verify", target=f"ops/moe/{name}")
        paths = catalog_paths.entry_paths(item, owner_rel, tmp_path)
        [check] = delivery.check_model_op_tests(
            tmp_path, owner_rel, [(Path(owner_rel).name, "moe", name)]
        )
    else:
        source = f"{_LAYOUT_PACKAGE}/catalog/moe/{name}.py"
        paths = catalog_paths.shared_entry_paths(f"moe/{name}", tmp_path)
        [check] = delivery.check_catalog_tests(tmp_path, [("moe", name)])
    assert paths == (source, expected)
    assert not check.passed and check.detail == f"missing: {expected}"

    _write_entry(tmp_path / source, tmp_path / expected)
    if owner_rel:
        [check] = delivery.check_model_op_tests(
            tmp_path, owner_rel, [(Path(owner_rel).name, "moe", name)]
        )
    else:
        [check] = delivery.check_catalog_tests(tmp_path, [("moe", name)])
    assert check.passed and check.detail == str(tmp_path / expected)


def test_shared_moe_matrix_variant_keeps_launcher_and_rank_body(tmp_path: Path) -> None:
    tests = tmp_path / _LAYOUT_TESTS / "moe/ops"
    tests.mkdir(parents=True)
    for name in ("test_modeling_v2_probe_op_matrix.py", "_probe_op_matrix.py"):
        (tests / name).write_text("# Matrix entry.\n", encoding="utf-8")
    item = catalog_paths.queue_item_from_ref("verify:moe/probe@site", "matrix_probe")
    assert item is not None and item.variant is not None
    name = item.variant.rsplit("/", 1)[-1]
    target_tests = f"{_LAYOUT_TESTS}/models/layout_family/layout_target/ops/moe"
    (tmp_path / _LAYOUT_TESTS / "models/layout_family/layout_target/moe").mkdir(parents=True)
    assert catalog_paths.entry_paths(item, _LAYOUT_TARGET, tmp_path) == (
        f"{_LAYOUT_TARGET}/{item.variant}.py",
        f"{target_tests}/test_modeling_v2_{name}_op_matrix.py",
        f"{target_tests}/_{name}_op_matrix.py",
    )


def test_dflash_keeps_the_speculative_config_its_options_declare(tmp_path):
    task = _reuse_task(tmp_path, features=["dflash"])
    drafter = {"decoding_type": "DFlash", "max_draft_len": 3, "speculative_model": "/drafter"}
    Path(task["accuracy"]["llm_api_options"]["core"]).write_text(
        yaml.safe_dump({**_OPTIONS, "speculative_config": drafter}), encoding="utf-8"
    )
    config = check_config.check_configuration(
        task,
        RunEnvironment(kind="local", python="/py", world_size=1, env={}),
        unit="core",
        features=["dflash"],
        max_tokens=8,
        member="tp1",
    )
    # Before, a dflash unit fell through to `speculative_config: None`: the engine ran with
    # no drafter, and the acceptance gate measured nothing it asked for.
    assert config["llm"]["speculative_config"] == drafter
    assert check_config.speculation_flags(["dflash"], max_draft_len=3) == [
        "--spec_decode_algo=DFlash",
        "--spec_decode_max_draft_len=3",
    ]
