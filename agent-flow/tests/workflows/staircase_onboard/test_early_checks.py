# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the early delivery checks (``early_checks.py``).

The fixtures are the DeepSeek-V4.1 run's delivery surprises: after every unit
was green, delivery failed on two comm entries whose tests were not named
``test_modeling_v2_<entry>.py``, a missing integration accuracy test and a
missing test-db line; and later its accuracy cases turned out to be listed
post-merge only, on a 527 GB checkpoint nobody had confirmed CI's model store
holds, with skip conditions that made several cases skip on CI.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from agent_flow.workflows.staircase_onboard import delivery, early_checks, envelope
from agent_flow.workflows.staircase_onboard.capacity_audit import audit_target
from agent_flow.workflows.staircase_onboard.state import load_state

from .test_workflow import Script, _install, _workflow, _write_vocabulary

_ACCURACY_TEST = """\
import os

import pytest

from tensorrt_llm._utils import get_sm_version

from ..conftest import llm_models_root

skip_not_sm103 = pytest.mark.skipif(get_sm_version() != 103, reason="certified on sm_103 only")


class TestModelingV2DeepseekV41(object):
    MODEL_PATH = f"{llm_models_root()}/DeepSeek-V4.1-Flash-NVFP4"

    @skip_not_sm103
    @pytest.mark.skip_less_device(4)
    def test_gsm8k(self):
        golden = os.environ.get("STAIRCASE_GOLDEN_DIR")
        if golden is None:
            pytest.skip("no golden reference")
"""


def _db(root: Path, name: str, stage: str, gpus, count, test: str) -> None:
    db = root / delivery.TEST_DB_DIR
    db.mkdir(parents=True, exist_ok=True)
    (db / name).write_text(
        yaml.safe_dump(
            {
                name.removesuffix(".yml"): [
                    {
                        "condition": {
                            "ranges": {"system_gpu_count": {"gte": count, "lte": count}},
                            "wildcards": {"gpu": gpus},
                            "terms": {"stage": stage},
                        },
                        "tests": [test],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )


def _repo(tmp_path: Path, *, accuracy: bool = True) -> Path:
    root = tmp_path / "repo"
    catalog = root / delivery.CATALOG_DIR
    tests = root / delivery.CATALOG_TESTS_DIR
    for category, name, test_name in (
        ("comm", "allreduce", "test_allreduce_op_matrix.py"),
        ("norm", "rmsnorm", "test_modeling_v2_rmsnorm.py"),
    ):
        (catalog / category).mkdir(parents=True, exist_ok=True)
        (catalog / category / f"{name}.py").write_text("def f(): ...\n", encoding="utf-8")
        if test_name:
            (tests / category).mkdir(parents=True, exist_ok=True)
            (tests / category / test_name).write_text("def test(): ...\n", encoding="utf-8")
    (catalog / "gemm").mkdir(parents=True)
    (catalog / "gemm" / "mm_mxfp8.py").write_text("def f(): ...\n", encoding="utf-8")
    if accuracy:
        acc = root / delivery.ACCURACY_TESTS_DIR
        acc.mkdir(parents=True)
        (acc / "test_modeling_v2_deepseek_v41.py").write_text(_ACCURACY_TEST, encoding="utf-8")
    return root


def _vocabulary(tmp_path: Path) -> Path:
    path = tmp_path / "vocabulary.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "steps": [
                    {"step": "tp all-reduce", "entry": "catalog/comm/allreduce.py"},
                    {"step": "norm", "entry": "catalog/norm/rmsnorm.py"},
                    {"step": "route", "entry": "ops/moe/router.py"},
                    {
                        "step": "gemm",
                        "entry": "catalog/gemm/mm_mxfp8.py",
                        "call_sites": [{"site": "x", "row_multiple": 0}],
                    },
                ],
                "queue": [{"n": 1, "mode": "onboard", "target": "sampling/draft_sample"}],
            }
        ),
        encoding="utf-8",
    )
    return path


def _names(report, level=early_checks.LEVEL_WARN):
    return {f.name for f in report.findings if f.level == level}


def test_test_names_ci_does_not_match_are_found_before_delivery(tmp_path):
    report = early_checks.run_early_checks(
        repo_root=_repo(tmp_path),
        family="deepseek_v41",
        stage="plan",
        vocabulary_path=_vocabulary(tmp_path),
    )
    warned = {f.name: f.detail for f in report.warnings}
    assert "test_allreduce_op_matrix.py" in warned["test-name:comm/allreduce"]
    assert "has no test" in warned["test-name:gemm/mm_mxfp8"]
    # Correctly named, the target's own op (delivery checks it at the target's
    # path), and an entry smith has not written yet: no warning.
    assert not {"test-name:norm/rmsnorm", "test-name:moe/router"} & set(warned)
    assert not any("draft_sample" in name for name in warned)
    assert "vocabulary:call_sites" in warned


def test_a_missing_accuracy_test_is_a_warning_and_no_assets_are_listed(tmp_path):
    report = early_checks.run_early_checks(
        repo_root=_repo(tmp_path, accuracy=False), family="deepseek_v41", stage="plan"
    )
    assert "tests:accuracy-file" in _names(report)
    assert report.assets == ()
    assert "(no accuracy test to read yet)" in report.render()


def test_the_required_ci_assets_are_read_from_the_test(tmp_path):
    root = _repo(tmp_path)
    assets = early_checks.analyze_skips(
        root / delivery.ACCURACY_TESTS_DIR / "test_modeling_v2_deepseek_v41.py"
    )
    found = {(a.kind, a.what) for a in assets}
    assert ("checkpoint", "$LLM_MODELS_ROOT/DeepSeek-V4.1-Flash-NVFP4") in found
    assert ("arch", "sm_103") in found
    assert ("gpus", ">= 4") in found
    assert ("env", "STAIRCASE_GOLDEN_DIR") in found
    assert ("skip", "pytest.skip(...) at runtime") in found


def test_ci_skips_are_predicted_from_the_test_db(tmp_path):
    root = _repo(tmp_path)
    test = "accuracy/test_modeling_v2_deepseek_v41.py::TestModelingV2DeepseekV41::test_gsm8k"
    _db(root, "l0_gb300_multi_gpus.yml", "post_merge", ["*gb300*"], 4, test)
    report = early_checks.run_early_checks(repo_root=root, family="deepseek_v41", stage="unit")
    warned = {f.name: f.detail for f in report.warnings}
    assert "post-merge only" in warned["tests:test-db"]
    assert "ci:env:STAIRCASE_GOLDEN_DIR" in warned
    assert "ci:arch" not in warned and "ci:gpus" not in warned

    _db(root, "l0_b200.yml", "pre_merge", ["*b200*"], 1, test)
    report = early_checks.run_early_checks(repo_root=root, family="deepseek_v41", stage="unit")
    details = [f.detail for f in report.warnings]
    assert any("needs sm_103" in d and "l0_b200.yml" in d and "always skips" in d for d in details)
    assert any("needs >= 4 GPUs and l0_b200.yml runs on at most 1" in d for d in details)
    assert "post-merge only" not in " ".join(details)
    assert "## Where CI runs it" in report.render()


def test_the_checkpoint_is_checked_where_the_model_store_is_known(tmp_path):
    root = _repo(tmp_path)
    store = tmp_path / "models"
    store.mkdir()
    env = {early_checks.MODELS_ROOT_ENV: str(store)}
    report = early_checks.run_early_checks(
        repo_root=root, family="deepseek_v41", stage="u", env=env
    )
    assert (
        "does not exist under this host's LLM_MODELS_ROOT"
        in {f.name: f.detail for f in report.warnings}["ci:checkpoint"]
    )

    (store / "DeepSeek-V4.1-Flash-NVFP4").mkdir()
    report = early_checks.run_early_checks(
        repo_root=root, family="deepseek_v41", stage="u", env=env
    )
    assert "ci:checkpoint" in _names(report, early_checks.LEVEL_OK)


def test_capacity_and_matrix_failures_surface_early(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    (target / "m.py").write_text("_MAX_ROWS = 8192\n", encoding="utf-8")
    env = envelope.ServingEnvelope(max_batch_size=8, max_num_tokens=16384, max_seq_len=4096)
    plan = envelope.CertificationPlan(
        envelope=env,
        cells=(
            envelope.MatrixCell(
                cell="m",
                features=(),
                member="dep8",
                evidence="boot",
                at="ceiling",
                status=envelope.CELL_EXTERNAL,
                reason="two nodes",
            ),
        ),
    )
    report = early_checks.run_early_checks(
        repo_root=_repo(tmp_path),
        family="deepseek_v41",
        stage="unit",
        capacity=audit_target(target, env),
        certification=plan,
        workspace=tmp_path / "ws",
    )
    assert {"capacity:_MAX_ROWS", "matrix:m__dep8__boot__ceiling"} <= _names(report)


def test_the_workflow_reports_after_the_plan_and_every_unit_and_does_not_stop(
    tmp_path, task_file, monkeypatch
):
    workflow = _workflow(tmp_path)
    _install(monkeypatch, workflow, Script(None))
    original = workflow._run_plan_phase

    def plan_then_write(state, log):
        (tmp_path / "ws").mkdir(parents=True, exist_ok=True)
        _write_vocabulary(tmp_path / "ws", [])
        return original(state, log)

    monkeypatch.setattr(workflow, "_run_plan_phase", plan_then_write)
    with workflow:
        workflow.run(task_file)

    workspace = tmp_path / "ws"
    journal = (workspace / "journal.md").read_text()
    assert "early delivery checks (end of the plan phase)" in journal
    assert "early delivery checks (after the core unit)" in journal
    text = (workspace / "early_checks.md").read_text()
    assert "# Early delivery checks (after the core unit)" in text
    assert "## Required CI assets" in text
    assert load_state(workspace / ".staircase_onboard_state.json").verdict == "delivered"
