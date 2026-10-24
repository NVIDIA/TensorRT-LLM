# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fast path: a request derived from the checkpoint, the site and a goal, checked in phase 0."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from agent_flow.logger import get_logger
from agent_flow.workflows.trtllm_modeling_v2 import cli
from agent_flow.workflows.trtllm_modeling_v2.run.state import (
    STAGE_PLANNED,
    STATE_FILENAME,
    WorkflowState,
    save_state,
)
from agent_flow.workflows.trtllm_modeling_v2.spec.intake import derive
from agent_flow.workflows.trtllm_modeling_v2.spec.intake.checks import blocking, run_checks
from agent_flow.workflows.trtllm_modeling_v2.spec.intake.schema import IntakeError
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import load_and_validate_task_yaml

from .conftest import SCRATCH
from .test_workflow import _workflow


def _checkpoint(root: Path) -> Path:
    path = root / "newmodel"
    path.mkdir(parents=True, exist_ok=True)
    base = {
        "architectures": ["NewModelForCausalLM"],
        "_name_or_path": "org/newmodel",
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "num_experts": 16,
        "max_position_embeddings": 32768,
        "num_nextn_predict_layers": 1,
        "vision_config": {"model_type": "vit"},
    }
    (path / "config.json").write_text(json.dumps(base))
    return path


def _repo(root: Path) -> Path:
    repo = root / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    return repo


def _site(root: Path, repo: Path) -> dict:
    return {
        "trtllm_repo_path": str(repo),
        "environment": {
            "kind": "session_spool",
            "python": "/opt/venv/bin/python",
            "spool": str(root / "spool"),
            "env": {"TMPDIR": SCRATCH, "CLAUDE_CODE_TMPDIR": SCRATCH},
        },
        "device": {"arch": "sm_100", "gpus_per_node": 4},
        "defaults": {"members": ["tp1", "tep4", "dep4"]},
    }


def _site_file(root: Path, site: dict) -> Path:
    (root / "spool").mkdir(exist_ok=True)
    path = root / "site.yaml"
    path.write_text(yaml.safe_dump(site))
    return path


def test_features_the_checkpoint_carries_are_built_with_their_provenance(tmp_path):
    repo = _repo(tmp_path)
    derivation = derive.derive_task(
        _checkpoint(tmp_path), _site(tmp_path, repo), derive.parse_goal("")
    )
    task = derivation.task
    assert task["features"]["speculative"] == {
        "status": "required",
        "params": {"algo": "mtp", "draft_len": 1},
    }
    assert task["features"]["multimodal"] == {"status": "required"}
    assert task["features"]["cuda_graph"] == {"status": "required"}
    assert [f"{p['kind']}{p['world']}" for p in task["parallelism"]] == ["tp1", "tep4", "dep4"]
    assert task["workload"] == derive.DEFAULT_WORKLOAD
    assert task["accuracy"]["model_key"] == "org/newmodel"
    assert task["accuracy"]["boot"] == derive.DEFAULT_BOOT
    assert task["provenance"]["features.speculative.params.algo"] == "inferred"
    assert task["provenance"]["workload.isl"] == "default"
    assert task["provenance"]["parallelism"] == "inferred"
    assert task["open_questions"] == []
    assert task["derived"]["goal"] == ""
    assert blocking(run_checks(task)) == []


def test_the_goal_overrides_what_the_checkpoint_and_the_site_say(tmp_path):
    repo = _repo(tmp_path)
    goal = derive.parse_goal("-mtp text_only tep4 isl=1024 osl=256 arch=sm_103")
    task = derive.derive_task(_checkpoint(tmp_path), _site(tmp_path, repo), goal).task
    assert task["features"]["speculative"] == {"status": "out_of_scope"}
    assert task["features"]["multimodal"] == {"status": "out_of_scope"}
    assert task["parallelism"] == [{"kind": "tep", "world": 4}]
    assert task["workload"]["isl"] == 1024 and task["provenance"]["workload.isl"] == "user"
    assert task["device"]["arch"] == "sm_103"


def _fast(tmp_path, site_path, goal=None, confirm_by=None, clean=False):
    return cli._derive_intake_task(
        _checkpoint(tmp_path),
        site_path,
        tmp_path / "ws",
        goal=goal,
        confirm_by=confirm_by,
        clean=clean,
    )


def _start(workspace: Path) -> None:
    save_state(
        workspace / STATE_FILENAME,
        WorkflowState(task_path=str(workspace / "task.yaml"), stage=STAGE_PLANNED),
    )


def test_an_edit_before_the_run_starts_is_taken_and_after_it_is_refused(tmp_path):
    site_path = _site_file(tmp_path, _site(tmp_path, _repo(tmp_path)))
    _, request = _fast(tmp_path, site_path, goal="-multimodal")
    text = request.read_text()
    request.write_text(text.replace("osl: 1024", "osl: 512"))
    resolved, _ = _fast(tmp_path, site_path)
    assert "--max_seq_len=4608" in load_and_validate_task_yaml(resolved)["runtime"]["extra_flags"]

    _start(tmp_path / "ws")
    _fast(tmp_path, site_path)  # unchanged since the run started
    request.write_text(request.read_text().replace("osl: 512", "osl: 256"))
    with pytest.raises(IntakeError, match="changed after its run started"):
        _fast(tmp_path, site_path)
    _fast(tmp_path, site_path, confirm_by="reviewer")
    request.write_text(request.read_text().replace("osl: 256", "osl: 128"))
    with pytest.raises(IntakeError, match="changed after it was confirmed"):
        _fast(tmp_path, site_path)


def test_another_goal_or_checkpoint_needs_clean(tmp_path):
    site_path = _site_file(tmp_path, _site(tmp_path, _repo(tmp_path)))
    _fast(tmp_path, site_path, goal="-multimodal")
    with pytest.raises(IntakeError, match="--clean"):
        _fast(tmp_path, site_path, goal="-multimodal -mtp")
    _, request = _fast(tmp_path, site_path, goal="-multimodal -mtp", clean=True)
    assert yaml.safe_load(request.read_text())["features"]["speculative"]["status"] == (
        "out_of_scope"
    )


def test_confirming_a_request_with_a_block_is_refused_with_every_block(tmp_path):
    site = _site(tmp_path, _repo(tmp_path))
    site.pop("device")
    site_path = _site_file(tmp_path, site)
    with pytest.raises(IntakeError) as raised:
        _fast(tmp_path, site_path, goal="disagg", confirm_by="reviewer")
    assert "device.arch" in str(raised.value) and "contrast_command" not in str(raised.value)


def _request(tmp_path, checkpoint: Path, **changes) -> Path:
    task = {
        "version": 1,
        "goals": ["onboard"],
        "model": {"checkpoint": str(checkpoint)},
        "device": {"arch": "sm_103"},
        "parallelism": [{"kind": "tp", "world": 1}],
        "workload": {"isl": 1024, "osl": 256},
        **changes,
    }
    path = tmp_path / derive.REQUEST_FILENAME
    path.write_text(yaml.safe_dump(task))
    return path


def test_phase_0_stops_on_every_intake_block_before_the_host_is_touched(
    tmp_path, task_file, checkpoint
):
    checkpoint.joinpath("config.json").write_text(
        json.dumps({"architectures": ["GptOssForCausalLM"], "num_attention_heads": 8})
    )
    workflow = _workflow(tmp_path)
    log = get_logger().console
    state = workflow._init_state(task_file, log)
    workflow.intake_request = _request(
        tmp_path,
        checkpoint,
        parallelism=[{"kind": "tp", "world": 3}],
        open_questions=["which benchmark?"],
    )
    assert workflow._run_preflight(state, log) is False
    journal = (tmp_path / "ws" / "journal.md").read_text()
    assert "intake checks failed: 3 block(s)" in journal
    assert "8 heads do not split 3" in journal and "which benchmark?" in journal
    assert workflow.environment is None
