# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures shared across the trtllm_modeling_v2 tests.

``repo`` is a minimal TensorRT-LLM checkout holding one merged target,
``checkpoint`` a HuggingFace checkpoint directory for it, and ``task_file`` a
task spec that points at both. pytest resolves them by name, so no test
module imports another to get them.
"""

from __future__ import annotations

import json

import pytest
import yaml

from .test_delivery import _build_repo

SCRATCH = "/scratch/mv2flow-run-tmp"


@pytest.fixture
def repo(tmp_path):
    return _build_repo(tmp_path)


@pytest.fixture
def checkpoint(tmp_path):
    path = tmp_path / "gpt-oss-120b"
    path.mkdir()
    (path / "config.json").write_text(json.dumps({"architectures": ["GptOssForCausalLM"]}))
    return path


@pytest.fixture
def task_file(tmp_path, repo, checkpoint):
    spec = {
        "checkpoint_path": str(checkpoint),
        "arch": "sm_103",
        "parallel": "tp1",
        "trtllm_repo_path": str(repo),
        "target": {"family": "gpt_oss", "checkpoint_segment": "gpt_oss_120b"},
        "environment": {
            "kind": "local",
            "python": "/opt/venv/bin/python",
            "env": {"TMPDIR": SCRATCH, "CLAUDE_CODE_TMPDIR": SCRATCH},
        },
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": "tests/integration/defs/accuracy/references/gsm8k.yaml",
            "model_key": "openai/gpt-oss-120b",
            "tolerance": 5.0,
            # The fake shells answer the unit gates only; the envelope boot
            # has its own tests (test_envelope_boot.py) and one through delivery.
            "envelope": {"enabled": False},
        },
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _planned_runs_start_from_the_planners_graph(monkeypatch):
    """A scripted planner here writes the whole graph; the survey is not started for it.

    A new run starts with the survey beside the planner's first turn (phase 0,
    ``PlanStage.survey_first``), which would add a survey turn every scripted
    run had to expect. Its own tests turn it back on.
    """
    from agent_flow.workflows.trtllm_modeling_v2.run.stages.plan import PlanStage

    monkeypatch.setattr(PlanStage, "survey_first", False)


@pytest.fixture(autouse=True)
def _git_stays_under_the_base_temp(tmp_path_factory, monkeypatch):
    """Stop git's repository search at pytest's base temp directory.

    The workflow asks git about the trees these tests build -- delivery, the
    commit audit and the target marker all do -- and git searches upward
    from where it starts. A fixture tree that is not a checkout has to read
    as not one; but when the base temp directory sits inside some other
    checkout (an ignored build directory, say), the search would reach that
    checkout and answer for it instead.

    git never enters a ceiling directory but still examines the ones below
    it, so a repository a test creates at its ``tmp_path`` -- a child of the
    base temp directory -- is found from anywhere inside it.
    """
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path_factory.getbasetemp()))


@pytest.fixture(autouse=True)
def _no_agent_probe(monkeypatch):
    """Keep preflight's agent probe from starting a real agent turn.

    These tests script every turn through fakes of the role layers, and the
    probe builds a layer of its own. Its tests (test_agent_probe.py) unset this.
    """
    monkeypatch.setenv("MV2FLOW_SKIP_AGENT_PROBE", "1")
