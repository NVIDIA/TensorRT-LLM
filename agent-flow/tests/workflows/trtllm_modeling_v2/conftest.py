# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fixtures shared across the trtllm_modeling_v2 tests.

``repo`` is a minimal TensorRT-LLM checkout holding one merged target,
``checkpoint`` a HuggingFace checkpoint directory for it, and ``task_file`` a
task spec that points at both.
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
            # The fake shells answer the unit gates only.
            "envelope": {"enabled": False},
        },
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _planned_runs_start_from_the_planners_graph(monkeypatch):
    """A scripted planner here writes the whole graph; the survey is not started for it."""
    from agent_flow.workflows.trtllm_modeling_v2.run.workflow import TrtllmModelingV2Workflow

    monkeypatch.setattr(TrtllmModelingV2Workflow, "survey_first", False)


@pytest.fixture(autouse=True)
def _git_stays_under_the_base_temp(tmp_path_factory, monkeypatch):
    """Stop git's repository search at pytest's base temp directory.

    A fixture tree that is not a checkout has to read as not one, even when
    the base temp directory sits inside some other checkout.
    """
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path_factory.getbasetemp()))


@pytest.fixture(autouse=True)
def _no_parent_codex_session(monkeypatch, tmp_path):
    """Pin fake models; offline tests never inspect the developer's live session.

    A Codex role seeds the run's Codex home from ``$CODEX_HOME`` and keeps its
    databases under ``$TMPDIR``: neither is the developer's.
    """
    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    monkeypatch.setenv("CLAUDE_CODE_DEFAULT_MODEL", "claude-test-model")
    monkeypatch.setenv("CODEX_DEFAULT_MODEL", "codex-test-model")
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "user-codex-home"))
    monkeypatch.setenv("TMPDIR", str(tmp_path))


@pytest.fixture(autouse=True)
def _no_agent_probe(monkeypatch):
    """Keep preflight's agent probe from starting a real agent turn; its own tests unset this."""
    monkeypatch.setenv("MV2FLOW_SKIP_AGENT_PROBE", "1")


@pytest.fixture(autouse=True)
def _no_dataset_downloads(monkeypatch):
    """Keep the run's dataset staging off the network; ``test_datasets.py`` calls it itself."""
    from agent_flow.workflows.trtllm_modeling_v2.run.startup import datasets

    monkeypatch.setattr(datasets, "start", lambda *args, **kwargs: None)
