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

from ._helpers import _build_repo

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
        "agents": {"defaults": {"backend": "claude-code", "model": "claude-test-model"}},
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
        },
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def _runs_start_past_the_survey(monkeypatch):
    """A scripted run here starts from the workspace's vocabulary; a survey test turns it on."""
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
    """Offline tests never touch the developer's Codex state or live session.

    ``$CODEX_HOME`` and ``$TMPDIR`` are the test's own.
    """
    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    monkeypatch.delenv("CODEX_SQLITE_HOME", raising=False)
    monkeypatch.setenv("CODEX_HOME", str(tmp_path / "user-codex-home"))
    monkeypatch.setenv("TMPDIR", str(tmp_path))


@pytest.fixture(autouse=True)
def _no_agent_probe(monkeypatch):
    """Keep preflight's agent probe from starting a real agent turn; its own tests unset this."""
    monkeypatch.setenv("MV2FLOW_SKIP_AGENT_PROBE", "1")


@pytest.fixture(autouse=True)
def _no_dataset_downloads(monkeypatch):
    """Keep the run's dataset staging off the network; its own tests call ``stage`` themselves."""
    from agent_flow.workflows.trtllm_modeling_v2.checks import datasets

    monkeypatch.setattr(datasets, "start", lambda *args, **kwargs: None)
