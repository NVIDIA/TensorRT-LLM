# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One decode CUDA-graph capture list per unit, whichever gate builds the engine.

The accuracy gate's ``trtllm-eval`` reads the unit's options file, and
``quickstart_advanced.py`` (boot, acceptance, envelope) builds its
``CudaGraphConfig`` from its own flags.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import survey
from agent_flow.workflows.trtllm_modeling_v2.checks import gates
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import resolve_environment
from agent_flow.workflows.trtllm_modeling_v2.run import preflight
from agent_flow.workflows.trtllm_modeling_v2.run.stages.plan import PlanStage

PADDED_19 = (1, 2, 4, *range(8, 129, 8))


def _options(tmp_path: Path, name: str, data: dict) -> Path:
    path = tmp_path / f"{name}.llm_api.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def _task(tmp_path: Path, *, options: dict | None = None, runtime: dict | None = None):
    task: dict = {
        "checkpoint_path": "/ckpt/model",
        "parallel": "tp1",
        "trtllm_repo_path": str(tmp_path / "repo"),
        "features": ["mtp"],
        "runtime": runtime
        if runtime is not None
        else {"cuda_graph": True, "extra_flags": ["--max_batch_size=128"]},
        "accuracy": {"benchmark": "gsm8k"},
    }
    if options is not None:
        task["accuracy"]["llm_api_options"] = {
            unit: str(_options(tmp_path, unit, data)) for unit, data in options.items()
        }
    return task


_PADDED_128 = {"cuda_graph_config": {"enable_padding": True, "max_batch_size": 128}}


def _env():
    return resolve_environment(
        {"environment": {"kind": "local", "python": "/opt/venv/bin/python", "env": {}}},
        devices=(0,),
        world_size=1,
    )


def test_boot_acceptance_and_accuracy_capture_the_same_list(tmp_path):
    """The live defect: a padded options file, and a runtime block that says nothing of it."""
    options = {**_PADDED_128, "speculative_config": {"decoding_type": "MTP", "max_draft_len": 3}}
    task = _task(
        tmp_path,
        options={"mtp": options},
        runtime={"cuda_graph": True, "extra_flags": ["--max_batch_size=128"]},
    )
    words = gates.quickstart_words(
        task, _env(), unit="mtp", features=["mtp"], max_tokens=16, member=None
    )
    quickstart = gates.quickstart_capture(words)
    accuracy = gates.options_capture(
        gates.load_llm_api_options(gates.resolve_llm_api_options(task["accuracy"], "mtp"))
    )
    assert quickstart == accuracy
    assert quickstart.batch_sizes == PADDED_19 and quickstart.enable_padding
    assert "--spec_decode_max_draft_len=3" in words


def test_preflight_refuses_runtime_flags_against_the_options_file(tmp_path):
    task = _task(
        tmp_path,
        options={"core": _PADDED_128},
        runtime={"cuda_graph": True, "extra_flags": ["--cuda_graph_batch_sizes 1 2 4"]},
    )
    with pytest.raises(preflight.CaptureListMismatchError) as excinfo:
        preflight.check_capture_lists(task)
    message = str(excinfo.value)
    assert "`core`" in message
    assert "3 batches [1, 2, 4], padded" in message
    assert "19 batches [1, 2, 4, 8, 16, 24," in message
    assert "core.llm_api.yaml" in message
    assert "`cuda_graph_config` in the unit's options file" in message
    assert isinstance(excinfo.value, preflight.SpecInconsistencyError)


class _Survey(PlanStage):
    def __init__(self, tmp_path: Path, task_data: dict) -> None:
        self.workspace = tmp_path / "ws"
        self.task_path = tmp_path / "task.yaml"
        self.plan_path = self.workspace / "plan.md"
        self.vocabulary_path = self.workspace / "vocabulary.yaml"
        self._surface_block = "## Surface survey\n"
        self._runtime_block = ""
        self.task_data = task_data

    def _report_instruction(self, role: str) -> str:
        return f"report as {role}"


def test_the_surveyor_turn_carries_the_block(tmp_path):
    task = _task(tmp_path, options={"core": _PADDED_128})
    prompt = _Survey(tmp_path, task)._surveyor_prompt()
    assert survey.CAPTURE_LISTS_HEADING in prompt
    assert "19 batches [1, 2, 4, 8, 16," in prompt
