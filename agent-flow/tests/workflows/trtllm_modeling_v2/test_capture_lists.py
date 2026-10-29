# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One decode CUDA-graph capture list per unit, whichever gate builds the engine.

The accuracy gate's ``trtllm-eval`` and boot/acceptance's public engine driver
receive the same resolved ``CudaGraphConfig`` options.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.agents.prompts import survey
from agent_flow.workflows.trtllm_modeling_v2.checks import check_config
from agent_flow.workflows.trtllm_modeling_v2.run.execution.tasks import _surveyor_prompt
from agent_flow.workflows.trtllm_modeling_v2.run.startup import preflight

from .test_gates import _env

PADDED_19 = (1, 2, 4, *range(8, 129, 8))


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
    for unit, data in (options or {}).items():
        path = tmp_path / f"{unit}.llm_api.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        task["accuracy"].setdefault("llm_api_options", {})[unit] = str(path)
    return task


_PADDED_128 = {"cuda_graph_config": {"enable_padding": True, "max_batch_size": 128}}


def test_boot_acceptance_and_accuracy_capture_the_same_list(tmp_path):
    """The live defect: a padded options file, and a runtime block that says nothing of it."""
    options = {**_PADDED_128, "speculative_config": {"decoding_type": "MTP", "max_draft_len": 3}}
    task = _task(tmp_path, options={"mtp": options})
    config = check_config.check_configuration(
        task, _env(), unit="mtp", features=["mtp"], max_tokens=16, member=None
    )
    quickstart = check_config.options_capture(config["llm"])
    accuracy = check_config.options_capture(
        check_config.load_llm_api_options(
            check_config.resolve_llm_api_options(task["accuracy"], "mtp")
        )
    )
    assert quickstart == accuracy
    assert quickstart.batch_sizes == PADDED_19 and quickstart.enable_padding
    assert config["llm"]["speculative_config"]["max_draft_len"] == 3


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
    assert "runtime batch sizes [1, 2, 4] conflict" in message
    assert "19 batches [1, 2, 4, 8, 16, 24," in message
    assert "core.llm_api.yaml" in message
    assert "`cuda_graph_config` in the unit's options file" in message
    assert isinstance(excinfo.value, preflight.SpecInconsistencyError)


class _Survey:
    def __init__(self, tmp_path: Path, task_data: dict) -> None:
        self.workspace = tmp_path / "ws"
        self.task_path = tmp_path / "task.yaml"
        self.plan_path = self.workspace / "plan.md"
        self.vocabulary_path = self.workspace / "vocabulary.yaml"
        self._surface_block = "## Surface survey\n"
        self.task_data = task_data

    def _report_instruction(self, role: str) -> str:
        return f"report as {role}"


def test_the_surveyor_turn_carries_the_block(tmp_path):
    task = _task(tmp_path, options={"core": _PADDED_128})
    prompt = _surveyor_prompt(_Survey(tmp_path, task))
    assert survey.CAPTURE_LISTS_HEADING in prompt
    assert "19 batches [1, 2, 4, 8, 16," in prompt
