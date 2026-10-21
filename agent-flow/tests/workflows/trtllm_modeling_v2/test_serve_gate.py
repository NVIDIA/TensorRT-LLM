# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The serve gate: `trtllm-serve` starts the unit's target and answers over HTTP."""

from __future__ import annotations

import json

from agent_flow.workflows.trtllm_modeling_v2.checks import serve_driver
from agent_flow.workflows.trtllm_modeling_v2.checks import serve_gate as sg
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import resolve_environment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult

_MEMBERS = ["tp1", "tp2", "tep2", "dep2", "tp4", "tep4", "dep4"]


def _task(tmp_path):
    options = tmp_path / "core.llm_api.yaml"
    options.write_text("kv_cache_config: {free_gpu_memory_fraction: 0.6}\n", encoding="utf-8")
    return {
        "checkpoint_path": "/ckpt/model",
        "trtllm_repo_path": str(tmp_path / "repo"),
        "parallel": "unified",
        "members": list(_MEMBERS),
        "features": ["mtp", "multimodal"],
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": "x.yaml",
            "llm_api_options": str(options),
            "max_seq_len": 9216,
            "max_batch_size": 128,
        },
    }


def _env(world_size):
    block = {"kind": "local", "python": "/venv/bin/python", "env": {"HF_HUB_OFFLINE": "1"}}
    return resolve_environment({"environment": block}, devices=(), world_size=world_size)


def _server(healthy=True, **extra):
    return f"{serve_driver.SERVER_MARKER} {json.dumps({'healthy': healthy, **extra})}\n"


def _answer(index, text="Paris", status=200, **extra):
    record = {"index": index, "status": status, "text": text, **extra}
    return f"{workload_io.RECORD_MARKER} {json.dumps(record)}\n"


def _recorder(output=""):
    seen = []

    def run(command, env=None, timeout=0):
        seen.append((command, timeout))
        return CommandResult(returncode=0, stdout=output)

    run.seen = seen  # type: ignore[attr-defined]
    return run


def test_the_gate_reads_the_driver_and_names_the_member(tmp_path):
    run = _recorder(_answer(0, " Paris.") + _server(boot_s=3.0))
    result = sg.run_serve_gate(
        _env(1),
        task_data=_task(tmp_path),
        runner=run,
        features=[],
        unit="core",
        member="tp1",
        workdir=tmp_path,
    )
    assert result.passed, result.detail
    assert result.summary().startswith("[PASS] serve@tp1: trtllm-serve healthy after 3.0s")
