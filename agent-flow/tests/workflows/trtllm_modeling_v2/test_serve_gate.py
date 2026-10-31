# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The serve gate: `trtllm-serve` starts the unit's target and answers over HTTP."""

from __future__ import annotations

import json

from agent_flow.workflows.trtllm_modeling_v2.checks import serve as sg
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import serve_driver, workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import resolve_environment


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
