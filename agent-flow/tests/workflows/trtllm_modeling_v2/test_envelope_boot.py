# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The envelope boot: the target served at its declared limits."""

from __future__ import annotations

import json
import re
from pathlib import Path

from agent_flow.workflows.trtllm_modeling_v2.checks import envelope_boot as eb
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import resolve_environment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult

_TASK = {
    "checkpoint_path": "/ckpt/Qwen3-30B-A3B",
    "parallel": "tp1",
    "trtllm_repo_path": "/repo",
    "runtime": {
        "cuda_graph": True,
        "chunked_prefill": True,
        "extra_flags": ["--max_seq_len=4096", "--max_batch_size=8", "--max_num_tokens=1024"],
    },
    "accuracy": {"benchmark": "gsm8k", "reference_yaml": "x.yaml"},
}


def _env():
    return resolve_environment(
        {"environment": {"kind": "local", "python": "/opt/venv/bin/python", "env": {}}},
        devices=(0,),
        world_size=1,
    )


def _engine():
    """A runner that answers each driver spec with one record per request."""
    seen: list[dict] = []

    def run(command, env=None, timeout=0):
        spec_path = re.search(r"--spec (\S+)", command).group(1)
        spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        seen.append(spec)
        lines = []
        for n, request in enumerate(spec["requests"]):
            ids = [7, 8, 9, 10, 11]
            tokens = len(request["token_ids"]) if "token_ids" in request else 5
            record = {"index": n, "id": request["id"], "token_ids": ids, "prompt_tokens": tokens}
            lines.append(f"{workload_io.RECORD_MARKER} {json.dumps(record)}")
        return CommandResult(returncode=0, stdout="\n".join(lines) + "\n")

    run.seen = seen  # type: ignore[attr-defined]
    return run


def test_it_passes_when_every_probe_agrees_with_batch_one(tmp_path):
    run = _engine()
    result = eb.run_envelope_boot(_env(), task_data=_TASK, runner=run, workdir=tmp_path)
    assert result.passed, result.detail
    assert result.name == "envelope"
    reference, loaded = run.seen  # type: ignore[attr-defined]
    assert reference["concurrency"] == 1
    assert loaded["concurrency"] == 8
    assert "TRTLLM_MODELING_V2=require" in result.command
    assert "TLLM_AUTOTUNER_CACHE_PATH" in result.command
