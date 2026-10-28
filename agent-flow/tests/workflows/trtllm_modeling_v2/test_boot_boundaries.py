# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordinary boot owns optional boundary phases, using one public LLM instance."""

from __future__ import annotations

import contextlib
import copy
import io
import json
import shlex
import sys
from types import SimpleNamespace

import pytest

from agent_flow.workflows.trtllm_modeling_v2.checks import boot as boot_checks
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import engine_driver, workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.run.validation import contract

from .test_gates import _env

_TASK = {
    "checkpoint_path": "/ckpt/Qwen3-30B-A3B",
    "parallel": "tp1",
    "trtllm_repo_path": "/repo",
    "runtime": {
        "cuda_graph": True,
        "chunked_prefill": True,
        "extra_flags": ["--max_seq_len=64", "--max_batch_size=8", "--max_num_tokens=32"],
    },
    "accuracy": {
        "benchmark": "gsm8k",
        "reference_yaml": "x.yaml",
        "envelope": {"enabled": True},
    },
}


def _run_boot(tmp_path, monkeypatch, *, mutate=None, task=None):
    events = []
    specs = []
    active = 0

    class Params:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class LLM:
        def __init__(self, **kwargs):
            events.append(("construct", kwargs))
            self.tokenizer = SimpleNamespace(
                apply_chat_template=lambda *args, **kwargs: "chat: " + args[0][0]["content"]
            )

        def generate_async(self, prompt, params, streaming=False):
            nonlocal active
            active += 1
            events.append(("submit", prompt, params, active))
            output = SimpleNamespace(
                prompt_token_ids=(prompt["prompt_token_ids"] if isinstance(prompt, dict) else [1]),
                cached_tokens=0,
                outputs=[
                    SimpleNamespace(
                        token_ids=list(range(params.max_tokens)),
                        text="Paris",
                        finish_reason="length",
                    )
                ],
            )

            def stream():
                nonlocal active
                yield output
                active -= 1
                events.append(("drain", prompt))

            return stream()

        def shutdown(self):
            events.append(("shutdown",))

    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm",
        SimpleNamespace(LLM=LLM, SamplingParams=Params, __file__="/fake"),
    )
    monkeypatch.setattr(engine_driver.engine_facts, "emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(engine_driver.reuse_probe, "probe", lambda *args: events.append(("reuse",)))

    def run(command, env=None, timeout=0):
        words = shlex.split(command)
        path = words[words.index("--spec") + 1]
        from pathlib import Path

        specs.append(json.loads(Path(path).read_text()))
        capture = io.StringIO()
        with contextlib.redirect_stdout(capture):
            assert engine_driver.main(["--spec", path]) == 0
        output = capture.getvalue()
        if mutate:
            lines = []
            for line in output.splitlines():
                if line.startswith(workload_io.RECORD_MARKER):
                    record = json.loads(line[len(workload_io.RECORD_MARKER) :])
                    mutate(record)
                    if record:
                        lines.append(f"{workload_io.RECORD_MARKER} {json.dumps(record)}")
                else:
                    lines.append(line)
            output = "\n".join(lines)
        return CommandResult(returncode=0, stdout=output)

    result = boot_checks.run_boot_gate(
        _env(), task_data=task or _TASK, runner=run, workdir=tmp_path
    )
    return result, specs, events


def test_ordinary_boot_runs_boundaries_in_one_engine_and_one_contract_check(tmp_path, monkeypatch):
    task = copy.deepcopy(_TASK)
    task["runtime"]["extra_flags"].append("--apply_chat_template")
    result, specs, events = _run_boot(tmp_path, monkeypatch, task=task)
    assert result.passed, result.detail
    assert result.name == "boot"
    assert len(specs) == 1
    assert [e[0] for e in events].count("construct") == 1
    assert events[-2:] == [("reuse",), ("shutdown",)]
    spec = specs[0]
    assert [p["id"] for p in spec["phases"]] == ["baseline", "loaded"]
    assert [p["concurrency"] for p in spec["phases"]] == [1, 8]
    submitted = [e for e in events if e[0] == "submit"]
    assert all(e[3] == 1 for e in submitted[:6])  # ordinary prompts, then baseline
    assert max(e[3] for e in submitted[6:]) == 8
    assert submitted[2][2].__dict__ == submitted[8][2].__dict__
    assert "TLLM_AUTOTUNER_CACHE_PATH" in result.command
    assert "actual scheduler batches and CUDA graph execution are not measured" in result.detail
    compiled = contract.compile_contract(task, 1)
    assert [c.gate for c in compiled.checks] == ["boot", "accuracy[gsm8k]"]
    disabled = copy.deepcopy(task)
    disabled["accuracy"]["envelope"]["enabled"] = False
    assert (
        compiled.checks[0].equivalence
        != contract.compile_contract(disabled, 1).checks[0].equivalence
    )


@pytest.mark.parametrize(
    "phase,request_id,field,value,missing,reason",
    [
        ("loaded", "probe0", "token_ids", [9, 1, 2, 3], False, "must agree"),
        ("loaded", "probe0", "token_ids", [0, 1], False, "must agree"),
        ("baseline", "probe0", "token_ids", [0, 1], False, "must agree"),
        ("loaded", "probe0", "token_ids", [], False, "generated nothing"),
        ("loaded", "probe0", "token_ids", None, True, "generated nothing"),
        ("loaded", "probe0", "record", None, True, "no record"),
        ("loaded", "long0", "prompt_tokens", None, True, "prompt tokens"),
        ("loaded", "long0", "prompt_tokens", 3, False, "prompt tokens"),
        ("loaded", "long0", "token_ids", [], False, "generated nothing"),
        ("loaded", "fill0", "token_ids", [], False, "generated nothing"),
    ],
)
def test_boot_refuses_incomplete_boundary_evidence(
    tmp_path, monkeypatch, phase, request_id, field, value, missing, reason
):
    def mutate(record):
        if record.get("phase") == phase and record["id"] == request_id:
            if field == "record":
                record.clear()
            elif missing:
                record.pop(field)
            else:
                record[field] = value

    result, _, _ = _run_boot(tmp_path, monkeypatch, mutate=mutate)
    assert not result.passed
    assert reason in result.detail


def test_boundaries_are_opt_in_and_unchunked_length_is_reported_honestly(tmp_path, monkeypatch):
    task = copy.deepcopy(_TASK)
    task["accuracy"].pop("envelope")
    result, specs, _ = _run_boot(tmp_path, monkeypatch, task=task)
    assert result.passed, result.detail
    assert "phases" not in specs[0]
    task["accuracy"]["envelope"] = {"enabled": True}
    task["runtime"]["chunked_prefill"] = False
    result, specs, _ = _run_boot(tmp_path, monkeypatch, task=task)
    assert result.passed, result.detail
    assert len(specs[0]["phases"][1]["requests"][0]["token_ids"]) == 31
    assert "exactly 31 input tokens" in result.detail
    assert "capped by unchunked" in result.detail
