# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Ordinary boot owns optional boundary phases, using one public LLM instance."""

from __future__ import annotations

import contextlib
import io
import json
import shlex
import sys
from types import SimpleNamespace

from agent_flow.workflows.trtllm_modeling_v2.checks import boot as boot_checks
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import engine_driver, workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult

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


def _run_boot(tmp_path, monkeypatch, *, mutate=None, task=None, unit=None, member=None):
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
        _env(), task_data=task or _TASK, runner=run, workdir=tmp_path, unit=unit, member=member
    )
    return result, specs, events


def _parted(gap):
    """Probe0's loaded leg takes another token at step 2, ``gap`` nats behind in both legs."""

    def mutate(record):
        if record.get("id") != "probe0":
            return
        own = 2 if record.get("phase") == "baseline" else 9
        other = 9 if own == 2 else 2
        if record.get("phase") == "loaded":
            record["token_ids"] = [0, 1, 9, 3]
        rows = [[[t, 0.0, 1]] for t in record["token_ids"]]
        rows[2] = [[own, -0.1, 1], [other, -0.1 - gap, 2]]
        record["top_logprobs"] = rows

    return mutate


def test_a_probe_that_parts_at_a_near_tie_passes_and_past_the_margin_fails(tmp_path, monkeypatch):
    result, specs, events = _run_boot(tmp_path, monkeypatch, mutate=_parted(0.05))
    assert result.passed, result.detail
    assert "near-tie(s) (probe0 step 2: baseline 2, loaded 9, 0.050 apart)" in result.detail
    # The probes ask the engine for their top log-probs; the fillers do not.
    probes = [r for p in specs[0]["phases"] for r in p["requests"] if r["id"].startswith("probe")]
    assert probes and all(r["logprobs"] == boot_checks.BOUNDARY_TOP_K for r in probes)
    asked = [e[2] for e in events if e[0] == "submit" and getattr(e[2], "logprobs", 0)]
    assert len(asked) == len(probes)
    result, _, _ = _run_boot(tmp_path, monkeypatch, mutate=_parted(2.0))
    assert not result.passed
    assert "must agree on 4 tokens" in result.detail
    assert "past the near-tie margin 0.150" in result.detail
