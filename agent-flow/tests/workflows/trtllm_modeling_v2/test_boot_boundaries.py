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
from agent_flow.workflows.trtllm_modeling_v2.checks import check_config
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
    "request_id,change,reason",
    [
        ("probe0", lambda r: r.update(token_ids=[9, 1, 2, 3]), "must agree"),
        ("probe0", lambda r: r.update(token_ids=[0, 1]), "must agree"),
        ("fill0", lambda r: r.update(token_ids=[]), "generated nothing"),
        ("probe0", dict.clear, "no record"),
        ("long0", lambda r: r.pop("prompt_tokens"), "prompt tokens"),
        ("long0", lambda r: r.update(prompt_tokens=3), "prompt tokens"),
    ],
)
def test_boot_refuses_incomplete_boundary_evidence(
    tmp_path, monkeypatch, request_id, change, reason
):
    def mutate(record):
        if record.get("phase") == "loaded" and record["id"] == request_id:
            change(record)

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


def test_unchunked_variant_owes_no_split_prompt(tmp_path, monkeypatch):
    """A variant that turns chunked prefill off is about chunking, but its engine never splits."""
    task = copy.deepcopy(_TASK)
    task["accuracy"].pop("envelope")
    task["variants"] = {
        "unchunked": {
            "unit": "core",
            "runtime": {
                "chunked_prefill": False,
                "extra_flags": ["--max_seq_len=64", "--max_batch_size=8", "--max_num_tokens=64"],
            },
        }
    }
    task = check_config.variant_spec(task, "unchunked", world_size=1)
    result, specs, _ = _run_boot(tmp_path, monkeypatch, task=task, unit="unchunked")
    # The engine ran (this fake prints no engine record, which the attestation then names);
    # before, the gate refused it for want of a prompt an unchunked engine would split.
    assert specs, result.detail
    assert "leave no room" not in result.detail
    assert not any(r.get("id") == "chunk" for r in specs[0]["requests"])


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


def test_boundaries_run_once_per_unit_at_the_member_that_batches_most(tmp_path, monkeypatch):
    assert boot_checks.boundary_member(["tp4", "tep4", "dep4"]) == "dep4"
    assert boot_checks.boundary_member(["tp4", "tep8"]) == "tep8"
    task = copy.deepcopy(_TASK)
    task.update(parallel="unified", members={"default": ["tp4", "tep4", "dep4"]})
    task["variants"] = {"kv_fp8": {"llm_api_options": {"kv_cache_config": {"dtype": "fp8"}}}}
    spec = boot_checks.BootBoundarySpec.from_task(task)
    assert [m for m in ("tp4", "tep4", "dep4") if spec.runs_at(task, "core", m, 4)] == ["dep4"]
    assert not spec.runs_at(task, "kv_fp8", "dep4", 4)
    task["accuracy"]["envelope"]["members"] = ["tp4"]
    spec = boot_checks.BootBoundarySpec.from_task(task)
    assert [m for m in ("tp4", "tep4", "dep4") if spec.runs_at(task, "core", m, 4)] == ["tp4"]
    # A boot at another member runs none.
    task = copy.deepcopy(_TASK)
    task["accuracy"]["envelope"]["members"] = ["tp2"]
    result, specs, _ = _run_boot(tmp_path, monkeypatch, task=task, member="tp1")
    assert result.passed, result.detail
    assert not specs[0].get("phases")
    assert "boot boundaries" not in result.detail


def test_the_spec_names_a_near_tie_margin_and_boundary_members():
    from agent_flow.workflows.trtllm_modeling_v2.spec import task_schema

    def errors(block):
        found = []
        task_schema._validate_envelope(block, found)
        return found

    assert errors({"enabled": True, "tie_margin": 0.2, "members": ["dep4"]}) == []
    assert any("tie_margin" in e for e in errors({"tie_margin": -1}))
    assert any("members" in e for e in errors({"members": ["four"]}))
