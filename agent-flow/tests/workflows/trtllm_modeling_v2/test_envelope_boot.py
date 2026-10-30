# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the envelope boot: the target served at its declared limits."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from agent_flow.workflows.trtllm_modeling_v2.checks import envelope_boot as eb
from agent_flow.workflows.trtllm_modeling_v2.checks import gates
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import resolve_environment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import _validate_accuracy

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


def _engine(*, under_load=None, exit_codes=(0, 0), prompt_tokens=None, log="", reused=None):
    """A runner that answers each driver spec with one record per request.

    ``under_load`` maps a probe id to the token ids it generates in the
    envelope leg; by default every probe generates the same ids in both legs.
    ``reused`` gives, per leg, the prompt tokens the reuse probe's second
    prompt resumed from the KV cache; without it no ``MV2FLOW_REUSE`` record
    is printed.
    """
    seen: list[dict] = []

    def run(command, env=None, timeout=0):
        spec_path = re.search(r"--spec (\S+)", command).group(1)
        spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
        seen.append(spec)
        leg = len(seen) - 1
        lines = []
        for n, request in enumerate(spec["requests"]):
            ids = [7, 8, 9, 10, 11]
            if leg == 1 and under_load and request["id"] in under_load:
                ids = under_load[request["id"]]
            tokens = len(request["token_ids"]) if "token_ids" in request else 5
            if prompt_tokens is not None and "token_ids" in request:
                tokens = prompt_tokens
            record = {"index": n, "id": request["id"], "token_ids": ids, "prompt_tokens": tokens}
            lines.append(f"{workload_io.RECORD_MARKER} {json.dumps(record)}")
        if reused is not None and spec.get("reuse_probe"):
            lines.append(_reuse_line(len(spec["requests"]), reused[leg]))
        return CommandResult(returncode=exit_codes[leg], stdout="\n".join(lines) + "\n", stderr=log)

    run.seen = seen  # type: ignore[attr-defined]
    return run


def _reuse_line(requests: int, reused: int) -> str:
    record = {
        "requests": requests,
        "reported": requests,
        "hits": 0,
        "reused_tokens": 0,
        "resumed_at": {},
        "min_resume": 32,
        "resumable": 0,
        "concurrency": 1,
        "probe": {"tokens": 96, "first": 0, "reused": reused},
    }
    return f"{workload_io.REUSE_MARKER} {json.dumps(record)}"


def test_the_envelope_is_read_from_the_gate_argv():
    words = [
        "--max_seq_len=4096",
        "--max_batch_size",
        "8",
        "--enable_chunked_prefill",
        "--use_cuda_graph",
        "--cuda_graph_batch_sizes",
        "1",
        "2",
        "8",
        "--max_tokens=16",
    ]
    envelope = eb.resolve_envelope(words)
    assert envelope.max_seq_len == 4096
    assert envelope.max_batch_size == 8
    assert envelope.max_num_tokens == eb.QUICKSTART_MAX_NUM_TOKENS
    assert envelope.chunked_prefill and envelope.cuda_graph
    assert envelope.capture_batches == (1, 2, 8)


def test_without_chunked_prefill_a_prompt_is_capped_at_one_chunk():
    chunked = eb.Envelope(8, 1024, 4096, True, False, ())
    unchunked = eb.Envelope(8, 1024, 4096, False, False, ())
    assert chunked.long_prompt_tokens(16) == 4096 - 16 - 1
    assert unchunked.long_prompt_tokens(16) == 1023


def test_the_loaded_leg_fills_the_batch_and_staggers_its_lengths():
    envelope = eb.Envelope(8, 1024, 4096, True, True, ())
    requests = eb.envelope_requests(envelope, eb.EnvelopeSpec(long_prompts=2))
    assert len(requests) == 8
    assert [r["id"] for r in requests[:2]] == ["long0", "long1"]
    assert len(requests[0]["token_ids"]) == 4096 - 16 - 1
    fillers = [r["max_tokens"] for r in requests if r["id"].startswith("fill")]
    assert len(set(fillers)) == len(fillers)


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


def test_a_probe_that_changes_under_load_fails_and_is_quoted(tmp_path):
    run = _engine(under_load={"probe1": [7, 8, 99, 10, 11]})
    result = eb.run_envelope_boot(_env(), task_data=_TASK, runner=run, workdir=tmp_path)
    assert not result.passed
    assert "probe1" in result.detail and "99" in result.detail


def test_a_divergence_after_the_compared_prefix_is_not_a_failure(tmp_path):
    run = _engine(under_load={"probe0": [7, 8, 9, 10, 99]})
    result = eb.run_envelope_boot(_env(), task_data=_TASK, runner=run, workdir=tmp_path)
    assert result.passed, result.detail


def test_a_crash_in_the_loaded_leg_fails_and_names_the_envelope(tmp_path):
    run = _engine(exit_codes=(0, 1))
    result = eb.run_envelope_boot(_env(), task_data=_TASK, runner=run, workdir=tmp_path)
    assert not result.passed
    assert "envelope leg exited 1" in result.detail
    assert "max_seq_len 4096" in result.detail


def test_a_long_prompt_that_did_not_reach_the_limit_fails(tmp_path):
    run = _engine(prompt_tokens=100)
    result = eb.run_envelope_boot(_env(), task_data=_TASK, runner=run, workdir=tmp_path)
    assert not result.passed
    assert "long0" in result.detail


def test_an_undeclared_sequence_limit_is_refused_before_any_job(tmp_path):
    task = {**_TASK, "runtime": {"extra_flags": ["--max_batch_size=8"]}}
    run = _engine()
    result = eb.run_envelope_boot(_env(), task_data=task, runner=run, workdir=tmp_path)
    assert not result.passed
    assert "--max_seq_len" in result.detail
    assert run.seen == []  # type: ignore[attr-defined]


def test_the_spec_block_is_read_and_validated():
    spec = eb.EnvelopeSpec.from_task(
        {"accuracy": {"envelope": {"max_tokens": 8, "probes": ["hi"], "enabled": False}}}
    )
    assert spec.max_tokens == 8 and spec.probes == ("hi",) and not spec.enabled
    errors: list[str] = []
    _validate_accuracy(
        {
            "benchmark": "gsm8k",
            "reference_yaml": "x.yaml",
            "envelope": {"probes": [], "max_tokens": 0, "bogus": 1},
        },
        errors,
    )
    joined = "\n".join(errors)
    assert "accuracy.envelope.probes" in joined
    assert "accuracy.envelope.max_tokens" in joined
    assert "bogus" in joined


@pytest.mark.parametrize("name", ["envelope"])
def test_the_gate_command_can_run_it(name):
    from agent_flow.workflows.trtllm_modeling_v2.checks.gate_cli import GATE_NAMES

    assert name in GATE_NAMES


_SNAPSHOTS = {"mamba_state_config": {"periodic_snapshot_interval": 2048}}


def _task_with_options(tmp_path):
    options = tmp_path / "core.llm_api.yaml"
    options.write_text(json.dumps({"kv_cache_config": _SNAPSHOTS}), encoding="utf-8")
    return {**_TASK, "accuracy": {**_TASK["accuracy"], "llm_api_options": {"core": str(options)}}}


def test_the_engine_gets_the_units_kv_cache_keys(tmp_path):
    run = _engine()
    result = eb.run_envelope_boot(
        _env(), task_data=_task_with_options(tmp_path), unit="core", runner=run, workdir=tmp_path
    )
    assert result.passed, result.detail
    assert gates.KV_CACHE_CONFIG_ENV in result.command
    assert "periodic_snapshot_interval" in result.command


def test_an_engine_that_turned_block_reuse_off_fails(tmp_path):
    log = (
        "[W] Disabling KV cache block reuse for the hybrid Mamba model because no Mamba "
        "state snapshot policy is configured.\n"
    )
    run = _engine(log=log)
    result = eb.run_envelope_boot(_env(), task_data=_TASK, runner=run, workdir=tmp_path)
    assert not result.passed
    assert result.detail.startswith("the reference leg: ")
    assert "mamba_state_config" in result.detail


def test_both_legs_send_the_reuse_probe_and_record_what_it_resumed(tmp_path):
    run = _engine(reused=(64, 64))
    result = eb.run_envelope_boot(_env(), task_data=_TASK, runner=run, workdir=tmp_path)
    assert result.passed, result.detail
    assert [spec.get("reuse_probe") for spec in run.seen] == [True, True]  # type: ignore[attr-defined]
    assert [e["reuse"]["probe"]["reused"] for e in result.engine] == [64, 64]


def test_an_envelope_leg_that_reused_nothing_fails_a_unit_that_asks_for_reuse(tmp_path):
    """An engine served at its limits that no longer resumes from its cache is caught."""
    run = _engine(reused=(64, 0))
    result = eb.run_envelope_boot(_env(), task_data=_TASK, runner=run, workdir=tmp_path)
    assert not result.passed
    assert result.detail.startswith(
        "the unit asks for KV-cache block reuse and the engine of the envelope leg reused nothing"
    )
    assert "96-token prompt twice" in result.detail
    assert result.engine[-1]["reuse"]["probe"]["reused"] == 0


def test_a_reference_leg_that_reused_nothing_stops_before_the_envelope_leg(tmp_path):
    run = _engine(reused=(0, 64))
    result = eb.run_envelope_boot(_env(), task_data=_TASK, runner=run, workdir=tmp_path)
    assert not result.passed
    assert "the engine of the reference leg reused nothing" in result.detail
    assert len(run.seen) == 1  # type: ignore[attr-defined]


def test_a_unit_that_turns_reuse_off_asks_for_none(tmp_path):
    task = {**_TASK, "runtime": {**_TASK["runtime"], "kv_cache_reuse": False}}
    run = _engine(reused=(0, 0))
    result = eb.run_envelope_boot(_env(), task_data=task, runner=run, workdir=tmp_path)
    assert result.passed, result.detail
    assert "--disable_kv_cache_reuse" in result.command
