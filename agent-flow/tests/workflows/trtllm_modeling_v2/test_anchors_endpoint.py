# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen anchors per deployment and benchmark, and the checks that go through an endpoint."""

from __future__ import annotations

import math
import re

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import anchors, endpoint, gate_cli, servers
from agent_flow.workflows.trtllm_modeling_v2.checks.boot import BootPrompt, BootSpec
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import endpoint_eval
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import resolve_environment
from agent_flow.workflows.trtllm_modeling_v2.spec import deployments
from agent_flow.workflows.trtllm_modeling_v2.spec.deployments import Benchmark, Deployment, Server

from ._helpers import _workflow
from .test_deployments import _delivered, _load, _native

REFERENCES = "tests/integration/defs/accuracy/references"


def _rows(tmp_path, benchmark: str, rows: dict) -> None:
    path = tmp_path / "repo" / REFERENCES / f"{benchmark}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(rows), encoding="utf-8")


# ------------------------------------------------------------------------------ anchors


def test_each_benchmark_of_each_deployment_gets_its_anchor_in_order(tmp_path):
    spec = _native(tmp_path)
    spec["deployments"].insert(
        1,
        {
            "name": "dep16_kv_fp8",
            "options": {"kv_cache_config": {"dtype": "fp8"}, "tensor_parallel_size": 16},
            "relative_to": "dep16",
            "benchmarks": [{"name": "gsm8k"}, {"name": "mmmu"}],
        },
    )
    spec["deployments"].append({"name": "lone", "options": {}, "benchmarks": [{"name": "mmmu"}]})
    _rows(
        tmp_path,
        "gsm8k",
        {
            "moonshotai/Kimi-K3": [
                {"accuracy": 96.5, "kv_cache_quant_algo": "NVFP4"},
                {"accuracy": 95.9, "kv_cache_quant_algo": "FP8"},
            ]
        },
    )
    loaded = _load(tmp_path, spec)
    found = anchors.freeze(tmp_path / "ws", loaded)
    shown = {key: (a.kind, a.value, a.relative_to) for key, a in found.items()}
    assert shown == {
        # The row whose tags are the deployment's own: dep16's options store the KV cache as NVFP4.
        "dep16:gsm8k": (anchors.REFERENCE, 96.5, None),
        # No reference file for mmmu and nothing declared: a score, no verdict.
        "dep16:mmmu": (anchors.NONE, None, None),
        "dep16:aa_lcr": (anchors.DECLARED, 0.62, None),
        "dep16_kv_fp8:gsm8k": (anchors.REFERENCE, 95.9, None),
        # No row fits, so it is judged against the deployment it names.
        "dep16_kv_fp8:mmmu": (anchors.RELATIVE, None, "dep16"),
        "pd:gsm8k": (anchors.REFERENCE, 96.5, None),
        "lone:mmmu": (anchors.NONE, None, None),
    }
    assert found["dep16:aa_lcr"].source.startswith("trtllm-eval aa_lcr at DEP16")
    assert "the row of `moonshotai/Kimi-K3` with {'kv_cache_quant_algo': 'NVFP4'}" in (
        found["dep16:gsm8k"].source
    )


def test_anchors_are_frozen_once_and_a_later_start_adds_only_what_is_absent(tmp_path):
    spec = _delivered(
        tmp_path,
        parallel="tp1",
        model_key="Qwen/Qwen3-0.6B",
        agent="claude-test-model",
        disagg=False,
    )
    _rows(tmp_path, "gsm8k", {"Qwen/Qwen3-0.6B": [{"accuracy": 41.77}]})
    loaded = _load(tmp_path, spec)
    workspace = tmp_path / "ws"
    first = anchors.freeze(workspace, loaded)
    assert {k: (a.kind, a.value) for k, a in first.items()} == {
        "tp1:gsm8k": (anchors.REFERENCE, 41.77),
        "kv_fp8@tp1:gsm8k": (anchors.RELATIVE, None),
    }
    # A commit that moves the row, and a spec that asks for more, change nothing frozen.
    _rows(tmp_path, "gsm8k", {"Qwen/Qwen3-0.6B": [{"accuracy": 12.0}]})
    spec["variants"]["chunked"] = {"runtime": {"chunked_prefill": True}, "gates": ["accuracy"]}
    again = anchors.freeze(workspace, _load(tmp_path, spec))
    assert again["tp1:gsm8k"].value == 41.77 and set(again) == {*first, "chunked@tp1:gsm8k"}
    assert anchors.frozen(workspace, "tp1", "gsm8k").value == 41.77
    assert anchors.frozen(None, "tp1", "gsm8k") is None
    assert anchors.differences(workspace, loaded) == [
        "tp1:gsm8k: frozen at 41.77; the tree's row now says 12"
    ]


def test_a_floor_is_two_standard_errors_and_a_relative_one_allows_for_both_scores():
    accuracy = {}
    gsm8k = Benchmark(name="gsm8k")
    reference = anchors.Anchor("tp1", "gsm8k", anchors.REFERENCE, 41.77)
    floor, source = anchors.floor(reference, accuracy, gsm8k)
    se = 100 * math.sqrt(0.4177 * (1 - 0.4177) / 1319)
    assert floor.value == pytest.approx(41.77 - 2 * se) and source == "the gsm8k split size"
    relative = anchors.Anchor("kv", "gsm8k", anchors.RELATIVE, relative_to="tp1")
    floor, _ = anchors.floor(relative, accuracy, gsm8k, relative_score=41.77)
    assert floor.value == pytest.approx(41.77 - 2 * math.sqrt(2) * se) and floor.relative
    assert anchors.floor(relative, accuracy, gsm8k)[0] is None  # its deployment has no score
    # The count that sets the error: what the run printed, then --num_samples, then the split.
    subset = Benchmark(name="gsm8k", args=("--num_samples", "200"))
    assert anchors.floor(reference, accuracy, subset)[0].samples == 200
    printed = "Partial scores after 150/150 responses"
    assert anchors.floor(reference, accuracy, subset, output=printed)[0].samples == 150
    # A benchmark that reports a fraction is held on its own scale.
    declared = anchors.Anchor("dep16", "aa_lcr", anchors.DECLARED, 0.62)
    floor, _ = anchors.floor(declared, {"samples": 100}, Benchmark(name="aa_lcr"))
    assert floor.value == pytest.approx(0.62 - 2 * math.sqrt(0.62 * 0.38 / 100))
    none = anchors.Anchor("lone", "mmmu", anchors.NONE)
    assert anchors.floor(none, accuracy, Benchmark(name="mmmu")) == (None, "no anchor")


def test_a_run_freezes_its_anchors_and_its_checks_read_the_frozen_copy(
    tmp_path, task_file, repo, monkeypatch
):
    workflow = _workflow(tmp_path, dry_run=True)
    with workflow:
        assert workflow.run(task_file)
    workspace = tmp_path / "ws"
    assert anchors.frozen(workspace, "tp1", "gsm8k").value == 85.6
    assert "anchors: 1 frozen" in (workspace / "journal.md").read_text()
    # The tree's row moves; the check is still held to the frozen 85.6.
    (repo / REFERENCES / "gsm8k.yaml").write_text("openai/gpt-oss-120b:\n  - accuracy: 1.0\n")
    held = {}

    def gate(environment, **kwargs):
        held.update(kwargs)
        return gate_cli.check_result.GateResult(name="accuracy[gsm8k]", passed=True, detail="ok")

    monkeypatch.setattr(gate_cli.accuracy_checks, "run_accuracy_gate", gate)
    common = dict(workdir=tmp_path / "w", runner=lambda *a, **k: CommandResult(0, ""))
    gate_cli.run_check(
        workflow.task_data, "accuracy[gsm8k]", "core", "tp1", workspace=workspace, **common
    )
    assert held["reference"] == 85.6 and held["deployment"].name == "tp1"
    gate_cli.run_check(workflow.task_data, "accuracy[gsm8k]", "core", "tp1", **common)
    assert held["reference"] == 1.0  # no workspace: the tree's row, as a bare command reads it


# ------------------------------------------------------------------ through the endpoint


def _pd() -> Deployment:
    return Deployment(
        name="pd",
        ctx=Server(options={"tensor_parallel_size": 1}),
        gen=Server(options={"tensor_parallel_size": 1}),
        benchmarks=(Benchmark(name="gsm8k", args=("--num_samples", "200")),),
    )


def _task(tmp_path) -> dict:
    return {"checkpoint_path": "/ckpt/model", "trtllm_repo_path": str(tmp_path)}


def test_a_boot_through_the_endpoint_needs_answers_with_their_keywords(site):
    workspace = site / "ws"
    spec = BootSpec(
        prompts=("In one sentence, photosynthesis is",),
        items=(BootPrompt(prompt="The capital of France is", keywords_any=("Paris",)),),
    )
    result = endpoint.run_endpoint_boot(
        _pd(), task_data=_task(site), workspace=workspace, spec=spec
    )
    assert result.passed, result.detail
    assert "disaggregated deployment `pd` started and its endpoint answered 2 prompt(s)" in (
        result.detail
    )
    assert servers.load_owned(workspace) == []  # one load: started, asked, stopped
    wrong = BootSpec(prompts=("x",), keywords=("Berlin",))
    result = endpoint.run_endpoint_boot(
        _pd(), task_data=_task(site), workspace=workspace, spec=wrong
    )
    assert not result.passed and "missing 1 expected keyword(s): ['Berlin']" in result.detail


def test_a_boot_through_the_endpoint_fails_when_nothing_starts_or_ranks_are_left(
    site,
    monkeypatch,
):
    workspace = site / "ws"
    monkeypatch.setenv("FAKE_SERVE_EXIT", "1")
    result = endpoint.run_endpoint_boot(_pd(), task_data=_task(site), workspace=workspace)
    assert not result.passed and "the deployment did not start: pd.ctx exited" in result.detail
    monkeypatch.delenv("FAKE_SERVE_EXIT")
    holders = site / "holders"

    def ask(running, prompt, **kwargs):
        holders.write_text("4242\n")  # a rank that outlives the stop
        return "Paris"

    monkeypatch.setattr(servers, "ask", ask)
    result = endpoint.run_endpoint_boot(_pd(), task_data=_task(site), workspace=workspace)
    assert not result.passed and result.infrastructure.startswith("ranks were left holding")


def test_accuracy_through_the_endpoint_runs_the_trees_evaluator_and_is_held_to_the_anchor(
    site,
):
    workspace = site / "ws"
    pd = _pd()
    environment = resolve_environment(
        {"environment": {"python": "/opt/venv/bin/python"}}, devices=(), world_size=1
    )
    seen: list[str] = []

    def runner(command, env=None, timeout=0):
        seen.append(command)
        return CommandResult(returncode=0, stdout="lm-eval gsm8k average accuracy: 40.50\n")

    anchor = anchors.Anchor("pd", "gsm8k", anchors.REFERENCE, 41.77)
    result = endpoint.run_endpoint_accuracy(
        pd, pd.benchmarks[0], environment, task_data=_task(site), workspace=workspace,
        anchor=anchor, runner=runner, member="tp1",
    )  # fmt: skip
    # 200 samples: SE 3.49, floor 34.80.
    assert (
        result.passed and result.samples == 200 and result.floor == pytest.approx(34.795, abs=0.01)
    )
    assert "through the endpoint" in result.detail and "anchor: reference" in result.detail
    [command] = seen
    assert re.search(
        r"endpoint_eval\.py --url http://127\.0\.0\.1:\d+ --model /ckpt/model", command
    )
    assert command.endswith("--benchmark gsm8k --num_samples 200")
    assert servers.load_owned(workspace) == []
    # Relative: against the other deployment's score, with the error of both.
    relative = anchors.Anchor("pd", "gsm8k", anchors.RELATIVE, relative_to="tp1")
    result = endpoint.run_endpoint_accuracy(
        pd, pd.benchmarks[0], environment, task_data=_task(site), workspace=workspace,
        anchor=relative, relative_score=52.0, runner=runner,
    )  # fmt: skip
    assert not result.passed and "of the difference" in result.floor_rule
    # A benchmark the tree has not driven through an endpoint starts nothing.
    mmmu = Benchmark(name="mmmu")
    result = endpoint.run_endpoint_accuracy(
        pd, mmmu, environment, task_data=_task(site), workspace=workspace,
        anchor=anchors.Anchor("pd", "mmmu", anchors.NONE), runner=runner,
    )  # fmt: skip
    assert not result.passed and result.detail.startswith("not judged through the hop")
    assert len(seen) == 2


def test_the_evaluators_request_states_its_sampling():
    class Sampling:
        max_tokens, stop, truncate_prompt_tokens = 256, ["Question:"], 4096

    body = endpoint_eval.request_body("/ckpt/model", "Q: 1+1?", Sampling())
    assert body == {
        "model": "/ckpt/model",
        "prompt": "Q: 1+1?",
        "max_tokens": 256,
        "temperature": 0.0,
        "top_k": 1,
        "n": 1,
        "stream": False,
        "stop": ["Question:"],
        "truncate_prompt_tokens": 4096,
    }
    assert endpoint.ENDPOINT_BENCHMARKS == ("gsm8k",)


def test_a_disaggregated_deployments_boot_goes_through_its_endpoint_under_the_lock(
    site,
    monkeypatch,
):
    spec = _load(
        site,
        _delivered(
            site,
            parallel="tp2",
            model_key="Qwen/Qwen3-0.6B",
            agent="claude-test-model",
            disagg=True,
        ),
    )
    assert deployments.resolve(spec)[1].kind == deployments.DISAGGREGATED
    locked = []
    real = servers.engine_lock

    def lock(workspace):
        locked.append(workspace)
        return real(workspace)

    monkeypatch.setattr(servers, "engine_lock", lock)
    result = gate_cli.run_check(
        spec, "boot", "disagg", "tp2", workdir=site / "w", workspace=site / "ws",
        runner=lambda *a, **k: CommandResult(0, ""),
    )  # fmt: skip
    assert result.passed, result.detail
    assert "disaggregated deployment `disagg@tp2` started" in result.detail
    assert locked == [site / "ws"]
    started = (site / "ws" / "deployments" / "disagg@tp2" / "address.ctx.started.json").read_text()
    assert '"--server_role", "context"' in started
    # Its accuracy names the benchmark; one the tree has not driven through an endpoint is
    # reported, with no load.
    result = gate_cli.run_check(
        spec, "accuracy[mmmu]", "disagg", "tp2", workdir=site / "w", workspace=site / "ws",
        runner=lambda *a, **k: CommandResult(0, ""),
    )  # fmt: skip
    assert not result.passed and result.detail.startswith("not judged through the hop")
