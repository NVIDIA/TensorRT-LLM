# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Frozen anchors per deployment and benchmark, and the checks that go through an endpoint."""

from __future__ import annotations

import json
import math
import re
import sys
import types

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import (
    accuracy,
    anchors,
    endpoint,
    gate_cli,
    servers,
)
from agent_flow.workflows.trtllm_modeling_v2.checks.boot import (
    BootPrompt,
    BootSpec,
    resolve_boot_spec,
)
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import endpoint_eval
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.commands import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.infra.compute.environment import resolve_environment
from agent_flow.workflows.trtllm_modeling_v2.spec import advice, deployments
from agent_flow.workflows.trtllm_modeling_v2.spec.deployments import Benchmark, Deployment, Server
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import TaskSchemaError

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
    # Of a context and a generation server that differ, the generation side's options name
    # the row: it produces what is scored.
    sides = Deployment(
        name="pd",
        ctx=Server(options={"kv_cache_config": {"dtype": "auto"}}),
        gen=Server(options={"kv_cache_config": {"dtype": "fp8"}}),
    )
    assert anchors.tags(sides, Benchmark(name="gsm8k")) == {"kv_cache_quant_algo": "FP8"}
    assert anchors.resolve(loaded, sides, Benchmark(name="gsm8k")).value == 95.9
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
        "kv_fp8:gsm8k": (anchors.RELATIVE, None),
    }
    # A commit that moves the row, and a spec that asks for more, change nothing frozen.
    _rows(tmp_path, "gsm8k", {"Qwen/Qwen3-0.6B": [{"accuracy": 12.0}]})
    spec["variants"]["chunked"] = {"runtime": {"chunked_prefill": True}, "gates": ["accuracy"]}
    again = anchors.freeze(workspace, _load(tmp_path, spec))
    assert again["tp1:gsm8k"].value == 41.77 and set(again) == {*first, "chunked:gsm8k"}
    assert anchors.frozen(workspace, "tp1", "gsm8k").value == 41.77
    assert anchors.frozen(None, "tp1", "gsm8k") is None
    assert anchors.differences(workspace, loaded) == [
        "tp1:gsm8k: frozen at 41.77; the tree's row now says 12"
    ]
    # Frozen with each: the scale its scores are on and the count its floor is computed over.
    assert (again["tp1:gsm8k"].unit, again["tp1:gsm8k"].samples) == ("percent", 1319)


def test_frozen_means_frozen_a_none_stays_none_and_an_unreadable_file_is_not_read_as_none(
    tmp_path,
):
    spec = _delivered(
        tmp_path, parallel="tp1", model_key="Qwen/Qwen3-0.6B", agent="claude-test-model",
        disagg=False,
    )  # fmt: skip
    loaded = _load(tmp_path, spec)
    workspace = tmp_path / "ws"
    assert anchors.freeze(workspace, loaded)["tp1:gsm8k"].kind == anchors.NONE
    # A row written later does not turn a score with no verdict into a judged one.
    _rows(tmp_path, "gsm8k", {"Qwen/Qwen3-0.6B": [{"accuracy": 41.77}]})
    assert anchors.freeze(workspace, loaded)["tp1:gsm8k"].kind == anchors.NONE
    assert anchors.frozen(workspace, "tp1", "gsm8k").kind == anchors.NONE
    # A file that does not hold the anchors is not read as "nothing frozen yet".
    for torn in ("- [not, an, anchor\n", "", "tp1: 41.77\n"):
        (workspace / anchors.ANCHORS_FILENAME).write_text(torn)
        with pytest.raises(anchors.AnchorError, match="delete it to freeze them again"):
            anchors.freeze(workspace, loaded)


def test_a_run_whose_frozen_anchors_cannot_be_read_stops_at_its_start(tmp_path, task_file):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / anchors.ANCHORS_FILENAME).write_text("- [not, an, anchor\n")
    workflow = _workflow(tmp_path, dry_run=True)
    with workflow:
        assert not workflow.run(task_file)
    journal = (workspace / "journal.md").read_text()
    assert "anchors: the frozen file cannot be read" in journal
    assert "delete it to freeze them again" in journal


def test_a_benchmark_states_the_unit_of_its_scores(tmp_path):
    spec = _native(tmp_path)
    lcr = next(b for b in spec["deployments"][0]["benchmarks"] if b["name"] == "aa_lcr")
    assert "unit" not in lcr
    # 0.62 with the default unit reads as 0.62 percent: said at load, not guessed around.
    loaded = _load(tmp_path, spec)
    assert any(
        w.startswith("anchor under one percent: `dep16` on `aa_lcr` declares 0.62")
        for w in advice.spec_warnings(loaded)
    )
    lcr["unit"] = "fraction"
    loaded = _load(tmp_path, spec)
    assert deployments.resolve(loaded)[0].benchmark("aa_lcr").unit == "fraction"
    assert not [w for w in advice.spec_warnings(loaded) if w.startswith("anchor under one")]
    assert anchors.freeze(tmp_path / "ws", loaded)["dep16:aa_lcr"].unit == "fraction"
    lcr["unit"] = "points"
    with pytest.raises(TaskSchemaError, match="unit"):
        _load(tmp_path, spec)


def test_a_floor_is_two_standard_errors_and_a_relative_one_allows_for_both_scores():
    gsm8k = Benchmark(name="gsm8k")
    reference = anchors.Anchor("tp1", "gsm8k", anchors.REFERENCE, 41.77)
    floor, source = anchors.floor(reference, gsm8k)
    se = 100 * math.sqrt(0.4177 * (1 - 0.4177) / 1319)
    assert floor.value == pytest.approx(41.77 - 2 * se) and source == "the gsm8k split size"
    relative = anchors.Anchor("kv", "gsm8k", anchors.RELATIVE, relative_to="tp1")
    floor, _ = anchors.floor(relative, gsm8k, relative_score=41.77)
    assert floor.value == pytest.approx(41.77 - 2 * math.sqrt(2) * se) and floor.relative
    assert anchors.floor(relative, gsm8k)[0] is None  # its deployment has no score
    # The count that sets the error: what the run printed, then --num_samples, then the split.
    subset = Benchmark(name="gsm8k", args=("--num_samples", "200"))
    assert anchors.floor(reference, subset)[0].samples == 200
    printed = "Partial scores after 150/150 responses"
    assert anchors.floor(reference, subset, output=printed)[0].samples == 150
    # The count frozen with the anchor is the floor's, whatever a later run prints.
    fixed = anchors.Anchor("tp1", "gsm8k", anchors.REFERENCE, 41.77, samples=1319)
    floor, source = anchors.floor(fixed, subset, output=printed)
    assert (floor.samples, source) == (1319, "the count fixed with the anchor")
    # A benchmark that states it reports a fraction is held on that scale; nothing is
    # guessed from a score's size: 0.62 percent is a floor near zero.
    declared = anchors.Anchor("dep16", "aa_lcr", anchors.DECLARED, 0.62, unit="fraction")
    floor, _ = anchors.floor(declared, Benchmark(name="aa_lcr", samples=100))
    assert floor.value == pytest.approx(0.62 - 2 * math.sqrt(0.62 * 0.38 / 100))
    # A score that is no fraction is not held to a fraction's floor, which it would clear.
    scored = accuracy.judged(
        62.0, declared, Benchmark(name="aa_lcr", samples=100), relative_score=None, output=""
    )
    assert not scored["passed"] and "which is no fraction" in scored["detail"]
    assert accuracy.judged(
        0.61, declared, Benchmark(name="aa_lcr", samples=100), relative_score=None, output=""
    )["passed"]
    percent = anchors.Anchor("dep16", "aa_lcr", anchors.DECLARED, 0.62)
    floor, _ = anchors.floor(percent, Benchmark(name="aa_lcr", samples=100))
    assert floor.value == pytest.approx(0.62 - 2 * 100 * math.sqrt(0.0062 * 0.9938 / 100))
    none = anchors.Anchor("lone", "mmmu", anchors.NONE)
    assert anchors.floor(none, Benchmark(name="mmmu")) == (None, "no anchor")
    # A benchmark whose size nobody states has no error to hold it to.
    uncounted = anchors.Anchor("dep16", "aa_lcr", anchors.DECLARED, 0.62)
    assert anchors.floor(uncounted, Benchmark(name="aa_lcr")) == (
        None,
        "no sample count for `aa_lcr`: state its `samples`",
    )


def test_a_run_freezes_its_anchors_and_its_checks_read_the_frozen_copy(
    tmp_path, task_file, repo, monkeypatch
):
    workflow = _workflow(tmp_path, dry_run=True)
    with workflow:
        assert workflow.run(task_file)
    workspace = tmp_path / "ws"
    assert anchors.frozen(workspace, "tp1", "gsm8k").value == 85.6
    assert "gsm8k (reference 85.6: " in (workspace / "journal.md").read_text()
    # The tree's row moves; the check is still held to the frozen 85.6.
    (repo / REFERENCES / "gsm8k.yaml").write_text("openai/gpt-oss-120b:\n  - accuracy: 1.0\n")
    held = {}

    def gate(environment, **kwargs):
        held.update(kwargs)
        return gate_cli.check_result.GateResult(name="accuracy[gsm8k]", passed=True, detail="ok")

    monkeypatch.setattr(gate_cli.accuracy_checks, "run_accuracy_gate", gate)
    common = dict(workdir=tmp_path / "w", runner=lambda *a, **k: CommandResult(0, ""))
    gate_cli.run_check(workflow.task_data, "accuracy[gsm8k]", workspace=workspace, **common)
    assert held["anchor"].value == 85.6 and held["deployment"].name == "tp1"
    gate_cli.run_check(workflow.task_data, "accuracy[gsm8k]", "core", **common)
    assert held["anchor"].value == 1.0  # no workspace: the tree's row, as a bare command reads it


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
    # Every rank of both servers said what it ran, from the tree being measured.
    assert [(r["role"], r["status"], r["ranks"]) for r in result.engine] == [
        ("ctx", "match", 1),
        ("gen", "match", 1),
    ]
    wrong = BootSpec(prompts=("x",), keywords=("Berlin",))
    result = endpoint.run_endpoint_boot(
        _pd(), task_data=_task(site), workspace=workspace, spec=wrong
    )
    assert not result.passed and "missing 1 expected keyword(s): ['Berlin']" in result.detail


def test_a_boot_through_the_endpoint_holds_every_server_to_its_options(site, monkeypatch):
    workspace = site / "ws"
    reuse = {"kv_cache_config": {"dtype": "fp8", "enable_block_reuse": True}}
    quantized = Deployment(name="q", ctx=Server(options=reuse), gen=Server(options=reuse))

    def boot():
        return endpoint.run_endpoint_boot(quantized, task_data=_task(site), workspace=workspace)

    passed = boot()
    assert passed.passed
    # The proxy's usage is the context server's: the line says whose cache the repeat showed.
    assert "a repeated prompt resumed 448 cached tokens on the context server" in passed.detail
    assert "the generation server's reuse is not shown by the proxy" in passed.detail
    # Each server was started in the tree, with the switch on and the hook loaded.
    started = json.loads((workspace / "deployments/q/address.ctx.started.json").read_text())
    assert started["env"]["PYTHONPATH"].split(":")[1] == str(site.resolve())
    assert started["env"]["MV2FLOW_RANK_FACTS"] == "1"
    written = yaml.safe_load((workspace / "deployments/q/q.ctx.yaml").read_text())
    assert written == {**reuse, "modeling_v2": "require"}
    monkeypatch.setenv("FAKE_REUSE", "0")
    cold = boot()
    assert not cold.passed and "the engine reused nothing" in cold.detail
    monkeypatch.delenv("FAKE_REUSE")
    monkeypatch.setenv("FAKE_DTYPE", "auto")
    other = boot()
    assert not other.passed
    assert "the ctx server did not run what its options declare" in other.detail
    assert "kv_cache_config.dtype is 'auto', declared 'fp8'" in other.detail
    monkeypatch.delenv("FAKE_DTYPE")
    monkeypatch.setenv("FAKE_RANKS", "0")
    silent = boot()
    assert not silent.passed and "what the ctx server ran is unknown" in silent.detail


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

    real = servers.ask
    monkeypatch.setattr(servers, "ask", ask)
    result = endpoint.run_endpoint_boot(_pd(), task_data=_task(site), workspace=workspace)
    assert not result.passed and result.infrastructure.startswith("ranks were left holding")
    # They are still the run's until the devices are seen free: the next check starts nothing.
    again = endpoint.run_endpoint_boot(_pd(), task_data=_task(site), workspace=workspace)
    assert not again.passed and again.infrastructure.startswith("ranks an earlier check left")
    holders.write_text("")
    monkeypatch.setattr(servers, "ask", real)
    # Devices that could not be looked at: the verdict stands and its line says so.
    monkeypatch.setattr(servers, "_holders", lambda *a, **k: None)
    unseen = endpoint.run_endpoint_boot(_pd(), task_data=_task(site), workspace=workspace)
    assert unseen.passed and unseen.detail.endswith("; devices not verified free")


def test_an_answer_counts_once_it_completed_and_an_image_deployment_is_sent_an_image(
    site, monkeypatch
):
    workspace = site / "ws"
    picture = site / "tests/integration/test_input_files/merlion.png"
    picture.parent.mkdir(parents=True)
    picture.write_bytes(b"png")
    sees = Deployment(
        name="vl",
        ctx=Server(options={"a": 1}),
        gen=Server(options={"a": 1}),
        inputs=("text", "image"),
    )
    spec = resolve_boot_spec({}, sees)

    def boot():
        return endpoint.run_endpoint_boot(
            sees, task_data=_task(site), workspace=workspace, spec=spec
        )

    result = boot()
    assert result.passed, result.detail
    assert "answered 3 prompt(s) with their keywords, 1 of them with an image" in result.detail
    # A server that takes no image has not shown it serves one: said, and no pass.
    monkeypatch.setenv("FAKE_NO_IMAGES", "1")
    blind = boot()
    assert not blind.passed and blind.detail.startswith("image not checked: vl did not answer")
    # The server's own word on what is wrong is kept: the first 300 characters of the body.
    assert 'HTTP Error 400: Bad Request: {"error": "this model takes no image; ' in blind.detail
    assert "END" not in blind.detail
    monkeypatch.delenv("FAKE_NO_IMAGES")
    # An answer cut off before it finished is not an answer.
    monkeypatch.setenv("FAKE_UNFINISHED", "1")
    cut = boot()
    assert not cut.passed and "without a completed choice" in cut.detail


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
        anchor=anchor, runner=runner,
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
    # Offline, with the dataset the host holds, then the benchmark's own options.
    assert "--benchmark gsm8k --dataset_path " in command and command.endswith("--num_samples 200")
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
    # No stop is said as no stop: a server's own default is not the evaluator's.
    Sampling.stop = None
    assert endpoint_eval.request_body("/ckpt/model", "Q", Sampling())["stop"] == []


def test_the_evaluator_is_told_its_stop_strings_and_its_sample_count(monkeypatch, capsys):
    built: dict = {}

    class GSM8K:
        def __init__(self, **kwargs):
            built.update(kwargs)

        task_name = "gsm8k"

        def evaluate(self, llm, sampling):
            built["stop"] = sampling.stop
            return built.get("score", 41.7)

    class SamplingParams:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    package = types.ModuleType("tensorrt_llm")
    modules = {
        "tensorrt_llm": package,
        "tensorrt_llm.evaluate": types.SimpleNamespace(GSM8K=GSM8K),
        "tensorrt_llm.llmapi": types.SimpleNamespace(SamplingParams=SamplingParams),
        "tensorrt_llm.llmapi.tokenizer": types.SimpleNamespace(load_hf_tokenizer=lambda m: None),
    }
    package.evaluate = modules["tensorrt_llm.evaluate"]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    common = ["--url", "http://h:1", "--model", "/ckpt", "--benchmark", "gsm8k"]
    assert endpoint_eval.main([*common, "--stop", '["Question:"]']) == 0
    # The stop strings go into the task itself, which is what the evaluator generates by;
    # the count is the split's when the command names none, never left to a default.
    assert built["stop_strings"] == ["Question:"] and built["stop"] == ["Question:"]
    assert built["num_samples"] == 1319
    said = capsys.readouterr().out
    assert "endpoint-eval: gsm8k on 1319 samples, greedy, stop ['Question:']" in said
    # The score the evaluator returns is printed, and the check's own parser reads it.
    assert said.splitlines()[-1] == "lm-eval gsm8k average accuracy: 41.70"
    assert accuracy.parse_eval_score(said) == 41.7
    built.clear()
    assert endpoint_eval.main([*common, "--num_samples", "200"]) == 0
    assert built["num_samples"] == 200 and "stop_strings" not in built
    assert "on 200 samples, greedy, stop the task own" in capsys.readouterr().out
    # An evaluation that returns nothing is a failed command, never a silent "no score".
    GSM8K.evaluate = lambda self, llm, sampling: None
    assert endpoint_eval.main(common) == 1
    assert "returned no score" in capsys.readouterr().err


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
    common = dict(
        workdir=site / "w", workspace=site / "ws", runner=lambda *a, **k: CommandResult(0, "")
    )
    result = gate_cli.run_check(spec, "boot", "disagg", **common)
    assert result.passed, result.detail
    assert "disaggregated deployment `disagg` started" in result.detail
    # The verdict says which path judged.
    assert result.path == gate_cli.THROUGH_ENDPOINT and "[through the endpoint]" in result.summary()
    assert locked == [site / "ws"]
    started = (site / "ws" / "deployments" / "disagg" / "address.ctx.started.json").read_text()
    assert '"--server_role", "context"' in started
    # A benchmark the deployment is not scored on is a usage error, with no load.
    with pytest.raises(TaskSchemaError, match="`disagg` is not scored on `mmmu`"):
        gate_cli.run_check(spec, "accuracy[mmmu]", "disagg", **common)
