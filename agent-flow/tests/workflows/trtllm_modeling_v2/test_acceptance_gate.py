# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The acceptance gate's per-prompt comparisons, its workload, and the engine driver it runs."""

from __future__ import annotations

import json
import math
import shlex
import subprocess
import sys
import textwrap

import pytest
import yaml

from agent_flow.workflows.trtllm_modeling_v2.checks import gates
from agent_flow.workflows.trtllm_modeling_v2.checks.gate_records import KIND_ACCEPTANCE, GateRecords
from agent_flow.workflows.trtllm_modeling_v2.checks.gates import TreeState
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import workload_io
from agent_flow.workflows.trtllm_modeling_v2.infra.environment import resolve_environment
from agent_flow.workflows.trtllm_modeling_v2.infra.execution import CommandResult
from agent_flow.workflows.trtllm_modeling_v2.spec.task_schema import (
    TaskSchemaError,
    load_and_validate_task_yaml,
)

_TASK = {
    "checkpoint_path": "/ckpt/M",
    "parallel": "tep4",
    "trtllm_repo_path": "/repo",
    "features": ["dspark"],
    "accuracy": {"benchmark": "gsm8k", "reference_yaml": "r.yaml"},
}


def _env():
    return resolve_environment(
        {"environment": {"kind": "local", "python": "/py", "env": {}}}, devices=(0,), world_size=1
    )


def _driver_output(rates, *, prompts=None, iterations=100):
    """What the engine driver prints for prompts accepting at ``rates``."""
    lines = []
    for n, rate in enumerate(rates):
        tokens = int(round(rate * iterations))
        workload_io_line = {
            "index": n,
            "id": f"d{n}",
            "prompt_sha": (prompts or [f"p{i}" for i in range(len(rates))])[n],
            "tokens": tokens,
            "iterations": iterations,
            "token_ids": [],
            "text": "",
        }
        lines.append(f"{workload_io.RECORD_MARKER} {json.dumps(workload_io_line)}")
    return "\n".join(lines) + "\n"


def _legs(require, off=None):
    """A runner: the require leg prints ``require``; the off leg ``off`` or cannot load."""
    seen = []

    def run(command, env=None, timeout=0):
        seen.append(command)
        if "TRTLLM_MODELING_V2=off" in command:
            if off is None:
                return CommandResult(returncode=1, stderr="Unknown model architecture: X")
            return CommandResult(returncode=0, stdout=off)
        return CommandResult(returncode=0, stdout=require)

    run.seen = seen
    return run


def _task(**acceptance):
    return {**_TASK, "accuracy": {**_TASK["accuracy"], "acceptance": acceptance}}


_BASE = [2.6 + 0.5 * math.sin(i) for i in range(32)]


@pytest.fixture
def tree(monkeypatch):
    """Pin the tree a gate believes it measured, so a baseline can come from another."""
    current = {"name": "treeA"}

    def fake(task_data, *, allow_dirty=None):
        return TreeState(description=current["name"])

    monkeypatch.setattr(gates, "inspect_tree", fake)
    return current


# ------------------------------------------------------------- against stock


def test_a_drop_against_stock_that_the_paired_test_finds_fails():
    worse = [r - 0.4 for r in _BASE]
    result = gates.run_acceptance_gate(
        _env(), task_data=_task(), runner=_legs(_driver_output(worse), _driver_output(_BASE))
    )
    assert not result.passed
    assert "paired over 32 prompt(s)" in result.detail
    assert "subtly wrong" in result.detail


def test_a_ratio_under_the_floor_that_is_noise_passes():
    """Under 0.95 on the weighted mean, but the per-prompt test cannot tell it from noise."""
    base = [b + 1.0 for b in _BASE]
    noisy = [b + (1.4 if i % 2 == 0 else -1.8) for i, b in enumerate(base)]
    paired = gates.RegressionPolicy().compare(base, noisy)
    assert not paired.significant
    result = gates.run_acceptance_gate(
        _env(), task_data=_task(), runner=_legs(_driver_output(noisy), _driver_output(base))
    )
    weighted = sum(noisy) / sum(base)
    assert weighted < 0.95
    assert result.passed, result.detail


# ------------------------------------------------- against the previous tree


def test_the_first_verdict_becomes_the_baseline(tmp_path, tree):
    records = GateRecords(tmp_path / "r.jsonl")
    result = gates.run_acceptance_gate(
        _env(),
        task_data=_task(),
        runner=_legs(_driver_output(_BASE), _driver_output(_BASE)),
        member="tep4",
        records=records,
    )
    assert result.passed
    assert "this one is the baseline" in result.detail
    [record] = records.all(KIND_ACCEPTANCE)
    assert record.tree == "treeA" and len(record.values["per_prompt"]) == 32


def test_a_significant_drop_against_the_previous_tree_fails(tmp_path, tree):
    records = GateRecords(tmp_path / "r.jsonl")
    common = dict(task_data=_task(), member="tep4", records=records)
    gates.run_acceptance_gate(
        _env(), runner=_legs(_driver_output(_BASE), _driver_output(_BASE)), **common
    )
    tree["name"] = "treeB"
    dropped = [r - 0.3 for r in _BASE]
    result = gates.run_acceptance_gate(
        _env(), runner=_legs(_driver_output(dropped), _driver_output(dropped)), **common
    )
    assert not result.passed
    assert "against tree treeA: REGRESSED" in result.detail
    # A failed verdict is recorded, but is not the next tree's baseline.
    tree["name"] = "treeC"
    again = gates.run_acceptance_gate(
        _env(), runner=_legs(_driver_output(_BASE), _driver_output(_BASE)), **common
    )
    assert again.passed and "against tree treeA" in again.detail


def test_the_f051_change_passes_against_the_previous_tree(tmp_path, tree):
    """A change that only moves the generated text: per-prompt noise, no shift."""
    records = GateRecords(tmp_path / "r.jsonl")
    common = dict(task_data=_task(), member="tp4", records=records)
    gates.run_acceptance_gate(
        _env(), runner=_legs(_driver_output(_BASE), _driver_output(_BASE)), **common
    )
    tree["name"] = "post-fix"
    moved = [b - 0.06 + 0.25 * math.cos(3 * i) for i, b in enumerate(_BASE)]
    result = gates.run_acceptance_gate(
        _env(), runner=_legs(_driver_output(moved), _driver_output(moved)), **common
    )
    assert result.passed, result.detail
    assert "against tree treeA" in result.detail and "REGRESSED" not in result.detail


def test_a_rerun_of_the_same_tree_is_not_its_own_baseline(tmp_path, tree):
    records = GateRecords(tmp_path / "r.jsonl")
    common = dict(task_data=_task(), member="tep4", records=records)
    for _ in range(2):
        result = gates.run_acceptance_gate(
            _env(), runner=_legs(_driver_output(_BASE), _driver_output(_BASE)), **common
        )
    assert "this one is the baseline" in result.detail


def test_other_members_and_workloads_are_not_compared(tmp_path, tree):
    records = GateRecords(tmp_path / "r.jsonl")
    gates.run_acceptance_gate(
        _env(),
        task_data=_task(),
        runner=_legs(_driver_output(_BASE), _driver_output(_BASE)),
        member="tep4",
        records=records,
    )
    tree["name"] = "treeB"
    dropped = [r - 0.5 for r in _BASE]
    for task, member in ((_task(), "dep4"), (_task(num_prompts=40), "tep4")):
        result = gates.run_acceptance_gate(
            _env(),
            task_data=task,
            runner=_legs(_driver_output(dropped), _driver_output(dropped)),
            member=member,
            records=records,
        )
        assert "this one is the baseline" in result.detail


def test_the_regression_policy_is_read_from_the_task():
    policy = gates.resolve_regression_policy(
        {"acceptance": {"regression": {"test": "wilcoxon", "alpha": 0.01, "min_effect": 0.2}}}
    )
    assert (policy.test, policy.alpha, policy.min_effect) == ("wilcoxon", 0.01, 0.2)
    assert gates.resolve_regression_policy({}).min_effect == 0.1


# ----------------------------------------------------------------- the floor


def test_a_floor_without_a_measurement_warns():
    """A floor picked from the ceiling passes a wrong drafter, so an unsourced floor warns."""
    task = _task(dspark={"min_absolute": 2.0})
    result = gates.run_acceptance_gate(
        _env(), task_data=task, runner=_legs(_driver_output(_BASE)), unit="dspark"
    )
    assert result.passed
    assert result.warnings and "not tied to a measured reference" in result.warnings[0]
    assert "WARNING: the acceptance floor 2.00" in result.summary()


def test_a_floor_with_its_source_does_not_warn():
    task = _task(dspark={"min_absolute": 2.0, "floor_source": "05-units/x.md: reference 3.32"})
    result = gates.run_acceptance_gate(
        _env(), task_data=task, runner=_legs(_driver_output(_BASE)), unit="dspark"
    )
    assert result.passed and not result.warnings
    assert "from 05-units/x.md" in result.detail


def test_a_measured_reference_sets_the_floor():
    task = _task(dspark={"measured_reference": 3.32})
    floor = gates.resolve_acceptance_floor_spec(task["accuracy"], "dspark")
    assert floor.value == pytest.approx(3.32 * 0.95) and floor.warning() is None
    result = gates.run_acceptance_gate(
        _env(), task_data=task, runner=_legs(_driver_output(_BASE)), unit="dspark"
    )
    assert not result.passed  # 2.6-ish against 3.15
    assert result.reference == pytest.approx(3.154)


# -------------------------------------------------------------- the wiring


# ---------------------------------------------------------------- the schema


def _spec(tmp_path, acceptance):
    repo = tmp_path / "repo"
    repo.mkdir(exist_ok=True)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir(exist_ok=True)
    data = {
        "checkpoint_path": str(ckpt),
        "trtllm_repo_path": str(repo),
        "arch": "sm_100",
        "parallel": "tp1",
        "environment": {"kind": "local", "python": "/usr/bin/python3"},
        "accuracy": {
            "benchmark": "gsm8k",
            "reference_yaml": "r.yaml",
            "model_key": "M",
            "acceptance": acceptance,
        },
    }
    path = tmp_path / "task.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_the_schema_accepts_the_new_workload_and_comparison_keys(tmp_path):
    (tmp_path / "qs.jsonl").write_text('{"question": "q"}\n', encoding="utf-8")
    acceptance = {
        "dataset": "qs.jsonl",
        "num_prompts": 40,
        "text_field": "question",
        "prompt_template": "<User>{text}<Assistant>",
        "min_prompts": 32,
        "regression": {"test": "wilcoxon", "alpha": 0.05, "min_effect": 0.1},
        "dspark": {"measured_reference": 3.32, "floor_source": "05-units/x.md"},
    }
    loaded = load_and_validate_task_yaml(_spec(tmp_path, acceptance))
    assert loaded["accuracy"]["acceptance"]["dataset"] == str(tmp_path / "qs.jsonl")


def test_a_dataset_id_is_left_alone(tmp_path):
    loaded = load_and_validate_task_yaml(_spec(tmp_path, {"dataset": "openai/gsm8k"}))
    assert loaded["accuracy"]["acceptance"]["dataset"] == "openai/gsm8k"


@pytest.mark.parametrize(
    "acceptance,message",
    [
        ({"regression": {"test": "z"}}, "regression.test"),
        ({"regression": {"alpha": 1.5}}, "regression.alpha"),
        ({"regression": {"tails": 2}}, "tails"),
        ({"prompt_template": "no placeholder"}, "prompt_template"),
        ({"num_prompts": 0}, "num_prompts"),
        ({"dataset": "x", "prompts": ["a"]}, "single source"),
        ({"dspark": {"measured_reference": 0.5}}, "measured_reference"),
        ({"dspark": {"floor_source": ""}}, "floor_source"),
    ],
)
def test_the_schema_refuses_a_malformed_acceptance_block(tmp_path, acceptance, message):
    with pytest.raises(TaskSchemaError, match=message):
        load_and_validate_task_yaml(_spec(tmp_path, acceptance))


# ---------------------------------------------------------- the dataset reader


def test_rows_come_from_json_lines_json_yaml_and_a_directory(tmp_path):
    (tmp_path / "a.jsonl").write_text('{"question": "q1", "answer": "#### 3"}\n\n', "utf-8")
    (tmp_path / "b.json").write_text(json.dumps({"data": [{"prompt": "q2"}]}), "utf-8")
    (tmp_path / "c.yaml").write_text(yaml.safe_dump(["q3"]), "utf-8")
    assert workload_io.load_rows(tmp_path / "a.jsonl")[0]["question"] == "q1"
    assert workload_io.load_rows(tmp_path / "b.json") == [{"prompt": "q2"}]
    assert workload_io.load_rows(tmp_path / "c.yaml") == ["q3"]
    assert len(workload_io.load_rows(tmp_path, split=None)) == 3


def test_a_directory_prefers_the_split_named_in_its_files(tmp_path):
    (tmp_path / "train.jsonl").write_text('{"question": "train"}\n', "utf-8")
    (tmp_path / "test.jsonl").write_text('{"question": "test"}\n', "utf-8")
    assert workload_io.load_rows(tmp_path, split="test") == [{"question": "test"}]


def test_prompts_are_rendered_through_the_template():
    rows = [{"question": "Q0", "answer": "#### 1"}, {"other": "x"}, {"question": "Q2"}, "Q3"]
    chosen = workload_io.select_prompts(rows, num=2, template="<U>{text} step by step<A>")
    assert [c["prompt"] for c in chosen] == ["<U>Q0 step by step<A>", "<U>Q2 step by step<A>"]
    assert chosen[0]["answer"] == "#### 1"
    assert workload_io.select_prompts(rows, num=5, skip=3)[0]["prompt"] == "Q3"


def test_a_dataset_id_is_never_downloaded(monkeypatch):
    """With no local path, an id needs `datasets`, read offline; this host has neither."""
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(workload_io.DatasetError, match="datasets"):
        workload_io.load_rows("openai/gsm8k")
    with pytest.raises(workload_io.DatasetError, match="does not exist"):
        workload_io.load_rows("/nowhere/gsm8k")


def test_the_gsm8k_scorer_compares_the_last_number():
    assert workload_io.score("gsm8k", "so the total is 1,234.", "steps #### 1234")
    assert not workload_io.score("gsm8k", "it is 12", "#### 13")
    assert workload_io.score("contains", "The Merlion statue", "merlion")
    assert workload_io.score("gsm8k", "x", None) is None


def test_records_are_found_behind_any_prefix_and_a_torn_line_is_skipped():
    out = (
        f'0: [TRT] {workload_io.RECORD_MARKER} {{"index": 1, "text": "b"}}\n'
        f'CPUs 72-143 {workload_io.RECORD_MARKER} {{"index": 0, "text": "a"}}\n'
        f'{workload_io.RECORD_MARKER} {{"index": 2, "te\n'
    )
    assert [r["text"] for r in gates.parse_driver_records(out)] == ["a", "b"]


# --------------------------------------------------------- the engine driver

_FAKE_QUICKSTART = textwrap.dedent(
    """
    import argparse
    from types import SimpleNamespace


    class Params:
        def __init__(self, max_tokens):
            self.max_tokens = max_tokens
            self.ignore_eos = False


    class FakeLLM:
        tokenizer = SimpleNamespace(vocab_size=1000)

        def __init__(self, accept):
            self.accept = accept
            self.calls = []

        def _ids(self, inp, params):
            text = inp if isinstance(inp, str) else str(inp)
            return [(len(text) + k) % 97 for k in range(params.max_tokens)]

        def generate_async(self, inp, params, streaming=False):
            ids = self._ids(inp, params)
            prompt = inp if isinstance(inp, str) else inp.get("prompt_token_ids", [])
            ends = [1]
            while ends[-1] < len(ids):
                ends.append(min(ends[-1] + self.accept, len(ids)))
            for end in ends:
                got = ids[:end]
                yield SimpleNamespace(
                    outputs=[SimpleNamespace(token_ids=got, text=" ".join(map(str, got)),
                                             finish_reason="length")],
                    prompt_token_ids=list(range(len(prompt))),
                )

        def generate(self, inputs, params):
            return [list(self.generate_async(i, params))[-1] for i in inputs]

        def shutdown(self):
            print("FAKE shutdown")


    def parse_arguments():
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_dir")
        parser.add_argument("--max_tokens", type=int, default=8)
        parser.add_argument("--accept", type=int, default=1)
        return parser.parse_known_args()[0]


    def setup_llm(args):
        return FakeLLM(args.accept), Params(args.max_tokens)
    """
)


@pytest.fixture
def fake_tree(tmp_path):
    """A tree with a fake quickstart, and a fake `tensorrt_llm` on the path."""
    quickstart = tmp_path / "repo" / "examples" / "llm-api" / "quickstart_advanced.py"
    quickstart.parent.mkdir(parents=True)
    quickstart.write_text(_FAKE_QUICKSTART, encoding="utf-8")
    package = tmp_path / "site" / "tensorrt_llm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    return tmp_path


def _drive(fake_tree, spec):
    path = fake_tree / "spec.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(gates.ENGINE_DRIVER), "--spec", str(path)],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(fake_tree / "site"), "PATH": "/usr/bin:/bin"},
        timeout=120,
        check=False,
    )


def test_the_driver_runs_declared_prompts_and_a_local_dataset(fake_tree):
    data = fake_tree / "gsm.jsonl"
    data.write_text(
        "".join(json.dumps({"question": f"Q{i}", "answer": "#### 1"}) + "\n" for i in range(5)),
        "utf-8",
    )
    task = {"trtllm_repo_path": str(fake_tree / "repo"), "checkpoint_path": "/ckpt"}
    spec = json.loads(
        gates.write_driver_spec(
            fake_tree / "s.json",
            task_data=task,
            argv=["--model_dir=/ckpt", "--max_tokens=9", "--accept=3"],
            requests=[{"id": "p0", "prompt": "hello", "max_tokens": 6}],
            dataset={"source": str(data), "num": 3, "template": "<{text}>", "max_tokens": 9},
        ).read_text()
    )
    done = _drive(fake_tree, spec)
    assert done.returncode == 0, done.stderr
    records = gates.parse_driver_records(done.stdout)
    assert [r["id"] for r in records] == ["p0", "d0", "d1", "d2"]
    assert records[0]["tokens"] == 6 and records[0]["iterations"] == 3
    assert records[1]["tokens"] == 9 and records[1]["iterations"] == 4
    assert records[1]["prompt_sha"] == workload_io.prompt_digest("<Q0>")
    assert "FAKE shutdown" in done.stdout
    driver = workload_io.parse_marked(done.stdout, workload_io.DRIVER_MARKER)
    assert driver[0]["tensorrt_llm"].startswith(str(fake_tree / "site"))
    reading = gates.parse_acceptance(done.stdout)
    assert reading.value == pytest.approx((6 + 27) / (3 + 12)) and len(reading.prompts) == 4


def test_the_driver_runs_concurrent_windows_in_order(fake_tree):
    spec = {
        "quickstart": str(fake_tree / "repo/examples/llm-api/quickstart_advanced.py"),
        "quickstart_argv": ["--model_dir=/ckpt"],
        "requests": [{"prompt": f"p{i}", "max_tokens": 4} for i in range(5)],
        "concurrency": 2,
    }
    done = _drive(fake_tree, spec)
    assert done.returncode == 0, done.stderr
    assert [r["index"] for r in gates.parse_driver_records(done.stdout)] == [0, 1, 2, 3, 4]


def test_the_driver_measures_prefill_and_decode(fake_tree):
    spec = {
        "quickstart": str(fake_tree / "repo/examples/llm-api/quickstart_advanced.py"),
        "quickstart_argv": ["--model_dir=/ckpt", "--accept=2"],
        "perf": {"prefill_tokens": 16, "decode_tokens": 8, "repeats": 3, "warmup": 1},
    }
    done = _drive(fake_tree, spec)
    assert done.returncode == 0, done.stderr
    [perf] = workload_io.parse_marked(done.stdout, workload_io.PERF_MARKER)
    assert len(perf["prefill_samples"]) == 3 and perf["prefill_ms"] >= 0
    assert perf["decode_step_ms"] is not None and perf["tokens_per_step"] == pytest.approx(2.0)


def test_a_failing_driver_exits_non_zero_with_the_reason(fake_tree):
    spec = {
        "quickstart": str(fake_tree / "repo/examples/llm-api/quickstart_advanced.py"),
        "quickstart_argv": ["--model_dir=/ckpt"],
        "requests": [{"prompt": "x", "images": [str(fake_tree / "absent.png")]}],
    }
    done = _drive(fake_tree, spec)
    assert done.returncode != 0 and "image not found" in done.stderr


def test_an_lfs_pointer_image_is_named_as_one(fake_tree):
    pointer = fake_tree / "merlion.png"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:x\n", "utf-8")
    spec = {
        "quickstart": str(fake_tree / "repo/examples/llm-api/quickstart_advanced.py"),
        "quickstart_argv": ["--model_dir=/ckpt"],
        "requests": [{"prompt": "x", "images": [str(pointer)]}],
    }
    done = _drive(fake_tree, spec)
    assert done.returncode != 0 and "git-lfs pointer" in done.stderr


def test_the_gate_command_launches_the_driver_as_engine_work(tmp_path):
    command = gates.engine_driver_command(_env(), tmp_path / "s.json")
    words = shlex.split(command)
    assert words[words.index(str(gates.ENGINE_DRIVER)) - 1] == "/py"
    assert words[-1] == str(tmp_path / "s.json")
