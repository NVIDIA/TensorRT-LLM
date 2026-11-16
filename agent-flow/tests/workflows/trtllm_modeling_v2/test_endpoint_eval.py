# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The evaluator a session scores a benchmark with, through its endpoint (``harness/endpoint_eval.py``)."""

from __future__ import annotations

import sys
import types

from agent_flow.workflows.trtllm_modeling_v2.checks import accuracy, session
from agent_flow.workflows.trtllm_modeling_v2.checks.harness import endpoint_eval
from agent_flow.workflows.trtllm_modeling_v2.spec import deployments


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
    assert session.ENDPOINT_BENCHMARKS == ("gsm8k",)
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
    # Asked for its data alone, with no endpoint: the evaluator is built, which loads the
    # data, and nothing is scored.
    built.clear()
    asked = ["--benchmark", "gsm8k", "--dataset_only", "--dataset_path", "/data/gsm8k"]
    assert endpoint_eval.main(asked) == 0
    assert built["dataset_path"] == "/data/gsm8k" and "stop" not in built

    # A loader that cannot read is answered in its own words, with where it looked.
    def offline(self, **kwargs):
        raise ConnectionError("Couldn't reach 'openai/gsm8k' on the Hub (OfflineModeIsEnabled)")

    GSM8K.__init__ = offline
    assert endpoint_eval.main(asked) == endpoint_eval.NO_DATA
    said = capsys.readouterr().err.strip().splitlines()[-1]
    assert "(ConnectionError: Couldn't reach 'openai/gsm8k' on the Hub" in said
    assert "It reads `--dataset_path` /data/gsm8k: put the dataset there" in said


def test_what_the_evaluator_cannot_carry_is_known_before_any_load():
    carried = endpoint_eval.not_carried
    assert carried("gsm8k", []) == "" and carried("gsm8k", ["--num_samples", "200"]) == ""
    # An argument of `trtllm-eval`'s own, a value it cannot read, another benchmark.
    assert carried("gsm8k", ["--num_fewshot", "8"]) == "it does not take --num_fewshot 8"
    assert carried("gsm8k", ["--num_samples", "many"]).startswith(
        "it cannot read the benchmark's arguments: argument --num_samples: invalid int value"
    )
    assert carried("mmlu", []) == "it scores gsm8k, not `mmlu`"
    # By its arguments as by its name: such a benchmark is not the session's endpoint's.
    asked = deployments.Benchmark(name="gsm8k", args=("--num_fewshot", "8"))
    assert session.endpoint_cannot(asked) == "it does not take --num_fewshot 8"
