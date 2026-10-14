# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def gsm8k(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    # Avoid integration.defs.__init__ importing torch for these CPU-only checks.
    helper_dir = Path(__file__).resolve().parents[2] / "integration" / "defs"
    for name in ("deepseek_v41_serving", "deepseek_v41_gsm8k"):
        spec = importlib.util.spec_from_file_location(
            f"_dsv41_ci.{name}", helper_dir / f"{name}.py"
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, module)
        spec.loader.exec_module(module)
    return module


@pytest.mark.usefixtures("gsm8k")
def test_smoke_cases_format_long_recall_as_chat() -> None:
    serving = sys.modules["_dsv41_ci.deepseek_v41_serving"]
    tokenizer = Mock()
    tokenizer.encode.side_effect = lambda text, **kwargs: list(text)
    tokenizer.apply_chat_template.side_effect = (
        lambda messages, **kwargs: f"<chat>{messages[0]['content']}</chat>"
    )

    cases = serving.build_smoke_cases(tokenizer)

    assert tokenizer.apply_chat_template.call_count == len(cases) == 3
    assert cases[0].prompt.startswith("<chat>Ledger 47")
    assert cases[0].prompt.endswith("Reply with the digits only.\nAnswer:</chat>")
    assert 2 * serving.MAX_NUM_TOKENS < cases[0].input_tokens
    for call in tokenizer.apply_chat_template.call_args_list:
        assert call.args[0][0]["role"] == "user"
        assert call.kwargs == {
            "tokenize": False,
            "enable_thinking": False,
            "add_generation_prompt": True,
        }


def _results() -> dict:
    filters = ("strict-match", "flexible-extract")
    return {
        "n-samples": {"gsm8k": {"effective": 200}},
        "samples": {
            "gsm8k": [
                {"doc_id": doc_id, "filter": filter_name, "exact_match": 1}
                for filter_name in filters
                for doc_id in range(200)
            ]
        },
        "results": {"gsm8k": {f"exact_match,{name}": 100.0 for name in filters}},
    }


def test_check_results_accepts_complete_evidence(gsm8k: ModuleType) -> None:
    assert gsm8k._check_results(_results()) == {"strict-match": 100.0, "flexible-extract": 100.0}


@pytest.mark.parametrize(
    "corruption", ["missing", "duplicate", "filter", "count", "aggregate", "nan", "fractional"]
)
def test_check_results_rejects_invalid_evidence(gsm8k: ModuleType, corruption: str) -> None:
    results = _results()
    rows = results["samples"]["gsm8k"]
    if corruption == "missing":
        rows.pop()
    elif corruption == "duplicate":
        rows[1]["doc_id"] = rows[0]["doc_id"]
    elif corruption == "filter":
        rows[-1]["filter"] = "unknown"
    elif corruption == "count":
        results["n-samples"]["gsm8k"]["effective"] = 199
    elif corruption == "aggregate":
        results["results"]["gsm8k"]["exact_match,strict-match"] = 99.5
    elif corruption == "nan":
        results["results"]["gsm8k"]["exact_match,strict-match"] = float("nan")
    elif corruption == "fractional":
        rows[0]["exact_match"] = 0.5
    with pytest.raises(AssertionError):
        gsm8k._check_results(results)


@pytest.mark.parametrize("submitted", [199, 200])
def test_evaluate_gsm8k_recipe_and_gates(
    gsm8k: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, submitted: int
) -> None:
    monkeypatch.setenv("INTEGRATION_TEST", "1")
    monkeypatch.setenv("TLLM_EVAL_MAX_IN_FLIGHT", "1")
    monkeypatch.setenv("TLLM_EVAL_SPEC_STATS", "1")
    task = SimpleNamespace(
        dataset={"test": range(1319)},
        config=SimpleNamespace(generation_kwargs={"max_gen_toks": 256, "temperature": 0}),
        set_config=Mock(),
    )
    llm = SimpleNamespace(
        tokenizer=SimpleNamespace(encode=Mock(return_value=[1, 2, 3])), generate_async=Mock()
    )

    def evaluate(wrapper: SimpleNamespace, sampling: SimpleNamespace, **kwargs: object) -> None:
        assert os.environ["INTEGRATION_TEST"] == "1"
        assert os.environ["TLLM_EVAL_MAX_IN_FLIGHT"] == "8"
        assert "TLLM_EVAL_SPEC_STATS" not in os.environ
        assert kwargs == {"scores_filter": "exact_match,strict-match", "sampling_override": True}
        assert vars(sampling) == {
            "max_tokens": 1024,
            "truncate_prompt_tokens": 8192,
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
            "stop": [],
            "add_special_tokens": False,
        }
        for doc_id in range(submitted):
            wrapper.generate_async(f"prompt {doc_id}", sampling)
        (tmp_path / "samples_gsm8k.json").write_text(json.dumps(_results()))

    constructor = Mock(return_value=SimpleNamespace(task_dict={"gsm8k": task}, evaluate=evaluate))
    gates = [Mock(spec=["report", "assert_passing"]), Mock(spec=["report", "assert_passing"])]
    gate_constructor = Mock(side_effect=gates)
    for name, module in {
        "tensorrt_llm.evaluate": SimpleNamespace(GSM8K=constructor),
        "tensorrt_llm.llmapi": SimpleNamespace(SamplingParams=SimpleNamespace),
        "tensorrt_llm.logger": SimpleNamespace(logger=Mock()),
        "_dsv41_ci.accuracy.accuracy_core": SimpleNamespace(
            HypothesisTestingParams=gate_constructor
        ),
        "_dsv41_ci.conftest": SimpleNamespace(llm_models_root=lambda: "/models"),
    }.items():
        monkeypatch.setitem(sys.modules, name, module)
    check_results = Mock(wraps=gsm8k._check_results)
    monkeypatch.setattr(gsm8k, "_check_results", check_results)

    if submitted != 200:
        with pytest.raises(AssertionError, match="Expected 200 requests"):
            gsm8k.evaluate_gsm8k(llm, tmp_path)
        gate_constructor.assert_not_called()
        check_results.assert_not_called()
        return

    gsm8k.evaluate_gsm8k(llm, tmp_path)
    assert constructor.call_args.kwargs == {
        "dataset_path": "/models/datasets/openai/gsm8k",
        "num_samples": 200,
        "random_seed": 0,
        "num_fewshot": 32,
        "apply_chat_template": True,
        "fewshot_as_multiturn": False,
        "system_prompt": gsm8k.GSM8K_SYSTEM_PROMPT,
        "chat_template_kwargs": {"enable_thinking": False},
        "stop_strings": [],
        "log_samples": True,
        "output_path": str(tmp_path),
    }
    task.set_config.assert_called_once_with(
        key="generation_kwargs", value={"max_gen_toks": 1024, "temperature": 0}
    )
    assert llm.generate_async.call_count == 200
    check_results.assert_called_once_with(_results())
    for call, gate, filter_name in zip(
        gate_constructor.call_args_list, gates, gsm8k.GSM8K_REFERENCE_SCORES, strict=True
    ):
        assert call.kwargs == {
            "ref_accuracy": gsm8k.GSM8K_REFERENCE_SCORES[filter_name],
            "num_samples": 200,
            "metric_name": f"GSM8K {filter_name}",
        }
        gate.assert_passing.assert_called_once_with(100.0)
    assert os.environ["TLLM_EVAL_MAX_IN_FLIGHT"] == "1"
    assert os.environ["TLLM_EVAL_SPEC_STATS"] == "1"


@pytest.mark.parametrize("http_failure", [False, True])
def test_http_adapter_propagates_results_and_failures(
    gsm8k: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, http_failure: bool
) -> None:
    client = MagicMock()
    client.__enter__.return_value = client
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(text="#### 42", finish_reason="stop", avg_decoded_tokens_per_iter=4.0)
        ],
        usage=SimpleNamespace(prompt_tokens=3, completion_tokens=4),
    )
    client.completions.create.return_value = response
    if http_failure:
        client.completions.create.side_effect = RuntimeError("request failed")
    constructor = Mock(return_value=client)
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=constructor))
    tokenizer = SimpleNamespace(encode=Mock(return_value=[1, 2, 3]))

    def evaluate(llm: SimpleNamespace, output_dir: Path) -> None:
        assert output_dir == tmp_path
        sampling = SimpleNamespace(max_tokens=1024, temperature=0, top_p=1, seed=0, stop=[])
        result = llm.generate_async("question", sampling).result(timeout=5)
        assert result.outputs[0].text == "#### 42"

    monkeypatch.setattr(gsm8k, "evaluate_gsm8k", evaluate)
    if http_failure:
        with pytest.raises(RuntimeError, match="request failed"):
            gsm8k.evaluate_gsm8k_server("http://localhost:8000", "/model", tokenizer, tmp_path)
    else:
        gsm8k.evaluate_gsm8k_server("http://localhost:8000", "/model", tokenizer, tmp_path)
    constructor.assert_called_once_with(
        api_key="unused", base_url="http://localhost:8000/v1", timeout=600, max_retries=0
    )
    client.completions.create.assert_called_once_with(
        model="/model",
        prompt="question",
        max_tokens=1024,
        temperature=0,
        top_p=1,
        seed=0,
        stop=[],
        n=1,
        stream=False,
        extra_body={"add_special_tokens": False},
    )
