# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared, deterministic GSM8K gate for DS-V4.1 aggregate and disaggregate CI."""

import json
import math
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

from .deepseek_v41_serving import MAX_SEQ_LEN

if TYPE_CHECKING:
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import RequestOutput, SamplingParams
    from tensorrt_llm.tokenizer import TokenizerBase

GSM8K_TEST_TIMEOUT = 7200
GSM8K_NUM_SAMPLES = 200
GSM8K_MAX_OUTPUT_TOKENS = 1024
GSM8K_CONCURRENCY = 8
GSM8K_SYSTEM_PROMPT = (
    "Solve the problem carefully. "
    "End your response with a final line exactly in the form #### <answer>."
)
# First 200 seed-0 documents from pre-CSA2 run 435344: 194/200 for both filters.
GSM8K_REFERENCE_SCORES = {"strict-match": 97.0, "flexible-extract": 97.0}


def _check_results(results: dict) -> dict[str, float]:
    """Reject incomplete or duplicated samples before applying an accuracy gate."""
    assert results["n-samples"]["gsm8k"]["effective"] == GSM8K_NUM_SAMPLES
    samples = results["samples"]["gsm8k"]
    assert len(samples) == GSM8K_NUM_SAMPLES * len(GSM8K_REFERENCE_SCORES)
    scores = {}
    for filter_name in GSM8K_REFERENCE_SCORES:
        rows = [row for row in samples if row["filter"] == filter_name]
        assert len(rows) == GSM8K_NUM_SAMPLES, filter_name
        assert {row["doc_id"] for row in rows} == set(range(GSM8K_NUM_SAMPLES)), filter_name
        assert all(row["exact_match"] in (0, 1) for row in rows), filter_name
        score = results["results"]["gsm8k"][f"exact_match,{filter_name}"]
        expected = 100 * sum(row["exact_match"] for row in rows) / GSM8K_NUM_SAMPLES
        assert math.isfinite(score) and math.isclose(score, expected, abs_tol=1e-9), filter_name
        scores[filter_name] = score
    return scores


def evaluate_gsm8k(
    llm: "LLM | SimpleNamespace",
    output_dir: Path,
    *,
    request_outputs: "list[RequestOutput] | None" = None,
) -> None:
    """Evaluate the same seed-0 subset with the validated 32-shot, non-thinking recipe."""
    import pytest

    from tensorrt_llm.evaluate import GSM8K
    from tensorrt_llm.llmapi import SamplingParams
    from tensorrt_llm.logger import logger

    from .accuracy.accuracy_core import HypothesisTestingParams
    from .conftest import llm_models_root

    evaluator = GSM8K(
        dataset_path=f"{llm_models_root()}/datasets/openai/gsm8k",
        num_samples=GSM8K_NUM_SAMPLES,
        random_seed=0,
        num_fewshot=32,
        apply_chat_template=True,
        fewshot_as_multiturn=False,
        system_prompt=GSM8K_SYSTEM_PROMPT,
        chat_template_kwargs={"enable_thinking": False},
        stop_strings=[],
        log_samples=True,
        output_path=str(output_dir),
    )
    task = evaluator.task_dict["gsm8k"]
    assert len(task.dataset["test"]) == 1319, "Expected the complete GSM8K test split"
    # Task generation kwargs override SamplingParams in LmEvalWrapper.
    task.set_config(
        key="generation_kwargs",
        value={**task.config.generation_kwargs, "max_gen_toks": GSM8K_MAX_OUTPUT_TOKENS},
    )
    sampling = SamplingParams(
        max_tokens=GSM8K_MAX_OUTPUT_TOKENS,
        truncate_prompt_tokens=MAX_SEQ_LEN,
        temperature=0,
        top_p=1,
        seed=0,
        stop=[],
        add_special_tokens=False,
    )
    submitted = 0

    def generate_async(
        prompt: str, sampling_params: "SamplingParams", streaming: bool = False
    ) -> "RequestOutput | Future[SimpleNamespace]":
        nonlocal submitted
        assert not streaming
        assert sampling_params.max_tokens == GSM8K_MAX_OUTPUT_TOKENS
        prompt_tokens = len(llm.tokenizer.encode(prompt, add_special_tokens=False))
        assert prompt_tokens + GSM8K_MAX_OUTPUT_TOKENS <= MAX_SEQ_LEN, "Prompt would be truncated"
        submitted += 1
        output = llm.generate_async(prompt, sampling_params=sampling_params, streaming=False)
        if request_outputs is not None:
            request_outputs.append(output)
        return output

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setenv("TLLM_EVAL_MAX_IN_FLIGHT", str(GSM8K_CONCURRENCY))
        monkeypatch.delenv("TLLM_EVAL_SPEC_STATS", raising=False)
        # Direct evaluation must not inherit AccuracyTask's INTEGRATION_TEST one-sample shortcut.
        evaluator.evaluate(
            SimpleNamespace(tokenizer=llm.tokenizer, generate_async=generate_async),
            sampling,
            scores_filter="exact_match,strict-match",
            sampling_override=True,
        )
    assert submitted == GSM8K_NUM_SAMPLES, f"Expected 200 requests, submitted {submitted}"
    with (output_dir / "samples_gsm8k.json").open() as stream:
        scores = _check_results(json.load(stream))
    for filter_name, score in scores.items():
        gate = HypothesisTestingParams(
            ref_accuracy=GSM8K_REFERENCE_SCORES[filter_name],
            num_samples=GSM8K_NUM_SAMPLES,
            metric_name=f"GSM8K {filter_name}",
        )
        logger.info(gate.report(score))
        gate.assert_passing(score)


def evaluate_gsm8k_server(
    server_url: str, model_path: str, tokenizer: "TokenizerBase", output_dir: Path
) -> None:
    """Check GSM8K accuracy and DSpark AL over non-retrying HTTP completions."""
    from openai import OpenAI

    with OpenAI(
        api_key="unused", base_url=f"{server_url}/v1", timeout=600, max_retries=0
    ) as client:
        pool = ThreadPoolExecutor(max_workers=GSM8K_CONCURRENCY)
        generation_stats: list[tuple[int, int]] = []

        def complete(prompt: str, sampling_params: "SamplingParams") -> SimpleNamespace:
            response = client.completions.create(
                model=model_path,
                prompt=prompt,
                max_tokens=sampling_params.max_tokens,
                temperature=sampling_params.temperature,
                top_p=sampling_params.top_p,
                seed=sampling_params.seed,
                stop=sampling_params.stop,
                n=1,
                stream=False,
                extra_body={"add_special_tokens": False},
            )
            assert len(response.choices) == 1
            choice = response.choices[0]
            assert choice.finish_reason in ("stop", "length"), choice
            assert response.usage is not None
            assert response.usage.prompt_tokens == len(
                tokenizer.encode(prompt, add_special_tokens=False)
            )
            tokens = response.usage.completion_tokens
            assert 0 < tokens <= GSM8K_MAX_OUTPUT_TOKENS
            acceptance_length = getattr(choice, "avg_decoded_tokens_per_iter", None)
            assert acceptance_length is not None, "Missing DSpark acceptance length"
            assert math.isfinite(acceptance_length) and acceptance_length > 0
            # With no stop strings, usage and AL count the same tokens, including EOS.
            iterations = round(tokens / acceptance_length)
            assert iterations > 0 and math.isclose(
                tokens / iterations, acceptance_length, rel_tol=1e-6
            ), (tokens, acceptance_length, iterations)
            generation_stats.append((tokens, iterations))
            return SimpleNamespace(outputs=[SimpleNamespace(text=choice.text)])

        def generate_async(
            prompt: str, sampling_params: "SamplingParams", streaming: bool = False
        ) -> Future[SimpleNamespace]:
            assert not streaming
            return pool.submit(complete, prompt, sampling_params)

        try:
            evaluate_gsm8k(
                SimpleNamespace(tokenizer=tokenizer, generate_async=generate_async), output_dir
            )
            assert generation_stats
            acceptance_length = sum(tokens for tokens, _ in generation_stats) / sum(
                iterations for _, iterations in generation_stats
            )
            print(f"[AL] DS-V4.1 disagg acceptance_length = {acceptance_length:.3f}")
            assert acceptance_length >= 3.0, (
                f"DSpark acceptance length {acceptance_length:.3f} < 3.0"
            )
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
