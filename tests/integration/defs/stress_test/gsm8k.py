# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configurable GSM8K accuracy evaluation against an existing stress-test server."""

import json
import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorrt_llm import SamplingParams


@dataclass(frozen=True)
class Gsm8kConfig:
    # Accuracy is a fraction in [0, 1].
    threshold: float
    num_samples: int = 200
    num_fewshot: int = 5
    concurrency: int = 8
    max_gen_toks: int = 256
    max_length: int = 4096
    request_timeout: int = 600
    random_seed: int = 0
    tokenizer: str | None = None
    apply_chat_template: bool = False
    system_prompt: str | None = None
    chat_template_kwargs: dict[str, object] = field(default_factory=dict)
    stop_strings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("GSM8K threshold must be a fraction between 0 and 1")
        if min(self.num_samples, self.concurrency, self.max_gen_toks, self.request_timeout) <= 0:
            raise ValueError(
                "GSM8K sample count, concurrency, output budget and timeout must be positive"
            )
        if self.num_fewshot < 0 or self.max_length <= self.max_gen_toks:
            raise ValueError("GSM8K needs nonnegative few-shot count and room for input tokens")


@dataclass
class Gsm8kResult:
    scores: dict[str, float]
    generation_stats: list[tuple[int, int]]


def _check_results(path: Path, config: Gsm8kConfig) -> dict[str, float]:
    with path.open() as stream:
        results = json.load(stream)
    assert results["n-samples"]["gsm8k"]["effective"] == config.num_samples
    samples = results["samples"]["gsm8k"]
    filters = ("strict-match", "flexible-extract")
    assert len(samples) == config.num_samples * len(filters)
    scores = {}
    for name in filters:
        rows = [row for row in samples if row["filter"] == name]
        assert len(rows) == config.num_samples, name
        assert {row["doc_id"] for row in rows} == set(range(config.num_samples)), name
        assert all(row["exact_match"] in (0, 1) for row in rows), name
        score = results["results"]["gsm8k"][f"exact_match,{name}"] / 100
        expected = sum(row["exact_match"] for row in rows) / config.num_samples
        assert math.isfinite(score) and math.isclose(score, expected, abs_tol=1e-9), name
        print(f"[GSM8K] {name}: {score:.2%}, threshold: {config.threshold:.2%}", flush=True)
        assert score >= config.threshold, (
            f"GSM8K {name} accuracy {score:.2%} is below {config.threshold:.2%}"
        )
        scores[name] = score
    return scores


def check_acceptance_length(result: Gsm8kResult, threshold: float) -> None:
    """Apply the optional speculative-decoding gate to token-weighted statistics."""
    if not math.isfinite(threshold) or threshold <= 0:
        raise ValueError("Acceptance-length threshold must be finite and positive")
    assert result.generation_stats, "No speculative-decoding statistics were recorded"
    acceptance_length = sum(tokens for tokens, _ in result.generation_stats) / sum(
        iterations for _, iterations in result.generation_stats
    )
    print(f"[AL] acceptance_length = {acceptance_length:.3f}", flush=True)
    assert acceptance_length >= threshold, (
        f"Speculative acceptance length {acceptance_length:.3f} is below {threshold}"
    )


def evaluate_gsm8k_server(
    server_url: str,
    model_path: str,
    config: Gsm8kConfig,
    output_dir: Path,
    *,
    timeout: int = 1200,
    collect_acceptance_stats: bool = False,
) -> Gsm8kResult:
    """Score every requested sample and enforce the configured accuracy threshold.

    Requests use the existing OpenAI-compatible server without retries. Normal
    models do not need to return speculative-decoding statistics.
    """
    import pytest
    from defs.conftest import llm_models_root

    if timeout <= 0:
        raise ValueError("GSM8K evaluation timeout must be positive")
    deadline = time.monotonic() + timeout
    from openai import OpenAI

    from tensorrt_llm import SamplingParams
    from tensorrt_llm.evaluate import GSM8K
    from tensorrt_llm.tokenizer import load_custom_tokenizer, tokenizer_factory

    if collect_acceptance_stats and config.stop_strings:
        raise ValueError("Acceptance-length validation requires no stop-string trimming")
    if config.tokenizer is not None:
        tokenizer = load_custom_tokenizer(config.tokenizer, model_path, local_files_only=True)
    else:
        tokenizer = tokenizer_factory(model_path, local_files_only=True)
    assert tokenizer is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = GSM8K(
        dataset_path=str(Path(llm_models_root()) / "datasets/openai/gsm8k"),
        num_samples=config.num_samples,
        random_seed=config.random_seed,
        num_fewshot=config.num_fewshot,
        apply_chat_template=config.apply_chat_template,
        fewshot_as_multiturn=False,
        system_prompt=config.system_prompt,
        chat_template_kwargs=config.chat_template_kwargs,
        stop_strings=config.stop_strings,
        log_samples=True,
        output_path=str(output_dir),
    )
    task = evaluator.task_dict["gsm8k"]
    assert len(task.dataset["test"]) >= config.num_samples
    task.set_config(
        key="generation_kwargs",
        value={**task.config.generation_kwargs, "max_gen_toks": config.max_gen_toks},
    )
    sampling = SamplingParams(
        max_tokens=config.max_gen_toks,
        truncate_prompt_tokens=config.max_length,
        temperature=0,
        top_p=1,
        seed=config.random_seed,
        stop=config.stop_strings,
        add_special_tokens=False,
    )
    submitted = 0
    generation_stats = []
    with OpenAI(
        api_key="unused", base_url=f"{server_url}/v1", timeout=config.request_timeout, max_retries=0
    ) as client:
        pool = ThreadPoolExecutor(max_workers=config.concurrency)

        def complete(prompt: str, params: "SamplingParams") -> SimpleNamespace:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("GSM8K evaluation exceeded its configured timeout")
            response = client.completions.create(
                model=model_path,
                prompt=prompt,
                max_tokens=params.max_tokens,
                temperature=params.temperature,
                top_p=params.top_p,
                seed=params.seed,
                stop=params.stop,
                n=1,
                stream=False,
                extra_body={"add_special_tokens": False},
                timeout=min(config.request_timeout, remaining),
            )
            assert len(response.choices) == 1
            choice = response.choices[0]
            assert choice.finish_reason in ("stop", "length"), choice
            assert response.usage is not None
            assert response.usage.prompt_tokens == len(
                tokenizer.encode(prompt, add_special_tokens=False)
            )
            tokens = response.usage.completion_tokens
            assert 0 < tokens <= config.max_gen_toks
            if collect_acceptance_stats:
                length = getattr(choice, "avg_decoded_tokens_per_iter", None)
                assert length is not None and math.isfinite(length) and length > 0
                iterations = round(tokens / length)
                assert iterations > 0 and math.isclose(tokens / iterations, length, rel_tol=1e-6)
                generation_stats.append((tokens, iterations))
            return SimpleNamespace(outputs=[SimpleNamespace(text=choice.text)])

        def generate_async(
            prompt: str, sampling_params: "SamplingParams", streaming: bool = False
        ) -> Future[SimpleNamespace]:
            nonlocal submitted
            assert not streaming
            assert sampling_params.max_tokens == config.max_gen_toks
            prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            assert prompt_tokens + config.max_gen_toks <= config.max_length, (
                "Prompt would be truncated"
            )
            submitted += 1
            return pool.submit(complete, prompt, sampling_params)

        try:
            with pytest.MonkeyPatch.context() as monkeypatch:
                monkeypatch.setenv("TLLM_EVAL_MAX_IN_FLIGHT", str(config.concurrency))
                monkeypatch.delenv("TLLM_EVAL_SPEC_STATS", raising=False)
                evaluator.evaluate(
                    SimpleNamespace(tokenizer=tokenizer, generate_async=generate_async),
                    sampling,
                    scores_filter="exact_match,strict-match",
                    sampling_override=True,
                )
            assert submitted == config.num_samples
            if time.monotonic() >= deadline:
                raise TimeoutError("GSM8K evaluation exceeded its configured timeout")
            return Gsm8kResult(
                _check_results(output_dir / "samples_gsm8k.json", config), generation_stats
            )
        finally:
            pool.shutdown(wait=False, cancel_futures=True)
