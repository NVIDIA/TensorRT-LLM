# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.executor.result import CompletionOutput, GenerationResult
from tensorrt_llm.llmapi import CudaGraphConfig, NGramDecodingConfig
from tensorrt_llm.llmapi import KvCacheConfig as TRT_KvCacheConfig


@pytest.fixture(scope="module")
def model_path():
    return llm_models_root() / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


@dataclass(frozen=True)
class _PenaltyE2ECase:
    name: str
    prompt: str
    sampling_params: SamplingParams


def _penalty_sampling_params(
    max_tokens: int = 1,
    logprobs: int = 1,
    **penalties: float | int,
) -> SamplingParams:
    return SamplingParams(
        max_tokens=max_tokens,
        temperature=1.3,
        seed=12345,
        ignore_eos=True,
        logprobs=logprobs,
        logprobs_mode="processed",
        return_generation_logits=True,
        **penalties,
    )


def _make_penalty_e2e_cases() -> list[_PenaltyE2ECase]:
    repeated_answer_prompt = "The capital of France is Paris. The capital of France is"
    capital_prompt = "The capital of France is"
    repeated_token_prompt = "cat cat cat cat The capital of France is"

    return [
        _PenaltyE2ECase(
            "repetition_discourage",
            repeated_answer_prompt,
            _penalty_sampling_params(repetition_penalty=100.0),
        ),
        _PenaltyE2ECase(
            "repetition_encourage",
            capital_prompt,
            _penalty_sampling_params(repetition_penalty=0.01),
        ),
        _PenaltyE2ECase(
            "additive_reward",
            repeated_token_prompt,
            _penalty_sampling_params(presence_penalty=-10.0, frequency_penalty=-2.0),
        ),
        _PenaltyE2ECase(
            "frequency_count",
            repeated_token_prompt,
            _penalty_sampling_params(frequency_penalty=5.0),
        ),
        _PenaltyE2ECase(
            "additive_prompt_ignored",
            capital_prompt,
            _penalty_sampling_params(
                presence_penalty=100.0,
                frequency_penalty=100.0,
                prompt_ignore_length=10_000,
            ),
        ),
        _PenaltyE2ECase(
            "combined_penalties",
            repeated_answer_prompt,
            _penalty_sampling_params(
                max_tokens=6,
                logprobs=5,
                repetition_penalty=1.7,
                presence_penalty=2.0,
                frequency_penalty=0.75,
                prompt_ignore_length=2,
            ),
        ),
    ]


def _create_torch_llm(
    model_dir: Path,
    max_batch_size: int | None = None,
    speculative_config: NGramDecodingConfig | None = None,
    enable_iter_perf_stats: bool = False,
) -> LLM:
    llm_kwargs: dict[str, object] = {}
    if max_batch_size is not None:
        llm_kwargs["max_batch_size"] = max_batch_size
    if speculative_config is not None:
        llm_kwargs["speculative_config"] = speculative_config

    return LLM(
        model=str(model_dir),
        tensor_parallel_size=1,
        trust_remote_code=True,
        enable_chunked_prefill=True,
        cuda_graph_config=CudaGraphConfig(),
        sampler_type="TorchSampler",
        kv_cache_config=TRT_KvCacheConfig(enable_block_reuse=False),
        max_num_tokens=128,
        enable_iter_perf_stats=enable_iter_perf_stats,
        **llm_kwargs,
    )


def _run_penalty_e2e_cases(
    model_dir: Path,
    cases: list[_PenaltyE2ECase],
) -> tuple[dict[str, GenerationResult], dict[str, tuple[int, ...]]]:
    with _create_torch_llm(model_dir) as llm:
        outputs = llm.generate(
            [case.prompt for case in cases],
            sampling_params=[case.sampling_params for case in cases],
            use_tqdm=False,
        )

    results = dict(zip((case.name for case in cases), outputs, strict=True))
    prompt_token_ids = {
        case.name: tuple(int(token_id) for token_id in output.prompt_token_ids)
        for case, output in zip(cases, outputs, strict=True)
    }
    return results, prompt_token_ids


def _reference_penalized_logits(
    raw_logits: torch.Tensor,
    token_history: list[int],
    prompt_length: int,
    sampling_params: SamplingParams,
) -> torch.Tensor:
    """Apply the documented penalties independently of TorchSampler."""
    vocab_size = raw_logits.numel()
    history = torch.tensor(token_history, dtype=torch.int64)
    valid_history = history[(history >= 0) & (history < vocab_size)]
    adjusted_logits = raw_logits.float()

    repetition_penalty = sampling_params.repetition_penalty or 1.0
    if repetition_penalty != 1.0 and valid_history.numel() > 0:
        repetition_mask = torch.bincount(valid_history, minlength=vocab_size).bool()
        repetition_scaled_logits = torch.where(
            adjusted_logits < 0,
            adjusted_logits * repetition_penalty,
            adjusted_logits / repetition_penalty,
        )
        adjusted_logits = torch.where(repetition_mask, repetition_scaled_logits, adjusted_logits)

    prompt_ignore_length = sampling_params.prompt_ignore_length or 0
    occurrence_start = max(0, min(prompt_ignore_length, prompt_length))
    occurrence_history = history[occurrence_start:]
    valid_occurrences = occurrence_history[
        (occurrence_history >= 0) & (occurrence_history < vocab_size)
    ]
    occurrence_counts = torch.bincount(valid_occurrences, minlength=vocab_size).float()

    temperature = sampling_params.temperature
    effective_temperature = 1.0 if temperature is None or temperature == 0.0 else temperature
    presence_penalty = sampling_params.presence_penalty or 0.0
    frequency_penalty = sampling_params.frequency_penalty or 0.0
    adjusted_logits -= presence_penalty * effective_temperature * (occurrence_counts > 0)
    adjusted_logits -= frequency_penalty * effective_temperature * occurrence_counts

    dtype_limit = torch.finfo(raw_logits.dtype).max
    return adjusted_logits.clamp(min=-dtype_limit, max=dtype_limit).to(raw_logits.dtype)


def _reference_processed_logprobs(
    raw_logits: torch.Tensor,
    token_history: list[int],
    prompt_length: int,
    sampling_params: SamplingParams,
) -> torch.Tensor:
    penalized_logits = _reference_penalized_logits(
        raw_logits,
        token_history,
        prompt_length,
        sampling_params,
    )
    temperature = sampling_params.temperature
    if temperature is not None and temperature != 0.0:
        penalized_logits = penalized_logits / max(temperature, 1e-5)
    processed_logits = penalized_logits.float()
    sampling_probs = torch.softmax(processed_logits, dim=-1)
    processed_logits = processed_logits.masked_fill(sampling_probs == 0, float("-inf"))
    return torch.log_softmax(processed_logits, dim=-1)


def _assert_completion_penalty_logprobs(
    case: _PenaltyE2ECase,
    completion: CompletionOutput,
    prompt_token_ids: tuple[int, ...],
) -> None:
    assert completion.token_ids is not None, case.name
    assert completion.generation_logits is not None, case.name
    assert completion.logprobs is not None, case.name

    token_history = list(prompt_token_ids)
    expected_cumulative_logprob = 0.0
    for step, (token_id, raw_logits, actual_logprobs) in enumerate(
        zip(completion.token_ids, completion.generation_logits, completion.logprobs, strict=True)
    ):
        location = f"{case.name}/step_{step}"
        assert token_id in actual_logprobs, location
        expected_logprobs = _reference_processed_logprobs(
            raw_logits,
            token_history,
            len(prompt_token_ids),
            case.sampling_params,
        )

        for returned_token_id, actual in actual_logprobs.items():
            assert actual.logprob == pytest.approx(
                float(expected_logprobs[returned_token_id]),
                rel=2e-5,
                abs=2e-4,
            ), location

        num_logprobs = case.sampling_params.logprobs
        if num_logprobs:
            ranked_logprobs = {
                actual.rank: actual.logprob
                for actual in actual_logprobs.values()
                if actual.rank is not None and actual.rank <= num_logprobs
            }
            assert set(ranked_logprobs) == set(range(1, num_logprobs + 1)), location
            expected_top_logprobs = torch.topk(expected_logprobs, k=num_logprobs).values
            for rank, expected in enumerate(expected_top_logprobs, start=1):
                assert ranked_logprobs[rank] == pytest.approx(
                    float(expected), rel=2e-5, abs=2e-4
                ), location

        expected_cumulative_logprob += float(expected_logprobs[token_id])
        token_history.append(token_id)

    assert completion.cumulative_logprob == pytest.approx(
        expected_cumulative_logprob, rel=2e-5, abs=2e-4
    ), case.name


@pytest.mark.high_cuda_memory
def test_torch_sampler_penalty_logits_e2e(model_path):
    """Validate TorchSampler's processed logits against the penalty formulas."""
    cases = _make_penalty_e2e_cases()
    results, prompt_token_ids = _run_penalty_e2e_cases(model_path, cases)

    for case in cases:
        for completion in results[case.name].outputs:
            _assert_completion_penalty_logprobs(
                case,
                completion,
                prompt_token_ids[case.name],
            )


@pytest.mark.high_cuda_memory
def test_torch_sampler_speculative_penalty_e2e(model_path):
    """Validate penalties through overlap scheduling and confirmed-history updates."""
    case = _PenaltyE2ECase(
        "ngram_speculative_penalties",
        "red blue red blue red blue red blue red blue",
        SamplingParams(
            max_tokens=8,
            temperature=0.0,
            seed=12345,
            ignore_eos=True,
            presence_penalty=0.01,
            frequency_penalty=0.01,
            prompt_ignore_length=2,
        ),
    )
    speculative_config = NGramDecodingConfig(
        max_draft_len=3,
        max_matching_ngram_size=2,
        is_keep_all=True,
        is_use_oldest=True,
        is_public_pool=False,
    )
    with _create_torch_llm(
        model_path,
        max_batch_size=1,
        speculative_config=speculative_config,
        enable_iter_perf_stats=True,
    ) as llm:
        speculative_outputs = llm.generate(
            [case.prompt],
            sampling_params=[case.sampling_params],
            use_tqdm=False,
        )
        stats = llm.get_stats(timeout=5)

    with _create_torch_llm(model_path, max_batch_size=1) as llm:
        reference_outputs = llm.generate(
            [case.prompt],
            sampling_params=[case.sampling_params],
            use_tqdm=False,
        )

    assert len(speculative_outputs) == len(reference_outputs) == 1
    speculative_stats = [
        stat["specDecodingStats"]
        for stat in stats
        if stat.get("specDecodingStats", {}).get("numDraftTokens", 0) > 0
    ]
    assert speculative_stats, "NGram must produce draft tokens in this test"
    assert sum(stat["numAcceptedTokens"] for stat in speculative_stats) > 0

    speculative_completions = speculative_outputs[0].outputs
    reference_completions = reference_outputs[0].outputs
    assert [completion.token_ids for completion in speculative_completions] == [
        completion.token_ids for completion in reference_completions
    ]
