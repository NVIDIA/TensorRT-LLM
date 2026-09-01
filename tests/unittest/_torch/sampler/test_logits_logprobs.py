import os
from itertools import product
from typing import Any, Final, Generator, cast

import numpy as np
import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.llm_data import llm_models_root
from utils.util import force_ampere

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.pyexecutor.llm_request import (
    LlmRequest,
    LlmRequestState,
    get_draft_token_length,
)
from tensorrt_llm._torch.pyexecutor.sampler.sampler import Logprob, ScheduledRequests, TorchSampler
from tensorrt_llm._torch.pyexecutor.sampler.sampler_strategy import _StrategyImpls
from tensorrt_llm.bindings import SamplingConfig
from tensorrt_llm.executor.result import TokenLogprobs
from tensorrt_llm.llmapi.llm_utils import KvCacheConfig
from tensorrt_llm.sampling_params import LogprobMode

prompts = ["A B C"]
global_kvcache_config = KvCacheConfig(
    max_tokens=10000,
    enable_block_reuse=True,
)

global_kvcache_config_prompt_logprobs = KvCacheConfig(
    max_tokens=10000,
    # FIXME: block reuse is disabled for prompt logprobs
    # because prompt logprobs are computed from context logits
    # and context logits may not be calculated when using block reuse
    # See https://nvbugs/5577178
    enable_block_reuse=False,
)


@pytest.fixture(autouse=True)
def _dynamo_recompile_headroom():
    """Recompile headroom for the fullgraph=True beam-search step.

    The beam-search cases here build a fresh engine per parametrization in one
    process, and Dynamo counts recompiles per code object across the process,
    so the default limit (8) is exhausted partway through and the rest hard-fail
    under fullgraph. The limit guards against runaway recompilation and is not a
    correctness property; a served model has a fixed set of shapes.
    """
    import torch._dynamo

    with torch._dynamo.config.patch(recompile_limit=128):
        yield


@pytest.fixture(scope="module", params=[False, True])
def disable_overlap_scheduler_fixture(request) -> bool:
    return request.param


@pytest.fixture(scope="module", params=[False, True])
def enable_early_first_token_response_fixture(request) -> bool:
    return request.param


class CacheSalter:
    _salt = 0

    @classmethod
    def get_salt_unique(cls) -> str:
        cls._salt += 1
        return str(cls._salt)

    @classmethod
    def get_salt_shared(cls) -> str:
        return str(0)

    @classmethod
    def get_salt(cls, reuse_cache: bool) -> str:
        if reuse_cache:
            salt = cls.get_salt_shared()
        else:
            salt = cls.get_salt_unique()
        return salt


@pytest.fixture(scope="module")
def llm(
    disable_overlap_scheduler_fixture: bool,
    enable_early_first_token_response_fixture: bool,
):
    disable_overlap_scheduler = disable_overlap_scheduler_fixture
    enable_early_first_token_response = enable_early_first_token_response_fixture

    if enable_early_first_token_response and disable_overlap_scheduler:
        pytest.skip(
            "enable_early_first_token_response is relevant only when the overlap scheduler is enabled."
        )

    llm = LLM(
        model=os.path.join(llm_models_root(), "llama-models-v2", "TinyLlama-1.1B-Chat-v1.0"),
        kv_cache_config=global_kvcache_config,
        max_batch_size=128,  # reduce buffer sizes, specially for generation logits
        disable_overlap_scheduler=disable_overlap_scheduler,
        enable_early_first_token_response=enable_early_first_token_response,
    )
    with llm:
        yield llm


@pytest.fixture(scope="module")
def simple_llm() -> LLM:
    llm = LLM(
        model=os.path.join(llm_models_root(), "llama-models-v2", "TinyLlama-1.1B-Chat-v1.0"),
        max_batch_size=8,
        kv_cache_config=global_kvcache_config_prompt_logprobs,
    )
    return llm


def check_generated_output(
    gather_context_logits,
    gather_generation_logits,
    sampling_params,
    reuse_cache,
    return_log_probs,
    idx,
    output,
    streaming,
):
    if gather_context_logits:
        assert output.context_logits is not None
        # NOTE: prompt_token_ids of "A B C" becomes [1, 319, 350, 315]
        expected_len = len(prompts[0].split()) + 1
        assert expected_len == output.context_logits.shape[0]
    else:
        assert output.context_logits is None

    for sequence in output.outputs:
        if streaming:
            assert sequence.length == idx + 1
        else:
            assert sequence.length == sampling_params.max_tokens

        if gather_generation_logits:
            gen_logits = sequence.generation_logits
            assert gen_logits is not None
            assert gen_logits.ndim == 2
            if streaming:
                assert gen_logits.shape[0] == 1
                assert torch.argmax(gen_logits, dim=1).tolist()[0] == sequence.token_ids[-1]
            else:
                assert gen_logits.shape[0] == sampling_params.max_tokens
                assert torch.argmax(gen_logits, dim=1).tolist() == sequence.token_ids
        else:
            assert sequence.generation_logits is None
        if return_log_probs:
            assert len(sequence.logprobs) == sequence.length
        else:
            assert len(sequence.logprobs) == 0


@force_ampere  # Save H100 resource
@pytest.mark.parametrize("reuse_cache", [False, True])
@pytest.mark.parametrize("return_log_probs", [False, True])
@pytest.mark.parametrize("gather_generation_logits", [False, True])
@pytest.mark.parametrize("gather_context_logits", [False, True])
@pytest.mark.parametrize("async_generation", [False, True])
@pytest.mark.timeout(120, method="signal")
@pytest.mark.threadleak(enabled=False)
def test_generation_with_return_logits(
    llm,
    gather_context_logits: bool,
    gather_generation_logits: bool,
    reuse_cache: bool,
    return_log_probs: bool,
    async_generation: bool,
):
    if not (gather_context_logits or gather_generation_logits or return_log_probs):  # prune space
        pytest.skip("Nothing to test")
    if reuse_cache and gather_context_logits:
        pytest.skip("nvbugs/5577178")

    sampling_params = SamplingParams(
        max_tokens=8,
        return_context_logits=gather_context_logits,
        return_generation_logits=gather_generation_logits,
        logprobs=return_log_probs,
    )

    if async_generation:
        for idx, output in enumerate(
            llm.generate_async(
                prompts[0],
                sampling_params=sampling_params,
                streaming=True,
                cache_salt=CacheSalter.get_salt(reuse_cache),
            )
        ):
            check_generated_output(
                gather_context_logits=gather_context_logits,
                gather_generation_logits=gather_generation_logits,
                sampling_params=sampling_params,
                reuse_cache=reuse_cache,
                return_log_probs=return_log_probs,
                idx=idx,
                output=output,
                streaming=True,
            )
        assert idx == sampling_params.max_tokens - 1
    else:
        for idx, output in enumerate(
            llm.generate(
                prompts,
                sampling_params=sampling_params,
                cache_salt=[CacheSalter.get_salt(reuse_cache) for _ in prompts],
            )
        ):
            check_generated_output(
                gather_context_logits=gather_context_logits,
                gather_generation_logits=gather_generation_logits,
                sampling_params=sampling_params,
                reuse_cache=reuse_cache,
                return_log_probs=return_log_probs,
                idx=idx,
                output=output,
                streaming=False,
            )
        assert idx == len(prompts) - 1


@pytest.mark.parametrize("logprobs_k", [0, 1, 3], ids=["top_0", "top_1", "top_3"])
@pytest.mark.parametrize("logprobs_mode", ["raw", "processed"])
@pytest.mark.threadleak(enabled=False)
def test_sampled_token_always_in_logprobs(logprobs_k: int, logprobs_mode: str, simple_llm: LLM):
    """Two scenarios:
    - logprobs=0: Returns only sampled token (1 element)
    - logprobs=K (K>0): Returns top-K tokens + sampled token if not in top-K (up to K+1 elements)
    """

    sampling_params = SamplingParams(
        max_tokens=8,
        temperature=0.7,
        top_p=0.9,
        logprobs=logprobs_k,
        logprobs_mode=logprobs_mode,
    )

    for output in simple_llm.generate(["The future of AI is"], sampling_params=sampling_params):
        print(f"\n{'=' * 80}")
        print(f"Generated text: {output.outputs[0].text!r}")
        print(f"Generated token IDs: {output.outputs[0].token_ids}")

        logprobs = output.outputs[0].logprobs
        token_ids = output.outputs[0].token_ids

        assert len(logprobs) == sampling_params.max_tokens, (
            f"Expected {sampling_params.max_tokens} logprob entries, got {len(logprobs)}"
        )

        for token_idx, (sampled_token_id, token_logprobs) in enumerate(zip(token_ids, logprobs)):
            print(
                f"\n  Token {token_idx}: "
                f"ID={sampled_token_id}, "
                f"Text={simple_llm.tokenizer.decode([sampled_token_id])!r}"
            )

            assert sampled_token_id in token_logprobs, (
                f"Token {token_idx}: Sampled token ID {sampled_token_id} not in logprobs dict: {token_logprobs.keys()}"
            )

            if logprobs_k == 0:
                assert len(token_logprobs) == 1, (
                    f"Token {token_idx}: Expected 1 logprob (sampled only), got {len(token_logprobs)}"
                )
            else:
                assert len(token_logprobs) <= logprobs_k + 1, (
                    f"Token {token_idx}: Expected at most {logprobs_k + 1} logprobs, got {len(token_logprobs)}"
                )
                assert len(token_logprobs) >= max(logprobs_k, 1)

            sorted_tokens_by_prob = sorted(
                token_logprobs.items(), key=lambda x: (x[1].logprob, -x[1].rank), reverse=True
            )

            if logprobs_k > 0:
                sampled_token_rank = token_logprobs[sampled_token_id].rank
                sampled_in_topk = sampled_token_rank <= logprobs_k

                if not sampled_in_topk:
                    assert sorted_tokens_by_prob[-1][0] == sampled_token_id, (
                        f"Token {token_idx}: Sampled token (ID={sampled_token_id}, rank={sampled_token_rank}) "
                        f"not in top-{logprobs_k}, should be last in sorted list, "
                        f"but last token is ID={sorted_tokens_by_prob[-1][0]}"
                    )

            for rank_idx, (token_id, logprob_obj) in enumerate(sorted_tokens_by_prob, start=1):
                token_text = simple_llm.tokenizer.decode([token_id])
                is_sampled = "← SAMPLED" if token_id == sampled_token_id else ""
                print(
                    f"    • Token {token_id:5d} ({token_text:15s}): "
                    f"logprob={logprob_obj.logprob:8.4f}, "
                    f"rank={logprob_obj.rank} {is_sampled}"
                )

                if logprobs_k > 0 and sampled_in_topk:
                    assert logprob_obj.rank == rank_idx, (
                        f"Token {token_idx}: Token {token_id} rank mismatch. "
                        f"Expected rank {rank_idx} (by sorted position), got {logprob_obj.rank}"
                    )

        print(f"{'=' * 80}\n")


@pytest.mark.parametrize("logprobs_k", [0, 1, 3], ids=["top_0", "top_1", "top_3"])
@pytest.mark.threadleak(enabled=False)
def test_sampled_token_always_in_prompt_logprobs(logprobs_k: int, simple_llm: LLM):
    """Two scenarios:
    - logprobs=0: Returns only sampled token (1 element)
    - logprobs=K (K>0): Returns top-K tokens + sampled token if not in top-K (up to K+1 elements)
    """

    sampling_params = SamplingParams(
        max_tokens=1,
        prompt_logprobs=logprobs_k,
    )

    for output in simple_llm.generate(["The future of AI is"], sampling_params=sampling_params):
        print(f"\n{'=' * 80}")
        print(f"Prompt text: {output.prompt!r}")
        print(f"Prompt token IDs: {output.prompt_token_ids}")

        logprobs = output.outputs[0].prompt_logprobs
        token_ids = output.prompt_token_ids[1:] + output.outputs[0].token_ids[:1]

        assert len(logprobs) == len(token_ids), (
            f"Expected {len(token_ids)} logprob entries, got {len(logprobs)}"
        )

        for token_idx, (sampled_token_id, token_logprobs) in enumerate(zip(token_ids, logprobs)):
            print(
                f"\n  Token {token_idx}: "
                f"ID={sampled_token_id}, "
                f"Text={simple_llm.tokenizer.decode([sampled_token_id])!r}"
            )

            assert sampled_token_id in token_logprobs, (
                f"Token {token_idx}: Sampled token ID {sampled_token_id} not in logprobs dict: {token_logprobs.keys()}"
            )

            if logprobs_k == 0:
                assert len(token_logprobs) == 1, (
                    f"Token {token_idx}: Expected 1 logprob (sampled only), got {len(token_logprobs)}"
                )
            else:
                assert len(token_logprobs) <= logprobs_k + 1, (
                    f"Token {token_idx}: Expected at most {logprobs_k + 1} logprobs, got {len(token_logprobs)}"
                )
                assert len(token_logprobs) >= 1

            sorted_tokens_by_prob = sorted(
                token_logprobs.items(), key=lambda x: (x[1].logprob, -x[1].rank), reverse=True
            )

            if logprobs_k > 0:
                sampled_token_rank = token_logprobs[sampled_token_id].rank
                sampled_in_topk = sampled_token_rank <= logprobs_k

                if not sampled_in_topk:
                    assert sorted_tokens_by_prob[-1][0] == sampled_token_id, (
                        f"Token {token_idx}: Sampled token (ID={sampled_token_id}, rank={sampled_token_rank}) "
                        f"not in top-{logprobs_k}, should be last in sorted list, "
                        f"but last token is ID={sorted_tokens_by_prob[-1][0]}"
                    )

        print(f"{'=' * 80}\n")


@pytest.mark.threadleak(enabled=False)
def test_logprobs_simple_format(simple_llm: LLM):
    """When ``logprobs_simple_format=True`` and ``prompt_logprobs_simple_format=True``
    with the corresponding K==0, the per-token logprobs are returned as a flat
    ``list[float]`` instead of the default ``list[dict[int, Logprob]]`` and the
    numeric values match the dict-format path within tolerance."""

    prompt = "The future of AI is"
    common_kwargs = dict(max_tokens=8, temperature=0.0)

    dict_params = SamplingParams(logprobs=0, prompt_logprobs=0, **common_kwargs)
    simple_params = SamplingParams(
        logprobs=0,
        prompt_logprobs=0,
        logprobs_simple_format=True,
        prompt_logprobs_simple_format=True,
        **common_kwargs,
    )

    [dict_out] = list(simple_llm.generate([prompt], sampling_params=dict_params))
    [simple_out] = list(simple_llm.generate([prompt], sampling_params=simple_params))

    dict_gen_logprobs = dict_out.outputs[0].logprobs
    simple_gen_logprobs = simple_out.outputs[0].logprobs

    # Simple format must be list[float]; dict format must remain list[dict].
    assert all(isinstance(x, float) for x in simple_gen_logprobs), (
        f"Expected list[float], got element types: {[type(x) for x in simple_gen_logprobs]}"
    )
    assert all(isinstance(x, dict) for x in dict_gen_logprobs)

    for token_id, lp_simple, lp_dict in zip(
        dict_out.outputs[0].token_ids, simple_gen_logprobs, dict_gen_logprobs, strict=True
    ):
        torch.testing.assert_close(
            torch.tensor(lp_simple, dtype=torch.float32),
            torch.tensor(lp_dict[token_id].logprob, dtype=torch.float32),
            atol=1e-4,
            rtol=0,
        )

    dict_prompt_logprobs = dict_out.outputs[0].prompt_logprobs
    simple_prompt_logprobs = simple_out.outputs[0].prompt_logprobs
    assert all(isinstance(x, float) for x in simple_prompt_logprobs)
    assert all(isinstance(x, dict) for x in dict_prompt_logprobs)
    prompt_token_ids = dict_out.prompt_token_ids[1:] + dict_out.outputs[0].token_ids[:1]
    for token_id, lp_simple, lp_dict in zip(
        prompt_token_ids, simple_prompt_logprobs, dict_prompt_logprobs, strict=True
    ):
        torch.testing.assert_close(
            torch.tensor(lp_simple, dtype=torch.float32),
            torch.tensor(lp_dict[token_id].logprob, dtype=torch.float32),
            atol=1e-4,
            rtol=0,
        )


def test_logprobs_simple_format_validation():
    """``SamplingParams`` rejects incompatible combinations of the simple-format
    flag with non-zero / unset ``logprobs`` and with beam search."""
    SamplingParams(max_tokens=4, logprobs=0, logprobs_simple_format=True)
    SamplingParams(max_tokens=4, prompt_logprobs=0, prompt_logprobs_simple_format=True)

    with pytest.raises(ValueError, match=r"logprobs_simple_format=True requires logprobs == 0"):
        SamplingParams(max_tokens=4, logprobs=2, logprobs_simple_format=True)
    with pytest.raises(ValueError, match=r"logprobs_simple_format=True requires logprobs == 0"):
        SamplingParams(max_tokens=4, logprobs=None, logprobs_simple_format=True)
    with pytest.raises(
        ValueError, match=r"prompt_logprobs_simple_format=True requires prompt_logprobs == 0"
    ):
        SamplingParams(max_tokens=4, prompt_logprobs=3, prompt_logprobs_simple_format=True)
    with pytest.raises(
        ValueError, match=r"prompt_logprobs_simple_format=True requires prompt_logprobs == 0"
    ):
        SamplingParams(max_tokens=4, prompt_logprobs=None, prompt_logprobs_simple_format=True)
    with pytest.raises(ValueError, match="beam search"):
        SamplingParams(
            max_tokens=4,
            logprobs=0,
            logprobs_simple_format=True,
            use_beam_search=True,
            best_of=2,
            n=2,
        )


@pytest.mark.parametrize("logprobs_k", [None, 0, 3], ids=["None", "top_0", "top_3"])
@pytest.mark.parametrize("prompt_logprobs_k", [None, 0, 3], ids=["None", "top_0", "top_3"])
@pytest.mark.threadleak(enabled=False)
def test_logprobs_against_logits(
    logprobs_k: int | None, prompt_logprobs_k: int | None, simple_llm: LLM
):
    """
    Test combination of logprobs and prompt_logprobs against manually calculated log probabilities from logits
    """

    sampling_params = SamplingParams(
        max_tokens=8,
        logprobs=logprobs_k,
        prompt_logprobs=prompt_logprobs_k,
        return_generation_logits=True,
        return_context_logits=True,
    )

    def check_logprobs(
        num_logprobs: int,
        tokens: list[int],
        logprobs: TokenLogprobs,
        logits_cuda: torch.Tensor,
        case_str: str,
        logprobs_offset: int = 0,
    ):
        """Checks if the provided logprobs match the logprobs calculated from the logits"""
        expected_logprobs = torch.nn.functional.log_softmax(logits_cuda, dim=-1).to(device="cpu")
        sorted_expected_logprobs = torch.sort(expected_logprobs, descending=True, dim=-1)[0]
        for generation_idx, token_logprobs in enumerate(logprobs):
            assert len(token_logprobs) <= num_logprobs + 1, "too many logprobs provided"
            assert len(token_logprobs) >= num_logprobs, "not enough logprobs provided"
            expected_logprobs_per_token = expected_logprobs[generation_idx]
            sorted_expected_logprobs_per_token = sorted_expected_logprobs[generation_idx]
            # For each rank store the corresponding logprob to ensure that shared ranks have the same logprob
            processed_ranks_and_logprobs: dict[int, float] = {}
            for token_id, logprob_obj in token_logprobs.items():
                # the sampled token may have any rank > 0
                if token_id != tokens[generation_idx + logprobs_offset]:
                    # All other tokens should have a rank <= num_logprobs
                    assert logprob_obj.rank <= num_logprobs, (
                        f"{case_str} logprob rank is greater than {num_logprobs}"
                    )
                assert logprob_obj.rank >= 1, f"{case_str} logprob rank is smaller than 1"

                # Shared ranks should not exist
                assert logprob_obj.rank not in processed_ranks_and_logprobs, (
                    f"Found shared rank {logprob_obj.rank} with logprob {logprob_obj.logprob}"
                )
                processed_ranks_and_logprobs[logprob_obj.rank] = logprob_obj.logprob

                # Check if the logprob matches the top-rank logprob from the logits
                torch.testing.assert_close(
                    torch.tensor(logprob_obj.logprob, dtype=torch.float32),
                    torch.tensor(
                        sorted_expected_logprobs_per_token[logprob_obj.rank - 1],
                        dtype=torch.float32,
                    ),
                    msg=f"Returned {case_str} logprob {logprob_obj.logprob} does not match expected logprob \
                        {sorted_expected_logprobs_per_token[logprob_obj.rank - 1]} at rank {logprob_obj.rank}",
                )
                # Check if the logprob matches the token-id logprob from the logits
                torch.testing.assert_close(
                    torch.tensor(logprob_obj.logprob, dtype=torch.float32),
                    torch.tensor(expected_logprobs_per_token[token_id], dtype=torch.float32),
                    msg=f"Returned {case_str} logprob {logprob_obj.logprob} does not match expected logprob \
                        {expected_logprobs_per_token[token_id]} for token {token_id}",
                )

    for output in simple_llm.generate(["The future of AI is"], sampling_params=sampling_params):
        if logprobs_k is not None:
            generation_tokens = output.outputs[0].token_ids
            generation_logprobs = output.outputs[0].logprobs
            generation_logits = output.outputs[0].generation_logits.to(device="cuda")
            check_logprobs(
                logprobs_k,
                generation_tokens,
                generation_logprobs,
                generation_logits,
                "generation",
                logprobs_offset=0,
            )
        if prompt_logprobs_k is not None:
            context_tokens = output.prompt_token_ids + output.outputs[0].token_ids[:1]
            context_logprobs = output.outputs[0].prompt_logprobs
            context_logits = output.context_logits.to(device="cuda")
            check_logprobs(
                prompt_logprobs_k,
                context_tokens,
                context_logprobs,
                context_logits,
                "context",
                logprobs_offset=1,  # Prompt logprobs are offset by 1 relative to the prompt token ids
            )
        # The last context logprob dict and the first generation logprob dict should agree on
        # the top-n entries (n = min(prompt_logprobs_k, logprobs_k)) and the sampled token's logprob.
        if prompt_logprobs_k is not None and logprobs_k is not None:
            last_context_logprob = context_logprobs[-1]
            first_generation_logprob = generation_logprobs[0]
            less_prompt_logprobs = prompt_logprobs_k <= logprobs_k
            expected = last_context_logprob if less_prompt_logprobs else first_generation_logprob
            compare = first_generation_logprob if less_prompt_logprobs else last_context_logprob
            sampled_token_id = generation_tokens[0]
            assert sampled_token_id in last_context_logprob, (
                f"Sampled token {sampled_token_id} is not a valid key in the last entry "
                f"of the context logprob dict: {list(last_context_logprob.keys())}"
            )
            assert sampled_token_id in first_generation_logprob, (
                f"Sampled token {sampled_token_id} is not a valid key in the first entry "
                f"of the generation logprob dict: {list(first_generation_logprob.keys())}"
            )
            torch.testing.assert_close(
                last_context_logprob[sampled_token_id].logprob,
                first_generation_logprob[sampled_token_id].logprob,
                msg=(
                    f"logprob {last_context_logprob[sampled_token_id].logprob} in the last "
                    f"entry of the context logprob dict does not match the corresponding "
                    f"logprob {first_generation_logprob[sampled_token_id].logprob} in the "
                    f"first entry of the generation logprob dict for token {sampled_token_id}"
                ),
            )
            for token_id, logprob_obj in expected.items():
                assert token_id in compare, (
                    f"Token {token_id} is not a valid key in the other dict: {list(compare.keys())}"
                )
                expected_logprob = logprob_obj.logprob
                compare_logprob = compare[token_id].logprob
                torch.testing.assert_close(
                    expected_logprob,
                    compare_logprob,
                    msg=(
                        f"logprob {expected_logprob} does not match the corresponding "
                        f"logprob {compare_logprob} in the other dict for token {token_id}"
                    ),
                )


@pytest.mark.parametrize("logprobs_k", [0, 2], ids=["top_0", "top_2"])
@pytest.mark.threadleak(enabled=False)
def test_logprobs_with_grouped_samplings_strategies(logprobs_k: int, simple_llm: LLM):
    """Test logprobs when requests are reordered by sampling strategy grouping"""

    test_prompts = [
        "The capital of France is",
        "The future of AI is",
        "Hello, my name is",
        "Hello, my name is",
        "Write a short story about a cat",
    ]

    # Causes reordering: [0,1,2,3,4] → [0,2,3,1,4]
    sampling_params_list = [
        SamplingParams(
            max_tokens=6,
            temperature=0.8,
            top_k=50,
            logprobs=logprobs_k,
            return_generation_logits=True,
        ),
        SamplingParams(
            max_tokens=6,
            temperature=0.8,
            top_p=0.9,
            logprobs=logprobs_k,
            return_generation_logits=True,
        ),
        SamplingParams(
            max_tokens=6,
            temperature=0.8,
            top_k=50,
            logprobs=logprobs_k,
            return_generation_logits=True,
        ),
        SamplingParams(
            max_tokens=6, temperature=0.8, top_k=50, logprobs=None, return_generation_logits=True
        ),
        SamplingParams(
            max_tokens=6,
            temperature=0.8,
            top_p=0.9,
            logprobs=logprobs_k,
            return_generation_logits=True,
        ),
    ]

    outputs = list(simple_llm.generate(test_prompts, sampling_params=sampling_params_list))

    for req_idx, output in enumerate(outputs):
        generation_logits = output.outputs[0].generation_logits.to(device="cuda")
        token_ids = output.outputs[0].token_ids
        logprobs = output.outputs[0].logprobs
        if sampling_params_list[req_idx].logprobs is None:
            assert len(logprobs) == 0
            continue

        assert generation_logits is not None
        assert len(logprobs) == len(token_ids), "Logprobs length mismatch"

        # generation_logits might be shorter than token_ids
        num_logits = len(generation_logits)

        for token_idx, (sampled_token_id, token_logprobs_dict) in enumerate(
            zip(token_ids[:num_logits], logprobs[:num_logits])
        ):
            returned_logprob = token_logprobs_dict[sampled_token_id].logprob

            logits_for_token = generation_logits[token_idx]
            expected_logprobs = torch.nn.functional.log_softmax(logits_for_token, dim=-1).to(
                device="cpu"
            )
            expected_logprob = expected_logprobs[sampled_token_id]
            print(
                f"Req {req_idx}, Token {token_idx}: returned={returned_logprob:.6f}, "
                f"expected={expected_logprob.item():.6f}"
            )
            torch.testing.assert_close(
                torch.tensor(returned_logprob, dtype=torch.float32),
                expected_logprob,
            )


@pytest.mark.parametrize("logprobs_k", [-5], ids=["invalid_negative_value"])
@pytest.mark.threadleak(enabled=False)
def test_invalid_logprobs(logprobs_k: int):
    """Test invalid logprobs values"""
    with pytest.raises(ValueError):
        SamplingParams(logprobs=logprobs_k)
    with pytest.raises(ValueError):
        SamplingParams(prompt_logprobs=logprobs_k)


@pytest.mark.parametrize("logprobs_k", [0, 2], ids=["top_0", "top_2"])
@pytest.mark.threadleak(enabled=False)
def test_processed_logprobs_e2e(logprobs_k: int, simple_llm: LLM):
    """Test logprobs when requests are reordered by sampling strategy grouping"""
    test_prompts = [
        "The capital of France is",
        "The future of AI is",
        "Hello, my name is",
        "Write a short story about a cat",
        "Hello, my name is",
        "Write a short story about a cat",
    ]

    sampling_params_list = [
        # greedy decoding
        SamplingParams(
            max_tokens=6,
            temperature=0.0,
            logprobs=logprobs_k,
            return_generation_logits=True,
            logprobs_mode="processed",
        ),
        # temperature sampling
        SamplingParams(
            max_tokens=6,
            temperature=0.8,
            logprobs=logprobs_k,
            return_generation_logits=True,
            logprobs_mode="processed",
        ),
        # top-p sampling
        SamplingParams(
            max_tokens=6,
            temperature=0.8,
            top_p=0.9,
            logprobs=logprobs_k,
            return_generation_logits=True,
            logprobs_mode="processed",
        ),
        # top-k sampling
        SamplingParams(
            max_tokens=6,
            temperature=0.8,
            top_k=50,
            logprobs=logprobs_k,
            return_generation_logits=True,
            logprobs_mode="processed",
        ),
        # top-p sampling 2
        SamplingParams(
            max_tokens=6,
            temperature=0.8,
            top_p=0.9,
            logprobs=logprobs_k,
            return_generation_logits=True,
            logprobs_mode="processed",
        ),
        # top-p and top-k sampling
        SamplingParams(
            max_tokens=6,
            temperature=0.8,
            top_p=0.9,
            top_k=50,
            logprobs=logprobs_k,
            return_generation_logits=True,
            logprobs_mode="processed",
        ),
    ]

    outputs = list(simple_llm.generate(test_prompts, sampling_params=sampling_params_list))

    for req_idx, output in enumerate(outputs):
        generation_logits = output.outputs[0].generation_logits.to(device="cuda")
        token_ids = output.outputs[0].token_ids
        logprobs = output.outputs[0].logprobs

        assert generation_logits is not None
        assert len(logprobs) == len(token_ids), "Logprobs length mismatch"

        # generation_logits might be shorter than token_ids
        num_logits = len(generation_logits)

        for token_idx, token_logprobs_dict in enumerate(logprobs[:num_logits]):
            assert token_ids[token_idx] in token_logprobs_dict, "Sampled token not in logprobs"

            logits_for_token = generation_logits[token_idx : token_idx + 1]
            topk = sampling_params_list[req_idx].top_k
            topp = sampling_params_list[req_idx].top_p
            temperature = sampling_params_list[req_idx].temperature
            if sampling_params_list[req_idx]._greedy_decoding:
                probs = torch.zeros_like(logits_for_token)
                probs[0, token_ids[token_idx]] = 1.0
            else:
                topk = topk if topk is not None else logits_for_token.shape[-1]
                topp = topp if topp is not None else 1.0
                temperature = temperature if temperature is not None else 1.0

                # perform masking top-k top-p via the flashinfer strategy impl
                _, probs = _StrategyImpls.StrategyImplWithProbs._sample_with_probs(
                    logits_for_token,
                    group_logit_indices=None,
                    top_k=torch.tensor([topk], dtype=torch.int32, device="cuda"),
                    top_p=torch.tensor([topp], dtype=torch.float32, device="cuda"),
                    # None disables the min-p stage; no request here sets min_p.
                    min_p=None,
                    temperature=torch.tensor([temperature], dtype=torch.float32, device="cuda"),
                    generator=None,
                )

            if temperature != 0:
                logits_for_token /= temperature
            adjusted_logits_for_token = torch.where(probs != 0, logits_for_token, float("-inf"))[0]
            expected_logprobs = torch.nn.functional.log_softmax(
                adjusted_logits_for_token, dim=-1
            ).to(device="cpu")
            for logprob_token, logprob_values in token_logprobs_dict.items():
                expected_logprob = expected_logprobs[logprob_token]
                returned_logprob = logprob_values.logprob
                print(
                    f"Req {req_idx}, Token {token_idx}: "
                    f"returned={returned_logprob:.6f}, expected={expected_logprob.item():.6f}"
                )
                torch.testing.assert_close(
                    torch.tensor(returned_logprob, dtype=torch.float32),
                    expected_logprob,
                )


@force_ampere
@pytest.mark.gpu2
def test_logprobs_match_hf_tp2():
    model_path = os.path.join(llm_models_root(), "llama-models-v2", "TinyLlama-1.1B-Chat-v1.0")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
    )

    prompts = ["The future of the AI is"]

    sampling_params = SamplingParams(
        max_tokens=10,
        temperature=1.0,
        logprobs=0,
    )

    hf_model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to(
        "cuda"
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(model_path)

    output = list(llm.generate(prompts, sampling_params=sampling_params))[0]

    trtllm_token_ids = output.outputs[0].token_ids
    trtllm_logprobs = torch.tensor(
        [list(lp.values())[0].logprob for lp in output.outputs[0].logprobs]
    )

    base_ids = hf_tokenizer.encode(prompts[0], return_tensors="pt").to("cuda")
    hf_logprobs = []

    for i, token_id in enumerate(trtllm_token_ids):
        if i > 0:
            prev_tokens = torch.tensor([trtllm_token_ids[:i]], device="cuda")
            input_ids = torch.cat([base_ids, prev_tokens], dim=1)
        else:
            input_ids = base_ids
        with torch.no_grad():
            logits = hf_model(input_ids).logits[0, -1, :]
        hf_logprobs.append(torch.log_softmax(logits, dim=-1)[token_id].item())

    hf_logprobs = torch.tensor(hf_logprobs)

    print(f"\nTensorRT-LLM logprobs: {trtllm_logprobs}")
    print(f"HuggingFace logprobs:  {hf_logprobs}")
    print(f"Diff: {(trtllm_logprobs - hf_logprobs).abs()}")

    torch.testing.assert_close(trtllm_logprobs, hf_logprobs, atol=0.15, rtol=0)


@pytest.mark.gpu2
def test_logprobs_pp2():
    """Test that logprobs count matches generated token count with PP=2.

    Regression test for https://github.com/NVIDIA/TensorRT-LLM/issues/12444
    Without the fix, logprobs length = 2N-1 instead of N due to duplication
    in the PP ring broadcast diff mechanism.
    """
    model_path = os.path.join(llm_models_root(), "llama-models-v2", "TinyLlama-1.1B-Chat-v1.0")
    max_tokens = 16
    llm = LLM(
        model=model_path,
        pipeline_parallel_size=2,
        max_batch_size=1,
        max_num_tokens=128,
        max_seq_len=256,
    )

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        logprobs=5,
    )

    output = list(llm.generate(["The future of the AI is"], sampling_params=sampling_params))[0]

    num_tokens = len(output.outputs[0].token_ids)
    num_logprobs = len(output.outputs[0].logprobs)
    assert num_logprobs == num_tokens, (
        f"logprobs length {num_logprobs} != generated tokens {num_tokens} "
        f"(expected 1:1 ratio, got {num_logprobs / num_tokens:.2f}:1)"
    )


class TestLogsprobsInBatchedSampling:
    """Validate logprobs handling in batched/mixed sampling.

    This is adapted from TestBatchedSampling. Direct testing of the sampler is needed, because request batching in tests
    like test_processed_logprobs_e2e is non-deterministic.
    """

    VOCAB_SIZE = 123

    @staticmethod
    def _build_test_cases(max_draft_len: int) -> list[tuple[list[SamplingParams], str]]:
        """Return test cases for testing logprobs handling in batched sampling.

        Each test case consists of a list of sampling parameters and a human-readable
        test case name.
        """

        include_beam_search: Final[bool] = (
            max_draft_len == 0
        )  # logprobs for beam search with drafting not supported

        LOGPROBS_CASES = [
            ("nologprobs", {}),
            ("raw_logprobs0", {"logprobs": 0}),
            ("raw_logprobs3", {"logprobs": 3}),
            ("proc_logprobs0", {"logprobs": 0, "logprobs_mode": LogprobMode.PROCESSED}),
            ("proc_logprobs3", {"logprobs": 3, "logprobs_mode": LogprobMode.PROCESSED}),
        ]
        BASE_CASES = {
            **{
                f"single-{label}": SamplingParams(temperature=0.7, top_k=2, **kwargs)
                for label, kwargs in LOGPROBS_CASES
                if (
                    # FIXME: fails with "Beam search does not support returning multiple logprobs per request"
                    not include_beam_search or kwargs.get("logprobs") != 3
                )
            },
            **{
                f"beamsearch-{label}": SamplingParams(use_beam_search=True, best_of=5, **kwargs)
                for label, kwargs in LOGPROBS_CASES
                if (
                    include_beam_search
                    and kwargs.get("logprobs") != 3  # top-k logprobs not supported for beam search
                    and "logprobs_mode"
                    not in kwargs  # processed logprobs not defined for beam search
                )
            },
        }

        test_cases = []

        # Base cases (single-request batches)
        for base_label, params in BASE_CASES.items():
            test_cases.append(
                (
                    [params],
                    f"single_{base_label}",
                )
            )

        rng = np.random.default_rng(seed=42)

        # Homogeneous batches (all requests use the same sampling params)
        max_batch_size: Final = 24
        for base_label, params in BASE_CASES.items():
            batch_size = rng.integers(low=2, high=max_batch_size)
            test_cases.append(
                (
                    [params] * batch_size,
                    f"uniform_batch_{base_label}",
                )
            )

        # To contain combinatorial explosion, combine up to 3 of the base cases in a batch. This
        # includes raw + processed + nologprobs as well as mixes of regular and beam-search requests.
        #
        # Raw/processed logprobs handling does not interfere with 1-beam vs. n-beam handling.
        # => Consider mixes of raw and processed only for "single" strategy.
        #    Those cases might be ordering sensitive.
        # => Consider mixes of beam-search and single-beam requests only for raw logprobs=0,3
        #    (top-k logprobs not supported for beam search), including ordering.
        #
        # What remains are mixes of beam-search requests with logprobs=None and raw_logprobs=0.
        #
        # Test cases further multiplied by:
        #      seq_slot assignments  ->  testing with one random sparse assignment only
        #      draft token presence  ->  drafting and beam search not a supported combination,
        #                                so testing only two cases (drafting without beam search and beam search)
        #      all sub-batches contiguous, any one of the 3- sub-batches contiguous, or all requests shuffled

        def _shuffle_mixed_batches(
            sub_batches: tuple[list[SamplingParams], ...], labels: tuple[str, ...]
        ) -> list[tuple[list[SamplingParams], str]]:
            res = []

            # all sub-batches contiguous
            ordered_batch = [params for sub_batch in sub_batches for params in sub_batch]
            joint_label = "_".join(labels)
            res.append((ordered_batch, joint_label))

            # permute all but one sub-batch
            for unperm_sub_batch_idx in range(len(sub_batches)):
                head = [
                    params
                    for sub_batch in sub_batches[:unperm_sub_batch_idx]
                    for params in sub_batch
                ]
                tail = [
                    params
                    for sub_batch in sub_batches[(unperm_sub_batch_idx + 1) :]
                    for params in sub_batch
                ]
                rng.shuffle(head)  # inplace
                rng.shuffle(tail)  # inplace
                partially_perm_batch = head + sub_batches[unperm_sub_batch_idx] + tail

                partially_perm_label = "partially_permuted_" + "_".join(
                    list(labels[:unperm_sub_batch_idx])
                    + [f"{labels[unperm_sub_batch_idx]}_unpermuted"]
                    + list(labels[(unperm_sub_batch_idx + 1) :])
                )
                res.append((partially_perm_batch, partially_perm_label))

            # shuffle entire batch
            shuffled_batch = ordered_batch.copy()
            rng.shuffle(shuffled_batch)  # inplace
            res.append((shuffled_batch, f"shuffled-{joint_label}"))

            return res

        def _combine_up_to_3_configs(
            labeled_configs: list[tuple[str, SamplingParams]],
            *,
            allow_single_only: bool = True,
        ) -> list[tuple[list[SamplingParams], str]]:
            res = []
            for cfg_1 in labeled_configs:
                label_1, params_1 = cfg_1
                sub_batch_size_1 = rng.integers(low=2, high=max_batch_size)
                sub_batch_1 = [params_1] * sub_batch_size_1
                is_single_1 = label_1.startswith("single")
                label_1 = f"{label_1}_x{sub_batch_size_1}"

                # NB: singular configs were already covered above
                for cfg_2 in labeled_configs:
                    if cfg_2 is cfg_1:
                        continue

                    label_2, params_2 = cfg_2
                    sub_batch_size_2 = rng.integers(low=2, high=max_batch_size)
                    sub_batch_2 = [params_2] * sub_batch_size_2
                    is_single_2 = label_2.startswith("single")
                    label_2 = f"{label_2}_x{sub_batch_size_2}"
                    if allow_single_only or (not is_single_1 or not is_single_2):
                        res += _shuffle_mixed_batches(
                            (sub_batch_1, sub_batch_2), (label_1, label_2)
                        )

                    for cfg_3 in labeled_configs:
                        if cfg_3 is cfg_2 or cfg_3 is cfg_1:
                            continue

                        label_3, params_3 = cfg_3
                        sub_batch_size_3 = rng.integers(low=2, high=max_batch_size)
                        sub_batch_3 = [params_3] * sub_batch_size_3
                        is_single_3 = label_3.startswith("single")
                        label_3 = f"{label_3}_x{sub_batch_size_3}"
                        if allow_single_only or (
                            not is_single_1 or not is_single_2 or not is_single_3
                        ):
                            res += _shuffle_mixed_batches(
                                (sub_batch_1, sub_batch_2, sub_batch_3),
                                (label_1, label_2, label_3),
                            )
            return res

        # All ordered combinations of up to three configs from the "single" family
        single_configs = [
            (key, value) for key, value in BASE_CASES.items() if key.startswith("single")
        ]
        assert single_configs
        test_cases += _combine_up_to_3_configs(single_configs)

        if include_beam_search:
            # All ordered combinations of up to three configs from the "beam_search" family
            beam_configs = [
                (key, value) for key, value in BASE_CASES.items() if key.startswith("beamsearch")
            ]
            assert beam_configs
            test_cases += _combine_up_to_3_configs(beam_configs)

            # All ordered combinations of up to three configs requesting raw logprobs
            raw_configs = [
                (key, value)
                for key, value in BASE_CASES.items()
                if value.logprobs is not None and value.logprobs_mode != LogprobMode.PROCESSED
            ]
            assert raw_configs
            test_cases += _combine_up_to_3_configs(raw_configs, allow_single_only=False)

        return test_cases

    @pytest.fixture(scope="function")
    def draft_lens(
        self,
        max_draft_len: int,
        sampling_params_list: list[SamplingParams],
    ) -> list[int]:
        """Generate per-request draft lengths.

        Currently drawn at random, every draft length is between 0
        and max_draft_len.
        """
        draft_len = list(
            np.random.default_rng(seed=42).integers(
                0,
                max_draft_len + 1,
                size=(
                    len(
                        sampling_params_list,
                    )
                ),
            )
        )
        return draft_len

    @pytest.fixture(scope="function")
    def total_seq_slots(self) -> int:
        return 1234

    @pytest.fixture(scope="function")
    def seq_slots(
        self, sampling_params_list: list[SamplingParams], total_seq_slots: int
    ) -> list[int]:
        # Returns list of seq slots associated with each request.
        #
        # Using a single sparse assignment with margin/padding to limit test complexity.
        # Update selectivity is independently validated by poisoning unrelated entries
        # of the relevant tensors in the 'sampler' fixture.
        rng = np.random.default_rng(seed=42)

        margin: Final = 12

        start = rng.integers(2, margin).item()
        end = total_seq_slots - rng.integers(2, margin).item()
        allowed_slots = np.arange(start, end)
        num_seq_slots = len(sampling_params_list)
        assert num_seq_slots <= 2 * allowed_slots.size  # want a sparse assignment
        seq_slots = list(rng.choice(allowed_slots, num_seq_slots, replace=False))

        return seq_slots

    @pytest.fixture(scope="function")
    def mock_requests(
        self,
        sampling_params_list: list[SamplingParams],
        seq_slots: list[int],
        draft_lens: list[int],
    ) -> ScheduledRequests:
        return self._build_mock_requests(
            sampling_params_list=sampling_params_list,
            seq_slots=seq_slots,
            draft_lens=draft_lens,
        )

    def _build_mock_requests(
        self,
        sampling_params_list: list[SamplingParams],
        *,
        seq_slots: list[int],
        draft_lens: list[int],
    ) -> ScheduledRequests:
        """Build a batch of test requests consumable by sample_async."""
        with torch.inference_mode(True):
            scheduled_requests = ScheduledRequests()
            # Logprobs handling does not depend on context vs. generation requests in general
            # NB: For beam search, finished context requests have different input and output beam widths.
            #     The tests could be extended to cover this case in the future.
            scheduled_requests.context_requests_chunking = []
            scheduled_requests.context_requests_last_chunk = []
            scheduled_requests.generation_requests = [
                LlmRequest(
                    request_id=seq_slot,
                    max_new_tokens=draft_len,
                    input_tokens=[],  # not used by tested code
                    sampling_config=SamplingConfig(sampling_params._get_sampling_config()),
                    seq_slot=seq_slot,
                    is_streaming=False,  # not relevant for tested code
                    draft_tokens=(  # 'len(.py_draft_tokens)' is inspected by get_draft_token_length
                        torch.testing.make_tensor(
                            (draft_len,),
                            dtype=torch.int32,
                            device="cpu",
                        ).tolist()
                        if draft_len
                        else None
                    ),
                    logprobs_mode=sampling_params.logprobs_mode,
                    num_logprobs=sampling_params.logprobs or 0,
                    return_log_probs=sampling_params.logprobs is not None,
                )
                for sampling_params, seq_slot, draft_len in zip(
                    sampling_params_list, seq_slots, draft_lens, strict=True
                )
            ]
            # Patch up request state to correctly infer input beam width (inspects req.is_context_init_state)
            for req in scheduled_requests.generation_requests:
                req.state = LlmRequestState.GENERATION_IN_PROGRESS
            return scheduled_requests

    @pytest.fixture(scope="function")
    def model_outputs(
        self,
        mock_requests: ScheduledRequests,
        vocab_size: int,
    ) -> Generator[dict[str, torch.Tensor], None, None]:
        """Provide a batch of random logits for use as input to sample_async.

        This fixture also validates that the logits are not altered by the UUT.
        """
        total_steps = sum(
            req.py_beam_width * (get_draft_token_length(req) + 1)
            for req in mock_requests.all_requests()
        )
        logits = torch.testing.make_tensor(
            (total_steps, vocab_size),
            dtype=torch.float32,
            device="cuda",
        )
        logits_orig = logits.clone()
        try:
            yield {
                # No 'd2t': Not relevant for tested functionality
                "logits": logits,
            }
        finally:
            torch.testing.assert_close(logits, logits_orig)

    @pytest.fixture(scope="function")
    def sampler(
        self,
        max_draft_len: int,
        total_seq_slots: int,
        seq_slots: list[int],
    ) -> Generator[TorchSampler, None, None]:
        sampler = TorchSampler(
            TorchSampler.Args(
                max_seq_len=127,
                max_draft_len=max_draft_len,
                max_num_sequences=total_seq_slots,
                max_beam_width=1 if max_draft_len else 5,
                max_total_draft_tokens=max_draft_len,
                disable_overlap_scheduler=True,  # this only affects bad-words handling
            )
        )
        assert sampler.store.log_probs_store is not None
        log_probs_store = sampler.store.log_probs_store
        logprobs_buffers = [
            log_probs_store.sampled_log_prob_indices,
            log_probs_store.sampled_log_probs,
            log_probs_store.sampled_log_prob_ranks,
            log_probs_store.topk_indices,
            log_probs_store.topk_vals,
        ]
        # poison buffers
        bystander_seq_slots = torch.arange(total_seq_slots)
        bystander_seq_slots = bystander_seq_slots[
            ~torch.isin(bystander_seq_slots, torch.tensor(seq_slots, dtype=torch.int32))
        ]
        for logprobs_buffer in logprobs_buffers:
            logprobs_buffer[bystander_seq_slots] = torch.testing.make_tensor(
                (bystander_seq_slots.size(0), *logprobs_buffer.shape[1:]),
                dtype=logprobs_buffer.dtype,
                device=logprobs_buffer.device,
            )
        # snapshot buffers
        logprobs_buffer_snapshots = [
            logprobs_buffer.clone() for logprobs_buffer in logprobs_buffers
        ]
        try:
            yield sampler
        finally:
            # validate buffers
            for buffer, snapshot in zip(logprobs_buffers, logprobs_buffer_snapshots, strict=True):
                assert torch.equal(buffer[bystander_seq_slots], snapshot[bystander_seq_slots])

    def _validate_logprobs(
        self,
        *,
        mock_requests: ScheduledRequests,
        model_outputs: dict[str, Any],
        sampler: TorchSampler,
    ):
        logits: torch.Tensor = model_outputs["logits"]
        logits_offset = 0
        for req in mock_requests.all_requests():
            num_logits = req.py_beam_width * (get_draft_token_length(req) + 1)

            if not req.py_return_log_probs:
                logits_offset += num_logits
                continue
            assert req.py_num_logprobs >= 0

            is_processed_logprobs = req.py_logprobs_mode == LogprobMode.PROCESSED

            # compute probs
            req_logits = logits[logits_offset : (logits_offset + num_logits)]
            if is_processed_logprobs:
                # NB: Test considers only temperature + top-k or beam search without temperature
                if req.sampling_config.temperature is not None:
                    req_logits = req_logits / req.sampling_config.temperature
            req_log_probs = req_logits.log_softmax(dim=-1)
            if is_processed_logprobs:
                # NB: Test considers only temperature + top-k or beam search without temperature
                if req.sampling_config.top_k is not None:
                    # Computing req_log_probs by renormalizing without masking. This avoids issues
                    # with non-deterministic tie-breaking in top-k.
                    req_log_probs += torch.topk(req_log_probs, k=req.sampling_config.top_k)[
                        0
                    ].log_softmax(dim=-1).amax(dim=-1, keepdim=True) - req_log_probs.amax(
                        dim=-1, keepdim=True
                    )

            sampled_tokens = req.get_tokens()
            assert req.py_result is not None
            assert req.py_result.log_probs is not None
            returned_log_probs = cast(list[list[dict]], req.py_result.log_probs)
            for beam_idx in range(req.py_beam_width):
                num_steps = get_draft_token_length(req) + 1
                returned_log_probs_beam = returned_log_probs[beam_idx]

                sampled_tokens_beam = sampled_tokens[beam_idx][-num_steps:]
                for step_idx, sampled_token in zip(range(num_steps), sampled_tokens_beam):
                    if req.py_beam_width > 1:
                        assert sampler.store.beam_search_store is not None
                        pred_beam_idx = cast(
                            int,
                            sampler.store.beam_search_store.predecessor_beams[
                                req.py_seq_slot, beam_idx
                            ].item(),
                        )
                    else:
                        pred_beam_idx = beam_idx
                    returned_log_probs_step = returned_log_probs_beam[step_idx]
                    returned_sampled_logprob = returned_log_probs_step[sampled_token]

                    req_logit_offset = step_idx * req.py_beam_width + pred_beam_idx

                    def validate_logprob_and_rank(token_id: int, returned_logprob: Logprob):
                        # Validate logprob
                        recomputed_logprob = req_log_probs[req_logit_offset, token_id].item()
                        torch.testing.assert_close(
                            returned_logprob.logprob,
                            recomputed_logprob,
                            rtol=0,
                            atol=(5e-6 if is_processed_logprobs else 1e-6),  # equals rtol for probs
                        )

                        # Validate sampled rank
                        if req.py_beam_width == 1:
                            # FIXME: "rank" is not behaving as expected for beam search requests.
                            #   There are two factors. First, "rank" is not clearly specified for beam search
                            #   (could be logprob rank within beam or across all beams) and therefore
                            #   'rank=1' is returned for all finished beams
                            #   via finalize_beam and convert_logprobs_tensor_to_list. Second,
                            #   during decoding (unfinished beam, in general this is the case in this test),
                            #   request logprobs are set via handle_logprobs and store_logprobs_list_to_request,
                            #   which inspects uninitialized elements of log_probs_store.sampled_log_prob_ranks.
                            min_rank = (
                                req_log_probs[req_logit_offset] > recomputed_logprob
                            ).sum().item() + 1
                            tie_fuzz = (
                                (req_log_probs[req_logit_offset] == recomputed_logprob).sum().item()
                            )
                            assert min_rank <= cast(int, returned_logprob.rank)
                            assert cast(int, returned_logprob.rank) <= min_rank + tie_fuzz

                    validate_logprob_and_rank(sampled_token, returned_sampled_logprob)

                    # Validate top-k logprobs
                    if req.py_num_logprobs > 0:
                        # validate result size and remove sampled-token if more than k logprobs returned
                        returned_topk_logprobs = returned_log_probs_step.copy()
                        if len(returned_log_probs_step) != req.py_num_logprobs:
                            assert len(returned_log_probs_step) == req.py_num_logprobs + 1
                            returned_topk_logprobs.pop(sampled_token)

                        # validate ranks and suppress masked tokens
                        req_top_k = None
                        if (req_top_ks := req.sampling_config.top_k) is not None:
                            req_top_k = req_top_ks[0]
                        returned_ranks = set()
                        for token, logprob in returned_topk_logprobs.items():
                            assert logprob.rank not in returned_ranks
                            returned_ranks.add(logprob.rank)
                            if (
                                is_processed_logprobs
                                and req_top_k is not None
                                and logprob.rank > req_top_k
                            ):
                                assert logprob.logprob == float("-inf")
                        assert returned_ranks == set(range(1, req.py_num_logprobs + 1))

                        # validate that remaining set is a top-k set
                        returned_topk_tokens = [
                            token
                            for token, logprob in returned_topk_logprobs.items()
                            if not is_processed_logprobs
                            or req_top_k is None
                            or logprob.rank <= req_top_k
                        ]
                        returned_topk_tokens_tensor = torch.tensor(
                            returned_topk_tokens, dtype=torch.int32
                        )
                        returned_topk_mask = torch.zeros((self.VOCAB_SIZE,), dtype=torch.bool)
                        returned_topk_mask[returned_topk_tokens_tensor] = True
                        topk_min = req_log_probs[req_logit_offset, returned_topk_mask].amin().item()
                        other_max = (
                            req_log_probs[req_logit_offset, ~returned_topk_mask].amax().item()
                        )
                        assert topk_min >= other_max

                        # validate logprobs (-inf already validated above)
                        for token in returned_topk_tokens:
                            validate_logprob_and_rank(token, returned_topk_logprobs[token])

            logits_offset += num_logits

    @pytest.mark.parametrize(
        (
            "max_draft_len",
            "sampling_params_list",
            "vocab_size",
        ),
        [
            pytest.param(
                max_draft_len,
                sampling_params_list,
                vocab_size,
                id=(f"draft_len=0..{max_draft_len}-{params_label}"),
            )
            # https://stackoverflow.com/a/75421799, does not work with nested loops
            for (
                max_draft_len,
                _build_test_cases,
                vocab_size,
            ) in product(
                [0, 3],
                [_build_test_cases],
                [VOCAB_SIZE],
            )
            for (sampling_params_list, params_label) in _build_test_cases(
                max_draft_len=max_draft_len
            )
        ],
    )
    def test_logprobs(
        self,
        max_draft_len: int,  # used by fixtures
        vocab_size: int,  # used by fixtures
        sampling_params_list: list[SamplingParams],  # used by fixtures
        mock_requests: ScheduledRequests,
        sampler: TorchSampler,
        model_outputs: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ):
        # Override some caching smartness in setup_sampler_step
        def _eager_collect_new_requests_for_setup(
            scheduled_requests: ScheduledRequests,
        ) -> list[LlmRequest]:
            return scheduled_requests.all_requests()

        monkeypatch.setattr(
            sampler, "_collect_new_requests_for_setup", _eager_collect_new_requests_for_setup
        )

        sampler.setup_sampler_step(mock_requests)

        sample_state = sampler.sample_async(
            mock_requests,
            model_outputs,
            num_context_logits_prefix_sum=[0],
        )

        sampler.update_requests(sample_state)

        self._validate_logprobs(
            mock_requests=mock_requests,
            model_outputs=model_outputs,
            sampler=sampler,
        )
