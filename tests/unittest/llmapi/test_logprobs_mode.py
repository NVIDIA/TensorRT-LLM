import pytest
from utils.llm_data import llm_models_root

import tensorrt_llm
from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.sampling_params import SamplingParams

print(f"tensorrt_llm.__file__: {tensorrt_llm.__file__}")
# /home/dominicw/.local/lib/python3.12/site-packages/tensorrt_llm

MODEL_PATH = llm_models_root() / "DeepSeek-V3-Lite/bf16"


@pytest.mark.parametrize("temperature", [0.0, 0.8])
@pytest.mark.parametrize("top_k", [None, 50])
def test_logprobs_mode_basic(temperature, top_k):
    llm = LLM(
        MODEL_PATH,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.7),
    )

    sampling_params = SamplingParams(
        max_tokens=10,
        logprobs=3,
        temperature=temperature,
        top_k=top_k,
        logprobs_mode="processed_logprobs",
        return_context_logits=True,
        return_generation_logits=True,
        seed=42,
    )

    prompts = ["The future of AI is"]
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    print(f"outputs: {outputs}")

    assert len(outputs) == 1
    output = outputs[0]
    assert len(output.outputs) == 1
    completion = output.outputs[0]

    # Check that logprobs were returned
    assert completion.logprobs is not None
    assert len(completion.logprobs) > 0

    # Collect all logprob values
    all_logprob_values = []
    for token_logprobs in completion.logprobs:
        for token_id, logprob_obj in token_logprobs.items():
            all_logprob_values.append(logprob_obj.logprob)

    print(f"all_logprob_values: {all_logprob_values}")
    # Validate that processed_logprobs returns non-positive values (log probabilities)
    for val in all_logprob_values:
        assert val <= 0.0, f"processed_logprobs mode should have non-positive values, got {val}"

    del llm


@pytest.mark.parametrize("temperature", [0.5, 1.0, 1.5])
def test_processed_logprobs_with_temperature(temperature):
    llm = LLM(
        MODEL_PATH,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.7),
    )

    prompt = ["The capital of France is"]

    # Get processed logprobs (after temperature/top-k/top-p modifications)
    params = SamplingParams(
        max_tokens=3,
        logprobs=5,
        temperature=temperature,
        top_k=20,
        logprobs_mode="processed_logprobs",
        seed=42,
    )
    outputs = llm.generate(prompt, sampling_params=params)

    # Extract logprobs from first token
    first_token_logprobs = outputs[0].outputs[0].logprobs[0]
    assert len(first_token_logprobs) > 0, "Should have logprobs returned"

    # Validate that all values are non-positive (log probabilities)
    for token_id, logprob_obj in first_token_logprobs.items():
        assert logprob_obj.logprob <= 0.0, (
            f"processed_logprobs should have non-positive values, got {logprob_obj.logprob}"
        )

    del llm


def test_logprobs_mode_with_greedy_sampling():
    llm = LLM(
        MODEL_PATH,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )

    prompt = ["Once upon a time"]

    sampling_params = SamplingParams(
        max_tokens=4,
        logprobs=3,
        temperature=0.0,  # Greedy sampling
        logprobs_mode="processed_logprobs",
    )

    outputs = llm.generate(prompt, sampling_params=sampling_params)

    assert len(outputs) == 1
    assert len(outputs[0].outputs[0].logprobs) > 0, (
        "processed_logprobs should return logprobs even with greedy sampling"
    )

    # Check value ranges - all should be non-positive (log probabilities)
    logprob_vals = [
        logprob_obj.logprob
        for token_logprobs in outputs[0].outputs[0].logprobs
        for logprob_obj in token_logprobs.values()
    ]

    assert all(v <= 0.0 for v in logprob_vals), "processed_logprobs should have non-positive values"

    del llm


def test_backward_compatibility():
    llm = LLM(
        MODEL_PATH,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )

    prompt = ["Hello world"]

    # Test with explicit processed_logprobs
    explicit_params = SamplingParams(
        max_tokens=3,
        logprobs=2,
        temperature=0.8,
        logprobs_mode="processed_logprobs",
        seed=123,
    )
    explicit_outputs = llm.generate(prompt, sampling_params=explicit_params)

    # Test with default (should be processed_logprobs)
    default_params = SamplingParams(
        max_tokens=3,
        logprobs=2,
        temperature=0.8,
        seed=123,
    )
    default_outputs = llm.generate(prompt, sampling_params=default_params)

    # Results should be identical (same sampled tokens, same logprobs)
    explicit_tokens = explicit_outputs[0].outputs[0].token_ids
    default_tokens = default_outputs[0].outputs[0].token_ids

    assert explicit_tokens == default_tokens, (
        "Default mode should produce same results as explicit processed_logprobs"
    )

    del llm


def test_logprobs_mode_with_top_p():
    llm = LLM(
        MODEL_PATH,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )

    prompt = ["The weather today is"]

    # Test processed_logprobs mode (should have -inf for masked tokens after log_softmax)
    params = SamplingParams(
        max_tokens=2,
        logprobs=10,  # Request many logprobs to see the effect
        temperature=1.0,
        top_p=0.5,  # Restrict to top 50% probability mass
        logprobs_mode="processed_logprobs",
    )

    outputs = llm.generate(prompt, sampling_params=params)

    # Check that some logprobs are -inf (masked by top-p)
    first_token_logprobs = outputs[0].outputs[0].logprobs[0]
    logprob_values = [obj.logprob for obj in first_token_logprobs.values()]
    print(f"processed_logprobs values: {logprob_values}")
    assert any(val == float("-inf") for val in logprob_values), (
        "processed_logprobs should have -inf values for tokens masked by top-p"
    )
    # All non-inf values should be non-positive (log probabilities)
    non_inf_values = [v for v in logprob_values if v != float("-inf")]
    if non_inf_values:
        assert all(v <= 0.0 for v in non_inf_values), (
            "processed_logprobs non-inf values should be non-positive"
        )

    del llm


def test_prompt_logprobs_with_processed_logprobs():
    llm = LLM(
        MODEL_PATH,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )

    prompt = ["Hello world, how are you?"]

    params = SamplingParams(
        max_tokens=2,
        logprobs=3,
        prompt_logprobs=3,  # Request prompt logprobs
        logprobs_mode="processed_logprobs",
        temperature=0.8,
    )

    outputs = llm.generate(prompt, sampling_params=params)

    # Check that prompt logprobs were returned
    prompt_logprobs = outputs[0].outputs[0].prompt_logprobs
    assert prompt_logprobs is not None
    assert len(prompt_logprobs) > 0

    # Validate values - processed_logprobs should be non-positive
    for token_logprobs in prompt_logprobs:
        if token_logprobs:  # Can be None for first token
            for logprob_obj in token_logprobs.values():
                assert logprob_obj.logprob <= 0.0, (
                    "Prompt logprobs in processed_logprobs mode should be non-positive"
                )

    del llm


def test_processed_logprobs_with_top_k():
    llm = LLM(
        MODEL_PATH,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )

    prompt = ["The future of technology"]

    # Test with small top_k to ensure filtering is applied
    params = SamplingParams(
        max_tokens=2,
        logprobs=20,  # Request more logprobs than top_k to see filtering
        temperature=1.0,
        top_k=5,  # Only keep top 5 tokens
        logprobs_mode="processed_logprobs",
    )

    outputs = llm.generate(prompt, sampling_params=params)

    # Check that we have logprobs returned
    first_token_logprobs = outputs[0].outputs[0].logprobs[0]
    assert len(first_token_logprobs) > 0, "Should have logprobs returned"

    # With top_k=5, we should get at most 5 non-inf logprobs (plus potentially the sampled token)
    logprob_values = [obj.logprob for obj in first_token_logprobs.values()]
    non_inf_count = sum(1 for v in logprob_values if v != float("-inf"))

    # Should have at most top_k + 1 (top_k + sampled token if not in top_k)
    assert non_inf_count <= 6, (
        f"With top_k=5, should have at most 6 non-inf logprobs, got {non_inf_count}"
    )

    # All values should be non-positive (log probabilities)
    non_inf_values = [v for v in logprob_values if v != float("-inf")]
    if non_inf_values:
        assert all(v <= 0.0 for v in non_inf_values), (
            "processed_logprobs values should be non-positive"
        )

    del llm


if __name__ == "__main__":
    # Run a quick smoke test
    print("Running test for processed_logprobs mode...")
    test_logprobs_mode_basic(0.8, None)
    print("logprobs mode test passed!")
