import pytest
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig
from tensorrt_llm.sampling_params import SamplingParams

MODEL_PATH = llm_models_root() / "DeepSeek-V3-Lite/bf16"


@pytest.mark.parametrize(
    "logprobs_mode",
    [
        "raw_logits",
        "raw_logprobs",
        "processed_logits",
        "processed_logprobs",
    ],
)
@pytest.mark.parametrize("temperature", [0.0, 0.8])
@pytest.mark.parametrize("top_k", [None, 50])
def test_logprobs_mode_basic(logprobs_mode, temperature, top_k):
    llm = LLM(
        MODEL_PATH,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )

    sampling_params = SamplingParams(
        max_tokens=5,
        logprobs=3,
        temperature=temperature,
        top_k=top_k,
        logprobs_mode=logprobs_mode,
    )

    prompts = ["The future of AI is"]
    outputs = llm.generate(prompts, sampling_params=sampling_params)

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

    # Validate based on mode
    if "logprobs" in logprobs_mode:
        for val in all_logprob_values:
            assert val <= 0.0, (
                f"Logprobs mode {logprobs_mode} should have non-positive values, got {val}"
            )

    if "logits" in logprobs_mode:
        has_positive = any(val > 0 for val in all_logprob_values)
        assert has_positive, f"Logits mode {logprobs_mode} should have some positive values"

    del llm


@pytest.mark.parametrize("temperature", [0.5, 1.0])
def test_raw_vs_processed_logprobs_difference(temperature):
    llm = LLM(
        MODEL_PATH,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.7),
    )

    prompt = ["The capital of France is"]

    # Get raw logprobs
    raw_params = SamplingParams(
        max_tokens=3,
        logprobs=5,
        temperature=temperature,
        top_k=20,
        logprobs_mode="raw_logprobs",
        seed=42,  # Fix seed for reproducibility
    )
    raw_outputs = llm.generate(prompt, sampling_params=raw_params)

    # Get processed logprobs
    processed_params = SamplingParams(
        max_tokens=3,
        logprobs=5,
        temperature=temperature,
        top_k=20,
        logprobs_mode="processed_logprobs",
        seed=42,  # Same seed
    )
    processed_outputs = llm.generate(prompt, sampling_params=processed_params)

    # Extract logprobs from first token
    raw_first_token_logprobs = raw_outputs[0].outputs[0].logprobs[0]
    processed_first_token_logprobs = processed_outputs[0].outputs[0].logprobs[0]

    # Get a common token ID
    common_token_ids = set(raw_first_token_logprobs.keys()) & set(
        processed_first_token_logprobs.keys()
    )
    assert len(common_token_ids) > 0, "Should have some common token IDs"

    token_id = list(common_token_ids)[0]
    raw_val = raw_first_token_logprobs[token_id].logprob
    processed_val = processed_first_token_logprobs[token_id].logprob

    if temperature != 1.0:
        assert raw_val != processed_val, (
            f"Raw and processed logprobs should differ with temperature={temperature}"
        )

    del llm


def test_logprobs_mode_with_greedy_sampling():
    llm = LLM(
        MODEL_PATH,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )

    prompt = ["Once upon a time"]

    for mode in ["raw_logprobs", "processed_logprobs", "raw_logits", "processed_logits"]:
        sampling_params = SamplingParams(
            max_tokens=4,
            logprobs=3,
            temperature=0.0,  # Greedy sampling
            logprobs_mode=mode,
        )

        outputs = llm.generate(prompt, sampling_params=sampling_params)

        assert len(outputs) == 1
        assert len(outputs[0].outputs[0].logprobs) > 0, (
            f"Mode {mode} should return logprobs even with greedy sampling"
        )

        # Check value ranges
        logprob_vals = [
            logprob_obj.logprob
            for token_logprobs in outputs[0].outputs[0].logprobs
            for logprob_obj in token_logprobs.values()
        ]

        if "logprobs" in mode:
            assert all(v <= 0.0 for v in logprob_vals), (
                f"Mode {mode} should have non-positive values"
            )

        if "logits" in mode:
            assert any(v > 0 for v in logprob_vals), f"Mode {mode} should have some positive values"

    del llm


def test_backward_compatibility():
    llm = LLM(
        MODEL_PATH,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )

    prompt = ["Hello world"]

    # Test with explicit raw_logprobs
    explicit_params = SamplingParams(
        max_tokens=3,
        logprobs=2,
        temperature=0.8,
        logprobs_mode="raw_logprobs",
        seed=123,
    )
    explicit_outputs = llm.generate(prompt, sampling_params=explicit_params)

    # Test with default (should be raw_logprobs)
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
        "Default mode should produce same results as explicit raw_logprobs"
    )

    del llm


def test_logprobs_mode_with_top_p():
    """Test that processed modes correctly capture the effect of top-p sampling."""
    llm = LLM(
        MODEL_PATH,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )

    prompt = ["The weather today is"]

    params = SamplingParams(
        max_tokens=2,
        logprobs=10,  # Request many logprobs to see the effect
        temperature=1.0,
        top_p=0.5,  # Restrict to top 50% probability mass
        logprobs_mode="processed_logits",
    )

    outputs = llm.generate(prompt, sampling_params=params)

    # Check that some logits are -inf (masked by top-p)
    first_token_logprobs = outputs[0].outputs[0].logprobs[0]
    logprob_values = [obj.logprob for obj in first_token_logprobs.values()]
    print(f"logprob_values: {logprob_values}")
    assert any(val == float("-inf") for val in logprob_values)

    del llm


@pytest.mark.parametrize("mode", ["raw_logprobs", "processed_logprobs"])
def test_prompt_logprobs_with_modes(mode):
    """Test that logprobs modes also work for prompt logprobs."""
    llm = LLM(
        MODEL_PATH,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
    )

    prompt = ["Hello world, how are you?"]

    params = SamplingParams(
        max_tokens=2,
        logprobs=3,
        prompt_logprobs=3,  # Request prompt logprobs
        logprobs_mode=mode,
        temperature=0.8,
    )

    outputs = llm.generate(prompt, sampling_params=params)

    # Check that prompt logprobs were returned
    prompt_logprobs = outputs[0].outputs[0].prompt_logprobs
    assert prompt_logprobs is not None
    assert len(prompt_logprobs) > 0

    # Validate values based on mode
    for token_logprobs in prompt_logprobs:
        if token_logprobs:  # Can be None for first token
            for logprob_obj in token_logprobs.values():
                if "logprobs" in mode:
                    assert logprob_obj.logprob <= 0.0, (
                        f"Prompt logprobs in mode {mode} should be non-positive"
                    )

    del llm


if __name__ == "__main__":
    # Run a quick smoke test
    print("Running smoke test for logprobs modes...")
    test_logprobs_mode_basic("raw_logprobs", 0.8, None)
    test_logprobs_mode_basic("processed_logprobs", 0.8, None)
    print("Smoke test passed!")
