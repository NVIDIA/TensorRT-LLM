import pytest
from utils.llm_data import llm_models_root
from utils.util import similar

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import CudaGraphConfig
from tensorrt_llm.llmapi import KvCacheConfig as TRT_KvCacheConfig


@pytest.fixture(scope="module")
def model_path():
    return llm_models_root() / "llama-models-v2/TinyLlama-1.1B-Chat-v1.0"


def _create_llm_base(model_dir, enable_trtllm_sampler):
    """Base LLM creation with configurable sampler."""
    sampler_type = "TRTLLMSampler" if enable_trtllm_sampler else "TorchSampler"

    trt_kv_cache_config = TRT_KvCacheConfig(enable_block_reuse=False)

    return LLM(
        model=str(model_dir),
        tensor_parallel_size=1,
        trust_remote_code=True,
        enable_chunked_prefill=True,
        cuda_graph_config=CudaGraphConfig(),
        sampler_type=sampler_type,
        kv_cache_config=trt_kv_cache_config,
        max_num_tokens=
        128  # Only one request longer than max_num_tokens is required to test chunked prefill
    )


def create_llm(model_dir):
    """Create LLM with specific overlap scheduler setting"""
    return _create_llm_base(model_dir, enable_trtllm_sampler=True)


def create_llm_with_torch_sampler(model_dir):
    """Create LLM with TorchSampler."""
    return _create_llm_base(model_dir, enable_trtllm_sampler=False)


@pytest.mark.high_cuda_memory
def test_trtllm_sampler(model_path):
    prompts = [
        "Magellan and Elcano lead the first",
        "The capital of France is",
        "The capital of Bolivia is",
    ]

    expected_outputs = [["circumnavigation of the world"], ["Paris"],
                        ["La Paz"]]

    # Test configuration
    max_new_tokens = 10
    temperature = 1.0
    top_p = None
    stop_words = ["."]

    sampling_config = SamplingParams(max_tokens=max_new_tokens,
                                     n=1,
                                     stop=stop_words,
                                     temperature=temperature,
                                     top_p=top_p)

    # Test with overlap scheduler disabled
    llm = create_llm(model_path)
    outputs = llm.generate(prompts,
                           sampling_params=sampling_config,
                           use_tqdm=True)
    texts = [[completion.text for completion in request_output.outputs]
             for request_output in outputs]
    llm.shutdown()

    # Remove any text after \n\n, consider texts is a list of list of strings
    texts = [[text.split('\n\n')[0] for text in request_output]
             for request_output in texts]

    # Verify outputs are consistent
    for text, expected in zip(texts, expected_outputs):
        assert similar(text, expected), f"text: {text}, expected: {expected}"


@pytest.mark.high_cuda_memory
def test_trtllm_sampler_with_stop_token_ids(model_path):
    """Test sampler with stop_token_ids (fast path optimization)."""

    llm = create_llm_with_torch_sampler(model_path)
    tokenizer = llm.tokenizer

    prompt = "The capital of France is"
    target_sentence = "The capital of France is Paris"

    prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
    target_tokens = tokenizer.encode(target_sentence, add_special_tokens=False)

    # Use the first token after the prompt as the stop token
    assert len(target_tokens) > len(
        prompt_tokens), "Target must be longer than prompt"
    stop_token_id = target_tokens[len(prompt_tokens)]

    sampling_config = SamplingParams(max_tokens=100,
                                     n=1,
                                     stop_token_ids=[stop_token_id],
                                     temperature=0.0)

    outputs = llm.generate([prompt], sampling_params=sampling_config)
    text = outputs[0].outputs[0].text

    output_tokens = tokenizer.encode(text, add_special_tokens=False)

    llm.shutdown()
    assert stop_token_id not in output_tokens, f"Output should not contain stop token {stop_token_id}"
    assert len(output_tokens
               ) < 10, "Should stop very early with first-token stop_token_id"


@pytest.mark.high_cuda_memory
def test_torch_sampler_with_multi_token_stop_words(model_path):
    """Test TorchSampler with multi-token stop words (slow path)."""

    llm = create_llm_with_torch_sampler(model_path)
    tokenizer = llm.tokenizer

    prompt = "The capital of France is"

    # Use a string that will tokenize to multiple tokens
    stop_string = "\n\n"
    stop_tokens = tokenizer.encode(stop_string, add_special_tokens=False)

    assert len(
        stop_tokens
    ) > 1, f"Stop string should be multi-token, got {len(stop_tokens)} tokens"

    sampling_config = SamplingParams(
        max_tokens=100,
        n=1,
        stop=[stop_string],  # Use 'stop' parameter for multi-token
        temperature=0.0)

    outputs = llm.generate([prompt], sampling_params=sampling_config)
    text = outputs[0].outputs[0].text

    llm.shutdown()

    assert len(text) > 0, "Should generate some text"
    assert stop_string not in text, f"Stop string '{repr(stop_string)}' should not appear in the output"


@pytest.mark.high_cuda_memory
def test_trtllm_sampler_best_of_with_logprobs(model_path):
    """Test TRTLLMSampler with best_of > n and logprobs."""

    llm = create_llm(model_path)

    prompt = "The capital of France is"

    sampling_config = SamplingParams(
        max_tokens=10,
        temperature=1.0,
        top_k=2,
        n=2,  # Return 2 sequences
        best_of=3,  # Generate 3 candidates, pick best 2
        logprobs=1  # Return log probabilities
    )

    outputs = llm.generate([prompt], sampling_params=sampling_config)

    llm.shutdown()

    assert len(outputs) == 1, "Should return one request output"

    request_output = outputs[0]
    completion_outputs = request_output.outputs

    assert len(
        completion_outputs
    ) == 2, f"Expected 2 outputs (n=2), got {len(completion_outputs)}"

    for i, output in enumerate(completion_outputs):
        assert len(output.text) > 0, f"Output {i} should have generated text"

        assert output.finish_reason is not None, \
            f"Output {i} must have a finish_reason"

        assert output.cumulative_logprob is not None, \
            f"Output {i} should have cumulative_logprob when logprobs is requested"
        assert isinstance(output.cumulative_logprob, (float, int)), \
            f"Output {i} cumulative_logprob should be a number, got {type(output.cumulative_logprob)}"

        assert output.logprobs is not None, \
            f"Output {i} should have logprobs when logprobs=1"
        assert len(output.logprobs) == len(output.token_ids), \
            f"Output {i} should have logprobs for each token"

    if len(completion_outputs) >= 2:
        assert completion_outputs[0].cumulative_logprob >= completion_outputs[1].cumulative_logprob, \
            "Outputs should be sorted by cumulative log probability (best first)"
