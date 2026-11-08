import os

import pytest
from utils.llm_data import llm_models_root
from utils.util import force_ampere, similar

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.pyexecutor.llm_request import (LlmRequest,
                                                        SamplingConfig)
from tensorrt_llm.llmapi.llm_utils import KvCacheConfig


@pytest.fixture(scope="module")
def input_prompts():
    return [
        "Born in north-east France, Soyer trained as a",
        "The future of AI is",
    ]


@pytest.fixture(scope="module")
def expected_outputs():
    return {
        "Born in north-east France, Soyer trained as a": [
            "lawyer and was a member of the French Resistance",
            "cook before turning to painting."
        ],
        "The future of AI is": [
            "all about human-machine collaboration.",
            "more promising than you think."
        ],
    }


@pytest.fixture(scope="module")
def llm():
    return LLM(model=os.path.join(llm_models_root(), "llama-models-v2",
                                  "TinyLlama-1.1B-Chat-v1.0"),
               kv_cache_config=KvCacheConfig(max_tokens=1000),
               max_batch_size=8,
               max_seq_len=64,
               disable_overlap_scheduler=True)


@force_ampere  # Save H100 resource
@pytest.mark.parametrize("n", [1, 2, 3])
def test_create_child_request(n: int):
    sampling_config = SamplingConfig()
    setattr(sampling_config, 'top_p', [0.9])
    setattr(sampling_config, 'num_return_sequences', n)

    parent = LlmRequest(
        request_id=1,
        max_new_tokens=10,
        input_tokens=[1, 2, 3],
        sampling_config=sampling_config,
        is_streaming=False,
        client_id=50,
        return_log_probs=True,
        return_context_logits=True,
    )

    for child_id in range(
            parent.request_id + 1,
            parent.request_id + parent.sampling_config.num_return_sequences):
        parent.create_child_request(child_id)

    assert len(parent.child_requests
               ) == parent.sampling_config.num_return_sequences - 1

    for ind, child in enumerate(parent.child_requests):
        assert child.request_id == ind + parent.request_id + 1
        assert child.py_request_id == child.request_id
        assert child.parent_request_id == parent.request_id

        assert child.py_client_id == 50
        assert child.py_max_new_tokens == 10

        assert child.py_return_log_probs == parent.py_return_log_probs
        assert child.py_return_context_logits == parent.py_return_context_logits

        assert child.py_batch_idx is None

        # Verify parent - child independence
        assert child.py_result is not None
        assert child.py_result is not parent.py_result
        assert child.get_tokens() == parent.get_tokens()
        assert child.get_tokens() is not parent.get_tokens()

        assert child.child_requests == []


@force_ampere  # Save H100 resource
@pytest.mark.parametrize("n", [2])
@pytest.mark.parametrize("best_of", [None, 3])
@pytest.mark.threadleak(enabled=False)
def test_n_outputs(n: int, best_of: int, llm, input_prompts, expected_outputs):
    sampling_params = SamplingParams(
        n=n,
        best_of=best_of,
        temperature=0.8,  # ensure different outputs
        top_p=0.95,  # ensure different outputs
        use_beam_search=False)
    for output_idx, output in enumerate(
            llm.generate(input_prompts, sampling_params=sampling_params)):

        assert len(output.outputs) == n

        for idx, sequence in enumerate(output.outputs):
            if n == best_of:
                assert similar(sequence.text,
                               expected_outputs[input_prompts[output_idx]][idx])


@pytest.mark.parametrize("n", [3])
@pytest.mark.threadleak(enabled=False)
def test_async_n_outputs(n: int, llm, input_prompts):
    sampling_params = SamplingParams(
        n=n,
        temperature=0.8,  # ensure different outputs
        top_p=0.95,  # ensure different outputs
        use_beam_search=False)

    # Asynchronously submit many requests to exceed max batch size.
    futures = []
    for _ in range(5):
        for prompt in input_prompts:
            future = llm.generate_async(prompt, sampling_params)
            futures.append(future)

    # Expect no error raised and each result contains n outputs.
    for _, future in enumerate(futures):
        request_output = future.result()
        assert len(request_output.outputs) == n
