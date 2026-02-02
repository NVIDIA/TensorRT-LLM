import os
import sys
import unittest
from unittest.mock import Mock, patch

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm.llmapi import (CudaGraphConfig, Eagle3DecodingConfig,
                                 KvCacheConfig)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(scope="function")
def enforce_single_worker(monkeypatch):
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    yield


@pytest.mark.skip("https://nvbugs/5758449")
@pytest.mark.parametrize("disable_overlap_scheduler", [True, False])
@pytest.mark.high_cuda_memory
def test_dynamic_spec_decode(enforce_single_worker,
                             disable_overlap_scheduler: bool):
    # mock_should_use_spec_decode doesn't work with multiple processes,
    # so we use the enforce_single_worker fixture to set the environment variable.
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 35:
        pytest.skip("Not enough memory to load target + draft model")

    models_path = llm_models_root()
    eagle_model_dir = f"{models_path}/EAGLE3-LLaMA3.1-Instruct-8B"
    target_model_dir = f"{models_path}/llama-3.1-model/Llama-3.1-8B-Instruct"

    max_batch_size = 1
    max_draft_len = 4
    kv_cache_config = KvCacheConfig(enable_block_reuse=True, max_tokens=8192)
    cuda_graph_config = CudaGraphConfig(batch_sizes=[1])

    llm_common_config = dict(
        model=target_model_dir,
        attn_backend="TRTLLM",
        disable_overlap_scheduler=disable_overlap_scheduler,
        cuda_graph_config=cuda_graph_config,
        max_batch_size=max_batch_size,
        kv_cache_config=kv_cache_config,
        # This max_seq_len is larger than the one specified
        # in the llama 3 8B eagle's config. We want to make sure
        # that the draft model won't go above its max in warmup
        # in this test.
        max_seq_len=8192,
    )

    spec_config = Eagle3DecodingConfig(
        max_draft_len=max_draft_len,
        speculative_model=eagle_model_dir,
        # Llama 3 does not support one model eagle.
        eagle3_one_model=False,
    )

    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)
    sampling_params = SamplingParams(max_tokens=128, temperature=0)

    # Output tests
    prompts = [
        "The president of the United States is",
    ]
    sampling_params = SamplingParams(max_tokens=20, temperature=0)

    # Mock should_use_spec_decode to turn on/off spec decode dynamically.
    def mock_should_use_spec_decode(requests, max_batch_size, max_num_tokens,
                                    max_total_draft_tokens):
        for req in requests:
            if req.state != LlmRequestState.GENERATION_IN_PROGRESS:
                continue

            mock_should_use_spec_decode.call_count += 1
            # Turn spec decoding on/off alternately.
            # When call_count % 4 is 0 or 1, spec decoding is on.
            # When call_count % 4 is 2 or 3, spec decoding is off.
            # By doing this, we can cover all the cases of dynamic spec decoding.
            # 1. Using spec decoding in iteration i, then using spec decoding in iteration i+1.
            # 2. Using spec decoding in iteration i, then not using spec decoding in iteration i+1.
            # 3. Not using spec decoding in iteration i, then using spec decoding in iteration i+1.
            # 4. Not using spec decoding in iteration i, then not using spec decoding in iteration i+1.
            if mock_should_use_spec_decode.call_count % 4 < 2:
                return True
            else:
                return False
        return True

    # Create a Mock object with the mock function as side_effect
    mock_should_use_spec_decode = Mock(side_effect=mock_should_use_spec_decode)
    # Reset mock state before using it
    mock_should_use_spec_decode.reset_mock()
    mock_should_use_spec_decode.call_count = 0

    with patch(
            'tensorrt_llm._torch.speculative.model_drafter.ModelDrafter.should_use_spec_decode',
            mock_should_use_spec_decode):
        results_spec = llm_spec.generate(prompts, sampling_params)
    generated_text_spec = [result.outputs[0].text for result in results_spec]
    llm_spec.shutdown()

    llm_ref = LLM(**llm_common_config)
    results_ref = llm_ref.generate(prompts, sampling_params)
    generated_text_ref = [result.outputs[0].text for result in results_ref]
    llm_ref.shutdown()

    for text_spec, text_ref in zip(generated_text_spec, generated_text_ref):
        # The spec decode algorithm currently guarantees identical results
        assert text_spec == text_ref


def test_should_use_spec_decode():
    from tensorrt_llm._torch.speculative.drafter import Drafter

    class _DummyDrafter(Drafter):

        def prepare_draft_tokens(self,
                                 scheduled_requests,
                                 resource_manager=None) -> None:
            return

    drafter = _DummyDrafter(max_concurrency=6)

    # Compare min(len(requests), max_batch_size, token_cap) with max_concurrency

    # Small active_requests ON case: num_effective_requests = min(5, 8, very_large) = 5 <= 6 → True
    active_requests = [object()] * 5
    assert drafter.should_use_spec_decode(active_requests,
                                          max_batch_size=8,
                                          max_num_tokens=4096 * 8,
                                          max_total_draft_tokens=4)

    # Small batch size ON case: num_effective_requests = min(12, 5, very_large) = 5 <= 6 → True
    active_requests = [object()] * 12
    assert drafter.should_use_spec_decode(active_requests,
                                          max_batch_size=5,
                                          max_num_tokens=4096 * 8,
                                          max_total_draft_tokens=4)

    # Small token budget ON case: token_cap = 28 // (1+4) = 5 → min(12, 8, 5) = 5 <= 6 → True
    active_requests = [object()] * 12
    assert drafter.should_use_spec_decode(active_requests,
                                          max_batch_size=8,
                                          max_num_tokens=28,
                                          max_total_draft_tokens=4)

    # Generic OFF case: num_effective_requests = min(12, 8, very_large) = 8 > 6 → False
    active_requests = [object()] * 12
    assert not drafter.should_use_spec_decode(active_requests,
                                              max_batch_size=8,
                                              max_num_tokens=4096 * 8,
                                              max_total_draft_tokens=4)

    # Edge case - None active requests OFF case
    active_requests = []
    assert not drafter.should_use_spec_decode(active_requests,
                                              max_batch_size=8,
                                              max_num_tokens=4096 * 8,
                                              max_total_draft_tokens=4)

    # Edge case - Token cap equals 0 OFF case: token_cap = 4 // (1+4) = 0 → min(12, 8, 0) = 0 <= 6 → False
    active_requests = [object()] * 12
    assert not drafter.should_use_spec_decode(active_requests,
                                              max_batch_size=8,
                                              max_num_tokens=4,
                                              max_total_draft_tokens=4)


if __name__ == "__main__":
    unittest.main()
