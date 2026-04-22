import os
import sys
import unittest
from unittest.mock import Mock, patch

import pytest
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import (CudaGraphConfig, Eagle3DecodingConfig,
                                 KvCacheConfig)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(scope="function")
def enforce_single_worker(monkeypatch):
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    yield


def test_dynamic_draft_len(enforce_single_worker):

    def mock_get_draft_len_for_batch_size(draft_len_schedule, batch_size,
                                          max_total_draft_tokens):
        # The draft length for each iter will be 4-4-2-2-0-0-2-2-4-4-2-2-0-0-2-2-...
        # which tested:
        # (1) decrease draft length: 4->2,
        # (2) increase draft length: 2->4,
        # (3) disable and re-enable drafting: 2->0->2.
        if mock_get_draft_len_for_batch_size.call_count % 8 < 2:
            dynamic_draft_len = 4
        elif mock_get_draft_len_for_batch_size.call_count % 8 < 4:
            dynamic_draft_len = 2
        elif mock_get_draft_len_for_batch_size.call_count % 8 < 6:
            dynamic_draft_len = 0
        else:
            dynamic_draft_len = 2
        return dynamic_draft_len

    # Create a Mock object with the mock function as side_effect
    mock_get_draft_len_for_batch_size = Mock(
        side_effect=mock_get_draft_len_for_batch_size)
    # Reset mock state before using it
    mock_get_draft_len_for_batch_size.reset_mock()
    mock_get_draft_len_for_batch_size.call_count = 0

    pytorch_config = dict(
        disable_overlap_scheduler=True,
        cuda_graph_config=CudaGraphConfig(),
        max_batch_size=1,
    )
    kv_cache_config = KvCacheConfig(
        enable_block_reuse=True,
        free_gpu_memory_fraction=0.6,
    )
    eagle_model_dir = f"{llm_models_root()}/Qwen3/qwen3_8b_eagle3"
    target_model_dir = f"{llm_models_root()}/Qwen3/Qwen3-8B"

    spec_config = Eagle3DecodingConfig(
        max_draft_len=4,
        speculative_model=eagle_model_dir,
        eagle3_one_model=True,
        draft_len_schedule={
            1: 4
        },  # It doesn't matter which value is used here, as the draft length will be controlled by the mock function.
    )
    llm_common_config = dict(
        model=target_model_dir,
        **pytorch_config,
        kv_cache_config=kv_cache_config,
        enable_chunked_prefill=False,
        max_num_tokens=8192,
    )
    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)

    prompts = [
        "The president of the United States is",
    ]
    sampling_params = SamplingParams(max_tokens=50, temperature=0)
    with patch(
            'tensorrt_llm._torch.speculative.utils.get_draft_len_for_batch_size',
            mock_get_draft_len_for_batch_size):
        results_spec = llm_spec.generate(prompts, sampling_params)
        generated_text_spec = [
            result.outputs[0].text for result in results_spec
        ]
    llm_spec.shutdown()

    llm_ref = LLM(**llm_common_config)
    results_ref = llm_ref.generate(prompts, sampling_params)
    generated_text_ref = [result.outputs[0].text for result in results_ref]
    llm_ref.shutdown()

    for text_spec, text_ref in zip(generated_text_spec, generated_text_ref):
        assert text_spec == text_ref


if __name__ == "__main__":
    unittest.main()
