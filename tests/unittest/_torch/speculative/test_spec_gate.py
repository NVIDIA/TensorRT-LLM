import os
import sys
import unittest

import pytest
import torch
from utils.llm_data import llm_models_root
from utils.util import similar

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.speculative.speculation_gate import SpeculationGate
from tensorrt_llm.llmapi import (CudaGraphConfig, EagleDecodingConfig,
                                 KvCacheConfig)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


# It tests the end-to-end functionality of the SpeculationGate,
# which will turn off spec decode when the average acceptance length is below the threshold.
# It is set with acceptance window and acceptance threshold in spec_config.
# This test set the max_concurrency to a large value to prevent spec decode turned off due to number of effective requests > max_concurrency,
# So that we can only focus on the turning off effect from the SpeculationGate.
@pytest.mark.high_cuda_memory
def test_spec_gate_e2e():
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 35:
        pytest.skip("Not enough memory to load target + draft model")
    models_path = llm_models_root()
    eagle_model_dir = f"{models_path}/EAGLE3-LLaMA3.1-Instruct-8B"
    target_model_dir = f"{models_path}/llama-3.1-model/Llama-3.1-8B-Instruct"

    max_batch_size = 2
    max_draft_len = 4
    kv_cache_config = KvCacheConfig(enable_block_reuse=True, max_tokens=8192)
    cuda_graph_config = CudaGraphConfig(batch_sizes=[1])

    llm_common_config = dict(
        model=target_model_dir,
        attn_backend="TRTLLM",
        disable_overlap_scheduler=True,
        cuda_graph_config=cuda_graph_config,
        max_batch_size=max_batch_size,
        kv_cache_config=kv_cache_config,
        max_seq_len=4096,
    )

    spec_config = EagleDecodingConfig(
        max_draft_len=max_draft_len,
        speculative_model_dir=eagle_model_dir,
        # Llama 3 does not support one model eagle.
        eagle3_one_model=False,
        max_concurrency=10000,
        acceptance_window=5,
        acceptance_length_threshold=0.6,
    )

    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)
    # Output tests
    prompts = [
        "The capital of France is",
        "The president of the United States is",
        "What is the capital of Australia?",
        "Explain in one sentence why the sky is blue.",
        "Who wrote the book 'Pride and Prejudice'?",
        "List three U.S. national holidays in the year 2025.",
        "What is the currency of Japan?",
        "How many players are on a basketball court for one team?",
        "List three primary colors.",
    ]
    sampling_params = SamplingParams(max_tokens=32, temperature=0)

    results_spec = llm_spec.generate(prompts, sampling_params)
    generated_text_spec = [result.outputs[0].text for result in results_spec]
    llm_spec.shutdown()

    llm_ref = LLM(**llm_common_config)
    results_ref = llm_ref.generate(prompts, sampling_params)
    generated_text_ref = [result.outputs[0].text for result in results_ref]
    llm_ref.shutdown()

    for text_spec, text_ref in zip(generated_text_spec, generated_text_ref):
        assert similar(text_spec, text_ref)


def test_returns_none_until_window_and_enabled_when_above_threshold():
    gate = SpeculationGate(window=3, threshold=0.5)

    disabled, avg = gate.record_avg_decoded(2.0, request_id=1)
    assert disabled is False and avg is None
    assert gate.disabled is False

    disabled, avg = gate.record_avg_decoded(2.0, request_id=2)
    assert disabled is False and avg is None
    assert gate.disabled is False

    disabled, avg = gate.record_avg_decoded(2.0, request_id=3)
    assert disabled is False
    assert avg == pytest.approx(1.0, rel=1e-6)
    assert gate.disabled is False


def test_disables_when_avg_below_threshold_and_stays_disabled():
    gate = SpeculationGate(window=3, threshold=0.7)

    gate.record_avg_decoded(1.1)
    gate.record_avg_decoded(1.2)

    disabled, avg = gate.record_avg_decoded(1.3)
    assert disabled is True
    assert avg == pytest.approx(0.2, rel=1e-6)
    assert gate.disabled is True

    # Once disabled, subsequent calls do nothing and return (False, None)
    disabled, avg = gate.record_avg_decoded(100.0)
    assert disabled is False and avg is None
    assert gate.disabled is True

    disabled, avg = gate.record_avg_decoded(200.0)
    assert disabled is False and avg is None
    assert gate.disabled is True


def test_rolling_window_and_disable_on_drop():
    gate = SpeculationGate(window=3, threshold=0.8)

    # First three high-acceptance requests keep it enabled
    gate.record_avg_decoded(2.0)
    gate.record_avg_decoded(2.0)
    disabled, avg = gate.record_avg_decoded(2.0)
    assert disabled is False
    assert avg == pytest.approx(1.0, rel=1e-6)
    assert gate.disabled is False

    # Fourth lower value enters window -> average drops below threshold -> disable
    disabled, avg = gate.record_avg_decoded(1.2)
    assert disabled is True
    assert avg == pytest.approx((1.0 + 1.0 + 0.2) / 3.0, rel=1e-6)
    assert gate.disabled is True


if __name__ == "__main__":
    unittest.main()
