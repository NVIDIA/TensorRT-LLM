import os
import sys
import unittest
from unittest.mock import patch

import pytest
import torch
from utils.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch.speculative.speculation_gate import SpeculationGate
from tensorrt_llm.llmapi import (CudaGraphConfig, Eagle3DecodingConfig,
                                 KvCacheConfig)
from tensorrt_llm.logger import logger

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


@pytest.fixture(scope="function")
def enforce_single_worker(monkeypatch):
    """Mock functions don't work with multiple processes, so we enforce single worker."""
    monkeypatch.setenv("TLLM_WORKER_USE_SINGLE_PROCESS", "1")
    yield


# Tests that the SpeculationGate correctly disables speculative decoding
# when the average acceptance rate drops below the threshold.
# This test uses a mock to simulate low acceptance rates and verifies
# that the spec gate triggers and disables speculation.
@pytest.mark.high_cuda_memory
def test_spec_gate_e2e(enforce_single_worker):
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if total_mem_gb < 35:
        pytest.skip("Not enough memory to load target + draft model")
    models_path = llm_models_root()
    eagle_model_dir = f"{models_path}/EAGLE3-LLaMA3.1-Instruct-8B"
    target_model_dir = f"{models_path}/llama-3.1-model/Llama-3.1-8B-Instruct"

    max_batch_size = 2
    max_draft_len = 4
    acceptance_window = 3
    acceptance_threshold = 0.6
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

    spec_config = Eagle3DecodingConfig(
        max_draft_len=max_draft_len,
        speculative_model=eagle_model_dir,
        eagle3_one_model=False,
        max_concurrency=10000,
        acceptance_window=acceptance_window,
        acceptance_length_threshold=acceptance_threshold,
    )

    prompts = [
        "The capital of France is",
        "The president of the United States is",
        "What is the capital of Australia?",
    ]
    sampling_params = SamplingParams(max_tokens=20, temperature=0)

    # Track calls to record_avg_decoded and the disabled state
    gate_state = {"record_calls": [], "gate_disabled": False}

    original_record_avg_decoded = SpeculationGate.record_avg_decoded

    def mock_record_avg_decoded(self,
                                avg_decoded_tokens_per_iter,
                                request_id=None):
        """
        Mock that simulates low acceptance rate (1.2 tokens/iter = 0.2 accepted).
        This is below the threshold of 0.6, so the gate should trigger after the window fills.
        """
        # Simulate low acceptance: avg_decoded = 1.2 means accepted_len = 0.2
        # This is below threshold (0.6), so gate should trigger
        simulated_low_avg = 1.2
        disabled_now, avg = original_record_avg_decoded(self, simulated_low_avg,
                                                        request_id)

        gate_state["record_calls"].append({
            "original_avg": avg_decoded_tokens_per_iter,
            "simulated_avg": simulated_low_avg,
            "disabled_now": disabled_now,
            "avg_accept": avg,
            "request_id": request_id,
        })
        if disabled_now:
            gate_state["gate_disabled"] = True

        return disabled_now, avg

    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)

    try:
        with patch.object(SpeculationGate, 'record_avg_decoded',
                          mock_record_avg_decoded):
            llm_spec.generate(prompts, sampling_params)

        # Verify the mock was called (requests completed)
        assert len(gate_state["record_calls"]
                   ) > 0, "record_avg_decoded should have been called"

        # Verify the gate was disabled after enough requests with low acceptance
        assert gate_state["gate_disabled"], \
            f"Gate should have been disabled with simulated low acceptance. Calls: {gate_state['record_calls']}"

        # Verify the gate triggered at the right time (after window is filled)
        # The gate should trigger on the `acceptance_window`-th call (index = window - 1)
        disable_indices = [
            i for i, call in enumerate(gate_state["record_calls"])
            if call["disabled_now"]
        ]
        assert len(disable_indices) == 1, \
            f"Gate should have triggered exactly once, but triggered at indices: {disable_indices}"
        assert disable_indices[0] >= acceptance_window - 1, \
            f"Gate should trigger after window ({acceptance_window}) is filled, but triggered at index {disable_indices[0]}"

        # Verify the average acceptance was below threshold when disabled
        disable_call = gate_state["record_calls"][disable_indices[0]]
        assert disable_call["avg_accept"] is not None
        assert disable_call["avg_accept"] < acceptance_threshold, \
            f"Avg acceptance ({disable_call['avg_accept']}) should be below threshold ({acceptance_threshold})"

        logger.debug(
            f"Gate correctly triggered after {disable_indices[0] + 1} requests")
        logger.debug(
            f"Final avg acceptance: {disable_call['avg_accept']:.3f} < threshold {acceptance_threshold}"
        )
    finally:
        llm_spec.shutdown()


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
