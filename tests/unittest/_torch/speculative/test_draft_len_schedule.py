import os
import sys

import pytest
import torch

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.llmapi import (DraftTargetDecodingConfig, KvCacheConfig,
                                 NGramDecodingConfig)

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.llm_data import llm_models_root
from utils.util import similar


# # ============================================================================
# # Fixture: Force single-worker mode for all tests in this module
# # ============================================================================
@pytest.fixture(scope="module", autouse=True)
def enforce_single_worker():
    """Force single-worker mode for all tests in this module."""
    import os
    os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"] = "1"
    yield
    if "TLLM_WORKER_USE_SINGLE_PROCESS" in os.environ:
        del os.environ["TLLM_WORKER_USE_SINGLE_PROCESS"]


# # ============================================================================
# # test 1:  Generation correctness check
# # ============================================================================
@pytest.mark.parametrize("drafter_type,schedule", [
    ("ngram", {
        1: 3,
        4: 2,
        8: 1
    }),
    ("model_drafter", {
        1: 3,
        4: 2,
        8: 1
    }),
])
@pytest.mark.high_cuda_memory
@pytest.mark.no_xdist
def test_correctness_across_batch_sizes(drafter_type: str, schedule: dict):
    """
    Test output correctness with various schedules and batch sizes.

    This is the primary correctness test that validates:
    - Multiple different schedules work correctly
    - Output with draft_len_schedule matches output with fixed draft_len
    - Works across different batch size transitions
    - Both NGram and ModelDrafter function correctly
    """
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    memory_required = 30 if drafter_type == "model_drafter" else 20
    if total_mem_gb < memory_required:
        pytest.skip(
            f"Not enough memory (need {memory_required}GB, have {total_mem_gb:.1f}GB)"
        )

    models_path = llm_models_root()
    target_model = f"{models_path}/llama-3.1-model/Llama-3.1-8B-Instruct"
    draft_model = f"{models_path}/llama-3.2-models/Llama-3.2-3B-Instruct"

    max_batch_size = 4
    max_draft_len = max(schedule.values())  # Use max from schedule

    kv_cache_config = KvCacheConfig(enable_block_reuse=False,
                                    enable_partial_reuse=False,
                                    max_tokens=8192)

    llm_common_config = dict(
        model=target_model,
        backend='pytorch',
        attn_backend="TRTLLM",
        disable_overlap_scheduler=True,
        max_batch_size=max_batch_size,
        kv_cache_config=kv_cache_config,
        max_num_tokens=2048,
    )

    if drafter_type == "ngram":
        spec_config = NGramDecodingConfig(
            max_draft_len=max_draft_len,
            max_matching_ngram_size=2,
            draft_len_schedule=schedule,
            is_keep_all=True,
            is_use_oldest=True,
            is_public_pool=False,
        )
    else:
        spec_config = DraftTargetDecodingConfig(
            max_draft_len=max_draft_len,
            speculative_model_dir=str(draft_model),
            draft_len_schedule=schedule,
        )

    prompts = [
        "The capital of France is",
        "The president of the United States is",
        "Machine learning is",
        "The future of AI",
        "What is the capital of Australia?",
        "Explain in one sentence why the sky is blue.",
        "Who wrote the book 'Pride and Prejudice'?",
        "List three U.S. national holidays in the year 2025.",
    ]

    # Give each request different max_tokens so they finish at different times
    # This creates batch size transitions to test draft_len_schedule
    # Use deterministic sampling settings to maximize similarity with non-spec baseline
    sampling_params_list = [
        SamplingParams(
            max_tokens=i + 3,
            temperature=0,
            seed=42,
            ignore_eos=True,  # Prevent early stopping differences
            top_k=1,
            top_p=1.0,
        ) for i in range(len(prompts))
    ]

    # With dynamic draft_len_schedule
    llm_with_schedule = LLM(**llm_common_config, speculative_config=spec_config)
    results_with_schedule = llm_with_schedule.generate(prompts,
                                                       sampling_params_list)
    generated_text_with_schedule = [
        result.outputs[0].text for result in results_with_schedule
    ]
    llm_with_schedule.shutdown()

    # Reference: spec decode with fixed max_draft_len (no schedule)
    if drafter_type == "ngram":
        spec_config_fixed = NGramDecodingConfig(
            max_draft_len=max_draft_len,
            max_matching_ngram_size=2,
            draft_len_schedule=None,  # No schedule - fixed draft length
            is_keep_all=True,
            is_use_oldest=True,
            is_public_pool=False,
        )
    else:
        spec_config_fixed = DraftTargetDecodingConfig(
            max_draft_len=max_draft_len,
            speculative_model_dir=str(draft_model),
            draft_len_schedule=None,  # No schedule - fixed draft length
        )

    llm_fixed = LLM(**llm_common_config, speculative_config=spec_config_fixed)
    results_fixed = llm_fixed.generate(prompts, sampling_params_list)
    generated_text_fixed = [result.outputs[0].text for result in results_fixed]
    llm_fixed.shutdown()

    # Verify correctness: spec decode with schedule should match spec decode without schedule
    for text_schedule, text_fixed in zip(generated_text_with_schedule,
                                         generated_text_fixed):
        assert similar(text_schedule, text_fixed), \
            f"{drafter_type} output with draft_len_schedule should match output with fixed draft_len. Got:\n" \
            f"With schedule: {text_schedule}\n" \
            f"Fixed:         {text_fixed}"


# # ============================================================================
# # test 2:  Drafting side functionality check
# # ============================================================================
@pytest.mark.parametrize("drafter_type,spec_config_factory", [
    ("ngram", lambda: NGramDecodingConfig(
        max_draft_len=5,
        max_matching_ngram_size=2,
        draft_len_schedule={
            1: 5,
            4: 4,
            5: 3,
            6: 2,
            7: 1,
            8: 0
        },
    )),
    ("model_drafter", lambda: DraftTargetDecodingConfig(
        max_draft_len=5,
        speculative_model_dir=str(llm_models_root() / "llama-3.2-models" /
                                  "Llama-3.2-3B-Instruct"),
        draft_len_schedule={
            1: 5,
            4: 4,
            5: 3,
            6: 2,
            7: 1,
            8: 0
        },
    )),
])
@pytest.mark.high_cuda_memory
@pytest.mark.no_xdist
def test_draft_len_schedule_functionality(drafter_type: str,
                                          spec_config_factory):
    """
    Test that draft_len=0 in schedule properly disables speculation.

    Verifies:
    - When schedule maps to draft_len=0, speculation is disabled
    - System falls back to normal generation
    - Output is still correct
    - max_draft_tokens is set to 0 when batch_size triggers draft_len=0
    - Works for both NGram and ModelDrafter
    """

    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if drafter_type == "model_drafter" and total_mem_gb < 30:
        pytest.skip("Not enough memory for 2-model setup")
    elif total_mem_gb < 20:
        pytest.skip("Not enough memory")
    max_batch_size = 8

    llm_common_config = dict(
        model=llm_models_root() / "llama-3.1-model" / "Meta-Llama-3.1-8B",
        backend='pytorch',
        attn_backend="TRTLLM",
        disable_overlap_scheduler=True,
        max_batch_size=max_batch_size,
        max_num_tokens=2048,
    )
    spec_config = spec_config_factory()
    prompts = [f"Prompt {i}: The answer is" for i in range(8)]
    # Give each request different max_tokens so they finish at different times
    # This creates batch size transitions: 8 -> 7 -> 6 -> 5 -> 4 -> 3 -> 2 -> 1
    sampling_params_list = [
        SamplingParams(
            max_tokens=20 * (i + 1),
            temperature=0,
            seed=42,
            ignore_eos=True,  # Prevent early stopping
            top_k=1,
            top_p=1.0,
        ) for i in range(8)
    ]
    llm_spec = LLM(**llm_common_config, speculative_config=spec_config)

    drafter = llm_spec._executor.engine.drafter
    executor = llm_spec._executor.engine

    iteration_data = []

    # Store original methods
    original_update_max_draft_tokens = drafter.update_max_draft_tokens
    original_prepare_draft = drafter.prepare_draft_tokens
    original_should_use_spec_decode = drafter.should_use_spec_decode

    # 1. Mock should_use_spec_decode to always return True
    # This isolates draft_len_schedule testing from max_concurrency logic
    def mock_should_use_spec_decode(*args, **kwargs):
        return True  # Always allow speculation (draft_len_schedule controls it)

    drafter.should_use_spec_decode = mock_should_use_spec_decode

    # 2. Instrument update_max_draft_tokens to capture when draft_len changes
    def instrumented_update_max_draft_tokens(new_max_draft_tokens: int):
        batch_size_active = len(executor.active_requests)

        original_update_max_draft_tokens(new_max_draft_tokens)

        iteration_data.append({
            'batch_size_active': batch_size_active,
            'drafter_max_draft_tokens': new_max_draft_tokens,
            'use_spec_decode':
            None,  # Will be filled after _prepare_and_schedule_batch completes
            'actual_draft_lens':
            [],  # Will be filled after prepare_draft_tokens
        })

    drafter.update_max_draft_tokens = instrumented_update_max_draft_tokens

    # 3. Instrument prepare_draft_tokens - where actual draft tokens are produced
    def instrumented_prepare_draft(scheduled_batch, resource_manager):
        result = original_prepare_draft(scheduled_batch, resource_manager)

        if iteration_data and len(iteration_data) > 0:
            iteration_data[-1]['use_spec_decode'] = executor.use_spec_decode

            actual_draft_lens = []
            for req in scheduled_batch.generation_requests:
                draft_len = len(
                    req.py_draft_tokens) if req.py_draft_tokens else 0
                actual_draft_lens.append(draft_len)

            iteration_data[-1]['actual_draft_lens'] = actual_draft_lens

            # Filter out context-phase iterations (no generation requests = no draft tokens)
            # This happens when all requests are still in prefill/context phase
            if len(scheduled_batch.generation_requests) == 0:
                iteration_data.pop()

        return result

    drafter.prepare_draft_tokens = instrumented_prepare_draft

    try:
        llm_spec.generate(prompts, sampling_params_list)
    finally:
        # Restore methods in finally block to ensure cleanup even if generate() fails
        drafter.update_max_draft_tokens = original_update_max_draft_tokens
        drafter.prepare_draft_tokens = original_prepare_draft
        drafter.should_use_spec_decode = original_should_use_spec_decode
        llm_spec.shutdown()

    # ========================================================================
    # Verification Rule 1: batch_size_active → drafter_max_draft_tokens mapping
    # ========================================================================
    # Hardcoded expected mapping (floor lookup):
    # This matches what the authentic code does: len(executor.active_requests)
    expected_mapping = {
        1: 5,
        2: 5,
        3: 5,
        4: 4,
        5: 3,
        6: 2,
        7: 1,
        8: 0,  # >= 8
    }

    for idx, it in enumerate(iteration_data):
        bs = it['batch_size_active']
        drafter_tokens = it['drafter_max_draft_tokens']

        expected = expected_mapping.get(bs, 0)
        assert drafter_tokens == expected, \
            f"Iter {idx}: batch_size_gen={bs} → expected {expected} tokens, got {drafter_tokens}"

        if drafter_tokens == 0:
            assert not it['use_spec_decode'], \
                f"Iter {idx}: drafter_max_draft_tokens=0 but use_spec_decode={it['use_spec_decode']}"

    # ========================================================================
    # Verification Rule 2: actual_draft_lens (req.py_draft_tokens) vs drafter_max_draft_tokens
    # ========================================================================
    if drafter_type == "ngram":
        # NGram: all actual_draft_lens <= drafter_max_draft_tokens (because ngram drafting length not necessary to be the same as drafter_max_draft_tokens)
        for idx, it in enumerate(iteration_data):
            drafter_tokens = it['drafter_max_draft_tokens']
            for req_idx, actual_len in enumerate(it['actual_draft_lens']):
                assert actual_len <= drafter_tokens, \
                    f"Iter {idx}, req {req_idx}: NGram produced {actual_len} > max {drafter_tokens}"

    elif drafter_type == "model_drafter":
        # ModelDrafter: the drafter should produce full draft_len all the time
        for idx, it in enumerate(iteration_data):
            drafter_tokens = it['drafter_max_draft_tokens']
            actual_lens = it['actual_draft_lens']

            if drafter_tokens > 0:  # Only count when speculation is active (draft_len > 0)
                for req_idx, actual_len in enumerate(actual_lens):
                    assert actual_len == drafter_tokens, \
                        f"Iter {idx}, req {req_idx}: ModelDrafter produced {actual_len} != max_draft_tokens {drafter_tokens}"
