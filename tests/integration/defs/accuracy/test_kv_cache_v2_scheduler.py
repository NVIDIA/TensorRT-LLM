# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Integration tests for KVCacheV2Scheduler.

Tests cover: basic correctness (V2 vs V1), token budget limits, chunked prefill,
eviction, LoRA/PEFT, MTP draft tokens, block reuse, overlap scheduler, and
large-model accuracy via GSM8K.
"""

from unittest import mock

import pytest

from tensorrt_llm import LLM
from tensorrt_llm.executor import request as executor_request
from tensorrt_llm.llmapi import (CudaGraphConfig, Eagle3DecodingConfig,
                                 KvCacheConfig, MoeConfig, MTPDecodingConfig,
                                 SamplingParams, SchedulerConfig)
from tensorrt_llm.lora_helper import LoraConfig

from ..conftest import (llm_models_root, skip_pre_hopper)
from .accuracy_core import GSM8K, LlmapiAccuracyTestHarness

# ---------------------------------------------------------------------------
# Shared prompts
# ---------------------------------------------------------------------------
SHORT_PROMPTS = [
    "What is the capital of France?",
    "Explain gravity in one sentence.",
    "List three prime numbers.",
    "What color is the sky on a clear day?",
    "Name one planet in our solar system.",
    "What is 2 + 2?",
    "Who wrote Romeo and Juliet?",
    "What is the boiling point of water?",
    "Name a mammal that can fly.",
    "What language is spoken in Brazil?",
]

MEDIUM_PROMPTS = [
    "Describe the process of photosynthesis in detail, including the light-dependent and light-independent reactions.",
    "Explain the theory of general relativity and its implications for our understanding of space and time.",
    "Discuss the major causes and consequences of the French Revolution in European history.",
    "Compare and contrast the economic systems of capitalism and socialism with real-world examples.",
    "Describe the structure and function of DNA, including how it replicates and how mutations occur.",
]

SHARED_PREFIX_PROMPTS = [
    "The following is a summary of a scientific paper about climate change. "
    "Please answer the question below.\nQuestion: What is the main finding?",
    "The following is a summary of a scientific paper about climate change. "
    "Please answer the question below.\nQuestion: What methodology was used?",
    "The following is a summary of a scientific paper about climate change. "
    "Please answer the question below.\nQuestion: What are the limitations?",
    "The following is a summary of a scientific paper about climate change. "
    "Please answer the question below.\nQuestion: How does this compare to prior work?",
    "The following is a summary of a scientific paper about climate change. "
    "Please answer the question below.\nQuestion: What future research is suggested?",
]

# Construct a long prompt (~500 tokens) by repeating text
_LONG_BLOCK = (
    "Artificial intelligence has transformed many industries. "
    "From healthcare to finance, AI systems are becoming increasingly capable. "
    "Machine learning models can now process vast amounts of data. "
    "Natural language processing enables computers to understand human language. "
    "Computer vision allows machines to interpret visual information. "
)
LONG_PROMPT = (_LONG_BLOCK * 12 +
               "\nBased on the above, summarize the key themes.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# V2 scheduler requires MAX_UTILIZATION policy
_V2_SCHEDULER_CONFIG = SchedulerConfig(
    capacity_scheduler_policy="MAX_UTILIZATION")


def _run_generate(llm, prompts, sampling_params=None):
    """Run generation and return list of CompletionOutput."""
    if sampling_params is None:
        sampling_params = SamplingParams(max_tokens=64, temperature=0.0)
    return llm.generate(prompts, sampling_params=sampling_params)


def _assert_all_completed(outputs, expected_count=None):
    """Assert all outputs have non-empty generated text."""
    if expected_count is not None:
        assert len(outputs) == expected_count
    for i, out in enumerate(outputs):
        assert len(out.outputs) > 0, f"Output {i} has no outputs"
        assert len(
            out.outputs[0].token_ids) > 0, f"Output {i} has empty token_ids"


# ===========================================================================
# IT1-IT7, IT13-IT15: Functional tests on Llama-3.2-1B
# ===========================================================================
class TestKVCacheV2Llama:
    """Functional tests for V2 scheduler using Llama-3.2-1B (1 GPU)."""

    MODEL_PATH = f"{llm_models_root()}/llama-3.2-models/Llama-3.2-1B"

    # IT1: V2 vs V1 produce identical greedy output
    def test_v2_vs_v1_basic(self):
        sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
        prompts = SHORT_PROMPTS[:5]

        kv_v1 = KvCacheConfig(use_kv_cache_manager_v2=False)
        with LLM(self.MODEL_PATH, kv_cache_config=kv_v1) as llm:
            outputs_v1 = llm.generate(prompts,
                                      sampling_params=sampling_params)

        kv_v2 = KvCacheConfig(use_kv_cache_manager_v2=True)
        with LLM(self.MODEL_PATH, kv_cache_config=kv_v2,
                 scheduler_config=_V2_SCHEDULER_CONFIG) as llm:
            outputs_v2 = llm.generate(prompts,
                                      sampling_params=sampling_params)

        for i, (o1, o2) in enumerate(zip(outputs_v1, outputs_v2)):
            assert o1.outputs[0].token_ids == o2.outputs[0].token_ids, (
                f"Prompt {i}: V1 and V2 outputs differ.\n"
                f"V1: {o1.outputs[0].token_ids}\n"
                f"V2: {o2.outputs[0].token_ids}")

    # IT2: Token budget limited
    def test_token_budget_limited(self):
        kv_config = KvCacheConfig(use_kv_cache_manager_v2=True)
        sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_config,
                 scheduler_config=_V2_SCHEDULER_CONFIG,
                 max_num_tokens=64) as llm:
            outputs = _run_generate(llm, SHORT_PROMPTS, sampling_params)
            _assert_all_completed(outputs, expected_count=10)

    # IT3: Chunked prefill with a long prompt
    def test_chunked_prefill(self):
        kv_config = KvCacheConfig(use_kv_cache_manager_v2=True)
        sampling_params = SamplingParams(max_tokens=64, temperature=0.0)
        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_config,
                 scheduler_config=_V2_SCHEDULER_CONFIG,
                 enable_chunked_prefill=True,
                 max_num_tokens=128) as llm:
            outputs = _run_generate(llm, [LONG_PROMPT], sampling_params)
            _assert_all_completed(outputs, expected_count=1)

    # IT4: Chunked prefill with multiple requests
    def test_chunked_prefill_multi_request(self):
        kv_config = KvCacheConfig(use_kv_cache_manager_v2=True)
        sampling_params = SamplingParams(max_tokens=64, temperature=0.0)
        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_config,
                 scheduler_config=_V2_SCHEDULER_CONFIG,
                 enable_chunked_prefill=True,
                 max_num_tokens=256) as llm:
            outputs = _run_generate(llm, MEDIUM_PROMPTS, sampling_params)
            _assert_all_completed(outputs, expected_count=5)

    # IT5: Eviction under tight memory (no CUDA graph to avoid warmup
    # needing more blocks than max_tokens allows).
    # max_tokens=512 with 10 concurrent requests × ~84 tokens forces eviction.
    @pytest.mark.timeout(300)
    def test_eviction(self):
        kv_config = KvCacheConfig(use_kv_cache_manager_v2=True,
                                  max_tokens=512)
        sampling_params = SamplingParams(max_tokens=64, temperature=0.0)
        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_config,
                 scheduler_config=_V2_SCHEDULER_CONFIG,
                 cuda_graph_config=None,
                 max_batch_size=4,
                 max_num_tokens=256) as llm:
            outputs = _run_generate(llm, SHORT_PROMPTS, sampling_params)
            _assert_all_completed(outputs, expected_count=10)

    # IT6: Extreme eviction pressure (no CUDA graph — tests pure eviction)
    # max_tokens=256 with 20 prompts × ~148 tokens each forces heavy eviction.
    @pytest.mark.timeout(300)
    def test_extreme_eviction(self):
        kv_config = KvCacheConfig(use_kv_cache_manager_v2=True,
                                  max_tokens=256)
        sampling_params = SamplingParams(max_tokens=128, temperature=0.0)
        prompts = SHORT_PROMPTS + SHORT_PROMPTS  # 20 prompts
        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_config,
                 scheduler_config=_V2_SCHEDULER_CONFIG,
                 cuda_graph_config=None,
                 max_num_tokens=128) as llm:
            outputs = _run_generate(llm, prompts, sampling_params)
            _assert_all_completed(outputs, expected_count=20)

    # IT7: Batch size limited
    def test_batch_size_limited(self):
        kv_config = KvCacheConfig(use_kv_cache_manager_v2=True)
        sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_config,
                 scheduler_config=_V2_SCHEDULER_CONFIG,
                 max_batch_size=2,
                 max_num_tokens=8192) as llm:
            outputs = _run_generate(llm, SHORT_PROMPTS, sampling_params)
            _assert_all_completed(outputs, expected_count=10)

    # IT13: Overlap scheduler produces same output as non-overlap
    def test_overlap_scheduler(self):
        sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
        prompts = SHORT_PROMPTS[:5]

        kv_config = KvCacheConfig(use_kv_cache_manager_v2=True)

        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_config,
                 scheduler_config=_V2_SCHEDULER_CONFIG,
                 disable_overlap_scheduler=True) as llm:
            outputs_no_overlap = llm.generate(prompts,
                                              sampling_params=sampling_params)

        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_config,
                 scheduler_config=_V2_SCHEDULER_CONFIG,
                 disable_overlap_scheduler=False) as llm:
            outputs_overlap = llm.generate(prompts,
                                           sampling_params=sampling_params)

        for i, (o1, o2) in enumerate(
                zip(outputs_no_overlap, outputs_overlap)):
            assert o1.outputs[0].token_ids == o2.outputs[0].token_ids, (
                f"Prompt {i}: overlap vs non-overlap outputs differ.\n"
                f"No-overlap: {o1.outputs[0].token_ids}\n"
                f"Overlap: {o2.outputs[0].token_ids}")

    # IT14: Block reuse with shared-prefix prompts
    def test_block_reuse(self):
        kv_config = KvCacheConfig(use_kv_cache_manager_v2=True,
                                  enable_block_reuse=True)
        sampling_params = SamplingParams(max_tokens=64, temperature=0.0)
        with LLM(self.MODEL_PATH, kv_cache_config=kv_config,
                 scheduler_config=_V2_SCHEDULER_CONFIG) as llm:
            outputs = _run_generate(llm, SHARED_PREFIX_PROMPTS,
                                    sampling_params)
            _assert_all_completed(outputs, expected_count=5)

    # IT15: Chunked prefill with eviction (no CUDA graph to avoid warmup
    # needing more blocks than max_tokens allows).
    # max_tokens=384 with 5 medium prompts (~100 tokens) + chunked prefill.
    @pytest.mark.timeout(300)
    def test_chunked_prefill_with_eviction(self):
        kv_config = KvCacheConfig(use_kv_cache_manager_v2=True,
                                  max_tokens=384)
        sampling_params = SamplingParams(max_tokens=64, temperature=0.0)
        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_config,
                 scheduler_config=_V2_SCHEDULER_CONFIG,
                 cuda_graph_config=None,
                 max_batch_size=4,
                 enable_chunked_prefill=True,
                 max_num_tokens=128) as llm:
            outputs = _run_generate(llm, MEDIUM_PROMPTS, sampling_params)
            _assert_all_completed(outputs, expected_count=5)


# ===========================================================================
# IT8-IT9: LoRA tests on llama-7b-hf
# ===========================================================================
@pytest.mark.skip_less_device_memory(40000)
class TestKVCacheV2LoRA:
    """LoRA tests for V2 scheduler using llama-7b-hf (1 GPU, >=40GB)."""

    MODEL_PATH = f"{llm_models_root()}/llama-models/llama-7b-hf"
    LORA_DIR = f"{llm_models_root()}/llama-models/luotuo-lora-7b-0.1"

    # IT8: Single LoRA adapter with V2
    def test_lora_v2(self):
        lora_config = LoraConfig(
            lora_dir=[self.LORA_DIR],
            max_lora_rank=8,
            max_loras=1,
            max_cpu_loras=1,
            lora_target_modules=["attn_q", "attn_k", "attn_v"],
        )
        kv_config = KvCacheConfig(use_kv_cache_manager_v2=True,
                                  free_gpu_memory_fraction=0.4)
        sampling_params = SamplingParams(max_tokens=32, temperature=0.0)

        prompts_with_lora = SHORT_PROMPTS[:3]

        # Generate with LoRA
        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_config,
                 scheduler_config=_V2_SCHEDULER_CONFIG,
                 lora_config=lora_config) as llm:
            lora_request = executor_request.LoRARequest("lora-0", 0,
                                                        self.LORA_DIR)
            outputs_lora = llm.generate(prompts_with_lora,
                                        sampling_params=sampling_params,
                                        lora_request=lora_request)
            _assert_all_completed(outputs_lora, expected_count=3)

        # Generate without LoRA (baseline)
        kv_config_base = KvCacheConfig(use_kv_cache_manager_v2=True,
                                       free_gpu_memory_fraction=0.4)
        with LLM(self.MODEL_PATH, kv_cache_config=kv_config_base,
                 scheduler_config=_V2_SCHEDULER_CONFIG) as llm:
            outputs_base = llm.generate(prompts_with_lora,
                                        sampling_params=sampling_params)
            _assert_all_completed(outputs_base, expected_count=3)

        # Verify LoRA changes at least one output
        any_different = False
        for o_lora, o_base in zip(outputs_lora, outputs_base):
            if o_lora.outputs[0].token_ids != o_base.outputs[0].token_ids:
                any_different = True
                break
        assert any_different, "LoRA adapter did not change any output"

    # IT9: Multiple LoRA adapters with max_loras=1 (adapter swapping)
    def test_lora_multi_adapter_v2(self):
        lora_config = LoraConfig(
            lora_dir=[self.LORA_DIR],
            max_lora_rank=8,
            max_loras=1,
            max_cpu_loras=1,
            lora_target_modules=["attn_q", "attn_k", "attn_v"],
        )
        kv_config = KvCacheConfig(use_kv_cache_manager_v2=True,
                                  free_gpu_memory_fraction=0.4)
        sampling_params = SamplingParams(max_tokens=32, temperature=0.0)

        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_config,
                 scheduler_config=_V2_SCHEDULER_CONFIG,
                 lora_config=lora_config) as llm:
            # First request with LoRA
            lora_request = executor_request.LoRARequest("lora-0", 0,
                                                        self.LORA_DIR)
            outputs1 = llm.generate(SHORT_PROMPTS[:2],
                                    sampling_params=sampling_params,
                                    lora_request=lora_request)
            _assert_all_completed(outputs1, expected_count=2)

            # Second request without LoRA (adapter swap)
            outputs2 = llm.generate(SHORT_PROMPTS[2:4],
                                    sampling_params=sampling_params)
            _assert_all_completed(outputs2, expected_count=2)


# ===========================================================================
# IT10-IT11: MTP draft tokens on DeepSeek-V3-Lite
# ===========================================================================
@skip_pre_hopper
@pytest.mark.skip_less_device_memory(60000)
class TestKVCacheV2DSv3Lite:
    """MTP speculative decoding tests with V2 scheduler on DeepSeek-V3-Lite."""

    MODEL_PATH = f"{llm_models_root()}/DeepSeek-V3-Lite/bf16"

    # IT10: MTP draft tokens without chunked prefill
    def test_mtp_draft_tokens(self):
        kv_config = KvCacheConfig(use_kv_cache_manager_v2=True,
                                  free_gpu_memory_fraction=0.75)
        mtp_config = MTPDecodingConfig(num_nextn_predict_layers=2)
        sampling_params = SamplingParams(max_tokens=64, temperature=0.0)
        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_config,
                 scheduler_config=_V2_SCHEDULER_CONFIG,
                 speculative_config=mtp_config,
                 max_num_tokens=8192) as llm:
            outputs = _run_generate(llm, SHORT_PROMPTS[:5], sampling_params)
            _assert_all_completed(outputs, expected_count=5)

    # IT11: MTP draft tokens with chunked prefill
    def test_mtp_chunked_draft_tokens(self):
        kv_config = KvCacheConfig(use_kv_cache_manager_v2=True,
                                  free_gpu_memory_fraction=0.75)
        mtp_config = MTPDecodingConfig(num_nextn_predict_layers=2)
        sampling_params = SamplingParams(max_tokens=64, temperature=0.0)
        with LLM(self.MODEL_PATH,
                 kv_cache_config=kv_config,
                 scheduler_config=_V2_SCHEDULER_CONFIG,
                 speculative_config=mtp_config,
                 enable_chunked_prefill=True,
                 max_num_tokens=256) as llm:
            outputs = _run_generate(llm, [LONG_PROMPT], sampling_params)
            _assert_all_completed(outputs, expected_count=1)


# ===========================================================================
# IT16-IT17: Accuracy tests on GPT-OSS-120B (4 GPUs)
# ===========================================================================
@skip_pre_hopper
@pytest.mark.skip_less_device_memory(80000)
@pytest.mark.skip_less_device(4)
class TestKVCacheV2GPTOSS(LlmapiAccuracyTestHarness):
    """Accuracy tests for V2 scheduler on GPT-OSS-120B (4 GPUs)."""

    MODEL_PATH = f"{llm_models_root()}/gpt_oss/gpt-oss-120b"

    extra_evaluator_kwargs = {
        "fewshot_as_multiturn": True,
        "apply_chat_template": True,
    }

    # IT16: GSM8K accuracy with V2
    @mock.patch.dict(GSM8K.EVALUATE_KWARGS,
                     {"scores_filter": "exact_match,flexible-extract"})
    @mock.patch.object(GSM8K, "MAX_OUTPUT_LEN", 8192)
    def test_accuracy_v2(self):
        kv_config = KvCacheConfig(free_gpu_memory_fraction=0.7,
                                  use_kv_cache_manager_v2=True)
        pytorch_config = dict(
            disable_overlap_scheduler=False,
            cuda_graph_config=CudaGraphConfig(),
            moe_config=MoeConfig(backend="CUTLASS"),
        )
        with LLM(self.MODEL_PATH,
                 tensor_parallel_size=4,
                 kv_cache_config=kv_config,
                 scheduler_config=_V2_SCHEDULER_CONFIG,
                 max_batch_size=720,
                 **pytorch_config) as llm:
            model_name = "GPT-OSS/120B-MXFP4"
            task = GSM8K(model_name)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.extra_evaluator_kwargs)

    # IT17: Eagle3 speculative decoding with V2
    @mock.patch.dict(GSM8K.EVALUATE_KWARGS,
                     {"scores_filter": "exact_match,flexible-extract"})
    @mock.patch.object(GSM8K, "MAX_OUTPUT_LEN", 8192)
    def test_eagle3_v2(self):

        kv_config = KvCacheConfig(free_gpu_memory_fraction=0.4,
                                  enable_block_reuse=False,
                                  use_kv_cache_manager_v2=True)
        pytorch_config = dict(
            disable_overlap_scheduler=False,
            cuda_graph_config=CudaGraphConfig(),
            moe_config=MoeConfig(backend="CUTLASS"),
        )
        eagle_model_dir = f"{llm_models_root()}/gpt_oss/gpt-oss-120b-Eagle3"
        spec_config = Eagle3DecodingConfig(
            max_draft_len=3,
            speculative_model=eagle_model_dir,
            eagle3_one_model=True,
        )
        with LLM(self.MODEL_PATH,
                 tensor_parallel_size=4,
                 kv_cache_config=kv_config,
                 scheduler_config=_V2_SCHEDULER_CONFIG,
                 max_batch_size=720,
                 speculative_config=spec_config,
                 **pytorch_config) as llm:
            model_name = "GPT-OSS/120B-MXFP4"
            task = GSM8K(model_name)
            task.evaluate(llm,
                          extra_evaluator_kwargs=self.extra_evaluator_kwargs)
