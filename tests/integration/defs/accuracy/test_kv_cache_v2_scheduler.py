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

import gc
from unittest import mock

import pytest
import torch

from tensorrt_llm import LLM
from tensorrt_llm.executor import request as executor_request
from tensorrt_llm.llmapi import (
    CudaGraphConfig,
    Eagle3DecodingConfig,
    KvCacheConfig,
    MoeConfig,
    MTPDecodingConfig,
    SamplingParams,
    SchedulerConfig,
)
from tensorrt_llm.lora_helper import LoraConfig

from ..conftest import llm_models_root, parametrize_with_ids, skip_pre_hopper
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
LONG_PROMPT = _LONG_BLOCK * 12 + "\nBased on the above, summarize the key themes."

# Eviction prompt (~1000 tokens ≈ 32 blocks). Used by chunked-prefill eviction tests
# where chunking naturally creates high concurrency.
EVICTION_PROMPT = _LONG_BLOCK * 24 + "\nSummarize the key themes in one paragraph."

# Short eviction prompts (~307 tokens ≈ 10 blocks each). Used by non-chunked eviction
# tests to allow high concurrency: 12 concurrent × 10 blocks = 120 > 96 GPU blocks.
# Unique PREFIX prevents block_reuse from sharing blocks (radix tree is prefix-based).
EVICTION_PROMPTS_SHORT = [
    f"Topic {i}: " + _LONG_BLOCK * 6 + "\nSummarize the key themes." for i in range(16)
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# V2 scheduler requires MAX_UTILIZATION policy
_V2_SCHEDULER_CONFIG = SchedulerConfig(capacity_scheduler_policy="MAX_UTILIZATION")

# ---------------------------------------------------------------------------
# Eviction test parameters.
# Goal: CUDA graph warmup passes AND runtime triggers scheduler eviction.
#
# Model: Llama-3.2-1B (max_position_embeddings=131072, tokens_per_block=32)
# Per-block KV: 16 layers × 8 heads × 64 dim × 2(K+V) × 2(bf16) × 32 = 1 MB
#
# max_seq_len=2048  → per-request max 64 blocks. Caps warmup long dummy.
# max_tokens=3072   → 96 GPU blocks. Warmup: 11 short (11) + 1 long (65) = 76 < 96 ✓
# host_cache_size=32MB → 32 host blocks. Total = 96+32 = 128 blocks.
# max_batch_size=12  → high concurrency.
#
# Non-chunked tests use EVICTION_PROMPTS_SHORT (~256 tokens = 8 blocks each).
# 12 concurrent × 8 blocks = 96 = GPU pool, so any gen growth → eviction.
# Chunked tests use EVICTION_PROMPT (~1189 tokens) where chunking creates
# natural concurrency pressure with incremental block allocation.
# ---------------------------------------------------------------------------
_EVICT_MAX_SEQ_LEN = 2048
_EVICT_MAX_TOKENS = 3072
_EVICT_HOST_CACHE_SIZE = 32 * 1024 * 1024  # 32 MiB
_EVICT_MAX_BATCH_SIZE = 12


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
        assert len(out.outputs[0].token_ids) > 0, f"Output {i} has empty token_ids"


def _assert_outputs_match(outputs_a, outputs_b, label_a="A", label_b="B"):
    """Assert two output lists produce identical token sequences."""
    assert len(outputs_a) == len(outputs_b), (
        f"Output count mismatch: {label_a}={len(outputs_a)}, {label_b}={len(outputs_b)}"
    )
    for i, (oa, ob) in enumerate(zip(outputs_a, outputs_b)):
        assert oa.outputs[0].token_ids == ob.outputs[0].token_ids, (
            f"Prompt {i}: {label_a} vs {label_b} outputs differ.\n"
            f"{label_a}: {oa.outputs[0].token_ids}\n"
            f"{label_b}: {ob.outputs[0].token_ids}"
        )


def _run_v1_v2_compare(model_path, prompts, sampling_params, kv_extra=None,
                        **llm_kwargs):
    """Run same prompts on V1 and V2, assert identical greedy output.

    Args:
        model_path: HF model path.
        prompts: List of prompt strings.
        sampling_params: SamplingParams (should use temperature=0.0).
        kv_extra: Extra kwargs for both V1 and V2 KvCacheConfig (e.g. enable_block_reuse).
        **llm_kwargs: Extra kwargs for both V1 and V2 LLM (e.g. max_num_tokens).
    """
    kv_extra = kv_extra or {}

    kv_v1 = KvCacheConfig(use_kv_cache_manager_v2=False, **kv_extra)
    with LLM(model_path, kv_cache_config=kv_v1, **llm_kwargs) as llm:
        outputs_v1 = llm.generate(prompts, sampling_params=sampling_params)

    # Free GPU memory from V1 run before starting V2
    gc.collect()
    torch.cuda.empty_cache()

    kv_v2 = KvCacheConfig(use_kv_cache_manager_v2=True, **kv_extra)
    with LLM(model_path, kv_cache_config=kv_v2,
             scheduler_config=_V2_SCHEDULER_CONFIG, **llm_kwargs) as llm:
        outputs_v2 = llm.generate(prompts, sampling_params=sampling_params)

    _assert_all_completed(outputs_v1, expected_count=len(prompts))
    _assert_all_completed(outputs_v2, expected_count=len(prompts))
    _assert_outputs_match(outputs_v1, outputs_v2, "V1", "V2")
    return outputs_v1, outputs_v2


def _run_eviction_test(model_path, prompts, sampling_params, *,
                       enable_block_reuse=False, enable_chunked_prefill=False,
                       max_num_tokens=2048, disable_overlap_scheduler=None):
    """Run an eviction test with _EVICT_* memory constraints.

    Args:
        model_path: HF model path.
        prompts: List of prompt strings.
        sampling_params: SamplingParams.
        enable_block_reuse: Whether to enable block reuse.
        enable_chunked_prefill: Whether to enable chunked prefill.
        max_num_tokens: Max num tokens for the LLM.
        disable_overlap_scheduler: If not None, set disable_overlap_scheduler.
    """
    kv_config = KvCacheConfig(
        use_kv_cache_manager_v2=True,
        max_tokens=_EVICT_MAX_TOKENS,
        enable_block_reuse=enable_block_reuse,
        host_cache_size=_EVICT_HOST_CACHE_SIZE,
    )
    llm_kwargs = dict(
        max_batch_size=_EVICT_MAX_BATCH_SIZE,
        max_seq_len=_EVICT_MAX_SEQ_LEN,
        max_num_tokens=max_num_tokens,
    )
    if enable_chunked_prefill:
        llm_kwargs["enable_chunked_prefill"] = True
    if disable_overlap_scheduler is not None:
        llm_kwargs["disable_overlap_scheduler"] = disable_overlap_scheduler

    with LLM(model_path, kv_cache_config=kv_config,
             scheduler_config=_V2_SCHEDULER_CONFIG, **llm_kwargs) as llm:
        outputs = _run_generate(llm, prompts, sampling_params)

    _assert_all_completed(outputs, expected_count=len(prompts))
    return outputs



# ===========================================================================
# IT1-IT7, IT13-IT15: Functional tests on Llama-3.2-1B
# ===========================================================================
class TestKVCacheV2Llama:
    """Functional tests for V2 scheduler using Llama-3.2-1B (1 GPU)."""

    MODEL_PATH = f"{llm_models_root()}/llama-3.2-models/Llama-3.2-1B"

    # IT1: V2 vs V1 produce identical greedy output
    def test_v2_vs_v1_basic(self):
        _run_v1_v2_compare(
            self.MODEL_PATH, SHORT_PROMPTS[:5],
            SamplingParams(max_tokens=32, temperature=0.0),
        )

    # IT2: Token budget limited — V2 matches V1 under tight max_num_tokens
    def test_token_budget_limited(self):
        _run_v1_v2_compare(
            self.MODEL_PATH, SHORT_PROMPTS,
            SamplingParams(max_tokens=32, temperature=0.0),
            max_num_tokens=64,
        )

    # IT3: Chunked prefill — V2 matches V1 with chunked long prompt
    def test_chunked_prefill(self):
        _run_v1_v2_compare(
            self.MODEL_PATH, [LONG_PROMPT],
            SamplingParams(max_tokens=64, temperature=0.0),
            enable_chunked_prefill=True, max_num_tokens=128,
        )

    # IT4: Chunked prefill multi-request — V2 matches V1
    def test_chunked_prefill_multi_request(self):
        _run_v1_v2_compare(
            self.MODEL_PATH, MEDIUM_PROMPTS,
            SamplingParams(max_tokens=64, temperature=0.0),
            enable_chunked_prefill=True, max_num_tokens=256,
        )

    # IT5: Eviction under tight memory.
    # cuda_graph=True: uses _EVICT_* constants (see module-level comment).
    # cuda_graph=False: max_tokens=288, no host tier, resize() failure path.
    @pytest.mark.timeout(300)
    @pytest.mark.parametrize("use_cuda_graph", [True, False], ids=["cuda_graph", "no_cuda_graph"])
    def test_eviction(self, use_cuda_graph):
        sampling_params = SamplingParams(max_tokens=64, temperature=0.0)
        if use_cuda_graph:
            _run_eviction_test(
                self.MODEL_PATH, EVICTION_PROMPTS_SHORT, sampling_params,
            )
        else:
            # No CUDA graph, smaller config, no host tier
            kv_config = KvCacheConfig(
                use_kv_cache_manager_v2=True,
                max_tokens=288,
                enable_block_reuse=False,
            )
            with LLM(
                self.MODEL_PATH,
                kv_cache_config=kv_config,
                scheduler_config=_V2_SCHEDULER_CONFIG,
                max_batch_size=4,
                max_num_tokens=256,
                cuda_graph_config=None,
            ) as llm:
                outputs = _run_generate(llm, SHORT_PROMPTS, sampling_params)
            _assert_all_completed(outputs, expected_count=len(SHORT_PROMPTS))

    # IT7: Batch size limited — V2 matches V1 under tight max_batch_size
    def test_batch_size_limited(self):
        _run_v1_v2_compare(
            self.MODEL_PATH, SHORT_PROMPTS,
            SamplingParams(max_tokens=32, temperature=0.0),
            max_batch_size=2, max_num_tokens=8192,
        )

    # IT13: V2 overlap/non-overlap scheduler matches V1
    @pytest.mark.parametrize("disable_overlap", [True, False],
                             ids=["non_overlap", "overlap"])
    def test_overlap_scheduler(self, disable_overlap):
        _run_v1_v2_compare(
            self.MODEL_PATH, SHORT_PROMPTS[:5],
            SamplingParams(max_tokens=32, temperature=0.0),
            disable_overlap_scheduler=disable_overlap,
        )

    # IT14: Block reuse — V2 matches V1 with shared-prefix prompts
    def test_block_reuse(self):
        _run_v1_v2_compare(
            self.MODEL_PATH, SHARED_PREFIX_PROMPTS,
            SamplingParams(max_tokens=64, temperature=0.0),
            kv_extra={"enable_block_reuse": True},
        )

    # IT23: Partial block reuse — V2 matches V1
    # Tests the distinct code path where context_current_position is set from
    # num_committed_tokens at a partial-block boundary, and chunk alignment
    # to block boundary is performed.
    def test_partial_block_reuse(self):
        _run_v1_v2_compare(
            self.MODEL_PATH, SHARED_PREFIX_PROMPTS,
            SamplingParams(max_tokens=64, temperature=0.0),
            kv_extra={"enable_block_reuse": True, "enable_partial_reuse": True},
        )

    # IT15: Chunked prefill with eviction.
    @pytest.mark.timeout(300)
    def test_chunked_prefill_with_eviction(self):
        _run_eviction_test(
            self.MODEL_PATH, [EVICTION_PROMPT] * 16,
            SamplingParams(max_tokens=64, temperature=0.0),
            enable_chunked_prefill=True, max_num_tokens=256,
        )

    # IT18: Eviction with block reuse enabled.
    @pytest.mark.timeout(300)
    def test_eviction_with_block_reuse(self):
        _run_eviction_test(
            self.MODEL_PATH, EVICTION_PROMPTS_SHORT,
            SamplingParams(max_tokens=64, temperature=0.0),
            enable_block_reuse=True,
        )

    # IT19: Chunked prefill + eviction + block reuse.
    # Uses diverse prompts so block_reuse cannot collapse all into shared blocks.
    @pytest.mark.timeout(300)
    def test_chunked_prefill_eviction_block_reuse(self):
        _run_eviction_test(
            self.MODEL_PATH, EVICTION_PROMPTS_SHORT,
            SamplingParams(max_tokens=64, temperature=0.0),
            enable_block_reuse=True, enable_chunked_prefill=True,
            max_num_tokens=256,
        )

    # IT20: Eviction + overlap scheduler.
    # Overlap + eviction is the most timing-sensitive combination:
    # iteration N+1's scheduler may suspend N's requests while N's
    # update_resources hasn't run yet (BUG-009 scenario).
    @pytest.mark.timeout(300)
    def test_eviction_overlap(self):
        _run_eviction_test(
            self.MODEL_PATH, EVICTION_PROMPTS_SHORT,
            SamplingParams(max_tokens=64, temperature=0.0),
            disable_overlap_scheduler=False,
        )


# ===========================================================================
# IT8-IT9: LoRA tests on llama-7b-hf
# ===========================================================================
@pytest.mark.skip_less_device_memory(40000)
class TestKVCacheV2LoRA:
    """LoRA tests for V2 scheduler using llama-7b-hf (1 GPU, >=40GB)."""

    MODEL_PATH = f"{llm_models_root()}/llama-models/llama-7b-hf"
    LORA_DIR = f"{llm_models_root()}/llama-models/luotuo-lora-7b-0.1"

    # IT8: Single LoRA adapter — V2 matches V1
    def test_lora_v2(self):
        lora_config = LoraConfig(
            lora_dir=[self.LORA_DIR],
            max_lora_rank=8,
            max_loras=1,
            max_cpu_loras=1,
            lora_target_modules=["attn_q", "attn_k", "attn_v"],
        )
        sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
        prompts = SHORT_PROMPTS[:3]
        lora_request = executor_request.LoRARequest("lora-0", 0, self.LORA_DIR)

        # V1 with LoRA
        kv_v1 = KvCacheConfig(use_kv_cache_manager_v2=False, free_gpu_memory_fraction=0.4)
        with LLM(
            self.MODEL_PATH, kv_cache_config=kv_v1, lora_config=lora_config,
        ) as llm:
            outputs_v1 = llm.generate(
                prompts, sampling_params=sampling_params, lora_request=lora_request,
            )

        gc.collect()
        torch.cuda.empty_cache()

        # V2 with LoRA
        kv_v2 = KvCacheConfig(use_kv_cache_manager_v2=True, free_gpu_memory_fraction=0.4)
        with LLM(
            self.MODEL_PATH, kv_cache_config=kv_v2,
            scheduler_config=_V2_SCHEDULER_CONFIG, lora_config=lora_config,
        ) as llm:
            outputs_v2 = llm.generate(
                prompts, sampling_params=sampling_params, lora_request=lora_request,
            )

        _assert_all_completed(outputs_v1, expected_count=3)
        _assert_all_completed(outputs_v2, expected_count=3)
        _assert_outputs_match(outputs_v1, outputs_v2, "V1-LoRA", "V2-LoRA")

    # IT9: Multiple LoRA adapters with max_loras=1 (adapter swapping)
    def test_lora_multi_adapter_v2(self):
        lora_config = LoraConfig(
            lora_dir=[self.LORA_DIR],
            max_lora_rank=8,
            max_loras=1,
            max_cpu_loras=1,
            lora_target_modules=["attn_q", "attn_k", "attn_v"],
        )
        kv_config = KvCacheConfig(use_kv_cache_manager_v2=True, free_gpu_memory_fraction=0.4)
        sampling_params = SamplingParams(max_tokens=32, temperature=0.0)

        with LLM(
            self.MODEL_PATH,
            kv_cache_config=kv_config,
            scheduler_config=_V2_SCHEDULER_CONFIG,
            lora_config=lora_config,
        ) as llm:
            # First request with LoRA
            lora_request = executor_request.LoRARequest("lora-0", 0, self.LORA_DIR)
            outputs1 = llm.generate(
                SHORT_PROMPTS[:2], sampling_params=sampling_params, lora_request=lora_request
            )
            _assert_all_completed(outputs1, expected_count=2)

            # Second request without LoRA (adapter swap)
            outputs2 = llm.generate(SHORT_PROMPTS[2:4], sampling_params=sampling_params)
            _assert_all_completed(outputs2, expected_count=2)

    # IT24: LoRA + chunked prefill — V2 matches V1
    # Tests PEFT page accounting during incremental context allocation.
    def test_lora_chunked_prefill(self):
        lora_config = LoraConfig(
            lora_dir=[self.LORA_DIR],
            max_lora_rank=8,
            max_loras=1,
            max_cpu_loras=1,
            lora_target_modules=["attn_q", "attn_k", "attn_v"],
        )
        sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
        prompts = MEDIUM_PROMPTS[:3]
        lora_request = executor_request.LoRARequest("lora-0", 0, self.LORA_DIR)

        # V1 with LoRA + chunked prefill
        kv_v1 = KvCacheConfig(use_kv_cache_manager_v2=False, free_gpu_memory_fraction=0.4)
        with LLM(
            self.MODEL_PATH, kv_cache_config=kv_v1, lora_config=lora_config,
            enable_chunked_prefill=True, max_num_tokens=128,
        ) as llm:
            outputs_v1 = llm.generate(
                prompts, sampling_params=sampling_params, lora_request=lora_request,
            )

        gc.collect()
        torch.cuda.empty_cache()

        # V2 with LoRA + chunked prefill
        kv_v2 = KvCacheConfig(use_kv_cache_manager_v2=True, free_gpu_memory_fraction=0.4)
        with LLM(
            self.MODEL_PATH, kv_cache_config=kv_v2,
            scheduler_config=_V2_SCHEDULER_CONFIG, lora_config=lora_config,
            enable_chunked_prefill=True, max_num_tokens=128,
        ) as llm:
            outputs_v2 = llm.generate(
                prompts, sampling_params=sampling_params, lora_request=lora_request,
            )

        _assert_all_completed(outputs_v1, expected_count=3)
        _assert_all_completed(outputs_v2, expected_count=3)
        _assert_outputs_match(outputs_v1, outputs_v2, "V1-LoRA-chunked", "V2-LoRA-chunked")

    # IT25: LoRA + eviction — PEFT page accounting across eviction/resume.
    @pytest.mark.timeout(300)
    def test_lora_eviction(self):
        lora_config = LoraConfig(
            lora_dir=[self.LORA_DIR],
            max_lora_rank=8,
            max_loras=1,
            max_cpu_loras=1,
            lora_target_modules=["attn_q", "attn_k", "attn_v"],
        )
        # Tight memory to force eviction. llama-7b: ~14 heads, head_dim=128,
        # 32 layers, bf16, tokens_per_block=32 → ~7 MB/block.
        # max_tokens=4096 → ~585 blocks → ~4 GB. With 40GB GPU this is tight
        # enough with many concurrent LoRA requests.
        kv_config = KvCacheConfig(
            use_kv_cache_manager_v2=True,
            free_gpu_memory_fraction=0.2,
            host_cache_size=64 * 1024 * 1024,  # 64 MiB host tier
        )
        sampling_params = SamplingParams(max_tokens=64, temperature=0.0)
        lora_request = executor_request.LoRARequest("lora-0", 0, self.LORA_DIR)

        with LLM(
            self.MODEL_PATH,
            kv_cache_config=kv_config,
            scheduler_config=_V2_SCHEDULER_CONFIG,
            lora_config=lora_config,
            max_batch_size=8,
        ) as llm:
            outputs = llm.generate(
                SHORT_PROMPTS, sampling_params=sampling_params,
                lora_request=lora_request,
            )
        _assert_all_completed(outputs, expected_count=10)


# ===========================================================================
# IT10-IT11: MTP draft tokens on DeepSeek-V3-Lite
# ===========================================================================
@skip_pre_hopper
@pytest.mark.skip_less_device_memory(60000)
class TestKVCacheV2DSv3Lite:
    """MTP speculative decoding tests with V2 scheduler on DeepSeek-V3-Lite."""

    MODEL_PATH = f"{llm_models_root()}/DeepSeek-V3-Lite/bf16"

    # IT10: MTP draft tokens — V2 matches V1
    # MTP creates main + 2 draft KV cache managers, each consuming GPU memory.
    # V2's Python manager uses cuMemCreate to pre-allocate physical pages.
    # Use low free_gpu_memory_fraction to leave headroom for activations/cublas.
    def test_mtp_draft_tokens(self):
        _run_v1_v2_compare(
            self.MODEL_PATH, SHORT_PROMPTS[:5],
            SamplingParams(max_tokens=64, temperature=0.0),
            kv_extra={"free_gpu_memory_fraction": 0.3},
            speculative_config=MTPDecodingConfig(num_nextn_predict_layers=2),
            max_num_tokens=8192,
        )

    # IT11: MTP draft tokens with chunked prefill — V2 matches V1
    def test_mtp_chunked_draft_tokens(self):
        _run_v1_v2_compare(
            self.MODEL_PATH, [LONG_PROMPT],
            SamplingParams(max_tokens=64, temperature=0.0),
            kv_extra={"free_gpu_memory_fraction": 0.3},
            speculative_config=MTPDecodingConfig(num_nextn_predict_layers=2),
            enable_chunked_prefill=True, max_num_tokens=256,
        )

    # IT26: MTP + eviction — speculative decoding under memory pressure.
    # Tests draft token fitting when GPU pool is exhausted and eviction triggers.
    @pytest.mark.timeout(300)
    def test_mtp_eviction(self):
        # DSv3Lite MLA: 2 layers, kv_lora_rank=512, dtype=bf16, tokens_per_block=64
        # Per-block KV ≈ 2 * 512 * 2(bf16) * 64 = 128 KB/block
        # With MTP (2 draft layers): main + 2 draft managers.
        # Use tight free_gpu_memory_fraction to constrain GPU blocks.
        kv_config = KvCacheConfig(
            use_kv_cache_manager_v2=True,
            free_gpu_memory_fraction=0.3,
            host_cache_size=64 * 1024 * 1024,  # 64 MiB host tier
        )
        mtp_config = MTPDecodingConfig(num_nextn_predict_layers=2)
        sampling_params = SamplingParams(max_tokens=64, temperature=0.0)
        # Use many prompts to create memory pressure with MTP
        prompts = SHORT_PROMPTS * 2  # 20 prompts
        with LLM(
            self.MODEL_PATH,
            kv_cache_config=kv_config,
            scheduler_config=_V2_SCHEDULER_CONFIG,
            speculative_config=mtp_config,
            max_batch_size=16,
            max_num_tokens=4096,
        ) as llm:
            outputs = _run_generate(llm, prompts, sampling_params)
        _assert_all_completed(outputs, expected_count=20)


# ===========================================================================
# IT27: Attention DP with V2 on DeepSeek-V3-Lite (2 GPUs)
# ===========================================================================
@skip_pre_hopper
@pytest.mark.skip_less_device_memory(60000)
@pytest.mark.skip_less_device(2)
class TestKVCacheV2AttentionDP:
    """Attention data parallelism with V2 scheduler on DeepSeek-V3-Lite (2 GPUs)."""

    MODEL_PATH = f"{llm_models_root()}/DeepSeek-V3-Lite/bf16"

    # IT27: Attention DP — V2 matches V1
    # enable_attention_dp sets dp_size=tp_size, attn_tp_size=1.
    # Scheduler capacity is bumped +1 for attention DP dummy request.
    def test_attention_dp(self):
        sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
        prompts = SHORT_PROMPTS[:5]

        kv_v1 = KvCacheConfig(
            use_kv_cache_manager_v2=False, free_gpu_memory_fraction=0.75,
        )
        with LLM(
            self.MODEL_PATH, kv_cache_config=kv_v1,
            tensor_parallel_size=2, enable_attention_dp=True,
            moe_config=MoeConfig(backend="CUTLASS"),
        ) as llm:
            outputs_v1 = llm.generate(prompts, sampling_params=sampling_params)

        gc.collect()
        torch.cuda.empty_cache()

        kv_v2 = KvCacheConfig(
            use_kv_cache_manager_v2=True, free_gpu_memory_fraction=0.75,
        )
        with LLM(
            self.MODEL_PATH, kv_cache_config=kv_v2,
            scheduler_config=_V2_SCHEDULER_CONFIG,
            tensor_parallel_size=2, enable_attention_dp=True,
            moe_config=MoeConfig(backend="CUTLASS"),
        ) as llm:
            outputs_v2 = llm.generate(prompts, sampling_params=sampling_params)

        _assert_all_completed(outputs_v1, expected_count=5)
        _assert_all_completed(outputs_v2, expected_count=5)
        _assert_outputs_match(outputs_v1, outputs_v2, "V1-ADP", "V2-ADP")


# ===========================================================================
# IT21: V2 non-overlap accuracy on DeepSeek-V3-Lite
# ===========================================================================
@skip_pre_hopper
@pytest.mark.skip_less_device_memory(60000)
class TestKVCacheV2DSv3LiteAccuracy(LlmapiAccuracyTestHarness):
    """V2 non-overlap accuracy test on DeepSeek-V3-Lite (1 GPU, >=60GB)."""

    MODEL_NAME = "deepseek-ai/DeepSeek-V3-Lite"
    MODEL_PATH = f"{llm_models_root()}/DeepSeek-V3-Lite/bf16"

    # IT21/IT22: GSM8K accuracy with V2 non-overlap / overlap scheduler
    @parametrize_with_ids("disable_overlap", [True, False])
    def test_accuracy_v2(self, disable_overlap):
        kv_config = KvCacheConfig(free_gpu_memory_fraction=0.75, use_kv_cache_manager_v2=True)
        pytorch_config = dict(
            disable_overlap_scheduler=disable_overlap,
            cuda_graph_config=CudaGraphConfig(),
            moe_config=MoeConfig(backend="CUTLASS"),
        )
        with LLM(
            self.MODEL_PATH,
            kv_cache_config=kv_config,
            scheduler_config=_V2_SCHEDULER_CONFIG,
            **pytorch_config,
        ) as llm:
            task = GSM8K(self.MODEL_NAME)
            task.evaluate(llm)


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
    @mock.patch.dict(GSM8K.EVALUATE_KWARGS, {"scores_filter": "exact_match,flexible-extract"})
    @mock.patch.object(GSM8K, "MAX_OUTPUT_LEN", 8192)
    def test_accuracy_v2(self):
        kv_config = KvCacheConfig(free_gpu_memory_fraction=0.7, use_kv_cache_manager_v2=True)
        pytorch_config = dict(
            disable_overlap_scheduler=False,
            cuda_graph_config=CudaGraphConfig(),
            moe_config=MoeConfig(backend="CUTLASS"),
        )
        with LLM(
            self.MODEL_PATH,
            tensor_parallel_size=4,
            kv_cache_config=kv_config,
            scheduler_config=_V2_SCHEDULER_CONFIG,
            max_batch_size=720,
            **pytorch_config,
        ) as llm:
            model_name = "GPT-OSS/120B-MXFP4"
            task = GSM8K(model_name)
            task.evaluate(llm, extra_evaluator_kwargs=self.extra_evaluator_kwargs)

    # IT17: Eagle3 speculative decoding with V2
    @mock.patch.dict(GSM8K.EVALUATE_KWARGS, {"scores_filter": "exact_match,flexible-extract"})
    @mock.patch.object(GSM8K, "MAX_OUTPUT_LEN", 8192)
    def test_eagle3_v2(self):
        kv_config = KvCacheConfig(
            free_gpu_memory_fraction=0.4, enable_block_reuse=False, use_kv_cache_manager_v2=True
        )
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
        with LLM(
            self.MODEL_PATH,
            tensor_parallel_size=4,
            kv_cache_config=kv_config,
            scheduler_config=_V2_SCHEDULER_CONFIG,
            max_batch_size=720,
            speculative_config=spec_config,
            **pytorch_config,
        ) as llm:
            model_name = "GPT-OSS/120B-MXFP4"
            task = GSM8K(model_name)
            task.evaluate(llm, extra_evaluator_kwargs=self.extra_evaluator_kwargs)
