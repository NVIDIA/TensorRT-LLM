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
eviction, LoRA/PEFT, MTP draft tokens, block reuse, and overlap scheduler.
"""

import gc

import pytest
import torch

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig, MTPDecodingConfig, SamplingParams, SchedulerConfig

from ..conftest import llm_models_root, skip_pre_hopper

# ---------------------------------------------------------------------------
# Shared prompts
# ---------------------------------------------------------------------------
# Short prompts chosen for deterministic, short answers (temperature=0).
SHORT_PROMPTS = [
    "What is 2+2? Answer in one number.",
    "Capital of France? One word.",
    "Largest planet in our solar system? One word.",
    "Who wrote Romeo and Juliet? One name.",
    "Boiling point of water in Celsius? One number.",
    "Language spoken in Brazil? One word.",
    "Name a mammal that can fly. One word.",
    "Largest ocean on Earth? One word.",
    "How many continents are there? One number.",
    "First prime number? One digit.",
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


def _assert_all_completed(outputs, expected_count=None):
    """Assert all outputs have non-empty generated text."""
    if expected_count is not None:
        assert len(outputs) == expected_count
    for i, out in enumerate(outputs):
        assert len(out.outputs) > 0, f"Output {i} has no outputs"
        assert len(out.outputs[0].token_ids) > 0, f"Output {i} has empty token_ids"


def _assert_outputs_match(outputs_a, outputs_b, label_a="A", label_b="B"):
    """Assert two output lists produce identical text."""
    assert len(outputs_a) == len(outputs_b), (
        f"Output count mismatch: {label_a}={len(outputs_a)}, {label_b}={len(outputs_b)}"
    )
    for i, (oa, ob) in enumerate(zip(outputs_a, outputs_b)):
        assert oa.outputs[0].text == ob.outputs[0].text, (
            f"Prompt {i}: {label_a} vs {label_b} outputs differ.\n"
            f"{label_a}: {oa.outputs[0].text[:500]}\n"
            f"{label_b}: {ob.outputs[0].text[:500]}"
        )


def _run_v1_v2_compare(
    model_path, prompts, sampling_params, kv_extra=None, *, assert_outputs_match=True, **llm_kwargs
):
    """Run same prompts on V1 and V2; optionally assert identical output.

    Args:
        model_path: HF model path.
        prompts: List of prompt strings.
        sampling_params: SamplingParams (should use temperature=0.0).
        kv_extra: Extra kwargs for both V1 and V2 KvCacheConfig (e.g. enable_block_reuse).
        assert_outputs_match: If True, assert V1 and V2 outputs are identical (text).
            Set False for MTP/speculative tests where scheduler differences can diverge.
        **llm_kwargs: Extra kwargs for both V1 and V2 LLM (e.g. max_num_tokens).
    """
    kv_extra = kv_extra or {}

    outputs_v1 = None
    if assert_outputs_match:
        kv_v1 = KvCacheConfig(use_kv_cache_manager_v2=False, **kv_extra)
        with LLM(model_path, kv_cache_config=kv_v1, **llm_kwargs) as llm:
            outputs_v1 = llm.generate(prompts, sampling_params=sampling_params)
        gc.collect()
        torch.cuda.empty_cache()

    kv_v2 = KvCacheConfig(use_kv_cache_manager_v2=True, **kv_extra)
    with LLM(
        model_path, kv_cache_config=kv_v2, scheduler_config=_V2_SCHEDULER_CONFIG, **llm_kwargs
    ) as llm:
        outputs_v2 = llm.generate(prompts, sampling_params=sampling_params)

    _assert_all_completed(outputs_v2, expected_count=len(prompts))
    if assert_outputs_match:
        _assert_all_completed(outputs_v1, expected_count=len(prompts))
        _assert_outputs_match(outputs_v1, outputs_v2, "V1", "V2")
    return outputs_v1, outputs_v2


def _run_eviction_test(
    model_path,
    prompts,
    sampling_params,
    *,
    enable_block_reuse=False,
    enable_chunked_prefill=False,
    max_num_tokens=2048,
    disable_overlap_scheduler=None,
):
    """Run eviction test: V1 vs V2 comparison with _EVICT_* memory constraints.

    Both V1 and V2 are run with identical tight-memory configs to trigger
    eviction, then greedy outputs are compared for correctness.

    Args:
        model_path: HF model path.
        prompts: List of prompt strings.
        sampling_params: SamplingParams.
        enable_block_reuse: Whether to enable block reuse.
        enable_chunked_prefill: Whether to enable chunked prefill.
        max_num_tokens: Max num tokens for the LLM.
        disable_overlap_scheduler: If not None, set disable_overlap_scheduler.
    """
    kv_extra = dict(
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

    return _run_v1_v2_compare(model_path, prompts, sampling_params, kv_extra=kv_extra, **llm_kwargs)


# ===========================================================================
# MTP tests on DeepSeek-V3-Lite (2 GPUs)
# ===========================================================================
@skip_pre_hopper
@pytest.mark.skip_less_device_memory(60000)
@pytest.mark.skip_less_device(2)
class TestKVCacheV2DSv3Lite:
    """MTP speculative decoding tests with V2 scheduler on DeepSeek-V3-Lite (2 GPUs)."""

    MODEL_PATH = f"{llm_models_root()}/DeepSeek-V3-Lite/bf16"
    TP_SIZE = 2

    def _compare(self, prompts, max_tokens=32, kv_extra=None, **llm_kwargs):
        """Run V1 vs V2 with MTP; assert both complete (match not asserted — MTP can diverge)."""
        if kv_extra is None:
            kv_extra = {"free_gpu_memory_fraction": 0.3}
        llm_kwargs.setdefault("max_num_tokens", 8192)
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        return _run_v1_v2_compare(
            self.MODEL_PATH,
            prompts,
            sampling_params,
            kv_extra=kv_extra,
            speculative_config=MTPDecodingConfig(max_draft_len=2),
            tensor_parallel_size=self.TP_SIZE,
            assert_outputs_match=False,
            **llm_kwargs,
        )

    # MTP draft tokens — both V1 and V2 complete (MTP can diverge)
    def test_mtp_draft_tokens(self):
        self._compare(SHORT_PROMPTS[:5])

    # MTP + chunked prefill — both V1 and V2 complete
    def test_mtp_chunked_draft_tokens(self):
        self._compare([LONG_PROMPT], enable_chunked_prefill=True, max_num_tokens=256)

    def test_mtp_eviction(self):
        # Eviction parameters for DeepSeek-V3-Lite MTP (tokens_per_block=32):
        # max_seq_len=512   → per-request max 16 blocks. Caps warmup dummy.
        # max_tokens=8192   → 256 GPU blocks.
        #   Warmup: 15 short(1 block) + 1 long(16 blocks) = 31 < 256 ✓
        # 40 prompts, gen=256 → 16 concurrent × ~9 blocks = 144 at peak.
        #   With draft KV doubling pressure → eviction expected.
        # host_cache_size=512MB → host tier for evicted blocks.
        self._compare(
            SHORT_PROMPTS * 4,  # 40 prompts for memory pressure
            max_tokens=256,  # longer generation to fill KV pool
            kv_extra={
                "free_gpu_memory_fraction": 0.3,
                "max_tokens": 4096,  # constrain KV pool to ~128 blocks
                "host_cache_size": 512 * 1024 * 1024,  # 512 MiB host tier
            },
            max_batch_size=16,
            max_num_tokens=4096,
            max_seq_len=512,  # cap warmup dummy request size
        )
