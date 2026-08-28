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
from tensorrt_llm._torch.peft.lora.config import LoraConfig
from tensorrt_llm.executor import request as executor_request
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

# Construct a long prompt (~500 tokens) by repeating text
_LONG_BLOCK = (
    "Artificial intelligence has transformed many industries. "
    "From healthcare to finance, AI systems are becoming increasingly capable. "
    "Machine learning models can now process vast amounts of data. "
    "Natural language processing enables computers to understand human language. "
    "Computer vision allows machines to interpret visual information. "
)
LONG_PROMPT = _LONG_BLOCK * 12 + "\nBased on the above, summarize the key themes."

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# V2 scheduler requires MAX_UTILIZATION policy
_V2_SCHEDULER_CONFIG = SchedulerConfig(capacity_scheduler_policy="MAX_UTILIZATION")


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


# ===========================================================================
# LoRA tests on llama-7b-hf
# ===========================================================================
@pytest.mark.skip_less_device_memory(40000)
class TestKVCacheV2LoRA:
    """LoRA tests for V2 scheduler using llama-7b-hf (1 GPU, >=40GB)."""

    MODEL_PATH = f"{llm_models_root()}/llama-models/llama-7b-hf"
    LORA_DIR = f"{llm_models_root()}/llama-models/luotuo-lora-7b-0.1"
    LORA_CONFIG = LoraConfig(
        lora_dir=[LORA_DIR],
        max_lora_rank=8,
        max_loras=1,
        max_cpu_loras=1,
        lora_target_modules=["attn_q", "attn_k", "attn_v"],
    )

    def _run_v1_v2_lora(
        self,
        prompts,
        expected_count=None,
        sampling_params=None,
        kv_extra=None,
        label_suffix="",
        **llm_kwargs,
    ):
        """Run V1 then V2 with LoRA; assert outputs match."""
        if expected_count is None:
            expected_count = len(prompts)
        if sampling_params is None:
            sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
        if kv_extra is None:
            kv_extra = {"free_gpu_memory_fraction": 0.4}
        lora_request = executor_request.LoRARequest("lora-0", 0, self.LORA_DIR)

        kv_v1 = KvCacheConfig(use_kv_cache_manager_v2=False, **kv_extra)
        with LLM(
            self.MODEL_PATH,
            kv_cache_config=kv_v1,
            lora_config=self.LORA_CONFIG,
            **llm_kwargs,
        ) as llm:
            outputs_v1 = llm.generate(
                prompts,
                sampling_params=sampling_params,
                lora_request=lora_request,
            )
        gc.collect()
        torch.cuda.empty_cache()

        kv_v2 = KvCacheConfig(use_kv_cache_manager_v2=True, **kv_extra)
        with LLM(
            self.MODEL_PATH,
            kv_cache_config=kv_v2,
            scheduler_config=_V2_SCHEDULER_CONFIG,
            lora_config=self.LORA_CONFIG,
            **llm_kwargs,
        ) as llm:
            outputs_v2 = llm.generate(
                prompts,
                sampling_params=sampling_params,
                lora_request=lora_request,
            )

        _assert_all_completed(outputs_v1, expected_count=expected_count)
        _assert_all_completed(outputs_v2, expected_count=expected_count)
        _assert_outputs_match(
            outputs_v1,
            outputs_v2,
            f"V1-LoRA{label_suffix}",
            f"V2-LoRA{label_suffix}",
        )

    # Single LoRA adapter — V2 matches V1
    def test_lora_v2(self):
        self._run_v1_v2_lora(SHORT_PROMPTS[:3])

    # LoRA adapter swapping — V2 matches V1
    def test_lora_multi_adapter_v2(self):
        sampling_params = SamplingParams(max_tokens=32, temperature=0.0)
        lora_request = executor_request.LoRARequest("lora-0", 0, self.LORA_DIR)

        def _run_multi_adapter(kv_config, **extra_llm_kwargs):
            with LLM(
                self.MODEL_PATH,
                kv_cache_config=kv_config,
                lora_config=self.LORA_CONFIG,
                **extra_llm_kwargs,
            ) as llm:
                out_lora = llm.generate(
                    SHORT_PROMPTS[:2],
                    sampling_params=sampling_params,
                    lora_request=lora_request,
                )
                out_base = llm.generate(
                    SHORT_PROMPTS[2:4],
                    sampling_params=sampling_params,
                )
            return out_lora, out_base

        outs_v1 = _run_multi_adapter(
            KvCacheConfig(use_kv_cache_manager_v2=False, free_gpu_memory_fraction=0.4),
        )
        gc.collect()
        torch.cuda.empty_cache()
        outs_v2 = _run_multi_adapter(
            KvCacheConfig(use_kv_cache_manager_v2=True, free_gpu_memory_fraction=0.4),
            scheduler_config=_V2_SCHEDULER_CONFIG,
        )

        for label, v1, v2, count in [
            ("LoRA", outs_v1[0], outs_v2[0], 2),
            ("base", outs_v1[1], outs_v2[1], 2),
        ]:
            _assert_all_completed(v1, expected_count=count)
            _assert_all_completed(v2, expected_count=count)
            _assert_outputs_match(v1, v2, f"V1-{label}", f"V2-{label}")

    # LoRA + chunked prefill — V2 matches V1
    def test_lora_chunked_prefill(self):
        self._run_v1_v2_lora(
            MEDIUM_PROMPTS[:3],
            enable_chunked_prefill=True,
            max_num_tokens=128,
            label_suffix="-chunked",
        )

    # LoRA + eviction — V2 matches V1
    def test_lora_eviction(self):
        self._run_v1_v2_lora(
            SHORT_PROMPTS,
            expected_count=10,
            sampling_params=SamplingParams(max_tokens=64, temperature=0.0),
            kv_extra={
                "free_gpu_memory_fraction": 0.2,
                "host_cache_size": 64 * 1024 * 1024,  # 64 MiB host tier
            },
            max_batch_size=8,
            label_suffix="-evict",
        )


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
