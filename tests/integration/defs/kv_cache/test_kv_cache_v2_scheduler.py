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
"""DeepSeek MTP integration tests for KVCacheV2Scheduler."""

import pytest

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


def _run_v2(model_path, prompts, sampling_params, kv_extra=None, **llm_kwargs):
    """Run prompts with the V2 KV-cache manager and assert completion.

    Args:
        model_path: HF model path.
        prompts: List of prompt strings.
        sampling_params: SamplingParams (should use temperature=0.0).
        kv_extra: Extra kwargs for the V2 KvCacheConfig.
        **llm_kwargs: Extra kwargs for the V2 LLM.
    """
    kv_extra = kv_extra or {}
    kv_v2 = KvCacheConfig(use_kv_cache_manager_v2=True, **kv_extra)
    with LLM(
        model_path, kv_cache_config=kv_v2, scheduler_config=_V2_SCHEDULER_CONFIG, **llm_kwargs
    ) as llm:
        outputs_v2 = llm.generate(prompts, sampling_params=sampling_params)

    _assert_all_completed(outputs_v2, expected_count=len(prompts))
    return outputs_v2


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

    def _run(self, prompts, max_tokens=32, kv_extra=None, **llm_kwargs):
        """Run V2 with MTP and assert all requests complete."""
        if kv_extra is None:
            kv_extra = {"free_gpu_memory_fraction": 0.3}
        llm_kwargs.setdefault("max_num_tokens", 8192)
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0.0)
        return _run_v2(
            self.MODEL_PATH,
            prompts,
            sampling_params,
            kv_extra=kv_extra,
            speculative_config=MTPDecodingConfig(max_draft_len=2),
            tensor_parallel_size=self.TP_SIZE,
            **llm_kwargs,
        )

    def test_mtp_draft_tokens(self):
        self._run(SHORT_PROMPTS[:5])

    def test_mtp_chunked_draft_tokens(self):
        self._run([LONG_PROMPT], enable_chunked_prefill=True, max_num_tokens=256)

    def test_mtp_eviction(self):
        # Eviction parameters for DeepSeek-V3-Lite MTP (tokens_per_block=32):
        # max_seq_len=512   → per-request max 16 blocks. Caps warmup dummy.
        # max_tokens=8192   → 256 GPU blocks.
        #   Warmup: 15 short(1 block) + 1 long(16 blocks) = 31 < 256 ✓
        # 40 prompts, gen=256 → 16 concurrent × ~9 blocks = 144 at peak.
        #   With draft KV doubling pressure → eviction expected.
        # host_cache_size=512MB → host tier for evicted blocks.
        self._run(
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
