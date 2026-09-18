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

"""Integration tests for stream localization equivalence.

Verifies that running inference with locality domains enabled produces
identical outputs to running without them, using the LLM API end-to-end.

Rubin only: locality domains need hardware that exposes two locality domains.
``TRT_LLM_MOCK_LOCALIZATION_SUPPORT`` is deliberately not used here -- it only
reaches the KV cache allocator, so the fork/join runtime would still fail.

Model: Qwen3-4B.
"""

from ..conftest import llm_models_root, skip_no_rubin

# Qwen3-4B: dense GQA, so attention arrives as a single mixed call -- the path
# an MLA model never exercises. Suggested by the feature author over Llama,
# which is P1.
_MODEL_PATH = f"{llm_models_root()}/Qwen3/Qwen3-4B"

SHORT_PROMPTS = [
    "What is 2+2? Answer in one number.",
    "Capital of France? One word.",
    "Largest planet in our solar system? One word.",
    "Who wrote Romeo and Juliet? One name.",
]

MIXED_LENGTH_PROMPTS = [
    "Hi",
    "The quick brown fox jumps over the lazy dog and then runs across the field",
    "Once upon a time in a land far far away there lived a brave knight who fought dragons every day",
    "A",
]


def _generate(model_path, prompts, *, localized: bool):
    """Run generation with or without locality domains."""
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams, SchedulerConfig

    kv_config = KvCacheConfig(use_kv_cache_manager_v2=True)
    scheduler_config = SchedulerConfig(capacity_scheduler_policy="MAX_UTILIZATION")
    sampling = SamplingParams(temperature=0.0, max_tokens=32)

    with LLM(
        model_path,
        kv_cache_config=kv_config,
        scheduler_config=scheduler_config,
        # Without this the localized arm runs ordinary mode and the comparison
        # holds no matter what locality domains do. The capability is probed
        # from the device rather than mocked: the mock only reaches the KV cache
        # allocator, so the fork/join runtime still requires real hardware.
        enable_locality_domains=localized,
        env_overrides={
            # Locality domains are only implemented in the Python backend.
            "TLLM_KV_CACHE_MANAGER_V2_BACKEND": "python",
        },
    ) as llm:
        return llm.generate(prompts, sampling_params=sampling)


def _assert_outputs_match(baseline, localized, prompts):
    """Assert that baseline and localized outputs produce identical text."""
    assert len(baseline) == len(localized) == len(prompts)
    for i, (base_output, localized_output) in enumerate(zip(baseline, localized)):
        assert base_output.outputs[0].text == localized_output.outputs[0].text, (
            f"Prompt {i}: baseline={base_output.outputs[0].text!r} "
            f"vs localized={localized_output.outputs[0].text!r}"
        )


@skip_no_rubin
class TestStreamLocalizationEquivalence:
    """Verify that localized (split->forward->merge) matches single-batch forward.

    Uses the LLM API with Qwen3-4B to run the same prompts with and
    without locality domains, comparing generated text at temperature=0.
    """

    def test_generation_only_equivalence(self):
        """Short prompts -- all requests complete context quickly."""
        baseline = _generate(_MODEL_PATH, SHORT_PROMPTS, localized=False)
        localized = _generate(_MODEL_PATH, SHORT_PROMPTS, localized=True)
        _assert_outputs_match(baseline, localized, SHORT_PROMPTS)

    def test_mixed_length_equivalence(self):
        """Varying prompt lengths -- exercises context chunking + generation."""
        baseline = _generate(_MODEL_PATH, MIXED_LENGTH_PROMPTS, localized=False)
        localized = _generate(_MODEL_PATH, MIXED_LENGTH_PROMPTS, localized=True)
        _assert_outputs_match(baseline, localized, MIXED_LENGTH_PROMPTS)
