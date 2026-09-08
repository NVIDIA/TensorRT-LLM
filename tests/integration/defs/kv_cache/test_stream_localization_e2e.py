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

Verifies that running inference with mock locality domain localization
(TRT_LLM_MOCK_LOCALIZATION_SUPPORT=1) produces identical outputs
to running without localization, using the LLM API end-to-end.

Model: Llama-3.2-1B (same as test_kv_cache_v2_scheduler.py).
"""

import gc
import os

import pytest
import torch

from ..conftest import llm_models_root

# Model: Llama-3.2-1B (same as TestKVCacheV2Llama)
_MODEL_PATH = f"{llm_models_root()}/llama-3.2-models/Llama-3.2-1B"

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


@pytest.fixture(autouse=True)
def _gc_cleanup():
    """Free GPU memory between tests."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _generate(model_path, prompts, *, localized: bool):
    """Run generation with or without mock localization."""
    from tensorrt_llm import LLM
    from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams, SchedulerConfig

    env_backup = os.environ.get("TRT_LLM_MOCK_LOCALIZATION_SUPPORT")
    try:
        localization_override = "1" if localized else "0"
        os.environ["TRT_LLM_MOCK_LOCALIZATION_SUPPORT"] = localization_override

        kv_config = KvCacheConfig(use_kv_cache_manager_v2=True)
        scheduler_config = SchedulerConfig(capacity_scheduler_policy="MAX_UTILIZATION")
        sampling = SamplingParams(temperature=0.0, max_tokens=32)

        with LLM(
            model_path,
            kv_cache_config=kv_config,
            scheduler_config=scheduler_config,
            env_overrides={
                "TRT_LLM_MOCK_LOCALIZATION_SUPPORT": localization_override,
            },
        ) as llm:
            return llm.generate(prompts, sampling_params=sampling)
    finally:
        if env_backup is not None:
            os.environ["TRT_LLM_MOCK_LOCALIZATION_SUPPORT"] = env_backup
        elif "TRT_LLM_MOCK_LOCALIZATION_SUPPORT" in os.environ:
            del os.environ["TRT_LLM_MOCK_LOCALIZATION_SUPPORT"]


def _assert_outputs_match(baseline, localized, prompts):
    """Assert that baseline and localized outputs produce identical text."""
    assert len(baseline) == len(localized) == len(prompts)
    for i, (base_output, localized_output) in enumerate(zip(baseline, localized)):
        assert base_output.outputs[0].text == localized_output.outputs[0].text, (
            f"Prompt {i}: baseline={base_output.outputs[0].text!r} "
            f"vs localized={localized_output.outputs[0].text!r}"
        )


class TestStreamLocalizationEquivalence:
    """Verify that localized (split->forward->merge) matches single-batch forward.

    Uses the LLM API with Llama-3.2-1B to run the same prompts with and
    without mock localization, comparing generated text at temperature=0.
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
