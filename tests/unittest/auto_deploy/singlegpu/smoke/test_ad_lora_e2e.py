# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""E2E smoke test for LoRA in AutoDeploy.

Tests the full AD pipeline with TinyLlama-1.1B + tarot LoRA adapter.
The tarot adapter produces clearly different output (tarot card readings)
compared to the base model, making it easy to verify LoRA is working.

Uses LLM API directly with lora_request (same pattern as PyTorch backend
tests in test_llm_pytorch.py).
"""

import os

import pytest

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm.executor.request import LoRARequest
from tensorrt_llm.lora_helper import LoraConfig

_MODELS_ROOT = os.environ.get("LLM_MODELS_ROOT", os.path.expanduser("~/dev/model-symlinks"))
_MODEL_DIR = os.path.join(_MODELS_ROOT, "TinyLlama-1.1B-Chat-v1.0")
_TAROT_LORA_DIR = os.path.join(_MODELS_ROOT, "tinyllama-tarot-v1")

# The Tower card prompt: base model thinks it's a card game, tarot adapter gives correct reading
_PROMPT = "<|user|>\nWhat does The Tower tarot card mean?\n<|assistant|>\n"


def _model_available():
    return os.path.isdir(_MODEL_DIR) and os.path.isdir(_TAROT_LORA_DIR)


@pytest.mark.skipif(not _model_available(), reason="TinyLlama + tarot adapter not found")
def test_ad_lora_tarot_changes_output():
    """LoRA adapter produces different, semantically correct output.

    The tarot adapter makes the model interpret "The Tower" as a tarot card
    (upheaval, transformation) instead of a generic/game concept.
    """
    sp = SamplingParams(max_tokens=60, temperature=0.0)
    lora_config = LoraConfig(
        lora_dir=[_TAROT_LORA_DIR],
        max_lora_rank=32,
        max_loras=1,
        max_cpu_loras=1,
    )

    llm = LLM(
        model=_MODEL_DIR,
        backend="_autodeploy",
        lora_config=lora_config,
        max_batch_size=1,
        cuda_graph_config={"max_batch_size": 1},
    )

    try:
        # Base model output (no LoRA)
        base = llm.generate([_PROMPT], sp)
        base_text = base[0].outputs[0].text

        # LoRA output (tarot adapter)
        lora_req = LoRARequest("tarot", 0, _TAROT_LORA_DIR)
        lora = llm.generate([_PROMPT], sp, lora_request=[lora_req])
        lora_text = lora[0].outputs[0].text
    finally:
        llm.shutdown()

    print(f"Base: {repr(base_text[:200])}")
    print(f"LoRA: {repr(lora_text[:200])}")

    # LoRA output should differ from base
    assert base_text != lora_text, (
        f"LoRA output should differ from base.\nBase: {repr(base_text)}\nLoRA: {repr(lora_text)}"
    )

    # LoRA output should contain tarot-related keywords
    lora_lower = lora_text.lower()
    tarot_keywords = ["change", "upheaval", "transform", "tower", "significant", "loss", "crisis"]
    has_tarot_content = any(kw in lora_lower for kw in tarot_keywords)
    assert has_tarot_content, (
        f"LoRA output should contain tarot-related content.\n"
        f"Expected keywords: {tarot_keywords}\n"
        f"Got: {repr(lora_text)}"
    )
