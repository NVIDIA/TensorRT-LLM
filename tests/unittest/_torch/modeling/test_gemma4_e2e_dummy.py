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
"""E2E tests for Gemma4 models using dummy configs and dummy weights.

Workaround for FlashInfer head_dim=512 limitation on B200: patch real model
configs to use head_dim=128/global_head_dim=256 (FlashInfer-supported), then
test via LLM API with load_format='dummy' (random weights).

Validates: full pipeline (load → generate), all Gemma4 features (MoE, K=V,
KV sharing, PLE, bidirectional mask, softcapping), hybrid attention, multimodal.
"""

import json
import os
import shutil
import tempfile

import pytest
import transformers
from packaging.version import Version

# Gemma4 requires transformers>=5.5.0 (native Gemma4 config/model classes).
# Applied per-test so unrelated helpers in this module can still be collected.
requires_gemma4_transformers = pytest.mark.skipif(
    Version(transformers.__version__) < Version("5.5.0"),
    reason="Gemma4 requires transformers>=5.5.0",
)

# Models root: require LLM_MODELS_ROOT env var (skip module if unset).
_LLM_MODELS_ROOT = os.environ.get("LLM_MODELS_ROOT")
if _LLM_MODELS_ROOT is None:
    pytest.skip("LLM_MODELS_ROOT not set", allow_module_level=True)
_GEMMA4_MODELS = os.path.join(_LLM_MODELS_ROOT, "gemma4")

# Real model paths — used for tokenizer + base config
MODEL_PATHS = {
    "26B": os.path.join(_GEMMA4_MODELS, "gemma-4-26B-A4B-it"),
    "E2B": os.path.join(_GEMMA4_MODELS, "gemma-4-E2B-it"),
    "31B": os.path.join(_GEMMA4_MODELS, "gemma-4-31B-it"),
    "E4B": os.path.join(_GEMMA4_MODELS, "gemma-4-E4B-it"),
}


def _model_available(name: str) -> bool:
    path = MODEL_PATHS.get(name, "")
    return os.path.isfile(os.path.join(path, "config.json"))


def _make_dummy_config_dir(
    model_path: str,
    dummy_head_dim: int = 128,
    dummy_global_head_dim: int = 256,
    max_layers: int = 8,
    shrink_hidden: bool = True,
) -> str:
    """Create a temp dir with patched config.json + real tokenizer files.

    Patches head_dim/global_head_dim to FlashInfer-supported values.
    Optionally shrinks hidden_size/layers for faster testing.
    Copies tokenizer and processor files from the real model.

    Returns path to the temp directory.
    """
    tmp_dir = tempfile.mkdtemp(prefix="gemma4_dummy_")

    # Load and patch config
    with open(os.path.join(model_path, "config.json")) as f:
        config = json.load(f)

    tc = config.get("text_config", config)
    is_nested = "text_config" in config

    # Patch head dims and ensure dtype is set
    tc["head_dim"] = dummy_head_dim
    tc["global_head_dim"] = dummy_global_head_dim
    if tc.get("torch_dtype") is None:
        tc["torch_dtype"] = "bfloat16"

    # Shrink model for fast testing
    if shrink_hidden:
        orig_layers = tc["num_hidden_layers"]
        orig_layer_types = tc.get("layer_types", [])

        # Reduce layers, preserving feature coverage
        num_kv_shared = tc.get("num_kv_shared_layers", 0)
        if num_kv_shared > 0:
            # KV sharing: keep enough non-shared + some shared layers
            non_shared = min(4, orig_layers - num_kv_shared)
            shared = min(4, num_kv_shared)
            new_layers = non_shared + shared
            tc["num_kv_shared_layers"] = shared
        else:
            new_layers = min(max_layers, orig_layers)

        tc["num_hidden_layers"] = new_layers

        # Rebuild layer_types for new layer count: alternate sliding/full,
        # last layer must be full_attention (transformers validation).
        if orig_layer_types:
            new_lt = []
            for i in range(new_layers):
                new_lt.append("full_attention" if (i + 1) % 3 == 0 else "sliding_attention")
            # Last layer must always be full_attention
            new_lt[-1] = "full_attention"
            tc["layer_types"] = new_lt

        # Shrink all dimensions consistently:
        # hidden_size must >= num_attention_heads * max(head_dim, global_head_dim)
        num_heads = 4
        num_kv_heads = min(tc.get("num_key_value_heads", 2), 2)
        max_hd = max(dummy_head_dim, dummy_global_head_dim)
        hidden = num_heads * max_hd  # e.g., 4 * 256 = 1024

        tc["hidden_size"] = hidden
        tc["intermediate_size"] = hidden * 2
        tc["num_attention_heads"] = num_heads
        tc["num_key_value_heads"] = num_kv_heads
        if tc.get("num_global_key_value_heads") is not None:
            tc["num_global_key_value_heads"] = min(
                num_kv_heads, max(1, tc["num_global_key_value_heads"])
            )

        # Scale PLE dim proportionally
        if tc.get("hidden_size_per_layer_input", 0) > 0:
            tc["hidden_size_per_layer_input"] = 64

        # Reduce MoE experts for speed (must cap top_k too)
        if tc.get("enable_moe_block"):
            tc["num_experts"] = 4
            if tc.get("top_k_experts") is not None:
                tc["top_k_experts"] = min(tc["top_k_experts"], 4)
            if tc.get("num_experts_per_tok") is not None:
                tc["num_experts_per_tok"] = min(tc["num_experts_per_tok"], 4)

        # Reduce sliding window and vocab for faster testing
        tc["sliding_window"] = 256
        tc["vocab_size"] = min(tc.get("vocab_size", 32000), 32000)

    if is_nested:
        config["text_config"] = tc
    else:
        config = tc

    with open(os.path.join(tmp_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Copy tokenizer, processor, and chat template files from real model
    for fname in os.listdir(model_path):
        if (
            fname.startswith("tokenizer")
            or fname.startswith("chat_template")
            or fname in ("processor_config.json", "special_tokens_map.json", "added_tokens.json")
        ):
            src = os.path.join(model_path, fname)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(tmp_dir, fname))

    return tmp_dir


# ---------------------------------------------------------------------------
# Text-only E2E tests
# ---------------------------------------------------------------------------


@requires_gemma4_transformers
@pytest.mark.skipif(not _model_available("26B"), reason="gemma-4-26B-A4B-it not found")
def test_e2e_text_26b_dummy():
    """E2E text generation for 26B-A4B (MoE + K=V + softcap + hybrid attn)."""
    from tensorrt_llm.llmapi import LLM, SamplingParams

    dummy_dir = _make_dummy_config_dir(MODEL_PATHS["26B"])
    try:
        llm = LLM(dummy_dir, load_format="dummy", attn_backend="FLASHINFER", dtype="bfloat16")
        with llm:
            output = llm.generate(["Hello"], SamplingParams(max_tokens=4))
            assert len(output) == 1
            assert len(output[0].outputs[0].token_ids) > 0
    finally:
        shutil.rmtree(dummy_dir, ignore_errors=True)


@requires_gemma4_transformers
@pytest.mark.skipif(not _model_available("E2B"), reason="gemma-4-E2B-it not found")
def test_e2e_text_e2b_dummy():
    """E2E text generation for E2B (KV sharing + PLE + double-wide MLP)."""
    from tensorrt_llm.llmapi import LLM, SamplingParams

    dummy_dir = _make_dummy_config_dir(MODEL_PATHS["E2B"])
    try:
        llm = LLM(dummy_dir, load_format="dummy", attn_backend="FLASHINFER", dtype="bfloat16")
        with llm:
            output = llm.generate(["Hello"], SamplingParams(max_tokens=4))
            assert len(output) == 1
            assert len(output[0].outputs[0].token_ids) > 0
    finally:
        shutil.rmtree(dummy_dir, ignore_errors=True)


@requires_gemma4_transformers
@pytest.mark.skipif(not _model_available("31B"), reason="gemma-4-31B-it not found")
def test_e2e_text_31b_dummy():
    """E2E text generation for 31B (K=V + hybrid attn + softcap)."""
    from tensorrt_llm.llmapi import LLM, SamplingParams

    dummy_dir = _make_dummy_config_dir(MODEL_PATHS["31B"])
    try:
        llm = LLM(dummy_dir, load_format="dummy", attn_backend="FLASHINFER", dtype="bfloat16")
        with llm:
            output = llm.generate(["Hello"], SamplingParams(max_tokens=4))
            assert len(output) == 1
            assert len(output[0].outputs[0].token_ids) > 0
    finally:
        shutil.rmtree(dummy_dir, ignore_errors=True)


@requires_gemma4_transformers
@pytest.mark.skipif(not _model_available("E4B"), reason="gemma-4-E4B-it not found")
def test_e2e_text_e4b_dummy():
    """E2E text generation for E4B (KV sharing + hybrid attn)."""
    from tensorrt_llm.llmapi import LLM, SamplingParams

    dummy_dir = _make_dummy_config_dir(MODEL_PATHS["E4B"])
    try:
        llm = LLM(dummy_dir, load_format="dummy", attn_backend="FLASHINFER", dtype="bfloat16")
        with llm:
            output = llm.generate(["Hello"], SamplingParams(max_tokens=4))
            assert len(output) == 1
            assert len(output[0].outputs[0].token_ids) > 0
    finally:
        shutil.rmtree(dummy_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Multimodal E2E test
# ---------------------------------------------------------------------------


@requires_gemma4_transformers
@pytest.mark.skipif(not _model_available("26B"), reason="gemma-4-26B-A4B-it not found")
def test_e2e_multimodal_26b_dummy():
    """E2E multimodal: image → vision tower → embedder → LLM → output."""
    import numpy as np
    from PIL import Image

    from tensorrt_llm.llmapi import LLM, SamplingParams

    dummy_dir = _make_dummy_config_dir(MODEL_PATHS["26B"])
    try:
        # Format prompt with image placeholder via tokenizer chat template
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(dummy_dir)
        formatted_prompt = tok.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": "Describe."},
                    ],
                }
            ],
            add_generation_prompt=True,
            tokenize=False,
        )

        llm = LLM(dummy_dir, load_format="dummy", attn_backend="FLASHINFER", dtype="bfloat16")
        with llm:
            img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
            prompt = {
                "prompt": formatted_prompt,
                "multi_modal_data": {"image": [img]},
            }
            output = llm.generate([prompt], SamplingParams(max_tokens=4))
            assert len(output) == 1
            assert len(output[0].outputs[0].token_ids) > 0
    finally:
        shutil.rmtree(dummy_dir, ignore_errors=True)
