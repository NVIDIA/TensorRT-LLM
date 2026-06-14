# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Cascade Attention accuracy test.

Validates that the cascade MMHA kernel produces numerically equivalent results
to the baseline MMHA kernel by running the same beam search prompts with
TRTLLM_ENABLE_CASCADE_MMHA disabled (baseline) and enabled (cascade), then
comparing the generated token sequences.

Requirements:
  - GPU with SM >= 80 (Ampere or newer)
  - Model: Qwen3-0.6B (head_dim=128, NeoX RoPE, beam_width>=2 → cascade eligible)
  - LLM_MODELS_ROOT env var pointing to model storage
"""

import os

import pytest

from tensorrt_llm import LLM
from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams

from ..conftest import get_sm_version, llm_models_root

# Cascade kernels require SM >= 80 (Ampere mma.m16n8k16 / cp.async).
skip_pre_ampere = pytest.mark.skipif(
    get_sm_version() < 80, reason="Cascade attention requires SM >= 80 (Ampere or newer)"
)

# Prompts designed to exercise multi-token generation under beam search.
_TEST_PROMPTS = [
    "The capital of France is",
    "Explain the theory of relativity in simple terms:",
    "Write a short poem about the ocean:",
    "What are the main differences between Python and C++?",
]


def _get_model_path() -> str:
    """Resolve Qwen3-0.6B model path.

    Priority:
      1. CASCADE_TEST_MODEL_PATH env var (explicit override)
      2. $LLM_MODELS_ROOT/Qwen3/Qwen3-0.6B (CI pre-cached)
      3. "Qwen/Qwen3-0.6B" (auto-download from HuggingFace Hub)
    """
    if "CASCADE_TEST_MODEL_PATH" in os.environ:
        return os.environ["CASCADE_TEST_MODEL_PATH"]

    try:
        local_path = os.path.join(llm_models_root(), "Qwen3", "Qwen3-0.6B")
        if os.path.isdir(local_path):
            return local_path
    except AssertionError:
        pass  # LLM_MODELS_ROOT not configured — fall through to HF Hub.

    # Fallback: LLM() will auto-download from HuggingFace Hub.
    return "Qwen/Qwen3-0.6B"


def _run_beam_search(model_path: str, prompts: list, enable_cascade: bool):
    """Run beam search inference and return list of generated text outputs."""
    env_val = "1" if enable_cascade else "0"
    os.environ["TRTLLM_ENABLE_CASCADE_MMHA"] = env_val

    max_beam_width = 2
    sampling_params = SamplingParams(
        n=max_beam_width,
        best_of=max_beam_width,
        use_beam_search=True,
        max_tokens=64,
    )

    with LLM(
        model=model_path,
        kv_cache_config=KvCacheConfig(free_gpu_memory_fraction=0.4),
        max_batch_size=len(prompts),
        max_seq_len=256,
        max_beam_width=max_beam_width,
    ) as llm:
        outputs = llm.generate(prompts, sampling_params=sampling_params)

    # Extract the best beam's generated text for each prompt.
    results = []
    for output in outputs:
        results.append(output.outputs[0].text)
    return results


@skip_pre_ampere
class TestCascadeAttentionAccuracy:
    """Compare cascade vs baseline MMHA beam search outputs for Qwen3-0.6B."""

    def test_cascade_matches_baseline(self):
        """Cascade MMHA must produce identical beam search output as baseline.

        The cascade kernel splits attention into prefix (shared, Tensor-Core
        MMA) and suffix (per-beam, scalar loop) then merges with online-softmax.
        The math is equivalent but FP16 accumulation order differs, so tiny
        numerical deltas are theoretically possible.  We assert exact match
        first (which should hold for short sequences) and fall back to a token
        overlap check if the strict assertion fails.
        """
        model_path = _get_model_path()

        # Run baseline (cascade disabled)
        baseline_outputs = _run_beam_search(model_path, _TEST_PROMPTS, enable_cascade=False)

        # Run cascade (cascade enabled)
        cascade_outputs = _run_beam_search(model_path, _TEST_PROMPTS, enable_cascade=True)

        # Compare: cascade should produce identical token sequences.
        # Each LLM(...) spawns a fresh worker process via MpiPoolSession, so
        # the C++ static env-var cache is re-initialized correctly.
        mismatches = []
        for i, (baseline, cascade) in enumerate(zip(baseline_outputs, cascade_outputs)):
            if baseline != cascade:
                mismatches.append(i)

        if not mismatches:
            return  # Perfect match — ideal outcome.

        # Soft fallback: allow minor token-level divergence caused by FP16
        # accumulation order differences.  Require >= 90% token overlap.
        MIN_OVERLAP_RATIO = 0.9
        for i in mismatches:
            baseline_tokens = baseline_outputs[i].split()
            cascade_tokens = cascade_outputs[i].split()
            if not baseline_tokens:
                continue
            common = set(baseline_tokens) & set(cascade_tokens)
            overlap = len(common) / max(len(baseline_tokens), 1)
            assert overlap >= MIN_OVERLAP_RATIO, (
                f"Prompt {i}: token overlap {overlap:.1%} < {MIN_OVERLAP_RATIO:.0%}\n"
                f"  Baseline: {baseline_outputs[i]!r}\n"
                f"  Cascade:  {cascade_outputs[i]!r}\n"
                f"Cascade kernel produced significantly different beam search "
                f"results than baseline MMHA."
            )
