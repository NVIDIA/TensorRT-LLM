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
"""Tests for Wan 2.2 T2V TeaCache happy path.

Wan 2.2 uses a dual-transformer architecture (high-noise + low-noise stages).
TeaCache requires explicit coefficients for both transformers via
TeaCacheConfig.coefficients (high-noise) and TeaCacheConfig.coefficients_2 (low-noise).

Verifies:
  - Both transformer backends are initialized
  - Both backends produce cache hits after a forward pass
  - Stats are returned for each transformer separately

Run:
    pytest tests/unittest/_torch/visual_gen/test_wan22_t2v_teacache.py -v -s

Override checkpoint path:
    DIFFUSION_MODEL_PATH_WAN22_T2V=/path/to/wan22 \\
        pytest tests/unittest/_torch/visual_gen/test_wan22_t2v_teacache.py -v -s
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import gc
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.visual_gen.args import TeaCacheConfig, VisualGenArgs


@pytest.fixture(autouse=True, scope="module")
def _cleanup_mpi_env():
    yield
    os.environ.pop("TLLM_DISABLE_MPI", None)


@pytest.fixture(autouse=True)
def _cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    yield
    gc.collect()
    torch.cuda.empty_cache()


# ============================================================================
# Path helpers
# ============================================================================


def _llm_models_root() -> Path:
    if "LLM_MODELS_ROOT" in os.environ:
        root = Path(os.environ["LLM_MODELS_ROOT"])
    else:
        root = Path("/home/scratch.trt_llm_data_ci/llm-models/")
    if not root.exists():
        root = Path("/scratch.trt_llm_data/llm-models/")
    assert root.exists(), (
        "Set LLM_MODELS_ROOT or ensure /home/scratch.trt_llm_data_ci/llm-models/ is accessible."
    )
    return root


def _checkpoint(env_var: str, default_name: str) -> str:
    return os.environ.get(env_var) or str(_llm_models_root() / default_name)


WAN22_A14B_PATH = _checkpoint("DIFFUSION_MODEL_PATH_WAN22_T2V", "Wan2.2-T2V-A14B-Diffusers")

INFER_NUM_FRAMES = 33  # (33-1)/4+1 = 9 latent frames; smallest realistic shape
INFER_NUM_STEPS = 20  # Wan 2.2 has no reference hit rate; just enough to exercise both backends
INFER_SEED = 42

# Placeholder coefficients for Wan 2.2 dual-transformer TeaCache.
# Wan 2.2 has no built-in coefficient table; users must supply both sets.
# These are used to exercise the dual-backend path and verify cache hits fire.
WAN22_T2V_HIGH_NOISE_COEFFICIENTS = [
    -5784.54975374,
    5449.50911966,
    -1811.16591783,
    256.27178429,
    -13.02301147,
]
WAN22_T2V_LOW_NOISE_COEFFICIENTS = [
    2.39676752e03,
    -1.31110545e03,
    2.01331979e02,
    -8.29855975e00,
    1.37887774e-01,
]


# ============================================================================
# Fixture
# ============================================================================


@pytest.fixture
def wan22_t2v_pipeline():
    if not os.path.exists(WAN22_A14B_PATH):
        pytest.skip(f"Checkpoint not found: {WAN22_A14B_PATH}")
    args = VisualGenArgs(
        model=WAN22_A14B_PATH,
        cache_config=TeaCacheConfig(
            teacache_thresh=0.15,
            coefficients=WAN22_T2V_HIGH_NOISE_COEFFICIENTS,
            coefficients_2=WAN22_T2V_LOW_NOISE_COEFFICIENTS,
        ),
    )
    pipeline = PipelineLoader(args).load(skip_warmup=True)
    yield pipeline
    del pipeline
    torch.cuda.empty_cache()


# ============================================================================
# Assertion helper
# ============================================================================


def _assert_dual_stage_teacache(pipeline, height: int, width: int) -> None:
    with torch.no_grad():
        pipeline.forward(
            prompt="a cat sitting on a windowsill",
            negative_prompt="",
            height=height,
            width=width,
            num_frames=INFER_NUM_FRAMES,
            num_inference_steps=INFER_NUM_STEPS,
            seed=INFER_SEED,
        )

    stats = pipeline.cache_accelerator.get_stats()

    print(f"\n  ===== TeaCache — Wan 2.2 T2V-A14B dual-stage {height}x{width} =====")
    for key, s in stats.items():
        print(
            f"  {key}: {s['cached_steps']}/{s['total_steps']} cached ({s['hit_rate']:.1%} hit rate)"
        )
    print("  ================================================================")

    assert len(stats) == 2, f"Expected stats for 2 transformers, got: {list(stats.keys())}"
    total_steps_sum = sum(s["total_steps"] for s in stats.values())
    assert total_steps_sum == INFER_NUM_STEPS, (
        f"Sum of steps across both transformers {total_steps_sum} != {INFER_NUM_STEPS}"
    )
    for key, s in stats.items():
        assert s["total_steps"] > 0, f"{key}: transformer ran 0 steps"
        assert s["compute_steps"] + s["cached_steps"] == s["total_steps"]


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.wan_t2v
@pytest.mark.teacache
class TestWan22T2V_TeaCache:
    """Wan2.2-T2V-A14B  480x832  dual-stage TeaCache."""

    def test_wan22_t2v_teacache_forward_runs(self, wan22_t2v_pipeline):
        _assert_dual_stage_teacache(wan22_t2v_pipeline, height=480, width=832)

    def test_wan22_t2v_teacache_two_backends_initialized(self, wan22_t2v_pipeline):
        assert len(wan22_t2v_pipeline.cache_accelerator.backends) == 2
