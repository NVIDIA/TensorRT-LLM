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
"""Confirm that user-configured TeaCache coefficients affect the cache rate on Wan 2.1 1.3B.

Loads model weights and runs two forward passes back-to-back, each with a
different user-supplied TeaCacheConfig.coefficients list, then prints and compares
the cached step counts to confirm the user override is respected.

Run:
    pytest tests/unittest/_torch/visual_gen/test_wan13b_teacache_coefficients.py -v -s

Override checkpoint:
    DIFFUSION_MODEL_PATH_WAN21_1_3B=/path/to/weights \\
        pytest tests/unittest/_torch/visual_gen/test_wan13b_teacache_coefficients.py -v -s
"""

import gc
import os

os.environ["TLLM_DISABLE_MPI"] = "1"

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


WAN21_1_3B_PATH = os.environ.get(
    "DIFFUSION_MODEL_PATH_WAN21_1_3B",
    str(_llm_models_root() / "Wan2.1-T2V-1.3B-Diffusers"),
)

PROMPT = "a cat sitting on a windowsill"
HEIGHT, WIDTH = 480, 832
NUM_FRAMES = 33  # (33-1)/4+1 = 9 latent frames; smallest realistic shape
NUM_STEPS = 50  # enough steps for coefficients to produce meaningful cache hits
SEED = 42

# Two user-supplied coefficient lists passed explicitly via TeaCacheConfig.coefficients,
# bypassing the built-in variant-lookup table entirely, testing the user-override code path.

COEFFICIENTS_CALIBRATED = [
    2.39676752e03,
    -1.31110545e03,
    2.01331979e02,
    -8.29855975e00,
    1.37887774e-01,
]

COEFFICIENTS_IDENTITY_LINEAR = [
    1.0,
    0.0,
]


def _run_forward(coefficients: list, thresh: float, label: str) -> dict:
    """Load the pipeline with the given user-supplied coefficients, run one forward pass."""
    if not os.path.exists(WAN21_1_3B_PATH):
        pytest.skip(f"Checkpoint not found: {WAN21_1_3B_PATH}")

    args = VisualGenArgs(
        model=WAN21_1_3B_PATH,
        cache_config=TeaCacheConfig(
            teacache_thresh=thresh,
            coefficients=coefficients,
        ),
    )
    pipeline = PipelineLoader(args).load(skip_warmup=True)
    try:
        with torch.no_grad():
            pipeline.forward(
                prompt=PROMPT,
                negative_prompt="",
                height=HEIGHT,
                width=WIDTH,
                num_frames=NUM_FRAMES,
                num_inference_steps=NUM_STEPS,
                seed=SEED,
            )
        stats = pipeline.cache_accelerator.get_stats()
    finally:
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()

    print(
        f"  {label:<30s}  cached={stats['cached_steps']:>3}/{stats['total_steps']}  "
        f"({stats['hit_rate']:.1%} cache rate)"
    )
    return stats


@pytest.mark.integration
@pytest.mark.wan_t2v
@pytest.mark.teacache
class TestWan13BUserCoefficientsAffectCacheRate:
    """User-supplied TeaCacheConfig.coefficients override the built-in table and change cache rate."""

    def test_different_user_coefficients_produce_different_cache_rates(self):
        print(f"\n  {'coefficient set':<30s}  {'cached / total':>15}  hit rate")
        print(f"  {'-' * 65}")

        stats_a = _run_forward(
            COEFFICIENTS_CALIBRATED,
            thresh=0.2,
            label="calibrated 1.3B standard",
        )
        stats_b = _run_forward(
            COEFFICIENTS_IDENTITY_LINEAR,
            thresh=0.2,
            label="identity linear [1, 0]",
        )

        print(f"  {'-' * 65}")
        print(
            f"  difference in cached steps: "
            f"{abs(stats_a['cached_steps'] - stats_b['cached_steps'])}"
        )

        assert stats_a["cached_steps"] != stats_b["cached_steps"], (
            f"Both coefficient sets produced {stats_a['cached_steps']} cached steps — "
            f"expected user-configured coefficients to produce different cache rates."
        )
