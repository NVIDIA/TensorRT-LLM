# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Wan I2V TeaCache using the full pipeline.

Wan 2.1 models are tested with TeaCache enabled:
  - Wan2.1-I2V-14B-480P-Diffusers   480x832
  - Wan2.1-I2V-14B-720P-Diffusers   720x1280

Wan 2.2 is tested to confirm that enabling TeaCache raises a ValueError.

Loads all components (VAE, text encoder, scheduler) and calls pipeline.forward()
so that TeaCache runs on the actual scheduler timesteps.

Run all:
    pytest tests/unittest/_torch/visual_gen/test_wan21_i2v_teacache.py -v -s

Run one model:
    pytest tests/unittest/_torch/visual_gen/test_wan21_i2v_teacache.py -v -s -k wan21_i2v_480p
    pytest tests/unittest/_torch/visual_gen/test_wan21_i2v_teacache.py -v -s -k wan21_i2v_720p
    pytest tests/unittest/_torch/visual_gen/test_wan21_i2v_teacache.py -v -s -k wan22_raises

Override checkpoint paths:
    DIFFUSION_MODEL_PATH_WAN21_I2V_480P=/path/to/480p \\
    DIFFUSION_MODEL_PATH_WAN21_I2V_720P=/path/to/720p \\
    DIFFUSION_MODEL_PATH_WAN22_I2V=/path/to/wan22 \\
        pytest tests/unittest/_torch/visual_gen/test_wan21_i2v_teacache.py -v -s
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import gc
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from tensorrt_llm._torch.visual_gen.config import TeaCacheConfig, VisualGenArgs
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader


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


WAN21_I2V_480P_PATH = _checkpoint(
    "DIFFUSION_MODEL_PATH_WAN21_I2V_480P", "Wan2.1-I2V-14B-480P-Diffusers"
)
WAN21_I2V_720P_PATH = _checkpoint(
    "DIFFUSION_MODEL_PATH_WAN21_I2V_720P", "Wan2.1-I2V-14B-720P-Diffusers"
)

WAN22_I2V_A14B_PATH = _checkpoint("DIFFUSION_MODEL_PATH_WAN22_I2V", "Wan2.2-I2V-A14B-Diffusers")

INFER_NUM_FRAMES = 33  # (33-1)/4+1 = 9 latent frames; smallest realistic shape
INFER_NUM_STEPS = 50  # Required for meaningful cache hits with calibrated coefficients
INFER_SEED = 42


# ============================================================================
# Pipeline fixture factory
# ============================================================================


def _make_pipeline(checkpoint_path: str, use_ret_steps: bool = False):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        pytest.skip(f"Checkpoint not found: {checkpoint_path}")
    args = VisualGenArgs(
        checkpoint_path=checkpoint_path,
        device="cuda",
        dtype="bfloat16",
        cache=TeaCacheConfig(
            teacache_thresh=0.2,
            use_ret_steps=use_ret_steps,
        ),
    )
    pipeline = PipelineLoader(args).load(skip_warmup=True)
    return pipeline


@pytest.fixture
def wan21_i2v_480p_pipeline():
    pipeline = _make_pipeline(WAN21_I2V_480P_PATH)
    yield pipeline
    del pipeline
    torch.cuda.empty_cache()


@pytest.fixture
def wan21_i2v_480p_ret_steps_pipeline():
    pipeline = _make_pipeline(WAN21_I2V_480P_PATH, use_ret_steps=True)
    yield pipeline
    del pipeline
    torch.cuda.empty_cache()


@pytest.fixture
def wan21_i2v_720p_pipeline():
    pipeline = _make_pipeline(WAN21_I2V_720P_PATH)
    yield pipeline
    del pipeline
    torch.cuda.empty_cache()


# ============================================================================
# Shared helpers
# ============================================================================


def _make_test_image(height: int, width: int) -> Image.Image:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = np.linspace(0, 255, height, dtype=np.uint8)[:, None]
    return Image.fromarray(img, mode="RGB")


def _assert_i2v_teacache(
    pipeline,
    height: int,
    width: int,
    model: str = "",
    expected_hit_rate: float = None,
    atol: float = 0.02,
) -> None:
    """Run forward and verify TeaCache produces cache hits (single-stage Wan 2.1 I2V)."""
    test_image = _make_test_image(height, width)

    with torch.no_grad():
        pipeline.forward(
            image=test_image,
            prompt="a cat sitting on a windowsill",
            negative_prompt="",
            height=height,
            width=width,
            num_frames=INFER_NUM_FRAMES,
            num_inference_steps=INFER_NUM_STEPS,
            seed=INFER_SEED,
        )

    stats = pipeline.transformer_cache_backend.get_stats()

    print(f"\n  ===== TeaCache — Wan 2.1 {model} single-stage {height}x{width} =====")
    print(
        f"  transformer: {stats['cached_steps']}/{stats['total_steps']} cached "
        f"({stats['hit_rate']:.1%} hit rate)"
    )
    if expected_hit_rate is not None:
        # Reference hit rates derived from vFly reference runs
        print(f"  expected:    {expected_hit_rate:.1%}  (vFly reference, atol={atol:.0%})")
        delta = stats["hit_rate"] - expected_hit_rate
        print(f"  delta:       {delta:+.1%}")
    print("  ================================================")

    assert stats["total_steps"] == INFER_NUM_STEPS, (
        f"total_steps {stats['total_steps']} != {INFER_NUM_STEPS}"
    )
    assert stats["compute_steps"] + stats["cached_steps"] == stats["total_steps"]
    assert stats["cached_steps"] > 0, (
        f"0 cache hits after {stats['total_steps']} steps. TeaCache is not working. Stats: {stats}"
    )
    if expected_hit_rate is not None:
        assert abs(stats["hit_rate"] - expected_hit_rate) <= atol + 1e-9, (
            f"Hit rate {stats['hit_rate']:.1%} not within {atol:.0%} "
            f"of expected {expected_hit_rate:.1%} (vFly reference)"
        )


# ============================================================================
# Tests
# ============================================================================


@pytest.mark.integration
@pytest.mark.wan_i2v
@pytest.mark.teacache
class TestWan21I2V_480P_TeaCache:
    """Wan2.1-I2V-14B-480P  480x832  single-stage."""

    def test_wan21_i2v_480p_teacache(self, wan21_i2v_480p_pipeline):
        _assert_i2v_teacache(
            wan21_i2v_480p_pipeline, height=480, width=832, model="I2V-14B", expected_hit_rate=0.54
        )

    def test_wan21_i2v_480p_teacache_ret_steps(self, wan21_i2v_480p_ret_steps_pipeline):
        _assert_i2v_teacache(
            wan21_i2v_480p_ret_steps_pipeline,
            height=480,
            width=832,
            model="I2V-14B",
            expected_hit_rate=0.50,
        )


@pytest.mark.integration
@pytest.mark.wan_i2v
@pytest.mark.teacache
class TestWan21I2V_720P_TeaCache:
    """Wan2.1-I2V-14B-720P  720x1280  single-stage."""

    def test_wan21_i2v_720p_teacache(self, wan21_i2v_720p_pipeline):
        _assert_i2v_teacache(
            wan21_i2v_720p_pipeline, height=720, width=1280, model="I2V-14B", expected_hit_rate=0.54
        )


@pytest.mark.integration
@pytest.mark.wan_i2v
@pytest.mark.teacache
class TestWan22_I2V_TeaCacheRaisesError:
    """Wan2.2-I2V-A14B must raise ValueError when TeaCache is enabled."""

    def test_wan22_raises_if_teacache_enabled(self):
        if not os.path.exists(WAN22_I2V_A14B_PATH):
            pytest.skip(
                f"Checkpoint not found: {WAN22_I2V_A14B_PATH} (set DIFFUSION_MODEL_PATH_WAN22_I2V)"
            )
        args = VisualGenArgs(
            checkpoint_path=WAN22_I2V_A14B_PATH,
            device="cuda",
            dtype="bfloat16",
            cache=TeaCacheConfig(),
        )
        with pytest.raises(ValueError, match="TeaCache is not supported for Wan 2\\.2"):
            PipelineLoader(args).load(skip_warmup=True)
