# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Wan T2V TeaCache using the full pipeline.

Wan 2.1 models are tested with TeaCache enabled:
  - Wan2.1-T2V-1.3B-Diffusers   480x832   single-stage
  - Wan2.1-T2V-14B-Diffusers    720x1280  single-stage

Wan 2.2 is tested to confirm that enabling TeaCache raises a ValueError.

Each test loads all pipeline components (VAE, text encoder, scheduler) and calls
pipeline.forward() so TeaCache runs on the actual scheduler timesteps.

Run all:
    pytest tests/unittest/_torch/visual_gen/test_wan21_t2v_teacache.py -v -s

Run one model:
    pytest tests/unittest/_torch/visual_gen/test_wan21_t2v_teacache.py -v -s -k wan21_1_3b
    pytest tests/unittest/_torch/visual_gen/test_wan21_t2v_teacache.py -v -s -k wan21_14b
    pytest tests/unittest/_torch/visual_gen/test_wan21_t2v_teacache.py -v -s -k wan22_raises
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import gc

import pytest
import torch
from utils.llm_data import get_checkpoint

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
# Checkpoints
# ============================================================================

WAN21_1_3B_SUBDIR = "Wan2.1-T2V-1.3B-Diffusers"
WAN21_14B_SUBDIR = "Wan2.1-T2V-14B-Diffusers"
WAN22_A14B_SUBDIR = "Wan2.2-T2V-A14B-Diffusers"

INFER_NUM_FRAMES = 33  # (33-1)/4+1 = 9 latent frames; smallest realistic shape
INFER_NUM_STEPS = 50  # Required for meaningful cache hits with calibrated coefficients
INFER_SEED = 42


# ============================================================================
# Pipeline fixture factory
# ============================================================================


def _make_pipeline(checkpoint_path: str, use_ret_steps: bool = False):
    args = VisualGenArgs(
        model=checkpoint_path,
        cache_config=TeaCacheConfig(
            teacache_thresh=0.2,
            use_ret_steps=use_ret_steps,
        ),
    )
    pipeline = PipelineLoader(args).load(skip_warmup=True)
    return pipeline


@pytest.fixture
def wan21_1_3b_pipeline():
    pipeline = _make_pipeline(get_checkpoint(WAN21_1_3B_SUBDIR))
    yield pipeline
    del pipeline
    torch.cuda.empty_cache()


@pytest.fixture
def wan21_1_3b_ret_steps_pipeline():
    pipeline = _make_pipeline(get_checkpoint(WAN21_1_3B_SUBDIR), use_ret_steps=True)
    yield pipeline
    del pipeline
    torch.cuda.empty_cache()


@pytest.fixture
def wan21_14b_pipeline():
    pipeline = _make_pipeline(get_checkpoint(WAN21_14B_SUBDIR))
    yield pipeline
    del pipeline
    torch.cuda.empty_cache()


# ============================================================================
# Shared assertion helper
# ============================================================================


def _assert_single_stage_teacache(
    pipeline,
    height: int,
    width: int,
    model: str = "",
    expected_hit_rate: float = None,
    atol: float = 0.02,
) -> None:
    """Run forward and verify TeaCache produces cache hits (single-stage Wan 2.1)."""
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
    print("  ================================================================")

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
@pytest.mark.wan_t2v
@pytest.mark.teacache
class TestWan21_1_3B_TeaCache:
    """Wan2.1-T2V-1.3B  480x832  single-stage."""

    def test_wan21_1_3b_teacache(self, wan21_1_3b_pipeline):
        _assert_single_stage_teacache(
            wan21_1_3b_pipeline, height=480, width=832, model="T2V-1.3B", expected_hit_rate=0.68
        )

    def test_wan21_1_3b_teacache_ret_steps(self, wan21_1_3b_ret_steps_pipeline):
        _assert_single_stage_teacache(
            wan21_1_3b_ret_steps_pipeline,
            height=480,
            width=832,
            model="T2V-1.3B",
            expected_hit_rate=0.72,
        )


@pytest.mark.integration
@pytest.mark.wan_t2v
@pytest.mark.teacache
class TestWan21_14B_TeaCache:
    """Wan2.1-T2V-14B  720x1280  single-stage."""

    def test_wan21_14b_teacache(self, wan21_14b_pipeline):
        _assert_single_stage_teacache(
            wan21_14b_pipeline, height=720, width=1280, model="T2V-14B", expected_hit_rate=0.48
        )


@pytest.mark.integration
@pytest.mark.wan_t2v
@pytest.mark.teacache
class TestWan22_T2V_TeaCacheRaisesError:
    """Wan2.2-T2V-A14B must raise ValueError when TeaCache is enabled."""

    def test_wan22_raises_if_teacache_enabled(self):
        args = VisualGenArgs(
            model=get_checkpoint(WAN22_A14B_SUBDIR),
            cache_config=TeaCacheConfig(),
        )
        with pytest.raises(ValueError, match=r"Wan 2\.2 TeaCache requires explicit"):
            PipelineLoader(args).load(skip_warmup=True)
