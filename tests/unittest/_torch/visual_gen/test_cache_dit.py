# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for Cache-DiT in visual generation.

Wan 2.2 step-split logic is covered with small CPU-side tests. Wan and FLUX integration
tests run on GPU only when cache_dit is installed, CUDA is available, and a checkpoint
can be resolved (TRTLLM_CACHE_DIT_* env vars or the fallbacks inside each test).
"""

from __future__ import annotations

import contextlib
import importlib.util
import logging
import os
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.visual_gen.cache.cache_dit_enablers import split_wan22_inference_steps

# ---------------------------------------------------------------------------
# Integration prerequisites
# ---------------------------------------------------------------------------

_CACHE_DIT_SPEC = importlib.util.find_spec("cache_dit")

requires_cache_dit = pytest.mark.skipif(
    _CACHE_DIT_SPEC is None,
    reason="optional dependency cache-dit not installed (pip install cache-dit)",
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

_WAN_SUBPATH = "Wan2.1-T2V-1.3B-Diffusers"
_FLUX_SUBPATH = "FLUX.1-dev"
# Default NFS layout used on CI runners (override via TRTLLM_CACHE_DIT_*_CHECKPOINT).
_CI_DEFAULT_LLM_MODELS = "/home/scratch.trt_llm_data_ci/llm-models"
_DEFAULT_WAN_CHECKPOINT = os.path.join(_CI_DEFAULT_LLM_MODELS, _WAN_SUBPATH)
_DEFAULT_FLUX_CHECKPOINT = os.path.join(_CI_DEFAULT_LLM_MODELS, _FLUX_SUBPATH)


def _resolve_wan_checkpoint() -> str | None:
    """Wan 2.1 1.3B: explicit env, then CI default tree, then LLM_MODELS_ROOT."""
    explicit = os.environ.get("TRTLLM_CACHE_DIT_WAN_CHECKPOINT", "").strip()
    if explicit and os.path.isdir(explicit):
        return os.path.abspath(explicit)
    if os.path.isdir(_DEFAULT_WAN_CHECKPOINT):
        return os.path.abspath(_DEFAULT_WAN_CHECKPOINT)
    root = os.environ.get("LLM_MODELS_ROOT", "").strip()
    if root:
        cand = os.path.join(root, _WAN_SUBPATH)
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    return None


def _resolve_flux_checkpoint() -> str | None:
    """FLUX.1 dev tree: explicit env, FLUX1_MODEL_PATH, CI default, then LLM_MODELS_ROOT."""
    explicit = os.environ.get("TRTLLM_CACHE_DIT_FLUX_CHECKPOINT", "").strip()
    if explicit and os.path.isdir(explicit):
        return os.path.abspath(explicit)
    flux1 = os.environ.get("FLUX1_MODEL_PATH", "").strip()
    if flux1 and os.path.isdir(flux1):
        return os.path.abspath(flux1)
    if os.path.isdir(_DEFAULT_FLUX_CHECKPOINT):
        return os.path.abspath(_DEFAULT_FLUX_CHECKPOINT)
    root = os.environ.get("LLM_MODELS_ROOT", "").strip()
    if root:
        cand = os.path.join(root, _FLUX_SUBPATH)
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    return None


@contextlib.contextmanager
def _suppress_stdlib_logging_for_cache_dit():
    """Silence cache_dit's stdlib logging under pytest capture (closed stdout/stderr).

    Wrap the full Cache-DiT window: load (wrap), denoise (get_stats), and teardown
    (disable_cache)—not only right after forward.

    Do not nest this context manager: logging.disable(NOTSET) in finally resets the
    process-wide disable level and would undo an outer scope.
    """
    logging.disable(logging.CRITICAL + 1)
    try:
        yield
    finally:
        logging.disable(logging.NOTSET)


def _total_accumulated_cached_steps(stats: dict) -> int:
    """Sum accumulated_cached_steps across CacheDiTAccelerator.get_stats() values."""
    total = 0
    for _key, val in stats.items():
        entries = val if isinstance(val, list) else [val]
        for entry in entries:
            if hasattr(entry, "accumulated_cached_steps"):
                total += int(entry.accumulated_cached_steps)
    return total


# ---------------------------------------------------------------------------
# 1) Wan 2.2 high/low step split (no mocks)
# ---------------------------------------------------------------------------


class _FixedTimestepScheduler:
    """Scheduler stub: set_timesteps is a no-op; timesteps stay fixed for the test."""

    def __init__(self, timesteps: torch.Tensor):
        self.timesteps = timesteps
        self.config = SimpleNamespace(num_train_timesteps=1000)

    def set_timesteps(self, n: int, device=None) -> None:
        del n, device


class _MiniWan22Pipeline:
    """Minimal object satisfying the split_wan22_inference_steps protocol."""

    def __init__(self, boundary_ratio: float | None, timestep_values: list[float]):
        self.boundary_ratio = boundary_ratio
        self.scheduler = _FixedTimestepScheduler(torch.tensor(timestep_values, dtype=torch.float32))
        self.transformer = torch.nn.Linear(1, 1)


class TestSplitWan22InferenceSteps:
    def test_counts_high_noise_at_default_boundary(self):
        pipeline = _MiniWan22Pipeline(
            0.42,
            [999.0, 800.0, 400.0, 200.0, 50.0, 10.0],
        )
        high, low = split_wan22_inference_steps(pipeline, 6)
        assert high == 2
        assert low == 4

    def test_boundary_ratio_none_counts_all_as_high(self):
        pipeline = _MiniWan22Pipeline(None, [500.0, 400.0, 300.0])
        high, low = split_wan22_inference_steps(pipeline, 3)
        assert high == 3
        assert low == 0

    def test_boundary_zero_all_timesteps_high(self):
        pipeline = _MiniWan22Pipeline(0.0, [999.0, 1.0])
        high, low = split_wan22_inference_steps(pipeline, 2)
        assert high == 2
        assert low == 0

    def test_boundary_one_all_timesteps_low(self):
        pipeline = _MiniWan22Pipeline(1.0, [999.0, 500.0])
        high, low = split_wan22_inference_steps(pipeline, 2)
        assert high == 0
        assert low == 2


# ---------------------------------------------------------------------------
# 2) Real pipeline + real cache-dit
# ---------------------------------------------------------------------------


@requires_cache_dit
@requires_cuda
class TestCacheDiTRealPipelineForward:
    """Wan and FLUX.1 use the CI llm-models tree when a checkpoint is present.

    Each test calls _teardown_cache_dit in finally; cache_dit treats caching as
    process-global, so a second loaded pipeline would otherwise skip setup.
    """

    @staticmethod
    def _teardown_cache_dit(pipeline: object) -> None:
        """cache_dit keeps process-global state; unwrap so a later pipeline can enable_cache.

        Call only while _suppress_stdlib_logging_for_cache_dit is active (disable_cache
        also logs); do not add a nested suppress here.
        """
        acc = getattr(pipeline, "cache_accelerator", None)
        if acc is not None and acc.is_enabled():
            acc.unwrap()
        if hasattr(pipeline, "cache_accelerator"):
            pipeline.cache_accelerator = None

    @staticmethod
    def _load_visual_gen_pipeline(checkpoint_dir: str):
        from tensorrt_llm._torch.visual_gen.config import (
            CacheDiTConfig,
            TorchCompileConfig,
            VisualGenArgs,
        )
        from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader

        args = VisualGenArgs(
            checkpoint_path=checkpoint_dir,
            cache_backend="cache_dit",
            cache_dit=CacheDiTConfig(
                max_warmup_steps=0,
                Fn_compute_blocks=1,
                Bn_compute_blocks=0,
                residual_diff_threshold=0.25,
            ),
            torch_compile=TorchCompileConfig(
                enable_torch_compile=False,
                enable_autotune=False,
            ),
            skip_warmup=True,
        )
        loader = PipelineLoader(args)
        return loader.load(skip_warmup=True)

    def test_wan_cache_dit_skips_blocks_after_forward(self):
        ckpt = _resolve_wan_checkpoint()
        if ckpt is None:
            pytest.skip(
                "Wan 2.1 1.3B not found: set TRTLLM_CACHE_DIT_WAN_CHECKPOINT, "
                f"install under {_DEFAULT_WAN_CHECKPOINT}, "
                f"or under $LLM_MODELS_ROOT/{_WAN_SUBPATH}"
            )

        pipeline = None
        with _suppress_stdlib_logging_for_cache_dit():
            try:
                pipeline = self._load_visual_gen_pipeline(ckpt)
                assert pipeline.cache_accelerator is not None
                assert pipeline.cache_accelerator.is_enabled()

                with torch.inference_mode():
                    pipeline.forward(
                        prompt="cache dit validation",
                        negative_prompt="",
                        height=480,
                        width=832,
                        num_frames=33,
                        num_inference_steps=16,
                        guidance_scale=5.0,
                        seed=0,
                        max_sequence_length=256,
                    )

                stats = pipeline.cache_accelerator.get_stats()
                cached = _total_accumulated_cached_steps(stats)
                assert cached > 0, (
                    "Expected Cache-DiT accumulated_cached_steps > 0 after forward; "
                    f"stats={stats!r}. Try more steps or a looser residual_diff_threshold."
                )
            finally:
                if pipeline is not None:
                    self._teardown_cache_dit(pipeline)

    def test_flux_cache_dit_skips_blocks_after_forward(self):
        ckpt = _resolve_flux_checkpoint()
        if ckpt is None:
            pytest.skip(
                "FLUX.1-dev not found: set TRTLLM_CACHE_DIT_FLUX_CHECKPOINT or FLUX1_MODEL_PATH, "
                f"install under {_DEFAULT_FLUX_CHECKPOINT}, "
                f"or under $LLM_MODELS_ROOT/{_FLUX_SUBPATH}"
            )

        pipeline = None
        with _suppress_stdlib_logging_for_cache_dit():
            try:
                pipeline = self._load_visual_gen_pipeline(ckpt)
                name = pipeline.__class__.__name__
                if name not in ("FluxPipeline", "Flux2Pipeline"):
                    pytest.skip(f"Checkpoint resolved to {name}, not a FLUX visual_gen pipeline")

                assert pipeline.cache_accelerator is not None
                assert pipeline.cache_accelerator.is_enabled()

                with torch.inference_mode():
                    pipeline.forward(
                        prompt="cache dit validation",
                        height=512,
                        width=512,
                        num_inference_steps=16,
                        guidance_scale=3.5,
                        seed=0,
                        max_sequence_length=256,
                    )

                stats = pipeline.cache_accelerator.get_stats()
                cached = _total_accumulated_cached_steps(stats)
                assert cached > 0, (
                    "Expected Cache-DiT accumulated_cached_steps > 0 after forward; "
                    f"stats={stats!r}. Try more steps or a looser residual_diff_threshold."
                )
            finally:
                if pipeline is not None:
                    self._teardown_cache_dit(pipeline)
