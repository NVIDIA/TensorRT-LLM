# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused CPU tests for the native LTX-2.3 retake pipeline."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.patchifier import VideoLatentPatchifier
from tensorrt_llm._torch.visual_gen.models.ltx23.pipeline_ltx23 import LTX23Pipeline
from tensorrt_llm._torch.visual_gen.models.ltx23.pipeline_ltx23_retake import (
    LTX23RetakePipeline,
    _init_retake_patchified_latents,
    _retake_conditioned_latent_ranges,
    _retake_pixel_window,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY, AutoPipeline
from tensorrt_llm.visual_gen.args import VisualGenArgs

pytestmark = pytest.mark.cpu_only


def test_retake_window_maps_to_unconditioned_latents() -> None:
    pixel_start, pixel_end = _retake_pixel_window(
        start_time=2.9667,
        end_time=3.9333,
        fps=30.0,
        num_frames=209,
    )

    assert (pixel_start, pixel_end) == (89, 118)
    assert _retake_conditioned_latent_ranges(
        pixel_start=pixel_start,
        pixel_end=pixel_end,
        num_frames=209,
        temporal_ratio=8,
    ) == [(0, 12), (16, 27)]


@pytest.mark.parametrize(
    ("start_time", "end_time", "expected"),
    [
        (-1.0, 1.0, (0, 30)),
        (1.0, 99.0, (30, 209)),
        (4.0, 2.0, (120, 120)),
        (2.0, 2.0, (60, 60)),
    ],
)
def test_retake_window_clamps_to_source(
    start_time: float,
    end_time: float,
    expected: tuple[int, int],
) -> None:
    assert _retake_pixel_window(start_time, end_time, fps=30.0, num_frames=209) == expected


@pytest.mark.parametrize(
    ("fps", "num_frames", "error"),
    [
        (0.0, 209, "fps must be positive"),
        (30.0, -1, "num_frames must be non-negative"),
    ],
)
def test_retake_window_rejects_invalid_source_metadata(
    fps: float,
    num_frames: int,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        _retake_pixel_window(1.0, 2.0, fps=fps, num_frames=num_frames)


def test_empty_retake_window_conditions_all_latents() -> None:
    assert _retake_conditioned_latent_ranges(
        pixel_start=60,
        pixel_end=60,
        num_frames=209,
        temporal_ratio=8,
    ) == [(0, 27)]


def test_retake_initial_noise_preserves_conditioned_tokens() -> None:
    patchifier = VideoLatentPatchifier(patch_size=1)
    source = patchifier.patchify(
        torch.arange(1 * 4 * 3 * 2 * 2, dtype=torch.float32).reshape(1, 4, 3, 2, 2)
    )
    noise = torch.randn(source.shape, generator=torch.Generator().manual_seed(42))
    denoise_mask = torch.tensor([[0.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0]])

    initialized = _init_retake_patchified_latents(noise, source, denoise_mask)

    conditioned = denoise_mask[0] == 0
    regenerated = ~conditioned
    assert torch.equal(initialized[:, conditioned], source[:, conditioned])
    assert torch.equal(initialized[:, regenerated], noise[:, regenerated])


def test_retake_pipeline_registration_and_config_schema(monkeypatch) -> None:
    retake_entry = PIPELINE_REGISTRY["LTX23RetakePipeline"]
    assert retake_entry.pipeline_cls is LTX23RetakePipeline
    assert retake_entry.defaults == {}

    ltx23_defaults = PIPELINE_REGISTRY["LTX23Pipeline"].defaults
    assert ltx23_defaults["workflow"] == "generation"

    monkeypatch.setattr(AutoPipeline, "_detect_from_checkpoint", lambda _: "LTX23Pipeline")

    args = VisualGenArgs(
        model="/tmp/ltx23-retake.safetensors",
        pipeline_config={"workflow": "retake"},
    )
    resolved = PipelineLoader(args)._resolve_pipeline_config(args.model)
    assert resolved["workflow"] == "retake"
    assert (
        LTX23Pipeline.resolve_variant(SimpleNamespace(extra_attrs=resolved)) is LTX23RetakePipeline
    )

    config_dir = Path(__file__).resolve().parents[4] / "examples" / "visual_gen" / "configs"
    recipe_args = VisualGenArgs.from_yaml(config_dir / "ltx23-retake-1gpu.yaml")
    assert getattr(recipe_args.quant_config, "quant_algo", None) is None
    assert recipe_args.pipeline_config == {"workflow": "retake"}
