# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused CPU tests for the native LTX-2 retake pipeline."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.patchifier import VideoLatentPatchifier
from tensorrt_llm._torch.visual_gen.models.ltx2.ltx2_core.rope import LTXRopeType
from tensorrt_llm._torch.visual_gen.models.ltx2.pipeline_ltx2 import LTX2Pipeline
from tensorrt_llm._torch.visual_gen.models.ltx2_retake.ltx2_retake_core.connector import (
    _connector_from_config,
)
from tensorrt_llm._torch.visual_gen.models.ltx2_retake.ltx2_retake_core.transformer_args import (
    MultiModalTransformerArgsPreprocessor,
)
from tensorrt_llm._torch.visual_gen.models.ltx2_retake.pipeline_ltx2_retake import (
    LTX2RetakePipeline,
    _init_retake_patchified_latents,
    _retake_conditioned_latent_ranges,
    _retake_pixel_window,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm._torch.visual_gen.pipeline_registry import PIPELINE_REGISTRY, AutoPipeline
from tensorrt_llm.visual_gen.args import VisualGenArgs

pytestmark = pytest.mark.cpu_only


class _RecordingAdaLN:
    def __init__(self, width: int) -> None:
        self.width = width
        self.inputs: list[torch.Tensor] = []

    def __call__(
        self, timestep: torch.Tensor, *, hidden_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.inputs.append(timestep.clone())
        output = timestep.to(hidden_dtype).unsqueeze(-1).expand(-1, self.width).contiguous()
        return output, output


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


def test_connector_uses_normalized_connector_rope_type() -> None:
    connector = _connector_from_config(
        {
            "transformer": {
                "connector_num_attention_heads": 1,
                "connector_attention_head_dim": 2,
                "connector_num_layers": 0,
                "connector_rope_type": "SPLIT",
            }
        }
    )

    assert connector.rope_type is LTXRopeType.SPLIT


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


def test_cross_attention_gate_uses_cross_modality_sigma() -> None:
    scale_shift_adaln = _RecordingAdaLN(width=8)
    gate_adaln = _RecordingAdaLN(width=4)
    preprocessor = MultiModalTransformerArgsPreprocessor.__new__(
        MultiModalTransformerArgsPreprocessor
    )
    preprocessor.cross_scale_shift_adaln = scale_shift_adaln
    preprocessor.cross_gate_adaln = gate_adaln
    preprocessor.av_ca_timestep_scale_multiplier = 10

    modality_timesteps = torch.tensor(
        [[0.0, 0.25, 0.5], [0.1, 0.2, 0.3]],
        dtype=torch.bfloat16,
    )
    cross_modality_sigma = torch.tensor([0.25, 0.75], dtype=torch.bfloat16)
    _, gate = preprocessor._prepare_cross_attention_timestep(
        modality_timesteps=modality_timesteps,
        cross_modality_sigma=cross_modality_sigma,
        timestep_scale_multiplier=1000,
        batch_size=2,
        hidden_dtype=torch.bfloat16,
    )

    torch.testing.assert_close(gate_adaln.inputs[0], cross_modality_sigma * 10)
    assert gate.shape == (2, 3, 4)


def test_retake_pipeline_registration_and_config_schema(monkeypatch) -> None:
    retake_entry = PIPELINE_REGISTRY["LTX2RetakePipeline"]
    assert retake_entry.pipeline_cls is LTX2RetakePipeline
    assert retake_entry.defaults == {}

    ltx2_defaults = PIPELINE_REGISTRY["LTX2Pipeline"].defaults
    assert ltx2_defaults["workflow"] == "generation"

    monkeypatch.setattr(AutoPipeline, "_detect_from_checkpoint", lambda _: "LTX2Pipeline")

    args = VisualGenArgs(
        model="/tmp/ltx2-retake.safetensors",
        pipeline_config={"workflow": "retake"},
    )
    resolved = PipelineLoader(args)._resolve_pipeline_config(args.model)
    assert resolved["workflow"] == "retake"
    assert LTX2Pipeline.resolve_variant(SimpleNamespace(extra_attrs=resolved)) is LTX2RetakePipeline

    config_dir = Path(__file__).resolve().parents[4] / "examples" / "visual_gen" / "configs"
    recipe_args = VisualGenArgs.from_yaml(config_dir / "ltx2-retake-1gpu.yaml")
    assert getattr(recipe_args.quant_config, "quant_algo", None) is None
    assert recipe_args.pipeline_config == {"workflow": "retake"}
