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

"""Synthetic pipeline-level tests for MiniMax-H3."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from diffusers import MiniMaxH3Scheduler

from tensorrt_llm._torch.visual_gen.config import DiffusionPipelineConfig
from tensorrt_llm._torch.visual_gen.models.minimax_h3_v2.default_pipeline.pipeline import (
    MiniMaxH3Pipeline,
)
from tensorrt_llm._torch.visual_gen.models.minimax_h3_v2.fasth3.pipeline import FastH3Pipeline
from tensorrt_llm._torch.visual_gen.models.minimax_h3_v2.pipeline_registry import route
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent
from tensorrt_llm.visual_gen.args import AttentionConfig

from .._pipeline_fixtures import SyntheticFastH3Pipeline, SyntheticH3Pipeline

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "manifest, expected_steps",
    [
        (None, 50),
        ({"schema_version": "fasth3-inference-contract-v1", "num_inference_steps": 9}, 9),
        ({"schema_version": "fasth3-inference-contract-v1"}, 50),
        (
            {
                "schema_version": "fasth3-inference-contract-v1",
                "dmd_denoising_steps": list(range(999, 972, -1)),
            },
            28,
        ),
        (
            {
                "schema_version": "fasth3-inference-contract-v1",
                "dmd_denoising_steps": [999, 500],
            },
            3,
        ),
        (
            {
                "schema_version": "fasth3-inference-contract-v1",
                "dmd_denoising_steps": [999, 500],
                "num_inference_steps": 9,
            },
            9,
        ),
        ({"schema_version": "other-model-v1", "num_inference_steps": 5}, 50),
    ],
)
def test_fasth3_generation_steps_follow_inference_contract(
    tmp_path: Path, manifest: dict | None, expected_steps: int
) -> None:
    if manifest is not None:
        (tmp_path / "fastvideo_inference.json").write_text(json.dumps(manifest))
    config = SimpleNamespace(
        attention=AttentionConfig(),
        model_configs={},
        attention_metadata_state=None,
        primary_pretrained_config=SimpleNamespace(),
    )
    target = route(config, str(tmp_path))
    pipeline = SyntheticFastH3Pipeline() if target is FastH3Pipeline else SyntheticH3Pipeline()
    pipeline.pipeline_config = config
    pipeline.load_standard_components(
        str(tmp_path), torch.device("cpu"), skip_components=list(PipelineComponent)
    )

    assert pipeline.default_generation_params["num_inference_steps"] == expected_steps


@pytest.mark.parametrize("fasth3", [False, True])
def test_warmup_uses_model_schedule(fasth3: bool) -> None:
    pipeline = SyntheticFastH3Pipeline() if fasth3 else SyntheticH3Pipeline()
    pipeline.pipeline_config.compilation = SimpleNamespace(resolutions=[(32, 32)], num_frames=[124])
    if fasth3:
        pipeline.scheduler = MiniMaxH3Scheduler(shift=10.0)
        pipeline._fastvideo_inference = {
            "dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125],
            "video_scheduler_shift": 10.0,
            "audio_scheduler_shift": 3.0,
        }
    pipeline.post_load_weights()
    shapes, steps = pipeline.resolve_warmup_plan()
    assert shapes == [(32, 32, 124)]
    assert steps == (9 if fasth3 else 2)

    for height, width, num_frames in shapes:
        pipeline._run_warmup(height, width, num_frames, steps)

    assert pipeline.transformer.forward_calls == steps - 1
    assert len(pipeline.scheduler.timesteps) == steps - 1
    assert len(pipeline.audio_scheduler.timesteps) == steps - 1


def test_fasth3_v2_trained_schedule() -> None:
    pipeline = SyntheticFastH3Pipeline()
    pipeline.scheduler = MiniMaxH3Scheduler(shift=10.0)
    pipeline.audio_scheduler = MiniMaxH3Scheduler(shift=3.0)
    pipeline._fastvideo_inference = {
        "dmd_denoising_steps": [999, 874, 749, 624, 500, 375, 250, 125],
        "video_scheduler_shift": 10.0,
        "audio_scheduler_shift": 3.0,
    }
    pipeline.post_load_weights()
    pipeline._set_timesteps(9)
    base = torch.tensor([0.999, 0.874, 0.749, 0.624, 0.5, 0.375, 0.25, 0.125, 0.0])
    for scheduler, shift in ((pipeline.scheduler, 10.0), (pipeline.audio_scheduler, 3.0)):
        expected = shift * base / (1 + (shift - 1) * base)
        torch.testing.assert_close(scheduler.sigmas.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(scheduler.timesteps.cpu(), 1 - expected[:-1], rtol=0, atol=0)
        assert len(scheduler.timesteps) == 8
    with pytest.raises(ValueError, match="9 sigma-grid points"):
        pipeline._set_timesteps(8)
    pipeline.scheduler.set_shift(12.0)
    with pytest.raises(ValueError, match="video_scheduler_shift"):
        pipeline._set_timesteps(9)
    pipeline._fastvideo_inference = {}
    pipeline._set_timesteps(50)
    assert len(pipeline.scheduler.timesteps) == 49


@pytest.mark.parametrize("steps", [[], [999, 999], [999, 1000], [1001, 500], [999, 0]])
def test_fasth3_rejects_invalid_dmd_steps(steps: list[int]) -> None:
    pipeline = SyntheticFastH3Pipeline()
    pipeline._fastvideo_inference = {"dmd_denoising_steps": steps}
    with pytest.raises(ValueError, match="strictly decreasing"):
        pipeline._set_timesteps(len(steps) + 1)


@pytest.mark.parametrize(
    "attention_kwargs,expected_backend,expected_sparsity",
    [
        ({}, "TRTLLM", 0.8),
        ({"backend": "TRTLLM"}, "TRTLLM", 0.8),
        ({"backend": "CUTEDSL"}, "CUTEDSL", 0.8),
        (
            {
                "backend": "CUTEDSL",
                "sparse_attention_config": {"algorithm": "vsa", "vsa_sparsity": 0.5},
            },
            "CUTEDSL",
            0.5,
        ),
    ],
)
def test_fasth3_manifest_metadata_controls_pipeline_defaults(
    tmp_path: Path, attention_kwargs: dict, expected_backend: str, expected_sparsity: float
) -> None:
    (tmp_path / "transformer").mkdir()
    (tmp_path / "transformer" / "config.json").write_text(
        json.dumps({"hidden_size": 32}),
        encoding="utf-8",
    )
    (tmp_path / "modular_model_index.json").write_text(
        json.dumps(
            {
                "_class_name": "MiniMaxH3ModularPipeline",
                "transformer": ["diffusers", "MiniMaxH3Transformer3DModel"],
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "fastvideo_inference.json").write_text(
        json.dumps(
            {
                "schema_version": "fasth3-inference-contract-v1",
                "model_id": "FastVideo/FastVideo-FastH3-8-Step-V2",
                "num_inference_steps": 9,
                "vsa_sparsity": 0.8,
            }
        ),
        encoding="utf-8",
    )

    config = DiffusionPipelineConfig.from_pretrained(str(tmp_path))
    config.attention = AttentionConfig(**attention_kwargs)
    assert route(config, str(tmp_path)) is FastH3Pipeline
    assert config.attention.backend == expected_backend
    assert config.attention.sparse_attention_config.algorithm == "vsa"
    assert config.attention.sparse_attention_config.vsa_sparsity == expected_sparsity
    assert config.primary_model_config.attention == config.attention
    if expected_backend == "TRTLLM":
        assert config.attention_metadata_state == {"metadata_cache": {}}
        assert config.primary_model_config.attention_metadata_state == {"metadata_cache": {}}
        assert (
            config.primary_model_config.attention_metadata_state
            is not config.attention_metadata_state
        )

    assert "fastvideo_inference" not in config.extra_attrs
    pipeline = SyntheticFastH3Pipeline()
    pipeline.pipeline_config = config
    pipeline.load_standard_components(
        str(tmp_path), torch.device("cpu"), skip_components=list(PipelineComponent)
    )
    assert pipeline.default_generation_params["num_inference_steps"] == 9

    config.attention = AttentionConfig(backend="VANILLA")
    with pytest.raises(ValueError, match="FastH3 requires VSA"):
        route(config, str(tmp_path))

    (tmp_path / "fastvideo_inference.json").unlink()
    assert route(config, str(tmp_path)) is MiniMaxH3Pipeline
    assert config.attention.backend == "VANILLA"
    assert config.attention.sparse_attention_config is None
