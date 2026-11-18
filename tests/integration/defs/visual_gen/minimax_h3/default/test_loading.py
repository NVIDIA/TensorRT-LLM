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

"""Real-checkpoint loading, compilation and public API coverage for base H3."""

import pytest
import torch
from PIL import Image

from defs.examples.visual_gen.visual_gen_test_utils import (
    _cleanup_cuda,
    _disable_inductor_compile_worker_quiesce,
)
from defs.visual_gen.minimax_h3.quality_utils import (
    MINIMAX_H3_SEED,
    MINIMAX_H3_SMOKE_NUM_FRAMES,
    _minimax_h3_checkpoint_path,
)
from tensorrt_llm import VisualGen, VisualGenParams
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.visual_gen.args import CompilationConfig, TorchCompileConfig, VisualGenArgs


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_minimax_h3_released_transformer_loads_and_runs() -> None:
    """Load every released transformer tensor and compile all model layers."""
    checkpoint_path = _minimax_h3_checkpoint_path()
    _disable_inductor_compile_worker_quiesce()
    pipeline = PipelineLoader(
        VisualGenArgs(
            model=checkpoint_path,
            torch_compile_config=TorchCompileConfig(enable=True),
        )
    ).load(
        skip_warmup=True,
        skip_components=[
            PipelineComponent.TOKENIZER,
            PipelineComponent.PROCESSOR,
            PipelineComponent.TEXT_ENCODER,
            PipelineComponent.VAE,
            PipelineComponent.AUDIO_VAE,
            PipelineComponent.SCHEDULER,
            PipelineComponent.AUDIO_SCHEDULER,
        ],
    )
    transformer = None
    output = None
    try:
        transformer = pipeline.transformer
        assert all(
            isinstance(block, torch._dynamo.OptimizedModule)
            for block in transformer.transformer_blocks
        )
        position_ids = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 0, 1], [3, 0, 0]],
            device="cuda",
        )
        # Clear code cached by earlier tests so this forward records its own
        # newly compiled frames.
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        with torch.inference_mode():
            output = transformer(
                hidden_states=torch.randn(1, 2, 96, device="cuda"),
                audio_hidden_states=torch.randn(1, 1, 32, device="cuda"),
                encoder_hidden_states=torch.randn(
                    1,
                    2,
                    5120,
                    device="cuda",
                    dtype=torch.bfloat16,
                ),
                timestep=torch.tensor([1.0], device="cuda"),
                conditioning_timesteps=torch.tensor([1.0], device="cuda"),
                timestep_indices=torch.zeros(5, dtype=torch.long, device="cuda"),
                token_tags=torch.tensor([1, 1, 0, 0, 2], device="cuda"),
                position_ids=position_ids,
                video_indices=torch.tensor([2, 3], device="cuda"),
                audio_indices=torch.tensor([4], device="cuda"),
                text_indices=torch.tensor([0, 1], device="cuda"),
            )

        assert output.sample.shape == (1, 2, 96)
        assert output.audio_sample.shape == (1, 1, 32)
        assert torch.isfinite(output.sample).all()
        assert torch.isfinite(output.audio_sample).all()
        assert torch._dynamo.utils.counters["frames"]["ok"] > 0
        assert not any(parameter.is_meta for parameter in transformer.parameters())
    finally:
        del output
        del transformer
        del pipeline
        _cleanup_cuda()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("quant_algo", ["FP8", "FP8_BLOCK_SCALES", "NVFP4"])
def test_minimax_h3_released_transformer_dynamic_quant_compile_smoke(quant_algo: str) -> None:
    """Dynamically quantize the released BF16 weights and run compiled H3."""
    checkpoint_path = _minimax_h3_checkpoint_path()
    _disable_inductor_compile_worker_quiesce()
    pipeline = PipelineLoader(
        VisualGenArgs(
            model=checkpoint_path,
            quant_config={"quant_algo": quant_algo, "dynamic": True},
            torch_compile_config=TorchCompileConfig(enable=True),
        )
    ).load(
        skip_warmup=True,
        skip_components=[
            PipelineComponent.TOKENIZER,
            PipelineComponent.PROCESSOR,
            PipelineComponent.TEXT_ENCODER,
            PipelineComponent.VAE,
            PipelineComponent.AUDIO_VAE,
            PipelineComponent.SCHEDULER,
            PipelineComponent.AUDIO_SCHEDULER,
        ],
    )
    transformer = None
    output = None
    try:
        transformer = pipeline.transformer
        assert transformer.model_config.quant_config.quant_algo == QuantAlgo(quant_algo)
        assert transformer.model_config.dynamic_weight_quant
        assert all(
            isinstance(block, torch._dynamo.OptimizedModule)
            for block in transformer.transformer_blocks
        )
        assert transformer.context_embedder.has_any_quant
        for module in (
            transformer.proj_in,
            transformer.audio_proj_in,
            transformer.proj_out,
            transformer.audio_proj_out,
        ):
            assert module.has_any_quant == (quant_algo != "FP8_BLOCK_SCALES")

        position_ids = torch.tensor(
            [[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 0, 1], [3, 0, 0]],
            device="cuda",
        )
        torch._dynamo.utils.counters.clear()
        with torch.inference_mode():
            output = transformer(
                hidden_states=torch.randn(1, 2, 96, device="cuda"),
                audio_hidden_states=torch.randn(1, 1, 32, device="cuda"),
                encoder_hidden_states=torch.randn(
                    1,
                    2,
                    5120,
                    device="cuda",
                    dtype=torch.bfloat16,
                ),
                timestep=torch.tensor([1.0], device="cuda"),
                conditioning_timesteps=torch.tensor([1.0], device="cuda"),
                timestep_indices=torch.zeros(5, dtype=torch.long, device="cuda"),
                token_tags=torch.tensor([1, 1, 0, 0, 2], device="cuda"),
                position_ids=position_ids,
                video_indices=torch.tensor([2, 3], device="cuda"),
                audio_indices=torch.tensor([4], device="cuda"),
                text_indices=torch.tensor([0, 1], device="cuda"),
            )

        assert output.sample.shape == (1, 2, 96)
        assert output.audio_sample.shape == (1, 1, 32)
        assert torch.isfinite(output.sample).all()
        assert torch.isfinite(output.audio_sample).all()
        assert torch._dynamo.utils.counters["frames"]["ok"] > 0
    finally:
        del output
        del transformer
        del pipeline
        torch._dynamo.reset()
        _cleanup_cuda()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_minimax_h3_released_pipeline_smoke() -> None:
    """Load all components and execute both T2VA and first/last-frame FL2VA."""
    checkpoint_path = _minimax_h3_checkpoint_path()
    _disable_inductor_compile_worker_quiesce()
    pipeline = PipelineLoader(
        VisualGenArgs(
            model=checkpoint_path,
            torch_compile_config=TorchCompileConfig(enable=True),
        )
    ).load(skip_warmup=True)
    output = None
    conditioned_output = None
    try:
        output = pipeline.forward(
            prompt="A red fox walking through snow",
            seed=MINIMAX_H3_SEED,
            height=128,
            width=128,
            num_frames=MINIMAX_H3_SMOKE_NUM_FRAMES,
            frame_rate=24.0,
            # MiniMax-H3 counts both endpoints in its sigma grid, so two
            # grid points execute one denoising evaluation.
            num_inference_steps=2,
        )

        assert output.video.shape == (1, MINIMAX_H3_SMOKE_NUM_FRAMES, 128, 128, 3)
        assert output.video.dtype == torch.uint8
        assert output.audio.ndim == 3
        assert output.audio.shape[:2] == (1, 2)
        assert output.audio.shape[-1] > 0
        assert torch.isfinite(output.audio).all()
        assert output.frame_rate == 24.0
        assert output.audio_sample_rate > 0

        conditioned_output = pipeline.forward(
            prompt="Move smoothly between the supplied keyframes",
            seed=MINIMAX_H3_SEED,
            height=128,
            width=128,
            num_frames=MINIMAX_H3_SMOKE_NUM_FRAMES,
            frame_rate=24.0,
            num_inference_steps=2,
            keyframes=[
                Image.new("RGB", (128, 128), "navy"),
                Image.new("RGB", (128, 128), "orange"),
            ],
            keyframe_anchors=("first", "last"),
        )
        assert conditioned_output.video.shape == (
            1,
            MINIMAX_H3_SMOKE_NUM_FRAMES,
            128,
            128,
            3,
        )
        assert conditioned_output.audio.shape[:2] == (1, 2)
        assert torch.isfinite(conditioned_output.audio).all()
    finally:
        del conditioned_output
        del output
        del pipeline
        _cleanup_cuda()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_minimax_h3_public_visual_gen_api_smoke() -> None:
    """Generate joint video/audio through the public VisualGen executor API."""
    checkpoint_path = _minimax_h3_checkpoint_path()
    _disable_inductor_compile_worker_quiesce()
    visual_gen = VisualGen(
        model=checkpoint_path,
        args=VisualGenArgs(
            model=checkpoint_path,
            compilation_config=CompilationConfig(skip_warmup=True),
            torch_compile_config=TorchCompileConfig(enable=True),
        ),
    )
    try:
        output = visual_gen.generate(
            inputs="A red fox walking through snow",
            params=VisualGenParams(
                seed=MINIMAX_H3_SEED,
                height=128,
                width=128,
                num_frames=MINIMAX_H3_SMOKE_NUM_FRAMES,
                frame_rate=24.0,
                num_inference_steps=2,
            ),
        )
        assert output.video.shape == (
            1,
            MINIMAX_H3_SMOKE_NUM_FRAMES,
            128,
            128,
            3,
        )
        assert output.video.dtype == torch.uint8
        assert output.audio.shape[:2] == (1, 2)
        assert torch.isfinite(output.audio).all()
        assert output.frame_rate == 24.0
        assert output.audio_sample_rate == 32000
    finally:
        visual_gen.shutdown()
        del visual_gen
        _cleanup_cuda()
