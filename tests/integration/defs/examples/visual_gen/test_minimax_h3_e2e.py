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

"""Real-checkpoint MiniMax-H3 tests against an on-the-fly Diffusers reference.

Weights must already exist in the shared model store, or at
MINIMAX_H3_CHECKPOINT. Reference and candidate run sequentially on one GPU.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import diffusers
import lpips as lpips_package
import pytest
import torch
import torch.nn.functional as F
from defs.examples.visual_gen.visual_gen_test_utils import (
    _cleanup_cuda,
    _disable_inductor_compile_worker_quiesce,
    _llm_models_root,
)
from diffusers import ModularPipeline
from PIL import Image

from tensorrt_llm import VisualGen, VisualGenParams
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm._torch.visual_gen.pipeline_registry import PipelineComponent
from tensorrt_llm.quantization.mode import QuantAlgo
from tensorrt_llm.visual_gen.args import CompilationConfig, TorchCompileConfig, VisualGenArgs

MINIMAX_H3_PROMPT = "\n".join(
    (
        " ".join(
            (
                "integrated_multimodal_description: [Shot 1] Cinematic, medium wide shot, pushing in",
                "slowly. In the cavernous, dimly lit bridge of a starship, sleek metallic consoles with",
                "glowing amber displays flank a massive, curved observation window. A female captain, in",
                "her late 40s with an athletic build and short silver-streaked black hair, stands in the",
                "center midground. She wears a structured, high-collared dark navy military tunic with",
                "silver chest insignias. Her back is to the camera, silhouetted against the cool, ambient",
                "starlight pouring through the thick glass. She stands perfectly still with her hands",
                "clasped tightly behind her back. Outside the window, a massive armada of jagged, dark",
                "grey dreadnoughts hovers in tight formation against a deep purple space nebula. The",
                "fleet's massive rear thrusters begin to glow with an intense, escalating bright blue",
                "light. [Shot 2] At 00:04.500, the camera cuts to a close-up of the captain's face and",
                "shakes strongly. The brilliant blue-white light from the fleet's gathering energy",
                "reflects vividly in her dark eyes. Suddenly, a blinding white flash floods through the",
                "window, completely washing out the background as the fleet jumps to hyperspace. The",
                "sheer spatial force violently jolts the bridge, causing the captain from Shot 1 to",
                "stagger slightly forward, her shoulders tensing as she visibly braces herself against",
                "the physical tremors. As the intense white light fades abruptly, leaving only the dim,",
                "empty expanse of the purple nebula reflected on her starkly lit skin, her jaw clenches,",
                "and she slowly closes her eyes in the newly emptied space.",
            )
        ),
        " ".join(
            (
                "overall_soundscape: A low, resonant hum of the ship's ambient life support systems",
                "serves as the baseline, soon drowned out by an audible, escalating, high-pitched",
                "electronic whine as the fleet outside charges its hyperdrives. A massive, deafening,",
                "bass-heavy boom and sharp crackle erupts during the blinding flash, accompanied by the",
                "loud metallic creaking, rattling, and deep thuds of the bridge's bulkheads vibrating",
                "under immense physical stress. The intense roaring impact then cuts abruptly back to a",
                "hollow, echoing room tone, leaving only the faint, steady hum of the isolated bridge.",
            )
        ),
        " ".join(
            (
                "non_diegetic_music: Cinematic space-opera orchestral score, slow tempo, featuring a",
                "solitary, mournful French horn melody over deep, sustained string dissonances that build",
                "rapidly in volume and intensity, swelling to a massive orchestral peak before snapping",
                "immediately into silence right after the jump.",
            )
        ),
    )
)
MINIMAX_H3_QUALITY_HEIGHT = 128
MINIMAX_H3_QUALITY_WIDTH = 128
MINIMAX_H3_SMOKE_NUM_FRAMES = 124
MINIMAX_H3_QUALITY_NUM_FRAMES = 124
MINIMAX_H3_NUM_INFERENCE_STEPS = 28
MINIMAX_H3_SEED = 0
# Keep explicit quality bounds; report measured distances for every run.
MINIMAX_H3_LPIPS_THRESHOLD = 0.15
# Provisional regression tolerance allowing quantization drift, not a calibrated
# perceptual acceptance score. Its scale is independent of LPIPS.
MINIMAX_H3_AUDIO_LOG_STFT_THRESHOLD = 0.10
MINIMAX_H3_LPIPS_BATCH_SIZE = 16


@pytest.fixture
def _full_cuda_memory_budget() -> Iterator[None]:
    # Earlier LLM tests can leave a process-wide allocator cap after shutdown.
    # The reference and candidate run sequentially and need the full GPU budget.
    device = torch.cuda.current_device()
    previous_fraction = torch.cuda.get_per_process_memory_fraction(device)
    torch.cuda.set_per_process_memory_fraction(1.0, device)
    try:
        yield
    finally:
        torch.cuda.set_per_process_memory_fraction(previous_fraction, device)


def _minimax_h3_checkpoint_path() -> str:
    """Resolve the MiniMax-H3 checkpoint from the shared CI model store.

    ``MINIMAX_H3_CHECKPOINT`` overrides the location for a local run against a
    copy outside that store.  Missing weights fail rather than skip: these tests
    are registered in the CI list, so a silent skip would report coverage that
    never ran.
    """
    override = os.environ.get("MINIMAX_H3_CHECKPOINT", "")
    path = override or os.path.join(_llm_models_root(), "MiniMax-H3")
    if not os.path.isdir(path):
        pytest.fail(f"MiniMax-H3 checkpoint not found: {path}")
    return path


def _multi_resolution_log_stft_distance(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Audio shape mismatch: generated {tuple(actual.shape)}, "
            f"golden {tuple(expected.shape)}."
        )
    actual = actual.float().reshape(-1, actual.shape[-1])
    expected = expected.float().reshape(-1, expected.shape[-1])
    distances: list[torch.Tensor] = []
    for n_fft in (512, 1024, 2048):
        hop_length = n_fft // 4
        window = torch.hann_window(n_fft)
        actual_stft = torch.stft(
            actual,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        ).abs()
        expected_stft = torch.stft(
            expected,
            n_fft=n_fft,
            hop_length=hop_length,
            window=window,
            return_complex=True,
        ).abs()
        distances.append(F.l1_loss(torch.log1p(actual_stft), torch.log1p(expected_stft)))
    return float(torch.stack(distances).mean())


def _generate_diffusers_reference(
    checkpoint_path: str,
    keyframes: list[Image.Image],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a fresh reference with the installed, repository-pinned Diffusers."""
    pipeline = ModularPipeline.from_pretrained(
        checkpoint_path,
        workflow="fl2va" if keyframes else "t2va",
        local_files_only=True,
    )
    pipeline.load_components(
        dtype=torch.bfloat16,
        pretrained_model_name_or_path=checkpoint_path,
        local_files_only=True,
    )
    pipeline.to("cuda")
    try:
        keyframe_inputs = {"image": keyframes[0], "last_image": keyframes[1]} if keyframes else {}
        reference = pipeline(
            prompt=MINIMAX_H3_PROMPT,
            height=MINIMAX_H3_QUALITY_HEIGHT,
            width=MINIMAX_H3_QUALITY_WIDTH,
            num_frames=MINIMAX_H3_QUALITY_NUM_FRAMES,
            num_inference_steps=MINIMAX_H3_NUM_INFERENCE_STEPS,
            generator=torch.Generator(device="cpu").manual_seed(MINIMAX_H3_SEED),
            output_type="pt",
            output=["videos", "audio"],
            **keyframe_inputs,
        )
        video = reference["videos"].detach().float().cpu()
        audio = reference["audio"].detach().float().cpu()
        assert torch.isfinite(video).all() and torch.isfinite(audio).all()
        print(f"MiniMax-H3 reference: diffusers={diffusers.__version__}")
        return video, audio
    finally:
        del pipeline
        _cleanup_cuda()


def _mean_lpips_distance(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> float:
    if actual.dtype != torch.uint8:
        raise AssertionError(f"Generated video must be uint8, got {actual.dtype}.")
    actual = actual.permute(0, 1, 4, 2, 3).float().div(255.0)
    if actual.shape != expected.shape:
        raise AssertionError(
            f"Video shape mismatch: generated {tuple(actual.shape)}, "
            f"golden {tuple(expected.shape)}."
        )

    actual = actual.flatten(0, 1)
    expected = expected.flatten(0, 1)
    metric = lpips_package.LPIPS(net="alex", verbose=False).cuda().eval()
    scores: list[torch.Tensor] = []
    try:
        with torch.inference_mode():
            for start in range(0, actual.shape[0], MINIMAX_H3_LPIPS_BATCH_SIZE):
                end = start + MINIMAX_H3_LPIPS_BATCH_SIZE
                scores.append(
                    metric(
                        expected[start:end].cuda().mul(2).sub(1),
                        actual[start:end].cuda().mul(2).sub(1),
                    ).cpu()
                )
        return float(torch.cat(scores).mean())
    finally:
        del metric
        _cleanup_cuda()


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    ("task", "quant_algo"),
    [("t2va", None), ("fl2va", None), ("t2va", "FP8_BLOCK_SCALES")],
    ids=["t2va", "fl2va", "t2va-fp8-blockwise"],
)
def test_minimax_h3_diffusers_lpips_and_audio_reference(
    task: str, quant_algo: str | None, tmp_path: Path, _full_cuda_memory_budget: None
) -> None:
    """Gate both supported tasks against a fresh Diffusers reference."""
    checkpoint_path = _minimax_h3_checkpoint_path()
    _disable_inductor_compile_worker_quiesce()
    keyframes = (
        [Image.new("RGB", (128, 128), "navy"), Image.new("RGB", (128, 128), "orange")]
        if task == "fl2va"
        else []
    )
    golden_video, golden_audio = _generate_diffusers_reference(checkpoint_path, keyframes)

    pipeline = PipelineLoader(
        VisualGenArgs(
            model=checkpoint_path,
            quant_config={"quant_algo": quant_algo, "dynamic": True} if quant_algo else None,
            torch_compile_config=TorchCompileConfig(enable=True),
        )
    ).load(skip_warmup=True)
    try:
        output = pipeline.forward(
            prompt=MINIMAX_H3_PROMPT,
            keyframes=keyframes,
            keyframe_anchors=("first", "last") if keyframes else (),
            seed=MINIMAX_H3_SEED,
            height=MINIMAX_H3_QUALITY_HEIGHT,
            width=MINIMAX_H3_QUALITY_WIDTH,
            num_frames=MINIMAX_H3_QUALITY_NUM_FRAMES,
            frame_rate=24.0,
            num_inference_steps=MINIMAX_H3_NUM_INFERENCE_STEPS,
        )
        generated_video = output.video.detach().cpu()
        generated_audio = output.audio.detach().cpu()
    finally:
        del pipeline
        _cleanup_cuda()

    # Keep local tensors for inspecting quality failures; do not upload model output.
    torch.save(
        {
            "reference_video": golden_video,
            "reference_audio": golden_audio,
            "video": generated_video,
            "audio": generated_audio,
        },
        tmp_path / "comparison.pt",
    )
    lpips_distance = _mean_lpips_distance(generated_video, golden_video)
    audio_distance = _multi_resolution_log_stft_distance(
        generated_audio,
        golden_audio,
    )
    print(
        "MiniMax-H3 quality: "
        f"LPIPS={lpips_distance:.6f} "
        f"(threshold={MINIMAX_H3_LPIPS_THRESHOLD:.6f}), "
        f"audio_log_stft={audio_distance:.6f} "
        f"(threshold={MINIMAX_H3_AUDIO_LOG_STFT_THRESHOLD:.6f})"
    )

    assert lpips_distance < MINIMAX_H3_LPIPS_THRESHOLD
    assert audio_distance < MINIMAX_H3_AUDIO_LOG_STFT_THRESHOLD
