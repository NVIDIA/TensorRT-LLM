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

"""Real-checkpoint release, quality and parallelism cases for MiniMax-H3 V2."""

import os
from collections.abc import Iterator
from pathlib import Path

import diffusers
import lpips as lpips_package
import pytest
import torch
from diffusers import ModularPipeline
from PIL import Image
from test_common.llm_data import get_checkpoint
from test_common.visual_gen.h3_loading import load_default_h3_visual_gen

from defs.examples.visual_gen.visual_gen_test_utils import (
    _cleanup_cuda,
    _disable_inductor_compile_worker_quiesce,
    _multi_resolution_log_stft_distance,
)
from tensorrt_llm import VisualGenParams
from tensorrt_llm._torch.visual_gen.modeling_v2.models.minimax_h3.h3_pipeline.pipeline import (
    MiniMaxH3Pipeline,
)
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
MINIMAX_H3_NUM_INFERENCE_STEPS = 50
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
    override = os.environ.get("MINIMAX_H3_CHECKPOINT")
    if override and not Path(override).is_dir():
        raise FileNotFoundError(override)
    return override or get_checkpoint("MiniMax-H3")


def _generate_diffusers_reference(
    checkpoint_path: str,
    keyframes: list[Image.Image],
    references: list | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate a fresh reference with the installed, repository-pinned Diffusers."""
    pipeline = ModularPipeline.from_pretrained(
        checkpoint_path,
        workflow="ref2va" if references else ("fl2va" if keyframes else "t2va"),
        local_files_only=True,
    )
    pipeline.load_components(
        dtype={"default": torch.bfloat16, "vae": torch.float32, "audio_vae": torch.float32}
        if references
        else torch.bfloat16,
        pretrained_model_name_or_path=checkpoint_path,
        local_files_only=True,
    )
    pipeline.to("cuda")
    try:
        keyframe_inputs = {"image": keyframes[0], "last_image": keyframes[1]} if keyframes else {}
        if references:
            keyframe_inputs = {"references": references}
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


def check_released_transformer_loads_and_runs() -> None:
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
        assert type(pipeline) is MiniMaxH3Pipeline
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


def check_released_transformer_dynamic_quant_compile_smoke(quant_algo: str) -> None:
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
        assert type(pipeline) is MiniMaxH3Pipeline
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


def check_released_pipeline_smoke() -> None:
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
        assert type(pipeline) is MiniMaxH3Pipeline
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


def check_public_visual_gen_api_smoke() -> None:
    """Generate joint video/audio through the public VisualGen executor API."""
    checkpoint_path = _minimax_h3_checkpoint_path()
    _disable_inductor_compile_worker_quiesce()
    visual_gen = load_default_h3_visual_gen(
        VisualGenArgs(
            model=checkpoint_path,
            compilation_config=CompilationConfig(skip_warmup=True),
            torch_compile_config=TorchCompileConfig(enable=True),
        ),
    )
    try:
        assert visual_gen.default_params.num_inference_steps == 50
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


def check_diffusers_lpips_and_audio_reference(
    task: str,
    quant_algo: str | None,
    tmp_path: Path,
    _full_cuda_memory_budget: None,
) -> None:
    """Gate both supported tasks against a fresh Diffusers reference."""
    checkpoint_path = _minimax_h3_checkpoint_path()
    _disable_inductor_compile_worker_quiesce()
    keyframes = (
        [Image.new("RGB", (128, 128), "navy"), Image.new("RGB", (128, 128), "orange")]
        if task == "fl2va"
        else []
    )
    references = None
    if task == "ref2va":
        from diffusers.modular_pipelines.minimax_h3.references import (
            MiniMaxH3AudioReference,
            MiniMaxH3ImageReference,
        )

        # Exercise both frozen conditioning prefixes against a fresh reference.
        waveform = 0.1 * torch.sin(torch.arange(64000) * (2 * torch.pi * 440 / 32000))
        references = [
            MiniMaxH3ImageReference(Image.new("RGB", (128, 128), "navy")),
            MiniMaxH3AudioReference(waveform.repeat(2, 1), sample_rate=32000),
        ]
    golden_video, golden_audio = _generate_diffusers_reference(
        checkpoint_path, keyframes, references
    )

    pipeline = PipelineLoader(
        VisualGenArgs(
            model=checkpoint_path,
            pipeline_config={"workflow": "ref2va"} if references else {},
            quant_config={"quant_algo": quant_algo, "dynamic": True} if quant_algo else None,
            torch_compile_config=TorchCompileConfig(enable=True, enable_autotune=True),
            compilation_config=CompilationConfig(
                resolutions=[(MINIMAX_H3_QUALITY_HEIGHT, MINIMAX_H3_QUALITY_WIDTH)],
                num_frames=[MINIMAX_H3_QUALITY_NUM_FRAMES],
            ),
        )
    ).load(skip_warmup=quant_algo != "FP8_BLOCK_SCALES")
    try:
        assert type(pipeline) is MiniMaxH3Pipeline
        output = pipeline.forward(
            prompt=MINIMAX_H3_PROMPT,
            keyframes=keyframes,
            references=references,
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


MINIMAX_H3_MULTI_GPU_LPIPS_THRESHOLD = 0.30


def check_text_encoder_tp_lpips(tmp_path, monkeypatch, world_size):
    """Compare encoder TP with HF encoding while keeping DiT parallelism fixed."""
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"Requires {world_size} CUDA GPUs")
    monkeypatch.setenv("TLLM_DISABLE_MPI", "1")
    checkpoint = _minimax_h3_checkpoint_path()
    videos, audios = [], []
    for encoder_tp in (1, 2):
        args = VisualGenArgs(
            model=checkpoint,
            parallel_config={"ulysses_size": world_size, "text_encoder_tp_size": encoder_tp},
            attention_config={"backend": "VANILLA"},
            torch_compile_config={"enable": False},
            cuda_graph_config={"enable": False},
            compilation_config={"skip_warmup": True},
        )
        engine = load_default_h3_visual_gen(args)
        try:
            assert engine.default_params.num_inference_steps == 50
            output = engine.generate(
                inputs="A woman smiles and waves in a sunlit park, with birds chirping.",
                params=VisualGenParams(
                    height=128, width=128, num_frames=124, num_inference_steps=4, seed=42
                ),
            )
            video = torch.as_tensor(output.video).cpu()
            if video.ndim == 4:
                video = video.unsqueeze(0)
            audio = torch.as_tensor(output.audio).cpu()
            assert video.shape == (1, 124, 128, 128, 3)
            assert video.float().std() > 1
            assert torch.isfinite(audio).all()
            videos.append(video)
            audios.append(audio)
            torch.save(
                {"video": video, "audio": audio},
                tmp_path / f"h3_world{world_size}_encoder_tp{encoder_tp}.pt",
            )
        finally:
            engine.shutdown()
    reference = videos[0].permute(0, 1, 4, 2, 3).float().div(255)
    score = _mean_lpips_distance(videos[1], reference)
    audio_score = _multi_resolution_log_stft_distance(audios[1], audios[0])
    print(f"H3 encoder TP=2/{world_size}: LPIPS={score:.6f}, audio log-STFT={audio_score:.6f}")
    assert score < 0.05, f"Encoder TP changed video: LPIPS={score:.6f}"
    assert audio_score < MINIMAX_H3_AUDIO_LOG_STFT_THRESHOLD


def check_ulysses_parallel_vae_lpips(tmp_path, ulysses_size, vae_size, backend):
    """Compare real Ulysses + tiled VAE collectives with a fresh one-GPU baseline."""
    if torch.cuda.device_count() < ulysses_size:
        pytest.skip(f"Requires {ulysses_size} CUDA GPUs")
    prompt = "A woman smiles and waves in a sunlit park, with birds chirping."
    videos, audios = [], []
    for size, decode_size in ((1, 1), (ulysses_size, 1), (ulysses_size, vae_size)):
        args = VisualGenArgs(
            model=_minimax_h3_checkpoint_path(),
            parallel_config={"ulysses_size": size, "parallel_vae_size": decode_size},
            attention_config={"backend": backend},
            torch_compile_config={"enable": True},
            cuda_graph_config={"enable": False},
            compilation_config={"skip_warmup": True},
        )
        engine = load_default_h3_visual_gen(args)
        try:
            assert engine.default_params.num_inference_steps == 50
            output = engine.generate(
                inputs=prompt,
                params=VisualGenParams(
                    height=384,
                    width=384,
                    num_frames=124,
                    num_inference_steps=MINIMAX_H3_NUM_INFERENCE_STEPS,
                    seed=42,
                ),
            )
            video = torch.as_tensor(output.video).cpu()
            if video.ndim == 4:
                video = video.unsqueeze(0)
            assert video.shape == (1, 124, 384, 384, 3)
            assert video.float().std() > 1
            assert torch.isfinite(torch.as_tensor(output.audio)).all()
            torch.save(
                {"video": video, "audio": output.audio},
                tmp_path / f"h3_ulysses{size}_vae{decode_size}.pt",
            )
            videos.append(video)
            audios.append(torch.as_tensor(output.audio).cpu())
        finally:
            engine.shutdown()
    # The tiled decoder batches a rank's tiles into one call, so the two runs decode with
    # different batch shapes; compiled kernels may then round a pixel to the adjacent uint8
    # level. Audio takes no tile batching and stays exact.
    torch.testing.assert_close(videos[2], videos[1], atol=1, rtol=0)
    torch.testing.assert_close(audios[2], audios[1])
    for decode_size, audio in zip((1, vae_size), audios[1:]):
        audio_score = _multi_resolution_log_stft_distance(audio, audios[0])
        print(
            f"H3 Ulysses={ulysses_size} + parallel_vae_size={decode_size} vs single GPU: "
            f"audio log-STFT={audio_score:.6f}"
        )
        assert audio_score < MINIMAX_H3_AUDIO_LOG_STFT_THRESHOLD, (
            f"H3 parallelism changed audio: log-STFT={audio_score:.6f} "
            f">= {MINIMAX_H3_AUDIO_LOG_STFT_THRESHOLD}"
        )
    reference = videos[0].permute(0, 1, 4, 2, 3).float().div(255)
    score = _mean_lpips_distance(videos[2], reference)
    print(
        f"H3 Ulysses={ulysses_size} + parallel_vae_size={vae_size} vs single GPU: mean LPIPS={score:.6f}"
    )
    assert score < MINIMAX_H3_MULTI_GPU_LPIPS_THRESHOLD, (
        f"H3 parallelism changed output: mean LPIPS={score:.6f} "
        f">= {MINIMAX_H3_MULTI_GPU_LPIPS_THRESHOLD}"
    )


def check_tp2_example_lpips(tmp_path):
    """Check the shipped TP2 configuration against a same-build TP1 pipeline."""
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA GPUs")
    config_path = (
        Path(__file__).resolve().parents[6] / "examples/visual_gen/configs/minimax-h3-bf16-tp2.yaml"
    )
    model = _minimax_h3_checkpoint_path()
    prompt = "A woman smiles and waves in a sunlit park, with birds chirping."
    videos, audios = [], []
    for tp_size in (1, 2):
        args = VisualGenArgs.from_yaml(config_path, model=model)
        assert args.parallel_config.tp_size == 2
        assert args.attention_config.backend == "VANILLA"
        assert args.torch_compile_config.enable
        assert not args.cuda_graph_config.enable
        # Keep the reference identical to the example except for TP degree.
        args.parallel_config.tp_size = tp_size
        engine = load_default_h3_visual_gen(args)
        try:
            assert engine.default_params.num_inference_steps == 50
            output = engine.generate(
                inputs=prompt,
                params=VisualGenParams(
                    height=128,
                    width=128,
                    num_frames=124,
                    num_inference_steps=MINIMAX_H3_NUM_INFERENCE_STEPS,
                    seed=42,
                ),
            )
            video = torch.as_tensor(output.video).cpu()
            if video.ndim == 4:
                video = video.unsqueeze(0)
            audio = torch.as_tensor(output.audio).cpu()
            assert video.shape == (1, 124, 128, 128, 3)
            assert video.float().std() > 1
            assert torch.isfinite(audio).all()
            assert audio.float().std() > 0
            torch.save({"video": video, "audio": audio}, tmp_path / f"h3_tp{tp_size}.pt")
            videos.append(video)
            audios.append(audio)
        finally:
            engine.shutdown()

    reference = videos[0].permute(0, 1, 4, 2, 3).float().div(255)
    video_score = _mean_lpips_distance(videos[1], reference)
    audio_score = _multi_resolution_log_stft_distance(audios[1], audios[0])
    print(f"H3 TP2 example vs TP1: LPIPS={video_score:.6f}, audio log-STFT={audio_score:.6f}")
    assert video_score < MINIMAX_H3_LPIPS_THRESHOLD, (
        f"H3 TP2 changed video: LPIPS={video_score:.6f} >= {MINIMAX_H3_LPIPS_THRESHOLD}"
    )
    assert audio_score < MINIMAX_H3_AUDIO_LOG_STFT_THRESHOLD, (
        f"H3 TP2 changed audio: log-STFT={audio_score:.6f} >= {MINIMAX_H3_AUDIO_LOG_STFT_THRESHOLD}"
    )
