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

"""Base H3 video and audio quality against a fresh Diffusers reference."""

from collections.abc import Iterator
from pathlib import Path

import diffusers
import pytest
import torch
from diffusers import ModularPipeline
from PIL import Image

from defs.examples.visual_gen.visual_gen_test_utils import (
    _cleanup_cuda,
    _disable_inductor_compile_worker_quiesce,
    _multi_resolution_log_stft_distance,
)
from defs.visual_gen.minimax_h3.quality_utils import (
    MINIMAX_H3_AUDIO_LOG_STFT_THRESHOLD,
    MINIMAX_H3_LPIPS_THRESHOLD,
    MINIMAX_H3_NUM_INFERENCE_STEPS,
    MINIMAX_H3_PROMPT,
    MINIMAX_H3_QUALITY_HEIGHT,
    MINIMAX_H3_QUALITY_NUM_FRAMES,
    MINIMAX_H3_QUALITY_WIDTH,
    MINIMAX_H3_SEED,
    _mean_lpips_distance,
    _minimax_h3_checkpoint_path,
)
from tensorrt_llm._torch.visual_gen.pipeline_loader import PipelineLoader
from tensorrt_llm.visual_gen.args import TorchCompileConfig, VisualGenArgs


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    ("task", "quant_algo"),
    [("t2va", None), ("fl2va", None), ("t2va", "FP8_BLOCK_SCALES"), ("ref2va", None)],
    ids=["t2va", "fl2va", "t2va-fp8-blockwise", "ref2va"],
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
            torch_compile_config=TorchCompileConfig(enable=True),
        )
    ).load(skip_warmup=True)
    try:
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
