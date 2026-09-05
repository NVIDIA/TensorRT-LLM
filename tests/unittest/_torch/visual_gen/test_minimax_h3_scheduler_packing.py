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

"""CPU unit tests for MiniMax-H3 scheduling and packed-sequence geometry."""

import json
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.minimax_h3.packing import (
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    keyframe_condition_noise,
    patchify_video_latents,
    resolve_canvas_size,
    unpack_audio_tokens,
    unpatchify_video_tokens,
    video_latent_num_frames,
)
from tensorrt_llm._torch.visual_gen.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from tensorrt_llm._torch.visual_gen.models.minimax_h3.scheduler import MiniMaxH3Scheduler


@pytest.mark.parametrize(
    ("shift", "expected_sigmas"),
    [
        (12.0, [1.0, 0.96, 6.0 / 7.0, 0.0]),
        (3.0, [1.0, 6.0 / 7.0, 0.6, 0.0]),
    ],
)
def test_scheduler_builds_h3_shifted_schedule(
    shift: float,
    expected_sigmas: list[float],
) -> None:
    scheduler = MiniMaxH3Scheduler(shift=shift)

    scheduler.set_timesteps(4)

    torch.testing.assert_close(
        scheduler.sigmas,
        torch.tensor(expected_sigmas),
    )
    torch.testing.assert_close(scheduler.timesteps, 1.0 - scheduler.sigmas[:-1])
    assert scheduler.num_inference_steps == 3


def test_scheduler_uses_data_ward_plus_sign_and_fp32_euler() -> None:
    scheduler = MiniMaxH3Scheduler()
    scheduler.set_timesteps(sigmas=[1.0, 0.5, 0.0])
    sample = torch.tensor([2.0], dtype=torch.bfloat16)
    velocity = torch.tensor([0.5], dtype=torch.bfloat16)

    first = scheduler.step(velocity, scheduler.timesteps[0], sample).prev_sample
    second = scheduler.step(velocity, scheduler.timesteps[1], first, return_dict=False)[0]

    assert first.dtype == torch.bfloat16
    torch.testing.assert_close(first.float(), torch.tensor([2.25]), atol=0.0, rtol=0.0)
    torch.testing.assert_close(second.float(), torch.tensor([2.5]), atol=0.0, rtol=0.0)

    scheduler.set_timesteps(sigmas=[0.7, 0.21, 0.0])
    fp16_result = scheduler.step(
        torch.tensor([0.2], dtype=torch.float16),
        scheduler.timesteps[0],
        torch.tensor([0.1], dtype=torch.float16),
    ).prev_sample
    assert fp16_result.item() == 0.197998046875


def test_scheduler_scale_noise_uses_clean_at_t_one() -> None:
    scheduler = MiniMaxH3Scheduler()
    sample = torch.tensor([[2.0, 4.0]])
    noise = torch.tensor([[10.0, 20.0]])

    torch.testing.assert_close(scheduler.scale_noise(sample, 1.0, noise), sample)
    torch.testing.assert_close(
        scheduler.scale_noise(sample, torch.tensor([0.25]), noise),
        torch.tensor([[8.0, 16.0]]),
    )


def test_scheduler_loads_local_diffusers_config(tmp_path: Path) -> None:
    scheduler_dir = tmp_path / "audio_scheduler"
    scheduler_dir.mkdir()
    (scheduler_dir / "scheduler_config.json").write_text(
        json.dumps({"_class_name": "MiniMaxH3Scheduler", "shift": 3.0}),
        encoding="utf-8",
    )

    scheduler = MiniMaxH3Scheduler.from_pretrained(
        tmp_path,
        subfolder="audio_scheduler",
    )

    assert scheduler.shift == 3.0


@pytest.mark.parametrize("sigmas", [[1.0], [1.0, 0.5], [1.0, 0.5, 0.5, 0.0]])
def test_scheduler_rejects_invalid_explicit_sigmas(sigmas: list[float]) -> None:
    with pytest.raises(ValueError):
        MiniMaxH3Scheduler().set_timesteps(sigmas=sigmas)


def test_frame_alignment_and_canvas_contract() -> None:
    assert align_num_frames(90) == 90
    assert align_num_frames(91) == 107
    assert video_latent_num_frames(90) == 27
    assert audio_latent_num_frames(90) == 150
    assert resolve_canvas_size(16, 9) == (768, 1344)
    assert resolve_canvas_size(1, 1) == (768, 768)

    with pytest.raises(ValueError, match="17 \\* n \\+ 5"):
        video_latent_num_frames(91)
    with pytest.raises(ValueError, match="1:4 to 4:1"):
        resolve_canvas_size(5, 1)


@pytest.mark.parametrize(
    ("num_frames", "expected_aligned_num_frames"),
    [
        (108, 124),
        (345, 345),
    ],
)
def test_duration_boundaries_are_accepted_after_frame_alignment(
    num_frames: int,
    expected_aligned_num_frames: int,
) -> None:
    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)

    aligned_num_frames = pipeline._validate_request(
        "a synthetic prompt",
        height=768,
        width=1344,
        num_frames=num_frames,
        frame_rate=24.0,
    )

    assert aligned_num_frames == expected_aligned_num_frames


@pytest.mark.parametrize(
    ("num_frames", "expected_aligned_num_frames"),
    [
        (107, 107),
        (346, 362),
    ],
)
def test_duration_boundaries_are_rejected_after_frame_alignment(
    num_frames: int,
    expected_aligned_num_frames: int,
) -> None:
    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)

    with pytest.raises(
        ValueError,
        match=f"got {expected_aligned_num_frames} frames",
    ):
        pipeline._validate_request(
            "a synthetic prompt",
            height=768,
            width=1344,
            num_frames=num_frames,
            frame_rate=24.0,
        )


def test_video_patchify_round_trip() -> None:
    latents = torch.arange(2 * 3 * 2 * 4 * 6, dtype=torch.float32).reshape(
        2,
        3,
        2,
        4,
        6,
    )

    rows = patchify_video_latents(latents, (1, 2, 2))
    restored = unpatchify_video_tokens(rows, 2, 4, 6, 3, (1, 2, 2))

    assert rows.shape == (24, 12)
    torch.testing.assert_close(restored, latents)


def test_unpack_audio_tokens_restores_stereo_batch() -> None:
    rows = torch.arange(2 * 5 * 3, dtype=torch.float32).reshape(10, 3)

    audio = unpack_audio_tokens(rows, num_audio_latents=5)

    assert audio.shape == (2, 3, 5)
    torch.testing.assert_close(audio[0, :, 0], rows[0])
    torch.testing.assert_close(audio[1, :, 0], rows[5])


def test_build_packed_sequence_places_modalities_and_keyframe_anchors() -> None:
    text_tags = torch.tensor([MINIMAX_H3_TEXT_TAG, MINIMAX_H3_VIDEO_TAG, MINIMAX_H3_TEXT_TAG])

    layout = build_packed_sequence(
        text_token_tags=text_tags,
        num_latent_frames=2,
        latent_height=4,
        latent_width=4,
        num_audio_latents=2,
        patch_size=(1, 2, 2),
        keyframe_anchors=("first", "last"),
    )

    assert layout.sequence_length == 23
    assert layout.num_condition_video_rows == 8
    assert layout.num_condition_audio_rows == 0
    torch.testing.assert_close(layout.text_indices, torch.arange(3))
    torch.testing.assert_close(layout.audio_indices, torch.arange(11, 15))
    torch.testing.assert_close(
        layout.video_indices,
        torch.cat([torch.arange(3, 11), torch.arange(15, 23)]),
    )
    torch.testing.assert_close(layout.token_tags[:3], text_tags)
    assert torch.all(layout.token_tags[layout.audio_indices] == MINIMAX_H3_AUDIO_TAG)
    assert torch.all(layout.token_tags[layout.video_indices] == MINIMAX_H3_VIDEO_TAG)
    assert layout.position_ids.dtype == torch.float64
    torch.testing.assert_close(layout.position_ids[:3, 0], torch.arange(3, dtype=torch.float64))
    torch.testing.assert_close(
        layout.position_ids[3:7, 0],
        torch.full((4,), 3.0, dtype=torch.float64),
    )
    assert layout.position_ids[7, 0] > layout.position_ids[3, 0]
    torch.testing.assert_close(
        layout.position_ids[11:15, 0],
        torch.tensor([3.0, 4.0, 3.0, 4.0], dtype=torch.float64),
    )


def test_build_row_timesteps_keeps_conditions_pinned() -> None:
    layout = build_packed_sequence(
        text_token_tags=torch.tensor([MINIMAX_H3_TEXT_TAG]),
        num_latent_frames=2,
        latent_height=2,
        latent_width=2,
        num_audio_latents=1,
        patch_size=(1, 2, 2),
        keyframe_anchors=("first",),
    )

    timesteps, timestep_indices = build_row_timesteps(
        layout,
        video_timestep=0.1,
        audio_timestep=0.2,
        condition_video_timestep=0.999,
        condition_audio_timestep=1.0,
    )
    expanded = timesteps[timestep_indices]

    torch.testing.assert_close(timesteps, torch.tensor([0.1, 0.2, 0.999]))
    torch.testing.assert_close(
        expanded[layout.video_indices[: layout.num_condition_video_rows]],
        torch.tensor([0.999]),
    )
    torch.testing.assert_close(expanded[layout.audio_indices], torch.tensor([0.2, 0.2]))
    torch.testing.assert_close(
        expanded[layout.video_indices[layout.num_condition_video_rows :]],
        torch.tensor([0.1, 0.1]),
    )


def test_keyframe_condition_noise_is_seeded_and_patchified() -> None:
    first_generator = torch.Generator().manual_seed(123)
    second_generator = torch.Generator().manual_seed(123)

    first = keyframe_condition_noise(
        ((1, 4, 4),),
        patch_size=(1, 2, 2),
        latent_channels=3,
        generator=first_generator,
    )
    second = keyframe_condition_noise(
        ((1, 4, 4),),
        patch_size=(1, 2, 2),
        latent_channels=3,
        generator=second_generator,
    )

    assert first.shape == (4, 12)
    torch.testing.assert_close(first, second)
