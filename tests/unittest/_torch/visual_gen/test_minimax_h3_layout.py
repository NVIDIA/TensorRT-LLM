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

"""Unit tests for the MiniMax-H3 packed-sequence layout and schedules.

These run on the CPU with no model weights: the packed layout, the frame/canvas
arithmetic, the per-row timestep plan and the patchify round trip are pure
bookkeeping, and they are what every other part of the pipeline addresses rows
through. Where the Diffusers reference blocks are importable the tests compare
against them directly, so a divergence in either implementation is caught.

Run:
    pytest tests/unittest/_torch/visual_gen/test_minimax_h3_layout.py -v
"""

import os

os.environ["TLLM_DISABLE_MPI"] = "1"

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.minimax_h3.pipeline_minimax_h3 import (
    MINIMAX_H3_AUDIO_CHANNELS,
    MINIMAX_H3_AUDIO_TAG,
    MINIMAX_H3_TEXT_TAG,
    MINIMAX_H3_VIDEO_TAG,
    align_num_frames,
    audio_latent_num_frames,
    build_packed_sequence,
    build_row_timesteps,
    patchify_video_latents,
    resolve_canvas_size,
    unpatchify_video_latents,
    video_latent_num_frames,
)

PATCH_SIZE = (1, 2, 2)
CANVAS_MULTIPLE = 32
SHORT_EDGE = 768
MAX_PIXELS = 768 * 1344


class TestCanvas:
    @pytest.mark.parametrize(
        "aspect, expected",
        [
            ((16, 9), (768, 1344)),
            ((9, 16), (1344, 768)),
            ((1, 1), (768, 768)),
            ((4, 3), (768, 1024)),
        ],
    )
    def test_resolve_canvas_size(self, aspect, expected):
        assert resolve_canvas_size(*aspect, CANVAS_MULTIPLE, SHORT_EDGE, MAX_PIXELS) == expected

    def test_canvas_is_multiple_of_32(self):
        height, width = resolve_canvas_size(21, 9, CANVAS_MULTIPLE, SHORT_EDGE, MAX_PIXELS)
        assert height % CANVAS_MULTIPLE == 0 and width % CANVAS_MULTIPLE == 0

    @pytest.mark.parametrize("aspect", [(5, 1), (1, 5), (0, 1), (-16, 9)])
    def test_rejects_unsupported_aspect_ratios(self, aspect):
        with pytest.raises(ValueError):
            resolve_canvas_size(*aspect, CANVAS_MULTIPLE, SHORT_EDGE, MAX_PIXELS)


class TestFrameArithmetic:
    @pytest.mark.parametrize(
        "requested, aligned",
        [(124, 124), (120, 124), (106, 107), (5, 5), (23, 39)],
    )
    def test_align_num_frames(self, requested, aligned):
        assert align_num_frames(requested, 17, 5) == aligned

    def test_aligned_frames_are_decodable(self):
        for requested in range(5, 400):
            aligned = align_num_frames(requested, 17, 5)
            assert aligned >= requested
            assert aligned % 17 == 5
            # Decodable frame counts map onto a whole number of latent frames.
            video_latent_num_frames(aligned, 17, 5)

    def test_video_latent_num_frames(self):
        assert video_latent_num_frames(124, 17, 5) == 37
        assert video_latent_num_frames(5, 17, 5) == 2

    def test_video_latent_num_frames_rejects_unaligned(self):
        with pytest.raises(ValueError):
            video_latent_num_frames(123, 17, 5)

    def test_audio_latent_num_frames(self):
        # 124 frames at 24 fps is 5.1667 s, and the audio VAE emits 40 latents/s.
        assert audio_latent_num_frames(124) == 207
        assert audio_latent_num_frames(24) == 40


class TestPackedSequence:
    @staticmethod
    def _layout(num_text=16, keyframe_anchors=()):
        text_tags = torch.full((num_text,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
        return build_packed_sequence(
            text_tags,
            num_latent_frames=37,
            latent_height=24,
            latent_width=40,
            num_audio_latents=207,
            patch_size=PATCH_SIZE,
            keyframe_anchors=keyframe_anchors,
        )

    def test_row_counts_and_tags(self):
        pos, tags, video_idx, audio_idx, text_idx, num_cond = self._layout()
        rows_per_frame = (24 // 2) * (40 // 2)
        expected_len = 16 + 207 * MINIMAX_H3_AUDIO_CHANNELS + 37 * rows_per_frame

        assert pos.shape == (expected_len, 3)
        assert pos.dtype == torch.float64, "the rotary grid is float64 by checkpoint contract"
        assert video_idx.numel() == 37 * rows_per_frame
        assert audio_idx.numel() == 207 * MINIMAX_H3_AUDIO_CHANNELS
        assert text_idx.numel() == 16
        assert num_cond == 0

        # Every row is addressed exactly once and tagged by its modality.
        all_rows = torch.cat([video_idx, audio_idx, text_idx]).sort().values
        assert torch.equal(all_rows, torch.arange(expected_len))
        assert set(tags[video_idx].tolist()) == {MINIMAX_H3_VIDEO_TAG}
        assert set(tags[audio_idx].tolist()) == {MINIMAX_H3_AUDIO_TAG}
        assert set(tags[text_idx].tolist()) == {MINIMAX_H3_TEXT_TAG}

    def test_text_rows_lead_the_sequence(self):
        pos, _, _, _, text_idx, _ = self._layout(num_text=16)
        assert torch.equal(text_idx, torch.arange(16))
        # Text sits on the time axis at its row index and carries no h/w.
        assert torch.equal(pos[:16, 0], torch.arange(16, dtype=torch.float64))
        assert torch.all(pos[:16, 1:] == 0)

    def test_audio_rows_are_channel_major_on_two_width_extremes(self):
        pos, _, _, audio_idx, _, _ = self._layout()
        audio_pos = pos[audio_idx]
        left, right = audio_pos[0, 2].item(), audio_pos[-1, 2].item()
        assert left != right, "the two stereo blocks sit on opposite width extremes"
        assert torch.all(audio_pos[:207, 2] == left)
        assert torch.all(audio_pos[207:, 2] == right)
        assert torch.all(audio_pos[:, 1] == 0), "audio rows carry no height coordinate"
        # The two channels share one clock.
        assert torch.equal(audio_pos[:207, 0], audio_pos[207:, 0])

    @pytest.mark.parametrize("anchors", [("first",), ("last",), ("first", "last")])
    def test_keyframe_conditioning_rows(self, anchors):
        _, tags, video_idx, _, _, num_cond = self._layout(keyframe_anchors=anchors)
        rows_per_frame = (24 // 2) * (40 // 2)
        assert num_cond == len(anchors) * rows_per_frame
        # Conditioning rows lead the video rows and are tagged as video.
        assert set(tags[video_idx[:num_cond]].tolist()) == {MINIMAX_H3_VIDEO_TAG}

    def test_rejects_unknown_anchor(self):
        with pytest.raises(ValueError):
            self._layout(keyframe_anchors=("middle",))


class TestRowTimesteps:
    @staticmethod
    def _plan(num_cond_video=0, num_cond_audio=0):
        text_tags = torch.full((8,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
        _, _, video_idx, audio_idx, text_idx, _ = build_packed_sequence(
            text_tags,
            7,
            8,
            8,
            5,
            PATCH_SIZE,
            keyframe_anchors=("first",) * (1 if num_cond_video else 0),
        )
        cond_rows = num_cond_video and (8 // 2) * (8 // 2)
        return (
            build_row_timesteps(
                video_idx,
                audio_idx,
                cond_rows,
                num_cond_audio,
                text_idx.numel(),
                video_timestep=0.4,
                audio_timestep=0.7,
                condition_video_timestep=0.999,
                condition_audio_timestep=1.0,
            ),
            video_idx,
            audio_idx,
            text_idx,
            cond_rows,
        )

    def test_generated_rows_get_their_own_schedule(self):
        (unique, inverse), video_idx, audio_idx, text_idx, _ = self._plan()
        assert torch.allclose(unique, torch.tensor([0.4, 0.7]))
        # Text rows never reach an output head and inherit the video timestep.
        assert torch.all(unique[inverse[text_idx]] == 0.4)
        assert torch.all(unique[inverse[video_idx]] == 0.4)
        assert torch.all(unique[inverse[audio_idx]] == 0.7)

    def test_conditioning_rows_stay_pinned(self):
        (unique, inverse), video_idx, _, _, cond_rows = self._plan(num_cond_video=1)
        assert torch.allclose(unique, torch.tensor([0.4, 0.7, 0.999]))
        cond_t = unique[inverse[video_idx[:cond_rows]]]
        gen_t = unique[inverse[video_idx[cond_rows:]]]
        assert torch.allclose(cond_t, torch.full_like(cond_t, 0.999))
        assert torch.allclose(gen_t, torch.full_like(gen_t, 0.4))

    def test_inverse_indexes_every_row(self):
        (unique, inverse), video_idx, audio_idx, text_idx, _ = self._plan()
        assert inverse.shape == (video_idx.numel() + audio_idx.numel() + text_idx.numel(),)
        assert int(inverse.max()) < unique.numel()


class TestPatchify:
    def test_round_trip(self):
        torch.manual_seed(0)
        latents = torch.randn(1, 24, 7, 8, 12)
        rows = patchify_video_latents(latents, PATCH_SIZE)
        assert rows.shape == (7 * 4 * 6, 24 * 4)
        back = unpatchify_video_latents(rows, 7, 8, 12, 24, PATCH_SIZE)
        assert torch.equal(latents, back)

    def test_rejects_indivisible_shapes(self):
        with pytest.raises(ValueError):
            patchify_video_latents(torch.zeros(1, 24, 7, 9, 12), PATCH_SIZE)


class TestAgainstDiffusersReference:
    """Compare against the Diffusers blocks when they are available."""

    @staticmethod
    def _reference_blocks():
        try:
            from diffusers.modular_pipelines.minimax_h3.before_denoise import (
                MiniMaxH3PrepareLayoutStep,
                MiniMaxH3SetTimestepsStep,
            )
            from diffusers.modular_pipelines.minimax_h3.before_denoise import (
                patchify_video_latents as ref_patchify,
            )
        except ImportError:
            pytest.skip("diffusers build without the MiniMax-H3 modular blocks")
        return MiniMaxH3PrepareLayoutStep, MiniMaxH3SetTimestepsStep, ref_patchify

    def test_packed_sequence_matches_reference(self):
        layout_step, _, _ = self._reference_blocks()
        text_tags = torch.full((16,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
        ref = layout_step.build_packed_sequence(
            text_tags,
            37,
            24,
            40,
            207,
            PATCH_SIZE,
            MINIMAX_H3_AUDIO_CHANNELS,
            MINIMAX_H3_AUDIO_TAG,
            MINIMAX_H3_VIDEO_TAG,
        )
        got = build_packed_sequence(text_tags, 37, 24, 40, 207, PATCH_SIZE)
        for name, a, b in zip(
            ("position_ids", "token_tags", "video_indices", "audio_indices", "text_indices"),
            ref[:5],
            got[:5],
        ):
            assert torch.equal(a, b), f"{name} differs from the Diffusers reference"
        assert ref[5] == got[5]

    def test_row_timesteps_match_reference(self):
        layout_step, timesteps_step, _ = self._reference_blocks()
        text_tags = torch.full((16,), MINIMAX_H3_TEXT_TAG, dtype=torch.long)
        ref_layout = layout_step.build_packed_sequence(
            text_tags,
            37,
            24,
            40,
            207,
            PATCH_SIZE,
            MINIMAX_H3_AUDIO_CHANNELS,
            MINIMAX_H3_AUDIO_TAG,
            MINIMAX_H3_VIDEO_TAG,
        )
        _, _, video_idx, audio_idx, text_idx, ncv, nca = ref_layout
        ref_unique, ref_inverse = timesteps_step.build_row_timesteps(
            video_idx, audio_idx, ncv, nca, text_idx.numel(), 0.4, 0.7, 0.999, 1.0
        )
        unique, inverse = build_row_timesteps(
            video_idx, audio_idx, ncv, nca, text_idx.numel(), 0.4, 0.7, 0.999, 1.0
        )
        assert torch.equal(ref_unique, unique)
        assert torch.equal(ref_inverse, inverse)

    def test_patchify_matches_reference(self):
        _, _, ref_patchify = self._reference_blocks()
        torch.manual_seed(1)
        latents = torch.randn(1, 24, 7, 8, 12)
        assert torch.equal(
            ref_patchify(latents, PATCH_SIZE), patchify_video_latents(latents, PATCH_SIZE)
        )
