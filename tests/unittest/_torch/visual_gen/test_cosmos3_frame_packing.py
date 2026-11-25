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
"""Cosmos3 Nano Sim: frame-major packing round-trips and absolute rotary positions."""

import pytest
import torch

from tensorrt_llm._torch.visual_gen.models.cosmos3.frame_packing import FramePacking
from tensorrt_llm._torch.visual_gen.models.cosmos3.transformer_cosmos3 import (
    compute_mrope_position_ids_action,
    compute_mrope_position_ids_vision,
    sim_rotary_positions,
)

pytestmark = pytest.mark.cpu_only

TEXT_LEN, MARGIN, BASE_FPS, TCF = 37, 15000, 24.0, 4
A, H, W = 4, 15, 26  # 480p: 15 x 26 = 390 vision tokens


def test_interleave_packs_frame_major_and_vision_unpacks():
    packing = FramePacking(num_frames=3, action_tokens=A, vision_tokens=H * W)
    assert (packing.tokens_per_frame, packing.num_tokens) == (394, 3 * 394)
    dim = 8
    vision = torch.arange(3 * H * W, dtype=torch.float32).view(1, -1, 1).expand(1, -1, dim)
    action = -torch.arange(1, 3 * A + 1, dtype=torch.float32).view(1, -1, 1).expand(1, -1, dim)
    packed = packing.interleave(vision, action)
    assert packed.shape == (1, 3 * 394, dim)
    for f in range(3):
        a_slice, v_slice = packing.frame_slices(f)
        assert torch.equal(packed[0, a_slice, 0], action[0, f * A : (f + 1) * A, 0])
        assert torch.equal(packed[0, v_slice, 0], vision[0, f * H * W : (f + 1) * H * W, 0])
    assert torch.equal(packing.vision(packed), vision)
    # One frame's vision tokens are contiguous in the packed chunk: a view, no copy.
    one = FramePacking(num_frames=1, action_tokens=A, vision_tokens=H * W)
    packed_one = one.interleave(vision[:, : H * W], action[:, :A])
    assert torch.equal(one.vision(packed_one), vision[:, : H * W])
    assert one.vision(packed_one).data_ptr() == packed_one[:, A:].data_ptr()
    with pytest.raises(ValueError):
        packing.interleave(vision[:, :-1], action)


@pytest.mark.parametrize("first_frame, num_frames", [(0, 1), (1, 4), (5, 4), (237, 4)])
@pytest.mark.parametrize("fps", [24.0, 16.0])
def test_positions_follow_the_spec(first_frame, num_frames, fps):
    """Vision time of absolute frame f is text_len + margin + f * base_fps / fps; its
    action rows sit a quarter latent frame apart, ending at the frame's own time.
    Height and width restart inside every frame."""
    packing = FramePacking(num_frames=num_frames, action_tokens=A, vision_tokens=H * W)
    ids = sim_rotary_positions(
        packing,
        first_frame=first_frame,
        grid_h=H,
        grid_w=W,
        text_len=TEXT_LEN,
        temporal_modality_margin=MARGIN,
        fps=fps,
        action_fps=fps,
        base_fps=BASE_FPS,
        temporal_compression_factor=TCF,
        enable_fps_modulation=True,
    ).double()
    assert ids.shape == (3, packing.num_tokens)
    media = TEXT_LEN + MARGIN
    for i in range(num_frames):
        f = first_frame + i
        frame_time = media + f * BASE_FPS / fps
        a_slice, v_slice = packing.frame_slices(i)
        expected_action = torch.tensor(
            [frame_time - (A - 1 - j) * (BASE_FPS / fps) / TCF for j in range(A)],
            dtype=torch.double,
        )
        if f == 0:
            expected_action = torch.full((A,), float(frame_time), dtype=expected_action.dtype)
        torch.testing.assert_close(ids[0, a_slice], expected_action, atol=1e-4, rtol=0)
        assert torch.all(ids[0, v_slice] == frame_time)
        assert torch.equal(ids[1, v_slice], torch.arange(H).repeat_interleave(W).double())
        assert torch.equal(ids[2, v_slice], torch.arange(W).repeat(H).double())
        assert torch.all(ids[1:, a_slice] == 0)


def test_positions_match_one_long_clip():
    """A chunk at frame f0 takes exactly the positions frames f0.. would have in a
    single bidirectional clip with action_start_frame_offset = 1: eviction never
    renumbers anything."""
    f0, n, total = 9, 4, 20
    media = TEXT_LEN + MARGIN
    clip_vision, _ = compute_mrope_position_ids_vision(
        total,
        H,
        W,
        temporal_offset=media,
        fps=24.0,
        base_fps=BASE_FPS,
        temporal_compression_factor=TCF,
        enable_fps_modulation=True,
    )
    clip_action, _ = compute_mrope_position_ids_action(
        (total - 1) * A,
        temporal_offset=media,
        action_fps=24.0,
        base_fps=BASE_FPS,
        base_temporal_compression_factor=TCF,
        enable_fps_modulation=True,
        start_frame_offset=1,
    )
    packing = FramePacking(num_frames=n, action_tokens=A, vision_tokens=H * W)
    ids = sim_rotary_positions(
        packing,
        first_frame=f0,
        grid_h=H,
        grid_w=W,
        text_len=TEXT_LEN,
        temporal_modality_margin=MARGIN,
        fps=24.0,
        action_fps=24.0,
        base_fps=BASE_FPS,
        temporal_compression_factor=TCF,
        enable_fps_modulation=True,
    )
    for i in range(n):
        f = f0 + i
        a_slice, v_slice = packing.frame_slices(i)
        torch.testing.assert_close(
            ids[:, v_slice], clip_vision[:, f * H * W : (f + 1) * H * W].to(ids.dtype)
        )
        # the clip's action rows for frame f are rows (f-1)*A .. f*A
        torch.testing.assert_close(
            ids[:, a_slice], clip_action[:, (f - 1) * A : f * A].to(ids.dtype)
        )
