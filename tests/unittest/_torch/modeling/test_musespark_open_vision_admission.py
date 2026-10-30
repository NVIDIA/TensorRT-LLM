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
"""Admission-time validation of vision segment lengths.

The vision tower presents each temporal clip to full attention as one segment
of `grid_h * grid_w` patches and raises when a segment exceeds the
configured maximum. Before this check existed at admission, that condition
was first caught inside the tower forward, where the executor's error
handling fails every request in the scheduled batch; on the overlap loop the
None forward result then crashed the executor loop on every rank. These tests
pin the admission-side arithmetic and the model hook that the executor's
`_validate_request` calls.
"""

import types

import pytest
import torch

from tensorrt_llm._torch.models.modeling_musespark_open_omni import (
    MuseSparkForConditionalGeneration,
    validate_vision_segment_lengths,
)

# The CPU-* CI stages run pytest with -m 'cpu_only'. Without this marker every
# test in the file is deselected, which pytest reports as exit code 5 and the
# stage reports as a failure.
pytestmark = pytest.mark.cpu_only


def _vision_payload(modality, grids):
    grid = torch.tensor(grids, dtype=torch.int32)
    patches = int(sum(t * h * w for t, h, w in grids))
    return {
        modality: {
            "pixel_values": torch.zeros((patches, 4), dtype=torch.bfloat16),
            f"{modality}_grid_thw": grid,
        }
    }


# ---------------------------------------------------------------------------
# validate_vision_segment_lengths
# ---------------------------------------------------------------------------


def test_under_budget_image_passes():
    validate_vision_segment_lengths(_vision_payload("image", [(1, 64, 128)]), max_seq_len=8192)


def test_at_budget_image_passes():
    validate_vision_segment_lengths(_vision_payload("image", [(1, 64, 128)]), max_seq_len=64 * 128)


def test_over_budget_image_is_rejected():
    """The live-serve crash shape: one image of 16112 patches vs a budget
    of 8192 sailed through admission and killed the engine mid-forward."""
    with pytest.raises(
        ValueError,
        match="vision attention segment length 16112 exceeds the configured maximum 8192",
    ):
        validate_vision_segment_lengths(_vision_payload("image", [(1, 106, 152)]), max_seq_len=8192)


def test_over_budget_video_is_rejected():
    with pytest.raises(ValueError, match="exceeds the configured maximum"):
        validate_vision_segment_lengths(_vision_payload("video", [(4, 100, 100)]), max_seq_len=8192)


def test_video_segment_is_per_clip_not_per_item():
    """A clip is one temporal frame's h*w patches; t clips of h*w each must
    NOT be summed into t*h*w. 8 clips of 4096 fit an 8192 budget even though
    the item totals 32768 patches."""
    validate_vision_segment_lengths(_vision_payload("video", [(8, 64, 64)]), max_seq_len=8192)


def test_second_grid_row_is_checked():
    with pytest.raises(ValueError, match="exceeds the configured maximum"):
        validate_vision_segment_lengths(
            _vision_payload("image", [(1, 10, 10), (1, 200, 200)]), max_seq_len=8192
        )


def test_precomputed_embedding_payload_is_skipped():
    """No pixel_values means the tower will not run; nothing to bound."""
    validate_vision_segment_lengths(
        {"image": {"multimodal_embedding": torch.zeros(4, 8)}}, max_seq_len=1
    )


def test_text_only_payload_passes():
    validate_vision_segment_lengths({}, max_seq_len=8192)
    validate_vision_segment_lengths({"audio": {"waveforms": torch.zeros(8)}}, max_seq_len=8192)


def test_unset_budget_disables_the_check():
    validate_vision_segment_lengths(_vision_payload("image", [(1, 200, 200)]), max_seq_len=0)


# ---------------------------------------------------------------------------
# The model hook the executor calls from _validate_request
# ---------------------------------------------------------------------------


def _hook(mm_encoder):
    """Call the (plain) method without instantiating the model."""
    fake_self = types.SimpleNamespace(mm_encoder=mm_encoder)
    return lambda mm_data: (
        MuseSparkForConditionalGeneration.validate_multimodal_request_data(fake_self, mm_data)
    )


def test_model_hook_rejects_over_budget_image():
    tower = types.SimpleNamespace(attn_max_seq_len=8192)
    encoder = types.SimpleNamespace(tower=tower)
    with pytest.raises(ValueError, match="exceeds the configured maximum"):
        _hook(encoder)(_vision_payload("image", [(1, 106, 152)]))


def test_model_hook_accepts_under_budget_image():
    tower = types.SimpleNamespace(attn_max_seq_len=8192)
    encoder = types.SimpleNamespace(tower=tower)
    _hook(encoder)(_vision_payload("image", [(1, 64, 64)]))


def test_model_hook_without_encoder_accepts_everything():
    """Text-only builds (mm_encoder=None) must not reject anything."""
    _hook(None)(_vision_payload("image", [(1, 500, 500)]))
