# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for Qwen3-VL preprocess control flow.

Covers two pieces of logic that decide what reaches HF's video processor:

  1. ``_decide_do_sample_frames`` — the per-video sampling decision.
  2. ``_preprocess`` — metadata rewriting and kwargs threading around the
     processor call.

The tests use a production adapter instance and a mock HF processor to check
metadata and kwargs forwarding. Real HF processing is covered by the
multimodal video sampling tests.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensorrt_llm._torch.models.modeling_qwen3vl import (
    Qwen3VLInputProcessorBase,
    _decide_do_sample_frames,
)

pytestmark = pytest.mark.cpu_only


def _fake_video(metadata, *, n_frames=8):
    """Stand-in for VideoData: only `.frames` (count) and `.metadata` are read."""
    return SimpleNamespace(frames=[object()] * n_frames, metadata=metadata)


def _call_preprocess(mm_data, mm_processor_kwargs):
    fake_processor = MagicMock(return_value={})
    fake_processor._merge_kwargs.return_value = {
        "images_kwargs": {},
        "videos_kwargs": dict(mm_processor_kwargs),
    }
    fake_processor.video_processor.do_sample_frames = True
    fake_processor.video_processor.num_frames = None
    fake_processor.video_processor.fps = 2
    fake_processor.video_processor.max_frames = 768
    fake_processor.video_processor.sample_frames.return_value = [0, 2, 4, 7]
    instance = object.__new__(Qwen3VLInputProcessorBase)
    instance._processor = fake_processor
    instance._preprocess(
        "the prompt",
        mm_data,
        mm_processor_kwargs,
    )
    return fake_processor


class TestDecideDoSampleFrames:
    """Decision tree for whether HF should run ``sample_frames``."""

    def test_explicit_true_wins_even_when_counts_match(self):
        # Even with a matching target, caller's explicit True is honored.
        vd = _fake_video({"duration": 4.0}, n_frames=8)
        assert _decide_do_sample_frames([vd], {"do_sample_frames": True}) is True

    def test_explicit_false_wins_even_when_counts_differ(self):
        # Even when num_frames would force sampling, caller's False is honored.
        vd = _fake_video({"duration": 4.0}, n_frames=8)
        assert (
            _decide_do_sample_frames([vd], {"do_sample_frames": False, "num_frames": 16}) is False
        )

    def test_num_frames_matches_decoded_no_sampling(self):
        vd = _fake_video({}, n_frames=8)
        assert _decide_do_sample_frames([vd], {"num_frames": 8}) is False

    def test_num_frames_differs_triggers_sampling(self):
        vd = _fake_video({}, n_frames=8)
        assert _decide_do_sample_frames([vd], {"num_frames": 4}) is True

    @pytest.mark.parametrize("target", ["num_frames", "fps"])
    @pytest.mark.parametrize("value", [0, -1, -2])
    def test_nonpositive_single_target_disables_limit(self, target, value):
        vd = _fake_video(
            {"fps": 24.0, "duration": 10.0, "frames_indices": list(range(40))}, n_frames=40
        )
        assert _decide_do_sample_frames([vd], {target: value}) is False

    @pytest.mark.parametrize("num_frames", [None, 0, 40])
    def test_fps_target_uses_decoded_prefix_duration(self, num_frames):
        vd = _fake_video(
            {
                "fps": 24.0,
                "total_num_frames": 240,
                "duration": 10.0,
                "frames_indices": list(range(40)),
            },
            n_frames=40,
        )
        kwargs = {"fps": 4}
        if num_frames is not None:
            kwargs["num_frames"] = num_frames
        assert _decide_do_sample_frames([vd], kwargs) is True

    def test_fps_target_computed_from_duration(self):
        # Target = floor(duration * fps) = floor(4.0 * 2.0) = 8 → no resample.
        vd_match = _fake_video({"duration": 4.0}, n_frames=8)
        assert _decide_do_sample_frames([vd_match], {"fps": 2.0}) is False
        # FPS sampling still delegates its frame cap to HF.
        vd_diff = _fake_video({"duration": 4.0}, n_frames=8)
        assert _decide_do_sample_frames([vd_diff], {"fps": 4.0}) is True

    @pytest.mark.parametrize("n,target", [(10, 32), (40, 100)])
    def test_oversized_num_frames_does_not_request_upsampling(self, n, target):
        assert not _decide_do_sample_frames([_fake_video({}, n_frames=n)], {"num_frames": target})

    def test_silent_caller_io_loaded_all_triggers_fallback_sampling(self):
        # No num_frames/fps from caller, IO loaded every source frame
        # (decoded count == total_num_frames): defer to HF's class-default
        # sampling.
        vd = _fake_video({"total_num_frames": 240}, n_frames=240)
        assert _decide_do_sample_frames([vd], {}) is True

    def test_silent_caller_io_subsampled_triggers_sampling(self):
        # No num_frames/fps from caller: defer to HF's class-default sampling
        # even when IO already subsampled (decoded count < total_num_frames).
        vd = _fake_video({"total_num_frames": 240}, n_frames=8)
        assert _decide_do_sample_frames([vd], {}) is True

    def test_batch_reduction_is_any_video_needs_sampling(self):
        # One video matches, one needs resampling → batch is sampled.
        match = _fake_video({}, n_frames=8)
        differ = _fake_video({}, n_frames=16)
        assert _decide_do_sample_frames([match, differ], {"num_frames": 8}) is True


class TestPreprocessMetadataForwarding:
    """`_preprocess` forwards per-video metadata with controlled rewrites."""

    def test_source_total_num_frames_retained(self):
        # Source clip had 240 frames; IO subsampled to 8.
        # The source total is retained; sampling is handled before HF processes
        # the decoded frames with its own sampling switch disabled.
        vd = _fake_video({"total_num_frames": 240, "fps": 24.0, "duration": 10.0}, n_frames=8)
        processor = _call_preprocess({"video": [vd]}, {"num_frames": 8})
        m = processor.call_args.kwargs["videos_kwargs"]["video_metadata"][0]
        assert m["total_num_frames"] == 240
        assert m["fps"] == 24.0  # other fields unchanged

    def test_metadata_is_none_when_no_videos(self):
        processor = _call_preprocess({}, {"fps": 2.0})
        assert "video_metadata" not in processor.call_args.kwargs["videos_kwargs"]


class TestPreprocessKwargsForwarding:
    """Videos are sampled per item, then HF's batch sampler is disabled."""

    def test_sampling_kwargs_forwarded_when_sampling(self):
        # Sampling receives the requested target, then the batch switch is disabled.
        vd = _fake_video({}, n_frames=8)
        processor = _call_preprocess({"video": [vd]}, {"num_frames": 4})
        kwargs = processor.call_args.kwargs["videos_kwargs"]
        assert kwargs["do_sample_frames"] is False
        assert "num_frames" not in kwargs
        assert processor.video_processor.sample_frames.call_args.kwargs["num_frames"] == 4
        assert kwargs["video_metadata"][0]["frames_indices"] == [0, 2, 4, 7]

    def test_sampling_kwargs_dropped_when_not_sampling(self):
        # num_frames matches frames count → no sampling → num_frames NOT forwarded
        # (HF's class-default ``fps=2`` cannot collide because do_sample_frames=False).
        vd = _fake_video({}, n_frames=8)
        processor = _call_preprocess({"video": [vd]}, {"num_frames": 8})
        kwargs = processor.call_args.kwargs["videos_kwargs"]
        assert kwargs["do_sample_frames"] is False
        assert "num_frames" not in kwargs
        assert "fps" not in kwargs

    def test_caller_dict_not_mutated(self):
        original = {"num_frames": 8, "max_pixels": 1234}
        snapshot = dict(original)
        _call_preprocess({}, original)
        assert original == snapshot

    def test_unrelated_kwargs_pass_through(self):
        # Resize/normalize knobs flow through unchanged.
        processor = _call_preprocess({}, {"max_pixels": 1234})
        assert processor.call_args.kwargs["videos_kwargs"]["max_pixels"] == 1234
