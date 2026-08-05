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

  1. ``_decide_do_sample_frames`` — the decision tree that picks the single
     ``do_sample_frames`` flag for the whole batch.
  2. ``_preprocess`` — metadata rewriting and kwargs threading around the
     processor call.

The tests bind ``_preprocess`` to a stand-in object so they exercise the
control flow without constructing a real HF processor.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from tensorrt_llm._torch.models.modeling_qwen3vl import (
    Qwen3VLInputProcessorBase,
    _decide_do_sample_frames,
)


def _fake_video(metadata, *, n_frames=8):
    """Stand-in for VideoData: only `.frames` (count) and `.metadata` are read."""
    return SimpleNamespace(frames=[object()] * n_frames, metadata=metadata)


def _call_preprocess(mm_data, mm_processor_kwargs):
    fake_processor = MagicMock(return_value={})
    fake_self = SimpleNamespace(processor=fake_processor)
    Qwen3VLInputProcessorBase._preprocess(
        fake_self,
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

    def test_fps_target_computed_from_duration(self):
        # Target = floor(duration * fps) = floor(4.0 * 2.0) = 8 → no resample.
        vd_match = _fake_video({"duration": 4.0}, n_frames=8)
        assert _decide_do_sample_frames([vd_match], {"fps": 2.0}) is False
        # Target = floor(4.0 * 4.0) = 16 ≠ 8 → resample.
        vd_diff = _fake_video({"duration": 4.0}, n_frames=8)
        assert _decide_do_sample_frames([vd_diff], {"fps": 4.0}) is True

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

    def test_total_num_frames_rewritten_to_decoded_count(self):
        # Source clip had 240 frames; IO subsampled to 8.
        # `total_num_frames` must be rewritten so HF's sample_frames indices
        # stay in range of the 8-frame tensor we hand it.
        vd = _fake_video({"total_num_frames": 240, "fps": 24.0, "duration": 10.0}, n_frames=8)
        processor = _call_preprocess({"video": [vd]}, {"num_frames": 8})
        m = processor.call_args.kwargs["video_metadata"][0]
        assert m["total_num_frames"] == 8
        assert m["fps"] == 24.0  # other fields unchanged

    def test_metadata_is_none_when_no_videos(self):
        processor = _call_preprocess({}, {"fps": 2.0})
        assert processor.call_args.kwargs["video_metadata"] is None


class TestPreprocessKwargsForwarding:
    """`do_sample_frames` is always passed; sampling kwargs are conditional."""

    def test_sampling_kwargs_forwarded_when_sampling(self):
        # num_frames mismatches frames count → sampling fires → num_frames forwarded.
        vd = _fake_video({}, n_frames=8)
        processor = _call_preprocess({"video": [vd]}, {"num_frames": 4})
        kwargs = processor.call_args.kwargs
        assert kwargs["do_sample_frames"] is True
        assert kwargs["num_frames"] == 4

    def test_sampling_kwargs_dropped_when_not_sampling(self):
        # num_frames matches frames count → no sampling → num_frames NOT forwarded
        # (HF's class-default ``fps=2`` cannot collide because do_sample_frames=False).
        vd = _fake_video({}, n_frames=8)
        processor = _call_preprocess({"video": [vd]}, {"num_frames": 8})
        kwargs = processor.call_args.kwargs
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
        assert processor.call_args.kwargs["max_pixels"] == 1234
