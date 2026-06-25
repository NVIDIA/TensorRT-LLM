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
"""Unit tests for Qwen3VLInputProcessorBase._preprocess kwargs handling.

Covers the per-request ``mm_processor_kwargs`` plumbing added to Qwen3-VL:
  * Video metadata from the IO loader is forwarded to the HF processor so
    sample_frames sees the real source fps (rather than HF's 24 fps default).
  * ``num_frames`` and ``fps`` are mutually exclusive in HF's sample_frames; if
    the caller sets ``num_frames`` without ``fps``, ``_preprocess`` must
    explicitly null ``fps`` so the processor's class-level ``fps=2`` default
    does not interfere.
  * The caller-supplied ``mm_processor_kwargs`` dict must not be mutated.

The tests bind ``_preprocess`` to a stand-in object so they avoid the heavy
``__init__`` (which would download an HF processor) and only exercise the new
control-flow.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLInputProcessorBase


def _fake_video(metadata):
    """Minimal stand-in for VideoData: just needs ``.frames`` and ``.metadata``."""
    # A single non-Tensor frame so the do_rescale branch stays in its default.
    return SimpleNamespace(frames=[object()], metadata=metadata)


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


class TestVideoMetadataForwarding:
    """``video_metadata`` is forwarded only when the caller passes
    ``mm_processor_kwargs``."""

    def test_metadata_forwarded_when_opted_in(self):
        """Opted-in path: list built per-video, in order."""
        md0 = {"total_num_frames": 64, "fps": 25.0, "duration": 2.5}
        md1 = {"total_num_frames": 32, "fps": 30.0, "duration": 1.0}
        mm_data = {"video": [_fake_video(md0), _fake_video(md1)]}

        processor = _call_preprocess(mm_data, {"fps": 2.0})

        kwargs = processor.call_args.kwargs
        assert kwargs["video_metadata"] == [md0, md1]
        assert len(kwargs["videos"]) == 2

    def test_metadata_not_forwarded_with_empty_kwargs(self):
        """Empty kwargs: videos present but metadata is not forwarded."""
        mm_data = {"video": [_fake_video({"fps": 30.0})]}

        processor = _call_preprocess(mm_data, {})

        assert processor.call_args.kwargs["video_metadata"] is None

    def test_metadata_is_none_when_no_videos(self):
        """No videos: metadata is None regardless of kwargs."""
        processor = _call_preprocess({}, {"fps": 2.0})
        kwargs = processor.call_args.kwargs
        assert kwargs["video_metadata"] is None
        assert kwargs["videos"] is None


class TestNumFramesFpsMutualExclusivity:
    """Truth table over ("num_frames" present, "fps" present).

    HF's Qwen3VLProcessor.sample_frames treats num_frames and fps as mutually
    exclusive; the class-level default ``fps=2`` would otherwise win over the
    caller's ``num_frames``. The fix injects ``fps=None`` only in the
    (True, False) corner.
    """

    def test_num_frames_without_fps_injects_null_fps(self):
        """(True, False): the corner the fix exists for."""
        mm_data = {"video": [_fake_video({"fps": 30.0})]}

        processor = _call_preprocess(mm_data, {"num_frames": 8})

        kwargs = processor.call_args.kwargs
        assert kwargs["num_frames"] == 8
        assert "fps" in kwargs
        assert kwargs["fps"] is None

    def test_explicit_fps_is_preserved(self):
        """(True, True): the fix must not trample a user-supplied fps."""
        mm_data = {"video": [_fake_video({"fps": 30.0})]}

        processor = _call_preprocess(mm_data, {"num_frames": 8, "fps": 4.0})

        kwargs = processor.call_args.kwargs
        assert kwargs["num_frames"] == 8
        assert kwargs["fps"] == 4.0

    def test_fps_only_passes_through_unchanged(self):
        """(False, True): the condition must not fire for fps alone."""
        processor = _call_preprocess({}, {"fps": 4.0})

        kwargs = processor.call_args.kwargs
        assert kwargs["fps"] == 4.0
        assert "num_frames" not in kwargs

    def test_no_kwargs_means_no_num_frames_no_fps(self):
        """(False, False): the empty-kwargs case adds no spurious keys."""
        processor = _call_preprocess({}, {})

        kwargs = processor.call_args.kwargs
        assert "num_frames" not in kwargs
        assert "fps" not in kwargs


class TestInputDictNotMutated:
    """The caller's mm_processor_kwargs dict is copied, not mutated."""

    def test_num_frames_only_dict_not_mutated(self):
        original = {"num_frames": 8}
        snapshot = dict(original)
        _call_preprocess({}, original)
        assert original == snapshot


class TestUnrelatedKwargsForwarded:
    """Keys unrelated to the num_frames/fps fix pass through verbatim."""

    def test_unrelated_kwargs_reach_processor(self):
        processor = _call_preprocess({}, {"num_frames": 8, "max_pixels": 1234})
        assert processor.call_args.kwargs["max_pixels"] == 1234
