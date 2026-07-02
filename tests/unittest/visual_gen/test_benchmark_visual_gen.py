# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only tests for the VisualGen benchmark command helpers."""

from tensorrt_llm.bench.benchmark.visual_gen import _add_frame_rate_if_supported


def test_frame_rate_is_omitted_for_image_pipeline():
    kwargs = {"seed": 42, "height": 1024, "width": 1024}

    result = _add_frame_rate_if_supported(
        kwargs,
        fps=16,
        pipeline_defaults={"height": 1024, "width": 1024},
    )

    assert result == kwargs
    assert "frame_rate" not in result


def test_frame_rate_is_applied_for_video_pipeline():
    kwargs = {"seed": 42, "num_frames": 81}

    result = _add_frame_rate_if_supported(
        kwargs,
        fps=24,
        pipeline_defaults={"num_frames": 81, "frame_rate": None},
    )

    assert result == {"seed": 42, "num_frames": 81, "frame_rate": 24.0}
    assert kwargs == {"seed": 42, "num_frames": 81}
