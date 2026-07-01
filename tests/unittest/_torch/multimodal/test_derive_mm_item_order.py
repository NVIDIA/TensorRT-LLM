# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for BaseMultimodalInputProcessor.derive_mm_item_order.

The helper walks the pre-expansion prompt text and returns a prompt-order
manifest `[{"modality": ..., "index": ...}, ...]`. Each placeholder string
occurrence corresponds to exactly one media item, so tests cover:

- pure image-only prompts (one or more items)
- pure video-only prompts
- image-then-video and video-then-image interleaving
- image-video-image (three items in prompt order)
- graceful behavior when placeholders are missing (None) or absent from text
"""

from tensorrt_llm.inputs.registry import BaseMultimodalInputProcessor

_IMAGE_PH = "<|image_pad|>"
_VIDEO_PH = "<|video_pad|>"


def _fn():
    # `derive_mm_item_order` is a concrete instance method on the ABC; call
    # it as an unbound function with `None` self.
    return BaseMultimodalInputProcessor.derive_mm_item_order


def _wrap(inner: str) -> str:
    """Approximate the surrounding tokens Qwen-VL chat templates emit."""
    return f"<|vision_start|>{inner}<|vision_end|>"


def test_pure_image_single_item():
    text = f"describe this: {_wrap(_IMAGE_PH)} thanks"
    order = _fn()(
        None,
        text,
        image_placeholder=_IMAGE_PH,
        video_placeholder=_VIDEO_PH,
    )
    assert order == [{"modality": "image", "index": 0}]


def test_pure_image_two_items():
    text = f"{_wrap(_IMAGE_PH)} vs {_wrap(_IMAGE_PH)}"
    order = _fn()(
        None,
        text,
        image_placeholder=_IMAGE_PH,
        video_placeholder=_VIDEO_PH,
    )
    assert order == [
        {"modality": "image", "index": 0},
        {"modality": "image", "index": 1},
    ]


def test_pure_video_single_item():
    """Even a multi-frame video occupies ONE `<|video_pad|>` in the raw text —
    HF's processor is what later expands it into per-frame token runs. Text
    walk sees exactly one placeholder per item, so no bracketing needed.
    """
    text = f"clip: {_wrap(_VIDEO_PH)}"
    order = _fn()(
        None,
        text,
        image_placeholder=_IMAGE_PH,
        video_placeholder=_VIDEO_PH,
    )
    assert order == [{"modality": "video", "index": 0}]


def test_image_then_video():
    text = f"{_wrap(_IMAGE_PH)} then {_wrap(_VIDEO_PH)}"
    order = _fn()(
        None,
        text,
        image_placeholder=_IMAGE_PH,
        video_placeholder=_VIDEO_PH,
    )
    assert order == [
        {"modality": "image", "index": 0},
        {"modality": "video", "index": 0},
    ]


def test_video_then_image():
    text = f"{_wrap(_VIDEO_PH)} first, then {_wrap(_IMAGE_PH)}"
    order = _fn()(
        None,
        text,
        image_placeholder=_IMAGE_PH,
        video_placeholder=_VIDEO_PH,
    )
    assert order == [
        {"modality": "video", "index": 0},
        {"modality": "image", "index": 0},
    ]


def test_image_video_image_prompt_order():
    text = f"{_wrap(_IMAGE_PH)} a {_wrap(_VIDEO_PH)} b {_wrap(_IMAGE_PH)}"
    order = _fn()(
        None,
        text,
        image_placeholder=_IMAGE_PH,
        video_placeholder=_VIDEO_PH,
    )
    assert order == [
        {"modality": "image", "index": 0},
        {"modality": "video", "index": 0},
        {"modality": "image", "index": 1},
    ]


def test_empty_text():
    order = _fn()(
        None,
        "",
        image_placeholder=_IMAGE_PH,
        video_placeholder=_VIDEO_PH,
    )
    assert order == []


def test_no_placeholders_in_text():
    order = _fn()(
        None,
        "hello world with no media",
        image_placeholder=_IMAGE_PH,
        video_placeholder=_VIDEO_PH,
    )
    assert order == []


def test_missing_video_placeholder_only_images_recognized():
    text = f"{_wrap(_IMAGE_PH)} + {_wrap(_VIDEO_PH)}"
    order = _fn()(
        None,
        text,
        image_placeholder=_IMAGE_PH,
        video_placeholder=None,
    )
    assert order == [{"modality": "image", "index": 0}]


def test_missing_both_placeholders_returns_empty():
    text = f"{_wrap(_IMAGE_PH)} + {_wrap(_VIDEO_PH)}"
    order = _fn()(
        None,
        text,
        image_placeholder=None,
        video_placeholder=None,
    )
    assert order == []


def test_regex_special_chars_in_placeholder_are_literal():
    """Placeholders contain `<|...|>` characters that are special in regex.
    They must match literally, not as regex metacharacters."""
    text = "before <|image_pad|> after"
    order = _fn()(
        None,
        text,
        image_placeholder="<|image_pad|>",
        video_placeholder="<|video_pad|>",
    )
    assert order == [{"modality": "image", "index": 0}]
