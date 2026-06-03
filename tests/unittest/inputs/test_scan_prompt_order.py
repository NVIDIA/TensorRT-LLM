# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the shared `scan_prompt_order` placeholder walk.

`scan_prompt_order` is the single left-to-right placeholder scanner that both
Nano and Qwen3VL `_get_mm_item_order_from_text` delegate to. It walks the
decoded prompt from a moving cursor, at each step picking the nearest next
placeholder among modalities still below their expected count (longer
placeholder wins on a position tie), appends `(modality, running_index)`, and
advances the cursor past the match. At the end it raises `ValueError` if any
modality's observed count disagrees with its expected count.
"""

import pytest

from tensorrt_llm.inputs.multimodal import scan_prompt_order


def test_mixed_image_video_audio_left_to_right():
    # Interleaved image -> video -> audio with one item each, single placeholder
    # per modality (the Nano shape). Order follows prompt position, not the
    # modality iteration order of `placeholder_by_modality`.
    text = "a <IMG> b <VID> c <SND> d"
    placeholder_by_modality = {
        "image": "<IMG>",
        "video": "<VID>",
        "audio": "<SND>",
    }
    expected_counts = {"image": 1, "video": 1, "audio": 1}

    order = scan_prompt_order(text, placeholder_by_modality, expected_counts)

    assert order == [("image", 0), ("video", 0), ("audio", 0)]


def test_single_modality_two_items_running_index():
    # A single modality carrying two items: running index increments 0, 1 in
    # prompt order.
    text = "x <IMG> y <IMG> z"
    placeholder_by_modality = {"image": "<IMG>"}
    expected_counts = {"image": 2}

    order = scan_prompt_order(text, placeholder_by_modality, expected_counts)

    assert order == [("image", 0), ("image", 1)]


def test_count_mismatch_raises():
    # Fewer placeholders in the text than expected -> ValueError at end of walk.
    text = "only one <IMG> here"
    placeholder_by_modality = {"image": "<IMG>"}
    expected_counts = {"image": 2}

    with pytest.raises(ValueError):
        scan_prompt_order(text, placeholder_by_modality, expected_counts)


def test_longer_placeholder_wins_at_same_position():
    # Qwen3VL passes a long form and a short substring of it per modality. When
    # both match at the same cursor position, the longer placeholder must win so
    # the cursor advances past the whole long form (otherwise the short form
    # would be re-matched and inflate the count). One long-form image followed by
    # one bare short-form image -> exactly two image items.
    text = "<vision_start><image_pad><vision_end> mid <image_pad>"
    placeholder_by_modality = {
        "image": ("<vision_start><image_pad><vision_end>", "<image_pad>"),
    }
    expected_counts = {"image": 2}

    order = scan_prompt_order(text, placeholder_by_modality, expected_counts)

    assert order == [("image", 0), ("image", 1)]


def test_falsy_placeholder_is_skipped():
    # A modality whose placeholder string is empty/falsy is skipped entirely
    # (Nano's defensive `if not placeholder`), and contributes zero expected
    # items so it does not trip the count check.
    text = "p <IMG> q"
    placeholder_by_modality = {"image": "<IMG>", "audio": ""}
    expected_counts = {"image": 1, "audio": 0}

    order = scan_prompt_order(text, placeholder_by_modality, expected_counts)

    assert order == [("image", 0)]
