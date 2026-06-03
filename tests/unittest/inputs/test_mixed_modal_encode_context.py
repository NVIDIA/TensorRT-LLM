# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from tensorrt_llm.inputs.multimodal import MixedModalEncodeContext


def test_holds_order_and_embedding_lengths():
    ctx = MixedModalEncodeContext(
        order=(("image", 0), ("video", 0), ("image", 1)),
        embedding_lengths=(32, 48, 32),
    )
    assert ctx.order == (("image", 0), ("video", 0), ("image", 1))
    assert ctx.embedding_lengths == (32, 48, 32)
    # prompt_pos of item i is just its index in `order`.
    assert ctx.order[2] == ("image", 1)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="embedding_lengths"):
        MixedModalEncodeContext(order=(("image", 0),), embedding_lengths=(32, 48))


def test_duplicate_reference_raises():
    with pytest.raises(ValueError, match="more than once|duplicate"):
        MixedModalEncodeContext(order=(("image", 0), ("image", 0)), embedding_lengths=(32, 32))


def test_flatten_projects_per_modality_values_into_prompt_order():
    ctx = MixedModalEncodeContext(
        order=(("image", 0), ("video", 0), ("image", 1)),
        embedding_lengths=(1, 1, 1),
    )
    out = ctx.flatten({"image": ["i0", "i1"], "video": ["v0"]})
    assert out == ["i0", "v0", "i1"]


def test_flatten_uuids_handles_none():
    ctx = MixedModalEncodeContext(order=(("image", 0),), embedding_lengths=(1,))
    assert ctx.flatten_uuids(None) is None
    assert ctx.flatten_uuids({"image": ["u0"]}) == ["u0"]


def test_default_is_modality_major():
    ctx = MixedModalEncodeContext.default(
        {"image": ["a", "b"], "video": ["c"]}, embedding_lengths=(1, 1, 1)
    )
    assert ctx.order == (("image", 0), ("image", 1), ("video", 0))


def test_from_metadata_pairs_order_with_lengths():
    multimodal_data = {
        "multimodal_item_order": [
            {"modality": "image", "index": 0},
            {"modality": "video", "index": 0},
        ]
    }
    ctx = MixedModalEncodeContext.from_metadata(multimodal_data, (32, 48))
    assert ctx.order == (("image", 0), ("video", 0))
    assert ctx.embedding_lengths == (32, 48)
    assert MixedModalEncodeContext.from_metadata({}, ()) is None
