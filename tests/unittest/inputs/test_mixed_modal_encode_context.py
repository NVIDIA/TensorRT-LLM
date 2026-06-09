# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from tensorrt_llm.inputs.multimodal import MixedModalItemOrder


def test_holds_order_and_embedding_lengths():
    ctx = MixedModalItemOrder(
        order=(("image", 0), ("video", 0), ("image", 1)),
        embedding_lengths=(32, 48, 32),
    )
    assert ctx.order == (("image", 0), ("video", 0), ("image", 1))
    assert ctx.embedding_lengths == (32, 48, 32)
    # prompt_pos of item i is just its index in `order`.
    assert ctx.order[2] == ("image", 1)


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="embedding_lengths"):
        MixedModalItemOrder(order=(("image", 0),), embedding_lengths=(32, 48))


def test_duplicate_reference_raises():
    with pytest.raises(ValueError, match="more than once|duplicate"):
        MixedModalItemOrder(order=(("image", 0), ("image", 0)), embedding_lengths=(32, 32))


def test_default_order_is_modality_major():
    order = MixedModalItemOrder.default_order({"image": ["a", "b"], "video": ["c"]})
    assert order == (("image", 0), ("image", 1), ("video", 0))


def test_from_metadata_pairs_order_with_lengths():
    multimodal_data = {
        "multimodal_item_order": [
            {"modality": "image", "index": 0},
            {"modality": "video", "index": 0},
        ]
    }
    ctx = MixedModalItemOrder.from_metadata(multimodal_data, (32, 48))
    assert ctx.order == (("image", 0), ("video", 0))
    assert ctx.embedding_lengths == (32, 48)
    assert MixedModalItemOrder.from_metadata({}, ()) is None


# ---------------------------------------------------------------------------
# Order resolution / normalization (absorbed from the deleted MultimodalPromptOrder).
# `resolve_order` returns a bare prompt-order tuple; `from_metadata` pairs an
# explicit order with `embedding_lengths`. `_normalize`, `from_raw_entries`,
# `order_from_metadata`, and `validate_order` are the self-contained internals
# the producer drives.
# ---------------------------------------------------------------------------


class TestOrderConstructors:
    """Order-only constructors on MixedModalItemOrder (no embedding lengths)."""

    def test_default_order_single_modality(self):
        a, b, c = object(), object(), object()
        assert MixedModalItemOrder.default_order({"image": [a, b, c]}) == (
            ("image", 0),
            ("image", 1),
            ("image", 2),
        )

    def test_default_order_empty_returns_empty(self):
        assert MixedModalItemOrder.default_order({}) == ()

    def test_from_raw_entries_dict_form(self):
        assert MixedModalItemOrder.from_raw_entries(
            [{"modality": "image", "index": 1}, {"type": "video"}], source="x"
        ) == (("image", 1), ("video", 0))

    def test_from_raw_entries_tuple_form(self):
        assert MixedModalItemOrder.from_raw_entries([("image", 0), ("video", 2)], source="x") == (
            ("image", 0),
            ("video", 2),
        )

    def test_from_raw_entries_rejects_unsupported_types(self):
        with pytest.raises(ValueError):
            MixedModalItemOrder.from_raw_entries([3.5], source="x")
        with pytest.raises(ValueError):
            MixedModalItemOrder.from_raw_entries([7], source="x")

    def test_order_from_metadata_prefers_multimodal_item_order(self):
        order = MixedModalItemOrder.order_from_metadata(
            {
                "multimodal_item_order": [
                    {"modality": "image", "index": 1},
                    {"type": "video"},
                    ("image", 0),
                ]
            }
        )
        assert order == (("image", 1), ("video", 0), ("image", 0))

    def test_order_from_metadata_returns_none_when_absent(self):
        assert MixedModalItemOrder.order_from_metadata(None) is None
        assert MixedModalItemOrder.order_from_metadata({}) is None
        assert MixedModalItemOrder.order_from_metadata({"other": 1}) is None


class TestNormalize:
    """`_normalize` lifts modality payloads into list form for ordering."""

    def test_normalize_wraps_scalars_and_drops_non_modalities(self):
        a, b = object(), object()
        assert MixedModalItemOrder._normalize(
            {
                "image": a,
                "video": [b],
                "extra": "ignored",
                "audio": None,
            }
        ) == {"image": [a], "video": [b]}


class TestResolveOrder:
    """`resolve_order` reads explicit metadata when present, else the default order.

    Every mixed preprocess path now bakes `multimodal_item_order` into the
    processed metadata (the text paths always did; the token-id path does after
    the Nano-token-id bake), so `resolve_order` no longer parses prompts: it is
    purely `metadata | default_order`. There is no duck-typed `get_mm_item_order`
    hook on the framework seam.
    """

    def test_uses_metadata_when_present(self):
        a, b, c = object(), object(), object()
        order = MixedModalItemOrder.resolve_order(
            {"image": [a, b], "video": [c]},
            multimodal_data={
                "multimodal_item_order": [
                    {"modality": "video", "index": 0},
                    {"modality": "image", "index": 0},
                    {"modality": "image", "index": 1},
                ]
            },
        )
        assert order == (("video", 0), ("image", 0), ("image", 1))

    def test_uses_default_for_single_modality(self):
        a, b = object(), object()
        assert MixedModalItemOrder.resolve_order({"image": [a, b]}) == (
            ("image", 0),
            ("image", 1),
        )

    def test_uses_metadata_for_multi_modality(self):
        a, b = object(), object()
        # Multi-modality with explicit metadata: the baked order wins over the
        # modality-major default (which would be image-then-video).
        assert MixedModalItemOrder.resolve_order(
            {"image": [a], "video": [b]},
            multimodal_data={
                "multimodal_item_order": [
                    {"modality": "video", "index": 0},
                    {"modality": "image", "index": 0},
                ]
            },
        ) == (("video", 0), ("image", 0))

    def test_falls_back_to_default_when_metadata_absent(self):
        a, b = object(), object()
        # No `multimodal_item_order` in metadata: fall back to the modality-major
        # default order.
        assert MixedModalItemOrder.resolve_order(
            {"image": [a], "video": [b]}, multimodal_data=None
        ) == (("image", 0), ("video", 0))


class TestValidateOrder:
    """`validate_order` is the static coverage check used by `resolve_order`."""

    def test_rejects_unknown_modality(self):
        a = object()
        with pytest.raises(ValueError, match="modality 'audio'"):
            MixedModalItemOrder.validate_order((("audio", 0),), {"image": [a]})

    def test_rejects_out_of_bounds_index(self):
        a = object()
        with pytest.raises(ValueError, match=r"image\[5\]"):
            MixedModalItemOrder.validate_order((("image", 5),), {"image": [a]})

    def test_rejects_coverage_mismatch(self):
        a, b = object(), object()
        with pytest.raises(ValueError, match="expected 2"):
            MixedModalItemOrder.validate_order((("image", 0),), {"image": [a, b]})

    def test_rejects_duplicate_index(self):
        a, b = object(), object()
        with pytest.raises(ValueError, match=r"references image\[0\] more than once"):
            MixedModalItemOrder.validate_order((("image", 0), ("image", 0)), {"image": [a, b]})


class TestStaticProjections:
    """Static projections drive the producer (which holds a bare order tuple)."""

    def test_project_by_order_reorders_by_key(self):
        assert MixedModalItemOrder.project_by_order(
            (("image", 0), ("video", 0), ("image", 1)), {"image": [10, 11], "video": [20]}
        ) == [10, 20, 11]

    def test_project_uuids_by_order_passes_through_none(self):
        assert MixedModalItemOrder.project_uuids_by_order((("image", 0),), None) is None

    def test_project_uuids_by_order_handles_missing_modality(self):
        assert MixedModalItemOrder.project_uuids_by_order(
            (("image", 0), ("video", 0)), {"image": ["a"]}
        ) == ["a", None]


def test_resolve_order_for_single_modality():
    """`resolve_order` returns the bare prompt-order tuple for a single modality."""
    a, b = object(), object()
    order = MixedModalItemOrder.resolve_order({"image": [a, b]})
    assert order == (("image", 0), ("image", 1))
