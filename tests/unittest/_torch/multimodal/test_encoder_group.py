# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the generic encoder-group primitives.

Covers the pure helpers behind ``encode_multimodal_by_groups``:

* ``_lengths_by_modality`` — invert prompt-ordered ``multimodal_embedding_lengths``
  into per-modality per-item lists using ``item_order``.
* ``MultimodalParams.get_item_layout`` — item metadata with legacy fallbacks.
* ``_reorder_embeds_by_manifest`` — slice per-modality tensors and concat in
  each request's prompt-order manifest.
"""

import pytest
import torch

from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
    EncoderGroup,
    MultimodalEncoderContractError,
    _lengths_by_modality,
    _reorder_embeds_by_manifest,
    encode_multimodal_by_groups,
)
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalItemMetadata,
    MultimodalItemPartition,
    MultimodalParams,
)

pytestmark = pytest.mark.cpu_only


def _mp(*, item_order=None, embedding_lengths=None, buckets=None):
    """Build the same common item record supplied by input preparation."""
    data = dict(buckets or {})
    refs = (
        [(entry["modality"], entry["index"]) for entry in item_order]
        if item_order
        else [(next(iter(data)), i) for i in range(len(embedding_lengths or ()))]
    )
    metadata = MultimodalItemMetadata(
        refs,
        output_embedding_lengths=(
            () if not refs and embedding_lengths is None else embedding_lengths
        ),
    )
    metadata.validate()
    data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = metadata
    return MultimodalParams(multimodal_data=data)


class TestLengthsByModality:
    def test_mixed_manifest_image_video_image(self):
        # Manifest order: image#0 (4 rows), video#0 (6), image#1 (2).
        mp = _mp(
            item_order=[
                {"modality": "image", "index": 0},
                {"modality": "video", "index": 0},
                {"modality": "image", "index": 1},
            ],
            embedding_lengths=[4, 6, 2],
            buckets={"image": {}, "video": {}},
        )
        assert _lengths_by_modality([mp.get_item_layout()], ("image", "video")) == {
            "image": [4, 2],
            "video": [6],
        }

    def test_single_modality_no_manifest(self):
        # No item_order; flat list is already in per-modality order.
        mp = _mp(embedding_lengths=[64, 64], buckets={"image": {}})
        assert _lengths_by_modality([mp.get_item_layout()], ("image", "video")) == {
            "image": [64, 64],
            "video": [],
        }

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="references and lengths must align"):
            _mp(
                item_order=[
                    {"modality": "image", "index": 0},
                    {"modality": "video", "index": 0},
                ],
                embedding_lengths=[4],
                buckets={"image": {}, "video": {}},
            )

    def test_missing_item_record_does_not_infer_cross_group_order(self):
        mp = MultimodalParams(multimodal_data={"image": {}, "audio": {}})
        assert mp.get_item_layout() == ((), ())

    def test_single_modality_outside_group_is_skipped(self):
        # Request has only audio; called on the image group. Must silently
        # return the group's empty bucket rather than KeyError on
        # ``by_modality["audio"]``.
        mp = _mp(embedding_lengths=[3], buckets={"audio": {}})
        assert _lengths_by_modality([mp.get_item_layout()], ("image",)) == {"image": []}


class TestMultimodalItemLayout:
    def test_image_only(self):
        mp = _mp(embedding_lengths=[10, 20, 30], buckets={"image": {}})
        assert mp.get_item_layout() == ((("image", 0), ("image", 1), ("image", 2)), (10, 20, 30))

    def test_layout_includes_all_request_modalities(self):
        # Layout belongs to the request, independent of the encoder group.
        mp = _mp(embedding_lengths=[10], buckets={"audio": {}})
        assert mp.get_item_layout() == ((("audio", 0),), (10,))

    def test_item_metadata_owns_order_and_lengths(self) -> None:
        metadata = MultimodalItemMetadata(
            [("video", 0), ("image", 0), ("audio", 0)], [24, 16, 8], [6, 4, 2]
        )
        mp = _mp(
            item_order=[{"modality": "image", "index": 0}],
            embedding_lengths=[99],
            buckets={"image": {}, "video": {}, "audio": {}},
        )
        mp.multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = metadata

        item_refs, lengths = mp.get_item_layout()

        assert item_refs == tuple(metadata.item_refs)
        assert lengths == tuple(metadata.output_embedding_lengths)
        assert _lengths_by_modality([mp.get_item_layout()], ("image", "video")) == {
            "image": [4],
            "video": [6],
        }
        assert _lengths_by_modality([mp.get_item_layout()], ("audio",)) == {"audio": [2]}


class TestReorderEmbedsByManifest:
    @staticmethod
    def _marker_tensor(marker, rows, hidden=2):
        # Rows all share a marker so the reorder can be asserted by column-0.
        return torch.full((rows, hidden), float(marker))

    def test_mixed_image_video_image_in_one_request(self):
        # Prompt: image#0 (4 rows, marker 10), video#0 (6, 20), image#1 (2, 30).
        mp = _mp(
            item_order=[
                {"modality": "image", "index": 0},
                {"modality": "video", "index": 0},
                {"modality": "image", "index": 1},
            ],
            buckets={"image": {}, "video": {}},
        )
        # Per-modality tensors are cat'd in encounter order — image items 0
        # and 1 concatenated, then video item 0.
        per_modality_embeds = {
            "image": torch.cat([self._marker_tensor(10, 4), self._marker_tensor(30, 2)], dim=0),
            "video": self._marker_tensor(20, 6),
        }
        per_modality_lengths = {"image": [4, 2], "video": [6]}
        out = _reorder_embeds_by_manifest(
            [(mp.item_metadata.item_refs, ())], per_modality_embeds, per_modality_lengths
        )
        # Expected column-0: 10 (×4), 20 (×6), 30 (×2) in prompt order.
        expected = torch.tensor([10.0] * 4 + [20.0] * 6 + [30.0] * 2)
        assert torch.equal(out[:, 0], expected)

    def test_cross_request_cursors_advance_per_modality(self):
        # Request A: [image#0 (2)]; Request B: [video#0 (3), image#0 (1)].
        # Global per-modality indexing: image=[A#0, B#0], video=[B#0].
        # Manifests use per-request indices — cursor must translate them.
        mp_a = _mp(
            item_order=[{"modality": "image", "index": 0}],
            buckets={"image": {}},
        )
        mp_b = _mp(
            item_order=[
                {"modality": "video", "index": 0},
                {"modality": "image", "index": 0},
            ],
            buckets={"image": {}, "video": {}},
        )
        per_modality_embeds = {
            "image": torch.cat([self._marker_tensor(1, 2), self._marker_tensor(3, 1)], dim=0),
            "video": self._marker_tensor(2, 3),
        }
        per_modality_lengths = {"image": [2, 1], "video": [3]}
        out = _reorder_embeds_by_manifest(
            [(mp_a.item_metadata.item_refs, ()), (mp_b.item_metadata.item_refs, ())],
            per_modality_embeds,
            per_modality_lengths,
        )
        # A: image#0 → marker 1 (×2). B: video#0 → 2 (×3), then image#0 → 3 (×1).
        expected = torch.tensor([1.0] * 2 + [2.0] * 3 + [3.0] * 1)
        assert torch.equal(out[:, 0], expected)

    def test_one_populated_modality_ignores_empty_group_peer(self):
        # Two image items, no explicit manifest — reorder still works by
        # synthesizing a trivial manifest. Qwen's shared image/video group also
        # contributes an empty video view; it must not force an image copy.
        mp = _mp(embedding_lengths=[2, 3], buckets={"image": {}})
        image_embeddings = torch.cat([self._marker_tensor(7, 2), self._marker_tensor(9, 3)], dim=0)
        per_modality_embeds = {
            "image": image_embeddings,
            "video": image_embeddings.new_empty((0, image_embeddings.shape[1])),
        }
        per_modality_lengths = {"image": [2, 3], "video": []}
        out = _reorder_embeds_by_manifest(
            [(mp.item_metadata.item_refs, ())], per_modality_embeds, per_modality_lengths
        )
        expected = torch.tensor([7.0] * 2 + [9.0] * 3)
        assert torch.equal(out[:, 0], expected)
        assert out is image_embeddings

    def test_single_modality_noncanonical_manifest_is_reordered(self):
        mp = _mp(
            item_order=[
                {"modality": "image", "index": 1},
                {"modality": "image", "index": 0},
            ],
            buckets={"image": {}},
        )
        image_embeddings = torch.cat([self._marker_tensor(7, 2), self._marker_tensor(9, 3)], dim=0)
        out = _reorder_embeds_by_manifest(
            [(mp.item_metadata.item_refs, ())], {"image": image_embeddings}, {"image": [2, 3]}
        )

        expected = torch.tensor([9.0] * 3 + [7.0] * 2)
        assert torch.equal(out[:, 0], expected)
        assert out is not image_embeddings

    @pytest.mark.parametrize("b_item_order", [[0, 1], [1, 0]])
    def test_single_modality_batch_reorders_only_out_of_order_requests(self, b_item_order):
        # Request A uses a synthesized manifest; request B lists its two images
        # explicitly. The request-major encoder output is already in prompt
        # order unless B lists its second image first.
        mp_a = _mp(embedding_lengths=[2], buckets={"image": {}})
        mp_b = _mp(
            item_order=[{"modality": "image", "index": i} for i in b_item_order],
            buckets={"image": {}},
        )
        image_embeddings = torch.cat(
            [self._marker_tensor(1, 2), self._marker_tensor(2, 3), self._marker_tensor(3, 1)],
            dim=0,
        )
        out = _reorder_embeds_by_manifest(
            [(mp_a.item_metadata.item_refs, ()), (mp_b.item_metadata.item_refs, ())],
            {"image": image_embeddings},
            {"image": [2, 3, 1]},
        )
        b_rows = {0: [2.0] * 3, 1: [3.0] * 1}
        expected = torch.tensor([1.0] * 2 + b_rows[b_item_order[0]] + b_rows[b_item_order[1]])
        assert torch.equal(out[:, 0], expected)
        assert (out is image_embeddings) == (b_item_order == [0, 1])

    def test_empty_bookkeeping_returns_typed_empty(self):
        # Mirrors the executor KV-cache profiling pass: the dummy batch runs the
        # encoder but carries no ``multimodal_embedding_lengths``, so per-modality
        # lengths are empty and the sliced embeds are zero-row. Reorder must
        # return a correctly-typed empty tensor, not crash on ``torch.cat([])``.
        mp = _mp(buckets={"image": {}})  # modality present, no manifest, no lengths
        hidden = 5
        per_modality_embeds = {"image": torch.empty((0, hidden), dtype=torch.float16)}
        per_modality_lengths = {"image": []}
        out = _reorder_embeds_by_manifest(
            [(mp.item_metadata.item_refs, ())], per_modality_embeds, per_modality_lengths
        )
        assert out.shape == (0, hidden)
        assert out.dtype == torch.float16

    def test_no_embeds_returns_empty(self):
        # Defensive: no group produced embeddings at all — return an empty
        # tensor rather than crashing.
        mp = _mp(buckets={})
        out = _reorder_embeds_by_manifest([(mp.item_metadata.item_refs, ())], {}, {})
        assert out.numel() == 0


def _group_request(refs, lengths, buckets):
    metadata = MultimodalItemMetadata(
        refs,
        output_embedding_lengths=lengths,
        input_partitions={modality: {"pixels": MultimodalItemPartition()} for modality in buckets},
    )
    return MultimodalParams(
        multimodal_data={**buckets, MULTIMODAL_ENCODER_ITEM_METADATA_KEY: metadata}
    )


def test_default_builder_batches_compatible_shapes_within_each_modality():
    calls = []

    def encode(pixels):
        calls.append(pixels)
        return pixels[:, :1]

    image = torch.tensor([[1, 1], [2, 2]])
    video = torch.tensor([[3, 3]])
    wide = torch.tensor([[4, 4, 4]])
    requests = [
        _group_request(
            [("image", 1), ("video", 0), ("image", 0)],
            [1, 1, 1],
            {"image": {"pixels": image}, "video": {"pixels": video}},
        ),
        _group_request([("image", 0)], [1], {"image": {"pixels": wide}}),
        _group_request([("image", 0)], [1], {"image": {"pixels": torch.tensor([[5, 5]])}}),
    ]
    # An optional empty bucket has no items or partition declaration.
    requests[1].multimodal_data["video"] = {}
    out = encode_multimodal_by_groups([EncoderGroup(("image", "video"), encode)], requests)
    assert out.flatten().tolist() == [2, 3, 1, 4, 5]
    assert [tensor.shape for tensor in calls] == [(3, 2), (1, 3), (1, 2)]
    assert calls[1] is wide and calls[2] is video


def test_default_builder_rejects_item_counts_that_cancel_across_requests():
    pixels = torch.tensor([[1], [2]])
    requests = [
        _group_request([("image", 0)], [1], {"image": {"pixels": batch}})
        for batch in (pixels, pixels[:0])
    ]
    with pytest.raises(MultimodalEncoderContractError, match="item count"):
        encode_multimodal_by_groups([EncoderGroup(("image",), lambda pixels: pixels)], requests)


@pytest.mark.parametrize("nested", [False, True])
def test_default_builder_splits_list_items_within_a_request(nested):
    calls = []

    def encode(pixels):
        if nested:
            pixels = [frames[0] for frames in pixels]
        calls.append([value.shape for value in pixels])
        return torch.stack([value.sum() for value in pixels])

    small, wide = torch.tensor([1]), torch.tensor([2, 3])
    if nested:
        small, wide = [small], [wide]
    requests = [
        _group_request([("image", 1), ("image", 0)], [1, 1], {"image": {"pixels": [small, wide]}}),
        _group_request([("image", 0)], [1], {"image": {"pixels": [small]}}),
    ]
    out = encode_multimodal_by_groups([EncoderGroup(("image",), encode)], requests)
    assert calls == [[(1,), (1,)], [(2,)]]
    assert out.tolist() == [5, 1, 1]
