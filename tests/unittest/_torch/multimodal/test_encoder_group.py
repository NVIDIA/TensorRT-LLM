# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the generic encoder-group primitives.

Covers the pure helpers behind ``encode_multimodal_by_groups``:

* ``_lengths_by_modality`` — invert prompt-ordered ``multimodal_embedding_lengths``
  into per-modality per-item lists using ``item_order``.
* ``MultimodalParams.get_item_layout`` — item metadata with legacy fallbacks.
* ``_reorder_embeds_by_manifest`` — slice per-modality tensors and concat in
  each request's prompt-order manifest.
* ``EncoderGroup`` inputs — ``input_fields`` packing (concat, padding, dtype
  split, shared records) and ``build_batched_input`` pass-through.
"""

from typing import NamedTuple

import pytest
import torch

from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
    EncoderGroup,
    MultimodalEncoderContractError,
    _input_field_calls,
    _lengths_by_modality,
    _reorder_embeds_by_manifest,
    encode_multimodal_by_groups,
)
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalItemMetadata,
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


class _Scale(NamedTuple):
    """A hashable per-payload record, like a normalization record."""

    factor: float


def _single_call_inputs(group, params):
    """The kwargs of the one encoder call that `group.input_fields` packs."""
    _, calls = _input_field_calls(group, params)
    ((_, inputs),) = calls
    return inputs


class TestEncoderGroupInputs:
    _INPUT_FIELDS = {
        "values": {"image": "image_values", "video": "video_values"},
        "sizes": {"image": "image_sizes", "video": "video_sizes"},
    }

    @staticmethod
    def _encode(values, sizes):
        return values

    @staticmethod
    def _payload(modality, marker, rows):
        return {
            f"{modality}_values": torch.full((rows, 2), float(marker)),
            f"{modality}_sizes": torch.tensor([[rows]]),
        }

    def _group(self):
        return EncoderGroup(("image", "video"), self._encode, input_fields=self._INPUT_FIELDS)

    def test_requires_exactly_one_input_declaration(self):
        with pytest.raises(TypeError, match="exactly one"):
            EncoderGroup(("image",), self._encode)
        with pytest.raises(TypeError, match="exactly one"):
            EncoderGroup(
                ("image",),
                self._encode,
                lambda params: {},
                input_fields={"values": {"image": "image_values"}},
            )

    def test_concat_orders_modality_then_request_and_skips_absent_payloads(self):
        params = [
            MultimodalParams(multimodal_data={"video": self._payload("video", 1, 3)}),
            MultimodalParams(multimodal_data={"image": None, "audio": {}}),
            MultimodalParams(
                multimodal_data={
                    "image": self._payload("image", 2, 1),
                    "video": self._payload("video", 3, 2),
                }
            ),
            MultimodalParams(multimodal_data={"image": self._payload("image", 4, 2)}),
        ]
        payloads, calls = _input_field_calls(self._group(), params)
        assert payloads == [("image", 2), ("image", 3), ("video", 0), ("video", 2)]
        ((members, inputs),) = calls
        assert members == [0, 1, 2, 3]
        assert list(inputs) == ["values", "sizes"]
        assert torch.equal(inputs["values"][:, 0], torch.tensor([2.0, 4, 4, 1, 1, 1, 3, 3]))
        assert torch.equal(inputs["sizes"], torch.tensor([[1], [2], [3], [2]]))

    def test_concat_copies_once_per_field_even_for_one_request(self, monkeypatch):
        payload = self._payload("image", 1, 2)
        calls = []
        cat = torch.cat

        def counting_cat(tensors, *args, **kwargs):
            calls.append(len(tensors))
            return cat(tensors, *args, **kwargs)

        monkeypatch.setattr(torch, "cat", counting_cat)
        inputs = _single_call_inputs(
            self._group(), [MultimodalParams(multimodal_data={"image": payload})]
        )
        monkeypatch.undo()
        assert calls == [1, 1]
        assert torch.equal(inputs["values"], payload["image_values"])
        assert inputs["values"].data_ptr() != payload["image_values"].data_ptr()

    def test_equal_item_shapes_take_one_cat_and_no_padding(self, monkeypatch):
        params = [
            MultimodalParams(multimodal_data={"image": self._payload("image", marker, rows)})
            for marker, rows in ((1, 2), (2, 1), (3, 3))
        ]
        cats, allocations = [], []
        cat, new_zeros = torch.cat, torch.Tensor.new_zeros

        def counting_cat(tensors, *args, **kwargs):
            cats.append(len(tensors))
            return cat(tensors, *args, **kwargs)

        def counting_new_zeros(tensor, *args, **kwargs):
            allocations.append(args)
            return new_zeros(tensor, *args, **kwargs)

        monkeypatch.setattr(torch, "cat", counting_cat)
        monkeypatch.setattr(torch.Tensor, "new_zeros", counting_new_zeros)
        inputs = _single_call_inputs(self._group(), params)
        monkeypatch.undo()
        assert cats == [3, 3]
        assert allocations == []
        assert torch.equal(inputs["values"][:, 0], torch.tensor([1.0, 1, 2, 3, 3, 3]))

    def test_ragged_item_dims_are_zero_padded_at_their_end(self):
        group = EncoderGroup(
            ("audio",),
            lambda features, mask: features,
            input_fields={
                "features": {"audio": "features"},
                "mask": {"audio": "mask"},
            },
        )
        first = torch.arange(30, dtype=torch.float32).reshape(2, 5, 3) + 1
        second = torch.arange(9, dtype=torch.float32).reshape(1, 3, 3) + 100
        params = [
            MultimodalParams(
                multimodal_data={"audio": {"features": first, "mask": torch.ones(2, 5, dtype=bool)}}
            ),
            MultimodalParams(
                multimodal_data={
                    "audio": {"features": second, "mask": torch.ones(1, 3, dtype=torch.bool)}
                }
            ),
        ]
        inputs = _single_call_inputs(group, params)
        features, mask = inputs["features"], inputs["mask"]
        assert features.shape == (3, 5, 3) and features.dtype == torch.float32
        assert torch.equal(features[:2], first)
        assert torch.equal(features[2, :3], second[0])
        assert torch.equal(features[2, 3:], torch.zeros(2, 3))
        assert mask.dtype == torch.bool
        assert mask.tolist() == [[True] * 5, [True] * 5, [True] * 3 + [False] * 2]

    def test_values_that_differ_in_two_dims_are_padded_in_both(self):
        group = EncoderGroup(
            ("image",), lambda values: values, input_fields={"values": {"image": "v"}}
        )
        params = [
            MultimodalParams(
                multimodal_data={"image": {"v": torch.ones(1, 2, 4, dtype=torch.int64)}}
            ),
            MultimodalParams(multimodal_data={"image": {"v": torch.full((2, 3, 1), 7)}}),
        ]
        values = _single_call_inputs(group, params)["values"]
        assert values.shape == (3, 3, 4) and values.dtype == torch.int64
        assert values[0].tolist() == [[1, 1, 1, 1], [1, 1, 1, 1], [0, 0, 0, 0]]
        assert values[1].tolist() == [[7, 0, 0, 0]] * 3

    def test_missing_field_reads_as_none_in_its_own_call(self):
        payload = self._payload("image", 1, 2)
        other = self._payload("image", 2, 1)
        del payload["image_sizes"]
        payloads, calls = _input_field_calls(
            self._group(),
            [
                MultimodalParams(multimodal_data={"image": payload}),
                MultimodalParams(multimodal_data={"image": other}),
            ],
        )
        assert payloads == [("image", 0), ("image", 1)]
        assert [members for members, _ in calls] == [[0], [1]]
        assert calls[0][1]["sizes"] is None
        assert torch.equal(calls[1][1]["sizes"], other["image_sizes"])

    def test_list_fields_are_rejected(self):
        payload = self._payload("image", 1, 2)
        payload["image_sizes"] = [2]
        with pytest.raises(TypeError, match="emit per-item values as a tensor"):
            _input_field_calls(
                self._group(), [MultimodalParams(multimodal_data={"image": payload})]
            )

    def test_dtype_split_encodes_each_dtype_apart_and_restores_the_layout(self):
        # Raw uint8 pixels next to already normalized ones, as with device normalization.
        record = _Scale(0.5)
        calls = []

        def encode(values, sizes, normalization=None):
            calls.append((values.dtype, normalization))
            if values.dtype == torch.uint8:
                values = values.to(torch.bfloat16) * normalization.factor
            return values

        group = EncoderGroup(
            ("image", "video"),
            encode,
            input_fields={
                **self._INPUT_FIELDS,
                "normalization": {"image": "normalization", "video": "normalization"},
            },
        )

        def payload(modality, marker, rows, raw):
            data = self._payload(modality, marker, rows)
            if raw:
                data[f"{modality}_values"] = data[f"{modality}_values"].mul(2).to(torch.uint8)
                data["normalization"] = record
            else:
                data[f"{modality}_values"] = data[f"{modality}_values"].to(torch.bfloat16)
            return data

        layout = (("image", 1, 2, True), ("video", 2, 1, False), ("image", 3, 1, False))
        params = [
            _mp(embedding_lengths=[rows], buckets={modality: payload(modality, marker, rows, raw)})
            for modality, marker, rows, raw in layout
        ]
        params.append(
            _mp(
                item_order=[{"modality": "image", "index": 0}, {"modality": "video", "index": 0}],
                embedding_lengths=[2, 3],
                buckets={
                    "image": payload("image", 4, 2, False),
                    "video": payload("video", 5, 3, True),
                },
            )
        )
        out = encode_multimodal_by_groups([group], params)
        assert calls == [(torch.uint8, record), (torch.bfloat16, None)]
        assert out.dtype == torch.bfloat16
        assert out[:, 0].tolist() == [1.0, 1, 2, 3, 4, 4, 5, 5, 5]
        # A uniform batch makes exactly one call with the original single-call packing.
        calls.clear()
        encode_multimodal_by_groups([group], params[1:3])
        assert calls == [(torch.bfloat16, None)]

    def test_payloads_with_different_records_are_encoded_apart(self):
        calls = []
        group = EncoderGroup(
            ("image",),
            lambda values, normalization: calls.append(normalization) or values,
            input_fields={
                "values": {"image": "values"},
                "normalization": {"image": "normalization"},
            },
        )
        params = [
            _mp(
                embedding_lengths=[1],
                buckets={"image": {"values": torch.full((1, 1), float(i)), "normalization": n}},
            )
            for i, n in enumerate(("a", "b", "a"))
        ]
        out = encode_multimodal_by_groups([group], params)
        assert calls == ["a", "b"]
        assert out[:, 0].tolist() == [0.0, 1.0, 2.0]

    def test_builder_receives_original_params(self):
        received = []

        def build(params):
            received.extend(params)
            return {
                "values": torch.cat([p.multimodal_data["image"]["image_values"] for p in params])
            }

        group = EncoderGroup(("image",), lambda values: values, build)
        assert group.build_batched_input is build
        assert group.input_fields is None
        params = [
            _mp(embedding_lengths=[rows], buckets={"image": self._payload("image", marker, rows)})
            for marker, rows in ((1, 2), (2, 1))
        ]
        out = encode_multimodal_by_groups([group], params)
        assert len(received) == len(params)
        assert all(actual is expected for actual, expected in zip(received, params))
        assert torch.equal(out[:, 0], torch.tensor([1.0, 1, 2]))

    def test_encoder_rows_must_match_item_metadata(self):
        # Extra rows, such as padding an encoder kept, must not be sliced away silently.
        group = EncoderGroup(
            ("image",),
            lambda values, sizes: torch.cat([values, values[:1]]),
            input_fields=self._INPUT_FIELDS,
        )
        params = [_mp(embedding_lengths=[2], buckets={"image": self._payload("image", 1, 2)})]
        with pytest.raises(MultimodalEncoderContractError, match="returned 3 rows"):
            encode_multimodal_by_groups([group], params)
