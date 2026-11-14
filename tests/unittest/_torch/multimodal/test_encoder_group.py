# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the generic encoder-group primitives.

Covers the pure helpers behind ``encode_multimodal_by_groups``:

* ``_lengths_by_modality`` — invert prompt-ordered ``multimodal_embedding_lengths``
  into per-modality per-item lists using ``item_order``.
* ``MultimodalParams.get_item_layout`` — item metadata with legacy fallbacks.
* ``_reorder_embeds_by_manifest`` — slice per-modality tensors and concat in
  each request's prompt-order manifest.
* ``EncoderGroup`` batching — the default builder, splitting payloads that
  cannot share a call, and the encoder row-count check.
"""

import functools
from types import SimpleNamespace
from typing import NamedTuple

import pytest
import torch

from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
    EncoderGroup,
    MultimodalEncoderContractError,
    _device_keys,
    _lengths_by_modality,
    _reorder_embeds_by_manifest,
    build_batched_input_by_partitions,
    encode_multimodal_by_groups,
    release_encoded_media,
    request_device_paths,
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


def _rows(*markers_and_rows, dtype=torch.float32):
    """A packed payload: ``rows`` rows holding ``marker`` per item."""
    return torch.cat(
        [torch.full((rows, 1), float(marker)) for marker, rows in markers_and_rows]
    ).to(dtype)


class _Scale(NamedTuple):
    """A per-processor record, like a normalization record."""

    factor: float


def _request(items, buckets, partitions):
    """One request: prompt-ordered ``(modality, index, rows)`` items and its payloads."""
    metadata = MultimodalItemMetadata(
        [(modality, index) for modality, index, _ in items],
        output_embedding_lengths=[rows for *_, rows in items],
        input_partitions=partitions,
    )
    metadata.validate()
    return MultimodalParams(
        multimodal_data={**buckets, MULTIMODAL_ENCODER_ITEM_METADATA_KEY: metadata}
    )


def _image(markers_and_rows, dtype=torch.float32, scale=None):
    """An image payload with one packed item per (marker, rows) and its partitions."""
    data = {"pixels": _rows(*markers_and_rows, dtype=dtype), "undeclared": torch.zeros(1)}
    partitions = {
        "pixels": MultimodalItemPartition(item_lengths=tuple(r for _, r in markers_and_rows))
    }
    if scale is not None:
        data["scale"] = scale
        partitions["scale"] = MultimodalItemPartition(item_axis=None)
    return data, partitions


class TestDefaultBuilder:
    def test_group_binds_the_default_builder_and_keeps_its_own(self):
        group = EncoderGroup(("image", "video"), lambda pixels: pixels)
        builder = group.build_batched_input
        assert isinstance(builder, functools.partial)
        assert builder.func is build_batched_input_by_partitions
        assert builder.args == (("image", "video"),)

        def build(params):
            return {}

        # Positional, as models construct groups on main.
        assert EncoderGroup(("image",), lambda: None, build).build_batched_input is build

    def test_modality_then_request_order_with_declared_keys_only(self):
        image_a, image_a_partitions = _image([(1, 2)])
        image_b, image_b_partitions = _image([(3, 1), (4, 2)])
        video = {"pixels": _rows((2, 3)), "frames": torch.tensor([3])}
        video_partitions = {"pixels": MultimodalItemPartition(item_lengths=(3,))}
        params = [
            _request(
                [("video", 0, 3), ("image", 0, 2)],
                {"image": image_a, "video": video},
                {"image": image_a_partitions, "video": video_partitions},
            ),
            _request([("audio", 0, 1)], {"audio": {"pixels": torch.ones(1, 1)}}, {}),
            _request(
                [("image", 0, 1), ("image", 1, 2)],
                {"image": image_b},
                {"image": image_b_partitions},
            ),
        ]
        group_params = [params[0], params[2]]
        inputs = build_batched_input_by_partitions(("image", "video"), group_params)
        # Image items of both requests, then the video item, all batched in `pixels`.
        assert list(inputs) == ["pixels"]
        assert inputs["pixels"][:, 0].tolist() == [1, 1, 3, 4, 4, 2, 2, 2]

    def test_different_keys_per_modality_stay_apart(self):
        image, image_partitions = _image([(1, 2)])
        video = {"video_pixels": _rows((2, 1)), "grid": torch.tensor([[1, 1, 1]])}
        params = [
            _request(
                [("image", 0, 2), ("video", 0, 1)],
                {"image": image, "video": video},
                {
                    "image": image_partitions,
                    "video": {
                        "video_pixels": MultimodalItemPartition(item_lengths=(1,)),
                        "grid": MultimodalItemPartition(),
                    },
                },
            )
        ]
        inputs = build_batched_input_by_partitions(("image", "video"), params)
        assert sorted(inputs) == ["grid", "pixels", "video_pixels"]
        assert torch.equal(inputs["pixels"], image["pixels"])
        assert torch.equal(inputs["video_pixels"], video["video_pixels"])

    def test_padding_value_of_a_key_reaches_the_encoder(self):
        partitions = {"positions": MultimodalItemPartition(padding_value=-1)}
        params = [
            _request(
                [("image", 0, 1)],
                {"image": {"positions": torch.tensor([[1, 2, 3]])}},
                {"image": partitions},
            ),
            _request(
                [("image", 0, 1)],
                {"image": {"positions": torch.tensor([[4]])}},
                {"image": partitions},
            ),
        ]
        inputs = build_batched_input_by_partitions(("image",), params)
        assert inputs["positions"].tolist() == [[1, 2, 3], [4, -1, -1]]

    @pytest.mark.parametrize(
        "other",
        [MultimodalItemPartition(1), MultimodalItemPartition(padding_value=-1)],
        ids=["item_axis", "padding_value"],
    )
    def test_payloads_must_share_how_a_key_is_batched(self, other):
        params = [
            _request(
                [("image", 0, 1)],
                {"image": {"pixels": torch.ones(1, 2)}},
                {"image": {"pixels": partition}},
            )
            for partition in (MultimodalItemPartition(), other)
        ]
        with pytest.raises(ValueError, match="share their item axis and padding value"):
            build_batched_input_by_partitions(("image",), params)
        group = EncoderGroup(("image",), lambda pixels: pixels)
        with pytest.raises(ValueError, match="share their item axis and padding value"):
            encode_multimodal_by_groups([group], params)

    def test_payload_without_partitions_is_rejected(self):
        params = [_request([("image", 0, 1)], {"image": {"pixels": torch.ones(1, 1)}}, {})]
        with pytest.raises(MultimodalEncoderContractError, match="declares no input partitions"):
            build_batched_input_by_partitions(("image",), params)

    def test_packed_size_must_match_item_lengths(self):
        # 3 rows for one item of 2 rows; the encoder returns the 2 rows the metadata expects,
        # so only the size check finds it.
        params = [
            _request(
                [("image", 0, 2)],
                {"image": {"pixels": torch.arange(6.0).view(3, 2)}},
                {"image": {"pixels": MultimodalItemPartition(item_lengths=(2,))}},
            )
        ]
        with pytest.raises(ValueError, match="does not match its item lengths"):
            build_batched_input_by_partitions(("image",), params)
        group = EncoderGroup(("image",), lambda pixels: pixels[:2])
        with pytest.raises(ValueError, match="does not match its item lengths"):
            encode_multimodal_by_groups([group], params)

    def test_item_metadata_must_be_the_record_type(self):
        record = _request(
            [("image", 0, 1)],
            {"image": {"pixels": torch.ones(1, 1)}},
            {"image": {"pixels": MultimodalItemPartition()}},
        )
        stand_in = SimpleNamespace(
            item_refs=record.item_metadata.item_refs,
            output_embedding_lengths=record.item_metadata.output_embedding_lengths,
            input_partitions=record.item_metadata.input_partitions,
        )
        params = [
            MultimodalParams(
                multimodal_data={
                    "image": record.multimodal_data["image"],
                    MULTIMODAL_ENCODER_ITEM_METADATA_KEY: stand_in,
                }
            )
        ]
        with pytest.raises(TypeError, match="must be a MultimodalItemMetadata"):
            build_batched_input_by_partitions(("image",), params)
        group = EncoderGroup(("image",), lambda pixels: pixels)
        with pytest.raises(TypeError, match="must be a MultimodalItemMetadata"):
            encode_multimodal_by_groups([group], params)


class TestEncodeByGroups:
    @staticmethod
    def _recording_group(modalities=("image", "video")):
        calls = []

        def encode(pixels, scale=None):
            calls.append((pixels.dtype, scale, pixels[:, 0].tolist()))
            return pixels.to(torch.float32) * (1 if scale is None else scale.factor)

        return EncoderGroup(modalities, encode), calls

    def test_compatible_payloads_make_one_call_and_restore_prompt_order(self):
        group, calls = self._recording_group()
        image_a, image_a_partitions = _image([(1, 2), (3, 1)])
        video, video_partitions = _image([(2, 3)])
        image_b, image_b_partitions = _image([(4, 1)])
        params = [
            _request(
                [("image", 1, 1), ("video", 0, 3), ("image", 0, 2)],
                {"image": image_a, "video": video},
                {"image": image_a_partitions, "video": video_partitions},
            ),
            _request([("image", 0, 1)], {"image": image_b}, {"image": image_b_partitions}),
        ]
        out = encode_multimodal_by_groups([group], params)
        assert len(calls) == 1
        assert calls[0][2] == [1, 1, 3, 4, 2, 2, 2]
        assert out[:, 0].tolist() == [3, 2, 2, 2, 1, 1, 4]

    def test_payloads_with_different_shared_values_are_encoded_apart(self):
        group, calls = self._recording_group(("image",))
        params = []
        for marker, factor in ((1, 2.0), (2, 3.0), (3, 2.0)):
            image, partitions = _image([(marker, 1)], scale=_Scale(factor))
            params.append(_request([("image", 0, 1)], {"image": image}, {"image": partitions}))
        out = encode_multimodal_by_groups([group], params)
        assert [(scale, rows) for _, scale, rows in calls] == [
            (_Scale(2.0), [1.0, 3.0]),
            (_Scale(3.0), [2.0]),
        ]
        assert out[:, 0].tolist() == [2.0, 6.0, 6.0]

    def test_equal_shared_records_share_one_call(self):
        # Each request carries its own record object; equal records still share a call.
        group, calls = self._recording_group(("image",))
        params = []
        for marker in (1, 2):
            image, partitions = _image([(marker, 1)], scale=_Scale(2.0))
            params.append(_request([("image", 0, 1)], {"image": image}, {"image": partitions}))
        assert (
            params[0].multimodal_data["image"]["scale"]
            is not (params[1].multimodal_data["image"]["scale"])
        )
        out = encode_multimodal_by_groups([group], params)
        assert [(scale, rows) for _, scale, rows in calls] == [(_Scale(2.0), [1.0, 2.0])]
        assert out[:, 0].tolist() == [2.0, 4.0]

    def test_payloads_with_different_dtypes_are_encoded_apart(self):
        # Raw uint8 pixels next to already normalized ones, also inside one request.
        group, calls = self._recording_group()
        raw, raw_partitions = _image([(1, 2)], dtype=torch.uint8, scale=_Scale(0.5))
        image, image_partitions = _image([(3, 1)], dtype=torch.bfloat16)
        video, video_partitions = _image([(5, 3)], dtype=torch.uint8, scale=_Scale(0.5))
        params = [
            _request([("image", 0, 2)], {"image": raw}, {"image": raw_partitions}),
            _request(
                [("video", 0, 3), ("image", 0, 1)],
                {"image": image, "video": video},
                {"image": image_partitions, "video": video_partitions},
            ),
        ]
        out = encode_multimodal_by_groups([group], params)
        assert [(dtype, scale, rows) for dtype, scale, rows in calls] == [
            (torch.uint8, _Scale(0.5), [1, 1, 5, 5, 5]),
            (torch.bfloat16, None, [3.0]),
        ]
        assert out.dtype == torch.float32
        assert out[:, 0].tolist() == [0.5, 0.5, 2.5, 2.5, 2.5, 3.0]

    def test_own_builder_gets_the_original_params_in_one_call(self):
        received = []

        def build(params):
            received.append(params)
            return {
                "pixels": torch.cat([p.multimodal_data["image"]["pixels"].float() for p in params])
            }

        group = EncoderGroup(("image",), lambda pixels: pixels, build)
        raw, raw_partitions = _image([(1, 2)], dtype=torch.uint8, scale=_Scale(0.5))
        image, image_partitions = _image([(3, 1)], dtype=torch.bfloat16)
        params = [
            _request([("image", 0, 2)], {"image": raw}, {"image": raw_partitions}),
            _request([("image", 0, 1)], {"image": image}, {"image": image_partitions}),
        ]
        out = encode_multimodal_by_groups([group], params)
        assert len(received) == 1
        assert all(actual is expected for actual, expected in zip(received[0], params, strict=True))
        assert out[:, 0].tolist() == [1.0, 1.0, 3.0]

    @pytest.mark.parametrize(
        "factors, message",
        [((1.0, 1.0), "returned 5 rows, item metadata expects 4"), ((1.0, 2.0), "returned 3 rows")],
        ids=["one_call", "split"],
    )
    def test_encoder_rows_must_match_item_metadata(self, factors, message):
        # Extra rows, such as padding an encoder kept, must not be sliced away silently.
        group = EncoderGroup(("image",), lambda pixels, scale=None: torch.cat([pixels, pixels[:1]]))
        params = []
        for factor in factors:
            image, partitions = _image([(1, 2)], scale=_Scale(factor))
            params.append(_request([("image", 0, 2)], {"image": image}, {"image": partitions}))
        with pytest.raises(MultimodalEncoderContractError, match=message):
            encode_multimodal_by_groups([group], params)


class TestDevicePaths:
    """Which paths the executor moves per request, and which the encoder call batches."""

    PATHS = [
        "image.pixel_values",
        "video.pixel_values_videos",
        "image",
        "audio.audio_features.data",
        "multimodal_embedding",
        "mrope_config.mrope_position_ids",
    ]

    @staticmethod
    def _model(groups, paths=PATHS):
        return SimpleNamespace(mm_encoder_groups=groups, multimodal_data_device_paths=paths)

    def test_default_builder_models_leave_batched_media_keys_to_the_encoder_call(self):
        model = self._model((EncoderGroup(("image", "video"), lambda **kwargs: None),))
        # A whole modality or a nested key is not a key the builder batches: still per request.
        assert request_device_paths(model) == [
            "image",
            "audio.audio_features.data",
            "multimodal_embedding",
            "mrope_config.mrope_position_ids",
        ]

    def test_media_of_a_modality_without_a_group_still_moves_per_request(self):
        model = self._model((EncoderGroup(("image",), lambda **kwargs: None),))
        assert "video.pixel_values_videos" in request_device_paths(model)
        assert "image.pixel_values" not in request_device_paths(model)

    @pytest.mark.parametrize(
        "groups",
        [
            (),
            (EncoderGroup(("image",), lambda **kwargs: None, lambda params: {}),),
            (
                EncoderGroup(("image",), lambda **kwargs: None),
                EncoderGroup(("video",), lambda **kwargs: None, lambda params: {}),
            ),
            (SimpleNamespace(modalities=("image",)),),
            None,
        ],
        ids=["no_groups", "own_builder", "one_own_builder", "duck_typed", "not_a_sequence"],
    )
    def test_other_models_move_every_path_per_request(self, groups):
        assert request_device_paths(self._model(groups)) == self.PATHS

    def test_no_path_list_moves_everything(self):
        group = EncoderGroup(("image",), lambda **kwargs: None)
        assert request_device_paths(SimpleNamespace(mm_encoder_groups=(group,))) is None
        assert request_device_paths(self._model((group,), paths=None)) is None

    def test_only_media_paths_give_an_empty_list(self):
        model = self._model((EncoderGroup(("audio",), lambda **kwargs: None),), ["audio.features"])
        assert request_device_paths(model) == []

    def test_device_keys_tell_the_call_from_the_per_request_paths(self):
        # True: the encoder call makes the key on the GPU; False: the executor moves it per
        # request (a whole modality is key ""; a nested path names its payload key).
        assert _device_keys(self.PATHS) == {
            "image": {"pixel_values": True, "": False},
            "video": {"pixel_values_videos": True},
            "audio": {"audio_features": False},
        }
        assert _device_keys(["image.a.b", "image.a"]) == {"image": {"a": True}}
        assert _device_keys(None) == {}


class TestBatchAs:
    @staticmethod
    def _image_video(video_dtype=torch.float32):
        image, image_partitions = _image([(1, 2), (2, 1)])
        video = {"frames": _rows((5, 3)).to(video_dtype), "frame_grid": torch.tensor([[3]])}
        video_partitions = {
            "frames": MultimodalItemPartition(item_lengths=(3,)),
            "frame_grid": MultimodalItemPartition(),
        }
        image["grid"] = torch.tensor([[2], [1]])
        image_partitions["grid"] = MultimodalItemPartition()
        return _request(
            [("video", 0, 3), ("image", 0, 2), ("image", 1, 1)],
            {"image": image, "video": video},
            {"image": image_partitions, "video": video_partitions},
        )

    def test_mapped_keys_are_batched_after_the_target_key(self):
        calls = []

        def encode(pixels, grid):
            calls.append((pixels[:, 0].tolist(), grid[:, 0].tolist()))
            return pixels

        group = EncoderGroup(
            ("image", "video"), encode, batch_as={"frames": "pixels", "frame_grid": "grid"}
        )
        out = encode_multimodal_by_groups([group], [self._image_video()])
        assert calls == [([1, 1, 2, 5, 5, 5], [2, 1, 3])]
        assert out[:, 0].tolist() == [5, 5, 5, 1, 1, 2]

    def test_mapped_keys_with_another_dtype_are_encoded_apart(self):
        calls = []

        def encode(pixels, grid):
            calls.append((pixels.dtype, pixels[:, 0].tolist()))
            return pixels.float()

        group = EncoderGroup(
            ("image", "video"), encode, batch_as={"frames": "pixels", "frame_grid": "grid"}
        )
        out = encode_multimodal_by_groups([group], [self._image_video(torch.bfloat16)])
        assert calls == [(torch.float32, [1, 1, 2]), (torch.bfloat16, [5, 5, 5])]
        assert out[:, 0].tolist() == [5, 5, 5, 1, 1, 2]

    def test_mapped_keys_with_another_device_policy_are_encoded_apart(self):
        """A key the owner sends to the GPU never shares a batch with one it keeps on the host."""
        calls = []

        def encode(pixels, grid):
            calls.append(pixels[:, 0].tolist())
            return pixels

        group = EncoderGroup(
            ("image", "video"), encode, batch_as={"frames": "pixels", "frame_grid": "grid"}
        )
        # The executor moves the whole image modality per request (here it stays where the test
        # put it); the video frames are kept on the host, so they get their own call.
        out = encode_multimodal_by_groups([group], [self._image_video()], ["image"])
        assert calls == [[1, 1, 2], [5, 5, 5]]
        assert out[:, 0].tolist() == [5, 5, 5, 1, 1, 2]


class TestReleaseEncodedMedia:
    """Raw media leaves the request once its whole output is held and nothing encodes again."""

    @staticmethod
    def _request(media_device="cpu", **extra):
        data = {
            "image": {"pixels": torch.ones(4, 2, device=media_device), "grid": torch.ones(1, 3)},
            "mrope_config": {"mrope_position_deltas": torch.zeros(1)},
            "multimodal_embed_mask_cumsum": torch.arange(4),
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata([("image", 0)], [4], [4]),
            "multimodal_embedding": torch.zeros(4, 2),
            **extra,
        }
        return SimpleNamespace(py_multimodal_data=data, py_mm_encoder_state=None)

    def test_held_output_drops_host_media_and_keeps_the_rest(self):
        request = self._request()
        kept = {key: value for key, value in request.py_multimodal_data.items() if key != "image"}
        release_encoded_media(request)
        assert request.py_multimodal_data == kept

    @pytest.mark.parametrize(
        "change",
        ["no_output", "chunk_output", "item_scheduling", "media_on_a_device"],
    )
    def test_media_stays_while_an_encoder_call_may_read_it(self, change):
        request = self._request(media_device="meta" if change == "media_on_a_device" else "cpu")
        data = request.py_multimodal_data
        if change == "no_output":
            del data["multimodal_embedding"]
        elif change == "chunk_output":
            data["multimodal_embedding_is_chunk"] = True
        elif change == "item_scheduling":
            request.py_mm_encoder_state = object()
        release_encoded_media(request)
        assert "image" in data

    def test_requests_without_media_are_left_alone(self):
        request = SimpleNamespace(py_multimodal_data=None, py_mm_encoder_state=None)
        release_encoded_media(request)
        assert request.py_multimodal_data is None
