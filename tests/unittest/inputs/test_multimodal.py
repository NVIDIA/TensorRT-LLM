# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for MultimodalRuntimeData cumsum math and the flat-mask producer."""

from types import SimpleNamespace
from typing import get_args
from unittest.mock import patch

import numpy as np
import pytest
import torch

from tensorrt_llm.inputs.media_io import MEDIA_IO_REGISTRY, MediaModality
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalEncoderItemMetadata,
    MultimodalInput,
    MultimodalItemMetadata,
    MultimodalItemPartition,
    MultimodalParams,
    MultimodalRuntimeData,
    _find_mm_embedding_lengths_from_masks,
    find_mm_token_lengths,
    select_multimodal_fields,
)
from tensorrt_llm.inputs.multimodal_data import MEDIA_MODALITIES, AudioData, VideoData
from tensorrt_llm.inputs.registry import (
    create_input_processor_with_hash,
    maybe_compute_mm_embed_cumsum,
)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("indices", [[2, 0], [1, 1, 0], []])
def test_field_layout_selects_tensor_item_axis(axis: int, indices: list[int]) -> None:
    values = torch.arange(9).reshape(3, 3)
    selected = MultimodalItemPartition(item_axis=axis).select(values, indices)
    expected = torch.index_select(values, axis, torch.tensor(indices, dtype=torch.long))
    torch.testing.assert_close(selected, expected)


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize("indices", [[0, 1], [-2, -1], [2], []])
def test_batched_item_partition_preserves_contiguous_views(axis, indices) -> None:
    values = torch.arange(9).reshape(3, 3)
    selected = MultimodalItemPartition(item_axis=axis).select(values, indices)
    expected = values[indices] if axis == 0 else values[:, indices]

    torch.testing.assert_close(selected, expected)
    assert selected.untyped_storage().data_ptr() == values.untyped_storage().data_ptr()


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize(
    ("indices", "is_view"),
    [
        ([2, 0], False),
        ([1, 2], True),
        ([1, 1, 0], False),
        ([], True),
        ([-1], True),
        ([-2, -1], True),
        ([-1, 0], False),
    ],
)
def test_item_partition_selects_packed_items_and_preserves_contiguous_views(
    axis: int, indices: list[int], is_view: bool
) -> None:
    values = torch.arange(12).reshape(6, 2)
    if axis != 0:
        values = values.T
    layout = MultimodalItemPartition(item_axis=axis, item_lengths=(2, 3, 1))

    selected = layout.select(values, indices)

    items = torch.split(values, (2, 3, 1), dim=axis)
    expected = (
        torch.cat([items[i] for i in indices], dim=axis) if indices else values.narrow(axis, 0, 0)
    )
    torch.testing.assert_close(selected, expected)
    if is_view:
        assert selected.untyped_storage().data_ptr() == values.untyped_storage().data_ptr()
    else:
        assert selected.untyped_storage().data_ptr() != values.untyped_storage().data_ptr()


@pytest.mark.parametrize("sequence_type", [list, tuple])
def test_field_layout_selects_item_sequences(sequence_type) -> None:
    values = sequence_type(["a", "b", "c"])
    assert MultimodalItemPartition().select(values, [2, 0]) == sequence_type(["c", "a"])


def test_field_layout_keeps_shared_and_unlisted_values_even_when_sizes_match() -> None:
    shared = torch.arange(3)
    unlisted = ["a", "b", "c"]
    data = {
        "frames": torch.arange(6).reshape(3, 2),
        "shared": shared,
        "other": unlisted,
        "nullable_mask": None,
    }

    selected = select_multimodal_fields(
        data,
        {
            "frames": MultimodalItemPartition(),
            "shared": MultimodalItemPartition(item_axis=None),
            "optional_mask": MultimodalItemPartition(),
            "nullable_mask": MultimodalItemPartition(),
        },
        [2, 0],
    )

    torch.testing.assert_close(selected["frames"], data["frames"][[2, 0]])
    assert selected["shared"] is shared
    assert selected["other"] is unlisted
    assert "optional_mask" not in selected
    assert selected["nullable_mask"] is None
    assert data["frames"].shape == (3, 2)


def test_packed_field_layout_rejects_inconsistent_lengths() -> None:
    with pytest.raises(ValueError, match="Packed field size"):
        MultimodalItemPartition(item_lengths=(2, 3)).select(torch.zeros(4, 2), [0])


@pytest.mark.parametrize("index", [-4, 3])
@pytest.mark.parametrize("packed", [False, True])
def test_item_partition_rejects_out_of_bounds_indices(index: int, packed: bool) -> None:
    lengths = (2, 3, 1) if packed else None
    values = torch.zeros(6 if packed else 3, 2)
    with pytest.raises(IndexError):
        MultimodalItemPartition(item_lengths=lengths).select(values, [index])


def test_media_modalities_match_io_and_validation_contract() -> None:
    assert MEDIA_MODALITIES == get_args(MediaModality) == ("image", "video", "audio")
    assert set(MEDIA_MODALITIES) == set(MEDIA_IO_REGISTRY)


class _ItemMetadataFakeProcessor:
    multimodal_hashing_supported = False

    def __init__(self, existing_embedding_lengths):
        self.existing_embedding_lengths = existing_embedding_lengths

    def __call__(self, inputs, sampling_params):
        return [101, 102], {
            "multimodal_data": {
                "image": {},
                "multimodal_embedding_lengths": self.existing_embedding_lengths,
            }
        }

    def get_mm_encoder_item_metadata(self, prompt_token_ids, multimodal_data):
        assert prompt_token_ids == [101, 102]
        assert "image" in multimodal_data
        return MultimodalEncoderItemMetadata(
            item_refs=[("image", 0)],
            encoder_token_lengths=[4],
            output_embedding_lengths=[2],
        )

    def get_vocab_size(self):
        return 100

    def get_mm_token_ids(self):
        return None

    def get_mm_special_token_ids(self):
        return None


def test_mm_item_metadata_is_materialized_when_embedding_lengths_match():
    input_processor = create_input_processor_with_hash(_ItemMetadataFakeProcessor([2]))

    _, extra = input_processor({"prompt": "unused"}, sampling_params=None)

    multimodal_data = extra["multimodal_data"]
    item_metadata = multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY]
    assert item_metadata == MultimodalItemMetadata(
        item_refs=[("image", 0)],
        encoder_token_lengths=[4],
        output_embedding_lengths=[2],
    )
    assert "multimodal_embedding_lengths" not in multimodal_data
    assert "multimodal_item_refs" not in multimodal_data
    assert "multimodal_encoder_token_lengths" not in multimodal_data

    params = MultimodalParams(multimodal_data=multimodal_data)
    params.to_handle("multimodal_data")
    assert isinstance(
        params.multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY],
        MultimodalItemMetadata,
    )


def test_mm_item_metadata_rejects_mismatched_embedding_lengths():
    input_processor = create_input_processor_with_hash(_ItemMetadataFakeProcessor([1]))

    with pytest.raises(ValueError, match="do not match"):
        input_processor({"prompt": "unused"}, sampling_params=None)


def test_mm_item_metadata_none_falls_back_to_non_item_path():
    # Generic item order does not imply that physical encoder costs or
    # item scheduling are supported.
    class NoMetadataProcessor(_ItemMetadataFakeProcessor):
        def get_mm_encoder_item_metadata(self, prompt_token_ids, multimodal_data):
            return None

    input_processor = create_input_processor_with_hash(NoMetadataProcessor([2]))

    _, extra = input_processor({"prompt": "unused"}, sampling_params=None)

    metadata = extra["multimodal_data"][MULTIMODAL_ENCODER_ITEM_METADATA_KEY]
    assert metadata.item_refs == (("image", 0),)
    assert metadata.encoder_token_lengths is None
    assert metadata.output_embedding_lengths == (2,)


def test_mm_item_metadata_rejects_wrong_return_type():
    # A non-None, non-metadata return is a genuine contract violation.
    class BadMetadataProcessor(_ItemMetadataFakeProcessor):
        def get_mm_encoder_item_metadata(self, prompt_token_ids, multimodal_data):
            return ("image", 0)

    input_processor = create_input_processor_with_hash(BadMetadataProcessor([2]))

    with pytest.raises(TypeError, match="must return a"):
        input_processor({"prompt": "unused"}, sampling_params=None)


def test_mm_item_metadata_retains_three_field_tuple_contract() -> None:
    metadata = MultimodalEncoderItemMetadata([("image", 0)], [4], [2])

    item_refs, encoder_lengths, output_lengths = metadata

    assert metadata._fields == ("item_refs", "encoder_token_lengths", "output_embedding_lengths")
    assert item_refs == [("image", 0)]
    assert encoder_lengths == [4]
    assert output_lengths == [2]


@pytest.mark.parametrize(
    "selected, expected_refs, expected_costs, expected_lengths",
    [
        ([3, 0, 2], [("audio", 0), ("image", 0), ("image", 1)], [40, 20, 30], [4, 2, 3]),
        ([2, 0], [("image", 0), ("image", 1)], [30, 20], [3, 2]),
        ([], [], [], []),
    ],
)
def test_mm_item_metadata_selection_rebases_refs_without_mutating_source(
    selected, expected_refs, expected_costs, expected_lengths
) -> None:
    metadata = MultimodalEncoderItemMetadata(
        [("image", 1), ("video", 0), ("image", 0), ("audio", 0)],
        [20, 50, 30, 40],
        [2, 5, 3, 4],
    )

    subset = metadata.select(selected)

    assert subset == MultimodalEncoderItemMetadata(expected_refs, expected_costs, expected_lengths)
    assert metadata.item_refs == [("image", 1), ("video", 0), ("image", 0), ("audio", 0)]
    assert metadata.encoder_token_lengths == [20, 50, 30, 40]
    assert metadata.output_embedding_lengths == [2, 5, 3, 4]
    assert all(
        selected_field is not source_field for selected_field, source_field in zip(subset, metadata)
    )


def test_mm_item_metadata_legacy_import_and_pickle() -> None:
    import pickle

    from tensorrt_llm.inputs.registry import MultimodalEncoderItemMetadata as LegacyItemMetadata

    assert LegacyItemMetadata is MultimodalEncoderItemMetadata
    # Existing serialized records resolve the original registry module path.
    legacy_class = pickle.loads(b"ctensorrt_llm.inputs.registry\nMultimodalEncoderItemMetadata\n.")
    metadata = legacy_class([("audio", 0)], [8], [2])
    assert pickle.loads(pickle.dumps(metadata)) == metadata


@pytest.mark.parametrize("indices", [[3, 0, 2], [2, 0], [-1, 0], [], [1, 1]])
def test_item_metadata_selection_keeps_partitions_in_payload_order(indices) -> None:
    metadata = MultimodalItemMetadata(
        [("image", 1), ("audio", 0), ("image", 0), ("audio", 1)],
        output_embedding_lengths=[3, 1, 2, 4],
        input_partitions={
            "image": {"pixels": MultimodalItemPartition(item_lengths=(2, 3))},
            "audio": {
                "features": MultimodalItemPartition(1, (1, 4)),
                "shared": MultimodalItemPartition(item_axis=None),
            },
        },
    )
    metadata.validate()
    values = {"image": torch.arange(5), "audio": torch.arange(10).reshape(2, 5)}
    selected = metadata.select(indices)
    selected.validate()
    assert selected.encoder_token_lengths is None
    assert metadata.input_partitions["image"]["pixels"].item_lengths == (2, 3)
    assert metadata.input_partitions["audio"]["features"].item_lengths == (1, 4)
    for modality, partitions in selected.input_partitions.items():
        local_indices = [
            metadata.item_refs[i][1] for i in indices if metadata.item_refs[i][0] == modality
        ]
        name = "pixels" if modality == "image" else "features"
        source_partition = metadata.input_partitions[modality][name]
        value = source_partition.select(values[modality], local_indices)
        # A second selection uses the view's local item indices and lengths.
        first = partitions[name].select(value, [0])
        expected = source_partition.select(values[modality], local_indices[:1])
        torch.testing.assert_close(first, expected)


def test_unknown_item_costs_and_rows_remain_explicit() -> None:
    metadata = MultimodalItemMetadata([("audio", 0)])
    metadata.validate()
    params = MultimodalParams(multimodal_data={MULTIMODAL_ENCODER_ITEM_METADATA_KEY: metadata})
    assert params.item_metadata is metadata
    assert params.item_metadata.encoder_token_lengths is None
    with pytest.raises(ValueError, match="requires item output embedding lengths"):
        params.get_item_layout()


def test_item_metadata_roundtrips_with_shared_tensor_handles() -> None:
    import pickle

    values = torch.arange(10, dtype=torch.float32).reshape(2, 5)
    metadata = MultimodalItemMetadata(
        [("audio", 0), ("audio", 1)],
        output_embedding_lengths=[2, 3],
        input_partitions={"audio": {"features": MultimodalItemPartition(1, (2, 3))}},
    )
    params = MultimodalParams(
        multimodal_data={
            "audio": {"features": values},
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: metadata,
        }
    )
    params.to_handle("multimodal_data")
    assert params.item_metadata is metadata
    restored = pickle.loads(pickle.dumps(params))
    restored.to_tensor("multimodal_data")
    torch.testing.assert_close(restored.multimodal_data["audio"]["features"], values)
    assert restored.item_metadata == metadata
    assert restored.encoder_state is None


@pytest.mark.parametrize("precomputed", [False, True])
def test_generic_metadata_normalizes_mixed_order_without_encoder_costs(precomputed) -> None:
    from tensorrt_llm.inputs.registry import BaseMultimodalInputProcessor

    class Processor(_ItemMetadataFakeProcessor):
        get_mm_item_metadata = BaseMultimodalInputProcessor.get_mm_item_metadata

        def __call__(self, inputs, sampling_params):
            return [101, 102], {
                "multimodal_data": {
                    "image": {},
                    "audio": {},
                    "multimodal_embedding_lengths": [1, 1],
                    **({"multimodal_embedding": torch.zeros(2, 4)} if precomputed else {}),
                }
            }

        def get_mm_encoder_item_metadata(self, prompt_token_ids, multimodal_data):
            assert not precomputed
            return None

    order = [{"modality": "audio", "index": 0}, {"modality": "image", "index": 0}]
    _, extra = create_input_processor_with_hash(Processor(None))(
        {"prompt": "unused", "mm_item_order": order}, None
    )
    metadata = extra["multimodal_data"][MULTIMODAL_ENCODER_ITEM_METADATA_KEY]
    assert metadata.item_refs == (("audio", 0), ("image", 0))
    assert metadata.encoder_token_lengths is None
    assert metadata.output_embedding_lengths == (1, 1)


def test_maybe_compute_mm_embed_cumsum_populates_py_multimodal_data():
    """Producer writes a flat int64 cumsum tensor at py_multimodal_data[multimodal_embed_mask_cumsum]."""

    class FakeProcessor:
        def get_vocab_size(self):
            return 1000

        def get_mm_token_ids(self):
            return None

        def get_mm_special_token_ids(self):
            return torch.tensor([2000])

    # [text, img, img, special, img, img, img, text]
    prompt_token_ids = [10, 1001, 1002, 2000, 1003, 1004, 1005, 20]
    extra = {"multimodal_data": {}}
    maybe_compute_mm_embed_cumsum(prompt_token_ids, extra, FakeProcessor())

    cumsum = extra["multimodal_data"]["multimodal_embed_mask_cumsum"]
    torch.testing.assert_close(
        cumsum,
        torch.tensor([0, 1, 2, 2, 3, 4, 5, 5], dtype=torch.int64),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("cached_cumsum", [None, torch.tensor([0, 1], dtype=torch.int64)])
def test_existing_mm_cumsum_preserves_storage_and_backfills_block_alignment(
    cached_cumsum: torch.Tensor | None,
) -> None:
    processor = SimpleNamespace(mm_bidirectional_blocks=True)
    data = {"multimodal_embed_mask_cumsum": cached_cumsum}

    maybe_compute_mm_embed_cumsum([10, 101], {"multimodal_data": data}, processor)

    assert data["multimodal_embed_mask_cumsum"] is cached_cumsum
    assert data["mm_bidirectional_blocks"] is True


def test_tokenized_multimodal_overwrites_stale_embedding_lengths():
    """Final expanded prompt layout wins over dummy-placeholder metadata."""

    class FakeProcessor:
        multimodal_hashing_supported = True

        def get_text_with_mm_placeholders(self, mm_counts):
            assert mm_counts == {"image": 1}
            return "<image>"

        def __call__(self, inputs, sampling_params):
            # Tokenized fast path: multimodal_hashing_process forwards the raw
            # request dict (prompt_token_ids + multi_modal_data), with no
            # synthesized "prompt" text key. The real call_with_token_ids
            # returns the EXPANDED token ids (placeholder tokens replaced with
            # MM feature tokens) alongside stale dummy-placeholder metadata, and
            # downstream mask math runs on that post-expansion layout. Mirror it
            # by routing through this processor's own expansion hook.
            assert inputs["prompt_token_ids"] == [10, 98, 20]
            expanded_ids, _ = self.expand_prompt_token_ids_for_mm(inputs["prompt_token_ids"], [4])
            return expanded_ids, {
                "multimodal_data": {
                    # Stale dummy-placeholder value; recomputed to [3] downstream.
                    "multimodal_embedding_lengths": [999],
                }
            }

        def get_num_tokens_per_image(self, *, image):
            torch.testing.assert_close(image, torch.tensor([1]))
            return 4

        def expand_prompt_token_ids_for_mm(self, prompt_token_ids, num_mm_tokens, **kwargs):
            assert prompt_token_ids == [10, 98, 20]
            assert num_mm_tokens == [4]
            return [10, 101, 102, 200, 103, 20], None

        def get_vocab_size(self):
            return 100

        def get_mm_token_ids(self):
            return None

        def get_mm_special_token_ids(self):
            return torch.tensor([200])

    input_processor = create_input_processor_with_hash(FakeProcessor())

    prompt_token_ids, extra = input_processor(
        {
            "prompt_token_ids": [10, 98, 20],
            "multi_modal_data": {"image": [torch.tensor([1])]},
        },
        sampling_params=None,
    )

    assert prompt_token_ids == [10, 101, 102, 200, 103, 20]
    assert extra["multimodal_data"]["multimodal_embedding_lengths"] == [3]


class _KwargsHashFakeProcessor:
    multimodal_hashing_supported = True

    def __call__(self, inputs, sampling_params):
        return [10, 101, 102, 20], {"multimodal_data": {}}

    def get_num_tokens_per_image(self, *, image):
        return 2

    def get_vocab_size(self):
        return 100

    def get_mm_token_ids(self):
        return None

    def get_mm_special_token_ids(self):
        return None


_CONTENT_HASH_SNAPSHOT = {
    "image": [
        1704823964,
        362764787,
        1556099772,
        -86050102,
        551081767,
        1806967272,
        -1857978322,
        -1074313326,
    ],
    "video": [
        -839685430,
        -760767140,
        -2055508912,
        -1577466732,
        1582489614,
        1138227298,
        1889150221,
        1556646618,
    ],
    "audio": [
        -1693994058,
        897664400,
        754889986,
        -1726904048,
        -495512501,
        952543145,
        -87795905,
        -419978698,
    ],
}
_UUID_HASH_SNAPSHOT = {
    "image": [
        1777607066,
        -1240710263,
        -116192886,
        1440160746,
        120140059,
        -2062281779,
        960024653,
        2041693752,
    ],
    "video": [
        2070136680,
        -523271134,
        1570526135,
        1209057230,
        1458965967,
        334689758,
        1673654996,
        -1634814920,
    ],
    "audio": [
        -1172540155,
        -1149723152,
        1994549875,
        -2028680509,
        93572480,
        1551260061,
        -1949652498,
        -1123551599,
    ],
}


@pytest.mark.parametrize("hashing_enabled", [False, True])
def test_mm_item_metadata_rejects_conflicting_prompt_order(hashing_enabled: bool) -> None:
    class Processor(_KwargsHashFakeProcessor):
        multimodal_hashing_supported = hashing_enabled

        def __call__(self, inputs: dict, sampling_params: object) -> tuple[list[int], dict]:
            return [10, 101, 102, 20, 101, 102, 30], {
                "multimodal_data": {
                    "image": {},
                    "video": {},
                    "multimodal_embedding_lengths": [2, 2],
                }
            }

        def get_num_tokens_per_video(self, *, video: list[np.ndarray], **kwargs: object) -> int:
            return 2

        def get_mm_encoder_item_metadata(
            self, prompt_token_ids: list[int], multimodal_data: dict
        ) -> MultimodalEncoderItemMetadata:
            return MultimodalEncoderItemMetadata([("video", 0), ("image", 0)], [4, 4], [2, 2])

    processor = Processor()
    inputs = {
        "prompt": "unused",
        "multi_modal_data": {
            "image": [np.array([1], dtype=np.uint8)],
            "video": [VideoData(frames=[np.full((2, 2, 3), 2, dtype=np.uint8)], metadata={})],
        },
        "mm_item_order": [{"modality": "image", "index": 0}, {"modality": "video", "index": 0}],
    }

    with pytest.raises(ValueError, match="item references do not match the prompt item order"):
        create_input_processor_with_hash(processor)(inputs, None)

    assert processor.multimodal_hashing_supported is hashing_enabled


@pytest.mark.parametrize("hashing_enabled", [False, True], ids=["fallback", "hashing"])
@pytest.mark.parametrize("use_uuids", [False, True], ids=["content", "uuid"])
@pytest.mark.parametrize(
    "modalities,token_ids,explicit_ids,special_ids,expected_cumsum,expected_runs",
    [
        pytest.param(
            ["image"],
            [10, 101, 102, 20],
            None,
            None,
            [0, 1, 2, 2],
            ([0, 1], [1], [2]),
            id="out-of-vocab-image",
        ),
        pytest.param(
            ["image"],
            [10, 101, 102, 20],
            [101, 102],
            None,
            [0, 1, 2, 2],
            ([0, 1], [1], [2]),
            id="explicit-image-tokens",
        ),
        pytest.param(
            ["image"],
            [10, 101, 200, 102, 20],
            [101, 102],
            [200],
            [0, 1, 1, 2, 2],
            ([0, 1], [1], [3]),
            id="image-framing-token",
        ),
        pytest.param(
            ["video"],
            [10, 101, 20, 102, 30],
            None,
            None,
            [0, 1, 1, 2, 2],
            ([0, 2], [1, 3], [1, 1]),
            id="video-interleaved-text",
        ),
        pytest.param(
            ["audio"],
            [10, 101, 102, 20],
            [101, 102],
            None,
            [0, 1, 2, 2],
            ([0, 1], [1], [2]),
            id="audio",
        ),
        pytest.param(
            ["video", "image", "audio"],
            [10, 101, 102, 20, 101, 102, 30, 101, 102, 40],
            [101, 102],
            None,
            [0, 1, 2, 2, 3, 4, 4, 5, 6, 6],
            ([0, 1, 2, 3], [1, 4, 7], [2, 2, 2]),
            id="mixed-prompt-order",
        ),
    ],
)
def test_multimodal_input_contract_snapshot(
    modalities: list[str],
    token_ids: list[int],
    explicit_ids: list[int] | None,
    special_ids: list[int] | None,
    expected_cumsum: list[int],
    expected_runs: tuple[list[int], list[int], list[int]],
    hashing_enabled: bool,
    use_uuids: bool,
) -> None:
    """Pin framework metadata independently of model-specific media encoding."""
    item_refs = [(modality, 0) for modality in modalities]
    num_items = len(item_refs)

    class Processor(_KwargsHashFakeProcessor):
        multimodal_hashing_supported = hashing_enabled
        mm_bidirectional_blocks = True

        def __call__(self, inputs: dict, sampling_params: object) -> tuple[list[int], dict]:
            return list(token_ids), {
                "multimodal_data": {
                    **{modality: {} for modality in modalities},
                    "multimodal_embedding_lengths": [2] * num_items,
                }
            }

        def get_num_tokens_per_image(self, *, image: np.ndarray) -> int:
            return 2 + len(special_ids or [])

        def get_num_tokens_per_video(self, *, video: list[np.ndarray], **kwargs: object) -> int:
            return 2

        def get_num_tokens_per_audio(self, *, audio: AudioData) -> int:
            return 2

        def get_mm_token_ids(self) -> torch.Tensor | None:
            return torch.tensor(explicit_ids) if explicit_ids is not None else None

        def get_mm_special_token_ids(self) -> torch.Tensor | None:
            return torch.tensor(special_ids) if special_ids is not None else None

        def get_mm_encoder_item_metadata(
            self, prompt_token_ids: list[int], multimodal_data: dict
        ) -> MultimodalEncoderItemMetadata:
            return MultimodalEncoderItemMetadata(item_refs, [4] * num_items, [2] * num_items)

    media = {
        "image": [np.array([1], dtype=np.uint8)],
        "video": [VideoData(frames=[np.full((2, 2, 3), 2, dtype=np.uint8)], metadata={})],
        "audio": [AudioData(samples=np.array([0.25, -0.5], dtype=np.float32), sample_rate=16000)],
    }
    inputs = {
        "prompt": "unused",
        "multi_modal_data": {key: value for key, value in media.items() if key in modalities},
        "mm_item_order": [
            {"modality": modality, "index": 0, "placeholder": f"<{modality}>"}
            for modality in modalities
        ],
    }
    if use_uuids:
        inputs["multi_modal_uuids"] = {modality: [f"{modality}-id"] for modality in modalities}

    actual_token_ids, extra = create_input_processor_with_hash(Processor())(inputs, None)

    assert actual_token_ids == token_ids
    data = extra["multimodal_data"]
    cumsum = data["multimodal_embed_mask_cumsum"]
    assert cumsum.dtype == torch.int64
    assert cumsum.device.type == "cpu"
    assert cumsum.tolist() == expected_cumsum
    assert data["mm_bidirectional_blocks"] is True
    assert "multimodal_embedding_lengths" not in data
    assert data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] == MultimodalItemMetadata(
        item_refs, [4] * num_items, [2] * num_items
    )
    if not hashing_enabled:
        assert "multimodal_input" not in extra
        return

    mm_input = extra["multimodal_input"]
    hashes = _UUID_HASH_SNAPSHOT if use_uuids else _CONTENT_HASH_SNAPSHOT
    assert mm_input.multimodal_hashes == [hashes[modality] for modality in modalities]
    assert mm_input.multimodal_uuids == (
        [f"{modality}-id" for modality in modalities] if use_uuids else None
    )
    assert mm_input.multimodal_positions == [1 + 3 * index for index in range(num_items)]
    assert mm_input.multimodal_lengths == [2 + len(special_ids or [])] * num_items
    assert (
        mm_input.multimodal_item_run_cu_offsets,
        mm_input.multimodal_run_positions,
        mm_input.multimodal_run_lengths,
    ) == expected_runs
    if special_ids:
        assert data["special_token_offsets"] == [1]


def test_mm_processor_kwargs_hash_is_stable_for_canonical_values():
    input_processor = create_input_processor_with_hash(
        _KwargsHashFakeProcessor(),
        encoder_cache_enabled=True,
    )
    mm_data = {"image": [torch.tensor([1])]}

    _, first = input_processor(
        {
            "prompt": "unused",
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": {
                "b": [2, 3],
                "a": torch.tensor([1]),
            },
        },
        sampling_params=None,
    )
    _, second = input_processor(
        {
            "prompt": "unused",
            "multi_modal_data": mm_data,
            "mm_processor_kwargs": {
                "a": torch.tensor([1]),
                "b": [2, 3],
            },
        },
        sampling_params=None,
    )

    assert first["multimodal_data"]["mm_processor_kwargs_hash"] is not None
    assert (
        first["multimodal_data"]["mm_processor_kwargs_hash"]
        == second["multimodal_data"]["mm_processor_kwargs_hash"]
    )


def test_unserializable_mm_processor_kwargs_disable_persistent_cache_keying():
    input_processor = create_input_processor_with_hash(
        _KwargsHashFakeProcessor(),
        encoder_cache_enabled=True,
    )

    _, extra = input_processor(
        {
            "prompt": "unused",
            "multi_modal_data": {"image": [torch.tensor([1])]},
            "mm_processor_kwargs": {"custom": object()},
        },
        sampling_params=None,
    )

    assert extra["multimodal_data"]["mm_processor_kwargs_hash"] is None


def test_disabled_encoder_cache_skips_mm_processor_kwargs_hash():
    input_processor = create_input_processor_with_hash(_KwargsHashFakeProcessor())

    with patch("tensorrt_llm.inputs.registry._hash_mm_processor_kwargs") as kwargs_hash:
        _, extra = input_processor(
            {
                "prompt": "unused",
                "multi_modal_data": {"image": [torch.tensor([1])]},
                "mm_processor_kwargs": {"image_sizes": torch.tensor([1, 2])},
            },
            sampling_params=None,
        )

    kwargs_hash.assert_not_called()
    assert "mm_processor_kwargs_hash" not in extra["multimodal_data"]


def test_multimodal_embedding_lengths_exclude_special_tokens():
    """Embedding lengths omit multimodal wrapper tokens used only in prompts."""
    mm_mask = torch.tensor([False, True, True, True, True, True, False, True, True, True, True])
    embed_mask = torch.tensor(
        [
            False,
            False,
            True,
            True,
            False,
            True,
            False,
            True,
            False,
            True,
            True,
        ]
    )

    embedding_lengths = _find_mm_embedding_lengths_from_masks(mm_mask, embed_mask, [5, 4])

    assert embedding_lengths == [3, 3]


def test_find_mm_token_lengths_passes_all_grid_rows_for_single_video():
    """A single logical video can have multiple processor grid rows."""

    class FakeVideoProcessor:
        def get_num_tokens_per_video(self, *, video, video_grid_thw=None):
            assert video == ["frame"]
            assert isinstance(video_grid_thw, torch.Tensor)
            assert video_grid_thw.device.type == "cpu"
            torch.testing.assert_close(
                video_grid_thw,
                torch.tensor([[1, 8, 10], [1, 8, 10]]),
                rtol=0,
                atol=0,
            )
            return 40

    assert find_mm_token_lengths(
        {"video": [["frame"]]},
        FakeVideoProcessor(),
        multimodal_data={"video": {"video_grid_thw": [[1, 8, 10], [1, 8, 10]]}},
    ) == {"video": [40]}


def test_find_mm_token_lengths_preserves_2d_grid_rows_per_video():
    """One-row-per-video grids stay as [1, 3] CPU tensors."""

    class FakeVideoProcessor:
        def get_num_tokens_per_video(self, *, video, video_grid_thw=None):
            assert isinstance(video_grid_thw, torch.Tensor)
            assert video_grid_thw.device.type == "cpu"
            assert tuple(video_grid_thw.shape) == (1, 3)
            return int(video_grid_thw[0, 0].item())

    assert find_mm_token_lengths(
        {"video": [["frame0"], ["frame1"]]},
        FakeVideoProcessor(),
        multimodal_data={"video": {"video_grid_thw": torch.tensor([[1, 8, 10], [2, 8, 10]])}},
    ) == {"video": [1, 2]}


def test_find_mm_token_lengths_accepts_stacked_numpy_video_frames():
    """VideoMediaIO's optimized numpy format is normalized to frame lists."""

    class FakeVideoProcessor:
        def get_num_tokens_per_video(self, *, video, **kwargs):
            assert isinstance(video, list)
            assert len(video) == 2
            assert all(isinstance(frame, np.ndarray) for frame in video)
            assert all(frame.shape == (4, 6, 3) for frame in video)
            return len(video)

    video = VideoData(
        frames=np.zeros((2, 4, 6, 3), dtype=np.uint8),
        metadata={},
    )

    assert find_mm_token_lengths(
        {"video": [video]},
        FakeVideoProcessor(),
    ) == {"video": [2]}


def test_find_mm_token_lengths_rejects_undecoded_video_string():
    class FakeVideoProcessor:
        def get_num_tokens_per_video(self, *, video, **kwargs):
            raise AssertionError("Processor must not receive undecoded video")

    with pytest.raises(ValueError, match="decoded frames"):
        find_mm_token_lengths(
            {"video": ["video.mp4"]},
            FakeVideoProcessor(),
        )


def test_attached_embeddings_share_item_metadata_and_prompt_layout():
    """Sparse E/P inputs use the same params as local encoder results."""
    handles = [{"tensor_size": (3, 16)}]
    metadata = MultimodalItemMetadata([("video", 0)], output_embedding_lengths=[3])
    params = MultimodalParams.from_embeddings(
        handles,
        metadata,
        multimodal_lengths=[4],
        multimodal_positions=[1],
        multimodal_hashes=[[1, 2, 3, 4]],
        multimodal_item_run_cu_offsets=[0, 2],
        multimodal_run_positions=[1, 4],
        multimodal_run_lengths=[2, 2],
        special_token_offsets=[2],
    )

    assert params.item_metadata is metadata
    assert params.get_item_layout() == ((("video", 0),), (3,))
    assert params.multimodal_data["multimodal_embedding"] is handles
    assert "multimodal_embedding_lengths" not in params.multimodal_data
    assert params.multimodal_data["special_token_offsets"] == [2]
    assert params.multimodal_input.multimodal_positions == [1]
    assert params.multimodal_input.multimodal_lengths == [4]
    assert params.multimodal_input.multimodal_item_run_cu_offsets == [0, 2]
    assert params.multimodal_input.multimodal_run_positions == [1, 4]
    assert params.multimodal_input.multimodal_run_lengths == [2, 2]
    assert params.encoder_state is None
    assert params.encoder_event is None


@pytest.mark.parametrize("modality", ["image", "video", "audio"])
def test_attached_embeddings_accept_tensors_without_cache_hashes(modality):
    embedding = torch.ones(3, 16)
    params = MultimodalParams.from_embeddings(
        [embedding],
        MultimodalItemMetadata([(modality, 0)], output_embedding_lengths=[3]),
        multimodal_positions=[1],
        multimodal_lengths=[3],
    )

    assert params.multimodal_input is None
    assert params.item_metadata.item_refs == ((modality, 0),)
    assert params.multimodal_data["multimodal_embedding"][0] is embedding
    assert params.item_metadata.encoder_token_lengths is None


@pytest.mark.parametrize("with_hashes", [False, True])
@pytest.mark.parametrize(
    "updates,match",
    [
        ({"multimodal_positions": [-1]}, "non-negative"),
        ({"multimodal_positions": []}, "length|align"),
        ({"multimodal_lengths": [0]}, "positive"),
        ({"multimodal_run_lengths": [2]}, "sum"),
        ({"special_token_offsets": [-1]}, "non-negative"),
        ({"embeddings": []}, "align"),
    ],
)
def test_attached_embeddings_validate_prompt_and_item_alignment(with_hashes, updates, match):
    kwargs = dict(
        embeddings=[{"tensor_size": (3, 16)}],
        item_metadata=MultimodalItemMetadata([("audio", 0)], output_embedding_lengths=[3]),
        multimodal_positions=[1],
        multimodal_lengths=[3],
        multimodal_item_run_cu_offsets=[0, 1],
        multimodal_run_positions=[1],
        multimodal_run_lengths=[3],
    )
    if with_hashes:
        kwargs["multimodal_hashes"] = [[1] * 8]
    kwargs.update(updates)
    with pytest.raises(ValueError, match=match):
        MultimodalParams.from_embeddings(**kwargs)


def test_multimodal_input_rejects_invalid_prompt_spans():
    """Prompt span validation rejects negative starts and zero lengths."""
    with pytest.raises(ValueError, match="multimodal_positions must be non-negative"):
        MultimodalInput.from_components([[1, 2, 3, 4]], [-1], [1])

    with pytest.raises(ValueError, match="multimodal_lengths must be positive"):
        MultimodalInput.from_components([[1, 2, 3, 4]], [0], [0])


def test_runtime_data_cumsum_math_simplest():
    """All-True mask, full request, no cache."""
    is_embed = torch.ones(5, dtype=torch.bool)
    rt = MultimodalRuntimeData(
        past_seen_token_num=0,
        chunk_end_pos=5,
        embed_mask_cumsum=is_embed.cumsum(0, dtype=torch.int64),
    )
    assert rt.num_cached_mm_tokens == 0
    assert rt.num_mm_tokens_in_chunk == 5
    assert rt.total_embeds_in_request == 5


def test_runtime_data_cumsum_math_partial_chunk():
    """Chunk ends before end of mask."""
    is_embed = torch.tensor([True, True, False, True, True, False, True])
    rt = MultimodalRuntimeData(
        past_seen_token_num=0,
        chunk_end_pos=4,
        embed_mask_cumsum=is_embed.cumsum(0, dtype=torch.int64),
    )
    assert rt.num_cached_mm_tokens == 0
    assert rt.num_mm_tokens_in_chunk == 3
    assert rt.total_embeds_in_request == 5


def test_runtime_data_cumsum_math_partial_cache():
    """past_seen_token_num > 0: cached counts embeds before watermark."""
    is_embed = torch.tensor([True, True, False, True, True, False, True])
    rt = MultimodalRuntimeData(
        past_seen_token_num=3,
        chunk_end_pos=7,
        embed_mask_cumsum=is_embed.cumsum(0, dtype=torch.int64),
    )
    assert rt.num_cached_mm_tokens == 2
    assert rt.num_mm_tokens_in_chunk == 3
    assert rt.total_embeds_in_request == 5


def test_runtime_data_cumsum_math_with_specials_mistral_shape():
    """Chunk boundary inside a unit with inline special (Mistral-shape)."""
    # [text, img, img, special, img, img, img, text]
    is_embed = torch.tensor([False, True, True, False, True, True, True, False])
    cumsum = is_embed.cumsum(0, dtype=torch.int64)

    rt0 = MultimodalRuntimeData(past_seen_token_num=0, chunk_end_pos=5, embed_mask_cumsum=cumsum)
    assert rt0.num_cached_mm_tokens == 0
    assert rt0.num_mm_tokens_in_chunk == 3
    assert rt0.total_embeds_in_request == 5

    rt1 = MultimodalRuntimeData(past_seen_token_num=5, chunk_end_pos=8, embed_mask_cumsum=cumsum)
    assert rt1.num_cached_mm_tokens == 3
    assert rt1.num_mm_tokens_in_chunk == 2
    assert rt1.total_embeds_in_request == 5


def test_runtime_data_requires_cumsum():
    """embed_mask_cumsum is required."""
    with pytest.raises(TypeError):
        MultimodalRuntimeData(past_seen_token_num=0, chunk_end_pos=5)


def test_normalized_params_keep_one_order_and_output_length_record():
    refs = [("audio", 0), ("image", 0)]
    costs = [4, 8]
    rows = [2, 3]
    metadata = MultimodalItemMetadata(refs, costs, rows)
    data = {
        MULTIMODAL_ENCODER_ITEM_METADATA_KEY: metadata,
        "multimodal_embedding_lengths": list(rows),
    }
    params = MultimodalParams(
        multimodal_data=data,
        mm_item_order=[{"modality": m, "index": i} for m, i in refs],
    )
    refs.reverse()
    costs[0] = 100
    rows[0] = 100

    assert params.item_metadata is metadata
    assert params.mm_item_order is None
    assert "multimodal_embedding_lengths" not in data
    assert metadata.item_refs == (("audio", 0), ("image", 0))
    assert metadata.encoder_token_lengths == (4, 8)
    assert metadata.output_embedding_lengths == (2, 3)
    with pytest.raises(TypeError):
        metadata.output_embedding_lengths[0] = 100
    with pytest.raises(AttributeError):
        metadata.encoder_token_lengths = (100, 8)


def test_unknown_item_layout_preserves_legacy_order_at_input_boundary():
    order = [{"modality": "audio", "index": 0}]
    params = MultimodalParams(multimodal_data={"audio": {}}, mm_item_order=order)
    assert params.item_metadata is None
    assert params.mm_item_order is order
