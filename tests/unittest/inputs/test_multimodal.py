# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Tests for MultimodalRuntimeData cumsum math and the flat-mask producer."""

from types import SimpleNamespace
from typing import get_args
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

from tensorrt_llm.inputs import registry
from tensorrt_llm.inputs.media_io import MEDIA_IO_REGISTRY, MediaModality
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_CACHE_SALT_KEY,
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MULTIMODAL_NO_CONTENT_HASHES_KEY,
    MultimodalInput,
    MultimodalItemMetadata,
    MultimodalItemPartition,
    MultimodalParams,
    MultimodalRuntimeData,
    _find_mm_embedding_lengths_from_masks,
    find_mm_token_lengths,
    modalities_without_token_count,
    request_cache_salt,
    select_multimodal_fields,
)
from tensorrt_llm.inputs.multimodal_data import MEDIA_MODALITIES, AudioData, VideoData
from tensorrt_llm.inputs.registry import (
    BaseMultimodalInputProcessor,
    DefaultInputProcessor,
    create_input_processor_with_hash,
    maybe_compute_mm_embed_cumsum,
)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("axis", [0, 1, -1])
@pytest.mark.parametrize(
    "indices,is_view",
    [
        ([0, 1], True),
        ([2, 0], False),
        ([1, 1, 0], False),
        ([], True),
        ([-2, -1], True),
        ([2], True),
    ],
)
def test_batched_item_partition_selects_items_and_preserves_views(
    axis: int, indices: list[int], is_view: bool
) -> None:
    values = torch.arange(9).reshape(3, 3)
    selected = MultimodalItemPartition(item_axis=axis).select(values, indices)
    expected = values[indices] if axis == 0 else values[:, indices]

    torch.testing.assert_close(selected, expected)
    shares_storage = selected.untyped_storage().data_ptr() == values.untyped_storage().data_ptr()
    assert shares_storage is is_view


@pytest.mark.parametrize("targeted", [False, True])
def test_audio_geometry_stays_on_cpu_during_nested_device_staging(targeted: bool) -> None:
    audio = {
        "input_audio_features": torch.zeros(3, 4, 2),
        "audio_num_clips": torch.tensor([2, 1]),
        "audio_output_lengths": torch.tensor([4, 2, 3]),
    }
    params = MultimodalParams(multimodal_data={"video": {"audio": audio}})
    targets = [f"video.audio.{name}" for name in audio] if targeted else None

    params.to_device("multimodal_data", "meta", target_keywords=targets)

    staged = params.multimodal_data["video"]["audio"]
    assert staged["input_audio_features"].device.type == "meta"
    for name in ("audio_num_clips", "audio_output_lengths"):
        assert staged[name] is audio[name]
        assert staged[name].device.type == "cpu"


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
                MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                    [("image", i) for i in range(len(self.existing_embedding_lengths))],
                    output_embedding_lengths=self.existing_embedding_lengths,
                ),
            }
        }

    def get_mm_item_metadata(self, prompt_token_ids, multimodal_data, mm_item_order=None):
        assert prompt_token_ids == [101, 102]
        assert "image" in multimodal_data
        return MultimodalItemMetadata(
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


def test_removed_encoder_metadata_hook_fails_before_processing() -> None:
    class OldMetadataProcessor(_ItemMetadataFakeProcessor):
        def get_mm_encoder_item_metadata(
            self, prompt_token_ids: list[int], multimodal_data: dict
        ) -> None:
            raise AssertionError("The removed hook must not be called")

    with pytest.raises(TypeError, match=r"implement get_mm_item_metadata\(\)"):
        create_input_processor_with_hash(OldMetadataProcessor([2]))


def test_mm_item_metadata_none_falls_back_to_non_item_path():
    # Generic item order does not imply that physical encoder costs or
    # item scheduling are supported.
    class NoMetadataProcessor(_ItemMetadataFakeProcessor):
        def get_mm_item_metadata(self, prompt_token_ids, multimodal_data, mm_item_order=None):
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
        def get_mm_item_metadata(self, prompt_token_ids, multimodal_data, mm_item_order=None):
            return ("image", 0)

    input_processor = create_input_processor_with_hash(BadMetadataProcessor([2]))

    with pytest.raises(TypeError, match="must return a"):
        input_processor({"prompt": "unused"}, sampling_params=None)


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
    metadata = MultimodalItemMetadata(
        [("image", 1), ("video", 0), ("image", 0), ("audio", 0)],
        [20, 50, 30, 40],
        [2, 5, 3, 4],
    )

    subset = metadata.select(selected)

    assert subset == MultimodalItemMetadata(expected_refs, expected_costs, expected_lengths)
    assert metadata.item_refs == (("image", 1), ("video", 0), ("image", 0), ("audio", 0))
    assert metadata.encoder_token_lengths == (20, 50, 30, 40)
    assert metadata.output_embedding_lengths == (2, 5, 3, 4)
    assert subset is not metadata


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


@pytest.mark.parametrize("modality", MEDIA_MODALITIES)
def test_prepared_metadata_fills_unknown_model_rows(modality) -> None:
    partitions = {modality: {"features": MultimodalItemPartition()}}
    original = MultimodalItemMetadata([(modality, 0)], input_partitions=partitions)

    class Processor(_ItemMetadataFakeProcessor):
        def __call__(self, inputs, sampling_params):
            return [101, 102], {
                "multimodal_data": {
                    modality: {},
                    MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                        [(modality, 0)], output_embedding_lengths=[2]
                    ),
                }
            }

        def get_mm_item_metadata(self, prompt_token_ids, multimodal_data, mm_item_order=None):
            return original

    _, extra = create_input_processor_with_hash(Processor(None))({"prompt": "unused"}, None)
    params = MultimodalParams(multimodal_data=extra["multimodal_data"])
    assert params.get_item_layout() == (((modality, 0),), (2,))
    assert params.item_metadata.encoder_token_lengths is None
    assert params.item_metadata.input_partitions is partitions
    assert original.output_embedding_lengths is None
    assert "multimodal_embedding_lengths" not in params.multimodal_data
    assert (
        MultimodalParams(multimodal_data=params.multimodal_data).item_metadata
        is params.item_metadata
    )


@pytest.mark.parametrize("lengths", [[2, 3], [-1], ["2"]])
def test_prepared_metadata_validates_model_rows(lengths) -> None:
    class Processor(_ItemMetadataFakeProcessor):
        def get_mm_item_metadata(self, prompt_token_ids, multimodal_data, mm_item_order=None):
            return MultimodalItemMetadata([("image", 0)], output_embedding_lengths=lengths)

    with pytest.raises((TypeError, ValueError)):
        create_input_processor_with_hash(Processor([2]))({"prompt": "unused"}, None)


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
def test_generic_metadata_keeps_mixed_order_without_encoder_costs(precomputed) -> None:
    from tensorrt_llm.inputs.registry import BaseMultimodalInputProcessor

    class Processor(_ItemMetadataFakeProcessor):
        get_mm_item_metadata = BaseMultimodalInputProcessor.get_mm_item_metadata

        def __call__(self, inputs, sampling_params):
            refs = [(entry["modality"], entry["index"]) for entry in inputs["mm_item_order"]]
            return [101, 102], {
                "multimodal_data": {
                    "image": {},
                    "audio": {},
                    MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                        refs, output_embedding_lengths=[1, 1]
                    ),
                    **({"multimodal_embedding": torch.zeros(2, 4)} if precomputed else {}),
                }
            }

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


def test_tokenized_multimodal_builds_metadata_from_final_embedding_slots():
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
                    MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                        [("image", 0)], output_embedding_lengths=None
                    ),
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
    assert extra["multimodal_data"][
        MULTIMODAL_ENCODER_ITEM_METADATA_KEY
    ].output_embedding_lengths == (3,)


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


def test_mixed_hashing_uses_processor_metadata_without_frontend_order() -> None:
    class Processor(_KwargsHashFakeProcessor):
        multimodal_hashing_supported = None

        def __call__(self, inputs: dict, sampling_params: object) -> tuple[list[int], dict]:
            return [10, 101, 102, 20, 101, 102, 30], {"multimodal_data": {"image": {}, "video": {}}}

        def get_num_tokens_per_video(self, *, video: VideoData, **kwargs: object) -> int:
            return 2

        def get_mm_item_metadata(
            self,
            prompt_token_ids: list[int],
            multimodal_data: dict,
            mm_item_order: list[dict[str, object]] | None = None,
        ) -> MultimodalItemMetadata:
            return MultimodalItemMetadata([("video", 0), ("image", 0)], [16, 8], [2, 2])

    processor = Processor()
    wrapped = create_input_processor_with_hash(processor)
    inputs = {
        "prompt": "unused",
        "multi_modal_data": {
            "image": [np.array([1], dtype=np.uint8)],
            "video": [VideoData(frames=[np.full((2, 2, 3), 2, dtype=np.uint8)], metadata={})],
        },
    }

    # Exercise both the first hashing probe and a subsequent enabled request.
    for _ in range(2):
        _, extra = wrapped(inputs, None)
        assert processor.multimodal_hashing_supported is True
        mm_input = extra["multimodal_input"]
        assert mm_input.multimodal_hashes == [
            _CONTENT_HASH_SNAPSHOT["video"],
            _CONTENT_HASH_SNAPSHOT["image"],
        ]
        assert mm_input.multimodal_positions == [1, 4]
        metadata = extra["multimodal_data"][MULTIMODAL_ENCODER_ITEM_METADATA_KEY]
        assert metadata.item_refs == (("video", 0), ("image", 0))
        assert metadata.encoder_token_lengths == (16, 8)
        assert metadata.output_embedding_lengths == (2, 2)


@pytest.mark.parametrize("hashing_enabled", [False, True])
def test_mm_item_metadata_rejects_conflicting_prompt_order(hashing_enabled: bool) -> None:
    class Processor(_KwargsHashFakeProcessor):
        multimodal_hashing_supported = hashing_enabled

        def __call__(self, inputs: dict, sampling_params: object) -> tuple[list[int], dict]:
            return [10, 101, 102, 20, 101, 102, 30], {
                "multimodal_data": {
                    "image": {},
                    "video": {},
                    MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                        [("video", 0), ("image", 0)], output_embedding_lengths=[2, 2]
                    ),
                }
            }

        def get_num_tokens_per_video(self, *, video: list[np.ndarray], **kwargs: object) -> int:
            return 2

        def get_mm_item_metadata(
            self, prompt_token_ids: list[int], multimodal_data: dict, mm_item_order=None
        ) -> MultimodalItemMetadata:
            return MultimodalItemMetadata([("video", 0), ("image", 0)], [4, 4], [2, 2])

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


class _ProbeFakeProcessor(_KwargsHashFakeProcessor):
    multimodal_hashing_supported: bool | None = None

    def __init__(
        self,
        call_error: Exception | None = None,
        count_error: Exception | None = None,
        vocab_size: int | None = 100,
    ) -> None:
        self.call_error = call_error
        self.count_error = count_error
        self.vocab_size = vocab_size
        self.calls = 0

    def __call__(self, inputs: dict, sampling_params: object) -> tuple[list[int], dict]:
        self.calls += 1
        if self.call_error is not None:
            raise self.call_error
        return super().__call__(inputs, sampling_params)

    def get_num_tokens_per_image(self, *, image: np.ndarray) -> int:
        if self.count_error is not None:
            raise self.count_error
        return super().get_num_tokens_per_image(image=image)

    def get_vocab_size(self) -> int | None:
        return self.vocab_size


_PROBE_INPUTS = {"prompt": "unused", "multi_modal_data": {"image": [np.array([1], dtype=np.uint8)]}}
_AUDIO_INPUTS = {"prompt": "unused", "multi_modal_data": {"audio": [np.zeros(2, dtype=np.float32)]}}


def _assert_isolated(extra: dict) -> str:
    """The request has no item spans and a cache salt of its own; returns that salt."""
    data = extra["multimodal_data"]
    assert "multimodal_input" not in extra
    assert data[MULTIMODAL_NO_CONTENT_HASHES_KEY] is True
    salt = request_cache_salt(None, MultimodalParams(multimodal_data=data))
    assert salt is not None
    assert request_cache_salt("user", MultimodalParams(multimodal_data=data)) == f"user/{salt}"
    return salt


@pytest.mark.parametrize(
    "call_error",
    [ValueError("bad image"), NotImplementedError("bad image")],
    ids=["processor_error", "processor_unsupported_input"],
)
def test_processor_error_runs_the_processor_once_and_keeps_hashing_on(call_error) -> None:
    """A request the processor rejects fails after one run and does not turn hashing off."""
    processor = _ProbeFakeProcessor(call_error)
    wrapped = create_input_processor_with_hash(processor)

    with pytest.raises(type(call_error), match="bad image"):
        wrapped(_PROBE_INPUTS, None)
    assert processor.calls == 1
    assert processor.multimodal_hashing_supported is None

    processor.call_error = None
    _, extra = wrapped(_PROBE_INPUTS, None)
    assert processor.calls == 2
    assert processor.multimodal_hashing_supported is True
    assert extra["multimodal_input"].multimodal_hashes == [_CONTENT_HASH_SNAPSHOT["image"]]


@pytest.mark.parametrize("hashing_supported", [None, True], ids=["probe", "hashing_on"])
def test_token_count_error_isolates_the_request(hashing_supported) -> None:
    """The token count fails for one request (other processor kwargs, say): no item spans.

    The request is served after one processor run, with a cache salt of its own, so it shares
    no KV cache blocks keyed by placeholder tokens. Hashing stays as it was.
    """
    processor = _ProbeFakeProcessor(count_error=ValueError("other max_pixels"))
    processor.multimodal_hashing_supported = hashing_supported
    wrapped = create_input_processor_with_hash(processor)

    salts = {_assert_isolated(wrapped(_PROBE_INPUTS, None)[1]) for _ in range(2)}
    assert len(salts) == 2
    assert processor.calls == 2
    assert processor.multimodal_hashing_supported is hashing_supported

    processor.count_error = None
    _, extra = wrapped(_PROBE_INPUTS, None)
    assert processor.multimodal_hashing_supported is True
    assert extra["multimodal_input"].multimodal_hashes == [_CONTENT_HASH_SNAPSHOT["image"]]
    assert MULTIMODAL_CACHE_SALT_KEY not in extra["multimodal_data"]


@pytest.mark.parametrize(
    "count_error",
    [NotImplementedError("bug"), AttributeError("bug"), TypeError("bug")],
    ids=["not_implemented", "attribute_error", "type_error"],
)
def test_other_token_count_errors_fail_the_request(count_error) -> None:
    """Only a ValueError from the token count means "no spans"; anything else is a bug to see."""
    processor = _ProbeFakeProcessor(count_error=count_error)
    wrapped = create_input_processor_with_hash(processor)

    with pytest.raises(type(count_error), match="bug"):
        wrapped(_PROBE_INPUTS, None)
    assert processor.calls == 1
    assert processor.multimodal_hashing_supported is None


@pytest.mark.parametrize("no_items", [[], None], ids=["empty_list", "none"])
def test_modality_without_items_keeps_hashing_on(no_items) -> None:
    """An empty modality says nothing about whether the processor can hash.

    An image request with ``"audio": []`` hashes its image and turns hashing on.
    """
    processor = _ProbeFakeProcessor()  # Counts image tokens only.
    wrapped = create_input_processor_with_hash(processor)
    image = _PROBE_INPUTS["multi_modal_data"]["image"]

    _, extra = wrapped(
        {"prompt": "unused", "multi_modal_data": {"image": image, "audio": no_items}}, None
    )
    assert processor.multimodal_hashing_supported is True
    assert extra["multimodal_input"].multimodal_hashes == [_CONTENT_HASH_SNAPSHOT["image"]]


@pytest.mark.parametrize(
    "hashing_supported", [None, True, False], ids=["unknown", "hashing_on", "hashing_off"]
)
@pytest.mark.parametrize("inputs", [_PROBE_INPUTS, _AUDIO_INPUTS], ids=["image", "audio"])
@pytest.mark.parametrize(
    "uuids",
    [{"image": [123]}, {"image": ["a", "b"]}, ["image-uuid"], {"images": ["a"]}],
    ids=["int_uuid", "length_mismatch", "not_a_dict", "unknown_modality"],
)
def test_bad_multimodal_uuids_are_a_request_error(uuids, inputs, hashing_supported) -> None:
    """Bad UUIDs fail the request before any processing and do not change the hashing flag.

    Also for audio, whose tokens the processor cannot count, so it is never hashed.
    """
    modality = next(iter(inputs["multi_modal_data"]))
    if isinstance(uuids, dict) and "image" in uuids:
        uuids = {modality: uuids["image"]}
    processor = _ProbeFakeProcessor()
    processor.multimodal_hashing_supported = hashing_supported
    wrapped = create_input_processor_with_hash(processor)

    with pytest.raises(ValueError, match="(?i)uuid"):
        wrapped({**inputs, "multi_modal_uuids": uuids}, None)
    assert processor.calls == 0
    assert processor.multimodal_hashing_supported is hashing_supported


class _NoHfTokenCountProcessor(_ProbeFakeProcessor):
    """Counts image tokens with the base processor, whose HF processor cannot."""

    processor = object()
    get_num_multimodal_tokens = BaseMultimodalInputProcessor.get_num_multimodal_tokens
    get_num_tokens_per_image = BaseMultimodalInputProcessor.get_num_tokens_per_image


@pytest.mark.parametrize(
    "processor_cls,vocab_size",
    [(_NoHfTokenCountProcessor, 100), (_ProbeFakeProcessor, None)],
    ids=["no_hf_token_count", "no_vocab_or_mm_ids"],
)
def test_unsupported_hashing_turns_hashing_off(processor_cls, vocab_size) -> None:
    """A processor that cannot find item spans turns hashing off; its requests are isolated."""
    processor = processor_cls(vocab_size=vocab_size)
    wrapped = create_input_processor_with_hash(processor)
    inputs = {
        "prompt": "unused",
        "multi_modal_data": {"image": [np.zeros((2, 2, 3), dtype=np.uint8)]},
    }

    salts = set()
    for _ in range(2):
        salts.add(_assert_isolated(wrapped(inputs, None)[1]))
        assert processor.multimodal_hashing_supported is False
    assert len(salts) == 2
    assert processor.calls == 2  # One processor run per request.


def test_modality_without_token_count_hook_does_not_change_other_modalities() -> None:
    """A bad image, a good image, then audio, which the processor cannot count (as Phi-4-MM).

    The audio request is served after one processor run, with a cache salt of its own, and
    image requests keep hashing.
    """
    processor = _ProbeFakeProcessor(count_error=ValueError("bad image"))  # Counts images only.
    wrapped = create_input_processor_with_hash(processor)

    _assert_isolated(wrapped(_PROBE_INPUTS, None)[1])
    processor.count_error = None
    _, extra = wrapped(_PROBE_INPUTS, None)
    assert extra["multimodal_input"].multimodal_hashes == [_CONTENT_HASH_SNAPSHOT["image"]]

    calls = processor.calls
    _assert_isolated(wrapped(_AUDIO_INPUTS, None)[1])
    assert processor.calls == calls + 1
    assert processor.multimodal_hashing_supported is True

    _, extra = wrapped(_PROBE_INPUTS, None)
    assert extra["multimodal_input"].multimodal_hashes == [_CONTENT_HASH_SNAPSHOT["image"]]


def test_phi4mm_has_no_audio_token_count() -> None:
    """Phi-4-MM audio takes the no-span path above: it counts image tokens but not audio."""
    from tensorrt_llm._torch.models.modeling_phi4mm import Phi4MMInputProcessor

    image_and_audio = {"image": [np.zeros((2, 2, 3))], "audio": [np.zeros(2)]}
    assert modalities_without_token_count(image_and_audio, Phi4MMInputProcessor) == ["audio"]


class _PilOnlyKimiImageProcessor:
    """Counts tokens as the Kimi K3 checkpoint does: PIL images only, else ValueError."""

    media_proc_cfg = {"temporal_merge_kernel_size": 2}

    def media_tokens_calculator(self, media: dict) -> int:
        from PIL import Image

        items = [media["image"]] if media["type"] == "image" else media["video_chunk"]
        if not all(isinstance(item, Image.Image) for item in items):
            raise ValueError(f"Unsupported data type: {type(items[0])}")
        width, height = items[0].size
        return (width // 28) * (height // 28)


def test_kimi_counts_the_tokens_of_tensor_images_and_frames() -> None:
    """trtllm-serve passes images and video frames as tensors (format="pt").

    A count that rejects them leaves every such request without item spans, so it would reuse
    no KV cache at all.
    """
    from tensorrt_llm._torch.models.modeling_kimi_k25 import KimiK25InputProcessor

    processor = object.__new__(KimiK25InputProcessor)
    processor._processor = SimpleNamespace(image_processor=_PilOnlyKimiImageProcessor())
    frame = torch.zeros(3, 56, 84, dtype=torch.uint8)

    assert processor.get_num_tokens_per_image(image=frame) == 6
    assert processor.get_num_tokens_per_video(video=[frame] * 3) == 12


@pytest.mark.parametrize("hashing_off", [True, False], ids=["hashing_off", "hashing_failed"])
def test_request_without_content_hashes_gets_random_hashes(monkeypatch, hashing_off) -> None:
    """Hashing off, or hashing failed for this request: the items get random hashes.

    Two requests with the same image and prompt get the same spans and other hashes, which are
    not the image's content hash, so their KV cache blocks get other keys. A supplied UUID is kept.
    """
    processor = _ProbeFakeProcessor()
    if hashing_off:
        processor.multimodal_hashing_supported = False
    else:

        def cannot_hash(*args: object, **kwargs: object) -> None:
            raise ValueError("cannot hash this image")

        monkeypatch.setattr(registry, "apply_mm_hashes", cannot_hash)
    wrapped = create_input_processor_with_hash(processor)

    inputs = [dict(_PROBE_INPUTS), dict(_PROBE_INPUTS)]
    inputs[0]["multi_modal_uuids"] = {"image": ["image-0"]}
    extras = [wrapped(request, None)[1] for request in inputs]
    mm_inputs = [extra["multimodal_input"] for extra in extras]

    assert processor.calls == 2
    assert processor.multimodal_hashing_supported is (False if hashing_off else None)
    for extra, mm_input in zip(extras, mm_inputs):
        assert mm_input.multimodal_positions == [1]
        assert mm_input.multimodal_lengths == [2]
        assert mm_input.multimodal_hashes != [_CONTENT_HASH_SNAPSHOT["image"]]
        assert extra["multimodal_data"][MULTIMODAL_NO_CONTENT_HASHES_KEY] is True
        assert MULTIMODAL_CACHE_SALT_KEY not in extra["multimodal_data"]
    assert mm_inputs[0].multimodal_hashes != mm_inputs[1].multimodal_hashes
    assert [mm_input.multimodal_uuids for mm_input in mm_inputs] == [["image-0"], None]


def test_random_hashes_keep_uuids_of_one_modality_in_prompt_order() -> None:
    """Hashing off, an image and a video, a UUID for the video only."""

    class Processor(_ProbeFakeProcessor):
        def __call__(self, inputs: dict, sampling_params: object) -> tuple[list[int], dict]:
            self.calls += 1
            return [10, 101, 102, 20, 101, 102, 30], {"multimodal_data": {"image": {}, "video": {}}}

        def get_num_tokens_per_video(self, *, video: list[np.ndarray], **kwargs: object) -> int:
            return 2

    processor = Processor()
    processor.multimodal_hashing_supported = False
    _, extra = create_input_processor_with_hash(processor)(
        {
            "prompt": "unused",
            "multi_modal_data": {
                "image": [np.array([1], dtype=np.uint8)],
                "video": [[np.zeros((2, 2, 3), dtype=np.uint8)]],
            },
            "multi_modal_uuids": {"video": ["video-0"]},
            "mm_item_order": [{"modality": "video", "index": 0}, {"modality": "image", "index": 0}],
        },
        None,
    )

    mm_input = extra["multimodal_input"]
    assert mm_input.multimodal_positions == [1, 4]
    assert mm_input.multimodal_uuids == ["video-0", None]
    assert len({tuple(h) for h in mm_input.multimodal_hashes}) == 2


@pytest.mark.parametrize("hashing_off", [False, True], ids=["hashing_on", "hashing_off"])
def test_modality_set_to_none_is_like_no_modality(hashing_off) -> None:
    """``"audio": None`` next to an image changes nothing: the image gets the same span and hash.

    The processor counts image tokens only, so counting the None as an audio item would fail.
    """
    processor = _ProbeFakeProcessor()
    processor.multimodal_hashing_supported = False if hashing_off else None
    wrapped = create_input_processor_with_hash(processor)
    image = _PROBE_INPUTS["multi_modal_data"]["image"]

    mm_inputs = [
        wrapped({"prompt": "unused", "multi_modal_data": data}, None)[1]["multimodal_input"]
        for data in ({"image": image}, {"image": image, "audio": None})
    ]

    for mm_input in mm_inputs:
        assert mm_input.multimodal_positions == [1]
        assert mm_input.multimodal_lengths == [2]
        assert len(mm_input.multimodal_hashes) == 1
    if not hashing_off:
        assert mm_inputs[1].multimodal_hashes == [_CONTENT_HASH_SNAPSHOT["image"]]
        assert processor.multimodal_hashing_supported is True


def _llm_with_mock_executor(input_processor: object = None) -> object:
    """A BaseLLM whose generate_async submits to a mock executor."""
    from tensorrt_llm.llmapi.llm import BaseLLM
    from tensorrt_llm.sampling_params import SamplingParams

    llm = object.__new__(BaseLLM)
    llm.shutdown = Mock()
    llm._encode_only = False
    llm._startup_metrics = {}
    llm.args = SimpleNamespace(mm_encoder_only=False, return_perf_metrics=False)
    llm._executor = Mock()
    llm._executor.is_shutdown.return_value = False
    llm._prepare_sampling_params = Mock(return_value=SamplingParams(max_tokens=1))
    llm._check_arguments = Mock()
    llm._post_processor_hook = None
    llm._tokenizer = None
    llm._hf_model_config = None
    llm.input_processor = input_processor
    return llm


def test_generate_async_submits_the_multimodal_cache_salt() -> None:
    """LLM.generate_async adds the request's multimodal salt to the caller's salt."""
    llm = _llm_with_mock_executor()
    wrapped = create_input_processor_with_hash(_ProbeFakeProcessor(count_error=ValueError("x")))
    submitted = []
    for inputs in (_PROBE_INPUTS, {"prompt": "text only"}):
        tokens, extra = wrapped(inputs, None) if "multi_modal_data" in inputs else ([1], None)
        params = None if extra is None else MultimodalParams(**extra)
        llm._preprocess = Mock(return_value=(tokens, None, params, None))
        with patch("tensorrt_llm.llmapi.llm.RequestOutput._from_generation_result"):
            llm.generate_async(inputs, cache_salt="user")
        submitted.append(llm._executor.generate_async.call_args.kwargs["cache_salt"])

    assert submitted[0].startswith("user/mm-no-item-spans-")
    assert submitted[1] == "user"


@pytest.mark.parametrize("route", ["raw", "preprocessed", "processor_tensors"])
def test_phi4mm_single_audio_encoder_inputs(route: str) -> None:
    from tensorrt_llm._torch.models.modeling_phi4mm import (
        HFPhi4MultimodalEncoder,
        Phi4MMInputProcessor,
    )
    from tensorrt_llm.llmapi.llm import PreprocessedInputs

    config = SimpleNamespace(model_type="phi4mm", vocab_size=200064, torch_dtype=torch.float32)
    hf_processor = Mock(
        spec=["image_processor", "audio_processor", "_convert_images_audios_text_to_inputs"]
    )
    with patch("transformers.AutoProcessor.from_pretrained", return_value=hf_processor):
        processor = Phi4MMInputProcessor("unused", config, Mock())
    features = torch.tensor([[[2.0, 3.0]]])
    audio = {"input_audio_embeds": features, "audio_embed_sizes": torch.tensor([1])}
    hf_processor.audio_processor.return_value = audio
    hf_processor._convert_images_audios_text_to_inputs.return_value = {
        "input_ids": processor.get_mm_token_ids()[1:].reshape(1, 1).long(),
        "input_image_embeds": torch.empty(0),
        "image_sizes": torch.empty(0),
        "image_attention_mask": torch.empty(0),
        "audio_attention_mask": None,
        **audio,
    }
    llm = _llm_with_mock_executor(processor)
    llm.args.multimodal_config = None
    inputs = {"prompt": "<|audio_1|>", "multi_modal_data": {"audio": [(np.zeros(320), 16000)]}}
    if route == "preprocessed":
        inputs = llm.preprocess(inputs)
    elif route == "processor_tensors":
        ids, extra = processor(inputs, None)
        inputs = PreprocessedInputs(ids, MultimodalParams(**extra))
    with patch("tensorrt_llm.llmapi.llm.RequestOutput._from_generation_result"):
        llm.generate_async(inputs)
    params = llm._executor.generate_async.call_args.kwargs["multimodal_params"]
    params.to_tensor("multimodal_data")

    encoder = HFPhi4MultimodalEncoder.__new__(HFPhi4MultimodalEncoder)
    torch.nn.Module.__init__(encoder)
    encoder.embed_tokens = torch.nn.Identity()
    encoder.audio_embed = Mock(side_effect=lambda input_embeds, **kwargs: input_embeds)
    (embedding,) = encoder._encoding_batch_request([params], processor.get_mm_token_ids())
    torch.testing.assert_close(embedding, features[0])


class _AttachWithoutSpansProcessor(_ProbeFakeProcessor):
    """Attaches embeddings as LLaVA-NeXT and Llama 4 do: no hashes and no item spans."""

    def attach_multimodal_embeddings(
        self, inputs: dict, multimodal_embedding: dict, sampling_params: object
    ) -> tuple[list[int], dict]:
        return [10, 101, 102, 20], {
            "multimodal_data": {"multimodal_embedding": multimodal_embedding["image"][0]}
        }


def test_attached_embeddings_without_item_spans_get_a_cache_salt() -> None:
    """Two requests with the same prompt and other embeddings of the same size.

    Their tokens are the same, so without item spans only a cache salt of their own keeps the
    second from reusing the KV cache of the first.
    """
    llm = _llm_with_mock_executor(_AttachWithoutSpansProcessor())
    submitted = []
    for embedding in (torch.zeros(2, 4), torch.ones(2, 4)):
        inputs = {"prompt": "unused", "multi_modal_embeddings": {"image": [embedding]}}
        with (
            patch.object(MultimodalParams, "to_handle"),
            patch("tensorrt_llm.llmapi.llm.RequestOutput._from_generation_result"),
        ):
            llm.generate_async(inputs, cache_salt="user")
        kwargs = llm._executor.generate_async.call_args.kwargs
        submitted.append(kwargs["cache_salt"])
        assert kwargs["multimodal_params"].multimodal_data[MULTIMODAL_NO_CONTENT_HASHES_KEY]

    assert all(salt.startswith("user/mm-no-item-spans-") for salt in submitted)
    assert submitted[0] != submitted[1]


@pytest.mark.parametrize(
    "method,inputs",
    [
        ("generate", {"prompt": "p", "multi_modal_data": {"image": [torch.zeros(3, 4, 4)]}}),
        (
            "preprocess",
            {"prompt_token_ids": [1, 2], "multi_modal_data": {"image": [torch.zeros(3, 4, 4)]}},
        ),
        (
            "generate_async",
            {"prompt": "p", "multi_modal_embeddings": {"image": [torch.zeros(2, 4)]}},
        ),
        (
            "encode",
            {"prompt_token_ids": [1, 2], "multi_modal_data": {"image": [torch.zeros(3, 4, 4)]}},
        ),
    ],
    ids=["generate", "preprocess", "generate_async", "encode"],
)
def test_text_only_model_rejects_multimodal_inputs(method: str, inputs: dict) -> None:
    llm = _llm_with_mock_executor(DefaultInputProcessor("unused", None, Mock()))
    llm.args.multimodal_config = None
    if method == "encode":
        llm._encode_only = True
        llm._encoder_executor = Mock(max_batch_size=1, max_num_tokens=2, max_seq_len=2)
        llm._encoder_executor.batch_forward.return_value = {"logits": torch.zeros(1)}
    with pytest.raises(ValueError, match="supports text only"):
        getattr(llm, method)(inputs)
    llm._executor.generate_async.assert_not_called()


@pytest.mark.parametrize(
    "inputs",
    [
        {"prompt_token_ids": [1, 2], "multi_modal_data": {"image": [], "audio": None}},
        {"prompt": "p", "multi_modal_embeddings": {}},
        {"prompt": "p", "multi_modal_data": {"image": {}}, "multi_modal_uuids": {"image": []}},
    ],
    ids=["token_ids", "text", "empty_image_dict"],
)
def test_text_only_model_accepts_empty_multimodal_inputs(inputs: dict) -> None:
    tokenizer = Mock()
    tokenizer.encode.return_value = [1, 2]
    llm = _llm_with_mock_executor(DefaultInputProcessor("unused", None, tokenizer))
    llm.args.multimodal_config = None
    result = llm.preprocess(inputs)
    assert result.prompt_token_ids == [1, 2]
    assert result.multimodal_params is None
    assert "multi_modal_data" in inputs or "multi_modal_embeddings" in inputs


@pytest.mark.parametrize(
    "media, error",
    [
        ({}, "which multi_modal_data does not have"),
        ({"multi_modal_data": {"image": []}}, "UUID list length"),
        ({"multi_modal_data": {"image": {}}}, r"data items length \(0\)"),
        ({"multi_modal_data": {"audio": {}}}, "which multi_modal_data does not have"),
    ],
    ids=["no_media_field", "empty_image_list", "empty_image_dict", "unknown_modality"],
)
def test_text_only_model_validates_uuids_for_empty_media(media, error) -> None:
    llm = _llm_with_mock_executor(DefaultInputProcessor("unused", None, Mock()))
    llm.args.multimodal_config = None
    with pytest.raises(ValueError, match=error):
        llm.preprocess(
            {
                "prompt_token_ids": [1, 2],
                **media,
                "multi_modal_uuids": {"image": ["unused"]},
            }
        )


def test_text_only_admission_of_preprocessed_media_and_decode_metadata() -> None:
    from tensorrt_llm.llmapi.llm import PreprocessedInputs

    llm = _llm_with_mock_executor(DefaultInputProcessor("unused", None, Mock()))
    for embedding in (torch.zeros(2, 4), {"method_key": -1, "tensor_size": [0, 4]}):
        params = MultimodalParams(multimodal_data={"multimodal_embedding": embedding})
        with pytest.raises(ValueError, match="supports text only"):
            llm.generate_async(PreprocessedInputs([1, 2], params))
    llm._executor.generate_async.assert_not_called()

    extra = {"multimodal_data": {"mrope_config": {"mrope_position_deltas": [0]}}}
    maybe_compute_mm_embed_cumsum([1, 2], extra, _ProbeFakeProcessor())
    extra["multimodal_data"]["mm_processor_kwargs_hash"] = registry._hash_mm_processor_kwargs({})
    params = MultimodalParams(**extra)
    with patch("tensorrt_llm.llmapi.llm.RequestOutput._from_generation_result"):
        llm.generate_async(PreprocessedInputs([1, 2], params))
    submitted = llm._executor.generate_async.call_args.kwargs
    assert submitted["multimodal_params"] is params
    assert submitted["cache_salt"] is None

    metadata = dict(params.multimodal_data)
    params.multimodal_data["image"] = {}
    params.multimodal_input = MultimodalInput([[1] * 8], [0], [1])
    with patch("tensorrt_llm.llmapi.llm.RequestOutput._from_generation_result"):
        llm.generate_async(PreprocessedInputs([1, 2], params), cache_salt="tenant")
    submitted = llm._executor.generate_async.call_args.kwargs
    assert submitted["cache_salt"] == "tenant"
    assert submitted["multimodal_params"].multimodal_data == metadata
    assert submitted["multimodal_params"].multimodal_input is params.multimodal_input
    assert "image" in params.multimodal_data


@pytest.mark.parametrize(
    "data",
    [{"image": None}, {"image": []}, {"image": {}}, {"multimodal_embedding": None}],
)
def test_text_only_admission_keeps_empty_prepared_media_unsalted(data) -> None:
    from tensorrt_llm.llmapi.llm import PreprocessedInputs

    llm = _llm_with_mock_executor(DefaultInputProcessor("unused", None, Mock()))
    params = MultimodalParams(multimodal_data=data)
    with patch("tensorrt_llm.llmapi.llm.RequestOutput._from_generation_result"):
        llm.generate_async(PreprocessedInputs([1, 2], params))
    submitted = llm._executor.generate_async.call_args.kwargs
    assert submitted["cache_salt"] is None
    assert submitted["multimodal_params"] is None
    assert params.multimodal_data == data


def test_prepared_media_admission_uses_checkpoint_modalities() -> None:
    from transformers import Gemma4Config
    from transformers.models.gemma4.configuration_gemma4 import Gemma4AudioConfig

    from tensorrt_llm._torch.models.modeling_gemma4mm import Gemma4InputProcessor
    from tensorrt_llm.llmapi.llm import PreprocessedInputs

    processor = object.__new__(Gemma4InputProcessor)
    processor._config = Gemma4Config()

    processor._config.vision_config = processor._config.audio_config = None
    llm = _llm_with_mock_executor(processor)
    image = PreprocessedInputs(
        [1, 2], MultimodalParams(multimodal_data={"image": {"pixels": torch.ones(2)}})
    )
    with pytest.raises(ValueError, match="supports text only"):
        llm.generate_async(image)

    processor._config.audio_config = Gemma4AudioConfig()
    with pytest.raises(ValueError, match="image"):
        llm.generate_async(image)
    llm._executor.generate_async.assert_not_called()

    audio = MultimodalParams(multimodal_data={"audio": {"input_features": torch.ones(2)}})
    with patch("tensorrt_llm.llmapi.llm.RequestOutput._from_generation_result"):
        llm.generate_async(PreprocessedInputs([1, 2], audio))
    submitted = llm._executor.generate_async.call_args.kwargs["multimodal_params"]
    assert submitted.multimodal_data["audio"] is audio.multimodal_data["audio"]


@pytest.mark.parametrize(
    "target_path, target_name",
    [
        ("kimi_k3_vl.kimi_k3_nvfp4__sm_107__tp1", "ModelingV2KimiK3Nvfp4Sm107Tp1"),
        ("minimax_m3.minimax_m3_nvfp4__sm_103__tp4", "ModelingV2MiniMaxM3Nvfp4Sm103Tp4"),
        ("minimax_m3.minimax_m3_nvfp4__sm_107__tp2", "ModelingV2MiniMaxM3Nvfp4Sm107Tp2"),
    ],
)
def test_text_only_target_rejects_prepared_embeddings_before_submission(target_path, target_name):
    from importlib import import_module

    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader
    from tensorrt_llm.llmapi.llm import PreprocessedInputs

    target = getattr(
        import_module(f"tensorrt_llm._torch.modeling_v2.models.{target_path}.modeling"), target_name
    )
    model = target.__new__(target)
    torch.nn.Module.__init__(model)
    llm = _llm_with_mock_executor(_ProbeFakeProcessor())
    loader = ModelLoader(llm.args, SimpleNamespace(rank=0), None, None, 8, 8)
    with patch.object(loader, "_timed_load", return_value=(model, None)):
        loader.load("unused", None)
    llm._startup_metrics = {"model_loader": loader.startup_metadata}
    params = MultimodalParams(multimodal_data={"multimodal_embedding": torch.ones(2, 4)})
    with (
        patch("tensorrt_llm.llmapi.llm.RequestOutput._from_generation_result"),
        pytest.raises(ValueError, match="supports text only"),
    ):
        llm.generate_async(PreprocessedInputs([1, 2], params))
    llm._executor.generate_async.assert_not_called()


def test_raw_media_admission_uses_the_loaded_encoder() -> None:
    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader
    from tensorrt_llm.disaggregated_params import DisaggregatedParams
    from tensorrt_llm.llmapi.llm import PreprocessedInputs

    llm = _llm_with_mock_executor(_ProbeFakeProcessor())
    llm.args.disable_mm_encoder = True
    llm.args.multimodal_config = None

    def load_encoder(encoder, accepts_multimodal_inputs=True):
        model = torch.nn.Module()
        model.accepts_multimodal_inputs = accepts_multimodal_inputs
        model.mm_encoder = encoder
        loader = ModelLoader(llm.args, SimpleNamespace(rank=0), None, None, 8, 8)
        with patch.object(loader, "_timed_load", return_value=(model, None)):
            loader.load("unused", None)
        llm._startup_metrics = {"model_loader": loader.startup_metadata}

    load_encoder(None)
    with pytest.raises(ValueError, match="require a local encoder"):
        llm.preprocess(_PROBE_INPUTS)

    media = MultimodalParams(multimodal_data={"image": {"pixel_values": torch.zeros(3, 2, 2)}})
    with pytest.raises(ValueError, match="require a local encoder"):
        llm.generate_async(PreprocessedInputs([1, 2], media))
    llm._executor.generate_async.assert_not_called()

    llm.input_processor.supported_modalities = frozenset()
    key = "multimodal_embedding"
    embedding = MultimodalParams(multimodal_data={key: torch.zeros(2, 4)})
    with patch("tensorrt_llm.llmapi.llm.RequestOutput._from_generation_result"):
        llm.generate_async(PreprocessedInputs([1, 2], embedding))
        submitted = llm._executor.generate_async.call_args.kwargs["multimodal_params"]
        assert submitted.multimodal_data[key] is embedding.multimodal_data[key]
        embedding.to_handle("multimodal_data")
        try:
            llm.generate_async(PreprocessedInputs([1, 2], embedding))
            submitted = llm._executor.generate_async.call_args.kwargs["multimodal_params"]
            assert submitted.multimodal_data[key] is embedding.multimodal_data[key]
        finally:
            embedding.to_tensor("multimodal_data")

    gen_only = DisaggregatedParams(request_type="generation_only")
    with (
        patch.object(MultimodalParams, "to_handle"),
        patch("tensorrt_llm.llmapi.llm.RequestOutput._from_generation_result"),
    ):
        prepared = llm.preprocess(_PROBE_INPUTS, disaggregated_params=gen_only)
        llm.generate_async(prepared, disaggregated_params=gen_only)
    assert llm._executor.generate_async.call_args.kwargs["disaggregated_params"] is gen_only

    load_encoder(torch.nn.Identity())
    with patch.object(MultimodalParams, "to_handle"):
        assert llm.preprocess(_PROBE_INPUTS).multimodal_params.multimodal_input is not None

    llm._executor.generate_async.reset_mock()
    llm.args.disable_mm_encoder = False
    load_encoder(None, accepts_multimodal_inputs=False)
    with pytest.raises(ValueError, match="supports text only"):
        llm.generate_async(PreprocessedInputs([1, 2], embedding))
    llm._executor.generate_async.assert_not_called()


_MM_SALT = "mm-no-item-spans-" + "0" * 32


@pytest.mark.parametrize(
    "cache_salt",
    ["a" * 206, "a" * 207, "a" * 256, "\u00e9" * 128],
    ids=["fits", "one_byte_over", "at_limit", "multibyte_at_limit"],
)
def test_multimodal_cache_salt_fits_the_salt_limit(cache_salt: str) -> None:
    """A caller salt any request may have stays valid with the multimodal salt added."""
    from tensorrt_llm.executor.request import GenerationRequest
    from tensorrt_llm.sampling_params import SamplingParams

    params = MultimodalParams(multimodal_data={MULTIMODAL_CACHE_SALT_KEY: _MM_SALT})
    salt = request_cache_salt(cache_salt, params)

    assert len(salt.encode("utf-8")) <= GenerationRequest.MAX_CACHE_SALT_LEN
    assert salt.endswith(_MM_SALT)
    if len(cache_salt.encode("utf-8")) + 1 + len(_MM_SALT) <= GenerationRequest.MAX_CACHE_SALT_LEN:
        assert salt == f"{cache_salt}/{_MM_SALT}"
    else:
        assert salt != request_cache_salt("b" * 256, params)
        assert salt != request_cache_salt(salt.split("/", 1)[0], params)
    GenerationRequest([1], SamplingParams(max_tokens=1), cache_salt=salt)


@pytest.mark.parametrize(
    "cache_salt,error",
    [(123, TypeError), ("a" * 257, ValueError)],
    ids=["not_a_string", "too_long"],
)
def test_bad_caller_cache_salt_still_fails_with_a_multimodal_salt(cache_salt, error) -> None:
    """A caller salt no request may have fails as it does without a multimodal salt."""
    from tensorrt_llm.executor.request import GenerationRequest
    from tensorrt_llm.sampling_params import SamplingParams

    params = MultimodalParams(multimodal_data={MULTIMODAL_CACHE_SALT_KEY: _MM_SALT})
    with pytest.raises(error, match="cache_salt"):
        GenerationRequest(
            [1], SamplingParams(max_tokens=1), cache_salt=request_cache_salt(cache_salt, params)
        )


def _process_mm_contract(
    modalities: list[str],
    token_ids: list[int],
    explicit_ids: list[int] | None,
    special_ids: list[int] | None,
    *,
    hashing_enabled: bool = True,
    use_uuids: bool = False,
    frontend_order: bool = True,
) -> tuple[list[int], dict]:
    item_refs = [(modality, 0) for modality in modalities]
    num_items = len(item_refs)

    class Processor(_KwargsHashFakeProcessor):
        multimodal_hashing_supported = hashing_enabled
        mm_bidirectional_blocks = True

        def __call__(self, inputs: dict, sampling_params: object) -> tuple[list[int], dict]:
            return list(token_ids), {
                "multimodal_data": {
                    **{modality: {} for modality in modalities},
                    MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                        item_refs, output_embedding_lengths=[2] * num_items
                    ),
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

        def get_mm_item_metadata(
            self, prompt_token_ids: list[int], multimodal_data: dict, mm_item_order=None
        ) -> MultimodalItemMetadata:
            return MultimodalItemMetadata(item_refs, [4] * num_items, [2] * num_items)

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
    if not frontend_order:
        inputs.pop("mm_item_order")
    if use_uuids:
        inputs["multi_modal_uuids"] = {modality: [f"{modality}-id"] for modality in modalities}

    return create_input_processor_with_hash(Processor())(inputs, None)


@pytest.mark.parametrize("hashing_enabled", [False, True], ids=["fallback", "hashing"])
@pytest.mark.parametrize(
    "modalities,token_ids,explicit_ids,special_ids,expected_cumsum,expected_runs",
    [
        (["image"], [10, 101, 102, 20], None, None, [0, 1, 2, 2], ([0, 1], [1], [2])),
        (["image"], [10, 101, 102, 20], [101, 102], None, [0, 1, 2, 2], ([0, 1], [1], [2])),
        (
            ["image"],
            [10, 101, 200, 102, 20],
            [101, 102],
            [200],
            [0, 1, 1, 2, 2],
            ([0, 1], [1], [3]),
        ),
        (["video"], [10, 101, 20, 102, 30], None, None, [0, 1, 1, 2, 2], ([0, 2], [1, 3], [1, 1])),
        (["audio"], [10, 101, 102, 20], [101, 102], None, [0, 1, 2, 2], ([0, 1], [1], [2])),
        (
            ["video", "image", "audio"],
            [10, 101, 102, 20, 101, 102, 30, 101, 102, 40],
            [101, 102],
            None,
            [0, 1, 2, 2, 3, 4, 4, 5, 6, 6],
            ([0, 1, 2, 3], [1, 4, 7], [2, 2, 2]),
        ),
    ],
    ids=[
        "out-of-vocab-image",
        "explicit-image-tokens",
        "image-framing-token",
        "video-interleaved-text",
        "audio",
        "mixed-prompt-order",
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
) -> None:
    """Pin framework metadata independently of model-specific media encoding."""
    actual_token_ids, extra = _process_mm_contract(
        modalities, token_ids, explicit_ids, special_ids, hashing_enabled=hashing_enabled
    )
    item_refs = [(modality, 0) for modality in modalities]
    num_items = len(item_refs)

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
    mm_input = extra["multimodal_input"]
    assert MULTIMODAL_CACHE_SALT_KEY not in data
    if hashing_enabled:
        hashes = _CONTENT_HASH_SNAPSHOT
        assert mm_input.multimodal_hashes == [hashes[modality] for modality in modalities]
        assert MULTIMODAL_NO_CONTENT_HASHES_KEY not in data
    else:
        # Without content hashes, each item gets a random hash and the same spans.
        assert len(mm_input.multimodal_hashes) == num_items
        assert not any(h in _CONTENT_HASH_SNAPSHOT.values() for h in mm_input.multimodal_hashes)
        assert data[MULTIMODAL_NO_CONTENT_HASHES_KEY] is True
    assert mm_input.multimodal_uuids is None
    assert mm_input.multimodal_positions == [1 + 3 * index for index in range(num_items)]
    assert mm_input.multimodal_lengths == [2 + len(special_ids or [])] * num_items
    assert (
        mm_input.multimodal_item_run_cu_offsets,
        mm_input.multimodal_run_positions,
        mm_input.multimodal_run_lengths,
    ) == expected_runs
    if special_ids:
        assert data["special_token_offsets"] == [1]


@pytest.mark.parametrize("use_uuids", [False, True], ids=["content", "uuid"])
@pytest.mark.parametrize("frontend_order", [False, True], ids=["record-order", "frontend-order"])
def test_hashes_and_uuids_follow_common_item_refs(use_uuids: bool, frontend_order: bool) -> None:
    modalities = ["video", "image", "audio"]
    _, extra = _process_mm_contract(
        modalities,
        [10, 101, 102, 20, 101, 102, 30, 101, 102, 40],
        [101, 102],
        None,
        use_uuids=use_uuids,
        frontend_order=frontend_order,
    )
    mm_input = extra["multimodal_input"]
    hashes = _UUID_HASH_SNAPSHOT if use_uuids else _CONTENT_HASH_SNAPSHOT
    assert mm_input.multimodal_hashes == [hashes[modality] for modality in modalities]
    assert mm_input.multimodal_uuids == (
        [f"{modality}-id" for modality in modalities] if use_uuids else None
    )
    assert mm_input.multimodal_positions == [1, 4, 7]
    assert extra["multimodal_data"][MULTIMODAL_ENCODER_ITEM_METADATA_KEY].item_refs == tuple(
        (modality, 0) for modality in modalities
    )


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

    # Without content hashes, the item gets a random hash for its KV cache keys.
    assert len(params.multimodal_input.multimodal_hashes) == 1
    assert params.multimodal_input.multimodal_positions == [1]
    assert params.multimodal_input.multimodal_lengths == [3]
    assert params.multimodal_data[MULTIMODAL_NO_CONTENT_HASHES_KEY] is True
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
        # pyrefly: ignore [missing-argument]
        MultimodalRuntimeData(past_seen_token_num=0, chunk_end_pos=5)


def test_normalized_params_keep_one_order_and_output_length_record():
    refs = [("audio", 0), ("image", 0)]
    costs = [4, 8]
    rows = [2, 3]
    metadata = MultimodalItemMetadata(refs, costs, rows)
    data = {
        MULTIMODAL_ENCODER_ITEM_METADATA_KEY: metadata,
    }
    params = MultimodalParams(
        multimodal_data=data,
    )
    refs.reverse()
    costs[0] = 100
    rows[0] = 100

    assert params.item_metadata is metadata
    assert "multimodal_embedding_lengths" not in data
    assert metadata.item_refs == (("audio", 0), ("image", 0))
    assert metadata.encoder_token_lengths == (4, 8)
    assert metadata.output_embedding_lengths == (2, 3)
    with pytest.raises(TypeError):
        metadata.output_embedding_lengths[0] = 100
    with pytest.raises(AttributeError):
        metadata.encoder_token_lengths = (100, 8)


def test_params_do_not_infer_item_layout_from_raw_buckets():
    params = MultimodalParams(multimodal_data={"audio": {}, "image": {}})
    assert params.item_metadata is None
    assert params.get_item_layout() == ((), ())
