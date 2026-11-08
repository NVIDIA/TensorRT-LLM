# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normal-package CPU regressions for the actual mixed-media input path."""

import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tensorrt_llm._torch.models.modeling_minimaxm3_vl import (
    MiniMaxM3VLInputProcessor,
    extract_multimodal_items_in_request_order,
    get_minimax_m3_vl_input_processor_cls,
)
from tensorrt_llm.inputs import registry
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalItemMetadata,
    MultimodalParams,
    _has_mm_payload_keys,
    check_mm_embed_cumsum_if_needed,
    get_multimodal_item_metadata,
)
from tensorrt_llm.inputs.multimodal_data import VideoData

IMAGE, VIDEO, START, END = 200025, 200026, 200029, 200030


pytestmark = pytest.mark.cpu_only


def processor_and_media(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[MiniMaxM3VLInputProcessor, dict[str, list[np.ndarray | VideoData]]]:
    cls = get_minimax_m3_vl_input_processor_cls()
    processor = cls.__new__(cls)
    processor._image_token_id, processor._video_token_id = IMAGE, VIDEO
    processor._vision_start_token_id, processor._vision_end_token_id = START, END
    processor._config = SimpleNamespace(vocab_size=200064)
    processor._multimodal_hashing_supported = None
    processor._processor = SimpleNamespace(
        image_processor=SimpleNamespace(merge_size=1, get_number_of_image_patches=lambda h, w: 4),
        video_processor=SimpleNamespace(merge_size=1, temporal_patch_size=2),
    )
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    video = VideoData(frames=[image.copy(), image.copy()], metadata={}, audio=None)
    media = {"image": [image], "video": [video]}
    image_pixels = torch.full((4, 6), 11.0)
    video_pixels = torch.full((4, 6), 29.0)

    def encoded_payload(*args, **kwargs) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
        # Only the unrelated HF encoder is a test double. Expansion, per-item
        # token counting, order propagation and feature concatenation stay real.
        return {
            "multimodal_data": {
                "image": {
                    "pixel_values": image_pixels,
                    "image_grid_thw": torch.tensor([[1, 2, 2]]),
                },
                "video": {
                    "pixel_values_videos": video_pixels,
                    "video_grid_thw": torch.tensor([[1, 2, 2]]),
                },
            }
        }

    monkeypatch.setattr(registry, "_process_multimodal_with_dummy_placeholders", encoded_payload)
    return processor, media


@pytest.mark.parametrize("first,second", [(IMAGE, VIDEO), (VIDEO, IMAGE)])
def test_mixed_placeholder_and_vision_features_follow_actual_prompt_order(
    monkeypatch, first, second
) -> None:
    processor, media = processor_and_media(monkeypatch)
    source = {"prompt_token_ids": [3, first, 4, second, 5], "multi_modal_data": media}
    expanded, extra = processor.call_with_token_ids(source, SimpleNamespace())
    assert expanded == [3, START, *([first] * 4), END, 4, START, *([second] * 4), END, 5]
    expected_order = ["image" if token == IMAGE else "video" for token in (first, second)]
    metadata = get_multimodal_item_metadata(extra["multimodal_data"])
    assert metadata.item_refs == tuple((modality, 0) for modality in expected_order)
    assert metadata.encoder_token_lengths is None
    assert "layout_metadata" not in extra["multimodal_data"]
    items = extract_multimodal_items_in_request_order(
        [MultimodalParams(multimodal_data=extra["multimodal_data"])]
    )
    assert [item["modality"] for item in items] == expected_order
    assert [float(item["pixel_values"][0, 0]) for item in items] == [
        11.0 if modality == "image" else 29.0 for modality in expected_order
    ]
    assert source["prompt_token_ids"] == [3, first, 4, second, 5]
    assert "mm_item_order" not in source


def test_mixed_video_retains_each_temporal_bracket(monkeypatch) -> None:
    processor, media = processor_and_media(monkeypatch)
    media["video"][0].frames.extend([media["image"][0].copy(), media["image"][0].copy()])

    def encoded_payload(*args, **kwargs) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
        return {
            "multimodal_data": {
                "image": {
                    "pixel_values": torch.zeros((4, 6)),
                    "image_grid_thw": torch.tensor([[1, 2, 2]]),
                },
                "video": {
                    "pixel_values_videos": torch.zeros((8, 6)),
                    "video_grid_thw": torch.tensor([[2, 2, 2]]),
                },
            }
        }

    monkeypatch.setattr(registry, "_process_multimodal_with_dummy_placeholders", encoded_payload)
    expanded, _ = processor.call_with_token_ids(
        {"prompt_token_ids": [IMAGE, VIDEO], "multi_modal_data": media}
    )
    assert expanded == [
        START,
        *([IMAGE] * 4),
        END,
        START,
        *([VIDEO] * 4),
        END,
        START,
        *([VIDEO] * 4),
        END,
    ]


def test_real_hashing_wrapper_uses_encoded_prompt_order_and_both_lengths(monkeypatch) -> None:
    processor, media = processor_and_media(monkeypatch)

    def encoded_payload(*args, **kwargs) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
        return {
            "multimodal_data": {
                "image": {
                    "pixel_values": torch.zeros((4, 6)),
                    "image_grid_thw": torch.tensor([[1, 2, 2]]),
                },
                "video": {
                    "pixel_values_videos": torch.zeros((8, 6)),
                    "video_grid_thw": torch.tensor([[1, 2, 4]]),
                },
            }
        }

    monkeypatch.setattr(registry, "_process_multimodal_with_dummy_placeholders", encoded_payload)
    wrapped = registry.create_input_processor_with_hash(processor)
    _, extra = wrapped(
        {"prompt_token_ids": [3, VIDEO, 4, IMAGE, 5], "multi_modal_data": media}, SimpleNamespace()
    )
    metadata = extra["multimodal_input"]
    hashes, _ = registry.apply_mm_hashes(media, None, registry.default_hasher)
    assert metadata.multimodal_lengths == [10, 6]
    assert metadata.multimodal_hashes == [
        registry.hexdigest_to_int32(hashes["video"][0]),
        registry.hexdigest_to_int32(hashes["image"][0]),
    ]
    assert processor.multimodal_hashing_supported is True


def test_real_hashing_wrapper_rejects_conflicting_incoming_order(monkeypatch) -> None:
    processor, media = processor_and_media(monkeypatch)
    wrapped = registry.create_input_processor_with_hash(processor)
    incoming = [{"modality": "image", "index": 0}, {"modality": "video", "index": 0}]
    with pytest.raises(ValueError, match="conflicts"):
        wrapped(
            {
                "prompt_token_ids": [VIDEO, IMAGE],
                "multi_modal_data": media,
                "mm_item_order": incoming,
            },
            SimpleNamespace(),
        )


def test_matching_input_manifest_may_retain_placeholder_text_metadata(monkeypatch) -> None:
    processor, media = processor_and_media(monkeypatch)
    source_order = [
        {"modality": "video", "index": 0, "placeholder": "]<]video[>["},
        {"modality": "image", "index": 0, "placeholder": "]<]image[>["},
    ]
    _, extra = processor.call_with_token_ids(
        {
            "prompt_token_ids": [VIDEO, IMAGE],
            "multi_modal_data": media,
            "mm_item_order": source_order,
        }
    )
    assert get_multimodal_item_metadata(extra["multimodal_data"]).item_refs == (
        ("video", 0),
        ("image", 0),
    )


def test_timestamped_video_and_interleaved_image_keep_tokens_features_and_hashes_aligned(
    monkeypatch,
) -> None:
    processor, media = processor_and_media(monkeypatch)
    frame = media["image"][0]
    media["video"][0] = VideoData(
        frames=[frame.copy() for _ in range(4)],
        metadata={"fps": 1.0, "frames_indices": [4, 6, 8, 10], "duration": 12.0},
        audio=None,
    )
    encoded_stamps = []

    def encode_timestamp(text: str, *, add_special_tokens: bool) -> list[int]:
        assert add_special_tokens is False
        encoded_stamps.append(text)
        return {
            "]<]4.0 seconds[>[": [9004],
            "]<]8.0 seconds[>[": [9008],
        }[text]

    # Match the checkpoint's actual timestamp spelling exactly, while keeping
    # tokenizer vocabulary data outside this CPU unit fixture.
    processor._tokenizer = SimpleNamespace(encode=encode_timestamp)

    def encoded_payload(*args, **kwargs) -> dict[str, dict[str, dict[str, torch.Tensor]]]:
        return {
            "multimodal_data": {
                "image": {
                    "pixel_values": torch.full((4, 6), 11.0),
                    "image_grid_thw": torch.tensor([[1, 2, 2]]),
                },
                "video": {
                    "pixel_values_videos": torch.full((8, 6), 29.0),
                    "video_grid_thw": torch.tensor([[2, 2, 2]]),
                },
            }
        }

    monkeypatch.setattr(registry, "_process_multimodal_with_dummy_placeholders", encoded_payload)
    wrapped = registry.create_input_processor_with_hash(processor)
    expanded, extra = wrapped(
        {
            "prompt_token_ids": [3, VIDEO, 4, IMAGE, 5],
            "multi_modal_data": media,
            "mm_item_order": [
                {"modality": "video", "index": 0, "placeholder": "]<]video[>["},
                {"modality": "image", "index": 0, "placeholder": "]<]image[>["},
            ],
        },
        SimpleNamespace(),
    )
    assert encoded_stamps == ["]<]4.0 seconds[>[", "]<]8.0 seconds[>["]
    assert expanded == [
        3,
        9004,
        START,
        *([VIDEO] * 4),
        END,
        9008,
        START,
        *([VIDEO] * 4),
        END,
        4,
        START,
        *([IMAGE] * 4),
        END,
        5,
    ]
    metadata = extra["multimodal_input"]
    assert metadata.multimodal_lengths == [12, 6]
    assert get_multimodal_item_metadata(extra["multimodal_data"]).output_embedding_lengths == (8, 4)
    assert metadata.multimodal_item_run_cu_offsets == [0, 2, 3]
    assert metadata.multimodal_run_positions == [2, 9, 16]
    assert metadata.multimodal_run_lengths == [6, 6, 6]
    hashes, _ = registry.apply_mm_hashes(media, None, registry.default_hasher)
    assert metadata.multimodal_hashes == [
        registry.hexdigest_to_int32(hashes["video"][0]),
        registry.hexdigest_to_int32(hashes["image"][0]),
    ]
    items = extract_multimodal_items_in_request_order(
        [MultimodalParams(multimodal_data=extra["multimodal_data"])]
    )
    assert [item["modality"] for item in items] == ["video", "image"]
    assert [item["pixel_values"].shape[0] for item in items] == [8, 4]
    assert [float(item["pixel_values"][0, 0]) for item in items] == [29.0, 11.0]


@pytest.mark.parametrize("ids", [[IMAGE], [VIDEO], [IMAGE, IMAGE, VIDEO]])
def test_mixed_duplicate_or_missing_placeholder_is_rejected(monkeypatch, ids) -> None:
    processor, media = processor_and_media(monkeypatch)
    with pytest.raises(ValueError, match="one placeholder"):
        processor.call_with_token_ids({"prompt_token_ids": ids, "multi_modal_data": media})


def test_legacy_unqualified_mixed_length_list_is_still_rejected(monkeypatch) -> None:
    processor, media = processor_and_media(monkeypatch)
    # Preserve the old fail-closed API: one modality's anonymous length list is
    # not sufficient to enable the mixed path, even after the new full route.
    with pytest.raises(ValueError, match="mixed image"):
        processor.expand_prompt_token_ids_for_mm([IMAGE, VIDEO], [6], mm_data=media)


def test_mixed_multiple_videos_remain_explicitly_unsupported(monkeypatch) -> None:
    processor, media = processor_and_media(monkeypatch)
    media["video"].append(media["video"][0])
    with pytest.raises(ValueError, match="exactly one image and one video"):
        processor.call_with_token_ids(
            {"prompt_token_ids": [IMAGE, VIDEO, VIDEO], "multi_modal_data": media}
        )


def test_encoded_order_cannot_name_missing_or_duplicate_modalities() -> None:
    payload = {
        "image": {"pixel_values": torch.zeros((1, 6)), "image_grid_thw": torch.tensor([[1, 1, 1]])},
        "video": {
            "pixel_values_videos": torch.zeros((1, 6)),
            "video_grid_thw": torch.tensor([[1, 1, 1]]),
        },
        MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata([("image", 0), ("audio", 0)]),
    }
    with pytest.raises(ValueError, match="does not match"):
        extract_multimodal_items_in_request_order([MultimodalParams(multimodal_data=payload)])


def test_order_manifest_alone_remains_metadata_during_cached_or_chunked_decode() -> None:
    metadata = {
        MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata([("video", 0), ("image", 0)])
    }
    assert _has_mm_payload_keys(metadata) is False
    check_mm_embed_cumsum_if_needed(metadata, begin_compute=5, end_compute=8, prompt_len=10)


def test_mixed_order_survives_common_params_serialization(monkeypatch) -> None:
    processor, media = processor_and_media(monkeypatch)
    _, extra = processor.call_with_token_ids(
        {"prompt_token_ids": [VIDEO, IMAGE], "multi_modal_data": media}
    )
    restored = pickle.loads(
        pickle.dumps(MultimodalParams(multimodal_data=extra["multimodal_data"]))
    )
    assert restored.item_metadata.item_refs == (("video", 0), ("image", 0))
    assert [item["modality"] for item in extract_multimodal_items_in_request_order([restored])] == [
        "video",
        "image",
    ]


def test_single_modality_common_metadata_retains_grouped_rows() -> None:
    pixels = torch.zeros((8, 6))
    params = MultimodalParams(
        multimodal_data={
            "image": {
                "pixel_values": pixels,
                "image_grid_thw": torch.tensor([[1, 2, 2], [1, 2, 2]]),
            },
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                [("image", 0), ("image", 1)]
            ),
        }
    )
    items = extract_multimodal_items_in_request_order([params])
    assert len(items) == 1 and items[0]["pixel_values"] is pixels


def test_mixed_processor_rejects_conflicting_prepared_common_metadata(monkeypatch) -> None:
    processor, media = processor_and_media(monkeypatch)
    original = registry._process_multimodal_with_dummy_placeholders

    def encoded_with_conflict(*args, **kwargs) -> dict[str, dict[str, object]]:
        extra = original(*args, **kwargs)
        extra["multimodal_data"][MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = MultimodalItemMetadata(
            [("image", 0), ("video", 0)]
        )
        return extra

    monkeypatch.setattr(
        registry, "_process_multimodal_with_dummy_placeholders", encoded_with_conflict
    )
    with pytest.raises(ValueError, match="conflicts"):
        processor.call_with_token_ids(
            {"prompt_token_ids": [VIDEO, IMAGE], "multi_modal_data": media}
        )


@pytest.mark.parametrize("provided_rows", [None, (4, 4), (9, 4)])
def test_mixed_expansion_merges_common_output_rows_without_changing_source_metadata(
    monkeypatch, provided_rows
) -> None:
    processor, media = processor_and_media(monkeypatch)
    original = registry._process_multimodal_with_dummy_placeholders
    prepared = MultimodalItemMetadata(
        [("video", 0), ("image", 0)], output_embedding_lengths=provided_rows
    )

    def encoded_with_metadata(*args, **kwargs) -> dict[str, dict[str, object]]:
        extra = original(*args, **kwargs)
        extra["multimodal_data"][MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = prepared
        return extra

    monkeypatch.setattr(
        registry, "_process_multimodal_with_dummy_placeholders", encoded_with_metadata
    )
    inputs = {"prompt_token_ids": [VIDEO, IMAGE], "multi_modal_data": media}
    if provided_rows == (9, 4):
        with pytest.raises(ValueError, match="rows conflict"):
            processor.call_with_token_ids(inputs)
    else:
        _, extra = processor.call_with_token_ids(inputs)
        actual = get_multimodal_item_metadata(extra["multimodal_data"])
        assert actual.output_embedding_lengths == (4, 4)
        assert actual.input_partitions is prepared.input_partitions
        assert actual.encoder_token_lengths is None
        assert prepared.output_embedding_lengths == provided_rows


def test_mixed_single_items_are_normalized_without_mutating_the_caller(monkeypatch) -> None:
    processor, media = processor_and_media(monkeypatch)
    image, video = media["image"][0], media["video"][0]
    raw = {"image": image, "video": video}
    expanded, extra = processor.call_with_token_ids(
        {"prompt_token_ids": [VIDEO, IMAGE], "multi_modal_data": raw}
    )
    assert expanded == [START, *([VIDEO] * 4), END, START, *([IMAGE] * 4), END]
    assert get_multimodal_item_metadata(extra["multimodal_data"]).item_refs == (
        ("video", 0),
        ("image", 0),
    )
    assert raw["image"] is image and raw["video"] is video
