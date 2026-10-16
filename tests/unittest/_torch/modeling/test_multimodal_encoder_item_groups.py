# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-request and selected-item encoding share prompt-order item metadata.

Group outputs follow modality-local payload order. Selecting items must slice
the tensors and their metadata together, so output assembly returns the same
rows and order as encoding those items in the full request.
"""

from functools import partial

import pytest
import torch

from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
    EncoderGroup,
    MultimodalEncoderContractError,
    MultimodalModelMixin,
    encode_multimodal_by_groups,
)
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalEncoderItemMetadata,
    MultimodalParams,
)

HIDDEN = 4
pytestmark = pytest.mark.cpu_only
# Rows the encoder emits per patch-grid item. Distinct values catch splits that
# happen to line up only for uniform sizes.
ITEM_ROWS = {"image": [2, 3], "video": [4]}


def _grid_for_rows(rows: int) -> list[int]:
    """A (t, h, w) grid whose product is the item's patch count."""
    return [1, 1, rows]


class _GroupedEncoderModel(MultimodalModelMixin):
    """Minimal model whose encoder serves image and video in one call.

    Mirrors the Qwen3-VL shape: `encode_multimodal_inputs` delegates to
    `encode_multimodal_by_groups`, which is what makes the
    `multimodal_embedding_lengths` contract load-bearing.
    """

    supports_mm_encoder_item_scheduling = True

    def __init__(self):
        self.encoder_calls: list[int] = []
        self.mm_encoder_groups = (
            EncoderGroup(
                modalities=("image", "video"),
                encoder_fn=self._encode,
                build_batched_input=self._build_batched_input,
            ),
        )

    @staticmethod
    def _build_batched_input(params: list[MultimodalParams]) -> dict:
        # Per the EncoderGroup contract: all items of the first modality across
        # requests, then all items of the second.
        pixels = []
        for modality, pixel_key in (("image", "pixel_values"), ("video", "pixel_values_videos")):
            for mp in params:
                data = mp.multimodal_data.get(modality)
                if data is not None:
                    pixels.append(data[pixel_key])
        return {"pixel_values": torch.cat(pixels, dim=0)}

    def _encode(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.encoder_calls.append(pixel_values.shape[0])
        # One output row per input patch row, carrying the input value so the
        # test can assert *which* rows came back, not just how many.
        return pixel_values.clone()

    def encode_multimodal_inputs(self, multimodal_params) -> torch.Tensor:
        return encode_multimodal_by_groups(self.mm_encoder_groups, list(multimodal_params))


def _make_request(modalities: list[str], *, value_start: float = 1.0) -> MultimodalParams:
    """Build a request whose items carry unique, checkable patch values."""
    per_modality_rows: dict[str, list[int]] = {}
    item_refs: list[tuple[str, int]] = []
    lengths: list[int] = []
    for modality in modalities:
        local_idx = len(per_modality_rows.setdefault(modality, []))
        rows = ITEM_ROWS[modality][local_idx % len(ITEM_ROWS[modality])]
        per_modality_rows[modality].append(rows)
        item_refs.append((modality, local_idx))
        lengths.append(rows)

    multimodal_data: dict = {}
    value = value_start
    for modality, rows_per_item in per_modality_rows.items():
        pixel_key = "pixel_values" if modality == "image" else "pixel_values_videos"
        grid_key = "image_grid_thw" if modality == "image" else "video_grid_thw"
        blocks = []
        for rows in rows_per_item:
            blocks.append(torch.full((rows, HIDDEN), value))
            value += 1.0
        multimodal_data[modality] = {
            pixel_key: torch.cat(blocks, dim=0),
            grid_key: torch.tensor([_grid_for_rows(r) for r in rows_per_item]),
        }
    multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = MultimodalEncoderItemMetadata(
        item_refs=item_refs,
        encoder_token_lengths=list(lengths),
        output_embedding_lengths=list(lengths),
    )
    # Processors retain this field for legacy consumers. Encoder groups use
    # the typed item metadata when available.
    multimodal_data["multimodal_embedding_lengths"] = list(lengths)
    return MultimodalParams(multimodal_data=multimodal_data)


def _expected_rows(request: MultimodalParams, item_idx: int) -> torch.Tensor:
    """The patch rows the encoder should return for one item."""
    metadata = request.multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY]
    modality, local_idx = metadata.item_refs[item_idx]
    data = request.multimodal_data[modality]
    pixel_key = "pixel_values" if modality == "image" else "pixel_values_videos"
    grids = data["image_grid_thw" if modality == "image" else "video_grid_thw"]
    counts = torch.prod(grids, dim=1).tolist()
    start = sum(int(c) for c in counts[:local_idx])
    return data[pixel_key][start : start + int(counts[local_idx])]


def _run_items(model, selected):
    encoder_inputs = model.prepare_multimodal_encoder_inputs(selected)
    return model.forward_multimodal_encoder_items(encoder_inputs), encoder_inputs


def test_sliced_items_survive_the_grouped_encoder_path():
    """Regression: without `multimodal_embedding_lengths` on the slice, the
    grouped path splits to zero rows and returns nothing for every item."""
    model = _GroupedEncoderModel()
    request = _make_request(["image", "image"])
    selected = [(request, 0), (request, 1)]

    outputs, _ = _run_items(model, selected)

    assert len(outputs) == 2
    for item_idx, output in enumerate(outputs):
        expected = _expected_rows(request, item_idx)
        assert output.shape == expected.shape
        torch.testing.assert_close(output, expected)


def test_adjacent_same_modality_items_share_one_encoder_call():
    """`forward_multimodal_encoder_items` batches consecutive same-modality
    inputs into a single grouped-encoder invocation."""
    model = _GroupedEncoderModel()
    request = _make_request(["image", "image"])

    outputs, encoder_inputs = _run_items(model, [(request, 0), (request, 1)])

    assert len(encoder_inputs) == 1, "adjacent same-modality items must be sliced together"
    assert encoder_inputs[0][1] == ITEM_ROWS["image"][:2]
    assert model.encoder_calls == [sum(ITEM_ROWS["image"][:2])]
    assert len(outputs) == 2


def test_mixed_modality_items_keep_scheduler_order():
    """A request interleaving modalities must still get one output per item,
    in scheduler order -- this is where the per-modality reorder could bite."""
    model = _GroupedEncoderModel()
    request = _make_request(["image", "video", "image"])
    selected = [(request, idx) for idx in range(3)]

    outputs, encoder_inputs = _run_items(model, selected)

    # image / video / image cannot merge across the modality change.
    assert [modality for _, _, modality in encoder_inputs] == ["image", "video", "image"]
    assert len(outputs) == 3
    for item_idx, output in enumerate(outputs):
        torch.testing.assert_close(output, _expected_rows(request, item_idx))


@pytest.mark.parametrize("modalities", [["video", "image", "image"], ["image", "video", "image"]])
@pytest.mark.parametrize("legacy_order", [False, True])
def test_full_and_item_encoder_paths_share_item_metadata(
    modalities: list[str], legacy_order: bool
) -> None:
    model = _GroupedEncoderModel()
    request = _make_request(modalities)
    metadata = request.multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY]
    if legacy_order:
        request.mm_item_order = [
            {"modality": modality, "index": index} for modality, index in metadata.item_refs
        ]
    # Full-request encoding reads both order and lengths from the same
    # metadata that drives the scheduler's item selection.
    request.multimodal_data.pop("multimodal_embedding_lengths")

    full_output = model.encode_multimodal_inputs([request])
    item_outputs, _ = _run_items(model, [(request, i) for i in range(len(modalities))])
    expected = torch.cat([_expected_rows(request, i) for i in range(len(modalities))])

    torch.testing.assert_close(full_output, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(item_outputs), expected, rtol=0, atol=0)


def test_metadata_is_rebased_for_partial_and_reordered_items() -> None:
    model = _GroupedEncoderModel()
    request = _make_request(["image", "video", "image"])
    original = request.multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY]

    outputs, encoder_inputs = _run_items(model, [(request, 2), (request, 0)])

    assert len(encoder_inputs) == 1
    residual = encoder_inputs[0][0]
    metadata = residual.multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY]
    assert metadata == MultimodalEncoderItemMetadata([("image", 0), ("image", 1)], [3, 2], [3, 2])
    assert residual.multimodal_data["multimodal_embedding_lengths"] == [3, 2]
    assert request.multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] is original
    torch.testing.assert_close(outputs[0], _expected_rows(request, 2), rtol=0, atol=0)
    torch.testing.assert_close(outputs[1], _expected_rows(request, 0), rtol=0, atol=0)


@pytest.mark.parametrize("use_metadata", [False, True])
def test_full_and_item_order_can_differ_from_raw_payload_order(use_metadata: bool) -> None:
    model = _GroupedEncoderModel()
    request = _make_request(["image", "image"])
    request.multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = MultimodalEncoderItemMetadata(
        [("image", 1), ("image", 0)], [3, 2], [3, 2]
    )
    expected = torch.cat([_expected_rows(request, 0), _expected_rows(request, 1)])
    request.multimodal_data["multimodal_embedding_lengths"] = [3, 2]
    if use_metadata:
        item_outputs, _ = _run_items(model, [(request, 0), (request, 1)])
        torch.testing.assert_close(torch.cat(item_outputs), expected, rtol=0, atol=0)
    else:
        request.multimodal_data.pop(MULTIMODAL_ENCODER_ITEM_METADATA_KEY)
        request.mm_item_order = [
            {"modality": "image", "index": 1},
            {"modality": "image", "index": 0},
        ]

    torch.testing.assert_close(model.encode_multimodal_inputs([request]), expected, rtol=0, atol=0)


@pytest.mark.parametrize("shared_vision_encoder", [False, True])
def test_vision_and_audio_groups_share_full_and_item_ordering(shared_vision_encoder: bool) -> None:
    model = _GroupedEncoderModel()

    def build_vision_input(
        params: list[MultimodalParams], modality: str
    ) -> dict[str, torch.Tensor]:
        pixel_key = "pixel_values" if modality == "image" else "pixel_values_videos"
        return {
            "pixel_values": torch.cat([mp.multimodal_data[modality][pixel_key] for mp in params])
        }

    if not shared_vision_encoder:
        model.mm_encoder_groups = tuple(
            EncoderGroup((modality,), model._encode, partial(build_vision_input, modality=modality))
            for modality in ("image", "video")
        )

    def build_audio_input(params: list[MultimodalParams]) -> dict[str, torch.Tensor]:
        return {
            "pixel_values": torch.cat(
                [mp.multimodal_data["audio"]["input_features"].flatten(0, 1) for mp in params]
            )
        }

    model.mm_encoder_groups += (EncoderGroup(("audio",), model._encode, build_audio_input),)
    request = _make_request(["video", "image"])
    audio = torch.stack([torch.full((2, HIDDEN), value) for value in (10.0, 20.0)])
    request.multimodal_data["audio"] = {"input_features": audio}
    request.multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = MultimodalEncoderItemMetadata(
        [("video", 0), ("audio", 0), ("image", 0), ("audio", 1)], [4, 2, 2, 2], [4, 2, 2, 2]
    )
    request.multimodal_data.pop("multimodal_embedding_lengths")
    expected = torch.cat(
        [_expected_rows(request, 0), audio[0], _expected_rows(request, 2), audio[1]]
    )

    full_output = model.encode_multimodal_inputs([request])
    item_outputs, _ = _run_items(model, [(request, i) for i in range(4)])

    torch.testing.assert_close(full_output, expected, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(item_outputs), expected, rtol=0, atol=0)


def test_partial_selection_only_encodes_selected_items():
    """The scheduler may pick a subset (the rest being cache hits or deferred);
    unselected items must not leak into the encoder input."""
    model = _GroupedEncoderModel()
    request = _make_request(["image", "image"])

    outputs, _ = _run_items(model, [(request, 1)])

    assert len(outputs) == 1
    torch.testing.assert_close(outputs[0], _expected_rows(request, 1))
    assert model.encoder_calls == [ITEM_ROWS["image"][1]]


def test_wrong_encoder_output_rows_are_request_contract_error():
    class _WrongRowsModel(_GroupedEncoderModel):
        def encode_multimodal_inputs(self, multimodal_params):
            output = super().encode_multimodal_inputs(multimodal_params)
            return output[:-1]

    model = _WrongRowsModel()
    request = _make_request(["image"])
    encoder_inputs = model.prepare_multimodal_encoder_inputs([(request, 0)])

    with pytest.raises(MultimodalEncoderContractError, match="rows declared"):
        model.forward_multimodal_encoder_items(encoder_inputs)


def test_items_from_multiple_requests_stay_separated():
    model = _GroupedEncoderModel()
    first = _make_request(["image"])
    second = _make_request(["image"], value_start=9.0)

    outputs, encoder_inputs = _run_items(model, [(first, 0), (second, 0)])

    assert len(encoder_inputs) == 2
    assert model.encoder_calls == [2 * ITEM_ROWS["image"][0]]
    torch.testing.assert_close(outputs[0], _expected_rows(first, 0))
    torch.testing.assert_close(outputs[1], _expected_rows(second, 0))
