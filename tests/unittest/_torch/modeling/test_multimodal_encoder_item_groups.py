# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Junction tests: item-level MM encoder scheduling over `EncoderGroup` models.

The two features are developed independently and each has its own coverage, but
nothing exercised them *together* until this file. They meet at a specific
contract: `encode_multimodal_by_groups` reads `multimodal_embedding_lengths`
off each `MultimodalParams` to split the encoder output back per modality, while
item scheduling feeds it freshly sliced single-item params. A slice that omits
that key splits the encoder output to zero rows, so every scheduled item comes
back empty; the row-count check in `forward_multimodal_encoder_items` turns that
into a hard error, but only after the wasted encoder forward.
"""

import pytest
import torch

from tensorrt_llm._torch.models.modeling_multimodal_mixin import (
    EncoderGroup,
    MultimodalEncoderContractError,
    MultimodalModelMixin,
    encode_multimodal_by_groups,
)
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.inputs.registry import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalEncoderItemMetadata,
)

HIDDEN = 4
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
    # Production attaches this alongside the item metadata (see
    # `inputs/registry.py`); `_apply_metadata_slice` and
    # `encode_multimodal_by_groups` both read it from the top level.
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
