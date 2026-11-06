# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for MiniMax-M3 shared encoder input batching."""

from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3VLForConditionalGeneration
from tensorrt_llm._torch.models.modeling_minimaxm3_vl import (
    CLIPVisionConfig,
    MiniMaxM3VLInputProcessor,
    MiniMaxVLVisionModel,
)
from tensorrt_llm._torch.models.modeling_multimodal_mixin import encode_multimodal_by_groups
from tensorrt_llm.inputs.multimodal import MULTIMODAL_ENCODER_ITEM_METADATA_KEY, MultimodalParams

pytestmark = pytest.mark.cpu_only

_IMAGE = 50
_VIDEO = 51


def _main_extract_multimodal_items_in_request_order(
    multimodal_params: Sequence[Any],
) -> List[Dict[str, Any]]:
    """Walk ``multimodal_params`` and return per-item entries in request order.

    Where :func:`extract_multimodal_data_from_params` concatenates all
    image rows together and all video rows together (image-first
    bucketing), this helper keeps the natural left-to-right request
    order. Each MultimodalParams contributes at most one image item and
    at most one video item — within a single request the M3 processor
    expands all image placeholders first, then all video placeholders,
    so a request that carries both modalities still emits image then
    video in left-to-right order.

    Returned list element schema:
        {
            "modality":     "image" | "video",
            "pixel_values": Tensor [N_rows, C*T*P*P],
            "grid_thw":     Tensor [n_grids, 3],
        }
    """
    items: List[Dict[str, Any]] = []
    for params in multimodal_params:
        if params is None:
            continue
        data = getattr(params, "multimodal_data", None)
        if not data:
            continue
        img = data.get("image") if isinstance(data, Mapping) else None
        if isinstance(img, Mapping):
            pv = img.get("pixel_values")
            gthw = img.get("image_grid_thw")
            if pv is not None and gthw is not None:
                items.append({"modality": "image", "pixel_values": pv, "grid_thw": gthw})
        vid = data.get("video") if isinstance(data, Mapping) else None
        if isinstance(vid, Mapping):
            pv = vid.get("pixel_values_videos")
            if pv is None:
                pv = vid.get("pixel_values")
            gthw = vid.get("video_grid_thw")
            if pv is not None and gthw is not None:
                items.append({"modality": "video", "pixel_values": pv, "grid_thw": gthw})
    return items


class _RecordingTower(nn.Module):
    def __init__(self, real):
        super().__init__()
        self.dtype = torch.float32
        self.calls = []
        if real:
            config = CLIPVisionConfig(
                hidden_size=24,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                patch_size=2,
                image_size=8,
                img_token_compression_config={"spatial_merge_size": 2, "temporal_patch_size": 2},
            )
            self.tower = MiniMaxVLVisionModel(config, text_hidden_size=8, dtype=torch.float32)
        else:
            self.tower = None

    def forward(self, pixel_values, grid_thw):
        self.calls.append((pixel_values, grid_thw))
        if self.tower is not None:
            return self.tower(pixel_values=pixel_values, grid_thw=grid_thw)
        return pixel_values.reshape(-1, 4, pixel_values.shape[-1]).mean(dim=1)[:, :8]


class _CpuEncoder:
    mm_encoder_groups = MiniMaxM3VLForConditionalGeneration.mm_encoder_groups
    _encode_vision = MiniMaxM3VLForConditionalGeneration._encode_vision

    def __init__(self, real=False):
        torch.manual_seed(1234)
        self.model = SimpleNamespace(embed_tokens=nn.Embedding(64, 8))
        self.vision_tower = _RecordingTower(real)


def _processor():
    processor = object.__new__(MiniMaxM3VLInputProcessor)
    processor._config = SimpleNamespace(
        vision_config={"img_token_compression_config": {"spatial_merge_size": 2}}
    )
    processor._image_token_id = _IMAGE
    processor._video_token_id = _VIDEO
    return processor


def _request(items, *, manifest=True):
    data = {}
    order = []
    tokens = []
    for modality, grid, tag in items:
        pixel_key, grid_key = (
            ("pixel_values", "image_grid_thw")
            if modality == "image"
            else ("pixel_values_videos", "video_grid_thw")
        )
        payload = data.setdefault(modality, {pixel_key: [], grid_key: []})
        order.append({"modality": modality, "index": len(payload[grid_key])})
        rows = grid[0] * grid[1] * grid[2]
        payload[pixel_key].append(
            torch.arange(rows * 24, dtype=torch.float32).reshape(rows, 24) / 100 + tag
        )
        payload[grid_key].append(torch.tensor([grid], dtype=torch.int64))
        tokens.extend([_IMAGE if modality == "image" else _VIDEO] * (rows // 4))
        tokens.append(7)
    for payload in data.values():
        for key, values in payload.items():
            payload[key] = torch.cat(values)
    metadata = _processor().get_mm_item_metadata(tokens, data, order if manifest else None)
    if metadata is not None:
        data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = metadata
    return MultimodalParams(multimodal_data=data), tokens


def _encode_like_main(model, params):
    items = _main_extract_multimodal_items_in_request_order(params)
    embed_tokens = model.model.embed_tokens
    mm_embeds: List[torch.Tensor] = []
    vision_dtype = getattr(model.vision_tower, "dtype", embed_tokens.weight.dtype)
    vision_device = embed_tokens.weight.device
    for item in items:
        feats = model.vision_tower(
            pixel_values=item["pixel_values"].to(device=vision_device, dtype=vision_dtype),
            grid_thw=item["grid_thw"],
        )
        mm_embeds.append(feats.to(dtype=embed_tokens.weight.dtype, device=vision_device))
    return torch.cat(mm_embeds, dim=0)


@pytest.mark.parametrize("real", [False, True], ids=["stub-exact", "tiny-random-tower"])
@pytest.mark.parametrize(
    "requests",
    [
        [[("image", (1, 2, 4), 1)]],
        [[("image", (1, 2, 2), 1)], [("image", (1, 4, 4), 2)]],
        [[("video", (2, 2, 4), 1)], [("image", (1, 4, 2), 2)]],
        [[("image", (1, 2, 2), 1), ("video", (2, 2, 4), 2)]],
        [
            [("image", (1, 2, 2), 1), ("image", (1, 4, 4), 2)],
            [("video", (3, 2, 2), 3)],
            [("image", (1, 2, 4), 4)],
        ],
    ],
)
def test_minimax_groups_match_main(requests, real):
    params = [_request(items)[0] for items in requests]
    model = _CpuEncoder(real)
    with torch.no_grad():
        expected = _encode_like_main(model, params)
        oracle_inputs = model.vision_tower.calls[:]
        model.vision_tower.calls.clear()
        actual = encode_multimodal_by_groups(model.mm_encoder_groups, params)
    assert len(model.vision_tower.calls) == 1
    pixels, grids = model.vision_tower.calls[0]
    items = _main_extract_multimodal_items_in_request_order(params)
    grouped = sorted(zip(items, oracle_inputs), key=lambda pair: pair[0]["modality"] != "image")
    for actual_input, field in ((pixels, 0), (grids, 1)):
        expected_input = torch.cat([inputs[field] for _, inputs in grouped])
        assert torch.equal(actual_input, expected_input)
        assert actual_input.dtype == expected_input.dtype
        assert actual_input.device == expected_input.device
    if real:
        # Cross-request linear projections can change floating-point accumulation order.
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    else:
        assert torch.equal(actual, expected)


def test_minimax_group_restores_interleaved_prompt_order():
    items = [
        ("video", (2, 2, 2), 1),
        ("image", (1, 2, 4), 2),
        ("video", (1, 2, 4), 3),
    ]
    param, _ = _request(items, manifest=False)
    model = _CpuEncoder()
    expected = torch.cat([_encode_like_main(model, [_request([item])[0]]) for item in items])
    model.vision_tower.calls.clear()
    actual = encode_multimodal_by_groups(model.mm_encoder_groups, [param])
    assert len(model.vision_tower.calls) == 1
    assert torch.equal(actual, expected)
    assert param.item_metadata.item_refs == (("video", 0), ("image", 0), ("video", 1))


def test_minimax_metadata_fallback_and_video_field_alias():
    param, tokens = _request([("video", (2, 2, 4), 1), ("image", (1, 2, 2), 2)])
    param.multimodal_data.pop(MULTIMODAL_ENCODER_ITEM_METADATA_KEY)
    video = param.multimodal_data["video"]
    video["pixel_values"] = video.pop("pixel_values_videos")
    metadata = _processor().get_mm_item_metadata(tokens, param.multimodal_data)
    assert metadata.item_refs == (("video", 0), ("image", 0))
    assert metadata.output_embedding_lengths == (4, 1)
    assert video["pixel_values_videos"] is video["pixel_values"]
    param.multimodal_data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = metadata
    assert _processor().get_mm_item_metadata(tokens, param.multimodal_data) is metadata
    assert _processor().get_mm_item_metadata([7], {}) is None


def test_minimax_groups_split_pixel_dtypes():
    first, _ = _request([("image", (1, 2, 4), 1)])
    second, _ = _request([("video", (2, 2, 2), 2)])
    second.multimodal_data["video"]["pixel_values_videos"] = second.multimodal_data["video"][
        "pixel_values_videos"
    ].to(torch.float64)
    model = _CpuEncoder()
    expected = _encode_like_main(model, [first, second])
    model.vision_tower.calls.clear()
    actual = encode_multimodal_by_groups(model.mm_encoder_groups, [first, second])
    assert len(model.vision_tower.calls) == 2
    assert torch.equal(actual, expected)
