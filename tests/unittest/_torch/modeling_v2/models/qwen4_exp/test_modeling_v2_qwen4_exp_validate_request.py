# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""qwen4_exp's admission check fails a bad media request before the forward would assert.

The check reads only the request's host-side payload, so these run on fake request
objects with meta pixel tensors: no GPU, vision ops or engine. Every refusal is a
`ValueError`, which the executor turns into that one request's error.
"""

from __future__ import annotations

import types
from types import SimpleNamespace

import pytest
import torch

__extra_import_path__ = ["../../.."]  # noqa: F841 -- repo's file-scoped import hook

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalItemMetadata,
)

pytestmark = pytest.mark.cpu_only

Shell = M.ModelingV2Qwen38FlashNextNvfp4Sm103Tp1

MERGE = 2
PATCH = 16
PIXEL_WIDTH = 4
TEXT_TOKENS = 5


def _shell():
    shell = SimpleNamespace(
        model=SimpleNamespace(v_merge=MERGE, v_patch=PATCH, v_pixel_width=PIXEL_WIDTH)
    )
    shell.validate_request = types.MethodType(Shell.validate_request, shell)
    shell._media_grid = types.MethodType(Shell._media_grid, shell)
    shell._check_item_order = Shell._check_item_order
    return shell


def _media(modality, grids):
    """A payload whose pixel tensor has the right shape and no storage."""
    patches = sum(t * h * w for t, h, w in grids)
    key = "pixel_values" if modality == "image" else "pixel_values_videos"
    grid_key = "image_grid_thw" if modality == "image" else "video_grid_thw"
    return {
        key: torch.empty(patches, PIXEL_WIDTH, device="meta"),
        grid_key: torch.tensor(grids),
    }


def _request(image=None, video=None, order=None, **overrides):
    """A request whose prompt is `TEXT_TOKENS` text tokens around the placeholders the
    media produce, with the position ids and placeholder cumsum the processor emits."""
    rows = sum(t * h * w for t, h, w in (image or []) + (video or [])) // (MERGE * MERGE)
    prompt_len = rows + TEXT_TOKENS
    data = {
        "mrope_config": {"mrope_position_ids": torch.zeros(3, 1, prompt_len, dtype=torch.int32)},
        "multimodal_embed_mask_cumsum": torch.cat(
            [torch.arange(1, rows + 1), torch.full((TEXT_TOKENS,), rows)]
        ),
    }
    if image is not None:
        data["image"] = _media("image", image)
    if video is not None:
        data["video"] = _media("video", video)
    if order is not None:
        data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = MultimodalItemMetadata(item_refs=order)
    fields = {"py_multimodal_data": data, "prompt_len": prompt_len}
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _admit(request):
    _shell().validate_request(request)


def _item(modality, index):
    return modality, index


IMAGES = [(1, 4, 4), (1, 2, 2)]
VIDEOS = [(2, 2, 4), (2, 4, 4)]
MIXED_ORDER = [_item("video", 0), _item("image", 0), _item("video", 1), _item("image", 1)]


def test_clean_requests_are_admitted():
    # A single-modality request needs no manifest.
    _admit(_request(image=IMAGES))
    _admit(_request(video=VIDEOS))
    _admit(_request(image=IMAGES, video=VIDEOS, order=MIXED_ORDER))


def test_a_request_without_media_costs_nothing():
    # Not even the other request attributes are read.
    _admit(SimpleNamespace(py_multimodal_data=None))
    _admit(SimpleNamespace(py_multimodal_data={}))
    # A warmup request carries position ids and no media.
    _admit(
        SimpleNamespace(
            py_multimodal_data={"mrope_config": {"mrope_position_ids": torch.zeros(3, 1, 8)}}
        )
    )
    # Precomputed embeddings arrive without pixels; the splice skips them.
    _admit(SimpleNamespace(py_multimodal_data={"multimodal_embedding": torch.zeros(4, 3)}))


def test_the_longest_admitted_frame_is_the_calls_ceiling(monkeypatch):
    monkeypatch.setattr(M, "_VISION_CALL_MAX_PATCHES", 64)
    _admit(_request(image=[(1, 8, 8)]))
    _admit(_request(video=[(3, 4, 16)]))
    with pytest.raises(ValueError, match="frames of 8x10 = 80 patches"):
        _admit(_request(image=[(1, 8, 8), (1, 8, 10)]))


def test_a_frame_past_the_ceiling_fails_the_request():
    # 3072x3072 pixels: a 192x192 grid, 36864 patches in one frame.
    request = _request(video=[(1, 192, 192)])
    with pytest.raises(ValueError, match=r"video 0 has frames of 192x192 = 36864 patches"):
        _admit(request)
    # The message names the pixels and the bound.
    with pytest.raises(ValueError, match=r"\(3072x3072 pixels\).*at most 32768 patches"):
        _admit(request)


def test_the_frame_bound_is_per_frame_not_per_item():
    # 33 temporal groups of 32x32 patches: 33792 rows in all, 1024 past one call.
    _admit(_request(video=[(33, 32, 32)]))
    # 4 frames of 128x128 patches (16384 each) are fine; 192x192 in the second item is not.
    _admit(_request(video=[(4, 128, 128)]))
    with pytest.raises(ValueError, match="video 1 has frames of 192x192"):
        _admit(_request(video=[(4, 128, 128), (1, 192, 192)]))


def test_a_malformed_grid_is_refused():
    request = _request(image=[(1, 4, 4)])
    request.py_multimodal_data["image"]["image_grid_thw"] = torch.tensor([(1, 4, 5)])
    with pytest.raises(ValueError, match="spatial_merge_size 2"):
        _admit(request)


def test_a_grid_that_is_not_one_row_per_item_is_refused():
    request = _request(image=IMAGES)
    request.py_multimodal_data["image"]["image_grid_thw"] = torch.tensor([1, 4, 4])
    with pytest.raises(ValueError, match="one \\(t, h, w\\) row per image"):
        _admit(request)


def test_pixels_that_disagree_with_the_grid_are_refused():
    request = _request(video=VIDEOS)
    video = request.py_multimodal_data["video"]
    video["pixel_values_videos"] = torch.empty(5, PIXEL_WIDTH, device="meta")
    with pytest.raises(ValueError, match="pixel_values_videos is .*describe 48 patch rows"):
        _admit(request)


def test_a_payload_without_its_pixels_or_grid_is_refused():
    request = _request(video=VIDEOS)
    del request.py_multimodal_data["video"]["video_grid_thw"]
    with pytest.raises(ValueError, match="this target reads 'pixel_values_videos'"):
        _admit(request)


def test_missing_or_mis_sized_position_ids_fail_the_request():
    request = _request(video=VIDEOS)
    request.py_multimodal_data["mrope_config"]["mrope_position_ids"] = torch.zeros(3, 1, 40)
    with pytest.raises(ValueError, match="one position per prompt token"):
        _admit(request)
    del request.py_multimodal_data["mrope_config"]
    with pytest.raises(ValueError, match="mrope_position_ids.*got None"):
        _admit(request)


def test_a_mixed_request_needs_its_manifest():
    with pytest.raises(ValueError, match="needs item metadata"):
        _admit(_request(image=IMAGES, video=VIDEOS))
    with pytest.raises(ValueError, match="needs item metadata"):
        _admit(_request(image=IMAGES, video=VIDEOS, order=[]))


def test_a_manifest_that_disagrees_with_the_payload_is_refused():
    with pytest.raises(ValueError, match="lists image items \\[0\\]; the payload carries 2"):
        _admit(_request(image=IMAGES, video=VIDEOS, order=MIXED_ORDER[:3]))
    with pytest.raises(ValueError, match="lists video items \\[0, 1, 0\\]"):
        _admit(_request(image=IMAGES, video=VIDEOS, order=MIXED_ORDER + [_item("video", 0)]))
    with pytest.raises(ValueError, match="is not an item of this request"):
        _admit(_request(image=IMAGES, video=VIDEOS, order=MIXED_ORDER + [_item("audio", 0)]))
    with pytest.raises(ValueError, match="is not an item of this request"):
        _admit(_request(image=IMAGES, video=VIDEOS, order=[("image",)]))


def test_the_placeholder_count_is_the_sum_over_both_modalities():
    request = _request(image=IMAGES, video=VIDEOS, order=MIXED_ORDER)
    request.py_multimodal_data["multimodal_embed_mask_cumsum"] = torch.arange(1, 14)
    with pytest.raises(ValueError, match="holds 13 image/video placeholder tokens.*17"):
        _admit(request)


def test_a_request_without_a_cumsum_skips_the_count():
    request = _request(video=VIDEOS)
    del request.py_multimodal_data["multimodal_embed_mask_cumsum"]
    _admit(request)
