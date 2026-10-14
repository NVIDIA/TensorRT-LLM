# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""qwen4_exp's multimodal splice runs the vision tower once per modality a request
carries and puts the rows back in prompt order.

The tower is a stub whose rows name the item they came from, so the bookkeeping --
modality dispatch, `mm_item_order`, the chunk slice, the per-call row ceiling -- runs on
the CPU without the vision ops or an engine.
"""

from __future__ import annotations

import types
from types import SimpleNamespace

import pytest
import torch

__extra_import_path__ = [".."]  # noqa: F841 -- repo's file-scoped import hook

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1 import (  # noqa: E501
    modeling as M,
)

Shell = M.ModelingV2Qwen38FlashNextNvfp4Sm103Tp1

HIDDEN = 3
MERGE = 2
IMAGE, VIDEO = 1, 2


class _Tower:
    """Stands in for `VisionTower`: every row of item `i` of modality `m` holds
    `100 * m + i`, so a spliced row says where it came from."""

    def __init__(self):
        self.calls = 0

    def __call__(self, pixel_values, grid_thw):
        self.calls += 1
        modality = int(pixel_values[0, 0])
        return torch.cat(
            [
                torch.full((t * h * w // (MERGE * MERGE), HIDDEN), float(100 * modality + item))
                for item, (t, h, w) in enumerate(grid_thw.tolist())
            ]
        )


def _shell(tower):
    shell = SimpleNamespace(
        model=SimpleNamespace(
            v_merge=MERGE, w={"embed": torch.zeros(16, HIDDEN)}, vision_tower=tower
        )
    )
    shell._run_tower = types.MethodType(Shell._run_tower, shell)
    shell._in_prompt_order = types.MethodType(Shell._in_prompt_order, shell)
    return shell


def _media(modality, grids):
    """One `multimodal_data` entry; `pixel_values` only carries the stub's modality."""
    patches = sum(t * h * w for t, h, w in grids)
    key = "pixel_values" if modality == IMAGE else "pixel_values_videos"
    grid_key = "image_grid_thw" if modality == IMAGE else "video_grid_thw"
    return {key: torch.full((patches, 1), float(modality)), grid_key: torch.tensor(grids)}


def _params(image=None, video=None, order=None, runtime=None, mrope=True):
    data = {}
    if mrope:
        data["mrope_config"] = {"mrope_position_ids": torch.zeros(3, 1, 8, dtype=torch.int32)}
    if image is not None:
        data["image"] = _media(IMAGE, image)
    if video is not None:
        data["video"] = _media(VIDEO, video)
    return SimpleNamespace(multimodal_data=data, mm_item_order=order, multimodal_runtime=runtime)


def _splice(tower, params, num_rows):
    """The splice over a prompt whose tokens 1..num_rows are the placeholders; returns
    the rows that landed there, one value per row, or None when nothing was spliced."""
    input_ids = torch.zeros(num_rows + 2, dtype=torch.int64)
    kwargs = {
        "multimodal_params": [params],
        "mm_token_indices": torch.arange(1, 1 + num_rows, dtype=torch.int64),
    }
    out = Shell._multimodal_embeds(_shell(tower), input_ids, kwargs)
    return None if out is None else out[1 : 1 + num_rows, 0].tolist()


def _item(modality, index):
    return {"modality": modality, "index": index}


# Two videos of 4 and 8 rows, two images of 4 and 1.
VIDEOS = [(2, 2, 4), (2, 4, 4)]
IMAGES = [(1, 4, 4), (1, 2, 2)]
MIXED_ORDER = [_item("video", 0), _item("image", 0), _item("video", 1), _item("image", 1)]
MIXED_ROWS = [200.0] * 4 + [100.0] * 4 + [201.0] * 8 + [101.0]


def test_image_request_is_unchanged():
    tower = _Tower()
    assert _splice(tower, _params(image=IMAGES), 5) == [100.0] * 4 + [101.0]
    assert tower.calls == 1


def test_video_rows_reach_the_placeholders_item_by_item():
    tower = _Tower()
    assert _splice(tower, _params(video=VIDEOS), 12) == [200.0] * 4 + [201.0] * 8
    assert tower.calls == 1


def test_a_mixed_request_is_reassembled_in_prompt_order():
    tower = _Tower()
    params = _params(image=IMAGES, video=VIDEOS, order=MIXED_ORDER)
    assert _splice(tower, params, len(MIXED_ROWS)) == MIXED_ROWS
    assert tower.calls == 2


def test_a_mixed_request_needs_its_manifest():
    with pytest.raises(AssertionError, match="needs mm_item_order"):
        _splice(_Tower(), _params(image=IMAGES, video=VIDEOS), len(MIXED_ROWS))


def test_a_manifest_that_disagrees_with_the_payload_is_refused():
    params = _params(image=IMAGES, video=VIDEOS, order=MIXED_ORDER[:3])
    with pytest.raises(AssertionError, match="mm_item_order lists image items"):
        _splice(_Tower(), params, len(MIXED_ROWS))


def test_a_chunk_takes_its_slice_of_the_prompt_order_rows():
    runtime = SimpleNamespace(
        num_cached_mm_tokens=3, num_mm_tokens_in_chunk=6, total_embeds_in_request=len(MIXED_ROWS)
    )
    params = _params(image=IMAGES, video=VIDEOS, order=MIXED_ORDER, runtime=runtime)
    assert _splice(_Tower(), params, 6) == MIXED_ROWS[3:9]


def test_a_chunk_without_placeholders_skips_the_tower():
    runtime = SimpleNamespace(
        num_cached_mm_tokens=len(MIXED_ROWS),
        num_mm_tokens_in_chunk=0,
        total_embeds_in_request=len(MIXED_ROWS),
    )
    tower = _Tower()
    params = _params(image=IMAGES, video=VIDEOS, order=MIXED_ORDER, runtime=runtime)
    assert _splice(tower, params, 0) is None
    assert tower.calls == 0


def test_the_video_call_ceiling_is_admitted_and_one_patch_past_it_is_refused():
    side = M._VIDEO_CALL_MAX_PATCHES // 2 // 128
    at_ceiling = [(2, 128, side)]
    assert sum(t * h * w for t, h, w in at_ceiling) == M._VIDEO_CALL_MAX_PATCHES
    assert len(_splice(_Tower(), _params(video=at_ceiling), M._VIDEO_CALL_MAX_PATCHES // 4)) == (
        M._VIDEO_CALL_MAX_PATCHES // 4
    )
    past = [(2, 128, side), (1, 1, 1)]
    with pytest.raises(AssertionError, match="certified up to"):
        _splice(_Tower(), _params(video=past), M._VIDEO_CALL_MAX_PATCHES // 4)


def test_a_media_request_needs_the_processors_position_ids():
    with pytest.raises(AssertionError, match="mrope_position_ids"):
        _splice(_Tower(), _params(video=VIDEOS, mrope=False), 12)
