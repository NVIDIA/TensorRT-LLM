# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""qwen4_exp's multimodal splice runs the vision tower once per modality a request
carries and puts the rows back in prompt order; the tower itself runs a request's frames
in calls of whole frames under a row ceiling.

The tower is a stub whose rows name the item they came from, so the bookkeeping --
modality dispatch, item metadata, the chunk slice, the frame grouping and the order of
the concatenated calls -- runs on the CPU without the vision ops or an engine.
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
    MultimodalParams,
)

pytestmark = pytest.mark.cpu_only

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
    if order is not None:
        data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = MultimodalItemMetadata(item_refs=order)
    return MultimodalParams(multimodal_data=data, multimodal_runtime=runtime)


def _splice(tower, params, num_rows, padding=0):
    """The tower's rows spliced, as the prefill target's break does, over a prompt whose
    tokens 1..num_rows are the placeholders and which `padding` rows follow; returns the
    rows that landed there, one value per row, or None when the step carries no media."""
    kwargs = {
        "multimodal_params": [params],
        "mm_token_indices": torch.arange(1, 1 + num_rows, dtype=torch.int64),
    }
    vision = Shell._multimodal_embeds(_shell(tower), kwargs)
    if vision is None:
        return None
    total = num_rows + 2
    h = torch.full((total + padding, HIDDEN), -1.0)
    target = SimpleNamespace(_step=SimpleNamespace(vision=vision, num_tokens=total))
    M.PrefillTarget._splice(target, h)
    assert torch.count_nonzero(h[total:]) == 0, "padding rows must be zeroed"
    assert torch.all(h[0] == -1.0) and torch.all(h[total - 1] == -1.0)
    return h[1 : 1 + num_rows, 0].tolist()


def _item(modality, index):
    return modality, index


# Two videos of 4 and 8 rows, two images of 4 and 1.
VIDEOS = [(2, 2, 4), (2, 4, 4)]
IMAGES = [(1, 4, 4), (1, 2, 2)]
MIXED_ORDER = [_item("video", 0), _item("image", 0), _item("video", 1), _item("image", 1)]
MIXED_ROWS = [200.0] * 4 + [100.0] * 4 + [201.0] * 8 + [101.0]


def test_the_splice_fills_the_placeholders_and_zeroes_the_padding():
    """An image request's rows land on its placeholders in one tower call and the padding
    rows after the step are zeroed (`_splice` checks); a text step writes only the padding."""
    tower = _Tower()
    assert _splice(tower, _params(image=IMAGES), 5, padding=3) == [100.0] * 4 + [101.0]
    assert tower.calls == 1
    target = SimpleNamespace(_step=SimpleNamespace(vision=None, num_tokens=2))
    h = torch.full((4, HIDDEN), -1.0)
    M.PrefillTarget._splice(target, h)
    assert torch.all(h[:2] == -1.0) and torch.count_nonzero(h[2:]) == 0


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
    with pytest.raises(AssertionError, match="needs item metadata"):
        _splice(_Tower(), _params(image=IMAGES, video=VIDEOS), len(MIXED_ROWS))


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


def _frames(grid):
    """The item-by-item frame list a grid describes: one `(h, w)` per frame."""
    return [(h, w) for t, h, w in grid for _ in range(t)]


def _random_grid(gen):
    return [
        (int(torch.randint(1, 6, (1,), generator=gen)), 2 * h, 2 * w)
        for h, w in torch.randint(
            1, 5, (int(torch.randint(1, 6, (1,), generator=gen)), 2), generator=gen
        ).tolist()
    ]


def test_frames_are_grouped_whole_and_in_order():
    # 16-row frames, 32 rows to a call: one item's frames split across calls.
    assert M._frame_calls([(5, 4, 4)], 32) == [[(2, 4, 4)], [(2, 4, 4)], [(1, 4, 4)]]
    # A call takes the end of one item and the start of the next.
    grid = [(1, 8, 4), (3, 4, 4), (1, 2, 2)]
    assert M._frame_calls(grid, 48) == [[(1, 8, 4), (1, 4, 4)], [(2, 4, 4), (1, 2, 2)]]
    assert M._frame_calls(grid, 1 << 20) == [grid]


def test_a_call_is_filled_to_the_ceiling_and_never_past_it():
    gen = torch.Generator().manual_seed(0)
    for _ in range(200):
        grid = _random_grid(gen)
        cap = int(torch.randint(64, 300, (1,), generator=gen))
        calls = M._frame_calls(grid, cap)
        assert [f for call in calls for f in _frames(call)] == _frames(grid)
        sizes = [sum(t * h * w for t, h, w in call) for call in calls]
        assert all(0 < size <= cap for size in sizes)
        # Greedy: the next call's first frame would not have fit.
        for size, call in zip(sizes, calls[1:]):
            h, w = _frames(call)[0]
            assert size + h * w > cap


def test_one_frame_past_the_ceiling_is_refused_and_one_at_it_is_not():
    assert M._frame_calls([(2, 8, 8)], 64) == [[(1, 8, 8)], [(1, 8, 8)]]
    with pytest.raises(AssertionError, match="one frame is 8x10 = 80 patch rows"):
        M._frame_calls([(1, 4, 4), (3, 8, 10)], 64)


PIXEL_WIDTH = 4


def _tower(monkeypatch, cap):
    """A `VisionTower` whose per-call body is a stub: merged row `j` of a call carries the
    pixel value of the call's patch row `4 * j`, which `_pixels` set to the patch's
    index in the request."""
    monkeypatch.setattr(M, "_VISION_CALL_MAX_PATCHES", cap)
    tower = object.__new__(M.VisionTower)
    tower.core = SimpleNamespace(
        w={"embed": torch.zeros(1)}, v_pixel_width=PIXEL_WIDTH, v_merge=MERGE
    )
    tower.calls = []

    def run(pixel_values, grid, grid_thw):
        tower.calls.append((pixel_values, grid, grid_thw))
        return pixel_values[:: MERGE * MERGE, :1].expand(-1, HIDDEN).clone()

    tower._run = run
    return tower


def _pixels(grid):
    patches = sum(t * h * w for t, h, w in grid)
    return torch.arange(patches, dtype=torch.float32)[:, None].expand(-1, PIXEL_WIDTH).clone()


def test_the_tower_calls_whole_frames_and_concatenates_their_rows_in_order(monkeypatch):
    grid = [(1, 8, 4), (3, 4, 4), (2, 2, 6), (1, 2, 2)]
    tower = _tower(monkeypatch, 48)
    out = tower(_pixels(grid), torch.tensor(grid))
    patches = sum(t * h * w for t, h, w in grid)
    assert out[:, 0].tolist() == [float(4 * j) for j in range(patches // 4)]
    assert [call[1] for call in tower.calls] == M._frame_calls(grid, 48)
    start = 0
    for pixel_values, call, grid_thw in tower.calls:
        rows = sum(t * h * w for t, h, w in call)
        assert pixel_values[:, 0].tolist() == [float(i) for i in range(start, start + rows)]
        assert rows <= 48
        assert grid_thw.tolist() == [list(row) for row in call]
        assert grid_thw.dtype == torch.int64
        start += rows
    assert start == patches


def test_a_request_past_the_old_whole_request_ceiling_is_served(monkeypatch):
    # 33 temporal groups of 32x32 patches (66 frames of 512x512 pixels): 33792 rows, 1024
    # rows past what a single call may carry.
    ceiling = M._VISION_CALL_MAX_PATCHES
    grid = [(33, 32, 32)]
    tower = _tower(monkeypatch, ceiling)
    out = tower(_pixels(grid), torch.tensor(grid))
    assert [call[1] for call in tower.calls] == [[(32, 32, 32)], [(1, 32, 32)]]
    assert out.shape[0] == 33 * 32 * 32 // 4
    assert out[:, 0].tolist() == [float(4 * j) for j in range(out.shape[0])]


def test_a_media_request_needs_the_processors_position_ids():
    with pytest.raises(AssertionError, match="mrope_position_ids"):
        _splice(_Tower(), _params(video=VIDEOS, mrope=False), 12)
