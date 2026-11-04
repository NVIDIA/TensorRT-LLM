# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The FP8 unified target's vision path: the tower against HuggingFace, its two own entries, its
calls and angle table, the splice's bookkeeping, the shell's wiring and the admission check.

On one GPU (unmarked tests): the tower runs on the checkpoint's own `model.visual.*` weights,
read through the weight table's `vis_*` sources, against HuggingFace's `Qwen3_5VisionModel`
(this tower's geometry, no deepstack) loaded with the same tensors. The inputs are the
checkpoint's own image and video processors' output for two images and a four-frame video
(grids 1x28x42, 1x14x20 and 2x16x24). The reference is the HF model in fp32 with IEEE fp32
GEMMs, convolutions and attention; the band is the worst the same HF model in bf16 reaches
against it over four SDPA backends, so the tower is held to the error of a bf16 forward of
this model, not to a number. Two wrong towers (the rotary at twice its base, a video's frames
in one attention segment) are shown to fall outside that band.

The tower's entries, each over its certified cells (GPU):

- `vision_rope_2d_packed_qk`, the 2-D rotary, one Triton launch in place on the q|k prefix of
  the packed `[T, 3 * 16, 72]` qkv projection, over the caller's angle table (`_angle_table`:
  spatial-merge positions, fp64 angles rounded once to fp32): bitwise against the entry's
  `reference` on the same table and against the whole op from the grid (the NVFP4 tp1 target's
  `VisionRope2d.reference`, copied as `_whole_op_reference` since a target's test does not
  import a sibling's code), the v heads of every row untouched.
- `triton_pos_embed_interpolate`, the position table's bilinear resize: `reference` is the
  explicit four-corner sum in fp64, returned unrounded as fp32, so the band measures the op's
  own bf16 arithmetic. `_resize64` runs torch's `F.interpolate` in fp64 and shares no
  arithmetic with either: each cell holds the reference to it (which axis is which, corner
  alignment, merge row order), and its wrong variants (`align_corners=False`, raster rows,
  swapped table axes) must fail the band. The checkpoint's own table sits in the band too.

Each refusal of an entry's `is_valid` is first shown to be a silent wrong answer on the raw op,
then refused under `validating`: once per check the guard makes.

On the CPU (`cpu_only`): the tower is built over the weight table's own `vis_*`
declarations on the meta device, so what it reads is what `weights.py` declares. Its per-call
body is a stub where a test needs one, and so is the whole tower where the splice's
bookkeeping is under test: every row of a stub tower names the item it came from, so the
modality dispatch, item metadata, chunk slice, frame grouping and the order of concatenated
calls run with no engine. The shell's forward is driven up to `begin_step`, to show the rows
it hands the prefill target's splice; its `validate_request`, over the tower built from the
table, admits text and the media the tower serves and is driven through each refusal.
"""

from __future__ import annotations

import contextlib
import functools
import json
import zlib
from types import SimpleNamespace
from typing import List, Sequence, Tuple

import numpy as np
import pytest
import safetensors
import torch
import torch.nn.functional as F
from utils.llm_data import llm_models_root

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified import (  # noqa: E501
    vision as V,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified import (  # noqa: E501
    weights as W,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.attention.vision_rope_2d_packed_qk import (  # noqa: E501
    HEAD_DIM,
    NUM_HEADS,
    vision_rope_2d_packed_qk,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_fp8__sm_103__unified.ops.vision.triton_pos_embed_interpolate import (  # noqa: E501
    triton_pos_embed_interpolate,
)
from tensorrt_llm._torch.configs import Qwen4ExpVisionConfig
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalItemMetadata,
    MultimodalParams,
)

__extra_import_path__ = ["../../.."]
from _validating import (  # noqa: E402,F401 — path above; fixture by name
    ULP,
    assert_within_ulp,
    certifying,
    validating,
)

CHECKPOINT = "Qwen3.8-Flash-Next-FP8"

Shell = M.ModelingV2Qwen38FlashNextFp8Sm103Unified

# Qwen3.8-Flash-Next's vision config.
_VISION = dict(
    deepstack_visual_indexes=[],
    depth=27,
    dtype="bfloat16",
    hidden_act="gelu_pytorch_tanh",
    hidden_size=1152,
    in_channels=3,
    intermediate_size=4304,
    num_heads=16,
    num_position_embeddings=2304,
    out_hidden_size=2560,
    patch_size=16,
    spatial_merge_size=2,
    temporal_patch_size=2,
)
HIDDEN_OUT = 2560

_needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _dims(vcfg) -> SimpleNamespace:
    """The vision widths of `weights.rank_dims` (the tower is replicated: every rank's)."""
    return SimpleNamespace(
        hidden=HIDDEN_OUT,
        dtype=torch.bfloat16,
        v_depth=vcfg.depth,
        v_hidden=vcfg.hidden_size,
        v_inter=vcfg.intermediate_size,
        v_channels=vcfg.in_channels,
        v_tpatch=vcfg.temporal_patch_size,
        v_patch=vcfg.patch_size,
        v_pos_rows=vcfg.num_position_embeddings,
        v_merge_dim=vcfg.hidden_size * vcfg.spatial_merge_size**2,
    )


def _vision_entries():
    return [entry for entry in W.MODEL_WEIGHTS.WEIGHTS if entry.name.startswith("vis_")]


def _tower_over_the_weight_table() -> V.VisionTower:
    """The tower over every `vis_*` weight the table declares, at the checkpoint's shapes,
    on meta; `embed` stands in for the activation dtype the tower reads off it."""
    vcfg = Qwen4ExpVisionConfig(**_VISION)
    d = _dims(vcfg)
    w = {"embed": torch.empty(8, HIDDEN_OUT, dtype=torch.bfloat16, device="meta")}
    for entry in _vision_entries():
        for key, _, _ in W.MODEL_WEIGHTS._targets(entry, d):
            w[key] = torch.empty(*entry.shape(d), dtype=entry.dtype or d.dtype, device="meta")
    return V.VisionTower(w, vcfg)


@pytest.mark.cpu_only
def test_the_tower_reads_the_weight_tables_vision_rows_at_the_ops_layouts():
    """Every view the tower builds exists in the table, and each GEMM operand is the
    column-major `[K, N]` view `cublas_mm` reads: the packed qkv row is `3 * 1152` wide and
    the merger reads `2 x 2` patches of 1152."""
    tower = _tower_over_the_weight_table()
    assert (tower.depth, tower.heads, tower.head_dim, tower.merge) == (27, 16, 72, 2)
    assert (tower.pixel_width, tower.merge_dim, tower.grid_side) == (1536, 4608, 48)
    assert tower.dtype == torch.bfloat16
    views = tower._w
    assert tuple(views.patch_w.shape) == (1152, 3, 2, 16, 16)
    assert tuple(views.pos.shape) == (2304, 1152)
    assert len(views.blocks) == 27
    for block in views.blocks:
        for gemm, (k, n) in (
            (block.qkv, (1152, 3456)),
            (block.proj, (1152, 1152)),
            (block.fc1, (1152, 4304)),
            (block.fc2, (4304, 1152)),
        ):
            assert tuple(gemm.shape) == (k, n) and gemm.stride() == (1, k)
        assert tuple(block.qkv_b.shape) == (3456,) and tuple(block.fc1_b.shape) == (4304,)
    assert tuple(views.m_fc1.shape) == (4608, 4608) and views.m_fc1.stride() == (1, 4608)
    assert tuple(views.m_fc2.shape) == (4608, HIDDEN_OUT) and views.m_fc2.stride() == (1, 4608)


@pytest.mark.cpu_only
def test_the_tower_reads_every_vision_entry_of_the_table():
    """The tower's views are exactly the table's `vis_*` parameters: none it reads is
    missing from the table, and none the table loads goes unread."""
    vcfg = Qwen4ExpVisionConfig(**_VISION)
    d = _dims(vcfg)
    declared = {
        key for entry in _vision_entries() for key, _, _ in W.MODEL_WEIGHTS._targets(entry, d)
    }
    assert len(declared) == 3 + 12 * vcfg.depth + 6

    class _Recording(dict):
        def __init__(self, *args):
            super().__init__(*args)
            self.read = set()

        def __getitem__(self, key):
            self.read.add(key)
            return super().__getitem__(key)

    w = _Recording(
        {key: torch.empty(1, 1, device="meta") for key in declared}
        | {"embed": torch.empty(1, 1, dtype=torch.bfloat16, device="meta")}
    )
    V.VisionTower(w, vcfg)
    assert w.read - {"embed"} == declared


# --------------------------------------------------------------- angle table ---

MERGE = 2
THETA = 10000.0
HALF = HEAD_DIM // 2


def _num_tokens(grid) -> int:
    return sum(t * h * w for t, h, w in grid)


def _spatial_merge_positions(grid: Sequence[Sequence[int]], merge: int) -> List[Tuple[int, int]]:
    """Per-token `(row, col)` by four nested loops, in spatial-merge order: the template the
    rotary entry's cells build their tables from."""
    positions: List[Tuple[int, int]] = []
    for frames, height, width in grid:
        frame = [
            (block_row * merge + r, block_col * merge + c)
            for block_row in range(height // merge)
            for block_col in range(width // merge)
            for r in range(merge)
            for c in range(merge)
        ]
        positions.extend(frame * frames)
    return positions


def _angle_table(grid_thw: torch.Tensor, merge: int, theta: float, head_dim: int, device="cuda"):
    """The caller's glue: per-token fp32 `cos` / `sin` `[T, head_dim // 2]`, height angles
    then width, from fp64 angles rounded once to fp32."""
    rope_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float64) / rope_dim))
    pos = torch.tensor(_spatial_merge_positions(grid_thw.tolist(), merge), dtype=torch.float64)
    angles = (pos[:, :, None] * inv_freq).flatten(1).to(device=device, dtype=torch.float32)
    return angles.cos().contiguous(), angles.sin().contiguous()


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "grid", [[(1, 62, 90)], [(3, 8, 8), (1, 16, 24)], [(2, 4, 6)]], ids=["62x90", "mixed", "t2"]
)
def test_the_angle_table_is_the_rotary_entrys_certified_table(grid):
    """`_rope_tables` builds, bit for bit, the table `vision_rope_2d_packed_qk`'s cells are
    certified with (`_angle_table`): positions in spatial-merge order, fp64 angles rounded once
    to fp32, height angles then width angles, `[rows, 36]` contiguous."""
    assert V._merge_order_positions(grid, MERGE).tolist() == [
        list(p) for p in _spatial_merge_positions(grid, MERGE)
    ]
    assert V.VISION_ROPE_THETA == THETA and HEAD_DIM == 72
    cos, sin = V._rope_tables(grid, MERGE, V.VISION_ROPE_THETA, HEAD_DIM, "cpu")
    want = _angle_table(torch.tensor(grid), MERGE, THETA, HEAD_DIM, device="cpu")
    for got, table in zip((cos, sin), want):
        assert got.dtype == torch.float32 and got.is_contiguous()
        assert tuple(got.shape) == (_num_tokens(grid), HALF)
        assert torch.equal(got, table)


# ---------------------------------------------------------------- tower calls ---

HIDDEN = 3


PIXEL_WIDTH = 4


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


@pytest.mark.cpu_only
def test_frames_are_grouped_whole_and_in_order():
    # 16-row frames, 32 rows to a call: one item's frames split across calls.
    assert V._frame_calls([(5, 4, 4)], 32) == [[(2, 4, 4)], [(2, 4, 4)], [(1, 4, 4)]]
    # A call takes the end of one item and the start of the next.
    grid = [(1, 8, 4), (3, 4, 4), (1, 2, 2)]
    assert V._frame_calls(grid, 48) == [[(1, 8, 4), (1, 4, 4)], [(2, 4, 4), (1, 2, 2)]]
    assert V._frame_calls(grid, 1 << 20) == [grid]


@pytest.mark.cpu_only
def test_a_call_is_filled_to_the_ceiling_and_never_past_it():
    gen = torch.Generator().manual_seed(0)
    for _ in range(200):
        grid = _random_grid(gen)
        cap = int(torch.randint(64, 300, (1,), generator=gen))
        calls = V._frame_calls(grid, cap)
        assert [f for call in calls for f in _frames(call)] == _frames(grid)
        sizes = [sum(t * h * w for t, h, w in call) for call in calls]
        assert all(0 < size <= cap for size in sizes)
        # Greedy: the next call's first frame would not have fit.
        for size, call in zip(sizes, calls[1:]):
            h, w = _frames(call)[0]
            assert size + h * w > cap


@pytest.mark.cpu_only
def test_one_frame_past_the_ceiling_is_refused_and_one_at_it_is_not():
    assert V._frame_calls([(2, 8, 8)], 64) == [[(1, 8, 8)], [(1, 8, 8)]]
    with pytest.raises(AssertionError, match="one frame is 8x10 = 80 patch rows"):
        V._frame_calls([(1, 4, 4), (3, 8, 10)], 64)


def _stub_tower(monkeypatch, cap):
    """A `VisionTower` whose per-call body is a stub: merged row `j` of a call carries the
    pixel value of the call's patch row `4 * j`, which `_pixels` set to the patch's index in
    the request."""
    monkeypatch.setattr(V, "VISION_CALL_MAX_PATCHES", cap)
    tower = object.__new__(V.VisionTower)
    tower.dtype = torch.float32
    tower.pixel_width = PIXEL_WIDTH
    tower.merge = MERGE
    tower.calls = []

    def run(pixel_values, grid):
        tower.calls.append((pixel_values, grid))
        return pixel_values[:: MERGE * MERGE, :1].expand(-1, HIDDEN).clone()

    tower._run = run
    return tower


def _pixels(grid):
    patches = sum(t * h * w for t, h, w in grid)
    return torch.arange(patches, dtype=torch.float32)[:, None].expand(-1, PIXEL_WIDTH).clone()


@pytest.mark.cpu_only
def test_the_tower_calls_whole_frames_and_concatenates_their_rows_in_order(monkeypatch):
    grid = [(1, 8, 4), (3, 4, 4), (2, 2, 6), (1, 2, 2)]
    tower = _stub_tower(monkeypatch, 48)
    out = tower(_pixels(grid), torch.tensor(grid))
    patches = sum(t * h * w for t, h, w in grid)
    assert out[:, 0].tolist() == [float(4 * j) for j in range(patches // 4)]
    assert [call[1] for call in tower.calls] == V._frame_calls(grid, 48)
    start = 0
    for pixel_values, call in tower.calls:
        rows = sum(t * h * w for t, h, w in call)
        assert pixel_values[:, 0].tolist() == [float(i) for i in range(start, start + rows)]
        assert rows <= 48
        start += rows
    assert start == patches


@pytest.mark.cpu_only
def test_a_request_past_one_calls_ceiling_is_served_in_two(monkeypatch):
    # 33 temporal groups of 32x32 patches (66 frames of 512x512 pixels): 33792 rows, 1024
    # rows past what a single call may carry.
    ceiling = V.VISION_CALL_MAX_PATCHES
    grid = [(33, 32, 32)]
    tower = _stub_tower(monkeypatch, ceiling)
    out = tower(_pixels(grid), torch.tensor(grid))
    assert [call[1] for call in tower.calls] == [[(32, 32, 32)], [(1, 32, 32)]]
    assert out.shape[0] == 33 * 32 * 32 // 4
    assert out[:, 0].tolist() == [float(4 * j) for j in range(out.shape[0])]


@pytest.mark.cpu_only
def test_the_tower_refuses_pixels_its_grid_does_not_describe(monkeypatch):
    tower = _stub_tower(monkeypatch, 64)
    with pytest.raises(AssertionError, match="describes 16 patch rows"):
        tower(_pixels([(1, 4, 4)])[:12], torch.tensor([(1, 4, 4)]))
    with pytest.raises(AssertionError, match="divide by spatial_merge_size 2"):
        tower(_pixels([(1, 4, 3)]), torch.tensor([(1, 4, 3)]))
    with pytest.raises(AssertionError, match="activation dtype"):
        tower(_pixels([(1, 4, 4)]).to(torch.bfloat16), torch.tensor([(1, 4, 4)]))


# ------------------------------------------------------------------- splice ---

IMAGE, VIDEO = 1, 2


class _Tower:
    """Stands in for `VisionTower`: every row of item `i` of modality `m` holds `100 * m +
    i`, so a spliced row says where it came from."""

    merge = MERGE

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
    """The tower's rows spliced, as this target's prefill break does, over a prompt whose
    tokens 1..num_rows are the placeholders and which `padding` rows follow; returns the rows
    that landed there, one value per row, or None when the step carries no media."""
    kwargs = {
        "multimodal_params": [params],
        "mm_token_indices": torch.arange(1, 1 + num_rows, dtype=torch.int64),
    }
    vision = V.multimodal_embeds(tower, kwargs)
    if vision is None:
        return None
    total = num_rows + 2
    h = torch.full((total + padding, HIDDEN), -1.0)
    target = SimpleNamespace(_step=SimpleNamespace(vision=vision, num_tokens=total))
    M.PrefillTarget._splice(target, h)
    assert torch.count_nonzero(h[total:]) == 0, "padding rows must be zeroed"
    assert torch.all(h[0] == -1.0) and torch.all(h[total - 1] == -1.0)
    return h[1 : 1 + num_rows, 0].tolist()


# Two videos of 4 and 8 rows, two images of 4 and 1.
VIDEOS = [(2, 2, 4), (2, 4, 4)]
IMAGES = [(1, 4, 4), (1, 2, 2)]
MIXED_ORDER = [("video", 0), ("image", 0), ("video", 1), ("image", 1)]
MIXED_ROWS = [200.0] * 4 + [100.0] * 4 + [201.0] * 8 + [101.0]


@pytest.mark.cpu_only
def test_a_text_step_runs_no_tower():
    assert V.multimodal_embeds(None, {}) is None
    assert V.multimodal_embeds(None, {"multimodal_params": []}) is None
    # Position ids alone (a text prompt through a multimodal processor) are not media.
    assert V.multimodal_embeds(None, {"multimodal_params": [_params()]}) is None


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
def test_video_rows_reach_the_placeholders_item_by_item():
    tower = _Tower()
    assert _splice(tower, _params(video=VIDEOS), 12) == [200.0] * 4 + [201.0] * 8
    assert tower.calls == 1


@pytest.mark.cpu_only
def test_a_mixed_request_is_reassembled_in_prompt_order():
    tower = _Tower()
    params = _params(image=IMAGES, video=VIDEOS, order=MIXED_ORDER)
    assert _splice(tower, params, len(MIXED_ROWS)) == MIXED_ROWS
    assert tower.calls == 2


@pytest.mark.cpu_only
def test_a_mixed_request_needs_its_manifest():
    with pytest.raises(AssertionError, match="needs item metadata"):
        _splice(_Tower(), _params(image=IMAGES, video=VIDEOS), len(MIXED_ROWS))


@pytest.mark.cpu_only
def test_a_chunk_takes_its_slice_of_the_prompt_order_rows():
    runtime = SimpleNamespace(
        num_cached_mm_tokens=3, num_mm_tokens_in_chunk=6, total_embeds_in_request=len(MIXED_ROWS)
    )
    params = _params(image=IMAGES, video=VIDEOS, order=MIXED_ORDER, runtime=runtime)
    assert _splice(_Tower(), params, 6) == MIXED_ROWS[3:9]


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
def test_the_executors_row_count_must_match_the_towers():
    runtime = SimpleNamespace(
        num_cached_mm_tokens=0, num_mm_tokens_in_chunk=4, total_embeds_in_request=20
    )
    params = _params(image=IMAGES, video=VIDEOS, order=MIXED_ORDER, runtime=runtime)
    with pytest.raises(AssertionError, match="counted 20 placeholder rows"):
        _splice(_Tower(), params, 4)


@pytest.mark.cpu_only
def test_a_media_request_needs_the_processors_position_ids():
    with pytest.raises(AssertionError, match="mrope_position_ids"):
        _splice(_Tower(), _params(video=VIDEOS, mrope=False), 12)


@pytest.mark.cpu_only
def test_the_placeholder_rows_must_match_the_towers_rows():
    kwargs = {
        "multimodal_params": [_params(image=IMAGES)],
        "mm_token_indices": torch.arange(4, dtype=torch.int64),
    }
    with pytest.raises(AssertionError, match="4 placeholder rows and the tower produced 5"):
        V.multimodal_embeds(_Tower(), kwargs)


@pytest.mark.cpu_only
def test_the_grids_stay_on_the_host():
    """The executor is to move only the pixels and the position ids (the shell's
    `multimodal_data_device_paths`); the grids the tower and the admission check read on the
    host stay there, and the tower refuses a grid on the device."""
    assert set(V.DEVICE_PATHS) == {
        "image.pixel_values",
        "video.pixel_values_videos",
        "mrope_config.mrope_position_ids",
        "mrope_config.mrope_position_deltas",
    }
    assert not any("grid_thw" in path for path in V.DEVICE_PATHS)
    assert {modality: keys[0] for modality, keys in V.MEDIA_KEYS.items()} == {
        "image": "pixel_values",
        "video": "pixel_values_videos",
    }
    assert Shell.multimodal_data_device_paths.fget(None) == list(V.DEVICE_PATHS)


# ------------------------------------------------------------ the shell's wiring ---

# The composite config's placeholder ids (`image_token_id`, `video_token_id`).
IMAGE_TOKEN_ID, VIDEO_TOKEN_ID = 248056, 248057


class _StepBegun(Exception):
    """Raised by the stub core's `begin_step`, so the shell's forward stops there."""


def _forward_to_begin_step(tower, kwargs):
    """The `vision` argument the shell's forward hands the core's `begin_step` for a step
    whose `**kwargs` are `kwargs`, with `tower` as the shell's vision tower."""
    seen = {}

    def begin_step(attn_metadata, input_ids, position_ids, **step):
        seen.update(step)
        raise _StepBegun

    shell = SimpleNamespace(model=SimpleNamespace(begin_step=begin_step), vision_tower=tower)
    with pytest.raises(_StepBegun):
        Shell.forward(shell, None, torch.zeros(1, dtype=torch.int32), None, **kwargs)
    return seen["vision"]


@pytest.mark.cpu_only
def test_the_shell_hands_a_text_step_no_vision_rows():
    """A text step reaches the core as it did before the tower: `vision=None`, and the tower
    is never called, whether or not the executor ships multimodal params."""
    tower = _Tower()
    assert _forward_to_begin_step(tower, {}) is None
    assert _forward_to_begin_step(tower, {"multimodal_params": [_params()]}) is None
    assert tower.calls == 0


@pytest.mark.cpu_only
def test_the_shell_hands_the_splice_the_rows_in_prompt_order():
    """The shell's forward runs the tower over the step's media and hands `begin_step` the
    rows with the placeholder indices the executor computed; the prefill target's splice
    writes them over the placeholders in prompt order."""
    tower = _Tower()
    num_rows = len(MIXED_ROWS)
    kwargs = {
        "multimodal_params": [_params(image=IMAGES, video=VIDEOS, order=MIXED_ORDER)],
        "mm_token_indices": torch.arange(1, 1 + num_rows, dtype=torch.int64),
        "resource_manager": object(),
    }
    vision = _forward_to_begin_step(tower, kwargs)
    assert tower.calls == 2
    rows, mm_token_indices = vision
    assert mm_token_indices is kwargs["mm_token_indices"]
    total = num_rows + 2
    h = torch.full((total + 3, HIDDEN), -1.0)
    target = SimpleNamespace(_step=SimpleNamespace(vision=vision, num_tokens=total))
    M.PrefillTarget._splice(target, h)
    assert h[1 : 1 + num_rows, 0].tolist() == MIXED_ROWS
    assert torch.all(h[0] == -1.0) and torch.all(h[total - 1] == -1.0)
    assert torch.count_nonzero(h[total:]) == 0


@pytest.mark.cpu_only
def test_the_shell_names_the_placeholder_ids_once():
    """`multimodal_token_ids` is the image and video placeholder ids as one int32 tensor,
    built on first read and reused after (the shell is constructed under meta init)."""
    shell = SimpleNamespace(
        _mm_token_ids_value=(IMAGE_TOKEN_ID, VIDEO_TOKEN_ID), _mm_token_ids_cache=None
    )
    ids = Shell.multimodal_token_ids.fget(shell)
    assert ids.dtype == torch.int32 and ids.device.type == "cpu"
    assert ids.tolist() == [IMAGE_TOKEN_ID, VIDEO_TOKEN_ID]
    assert Shell.multimodal_token_ids.fget(shell) is ids


# ---------------------------------------------------------------- admission ---

TEXT_TOKENS = 5


@functools.cache
def _assembled_tower() -> V.VisionTower:
    """The tower the shell's `post_load_weights` builds, over the weight table's `vis_*`
    declarations on meta: admission reads its `merge`, `patch` and `pixel_width`."""
    return _tower_over_the_weight_table()


def _request_media(modality, grids):
    """A payload whose pixel tensor has the assembled tower's patch width and no storage."""
    patches = sum(t * h * w for t, h, w in grids)
    key = "pixel_values" if modality == "image" else "pixel_values_videos"
    grid_key = "image_grid_thw" if modality == "image" else "video_grid_thw"
    return {
        key: torch.empty(patches, _assembled_tower().pixel_width, device="meta"),
        grid_key: torch.tensor(grids),
    }


def _request(image=None, video=None, order=None, **overrides):
    """A request whose prompt is `TEXT_TOKENS` text tokens around the placeholders the media
    produce, with the position ids and placeholder cumsum the processor emits."""
    rows = sum(t * h * w for t, h, w in (image or []) + (video or [])) // (MERGE * MERGE)
    prompt_len = rows + TEXT_TOKENS
    data = {
        "mrope_config": {"mrope_position_ids": torch.zeros(3, 1, prompt_len, dtype=torch.int32)},
        "multimodal_embed_mask_cumsum": torch.cat(
            [torch.arange(1, rows + 1), torch.full((TEXT_TOKENS,), rows)]
        ),
    }
    if image is not None:
        data["image"] = _request_media("image", image)
    if video is not None:
        data["video"] = _request_media("video", video)
    if order is not None:
        data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = MultimodalItemMetadata(item_refs=order)
    fields = {"py_multimodal_data": data, "prompt_len": prompt_len}
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _admit(request, *, without_tower=False):
    """The shell's admission check (`vision.validate_request` over the shell's tower): the
    assembled tower (`_assembled_tower`), or none at all with `without_tower`, for a request
    whose check must not read it."""
    tower = None if without_tower else _assembled_tower()
    Shell.validate_request(SimpleNamespace(vision_tower=tower), request)


@pytest.mark.cpu_only
def test_clean_requests_are_admitted():
    """Images, videos and both in one request, at the tower's patch width, are served."""
    # A single-modality request needs no manifest.
    _admit(_request(image=IMAGES))
    _admit(_request(video=VIDEOS))
    _admit(_request(image=IMAGES, video=VIDEOS, order=MIXED_ORDER))


@pytest.mark.cpu_only
def test_a_request_without_media_costs_nothing():
    """A text request is admitted without reading the tower."""
    # Not even the other request attributes, or the tower, are read.
    _admit(SimpleNamespace(py_multimodal_data=None), without_tower=True)
    _admit(SimpleNamespace(py_multimodal_data={}), without_tower=True)
    # Position ids without media (a text request's empty mrope config, a warmup request's
    # position ids) are no media either.
    _admit(SimpleNamespace(py_multimodal_data={"mrope_config": {}}), without_tower=True)
    _admit(
        SimpleNamespace(
            py_multimodal_data={"mrope_config": {"mrope_position_ids": torch.zeros(3, 1, 8)}}
        ),
        without_tower=True,
    )


@pytest.mark.cpu_only
def test_a_frame_past_the_ceiling_fails_the_request():
    # 2048x4096 pixels: a 128x256 grid, 32768 patches in one frame -- the call's ceiling, served.
    _admit(_request(image=[(1, 128, 256)]))
    # 3072x3072 pixels: a 192x192 grid, 36864 patches in one frame.
    request = _request(video=[(1, 192, 192)])
    with pytest.raises(ValueError, match=r"video 0 has frames of 192x192 = 36864 patches"):
        _admit(request)
    # The message names the pixels and the bound.
    with pytest.raises(ValueError, match=r"\(3072x3072 pixels\).*at most 32768 patches"):
        _admit(request)


@pytest.mark.cpu_only
def test_the_frame_bound_is_per_frame_not_per_item():
    # 33 temporal groups of 32x32 patches: 33792 rows in all, 1024 past one call.
    _admit(_request(video=[(33, 32, 32)]))
    with pytest.raises(ValueError, match="video 1 has frames of 192x192"):
        _admit(_request(video=[(4, 128, 128), (1, 192, 192)]))


@pytest.mark.cpu_only
def test_a_malformed_grid_is_refused():
    request = _request(image=[(1, 4, 4)])
    request.py_multimodal_data["image"]["image_grid_thw"] = torch.tensor([(1, 4, 5)])
    with pytest.raises(ValueError, match="spatial_merge_size 2"):
        _admit(request)
    request.py_multimodal_data["image"]["image_grid_thw"] = torch.tensor([(0, 4, 4)])
    with pytest.raises(ValueError, match="every entry must be positive"):
        _admit(request)
    request.py_multimodal_data["image"]["image_grid_thw"] = torch.tensor([1, 4, 4])
    with pytest.raises(ValueError, match="one \\(t, h, w\\) row per image"):
        _admit(request)
    request.py_multimodal_data["image"]["image_grid_thw"] = torch.zeros(0, 3, dtype=torch.int64)
    with pytest.raises(ValueError, match="one \\(t, h, w\\) row per image"):
        _admit(request)


@pytest.mark.cpu_only
def test_pixels_that_disagree_with_the_grid_are_refused():
    request = _request(video=VIDEOS)
    video = request.py_multimodal_data["video"]
    width = _assembled_tower().pixel_width
    video["pixel_values_videos"] = torch.empty(5, width, device="meta")
    with pytest.raises(ValueError, match="pixel_values_videos is .*describe 48 patch rows"):
        _admit(request)
    video["pixel_values_videos"] = torch.empty(48, width + 1, device="meta")
    with pytest.raises(ValueError, match=f"48 patch rows of {width}"):
        _admit(request)
    del video["video_grid_thw"]
    with pytest.raises(ValueError, match="this target reads 'pixel_values_videos'"):
        _admit(request)


@pytest.mark.cpu_only
def test_missing_or_mis_sized_position_ids_fail_the_request():
    request = _request(video=VIDEOS)
    request.py_multimodal_data["mrope_config"]["mrope_position_ids"] = torch.zeros(3, 1, 40)
    with pytest.raises(ValueError, match="one position per prompt token"):
        _admit(request)
    request.py_multimodal_data["mrope_config"]["mrope_position_ids"] = torch.zeros(1, 17)
    with pytest.raises(ValueError, match="got \\(1, 17\\)"):
        _admit(request)
    del request.py_multimodal_data["mrope_config"]
    with pytest.raises(ValueError, match="mrope_position_ids.*got None"):
        _admit(request)


@pytest.mark.cpu_only
def test_a_mixed_request_lists_its_items():
    with pytest.raises(ValueError, match="needs item metadata"):
        _admit(_request(image=IMAGES, video=VIDEOS))
    with pytest.raises(ValueError, match="lists image items \\[0\\]; the payload carries 2"):
        _admit(_request(image=IMAGES, video=VIDEOS, order=MIXED_ORDER[:3]))
    with pytest.raises(ValueError, match="is not an item of this request"):
        _admit(_request(image=IMAGES, video=VIDEOS, order=MIXED_ORDER + [("audio", 0)]))


@pytest.mark.cpu_only
def test_the_placeholder_count_is_the_sum_over_both_modalities():
    request = _request(image=IMAGES, video=VIDEOS, order=MIXED_ORDER)
    request.py_multimodal_data["multimodal_embed_mask_cumsum"] = torch.arange(1, 14)
    with pytest.raises(ValueError, match="holds 13 image/video placeholder tokens.*17"):
        _admit(request)
    del request.py_multimodal_data["multimodal_embed_mask_cumsum"]
    _admit(request)


# ------------------------------------------------- vision_rope_2d_packed_qk ---


def _whole_op_reference(q, k, grid_thw, spatial_merge_size, theta=THETA):
    """`VisionRope2d.reference`, verbatim: the whole op from the grid, as a fresh
    `(q_out, k_out)`."""
    rope_dim = q.shape[-1] // 2
    axis_dim = rope_dim // 2
    positions = _spatial_merge_positions(grid_thw.tolist(), spatial_merge_size)
    inverse = torch.tensor(
        [theta ** (-(2.0 * j) / rope_dim) for j in range(axis_dim)], dtype=torch.float64
    )
    pos = torch.tensor(positions, dtype=torch.float64).reshape(-1, 2)
    angles = (pos[:, :, None] * inverse).flatten(1).to(device=q.device, dtype=torch.float32)
    rotor = torch.polar(torch.ones_like(angles), angles).unsqueeze(-2)

    def rotate(x: torch.Tensor) -> torch.Tensor:
        fp32 = x.float()
        rotated = torch.complex(fp32[..., :rope_dim], fp32[..., rope_dim:]) * rotor
        return torch.cat((rotated.real, rotated.imag), dim=-1).to(x.dtype)

    return rotate(q), rotate(k)


def _packed_qkv(tokens: int, dtype: torch.dtype, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(tokens, 3 * NUM_HEADS, HEAD_DIM, generator=g, device="cuda").to(dtype)


def _qk(qkv: torch.Tensor) -> torch.Tensor:
    qk = qkv[:, : 2 * NUM_HEADS]
    assert qk.stride() == (3 * NUM_HEADS * HEAD_DIM, HEAD_DIM, 1) and not qk.is_contiguous()
    return qk


def _distribution(got: torch.Tensor, want: torch.Tensor) -> Tuple[float, float, float]:
    """(cosine, relative L2, p99.9 absolute error) of `got` against `want`, flattened."""
    g, w = got.float().reshape(-1), want.float().reshape(-1)
    cos = torch.nn.functional.cosine_similarity(g, w, dim=0).item()
    rel_l2 = ((g - w).norm() / w.norm().clamp_min(1e-30)).item()
    err = (g - w).abs()
    # torch.quantile caps its input size; a strided sample of 2^24 is plenty for p99.9.
    if err.numel() > 2**24:
        err = err[:: -(-err.numel() // 2**24)]
    return cos, rel_l2, torch.quantile(err, 0.999).item()


#: The whole-op reference's distribution gates: both sides round the same fp32 rotation to bf16
#: once, so they differ by at most one bf16 ulp per element, at most 2^-7 of its value: the
#: relative L2 is bounded by that, and 1 - cosine by half its square.
_REL_L2_BOUND = 2.0**-7
_COS_BOUND = 1.0 - 2.0**-15


def _mismatch(got: torch.Tensor, want: torch.Tensor) -> Tuple[int, float]:
    """(elements differing in their bits, largest absolute difference)."""
    differing = (got.reshape(-1).view(torch.int16) != want.reshape(-1).view(torch.int16)).sum()
    return int(differing.item()), (got.float() - want.float()).abs().max().item()


@_needs_gpu
@pytest.mark.parametrize(
    "cell", vision_rope_2d_packed_qk.CELLS, ids=[c.why[:44] for c in vision_rope_2d_packed_qk.CELLS]
)
def test_vision_rope_2d_packed_qk_certified_cells(cell, certifying):  # noqa: F811 — pytest resolves the fixture by this name
    certifying(vision_rope_2d_packed_qk)
    spec = cell.spec
    grid_thw = torch.tensor([list(row) for row in spec["grid"]], dtype=spec["grid_dtype"])
    tokens = _num_tokens(spec["grid"])
    qkv = _packed_qkv(tokens, spec["dtype"], seed=zlib.crc32(cell.why.encode()) % 2**31)
    before = qkv.clone()
    cos, sin = _angle_table(grid_thw, MERGE, THETA, HEAD_DIM)
    qk = _qk(qkv)

    want = vision_rope_2d_packed_qk.reference(qk, cos, sin)
    assert torch.equal(qkv, before), "reference must not mutate its input"
    out = vision_rope_2d_packed_qk(qk, cos, sin)
    torch.cuda.synchronize()

    assert out is qk, "the op returns the view it rotated"
    assert torch.equal(qkv[:, 2 * NUM_HEADS :], before[:, 2 * NUM_HEADS :]), "v was written"
    assert not torch.equal(qk, before[:, : 2 * NUM_HEADS]), "q|k was not rotated"

    # 1. The entry's own reference on the same table: bitwise.
    differing, max_dev = _mismatch(qk, want)
    print(f"[{cell.why}] tokens={tokens} vs reference: {differing} differing, max {max_dev:.3e}")
    assert differing == 0, f"{differing} elements differ from reference (max {max_dev:.3e})"

    # 2. The whole-op reference from the grid: bitwise, plus its distribution gates.
    q_whole, k_whole = _whole_op_reference(
        before[:, :NUM_HEADS], before[:, NUM_HEADS : 2 * NUM_HEADS], grid_thw, MERGE, THETA
    )
    for name, got, ref in (
        ("q", qkv[:, :NUM_HEADS], q_whole),
        ("k", qkv[:, NUM_HEADS : 2 * NUM_HEADS], k_whole),
    ):
        differing, max_dev = _mismatch(got, ref)
        cos_sim, rel_l2, p999 = _distribution(got, ref)
        print(
            f"[{cell.why}] {name} vs whole op: {differing} differing, max {max_dev:.3e}, "
            f"cos={cos_sim:.9f} rel_l2={rel_l2:.3e} p99.9={p999:.3e}"
        )
        assert cos_sim >= _COS_BOUND and rel_l2 <= _REL_L2_BOUND, (name, cos_sim, rel_l2)
        assert differing == 0, (
            f"{name}: {differing} elements differ from the whole op (max {max_dev:.3e})"
        )


def _rope_small():
    """A 96-row image (one 8 x 12 frame): the packed qkv and its correct angle table."""
    grid_thw = torch.tensor([(1, 8, 12)], dtype=torch.int64)
    qkv = _packed_qkv(96, torch.bfloat16, seed=17)
    cos, sin = _angle_table(grid_thw, MERGE, THETA, HEAD_DIM)
    return qkv, cos, sin


def _rope_refused(cos: torch.Tensor, sin: torch.Tensor, match: str) -> None:
    """The guarded op refuses the table before the kernel runs: the qkv is untouched."""
    qkv, _, _ = _rope_small()
    before = qkv.clone()
    with pytest.raises(AssertionError, match=match):
        with validating(vision_rope_2d_packed_qk):
            vision_rope_2d_packed_qk(_qk(qkv), cos, sin)
    assert torch.equal(qkv, before), "the guard let the kernel write"


@_needs_gpu
def test_the_rope_guard_refuses_a_bf16_table():
    """The kernel widens a bf16 table and rotates by its rounded angles: a rotation close
    to the right one, and wrong."""
    qkv, cos, sin = _rope_small()
    want = vision_rope_2d_packed_qk.reference(_qk(qkv), cos, sin)
    vision_rope_2d_packed_qk(_qk(qkv), cos.to(torch.bfloat16), sin.to(torch.bfloat16))
    torch.cuda.synchronize()
    differing, max_dev = _mismatch(_qk(qkv), want)
    cos_sim, rel_l2, _ = _distribution(_qk(qkv), want)
    print(f"bf16 table: {differing} differing, max {max_dev:.3e}, rel_l2={rel_l2:.3e}")
    assert differing > 0 and cos_sim > 0.999, (differing, cos_sim)

    _rope_refused(cos.to(torch.bfloat16), sin.to(torch.bfloat16), "must be fp32")


@_needs_gpu
def test_the_rope_guard_refuses_a_table_of_another_shape():
    """The guard holds the table to `[T, head_dim // 2]`. A short table leaves the tail tokens
    unrotated and a narrow one rotates a prefix of each head; a 40-wide one, shown here, pairs
    `(i, i + 40)`, so the second element of the last 8 pairs lies past the head: the last k
    head writes into the first v head, which nothing should address."""
    qkv, cos, sin = _rope_small()
    before = qkv.clone()
    wide_cos = torch.cat((cos, cos[:, :4]), dim=-1).contiguous()
    wide_sin = torch.cat((sin, sin[:, :4]), dim=-1).contiguous()
    vision_rope_2d_packed_qk(_qk(qkv), wide_cos, wide_sin)
    torch.cuda.synchronize()
    v0 = 2 * NUM_HEADS
    written = (qkv[:, v0, :8] != before[:, v0, :8]).any(dim=-1).sum().item()
    print(f"wide table: v head 0 written in {written} of {qkv.shape[0]} rows")
    assert written > 0, "the wide table did not reach the v head"
    assert torch.equal(qkv[:, v0:, 8:], before[:, v0:, 8:]) and torch.equal(
        qkv[:, v0 + 1 :], before[:, v0 + 1 :]
    ), "the wide table wrote past the first 8 elements of v head 0"

    _rope_refused(wide_cos, wide_sin, r"must be \[T, head_dim // 2\]")


@_needs_gpu
def test_the_rope_guard_refuses_a_strided_table():
    """The first half of a duplicated `[T, head_dim]` table has the right shape, but the
    kernel indexes it as `t * half + i` from its base: row `t` reads token `t // 2`'s
    angles."""
    qkv, cos, sin = _rope_small()
    want = vision_rope_2d_packed_qk.reference(_qk(qkv), cos, sin)
    strided_cos = torch.cat((cos, cos), dim=-1)[:, :HALF]
    strided_sin = torch.cat((sin, sin), dim=-1)[:, :HALF]
    assert strided_cos.shape == cos.shape and not strided_cos.is_contiguous()
    vision_rope_2d_packed_qk(_qk(qkv), strided_cos, strided_sin)
    torch.cuda.synchronize()
    rows = (_qk(qkv) != want).flatten(1).any(dim=-1)
    print(f"strided table: {rows.sum().item()} of {rows.numel()} rows wrong")
    assert not rows[0] and rows[1:].any(), "row 0 reads its own angles, the rest do not"

    _rope_refused(strided_cos, strided_sin, "must be contiguous")


# ------------------------------------------- the tower against HuggingFace (GPU) ---

#: (height, width) in pixels of the two test images: grids 1x28x42 and 1x14x20.
_IMAGE_SIZES = ((448, 672), (224, 320))
#: Four 256x384 frames: one video item, grid 2x16x24 (two temporal groups).
_VIDEO_FRAMES, _VIDEO_SIZE = 4, (256, 384)
#: The SDPA backends the bf16 HuggingFace forward runs on: four bf16 implementations of the
#: same model, whose spread is what "a bf16 forward of this model" means.
_HF_BACKENDS = ("MATH", "EFFICIENT_ATTENTION", "FLASH_ATTENTION", "CUDNN_ATTENTION")
#: How far past the worst of those forwards (each statistic against the fp32 reference) the
#: tower may land. The bf16 forwards differ among themselves by up to 1.5x on one item
#: (measured on GB300: relative L2 0.032-0.048 on the 1x14x20 image, max abs 0.036-0.055 on
#: the video, where the math backend's worst element is the tower's); the tower runs the same
#: math with other kernels (fused GEMM epilogues, a fused attention, rope and
#: position-interpolation kernels), so it is one more sample of that spread.
_BAND = 1.25


def _picture(height: int, width: int, seed: int) -> np.ndarray:
    """A deterministic `[H, W, 3]` uint8 picture: smooth colour fields, a few hard-edged
    shapes and a little noise, so the tower sees both low and high spatial frequencies."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width] / max(height, width)
    rgb = np.stack(
        [np.sin(6 * x + seed), np.cos(5 * y - seed), np.sin(3 * (x + y) + 0.5 * seed)], -1
    )
    rgb = 0.5 + 0.5 * rgb
    for _ in range(4):
        top, left = rng.integers(0, height // 2), rng.integers(0, width // 2)
        rgb[top : top + height // 4, left : left + width // 4] = rng.random(3)
    rgb += 0.04 * rng.standard_normal(rgb.shape)
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


@contextlib.contextmanager
def _ieee_fp32():
    """The fp32 reference in IEEE fp32: GEMMs and the patch convolution off TF32, and SDPA on
    its math backend (plain matmuls, so the same switch covers it)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    matmul, conv = torch.backends.cuda.matmul, torch.backends.cudnn.conv
    previous = (matmul.fp32_precision, conv.fp32_precision)
    matmul.fp32_precision = "ieee"
    conv.fp32_precision = "ieee"
    try:
        with sdpa_kernel(SDPBackend.MATH):
            yield
    finally:
        matmul.fp32_precision, conv.fp32_precision = previous


def _stats(out: torch.Tensor, ref: torch.Tensor) -> dict:
    """Max abs error, max abs error over max |ref| ("max rel"), relative L2 and cosine."""
    o, r = out.double().flatten(), ref.double().flatten()
    diff = (o - r).abs()
    return dict(
        max_abs=diff.max().item(),
        max_rel=(diff.max() / r.abs().max()).item(),
        rel_l2=(torch.linalg.vector_norm(o - r) / torch.linalg.vector_norm(r)).item(),
        cos=torch.nn.functional.cosine_similarity(o, r, dim=0).item(),
    )


@pytest.fixture(scope="module")
def checkpoint():
    root = llm_models_root()
    path = None if root is None else root / CHECKPOINT
    if path is None or not (path / "model.safetensors.index.json").is_file():
        pytest.skip(f"{CHECKPOINT} is not under LLM_MODELS_ROOT")
    return path


@pytest.fixture(scope="module")
def models(checkpoint):
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    return _load_models(checkpoint)


@pytest.fixture(scope="module")
def media(checkpoint):
    return _load_media(checkpoint)


@pytest.fixture(scope="module")
def references(models, media):
    return _reference_outputs(models, media)


def _load_models(checkpoint):
    """`(tower, hf32, hf16)` on the checkpoint's own vision weights: this target's tower over
    the weight table's `vis_*` sources, and HuggingFace's `Qwen3_5VisionModel` in fp32 and
    bf16 loaded strictly with the same tensors."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    vision = json.loads((checkpoint / "config.json").read_text())["vision_config"]
    assert vision["deepstack_visual_indexes"] == [], "the reference has no deepstack mergers"
    vcfg = Qwen4ExpVisionConfig(**{k: v for k, v in vision.items() if k != "model_type"})
    d = _dims(vcfg)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]

    # The weight table's own sources for every `vis_*` parameter: one whole checkpoint
    # tensor each, no cut and no transform (the tower is replicated and bf16).
    sources = {}
    for entry in _vision_entries():
        assert entry.transform is None and entry.cut is None, entry.name
        for key, i, prefix in W.MODEL_WEIGHTS._targets(entry, d):
            [(ckpt_key, at, cut)] = entry.sources(d, i, prefix)
            assert at is None and cut is None, (key, ckpt_key)
            sources[key] = (ckpt_key, entry.shape(d), entry.dtype or d.dtype)
    visual = {k for k in index if k.startswith("model.visual.")}
    assert {ckpt_key for ckpt_key, _, _ in sources.values()} == visual, (
        "the table's vis_* sources are not exactly the checkpoint's model.visual.* tensors"
    )
    assert len(sources) == len(visual)

    tensors = {}
    for shard in sorted({index[k] for k in visual}):
        with safetensors.safe_open(str(checkpoint / shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in visual:
                    tensors[key] = f.get_tensor(key)
    w = {}
    for key, (ckpt_key, shape, dtype) in sources.items():
        tensor = tensors[ckpt_key]
        assert tuple(tensor.shape) == tuple(shape) and tensor.dtype == dtype, (
            key,
            tuple(tensor.shape),
            tensor.dtype,
        )
        w[key] = tensor.cuda()
    w["embed"] = torch.empty(1, d.hidden, dtype=torch.bfloat16, device="cuda")
    tower = V.VisionTower(w, vcfg)

    fields = (
        "depth hidden_size hidden_act intermediate_size num_heads in_channels patch_size "
        "spatial_merge_size temporal_patch_size out_hidden_size num_position_embeddings"
    ).split()
    hf_cfg = Qwen3_5VisionConfig(**{k: vision[k] for k in fields})
    hf_cfg._attn_implementation = "sdpa"
    state = {k.removeprefix("model.visual."): t for k, t in tensors.items()}
    out = []
    for dtype in (torch.float32, torch.bfloat16):
        with torch.device("cuda"):
            hf = Qwen3_5VisionModel(hf_cfg)
        hf.load_state_dict(state, strict=True)
        # The rotary's `inv_freq` is an fp32 buffer HF builds explicitly; `from_pretrained`
        # at bf16 leaves it fp32, while `.to(bf16)` would round it.
        inv_freq = hf.rotary_pos_emb.inv_freq.clone()
        hf = hf.to(dtype).eval()
        hf.rotary_pos_emb.inv_freq = inv_freq
        out.append(hf)
    return tower, out[0], out[1]


def _load_media(checkpoint):
    """The checkpoint's own processors' output, `{name: (pixel_values bf16 [rows, 1536] on
    the device, grid_thw int64 on the host)}`: two images and one four-frame video. The
    pixels are rounded to bf16 once, and every model reads those same values."""
    from PIL import Image
    from transformers import AutoImageProcessor, AutoVideoProcessor

    images = AutoImageProcessor.from_pretrained(str(checkpoint))
    videos = AutoVideoProcessor.from_pretrained(str(checkpoint))
    out = {}
    for n, (height, width) in enumerate(_IMAGE_SIZES):
        batch = images(images=[Image.fromarray(_picture(height, width, n))], return_tensors="pt")
        out[f"image{n}"] = (batch["pixel_values"], batch["image_grid_thw"])
    frames = np.stack([_picture(*_VIDEO_SIZE, 10 + f) for f in range(_VIDEO_FRAMES)])
    batch = videos(videos=[frames], return_tensors="pt", do_sample_frames=False)
    out["video"] = (batch["pixel_values_videos"], batch["video_grid_thw"])
    assert [grid.tolist() for _, grid in out.values()] == [
        [[1, 28, 42]],
        [[1, 14, 20]],
        [[2, 16, 24]],
    ]
    return {
        name: (px.to(device="cuda", dtype=torch.bfloat16), grid) for name, (px, grid) in out.items()
    }


def _reference_outputs(models, media):
    """Per media item `(ref, envelope)`: the fp32 HF reference `[rows, 2560]`, and the worst
    of each statistic (`_stats`) the bf16 HF forward reaches against it over `_HF_BACKENDS`."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    _, hf32, hf16 = models
    out = {}
    with torch.inference_mode():
        for name, (px, grid) in media.items():
            with _ieee_fp32():
                ref = hf32(px.float(), grid_thw=grid).pooler_output
            floors = {}
            for backend in _HF_BACKENDS:
                with sdpa_kernel(getattr(SDPBackend, backend)):
                    floors[backend] = _stats(hf16(px, grid_thw=grid).pooler_output, ref)
            for backend, st in floors.items():
                print(
                    f"\n  {name}: bf16 HF ({backend}) vs fp32 HF: max abs {st['max_abs']:.4e}, "
                    f"max rel {st['max_rel']:.4e}, rel L2 {st['rel_l2']:.4e}, cos {st['cos']:.7f}"
                )
            envelope = {
                key: max(st[key] for st in floors.values()) for key in ("max_abs", "rel_l2")
            }
            out[name] = (ref, envelope)
    return out


def _check(name: str, out: torch.Tensor, ref: torch.Tensor, envelope: dict) -> dict:
    """`out` within `_BAND` of the worst bf16 HF forward on every gated statistic."""
    tower = _stats(out, ref)
    print(
        f"\n  {name}: tower vs fp32 HF: max abs {tower['max_abs']:.4e}, max rel "
        f"{tower['max_rel']:.4e}, rel L2 {tower['rel_l2']:.4e}, cos {tower['cos']:.7f} "
        f"(worst bf16 HF: max abs {envelope['max_abs']:.4e}, rel L2 {envelope['rel_l2']:.4e})"
    )
    assert out.dtype == torch.bfloat16 and out.shape == ref.shape
    assert torch.isfinite(out).all()
    for key in ("rel_l2", "max_abs"):
        assert tower[key] <= _BAND * envelope[key], (name, key, tower, envelope)
    return tower


@_needs_gpu
@pytest.mark.parametrize("name", ["image0", "image1", "video"])
def test_the_tower_is_huggingfaces_vision_model(models, media, references, name):
    """One image (1x28x42, 294 placeholder rows), a second at another grid (1x14x20, 70)
    and a four-frame video (2x16x24, two frames of 384 patches attending only to
    themselves, 192 rows): within the bf16 HF forward's own error of the fp32 reference."""
    tower = models[0]
    px, grid = media[name]
    with torch.inference_mode():
        out = tower(px, grid)
    rows = int(grid.prod(-1).sum()) // tower.merge**2
    assert out.shape == (rows, HIDDEN_OUT)
    _check(name, out, *references[name])


@_needs_gpu
def test_two_images_in_one_call_are_two_attention_segments(models, media, references):
    """Both images in one tower call (one `grid_thw` of two rows, 1456 patch rows): each
    image's rows are its own, so the call's rows are the two single-image references."""
    tower = models[0]
    (px0, grid0), (px1, grid1) = media["image0"], media["image1"]
    with torch.inference_mode():
        out = tower(torch.cat([px0, px1]), torch.cat([grid0, grid1]))
    split = int(grid0.prod()) // tower.merge**2
    _check("image0 in a pair", out[:split], *references["image0"])
    _check("image1 in a pair", out[split:], *references["image1"])


@_needs_gpu
def test_calls_cut_between_frames_give_the_same_rows(models, media, references, monkeypatch):
    """With the per-call ceiling lowered to one 384-patch frame, the video runs as two
    calls and the image pair as two; the rows are the references' all the same."""
    tower = models[0]
    px, grid = media["video"]
    assert V._frame_calls([(2, 16, 24)], 384) == [[(1, 16, 24)], [(1, 16, 24)]]
    monkeypatch.setattr(V, "VISION_CALL_MAX_PATCHES", 384)
    with torch.inference_mode():
        out = tower(px, grid)
    _check("video in two calls", out, *references["video"])

    (px0, grid0), (px1, grid1) = media["image0"], media["image1"]
    assert V._frame_calls([(1, 28, 42), (1, 14, 20)], 1176) == [[(1, 28, 42)], [(1, 14, 20)]]
    monkeypatch.setattr(V, "VISION_CALL_MAX_PATCHES", 1176)
    with torch.inference_mode():
        out = tower(torch.cat([px0, px1]), torch.cat([grid0, grid1]))
    split = int(grid0.prod()) // tower.merge**2
    _check("image0, own call", out[:split], *references["image0"])
    _check("image1, own call", out[split:], *references["image1"])


@_needs_gpu
def test_a_mixed_request_reaches_its_placeholders_in_prompt_order(models, media, references):
    """`multimodal_embeds` over one request carrying the 1x14x20 image and the video, the
    video first in the prompt: one tower run per modality, reassembled by item metadata."""
    tower = models[0]
    (ipx, igrid), (vpx, vgrid) = media["image1"], media["video"]
    params = MultimodalParams(
        multimodal_data={
            "mrope_config": {"mrope_position_ids": torch.zeros(3, 1, 300, dtype=torch.int32)},
            "image": {"pixel_values": ipx, "image_grid_thw": igrid},
            "video": {"pixel_values_videos": vpx, "video_grid_thw": vgrid},
            MULTIMODAL_ENCODER_ITEM_METADATA_KEY: MultimodalItemMetadata(
                item_refs=[("video", 0), ("image", 0)]
            ),
        }
    )
    rows = (int(vgrid.prod()) + int(igrid.prod())) // tower.merge**2
    indices = torch.arange(5, 5 + rows, dtype=torch.int64)
    with torch.inference_mode():
        vision, mm_token_indices = V.multimodal_embeds(
            tower, {"multimodal_params": [params], "mm_token_indices": indices}
        )
    assert mm_token_indices is indices
    split = int(vgrid.prod()) // tower.merge**2
    _check("video, mixed request", vision[:split], *references["video"])
    _check("image1, mixed request", vision[split:], *references["image1"])


@_needs_gpu
def test_the_band_sees_a_wrong_rotary(models, media, references, monkeypatch):
    """The band is tight enough to see a plausible wrong tower: the 2-D rotary at twice its
    base fails it on the smaller image."""
    tower = models[0]
    px, grid = media["image1"]
    monkeypatch.setattr(V, "VISION_ROPE_THETA", 2 * V.VISION_ROPE_THETA)
    with torch.inference_mode():
        out = tower(px, grid)
    with pytest.raises(AssertionError, match="rel_l2"):
        _check("image1, rotary base x2", out, *references["image1"])


@_needs_gpu
def test_the_band_sees_frames_attending_to_each_other(models, media, references, monkeypatch):
    """Each frame is its own attention segment: the video's two frames run as one segment
    (every other operand unchanged) fail the band."""
    tower = models[0]
    px, grid = media["video"]
    attention = V.thop_attention

    def one_segment(**kwargs):
        rows = kwargs["num_ctx_tokens"]
        host = torch.tensor([rows], dtype=torch.int32)
        dev = host.to(kwargs["q"].device)
        kwargs.update(
            sequence_length=dev,
            host_past_key_value_lengths=host,
            host_total_kv_lens=torch.tensor([rows, 0], dtype=torch.int32),
            context_lengths=dev,
            host_context_lengths=host,
            host_request_types=torch.zeros([1], dtype=torch.int32),
            max_num_requests=1,
            max_context_length=rows,
            max_seq_len=rows,
            attention_window_size=rows,
            num_contexts=1,
        )
        return attention(**kwargs)

    monkeypatch.setattr(V, "thop_attention", one_segment)
    with torch.inference_mode():
        out = tower(px, grid)
    with pytest.raises(AssertionError, match="rel_l2"):
        _check("video, frames as one segment", out, *references["video"])


# --------------------------------------------- triton_pos_embed_interpolate ---

NUM_GRID = 48
VISION_HIDDEN = 1152


def _pos_embed_compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """4 bf16 ulp per element against the row scale, 1 ulp relative RMS.

    The op rounds its four corner weights to bf16 and multiply-adds in bf16 (the
    HF reference's arithmetic); `ref` is the exact value. Measured worst over the
    cells: 3.494 / 0.752 across 50 seeds of `_pos_table`, 3.349 / 0.745 on this
    checkpoint's own table. The nearest wrong variant, `align_corners=False`,
    lands at 39.8 / 15.3 or more, raster order and swapped axes above 400 / 300.
    """
    assert out.dtype == torch.bfloat16, out.dtype
    assert_within_ulp(out.float(), ref, element_ulp=4.0, rms_ulp=1.0, ulp=ULP[torch.bfloat16])


def _pos_embed_ulp(out: torch.Tensor, ref: torch.Tensor) -> tuple[float, float]:
    """`_pos_embed_compare`'s two statistics: (row-scale element error, relative RMS) in bf16 ulp."""
    o, r = out.float(), ref.float()
    scale = r.abs().amax(dim=-1, keepdim=True).clamp_min(1e-9)
    elt = ((o - r) / scale).abs().max().item() / ULP[torch.bfloat16]
    rms = ((o - r).pow(2).mean().sqrt() / r.pow(2).mean().sqrt()).item() / ULP[torch.bfloat16]
    return elt, rms


def _pos_table(seed: int, num_grid: int = NUM_GRID, hidden: int = VISION_HIDDEN) -> torch.Tensor:
    """A smooth `[num_grid**2, hidden]` bf16 table, the way a learned 2-D
    position table is: a 12x12 normal field upsampled bicubically, plus 0.1
    noise. Neighbouring rows along w have cosine 0.913; this checkpoint's
    table's have 0.929.

    Contiguous on purpose: the upsampled field's permuted view reshapes to a
    strided table, which the kernel reads wrongly (`is_valid` refuses it).
    """
    g = torch.Generator(device="cuda").manual_seed(seed)
    coarse = torch.randn(1, hidden, 12, 12, generator=g, device="cuda")
    up = F.interpolate(coarse, size=(num_grid, num_grid), mode="bicubic", align_corners=True)
    up = up[0].permute(1, 2, 0) + 0.1 * torch.randn(
        num_grid, num_grid, hidden, generator=g, device="cuda"
    )
    return up.reshape(num_grid * num_grid, hidden).to(torch.bfloat16).contiguous()


def _resize64(table, t, h, w, m_size, *, align=True, raster=False, swap=False):
    """torch's bilinear resize in fp64, in merge order unless `raster`."""
    num_grid = int(round(table.shape[0] ** 0.5))
    grid = table.double().view(num_grid, num_grid, -1)
    if swap:
        grid = grid.transpose(0, 1)
    src = grid.permute(2, 0, 1).unsqueeze(0).contiguous()
    out = F.interpolate(src, size=(h, w), mode="bilinear", align_corners=align)[0].permute(1, 2, 0)
    if raster:
        rows = out.reshape(h * w, -1)
    else:
        rows = (
            out.reshape(h // m_size, m_size, w // m_size, m_size, -1)
            .permute(0, 2, 1, 3, 4)
            .reshape(h * w, -1)
        )
    return rows.repeat(t, 1) if t > 1 else rows


def _pos_embed_args(spec, table):
    return (
        table,
        spec["t"],
        spec["h"],
        spec["w"],
        spec["num_grid_per_side"],
        spec["m_size"],
        spec["dtype"],
    )


@_needs_gpu
@pytest.mark.parametrize(
    "cell",
    triton_pos_embed_interpolate.CELLS,
    ids=[c.why[:44] for c in triton_pos_embed_interpolate.CELLS],
)
def test_triton_pos_embed_interpolate_certified_cells(cell, certifying):  # noqa: F811
    certifying(triton_pos_embed_interpolate)
    spec = cell.spec
    t, h, w, m_size = spec["t"], spec["h"], spec["w"], spec["m_size"]
    table = _pos_table(
        zlib.crc32(cell.why.encode()) % 2**31, spec["num_grid_per_side"], spec["hidden"]
    )
    before = table.clone()

    out = triton_pos_embed_interpolate(*_pos_embed_args(spec, table))
    assert out.shape == (t * h * w, spec["hidden"]) and out.dtype == spec["dtype"]
    assert torch.equal(table, before) and out.data_ptr() != table.data_ptr()

    ref = triton_pos_embed_interpolate.reference(*_pos_embed_args(spec, table))
    elt, rms = _pos_embed_ulp(out, ref)
    print(f"  {cell.why[:44]}: {elt:.3f} ulp element-wise, {rms:.3f} ulp RMS")
    _pos_embed_compare(out, ref)

    # The reference's structure, from an implementation that shares nothing with it.
    torch.testing.assert_close(ref, _resize64(table, t, h, w, m_size).float())

    # The band refuses each wrong variant of the function.
    for wrong in (dict(align=False), dict(raster=True), dict(swap=True)):
        with pytest.raises(AssertionError):
            _pos_embed_compare(out, _resize64(table, t, h, w, m_size, **wrong).float())


@pytest.fixture(scope="module")
def checkpoint_table(checkpoint):
    """`(vision_config, model.visual.pos_embed.weight)` of this target's checkpoint."""
    vision = json.loads((checkpoint / "config.json").read_text())["vision_config"]
    key = "model.visual.pos_embed.weight"
    shard = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"][key]
    with safetensors.safe_open(str(checkpoint / shard), framework="pt", device="cpu") as handle:
        table = handle.get_tensor(key)
    return vision, table.cuda().contiguous()


@_needs_gpu
def test_the_checkpoints_own_position_table_stays_in_the_band(checkpoint_table, certifying):  # noqa: F811
    """The cells' geometry is this checkpoint's, and its learned table sits in the band
    at every cell's grid: measured worst 3.349 / 0.745 ulp, at the 32768-row grid."""
    certifying(triton_pos_embed_interpolate)
    vision, table = checkpoint_table
    assert vision["num_position_embeddings"] == NUM_GRID * NUM_GRID
    assert vision["hidden_size"] == VISION_HIDDEN and vision["spatial_merge_size"] == MERGE
    assert table.shape == (NUM_GRID * NUM_GRID, VISION_HIDDEN) and table.dtype == torch.bfloat16
    for cell in triton_pos_embed_interpolate.CELLS:
        out = triton_pos_embed_interpolate(*_pos_embed_args(cell.spec, table))
        ref = triton_pos_embed_interpolate.reference(*_pos_embed_args(cell.spec, table))
        elt, rms = _pos_embed_ulp(out, ref)
        print(f"  checkpoint table, {cell.why[:44]}: {elt:.3f} / {rms:.3f} ulp")
        _pos_embed_compare(out, ref)


_SERVING = triton_pos_embed_interpolate.CELLS[1].spec  # the 62x90 serving grid


@_needs_gpu
def test_the_pos_embed_guard_refuses_a_strided_table():
    """The kernel indexes the table as dense rows of `hidden`: a column slice of a wider
    buffer, the same values as a contiguous table, is read wrongly without an error."""
    table = _pos_table(7)
    strided = torch.cat([table, _pos_table(8)], dim=1)[:, :VISION_HIDDEN]
    assert not strided.is_contiguous() and torch.equal(strided, table)
    unguarded = triton_pos_embed_interpolate(*_pos_embed_args(_SERVING, strided))
    with pytest.raises(AssertionError):
        _pos_embed_compare(
            unguarded, triton_pos_embed_interpolate.reference(*_pos_embed_args(_SERVING, strided))
        )
    with pytest.raises(AssertionError, match="must be contiguous"):
        with validating(triton_pos_embed_interpolate):
            triton_pos_embed_interpolate(*_pos_embed_args(_SERVING, strided))


@_needs_gpu
def test_the_pos_embed_guard_refuses_a_grid_side_that_is_not_the_tables():
    """The grid side is an argument, not read off the table: 47 on the 2304-row table
    resamples a 47x47 grid out of it and returns a correctly shaped wrong result."""
    table = _pos_table(9)
    spec = dict(_SERVING, num_grid_per_side=NUM_GRID - 1)
    unguarded = triton_pos_embed_interpolate(*_pos_embed_args(spec, table))
    assert unguarded.shape == (_SERVING["h"] * _SERVING["w"], VISION_HIDDEN)
    with pytest.raises(AssertionError):
        _pos_embed_compare(
            unguarded, triton_pos_embed_interpolate.reference(*_pos_embed_args(_SERVING, table))
        )
    with pytest.raises(AssertionError, match="does not describe"):
        with validating(triton_pos_embed_interpolate):
            triton_pos_embed_interpolate(*_pos_embed_args(spec, table))


@_needs_gpu
def test_the_pos_embed_guard_refuses_an_empty_grid_side():
    """A side of 0 returns `[0, hidden]` rather than raising (one check covers t, h and w)."""
    table = _pos_table(10)
    spec = dict(triton_pos_embed_interpolate.CELLS[0].spec, h=0)
    assert triton_pos_embed_interpolate(*_pos_embed_args(spec, table)).shape == (0, VISION_HIDDEN)
    with pytest.raises(AssertionError, match="empty grid side"):
        with validating(triton_pos_embed_interpolate):
            triton_pos_embed_interpolate(*_pos_embed_args(spec, table))


@_needs_gpu
def test_the_pos_embed_guard_refuses_a_side_that_is_not_whole_merge_blocks():
    """The wrapper's only check on a 6x9 grid is a bare `assert`, gone under `python -O`
    (where the grid is read outside itself: recorded in the entry's note, not run here)."""
    table = _pos_table(11)
    spec = dict(triton_pos_embed_interpolate.CELLS[0].spec, h=6, w=9)
    with pytest.raises(AssertionError, match="must be divisible"):
        triton_pos_embed_interpolate(*_pos_embed_args(spec, table))
    with pytest.raises(AssertionError, match="whole m_size"):
        with validating(triton_pos_embed_interpolate):
            triton_pos_embed_interpolate(*_pos_embed_args(spec, table))
