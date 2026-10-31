# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The unified target's vision path on the CPU: the tower's weights, its calls and angle table,
the splice's bookkeeping and the admission check.

The tower is built over the weight table's own `vis_*` declarations on the meta device, so
what it reads is what `weights.py` declares. Its per-call body is a stub where a test needs one
(the vision ops need a GPU), and so is the whole tower where the shell's bookkeeping is under
test: every row of a stub tower names the item it came from, so the modality dispatch, item
metadata, chunk slice, frame grouping and the order of concatenated calls run with no engine.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified import (  # noqa: E501
    vision as V,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp_for_conditional_generation.hf_fc694b5_orig__sm_103__unified import (  # noqa: E501
    weights as W,
)
from tensorrt_llm._torch.configs import Qwen4ExpVisionConfig
from tensorrt_llm.inputs.multimodal import (
    MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
    MultimodalItemMetadata,
    MultimodalParams,
)

pytestmark = pytest.mark.cpu_only

Shell = M.ModelingV2HfFc694b5OrigSm103Unified

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


def _tower_over_the_weight_table() -> V.VisionTower:
    """The tower over every `vis_*` weight the table declares, at the checkpoint's shapes,
    on meta; `embed` stands in for the activation dtype the tower reads off it."""
    vcfg = Qwen4ExpVisionConfig(**_VISION)
    d = SimpleNamespace(
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
    w = {"embed": torch.empty(8, HIDDEN_OUT, dtype=torch.bfloat16, device="meta")}
    for entry in W.MODEL_WEIGHTS.WEIGHTS:
        if not entry.name.startswith("vis_"):
            continue
        for key, _, _ in W.MODEL_WEIGHTS._targets(entry, d):
            w[key] = torch.empty(*entry.shape(d), dtype=entry.dtype or d.dtype, device="meta")
    return V.VisionTower(w, vcfg)


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


# --------------------------------------------------------------- angle table ---


def _nested_loop_positions(grid, merge):
    """Per-row `(row, col)` by four nested loops, in spatial-merge order: the template the
    rotary entry's test builds its certified tables from."""
    positions = []
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


@pytest.mark.parametrize(
    "grid", [[(1, 62, 90)], [(3, 8, 8), (1, 16, 24)], [(2, 4, 6)]], ids=["62x90", "mixed", "t2"]
)
def test_the_angle_table_is_the_rotary_entrys_certified_table(grid):
    """`_rope_tables` builds, bit for bit, the table `vision_rope_2d_packed_qk` is
    certified with: positions in spatial-merge order, fp64 angles rounded once to fp32,
    height angles then width angles, `[rows, 36]` contiguous."""
    assert V._merge_order_positions(grid, 2).tolist() == [
        list(p) for p in _nested_loop_positions(grid, 2)
    ]
    cos, sin = V._rope_tables(grid, 2, V.VISION_ROPE_THETA, 72, "cpu")
    rope_dim = 36
    inv_freq = 1.0 / (
        V.VISION_ROPE_THETA ** (torch.arange(0, rope_dim, 2, dtype=torch.float64) / rope_dim)
    )
    pos = torch.tensor(_nested_loop_positions(grid, 2), dtype=torch.float64)
    angles = (pos[:, :, None] * inv_freq).flatten(1).to(torch.float32)
    rows = sum(t * h * w for t, h, w in grid)
    for got, want in ((cos, angles.cos()), (sin, angles.sin())):
        assert got.dtype == torch.float32 and got.is_contiguous()
        assert tuple(got.shape) == (rows, rope_dim)
        assert torch.equal(got, want)


# ---------------------------------------------------------------- tower calls ---

HIDDEN = 3
MERGE = 2
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


def test_frames_are_grouped_whole_and_in_order():
    # 16-row frames, 32 rows to a call: one item's frames split across calls.
    assert V._frame_calls([(5, 4, 4)], 32) == [[(2, 4, 4)], [(2, 4, 4)], [(1, 4, 4)]]
    # A call takes the end of one item and the start of the next.
    grid = [(1, 8, 4), (3, 4, 4), (1, 2, 2)]
    assert V._frame_calls(grid, 48) == [[(1, 8, 4), (1, 4, 4)], [(2, 4, 4), (1, 2, 2)]]
    assert V._frame_calls(grid, 1 << 20) == [grid]


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


def test_one_frame_past_the_ceiling_is_refused_and_one_at_it_is_not():
    assert V._frame_calls([(2, 8, 8)], 64) == [[(1, 8, 8)], [(1, 8, 8)]]
    with pytest.raises(AssertionError, match="one frame is 8x10 = 80 patch rows"):
        V._frame_calls([(1, 4, 4), (3, 8, 10)], 64)


def test_the_tower_call_ceiling_is_the_vision_entries_largest_cell():
    """32768 patch rows (8192 merged rows) per call: the ceiling the patch embedding,
    position interpolation, layer norm, rotary, GEMM and attention cells reach."""
    assert V.VISION_CALL_MAX_PATCHES == 32768


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
    """The tower's rows spliced, as the prefill target's break does, over a prompt whose
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


def _item(modality, index):
    return modality, index


# Two videos of 4 and 8 rows, two images of 4 and 1.
VIDEOS = [(2, 2, 4), (2, 4, 4)]
IMAGES = [(1, 4, 4), (1, 2, 2)]
MIXED_ORDER = [_item("video", 0), _item("image", 0), _item("video", 1), _item("image", 1)]
MIXED_ROWS = [200.0] * 4 + [100.0] * 4 + [201.0] * 8 + [101.0]


def test_a_text_step_runs_no_tower():
    assert V.multimodal_embeds(None, {}) is None
    assert V.multimodal_embeds(None, {"multimodal_params": []}) is None
    # Position ids alone (a text prompt through a multimodal processor) are not media.
    assert V.multimodal_embeds(None, {"multimodal_params": [_params()]}) is None


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


def test_a_media_request_needs_the_processors_position_ids():
    with pytest.raises(AssertionError, match="mrope_position_ids"):
        _splice(_Tower(), _params(video=VIDEOS, mrope=False), 12)


def test_the_placeholder_rows_must_match_the_towers_rows():
    kwargs = {
        "multimodal_params": [_params(image=IMAGES)],
        "mm_token_indices": torch.arange(4, dtype=torch.int64),
    }
    with pytest.raises(AssertionError, match="4 placeholder rows and the tower produced 5"):
        V.multimodal_embeds(_Tower(), kwargs)


def test_the_grids_stay_on_the_host():
    """The executor moves only the pixels and the position ids; the grids the tower and the
    admission check read on the host stay there."""
    paths = Shell.multimodal_data_device_paths.fget(None)
    assert set(paths) == {
        "image.pixel_values",
        "video.pixel_values_videos",
        "mrope_config.mrope_position_ids",
        "mrope_config.mrope_position_deltas",
    }
    assert not any("grid_thw" in path for path in paths)


# ---------------------------------------------------------------- admission ---

PATCH = 16
TEXT_TOKENS = 5

_GEOMETRY = SimpleNamespace(merge=MERGE, patch=PATCH, pixel_width=PIXEL_WIDTH)


def _request_media(modality, grids):
    """A payload whose pixel tensor has the right shape and no storage."""
    patches = sum(t * h * w for t, h, w in grids)
    key = "pixel_values" if modality == "image" else "pixel_values_videos"
    grid_key = "image_grid_thw" if modality == "image" else "video_grid_thw"
    return {
        key: torch.empty(patches, PIXEL_WIDTH, device="meta"),
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


def _admit(request):
    """Through the shell's own hook, as the executor calls it."""
    Shell.validate_request(SimpleNamespace(vision_tower=_GEOMETRY), request)


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


def test_the_longest_admitted_frame_is_the_calls_ceiling(monkeypatch):
    monkeypatch.setattr(V, "VISION_CALL_MAX_PATCHES", 64)
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
    with pytest.raises(ValueError, match="video 1 has frames of 192x192"):
        _admit(_request(video=[(4, 128, 128), (1, 192, 192)]))


def test_a_malformed_grid_is_refused():
    request = _request(image=[(1, 4, 4)])
    request.py_multimodal_data["image"]["image_grid_thw"] = torch.tensor([(1, 4, 5)])
    with pytest.raises(ValueError, match="spatial_merge_size 2"):
        _admit(request)
    request.py_multimodal_data["image"]["image_grid_thw"] = torch.tensor([1, 4, 4])
    with pytest.raises(ValueError, match="one \\(t, h, w\\) row per image"):
        _admit(request)


def test_pixels_that_disagree_with_the_grid_are_refused():
    request = _request(video=VIDEOS)
    video = request.py_multimodal_data["video"]
    video["pixel_values_videos"] = torch.empty(5, PIXEL_WIDTH, device="meta")
    with pytest.raises(ValueError, match="pixel_values_videos is .*describe 48 patch rows"):
        _admit(request)
    del video["video_grid_thw"]
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


def test_a_mixed_request_lists_its_items():
    with pytest.raises(ValueError, match="needs item metadata"):
        _admit(_request(image=IMAGES, video=VIDEOS))
    with pytest.raises(ValueError, match="lists image items \\[0\\]; the payload carries 2"):
        _admit(_request(image=IMAGES, video=VIDEOS, order=MIXED_ORDER[:3]))
    with pytest.raises(ValueError, match="is not an item of this request"):
        _admit(_request(image=IMAGES, video=VIDEOS, order=MIXED_ORDER + [_item("audio", 0)]))


def test_the_placeholder_count_is_the_sum_over_both_modalities():
    request = _request(image=IMAGES, video=VIDEOS, order=MIXED_ORDER)
    request.py_multimodal_data["multimodal_embed_mask_cumsum"] = torch.arange(1, 14)
    with pytest.raises(ValueError, match="holds 13 image/video placeholder tokens.*17"):
        _admit(request)
    del request.py_multimodal_data["multimodal_embed_mask_cumsum"]
    _admit(request)
