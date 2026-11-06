# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""qwen4_exp's multimodal splice runs the vision tower once over the step's images and
videos and puts the rows back in request order, then prompt order; the tower itself runs
the frames in calls of whole frames under a row ceiling.

The tower is a stub whose rows name the item they came from, so the bookkeeping --
the joined call, item metadata, the chunk slice, the frame grouping and the order of
the concatenated calls -- runs on the CPU without the vision ops or an engine.
"""

from __future__ import annotations

import math
import types
from types import SimpleNamespace

import pytest
import torch

__extra_import_path__ = ["../../.."]  # noqa: F841 -- repo's file-scoped import hook

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalEncoderContractError
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
    """Stands in for `VisionTower`: every row of an item holds the value of the item's
    first patch row, which `_media` sets to `100 * m + i` for item `i` of modality `m`,
    so a spliced row says where it came from."""

    def __init__(self):
        self.calls = 0

    def __call__(self, pixel_values, grid_thw):
        self.calls += 1
        rows, start = [], 0
        for t, h, w in grid_thw.tolist():
            rows.append(
                torch.full((t * h * w // (MERGE * MERGE), HIDDEN), float(pixel_values[start, 0]))
            )
            start += t * h * w
        assert start == pixel_values.shape[0]
        return torch.cat(rows)


def _shell(tower):
    shell = SimpleNamespace(
        model=SimpleNamespace(
            v_merge=MERGE, w={"embed": torch.zeros(16, HIDDEN)}, vision_tower=tower
        )
    )
    shell._vision_rows = types.MethodType(Shell._vision_rows, shell)
    shell.mm_encoder_groups = Shell.mm_encoder_groups.fget(shell)
    return shell


def _media(modality, grids):
    """One `multimodal_data` entry; item `i`'s patch rows hold `100 * modality + i`."""
    key = "pixel_values" if modality == IMAGE else "pixel_values_videos"
    grid_key = "image_grid_thw" if modality == IMAGE else "video_grid_thw"
    pixels = [
        torch.full((t * h * w, 1), float(100 * modality + item))
        for item, (t, h, w) in enumerate(grids)
    ]
    return {key: torch.cat(pixels), grid_key: torch.tensor(grids)}


def _params(image=None, video=None, order=None, runtime=None, mrope=True, manifest=True):
    """A media request with the item metadata the processor emits: `order` lists its
    items in prompt order (default: images, then videos), each with its embedding rows."""
    data = {}
    if mrope:
        data["mrope_config"] = {"mrope_position_ids": torch.zeros(3, 1, 8, dtype=torch.int32)}
    grids = {}
    if image is not None:
        data["image"] = _media(IMAGE, image)
        grids["image"] = image
    if video is not None:
        data["video"] = _media(VIDEO, video)
        grids["video"] = video
    if manifest:
        if order is None:
            order = [(modality, i) for modality, grid in grids.items() for i in range(len(grid))]
        data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = MultimodalItemMetadata(
            item_refs=order,
            output_embedding_lengths=[
                math.prod(grids[modality][index]) // (MERGE * MERGE) for modality, index in order
            ],
        )
    return MultimodalParams(multimodal_data=data, multimodal_runtime=runtime)


def _splice(tower, params, num_rows, padding=0):
    """The tower's rows spliced, as the prefill target's break does, over a prompt whose
    tokens 1..num_rows are the placeholders and which `padding` rows follow; returns the
    rows that landed there, one value per row, or None when the step carries no media.
    `params` is one request's or a list of requests'."""
    kwargs = {
        "multimodal_params": params if isinstance(params, list) else [params],
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
    # One tower call carries the images and the videos.
    assert tower.calls == 1


def test_a_steps_requests_share_one_tower_call_in_request_order():
    """A video request, a mixed one and an image one: one tower call, and each request's
    rows land on its own placeholders in its prompt order."""
    tower = _Tower()
    params = [
        _params(video=VIDEOS[:1]),
        _params(image=IMAGES, video=VIDEOS, order=MIXED_ORDER),
        _params(image=IMAGES[1:]),
    ]
    rows = [200.0] * 4 + MIXED_ROWS + [100.0]
    assert _splice(tower, params, len(rows)) == rows
    assert tower.calls == 1


def test_chunks_of_several_requests_take_their_slices():
    """Each request keeps only its chunk's rows; a request with no rows in this chunk is
    not encoded."""
    tower = _Tower()

    def runtime(cached, count, total):
        return SimpleNamespace(
            num_cached_mm_tokens=cached, num_mm_tokens_in_chunk=count, total_embeds_in_request=total
        )

    params = [
        _params(image=IMAGES, video=VIDEOS, order=MIXED_ORDER, runtime=runtime(3, 6, 17)),
        _params(video=VIDEOS, runtime=runtime(12, 0, 12)),
        _params(image=IMAGES, runtime=runtime(0, 5, 5)),
    ]
    assert _splice(tower, params, 11) == MIXED_ROWS[3:9] + [100.0] * 4 + [101.0]
    assert tower.calls == 1


@pytest.mark.parametrize(
    "params",
    [
        _params(image=IMAGES, manifest=False),
        _params(video=VIDEOS, manifest=False),
        _params(image=IMAGES, video=VIDEOS, manifest=False),
    ],
    ids=["image", "video", "mixed"],
)
def test_a_media_request_needs_its_manifest(params):
    # `validate_request` refuses such a request at admission; the shared encode refuses it
    # as well rather than guess the rows.
    with pytest.raises(MultimodalEncoderContractError, match="item metadata"):
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


# Verbatim copies of main's splice (`_multimodal_embeds`, `_run_tower` and
# `_in_prompt_order`), which ran the tower once per modality of each request; module
# names are qualified with `M.`.


def _main_multimodal_embeds(self, kwargs):
    """Run the vision tower for this batch; returns `(rows, mm_token_indices)` (the
    tower's `[R, hidden]` rows in packed-batch order and the `[R]` placeholder rows they
    replace, which the prefill target splices into its embedding), or None when the step
    carries no media.

    Lives in the shell because the inherited `DecoderModelForCausalLM.forward`
    drops `multimodal_params` from `**kwargs` on the no-speculation path (a
    booting, fluent, blind engine). Host-side bookkeeping: `multimodal_params`
    is a subset of the context requests, so never zip it positionally;
    `multimodal_runtime` (chunked prefill / KV reuse only) gives the cached and
    in-chunk row counts over the request's images and videos together, in prompt
    order; `mm_token_indices` names every placeholder row.

    The tower runs once per modality a request carries, and `_in_prompt_order`
    merges the two outputs. Positions are not derived here: every coordinate of
    a vision span, a video's temporal one included, is in the payload's
    `mrope_config.mrope_position_ids`, which the engine slices into
    `position_ids`."""
    mm_params = kwargs.get("multimodal_params") or []
    if not mm_params:
        return None
    rows = []
    for params in mm_params:
        data = getattr(params, "multimodal_data", None) or {}
        image, video = data.get("image"), data.get("video")
        if image is None and video is None:
            continue
        mrope_ids = (data.get("mrope_config") or {}).get("mrope_position_ids")
        assert (
            mrope_ids is not None
            and mrope_ids.dim() == 3
            and mrope_ids.shape[0] == M._QSA_POSITION_AXES
        ), (
            f"a media request needs mrope_config.mrope_position_ids of shape "
            f"[{M._QSA_POSITION_AXES}, 1, L]; got "
            f"{None if mrope_ids is None else tuple(mrope_ids.shape)}"
        )
        runtime = getattr(params, "multimodal_runtime", None)
        if runtime is not None and int(runtime.num_mm_tokens_in_chunk) == 0:
            # Every placeholder row of this request lies in another chunk.
            continue
        towers = {}
        if image is not None:
            towers["image"] = self._run_tower(image, *M._MEDIA_KEYS["image"])
        if video is not None:
            towers["video"] = self._run_tower(video, *M._MEDIA_KEYS["video"])
        embeds = self._in_prompt_order(params, towers)
        if runtime is not None:
            # Chunked prefill or reused prefix: splice only this chunk's slice.
            start = int(runtime.num_cached_mm_tokens)
            count = int(runtime.num_mm_tokens_in_chunk)
            total = embeds.shape[0]
            assert int(runtime.total_embeds_in_request) == total, (
                f"the executor counted {runtime.total_embeds_in_request} "
                f"placeholder rows for this request and the tower produced "
                f"{total} over its {sorted(towers)}; the two are derived from the "
                "same grids and a disagreement means the merger's row count is wrong"
            )
            assert 0 <= start <= start + count <= total, (start, count, total)
            embeds = embeds[start : start + count]
        rows.append(embeds)
    if not rows:
        return None
    vision = rows[0] if len(rows) == 1 else torch.cat(rows, 0)
    mm_token_indices = kwargs.get("mm_token_indices")
    assert mm_token_indices is not None, (
        "the executor shipped multimodal params but no mm_token_indices; "
        "that pairing is what `multimodal_token_ids` on this class buys, and "
        "without it there is no sync-free way to find the placeholder rows"
    )
    assert mm_token_indices.shape[0] == vision.shape[0], (
        f"the batch has {mm_token_indices.shape[0]} placeholder rows and the "
        f"tower produced {vision.shape[0]} embedding rows"
    )
    return vision, mm_token_indices


def _main_run_tower(self, media, pixel_key, grid_key):
    """The tower over one modality's payload: `(rows, grid_thw)`, the rows one per
    placeholder, item by item in the order the processor emitted them."""
    pixel_values, grid_thw = media.get(pixel_key), media.get(grid_key)
    assert pixel_values is not None and grid_thw is not None, (
        f"the multimodal payload carries {sorted(media)}; this target "
        f"reads {pixel_key!r} and {grid_key!r} from it"
    )
    assert not grid_thw.is_cuda, (
        f"{grid_key} must stay on the host; see multimodal_data_device_paths"
    )
    return self.model.vision_tower(pixel_values, grid_thw), grid_thw


def _main_in_prompt_order(self, params, towers):
    """One request's tower rows in the order its placeholders appear.

    `towers` maps each modality the request carries to `(rows, grid_thw)`; each
    grid row is one item of `t*h*w / merge**2` rows. One modality is already in
    prompt order. Two are reassembled item by item from `item_metadata.item_refs`
    (`(modality, index)` per item in prompt order, `index` counting that
    modality's items); without it there is no host-side way to tell which
    placeholder run is an image and which a video."""
    if len(towers) == 1:
        return next(iter(towers.values()))[0]
    metadata = params.item_metadata
    order = metadata.item_refs if metadata is not None else None
    assert order, (
        f"a request carrying {sorted(towers)} needs item metadata to splice them "
        "in prompt order; the payload holds each modality's rows apart"
    )
    merge_unit = self.model.v_merge * self.model.v_merge
    items = {}
    for modality, (embeds, grid_thw) in towers.items():
        counts = [t * h * w // merge_unit for t, h, w in grid_thw.tolist()]
        assert sum(counts) == embeds.shape[0], (
            f"the {modality} grid describes {sum(counts)} rows and the tower "
            f"produced {embeds.shape[0]}"
        )
        items[modality] = torch.split(embeds, counts, 0)
    listed = {modality: [] for modality in items}
    for modality, index in order:
        assert modality in items, (
            f"item_refs lists a {modality!r} item; this request carries {sorted(items)}"
        )
        listed[modality].append(int(index))
    for modality, parts in items.items():
        assert sorted(listed[modality]) == list(range(len(parts))), (
            f"item_refs lists {modality} items {listed[modality]}; the payload carries {len(parts)}"
        )
    return torch.cat([items[modality][int(index)] for modality, index in order], 0)


def _main_shell(tower):
    shell = SimpleNamespace(
        model=SimpleNamespace(
            v_merge=MERGE, w={"embed": torch.zeros(16, HIDDEN)}, vision_tower=tower
        )
    )
    shell._run_tower = types.MethodType(_main_run_tower, shell)
    shell._in_prompt_order = types.MethodType(_main_in_prompt_order, shell)
    return shell


def _runtime(cached, count, total):
    return SimpleNamespace(
        num_cached_mm_tokens=cached, num_mm_tokens_in_chunk=count, total_embeds_in_request=total
    )


_MAIN_BATCHES = {
    "one_image_request": lambda: [_params(image=IMAGES)],
    "requests_of_each_kind": lambda: [
        _params(video=VIDEOS[:1]),
        _params(image=IMAGES, video=VIDEOS, order=MIXED_ORDER),
        _params(image=IMAGES[1:]),
    ],
    "chunks": lambda: [
        _params(image=IMAGES, video=VIDEOS, order=MIXED_ORDER, runtime=_runtime(3, 6, 17)),
        _params(video=VIDEOS, runtime=_runtime(12, 0, 12)),
        _params(image=IMAGES, runtime=_runtime(0, 5, 5)),
        _params(video=VIDEOS, runtime=_runtime(4, 8, 12)),
    ],
}


@pytest.mark.parametrize("batch", list(_MAIN_BATCHES))
def test_the_splice_matches_main_per_request_towers(batch):
    """The joined call splices the same rows as main's per-request, per-modality calls."""
    params = _MAIN_BATCHES[batch]()
    num_rows = sum(
        int(p.multimodal_runtime.num_mm_tokens_in_chunk)
        if p.multimodal_runtime is not None
        else sum(p.item_metadata.output_embedding_lengths)
        for p in params
    )
    kwargs = {
        "multimodal_params": params,
        "mm_token_indices": torch.arange(num_rows, dtype=torch.int64),
    }
    main_tower, tower = _Tower(), _Tower()
    expected, _ = _main_multimodal_embeds(_main_shell(main_tower), kwargs)
    actual, _ = Shell._multimodal_embeds(_shell(tower), kwargs)
    assert torch.equal(actual, expected)
    # Main called the tower once per modality of each request with rows in this chunk.
    assert main_tower.calls == sum(
        ("image" in p.multimodal_data) + ("video" in p.multimodal_data)
        for p in params
        if p.multimodal_runtime is None or p.multimodal_runtime.num_mm_tokens_in_chunk
    )
    assert tower.calls == 1


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
