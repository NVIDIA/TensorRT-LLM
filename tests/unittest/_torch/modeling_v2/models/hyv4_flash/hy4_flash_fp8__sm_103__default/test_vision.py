# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The target's vision tower: the checkpoint's `vit.*` tensors, the tower's own math, and the
row accounting the image processor, the executor and the tower have to agree on.

The tower's weights come from the table's `vis_*` entries through the loader's own
`manifest`/`read`, so a renamed key, a wrong row split of the packed q|k|v or a stale
transpose fails here. The rows the tower produces are the placeholder rows the checkpoint's
image processor counts for the same image, and the executor allocates. And the tower's output
is compared against the reference `HYV4VLVisionModel`'s math, transcribed here from
`transformers/models/hy_v4_vl/modeling_hy_v4_vl.py` and run over the checkpoint's weights:
patch embedding, interpolated position table, 27 pre-norm blocks with bidirectional
per-frame attention, the RMSNorm merger with its newline column, and no rotary anywhere.

The CPU half needs the checkpoint; the tower comparison needs a CUDA device.
"""

import contextlib
import functools
import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from _target import CKPT, core, requires_checkpoint

from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default import (  # noqa: E501
    vision as V,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default import (  # noqa: E501
    weights as W,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.dims import (  # noqa: E501
    dims,
)
from tensorrt_llm._torch.modeling_v2.models.hyv4_flash.hy4_flash_fp8__sm_103__default.vision import (  # noqa: E501
    VISION_CALL_MAX_PATCHES,
    _chunk_rows,
    _frame_calls,
    _frame_rows,
    _tower_rows,
    multimodal_embeds,
    validate_request,
)
from tensorrt_llm._torch.models import modeling_hyv4_flash as M

pytestmark = requires_checkpoint

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

#: Two images the image processor resizes to nothing: patch- and merge-aligned and inside the
#: pixel budget, so the grids are (1, 4, 6) and (1, 8, 4) and the pixels are the raw ones.
IMAGE_SIZES = ((64, 96), (128, 64))


def _processor():
    """The checkpoint's own processor, as the engine's LLM API resolves it."""
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(CKPT)


@functools.cache
def _image_media() -> tuple:
    """Every test image through the checkpoint's image processor: `(pixels, grid rows)`."""
    image_processor = _processor().image_processor
    out = []
    for seed, (height, width) in enumerate(IMAGE_SIZES):
        raw = _raw_image(height, width, seed)
        batch = image_processor(images=raw, return_tensors="pt")
        grid = [tuple(int(v) for v in row) for row in batch["image_grid_thw"].tolist()]
        assert grid == [(1, height // 16, width // 16)], grid
        out.append((batch["pixel_values"], grid))
    return tuple(out)


def _raw_image(height: int, width: int, seed: int) -> torch.Tensor:
    """A deterministic `[3, H, W]` uint8 image with both low and high spatial frequencies."""
    y, x = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing="ij")
    rgb = torch.stack(
        [
            torch.sin(6 * x + seed),
            torch.cos(5 * y - seed),
            torch.sin(3 * (x + y) + 0.5 * seed),
        ]
    )
    rgb = 0.5 + 0.5 * rgb
    rgb[:, : height // 2, : width // 2] = 0.25 * (seed + 1)
    return (rgb.clamp(0, 1) * 255).to(torch.uint8)


def _video_media() -> tuple:
    """The second image as a two-frame video item: flat patch rows, one grid row."""
    pixels, grid = _image_media()[1]
    _, height, width = grid[0]
    return torch.cat([pixels, pixels.flip(0)], 0), [(2, height, width)]


# ---------------------------------------------------------------- weights and geometry


@functools.cache
def _checkpoint_tower_tensors() -> tuple:
    """`(vit.* keys, tensors)` of the checkpoint, reading only the shards that hold them."""
    from safetensors import safe_open

    with open(os.path.join(CKPT, "model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    keys = {key for key in weight_map if key.startswith("vit.")}
    tensors = {}
    for shard in sorted({weight_map[key] for key in keys}):
        with safe_open(os.path.join(CKPT, shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in keys:
                    tensors[key] = f.get_tensor(key)
    assert set(tensors) == keys, sorted(keys - set(tensors))[:8]
    return keys, tensors


@functools.cache
def _core():
    """`_target.core()` with the `dims` the weight table reads: `weights.manifest` and
    `weights.read` take the core, and `dims` takes the model config it carries."""
    stand_in = core()
    stand_in.dims = dims(stand_in)
    return stand_in


def _tower_keys(core_) -> set:
    """`declare`'s keys for the tower: the global `vis_*` entries by name, a block's as
    `l{i}_vis_*`, the `l{i}_` prefix the table gives every layer-keyed parameter."""
    return {
        key
        for entry in W.MODEL_WEIGHTS.WEIGHTS
        if entry.name.startswith("vis_")
        for key, _ in W.MODEL_WEIGHTS._targets(entry, core_.dims)
    }


def _tower_weights(core_, device=None) -> dict:
    """The `vis_*` parameters as `declare` allocates them, empty and unfilled.

    `device` is where `declare`'s parameters end up in the engine; the tower's own ops need
    the weights on the pixels' device, so the numeric tests build them on CUDA.
    """
    d = core_.dims
    w = {}
    for entry in W.MODEL_WEIGHTS.WEIGHTS:
        if not entry.name.startswith("vis_"):
            continue
        dtype = entry.dtype or d.dtype
        for key, i in W.MODEL_WEIGHTS._targets(entry, d):
            w[key] = torch.empty(*W.MODEL_WEIGHTS._shape(entry, d, i), dtype=dtype, device=device)
    return w


def _fill_tower_weights(w: dict, core_, tensors: dict) -> dict:
    """Every tower parameter of its manifest rows, through the loader's own cut and transform:
    `vis_qkv` is the checkpoint's three separate projections packed, so the row layout
    `is_fused_qkv` reads is checked here, and a wrong packing order is a wrong tower."""
    tower = _tower_keys(core_)
    for key, sources in W.MODEL_WEIGHTS.manifest(core_).items():
        if key not in tower:
            continue
        for ckpt_key, index, cut, transform in sources:
            dst = w[key] if index is None else w[key][index]
            src = W.MODEL_WEIGHTS.read(tensors, ckpt_key, cut, transform, core_, dst.device)
            assert dst.shape == src.shape, (ckpt_key, tuple(dst.shape), tuple(src.shape))
            assert src.dtype == dst.dtype, (ckpt_key, src.dtype, dst.dtype)
            dst.copy_(src)
    return w


@functools.cache
def _tower(device=None) -> V.VisionTower:
    """The target's tower over the checkpoint's own vision tensors.

    Built on the host by default, which is all the construction checks need; `device="cuda"`
    is the tower the numeric tests run, where every `read` moves its source to that device.
    """
    core_ = _core()
    _, tensors = _checkpoint_tower_tensors()
    return V.VisionTower(
        _fill_tower_weights(_tower_weights(core_, device), core_, tensors), core_.dims.vision
    )


def test_manifest_sources_are_exactly_the_checkpoint_tower():
    """One manifest row per `vit.*` tensor, no tensor twice, none invented."""
    core_ = _core()
    keys, _ = _checkpoint_tower_tensors()
    manifest = W.MODEL_WEIGHTS.manifest(core_)
    tower = _tower_keys(core_)
    sources = [
        ckpt_key for param, rows in manifest.items() if param in tower for ckpt_key, _, _, _ in rows
    ]
    assert len(sources) == len(set(sources)) == len(keys)
    assert set(sources) == keys, sorted(keys ^ set(sources))[:8]


def test_the_tower_accepts_the_checkpoint_tensors():
    """Building the tower is itself a check: the position table's rows, the LayerNorm eps and
    the block count are asserted inside `__init__` against the loaded tensors."""
    tower = _tower()
    v = dims(core()).vision
    assert (tower.merge, tower.patch, tower.out_hidden, tower.pixel_width) == (
        v.merge,
        v.patch,
        v.out_hidden,
        v.pixel_width,
    )
    assert tower.dtype == torch.bfloat16
    assert VISION_CALL_MAX_PATCHES == v.position_edge * v.position_edge


def test_frame_rows_are_the_processors_placeholder_count():
    """One frame's rows: the tower's own formula against the placeholder run the checkpoint's
    processor counts for the same image, and the pixel rows it hands the tower."""
    v = dims(core()).vision
    processor = _processor()
    assert isinstance(processor, M.HYV4FlashProcessor), type(processor)
    for pixels, grid in _image_media():
        frames, height, width = grid[0]
        assert frames == 1
        assert pixels.shape == (height * width, v.pixel_width)
        counted = processor._get_num_multimodal_tokens(
            image_sizes=[(height * v.patch, width * v.patch)]
        )["num_image_tokens"][0]
        assert counted == _frame_rows(frames, height, width, v.merge)


# ---------------------------------------------------------------------------------- rows


class _StubTower:
    """Tower stand-in: a merged row carries the frame's patch at its own merged column, and
    the newline column carries the frame's last patch, negated. Every row of a frame is a
    function of that frame's own pixels, as the tower's rows are."""

    merge = 2
    patch = 16
    pixel_width = 3 * 16 * 16

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        assert sum(t * h * w for t, h, w in grid_thw.tolist()) == pixel_values.shape[0]
        rows, start = [], 0
        for frames, height, width in grid_thw.tolist():
            for _ in range(frames):
                frame = pixel_values[start : start + height * width]
                start += height * width
                merged = frame.view(height, width, -1)[:: self.merge, :: self.merge]
                newline = -frame[-1:].expand(merged.shape[0], 1, frame.shape[-1])
                rows.append(torch.cat([merged, newline], 1).reshape(-1, frame.shape[-1]))
        return torch.cat(rows, 0)


def _items(pixels: torch.Tensor, grids) -> list:
    """The `(pixel_values, first patch row, grid)` items `_tower_rows` takes, in order."""
    items, start = [], 0
    for grid in grids:
        items.append((pixels, start, grid))
        start += grid[0] * grid[1] * grid[2]
    return items


def test_frame_calls_split_between_frames_only():
    """A call is whole frames under the ceiling, and every patch row lands in exactly one."""
    calls = _frame_calls([(5, 64, 64)], VISION_CALL_MAX_PATCHES)
    assert calls == [[(4, 64, 64)], [(1, 64, 64)]]
    assert sum(t * h * w for call in calls for t, h, w in call) == 5 * 64 * 64
    # Frames of one item keep their order across the split, and a frame is never cut.
    assert all(h * w <= VISION_CALL_MAX_PATCHES for call in calls for _, h, w in call)
    # One frame over the ceiling has no valid call to go in.
    with pytest.raises(AssertionError):
        _frame_calls([(1, 256, 256)], VISION_CALL_MAX_PATCHES)


def test_chunked_rows_are_the_one_pass_rows():
    """The executor's chunk walk over one request: `_chunk_rows` in order, sharing the request's
    own dict, reassembles bitwise what one call over the whole prompt produces. Each row of the
    stub's output is a function of its own frame, so a slice off by a frame or a row shows."""
    tower = _StubTower()
    grids = [(2, 4, 4), (2, 8, 4)]
    pixels = torch.arange(2 * 4 * 4 + 2 * 8 * 4, dtype=torch.float32).view(-1, 1)
    items = _items(pixels, grids)
    total = sum(_frame_rows(*grid, tower.merge) for grid in grids)
    one_pass = _tower_rows(tower, items, 0, total)[1]
    calls_one_pass = tower.calls

    data: dict = {}
    got = []
    for start in range(0, total, 7):
        count = min(7, total - start)
        got.append(_chunk_rows(tower, data, items, start, count))
    assert torch.equal(torch.cat(got, 0), one_pass)
    frames = sum(t for t, _, _ in grids)
    assert tower.calls - calls_one_pass <= frames, "one tower call per frame, never one per chunk"
    assert not data, "the last chunk consumes the parked rows"


def test_multimodal_embeds_splices_items_in_prompt_order():
    """Two modalities in one step: the rows come out in `item_refs` order, paired with the `[R]`
    placeholder rows they replace."""
    from tensorrt_llm.inputs.multimodal import (
        MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
        MultimodalItemMetadata,
        MultimodalParams,
    )

    tower = _StubTower()
    grids = {"video": [(2, 4, 4)], "image": [(1, 4, 4), (1, 2, 2)]}
    data = {}
    for modality, (pixel_key, grid_key) in V.MEDIA_KEYS.items():
        rows = sum(t * h * w for t, h, w in grids[modality])
        data[modality] = {
            pixel_key: torch.zeros(rows, tower.pixel_width),
            grid_key: torch.tensor(grids[modality]),
        }
    # Item `i` of a modality holds its own tag in every patch row: 100 for the video, 200 and
    # 300 for the two images, so the merged rows read back as the order they were spliced in.
    for modality, (pixel_key, grid_key) in V.MEDIA_KEYS.items():
        start = 0
        for i, (t, h, w) in enumerate(grids[modality]):
            data[modality][pixel_key][start : start + t * h * w, 0] = 100.0 * (i + 1)
            start += t * h * w
    data[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = MultimodalItemMetadata(
        item_refs=[("video", 0), ("image", 0), ("image", 1)]
    )

    spliced = [
        (_frame_rows(*grids[modality][i], tower.merge), 100.0 * (i + 1))
        for modality, i in (("video", 0), ("image", 0), ("image", 1))
    ]
    # A second request's media follows the first's, and a payload holding one modality
    # needs no `item_refs`: its items splice in payload order.
    following = [(_frame_rows(*grids["image"][i], tower.merge), 100.0 * (i + 1)) for i in range(2)]
    total = sum(rows for rows, _ in spliced)
    kwargs = {
        "multimodal_params": [
            MultimodalParams(multimodal_data=data),
            MultimodalParams(multimodal_data={"image": data["image"]}),
        ],
        "mm_token_indices": torch.arange(11, 11 + total + sum(r for r, _ in following)),
    }
    rows, indices = multimodal_embeds(tower, kwargs)
    assert rows.shape == (total + sum(r for r, _ in following), tower.pixel_width)
    assert indices.tolist() == list(range(11, 11 + rows.shape[0]))
    assert rows[:, 0].abs().tolist() == [  # the stub's newline rows carry the tag negated
        tag for rows_of_item, tag in spliced + following for _ in range(rows_of_item)
    ]
    # Frames of one modality are one call's pixels; the two params objects are two calls.
    assert tower.calls == 2


def test_multimodal_embeds_declines_steps_without_media():
    """A decode step, a text-only step and a chunk whose rows all lie elsewhere run no tower:
    the executor ships no media for them, or a runtime that says so."""
    from tensorrt_llm.inputs.multimodal import MultimodalParams

    tower = _StubTower()
    media = {
        "image": {
            "pixel_values": torch.zeros(32, tower.pixel_width),
            "image_grid_thw": torch.tensor([[2, 4, 4]]),
        }
    }
    assert multimodal_embeds(tower, {}) is None
    assert (
        multimodal_embeds(tower, {"multimodal_params": [MultimodalParams(multimodal_data={})]})
        is None
    )
    elsewhere = MultimodalParams(
        multimodal_data=media,
        multimodal_runtime=SimpleNamespace(num_mm_tokens_in_chunk=0),
    )
    assert multimodal_embeds(tower, {"multimodal_params": [elsewhere]}) is None
    assert tower.calls == 0
    # The executor ships the placeholder indices with the media; without them the rows have
    # nowhere to go.
    with pytest.raises(AssertionError):
        multimodal_embeds(tower, {"multimodal_params": [MultimodalParams(multimodal_data=media)]})


# ------------------------------------------------------------------------ admission


def _request(pixels, grid, **extra) -> SimpleNamespace:
    data = {"image": {"pixel_values": pixels, "image_grid_thw": torch.tensor(grid)}}
    data.update(extra)
    return SimpleNamespace(py_multimodal_data=data)


def test_validate_request_refuses_what_the_forward_would_assert():
    """The executor runs this on every new request, from the host payload alone: a payload it
    lets through and the forward refuses would end the executor."""
    v = dims(core()).vision
    tower = SimpleNamespace(merge=v.merge, patch=v.patch, pixel_width=v.pixel_width)
    pixels = torch.zeros(4 * 6, v.pixel_width)
    grid = [[1, 4, 6]]
    rows = _frame_rows(1, 4, 6, v.merge)
    run = torch.cat([torch.arange(1, rows + 1), torch.full((5,), rows)])

    validate_request(tower, _request(pixels, grid))
    validate_request(tower, _request(pixels, grid, multimodal_embed_mask_cumsum=run))

    for case in (
        # A prompt whose placeholder run is not what its grids produce.
        _request(pixels, grid, multimodal_embed_mask_cumsum=run + 1),
        # Precomputed embeddings: the tower cannot splice them.
        _request(pixels, grid, multimodal_embedding=torch.zeros(rows, v.out_hidden)),
        # Sides that do not divide by the merge size.
        _request(pixels[: 3 * 6], [[1, 3, 6]]),
        # Patch rows that are not the grid's.
        _request(pixels[:5], grid),
        # A grid row that is not (t, h, w).
        _request(pixels, [[1, 4]]),
        # A grid that is not one row per image.
        _request(pixels, []),
    ):
        with pytest.raises(ValueError):
            validate_request(tower, case)

    # A frame over the tower's per-call ceiling: the pixel tensor is far too small, but the
    # ceiling is checked before the row count, and it is the error that matters.
    with pytest.raises(ValueError):
        validate_request(tower, _request(pixels, [[1, 129, 128]]))


def test_validate_request_refuses_a_second_modality_without_item_order():
    """Two modalities arrive as separate payloads; only `item_refs` says how they interleave."""
    from tensorrt_llm.inputs.multimodal import (
        MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
        MultimodalItemMetadata,
    )

    v = dims(core()).vision
    tower = SimpleNamespace(merge=v.merge, patch=v.patch, pixel_width=v.pixel_width)
    image = {
        "pixel_values": torch.zeros(4 * 6, v.pixel_width),
        "image_grid_thw": torch.tensor([[1, 4, 6]]),
    }
    video = {
        "pixel_values_videos": torch.zeros(2 * 2 * 2, v.pixel_width),
        "video_grid_thw": torch.tensor([[2, 2, 2]]),
    }
    rows = _frame_rows(1, 4, 6, v.merge) + _frame_rows(2, 2, 2, v.merge)
    run = torch.cat([torch.arange(1, rows + 1), torch.full((5,), rows)])

    both = {"image": image, "video": video, "multimodal_embed_mask_cumsum": run}
    with pytest.raises(ValueError):
        validate_request(tower, SimpleNamespace(py_multimodal_data=both))
    # The same payload with an order that lists every item exactly once is admitted.
    both[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = MultimodalItemMetadata(
        item_refs=[("video", 0), ("image", 0)]
    )
    validate_request(tower, SimpleNamespace(py_multimodal_data=both))
    # ... and one that leaves an item out is not.
    both[MULTIMODAL_ENCODER_ITEM_METADATA_KEY] = MultimodalItemMetadata(
        item_refs=[("video", 0), ("image", 0), ("image", 0)]
    )
    with pytest.raises(ValueError):
        validate_request(tower, SimpleNamespace(py_multimodal_data=both))


def test_validate_request_ignores_a_text_only_request():
    tower = SimpleNamespace(merge=2, patch=16, pixel_width=768)
    validate_request(tower, SimpleNamespace(py_multimodal_data={}))
    validate_request(
        tower,
        SimpleNamespace(py_multimodal_data={"image": None, "multimodal_embed_mask_cumsum": None}),
    )


# ------------------------------------------------------------------- the tower's math


def _reference(v, tensors: dict, grid, pixel_values, dtype) -> torch.Tensor:
    """`HYV4VLVisionModel.forward` transcribed from `modeling_hy_v4_vl.py`.

    The patch embedding is `nn.Conv2d(3, hidden, patch, stride=patch)`; the position table is
    one `nn.Embedding(position_edge**2 + 1, hidden)` read from row 1 (the cls row is skipped)
    and resampled bilinearly with the reference's `scale_factor = (size + 0.1) / position_edge`
    at `align_corners=False`; each block is pre-norm with `is_causal=False` attention over one
    sequence per frame; the merger is a pre-norm RMSNorm, a 2x2 conv, the erf GELU, a 1x1 conv,
    and one newline column per merged row. The checkpoint's own `q_proj`/`k_proj`/`v_proj` feed
    this reference, while the tower reads the packed rows `weights.py` builds from them.
    """
    hidden, heads, head_dim = v.hidden, v.heads, v.head_dim

    def weight(key):
        """The checkpoint tensor on the pixels' device, still in its own dtype."""
        return tensors[key].to(pixel_values.device)

    patches = pixel_values.to(dtype).reshape(-1, v.channels, v.patch, v.patch)
    seq = F.conv2d(
        patches,
        weight("vit.embeddings.patch_embedding.weight").to(dtype),
        weight("vit.embeddings.patch_embedding.bias").to(dtype),
    ).flatten(1)

    table = weight("vit.embeddings.position_embedding.weight")[1:].float()
    table = table.reshape(v.position_edge, v.position_edge, hidden).permute(2, 0, 1)[None]
    positions = []
    for t, height, width in grid:
        resampled = F.interpolate(
            table,
            scale_factor=((height + 0.1) / v.position_edge, (width + 0.1) / v.position_edge),
            mode="bilinear",
            align_corners=False,
        )
        assert resampled.shape[-2:] == (height, width), (resampled.shape, (height, width))
        positions.append(resampled.permute(0, 2, 3, 1).reshape(-1, hidden).repeat(t, 1))
    seq = (seq + torch.cat(positions).to(dtype))[None]

    segments = [height * width for t, height, width in grid for _ in range(t)]
    for layer in range(v.depth):
        at = f"vit.layers.{layer}."
        normed = F.layer_norm(
            seq,
            (hidden,),
            weight(at + "input_layernorm.weight").to(dtype),
            weight(at + "input_layernorm.bias").to(dtype),
            v.eps,
        )
        q, k, value = (
            F.linear(
                normed,
                weight(at + f"self_attn.{name}_proj.weight").to(dtype),
                weight(at + f"self_attn.{name}_proj.bias").to(dtype),
            )
            .view(1, -1, heads, head_dim)
            .transpose(1, 2)
            for name in ("q", "k", "v")
        )
        outs, start = [], 0
        for length in segments:
            outs.append(
                F.scaled_dot_product_attention(
                    q[:, :, start : start + length],
                    k[:, :, start : start + length],
                    value[:, :, start : start + length],
                    is_causal=False,
                    scale=head_dim**-0.5,
                )
            )
            start += length
        attn = torch.cat(outs, 2).transpose(1, 2).reshape(1, -1, hidden)
        seq = seq + F.linear(
            attn,
            weight(at + "self_attn.o_proj.weight").to(dtype),
            weight(at + "self_attn.o_proj.bias").to(dtype),
        )
        normed = F.layer_norm(
            seq,
            (hidden,),
            weight(at + "post_attention_layernorm.weight").to(dtype),
            weight(at + "post_attention_layernorm.bias").to(dtype),
            v.eps,
        )
        normed = F.gelu(
            F.linear(
                normed,
                weight(at + "mlp.dense_h_to_4h.weight").to(dtype),
                weight(at + "mlp.dense_h_to_4h.bias").to(dtype),
            )
        )
        seq = seq + F.linear(
            normed,
            weight(at + "mlp.dense_4h_to_h.weight").to(dtype),
            weight(at + "mlp.dense_4h_to_h.bias").to(dtype),
        )

    # `HYV4VLRMSNorm`: the variance in fp32, the scale on the rounded value. No bias.
    seq = seq.to(torch.float32)
    seq = weight("vit.perceive.norm.weight").to(dtype) * (
        seq * torch.rsqrt(seq.pow(2).mean(-1, keepdim=True) + v.eps)
    ).to(dtype)
    seq = seq[0]

    items, start = [], 0
    for t, height, width in grid:
        frame = seq[start : start + t * height * width].reshape(t, height, width, hidden)
        start += t * height * width
        merged = F.gelu(
            F.conv2d(
                frame.permute(0, 3, 1, 2),
                weight("vit.perceive.proj.0.weight").to(dtype),
                weight("vit.perceive.proj.0.bias").to(dtype),
                stride=v.merge,
            )
        )
        merged = F.conv2d(
            merged,
            weight("vit.perceive.proj.2.weight").to(dtype),
            weight("vit.perceive.proj.2.bias").to(dtype),
        )
        frames, channels, rows, cols = merged.shape
        newline = weight("vit.perceive.image_newline").to(dtype).reshape(1, channels, 1, 1)
        merged = torch.cat([merged, newline.expand(frames, channels, rows, 1)], dim=-1)
        items.append(merged.reshape(frames, channels, -1).permute(0, 2, 1).reshape(-1, channels))
    return torch.cat(items, 0)


@contextlib.contextmanager
def _ieee_fp32():
    """IEEE fp32 for the reference: GEMMs and convolutions off TF32, SDPA on its math backend."""
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
    """Max abs error, that over `max |ref|`, relative L2 and cosine, all in fp64."""
    o, r = out.double().flatten(), ref.double().flatten()
    diff = (o - r).abs()
    return dict(
        max_abs=diff.max().item(),
        max_rel=(diff.max() / r.abs().max()).item(),
        rel_l2=(torch.linalg.vector_norm(o - r) / torch.linalg.vector_norm(r)).item(),
        cos=torch.nn.functional.cosine_similarity(o, r, dim=0).item(),
    )


#: The tower and the bf16 reference are two accumulation orders of the same bf16 math over 27
#: blocks; the fp32 reference is that math with a single rounding per op. Measured on the
#: checkpoint's weights over the three cases below (image0 / image1 / video), with the test's
#: own print: tower vs bf16 rel_l2 9.7e-3 / 1.6e-2 / 1.9e-2 and max_rel 8.5e-3 / 1.6e-2 /
#: 2.8e-2; bf16 vs fp32 rel_l2 1.4e-2 / 1.7e-2 / 1.4e-2 and max_rel 1.0e-2 / 3.7e-2 / 1.5e-2.
#: So the tower sits inside the reference's own rounding noise, and the band is the measured
#: worst with half again: a wrong q|k|v packing, a transposed projection, a missing position
#: table or a misplaced newline is an order of magnitude out, while the accumulation order a
#: different device picks is not.
_BAND = dict(rel_l2=3.0e-2, max_rel=6.0e-2)


@needs_gpu
def test_the_tower_is_the_references_vision_model():
    """The tower's rows against the reference's math, over the checkpoint's own weights.

    The reference runs twice: in fp32, where every op is the reference's own, and in bf16,
    which is the same code at the tower's precision. The tower is gated against the bf16
    reference -- the two differ by accumulation order over 27 blocks, not by math, so a wrong
    q|k|v packing, a transposed projection, a missing position table or a newline in the wrong
    place lands orders of magnitude outside the band -- and both are reported against fp32.
    """
    v = dims(core()).vision
    tower = _tower("cuda")
    _, tensors = _checkpoint_tower_tensors()

    cases = [(f"image{i}", *media) for i, media in enumerate(_image_media())]
    cases.append(("video", *_video_media()))
    for name, pixels, grid in cases:
        with torch.inference_mode(), _ieee_fp32():
            ref32 = _reference(v, tensors, grid, pixels.to("cuda"), torch.float32)
        with torch.inference_mode():
            ref16 = _reference(v, tensors, grid, pixels.to("cuda"), torch.bfloat16)
            out = tower(pixels.to("cuda"), torch.tensor(grid))
        assert out.shape == ref16.shape, (out.shape, ref16.shape)
        assert out.shape[0] == sum(_frame_rows(*item, v.merge) for item in grid)
        measured, noise = _stats(out, ref16), _stats(ref16, ref32)
        print(f"{name}: tower vs bf16 {measured} | bf16 vs fp32 {noise}")
        assert measured["rel_l2"] <= _BAND["rel_l2"], (name, measured, noise)
        assert measured["max_rel"] <= _BAND["max_rel"], (name, measured, noise)


@needs_gpu
def test_a_split_call_reproduces_the_whole_call_row_for_row():
    """Frames are independent in every op the tower runs, which is what chunked prefill and
    block reuse cut on: two frames in one call and in two calls are bitwise the same rows."""
    v = dims(core()).vision
    tower = _tower("cuda")
    pixels, grid = _image_media()[1]
    _, height, width = grid[0]
    both = torch.cat([pixels, pixels.flip(0)], 0).to("cuda")
    whole = tower(both, torch.tensor([(2, height, width)]))
    first = tower(both[: height * width], torch.tensor([(1, height, width)]))
    second = tower(both[height * width :], torch.tensor([(1, height, width)]))
    rows = _frame_rows(1, height, width, v.merge)
    assert whole.shape == (2 * rows, v.out_hidden)
    assert torch.equal(whole[:rows], first)
    assert torch.equal(whole[rows:], second)
