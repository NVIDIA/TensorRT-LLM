# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The vision tower against HF `HYV4VLVisionModel`'s math over the checkpoint's weights, and the
multimodal splice's row bookkeeping."""

import contextlib
import functools
import json
import os

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
    _chunk_rows,
    _frame_rows,
    _tower_rows,
    multimodal_embeds,
)

pytestmark = requires_checkpoint

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

#: Patch- and merge-aligned images the processor does not resize: grids (1, 4, 6) and (1, 8, 4).
IMAGE_SIZES = ((64, 96), (128, 64))


@functools.cache
def _image_media() -> tuple:
    """Every test image through the checkpoint's image processor: `(pixels, grid rows)`."""
    from transformers import AutoProcessor

    image_processor = AutoProcessor.from_pretrained(CKPT).image_processor
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
    """The second image as a two-frame video item."""
    pixels, grid = _image_media()[1]
    _, height, width = grid[0]
    return torch.cat([pixels, pixels.flip(0)], 0), [(2, height, width)]


@functools.cache
def _checkpoint_tower_tensors() -> dict:
    """The checkpoint's `vit.*` tensors, reading only the shards that hold them."""
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
    return tensors


@functools.cache
def _tower() -> V.VisionTower:
    """The target's tower on CUDA, filled through the loader's own manifest, cut and transform."""
    core_ = core()
    core_.dims = dims(core_)
    d = core_.dims
    w = {}
    for entry in W.MODEL_WEIGHTS.WEIGHTS:
        if not entry.name.startswith("vis_"):
            continue
        dtype = entry.dtype or d.dtype
        for key, i in W.MODEL_WEIGHTS._targets(entry, d):
            w[key] = torch.empty(*W.MODEL_WEIGHTS._shape(entry, d, i), dtype=dtype, device="cuda")
    tensors = _checkpoint_tower_tensors()
    for key, sources in W.MODEL_WEIGHTS.manifest(core_).items():
        if key not in w:
            continue
        for ckpt_key, index, cut, transform in sources:
            dst = w[key] if index is None else w[key][index]
            src = W.MODEL_WEIGHTS.read(tensors, ckpt_key, cut, transform, core_, dst.device)
            assert dst.shape == src.shape and src.dtype == dst.dtype, ckpt_key
            dst.copy_(src)
    return V.VisionTower(w, d.vision)


def _reference(v, tensors: dict, grid, pixel_values, dtype) -> torch.Tensor:
    """`HYV4VLVisionModel.forward`, transcribed from `modeling_hy_v4_vl.py`, on the checkpoint's
    separate q/k/v projections."""
    hidden, heads, head_dim = v.hidden, v.heads, v.head_dim

    def weight(key):
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
    o, r = out.double().flatten(), ref.double().flatten()
    diff = (o - r).abs()
    return dict(
        max_rel=(diff.max() / r.abs().max()).item(),
        rel_l2=(torch.linalg.vector_norm(o - r) / torch.linalg.vector_norm(r)).item(),
    )


#: The tower vs the bf16 reference differ by accumulation order over 27 blocks; a wrong q|k|v
#: packing, transposed projection, missing position table or misplaced newline is far outside.
_BAND = dict(rel_l2=3.0e-2, max_rel=6.0e-2)


@needs_gpu
def test_the_tower_is_the_references_vision_model():
    """The tower against the bf16 reference (gated) and the fp32 reference (reported)."""
    v = dims(core()).vision
    tower = _tower()
    tensors = _checkpoint_tower_tensors()

    cases = [(f"image{i}", *media) for i, media in enumerate(_image_media())]
    cases.append(("video", *_video_media()))
    for name, pixels, grid in cases:
        with torch.inference_mode(), _ieee_fp32():
            ref32 = _reference(v, tensors, grid, pixels.to("cuda"), torch.float32)
        with torch.inference_mode():
            ref16 = _reference(v, tensors, grid, pixels.to("cuda"), torch.bfloat16)
            out = tower(pixels.to("cuda"), torch.tensor(grid))
        assert out.shape == ref16.shape, (out.shape, ref16.shape)
        measured, noise = _stats(out, ref16), _stats(ref16, ref32)
        print(f"{name}: tower vs bf16 {measured} | bf16 vs fp32 {noise}")
        assert measured["rel_l2"] <= _BAND["rel_l2"], (name, measured, noise)
        assert measured["max_rel"] <= _BAND["max_rel"], (name, measured, noise)


class _StubTower:
    """A merged row carries its frame's patch at its merged column; the newline column carries the
    frame's last patch, negated. Every row is a function of its own frame's pixels."""

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


def test_chunked_rows_are_the_one_pass_rows():
    """`_chunk_rows` over a request's chunks reassembles bitwise the one-pass rows, one call per frame."""
    tower = _StubTower()
    grids = [(2, 4, 4), (2, 8, 4)]
    pixels = torch.arange(2 * 4 * 4 + 2 * 8 * 4, dtype=torch.float32).view(-1, 1)
    items, start = [], 0
    for grid in grids:
        items.append((pixels, start, grid))
        start += grid[0] * grid[1] * grid[2]
    total = sum(_frame_rows(*grid, tower.merge) for grid in grids)
    one_pass = _tower_rows(tower, items, 0, total)[1]
    calls_one_pass = tower.calls

    data: dict = {}
    got = []
    for start in range(0, total, 7):
        count = min(7, total - start)
        got.append(_chunk_rows(tower, data, items, start, count))
    assert torch.equal(torch.cat(got, 0), one_pass)
    assert tower.calls - calls_one_pass <= sum(t for t, _, _ in grids)
    assert not data, "the last chunk consumes the parked rows"


def test_multimodal_embeds_splices_items_in_prompt_order():
    """Two modalities in one request come out in `item_refs` order; a second request follows."""
    from tensorrt_llm.inputs.multimodal import (
        MULTIMODAL_ENCODER_ITEM_METADATA_KEY,
        MultimodalItemMetadata,
        MultimodalParams,
    )

    tower = _StubTower()
    grids = {"video": [(2, 4, 4)], "image": [(1, 4, 4), (1, 2, 2)]}
    data = {}
    # Item `i` of a modality tags every patch row with 100 * (i + 1).
    for modality, (pixel_key, grid_key) in V.MEDIA_KEYS.items():
        rows = sum(t * h * w for t, h, w in grids[modality])
        data[modality] = {
            pixel_key: torch.zeros(rows, tower.pixel_width),
            grid_key: torch.tensor(grids[modality]),
        }
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
    following = [(_frame_rows(*grids["image"][i], tower.merge), 100.0 * (i + 1)) for i in range(2)]
    total = sum(rows for rows, _ in spliced + following)
    kwargs = {
        "multimodal_params": [
            MultimodalParams(multimodal_data=data),
            MultimodalParams(multimodal_data={"image": data["image"]}),
        ],
        "mm_token_indices": torch.arange(11, 11 + total),
    }
    rows, indices = multimodal_embeds(tower, kwargs)
    assert rows.shape == (total, tower.pixel_width)
    assert indices.tolist() == list(range(11, 11 + total))
    assert rows[:, 0].abs().tolist() == [
        tag for rows_of_item, tag in spliced + following for _ in range(rows_of_item)
    ]
