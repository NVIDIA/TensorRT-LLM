# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The FP8 unified target's vision path: the tower's two entries over their certified cells, the
tower against HuggingFace on the checkpoint's weights, request admission and the multimodal splice.
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
)

CHECKPOINT = "Qwen3.8-Flash-Next-FP8"

Shell = M.ModelingV2Qwen38FlashNextFp8Sm103Unified

_VISION = dict(
    deepstack_visual_indexes=[], depth=27, dtype="bfloat16", hidden_act="gelu_pytorch_tanh",
    hidden_size=1152, in_channels=3, intermediate_size=4304, num_heads=16,
    num_position_embeddings=2304, out_hidden_size=2560, patch_size=16, spatial_merge_size=2,
    temporal_patch_size=2,
)  # fmt: skip
HIDDEN_OUT = 2560

_needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")


def _dims(vcfg) -> SimpleNamespace:
    """The vision widths of `weights.rank_dims` (the tower is replicated: every rank's)."""
    return SimpleNamespace(
        hidden=HIDDEN_OUT, dtype=torch.bfloat16, v_depth=vcfg.depth, v_hidden=vcfg.hidden_size,
        v_inter=vcfg.intermediate_size, v_channels=vcfg.in_channels,
        v_tpatch=vcfg.temporal_patch_size, v_patch=vcfg.patch_size,
        v_pos_rows=vcfg.num_position_embeddings,
        v_merge_dim=vcfg.hidden_size * vcfg.spatial_merge_size**2,
    )  # fmt: skip


def _vision_entries():
    return [entry for entry in W.MODEL_WEIGHTS.WEIGHTS if entry.name.startswith("vis_")]


def _tower_over_the_weight_table() -> V.VisionTower:
    """The tower over the table's `vis_*` weights on meta; `embed` only supplies the dtype."""
    vcfg = Qwen4ExpVisionConfig(**_VISION)
    d = _dims(vcfg)
    w = {"embed": torch.empty(8, HIDDEN_OUT, dtype=torch.bfloat16, device="meta")}
    for entry in _vision_entries():
        for key, _, _ in W.MODEL_WEIGHTS._targets(entry, d):
            w[key] = torch.empty(*entry.shape(d), dtype=entry.dtype or d.dtype, device="meta")
    return V.VisionTower(w, vcfg)


MERGE = 2
THETA = 10000.0


def _num_tokens(grid) -> int:
    return sum(t * h * w for t, h, w in grid)


def _spatial_merge_positions(grid: Sequence[Sequence[int]], merge: int) -> List[Tuple[int, int]]:
    """Per-token `(row, col)` in spatial-merge order, by plain nested loops."""
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
    """fp32 `cos` / `sin` `[T, head_dim // 2]`, height then width, rounded once from fp64."""
    rope_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float64) / rope_dim))
    pos = torch.tensor(_spatial_merge_positions(grid_thw.tolist(), merge), dtype=torch.float64)
    angles = (pos[:, :, None] * inv_freq).flatten(1).to(device=device, dtype=torch.float32)
    return angles.cos().contiguous(), angles.sin().contiguous()


HIDDEN = 3

IMAGE, VIDEO = 1, 2


class _Tower:
    """Stub `VisionTower`: a merged row carries its first patch's pixel value (`_media`)."""

    merge = MERGE

    def __init__(self):
        self.calls = 0
        self.patches = 0

    def __call__(self, pixel_values, grid_thw):
        self.calls += 1
        self.patches += pixel_values.shape[0]
        assert sum(t * h * w for t, h, w in grid_thw.tolist()) == pixel_values.shape[0]
        return pixel_values[:: MERGE * MERGE].expand(-1, HIDDEN)


def _media(modality, grids):
    """One `multimodal_data` entry; every patch row of item `i` holds `100 * modality + i`."""
    key = "pixel_values" if modality == IMAGE else "pixel_values_videos"
    grid_key = "image_grid_thw" if modality == IMAGE else "video_grid_thw"
    pixels = torch.cat(
        [
            torch.full((t * h * w, 1), float(100 * modality + item))
            for item, (t, h, w) in enumerate(grids)
        ]
    )
    return {key: pixels, grid_key: torch.tensor(grids)}


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


# Two videos of 4 and 8 rows, two images of 4 and 1.
VIDEOS = [(2, 2, 4), (2, 4, 4)]
IMAGES = [(1, 4, 4), (1, 2, 2)]
MIXED_ORDER = [("video", 0), ("image", 0), ("video", 1), ("image", 1)]
MIXED_ROWS = [200.0] * 4 + [100.0] * 4 + [201.0] * 8 + [101.0]


class _StepBegun(Exception):
    """Raised by the stub core's `begin_step`, so the shell's forward stops there."""


def _forward_to_begin_step(tower, kwargs):
    """The `vision` argument the shell's forward hands `begin_step` for a step with `kwargs`."""
    seen = {}

    def begin_step(attn_metadata, input_ids, position_ids, **step):
        seen.update(step)
        raise _StepBegun

    shell = SimpleNamespace(model=SimpleNamespace(begin_step=begin_step), vision_tower=tower)
    with pytest.raises(_StepBegun):
        Shell.forward(shell, None, torch.zeros(1, dtype=torch.int32), None, **kwargs)
    return seen["vision"]


TEXT_TOKENS = 5


@functools.cache
def _assembled_tower() -> V.VisionTower:
    return _tower_over_the_weight_table()


def _request_media(modality, grids):
    """A payload whose pixel tensor has the assembled tower's patch width and dtype and no
    storage."""
    patches = sum(t * h * w for t, h, w in grids)
    key = "pixel_values" if modality == "image" else "pixel_values_videos"
    grid_key = "image_grid_thw" if modality == "image" else "video_grid_thw"
    tower = _assembled_tower()
    return {
        key: torch.empty(patches, tower.pixel_width, dtype=tower.dtype, device="meta"),
        grid_key: torch.tensor(grids),
    }


def _request(image=None, video=None, order=None, **overrides):
    """Placeholders plus `TEXT_TOKENS` text tokens, with the mrope ids and placeholder cumsum."""
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


def _whole_op_reference(q, k, grid_thw, spatial_merge_size, theta=THETA):
    """`VisionRope2d.reference` verbatim: the whole op from the grid, as fresh `(q_out, k_out)`."""
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


def _mismatch(got: torch.Tensor, want: torch.Tensor) -> Tuple[int, float]:
    """(elements differing in their bits, largest absolute difference)."""
    differing = (got.reshape(-1).view(torch.int16) != want.reshape(-1).view(torch.int16)).sum()
    return int(differing.item()), (got.float() - want.float()).abs().max().item()


def _check_rope(cell) -> None:
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

    differing, max_dev = _mismatch(qk, want)
    assert differing == 0, f"{differing} elements differ from reference (max {max_dev:.3e})"

    q_whole, k_whole = _whole_op_reference(
        before[:, :NUM_HEADS], before[:, NUM_HEADS : 2 * NUM_HEADS], grid_thw, MERGE, THETA
    )
    for name, got, ref in (
        ("q", qkv[:, :NUM_HEADS], q_whole),
        ("k", qkv[:, NUM_HEADS : 2 * NUM_HEADS], k_whole),
    ):
        differing, max_dev = _mismatch(got, ref)
        assert differing == 0, (
            f"{name}: {differing} elements differ from the whole op (max {max_dev:.3e})"
        )


#: (height, width) in pixels of the two test images: grids 1x28x42 and 1x14x20.
_IMAGE_SIZES = ((448, 672), (224, 320))
#: Four 256x384 frames: one video item, grid 2x16x24 (two temporal groups).
_VIDEO_FRAMES, _VIDEO_SIZE = 4, (256, 384)
#: SDPA backends of the bf16 HF forward; their spread defines the bf16 error envelope.
_HF_BACKENDS = ("MATH", "EFFICIENT_ATTENTION", "FLASH_ATTENTION", "CUDNN_ATTENTION")
#: Allowed factor over the worst bf16 HF forward: the backends differ among themselves by up to
#: 1.5x on one item, and the tower is one more sample of that spread.
_BAND = 1.25


def _picture(height: int, width: int, seed: int) -> np.ndarray:
    """A deterministic `[H, W, 3]` uint8 picture with both low and high spatial frequencies."""
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
    """IEEE fp32 for the reference: GEMMs and the patch conv off TF32, SDPA on its math backend."""
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
    """`(tower, hf32, hf16)`: this target's tower over the table's `vis_*` sources and HF's
    `Qwen3_5VisionModel` in fp32 and bf16, all on the checkpoint's same vision tensors."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

    vision = json.loads((checkpoint / "config.json").read_text())["vision_config"]
    assert vision["deepstack_visual_indexes"] == [], "the reference has no deepstack mergers"
    vcfg = Qwen4ExpVisionConfig(**{k: v for k, v in vision.items() if k != "model_type"})
    d = _dims(vcfg)
    index = json.loads((checkpoint / "model.safetensors.index.json").read_text())["weight_map"]

    # Each `vis_*` parameter is one whole checkpoint tensor: the tower is replicated and bf16.
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
        assert tuple(tensor.shape) == tuple(shape) and tensor.dtype == dtype, key
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
        # Keep the rotary's `inv_freq` fp32 as `from_pretrained` does; `.to(bf16)` rounds it.
        inv_freq = hf.rotary_pos_emb.inv_freq.clone()
        hf = hf.to(dtype).eval()
        hf.rotary_pos_emb.inv_freq = inv_freq
        out.append(hf)
    return tower, out[0], out[1]


def _load_media(checkpoint):
    """`{name: (pixel_values, grid_thw)}` from the checkpoint's processors for two images and a
    four-frame video, the pixels rounded to bf16 once so every model reads the same values."""
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
            envelope = {
                key: max(st[key] for st in floors.values()) for key in ("max_abs", "rel_l2")
            }
            out[name] = (ref, envelope)
    return out


def _check(name: str, out: torch.Tensor, ref: torch.Tensor, envelope: dict) -> dict:
    """`out` within `_BAND` of the worst bf16 HF forward on every gated statistic."""
    tower = _stats(out, ref)
    assert out.dtype == torch.bfloat16 and out.shape == ref.shape
    assert torch.isfinite(out).all()
    for key in ("rel_l2", "max_abs"):
        assert tower[key] <= _BAND * envelope[key], (name, key, tower, envelope)
    return tower


def _hf_single(name):
    def run(tower, media, references, monkeypatch):
        px, grid = media[name]
        with torch.inference_mode():
            out = tower(px, grid)
        assert out.shape == (int(grid.prod(-1).sum()) // tower.merge**2, HIDDEN_OUT)
        _check(name, out, *references[name])

    return run


def _hf_pair(cap=None):
    """Both images in one call (or one per call under `cap`), each its own attention segment."""

    def run(tower, media, references, monkeypatch):
        (px0, grid0), (px1, grid1) = media["image0"], media["image1"]
        if cap is not None:
            assert V._frame_calls([(1, 28, 42), (1, 14, 20)], cap) == [[(1, 28, 42)], [(1, 14, 20)]]
            monkeypatch.setattr(V, "VISION_CALL_MAX_PATCHES", cap)
        with torch.inference_mode():
            out = tower(torch.cat([px0, px1]), torch.cat([grid0, grid1]))
        split = int(grid0.prod()) // tower.merge**2
        _check("image0 in a pair", out[:split], *references["image0"])
        _check("image1 in a pair", out[split:], *references["image1"])

    return run


def _hf_video_in_two_calls(tower, media, references, monkeypatch):
    """The per-call ceiling lowered to one 384-patch frame: the video runs as two calls."""
    px, grid = media["video"]
    assert V._frame_calls([(2, 16, 24)], 384) == [[(1, 16, 24)], [(1, 16, 24)]]
    monkeypatch.setattr(V, "VISION_CALL_MAX_PATCHES", 384)
    with torch.inference_mode():
        out = tower(px, grid)
    _check("video in two calls", out, *references["video"])


def _hf_video_in_chunks(tower, media, references, monkeypatch):
    """The video spliced over three chunked-prefill chunks whose boundaries cut its two 96-row
    frames: each chunk runs only the frames holding its rows, and together they are the one-pass
    rows bitwise."""
    px, grid = media["video"]
    total = int(grid.prod(-1).sum()) // tower.merge**2
    data = {
        "mrope_config": {"mrope_position_ids": torch.zeros(3, 1, 8, dtype=torch.int32)},
        "video": {"pixel_values_videos": px, "video_grid_thw": grid},
    }
    chunks = []
    with torch.inference_mode():
        whole = tower(px, grid)
        for start, end in ((0, 50), (50, 150), (150, total)):
            runtime = SimpleNamespace(
                num_cached_mm_tokens=start,
                num_mm_tokens_in_chunk=end - start,
                total_embeds_in_request=total,
            )
            params = MultimodalParams(multimodal_data=data, multimodal_runtime=runtime)
            kwargs = {"multimodal_params": [params], "mm_token_indices": torch.arange(end - start)}
            chunks.append(V.multimodal_embeds(tower, kwargs)[0])
    out = torch.cat(chunks)
    assert torch.equal(out, whole), _mismatch(out, whole)
    _check("video in three chunks", out, *references["video"])


_HF_CASES = {
    "image0": _hf_single("image0"),
    "image1": _hf_single("image1"),
    "video": _hf_single("video"),
    "image_pair": _hf_pair(),
    "image_pair_two_calls": _hf_pair(cap=1176),
    "video_two_calls": _hf_video_in_two_calls,
    "video_chunks": _hf_video_in_chunks,
}


@_needs_gpu
@pytest.mark.parametrize("case", list(_HF_CASES))
def test_the_tower_is_huggingfaces_vision_model(models, media, references, case, monkeypatch):
    """Each case lands within the bf16 HF forward's own error of the fp32 HF reference."""
    _HF_CASES[case](models[0], media, references, monkeypatch)


NUM_GRID = 48
VISION_HIDDEN = 1152


def _pos_embed_compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """4 bf16 ulp per element, 1 ulp relative RMS: the op rounds its corner weights to bf16 and
    multiply-adds in bf16, against the exact value (measured worst 3.5 / 0.75)."""
    assert out.dtype == torch.bfloat16, out.dtype
    assert_within_ulp(out.float(), ref, element_ulp=4.0, rms_ulp=1.0, ulp=ULP[torch.bfloat16])


def _pos_table(seed: int, num_grid: int = NUM_GRID, hidden: int = VISION_HIDDEN) -> torch.Tensor:
    """A smooth `[num_grid**2, hidden]` bf16 table, as a learned 2-D position table is."""
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
    return (table, *(spec[k] for k in ("t", "h", "w", "num_grid_per_side", "m_size", "dtype")))


def _check_pos_embed(cell) -> None:
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
    _pos_embed_compare(out, ref)

    # An independent fp64 implementation confirms the reference.
    torch.testing.assert_close(ref, _resize64(table, t, h, w, m_size).float())

    # The band refuses each wrong variant of the function.
    for wrong in (dict(align=False), dict(raster=True), dict(swap=True)):
        with pytest.raises(AssertionError):
            _pos_embed_compare(out, _resize64(table, t, h, w, m_size, **wrong).float())


_ENTRIES = (
    (vision_rope_2d_packed_qk, _check_rope),
    (triton_pos_embed_interpolate, _check_pos_embed),
)
CASES = [(op, check, cell) for op, check in _ENTRIES for cell in op.CELLS]


@_needs_gpu
@pytest.mark.parametrize(
    "entry,check,cell",
    CASES,
    ids=[f"{type(op).__name__}-{i}" for op, _ in _ENTRIES for i in range(len(op.CELLS))],
)
def test_certified_cells(entry, check, cell, certifying) -> None:  # noqa: F811 — fixture by name
    """Every certified cell of the tower's two entries against the entry's own reference."""
    certifying(entry)
    check(cell)


def _text(data):
    return lambda: SimpleNamespace(py_multimodal_data=data)


def _req(edit=None, **media):
    """`_request(**media)`, then `edit` on its `py_multimodal_data`."""

    def build():
        request = _request(**media)
        if edit is not None:
            edit(request.py_multimodal_data)
        return request

    return build


def _grid(value):
    return lambda d: d["image"].update(image_grid_thw=torch.tensor(value))


def _pixels(rows, extra=0, dtype=None):
    def edit(d):
        tower = _assembled_tower()
        d["video"]["pixel_values_videos"] = torch.empty(
            rows, tower.pixel_width + extra, dtype=dtype or tower.dtype, device="meta"
        )

    return edit


def _positions(*shape):
    return lambda d: d["mrope_config"].update(mrope_position_ids=torch.zeros(*shape))


def _cumsum(d):
    d["multimodal_embed_mask_cumsum"] = torch.arange(1, 14)


_MIXED = dict(image=IMAGES, video=VIDEOS, order=MIXED_ORDER)
_TILE = [(1, 4, 4)]
#: Served. The per-call ceiling of 32768 patches bounds a frame, not a whole video.
_SERVED = [
    ("image", _req(image=IMAGES)),
    ("video", _req(video=VIDEOS)),
    ("mixed", _req(**_MIXED)),
    ("frame_at_ceiling", _req(image=[(1, 128, 256)])),
    ("long_video", _req(video=[(33, 32, 32)])),
    ("no_cumsum", _req(lambda d: d.pop("multimodal_embed_mask_cumsum"), **_MIXED)),
]
#: Text, and position ids without media (a warmup request's): admitted without the tower.
_POSITIONS_ONLY = {"mrope_config": {"mrope_position_ids": torch.zeros(3, 1, 8)}}
_TEXT = [("text", _text(None)), ("positions_only", _text(_POSITIONS_ONLY))]
#: Refused: what the tower or the splice would assert on, or read wrongly.
_REFUSED = [
    ("frame_past_ceiling", _req(video=[(1, 192, 192)]),
     r"video 0 has frames of 192x192 = 36864 patches \(3072x3072 pixels\).*at most 32768"),
    ("grid_not_merge_blocks", _req(_grid([(1, 4, 5)]), image=_TILE), "spatial_merge_size 2"),
    ("grid_zero", _req(_grid([(0, 4, 4)]), image=_TILE), "every entry must be positive"),
    ("grid_1d", _req(_grid([1, 4, 4]), image=_TILE), r"one \(t, h, w\) row per image"),
    ("pixel_rows", _req(_pixels(5), video=VIDEOS), "describe 48 patch rows"),
    ("pixel_width", _req(_pixels(48, 1), video=VIDEOS), "48 patch rows of"),
    ("pixel_dtype", _req(_pixels(48, dtype=torch.uint8), video=VIDEOS),
     "torch.uint8; the vision tower runs at torch.bfloat16"),
    ("precomputed_embedding",
     _req(lambda d: d.update(multimodal_embedding=[torch.zeros(4, 8)]), image=_TILE),
     "precomputed image embeddings"),
    ("no_grid", _req(lambda d: d["video"].pop("video_grid_thw"), video=VIDEOS),
     "this target reads 'pixel_values_videos'"),
    ("positions_short", _req(_positions(3, 1, 40), video=VIDEOS), "one position per prompt token"),
    ("positions_2d", _req(_positions(1, 17), video=VIDEOS), r"got \(1, 17\)"),
    ("no_positions", _req(lambda d: d.pop("mrope_config"), video=VIDEOS),
     "mrope_position_ids.*got None"),
    ("no_manifest", _req(image=IMAGES, video=VIDEOS), "needs item metadata"),
    ("short_manifest", _req(**dict(_MIXED, order=MIXED_ORDER[:3])),
     r"lists image items \[0\]; the payload carries 2"),
    ("unknown_item", _req(**dict(_MIXED, order=MIXED_ORDER + [("audio", 0)])),
     "is not an item of this request"),
    ("placeholder_count", _req(_cumsum, **_MIXED), "holds 13 image/video placeholder tokens.*17"),
]  # fmt: skip
_ADMISSIONS = (
    [(i, b, None, True) for i, b in _SERVED]
    + [(i, b, None, False) for i, b in _TEXT]
    + [(i, b, r, True) for i, b, r in _REFUSED]
)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "build,refusal,reads_tower", [a[1:] for a in _ADMISSIONS], ids=[a[0] for a in _ADMISSIONS]
)
def test_admission_serves_what_the_tower_serves(build, refusal, reads_tower):
    """`validate_request` admits text without reading the tower and the media it serves, and
    refuses what the tower or the splice would assert on, naming the item and the bound."""
    shell = SimpleNamespace(vision_tower=_assembled_tower() if reads_tower else None)
    if refusal is None:
        Shell.validate_request(shell, build())
    else:
        with pytest.raises(ValueError, match=refusal):
            Shell.validate_request(shell, build())


def _chunk(cached, in_chunk, total=len(MIXED_ROWS)):
    return SimpleNamespace(
        num_cached_mm_tokens=cached,
        num_mm_tokens_in_chunk=in_chunk,
        total_embeds_in_request=total,
    )


#: `(id, _params kwargs, placeholder rows, rows that land there (None: no media), tower calls,
#: patch rows the tower runs)`. The mixed chunk takes prompt-order rows 3..9: the second frame of
#: video 0, image 0 and the first frame of video 1 (40 of the request's 68 patch rows); the video
#: chunk rows 5..7, inside video 1's first frame.
_SPLICES = [
    ("text", dict(), 0, None, 0, 0),
    ("image", dict(image=IMAGES), 5, [100.0] * 4 + [101.0], 1, 20),
    ("video", dict(video=VIDEOS), 12, [200.0] * 4 + [201.0] * 8, 1, 48),
    ("mixed", _MIXED, len(MIXED_ROWS), MIXED_ROWS, 1, 68),
    ("chunk", dict(_MIXED, runtime=_chunk(3, 6)), 6, MIXED_ROWS[3:9], 1, 40),
    ("video_chunk", dict(video=VIDEOS, runtime=_chunk(5, 2, 12)), 2, [201.0] * 2, 1, 16),
    ("chunk_without_placeholders", dict(_MIXED, runtime=_chunk(len(MIXED_ROWS), 0)), 0, None, 0, 0),
]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "params,num_rows,want,calls,patches", [s[1:] for s in _SPLICES], ids=[s[0] for s in _SPLICES]
)
def test_the_splice_puts_each_items_rows_on_its_placeholders(
    params, num_rows, want, calls, patches
):
    """The tower's rows reach `begin_step` with the placeholder indices, and the splice writes them
    over the placeholders in prompt order and zeroes the padding rows after the step. A chunk runs
    the tower on only the frames that hold its rows."""
    tower = _Tower()
    indices = torch.arange(1, 1 + num_rows, dtype=torch.int64)
    kwargs = {
        "multimodal_params": [_params(**params)],
        "mm_token_indices": indices,
        "resource_manager": object(),
    }
    vision = _forward_to_begin_step(tower, kwargs)
    assert (tower.calls, tower.patches) == (calls, patches)
    total = num_rows + 2
    h = torch.full((total + 3, HIDDEN), -1.0)
    M.PrefillTarget._splice(
        SimpleNamespace(_step=SimpleNamespace(vision=vision, num_tokens=total)), h
    )
    assert torch.count_nonzero(h[total:]) == 0, "padding rows must be zeroed"
    if want is None:
        assert vision is None and torch.all(h[:total] == -1.0)
        return
    assert vision[1] is indices
    assert h[1 : 1 + num_rows, 0].tolist() == want
    assert torch.all(h[0] == -1.0) and torch.all(h[total - 1] == -1.0)
