# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1 multimodal input processor: image preprocessing and image-span expansion.

This is the checkpoint's ``inference/image_processor.py`` (``prepare_vl_inputs``) as a TensorRT-LLM
input processor, plus the host-side tables the model's vision path indexes with. An image becomes
an ``n_vit_h x n_vit_w`` grid of 14x14 patches for the ViT and an ``n_llm_h x n_llm_w`` grid after
the aligner's 3x3 downsample, which the language model sees as the span

    [START] + ([IMAGE] * n_llm_w + [NEW_LINE]) * n_llm_h + [END]

of ``n_llm_h * (n_llm_w + 1) + 2`` positions. Every span position carries ``image_token_id``
(``<｜deepseek_image｜>``, 129264) in ``input_ids``; only the per-position type tells them apart.

Preprocessing, exactly as the checkpoint does it:

* an image under ``min_pixels`` (544 x 544) is scaled up to about that area, and both sides are
  rounded up to multiples of the patch size;
* if the resulting span would exceed ``max_image_tokens`` (1024), the size shrinks to the largest
  aspect-preserving one that fits;
* the image is then fitted into that size with ``ImageOps.pad`` (aspect-preserving resize, then
  gray (127, 127, 127) padding). A plain resize to the target size is used only when
  ``max_wh_ratio`` is set and the image is at least that wide; this checkpoint sets it to None, so
  it never happens;
* pixels become ``(x / 255 - 0.5) / 0.5`` in fp32, then bf16, then row-major patches with the
  channels first inside each patch.

Text is never rendered here. A prompt arrives already rendered (the ``custom_tokenizer:
deepseek_v4`` chat renderer produces the V4.1 prompt byte-for-byte, and the placeholder registry
puts one ``<｜deepseek_image｜>`` per image before the text, joined by a blank line); this module
tokenizes it and expands each placeholder into its span.

What the model receives: ``multimodal_data`` of a request with images
(``P`` = patches over all images of the request, ``C`` = aligner cells = IMAGE slots over all
images, ``S`` = span positions over all images, ``L`` = prompt length, ``N`` = images)

``multimodal_data["image"]``, the vision path's inputs, in prompt order:

* ``pixel_values``: bf16 ``[P, 3, 14, 14]``. ``patch_embed`` reads ``pixel_values.flatten(1)``.
* ``vit_grid_hw``: ``list[list[int]]``, ``[n_vit_h, n_vit_w]`` per image.
* ``llm_grid_hw``: ``list[list[int]]``, ``[n_llm_h, n_llm_w]`` per image.
* ``vit_cu_seqlens``: ``list[int]`` of length ``N + 1``, the patch offset of each image. ViT
  attention is bidirectional within one image and never across images.
* ``vit_rope_cos`` / ``vit_rope_sin``: fp32 ``[P, 32]``, the ViT's 2D rotary table: per patch the
  angles ``[row * inv_freq(16), col * inv_freq(16)]``, ``inv_freq = 1 / theta ** (arange(0, 32,
  2) / 32)``, applied rotate-half style to each 64-wide head.
* ``aligner_index``: int64 ``[C, 9]``. Row ``k`` lists, in ``(kh, kw)`` order, the ViT output rows
  of the 3x3 block that aligner cell ``k`` reads; the value ``P`` is the zero padding the aligner
  adds on the bottom and right edges. With ``vit_out`` ``[P, 1024]``,
  ``cat([vit_out, zeros(1, 1024)])[aligner_index].transpose(1, 2).reshape(C, 9216)`` is exactly
  the aligner's ``F.unfold`` input ``(channel, kh, kw)``, for all images at once.
* ``span_types``: int64 ``[S]``, the types of every span position (``IMAGE_START`` 0,
  ``IMAGE`` 1, ``IMAGE_NEW_LINE`` 2, ``IMAGE_END`` 3).
* ``span_source_index``: int64 ``[S]``. With ``aligner_out`` ``[C, hidden]``,
  ``cat([aligner_out, image_start[None], image_end[None], image_newline[None]])[span_source_index]``
  is the ``[S, hidden]`` span embedding: IMAGE slots take aligner rows in reading order and the
  delimiters take the learned vectors (rows ``C``, ``C + 1``, ``C + 2``).
* ``span_starts`` / ``span_lengths``: ``list[int]``, each span's first prompt position and length.
* ``aligner_cu_cells``: ``list[int]`` of length ``N + 1``, the aligner-cell offset of each image.
* ``aligner_index_local`` / ``span_source_local``: the same two tables with every index local to its
  own image, so any run of whole images can be assembled without the others: row ``k`` of image
  ``i``'s cells indexes ``cat([vit_out_i, zeros(1, 1024)])`` (``P_i`` = that image's patch count is
  the zero pad), and its span positions index ``cat([aligner_out_i, image_start, image_end,
  image_newline])`` (delimiters at ``C_i``, ``C_i + 1``, ``C_i + 2``).

``multimodal_data["token_layout"]``, per prompt position (index by absolute prompt position):

* ``token_types``: int8 ``[L]``, ``TEXT`` (-1) outside spans, the span type inside.
* ``image_mask``: bool ``[L]``, True on every span position; it selects the MoE gate's ``bias_vl``.
* ``engram_dead_mask``: bool ``[L]``, True on every span position; Engram stores those positions as
  DEAD (no n-gram crosses them, and their gate is 0). The reference's Engram ``token_mask`` is its
  negation.

Every ``image_token_id`` in a processed prompt is a span position and every span position is one,
so ``image_mask == engram_dead_mask == (input_ids == image_token_id)`` on the prompt.

The span embedding is one row per span position, which is what the runtime's multimodal splice
expects: ``get_mm_token_ids`` is ``[image_token_id]`` and there are no id-level special tokens, so
``multimodal_embed_mask_cumsum`` counts every span position.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageOps
from transformers import AutoTokenizer, PretrainedConfig

from tensorrt_llm.inputs.content_format import ContentFormat
from tensorrt_llm.inputs.data import TextPrompt
from tensorrt_llm.inputs.registry import (
    BaseMultimodalInputProcessor,
    DefaultInputProcessor,
    ExtraProcessedInputs,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
)
from tensorrt_llm.sampling_params import SamplingParams

__all__ = [
    "IMAGE",
    "IMAGE_END",
    "IMAGE_NEW_LINE",
    "IMAGE_PLACEHOLDER",
    "IMAGE_START",
    "INPUT_PROCESSOR",
    "MODEL_TYPE",
    "PLACEHOLDER_METADATA",
    "PAD_COLOR",
    "TEXT",
    "DeepseekV41InputProcessor",
    "ImageGrid",
    "PreprocessedImage",
    "VisionGeometry",
    "build_vision_inputs",
    "expand_image_placeholders",
    "image_span_types",
    "num_image_tokens",
    "plan_image_grid",
    "preprocess_image",
    "to_rgb_image",
    "vit_rope_table",
]

# Per-position type codes, the checkpoint's own values.
TEXT = -1
IMAGE_START, IMAGE, IMAGE_NEW_LINE, IMAGE_END = range(4)

IMAGE_PLACEHOLDER = "<｜deepseek_image｜>"
PAD_COLOR = (127, 127, 127)


@dataclass(frozen=True)
class VisionGeometry:
    """The vision numbers preprocessing and the tables depend on, read from the checkpoint config.

    Attributes:
        patch_size: ViT patch side in pixels (``vision_config.patch_size``).
        downsample_ratio: aligner block side in patches (``vision_config.downsample_ratio``).
        max_image_tokens: largest span an image may cost (``vision_config.max_image_tokens``).
        min_pixels: smaller images are scaled up to about this area (``vision_config.min_pixels``).
        max_wh_ratio: plain-resize threshold; None pads every image (``vision_config.max_wh_ratio``).
        hidden_size: ViT width (``vision_config.hidden_size``).
        num_heads: ViT heads (``vision_config.num_attention_heads``).
        num_layers: ViT depth; 0 means the checkpoint has no vision tower.
        rope_theta: base of the ViT's 2D rotary table (``vision_config.rope_theta``).
        image_token_id: the id every span position carries (top-level ``image_token_id``).
    """

    patch_size: int
    downsample_ratio: int
    max_image_tokens: int
    min_pixels: int
    max_wh_ratio: Optional[float]
    hidden_size: int
    num_heads: int
    num_layers: int
    rope_theta: float
    image_token_id: int

    @classmethod
    def from_config(cls, config: PretrainedConfig) -> "VisionGeometry":
        """Read the geometry from a ``deepseek_v41`` composite config."""
        vision = config.vision_config
        if isinstance(vision, dict):
            vision = PretrainedConfig(**vision)
        return cls(
            patch_size=int(vision.patch_size),
            downsample_ratio=int(vision.downsample_ratio),
            max_image_tokens=int(vision.max_image_tokens),
            min_pixels=int(vision.min_pixels),
            max_wh_ratio=vision.max_wh_ratio,
            hidden_size=int(vision.hidden_size),
            num_heads=int(vision.num_attention_heads),
            num_layers=int(vision.num_hidden_layers),
            rope_theta=vision.rope_theta,
            image_token_id=int(config.image_token_id),
        )

    @property
    def rope_dim(self) -> int:
        """Rotary angles per patch: half of one head's width."""
        return self.hidden_size // self.num_heads // 2


@dataclass(frozen=True)
class ImageGrid:
    """Where an image of a given size lands: its padded pixel size and both grids."""

    n_llm_h: int
    n_llm_w: int
    best_height: int
    best_width: int
    n_vit_h: int
    n_vit_w: int

    @property
    def num_span_tokens(self) -> int:
        return num_image_tokens(self.n_llm_h, self.n_llm_w)


@dataclass(frozen=True)
class PreprocessedImage:
    """One image, ready for the ViT.

    Attributes:
        patches: bf16 ``[n_vit_h * n_vit_w, 3, patch, patch]``, row-major, channels first.
        grid: the image's grid.
    """

    patches: torch.Tensor
    grid: ImageGrid


def num_image_tokens(n_llm_h: int, n_llm_w: int) -> int:
    """Span length of an ``n_llm_h x n_llm_w`` token grid, delimiters included."""
    return n_llm_h * (n_llm_w + 1) + 2


def _llm_grid(best_height: int, best_width: int, patch_size: int, downsample_ratio: int):
    return math.ceil((best_height // patch_size) / downsample_ratio), math.ceil(
        (best_width // patch_size) / downsample_ratio
    )


def _solve_resize_ratio(height, width, patch_size, downsample_ratio, max_n_token):
    """Largest aspect-preserving pixel size whose span still fits in ``max_n_token``."""
    r = height / width
    max_w_float = math.sqrt((max_n_token - 2) / r + 0.25) - 0.5
    max_h_float = max_w_float * r
    cell = patch_size * downsample_ratio
    if max_w_float < 1.0:  # very tall: a single column
        return (max_n_token - 2) // 2 * cell, cell
    if max_h_float < 1.0:  # very wide: a single row
        return cell, (max_n_token - 3) * cell
    beta = min(math.floor(max_w_float) * cell / width, math.floor(max_h_float) * cell / height)
    return (
        math.floor(height * beta / patch_size) * patch_size,
        math.floor(width * beta / patch_size) * patch_size,
    )


def _safe_resize(height, width, best_height, best_width, patch_size, downsample_ratio, max_n_token):
    n_llm_h, n_llm_w = _llm_grid(best_height, best_width, patch_size, downsample_ratio)
    if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
        best_height, best_width = _solve_resize_ratio(
            height, width, patch_size, downsample_ratio, max_n_token
        )
        n_llm_h, n_llm_w = _llm_grid(best_height, best_width, patch_size, downsample_ratio)
        if num_image_tokens(n_llm_h, n_llm_w) > max_n_token:
            raise ValueError(
                f"image grid {n_llm_h}x{n_llm_w} still exceeds {max_n_token} tokens after shrinking"
            )
    return n_llm_h, n_llm_w, best_height, best_width


def plan_image_grid(width: int, height: int, geometry: VisionGeometry) -> ImageGrid:
    """The grid an image of this original pixel size gets; depends on the size alone."""
    p = geometry.patch_size
    if geometry.max_wh_ratio is not None and width > height * geometry.max_wh_ratio:
        width = height * geometry.max_wh_ratio
    if 0 < width * height < geometry.min_pixels:
        ratio = (geometry.min_pixels / (width * height)) ** 0.5
        width = int(width * ratio)
        height = int(height * ratio)
    best_width = math.ceil(width / p) * p
    best_height = math.ceil(height / p) * p
    n_llm_h, n_llm_w, best_height, best_width = _safe_resize(
        height,
        width,
        best_height,
        best_width,
        p,
        geometry.downsample_ratio,
        geometry.max_image_tokens,
    )
    return ImageGrid(
        n_llm_h=n_llm_h,
        n_llm_w=n_llm_w,
        best_height=best_height,
        best_width=best_width,
        n_vit_h=best_height // p,
        n_vit_w=best_width // p,
    )


def to_rgb_image(image: Any) -> Image.Image:
    """Bring an image from any runtime media format to an RGB ``PIL.Image``.

    Accepts a ``PIL.Image`` (converted with ``convert("RGB")``, as the checkpoint does), a
    ``torch.Tensor`` ``[C, H, W]`` (float in ``[0, 1]``, the runtime's ``"pt"`` format, or uint8), or
    a ``numpy.ndarray`` ``[H, W, C]`` of uint8 (the runtime's ``"np"`` format). Float tensors are
    mapped back to the uint8 values they were made from, so the pixels are the decoder's.
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, torch.Tensor):
        tensor = image.detach().to("cpu")
        if tensor.dim() != 3:
            raise ValueError(f"image tensor must be [C, H, W], got shape {tuple(tensor.shape)}")
        if tensor.is_floating_point():
            tensor = (tensor.float().clamp(0.0, 1.0) * 255.0).round().to(torch.uint8)
        elif tensor.dtype != torch.uint8:
            raise ValueError(f"integer image tensors must be uint8, got {tensor.dtype}")
        array = tensor.permute(1, 2, 0).contiguous().numpy()
    elif isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.ndim not in (2, 3):
            raise ValueError(
                f"numpy images must be uint8 [H, W] or [H, W, C], got {image.dtype} {image.shape}"
            )
        array = image
    else:
        raise TypeError(f"unsupported image type {type(image).__name__}")
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    return Image.fromarray(np.ascontiguousarray(array)).convert("RGB")


def _image_size(image: Any) -> Tuple[int, int]:
    """(width, height) without decoding pixels."""
    if isinstance(image, Image.Image):
        return image.width, image.height
    if isinstance(image, torch.Tensor):
        return int(image.shape[-1]), int(image.shape[-2])
    if isinstance(image, np.ndarray):
        return int(image.shape[1]), int(image.shape[0])
    raise TypeError(f"unsupported image type {type(image).__name__}")


def preprocess_image(image: Any, geometry: VisionGeometry) -> PreprocessedImage:
    """Fit one image into its grid and cut it into normalized bf16 patches."""
    image = to_rgb_image(image)
    grid = plan_image_grid(image.width, image.height, geometry)
    p = geometry.patch_size
    target = (grid.best_width, grid.best_height)
    if geometry.max_wh_ratio is not None and image.width >= geometry.max_wh_ratio * image.height:
        image = image.resize(target)
    else:
        image = ImageOps.pad(image, target, color=PAD_COLOR)
    x = torch.from_numpy(np.asarray(image, dtype=np.float32)).permute(2, 0, 1) / 255
    x = ((x - 0.5) / 0.5).to(torch.bfloat16)
    patches = (
        x.reshape(3, grid.n_vit_h, p, grid.n_vit_w, p)
        .permute(1, 3, 0, 2, 4)
        .reshape(grid.n_vit_h * grid.n_vit_w, 3, p, p)
    )
    return PreprocessedImage(patches=patches, grid=grid)


def image_span_types(n_llm_h: int, n_llm_w: int) -> torch.Tensor:
    """int64 ``[num_image_tokens]``: the span's per-position types in reading order."""
    types = [IMAGE_START]
    types += ([IMAGE] * n_llm_w + [IMAGE_NEW_LINE]) * n_llm_h
    types.append(IMAGE_END)
    return torch.tensor(types, dtype=torch.int64)


def vit_rope_table(n_h: int, n_w: int, dim: int, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp32 ``[n_h * n_w, dim]`` cos and sin of the ViT's 2D rotary angles, patches row-major."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    hpos = torch.arange(n_h).unsqueeze(1).expand(n_h, n_w)
    wpos = torch.arange(n_w).unsqueeze(0).expand(n_h, n_w)
    freqs = torch.stack([hpos, wpos], dim=-1).reshape(-1, 2, 1).float() * inv_freq
    freqs = freqs.flatten(1)
    return freqs.cos(), freqs.sin()


def _aligner_index(grid: ImageGrid, ratio: int, patch_base: int) -> torch.Tensor:
    """int64 ``[n_llm_h * n_llm_w, ratio**2]`` patch rows per aligner cell; -1 marks padding."""
    rows = torch.arange(grid.n_llm_h * ratio).view(grid.n_llm_h, 1, ratio, 1)
    cols = torch.arange(grid.n_llm_w * ratio).view(1, grid.n_llm_w, 1, ratio)
    index = patch_base + rows * grid.n_vit_w + cols
    valid = (rows < grid.n_vit_h) & (cols < grid.n_vit_w)
    index = torch.where(valid, index, torch.full_like(index, -1))
    return index.reshape(grid.n_llm_h * grid.n_llm_w, ratio * ratio)


def build_vision_inputs(
    images: Sequence[PreprocessedImage], span_starts: Sequence[int], geometry: VisionGeometry
) -> Dict[str, Any]:
    """The ``multimodal_data["image"]`` payload for a request's images, in prompt order."""
    pixel_values, cos, sin, aligner, span_types, span_image_rows = [], [], [], [], [], []
    aligner_local, span_local = [], []
    vit_cu_seqlens = [0]
    aligner_cu_cells = [0]
    num_cells = 0
    for image in images:
        grid = image.grid
        pixel_values.append(image.patches)
        c, s = vit_rope_table(grid.n_vit_h, grid.n_vit_w, geometry.rope_dim, geometry.rope_theta)
        cos.append(c)
        sin.append(s)
        aligner.append(_aligner_index(grid, geometry.downsample_ratio, vit_cu_seqlens[-1]))
        num_image_patches = grid.n_vit_h * grid.n_vit_w
        local = _aligner_index(grid, geometry.downsample_ratio, 0)
        local[local < 0] = num_image_patches
        aligner_local.append(local)
        types = image_span_types(grid.n_llm_h, grid.n_llm_w)
        span_types.append(types)
        rows = torch.full_like(types, -1)
        cells = grid.n_llm_h * grid.n_llm_w
        rows[types == IMAGE] = torch.arange(num_cells, num_cells + cells)
        span_image_rows.append(rows)
        rows_local = torch.full_like(types, -1)
        rows_local[types == IMAGE] = torch.arange(cells)
        rows_local[types == IMAGE_START] = cells
        rows_local[types == IMAGE_END] = cells + 1
        rows_local[types == IMAGE_NEW_LINE] = cells + 2
        span_local.append(rows_local)
        num_cells += cells
        vit_cu_seqlens.append(vit_cu_seqlens[-1] + num_image_patches)
        aligner_cu_cells.append(num_cells)

    num_patches = vit_cu_seqlens[-1]
    aligner_index = torch.cat(aligner)
    aligner_index[aligner_index < 0] = num_patches
    span_types_all = torch.cat(span_types)
    span_source_index = torch.cat(span_image_rows)
    span_source_index[span_types_all == IMAGE_START] = num_cells
    span_source_index[span_types_all == IMAGE_END] = num_cells + 1
    span_source_index[span_types_all == IMAGE_NEW_LINE] = num_cells + 2
    return {
        "pixel_values": torch.cat(pixel_values),
        "vit_grid_hw": [[image.grid.n_vit_h, image.grid.n_vit_w] for image in images],
        "llm_grid_hw": [[image.grid.n_llm_h, image.grid.n_llm_w] for image in images],
        "vit_cu_seqlens": vit_cu_seqlens,
        "vit_rope_cos": torch.cat(cos),
        "vit_rope_sin": torch.cat(sin),
        "aligner_index": aligner_index,
        "span_types": span_types_all,
        "span_source_index": span_source_index,
        "span_starts": list(span_starts),
        "span_lengths": [image.grid.num_span_tokens for image in images],
        "aligner_cu_cells": aligner_cu_cells,
        "aligner_index_local": torch.cat(aligner_local),
        "span_source_local": torch.cat(span_local),
    }


def expand_image_placeholders(
    prompt_token_ids: Sequence[int], images: Sequence[Any], geometry: VisionGeometry
) -> Tuple[List[int], List[int], List[PreprocessedImage], List[int]]:
    """Expand every placeholder token into its image's span, preprocessing images in order.

    Returns:
        ``(token_ids, token_types, images, span_starts)``: the expanded ids, the per-position types
        (``TEXT`` outside spans), the preprocessed images and each span's first position.

    Raises:
        ValueError: the number of placeholder tokens is not the number of images, or the
            checkpoint has no vision tower and images were given.
    """
    image_token_id = geometry.image_token_id
    num_placeholders = sum(token == image_token_id for token in prompt_token_ids)
    if num_placeholders != len(images):
        raise ValueError(
            f"Found {num_placeholders} image placeholder tokens ({IMAGE_PLACEHOLDER}, id "
            f"{image_token_id}) but got {len(images)} images"
        )
    if num_placeholders and geometry.num_layers == 0:
        raise ValueError(
            "The model config has no vision tower (vision_config.num_hidden_layers == 0) but the "
            "prompt contains images"
        )
    tokens: List[int] = []
    token_types: List[int] = []
    processed: List[PreprocessedImage] = []
    span_starts: List[int] = []
    image_iter = iter(images)
    for token in prompt_token_ids:
        if token != image_token_id:
            tokens.append(token)
            token_types.append(TEXT)
            continue
        image = preprocess_image(next(image_iter), geometry)
        types = image_span_types(image.grid.n_llm_h, image.grid.n_llm_w)
        span_starts.append(len(tokens))
        processed.append(image)
        tokens += [image_token_id] * types.numel()
        token_types += types.tolist()
    return tokens, token_types, processed, span_starts


class DeepseekV41InputProcessor(BaseMultimodalInputProcessor):
    """Tokenizes a rendered DeepSeek-V4.1 prompt and expands its image placeholders.

    A prompt without images is tokenized exactly as ``DefaultInputProcessor`` does, so text-only
    requests get the token ids they would get with no multimodal registration at all.

    Args:
        model_path: checkpoint directory; the tokenizer is loaded from it when none is given.
        config: the checkpoint's ``deepseek_v41`` config (``vision_config`` and ``image_token_id``
            are read).
        tokenizer: the engine's tokenizer, returned unchanged by ``tokenizer`` so the engine keeps
            its chat renderer; None (per-rank construction) loads the checkpoint's on first use.
        trust_remote_code: passed to ``AutoTokenizer`` when the tokenizer is loaded here.
    """

    # The language model attends causally over image spans, so a span may be split across
    # prefill chunks.
    mm_bidirectional_blocks: ClassVar[bool] = False
    # Expansion works on token ids, so prompt_token_ids + images never needs a detokenize.
    supports_token_id_mm_expansion: ClassVar[bool] = True

    def __init__(
        self,
        model_path: str,
        config: PretrainedConfig,
        tokenizer: Any = None,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            model_path=model_path,
            config=config,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        self._geometry = VisionGeometry.from_config(config)
        self._text_processor: Optional[DefaultInputProcessor] = None
        self._placeholder_checked = False

    @property
    def geometry(self) -> VisionGeometry:
        return self._geometry

    @property
    def config(self) -> PretrainedConfig:
        return self._config

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_path, trust_remote_code=self._trust_remote_code
            )
        return self._tokenizer

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def processor(self) -> None:
        """The checkpoint ships no HF processor; this class is the whole preprocessing path."""
        return None

    @property
    def dtype(self) -> torch.dtype:
        """The checkpoint's dtype, and the dtype ``pixel_values`` are produced in."""
        return torch.bfloat16

    def get_mm_token_ids(self) -> torch.Tensor:
        return torch.tensor([self._geometry.image_token_id])

    def get_mm_special_token_ids(self) -> None:
        # START, NEW_LINE and END share the image id, so none is distinguishable by id; each takes
        # an embedding row like an IMAGE slot does.
        return None

    def get_num_tokens_per_image(self, *, image: Any, **kwargs) -> int:
        """Span length of an image, delimiters included; depends only on its size."""
        width, height = _image_size(image)
        return plan_image_grid(width, height, self._geometry).num_span_tokens

    def get_preferred_media_io_kwargs(self) -> Dict[str, Dict[str, Any]]:
        """Per-modality media-IO decode defaults: none, so the server's own settings apply.

        ``OpenAIServer`` asks every ``BaseMultimodalInputProcessor`` for these when it starts.
        The registry's default lives on ``BaseMultimodalDummyInputsBuilder``, which this
        processor does not implement, so it answers here with that same empty default:
        ``to_rgb_image`` takes the runtime's PIL, ``"pt"`` and ``"np"`` image formats alike.
        """
        return {}

    def call_with_text_prompt(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        images = self._images(inputs)
        if not images:
            if self._text_processor is None:
                self._text_processor = DefaultInputProcessor(None, None, self.tokenizer)
            token_ids, _ = self._text_processor(inputs, sampling_params)
            self._check_no_placeholders(token_ids)
            return token_ids, None
        self._check_placeholder_token()
        prompt_token_ids = self.tokenizer.encode(
            inputs["prompt"], add_special_tokens=sampling_params.add_special_tokens
        )
        return self._expand(prompt_token_ids, images, sampling_params)

    def call_with_token_ids(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        return self._expand(list(inputs["prompt_token_ids"]), self._images(inputs), sampling_params)

    def _expand(
        self,
        prompt_token_ids: List[int],
        images: List[Any],
        sampling_params: SamplingParams,
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        tokens, token_types, processed, span_starts = expand_image_placeholders(
            prompt_token_ids, images, self._geometry
        )
        if not processed:
            return tokens, None
        limit = sampling_params.truncate_prompt_tokens
        if limit is not None and len(tokens) > limit:
            raise ValueError(
                f"The prompt is {len(tokens)} tokens with its image spans expanded, more than "
                f"truncate_prompt_tokens={limit}; truncating it would cut an image span"
            )
        types = torch.tensor(token_types, dtype=torch.int8)
        image_mask = types >= 0
        multimodal_data = {
            "image": build_vision_inputs(processed, span_starts, self._geometry),
            "token_layout": {
                "token_types": types,
                "image_mask": image_mask,
                "engram_dead_mask": image_mask.clone(),
            },
        }
        return tokens, {"multimodal_data": multimodal_data}

    @staticmethod
    def _images(inputs: TextPrompt) -> List[Any]:
        mm_data = inputs.get("multi_modal_data") or {}
        unsupported = sorted(k for k, v in mm_data.items() if k != "image" and v is not None)
        if unsupported:
            raise ValueError(f"DeepSeek-V4.1 takes images only, got {unsupported}")
        images = mm_data.get("image")
        if images is None:
            return []
        return list(images) if isinstance(images, (list, tuple)) else [images]

    def _check_no_placeholders(self, token_ids: Sequence[int]) -> None:
        count = sum(token == self._geometry.image_token_id for token in token_ids)
        if count:
            raise ValueError(
                f"Found {count} image placeholder tokens ({IMAGE_PLACEHOLDER}, id "
                f"{self._geometry.image_token_id}) but got 0 images"
            )

    def _check_placeholder_token(self) -> None:
        """The placeholder string must tokenize to the config's image id."""
        if self._placeholder_checked:
            return
        placeholder_id = self.tokenizer.convert_tokens_to_ids(IMAGE_PLACEHOLDER)
        unk_id = getattr(self.tokenizer, "unk_token_id", None)
        if placeholder_id is not None and placeholder_id != unk_id:
            if placeholder_id != self._geometry.image_token_id:
                raise ValueError(
                    f"{IMAGE_PLACEHOLDER} is token {placeholder_id} in this tokenizer but the "
                    f"config's image_token_id is {self._geometry.image_token_id}"
                )
        self._placeholder_checked = True


# What ``modeling_v2_input_processor`` registers when a target claims the
# checkpoint: the processor, the multimodal model type the placeholder registry
# keys it by, and the placement, which follows the checkpoint's own prompt
# encoder (``encoding/encoding.py``): one ``<｜deepseek_image｜>`` per image, all
# before the text, blocks joined by a blank line, the content handed to the chat
# renderer as a plain string.
MODEL_TYPE = "deepseek_v41"
PLACEHOLDER_METADATA = MultimodalPlaceholderMetadata(
    placeholder_map={"image": IMAGE_PLACEHOLDER},
    placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    placeholders_separator="\n\n",
    content_format=ContentFormat.STRING,
)
INPUT_PROCESSOR = DeepseekV41InputProcessor
