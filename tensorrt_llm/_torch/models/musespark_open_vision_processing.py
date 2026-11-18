# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""MuseSparkOpen vision preprocessing and prompt placeholder expansion.

The framework-light half of the MuseSparkOpen multimodal input path: everything
between "a PIL image / a decoded video" and "the patch tensor the tower
consumes plus the exact prompt token ids the decoder sees". It imports only
`torch`, `numpy` and `PIL`, so it is importable on a machine with no GPU, no
engine and no tokenizer.

Not in this module: the NaViT tower, the input-processor class, the embedding
merge, the weight mapper.

Image path
----------
Native variable resolution -- no tiling, no cropping, no padding, no fixed 448
resize. Each image is converted to RGB (alpha is composited over white *before*
resampling, so transparent pixels cannot bleed into the visible edge), resized
once with PIL LANCZOS onto a canvas whose sides are multiples of the token patch
`patch_size * downsample_factor = 14 * 2 = 28`, and normalized in fp32 as
`(x / 255 - mean) / std` with `mean = std = 0.5`.

The canvas comes from :func:`least_distortion_fit`: the ideal fractional cell
grid `(H/28, W/28)` is rescaled if its product falls outside
`[min_num_tokens, max_num_image_tokens]`, then the floor/ceil combination that
minimizes aspect-ratio distortion wins. :func:`find_resize_target` is the same
computation in the reference's own argument order and also returns the cell
grid; a fallback scan handles aspect ratios so extreme that no floor/ceil
combination fits the budget.

`image_size = 448` in the config is *not* a resize target. It is the side of
the learned positional-embedding grid the tower interpolates from.

Video path
----------
Frames are sampled at `sampling_fps = 2.0`, capped at
`num_frames_per_video = 96`, floored to a multiple of
`max(frame_factor, patch_temporal) = 2` and picked with `np.linspace` over
the source frame range -- uniform, not a stride, so the last frame is always
kept. Every frame is resized to one canvas under a per-frame budget of
`max_num_tokens_per_frame = 144` tokens. Two consecutive frames form one
temporal clip, and each clip carries the timestamp of its *first* frame.

`max_num_video_tokens` (4096) is deliberately not enforced while
`max_num_tokens_per_frame > 0`: that branch wins outright in the reference, so
a 96-frame video really does produce `48 * 144 = 6912` video rows. Matching
the reference matters more here than honouring the declared cap, but the cap is
what the reference reports to its scheduler, so an encoder budget sized from it
will be ~68% short. The two numbers are therefore NOT alternatives: 4096 is a
per-item cap that only the scheduler sees, and the per-frame budget is what the
canvas is actually fitted to.

Where the per-frame budget comes from
-------------------------------------
144 is a code default, not a checkpoint value -- no export has ever declared it,
and it is pinned here because it reproduces the reference runtime token for
token: a captured reference prompt's video expands to `48 * 144 = 6912` rows.

Newer exports do declare it. They drop `config.json ->
vision_config.navit_config` and ship a HuggingFace-style
`processor_config.json` instead, renaming the same knobs on the way:
`max_num_image_tokens` -> `image_processor.max_image_tokens`, and a
per-frame video budget that never had a config home before as
`video_processor.max_video_frame_tokens`. :func:`with_processor_config` reads
that file so the checkpoint's own declaration governs;
:data:`_PROCESSOR_ALIASES` is the rename map, and a checkpoint that ships no
such file keeps the defaults above unchanged.

Placeholder expansion
---------------------
The chat template emits the bare strings `<image>` / `<video>`; the token run
of each marker is replaced, in prompt order, with::

    image: <|image_start|> <|patch|>*N <|image_end|>
    video: <|vid_start|> ( "Time: {s:.1f}s" <|video|>*M
                           <|vid_frame_separator|> "Time: {s:.1f}s" <|video|>*M
                           ... ) <|vid_end|>

No row/column separators inside an image, no newline after a timestamp, and
`<|image|>` (200090) is **not** emitted. The checkpoint config declares
`image_token_id: 200090` but the token that actually carries image embeddings
is `<|patch|>` (200092); wiring the merge to 200090 would silently place
vision rows on the wrong positions. Ids are therefore taken from
:data:`VISION_TOKEN_IDS`, never from `config.image_token_id`.

Shapes and conventions
----------------------
`grid_thw = (t, h, w)` describes one medium: `t` temporal clips, `h` and
`w` 14-px patches per frame (both even, because the canvas is a multiple of
28). One medium is `t * h * w` tower patches and
`t * (h // 2) * (w // 2)` decoder rows -- the 2x2 pixel shuffle merges four
patches into one row. :func:`num_placeholder_rows` is that second number and
must equal the tower's output first dim for the same medium.

`pixel_values` is fp32 `[num_patches, patch_dim]` with
`patch_dim = num_channels * patch_temporal * patch_size ** 2 = 3*2*14*14 =
1176`. Each row is one 14x14 patch of one clip, flattened `[t][c][py][px]`
with the temporal index outermost; rows run clip-major, then patch row-major
(`h` outer, `w` inner). A still image is *expanded* to `patch_temporal`
identical copies, never zero-padded -- zero padding would halve the energy of
every patch vector.

Every function that takes a `cfg` accepts the parsed vision config, the
top-level model config, a plain dict, or `None` for the checkpoint defaults;
:meth:`MuseSparkOpenPreprocessConfig.from_config` does the walking.
"""

from __future__ import annotations

import copy
import itertools
import json
import math
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch
from PIL import Image

from tensorrt_llm.logger import logger

__all__ = [
    "VISION_TOKEN_IDS",
    "RGBA_BACKGROUND_COLOR",
    "PROCESSOR_CONFIG_FILE",
    "MuseSparkOpenPreprocessConfig",
    "processor_config_overrides",
    "with_processor_config",
    "ExpandedPrompt",
    "VisionPreprocessError",
    "least_distortion_fit",
    "find_resize_target",
    "preprocess_image",
    "preprocess_video",
    "image_placeholder_ids",
    "video_placeholder_ids",
    "num_placeholder_rows",
    "num_patches",
    "patch_dim",
    "is_embed_mask",
    "sample_video_frame_indices",
    "video_grid_thw",
    "to_rgb",
    "resize_lanczos",
    "normalize_pixels",
    "patchify",
    "expand_temporal",
    "expand_prompt_placeholders",
]

# Special-token ids, resolved by token *name* from the shipped tokenizer and
# pinned here as the single source of truth for the vision path. Deliberately
# not read from the model config: `config.image_token_id` is 200090
# (`<|image|>`), which this expansion never emits.
VISION_TOKEN_IDS: dict[str, int] = {
    "image_start": 200080,  # <|image_start|>
    "image_end": 200081,  # <|image_end|>
    "patch": 200092,  # <|patch|>       -- carries image embeddings
    "image": 200090,  # <|image|>       -- never emitted, see module docstring
    "vid_start": 200082,  # <|vid_start|>
    "vid_end": 200083,  # <|vid_end|>
    "vid_frame_separator": 200087,  # <|vid_frame_separator|>
    "video": 200091,  # <|video|>       -- carries video embeddings
    "audio": 200038,  # <|audio|>       -- carries audio embeddings; the audio
    # path (expansion, delimiters, tower) lives in musespark_open_audio.py, whose
    # AUDIO_TOKEN_IDS is the audio-side pin of this same id.
}

# Ids whose positions the merge overwrites with encoder rows. Everything else in
# an expansion (markers, timestamp text) keeps its own token embedding.
_EMBED_TOKEN_IDS: frozenset[int] = frozenset((VISION_TOKEN_IDS["patch"], VISION_TOKEN_IDS["video"]))

# Colour an alpha channel is composited over. The two reference runtimes
# disagree here -- one composites over white, the other drops alpha -- and the
# frozen oracle was captured with the compositing one.
RGBA_BACKGROUND_COLOR: tuple[int, int, int] = (255, 255, 255)


class VisionPreprocessError(Exception):
    """Raised when resolution fitting or placeholder expansion fails."""


@dataclass(frozen=True)
class MuseSparkOpenPreprocessConfig:
    """Preprocessing knobs, as read from `vision_config.navit_config`.

    `num_frames_per_video`, `sampling_fps`, `frame_factor`,
    `max_num_tokens_per_frame` and `video_frame_timestamp_format` are absent
    from the shipped checkpoint config, so these defaults are what actually
    runs; the rest match the checkpoint.
    """

    patch_size: int = 14
    patch_temporal: int = 2
    downsample_factor: int = 2
    num_channels: int = 3
    min_num_tokens: int = 1
    max_num_image_tokens: int = 4096
    max_num_video_tokens: int = 4096
    use_per_video_token_budget: bool = False
    image_mean: tuple[float, ...] = (0.5, 0.5, 0.5)
    image_std: tuple[float, ...] = (0.5, 0.5, 0.5)
    video_downsample_factor: int = 2
    num_frames_per_video: int = 96
    sampling_fps: float | None = 2.0
    frame_factor: int | None = 2
    max_num_tokens_per_frame: int = 144
    video_frame_timestamp_format: str = "Time: {s:.1f}s"

    @property
    def token_patch(self) -> int:
        """Canvas alignment in pixels: `patch_size * downsample_factor` = 28."""
        return self.patch_size * self.downsample_factor

    @property
    def patch_dim(self) -> int:
        """Width of one patch vector: `C * patch_temporal * patch_size**2`."""
        return self.num_channels * self.patch_temporal * self.patch_size * self.patch_size

    @property
    def frame_alignment(self) -> int:
        """Sampled frame counts stay divisible by this."""
        return max(self.frame_factor or 2, self.patch_temporal)

    @classmethod
    def from_config(cls, cfg: Any) -> "MuseSparkOpenPreprocessConfig":
        """Build from a vision config, a model config, a dict, or `None`.

        Walks `cfg` -> `cfg.vision_config` -> `.navit_config` and takes the
        first level that defines each key, which is how both reference runtimes
        resolve them.
        """
        if isinstance(cfg, cls):
            return cfg
        levels: list[Any] = []
        node: Any = cfg
        for _ in range(3):
            if node is None:
                break
            levels.append(node)
            node = _lookup(node, "navit_config", _lookup(node, "vision_config"))
        levels.reverse()  # innermost (navit_config) wins

        def get(name: str, default: Any) -> Any:
            for level in levels:
                value = _lookup(level, name)
                if value is not None:
                    return value
            return default

        mean = tuple(float(x) for x in get("image_mean", (0.5, 0.5, 0.5)))
        std = tuple(float(x) for x in get("image_std", (0.5, 0.5, 0.5)))
        return cls(
            patch_size=int(get("patch_size", 14)),
            patch_temporal=int(get("patch_temporal", 2)),
            downsample_factor=int(get("downsample_factor", 2)),
            num_channels=len(mean),
            min_num_tokens=int(get("min_num_image_tokens", 1)),
            max_num_image_tokens=int(get("max_num_image_tokens", 4096)),
            max_num_video_tokens=int(get("max_num_video_tokens", 4096)),
            use_per_video_token_budget=bool(get("use_per_video_token_budget", False)),
            image_mean=mean,
            image_std=std,
            video_downsample_factor=int(get("video_downsample_factor", 2)),
            num_frames_per_video=int(get("num_frames_per_video", 96)),
            sampling_fps=get("sampling_fps", 2.0),
            frame_factor=get("frame_factor", 2),
            max_num_tokens_per_frame=int(get("max_num_tokens_per_frame", 144)),
            video_frame_timestamp_format=get("video_frame_timestamp_format", "Time: {s:.1f}s"),
        )


def _lookup(node: Any, name: str, default: Any = None) -> Any:
    """Read `name` off an attribute bag or a mapping."""
    if isinstance(node, dict):
        return node.get(name, default)
    return getattr(node, name, default)


def _cfg(cfg: Any) -> MuseSparkOpenPreprocessConfig:
    return MuseSparkOpenPreprocessConfig.from_config(cfg)


PROCESSOR_CONFIG_FILE = "processor_config.json"

# HuggingFace-style spellings in `processor_config.json`, per block, mapped
# onto this module's names. Only keys this module acts on appear here; a key
# outside the map is ignored rather than guessed at.
_PROCESSOR_ALIASES: dict[str, dict[str, str]] = {
    "image_processor": {
        "max_image_tokens": "max_num_image_tokens",
    },
    "video_processor": {
        "max_video_frame_tokens": "max_num_tokens_per_frame",
        "fps": "sampling_fps",
    },
}

# Declared by newer exports and deliberately NOT adopted, each with the reason.
# Recorded here rather than left as silence, so the next reader finds out that
# ignoring them was a decision and what would have to change to revisit it.
_PROCESSOR_KEYS_NOT_ADOPTED: dict[str, str] = {
    # 600 frames at a 256-row budget claims 300 * 256 = 76 800 decoder rows for
    # a single video, against a per-item cap (max_num_video_tokens) that this
    # module deliberately does not enforce. Raising the frame cap is only safe
    # once that cap question is settled, which is a separate decision.
    "num_frames": "would raise num_frames_per_video past what any cap bounds",
    # Changes the timestamp TEXT inside the expansion, so it moves prompt
    # tokens. The only captured reference expansion pins the current
    # one-decimal form; nothing measures the declared two-decimal form.
    "frame_timestamp_format": "moves prompt tokens; no reference capture for it",
}


def processor_config_overrides(model_path: str | os.PathLike[str] | None) -> dict[str, Any]:
    """Preprocessing values `model_path`'s `processor_config.json` declares.

    Returns them under this module's names, ready to hand to
    :meth:`MuseSparkOpenPreprocessConfig.from_config`. An absent or unreadable
    file yields `{}` -- a checkpoint that ships no processor config is the
    normal case for older exports and must not become a hard failure.
    """
    if not model_path:
        return {}
    path = os.path.join(str(model_path), PROCESSOR_CONFIG_FILE)
    try:
        with open(path, encoding="utf-8") as handle:
            declared = json.load(handle)
    except (OSError, ValueError):
        return {}
    if not isinstance(declared, dict):
        return {}

    overrides: dict[str, Any] = {}
    for block, aliases in _PROCESSOR_ALIASES.items():
        section = declared.get(block)
        if not isinstance(section, dict):
            continue
        for declared_name, local_name in aliases.items():
            if declared_name in section and section[declared_name] is not None:
                overrides[local_name] = section[declared_name]
    return overrides


def with_processor_config(vision_config: Any, model_path: str | os.PathLike[str] | None) -> Any:
    """`vision_config` carrying whatever the checkpoint declares about preprocessing.

    The open re-export moved the preprocessing knobs out of `config.json`
    (dropping `vision_config.navit_config` wholesale) into
    `processor_config.json`, so a config object built from `config.json`
    alone no longer sees them and every value silently falls back to this
    module's defaults. Reading them here is what lets the CHECKPOINT decide its
    own supported preprocessing rather than a constant in this file.

    Returns `vision_config` UNCHANGED, same object, when the checkpoint
    declares nothing -- so exports without a `processor_config.json` keep
    byte-identical behaviour and stay comparable with captured reference runs.
    Otherwise returns a copy carrying the declared values; the original is not
    mutated, because the tower and the weight mapper share it.

    Accepts either config shape. The serve path hands over the checkpoint's
    `MuseSparkOpenVisionConfig` (a mutable attribute bag), but an already
    resolved :class:`MuseSparkOpenPreprocessConfig` is frozen AND short-circuits
    :meth:`MuseSparkOpenPreprocessConfig.from_config`, so attaching attributes to
    it would both raise and, if it did not, be ignored -- it has to be rebuilt
    with :func:`dataclasses.replace` instead.
    """
    overrides = processor_config_overrides(model_path)
    if not overrides:
        return vision_config
    if isinstance(vision_config, MuseSparkOpenPreprocessConfig):
        fields = {
            name: value
            for name, value in overrides.items()
            if name in vision_config.__dataclass_fields__
        }
        enriched = replace(vision_config, **fields)
    else:
        enriched = copy.copy(vision_config)
        for name, value in overrides.items():
            setattr(enriched, name, value)
    logger.info(
        "MuseSparkOpen: preprocessing values taken from %s: %s",
        PROCESSOR_CONFIG_FILE,
        ", ".join(f"{k}={v!r}" for k, v in sorted(overrides.items())),
    )
    return enriched


@dataclass
class ExpandedPrompt:
    """A prompt after placeholder expansion."""

    token_ids: list[int]
    # True exactly on the positions the merge overwrites with encoder rows.
    is_embed: list[bool]
    # (modality, start, stop) of each expanded run, in prompt order.
    spans: list[tuple[str, int, int]] = field(default_factory=list)


# --------------------------------------------------------------------------
# Resolution fitting
# --------------------------------------------------------------------------


def _least_distortion_grid(
    ideal_nph: float,
    ideal_npw: float,
    img_h: int,
    img_w: int,
    token_patch: int,
    min_num_tokens: int,
    max_num_tokens: int,
) -> tuple[int, int]:
    """Floor/ceil cell grid with the smallest aspect-ratio distortion.

    Candidates are the four floor/ceil combinations; those with a non-positive
    side or a token count outside the budget are dropped, and the survivor
    closest to the source aspect ratio wins.

    The candidate list is built as `list(set(product(...)))` and scanned with
    `min`, exactly as the reference does, because roughly one image shape in
    150 produces an *exact* tie -- a near-square image whose ideal grid falls
    between two integers ties `(floor, floor)` against `(ceil, ceil)`, and
    those two differ by up to a factor of four in token count. Which one wins is
    then decided by set iteration order, so the construction is transcribed
    rather than tidied into a sort: a "cleaner" ordering silently changes the
    prompt length of every square image. (Deterministic in practice: tuples of
    small ints hash identically on every CPython build and are inserted in a
    fixed order.)
    """
    candidates = list(
        set(
            itertools.product(
                (math.floor(ideal_nph), math.ceil(ideal_nph)),
                (math.floor(ideal_npw), math.ceil(ideal_npw)),
            )
        )
    )
    candidates = [
        (nph, npw)
        for nph, npw in candidates
        if nph > 0 and npw > 0 and min_num_tokens <= nph * npw <= max_num_tokens
    ]
    if not candidates:
        raise VisionPreprocessError(
            f"least distortion fit failed on ideal_nph={ideal_nph}, "
            f"ideal_npw={ideal_npw}, img_h={img_h}, img_w={img_w}, "
            f"max_num_tokens={max_num_tokens}"
        )

    source_aspect = img_h / img_w

    def distortion(grid: tuple[int, int]) -> float:
        nph, npw = grid
        # Square patches, so the token_patch factors cancel; kept explicit to
        # mirror the reference formula.
        return abs((nph * token_patch) / (npw * token_patch) - source_aspect)

    return min(candidates, key=distortion)


def _least_distance_grid(
    ideal_nph: float,
    ideal_npw: float,
    token_patch: int,
    min_num_tokens: int,
    max_num_tokens: int,
) -> tuple[int, int]:
    """Fallback fit: nearest grid point between the min/max token hyperbolas.

    Only reachable for aspect ratios so extreme that every floor/ceil
    combination violates the token budget.
    """
    nph_axis = np.arange(1, max_num_tokens + 1)
    npw_upper = np.floor(max_num_tokens / nph_axis)
    npw_lower = np.ceil(min_num_tokens / nph_axis)
    valid = npw_upper >= npw_lower
    if not valid.any():
        raise VisionPreprocessError(
            "least distance fit failed: no valid grid point between "
            f"{min_num_tokens} and {max_num_tokens}"
        )

    # Round before clipping: truncating after the clip is off by one for
    # fractional parts above 0.5.
    npw_axis = np.clip(round(ideal_npw), a_min=npw_lower, a_max=npw_upper)
    nph_axis = nph_axis[valid]
    npw_axis = npw_axis[valid]

    l2 = ((nph_axis - ideal_nph) * token_patch) ** 2 + ((npw_axis - ideal_npw) * token_patch) ** 2
    best = int(np.argmin(l2))
    return int(nph_axis[best]), int(npw_axis[best])


def find_resize_target(
    img_h: int,
    img_w: int,
    token_patch: int,
    min_num_tokens: int,
    max_num_tokens: int,
    resize_to_max_canvas: bool = False,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Resolve the resize target that best preserves the aspect ratio.

    `token_patch` is the 28-px token cell, not the 14-px encoder patch.
    Returns `((canvas_h, canvas_w), (nph, npw))` -- height first, matching the
    tensor layout the rest of the module uses.

    When the ideal grid overshoots `max_num_tokens` the fractional grid is
    rescaled so its product is exactly `max_num_tokens` at the original aspect
    ratio, and the floor/ceil fit then lands inside the budget; the symmetric
    branch grows an undersized image up to `min_num_tokens`.
    """
    if img_h <= 0 or img_w <= 0:
        raise VisionPreprocessError(f"image must have positive extent, got {img_h}x{img_w}")

    ideal_nph = img_h / token_patch
    ideal_npw = img_w / token_patch
    ideal_ratio = ideal_npw / ideal_nph

    if resize_to_max_canvas or ideal_nph * ideal_npw > max_num_tokens:
        ideal_nph = (max_num_tokens / ideal_ratio) ** 0.5
        ideal_npw = ideal_nph * ideal_ratio
    elif ideal_nph * ideal_npw < min_num_tokens:
        ideal_nph = (min_num_tokens / ideal_ratio) ** 0.5
        ideal_npw = ideal_nph * ideal_ratio

    try:
        nph, npw = _least_distortion_grid(
            ideal_nph, ideal_npw, img_h, img_w, token_patch, min_num_tokens, max_num_tokens
        )
    except VisionPreprocessError:
        nph, npw = _least_distance_grid(
            ideal_nph, ideal_npw, token_patch, min_num_tokens, max_num_tokens
        )

    if not min_num_tokens <= nph * npw <= max_num_tokens:
        raise VisionPreprocessError(
            f"token count {nph * npw} outside bounds [{min_num_tokens}, {max_num_tokens}]"
        )

    return (nph * token_patch, npw * token_patch), (nph, npw)


def least_distortion_fit(
    width: int,
    height: int,
    *,
    patch_size: int = 14,
    downsample_factor: int = 2,
    min_tokens: int = 1,
    max_tokens: int = 4096,
) -> tuple[int, int]:
    """Canvas size for one image, both sides a multiple of 28 px.

    Argument and result order are **width first**, matching `PIL.Image.size`;
    everything else in this module is height-first because that is the tensor
    and `grid_thw` layout. Returns `(canvas_width, canvas_height)`.
    """
    (canvas_h, canvas_w), _ = find_resize_target(
        img_h=height,
        img_w=width,
        token_patch=patch_size * downsample_factor,
        min_num_tokens=min_tokens,
        max_num_tokens=max_tokens,
    )
    return canvas_w, canvas_h


# --------------------------------------------------------------------------
# Pixel handling
# --------------------------------------------------------------------------


def to_rgb(image: Image.Image) -> Image.Image:
    """Convert a PIL image of any mode to 3-channel RGB.

    RGBA alpha is composited over :data:`RGBA_BACKGROUND_COLOR`; every other
    non-RGB mode takes `.convert("RGB")`. Compositing must happen before the
    resize: LANCZOS on an RGBA image interpolates against fully-transparent
    pixels whose RGB is arbitrary, which bleeds them into the visible edge.
    """
    if image.mode == "RGB":
        return image
    if image.mode == "RGBA":
        composited = Image.new("RGB", image.size, RGBA_BACKGROUND_COLOR)
        composited.paste(image, mask=image.getchannel("A"))
        return composited
    return image.convert("RGB")


def resize_lanczos(image: Image.Image, canvas_h: int, canvas_w: int) -> torch.Tensor:
    """PIL LANCZOS resize to a uint8 `[C, H, W]` tensor."""
    resized = image.resize((canvas_w, canvas_h), resample=Image.Resampling.LANCZOS)
    return torch.from_numpy(np.asarray(resized).copy()).permute(2, 0, 1)


def normalize_pixels(pixels: torch.Tensor, cfg: Any = None) -> torch.Tensor:
    """Scale uint8 pixels to `[0, 1]`, then apply mean/std normalization.

    The cast is an explicit `.float()`: dividing an integral tensor by a
    Python float resolves to `torch.get_default_dtype()`, which a worker may
    have set to the model dtype, and this stage is fp32.
    """
    config = _cfg(cfg)
    if pixels.dtype != torch.uint8:
        raise VisionPreprocessError(f"expected uint8 pixels for normalization, got {pixels.dtype}")
    if pixels.shape[-3] != len(config.image_mean):
        raise VisionPreprocessError(
            f"expected {len(config.image_mean)}-channel pixels, got "
            f"{pixels.shape[-3]} (shape {tuple(pixels.shape)})"
        )

    scaled = pixels.float().div_(255.0)
    if len(set(config.image_mean)) == 1 and len(set(config.image_std)) == 1:
        # Channel-uniform mean/std as Python scalars: vectorizes better and is
        # bit-identical to the tensor form.
        return scaled.sub_(config.image_mean[0]).div_(config.image_std[0])
    mean = torch.tensor(config.image_mean, dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor(config.image_std, dtype=torch.float32).view(-1, 1, 1)
    return scaled.sub_(mean).div_(std)


def expand_temporal(pixel_values: torch.Tensor, cfg: Any = None) -> torch.Tensor:
    """Give a still image the temporal extent the patch vector needs.

    `patch_dim = C * patch_temporal * patch_size**2`, so a single frame is
    *duplicated* to `patch_temporal` identical copies -- never zero-padded,
    which would halve the energy of every patch vector. `[C, H, W]` in,
    `[patch_temporal, C, H, W]` out; a tensor that already has a temporal axis
    of a valid size passes through.
    """
    config = _cfg(cfg)
    if pixel_values.dim() == 3:
        pixel_values = pixel_values.unsqueeze(0)
    if pixel_values.dim() != 4:
        raise VisionPreprocessError(
            f"expected [C, H, W] or [T, C, H, W], got shape {tuple(pixel_values.shape)}"
        )
    if pixel_values.shape[0] == 1 and config.patch_temporal > 1:
        pixel_values = pixel_values.expand(config.patch_temporal, -1, -1, -1)
    if pixel_values.shape[0] % config.patch_temporal != 0:
        raise VisionPreprocessError(
            f"frame count {pixel_values.shape[0]} is not a multiple of "
            f"patch_temporal {config.patch_temporal}"
        )
    return pixel_values


def patchify(frames: torch.Tensor, cfg: Any = None) -> torch.Tensor:
    """Vectorize `[T, C, H, W]` frames into `[num_patches, patch_dim]`.

    `T` must be a multiple of `patch_temporal`; consecutive frames are fused
    into clips. Each output row is one 14x14 patch of one clip, flattened
    `[t][c][py][px]` with the temporal index outermost, and the rows run
    clip-major then patch row-major (`h` outer, `w` inner) -- the order the
    tower's positional embedding and 2D RoPE index with.
    """
    config = _cfg(cfg)
    if frames.dim() != 4:
        raise VisionPreprocessError(
            f"expected [T, C, H, W] frames, got shape {tuple(frames.shape)}"
        )
    num_frames, channels, height, width = frames.shape
    patch = config.patch_size
    temporal = config.patch_temporal
    if num_frames % temporal != 0:
        raise VisionPreprocessError(
            f"frame count {num_frames} is not a multiple of patch_temporal {temporal}"
        )
    if height % patch or width % patch:
        raise VisionPreprocessError(
            f"canvas {height}x{width} is not a multiple of patch_size {patch}"
        )

    clips = num_frames // temporal
    grid_h, grid_w = height // patch, width // patch
    out = frames.reshape(clips, temporal, channels, grid_h, patch, grid_w, patch)
    out = out.permute(0, 3, 5, 1, 2, 4, 6)
    return out.reshape(clips * grid_h * grid_w, temporal * channels * patch * patch)


# --------------------------------------------------------------------------
# Grid arithmetic
# --------------------------------------------------------------------------


def _check_grid(
    grid_thw: Sequence[int], cfg: MuseSparkOpenPreprocessConfig
) -> tuple[int, int, int]:
    if len(grid_thw) != 3:
        raise VisionPreprocessError(f"grid_thw must be (t, h, w), got {tuple(grid_thw)}")
    t, h, w = (int(x) for x in grid_thw)
    if t <= 0 or h <= 0 or w <= 0:
        raise VisionPreprocessError(f"grid_thw must be positive, got {(t, h, w)}")
    ds = cfg.downsample_factor
    if h % ds or w % ds:
        raise VisionPreprocessError(
            f"grid_thw spatial extent {(h, w)} is not a multiple of the "
            f"downsample factor {ds}; the pixel shuffle needs an even grid"
        )
    return t, h, w


def num_placeholder_rows(grid_thw: Sequence[int], cfg: Any = None) -> int:
    """Decoder rows one medium occupies: `t * (h // 2) * (w // 2)`.

    Equivalently `(H // 14) * (W // 14) // 4` per clip. This MUST equal the
    tower's output first dim for the same medium: the encoder wrapper compares
    the two, and a mismatch silently disables multimodal embedding caching, so
    the tower re-runs on every chunked-prefill chunk.
    """
    config = _cfg(cfg)
    t, h, w = _check_grid(grid_thw, config)
    ds = config.downsample_factor
    return t * (h // ds) * (w // ds)


def num_patches(grid_thw: Sequence[int], cfg: Any = None) -> int:
    """Tower patch rows one medium occupies: `t * h * w`."""
    config = _cfg(cfg)
    t, h, w = _check_grid(grid_thw, config)
    return t * h * w


def patch_dim(cfg: Any = None) -> int:
    """Width of one patch vector: `C * patch_temporal * patch_size**2` = 1176."""
    return _cfg(cfg).patch_dim


# --------------------------------------------------------------------------
# Preprocessing entry points
# --------------------------------------------------------------------------


def preprocess_image(
    image: Image.Image,
    cfg: Any = None,
    resize_to_max_canvas: bool = False,
) -> tuple[torch.Tensor, tuple[int, int, int]]:
    """Preprocess one image for the NaViT tower.

    Returns `(pixel_values, grid_thw)` with `pixel_values` fp32
    `[num_patches, patch_dim]` and `grid_thw = (1, H // 14, W // 14)` of the
    resized canvas. Each image is sized independently: there is no shared canvas
    and no global token budget across the images of one prompt, so multi-image
    prompts simply call this once per image.
    """
    config = _cfg(cfg)
    image = to_rgb(image)
    width, height = image.size

    (canvas_h, canvas_w), _ = find_resize_target(
        img_h=height,
        img_w=width,
        token_patch=config.token_patch,
        min_num_tokens=config.min_num_tokens,
        max_num_tokens=config.max_num_image_tokens,
        resize_to_max_canvas=resize_to_max_canvas,
    )

    pixels = normalize_pixels(resize_lanczos(image, canvas_h, canvas_w), config)
    frames = expand_temporal(pixels, config)
    grid_thw = (1, canvas_h // config.patch_size, canvas_w // config.patch_size)
    return patchify(frames, config), grid_thw


def sample_video_frame_indices(
    num_frames: int,
    source_fps: float,
    cfg: Any = None,
    total_num_frames: int | None = None,
) -> list[int]:
    """Indices to keep from a decoded video, as positions into `frames`.

    `target = int(total_num_frames * sampling_fps / source_fps)`, clamped into
    `[frame_alignment, min(num_frames_per_video, num_frames)]`, floored to a
    multiple of `frame_alignment`, then `np.linspace` over
    `[0, num_frames - 1]`. Uniform, not a stride, so the last frame is always
    kept.

    `total_num_frames` defaults to `num_frames` and differs only when the
    decoder already subsampled: the fps target is then computed against the
    *source* frame count while the linspace runs over what was decoded, which is
    what stops a pre-sampling decoder from sampling twice.
    """
    config = _cfg(cfg)
    if num_frames <= 0:
        raise VisionPreprocessError("video must have at least one frame")
    if total_num_frames is None:
        total_num_frames = num_frames

    align = config.frame_alignment
    fps = config.sampling_fps
    if fps is not None and fps > 0 and source_fps > 0:
        target = int(total_num_frames * fps / source_fps)
    else:
        target = num_frames
    target = min(max(target, align), config.num_frames_per_video, num_frames)
    target = max(align, (target // align) * align)

    if num_frames <= target:
        return list(range(num_frames))
    return [int(i) for i in np.linspace(0, num_frames - 1, target, dtype=int)]


def _frame_to_pil(frame: Any) -> Image.Image:
    """Accept a PIL image or a uint8 `[H, W, C]` array/tensor."""
    if isinstance(frame, Image.Image):
        return to_rgb(frame)
    array = frame.numpy() if isinstance(frame, torch.Tensor) else np.asarray(frame)
    if array.ndim != 3 or array.shape[-1] not in (1, 3, 4):
        raise VisionPreprocessError(f"expected an [H, W, C] uint8 frame, got shape {array.shape}")
    if array.shape[-1] == 4:
        raise VisionPreprocessError(
            "video frames must be 3-channel RGB, got 4 channels (alpha); "
            "whether the alpha is straight or premultiplied is not recoverable "
            "from a bare frame array, so it cannot be composited here"
        )
    if array.dtype != np.uint8:
        raise VisionPreprocessError(f"expected uint8 video frames, got {array.dtype}")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return Image.fromarray(np.ascontiguousarray(array))


def _aligned_frame_count(num_kept: int, config: Any) -> int:
    """Frames a video keeps: divisible by the video downsample factor.

    A video with fewer frames than that repeats its last frame rather than
    losing it, so the count never falls to zero.
    """
    vds = config.video_downsample_factor
    return (num_kept // vds) * vds or vds


def _video_token_budget(aligned: int, config: Any) -> int:
    """Per-clip token budget the canvas fit is solved against."""
    if config.max_num_tokens_per_frame > 0:
        # This branch wins outright over max_num_video_tokens, matching the
        # reference: a 96-frame video really does exceed the declared cap.
        return config.max_num_tokens_per_frame
    if config.use_per_video_token_budget:
        clips = aligned // config.patch_temporal
        return config.max_num_video_tokens // clips if clips > 0 else config.max_num_video_tokens
    return config.max_num_video_tokens


def video_grid_thw(
    num_frames: int,
    source_fps: float,
    frame_width: int,
    frame_height: int,
    cfg: Any = None,
    *,
    total_num_frames: int | None = None,
    resize_to_max_canvas: bool = False,
) -> tuple[int, int, int]:
    """The `grid_thw` :func:`preprocess_video` will produce, from geometry alone.

    The runtime asks how many prompt tokens a video will occupy *before* it is
    preprocessed, once per item per request. Answering by calling
    :func:`preprocess_video` would resize and patchify every frame to count
    them; answering with a second copy of the sampling rule would drift from it.
    So both go through the same three decisions -- which frames survive
    sampling, how many of those survive alignment, and what token budget the
    canvas is fitted to -- and only the pixel work is left to
    :func:`preprocess_video`.
    """
    config = _cfg(cfg)
    if num_frames <= 0:
        raise VisionPreprocessError("video must have at least one frame")
    keep = sample_video_frame_indices(num_frames, source_fps, config, total_num_frames)
    aligned = _aligned_frame_count(len(keep), config)
    (canvas_h, canvas_w), _ = find_resize_target(
        img_h=frame_height,
        img_w=frame_width,
        token_patch=config.token_patch,
        min_num_tokens=config.min_num_tokens,
        max_num_tokens=_video_token_budget(aligned, config),
        resize_to_max_canvas=resize_to_max_canvas,
    )
    return (
        aligned // config.patch_temporal,
        canvas_h // config.patch_size,
        canvas_w // config.patch_size,
    )


def preprocess_video(
    frames: Sequence[Any],
    fps: float,
    cfg: Any = None,
    *,
    frame_indices: Sequence[int] | None = None,
    total_num_frames: int | None = None,
    resize_to_max_canvas: bool = False,
) -> tuple[torch.Tensor, tuple[int, int, int], list[float]]:
    """Preprocess a decoded video for the NaViT tower.

    `frames` are the decoded frames (PIL images or uint8 `[H, W, C]`
    arrays) and `fps` is the source frame rate. Sampling happens here: 2.0
    fps, at most 96 frames, floored to a multiple of 2, `np.linspace`.

    Returns `(pixel_values, grid_thw, clip_timestamps)` with `pixel_values`
    fp32 `[num_patches, patch_dim]`, `grid_thw = (num_clips, H // 14,
    W // 14)` and one timestamp per clip, taken from the clip's *first* frame.

    `frame_indices` / `total_num_frames` are for a decoder that already
    subsampled: `frame_indices` are the original indices of the frames handed
    in (timestamps are `original_index / fps`) and `total_num_frames` is the
    source frame count the fps target is computed against. Both default to
    "`frames` is the whole video".
    """
    config = _cfg(cfg)
    if len(frames) == 0:
        raise VisionPreprocessError("video must have at least one frame")
    if frame_indices is None:
        frame_indices = range(len(frames))
    if len(frame_indices) != len(frames):
        raise VisionPreprocessError(
            f"got {len(frames)} frames but {len(frame_indices)} frame indices"
        )

    keep = sample_video_frame_indices(len(frames), fps, config, total_num_frames)
    kept_frames = [frames[i] for i in keep]
    timestamps = [float(frame_indices[i]) / fps if fps > 0 else 0.0 for i in keep]

    aligned = _aligned_frame_count(len(kept_frames), config)
    if aligned > len(kept_frames):
        pad = aligned - len(kept_frames)
        kept_frames = list(kept_frames) + [kept_frames[-1]] * pad
        timestamps = timestamps + [timestamps[-1]] * pad
    kept_frames = kept_frames[:aligned]
    timestamps = timestamps[:aligned]

    pil_frames = [_frame_to_pil(frame) for frame in kept_frames]
    frame_w, frame_h = pil_frames[0].size

    budget = _video_token_budget(aligned, config)

    (canvas_h, canvas_w), _ = find_resize_target(
        img_h=frame_h,
        img_w=frame_w,
        token_patch=config.token_patch,
        min_num_tokens=config.min_num_tokens,
        max_num_tokens=budget,
        resize_to_max_canvas=resize_to_max_canvas,
    )

    resized = torch.stack([resize_lanczos(frame, canvas_h, canvas_w) for frame in pil_frames])
    pixels = normalize_pixels(resized, config)

    num_clips = aligned // config.patch_temporal
    clip_timestamps = [timestamps[i * config.patch_temporal] for i in range(num_clips)]
    grid_thw = (num_clips, canvas_h // config.patch_size, canvas_w // config.patch_size)
    return patchify(pixels, config), grid_thw, clip_timestamps


# --------------------------------------------------------------------------
# Placeholder expansion
# --------------------------------------------------------------------------


def image_placeholder_ids(grid_thw: Sequence[int], cfg: Any = None) -> list[int]:
    """`<|image_start|> + <|patch|> * N + <|image_end|>`.

    `N = num_placeholder_rows(grid_thw)` = `(H // 14) * (W // 14) // 4`.
    There are no row, column or tile separators and no newline tokens, and
    `<|image|>` (200090) is not emitted.
    """
    config = _cfg(cfg)
    rows = num_placeholder_rows(grid_thw, config)
    return (
        [VISION_TOKEN_IDS["image_start"]]
        + [VISION_TOKEN_IDS["patch"]] * rows
        + [VISION_TOKEN_IDS["image_end"]]
    )


def _encode_text(tokenizer: Any, text: str) -> list[int]:
    """Tokenize ordinary text with whatever tokenizer object we were handed.

    Accepts an HF tokenizer, a raw `tokenizers.Tokenizer` (whose `encode`
    returns an `Encoding`), or a plain `str -> ids` callable. Special tokens
    are never added: the timestamp is ordinary text in the middle of a prompt.
    """
    if tokenizer is None:
        raise VisionPreprocessError(
            "video placeholder expansion needs a tokenizer for the timestamp text"
        )
    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        if not callable(tokenizer):
            raise VisionPreprocessError(
                f"tokenizer {type(tokenizer).__name__} has no encode() and is not callable"
            )
        encoded = tokenizer(text)
    else:
        try:
            encoded = encode(text, add_special_tokens=False)
        except TypeError:
            encoded = encode(text)
    ids = getattr(encoded, "ids", encoded)
    return [int(i) for i in ids]


def video_placeholder_ids(
    grid_thw: Sequence[int],
    timestamps: Sequence[float],
    cfg: Any = None,
    tokenizer: Any = None,
) -> list[int]:
    """`<|vid_start|>`, the per-clip body, then `<|vid_end|>`.

    Per clip `c` of `patch_temporal` frames: a
    `<|vid_frame_separator|>` for every clip after the first, the timestamp
    text `"Time: {s:.1f}s"` tokenized as ordinary text with no trailing
    newline, and `M = (h // 2) * (w // 2)` `<|video|>` fillers.
    """
    config = _cfg(cfg)
    t, h, w = _check_grid(grid_thw, config)
    if len(timestamps) != t:
        raise VisionPreprocessError(f"grid_thw has {t} clips but {len(timestamps)} timestamps")
    ds = config.downsample_factor
    per_clip = (h // ds) * (w // ds)

    ids: list[int] = [VISION_TOKEN_IDS["vid_start"]]
    for clip in range(t):
        if clip:
            ids.append(VISION_TOKEN_IDS["vid_frame_separator"])
        ids.extend(
            _encode_text(
                tokenizer, config.video_frame_timestamp_format.format(s=float(timestamps[clip]))
            )
        )
        ids.extend([VISION_TOKEN_IDS["video"]] * per_clip)
    ids.append(VISION_TOKEN_IDS["vid_end"])
    return ids


def is_embed_mask(token_ids: Sequence[int]) -> list[bool]:
    """True exactly where the merge overwrites a position with an encoder row.

    Markers and timestamp text keep their own token embeddings, so only
    `<|patch|>` and `<|video|>` are True.
    """
    return [int(i) in _EMBED_TOKEN_IDS for i in token_ids]


def _find_next_marker(
    token_ids: Sequence[int],
    start: int,
    markers: Sequence[tuple[str, tuple[int, ...]]],
) -> tuple[str, int, int] | None:
    """Leftmost occurrence of any marker at or after `start`.

    Ties on position break toward the longest marker, so a marker that is a
    prefix of another cannot shadow it.
    """
    best: tuple[str, int, int] | None = None
    for modality, marker in markers:
        width = len(marker)
        if not width:
            continue
        for pos in range(start, len(token_ids) - width + 1):
            if tuple(token_ids[pos : pos + width]) == marker:
                if best is None or pos < best[1] or (pos == best[1] and width > best[2] - best[1]):
                    best = (modality, pos, pos + width)
                break
    return best


def expand_prompt_placeholders(
    prompt_token_ids: Sequence[int],
    expansions: Sequence[tuple[str, Sequence[int]]],
    image_marker_ids: Iterable[int] = (),
    video_marker_ids: Iterable[int] = (),
) -> ExpandedPrompt:
    """Replace every `<image>` / `<video>` marker run, in prompt order.

    `expansions` is the media of the prompt in order, each as
    `("image" | "video", ids)` from :func:`image_placeholder_ids` /
    :func:`video_placeholder_ids`. The i-th marker of a modality takes the i-th
    expansion of that modality. Multiple images per prompt are the normal case,
    so a count mismatch in either direction is an error, not a silent
    truncation.

    `image_marker_ids` / `video_marker_ids` are the *tokenized* forms of the
    literal strings the chat template emits; they are ordinary text, not special
    tokens, so each usually spans several ids.
    """
    markers: list[tuple[str, tuple[int, ...]]] = []
    if image_marker_ids:
        markers.append(("image", tuple(image_marker_ids)))
    if video_marker_ids:
        markers.append(("video", tuple(video_marker_ids)))
    if not markers:
        raise VisionPreprocessError("no marker token ids given; nothing could be expanded")

    queues: dict[str, list[Sequence[int]]] = {"image": [], "video": []}
    for modality, ids in expansions:
        if modality not in queues:
            raise VisionPreprocessError(f"unknown modality {modality!r}")
        queues[modality].append(ids)
    consumed = {"image": 0, "video": 0}

    out_ids: list[int] = []
    spans: list[tuple[str, int, int]] = []

    cursor = 0
    while True:
        found = _find_next_marker(prompt_token_ids, cursor, markers)
        if found is None:
            break
        modality, start, stop = found
        out_ids.extend(prompt_token_ids[cursor:start])

        index = consumed[modality]
        if index >= len(queues[modality]):
            raise VisionPreprocessError(
                f"prompt has more <{modality}> markers than {modality} items "
                f"({len(queues[modality])})"
            )
        ids = queues[modality][index]
        spans.append((modality, len(out_ids), len(out_ids) + len(ids)))
        out_ids.extend(ids)
        consumed[modality] = index + 1
        cursor = stop

    out_ids.extend(prompt_token_ids[cursor:])

    for modality, queue in queues.items():
        if consumed[modality] != len(queue):
            raise VisionPreprocessError(
                f"prompt has {consumed[modality]} <{modality}> markers but "
                f"{len(queue)} {modality} items were provided"
            )

    return ExpandedPrompt(token_ids=out_ids, is_embed=is_embed_mask(out_ids), spans=spans)
