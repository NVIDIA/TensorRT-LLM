# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MiniMax-M3 VL vision-side modules and multimodal runtime wiring.

Contains the vision tower (patch embed -> pre-LN -> 32 encoder layers
with 3D RoPE attention -> projector -> patch merger), the placeholder
expansion helpers (``image_grid_thw`` / ``video_grid_thw`` ->
``grid_t * (grid_h // merge) * (grid_w // merge)`` token count), the
per-item ``pad_value`` rewrite, and the embedding-merge helper that
replaces placeholder positions in ``input_embeds`` with the visual
features produced by the tower.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# MiniMax-M3 VL special tokens.
#
# The image / video placeholder token IDs live on the HF model config as
# ``image_token_index`` / ``video_token_index`` — read those directly.
# The vision-start / vision-end markers are not exposed on the config; they
# are resolved through ``tokenizer.convert_tokens_to_ids`` using the textual
# tokens below, which are the canonical strings carried by the checkpoint's
# ``tokenizer_config.json`` ``added_tokens_decoder``. The checkpoint
# processor (``processing_minimax.py``) wraps BOTH images and videos with
# the *image* start/end markers; separate video start/end markers exist in
# the tokenizer but are intentionally unused by the serving path.
# ---------------------------------------------------------------------------


MINIMAX_M3_VL_VISION_START_TOKEN = "]<]start of image[>["  # nosec B105 - vision delimiter token, not a password
MINIMAX_M3_VL_VISION_END_TOKEN = "]<]end of image[>["  # nosec B105 - vision delimiter token, not a password


# ---------------------------------------------------------------------------
# Multimodal placeholder-count math.
# ---------------------------------------------------------------------------


def compute_visual_token_count(
    grid_t: int,
    grid_h: int,
    grid_w: int,
    spatial_merge_size: int,
) -> int:
    """Return the number of placeholder tokens this image/video must consume.

    After the Conv3d patch embedding produces a ``grid_t * grid_h * grid_w``
    token grid, the patch merger compresses by ``spatial_merge_size**2`` so
    the merged grid carries ``grid_t * (grid_h // spatial_merge_size) *
    (grid_w // spatial_merge_size)`` tokens. Each merged token consumes
    exactly one placeholder position in the text input.

    Raises ``ValueError`` if ``grid_h`` or ``grid_w`` is not a multiple
    of ``spatial_merge_size`` (the processor enforces that via the
    ``get_hw_multiple_of`` resize so divisibility is part of the
    contract).
    """
    if grid_t <= 0 or grid_h <= 0 or grid_w <= 0:
        raise ValueError(
            f"compute_visual_token_count requires positive grid dims; "
            f"got grid_thw=({grid_t}, {grid_h}, {grid_w})"
        )
    if spatial_merge_size <= 0:
        raise ValueError(
            f"compute_visual_token_count requires positive spatial_merge_size; "
            f"got {spatial_merge_size}"
        )
    if grid_h % spatial_merge_size != 0 or grid_w % spatial_merge_size != 0:
        raise ValueError(
            "compute_visual_token_count requires grid_h/grid_w to be a "
            f"multiple of spatial_merge_size={spatial_merge_size}; got "
            f"grid_thw=({grid_t}, {grid_h}, {grid_w})"
        )
    return grid_t * (grid_h // spatial_merge_size) * (grid_w // spatial_merge_size)


def compute_visual_token_counts(
    grid_thws: Sequence[Sequence[int]],
    spatial_merge_size: int,
) -> List[int]:
    """Per-item placeholder counts for a batch of ``grid_thw`` triples."""
    counts: List[int] = []
    for grid_thw in grid_thws:
        if len(grid_thw) != 3:
            raise ValueError(f"grid_thw must be a (t, h, w) triple; got {tuple(grid_thw)}")
        counts.append(
            compute_visual_token_count(grid_thw[0], grid_thw[1], grid_thw[2], spatial_merge_size)
        )
    return counts


# ---------------------------------------------------------------------------
# Placeholder expansion + multimodal embedding merge.
# ---------------------------------------------------------------------------


def expand_multimodal_placeholders(
    input_ids: Sequence[int],
    image_token_id: Optional[int],
    image_repeats: Sequence[int],
    video_token_id: Optional[int] = None,
    video_repeats: Sequence[int] = (),
) -> Tuple[List[int], List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Expand single-token image/video placeholders into per-item spans.

    The HF processor emits a single image placeholder token per image
    (``image_token_id``) and a single video placeholder token per video
    (``video_token_id``). The runtime then rewrites each single token
    into a span of N placeholder tokens whose length matches the visual
    feature count for that item.

    Returns ``(expanded_input_ids, image_spans, video_spans)``:
    - ``expanded_input_ids`` is the new token list with every single
      placeholder replaced by ``repeat`` copies of the same token id.
    - ``image_spans`` lists ``(start, end_exclusive)`` ranges in the
      expanded sequence where each image's visual embeddings will be
      placed.
    - ``video_spans`` is the same for videos.

    Image and video placeholders are matched in left-to-right order;
    ``image_repeats`` is consumed in order against image placeholders
    and ``video_repeats`` is consumed in order against video
    placeholders. ``ValueError`` is raised if the counts disagree.
    """
    image_repeats = list(image_repeats)
    video_repeats = list(video_repeats)

    expanded: List[int] = []
    image_spans: List[Tuple[int, int]] = []
    video_spans: List[Tuple[int, int]] = []
    image_idx = 0
    video_idx = 0

    for token in input_ids:
        if image_token_id is not None and token == image_token_id:
            if image_idx >= len(image_repeats):
                raise ValueError(
                    "expand_multimodal_placeholders found more image tokens "
                    f"({image_idx + 1}) than image_repeats entries "
                    f"({len(image_repeats)})"
                )
            repeat = int(image_repeats[image_idx])
            if repeat <= 0:
                raise ValueError(f"image_repeats[{image_idx}] must be positive; got {repeat}")
            start = len(expanded)
            expanded.extend([image_token_id] * repeat)
            image_spans.append((start, start + repeat))
            image_idx += 1
            continue
        if video_token_id is not None and token == video_token_id:
            if video_idx >= len(video_repeats):
                raise ValueError(
                    "expand_multimodal_placeholders found more video tokens "
                    f"({video_idx + 1}) than video_repeats entries "
                    f"({len(video_repeats)})"
                )
            repeat = int(video_repeats[video_idx])
            if repeat <= 0:
                raise ValueError(f"video_repeats[{video_idx}] must be positive; got {repeat}")
            start = len(expanded)
            expanded.extend([video_token_id] * repeat)
            video_spans.append((start, start + repeat))
            video_idx += 1
            continue
        expanded.append(int(token))

    if image_idx != len(image_repeats):
        raise ValueError(
            f"expand_multimodal_placeholders consumed {image_idx} image "
            f"placeholders but image_repeats has {len(image_repeats)} entries"
        )
    if video_idx != len(video_repeats):
        raise ValueError(
            f"expand_multimodal_placeholders consumed {video_idx} video "
            f"placeholders but video_repeats has {len(video_repeats)} entries"
        )

    return expanded, image_spans, video_spans


def merge_multimodal_embeddings(
    input_embeds: torch.Tensor,
    *,
    image_embeds: Optional[torch.Tensor] = None,
    image_spans: Sequence[Tuple[int, int]] = (),
    video_embeds: Optional[torch.Tensor] = None,
    video_spans: Sequence[Tuple[int, int]] = (),
) -> torch.Tensor:
    """Replace placeholder spans in ``input_embeds`` with visual features.

    ``input_embeds`` is the text-embedded prompt of shape
    ``[seq, hidden]`` (no batch dim — multimodal inputs in M3 are flat
    per request). ``image_embeds`` (resp. ``video_embeds``) is a single
    concatenation of all images' (resp. videos') merged visual features
    in the same order as ``image_spans`` (resp. ``video_spans``). For
    each ``(start, end)`` span we slice the matching chunk out of the
    concatenated visual features and assign it into
    ``input_embeds[start:end]`` (returning a new tensor that does not
    alias ``input_embeds``).

    Raises ``ValueError`` if a span length disagrees with the visual
    feature count for that item, or if total visual rows do not match
    the sum of span lengths.
    """
    if input_embeds.dim() != 2:
        raise ValueError(
            f"input_embeds must be [seq, hidden]; got shape {tuple(input_embeds.shape)}"
        )

    out = input_embeds.clone()

    def _splice(
        embeds: Optional[torch.Tensor],
        spans: Sequence[Tuple[int, int]],
        label: str,
    ) -> None:
        if not spans:
            if embeds is not None and embeds.numel() != 0:
                raise ValueError(
                    f"merge_multimodal_embeddings received nonempty {label}_embeds "
                    f"({embeds.shape}) but no {label}_spans"
                )
            return
        if embeds is None:
            raise ValueError(f"merge_multimodal_embeddings has {label}_spans but no {label}_embeds")
        if embeds.dim() != 2:
            raise ValueError(
                f"{label}_embeds must be [num_visual_tokens, hidden]; got shape "
                f"{tuple(embeds.shape)}"
            )
        if embeds.shape[-1] != out.shape[-1]:
            raise ValueError(
                f"{label}_embeds hidden_size {embeds.shape[-1]} must match input "
                f"hidden_size {out.shape[-1]}"
            )
        expected_total = sum(end - start for start, end in spans)
        if expected_total != embeds.shape[0]:
            raise ValueError(
                f"sum of {label}_span lengths {expected_total} != "
                f"{label}_embeds count {embeds.shape[0]}"
            )
        cursor = 0
        for span_idx, (start, end) in enumerate(spans):
            if start < 0 or end > out.shape[0] or end <= start:
                raise ValueError(
                    f"{label}_spans[{span_idx}]={(start, end)!r} out of bounds for seq={out.shape[0]}"
                )
            chunk_len = end - start
            out[start:end] = embeds[cursor : cursor + chunk_len].to(
                dtype=out.dtype, device=out.device
            )
            cursor += chunk_len

    _splice(image_embeds, list(image_spans), "image")
    _splice(video_embeds, list(video_spans), "video")
    return out


# ---------------------------------------------------------------------------
# input_ids builder + pad/merge helpers (inclusive-end offsets).
#
# These match the processor-side contract used by the published checkpoint:
# - ``processing_minimax.py`` wraps each image/video string with
#   ``]<]start of image[>[`` (id 200029) + N * placeholder + ``]<]end of image[>[``
#   (id 200030), then tokenizes — so the final ``input_ids`` already has
#   ``[..., VISION_START, IMG_TOK * N, VISION_END, ...]`` for each item.
#   Videos use the *image* start/end markers.
# - ``build_multimodal_input_ids`` operates on a pre-tokenized prompt that
#   contains a SINGLE image/video token between the start/end markers and
#   expands it into N copies in place. After the call, ``offsets`` carries
#   per-item ``(start_inclusive, end_inclusive)`` spans pointing at the
#   expanded placeholder run between the markers.
# - ``pad_multimodal_input_tokens`` replaces ``input_ids[start:end+1]``
#   (inclusive end) with each item's unique ``pad_value`` — the hash used
#   by the radix-attention prefix matcher.
#
# Both helpers agree on the final tokenised form:
#   ``[..., VISION_START_TOKEN_ID, IMG_TOKEN_ID * N, VISION_END_TOKEN_ID, ...]``
# ---------------------------------------------------------------------------


_IMAGE_MODALITY = "image"
_VIDEO_MODALITY = "video"


def build_multimodal_input_ids(
    prompt: Sequence[int],
    *,
    image_token_id: Optional[int],
    image_grid_thws: Sequence[Sequence[int]] = (),
    video_token_id: Optional[int] = None,
    video_grid_thws: Sequence[Sequence[int]] = (),
    spatial_merge_size: int = 2,
) -> Tuple[List[int], List[Tuple[int, int]], List[Tuple[int, int]], List[str]]:
    """Expand bracketed multimodal input_ids.

    Accepts a *pre-tokenized* prompt that contains the canonical
    bracketed form:
    ``[..., VISION_START_TOKEN_ID, image_token_id, VISION_END_TOKEN_ID, ...]``
    or ``[..., VISION_START_TOKEN_ID, video_token_id, VISION_END_TOKEN_ID, ...]``
    where ``VISION_START_TOKEN_ID``/``VISION_END_TOKEN_ID`` are the
    checkpoint's image start/end markers (used for BOTH images and videos
    by the M3 serving path) and ``image_token_id``/``video_token_id``
    appears as a single placeholder between the markers (the form
    ``MiniMaxVLProcessor`` produces *before* its in-place string-level
    expansion).

    For each placeholder found, the single token is expanded into ``N``
    copies of the same token id, where ``N = grid_t * (grid_h // merge) *
    (grid_w // merge)`` for that item. The VISION_START/VISION_END markers
    stay in place.

    Returns
    -------
    expanded_input_ids
        New input_ids list with each placeholder replaced by ``N`` copies.
    image_offsets
        Per-image ``(start_inclusive, end_inclusive)`` spans pointing at
        the run of expanded image_token_ids between the start/end markers.
    video_offsets
        Same for video items.
    modality_list
        Per-item modality label (``"image"`` or ``"video"``) in
        left-to-right encounter order.

    Raises ``ValueError`` on mismatched grid counts or non-divisible grid
    dims (delegating divisibility to :func:`compute_visual_token_count`).
    """
    if spatial_merge_size <= 0:
        raise ValueError(f"spatial_merge_size must be positive; got {spatial_merge_size}")

    prompt_list = list(prompt)
    image_grid_thws = list(image_grid_thws)
    video_grid_thws = list(video_grid_thws)

    # First pass: collect (position, modality) of every multimodal placeholder.
    vision_start_indices: List[Tuple[int, str]] = []
    for i in range(len(prompt_list) - 1):
        nxt = prompt_list[i + 1]
        if image_token_id is not None and nxt == image_token_id:
            vision_start_indices.append((i, _IMAGE_MODALITY))
        elif video_token_id is not None and nxt == video_token_id:
            vision_start_indices.append((i, _VIDEO_MODALITY))
    modality_list = [modality for _, modality in vision_start_indices]

    # Validate grid counts match placeholder counts.
    image_placeholders = sum(1 for _, m in vision_start_indices if m == _IMAGE_MODALITY)
    video_placeholders = sum(1 for _, m in vision_start_indices if m == _VIDEO_MODALITY)
    if image_placeholders != len(image_grid_thws):
        raise ValueError(
            f"build_multimodal_input_ids found {image_placeholders} image "
            f"placeholders but image_grid_thws has {len(image_grid_thws)} entries"
        )
    if video_placeholders != len(video_grid_thws):
        raise ValueError(
            f"build_multimodal_input_ids found {video_placeholders} video "
            f"placeholders but video_grid_thws has {len(video_grid_thws)} entries"
        )

    input_ids: List[int] = []
    image_offsets: List[Tuple[int, int]] = []
    video_offsets: List[Tuple[int, int]] = []
    cur_idx = 0
    img_idx = 0
    video_idx = 0

    for mm_start_idx, modality in vision_start_indices:
        if modality == _IMAGE_MODALITY:
            grid_t, grid_h, grid_w = image_grid_thws[img_idx]
            mm_token_num = compute_visual_token_count(grid_t, grid_h, grid_w, spatial_merge_size)
            mm_token_id = image_token_id
            img_idx += 1
        else:  # video
            grid_t, grid_h, grid_w = video_grid_thws[video_idx]
            mm_token_num = compute_visual_token_count(grid_t, grid_h, grid_w, spatial_merge_size)
            mm_token_id = video_token_id
            video_idx += 1

        assert cur_idx <= mm_start_idx, (cur_idx, mm_start_idx)
        # Append prompt up to and including the VISION_START marker.
        input_ids.extend(prompt_list[cur_idx : mm_start_idx + 1])
        mm_offset_start = len(input_ids)
        input_ids.extend([mm_token_id] * mm_token_num)
        cur_idx = mm_start_idx + 2  # skip the original placeholder at i+1

        offset = (mm_offset_start, len(input_ids) - 1)  # inclusive end
        if modality == _IMAGE_MODALITY:
            image_offsets.append(offset)
        else:
            video_offsets.append(offset)

    # Tail (including everything after the last placeholder, e.g. VISION_END
    # markers and any trailing text).
    input_ids.extend(prompt_list[cur_idx:])

    return input_ids, image_offsets, video_offsets, modality_list


def pad_multimodal_input_tokens(
    input_ids: Sequence[int],
    *,
    image_pad_values: Sequence[int] = (),
    image_offsets: Sequence[Tuple[int, int]] = (),
    video_pad_values: Sequence[int] = (),
    video_offsets: Sequence[Tuple[int, int]] = (),
) -> List[int]:
    """Inclusive-end pad rewrite.

    For each item, every position in the closed range ``[start, end]``
    of its offset is overwritten with the item's ``pad_value`` (the
    radix-attention prefix hash).
    """
    if len(image_pad_values) != len(image_offsets):
        raise ValueError(
            f"image_pad_values ({len(image_pad_values)}) must match image_offsets "
            f"({len(image_offsets)})"
        )
    if len(video_pad_values) != len(video_offsets):
        raise ValueError(
            f"video_pad_values ({len(video_pad_values)}) must match video_offsets "
            f"({len(video_offsets)})"
        )
    out = [int(t) for t in input_ids]
    for pad_value, (start, end) in zip(image_pad_values, image_offsets):
        for i in range(start, end + 1):
            out[i] = int(pad_value)
    for pad_value, (start, end) in zip(video_pad_values, video_offsets):
        for i in range(start, end + 1):
            out[i] = int(pad_value)
    return out


def merge_multimodal_embeddings_inclusive(
    input_embeds: torch.Tensor,
    *,
    image_embeds: Optional[torch.Tensor] = None,
    image_offsets: Sequence[Tuple[int, int]] = (),
    video_embeds: Optional[torch.Tensor] = None,
    video_offsets: Sequence[Tuple[int, int]] = (),
) -> torch.Tensor:
    """Replace inclusive-end multimodal spans with visual features.

    Same semantics as :func:`merge_multimodal_embeddings` but accepts
    ``(start_inclusive, end_inclusive)`` offsets so the output of
    :func:`build_multimodal_input_ids` can flow directly into the
    embedding-merge without converting offset conventions in between.
    """
    spans_image = [(s, e + 1) for s, e in image_offsets]
    spans_video = [(s, e + 1) for s, e in video_offsets]
    return merge_multimodal_embeddings(
        input_embeds,
        image_embeds=image_embeds,
        image_spans=spans_image,
        video_embeds=video_embeds,
        video_spans=spans_video,
    )


def apply_multimodal_pad_values(
    input_ids: Sequence[int],
    *,
    image_pad_values: Sequence[int] = (),
    image_spans: Sequence[Tuple[int, int]] = (),
    video_pad_values: Sequence[int] = (),
    video_spans: Sequence[Tuple[int, int]] = (),
) -> List[int]:
    """Per-item ``pad_value`` rewrite using half-open spans.

    Replaces each multimodal item's span with a unique per-item
    ``pad_value`` (a hash used by the radix-attention prefix matcher;
    the embedding is later rewritten to the real visual features).
    """
    out = [int(t) for t in input_ids]
    if len(image_pad_values) != len(image_spans):
        raise ValueError(
            f"image_pad_values ({len(image_pad_values)}) must match image_spans "
            f"({len(image_spans)})"
        )
    if len(video_pad_values) != len(video_spans):
        raise ValueError(
            f"video_pad_values ({len(video_pad_values)}) must match video_spans "
            f"({len(video_spans)})"
        )
    for pad_value, (start, end) in zip(image_pad_values, image_spans):
        for i in range(start, end):
            out[i] = int(pad_value)
    for pad_value, (start, end) in zip(video_pad_values, video_spans):
        for i in range(start, end):
            out[i] = int(pad_value)
    return out


# ---------------------------------------------------------------------------
# Multimodal forward-time helpers: build merged ``inputs_embeds`` from text
# ``input_ids`` + per-item ``pixel_values`` + ``grid_thws`` by running the
# vision tower and splicing into the text-embedding tensor at the
# (already expanded) placeholder positions.
# ---------------------------------------------------------------------------


def _flatten_grids(grid_thws: Sequence[Sequence[int]]) -> List[Tuple[int, int, int]]:
    """Normalize ``grid_thws`` to a list of integer triples."""
    out: List[Tuple[int, int, int]] = []
    if grid_thws is None:
        return out
    if isinstance(grid_thws, torch.Tensor):
        if grid_thws.dim() == 2 and grid_thws.shape[-1] == 3:
            for row in grid_thws.tolist():
                out.append((int(row[0]), int(row[1]), int(row[2])))
            return out
        if grid_thws.dim() == 1 and grid_thws.numel() == 3:
            return [(int(grid_thws[0]), int(grid_thws[1]), int(grid_thws[2]))]
        raise ValueError(
            f"grid_thws tensor must be [N, 3] or [3]; got shape {tuple(grid_thws.shape)}"
        )
    for g in grid_thws:
        if isinstance(g, torch.Tensor):
            g = g.tolist()
        out.append((int(g[0]), int(g[1]), int(g[2])))
    return out


def _find_placeholder_runs(input_ids: torch.Tensor, token_id: int) -> List[Tuple[int, int]]:
    """Return ``[(start_inclusive, end_inclusive), ...]`` runs of ``token_id`` in ``input_ids``.

    Runs are returned in left-to-right order. Each run is a maximal
    consecutive sequence of positions where ``input_ids == token_id``.
    Operates on a flat 1-D tensor.
    """
    if input_ids.dim() != 1:
        raise ValueError(
            f"_find_placeholder_runs expects a 1-D tensor; got shape {tuple(input_ids.shape)}"
        )
    runs: List[Tuple[int, int]] = []
    seq = input_ids.tolist()
    n = len(seq)
    i = 0
    while i < n:
        if int(seq[i]) == int(token_id):
            j = i
            while j < n and int(seq[j]) == int(token_id):
                j += 1
            runs.append((i, j - 1))  # inclusive end
            i = j
        else:
            i += 1
    return runs


def prepare_multimodal_inputs_embeds(
    *,
    input_ids: torch.Tensor,
    embed_tokens: nn.Module,
    vision_tower: Optional[nn.Module],
    image_token_id: int,
    video_token_id: int,
    image_pixel_values: Optional[torch.Tensor] = None,
    image_grid_thws: Optional[Sequence[Sequence[int]]] = None,
    video_pixel_values: Optional[torch.Tensor] = None,
    video_grid_thws: Optional[Sequence[Sequence[int]]] = None,
    mm_token_indices: Optional[torch.Tensor] = None,
    mm_modality_order: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    """Run the vision tower and splice its output into text embeddings.

    Used by :class:`MiniMaxM3VLForConditionalGeneration.forward` to
    materialise ``inputs_embeds`` for a multimodal request. The
    ``input_ids`` are the **already-expanded** HF processor output —
    i.e. each placeholder run has length equal to the visual token count
    derived from its ``grid_thw`` (see
    :func:`build_multimodal_input_ids`). Each placeholder run is rewritten
    with the matching slice of the vision tower's output; non-placeholder
    positions keep their ``embed_tokens(input_ids)`` value.

    Two index resolution paths are supported:

    1. **Explicit ``mm_token_indices``** (preferred for production
       runtime). The TRT-LLM LLM API runtime knows the multimodal
       positions in the in-flight-batched flat ``input_ids`` even after
       it has rewritten them with per-item radix-attention pad-value
       hashes; it conveys that via ``MultimodalParams`` /
       ``mm_token_indices``. When ``mm_token_indices`` is provided we
       use it directly, ignoring the token-id content at those
       positions. This is what makes the helper correct under
       pad-value rewriting.

    2. **Token-id run search** (used by the unit-test/standalone path
       where the caller has not pad-rewritten the input_ids and the
       canonical image/video token ids are still present at the
       placeholder positions). When ``mm_token_indices`` is ``None`` we
       locate runs by scanning for ``image_token_id`` /
       ``video_token_id`` in ``input_ids``. The total number of runs
       must equal the number of grid_thws entries per modality.

    Returns ``inputs_embeds`` of shape ``[seq, hidden]`` on the same
    device/dtype as the text embeddings.
    """
    if input_ids.dim() != 1:
        raise ValueError(
            f"prepare_multimodal_inputs_embeds expects 1-D input_ids; "
            f"got shape {tuple(input_ids.shape)}"
        )

    inputs_embeds = embed_tokens(input_ids)

    # _flatten_grids handles None, tensor, and list inputs uniformly; avoid
    # truthiness on a multi-element tensor (which raises in PyTorch).
    image_grid_list = _flatten_grids(image_grid_thws)
    video_grid_list = _flatten_grids(video_grid_thws)
    has_image = (
        image_pixel_values is not None
        and image_pixel_values.numel() > 0
        and len(image_grid_list) > 0
    )
    has_video = (
        video_pixel_values is not None
        and video_pixel_values.numel() > 0
        and len(video_grid_list) > 0
    )
    if not has_image and not has_video:
        return inputs_embeds

    if vision_tower is None:
        raise RuntimeError(
            "prepare_multimodal_inputs_embeds received multimodal data "
            "but vision_tower is None; the model was constructed without "
            "a vision branch."
        )

    # Locate placeholder runs in input_ids. Prefer explicit indices when the
    # caller passed them (pad-value-safe production path); fall back to
    # canonical-token run search (unit-test/standalone path).
    if mm_token_indices is not None:
        # Explicit indices: one contiguous run per modality item is
        # assumed (matches the HF processor contract). We build per-item
        # runs by carving the explicit index list into N slices whose
        # lengths come from grid_thws -> compute_visual_token_count, and
        # the slice is assumed to be contiguous in input_ids order.
        if mm_token_indices.dim() != 1:
            raise ValueError(
                f"mm_token_indices must be a 1-D tensor; got shape {tuple(mm_token_indices.shape)}"
            )
        idx_list = [int(i) for i in mm_token_indices.tolist()]
        # Verify all indices are within range and strictly increasing.
        if any(i < 0 or i >= int(input_ids.shape[0]) for i in idx_list):
            raise ValueError(
                "mm_token_indices contains out-of-range positions for "
                f"input_ids of length {int(input_ids.shape[0])}"
            )
        merge = getattr(vision_tower, "spatial_merge_size", 2)
        image_runs: List[Tuple[int, int]] = []
        video_runs: List[Tuple[int, int]] = []
        cursor = 0
        # When ``mm_modality_order`` is supplied, walk per-item left-to-right
        # so a prompt with [video][image] gets video positions assigned to
        # video_runs and image positions to image_runs. Without the order
        # hint, fall back to the legacy image-then-video bucketing (correct
        # for single-modality requests and the common image-before-video
        # prompt shape).
        if mm_modality_order is not None:
            img_idx, vid_idx = 0, 0
            for modality in mm_modality_order:
                if modality == "image":
                    if img_idx >= len(image_grid_list):
                        raise ValueError(
                            "mm_modality_order references more 'image' items "
                            f"than image_grid_thws ({len(image_grid_list)})"
                        )
                    g = image_grid_list[img_idx]
                    img_idx += 1
                elif modality == "video":
                    if vid_idx >= len(video_grid_list):
                        raise ValueError(
                            "mm_modality_order references more 'video' items "
                            f"than video_grid_thws ({len(video_grid_list)})"
                        )
                    g = video_grid_list[vid_idx]
                    vid_idx += 1
                else:
                    raise ValueError(
                        f"mm_modality_order entries must be 'image' or 'video'; got {modality!r}"
                    )
                n = compute_visual_token_count(g[0], g[1], g[2], merge)
                chunk = idx_list[cursor : cursor + n]
                if len(chunk) != n:
                    raise ValueError(
                        "mm_token_indices length is shorter than the sum of "
                        "per-item visual token counts derived from grid_thws"
                    )
                if modality == "image":
                    image_runs.append((chunk[0], chunk[-1]))
                else:
                    video_runs.append((chunk[0], chunk[-1]))
                cursor += n
            if img_idx != len(image_grid_list):
                raise ValueError(
                    f"mm_modality_order consumed {img_idx} image items but "
                    f"image_grid_thws has {len(image_grid_list)}"
                )
            if vid_idx != len(video_grid_list):
                raise ValueError(
                    f"mm_modality_order consumed {vid_idx} video items but "
                    f"video_grid_thws has {len(video_grid_list)}"
                )
        else:
            for g in image_grid_list:
                n = compute_visual_token_count(g[0], g[1], g[2], merge)
                chunk = idx_list[cursor : cursor + n]
                if len(chunk) != n:
                    raise ValueError(
                        "mm_token_indices length is shorter than the sum of "
                        "per-item visual token counts derived from grid_thws"
                    )
                image_runs.append((chunk[0], chunk[-1]))
                cursor += n
            for g in video_grid_list:
                n = compute_visual_token_count(g[0], g[1], g[2], merge)
                chunk = idx_list[cursor : cursor + n]
                if len(chunk) != n:
                    raise ValueError(
                        "mm_token_indices length is shorter than the sum of "
                        "per-item visual token counts derived from grid_thws"
                    )
                video_runs.append((chunk[0], chunk[-1]))
                cursor += n
        if cursor != len(idx_list):
            raise ValueError(
                f"mm_token_indices has {len(idx_list)} entries but the "
                f"grid_thws sum to {cursor} positions"
            )
    else:
        # Token-id run search; works only when input_ids still carry the
        # canonical placeholder token ids.
        image_runs = _find_placeholder_runs(input_ids, image_token_id) if has_image else []
        video_runs = _find_placeholder_runs(input_ids, video_token_id) if has_video else []

        if has_image and len(image_runs) != len(image_grid_list):
            raise ValueError(
                f"prepare_multimodal_inputs_embeds found {len(image_runs)} "
                f"image placeholder runs in input_ids but received "
                f"{len(image_grid_list)} image_grid_thws entries"
            )
        if has_video and len(video_runs) != len(video_grid_list):
            raise ValueError(
                f"prepare_multimodal_inputs_embeds found {len(video_runs)} "
                f"video placeholder runs in input_ids but received "
                f"{len(video_grid_list)} video_grid_thws entries"
            )

    # Run vision tower per-modality, batched.
    def _run_tower(
        pixel_values: torch.Tensor, grid_thws: List[Tuple[int, int, int]]
    ) -> torch.Tensor:
        feats = vision_tower(
            pixel_values=pixel_values.to(
                device=inputs_embeds.device,
                dtype=getattr(vision_tower, "dtype", inputs_embeds.dtype),
            ),
            grid_thw=grid_thws,
        )
        return feats.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)

    image_feats = _run_tower(image_pixel_values, image_grid_list) if has_image else None
    video_feats = _run_tower(video_pixel_values, video_grid_list) if has_video else None

    # Validate per-modality total feature count matches sum of run lengths.
    if image_feats is not None:
        expected = sum(end - start + 1 for start, end in image_runs)
        if image_feats.shape[0] != expected:
            raise ValueError(
                f"image vision-tower output has {image_feats.shape[0]} rows "
                f"but image placeholder runs sum to {expected}"
            )
    if video_feats is not None:
        expected = sum(end - start + 1 for start, end in video_runs)
        if video_feats.shape[0] != expected:
            raise ValueError(
                f"video vision-tower output has {video_feats.shape[0]} rows "
                f"but video placeholder runs sum to {expected}"
            )

    # Splice into inputs_embeds (non-aliased; clone first).
    out = inputs_embeds.clone()

    def _splice(feats: Optional[torch.Tensor], runs: List[Tuple[int, int]]) -> None:
        if feats is None or not runs:
            return
        cursor = 0
        for start, end in runs:
            length = end - start + 1
            out[start : end + 1] = feats[cursor : cursor + length]
            cursor += length

    _splice(image_feats, image_runs)
    _splice(video_feats, video_runs)
    return out


def extract_multimodal_data_from_params(
    multimodal_params: Sequence[Any],
) -> Dict[str, Optional[torch.Tensor]]:
    """Flatten ``MultimodalParams`` list to a single batched dict for forward.

    Accepts the ``multimodal_params`` list TensorRT-LLM passes to the
    model forward as ``kwargs["multimodal_params"]`` (each element has a
    ``multimodal_data`` dict with ``image`` / ``video`` entries). Returns
    a dict with keys ``image_pixel_values``, ``image_grid_thws``,
    ``video_pixel_values``, ``video_grid_thws`` suitable to pass into
    :func:`prepare_multimodal_inputs_embeds`. Per-request batches are
    concatenated; ``None`` is returned for missing modalities.

    Note: this function buckets all images first then all videos, which
    only matches left-to-right ``mm_token_indices`` for single-modality
    requests. For mixed image+video ordering across requests within a
    batch, use :func:`extract_multimodal_items_in_request_order` to
    preserve per-request, per-modality order.
    """
    image_pixel_list: List[torch.Tensor] = []
    image_grid_list: List[torch.Tensor] = []
    video_pixel_list: List[torch.Tensor] = []
    video_grid_list: List[torch.Tensor] = []
    for params in multimodal_params:
        if params is None:
            continue
        data = getattr(params, "multimodal_data", None)
        if not data:
            continue
        img = data.get("image") if isinstance(data, Mapping) else None
        if isinstance(img, Mapping):
            pv = img.get("pixel_values")
            gthw = img.get("image_grid_thw")
            if pv is not None and gthw is not None:
                image_pixel_list.append(pv)
                image_grid_list.append(gthw)
        vid = data.get("video") if isinstance(data, Mapping) else None
        if isinstance(vid, Mapping):
            pv = vid.get("pixel_values_videos")
            if pv is None:
                pv = vid.get("pixel_values")
            gthw = vid.get("video_grid_thw")
            if pv is not None and gthw is not None:
                video_pixel_list.append(pv)
                video_grid_list.append(gthw)

    def _cat(items: List[torch.Tensor]) -> Optional[torch.Tensor]:
        if not items:
            return None
        if len(items) == 1:
            return items[0]
        return torch.cat(items, dim=0)

    return {
        "image_pixel_values": _cat(image_pixel_list),
        "image_grid_thws": _cat(image_grid_list),
        "video_pixel_values": _cat(video_pixel_list),
        "video_grid_thws": _cat(video_grid_list),
    }


def extract_multimodal_items_in_request_order(
    multimodal_params: Sequence[Any],
) -> List[Dict[str, Any]]:
    """Walk ``multimodal_params`` and return per-item entries in request order.

    Where :func:`extract_multimodal_data_from_params` concatenates all
    image rows together and all video rows together (image-first
    bucketing), this helper keeps the natural left-to-right request
    order. Each MultimodalParams contributes at most one image item and
    at most one video item — within a single request the M3 processor
    expands all image placeholders first, then all video placeholders,
    so a request that carries both modalities still emits image then
    video in left-to-right order.

    Returned list element schema:
        {
            "modality":     "image" | "video",
            "pixel_values": Tensor [N_rows, C*T*P*P],
            "grid_thw":     Tensor [n_grids, 3],
        }
    """
    items: List[Dict[str, Any]] = []
    for params in multimodal_params:
        if params is None:
            continue
        data = getattr(params, "multimodal_data", None)
        if not data:
            continue
        img = data.get("image") if isinstance(data, Mapping) else None
        if isinstance(img, Mapping):
            pv = img.get("pixel_values")
            gthw = img.get("image_grid_thw")
            if pv is not None and gthw is not None:
                items.append({"modality": "image", "pixel_values": pv, "grid_thw": gthw})
        vid = data.get("video") if isinstance(data, Mapping) else None
        if isinstance(vid, Mapping):
            pv = vid.get("pixel_values_videos")
            if pv is None:
                pv = vid.get("pixel_values")
            gthw = vid.get("video_grid_thw")
            if pv is not None and gthw is not None:
                items.append({"modality": "video", "pixel_values": pv, "grid_thw": gthw})
    return items


@dataclass
class CLIPVisionConfig:
    """Plain dataclass mirror of the M3-VL ``vision_config`` block.

    Built once at model construction from the checkpoint's
    ``vision_config`` section. Carries only the fields downstream modules
    actually use, so a config dict with extra keys still constructs
    cleanly.
    """

    hidden_size: int = 1280
    intermediate_size: int = 5120
    num_hidden_layers: int = 32
    num_attention_heads: int = 16
    num_channels: int = 3
    image_size: int = 672
    patch_size: int = 14
    hidden_act: str = "gelu"
    layer_norm_eps: float = 1e-5
    img_token_compression_config: Dict[str, Any] = field(default_factory=dict)
    position_embedding_type: str = "rope"
    rope_mode: str = "3d"
    rope_theta: float = 10000.0
    vision_segment_max_frames: Optional[int] = 4
    projector_hidden_act: str = "gelu"
    multimodal_projector_bias: bool = True
    patch_merge_bias: bool = True

    @classmethod
    def from_dict_or_obj(cls, source: Any) -> "CLIPVisionConfig":
        """Build a :class:`CLIPVisionConfig` from a dict / object / ``None``.

        Treats missing fields as defaults rather than raising, so an
        incomplete test config still constructs.
        """
        if source is None:
            return cls()
        if hasattr(source, "to_dict"):
            data = source.to_dict()
        elif isinstance(source, Mapping):
            data = dict(source)
        else:
            data = {
                k: getattr(source, k)
                for k in dir(source)
                if not k.startswith("_") and not callable(getattr(source, k))
            }
        valid = {f for f in cls.__dataclass_fields__.keys()}
        filtered = {k: v for k, v in data.items() if k in valid}
        return cls(**filtered)


# ---------------------------------------------------------------------------
# Patch embedding (Conv3d).
# ---------------------------------------------------------------------------


class MiniMaxVLPatchEmbedding(nn.Module):
    """Conv3d patch embedding that produces ``[N, hidden_size]`` patches.

    A single 3D conv with bias-less weights, kernel/stride =
    ``(temporal_patch_size, patch_size, patch_size)``, where
    ``temporal_patch_size`` is read from ``img_token_compression_config``.

    The ``patch_embedding.weight`` slot has shape
    ``[hidden_size, num_channels, temporal_patch_size, patch_size, patch_size]``
    and matches the checkpoint key
    ``vision_tower.vision_model.embeddings.patch_embedding.weight``.
    """

    def __init__(self, config: CLIPVisionConfig, dtype: torch.dtype):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.patch_size = config.patch_size
        self.input_num_channels = config.num_channels
        self.temporal_patch_size = config.img_token_compression_config.get("temporal_patch_size", 2)

        self.patch_embedding = nn.Conv3d(
            in_channels=self.input_num_channels,
            out_channels=self.embed_dim,
            kernel_size=(self.temporal_patch_size, self.patch_size, self.patch_size),
            stride=(self.temporal_patch_size, self.patch_size, self.patch_size),
            bias=False,
            dtype=dtype,
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``[N, C*T*P*P]`` BF16 pixels -> ``[N, embed_dim]`` patch tokens.

        The processor emits a flat 2D pixel tensor of shape
        ``[N, C*T*P*P]``; the module reshapes to ``[N, C, T, P, P]`` and
        applies the Conv3d with kernel/stride equal to the patch dims so
        each row produces a single ``embed_dim`` vector.
        """
        if pixel_values.dim() != 2:
            raise ValueError(
                f"pixel_values must be 2D [N, C*T*P*P]; got shape {tuple(pixel_values.shape)}"
            )
        n = pixel_values.shape[0]
        expected_cols = (
            self.input_num_channels * self.temporal_patch_size * self.patch_size * self.patch_size
        )
        if pixel_values.shape[1] != expected_cols:
            raise ValueError(
                "pixel_values second dim must equal C*T*P*P="
                f"{expected_cols}; got {pixel_values.shape[1]}"
            )
        # Cast the conv weights to the input dtype if they disagree
        # (the published checkpoint is BF16 but tests can exercise
        # float32 paths).
        if self.patch_embedding.weight.dtype != pixel_values.dtype:
            self.patch_embedding = self.patch_embedding.to(pixel_values.dtype)
        x = pixel_values.reshape(
            n,
            self.input_num_channels,
            self.temporal_patch_size,
            self.patch_size,
            self.patch_size,
        )
        x = self.patch_embedding(x)
        x = x.reshape(n, -1)
        return x


# ---------------------------------------------------------------------------
# 3D RoPE helpers.
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the last dimension halfway: ``[x1, x2] -> [-x2, x1]``."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _prepare_rotary_cos_sin(freqs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``[seq, rope_dim/2]`` -> ``(cos, sin)`` each ``[seq, 1, rope_dim]``.

    Doubles the last dim by concatenating, then adds a head-broadcast
    dimension.
    """
    cos = freqs.cos().repeat(1, 2).unsqueeze(-2).float()
    sin = freqs.sin().repeat(1, 2).unsqueeze(-2).float()
    return cos, sin


def _apply_minimax_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply 3D RoPE to Q/K.

    Splits the last dim into rotated + passthrough halves so RoPE
    dimensions different from ``head_dim`` are handled cleanly (3D RoPE
    on M3 uses ``rope_dims < head_dim``).
    """
    rot_dim = cos.shape[-1]
    q_rot = q[..., :rot_dim].float()
    q_pass = q[..., rot_dim:]
    k_rot = k[..., :rot_dim].float()
    k_pass = k[..., rot_dim:]

    q_rot = (q_rot * cos) + (_rotate_half(q_rot) * sin)
    k_rot = (k_rot * cos) + (_rotate_half(k_rot) * sin)

    q = torch.cat((q_rot.to(q_pass.dtype), q_pass), dim=-1)
    k = torch.cat((k_rot.to(k_pass.dtype), k_pass), dim=-1)
    return q, k


# ---------------------------------------------------------------------------
# Encoder layer modules — separate Q/K/V projections to match checkpoint.
# ---------------------------------------------------------------------------


class MiniMaxVLEncoderSelfAttention(nn.Module):
    """Vision tower self-attention with separate Q/K/V projections.

    The checkpoint stores ``q_proj``, ``k_proj``, ``v_proj`` and
    ``out_proj`` separately for each of the 32 encoder layers; the
    module keeps them separate to match the published weight layout.
    Runtime QKV fusion is an optional later optimization that does not
    affect weight accounting.
    """

    def __init__(self, config: CLIPVisionConfig, dtype: torch.dtype):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True, dtype=dtype)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True, dtype=dtype)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True, dtype=dtype)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        """Per-image SDPA with 3D RoPE applied to Q/K.

        ``hidden_states``: ``[seq, embed_dim]``. ``cos/sin``: per-token
        rotary tables of shape ``[seq, 1, rope_dim]`` (after
        :func:`_prepare_rotary_cos_sin`). ``cu_seqlens`` is the
        cumulative token boundary per image so we can run a separate
        SDPA on each image without leaking attention across image
        boundaries.

        We use ``torch.nn.functional.scaled_dot_product_attention`` per
        image so we do not pull in a flash-attn dependency; for the
        single-image smoke tests this is the cheapest CUDA path that
        gives us a numerically-meaningful forward.
        """
        seq, embed_dim = hidden_states.shape
        if embed_dim != self.embed_dim:
            raise ValueError(f"hidden_states dim {embed_dim} != module embed_dim {self.embed_dim}")

        q = self.q_proj(hidden_states).reshape(seq, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).reshape(seq, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).reshape(seq, self.num_heads, self.head_dim)

        q, k = _apply_minimax_rope(q, k, cos, sin)

        # Per-image SDPA: cu_seqlens is a [num_images+1] cumulative tensor
        # of token boundaries.
        cu = cu_seqlens.detach().to("cpu").tolist()
        out = torch.empty_like(q)
        orig_dtype = hidden_states.dtype
        for i in range(len(cu) - 1):
            start, end = int(cu[i]), int(cu[i + 1])
            if start == end:
                continue
            # SDPA expects [batch, heads, seq, head_dim].
            q_i = q[start:end].permute(1, 0, 2).unsqueeze(0).to(orig_dtype)
            k_i = k[start:end].permute(1, 0, 2).unsqueeze(0).to(orig_dtype)
            v_i = v[start:end].permute(1, 0, 2).unsqueeze(0).to(orig_dtype)
            o_i = F.scaled_dot_product_attention(q_i, k_i, v_i)
            # [1, heads, seq, head_dim] -> [seq, heads, head_dim]
            out[start:end] = o_i.squeeze(0).permute(1, 0, 2)

        out = out.reshape(seq, embed_dim)
        return self.out_proj(out)


class MiniMaxVLEncoderMLP(nn.Module):
    """Two-layer MLP with GELU activation.

    Checkpoint keys:
    ``vision_tower.vision_model.encoder.layers.{i}.mlp.fc1.{weight,bias}``
    and ``...mlp.fc2.{weight,bias}``.
    """

    def __init__(self, config: CLIPVisionConfig, dtype: torch.dtype):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True, dtype=dtype)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True, dtype=dtype)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        x = self.fc1(hidden_states)
        x = F.gelu(x)
        return self.fc2(x)


class MiniMaxVLEncoderLayer(nn.Module):
    """Vision encoder layer: pre-norm self-attention + pre-norm MLP.

    Layer norm slots ``layer_norm1`` / ``layer_norm2`` keep the canonical
    underscore form used in the published checkpoint.
    """

    def __init__(self, config: CLIPVisionConfig, dtype: torch.dtype):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.self_attn = MiniMaxVLEncoderSelfAttention(config, dtype)
        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps, dtype=dtype)
        self.mlp = MiniMaxVLEncoderMLP(config, dtype)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_eps, dtype=dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, cu_seqlens)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class MiniMaxVLEncoder(nn.Module):
    """Stack of vision encoder layers."""

    def __init__(self, config: CLIPVisionConfig, dtype: torch.dtype):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [MiniMaxVLEncoderLayer(config, dtype) for _ in range(config.num_hidden_layers)]
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, cu_seqlens)
        return hidden_states


# ---------------------------------------------------------------------------
# Vision transformer.
# ---------------------------------------------------------------------------


class MiniMaxVLVisionTransformer(nn.Module):
    """ViT with Conv3d patch embedding, pre-norm typo intact, and 3D RoPE.

    The ``pre_layrnorm`` typo (missing "e") matches the published
    checkpoint key ``vision_tower.vision_model.pre_layrnorm.{weight,bias}``;
    do not "fix" it.

    3D RoPE inv-freq buffers (``inv_freq_t/h/w``) are computed at init
    time from ``rope_theta`` and the per-axis rope-dim split. Stored as
    non-persistent buffers so they are not part of the loaded state dict.
    """

    def __init__(self, config: CLIPVisionConfig, dtype: torch.dtype):
        super().__init__()
        self.config = config
        embed_dim = config.hidden_size

        self.temporal_patch_size = config.img_token_compression_config.get("temporal_patch_size", 2)
        self.spatial_merge_size = config.img_token_compression_config.get("spatial_merge_size", 2)

        self.embeddings = MiniMaxVLPatchEmbedding(config, dtype)
        # NOTE: the typo "layrnorm" matches the published checkpoint key.
        self.pre_layrnorm = nn.LayerNorm(embed_dim, eps=config.layer_norm_eps, dtype=dtype)
        self.encoder = MiniMaxVLEncoder(config, dtype)

        if config.position_embedding_type != "rope" or config.rope_mode != "3d":
            raise ValueError(
                "MiniMax VL vision tower expects rope position embeddings with "
                f"3D mode; got position_embedding_type={config.position_embedding_type!r}, "
                f"rope_mode={config.rope_mode!r}"
            )

        head_dim = embed_dim // config.num_attention_heads
        rope_dims = 2 * (head_dim // 2)
        # 3D RoPE: split rope dims evenly across t / h / w axes (each
        # rounded down to an even number).
        self.t_dim = int(2 * ((rope_dims // 3) // 2))
        self.h_dim = int(2 * ((rope_dims // 3) // 2))
        self.w_dim = int(2 * ((rope_dims // 3) // 2))

        rope_theta = float(config.rope_theta)
        inv_freq_t = 1.0 / (
            rope_theta ** (torch.arange(0, self.t_dim, 2, dtype=torch.float32) / self.t_dim)
        )
        inv_freq_h = 1.0 / (
            rope_theta ** (torch.arange(0, self.h_dim, 2, dtype=torch.float32) / self.h_dim)
        )
        inv_freq_w = 1.0 / (
            rope_theta ** (torch.arange(0, self.w_dim, 2, dtype=torch.float32) / self.w_dim)
        )
        self.register_buffer("inv_freq_t", inv_freq_t, persistent=False)
        self.register_buffer("inv_freq_h", inv_freq_h, persistent=False)
        self.register_buffer("inv_freq_w", inv_freq_w, persistent=False)

    def _compute_cu_seqlens(
        self, grid_thw: Sequence[Sequence[int]], device: torch.device
    ) -> torch.Tensor:
        """Cumulative per-image token boundary tensor for SDPA segmentation.

        For each image we add ``grid_t * grid_h * grid_w`` tokens to the
        running total. Returns a 1D int32 tensor of length
        ``num_images + 1`` on the requested device.
        """
        seqlens = [0]
        for grid_t, grid_h, grid_w in grid_thw:
            seqlens.append(grid_t * grid_h * grid_w)
        return torch.cumsum(torch.tensor(seqlens, dtype=torch.int32, device=device), dim=0).to(
            torch.int32
        )

    def _get_3d_rope_freqs(
        self,
        grid_t: int,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        """Per-token 3D RoPE frequency tensor.

        Returns ``[grid_t * grid_h * grid_w, rope_dim/2]`` where
        ``rope_dim = t_dim + h_dim + w_dim``. The layout is the per-token
        ``[freq_t, freq_h, freq_w]`` concatenation, with the per-axis
        position ids built via the standard ``arange + spatial_merge``
        reshape pattern.
        """
        device_t = self.inv_freq_t.device
        device_h = self.inv_freq_h.device
        device_w = self.inv_freq_w.device

        tokens_per_frame = grid_h * grid_w
        # tpos_ids: [grid_t * tokens_per_frame] with each frame's tokens
        # carrying the same temporal index.
        tpos_ids = (
            torch.arange(grid_t, device=device_t)
            .unsqueeze(1)
            .expand(-1, tokens_per_frame)
            .flatten()
        )

        spatial_merge_size = self.spatial_merge_size

        hpos_ids = torch.arange(grid_h, device=device_h).unsqueeze(1).expand(-1, grid_w)
        hpos_ids = hpos_ids.reshape(
            grid_h // spatial_merge_size,
            spatial_merge_size,
            grid_w // spatial_merge_size,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.permute(0, 2, 1, 3)
        hpos_ids = hpos_ids.unsqueeze(0).expand(grid_t, -1, -1, -1, -1).flatten()

        wpos_ids = torch.arange(grid_w, device=device_w).unsqueeze(0).expand(grid_h, -1)
        wpos_ids = wpos_ids.reshape(
            grid_h // spatial_merge_size,
            spatial_merge_size,
            grid_w // spatial_merge_size,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.permute(0, 2, 1, 3)
        wpos_ids = wpos_ids.unsqueeze(0).expand(grid_t, -1, -1, -1, -1).flatten()

        max_t = max(grid_t, 1)
        max_hw = max(grid_h, grid_w)

        seq_t = torch.arange(max_t, device=device_t, dtype=self.inv_freq_t.dtype)
        seq_hw = torch.arange(max_hw, device=device_h, dtype=self.inv_freq_h.dtype)

        freqs_t = torch.outer(seq_t, self.inv_freq_t)
        freqs_h = torch.outer(seq_hw, self.inv_freq_h)
        freqs_w = torch.outer(seq_hw, self.inv_freq_w)

        emb_t = freqs_t[tpos_ids]
        emb_h = freqs_h[hpos_ids]
        emb_w = freqs_w[wpos_ids]

        return torch.cat([emb_t, emb_h, emb_w], dim=-1)

    def _get_rope_freqs(self, grid_thw: Sequence[Sequence[int]]) -> torch.Tensor:
        """Concatenated per-image RoPE frequency tensor."""
        chunks = [self._get_3d_rope_freqs(int(t), int(h), int(w)) for t, h, w in grid_thw]
        return torch.cat(chunks, dim=0)

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """Run the full ViT forward: patch embed -> pre-LN -> encoder.

        Output shape: ``[total_tokens, embed_dim]`` where
        ``total_tokens = sum(grid_t * grid_h * grid_w)`` across all
        images. The projector + patch-merger compression is applied by
        the parent :class:`MiniMaxVLVisionModel`.
        """
        if pixel_values.dim() != 2:
            raise ValueError(
                f"pixel_values must be 2D [N, C*T*P*P]; got shape {tuple(pixel_values.shape)}"
            )
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.pre_layrnorm(hidden_states)

        device = hidden_states.device
        cu_seqlens = self._compute_cu_seqlens(grid_thw, device=device)
        freqs = self._get_rope_freqs(grid_thw).to(device=device)
        cos, sin = _prepare_rotary_cos_sin(freqs)

        hidden_states = self.encoder(hidden_states, cos, sin, cu_seqlens)
        return hidden_states


# ---------------------------------------------------------------------------
# Multimodal projector + patch merger.
# ---------------------------------------------------------------------------


class MiniMaxVLMultiModalProjector(nn.Module):
    """Two-layer MLP that lifts vision features to text hidden size.

    ``linear_1 -> gelu -> linear_2`` with optional bias. Checkpoint keys:
    ``multi_modal_projector.linear_{1,2}.{weight,bias}`` (re-anchored
    under ``vision_tower.multi_modal_projector``; see
    :func:`load_minimax_m3_vl_state_dict`).
    """

    def __init__(
        self,
        *,
        vision_hidden_size: int,
        text_hidden_size: int,
        projector_hidden_act: str = "gelu",
        multimodal_projector_bias: bool = True,
        projector_hidden_size: Optional[int] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if projector_hidden_act != "gelu":
            raise ValueError(
                "MiniMax VL multimodal projector only supports gelu activation; "
                f"got {projector_hidden_act!r}"
            )
        mid_size = projector_hidden_size if projector_hidden_size is not None else text_hidden_size
        self.linear_1 = nn.Linear(
            vision_hidden_size, mid_size, bias=multimodal_projector_bias, dtype=dtype
        )
        self.linear_2 = nn.Linear(
            mid_size, text_hidden_size, bias=multimodal_projector_bias, dtype=dtype
        )

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        x = self.linear_1(image_features)
        x = F.gelu(x)
        return self.linear_2(x)


class MiniMaxVLPatchMerger(nn.Module):
    """Reshape + two-layer MLP that compresses by ``spatial_merge_size**2``.

    Checkpoint keys: ``patch_merge_mlp.linear_{1,2}.{weight,bias}``
    (re-anchored under ``vision_tower.patch_merge_mlp``).
    """

    def __init__(
        self,
        *,
        spatial_merge_size: int,
        text_hidden_size: int,
        projector_hidden_act: str = "gelu",
        patch_merge_bias: bool = True,
        projector_hidden_size: Optional[int] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if projector_hidden_act != "gelu":
            raise ValueError(
                "MiniMax VL patch merger only supports gelu activation; "
                f"got {projector_hidden_act!r}"
            )
        self.spatial_merge_size = spatial_merge_size
        mid_size = projector_hidden_size if projector_hidden_size is not None else text_hidden_size
        self.linear_1 = nn.Linear(
            text_hidden_size * spatial_merge_size * spatial_merge_size,
            mid_size,
            bias=patch_merge_bias,
            dtype=dtype,
        )
        self.linear_2 = nn.Linear(mid_size, text_hidden_size, bias=patch_merge_bias, dtype=dtype)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Reshape by ``spatial_merge_size**2`` then 2-layer MLP.

        Input ``image_features`` is ``[N, text_hidden]`` where N is
        divisible by ``spatial_merge_size**2``. The reshape compresses
        every ``spatial_merge_size**2`` adjacent tokens into one wider
        token before the MLP.
        """
        merge_factor = self.spatial_merge_size * self.spatial_merge_size
        if image_features.dim() != 2:
            raise ValueError(
                f"image_features must be 2D [N, hidden]; got {tuple(image_features.shape)}"
            )
        n, hidden = image_features.shape
        if n % merge_factor != 0:
            raise ValueError(
                f"patch merger expects N divisible by spatial_merge_size**2="
                f"{merge_factor}; got N={n}"
            )
        x = image_features.reshape(n // merge_factor, merge_factor * hidden)
        x = self.linear_1(x)
        x = F.gelu(x)
        return self.linear_2(x)


# ---------------------------------------------------------------------------
# Top-level vision module: ViT + projector + patch merger.
# ---------------------------------------------------------------------------


class MiniMaxVLVisionModel(nn.Module):
    """ViT + multimodal projector + patch merger.

    Owns the canonical ``vision_model`` (ViT), ``multi_modal_projector``,
    and ``patch_merge_mlp`` submodules so that ``state_dict()`` keys
    directly correspond to the checkpoint's ``vision_tower.*`` key
    namespace after re-anchoring the standalone projector and
    patch-merger blobs (see :func:`reanchor_multimodal_checkpoint_keys`).
    """

    def __init__(
        self,
        config: CLIPVisionConfig,
        text_hidden_size: int,
        projector_hidden_size: Optional[int] = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.config = config
        self.vision_config = config

        self.vision_model = MiniMaxVLVisionTransformer(config, dtype)
        self.multi_modal_projector = MiniMaxVLMultiModalProjector(
            vision_hidden_size=config.hidden_size,
            text_hidden_size=text_hidden_size,
            projector_hidden_act=config.projector_hidden_act,
            multimodal_projector_bias=config.multimodal_projector_bias,
            projector_hidden_size=projector_hidden_size,
            dtype=dtype,
        )
        spatial_merge_size = config.img_token_compression_config.get("spatial_merge_size", 2)
        self.spatial_merge_size = spatial_merge_size
        self.patch_merge_mlp = MiniMaxVLPatchMerger(
            spatial_merge_size=spatial_merge_size,
            text_hidden_size=text_hidden_size,
            projector_hidden_act=config.projector_hidden_act,
            patch_merge_bias=config.multimodal_projector_bias,
            projector_hidden_size=projector_hidden_size,
            dtype=dtype,
        )
        self.dtype = dtype
        self.text_hidden_size = text_hidden_size

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """End-to-end vision feature extraction.

        Output ``[total_merged_tokens, text_hidden_size]`` where
        ``total_merged_tokens = sum_i grid_t_i * (grid_h_i // merge) *
        (grid_w_i // merge)`` matches the placeholder span sizes.
        """
        hidden_states = self.vision_model(pixel_values=pixel_values, grid_thw=grid_thw)
        if hidden_states.dim() == 3:
            hidden_states = hidden_states.squeeze(0)
        hidden_states = self.multi_modal_projector(hidden_states)
        hidden_states = self.patch_merge_mlp(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Checkpoint key planning / re-anchoring + weight loading.
# ---------------------------------------------------------------------------


def reanchor_multimodal_checkpoint_keys(
    source_keys: Iterable[str],
) -> Dict[str, str]:
    """Plan the rename from raw checkpoint keys to module-relative keys.

    ``multi_modal_projector.*`` and ``patch_merge_mlp.*`` blobs are
    stored at the checkpoint top level but logically belong under
    ``vision_tower.``. The returned mapping has the form
    ``{source_key: target_key}`` and only contains keys that need
    rewriting; ``vision_tower.*`` keys map to themselves.
    """
    mapping: Dict[str, str] = {}
    for key in source_keys:
        if key.startswith("vision_tower."):
            mapping[key] = key
            continue
        if key.startswith("multi_modal_projector.") or key.startswith("patch_merge_mlp."):
            mapping[key] = "vision_tower." + key
    return mapping


def split_multimodal_weights(
    weights: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Split a checkpoint dict into ``(text_weights, vision_weights)``.

    Text weights keep their original keys (including the ``language_model.``
    prefix) so the text loader keeps working unchanged.

    Vision weights are returned with their keys re-anchored under
    ``vision_tower.`` so the keys correspond directly to the
    :class:`MiniMaxVLVisionModel`'s parameter namespace when that module
    is exposed as ``vision_tower`` on the parent model.
    """
    rename_plan = reanchor_multimodal_checkpoint_keys(weights.keys())
    text_weights: Dict[str, Any] = {}
    vision_weights: Dict[str, Any] = {}
    for key, value in weights.items():
        if key in rename_plan:
            vision_weights[rename_plan[key]] = value
        else:
            text_weights[key] = value
    return text_weights, vision_weights


def load_minimax_m3_vl_state_dict(
    module: nn.Module,
    weights: Mapping[str, Any],
    *,
    strict: bool = True,
) -> Tuple[List[str], List[str]]:
    """Load checkpoint-style weights into a :class:`MiniMaxVLVisionModel`.

    ``weights`` may contain raw checkpoint keys; they are re-anchored
    automatically. The module is expected to be exposed as
    ``vision_tower`` on its parent — keys are stripped of the
    ``vision_tower.`` prefix before loading into ``module.state_dict()``.

    Returns ``(loaded_keys, missing_keys)``: ``loaded_keys`` lists the
    state-dict keys that received a value from ``weights``;
    ``missing_keys`` lists state-dict slots not present in ``weights``.

    When ``strict=True`` (the default) a missing slot is an error; with
    ``strict=False`` the caller is responsible for handling partial loads.
    """
    rename_plan = reanchor_multimodal_checkpoint_keys(weights.keys())
    state_dict = module.state_dict()
    target_keys = set(state_dict.keys())

    loaded: List[str] = []
    unconsumed: List[str] = []
    for source_key, value in weights.items():
        target_key = rename_plan.get(source_key, source_key)
        if not target_key.startswith("vision_tower."):
            unconsumed.append(source_key)
            continue
        relative_key = target_key[len("vision_tower.") :]
        if relative_key not in target_keys:
            unconsumed.append(source_key)
            continue
        existing = state_dict[relative_key]
        if existing.shape != value.shape:
            raise ValueError(
                f"shape mismatch for {target_key}: module expects "
                f"{tuple(existing.shape)}, checkpoint has {tuple(value.shape)}"
            )
        state_dict[relative_key] = value.to(dtype=existing.dtype)
        loaded.append(relative_key)

    missing = sorted(target_keys - set(loaded))
    if strict and missing:
        raise RuntimeError(
            f"MiniMaxVLVisionModel.load_state_dict missing {len(missing)} keys; "
            f"first 5: {missing[:5]!r}"
        )
    if strict and unconsumed:
        raise RuntimeError(
            f"MiniMaxVLVisionModel.load_state_dict received {len(unconsumed)} "
            f"unconsumed keys; first 5: {unconsumed[:5]!r}"
        )
    module.load_state_dict(state_dict, strict=False)
    return loaded, missing


# ---------------------------------------------------------------------------
# Input processor — bridges the LLM API's
# ``llm.generate(prompt + multi_modal_data)`` call into TRT-LLM's
# ``MultimodalParams`` format that the model ``forward()`` consumes.
# ---------------------------------------------------------------------------


class MiniMaxM3VLInputProcessor:
    """LLM-API input processor for MiniMax-M3 VL multimodal requests.

    Designed to participate in TRT-LLM's multimodal input-processing
    pipeline via :class:`tensorrt_llm.inputs.registry.BaseMultimodalInputProcessor`
    (mixed in by :func:`get_minimax_m3_vl_input_processor_cls`). When
    ``llm.generate(prompt + multi_modal_data)`` is called, the processor
    runs the M3 VL HF processor on the prompt + image/video data and
    returns the expanded ``input_ids`` along with the per-modality
    ``pixel_values`` and ``grid_thw`` tensors packaged in TRT-LLM's
    ``multimodal_data`` schema.

    The processor honours the checkpoint contract that BOTH image and
    video placeholders are framed by the image start/end tokens
    (``MINIMAX_M3_VL_VISION_START_TOKEN`` /
    ``MINIMAX_M3_VL_VISION_END_TOKEN`` above, resolved via the tokenizer).
    """

    def __init__(
        self,
        model_path: str,
        config: Any,
        tokenizer: Any = None,
        trust_remote_code: bool = True,
        **kwargs: Any,
    ):
        from tensorrt_llm.inputs.registry import BaseMultimodalInputProcessor

        BaseMultimodalInputProcessor.__init__(
            self,
            model_path=model_path,
            config=config,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )

        from transformers import AutoProcessor, AutoTokenizer

        self._tokenizer = (
            tokenizer
            if tokenizer is not None
            else AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
        )
        self._model_path = model_path
        self._config = config
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=self._use_fast,
            trust_remote_code=trust_remote_code,
        )
        text_cfg = getattr(config, "text_config", None)
        if isinstance(text_cfg, dict):
            self._dtype = getattr(text_cfg, "torch_dtype", torch.bfloat16)
        else:
            self._dtype = getattr(
                text_cfg, "torch_dtype", getattr(config, "torch_dtype", torch.bfloat16)
            )
        if not isinstance(self._dtype, torch.dtype):
            self._dtype = getattr(torch, str(self._dtype).split(".")[-1], torch.bfloat16)

        # Resolve special token ids strictly:
        #   * image / video placeholder ids come from the model config
        #     (``image_token_index`` / ``video_token_index``); missing
        #     fields are a checkpoint contract violation.
        #   * vision-start / vision-end marker ids come from the tokenizer
        #     via ``convert_tokens_to_ids`` on the canonical token strings;
        #     an unresolved lookup is a tokenizer contract violation.
        if not hasattr(config, "image_token_index"):
            raise ValueError("MiniMax-M3 VL config is missing required field 'image_token_index'")
        if not hasattr(config, "video_token_index"):
            raise ValueError("MiniMax-M3 VL config is missing required field 'video_token_index'")
        self._image_token_id = int(config.image_token_index)
        self._video_token_id = int(config.video_token_index)
        self._vision_start_token_id = self._resolve_token_id(MINIMAX_M3_VL_VISION_START_TOKEN)
        self._vision_end_token_id = self._resolve_token_id(MINIMAX_M3_VL_VISION_END_TOKEN)

    def _resolve_token_id(self, token: str) -> int:
        """Look ``token`` up via the tokenizer. Raise if the tokenizer has
        no ``convert_tokens_to_ids`` or maps the token to its unk id."""
        convert = getattr(self._tokenizer, "convert_tokens_to_ids", None)
        if convert is None:
            raise ValueError(
                f"MiniMax-M3 VL tokenizer has no convert_tokens_to_ids; "
                f"cannot resolve special token {token!r}"
            )
        resolved = convert(token)
        unk = getattr(self._tokenizer, "unk_token_id", None)
        if resolved is None or (unk is not None and resolved == unk):
            raise ValueError(
                f"MiniMax-M3 VL tokenizer cannot resolve special token {token!r}; "
                f"got {resolved!r} (unk_token_id={unk!r})"
            )
        return int(resolved)

    # ----- registry contract properties -----------------------------------
    @property
    def processor(self):
        return self._processor

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def config(self):
        return self._config

    @property
    def model_path(self):
        return self._model_path

    @property
    def dtype(self):
        return self._dtype

    @property
    def use_fast(self) -> bool:
        return self._use_fast

    # ----- helper: extract image/video items from the request -------------
    @staticmethod
    def _coerce_to_list(value: Any) -> List[Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    # ----- tokenized + MM fast-path hooks ---------------------------------
    # Implementing these lets the runtime accept
    # ``(prompt_token_ids, multi_modal_data)`` directly. Without them the
    # runtime detokenizes prompt_token_ids back to text and re-runs the M3
    # chat template, double-inserting ``]<]image[>[`` markers and crashing
    # the HF processor at ``image_grid_thw[index].prod()``.

    def get_vocab_size(self) -> Optional[int]:
        # Top-level M3 VL config carries a legacy ``vocab_size=32000``;
        # the real LM vocab lives on ``text_config`` (200064 incl. image
        # / video / vision-start / vision-end tokens).
        text_cfg = getattr(self._config, "text_config", None)
        vocab = getattr(text_cfg, "vocab_size", None) or getattr(self._config, "vocab_size", None)
        return int(vocab) if vocab is not None else None

    def get_mm_token_ids(self) -> torch.Tensor:
        return torch.tensor([self._image_token_id, self._video_token_id], dtype=torch.int32)

    def get_mm_special_token_ids(self) -> torch.Tensor:
        # VISION_START / VISION_END frame each MM span; declared as
        # specials so they're counted in num_mm_tokens_per_placeholder
        # but excluded from the embed-mask.
        return torch.tensor(
            [self._vision_start_token_id, self._vision_end_token_id],
            dtype=torch.int32,
        )

    @staticmethod
    def _hw(media: Any) -> Tuple[int, int]:
        if isinstance(media, torch.Tensor):
            return int(media.shape[-2]), int(media.shape[-1])
        return int(media.height), int(media.width)

    def get_num_tokens_per_image(self, *, image: Any, **kwargs: Any) -> int:
        """Token count per image: ``(grid_h*grid_w)/merge^2 + 2`` (+VS/VE framing)."""
        ip = self._processor.image_processor
        h, w = self._hw(image)
        merge = int(ip.merge_size)
        return ip.get_number_of_image_patches(h, w) // (merge * merge) + 2

    def get_num_tokens_per_video(
        self,
        *,
        video: Any,
        video_grid_thw: Optional[Any] = None,
        **kwargs: Any,
    ) -> int:
        """Token count per video block (+ VS/VE framing).

        ``video_grid_thw`` is populated by the fast-path's dummy-text pass
        in :func:`find_mm_token_lengths`, so we don't duplicate the video
        resize / temporal-patch math here.
        """
        if video_grid_thw is None:
            raise RuntimeError(
                "MiniMaxM3VL fast path: get_num_tokens_per_video requires "
                "precomputed video_grid_thw from the dummy-text pass."
            )
        grid = (
            video_grid_thw.tolist() if isinstance(video_grid_thw, torch.Tensor) else video_grid_thw
        )
        merge = int(self._processor.image_processor.merge_size)
        grid_t, grid_h, grid_w = int(grid[0]), int(grid[1]), int(grid[2])
        return grid_t * (grid_h // merge) * (grid_w // merge) + 2

    def get_text_with_mm_placeholders(self, mm_counts: Dict[str, int]) -> str:
        # Return ``""``: ``__call__``'s chat-template branch already
        # inserts one ``]<]image[>[`` / ``]<]video[>[`` marker per
        # ``{"type": "image"}`` / ``{"type": "video"}`` content entry, so
        # adding our own markers here would double them.
        return ""

    def expand_prompt_token_ids_for_mm(
        self,
        prompt_token_ids: List[int],
        num_mm_tokens_per_placeholder: List[int],
        hf_processor_mm_kwargs: Optional[Dict[str, Any]] = None,
        mm_data: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[int], Optional[Dict[str, Dict[str, Any]]]]:
        """Rewrite chat-template placeholders into ``[VS, IMG*N, VE]`` blocks.

        Dynamo's prompt_token_ids carries one ``image_token_id`` /
        ``video_token_id`` per item with no VISION_START / VISION_END
        framing; the HF MiniMaxVLProcessor would normally add it on the
        slow path. We do the same here from ``num_mm_tokens_per_placeholder``
        (which already accounts for the +2 framing).
        """
        mm_data = mm_data or {}
        has_image = bool(mm_data.get("image"))
        has_video = bool(mm_data.get("video"))
        if has_image and has_video:
            raise ValueError(
                "MiniMaxM3VL fast path: mixed image + video in a single request is not supported."
            )
        if not (has_image or has_video):
            return list(prompt_token_ids), None
        placeholder_id = self._image_token_id if has_image else self._video_token_id
        expected = len(num_mm_tokens_per_placeholder)
        expanded: List[int] = []
        consumed = 0
        for tok in prompt_token_ids:
            if tok != placeholder_id:
                expanded.append(int(tok))
                continue
            if consumed >= expected:
                raise ValueError(
                    "MiniMaxM3VL fast path: prompt has more placeholders than "
                    f"num_mm_tokens entries ({expected})."
                )
            inner = int(num_mm_tokens_per_placeholder[consumed]) - 2
            if inner < 1:
                raise ValueError(
                    f"MiniMaxM3VL fast path: num_mm_tokens[{consumed}]={inner + 2} "
                    "must be >= 3 (VISION_START + >=1 placeholder + VISION_END)."
                )
            expanded.append(self._vision_start_token_id)
            expanded.extend([placeholder_id] * inner)
            expanded.append(self._vision_end_token_id)
            consumed += 1
        if consumed != expected:
            raise ValueError(
                f"MiniMaxM3VL fast path: prompt has {consumed} placeholders "
                f"but num_mm_tokens has {expected} entries."
            )
        return expanded, None

    # ----- main entry point -----------------------------------------------
    # Implements the ``call_with_text_prompt`` hook required by
    # :class:`BaseMultimodalInputProcessor`. The base class's ``__call__``
    # dispatches text-prompt (with or without MM data) requests here and
    # detokenizes token-id+MM requests to fall through to this method.
    def call_with_text_prompt(
        self,
        inputs: Dict[str, Any],
        sampling_params: Any = None,
    ) -> Tuple[List[int], Optional[Dict[str, Any]]]:
        """Render a multimodal LLM-API request to (input_ids, multimodal_data).

        1. Pull ``prompt`` (text) and ``multi_modal_data`` (dict with
           optional ``image`` / ``video`` lists) from the ``inputs``
           mapping.
        2. Drive the checkpoint's HF processor (loaded via AutoProcessor;
           backed by ``processing_minimax.MiniMaxVLProcessor`` on the
           real checkpoint) to expand placeholders and produce
           ``pixel_values`` / ``image_grid_thw`` (and the video
           equivalents).
        3. Return ``(input_ids_list, {"multimodal_data": {...}})`` in the
           shape TRT-LLM's runtime expects.
        """
        text_prompt = inputs.get("prompt") if isinstance(inputs, Mapping) else None
        mm_data = (inputs.get("multi_modal_data") if isinstance(inputs, Mapping) else None) or {}

        images = self._coerce_to_list(mm_data.get("image"))
        video_inputs = self._coerce_to_list(mm_data.get("video"))

        # The HF MiniMaxVLProcessor expects video frames as nested lists
        # of frames (one list per video). Accept either pre-extracted
        # frames or wrapper objects with a ``frames`` attribute.
        videos: List[Any] = []
        for v in video_inputs:
            if hasattr(v, "frames"):
                videos.append(list(v.frames))
            else:
                videos.append(list(v) if isinstance(v, (list, tuple)) else [v])

        # Apply the M3 chat template before invoking the processor. The
        # M3 ``chat_template.jinja`` inserts the canonical
        # ``IMAGE_TOKEN`` / ``VIDEO_TOKEN`` markers around each image
        # / video item in the user's content list. The
        # ``MiniMaxVLProcessor`` then expands those markers in place
        # using ``VISION_START_TOKEN + IMAGE_TOKEN*N + VISION_END_TOKEN``
        # before tokenization. Without the chat template, the raw user
        # prompt doesn't carry any image marker and the processor
        # emits no multimodal token positions in the final input_ids.
        templated_text: str
        if images or videos:
            content: List[Dict[str, Any]] = []
            if isinstance(text_prompt, str) and text_prompt:
                content.append({"type": "text", "text": text_prompt})
            for _ in range(len(images)):
                content.append({"type": "image"})
            for _ in range(len(videos)):
                content.append({"type": "video"})
            messages = [{"role": "user", "content": content}]
            try:
                templated_text = self._tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                # Fallback: explicitly inject IMAGE_TOKEN per image so
                # the processor still expands them even when the chat
                # template is unavailable.
                explicit: List[str] = []
                if isinstance(text_prompt, str) and text_prompt:
                    explicit.append(text_prompt)
                explicit.extend(["]<]image[>["] * len(images))
                explicit.extend(["]<]video[>["] * len(videos))
                templated_text = "\n".join(explicit)
        else:
            templated_text = text_prompt or ""

        # Run the HF processor. ``return_tensors='pt'`` yields tensors
        # in the BatchFeature output.
        processed = self._processor(
            text=[templated_text],
            images=images if images else None,
            videos=videos if videos else None,
            return_tensors="pt",
        )

        multimodal_data: Dict[str, Any] = {}
        pixel_values = processed.get("pixel_values", None)
        if pixel_values is not None:
            multimodal_data["image"] = {
                "pixel_values": pixel_values.to(self.dtype),
                "image_grid_thw": processed.get("image_grid_thw"),
            }

        pixel_values_videos = processed.get("pixel_values_videos", None)
        if pixel_values_videos is not None:
            multimodal_data["video"] = {
                "pixel_values_videos": pixel_values_videos.to(self.dtype),
                "video_grid_thw": processed.get("video_grid_thw"),
            }

        fused_input_ids = processed["input_ids"][0]
        return fused_input_ids.to(torch.int32).tolist(), {
            "multimodal_data": multimodal_data,
        }


def get_minimax_m3_vl_input_processor_cls():
    """Return ``MiniMaxM3VLInputProcessor`` re-typed as a proper subclass.

    The model module decorates the model class with
    ``@register_input_processor(MiniMaxM3VLInputProcessor, ...)``.
    ``register_input_processor`` only requires that the processor
    declares the same set of attributes as ``InputProcessor`` /
    ``BaseMultimodalInputProcessor``; runtime ``isinstance`` checks
    against ``BaseMultimodalInputProcessor`` for the multimodal-disagg
    path. Re-type by dynamic subclassing here so the registered class
    inherits from the base.
    """
    from tensorrt_llm.inputs.registry import (
        BaseMultimodalDummyInputsBuilder,
        BaseMultimodalInputProcessor,
    )

    class _Registered(
        MiniMaxM3VLInputProcessor, BaseMultimodalInputProcessor, BaseMultimodalDummyInputsBuilder
    ):
        pass

    _Registered.__name__ = "MiniMaxM3VLInputProcessor"
    _Registered.__qualname__ = "MiniMaxM3VLInputProcessor"
    return _Registered
