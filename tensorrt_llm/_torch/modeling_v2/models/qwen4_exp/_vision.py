# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Qwen4-Exp vision input layout and admission for both checkpoint precisions."""

import math
from functools import partial

import torch

from tensorrt_llm._torch.modeling_v2._router_index import NoTarget
from tensorrt_llm._torch.models.modeling_multimodal_encoder import MultimodalEncoderMixin
from tensorrt_llm._torch.models.modeling_multimodal_utils import (
    VisionNormalization,
    _qwen3vl_build_batched_input,
)
from tensorrt_llm.inputs.multimodal import get_multimodal_item_metadata

_MEDIA_KEYS = {
    "image": ("pixel_values", "image_grid_thw"),
    "video": ("pixel_values_videos", "video_grid_thw"),
}
_POSITION_AXES = 3

# The vision kernels serve whole frames of up to 8192 merged tokens.
VISION_CALL_MAX_PATCHES = 32768


def _frame_calls(
    grid: list[tuple[int, int, int]], max_rows: int
) -> list[list[tuple[int, int, int]]]:
    """Pack whole items, splitting long videos at the same frame boundaries as when encoded alone."""
    calls = [[]]
    room = max_rows
    for frames, height, width in grid:
        area = height * width
        assert area <= max_rows, (
            f"one frame is {height}x{width} = {area} patch rows; the vision ops are "
            f"certified up to {max_rows} rows in one call"
        )
        frames_per_call = max_rows // area
        while frames:
            take = min(frames, frames_per_call)
            cost = take * area
            if cost > room:
                calls.append([])
                room = max_rows
            calls[-1].append((take, height, width))
            room -= cost
            frames -= take
    return calls


def _project_items(op, inputs, lengths, *weights):
    """Keep each cuBLASLt projection at its independent item's matrix dimensions."""
    if len(lengths) == 1:
        return op(inputs, *weights)
    return torch.cat([op(item, *weights) for item in inputs.split(lengths)])


def _item_attention_args(grid, device):
    """Prepare encoder-only lengths once; each item keeps its own attention launch shape."""
    frame_lengths = [h * w for t, h, w in grid for _ in range(t)]
    host_lengths = torch.tensor(frame_lengths, dtype=torch.int32, device="cpu")
    device_lengths = host_lengths.to(device)
    host_types = torch.zeros_like(host_lengths)
    totals = torch.tensor([[t * h * w, 0] for t, h, w in grid], dtype=torch.int32, device="cpu")
    items = []
    first_frame = first_patch = 0
    for index, (frames, height, width) in enumerate(grid):
        patches_per_frame = height * width
        patches = frames * patches_per_frame
        span = slice(first_frame, first_frame + frames)
        seq_lens = device_lengths[span]
        host_lens = host_lengths[span]
        items.append(
            (
                slice(first_patch, first_patch + patches),
                dict(
                    sequence_length=seq_lens,
                    context_lengths=seq_lens,
                    host_past_key_value_lengths=host_lens,
                    host_context_lengths=host_lens,
                    host_request_types=host_types[span],
                    host_total_kv_lens=totals[index],
                    max_num_requests=frames,
                    max_context_length=patches_per_frame,
                    max_seq_len=patches_per_frame,
                    attention_window_size=patches_per_frame,
                    num_contexts=frames,
                    num_ctx_tokens=patches,
                ),
            )
        )
        first_frame += frames
        first_patch += patches
    return items


class _VisionInputs(MultimodalEncoderMixin):
    supports_encoder_cache = True
    supports_mm_encoder_item_scheduling = True
    supports_vision_gpu_normalize = True
    supported_encoder_attention_backends = frozenset()

    # Pixels and M-RoPE positions move to the device. The frame grids are host metadata.
    multimodal_data_device_paths = (
        "image.pixel_values",
        "video.pixel_values_videos",
        "mrope_config.mrope_position_ids",
        "mrope_config.mrope_position_deltas",
        "multimodal_embedding",
    )

    build_batched_input = staticmethod(_qwen3vl_build_batched_input)

    def _dummy_grid(self, input_processor, modality, max_num_encoder_tokens):
        if modality == "image":
            budget = VISION_CALL_MAX_PATCHES
            if max_num_encoder_tokens is not None:
                budget = min(budget, max_num_encoder_tokens)
            return input_processor._get_dummy_grid_for_modality(modality, budget)

        video = getattr(input_processor.processor, "video_processor", None)
        if modality != "video" or video is None:
            return None
        config = input_processor.config.vision_config
        temporal = config.temporal_patch_size
        try:
            min_pixels, max_pixels = input_processor._vision_pixel_bounds("video")
        except ValueError:
            return None
        max_tokens = max_pixels // (config.patch_size**2 * temporal)
        if max_num_encoder_tokens is not None:
            max_tokens = min(max_tokens, max_num_encoder_tokens)
        max_tokens -= max_tokens % config.spatial_merge_size**2
        # The video processor bounds the complete volume. Spread it over temporal
        # groups to exercise the full item while each frame fits the attention kernel.
        max_t = max_tokens // config.spatial_merge_size**2
        max_frames = getattr(video, "max_frames", None)
        if max_frames is not None:
            max_t = min(max_t, max_frames // temporal)
        min_t = max(1, math.ceil((getattr(video, "min_frames", None) or 1) / temporal))
        best = None
        best_score = (0, 0, 0)
        max_call = min(max_tokens, VISION_CALL_MAX_PATCHES)
        for t in range(min_t, max_t + 1):
            size = input_processor._size_for_max_tokens(
                max_tokens=min(max_tokens // t, VISION_CALL_MAX_PATCHES),
                min_pixels=math.ceil(min_pixels / (t * temporal)),
                max_pixels=max_pixels // (t * temporal),
            )
            if size is None:
                continue
            h, w = size["height"] // config.patch_size, size["width"] // config.patch_size
            frame = h * w
            # Among equally large items, exercise the largest packed kernel call,
            # then the longest attention sequence within that call.
            call = min(t, VISION_CALL_MAX_PATCHES // frame) * frame
            score = (t * frame, call, frame)
            if score > best_score:
                best, best_score = (t, h, w), score
            # Increasing t only shortens the frame for the same total.
            if score[:2] == (max_tokens, max_call):
                break
        return best

    def get_mm_max_tokens_per_item(self, input_processor, max_num_encoder_tokens=None):
        sizes = {}
        for modality in _MEDIA_KEYS:
            grid = self._dummy_grid(input_processor, modality, max_num_encoder_tokens)
            if grid is not None:
                sizes[modality] = math.prod(grid)
        return sizes

    def get_dummy_mm_data(self, input_processor, *, max_num_encoder_tokens, mm_counts, dtype=None):
        return input_processor._dummy_mm_data_for_grids(
            max_num_encoder_tokens, mm_counts, dtype, partial(self._dummy_grid, input_processor)
        )

    def setup_attn_metadata(self, max_num_tokens: int, attention_metadata_capacity=None) -> None:
        """Each tower call builds its own encoder-only attention arguments from its frames."""


def _validate_request(tower, request, max_patches: int) -> None:
    """Validate each media item's geometry, embedding length and M-RoPE prompt positions."""
    data = request.py_multimodal_data
    if not data:
        return
    grids = {}
    for modality, (pixel_key, grid_key) in _MEDIA_KEYS.items():
        media = data.get(modality)
        if media is not None:
            grids[modality] = _media_grid(tower, modality, media, pixel_key, grid_key, max_patches)
    if not grids:
        return

    mrope_ids = (data.get("mrope_config") or {}).get("mrope_position_ids")
    if (
        mrope_ids is None
        or mrope_ids.dim() != 3
        or tuple(mrope_ids.shape[:2]) != (_POSITION_AXES, 1)
        or mrope_ids.shape[2] != request.prompt_len
    ):
        raise ValueError(
            f"a media request needs the [{_POSITION_AXES}, 1, {request.prompt_len}] "
            "mrope_config.mrope_position_ids from the input processor, one position per prompt "
            f"token; got {None if mrope_ids is None else tuple(mrope_ids.shape)}"
        )

    metadata = get_multimodal_item_metadata(data)
    merge_unit = tower.merge**2
    other_tokens = _check_item_order(grids, metadata, data, merge_unit)

    cumsum = data.get("multimodal_embed_mask_cumsum")
    if cumsum is not None:
        num_mm_tokens = sum(t * h * w for grid in grids.values() for t, h, w in grid) // merge_unit
        placeholders = int(cumsum[-1]) - other_tokens
        if placeholders != num_mm_tokens:
            raise ValueError(
                f"the prompt holds {placeholders} image/video placeholder tokens and its "
                f"{sorted(grids)} payload produces {num_mm_tokens} multimodal embeddings"
            )


def _media_grid(tower, modality, media, pixel_key, grid_key, max_patches):
    """One modality's `(frames, height, width)` rows, checked on the host."""
    pixel_values, grid_thw = media.get(pixel_key), media.get(grid_key)
    if pixel_values is None or grid_thw is None:
        raise ValueError(
            f"the {modality} payload carries {sorted(media)}; this target reads {pixel_key!r} "
            f"and {grid_key!r} from it"
        )
    if grid_thw.dim() != 2 or grid_thw.shape[0] == 0 or grid_thw.shape[1] != 3:
        raise ValueError(
            f"{grid_key} must hold one (t, h, w) row per {modality}; got shape "
            f"{tuple(grid_thw.shape)}"
        )
    merge = tower.merge
    grid = [tuple(int(v) for v in row) for row in grid_thw.tolist()]
    for item, (frames, height, width) in enumerate(grid):
        if min(frames, height, width) < 1 or height % merge or width % merge:
            raise ValueError(
                f"{modality} {item} has grid {(frames, height, width)}; every entry must be "
                f"positive and the sides divisible by spatial_merge_size {merge}"
            )
        if height * width > max_patches:
            raise ValueError(
                f"{modality} {item} has frames of {height}x{width} = {height * width} patches "
                f"({height * tower.patch}x{width * tower.patch} pixels); the vision tower "
                f"serves frames of at most {max_patches} patches, so send it at a "
                "lower resolution"
            )
    patches = sum(t * h * w for t, h, w in grid)
    if tuple(pixel_values.shape) != (patches, tower.pixel_width):
        raise ValueError(
            f"{pixel_key} is {tuple(pixel_values.shape)}; its grids describe {patches} patch "
            f"rows of {tower.pixel_width}"
        )
    norm = media.get("vision_normalization")
    if (
        pixel_values.dtype == torch.uint8
        and isinstance(norm, VisionNormalization)
        and norm.out_dtype == tower.dtype
    ):
        return grid
    if pixel_values.dtype != tower.dtype:
        raise ValueError(
            f"{pixel_key} is {pixel_values.dtype}; the vision tower runs at {tower.dtype} "
            "and raw pixels require matching processor normalization"
        )
    return grid


def _check_item_order(grids, metadata, data, merge_unit: int) -> int:
    """Check the vision items and count placeholder tokens belonging to other towers."""
    if metadata is None or not metadata.item_refs:
        raise ValueError(f"a request carrying {sorted(grids)} needs item metadata in prompt order")
    listed = {modality: [] for modality in grids}
    if metadata.output_embedding_lengths is None:
        raise ValueError("Grouped vision encoding needs item output embedding lengths")
    other_tokens = 0
    for entry, length in zip(metadata.item_refs, metadata.output_embedding_lengths, strict=True):
        try:
            modality, index = entry
        except (TypeError, ValueError) as error:
            raise ValueError(f"item_refs entry {entry!r} is not an item of this request") from error
        if modality not in grids:
            if modality in _MEDIA_KEYS or data.get(modality) is None:
                raise ValueError(f"item_refs entry {entry!r} is not an item of this request")
            other_tokens += length
            continue
        if not isinstance(index, int) or not 0 <= index < len(grids[modality]):
            raise ValueError(f"item_refs entry {entry!r} is not an item of this request")
        listed[modality].append(index)
        t, h, w = grids[modality][index]
        expected = t * h * w // merge_unit
        if length != expected:
            raise ValueError(
                f"{modality} {index} has output embedding length {length}, expected {expected}"
            )
    for modality, grid in grids.items():
        if sorted(listed[modality]) != list(range(len(grid))):
            raise ValueError(
                f"item_refs lists {modality} items {listed[modality]}; the payload carries "
                f"{len(grid)}"
            )
    return other_tokens


def _check_vision_deployment(model_config, llm_args) -> None:
    mm_config = llm_args.multimodal_config
    if mm_config is None or model_config.disable_mm_encoder or model_config.mm_encoder_only:
        return
    if mm_config.encoder_cuda_graph:
        raise NoTarget(
            "multimodal_config.encoder_cuda_graph: the vision tower has no encoder CUDA graph"
        )
