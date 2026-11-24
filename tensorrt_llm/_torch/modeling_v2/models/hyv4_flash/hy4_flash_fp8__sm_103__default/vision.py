# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The vision tower (HF `HYV4VLVisionModel`) and the multimodal splice's host bookkeeping.

The tower has no rotary: the `rope_2d_*` keys under `vision_config` reach no weight or module in the
reference. Every rank runs the whole tower eagerly on the whole batch's media.
"""

from __future__ import annotations

from typing import NamedTuple

import torch

from tensorrt_llm._torch.modeling_v2.catalog.attention.thop_attention import thop_attention
from tensorrt_llm.inputs.multimodal import get_multimodal_item_metadata

from .ops.activation.gelu import gelu
from .ops.gemm.cublas_mm__r11 import cublas_mm
from .ops.norm.flashinfer_rmsnorm__r12 import flashinfer_rmsnorm
from .ops.norm.layer_norm import layer_norm
from .ops.vision.patch_embed_conv import patch_embed_conv
from .ops.vision.pos_embed_interpolate import pos_embed_interpolate

#: The eps the tower's LayerNorm and RMSNorm entries are certified at.
VISION_NORM_EPS = 1e-5

#: Most patch rows per tower call (one 2048x2048 image), the ceiling the vision entries are
#: certified at. Calls are cut between frames.
VISION_CALL_MAX_PATCHES = 16384

#: `warm_up` grids: a patch count that is a multiple of 16 and one that is not (the two layer-norm
#: row specializations).
WARM_UP_GRIDS = (((1, 4, 4),), ((1, 2, 2),))

# `thop_attention`'s encoder-only mode, selected by omitting the KV cache pool arguments;
# `update_kv_cache=False` is rejected. `mask_type=0` is bidirectional per segment (one per frame).
_VISION_CALL_CONSTANTS = dict(
    k=None,
    v=None,
    output_sf=None,
    kv_cache_block_offsets=None,
    host_kv_cache_pool_pointers=None,
    host_kv_cache_pool_mapping=None,
    tokens_per_block=None,
    use_paged_context_fmha=False,
    is_fused_qkv=True,
    update_kv_cache=True,
    predicted_tokens_per_seq=1,
    local_layer_idx=0,
    max_context_q_len_override=None,
    cache_indirection=None,
    kv_scale_orig_quant=None,
    kv_scale_quant_orig=None,
    out_scale=None,
    quant_mode=0,
    beam_width=1,
    mask_type=0,
    q_scaling=1.0,
    position_embedding_type=0,
    rotary_inv_freq=None,
    rotary_cos_sin=None,
    rope_dim=0,
    rope_base=10000.0,
    rope_scale_type=0,
    rope_scale=1.0,
    rope_short_m_scale=1.0,
    rope_long_m_scale=1.0,
    rope_max_positions=1024,
    rope_original_max_positions=1024,
    attention_input_type=0,
    is_mla_enable=False,
    chunked_prefill_buffer_batch_size=1,
    attention_sinks=None,
    block_ids_per_seq=None,
    latent_cache=None,
    q_pe=None,
    q_lora_rank=None,
    kv_lora_rank=None,
    qk_nope_head_dim=None,
    qk_rope_head_dim=None,
    v_head_dim=None,
    rope_append=None,
    attention_chunk_size=None,
    softmax_stats_tensor=None,
    is_spec_decoding_enabled=False,
    use_spec_decoding=False,
    is_spec_dec_tree=False,
    spec_decoding_generation_lengths=None,
    spec_decoding_position_offsets=None,
    spec_decoding_packed_mask=None,
    spec_decoding_bl_tree_mask_offset=None,
    spec_decoding_bl_tree_mask=None,
    spec_bl_tree_first_sparse_mask_offset_kv=None,
    sparse_kv_indices=None,
    sparse_kv_offsets=None,
    sparse_attn_indices=None,
    sparse_attn_offsets=None,
    sparse_attn_indices_block_size=0,
    mrope_rotary_cos_sin=None,
    mrope_position_deltas=None,
    helix_position_offsets=None,
    helix_is_inactive_rank=None,
    is_cross=False,
    cross_kv=None,
    relative_attention_bias=None,
    relative_attention_max_distance=0,
    kv_norm_weight=None,
    kv_norm_eps=1e-6,
    skip_correction_threshold=0.0,
)

#: Per modality: the `multimodal_data` keys of the flat patch rows and the host `(t, h, w)` grid.
MEDIA_KEYS = {
    "image": ("pixel_values", "image_grid_thw"),
    "video": ("pixel_values_videos", "video_grid_thw"),
}

#: `multimodal_data` tensors moved to the device; the grids stay on the host.
DEVICE_PATHS = (
    "image.pixel_values",
    "video.pixel_values_videos",
)

#: `multimodal_data` key of a chunked request's last tower call that later chunks still read:
#: `(first placeholder row, rows)`. The chunk that consumes its last row drops it.
_TOWER_ROWS_KEY = "hyv4_flash_tower_rows"


class _VisionBlockWeights(NamedTuple):
    n1_w: torch.Tensor
    n1_b: torch.Tensor
    qkv: torch.Tensor
    qkv_b: torch.Tensor
    proj: torch.Tensor
    proj_b: torch.Tensor
    n2_w: torch.Tensor
    n2_b: torch.Tensor
    fc1: torch.Tensor
    fc1_b: torch.Tensor
    fc2: torch.Tensor
    fc2_b: torch.Tensor


class _VisionWeights(NamedTuple):
    patch_w: torch.Tensor
    patch_b: torch.Tensor
    pos: torch.Tensor
    blocks: list
    m_norm: torch.Tensor
    m_conv_w: torch.Tensor
    m_conv_b: torch.Tensor
    m_out_w: torch.Tensor
    m_out_b: torch.Tensor
    m_newline: torch.Tensor


def _vision_views(w: dict, depth: int) -> _VisionWeights:
    """The weight table's `vis_*` entries as the views the ops read (GEMM weights `.t()`)."""
    return _VisionWeights(
        patch_w=w["vis_patch_w"],
        patch_b=w["vis_patch_b"],
        pos=w["vis_pos"],
        blocks=[
            _VisionBlockWeights(
                n1_w=w[f"l{b}_vis_n1_w"],
                n1_b=w[f"l{b}_vis_n1_b"],
                qkv=w[f"l{b}_vis_qkv"].t(),
                qkv_b=w[f"l{b}_vis_qkv_b"],
                proj=w[f"l{b}_vis_proj_w"].t(),
                proj_b=w[f"l{b}_vis_proj_b"],
                n2_w=w[f"l{b}_vis_n2_w"],
                n2_b=w[f"l{b}_vis_n2_b"],
                fc1=w[f"l{b}_vis_fc1_w"].t(),
                fc1_b=w[f"l{b}_vis_fc1_b"],
                fc2=w[f"l{b}_vis_fc2_w"].t(),
                fc2_b=w[f"l{b}_vis_fc2_b"],
            )
            for b in range(depth)
        ],
        m_norm=w["vis_m_norm"],
        m_conv_w=w["vis_m_conv_w"],
        m_conv_b=w["vis_m_conv_b"],
        m_out_w=w["vis_m_out_w"],
        m_out_b=w["vis_m_out_b"],
        m_newline=w["vis_m_newline"],
    )


def _frame_calls(
    grid: list[tuple[int, int, int]], max_rows: int
) -> list[list[tuple[int, int, int]]]:
    """Split `grid` into calls of whole frames, at most `max_rows` rows each, as grid rows."""
    calls: list[list[tuple[int, int, int]]] = [[]]
    room = max_rows
    for frames, height, width in grid:
        rows = height * width
        while frames:
            take = min(frames, room // rows)
            if take == 0:
                calls.append([])
                room = max_rows
                continue
            calls[-1].append((take, height, width))
            room -= take * rows
            frames -= take
    return calls


def _frame_rows(t: int, height: int, width: int, merge: int) -> int:
    """Placeholder rows of one grid row: merged rows plus one `image_newline` per merged row."""
    return t * (height // merge) * (width // merge + 1)


class VisionTower:
    """The ViT and its patch merger over one batch's images or videos."""

    def __init__(self, w: dict, v) -> None:
        """`w` is the core's loaded weight table; `v` the `dims.vision` geometry."""
        self.hidden = int(v.hidden)
        self.heads = int(v.heads)
        self.head_dim = int(v.head_dim)
        self.merge = int(v.merge)
        self.patch = int(v.patch)
        self.out_hidden = int(v.out_hidden)
        self.merge_dim = int(v.merge_dim)
        self.pixel_width = int(v.pixel_width)
        self.position_edge = int(v.position_edge)
        assert abs(float(v.eps) - VISION_NORM_EPS) < 1e-12, (
            f"the tower's norms are certified at eps {VISION_NORM_EPS}; config says {v.eps}"
        )
        # The checkpoint's `vit.*` tensors are BF16 even in this FP8 checkpoint.
        self.dtype = w["vis_patch_w"].dtype
        self._w = _vision_views(w, int(v.depth))
        # Persistent scratch `thop_attention` grows in place.
        self._ws: torch.Tensor | None = None

    def warm_up(self) -> None:
        """Run every `WARM_UP_GRIDS` grid on zero pixels so the kernels compile at load."""
        device = self._w.pos.device
        for grid in WARM_UP_GRIDS:
            rows = sum(t * h * w for t, h, w in grid)
            self._run(torch.zeros(rows, self.pixel_width, dtype=self.dtype, device=device), grid)

    def __call__(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """`[sum(t*h*w), pixel_width]` patch rows and the host grid -> placeholder rows, in grid order."""
        assert pixel_values.dim() == 2 and pixel_values.shape[1] == self.pixel_width, tuple(
            pixel_values.shape
        )
        grid: list[tuple[int, int, int]] = []
        for row in grid_thw.tolist():
            assert len(row) == 3, row
            frames, height, width = (int(v) for v in row)
            assert frames >= 1 and height >= 1 and width >= 1, row
            assert height % self.merge == 0 and width % self.merge == 0, row
            assert height * width <= VISION_CALL_MAX_PATCHES, row
            grid.append((frames, height, width))
        assert grid, "grid_thw describes no image or video"
        described = sum(t * h * w for t, h, w in grid)
        assert described == pixel_values.shape[0], (described, pixel_values.shape[0])

        # The processor hands float32; the reference casts inside the patch embedding.
        pixels = pixel_values.to(self.dtype)
        outputs = []
        start = 0
        for call in _frame_calls(grid, VISION_CALL_MAX_PATCHES):
            rows = sum(t * h * w for t, h, w in call)
            outputs.append(self._run(pixels[start : start + rows], call))
            start += rows
        return outputs[0] if len(outputs) == 1 else torch.cat(outputs, 0)

    def _run(self, pixel_values: torch.Tensor, grid: list[tuple[int, int, int]]) -> torch.Tensor:
        """One tower call over whole frames; `grid` describes exactly the rows of `pixel_values`."""
        dt, dev = pixel_values.dtype, pixel_values.device
        assert pixel_values.is_cuda, "pixel_values must reach the tower on the device"

        # One attention segment per frame: a video's frames do not attend to each other.
        seg_lens: list[int] = []
        for frames, height, width in grid:
            seg_lens.extend([height * width] * frames)
        num_patches = pixel_values.shape[0]
        num_segments = len(seg_lens)
        max_seg = max(seg_lens)

        host_lens = torch.tensor(seg_lens, dtype=torch.int32)
        host_types = torch.zeros([num_segments], dtype=torch.int32)
        host_total = torch.tensor([num_patches, 0], dtype=torch.int32)
        dev_lens = host_lens.to(dev)
        if self._ws is None:
            self._ws = torch.empty([0], dtype=torch.int8, device=dev)

        V = self._w
        pos = [pos_embed_interpolate(V.pos, t, h, w, self.position_edge, dt) for t, h, w in grid]
        h = torch.add(
            patch_embed_conv(pixel_values, V.patch_w, V.patch_b),
            pos[0] if len(pos) == 1 else torch.cat(pos, 0),
        )
        attn_out = torch.empty([num_patches, self.hidden], dtype=dt, device=dev)

        for B in V.blocks:
            qkv = cublas_mm(layer_norm(h, B.n1_w, B.n1_b, VISION_NORM_EPS), B.qkv, B.qkv_b)
            attn_out = thop_attention(
                q=qkv,
                output=attn_out,
                workspace=self._ws,
                sequence_length=dev_lens,
                host_past_key_value_lengths=host_lens,
                host_total_kv_lens=host_total,
                context_lengths=dev_lens,
                host_context_lengths=host_lens,
                host_request_types=host_types,
                max_num_requests=num_segments,
                max_context_length=max_seg,
                max_seq_len=max_seg,
                attention_window_size=max_seg,
                num_contexts=num_segments,
                num_ctx_tokens=num_patches,
                num_heads=self.heads,
                num_kv_heads=self.heads,
                head_size=self.head_dim,
                **_VISION_CALL_CONSTANTS,
            )
            h = torch.add(h, cublas_mm(attn_out, B.proj, B.proj_b))
            mlp = cublas_mm(layer_norm(h, B.n2_w, B.n2_b, VISION_NORM_EPS), B.fc1, B.fc1_b)
            h = torch.add(h, cublas_mm(gelu(mlp), B.fc2, B.fc2_b))

        return self._merge(h, grid)

    def _merge(self, h_rows: torch.Tensor, grid: list[tuple[int, int, int]]) -> torch.Tensor:
        """Per frame: RMSNorm, 2x2 conv, GELU, 1x1 projection, and an `image_newline` row per merged row."""
        V = self._w
        merge, out = self.merge, self.out_hidden
        start = 0
        items = []
        for frames, height, width in grid:
            patches = frames * height * width
            x = flashinfer_rmsnorm(h_rows[start : start + patches], V.m_norm, VISION_NORM_EPS)
            start += patches
            merged_h, merged_w = height // merge, width // merge
            # 2x2 blocks gathered `(channel, kh, kw)` from the frame read as an `(h, w)` image.
            blocks = (
                x.reshape(frames, height, width, self.hidden)
                .permute(0, 3, 1, 2)
                .reshape(frames, self.hidden, merged_h, merge, merged_w, merge)
                .permute(0, 2, 4, 1, 3, 5)
                .reshape(-1, self.merge_dim)
            )
            merged = patch_embed_conv(blocks, V.m_conv_w, V.m_conv_b)
            merged = cublas_mm(gelu(merged), V.m_out_w.view(out, -1).t(), V.m_out_b)
            newline = V.m_newline.view(1, 1, 1, out).expand(frames, merged_h, 1, out)
            items.append(
                torch.cat([merged.view(frames, merged_h, merged_w, out), newline], dim=2).reshape(
                    -1, out
                )
            )
        return items[0] if len(items) == 1 else torch.cat(items, 0)


def multimodal_embeds(tower: VisionTower | None, kwargs: dict):
    """Run the tower on this batch's media: `(rows, mm_token_indices)`, or None without media.

    `multimodal_params` covers a subset of the context requests (never zip it positionally); a
    chunk runs the tower only on the frames that hold its rows, once per frame across chunks."""
    mm_params = kwargs.get("multimodal_params") or []
    if not mm_params:
        return None
    rows = []
    for params in mm_params:
        data = getattr(params, "multimodal_data", None) or {}
        if data.get("image") is None and data.get("video") is None:
            continue
        runtime = getattr(params, "multimodal_runtime", None)
        if runtime is not None and int(runtime.num_mm_tokens_in_chunk) == 0:
            continue
        assert tower is not None, "post_load_weights builds the vision tower before any step"
        items = _items_in_prompt_order(data)
        total = sum(_frame_rows(*grid, tower.merge) for _, _, grid in items)
        if runtime is None:
            rows.append(_tower_rows(tower, items, 0, total)[1])
            continue
        # Chunked prefill or reused prefix: only this chunk's slice.
        start = int(runtime.num_cached_mm_tokens)
        count = int(runtime.num_mm_tokens_in_chunk)
        assert int(runtime.total_embeds_in_request) == total, (
            runtime.total_embeds_in_request,
            total,
        )
        assert 0 <= start <= start + count <= total, (start, count, total)
        rows.append(_chunk_rows(tower, data, items, start, count))
    if not rows:
        return None
    vision = rows[0] if len(rows) == 1 else torch.cat(rows, 0)
    mm_token_indices = kwargs.get("mm_token_indices")
    assert mm_token_indices is not None, "multimodal params shipped without mm_token_indices"
    assert mm_token_indices.shape[0] == vision.shape[0], (
        mm_token_indices.shape[0],
        vision.shape[0],
    )
    return vision, mm_token_indices


def _items_in_prompt_order(data: dict) -> list[tuple[torch.Tensor, int, tuple]]:
    """One request's media items in placeholder order: `(pixel_values, first patch row, (t, h, w))`.

    Two modalities interleave by `item_metadata.item_refs` (checked at admission)."""
    media = {}
    for modality, (pixel_key, grid_key) in MEDIA_KEYS.items():
        payload = data.get(modality)
        if payload is None:
            continue
        pixel_values, grid_thw = payload[pixel_key], payload[grid_key]
        grid = [tuple(int(v) for v in row) for row in grid_thw.tolist()]
        firsts, patches = [], 0
        for frames, height, width in grid:
            firsts.append(patches)
            patches += frames * height * width
        media[modality] = (pixel_values, grid, firsts)
    if len(media) == 1:
        modality, (_, grid, _) = next(iter(media.items()))
        order = [(modality, index) for index in range(len(grid))]
    else:
        order = get_multimodal_item_metadata(data).item_refs
    return [
        (media[modality][0], media[modality][2][int(index)], media[modality][1][int(index)])
        for modality, index in order
    ]


def _tower_rows(
    tower: VisionTower, items: list, start: int, count: int
) -> tuple[int, torch.Tensor]:
    """`(first row, rows)`: one tower call over the frames of `items` that hold placeholder rows
    `[start, start + count)`, from the first of those frames' first row on.

    A frame's rows depend on its own patches alone, so a chunk of a long video does not run the
    request's other frames."""
    merge = tower.merge
    end = start + count
    spans: list[list] = []  # [pixel_values, first patch row, end patch row]
    grid: list[tuple[int, int, int]] = []
    first_row = None
    row = 0  # the item's first placeholder row
    for pixel_values, first_patch, (frames, height, width) in items:
        frame_rows = _frame_rows(1, height, width, merge)
        lo = max(start - row, 0) // frame_rows
        hi = min(-(-(end - row) // frame_rows), frames)
        if lo < hi:
            if first_row is None:
                first_row = row + lo * frame_rows
            patch_lo = first_patch + lo * height * width
            patch_hi = first_patch + hi * height * width
            if spans and spans[-1][0] is pixel_values and spans[-1][2] == patch_lo:
                spans[-1][2] = patch_hi
            else:
                spans.append([pixel_values, patch_lo, patch_hi])
            grid.append((hi - lo, height, width))
        row += frames * frame_rows
    assert first_row is not None, (start, count, row)
    parts = [pixel_values[lo:hi] for pixel_values, lo, hi in spans]
    pixels = parts[0] if len(parts) == 1 else torch.cat(parts, 0)
    return first_row, tower(pixels, torch.tensor(grid))


def _chunk_rows(tower: VisionTower, data: dict, items: list, start: int, count: int):
    """Placeholder rows `[start, start + count)` of one prefill chunk.

    Rows the request's previous tower call produced (`_TOWER_ROWS_KEY` in `data`) are reused, so a
    frame several chunks cut runs once."""
    end = start + count
    kept = data.pop(_TOWER_ROWS_KEY, None)
    parts = []
    if kept is not None:
        first, out = kept
        if first <= start < first + out.shape[0]:
            take = min(end, first + out.shape[0])
            parts.append(out[start - first : take - first])
            start = take
    if start < end:
        kept = _tower_rows(tower, items, start, end - start)
        first, out = kept
        parts.append(out[start - first : end - first])
    if kept is not None and end < kept[0] + kept[1].shape[0]:
        data[_TOWER_ROWS_KEY] = kept
    return parts[0] if len(parts) == 1 else torch.cat(parts, 0)


def validate_request(tower, request) -> None:
    """Admission check from the host payload: refuse what the forward would assert on (an assert
    inside the step ends the executor) and precomputed embeddings, which the tower cannot splice."""
    data = request.py_multimodal_data
    if not data:
        return
    if data.get("multimodal_embedding") is not None:
        raise ValueError(
            "this target does not splice precomputed image embeddings; send the image/video pixels"
        )
    grids = {}
    for modality, (pixel_key, grid_key) in MEDIA_KEYS.items():
        media = data.get(modality)
        if media is not None:
            grids[modality] = _media_grid(tower, modality, media, pixel_key, grid_key)
    if not grids:
        return

    if len(grids) > 1:
        metadata = get_multimodal_item_metadata(data)
        _check_item_order(grids, metadata.item_refs if metadata is not None else None)

    cumsum = data.get("multimodal_embed_mask_cumsum")
    if cumsum is not None:
        rows = sum(
            _frame_rows(*item, tower.merge)
            for modality_grid in grids.values()
            for item in modality_grid
        )
        placeholders = int(cumsum[-1])
        if placeholders != rows:
            raise ValueError(
                f"the prompt holds {placeholders} image/video placeholder tokens and its "
                f"{' and '.join(sorted(grids))} payload produces {rows} embedding rows"
            )


def _media_grid(tower, modality, media, pixel_key, grid_key):
    """One modality's `(t, h, w)` rows, checked on the host."""
    pixel_values, grid_thw = media.get(pixel_key), media.get(grid_key)
    if pixel_values is None or grid_thw is None:
        raise ValueError(f"the {modality} payload needs {pixel_key!r} and {grid_key!r}")
    if grid_thw.dim() != 2 or grid_thw.shape[0] == 0 or grid_thw.shape[1] != 3:
        raise ValueError(f"{grid_key} must hold one (t, h, w) row per {modality}")
    merge = tower.merge
    grid = [tuple(int(v) for v in row) for row in grid_thw.tolist()]
    for item, (frames, height, width) in enumerate(grid):
        if min(frames, height, width) < 1 or height % merge or width % merge:
            raise ValueError(
                f"{modality} {item} has grid {(frames, height, width)}; entries must be positive "
                f"and the sides divisible by spatial_merge_size {merge}"
            )
        if height * width > VISION_CALL_MAX_PATCHES:
            raise ValueError(
                f"{modality} {item} has {height * tower.patch}x{width * tower.patch} pixel frames; "
                f"the tower serves at most {VISION_CALL_MAX_PATCHES} patches per frame"
            )
    patches = sum(t * h * w for t, h, w in grid)
    if tuple(pixel_values.shape) != (patches, tower.pixel_width):
        raise ValueError(
            f"{pixel_key} is {tuple(pixel_values.shape)}; its grids describe {patches} patch "
            f"rows of {tower.pixel_width}"
        )
    if not pixel_values.is_floating_point():
        raise ValueError(f"{pixel_key} is {pixel_values.dtype}; expected normalized float pixels")
    return grid


def _check_item_order(grids, order) -> None:
    """A request with several modalities lists each of its items once, in prompt order."""
    if not order:
        raise ValueError(f"a request carrying {sorted(grids)} needs item metadata (item_refs)")
    listed = {modality: [] for modality in grids}
    for entry in order:
        try:
            modality, index = entry
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"item_refs entry {entry!r} is not a (modality, index) pair"
            ) from error
        if modality not in listed:
            raise ValueError(
                f"item_refs lists a {modality!r} item; this request carries {sorted(grids)}"
            )
        listed[modality].append(int(index))
    for modality, grid in grids.items():
        if sorted(listed[modality]) != list(range(len(grid))):
            raise ValueError(
                f"item_refs lists {modality} items {listed[modality]}; the payload carries "
                f"{len(grid)}"
            )
