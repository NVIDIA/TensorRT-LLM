# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The vision tower and the multimodal splice's host bookkeeping.

The checkpoint's 27-block SigLIP2-like ViT (`vit.layers.*`, hidden 1152, 16 heads
of 72, FFN 4304) and its RMSNorm / 2x2-conv / 1x1-conv projector
(`vit.perceive.*`, out width 3072) turn a request's media into one
`[rows, out_hidden]` row per image/video placeholder token, which the core's
forward writes over the token embedding of those rows.

The reference is HuggingFace's `HYV4VLVisionModel` (the tower `hy_v4_flash`
reuses from `hy_v4_vl`): patch embedding plus a learned, bilinearly interpolated
position table, 27 pre-norm blocks, and a merger whose every merged row ends with
the `image_newline` row -- so a frame contributes `(h/merge) * (w/merge + 1)`
placeholder rows, the checkpoint's own image processor's `_num_frame_tokens`.
Attention is bidirectional per frame and carries **no rotary**: the reference's
`HYV4VLVisionAttention` has no rope, and the `rope_2d_percent` / `rope_2d_theta`
keys the checkpoint publishes under `vision_config` reach no weight or module in
it; this tower is rope-free to match.

Every rank holds the whole tower and runs it eagerly, outside any captured graph
(context steps are not captured -- `prefill_cuda_graph_backend` defaults to
disabled), on the media of the requests it serves: under attention TP, the whole
batch's, with identical rows on every rank.
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

#: The eps of the tower's two LayerNorms (the reference's `attribute_map` maps
#: `layer_norm_eps` onto `rms_norm_eps`) and of the merger's RMSNorm; the value
#: `vision_config` publishes and the LayerNorm entry is certified at.
VISION_NORM_EPS = 1e-5

#: Most patch rows per tower call: one 2048x2048 image (`max_image_size**2 /
#: patch_size**2`, the config's `max_vit_seq_len`), the ceiling the target's
#: vision entries are certified at. Calls are cut between frames, so a request
#: is bounded per frame.
VISION_CALL_MAX_PATCHES = 16384

#: The grids `VisionTower.warm_up` runs: one patch count that is a multiple of
#: 16 (4x4) and one that is not (2x2), which are the two layer-norm row
#: specializations.
WARM_UP_GRIDS = (((1, 4, 4),), ((1, 2, 2),))

#: `mask_type` 0: bidirectional within a batch segment, and one segment is one
#: frame (a video's frames do not attend to each other).
_VISION_MASK_PADDING = 0

# `thop_attention`'s encoder-only mode, selected by omitting the KV cache pool
# arguments. `update_kv_cache=False` is rejected, so it stays True. The head
# geometry (16 heads, head dimension 72) and the softmax scale are passed per
# call; `q_scaling=1.0` is the catalog entry's default `1 / sqrt(head_size)`,
# the reference's own `head_dim**-0.5`. Every rope argument is inert: the tower
# applies no rotary. Adapted from the qwen4_exp target's
# `qwen3_8_flash_next_fp8__sm_103__unified/vision.py` `_VISION_CALL_CONSTANTS`,
# which drives the same catalog entry in the same encoder mode.
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
    mask_type=_VISION_MASK_PADDING,
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

#: Per modality: the `multimodal_data` keys of the flat patch rows and the host
#: `(t, h, w)` grid.
MEDIA_KEYS = {
    "image": ("pixel_values", "image_grid_thw"),
    "video": ("pixel_values_videos", "video_grid_thw"),
}

#: `multimodal_data` tensors moved to the device; the grids stay on the host,
#: read sync-free. Hy4-Flash needs no mRoPE ids (`rope_scaling` publishes no
#: `mrope` type), so nothing else moves.
DEVICE_PATHS = (
    "image.pixel_values",
    "video.pixel_values_videos",
)

#: The `multimodal_data` key of a chunked request's last tower call that later
#: chunks still read: `(first placeholder row, rows)`. The executor hands every
#: chunk the request's own `py_multimodal_data` dict, so the entry lives on the
#: request; the chunk that consumes its last row drops it. Named for this target
#: so two targets' keys never collide on a shared payload.
_TOWER_ROWS_KEY = "hyv4_flash_tower_rows"


class _VisionBlockWeights(NamedTuple):
    """One vision block's weights (`.t()` GEMM views); its LayerNorms have biases."""

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
    """The tower's weights: patch embedding, position table, blocks and merger."""

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
    """The weight table's `vis_*` entries (`weights.py`) as the views the ops read."""
    return _VisionWeights(
        # Not transposed: `patch_embed_conv` takes the checkpoint's
        # `[out, in, kh, kw]` filter as it is.
        patch_w=w["vis_patch_w"],
        patch_b=w["vis_patch_b"],
        pos=w["vis_pos"],
        blocks=[
            _VisionBlockWeights(
                n1_w=w[f"l{b}_vis_n1_w"],
                n1_b=w[f"l{b}_vis_n1_b"],
                # `weights.py` packs the checkpoint's separate q|k|v projections
                # into the row layout `is_fused_qkv` reads: q's rows, then k's,
                # then v's.
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
        # The merger's RMSNorm has no bias, unlike the blocks' LayerNorms.
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
        assert rows <= max_rows, (
            f"one frame is {height}x{width} = {rows} patch rows; the vision ops are "
            f"certified up to {max_rows} rows in one call"
        )
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
    """One grid row's placeholder rows: `t * (h/merge) * (w/merge + 1)`.

    The merged row count plus one `image_newline` token per merged row, which is
    the checkpoint processor's `_num_frame_tokens`, and what the executor's
    placeholder accounting counts.
    """
    return t * (height // merge) * (width // merge + 1)


class VisionTower:
    """One request's images or videos through the 27-block ViT and its patch merger.

    Attention is bidirectional per frame, one context "sequence" per frame for
    `thop_attention`'s encoder mode. The tower adds a learned, interpolated
    position table to the patch embeddings and nothing else -- the reference has
    no rotary, hence no entry here rotates q or k. Calls of at most
    `VISION_CALL_MAX_PATCHES` rows are cut only between frames: every op is per
    frame or per row and merger blocks stay whole.
    """

    def __init__(self, w: dict, v) -> None:
        """`w` is the core's loaded weight table; `v` the `dims.vision` geometry."""
        self.depth = int(v.depth)
        self.hidden = int(v.hidden)
        self.heads = int(v.heads)
        self.head_dim = int(v.head_dim)
        self.inter = int(v.inter)
        self.merge = int(v.merge)
        self.patch = int(v.patch)
        self.out_hidden = int(v.out_hidden)
        self.merge_dim = int(v.merge_dim)
        #: One flat pixel row per patch, `(channels, patch, patch)`: the image
        #: processor's own row layout (`pixel_values` is `[patches, 3*16*16]`).
        self.pixel_width = int(v.pixel_width)
        #: `max_image_size // patch_size`: the position table's grid side, its
        #: `position_edge**2 + 1` rows led by the cls token.
        self.position_edge = int(v.position_edge)
        assert abs(float(v.eps) - VISION_NORM_EPS) < 1e-12, (
            f"the tower's LayerNorms run at the eps its entry is certified at "
            f"({VISION_NORM_EPS}); this config says {v.eps}"
        )
        # The checkpoint's `vit.*` tensors are BF16 even in this FP8 checkpoint,
        # and the tower runs at the activation dtype the weight table declares.
        self.dtype = w["vis_patch_w"].dtype
        assert w["vis_pos"].shape[0] == self.position_edge**2 + 1, (
            f"the position table has {w['vis_pos'].shape[0]} rows; a {self.position_edge}x"
            f"{self.position_edge} grid plus the cls row is {self.position_edge**2 + 1}"
        )
        self._w = _vision_views(w, self.depth)
        # Persistent scratch `thop_attention` grows in place; not the engine's own workspace.
        self._ws: torch.Tensor | None = None

    def warm_up(self) -> None:
        """Run the tower once per `WARM_UP_GRIDS` grid on zero pixels, so every kernel it
        launches is compiled at load rather than on the first media request."""
        device = self._w.pos.device
        for grid in WARM_UP_GRIDS:
            rows = sum(t * h * w for t, h, w in grid)
            self._run(torch.zeros(rows, self.pixel_width, dtype=self.dtype, device=device), grid)

    def __call__(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """The `[sum over items of t*(h/merge)*(w/merge+1), out_hidden]` placeholder rows, in
        grid order.

        `pixel_values` is `[sum(t*h*w), channels*patch*patch]`; `grid_thw` stays on the host.
        """
        assert pixel_values.dim() == 2 and pixel_values.shape[1] == self.pixel_width, (
            f"pixel_values must be [patches, {self.pixel_width}] flat patch rows; got "
            f"{tuple(pixel_values.shape)}"
        )
        grid: list[tuple[int, int, int]] = []
        for row in grid_thw.tolist():
            assert len(row) == 3, f"grid_thw rows are (t, h, w); got {row}"
            frames, height, width = (int(v) for v in row)
            assert frames >= 1 and height >= 1 and width >= 1, row
            assert height % self.merge == 0 and width % self.merge == 0, (
                f"grid sides must divide by spatial_merge_size {self.merge}; got "
                f"height={height}, width={width}"
            )
            assert height * width <= VISION_CALL_MAX_PATCHES, (
                f"a frame of {height}x{width} = {height * width} patches exceeds the "
                f"{VISION_CALL_MAX_PATCHES}-patch ceiling of one tower call"
            )
            grid.append((frames, height, width))
        assert grid, "grid_thw describes no image or video"
        described = sum(t * h * w for t, h, w in grid)
        assert described == pixel_values.shape[0], (
            f"grid_thw describes {described} patch rows and pixel_values carries "
            f"{pixel_values.shape[0]}"
        )

        # The reference casts the pixels inside the patch embedding
        # (`pixel_values.to(target_dtype)`); the checkpoint's processor hands float32.
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
            # No rotary: the packed q|k|v rows go to the kernel as they are.
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
            # Non-gated MLP with the erf GELU (`ACT2FN["gelu"]` at both call sites).
            h = torch.add(h, cublas_mm(gelu(mlp), B.fc2, B.fc2_b))

        return self._merge(h, grid)

    def _merge(self, h_rows: torch.Tensor, grid: list[tuple[int, int, int]]) -> torch.Tensor:
        """`[sum(t*h*w), hidden]` blocks -> `[sum(t*(h/merge)*(w/merge+1)), out_hidden]`.

        Per row: the merger's RMSNorm. Per frame: the 2x2 blocks gathered
        `(channel, kh, kw)`, one `patch_embed_conv` (proj.0), the erf GELU, the
        1x1 projection (proj.2), and the `image_newline` row the reference
        concatenates at the end of every merged row.
        """
        V = self._w
        merge, out = self.merge, self.out_hidden
        start = 0
        items = []
        for frames, height, width in grid:
            patches = frames * height * width
            x = flashinfer_rmsnorm(h_rows[start : start + patches], V.m_norm, VISION_NORM_EPS)
            start += patches
            merged_h, merged_w = height // merge, width // merge
            # The reference reads the frame's rows as an `(h, w)` image
            # (`reshape(-1, *size, C).permute(0, 3, 1, 2)`) and runs the 2x2
            # convolution on that layout; the rows are the processor's patch
            # order, in neither case re-sorted here.
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

    `rows` is `[R, out_hidden]` in packed-batch order, `mm_token_indices` the `[R]` rows they
    replace. The shell calls it: the inherited `DecoderModelForCausalLM.forward` drops
    `multimodal_params`, which covers a subset of the context requests (never zip it
    positionally). `multimodal_runtime` counts rows over a request's media together, in prompt
    order; a chunk runs the tower on only the frames that hold its rows (`_tower_rows`), once
    per frame across the request's chunks (`_chunk_rows`)."""
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
            # Every placeholder row of this request lies in another chunk.
            continue
        assert tower is not None, "post_load_weights builds the vision tower before any step"
        items = _items_in_prompt_order(params, data)
        total = sum(_frame_rows(*grid, tower.merge) for _, _, grid in items)
        if runtime is None:
            rows.append(_tower_rows(tower, items, 0, total)[1])
            continue
        # Chunked prefill or reused prefix: only this chunk's slice.
        start = int(runtime.num_cached_mm_tokens)
        count = int(runtime.num_mm_tokens_in_chunk)
        assert int(runtime.total_embeds_in_request) == total, (
            f"the executor counted {runtime.total_embeds_in_request} placeholder rows for "
            f"this request and its grids describe {total}; the two are derived from the "
            "same grids and a disagreement means the merger's row count is wrong"
        )
        assert 0 <= start <= start + count <= total, (start, count, total)
        rows.append(_chunk_rows(tower, data, items, start, count))
    if not rows:
        return None
    vision = rows[0] if len(rows) == 1 else torch.cat(rows, 0)
    mm_token_indices = kwargs.get("mm_token_indices")
    assert mm_token_indices is not None, (
        "the executor shipped multimodal params but no mm_token_indices; that pairing is what "
        "the shell's `multimodal_token_ids` buys, and without it there is no sync-free way to "
        "find the placeholder rows"
    )
    assert mm_token_indices.shape[0] == vision.shape[0], (
        f"the batch has {mm_token_indices.shape[0]} placeholder rows and the tower produced "
        f"{vision.shape[0]} embedding rows"
    )
    return vision, mm_token_indices


def _items_in_prompt_order(params, data: dict) -> list[tuple[torch.Tensor, int, tuple]]:
    """One request's media items in placeholder order: `(pixel_values, first patch row, (t, h, w))`.

    Two modalities interleave by `item_metadata.item_refs`, the only host-side record of order."""
    media = {}
    for modality, (pixel_key, grid_key) in MEDIA_KEYS.items():
        payload = data.get(modality)
        if payload is None:
            continue
        pixel_values, grid_thw = payload.get(pixel_key), payload.get(grid_key)
        assert pixel_values is not None and grid_thw is not None, (
            f"the multimodal payload carries {sorted(payload)}; this target reads {pixel_key!r} "
            f"and {grid_key!r} from it"
        )
        assert not grid_thw.is_cuda, f"{grid_key} must stay on the host; see DEVICE_PATHS"
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
        metadata = get_multimodal_item_metadata(data)
        order = metadata.item_refs if metadata is not None else None
        assert order, (
            f"a request carrying {sorted(media)} needs item metadata to splice them in prompt "
            "order; the payload holds each modality's rows apart"
        )
        listed = {modality: [] for modality in media}
        for modality, index in order:
            assert modality in media, (
                f"item_refs lists a {modality!r} item; this request carries {sorted(media)}"
            )
            listed[modality].append(int(index))
        for modality, (_, grid, _) in media.items():
            assert sorted(listed[modality]) == list(range(len(grid))), (
                f"item_refs lists {modality} items {listed[modality]}; the payload carries "
                f"{len(grid)}"
            )
    return [
        (media[modality][0], media[modality][2][int(index)], media[modality][1][int(index)])
        for modality, index in order
    ]


def _tower_rows(
    tower: VisionTower, items: list, start: int, count: int
) -> tuple[int, torch.Tensor]:
    """`(first row, rows)`: one tower call over the frames of one request's `items` (prompt
    order) that hold its placeholder rows `[start, start + count)`, from the first of those
    frames' first row on.

    A frame's rows depend on its own patches alone (attention is per frame, every other op per
    row), so a chunk of a long video does not run the request's other frames. Adjacent frames of
    one pixel tensor stay one view; items of two modalities are copied into one tensor."""
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
    """Placeholder rows `[start, start + count)` of one chunk of a request's prefill.

    Rows the request's previous tower call already produced (`_TOWER_ROWS_KEY` in `data`, the
    request's `multimodal_data`) are read from it; the tower runs only on the frames holding the
    rest. So an image or frame that several chunks cut runs once: a max-size image under
    512-token chunks runs in its first chunk and keeps its `[rows, out_hidden]` rows until the
    chunk with its last row. The rows are bitwise the one-pass rows."""
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
    """Admission check the executor runs on every new request, from its host-side payload only.

    It refuses what the forward would assert on: an assert inside the step ends the executor.
    It also refuses precomputed image embeddings, which the tower cannot splice."""
    data = request.py_multimodal_data
    if not data:
        return
    if data.get("multimodal_embedding") is not None:
        raise ValueError(
            "this target does not splice precomputed image embeddings "
            "(multimodal_data['multimodal_embedding']); send the image/video pixels "
            "(multimodal_data['image'/'video']) so the vision tower runs"
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
        if height * width > VISION_CALL_MAX_PATCHES:
            raise ValueError(
                f"{modality} {item} has frames of {height}x{width} = {height * width} patches "
                f"({height * tower.patch}x{width * tower.patch} pixels); the vision tower "
                f"serves frames of at most {VISION_CALL_MAX_PATCHES} patches, so send it at a "
                "lower resolution"
            )
    patches = sum(t * h * w for t, h, w in grid)
    if tuple(pixel_values.shape) != (patches, tower.pixel_width):
        raise ValueError(
            f"{pixel_key} is {tuple(pixel_values.shape)}; its grids describe {patches} patch "
            f"rows of {tower.pixel_width}"
        )
    if not pixel_values.is_floating_point():
        raise ValueError(
            f"{pixel_key} is {pixel_values.dtype}; the image processor's rescaled and "
            "normalized pixels are what the tower casts to its own dtype"
        )
    return grid


def _check_item_order(grids, order) -> None:
    """A request with several modalities lists its items in prompt order."""
    if not order:
        raise ValueError(
            f"a request carrying {sorted(grids)} needs item metadata to splice them in prompt "
            "order; the payload holds each modality's rows apart"
        )
    listed = {modality: [] for modality in grids}
    for entry in order:
        try:
            modality, index = entry
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"item_refs entry {entry!r} is not an item of this request, which carries "
                f"{sorted(grids)}; entries are (modality, index) pairs"
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
