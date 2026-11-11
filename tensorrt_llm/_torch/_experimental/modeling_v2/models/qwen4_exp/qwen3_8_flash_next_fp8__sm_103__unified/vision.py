# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The vision tower and the multimodal splice's host bookkeeping.

The checkpoint's 27-block bf16 ViT and its 2x2 patch merger turn a request's media into one
`[rows, hidden]` row per placeholder token, which the prefill target's `_splice` writes over the
token embedding. Every rank holds the whole tower and runs it eagerly, outside any captured graph,
on the media of the requests it serves (under attention TP the whole batch's, with identical rows
on every rank). The math is HuggingFace's `Qwen3_5VisionModel`.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    thop_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.activation.gelu_erf import (
    gelu_erf,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.activation.gelu_tanh import (
    gelu_tanh,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.norm.layer_norm import (
    layer_norm,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.vision.patch_embed_conv3d import (
    patch_embed_conv3d,
)
from tensorrt_llm.inputs.multimodal import get_multimodal_item_metadata

from .ops.attention.vision_rope_2d_packed_qk import vision_rope_2d_packed_qk
from .ops.vision.triton_pos_embed_interpolate import triton_pos_embed_interpolate

#: Tower rope base (`vision_config` has no rope field); unrelated to the text 1e7.
VISION_ROPE_THETA = 10000.0

#: LayerNorm epsilon of `norm1`, `norm2`, `merger.norm`: a literal in the reference.
VISION_NORM_EPS = 1e-6

#: Most patch rows per tower call (8192 placeholder tokens x `spatial_merge_size`**2), the vision
#: ops' certified ceiling. Calls are cut between frames, so a request is bounded per frame.
VISION_CALL_MAX_PATCHES = 32768

#: The grids `VisionTower.warm_up` runs: one patch count that is a multiple of 16 (4x4) and one
#: that is not (2x2).
WARM_UP_GRIDS = (((1, 4, 4),), ((1, 2, 2),))

#: `mask_type` 0: bidirectional within a batch segment (causal would depend on raster order).
_VISION_MASK_PADDING = 0

# `thop_attention`'s encoder-only mode, selected by omitting the KV cache pool arguments.
# `update_kv_cache=False` is rejected, so it stays True.
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
    # Rotation is applied before the op (`vision_rope_2d_packed_qk`); rope_* inert.
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

#: `multimodal_data` tensors moved to the device; the grids stay on the host, read sync-free.
DEVICE_PATHS = (
    "image.pixel_values",
    "video.pixel_values_videos",
    "mrope_config.mrope_position_ids",
    "mrope_config.mrope_position_deltas",
)

#: The position-ids axes of the mRoPE layout a media request carries.
_POSITION_AXES = 3


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
    """The tower's weights: patch embedding, position table, blocks and patch merger."""

    patch_w: torch.Tensor
    patch_b: torch.Tensor
    pos: torch.Tensor
    blocks: list
    m_n_w: torch.Tensor
    m_n_b: torch.Tensor
    m_fc1: torch.Tensor
    m_fc1_b: torch.Tensor
    m_fc2: torch.Tensor
    m_fc2_b: torch.Tensor


def _vision_views(w: dict, depth: int) -> _VisionWeights:
    """The weight table's `vis_*` entries (`weights.py`) as the views the ops read."""
    return _VisionWeights(
        # Not transposed: the entry takes the checkpoint's `[out, in, kt, kh, kw]`.
        patch_w=w["vis_patch_w"],
        patch_b=w["vis_patch_b"],
        pos=w["vis_pos"],
        blocks=[
            _VisionBlockWeights(
                n1_w=w[f"l{b}_vis_n1_w"],
                n1_b=w[f"l{b}_vis_n1_b"],
                # Row layout (q heads, k, v) is already thop_attention's packed form.
                qkv=w[f"l{b}_vis_qkv_w"].t(),
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
        m_n_w=w["vis_m_n_w"],
        m_n_b=w["vis_m_n_b"],
        m_fc1=w["vis_m_fc1_w"].t(),
        m_fc1_b=w["vis_m_fc1_b"],
        m_fc2=w["vis_m_fc2_w"].t(),
        m_fc2_b=w["vis_m_fc2_b"],
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


def _merge_order_positions(grid: list[tuple[int, int, int]], merge: int) -> torch.Tensor:
    """Each patch row's `(row, col)` in its frame, int64 `[rows, 2]` on the host, in
    spatial-merge order: whole `merge x merge` blocks row-major, row-major inside a block,
    the order the patch merger's 4-to-1 reshape consumes. A frame group repeats its frame."""
    intra = torch.arange(merge)
    parts = []
    for frames, height, width in grid:
        blocks_h, blocks_w = height // merge, width // merge
        shape = (blocks_h, blocks_w, merge, merge)
        rows = torch.arange(blocks_h)[:, None, None, None] * merge + intra[None, None, :, None]
        cols = torch.arange(blocks_w)[None, :, None, None] * merge + intra[None, None, None, :]
        coords = torch.stack((rows.expand(shape).reshape(-1), cols.expand(shape).reshape(-1)), 1)
        parts.append(coords.repeat(frames, 1) if frames > 1 else coords)
    return parts[0] if len(parts) == 1 else torch.cat(parts, 0)


def _rope_tables(
    grid: list[tuple[int, int, int]], merge: int, theta: float, head_dim: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row fp32 2-D rotary `cos` / `sin`, contiguous `[rows, head_dim // 2]` each.

    Height angles, then width angles; formed in fp64 and rounded once (fp32 math rounds twice)."""
    rope_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float64) / rope_dim))
    pos = _merge_order_positions(grid, merge).to(torch.float64)
    angles = (pos[:, :, None] * inv_freq).flatten(1).to(device=device, dtype=torch.float32)
    return angles.cos().contiguous(), angles.sin().contiguous()


class VisionTower:
    """One request's images or videos through the 27-block ViT and its patch merger.

    Attention is bidirectional per frame (a video's frame group is a frame): one context
    "sequence" per frame for `thop_attention`. The position embedding and the 2-D rope emit in
    spatial-merge order, which the merger's 4-to-1 reshape needs; raster order gives a correctly
    shaped, silently wrong tower. Calls of at most `VISION_CALL_MAX_PATCHES` rows are cut only
    between frames: every op is per frame or per row, and merger blocks stay whole."""

    def __init__(self, w: dict, vision_config) -> None:
        """`w` is the core's loaded weight table; `vision_config` the vision half of the config."""
        vcfg = vision_config
        self.depth = vcfg.depth
        self.hidden = vcfg.hidden_size
        self.heads = vcfg.num_heads
        self.head_dim = self.hidden // self.heads
        self.merge = vcfg.spatial_merge_size
        self.patch = vcfg.patch_size
        # One flat pixel row per patch, `(in_channels, kt, kh, kw)` channel-slowest.
        self.pixel_width = vcfg.in_channels * vcfg.temporal_patch_size * self.patch * self.patch
        # `spatial_merge_size ** 2` patches become one token at the merger.
        self.merge_dim = self.hidden * self.merge * self.merge
        self.grid_side = math.isqrt(vcfg.num_position_embeddings)
        self.dtype = w["embed"].dtype
        self._w = _vision_views(w, self.depth)
        # Persistent scratch `thop_attention` grows in place; not the engine's own workspace.
        self._ws: torch.Tensor | None = None

    def warm_up(self) -> None:
        """Run the tower once per `WARM_UP_GRIDS` grid on zero pixels, so every JIT kernel it
        launches is compiled at load rather than on the first media request. The rope and
        position-embedding kernels take the grid as unspecialized ints; the layer norm's row
        count is specialized only on divisibility by 16, and the two grids cover both."""
        device = self._w.pos.device
        for grid in WARM_UP_GRIDS:
            rows = sum(t * h * w for t, h, w in grid)
            self._run(torch.zeros(rows, self.pixel_width, dtype=self.dtype, device=device), grid)

    def __call__(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Return the `[sum(t*h*w) / merge**2, out_hidden_size]` placeholder rows, in grid order.

        `pixel_values` is `[sum(t*h*w), in_channels*kt*kh*kw]`; `grid_thw` stays on the host."""
        assert pixel_values.dtype == self.dtype, (
            f"the tower runs at the activation dtype {self.dtype}; the runtime handed "
            f"pixel_values as {pixel_values.dtype}. The input processor's `dtype` property is "
            "what decides this, and no cast happens here"
        )
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
            grid.append((frames, height, width))
        assert grid, "grid_thw describes no image or video"
        described = sum(t * h * w for t, h, w in grid)
        assert described == pixel_values.shape[0], (
            f"grid_thw describes {described} patch rows and pixel_values carries "
            f"{pixel_values.shape[0]}"
        )

        outputs = []
        start = 0
        for call in _frame_calls(grid, VISION_CALL_MAX_PATCHES):
            rows = sum(t * h * w for t, h, w in call)
            outputs.append(self._run(pixel_values[start : start + rows], call))
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
        cos, sin = _rope_tables(grid, self.merge, VISION_ROPE_THETA, self.head_dim, dev)

        V = self._w
        pos = [
            triton_pos_embed_interpolate(V.pos, t, h, w, self.grid_side, self.merge, dt)
            for t, h, w in grid
        ]
        h = torch.add(
            patch_embed_conv3d(pixel_values, V.patch_w, V.patch_b),
            pos[0] if len(pos) == 1 else torch.cat(pos, 0),
        )
        attn_out = torch.empty([num_patches, self.hidden], dtype=dt, device=dev)

        for B in V.blocks:
            qkv = cublas_mm(layer_norm(h, B.n1_w, B.n1_b, VISION_NORM_EPS), B.qkv, B.qkv_b)
            # The q|k prefix of each packed row, rotated in place; v is never addressed.
            qkv_heads = torch.reshape(qkv, [num_patches, 3 * self.heads, self.head_dim])
            vision_rope_2d_packed_qk(qkv_heads[:, : 2 * self.heads], cos, sin)
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
            # Non-gated MLP with the tanh GELU (not the merger's erf).
            h = torch.add(h, cublas_mm(gelu_tanh(mlp), B.fc2, B.fc2_b))

        # Patch merger: norm over `hidden`, then the 4-to-1 reshape (rows in spatial-merge order).
        merged = torch.reshape(
            layer_norm(h, V.m_n_w, V.m_n_b, VISION_NORM_EPS), [-1, self.merge_dim]
        )
        # The erf GELU (`nn.GELU()`); within one bf16 ulp of tanh, hence a separate entry.
        return cublas_mm(gelu_erf(cublas_mm(merged, V.m_fc1, V.m_fc1_b)), V.m_fc2, V.m_fc2_b)


def multimodal_embeds(tower: VisionTower | None, kwargs: dict):
    """Run the tower on this batch's media: `(rows, mm_token_indices)`, or None without media.

    `rows` is `[R, hidden]` in packed-batch order, `mm_token_indices` the `[R]` rows they replace.
    The shell calls it: the inherited `DecoderModelForCausalLM.forward` drops `multimodal_params`,
    which covers a subset of the context requests (never zip it positionally).
    `multimodal_runtime` counts rows over a request's media together, in prompt order; a chunk
    runs the tower on only the frames that hold its rows (`_tower_rows`)."""
    mm_params = kwargs.get("multimodal_params") or []
    if not mm_params:
        return None
    rows = []
    for params in mm_params:
        data = getattr(params, "multimodal_data", None) or {}
        if data.get("image") is None and data.get("video") is None:
            continue
        mrope_ids = (data.get("mrope_config") or {}).get("mrope_position_ids")
        assert (
            mrope_ids is not None and mrope_ids.dim() == 3 and mrope_ids.shape[0] == _POSITION_AXES
        ), (
            f"a media request needs mrope_config.mrope_position_ids of shape "
            f"[{_POSITION_AXES}, 1, L]; got {None if mrope_ids is None else tuple(mrope_ids.shape)}"
        )
        runtime = getattr(params, "multimodal_runtime", None)
        if runtime is not None and int(runtime.num_mm_tokens_in_chunk) == 0:
            # Every placeholder row of this request lies in another chunk.
            continue
        assert tower is not None, "post_load_weights builds the vision tower before any step"
        items = _items_in_prompt_order(params, data)
        merge_unit = tower.merge * tower.merge
        total = sum(t * h * w for _, _, (t, h, w) in items) // merge_unit
        start, count = 0, total
        if runtime is not None:
            # Chunked prefill or reused prefix: only this chunk's slice.
            start = int(runtime.num_cached_mm_tokens)
            count = int(runtime.num_mm_tokens_in_chunk)
            assert int(runtime.total_embeds_in_request) == total, (
                f"the executor counted {runtime.total_embeds_in_request} placeholder rows for "
                f"this request and its grids describe {total}; the two are derived from the "
                "same grids and a disagreement means the merger's row count is wrong"
            )
            assert 0 <= start <= start + count <= total, (start, count, total)
        rows.append(_tower_rows(tower, items, start, count))
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
        metadata = params.item_metadata
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


def _tower_rows(tower: VisionTower, items: list, start: int, count: int) -> torch.Tensor:
    """Placeholder rows `[start, start + count)` of one request's `items` (prompt order).

    One tower call over only the frames that hold them: a frame's rows depend on its own patches
    alone (attention is per frame, every other op per row), so a chunk of a long video does not
    rerun the request's other frames. A frame a chunk boundary cuts runs in both chunks. Adjacent
    frames of one pixel tensor stay one view; items of two modalities are copied into one tensor."""
    merge_unit = tower.merge * tower.merge
    end = start + count
    spans: list[list] = []  # [pixel_values, first patch row, end patch row]
    grid: list[tuple[int, int, int]] = []
    first_row = None
    row = 0  # the item's first placeholder row
    for pixel_values, first_patch, (frames, height, width) in items:
        frame_rows = height * width // merge_unit
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
    out = tower(parts[0] if len(parts) == 1 else torch.cat(parts, 0), torch.tensor(grid))
    return out[start - first_row : end - first_row]


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

    mrope_ids = (data.get("mrope_config") or {}).get("mrope_position_ids")
    if (
        mrope_ids is None
        or mrope_ids.dim() != 3
        or mrope_ids.shape[0] != _POSITION_AXES
        or mrope_ids.shape[2] != request.prompt_len
    ):
        raise ValueError(
            f"a media request needs the [{_POSITION_AXES}, 1, {request.prompt_len}] "
            "mrope_config.mrope_position_ids from the input processor, one position per prompt "
            f"token; got {None if mrope_ids is None else tuple(mrope_ids.shape)}"
        )

    if len(grids) > 1:
        metadata = get_multimodal_item_metadata(data)
        _check_item_order(grids, metadata.item_refs if metadata is not None else None)

    cumsum = data.get("multimodal_embed_mask_cumsum")
    if cumsum is not None:
        merge_unit = tower.merge * tower.merge
        rows = sum(t * h * w for grid in grids.values() for t, h, w in grid) // merge_unit
        placeholders = int(cumsum[-1])
        if placeholders != rows:
            raise ValueError(
                f"the prompt holds {placeholders} image/video placeholder tokens and its "
                f"{' and '.join(sorted(grids))} payload produces {rows} embedding rows"
            )


def _media_grid(tower, modality, media, pixel_key, grid_key):
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
    if pixel_values.dtype != tower.dtype:
        raise ValueError(
            f"{pixel_key} is {pixel_values.dtype}; the vision tower runs at {tower.dtype} "
            "(multimodal_config.vision_gpu_normalize ships uint8)"
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
            listed[modality].append(int(index))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"item_refs entry {entry!r} is not an item of this request, which carries "
                f"{sorted(grids)}; entries are (modality, index) pairs"
            ) from error
    for modality, grid in grids.items():
        if sorted(listed[modality]) != list(range(len(grid))):
            raise ValueError(
                f"item_refs lists {modality} items {listed[modality]}; the payload carries "
                f"{len(grid)}"
            )
