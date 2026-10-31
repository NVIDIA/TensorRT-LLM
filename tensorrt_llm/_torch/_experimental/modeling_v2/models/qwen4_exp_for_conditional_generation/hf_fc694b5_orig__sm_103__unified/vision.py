# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The vision tower and the multimodal splice's host bookkeeping.

The checkpoint's 27-block ViT (no KV cache, deepstack or quantization) and its 2x2 patch
merger turn a request's images and videos into one `[rows, hidden]` embedding row per
placeholder token, which the prefill target's `_splice` break writes over the token
embedding. The tower is replicated: every rank holds all of its weights and runs it on the
media of the requests it serves -- under attention TP every rank the whole batch's, and gets
the same rows back, since every op is a deterministic function of its operands; under
attention DP a rank its own requests'.

The shell calls `multimodal_embeds` on every step, before the core's `begin_step`, and the
executor calls `validate_request` on every new request. Neither runs inside a captured
graph: the tower is eager and reads the request's grid on the host.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.gelu_erf import gelu_erf
from tensorrt_llm._torch._experimental.modeling_v2.catalog.activation.gelu_tanh import gelu_tanh
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.vision_rope_2d_packed_qk import (
    vision_rope_2d_packed_qk,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.layer_norm import layer_norm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.vision.patch_embed_conv3d import (
    patch_embed_conv3d,
)
from tensorrt_llm.inputs.multimodal import get_multimodal_item_metadata

from .ops.attention.thop_attention__r2 import thop_attention
from .ops.gemm.cublas_mm__vision import cublas_mm
from .ops.vision.triton_pos_embed_interpolate__tower import triton_pos_embed_interpolate

#: Tower rope base (`vision_config` has no rope field); unrelated to the text 1e7.
VISION_ROPE_THETA = 10000.0

#: LayerNorm epsilon of `norm1`, `norm2`, `merger.norm`: a literal in the reference.
VISION_NORM_EPS = 1e-6

#: The most patch rows one tower call carries, the ceiling the vision entries' cells reach:
#: 8192 placeholder tokens x `spatial_merge_size`**2 patch rows each. `VisionTower` splits a
#: request's frames into calls under it, so the bound a request meets is per frame: one
#: frame's `h * w` patches.
VISION_CALL_MAX_PATCHES = 32768

#: `mask_type` 0 = padding mask: bidirectional within a batch segment. Causal (1) would make
#: a patch depend on raster order.
_VISION_MASK_PADDING = 0

# Per-call constants of the tower's attention: the encoder-only mode of `thop_attention`.
# Omitting `kv_cache_block_offsets`, `host_kv_cache_pool_pointers` and
# `host_kv_cache_pool_mapping` selects that mode. `update_kv_cache=False` is rejected ("KV
# cache update cannot be disabled now"), so it stays True.
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
    # No pool, so no layer to address.
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

#: Per modality, the keys under `multimodal_data[modality]`: the flat patch rows and the
#: host-side `(t, h, w)` grid.
MEDIA_KEYS = {
    "image": ("pixel_values", "image_grid_thw"),
    "video": ("pixel_values_videos", "video_grid_thw"),
}

#: Tensors under `multimodal_data` the executor moves to the device. Everything else stays
#: on the host: the grids are read there, and moved they would cost a synchronization each.
DEVICE_PATHS = (
    "image.pixel_values",
    "video.pixel_values_videos",
    "mrope_config.mrope_position_ids",
    "mrope_config.mrope_position_deltas",
)

#: The position-ids axes of the mRoPE layout a media request carries.
_POSITION_AXES = 3


# --------------------------------------------------------------- weight views ---


class _VisionBlockWeights(NamedTuple):
    """One vision block's weights (`.t()` GEMM views). LayerNorm has a bias, unlike every
    norm in the language tower."""

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


# ------------------------------------------------------------------- the tower ---


def _frame_calls(
    grid: list[tuple[int, int, int]], max_rows: int
) -> list[list[tuple[int, int, int]]]:
    """Cut `grid` (one `(frames, height, width)` row per image or video) into tower calls of
    whole frames, in order, each at most `max_rows` patch rows. A call is its grid rows: the
    frames of one item that fall in it are one `(frames, height, width)` row."""
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
    """The 2-D rotary's per-row fp32 `cos` / `sin`, `[rows, head_dim // 2]` each and
    contiguous: the height angles, then the width angles, as `vision_rope_2d_packed_qk`
    takes them. Built once per tower call and shared by every block.

    Each axis owns half of the `head_dim // 2` rotation pairs. The angles are formed in
    fp64 and rounded once to fp32: the largest is `(side - 1)` radians, and fp32 throughout
    rounds twice there."""
    rope_dim = head_dim // 2
    inv_freq = 1.0 / (theta ** (torch.arange(0, rope_dim, 2, dtype=torch.float64) / rope_dim))
    pos = _merge_order_positions(grid, merge).to(torch.float64)
    angles = (pos[:, :, None] * inv_freq).flatten(1).to(device=device, dtype=torch.float32)
    return angles.cos().contiguous(), angles.sin().contiguous()


class VisionTower:
    """One request's images or videos through the 27-block ViT and its patch merger.

    Not a target: runs only for a context request carrying media, from the shell, before
    the step. Attention is bidirectional per frame (a video's frame group is a frame),
    expressed to `thop_attention` as one context "sequence" per frame; running the rows as
    one sequence is far outside tolerance. The position-embedding interpolation and the 2-D
    rope both emit in spatial-merge order (what the merger's 4-to-1 reshape consumes); a
    raster-order emitter gives a correctly shaped, silently wrong tower.

    The frames run in calls of at most `VISION_CALL_MAX_PATCHES` rows, cut only between
    frames: attention is per frame, the position embedding and rope are per grid row and
    repeat per frame, the norms and MLPs are row-wise, and a frame's `h * w` is a multiple
    of `merge**2`, so a cut keeps every merger block whole. One frame past the bound cannot
    be served; `validate_request` refuses it at admission."""

    def __init__(self, w: dict, vision_config) -> None:
        """`w` is the core's weight table after load; `vision_config` the composite
        config's vision half, which `routing.py` matched against this checkpoint's."""
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
        # The position table is a square grid of this side.
        self.grid_side = math.isqrt(vcfg.num_position_embeddings)
        self.dtype = w["embed"].dtype
        self._w = _vision_views(w, self.depth)
        # Persistent scratch: `thop_attention` grows `workspace` in place; never the
        # engine's own workspace.
        self._ws: torch.Tensor | None = None

    def __call__(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """`pixel_values` is `[sum(t*h*w), in_channels*kt*kh*kw]`, one flat patch per row,
        channel-slowest. `grid_thw` is one `(frames, height, width)` row per image or video
        and stays on the host (no synchronization). Returns
        `[sum(t*h*w) / merge**2, out_hidden_size]`, one row per placeholder, in grid row
        order."""
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
        """One tower call over whole frames: `grid` describes exactly the rows of
        `pixel_values`, at most `VISION_CALL_MAX_PATCHES` of them."""
        dt, dev = pixel_values.dtype, pixel_values.device
        assert pixel_values.is_cuda, "pixel_values must reach the tower on the device"

        # ---- the frame segmentation, host-side ------------------------------
        # One attention segment per frame: a video's frames do not attend to each other.
        seg_lens: list[int] = []
        for frames, height, width in grid:
            seg_lens.extend([height * width] * frames)
        num_patches = pixel_values.shape[0]
        num_segments = len(seg_lens)
        max_seg = max(seg_lens)

        # Batch-state tensors for the encoder-only call; they describe the frames.
        host_lens = torch.tensor(seg_lens, dtype=torch.int32)
        host_types = torch.zeros([num_segments], dtype=torch.int32)
        host_total = torch.tensor([num_patches, 0], dtype=torch.int32)
        dev_lens = host_lens.to(dev)
        if self._ws is None:
            self._ws = torch.empty([0], dtype=torch.int8, device=dev)
        # The rotary's angles, once for all blocks.
        cos, sin = _rope_tables(grid, self.merge, VISION_ROPE_THETA, self.head_dim, dev)

        # ---- patch embedding + absolute position ----------------------------
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

        # ---- the patch merger -----------------------------------------------
        # The norm runs over `hidden` before the 4-to-1 shuffle; the reshape is legal only
        # because both order-bearing steps emitted in spatial-merge order.
        merged = torch.reshape(
            layer_norm(h, V.m_n_w, V.m_n_b, VISION_NORM_EPS), [-1, self.merge_dim]
        )
        # The erf GELU (`nn.GELU()`); within one bf16 ulp of tanh, hence a separate entry.
        return cublas_mm(gelu_erf(cublas_mm(merged, V.m_fc1, V.m_fc1_b)), V.m_fc2, V.m_fc2_b)


# ------------------------------------------------------------ the shell's side ---


def multimodal_embeds(tower: VisionTower | None, kwargs: dict):
    """Run the vision tower for this batch; returns `(rows, mm_token_indices)` (the tower's
    `[R, hidden]` rows in packed-batch order and the `[R]` placeholder rows they replace,
    which the prefill target splices into its embedding), or None when the step carries no
    media.

    Called by the shell, because the inherited `DecoderModelForCausalLM.forward` drops
    `multimodal_params` from `**kwargs` on the no-speculation path (a booting, fluent, blind
    engine). Host-side bookkeeping: `multimodal_params` is a subset of the context requests,
    so never zip it positionally; `multimodal_runtime` (chunked prefill / KV reuse only)
    gives the cached and in-chunk row counts over the request's images and videos together,
    in prompt order; `mm_token_indices` names every placeholder row.

    The tower runs once per modality a request carries, and `in_prompt_order` merges the
    two outputs. Positions are not derived here: every coordinate of a vision span, a
    video's temporal one included, is in the payload's `mrope_config.mrope_position_ids`,
    which the engine slices into `position_ids`."""
    mm_params = kwargs.get("multimodal_params") or []
    if not mm_params:
        return None
    rows = []
    for params in mm_params:
        data = getattr(params, "multimodal_data", None) or {}
        image, video = data.get("image"), data.get("video")
        if image is None and video is None:
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
        towers = {}
        if image is not None:
            towers["image"] = run_tower(tower, image, *MEDIA_KEYS["image"])
        if video is not None:
            towers["video"] = run_tower(tower, video, *MEDIA_KEYS["video"])
        embeds = in_prompt_order(tower.merge, params, towers)
        if runtime is not None:
            # Chunked prefill or reused prefix: splice only this chunk's slice.
            start = int(runtime.num_cached_mm_tokens)
            count = int(runtime.num_mm_tokens_in_chunk)
            total = embeds.shape[0]
            assert int(runtime.total_embeds_in_request) == total, (
                f"the executor counted {runtime.total_embeds_in_request} placeholder rows for "
                f"this request and the tower produced {total} over its {sorted(towers)}; the "
                "two are derived from the same grids and a disagreement means the merger's "
                "row count is wrong"
            )
            assert 0 <= start <= start + count <= total, (start, count, total)
            embeds = embeds[start : start + count]
        rows.append(embeds)
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


def run_tower(tower, media: dict, pixel_key: str, grid_key: str):
    """The tower over one modality's payload: `(rows, grid_thw)`, the rows one per
    placeholder, item by item in the order the processor emitted them."""
    pixel_values, grid_thw = media.get(pixel_key), media.get(grid_key)
    assert pixel_values is not None and grid_thw is not None, (
        f"the multimodal payload carries {sorted(media)}; this target reads {pixel_key!r} and "
        f"{grid_key!r} from it"
    )
    assert not grid_thw.is_cuda, f"{grid_key} must stay on the host; see DEVICE_PATHS"
    return tower(pixel_values, grid_thw), grid_thw


def in_prompt_order(merge: int, params, towers: dict) -> torch.Tensor:
    """One request's tower rows in the order its placeholders appear.

    `towers` maps each modality the request carries to `(rows, grid_thw)`; each grid row is
    one item of `t*h*w / merge**2` rows. One modality is already in prompt order. Two are
    reassembled item by item from `item_metadata.item_refs` (`(modality, index)` per item in
    prompt order, `index` counting that modality's items); without it there is no host-side
    way to tell which placeholder run is an image and which a video."""
    if len(towers) == 1:
        return next(iter(towers.values()))[0]
    metadata = params.item_metadata
    order = metadata.item_refs if metadata is not None else None
    assert order, (
        f"a request carrying {sorted(towers)} needs item metadata to splice them in prompt "
        "order; the payload holds each modality's rows apart"
    )
    merge_unit = merge * merge
    items = {}
    for modality, (embeds, grid_thw) in towers.items():
        counts = [t * h * w // merge_unit for t, h, w in grid_thw.tolist()]
        assert sum(counts) == embeds.shape[0], (
            f"the {modality} grid describes {sum(counts)} rows and the tower produced "
            f"{embeds.shape[0]}"
        )
        items[modality] = torch.split(embeds, counts, 0)
    listed = {modality: [] for modality in items}
    for modality, index in order:
        assert modality in items, (
            f"item_refs lists a {modality!r} item; this request carries {sorted(items)}"
        )
        listed[modality].append(int(index))
    for modality, parts in items.items():
        assert sorted(listed[modality]) == list(range(len(parts))), (
            f"item_refs lists {modality} items {listed[modality]}; the payload carries {len(parts)}"
        )
    return torch.cat([items[modality][int(index)] for modality, index in order], 0)


# ---------------------------------------------------------------- admission ---


def validate_request(tower, request) -> None:
    """Admission check, run by the executor on every new request. A media request the
    forward would assert on fails here as that request's own error; an assert inside the
    step ends the executor under the overlap scheduler. Reads only the request's host-side
    payload, and a request without media returns at once. `tower` supplies the geometry
    (`merge`, `patch`, `pixel_width`).

    Refused: a frame past `VISION_CALL_MAX_PATCHES` patches (the one bound the tower keeps),
    a malformed grid or pixel payload, media whose rows disagree with the prompt's
    placeholder count, missing or mis-sized `mrope_position_ids`, and a request with both
    images and videos whose item metadata is missing or does not list its items."""
    data = request.py_multimodal_data
    if not data:
        return
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
