# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The vision tower and the multimodal splice's host bookkeeping.

The replicated 27-block ViT and its patch merger return one multimodal embedding per
image or video placeholder. Each rank encodes the media for its requests. The mixin
caches the resulting embeddings and selects the active chunk. The language model writes
those embeddings at the placeholder positions during prefill.

The tower validates the request before scheduling and runs eagerly using the host grids.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn

from tensorrt_llm._torch.modeling_v2.catalog.attention.thop_attention import thop_attention
from tensorrt_llm._torch.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.activation.gelu_erf import gelu_erf
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.activation.gelu_tanh import gelu_tanh
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.norm.layer_norm import layer_norm
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.vision.patch_embed_conv3d import (
    patch_embed_conv3d,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default.ops.attention.vision_rope_2d_packed_qk import (  # noqa: E501
    vision_rope_2d_packed_qk,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default.ops.vision.triton_pos_embed_interpolate import (  # noqa: E501
    triton_pos_embed_interpolate,
)

from .._vision import (
    VISION_CALL_MAX_PATCHES,
    _frame_calls,
    _item_attention_args,
    _project_items,
    _validate_request,
    _VisionInputs,
)
from . import weights as _weights

#: Tower rope base (`vision_config` has no rope field); unrelated to the text 1e7.
VISION_ROPE_THETA = 10000.0

#: LayerNorm epsilon of `norm1`, `norm2`, `merger.norm`: a literal in the reference.
VISION_NORM_EPS = 1e-6
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

#: The grids `VisionTower.warm_up` runs: one call whose patch count is a multiple of 16 (4x4)
#: and one whose count is not (2x2), the two variants Triton compiles a row count into.
WARM_UP_GRIDS = (((1, 4, 4),), ((1, 2, 2),))


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


class VisionTower(nn.Module, _VisionInputs):
    """A batch of images and videos through the 27-block ViT and its patch merger.

    Not a target: the vision-language model's encoder, run only for a context request carrying
    media, before the step. Attention is bidirectional per frame (a video's frame group is a
    frame), expressed to `thop_attention` as one context sequence per frame. Position embedding
    interpolation and 2-D RoPE both emit in the spatial-merge order the merger consumes.

    The frames run in calls of at most `VISION_CALL_MAX_PATCHES` rows, cut only between
    frames: attention is per frame, the position embedding and rope are per grid row and
    repeat per frame, the norms and MLPs are row-wise, and a frame's `h * w` is a multiple
    of `merge**2`, so a cut keeps every merger block whole. One frame past the bound cannot
    be served; `validate_request` refuses it at admission.

    The tower declares and loads its own weights (`weights.VISION_WEIGHTS`), so a model
    built without it (`disable_mm_encoder`) allocates no vision weight."""

    def __init__(self, vision_config, dtype: torch.dtype) -> None:
        """`vision_config` is the composite config's vision half, which `check_deployment`
        matched against this checkpoint's; `dtype` the activation dtype the tower runs at."""
        super().__init__()
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
        self.dtype = dtype
        # Every weight, declared here (meta-init intercepts `torch.empty`) and filled by
        # `load_weights`; the views the ops read are built after load.
        self.dims = _weights.vision_dims(vcfg, dtype)
        self.w = _weights.VISION_WEIGHTS.declare(self.dims)
        self._w: _VisionWeights | None = None
        # Persistent scratch: `thop_attention` grows `workspace` in place; never the
        # engine's own workspace.
        self._workspaces: dict[torch.cuda.Stream, torch.Tensor] = {}

    def load_weights(self, weights) -> None:
        """Fill the tower's weights from the checkpoint's `model.visual.` keys."""
        _weights.VISION_WEIGHTS.load(self, weights)

    def post_load_weights(self) -> None:
        """Build the views the ops read, then warm the kernels up; once, though the
        engine's post-load walk may reach this module more than once."""
        if self._w is not None:
            return
        self._w = _vision_views(self.w, self.depth)
        self.warm_up()

    def warm_up(self) -> None:
        """Run the tower once per `WARM_UP_GRIDS` grid on zero pixels, so every JIT kernel it
        launches is compiled at load rather than on the first media request. The rope and
        position-embedding kernels take the grid as unspecialized ints; the layer norm's row
        count is specialized only on divisibility by 16, and the two grids cover both."""
        device = self._w.pos.device
        for grid in WARM_UP_GRIDS:
            rows = sum(t * h * w for t, h, w in grid)
            self._run(torch.zeros(rows, self.pixel_width, dtype=self.dtype, device=device), grid)

    def validate_request(self, request) -> None:
        _validate_request(self, request, VISION_CALL_MAX_PATCHES)

    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """`pixel_values` is `[sum(t*h*w), in_channels*kt*kh*kw]`, one flat patch per row,
        channel-slowest. `grid_thw` is one `(frames, height, width)` row per image or video
        and stays on the host (no synchronization). Returns
        `[sum(t*h*w) / merge**2, out_hidden_size]`, one embedding per placeholder in grid order."""
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
        item_patches = [t * h * w for t, h, w in grid]
        num_patches = pixel_values.shape[0]
        attention_items = _item_attention_args(grid, dev)
        stream = torch.cuda.current_stream(dev)
        if stream not in self._workspaces:
            self._workspaces[stream] = torch.empty(0, dtype=torch.int8, device=dev)
        workspace = self._workspaces[stream]
        # The rotary's angles, once for all blocks.
        cos, sin = _rope_tables(grid, self.merge, VISION_ROPE_THETA, self.head_dim, dev)

        # ---- patch embedding + absolute position ----------------------------
        V = self._w
        pos = [
            triton_pos_embed_interpolate(V.pos, t, h, w, self.grid_side, self.merge, dt)
            for t, h, w in grid
        ]
        h = torch.add(
            _project_items(patch_embed_conv3d, pixel_values, item_patches, V.patch_w, V.patch_b),
            pos[0] if len(pos) == 1 else torch.cat(pos, 0),
        )
        attn_out = torch.empty([num_patches, self.hidden], dtype=dt, device=dev)

        for B in V.blocks:
            qkv = _project_items(
                cublas_mm,
                layer_norm(h, B.n1_w, B.n1_b, VISION_NORM_EPS),
                item_patches,
                B.qkv,
                B.qkv_b,
            )
            # The q|k prefix of each packed row, rotated in place; v is never addressed.
            qkv_heads = torch.reshape(qkv, [num_patches, 3 * self.heads, self.head_dim])
            vision_rope_2d_packed_qk(qkv_heads[:, : 2 * self.heads], cos, sin)
            for span, args in attention_items:
                thop_attention(
                    q=qkv[span],
                    output=attn_out[span],
                    workspace=workspace,
                    num_heads=self.heads,
                    num_kv_heads=self.heads,
                    head_size=self.head_dim,
                    **args,
                    **_VISION_CALL_CONSTANTS,
                )
            h = torch.add(h, _project_items(cublas_mm, attn_out, item_patches, B.proj, B.proj_b))
            mlp = _project_items(
                cublas_mm,
                layer_norm(h, B.n2_w, B.n2_b, VISION_NORM_EPS),
                item_patches,
                B.fc1,
                B.fc1_b,
            )
            # Non-gated MLP with the tanh GELU (not the merger's erf).
            h = torch.add(
                h, _project_items(cublas_mm, gelu_tanh(mlp), item_patches, B.fc2, B.fc2_b)
            )

        # ---- the patch merger -----------------------------------------------
        # The norm runs over `hidden` before the 4-to-1 shuffle; the reshape is legal only
        # because both order-bearing steps emitted in spatial-merge order.
        merged = torch.reshape(
            layer_norm(h, V.m_n_w, V.m_n_b, VISION_NORM_EPS), [-1, self.merge_dim]
        )
        # The erf GELU (`nn.GELU()`); within one bf16 ulp of tanh, hence a separate entry.
        item_outputs = [n // self.merge**2 for n in item_patches]
        mlp = _project_items(cublas_mm, merged, item_outputs, V.m_fc1, V.m_fc1_b)
        return _project_items(cublas_mm, gelu_erf(mlp), item_outputs, V.m_fc2, V.m_fc2_b)
