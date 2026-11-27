# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The replicated bf16 vision tower for Qwen4-Exp FP8 checkpoints.

Each rank encodes the images and videos in its batch. The 27-block ViT and its 2x2
patch merger return one embedding per placeholder in grid order. The shared mixin
selects the embeddings for the active prefill chunk.
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

from .._vision import (
    VISION_CALL_MAX_PATCHES,
    _frame_calls,
    _item_attention_args,
    _project_items,
    _validate_request,
    _VisionInputs,
)
from . import weights as _weights
from .ops.attention.vision_rope_2d_packed_qk import vision_rope_2d_packed_qk
from .ops.vision.triton_pos_embed_interpolate import triton_pos_embed_interpolate

#: Tower rope base (`vision_config` has no rope field); unrelated to the text 1e7.
VISION_ROPE_THETA = 10000.0

#: LayerNorm epsilon of `norm1`, `norm2`, `merger.norm`: a literal in the reference.
VISION_NORM_EPS = 1e-6
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


class VisionTower(nn.Module, _VisionInputs):
    """A batch of images and videos through the 27-block ViT and its patch merger.

    Attention is bidirectional per frame (a video's frame group is a frame): one context
    "sequence" per frame for `thop_attention`. The position embedding and the 2-D rope emit in
    spatial-merge order, which the merger's 4-to-1 reshape needs; raster order gives a correctly
    shaped, silently wrong tower. Calls of at most `VISION_CALL_MAX_PATCHES` rows are cut only
    between frames: every op is per frame or per row, and merger blocks stay whole."""

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
        if self._w.pos.is_cuda:
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
        """Return the `[sum(t*h*w) / merge**2, out_hidden_size]` placeholder embeddings in grid order.

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

        item_patches = [t * h * w for t, h, w in grid]
        num_patches = pixel_values.shape[0]
        attention_items = _item_attention_args(grid, dev)
        stream = torch.cuda.current_stream(dev)
        if stream not in self._workspaces:
            self._workspaces[stream] = torch.empty(0, dtype=torch.int8, device=dev)
        workspace = self._workspaces[stream]
        cos, sin = _rope_tables(grid, self.merge, VISION_ROPE_THETA, self.head_dim, dev)

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

        # Patch merger: norm over `hidden`, then the 4-to-1 reshape (rows in spatial-merge order).
        merged = torch.reshape(
            layer_norm(h, V.m_n_w, V.m_n_b, VISION_NORM_EPS), [-1, self.merge_dim]
        )
        # The erf GELU (`nn.GELU()`); within one bf16 ulp of tanh, hence a separate entry.
        item_outputs = [n // self.merge**2 for n in item_patches]
        mlp = _project_items(cublas_mm, merged, item_outputs, V.m_fc1, V.m_fc1_b)
        return _project_items(cublas_mm, gelu_erf(mlp), item_outputs, V.m_fc2, V.m_fc2_b)
