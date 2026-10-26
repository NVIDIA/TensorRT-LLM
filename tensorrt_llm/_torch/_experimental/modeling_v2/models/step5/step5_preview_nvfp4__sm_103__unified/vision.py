# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Step-5 perception encoder and its projector: image tiles in, the rows that replace
the prompt's `<im_patch>` embeddings out.

One tile -- a 728 px overview or a 504 px crop -- is patch-embedded (conv 14/14, no
bias), its absolute position embedding added (the overview's 52 x 52 table, or the same
table resampled to a crop's 36 x 36 grid at load), then `ln_pre`, then 47 pre-norm
blocks:

    x = x + ls_1 * out_proj(attention(rope2d(in_proj(ln_1(x)))))
    x = x + ls_2 * c_proj(quick_gelu(c_fc(ln_2(x))))

with 16 bidirectional heads of 96 attending within the tile only, q and k turned by
the 2-D rope (first half by the patch's column, second by its row, theta 1e4). The
tile's grid then passes the two 3x3 stride-2 downsamplers (1536 -> 3072 -> 6144, no
activation between) and the bias-free projector (6144 -> 4096): 169 rows per overview
(13 x 13), 81 per crop (9 x 9). An image's rows are its crops' in order, then its
overview's -- the order of its `<im_patch>` run in the prompt.

**Attention.** No kernel takes a 96-wide head, so q, k and v are written into a buffer
whose heads are zero-padded to 128 and attended with `q_scaling` sqrt(96 / 128), which
restores the 96^-0.5 softmax scale; the output's pad columns come back zero and are
dropped. The call is `thop_attention`'s no-KV-cache mode, one context sequence per tile
under the padding mask, so tiles packed into one call never see each other.

**Calls.** The tiles run in calls of at most `rows_per_call` patch rows, cut only between
tiles: attention is per tile and everything else is row-wise or per tile, so a cut
changes no row. The bound is the engine's `max_num_tokens` (the encoder token budget the
engine profiles the tower at, `get_dummy_mm_data`), capped at the largest call the
vision entries are certified at, and never below one overview. Within a call the crop
tiles run first, then the overviews; the output is put back in prompt order.

**Precision.** bf16 throughout, as StepFun's reference runs its tower: the residual
adds and the layer scales in bf16, each GEMM accumulating in fp32. Two places differ
from that reference by construction: `quick_gelu` evaluates in fp32 and rounds once,
and the rope turns by exact angles where the reference rounds its angle table to bf16
first (`vision_rope_2d_interleaved`'s note).

Eager only: it runs on prefill steps that carry images, never inside a CUDA graph, and
reads only host-side sizes (`num_patches`, tensor shapes), so it never synchronizes.
"""

from __future__ import annotations

import math
from typing import NamedTuple, Sequence

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (
    thop_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch._experimental.modeling_v2.catalog.norm.layer_norm import layer_norm

from .ops.activation.quick_gelu import quick_gelu
from .ops.attention.vision_rope_2d_interleaved import vision_rope_2d_interleaved
from .ops.vision.conv2d import conv2d

#: The reference encoder's defaults; `vision_config` states neither.
_LN_EPS = 1e-5
_ROPE_THETA = 1e4

#: The downsamplers: 3x3 kernels, stride 2, padding 1.
_DOWNSAMPLE_STRIDE = 2
_DOWNSAMPLE_PADDING = 1

#: The head width the attention kernels take for the tower's 96-wide heads.
_PADDED_HEAD = 128

#: The largest call the vision entries are certified at: an overview and eight crops,
#: 2704 + 8 * 1296 patch rows (`layer_norm`, `cublas_mm`, `quick_gelu`,
#: `vision_rope_2d_interleaved` and `thop_attention` each carry a cell there).
_CALL_MAX_ROWS = 13072  # capacity: fixed: the certification bound of the vision entries

#: `mask_type` 0, the padding mask: bidirectional within one context sequence (a tile).
_MASK_PADDING = 0

# Per-call constants of the tower's attention: `thop_attention`'s no-KV-cache mode.
# Leaving `kv_cache_block_offsets`, `host_kv_cache_pool_pointers` and
# `host_kv_cache_pool_mapping` None selects it; `update_kv_cache=False` is rejected ("KV
# cache update cannot be disabled now"), so it stays True. Rope is applied outside, so the
# rope_* values are inert.
_ATTENTION_CONSTANTS = dict(
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
    mask_type=_MASK_PADDING,
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
    spec_decoding_position_offsets_for_cpp=None,
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


class _Block(NamedTuple):
    """One encoder block's weights; GEMM operands are `.t()` views."""

    ln1_w: torch.Tensor
    ln1_b: torch.Tensor
    qkv: torch.Tensor
    qkv_b: torch.Tensor
    o: torch.Tensor
    o_b: torch.Tensor
    ls1: torch.Tensor
    ln2_w: torch.Tensor
    ln2_b: torch.Tensor
    fc1: torch.Tensor
    fc1_b: torch.Tensor
    fc2: torch.Tensor
    fc2_b: torch.Tensor
    ls2: torch.Tensor


class _Tile(NamedTuple):
    """One tile of the output order: a crop (`overview` False) or an overview, and its
    index into `patch_pixel_values` or `pixel_values`."""

    overview: bool
    index: int


class VisionTower:
    """The tower over the core's realized weights. Not an `nn.Module`: its weights live in
    the core's `ParameterDict`, declared once by the weight table."""

    def __init__(self, core) -> None:
        d = core.dims
        w = core.w
        self.dims = d
        self.patch = w["v_patch"]
        self.pos = {d.v_grid: w["v_pos"], d.v_crop_grid: w["v_pos_crop"]}
        self.ln_pre = (w["v_ln_pre_w"], w["v_ln_pre_b"])
        self.blocks = [
            _Block(
                ln1_w=w[f"v{i}_ln1_w"],
                ln1_b=w[f"v{i}_ln1_b"],
                qkv=w[f"v{i}_qkv"].t(),
                qkv_b=w[f"v{i}_qkv_b"],
                o=w[f"v{i}_o"].t(),
                o_b=w[f"v{i}_o_b"],
                ls1=w[f"v{i}_ls1"],
                ln2_w=w[f"v{i}_ln2_w"],
                ln2_b=w[f"v{i}_ln2_b"],
                fc1=w[f"v{i}_fc1"].t(),
                fc1_b=w[f"v{i}_fc1_b"],
                fc2=w[f"v{i}_fc2"].t(),
                fc2_b=w[f"v{i}_fc2_b"],
                ls2=w[f"v{i}_ls2"],
            )
            for i in range(d.v_layers)
        ]
        self.down = ((w["v_down1"], w["v_down1_b"]), (w["v_down2"], w["v_down2_b"]))
        self.proj = w["v_proj"].t()
        self.q_scaling = math.sqrt(d.v_head_dim / _PADDED_HEAD)
        self.tile_rows = {True: d.v_grid**2, False: d.v_crop_grid**2}
        # Downsampled rows per tile: each stride-2 conv takes a grid of g to ceil(g / 2).
        self.out_rows = {
            overview: _downsampled(grid) ** 2
            for overview, grid in ((True, d.v_grid), (False, d.v_crop_grid))
        }
        self.rows_per_call = max(
            self.tile_rows[True], min(_CALL_MAX_ROWS, int(core.model_config.max_num_tokens))
        )
        self._positions: dict = {}
        # Persistent scratch for `thop_attention`, which grows it in place; never the
        # engine's own workspace.
        self._ws: torch.Tensor | None = None

    def __call__(
        self,
        pixel_values: torch.Tensor,
        patch_pixel_values: torch.Tensor | None,
        num_patches: Sequence[int],
    ) -> torch.Tensor:
        """`pixel_values` `[N, 3, 728, 728]` (one overview per image),
        `patch_pixel_values` `[sum(num_patches), 3, 504, 504]` or None, `num_patches`
        the crop count of each image (host ints). Returns `[R, hidden]`: per image, its
        crops' rows then its overview's."""
        tiles: list[_Tile] = []
        crop = 0
        for image, count in enumerate(num_patches):
            tiles.extend(_Tile(False, crop + j) for j in range(int(count)))
            tiles.append(_Tile(True, image))
            crop += int(count)
        out = []
        for call in self._calls(tiles):
            out.extend(self._run(call, pixel_values, patch_pixel_values))
        return out[0] if len(out) == 1 else torch.cat(out, 0)

    def _calls(self, tiles: list[_Tile]) -> list[list[_Tile]]:
        """`tiles` cut, in order, into runs of at most `rows_per_call` patch rows."""
        calls: list[list[_Tile]] = [[]]
        rows = 0
        for tile in tiles:
            n = self.tile_rows[tile.overview]
            if calls[-1] and rows + n > self.rows_per_call:
                calls.append([])
                rows = 0
            calls[-1].append(tile)
            rows += n
        return calls

    def _grid_positions(self, grid: int, device) -> torch.Tensor:
        """`[grid * grid, 2]` int64 `(row, col)` of each patch, row-major."""
        if grid not in self._positions:
            axis = torch.arange(grid, dtype=torch.int64, device=device)
            self._positions[grid] = torch.stack(
                (axis.repeat_interleave(grid), axis.repeat(grid)), dim=1
            )
        return self._positions[grid]

    def _embed(self, pixels: torch.Tensor, grid: int) -> torch.Tensor:
        """Patch embedding plus the grid's position embedding: `[n * grid * grid, width]`."""
        d = self.dims
        patches = conv2d(pixels, self.patch, None, d.v_patch, 0)
        rows = patches.flatten(2).transpose(1, 2) + self.pos[grid]
        return rows.reshape(-1, d.v_width)

    def _run(self, call: list[_Tile], pixel_values, patch_pixel_values) -> list[torch.Tensor]:
        """One tower call over whole tiles; returns each tile's projected rows in `call`'s
        order."""
        d = self.dims
        crops = [t.index for t in call if not t.overview]
        overviews = [t.index for t in call if t.overview]
        # Each kind's indices are consecutive: the tiles are in prompt order.
        parts, seg_lens, positions = [], [], []
        for overview, index, source, grid in (
            (False, crops, patch_pixel_values, d.v_crop_grid),
            (True, overviews, pixel_values, d.v_grid),
        ):
            if not index:
                continue
            parts.append(self._embed(source[index[0] : index[-1] + 1], grid))
            seg_lens.extend([grid * grid] * len(index))
            positions.append(self._grid_positions(grid, source.device).repeat(len(index), 1))
        h = parts[0] if len(parts) == 1 else torch.cat(parts, 0)
        pos = positions[0] if len(positions) == 1 else torch.cat(positions, 0)
        h = self._blocks(layer_norm(h, *self.ln_pre, _LN_EPS), pos, seg_lens)

        projected = {}
        split = len(crops) * d.v_crop_grid**2
        for overview, count, grid, rows in (
            (False, len(crops), d.v_crop_grid, h[:split]),
            (True, len(overviews), d.v_grid, h[split:]),
        ):
            if count:
                projected[overview] = list(
                    torch.split(self._project(rows, count, grid), self.out_rows[overview], 0)
                )
        return [projected[t.overview].pop(0) for t in call]

    def _blocks(self, h: torch.Tensor, pos: torch.Tensor, seg_lens: list[int]) -> torch.Tensor:
        """The 47 encoder blocks over packed tiles `h` `[T, width]`; one attention context
        per entry of `seg_lens`."""
        d = self.dims
        rows, dev = h.shape[0], h.device
        heads, hd = d.v_heads, d.v_head_dim
        host_lens = torch.tensor(seg_lens, dtype=torch.int32)
        dev_lens = host_lens.to(dev, non_blocking=True)
        host_types = torch.zeros(len(seg_lens), dtype=torch.int32)
        host_total = torch.tensor([rows, 0], dtype=torch.int32)
        longest = max(seg_lens)
        if self._ws is None:
            self._ws = torch.empty(0, dtype=torch.int8, device=dev)
        # q | k | v heads zero-padded to `_PADDED_HEAD`; the pad lanes are written once.
        qkv_pad = torch.zeros(rows, 3 * heads * _PADDED_HEAD, dtype=h.dtype, device=dev)
        padded = qkv_pad.view(rows, 3 * heads, _PADDED_HEAD)
        attn = torch.empty(rows, heads * _PADDED_HEAD, dtype=h.dtype, device=dev)
        for B in self.blocks:
            qkv = cublas_mm(layer_norm(h, B.ln1_w, B.ln1_b, _LN_EPS), B.qkv, B.qkv_b)
            split = qkv.view(rows, 3 * heads, hd)
            q, k = vision_rope_2d_interleaved(
                split[:, :heads], split[:, heads : 2 * heads], pos, _ROPE_THETA
            )
            padded[:, :heads, :hd].copy_(q)
            padded[:, heads : 2 * heads, :hd].copy_(k)
            padded[:, 2 * heads :, :hd].copy_(split[:, 2 * heads :])
            attn = thop_attention(
                q=qkv_pad,
                output=attn,
                workspace_=self._ws,
                sequence_length=dev_lens,
                host_past_key_value_lengths=host_lens,
                host_total_kv_lens=host_total,
                context_lengths=dev_lens,
                host_context_lengths=host_lens,
                host_request_types=host_types,
                max_num_requests=len(seg_lens),
                max_context_length=longest,
                max_seq_len=longest,
                attention_window_size=longest,
                num_contexts=len(seg_lens),
                num_ctx_tokens=rows,
                num_heads=heads,
                num_kv_heads=heads,
                head_size=_PADDED_HEAD,
                q_scaling=self.q_scaling,
                **_ATTENTION_CONSTANTS,
            )
            o_in = attn.view(rows, heads, _PADDED_HEAD)[:, :, :hd].reshape(rows, heads * hd)
            h = h + cublas_mm(o_in, B.o, B.o_b) * B.ls1
            mlp = cublas_mm(layer_norm(h, B.ln2_w, B.ln2_b, _LN_EPS), B.fc1, B.fc1_b)
            h = h + cublas_mm(quick_gelu(mlp), B.fc2, B.fc2_b) * B.ls2
        return h

    def _project(self, rows: torch.Tensor, count: int, grid: int) -> torch.Tensor:
        """`count` tiles' encoder rows `[count * grid * grid, width]` through the two
        downsamplers and the projector: `[count * g * g, hidden]`, g = grid after both."""
        d = self.dims
        x = rows.view(count, grid, grid, d.v_width).permute(0, 3, 1, 2)
        for weight, bias in self.down:
            x = conv2d(x, weight, bias, _DOWNSAMPLE_STRIDE, _DOWNSAMPLE_PADDING)
        features = x.flatten(2).transpose(1, 2).reshape(-1, x.shape[1])
        return cublas_mm(features, self.proj)


def _downsampled(grid: int) -> int:
    """The side of a grid after both 3x3 stride-2 padding-1 downsamplers."""
    for _ in range(2):
        grid = (grid + 2 * _DOWNSAMPLE_PADDING - 3) // _DOWNSAMPLE_STRIDE + 1
    return grid
