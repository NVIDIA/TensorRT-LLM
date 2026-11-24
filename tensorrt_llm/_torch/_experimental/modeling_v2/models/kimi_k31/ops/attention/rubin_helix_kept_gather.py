# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kept-entry remap and NVFP4 -> E4M3 gather of the helix DSA generation rows (Rubin).

One launch over the kept entries of every row writes what the sparse MLA reads: the compact index
of each kept column and, for a valid entry, the 576 E4M3 bytes of its NVFP4 latent row. These are
the bytes ``convert_req_index_to_global`` + ``nvfp4_mla_kv_cache_gather`` write for those entries.

Only an NVFP4 latent cache needs the gather; the sparse MLA reads an FP8 latent cache directly. So
the default target's helix8 generation with the NVFP4 cache is the only caller.

``__call__`` calls the Triton launcher of ``kernels/attention/rubin_helix_kept_gather.py``. No torch
op is registered. ``nvfp4_pool_views``, the launcher module's, gives the pool views the entry
reads from the addresses in the KV cache manager's host pointer table.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.attention.rubin_helix_kept_gather import helix_kept_gather as _launch
from ...kernels.attention.rubin_helix_kept_gather import nvfp4_pool_views

#: Latent row width (512 NoPE + 64 RoPE values) and the RoPE part.
HEAD_DIM = 576
ROPE_DIM = 64

#: E2M1 magnitudes by code (bit 3 is the sign).
_E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _spec(rows, num_req, local_len, lens, residual, holes=False) -> dict:
    """``lens``: how the row lengths are drawn (``typical``: 180-340 kept entries, a CP8 share of
    the 2048-wide Top-K; ``edges``: 0..2048 around the 16- and 512-column boundaries; ``full``)."""
    return dict(
        rows=rows,
        num_req=num_req,
        local_len=local_len,
        lens=lens,
        residual_dim=residual,
        holes=holes,
        top_k=2048,
        tokens_per_block=64,
    )


def _codes(data: torch.Tensor) -> torch.Tensor:
    """uint8 [..., n] -> int64 [..., 2n] E2M1 codes, low nibble first."""
    d = data.to(torch.int64)
    return torch.stack((d & 15, d >> 4), dim=-1).flatten(-2)


def _e2m1(codes: torch.Tensor) -> torch.Tensor:
    table = torch.tensor(_E2M1 + tuple(-v for v in _E2M1), dtype=torch.float32)
    return table.to(codes.device)[codes]


def _e4m3(x: torch.Tensor) -> torch.Tensor:
    """Round to nearest even, saturating to +-448 (the conversions' ``.satfinite``)."""
    return x.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)


def _one_level(codes: torch.Tensor, scale_bytes: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """e4m3(fp16(e2m1 * fp16(fp32(block_scale * G)))): 16 values per scale byte."""
    s16 = (scale_bytes.view(torch.float8_e4m3fn).float() * g).to(torch.float16)
    s16 = s16.repeat_interleave(16, dim=-1).float()
    return _e4m3((_e2m1(codes) * s16).to(torch.float16).float())


def dequantize_rows(data: torch.Tensor, scale: torch.Tensor, g: torch.Tensor, residual_dim: int):
    """E4M3 [n, 576] of NVFP4 latent rows: data uint8 [n, (576 + r) / 2], scale uint8 [n, (576 + r) / 16].

    ``residual_dim`` 0: one level for all 576 values. 64: NoPE one level; the RoPE part in 4 groups
    of 16, main and residual codes each with their own block scale, combined as
    ``fma(main, A, residual * B)`` in fp32 (A, B = block scale * G), the order the kernels use.
    """
    if residual_dim == 0:
        return _one_level(_codes(data), scale, g)
    nope = _one_level(_codes(data[:, :256]), scale[:, :32], g)
    rope = data[:, 256:].reshape(-1, 4, 2, 8)  # group, (main, residual), bytes
    main, res = _e2m1(_codes(rope[:, :, 0])), _e2m1(_codes(rope[:, :, 1]))
    sm = scale[:, 32:40:2].view(torch.float8_e4m3fn).float() * g
    sr = scale[:, 33:40:2].view(torch.float8_e4m3fn).float() * g
    tail = res * sr[..., None]  # fp32 product, rounded once
    # main * A has <= 26 significant bits and the two terms are within 2^22 of each other, so the
    # fp64 sum is exact and its one rounding to fp32 is the fma's.
    fused = (main.double() * sm[..., None].double() + tail.double()).float()
    rope = _e4m3(fused).reshape(-1, ROPE_DIM)
    return torch.cat((nope.view(torch.uint8), rope.view(torch.uint8)), dim=-1).view(nope.dtype)


class RubinHelixKeptGather(OpWrapper):
    """For row ``r`` and column ``c < row_lens[r]`` (clamped to [0, top_k]), with ``t =
    topk_indices[r, c]``, ``b = block_table[req_idx[r], t // P]`` (P = ``tokens_per_block``) and
    ``g = b * stride_factor + t % P + layer_offset * P``: the entry is valid when ``t >= 0``, the
    block is inside the table, ``b >= 0`` and ``0 <= g < num_pool_tokens``. Then
    ``compact_indices[r, c] = r * top_k + c`` and ``scratch[r, c]`` = the E4M3 row of pool token
    ``g``. Otherwise, and for every column at or past the row length, ``compact_indices[r, c] = -1``
    and ``scratch[r, c]`` is not written.

    ``data_pool`` / ``scale_pool``: the layer's NVFP4 pools as flat uint8 (``nvfp4_pool_views``),
    (576 + r) / 2 and (576 + r) / 16 bytes per token for ``residual_dim`` r in (64, 0);
    ``dequant_scale``: fp32 [1] global scale G; ``scratch``: E4M3 [>= rows, top_k, 576];
    ``compact_indices``: int32 [rows, top_k]. Returns ``(scratch, compact_indices)``.
    """

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="helix8 GEN b8 k7: 64 rows of 180-340 kept entries, one-level layout (residual 0)",
            spec=_spec(64, 8, 16384, "typical", 0),
        ),
        Cell(
            why="helix8 GEN b16 k5: 96 rows, residual 0 (the hybrid KDA / DSA cache manager's)",
            spec=_spec(96, 16, 16384, "typical", 0),
        ),
        Cell(
            why="helix8 GEN b24 k5: 144 rows; -1 holes, padded table entries, out-of-pool pages",
            spec=_spec(144, 24, 4096, "typical", 0, holes=True),
        ),
        Cell(
            why="row lengths 0..2048 at the 16- and 512-column edges of the strips",
            spec=_spec(64, 8, 16384, "edges", 0, holes=True),
        ),
        Cell(
            why="full rows: all 2048 columns kept, contexts shorter than the row",
            spec=_spec(40, 5, 3000, "full", 0),
        ),
        Cell(
            why="two-level RoPE layout (residual 64, DSACacheManagerV2), helix8 b16 k5 rows",
            spec=_spec(96, 16, 16384, "typical", 64, holes=True),
        ),
    )
    note = """
    The conversions are inline PTX calibrated on SM107 against the C++ gather over every input.

    Launch shape: the launcher's BLOCK_E (32) and 4 warps. CUDA-graph safe: a static grid;
    row lengths, indices and G are read on the device.
    """

    def __call__(
        self,
        topk_indices: torch.Tensor,
        row_lens: torch.Tensor,
        req_idx: torch.Tensor,
        block_table: torch.Tensor,
        tokens_per_block: int,
        stride_factor: int,
        layer_offset: int,
        data_pool: torch.Tensor,
        scale_pool: torch.Tensor,
        dequant_scale: torch.Tensor,
        num_pool_tokens: int,
        scratch: torch.Tensor,
        compact_indices: torch.Tensor,
        residual_dim: int,
    ):
        _launch(
            topk_indices,
            row_lens,
            req_idx,
            block_table,
            tokens_per_block,
            stride_factor,
            layer_offset,
            data_pool,
            scale_pool,
            dequant_scale,
            num_pool_tokens,
            scratch,
            compact_indices,
            residual_dim=residual_dim,
        )
        return scratch, compact_indices

    def reference(
        self,
        topk_indices: torch.Tensor,
        row_lens: torch.Tensor,
        req_idx: torch.Tensor,
        block_table: torch.Tensor,
        tokens_per_block: int,
        stride_factor: int,
        layer_offset: int,
        data_pool: torch.Tensor,
        scale_pool: torch.Tensor,
        dequant_scale: torch.Tensor,
        num_pool_tokens: int,
        scratch: torch.Tensor,
        compact_indices: torch.Tensor,
        residual_dim: int,
    ):
        """The remap and the dequantization in plain torch, into copies of ``scratch`` and
        ``compact_indices`` (both left untouched). Index arithmetic is int64."""
        rows, top_k = topk_indices.shape
        dev = topk_indices.device
        cols = torch.arange(top_k, device=dev)
        row_len = row_lens[:rows].long().clamp(0, top_k)
        kept = cols[None, :] < row_len[:, None]
        tok = torch.where(kept, topk_indices.long(), -1)
        has_tok = tok >= 0
        blk = torch.where(has_tok, tok // tokens_per_block, 0)
        in_table = has_tok & (blk < block_table.shape[1])
        req = req_idx[:rows].long()[:, None].expand(rows, top_k)
        base = torch.where(in_table, block_table[req, blk.clamp(max=block_table.shape[1] - 1)], -1)
        base = base.long()
        gidx = base * stride_factor + tok % tokens_per_block + layer_offset * tokens_per_block
        valid = in_table & (base >= 0) & (gidx >= 0) & (gidx < num_pool_tokens)
        pair = torch.arange(rows, device=dev)[:, None] * top_k + cols[None, :]
        compact = torch.where(valid, pair, -1).to(compact_indices.dtype)

        out = scratch.clone().view(torch.uint8)
        data_bytes, scale_bytes = (HEAD_DIM + residual_dim) // 2, (HEAD_DIM + residual_dim) // 16
        g_rows = gidx[valid]
        data = data_pool.view(-1, data_bytes)[g_rows]
        scale = scale_pool.view(-1, scale_bytes)[g_rows]
        g = dequant_scale.reshape(-1)[0].float()
        out[:rows][valid] = dequantize_rows(data, scale, g, residual_dim).view(torch.uint8)
        return out.view(scratch.dtype), compact

    def is_valid(
        self,
        topk_indices: torch.Tensor,
        row_lens: torch.Tensor,
        req_idx: torch.Tensor,
        block_table: torch.Tensor,
        tokens_per_block: int,
        stride_factor: int,
        layer_offset: int,
        data_pool: torch.Tensor,
        scale_pool: torch.Tensor,
        dequant_scale: torch.Tensor,
        num_pool_tokens: int,
        scratch: torch.Tensor,
        compact_indices: torch.Tensor,
        residual_dim: int,
    ) -> None:
        rows, top_k = topk_indices.shape
        # The scratch is written as 32-bit words of E4M3 bytes: another dtype of the same shape
        # takes them as its own values.
        assert scratch.dtype == torch.float8_e4m3fn, f"scratch must be E4M3, got {scratch.dtype}"
        # Compact indices are stored at row * top_k + column: a shorter or wider buffer is written
        # out of place.
        assert compact_indices.dtype == torch.int32 and compact_indices.shape == (rows, top_k), (
            f"compact_indices must be int32 {(rows, top_k)}, got "
            f"{compact_indices.dtype} {tuple(compact_indices.shape)}"
        )
        assert row_lens.numel() >= rows and req_idx.numel() >= rows, (
            "one length and request per row"
        )
        # A row's request selects its block-table row without a bound check.
        if rows:
            lo, hi = int(req_idx[:rows].min()), int(req_idx[:rows].max())
            assert 0 <= lo and hi < block_table.shape[0], "req_idx outside the block table"
            # The kernel forms pool indices in int32: a wider index wraps instead of failing.
            last = int(block_table.max()) * stride_factor + (layer_offset + 1) * tokens_per_block
            assert last < 2**31 and num_pool_tokens < 2**31, "pool indices must fit in int32"
        # The kernel has the C++ gather's G != 1 conversions only; G == 1 takes another branch there.
        assert (
            dequant_scale.dtype == torch.float32 and float(dequant_scale.reshape(-1)[0]) != 1.0
        ), "dequant_scale must be an fp32 global scale other than 1"


rubin_helix_kept_gather = RubinHelixKeptGather()

__all__ = ["RubinHelixKeptGather", "dequantize_rows", "nvfp4_pool_views", "rubin_helix_kept_gather"]
