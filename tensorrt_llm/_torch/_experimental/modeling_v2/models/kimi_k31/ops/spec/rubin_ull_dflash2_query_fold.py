# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 drafter's query path (q_a RMSNorm, absorbed q_b x W_UK, query RoPE) and block latent store of one
MLA layer in one CuTe DSL launch."""

from __future__ import annotations

from typing import Tuple

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.spec.rubin_ull_dflash2_query_fold import HEAD, KV_LORA, TOKEN_TILE
from .draft_context_cache import to_draft_fp8
from .mla_norm_rope import _torch_mla_latent_norm_rope, _torch_mla_rope_pairs

#: Kimi K3.1 DFlash2 drafter at helix16 headtp: 6 heads a rank, q_lora 1536.
_HEADS = 6
_Q_LORA = 1536


def fold_query_weight(q_b: torch.Tensor, k_b: torch.Tensor, q_a_norm: torch.Tensor) -> torch.Tensor:
    """``[heads * (kv_lora + rope), q_lora]`` BF16: per head ``(W_UK_h^T @ q_b_nope_h) * g`` then
    ``q_b_rope_h * g`` with ``g`` the q_a RMSNorm weight on the q_lora axis, in FP64 and rounded once to BF16.

    ``x @ weight^T * rsqrt(mean(x ** 2) + eps)``, viewed ``[rows, heads, kv_lora + rope]``, is the absorbed
    query of the normalized ``x`` with its rope columns not yet rotated. ``q_b`` is
    ``[heads * (nope + rope), q_lora]``, ``k_b`` (W_UK) ``[heads, nope, kv_lora]``, ``q_a_norm`` ``[q_lora]``.
    """
    heads, nope, _ = k_b.shape
    q = q_b.view(heads, -1, q_b.shape[-1]).double() * q_a_norm.double()
    latent = torch.matmul(k_b.double().transpose(1, 2), q[:, :nope])
    return (
        torch.cat((latent, q[:, nope:]), dim=1)
        .reshape(-1, q_b.shape[-1])
        .to(q_b.dtype)
        .contiguous()
    )


def _cell(why: str, rows: int, **overrides) -> Cell:
    spec = dict(rows=rows, heads=_HEADS, q_lora=_Q_LORA, page_size=64, positions=torch.int64,
                index_dtype=torch.int64)  # fmt: skip
    spec.update(overrides)
    return Cell(why=why, spec=spec)


class K31DFlash2QueryFold(OpWrapper):
    """``query`` and the block latents of one drafter MLA layer from its ``[q_a | kv_a]`` projection, one launch.

    ``engine`` is a ``PreparedQueryFold`` (``kernels/spec/rubin_ull_dflash2_query_fold.py``) holding the folded
    query weight (``fold_query_weight``) and the layer's kv_a RMSNorm weight. Writes and returns
    ``query [rows, heads, kv_lora + rope]`` (new; the rope columns rotated at ``positions``) and the pool, where
    ``pool[pages[t], offsets[t]] = to_draft_fp8([RMSNorm(kv_a_t) | RoPE(k_pe_t)])``. The block latents are bitwise
    ``dflash2_mla_block_rope``'s; the query is not the bits of the q_a norm, ``q_b`` (or the absorbed weight) and
    ``dflash2_mla_block_rope``: the norm weight is folded into the weight, the normalized q_a is not rounded to BF16
    and the sum of squares takes another FP32 order.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        _cell("helix16 b1: one request's 8-row block, 6 heads (the served drafter step)", 8),
        _cell("5 rows: a partial token tile", 5),
        _cell(
            "one row, int32 positions and indices, 32-token pages",
            1,
            positions=torch.int32,
            index_dtype=torch.int32,
            page_size=32,
        ),  # fmt: skip
    )

    note = """
    The Helix16 drafter runs it on the headtp layout at one request's block
    for layers 1..4 (layer 0 keeps the q_a norm, the query projection and ``dflash2_mla_block_rope``: the context
    overlap's join sits right before its block store). The engine compiles on first use, which must happen outside
    CUDA-graph capture. Only an FP8 pool, kv_lora 512, rope 64 and up to ``TOKEN_TILE`` rows.
    """

    def __call__(
        self,
        engine,
        q_kv: torch.Tensor,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
        pool: torch.Tensor,
        store_pages: torch.Tensor,
        store_offsets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        rows = q_kv.shape[0]
        query = torch.empty((rows, engine.heads, HEAD), dtype=q_kv.dtype, device=q_kv.device)
        engine.run(
            q_kv,
            query.view(rows, -1),
            torch.view_as_real(freqs_cis).reshape(-1),
            positions.reshape(-1),
            pool.view(torch.uint8).view(-1, HEAD),
            store_pages.reshape(-1),
            store_offsets.reshape(-1),
            pool.shape[1],
        )
        return query, pool

    def reference(
        self,
        engine,
        q_kv: torch.Tensor,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
        pool: torch.Tensor,
        store_pages: torch.Tensor,
        store_offsets: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The FP64 norm scale times the FP64 product, rounded once to BF16 and rotated in FP32; the block latents
        as ``dflash2_mla_block_rope``'s reference, applied to a copy of ``pool``."""
        rows, q_lora = q_kv.shape[0], engine.q_lora
        x = q_kv[:, :q_lora].double()
        r = torch.rsqrt(x.square().mean(-1, keepdim=True) + engine.eps)
        query = ((x @ engine.weight.double().t()) * r).to(q_kv.dtype).view(rows, engine.heads, HEAD)
        freqs = freqs_cis[positions.reshape(-1).long()]
        query[..., KV_LORA:] = _torch_mla_rope_pairs(query[..., KV_LORA:], freqs.unsqueeze(1))
        blk_kv = _torch_mla_latent_norm_rope(
            q_kv[:, q_lora:], engine.kv_norm, engine.eps, freqs, KV_LORA
        )
        pool = pool.clone()
        pages, offsets = store_pages.long(), store_offsets.long()
        pool.view(torch.uint8)[pages, offsets] = to_draft_fp8(blk_kv).view(torch.uint8)
        return query, pool

    def is_valid(
        self,
        engine,
        q_kv: torch.Tensor,
        freqs_cis: torch.Tensor,
        positions: torch.Tensor,
        pool: torch.Tensor,
        store_pages: torch.Tensor,
        store_offsets: torch.Tensor,
    ) -> None:
        rows = q_kv.shape[0]
        assert 1 <= rows <= TOKEN_TILE, f"1..{TOKEN_TILE} rows (one token tile), got {rows}"
        assert q_kv.dtype == torch.bfloat16 and q_kv.shape[1] == engine.q_lora + HEAD, (
            f"BF16 q_kv [rows, {engine.q_lora + HEAD}]"
        )
        assert q_kv.is_contiguous() and q_kv.data_ptr() % 32 == 0, (
            "q_kv must be packed and 32-byte aligned"
        )
        assert pool.dtype == torch.float8_e4m3fn and pool.dim() == 3 and pool.shape[-1] == HEAD, (
            "pool must be an FP8 E4M3 [pages, page_size, 576]"
        )
        assert pool.is_contiguous() and pool.data_ptr() % 32 == 0, (
            "pool must be packed and 32-byte aligned"
        )
        assert freqs_cis.dtype == torch.complex64 and freqs_cis.is_contiguous()
        for name, t in (
            ("positions", positions),
            ("pages", store_pages),
            ("offsets", store_offsets),
        ):
            assert (
                t.dtype in (torch.int32, torch.int64) and t.numel() == rows and t.is_contiguous()
            ), name
        assert store_pages.dtype == store_offsets.dtype, "pages and offsets share a dtype"


dflash2_query_fold = K31DFlash2QueryFold()

__all__ = ["K31DFlash2QueryFold", "dflash2_query_fold", "fold_query_weight"]
