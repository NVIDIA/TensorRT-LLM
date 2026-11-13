# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The block drafter's context K/V of a decode step: every drafter layer's K (k_norm, NeoX RoPE) and V of the step's
B (K + 1) context tokens into the drafter's paged context pool, the write mask, ``ctx_len += num_acc`` (clamped) and
the context length each request may advertise, in one launch."""

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.kernels.k3_ctx_kv import (  # noqa: F401, E501 — registers the op
    op as _k3_ctx_kv_op,
)

_HEAD = 64


def _columns(ctx_len, slots, rows, counts, page: int, k1: int) -> torch.Tensor:
    """Each token's pool column [B, K + 1]: ``clamp(min(ctx_len[slot] + j, counts[row] page - 1), 0)``."""
    offs = torch.arange(k1, device=ctx_len.device)
    col = ctx_len[slots][:, None] + offs[None, :]
    return torch.minimum(col, (counts[rows] * page - 1)[:, None]).clamp(min=0)


class K3CtxKv(OpWrapper):
    """Write the context K/V of every drafter layer for ``x`` [N = B (K + 1), H] bf16 into the pool addressed from
    ``pool`` (the one-element base view of ``kernels/k3_ctx_kv/op.pool_view``) at ``layer_off`` [L] int64 element
    offsets with the page / K-V / head strides in elements, update ``ctx_len`` [slots] int64 in place, and return
    ``num_ctx`` int32 [B]."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS = (
        Cell(
            why="one TP16 request of a DSpark block of K + 1 = 8 tokens",
            spec=dict(batch=1, k1=8, block=8),
        ),
        Cell(
            why="four requests of 8 tokens with mixed accepted counts and columns crossing a page",
            spec=dict(batch=4, k1=8, block=8),
        ),
        Cell(
            why="eight requests of 8 tokens: the manager's maximum batch, 64 tokens, the widest token tile",
            spec=dict(batch=8, k1=8, block=8),
        ),
        Cell(
            why="eight requests of a DSpark block of K = 7 tokens (block size 7): a token count that is no "
            "power of two",
            spec=dict(batch=8, k1=7, block=7),
        ),
        Cell(
            why="eight requests of one token each: the narrowest per-request split",
            spec=dict(batch=8, k1=1, block=8),
        ),
        Cell(
            why="three requests of four tokens: an odd batch with a shorter draft",
            spec=dict(batch=3, k1=4, block=8),
        ),
        Cell(
            why="two requests of 8, the first at its allocation's end: its last columns clamp to counts page - 1 "
            "and num_ctx to counts page - block",
            spec=dict(batch=2, k1=8, block=8, clamp=True),
        ),
        Cell(
            why="four requests of 8, the second two below max_ctx with every token accepted: ctx_len clamps to "
            "max_ctx",
            spec=dict(batch=4, k1=8, block=8, near_max=True),
        ),
    )
    note = """
    TP16 shape: 5 drafter layers, 1 KV head of 64, hidden 7168, page 64; the
    pool is per-layer HND [pages, 2, nkv, page, 64] views of one allocation
    with one set of strides (page-major or layer-major), as op.pool_view
    builds it once per bound pool outside capture. B <= 8, K + 1 <= 8,
    B (K + 1) <= 64; the kernel's op raises on other shapes. Masked tokens
    (j >= num_acc) still write zeros at their column. Several tokens whose
    columns clamp to one column all write it; which one lands is not fixed.
    The op compiles on the first call for each (B, K + 1) and raises inside
    CUDA-graph capture before that. Every call on a device shares one
    arrival counter, so calls must be ordered on one stream. Integer index
    arguments may be views that start at any element.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        k_norm: torch.Tensor,
        cos_sin: torch.Tensor,
        cpos: torch.Tensor,
        num_acc: torch.Tensor,
        ctx_len: torch.Tensor,
        slots: torch.Tensor,
        rows: torch.Tensor,
        table: torch.Tensor,
        counts: torch.Tensor,
        pool: torch.Tensor,
        layer_off: torch.Tensor,
        page_stride: int,
        kv_stride: int,
        head_stride: int,
        eps: float,
        max_ctx: int,
        page: int,
        block_size: int,
        nkv: int,
    ) -> torch.Tensor:
        return torch.ops.trtllm.k3_ctx_kv(
            x,
            weight,
            k_norm,
            cos_sin,
            cpos,
            num_acc,
            ctx_len,
            slots,
            rows,
            table,
            counts,
            pool,
            layer_off,
            page_stride,
            kv_stride,
            head_stride,
            eps,
            max_ctx,
            page,
            block_size,
            nkv,
        )

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        k_norm: torch.Tensor,
        cos_sin: torch.Tensor,
        cpos: torch.Tensor,
        num_acc: torch.Tensor,
        ctx_len: torch.Tensor,
        slots: torch.Tensor,
        rows: torch.Tensor,
        table: torch.Tensor,
        counts: torch.Tensor,
        pool: torch.Tensor,
        layer_off: torch.Tensor,
        page_stride: int,
        kv_stride: int,
        head_stride: int,
        eps: float,
        max_ctx: int,
        page: int,
        block_size: int,
        nkv: int,
    ) -> torch.Tensor:
        """The projection, k_norm and NeoX RoPE in fp32, rounded to bf16 once at the store; writes the pool and
        ``ctx_len`` in place like the op and returns ``num_ctx`` int32 [B]."""
        batch, k1 = cpos.shape
        n, layers = x.shape[0], layer_off.numel()
        with true_fp32_matmul():
            kv = (x.float() @ weight.float().t()).view(n, layers, 2, nkv, _HEAD)
        k = kv[:, :, 0]
        k = (
            k
            * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + eps)
            * k_norm.float().view(1, layers, 1, _HEAD)
        )
        cs = cos_sin.float()[cpos.reshape(-1)]
        c = cs[:, None, None, : _HEAD // 2]
        s = cs[:, None, None, _HEAD // 2 :]
        k_lo, k_hi = k[..., : _HEAD // 2], k[..., _HEAD // 2 :]
        k = torch.cat([k_lo * c - k_hi * s, k_hi * c + k_lo * s], dim=-1)
        vals = torch.stack([k, kv[:, :, 1]], dim=2)  # [N, L, 2, nkv, 64]
        offs = torch.arange(k1, device=x.device)
        live = (offs[None, :] < num_acc.long()[:, None]).reshape(n)
        vals = (vals * live.view(n, 1, 1, 1, 1).to(vals.dtype)).to(pool.dtype)

        col = _columns(ctx_len, slots, rows, counts, page, k1).reshape(n)
        pages = table[rows[:, None].expand(batch, k1).reshape(n), col // page].long()
        token_base = pages * page_stride + (col % page) * _HEAD  # [N]
        idx = (
            layer_off.view(1, layers, 1, 1, 1)
            + token_base.view(n, 1, 1, 1, 1)
            + torch.arange(2, device=x.device).view(1, 1, 2, 1, 1) * kv_stride
            + torch.arange(nkv, device=x.device).view(1, 1, 1, nkv, 1) * head_stride
            + torch.arange(_HEAD, device=x.device).view(1, 1, 1, 1, _HEAD)
        )
        size = pool.untyped_storage().nbytes() // pool.element_size() - pool.storage_offset()
        flat = pool.as_strided((size,), (1,), pool.storage_offset())
        flat[idx.reshape(-1)] = vals.reshape(-1)

        ctx_len[slots] = torch.clamp(ctx_len[slots] + num_acc.long(), max=max_ctx)
        allocated = (counts[rows] * page - block_size).clamp(min=0)
        return torch.minimum(ctx_len[slots], allocated).to(torch.int32)

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        k_norm: torch.Tensor,
        cos_sin: torch.Tensor,
        cpos: torch.Tensor,
        num_acc: torch.Tensor,
        ctx_len: torch.Tensor,
        slots: torch.Tensor,
        rows: torch.Tensor,
        table: torch.Tensor,
        counts: torch.Tensor,
        pool: torch.Tensor,
        layer_off: torch.Tensor,
        page_stride: int,
        kv_stride: int,
        head_stride: int,
        eps: float,
        max_ctx: int,
        page: int,
        block_size: int,
        nkv: int,
    ) -> None:
        """Raise on the inputs the op takes without a check and answers wrongly: repeated slots (the requests race
        on one ctx_len entry), RoPE positions outside ``cos_sin`` and table entries or columns that address memory
        outside the table row or the pool's allocation (the op reads and writes out of bounds)."""
        if slots.unique().numel() != slots.numel():
            raise ValueError("k3_ctx_kv: repeated slots race on one ctx_len entry")
        if cpos.numel() and (int(cpos.min()) < 0 or int(cpos.max()) >= cos_sin.shape[0]):
            raise ValueError(f"k3_ctx_kv: RoPE positions outside cos_sin's {cos_sin.shape[0]} rows")
        col = _columns(ctx_len, slots, rows, counts, page, cpos.shape[1])
        if int((col // page).max()) >= table.shape[1]:
            raise ValueError("k3_ctx_kv: a column's page lies past the table row (counts > width)")
        pages = table[rows[:, None].expand_as(col), col // page].long()
        if int(pages.min()) < 0:
            raise ValueError("k3_ctx_kv: a negative page in the table")
        size = pool.untyped_storage().nbytes() // pool.element_size()
        last = (
            pool.storage_offset()
            + int(layer_off.max())
            + int(pages.max()) * page_stride
            + kv_stride
            + (nkv - 1) * head_stride
            + page * _HEAD
        )
        if int(layer_off.min()) < 0 or last > size:
            raise ValueError("k3_ctx_kv: a page of the table lies outside the pool's allocation")


k3_ctx_kv = K3CtxKv()
