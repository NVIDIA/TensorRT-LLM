# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compressed-indexer pooling: normed token keys -> RMS-normed group keys, fp32 interior, one launch.

Adapted from SGLang's ``_pool_group`` (``hy4_flash_indexpool.py``, Apache-2.0), stopping at the group key.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper, true_fp32_matmul

#: Rows of the pooling tile: `tl.dot` needs at least 16 rows, and `kpool` is at most 16.
ROWS = 16
#: Input columns per `tl.dot` step, so the fp32 weight tiles stay out of local memory.
BK = 32


@triton.jit
def _indexer_pool_kernel(
    KEYS,  # [B, T, HD] bf16, dense
    VALUE_W,  # [HD, HD] fp32, nn.Linear [out, in]
    GATE_W,  # [HD, HD] fp32, nn.Linear [out, in]
    APE,  # [KPOOL, HD] fp32
    NORM_W,  # [HD] fp32
    OUT,  # [B, T // KPOOL, HD] bf16
    groups,
    tokens,
    eps,
    HD: tl.constexpr,
    KPOOL: tl.constexpr,
    ROWS: tl.constexpr,
    BK: tl.constexpr,
):
    """One group per program, ``(group, row)``. Slots ``>= KPOOL`` are masked; the ``T % KPOOL`` tail is dropped."""
    g = tl.program_id(0).to(tl.int64)
    b = tl.program_id(1).to(tl.int64)
    rows = tl.arange(0, ROWS)
    live = rows < KPOOL
    d = tl.arange(0, HD)
    base = KEYS + (b * tokens + g * KPOOL) * HD
    values = tl.zeros((ROWS, HD), dtype=tl.float32)
    gates = tl.zeros((ROWS, HD), dtype=tl.float32)
    for k0 in tl.static_range(0, HD, BK):
        kk = k0 + tl.arange(0, BK)
        x = tl.load(base + rows[:, None] * HD + kk[None, :], mask=live[:, None], other=0.0).to(
            tl.float32
        )
        # [k, d] of a [out, in] weight is weight[d, k].
        wv = tl.load(VALUE_W + d[None, :] * HD + kk[:, None])
        wg = tl.load(GATE_W + d[None, :] * HD + kk[:, None])
        values += tl.dot(x, wv, input_precision="ieee")
        gates += tl.dot(x, wg, input_precision="ieee")
    ape = tl.load(APE + rows[:, None] * HD + d[None, :], mask=live[:, None], other=0.0)
    logits = tl.where(live[:, None], gates + ape, -float("inf"))
    e = tl.exp(logits - tl.max(logits, axis=0)[None, :])
    p = e / tl.sum(e, axis=0)[None, :]
    pooled = tl.sum(tl.where(live[:, None], values * p, 0.0), axis=0)
    w = tl.load(NORM_W + d)
    pooled = pooled * tl.rsqrt(tl.sum(pooled * pooled, axis=0) / HD + eps) * w
    tl.store(OUT + (b * groups + g) * HD + d, pooled.to(tl.bfloat16))


class IndexerPool(OpWrapper):
    """BF16 ``[B, T, head_dim]`` keys + fp32 ``[out, in]`` weights -> BF16 ``[B, T // kpool, head_dim]``."""

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="prefill at max_seq_len: 1 x 5120 tokens -> 1280 groups, the largest single row",
            spec=dict(batch=1, tokens=5120),
        ),
        Cell(
            why="prefill at max_batch_size: 64 x 5120 tokens -> 81920 groups, the envelope's largest call",
            spec=dict(batch=64, tokens=5120),
        ),
        Cell(
            why="decode: 64 rows x 4 tokens -> one group each, the smallest call",
            spec=dict(batch=64, tokens=4),
        ),
        Cell(
            why="decode with an incomplete group: 8 rows x 7 tokens -> one group, the tail 3 dropped",
            spec=dict(batch=8, tokens=7),
        ),
    )

    def __call__(
        self,
        token_keys: torch.Tensor,
        value_weight: torch.Tensor,
        gate_weight: torch.Tensor,
        intra_group_ape: torch.Tensor,
        key_norm_weight: torch.Tensor,
        eps: float,
        *,
        kpool: int = 4,
    ) -> torch.Tensor:
        batch, tokens, hd = token_keys.shape
        if hd < BK or hd & (hd - 1) or hd % BK:
            raise ValueError(f"indexer_pool: head_dim {hd} must be a power of two, at least {BK}")
        if kpool < 1:
            raise ValueError(f"indexer_pool: kpool {kpool} must be positive")
        for name, weight in (("value_weight", value_weight), ("gate_weight", gate_weight)):
            if tuple(weight.shape) != (hd, hd):
                raise ValueError(f"indexer_pool: {name} {tuple(weight.shape)} != ({hd}, {hd})")
        if tuple(intra_group_ape.shape) != (kpool, hd):
            raise ValueError(
                f"indexer_pool: intra_group_ape {tuple(intra_group_ape.shape)} != ({kpool}, {hd})"
            )
        if tuple(key_norm_weight.shape) != (hd,):
            raise ValueError(
                f"indexer_pool: key_norm_weight {tuple(key_norm_weight.shape)} != ({hd},)"
            )
        groups = tokens // kpool
        out = torch.empty(batch, groups, hd, dtype=torch.bfloat16, device=token_keys.device)
        if batch == 0 or groups == 0:
            return out
        _indexer_pool_kernel[(groups, batch)](
            token_keys,
            value_weight,
            gate_weight,
            intra_group_ape,
            key_norm_weight,
            out,
            groups,
            tokens,
            eps,
            HD=hd,
            KPOOL=kpool,
            ROWS=ROWS,
            BK=BK,
            num_warps=4,
        )
        return out

    def reference(
        self,
        token_keys: torch.Tensor,
        value_weight: torch.Tensor,
        gate_weight: torch.Tensor,
        intra_group_ape: torch.Tensor,
        key_norm_weight: torch.Tensor,
        eps: float,
        *,
        kpool: int = 4,
    ) -> torch.Tensor:
        """The group keys in fp32 from the same operands, rounded to BF16 once at the end."""
        batch, tokens, hd = token_keys.shape
        groups = tokens // kpool
        grouped = token_keys[:, : groups * kpool].reshape(batch, groups, kpool, hd).float()
        with true_fp32_matmul():
            values = torch.matmul(grouped, value_weight.float().t())
            gates = torch.matmul(grouped, gate_weight.float().t()) + intra_group_ape.float()
        pooled = (values * gates.softmax(dim=2)).sum(dim=2)
        scale = torch.rsqrt(pooled.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (pooled * scale * key_norm_weight.float()).to(torch.bfloat16)

    def is_valid(
        self,
        token_keys: torch.Tensor,
        value_weight: torch.Tensor,
        gate_weight: torch.Tensor,
        intra_group_ape: torch.Tensor,
        key_norm_weight: torch.Tensor,
        eps: float,
        *,
        kpool: int = 4,
    ) -> None:
        # Read with dense strides; the callable checks only shapes.
        assert token_keys.is_contiguous(), (
            "token_keys must be contiguous: the kernel walks rows at head_dim stride, so a "
            "slice of the token-key cache (keys[:, :T]) would read the cache's next rows"
        )
        hd = token_keys.shape[2]
        assert tuple(value_weight.stride()) == (hd, 1), (
            f"value_weight must be dense [out, in] as the checkpoint stores it; got strides "
            f"{tuple(value_weight.stride())} (weight.t() is accepted and computes x @ w)"
        )
        assert tuple(gate_weight.stride()) == (hd, 1), (
            f"gate_weight must be dense [out, in] as the checkpoint stores it; got strides "
            f"{tuple(gate_weight.stride())}"
        )
        assert tuple(intra_group_ape.stride()) == (hd, 1), (
            f"intra_group_ape must be dense [kpool, head_dim]; got strides "
            f"{tuple(intra_group_ape.stride())}"
        )
        assert key_norm_weight.is_contiguous(), "key_norm_weight must be contiguous"
        # The pooling tile is ROWS rows: a larger kpool pools its first ROWS members only.
        assert kpool <= ROWS, (
            f"kpool {kpool} exceeds the {ROWS}-row pooling tile; the rest of each group "
            f"would be dropped without a sign"
        )


indexer_pool = IndexerPool()
