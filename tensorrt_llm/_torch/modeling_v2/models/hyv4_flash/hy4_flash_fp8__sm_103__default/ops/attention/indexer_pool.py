# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compressed-indexer pooling of the Hy4-Flash indexer: normed token keys to group keys, one
launch.

Group ``g`` of a row pools its ``kpool`` consecutive token keys through the checkpoint's two
fp32 projections, adds the intra-group position embedding to the gates, takes a softmax over
the ``kpool`` slots, weights the projected values and RMS-normalizes the sum::

    values  = token_keys[g] @ key_value_projection.weight.T          # fp32
    gates   = token_keys[g] @ key_gate_projection.weight.T + intra_group_ape
    pooled  = sum_m softmax_m(gates) * values_m                      # fp32, over the kpool slots
    group   = bf16(key_norm_weight * pooled * rsqrt(mean(pooled^2) + eps))

which is ``HYV4FlashCompressedIndexer.pool_groups`` in the checkpoint's own implementation.
Adapted from SGLang's ``_pool_group``
(``python/sglang/kernels/ops/attention/dsa/hy4_flash_indexpool.py``, Apache-2.0): its padded
16-row tile, its 32-wide input chunks around an IEEE fp32 ``tl.dot``, its masked fp32 softmax
and its RMSNorm. That kernel continues into the indexer's mRoPE and its FP8 store; this entry
stops at the group key, because the rotation and the scoring are the indexer-scoring entry's.
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
    """Pool one group per program: ``(program_id(0), program_id(1))`` is ``(group, row)``.

    ``values`` and ``gates`` accumulate in fp32 over ``HD / BK`` IEEE dots, so every product is
    the exact product of a bf16 key and an fp32 weight. Slots ``>= KPOOL`` are masked out of the
    loads and get ``-inf`` in the softmax, so a ``kpool`` below the tile pools only its own
    members. A row's keys start at ``row * tokens``, not at ``row * groups * KPOOL``: the tail
    ``T % KPOOL`` keys are dropped from the pooling but still occupy their row.
    """
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
    """``token_keys`` BF16 ``[B, T, head_dim]`` (dense) and the checkpoint's fp32 pooling
    weights -> BF16 ``[B, T // kpool, head_dim]`` group keys.

    The last ``T % kpool`` token keys are dropped: a group is only complete at ``kpool``
    members, as the checkpoint's ``pool_groups`` drops them (``token_keys[:, : num_groups *
    kpool]``). ``value_weight`` and ``gate_weight`` are the checkpoint's ``nn.Linear`` weights
    in their ``[out, in]`` layout, read directly; ``intra_group_ape`` is ``[kpool, head_dim]``
    and ``key_norm_weight`` is ``[head_dim]``, both fp32 as stored.
    """

    ARCHS = frozenset({Arch.SM_103})

    # This checkpoint's indexer: head_dim 128, kpool 4, so a 5120-token row is 1280 groups
    # (the vocabulary's largest call) and a 64-request batch is the envelope's largest row
    # count. The token keys reaching this op are already projected and normed by
    # `token_key_projection`/`token_key_norm`, so a cell is only its `[batch, tokens]`.
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

    note = """
    Called by the target's `Indexer.pool_and_select` in both phases over the keys its
    `token_key_projection` and `token_key_norm` produced, and the returned group keys are
    rotated with the indexer's own mRoPE table at group-start positions before
    `indexer_select` scores them. Pooling is per row and per group: `T` only sizes the grid
    (`(groups, batch)` programs), so a 512-token chunked-prefill chunk is the 5120-token
    cell's shape at a smaller grid and has no cell of its own.

    Adapted from SGLang's `_pool_group`; the tail of that kernel (mRoPE at the group's first
    position, the power-of-two FP8 scale, the paged FP8 store) is not part of this entry.

    Preconditions:
      - `token_keys` and the three weight tensors are read with their dense strides while
        the callable checks only shapes, so a strided one is accepted and wrong. The trap is
        a decode slice of the token-key cache (`keys[:, :T]`, whose row stride is the cache's
        width) and a weight passed as `weight.t()`, which both projections are square enough
        to accept. `is_valid` refuses them.
      - `kpool` is at most the 16-row tile (`ROWS`); a larger one silently pools its first 16
        members. `is_valid` refuses it. `kpool` 1..16 is computed correctly and only 4 is
        certified.
    Loud (not pinned by a test): `head_dim` must be a power of two of at least 32 and a
    multiple of the 32-wide dot chunk, `kpool` at least 1, and the weight, ape and norm
    shapes must agree with `head_dim` and `kpool`; `__call__` raises `ValueError` for each.
    `batch` or a group count of 0 returns an empty tensor before launching.

    Precision: the interior is fp32 throughout, exactly as the checkpoint computes it
    (`grouped.to(float32)`, both projections and the softmax in fp32, `key_norm` in fp32) and
    rounds to BF16 once, at the store. The kernel takes `tl.exp` (the hardware `exp2`) where
    the reference takes torch's softmax, and `tl.rsqrt` where torch takes its own; both are
    within an fp32 ulp of the reference, so only an element whose fp32 value sits on a BF16
    rounding boundary moves, and it moves one BF16 step. Measured on sm_103 (GB300) against
    `reference`, per cell: 0 of 8192 group keys differ at 64 x 4 tokens and 0 of 1024 at
    8 x 7; 9 of 163840 at 1 x 5120 (one step each, 2.4e-3 of the row's largest magnitude);
    682 of 10485760 at 64 x 5120 (relative RMS 2.0e-5, 7 elements past one BF16 step of the
    row scale and none past two). A float64 recompute of the worst element of each differing
    cell puts both the kernel and the reference one BF16 step from the true value in opposite
    directions, so the reference is not the more accurate side. The elements that move further
    in BF16 steps are ones the four-slot weighted sum cancels to near zero; their absolute
    error stays far inside the row-scale band. The entry's test states and derives the band
    (`element_ulp=2`, `rms_ulp=0.25`) and bounds the differing count at one in 10000.

    Plain `@triton.jit`, no autotune, JIT-compiled at first call; the cells drove that JIT
    path on sm_103. `head_dim`, `kpool`, the tile and the chunk are `tl.constexpr`, so each
    geometry is its own compiled specialization. CUDA-graph capturable at a fixed shape (a
    graph's rows are fixed by the batch).

    Not certified: `kpool` other than 4, `head_dim` other than 128, a bf16 or fp16
    `value_weight`/`gate_weight` (the checkpoint keeps both fp32), an fp32 or fp16
    `token_keys`, `eps` other than this checkpoint's 1e-5, batch or token counts past the
    envelope's 64 x 5120, and `kpool` 1..16 other than 4 (the tile computes them; no cell
    drives them).
    """

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
        # Read with the dense strides, while the callable checks only shapes. Each is
        # accepted and silently wrong.
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
