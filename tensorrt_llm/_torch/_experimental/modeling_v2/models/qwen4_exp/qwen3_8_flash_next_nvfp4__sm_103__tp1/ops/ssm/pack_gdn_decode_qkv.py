# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated-DeltaNet decode-phase q/k/v packing: the plain split of the causal
conv-update output into per-head q, k and v, with no normalization and no gates.

Wraps the in-tree module callable
``tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops.pack_gdn_decode_qkv``
(tier-2 sourcing: no op boundary exists). One Triton launch,
``_pack_gdn_decode_qkv_kernel``, after allocating the three outputs.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import (
    pack_gdn_decode_qkv as _pack_gdn_decode_qkv,
)

#: This checkpoint's linear-attention geometry at tp1.
NUM_K_HEADS, HEAD_K_DIM, NUM_V_HEADS, HEAD_V_DIM = 16, 128, 48, 128
CKPT_HEADS = (NUM_K_HEADS, HEAD_K_DIM, NUM_V_HEADS, HEAD_V_DIM)
QKV_DIM = 2 * NUM_K_HEADS * HEAD_K_DIM + NUM_V_HEADS * HEAD_V_DIM  # 10240


class PackGdnDecodeQkv(OpWrapper):
    """Cut a token-major `[num_tokens, >= qkv_dim]` conv output, packed
    `[q | k | v]` along features with
    `qkv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim`, into

        q, k  [1, num_tokens, num_k_heads, head_k_dim]
        v     [1, num_tokens, num_v_heads, head_v_dim]

    Every element is a bit-exact copy: no Q/K L2 norm, no gate, no dtype
    change. Columns at or past `qkv_dim` are dropped.

    Fusion boundary: only the split and per-head reshape are inside. The conv
    (and its activation) is the caller's, and so is the Q/K normalization --
    the recurrence consuming these must run `use_qk_l2norm_in_kernel=True`.
    That is the one way to misuse this op and still get plausible output.

    Row-wise, with no notion of a sequence: a verify step of `B` sequences x
    `P` tokens is passed as `B * P` rows, the token axis keeps the input row
    order, and `q.reshape(B, P, ...)` is a view, not a re-gather.

    Returns three fresh contiguous allocations in `mixed_qkv.dtype`;
    `mixed_qkv` is read through both of its strides and is not modified.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: num_tokens, or (num_decodes, draft) for a verify batch, whose rows
    # are the token-major `[n * p, qkv_dim]` view of the conv's `[n, C, p]` output.
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"target verify call: {b} seqs x {p} tokens, token-major rows",
            spec=dict(num_decodes=b, draft=p),
        )
        for b, p in ((32, 2), (8, 4))
    ) + (Cell(why="127 tokens, contiguous: a partial final block", spec=dict(num_tokens=127)),)

    note = """
    Metadata consumed: none. Stateless; sequence boundaries and draft grouping
    are carried by the caller's row order alone.

    Call site (`_gdn_verify`): `mixed_qkv` the `[n * p, 10240]` token-major
    conv output (contiguous: the conv result carries its input's strides, so
    transposing back is a view), heads `16, 128, 48, 128`, bf16. Certified bitwise
    at that call for 32 sequences of 2 tokens and 8 of 4, and at 127 contiguous
    rows (a partial final block).

    Raise loudly, so not guarded: `head_k_dim` and `head_v_dim` rounding up to
    different powers of two (`128 / 64`: triton `CompilationError`
    "Mismatched type for dim_mask", before any launch); a 1-D input
    (`IndexError` reading `stride(1)`).

    The feature axis is never masked against the input's own extent; the
    token axis is. `is_valid` guards that: `[32, 9216]` at qkv_dim 10240 was
    accepted and returned finite values (83.3% of `v` bit-equal to the block
    it should have read, the rest up to 6.25 off); over-declaring
    `num_v_heads` likewise; a 3-D `[4, 2, 10240]` input faulted with an
    illegal memory access that poisons the process.

    Hard limits
      none probed.
    """

    def __call__(
        self,
        mixed_qkv: torch.Tensor,
        num_k_heads: int,
        head_k_dim: int,
        num_v_heads: int,
        head_v_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return _pack_gdn_decode_qkv(mixed_qkv, num_k_heads, head_k_dim, num_v_heads, head_v_dim)

    def reference(
        self,
        mixed_qkv: torch.Tensor,
        num_k_heads: int,
        head_k_dim: int,
        num_v_heads: int,
        head_v_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """`[q | k | v]` cut with `narrow` and reshaped per head; views where torch allows."""
        n = mixed_qkv.shape[0]
        qk_width, v_width = num_k_heads * head_k_dim, num_v_heads * head_v_dim
        q = mixed_qkv.narrow(1, 0, qk_width).reshape(n, num_k_heads, head_k_dim)
        k = mixed_qkv.narrow(1, qk_width, qk_width).reshape(n, num_k_heads, head_k_dim)
        v = mixed_qkv.narrow(1, 2 * qk_width, v_width).reshape(n, num_v_heads, head_v_dim)
        return q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)

    def is_valid(
        self,
        mixed_qkv: torch.Tensor,
        num_k_heads: int,
        head_k_dim: int,
        num_v_heads: int,
        head_v_dim: int,
    ) -> None:
        # `shape[1]` rather than a rank check: a 1-D input still reaches the
        # op's own IndexError, and a 3-D one (whose shape[1] is unrelated) is
        # refused here instead of faulting.
        qkv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim
        assert mixed_qkv.shape[1] >= qkv_dim, (
            f"mixed_qkv must be at least qkv_dim={qkv_dim} wide "
            f"(2 * {num_k_heads} * {head_k_dim} + {num_v_heads} * {head_v_dim}); got "
            f"shape {tuple(mixed_qkv.shape)} -- the feature axis is not masked against "
            "the input's extent, so a short row is read out of bounds"
        )


pack_gdn_decode_qkv = PackGdnDecodeQkv()
