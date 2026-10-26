# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated-DeltaNet post-convolution preparation: the q/k/v split of the causal
conv output with Q/K L2 normalization, plus the decay gate and beta.

Wraps the in-tree module callable
``tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops.fused_gdn_post_conv``
(tier-2 sourcing: no op boundary exists). One Triton launch,
``_fused_gdn_post_conv_kernel``, after allocating its five outputs.
"""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm._torch.modules.mamba.fuse_elementwise_ops import (
    fused_gdn_post_conv as _fused_gdn_post_conv,
)

#: This checkpoint's linear-attention geometry at tp1.
NUM_K_HEADS, HEAD_K_DIM, NUM_V_HEADS, HEAD_V_DIM = 16, 128, 48, 128
QKV_DIM = 2 * NUM_K_HEADS * HEAD_K_DIM + NUM_V_HEADS * HEAD_V_DIM  # 10240

#: Read off this checkpoint's `linear_attn.A_log` / `linear_attn.dt_bias`
#: (bf16, 48 per layer): A_log spans [-5.5625, 3.5625] and dt_bias
#: [-8.9375, 19.25]. dt_bias reaching 19.25 makes the softplus threshold branch
#: a live path, not a corner case.
A_LOG_RANGE = (-5.5625, 3.5625)
DT_BIAS_RANGE = (-8.9375, 19.25)
SOFTPLUS_THRESHOLD = 20.0


class FusedGdnPostConv(OpWrapper):
    """Split a causal-conv output into q/k/v and build the GDN gates.

    The token axis is the `num_prefill_tokens` columns of the dim-major
    `prefill` `[>= qkv_dim, n_p]`, then the rows of the token-major `decode`
    `[n_d, >= qkv_dim]` (`None` for a prefill-only batch), with
    `qkv_dim = 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim` packed
    `[q | k | v]`. Per token and head, in fp32:

        q, k = x / sqrt(sum(x^2) + l2_norm_eps)          (per head)
        v    = copied bit-exact
        g    = -exp(A_log) * softplus(a + dt_bias)       (fp32 always)
        beta = sigmoid(b)                                (fp32, rounded once to beta_dtype)

    `softplus(x)` is the unstabilised `log(1 + exp(x))` below
    `softplus_threshold` and `x` above it.

    Fusion boundary: split, Q/K norm, `g` and `beta` are inside. The conv is
    not (`prefill`/`decode` are already its output), and the recurrence that
    consumes these must not normalize q/k again (`use_qk_l2norm_in_kernel=False`).

    Returns `(q, k, v, g, beta)`, five fresh contiguous tensors:
    `[1, T, num_k_heads, head_k_dim]` x2 and `[1, T, num_v_heads, head_v_dim]`
    in `prefill.dtype`, then `[1, T, num_v_heads]` fp32 and `beta_dtype`.
    No input is modified.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: num_prefill, num_decode; dt_bias_fill. Every cell is the target's
    # one [T, 16480] input projection: `decode` the q|k|v column view of its
    # decode rows, `a`/`b` its column views, `prefill` the transposing
    # extraction of its prefill rows.
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"target in_proj views, {p} prefill + {d} decode tokens",
                spec=dict(num_prefill=p, num_decode=d),
            )
            for p, d in ((200, 32), (2048, 0))
        ),
        Cell(
            why="every head above softplus_threshold: the linear branch",
            spec=dict(num_prefill=64, num_decode=0, dt_bias_fill=SOFTPLUS_THRESHOLD + 8.0),
        ),
    )

    note = """
    Metadata consumed: none. Stateless and per token; it never needs sequence
    boundaries.

    Call sites (`_gdn_block`): prefill-only `(mq_t, None, a, b, ...)`, mixed
    `(mq_p_t, mq_d, a, b, ...)` and the MTP-verify context half
    `(mq_p_t, None, a_p, b_p, ...)`, all with `l2_norm_eps=1e-6`,
    `softplus_threshold=20.0` and the fp32 `beta_dtype` default; `prefill` is
    `extract_transpose_prefill_slice`'s contiguous output; `decode` and `a`/`b`
    column views (row stride 16480) of the layer's one `[T, 16480]`
    input-projection output. The stock mixer's verify call passes
    `beta_dtype=b.dtype` (not certified here). Certified in that form at a
    mixed step of 200 prefill and 32 decode tokens, a 2048-token prefill-only
    step, and a 64-token step with every head past the softplus threshold.

    Preconditions: `prefill.shape[0]` and `decode.shape[1]` >= qkv_dim (extra
    trailing features ignored; a short one is read out of bounds); `decode`
    has `prefill`'s dtype. `a`/`b` may have extra columns (the in_proj
    case) and are read through their own strides. `head_k_dim`/`head_v_dim`
    are rounded up to a power of two and masked.

    The op raises on none of the metadata violations probed; `is_valid`
    guards the four the kernel reads out of bounds or through the wrong
    addresses, measured: stride-2 A_log moved g by 2.8e+02, stride-2 dt_bias by
    3.2e+02; A_log, dt_bias, a, b short by rows/elements moved g by 4.9e+12,
    2.9e+02, 3.2e+37 and made beta NaN.

    The softplus is unstabilised: for very negative `x` the fp32 `1 + exp(x)`
    loses `exp(x)` and `g` rounds to exactly 0. Absolute error stays ~1 fp32
    ulp of the log argument times `exp(A_log)` (4.8e-07 measured over
    x in [-9, 5]); relative error in that tail reaches 1.0. Inconsequential
    for a log-domain decay. `exp(x)` is evaluated on the linear side too
    (`tl.where` is a select); harmless below fp32 overflow at x ~ 88, which no
    checkpoint `a + dt_bias` reaches.

    Not certified: fp16/fp32 activations, other head geometries, a bf16
    `beta_dtype`, bf16 `A_log`/`dt_bias`, a non-contiguous `prefill`, contiguous
    `decode`/`a`/`b` (not views of the projection), single-token halves, an
    empty batch.

    Hard limits
      none probed.
    """

    def __call__(
        self,
        prefill: torch.Tensor,
        decode: Optional[torch.Tensor],
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        num_k_heads: int,
        head_k_dim: int,
        num_v_heads: int,
        head_v_dim: int,
        l2_norm_eps: float = 1e-6,
        softplus_threshold: float = 20.0,
        beta_dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return _fused_gdn_post_conv(
            prefill,
            decode,
            a,
            b,
            A_log,
            dt_bias,
            num_k_heads,
            head_k_dim,
            num_v_heads,
            head_v_dim,
            l2_norm_eps,
            softplus_threshold,
            beta_dtype,
        )

    def reference(
        self,
        prefill: torch.Tensor,
        decode: Optional[torch.Tensor],
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        num_k_heads: int,
        head_k_dim: int,
        num_v_heads: int,
        head_v_dim: int,
        l2_norm_eps: float = 1e-6,
        softplus_threshold: float = 20.0,
        beta_dtype: torch.dtype = torch.float32,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """q/k normalized in fp32 and rounded once; `g` in fp64 with the
        stabilised `log1p(exp(x))` (the kernel's unstabilised form differs
        inside the test's floor, not around it), rounded to fp32; `beta` is
        the fp32 sigmoid rounded once to `beta_dtype`."""
        blocks = [prefill.t()[:, : 2 * num_k_heads * head_k_dim + num_v_heads * head_v_dim]]
        if decode is not None:
            blocks.append(decode[:, : blocks[0].shape[1]])
        mixed = torch.cat(blocks, dim=0)
        n = mixed.shape[0]
        kd = num_k_heads * head_k_dim

        def l2norm(x):
            x = x.reshape(n, num_k_heads, head_k_dim).float()
            return (x / torch.sqrt(x.pow(2).sum(-1, keepdim=True) + l2_norm_eps)).to(prefill.dtype)

        q, k = l2norm(mixed[:, :kd]), l2norm(mixed[:, kd : 2 * kd])
        v = mixed[:, 2 * kd :].reshape(n, num_v_heads, head_v_dim).contiguous()
        x = a[:n, :num_v_heads].double() + dt_bias[:num_v_heads].double()
        softplus = torch.where(x <= softplus_threshold, torch.log1p(torch.exp(x)), x)
        g = (-torch.exp(A_log[:num_v_heads].double()) * softplus).float()
        beta = torch.sigmoid(b[:n, :num_v_heads].float()).to(beta_dtype)
        return tuple(t.unsqueeze(0) for t in (q, k, v, g, beta))

    def is_valid(
        self,
        prefill: torch.Tensor,
        decode: Optional[torch.Tensor],
        a: torch.Tensor,
        b: torch.Tensor,
        A_log: torch.Tensor,
        dt_bias: torch.Tensor,
        num_k_heads: int,
        head_k_dim: int,
        num_v_heads: int,
        head_v_dim: int,
        l2_norm_eps: float = 1e-6,
        softplus_threshold: float = 20.0,
        beta_dtype: torch.dtype = torch.float32,
    ) -> None:
        # The kernel indexes A_log/dt_bias as `ptr + head` with no stride and
        # no mask, and does not mask a/b's token axis against their extent.
        num_tokens = prefill.shape[1] + (0 if decode is None else decode.shape[0])
        assert A_log.stride(0) == 1 and dt_bias.stride(0) == 1, (
            f"A_log and dt_bias must be contiguous 1-D; got strides {A_log.stride()} "
            f"and {dt_bias.stride()} -- read as `ptr + head`, a strided view gives a "
            "finite wrong g"
        )
        assert A_log.numel() >= num_v_heads and dt_bias.numel() >= num_v_heads, (
            f"A_log and dt_bias must hold at least num_v_heads={num_v_heads} elements; "
            f"got {A_log.numel()} and {dt_bias.numel()} -- the load is unmasked"
        )
        assert a.shape[0] >= num_tokens and b.shape[0] >= num_tokens, (
            f"a and b must have at least num_tokens={num_tokens} rows (prefill + decode); "
            f"got {a.shape[0]} and {b.shape[0]} -- the token axis is unmasked"
        )


fused_gdn_post_conv = FusedGdnPostConv()
