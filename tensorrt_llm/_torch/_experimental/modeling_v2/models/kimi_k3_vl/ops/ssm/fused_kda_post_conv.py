# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA prefill's channel-major convolution output to token-major Q / K / V (Triton), Q and K
L2-normalized per head, optionally zero-padded to the prefill chunk with beta alongside."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_kda import _kda_kernels

# The prefill token counts the cells drive: a single token, a partial final chunk, and whole chunks.
_TOKENS = (1, 127, 1024)
# Generation targets: (target, KDA heads per rank).
_GENERATION = (("helix8", 12), ("helix16_moetp4ep4", 6), ("dep16", 96))


def l2_normalize_heads(t: torch.Tensor, eps: float) -> torch.Tensor:
    """``t / sqrt(sum(t^2) + eps)`` over the last dimension, in fp64."""
    td = t.double()
    return td / torch.sqrt(td.square().sum(-1, keepdim=True) + eps)


def pad_tokens(t: torch.Tensor, tokens: int) -> torch.Tensor:
    """``t`` ``[1, T, ...]`` zero-padded along dimension 1 to ``tokens`` rows."""
    if t.shape[1] == tokens:
        return t
    pad = t.new_zeros((t.shape[0], tokens - t.shape[1], *t.shape[2:]))
    return torch.cat([t, pad], dim=1)


class FusedKdaPostConv(OpWrapper):
    """Split ``packed`` ``[3 * H * D, T]`` (the convolution's Q, K, V sections, channel-major) into
    contiguous ``[1, T', H, D]`` Q, K, V in ``packed.dtype``; Q and K are ``t / sqrt(sum(t^2) +
    l2_norm_eps)`` per head. ``T'`` is ``padded_tokens`` when given, else ``T``; rows past ``T``
    are zeros. With ``beta`` (``[T, H]`` or ``[1, T, H]``, unit head stride) a fourth output is
    its fp32 ``[1, T', H]`` copy, zero-padded the same way."""

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = tuple(
        Cell(
            why=f"{target} prefill ({heads} heads), {tokens} tokens: the unfused convolution path",
            spec=dict(heads=heads, tokens=tokens),
        )
        for target, heads in _GENERATION
        for tokens in _TOKENS
    )
    note = """
    Kimi K3's KDA prefill calls it after the causal convolution where
    the target's kda_fused_conv_prefill is off (every generation target), with head_dim 128 and the
    default l2_norm_eps, on the convolution's contiguous output. kda_prepad_prefill
    asks for padded_tokens and beta (a single sequence padded to the prefill chunk), but only the
    tp1 target sets it, and that target runs fused_kda_conv_prefill instead: the padded form is
    accepted and tested by its kernel test, not certified here.

    No state and no metadata; inputs are not mutated, outputs are new. Triton compiles once per
    shape class (the token counts are not specialization keys). A non-contiguous packed takes
    the portable torch path, which takes no padding.
    """

    def __call__(
        self,
        packed: torch.Tensor,
        num_heads: int,
        head_dim: int,
        l2_norm_eps: float = 1e-6,
        padded_tokens: Optional[int] = None,
        beta: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, ...]:
        return _kda_kernels.fused_kda_post_conv(
            packed, num_heads, head_dim, l2_norm_eps, padded_tokens, beta
        )

    def reference(
        self,
        packed: torch.Tensor,
        num_heads: int,
        head_dim: int,
        l2_norm_eps: float = 1e-6,
        padded_tokens: Optional[int] = None,
        beta: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, ...]:
        tokens = packed.shape[1]
        out_tokens = tokens if padded_tokens is None else padded_tokens
        q, k, v = packed.reshape(3, num_heads, head_dim, tokens).permute(0, 3, 1, 2).unsqueeze(1)
        outputs = (
            l2_normalize_heads(q, l2_norm_eps).to(packed.dtype),
            l2_normalize_heads(k, l2_norm_eps).to(packed.dtype),
            v,
        )
        outputs = tuple(pad_tokens(t, out_tokens).contiguous() for t in outputs)
        if beta is None:
            return outputs
        beta_rows = beta.reshape(1, tokens, num_heads).float()
        return (*outputs, pad_tokens(beta_rows, out_tokens).contiguous())


fused_kda_post_conv = FusedKdaPostConv()
