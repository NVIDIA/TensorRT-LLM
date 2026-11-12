# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA prefill's short convolution, SiLU and Q / K L2 norm in one Triton pass over the token-major
QKV projection, reading and writing back each sequence's conv state."""

from __future__ import annotations

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.k3r_kda import _kda_kernels
from .fused_kda_post_conv import l2_normalize_heads, pad_tokens

PAD_SLOT_ID = _kda_kernels.PAD_SLOT_ID
TP1_HEADS = 96  # the tp1 context server runs every KDA head


class FusedKdaConvPrefill(OpWrapper):
    """Return ``(q, k, v)`` (and ``beta``) of packed sequences, writing their conv states.

    ``qkv`` ``[T, 3 * H * D]`` holds the Q, K, V sections of the projection; sequence ``b`` is
    rows ``query_start_loc[b] : query_start_loc[b + 1]``. Each channel is convolved causally with
    ``conv_weight`` ``[3 * H * D, W]`` over the sequence, preceded by ``conv_states[s]``
    (``s = cache_indices[b]``) where ``has_initial_state[b]`` else zeros, then SiLU and rounded
    to ``qkv.dtype``; Q and K are then ``t / sqrt(sum(t^2) + l2_norm_eps)`` per head (with
    ``normalize_qk``). Outputs are contiguous ``[1, T', H, D]``, ``T'`` = ``padded_tokens`` or
    ``T``, rows past ``T`` zero; with ``beta`` also its fp32 ``[1, T', H]`` copy. In place: each
    sequence's last ``W - 1`` inputs (after its initial state) go to ``conv_states[s]``; a
    sequence on ``pad_slot_id`` reads no state and writes none.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="tp1, one fresh 300-token sequence: Q/K/V and beta zero-padded to the 64 chunk",
            spec=dict(seq_lens=(300,), has_init=(False,), prepad=True),
        ),
        Cell(
            why="tp1, a continuation chunk of 2048 tokens from its conv state, chunk-aligned",
            spec=dict(seq_lens=(2048,), has_init=(True,), prepad=False),
        ),
        Cell(
            why="tp1, four mixed fresh / continuing sequences, one with fewer tokens than the width",
            spec=dict(seq_lens=(64, 2, 129, 63), has_init=(True, True, False, True), prepad=False),
        ),
    )
    note = """
    Kimi K3's KDA prefill calls it where the target's kda_fused_conv_prefill is on (the tp1 context
    target): qkv the [q | k | v] column view of the QKVG output (unit channel stride, row stride
    4 * H * D), the [W, 3 * H * D] weight copy transposed (unit channel stride), the bf16
    [slots, 3 * H * D, 3] conv pool, the metadata's int32 query_start_loc and cache_indices,
    96 heads of 128, width 4, normalize_qk; padded_tokens and beta (a [f_a | b] column view)
    for a single sequence whose length is not a chunk multiple (kda_prepad_prefill).

    Two launches: the token kernel, then the sequence-edge kernel, which rewrites the first
    W - 1 tokens of each sequence and updates its conv state after reading it. The conv
    accumulates in fp32 (bf16 operands with mixed-precision FMA on SM100+), SiLU and the norm
    use approximate exp / reciprocal: results are the unfused causal_conv1d + post-conv chain's
    to rounding. Outputs are new. Triton compiles once per shape class.
    """

    def __call__(
        self,
        qkv: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        num_heads: int,
        head_dim: int,
        l2_norm_eps: float = 1e-6,
        padded_tokens: Optional[int] = None,
        beta: Optional[torch.Tensor] = None,
        pad_slot_id: int = PAD_SLOT_ID,
        normalize_qk: bool = True,
    ) -> tuple[torch.Tensor, ...]:
        return _kda_kernels.fused_kda_conv_prefill(
            qkv,
            conv_weight,
            conv_states,
            query_start_loc,
            cache_indices,
            has_initial_state,
            num_heads,
            head_dim,
            l2_norm_eps,
            padded_tokens,
            beta,
            pad_slot_id,
            normalize_qk,
        )

    def reference(
        self,
        qkv: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        num_heads: int,
        head_dim: int,
        l2_norm_eps: float = 1e-6,
        padded_tokens: Optional[int] = None,
        beta: Optional[torch.Tensor] = None,
        pad_slot_id: int = PAD_SLOT_ID,
        normalize_qk: bool = True,
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        """fp64 convolution: ``(outputs, conv_states after the call)``; nothing is written."""
        tokens = qkv.shape[0]
        width = conv_weight.shape[1]
        weight = conv_weight.double()
        states = conv_states.clone()
        activations = torch.zeros(qkv.shape[1], tokens, dtype=torch.float64, device=qkv.device)
        starts = query_start_loc.tolist()
        for (lo, hi), slot, init in zip(
            zip(starts, starts[1:]), cache_indices.tolist(), has_initial_state.tolist()
        ):
            seq = qkv[lo:hi].double().t()
            padding = slot == pad_slot_id
            if init and not padding:
                hist = conv_states[slot].double()
            else:
                hist = seq.new_zeros(seq.shape[0], width - 1)
            full = torch.cat([hist, seq], dim=1)
            y = sum(weight[:, tap : tap + 1] * full[:, tap : tap + hi - lo] for tap in range(width))
            activations[:, lo:hi] = y * torch.sigmoid(y)
            if not padding:
                states[slot] = full[:, -(width - 1) :].to(states.dtype)
        rounded = activations.to(qkv.dtype).t().reshape(1, tokens, 3, num_heads, head_dim)
        q, k, v = rounded.unbind(2)
        if normalize_qk:
            q = l2_normalize_heads(q, l2_norm_eps).to(qkv.dtype)
            k = l2_normalize_heads(k, l2_norm_eps).to(qkv.dtype)
        out_tokens = tokens if padded_tokens is None else padded_tokens
        outputs = tuple(pad_tokens(t, out_tokens).contiguous() for t in (q, k, v))
        if beta is not None:
            beta_rows = beta.reshape(1, tokens, num_heads).float()
            outputs = (*outputs, pad_tokens(beta_rows, out_tokens).contiguous())
        return outputs, states

    def is_valid(
        self,
        qkv: torch.Tensor,
        conv_weight: torch.Tensor,
        conv_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        cache_indices: torch.Tensor,
        has_initial_state: torch.Tensor,
        num_heads: int,
        head_dim: int,
        l2_norm_eps: float = 1e-6,
        padded_tokens: Optional[int] = None,
        beta: Optional[torch.Tensor] = None,
        pad_slot_id: int = PAD_SLOT_ID,
        normalize_qk: bool = True,
    ) -> None:
        starts = query_start_loc.tolist()
        # The kernels take the sequence bounds as given: overlapping or out-of-range spans
        # convolve the wrong rows without an error.
        assert starts[0] == 0 and starts[-1] == qkv.shape[0], "query_start_loc must span qkv"
        assert all(lo <= hi for lo, hi in zip(starts, starts[1:])), "query_start_loc must ascend"
        slots = [s for s in cache_indices.tolist() if s != pad_slot_id]
        assert all(0 <= s < conv_states.shape[0] for s in slots), "slot outside conv_states"
        assert len(set(slots)) == len(slots), "two sequences on one conv-state slot race"


fused_kda_conv_prefill = FusedKdaConvPrefill()
