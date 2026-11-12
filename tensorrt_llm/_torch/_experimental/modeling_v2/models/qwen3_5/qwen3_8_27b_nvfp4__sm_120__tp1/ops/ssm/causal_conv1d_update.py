# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-step causal depthwise conv1d against a slot-indexed conv-state pool: the GDN
decode rows' conv (pure decode steps past the GEMV-fused rows, and a mixed step's
decode riders).

Wraps ``torch.ops.trtllm.causal_conv1d_update`` -- TensorRT-LLM's CUDA kernel, at every
row count (a 2-D ``x`` is handed to it as an ``unsqueeze(-1)`` view). Nothing else runs.
"""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: This checkpoint's GDN q|k|v width (the conv's channels), in_proj_qkvz's width, and
#: the conv: width 4 over a 3-column pool.
CONV_DIM, QKVZ_DIM, WIDTH = 10240, 16384, 4
#: A pool past the largest decode batch (128 rows) the target runs.
POOL_SLOTS = 160


def _cell(why: str, rows: int, **spec) -> Cell:
    return Cell(why=why, spec=dict(rows=rows, **spec))


class CausalConv1dUpdate(OpWrapper):
    """One causal depthwise conv step per batch row against that row's cached window,
    then the window's advance -- `x` and the pool both written in place.

    Row `b`, slot `s = conv_state_indices[b]`, channel `c`, width `W = weight.shape[1]`:
    `window = [conv_state[s, c, -(W - 1):], x[b, c]]`,
    `x[b, c] = act(bias[c] + sum_j weight[c, j] * window[j])` in fp32 rounded once to
    x's dtype (`act` SiLU with `silu_activation`, else identity), and
    `conv_state[s, c] = [conv_state[s, c, 1:], x_old[b, c]]` (the old input, shifted in).
    A row whose slot equals `pad_slot_id` is skipped: its x row and every slot stay as
    they were.

    Fusion boundary: window read, convolution, bias, SiLU, state advance. The projection
    producing `x`, slot allocation and the q/k/v split are the caller's.

    Returns `x` -- the same object, written in place.
    """

    ARCHS = frozenset({Arch.SM_120})

    # spec: rows (decode rows / riders); prefill_rows (P: context rows ahead of x in the
    # projection, default 0); slots ("contiguous": 0..rows-1 | "scattered"); pad (row 1
    # given slot -1); pool ("compact": POOL_SLOTS slots | "coalesced": the cache manager
    # v2's view, slot stride 48 layers' conv states). Every cell is the call site: x the
    # [rows, 10240] q|k|v column view of the rows [P, P + rows) of a [P + rows, 16384]
    # in_proj_qkvz output, bf16, weight [10240, 4], no bias, SiLU, no ring, pad id -1.
    CELLS: tuple[Cell, ...] = (
        _cell("decode 4 rows (first past the GEMV-fused conv), contiguous slots", 4),
        _cell("decode 16 rows, scattered slots, a pad row (-1)", 16, slots="scattered", pad=True),
        _cell("decode 33 rows, scattered slots", 33, slots="scattered"),
        _cell("decode 64 rows, contiguous slots", 64),
        _cell(
            "decode 128 rows (concurrency 128), scattered slots, coalesced pool",
            128,
            slots="scattered",
            pool="coalesced",
        ),
        _cell(
            "mixed step: 1 rider after 2048 context rows", 1, prefill_rows=2048, slots="scattered"
        ),
        _cell(
            "mixed step: 63 riders after 2048 context rows",
            63,
            prefill_rows=2048,
            slots="scattered",
        ),
    )

    note = """
    Metadata consumed: `conv_state_indices` -- the rows' int32 state indices (pure decode:
    `state_indices[:N]`; a mixed step's riders: `state_indices_d`).

    Call site (modeling): `causal_conv1d_update(x, conv_pool, conv_w, None, True, None,
    slots, -1)` with x the `[N, 10240]` column view of in_proj_qkvz's output rows (row
    stride 16384; a mixed step's rows `[P, T)`), the layer's whole bf16 conv pool
    `[S, 10240, 3]` (any slot stride: the kernel takes all three pool strides), the conv
    weight `[10240, 4]`. Pure decode steps at 4..128 rows (1..3 rows fold the conv into
    the in_proj GEMV) and every mixed step's riders; the context rows' conv is
    `gdn_prefill_onepass`'s.

    Launch: TensorRT-LLM's seqlen-1 kernel (SiLU, no ring, `state_len == W - 1`): a thread
    per (row, channel), the `__fmaf_rn` chain over the taps from 0 and
    `v * __frcp_rn(1 + __expf(-v))`, rounded once to bf16; the state shifted by the new
    input. No PDL.

    The op checks metadata itself, loudly: width 2..4, `state_len >= W - 1`, one dtype
    across x / conv_state / weight, weight `[dim, W]`, int32 contiguous length-batch
    `conv_state_indices`. Not checked by the op (is_valid covers it): two rows naming one
    slot race on its window; an out-of-range slot is an illegal access.

    Numerics: the bf16 output inside torch's default band of the fp64 reference (one
    rounding after an fp32 4-tap chain and an approximate exp); the pool exact (a copy
    of inputs).
    """

    def __call__(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        silu_activation: bool,
        cache_seqlens: Optional[torch.Tensor],
        conv_state_indices: Optional[torch.Tensor],
        pad_slot_id: int,
    ) -> torch.Tensor:
        torch.ops.trtllm.causal_conv1d_update(
            x.unsqueeze(-1) if x.dim() == 2 else x,
            conv_state,
            weight,
            bias,
            silu_activation,
            cache_seqlens,
            conv_state_indices,
            pad_slot_id,
        )
        return x

    def reference(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        silu_activation: bool,
        cache_seqlens: Optional[torch.Tensor],
        conv_state_indices: Optional[torch.Tensor],
        pad_slot_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(x, conv_state)` as the call leaves them, for one token a row (2-D `x`, no
        ring): float64 taps, bias and SiLU, rounded once to x's dtype; the window shifted
        by the old input. Mutates nothing. `cache_seqlens` (the ring form) is not what
        the target calls and is refused here."""
        if cache_seqlens is not None:
            raise NotImplementedError("the ring-buffer form is not certified")
        dt = torch.float64
        width = weight.shape[1]
        out = x.clone()
        new_state = conv_state.clone()
        slots = (
            list(range(x.shape[0])) if conv_state_indices is None else conv_state_indices.tolist()
        )
        for row, slot in enumerate(slots):
            if conv_state_indices is not None and slot == pad_slot_id:
                continue
            history = torch.cat([conv_state[slot], x[row].unsqueeze(-1)], dim=1)  # [dim, L + 1]
            window = history[:, -width:].to(dt)
            acc = (weight.to(dt) * window).sum(-1)
            if bias is not None:
                acc = acc + bias.to(dt)
            if silu_activation:
                acc = acc * torch.sigmoid(acc)
            out[row] = acc.to(x.dtype)
            new_state[slot] = history[:, -conv_state.shape[2] :]
        return out, new_state

    def is_valid(
        self,
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        silu_activation: bool,
        cache_seqlens: Optional[torch.Tensor],
        conv_state_indices: Optional[torch.Tensor],
        pad_slot_id: int,
    ) -> None:
        if conv_state_indices is None:
            return
        live = [s for s in conv_state_indices.tolist() if s != pad_slot_id]
        assert len(set(live)) == len(live), "two rows naming one slot race on its window"
        assert all(0 <= s < conv_state.shape[0] for s in live), (
            "a slot outside [0, S) is an illegal access"
        )


causal_conv1d_update = CausalConv1dUpdate()
