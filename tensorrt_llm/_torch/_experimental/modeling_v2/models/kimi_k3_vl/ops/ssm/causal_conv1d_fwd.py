# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prefill causal depthwise conv1d over variable-length sequences, with per-slot conv states."""

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class CausalConv1dFwd(OpWrapper):
    """Convolve each sequence of `x` with `weight` causally, from its slot's conv state, in place.

    Writes the result into `out` (or back into `x`) and each sequence's last `width - 1`
    inputs into `conv_states[cache_indices[b]]`. Returns None.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"Kimi K3 TP{36864 // dim}: mixed fresh and continuing varlen sequences",
                spec=dict(
                    dim=dim,
                    seq_lens=(1, 2, 3, 17, 300, 64),
                    slots=(5, 0, 11, 3, 15, 8),
                    has_init=(True, False, True, False, True, True),
                ),
            )
            for dim in (2304, 9216)
        ),
        Cell(
            why="padding slot skips its activation columns and all state writes",
            spec=dict(
                dim=2304, seq_lens=(9, 33, 5), slots=(2, -1, 7), has_init=(False, True, True)
            ),
        ),
        *(
            Cell(
                why=f"bias at 768 channels with SiLU {'enabled' if silu else 'disabled'}",
                spec=dict(
                    dim=768,
                    seq_lens=(40, 3),
                    slots=(1, 4),
                    has_init=(True, False),
                    silu=silu,
                    use_bias=True,
                ),
            )
            for silu in (False, True)
        ),
        Cell(
            why="channel-last input with a separate destination preserves the input",
            spec=dict(
                dim=2304,
                seq_lens=(17, 130, 2),
                slots=(3, 9, 12),
                has_init=(True, False, True),
                channel_last=True,
            ),
        ),
    )
    note = """
    Caller-owned conv pool, possibly padded between slots; only named slots and
    their first width - 1 history elements are written. query_start_loc,
    cache_indices and has_initial_state are caller-built batch metadata.
    A slot belongs to one sequence: continuation chunks must run in order with
    has_initial_state=True after the first, with no intervening sequence using
    the slot. Swapped chunks silently compute another history. The tests cover
    64 + 1 + 116 continuation and CUDA-graph replay.

    Kimi K3 uses packed q | k | v, width 4, bf16, SiLU and no bias. The
    projection and everything after convolution are outside this entry.
    Equal-length [batch, dim, seqlen], fp16/fp32 and widths 2/3 are accepted
    but not certified here. Channel-last requires a non-overlapping out;
    causal_conv1d_fn allocates that destination for its callers.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        conv_states: Optional[torch.Tensor],
        query_start_loc: Optional[torch.Tensor],
        cache_indices: Optional[torch.Tensor],
        has_initial_state: Optional[torch.Tensor],
        silu_activation: bool,
        pad_slot_id: int,
        out: Optional[torch.Tensor] = None,
    ) -> None:
        torch.ops.trtllm.causal_conv1d_fwd(
            x,
            weight,
            bias,
            conv_states,
            query_start_loc,
            cache_indices,
            has_initial_state,
            silu_activation,
            pad_slot_id,
            out,
        )

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        conv_states: Optional[torch.Tensor],
        query_start_loc: Optional[torch.Tensor],
        cache_indices: Optional[torch.Tensor],
        has_initial_state: Optional[torch.Tensor],
        silu_activation: bool,
        pad_slot_id: int,
        out: Optional[torch.Tensor] = None,
    ) -> None:
        width = weight.shape[-1]
        destination = x if out is None else out
        if query_start_loc is None:
            spans = [(row,) for row in range(x.shape[0])]
        else:
            starts = query_start_loc.tolist()
            spans = [(slice(None), slice(lo, hi)) for lo, hi in zip(starts, starts[1:])]
        slots = range(len(spans)) if cache_indices is None else cache_indices.tolist()
        carry = [False] * len(spans) if has_initial_state is None else has_initial_state.tolist()
        for where, slot, init in zip(spans, slots, carry):
            if cache_indices is not None and slot == pad_slot_id:
                continue
            seq = x[where].float()
            length = seq.shape[-1]
            hist = (
                conv_states[slot, :, : width - 1].float()
                if init
                else seq.new_zeros(seq.shape[0], width - 1)
            )
            full = torch.cat([hist, seq], dim=1)
            y = sum(
                weight[:, tap : tap + 1].float() * full[:, tap : tap + length]
                for tap in range(width)
            )
            if bias is not None:
                y = y + bias.float().unsqueeze(1)
            if silu_activation:
                y = y * torch.sigmoid(y)
            destination[where] = y.to(x.dtype)
            if conv_states is not None:
                conv_states[slot, :, : width - 1] = full[:, -(width - 1) :].to(conv_states.dtype)

    def is_valid(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        conv_states: Optional[torch.Tensor],
        query_start_loc: Optional[torch.Tensor],
        cache_indices: Optional[torch.Tensor],
        has_initial_state: Optional[torch.Tensor],
        silu_activation: bool,
        pad_slot_id: int,
        out: Optional[torch.Tensor] = None,
    ) -> None:
        width = weight.shape[-1]
        assert width in (2, 3, 4), "width outside 2..4 silently dispatches no kernel"
        assert weight.dtype == x.dtype, "weight is read as x.dtype without conversion"
        varlen = query_start_loc is not None
        if x.ndim != (2 if varlen else 3):
            return
        channel_axis, token_axis = (0, 1) if varlen else (1, 2)
        channel_last = (
            x.stride(channel_axis) == 1
            and x.stride(token_axis) > 1
            and x.shape[channel_axis] > 1
            and x.shape[token_axis] > 0
        )
        if not channel_last:
            assert x.shape[token_axis] == 0 or x.stride(token_axis) == 1, (
                "channel-major input must have token stride 1"
            )
            assert conv_states is None or conv_states.stride(-1) == 1, (
                "channel-major convolution ignores the state width stride"
            )
        if varlen:
            assert query_start_loc.ndim == 1, "query_start_loc must be one-dimensional"
        batch = query_start_loc.numel() - 1 if varlen else x.shape[0]
        slots = range(batch) if cache_indices is None else cache_indices.tolist()
        active = [slot for slot in slots if cache_indices is None or slot != pad_slot_id]
        if conv_states is not None:
            assert conv_states.ndim == 3 and conv_states.shape[1] == x.shape[channel_axis], (
                "conv_states must have shape [slots, dim, history]"
            )
            assert conv_states.shape[2] >= width - 1, (
                "conv_states history is shorter than width - 1"
            )
            assert all(0 <= slot < conv_states.shape[0] for slot in active), (
                "cache_indices must name slots inside conv_states"
            )
        elif has_initial_state is not None:
            assert not any(
                init and (cache_indices is None or slot != pad_slot_id)
                for slot, init in zip(slots, has_initial_state.tolist())
            ), "an initial state requires conv_states"


causal_conv1d_fwd = CausalConv1dFwd()
