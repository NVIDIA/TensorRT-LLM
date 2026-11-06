# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dilated depthwise short convolution over a packed, ragged serving batch, with
each sequence's convolution state read from and published to a paged pool.

Composed from native torch ops (sourcing tier 3): no TensorRT-LLM kernel takes
a dilation. The ladder, as measured on this machine in this paged form:

* tier 1 -- no registered op schema mentions `dilation`; the three convolutions
  (`causal_conv1d_fwd`, `causal_conv1d_update`, `mamba_conv1d`) are the
  non-dilated family. `causal_conv1d_fwd` fits the batch layout exactly, and a
  zero-interleaved width-10 weight (taps at 0, 3, 6, 9) over the very
  `[slots, channels, 9]` pool **returns normally and computes nothing** --
  activation and pool bitwise unchanged -- because its dispatcher has no branch
  past width 4 and the width check sits on the update entry point only.
  `causal_conv1d_update` raises on the same weight.
* tier 2 -- the vendored Triton `causal_conv1d_fn` does not compile at width
  10; engram's `ShortConv` is dilated but rectangular and stateless.
* tier 3 -- `F.conv1d(dilation=, groups=)` over a packed history plus index
  arithmetic. This entry.

Not taken: `d` phase-separated non-dilated convolutions through
`causal_conv1d_fwd` (three engine calls plus gathers -- not one entry), and a
rectangular `[num_seqs, channels, state_len + max_tokens]` buffer (5.4 GB at 32
x 8192 tokens). A fused dilation-aware paged kernel is the faster path.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: `hc_count * hidden_size = 4 * 2560` channels, kernel 4, dilation `ngram_size`
#: 3, hence the 9-column state the mamba cache manager allocates per slot.
CKPT_CHANNELS = 10240
CKPT_DILATION = 3
#: A PLE-conv-pool slot count of a full-size engine run. The kernel addresses a slot by its
#: index alone, so a pool of another size (the pool shares the recurrent pool's slot space,
#: and a bf16 recurrent pool holds more slots in the same memory) takes the same path. The
#: target hands the whole pool over unsliced.
ENGINE_SLOTS = 461


def _scattered(num_seqs: int) -> tuple[int, ...]:
    """`num_seqs` distinct slots over the whole engine pool, both ends in."""
    slots = ((0, ENGINE_SLOTS - 1) + tuple(i * 13 + 4 for i in range(num_seqs)))[:num_seqs]
    assert len(set(slots)) == num_seqs and max(slots) < ENGINE_SLOTS
    return slots


def _ckpt(why: str, lengths: tuple[int, ...], **spec) -> Cell:
    return Cell(
        why=why,
        spec=dict(channels=CKPT_CHANNELS, lengths=lengths, pool_slots=ENGINE_SLOTS, **spec),
    )


class DilatedShortConvPaged(OpWrapper):
    """For sequence `s` over columns `[b, e)` of the packed channel-major `x`,
    `slot = cache_indices[s]` (or `s`), `state_len = (kernel_size - 1) *
    dilation`: `window = [conv_states[slot] if has_initial_state[s] else 0,
    x[:, b:e]]`, `out[c, b + t] = silu(round(sum_k weight[c, 0, k] *
    window[c, t + k * dilation] + bias[c]))` -- taps in fp32, rounded by the
    convolution, then the SiLU reads that back and rounds again -- and
    `conv_states[slot] = window[:, L : L + state_len]`, the raw input tail.

    The highest tap index multiplies the token being emitted. A padded
    sequence (`cache_indices[s] == pad_slot_id`) publishes nothing and still
    gets output columns -- its tokens against a zero carry, defined and
    meaningless. Two sequences naming one slot: the last live one's state
    survives (defined by construction, not a race; uncertified, as the
    scheduler never names one slot twice).

    Fusion boundary: state read, zero fill for a fresh sequence, the dilated
    depthwise conv, bias, SiLU and the state publish are inside. The
    pre-conv norm producing `x`, slot allocation and reset, and the residual
    add are the caller's.

    Returns a fresh contiguous `[channels, num_tokens]` activation; `x`,
    `weight` and `bias` are read-only. `conv_states` is advanced in place and
    not returned.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec keys: lengths, slots, initial (default all True). Every cell is the
    # call site's own form: a bare .t() of the token-major [T, 10240]
    # activation, kernel 4, dilation 3, bf16, no bias, SiLU,
    # query_start_loc_long (int64), int32 slots, the whole 461-slot pool.
    CELLS: tuple[Cell, ...] = (
        _ckpt("decode: 32 one-token rows", (1,) * 32, slots=_scattered(32)),
        _ckpt(
            "mixed: a 2047-token context + 3 one-token rows", (2047, 1, 1, 1), slots=_scattered(4)
        ),
        _ckpt(
            "MTP verify: a fresh 37-token context + 8 rows of 3 drafts",
            (37,) + (3,) * 8,
            slots=_scattered(9),
            initial=(False,) + (True,) * 8,
        ),
    )

    note = """
    Metadata consumed: none -- the same explicit argument set as
    `causal_conv1d_fwd` (packed batch, pool, int32/int64 slot map, bool carry).

    Call site: the PLE short conv on one layer -- `x` a bare `.t()` of the
    `[T, 10240]` normed activation, weight `[10240, 1, 4]` bf16, no bias,
    dilation 3 (state 9), SiLU on, the whole `[slots, 10240, 9]` pool, int32
    slots, `query_start_loc_long` (int64, rebuilt from `seq_lens` under MTP),
    carry flags covering the whole batch, `pad_slot_id` -1. Certified in that
    form at the 461-slot pool: 32 one-token decode rows, a 2047-token context
    mixed with three one-token rows, and an MTP verify step (a fresh 37-token
    context and 8 requests of 3 tokens). A padded row is not certified.

    Preconditions: `query_start_loc[-1] == x.shape[1]`; a short one indexes
    past the internal history and fires a device-side assert that kills the
    CUDA context (loud, not guarded). An out-of-range slot reaches the same
    bounds check (reasoned, not probed). Pool, weight and bias share `x`'s
    dtype (each mismatch raises); `weight` is 3-D depthwise (2-D raises).
    Only the checkpoint's kernel size 4 and dilation 3 are certified; there is
    no width dispatch.

    CUDA-graph capture is certified (at 32 requests of 8 tokens): no host
    sync, all batch reads on device, a captured graph stays correct when the
    batch tensors' values change.
    Launch-bound on decode (8 rows: 0.461 ms eager vs 0.029 ms of conv; 0.094
    ms replayed from a graph).

    Not measured: pools above 461 slots (a bf16 recurrent state gives the engine
    larger ones; a slot is addressed by its index alone), token counts above 8192,
    `torch.compile`, a pool dtype unlike the model's.
    """

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        conv_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        cache_indices: Optional[torch.Tensor],
        has_initial_state: Optional[torch.Tensor],
        dilation: int,
        silu_activation: bool,
        pad_slot_id: int,
    ) -> torch.Tensor:
        channels, num_tokens = x.shape
        kernel_size = weight.shape[-1]
        state_len = (kernel_size - 1) * dilation
        num_slots = conv_states.shape[0]
        num_seqs = query_start_loc.numel() - 1
        device = x.device

        # Device-resident arithmetic only: no `.item()`, no boolean indexing,
        # no data-dependent shape -- which is what makes the call capturable.
        starts = query_start_loc[:-1].to(torch.int64)
        ends = query_start_loc[1:].to(torch.int64)
        rows = torch.arange(num_seqs, device=device)
        columns = torch.arange(num_tokens, device=device)
        taps = torch.arange(state_len, device=device)

        # The first sequence whose end is strictly past the column; skips
        # zero-token sequences by construction.
        seq_of_column = torch.bucketize(columns, ends.contiguous(), right=True)

        if cache_indices is None:
            slots = rows
            live = torch.ones(num_seqs, dtype=torch.bool, device=device)
        else:
            live = cache_indices != pad_slot_id
            slots = torch.where(live, cache_indices.to(torch.int64), torch.zeros_like(rows))
        carries = (
            torch.zeros_like(live) if has_initial_state is None else (has_initial_state & live)
        )

        # Each sequence's carry placed right ahead of its own tokens, so one
        # convolution over the buffer never reads across a sequence boundary.
        history = x.new_zeros((channels, num_tokens + num_seqs * state_len))
        history.index_copy_(1, columns + state_len * (seq_of_column + 1), x)

        previous = conv_states.index_select(0, slots)
        history.index_copy_(
            1,
            ((starts + rows * state_len).unsqueeze(1) + taps).reshape(-1),
            torch.where(carries.reshape(num_seqs, 1, 1), previous, torch.zeros_like(previous))
            .permute(1, 0, 2)
            .reshape(channels, -1),
        )

        convolved = F.conv1d(
            history,
            weight,
            bias=bias,
            stride=1,
            padding=0,
            dilation=dilation,
            groups=channels,
        )
        if silu_activation:
            convolved = F.silu(convolved)
        out = convolved.index_select(1, columns + state_len * seq_of_column)

        published = (
            history.index_select(1, ((ends + rows * state_len).unsqueeze(1) + taps).reshape(-1))
            .view(channels, num_seqs, state_len)
            .permute(1, 0, 2)
        )

        # Skip padded rows without a host sync: every row writes, but writes
        # the bytes of the highest-numbered live row owning that slot (or the
        # slot's own contents when no live row does), so duplicate
        # destinations carry identical bytes and landing order cannot matter.
        writer = torch.full((num_slots,), -1, dtype=torch.int64, device=device)
        writer.scatter_reduce_(
            0,
            slots,
            torch.where(live, rows, torch.full_like(rows, -1)),
            reduce="amax",
            include_self=True,
        )
        owner = writer.index_select(0, slots)
        conv_states.index_copy_(
            0,
            slots,
            torch.where(
                (owner >= 0).reshape(num_seqs, 1, 1),
                published.index_select(0, owner.clamp_min(0)),
                previous,
            ),
        )
        return out

    def reference(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        conv_states: torch.Tensor,
        query_start_loc: torch.Tensor,
        cache_indices: Optional[torch.Tensor],
        has_initial_state: Optional[torch.Tensor],
        dilation: int,
        silu_activation: bool,
        pad_slot_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(out, conv_states)` as the call leaves them, one sequence at a time
        as an explicit fp32 tap-sum (no convolution op); mutates nothing.

        Rounds where the entry's two calls round: once after the taps, again
        after the SiLU. Publishes in sequence order, so a duplicated slot ends
        with the last live sequence's state.
        """
        channels = x.shape[0]
        kernel_size = weight.shape[-1]
        state_len = (kernel_size - 1) * dilation
        starts = query_start_loc.tolist()
        slots = None if cache_indices is None else cache_indices.tolist()
        carry = None if has_initial_state is None else has_initial_state.tolist()
        w = weight.float()
        out = torch.zeros(channels, x.shape[1], dtype=x.dtype, device=x.device)
        pool = conv_states.clone()
        for s in range(len(starts) - 1):
            begin, end = starts[s], starts[s + 1]
            length = end - begin
            slot = s if slots is None else slots[s]
            live = slots is None or slot != pad_slot_id
            if live and carry is not None and carry[s]:
                prefix = conv_states[slot]
            else:
                prefix = x.new_zeros(channels, state_len)
            window = torch.cat([prefix, x[:, begin:end]], dim=1)
            xf = window.float()
            acc = torch.zeros(channels, length, dtype=torch.float32, device=x.device)
            for k in range(kernel_size):
                acc = acc + w[:, 0, k].reshape(-1, 1) * xf[:, k * dilation : k * dilation + length]
            if bias is not None:
                acc = acc + bias.float().reshape(-1, 1)
            y = acc.to(x.dtype)
            if silu_activation:
                y = F.silu(y.float()).to(x.dtype)
            out[:, begin:end] = y
            if live:
                pool[slot] = window[:, length : length + state_len]
        return out, pool


dilated_short_conv_paged = DilatedShortConvPaged()
