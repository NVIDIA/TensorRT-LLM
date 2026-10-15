# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Causal depthwise conv1d over a whole (ragged) batch of sequences, with each
sequence's convolution state read from and published to a paged state pool in
the same call."""

from __future__ import annotations

from typing import Optional

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: This checkpoint's GDN mixer: `[q|k|v]` is 10240 wide, the conv is width 4,
#: so each pool slot holds a 3-column window.
CKPT_DIM = 10240
#: The largest conv-pool slot count the cache manager was seen to allocate for
#: this checkpoint (10, 35, 366, 368, 396, 399, 432 and 461 across twelve
#: runtime-value dumps, while `max_num_sequences` never exceeds 32). The target
#: hands the pool over unsliced, so 461 tops the certified interval.
ENGINE_SLOTS = 461


def _engine_batch_slots() -> tuple[int, ...]:
    """32 distinct slots over the whole engine pool, slot 0 and the last included."""
    slots = (0, ENGINE_SLOTS - 1) + tuple((i * 13 + 5) % (ENGINE_SLOTS - 2) + 1 for i in range(30))
    assert len(set(slots)) == 32
    return slots


class CausalConv1dFwd(OpWrapper):
    """Depthwise causal conv1d (+ optional SiLU) of every sequence in a batch,
    against its carried window, publishing the new window -- one launch.

    For sequence `s` over tokens `[b, e)` and `slot = cache_indices[s]` (or `s`):
    `xpad = cat([conv_states[slot] if has_initial_state[s] else 0, x[:, b:e]])`,
    `out[c, b + t] = act(sum_k weight[c, k] * xpad[c, t + k] + bias[c])` with the
    taps accumulated in fp32 and rounded once, and
    `conv_states[slot] = xpad[:, L : L + W - 1]` -- the raw input tail, not the
    activation, exact in every dtype.

    Fusion boundary: the state read, the convolution, the bias, the SiLU and the
    state publish are inside. The projection that produces `x`, and slicing the
    result into q/k/v, are the caller's.

    Writes the activation into `out`, or into `x` in place when `out is None`,
    and returns that tensor. `conv_states` is advanced in place for every
    non-padded sequence and is not returned.

    Two kernels sit behind the op, selected from `x`'s layout: unit stride on
    the token axis is channel-major (in place allowed; `out`, if given, must
    have `x`'s exact strides); unit stride on the channel axis with a token
    stride > 1, `dim > 1` and at least one token is channel-last (`out`
    required, channel-last, disjoint from `x`). The two agree bitwise.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec keys: dim, width, dtype, lengths (varlen) or batch3d=(batch, seqlen),
    # pool_slots (None -> no conv_states), slots (None -> cache_indices None),
    # initial (None -> has_initial_state None), silu, bias, pad_slot_id,
    # layout: "in_place" | "out" | "channel_last".
    CELLS: tuple[Cell, ...] = (
        (
            Cell(
                why="checkpoint prefill 37+1+128 tokens in place, 8-slot pool",
                spec=dict(
                    dim=CKPT_DIM,
                    lengths=(37, 1, 128),
                    pool_slots=8,
                    slots=(3, 0, 5),
                    initial=(True, False, True),
                ),
            ),
            Cell(
                why="checkpoint prefill at the engine's 461-slot pool: slots 460, 0, 230",
                spec=dict(
                    dim=CKPT_DIM,
                    lengths=(37, 1, 128),
                    pool_slots=ENGINE_SLOTS,
                    slots=(ENGINE_SLOTS - 1, 0, ENGINE_SLOTS // 2),
                    initial=(True, False, True),
                ),
            ),
            Cell(
                why="32 sequences (the batch ceiling) scattered over the 461-slot pool",
                spec=dict(
                    dim=CKPT_DIM,
                    lengths=tuple(1 + (i * 7) % 96 for i in range(32)),
                    pool_slots=ENGINE_SLOTS,
                    slots=_engine_batch_slots(),
                    initial=tuple(i % 3 != 0 for i in range(32)),
                ),
            ),
            Cell(
                why="explicit channel-major out: result lands there, x is read-only",
                spec=dict(
                    dim=CKPT_DIM,
                    lengths=(64, 9),
                    pool_slots=4,
                    slots=(1, 2),
                    initial=(True, True),
                    layout="out",
                ),
            ),
            Cell(
                why="channel-last x (a bare .t() of token-major), into its own out",
                spec=dict(
                    dim=CKPT_DIM,
                    lengths=(37, 1, 128),
                    pool_slots=8,
                    slots=(3, 0, 5),
                    initial=(True, False, True),
                    layout="channel_last",
                ),
            ),
            Cell(
                why="pad_slot_id -1 (the engine's sentinel): sequence and slot skipped",
                spec=dict(
                    dim=1024,
                    lengths=(16, 8, 16),
                    pool_slots=4,
                    slots=(0, -1, 2),
                    initial=(True, True, True),
                ),
            ),
            Cell(
                why="pad_slot_id 99: the sentinel is compared by value",
                spec=dict(
                    dim=1024,
                    lengths=(16, 8, 16),
                    pool_slots=4,
                    slots=(0, 99, 2),
                    initial=(True, True, True),
                    pad_slot_id=99,
                ),
            ),
            Cell(
                why="sequences shorter than the window: state part carry, part tokens",
                spec=dict(
                    dim=512,
                    lengths=(2, 1, 2),
                    pool_slots=4,
                    slots=(0, 1, 2),
                    initial=(True, True, False),
                ),
            ),
            Cell(
                why="a zero-token sequence republishes its carry unchanged",
                spec=dict(
                    dim=512, lengths=(0, 40), pool_slots=4, slots=(2, 3), initial=(True, False)
                ),
            ),
        )
        + tuple(
            Cell(
                why=f"{str(dtype).removeprefix('torch.')} with bias, silu={silu}",
                spec=dict(
                    dim=256,
                    lengths=(30, 34),
                    pool_slots=2,
                    slots=(0, 1),
                    initial=(True, True),
                    dtype=dtype,
                    bias=True,
                    silu=silu,
                ),
            )
            for dtype in (torch.bfloat16, torch.float16, torch.float32)
            for silu in (True, False)
        )
        + (
            Cell(
                why="conv_states None: every sequence starts from zeros",
                spec=dict(dim=512, lengths=(12, 20), pool_slots=None, slots=None, initial=None),
            ),
            Cell(
                why="cache_indices/has_initial_state None: slot s, no carry",
                spec=dict(dim=512, lengths=(24, 40), pool_slots=2, slots=None, initial=None),
            ),
            Cell(
                why="dense [batch, dim, seqlen] layout, query_start_loc None",
                spec=dict(
                    dim=256, batch3d=(3, 40), pool_slots=3, slots=None, initial=(True, False, True)
                ),
            ),
            Cell(
                why="one 2048-token sequence: the vectorised channel-major load",
                spec=dict(dim=CKPT_DIM, lengths=(2048,), pool_slots=1, slots=(0,), initial=(True,)),
            ),
            Cell(
                why="one 2047-token sequence: the scalar channel-major load",
                spec=dict(dim=CKPT_DIM, lengths=(2047,), pool_slots=1, slots=(0,), initial=(True,)),
            ),
            Cell(
                why="several long sequences: the 128-thread block",
                spec=dict(
                    dim=1024,
                    lengths=(700, 701, 648),
                    pool_slots=3,
                    slots=(0, 1, 2),
                    initial=(True, True, True),
                ),
            ),
            Cell(
                why="every sequence a single token",
                spec=dict(
                    dim=1024,
                    lengths=(1, 1, 1, 1),
                    pool_slots=4,
                    slots=(0, 1, 2, 3),
                    initial=(True, True, True, True),
                ),
            ),
        )
        + tuple(
            Cell(
                why=f"conv width {width}",
                spec=dict(
                    dim=512,
                    width=width,
                    lengths=(33, 7),
                    pool_slots=2,
                    slots=(0, 1),
                    initial=(True, True),
                ),
            )
            for width in (2, 3)
        )
    )

    note = """
    Metadata consumed: none. Batch layout (`query_start_loc`, int32), pool
    (`conv_states`, `[slots, dim, W - 1]`), slot map (`cache_indices`, int32)
    and carry flags (`has_initial_state`, bool) are explicit arguments.

    Call site: the GDN prefill half -- `x` the `[10240, num_prefill_tokens]`
    channel-major output of `extract_transpose_prefill_slice`, W 4, no bias,
    SiLU on, the layer's whole conv pool unsliced, `pad_slot_id` -1, `out` None.
    The first three cells are that call; 461 slots tops the certified pool
    interval (8 and 461 bracket every width the cache manager was seen to
    allocate). At 461 the activation and state are bitwise identical to the
    8-slot result for the same slot content.

    Preconditions the op checks itself (all raise): dtypes in {bf16, fp16,
    fp32}, `conv_states`/`bias` dtype, int32 `query_start_loc`/`cache_indices`,
    bool `has_initial_state` of length batch, `out` dtype/shape/layout for the
    selected kernel, channel-last in place, `x` rank.

    Not checked, not guarded (content, not metadata, or an illegal access
    rather than a wrong number): a `cache_indices` value outside `[0, slots)`;
    `has_initial_state[s]` True with `conv_states=None` (reads through null);
    `conv_states`' shape (only its dtype is validated). Sequences of length 0
    and shorter than `W - 1` are legal. Unnamed slots are bitwise untouched.

    `TRTLLM_CAUSAL_CONV1D_SILU_MODE` selects faster SiLU forms on the
    channel-last kernel; only the default (unset) is certified.

    Hard limits
      x token/channel strides  uint32 stride fields  boundary 2^32 elements
        not probed.
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
    ) -> torch.Tensor:
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
        return x if out is None else out

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
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """`(activation, conv_states)` as the call leaves them; mutates nothing.

        The activation starts from `out` (or `x`) so a padded sequence's
        columns keep whatever was there, as the op leaves them.
        """
        width = weight.shape[-1]
        act = (x if out is None else out).clone()
        states = None if conv_states is None else conv_states.clone()
        if query_start_loc is not None:
            starts = query_start_loc.tolist()
            spans = [(slice(None), slice(starts[s], starts[s + 1])) for s in range(len(starts) - 1)]
        else:
            spans = [(s,) for s in range(x.shape[0])]
        slots = None if cache_indices is None else cache_indices.tolist()
        carry = None if has_initial_state is None else has_initial_state.tolist()
        for s, where in enumerate(spans):
            slot = s if slots is None else slots[s]
            if slots is not None and slot == pad_slot_id:
                continue
            tokens = x[where].float()  # [dim, L]
            length = tokens.shape[1]
            if conv_states is not None and carry is not None and carry[s]:
                prefix = conv_states[slot].float()
            else:
                prefix = tokens.new_zeros(tokens.shape[0], width - 1)
            padded = torch.cat([prefix, tokens], dim=1)
            acc = torch.zeros_like(tokens)
            for k in range(width):
                acc = acc + weight[:, k].float().unsqueeze(1) * padded[:, k : k + length]
            if bias is not None:
                acc = acc + bias.float().unsqueeze(1)
            if silu_activation:
                acc = acc * torch.sigmoid(acc)
            if length:
                act[where] = acc.to(act.dtype)
            if states is not None:
                states[slot] = padded[:, length : length + width - 1].to(states.dtype)
        return act, states

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
        # Widths outside 2..4 fall off the dispatcher with no launch at all:
        # measured at W 1 and W 5, destination and pool bitwise unchanged.
        assert weight.shape[-1] in (2, 3, 4), (
            f"width {weight.shape[-1]} has no kernel; only 2, 3 and 4 are dispatched, "
            "and any other width is a silent no-op"
        )
        # The dispatch reads weight (and bias) as x's type without converting:
        # an fp32 weight with bf16 x moved the result by 11.5.
        assert weight.dtype == x.dtype, (
            "weight must have x's dtype; the kernel reads it as x's type without converting"
        )
        varlen = query_start_loc is not None
        c_axis, l_axis = (0, 1) if varlen else (1, 2)
        if x.dim() != (2 if varlen else 3):
            return  # the op rejects a wrong rank itself
        channel_last = (
            x.stride(c_axis) == 1
            and x.stride(l_axis) > 1
            and x.shape[c_axis] > 1
            and x.shape[l_axis] > 0
        )
        if not channel_last:
            # The channel-major kernel walks the token axis and the state's last
            # axis contiguously, ignoring both strides: token stride 2 landed
            # 16.3 off; a state last-dim stride 2 landed 7.9 off (5.4 in the
            # published state). The channel-last kernel honours every stride.
            assert x.shape[l_axis] == 0 or x.stride(l_axis) == 1, (
                "the channel-major kernel loads the token axis contiguously; "
                "x must have stride 1 along it"
            )
            assert conv_states is None or conv_states.stride(-1) == 1, (
                "the channel-major kernel ignores conv_states' last-dim stride; it must be 1"
            )


causal_conv1d_fwd = CausalConv1dFwd()
