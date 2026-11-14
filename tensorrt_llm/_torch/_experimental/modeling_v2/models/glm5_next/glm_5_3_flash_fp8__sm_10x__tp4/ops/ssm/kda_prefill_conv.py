# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA prefill short convolution: the causal width-4 conv + SiLU of the packed ``[q | k | v]`` projection,
each sequence's conv history read from and written back to its slot of the live conv pool."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_custom_ops  # noqa: F401 -- registers trtllm::kda_prefill_conv
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Channels of one head (KDA's K = V); the op's q / k / v outputs are ``[1, T, H, 128]``.
_HEAD_DIM = 128


class KdaPrefillConv(OpWrapper):
    """``(q, k, v)`` = SiLU of the causal width-4 convolution of the packed projection, one launch.

    Sequence ``b`` is tokens ``starts[b]:starts[b + 1]`` of ``src``. Its raw inputs are preceded by a three-tap
    history: ``conv_state[slots[b]]`` (oldest tap first) when ``has_init[b]``, zeros otherwise. For channel ``c``
    and token ``t``, ``y[t, c] = silu(sum_j weight[c, j] * seq[t + j, c])`` over that history-prefixed sequence,
    tap 3 being token ``t`` itself. Channels ``[0, D)`` are q, ``[D, 2D)`` k and ``[2D, 3D)`` v (``D = H * 128``),
    returned as ``[1, T, H, 128]`` with head ``(c % D) // 128``. The slot's history becomes the last three entries
    of the history-prefixed sequence (raw bit copies), so a sequence shorter than three tokens keeps part of its
    old history (a fresh one, part zeros).

    Args:
        src: bf16 ``[T, >= 3D]`` token-major projection rows; columns past ``3D`` are ignored.
        weight: bf16 ``[3D, 4]`` conv taps of q, k and v stacked (``conv1d.weight.squeeze(1)``).
        conv_state: bf16 ``[slots, 3D, 3]`` live conv pool, updated in place; the slot stride is free.
        starts: int32 ``[B + 1]`` packed token offsets (``query_start_loc``).
        slots: int32 ``[B]`` unique pool slot of each sequence.
        has_init: bool ``[B]``, whether the sequence continues from its slot's history.
        max_seqlen: host int bound on the sequence lengths; it sizes the grid only (any value is correct).

    Returns:
        ``(q, k, v)``, new contiguous bf16 ``[1, T, H, 128]`` tensors, unnormalized.
    """

    ARCHS = frozenset({Arch.SM_100})
    # GLM-5.3-Flash's KDA at TP4: 16 of 64 heads, 3D = 6144 channels. The op's own edges; the prefill steps the
    # shipped target runs are in this entry's test.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="sequences of 1, 2, 3 and 5 tokens: the new history mixes the old window with new inputs",
            spec=dict(
                heads=16,
                seq_lens=(1, 2, 3, 5),
                slots=(6, 0, 9, 3),
                has_init=(True, True, False, True),
                pool=10,
            ),
        ),
        Cell(
            why="lengths off the 32-token run and 8-token group edges, fresh and continuing slots",
            spec=dict(
                heads=16,
                seq_lens=(31, 33, 257, 100),
                slots=(2, 7, 4, 5),
                has_init=(False, True, True, False),
                pool=8,
            ),
        ),
        Cell(
            why="a [q | k | v | g] source row wider than the conv channels (row stride 4D, g columns ignored)",
            spec=dict(
                heads=16,
                seq_lens=(64, 17),
                slots=(1, 0),
                has_init=(True, False),
                pool=4,
                src_width=4 * 16 * _HEAD_DIM,
            ),
        ),
    )
    note = """
    Fused: conv, SiLU and the q / k / v split, no normalization (the prefill
    recurrence normalizes q and k). starts is int32 (query_start_loc); the
    recurrence that follows takes the int64 copy (query_start_loc_long).

    SiLU: for x >= -4 the kernel uses x/2 + x/2 * tanh.approx(x/2) (relative error
    at most about 2^-11 for x >= 0, absolute error under 1e-3 for -4 <= x < 0);
    groups of tokens with any smaller x are recomputed with an exact form. Taps
    accumulate in fp32 with exactly halved bf16 weights; one rounding to bf16.

    Every sequence of a call must name a distinct slot (two sequences on one slot
    race on its history). Fresh sequences need no reset: has_init=False reads a
    zero history and the write-back replaces the whole window. The kernel never
    reads max_seqlen on the device; it sizes the grid, and CTAs loop over the
    device-side run count, so an understated bound is still correct.

    The first call per layout compiles (CuTe DSL); warm it outside CUDA-graph
    capture. Token count, batch size and pool capacity are runtime extents.
    """

    def __call__(
        self,
        src: torch.Tensor,
        weight: torch.Tensor,
        conv_state: torch.Tensor,
        starts: torch.Tensor,
        slots: torch.Tensor,
        has_init: torch.Tensor,
        max_seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.kda_prefill_conv(
            src, weight, conv_state, starts, slots, has_init, max_seqlen
        )

    def reference(
        self,
        src: torch.Tensor,
        weight: torch.Tensor,
        conv_state: torch.Tensor,
        starts: torch.Tensor,
        slots: torch.Tensor,
        has_init: torch.Tensor,
        max_seqlen: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(q, k, v, conv_state as the call leaves it)``; mutates nothing.

        Taps in fp32 in tap order and an exact fp32 SiLU, rounded to bf16 once. ``max_seqlen`` only sizes the
        kernel's grid and is ignored.
        """
        channels, width = weight.shape
        history = width - 1
        heads = channels // (3 * _HEAD_DIM)
        tokens = src.shape[0]
        x = src[:, :channels].float()
        w = weight.float()
        y = torch.zeros(tokens, channels, dtype=torch.float32, device=src.device)
        states = conv_state.clone()
        bounds = starts.tolist()
        for seq, (slot, carry) in enumerate(zip(slots.tolist(), has_init.tolist())):
            lo, hi = bounds[seq], bounds[seq + 1]
            length = hi - lo
            if carry:
                prefix = conv_state[slot].float().t()
            else:
                prefix = x.new_zeros(history, channels)
            padded = torch.cat([prefix, x[lo:hi]], dim=0)
            acc = torch.zeros(length, channels, dtype=torch.float32, device=src.device)
            for tap in range(width):
                acc = acc + padded[tap : tap + length] * w[:, tap]
            y[lo:hi] = torch.nn.functional.silu(acc)
            states[slot] = padded[length : length + history].t().to(states.dtype)
        q, k, v = (
            part.reshape(1, tokens, heads, _HEAD_DIM).to(src.dtype)
            for part in y.split(channels // 3, dim=1)
        )
        return q, k, v, states

    def is_valid(
        self,
        src: torch.Tensor,
        weight: torch.Tensor,
        conv_state: torch.Tensor,
        starts: torch.Tensor,
        slots: torch.Tensor,
        has_init: torch.Tensor,
        max_seqlen: int,
    ) -> None:
        # The op checks dtypes, shapes, contiguity and storage overlap itself; it does not read the metadata
        # values on the host, and these are the ones it gets wrong silently.
        bounds = starts.tolist()
        assert (
            bounds[0] == 0
            and bounds[-1] == src.shape[0]
            and all(lo <= hi for lo, hi in zip(bounds, bounds[1:]))
        ), (
            "starts must monotonically partition the packed tokens: uncovered rows of q / k / v stay unwritten"
        )
        slot_list = slots.tolist()
        assert len(set(slot_list)) == len(slot_list), (
            "slots must be unique: two sequences on one slot race on its history"
        )
        assert all(0 <= slot < conv_state.shape[0] for slot in slot_list), (
            "slots must lie in the pool: slot -1 copies the raw inputs through unconvolved and other "
            "negative slots index the pool from its end"
        )


kda_prefill_conv = KdaPrefillConv()
