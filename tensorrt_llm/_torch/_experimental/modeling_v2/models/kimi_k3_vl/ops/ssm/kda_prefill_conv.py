# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA prefill's short convolution: causal width-4 conv + SiLU of the packed q | k | v columns from each sequence's
conv window, to token-major q, k, v, the window updated in place."""

import torch

import tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_custom_ops  # noqa: F401 — registers the op
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class KdaPrefillConv(OpWrapper):
    """Return new contiguous ``(q, k, v)``, each bf16 [1, T, H, 128] and not normalized: SiLU of the causal conv of
    channels ``[0, C)`` of ``src`` [T, >= C] (C = ``weight.shape[0]`` = 3 H 128, q | k | v) with ``weight`` [C, 4],
    sequence b covering rows ``[starts[b], starts[b + 1])`` and starting from ``conv_state[slots[b]]`` [C, 3] (zeros
    when not ``has_init[b]``). Each sequence's last three raw inputs (history included when it is shorter) are
    written to ``conv_state[slots[b]]``."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"TP{96 // heads}: ragged packed prefill from the fused [q | k | v | g] rows, fresh and continuing",
                spec=dict(
                    heads=heads,
                    src_cols=4 * heads * 128,
                    seq_lens=(37, 1, 2, 3, 300, 64),
                    has_init=(False, True, False, True, True, False),
                ),
            )
            for heads in (6, 24)
        ),
        Cell(
            why="one long continuing sequence across many 8-token groups",
            spec=dict(heads=6, src_cols=4 * 6 * 128, seq_lens=(2048,), has_init=(True,)),
        ),
        Cell(
            why="a source exactly C wide, tile-boundary lengths",
            spec=dict(
                heads=6,
                src_cols=3 * 6 * 128,
                seq_lens=(15, 16, 17, 31, 32, 33),
                has_init=(True, False, True, False, True, False),
            ),
        ),
    )
    note = """
    Kimi K3's prefill passes the fused projection's [T, 4 H 128] rows
    ([q | k | v | g], contiguous), the conv weights [3 H 128, 4] bf16 (the
    module's q | k | v conv taps), the manager's bf16 conv pool
    [slots, 3 H 128, 3] (slot rows strided by the manager's coalescing),
    metadata query_start_loc int32 [N + 1], state_indices int32 [N] and
    has_initial_states bool [N], and kda_max_seqlen. Every tensor is on one
    CUDA device; outputs and the pool must not share storage with the inputs
    (the op raises).

    max_seqlen only sizes the grid; an overstated or understated bound still
    covers every token. Slots must be unique and in range: the op does not
    check on device. Slot -1 (skip the pool) is outside Kimi K3's use and not
    certified here.

    SiLU uses a fast tanh form with an exact recompute where it would lose
    accuracy; outputs are within 0.8% relative (1e-4 absolute) of the fp32
    reference, and the conv windows are raw inputs, so they are bit-exact.
    Q/K normalization, gates and the recurrence are kda_prefill's.
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """fp32 conv over each sequence's window; ``max_seqlen`` is ignored."""
        channels, width = weight.shape
        heads = channels // (3 * 128)
        w = weight.float()
        out = torch.zeros(src.shape[0], channels, dtype=torch.float32, device=src.device)
        bounds = starts.tolist()
        for b, (slot, init) in enumerate(zip(slots.tolist(), has_init.tolist())):
            lo, hi = bounds[b], bounds[b + 1]
            hist = (
                conv_state[slot].float()
                if init
                else torch.zeros(channels, width - 1, dtype=torch.float32, device=src.device)
            )
            full = torch.cat([hist, src[lo:hi, :channels].float().t()], dim=1)  # [C, W - 1 + L]
            for t in range(hi - lo):
                out[lo + t] = (full[:, t : t + width] * w).sum(-1)
            conv_state[slot] = full[:, -(width - 1) :].to(conv_state.dtype)
        out = torch.nn.functional.silu(out).to(src.dtype)
        return tuple(
            part.reshape(1, src.shape[0], heads, 128).contiguous()
            for part in out.split(channels // 3, dim=1)
        )

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
        bounds = starts.tolist()
        assert bounds[0] == 0 and bounds[-1] == src.shape[0], (
            "starts must partition the packed source rows"
        )
        assert all(lo <= hi for lo, hi in zip(bounds, bounds[1:])), "starts must be monotone"
        used = slots.tolist()
        assert len(set(used)) == len(used), "slots must be unique"
        assert all(0 <= s < conv_state.shape[0] for s in used), "slot out of range"


kda_prefill_conv = KdaPrefillConv()
