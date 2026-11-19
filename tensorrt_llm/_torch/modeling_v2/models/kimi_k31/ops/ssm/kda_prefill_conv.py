# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA prefill short convolution: causal width-4 conv + SiLU over packed raw Q/K/V, convolution pool updated."""

from typing import Tuple

import torch

import tensorrt_llm._torch.custom_ops.cute_dsl_kimi_k3_custom_ops  # noqa: F401 — registers the op
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class KdaPrefillConv(OpWrapper):
    """Return the convolved, SiLU-activated `(q, k, v)`, each a new contiguous bf16 `[1, T, H, 128]`.

    `src` [T, >= 3D] bf16 holds each token's raw `[q | k | v | ...]` projection (extra columns are
    ignored); `weight` [3D, 4] bf16 is the packed per-channel kernel. Sequence `i` occupies rows
    `[starts[i], starts[i + 1])` and continues the window `conv_state[slots[i]]` ([3D, 3] bf16 raw
    inputs) when `has_init[i]`, else starts from zeros. Each selected window is overwritten with the
    sequence's last 3 raw inputs (the old window's tail when the sequence is shorter).
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"{heads} KDA heads per rank: ragged packed prefill, fresh and continuation rows",
                spec=dict(
                    heads=heads,
                    seq_lens=(1, 2, 3, 17, 64, 300),
                    has_init=(True, False, True, False, True, False),
                    pool_slots=8,
                ),
            )
            for heads in (96, 48, 24, 12, 6, 3)
        ),
        Cell(
            why="one long chunked-prefill continuation (4096 tokens) at 96 heads",
            spec=dict(heads=96, seq_lens=(4096,), has_init=(True,), pool_slots=4),
        ),
        Cell(
            why="sequences shorter than the window keep the old window's tail",
            spec=dict(heads=6, seq_lens=(1, 1, 2), has_init=(True, True, True), pool_slots=4),
        ),
        Cell(
            why="sm_107 TEP8 context: one 16k-token chunk continuation at 12 heads",
            spec=dict(heads=12, seq_lens=(16384,), has_init=(True,), pool_slots=4),
        ),
    )
    note = """
    The fused CuTe DSL convolution of the indexed KDA prefill: it reads the
    token-major fused projection directly (the target passes the [T, 4D]
    [q | k | v | g] GEMM output; any row stride, base offset allowed) and
    writes the three outputs in the raw layout trtllm::kda_prefill consumes,
    without normalization. conv_state is the live [slots, 3D, W - 1] bf16
    pool (sections q | k | v, contiguous per slot, slot stride may be
    padded); only the selected rows change. starts are the metadata's int32
    query_start_loc (monotone, starts[-1] == T), slots unique nonnegative
    int32, has_init bool, max_seqlen the metadata's kda_max_seqlen (it only
    sizes the per-CTA token tiles and grid: CTAs loop over each sequence's
    device-side length, so a loose bound changes speed, not results). fp32
    accumulation from bf16 inputs and a fast SiLU (h + h * tanh.approx(h),
    h = x / 2, at most 1.3% relative where kept; groups with x < -4 are
    recomputed exactly), one rounding per output: within 1.5e-2 relative of
    an fp32 evaluation; the window update is a copy of raw
    inputs and is bitwise. Outputs and pool must not alias the inputs (the op
    rejects it). Compiled once per layout, warm before graph capture.
    ARCHS lists only the architectures the cells were driven on: sm_103 (GB300)
    and sm_107 (VR200).
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """fp32 causal convolution per sequence; `max_seqlen` is a tiling bound and is ignored."""
        channels, width = weight.shape
        heads = channels // 384
        total = src.shape[0]
        out = torch.zeros(total, channels, dtype=torch.float32, device=src.device)
        bounds = starts.tolist()
        for slot, init, lo, hi in zip(slots.tolist(), has_init.tolist(), bounds, bounds[1:]):
            history = (
                conv_state[slot].float()
                if init
                else torch.zeros(channels, width - 1, device=src.device)
            )
            window = torch.cat((history, src[lo:hi, :channels].float().t()), dim=1)
            taps = window.unfold(1, width, 1)  # [3D, hi - lo, width]
            out[lo:hi] = torch.nn.functional.silu((taps * weight.float().unsqueeze(1)).sum(-1)).t()
            conv_state[slot] = window[:, window.shape[1] - (width - 1) :].to(conv_state.dtype)
        sections = out.to(torch.bfloat16).split(channels // 3, dim=1)
        return tuple(s.reshape(1, total, heads, 128).contiguous() for s in sections)

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
        lengths = [hi - lo for lo, hi in zip(bounds, bounds[1:])]
        assert bounds[0] == 0 and bounds[-1] == src.shape[0] and min(lengths) >= 0, (
            "starts must monotonically partition the packed tokens"
        )
        indices = slots.tolist()
        assert len(set(indices)) == len(indices), "slots must be unique"
        assert all(0 <= slot < conv_state.shape[0] for slot in indices), "slot out of range"


kda_prefill_conv = KdaPrefillConv()
