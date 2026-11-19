# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA prefill short convolution on Rubin: causal width-4 conv + SiLU over the packed raw Q/K/V."""

from __future__ import annotations

from typing import Tuple

import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.ssm.rubin_kda_prefill_ops import kda_prefill_conv as _kda_prefill_conv

_RAGGED = dict(seq_lens=(1, 2, 3, 17, 64, 300), has_init=(True, False, True, False, True, False))


class RubinKdaPrefillConv(OpWrapper):
    """Return the convolved, SiLU-activated `(q, k, v)`, each a new contiguous bf16 `[1, T, H, 128]`.

    `src` [T, >= 3D] bf16 holds each token's raw `[q | k | v | ...]` projection (extra columns are
    ignored); `weight` [3D, 4] bf16 is the packed per-channel kernel. Sequence `i` occupies rows
    `[starts[i], starts[i + 1])` and continues the window `conv_state[slots[i]]` ([3D, 3] bf16 raw
    inputs) when `has_init[i]`, else starts from zeros. Each selected window is overwritten with the
    sequence's last 3 raw inputs (the old window's tail when the sequence is shorter).
    """

    ARCHS = frozenset({Arch.SM_107})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"{layout} context ({heads} KDA heads per rank): ragged batch, the [q|k|v|g] GEMM output as src",
                spec=dict(heads=heads, pool_slots=8, src_width_of=4, **_RAGGED),
            )
            for layout, heads in (("DEP16", 96), ("TEP8", 12))
        ),
        *(
            Cell(
                why=f"{layout} context: src is the whole stacked [q|k|v|g|f_a|b] GEMM output, row pitch {pitch}",
                spec=dict(heads=heads, pool_slots=8, stacked=True, **_RAGGED),
            )
            for layout, heads, pitch in (("DEP16", 96, 49376), ("TEP8", 12, 6288))
        ),
        Cell(
            why="one long chunked-prefill continuation (4096 tokens) at 96 heads",
            spec=dict(heads=96, seq_lens=(4096,), has_init=(True,), pool_slots=4, src_width_of=4),
        ),
        Cell(
            why="sequences shorter than the window keep the old window's tail",
            spec=dict(
                heads=12,
                seq_lens=(1, 1, 2),
                has_init=(True, True, True),
                pool_slots=4,
                src_width_of=4,
            ),
        ),
        Cell(
            why="dense src (exactly 3D columns) at 12 heads",
            spec=dict(heads=12, pool_slots=8, src_width_of=3, **_RAGGED),
        ),
    )
    note = """
    The fused CuTe DSL convolution of the indexed KDA prefill. src is
    token-major with any row stride and at least 3D columns: a [T, 4D]
    [q | k | v | g] projection, or the specialized targets' whole
    [q | k | v | g | f_a | b] GEMM output (N = 6288 or 49376 columns).
    The outputs are the raw layout rubin_kda_prefill takes, without
    normalization. conv_state is the live [slots, 3D, 3] bf16 pool
    (sections q | k | v, contiguous per slot, slot stride may be padded);
    only the selected rows change. starts are int32 query_start_loc
    (monotone, starts[-1] == T), slots unique nonnegative int32, has_init
    bool, max_seqlen an upper bound of the lengths (it sizes tiles only).

    fp32 accumulation and a fast SiLU, one rounding per output: within
    1.5e-2 relative of an fp32 evaluation; the window update copies raw
    inputs and is bitwise. Outputs and pool must not alias the inputs.
    Compiled once per layout, so warm it before graph capture.
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
        return _kda_prefill_conv(src, weight, conv_state, starts, slots, has_init, max_seqlen)

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
        """fp32 causal convolution per sequence; `max_seqlen` only sizes tiles and is ignored."""
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


rubin_kda_prefill_conv = RubinKdaPrefillConv()
