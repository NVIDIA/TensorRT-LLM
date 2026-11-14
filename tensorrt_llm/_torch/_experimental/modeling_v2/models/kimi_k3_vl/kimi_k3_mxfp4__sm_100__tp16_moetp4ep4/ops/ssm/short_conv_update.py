# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FLA's one-token short-convolution step (``fla.modules.conv.triton.ops.causal_conv1d_update``): the step of Kimi
K3's sequential KDA verify."""

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class ShortConvUpdate(OpWrapper):
    """One token of a depthwise causal conv per row: returns ``(y, cache)``.

    ``x`` [N, 1, D] is each row's new input and ``cache`` [N, D, W] its last W inputs, oldest first (column 0 the
    one that drops out). The cache shifts left by one and takes ``x`` as its last column, in place, and
    ``y = act(sum(cache * weight, -1))`` with ``weight`` [D, W]; the returned cache is the argument.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        *(
            Cell(
                why=f"TP16 q/k/v section, {rows} requests: one verify step's column of the projection",
                spec=dict(rows=rows, dim=768, width=4, strided_x=True),
            )
            for rows in (1, 4, 8)
        ),
        Cell(
            why="contiguous input, the TP4 section width",
            spec=dict(rows=3, dim=3072, width=4, strided_x=False),
        ),
    )
    note = """
    Kimi K3's sequential verify (drafts beyond kda_mtp_decode's limit) steps
    each of its q / k / v sections: x is the step's column of the [N, T, D]
    projection (a strided [N, 1, D] view, bf16), cache a dense bf16 [N, D, 4]
    copy of the slot windows left-padded with a zero column, weight the
    module's bf16 conv taps viewed [D, 4], activation "silu", no bias or
    residual. Every step's cache is the next step's.

    FLA is imported on first call. The Triton kernel accumulates in fp32 and
    rounds y to x.dtype; the cache holds raw inputs, so it is bit-exact.
    FLA_CONV_BACKEND=cuda (FLA's CUDA backend) is not this entry.
    """

    def __call__(
        self,
        x: torch.Tensor,
        cache: torch.Tensor,
        weight: torch.Tensor,
        activation: Optional[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Imported on the first call, so that importing the target does not need FLA.
        from fla.modules.conv.triton.ops import causal_conv1d_update

        return causal_conv1d_update(
            x=x, cache=cache, residual=None, weight=weight, bias=None, activation=activation
        )

    def reference(
        self,
        x: torch.Tensor,
        cache: torch.Tensor,
        weight: torch.Tensor,
        activation: Optional[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        shifted = torch.cat([cache[:, :, 1:], x.reshape(x.shape[0], -1, 1).to(cache.dtype)], dim=-1)
        cache.copy_(shifted)
        y = (cache.float() * weight.float()).sum(-1)
        if activation in ("silu", "swish"):
            y = torch.nn.functional.silu(y)
        return y.to(x.dtype).view(x.shape), cache

    def is_valid(
        self,
        x: torch.Tensor,
        cache: torch.Tensor,
        weight: torch.Tensor,
        activation: Optional[str],
    ) -> None:
        # FLA's input guard copies a strided cache and steps the copy: the result is right, the argument stale.
        assert cache.is_contiguous(), (
            "cache must be contiguous, or the step is not written back to it"
        )


short_conv_update = ShortConvUpdate()
