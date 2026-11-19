# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 drafter's dynamic block convolution as one CuTe DSL launch."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

from ...kernels.spec.dflash2_grouped_conv import VEC, dflash2_grouped_conv_into
from ..gemm.low_m_linear import k31_low_m_linear, low_m_choice

#: Kimi K3.1 DFlash2 drafter: hidden 7168, conv_kernel_size 2, conv_group_size 16.
_HIDDEN = 7168
_TAPS = 2
_GROUP = 16


def _spec(batch: int, width: int, side: int, fill: str = "randn") -> dict:
    return dict(
        batch=batch, width=width, side=side, hidden=_HIDDEN, taps=_TAPS, group=_GROUP, fill=fill
    )


def _torch_dflash2_grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base_kernel: torch.Tensor,
    block_size: int,
    group_size: int,
) -> torch.Tensor:
    """Short dynamic depthwise convolution across one drafted block.

    ``Conv(x)_t = sum_i k_{t,i} * x_{t-i}``, with ``k_{t,i} = base_kernel[i] +
    delta[t, i]`` (a per-position correction shared by every ``group_size``
    channels). Taps reaching past a block's start contribute nothing.

    Args:
        hidden_states: ``[B * block_size, hidden_size]``, request-major.
        delta: ``[B * block_size, taps, hidden_size // group_size]``.
        base_kernel: ``[taps, hidden_size]``.
    Returns:
        ``[B * block_size, hidden_size]``
    """
    taps, hidden_size = base_kernel.shape
    num_groups = hidden_size // group_size
    # FP32 math, one rounding at the end (what a fused kernel computes).
    blocks = hidden_states.float().unflatten(-1, (num_groups, group_size))
    coefficients = base_kernel.float().view(
        1, taps, num_groups, group_size
    ) + delta.float().unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    if taps > 1:
        position = torch.arange(hidden_states.shape[0], device=hidden_states.device) % block_size
        for tap in range(1, taps):
            shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
            # Select, not multiply: a tap past the block start reads the previous request's row,
            # which may hold anything (NaN in a padding row), and NaN * 0 is NaN.
            shifted = torch.where((position >= tap).view(-1, 1, 1), shifted, 0.0)
            output = output + coefficients[:, tap] * shifted
    return output.flatten(-2).to(hidden_states.dtype)


class K31DFlash2GroupedConv(OpWrapper):
    """``Conv(x)_t = sum_i (base[i] + delta[t, i]) * x[t - i]`` within each drafted block.

    One launch per call, bitwise the FP32 torch reference (``_torch_dflash2_grouped_conv``): the
    same products and sums in the same order, each rounded to nearest on its own,
    one round to the activation dtype at the end. A tap past a block's start multiplies an
    explicit ``0.0``, so the previous request's row (which may hold NaN in a CUDA-graph padding
    request) never contributes. Returns a new tensor.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    CELLS: tuple[Cell, ...] = (
        Cell(
            why="c1, width 5 (max_draft_len 4): the attention prepare side",
            spec=_spec(1, 5, side=0),
        ),
        Cell(
            why="c1, width 8 (max_draft_len 7): the finish side, a strided delta view at offset",
            spec=_spec(1, 8, side=1),
        ),
        Cell(why="64 requests at width 5: 320 rows", spec=_spec(64, 5, side=0)),
        Cell(why="64 requests at width 8: 512 rows, the largest decode step", spec=_spec(64, 8, 1)),
        Cell(
            why="odd batch with NaN rows: a padding request's NaN never leaks into the next block",
            spec=_spec(7, 8, side=0, fill="nan_prev"),
        ),
        Cell(
            why="exact zeros and negative coefficients: signed zeros match bit for bit",
            spec=_spec(13, 5, side=1, fill="zeros"),
        ),
        Cell(
            why="sm_107 DEP16 generation: 12 requests a rank at width 8 (96 rows)",
            spec=_spec(12, 8, side=0),
        ),
        Cell(
            why="sm_107 Helix CP 8 generation: 16 requests at width 6 (96 rows)",
            spec=_spec(16, 6, side=1),
        ),
    )

    note = """
    Shapes: TP does not change the convolution (it runs on the full 7168-wide hidden state on
    every rank), so the cells span the batch (1..64 requests) and the width (5 / 8) only. The
    drafter calls it 4 times per layer (attention prepare / finish, MLP prepare / finish), 20
    times per decode step: one launch each in place of the reference's 15.

    The kernel compiles once per (hidden, taps, group_size, dtype) on the first call; that call
    must be eager (the drafter's warmup), CUDA-graph capture only replays the launch. No host
    synchronization. BF16 activations (the cells); FP16 compiles too but is not a cell. The
    delta and the base kernel must share the activation dtype.
    """

    def __call__(
        self,
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
        base_kernel: torch.Tensor,
        block_size: int,
        group_size: int,
    ) -> torch.Tensor:
        out = torch.empty_like(hidden_states)
        return dflash2_grouped_conv_into(
            hidden_states, delta, base_kernel, block_size, group_size, out
        )

    def reference(
        self,
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
        base_kernel: torch.Tensor,
        block_size: int,
        group_size: int,
    ) -> torch.Tensor:
        """The torch convolution (FP32 math, one rounding at the end)."""
        return _torch_dflash2_grouped_conv(
            hidden_states, delta, base_kernel, block_size, group_size
        )

    def is_valid(
        self,
        hidden_states: torch.Tensor,
        delta: torch.Tensor,
        base_kernel: torch.Tensor,
        block_size: int,
        group_size: int,
    ) -> None:
        dtype = hidden_states.dtype
        assert dtype in (torch.bfloat16, torch.float16), f"BF16/FP16 activations, got {dtype}"
        assert delta.dtype == dtype and base_kernel.dtype == dtype, (
            "the kernel reads the delta and the base kernel in the activation dtype; got "
            f"{delta.dtype} / {base_kernel.dtype} for {dtype}"
        )
        assert hidden_states.dim() == 2 and hidden_states.is_contiguous(), (
            "hidden_states must be a contiguous [rows, hidden]"
        )
        rows, hidden = hidden_states.shape
        taps = base_kernel.shape[0]
        assert group_size % VEC == 0 and hidden % group_size == 0, (
            f"group_size {group_size} must be a multiple of {VEC} dividing hidden {hidden}"
        )
        assert base_kernel.shape == (taps, hidden) and base_kernel.is_contiguous(), (
            f"base_kernel must be a contiguous [taps, {hidden}], got {tuple(base_kernel.shape)}"
        )
        assert delta.shape == (rows, taps, hidden // group_size) and delta.stride(-1) == 1, (
            f"delta must be [{rows}, {taps}, {hidden // group_size}] with a unit last stride"
        )
        # 16-byte vectors: a misaligned base would fault or, worse, read the wrong channels.
        assert hidden_states.data_ptr() % 16 == 0 and base_kernel.data_ptr() % 16 == 0, (
            "hidden_states and base_kernel must be 16-byte aligned"
        )
        assert block_size >= 1, f"block_size must be positive, got {block_size}"


dflash2_grouped_conv = K31DFlash2GroupedConv()


class K31DFlash2BlockConv(nn.Module):
    """Parameter holder + op call; not a layer.

    The DFlash2 dynamic depthwise convolutions wrapped around one drafter sublayer.

    ``prepare`` convolves the sublayer's input and ``finish`` its output; one projection of the input
    (``kernel_projection``) gives the per-position coefficients of both. Each output row is
    ``sum_i (base_kernel[side, i] + delta[t, i]) * x[t - i]`` within a block (a delta per ``group_size`` channels),
    accumulated in FP32 by the CuTe DSL op ``ops/spec/dflash2_grouped_conv``.
    """

    # The drafter's low-M GEMM plan (set with its packed projections); None keeps the kernel projection's GEMM.
    _k31_low_m_plan = None

    def __init__(
        self,
        base_kernel: torch.Tensor,
        kernel_projection_weight: torch.Tensor,
        taps: int,
        group_size: int,
        hidden_size: int,
    ):
        super().__init__()
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        out_features = 2 * taps * self.num_groups
        if tuple(base_kernel.shape) != (2, taps, hidden_size) or tuple(
            kernel_projection_weight.shape
        ) != (out_features, hidden_size):
            raise ValueError(
                f"DFlash2 conv shapes base_kernel {tuple(base_kernel.shape)} / kernel_projection "
                f"{tuple(kernel_projection_weight.shape)}, expected {(2, taps, hidden_size)} / "
                f"{(out_features, hidden_size)}."
            )
        # Side 0 wraps the sublayer input, side 1 its output.
        self.base_kernel = nn.Parameter(base_kernel, requires_grad=False)
        self.kernel_projection = nn.Linear(
            hidden_size,
            out_features,
            bias=False,
            device=kernel_projection_weight.device,
            dtype=kernel_projection_weight.dtype,
        )
        self.kernel_projection.weight.data.copy_(kernel_projection_weight)

    def prepare(
        self, hidden_states: torch.Tensor, block_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convolve the sublayer input; also return ``finish``'s coefficients. The kernel projection runs on the
        low-M plan where it serves."""
        plan, weight = self._k31_low_m_plan, self.kernel_projection.weight
        if plan is not None and low_m_choice(plan, hidden_states, weight) is not None:
            projected = k31_low_m_linear(hidden_states, weight, plan, True)
        else:
            projected = self.kernel_projection(hidden_states)
        coefficients = projected.view(hidden_states.shape[0], 2, self.taps, self.num_groups)
        return (
            self._convolve(hidden_states, coefficients[:, 0], 0, block_size),
            coefficients[:, 1],
        )

    def finish(
        self, hidden_states: torch.Tensor, coefficients: torch.Tensor, block_size: int
    ) -> torch.Tensor:
        """Convolve the sublayer output with the coefficients from ``prepare``."""
        return self._convolve(hidden_states, coefficients, 1, block_size)

    def _convolve(
        self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int, block_size: int
    ) -> torch.Tensor:
        return dflash2_grouped_conv(
            hidden_states, delta, self.base_kernel[side], block_size, self.group_size
        )
