# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SiTU gated activation and its fused Triton implementation."""

from __future__ import annotations

import os
from typing import Mapping, Optional

import torch
import triton  # type: ignore[import]
import triton.language as tl  # type: ignore[import]
import triton.language.extra.libdevice as tldevice  # type: ignore[import]
from torch import nn

from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.logger import logger

from ..flashinfer_utils import get_env_enable_pdl
from .kimi_k3_mk1_gate import mk1_platform_supported

#: Kimi K3 on SM107, attention DP + EP (DEP16) decode: the SiTU activation of the shared expert, which runs on a side
#: stream beside the all-to-all dispatch, launches without programmatic dependent launch (PDL). Opt-in: unset or ``0``:
#: off (the PDL launch); ``auto``: on for batches of at least ``_NO_PDL_MIN_ROWS`` rows, where it pays; ``1``: on for
#: every decode batch.
ENV_SITU_NO_PDL = "TRTLLM_K3_DEP_SITU_NO_PDL"
_NO_PDL_MIN_ROWS = 128
_NO_PDL_MAX_ROWS = 1024  # decode batches only; prefill keeps the PDL launch


def situ_no_pdl_setting() -> str:
    """The switch of the no-PDL launch: ``"0"`` (also when the variable is unset: off), ``"1"`` or ``"auto"``. Any
    other value raises. Every consumer reads it through here, ahead of every decision that depends on the shape or
    the layer, so that all ranks of a server reject a bad value alike: :class:`SituAndMul` reads it when it is built
    (the start of the server), and :func:`situ_without_pdl` before it looks at anything else."""
    setting = os.environ.get(ENV_SITU_NO_PDL, "").strip()
    if setting not in ("", "0", "1", "auto"):
        raise ValueError(f"{ENV_SITU_NO_PDL} must be 0, 1 or auto, got {setting!r}")
    return setting or "0"


def situ_without_pdl(num_rows: int, overlaps_moe_dispatch: bool) -> bool:
    """Whether ``num_rows`` rows of the SiTU activation launch without PDL: the activation of a shared expert beside
    the all-to-all dispatch of an attention-DP MoE layer, a decode batch, on the platform the switch was validated on.
    """
    setting = situ_no_pdl_setting()
    if setting == "0":
        return False  # the default: the PDL launch
    if not overlaps_moe_dispatch or num_rows > _NO_PDL_MAX_ROWS or not mk1_platform_supported():
        return False
    return setting == "1" or num_rows >= _NO_PDL_MIN_ROWS


class SituAndMul(nn.Module):
    """SiTU activation with gate/up multiplicative gating.

    Runs the math in fp32 for numerical stability, then casts back to the
    input dtype. CUDA inputs optionally use the fused Triton implementation;
    CPU and meta inputs keep the eager reference path.

    ``overlaps_moe_dispatch`` marks the activation of a shared expert that runs on
    a side stream beside the all-to-all dispatch of an attention-DP MoE layer
    (see :data:`ENV_SITU_NO_PDL`). A PDL launch parks one waiting CTA per tile
    behind the shared expert's GEMM, and while they wait the dispatch of the same
    layer starts late and runs slow; without PDL it does not. The activation is
    the same bit for bit. A dense MLP or a TP shared expert keeps the PDL launch.
    """

    def __init__(
        self,
        *,
        beta: float = 1.0,
        linear_beta: Optional[float] = None,
        use_fused_activation: bool = False,
        overlaps_moe_dispatch: bool = False,
    ) -> None:
        super().__init__()
        situ_no_pdl_setting()  # a bad TRTLLM_K3_DEP_SITU_NO_PDL fails here, on every rank, and never inside a step
        self.beta = beta
        self.linear_beta = linear_beta
        self.use_fused_activation = use_fused_activation
        self.overlaps_moe_dispatch = overlaps_moe_dispatch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_fused_activation and x.is_cuda:
            return torch.ops.trtllm.situ_and_mul(
                x.reshape(-1, x.shape[-1]), self.beta, self.linear_beta, self.overlaps_moe_dispatch
            ).reshape(*x.shape[:-1], x.shape[-1] // 2)

        d = x.shape[-1] // 2
        gate = x[..., :d].to(torch.float32)
        up = x[..., d:].to(torch.float32)
        situ_a = self.beta * torch.tanh(gate / self.beta) * torch.sigmoid(gate)
        if self.linear_beta is not None:
            up = self.linear_beta * torch.tanh(up / self.linear_beta)
        return (situ_a * up).to(x.dtype)


@triton.jit
def situ_and_mul_kernel(
    o_ptr,
    o_stride,
    x_ptr,
    x_stride,
    d,
    beta,
    linear_beta,
    BLOCK_SIZE: tl.constexpr,
    HAS_LINEAR_BETA: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
) -> None:
    """Fused :class:`SituAndMul` on a packed ``[gate | up]`` row layout."""
    i = tl.program_id(axis=0).to(tl.int64)
    j = tl.program_id(axis=1)

    o_row_ptr = o_ptr + o_stride * i
    x_row_ptr = x_ptr + x_stride * i

    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_wait()

    offsets = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < d

    gate = tl.load(x_row_ptr + offsets, mask=mask).to(tl.float32)
    up = tl.load(x_row_ptr + offsets + d, mask=mask).to(tl.float32)

    situ_a = beta * tldevice.tanh(gate / beta) * tl.sigmoid(gate)
    if HAS_LINEAR_BETA:
        up = linear_beta * tldevice.tanh(up / linear_beta)
    result = situ_a * up

    tl.store(o_row_ptr + offsets, result, mask=mask)

    if LAUNCH_WITH_PDL:
        tl.extra.cuda.gdc_launch_dependents()


@torch.library.custom_op("trtllm::situ_and_mul", mutates_args=())
def situ_and_mul(
    x: torch.Tensor,
    beta: float,
    linear_beta: Optional[float] = None,
    overlaps_moe_dispatch: bool = False,
) -> torch.Tensor:
    """Run fused SiTU on a packed ``[num_tokens, 2 * d]`` input."""
    b, n = x.shape

    assert n % 2 == 0
    if x.stride(1) != 1:
        raise ValueError("situ_and_mul requires a contiguous last dimension")
    d = n // 2
    output = torch.empty((b, d), dtype=x.dtype, device=x.device)

    def grid(meta: Mapping[str, int]) -> tuple[int, int]:
        return (b, triton.cdiv(d, meta["BLOCK_SIZE"]))

    launch_with_pdl = get_env_enable_pdl() and get_sm_version() >= 90
    if launch_with_pdl and situ_without_pdl(b, overlaps_moe_dispatch):
        launch_with_pdl = False
        logger.info_once(
            "SiTU of the shared expert beside the MoE dispatch launches without PDL "
            f"(rows={b}, width={d}; set by {ENV_SITU_NO_PDL}; unset or 0 keeps the PDL launch)",
            key=f"situ_without_pdl_{b}_{d}",
        )
    situ_and_mul_kernel[grid](
        o_ptr=output,
        o_stride=output.stride(0),
        x_ptr=x,
        x_stride=x.stride(0),
        d=d,
        beta=float(beta),
        linear_beta=float(linear_beta) if linear_beta is not None else 1.0,
        BLOCK_SIZE=1024,
        HAS_LINEAR_BETA=linear_beta is not None,
        LAUNCH_WITH_PDL=launch_with_pdl,
        launch_pdl=launch_with_pdl,
    )
    return output


@situ_and_mul.register_fake
def _(
    x: torch.Tensor,
    beta: float,
    linear_beta: Optional[float] = None,
    overlaps_moe_dispatch: bool = False,
) -> torch.Tensor:
    b, n = x.shape
    assert n % 2 == 0
    return x.new_empty((b, n // 2))
