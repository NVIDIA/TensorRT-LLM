# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attention output's per-head norm and sigmoid gate, launched as a PDL kernel.

The built-in attention ends each decode step with
``MuseSparkOpenAttention.apply_output_gate``: one Triton program per
(row, head) normalizes the head's 64 attention outputs without weights, rounds
to BF16, multiplies by the BF16-rounded sigmoid of the matching gate head and
stores BF16 (``_attention_norm_gate_kernel``). It is launched as a plain
kernel, so the output projection after it can start only once it has
completed, and the projection's weight loads begin no earlier than that.

This entry is the same kernel -- statement for statement the built-in
arithmetic, the same grid, the same ``num_warps`` -- launched with
programmatic dependent launch. It waits (``griddepcontrol.wait``) before its
first global read, since the attention output is the previous kernel's, and it
may let its dependent launch early: a projection kernel that loads its weight
before its own ``griddepcontrol.wait`` (flashinfer's split-K GEMM does) then
streams that weight while this grid runs. Whatever the trigger position, the
dependent's wait returns only once this grid has completed and its stores are
visible, so a dependent that reads its activation after its wait reads this
kernel's output.

The kernel copies the built-in ``_attention_norm_gate_kernel`` and the launch
``MuseSparkOpenAttention.apply_output_gate`` makes; ``reference`` runs that
built-in path.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, OpWrapper

#: Width of one attention head.
HEAD_DIM = 64
#: Rows of a decode step this entry is certified for: one request's last token and 7 drafts.
MAX_ROWS = 8

#: Where the kernel lets its dependent launch. ``EXIT``: implicitly, once the
#: grid completes (a plain PDL launch). ``AFTER_WAIT``: right after its own
#: ``griddepcontrol.wait``, i.e. once its predecessor has completed, so the
#: dependent's prologue overlaps this grid only. ``ENTRY``: before its own
#: wait, so the dependent may also overlap the predecessor's tail.
EXIT, AFTER_WAIT, ENTRY = 0, 1, 2
TRIGGERS = {"exit": EXIT, "after_wait": AFTER_WAIT, "entry": ENTRY}


@triton.jit
def _attention_norm_gate_pdl_kernel(
    out,
    attention,
    gate,
    attention_stride,
    gate_token_stride,
    gate_head_stride,
    eps,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TRIGGER: tl.constexpr,
):
    if TRIGGER == 2:
        tl.extra.cuda.gdc_launch_dependents()
    # Before any global read: the attention output is the previous kernel's.
    tl.extra.cuda.gdc_wait()
    if TRIGGER == 1:
        tl.extra.cuda.gdc_launch_dependents()
    # From here on: the built-in _attention_norm_gate_kernel, statement for statement.
    row = tl.program_id(0)
    token = row // HEADS
    head = row % HEADS
    offsets = tl.arange(0, HEAD_DIM)
    x = tl.load(attention + token * attention_stride + head * HEAD_DIM + offsets).to(tl.float32)
    variance = tl.sum(x * x, axis=0) / HEAD_DIM
    normed = (x * tl.rsqrt(variance + eps)).to(tl.bfloat16)
    g = tl.load(gate + token * gate_token_stride + head * gate_head_stride + offsets).to(tl.float32)
    sigmoid = tl.sigmoid(g).to(tl.bfloat16)
    result = normed.to(tl.float32) * sigmoid.to(tl.float32)
    tl.store(out + row * HEAD_DIM + offsets, result)


@triton.jit
def _attention_norm_gate_pdl_heads_kernel(
    out,
    attention,
    gate,
    attention_stride,
    gate_token_stride,
    gate_head_stride,
    eps,
    HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    TRIGGER: tl.constexpr,
    HEADS_PER_PROGRAM: tl.constexpr,
):
    """``_attention_norm_gate_pdl_kernel`` over ``HEADS_PER_PROGRAM`` consecutive (row, head)
    pairs per program: each pair runs the same statements on the same layout."""
    if TRIGGER == 2:
        tl.extra.cuda.gdc_launch_dependents()
    tl.extra.cuda.gdc_wait()
    if TRIGGER == 1:
        tl.extra.cuda.gdc_launch_dependents()
    offsets = tl.arange(0, HEAD_DIM)
    for i in tl.static_range(HEADS_PER_PROGRAM):
        row = tl.program_id(0) * HEADS_PER_PROGRAM + i
        token = row // HEADS
        head = row % HEADS
        x = tl.load(attention + token * attention_stride + head * HEAD_DIM + offsets).to(tl.float32)
        variance = tl.sum(x * x, axis=0) / HEAD_DIM
        normed = (x * tl.rsqrt(variance + eps)).to(tl.bfloat16)
        g = tl.load(gate + token * gate_token_stride + head * gate_head_stride + offsets).to(
            tl.float32
        )
        sigmoid = tl.sigmoid(g).to(tl.bfloat16)
        result = normed.to(tl.float32) * sigmoid.to(tl.float32)
        tl.store(out + row * HEAD_DIM + offsets, result)


def gate_head_stride(gate: torch.Tensor) -> int:
    """The stride between heads of ``gate`` as the built-in launch computes it."""
    return gate.stride(1) if gate.ndim == 3 else HEAD_DIM


class AttentionNormGate(OpWrapper):
    """``bf16(rmsnorm(head)) * bf16(sigmoid(gate head))`` per (row, head), BF16, PDL launch.

    ``attention`` is the attention output ``[rows, H * 64]`` BF16 (unit last
    stride, any row stride), ``gate`` the gate ``[rows, H * 64]`` or
    ``[rows, H, 64]`` BF16 with unit last stride. Returns a fresh contiguous
    ``[rows, H * 64]`` BF16 tensor. ``trigger`` names when the dependent may
    launch (``TRIGGERS``); it changes nothing the kernel computes.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})

    note = """
    Certified bit for bit against the built-in `_attention_norm_gate_kernel`
    (raw BF16 bits over random heads across six decades, exact-zero heads,
    huge and tiny gates, infinities and NaN), eagerly and through CUDA-graph
    replays behind a PDL predecessor whose output lands late (a negative
    control that would read stale values if any load preceded the wait), and
    against `reference` within a band. Rows 1..8: the target calls it only on
    decode steps; the kernel itself has no row limit.
    """

    def __call__(
        self,
        attention: torch.Tensor,
        gate: torch.Tensor,
        eps: float,
        trigger: str = "after_wait",
        out: Optional[torch.Tensor] = None,
        heads_per_program: int = 1,
    ) -> torch.Tensor:
        if out is None:
            out = torch.empty_like(attention, memory_format=torch.contiguous_format)
        heads = attention.shape[1] // HEAD_DIM
        if heads_per_program > 1:
            if heads % heads_per_program:
                raise ValueError(
                    f"heads_per_program must divide the {heads} heads, got {heads_per_program}"
                )
            _attention_norm_gate_pdl_heads_kernel[
                (attention.shape[0] * heads // heads_per_program,)
            ](
                out,
                attention,
                gate,
                attention.stride(0),
                gate.stride(0),
                gate_head_stride(gate),
                eps,
                HEADS=heads,
                HEAD_DIM=HEAD_DIM,
                TRIGGER=TRIGGERS[trigger],
                HEADS_PER_PROGRAM=heads_per_program,
                num_warps=4,
                launch_pdl=True,
            )
            return out
        _attention_norm_gate_pdl_kernel[(attention.shape[0] * heads,)](
            out,
            attention,
            gate,
            attention.stride(0),
            gate.stride(0),
            gate_head_stride(gate),
            eps,
            HEADS=heads,
            HEAD_DIM=HEAD_DIM,
            TRIGGER=TRIGGERS[trigger],
            num_warps=4,
            launch_pdl=True,
        )
        return out

    def reference(
        self,
        attention: torch.Tensor,
        gate: torch.Tensor,
        eps: float,
        trigger: str = "after_wait",
        out: Optional[torch.Tensor] = None,
        heads_per_program: int = 1,
    ) -> torch.Tensor:
        """float64 between the op's BF16 rounding points, which it keeps.

        ``trigger``, ``out`` and ``heads_per_program`` change where the result goes, when a
        dependent may start and how the grid is cut, not the result.
        """
        rows, width = attention.shape
        heads = width // HEAD_DIM
        x = attention.reshape(rows, heads, HEAD_DIM).double()
        normed = (x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps)).to(torch.bfloat16)
        g = gate.reshape(rows, heads, HEAD_DIM).double()
        sigmoid = torch.sigmoid(g).to(torch.bfloat16)
        return (normed.double() * sigmoid.double()).to(torch.bfloat16).reshape(rows, width)

    def is_valid(
        self,
        attention: torch.Tensor,
        gate: torch.Tensor,
        eps: float,
        trigger: str = "after_wait",
        out: Optional[torch.Tensor] = None,
        heads_per_program: int = 1,
    ) -> None:
        """Shapes, dtypes and strides the kernel reads as if they were the built-in's, with no error.

        The kernel addresses heads of exactly 64 values at unit stride and
        writes a packed ``[rows, H * 64]`` output; anything else is read or
        written out of place.
        """
        rows = attention.shape[0] if attention.dim() == 2 else -1
        if (
            rows < 1
            or attention.dtype != torch.bfloat16
            or attention.stride(-1) != 1
            or attention.shape[1] % HEAD_DIM
        ):
            raise ValueError(
                f"attention must be BF16 [rows, H * {HEAD_DIM}] with unit last stride, got "
                f"{attention.dtype} {tuple(attention.shape)}"
            )
        if (
            gate.dtype != torch.bfloat16
            or gate.ndim not in (2, 3)
            or gate.stride(-1) != 1
            or gate.shape[0] != rows
            or gate.numel() != attention.numel()
            or (gate.ndim == 3 and gate.shape[-1] != HEAD_DIM)
        ):
            raise ValueError(
                f"gate must be BF16 [rows, H * {HEAD_DIM}] or [rows, H, {HEAD_DIM}] with unit "
                f"last stride, got {gate.dtype} {tuple(gate.shape)}"
            )
        if trigger not in TRIGGERS:
            raise ValueError(f"trigger must be one of {sorted(TRIGGERS)}, got {trigger!r}")
        if out is not None and (
            out.dtype != torch.bfloat16 or not out.is_contiguous() or out.shape != attention.shape
        ):
            raise ValueError("out must be a contiguous BF16 tensor shaped like attention")


attention_norm_gate = AttentionNormGate()
