# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The drafter MLP's ``silu(gate) * up`` as a programmatic-dependent-launch copy of the built-in kernel.

The DFlash 2 drafter's ``GatedMLP`` runs the built-in ``torch.ops.trtllm.silu_and_mul``
between its ``gate_up_proj`` and ``down_proj`` GEMMs: the Triton
``silu_and_mul_kernel`` of ``tensorrt_llm/_torch/modules/swiglu.py``, launched
without programmatic dependent launch. A plain launch between two GEMMs that
are launched with it is a full dependency on both sides: the down projection
starts streaming its weight only after this kernel has finished.

This entry is that kernel with the same statements, launched the same way
(grid ``(rows, cdiv(d, 1024))``, 1024 columns per program, four warps), plus
the launch protocol of the PDL chain's ``pdl_silu``: launched with programmatic dependent
launch, it triggers its dependents and then executes ``griddepcontrol.wait``
before its first load. The arithmetic is the built-in kernel's, statement for
statement, so the output is bitwise the built-in output; the test gates that
with ``torch.equal`` against ``torch.ops.trtllm.silu_and_mul``.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Columns per program and warps per program: the built-in launch's
#: (``torch_custom_ops.silu_and_mul``: ``BLOCK_SIZE=1024``, Triton's default four warps).
BLOCK_SIZE = 1024
NUM_WARPS = 4


@triton.jit
def _silu_and_mul_pdl_kernel(
    o_ptr,
    o_stride,
    x_ptr,
    x_stride,
    d,
    swiglu_alpha: tl.constexpr,
    swiglu_beta: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ENTRY_TRIGGER: tl.constexpr,
) -> None:
    if ENTRY_TRIGGER:
        # Release the dependent (a GEMM that streams its weight before its own
        # wait), then wait for the producer before the first load.
        tl.extra.cuda.gdc_launch_dependents()
        tl.extra.cuda.gdc_wait()
    # From here on the built-in silu_and_mul_kernel's statements, without its
    # output-scale and limit branches (HAS_O_SCALE and HAS_SWIGLU_LIMIT off).
    i = tl.program_id(axis=0).to(tl.int64)
    j = tl.program_id(axis=1)

    o_row_ptr = o_ptr + o_stride * i
    x_row_ptr = x_ptr + x_stride * i

    offsets = j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < d

    a = tl.load(x_row_ptr + offsets, mask=mask).to(tl.float32)
    b = tl.load(x_row_ptr + offsets + d, mask=mask).to(tl.float32)

    result = a * tl.sigmoid(swiglu_alpha * a) * (b + swiglu_beta)

    tl.store(o_row_ptr + offsets, result, mask=mask)


def takes(x: torch.Tensor) -> bool:
    """Whether this entry computes ``torch.ops.trtllm.silu_and_mul(x)`` for this operand."""
    return (
        x.dim() == 2
        and x.is_cuda
        and x.dtype in (torch.bfloat16, torch.float16, torch.float32)
        and x.shape[1] % 2 == 0
        and x.stride(1) == 1
    )


class DrafterSiluAndMul(OpWrapper):
    """``out[r, c] = silu(x[r, c]) * x[r, d + c]`` for ``x`` ``[rows, 2d]``; the built-in kernel's bits.

    Arguments: ``x`` ``[rows, 2d]`` (row stride free, unit column stride),
    ``entry_trigger``: launch with programmatic dependent launch, release the
    dependent at entry and wait for the producer before the first load
    (the PDL chain's ``pdl_silu``). Returns a fresh ``[rows, d]`` tensor in ``x.dtype``. Only the
    plain SwiGLU (``swiglu_alpha`` 1, ``swiglu_beta`` 0, no limit, no output
    scale), which the drafter runs.

    Fusion boundary: the activation only; both GEMMs stay outside.
    """

    ARCHS = frozenset({Arch.SM_103})

    # spec: rows, d (half width), dtype, entry_trigger, specials (inf / nan / signed zeros).
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="the drafter's served call: 8 rows (K = 7 block) x 3584 per TP8 rank, entry trigger",
            spec=dict(rows=8, d=3584, entry_trigger=True),
        ),
        Cell(
            why="the same call without the trigger (a plain launch)",
            spec=dict(rows=8, d=3584, entry_trigger=False),
        ),
        Cell(
            why="one row; d not a multiple of 1024 (masked tail of the last program)",
            spec=dict(rows=1, d=1000, entry_trigger=True),
        ),
        Cell(
            why="infinities, NaNs, signed zeros and large magnitudes in gate and up",
            spec=dict(rows=8, d=3584, entry_trigger=True, specials=True),
        ),
        Cell(
            why="fp16 operands", spec=dict(rows=4, d=2048, dtype=torch.float16, entry_trigger=True)
        ),
        Cell(
            why="fp32 operands", spec=dict(rows=4, d=2048, dtype=torch.float32, entry_trigger=True)
        ),
        Cell(
            why="0 rows: an empty result, no launch", spec=dict(rows=0, d=3584, entry_trigger=True)
        ),
    )

    note = """
    Metadata consumed: none. One Triton kernel, JIT-compiled per
    specialization: grid `(rows, cdiv(d, 1024))`, four warps, as the built-in
    `torch.ops.trtllm.silu_and_mul` launches `silu_and_mul_kernel`.

    Call site: the drafter's `GatedMLP._apply_activation` once re-classed by
    the PDL chain's `pdl_silu` (`drafter_pdl_chain/chain.py`), on 8-row blocks whose
    `down_proj` runs the `drafter_fixed_gemms` split-K tactic (which loads only its weight before its own
    `griddepcontrol.wait`).

    Numerics: bitwise the built-in kernel (same statements, same constexpr
    values: alpha 1.0, beta 0.0). `reference` is fp32 with one rounding.

    Hard limits
      unit column stride, even width (`takes`).
    """

    def __call__(self, x: torch.Tensor, entry_trigger: bool) -> torch.Tensor:
        rows, width = x.shape
        d = width // 2
        out = torch.empty((rows, d), dtype=x.dtype, device=x.device)
        if rows == 0:
            return out
        _silu_and_mul_pdl_kernel[(rows, triton.cdiv(d, BLOCK_SIZE))](
            out,
            out.stride(0),
            x,
            x.stride(0),
            d,
            swiglu_alpha=1.0,
            swiglu_beta=0.0,
            BLOCK_SIZE=BLOCK_SIZE,
            ENTRY_TRIGGER=entry_trigger,
            num_warps=NUM_WARPS,
            launch_pdl=entry_trigger,
        )
        return out

    def reference(self, x: torch.Tensor, entry_trigger: bool) -> torch.Tensor:
        """fp32 ``silu(gate) * up``, one rounding to ``x.dtype``. ``entry_trigger`` is ignored."""
        d = x.shape[1] // 2
        gate, up = x[:, :d].float(), x[:, d:].float()
        return (gate * torch.sigmoid(gate) * up).to(x.dtype)

    def is_valid(self, x: torch.Tensor, entry_trigger: bool) -> None:
        assert takes(x), (
            "x must be a [rows, 2d] CUDA tensor of bf16 / fp16 / fp32 with unit column stride, "
            f"got {tuple(x.shape)} {x.dtype} stride {x.stride()}"
        )


drafter_silu_and_mul = DrafterSiluAndMul()
