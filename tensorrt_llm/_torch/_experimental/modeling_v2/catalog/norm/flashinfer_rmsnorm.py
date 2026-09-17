# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RMS normalization over the last dim via the flashinfer rmsnorm kernel.

Everything a target author needs in order to decide whether this op covers
their case is below, and all of it is checked: `reference` is what CI
compares the kernel against, and `CELLS` is what CI drives. Read the cell
list for the certified range.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper


class _FlashinferRmsnorm(OpWrapper):
    """`x / sqrt(mean(x^2, -1) + eps) * weight`, one op call, new tensor out.

    Fusion boundary: normalization and weight scaling only. No residual add
    (`flashinfer_fused_add_rmsnorm`), no gemma `(1 + w)` form
    (`flashinfer_gemma_rmsnorm`), no quantization. `x` is not mutated.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The widths the shipped targets normalize over. gpt-oss reaches this op
    # only on its residual stream; deepseek reaches it there and on the two
    # lora ranks its MLA projects through, which is why the narrow cells are
    # here at all -- a 512-wide row is one 128-bit vector per lane and takes a
    # different path through the kernel than a 7168-wide one.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode-shaped, gpt-oss-120b hidden 2880",
            spec=dict(shape=(1, 2880), dtype=torch.bfloat16, eps=1e-5),
        ),
        Cell(
            why="decode-shaped, deepseek-r1 hidden 7168",
            spec=dict(shape=(1, 7168), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="prefill-shaped: enough rows to cross the kernel's grid tiling",
            spec=dict(shape=(2048, 7168), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="deepseek q_lora rank 1536, the q_norm this op is called on",
            spec=dict(shape=(16, 1536), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="deepseek kv_lora rank 512, the narrowest row any target asks for",
            spec=dict(shape=(16, 512), dtype=torch.bfloat16, eps=1e-6),
        ),
    )

    note = """
    Registered only when flashinfer is importable (`IS_FLASHINFER_AVAILABLE`).

    PDL is controlled by `TRTLLM_ENABLE_PDL` (default on) inside the trtllm
    custom op; it changes scheduling, not results, and is not a cell axis.

    fp16 and fp32 inputs are accepted by the op and were measured to work,
    but no shipped target uses them, so they are not cells. TRT-LLM's own
    `RMSNorm` module routes here for fp16/bf16 only.

    `float8_e4m3fn` is the trap this entry cannot guard: the CuTe DSL path's
    dtype table maps it and returns a plausible fp8 rmsnorm rather than
    raising, so a caller who forgets to dequantize gets a result. `is_valid`
    rejects it below. The CUDA-JIT path
    (`FLASHINFER_USE_CUDA_NORM=1`) does raise, but this entry does not
    certify that path.
    """

    def __call__(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        return torch.ops.trtllm.flashinfer_rmsnorm(x, weight, eps)

    def reference(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
        """fp32-accumulated; the kernel reduces in fp32 and rounds once at the end."""
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (normed * weight.float()).to(x.dtype)

    def is_valid(self, x: torch.Tensor, weight: torch.Tensor, eps: float) -> None:
        # Only the case the op takes and answers wrongly. Shape and dtype
        # mismatches it rejects itself, loudly, and duplicating that here
        # would only swap its error for a worse one.
        assert x.dtype is not torch.float8_e4m3fn, (
            "float8_e4m3fn is accepted by the kernel's dtype table and returns "
            "a plausible fp8 rmsnorm; dequantize before normalizing"
        )
        assert x.stride(-1) == 1, (
            f"the normalized dim must be contiguous; got stride {x.stride(-1)}"
        )


flashinfer_rmsnorm = _FlashinferRmsnorm()
