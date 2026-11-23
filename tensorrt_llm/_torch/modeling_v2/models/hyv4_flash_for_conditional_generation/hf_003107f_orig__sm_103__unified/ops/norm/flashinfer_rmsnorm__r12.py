# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RMS normalization over the last dim via the flashinfer rmsnorm kernel, at
this checkpoint's widths and row counts.

Local variant of ``catalog/norm/flashinfer_rmsnorm.py`` (``__r12``) owned by the
Hy4-Flash unified target: same op call, same reference and same guards, with
``CELLS`` replaced by the shapes this target calls it with. The shared entry and
its cell are unchanged; this file is what the target imports.

Everything a target author needs in order to decide whether this op covers
their case is below, and all of it is checked: `reference` is what CI compares
the kernel against, and `CELLS` is what CI drives. Read the cell list for the
certified range.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class FlashinferRmsnorm(OpWrapper):
    """`x / sqrt(mean(x^2, -1) + eps) * weight`, one op call, new tensor out.

    Fusion boundary: normalization and weight scaling only. No residual add
    (`flashinfer_fused_add_rmsnorm`), no gemma `(1 + w)` form
    (`flashinfer_gemma_rmsnorm`), no quantization. `x` is not mutated.
    """

    #: Driven on the run's sm_103 device only. The shared entry also carries
    #: SM_100 from its own certification; that arch is not measured here and
    #: this target's routing admits sm_103 alone.
    ARCHS = frozenset({Arch.SM_103})

    # The Hy4-Flash widths and row counts, each named by the call site that
    # reaches it. The widths span two very different kernel paths: a 96-wide
    # row is three 128-bit vectors per lane and a 16384-wide one is a long
    # streaming reduction, and the row counts span one decode row to the
    # 81920 head-rows the Engram branch normalizes at a 5120-token chunk.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: q_a_layernorm over the q_lora rank (q_a_proj 3072->1024)",
            spec=dict(shape=(1, 1024), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="64-row end: q_a_layernorm at the q_lora rank",
            spec=dict(shape=(64, 1024), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="prefill chunk: q_a_layernorm over 5120 rows",
            spec=dict(shape=(5120, 1024), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="decode: kv_a_layernorm over the kv_lora rank",
            spec=dict(shape=(1, 512), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="prefill chunk: kv_a_layernorm over 5120 rows",
            spec=dict(shape=(5120, 512), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="decode: post_proj_norm over 32 sparse heads x v_head 256",
            spec=dict(shape=(1, 8192), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="prefill chunk: post_proj_norm at 8192",
            spec=dict(shape=(5120, 8192), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="decode: post_proj_norm over 64 SWA heads x v_head 256",
            spec=dict(shape=(1, 16384), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="prefill chunk: post_proj_norm at 16384, the widest row",
            spec=dict(shape=(5120, 16384), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="decode: model.norm / combine_output_norm at hidden 3072",
            spec=dict(shape=(1, 3072), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="prefill chunk: model.norm at hidden 3072",
            spec=dict(shape=(5120, 3072), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="engram: 16 per-head norms over 96 at a 5120-token chunk, the most rows",
            spec=dict(shape=(81920, 96), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="indexer: token_key_norm/key_norm over 5120 context rows",
            spec=dict(shape=(5120, 128), dtype=torch.bfloat16, eps=1e-6),
        ),
        Cell(
            why="indexer: key_norm over the 5120/4 pooled groups",
            spec=dict(shape=(1280, 128), dtype=torch.bfloat16, eps=1e-6),
        ),
    )

    note = """
    Local variant (`__r12`) of `catalog/norm/flashinfer_rmsnorm.py`, owned by the
    Hy4-Flash unified target. The shared entry's own cell, shape (2048, 7168),
    stays with the shared entry: 7168 is not a width this checkpoint has.

    Registered only when flashinfer is importable (`IS_FLASHINFER_AVAILABLE`).

    The default path is FlashInfer's CuTe DSL rmsnorm, JIT-compiled for the
    device it runs on; the cells below were driven on sm_103. The CUDA-JIT path
    (`FLASHINFER_USE_CUDA_NORM=1`) is not certified.

    PDL is controlled by `TRTLLM_ENABLE_PDL` (default on) inside the trtllm
    custom op; it changes scheduling, not results, and is not a cell axis.

    Measured on sm_103 against `reference` over the cells below: worst element
    error 1.36 ulp of the row's largest element and 0.0053 ulp of relative RMS
    (1.06 / 0.0045 with the same cells re-seeded), cosine >= 0.9999999 in fp32.
    The test certifies at 3 ulp element / 0.5 ulp RMS.

    fp16 and fp32 inputs are accepted by the op; the shared entry records that
    they were measured to work. No shipped target uses them, so they are not
    cells here and this variant does not re-measure them. TRT-LLM's own
    `RMSNorm` module routes here for fp16/bf16 only.

    `float8_e4m3fn` is the trap this entry cannot guard: the CuTe DSL path's
    dtype table maps it and returns a plausible fp8 rmsnorm rather than
    raising, so a caller who forgets to dequantize gets a result. `is_valid`
    rejects it below. The CUDA-JIT path does raise, but this entry does not
    certify that path.

    Not certified: eps other than 1e-6, widths outside the 96/128/512/1024/
    3072/8192/16384 set above, and row counts above 81920.
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


flashinfer_rmsnorm = FlashinferRmsnorm()
