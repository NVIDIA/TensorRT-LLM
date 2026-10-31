# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated SiLU activation via the flashinfer silu_and_mul kernel.

Tier 1: the entry wraps `torch.ops.trtllm.flashinfer_silu_and_mul`. The qwen4_exp
shared-expert width (2 x 640) was merged in from the unified target's variant r44.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper


class FlashinferSiluAndMul(OpWrapper):
    """`silu(x[..., :d]) * x[..., d:]` with `d = x.shape[-1] // 2`, new tensor out.

    The gate and up projections arrive concatenated along the last dim, which
    is why the op takes one tensor and halves it rather than taking two.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    # deepseek-r1 is the only shipped caller. It reaches this op from two
    # widths -- the routed/shared expert intermediate and the dense layers'
    # -- at decode and prefill token counts.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode-shaped, 2 x moe_intermediate (2048)",
            spec=dict(shape=(1, 4096), dtype=torch.bfloat16),
        ),
        Cell(
            why="prefill-shaped: enough rows to cross the kernel's grid tiling",
            spec=dict(shape=(2048, 4096), dtype=torch.bfloat16),
        ),
        Cell(
            why="the dense layers' wider intermediate, 2 x 18432",
            spec=dict(shape=(16, 36864), dtype=torch.bfloat16),
        ),
    )

    note = """
    Registered only when flashinfer is importable (`IS_FLASHINFER_AVAILABLE`).

    fp16 and fp32 are accepted by the op. No shipped target uses them here, so
    they are not cells.

    Source: `trtllm::flashinfer_silu_and_mul` is a Python `torch.library`
    custom op (`_torch/custom_ops/flashinfer_custom_ops.py`) calling
    `flashinfer.activation.silu_and_mul(x, enable_pdl=get_env_enable_pdl())`;
    PDL follows `TRTLLM_ENABLE_PDL` (default on) and changes scheduling, not
    results. FlashInfer JIT-compiles the kernel on the first call in a process
    (`flashinfer.jit.gen_act_and_mul_module("silu")`): this installation has no
    AOT artifact for it, and on GB300 the build ran for the device's own
    architecture, `-gencode=arch=compute_103a,code=sm_103a`
    (`FLASHINFER_CUDA_ARCH_LIST` unset).

    Numerics: the kernel's SiLU is `g / (1 + __expf(-g))` in fp32, times `up`,
    rounded once to the input dtype. Measured against `reference` at width 1280
    (40 draws per row count, randn at scale 1 and 4): bitwise at 1 and 128 rows;
    1 of 26.2M elements at 1024 rows and 4 of 209.7M at 8192 rows differ, each by
    one bf16 step (relative L2 at most 6.6e-12). Torch's bf16 default band holds
    that and refuses swapped halves and GELU-tanh (the test's negative control).
    Gates at or above 20 give `g * up` exactly; at -100 `__expf` overflows and
    the output is a signed zero, as the reference's denormal rounds to in bf16.

    The qwen4_exp shared-expert cells (width 1280, rows 1, 128, 1024, 8192; the
    tp1 target's trunk and MTP shared expert) and their CUDA-graph replay at 1,
    128 and 1024 rows were driven on sm_103 only; width 1280 is unmeasured on
    sm_100.
    """

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.trtllm.flashinfer_silu_and_mul(x)

    def reference(self, x: torch.Tensor) -> torch.Tensor:
        """fp32-accumulated; the kernel computes in fp32 and rounds once."""
        gate, up = x.float().chunk(2, dim=-1)
        return (F.silu(gate) * up).to(x.dtype)

    def is_valid(self, x: torch.Tensor) -> None:
        # The op's own check is on the *row*, `x.shape[-1] * itemsize % 16 == 0`,
        # which a final dimension of 24 passes. The kernel vectorizes over the
        # two halves, so what has to be 16-byte aligned is the *half*: at 24 the
        # second half starts mid-vector and the launch dies with `CUDA
        # misaligned address`, poisoning the context rather than raising.
        width = x.shape[-1]
        assert width % 2 == 0, f"x.shape[-1] must be even to split in half; got {width}"
        half_bytes = (width // 2) * x.element_size()
        assert half_bytes % 16 == 0, (
            f"x.shape[-1] // 2 must be a whole number of 16-byte vectors; "
            f"{width} halves to {half_bytes} bytes, which is not a multiple of 16 "
            f"-- the kernel would fault on the misaligned second half"
        )
        assert half_bytes >= 16, (
            f"x.shape[-1] // 2 must hold at least one 16-byte vector; {width} "
            f"gives {half_bytes} bytes, for which the computed block size is 0"
        )


flashinfer_silu_and_mul = FlashinferSiluAndMul()
