# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Gated SiLU activation via the flashinfer silu_and_mul kernel, at this
checkpoint's widths and row counts.

Local variant of ``catalog/activation/flashinfer_silu_and_mul.py`` (``__r18``)
owned by the Hy4-Flash unified target: same op call, same reference and same
guards, with ``CELLS`` replaced by the shapes this target calls it with. The
shared entry and its cells are unchanged; this file is what the target imports.

Everything a target author needs in order to decide whether this op covers
their case is below, and all of it is checked: `reference` is what CI compares
the kernel against, and `CELLS` is what CI drives. Read the cell list for the
certified range.
"""

from __future__ import annotations

import torch
from torch.nn import functional as F

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class FlashinferSiluAndMul(OpWrapper):
    """`silu(x[..., :d]) * x[..., d:]` with `d = x.shape[-1] // 2`, new tensor out.

    The gate and up projections arrive concatenated along the last dim, which
    is why the op takes one tensor and halves it rather than taking two.
    """

    #: Driven on the run's sm_103 device only. The shared entry also carries
    #: SM_100 from its own certification; that arch is not measured here and
    #: this target's routing admits sm_103 alone.
    ARCHS = frozenset({Arch.SM_103})

    # The Hy4-Flash widths and row counts, each named by the call site that
    # reaches it: the layer-0 dense MLP's 2 x 13824, the shared expert's
    # 2 x 1536, and the drafter MLP's 2 x 9216 at its five-draft width. The
    # row counts span one decode row and the drafter's 6 rows to the 64-row
    # end and the 5120-token prefill chunk.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="decode: layer-0 dense MLP gate+up, 2 x 13824",
            spec=dict(shape=(1, 27648), dtype=torch.bfloat16),
        ),
        Cell(
            why="64-row end: layer-0 dense MLP gate+up, 2 x 13824",
            spec=dict(shape=(64, 27648), dtype=torch.bfloat16),
        ),
        Cell(
            why="prefill chunk: layer-0 dense MLP gate+up over 5120 rows",
            spec=dict(shape=(5120, 27648), dtype=torch.bfloat16),
        ),
        Cell(
            why="decode: shared expert gate+up, 2 x 1536",
            spec=dict(shape=(1, 3072), dtype=torch.bfloat16),
        ),
        Cell(
            why="prefill chunk: shared expert gate+up over 5120 rows",
            spec=dict(shape=(5120, 3072), dtype=torch.bfloat16),
        ),
        Cell(
            why="drafter MLP at the five-draft width, 2 x 9216",
            spec=dict(shape=(6, 18432), dtype=torch.bfloat16),
        ),
    )

    note = """
    Local variant (`__r18`) of `catalog/activation/flashinfer_silu_and_mul.py`,
    owned by the Hy4-Flash unified target. Same op call, reference and guards as
    the shared entry; the shared entry's 4096/36864 cells stay with it.

    Registered only when flashinfer is importable (`IS_FLASHINFER_AVAILABLE`).

    The kernel is FlashInfer's `silu_and_mul` activation
    (`flashinfer.activation`, `get_act_and_mul_module("silu")`), JIT-compiled by
    nvcc for the device it runs on and cached on disk; the cells below were
    driven on sm_103, so the JIT path that ran is the sm_103 one. The trtllm
    custom op (`tensorrt_llm/_torch/custom_ops/flashinfer_custom_ops.py`) only
    adds `enable_pdl=get_env_enable_pdl()`. PDL is controlled by
    `TRTLLM_ENABLE_PDL` (default on) inside that custom op; it changes
    scheduling, not results, and is not a cell axis.

    Measured on sm_103 against `reference` over the cells below: bit-identical
    on every cell (0 of 70,778,880 elements differ on the widest), 0.0 ulp of
    the row's largest element and 0.0 ulp of relative RMS. The test certifies at
    1.0 ulp element / 0.1 ulp RMS: both sides compute in fp32 and round once, so
    one bf16 rounding step is the most a different fp32 formulation could move
    an element, and torch's bf16 default (about 4.1 ulp) is wider than this band.

    The inherited guard (the half must be a whole number of 16-byte vectors) is
    conservative against this installed FlashInfer: its launcher picks the
    largest power-of-two vec_size dividing the half (`flashinfer/jit/activation.py`,
    `max_vec_size = 16 / sizeof(c_type)`), so any even width runs. Measured
    unguarded on sm_103 at widths 24, 8 and 2: each matched `reference`
    bit-exactly. Every half this target passes (13824, 1536, 9216 elements) is a
    multiple of 8, so the guard refuses no cell here; widening it is the shared
    entry's call, not this variant's.

    Not certified: widths other than 27648/3072/18432, rows other than
    1/64/5120/6, fp16 and fp32 inputs, and the `out=` in-place form (the trtllm
    custom op does not expose it).
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
        # second half starts mid-vector and an older kernel launch died with
        # `CUDA misaligned address`, poisoning the context rather than raising.
        # The installed FlashInfer picks a vec_size that divides the half, so
        # this is conservative there; it stays as the shared entry has it.
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
