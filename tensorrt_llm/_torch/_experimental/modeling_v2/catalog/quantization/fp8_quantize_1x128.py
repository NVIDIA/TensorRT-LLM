# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic 1x128 block-FP8 quantization of BF16 rows: e4m3 codes and one FP32 scale per 128
columns of a row, the scales returned K-block-major (`[ceil(K/128), M]`) on the sm_100 family."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

#: Columns per scale.
BLOCK = 128

#: Largest finite e4m3 magnitude.
E4M3_MAX = 448.0

#: The kernel's amax floor, as the bf16 constant it compares against:
#: bfloat16(1e-10) == 1.000444171950221e-10.
AMAX_FLOOR = float(torch.tensor(1e-10, dtype=torch.bfloat16))


def block_scales(input: torch.Tensor, use_ue8m0: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """`(quant_scale, dequant_scale)`, each `[M, ceil(K/128)]` fp32, exactly rounded.

    The amax runs in bf16, as the kernel's running maximum does (exact: `|x|` of a bf16 is a
    bf16), floored at `AMAX_FLOOR`; a short tail block takes the amax of its own elements. The
    kernel multiplies by the first before rounding to e4m3 and returns the second.

    Every quotient is a tensor-by-tensor division: torch evaluates `scalar / tensor` as a
    reciprocal times the scalar, which is two roundings and can sit one ulp off the quotient.
    """
    rows, cols = input.shape
    nb = (cols + BLOCK - 1) // BLOCK
    mag = torch.nn.functional.pad(input.abs(), (0, nb * BLOCK - cols))
    amax = mag.reshape(rows, nb, BLOCK).amax(dim=2).float().clamp(min=AMAX_FLOOR)
    one = torch.ones_like(amax)
    qs = torch.full_like(amax, E4M3_MAX) / amax
    if not use_ue8m0:
        return qs, one / qs
    raw = one / qs
    mant, exp = torch.frexp(raw)
    ds = torch.ldexp(one, torch.where(mant == 0.5, exp - 1, exp))
    return one / ds, ds


#: GLM-5.3-Flash's hidden width, the input of every routed and shared FFN GEMM.
_GLM_HIDDEN = 4096


class Fp8Quantize1x128(OpWrapper):
    """Quantize a bf16 `[M, K]` matrix to e4m3 with one fp32 scale per 1x128 row block.

    Per row `m` and block `b` of 128 consecutive columns (the last one short when
    `K % 128 != 0`): `amax = max(|x[m, block]|, bf16(1e-10))`, `data = e4m3(x * (448 / amax))`
    saturating at +-448, `scales[b, m] = 1 / (448 / amax)`. Returns `(data [M, K]
    float8_e4m3fn, scales [ceil(K/128), M] fp32)`; `data * scales.t()` (each scale repeated
    over its 128 columns) reconstructs `x`.

    The scales are TRANSPOSED, K-block-major: `[K/128, M]`, the layout
    `cute_dsl_fp8_gemm_blackwell` and `fp8_block_scale_moe_runner` read on the sm_100 family.
    Pass them on as returned. With `use_ue8m0=False` each scale is the plain `amax / 448`;
    `use_ue8m0=True` rounds every scale up to a power of two -- a different quantization of
    the same input, not a layout choice.

    Fusion boundary: the amax reduction, scaling, rounding and the compaction of the scales
    into `[K/128, M]` are inside; no GEMM is fused.
    """

    ARCHS = frozenset({Arch.SM_100})

    # Only GLM-5.3-Flash calls this entry, so its own cells are the op's row-count edges at
    # GLM's hidden width; the widths and row counts the target quantizes are in the test.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one row: the decode row count, at GLM's hidden width",
            spec=dict(rows=1, cols=_GLM_HIDDEN, use_ue8m0=False),
        ),
        Cell(
            why="5 rows: not a multiple of 4, so data is a view of a padded allocation and the "
            "scales a compacting copy",
            spec=dict(rows=5, cols=_GLM_HIDDEN, use_ue8m0=False),
        ),
        Cell(
            why="32 rows == K/128: the scale buffer is square, so its transpose is undetectable",
            spec=dict(rows=32, cols=_GLM_HIDDEN, use_ue8m0=False),
        ),
    )

    note = """
    Metadata consumed: none. The returned scale layout is chosen by an architecture query
    (`isSM100Family()` in thop/fp8Quantize.cpp): off the sm_100 family the op returns a 1-D
    padded buffer instead, which is why this entry is certified on sm_100 only.

    Preconditions, enforced loudly by the op: `input` bf16, 2-D (flatten `[B, S, K]` yourself),
    contiguous, CUDA, and `K % 16 == 0` -- the width, not the block count. A consumer needing
    `K % 128 == 0` (the block-scale MoE runner, the blockwise GEMM at every GLM shape) owns
    that constraint.

    `data` is a leading-row view of an allocation padded to a multiple of 4 rows; the padded
    rows are uninitialized, so a raw-pointer reader must stop at `M`. At a row count that is a
    multiple of 4 nothing is padded and `scales` is the kernel's own buffer; otherwise it is a
    compacting copy.

    At `M == K/128` the scales are square (32 rows at K 4096, 24 at 3072, 12 at 1536, 4 at
    512), so a consumer that reads them token-major passes every shape check and is silently
    wrong. Nothing here can see that; the consumers' entries say so too.

    Source: compiled-in C++, `torch.ops.trtllm.fp8_quantize_1x128` (thop/fp8Quantize.cpp:
    `fp8CS1x128` -> `launch_scale_1x128`, which takes `scale_1x128_vec_kernel` when the input
    is 16-byte aligned and K % 8 == 0, as at every cell). The kernel's two fp32 divisions
    (`448 / amax`, then `1 / that`) are compiled approximately, so neither output is
    reproducible to the last bit: the test holds scales to a few fp32 ulp of the exact ones and
    every code disagreement to a rounding tie.

    Not certified: `M == 0`, a short tail block (`K % 128 != 0`), all-zero blocks (the
    `AMAX_FLOOR` scale), inf / NaN inputs, `use_ue8m0=True`, the element-wise fallback kernel
    (an input whose storage offset breaks 16-byte alignment), CUDA-graph capture, sm_103.
    """

    def __call__(
        self,
        input: torch.Tensor,
        use_ue8m0: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.ops.trtllm.fp8_quantize_1x128(input, use_ue8m0)

    def reference(
        self,
        input: torch.Tensor,
        use_ue8m0: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """`(data, scales[ceil(K/128), M])` from exactly rounded fp32 divisions.

        No matmul, so nothing here depends on the TF32 setting.
        """
        qs, ds = block_scales(input, use_ue8m0)
        cols = qs.repeat_interleave(BLOCK, dim=1)[:, : input.shape[1]]
        data = (input.float() * cols).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        return data, ds.t().contiguous()


fp8_quantize_1x128 = Fp8Quantize1x128()
