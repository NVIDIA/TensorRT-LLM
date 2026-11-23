# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic 1x128 block-FP8 quantization of BF16 rows: e4m3 codes and one FP32 scale per 128
columns of a row, the scales returned K-block-major (`[ceil(K/128), M]`) on the sm_100 family.

Adapted from glm5_next's target entry
(`models/glm5_next/glm_5_3_flash_fp8__sm_10x__tp4/ops/quantization/fp8_quantize_1x128.py`),
which wraps the same op and whose `block_scales` -- tensor-by-tensor divisions, so every
quotient is exactly rounded -- and explicit +-448 clamp came from
`models/qwen4_exp/ops/quantization/fp8_quantize_1x128.py`. The arithmetic is unchanged; the
cells are at Hy4-Flash's widths and row counts, and this copy is certified on sm_103.
"""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Columns per scale.
BLOCK = 128

#: Largest finite e4m3 magnitude.
E4M3_MAX = 448.0

#: The kernel's amax floor, as the bf16 constant it compares against:
#: bfloat16(1e-10) == 1.000444171950221e-10.
AMAX_FLOOR = float(torch.tensor(1e-10, dtype=torch.bfloat16))

#: The q_lora width every MLA layer quantizes before `q_b`, sparse and SWA alike.
_Q_B = 1024

#: The FFN `linear_gate` input width (the routed and shared experts' hidden).
_GATE = 3072

#: `o_proj`'s input width on the sparse layers: 32 heads x 256 value channels.
_O_PROJ = 8192


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


class Fp8Quantize1x128(OpWrapper):
    """Quantize a bf16 `[M, K]` matrix to e4m3 with one fp32 scale per 1x128 row block.

    Per row `m` and block `b` of 128 consecutive columns (the last one short when
    `K % 128 != 0`): `amax = max(|x[m, block]|, bf16(1e-10))`, `data = e4m3(x * (448 / amax))`
    saturating at +-448, `scales[b, m] = 1 / (448 / amax)`. Returns `(data [M, K]
    float8_e4m3fn, scales [ceil(K/128), M] fp32)`; `data * scales.t()` (each scale repeated
    over its 128 columns) reconstructs `x`.

    The scales are TRANSPOSED, K-block-major: `[K/128, M]`, the layout
    `cute_dsl_fp8_gemm_blackwell` reads on the sm_100 family. Pass them on as returned. With
    `use_ue8m0=False` each scale is the plain `amax / 448`; `use_ue8m0=True` rounds every scale
    up to a power of two -- a different quantization of the same input, not a layout choice.

    Fusion boundary: the amax reduction, scaling, rounding and the compaction of the scales
    into `[K/128, M]` are inside; no GEMM is fused.
    """

    ARCHS = frozenset({Arch.SM_103})

    # The vocabulary's call sites are q_b's 1024 width at decode rows 1..64 and the drafter's
    # 6-row blocks; the same entry quantizes `linear_gate` (3072) and `o_proj` (8192) inputs,
    # so the cells walk those widths at the envelope's row counts.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="one decode row before q_b at 1024 wide: the vocabulary's decode call",
            spec=dict(rows=1, cols=_Q_B, use_ue8m0=False),
        ),
        Cell(
            why="64 rows before q_b: the widest decode batch the engine captures",
            spec=dict(rows=64, cols=_Q_B, use_ue8m0=False),
        ),
        Cell(
            why="6 rows before the drafter's q_b: five drafts plus the anchor",
            spec=dict(rows=6, cols=_Q_B, use_ue8m0=False),
        ),
        Cell(
            why="384 rows before the drafter's q_b: width 6 at 64 requests",
            spec=dict(rows=384, cols=_Q_B, use_ue8m0=False),
        ),
        Cell(
            why="512 rows before q_b: one chunked-prefill chunk",
            spec=dict(rows=512, cols=_Q_B, use_ue8m0=False),
        ),
        Cell(
            why="5120 rows before q_b: the prefill envelope's largest row count",
            spec=dict(rows=5120, cols=_Q_B, use_ue8m0=False),
        ),
        Cell(
            why="one decode row before linear_gate at 3072 wide",
            spec=dict(rows=1, cols=_GATE, use_ue8m0=False),
        ),
        Cell(
            why="5120 rows before linear_gate at the envelope's largest row count",
            spec=dict(rows=5120, cols=_GATE, use_ue8m0=False),
        ),
        Cell(
            why="one decode row before o_proj at 8192 wide",
            spec=dict(rows=1, cols=_O_PROJ, use_ue8m0=False),
        ),
        Cell(
            why="64 rows == 8192/128 before o_proj: the scale buffer is square",
            spec=dict(rows=64, cols=_O_PROJ, use_ue8m0=False),
        ),
        Cell(
            why="5120 rows before o_proj: the widest row count at the widest width",
            spec=dict(rows=5120, cols=_O_PROJ, use_ue8m0=False),
        ),
    )

    note = """
    Metadata consumed: none. The returned scale layout is chosen by an architecture query
    (`isSM100Family()` in thop/fp8Quantize.cpp): off the sm_100 family the op returns a 1-D
    padded buffer instead, which is why this entry is certified on sm_103 only.

    Preconditions, enforced loudly by the op: `input` bf16, 2-D (flatten `[B, S, K]` yourself),
    contiguous, CUDA, and `K % 16 == 0` -- the width, not the block count. Every Hy4-Flash
    caller is 128-aligned (1024, 3072, 8192), but the op would accept a width that is not.

    `data` is a leading-row view of an allocation padded to a multiple of 4 rows; the padded
    rows are uninitialized, so a raw-pointer reader must stop at `M`. At a row count that is a
    multiple of 4 nothing is padded and `scales` is the kernel's own buffer; otherwise it is a
    compacting copy. The decode row of 1 and the drafter's 6 rows take the padding path.

    At `M == K/128` the scales are square (8 rows at K 1024, 24 at 3072, 64 at 8192), so a
    consumer that reads them token-major passes every shape check and is silently wrong.
    Nothing here can see that; the blockwise GEMM's entry says so too.

    Source: compiled-in C++, `torch.ops.trtllm.fp8_quantize_1x128` (thop/fp8Quantize.cpp:
    `fp8CS1x128` -> `launch_scale_1x128`, which takes `scale_1x128_vec_kernel` when the input
    is 16-byte aligned and K % 8 == 0, as at every cell). The kernel's two fp32 divisions
    (`448 / amax`, then `1 / that`), its fp32 product and its saturating E4M3 conversion are
    what `block_scales` and `reference` compute: measured on an NVIDIA GB300 (sm_103) over all
    eleven cells, 100.0000% of the codes are bit-exact and no scale differs. The test's bands
    (4 fp32 ulp on a scale, at most one e4m3 code step, >=99.5% bit-exact, every miss a
    rounding tie) are slack for a `FAST_MATH=ON` build, which compiles each division within
    2 ulp.

    Decode rows 1..64 are the shapes the target captures; the test replays rows 1, 6 and 64 at
    q_b's width under CUDA-graph capture.

    Not certified: `M == 0`, a short tail block (`K % 128 != 0`), all-zero blocks (the
    `AMAX_FLOOR` scale), inf / NaN inputs, `use_ue8m0=True`, the element-wise fallback kernel
    (an input whose storage offset breaks 16-byte alignment), K 16384 (the SWA layers' `o_proj`
    input) and 9216 (the drafter's MLP down projection), and every arch but sm_103.
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
        """`(data, scales[ceil(K/128), M])` from exactly rounded fp32 divisions, as a fresh
        contiguous pair.

        No matmul, so nothing here depends on the TF32 setting.
        """
        qs, ds = block_scales(input, use_ue8m0)
        cols = qs.repeat_interleave(BLOCK, dim=1)[:, : input.shape[1]]
        data = (input.float() * cols).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
        return data, ds.t().contiguous()


fp8_quantize_1x128 = Fp8Quantize1x128()
