# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic 1x128 block-wise FP8 quantization: bf16 -> e4m3 data + transposed fp32 block scales."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

#: Largest finite e4m3 magnitude.
E4M3_MAX = 448.0

#: The kernel's amax floor, as the bf16 constant it compares against:
#: bfloat16(1e-10) == 1.000444171950221e-10.
AMAX_FLOOR = float(torch.tensor(1e-10, dtype=torch.bfloat16))

#: The two divisions in the kernel are approximate (`--use_fast_math`), each
#: within CUDA's documented 2 ulp; the returned scale is two deep.
SCALE_ULP_GATE = 4


def block_scales(input: torch.Tensor, use_ue8m0: bool) -> tuple[torch.Tensor, torch.Tensor]:
    """`(quant_scale, dequant_scale)`, each `[T, ceil(H/128)]` fp32, exactly rounded.

    The amax runs in bf16, as the kernel's running maximum does (exact: |x| of
    a bf16 is a bf16), floored at `AMAX_FLOOR`; a short tail block takes the
    amax of its own elements. The kernel multiplies by the first before
    rounding to e4m3 and returns the second.
    """
    t, n = input.shape
    nb = (n + 127) // 128
    mag = torch.nn.functional.pad(input.abs(), (0, nb * 128 - n))
    amax = mag.reshape(t, nb, 128).amax(dim=2).float().clamp(min=AMAX_FLOOR)
    qs = E4M3_MAX / amax
    if not use_ue8m0:
        return qs, 1.0 / qs
    raw = 1.0 / qs
    mant, exp = torch.frexp(raw)
    ds = torch.ldexp(torch.ones_like(raw), torch.where(mant == 0.5, exp - 1, exp))
    return 1.0 / ds, ds


#: The MTP draft layer's hidden width, quantized with the plain scale the
#: block-scale MoE runner wants.
_CKPT = dict(hidden=2560, use_ue8m0=False)


class Fp8Quantize1x128(OpWrapper):
    """Quantize a bf16 `[T, H]` matrix to e4m3 with one fp32 scale per 1x128 row block.

    Per token `t` and block `b` of 128 consecutive columns (the last one short
    when `H % 128 != 0`): `amax = max(|x[t, block]|, bf16(1e-10))`,
    `data = e4m3(x * (448 / amax))` saturating at +-448, `scale[b, t] =
    1 / (448 / amax)`. Returns `(data [T, H] float8_e4m3fn, scales [ceil(H/128),
    T] fp32)`.

    Three readings are each a silent wrong answer: `scales` is TRANSPOSED
    (`[NB, T]`, the layout `fp8_block_scale_moe_runner` consumes); it
    MULTIPLIES the code back (dividing is 6780 relative RMS off); and with
    `use_ue8m0=False` it is the plain `amax / 448`, while `use_ue8m0=True`
    rounds every scale up to a power of two -- a different quantization of the
    same input (99% of scales, 96% of codes change), not a layout choice.

    Fusion boundary: the amax reduction, scaling, rounding and (on the sm_100
    family) the compaction into `[NB, T]` are inside; no GEMM is fused.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        Cell(why="MTPLayer._moe decode: one token at hidden 2560", spec=dict(tokens=1, **_CKPT)),
        Cell(
            why="20 tokens == H/128: the scale buffer is square, its transpose undetectable",
            spec=dict(tokens=20, **_CKPT),
        ),
        Cell(
            why="16384 rows: max_num_tokens of the tuned config",
            spec=dict(tokens=16384, **_CKPT),
        ),
        Cell(
            why="128 rows: the MTP drafter's decode step at max_batch_size 128",
            spec=dict(tokens=128, **_CKPT),
        ),
        Cell(
            why="1024 rows: the first draft over mtp7's verify rows, 128 requests x 8",
            spec=dict(tokens=1024, **_CKPT),
        ),
        Cell(
            why="8192 rows: the MTP layer over one max_num_tokens context chunk",
            spec=dict(tokens=8192, **_CKPT),
        ),
    )

    note = """
    Metadata consumed: none. The returned scale layout is chosen by an
    architecture query (`isSM100Family()` in thop/fp8Quantize.cpp): off the
    sm_100 family the op returns a 1-D padded buffer instead. That branch was
    read, not run.

    Preconditions, enforced loudly by the op: `input` bf16, 2-D (flatten
    `[B, S, H]` yourself), contiguous (the test holds the column-slice
    rejection), CUDA, and `H % 16 == 0` -- the width, not the block count, so
    H need not be a multiple of 128. A consumer needing `H % 128 == 0` (the
    block-scale MoE runner) owns that constraint.

    `data` is a leading-row view of an allocation padded to a multiple of 4
    rows -- the padded rows are uninitialized, so a raw-pointer reader must
    stop at `T`. At a `T` that is a multiple of 4 (128, 1024, 8192) nothing is
    padded and `scales` is the kernel's own buffer, not a compacting copy.
    CUDA-graph capture and replay are certified at 64, 128 and 1024 rows.

    Source: compiled-in C++, `torch.ops.trtllm.fp8_quantize_1x128` from
    `thop/fp8Quantize.cpp` (`fp8_1x128_cs` -> `launch_scale_1x128`). The
    launcher takes `scale_1x128_vec_kernel` when `can_vectorize_1x128` holds
    (H % 8 == 0, 16-byte aligned input, 8-byte aligned output), as at every
    cell; the element-wise `scale_1x128_kernel` it falls back to (an input
    whose storage offset breaks the 16-byte alignment) is not certified. The
    grid is persistent, `kNumDeviceSMs * 8` blocks of 8 warps quantizing 2
    groups each: 19456 groups a sweep on GB300's 152 SMs. 128 rows (2560
    groups) fills 13% of one sweep, 1024 rows (20480) is the first certified
    count past one, 8192 rows (163840) takes nine.

    Measured on GB300 at H 2560 (codes bit-exact; worst miss from its rounding
    midpoint, relative): 1 row 100%; 20 rows 99.9629%, 1.122e-07; 128 rows
    99.9451%; 1024 rows 99.9437%; 8192 rows 99.9429%; 16384 rows 99.9453%;
    every row count above one at 1.122e-07, scales within the 4-ulp gate.

    Hard limits
      T   certified at 1, 20, 128, 1024, 8192 and 16384 at H 2560 (128: the
          drafter's decode at max_batch_size 128; 1024: the first draft over
          mtp7's 128 x 8 verify rows; 8192: a max_num_tokens context chunk;
          16384: max_num_tokens of the tuned config); above that not probed
      H   multiple of 16 (raises otherwise); certified at 2560, the draft layer's
          width

    Not certified: `T == 0` (the op accepts it and returns `[0, H]` /
    `[NB, 0]`); a short tail block (`H % 128 != 0`); all-zero blocks (the
    `AMAX_FLOOR` scale), `inf` / NaN inputs and saturation; `use_ue8m0=True`;
    the element-wise fallback kernel; the registered fake.
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
        """`(data, scales[NB, T])` from exactly-rounded fp32 divisions."""
        qs, ds = block_scales(input, use_ue8m0)
        cols = qs.repeat_interleave(128, dim=1)[:, : input.shape[1]]
        data = (input.float() * cols).to(torch.float8_e4m3fn)
        return data, ds.t().contiguous()


fp8_quantize_1x128 = Fp8Quantize1x128()
