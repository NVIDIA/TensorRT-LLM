# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dynamic 1x128 block-wise FP8 quantization: bf16 -> e4m3 data + transposed fp32 block scales."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

from .._op import Arch, Cell, OpWrapper

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


#: Mistral Large 4's routed-expert input: the full hidden width (the MoE input is
#: replicated across the TP4 attention ranks), the plain scale
#: `TRTLLMGenFp8BlockScalesBase.run_moe` asks for by leaving `use_ue8m0` at its
#: default.
_ML4 = dict(hidden=6144, use_ue8m0=False)


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

    At Mistral Large 4's routed-expert input the call is `[T, 6144]` bf16, T the
    step's token count (1..16 in captured decode, up to 9216 eager), and both
    outputs go straight to the block-scale MoE runner. At T == 48 the scale
    buffer is `[48, 48]`, so a transposed buffer passes every shape check.
    """

    ARCHS = frozenset({Arch.SM_103})

    CELLS: tuple[Cell, ...] = (
        # Mistral Large 4's routed-expert input, H 6144. Decode steps are captured
        # at the padded batches 1/2/4/8/16; each captured cell replays on new
        # data and must equal an eager call bitwise.
        Cell(
            why="Mistral Large 4 decode, captured batch 1: replays bitwise eager",
            spec=dict(tokens=1, captured=True, **_ML4),
        ),
        Cell(
            why="Mistral Large 4 decode, captured batch 2: replays bitwise eager",
            spec=dict(tokens=2, captured=True, **_ML4),
        ),
        Cell(
            why="Mistral Large 4 decode, captured batch 3: an odd batch of the unpadded 1..16 default",
            spec=dict(tokens=3, captured=True, **_ML4),
        ),
        Cell(
            why="Mistral Large 4 decode, captured batch 4: replays bitwise eager",
            spec=dict(tokens=4, captured=True, **_ML4),
        ),
        Cell(
            why="Mistral Large 4 decode, captured batch 8: replays bitwise eager",
            spec=dict(tokens=8, captured=True, **_ML4),
        ),
        Cell(
            why="Mistral Large 4 decode, captured batch 16 (max_batch_size): replays bitwise eager",
            spec=dict(tokens=16, captured=True, **_ML4),
        ),
        Cell(
            why="Mistral Large 4 T 48 == 6144/128: the [48, 48] scale buffer, transpose undetectable",
            spec=dict(tokens=48, **_ML4),
        ),
        Cell(
            why=(
                "Mistral Large 4 every eager T 1..9216 (max_num_tokens), each bitwise the leading "
                "rows of the 9216-row call: every 4-row pad phase, and the grid-stride wrap "
                "at T 203 on 152 SMs; about 2 s"
            ),
            spec=dict(tokens=9216, sweep=True, **_ML4),
        ),
        Cell(
            why=(
                "Mistral Large 4 T 9216 with 12 outlier channels at 150x: blocks whose amax one "
                "channel sets, the rest in e4m3's subnormal codes"
            ),
            spec=dict(tokens=9216, dist="outlier", **_ML4),
        ),
    )

    note = """
    Metadata consumed: none. The returned scale layout is chosen by an
    architecture query (`isSM100Family()` in thop/fp8Quantize.cpp): off the
    sm_100 family the op returns a 1-D padded buffer instead. That branch was
    read, not run.

    Preconditions, all enforced loudly by the op: `input` bf16 (fp16, fp32 and
    fp8 raise), 2-D (flatten `[B, S, H]` yourself), contiguous, CUDA, and
    `H % 16 == 0` -- the width, not the block count, so H need not be a
    multiple of 128. `T == 0` is accepted and returns `[0, H]` / `[NB, 0]`.
    A consumer needing `H % 128 == 0` (the block-scale MoE runner) owns that
    constraint.

    One precondition the op does not check: the input's base address must be
    4-byte aligned, because the kernel loads bf16 pairs. Measured on sm_103 in
    a child process: a contiguous `[16, 6144]` view one element (2 bytes) into
    its storage faults with `torch.AcceleratorError: CUDA error: misaligned
    address`, and the process's CUDA context is unusable afterwards; views 2
    and 8 elements in compute exactly what an aligned copy does. `is_valid`
    refuses the odd base. A fresh activation, or a row slice of one (H is a
    multiple of 16), is aligned.

    Launch (read in `fp8_1x128_cs`): a fixed grid of 8 blocks of 256 threads
    per SM, one warp per (token, block) and a grid-stride loop over the rest,
    with no PDL attribute. On sm_103 that is 9728 warps, so at H
    6144 (48 blocks per token) one pass covers T <= 202 and larger T loop.
    Measured: every T from 1 to 9216 at H 6144 is bitwise the leading rows of
    the 9216-row call, codes and scales -- nothing depends on the batch.

    Edge behaviour, measured: an all-zero block takes scale
    `bf16(1e-10) / 448` and all-zero codes; `inf` poisons its whole block
    (scale inf, its own code NaN, every other code 0); NaN stays in its own
    element (the running max ignores it); the block max maps to exactly +-448
    and nothing overflows. `data` is a leading-row view of an allocation padded
    to a multiple of 4 rows -- the padded rows are uninitialized, so a
    raw-pointer reader must stop at `T`. CUDA-graph capturable (replays at T
    1..16, H 6144, bitwise equal to eager); stream-independent; the registered
    fake matches eager shapes and strides.

    Accepted by the op but not certified here: `use_ue8m0=True`; `T == 0`;
    any H other than 6144, a short tail block included.

    Hard limits (sm_103)
      T             int64 on the host; the op raises above int32 ("M must be
                    within int32"), read, not probed; certified every T
                    1..9216 at H 6144
      H             multiple of 16 (raises otherwise); certified at 6144 only
      data offset   token * H as size_t in the kernel; no 32-bit product
      scale offset  block * ceil4(T) + token as size_t; no 32-bit product
      scale size    ceil4(T) * ceil(H/128) formed in 32-bit int when the
                    buffer is sized (getActScaleSize): 2^31 elements, i.e.
                    T >= 44,739,241 at H 6144 -- not reached (the envelope's
                    largest is 442,368), not probed; the op then compares the
                    size it needs against the buffer in 64 bits ("Scale tensor
                    size mismatch"), read, not probed
      No index passes through fp32. The envelope's largest input is
      9216 x 6144 = 56,623,104 elements.
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

    def is_valid(
        self,
        input: torch.Tensor,
        use_ue8m0: bool = False,
    ) -> None:
        """Refuse a base address the kernel's paired bf16 loads fault on."""
        if input.data_ptr() % 4:
            raise ValueError(
                f"fp8_quantize_1x128: input's base address is {input.data_ptr() % 4} bytes "
                "off 4-byte alignment; the kernel's paired bf16 loads fault there "
                "(CUDA error: misaligned address)"
            )


fp8_quantize_1x128 = Fp8Quantize1x128()
