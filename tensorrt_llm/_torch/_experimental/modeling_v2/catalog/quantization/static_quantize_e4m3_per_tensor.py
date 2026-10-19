# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static per-tensor FP8 quantization: bf16 -> float8_e4m3fn at a fixed input scale."""

from __future__ import annotations

import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — loads the extension registering torch.ops.tensorrt_llm.*
from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper

E4M3_MAX = 448.0

#: The vectorized kernel's address requirement: 128-bit loads of the input.
#: The output is the op's own fresh allocation, so only the input can miss it.
VEC_ALIGN_BYTES = 16

#: Each FP8 linear's checkpoint `input_scale`, the smallest and the largest
#: over the 48 layers that carry one.
IN_PROJ_SCALES = (0.0166015625, 0.3549107015132904)
OUT_PROJ_SCALES = (0.00788225419819355, 1.8214285373687744)
SHARED_UP_SCALES = (0.00177001953125, 0.1841517835855484)
SHARED_DOWN_SCALES = (0.004359654150903225, 6.035714149475098)


def _site(why: str, rows: int, cols: int, scale: float, source: str = "mixed") -> Cell:
    return Cell(
        why=why, spec=dict(shape=(rows, cols), scale=scale, source=source, misaligned=False)
    )


class StaticQuantizeE4m3PerTensor(OpWrapper):
    """`sat_e4m3(input * (1 / scales[0]))`, one rounding, saturating at +-448.

    Wraps `torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor` -- the
    `tensorrt_llm` namespace; there is no `trtllm::` alias.

    It DIVIDES by the scale: pass the checkpoint's `input_scale` as stored
    (modelopt's `amax / 448`), never its reciprocal. `scales` is an fp32
    tensor holding one value; the checkpoint's `[1]` and a 0-d tensor are both
    accepted unchanged. The output is a fresh float8_e4m3fn tensor of
    `input`'s shape.

    The op returns a second tensor, an `empty_like(scales)` in `input`'s
    dtype that it never writes. That is uninitialized memory, so this entry
    returns the quantized tensor alone.

    Two kernels sit behind the op and round differently:

    * bf16 input at a 16-byte-aligned address -- every call site, since each
      quantizes a fresh activation (or a row slice of one: every row is a
      multiple of 16 bytes at these widths) -- runs the vectorized kernel,
      which forms `1/s` once and multiplies.
    * anything else (a misaligned bf16 view, fp16, fp32) runs the scalar
      kernel, which divides per element.

    `x * (1/s)` and `x / s` round apart when the quotient lands within one
    fp32 ulp of an e4m3 rounding boundary. Measured at out_proj's largest
    scale (1.8214...) on a [8192, 4096] activation, 262519 of 33.5M codes
    (0.78%) differ by one between the two. `reference` states which kernel
    runs and computes that kernel's arithmetic, so both are certified
    bit-exactly rather than one within a band.
    """

    ARCHS = frozenset({Arch.SM_107})

    # The four FP8 linears' activations at their tep4 widths -- in_proj and
    # the shared-expert up_proj read the normed hidden state (8192), out_proj
    # the gated norm's output (4096 per rank), the shared-expert down_proj the
    # relu2 output (2560 per rank) -- each at its checkpoint scale extremes.
    # The op is elementwise, so the row counts are the decode, batch and
    # prefill heights rather than separate kernel paths; the prefill cells
    # also carry the largest grids (67M and 33.5M elements).
    CELLS: tuple[Cell, ...] = (
        _site("in_proj decode: x[1, 8192], smallest in_proj scale", 1, 8192, IN_PROJ_SCALES[0]),
        _site(
            "in_proj prefill: x[8192, 8192], largest in_proj scale", 8192, 8192, IN_PROJ_SCALES[1]
        ),
        _site(
            "out_proj decode: y[1, 4096] at 1.8214, where 1/s is inexact",
            1,
            4096,
            OUT_PROJ_SCALES[1],
        ),
        _site(
            "out_proj decode batch: y[32, 4096], smallest out_proj scale",
            32,
            4096,
            OUT_PROJ_SCALES[0],
        ),
        _site(
            "out_proj prefill: y[8192, 4096] at 1.8214 -- 0.78% of codes tell x*(1/s) from x/s",
            8192,
            4096,
            OUT_PROJ_SCALES[1],
        ),
        _site("shared up_proj: x[7, 8192], smallest shared-up scale", 7, 8192, SHARED_UP_SCALES[0]),
        _site(
            "shared up_proj prefill: x[8192, 8192], largest shared-up scale",
            8192,
            8192,
            SHARED_UP_SCALES[1],
        ),
        _site(
            "shared down_proj decode: relu2 output [1, 2560], non-negative, largest scale",
            1,
            2560,
            SHARED_DOWN_SCALES[1],
            source="relu2",
        ),
        _site(
            "shared down_proj prefill: relu2 output [8192, 2560], smallest scale",
            8192,
            2560,
            SHARED_DOWN_SCALES[0],
            source="relu2",
        ),
        # dep4 runs the same linears unsharded: out_proj reads the full gated
        # norm output (16384) and the shared down_proj the full relu2 output
        # (10240); in_proj and the shared up_proj keep 8192. The [8192, 16384]
        # prefill is 2^27 elements: the vectorized kernel's full grid (65536
        # blocks x 256 threads x 8 elements) with no grid-stride iteration.
        _site(
            "dep4 in_proj / shared up_proj decode batch: x[32, 8192], largest in_proj scale",
            32,
            8192,
            IN_PROJ_SCALES[1],
        ),
        _site(
            "dep4 out_proj decode: y[1, 16384] at 1.8214, where 1/s is inexact",
            1,
            16384,
            OUT_PROJ_SCALES[1],
        ),
        _site(
            "dep4 out_proj decode batch: y[32, 16384], smallest out_proj scale",
            32,
            16384,
            OUT_PROJ_SCALES[0],
        ),
        _site(
            "dep4 out_proj prefill: y[8192, 16384] at 1.8214, 2^27 elements",
            8192,
            16384,
            OUT_PROJ_SCALES[1],
        ),
        _site(
            "dep4 shared down_proj decode: relu2 output [1, 10240], largest scale",
            1,
            10240,
            SHARED_DOWN_SCALES[1],
            source="relu2",
        ),
        _site(
            "dep4 shared down_proj batch: relu2 output [7, 10240], smallest scale",
            7,
            10240,
            SHARED_DOWN_SCALES[0],
            source="relu2",
        ),
        _site(
            "dep4 shared down_proj prefill: relu2 output [8192, 10240], largest scale",
            8192,
            10240,
            SHARED_DOWN_SCALES[1],
            source="relu2",
        ),
        Cell(
            why="every non-NaN bf16 value (+-inf, subnormals, ties) at four scales; numel % 8 != 0 tail",
            spec=dict(
                shape=None,
                scales=(1.0, IN_PROJ_SCALES[0], OUT_PROJ_SCALES[1], SHARED_DOWN_SCALES[1]),
                source="all_bf16",
                misaligned=False,
            ),
        ),
        Cell(
            why="bf16 input at a non-16-byte-aligned address: the scalar kernel, x / s",
            spec=dict(
                shape=(8192, 4096), scale=OUT_PROJ_SCALES[1], source="mixed", misaligned=True
            ),
        ),
    )

    note = """
    Metadata consumed: none; stateless. The scale is read from device memory
    when the kernel runs, so a CUDA graph replay picks up a scale written in
    place (certified by the test).

    Preconditions the op enforces loudly (`e4m3_quantize_helper` in
    thop/fp8Op.cpp): `input` CUDA, contiguous, rank >= 2, non-empty, fp32 /
    fp16 / bf16; `scales` CUDA fp32 (a bf16 scale raises).

    Saturation: |x / s| beyond 448 -- infinities included -- becomes +-448.
    torch's own float -> e4m3fn cast is not saturating (it returns NaN from
    464 up), which is why `reference` clamps before casting. -0.0 stays -0.0.

    Not certified: fp16 and fp32 inputs (the scalar kernel; no call site),
    NaN inputs (payload and sign are not compared), and the grid-stride loop,
    which the vectorized kernel enters only above 2^27 elements (the dep4
    out_proj prefill cell sits exactly at that limit).
    """

    def __call__(self, input: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        return torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(input, scales)[0]

    def reference(self, input: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
        """The kernel that runs, in fp32: multiply by `1/s` or divide by `s`.

        Both are single IEEE fp32 operations on the exact bf16 value, then
        one round-to-nearest-even into e4m3 after clamping to +-448. The
        branch reads the input's dtype and address because the op does.
        """
        s = scales.float().reshape(-1)[0]
        if input.dtype is torch.bfloat16 and input.data_ptr() % VEC_ALIGN_BYTES == 0:
            scaled = input.float() * (1.0 / s)
        else:
            scaled = input.float() / s
        return scaled.clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)

    def is_valid(self, input: torch.Tensor, scales: torch.Tensor) -> None:
        # The kernel loads scales[0] and nothing else, so a per-token or
        # per-channel vector is applied as its first element to every row --
        # measured to return a valid-looking tensor, never to raise.
        assert scales.numel() == 1, (
            f"scales must hold exactly one value; got {scales.numel()}, and the "
            "op would silently apply scales[0] to every element"
        )


static_quantize_e4m3_per_tensor = StaticQuantizeE4m3PerTensor()
