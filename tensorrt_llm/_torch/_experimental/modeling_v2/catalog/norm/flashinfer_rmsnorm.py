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
from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE, get_env_enable_pdl

from .._op import Arch, Cell, OpWrapper

if IS_FLASHINFER_AVAILABLE:
    from flashinfer.norm import rmsnorm as _rmsnorm_into


class FlashinferRmsnorm(OpWrapper):
    """`x / sqrt(mean(x^2, -1) + eps) * weight`, one op call, new tensor out.

    Fusion boundary: normalization and weight scaling only. No residual add
    (`flashinfer_fused_add_rmsnorm`), no gemma `(1 + w)` form
    (`flashinfer_gemma_rmsnorm`), no quantization. `x` is not mutated.

    `x` may be a row-strided view -- the leading columns of a wider row, such
    as the latent `kv[:, :512]` of a 576-wide kv_a row -- and the result is
    still a new contiguous tensor; the `note` says which views are accepted.

    With `out`, the result is written there instead and `out` is returned:
    `x` itself (in place, the latent normalized inside its kv_a row, so the
    row is the `[RMSNorm(c_kv) | k_pe]` latent with no concatenation) or the
    leading columns of another buffer (a NoPE owner's 576-wide latent row,
    whose rope tail the caller keeps zero). The columns of the row outside
    `out` are not touched.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})

    # The widths the shipped targets normalize over are in this entry's test.
    # gpt-oss reaches this op only on its residual stream; deepseek reaches it
    # there and on the two lora ranks its MLA projects through, which is why
    # narrow rows are certified at all -- a 512-wide row is one 128-bit vector
    # per lane and takes a different path through the kernel than a 7168-wide
    # one.
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="prefill-shaped: enough rows to cross the kernel's grid tiling",
            spec=dict(shape=(2048, 7168), dtype=torch.bfloat16, eps=1e-6),
        ),
    )

    note = """
    Registered only when flashinfer is importable (`IS_FLASHINFER_AVAILABLE`).

    The path the cells certify is FlashInfer's CuTe DSL kernel:
    `flashinfer.norm.rmsnorm` runs `rmsnorm_cute`
    (`flashinfer/norm/kernels/rmsnorm.py`) unless `FLASHINFER_USE_CUDA_NORM=1`
    is set or the installed CuTe DSL cannot target the device; then it runs
    the CUDA-JIT kernel, which this entry does not certify. `rmsnorm_cute`
    compiles on the first call in a process for each (dtype, width, row
    layout): a compact kernel when input and output are contiguous as torch
    counts it, a strided one otherwise. A one-row view of a wider row is
    contiguous to torch, so it takes the compact kernel; measured, and again
    at 32768 rows on the 576-strided latent: compact at 1 row, strided from 2.
    The cells drive both at width 512 and the compact one at every other
    width.

    PDL is controlled by `TRTLLM_ENABLE_PDL` (default on) inside the trtllm
    custom op; it changes scheduling, not results, and is not a cell axis.
    The cells run with it on.

    Row-strided input. `x` may be a view whose last dim is contiguous, whose
    row stride is a multiple of 8 elements and whose data is 16-byte aligned:
    the leading columns of a wider row. The result is a new contiguous tensor
    and the columns of the row outside `x` are not touched. Every other layout
    was measured to raise before the launch rather than answer, so `is_valid`
    does not restate them (the test pins each). Measured at width 512; every
    message carries the same signature, which names the width:
      last dim not contiguous (`buf[..., 0]` of a [T, 512, 2] buffer):
        ValueError: Mismatched mX.strides[1] on argument #0 when calling:
        `__call__(mX: Tensor([n0, 512], bfloat16), mW: Tensor([512],
        bfloat16), mY: Tensor([n0, 512], bfloat16), M: int64, eps: float32)`,
        expected to be 1
      row stride not a multiple of 8 (`buf[:, :512]` of a [T, 516] buffer):
        ValueError: Invalid mX.strides[0] on argument #0 when calling:
        `__call__(mX: Tensor([n0, 512], bfloat16), mW: Tensor([512],
        bfloat16), mY: Tensor([n0, 512], bfloat16), M: int64, eps: float32)`,
        expected to be divisible by 8
      data not 16-byte aligned (`buf[:, 4:516]` of a [T, 576] buffer):
        ValueError: Misaligned Tensor data on argument #0 when calling:
        `__call__(mX: Tensor([n0, 512], bfloat16), mW: Tensor([512],
        bfloat16), mY: Tensor([n0, 512], bfloat16), M: int64, eps: float32)`,
        expected data alignment=16 bytes
    The first refusal in a process takes about 2 s inside the cute DSL's
    TVM-FFI call wrapper; later ones are immediate.

    `out` runs the same kernel through `flashinfer.norm.rmsnorm(..., out=out)`
    with the custom op's PDL setting (the op itself allocates its output), so
    the result is bitwise the call without `out` (the `out` cells hold it so);
    a 576-strided `out` takes the strided kernel from two rows, as `x` does.
    The compact and strided kernels are compiled from one `RMSNormKernel`
    program (`rmsnorm_cute`) and differ only in the layouts they are compiled
    for, the row stride a constant or a runtime int64: the same operations at
    the same precision, so a NoPE owner's latent written at row stride 576
    holds the bits the contiguous call returns.
    In place is safe: each row is read whole before it is written, by the
    same block (measured bitwise at every cell). An `out` that overlaps `x`
    without being `x` is refused by `is_valid`.

    Rows are computed independently: at 6144 and at 512 in both layouts,
    every row count in 1..9216 is bitwise equal to the leading rows of the
    9216-row call, every row count in 1..32768 to the leading rows of the
    32768-row call, and a CUDA-graph replay is bitwise equal to an eager call
    on the same data. An all-zero row was measured to return zeros. The
    32768-row call is at most one bf16 ulp from the reference, on 6.44e-6 of
    the elements at 6144 and 1.55e-6 to 1.85e-6 at 512.

    fp16 and fp32 inputs are accepted by the op and were measured to work,
    but no shipped target uses them, so they are not cells. TRT-LLM's own
    `RMSNorm` module routes here for fp16/bf16 only.

    `float8_e4m3fn` is the trap this entry cannot guard: the CuTe DSL path's
    dtype table maps it and returns a plausible fp8 rmsnorm rather than
    raising, so a caller who forgets to dequantize gets a result. `is_valid`
    rejects it below. The CUDA-JIT path
    (`FLASHINFER_USE_CUDA_NORM=1`) does raise, but this entry does not
    certify that path.

    Hard limits (sm_103). Computed, not probed: the envelope's largest call,
    32768 x 6144 (the NVFP4 target's step), is 201,326,592 elements, 9.4% of
    2^31; its last row starts at element 201,320,448.
      rows x width   the compact kernel computes row * width in int32;
                     `rmsnorm_cute` switches to the strided kernel (int64 row
                     stride) itself once rows * width > 2^31 - 1, at 349526
                     rows for 6144 and 4194304 for 512 (read in the source),
                     10.7x and 128x the 32768-row step
      row offset     the strided kernel's row stride is int64: no 32-bit
                     product at row stride 576 (the kv_a buffer of a
                     32768-row step is 18,874,368 elements)
      rows           int64 in the kernel; the grid is rows / rows-per-block,
                     16384 blocks at 32768 x 6144 and 8192 at 32768 x 512,
                     against the 2^31 - 1 grid limit
      fp32           no index passes through fp32
      KV pool        none read: the op takes this step's rows only, so the
                     pool size sets no limit
    """

    def __call__(
        self, x: torch.Tensor, weight: torch.Tensor, eps: float, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        if out is None:
            return torch.ops.trtllm.flashinfer_rmsnorm(x, weight, eps)
        return _rmsnorm_into(x, weight, eps, out=out, enable_pdl=get_env_enable_pdl())

    def reference(
        self, x: torch.Tensor, weight: torch.Tensor, eps: float, out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """fp32-accumulated; the kernel reduces in fp32 and rounds once at the end. A new
        tensor either way; `out` is not read."""
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
        return (normed * weight.float()).to(x.dtype)

    def is_valid(
        self, x: torch.Tensor, weight: torch.Tensor, eps: float, out: torch.Tensor | None = None
    ) -> None:
        # Only the case the op takes and answers wrongly. Shape, dtype and
        # layout mismatches it rejects itself, loudly (the `note` quotes the
        # layout errors), and duplicating that here would only swap its error
        # for a worse one.
        assert x.dtype is not torch.float8_e4m3fn, (
            "float8_e4m3fn is accepted by the kernel's dtype table and returns "
            "a plausible fp8 rmsnorm; dequantize before normalizing"
        )
        # A partial overlap reads rows the call has already written.
        if out is not None and not (out.data_ptr() == x.data_ptr() and out.stride() == x.stride()):
            x0, o0 = x.data_ptr(), out.data_ptr()
            x1 = x0 + ((x.shape[0] - 1) * x.stride(0) + x.shape[1]) * x.element_size()
            o1 = o0 + ((out.shape[0] - 1) * out.stride(0) + out.shape[1]) * out.element_size()
            assert not (x0 < o1 and o0 < x1), "out must be x itself or not overlap it"


flashinfer_rmsnorm = FlashinferRmsnorm()
