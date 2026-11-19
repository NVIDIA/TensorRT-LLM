# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Split of the context-step indexer projection: ``kb, wq = split(out, [128, 32])``
with the keys rounded to BF16, in one launch.

At context row counts the DSA indexer's two projections (keys ``wk`` and head
weights ``weights_proj``) run as one persistent GEMM into an FP32
``out[rows, 160]``.  The serving glue then needed two PyTorch elementwise
copies per indexer layer: ``kb.copy_(out[:, :128])`` (FP32 -> BF16) and
``wq.copy_(out[:, 128:])`` (FP32 -> FP32), about 6.6 us per layer at ~7.6k rows
for 7.6 MB of traffic.  This kernel reads each row once and
writes both outputs:

* ``kb[r, c] = bf16(out[r, c])`` for ``c < 128``, round-to-nearest-even, as
  ``Tensor.copy_`` converts (``cvt.rn.bf16.f32``);
* ``wq[r, c] = out[r, 128 + c]`` for ``c < 32``, bit-exact.

One warp per row, eight rows per CTA of 256 threads.  Lane ``l`` handles key
columns ``l, l + 32, l + 64, l + 96`` and weight column ``l``, so every warp
access is a coalesced 128-byte (FP32) or 64-byte (BF16) segment.  All three
operands take a runtime row pitch with a unit inner stride; the output rows
must not overlap the input.

Pattern: glm-indexer-prefill-wk-split
Device: SM100+ (Blackwell)
Shape: runtime rows; 160 = 128 key + 32 weight columns
"""

from collections.abc import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

KEY_COLS = 128
WEIGHT_COLS = 32
IN_COLS = KEY_COLS + WEIGHT_COLS
_WARP = 32
_ROWS_PER_CTA = 8
_THREADS = _WARP * _ROWS_PER_CTA

CompiledKernel = Callable[..., object]
_COMPILED: list[CompiledKernel] = []


@cute.kernel
def _split_kernel(
    src: cute.Tensor, keys: cute.Tensor, weights: cute.Tensor, rows: cutlass.Int32
) -> None:
    bidx, _, _ = cute.arch.block_idx()
    tidx, _, _ = cute.arch.thread_idx()
    row = bidx * _ROWS_PER_CTA + tidx // _WARP
    lane = tidx % _WARP
    if row < rows:
        for k in cutlass.range_constexpr(KEY_COLS // _WARP):
            col = lane + k * _WARP
            keys[row, col] = src[row, col].to(cutlass.BFloat16)
        weights[row, lane] = src[row, KEY_COLS + lane]


@cute.jit
def _launch(
    src: cute.Tensor,
    keys: cute.Tensor,
    weights: cute.Tensor,
    rows: cutlass.Int32,
    num_ctas: cutlass.Int32,
    stream: cuda.CUstream,
) -> None:
    _split_kernel(src, keys, weights, rows).launch(
        grid=(num_ctas, 1, 1), block=(_THREADS, 1, 1), stream=stream
    )


def _fake(dtype, cols: int, align: int) -> object:
    # Element alignment only: the accesses are scalar, and the outputs may be
    # column slices of wider rows.
    return make_fake_tensor(
        dtype,
        (cute.sym_int32(), cols),
        stride=(cute.sym_int64(), 1),
        assumed_align=align,
    )


def compile_split() -> CompiledKernel:
    """Compile (once) the TVM-FFI launcher ``fn(src, keys, weights, rows, num_ctas)``."""
    if not _COMPILED:
        z = cutlass.Int32(0)
        _COMPILED.append(
            cute.compile(
                _launch,
                _fake(cutlass.Float32, IN_COLS, 4),
                _fake(cutlass.BFloat16, KEY_COLS, 2),
                _fake(cutlass.Float32, WEIGHT_COLS, 4),
                z,
                z,
                make_fake_stream(use_tvm_ffi_env_stream=True),
                options="--enable-tvm-ffi",
            )
        )
    return _COMPILED[0]


def _check(src: torch.Tensor, keys: torch.Tensor, weights: torch.Tensor) -> int:
    """Validate the operands on the host; return the row count."""
    if src.dtype != torch.float32 or keys.dtype != torch.bfloat16 or weights.dtype != torch.float32:
        raise ValueError(
            f"dtypes must be fp32 / bf16 / fp32, got {src.dtype} / {keys.dtype} / {weights.dtype}"
        )
    if not (src.is_cuda and keys.device == src.device and weights.device == src.device):
        raise ValueError("operands must be on one CUDA device")
    rows = int(src.shape[0])
    if (
        src.ndim != 2
        or tuple(src.shape) != (rows, IN_COLS)
        or tuple(keys.shape) != (rows, KEY_COLS)
        or tuple(weights.shape) != (rows, WEIGHT_COLS)
    ):
        raise ValueError(
            f"shapes must be [R, {IN_COLS}] / [R, {KEY_COLS}] / [R, {WEIGHT_COLS}], got "
            f"{tuple(src.shape)} / {tuple(keys.shape)} / {tuple(weights.shape)}"
        )
    if src.stride(1) != 1 or keys.stride(1) != 1 or weights.stride(1) != 1:
        raise ValueError("every operand needs a unit inner stride")
    if rows > 1 and (keys.stride(0) < KEY_COLS or weights.stride(0) < WEIGHT_COLS):
        raise ValueError("output rows overlap (row pitch below the row width)")
    return rows


def split_(src: torch.Tensor, keys: torch.Tensor, weights: torch.Tensor) -> None:
    """``keys.copy_(src[:, :128]); weights.copy_(src[:, 128:])`` in one launch on
    torch's current stream."""
    rows = _check(src, keys, weights)
    if rows == 0:
        return
    num_ctas = (rows + _ROWS_PER_CTA - 1) // _ROWS_PER_CTA
    compile_split()(src, keys, weights, rows, num_ctas)


def warm(device: torch.device | str = "cuda") -> None:
    """Compile and launch once (eagerly, before any CUDA-graph capture)."""
    src = torch.zeros((3, IN_COLS), dtype=torch.float32, device=device)
    split_(
        src,
        torch.empty((3, KEY_COLS), dtype=torch.bfloat16, device=device),
        torch.empty((3, WEIGHT_COLS), dtype=torch.float32, device=device),
    )
    torch.cuda.synchronize()
