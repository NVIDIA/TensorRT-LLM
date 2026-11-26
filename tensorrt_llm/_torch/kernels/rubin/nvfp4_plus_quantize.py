# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""NVFP4_PLUS activation quantization on Rubin (SM107).

NVFP4_PLUS stores E2M1 values with one UE5M3 block scale per 32 values along K
and no per-tensor scale. UE5M3 spans 2**-17 to 114688, which covers
``amax / 6`` of any practical activation block.

Each block scale is ``amax / 6`` rounded up to UE5M3, so no value exceeds the
E2M1 range after the division by the decoded scale (see ``ue5m3_utils``).
"""

from typing import Callable

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass._mlir.dialects import math
from cutlass.cute.runtime import make_fake_stream

from tensorrt_llm.quantization.utils.fp4_utils import NVFP4_PLUS_SF_VEC_SIZE, pad_up

from .ue5m3_utils import f32_to_ue5m3, ue5m3_to_f32

# One thread quantizes four consecutive blocks: their scales are one 32-bit
# word of the 128x4 swizzled scale layout.
_BLOCKS_PER_THREAD = 4
_THREADS_PER_CTA = 128
_SF_ROW_TILE = 128

_CUTLASS_DTYPES = {torch.bfloat16: cutlass.BFloat16, torch.float16: cutlass.Float16}


class _Nvfp4PlusQuantize:
    """Quantize the rows of ``x`` [M, K] into ``fp4`` [M, K / 2] and ``sf``.

    ``sf`` is either the 128x4 swizzled layout as int32 words
    [pad_up(M, 128) * pad_up(K / 32, 4) / 4], with the padding rows and
    columns zeroed, or row-major bytes [M, K / 32].
    """

    def __init__(self, swizzled: bool) -> None:
        self.swizzled = swizzled

    @cute.jit
    def __call__(
        self, x: cute.Tensor, fp4: cute.Tensor, sf: cute.Tensor, stream: cuda.CUstream
    ) -> None:
        rows = x.shape[0]
        if cutlass.const_expr(self.swizzled):
            rows = (rows + _SF_ROW_TILE - 1) // _SF_ROW_TILE * _SF_ROW_TILE
        sf_cols = x.shape[1] // NVFP4_PLUS_SF_VEC_SIZE
        groups = (sf_cols + _BLOCKS_PER_THREAD - 1) // _BLOCKS_PER_THREAD
        threads = rows * groups
        self.kernel(x, fp4, sf).launch(
            grid=((threads + _THREADS_PER_CTA - 1) // _THREADS_PER_CTA, 1, 1),
            block=(_THREADS_PER_CTA, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, x: cute.Tensor, fp4: cute.Tensor, sf: cute.Tensor) -> None:
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        num_rows = x.shape[0]
        sf_cols = x.shape[1] // NVFP4_PLUS_SF_VEC_SIZE
        groups = (sf_cols + _BLOCKS_PER_THREAD - 1) // _BLOCKS_PER_THREAD
        rows = num_rows
        if cutlass.const_expr(self.swizzled):
            rows = (num_rows + _SF_ROW_TILE - 1) // _SF_ROW_TILE * _SF_ROW_TILE
        index = cutlass.Int64(block) * _THREADS_PER_CTA + thread
        row = index // groups
        group = index % groups

        if row < rows:
            src = cute.make_rmem_tensor(
                (NVFP4_PLUS_SF_VEC_SIZE * _BLOCKS_PER_THREAD,), x.element_type
            )
            scales = cute.make_rmem_tensor((_BLOCKS_PER_THREAD,), cutlass.Float32)
            scales.fill(0.0)
            for b in cutlass.range_constexpr(_BLOCKS_PER_THREAD):
                col = group * _BLOCKS_PER_THREAD + b
                if (row < num_rows) & (col < sf_cols):
                    src_block = cute.local_tile(src, (NVFP4_PLUS_SF_VEC_SIZE,), (b,))
                    cute.autovec_copy(
                        cute.local_tile(x[row, None], (NVFP4_PLUS_SF_VEC_SIZE,), (col,)),
                        src_block,
                    )
                    values = src_block.load().to(cutlass.Float32)
                    magnitudes = type(values)(
                        math.absf(values.ir_value()), values.shape, values.dtype
                    )
                    amax = magnitudes.reduce(cute.ReductionOp.MAX, cutlass.Float32(0.0), 0)
                    scales[b] = amax / cutlass.Float32(6.0)

            codes = cute.make_rmem_tensor((_BLOCKS_PER_THREAD,), cutlass.FloatNV8E5M3FNU)
            f32_to_ue5m3(scales, codes)
            code_bytes = cute.recast_tensor(codes, cutlass.Uint8)

            if cutlass.const_expr(self.swizzled):
                # Byte offset of (row, 4 * group) in the 128x4 layout, in words:
                # (row // 128 * groups + group) * 512 + row % 32 * 16 + row % 128 // 32 * 4.
                word = (row // _SF_ROW_TILE * groups + group) * _SF_ROW_TILE + (
                    row % 32 * 4 + row % _SF_ROW_TILE // 32
                )
                sf[word] = cute.recast_tensor(codes, cutlass.Int32)[0]
            else:
                for b in cutlass.range_constexpr(_BLOCKS_PER_THREAD):
                    col = group * _BLOCKS_PER_THREAD + b
                    if (row < num_rows) & (col < sf_cols):
                        sf[row, col] = code_bytes[b]

            for b in cutlass.range_constexpr(_BLOCKS_PER_THREAD):
                col = group * _BLOCKS_PER_THREAD + b
                if (row < num_rows) & (col < sf_cols):
                    scale = ue5m3_to_f32(cutlass.Uint32(code_bytes[b]))
                    inv_scale = cutlass.Float32(0.0)
                    if scale != cutlass.Float32(0.0):
                        inv_scale = cutlass.Float32(1.0) / scale
                    values = cute.local_tile(src, (NVFP4_PLUS_SF_VEC_SIZE,), (b,)).load()
                    quantized = cute.make_rmem_tensor(
                        (NVFP4_PLUS_SF_VEC_SIZE,), cutlass.Float4E2M1FN
                    )
                    quantized.store(
                        (values.to(cutlass.Float32) * inv_scale).to(cutlass.Float4E2M1FN)
                    )
                    cute.autovec_copy(
                        cute.recast_tensor(quantized, cutlass.Uint8),
                        cute.local_tile(fp4[row, None], (NVFP4_PLUS_SF_VEC_SIZE // 2,), (col,)),
                    )


_compiled_kernels: dict[tuple[torch.dtype, bool], Callable[..., None]] = {}


def _compile(dtype: torch.dtype, swizzled: bool) -> Callable[..., None]:
    num_rows = cute.sym_int64()
    x = cute.runtime.make_fake_tensor(
        _CUTLASS_DTYPES[dtype],
        (num_rows, cute.sym_int64(divisibility=NVFP4_PLUS_SF_VEC_SIZE)),
        (cute.sym_int64(divisibility=NVFP4_PLUS_SF_VEC_SIZE), 1),
        assumed_align=16,
    )
    fp4 = cute.runtime.make_fake_tensor(
        cutlass.Uint8,
        (num_rows, cute.sym_int64(divisibility=NVFP4_PLUS_SF_VEC_SIZE // 2)),
        (cute.sym_int64(divisibility=NVFP4_PLUS_SF_VEC_SIZE // 2), 1),
        assumed_align=16,
    )
    if swizzled:
        sf = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32, (cute.sym_int64(),), assumed_align=16
        )
    else:
        sf = cute.runtime.make_fake_tensor(
            cutlass.Uint8, (num_rows, cute.sym_int64()), (cute.sym_int64(), 1), assumed_align=1
        )
    return cute.compile(
        _Nvfp4PlusQuantize(swizzled),
        x,
        fp4,
        sf,
        make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


def nvfp4_plus_quantize(
    x: torch.Tensor, sf_swizzled: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``x`` [M, K] (bf16 or fp16, K % 32 == 0) to NVFP4_PLUS.

    Returns the packed E2M1 values [M, K // 2] (uint8, element 2j in the low
    nibble of byte j) and the UE5M3 scales (uint8): the 128x4 swizzled layout
    [pad_up(M, 128) * pad_up(K // 32, 4)] with zero padding when
    ``sf_swizzled``, else row-major [M, K // 32].
    """
    if x.dim() != 2 or x.dtype not in _CUTLASS_DTYPES:
        raise ValueError(
            f"nvfp4_plus_quantize expects a 2D bf16/fp16 tensor, got {x.dtype} {x.shape}"
        )
    m, k = x.shape
    if k % NVFP4_PLUS_SF_VEC_SIZE != 0:
        raise ValueError(
            f"nvfp4_plus_quantize needs K divisible by {NVFP4_PLUS_SF_VEC_SIZE}, got {k}"
        )
    if x.stride(1) != 1 or x.stride(0) % NVFP4_PLUS_SF_VEC_SIZE != 0 or x.data_ptr() % 16 != 0:
        x = x.contiguous()
    sf_cols = k // NVFP4_PLUS_SF_VEC_SIZE
    fp4 = torch.empty((m, k // 2), dtype=torch.uint8, device=x.device)
    if sf_swizzled:
        sf = torch.empty(
            (pad_up(m, _SF_ROW_TILE) * pad_up(sf_cols, 4),), dtype=torch.uint8, device=x.device
        )
    else:
        sf = torch.empty((m, sf_cols), dtype=torch.uint8, device=x.device)
    if m == 0:
        return fp4, sf

    key = (x.dtype, sf_swizzled, x.device.index)
    compiled = _compiled_kernels.get(key)
    if compiled is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "nvfp4_plus_quantize compiles on its first call, which must run outside CUDA-graph capture"
            )
        compiled = _compiled_kernels[key] = _compile(x.dtype, sf_swizzled)
    compiled(x, fp4, sf.view(torch.int32) if sf_swizzled else sf)
    return fp4, sf
