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
"""BF16 head RMSNorm and sigmoid gate with the FlashInfer H64 reduction order."""

from functools import lru_cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32


class _RmsNormGate:
    def __init__(self, heads: int):
        self.heads = heads

    @cute.jit
    def __call__(self, x, gate, out, eps: Float32, stream):
        self.kernel(x, gate, out, eps).launch(
            grid=(cute.ceil_div(x.shape[0] * self.heads, 16), 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, x, gate, out, eps: Float32):
        tid = cute.arch.thread_idx()[0]
        row = cute.arch.block_idx()[0] * 16 + tid // 8
        token = row // self.heads
        head = row % self.heads
        col = (tid % 8) * 8
        layout = cute.make_layout((8,))
        x_offset = cute.assume(token * x.stride[0] + head * 64 + col, divby=8)
        g_offset = cute.assume(token * gate.stride[0] + head * gate.stride[1] + col, divby=8)
        o_offset = cute.assume(token * self.heads * 64 + head * 64 + col, divby=8)
        gx = cute.make_tensor(x.iterator + x_offset, layout)
        gg = cute.make_tensor(gate.iterator + g_offset, layout)
        go = cute.make_tensor(out.iterator + o_offset, layout)
        rx = cute.make_rmem_tensor((8,), BFloat16)
        rg = cute.make_rmem_tensor((8,), BFloat16)
        ro = cute.make_rmem_tensor((8,), BFloat16)
        atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128)
        rx.fill(0)
        rg.fill(0)
        if row < x.shape[0] * self.heads:
            cute.copy(atom, gx, rx)
            cute.copy(atom, gg, rg)
        xv = rx.load().to(Float32)
        gv = rg.load().to(Float32)

        # FlashInfer H64 uses eight contiguous values per thread, followed
        # by ascending butterfly offsets. A different sum tree changes
        # values at BF16 rounding boundaries even with identical inputs.
        squared = xv * xv
        total = squared.reduce(cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0)
        for i in cutlass.range_constexpr(3):
            total = total + cute.arch.shuffle_sync_bfly(total, offset=1 << i)
        inv = cute.math.rsqrt(total / Float32(64) + eps, fastmath=True)
        normed = (xv * inv).to(BFloat16).to(Float32)
        sigmoid = (
            (Float32(1.0) / (Float32(1.0) + cute.math.exp(-gv, fastmath=True)))
            .to(BFloat16)
            .to(Float32)
        )
        ro.store((normed * sigmoid).to(BFloat16))
        if row < x.shape[0] * self.heads:
            cute.copy(atom, ro, go)


@lru_cache(maxsize=None)
def _compile_norm_gate(heads: int):
    tokens = cute.sym_int64()
    x = cute.runtime.make_fake_tensor(
        BFloat16,
        (tokens, heads * 64),
        (cute.sym_int64(divisibility=8), 1),
        assumed_align=16,
    )
    gate = cute.runtime.make_fake_tensor(
        BFloat16,
        (tokens, heads, 64),
        (cute.sym_int64(divisibility=8), cute.sym_int64(divisibility=8), 1),
        assumed_align=16,
    )
    out = cute.runtime.make_fake_compact_tensor(
        BFloat16, (tokens, heads * 64), stride_order=(1, 0), assumed_align=16
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        _RmsNormGate(heads), x, gate, out, Float32(1e-5), stream, options="--enable-tvm-ffi"
    )


def rmsnorm_gate(x: torch.Tensor, gate: torch.Tensor, eps: float) -> torch.Tensor:
    """Normalize BF16 H64 heads, round, and multiply by a BF16 sigmoid gate.

    Token count and aligned row/head strides remain dynamic in the compiled
    kernel, so mixed prefill batches do not each trigger a new compilation.
    """
    if not (
        x.is_cuda
        and x.dtype == torch.bfloat16
        and x.ndim == 2
        and x.shape[1] > 0
        and x.shape[1] % 64 == 0
        and x.stride(1) == 1
        and x.stride(0) % 8 == 0
        and x.data_ptr() % 16 == 0
        and gate.device == x.device
        and gate.dtype == x.dtype
        and gate.ndim in (2, 3)
        and gate.shape[0] == x.shape[0]
        and gate.numel() == x.numel()
        and gate.stride(-1) == 1
        and gate.data_ptr() % 16 == 0
    ):
        raise ValueError("RMSNorm gate requires aligned CUDA BF16 tensors with H64 heads")
    heads = x.shape[1] // 64
    if gate.ndim == 2:
        gate = gate.view(x.shape[0], heads, 64)
    if gate.shape[1:] != (heads, 64) or any(stride % 8 for stride in gate.stride()[:2]):
        raise ValueError("Gate head and token strides must be multiples of eight elements")
    out = torch.empty_like(x, memory_format=torch.contiguous_format)
    if x.shape[0]:
        _compile_norm_gate(heads)(x, gate, out, float(eps))
    return out
