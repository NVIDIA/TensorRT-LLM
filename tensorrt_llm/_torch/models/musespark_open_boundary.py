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
"""GB300 residual join and following prenorm across a CTA cluster.

Keep the BF16 branch-normalization and residual-to-prenorm rounding points.
Different reduction orders can still move values across BF16 thresholds.
Each reduction owns a separate DSM inbox/barrier; graph replays read live inputs.
"""

import functools

import torch

from .musespark_open_finalize_ar import HAS_CUTEDSL

if HAS_CUTEDSL:
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    from cutlass import Float32
    from cutlass.cute.runtime import make_ptr

    from .musespark_open_finalize_ar import _store_dsmem, _wait_cluster_tx, _warp_sum

    class _ClusterBoundaryKernel:
        def __init__(
            self, m: int, cluster: int, threads: int, gain: bool, post_eps: float, pre_eps: float
        ):
            self.m, self.cluster, self.threads, self.gain = m, cluster, threads, gain
            self.post_eps, self.pre_eps = float(post_eps), float(pre_eps)
            self.iters = 8192 // (cluster * threads * 8)
            self.warps = threads // 32
            self.slots = cluster * self.warps

        @cute.jit
        def __call__(
            self,
            branch: cute.Pointer,
            residual: cute.Pointer,
            gain: cute.Pointer,
            alpha: cute.Pointer,
            beta: cute.Pointer,
            weight: cute.Pointer,
            output: cute.Pointer,
            next_output: cute.Pointer,
            stream: cuda.CUstream,
        ):
            self.kernel(branch, residual, gain, alpha, beta, weight, output, next_output).launch(
                grid=[self.m * self.cluster, 1, 1],
                block=[self.threads, 1, 1],
                cluster=[self.cluster, 1, 1],
                smem=self.slots * 4 * 4 * 2 + 16,
                stream=stream,
                use_pdl=True,
            )

        @cute.jit
        def load(self, ptr: cute.Pointer, dtype: cutlass.Constexpr):
            layout = cute.make_layout((8, self.iters), stride=(1, self.threads * self.cluster * 8))
            reg = cute.make_rmem_tensor(cute.make_layout((8, self.iters)), dtype)
            atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=128)
            cute.copy(atom, cute.make_tensor(ptr.align(16), layout), reg)
            return reg

        @cute.jit
        def reduce(
            self, value: Float32, partials: cute.Tensor, bars: cute.Tensor, phase: cutlass.Constexpr
        ):
            tid = cute.arch.thread_idx()[0]
            lane, warp = tid % 32, tid // 32
            rank = cute.arch.block_idx()[0] % self.cluster
            value = _warp_sum(value)
            if cutlass.const_expr(self.cluster > 1):
                if lane < self.cluster:
                    _store_dsmem(
                        partials.iterator + (phase * self.slots + rank * self.warps + warp) * 4,
                        bars.iterator + phase,
                        lane,
                        value,
                        Float32(0),
                        Float32(0),
                        Float32(0),
                    )
                _wait_cluster_tx(bars.iterator + phase)
            else:
                if lane == 0:
                    partials[(phase * self.slots + warp) * 4] = value
                cute.arch.barrier()
            total = Float32(0)
            for slot in range(lane, self.slots, 32):
                total += partials[(phase * self.slots + slot) * 4]
            return _warp_sum(total)

        @cute.kernel
        def kernel(
            self,
            branch: cute.Pointer,
            residual: cute.Pointer,
            gain: cute.Pointer,
            alpha: cute.Pointer,
            beta: cute.Pointer,
            weight: cute.Pointer,
            output: cute.Pointer,
            next_output: cute.Pointer,
        ):
            tid = cute.arch.thread_idx()[0]
            rank = cute.arch.block_idx()[0] % self.cluster
            row = cute.arch.block_idx()[0] // self.cluster
            offset = cute.assume((rank * self.threads + tid) * 8, divby=8)
            smem = utils.SmemAllocator()
            partials = smem.allocate_tensor(
                Float32, cute.make_layout((self.slots * 4 * 2,)), byte_alignment=16
            )
            bars = smem.allocate_tensor(cutlass.Uint64, cute.make_layout((2,)), byte_alignment=8)
            if cutlass.const_expr(self.cluster > 1):
                if tid < 2:
                    cute.arch.mbarrier_init(bars.iterator + tid, 1)
                    cute.arch.mbarrier_arrive_and_expect_tx(bars.iterator + tid, self.slots * 16)
                cute.arch.mbarrier_init_fence()
                cute.arch.barrier()
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()
            # Coefficients may be produced dynamically when gate caching is off.
            # Wait before reading any input, including weights and coefficients.
            cute.arch.griddepcontrol_wait()
            a = self.load(alpha + offset, Float32)
            b = self.load(beta + offset, Float32)
            w = self.load(weight + offset, cutlass.BFloat16)
            g = self.load(gain + offset, cutlass.BFloat16)
            x = self.load(branch + row * 8192 + offset, cutlass.BFloat16)
            r = self.load(residual + row * 8192 + offset, Float32)
            scaled = cute.make_rmem_tensor(cute.make_layout((8, self.iters)), Float32)
            updated = cute.make_rmem_tensor(cute.make_layout((8, self.iters)), Float32)
            normed = cute.make_rmem_tensor(cute.make_layout((8, self.iters)), cutlass.BFloat16)
            total = Float32(0)
            for p in cutlass.range_constexpr(self.iters):
                for v in cutlass.range_constexpr(8):
                    value = Float32(x[v, p])
                    if cutlass.const_expr(self.gain):
                        value = Float32(cutlass.BFloat16(value * Float32(g[v, p])))
                    scaled[v, p] = value
                    total += value * value
            total = self.reduce(total, partials, bars, 0)
            inv = cute.math.rsqrt(total / Float32(8192) + Float32(self.post_eps), fastmath=True)
            total = Float32(0)
            for p in cutlass.range_constexpr(self.iters):
                for v in cutlass.range_constexpr(8):
                    value = Float32(cutlass.BFloat16(scaled[v, p] * inv))
                    u = a[v, p] * r[v, p] + b[v, p] * value
                    updated[v, p] = u
                    rounded = Float32(cutlass.BFloat16(u))
                    scaled[v, p] = rounded
                    total += rounded * rounded
            total = self.reduce(total, partials, bars, 1)
            inv = cute.math.rsqrt(total / Float32(8192) + Float32(self.pre_eps), fastmath=True)
            for p in cutlass.range_constexpr(self.iters):
                for v in cutlass.range_constexpr(8):
                    normed[v, p] = cutlass.BFloat16(
                        scaled[v, p] * inv * (Float32(w[v, p]) + Float32(1))
                    )
            layout = cute.make_layout((8, self.iters), stride=(1, self.threads * self.cluster * 8))
            atom32 = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), Float32, num_bits_per_copy=128
            )
            atom16 = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
            )
            cute.copy(
                atom32, updated, cute.make_tensor((output + row * 8192 + offset).align(16), layout)
            )
            cute.copy(
                atom16,
                normed,
                cute.make_tensor((next_output + row * 8192 + offset).align(16), layout),
            )
            if cutlass.const_expr(self.cluster > 1):
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()
            cute.arch.griddepcontrol_launch_dependents()

    @functools.lru_cache(maxsize=128)
    def _compile_boundary(
        tokens: int, has_gain: bool, post_eps: float, pre_eps: float, device_index: int
    ):
        cluster = 16 if tokens == 1 else (8 if tokens <= 8 else 4)
        threads = min(256, 8192 // (cluster * 8))
        dtypes = (
            cutlass.BFloat16,
            Float32,
            cutlass.BFloat16,
            Float32,
            Float32,
            cutlass.BFloat16,
            Float32,
            cutlass.BFloat16,
        )
        with torch.cuda.device(device_index):
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            return cute.compile(
                _ClusterBoundaryKernel(tokens, cluster, threads, has_gain, post_eps, pre_eps),
                *[
                    make_ptr(dtype, 16, cute.AddressSpace.gmem, assumed_align=16)
                    for dtype in dtypes
                ],
                stream,
                options="--enable-tvm-ffi",
            )


@functools.lru_cache(maxsize=16)
def cluster_boundary_available(device: torch.device) -> bool:
    """The measured dispatch is restricted to GB300."""
    return (
        HAS_CUTEDSL
        and device.type == "cuda"
        and torch.cuda.get_device_capability(device) == (10, 3)
    )


def cluster_boundary_supported(
    residual: torch.Tensor,
    branch: torch.Tensor,
    gain: torch.Tensor | None,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    next_weight: torch.Tensor,
) -> bool:
    """Check shape, dtype and vector-copy alignment before dispatch."""
    if residual.ndim != 2 or residual.shape[1] != 8192 or not 1 <= residual.shape[0] <= 32:
        return False
    if not cluster_boundary_available(residual.device):
        return False
    if (
        residual.dtype != torch.float32
        or branch.dtype != torch.bfloat16
        or branch.shape != residual.shape
    ):
        return False
    gain_arg = next_weight if gain is None else gain
    vectors = (gain_arg, alpha, beta, next_weight)
    if any(t.shape != (8192,) for t in vectors):
        return False
    if any(
        t.device != residual.device or not t.is_contiguous() or t.data_ptr() % 16
        for t in (residual, branch, *vectors)
    ):
        return False
    return (
        gain_arg.dtype == torch.bfloat16
        and next_weight.dtype == torch.bfloat16
        and alpha.dtype == torch.float32
        and beta.dtype == torch.float32
    )


def cluster_boundary(
    residual: torch.Tensor,
    branch: torch.Tensor,
    gain: torch.Tensor | None,
    alpha: torch.Tensor,
    beta: torch.Tensor,
    next_weight: torch.Tensor,
    post_eps: float,
    pre_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the measured M1–32/H8192 boundary; caller retains the fallback."""
    if not cluster_boundary_supported(residual, branch, gain, alpha, beta, next_weight):
        raise ValueError("Unsupported cluster boundary inputs, device, or alignment")
    tokens = residual.shape[0]
    gain_arg = next_weight if gain is None else gain
    updated, normalized = torch.empty_like(residual), torch.empty_like(branch)
    compiled = _compile_boundary(tokens, gain is not None, post_eps, pre_eps, residual.device.index)
    dtypes = (
        cutlass.BFloat16,
        Float32,
        cutlass.BFloat16,
        Float32,
        Float32,
        cutlass.BFloat16,
        Float32,
        cutlass.BFloat16,
    )
    args = (branch, residual, gain_arg, alpha, beta, next_weight, updated, normalized)
    # Resolve the stream here: CUDA graph capture can use a different stream
    # from the eager call that populated the compilation cache.
    stream = cuda.CUstream(torch.cuda.current_stream(residual.device).cuda_stream)
    compiled(*[make_ptr(dtype, tensor.data_ptr()) for dtype, tensor in zip(dtypes, args)], stream)
    return updated, normalized
