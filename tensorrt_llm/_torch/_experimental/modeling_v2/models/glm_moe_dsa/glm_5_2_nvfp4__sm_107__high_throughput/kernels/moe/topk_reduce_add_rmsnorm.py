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
"""The end of an attention-DP MoE layer in one kernel.

A MegaMoE launch whose top-k reduce runs outside the kernel leaves one FC2 row per (token, top-k
slot) in its combine plane. The layer then ends with three kernels: TopkReduce sums the plane over
the top-k axis, the shared-expert output is added to that sum, and the next layer's input norm adds
the residual and normalizes. ``TopkReduceAddRMSNormKernel`` computes, per token row ``t``,

    routed      = bf16(plane[t, 0] + plane[t, 1] + ... + plane[t, K - 1])   (FP32, slot order)
    moe         = bf16(shared[t] + routed)
    h           = moe + residual[t]                                         (FP32)
    residual[t] = bf16(h)
    shared[t]   = bf16(h * rsqrt(mean(h * h) + eps) * weight)

with the rounding points of those three kernels: TopkReduce's FP32 slot-order sum (no top-k
weights), PyTorch's BF16 add, and FlashInfer's CuTe DSL ``FusedAddRMSNormKernel`` for the norm. The
sum of squares runs FlashInfer's ``row_reduce_sum_multirow`` over the per-thread column sets of that
kernel (64 threads per row, 8-element vectors, the same column order per thread) and the remaining
statements are FlashInfer's, so the outputs are bit-identical to the three-kernel sequence where
FlashInfer runs that kernel with one CTA per row.

One CTA owns one token row. Each thread loads whole 8-element vectors of every input, so the K
plane rows of a vector are in flight together. ``h`` goes through shared memory to the 64 threads
that take FlashInfer's per-thread column sets for the reduction; the other warps only meet the
reduction's block barrier. With PDL, the kernel loads the norm weight before its
``griddepcontrol.wait`` and releases its dependents right after it: a dependent reads the outputs
only after its own wait, which returns once this grid has completed.
"""

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int64
from flashinfer.norm.kernels.fused_add_rmsnorm import FusedAddRMSNormKernel
from flashinfer.norm.utils import row_reduce_sum_multirow

_VEC = 8  # BF16 elements per 16-byte vector


def _aligned(tensor: cute.Tensor, byte_alignment: int) -> cute.Tensor:
    pointer = tensor.iterator
    return cute.make_tensor(
        cute.make_ptr(
            pointer.dtype, pointer.toint(), pointer.memspace, assumed_align=byte_alignment
        ),
        tensor.layout,
    )


class TopkReduceAddRMSNormKernel:
    """Top-k reduce, shared add and residual add + RMSNorm of BF16 token rows.

    Args:
        hidden_size: Row length H.
        num_topk: Top-k slots K per token in the combine plane.
        num_threads: Threads per CTA; H / 8 must be a multiple of it.
    """

    def __init__(self, hidden_size: int, num_topk: int, num_threads: int):
        reference = FusedAddRMSNormKernel(cutlass.BFloat16, hidden_size, weight_bias=0.0)
        if reference.cluster_n != 1 or reference.vec_size != _VEC:
            raise ValueError(
                f"hidden size {hidden_size} does not map to one CTA per row with 8-element "
                "vectors in FusedAddRMSNormKernel"
            )
        num_vectors = hidden_size // _VEC
        if num_vectors % num_threads != 0 or num_threads % 32 != 0:
            raise ValueError(
                f"{num_threads} threads do not tile the {num_vectors} vectors of a row"
            )
        if reference.threads_per_row > num_threads:
            raise ValueError("the row reduction needs more threads than the CTA has")
        if reference.threads_per_row * reference.num_vec_blocks * _VEC != hidden_size:
            raise ValueError("FusedAddRMSNormKernel does not cover the row exactly")
        self.H = hidden_size
        self.num_topk = num_topk
        self.num_threads = num_threads
        self.vectors_per_thread = num_vectors // num_threads
        self.weight_bias = reference.weight_bias
        # FlashInfer's per-thread geometry for this H: thread t of the row holds the vectors
        # t, t + threads_per_row, ... (num_vec_blocks of them).
        self.ref_threads_per_row = reference.threads_per_row
        self.ref_warps_per_row = reference.warps_per_row
        self.ref_vec_blocks = reference.num_vec_blocks

    def _smem_size_in_bytes(self) -> int:
        return self.H * 4 + self.ref_warps_per_row * 4 + 16

    @cute.jit
    def __call__(
        self,
        mPlane: cute.Tensor,
        mShared: cute.Tensor,
        mResidual: cute.Tensor,
        mW: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        self.kernel(mPlane, mShared, mResidual, mW, eps, enable_pdl).launch(
            grid=[M, 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mPlane: cute.Tensor,
        mShared: cute.Tensor,
        mResidual: cute.Tensor,
        mW: cute.Tensor,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()

        H = self.H
        num_topk: cutlass.Constexpr[int] = self.num_topk
        num_threads: cutlass.Constexpr[int] = self.num_threads
        vectors_per_thread: cutlass.Constexpr[int] = self.vectors_per_thread
        ref_threads_per_row: cutlass.Constexpr[int] = self.ref_threads_per_row
        ref_warps_per_row: cutlass.Constexpr[int] = self.ref_warps_per_row
        ref_vec_blocks: cutlass.Constexpr[int] = self.ref_vec_blocks
        weight_bias = self.weight_bias

        smem = cutlass.utils.SmemAllocator()
        sH = smem.allocate_tensor(Float32, cute.make_layout((H,)), byte_alignment=16)
        reduction_buffer = smem.allocate_tensor(
            Float32, cute.make_layout((1, ref_warps_per_row)), byte_alignment=4
        )
        sRstd = smem.allocate_tensor(Float32, cute.make_layout((1,)), byte_alignment=4)
        vH = cute.zipped_divide(sH, (_VEC,))

        bf16_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
        )
        vec_bytes = _VEC * cutlass.BFloat16.width // 8

        gW = cute.zipped_divide(mW, (_VEC,))
        gShared = cute.zipped_divide(mShared[row, None], (_VEC,))
        gResidual = cute.zipped_divide(mResidual[row, None], (_VEC,))
        gPlane = cute.zipped_divide(mPlane[row, None, None], (num_topk, _VEC))

        # The norm weight is a parameter, written before any forward: load it before the wait.
        w_vec = cute.make_rmem_tensor((_VEC, vectors_per_thread), cutlass.BFloat16)
        for i in cutlass.range_constexpr(vectors_per_thread):
            v = tidx + i * num_threads
            cute.copy(bf16_atom, _aligned(gW[(None,), v], vec_bytes), w_vec[None, i])

        # PDL: the plane, the shared-expert output and the residual come from earlier grids
        # (the MegaMoE kernel, the shared expert on its stream, the previous layers); read them
        # only after the wait. Dependents read this kernel's outputs only after their own wait.
        if cutlass.const_expr(enable_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        terms = cute.make_rmem_tensor((_VEC, num_topk, vectors_per_thread), cutlass.BFloat16)
        shared_vec = cute.make_rmem_tensor((_VEC, vectors_per_thread), cutlass.BFloat16)
        residual_vec = cute.make_rmem_tensor((_VEC, vectors_per_thread), cutlass.BFloat16)
        for i in cutlass.range_constexpr(vectors_per_thread):
            v = tidx + i * num_threads
            plane_vec = gPlane[(None, None), (0, v)]
            for k in cutlass.range_constexpr(num_topk):
                cute.copy(bf16_atom, _aligned(plane_vec[k, None], vec_bytes), terms[None, k, i])
            cute.copy(bf16_atom, _aligned(gShared[(None,), v], vec_bytes), shared_vec[None, i])
            cute.copy(bf16_atom, _aligned(gResidual[(None,), v], vec_bytes), residual_vec[None, i])

        h_vec = cute.make_rmem_tensor((_VEC, vectors_per_thread), Float32)
        for i in cutlass.range_constexpr(vectors_per_thread):
            v = tidx + i * num_threads
            # TopkReduce: FP32 sum over the slots in order 0..K-1, rounded to BF16.
            acc = cute.make_rmem_tensor((_VEC,), Float32)
            for j in cutlass.range_constexpr(_VEC):
                acc[j] = Float32(terms[j, 0, i])
            for k in cutlass.range_constexpr(1, num_topk):
                for j in cutlass.range_constexpr(_VEC):
                    acc[j] = acc[j] + Float32(terms[j, k, i])
            routed = cute.make_rmem_tensor((_VEC,), cutlass.BFloat16)
            routed.store(acc.load().to(cutlass.BFloat16))
            # shared + routed in FP32, rounded to BF16 (PyTorch's BF16 add).
            moe_f32 = cute.make_rmem_tensor((_VEC,), Float32)
            for j in cutlass.range_constexpr(_VEC):
                moe_f32[j] = Float32(shared_vec[j, i]) + Float32(routed[j])
            moe = cute.make_rmem_tensor((_VEC,), cutlass.BFloat16)
            moe.store(moe_f32.load().to(cutlass.BFloat16))
            # FusedAddRMSNormKernel: h = input + residual in FP32; the residual keeps bf16(h).
            h = moe.load().to(Float32) + residual_vec[None, i].load().to(Float32)
            h_vec[None, i].store(h)
            residual_out = cute.make_rmem_tensor((_VEC,), cutlass.BFloat16)
            residual_out.store(h.to(cutlass.BFloat16))
            cute.copy(bf16_atom, residual_out, _aligned(gResidual[(None,), v], vec_bytes))
            cute.autovec_copy(h_vec[None, i], vH[(None,), v])

        cute.arch.barrier()

        # Sum of squares as FusedAddRMSNormKernel computes it: thread t of the first
        # ref_threads_per_row threads holds FlashInfer thread t's columns, vector blocks in order.
        if warp_idx < ref_warps_per_row:
            fragment = cute.make_rmem_tensor((_VEC, ref_vec_blocks), Float32)
            for b in cutlass.range_constexpr(ref_vec_blocks):
                cute.autovec_copy(vH[(None,), tidx + b * ref_threads_per_row], fragment[None, b])
            h_rows = fragment.load()
            h_sq = h_rows * h_rows
            sum_sq = row_reduce_sum_multirow(h_sq, ref_threads_per_row, reduction_buffer, None, 1)
            mean_sq = sum_sq / Float32(H)
            rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)
            if tidx == 0:
                sRstd[0] = rstd
        else:
            # The reduction's block barrier.
            cute.arch.barrier()

        cute.arch.barrier()
        rstd = sRstd[0]

        for i in cutlass.range_constexpr(vectors_per_thread):
            v = tidx + i * num_threads
            w = w_vec[None, i].load().to(Float32)
            y = h_vec[None, i].load() * rstd * (w + Float32(weight_bias))
            out = cute.make_rmem_tensor((_VEC,), cutlass.BFloat16)
            out.store(y.to(cutlass.BFloat16))
            cute.copy(bf16_atom, out, _aligned(gShared[(None,), v], vec_bytes))


def _num_threads(hidden_size: int, ref_threads_per_row: int) -> int:
    """The largest CTA (at most 1024 threads, whole warps) giving each thread whole vectors."""
    num_vectors = hidden_size // _VEC
    for warps in range(32, 0, -1):
        threads = warps * 32
        if num_vectors % threads == 0 and threads >= ref_threads_per_row:
            return threads
    return 0


def topk_reduce_add_rmsnorm_matches_flashinfer(device: torch.device, hidden_size: int) -> bool:
    """Whether ``flashinfer.norm.fused_add_rmsnorm`` runs the kernel whose numerics this one copies.

    FlashInfer's CUDA JIT norm and the cluster variant of ``FusedAddRMSNormKernel`` (very large
    hidden sizes) are not covered.
    """
    import flashinfer.norm

    use_cuda_norm = getattr(flashinfer.norm, "_use_cuda_norm", None)
    if use_cuda_norm is None or use_cuda_norm(device) or hidden_size % _VEC != 0:
        return False
    with torch.cuda.device(device):
        reference = FusedAddRMSNormKernel(cutlass.BFloat16, hidden_size, weight_bias=0.0)
    return (
        reference.cluster_n == 1
        and reference.vec_size == _VEC
        and reference.threads_per_row * reference.num_vec_blocks * _VEC == hidden_size
        and _num_threads(hidden_size, reference.threads_per_row) > 0
    )


@functools.cache
def _compile_topk_reduce_add_rmsnorm(hidden_size: int, num_topk: int, enable_pdl: bool):
    reference = FusedAddRMSNormKernel(cutlass.BFloat16, hidden_size, weight_bias=0.0)
    num_threads = _num_threads(hidden_size, reference.threads_per_row)
    kernel_obj = TopkReduceAddRMSNormKernel(hidden_size, num_topk, num_threads)
    sym_m = cute.sym_int(64)
    plane_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (sym_m, num_topk, hidden_size), stride_order=(2, 1, 0), assumed_align=16
    )
    shared_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (sym_m, hidden_size), stride_order=(1, 0), assumed_align=16
    )
    residual_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (sym_m, hidden_size), stride_order=(1, 0), assumed_align=16
    )
    w_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (hidden_size,), assumed_align=16
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    # Same compile options as FlashInfer's norm kernels.
    return cute.compile(
        kernel_obj,
        plane_fake,
        shared_fake,
        residual_fake,
        w_fake,
        Int64(1),
        Float32(1e-6),
        enable_pdl,
        stream_fake,
        options="--enable-tvm-ffi",
    )


def topk_reduce_add_rmsnorm(
    plane: torch.Tensor,
    shared: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    enable_pdl: bool,
) -> None:
    """Reduce the top-k plane, add the shared output and the residual, RMS-normalize, in place.

    Same values as ``shared.add_(TopkReduce(plane))`` followed by
    ``flashinfer.norm.fused_add_rmsnorm(shared, residual, weight, eps)``.

    Args:
        plane: Combine plane [M, K, H], BF16, contiguous: one FC2 row per (token, top-k slot),
            top-k weights already applied.
        shared: Shared-expert output [M, H], BF16, contiguous. Receives the normalized rows.
        residual: Residual [M, H], BF16, contiguous. Receives the new residual.
        weight: RMSNorm weight [H], BF16, contiguous.
        eps: RMSNorm epsilon.
        enable_pdl: Launch with programmatic dependent launch.
    """
    num_rows, hidden_size = shared.shape
    if num_rows == 0:
        return
    kernel = _compile_topk_reduce_add_rmsnorm(hidden_size, plane.shape[1], enable_pdl)
    kernel(plane, shared, residual, weight, num_rows, eps)


__all__ = [
    "TopkReduceAddRMSNormKernel",
    "topk_reduce_add_rmsnorm",
    "topk_reduce_add_rmsnorm_matches_flashinfer",
]
