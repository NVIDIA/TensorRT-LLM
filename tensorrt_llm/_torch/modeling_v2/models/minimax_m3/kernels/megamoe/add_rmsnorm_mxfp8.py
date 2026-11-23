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
"""FlashInfer's fused residual add + RMSNorm of BF16 rows, plus other forms of the normalized rows, in one kernel:
their MXFP8 form (``add_rmsnorm_mxfp8``), copies of them side by side (``add_rmsnorm_replicated``), or both
(``add_rmsnorm_replicated_mxfp8``).

Per token row t:
    h           = input[t] + residual[t]                                   (FP32)
    residual[t] = bf16(h)
    input[t]    = bf16(h * rsqrt(mean(h * h) + eps) * weight)
as FlashInfer's CuTe DSL ``FusedAddRMSNormKernel`` computes it, which ``flashinfer.norm.fused_add_rmsnorm`` runs
where ``megamoe_topk_reduce_add_rmsnorm_matches_flashinfer`` holds: the sum of squares is FlashInfer's
``row_reduce_sum_multirow`` over that kernel's per-thread column sets, as ``megamoe_topk_reduce_tail``'s norm form
computes it. The normalized row's MXFP8 form is then ``mxfp8_rows.store_mxfp8_vector``'s, bit for bit
``trtllm::mxfp8_quantize(input, True)``; its copies are the BF16 values stored to ``input[t]``. One CTA owns one row.
"""

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64

from .megamoe_topk_reduce_tail import _aligned, _no_mxfp8_rows, _num_threads
from .mxfp8_rows import (
    VEC,
    check_mxfp8_rows,
    mxfp8_fakes,
    rcp_approx_ftz,
    store_mxfp8_vector,
    zero_mxfp8_pad_rows,
)


class AddRMSNormKernel:
    """FlashInfer's fused add + RMSNorm of BF16 rows with a second form of the normalized rows.

    Args:
        hidden_size: Row length H, a multiple of 128.
        num_threads: Threads per CTA; H / 8 must be a multiple of it.
        emit_mxfp8: Write the normalized rows' MXFP8 form into the q / sf arguments.
        num_replicas: Write each normalized row this many times side by side into the replicas argument
            ([M, num_replicas * H]); 0 writes none.
    """

    def __init__(
        self, hidden_size: int, num_threads: int, emit_mxfp8: bool = True, num_replicas: int = 0
    ):
        from flashinfer.norm.kernels.fused_add_rmsnorm import FusedAddRMSNormKernel
        from flashinfer.norm.utils import row_reduce_sum_multirow

        num_vectors = hidden_size // VEC
        if hidden_size % 128 != 0 or num_threads % 32 != 0 or num_vectors % num_threads != 0:
            raise ValueError(
                f"{num_threads} threads do not tile the {num_vectors} vectors of a {hidden_size}-wide row"
            )
        reference = FusedAddRMSNormKernel(cutlass.BFloat16, hidden_size, weight_bias=0.0)
        if reference.cluster_n != 1 or reference.vec_size != VEC:
            raise ValueError(
                f"hidden size {hidden_size} does not map to one CTA per row with 8-element vectors in "
                "FusedAddRMSNormKernel"
            )
        if reference.threads_per_row > num_threads:
            raise ValueError("the row reduction needs more threads than the CTA has")
        if reference.threads_per_row * reference.num_vec_blocks * VEC != hidden_size:
            raise ValueError("FusedAddRMSNormKernel does not cover the row exactly")
        if num_replicas < 0:
            raise ValueError(f"num_replicas must be >= 0, got {num_replicas}")
        self.H = hidden_size
        self.num_threads = num_threads
        self.vectors_per_thread = num_vectors // num_threads
        self.emit_mxfp8 = emit_mxfp8
        self.num_replicas = num_replicas
        self._row_reduce_sum_multirow = row_reduce_sum_multirow
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
        mInput: cute.Tensor,
        mResidual: cute.Tensor,
        mW: cute.Tensor,
        mQ: cute.Tensor,
        mSF: cute.Tensor,
        mReplicas: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        self.kernel(mInput, mResidual, mW, mQ, mSF, mReplicas, Int32(M), eps, enable_pdl).launch(
            grid=[M, 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mInput: cute.Tensor,
        mResidual: cute.Tensor,
        mW: cute.Tensor,
        mQ: cute.Tensor,
        mSF: cute.Tensor,
        mReplicas: cute.Tensor,
        num_rows: Int32,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
    ):
        row_reduce_sum_multirow = self._row_reduce_sum_multirow
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.warp_idx()

        H = self.H
        num_threads: cutlass.Constexpr[int] = self.num_threads
        vectors_per_thread: cutlass.Constexpr[int] = self.vectors_per_thread
        ref_threads_per_row: cutlass.Constexpr[int] = self.ref_threads_per_row
        ref_warps_per_row: cutlass.Constexpr[int] = self.ref_warps_per_row
        ref_vec_blocks: cutlass.Constexpr[int] = self.ref_vec_blocks
        num_k_tiles: cutlass.Constexpr[int] = (H // 32 + 3) // 4
        emit_mxfp8: cutlass.Constexpr[bool] = self.emit_mxfp8
        num_replicas: cutlass.Constexpr[int] = self.num_replicas
        num_vectors: cutlass.Constexpr[int] = H // VEC
        weight_bias = self.weight_bias

        smem = cutlass.utils.SmemAllocator()
        sH = smem.allocate_tensor(Float32, cute.make_layout((H,)), byte_alignment=16)
        reduction_buffer = smem.allocate_tensor(
            Float32, cute.make_layout((1, ref_warps_per_row)), byte_alignment=4
        )
        sRstd = smem.allocate_tensor(Float32, cute.make_layout((1,)), byte_alignment=4)
        vH = cute.zipped_divide(sH, (VEC,))

        bf16_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
        )
        vec_bytes = VEC * cutlass.BFloat16.width // 8

        gW = cute.zipped_divide(mW, (VEC,))
        gInput = cute.zipped_divide(mInput[row, None], (VEC,))
        gResidual = cute.zipped_divide(mResidual[row, None], (VEC,))

        # The norm weight is a parameter, written before any forward: load it before the wait.
        w_vec = cute.make_rmem_tensor((VEC, vectors_per_thread), cutlass.BFloat16)
        for i in cutlass.range_constexpr(vectors_per_thread):
            v = tidx + i * num_threads
            cute.copy(bf16_atom, _aligned(gW[(None,), v], vec_bytes), w_vec[None, i])

        if cutlass.const_expr(enable_pdl):
            cute.arch.griddepcontrol_wait()
            cute.arch.griddepcontrol_launch_dependents()

        input_vec = cute.make_rmem_tensor((VEC, vectors_per_thread), cutlass.BFloat16)
        residual_vec = cute.make_rmem_tensor((VEC, vectors_per_thread), cutlass.BFloat16)
        for i in cutlass.range_constexpr(vectors_per_thread):
            v = tidx + i * num_threads
            cute.copy(bf16_atom, _aligned(gInput[(None,), v], vec_bytes), input_vec[None, i])
            cute.copy(bf16_atom, _aligned(gResidual[(None,), v], vec_bytes), residual_vec[None, i])

        h_vec = cute.make_rmem_tensor((VEC, vectors_per_thread), Float32)
        for i in cutlass.range_constexpr(vectors_per_thread):
            v = tidx + i * num_threads
            # FusedAddRMSNormKernel: h = input + residual in FP32; the residual keeps bf16(h).
            h = input_vec[None, i].load().to(Float32) + residual_vec[None, i].load().to(Float32)
            h_vec[None, i].store(h)
            residual_out = cute.make_rmem_tensor((VEC,), cutlass.BFloat16)
            residual_out.store(h.to(cutlass.BFloat16))
            cute.copy(bf16_atom, residual_out, _aligned(gResidual[(None,), v], vec_bytes))
            cute.autovec_copy(h_vec[None, i], vH[(None,), v])

        cute.arch.barrier()

        # Sum of squares as FusedAddRMSNormKernel computes it: thread t of the first
        # ref_threads_per_row threads holds FlashInfer thread t's columns, vector blocks in order.
        if warp_idx < ref_warps_per_row:
            fragment = cute.make_rmem_tensor((VEC, ref_vec_blocks), Float32)
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

        rcp_448 = rcp_approx_ftz(Float32(448.0))
        gReplicas = cute.zipped_divide(mReplicas[row, None], (VEC,))
        for i in cutlass.range_constexpr(vectors_per_thread):
            v = tidx + i * num_threads
            w = w_vec[None, i].load().to(Float32)
            y = h_vec[None, i].load() * rstd * (w + Float32(weight_bias))
            out = cute.make_rmem_tensor((VEC,), cutlass.BFloat16)
            out.store(y.to(cutlass.BFloat16))
            cute.copy(bf16_atom, out, _aligned(gInput[(None,), v], vec_bytes))
            if cutlass.const_expr(emit_mxfp8):
                store_mxfp8_vector(out, mQ, mSF, row, v, num_k_tiles, rcp_448)
            for r in cutlass.range_constexpr(num_replicas):
                cute.copy(
                    bf16_atom, out, _aligned(gReplicas[(None,), v + r * num_vectors], vec_bytes)
                )

        if cutlass.const_expr(emit_mxfp8):
            zero_mxfp8_pad_rows(mSF, row, num_rows, tidx, num_threads, num_k_tiles)


@functools.cache
def _compile_add_rmsnorm(hidden_size: int, enable_pdl: bool, emit_mxfp8: bool, num_replicas: int):
    from flashinfer.norm.kernels.fused_add_rmsnorm import FusedAddRMSNormKernel

    reference = FusedAddRMSNormKernel(cutlass.BFloat16, hidden_size, weight_bias=0.0)
    num_threads = _num_threads(hidden_size, reference.threads_per_row)
    kernel_obj = AddRMSNormKernel(hidden_size, num_threads, emit_mxfp8, num_replicas)
    sym_m = cute.sym_int(64)
    rows = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (sym_m, hidden_size), stride_order=(1, 0), assumed_align=16
    )
    residual = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (sym_m, hidden_size), stride_order=(1, 0), assumed_align=16
    )
    weight = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (hidden_size,), assumed_align=16
    )
    # Own row counts: without its epilogue the kernel takes one-row placeholders for that form.
    q, sf = mxfp8_fakes(hidden_size, cute.sym_int())
    replicas = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16,
        (cute.sym_int(), max(num_replicas, 1) * hidden_size),
        stride_order=(1, 0),
        assumed_align=16,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    # Same compile options as FlashInfer's norm kernels.
    return cute.compile(
        kernel_obj,
        rows,
        residual,
        weight,
        q,
        sf,
        replicas,
        Int64(1),
        Float32(1e-6),
        enable_pdl,
        stream,
        options="--enable-tvm-ffi",
    )


@functools.cache
def _no_replicas(hidden_size: int, device: torch.device) -> torch.Tensor:
    """One-row placeholder for the replicas argument of a kernel compiled without copies."""
    return torch.empty((1, hidden_size), dtype=torch.bfloat16, device=device)


def _check_rows(input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor):
    if input.dim() != 2:
        raise ValueError("input is [M, H]")
    num_rows, hidden_size = input.shape
    if tuple(residual.shape) != (num_rows, hidden_size) or tuple(weight.shape) != (hidden_size,):
        raise ValueError(
            f"residual {tuple(residual.shape)} and weight {tuple(weight.shape)} do not match "
            f"input {tuple(input.shape)}"
        )
    for name, tensor in (("input", input), ("residual", residual), ("weight", weight)):
        if tensor.dtype != torch.bfloat16:
            raise ValueError(f"{name} must be BF16")
        if not tensor.is_contiguous() or tensor.data_ptr() % 16 != 0:
            raise ValueError(f"{name} must be contiguous and 16-byte aligned")
    return num_rows, hidden_size


def add_rmsnorm_mxfp8(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    q: torch.Tensor,
    sf: torch.Tensor,
    *,
    enable_pdl: bool,
) -> None:
    """Residual add + RMSNorm in place, plus the MXFP8 form of the normalized rows.

    Same values as ``flashinfer.norm.fused_add_rmsnorm(input, residual, weight, eps)`` followed by
    ``trtllm::mxfp8_quantize(input, True)`` where ``megamoe_topk_reduce_add_rmsnorm_matches_flashinfer`` holds.

    Args:
        input: Rows [M, H], BF16, contiguous. Receives the normalized rows.
        residual: Residual [M, H], BF16, contiguous. Receives the new residual.
        weight: RMSNorm weight [H], BF16, contiguous.
        eps: RMSNorm epsilon.
        q: E4M3 [M, H], contiguous. Receives the MXFP8 rows.
        sf: UINT8, at least ``mxfp8_rows.scale_bytes(M, H)``. Receives the swizzled UE8M0 scales.
        enable_pdl: Launch with programmatic dependent launch.
    """
    num_rows, hidden_size = _check_rows(input, residual, weight)
    check_mxfp8_rows(q, sf, num_rows, hidden_size)
    if num_rows == 0:
        return
    kernel = _compile_add_rmsnorm(hidden_size, enable_pdl, True, 0)
    replicas = _no_replicas(hidden_size, input.device)
    kernel(input, residual, weight, q.view(torch.int64), sf, replicas, num_rows, eps)


def add_rmsnorm_replicated(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    replicas: torch.Tensor,
    *,
    enable_pdl: bool,
) -> None:
    """Residual add + RMSNorm in place, plus each normalized row written n times side by side.

    Same values as ``flashinfer.norm.fused_add_rmsnorm(input, residual, weight, eps)`` followed by
    ``replicas.copy_(torch.cat([input] * n, dim=-1))`` where ``megamoe_topk_reduce_add_rmsnorm_matches_flashinfer``
    holds.

    Args:
        input: Rows [M, H], BF16, contiguous. Receives the normalized rows.
        residual: Residual [M, H], BF16, contiguous. Receives the new residual.
        weight: RMSNorm weight [H], BF16, contiguous.
        eps: RMSNorm epsilon.
        replicas: [M, n * H] with n >= 1, BF16, contiguous. Receives the n copies.
        enable_pdl: Launch with programmatic dependent launch.
    """
    num_rows, hidden_size = _check_rows(input, residual, weight)
    _check_replicas(replicas, num_rows, hidden_size)
    if num_rows == 0:
        return
    kernel = _compile_add_rmsnorm(hidden_size, enable_pdl, False, replicas.shape[1] // hidden_size)
    q_words, sf = _no_mxfp8_rows(hidden_size, input.device)
    kernel(input, residual, weight, q_words, sf, replicas, num_rows, eps)


def add_rmsnorm_replicated_mxfp8(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    replicas: torch.Tensor,
    q: torch.Tensor,
    sf: torch.Tensor,
    *,
    enable_pdl: bool,
) -> None:
    """``add_rmsnorm_replicated`` and the MXFP8 form of ``add_rmsnorm_mxfp8`` in one launch.

    Args:
        input: Rows [M, H], BF16, contiguous. Receives the normalized rows.
        residual: Residual [M, H], BF16, contiguous. Receives the new residual.
        weight: RMSNorm weight [H], BF16, contiguous.
        eps: RMSNorm epsilon.
        replicas: [M, n * H] with n >= 1, BF16, contiguous. Receives the n copies.
        q: E4M3 [M, H], contiguous. Receives the MXFP8 rows.
        sf: UINT8, at least ``mxfp8_rows.scale_bytes(M, H)``. Receives the swizzled UE8M0 scales.
        enable_pdl: Launch with programmatic dependent launch.
    """
    num_rows, hidden_size = _check_rows(input, residual, weight)
    _check_replicas(replicas, num_rows, hidden_size)
    check_mxfp8_rows(q, sf, num_rows, hidden_size)
    if num_rows == 0:
        return
    kernel = _compile_add_rmsnorm(hidden_size, enable_pdl, True, replicas.shape[1] // hidden_size)
    kernel(input, residual, weight, q.view(torch.int64), sf, replicas, num_rows, eps)


def _check_replicas(replicas: torch.Tensor, num_rows: int, hidden_size: int) -> None:
    if (
        replicas.dtype != torch.bfloat16
        or replicas.dim() != 2
        or replicas.shape[0] != num_rows
        or replicas.shape[1] == 0
        or replicas.shape[1] % hidden_size != 0
    ):
        raise ValueError(
            f"replicas {tuple(replicas.shape)} {replicas.dtype} is not BF16 [{num_rows}, n * {hidden_size}]"
        )
    if not replicas.is_contiguous() or replicas.data_ptr() % 16 != 0:
        raise ValueError("replicas must be contiguous and 16-byte aligned")


__all__ = [
    "AddRMSNormKernel",
    "add_rmsnorm_mxfp8",
    "add_rmsnorm_replicated",
    "add_rmsnorm_replicated_mxfp8",
]
