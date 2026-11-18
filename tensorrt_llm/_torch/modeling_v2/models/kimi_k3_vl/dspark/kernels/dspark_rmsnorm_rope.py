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
"""Fused DSpark RMSNorm and adjacent-pair RoPE for Blackwell: one warp per row.

``DSparkRMSNormRoPEKernel`` is the MLA DSpark drafter's RMSNorm + RoPE of the query rope slice and
the latents (``ops/spec/k3_dspark_rmsnorm_rope*``); ``is_fused_dspark_rmsnorm_rope_supported`` is
the operand contract its callers check before taking it.
"""

import functools
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch

from tensorrt_llm._utils import get_sm_version

try:
    from cuda.bindings import driver as cuda
except ImportError:
    from cuda import cuda

# The fused RMSNorm / RoPE kernel's devices.
_DSPARK_RMSNORM_ROPE_SM_VERSIONS = frozenset((100, 103, 107))


class DSparkRMSNormRoPEKernel:
    """Apply optional RMSNorm and last-dimension RoPE with one warp per row."""

    num_threads = cute.arch.WARP_SIZE

    def __init__(
        self,
        hidden_dim: int,
        rope_dim: int,
        num_heads: int,
        eps: float,
        apply_weight: bool,
        apply_rmsnorm: bool,
        inverse_rope: bool,
        norm_dim: int | None = None,
        out_rope_offset: int | None = None,
        write_nope: bool = True,
    ):
        if hidden_dim % self.num_threads != 0:
            raise ValueError(
                f"hidden_dim must be divisible by {self.num_threads}; got {hidden_dim}"
            )
        if rope_dim < 0 or rope_dim > hidden_dim or rope_dim % 2 != 0:
            raise ValueError(f"rope_dim must be even and in [0, {hidden_dim}]; got {rope_dim}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive; got {num_heads}")
        self.hidden_dim = hidden_dim
        self.rope_dim = rope_dim
        self.nope_dim = hidden_dim - rope_dim
        self.rope_pairs = rope_dim // 2
        self.num_heads = num_heads
        self.eps = eps
        self.apply_weight = apply_weight
        self.apply_rmsnorm = apply_rmsnorm
        self.inverse_rope = inverse_rope
        # Which prefix the RMS is taken over, and how far the norm scale and the
        # weight reach. DSpark normalizes the whole row (the default, bit-identical
        # to before this knob existed); DeepSeek-style MLA normalizes only the
        # kv_lora_rank latent and leaves k_pe raw, which is nope_dim here.
        self.norm_dim = hidden_dim if norm_dim is None else norm_dim
        if self.norm_dim not in (hidden_dim, self.nope_dim):
            raise ValueError(
                f"norm_dim must be hidden_dim ({hidden_dim}) or nope_dim "
                f"({self.nope_dim}); got {self.norm_dim}"
            )
        self.norm_covers_rope = self.norm_dim == hidden_dim
        # Where the rotated pairs land in `output`. None means "same column as in
        # the input", which is every caller that rewrites a row in place. The MLA
        # drafter instead reads a 192-wide q row and writes into the rope half of
        # a 576-wide fused query, so its input and output offsets differ and it
        # wants the nope half left to bmm_out.
        self.out_rope_offset = self.nope_dim if out_rope_offset is None else out_rope_offset
        if self.out_rope_offset < 0 or self.out_rope_offset % 2 != 0:
            raise ValueError(
                f"out_rope_offset must be even and non-negative; got {self.out_rope_offset}"
            )
        self.write_nope = write_nope
        if not self.write_nope and self.norm_covers_rope:
            # The rope lanes would need the RMS scale computed over a row whose
            # nope half is never written -- readable, but no caller wants it and
            # it would silently pair a raw-passthrough offset with a scaled rope.
            raise ValueError("write_nope=False requires norm_dim == nope_dim")
        if self.norm_dim % self.num_threads != 0:
            raise ValueError(
                f"norm_dim must be divisible by {self.num_threads}; got {self.norm_dim}"
            )
        self.norm_elements_per_thread = self.norm_dim // self.num_threads
        if self.nope_dim % self.num_threads != 0:
            raise ValueError(
                f"nope_dim must be divisible by {self.num_threads}; got {self.nope_dim}"
            )
        if self.rope_pairs % self.num_threads != 0:
            raise ValueError(
                f"rope pairs must be divisible by {self.num_threads}; got {self.rope_pairs}"
            )
        self.elements_per_thread = hidden_dim // self.num_threads
        self.nope_elements_per_thread = self.nope_dim // self.num_threads
        self.pairs_per_thread = self.rope_pairs // self.num_threads

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: Optional[cute.Tensor],
        freqs: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """weight is None exactly when apply_weight is False.

        Both reads sit behind const_expr(self.apply_weight), so with the flag
        off the operand never reaches the generated signature -- which is the
        point: a rope-only caller has no weight and should not have to
        materialize one. The compile wrappers derive both from the same flag,
        so they cannot disagree.
        """
        self.kernel(x, weight, freqs, output).launch(
            grid=[x.shape[0], 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: Optional[cute.Tensor],
        freqs: cute.Tensor,
        output: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()

        inverse_rms = cutlass.Float32(1.0)
        if cutlass.const_expr(self.apply_rmsnorm):
            sum_sq = cutlass.Float32(0.0)
            for item in cutlass.range_constexpr(self.norm_elements_per_thread):
                dim = tidx + item * self.num_threads
                value = cutlass.Float32(x[row, dim])
                sum_sq += value * value
            sum_sq = cute.arch.warp_reduction_sum(sum_sq)
            inverse_rms = cute.math.rsqrt(sum_sq / self.norm_dim + self.eps)

        if cutlass.const_expr(self.write_nope):
            for item in cutlass.range_constexpr(self.nope_elements_per_thread):
                dim = tidx + item * self.num_threads
                value = cutlass.Float32(x[row, dim]) * inverse_rms
                if cutlass.const_expr(self.apply_weight):
                    value *= cutlass.Float32(weight[dim])
                output[row, dim] = value.to(output.element_type)

        if cutlass.const_expr(self.rope_pairs > 0):
            freq_row = row // self.num_heads
            for item in cutlass.range_constexpr(self.pairs_per_thread):
                pair = tidx + item * self.num_threads
                real_dim = self.nope_dim + pair * 2
                imag_dim = real_dim + 1
                out_real = self.out_rope_offset + pair * 2
                out_imag = out_real + 1
                real = cutlass.Float32(x[row, real_dim])
                imag = cutlass.Float32(x[row, imag_dim])
                # Outside norm_dim the rope lanes are passed through raw: no RMS
                # scale, no weight. weight is only norm_dim long in that case.
                if cutlass.const_expr(self.norm_covers_rope):
                    real *= inverse_rms
                    imag *= inverse_rms
                    if cutlass.const_expr(self.apply_weight):
                        real *= cutlass.Float32(weight[real_dim])
                        imag *= cutlass.Float32(weight[imag_dim])
                cos = cutlass.Float32(freqs[freq_row, pair, 0])
                sin = cutlass.Float32(freqs[freq_row, pair, 1])
                if cutlass.const_expr(self.inverse_rope):
                    sin = -sin
                output[row, out_real] = (real * cos - imag * sin).to(output.element_type)
                output[row, out_imag] = (imag * cos + real * sin).to(output.element_type)


@functools.cache
def _is_dspark_rmsnorm_rope_sm_supported(sm_version: int | None = None) -> bool:
    """Return whether the fused RMSNorm/RoPE kernels support this GPU."""
    if sm_version is None:
        sm_version = get_sm_version()
    return sm_version in _DSPARK_RMSNORM_ROPE_SM_VERSIONS


def _has_regular_row_stride(x: torch.Tensor) -> bool:
    """Return whether leading dimensions flatten to non-overlapping strided rows."""
    if x.is_contiguous():
        return True
    if x.stride(-1) != 1:
        return False

    outer_dims = [dim for dim in range(x.ndim - 1) if x.shape[dim] > 1]
    if not outer_dims:
        return True

    row_dim = outer_dims[-1]
    if x.stride(row_dim) < x.shape[-1]:
        return False
    return all(
        x.stride(dim) == x.stride(next_dim) * x.shape[next_dim]
        for dim, next_dim in zip(outer_dims, outer_dims[1:])
    )


def is_fused_dspark_rmsnorm_rope_supported(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    freqs: torch.Tensor,
    num_heads: int,
    rope_dim: int,
    norm_dim: int | None = None,
) -> bool:
    """Return whether tensors satisfy the production fused-op contract.

    ``weight`` is None for a kernel built with ``apply_weight=False``; the
    weight checks are then vacuous rather than a reason to reject, and the
    caller has no tensor to offer in the first place.
    """
    operands = (x, freqs) if weight is None else (x, weight, freqs)
    if not _is_dspark_rmsnorm_rope_sm_supported() or not all(t.is_cuda for t in operands):
        return False
    if x.dtype != torch.bfloat16 or (weight is not None and weight.dtype != torch.bfloat16):
        return False
    if freqs.dtype != torch.float32:
        return False
    if x.ndim < 2 or x.shape[-1] % 32 != 0:
        return False
    if rope_dim < 0 or rope_dim > x.shape[-1] or rope_dim % 2 != 0:
        return False
    # norm_dim defaults to the whole row; the only other supported value is the
    # nope prefix, where the weight spans just that prefix.
    effective_norm_dim = x.shape[-1] if norm_dim is None else norm_dim
    if effective_norm_dim not in (x.shape[-1], x.shape[-1] - rope_dim):
        return False
    if effective_norm_dim % 32 != 0:
        return False
    if weight is not None and weight.shape != (effective_norm_dim,):
        return False
    if (x.shape[-1] - rope_dim) % 32 != 0 or (rope_dim // 2) % 32 != 0:
        return False
    rows = x.numel() // x.shape[-1]
    if num_heads <= 0 or rows % num_heads != 0:
        return False
    return (
        freqs.ndim == 3
        and freqs.shape[0] == rows // num_heads
        and freqs.shape[1] >= max(1, rope_dim // 2)
        and freqs.shape[2] == 2
        and _has_regular_row_stride(x)
        and (weight is None or weight.is_contiguous())
        and freqs.is_contiguous()
    )
