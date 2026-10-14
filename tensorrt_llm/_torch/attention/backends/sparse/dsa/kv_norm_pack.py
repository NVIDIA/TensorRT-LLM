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
"""Fused DSA KV RMSNorm and latent-cache packing."""

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.flashinfer_utils import IS_FLASHINFER_AVAILABLE
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.utils import is_nvfp4_marlin_enabled


def is_kv_norm_pack_supported(
    compressed_kv: torch.Tensor, k_pe: torch.Tensor, norm: RMSNorm
) -> bool:
    return (
        type(norm) is RMSNorm
        and IS_FLASHINFER_AVAILABLE
        and not is_nvfp4_marlin_enabled()
        and not norm.use_gemma
        and not norm.use_cuda_tile
        and not norm.is_nvfp4
        and not norm.return_hp_output
        and compressed_kv.is_cuda
        and compressed_kv.dtype == torch.bfloat16
        and compressed_kv.ndim == 2
        and compressed_kv.shape[1] == 512
        and compressed_kv.stride(1) == 1
        and compressed_kv.stride(0) >= 512
        and k_pe.device == compressed_kv.device
        and k_pe.dtype == compressed_kv.dtype
        and k_pe.ndim == 2
        and k_pe.shape == (compressed_kv.shape[0], 64)
        and k_pe.stride(1) == 1
        and k_pe.stride(0) >= 64
        and norm.weight.device == compressed_kv.device
        and norm.weight.dtype == compressed_kv.dtype
        and norm.weight.shape == (512,)
        and norm.weight.is_contiguous()
    )


@triton.jit
def _kv_norm_pack_kernel(
    kv_ptr, k_pe_ptr, weight_ptr, norm_ptr, latent_ptr, num_rows, kv_stride, k_pe_stride, eps
):
    rows = (tl.program_id(0) + tl.arange(0, 1)).to(tl.int64)
    cols = tl.arange(0, 512)
    row_mask = rows < num_rows
    kv = tl.load(
        kv_ptr + rows[:, None] * kv_stride + cols[None, :],
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    variance = tl.sum(kv * kv, axis=1) / 512
    # Match FlashInfer's FP32 rsqrt and single final BF16 rounding.
    rcp = tl.inline_asm_elementwise(
        "rsqrt.approx.ftz.f32 $0, $1;",
        "=f,f",
        [variance + eps],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    weight = tl.load(weight_ptr + cols).to(tl.float32)
    normed = ((kv * rcp[:, None]) * weight[None, :]).to(tl.bfloat16)
    tl.store(norm_ptr + rows[:, None] * 512 + cols[None, :], normed, row_mask[:, None])
    tl.store(latent_ptr + rows[:, None] * 576 + cols[None, :], normed, row_mask[:, None])

    rope_cols = tl.arange(0, 64)
    k_pe = tl.load(
        k_pe_ptr + rows[:, None] * k_pe_stride + rope_cols[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    tl.store(
        latent_ptr + rows[:, None] * 576 + 512 + rope_cols[None, :],
        k_pe,
        row_mask[:, None],
    )


@torch.library.custom_op("trtllm::dsa_kv_norm_pack", mutates_args=())
def kv_norm_pack(
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize BF16 KV[M,512] and pack K-RoPE[M,64] into contiguous outputs.

    Inputs have unit inner strides; weight is contiguous BF16 [512]. Both
    outputs own their storage, preserving the normalized KV's 512-column stride.
    """
    num_rows = compressed_kv.shape[0]
    normed = compressed_kv.new_empty((num_rows, 512))
    latent_cache = compressed_kv.new_empty((num_rows, 576))
    if num_rows:
        _kv_norm_pack_kernel[(num_rows,)](
            compressed_kv,
            k_pe,
            weight,
            normed,
            latent_cache,
            num_rows,
            compressed_kv.stride(0),
            k_pe.stride(0),
            eps,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return normed, latent_cache


@kv_norm_pack.register_fake
def _kv_norm_pack_fake(
    compressed_kv: torch.Tensor,
    k_pe: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        compressed_kv.new_empty((compressed_kv.shape[0], 512)),
        compressed_kv.new_empty((compressed_kv.shape[0], 576)),
    )
