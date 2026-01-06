# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({"TOKENS_PER_BLOCK": 4}, num_warps=4, num_stages=2),
        triton.Config({"TOKENS_PER_BLOCK": 8}, num_warps=4, num_stages=2),
        triton.Config({"TOKENS_PER_BLOCK": 16}, num_warps=8, num_stages=2),
        triton.Config({"TOKENS_PER_BLOCK": 4}, num_warps=8, num_stages=2),
        triton.Config({"TOKENS_PER_BLOCK": 1}, num_warps=1, num_stages=1),
    ],
    key=["L"],
)
@triton.jit
def quant_per_token_block128_int8_kernel(
    Input,
    Output,
    Scale,
    L,
    stride_iz,
    stride_ih,
    stride_in,
    stride_oz,
    stride_oh,
    stride_on,
    stride_sz,
    stride_sh,
    stride_sn,
    C: tl.constexpr,
    BLK: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
):
    """
    Quantize bfloat16 input to int8 per token with block size 128 along head_dim dimension.
    Since head_dim=128 and BLK=128, each token has exactly 1 block.

    Each block processes TOKENS_PER_BLOCK=4 tokens in parallel using 2D tensor operations.
    - 2D tensor shape: [4, 128] = 512 elements per iteration
    - No nested loops, fully vectorized

    Input: bfloat16 tensor
    Output: int8 tensor + float32 scale
    Grid: (num_heads, num_sm, batch_size) with grid-stride loop over tokens
    """
    # Get program IDs
    pid_h = tl.program_id(0)  # head index
    pid_sm = tl.program_id(1)  # SM index
    pid_b = tl.program_id(2)  # batch index

    # Grid dimensions
    num_sm = tl.num_programs(1)

    # Pre-compute loop-invariant values
    offs_token = tl.arange(0, TOKENS_PER_BLOCK)[:, None]  # [4, 1]
    offs_c = tl.arange(0, BLK)[None, :]  # [1, 128]
    mask_c = offs_c < C  # [1, 128]
    offs_token_1d = tl.arange(0, TOKENS_PER_BLOCK)  # [4]

    base_ptr_in = Input + pid_b * stride_iz + pid_h * stride_ih
    base_ptr_out = Output + pid_b * stride_oz + pid_h * stride_oh
    base_ptr_scale = Scale + pid_b * stride_sz + pid_h * stride_sh

    # Grid-stride loop over tokens, processing TOKENS_PER_BLOCK at a time
    for off_n_base in range(pid_sm * TOKENS_PER_BLOCK, L, num_sm * TOKENS_PER_BLOCK):
        # Compute token indices and masks
        token_ids = off_n_base + offs_token  # [4, 1]
        mask_token = token_ids < L  # [4, 1]
        mask_2d = mask_token & mask_c  # [4, 128]

        # Compute pointers for 2D access
        input_ptrs = base_ptr_in + token_ids * stride_in + offs_c  # [4, 128]
        output_ptrs = base_ptr_out + token_ids * stride_on + offs_c  # [4, 128]

        # Load all data at once: [4, 128]
        x = tl.load(input_ptrs, mask=mask_2d, other=0.0, eviction_policy="evict_last")
        x = x.to(tl.float32)

        # Compute scale for each token (reduce along axis=1)
        abs_x = tl.abs(x)  # [4, 128]
        scale = tl.max(abs_x, axis=1, keep_dims=True) / 127.0  # [4, 1]
        scale = tl.maximum(scale, 1e-12)  # Avoid division by zero

        # Quantize all tokens in parallel
        x_int8 = x / scale  # Broadcasting: [4, 128] / [4, 1]
        # Optimized rounding: round away from zero
        x_int8 = tl.where(x_int8 >= 0, x_int8 + 0.5, x_int8 - 0.5)
        x_int8 = x_int8.to(tl.int8)

        # Store int8 output for all tokens
        tl.store(output_ptrs, x_int8, mask=mask_2d, eviction_policy="evict_first")

        # Store scales for all tokens
        token_ids_1d = off_n_base + offs_token_1d  # [4]
        mask_token_1d = token_ids_1d < L  # [4]

        scale_ptrs = base_ptr_scale + token_ids_1d * stride_sn  # [4]
        scale_to_store = tl.reshape(scale, [TOKENS_PER_BLOCK])  # [4, 1] -> [4]
        tl.store(scale_ptrs, scale_to_store, mask=mask_token_1d, eviction_policy="evict_first")


@triton.autotune(
    configs=[
        triton.Config({"TOKENS_PER_BLOCK": 1}, num_warps=1, num_stages=1),  # Very short sequences
        triton.Config({"TOKENS_PER_BLOCK": 4}, num_warps=4, num_stages=2),
        triton.Config({"TOKENS_PER_BLOCK": 8}, num_warps=4, num_stages=2),
        triton.Config({"TOKENS_PER_BLOCK": 16}, num_warps=8, num_stages=2),
        triton.Config({"TOKENS_PER_BLOCK": 4}, num_warps=8, num_stages=2),
    ],
    key=["L"],
)
@triton.jit
def dequant_per_token_block128_kernel(
    Input,
    Scale,
    Output,
    L,
    stride_iz,
    stride_ih,
    stride_in,
    stride_sz,
    stride_sh,
    stride_sn,
    stride_oz,
    stride_oh,
    stride_on,
    C: tl.constexpr,
    BLK: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
):
    """
    Dequantize int8 tensor back to bfloat16 using per-token block-128 scales.
    Since head_dim=128 and BLK=128, each token has exactly 1 block.

    Each block processes TOKENS_PER_BLOCK=4 tokens in parallel using 2D tensor operations.
    - 2D tensor shape: [4, 128] = 512 elements per iteration
    - No nested loops, fully vectorized

    Grid: (num_heads, num_sm, batch_size) with grid-stride loop over tokens
    """
    # Get program IDs
    pid_h = tl.program_id(0)  # head index
    pid_sm = tl.program_id(1)  # SM index
    pid_b = tl.program_id(2)  # batch index

    # Grid dimensions
    num_sm = tl.num_programs(1)

    # Pre-compute loop-invariant values
    offs_token = tl.arange(0, TOKENS_PER_BLOCK)[:, None]  # [4, 1]
    offs_c = tl.arange(0, BLK)[None, :]  # [1, 128]
    mask_c = offs_c < C  # [1, 128]
    offs_token_1d = tl.arange(0, TOKENS_PER_BLOCK)  # [4]

    base_ptr_in = Input + pid_b * stride_iz + pid_h * stride_ih
    base_ptr_scale = Scale + pid_b * stride_sz + pid_h * stride_sh
    base_ptr_out = Output + pid_b * stride_oz + pid_h * stride_oh

    # Grid-stride loop over tokens, processing TOKENS_PER_BLOCK at a time
    for off_n_base in range(pid_sm * TOKENS_PER_BLOCK, L, num_sm * TOKENS_PER_BLOCK):
        # Compute token indices and masks
        token_ids = off_n_base + offs_token  # [4, 1]
        mask_token = token_ids < L  # [4, 1]
        mask_2d = mask_token & mask_c  # [4, 128]

        # Compute pointers for 2D access
        input_ptrs = base_ptr_in + token_ids * stride_in + offs_c  # [4, 128]

        # Load int8 data: [4, 128]
        x_int8 = tl.load(input_ptrs, mask=mask_2d, other=0, eviction_policy="evict_last")

        # Load scales for all tokens
        token_ids_1d = off_n_base + offs_token_1d  # [4]
        mask_token_1d = token_ids_1d < L  # [4]

        scale_ptrs = base_ptr_scale + token_ids_1d * stride_sn  # [4]
        scale_1d = tl.load(scale_ptrs, mask=mask_token_1d, other=1.0, eviction_policy="evict_last")  # [4]
        scale = tl.reshape(scale_1d, [TOKENS_PER_BLOCK, 1])  # [4] -> [4, 1] for broadcasting

        # Dequantize all tokens in parallel
        x_bf16 = (x_int8.to(tl.float32) * scale).to(tl.bfloat16)  # [4, 128]

        # Store output for all tokens
        output_ptrs = base_ptr_out + token_ids * stride_on + offs_c  # [4, 128]
        tl.store(output_ptrs, x_bf16, mask=mask_2d, eviction_policy="evict_first")


def quantize_per_token_block128(input_tensor, tensor_layout="HND"):
    """
    Quantize a single input tensor per token with block size 128 along head_dim dimension.
    Each token is divided into blocks of 128 elements along the head_dim dimension.

    Quantization granularity: 1 x 128 (per token, per block in head_dim)
    This function processes ONE input tensor at a time for maximum flexibility.

    Uses grid-stride loop for better SM utilization:
    - Grid: (num_heads, num_sm, batch) where num_sm is the GPU's SM count
    - Each SM processes multiple tokens using a stride loop
    - Avoids grid size limits for large sequence lengths

    Scale layout matches input layout:
    - HND input [b, h, seq_len, 128] -> scale [b, h, seq_len, 1]
    - NHD input [b, seq_len, h, 128] -> scale [b, seq_len, h, 1]

    Args:
        input_tensor: BFloat16 input tensor to quantize
                     HND layout: [batch, heads, seq_len, 128]
                     NHD layout: [batch, seq_len, heads, 128]
        tensor_layout: "HND" or "NHD"

    Returns:
        output_int8: Quantized int8 tensor (same shape as input)
        output_scale: Scale tensor (layout matches input)
                     HND: [batch, heads, seq_len, 1]
                     NHD: [batch, seq_len, heads, 1]

    Example:
        >>> # HND layout
        >>> q = torch.randn(2, 4, 8, 128, dtype=torch.bfloat16, device='cuda')
        >>> q_int8, q_scale = quantize_per_token_block128(q, tensor_layout="HND")
        >>> print(q_int8.shape)  # torch.Size([2, 4, 8, 128])
        >>> print(q_scale.shape)  # torch.Size([2, 4, 8, 1])  # 1 block per token

        >>> # NHD layout
        >>> q = torch.randn(2, 8, 4, 128, dtype=torch.bfloat16, device='cuda')
        >>> q_int8, q_scale = quantize_per_token_block128(q, tensor_layout="NHD")
        >>> print(q_int8.shape)  # torch.Size([2, 8, 4, 128])
        >>> print(q_scale.shape)  # torch.Size([2, 8, 4, 1])  # scale layout matches input!

        >>> # Multi-stream async execution (Triton uses current stream automatically)
        >>> s1 = torch.cuda.Stream()
        >>> with torch.cuda.stream(s1):
        >>>     q_int8, q_scale = quantize_per_token_block128(q, tensor_layout="NHD")
    """
    output_int8 = torch.empty(input_tensor.shape, dtype=torch.int8, device=input_tensor.device)

    if tensor_layout == "HND":
        b, h, seq_len, _ = input_tensor.shape

        stride_bz, stride_h, stride_seq = input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2)
        stride_bz_out, stride_h_out, stride_seq_out = (
            output_int8.stride(0),
            output_int8.stride(1),
            output_int8.stride(2),
        )

        # Scale tensor: [b, h, seq_len, 1] - matches HND layout, 1 block per token
        output_scale = torch.empty((b, h, seq_len, 1), device=input_tensor.device, dtype=torch.float32)

    elif tensor_layout == "NHD":
        b, seq_len, h, _ = input_tensor.shape

        stride_bz, stride_h, stride_seq = input_tensor.stride(0), input_tensor.stride(2), input_tensor.stride(1)
        stride_bz_out, stride_h_out, stride_seq_out = (
            output_int8.stride(0),
            output_int8.stride(2),
            output_int8.stride(1),
        )

        # Scale tensor: [b, seq_len, h, 1] - matches NHD layout, 1 block per token
        output_scale = torch.empty((b, seq_len, h, 1), device=input_tensor.device, dtype=torch.float32)

    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    # Get number of SMs on current GPU
    device_props = torch.cuda.get_device_properties(input_tensor.device)
    num_sm = device_props.multi_processor_count

    # Grid: (num_heads, num_sm, batch_size)
    # Use grid-stride loop to handle all tokens
    # This allows better SM utilization and avoids grid size limits
    if seq_len // num_sm < 100:
        grid = (h, num_sm, b)
    else:
        grid = (h, num_sm * 20, b)

    # Prepare scale strides for kernel
    # Kernel expects: stride_batch, stride_head, stride_seq
    if tensor_layout == "HND":
        # Scale: [b, h, seq_len, 1]
        stride_scale_batch = output_scale.stride(0)
        stride_scale_head = output_scale.stride(1)
        stride_scale_seq = output_scale.stride(2)
    else:  # NHD
        # Scale: [b, seq_len, h, 1]
        stride_scale_batch = output_scale.stride(0)
        stride_scale_head = output_scale.stride(2)  # head is at dim 2
        stride_scale_seq = output_scale.stride(1)  # seq is at dim 1

    quant_per_token_block128_int8_kernel[grid](
        input_tensor,
        output_int8,
        output_scale,
        seq_len,
        stride_bz,
        stride_h,
        stride_seq,
        stride_bz_out,
        stride_h_out,
        stride_seq_out,
        stride_scale_batch,
        stride_scale_head,
        stride_scale_seq,
        C=128,
        BLK=128,
    )

    return output_int8, output_scale


def dequantize_per_token_block128(q_int8, q_scale, tensor_layout="HND"):
    """
    Dequantize int8 tensor back to bfloat16 using per-token block-128 scales
    Since head_dim=128 and BLK=128, each token has exactly 1 block.

    Uses grid-stride loop for better SM utilization (same as quantize):
    - Grid: (num_heads, num_sm, batch) where num_sm is the GPU's SM count
    - Each SM processes multiple tokens using a stride loop

    Scale layout matches input layout:
    - HND: q_int8 [b, h, seq_len, 128], q_scale [b, h, seq_len, 1]
    - NHD: q_int8 [b, seq_len, h, 128], q_scale [b, seq_len, h, 1]

    Args:
        q_int8: int8 tensor (HND: [b, h, seq_len, 128] or NHD: [b, seq_len, h, 128])
        q_scale: float32 scale tensor (layout matches q_int8)
                 HND: [b, h, seq_len, 1]
                 NHD: [b, seq_len, h, 1]
        tensor_layout: "HND" or "NHD"

    Returns:
        q_bf16: dequantized bfloat16 tensor in same layout as q_int8
    """
    # Check scale dimensions
    if q_scale.ndim == 3:
        raise ValueError(
            f"ERROR: q_scale has {q_scale.ndim} dimensions (shape: {q_scale.shape}), "
            f"but this function expects 4 dimensions.\n"
            f"You are likely using an OLD quantization function that generates 3D scales.\n"
            f"Please use 'quantize_per_token_block128()' which generates 4D scales:\n"
            f"  - For HND: scale shape should be [b, h, seq_len, 1]\n"
            f"  - For NHD: scale shape should be [b, seq_len, h, 1]\n"
            f"Current scale shape: {q_scale.shape}"
        )

    if q_scale.ndim != 4:
        raise ValueError(f"q_scale must be 4-dimensional, got shape {q_scale.shape} ({q_scale.ndim}D)")

    output_bf16 = torch.empty(q_int8.shape, dtype=torch.bfloat16, device=q_int8.device)

    if tensor_layout == "HND":
        b, h, seq_len, _ = q_int8.shape

        stride_bz, stride_h, stride_seq = q_int8.stride(0), q_int8.stride(1), q_int8.stride(2)
        stride_bz_out, stride_h_out, stride_seq_out = (
            output_bf16.stride(0),
            output_bf16.stride(1),
            output_bf16.stride(2),
        )

        # Scale strides
        stride_scale_batch = q_scale.stride(0)
        stride_scale_head = q_scale.stride(1)
        stride_scale_seq = q_scale.stride(2)

    elif tensor_layout == "NHD":
        b, seq_len, h, _ = q_int8.shape

        stride_bz, stride_h, stride_seq = q_int8.stride(0), q_int8.stride(2), q_int8.stride(1)
        stride_bz_out, stride_h_out, stride_seq_out = (
            output_bf16.stride(0),
            output_bf16.stride(2),
            output_bf16.stride(1),
        )

        # Scale strides
        stride_scale_batch = q_scale.stride(0)
        stride_scale_head = q_scale.stride(2)  # head is at dim 2
        stride_scale_seq = q_scale.stride(1)  # seq is at dim 1

    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    # Get number of SMs on current GPU
    device_props = torch.cuda.get_device_properties(q_int8.device)
    num_sm = device_props.multi_processor_count

    # Grid: (num_heads, num_sm, batch_size)
    # Use grid-stride loop to handle all tokens
    if seq_len // num_sm < 100:
        grid = (h, num_sm, b)
    else:
        grid = (h, num_sm * 20, b)

    dequant_per_token_block128_kernel[grid](
        q_int8,
        q_scale,
        output_bf16,
        seq_len,
        stride_bz,
        stride_h,
        stride_seq,
        stride_scale_batch,
        stride_scale_head,
        stride_scale_seq,
        stride_bz_out,
        stride_h_out,
        stride_seq_out,
        C=128,
        BLK=128,
    )

    return output_bf16


@triton.autotune(
    configs=[
        triton.Config({"TOKENS_PER_BLOCK": 1}, num_warps=1, num_stages=1),  # Very short sequences
        triton.Config({"TOKENS_PER_BLOCK": 4}, num_warps=4, num_stages=2),
        triton.Config({"TOKENS_PER_BLOCK": 8}, num_warps=4, num_stages=2),
        triton.Config({"TOKENS_PER_BLOCK": 16}, num_warps=8, num_stages=2),
        triton.Config({"TOKENS_PER_BLOCK": 4}, num_warps=8, num_stages=2),
    ],
    key=["L"],
)
@triton.jit
def quant_per_token_block128_packed_kernel(
    Input,
    Output,
    L,
    stride_iz,
    stride_ih,
    stride_in,
    stride_oz,
    stride_oh,
    stride_on,
    C: tl.constexpr,
    BLK: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
):
    """
    Quantize bfloat16 input to packed int8 tensor (int8 data + scale bytes).
    Output format: [..., seq_len, 132] where each token has:
      - 128 bytes: int8 quantized data
      - 4 bytes: float32 scale as bytes

    This avoids later view() operations and packing overhead.
    """
    # Get program IDs
    pid_h = tl.program_id(0)  # head index
    pid_sm = tl.program_id(1)  # SM index
    pid_b = tl.program_id(2)  # batch index

    # Grid dimensions
    num_sm = tl.num_programs(1)

    # Pre-compute loop-invariant values
    offs_token = tl.arange(0, TOKENS_PER_BLOCK)[:, None]  # [4, 1]
    offs_c = tl.arange(0, BLK)[None, :]  # [1, 128]
    mask_c = offs_c < C  # [1, 128]
    offs_token_1d = tl.arange(0, TOKENS_PER_BLOCK)  # [4]

    base_ptr_in = Input + pid_b * stride_iz + pid_h * stride_ih
    base_ptr_out = Output + pid_b * stride_oz + pid_h * stride_oh

    # Grid-stride loop over tokens, processing TOKENS_PER_BLOCK at a time
    for off_n_base in range(pid_sm * TOKENS_PER_BLOCK, L, num_sm * TOKENS_PER_BLOCK):
        # Compute token indices and masks
        token_ids = off_n_base + offs_token  # [4, 1]
        mask_token = token_ids < L  # [4, 1]
        mask_2d = mask_token & mask_c  # [4, 128]

        # Compute pointers for 2D access
        input_ptrs = base_ptr_in + token_ids * stride_in + offs_c  # [4, 128]

        # Load all data at once: [4, 128]
        x = tl.load(input_ptrs, mask=mask_2d, other=0.0, eviction_policy="evict_last")
        x = x.to(tl.float32)

        # Compute scale for each token (reduce along axis=1)
        abs_x = tl.abs(x)  # [4, 128]
        scale = tl.max(abs_x, axis=1, keep_dims=True) / 127.0  # [4, 1]
        scale = tl.maximum(scale, 1e-12)  # Avoid division by zero

        # Quantize all tokens in parallel
        x_int8 = x / scale  # Broadcasting: [4, 128] / [4, 1]
        # Optimized rounding: round away from zero
        x_int8 = tl.where(x_int8 >= 0, x_int8 + 0.5, x_int8 - 0.5)
        x_int8 = x_int8.to(tl.int8)

        # Store int8 output for all tokens (first 128 bytes)
        output_ptrs = base_ptr_out + token_ids * stride_on + offs_c  # [4, 128]
        tl.store(output_ptrs, x_int8, mask=mask_2d, eviction_policy="evict_first")

        # Store scale as 4 bytes (last 4 bytes of the 132-byte block)
        # Directly store float32 scale as int32 (1 store instead of 4)
        scale_1d = tl.reshape(scale, [TOKENS_PER_BLOCK])  # [4, 1] -> [4]

        token_ids_1d = off_n_base + offs_token_1d  # [4]
        mask_token_1d = token_ids_1d < L  # [4]

        # Store scale after the 128 int8 values (cast pointer to float32 type)
        scale_byte_offset = (token_ids_1d * stride_on + 128).to(tl.int64)
        scale_ptrs = (base_ptr_out + scale_byte_offset).to(tl.pointer_type(tl.float32))
        tl.store(scale_ptrs, scale_1d, mask=mask_token_1d, eviction_policy="evict_first")


@triton.autotune(
    configs=[
        triton.Config({"TOKENS_PER_BLOCK": 1}, num_warps=1, num_stages=1),  # Very short sequences
        triton.Config({"TOKENS_PER_BLOCK": 4}, num_warps=4, num_stages=2),
        triton.Config({"TOKENS_PER_BLOCK": 8}, num_warps=4, num_stages=2),
        triton.Config({"TOKENS_PER_BLOCK": 16}, num_warps=8, num_stages=2),
        triton.Config({"TOKENS_PER_BLOCK": 4}, num_warps=8, num_stages=2),
    ],
    key=["L"],
)
@triton.jit
def dequant_per_token_block128_packed_kernel(
    Input,
    Output,
    L,
    stride_iz,
    stride_ih,
    stride_in,
    stride_oz,
    stride_oh,
    stride_on,
    C: tl.constexpr,
    BLK: tl.constexpr,
    TOKENS_PER_BLOCK: tl.constexpr,
):
    """
    Dequantize packed int8 tensor (int8 data + scale bytes) back to bfloat16.
    Input format: [..., seq_len, 132] where each token has:
      - 128 bytes: int8 quantized data
      - 4 bytes: float32 scale as bytes
    """
    # Get program IDs
    pid_h = tl.program_id(0)  # head index
    pid_sm = tl.program_id(1)  # SM index
    pid_b = tl.program_id(2)  # batch index

    # Grid dimensions
    num_sm = tl.num_programs(1)

    # Pre-compute loop-invariant values
    offs_token = tl.arange(0, TOKENS_PER_BLOCK)[:, None]  # [4, 1]
    offs_c = tl.arange(0, BLK)[None, :]  # [1, 128]
    mask_c = offs_c < C  # [1, 128]
    offs_token_1d = tl.arange(0, TOKENS_PER_BLOCK)  # [4]

    base_ptr_in = Input + pid_b * stride_iz + pid_h * stride_ih
    base_ptr_out = Output + pid_b * stride_oz + pid_h * stride_oh

    # Grid-stride loop over tokens, processing TOKENS_PER_BLOCK at a time
    for off_n_base in range(pid_sm * TOKENS_PER_BLOCK, L, num_sm * TOKENS_PER_BLOCK):
        # Compute token indices and masks
        token_ids = off_n_base + offs_token  # [4, 1]
        mask_token = token_ids < L  # [4, 1]
        mask_2d = mask_token & mask_c  # [4, 128]

        # Load int8 data (first 128 bytes)
        input_ptrs = base_ptr_in + token_ids * stride_in + offs_c  # [4, 128]
        x_int8 = tl.load(input_ptrs, mask=mask_2d, other=0, eviction_policy="evict_last")

        # Load scale from 4 bytes (last 4 bytes of the 132-byte block)
        # Directly load float32 scale (1 load instead of 4)
        token_ids_1d = off_n_base + offs_token_1d  # [4]
        mask_token_1d = token_ids_1d < L  # [4]

        # Load scale after the 128 int8 values (cast pointer to float32 type)
        scale_byte_offset = (token_ids_1d * stride_in + 128).to(tl.int64)
        scale_ptrs = (base_ptr_in + scale_byte_offset).to(tl.pointer_type(tl.float32))
        scale_1d = tl.load(scale_ptrs, mask=mask_token_1d, other=1e-12, eviction_policy="evict_last")
        scale = tl.reshape(scale_1d, [TOKENS_PER_BLOCK, 1])  # [4] -> [4, 1] for broadcasting

        # Dequantize all tokens in parallel
        x_bf16 = (x_int8.to(tl.float32) * scale).to(tl.bfloat16)  # [4, 128]

        # Store output for all tokens
        output_ptrs = base_ptr_out + token_ids * stride_on + offs_c  # [4, 128]
        tl.store(output_ptrs, x_bf16, mask=mask_2d, eviction_policy="evict_first")


def quantize_per_token_block128_packed(input_tensor, tensor_layout="HND"):
    """
    Quantize input tensor and return packed format (int8 + scale bytes).
    Output shape: [..., seq_len, 132] where each token has:
      - 128 bytes: int8 quantized data
      - 4 bytes: float32 scale as bytes

    This avoids separate scale tensor and later packing operations.

    Args:
        input_tensor: BFloat16 input tensor to quantize
                     HND layout: [batch, heads, seq_len, 128]
                     NHD layout: [batch, seq_len, heads, 128]
        tensor_layout: "HND" or "NHD"

    Returns:
        output_packed: Packed int8 tensor
                      HND: [batch, heads, seq_len, 132]
                      NHD: [batch, seq_len, heads, 132]
    """
    if tensor_layout == "HND":
        b, h, seq_len, _ = input_tensor.shape
        output_shape = (b, h, seq_len, 132)
        stride_bz, stride_h, stride_seq = input_tensor.stride(0), input_tensor.stride(1), input_tensor.stride(2)
    elif tensor_layout == "NHD":
        b, seq_len, h, _ = input_tensor.shape
        output_shape = (b, seq_len, h, 132)
        stride_bz, stride_h, stride_seq = input_tensor.stride(0), input_tensor.stride(2), input_tensor.stride(1)
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    # Create output tensor
    output_packed = torch.empty(output_shape, dtype=torch.int8, device=input_tensor.device)

    if tensor_layout == "HND":
        stride_bz_out, stride_h_out, stride_seq_out = (
            output_packed.stride(0),
            output_packed.stride(1),
            output_packed.stride(2),
        )
    else:  # NHD
        stride_bz_out, stride_h_out, stride_seq_out = (
            output_packed.stride(0),
            output_packed.stride(2),
            output_packed.stride(1),
        )

    # Get number of SMs on current GPU
    device_props = torch.cuda.get_device_properties(input_tensor.device)
    num_sm = device_props.multi_processor_count

    if seq_len // num_sm < 100:
        grid = (h, num_sm, b)
    else:
        grid = (h, num_sm * 20, b)

    quant_per_token_block128_packed_kernel[grid](
        input_tensor,
        output_packed,
        seq_len,
        stride_bz,
        stride_h,
        stride_seq,
        stride_bz_out,
        stride_h_out,
        stride_seq_out,
        C=128,
        BLK=128,
    )

    return output_packed


def dequantize_per_token_block128_packed(packed_tensor, tensor_layout="HND"):
    """
    Dequantize packed tensor (int8 + scale bytes) back to bfloat16.

    Args:
        packed_tensor: Packed int8 tensor
                      HND: [batch, heads, seq_len, 132]
                      NHD: [batch, seq_len, heads, 132]
        tensor_layout: "HND" or "NHD"

    Returns:
        output: Dequantized bfloat16 tensor
               HND: [batch, heads, seq_len, 128]
               NHD: [batch, seq_len, heads, 128]
    """
    if tensor_layout == "HND":
        b, h, seq_len, _ = packed_tensor.shape
        output_shape = (b, h, seq_len, 128)
        stride_bz, stride_h, stride_seq = packed_tensor.stride(0), packed_tensor.stride(1), packed_tensor.stride(2)
    elif tensor_layout == "NHD":
        b, seq_len, h, _ = packed_tensor.shape
        output_shape = (b, seq_len, h, 128)
        stride_bz, stride_h, stride_seq = packed_tensor.stride(0), packed_tensor.stride(2), packed_tensor.stride(1)
    else:
        raise ValueError(f"Unknown tensor layout: {tensor_layout}")

    output_bf16 = torch.empty(output_shape, dtype=torch.bfloat16, device=packed_tensor.device)

    if tensor_layout == "HND":
        stride_bz_out, stride_h_out, stride_seq_out = (
            output_bf16.stride(0),
            output_bf16.stride(1),
            output_bf16.stride(2),
        )
    else:  # NHD
        stride_bz_out, stride_h_out, stride_seq_out = (
            output_bf16.stride(0),
            output_bf16.stride(2),
            output_bf16.stride(1),
        )

    # Get number of SMs on current GPU
    device_props = torch.cuda.get_device_properties(packed_tensor.device)
    num_sm = device_props.multi_processor_count

    if seq_len // num_sm < 100:
        grid = (h, num_sm, b)
    else:
        grid = (h, num_sm * 20, b)

    dequant_per_token_block128_packed_kernel[grid](
        packed_tensor,
        output_bf16,
        seq_len,
        stride_bz,
        stride_h,
        stride_seq,
        stride_bz_out,
        stride_h_out,
        stride_seq_out,
        C=128,
        BLK=128,
    )

    return output_bf16


def compute_cosine_similarity(a, b):
    """
    Compute cosine similarity between two tensors
    Args:
        a, b: tensors of same shape
    Returns:
        cosine similarity (scalar)
    """
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    return F.cosine_similarity(a_flat.unsqueeze(0), b_flat.unsqueeze(0)).item()


def test_hnd_layout():
    """Test basic functionality with HND layout"""
    print("\n[Test: HND layout]")

    device = "cuda"
    dtype = torch.bfloat16
    batch, heads, seq_len = 2, 40, 3510

    # Generate data in range [-2000, 2000]
    q = (torch.rand(batch, heads, seq_len, 128, dtype=torch.float32, device=device) - 0.5) * 400
    k = (torch.rand(batch, heads, seq_len, 128, dtype=torch.float32, device=device) - 0.5) * 400
    q = q.to(dtype)
    k = k.to(dtype)

    # Run quantization
    q_int8, q_scale = quantize_per_token_block128(q, tensor_layout="HND")
    k_int8, k_scale = quantize_per_token_block128(k, tensor_layout="HND")

    # Assertions
    assert q_int8.dtype == torch.int8
    assert q_scale.shape == (batch, heads, seq_len, 1)
    assert q_int8.min() >= -128 and q_int8.max() <= 127

    # Dequantize and check reconstruction
    q_dequant = dequantize_per_token_block128(q_int8, q_scale, tensor_layout="HND")
    k_dequant = dequantize_per_token_block128(k_int8, k_scale, tensor_layout="HND")

    q_expected = q.float()
    k_expected = k.float()

    q_cos_sim = compute_cosine_similarity(q_dequant, q_expected)
    k_cos_sim = compute_cosine_similarity(k_dequant, k_expected)

    q_error = torch.abs(q_dequant - q_expected).max().item()
    q_rel_error = (torch.abs(q_dequant - q_expected) / (torch.abs(q_expected) + 1e-6)).mean().item()

    print(f"  Q: cos_sim={q_cos_sim:.6f}, max_err={q_error:.2f}, rel_err={q_rel_error:.4f}")
    print(
        f"  K: cos_sim={k_cos_sim:.6f}, max_err={torch.abs(k_dequant - k_expected).max().item():.2f}, rel_err={(torch.abs(k_dequant - k_expected) / (torch.abs(k_expected) + 1e-6)).mean().item():.4f}"
    )

    assert q_cos_sim > 0.99, f"Q cosine similarity too low: {q_cos_sim:.6f}"
    assert k_cos_sim > 0.99, f"K cosine similarity too low: {k_cos_sim:.6f}"

    return True


def test_hnd_layout_packed():
    """Test packed quantization version"""
    print("\n[Test: Packed version (HND layout)]")

    device = "cuda"
    dtype = torch.bfloat16
    batch, heads, seq_len = 2, 40, 3510

    # Generate data in range [-2000, 2000]
    q = (torch.rand(batch, heads, seq_len, 128, dtype=torch.float32, device=device) - 0.5) * 923
    k = (torch.rand(batch, heads, seq_len, 128, dtype=torch.float32, device=device) - 0.5) * 923
    q = q.to(dtype)
    k = k.to(dtype)

    # Test packed version
    q_packed = quantize_per_token_block128_packed(q, tensor_layout="HND")
    k_packed = quantize_per_token_block128_packed(k, tensor_layout="HND")

    # Assertions
    assert q_packed.shape == (
        batch,
        heads,
        seq_len,
        132,
    ), f"Expected shape {(batch, heads, seq_len, 132)}, got {q_packed.shape}"
    assert q_packed.dtype == torch.int8, f"Expected dtype int8, got {q_packed.dtype}"

    # Dequantize
    q_dequant_packed = dequantize_per_token_block128_packed(q_packed, tensor_layout="HND")
    k_dequant_packed = dequantize_per_token_block128_packed(k_packed, tensor_layout="HND")

    # Compare with original data
    q_expected = q.float()
    k_expected = k.float()

    q_cos_sim = compute_cosine_similarity(q_dequant_packed, q_expected)
    k_cos_sim = compute_cosine_similarity(k_dequant_packed, k_expected)

    q_error = torch.abs(q_dequant_packed - q_expected).max().item()
    q_rel_error = (torch.abs(q_dequant_packed - q_expected) / (torch.abs(q_expected) + 1e-6)).mean().item()

    print(f"  Q: cos_sim={q_cos_sim:.6f}, max_err={q_error:.2f}, rel_err={q_rel_error:.4f}")
    print(
        f"  K: cos_sim={k_cos_sim:.6f}, max_err={torch.abs(k_dequant_packed - k_expected).max().item():.2f}, rel_err={(torch.abs(k_dequant_packed - k_expected) / (torch.abs(k_expected) + 1e-6)).mean().item():.4f}"
    )

    assert q_cos_sim > 0.99, f"Q cosine similarity too low: {q_cos_sim:.6f}"
    assert k_cos_sim > 0.99, f"K cosine similarity too low: {k_cos_sim:.6f}"

    return True


def test_nhd_layout():
    """Test with NHD layout"""
    print("Test: NHD layout (standard version)")

    device = "cuda"
    dtype = torch.bfloat16

    # NHD layout: [batch, seq_len, heads, 128]
    batch, seq_len, heads = 2, 3510, 40

    # Generate data in range [-2000, 2000]
    q = (torch.rand(batch, seq_len, heads, 128, dtype=torch.float32, device=device) - 0.5) * 1500
    k = (torch.rand(batch, seq_len, heads, 128, dtype=torch.float32, device=device) - 0.5) * 1500
    q = q.to(dtype)
    k = k.to(dtype)

    # Run standard quantization
    q_int8, q_scale = quantize_per_token_block128(q, tensor_layout="NHD")
    k_int8, k_scale = quantize_per_token_block128(k, tensor_layout="NHD")

    # Assertions
    assert q_int8.dtype == torch.int8
    assert q_scale.shape == (batch, seq_len, heads, 1)
    assert q_int8.min() >= -128 and q_int8.max() <= 127

    # Dequantize and check reconstruction
    q_dequant = dequantize_per_token_block128(q_int8, q_scale, tensor_layout="NHD")
    k_dequant = dequantize_per_token_block128(k_int8, k_scale, tensor_layout="NHD")

    q_expected = q.float()
    k_expected = k.float()

    q_cos_sim = compute_cosine_similarity(q_dequant, q_expected)
    k_cos_sim = compute_cosine_similarity(k_dequant, k_expected)

    q_error = torch.abs(q_dequant - q_expected).max().item()
    q_rel_error = (torch.abs(q_dequant - q_expected) / (torch.abs(q_expected) + 1e-6)).mean().item()

    print(f"  Q: cos_sim={q_cos_sim:.6f}, max_err={q_error:.2f}, rel_err={q_rel_error:.4f}")
    print(
        f"  K: cos_sim={k_cos_sim:.6f}, max_err={torch.abs(k_dequant - k_expected).max().item():.2f}, rel_err={(torch.abs(k_dequant - k_expected) / (torch.abs(k_expected) + 1e-6)).mean().item():.4f}"
    )

    assert q_cos_sim > 0.99, f"Q cosine similarity too low: {q_cos_sim:.6f}"
    assert k_cos_sim > 0.99, f"K cosine similarity too low: {k_cos_sim:.6f}"

    return True


def test_nhd_layout_packed():
    """Test with NHD layout - packed version"""
    print("Test: NHD layout (packed version)")

    device = "cuda"
    dtype = torch.bfloat16

    # NHD layout: [batch, seq_len, heads, 128]
    batch, seq_len, heads = 2, 3510, 40

    # Generate data in range [-2000, 2000]
    q = (torch.rand(batch, seq_len, heads, 128, dtype=torch.float32, device=device) - 0.5) * 200
    k = (torch.rand(batch, seq_len, heads, 128, dtype=torch.float32, device=device) - 0.5) * 200
    q = q.to(dtype)
    k = k.to(dtype)

    # Test packed version
    q_packed = quantize_per_token_block128_packed(q, tensor_layout="NHD")
    k_packed = quantize_per_token_block128_packed(k, tensor_layout="NHD")

    # Assertions
    assert q_packed.shape == (
        batch,
        seq_len,
        heads,
        132,
    ), f"Expected shape {(batch, seq_len, heads, 132)}, got {q_packed.shape}"
    assert q_packed.dtype == torch.int8

    # Dequantize
    q_dequant_packed = dequantize_per_token_block128_packed(q_packed, tensor_layout="NHD")
    k_dequant_packed = dequantize_per_token_block128_packed(k_packed, tensor_layout="NHD")

    # Compare with original data
    q_expected = q.float()
    k_expected = k.float()

    q_cos_sim = compute_cosine_similarity(q_dequant_packed, q_expected)
    k_cos_sim = compute_cosine_similarity(k_dequant_packed, k_expected)

    q_error = torch.abs(q_dequant_packed - q_expected).max().item()
    q_rel_error = (torch.abs(q_dequant_packed - q_expected) / (torch.abs(q_expected) + 1e-6)).mean().item()

    k_error = torch.abs(k_dequant_packed - k_expected).max().item()
    k_rel_error = (torch.abs(k_dequant_packed - k_expected) / (torch.abs(k_expected) + 1e-6)).mean().item()

    print(f"  Q: cos_sim={q_cos_sim:.6f}, max_err={q_error:.2f}, rel_err={q_rel_error:.4f}")
    print(f"  K: cos_sim={k_cos_sim:.6f}, max_err={k_error:.2f}, rel_err={k_rel_error:.4f}")

    assert q_cos_sim > 0.99, f"Q cosine similarity too low: {q_cos_sim:.6f}"
    assert k_cos_sim > 0.99, f"K cosine similarity too low: {k_cos_sim:.6f}"

    return True


def test_nhd_profile(batch, seq_len, heads):
    """Test with NHD layout - performance profiling (standard version)"""
    print(f"\n[Performance: NHD [{batch}, {seq_len}, {heads}, 128] - Standard Version]")

    device = "cuda"
    dtype = torch.bfloat16

    # Pre-generate multiple sets of data to avoid L2 cache hits
    warm_up_time = 2
    run_time = 10

    print(f"Pre-generating {warm_up_time + run_time} sets of data...")
    q_list = []
    for i in range(warm_up_time + run_time):
        q = (torch.rand(batch, seq_len, heads, 128, dtype=torch.float32, device=device) - 0.5) * 4000
        q = q.to(dtype)
        q_list.append(q)

    # Run quantization (using new single-input function)
    import time

    # Warm-up with first warm_up_time datasets
    for i in range(warm_up_time):
        q_int8, q_scale = quantize_per_token_block128(q_list[i], tensor_layout="NHD")

    torch.cuda.synchronize()
    start_time = time.time()
    # Performance test with next run_time datasets
    for i in range(run_time):
        q_int8, q_scale = quantize_per_token_block128(q_list[warm_up_time + i], tensor_layout="NHD")
    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    avg_time = total_time / run_time

    # Calculate bandwidth
    input_size = batch * seq_len * heads * 128 * 2  # bf16 = 2 bytes
    output_size = batch * seq_len * heads * 128 * 1  # int8 = 1 byte
    scale_size = batch * seq_len * heads * 1 * 4  # fp32 = 4 bytes
    total_bytes = input_size + output_size + scale_size
    bandwidth_gb_s = (total_bytes / 1e9) / avg_time

    print(f"Quantize:   {avg_time*1000:.3f} ms, {bandwidth_gb_s:.1f} GB/s")

    # Pre-quantize all datasets for dequantization test
    print(f"Pre-quantizing {warm_up_time + run_time} sets of data...")
    q_int8_list = []
    q_scale_list = []
    for i in range(warm_up_time + run_time):
        q_int8_tmp, q_scale_tmp = quantize_per_token_block128(q_list[i], tensor_layout="NHD")
        q_int8_list.append(q_int8_tmp)
        q_scale_list.append(q_scale_tmp)

    # Warm-up dequantization
    for i in range(warm_up_time):
        q_dequant = dequantize_per_token_block128(q_int8_list[i], q_scale_list[i], tensor_layout="NHD")

    torch.cuda.synchronize()
    start_time = time.time()
    # Performance test dequantization
    for i in range(run_time):
        q_dequant = dequantize_per_token_block128(
            q_int8_list[warm_up_time + i], q_scale_list[warm_up_time + i], tensor_layout="NHD"
        )
    torch.cuda.synchronize()
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / run_time

    # Calculate bandwidth for dequant
    input_size = batch * seq_len * heads * 128 * 1  # int8 = 1 byte
    scale_size = batch * seq_len * heads * 1 * 4  # fp32 = 4 bytes
    output_size = batch * seq_len * heads * 128 * 2  # bf16 = 2 bytes
    total_bytes = input_size + scale_size + output_size
    bandwidth_gb_s = (total_bytes / 1e9) / avg_time

    print(f"Dequantize: {avg_time*1000:.3f} ms, {bandwidth_gb_s:.1f} GB/s")

    return True


def test_nhd_profile_packed(batch, seq_len, heads):
    """Test with NHD layout - performance profiling (packed version)"""
    print(f"\n[Performance: NHD [{batch}, {seq_len}, {heads}, 128] - Packed Version]")

    device = "cuda"
    dtype = torch.bfloat16

    # Pre-generate multiple sets of data to avoid L2 cache hits
    warm_up_time = 2
    run_time = 10

    print(f"Pre-generating {warm_up_time + run_time} sets of data...")
    q_list = []
    for i in range(warm_up_time + run_time):
        q = (torch.rand(batch, seq_len, heads, 128, dtype=torch.float32, device=device) - 0.5) * 4000
        q = q.to(dtype)
        q_list.append(q)

    import time

    # Warm-up packed quantization
    for i in range(warm_up_time):
        q_packed = quantize_per_token_block128_packed(q_list[i], tensor_layout="NHD")

    torch.cuda.synchronize()
    start_time = time.time()
    # Performance test packed quantization
    for i in range(run_time):
        q_packed = quantize_per_token_block128_packed(q_list[warm_up_time + i], tensor_layout="NHD")
    torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_packed_quant = total_time / run_time

    # Calculate bandwidth for packed quantization
    input_size = batch * seq_len * heads * 128 * 2  # bf16 = 2 bytes
    output_size = batch * seq_len * heads * 132 * 1  # packed int8 = 132 bytes (128 + 4)
    total_bytes = input_size + output_size
    bandwidth_gb_s = (total_bytes / 1e9) / avg_time_packed_quant

    print(f"Quantize:   {avg_time_packed_quant*1000:.3f} ms, {bandwidth_gb_s:.1f} GB/s")

    # Pre-quantize packed datasets for dequantization test
    print(f"Pre-quantizing {warm_up_time + run_time} sets of packed data...")
    q_packed_list = []
    for i in range(warm_up_time + run_time):
        q_packed_tmp = quantize_per_token_block128_packed(q_list[i], tensor_layout="NHD")
        q_packed_list.append(q_packed_tmp)

    # Warm-up packed dequantization
    for i in range(warm_up_time):
        q_dequant_packed = dequantize_per_token_block128_packed(q_packed_list[i], tensor_layout="NHD")

    torch.cuda.synchronize()
    start_time = time.time()
    # Performance test packed dequantization
    for i in range(run_time):
        q_dequant_packed = dequantize_per_token_block128_packed(q_packed_list[warm_up_time + i], tensor_layout="NHD")
    torch.cuda.synchronize()
    end_time = time.time()
    total_time = end_time - start_time
    avg_time_packed_dequant = total_time / run_time

    # Calculate bandwidth for packed dequantization
    input_size = batch * seq_len * heads * 132 * 1  # packed int8 = 132 bytes
    output_size = batch * seq_len * heads * 128 * 2  # bf16 = 2 bytes
    total_bytes = input_size + output_size
    bandwidth_gb_s = (total_bytes / 1e9) / avg_time_packed_dequant

    print(f"Dequantize: {avg_time_packed_dequant*1000:.3f} ms, {bandwidth_gb_s:.1f} GB/s")

    return True


def main():
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        return

    try:
        # Run all tests
        all_passed = True
        all_passed &= test_hnd_layout()
        all_passed &= test_hnd_layout_packed()
        all_passed &= test_nhd_layout()
        all_passed &= test_nhd_layout_packed()

        print("\n" + "=" * 80)
        if all_passed:
            print("✓ ALL TESTS PASSED!")
        else:
            print("✗ SOME TESTS FAILED!")
        print("=" * 80)

        # Run performance profiling
        test_nhd_profile(1, 75600, 40)
        test_nhd_profile_packed(1, 75600, 40)
        test_nhd_profile(1, 3510, 40)
        test_nhd_profile_packed(1, 3510, 40)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
        return False

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
