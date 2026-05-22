# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import triton
import triton.language as tl


@triton.jit
def rope_fwd_interleaved_kernel(
    q_ptr,  # [B, S, H_Q, D] input Q (interleaved layout)
    k_ptr,  # [B, S, H_K, D] input K (interleaved layout)
    cos_cache_ptr,  # [max_seq_len, D] cos cache
    sin_cache_ptr,  # [max_seq_len, D] sin cache
    position_ids_ptr,  # [B, S] position IDs
    q_out_ptr,  # [B, S, H_Q, D] output Q
    k_out_ptr,  # [B, S, H_K, D] output K
    B,  # batch size
    S,  # sequence length
    H_Q,  # number of Q heads
    H_K,  # number of K heads (typically 1 for MQA)
    D: tl.constexpr,  # head dimension
    stride_qb,  # Q batch stride
    stride_qs,  # Q seq stride
    stride_qh,  # Q head stride
    stride_qd,  # Q dim stride
    stride_kb,  # K batch stride
    stride_ks,  # K seq stride
    stride_kh,  # K head stride
    stride_kd,  # K dim stride
    stride_pos_b,  # position_ids batch stride
    stride_pos_s,  # position_ids seq stride
    stride_cache_s,  # cache seq stride
    stride_cache_d,  # cache dim stride
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_S: tl.constexpr,
):
    """
    Fused RoPE kernel for DeepSeek-style interleaved Q/K inputs.

    Input layout (interleaved): [q0_r, q0_i, q1_r, q1_i, ...]
    After de-interleave: [q0_r, q1_r, ..., q0_i, q1_i, ...]

    This kernel does:
    1. Position ID lookup for cos/sin
    2. Reads even and odd indices directly from the interleaved input using strided loads
    3. RoPE application directly on the separated a/b values: y_first = a*cos - b*sin, y_second = b*cos + a*sin
    4. Writes the output in contiguous "half-split" layout — first half and second half stored separately

    Grid: (B, cdiv(H_Q, BLOCK_SIZE_H), cdiv(S, BLOCK_SIZE_S))
    """
    D2: tl.constexpr = D // 2
    D2_PADDED: tl.constexpr = triton.next_power_of_2(D2)

    # Program IDs
    batch_id = tl.program_id(0)
    head_block_id = tl.program_id(1)
    seq_block_id = tl.program_id(2)

    # Head offsets and mask
    head_offsets = head_block_id * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    head_mask = head_offsets < H_Q

    # Sequence offsets and mask
    seq_offsets = seq_block_id * BLOCK_SIZE_S + tl.arange(0, BLOCK_SIZE_S)
    seq_mask = seq_offsets < S

    # Dimension offsets for half the head dim (we read pairs)
    dim_offsets = tl.arange(0, D2_PADDED)
    dim_mask = dim_offsets < D2

    # ========== LOAD POSITION IDS ==========
    # position_ids: [B, S]
    pos_ptr = position_ids_ptr + batch_id * stride_pos_b + seq_offsets * stride_pos_s
    pos_ids = tl.load(pos_ptr, mask=seq_mask, other=0)  # [BLOCK_SIZE_S]

    # ========== LOAD COS/SIN FROM CACHE ==========
    # cos_cache, sin_cache: [max_seq_len, D]
    # For each position, load the corresponding cos/sin values
    # We need cos/sin for both halves of head_dim
    cache_offsets = (
        pos_ids[:, None] * stride_cache_s + dim_offsets[None, :] * stride_cache_d
    )  # [BLOCK_SIZE_S, D2_PADDED]

    cache_mask = seq_mask[:, None] & dim_mask[None, :]  # [BLOCK_SIZE_S, D2_PADDED]

    cos_first = tl.load(cos_cache_ptr + cache_offsets, mask=cache_mask)  # [BLOCK_SIZE_S, D2_PADDED]
    sin_first = tl.load(sin_cache_ptr + cache_offsets, mask=cache_mask)  # [BLOCK_SIZE_S, D2_PADDED]

    # Second half of cos/sin (offset by D2)
    cache_offsets_second = (
        pos_ids[:, None] * stride_cache_s + (dim_offsets[None, :] + D2) * stride_cache_d
    )
    cos_second = tl.load(cos_cache_ptr + cache_offsets_second, mask=cache_mask)
    sin_second = tl.load(sin_cache_ptr + cache_offsets_second, mask=cache_mask)

    # ========== PROCESS Q ==========
    # Q layout: [B, S, H, D] with interleaved D
    # Read even indices (a) and odd indices (b) for de-interleaving
    # Input: [q0_r, q0_i, q1_r, q1_i, ...] -> a=[q0_r, q1_r, ...], b=[q0_i, q1_i, ...]

    q_base = batch_id * stride_qb

    # Compute offsets for reading interleaved data
    # even_offsets: positions 0, 2, 4, ... (stride 2)
    # odd_offsets: positions 1, 3, 5, ... (stride 2)
    q_offsets_base = (
        seq_offsets[:, None, None] * stride_qs
        + head_offsets[None, :, None] * stride_qh
        + dim_offsets[None, None, :] * 2 * stride_qd  # stride 2 for interleaved
    )  # [BLOCK_SIZE_S, BLOCK_SIZE_H, D2_PADDED]

    even_offsets = q_base + q_offsets_base
    odd_offsets = q_base + q_offsets_base + stride_qd

    # Combined mask
    load_mask = seq_mask[:, None, None] & head_mask[None, :, None] & dim_mask[None, None, :]

    # Load Q values (even = a, odd = b)
    q_a = tl.load(q_ptr + even_offsets, mask=load_mask)  # [BLOCK_SIZE_S, BLOCK_SIZE_H, D2_PADDED]
    q_b = tl.load(q_ptr + odd_offsets, mask=load_mask)  # [BLOCK_SIZE_S, BLOCK_SIZE_H, D2_PADDED]

    # Broadcast cos/sin for heads: [BLOCK_SIZE_S, D2_PADDED] -> [BLOCK_SIZE_S, 1, D2_PADDED]
    cos_first_bc = cos_first[:, None, :]
    sin_first_bc = sin_first[:, None, :]
    cos_second_bc = cos_second[:, None, :]
    sin_second_bc = sin_second[:, None, :]

    # Apply RoPE formula
    # y_first_half = a * cos - b * sin
    # y_second_half = b * cos + a * sin
    q_y1 = q_a * cos_first_bc - q_b * sin_first_bc
    q_y2 = q_b * cos_second_bc + q_a * sin_second_bc

    # Store Q output (CONTIGUOUS layout)
    # Output layout: [B, S, H_Q, D] with first half = y1, second half = y2
    # Compute contiguous strides: stride_b=S*H_Q*D, stride_s=H_Q*D, stride_h=D, stride_d=1
    q_out_stride_b = S * H_Q * D
    q_out_stride_s = H_Q * D
    q_out_stride_h = D
    q_out_offsets_first = (
        batch_id * q_out_stride_b
        + seq_offsets[:, None, None] * q_out_stride_s
        + head_offsets[None, :, None] * q_out_stride_h
        + dim_offsets[None, None, :]  # stride_d = 1
    )
    q_out_offsets_second = q_out_offsets_first + D2  # D2 * 1

    tl.store(q_out_ptr + q_out_offsets_first, q_y1, mask=load_mask)
    tl.store(q_out_ptr + q_out_offsets_second, q_y2, mask=load_mask)

    # ========== PROCESS K ==========
    # K typically has H_K=1 for MQA, but we handle general case
    # Use head_offsets < H_K for K's mask
    head_mask_k = head_offsets < H_K

    k_base = batch_id * stride_kb

    k_offsets_base = (
        seq_offsets[:, None, None] * stride_ks
        + head_offsets[None, :, None] * stride_kh
        + dim_offsets[None, None, :] * 2 * stride_kd
    )

    k_even_offsets = k_base + k_offsets_base
    k_odd_offsets = k_base + k_offsets_base + stride_kd

    load_mask_k = seq_mask[:, None, None] & head_mask_k[None, :, None] & dim_mask[None, None, :]

    k_a = tl.load(k_ptr + k_even_offsets, mask=load_mask_k)
    k_b = tl.load(k_ptr + k_odd_offsets, mask=load_mask_k)

    k_y1 = k_a * cos_first_bc - k_b * sin_first_bc
    k_y2 = k_b * cos_second_bc + k_a * sin_second_bc

    # Store K output (CONTIGUOUS layout)
    # Output layout: [B, S, H_K, D] with first half = y1, second half = y2
    # Compute contiguous strides: stride_b=S*H_K*D, stride_s=H_K*D, stride_h=D, stride_d=1
    k_out_stride_b = S * H_K * D
    k_out_stride_s = H_K * D
    k_out_stride_h = D
    k_out_offsets_first = (
        batch_id * k_out_stride_b
        + seq_offsets[:, None, None] * k_out_stride_s
        + head_offsets[None, :, None] * k_out_stride_h
        + dim_offsets[None, None, :]  # stride_d = 1
    )
    k_out_offsets_second = k_out_offsets_first + D2  # D2 * 1

    tl.store(k_out_ptr + k_out_offsets_first, k_y1, mask=load_mask_k)
    tl.store(k_out_ptr + k_out_offsets_second, k_y2, mask=load_mask_k)


@triton.jit
def rope_fwd_kernel(
    x_ptr,
    input_pos_ptr,
    f_ptr,
    output_ptr,
    N,
    L,
    H,
    D: tl.constexpr,
    stride_n,
    stride_l,
    stride_h,
    stride_d,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
):
    """Grids(N, H // BLOCK_SIZE_H, BLOCK_SIZE_L).

    Each block gets 2 blocks of (1, H, BLOCK_SIZE_L, D) input data.
    """
    D2: tl.constexpr = D // 2
    D2_PADDED: tl.constexpr = triton.next_power_of_2(D2)
    batch = tl.program_id(0)
    x_ptr += batch * stride_n
    output_ptr += batch * stride_n

    # frequencies tensor is not sliced.
    # layout: [1,max_seq_len,D//2,2]
    input_offset = tl.load(input_pos_ptr + batch) * D
    head_offsets = tl.program_id(1) * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    head_mask = head_offsets < H

    # input is interleaved as [2,D//2].
    col_offsets1 = tl.arange(0, D2_PADDED)
    col_mask1 = col_offsets1 < D2
    col_offsets2 = col_offsets1 + D2
    col_mask2 = col_offsets2 < D

    row_start = tl.program_id(2) * BLOCK_SIZE_L
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_L)
    row_mask = row_offsets < L

    offsets1 = (
        head_offsets[:, None, None] * stride_h
        + row_offsets[None, :, None] * stride_l
        + col_offsets1[None, None, :] * stride_d
    )
    offsets2 = (
        head_offsets[:, None, None] * stride_h
        + row_offsets[None, :, None] * stride_l
        + col_offsets2[None, None, :] * stride_d
    )

    nohead_mask1 = row_mask[None, :, None] * col_mask1[None, None, :]
    nohead_mask2 = row_mask[None, :, None] * col_mask2[None, None, :]

    mask1 = head_mask[:, None, None] * nohead_mask1
    mask2 = head_mask[:, None, None] * nohead_mask2

    a = tl.load(x_ptr + offsets1, mask=mask1).to(tl.float32)
    b = tl.load(x_ptr + offsets2, mask=mask2).to(tl.float32)
    # -----------------------------------
    # torch version sin/cos
    # cos and sin values are interleaved in frequencies tensor.
    col_offsets = tl.arange(0, D2_PADDED)
    offsets = row_offsets[None, :, None] * D2 + col_offsets[None, None, :]
    cos_ref = tl.load(f_ptr + input_offset + offsets * 2, mask=nohead_mask1).to(dtype=tl.float32)
    sin_ref = tl.load(f_ptr + input_offset + offsets * 2 + 1, mask=nohead_mask2).to(
        dtype=tl.float32
    )

    y1 = cos_ref * a - sin_ref * b
    y2 = sin_ref * a + cos_ref * b

    # -----------------------------------
    # triton version sin/cos
    # m = row_offsets + 1.
    # theta = tl.exp(-2. * (col_start + tl.arange(0, BLOCK_SIZE_D)) / D * LOG_BASE)
    # mtheta = m[None, :, None] * theta[None, None, :]
    # cos = tl.cos(mtheta)
    # sin = tl.sin(mtheta)

    # y1 = cos * a - sin * b
    # y2 = sin * a + cos * b
    # -----------------------------------

    tl.store(output_ptr + offsets1, y1, mask=mask1)
    tl.store(output_ptr + offsets2, y2, mask=mask2)


@triton.jit
def rope_fwd_flattened_kernel(
    x_ptr,  # [B*S, N, D]
    seq_lens_ptr,  # [B]
    seq_start_indices_ptr,  # [B]
    input_pos_ptr,  # [B]
    f_ptr,
    output_ptr,
    H: tl.constexpr,  # number of heads
    D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_L: tl.constexpr,
):
    """Rope that works with flattened Q/K."""
    D2: tl.constexpr = D // 2
    D2_PADDED: tl.constexpr = triton.next_power_of_2(D2)
    batch = tl.program_id(0)
    seq_len = tl.load(seq_lens_ptr + batch)
    seq_start_index = tl.load(seq_start_indices_ptr + batch)

    x_ptr += seq_start_index * H * D
    output_ptr += seq_start_index * H * D

    # frequencies tensor is not sliced.
    # layout: [1,max_seq_len,D//2,2]
    input_offset = tl.load(input_pos_ptr + batch) * D
    head_offsets = tl.program_id(1) * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    head_mask = head_offsets < H

    # input is interleaved as [2,D//2].
    col_offsets1 = tl.arange(0, D2_PADDED)
    col_mask1 = col_offsets1 < D2
    col_offsets2 = col_offsets1 + D2
    col_mask2 = col_offsets2 < D

    row_start = tl.program_id(2) * BLOCK_SIZE_L
    row_offsets = row_start + tl.arange(0, BLOCK_SIZE_L)
    row_mask = row_offsets < seq_len

    offsets1 = (
        head_offsets[:, None, None] * D
        + row_offsets[None, :, None] * D * H
        + col_offsets1[None, None, :]
    )
    offsets2 = (
        head_offsets[:, None, None] * D
        + row_offsets[None, :, None] * D * H
        + col_offsets2[None, None, :]
    )

    nohead_mask1 = row_mask[None, :, None] * col_mask1[None, None, :]
    nohead_mask2 = row_mask[None, :, None] * col_mask2[None, None, :]

    mask1 = head_mask[:, None, None] * nohead_mask1
    mask2 = head_mask[:, None, None] * nohead_mask2

    a = tl.load(x_ptr + offsets1, mask=mask1).to(tl.float32)
    b = tl.load(x_ptr + offsets2, mask=mask2).to(tl.float32)
    # -----------------------------------
    # torch version sin/cos
    # cos and sin values are interleaved in frequencies tensor.
    col_offsets = tl.arange(0, D2_PADDED)
    offsets = row_offsets[None, :, None] * D2 + col_offsets[None, None, :]
    cos_ref = tl.load(f_ptr + input_offset + offsets * 2, mask=nohead_mask1).to(dtype=tl.float32)
    sin_ref = tl.load(f_ptr + input_offset + offsets * 2 + 1, mask=nohead_mask2).to(
        dtype=tl.float32
    )

    y1 = cos_ref * a - sin_ref * b
    y2 = sin_ref * a + cos_ref * b

    # -----------------------------------
    # triton version sin/cos
    # m = row_offsets + 1.
    # theta = tl.exp(-2. * (col_start + tl.arange(0, BLOCK_SIZE_D)) / D * LOG_BASE)
    # mtheta = m[None, :, None] * theta[None, None, :]
    # cos = tl.cos(mtheta)
    # sin = tl.sin(mtheta)

    # y1 = cos * a - sin * b
    # y2 = sin * a + cos * b
    # -----------------------------------

    tl.store(output_ptr + offsets1, y1, mask=mask1)
    tl.store(output_ptr + offsets2, y2, mask=mask2)
