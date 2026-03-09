/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

namespace tensorrt_llm::plugins
{
namespace aether
{

// Core Forward Kernel for Aether Block-Sparse Flash Attention
// 
// Computes attention using a pre-computed array of active indices.
void invokeAetherSparseFlashFwd(
    void const* query,          // [B, H, q_len, D]
    void const* keys,           // [B, H_kv, max_seq_len, D]
    void const* values,         // [B, H_kv, max_seq_len, D]
    int const* active_indices,  // [B, H, max_active_blocks]
    int const* active_counts,   // [B, H]
    void* output,               // [B, H, q_len, D]
    float scale,
    int batch_size,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    int block_size,
    int q_len,
    int max_seq_len,
    int max_active_blocks,
    cudaStream_t stream);

} // namespace aether
} // namespace tensorrt_llm::plugins
