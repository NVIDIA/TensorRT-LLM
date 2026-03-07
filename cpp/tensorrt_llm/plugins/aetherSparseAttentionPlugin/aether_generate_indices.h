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

// Computes dot product of query and block centroids, and extracts indices of blocks
// where dot_product >= cos_threshold.
// 
// query: [B, H, D]
// block_centroids: [B, H, num_blocks, D]
// active_indices: [B, H, max_active_blocks]  (Output)
// active_counts: [B, H] (Output)
// 
void invokeAetherGenerateIndices(
    void const* query,
    void const* block_centroids,
    int* active_indices,
    int* active_counts,
    float cos_threshold,
    int batch_size,
    int num_heads,
    int head_dim,
    int num_blocks,
    int max_active_blocks,
    cudaStream_t stream);

} // namespace aether
} // namespace tensorrt_llm::plugins
