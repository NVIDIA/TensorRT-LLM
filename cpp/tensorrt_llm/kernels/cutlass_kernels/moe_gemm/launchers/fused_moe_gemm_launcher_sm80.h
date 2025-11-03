/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: NVIDIA TensorRT Source Code License Agreement
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

namespace tensorrt_llm::kernels::cutlass_kernels
{
template <typename ElementType_, typename CutlassWeightType_, int MaxTileM_, int TileN_, int TileK_, int Stages_,
    typename EpilogueTag>
void sm80_generic_fused_moe_gemm_kernelLauncher(ElementType_ const* A, CutlassWeightType_ const* B,
    ElementType_ const* biases, bool bias_is_broadcast, ElementType_* C, int64_t const* total_tokens_including_expert,
    int64_t num_rows, int64_t gemm_n, int64_t gemm_k, int num_experts, int multi_processor_count, cudaStream_t stream,
    int* kernel_occupancy);
}
