/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace marlin_nvfp4
{

// Dequantize FP4 E2M1 activations to BF16.
// act_fp4:  [M, K/2] packed FP4 (uint8)
// act_sf:   activation block scales (FP8 E4M3, swizzled, uint8)
// alpha:    global scale (float32, device pointer)
// act_bf16: [M, K] output BF16 buffer (pre-allocated by caller)
void dequantFp4Activations(
    void const* act_fp4, void const* act_sf, float const* alpha, void* act_bf16, int m, int k, cudaStream_t stream);

// Run Marlin W4A16 GEMM: BF16 activations + FP4 weights.
// act_bf16:  [M, K] BF16 activations (pre-dequantized by caller)
// weight:    [N, K/2] FP4 weights (row-major packed)
// output:    [M, N] output buffer
// weight_sf: weight block scales (FP8 E4M3, swizzled)
// global_scale_bf16: global scale as BF16 (device pointer, uint16_t)
// workspace: [sms] int32 lock buffer (pre-allocated, zeroed by caller)
void marlinNvfp4Gemm(void const* act_bf16, void const* weight, void* output, void* C_tmp, void const* weight_sf,
    void const* global_scale_bf16, int m, int n, int k, int* workspace, int num_groups, int group_size,
    bool use_fp32_reduce, cudaStream_t stream);

} // namespace marlin_nvfp4
