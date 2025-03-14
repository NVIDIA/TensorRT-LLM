/*
 * Copyright (c) 2022-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <sstream>

namespace tensorrt_llm
{
namespace kernels
{
template <int HeadSize, int paddedHeadSize, int BlockSizeQ, int BlockSizeK, int BlockSizeV, typename T, typename TQuant,
    typename TSmoothK>
void sage_quant(
    // host var
    unsigned int batch_size, unsigned int head_num, unsigned int max_seq_len, bool smooth_k, bool is_padded,
    // device input
    void const* q, void const* k, void const* v, int const stride_q, int const stride_k, int const stride_v,
    int const* cu_seqlens_q, int const* cu_seqlens_kv,
    // sizeof(workspace) = batch_size * head_num * head_size * sizeof(TSmoothK)
    void* workspace,
    // device output
    void* quant_q, void* quant_k, void* quant_v, int const stride_quant_q, int const stride_quant_k,
    int const stride_quant_v, float* scales_q, float* scales_k, float* scales_v, cudaStream_t stream);

template <int PaddedHeadSize, int HeadSize, typename T>
void unpadding(
    // host var
    unsigned int batch_size, unsigned int head_num, unsigned int max_seq_len,
    // device input
    void const* padded_output, int const stride_output, int const stride_padded_output, int const* cu_seqlens,
    // device output
    void* output, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
