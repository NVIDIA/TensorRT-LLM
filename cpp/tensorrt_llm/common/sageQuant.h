/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION &
 * AFFILIATES. All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "tensorrt_llm/common/cudaBf16Wrapper.h"
#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"
#include <cuda.h>
#include <cuda_runtime.h>
#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif

namespace tensorrt_llm::common
{

// How Q or K tokens are grouped into scales. See sagePartition.h.
enum class SageScalePartition
{
    // tokenBlockSize consecutive tokens share a scale, one group per tile.
    Contiguous = 0,
    // 2 rows 8 apart share a scale; 32 groups per 64-row tile.
    HopperQ,
    // 16 strided keys share a scale; 16 groups per 256-key tile.
    HopperK,
};

struct SageQuantParams
{
    // Required arguments for SageQuantQk (Q or K):
    int sumSeqLensQk{};
    int batchSize{};
    int numHeads{};
    int headDim{};
    // Only read when partition == Contiguous; the Hopper partitions fix their own group size.
    int tokenBlockSize{};
    SageScalePartition partition{SageScalePartition::Contiguous};
    bool kSmooth{false};
    int const* ptrCuSeqLensQk{nullptr};
    void const* ptrQk{nullptr};
    void* ptrQkQuant{nullptr};
    kernels::Data_type inputType{kernels::DATA_TYPE_FP16};
    kernels::Data_type quantType{kernels::DATA_TYPE_E4M3};
    float* ptrQkScale{nullptr};
    // Optional source and scratch mean used to perform K-smoothing.
    // (See below) collected at vStage==0, applied at vStage==1.
    void const* ptrKForMean{nullptr};
    float* ptrKMean{nullptr};
    // Optional arguments for SageQuantV:
    // vStage: 0: disabled, 1: collect scales, 2: quantize
    int vStage{};
    int sumSeqLensV{};
    int numHeadsV{};
    void const* ptrV{nullptr};
    void* ptrVQuant{nullptr};
    float* ptrVScale{nullptr};
    // Hardware info. Required.
    int smCount{};
    cudaStream_t stream{};
};

void invokeSageQuant(SageQuantParams const& params);

// The scale grouping a consumer kernel expects. SM90 scales the tokens a single thread owns; the
// other architectures scale contiguous token blocks.
inline SageScalePartition getSageQPartition(bool perThread)
{
    return perThread ? SageScalePartition::HopperQ : SageScalePartition::Contiguous;
}

inline SageScalePartition getSageKPartition(bool perThread)
{
    return perThread ? SageScalePartition::HopperK : SageScalePartition::Contiguous;
}

// Scale-buffer geometry for a partition, so that callers can size the scale buffers and fill in
// the max_nblock the consumer kernel reads. Returns 0 when the tensor is not sage-quantized
// (tokenBlockSize 0), when the shape is empty, or when the (partition, tokenBlockSize) pair has no
// kernel instantiation, so it is safe to call from noexcept sizing paths.

// Scales per head, i.e. the head stride of the scale buffer. Grows with the batch size as well as
// the token count, since every sequence reserves a spare tile.
int getSageScaleHeadStride(SageScalePartition partition, int tokenBlockSize, int sumSeqLens, int batchSize);

} // namespace tensorrt_llm::common
