/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#include <NvInferRuntime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/quantization.h"
#include "tensorrt_llm/runtime/ipcUtils.h"

namespace tensorrt_llm::kernels::ar_fusion
{
template <typename DType>
struct ElemsPerAccess;

template <>
struct ElemsPerAccess<half>
{
    static constexpr int value = 8;
};

template <>
struct ElemsPerAccess<nv_bfloat16>
{
    static constexpr int value = 8;
};

template <>
struct ElemsPerAccess<float>
{
    static constexpr int value = 4;
};

template <typename DType>
static constexpr int kElemsPerAccess = ElemsPerAccess<DType>::value;
static constexpr int kOneShotMaxToken = 128;
static constexpr int kBarrierFlagCount = 256;

enum class AllReduceFusionPattern : int
{
    kAllReduce = 0,
    kARResidualRMSNorm = 1,
    kARResidualRMSNormFP8Quant = 2,
    kARResidualRMSNormFP4Quant = 3,
    // The difference between these two and the standard version is that the NormOut version outputs the result of the
    // norm.
    kARResidualRMSNormOutFP8Quant = 4,
    kARResidualRMSNormOutFP4Quant = 5
};

enum class QuantType : int
{
    kNone = 0,
    kFP8 = 1,
    kFP4 = 2,
};

template <AllReduceFusionPattern Pattern>
struct FusionPatternTraits;

#define DEFINE_FUSION_PATTERN_TRAITS(                                                                                  \
    pattern, hasAllReduceOut, hasResidual, hasResidualOut, hasRMSNorm, hasNormOut, quantType)                          \
    template <>                                                                                                        \
    struct FusionPatternTraits<pattern>                                                                                \
    {                                                                                                                  \
        static constexpr bool kHasAllReduceOut = hasAllReduceOut;                                                      \
        static constexpr bool kHasResidual = hasResidual;                                                              \
        static constexpr bool kHasResidualOut = hasResidualOut;                                                        \
        static constexpr bool kHasRMSNorm = hasRMSNorm;                                                                \
        static constexpr bool kHasNormOut = hasNormOut;                                                                \
        static constexpr QuantType kQuantType = quantType;                                                             \
    };

DEFINE_FUSION_PATTERN_TRAITS(AllReduceFusionPattern::kAllReduce, true, false, false, false, false, QuantType::kNone);
DEFINE_FUSION_PATTERN_TRAITS(
    AllReduceFusionPattern::kARResidualRMSNorm, false, true, true, true, true, QuantType::kNone);
DEFINE_FUSION_PATTERN_TRAITS(
    AllReduceFusionPattern::kARResidualRMSNormFP8Quant, false, true, true, true, false, QuantType::kFP8);
DEFINE_FUSION_PATTERN_TRAITS(
    AllReduceFusionPattern::kARResidualRMSNormFP4Quant, false, true, true, true, false, QuantType::kFP4);
DEFINE_FUSION_PATTERN_TRAITS(
    AllReduceFusionPattern::kARResidualRMSNormOutFP8Quant, false, true, true, true, true, QuantType::kFP8);
DEFINE_FUSION_PATTERN_TRAITS(
    AllReduceFusionPattern::kARResidualRMSNormOutFP4Quant, false, true, true, true, true, QuantType::kFP4);
#undef DEFINE_FUSION_PATTERN_TRAITS

template <AllReduceFusionPattern Pattern>
constexpr bool HasResidual = FusionPatternTraits<Pattern>::kHasResidual;
template <AllReduceFusionPattern Pattern>
constexpr bool HasRMSNorm = FusionPatternTraits<Pattern>::kHasRMSNorm;
template <AllReduceFusionPattern Pattern>
constexpr bool HasAllReduceOut = FusionPatternTraits<Pattern>::kHasAllReduceOut;
template <AllReduceFusionPattern Pattern>
constexpr bool HasResidualOut = FusionPatternTraits<Pattern>::kHasResidualOut;
template <AllReduceFusionPattern Pattern>
constexpr bool HasNormOut = FusionPatternTraits<Pattern>::kHasNormOut;
template <AllReduceFusionPattern Pattern>
constexpr QuantType GetQuantType = FusionPatternTraits<Pattern>::kQuantType;

struct AllReduceFusionParams
{
    int nranks;
    int rank;
    nvinfer1::DataType dtype;
    int size;
    int hidden_dim;
    void** workspace;
    void* allreduce_in;
    void* residual_in;
    void* allreduce_out;
    void* residual_out;
    void* norm_out;
    void* quant_out;
    void* scale_out;
    void* rms_gamma;
    float rms_eps;
    float* scale_factor;
    bool use_oneshot;
    QuantizationSFLayout layout = QuantizationSFLayout::SWIZZLED;
    cudaStream_t stream;
    AllReduceFusionPattern pattern;
    bool trigger_completion_at_end = true;
};

void allreduce_fusion_op(AllReduceFusionParams const& params);
} // namespace tensorrt_llm::kernels::ar_fusion
