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
};

enum class QuantType : int
{
    kNone = 0,
    kFP8 = 1,
    kFP4 = 2,
};

template <AllReduceFusionPattern Pattern>
struct FusionPatternTraits;

#define DEFINE_FUSION_PATTERN_TRAITS(pattern, hasResidual, hasRMSNorm, quantType)                                      \
    template <>                                                                                                        \
    struct FusionPatternTraits<pattern>                                                                                \
    {                                                                                                                  \
        static constexpr bool kHasResidual = hasResidual;                                                              \
        static constexpr bool kHasRMSNorm = hasRMSNorm;                                                                \
        static constexpr QuantType kQuantType = quantType;                                                             \
    };

DEFINE_FUSION_PATTERN_TRAITS(AllReduceFusionPattern::kAllReduce, false, false, QuantType::kNone);
DEFINE_FUSION_PATTERN_TRAITS(AllReduceFusionPattern::kARResidualRMSNorm, true, true, QuantType::kNone);
DEFINE_FUSION_PATTERN_TRAITS(AllReduceFusionPattern::kARResidualRMSNormFP8Quant, true, true, QuantType::kFP8);
DEFINE_FUSION_PATTERN_TRAITS(AllReduceFusionPattern::kARResidualRMSNormFP4Quant, true, true, QuantType::kFP4);
#undef DEFINE_FUSION_PATTERN_TRAITS

template <AllReduceFusionPattern Pattern>
constexpr bool HasResidual = FusionPatternTraits<Pattern>::kHasResidual;
template <AllReduceFusionPattern Pattern>
constexpr bool HasRMSNorm = FusionPatternTraits<Pattern>::kHasRMSNorm;
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
    void* residual_out;
    void* norm_out;
    void* quant_out;
    void* scale_out;
    void* rms_gamma;
    float rms_eps;
    float* scale_factor;
    cudaStream_t stream;
    AllReduceFusionPattern pattern;
};

void allreduce_fusion_op(AllReduceFusionParams const& params);

class Workspace
{
public:
    Workspace(int rank, int tp_size, int max_token_num, int hidden_dim,
        std::shared_ptr<tensorrt_llm::runtime::CudaStream> stream_ptr);
    ~Workspace();
    void** get_workspace();

private:
    tensorrt_llm::runtime::WorldConfig m_world_config;
    std::shared_ptr<tensorrt_llm::runtime::BufferManager> m_buffer_mgr;
    std::vector<tensorrt_llm::runtime::IpcMemory> m_ipc_mem_handles;
    void* m_workspace;
    std::shared_ptr<tensorrt_llm::runtime::CudaStream> m_cuda_stream;
    void* m_flag_d_ptr;
};

void lamport_initialize(void* ptr, int bytes, cudaStream_t stream);
} // namespace tensorrt_llm::kernels::ar_fusion
