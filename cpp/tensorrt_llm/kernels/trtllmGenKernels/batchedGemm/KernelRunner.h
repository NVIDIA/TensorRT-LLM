/*
 * Copyright (c) 2020-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda.h>
#include <optional>

#include "tensorrt_llm/kernels/trtllmGenKernels/common/Dtype.h"
#include "trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h"

namespace tensorrt_llm
{
namespace kernels
{

struct TrtllmGenBatchedGemmRunnerOptions
{
    trtllm::gen::Dtype eltType;
    trtllm::gen::Dtype outputType;
    bool deepSeekFp8{false};
    bool fusedAct{false};
    bool routeAct{false};
    bool staticBatch{false};
    bool transposeMmaOutput{false};
    int32_t tileSize{8};
    int32_t epilogueTileM{128};
};

class TrtllmGenBatchedGemmRunner
{
public:
    explicit TrtllmGenBatchedGemmRunner(TrtllmGenBatchedGemmRunnerOptions const& options);

    [[nodiscard]] size_t getWorkspaceSizeInBytes(int32_t m, int32_t n, int32_t k,
        std::vector<int32_t> const& batchedTokens, int32_t numTokens, int32_t numBatches, int32_t maxNumCtasInBatchDim);

    void run(int32_t m, int32_t n, int32_t k, std::vector<int32_t> const& batchedTokens, int32_t numTokens,
        int32_t numBatches, int32_t maxNumCtasInBatchDim, void const* a, void const* sfA, void const* b,
        void const* sfB, void const* perTokensSfA, void const* perTokensSfB, float const* scaleC,
        float const* scaleGateC, void* c, void* outSfC, int32_t const* routeMap, int32_t const* totalNumPaddedTokens,
        int32_t const* ctaIdxXyToBatchIdx, int32_t const* ctaIdxXyToMnLimit, int32_t const* numNonExitingCtas,
        void* workspace, CUstream stream, int device);

    void run(int32_t m, int32_t n, int32_t k, std::vector<int32_t> const& batchedTokens, void const* a, void const* sfA,
        void const* b, void const* sfB, void* c, void* outSfC, void* workspace, CUstream stream, int device);

    void run(int32_t m, int32_t n, int32_t k, std::vector<int32_t> const& batchedTokens, void const* a, void const* b,
        float const* scaleC, float const* scaleGateC, void* c, void* workspace, CUstream stream, int device);

private:
    void selectGemmConfig(int32_t m, int32_t n, int32_t k, std::vector<int32_t> const& batchedTokens, int32_t numTokens,
        int32_t numBatches, int32_t maxNumCtasInBatchDim);

private:
    TrtllmGenBatchedGemmRunnerOptions mOptions;
    std::optional<int> mSelectedConfigIndex;
    std::vector<int32_t> mPassingConfigIndices;
};
} // namespace kernels
} // namespace tensorrt_llm
