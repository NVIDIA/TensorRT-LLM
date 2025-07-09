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

#include <cstdint>
#include <cuda.h>
#include <vector>

#include "trtllmGen_bmm_export/trtllm/gen/DtypeDecl.h"

namespace tensorrt_llm
{
namespace kernels
{

struct TrtllmGenBatchedGemmRunnerOptions
{
    batchedGemm::trtllm::gen::Dtype eltType;
    batchedGemm::trtllm::gen::Dtype outputType;
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
        std::vector<int32_t> const& batchedTokens, int32_t numTokens, int32_t numBatches, int32_t maxNumCtasInBatchDim,
        int32_t configIndex) const;

    // Generic GEMM interface
    void run(int32_t m, int32_t n, int32_t k, std::vector<int32_t> const& batchedTokens, int32_t numTokens,
        int32_t numBatches, int32_t maxNumCtasInBatchDim, void const* a, void const* sfA, void const* b,
        void const* sfB, void const* perTokensSfA, void const* perTokensSfB, float const* scaleC,
        float const* scaleGateC, void* c, void* outSfC, int32_t const* routeMap, int32_t const* totalNumPaddedTokens,
        int32_t const* ctaIdxXyToBatchIdx, int32_t const* ctaIdxXyToMnLimit, int32_t const* numNonExitingCtas,
        void* workspace, CUstream stream, int device, int32_t configIndex);

    // NVFP4 per-block scaling GEMM
    void run(int32_t m, int32_t n, int32_t k, std::vector<int32_t> const& batchedTokens, void const* a, void const* sfA,
        void const* b, void const* sfB, void* c, void* outSfC, void* workspace, CUstream stream, int device,
        int32_t configIndex);

    // FP8 per-tensor scaling GEMM
    void run(int32_t m, int32_t n, int32_t k, std::vector<int32_t> const& batchedTokens, void const* a, void const* b,
        float const* scaleC, float const* scaleGateC, void* c, void* workspace, CUstream stream, int device,
        int32_t configIndex);

    // Get the list of configs that passed the validation based on the constructor options
    [[nodiscard]] std::vector<int64_t> getPassingConfigIndices() const
    {
        return mPassingConfigIndices;
    }

    // Get the list of config indices that are valid for the given problem shape
    [[nodiscard]] std::vector<int64_t> getValidConfigIndices(int32_t m, int32_t n, int32_t k,
        std::vector<int32_t> const& batchedTokens, int32_t numTokens, int32_t numBatches,
        int32_t maxNumCtasInBatchDim) const;

    // Get a default config index that is valid for the given problem shape
    // This will be used as the fallback config if using auto-tuning
    [[nodiscard]] int64_t getDefaultValidConfigIndex(int32_t m, int32_t n, int32_t k,
        std::vector<int32_t> const& batchedTokens, int32_t numTokens, int32_t numBatches,
        int32_t maxNumCtasInBatchDim) const;

    [[nodiscard]] bool isValidConfigIndex(int32_t configIndex, int32_t m, int32_t n, int32_t k,
        std::vector<int32_t> const& batchedTokens, int32_t numTokens, int32_t numBatches,
        int32_t maxNumCtasInBatchDim) const;

private:
    void selectGemmConfig(int32_t m, int32_t n, int32_t k, std::vector<int32_t> const& batchedTokens, int32_t numTokens,
        int32_t numBatches, int32_t maxNumCtasInBatchDim);

private:
    TrtllmGenBatchedGemmRunnerOptions mOptions;
    std::vector<int64_t> mPassingConfigIndices;
};
} // namespace kernels
} // namespace tensorrt_llm
