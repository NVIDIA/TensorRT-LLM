/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/common/cudaDriverWrapper.h"
#include "tensorrt_llm/kernels/multiHeadAttentionCommon.h"
#include "tensorrt_llm/kernels/trtllmGenKernels/batchedGemm/kernelList.h"

namespace tensorrt_llm::kernels
{
class TrtllmGenBatchedGemmRunner
{
public:
    explicit TrtllmGenBatchedGemmRunner(
        Data_type outputType, int64_t gemmBatchSize, int64_t tileSize, bool useDeepSeekFp8, bool batchModeM);

    void run(int32_t m, int32_t n, int32_t k, void* a, void* b, void* c, float const* cScale, float* dDqSfsA,
        float* dDqSfsB, float* dDqSfsC, std::vector<int32_t> const& batchedM, std::vector<int32_t> const& batchedN,
        CUstream stream);

private:
    Data_type mOutputType;
    int32_t mGemmBatchSize;
    bool mUseDeepSeekFp8;
    BatchMode mBatchMode;
    bool mBatchModeM;
    int32_t mTileSize;
    TrtllmGenBatchedStridedGemmInfo const* mKernelInfo;
    std::shared_ptr<tensorrt_llm::common::CUDADriverWrapper> mDriver;
    CUmodule mModule;
    CUfunction mFunction;

    static std::array<int, 16> const srcToDstBlk16RowMap;
    static std::array<int, 32> const srcToDstBlk32RowMap;

    template <Data_type T>
    void shuffleMatrixA(void const* input, void* output, int B, int M, int K, int epilogueTileM);
};

// clang-format off
inline const std::array<int, 16> TrtllmGenBatchedGemmRunner::srcToDstBlk16RowMap =
{
    0,  8,
    1,  9,
    2, 10,
    3, 11,
    4, 12,
    5, 13,
    6, 14,
    7, 15
};

inline const std::array<int, 32> TrtllmGenBatchedGemmRunner::srcToDstBlk32RowMap =
{
    0,  8, 16, 24,
    1,  9, 17, 25,
    2, 10, 18, 26,
    3, 11, 19, 27,
    4, 12, 20, 28,
    5, 13, 21, 29,
    6, 14, 22, 30,
    7, 15, 23, 31
};

// clang-format on

template <Data_type T>
void TrtllmGenBatchedGemmRunner::shuffleMatrixA(void const* input, void* output, int B, int M, int K, int epilogueTileM)
{
    int shuffleBlockSize = 16;
    if (epilogueTileM % 128 == 0)
    {
        shuffleBlockSize = 32;
    }
    int numBytesPerRow = K * get_size_in_bytes(T);

    std::vector<uint8_t> tmp(M * numBytesPerRow);
    for (int batchIndex = 0; batchIndex < B; ++batchIndex)
    {
        int const batchRowStride = batchIndex * M;
        for (int mi = 0; mi < M; ++mi)
        {
            int const dstRowBlockIdx = mi / shuffleBlockSize;
            int const srcRowInBlockIdx = mi % shuffleBlockSize;
            int const dstRowInBlockIdx = shuffleBlockSize == 16 ? srcToDstBlk16RowMap[srcRowInBlockIdx]
                                                                : srcToDstBlk32RowMap[srcRowInBlockIdx];
            int const dstRowIdx = dstRowBlockIdx * shuffleBlockSize + dstRowInBlockIdx;
            std::memcpy(&tmp[dstRowIdx * numBytesPerRow],
                &reinterpret_cast<uint8_t const*>(input)[(batchRowStride + mi) * numBytesPerRow], numBytesPerRow);
        }

        // Copy tmp data to the output pointer.
        std::memcpy(
            &reinterpret_cast<uint8_t*>(output)[batchRowStride * numBytesPerRow], tmp.data(), M * numBytesPerRow);
    }
}
} // namespace tensorrt_llm::kernels
