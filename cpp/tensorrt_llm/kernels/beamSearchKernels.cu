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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/beamSearchKernels.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

template <typename T, int PBM, bool IS_V2>
void beamSearchKernelLauncher(
    T const* logProbs, T const* bias, void* workspace, BeamHypotheses& bh, cudaStream_t stream);

#define CASE_K(PBM)                                                                                                    \
    {                                                                                                                  \
        beamSearchKernelLauncher<T, PBM, IS_V2>(logProbs, bias, workspace, bh, stream);                                \
        break;                                                                                                         \
    }

template <typename T, bool IS_V2>
void invokeTopkBeamSearch(T const* logProbs, T const* bias, void* workspace, BeamHypotheses& bh, cudaStream_t stream)
{
    int const nPadBeamWidth = padToNextPowerOfTwo(bh.nBeamWidth);

    // case X means X/2 < beam_width <= X
    if constexpr (IS_V2)
    {
        switch (nPadBeamWidth)
        {
        case 1:
        case 2:
        case 4: CASE_K(4)
        case 8: CASE_K(8)
        case 16: CASE_K(16)
#ifndef FAST_BUILD // Skip beam_width larger than 16
        case 32: CASE_K(32)
        case 64: CASE_K(64)
        case 128: CASE_K(128)
        case 256: CASE_K(256)
        case 512: CASE_K(512)
        case 1024: CASE_K(1024)
#endif // FAST_BUILD
        }
    }
    else // V1, only use kernels of `beam_width <= nMaxBeamWidthForV1`
    {
        switch (nPadBeamWidth)
        {
        case 1:
        case 2:
        case 4: CASE_K(4)
        case 8: CASE_K(8)
        }
    }
}

#undef CASE_K

template void invokeTopkBeamSearch<float, false>(
    float const* logProbs, float const* bias, void* workspace, BeamHypotheses& bh, cudaStream_t stream);

template void invokeTopkBeamSearch<float, true>(
    float const* logProbs, float const* bias, void* workspace, BeamHypotheses& bh, cudaStream_t stream);

template void invokeTopkBeamSearch<half, false>(
    half const* logProbs, half const* bias, void* workspace, BeamHypotheses& bh, cudaStream_t stream);

template void invokeTopkBeamSearch<half, true>(
    half const* logProbs, half const* bias, void* workspace, BeamHypotheses& bh, cudaStream_t stream);

template <typename T>
__global__ void addCumLogProbs(T* __restrict pStage1Probs, float const* __restrict cumLogProbs,
    FinishedState const* finished, int const* endIds, float const* diversityRates,
    runtime::SizeType32 const* batchSlots, size_t const nBS, size_t const nBM)
{
    int const bid = blockIdx.x;
    float const diversityRate{diversityRates[batchSlots[bid]]};
    T* pLocalProbs = pStage1Probs + bid * nBM * nBM * 2;

    for (int index = threadIdx.x; index < nBM * nBM * 2; index += blockDim.x)
    {
        int const indexBM = index / (nBM * 2);
        if (finished[bid * nBM + indexBM].isFinished())
        {
            pLocalProbs[index] += (index == endIds[bid]) ? 1.0f : 0.0f;
        }
        else
        {
            pLocalProbs[index] += cumLogProbs[bid * nBM + indexBM] + diversityRate * indexBM;
        }
    }
    return;
}

template __global__ void addCumLogProbs<float>(float* __restrict pStage1Probs, float const* __restrict cumLogProbs,
    FinishedState const* finished, int const* endIds, float const* diversityRates,
    runtime::SizeType32 const* batchSlots, size_t const nBS, size_t const nBM);

template __global__ void addCumLogProbs<half>(half* __restrict pStage1Probs, float const* __restrict cumLogProbs,
    FinishedState const* finished, int const* endIds, float const* diversityRates,
    runtime::SizeType32 const* batchSlots, size_t const nBS, size_t const nBM);

__global__ void gatherId(
    int const* __restrict pStage1Id, int* __restrict pStage2Id, size_t const nBS, size_t const nBM, size_t const nV)
{
    // Example (definition of the processes and variables and are in `beamSearchKernelsTemplate.h`).
    // nBS = 3, nBM = 2, nV = 5, use logProbs with integer values here for simplicity.
    // ┏┏ 46 35 47 18 67 ┓┓      ┏┏ 67 47 46 35 ┓┓ ┏┏ 4 2 0 1 ┓┓
    // ┃┗ 76 23 74 73 17 ┛┃      ┃┗ 76 74 73 23 ┛┃ ┃┗ 0 2 3 1 ┛┃
    // ┃┏ 67 49 98 88 74 ┓┃  A   ┃┏ 98 88 74 67 ┓┃ ┃┏ 2 3 4 0 ┓┃  C   ┏ 76 74 73 67 ┓ ┏ 4 5 6 0 ┓  D   ┏ 5 7 8 4 ┓
    // ┃┗ 12 70 77 22 88 ┛┃ ---> ┃┗ 88 77 70 22 ┛┃ ┃┗ 4 2 1 3 ┛┃ ---> ┃ 98 88 88 77 ┃ ┃ 0 1 4 5 ┃ ---> ┃ 2 3 9 7 ┃
    // ┃┏ 55 15 72  3 84 ┓┃      ┃┏ 74 72 55 15 ┓┃ ┃┏ 4 2 0 1 ┓┃      ┗ 98 93 84 77 ┛ ┗ 4 5 0 6 ┛      ┗ 9 6 4 5 ┛
    // ┗┗ 77 93 14 60 98 ┛┛      ┗┗ 98 93 77 60 ┛┛ ┗┗ 4 1 0 3 ┛┛
    //       logProbs               stage1Probs      stage1Id           stage2Probs    stage2Id      output-stage2Id
    //
    // For `stage2Probs[2][3] == 77`,
    //     original batch index in logProbs:    bid                         -> 2    (a)
    //     original beam index in logProbs:     stage2Id[2][3] / (nBM * 2)  -> 1    (b)
    //     row index in stage1Probs:            a * nBM + b                 -> 5    (c)
    //     column index in stage1*:             stage2Id[2][3] % (nBM * 2)  -> 2    (d)
    //     column index in logProbs:            stage1Id[c][d]              -> 0    (e)
    //     pad for previous tokens:             b * nV                      -> 5    (f)
    //     final output:                        e + f                       -> 5

    int const bid = blockIdx.x;
    for (int j = threadIdx.x; j < nBM * 2; j += blockDim.x)
    {
        int const index = pStage2Id[bid * nBM * 2 + j];
        int const iBM = index / (nBM * 2);
        int const jBM = index % (nBM * 2);
        pStage2Id[bid * nBM * 2 + j] = pStage1Id[(bid * nBM + iBM) * (nBM * 2) + jBM] + iBM * nV;
    }
    return;
}

} // namespace kernels
} // namespace tensorrt_llm
