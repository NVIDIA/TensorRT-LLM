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
    int const nPadBeamWidth{padToNextPowerOfTwo(bh.nBeamWidth)};

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
#ifndef FAST_BUILD // Skip beam width > 16
        case 32: CASE_K(32)
        case 64: CASE_K(64)
        case 128: CASE_K(128)
        case 256: CASE_K(256)
        case 512: CASE_K(512)
        case 1024: CASE_K(1024)
#endif // FAST_BUILD
        }
    }
    else // V1, only use kernels of `beam_width <= kMaxBeamWidthForV1`
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

__global__ void updateCacheIndirectionKernel(
    int* tgtCI, int const* srcCI, BeamHypotheses bh, int const nMaxAttentionWindow, int const nSinkTokenLength)
{
    // Update cache indirections which steps are between `bh.inputLength[x]` to `sequenceLengths[x]`
    int const step = blockIdx.x * blockDim.x + threadIdx.x;
    size_t const nBM{bh.nBeamWidth};
    size_t const nBMIn{bh.nBeamWidthIn};
    size_t const nBMOut{bh.nBeamWidthOut};
    size_t const nMSL{bh.nMaxSeqLen};
    int const indexBatch = blockIdx.y;
    int const batchSlot = bh.batchSlots[indexBatch];
    int const tgtIndexBeam = blockIdx.z;
    int const tgtIndexBatchBeam = batchSlot * nBM + tgtIndexBeam;
    int const lastStep{bh.sequenceLengths[tgtIndexBatchBeam] - 1}; // minus 1 since it is updated in stage 3 kernel

    // Return early when at least one of the conditions is true:
    // 1. `step` is out of the bound
    // 2. `step` is inside of input part (since context KV Cache is shared)
    // 3. `step` is outside of attention widow
    if (step >= nMSL || step < bh.inputLengths[tgtIndexBatchBeam] || step < (nMSL - nMaxAttentionWindow))
    {
        return;
    }

    // Keep all past tokens by parentIdsPtr
    int const srcIndexBeam = bh.parentIdsPtr[batchSlot][tgtIndexBeam * nMSL + lastStep];
    // Return early when the source beam isfinished
    if (bh.finished[tgtIndexBatchBeam].isFinished())
    {
        return;
    }

    int const stepCirc = (step >= nSinkTokenLength)
        ? nSinkTokenLength + (step - nSinkTokenLength) % (nMaxAttentionWindow - nSinkTokenLength)
        : step;
    // Consider cyclic kv cache for the indir tables
    uint32_t const tgtOffset = batchSlot * nBMOut * nMaxAttentionWindow + tgtIndexBeam * nMaxAttentionWindow + stepCirc;
    uint32_t const srcOffset = batchSlot * nBMIn * nMaxAttentionWindow + srcIndexBeam * nMaxAttentionWindow + stepCirc;
    tgtCI[tgtOffset] = (step == lastStep) ? tgtIndexBeam : srcCI[srcOffset];
}

void invokeUpdateCacheIndirection(int* tgtCI, int const* srcCI, BeamHypotheses& bh,
    runtime::SizeType32 const maxAttentionWindow, runtime::SizeType32 sinkTokenLength, cudaStream_t stream)
{
    dim3 const grid(common::roundUp(bh.nMaxSeqLen, 32), bh.nBatchSize, bh.nBeamWidthOut);
    updateCacheIndirectionKernel<<<grid, 32, 0, stream>>>(tgtCI, srcCI, bh, maxAttentionWindow, sinkTokenLength);
    sync_check_cuda_error(stream);
}

__global__ void addCumLogProbs(float* __restrict pStage1LogProbs, float const* __restrict cumLogProbs,
    FinishedState const* finished, int const* endIds, float const* diversityRates,
    runtime::SizeType32 const* batchSlots, size_t const nBS, size_t const nBMIn, size_t const nBMOut, size_t const nBM)
{
    int const bid = blockIdx.x; // Index of request in batch
    runtime::SizeType32 const slot = batchSlots[bid];
    float const diversityRate{diversityRates[slot]};
    float* pLocalLogProbs = pStage1LogProbs + bid * nBMIn * nBMOut * 2;

    for (int i = threadIdx.x; i < nBMIn * nBMOut * 2; i += blockDim.x)
    {
        int const iBMIn = i / (nBMOut * 2);
        if (finished[slot * nBMIn + iBMIn].isFinished())
        {
            pLocalLogProbs[i] += (i == endIds[slot]) ? 1.0f : 0.0f;
        }
        else
        {
            // nBM is used in VBWS since `cumLogProbs` is initialized with kMaxBeamWidth earlier than BeamSearchLayer
            pLocalLogProbs[i] += cumLogProbs[slot * nBM + iBMIn] + diversityRate * iBMIn;
        }
    }
    return;
}

__global__ void addCumLogProbs(half* __restrict pStage1LogProbs, float const* __restrict cumLogProbs,
    FinishedState const* finished, int const* endIds, float const* diversityRates,
    runtime::SizeType32 const* batchSlots, size_t const nBS, size_t const nBMIn, size_t const nBMOut, size_t const nBM)
{
    int const bid = blockIdx.x; // Index of request in batch
    runtime::SizeType32 const slot = batchSlots[bid];
    float const diversityRate{diversityRates[slot]};
    half* pLocalLogProbs = pStage1LogProbs + bid * nBMIn * nBMOut * 2;

    for (int i = threadIdx.x; i < nBMIn * nBMOut * 2; i += blockDim.x)
    {
        int const iBMIn = i / (nBMOut * 2);
        if (finished[slot * nBMIn + iBMIn].isFinished())
        {
            pLocalLogProbs[i] += (i == endIds[slot]) ? 1.0f : 0.0f;
        }
        else
        {
            // nBM is used in VBWS since `cumLogProbs` is initialized with kMaxBeamWidth earlier than BeamSearchLayer
            pLocalLogProbs[i] += cumLogProbs[slot * nBM + iBMIn] + diversityRate * iBMIn;
        }
    }
    return;
}

__global__ void gatherId(int const* __restrict pStage1Id, int* __restrict pStage2Id, size_t const nBS,
    size_t const nBMIn, size_t const nBMOut, size_t const nV)
{
    // Use topK output `pStage1Id` and `pStage1Id` to get the index of a new token in `logProbs` for each beam.
    //
    // clang-format off
    //
    // Example for normal beam search:
    // nBS = 3, nBM = 2, nV = 5, use logProbs with integer values here for simplicity.
    // ┏┏ 46 35 47 18 67 ┓┓      ┏┏ 67 47 46 35 ┓┓ ┏┏ 4 2 0 1 ┓┓
    // ┃┗ 76 23 74 73 17 ┛┃      ┃┗ 76 74 73 23 ┛┃ ┃┗ 0 2 3 1 ┛┃
    // ┃┏ 67 49 98 88 74 ┓┃  A   ┃┏ 98 88 74 67 ┓┃ ┃┏ 2 3 4 0 ┓┃  C   ┏ 76 74 73 67 ┓ ┏ 4 5 6 0 ┓  D   ┏ 5 7 8 4 ┓
    // ┃┗ 12 70 77 22 88 ┛┃ ---> ┃┗ 88 77 70 22 ┛┃ ┃┗ 4 2 1 3 ┛┃ ---> ┃ 98 88 88 77 ┃ ┃ 0 1 4 5 ┃ ---> ┃ 2 3 9 7 ┃
    // ┃┏ 55 15 72  3 84 ┓┃      ┃┏ 74 72 55 15 ┓┃ ┃┏ 4 2 0 1 ┓┃      ┗ 98 93 84 77 ┛ ┗ 4 5 0 6 ┛      ┗ 9 6 4 5 ┛
    // ┗┗ 77 93 14 60 98 ┛┛      ┗┗ 98 93 77 60 ┛┛ ┗┗ 4 1 0 3 ┛┛
    //       logProbs              stage1LogProbs     stage1Id         stage2LogProbs   stage2Id     output-stage2Id
    //
    // For `stage2LogProbs[2][3] == 77`,
    //     original batch index in logProbs:    blockIdx.x                  -> 2    (a)
    //     original beam index in logProbs:     stage2Id[2][3] / (nBM * 2)  -> 1    (b)
    //     row index in stage1Probs:            a * nBM + b                 -> 5    (c)
    //     column index in stage1*:             stage2Id[2][3] % (nBM * 2)  -> 2    (d)
    //     column index in logProbs:            stage1Id[c][d]              -> 0    (e)
    //     pad for previous tokens:             b * nV                      -> 5    (f)
    //     final output:                        e + f                       -> 5
    //
    // ========================================================================================================
    // Example for VBWS:
    // nBS = 2, nBMIn = 3, nBMOut = 5, nBM = 7, nV = 11, use logProbs with integer values here for simplicity.
    // ┏┏ 46 35 47 18 67 76 23 74 73 17 67 ┓┓      ┏┏ 76 74 73 67 67 47 46 35 23 18 ┓┓ ┏┏  5  7  8  4 10  2  0  1  6  3 ┓┓
    // ┃┃ 49 98 88 74 12 70 77 22 88 55 15 ┃┃      ┃┃ 98 88 88 77 74 70 55 49 22 15 ┃┃ ┃┃  1  2  8  6  3  5  9  0  7 10 ┃┃
    // ┃┗ 72  3 84 77 93 14 60 98 65  4 20 ┛┃  A   ┃┗ 98 93 84 77 72 65 60 20 14  4 ┛┃ ┃┗  7  4  2  3  0  8  6 10  5  9 ┛┃  C
    // ┃┏ 16 34 71 38 19 91  5 81 97 43 79 ┓┃ ---> ┃┏ 97 91 81 79 71 43 38 34 19 16 ┓┃ ┃┏  8  5  7 10  2  9  3  1  4  0 ┓┃ --->
    // ┃┃  2 22 77 37 57 33 57 41 27 73 88 ┃┃      ┃┃ 88 77 73 57 57 41 37 33 27 22 ┃┃ ┃┃ 10  2  9  4  6  7  3  5  8  1 ┃┃
    // ┗┗ 77 16 23 22 82 89  6 77 67 15 31 ┛┛      ┗┗ 89 82 77 77 67 31 23 22 16 15 ┛┛ ┗┗  5  4  0  7  8 10  2  3  1  9 ┛┛
    //                logProbs                                stage1LogProbs                         stage1Id
    //
    //  C    ┏ 98 98 93 88 88 84 77 77 76 74 ┓ ┏ 10 20 21 11 12 22 13 23  0  1 ┓  D   ┏ 12 29 26 13 19 24 17 25  5  7 ┓
    // --->  ┗ 97 91 89 88 82 81 79 77 77 77 ┛ ┗  0  1 20 10 21  2  3 11 22 23 ┛ ---> ┗  8  5 27 21 26  7 10 13 22 29 ┛
    //                 stage2LogProbs                        stage2Id                          output-stage2Id
    //
    // For `stage2LogProbs[1][4] == 82`,
    //     original batch index in logProbs:    blockIdx.x                      ->  1   (a)
    //     original beam index in logProbs:     stage2Id[1][4] / (nBMOut * 2)   ->  2   (b)
    //     row index in stage1LogProbs:         a * nBMIn + b                   ->  5   (c)
    //     column index in stage1*:             stage2Id[1][4] % (nBMOut * 2)   ->  1   (d)
    //     column index in logProbs:            stage1Id[c][d]                  ->  4   (e)
    //     pad for previous tokens:             b * nV                          -> 22   (f)
    //     final output:                        e + f                           -> 26   output-stage2Id[1][4]
    //
    // clang-format on
    int const a = blockIdx.x; // Index of request in batch
    for (int j = threadIdx.x; j < nBMOut * 2; j += blockDim.x)
    {
        int const index = a * (nBMOut * 2) + j;
        int const stage2Id = pStage2Id[index];
        int const b = stage2Id / (nBMOut * 2);
        int const c = a * nBMIn + b;
        int const d = stage2Id % (nBMOut * 2);
        int const e = pStage1Id[c * (nBMOut * 2) + d];
        int const f = b * nV;
        pStage2Id[index] = e + f;
    }
    return;
}

void BeamHypotheses::print()
{
#if BEAM_SEARCH_DEBUG
    cudaDeviceSynchronize();
    printf("================ print BeamHypotheses start\n");

    PRINT(this->bReturnNormedScore);
    PRINT(this->bVBWS);
    PRINT(this->nMaxBatchSize);
    PRINT(this->nBatchSize);
    PRINT(this->nBeamWidth);
    PRINT(this->nBeamWidthIn);
    PRINT(this->nBeamWidthOut);
    PRINT(this->nMaxSeqLen);
    PRINT(this->nVocabSize);
    PRINT(this->nVPart);
    PRINT(this->nByteMaxSharedMemoryPerBlock);
    PRINT(this->nByteSharedMemoryStage1);
    PRINT(this->nByteSharedMemoryStage3);
    size_t const mbs = this->nMaxBatchSize;
    size_t const nbs = this->nBatchSize;
    size_t const nbm = this->nBeamWidth;
    size_t const nbmo = this->nBeamWidthOut;
    size_t const msl = this->nMaxSeqLen;

    PH2(this->diversityRates, nbs);
    PH2(this->lengthPenalties, nbs);
    PH2(this->earlyStoppings, nbs);
    PH3(this->beamWidthArraysHost, nbs * kMaxBeamWidthArrayLength, kMaxBeamWidthArrayLength);
    PH2(this->nBeamWidthInHost, nbs);
    PH2(this->nBeamWidthOutHost, nbs);

    PH2(this->inputLengths, nbs * nbm);
    PH2(this->endIds, nbs);
    PH2(this->batchSlots, nbs);

    PH3(this->outputIds, nbs * nbm * msl, msl);
    PH3(this->logProbs, nbs * nbm * msl, msl);
    PH3(this->sequenceLengths, nbs * nbm, nbm);
    PH3(this->cumLogProbs, nbs * nbm, nbm);

    PH3(this->outputIdsCBA, mbs * nbmo * 2 * msl, msl);
    PH3(this->logProbsCBA, mbs * nbmo * 2 * msl, msl);
    PH3(this->sequenceLengthsCBA, mbs * nbmo * 2, nbmo * 2);
    PH3(this->cumLogProbsCBA, mbs * nbmo * 2, nbmo * 2);
    PH3(this->normedScoresCBA, mbs * nbmo * 2, nbmo * 2);
    PH2(this->numBeamsCBA, mbs);
    PH2(this->minNormedScoresCBA, mbs);

    // PH2(this->batchDones, nbs);
    uint8_t* finished = reinterpret_cast<uint8_t*>(this->finished);
    PH2(finished, nbs * nbm);

    std::vector<runtime::SizeType32> batchSlots(nbs, 0);
    cudaMemcpy(batchSlots.data(), this->batchSlots, sizeof(runtime::SizeType32) * nbs, cudaMemcpyDeviceToHost);

    std::vector<int*> outputIdsPtr(nbs, 0);
    cudaMemcpy(outputIdsPtr.data(), this->outputIdsPtr, sizeof(int*) * nbs, cudaMemcpyDeviceToHost);

    std::vector<int*> parentIdsPtr(nbs, 0);
    cudaMemcpy(parentIdsPtr.data(), this->parentIdsPtr, sizeof(int*) * nbs, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    for (int i = 0; i < nbs; ++i)
    {
        int slot = batchSlots[i];
        printf("slot=%d\n", slot);
        printf("outputIdsPtr[slot]=%p\n", outputIdsPtr[slot]);
        PH3(outputIdsPtr[slot], nbm * msl, msl);
    }
    for (int i = 0; i < nbs; ++i)
    {
        int slot = batchSlots[i];
        printf("slot=%d\n", slot);
        printf("parentIdsPtr[slot]=%p\n", parentIdsPtr[slot]);
        PH3(parentIdsPtr[slot], nbm * msl, msl);
    }

    // May not available in some context
    // PH3(this->outputIdsUnfinish, nbs * nbm * msl, msl);
    // PH3(this->parentIdsUnfinish, nbs * nbm * msl, msl);

    printf("================ print BeamHypotheses stop\n");
#endif
}

template <typename T>
void printLogProbs(T const* x, int const nBS, int const nBMIn, int const nBM, int const nV)
{
    for (int bs = 0; bs < nBS; ++bs)
    {
        T const* ptrBatch = x + bs * nBM * nV;
        printArrayInfo(ptrBatch, nBMIn * nV, std::string("Request ") + std::to_string(bs));
        for (int bm = 0; bm < nBMIn; ++bm)
        {
            T const* ptrBeam = ptrBatch + bm * nV;
            printArrayInfo(ptrBeam, nV, std::string("Beam ") + std::to_string(bm), true);
        }
    }
}

template void printLogProbs<float>(float const* x, int const nBS, int const nBMIn, int const nBM, int const nV);
template void printLogProbs<half>(half const* x, int const nBS, int const nBMIn, int const nBM, int const nV);

} // namespace kernels
} // namespace tensorrt_llm
