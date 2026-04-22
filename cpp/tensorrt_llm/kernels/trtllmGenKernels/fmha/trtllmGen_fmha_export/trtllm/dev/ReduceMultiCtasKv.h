/*
 * Copyright (c) 2011-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda_ptx/cuda_ptx.h>

#include <cuda/atomic>
#include <cuda/cmath>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "CutlassPipeline.h"
#include "CutlassBarrier.h"
#include "CutlassUtils.h"
#include "FastMath.h"
#include <float.h>
#include "StoreGmemO.h"
#include "ReduceMultiCtasKvUtils.h"
#include <math_constants.h>

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////

// Map the thread's rowIdxO to headIdxO and tokenIdxO.
template <bool GroupHeadsQ, bool GroupTokensQ>
inline __device__ void mapRowToHeadTokenIdx(int32_t rowIdx,
                                            int32_t& headIdxO,
                                            int32_t& tokenIdxO,
                                            int32_t numHeadsQPerKv) {
  static_assert(false, "Not implemented.");
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ void mapRowToHeadTokenIdx<false, true>(int32_t rowIdx,
                                                         int32_t& headIdxO,
                                                         int32_t& tokenIdxO,
                                                         int32_t) {
  // The stats has shape of [numTokensQ, numHeadsQ].
  // Only the tokens are grouped in the row dimension.
  tokenIdxO = rowIdx;
  headIdxO = 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ void mapRowToHeadTokenIdx<true, false>(int32_t rowIdx,
                                                         int32_t& headIdxO,
                                                         int32_t& tokenIdxO,
                                                         int32_t) {
  // The stats has shape of [numTokensQ, numHeadsQ].
  // Only the heads are grouped in the row dimension.
  tokenIdxO = 0;
  headIdxO = rowIdx;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ void mapRowToHeadTokenIdx<true, true>(int32_t rowIdx,
                                                        int32_t& headIdxO,
                                                        int32_t& tokenIdxO,
                                                        int32_t numHeadsQPerKv) {
  // The stats has shape of [numTokensQ, numHeadsQ].
  // If GroupTokensHeadsQ is true, the rowIdx needs to be mapped into tokenIdx and headIdxInGrp.
  // The token index.
  tokenIdxO = rowIdx / numHeadsQPerKv;
  // The head index in the group.
  headIdxO = rowIdx % numHeadsQPerKv;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// This function updates the number of complete CTAs and return the ctaIdxKv that will do the final
// reduction work.

inline __device__ int32_t recordCtaCompletion(int32_t* counter,
                                              int32_t* smemPtr,
                                              int32_t warpGrpThreadIdx,
                                              int32_t numCtasKv,
                                              int32_t numCtasKvForReduction,
                                              int32_t numWarpGrpThreads,
                                              int32_t namedBarId) {
  // Atomic inc to the counter in order to check if it is the last CtaKv.
  cuda::atomic_ref<int32_t, cuda::thread_scope_device> atomicCounter(*counter);

  // Sync the threads in the warp-group to make sure memory stores are moved after the atomic inc.
  trtllm::dev::CutlassNamedBarrier::sync(numWarpGrpThreads, namedBarId);

  // The first thread updates the number of CTAs that have completed their work.
  int32_t numCompleteCtas{-1};
  if (warpGrpThreadIdx == 0) {
    // Use atom.inc to set the counter to zero if it is the last CTA.
    asm volatile("atom.acq_rel.gpu.global.inc.u32 %0, [%1], %2;\n"
                 : "=r"(numCompleteCtas)
                 : "l"(counter), "r"(numCtasKv - 1));

    // The ctaIdxKv that will do the final reduction work (which starts from 0).
    smemPtr[0] = numCtasKv - 1 - numCompleteCtas;

    // If it is not the last CTA, wait for the last CTA to reset the counter.
    if (numCompleteCtas >= (numCtasKv - numCtasKvForReduction) &&
        numCompleteCtas < (numCtasKv - 1)) {
      while (atomicCounter.load() != 0) {
      };
    }
  }

  // Sync the threads in the warp-group to make sure memory stores are visiable to all threads.
  trtllm::dev::CutlassNamedBarrier::sync(numWarpGrpThreads, namedBarId);

  // Return the ctaIdxKv that will do the final reduction work.
  return smemPtr[0];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void storeStatsForAb(float const* maxPtr,
                                       float const* sumPtr,
                                       float* statsPtr,
                                       int32_t statsRowIdx,
                                       int32_t localStatsRowIdx,
                                       bool isValidThread,
                                       int32_t numValidElts) {
  // The shape of stats buffer: [numValidElts], where each element is a float2 (max/sum).

  // The layout depends on the instruction used to load from TMEM. When A and B tensors are not
  // swapped for Mmas, it is LDTM.32dp32bit/LDTM.16dp32bitx2.
  // 1. LDTM.32dp32bit: each thread holds one independent stats values.
  // 2. LDTM.16dp32bitx2: each thread holds one independent stats values.
  //    Only threads with (threadIdx % 32) < 16 will store the stats values.
  if (localStatsRowIdx < numValidElts && isValidThread) {
    reinterpret_cast<float2*>(statsPtr)[statsRowIdx] = make_float2(maxPtr[0], sumPtr[0]);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Barrier>
inline __device__ void storeStatsForAb(float const* maxPtr,
                                       float const* sumPtr,
                                       float* statsPtr,
                                       Barrier* transactionBarrier,
                                       int32_t statsRowIdx,
                                       int32_t remoteSmemCtaRank,
                                       int32_t remoteSmemRowIdx,
                                       bool isValidThread,
                                       int32_t numValidElts) {
  // The shape of stats buffer: [numValidElts], where each element is a float2 (max/sum).

  // The layout depends on the instruction used to load from TMEM. When A and B tensors are not
  // swapped for Mmas, it is LDTM.32dp32bit/LDTM.16dp32bitx2.
  // 1. LDTM.32dp32bit: each thread holds one independent stats values.
  // 2. LDTM.16dp32bitx2: each thread holds one independent stats values.
  //    Only threads with (threadIdx % 32) < 16 will store the stats values.
  if (statsRowIdx < numValidElts && isValidThread) {
    // The underlying barrier in the smem.
    uint64_t* barrier = transactionBarrier->getBarrierPtr();
    // Map the smem address of barrier and max/sum to the remote smem space.
    barrier = cuda_ptx::mapa(cuda_ptx::space_cluster_t{}, barrier, remoteSmemCtaRank);
    statsPtr = cuda_ptx::mapa(cuda_ptx::space_cluster_t{}, statsPtr, remoteSmemCtaRank);
    // Store the max/sum values to the remote smem.
    cuda_ptx::st_async(reinterpret_cast<float2*>(statsPtr) + remoteSmemRowIdx,
                       make_float2(maxPtr[0], sumPtr[0]),
                       barrier);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t NumStatsVals, bool MapRowToHeadTokenIdx, typename Barrier>
inline __device__ void storeStatsForSwappedAb(float const* maxPtr,
                                              float const* sumPtr,
                                              float* partialStatsPtr,
                                              Barrier* transactionBarrier,
                                              int32_t numStatsEltsPerCta,
                                              int32_t warpGrpThreadIdx,
                                              int32_t numValidElts) {
  // The shape of partialStatsPtr buffer: [TileSizePerCtaQ], where each element is a float2
  // (max/sum).

  // The layout depends on the instruction used to load from TMEM. When A and B tensors are swapped
  // for Mmas, it is LDTM.16dp256bit. In this case, the threads are reorganized as 8 x 4 layout, so
  // all quads (4 contiguous threads) have the same stats values after reduction. Only the first
  // quad needs to write the final max/sum values to global memory.

  // This stores the max/sum values to the remote smem.
  // Each CTA has its own smem buffer, and we will split the stats values to different CTA's smem
  // buffer in order to parallelize the reduction work.

  // The mapRowToHeadTokenIdx flag.
  static_assert(!MapRowToHeadTokenIdx, "MapRowToHeadTokenIdx should be false for this function.");

  // The underlying barrier in the smem.
  uint64_t* barrier = transactionBarrier->getBarrierPtr();
  // Store the max/sum values to the remote smem.
  if (warpGrpThreadIdx < 4 * NumStatsVals) {
    // The max/sum values.
    float2 stats{maxPtr[0], sumPtr[0]};
// Select the max/sum values for the current quad of threads.
#pragma unroll
    for (int ii = 1; ii < NumStatsVals; ++ii) {
      bool isSelectedQuad = (warpGrpThreadIdx >= 4 * ii) && (warpGrpThreadIdx < 4 * (ii + 1));
      stats.x = isSelectedQuad ? maxPtr[ii] : stats.x;
      stats.y = isSelectedQuad ? sumPtr[ii] : stats.y;
    }
    // The stats value index.
    int32_t statsIdx = warpGrpThreadIdx / 4;
    // The thread index in the quad.
    int32_t quadThreadIdx = warpGrpThreadIdx % 4;
    // The offset to store the max/sum values.
    int32_t statsOffset = (quadThreadIdx * 2) + (statsIdx / 2) * 8 + (statsIdx % 2);
    // The target cta_rank that will process the current stats.
    int32_t ctaRank = statsOffset / numStatsEltsPerCta;
    // The remote smem rowIdx.
    int32_t remoteSmemRowIdx = statsOffset % numStatsEltsPerCta;
    // Map the smem address of barrier and max/sum to the remote smem space.
    barrier = cuda_ptx::mapa(cuda_ptx::space_cluster_t{}, barrier, ctaRank);
    partialStatsPtr = cuda_ptx::mapa(cuda_ptx::space_cluster_t{}, partialStatsPtr, ctaRank);
    // Store the valid max/sum values to the remote smem.
    if (statsOffset < numValidElts) {
      cuda_ptx::st_async(reinterpret_cast<float2*>(partialStatsPtr) + remoteSmemRowIdx,
                         stats,
                         barrier);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t NumStatsVals, bool MapRowToHeadTokenIdx>
inline __device__ void storeStatsForSwappedAb(float const* maxPtr,
                                              float const* sumPtr,
                                              float* statsGmemPtr,
                                              int32_t numHeadsQ,
                                              int32_t numHeadsQPerKv,
                                              int32_t warpGrpThreadIdx,
                                              int32_t numValidElts) {
  // The shape of stats buffer: [numValidElts], where each element is a float2 (max/sum).

  // The layout depends on the instruction used to load from TMEM. When A and B tensors are swapped
  // for Mmas, it is LDTM.16dp256bit. In this case, the threads are reorganized as 8 x 4 layout, so
  // all quads (4 contiguous threads) have the same stats values after reduction. Only the first
  // quad needs to write the final max/sum values to global memory.
  if (warpGrpThreadIdx < 4 * NumStatsVals) {
    // The max/sum values.
    float2 stats{maxPtr[0], sumPtr[0]};
// Select the max/sum values for the current quad of threads.
#pragma unroll
    for (int ii = 1; ii < NumStatsVals; ++ii) {
      bool isSelectedQuad = (warpGrpThreadIdx >= 4 * ii);
      stats.x = isSelectedQuad ? maxPtr[ii] : stats.x;
      stats.y = isSelectedQuad ? sumPtr[ii] : stats.y;
    }
    // The stats value index.
    int32_t statsIdx = warpGrpThreadIdx / 4;
    // The thread index in the quad.
    int32_t quadThreadIdx = warpGrpThreadIdx % 4;
    // The gmem offset to store the max/sum values.
    int32_t statsGmemOffset = (quadThreadIdx * 2) + (statsIdx / 2) * 8 + (statsIdx % 2);
    // Is it a valid stats row?
    bool isValidStatsRow = statsGmemOffset < numValidElts;
    // If grouping both tokens and headsQ, map statsGmemOffset into tokenIdx and headIdxInGrp and
    // assume numHeadsQPerKv will be handled in one CTA.
    if constexpr (MapRowToHeadTokenIdx) {
      int32_t tokenIdx{statsGmemOffset / numHeadsQPerKv};
      int32_t headIdxInGrp{statsGmemOffset % numHeadsQPerKv};
      statsGmemOffset = tokenIdx * numHeadsQ + headIdxInGrp;
    }
    if (isValidStatsRow) {
      reinterpret_cast<float2*>(statsGmemPtr)[statsGmemOffset] = stats;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t TileSizePerCtaQ,
          int32_t HeadDim,
          int32_t HeadDimPerCta,
          int32_t NumWarpGrpThreads,
          bool GroupsTokensHeadsQ,
          bool IsE4m3Bmm,
          bool UsesCgaReduction,
          typename DtypeO,
          typename DtypePartialO>
inline __device__ void reducePartialO(DtypeO* oPtr,
                                      DtypePartialO const* partialOPtr,
                                      float const* partialStatsPtr,
                                      float const* attentionSinksPtr,
                                      float* softmaxStatsPtr,
                                      float softmaxScaleLog2,
                                      int32_t numCtasKv,
                                      int32_t warpGrpThreadIdx,
                                      int32_t ctaIdxKvForReduction,
                                      int32_t headIdxO,
                                      int32_t numHeadsQ,
                                      trtllm::dev::fast_mod_div numHeadsQPerKvDivisor,
                                      int32_t numValidRows,
                                      bool storesSoftmaxStats,
                                      cutlass::float_e4m3_t* oSfPtr = nullptr,
                                      float sfScale = 0.f,
                                      int32_t sfBaseRowIdx = 0) {

  // clang-format off
  // The shape of partialO buffer: [numCtasKv, TileSizePerCtaQ, headDimPerCta].
  // The shape of final O buffer: [numValidRows, numHeadsKv, headDim].
  // The shape of attentionSinks buffer: [numHeadsQ].
  // The shape of partialStats buffer: [numCtasKv, TileSizePerCtaQ], where each element is a float2 (max/sum).
  // The shape of softmaxStats buffer: [numValidRows], where each element is a float2 (max/sum).
  // The shape of smem buffer: [numCtasKv, NumWarpGrpThreads]. 
  // Note that numValidRows includes both numValidTokens and numHeadsQPerKv if grouping headsQ.
  // clang-format on

  int32_t constexpr NumBytesPerPartialElt{sizeof(DtypePartialO)};
  static_assert(NumBytesPerPartialElt == 2,
                "The data type of partialO should be either fp16 or bf16.");

  // The threads in the warp-group should load different values from one partial output
  // [numValidRows, headDim], and then iterate over partial outputs from different CTAs.
  int32_t constexpr NumEltsPer16BVec{16 / NumBytesPerPartialElt};
  static_assert((HeadDimPerCta * NumBytesPerPartialElt) % 16 == 0, "Not implemented");

  // The number of unrolled instructions to load partialO and partialStats.
  int32_t constexpr UnrollSize{4};

  // The number of processed rows in one slice.
  int32_t constexpr NumBytesPerHeadDim{HeadDimPerCta * NumBytesPerPartialElt};
  int32_t constexpr NumBytePerSlice{NumWarpGrpThreads * 16};
  static_assert(NumBytePerSlice % NumBytesPerHeadDim == 0, "Not implemented");
  int32_t constexpr NumRowsPerSlice{NumBytePerSlice / NumBytesPerHeadDim};
  // The actual number of tensor slices for the reduction.
  int32_t numSlices{(numValidRows + NumRowsPerSlice - 1) / NumRowsPerSlice};

  // The number of slices that each CTA will process.
  int32_t numSlicesPerCta{(numSlices + numCtasKv - 1) / numCtasKv};
  // The start slice index for the current CTA.
  int32_t startSliceIdx{ctaIdxKvForReduction * numSlicesPerCta};
  // The end slice index for the current CTA.
  int32_t endSliceIdx{std::min(startSliceIdx + numSlicesPerCta, numSlices)};

  // The total number of rows in the partial buffers.
  int32_t numRowsInPartialBuffers{TileSizePerCtaQ};
  if constexpr (UsesCgaReduction) {
    numRowsInPartialBuffers = numSlicesPerCta * NumRowsPerSlice;
  }

  //
  // Note: currently we assume TileSizePerCtaQ is relatively small like 8 or 16, so
  // duplicate work of reducing max/sum should make minor difference to the performance.
  // When TileSizePerCtaQ is larger like 64/128, we might need to consider each thread reducing
  // different rows' sum/max values, and make sure they can be reused later when reducing
  // partialO values.
  // Currently, the second read from partialStatsPtr relies on L1 cache as it is relatively small.
  //

  // Iterate over different slices.
  // Split the reduction work across multiple CtasKv to reduce the latency.
  for (int32_t sliceIdx = startSliceIdx; sliceIdx < endSliceIdx; ++sliceIdx) {
    // The base offset that each thread points to.
    int32_t const baseOffset{warpGrpThreadIdx * NumEltsPer16BVec};
    // The index in the row dimension.
    int32_t const rowIdx{sliceIdx * NumRowsPerSlice + (baseOffset / HeadDimPerCta)};
    // Does this thread point to a valid row ?
    bool const isValidRow{rowIdx < numValidRows};
    int32_t validRowIdx{std::min(rowIdx, numValidRows - 1)};
    // If the CGA reduction is used, each CTA has its own smem buffer, and we will split the slices
    // to different CTA's smem buffer in order to parallelize the reduction work, which means the
    // rowIdx should be mapped to the correct CTA's smem buffer (starts from 0).
    int32_t loadRowIdx{validRowIdx};
    if constexpr (UsesCgaReduction) {
      // The index in the headDim dimension.
      loadRowIdx -= (startSliceIdx * NumRowsPerSlice);
    }
    // The index in the headDim dimension.
    int32_t const headDimIdx{baseOffset % HeadDimPerCta};
    // The memory load offfset (can be from remote shared memory or global memory).
    int64_t const destMemOffset{loadRowIdx * HeadDimPerCta + headDimIdx};
    // The memory store offset.
    int64_t gmemStoreOffset{validRowIdx * HeadDim + headDimIdx};
    // The local headIdxO.
    int32_t localHeadIdxO{headIdxO + validRowIdx};
    // The rowIdx of softmaxStats.
    int32_t softmaxStatsRowIdx{validRowIdx};
    // If grouping both tokens and headsQ, map validRowIdx into tokenIdx and headIdxInGrp and
    // assume numHeadsQPerKv will be handled in one CTA.
    if constexpr (GroupsTokensHeadsQ) {
      int32_t tokenIdx{validRowIdx / numHeadsQPerKvDivisor};
      int32_t headIdxInGrp{validRowIdx % numHeadsQPerKvDivisor};
      localHeadIdxO = headIdxO + headIdxInGrp;
      softmaxStatsRowIdx = tokenIdx * numHeadsQ + headIdxInGrp;
      gmemStoreOffset = int64_t(softmaxStatsRowIdx) * HeadDim + headDimIdx;
    }

    // Reduce max, sum and partialO vectors from different CtasKv.
    float sumVal{0.f};
    float oldMaxVal{-FLT_MAX}, maxVal{-FLT_MAX};
    float outputVals[NumEltsPer16BVec] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int32_t ii = 0; ii < numCtasKv; ii += UnrollSize) {
      // The partialStats array and partialO array.
      float2 partialStatsArray[UnrollSize];
      uint4 partialOArray[UnrollSize];
#pragma unroll
      for (int32_t jj = 0; jj < UnrollSize; ++jj) {
        int32_t ctaIdxKv = min(ii + jj, numCtasKv - 1);
        partialStatsArray[jj] = reinterpret_cast<float2 const*>(
          partialStatsPtr)[ctaIdxKv * numRowsInPartialBuffers + loadRowIdx];
        partialOArray[jj] = *reinterpret_cast<uint4 const*>(
          partialOPtr + destMemOffset + ctaIdxKv * numRowsInPartialBuffers * HeadDimPerCta);
      }
#pragma unroll
      for (int32_t jj = 0; jj < UnrollSize; ++jj) {
        // Whether the ctaIdxKv is valid.
        bool const isValidCtaIdxKv = (ii + jj) < numCtasKv;
        // The local max and sum values.
        auto partialStats = partialStatsArray[jj];
        float localMax = partialStats.x;
        float localSum = partialStats.y;
        // Update the max value.
        maxVal = fmaxf(maxVal, localMax);
        // Compute the correction scales.
        float corrScale0 = isValidCtaIdxKv ? exp2f(softmaxScaleLog2 * (oldMaxVal - maxVal)) : 1.f;
        float corrScale1 = isValidCtaIdxKv ? exp2f(softmaxScaleLog2 * (localMax - maxVal)) : 0.f;
        // Update the old max value.
        oldMaxVal = maxVal;
        // The partialO value.
        uint4 vec = partialOArray[jj];
        // Reduce sum and finalO.
        sumVal = sumVal * corrScale0 + localSum * corrScale1;
        trtllm::dev::convertToFloatAndAccumulate<DtypePartialO>(outputVals,
                                                                vec,
                                                                corrScale0,
                                                                corrScale1);
      }
    }

    // Update the sums with the attention sink value.
    if (attentionSinksPtr != nullptr) {
      float attentionSinkVal =
        exp2f(attentionSinksPtr[localHeadIdxO] * CUDART_L2E_F - maxVal * softmaxScaleLog2);
      // Multiply the attention sink value by 448.f if the MMA data type is e4m3 as the sum value
      // has also included the 448.f quantization scale.
      sumVal += IsE4m3Bmm ? attentionSinkVal * 448.f : attentionSinkVal;
    }

    // Stores the final softmax stats values to global memory if needed (Helix attention, which
    // splits seqLenKv across GPUs).
    if (storesSoftmaxStats && isValidRow && headDimIdx == 0) {
      // The softmaxScale.
      float softmaxScale = (softmaxScaleLog2 * (1.f / CUDART_L2E_F));
      // The sumScale to unscale the 448.f quantization scale from P.
      float sumScale = IsE4m3Bmm ? (1.f / 448.f) : 1.f;
      // The final max and sum values.
      float2 stats{maxVal * softmaxScale, sumVal * sumScale};
      // Store the final max and sum values to global memory.
      reinterpret_cast<float2*>(softmaxStatsPtr)[softmaxStatsRowIdx] = stats;
    }

    // The final normalized scale.
    // If the output data type is e4m3, make sure that sumVal is divided by the quantization scale
    // (448.f), so 1.0f / (sumVal / 448.f) = 448.f / sumVal.
    float normalizedScale{IsE4m3Bmm ? (448.f / sumVal) : (1.0f / sumVal)};
    float2 normalizedScale2{normalizedScale, normalizedScale};

    // Apply the normalized scale to the reduced O values.
    for (int ii = 0; ii < NumEltsPer16BVec / 2; ++ii) {
      float2& f2 = reinterpret_cast<float2*>(outputVals)[ii];
      cute::mul(f2, f2, normalizedScale2);
    }

    // Convert the float values to DtypeO, and Store it to global memory.
    if constexpr (std::is_same_v<DtypeO, cutlass::float_e2m1_t>) {
      // The number of E2m1 elements packed in a byte.
      int32_t constexpr NumE2m1EltsPerByte = 2;
      // The number of elements per sf.
      int32_t constexpr NumEltsPerSf = 16;
      // The number of cols of SF per block.
      int32_t constexpr NumColsPerSfBlock = 4;
      // The size of each SF block.
      int32_t constexpr NumBytesPerSfBlock = 512;

      // The number of elements per sf.
      int32_t constexpr HeadDimSf = HeadDim / NumEltsPerSf;
      // The offset to store the SF values.
      int64_t storeGmemSfOffset;

      // When grouping tokens and headsQ, each row packs (tokenIdx, headIdxInGrp).
      // Use getSfOffset with absolute indices for correct Layout128x4 mapping,
      // since different rows map to different SF rows (tokens) and columns (head groups).
      if constexpr (GroupsTokensHeadsQ) {
        int32_t tokenIdx{validRowIdx / numHeadsQPerKvDivisor};
        int32_t headIdxInGrp{validRowIdx % numHeadsQPerKvDivisor};
        int32_t sfCol = headIdxInGrp * HeadDimSf + headDimIdx / NumEltsPerSf;
        int32_t numSfPerRow{numHeadsQ * HeadDimSf};
        storeGmemSfOffset = getSfOffset(sfBaseRowIdx + tokenIdx, sfCol, numSfPerRow);
      } else {
        // Without GroupsTokensHeadsQ, all rows share the same head and the SF column
        // is simply derived from gmemStoreOffset. The oSfPtr already accounts for the
        // base head and token offsets.
        int32_t sfColIdx = gmemStoreOffset / NumEltsPerSf;
        storeGmemSfOffset =
          sfColIdx / NumColsPerSfBlock * NumBytesPerSfBlock + sfColIdx % NumColsPerSfBlock;
      }
      convertAndStoreToGmem<DtypeO>(
        reinterpret_cast<char*>(oPtr + gmemStoreOffset / NumE2m1EltsPerByte),
        reinterpret_cast<char*>(oSfPtr) + storeGmemSfOffset,
        outputVals,
        sfScale,
        isValidRow);
    } else {
      if (isValidRow) {
        convertAndStoreToGmem<DtypeO>(reinterpret_cast<char*>(oPtr + gmemStoreOffset), outputVals);
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t TileSizePerCtaQ,
          int32_t HeadDim,
          int32_t HeadDimPerCta,
          int32_t NumWarpGrpThreads,
          bool GroupsTokensHeadsQ,
          bool IsE4m3Bmm,
          bool UsesCgaReduction,
          typename DtypeO,
          typename DtypePartialO,
          typename Barrier>
inline __device__ void reducePartialO(DtypeO* oPtr,
                                      DtypePartialO const* partialOPtr,
                                      float const* partialStatsPtr,
                                      float const* attentionSinksPtr,
                                      float* softmaxStatsPtr,
                                      Barrier* transactionBarrier,
                                      int32_t completionBytes,
                                      float softmaxScaleLog2,
                                      int32_t numCtasKv,
                                      int32_t warpGrpThreadIdx,
                                      int32_t ctaIdxKvForReduction,
                                      int32_t headIdxO,
                                      int32_t numHeadsQ,
                                      trtllm::dev::fast_mod_div numHeadsQPerKvDivisor,
                                      int32_t numValidRows,
                                      bool storesSoftmaxStats,
                                      cutlass::float_e4m3_t* oSfPtr = nullptr,
                                      float sfScale = 0.f,
                                      int32_t sfBaseRowIdx = 0) {

  //
  // If the CGA reduction is used, we need to use the transaction barrier to synchronize the
  // reduction.
  //
  static_assert(UsesCgaReduction, "Not implemented");

  // Perform the completion of the unused bytes.
  if (warpGrpThreadIdx == 0) {
    transactionBarrier->complete_transaction(cute::block_rank_in_cluster(), completionBytes);
  }
  // Wait for all the CTAs in the cluster to complete the remote store.
  transactionBarrier->wait();

  // Perform the reduction on the smem.
  reducePartialO<TileSizePerCtaQ,
                 HeadDim,
                 HeadDimPerCta,
                 NumWarpGrpThreads,
                 GroupsTokensHeadsQ,
                 IsE4m3Bmm,
                 UsesCgaReduction>(oPtr,
                                   partialOPtr,
                                   partialStatsPtr,
                                   attentionSinksPtr,
                                   softmaxStatsPtr,
                                   softmaxScaleLog2,
                                   numCtasKv,
                                   warpGrpThreadIdx,
                                   ctaIdxKvForReduction,
                                   headIdxO,
                                   numHeadsQ,
                                   numHeadsQPerKvDivisor,
                                   numValidRows,
                                   storesSoftmaxStats,
                                   oSfPtr,
                                   sfScale,
                                   sfBaseRowIdx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace dev
} // namespace trtllm
