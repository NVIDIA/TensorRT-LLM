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

#include "CutlassBarrier.h"
#include "FastMath.h"
#include "StoreSmemP.h"
#include "Fp4Utils.h"
#include <cuda_bf16.h>

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t NumRegs>
inline __device__ void copyFromSmemToGmemAndConvertToE2m1(char const* src,
                                                          char* dst,
                                                          char* dstSf,
                                                          float sfScale,
                                                          bool isValidRow) {
  static_assert(false, "Not implemented.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ void copyFromSmemToGmemAndConvertToE2m1<2>(char const* src,
                                                             char* dst,
                                                             char* dstSf,
                                                             float sfScale,
                                                             bool isValidRow) {
  // Load from SMEM.
  auto in = reinterpret_cast<uint64_t const*>(src)[0];

  // Convert to E2m1.
  cutlass::float_e4m3_t sfOut;
  uint16_t valOut;
  convertFp16ToE2m1<4>(valOut, sfOut, reinterpret_cast<uint32_t(&)[2]>(in), sfScale);
  // Each group of 4 threads maps to the same SF.
  if (isValidRow) {
    // Store the output to GMEM. Each group of 4 threads maps to the same SF.
    reinterpret_cast<uint16_t*>(dst)[0] = valOut;
    if (threadIdx.x % 4 == 0) {
      reinterpret_cast<cutlass::float_e4m3_t*>(dstSf)[0] = sfOut;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
inline __device__ void copyFromSmemToGmemAndConvertToE2m1<4>(char const* src,
                                                             char* dst,
                                                             char* dstSf,
                                                             float sfScale,
                                                             bool isValidRow) {
  // Load from SMEM.
  auto in = reinterpret_cast<cutlass::uint128_t const*>(src)[0];

  // Convert to E2m1.
  cutlass::float_e4m3_t sfOut;
  uint32_t valOut;
  convertFp16ToE2m1<8>(valOut, sfOut, reinterpret_cast<uint32_t(&)[4]>(in), sfScale);
  // Each pair of threads maps to the same SF.
  if (isValidRow) {
    // Store the output to GMEM. Each pair of threads maps to the same SF.
    reinterpret_cast<uint32_t*>(dst)[0] = valOut;
    if (threadIdx.x % 2 == 0) {
      reinterpret_cast<cutlass::float_e4m3_t*>(dstSf)[0] = sfOut;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t NumRegs, bool StoreToRemoteSmem> struct StoreVec {
  static inline __device__ void store(char* dstMemPtr, char* srcMemPtr, uint64_t*, int32_t) {
    static_assert(false, "Not implemented.");
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t NumRegs> struct StoreVec<NumRegs, false> {
  static inline __device__ void store(char* gmemPtr, char* srcMemPtr, uint64_t*, int32_t) {
    // The vectorized store.
    using VecType = cutlass::AlignedArray<uint32_t, NumRegs>;
    reinterpret_cast<VecType*>(gmemPtr)[0] = reinterpret_cast<VecType const*>(srcMemPtr)[0];
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t NumRegs> struct StoreVec<NumRegs, true> {
  static inline __device__ void store(char* remoteSmemPtr,
                                      char* srcMemPtr,
                                      uint64_t* barrier,
                                      int32_t targetCtaRank) {
    // The vectorized store.
    using VecType = cutlass::AlignedArray<uint32_t, NumRegs>;
    // The srcMem vector.
    VecType srcMemVec = reinterpret_cast<VecType const*>(srcMemPtr)[0];
    // Map the smem barrier and buffer to the lead CTA (0).
    barrier = cuda_ptx::mapa(cuda_ptx::space_cluster_t{}, barrier, targetCtaRank);
    remoteSmemPtr = cuda_ptx::mapa(cuda_ptx::space_cluster_t{}, remoteSmemPtr, targetCtaRank);
    // Store the vector to remote smem.
    static_assert(NumRegs <= 4, "Not implemented.");
    cuda_ptx::st_async(reinterpret_cast<uint32_t*>(remoteSmemPtr),
                       reinterpret_cast<uint32_t(&)[NumRegs]>(srcMemVec),
                       barrier);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Vectorized copy from swizzled shared memory to destination memory (global memory or remote shared
// memory).

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t NumRegs,
          int32_t NumRows,
          int32_t NumBytesPerRow,
          int32_t RowStrideSf,
          int32_t NumSfPerHead,
          bool CastToE2m1,
          bool StoreToRemoteSmem,
          bool MapRowToHeadTokenIdx>
inline __device__ void copyFromSmemToDstMem(char* smemPtr,
                                            char* dstMemPtr,
                                            uint64_t* transactionBarrier,
                                            int32_t baseOffset,
                                            int32_t dstMemRowStrideInBytes,
                                            int32_t numValidRows,
                                            int32_t numReductionRowsPerCta,
                                            int32_t numHeadsQ,
                                            trtllm::dev::fast_mod_div numHeadsQPerKv,
                                            char* dstMemSf = nullptr,
                                            float sfScale = 0.f,
                                            int32_t sfBaseRowIdx = 0) {
  // The number of bytes per row in smem.
  // The leading dimension of smem will be split into multiple slices if it exceeds 128B.
  int32_t constexpr NumBytesPerSmemRow{std::min(NumBytesPerRow, 128)};
  // The number of bytes per row in dstMem.
  int32_t constexpr NumBytesPerDstMemRow{NumBytesPerRow};
  // Each thread loads one 16B vector from shared memory, and stores it to global memory.
  int32_t constexpr NumBytesPerVec{16};
  // The number of packed rows for swizzling.
  int32_t constexpr NumPackedSmemRows{128 / NumBytesPerSmemRow};

  // Calculate the shared memory offset.
  int32_t const smemRowIdx{baseOffset / NumBytesPerSmemRow};
  // There will be 8 swizzle rows considering packed rows.
  int64_t const loadSmemOffset{baseOffset ^
                               (((smemRowIdx / NumPackedSmemRows) % 8) * NumBytesPerVec)};

  // The destination memory row index.
  int32_t dstMemRowIdx{smemRowIdx};
  // The destination memory column offset.
  int32_t dstMemColOffset{(baseOffset % NumBytesPerSmemRow)};

  // The leading dimension will be split into multiple slices if it exceeds 128B.
  // The smem shape [NumLeadingDimSlices, NumRows, NumBytesPerSmemRow].
  if constexpr (NumBytesPerDstMemRow > 128) {
    dstMemRowIdx = smemRowIdx % NumRows;
    dstMemColOffset =
      (smemRowIdx / NumRows) * NumBytesPerSmemRow + (baseOffset % NumBytesPerSmemRow);
  }

  // Whether the row is valid.
  bool isValidRow = dstMemRowIdx < numValidRows;

  // Copy from shared memory to global memory.
  if constexpr (CastToE2m1) {

    //
    // Store the SFs in Layout128x4 (see trtllm/gen/DtypeUtils.h for details).
    // All elements in the tile [numHeadsQPerKv (row), headDim (col)] are mapped to the col
    // dimension of SFs.
    //

    // Assume elements are stored in SMEM as FP16.
    int32_t constexpr NumBytesPerFp16Elt = 2;
    // The number of E2m1 elements packed in a byte.
    int32_t constexpr NumE2m1EltsPerByte = 2;
    // The number of elements per sf.
    int32_t constexpr NumEltsPerSf = 16;
    // The number of cols of SF per block.
    int32_t constexpr NumColsPerSfBlock = 4;
    // The size of each SF block.
    int32_t constexpr NumBytesPerSfBlock = 512;

    // The offset to store the SF values.
    int64_t dstMemSfOffset;

    // When grouping tokens and headsQ, unpack the row into (tokenIdx, headIdxInGrp)
    // and use getSfOffset with absolute indices for correct Layout128x4 mapping.
    if constexpr (MapRowToHeadTokenIdx) {
      // The token index.
      int32_t tokenIdx{dstMemRowIdx / numHeadsQPerKv};
      // The head index in the group.
      int32_t headIdxInGrp{dstMemRowIdx % numHeadsQPerKv};
      // The new row index after unpacking.
      dstMemRowIdx = headIdxInGrp + tokenIdx * numHeadsQ;
      // The local SF column index.
      int32_t localSfCol{dstMemColOffset / NumBytesPerFp16Elt / NumEltsPerSf};
      // The SF column index.
      int32_t sfHeadCol{headIdxInGrp * NumSfPerHead};
      // The number of SFs per row.
      int32_t numSfPerRow{numHeadsQ * NumSfPerHead};
      // The offset to store the SF values.
      dstMemSfOffset = getSfOffset(sfBaseRowIdx + tokenIdx, sfHeadCol + localSfCol, numSfPerRow);
    } else {
      // Without MapRowToHeadTokenIdx, all rows share the same head and the SF column
      // is simply derived from the column offset. The dstMemSf pointer already accounts
      // for the base head and token offsets.
      auto sfColIdx = dstMemColOffset / NumBytesPerFp16Elt / NumEltsPerSf;
      dstMemSfOffset = dstMemRowIdx * RowStrideSf +
                       sfColIdx / NumColsPerSfBlock * NumBytesPerSfBlock +
                       sfColIdx % NumColsPerSfBlock;
    }

    // Compute data destination offset.
    int64_t dstMemOffset{dstMemRowIdx * static_cast<int64_t>(dstMemRowStrideInBytes) +
                         dstMemColOffset};
    dstMemOffset = dstMemOffset / NumBytesPerFp16Elt / NumE2m1EltsPerByte;

    copyFromSmemToGmemAndConvertToE2m1<NumRegs>(smemPtr + loadSmemOffset,
                                                dstMemPtr + dstMemOffset,
                                                dstMemSf + dstMemSfOffset,
                                                sfScale,
                                                isValidRow);
  } else {
    // If it groups both headsQ and tokensQ into one CTA, we need to unpack the row index to the
    // valid range if values are stored to finalO.
    if constexpr (MapRowToHeadTokenIdx) {
      dstMemRowIdx = (dstMemRowIdx % numHeadsQPerKv) + (dstMemRowIdx / numHeadsQPerKv) * numHeadsQ;
    }

    // The destination memory offset.
    int64_t dstMemOffset{dstMemRowIdx * static_cast<int64_t>(dstMemRowStrideInBytes) +
                         dstMemColOffset};

    // If the CGA reduction is used, we store the output to remote smem.
    int32_t targetCtaRank = 0;
    if constexpr (StoreToRemoteSmem) {
      targetCtaRank = dstMemRowIdx / numReductionRowsPerCta;
      int32_t remoteSmemRowIdx = dstMemRowIdx % numReductionRowsPerCta;
      dstMemOffset =
        remoteSmemRowIdx * static_cast<int64_t>(dstMemRowStrideInBytes) + dstMemColOffset;
    }

    if (isValidRow) {
      // The destination in GMEM/Remote SMEM.
      dstMemPtr += dstMemOffset;
      // The source in SMEM.
      smemPtr += loadSmemOffset;
      // Create a vectorized store.
      StoreVec<NumRegs, StoreToRemoteSmem>::store(dstMemPtr,
                                                  smemPtr,
                                                  transactionBarrier,
                                                  targetCtaRank);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Vectorized copy from registers to destination memory (global memory or remote shared memory).

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int32_t NumRegs>
inline __device__ void storeVec(T* dstMemPtr, uint32_t (&srcVecs)[NumRegs]) {

  // Create a vectorized store.
  StoreVec<NumRegs, false>::store(reinterpret_cast<char*>(dstMemPtr),
                                  reinterpret_cast<char*>(srcVecs),
                                  nullptr,
                                  0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, typename Barrier, int32_t NumRegs>
inline __device__ void storeVecToRemoteSmem(T* dstMemPtr,
                                            uint32_t (&srcVecs)[NumRegs],
                                            Barrier* transactionBarrier,
                                            int32_t targetCtaRank) {

  // Create a vectorized store.
  StoreVec<NumRegs, true>::store(reinterpret_cast<char*>(dstMemPtr),
                                 reinterpret_cast<char*>(srcVecs),
                                 transactionBarrier->getBarrierPtr(),
                                 targetCtaRank);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Reorganize data in shared memory for coalesced memory store to global memory.

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t NumRows,
          int32_t NumCols,
          int32_t RowStrideSf,
          int32_t NumSfPerHead,
          class DtypeO,
          int NumRegs,
          bool CastToE2m1 = false,
          bool StoreToRemoteSmem = false,
          bool MapRowToHeadTokenIdx = false>
inline __device__ void reorganizeInSmemAndStoreToDstMemImpl_(
  DtypeO* smemPtrO,
  void* dstMemPtrO,
  uint64_t* transactionBarrier,
  uint32_t (&arrayO)[NumRegs],
  int32_t dstMemRowStride,
  int32_t numValidTokens,
  int32_t numReductionRowsPerCta,
  int32_t numHeadsQ,
  trtllm::dev::fast_mod_div numHeadsQPerKv,
  int32_t numWarpGrpThreads,
  int32_t warpGrpThreadIdx,
  int32_t namedBarId,
  cutlass::float_e4m3_t* dstMemPtrOSf = nullptr,
  float sfScale = 0.f,
  int32_t sfBaseRowIdx = 0) {

  static_assert((NumRegs & (NumRegs - 1)) == 0, "Not implemented.");

  // The allowed data types.
  static_assert(std::is_same_v<DtypeO, cutlass::float_e4m3_t> ||
                  std::is_same_v<DtypeO, cutlass::half_t> ||
                  std::is_same_v<DtypeO, cutlass::bfloat16_t>,
                "Not implemented.");
  static_assert(!CastToE2m1 || std::is_same_v<DtypeO, cutlass::half_t>,
                "DtypeO should be Fp16 if cast to E2M1.");

  // The number of rows and cols after transposing.
  int32_t constexpr NumTransRows{NumCols};
  // Assume that each warp in the warpgroup (4 warps in total) stores 16/32 transposed columns.
  int32_t constexpr NumTransCols{NumRows};
  // Only 64 or 128 transposed columns are allowed.
  static_assert(NumTransCols == 64 || NumTransCols == 128, "Not implemented.");
  // The number of transposed rows must be power of 2 and greater than or equal to 8.
  static_assert(NumTransRows >= 8 && (NumTransRows & (NumTransRows - 1)) == 0, "Not implemented.");

  // Whether or not to force 16B transactions.
  bool constexpr Force16B{true};
  // The maxiumum number of registers per copy per thread.
  int32_t constexpr MaxNumRegsPerCopy{4};
  // The required number of registers per copy.
  int32_t constexpr RequiredNumRegsPerCopy{std::min(NumRegs, MaxNumRegsPerCopy)};
  // The number of copy calls.
  int32_t constexpr NumCopyCalls{NumRegs / RequiredNumRegsPerCopy};
  // The number of registers per copy to actually use.
  int32_t constexpr NumRegsPerCopy{Force16B ? MaxNumRegsPerCopy : RequiredNumRegsPerCopy};
  // The number of bytes per copy.
  int32_t constexpr NumBytesPerCopy{NumRegsPerCopy * 4};
  // The portion of threads participating in the write
  int32_t constexpr ThreadsToUse{NumRegsPerCopy / RequiredNumRegsPerCopy};

  // The number of bytes per element.
  int32_t constexpr NumBytesPerElt{sizeof(DtypeO)};

  // The base smem/dstMem pointer in char.
  char* const baseSmemPtr{reinterpret_cast<char*>(smemPtrO)};
  char* const baseDstMemPtr{reinterpret_cast<char*>(dstMemPtrO)};

  // The dstMem row stride in bytes.
  int32_t dstMemRowStrideInBytes{dstMemRowStride * NumBytesPerElt};

  // Store transposed tensor to smem (NumRows = 128).
  if constexpr (std::is_same_v<DtypeO, cutlass::float_e4m3_t>) {
    storeTransposedSmem8b<NumTransRows, NumTransCols>(smemPtrO, arrayO, warpGrpThreadIdx);
  } else if constexpr (NumTransCols == 128) {
    storeTransposedSmem128x16b<NumTransRows>(smemPtrO, arrayO, warpGrpThreadIdx);
  } else {
    storeTransposedSmem64x16b<NumTransRows>(smemPtrO, arrayO, warpGrpThreadIdx);
  }
  // Sync the threads in the warpgroup.
  trtllm::dev::CutlassNamedBarrier::sync(numWarpGrpThreads, namedBarId);

  // Multiple copy instructions are needed if there are more than one 16B vec per thread.
  for (int32_t ii = 0; ii < NumCopyCalls; ++ii) {
    // The base offset for each thread.
    // Each thread copies at most 16B.
    int32_t const baseOffset{warpGrpThreadIdx * NumBytesPerCopy +
                             ii * numWarpGrpThreads * NumBytesPerCopy};

    // Copy from smem to dstMem.
    copyFromSmemToDstMem<NumRegsPerCopy,
                         NumTransRows,
                         NumTransCols * NumBytesPerElt,
                         RowStrideSf,
                         NumSfPerHead,
                         CastToE2m1,
                         StoreToRemoteSmem,
                         MapRowToHeadTokenIdx>(baseSmemPtr,
                                               baseDstMemPtr,
                                               transactionBarrier,
                                               baseOffset,
                                               dstMemRowStrideInBytes,
                                               numValidTokens,
                                               numReductionRowsPerCta,
                                               numHeadsQ,
                                               numHeadsQPerKv,
                                               reinterpret_cast<char*>(dstMemPtrOSf),
                                               sfScale,
                                               sfBaseRowIdx);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t NumRows, int32_t NumCols, bool MapRowToHeadTokenIdx, class DtypeO, int NumRegs>
inline __device__ void reorganizeInSmemAndStoreToDstMem(DtypeO* smemPtrO,
                                                        void* dstMemPtrO,
                                                        uint32_t (&arrayO)[NumRegs],
                                                        int32_t dstMemRowStride,
                                                        int32_t numValidTokens,
                                                        int32_t numHeadsQ,
                                                        trtllm::dev::fast_mod_div numHeadsQPerKv,
                                                        int32_t numWarpGrpThreads,
                                                        int32_t warpGrpThreadIdx,
                                                        int32_t namedBarId) {
  reorganizeInSmemAndStoreToDstMemImpl_<NumRows,
                                        NumCols,
                                        0,
                                        0,
                                        DtypeO,
                                        NumRegs,
                                        false,
                                        false,
                                        MapRowToHeadTokenIdx>(smemPtrO,
                                                              dstMemPtrO,
                                                              nullptr,
                                                              arrayO,
                                                              dstMemRowStride,
                                                              numValidTokens,
                                                              numValidTokens,
                                                              numHeadsQ,
                                                              numHeadsQPerKv,
                                                              numWarpGrpThreads,
                                                              warpGrpThreadIdx,
                                                              namedBarId);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t NumRows,
          int32_t NumCols,
          bool MapRowToHeadTokenIdx,
          class DtypeO,
          int NumRegs,
          typename Barrier>
inline __device__ void reorganizeInSmemAndStoreToDstMem(DtypeO* smemPtrO,
                                                        void* dstMemPtrO,
                                                        Barrier* transactionBarrier,
                                                        uint32_t (&arrayO)[NumRegs],
                                                        int32_t dstMemRowStride,
                                                        int32_t numValidTokens,
                                                        int32_t numReductionRowsPerCta,
                                                        int32_t numHeadsQ,
                                                        trtllm::dev::fast_mod_div numHeadsQPerKv,
                                                        int32_t numWarpGrpThreads,
                                                        int32_t warpGrpThreadIdx,
                                                        int32_t namedBarId) {
  static_assert(!MapRowToHeadTokenIdx, "MapRowToHeadTokenIdx should be false for this function.");
  reorganizeInSmemAndStoreToDstMemImpl_<NumRows,
                                        NumCols,
                                        0,
                                        0,
                                        DtypeO,
                                        NumRegs,
                                        false,
                                        true,
                                        MapRowToHeadTokenIdx>(smemPtrO,
                                                              dstMemPtrO,
                                                              transactionBarrier->getBarrierPtr(),
                                                              arrayO,
                                                              dstMemRowStride,
                                                              numValidTokens,
                                                              numReductionRowsPerCta,
                                                              numHeadsQ,
                                                              numHeadsQPerKv,
                                                              numWarpGrpThreads,
                                                              warpGrpThreadIdx,
                                                              namedBarId);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t NumRows,
          int32_t NumCols,
          int32_t RowStrideSf,
          int32_t NumSfPerHead,
          bool MapRowToHeadTokenIdx,
          class DtypeO,
          int NumRegs>
inline __device__ void reorganizeInSmemAndStoreToDstMemAsE2m1(
  DtypeO* smemPtrO,
  void* dstMemPtrO,
  uint32_t (&arrayO)[NumRegs],
  int32_t dstMemRowStride,
  int32_t numValidTokens,
  int32_t numHeadsQ,
  trtllm::dev::fast_mod_div numHeadsQPerKv,
  int32_t numWarpGrpThreads,
  int32_t warpGrpThreadIdx,
  int32_t namedBarId,
  cutlass::float_e4m3_t* dstMemPtrOSf,
  float sfScale,
  int32_t sfBaseRowIdx = 0) {

  reorganizeInSmemAndStoreToDstMemImpl_<NumRows,
                                        NumCols,
                                        RowStrideSf,
                                        NumSfPerHead,
                                        DtypeO,
                                        NumRegs,
                                        true,
                                        false,
                                        MapRowToHeadTokenIdx>(smemPtrO,
                                                              dstMemPtrO,
                                                              static_cast<uint64_t*>(nullptr),
                                                              arrayO,
                                                              dstMemRowStride,
                                                              numValidTokens,
                                                              numValidTokens,
                                                              numHeadsQ,
                                                              numHeadsQPerKv,
                                                              numWarpGrpThreads,
                                                              warpGrpThreadIdx,
                                                              namedBarId,
                                                              dstMemPtrOSf,
                                                              sfScale,
                                                              sfBaseRowIdx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace dev
} // namespace trtllm
