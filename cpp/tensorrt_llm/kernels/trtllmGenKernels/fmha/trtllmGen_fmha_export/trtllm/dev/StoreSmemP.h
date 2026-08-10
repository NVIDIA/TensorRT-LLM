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

#include <cutlass/numeric_conversion.h>
#include <cuda_fp8.h>
#include <algorithm>

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////

// STSM 8b functions to store transposed 16 (rows) x 8 (cols) to shared memory.

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void storeTrans8bMatrix(uint32_t* smemDst, uint32_t const (&src)[1]) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
  asm volatile("stmatrix.sync.aligned.m16n8.x1.trans.shared.b8 [%0], {%1};\n" ::"l"(smemDst),
               "r"(src[0]));
#else
  static_assert(false, "Not implemented.");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void storeTrans8bMatrix(uint32_t* smemDst, uint32_t const (&src)[2]) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
  asm volatile("stmatrix.sync.aligned.m16n8.x2.trans.shared.b8 [%0], {%1, %2};\n" ::"l"(smemDst),
               "r"(src[0]),
               "r"(src[1]));
#else
  assert(false && "Not implemented.");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void storeTrans8bMatrix(uint32_t* smemDst, uint32_t const (&src)[4]) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
  asm volatile(
    "stmatrix.sync.aligned.m16n8.x4.trans.shared.b8 [%0], {%1, %2, %3, %4};\n" ::"l"(smemDst),
    "r"(src[0]),
    "r"(src[1]),
    "r"(src[2]),
    "r"(src[3]));
#else
  assert(false && "Not implemented.");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// This function stores 64/128 rows x 8/16 cols to shared memory with transpose.
//
// The function assumes that the data is distributed amongst the threads after loading from TMEM
// using the LDTM.16dp256bit instructions.
//
// Distribution of threads:
//
// Row ..0: ..0 ..1 ..2 ..3
// Row ..1: ..4 ..5 ..6 ..7
// Row ..2: ..8 ..9 .10 .11
// Row ..3: .12 .13 .14 .15
// Row ..4: .16 .17 .18 .19
// Row ...: ...
// Row ..7: .28 .29 .30 .31
// Row ..8: .32 .33 .34 .35
// Row ...: ...
// Row .31: 124 125 126 127
//
// Distribution of values (even if it does not matter too much). The indices in the following dia-
// gram are the indices of the thread owners.
//
// Row ..0: ..0 ..0 ..1 ..1 ..2 ..2 ..3 ..3
// Row ..1: ..4 ..4 ..5 ..5 ..6 ..6 ..7 ..7
// Row ..2: ..8 ..8 ..9 ..9 .10 .10 .11 .11
// Row ..3: .12 .12 .13 .13 .14 .14 .15 .15
// Row ..4: .16 .16 .17 .17 .18 .18 .19 .19
// Row ...: ... ...
// Row ..7: .28 .28 .29 .29 .30 .30 .31 .31
// Row ..8: ..0 ..0 ..1 ..1 ..2 ..2 ..3 ..3
// Row ..9: ..4 ..4 ..5 ..5 ..6 ..6 ..7 ..7
// Row ...: ... ...
// Row .16: ..0 ..0 ..1 ..1 ..2 ..2 ..3 ..3
// Row .17: ..4 ..4 ..5 ..5 ..6 ..6 ..7 ..7
// Row ...: ... ...
// Row .24: ..0 ..0 ..1 ..1 ..2 ..2 ..3 ..3
// Row .25: ..4 ..4 ..5 ..5 ..6 ..6 ..7 ..7
// Row ...: ... ...
// Row .32: .32 .32 .33 .33 .34 .34 .35 .35
// Row ...: ... ...
// Row .64: .64 .64 .65 .65 .66 .66 .67 .67
// Row ...: ... ...
// Row .96: .96 .96 .97 .97 .98 .98 .99 .99
// Row ...: ... ...
// Row 127: 124 124 125 125 126 126 127 127
//
// Distribution of values in each thread. Every 4 contiguous values will be packed as one uint32_t.

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t NumRows, int32_t NumCols, int NumRegs>
inline __device__ void storeTransposedSmem8b(cutlass::float_e4m3_t* smemPtr,
                                             uint32_t const (&regsP)[NumRegs],
                                             int32_t warpGrpThreadIdx) {

  // Make sure NumRows is power of 2 and greater than or equal to 8.
  static_assert(NumRows >= 8 && (NumRows & (NumRows - 1)) == 0, "Not implemented.");
  // Make sure we have 64 or 128 cols after transposition.
  static_assert(NumCols == 64 || NumCols == 128, "Not implemented.");

  // The number of warps per warp-group.
  int32_t constexpr NumWarps{4};
  // The number of bytes per row after transposing.
  int32_t constexpr NumBytesPerRow{NumCols};
  // The number of rows packed in 128B.
  int32_t constexpr NumRowsPer128B{128 / NumBytesPerRow};
  // The number of 16B segments per warp and per row after transposing.
  int32_t constexpr NumSegsPerWarpPerRow{NumBytesPerRow / 16 / NumWarps};

  // The warp in the warp-group.
  int32_t const warpIdx{warpGrpThreadIdx / 32};
  // The lane in the warp-group.
  int32_t const laneIdx{warpGrpThreadIdx % 32};

  // Each "repeat" of the STSM instruction stores 8x segments of 16B. For the 1st repeat, the
  // threads 0-7 bring the addresses of the first 8 segments (one address per thread). For the 2nd
  // repeat, the threads 8-15 are responsible for the addresses of the 8 segments. And so on, ...

  // The number of STSM instructions where each STSM instruction writes at most 16 x 128 fp8
  // elements.
  int32_t constexpr NumStsmInsts{std::max(NumRows * NumCols / (16 * 128), 1)};
  // The number of registers per STSM instruction.
  static_assert(NumRegs % NumStsmInsts == 0, "Not supported");
  int32_t constexpr NumRegsPerStsm{NumRegs / NumStsmInsts};
  // The number of STSM instructions per row where each STSM instruction processes at most 32 rows.
  int32_t constexpr NumStsmPerRow{std::max(NumRows / 32, 1)};
  // The number of transposed STSM.MT168 matrices per col. Each transposed matrix has 8 rows.
  int32_t constexpr NumMtxPerCol{NumRows / 8};
  // Each STSM.MT168 matrix is handled by 8 threads.
  int32_t const mtxIdx{laneIdx / 8};
  // The position of the matrix in the column. The matrices are stored column-major.
  int32_t const mtxRowIdx{mtxIdx % NumMtxPerCol};
  // The position of the matrix in the row.
  int32_t const mtxColIdx{mtxIdx / NumMtxPerCol};
  // After transposition, each warp handles all the elements of its "assigned" columns.
  int32_t const thrRowIdx{laneIdx % 8};
// Iterate over the STSM instructions.
// The STSM instruction will process row matrices first and then column matrices in order
// to align with ldtm.16dp256bit instructions.
// When NumRows = 16 NumCols = 128, it will process 2 row x 2 column matrices with one instruction.
// When NumRows >= 32, it will process 4 row x 1 column matrices with one instruction.
#pragma unroll
  for (int32_t stsmIdx = 0; stsmIdx < NumStsmInsts; ++stsmIdx) {

    // The STSM instruction will process row matrices first and then column matrices
    int32_t const stsmRowIdx = stsmIdx % NumStsmPerRow;
    int32_t const stsmColIdx = stsmIdx / NumStsmPerRow;

    // The XOR mask.
    int32_t const xorMask = thrRowIdx / NumRowsPer128B;
    // Swizzle the offsets based on the swizzle mode.
    int32_t const segColIdx = (warpIdx * NumSegsPerWarpPerRow + mtxColIdx + stsmColIdx) ^ xorMask;

    // Assemble the final offset after swizzling.
    int smemOffset =
      (mtxRowIdx * 8 + thrRowIdx + stsmRowIdx * 32) * NumBytesPerRow + segColIdx * 16;
    // Call the STSM instruction.
    storeTrans8bMatrix(
      reinterpret_cast<uint32_t*>(smemPtr + smemOffset),
      reinterpret_cast<uint32_t const(&)[NumStsmInsts][NumRegsPerStsm]>(regsP)[stsmIdx]);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// STSM 16b functions to store transposed 8 (rows) x 8 (cols) to shared memory.

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void storeTrans16bMatrix(uint32_t* smemDst, uint32_t const (&src)[2]) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile("stmatrix.sync.aligned.x2.trans.m8n8.shared.b16 [%0], {%1, %2};\n" ::"l"(smemDst),
               "r"(src[0]),
               "r"(src[1]));
#else
  static_assert(false, "Not implemented.");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ void storeTrans16bMatrix(uint32_t* smemDst, uint32_t const (&src)[4]) {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
  asm volatile(
    "stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"l"(smemDst),
    "r"(src[0]),
    "r"(src[1]),
    "r"(src[2]),
    "r"(src[3]));
#else
  static_assert(false, "Not implemented.");
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// This function stores 64 rows x 8/16 cols to shared memory with transpose.
//
// The function assumes that the data is distributed amongst the threads after loading from TMEM
// using the LDTM.16dp256bit instructions.

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t NumRows, class Dtype, int NumRegs>
inline __device__ void storeTransposedSmem64x16b(Dtype* smemPtr,
                                                 uint32_t const (&regsP)[NumRegs],
                                                 int32_t warpGrpThreadIdx) {

  // Make sure NumRows is power of 2 and greater than or equal to 8.
  static_assert(NumRows >= 8 && (NumRows & (NumRows - 1)) == 0, "Not implemented.");


  // The number of cols after transposing.
  int32_t constexpr NumCols{64};

  // The number of warps per warp-group.
  int32_t constexpr NumWarps{4};
  // The number of bytes per row after transposing.
  int32_t constexpr NumBytesPerRow{NumCols * sizeof(uint16_t) /*64*2 = 128B*/};

  // The warp in the warp-group.
  int32_t const warpIdx{warpGrpThreadIdx / 32};
  // The lane in the warp-group.
  int32_t const laneIdx{warpGrpThreadIdx % 32};

  // After each repeat of LDTM.16dp256bit (when copying data from TMEM to registers in TmemS), each
  // thread owns two consecutive columns in Reg0 and Reg1 from the same row and two consecutive
  // columns in Reg2 and Reg3 from another row (there are 8 rows between those two rows). For exam-
  // ple threads 0, 1, 2 and 3 resp. holds columns 0-1, 2-3, 4-5 and 6-7 from rows 0 and 8. Threads
  // 4-7 have the same columns from rows 1 and 9. See LDTM documentation for details. The values
  // are FP32 values.
  //
  // In TmemP, the 4x FP32 values (per LDTM repeat) are converted to FP16 and packed into 2 regis-
  // ters (Reg0 and Reg1 are converted and packed into a first register - Reg2 and Reg3 go to
  // another one). The 1st two STSM repeats will produce 16 columns of the same row in SMEM after
  // transposition. Since there are 64 columns and 4 warps. Each warp writes 2 matrices per row.
  int32_t constexpr NumMtxPerRow{2};

  // The number of STSM instructions, where each STSM instruction can at most write 16 rows.
  int32_t constexpr NumStsmInsts{std::max(NumRows / 16, 1)};
  // The number of registers per STSM instruction.
  static_assert(NumRegs % NumStsmInsts == 0, "Not supported");
  int32_t constexpr NumRegsPerStsm{NumRegs / NumStsmInsts};

  // Each STSM.MT88 matrix is handled by 8 threads.
  int32_t const mtxIdx{laneIdx / 8};
  // The position of the matrix in the col (after transposition). Matrices are stored row-major.
  int32_t const mtxRowIdx{mtxIdx / NumMtxPerRow};
  // The position of the matrix in the row.
  int32_t const mtxColIdx{mtxIdx % NumMtxPerRow};

  // After transposition, each warp handles all the elements of its "assigned" columns.
  int32_t const thrRowIdx{laneIdx % 8};

  // The XOR mask (for SWIZZLE_128B).
  int32_t const xorMask = thrRowIdx;
  // Swizzle the offsets based on the swizzle mode.
  int32_t const segColIdx = (warpIdx * NumMtxPerRow + mtxColIdx) ^ xorMask;

  // The base smem pointer in bytes.
  char* const smemPtrBytes{reinterpret_cast<char*>(smemPtr)};

// Iterate over the STSM instructions where each STSM instruction writes at most 16 rows.
#pragma unroll
  for (int32_t stsmIdx = 0; stsmIdx < NumStsmInsts; ++stsmIdx) {
    // Assemble the final offset after swizzling.
    int smemOffset = (mtxRowIdx * 8 + thrRowIdx + stsmIdx * 16) * NumBytesPerRow + segColIdx * 16;
    // Call the STSM instruction.
    auto regsPArray = reinterpret_cast<uint32_t const(*)[NumRegsPerStsm]>(&regsP[0]);
    storeTrans16bMatrix(reinterpret_cast<uint32_t*>(smemPtrBytes + smemOffset),
                        regsPArray[stsmIdx]);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// This function stores 128 rows x (8 * numLdtmReps) cols to shared memory with transpose.
//
// The function assumes that the data is distributed amongst the threads after loading from TMEM
// using the LDTM.16dp256bit instructions.

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int32_t NumRows, class Dtype, int NumRegs>
inline __device__ void storeTransposedSmem128x16b(Dtype* smemPtr,
                                                  uint32_t const (&regsP)[NumRegs],
                                                  int32_t warpGrpThreadIdx) {

  // Make sure NumRows is power of 2 and greater than or equal to 8.
  static_assert(NumRows >= 8 && (NumRows & (NumRows - 1)) == 0, "Not implemented.");


  // The number of cols after transposing.
  int32_t constexpr NumCols{128};

  // The warp in the warp-group.
  int32_t const warpIdx{warpGrpThreadIdx / 32};
  // The lane in the warp-group.
  int32_t const laneIdx{warpGrpThreadIdx % 32};

  // Since a transposed row has 128 columns of 16-bit elements, it occupies 256B. That's more than
  // the 128B per row limit per slice in SMEM. So, a transposed row spans two slices. For that
  // reason, we decompose the warp-group into 2 groups of 2 warps. Warps 0-1 write to the 1st slice
  // and warps 2-3 write to the 2nd slice.

  // The slice written by the warp.
  int32_t const sliceIdx{warpIdx / 2};
  // The warp index in the slice.
  int32_t const warpIdxInSlice{warpIdx % 2};

  // After each repeat of LDTM.16dp256bit (when copying data from TMEM to registers in TmemS), each
  // thread owns two consecutive columns in Reg0 and Reg1 from the same row and two consecutive
  // columns in Reg2 and Reg3 from another row (there are 8 rows between those two rows). For exam-
  // ple threads 0, 1, 2 and 3 resp. holds columns 0-1, 2-3, 4-5 and 6-7 from rows 0 and 8. Threads
  // 4-7 have the same columns from rows 1 and 9. See LDTM documentation for details. The values
  // are FP32 values.
  //
  // In TmemP, the 4x FP32 values (per LDTM repeat) are converted to FP16 and packed into 2 regis-
  // ters (Reg0 and Reg1 are converted and packed into a first register - Reg2 and Reg3 go to
  // another one). The 1st two STSM repeats will produce 16 columns of the same row in SMEM after
  // transposition.
  //
  // Unfortunately, to make things harder, each LDTM repeat is in row-dimension of TMEM. It means
  // that when we have more than one LDTM repeat, the next block of 4 registers from LDTM corres-
  // ponds to another set of rows in the transposed matrix in SMEM (columns 8-15 in TMEM map to
  // rows 8-15 in SMEM after the transposition). It means that if we have 2 LDTM repeats, each
  // STSM will write two matrices per column of SMEM (i.e. 16 rows).
  // It explains why the below code is a little bit more complicated than we would have liked.

  // The number of LDTM repeats, and each repeat needs one STSM.MT88.x4 instruction.
  int32_t constexpr NumLdtmReps{NumRows / 8};
  // The number of rows must be a multiple of 8.
  static_assert(NumRows % 8 == 0, "Not supported");

  //
  // Each STSM instruction can write 4 matrices and it will be reshaped differently based on the
  // number of LDTM repeats.
  //

  // The number of matrices per row in SMEM per STSM (i.e. block of 4 registers).
  int32_t constexpr NumMtxPerRowPerStsm{NumLdtmReps == 1 ? 4 : 2};
  // The number of matrices per col in SMEM per STSM (i.e. block of 4 registers).
  int32_t constexpr NumMtxPerColPerStsm{NumLdtmReps == 1 ? 1 : 2};
  // Make sure the number of row*col is 4.
  static_assert(NumMtxPerRowPerStsm * NumMtxPerColPerStsm == 4);

  // The number of STSM instructions per row after transposition (32 columns per warp, i.e. 4
  // matrices).
  int32_t constexpr NumStsmPerRow{4 / NumMtxPerRowPerStsm};
  // The number of STSM instructions per col after transposition.
  int32_t constexpr NumStsmPerCol{std::max(NumLdtmReps / NumMtxPerColPerStsm, 1)};

  // The number of registers per STSM instruction.
  static_assert(NumRegs % (NumStsmPerRow * NumStsmPerCol) == 0, "Not supported");
  int32_t constexpr NumRegsPerStsm{NumRegs / (NumStsmPerRow * NumStsmPerCol)};

  // Each STSM.MT88 matrix is handled by 8 threads.
  int32_t const mtxIdx{laneIdx / 8};
  // The position of the matrix in the col (after transposition).
  int32_t const mtxRowIdx{mtxIdx / NumMtxPerRowPerStsm};
  // The position of the matrix in the row. Each warp stores 4 matrices per row (4x16 cols/warp).
  int32_t const mtxColIdx{(warpIdxInSlice * 4) + (mtxIdx % NumMtxPerRowPerStsm)};
  // After transposition, each warp handles all the elements of its "assigned" columns.
  int32_t const thrRowIdx{laneIdx % 8};

  // The number of bytes per row.
  int32_t constexpr NumBytesPerRow{128};
  // The offset to the slice in SMEM.
  int32_t const sliceOffset{sliceIdx * NumRows * NumBytesPerRow};
  // The offset to the row in the slice in SMEM.
  int32_t const rowOffset{(mtxRowIdx * 8 + thrRowIdx) * NumBytesPerRow};

// Each STSM corresponds to one LDTM.
#pragma unroll
  for (int32_t ii = 0; ii < NumStsmPerRow; ++ii) {

#pragma unroll
    for (int32_t jj = 0; jj < NumStsmPerCol; ++jj) {

      // The XOR mask (for SWIZZLE_128B).
      int32_t const xorMask = thrRowIdx;
      // Swizzle the offsets based on the swizzle mode.
      int32_t const segColIdx = (mtxColIdx + ii * NumMtxPerRowPerStsm) ^ xorMask;

      // The row offset of the current STSM instruction.
      int32_t stsmRowOffset = jj * NumMtxPerColPerStsm * 8 * NumBytesPerRow;

      // The base smem pointer in char.
      char* const smemPtrBytes{reinterpret_cast<char*>(smemPtr)};
      // Assemble the final offset after swizzling.
      int smemOffset{sliceOffset + rowOffset + stsmRowOffset + segColIdx * 16};
      // The dst pointers as UInt32.
      auto dstU32 = reinterpret_cast<uint32_t*>(smemPtrBytes + smemOffset);
      // Call the STSM instruction.
      auto regsPArray = reinterpret_cast<uint32_t const(*)[NumRegsPerStsm]>(&regsP[0]);
      storeTrans16bMatrix(dstU32, regsPArray[ii * NumStsmPerCol + jj]);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// P is stored in shared memory. Within a warp-group (4 warps), each warp stores 32 rows. For each
// row, the elements are packed into NumU32s uint32s. When the total size per row exceeds 128B,
// multiple SMEM rows are used to store the elements. The usual swizzling is used for storing the
// elements in SMEM.
template <int32_t NumKPartitionsMmaPv, int32_t NumU32s>
inline __device__ void storeWarpGrp2x2SmemP(uint32_t* smemDst,
                                            uint32_t const (&src)[NumU32s],
                                            int mWarpGrpThreadIdx) {
  //
  // The 2x2 warp-group layout.
  //
  // warp0   warp2
  // warp1   warp3
  //
  // Note that if we split the K dimension of P into 2 partitions (NumKPartitionsMmaPv = 2), warp2
  // and warp3 naturally store values to the 2nd partition of P. Each partition will swizzle its own
  // values based on the size of its K dimension. This is aligned with how we generate the BMM2
  // codes in TmemO.h.

  // NumKPartitionsMmaPv must be 1 or 2.
  static_assert(NumKPartitionsMmaPv == 1 || NumKPartitionsMmaPv == 2,
                "NumKPartitionsMmaPv must be 1 or 2");

  // The warpColIdx.
  int32_t const warpColIdx{mWarpGrpThreadIdx / 64};
  // The warpRowIdx.
  int32_t const warpRowIdx{mWarpGrpThreadIdx % 64};
  // The numUint32s per row.
  int32_t const numUint32sPerRow{std::min(NumU32s * (2 / NumKPartitionsMmaPv), 32)};

  // The base offset of the shared memory.
  // There is at most 128B (32 uint32_t) per row considering the swizzling.
  int32_t smemBaseOffset{warpRowIdx * numUint32sPerRow};
  // If we split the K dimension of P into 2 partitions, we need to adjust the base offset based on
  // the partition.
  if constexpr (NumKPartitionsMmaPv == 2) {
    smemBaseOffset += warpColIdx * 64 * numUint32sPerRow;
  }

  // The number of uint4 elements in the source.
  static_assert(NumU32s % 4 == 0, "The number of elements must be a multiple of 4");
  int32_t const numUint4Elts{NumU32s / 4};

  // The xor factor.
  int32_t const xorFactor = std::min(8, numUint32sPerRow / 4);
  // The number of packed rows.
  int32_t const numPackedRows{8 / xorFactor};

#pragma unroll
  for (int32_t ii = 0; ii < numUint4Elts; ++ii) {
    // The smemColIdx.
    int32_t smemColIdx{ii};
    // If we don't split the K dimension of P into 2 partitions, we need to adjust the block index
    // based on the warpColIdx.
    if constexpr (NumKPartitionsMmaPv == 1) {
      smemColIdx += warpColIdx * numUint4Elts;
    }
    // The smemBlock idx, where each block only contains 128B (32 uint32_t) in the leading
    // dimension.
    int32_t const smemBlkIdx{smemColIdx / 8};
    // Swizzle the column index.
    int32_t swizzledColIdx = (smemColIdx % 8) ^ ((mWarpGrpThreadIdx % 8) / numPackedRows);
    // The smem offset in uint32_t.
    int32_t const smemOffset{smemBaseOffset + smemBlkIdx * 64 * 32 + swizzledColIdx * 4};
    // Store the data.
    *reinterpret_cast<uint4*>(smemDst + smemOffset) = reinterpret_cast<uint4 const*>(src)[ii];
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace dev
} // namespace trtllm
