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

namespace trtllm {
namespace dev {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int BitsPerElementB, int N> struct Rows_per_xor_pattern_ampere_b {
  // The size in bits.
  enum { NInBits = N * BitsPerElementB };
  // The number of rows.
  enum { Value = NInBits <= 256 ? 2 : (NInBits <= 512 ? 4 : 8) };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ void ldsmt(uint4& dst, uint32_t ptr) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 730
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(dst.x), "=r"(dst.y), "=r"(dst.z), "=r"(dst.w)
               : "r"(ptr));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ void swizzleRows(uint32_t& a, uint32_t& b, uint32_t c, uint32_t d) {
  asm volatile("prmt.b32 %0, %1, %2, 0x6420;\n" : "=r"(a) : "r"(c), "r"(d));
  asm volatile("prmt.b32 %0, %1, %2, 0x7531;\n" : "=r"(b) : "r"(c), "r"(d));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ void stsm(uint32_t ptr, const uint4& src) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  asm volatile("stmatrix.sync.aligned.m8n8.x4.shared.b16 [%0], {%1, %2, %3, %4};\n" ::"r"(ptr),
               "r"(src.x),
               "r"(src.y),
               "r"(src.z),
               "r"(src.w));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Pass these for BMM2:
// TileSizeN - HeadDim
// TileSizeK - TileSizeKv
template <int TileSizeN, int TileSizeK, int ByesPerElementB = 1, int WarpCount = 4, int UnrollN = 1>
struct SmemTransposer {

  static_assert(TileSizeK % 128 == 0);
  static_assert(4 % WarpCount == 0);

  enum { PassCount = 4 / WarpCount };

  enum {
    WarpsM = 4,
    WarpsN = 1,
    WarpsK = 1,
  };

  enum {
    Warps4x1x1 = (WarpsM == 4 && WarpsN == 1 && WarpsK == 1),
    Warps4x1x2 = (WarpsM == 4 && WarpsN == 1 && WarpsK == 2),
  };

  enum { BytesPerLds = 16 };
  enum { BytesPerRow = 128 };

  // D=64 and 4 warps.
  // Per warp we load 32 rows x 16 columns with LDSM.Tx4, 128 rows per CTA.
  enum { S = TileSizeK >= 128 ? 128 : TileSizeK }; // The sequence length.
  enum { D = TileSizeN >= 128 ? 128 : TileSizeN }; // The head dimension.

  // static_assert(S % 128 == 0);
  static_assert(Warps4x1x1 || Warps4x1x2);
  static_assert(D % (BytesPerLds * WarpsK) == 0);

  enum { RowsPerLdsmPerCtaWithoutPacking = 128 }; // LDSMx4
  enum { RowPacking = BytesPerRow / (D * ByesPerElementB) };
  enum { RowsPerLdsmPerCta = RowsPerLdsmPerCtaWithoutPacking / RowPacking };
  enum { RowsPerXorPattern = Rows_per_xor_pattern_ampere_b<ByesPerElementB * 8, S>::Value };
  static_assert(RowsPerXorPattern == 8);

  // The number of loads in K dimension.
  enum { K = S / RowsPerLdsmPerCtaWithoutPacking };
  // static_assert(K * RowsPerLdsmPerCtaWithoutPacking == S);
  // static_assert(K == 3);
  //  The number of loads in the D dimension.
  enum { N = D / (BytesPerLds * WarpsK) }; // 16 bytes per load
  static_assert(N * BytesPerLds * WarpsK == D);

  uint4 mRegs[UnrollN][K];

  uint32_t mReadOffsets[PassCount];
  uint32_t mWriteOffsets[PassCount];

  inline __device__ SmemTransposer(int threadIdx) {

    for (int pass = 0; pass < PassCount; ++pass) {
      int tidx = threadIdx + pass * WarpCount * 32;
      int readRow, readCol;

      if (Warps4x1x1 && N == 8) { // D=128, 1 warp  in N
        readRow = (tidx & 0x7f);
        readCol = (tidx & 0x07);
      } else if (Warps4x1x1 && N == 4) { // D=64, 1 warp  in N
        readRow = (tidx & 0xe0) / 2 + (tidx & 0x1e) / 2;
        readCol = (tidx & 0x01) * 4 + (tidx & 0x06) / 2;
      } else if (Warps4x1x1 && N == 2) { // D=32, 1 warp  in N
        readRow = (tidx & 0x60) / 4 + (tidx & 0x1c) / 4;
        readCol = (tidx & 0x03) * 2;
        readCol ^= (readRow & 0x01);
      } else if (Warps4x1x2 && N == 4) { // D=128, 2 warps in N
        readRow = (tidx & 0x7f);
        readCol = (tidx & 0x07);
        // For two warpgroups we do two steps in N at once.
        readCol ^= (tidx & 0x80) / 128;
      } else if (Warps4x1x2 && N == 2) { // D=64, 2 warps in N
        readRow = (tidx & 0x60) / 2 + (tidx & 0x1e) / 2;
        readCol = (tidx & 0x01) * 4 + (tidx & 0x06) / 2;
        // For two warpgroups we do two steps in N at once.
        readCol ^= (tidx & 0x80) / 128;
      } else if (Warps4x1x2 && N == 1) { // D=32, 2 warps  in N
        readRow = (tidx & 0x60) / 4 + (tidx & 0x1c) / 4;
        readCol = (tidx & 0x03) * 2;
        readCol ^= (readRow & 0x01);
        // For two warpgroups we do two steps in N at once.
        readCol ^= (tidx & 0x80) / 128;
      } else {
        assert(false);
      }

      mReadOffsets[pass] = readRow * BytesPerRow + readCol * BytesPerLds;

      int writeRow, writeCol;
      if (Warps4x1x1) { // swizzle_128byte
        writeRow = (tidx & 0x10) / 2 + (tidx & 0x07);
        writeCol = (tidx & 0x60) / 16 + (tidx & 0x08) / 8;
      } else if (Warps4x1x2) {
        // Same as above, with second warp group writing next 16 rows.
        writeRow = (tidx & 0x80) / 8 + (tidx & 0x10) / 2 + (tidx & 0x07);
        writeCol = (tidx & 0x60) / 16 + (tidx & 0x08) / 8;
      } else {
        assert(false);
      }

      writeCol ^= (writeRow & 0x07);

      mWriteOffsets[pass] = writeRow * BytesPerRow + writeCol * BytesPerLds;
    }
  }

  inline __device__ void transpose(const void* smemSrcPtr, void* smemDstPtr) {
    uint32_t smemSrc = static_cast<std::uint32_t>(__cvta_generic_to_shared(smemSrcPtr));
    uint32_t smemDst = static_cast<std::uint32_t>(__cvta_generic_to_shared(smemDstPtr));
#pragma unroll
    for (int pass = 0; pass < PassCount; ++pass) {
      uint32_t smemReadLoc;
      uint32_t smemWriteLoc;
#pragma unroll
      for (int n_begin = 0; n_begin < N; n_begin += UnrollN) {
        transposeLdMatrix(n_begin, smemSrc, smemReadLoc, mReadOffsets[pass]);
        transposeStMatrix(n_begin, smemDst, smemWriteLoc, mWriteOffsets[pass]);
      }
    }
  }

  inline __device__ void transposeLdMatrix(int n_begin,
                                           uint32_t smemSrc,
                                           uint32_t& smemReadLoc,
                                           uint32_t readOffset) {
    static_assert(N % UnrollN == 0, "");

    uint4 tmp[UnrollN][K];
    if (n_begin == 0) {
      smemReadLoc = smemSrc + readOffset;
    }

#pragma unroll
    for (int ni = n_begin; ni < n_begin + UnrollN; ni++) {
      const int nii = ni - n_begin;
#pragma unroll
      for (int ki = 0; ki < K; ki++) { // 2
        ldsmt(tmp[nii][ki], smemReadLoc + ki * RowsPerLdsmPerCta * BytesPerRow);
      }

      if (Warps4x1x1 && N == 4) { // D=64, 1 warp  in N
        smemReadLoc ^= (ni % 2 == 0 ? 1 : 3) * 16;
      } else if (Warps4x1x1 && N == 2) { // D=32, 1 warp  in N
        smemReadLoc ^= 16;
      } else if (Warps4x1x2 && N == 2) { // D=64, 2 warps in N
        smemReadLoc ^= 32;
      } else if (Warps4x1x2 && N == 4) { // D=128, 2 warps in N
        smemReadLoc ^= (ni % 2 == 0 ? 1 : 3) * 32;
      } else if (Warps4x1x1 && N == 8) { // D=128, 1 warp  in N
        smemReadLoc ^= ((ni % 4 == 3) ? 7 : (ni % 2 == 1 ? 3 : 1)) * 16;
      } else if (N != 1) {
        assert(false);
      }
    }

#pragma unroll
    for (int ni = n_begin; ni < n_begin + UnrollN; ni++) {
      const int nii = ni - n_begin;
#pragma unroll
      for (int ki = 0; ki < K; ki++) {
        swizzleRows(mRegs[nii][ki].x, mRegs[nii][ki].z, tmp[nii][ki].x, tmp[nii][ki].y); // PRMT
                                                                                         // 0+1
        swizzleRows(mRegs[nii][ki].y, mRegs[nii][ki].w, tmp[nii][ki].z, tmp[nii][ki].w); // PRMT
                                                                                         // 2+3
      }
    }
  }

  inline __device__ void transposeStMatrix(int n_begin,
                                           uint32_t smemDst,
                                           uint32_t& smemWriteLoc,
                                           uint32_t writeOffset) {

    // After LDSM.Tx4 registers hold 2x2 elts:
    // [00, 01]
    // [10, 11]
    // With row offsets
    // x: + 0
    // y: + 8
    // z: +16 (g)
    // w: +24 (o)
    //
    // After PRMT 0, the :
    // [00, 01] [80, 81] => x: [00, 10, 80, 90], i.e. col 0
    // [10, 11] [90, 91] => z: [01, 11, 81, 91], i.e. col 1
    //
    // [g0, g1] [o0, o1] => y: [g0, h0, o0, p0], i.e. col 0
    // [h0, h1] [p0, p1] => w: [g1, h1, o1, p1], i.e. col 1
    //
    // Therefore, when looking at the transpose, quad q holds cols 2 * q + [0, 1], i.e.
    // - quad 0 holds cols 0, 1
    // - quad 1 holds cols 2, 3
    // - etc.
    //
    // This fits with the accumulator layout, since N strides in steps of 8 per thread.

    if (n_begin == 0) {
      smemWriteLoc = smemDst + writeOffset;
    }

#pragma unroll
    for (int ni = n_begin; ni < n_begin + UnrollN; ni++) {
      const int nii = ni - n_begin;
#pragma unroll
      for (int ki = 0; ki < K; ki++) {
        stsm(smemWriteLoc + ki * BytesPerRow * D, mRegs[nii][ki]);
      }
      if (Warps4x1x1) { // D=64, 1 warp in N.
        smemWriteLoc += 16 * BytesPerRow;
      } else if (Warps4x1x2) { // D=64, 2 warps in N.
        smemWriteLoc += 32 * BytesPerRow;
      } else {
        assert(false);
      }
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace dev
} // namespace trtllm
