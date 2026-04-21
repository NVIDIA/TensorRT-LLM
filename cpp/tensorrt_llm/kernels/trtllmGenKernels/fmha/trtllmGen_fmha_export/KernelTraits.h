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

#ifdef TLLM_FMHA_TRTLLM_COMPAT
#include "trtllmGenExportCompat.h"
#else
#include <trtllm/gen/GenCtx.h>
#include "KernelConfigBase.h"
#include "KernelParams.h"
#include <trtllm/gen/TmemTile.h>
#include <trtllm/gen/Expr.h>
#include <trtllm/gen/Kernel.h>
#endif // TLLM_FMHA_TRTLLM_COMPAT
#include <nlohmann/json.hpp>
#include <cassert>
#include <numeric>
#include <sstream>

namespace fmha {

struct KernelConfig : public KernelConfigBase {
  // The data type of softmax.
  tg::Dtype mDtypeSoftmax;
  // The maximum attention head size for K and V.
  int32_t mMaxHeadDimKv;
  // The number of elements in the packed uint32_t element for Q.
  int32_t mNumEltsPerUInt32Q;
  // The number of elements in the packed uint32_t element for K/V.
  int32_t mNumEltsPerUInt32Kv;
  // The number of elements in the packed uint32_t element for O.
  int32_t mNumEltsPerUInt32O;
  // The number of stages of the Epilogue buffer.
  int32_t mNumStagesEpilogue;
  // The number of stages of the persistent work id pipeline.
  int32_t mNumStagesWorkId;
  // The number of stages of the KV shared memory buffer.
  int32_t mNumStagesKv;
  // The number of stages of the K shared memory buffer.
  int32_t mNumStagesK;
  // The number of stages of the V shared memory buffer.
  int32_t mNumStagesV;
  // The number of stages of the pageOffsetsKv shared memory buffer.
  int32_t mNumStagesPageOffsetsKv;
  // The number of stages of the Q shared memory buffer.
  int32_t mNumStagesQ;
  // The number of stages of the transformed KV shared memory buffer.
  int32_t mNumStagesTransformedKv;
  // The number of stages of the transposed V shared memory buffer.
  int32_t mNumStagesTransposedV;
  // The number of tmem cols we will allocate once.
  int32_t mNumTmemCols;
  // Does it support pagedKv tensors?
  bool mSupportsPagedKv;
  // The softmax statistics tile size.
  int32_t mTileSizeStats;
  // Epilogue tile sizes for each instance in the M dimension.
  int32_t mTileSizeEpilogueM;
  // Epilogue tile sizes for each instance in the N dimension.
  int32_t mTileSizeEpilogueN;
  // The TileSizeEpilogueN in uint32_t.
  int32_t mTileSizeEpilogueNInUInt32;
  // Split epilogueM into multiple tiles to use fewer registers. Each thread handles one uint32_t.
  int32_t mTileSizeStgEpilogueM;
  // The Tmem to Smem copy atom size of each iteration in a epilogue tile.
  int32_t mTmemCpAtomSizeEpilogue;
  // The copy atom size for copying softmax output to TMEM.
  int32_t mTmemCpAtomSizeP;
  // The BMM1 accumulator dtype in TMEM S: Int32 for Int8 MMA, Fp32 otherwise.
  tg::Dtype mDtypeBmm1Acc;

  // The default constructor.
  KernelConfig() {}

  // The constructor where values are set.
  template <typename FmhaOptions_>
  KernelConfig(FmhaOptions_ const& options)
    : KernelConfigBase(options) // Copy all base class members from options.
    , mSupportsPagedKv{options.mQkvLayout == QkvLayout::PagedKv} {

    // Derive mDtypeKv: equals mDtypeK when K and V share the same dtype, else Void.
    mDtypeKv = (mDtypeK == mDtypeV) ? mDtypeK : tg::Dtype::Void;

    // Override mHeadDimPerCtaV with computed value if it was 0.
    if (mHeadDimPerCtaV == 0) {
      mHeadDimPerCtaV = mHeadDimV;
    }

    // The data type of softmax computation.
    if (options.mEnablesFp16Softmax) {
      // E4m3 kernels will also use Fp16 for softmax computation.
      mDtypeSoftmax = (mDtypeQ == tg::Dtype::Bfloat16) ? tg::Dtype::Bfloat16 : tg::Dtype::Fp16;
    }

    // The maximum headDim for K and V.
    mMaxHeadDimKv = std::max(mHeadDimQk, mHeadDimV);

    // The tileSizeN of the tileB in each CTA, which is computed as tileSizeND / clusterDimX.
    int32_t tileSizeNB = mTileSizeKv / options.mClusterDimX;

    // Is it the DS MLA-generation kernel with keepsMmaAb?
    bool keepsMmaAbForDsMlaGen = mIsMlaGen &&
                                 isKeepsMmaAbForGenerationKernel(options.mFmhaKernelType) &&
                                 (mHeadDimQk == 576 && mHeadDimV == 512);

    // Set the number of stages of the Q shared memory buffer.
    // When numTileKv is equal to 2, setting mNumStagesQ to 2 can also improve performance with the
    // persistent tile scheduler.
    mNumStagesQ = 2;
    if (mSwapsMmaAb) {
      // Assume maximum 32KB of smemQ can be used considering smemP and smemKv usage.
      // This will be finetuned later.
      int64_t totalNumBitsQ = 32 * 1024 * 8;
      // The number of bits per tileQ.
      int64_t numBitsPerTileQ = tg::dtypeGetNumBits(mDtypeQ) * mTileSizeQ * mHeadDimQk;
      // The number of stages.
      mNumStagesQ = std::min(2, static_cast<int32_t>(ceilDiv(totalNumBitsQ, numBitsPerTileQ)));
    } else if (mNumInstsQ == 1 && options.mTileScheduler == TileScheduler::Static) {
      mNumStagesQ = 1;
    }

    // Set numStagesQ for headDim > 128 kernels.
    bool const isGenerationSkipSoftmax =
      options.mSkipsSoftmaxWhenPossible && !isContextKernel(options.mFmhaKernelType);
    if (mNumInstsQ * mNumInstsKv == 1 && !isGenerationSkipSoftmax) {
      TLLM_CHECK_INFO(mTileSizeQ == 64 || (mHeadDimQk > 128 && mHeadDimV > 128),
                      "Consider using numInstsQ = 2 for better performance.");
    }
    if (mNumInstsQ * mNumInstsKv == 1) {
      // There is no enough shared memory for 2 stages when the headDim is not split into multiple
      // stages.
      if (mHeadDimPerStageKv == 0 && keepsMmaAbForDsMlaGen) {
        mNumStagesQ = 1;
      } else {
        // There is no enough shared memory for 2 stages when 16bit kv data type is used.
        mNumStagesQ = int32_t{2 * 8 /*bits*/ / std::max(tg::dtypeGetNumBits(mDtypeQ), 8)};
      }
    }

    // The number of stages of the KV shared memory buffer.
    if (tg::isArchHopper(options.mCudaArch)) {
      mNumStagesTransformedKv = 0;
      mNumStagesQ = mNumInstsQ;

      // The minimal amount of stages for the kernel to run
      mNumStagesK = 1;
      mNumStagesV = 1;
      mNumStagesTransposedV = mTransposeSmemV ? 1 : 0;

      // How much data in smem eash stage takes
      int32_t numBytesPerSmemStageQ = mTileSizeQ * mHeadDimQk * dtypeGetNumBits(mDtypeQ) / 8;
      int32_t numBytesPerSmemStageK =
        mTileSizeKv * std::max(mHeadDimQk, mHeadDimV) * dtypeGetNumBits(mDtypeK) / 8;
      int32_t numBytesPerSmemStageV =
        mTileSizeKv * std::max(mHeadDimQk, mHeadDimV) * dtypeGetNumBits(mDtypeV) / 8;

      // How much smem we have left
      int32_t numFreeBytes = 227 * 1024;
      // Take into account barriers in smem
      numFreeBytes -= 4 * 1024;
      // Take into account the minimal amount of stages
      numFreeBytes -= mNumInstsQ * numBytesPerSmemStageQ;
      numFreeBytes -= mNumStagesK * numBytesPerSmemStageK;
      numFreeBytes -= mNumStagesV * numBytesPerSmemStageV;
      numFreeBytes -= mNumStagesTransposedV * numBytesPerSmemStageV;

      // Increment all the stages if smem allows
      if (numFreeBytes - numBytesPerSmemStageK >= 0) {
        ++mNumStagesK;
        numFreeBytes -= numBytesPerSmemStageK;
      }

      if (mTransposeSmemV && (numFreeBytes - numBytesPerSmemStageV >= 0)) {
        ++mNumStagesTransposedV;
        numFreeBytes -= numBytesPerSmemStageV;
      }

      if (numFreeBytes - numBytesPerSmemStageV >= 0) {
        ++mNumStagesV;
        numFreeBytes -= numBytesPerSmemStageV;
      }

      mNumStagesKv = mNumStagesK + mNumStagesV;
    } else {


      // If dtypeQ != dtypeKv, the kv elements will be converted to dtypeQ in smemTransformedKv,
      // so the number of stages will be computed based on dtypeQ.
      mNumStagesKv = 3;

      // When the headDim is not split into multiple stages, we can use at most 4 stages for e4m3
      // data type.
      if (mHeadDimPerStageKv == 0 && keepsMmaAbForDsMlaGen) {
        TLLM_CHECK_ERROR(options.mSeparateSmemKv, "Not supported");
        mNumStagesKv = int32_t{4 * 8 /*bits*/ / std::max(tg::dtypeGetNumBits(mDtypeQ), 8)};
      } else if (keepsMmaAbForDsMlaGen) {
        // For DS MLA-generation kernels with keepsMmaAb, we can only have at most 8 stages for e4m3
        // data type.
        mNumStagesKv = int32_t{(tileSizeNB == 64 ? 14 : 8) * 8 /*bits*/ /
                               std::max(tg::dtypeGetNumBits(mDtypeQ), 8)};
      } else if (mReuseSmemKForV) {
        // Only MlaGen kernels support reusing smemK for V. More limitations can be found in
        // checkFmhaOptions.
        mNumStagesKv = (mHeadDimQk + mHeadDimPerStageKv - 1) / mHeadDimPerStageKv * 2;
      } else if (mNumInstsQ == 1) {
        // The numHeadDimBytes (padded to multiple of 128B)
        int32_t numHeadDimBytes = ceilDiv(tg::dtypeGetNumBits(mDtypeQ) * mHeadDimQk / 8, 128) * 128;
        // The total number of KB for smemQ.
        int32_t totalNumKBSmemQ = numHeadDimBytes * mTileSizeQ / 1024;
        // The maximum buffer size for smemKv (at most 144KB for KV, and at most 218KB for Qkv).
        int32_t maxBufferSizeKBForSmemKv = std::min(144, 218 - totalNumKBSmemQ * mNumStagesQ);
        // Calculate the number of stages for smemKv.
        mNumStagesKv = calculateNumStagesKv(maxBufferSizeKBForSmemKv);
      }
      mNumStagesTransformedKv = tg::dtypeGetNumBits(mDtypeKv) >= 8 ? 2 : 4;
      mNumStagesTransposedV = 0;
      // Set mNumStagesK and mNumStagesV to reasonable values.
      mNumStagesK = mNumStagesKv / 2;
      mNumStagesV = mNumStagesKv - mNumStagesK;
    }

    // The number of stages of the pageOffsetsKv shared memory buffer.
    mNumStagesPageOffsetsKv = 6;
    TLLM_CHECK_ERROR(mNumStagesPageOffsetsKv >= 2,
                     "The numStagesPageOffsetsKv must be >= 2 as it is shared by both K and V");

    // Set the number of tmem cols we will allocate once.
    mNumTmemCols = 512;
    // Set the softmax statistics tile size.
    mTileSizeStats = 32;
    // Set epilogue tile sizes for each instance in the M dimension.
    mTileSizeEpilogueM = mTileSizeQ;
    // Set epilogue tile sizes for each instance in the N dimension. Preferably 128B.
    mTileSizeEpilogueN =
      std::min(int32_t{128 * 8 /*bits*/ / tg::dtypeGetNumBits(mDtypeQ)}, mHeadDimV);
    // Set the Tmem to Smem copy atom size of each iteration in a epilogue tile.
    mTmemCpAtomSizeEpilogue = 16;

    // Set the number of stages of the Epilogue buffer.
    mNumStagesEpilogue = std::min(2, std::max(1, mHeadDimV / mTileSizeEpilogueN));
    // The number of stages of the persistent work id pipeline.
    mNumStagesWorkId = 2;

    // Make sure Tmem copy atom size never overreaches the size of the tile.
    TLLM_CHECK_ERROR(mTmemCpAtomSizeEpilogue <= mTileSizeEpilogueN, "Invalid size");

    // Set the number of Q elements in the packed uint32_t element.
    mNumEltsPerUInt32Q = 32 / tg::dtypeGetNumBits(mDtypeQ);
    // Set the number of KV elements in the packed uint32_t element.
    mNumEltsPerUInt32Kv = 32 / tg::dtypeGetNumBits(mDtypeK);
    // Set the number of O elements in the packed uint32_t element.
    mNumEltsPerUInt32O = 32 / tg::dtypeGetNumBits(mDtypeOut);

    // The tmemTileSizeND.
    int32_t tmemTileSizeND = mTileSizeKv;
    // If the tileSizeQ is 64, the 2x2 warp layout (M=128 2CTA MMA) results in only half of the
    // TMEM columns being processed by each thread.
    if (mTileSizeQ == 64) {
      tmemTileSizeND /= options.mClusterDimX;
    }

    // The accumulator type for Bmm1.
    mDtypeBmm1Acc = (mDtypeQ == tg::Dtype::Int8) ? tg::Dtype::Int32 : tg::Dtype::Fp32;
    // Input type for Bmm2.
    auto const dtypeBmm2 = (mDtypeK != mDtypeQ) ? mDtypeQ : mDtypeV;
    // Number of Bmm2 input elements per 32 bit pack.
    int32_t numEltsPerUInt32P = 32 / tg::dtypeGetNumBits(dtypeBmm2);
    // The maximum tmemCpAtomSizeP.
    int32_t maxTmemCpAtomSizeP = tmemTileSizeND / numEltsPerUInt32P;
    TLLM_CHECK_ERROR(tmemTileSizeND % numEltsPerUInt32P == 0, "The tileSizeKv is not supported");
    // Set the copy atom size for copying softmax output to TMEM.
    if (dtypeBmm2 == tg::Dtype::E4m3 || dtypeBmm2 == tg::Dtype::Int8) {
      // !!!
      // WARNING: On e4m3 kernels, 16 leads to more scoreboard dependencies in softmax SASS
      // sequence.
      // !!!
      mTmemCpAtomSizeP = std::min(32, maxTmemCpAtomSizeP);
    } else if (dtypeBmm2 == tg::Dtype::Fp16 || dtypeBmm2 == tg::Dtype::Bfloat16) {
      // !!!
      // WARNING: On half-precision kernels, 32 leads to excessive spill in softmax SASS sequence.
      // !!!
      mTmemCpAtomSizeP = std::min(16, maxTmemCpAtomSizeP);
    } else {
      TLLM_LOG_ERROR("Unexpected dtype ", static_cast<int32_t>(mDtypeQ));
    }

    // The paged-kv configurations. The number of tokens in one pageKv.
    mNumTokensPerPage = mSupportsPagedKv ? options.mNumTokensPerPage : mTileSizeKv;

    // The swizzle patterns require the row dimension (seqLen dimension) to be equal to or larger
    // than 8. Otherwise, the final shared memory tensor, after multiple copies, won't match the
    // expected swizzle pattern.
    TLLM_CHECK_ERROR(mSupportsPagedKv || (((mNumTokensPerPage & (mNumTokensPerPage - 1)) == 0) &&
                                          (mNumTokensPerPage >= 8)),
                     "NumTokensPerPage must be power-of-2 and equal or large than 8");

    // The TileSizeEpilogueN in uint32_t.
    TLLM_CHECK_ERROR(mTileSizeEpilogueN % mNumEltsPerUInt32Q == 0, "Invalid size");
    mTileSizeEpilogueNInUInt32 = mTileSizeEpilogueN / mNumEltsPerUInt32Q;

    // Split epilogueM into multiple tiles to use fewer registers. Each thread handles one
    // uint32_t.
    mTileSizeStgEpilogueM = 32 / mTileSizeEpilogueNInUInt32;
  }

  // Prevent accidental use of base-class operator== on KernelConfig.
  bool operator==(KernelConfig const&) const = delete;
  bool operator!=(KernelConfig const&) const = delete;

  // For K/V dtypes, headDim staging in SmemKv are always calculated based on the smaller-sized
  // dtype (e.g. e4m3), while this mHeadDimPerStageKv needs to be bit-reinterpreted to dtypeK which
  // could be a larger-sized datatype.
  int32_t getHeadDimPerStageK() const {
    int32_t headDimPerStageK = mHeadDimPerStageKv;
    if (headDimPerStageK > 0 && mDtypeK == mDtypeQ && mDtypeK != mDtypeV && !mSeparateSmemKv) {
      headDimPerStageK =
        headDimPerStageK * tg::dtypeGetNumBits(mDtypeV) / tg::dtypeGetNumBits(mDtypeK);
      TLLM_CHECK_ERROR(headDimPerStageK > 0, "Invalid effective headDimPerStageK.");
    }
    return headDimPerStageK;
  }

private:
  // Calculate the number of stages for K/V.
  int32_t calculateNumStagesKv(int32_t bufferSizeKB) {
    // The total number of bits for K/V.
    int64_t totalNumBitsKv = bufferSizeKB * 1024 * 8;
    // The head dimension per stage for K/V.
    int32_t headDimPerStageKv = mHeadDimPerStageKv == 0 ? mHeadDimQk : getHeadDimPerStageK();
    // The number of bits per tileKv.
    int64_t numBitsPerTileKv =
      std::max(tg::dtypeGetNumBits(mDtypeQ), 8) * mTileSizeKv * headDimPerStageKv;
    // The number of stages.
    return std::max(1, static_cast<int32_t>(totalNumBitsKv / numBitsPerTileKv));
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct MmaTraits {
  // The data type of Bmm1.
  tg::Dtype mDtypeBmm1;
  // The data type of Bmm2.
  tg::Dtype mDtypeBmm2;
  // The Atom Mn for Q tensor.
  int32_t mAtomQMn;
  // The Atom Mn for K tensor.
  int32_t mAtomKMn;
  // The Atom Mn for P tensor (softmax output).
  int32_t mAtomPMn;
  int32_t mTilePMn;
  // The Atom Mn for V tensor.
  int32_t mAtomVMn;
  // The Atom Mma for Q * K^T.
  int32_t mAtomQkM;
  int32_t mAtomQkN;
  int32_t mAtomQkK;
  // The Tile shape for Q * K^T accumulators (same as Atom shape).
  int32_t mTileQkM;
  int32_t mTileQkN;
  // The number of accumulator registers per thread.
  int32_t mNumQkEltsPerThread;
  // The Atom Mma for P * V.
  int32_t mAtomPvM;
  int32_t mAtomPvN;
  int32_t mAtomPvK;
  // The Tile shape for P * V accumulators (same as Atom shape).
  int32_t mTilePvM;
  int32_t mTilePvN;
  int32_t mValidTilePvN;
  // Whether to use UTCMMA in 2-CTA mode.
  bool mUseUtcmma2CtaMode;
  // The number of accumulator registers per thread.
  int32_t mNumPvEltsPerThread;
  // The default constructor.
  MmaTraits() {}

  // The constructor.
  template <typename FmhaOptions_>
  MmaTraits(FmhaOptions_ const& options)
    : mDtypeBmm1{options.mDtypeQ}
    , mDtypeBmm2(options.mDtypeK != options.mDtypeQ ? options.mDtypeQ : options.mDtypeV) {

    // Whether to use 2-CTA mode for UTCMMA.
    mUseUtcmma2CtaMode = options.mClusterDimX == 2;
    // For now, 128x128x16 is not supported. It requires 228KB of shared memory but the way we
    // generate code leads to an oversubscription of shared memory. To fix that limitation, we
    // would have to separate how we allocate shared memory to avoid aligning barriers on 1,024B
    // boundaries.
    TLLM_CHECK_ERROR(options.mHeadDimV == 32 || options.mHeadDimV == 64 ||
                       options.mHeadDimV == 128 || options.mHeadDimV % 128 == 0,
                     "Unsupported HeadDim for BMM2-N ",
                     options.mHeadDimV);

    // Whether MMA dtype is an 8-bit type (FP8/INT8).
    bool isMma8BitBmm1 = tg::dtypeGetNumBits(mDtypeBmm1) == 8;
    bool isMma8BitBmm2 = tg::dtypeGetNumBits(mDtypeBmm2) == 8;

    // The Atom Mn for Q tensor.
    mAtomQMn = options.mTileSizeQ;
    // The Atom Mn for K tensor.
    mAtomKMn = options.mTileSizeKv;

    // The Atom Mn for P tensor (softmax output).
    mAtomPMn = mAtomQMn;
    mTilePMn = mAtomPMn;

    // The Atom Mn for V tensor.
    mAtomVMn = options.mHeadDimV;

    // The Atom Mma for Q * K^T.
    mAtomQkM = options.mSwapsMmaAb ? options.mTileSizeKv : options.mTileSizeQ;
    mAtomQkN = options.mSwapsMmaAb ? options.mTileSizeQ : options.mTileSizeKv;
    {
      mAtomQkK = isMma8BitBmm1 ? 32 : 16;
    }

    if (tg::isArchHopper(options.mCudaArch) && options.mTransposeSmemV &&
        (options.mNumTransposeWarps == 4)) {
      // Make sure the accumulator of a single MMA instruction is smaller than 128 regs,
      // otherwise it would not compile with ctaSize=512,
      // even if we increase registers for the warp with setmaxnreg
      mAtomQkN = std::min(mAtomQkN, 128);
    }

    // The Tile shape for Q * K^T accumulators (same as Atom shape).
    mTileQkM = mAtomQkM;
    mTileQkN = mAtomQkN;

    // The number of accumulator registers per thread.
    mNumQkEltsPerThread = mAtomQkN;

    // The Atom Mma for P * V - M/N dimensions.
    //
    // For DeepSeek MLA, we force the MMA's headDim to be 128 as we distribute the headDim
    // over multiple MMAs.
    mAtomPvM = options.mSwapsMmaAb ? std::min(128, options.mHeadDimV) : options.mTileSizeQ;
    mAtomPvN = options.mSwapsMmaAb ? options.mTileSizeQ
                                   : std::min(128 * options.mClusterDimX, options.mHeadDimV);

    if (tg::isArchHopper(options.mCudaArch) && (options.mNumInstsQ == 1)) {
      mAtomPvN = std::min(128 * options.mClusterDimX, options.mHeadDimV / 2);
    }

    if (tg::isArchHopper(options.mCudaArch) && options.mTransposeSmemV &&
        (options.mNumTransposeWarps == 4)) {
      // Make sure the accumulator of a single MMA instruction is smaller than 128 regs,
      // otherwise it would not compile with ctaSize=512,
      // even if we increase registers for the warp with setmaxnreg
      mAtomPvN = std::min(mAtomPvN, 128);
    }

    // The K dimension.
    {
      mAtomPvK = isMma8BitBmm2 ? 32 : 16;
    }

    // The Tile shape for P * V accumulators (same as Atom shape).
    mTilePvM = mAtomPvM;
    mTilePvN = mAtomPvN;
    mValidTilePvN = options.mSwapsMmaAb ? mAtomQMn : mAtomPvN;

    // Check if TilePvM is valid or not.
    TLLM_CHECK_ERROR(mTilePvM == 64 || mTilePvM == 128,
                     "Invalid TilePvM as MMA only supports 64 or 128. mTilePvM=",
                     mTilePvM);

    // The number of accumulator registers per thread.
    // Note that when TilePvM is 64, each threads only load 16 data paths from TMEM.
    mNumPvEltsPerThread = (mValidTilePvN * mTilePvM) / 128;

    // Update the MMA instruction shapes considering the 2Cta mode.
    if (mUseUtcmma2CtaMode) {
      // The base MMA instruction shape is set to tileD's shape, which is (M / 2) x N.
      // As a result, we need to multiply the M dimension by 2.
      mAtomQkM *= 2;
      mAtomPvM *= 2;
    }
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper functions to get the number of head dimension stages for V.
// When the head dimension exceeds headDimPerStageKv, we must wrap it over multiple columns.
inline int32_t getNumHeadDimStagesV(int headDimPerCtaV, int headDimPerStageKv) {
  int numHeadDimStages{1};
  if (headDimPerCtaV > headDimPerStageKv && headDimPerStageKv > 0) {
    numHeadDimStages *= (headDimPerCtaV + headDimPerStageKv - 1) / headDimPerStageKv;
  }
  return numHeadDimStages;
}

struct KernelTraits : public KernelConfig, public MmaTraits {

  // The tile size for the correction step.
  int32_t mCorrTileSize;
  // Whether to cross-interleave TMEM columns for S and P across instances. When true, tmemP0
  // shares columns with tmemS1 and tmemP1 shares columns with tmemS0. Used when P does not have
  // dedicated columns (mSeparateTmemColsForSAndP false) to avoid data races. orderedSequence syncs
  // LDTM/STTM across instances.
  bool mInterleavesTmemSAndP;
  // The reshape factor for K/V TMA box width to be a multiple of 128B.
  int32_t mReshapeFactorKv;
  // The reshape factor for K/V scaling factor TMA box width.
  int32_t mReshapeFactorKvSf;
  // The number of head elts (per token) in each block of shared memory (see above explanation).
  int32_t mNumEltsInClampedHeadDimKv;
  // The number of head elts (per token) in each block of shared memory (see above explanation).
  int32_t mNumEltsInClampedHeadDimQ;
  // The number of FP4 elements per scaling factor.
  int32_t mNumEltsPerSf;
  // The K SMEM tile size.
  int32_t mNumEltsPerSmemStageK;
  // The KV SMEM tile size.
  int32_t mNumEltsPerSmemStageKv;
  // The V SMEM tile size.
  int32_t mNumEltsPerSmemStageV;
  // The number of keys per tile.
  int32_t mNumKeysPerTile;
  // The number of page offsets loaded per stage.
  int32_t mNumLoadedPageOffsetsPerStage;
  // The number of warps for loadTask.
  int32_t mNumLoadTaskWarps;
  // The number of transform stages for SmemTransformedKv.
  int32_t mNumSmemTransformStages;
  // The number of TMEM columns for each stage of K/V.
  int32_t mNumTmemColsPerStageKv;
  // The number of TMEM columns for P.
  int32_t mNumTmemColsP;
  // The number of TMEM columns for S.
  int32_t mNumTmemColsS;
  // The K SMEM transaction size.
  int32_t mNumTxBytesK;
  // The KV SMEM transaction size.
  int32_t mNumTxBytesKv;
  // The V SMEM transaction size.
  int32_t mNumTxBytesV;
  // Whether to use separate TMEM columns for S and P to avoid potential data races.
  bool mSeparateTmemColsForSAndP;
  // Use separate TMEM columns for TmemS and TmemStat to avoid potential data races.
  bool mSeparateTmemColsForSAndStats;
  // The size of the K/V tile per stage.
  int32_t mSmemKvTileSize;
  // Whether to store the transformed K/V in the TMEM.
  bool mStoreTransformedKvInTmem;
  // Whether to store the softmax local in shared memory (true: SMEM, false: TMEM).
  bool mStoresSoftmaxLocalInSmem;
  // Whether swizzle is needed for K/V.
  bool mSwizzleKv;

  // The default constructor.
  KernelTraits()
    : KernelConfig()
    , MmaTraits() {}

  // The constructor.
  template <typename FmhaOptions_>
  KernelTraits(FmhaOptions_ const& options)
    : KernelConfig(options)
    , MmaTraits(options) {

    // The tile size for the correction step.
    mCorrTileSize = std::min(mValidTilePvN, 64);

    // The number of keys per tile.
    mNumKeysPerTile = std::min(mNumTokensPerPage, mTileSizeKv);

    // The number of page offsets loaded by all threads.
    mNumLoadedPageOffsetsPerStage = 32;
    // Sparse attention uses numTokensPerPage = 1 so more pageOffsets need to be loaded per stage.
    if (isTokenSparse(options.mSparseType)) {
      TLLM_CHECK_ERROR((mTileSizeKv % mNumKeysPerTile) == 0,
                       "mTileSizeKv must be divisible by mNumKeysPerTile");
      // FIXME(perkzz): In loadScheduleForTwoInstsOrTwoStagesKv, K0 and K1 are prefetched at the
      // start, but freqInfo for the Task does not properly skip the initial unrolled steps, which
      // can result in a deadlock. Additionally, for robustness, the KV ordering should be made
      // consistent between loadPageOffsetsKvTask (KVKV) and loadTask (KKVKVK).
      mNumLoadedPageOffsetsPerStage = 2 * (mTileSizeKv / mNumKeysPerTile);
    }

    // The number of warps for loadTask.
    // Sparse attention kernels use 4 warps as they use gather4 to gather sparse kv in the token
    // granularity (same as page size 1).
    mNumLoadTaskWarps = (isTokenSparse(options.mSparseType) ? 4 : 1);

    if (!tg::isArchHopper(options.mCudaArch)) {
      // Make sure the sizes for the epilogue satisfy the following conditions:
      TLLM_CHECK_ERROR(mSwapsMmaAb ||
                         (mTileSizeEpilogueM == mTileSizeQ && mHeadDimV % mTileSizeEpilogueN == 0),
                       "Not supported");
    }

    // Make sure the two MMAs consume the same K width in bits.
    int32_t bmm1KBits = mAtomQkK * tg::dtypeGetNumBits(mDtypeBmm1);
    int32_t bmm2KBits = mAtomPvK * tg::dtypeGetNumBits(mDtypeBmm2);
    {
      // For other architectures, the K width in bits of BMM1 and BMM2 must be the same.
      TLLM_CHECK_ERROR(bmm1KBits == bmm2KBits,
                       "BMM1-K and BMM2-K must have the same K width in bits.");
    }

    // Verify that the head dim is compatible with the K dimension of the MMA atom.
    TLLM_CHECK_ERROR((mHeadDimQk % mAtomQkK) == 0, "HeadDim % BMM1-K must be 0");

    auto headDimK = options.mHeadDimPerStageKv != 0 ? getHeadDimPerStageK() : mHeadDimQk;
    auto headDimV = options.mHeadDimPerStageKv != 0 ? options.mHeadDimPerStageKv : mHeadDimV;
    // The bytes per head for K/V. Shared KV storage is packed using the larger byte width.
    auto headSizeBytesK = headDimK * tg::dtypeGetNumBits(mDtypeK) / 8;
    auto headSizeBytesV = headDimV * tg::dtypeGetNumBits(mDtypeV) / 8;
    auto maxHeadSizeBytesKv = std::max(headSizeBytesK, headSizeBytesV);
    // The packed head size for shared KV storage, expressed in dtypeK elements.
    auto headSizeKvInDtypeKElts = maxHeadSizeBytesKv * 8 / tg::dtypeGetNumBits(mDtypeK);

    // In shared memory, the elements are stored as 2D blocks. For each one of those blocks, the
    // fastest moving dimension is the head dimension (even for V where the transposition happens
    // in the MMA). To be able to issue fast copies from shared memory to the tensor cores within
    // the main loop (and avoid bank conflicts), the number of contiguous channel elements must
    // fit in at most 128B. If the head dimension is larger than what fits in 128B, multiple
    // blocks are needed to store the elements. If the head dimension is not large enough to fill
    // 128B worth of data, the number of channels per token in each SMEM block must be clamped
    // (the hardware can efficiently deal with 32B, 64B and 128B per token).

    // The number of elements in 128B for Q.
    int32_t numEltsIn128BQ = mNumEltsPerUInt32Q /*4B*/ * 32;
    // The number of head elts (per token) in each block of shared memory (see above explanation).
    mNumEltsInClampedHeadDimQ = std::min(numEltsIn128BQ, mHeadDimQk);

    // The number of elements in 128B for Q.
    int32_t numEltsIn128BKv = mNumEltsPerUInt32Kv /*4B*/ * 32;
    // The number of head elts (per token) in each block of shared memory (see above explanation).
    TLLM_CHECK_ERROR(mHeadDimQk == mHeadDimV || headSizeKvInDtypeKElts >= numEltsIn128BKv,
                     "Different mNumEltsInClampedHeadDim for K and V might be needed");
    mNumEltsInClampedHeadDimKv =
      std::min(numEltsIn128BKv, static_cast<int32_t>(headSizeKvInDtypeKElts));

    // The HW is designed to have NumEltsIn128B elements consumed by 4 UTC?Mmas in the K
    // dimension.
    {
      TLLM_CHECK_ERROR(numEltsIn128BQ == mAtomQkK * 4, "Internal error");
    }

    // TmemS and TmemStats can use different TMEM columns to avoid potential data-races when
    // (numTmemSCols + numTmemStatsCols) * mNumInstsQ * mNumInstsKv <= 256.
    mNumTmemColsS = mTileQkN;
    if (mUseUtcmma2CtaMode && mAtomQkM == 128) {
      // Only need half the number of TMEM columns for S if M=128 2CTA UTCMMA instruction is used.
      mNumTmemColsS /= 2;
    }

    // The number of transform stages for SmemTransformedKv.
    mNumSmemTransformStages = 2;

    // Note: mInterleaveSfV and mUsesSharedPagedKvIdx are inherited from KernelConfigBase.

    // The number of FP4 elements per scaling factor.
    mNumEltsPerSf = 16;

    // The number of FP4 SF per row.
    int32_t numSfPerRow = mMaxHeadDimKv / mNumEltsPerSf;

    // The head dimension bytes for K/V.
    auto headDimBytesKv = maxHeadSizeBytesKv;
    auto headDimBytesK = headDimK * tg::dtypeGetNumBits(mDtypeK) / 8;
    auto headDimBytesV = headDimV * tg::dtypeGetNumBits(mDtypeV) / 8;
    // The tileSizeNB of the BMM1, which is computed as tileSizeND / clusterDimX.
    int32_t tileSizeNB = mTileSizeKv / options.mClusterDimX;

    // Whether to store the transformed K/V in the TMEM. Currently only swapsMmaAb generation
    // phase kernels are supported. Only supports E2M1 Kv and E4M3 Q. HeadDimPerStageKv should be
    // 128, this is a limitation of TMA load of type CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B.
    auto transformedKvHeadDim = mHeadDimPerStageKv == 0 ? mHeadDimQk : mHeadDimPerStageKv;
    mStoreTransformedKvInTmem = options.mDtypeKv == tg::Dtype::E2m1 &&
                                options.mDtypeQ == tg::Dtype::E4m3 &&
                                (transformedKvHeadDim == 128) && mSwapsMmaAb;

    // The number of TMEM columns for S.
    int32_t numTmemColsS = mNumTmemColsS * 2;
    // The number of instances Qkv.
    auto numInstsQkv = mNumInstsQ * mNumInstsKv;
    // The number of TMEM columns for O.
    int32_t numTmemColsO =
      getNumHeadDimStagesV(mHeadDimPerCtaV, transformedKvHeadDim) * mAtomPvN * numInstsQkv;
    // Number of TMEM columns for P (each TMEM column is 32-bit wide).
    // SwapsMmaAb uses shared memory for P, so no need to allocate TMEM columns for P.
    mNumTmemColsP = mSwapsMmaAb ? 0 : ceilDiv(mNumTmemColsS * tg::dtypeGetNumBits(mDtypeQ), 32);

    // Check if enough TMEM columns are available for separate tmemP and tmemStats.
    {
      // Separate TMEM columns for S and P have only been implemented for context kernels.
      // Using both mSeparateTmemColsForSAndP and mStoresSoftmaxLocalInSmem degrades performance
      // with Pv0_Qk0_Pv1_Qk1 order. Separate TMEM for S and P is not used in this scenario.
      // TODO: Add support for separate TMEM S and P in keepsMmaAb kernels.
      mSeparateTmemColsForSAndP = (mNumInstsQ == 2) && (mMmaOrder == MmaOrder::Qk0_Pv0_Qk1_Pv1 ||
                                                        mMmaOrder == MmaOrder::Qk0_Qk1_Pv0_Pv1);
      mInterleavesTmemSAndP = false;
      mSeparateTmemColsForSAndStats = false;
      if (!isValidTmemAllocation(numTmemColsS, numTmemColsO) && mSeparateTmemColsForSAndP) {
        mSeparateTmemColsForSAndP = false;
        // If not enough TMEM columns for separate tmemP and tmemStats, try to use cross-interleaved
        // S and P to save TMEM columns.
        if ((mMmaOrder == MmaOrder::Qk0_Pv0_Qk1_Pv1 || mMmaOrder == MmaOrder::Qk0_Qk1_Pv0_Pv1)) {
          mInterleavesTmemSAndP = true;
        }
      }
      mSeparateTmemColsForSAndStats = true;
      mStoresSoftmaxLocalInSmem = false;
      // Store softmaxStats to shared memory for mInterleavesTmemSAndP or mSeparateTmemColsForSAndP
      // is true. as it gives slight performance improvement (the hypothesis is that using TMEM for
      // stats has more contention with tmemP/tmemS).
      if (mInterleavesTmemSAndP || mSeparateTmemColsForSAndP) {
        mStoresSoftmaxLocalInSmem = true;
        mSeparateTmemColsForSAndStats = false;
      } else if (!isValidTmemAllocation(numTmemColsS, numTmemColsO)) {
        mSeparateTmemColsForSAndStats = false;
      }
    }

    // The number of TMEM columns for Stats.
    int32_t numTmemColsStats = mSeparateTmemColsForSAndStats ? mTileSizeStats * 2 : 0;

    // Determine the number of stages if mStoreTransformedKvInTmem is true.
    mNumTmemColsPerStageKv = 0;
    if (mStoreTransformedKvInTmem) {
      // The number of TMEM columns for each stage of E2m1 K/V.
      mNumTmemColsPerStageKv = (transformedKvHeadDim * mTileSizeKv) / (4 * 128);
      // The number of TMEM columns for K/V.
      int32_t numTmemColsKv = mNumTmemCols - numTmemColsS - numTmemColsStats - numTmemColsO;
      // The number of stages for K/V.
      mNumStagesTransformedKv = numTmemColsKv / mNumTmemColsPerStageKv;
    }

    // Sanity checks if we're using separate TMEM columns for S and P.
    if (mSeparateTmemColsForSAndP) {
      // Check that enough TMEM columns are available.
      int32_t numTmemColsNeeded = numTmemColsS + numTmemColsStats + numTmemColsO +
                                  (mInterleavesTmemSAndP ? 0 : 2 * mNumTmemColsP);
      TLLM_CHECK_ERROR_FMT(
        mNumTmemCols - numTmemColsNeeded >= 0,
        "Not enough TMEM columns available. Have %d, need %d (S: %d, stats: %d, O: %d, P: %d)",
        mNumTmemCols,
        numTmemColsNeeded,
        numTmemColsS,
        numTmemColsStats,
        numTmemColsO,
        2 * mNumTmemColsP);
      // Check that we're not storing transformed K/V in TMEM.
      TLLM_CHECK_ERROR(!mStoreTransformedKvInTmem,
                       "StoreTransformedKvInTmem incompatible with SeparateTmemColsForSAndP");
    }
    // Sanity checks for cross-interleaved S and P.
    if (mInterleavesTmemSAndP) {
      TLLM_CHECK_ERROR(mNumInstsQ == 2 && mNumInstsKv == 1,
                       "InterleaveTmemSAndP requires numInstsQ == 2 and numInstsKv == 1");
      TLLM_CHECK_ERROR(!mStoreTransformedKvInTmem,
                       "StoreTransformedKvInTmem incompatible with InterleaveTmemSAndP");
    }

    // The size of the K/V tile per stage.
    // Each CtaX in the cluster only needs to load half N of tensor B if the 2Cta Utcmma
    // instruction is used.
    mSmemKvTileSize = tileSizeNB * headSizeKvInDtypeKElts;
    // Allow NVFP4 only for KV-cache-like input.
    if (mDtypeK == tg::Dtype::E2m1 || mDtypeV == tg::Dtype::E2m1) {
      TLLM_CHECK_ERROR(mDtypeK == mDtypeV, "E2m1 requires dtypeK == dtypeV.");
    }
    // The size of the K/V scaling factor tile per stage.
    int32_t smemKvSfTileSize = mDtypeKv == tg::Dtype::E2m1 ? tileSizeNB * numSfPerRow : 0;
    // Calculate the mNumEltsPerSmemStageKv.
    if (mDtypeKv == tg::Dtype::E2m1 && !mStoreTransformedKvInTmem) {
      // When mStoreTransformedKvInTmem is false, input FP4 elements are packed in 8-bit
      // containers. Therefore the number of elements should be halved. When
      // mStoreTransformedKvInTmem is true, since CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B is used
      // for TMA loading, each 8 4-bit elements are padded to 16B. In average, each element
      // occupies 8 bit, therefore no need to halve the number of elements.
      mNumEltsPerSmemStageKv = mSmemKvTileSize / 2 + smemKvSfTileSize;
    } else {
      // The number of elements per stage for K/V.
      mNumEltsPerSmemStageKv = mSmemKvTileSize + smemKvSfTileSize;
    }

    // Align each SMEM stage to 1024B when mStoreTransformedKvInTmem is true, as SWIZZLE_128B is
    // used for TMA loading.
    if (mStoreTransformedKvInTmem) {
      mNumEltsPerSmemStageKv = ceilDiv(mNumEltsPerSmemStageKv, 1024) * 1024;
    }

    // Do we have to transform K/V before MMA?
    bool const transformsKv{mDtypeK != mDtypeQ};
    // Whether swizzle is needed for K/V.
    mSwizzleKv = mStoreTransformedKvInTmem ? true : !transformsKv;

    // Whether we can reshape the TMA box width.
    bool canReshapeTmaKv = mSupportsPagedKv && mHeadDimQk == mHeadDimV && !mSwizzleKv;
    // The reshape factor for K/V TMA box width:
    //  - Aim for box width 128B.
    //  - But the box width must also be <= 128 elts for CU_TENSOR_MAP_SWIZZLE_128B.
    mReshapeFactorKv = canReshapeTmaKv
                         ? std::max(1,
                                    std::min({128 / mHeadDimQk,
                                              128 / (mHeadDimQk * tg::dtypeGetNumBits(mDtypeK) / 8),
                                              mNumKeysPerTile}))
                         : 1;
    // The reshape factor for K/V SF: aim for box width 128B, limit to numKeysPerTile.
    mReshapeFactorKvSf = mDtypeKv == tg::Dtype::E2m1
                           ? std::min(128 / (mMaxHeadDimKv / mNumEltsPerSf), mNumKeysPerTile)
                           : 1;

    // Calculate the transaction size for K/V.
    mNumTxBytesKv = tileSizeNB * maxHeadSizeBytesKv + smemKvSfTileSize * 1 /*byte per sf element*/;
    // Align the transaction size to 128B for TMA loading.
    if (headDimBytesKv > 128) {
      mNumTxBytesKv = (mNumTxBytesKv / headDimBytesKv) * ceilDiv(headDimBytesKv, 128) * 128;
    }

    // Calculate the transaction size for K.
    mNumTxBytesK = tileSizeNB * headDimBytesK;
    // Align the transaction size to 128B for TMA loading.
    if (headDimBytesK > 128) {
      mNumTxBytesK = tileSizeNB * ceilDiv(headDimBytesK, 128) * 128;
    }
    // The K SMEM tile size.
    mNumEltsPerSmemStageK = (mNumTxBytesK * 8) / tg::dtypeGetNumBits(mDtypeK);

    // For 2CTA M=256 MLA gen, a CTA cluster only loads half of headDimV.
    if (options.mClusterDimX == 2 && mIsMlaGen && mAtomQkM == 256 && mHeadDimV == 512) {
      headDimBytesV /= 2;
    }

    // Calculate the transaction size for V.
    mNumTxBytesV = tileSizeNB * headDimBytesV;
    // Align the transaction size to 128B for TMA loading.
    if (headDimBytesV > 128) {
      mNumTxBytesV = tileSizeNB * ceilDiv(headDimBytesV, 128) * 128;
    }
    // The V SMEM tile size.
    mNumEltsPerSmemStageV = (mNumTxBytesV * 8) / tg::dtypeGetNumBits(mDtypeV);
  }

  // Check if the TMEM allocation is valid.
  bool isValidTmemAllocation(int32_t numTmemColsS, int32_t numTmemColsO) const {
    // The number of TMEM columns needed when separate TMEM columns for S and Stats are used.
    int32_t numTmemColsNeeded = numTmemColsS + numTmemColsO;
    if (mSeparateTmemColsForSAndStats) {
      numTmemColsNeeded += mTileSizeStats * 2;
    }
    if (mSeparateTmemColsForSAndP) {
      numTmemColsNeeded += 2 * mNumTmemColsP;
    }
    // Check if the TMEM allocation is valid.
    return mNumTmemCols >= numTmemColsNeeded;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Create KernelTraits from FmhaOptions (or any options type that KernelTraits can be constructed
// from).
template <typename FmhaOptions_>
inline KernelTraits getKernelTraitsFromOptions(FmhaOptions_ const& options) {
  return KernelTraits(options);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

//
// clang-format off
// The TMEM allocation is organized as follows.
//
// If mSeparateTmemColsForSAndStats is true:
// [...................................FullTmem......................................................]
// [...TmemS0....][...TmemS1....][TmemStat0][TmemStat1][..TmemO0..][..TmemO1..][..TmemTransformedKv..]
// [TmemP0]       [TmemP1]
//
// If mSeparateTmemColsForSAndP is true:
// [...................................FullTmem.....................................................]
// [.......TmemS0........][.......TmemS1........][TmemP0][TmemP1][.....TmemO0.....][.....TmemO1.....]
// [TmemStat0]            [TmemStat1]
// 
// If mSeparateTmemColsForSAndP and mSeparateTmemColsForSAndStats are both true:
// [...................................FullTmem.....................................................]
// [..TmemS0..][..TmemS1..][TmemStat0][TmemStat1][..TmemP0..][..TmemP1..][...TmemO0...][...TmemO1...]
//
// If mInterleavesTmemSAndP is true (cross-interleaved S and P across instances):
// [...................................FullTmem.....................................................]
// [.......TmemS0........][........TmemS1........][.........TmemO0.........][.......TmemO1..........]
// When not enough tmem columns for separate stats: [TmemStat1][TmemP1] in S0, [TmemStat0][TmemP0] in S1.
// When enough tmem columns for stats (mSeparateTmemColsForSAndStats): stats use dedicated columns
// after S ([TmemStat0][TmemStat1]); only P shares with S: tmemP0 with tmemS1, tmemP1 with tmemS0.
// Used when mSeparateTmemColsForSAndP is false. orderedSequence syncs LDTM/STTM across instances.
//
// Otherwise, the TMEM allocation is organized as follows:
// [...................................FullTmem.....................................................]
// [.......TmemS0........][.......TmemS1........][..TmemO0..][..TmemO1..][....TmemTransformedKv.....]
// [TmemStat0][TmemP0]    [TmemStat1][TmemP1]
// clang-format on
//
inline int32_t getTmemAllocationS0(KernelTraits) {
  return 0;
}

inline int32_t getTmemAllocationS1(KernelTraits traits) {
  return getTmemAllocationS0(traits) + traits.mNumTmemColsS;
}

inline int32_t getTmemAllocationS(KernelTraits traits, int ii) {
  return ii == 0 ? getTmemAllocationS0(traits) : getTmemAllocationS1(traits);
}

// Code-gen-only Expr overloads (require Expr.h, Kernel.h, TmemTile.h).
#ifndef TLLM_FMHA_TRTLLM_COMPAT

inline tg::Expr const* getTmemAllocationS(tg::Kernel* kernel,
                                          KernelTraits traits,
                                          tg::Expr const* instId) {
  return TLLM_MAKE_OBJ(tg::ExprFromTernaryOp,
                       kernel,
                       tg::Dtype::Int32,
                       TLLM_MAKE_OBJ(tg::BinEq, kernel, instId, kernel->getInt0()),
                       kernel->getInt(getTmemAllocationS0(traits)),
                       kernel->getInt(getTmemAllocationS1(traits)));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getTmemAllocationStats0(KernelTraits traits) {
  // TmemS and TmemStats can use different TMEM columns to avoid potential data-races.
  return traits.mSeparateTmemColsForSAndStats ? (getTmemAllocationS1(traits) + traits.mNumTmemColsS)
                                              : getTmemAllocationS0(traits);
}

inline int32_t getTmemAllocationStats1(KernelTraits traits) {
  // TmemS and TmemStats can use different TMEM columns to avoid potential data-races.
  return traits.mSeparateTmemColsForSAndStats
           ? (getTmemAllocationStats0(traits) + traits.mTileSizeStats)
           : getTmemAllocationS1(traits);
}

inline int32_t getTmemAllocationStats(KernelTraits traits, int ii) {
  return ii == 0 ? getTmemAllocationStats0(traits) : getTmemAllocationStats1(traits);
}

inline tg::Expr const* getTmemAllocationStats(tg::Kernel* kernel,
                                              KernelTraits traits,
                                              tg::Expr const* instId) {
  return TLLM_MAKE_OBJ(tg::ExprFromTernaryOp,
                       kernel,
                       tg::Dtype::Int32,
                       TLLM_MAKE_OBJ(tg::BinEq, kernel, instId, kernel->getInt0()),
                       kernel->getInt(getTmemAllocationStats0(traits)),
                       kernel->getInt(getTmemAllocationStats1(traits)));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getTmemAllocationP0(KernelTraits traits) {
  if (traits.mInterleavesTmemSAndP) {
    // P0 shares columns with S1 (cross-interleaved across instances).
    return getTmemAllocationS1(traits) +
           ((traits.mSeparateTmemColsForSAndStats || traits.mStoresSoftmaxLocalInSmem)
              ? 0
              : traits.mTileSizeStats);
  } else if (traits.mSeparateTmemColsForSAndP) {
    return traits.mSeparateTmemColsForSAndStats
             ? (getTmemAllocationStats1(traits) + traits.mTileSizeStats)
             : getTmemAllocationS1(traits) + traits.mNumTmemColsS;
  } else {
    return getTmemAllocationS0(traits) +
           (traits.mSeparateTmemColsForSAndStats ? 0 : traits.mTileSizeStats);
  }
}

inline int32_t getTmemAllocationP1(KernelTraits traits) {
  if (traits.mInterleavesTmemSAndP) {
    // P1 shares columns with S0 (cross-interleaved across instances).
    return getTmemAllocationS0(traits) +
           ((traits.mSeparateTmemColsForSAndStats || traits.mStoresSoftmaxLocalInSmem)
              ? 0
              : traits.mTileSizeStats);
  } else if (traits.mSeparateTmemColsForSAndP) {
    return getTmemAllocationP0(traits) + traits.mNumTmemColsP;
  } else {
    return getTmemAllocationS1(traits) +
           (traits.mSeparateTmemColsForSAndStats ? 0 : traits.mTileSizeStats);
  }
}

inline int32_t getTmemAllocationP(KernelTraits traits, int ii) {
  return ii == 0 ? getTmemAllocationP0(traits) : getTmemAllocationP1(traits);
}

inline tg::Expr const* getTmemAllocationP(tg::Kernel* kernel,
                                          KernelTraits traits,
                                          tg::Expr const* instId) {
  return TLLM_MAKE_OBJ(tg::ExprFromTernaryOp,
                       kernel,
                       tg::Dtype::Int32,
                       TLLM_MAKE_OBJ(tg::BinEq, kernel, instId, kernel->getInt0()),
                       kernel->getInt(getTmemAllocationP0(traits)),
                       kernel->getInt(getTmemAllocationP1(traits)));
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getTmemAllocationO0(KernelTraits traits) {
  // Start with the end of TmemS/TmemStats/TmemP.
  if (traits.mSeparateTmemColsForSAndP) {
    return getTmemAllocationP1(traits) + traits.mNumTmemColsP;
  } else if (traits.mSeparateTmemColsForSAndStats) {
    return getTmemAllocationStats1(traits) + traits.mTileSizeStats;
  } else {
    return getTmemAllocationS1(traits) + traits.mNumTmemColsS;
  }
}

inline int32_t getTmemAllocationO1(KernelTraits traits) {
  // When the head dimension exceeds headDimPerStageKv, we must wrap it over multiple columns.
  return getTmemAllocationO0(traits) +
         traits.mAtomPvN * getNumHeadDimStagesV(traits.mHeadDimPerCtaV, traits.mHeadDimPerStageKv);
}

inline int32_t getTmemAllocationO(KernelTraits traits, int ii) {
  return ii == 0 ? getTmemAllocationO0(traits) : getTmemAllocationO1(traits);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

inline int32_t getTmemAllocationTransformedKv(KernelTraits traits) {
  // The number of instances Qkv.
  auto numInstsQkv = traits.mNumInstsQ * traits.mNumInstsKv;
  // The TMEM start offset of the transformed K/V.
  return getTmemAllocationO1(traits) +
         (getTmemAllocationO1(traits) - getTmemAllocationO0(traits)) * (numInstsQkv - 1);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

//
// For multi-stage processing (stages > 1), adjust column and row positions accordingly.
// Each headDimStage's TMEM D data is organized first across different columns, then across
// different rows (across the rows is only supported when tileSizeQ = 64).
// clang-format off
// Layout example:
// stage0: [row0,  col0-127] 
// stage1: [row0,  col128-255] 
// stage2: [row16, col0-127] 
// stage3: [row16, col128-255]
// clang-format on

inline int32_t getHeadDimStageTmemOffset(KernelTraits traits,
                                         int headDimStageIdx,
                                         bool isP = false,
                                         int numTmemColsPerStage = 0) {
  // The cluster dimension X.
  int clusterDimX{traits.mUseUtcmma2CtaMode ? 2 : 1};
  // When the head dimension exceeds headDimPerStageKv, we must wrap it over multiple columns.
  int numHeadDimStages{1};
  if (traits.mHeadDimPerCtaV > traits.mHeadDimPerStageKv && traits.mHeadDimPerStageKv > 0) {
    numHeadDimStages *=
      (traits.mHeadDimPerCtaV + traits.mHeadDimPerStageKv - 1) / traits.mHeadDimPerStageKv;
  } else if (traits.mHeadDimPerCtaV * clusterDimX > 256 && !isP && !traits.mSwapsMmaAb) {
    // The MmaPv might need multiple instructions to process the full headDimV, so the headDimOffset
    // is also needed in TmemCorr.
    // When swapsMmaAb is enabled, BMM2-M is always split into headDimStages (headDimPerStageKv > 0)
    // when headDimV > 128, so no need to split into headDimStages here.
    numHeadDimStages = ceilDiv(traits.mHeadDimPerCtaV * clusterDimX, traits.mAtomPvN);
  }

  // Return 0 if the number of stages is 1.
  if (numHeadDimStages == 1) {
    return 0;
  }

  // The number of TMEM columns per stage.
  if (numTmemColsPerStage == 0) {
    numTmemColsPerStage = traits.mAtomPvN;
  }

  // The maximum number of headDimStages per row.
  auto maxNumHeadDimStagesPerRow = 256 / numTmemColsPerStage;
  // Make sure the headDimStageIdx is within valid range.
  TLLM_CHECK_ERROR((headDimStageIdx + 1) * traits.mTileSizeQ * traits.mHeadDimPerStageKv <=
                     /* tmemRows */ 128 * /* tmemCols */ 256,
                   "Invalid headDimStageIdx");
  // The TMEM row offset.
  int32_t tmemRowOffset{(headDimStageIdx / maxNumHeadDimStagesPerRow) * 16 *
                        tg::TmemTile::TmemRowScale};
  // The TMEM column offset.
  int32_t tmemColOffset{(headDimStageIdx % maxNumHeadDimStagesPerRow) * numTmemColsPerStage};
  // The TMEM offset.
  // Note that the tmemRowOffsetP must be aligned to the tmemRowOffsetO.
  return isP ? tmemRowOffset : (tmemRowOffset + tmemColOffset);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline tg::Expr const* getHeadDimStageTmemOffset(tg::Kernel* kernel,
                                                 KernelTraits traits,
                                                 tg::Expr const* headDimStageIdx,
                                                 bool isP = false,
                                                 int numTmemColsPerStage = 0) {

  // The cluster dimension X.
  int clusterDimX{traits.mUseUtcmma2CtaMode ? 2 : 1};
  // When the head dimension exceeds headDimPerStageKv, we must wrap it over multiple columns.
  int numHeadDimStages{1};
  if (traits.mHeadDimPerCtaV > traits.mHeadDimPerStageKv && traits.mHeadDimPerStageKv > 0) {
    numHeadDimStages *=
      (traits.mHeadDimPerCtaV + traits.mHeadDimPerStageKv - 1) / traits.mHeadDimPerStageKv;
  } else if (traits.mHeadDimPerCtaV * clusterDimX > 256 && !isP && !traits.mSwapsMmaAb) {
    // The MmaPv might need multiple instructions to process the full headDimV, so the headDimOffset
    // is also needed in TmemCorr.
    // When swapsMmaAb is enabled, BMM2-M is always split into headDimStages (headDimPerStageKv > 0)
    // when headDimV > 128, so no need to split into headDimStages here.
    numHeadDimStages = ceilDiv(traits.mHeadDimPerCtaV * clusterDimX, traits.mAtomPvN);
  }
  // Return 0 if the number of stages is 1.
  if (numHeadDimStages == 1) {
    return kernel->getInt0();
  }

  // The number of TMEM columns per stage.
  if (numTmemColsPerStage == 0) {
    numTmemColsPerStage = traits.mAtomPvN;
  }

  // The maximum number of headDimStages per row.
  auto maxNumHeadDimStagesPerRow = 256 / numTmemColsPerStage;

  // Make sure the headDimStageIdx is within valid range.
  if (headDimStageIdx->isCompTimeCstIntScalar()) {
    tg::Int i{headDimStageIdx};
    TLLM_CHECK_ERROR((i.getVal() + 1) * traits.mTileSizeQ * traits.mHeadDimPerStageKv <=
                       /* tmemRows */ 128 * /* tmemCols */ 256,
                     "Invalid headDimStageIdx");
  }

  // The TMEM row offset.
  auto tmemRowOffset = TLLM_MAKE_OBJ(
    tg::BinMul,
    kernel,
    TLLM_MAKE_OBJ(tg::BinDiv, kernel, headDimStageIdx, kernel->getInt(maxNumHeadDimStagesPerRow)),
    kernel->getInt(16 * tg::TmemTile::TmemRowScale));

  // The TMEM column offset.
  auto tmemColOffset = TLLM_MAKE_OBJ(
    tg::BinMul,
    kernel,
    TLLM_MAKE_OBJ(tg::BinMod, kernel, headDimStageIdx, kernel->getInt(maxNumHeadDimStagesPerRow)),
    kernel->getInt(numTmemColsPerStage));
  // The TMEM offset.
  // Note that the tmemRowOffsetP must be aligned to the tmemRowOffsetO.
  return isP ? tmemRowOffset : TLLM_MAKE_OBJ(tg::BinAdd, kernel, tmemRowOffset, tmemColOffset);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

//
// Return a diagram string describing how TMEM (tensor memory) is allocated for the FMHA kernel.
//
// TMEM is partitioned into column ranges used by:
//   - tmemS:     Q*K^T scores (BMM1 output), two instances (S0, S1) for pipelining.
//   - tmemStats: Softmax row stats (e.g. max, sum); may share columns with S or be separate.
//   - tmemP:     Softmax output (P = softmax(S)), fed into BMM2; may share columns with S or be
//   separate.
//   - tmemO:     P*V output (BMM2 result), two instances (O0, O1) for pipelining.
//   - tmemTransformedKv: Optional storage for transformed K/V when mStoreTransformedKvInTmem is
//   set.
//
// The exact layout depends on kernel options (SeparateTmemColsForSAndStats,
// SeparateTmemColsForSAndP, InterleavesTmemSAndP). Uses the same getTmemAllocation* helpers so the
// diagram matches codegen layout.
//
template <typename FmhaOptions_>
inline std::string getTmemAllocationDiagram(FmhaOptions_ const& options) {
  KernelTraits traits(options);
  int32_t const totalCols = traits.mNumTmemCols;

  int32_t const s0 = getTmemAllocationS0(traits);
  int32_t const s1 = getTmemAllocationS1(traits);
  int32_t const numColsS = traits.mNumTmemColsS;
  int32_t const stats0 = getTmemAllocationStats0(traits);
  int32_t const stats1 = getTmemAllocationStats1(traits);
  int32_t const p0 = getTmemAllocationP0(traits);
  int32_t const p1 = getTmemAllocationP1(traits);
  int32_t const o0 = getTmemAllocationO0(traits);
  int32_t const o1 = getTmemAllocationO1(traits);
  int32_t const transformedKv = getTmemAllocationTransformedKv(traits);

  // Log kernelTraits related to TMEM allocation.
  TLLM_LOG_INFO("KernelTraits related to TMEM allocation:");
  TLLM_LOG_INFO("  mSeparateTmemColsForSAndP: ", toString(traits.mSeparateTmemColsForSAndP));
  TLLM_LOG_INFO("  mInterleavesTmemSAndP: ", toString(traits.mInterleavesTmemSAndP));
  TLLM_LOG_INFO("  mSeparateTmemColsForSAndStats: ",
                toString(traits.mSeparateTmemColsForSAndStats));
  TLLM_LOG_INFO("  mStoresSoftmaxLocalInSmem: ", toString(traits.mStoresSoftmaxLocalInSmem));
  TLLM_LOG_INFO("  mStoreTransformedKvInTmem: ", toString(traits.mStoreTransformedKvInTmem));

  // Print the TMEM allocation diagram.
  std::ostringstream oss;
  oss << "TMEM allocation diagram (columns 0.." << totalCols << "):\n";

  if (traits.mInterleavesTmemSAndP) {
    oss << "  Layout: cross-interleaved S and P (tmemP0 shares cols with tmemS1, tmemP1 with "
           "tmemS0).\n";
  } else if (traits.mSeparateTmemColsForSAndP) {
    oss << "  Layout: separate columns for S and P.\n";
  } else if (traits.mSeparateTmemColsForSAndStats) {
    oss << "  Layout: separate columns for S and Stats.\n";
  } else {
    oss << "  Layout: default (Stats/P alias S columns).\n";
  }

  auto range = [](int32_t start, int32_t end) {
    return "[" + std::to_string(start) + ".." + std::to_string(end) + ")";
  };

  oss << "  tmemS0:      " << range(s0, s0 + numColsS) << " (" << numColsS << " cols)\n";
  oss << "  tmemS1:      " << range(s1, s1 + numColsS) << " (" << numColsS << " cols)\n";
  if (traits.mSeparateTmemColsForSAndStats && traits.mTileSizeStats > 0) {
    oss << "  tmemStats0:  " << range(stats0, stats0 + traits.mTileSizeStats) << " ("
        << traits.mTileSizeStats << " cols)\n";
    oss << "  tmemStats1:  " << range(stats1, stats1 + traits.mTileSizeStats) << " ("
        << traits.mTileSizeStats << " cols)\n";
  }
  if (traits.mSeparateTmemColsForSAndP) {
    oss << "  tmemP0:      " << range(p0, p0 + traits.mNumTmemColsP) << " (" << traits.mNumTmemColsP
        << " cols)\n";
    oss << "  tmemP1:      " << range(p1, p1 + traits.mNumTmemColsP) << " (" << traits.mNumTmemColsP
        << " cols)\n";
  } else if (traits.mInterleavesTmemSAndP) {
    oss << "  tmemP0:      (aliased with tmemS1)\n";
    oss << "  tmemP1:      (aliased with tmemS0)\n";
  } else if (!traits.mSwapsMmaAb) {
    oss << "  tmemP:       (aliased with tmemS)\n";
  }
  int32_t const numColsO0 = o1 - o0;
  oss << "  tmemO0:      " << range(o0, o1) << " (" << numColsO0 << " cols)\n";
  oss << "  tmemO1:      " << range(o1, o1 + numColsO0) << " (" << numColsO0 << " cols)\n";
  if (traits.mStoreTransformedKvInTmem) {
    int32_t const numColsKv = totalCols - transformedKv;
    oss << "  tmemTransformedKv: " << range(transformedKv, totalCols) << " (" << numColsKv
        << " cols)\n";
  }
  oss << "  Total TMEM columns: " << totalCols;
  return oss.str();
}

#endif // !TLLM_FMHA_TRTLLM_COMPAT (code-gen-only Expr overloads)

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha
