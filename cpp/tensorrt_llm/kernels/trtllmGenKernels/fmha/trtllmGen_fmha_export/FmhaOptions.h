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

#include "KernelTraits.h"
#include <trtllm/gen/CudaArchDecl.h>
#include <trtllm/gen/CudaRunner.h>
#include <nlohmann/json.hpp>
#include <cfloat>
#include <string>

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// FmhaOptions class.
// Inherits from KernelConfigBase to share common configuration variables with KernelConfig.
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

struct FmhaOptions : public KernelConfigBase {
  // Relative error tolerance.
  float mAtol{1e-2f};
  // Attention windows size of sliding window attention. Disabled by default.
  int mAttentionWindowSize{0};
  // Batch size.
  int mBatchSize{2};
  // Whether to verify the correctness. 0: No check, 1: partial, 2: full.
  int mChecksResults{2};
  // The chunked attention size (default 0 means no chunking).
  int mChunkedAttentionSize{0};
  // Dry-run: print a log but does not actually generate anything
  bool mDryRun{false};
  // Enable the auto tuner.
  bool mEnablesAutoTuner{false};
  // Whether is exporting cubin.
  bool mIsExportingCubin{false};

  // Whether running inside TRTLLM (affects KV stride computation for MLA).
  bool mIsTrtllmLayout{false};
  // Whether the kernel is under tracing
  bool mIsTracing{false};
  // The maximum number of CTAs per sequenceKv (multiCtasKvMode).
  // This is used to limit the number of CTAs per sequenceKv for the multiCtasKvMode.
  int mMaxNumCtasPerSeqKv{INT_MAX};
  // The maximum number of CTAs for Q.
  int mMaxNumCtasQ{1};
  // The maximum number of CTAs for K/V.
  int mMaxNumCtasKv{1};
  // The maximum number of pages per sequence in the paged-kv buffer.
  int mMaxNumPagesPerSeqKv{512 / 32};
  // Sequence length for Q and K/V.
  int mMaxSeqLenQ{512}, mMaxSeqLenKv{512};
  // The minimum first sparseMask offset in the Kv sequence dimension.
  // Default 0 means that all tokensKv need custom masking.
  int mMinFirstSparseMaskOffsetKv{0};
  // The minimum sequence length (used to generate variable Q sequence length).
  int mMinSeqLenQ{INT_MAX};
  // The minimum sequence length (used to generate variable Kv sequence length).
  int mMinSeqLenKv{INT_MAX};
  // The minimum sparse MLA topK length.
  int mMinSparseMlaTopK{1};
  // Benchmark steps.
  int mNumBenchmarkSteps{1};
  // The number of Ctas per sequenceKv from the arguments.
  // It is used to fine-tune the multiCtasKvMode performance.
  int mNumCtasPerSeqKv{-1};
  // The number of loop iterations for schedulePrinter, which can helps identify deadlocks.
  int mNumLoopItersForPrint{2};
  // The number of pages in memory pool.
  int mNumPagesInMemPool{0};
  // The number of causal-mask spec-decoding tokens (it is fixed in the batch).
  int mNumSpecDecodingTokens{0};
  // Warmup steps.
  int mNumWarmUpSteps{0};
  // The maximum number of waves for the multiCtasKvMode.
  int mMaxNumWavesForCtasKvMode{1};
  // The attention output scale.
  float mOutputScale{1.f};
  // Relative error tolerance.
  float mRtol{1e-1f};
  // Whether to skip kernel generation (for debug purpose).
  bool mSkipsKernelGen{false};
  // The threshold to skip softmax operations when possible according to the below expression.
  float mSkipSoftmaxThresholdScaleFactor{0};
  // The topK value for sparse attention kernels.
  int mSparseAttnTopK{2048};
  // The sum of sequence lengths for Q and K/V.
  int mSumOfSeqLensQ{512 * 2}, mSumOfSeqLensKv{512 * 2};
  // Whether the indices for K & V pages are shared as unified index (vLLM/FlashInfer).
  bool mUsesSharedPagedKvIdx{false};
  // Level of verbose information.
  int mVerbosity{1};

  // Prevent accidental use of base-class operator== on FmhaOptions
  bool operator==(FmhaOptions const&) const = delete;
  bool operator!=(FmhaOptions const&) const = delete;

  // Convert the fmhaOptions to a JSON object.
  void toJson(nlohmann::json& j) const {
    // First, serialize the base class members.
    KernelConfigBase::toJson(j);

    // Then, serialize the FmhaOptions-specific members.
    TO_JSON(mAtol);
    TO_JSON(mAttentionWindowSize);
    TO_JSON(mBatchSize);
    TO_JSON(mChecksResults);
    TO_JSON(mChunkedAttentionSize);
    TO_JSON(mDryRun);
    TO_JSON(mEnablesAutoTuner);
    TO_JSON(mIsExportingCubin);
    TO_JSON(mIsTracing);
    TO_JSON(mMaxNumCtasPerSeqKv);
    TO_JSON(mMaxNumCtasQ);
    TO_JSON(mMaxNumCtasKv);
    TO_JSON(mMaxNumPagesPerSeqKv);
    TO_JSON(mMaxSeqLenQ);
    TO_JSON(mMaxSeqLenKv);
    TO_JSON(mMinFirstSparseMaskOffsetKv);
    TO_JSON(mMinSeqLenQ);
    TO_JSON(mMinSeqLenKv);
    TO_JSON(mMinSparseMlaTopK);
    TO_JSON(mNumBenchmarkSteps);
    TO_JSON(mNumCtasPerSeqKv);
    TO_JSON(mNumLoopItersForPrint);
    TO_JSON(mNumPagesInMemPool);
    TO_JSON(mNumSpecDecodingTokens);
    TO_JSON(mNumWarmUpSteps);
    TO_JSON(mMaxNumWavesForCtasKvMode);
    TO_JSON(mOutputScale);
    TO_JSON(mRtol);
    TO_JSON(mSkipsKernelGen);
    TO_JSON(mSkipSoftmaxThresholdScaleFactor);
    TO_JSON(mSparseAttnTopK);
    TO_JSON(mSumOfSeqLensQ);
    TO_JSON(mSumOfSeqLensKv);
    TO_JSON(mUsesSharedPagedKvIdx);
    TO_JSON(mVerbosity);
  }

#undef TO_JSON
};

////////////////////////////////////////////////////////////////////////////////////////////////////
struct FmhaOptionsFromArgs {
  // Attention window size.
  bool mIsAttentionWindowSizeSet{false};
  // Relative error tolerance.
  bool mIsAtolSet{false};
  // The head dimension per stage for Kv.
  bool mIsHeadDimPerStageKvSet{false};
  // The head dimension for Q and K.
  bool mIsHeadDimQkSet{false};
  // The head dimension for Q and K.
  bool mIsHeadDimVSet{false};
  // Whether to interleave MUFU and sums.
  bool mIsInterleavesMufuAndSumsSet{false};
  // Whether to use interleaved SF layout for tileV. (Only valid when KV cache is NVFP4).
  bool mIsInterleaveSfVSet{false};
  // The MMA order.
  bool mIsMmaOrderSet{false};
  // The number of delayed cvt elts.
  bool mIsNumDelayedCvtEltsSet{false};
  // The number of instances for Kv.
  bool mIsNumInstsKvSet{false};
  // The number of instances for Q.
  bool mIsNumInstsQSet{false};
  // The number of leading exp elts.
  bool mIsNumLeadingExpEltsSet{false};
  // The number of prefetched fmas.
  bool mIsNumPrefetchedFmasSet{false};
  // Relative error tolerance.
  bool mIsRtolSet{false};
  // Tile scheduler type.
  bool mIsTileSchedulerSet{false};
  // Whether to use an ordered sequence between softmax0 and softmax1.
  bool mIsUsesOrderedSequenceSet{false};
  // The tileSize for Q.
  bool mIsTileSizeQSet{false};
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// FmhaConfig
//
////////////////////////////////////////////////////////////////////////////////////////////////////
struct FmhaConfig {
  // The generated configuration JSON string.
  std::string mGenCfgJsonStr{""};
  // The function name of the kernel.
  std::string mFunctionName{""};
  // The execution path of the kernel.
  char const* mExecPath{nullptr};
  // The CUDA runner.
  tg::CudaRunner* mCudaRunner{nullptr};
  // The GenCfg object.
  tg::GenCfg* mGenCfg{nullptr};
  // The number of threads per CTA.
  int32_t mCtaDim{0};
  // The grid dimensions.
  tg::CudaRunner::Grid mGrid;
  // The cluster dimensions.
  tg::CudaRunner::Cluster mCluster;
  // The Fmha options.
  FmhaOptions mOptions{};
  // The CUDA architecture.
  tg::CudaArch mSm{tg::CudaArch::Sm100a};
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Check if the options are valid or not.
inline void checkFmhaOptions(FmhaOptions const& options,
                             FmhaOptionsFromArgs const& optionsFromArgs) {

  TLLM_CHECK_ERROR(!(options.mGroupsHeadsQ && isPackedQkv(options.mQkvLayout)),
                   "Grouping Q heads doesn't work with the packedQkv layout");
  // Only mNumInstsQ = 2, mNumInstsKv = 1 or mNumInstsQ = 1, mNumInstsKv = 2 or mNumInstsQ == 1,
  // mNumInstsKv == 1 are supported.
  TLLM_CHECK_ERROR(((options.mNumInstsQ * options.mNumInstsKv) <= 2),
                   "Only two tile instances are supported");

  // The number of instances for Q and Kv must be set together.
  TLLM_CHECK_ERROR(optionsFromArgs.mIsNumInstsQSet == optionsFromArgs.mIsNumInstsKvSet,
                   "The number of instances for Q and Kv must be set together");

  // Do we swap A/B for the generation kernel.
  bool const swapsMmaAb{isSwapsMmaAbForGenerationKernel(options.mFmhaKernelType)};
  // Check if tileSizeQ is valid.
  if (swapsMmaAb) {
    // NumHeadsQ can be distributed to multiple CTAs for MQA kernels.
    TLLM_CHECK_ERROR(options.mNumHeadsKv == 1 || options.mNumHeadsQPerKv <= 64,
                     "The number of heads per group cannot exceed 64");
  }

  // Check if the tileSizeKv is valid or not.
  int32_t tileSizeKv = options.mTileSizeKv;
  if (swapsMmaAb) {
    TLLM_CHECK_ERROR(tileSizeKv == 128 || tileSizeKv == 64,
                     "The SwapsMmaAbForGeneration kernels only support tileSizeKv 64 or 128");
  } else {
    // TileSizeKv must be a power of 2.
    bool const isPowOf2{(tileSizeKv & (tileSizeKv - 1)) == 0};
    int const maxTileSizeKv = tg::isArchHopper(options.mCudaArch) ? 256 : 128;
    TLLM_CHECK_ERROR(tileSizeKv >= 16 && tileSizeKv <= maxTileSizeKv && isPowOf2, "Not supported");
  }

  // Check if head dim is valid.
  auto headDimQk{options.mHeadDimQk}, headDimV{options.mHeadDimV};
  if (swapsMmaAb && headDimQk == headDimV) {
    TLLM_CHECK_ERROR(headDimQk == 64 || headDimQk == 80 || headDimQk == 128 || headDimQk == 256 ||
                       headDimQk == 512,
                     "The headDim must be 64, 80, 128, 256 or 512");
  }
  // MLA kernels.
  if (headDimQk != headDimV) {
    if (tg::isArchHopper(options.mCudaArch)) {
      TLLM_CHECK_ERROR(headDimQk > headDimV,
                       "Only headDimQk > headDimV MLA kernels have been verified for Hopper");
    } else {
      if (isContextKernel(options.mFmhaKernelType)) {
        TLLM_CHECK_ERROR(headDimQk == 192 && headDimV == 128,
                         "Only headDimQk = 192, headDimV = 128 MLA kernels have been verified");
      } else {
        TLLM_CHECK_ERROR(options.mIsMlaGen && ((headDimQk == 576 && headDimV == 512) ||
                                               (headDimQk == 320 && headDimV == 256)),
                         "Only headDimQk = 576, headDimV = 512 or headDimQk = 320, headDimV = 256 "
                         "MLA kernels have been verified");
      }
    }
  }

  // Check if headDimPerCtaV is valid.
  if (options.mHeadDimPerCtaV != options.mHeadDimV && options.mHeadDimPerCtaV != 0 &&
      options.mHeadDimPerStageKv != 0) {
    // The number of elements per CTA must be a multiple of the number of elements per stage.
    TLLM_CHECK_ERROR(options.mHeadDimPerCtaV % options.mHeadDimPerStageKv == 0,
                     "mHeadDimPerStageKv=",
                     options.mHeadDimPerStageKv,
                     " must divide mHeadDimPerCtaV=",
                     options.mHeadDimPerCtaV);
    // The head dimension must be a multiple of the number of elements per CTA.
    TLLM_CHECK_ERROR(options.mHeadDimV % options.mHeadDimPerCtaV == 0,
                     "mHeadDimPerCtaV=",
                     options.mHeadDimPerCtaV,
                     " must divide mHeadDimV=",
                     options.mHeadDimV);
  }

  // Check if headDimPerStageKv is valid.
  if (swapsMmaAb && (options.mHeadDimQk > 128 || options.mHeadDimV > 128) &&
      tg::isArchBlackwell(options.mCudaArch) && optionsFromArgs.mIsHeadDimPerStageKvSet) {
    TLLM_CHECK_ERROR(
      options.mHeadDimPerStageKv == 128,
      "HeadDimPerStageKv must be 128 for swapsMmaAbForGeneration kernels on Blackwell");
  }

  // Make sure the 2Cta option is valid.
  if (options.mClusterDimX == 2) {
    // Note that the tileSizeQ and tileSizeKv should be the tileSizes of D in BMM1.
    TLLM_CHECK_ERROR(options.mTileSizeQ == 64 || options.mTileSizeQ == 128,
                     "The tileSizeQ must be 64 or 128 for 2Cta option");
    TLLM_CHECK_ERROR(options.mTileSizeKv == 128 || options.mTileSizeKv == 256,
                     "The tileSizeKv must be 128 or 256 for 2Cta option");
  }

  // Make sure numSpecDecodingTokens is only enabled with causal-mask/custom-mask spec-decoding
  // kernels.
  if (options.mNumSpecDecodingTokens > 1) {
    // Make sure it is causal-mask spec-decoding kernel.
    TLLM_CHECK_ERROR(
      options.mIsCausalSpecDecodingGen || options.mIsCustomSpecDecodingGen,
      "The numSpecDecodingTokens > 1 requires causal-mask/custom-mask spec-decoding kernels");
    // Make sure it is not a context-phase kernel.
    TLLM_CHECK_ERROR(!isContextKernel(options.mFmhaKernelType),
                     "The numSpecDecodingTokens > 1 requires generation-phase kernels");
  }

  // Make sure ReuseSMemKForV option is valid.
  if (options.mReuseSmemKForV) {
    TLLM_CHECK_ERROR(options.mIsMlaGen && swapsMmaAb,
                     "Only MlaGen kernels support reusing smemK for V.");
    TLLM_CHECK_ERROR(
      options.mHeadDimV == options.mHeadDimPerCtaV || options.mHeadDimPerCtaV == 0,
      "Spliting headDimV across multiple CTAs doesn't work with reusing smemK for V.");
    TLLM_CHECK_ERROR((tg::dtypeGetNumBits(options.mDtypeK) * options.mTileSizeKv) <=
                       8 * 128 /*16*64*/,
                     "The shared memory size is not sufficient to support reusing smemK for V. "
                     "Consider using smaller tileSizeKv.");
  }

#ifdef TLLM_PUBLIC_RELEASE
  if (options.mDtypeKv == tg::Dtype::E2m1 && !options.mIsTrtllmLayout) {
    TLLM_CHECK_ERROR(false, "E2m1 KV cache is not supported with public compiler.");
  }
#endif // TLLM_PUBLIC_RELEASE

  // PackedQkv layout does not support supportsDiffSeqLensForQAndKv.
  TLLM_CHECK_ERROR(!(isPackedQkv(options.mQkvLayout) && options.mSupportsDiffSeqLensForQAndKv),
                   "PackedQkv layout does not support supportsDiffSeqLensForQAndKv");
  // Q does not support E2m1 dtype.
  TLLM_CHECK_ERROR(options.mDtypeQ != tg::Dtype::E2m1, "Q does not suppot E2m1 dtype");
  // Make sure correct attention window size is set.
  TLLM_CHECK_ERROR(!isSlidingOrChunkedCausalMask(options.mMaskType) ||
                     options.mAttentionWindowSize > 0 || options.mChunkedAttentionSize > 0,
                   "Please set correct sliding attention window size or chunked attention size");
  if (options.mChunkedAttentionSize > 0) {
    TLLM_CHECK_ERROR(options.mAttentionWindowSize >= options.mMaxSeqLenKv,
                     "sliding attention window size must be greater than or equal to maxSeqLenKv");
    TLLM_CHECK_ERROR(options.mChunkedAttentionSize % (options.mTileSizeKv * options.mNumInstsKv) ==
                       0,
                     "Chunked attention size must be a multiple of the tileSizePerCtaKv");
    TLLM_CHECK_ERROR((options.mChunkedAttentionSize & (options.mChunkedAttentionSize - 1)) == 0,
                     "Chunked attention size must be power of 2");
  }

  // Special options for FP4.
  if (options.mDtypeOut == tg::Dtype::E2m1) {
    // FP4 output only supports fuseEpilogueIntoCorr.
    TLLM_CHECK_ERROR(options.mFuseEpilogueIntoCorr,
                     "FP4 output only supports fuseEpilogueIntoCorr");
    // Make sure the number of sf per row can be divided by 4, required for interleaved SF layout.
    // Details can be seen in DtypeUtils.h: E2m1Utils::getSfOffset.
    int32_t hiddenDim = options.mNumHeadsQ * headDimQk;
    auto kernelTraits = getKernelTraitsFromOptions(options);
    TLLM_CHECK_ERROR((hiddenDim / kernelTraits.mNumEltsPerSf) % 4 == 0,
                     "Current hiddenDim is not supported for FP4 output");
  }

  // If we decide to use Sage Attention, the number of elements per block must be a power-of-two.
  if (options.mNumEltsPerSageAttnBlkQ != 0) {
    int numEltsPerBlk{options.mNumEltsPerSageAttnBlkQ};
    TLLM_CHECK_ERROR((numEltsPerBlk & (numEltsPerBlk - 1)) == 0,
                     "mNumEltsPerSageAttnBlkQ=",
                     options.mNumEltsPerSageAttnBlkQ,
                     " must be a power-of-two (or 0)");
  }
  if (options.mNumEltsPerSageAttnBlkK != 0) {
    int numEltsPerBlk{options.mNumEltsPerSageAttnBlkK};
    TLLM_CHECK_ERROR((numEltsPerBlk & (numEltsPerBlk - 1)) == 0,
                     "mNumEltsPerSageAttnBlkK=",
                     options.mNumEltsPerSageAttnBlkK,
                     " must be a power-of-two (or 0)");
  }
  if (options.mNumEltsPerSageAttnBlkP != 0) {
    int numEltsPerBlk{options.mNumEltsPerSageAttnBlkP};
    TLLM_CHECK_ERROR((numEltsPerBlk & (numEltsPerBlk - 1)) == 0,
                     "mNumEltsPerSageAttnBlkP=",
                     options.mNumEltsPerSageAttnBlkP,
                     " must be a power-of-two (or 0)");
  }
  if (options.mNumEltsPerSageAttnBlkV != 0) {
    int numEltsPerBlk{options.mNumEltsPerSageAttnBlkV};
    TLLM_CHECK_ERROR((numEltsPerBlk & (numEltsPerBlk - 1)) == 0,
                     "mNumEltsPerSageAttnBlkV=",
                     options.mNumEltsPerSageAttnBlkV,
                     " must be a power-of-two (or 0)");
  }

  // The CGA reduction.
  if (isCgaSmemReduction(options.mMultiCtasKvMode)) {
    TLLM_CHECK_ERROR(options.mTileScheduler == TileScheduler::Static,
                     "CGA reduction is only supported with static tile scheduler.");
  }

  // Make sure block sparse attention is only enabled with paged Kv layout.
  if (options.mUseBlockSparseAttention) {
    TLLM_CHECK_ERROR(isPagedKv(options.mQkvLayout),
                     "Block sparse attention is only supported with paged Kv layout.");
  }

  // Currently, for performance reason, only numTokensPerPage >= tileSizeKv is supported for
  // dynamic numTokensPerPage.
  if (options.mDynamicNumTokensPerPage) {
    TLLM_CHECK_ERROR(options.mNumTokensPerPage >= options.mTileSizeKv,
                     "NumTokensPerPage must be larger than or equal to tileSizeKv");
  }

  // Make sure the multiCtasKvMode is valid if numCtasPerSeqKv is set.
  if (options.mNumCtasPerSeqKv > 1) {
    TLLM_CHECK_ERROR(!isDisabled(options.mMultiCtasKvMode),
                     "Please set the correct multiCtasKvMode for numCtasPerSeqKv > 1.");
  }

  // The sparse attention kernels.
  if (isTokenSparse(options.mSparseType)) {
    TLLM_CHECK_ERROR(isPagedKv(options.mQkvLayout),
                     "PagedKv layout is required for sparse attention kernels.");
    TLLM_CHECK_ERROR(
      options.mSparseAttnTopK % 4 == 0,
      "SparseAttnTopK must be a multiple of 4 in order to use 16bytes cpAsync loads");
  }
  if (options.mHasSlidingWindowKvPool) {
    TLLM_CHECK_ERROR(
      supportsVarSparseMlaTopKLens(options),
      "The sliding-window KV pool is only supported by dynamic-token sparse MLA kernels.");
    TLLM_CHECK_ERROR(options.mSingleTokenQPerCta,
                     "mSingleTokenQPerCta must be true when sliding-window KV pool is enabled.");
    TLLM_CHECK_ERROR(options.mAttentionWindowSize == options.mTileSizeKv,
                     "attentionWindowSize must equal tileSizeKv when sliding-window KV pool is "
                     "enabled.");
  }

  // Always enable skipsSoftmaxWhenPossible for outputSkipSoftmaxStats.
  if (options.mOutputSkipSoftmaxStats) {
    TLLM_CHECK_ERROR(options.mSkipsSoftmaxWhenPossible,
                     "The outputSkipSoftmaxStats option requires skipsSoftmaxWhenPossible to be "
                     "enabled.");
  }

  // For headDim 256, mixed precision Qkv or tileSizeQ = 128 requires numInstsQ == 1 and numInstsKv
  // == 1.
  if (options.mHeadDimQk == 256 && options.mHeadDimV == 256 && optionsFromArgs.mIsNumInstsQSet &&
      optionsFromArgs.mIsNumInstsKvSet) {
    if (options.mDtypeQ != options.mDtypeK ||
        (optionsFromArgs.mIsTileSizeQSet && options.mTileSizeQ == 128)) {
      TLLM_CHECK_ERROR(options.mNumInstsQ == 1 && options.mNumInstsKv == 1,
                       "For headDim 256, mixed precision (dtypeQ != dtypeK) or tileSizeQ = 128 "
                       "requires numInstsQ == "
                       "1 and numInstsKv == 1.");
    }
  }

  // The mGroupsTokensHeadsQ only works with GQA generation kernels.
  if (options.mGroupsTokensHeadsQ) {
    TLLM_CHECK_ERROR(!isContextKernel(options.mFmhaKernelType),
                     "mGroupsTokensHeadsQ should only be enabled for generation kernels.");
    TLLM_CHECK_ERROR(options.mDtypeKv != tg::Dtype::E2m1,
                     "mGroupsTokensHeadsQ doesn't work with E2m1 dtypeKv.");
    TLLM_CHECK_ERROR(!options.mIsMlaGen,
                     "MLA gen kernels haven't supported mGroupsTokensHeadsQ yet.");
  }



  // For transformed K/V, MmaOrder must be Pv0_Qk0_Pv1_Qk1.
  if (options.mDtypeQ != options.mDtypeKv) {
    TLLM_CHECK_ERROR(options.mMmaOrder == MmaOrder::Pv0_Qk0_Pv1_Qk1,
                     "Only MMA order Pv0_Qk0_Pv1_Qk1 is supported for transformed K/V.");
  }

  if (options.mMmaOrder == MmaOrder::Qk0_Qk1_Pv0_Pv1) {
    TLLM_CHECK_ERROR(options.mNumInstsQ == 2,
                     "MMA order Qk0_Qk1_Pv0_Pv1 is only supported with numInstsQ=2.");
  }
  if (options.mMmaOrder == MmaOrder::Qk0_Pv0_Qk1_Pv1) {
    TLLM_CHECK_ERROR(
      options.mNumInstsQ == 2 || options.mNumInstsKv == 2,
      "MMA order Qk0_Pv0_Qk1_Pv1 is only supported with numInstsQ=2 or numInstsKv=2.");
    TLLM_CHECK_ERROR(!isKeepsMmaAbForGenerationKernel(options.mFmhaKernelType),
                     "MmaOrder Qk0_Pv0_Qk1_Pv1 is not supported with "
                     "keepsMmaAbForGeneration kernels.");
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Update the fmha options if needed.
inline void updateFmhaOptions(FmhaOptions& options, FmhaOptionsFromArgs const& optionsFromArgs) {
  // Set default absolute/relative tolerance for different data types.
  if ((options.mDtypeQ == tg::Dtype::Fp16) || (options.mDtypeQ == tg::Dtype::Bfloat16)) {
    // Use smaller tolerance for float16/bfloat16 if it is not set.
    if (options.mDtypeOut == tg::Dtype::E4m3) {
      if (!optionsFromArgs.mIsAtolSet) {
        options.mAtol = 2e-2f;
      }
      if (!optionsFromArgs.mIsRtolSet) {
        options.mRtol = 2e-3f;
      }
    } else if (options.mDtypeOut == tg::Dtype::E2m1) {
      if (!optionsFromArgs.mIsAtolSet) {
        options.mAtol = 0.15f;
      }
      if (!optionsFromArgs.mIsRtolSet) {
        options.mRtol = 0.01f;
      }
    } else {
      if (!optionsFromArgs.mIsAtolSet) {
        options.mAtol = 5e-3f;
      }
      if (!optionsFromArgs.mIsRtolSet) {
        options.mRtol = 1e-3f;
      }
    }
  } else if (options.mDtypeKv == tg::Dtype::E2m1) {
    if (!optionsFromArgs.mIsAtolSet) {
      options.mAtol = 0.3f;
    }
    if (!optionsFromArgs.mIsRtolSet) {
      options.mRtol = 0.1f;
    }
  } else if (options.mDtypeOut == tg::Dtype::E2m1) {
    if (!optionsFromArgs.mIsAtolSet) {
      options.mAtol = 0.15f;
    }
    if (!optionsFromArgs.mIsRtolSet) {
      options.mRtol = 0.1f;
    }
  }
  TLLM_LOG_TRACE("Reference atol = ", options.mAtol, ", rtol = ", options.mRtol, " is used.");

  //
  // Update sequence lengths parameters.
  //

  // Update maxSeqLenQ.
  if (options.mSingleTokenQ) {
    options.mMaxSeqLenQ = 1;
  }

  // Update minSeqLenQ, minSeqLenK.
  options.mMinSeqLenQ = std::min(options.mMinSeqLenQ, options.mMaxSeqLenQ);
  options.mMinSeqLenKv = std::min(options.mMinSeqLenKv, options.mMaxSeqLenKv);

  // Check if sequence lengths are valid.
  TLLM_CHECK_ERROR(options.mMinSeqLenQ > 0 && options.mMinSeqLenKv > 0,
                   "Invalid sequence lengths.");

  // Enable variable sequence if minSeqLenQ < maxSeqLenQ or minSeqLenKv < maxSeqLenKv.
  options.mSupportsVarSeqLens |=
    (options.mMinSeqLenQ < options.mMaxSeqLenQ) || (options.mMinSeqLenKv < options.mMaxSeqLenKv);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha
