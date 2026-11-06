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

#include "Dsv4Constants.h"
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
  // Batch size.
  int mBatchSize{2};
  // Whether to verify the correctness. 0: No check, 1: partial, 2: full.
  int mChecksResults{2};
  // The chunked attention size. A value > 0 enables chunked attention under the shared
  // SlidingOrChunkedCausal mask type.
  int32_t mChunkedAttentionSize{0};
  // Dry-run: print a log but does not actually generate anything
  bool mDryRun{false};
  // Token dimension reserved by the DSv4 FP32 or packed UE8M0 scale tensor. May be padded.
  int32_t mDsv4ScaleBufM{0};
  // Enable the auto tuner.
  bool mEnablesAutoTuner{false};
  // Select the grouped MLA generation kernel in the auto tuner.
  bool mSelectsGroupedMla{false};
  // Enable the BF16Q+FP8KV K-only transform path. Disabled by default.
  bool mEnablesBf16QFp8KvKOnlyTransform{false};
  // Whether is exporting cubin.
  bool mIsExportingCubin{false};
  // Whether the kernel is under tracing
  bool mIsTracing{false};
  // Whether running inside TRTLLM (affects KV stride computation for MLA).
  bool mIsTrtllmLayout{false};
  // Sliding-window left bound. -1 means unbounded left; otherwise each Q row can attend to K
  // positions no earlier than q - mLeftSlidingWindow, clamped to the sequence length.
  int32_t mLeftSlidingWindow{-1};
  // The maximum number of CTAs for K/V.
  int mMaxNumCtasKv{1};
  // The maximum number of CTAs per sequenceKv (multiCtasKvMode).
  // This is used to limit the number of CTAs per sequenceKv for the multiCtasKvMode.
  int mMaxNumCtasPerSeqKv{INT_MAX};
  // The maximum number of CTAs for Q.
  int mMaxNumCtasQ{1};
  // The maximum number of pages per sequence in the paged-kv buffer.
  int mMaxNumPagesPerSeqKv{512 / 32};
  // The maximum number of waves for the multiCtasKvMode.
  int mMaxNumWavesForCtasKvMode{1};
  // Sequence length for K/V.
  int mMaxSeqLenKv{512};
  // Sequence length for Q.
  int mMaxSeqLenQ{512};
  // The minimum first sparseMask offset in the Kv sequence dimension.
  // Default 0 means that all tokensKv need custom masking.
  int mMinFirstSparseMaskOffsetKv{0};
  // The minimum sequence length (used to generate variable Kv sequence length).
  int mMinSeqLenKv{INT_MAX};
  // The minimum sequence length (used to generate variable Q sequence length).
  int mMinSeqLenQ{INT_MAX};
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
  // The attention output scale.
  float mOutputScale{1.f};
  // Sliding-window right bound. -1 means unbounded right before option validation; otherwise each Q
  // row can attend to K positions no later than q + mRightSlidingWindow, clamped to the sequence.
  int32_t mRightSlidingWindow{-1};
  // Relative error tolerance.
  float mRtol{1e-1f};
  // Whether to skip kernel generation (for debug purpose).
  bool mSkipsKernelGen{false};
  // The threshold to skip softmax operations when possible according to the below expression.
  float mSkipSoftmaxThresholdScaleFactor{0};
  // The topK value for sparse attention kernels.
  int mSparseAttnTopK{2048};
  // For tree-based custom spec-decoding only: equals max_total_draft_tokens + 1,
  // fixed at config time. When set with mIsCustomSpecDecodingGen, FmhaAutoTuner
  // uses it as a deterministic upper bound for kernel selection.
  int mSpecDecodingTargetMaxGenLen{0};
  // The sum of sequence lengths for K/V.
  int mSumOfSeqLensKv{512 * 2};
  // The sum of sequence lengths for Q.
  int mSumOfSeqLensQ{512 * 2};
  // Whether the indices for K & V pages are shared as unified index (vLLM/FlashInfer).
  bool mUsesSharedPagedKvIdx{false};
  // Select the 2Qx1KV grouped-token schedule for GQA generation (decode) kernels. A fully
  // populated 2Q tile halves the CTA count, which wins when the 2Q grid fits in one wave and the
  // 1Q grid does not; the caller (e.g. FlashInfer's kernel selection) owns that shape/occupancy
  // heuristic and sets this flag.
  bool mUses2InstsQDecodeKernels{false};
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
    TO_JSON(mBatchSize);
    TO_JSON(mChecksResults);
    TO_JSON(mChunkedAttentionSize);
    TO_JSON(mDryRun);
    TO_JSON(mDsv4ScaleBufM);
    TO_JSON(mEnablesAutoTuner);
    TO_JSON(mSelectsGroupedMla);
    TO_JSON(mEnablesBf16QFp8KvKOnlyTransform);
    TO_JSON(mIsExportingCubin);
    TO_JSON(mIsTracing);
    TO_JSON(mLeftSlidingWindow);
    TO_JSON(mMaxNumCtasKv);
    TO_JSON(mMaxNumCtasPerSeqKv);
    TO_JSON(mMaxNumCtasQ);
    TO_JSON(mMaxNumPagesPerSeqKv);
    TO_JSON(mMaxNumWavesForCtasKvMode);
    TO_JSON(mMaxSeqLenKv);
    TO_JSON(mMaxSeqLenQ);
    TO_JSON(mMinFirstSparseMaskOffsetKv);
    TO_JSON(mMinSeqLenKv);
    TO_JSON(mMinSeqLenQ);
    TO_JSON(mMinSparseMlaTopK);
    TO_JSON(mNumBenchmarkSteps);
    TO_JSON(mNumCtasPerSeqKv);
    TO_JSON(mNumLoopItersForPrint);
    TO_JSON(mNumPagesInMemPool);
    TO_JSON(mNumSpecDecodingTokens);
    TO_JSON(mNumWarmUpSteps);
    TO_JSON(mOutputScale);
    TO_JSON(mRightSlidingWindow);
    TO_JSON(mRtol);
    TO_JSON(mSkipsKernelGen);
    TO_JSON(mSkipSoftmaxThresholdScaleFactor);
    TO_JSON(mSparseAttnTopK);
    TO_JSON(mSpecDecodingTargetMaxGenLen);
    TO_JSON(mSumOfSeqLensKv);
    TO_JSON(mSumOfSeqLensQ);
    TO_JSON(mUsesSharedPagedKvIdx);
    TO_JSON(mUses2InstsQDecodeKernels);
    TO_JSON(mVerbosity);
  }

#undef TO_JSON
};

////////////////////////////////////////////////////////////////////////////////////////////////////
struct FmhaOptionsFromArgs {
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
  // Whether to use separate transformed K/V resources.
  bool mIsSeparateTransformedKvSet{false};
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

// Whether the output dtype produces per-block scale factors.
inline bool hasOutputSfs(tg::Dtype dtype) {
  return dtype == tg::Dtype::E2m1 || dtype == tg::Dtype::MxE4m3;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Whether token-sparse attention was requested together with window parameters it cannot support
// (sliding-window mask, chunked attention, or VariableWindow).
inline bool hasTokenSparseUnsupportedWindowParams(FmhaOptions const& options) {
  return isAnySlidingWindowMask(options.mMaskType) || options.mLeftSlidingWindow != -1 ||
         options.mRightSlidingWindow != -1 || options.mChunkedAttentionSize > 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Check whether the runtime skip-correction threshold is valid for the selected kernel.
inline void checkSkipCorrThreshold(FmhaOptions const& options, float skipCorrThreshold) {
  TLLM_CHECK_ERROR(skipCorrThreshold >= 0.f, "skipCorrThreshold must be non-negative.");
  TLLM_CHECK_ERROR(skipCorrThreshold <= 0.f || options.mSkipsCorrWhenPossible,
                   "A positive skipCorrThreshold requires skipsCorrWhenPossible.");

  if (skipCorrThreshold > 0.f) {
    auto const dtypeBmm2 = getDtypeBmm2(options);
    TLLM_CHECK_ERROR(dtypeBmm2 != tg::Dtype::E2m1,
                     "Threshold-based correction skipping does not support E2M1 BMM2.");
    if (dtypeBmm2 == tg::Dtype::E4m3) {
      // A skipped correction can increase the scaled P value by up to 2^threshold. The
      // threshold-enabled P scale is 1.75, and 1.75 * 2^8 == 448, the E4M3 maximum, making 8 the
      // recommended threshold.
      TLLM_CHECK_ERROR(skipCorrThreshold <= 8.f, "skipCorrThreshold must be <= 8 for E4M3 BMM2.");
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Check if the options are valid or not. Pure validation, does not mutate options.
inline void checkFmhaOptions(FmhaOptions const& options,
                             FmhaOptionsFromArgs const& optionsFromArgs) {

  bool const isSlidingOrChunked{isSlidingOrChunkedCausalMask(options.mMaskType)};
  bool const isSlidingWindowCustom{isSlidingWindowCustomMask(options.mMaskType)};
  bool const isChunkedAttention{options.mChunkedAttentionSize > 0};
  bool const hasSlidingOrChunkedParams{options.mLeftSlidingWindow != -1 ||
                                       options.mRightSlidingWindow != -1 ||
                                       isChunkedAttention};
  if (isTokenSparse(options.mSparseType)) {
    TLLM_CHECK_ERROR(!hasTokenSparseUnsupportedWindowParams(options),
                     "Token-sparse attention does not support sliding-window, chunked, or "
                     "VariableWindow mask parameters.");
  }

  bool const hasValidSlidingOrChunkedMaskType{
    !hasSlidingOrChunkedParams || isSlidingOrChunked || isSlidingWindowCustom};
  TLLM_CHECK_ERROR(hasValidSlidingOrChunkedMaskType,
                   "Sliding window or chunked attention parameters are only supported with "
                   "SlidingOrChunkedCausal or SlidingWindowCustom mask type.");
  TLLM_CHECK_ERROR(options.mChunkedAttentionSize >= 0,
                   "chunkedAttentionSize must be >= 0.");

  // Validate chunked attention. In this mode, mChunkedAttentionSize is the exact chunk size
  // including the current token. Do not also pass left/right sliding-window bounds.
  if (isChunkedAttention) {
    bool const hasSlidingWindowBounds{options.mLeftSlidingWindow != -1 ||
                                      options.mRightSlidingWindow != -1};
    TLLM_CHECK_ERROR(!hasSlidingWindowBounds,
                     "Chunked attention uses mChunkedAttentionSize; do not also set "
                     "leftSlidingWindow or rightSlidingWindow.");

    bool const isChunkedAttentionActuallyCausal{
      options.mMaxSeqLenKv > 0 && options.mChunkedAttentionSize >= options.mMaxSeqLenKv};
    TLLM_CHECK_ERROR(!isChunkedAttentionActuallyCausal,
                     "Chunked attention with mChunkedAttentionSize >= mMaxSeqLenKv is Causal; "
                     "use -maskType Causal.");
  }

  // Validate non-chunked SlidingOrChunkedCausal. In this mode, mLeftSlidingWindow and
  // mRightSlidingWindow are sliding reaches that exclude the current token, so the attended K
  // range is [q - leftSlidingWindow, q + rightSlidingWindow].
  if (isSlidingOrChunked && !isChunkedAttention) {
    bool const isSlidingOrChunkedActuallyDense{
      options.mLeftSlidingWindow == -1 && options.mRightSlidingWindow == -1};
    TLLM_CHECK_ERROR(!isSlidingOrChunkedActuallyDense,
                     "SlidingOrChunkedCausal with unbounded left and right sides is Dense; "
                     "use -maskType Dense.");

    bool const isSlidingOrChunkedActuallyCausal{
      options.mLeftSlidingWindow == -1 && options.mRightSlidingWindow == 0};
    TLLM_CHECK_ERROR(!isSlidingOrChunkedActuallyCausal,
                     "SlidingOrChunkedCausal with unbounded left and rightSlidingWindow = 0 is "
                     "Causal; use -maskType Causal.");

    bool const isSlidingOrChunkedEffectivelyCausal{
      options.mMaxSeqLenKv > 0 && options.mLeftSlidingWindow >= options.mMaxSeqLenKv - 1 &&
      options.mRightSlidingWindow == 0};
    TLLM_CHECK_ERROR(!isSlidingOrChunkedEffectivelyCausal,
                     "SlidingOrChunkedCausal with leftSlidingWindow >= mMaxSeqLenKv - 1 and "
                     "rightSlidingWindow = 0 is Causal; use -maskType Causal.");

    bool const isSlidingOrChunkedEffectivelyDense{
      options.mMaxSeqLenKv > 0 && options.mLeftSlidingWindow >= options.mMaxSeqLenKv - 1 &&
      options.mRightSlidingWindow > 0 && options.mRightSlidingWindow >= options.mMaxSeqLenKv - 1};
    TLLM_CHECK_ERROR(!isSlidingOrChunkedEffectivelyDense,
                     "SlidingOrChunkedCausal with leftSlidingWindow and rightSlidingWindow both "
                     ">= mMaxSeqLenKv - 1 is Dense; use -maskType Dense.");

    bool const hasInvalidLeftSlidingWindow{options.mLeftSlidingWindow < -1};
    TLLM_CHECK_ERROR(!hasInvalidLeftSlidingWindow,
                     "leftSlidingWindow must be >= -1 for SlidingOrChunkedCausal.");

    bool const hasOversizedLeftSlidingWindow{options.mLeftSlidingWindow > options.mMaxSeqLenKv};
    TLLM_CHECK_ERROR(!hasOversizedLeftSlidingWindow,
                     "leftSlidingWindow must be <= mMaxSeqLenKv for SlidingOrChunkedCausal.");

    bool const hasInvalidRightSlidingWindow{options.mRightSlidingWindow < 0};
    TLLM_CHECK_ERROR(!hasInvalidRightSlidingWindow,
                     "rightSlidingWindow must be >= 0 for SlidingOrChunkedCausal.");

    bool const isGenerationSlidingOrChunked{!isContextKernel(options.mFmhaKernelType)};
    TLLM_CHECK_ERROR(!isGenerationSlidingOrChunked || options.mRightSlidingWindow == 0,
                     "SlidingOrChunkedCausal generation kernels require rightSlidingWindow = 0.");

    bool const hasOversizedRightSlidingWindow{
      options.mRightSlidingWindow > options.mMaxSeqLenKv};
    TLLM_CHECK_ERROR(!hasOversizedRightSlidingWindow,
                     "rightSlidingWindow must be <= mMaxSeqLenKv for SlidingOrChunkedCausal.");
  }

  // SlidingWindowCustom is the generation custom-mask mode with an analytic left sliding-window
  // bound. The right/tree visibility still comes from the packed custom mask.
  if (isSlidingWindowCustom) {
    TLLM_CHECK_ERROR(!isContextKernel(options.mFmhaKernelType),
                     "SlidingWindowCustom is only supported with generation kernels.");
    TLLM_CHECK_ERROR(!isChunkedAttention,
                     "SlidingWindowCustom requires sliding-window bounds without chunked attention.");
    TLLM_CHECK_ERROR(options.mLeftSlidingWindow >= 0 && options.mRightSlidingWindow == 0,
                     "SlidingWindowCustom requires leftSlidingWindow >= 0 and "
                     "rightSlidingWindow = 0.");
  }

  // VariableWindow extra constraints.
  if (isVariableWindowMask(options.mMaskType)) {
    TLLM_CHECK_ERROR(isContextKernel(options.mFmhaKernelType),
                     "VariableWindow is only supported with context (prefill) kernels.");
    TLLM_CHECK_ERROR(!options.mGroupsTokensHeadsQ,
                     "VariableWindow does not support groupsTokensHeadsQ because "
                     "mGroupsTokensHeadsQ is only supported by generation kernels.");
  }

  TLLM_CHECK_ERROR(!(options.mGroupsHeadsQ && isPackedQkv(options.mQkvLayout)),
                   "Grouping Q heads doesn't work with the packedQkv layout");
  // Only mNumInstsQ = 2, mNumInstsKv = 1 or mNumInstsQ = 1, mNumInstsKv = 2 or mNumInstsQ == 1,
  // mNumInstsKv == 1 are supported.
  TLLM_CHECK_ERROR(((options.mNumInstsQ * options.mNumInstsKv) <= 2),
                   "Only two tile instances are supported");
  if (isBf16QFp8KvFullTransformGeneration(options)) {
    TLLM_CHECK_ERROR(options.mNumInstsQ == 1 && options.mNumInstsKv == 1,
                     "BF16Q+FP8KV full-transform kernels require numInstsQ == 1 and "
                     "numInstsKv == 1.");
  }

  if (options.mFusesDsv4InvRopeFp8Quant) {
    bool const isSpecDecTree =
      options.mIsCustomSpecDecodingGen && options.mSpecDecodingTargetMaxGenLen > 0;
    bool const isSupportedDsv4FusionConfig =
      options.mIsMlaGen && options.mSparseType == SparseType::DynamicTokenSparse &&
      options.mFuseEpilogueIntoCorr && isKeepsMmaAbForGenerationKernel(options.mFmhaKernelType) &&
      options.mQkvLayout == QkvLayout::PagedKv && options.mDtypeQ == tg::Dtype::E4m3 &&
      options.mDtypeK == tg::Dtype::E4m3 && options.mDtypeV == tg::Dtype::E4m3 &&
      options.mDtypeOut == tg::Dtype::E4m3 && options.mHeadDimQk == kDsv4HeadDimQk &&
      options.mHeadDimV == kDsv4HeadDimV && options.mHeadDimPerCtaV == kDsv4HeadDimPerCtaV &&
      options.mHeadDimPerCtaV * options.mClusterDimX == options.mHeadDimV &&
      options.mClusterDimX == 2 && options.mTileSizeQ == 64 && options.mTileSizeKv == 128 &&
      options.mNumInstsQ == 1 && options.mNumInstsKv == 1 && !options.mSwapsMmaAb &&
      options.mHeadDimPerStageKv == 0 && options.mMultiCtasKvMode == MultiCtasKvMode::Disabled &&
      !options.mUseBlockSparseAttention && !isSpecDecTree;
    TLLM_CHECK_ERROR(isSupportedDsv4FusionConfig,
                     "DSv4 inverse-RoPE FP8 quant fusion only supports the fixed DSv4 sparse MLA "
                     "generation keep-AB paged-KV E4M3 configuration with standard non-tree "
                     "causal generation/context position semantics.");
  }

  TLLM_CHECK_ERROR(!options.mUsesDsv4Ue8m0ScaleO || options.mFusesDsv4InvRopeFp8Quant,
                   "-dsv4ScaleFormat ue8m0 requires -fusesDsv4InvRopeFp8Quant true.");

  // The number of instances for Q and Kv must be set together.
  TLLM_CHECK_ERROR(optionsFromArgs.mIsNumInstsQSet == optionsFromArgs.mIsNumInstsKvSet,
                   "The number of instances for Q and Kv must be set together");

  TLLM_CHECK_ERROR(options.mNumStagesKv >= 0, "numStagesKv must be >= 0");
  TLLM_CHECK_ERROR(options.mNumStagesQ >= 0, "numStagesQ must be >= 0");

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
        TLLM_CHECK_ERROR((headDimQk == 192 && headDimV == 128) ||
                           (headDimQk == 128 && headDimV == 64),
                         "Only headDimQk = 192, headDimV = 128 (DeepSeek context MLA) or "
                         "headDimQk = 128, headDimV = 64 (Mistral Small 4 context MLA) kernels "
                         "have been verified");
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

  // PackedQkv layout does not support supportsDiffSeqLensForQAndKv.
  TLLM_CHECK_ERROR(!(isPackedQkv(options.mQkvLayout) && options.mSupportsDiffSeqLensForQAndKv),
                   "PackedQkv layout does not support supportsDiffSeqLensForQAndKv");
  // Q does not support E2m1 dtype.
  TLLM_CHECK_ERROR(options.mDtypeQ != tg::Dtype::E2m1, "Q does not suppot E2m1 dtype");
  if (isChunkedAttention) {
    int32_t const chunkSize = options.mChunkedAttentionSize;
    TLLM_CHECK_ERROR(chunkSize % (options.mTileSizeKv * options.mNumInstsKv) == 0,
                     "Chunked attention size must be a multiple of the tileSizePerCtaKv");
    TLLM_CHECK_ERROR((chunkSize & (chunkSize - 1)) == 0,
                     "Chunked attention size must be power of 2");
  }

  // Special options for block-scaled outputs.
  if (fmha::hasOutputSfs(options.mDtypeOut)) {
    TLLM_CHECK_ERROR(options.mFuseEpilogueIntoCorr,
                     "E2m1 / MxE4m3 output only supports fuseEpilogueIntoCorr");

    // Make sure the number of SFs per row can be divided by 4, required for interleaved SF layout.
    int32_t numEltsPerSfO = tg::dtypeNumEltsPerSf(options.mDtypeOut);
    int32_t hiddenDim = options.mNumHeadsQ * options.mHeadDimV;
    TLLM_CHECK_ERROR(options.mHeadDimV % numEltsPerSfO == 0,
                     "headDimV must be divisible by the output SF group size");
    TLLM_CHECK_ERROR(hiddenDim % numEltsPerSfO == 0,
                     "hiddenDim must be divisible by the output SF group size");
    TLLM_CHECK_ERROR((hiddenDim / numEltsPerSfO) % 4 == 0,
                     "Current hiddenDim is not compatible with interleaved SF layout");
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

  // groupsTokensHeadsQ is allowed for GQA gen and explicitly selected MLA gen kernels.
  if (options.mGroupsTokensHeadsQ) {
    TLLM_CHECK_ERROR(!isContextKernel(options.mFmhaKernelType),
                     "mGroupsTokensHeadsQ should only be enabled for generation kernels.");
    TLLM_CHECK_ERROR(!options.mIsMlaGen || options.mSelectsGroupedMla,
                     "MLA generation with mGroupsTokensHeadsQ requires mSelectsGroupedMla.");
  }

#ifdef TLLM_RUBIN_FEATURES
  if (options.mFineGrainedForceValid) {
    TLLM_CHECK_ERROR(options.mFuseEpilogueIntoCorr,
                     "fineGrainedForceValid is only supported with fuseEpilogueIntoCorr");
    TLLM_CHECK_ERROR(!tg::dtypeIsBlockFmt(options.mDtypeOut),
                     "fineGrainedForceValid does not support block scaling outputs");
    TLLM_CHECK_ERROR(isDisabled(options.mMultiCtasKvMode),
                     "fineGrainedForceValid does not support multi-CTA mode");
    TLLM_CHECK_ERROR(
      isSwapsMmaAbForGenerationKernel(options.mFmhaKernelType) ||
        isKeepsMmaAbForGenerationKernel(options.mFmhaKernelType),
      "fineGrainedForceValid has not been tested with context or generation kernel types.");

    TLLM_CHECK_ERROR(options.mClusterDimX == 1,
                     "FineGrained producer is not compatible with 2 CTA mode");

    TLLM_CHECK_ERROR(
      options.mHeadDimPerStageKv == 0,
      "FineGrained producer is only compatible with a single iteration of the head dim loop");

    // TODO Are there more features that are not compatible?
  }
  if (options.mFineGrainedProducer) {
    TLLM_CHECK_ERROR(options.mFineGrainedForceValid,
                     "fineGrainedProducer requires fineGrainedForceValid");
  }
#endif // TLLM_RUBIN_FEATURES

#ifdef TLLM_RUBIN_FEATURES
  if (options.mUsesSpcompress) {
    TLLM_CHECK_ERROR(options.mCudaArch == tg::CudaArch::Sm107a,
                     "Sparse attention is only supported on sm_107a.");
    TLLM_CHECK_ERROR(options.mFmhaKernelType == FmhaKernelType::Context,
                     "Sparse attention is only supported with context kernel.");
    TLLM_CHECK_ERROR(options.mDtypeQ == tg::Dtype::E4m3 || options.mDtypeQ == tg::Dtype::E4m3,
                     "Sparse attention is only supported with e4m3.");
    TLLM_CHECK_ERROR(options.mClusterDimX == 1,
                     "Sparse attention is not compatible with 2 CTA mode.");
    TLLM_CHECK_ERROR(!isCustomMask(options.mMaskType),
                     "Sparse attention is not compatible with custom mask.");
  }
#endif // TLLM_RUBIN_FEATURES

  // For transformed K/V, MmaOrder must be Pv0_Qk0_Pv1_Qk1.
  if (options.mDtypeQ != options.mDtypeK || getDtypeBmm2(options) != options.mDtypeV) {
    TLLM_CHECK_ERROR(options.mMmaOrder == MmaOrder::Pv0_Qk0_Pv1_Qk1,
                     "Only MMA order Pv0_Qk0_Pv1_Qk1 is supported for transformed K/V.");
  }
  if (isFp8KNvFp4V(options)) {
    TLLM_CHECK_ERROR(isFp8QFp8KNvFp4V(options),
                     "FP8-K/NVFP4-V is supported only with FP8 Q on Blackwell.");
    TLLM_CHECK_ERROR(options.mDtypeOut == tg::Dtype::Bfloat16,
                     "FP8-K/NVFP4-V requires BF16 output.");
    TLLM_CHECK_ERROR(isContextKernel(options.mFmhaKernelType) || !options.mGroupsTokensHeadsQ,
                     "FP8-K/NVFP4-V generation does not support groupsTokensHeadsQ.");
  }
  if (options.mEnablesBf16QFp8KvKOnlyTransform) {
    TLLM_CHECK_ERROR(usesKOnlyTransformPipeline(options),
                     "BF16Q+FP8KV K-only transform is only supported for non-MLA Blackwell "
                     "generation kernels with BF16 Q, E4M3 K/V, and H64/H128/H256.");
    TLLM_CHECK_ERROR(!options.mSeparateTransformedKv,
                     "BF16Q+FP8KV K-only transform cannot be combined with separateTransformedKv.");
  }
  if (options.mSeparateTransformedKv) {
    TLLM_CHECK_ERROR(!usesKOnlyTransformPipeline(options),
                     "BF16Q+FP8KV K-only transform cannot be combined with separateTransformedKv.");
    TLLM_CHECK_ERROR(
      supportsSeparateTransformedKv(options),
      "separateTransformedKv is only supported by BF16-Q full-transform generation kernels "
      "with E4M3 K/V on Blackwell, numInstsQ=1, numInstsKv=1, and equal "
      "H64/H128/H256 K/V heads.");
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
  if (options.mDtypeK == tg::Dtype::E2m1 || options.mDtypeV == tg::Dtype::E2m1) {
    if (!optionsFromArgs.mIsAtolSet) {
      options.mAtol = 0.3f;
    }
    if (!optionsFromArgs.mIsRtolSet) {
      options.mRtol = 0.1f;
    }
  } else if ((options.mDtypeQ == tg::Dtype::Fp16) || (options.mDtypeQ == tg::Dtype::Bfloat16)) {
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

// Update the runtime DSv4 inverse-RoPE + FP8 quant layout. This must run after the sample sequence
// lengths have been generated because the compact benchmark layout depends on mSumOfSeqLensQ.
inline void updateDsv4InvRopeFp8QuantOptions(FmhaOptions& options) {
  if (!options.mFusesDsv4InvRopeFp8Quant) {
    return;
  }

  TLLM_CHECK_ERROR(options.mNumHeadsQ % kDsv4HeadsPerGroup == 0,
                   "numHeadsQ must be divisible by the DSv4 packed output head group size.");

  if (options.mDsv4ScaleBufM == 0) {
    int32_t constexpr scaleTokenAlignment = 4;
    int32_t const numPackedTokens = options.mSumOfSeqLensQ;
    int32_t const paddedScaleBufM =
      (numPackedTokens + scaleTokenAlignment - 1) / scaleTokenAlignment * scaleTokenAlignment;
    options.mDsv4ScaleBufM = paddedScaleBufM;
  }
  TLLM_CHECK_ERROR(options.mDsv4ScaleBufM > 0, "Dsv4ScaleBufM must be initialized.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha
