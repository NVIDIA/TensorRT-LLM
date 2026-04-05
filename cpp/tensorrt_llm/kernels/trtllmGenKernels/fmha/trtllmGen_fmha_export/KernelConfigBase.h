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

#include <trtllm/gen/CudaArchDecl.h>
#include <trtllm/gen/DtypeDecl.h>
#ifndef TLLM_FMHA_TRTLLM_COMPAT
#include <trtllm/gen/GenCtx.h>
#endif // TLLM_FMHA_TRTLLM_COMPAT
#include <nlohmann/json.hpp>
#include <cassert>
#include <cstring>
#include <string>
#include <cuda_runtime.h>

namespace tg = trtllm::gen;

namespace fmha {

////////////////////////////////////////////////////////////////////////////////////////////////////
// When TLLM_FMHA_TRTLLM_COMPAT is defined, enum types, toString specializations, and utility
// functions are provided by trtllmGenExportCompat.h instead (using TRT-LLM type aliases).
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef TLLM_FMHA_TRTLLM_COMPAT

////////////////////////////////////////////////////////////////////////////////////////////////////
// Macros to simplify JSON serialization
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef TO_JSON
#define TO_JSON(field) j[#field] = toString(field);
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// Enum types.
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

enum class AttentionMaskType {
  // Dense mask.
  Dense = 0,
  // Causal mask.
  Causal,
  // Sliding window causal mask or chunked attention causal mask.
  SlidingOrChunkedCausal,
  // Custom mask.
  Custom
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper functions to check the mask type.

#define ATTENTION_MASK_TYPE_FUNCTION(MaskType)                                                     \
  __host__ __device__ inline bool is##MaskType##Mask(AttentionMaskType maskType) {                 \
    return (maskType == AttentionMaskType::MaskType);                                              \
  }

ATTENTION_MASK_TYPE_FUNCTION(Dense)
ATTENTION_MASK_TYPE_FUNCTION(Causal)
ATTENTION_MASK_TYPE_FUNCTION(SlidingOrChunkedCausal)
ATTENTION_MASK_TYPE_FUNCTION(Custom)

#undef ATTENTION_MASK_TYPE_FUNCTION

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class FmhaKernelType {
  // The context-phase kernels.
  Context = 0,
  // Choose the best generation kernel based on the heuristic:
  // use SwapsMmaAbForGeneration kernels when numTokensQ (including grouped
  // headsQ) <= 16, otherwise not.
  Generation,
  // Swap tensor A and tensor B of Mma, which only supports numTokensQ (including grouped headsQ)
  // up to 16.
  SwapsMmaAbForGeneration,
  // Keep tensor A and tensor B of Mma when there are enough numTokensQ (including grouped
  // headsQ) for the softmaxTask warps and correctionTask warps
  // in order to make sure all threads have work to do.
  KeepsMmaAbForGeneration
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper functions to check the fmha kernel type.

#define FMHA_KERNEL_TYPE_FUNCTION(KernelType)                                                      \
  __host__ __device__ inline bool is##KernelType##Kernel(FmhaKernelType kernelType) {              \
    return (kernelType == FmhaKernelType::KernelType);                                             \
  }

FMHA_KERNEL_TYPE_FUNCTION(Context)
FMHA_KERNEL_TYPE_FUNCTION(Generation)
FMHA_KERNEL_TYPE_FUNCTION(SwapsMmaAbForGeneration)
FMHA_KERNEL_TYPE_FUNCTION(KeepsMmaAbForGeneration)

#undef FMHA_KERNEL_TYPE_FUNCTION

////////////////////////////////////////////////////////////////////////////////////////////////////

// This enum defines the order of operations in each iteration of the MMA main loop.
// Only used when numInstsQ == 2.
enum class MmaOrder {
  // 2 instances: PV0[i] - QK0[i+1] - PV1[i] - QK1[i+1].
  Pv0_Qk0_Pv1_Qk1 = 0,
  // 2 instances: QK0[i+1] - PV0[i] - QK1[i+1] - PV1[i].
  Qk0_Pv0_Qk1_Pv1,
  // 2 instances: QK0[i+1] - QK1[i+1] - PV0[i] - PV1[i].
  Qk0_Qk1_Pv0_Pv1,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

inline std::string mmaOrderToString(MmaOrder mmaOrder) {
  switch (mmaOrder) {
  case MmaOrder::Pv0_Qk0_Pv1_Qk1:
    return "Pv0_Qk0_Pv1_Qk1";
  case MmaOrder::Qk0_Pv0_Qk1_Pv1:
    return "Qk0_Pv0_Qk1_Pv1";
  case MmaOrder::Qk0_Qk1_Pv0_Pv1:
    return "Qk0_Qk1_Pv0_Pv1";
  default:
    assert(false);
    return "Invalid MmaOrder";
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline MmaOrder stringToMmaOrder(const char* str) {
  if (!strcmp(str, "Pv0_Qk0_Pv1_Qk1")) {
    return MmaOrder::Pv0_Qk0_Pv1_Qk1;
  } else if (!strcmp(str, "Qk0_Pv0_Qk1_Pv1")) {
    return MmaOrder::Qk0_Pv0_Qk1_Pv1;
  } else if (!strcmp(str, "Qk0_Qk1_Pv0_Pv1")) {
    return MmaOrder::Qk0_Qk1_Pv0_Pv1;
  } else {
    assert(false);
    return MmaOrder::Pv0_Qk0_Pv1_Qk1;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// The multiCtasKvMode.
enum class MultiCtasKvMode {
  // Disable the multiCtasKvMode.
  Disabled = 0,
  // Do the reduction through the global memory and atomic counters.
  GmemReduction,
  // Same as GmemReduction, but use a separate kernel for the reduction.
  // It is only supported/needed for 2-CTA or 1-CTA keepsMmaAbForGeneration MLA kernels with large
  // reduction tiles.
  GmemReductionWithSeparateKernel,
  // Do the reduction through the CGA remote shared memory.
  CgaSmemReduction
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper functions to check the multiCtasKvMode.
#define MULTI_CTAS_KV_MODE_FUNCTION(Mode)                                                          \
  __host__ __device__ inline bool is##Mode(MultiCtasKvMode multiCtasKvMode) {                      \
    return (multiCtasKvMode == MultiCtasKvMode::Mode);                                             \
  }

MULTI_CTAS_KV_MODE_FUNCTION(Disabled)
MULTI_CTAS_KV_MODE_FUNCTION(GmemReduction)
MULTI_CTAS_KV_MODE_FUNCTION(GmemReductionWithSeparateKernel)
MULTI_CTAS_KV_MODE_FUNCTION(CgaSmemReduction)

#undef MULTI_CTAS_KV_MODE_FUNCTION

// Helper functions to check if the multiCtasKvMode is supported.
__host__ __device__ inline bool supportsMultiCtasKvMode(MultiCtasKvMode multiCtasKvMode) {
  return (multiCtasKvMode != MultiCtasKvMode::Disabled);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline std::string multiCtasKvModeToString(MultiCtasKvMode multiCtasKvMode) {
  switch (multiCtasKvMode) {
  case MultiCtasKvMode::Disabled:
    return "Disabled";
  case MultiCtasKvMode::GmemReduction:
    return "GmemReduction";
  case MultiCtasKvMode::GmemReductionWithSeparateKernel:
    return "GmemReductionWithSeparateKernel";
  case MultiCtasKvMode::CgaSmemReduction:
    return "CgaSmemReduction";
  default:
    TLLM_LOG_ERROR("Unsupported multiCtasKvMode.");
    return "";
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline MultiCtasKvMode stringToMultiCtasKvMode(const char* str) {
  if (!strcmp(str, "disabled")) {
    return MultiCtasKvMode::Disabled;
  } else if (!strcmp(str, "gmemReduction")) {
    return MultiCtasKvMode::GmemReduction;
  } else if (!strcmp(str, "gmemReductionWithSeparateKernel")) {
    return MultiCtasKvMode::GmemReductionWithSeparateKernel;
  } else if (!strcmp(str, "cgaSmemReduction")) {
    return MultiCtasKvMode::CgaSmemReduction;
  } else {
    TLLM_LOG_ERROR("Unsupported multiCtasKvMode ", str);
    return MultiCtasKvMode::Disabled;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Note that (batchSize, seqLen) dimensions will be packed as sumOfSeqLens without paddings for
// variable sequence lengths.
enum class QkvLayout {
  // Separate Q, K and V buffers. The shape is [batchSize, seqLen, numHeads, headDim].
  SeparateQkv = 0,
  // Single buffer for Q, K and V. Shape: [batchSize, seqLen, numHeadsQ + 2*numHeadsKv, headDim].
  PackedQkv,
  // Paged buffer for K and V. Its shape is [batchSize, 2, maxNumPagesPerSeq] where 2 corresponds
  // to K and V. That buffer stores the logical page indices of the paged-KV memory pool. Each
  // "page" of that pool is a contiguous buffer of shape [numHeadsKv, pageSize, headDim].
  PagedKv,
  // Contiguous buffer for Q with shape [batchSize, seqLen, numHeads, headDim] and contiguous buf-
  // fer for K/V with shape [batchSize, 2, numHeads, maxSeqLen, headDim].
  ContiguousKv,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper functions to check the QkvLayout type.

#define QKV_LAYOUT_FUNCTION(LayoutType)                                                            \
  __host__ __device__ inline bool is##LayoutType(QkvLayout qkvLayout) {                            \
    return (qkvLayout == QkvLayout::LayoutType);                                                   \
  }

QKV_LAYOUT_FUNCTION(SeparateQkv)
QKV_LAYOUT_FUNCTION(PackedQkv)
QKV_LAYOUT_FUNCTION(PagedKv)
QKV_LAYOUT_FUNCTION(ContiguousKv)

#undef QKV_LAYOUT_FUNCTION

////////////////////////////////////////////////////////////////////////////////////////////////////

inline std::string qkvLayoutToString(QkvLayout qkvLayout) {
  switch (qkvLayout) {
  case QkvLayout::SeparateQkv:
    return "SeparateQkv";
  case QkvLayout::PackedQkv:
    return "PackedQkv";
  case QkvLayout::PagedKv:
    return "PagedKv";
  case QkvLayout::ContiguousKv:
    return "ContiguousKv";
  default:
    TLLM_LOG_ERROR("Unsupported qkv layout.");
    return "";
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class TileScheduler {
  // Static scheduler (Non-persistent).
  Static = 0,
  // Persistent scheduler.
  Persistent
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> __host__ __device__ inline T ceilDiv(T m, T n) {
  return (m + n - 1) / n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// Serialization helpers (toString functions).
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

// Generic toString for numeric types.
template <typename T> inline std::string toString(T e) {
  return std::to_string(e);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <> inline std::string toString(bool flag) {
  return (flag ? "true" : "false");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <> inline std::string toString(AttentionMaskType e) {
  switch (e) {
  case AttentionMaskType::Dense:
    return "Dense";
  case AttentionMaskType::Causal:
    return "Causal";
  case AttentionMaskType::SlidingOrChunkedCausal:
    return "SlidingOrChunkedCausal";
  case AttentionMaskType::Custom:
    return "Custom";
  default:
    TLLM_LOG_ERROR("Unsupported enum.");
    return "";
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <> inline std::string toString(tg::CudaArch arch) {
  return tg::cudaArchToString(arch);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <> inline std::string toString(FmhaKernelType e) {
  switch (e) {
  case FmhaKernelType::Context:
    return "Context";
  case FmhaKernelType::Generation:
    return "Gen";
  case FmhaKernelType::SwapsMmaAbForGeneration:
    return "SwapsAbForGen";
  case FmhaKernelType::KeepsMmaAbForGeneration:
    return "KeepsAbForGen";
  default:
    TLLM_LOG_ERROR("Unsupported enum.");
    return "";
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <> inline std::string toString(MultiCtasKvMode mode) {
  return multiCtasKvModeToString(mode);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <> inline std::string toString(TileScheduler scheduler) {
  switch (scheduler) {
  case TileScheduler::Static:
    return "Static";
  case TileScheduler::Persistent:
    return "Persistent";
  default:
    TLLM_LOG_ERROR("Unsupported enum.");
    return "";
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <> inline std::string toString(QkvLayout e) {
  return qkvLayoutToString(e);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <> inline std::string toString(tg::Dtype e) {
  return tg::dtypeToString(e);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <> inline std::string toString(MmaOrder e) {
  return mmaOrderToString(e);
}

#endif // !TLLM_FMHA_TRTLLM_COMPAT (enum types, toString, helpers)

////////////////////////////////////////////////////////////////////////////////////////////////////
// *************************************************************************************************
// KernelConfigBase: Base class shared between FmhaOptions and KernelConfig.
// This struct contains all common configuration variables that are directly shared
// between FmhaOptions (user-facing configuration) and KernelConfig (kernel compilation config).
// FmhaOptions is used to initialize KernelConfig.
// *************************************************************************************************
////////////////////////////////////////////////////////////////////////////////////////////////////

struct KernelConfigBase {
  // Whether to balance the workload for causal mask.
  bool mBalancesWorkloadForCausalMask{false};
  // The cluster dimension in the X dimension.
  int mClusterDimX{1};
  // Annotate the masking code path with ColdBlock to indicate less traversed path.
  bool mColdBlockTmemS{true};
  // The CUDA arch.
  tg::CudaArch mCudaArch{tg::CudaArch::Sm100a};
  // The data type of MMA accumulators.
  tg::Dtype mDtypeAcc{tg::Dtype::Fp32};
  // The data type of the elements of Q.
  tg::Dtype mDtypeQ{tg::Dtype::E4m3};
  // The data type of the elements of K/V.
  tg::Dtype mDtypeKv{tg::Dtype::E4m3};
  // The data type of the elements of O.
  tg::Dtype mDtypeOut{tg::Dtype::E4m3};
  // Whether to use dynamic numTokensPerPage.
  bool mDynamicNumTokensPerPage{false};
  // Whether to use fp16 softmax.
  bool mEnablesFp16Softmax{false};
  // Do we enable max value inflation?
  bool mEnablesInflateMax{false};
  // Whether 2 instances of the softmax task could be merged ?
  bool mEnablesSoftmaxTaskMerge{true};
  // Whether to enable the programmatic dependent launch.
  bool mEnablesPdl{false};
  // Whether to enable detection of repeated sequences.
  bool mEnablesRepeatedSeqDetect{false};
  // Is it causal-mask spec-decoding generation kernel ?
  bool mIsCausalSpecDecodingGen{false};
  // Is it custom-mask spec-decoding generation kernel ?
  bool mIsCustomSpecDecodingGen{false};
  // Is it Multi-Latent Attention (MLA) generation kernel ?
  bool mIsMlaGen{false};
  // Is it Multi-Latent Attention (MLA) sparse generation kernel ?
  bool mIsSparseMla{false};
  // The kernel type under different cases.
  FmhaKernelType mFmhaKernelType{FmhaKernelType::Context};
  // Always skip correction
  bool mForcesSkipCorr{false};
  // Store tensor to gmem directly in the end of the correction task.
  // True: vectorized store. False: TMA store using dedicated warp.
  bool mFuseEpilogueIntoCorr{true};
  // Whether to transform K/V in the correction task.
  bool mFuseTransformKvIntoCorr{true};
  // Whether to group the headsQ into one CTA.
  bool mGroupsHeadsQ{false};
  // Whether to group both tokensQ and headsQ into one CTA.
  bool mGroupsTokensHeadsQ{false};
  // The head dimension per CTA for V.
  int32_t mHeadDimPerCtaV{0};
  // The head dimension per stage for K/V.
  int32_t mHeadDimPerStageKv{0};
  // The attention head size for Q and K.
  int32_t mHeadDimQk{128};
  // The attention head size for V.
  int32_t mHeadDimV{128};
  // Whether to interleave MUFU and sums in the softmax loop.
  bool mInterleavesMufuAndSums{false};
  // Whether to use interleaved SF layout for tileV. (Only valid when KV cache is NVFP4)
  bool mInterleaveSfV{false};
  // Attention mask type.
  AttentionMaskType mMaskType{AttentionMaskType::Dense};
  // The MMA order (only used when numInstsQ == 2).
  MmaOrder mMmaOrder{MmaOrder::Pv0_Qk0_Pv1_Qk1};
  // The multiCtasKvMode.
  MultiCtasKvMode mMultiCtasKvMode{MultiCtasKvMode::Disabled};
  // Delay downcast until N-th exps to hide exp latency (17 clks); must be a multiple of 4.
  int mNumDelayedCvtElts{12};
  // The number of elements per block for Sage Attention (for K). 0 => we don't use Sage Attention.
  int mNumEltsPerSageAttnBlkK{0};
  // The number of elements per block for Sage Attention (for P). 0 => we don't use Sage Attention.
  int mNumEltsPerSageAttnBlkP{0};
  // The number of elements per block for Sage Attention (for Q). 0 => we don't use Sage Attention.
  int mNumEltsPerSageAttnBlkQ{0};
  // The number of elements per block for Sage Attention (for V). 0 => we don't use Sage Attention.
  int mNumEltsPerSageAttnBlkV{0};
  // The number of elements per SMEM stage for K/V.
  int mNumEltsPerSmemStageKv{0};
  // Use FlashAttention4-style softmax emulation for the last N exp2 calls.
  int mNumEmuExp2Elts{0};
  // Number of Q heads.
  int mNumHeadsQ{4};
  // Number of Q heads per KV head.
  int mNumHeadsQPerKv{1};
  // Number of KV heads.
  int mNumHeadsKv{4};
  // The number of instances of tileQ, where each warpgroup handles one instance.
  int32_t mNumInstsQ{2};
  // The number of instances of tileKv, where each warpgroup handles one instance.
  int32_t mNumInstsKv{1};
  // The k-partition factor for BMM2 O = P * V. Can start BMM2 while softmax is in flight.
  // Beneficial only when BMM1 cycles == softmax cycles where neither dominates.
  int mNumKPartitionsMmaPv{2};
  // Signal at the last N remaining exps to cover the OrderedSequenceBarriers latency (~40 clks).
  int mNumLeadingExpElts{6};
  // Delay exp until N-th FMAs to hide FMA latency (7 clks); must be a multiple of 2.
  int mNumPrefetchedFmas{4};
  // The paged-kv configurations. The number of tokens in one pageKv.
  int32_t mNumTokensPerPage{32};
  // How many warps are doing V transposition
  int mNumTransposeWarps{4};
  // Whether to output debug tensors.
  bool mOutputDebugTensors{false};
  // Does the kernel output stats on skipped softmax blocks?
  bool mOutputSkipSoftmaxStats{false};
  // Input layout.
  QkvLayout mQkvLayout{QkvLayout::SeparateQkv};
  // Do we reuse the shared memory buffer of K for V (for MLA Gen).
  bool mReuseSmemKForV{false};
  // Do we schedule exps in pairs?
  int mSchedExpPairs{false};
  // Separate task for loadK and loadV
  bool mSeparateLoadKvTask{false};
  // Separate resource for smemK and smemV
  bool mSeparateSmemKv{false};
  // Is only one tokenQ used ? (normally generation kernels).
  bool mSingleTokenQ{false};
  // Whether to only process one tokenQ per CTA in the Q dimension.
  bool mSingleTokenQPerCta{false};
  // Do we skip the correction step when possible?
  bool mSkipsCorrWhenPossible{false};
  // Do we skip the softmax operations when possible?
  bool mSkipsSoftmaxWhenPossible{false};
  // Whether to store the softmax stats to global memory.
  bool mStoresSoftmaxStats{false};
  // Whether to enable different sequence length for Q and K/V.
  bool mSupportsDiffSeqLensForQAndKv{false};
  // Whether to use variable sequence length.
  bool mSupportsVarSeqLens{false};
  // Whether to swap A and B tensors for Mmas. This should be used for small-M cases like the
  // generation phase.
  bool mSwapsMmaAb{false};
  // Tile scheduler type.
  TileScheduler mTileScheduler{TileScheduler::Static};
  // The tile size (in the sequence dimension) for K/V, which is considered as the tileSizeND of
  // BMM1.
  int32_t mTileSizeKv{128};
  // The tile size (in the sequence dimension) for Q, which is considered as the tileSizeMD of BMM1.
  int32_t mTileSizeQ{128};
  // Transpose V in smem for MMA.
  bool mTransposeSmemV{false};
  // Unroll first and last iteration to reduce branching at the expense of code size.
  bool mUnrollFirstLastIter{true};
  // The number of MMA-K partitions for the P.
  int mNumKPartitionsTileP{2};
  // The number of MMA-K partitions for the V.
  int mNumKPartitionsTileV{1};
  // Whether to use attention sinks (additional value in the denominator of the softmax).
  bool mUsesAttentionSinks{false};
  // Whether to use block sparse attention.
  bool mUseBlockSparseAttention{false};
  // Whether to use an ordered sequence between softmax0 and softmax1.
  bool mUsesOrderedSequence{true};
  // Whether to use CGA reduction (deprecated, kept for benchmarking).
  bool mUsesCgaReduction{false};
  // Switch warps opportunistically during rowsum to unblock softmax sequence.
  bool mWarpSwitchTmemSoftmax{true};
  // Switch warps opportunistically during correction to unblock softmax sequence.
  bool mWarpSwitchTmemCorr{true};
  // Switch warps opportunistically during max reduction to unblock softmax sequence.
  bool mWarpSwitchTmemS{true};

  // Default constructor.
  KernelConfigBase() = default;

  // Convert the base options to a JSON object.
  void toJson(nlohmann::json& j) const {
    TO_JSON(mBalancesWorkloadForCausalMask)
    TO_JSON(mClusterDimX)
    TO_JSON(mColdBlockTmemS)
    TO_JSON(mCudaArch)
    TO_JSON(mDtypeAcc)
    TO_JSON(mDtypeQ)
    TO_JSON(mDtypeKv)
    TO_JSON(mDtypeOut)
    TO_JSON(mDynamicNumTokensPerPage)
    TO_JSON(mEnablesFp16Softmax)
    TO_JSON(mEnablesInflateMax)
    TO_JSON(mEnablesSoftmaxTaskMerge)
    TO_JSON(mEnablesPdl)
    TO_JSON(mEnablesRepeatedSeqDetect)
    TO_JSON(mIsCausalSpecDecodingGen)
    TO_JSON(mIsCustomSpecDecodingGen)
    TO_JSON(mIsMlaGen)
    TO_JSON(mIsSparseMla)
    TO_JSON(mFmhaKernelType)
    TO_JSON(mForcesSkipCorr)
    TO_JSON(mFuseEpilogueIntoCorr)
    TO_JSON(mFuseTransformKvIntoCorr)
    TO_JSON(mGroupsHeadsQ)
    TO_JSON(mGroupsTokensHeadsQ)
    TO_JSON(mHeadDimPerCtaV)
    TO_JSON(mHeadDimPerStageKv)
    TO_JSON(mHeadDimQk)
    TO_JSON(mHeadDimV)
    TO_JSON(mInterleavesMufuAndSums)
    TO_JSON(mInterleaveSfV)
    TO_JSON(mMaskType)
    TO_JSON(mMmaOrder)
    TO_JSON(mMultiCtasKvMode)
    TO_JSON(mNumDelayedCvtElts)
    TO_JSON(mNumEltsPerSageAttnBlkK)
    TO_JSON(mNumEltsPerSageAttnBlkP)
    TO_JSON(mNumEltsPerSageAttnBlkQ)
    TO_JSON(mNumEltsPerSageAttnBlkV)
    TO_JSON(mNumEltsPerSmemStageKv)
    TO_JSON(mNumEmuExp2Elts)
    TO_JSON(mNumHeadsQ)
    TO_JSON(mNumHeadsQPerKv)
    TO_JSON(mNumHeadsKv)
    TO_JSON(mNumInstsQ)
    TO_JSON(mNumInstsKv)
    TO_JSON(mNumKPartitionsMmaPv)
    TO_JSON(mNumLeadingExpElts)
    TO_JSON(mNumPrefetchedFmas)
    TO_JSON(mNumTokensPerPage)
    TO_JSON(mNumTransposeWarps)
    TO_JSON(mOutputDebugTensors)
    TO_JSON(mOutputSkipSoftmaxStats)
    TO_JSON(mQkvLayout)
    TO_JSON(mReuseSmemKForV)
    TO_JSON(mSchedExpPairs)
    TO_JSON(mSeparateLoadKvTask)
    TO_JSON(mSeparateSmemKv)
    TO_JSON(mSingleTokenQ)
    TO_JSON(mSingleTokenQPerCta)
    TO_JSON(mSkipsCorrWhenPossible)
    TO_JSON(mSkipsSoftmaxWhenPossible)
    TO_JSON(mStoresSoftmaxStats)
    TO_JSON(mSupportsDiffSeqLensForQAndKv)
    TO_JSON(mSupportsVarSeqLens)
    TO_JSON(mSwapsMmaAb)
    TO_JSON(mTileScheduler)
    TO_JSON(mTileSizeKv)
    TO_JSON(mTileSizeQ)
    TO_JSON(mTransposeSmemV)
    TO_JSON(mUnrollFirstLastIter)
    TO_JSON(mNumKPartitionsTileP)
    TO_JSON(mNumKPartitionsTileV)
    TO_JSON(mUsesAttentionSinks)
    TO_JSON(mUseBlockSparseAttention)
    TO_JSON(mUsesCgaReduction)
    TO_JSON(mUsesOrderedSequence)
    TO_JSON(mWarpSwitchTmemSoftmax)
    TO_JSON(mWarpSwitchTmemCorr)
    TO_JSON(mWarpSwitchTmemS)
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha
