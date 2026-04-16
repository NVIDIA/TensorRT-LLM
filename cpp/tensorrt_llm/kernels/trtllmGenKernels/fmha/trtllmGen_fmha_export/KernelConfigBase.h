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
#include <functional>
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

enum class SparseType : int32_t {
  None = 0,
  StaticTokenSparse = 1,
  DynamicTokenSparse = 2,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper functions to check the sparse type.

#define SPARSE_TYPE_FUNCTION(SType)                                                                \
  __host__ __device__ inline bool is##SType(SparseType sparseType) {                               \
    return (sparseType == SparseType::SType);                                                      \
  }

SPARSE_TYPE_FUNCTION(StaticTokenSparse)
SPARSE_TYPE_FUNCTION(DynamicTokenSparse)

#undef SPARSE_TYPE_FUNCTION

__host__ __device__ inline bool isTokenSparse(SparseType sparseType) {
  return (sparseType != SparseType::None);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline std::string sparseTypeToString(SparseType sparseType) {
  switch (sparseType) {
  case SparseType::None:
    return "none";
  case SparseType::StaticTokenSparse:
    return "static";
  case SparseType::DynamicTokenSparse:
    return "dynamic";
  default:
    TLLM_LOG_ERROR("Unsupported sparseType.");
    return "";
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

inline SparseType stringToSparseType(const char* str) {
  if (!strcmp(str, "none")) {
    return SparseType::None;
  } else if (!strcmp(str, "static")) {
    return SparseType::StaticTokenSparse;
  } else if (!strcmp(str, "dynamic")) {
    return SparseType::DynamicTokenSparse;
  } else {
    TLLM_LOG_ERROR("Unsupported sparseType ", str);
    return SparseType::None;
  }
}

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
  // Swap tensor A and tensor B of Mma.
  SwapsMmaAbForGeneration,
  // Keep tensor A and tensor B of Mma when there are enough numTokensQ (including grouped
  // headsQ) for the softmaxTask warps and correctionTask warps in order to make sure all threads
  // have work to do.
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

template <> inline std::string toString(SparseType e) {
  return sparseTypeToString(e);
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

#define KERNEL_CONFIG_BASE_FIELDS_BASE(X)                                                          \
  /* Whether to balance the workload for causal mask. */                                           \
  X(bool, mBalancesWorkloadForCausalMask, false, bool)                                             \
  /* The cluster dimension in the X dimension. */                                                  \
  X(int, mClusterDimX, 1, int)                                                                     \
  X(bool, mKnob1, true, bool)                                                             \
  /* The CUDA arch. */                                                                             \
  X(tg::CudaArch, mCudaArch, tg::CudaArch::Sm100a, int)                                            \
  /* The data type of MMA accumulators. */                                                         \
  X(tg::Dtype, mDtypeAcc, tg::Dtype::Fp32, uint32_t)                                               \
  /* The data type of the elements of Q. */                                                        \
  X(tg::Dtype, mDtypeQ, tg::Dtype::E4m3, uint32_t)                                                 \
  /* The data type of the elements of K/V when dtypeK == dtypeV, else Dtype::Void. */              \
  X(tg::Dtype, mDtypeKv, tg::Dtype::E4m3, uint32_t)                                                \
  /* The data type of the elements of K. */                                                        \
  X(tg::Dtype, mDtypeK, tg::Dtype::E4m3, uint32_t)                                                 \
  /* The data type of the elements of V. */                                                        \
  X(tg::Dtype, mDtypeV, tg::Dtype::E4m3, uint32_t)                                                 \
  /* The data type of the elements of O. */                                                        \
  X(tg::Dtype, mDtypeOut, tg::Dtype::E4m3, uint32_t)                                               \
  /* Whether to use dynamic numTokensPerPage. */                                                   \
  X(bool, mDynamicNumTokensPerPage, false, bool)                                                   \
  /* Whether to use fp16 softmax. */                                                               \
  X(bool, mEnablesFp16Softmax, false, bool)                                                        \
  /* Do we enable max value inflation? */                                                          \
  X(bool, mEnablesInflateMax, false, bool)                                                         \
  /* Whether 2 instances of the softmax task could be merged ? */                                  \
  X(bool, mEnablesSoftmaxTaskMerge, true, bool)                                                    \
  /* Whether to enable the programmatic dependent launch. */                                       \
  X(bool, mEnablesPdl, false, bool)                                                                \
  /* Whether to enable detection of repeated sequences. */                                         \
  X(bool, mEnablesRepeatedSeqDetect, false, bool)                                                  \
  /* Is it causal-mask spec-decoding generation kernel ? */                                        \
  X(bool, mIsCausalSpecDecodingGen, false, bool)                                                   \
  /* Is it custom-mask spec-decoding generation kernel ? */                                        \
  X(bool, mIsCustomSpecDecodingGen, false, bool)                                                   \
  /* Is it Multi-Latent Attention (MLA) generation kernel ? */                                     \
  X(bool, mIsMlaGen, false, bool)                                                                  \
  /* The kernel type under different cases. */                                                     \
  X(FmhaKernelType, mFmhaKernelType, FmhaKernelType::Context, int)                                 \
  /* Always skip correction */                                                                     \
  X(bool, mForcesSkipCorr, false, bool)                                                            \
  /* Store tensor to gmem directly in the end of the correction task. */                           \
  /* True: vectorized store. False: TMA store using dedicated warp. */                             \
  X(bool, mFuseEpilogueIntoCorr, true, bool)                                                       \
  /* Whether to transform K/V in the correction task. */                                           \
  X(bool, mFuseTransformKvIntoCorr, true, bool)                                                    \
  /* Whether to group the headsQ into one CTA. */                                                  \
  X(bool, mGroupsHeadsQ, false, bool)                                                              \
  /* Whether to group both tokensQ and headsQ into one CTA. */                                     \
  X(bool, mGroupsTokensHeadsQ, false, bool)                                                        \
  /* The head dimension per CTA for V. */                                                          \
  X(int32_t, mHeadDimPerCtaV, 0, int32_t)                                                          \
  /* The head dimension per stage for K/V. */                                                      \
  X(int32_t, mHeadDimPerStageKv, 0, int32_t)                                                       \
  /* The attention head size for Q and K. */                                                       \
  X(int32_t, mHeadDimQk, 128, int32_t)                                                             \
  /* The attention head size for V. */                                                             \
  X(int32_t, mHeadDimV, 128, int32_t)                                                              \
  /* Whether to interleave MUFU and sums in the softmax loop. */                                   \
  X(bool, mInterleavesMufuAndSums, false, bool)                                                    \
  /* Whether to use interleaved SF layout for tileV. (Only valid when KV cache is NVFP4) */        \
  X(bool, mInterleaveSfV, false, bool)                                                             \
  /* Attention mask type. */                                                                       \
  X(AttentionMaskType, mMaskType, AttentionMaskType::Dense, int)                                   \
  /* The MMA order (only used when numInstsQ == 2). */                                             \
  X(MmaOrder, mMmaOrder, MmaOrder::Pv0_Qk0_Pv1_Qk1, int)                                           \
  /* The multiCtasKvMode. */                                                                       \
  X(MultiCtasKvMode, mMultiCtasKvMode, MultiCtasKvMode::Disabled, int)                             \
  X(int, mNumDelayedCvtElts, 12, int)                                                              \
  /* The number of elements per block for Sage Attention (for K). */                               \
  /* 0 => we don't use Sage Attention. */                                                          \
  X(int, mNumEltsPerSageAttnBlkK, 0, int)                                                          \
  /* The number of elements per block for Sage Attention (for P). */                               \
  /* 0 => we don't use Sage Attention. */                                                          \
  X(int, mNumEltsPerSageAttnBlkP, 0, int)                                                          \
  /* The number of elements per block for Sage Attention (for Q). */                               \
  /* 0 => we don't use Sage Attention. */                                                          \
  X(int, mNumEltsPerSageAttnBlkQ, 0, int)                                                          \
  /* The number of elements per block for Sage Attention (for V). */                               \
  /* 0 => we don't use Sage Attention. */                                                          \
  X(int, mNumEltsPerSageAttnBlkV, 0, int)                                                          \
  /* The number of elements per SMEM stage for K/V. */                                             \
  X(int, mNumEltsPerSmemStageKv, 0, int)                                                           \
  /* Use FlashAttention4-style softmax emulation for the last N exp2 calls. */                     \
  X(int, mNumEmuExp2Elts, 0, int)                                                                  \
  /* Number of Q heads. */                                                                         \
  X(int, mNumHeadsQ, 4, int)                                                                       \
  /* Number of Q heads per KV head. */                                                             \
  X(int, mNumHeadsQPerKv, 1, int)                                                                  \
  /* Number of KV heads. */                                                                        \
  X(int, mNumHeadsKv, 4, int)                                                                      \
  /* The number of instances of tileQ, where each warpgroup handles one instance. */               \
  X(int32_t, mNumInstsQ, 2, int32_t)                                                               \
  /* The number of instances of tileKv, where each warpgroup handles one instance. */              \
  X(int32_t, mNumInstsKv, 1, int32_t)                                                              \
  /* The k-partition factor for BMM2 O = P * V. Can start BMM2 while softmax is in flight. */      \
  /* Beneficial only when BMM1 cycles == softmax cycles where neither dominates. */                \
  X(int, mNumKPartitionsMmaPv, 2, int)                                                             \
  /* Signal at the last N remaining exps to cover the */                                           \
  X(int, mNumLeadingExpElts, 6, int)                                                               \
  X(int, mNumPrefetchedFmas, 4, int)                                                               \
  /* The paged-kv configurations. The number of tokens in one pageKv. */                           \
  X(int32_t, mNumTokensPerPage, 32, int32_t)                                                       \
  /* How many warps are doing V transposition */                                                   \
  X(int, mNumTransposeWarps, 4, int)                                                               \
  /* Whether to output debug tensors. */                                                           \
  X(bool, mOutputDebugTensors, false, bool)                                                        \
  /* Does the kernel output stats on skipped softmax blocks? */                                    \
  X(bool, mOutputSkipSoftmaxStats, false, bool)                                                    \
  /* Input layout. */                                                                              \
  X(QkvLayout, mQkvLayout, QkvLayout::SeparateQkv, int)                                            \
  /* Do we reuse the shared memory buffer of K for V (for MLA Gen). */                             \
  X(bool, mReuseSmemKForV, false, bool)                                                            \
  /* Do we schedule exps in pairs? */                                                              \
  X(int, mSchedExpPairs, false, int)                                                               \
  /* Separate task for loadK and loadV. */                                                         \
  X(bool, mSeparateLoadKvTask, false, bool)                                                        \
  /* Separate resource for smemK and smemV. */                                                     \
  X(bool, mSeparateSmemKv, false, bool)                                                            \
  /* Is only one tokenQ used ? (normally generation kernels). */                                   \
  X(bool, mSingleTokenQ, false, bool)                                                              \
  /* Whether to only process one tokenQ per CTA in the Q dimension. */                             \
  X(bool, mSingleTokenQPerCta, false, bool)                                                        \
  /* The sparse attention type. */                                                                 \
  X(SparseType, mSparseType, SparseType::None, int)                                                \
  /* Do we skip the correction step when possible? */                                              \
  X(bool, mSkipsCorrWhenPossible, false, bool)                                                     \
  /* Do we skip the softmax operations when possible? */                                           \
  X(bool, mSkipsSoftmaxWhenPossible, false, bool)                                                  \
  /* Whether to store the softmax stats to global memory. */                                       \
  X(bool, mStoresSoftmaxStats, false, bool)                                                        \
  /* Whether to enable different sequence length for Q and K/V. */                                 \
  X(bool, mSupportsDiffSeqLensForQAndKv, false, bool)                                              \
  /* Whether to use variable sequence length. */                                                   \
  X(bool, mSupportsVarSeqLens, false, bool)                                                        \
  /* Whether to swap A and B tensors for Mmas. This should be used for small-M cases like */       \
  /* the generation phase. */                                                                      \
  X(bool, mSwapsMmaAb, false, bool)                                                                \
  /* Tile scheduler type. */                                                                       \
  X(TileScheduler, mTileScheduler, TileScheduler::Static, int)                                     \
  /* The tile size (in the sequence dimension) for K/V, which is considered as */                  \
  /* the tileSizeND of BMM1. */                                                                    \
  X(int32_t, mTileSizeKv, 128, int32_t)                                                            \
  /* The tile size (in the sequence dimension) for Q, which is considered as */                    \
  /* the tileSizeMD of BMM1. */                                                                    \
  X(int32_t, mTileSizeQ, 128, int32_t)                                                             \
  /* Transpose V in smem for MMA. */                                                               \
  X(bool, mTransposeSmemV, false, bool)                                                            \
  /* Unroll first and last iteration to reduce branching at the expense of code size. */           \
  X(bool, mUnrollFirstLastIter, true, bool)                                                        \
  /* The number of MMA-K partitions for the P. */                                                  \
  X(int, mNumKPartitionsTileP, 2, int)                                                             \
  /* The number of MMA-K partitions for the V. */                                                  \
  X(int, mNumKPartitionsTileV, 1, int)                                                             \
  /* Whether to use attention sinks (additional value in the denominator of the softmax). */       \
  X(bool, mUsesAttentionSinks, false, bool)                                                        \
  /* Whether to use block sparse attention. */                                                     \
  X(bool, mUseBlockSparseAttention, false, bool)                                                   \
  /* Whether to use an ordered sequence between softmax0 and softmax1. */                          \
  X(bool, mUsesOrderedSequence, true, bool)                                                        \
  /* Whether to use CGA reduction (deprecated, kept for benchmarking). */                          \
  X(bool, mUsesCgaReduction, false, bool)                                                          \
  X(bool, mKnob2, true, bool)                                                      \
  X(bool, mKnob3, true, bool)                                                         \
  X(bool, mKnob4, true, bool)

////////////////////////////////////////////////////////////////////////////////////////////////////

#define KERNEL_CONFIG_BASE_FIELDS_EXTRA(X)

#define KERNEL_CONFIG_BASE_FIELDS(X)                                                               \
  KERNEL_CONFIG_BASE_FIELDS_BASE(X)                                                                \
  KERNEL_CONFIG_BASE_FIELDS_EXTRA(X)

////////////////////////////////////////////////////////////////////////////////////////////////////

struct KernelConfigBase {
#define KERNEL_CONFIG_BASE_DECLARE_FIELD(type, name, defaultValue, hashType)                       \
  type name{defaultValue};

  KERNEL_CONFIG_BASE_FIELDS(KERNEL_CONFIG_BASE_DECLARE_FIELD)
#undef KERNEL_CONFIG_BASE_DECLARE_FIELD

  // Default constructor.
  KernelConfigBase() = default;

  // Convert the base options to a JSON object.
  void toJson(nlohmann::json& j) const {
#define KERNEL_CONFIG_BASE_TO_JSON_FIELD(type, name, defaultValue, hashType) TO_JSON(name)
    KERNEL_CONFIG_BASE_FIELDS(KERNEL_CONFIG_BASE_TO_JSON_FIELD)
#undef KERNEL_CONFIG_BASE_TO_JSON_FIELD
  }

  // Equality comparison over all fields (used for kernel cache key lookup).
  bool operator==(KernelConfigBase const& o) const {
    return
#define KERNEL_CONFIG_BASE_EQ_FIELD(type, name, defaultValue, hashType) (name == o.name)&&
      KERNEL_CONFIG_BASE_FIELDS(KERNEL_CONFIG_BASE_EQ_FIELD)
#undef KERNEL_CONFIG_BASE_EQ_FIELD
        true;
  }

  bool operator!=(KernelConfigBase const& o) const { return !(*this == o); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha

////////////////////////////////////////////////////////////////////////////////////////////////////
// std::hash specialization for KernelConfigBase (kernel cache key).
////////////////////////////////////////////////////////////////////////////////////////////////////

namespace std {
template <> struct hash<fmha::KernelConfigBase> {
  size_t operator()(fmha::KernelConfigBase const& k) const {
    size_t h = 0;
    // Combine algorithm from boost::hash_combine
    // https://www.boost.org/doc/libs/1_36_0/doc/html/hash/reference.html#boost.hash_combine
    auto combine = [&h](size_t v) { h ^= v + 0x9e3779b9 + (h << 6) + (h >> 2); };
#define KERNEL_CONFIG_BASE_HASH_COMBINE_FIELD(type, name, defaultValue, hashType)                  \
  combine(std::hash<hashType>{}(static_cast<hashType>(k.name)));

    KERNEL_CONFIG_BASE_FIELDS(KERNEL_CONFIG_BASE_HASH_COMBINE_FIELD)
#undef KERNEL_CONFIG_BASE_HASH_COMBINE_FIELD
    return h;
  }
};
} // namespace std

#undef KERNEL_CONFIG_BASE_FIELDS
#undef KERNEL_CONFIG_BASE_FIELDS_BASE
#undef KERNEL_CONFIG_BASE_FIELDS_EXTRA
