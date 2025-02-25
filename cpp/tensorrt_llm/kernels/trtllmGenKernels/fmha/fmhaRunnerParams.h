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

#pragma once

#include <cuda_runtime.h>

namespace tensorrt_llm
{
namespace kernels
{

////////////////////////////////////////////////////////////////////////////////////////////////////

// The attention mask types.
enum class TrtllmGenAttentionMaskType
{
    // Dense mask.
    Dense = 0,
    // Causal mask.
    Causal,
    // Sliding window causal mask.
    SlidingWindowCausal,
    // Custom mask.
    Custom
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper functions to check the mask type.

#define ATTENTION_MASK_TYPE_FUNCTION(MaskType)                                                                         \
    inline bool is##MaskType##Mask(TrtllmGenAttentionMaskType maskType)                                                \
    {                                                                                                                  \
        return (maskType == TrtllmGenAttentionMaskType::MaskType);                                                     \
    }

ATTENTION_MASK_TYPE_FUNCTION(Dense)
ATTENTION_MASK_TYPE_FUNCTION(Causal)
ATTENTION_MASK_TYPE_FUNCTION(SlidingWindowCausal)
ATTENTION_MASK_TYPE_FUNCTION(Custom)

#undef ATTENTION_MASK_TYPE_FUNCTION

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class FmhaKernelType
{
    // The context-phase kernels.
    Context = 0,
    // Choose the best generation kernel based on the heuristic:
    // use SwapsMmaAbForGeneration kernels when numHeadsQPerKv <= 16, otherwise KeepsMmaAbForGeneration.
    Generation,
    // Swap tensor A and tensor B of Mma, which only supports numHeadsQPerKv <= 16.
    SwapsMmaAbForGeneration,
    // Keep tensor A and tensor B of Mma.
    KeepsMmaAbForGeneration,
    // Speculative decoding (Medusa and Eagle) generation-phase attention kernels, where seqLenQ > 1.
    SpecDecodingGeneration
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper functions to check the fmha kernel type.

#define FMHA_KERNEL_TYPE_FUNCTION(KernelType)                                                                          \
    inline bool is##KernelType##Kernel(FmhaKernelType kernelType)                                                      \
    {                                                                                                                  \
        return (kernelType == FmhaKernelType::KernelType);                                                             \
    }

FMHA_KERNEL_TYPE_FUNCTION(Context)
FMHA_KERNEL_TYPE_FUNCTION(Generation)
FMHA_KERNEL_TYPE_FUNCTION(SwapsMmaAbForGeneration)
FMHA_KERNEL_TYPE_FUNCTION(KeepsMmaAbForGeneration)
FMHA_KERNEL_TYPE_FUNCTION(SpecDecodingGeneration)

#undef QKV_LAYOUT_FUNCTION

////////////////////////////////////////////////////////////////////////////////////////////////////

// Note that (batchSize, seqLen) dimensions will be packed as sumOfSeqLens without paddings for
// variable sequence lengths.
enum class QkvLayout
{
    // SeparateQkv: separate Q, K and V buffers.
    // Each has the shape: [batchSize, seqLen, numHeads, headDim].
    SeparateQkv = 0,
    // PackedQkv: single buffer for Q, K and V.
    // Shape: [batchSize, seqLen, numHeadsQ + 2*numHeadsKv, headDim].
    PackedQkv,
    // Paged buffer for K and V. Its shape is [batchSize, 2, maxNumPagesPerSeq]. The 2 corresponds to
    // K
    // and V. That buffer stores the logical page index of the paged-KV memory pool. Each "page" of
    // that
    // pool is a contiguous buffer of shape [numHeadsKv, pageSize, headDim].
    PagedKv,
    // ContiguousKv:
    // Contiguous buffer for Q with shape [batchSize, seqLen, numHeads, headDim].
    // Contiguous buffer for Kv with shape [batchSize, seqLen, 2 * numHeads, headDim].
    ContiguousKv,
};

// Helper functions to check the QkvLayout type.

#define QKV_LAYOUT_FUNCTION(LayoutType)                                                                                \
    inline bool is##LayoutType(QkvLayout qkvLayout)                                                                    \
    {                                                                                                                  \
        return (qkvLayout == QkvLayout::LayoutType);                                                                   \
    }

QKV_LAYOUT_FUNCTION(SeparateQkv)
QKV_LAYOUT_FUNCTION(PackedQkv)
QKV_LAYOUT_FUNCTION(PagedKv)
QKV_LAYOUT_FUNCTION(ContiguousKv)

#undef QKV_LAYOUT_FUNCTION

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class TileScheduler
{
    // Static scheduler (Non-persistent).
    Static = 0,
    // Persistent scheduler.
    Persistent
};

////////////////////////////////////////////////////////////////////////////////////////////////////

struct TllmGenFmhaRunnerParams
{
    // Input layout.
    QkvLayout mQkvLayout;
    // Attention mask type.
    TrtllmGenAttentionMaskType mMaskType;
    // The kernel type.
    FmhaKernelType mKernelType;
    // The tile scheduler.
    TileScheduler mTileScheduler;
    // The multiCtasKvMode (i.e. multiBlockMode).
    bool mMultiCtasKvMode;

    // Input QKV buffers.
    void const* qPtr;
    void const* kPtr;
    void const* vPtr;
    // Packed KV buffer
    void const* kvPtr;
    // Packed QKV buffer
    void const* qkvPtr;
    // The scaling factor pointer of K.
    void const* kSfBasePtr;
    // The scaling factor pointer of V.
    void const* vSfBasePtr;
    // The custom mask ptr.
    uint32_t const* customMaskPtr;
    // The packed custom mask's offsets of each sequence.
    int64_t const* customMaskOffsetsPtr;
    // The first sparseMask offsets in the Kv sequence dimension.
    int32_t const* firstSparseMaskOffsetsKvPtr;
    // The counter for the multiCtasKv mode.
    int32_t* multiCtasKvCounterPtr;
    // The sequence length buffer for K/V.
    int const* seqLensKvPtr;
    // The cumulative sequence length buffer for Q and K/V
    int const* cumSeqLensQPtr;
    int const* cumSeqLensKvPtr;
    // The kv page idx
    int const* kvPageIdxPtr;
    // The device output scale for FP8 quantization.
    float const* outputScalePtr;
    // The device scaling factor for softmax (multiplied by log2 to use faster exp2)
    float const* scaleSoftmaxLog2Ptr;
    // The device scale for KV scaling factor.
    float const* kvSfScalePtr;
    // The device scale for O scaling factor.
    float const* oSfScalePtr;
    // The scratch space for each CtaKv when the multiCtasKv mode is enabled.
    // PartialO, partialMax and partialSum will be stored to the scratch space.
    void* multiCtasKvScratchPtr;
    // The output buffer.
    void* oPtr;
    // The output scaling factor buffer.
    void* oSfPtr;

    // Head dimension for Q and K.
    int mHeadDimQk;
    // Head dimension for V.
    int mHeadDimV;
    // Number of heads for Q and K/V.
    int mNumHeadsQ, mNumHeadsKv, mNumHeadsQPerKv;
    // The batch size.
    int mBatchSize;
    // The max sequence length in the contiguous Kv cache.
    int mMaxSeqLenCacheKv;
    // The max q sequence length.
    int mMaxSeqLenQ;
    // The max kv sequence length.
    int mMaxSeqLenKv;
    // The attention window size for sliding window attention.
    int mAttentionWindowSize;
    // The sum of sequence lengths for Q and K/V. (Only used when mSupportsVarSeqLens = true)
    int mSumOfSeqLensQ;
    int mSumOfSeqLensKv;
    // The maximum number of pages per sequence in the paged-kv buffer.
    int mMaxNumPagesPerSeqKv;
    // The number of tokens per pageKv.
    int mNumTokensPerPage;
    // The number of pages in memory pool.
    int mNumPagesInMemPool;
    // The number of multiProcessor for the GPU.
    int mMultiProcessorCount;
    // Scaling factor for Q.
    float mScaleQ;
    // The start token index in SF tensor. Used for FP4 SF offset calculation in generation phase kernel when inflight
    // batching is enabled.
    int mSfStartTokenIdx;

    // The SF scale for Kv.
    float mScaleSfKv;
    // The cuda stream.
    cudaStream_t stream;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Parameters that might be updated when selecting kernels.
struct TllmGenSelectKernelParams
{
    // The headDimV per CTA, which is only used by MLA generation kernels currently.
    int mHeadDimPerCtaV;
    // The maximum number of headsQPerCta that will be processed in one CTA.
    int mMaxNumHeadsQPerKvInCta;
    // Enable the multiCtasKvMode or not.
    bool mMultiCtasKvMode;
    // Reuse smemK for V or not (only work with MLA generation kernels).
    bool mReuseSmemKForV;
    // Do we need to select a new kernel as the parameters have been updated.
    bool mSelectNewKernel;
    // The tile scheduler.
    TileScheduler mTileScheduler;

    // The constructor.
    TllmGenSelectKernelParams(TllmGenFmhaRunnerParams params)
        : mHeadDimPerCtaV(params.mHeadDimV)
        , mMaxNumHeadsQPerKvInCta(1)
        , mMultiCtasKvMode(params.mMultiCtasKvMode)
        , mReuseSmemKForV(false)
        , mSelectNewKernel(false)
        , mTileScheduler(params.mTileScheduler){};
};

} // namespace kernels
} // namespace tensorrt_llm
