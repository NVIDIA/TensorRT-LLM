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

#include <cuda.h>
#include <trtllm/dev/FastMath.h>
#include <vector_types.h>

// NOTE: Keep this code dependency free. It has to be included by the device code and has to be
// compilable with NVRTC.

namespace fmha {

struct KernelParams {
#ifdef TLLM_ENABLE_CUDA
  // TMA descriptor for Q.
  CUtensorMap tmaQ_;
  // TMA descriptor for K.
  CUtensorMap tmaK_;
  // TMA descriptor for DSv4 sparse MLA sliding-window KV pool. Same format as tmaK_.
  CUtensorMap tmaKSlidingWindowKvPool_;
  // TMA descriptor for V.
  CUtensorMap tmaV_;
  // The descriptor for O.
  CUtensorMap tmaO_;

  // For FP4 KV cache, additional scaling factors are needed.

  // TMA descriptor for K scaling factor.
  CUtensorMap tmaKSf_;
  // TMA descriptor for V scaling factor.
  CUtensorMap tmaVSf_;
#endif

  // grid dimensions, these might differ from actual grid the kernel is launched with
  // for persistent kernels on Hopper GPUs.
  int32_t logicalGridDimX, logicalGridDimY, logicalGridDimZ;

  // The output pointer (used by STG for last tile).
  void* ptrO;
  // The output SF pointer (used for FP4 output).
  void* ptrSfO;
  // The attention sinks pointer (additional value per head in the denominator of the softmax).
  float const* ptrAttentionSinks;
  // The cumulative sequence lengths for Q.
  int32_t const* ptrCumSeqLensQ;
  // The cumulative sequence lengths for K/V.
  int32_t const* ptrCumSeqLensKv;
  // The packed custom mask.
  uint32_t const* ptrCustomMask;
  // The packed custom mask's offsets of each sequence.
  int64_t const* ptrCustomMaskOffsets;
  // The debug output matrix O
  float* ptrDebugO;
  // The first sparseMask offsets in the Kv sequence dimension.
  int32_t const* ptrFirstSparseMaskOffsetsKv;
  // The counter for the multiCtasKv mode.
  int32_t* ptrMultiCtasKvCounter;
  // The device output scale for FP8 quantization. Only needed by trt-llm fp8 kernels as the sca-
  // les have to be on the device currently.
  float const* ptrOutputScale;
  // The page indexes of the paged-kv buffer with shape of [batchSize, 2, maxNumPagesPerSeq].
  int32_t const* ptrPageIdxKv;
  // The partial matrix O for each CtaKv when the multiCtasKv mode is enabled.
  void* ptrPartialO;
  // The partial softmax stats (max/sum)for each CtaKv when the multiCtasKv mode is enabled.
  float2* ptrPartialStats;
  // The scaling factors for K.
  float const* ptrSageAttnSfsK;
  // The scaling factors for P.
  float const* ptrSageAttnSfsP;
  // The scaling factors for Q.
  float const* ptrSageAttnSfsQ;
  // The scaling factors for V.
  float const* ptrSageAttnSfsV;
  // The device scaling factor for softmax (multiplied by log2 to use faster exp2). Only needed by
  // trt-llm fp8 kernels as the scales have to be on the device currently.
  float const* ptrScaleSoftmaxLog2;
  // The SF scale for Kv on device. Only needed by trt-llm kernels as the scales have to be on the
  // device currently.
  float const* ptrScaleSfKv;
  // The SF scale for O on device. Only needed by trt-llm kernels as the scales have to be on the
  // device currently.
  float const* ptrScaleSfO;
  // The sequence lengths for K/V. Required by pagedKv kernels to avoid unnecessary computation
  // based on (ptrCumSeqLensKv[batchIdx + 1] - ptrCumSeqLensKv[batchIdx]).
  int32_t const* ptrSeqLensKv;
  // When collecting skip softmax stats, store them here
  // Note that softmax and BMM2 are skipped at different granularity (warp vs tile)
  // [0] -> skipped softmax warp blocks; [1] -> total softmax warp blocks.
  // [2] -> skipped BMM2s; [3] -> total BMM2s.
  int32_t* ptrSkipSoftmaxStats;
  // The softmax stats buffer.
  float2* ptrSoftmaxStats;
  // The variable sparseMla topK lengths with shape of [numTokensQ]
  //  where each tokenQ has a corresponding topK length.
  int32_t const* ptrSparseMlaTopKLens;

  // The attention window size for sliding window attention.
  int32_t mAttentionWindowSize;
  // The batch size
  int32_t mBatchSize;
  // The chunked attention size in log2.
  int32_t mChunkedAttentionSizeLog2;
  // The factor to add to the maximum value to increase the probability
  //   of skip correction during next iterations.
  float mInflateMax;
  // The log of the Sage Attention block size for K.
  int32_t mLogNumEltsPerSageAttnBlkK;
  // The log of the Sage Attention block size for P.
  int32_t mLogNumEltsPerSageAttnBlkP;
  // The log of the Sage Attention block size for Q.
  int32_t mLogNumEltsPerSageAttnBlkQ;
  // The log of the Sage Attention block size for V.
  int32_t mLogNumEltsPerSageAttnBlkV;
  // The sequence lengths for Q and K/V.
  int32_t mMaxSeqLenQ, mMaxSeqLenKv;
  // The maximum number of CTAs for Q.
  int32_t mMaxNumCtasQ;
  // The maximum number of CTAs for K/V.
  int32_t mMaxNumCtasKv;
  // The maximum number of pages per sequence for paged-kv buffer.
  int32_t mMaxNumPagesPerSeqKv;
  // The number of heads for K/V.
  int32_t mNumHeadsKv;
  // The number of heads for Q.
  int32_t mNumHeadsQ;
  // The number of Q heads per K/V head (i.e. mNumHeadsQ / mNumHeadsKv).
  int32_t mNumHeadsQPerKv;
  // The number of headsQ per K/V head as a fast_mod_div divisor.
  trtllm::dev::fast_mod_div mNumHeadsQPerKvDivisor{1};
  // The hidden size of O.
  int64_t mNumHiddenEltsO;
  // The total number of pages in the paged-kv memory pool.
  int32_t mNumPagesInMemPool;
  // The number of tokensQ per CTA (used for groupsHeadsTokensQ generation kernel).
  int32_t mNumTokensPerCtaQ;
  // The number of tokens per page (used if dynamic numTokensPerPage is enabled).
  int32_t mNumTokensPerPageLog2;
  // The output scale for FP8 quantization.
  float mOutputScale;
  // The scaling factor for softmax (multiplied by log2 to use faster exp2).
  float mScaleSoftmaxLog2;
  // The SF scale for Kv.
  float mScaleSfKv;
  // The SF scale for O.
  float mScaleSfO;
  // Threshold to decide whether warp skips softmax ops
  float mSkipSoftmaxThresholdScaleFactor;
  // The sparse attention topK value.
  int32_t mSparseAttnTopK;
  // The start token index in SF tensor. Used for FP4 SF offset calculation in generation phase
  // kernel when inflight batching is enabled in TRT-LLM.
  int32_t mStartTokenIdxSfO;
  // The sum of sequence lengths for Q and K/V.
  int32_t mSumOfSeqLensQ, mSumOfSeqLensKv;
  // The flag to use block sparse attention.
  bool mUseBlockSparseAttention;
  // Whether the indices for K & V pages are shared as unified index (vLLM/FlashInfer).
  bool mUsesSharedPagedKvIdx{false};

  // Note: No implementation functions here as they use STL which is not available in NVRTC
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace fmha
