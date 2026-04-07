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

#include "FmhaOptions.h"
#include <unordered_map>

#ifdef TLLM_GEN_EXPORT_INTERFACE
#ifdef TLLM_GEN_EXPORT_FLASHINFER
#include "flashinferMetaInfo.h"
#endif // TLLM_GEN_EXPORT_FLASHINFER
#endif // TLLM_GEN_EXPORT_INTERFACE

namespace fmha {
////////////////////////////////////////////////////////////////////////////////////////////////////
//
// FmhaData
//
////////////////////////////////////////////////////////////////////////////////////////////////////

struct FmhaData {
  struct MetaData {
    // The cumulative sequence lengths for Q. The shape is [batchSize + 1].
    int32_t const* cumSeqLensQPtrD;

    // The cumulative sequence lengths for K/V. The shape is [batchSize + 1], or [numHeadsKv *
    // (batchSize + 1)] for sparse attention.
    int32_t const* cumSeqLensKvPtrD;

    // The first sparseMask tile offset in the Kv sequence dimension, and adjusted to multiple
    // of tileSizeKvPerCta. The shape is [batchSize]
    int32_t const* firstSparseMaskOffsetsKvPtrD;

    // The value added to the new maximum value in case it is larger than the previous maximum.
    // This way we artificially inflate the maximum value (without additional compute cost)
    // to increase the probability of skip correction at next iterations of the pipeline loop.
    // [0..8] is a reasonable range of values for this.
    float inflateMax;

    // page indexes of Kv. The shape is [batchSize, 2, maxNumPagesPerSeqKv], or [numHeadsKv *
    // batchSize, 2, maxNumPagesPerSeqKv] for sparse attention.
    int32_t const* kvPageIdxD;

    // The sequence lengths of K/V. The shape is [batchSize], or [numHeadsKv * batchSize] for sparse
    // attention.
    int32_t const* seqLensKvD;

    // Start token index in the O scaling-factor tensor. Used for FP4 SF offset in generation when
    // inflight batching is enabled (TRT-LLM). Context uses 0.
    int32_t startTokenIdxSfO{0};
  };

  struct Scales {
    // FP4 scaling factors for KV cache
    void const* kSfBasePtr;
    void const* vSfBasePtr;

    // FP4 scaling factors for input/output
    float kvSfScale;
    float oSfScale;
    float const* kvSfScaleD;
    float const* oSfScaleD;

    // output scale for FP8 per tensor attention
    float const* outputScaleD;

    // Sage Attn scaling factors for Q, K, P, V
    float const* sageAttnSfsQPtrD;
    float const* sageAttnSfsKPtrD;
    float const* sageAttnSfsPPtrD;
    float const* sageAttnSfsVPtrD;

    // The scaling factor applied after Q*K^T (i.e. 1/sqrt(headDim).
    float softmaxScale;
    float const* scaleSoftmaxLog2D;

    // FP4 scaling factors for O
    void* oSfPtrD;
  };

  struct InputBuffers {
    // The input tensor for Q. The shape is [sumOfSeqLensQ, hiddenDimQ]
    void const* qBasePtr;

    // The input tensor for K, V. The shape is [sumOfSeqLensKv, hiddenDimKv] or
    // [sumOfSeqLensKv * numHeadsKv, hiddenDimKv] for sparse attention.
    void const* kBasePtr;
    void const* vBasePtr;

    // Attention sinks
    float const* attentionSinksPtrD;

    // Custom mask
    uint32_t const* customMaskPtrD;
    int64_t const* customMaskOffsetsPtrD;
  };

  struct OutputBuffers {
    // Device counters used to synchronize CTAs before the final reduction in multiCtasKvMode.
    int32_t* multiCtasKvCounterPtrD;

    // Partial output and softmaxStats for each CtaKv when the multiCtasKv mode is enabled.
    void* partialOPtrD;
    float2* partialStatsPtrD;

    // Skip-softmax statistics.
    int* skipSoftmaxStatsPtrD;

    // Softmax stats.
    float2* softmaxStatsD;

    // Debug
    void* oDebugPtrD;

    // The output tensor for O. The shape is [sum(seqLensQ), hiddenDimO]
    void* oPtrD;

  };

  MetaData mMetaData;
  Scales mScales;
  InputBuffers mInputBuffers;
  OutputBuffers mOutputBuffers;
};

////////////////////////////////////////////////////////////////////////////////////////////////////
//
// FmhaInterface
//
////////////////////////////////////////////////////////////////////////////////////////////////////

class FmhaInterface {
public:
  using ModuleCache = std::unordered_map<std::string, std::tuple<CUmodule, CUfunction>>;

  FmhaInterface(bool exportsCubin = false, int32_t numRotations = 1, bool verbose = false)
    : mExportsCubin(exportsCubin)
    , mNumRotations(numRotations)
    , mVerbose(verbose) {
    setVerbose(verbose);
  }

  // Set the verbosity level for logging. When false, TLLM_LOG_INFO messages are suppressed.
  void setVerbose(bool verbose);

  void generateAndCompileKernel(FmhaConfig& fmhaConfig) const;

  std::string getKernelName(FmhaConfig const& fmhaConfig) const;

  int32_t run(FmhaConfig const& config,
              FmhaData& data,
              cudaStream_t cudaStream,
              int32_t multiProcessorCount,
              int32_t rotatedKernelInstanceIdx) const;

private:
  // Whether to export the cubin file.
  bool mExportsCubin;
  // The number of rotations
  int32_t mNumRotations;
  // Whether to log info messages.
  bool mVerbose;
};

} // namespace fmha
