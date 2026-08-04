/*
 * Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/kvCacheType.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"

namespace tensorrt_llm::runtime
{
class TllmRuntime;
class MulticastTensor;
} // namespace tensorrt_llm::runtime

namespace tensorrt_llm::batch_manager
{

namespace kv_cache_manager
{
class BaseKVCacheManager;
}

class TransformerBuffers
{
public:
    using SizeType32 = runtime::SizeType32;
    using TensorPtr = runtime::ITensor::SharedPtr;
    using TensorMap = runtime::StringPtrMap<runtime::ITensor>;

    static constexpr auto kCrossAttentionMaskTensorName = "cross_attention_mask";
    static constexpr auto kCrossAttentionPackedMaskTensorName = "cross_attention_packed_mask";
    static constexpr auto kPositionIdsTensorName = "position_ids";
    static constexpr auto kCacheIndirectionsTensorName = "cache_indirection";
    static constexpr auto kHostPastKeyValueLengthsTensorName = "host_past_key_value_lengths";
    static constexpr auto kHostSinkTokenLengthTensorName = "host_sink_token_length";
    static constexpr auto kHostMaxAttentionWindowSizesTensorName = "host_max_attention_window_sizes";
    static constexpr auto kHostContextProgressTensorName = "host_context_progress";
    static constexpr auto kKvCacheBlockOffsetsTensorName = "kv_cache_block_offsets";
    static constexpr auto kHostKvCacheBlockOffsetsTensorName = "host_kv_cache_block_offsets";
    static constexpr auto kCrossKvCacheBlockOffsetsTensorName = "cross_kv_cache_block_offsets";
    static constexpr auto kHostCrossKvCacheBlockOffsetsTensorName = "host_cross_kv_cache_block_offsets";
    static constexpr auto kHostCrossKvCachePoolPointersTensorName = "host_cross_kv_cache_pool_pointers";
    static constexpr auto kHostCrossKvCachePoolMappingTensorName = "host_cross_kv_cache_pool_mapping";
    static constexpr auto kSkipCrossAttentionBlocksTensorName = "skip_cross_attn_blocks";

    TensorPtr pastKeyValueLengths; // Host tensor
    TensorPtr positionIds;

    // max kv cache lengths.
    TensorPtr maxAttentionWindows;
    // sink token lengths.
    TensorPtr sinkTokenLengths;
    TensorPtr cacheIndirection;
    TensorPtr kvCacheBlockOffsetsHost;   // [numPools, maxBatch * maxBeamWidth, 2, maxBlocksPerSeq]
    TensorPtr kvCacheBlockOffsetsDevice; // [numPools, maxBatch * maxBeamWidth, 2, maxBlocksPerSeq]
    TensorPtr contextProgressHost;

    // Cross attention buffers
    TensorPtr crossKvCacheBlockPoolPointers = nullptr;
    TensorPtr crossKvCacheBlockPoolMapping = nullptr;
    TensorPtr crossKvCacheBlockOffsetsHost = nullptr;
    TensorPtr crossKvCacheBlockOffsetsDevice = nullptr;
    TensorPtr crossAttentionMaskCopySrcOffsets = nullptr; // [maxNumRequest] pinned memory.
    TensorPtr crossAttentionMaskCopyDstOffsets = nullptr; // [maxNumRequest] pinned memory.
    TensorPtr crossAttentionMaskCopySizes = nullptr;      // [maxNumRequest] pinned memory.
    TensorPtr crossAttentionMaskDevice = nullptr;         // [maxNumTokens, maxEncoderOutputLen]
    // This is created to allow mixed memory types of crossAttentionMask (i.e. CPU and GPU).
    TensorPtr crossAttentionMaskPinnedHost = nullptr; // [maxNumTokens, maxEncoderOutputLen]
    // See more details in tensorrt_llm/kernels/contextFusedMultiHeadAttention/fmhaPackedMask.cu.
    // The attention packed mask for FMHA where each bit represents one mask.
    TensorPtr crossAttentionPackedMaskDevice
        = nullptr; // [maxBatchSize, maxInputLengthInBatch, roundUp(maxEncoderOutputLen, 32)]
    // The number of cumulative Q sequence lengths in the mask input, which is used to get mask offsets for different
    // requests.
    TensorPtr crossAttentionCuQSeqLensDevice = nullptr; // [maxBatchSize + 1]
    // The number of cumulative Q sequence lengths in the packed mask, which is used to get mask offsets for different
    // requests.
    TensorPtr crossAttentionPackedMaskCuMaskRowsDevice = nullptr; // [maxBatchSize + 1]

    TensorPtr cacheIndirBatchedCopySrcOffsets;
    TensorPtr cacheIndirBatchedCopyDstOffsets;
    TensorPtr cacheIndirBatchedCopySizes;

    TensorPtr fillValuesAlt;
    TensorPtr fillValuesAltDevice;
    TensorPtr seqSlotsAlt;
    TensorPtr seqSlotsAltDevice;
    TensorPtr skipCrossAttnBlocks;

    std::shared_ptr<tensorrt_llm::runtime::MulticastTensor> gemmAllReduceOutput;

    TransformerBuffers(SizeType32 maxBatchSize, SizeType32 maxBeamWidth,
        std::vector<SizeType32> const& maxAttentionWindowVec, SizeType32 maxAttentionWindow, SizeType32 sinkTokenLen,
        runtime::TllmRuntime const& runtime, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig);

    void reshape(SizeType32 numSequences, SizeType32 numInputTokens);

    void reshapeKvTensors(SizeType32 maxBatchSize, SizeType32 maxBeamWidth, SizeType32 maxBlocksPerSeq,
        kv_cache_manager::CacheType kvCacheType, SizeType32 numPools, runtime::BufferManager const& manager);

    void getBuffers(TensorMap& inputBuffers, TensorMap& outputBuffers, runtime::ModelConfig const& modelConfig) const;

    void copyPositionIds(runtime::TllmRuntime const& runtime, std::vector<SizeType32> const& positionIdsHost,
        bool isChatGlm, TensorPtr const& decoderPositionIds);

    void copyKvBlockOffsets(RequestVector const& contextRequests, RequestVector const& genRequests,
        kv_cache_manager::BaseKVCacheManager const* kvCacheManager,
        kv_cache_manager::BaseKVCacheManager const* crossKvCacheManager, runtime::BufferManager const& manager);

    // Copy CacheIndirection from `decoderCacheIndirectionOutput` to `this->cacheIndirection`
    void copyCacheIndirection(RequestVector const& genRequests, TensorPtr const& decoderCacheIndirectionOutput,
        runtime::CudaStream const& stream);

    void copyCrossAttentionMasks(RequestVector const& contextRequests, RequestVector const& genRequests,
        TensorPtr const& decoderContextLengthsDevice, TensorPtr const& encoderInputLengths,
        SizeType32 maxDecoderContextLength, SizeType32 maxEncoderInputLengthInBatch,
        runtime::TllmRuntime const& runtime);

    void copySkipCrossAttnBlocks(bool const& _skipCrossAttnBlocks, runtime::TllmRuntime const& runtime);

private:
    SizeType32 maxInputLen;
    SizeType32 maxEncoderOutputLen;
    SizeType32 maxNumTokens;
};

} // namespace tensorrt_llm::batch_manager
