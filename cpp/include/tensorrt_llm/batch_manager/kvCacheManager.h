/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/batch_manager/kvCacheConnector.h"
#include "tensorrt_llm/batch_manager/kvCacheEventManager.h"
#include "tensorrt_llm/batch_manager/kvCacheType.h"
#include "tensorrt_llm/batch_manager/llmRequest.h" // TODO forward declare
#include "tensorrt_llm/common/optionalRef.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/executor/transferAgent.h"
#include "tensorrt_llm/kernels/kvCacheIndex.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include <NvInferRuntime.h>

#include <array>
#include <cstdint>
#include <limits>
#include <list>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>

namespace kvc = tensorrt_llm::executor::kv_cache;

namespace tensorrt_llm::batch_manager::eviction_policy
{
class BaseEvictionPolicy;
} // namespace tensorrt_llm::batch_manager::eviction_policy

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

static constexpr SizeType32 kPrimaryLevel = 0;

static constexpr SizeType32 kSecondaryLevel = 1;

// Extra block buffer allocated for SWA to be able to always keep "window size"
// tokens held in the blocks.
static constexpr SizeType32 kSWAExtraBlock = 1;

class KVCacheBlock;
class BlockManager;
class KVCacheManager;
class KVCacheTransferManager;

using SizeType32 = tensorrt_llm::runtime::SizeType32;
using TokenIdType = tensorrt_llm::runtime::TokenIdType;
using VecTokens = std::vector<TokenIdType>;
using BeamTokens = std::vector<VecTokens>;
using BlockPtr = std::shared_ptr<KVCacheBlock>;
using FreeBlocksQueue = std::list<BlockPtr>;
using UniqueToken = tensorrt_llm::runtime::UniqueToken;
using VecUniqueTokens = tensorrt_llm::runtime::VecUniqueTokens;
using LoraTaskIdType = tensorrt_llm::runtime::LoraTaskIdType;
using BlocksPerWindow = std::map<SizeType32, std::tuple<SizeType32, SizeType32>>;
using CacheSaltIDType = tensorrt_llm::runtime::CacheSaltIDType;

// Type alias for multimodal hash key (hash array + start offset)
using MmKey = std::pair<std::array<uint8_t, 32>, SizeType32>;

template <typename T>
using OptionalRef = tensorrt_llm::common::OptionalRef<T>;

//! \brief Split vector into list of blocks of given size.
//! \param vec vector to split
//! \param usableSize part of the vector that is processed
//! \param elementsPerBlock desired size of blocks
//! \param allowPartial whether to append a block smaller than `elementsPerBlock` at the end
//! \return list of blocks
template <typename T>
std::list<std::vector<T>> chopVectorIntoBlocks(
    std::vector<T> const& vec, SizeType32 usableSize, SizeType32 elementsPerBlock, bool allowPartial)
{
    TLLM_CHECK_WITH_INFO(
        usableSize <= static_cast<SizeType32>(vec.size()), "usableSize=%d > %ld=vec.size()", usableSize, vec.size());
    std::list<std::vector<T>> blockedVectors;
    auto const vecEnd = vec.begin() + usableSize;
    for (auto begin = vec.begin(); begin < vecEnd; begin += elementsPerBlock)
    {
        auto blockSize = std::min(elementsPerBlock, static_cast<SizeType32>(std::distance(begin, vecEnd)));
        auto end = begin + blockSize;
        if (blockSize == elementsPerBlock || allowPartial)
        {
            blockedVectors.emplace_back(begin, end);
        }
    }
    return blockedVectors;
}

struct TempAttentionWindowInputs
{
    bool pagedContextFMHA;
    SizeType32 maxInputLen;
    SizeType32 maxNumTokens;
};

struct WindowSizeMetadata
{
    SizeType32 allottedPrimaryBlocks;    // Number of primary blocks allotted to the windowSize
    SizeType32 allottedSecondaryBlocks;  // Number of secondary blocks allotted to the windowSize
    SizeType32 absolutePoolsOffset;      // cumulative number of pools up to manager
    SizeType32 numPools;                 // number of managed pools
    SizeType32 maxTokenNum;              // Maximum token length per sequence (TODO: account for streamLLM)
    SizeType32 maxBlocksPerSeq;          // Maximum number of blocks per sequence
    SizeType32 maxNumBlocks;             // Number of primary+secondary blocks allotted to the windowSize
    SizeType32 temporaryAttentionWindow; // Temporary kv cache length per sequence.
                                         // Only needed when chunked context + sliding window attention are used
                                         // together. And it should only be considered when allocating blocks.
    SizeType32 windowSize;
    bool isSWA;

    std::string toString()
    {
        return tensorrt_llm::common::fmtstr(
            "WindowSizeMetadata{ .allottedPrimaryBlocks=%d, .allottedSecondaryBlocks=%d, .absolutePoolsOffset=%d, "
            ".numPools=%d, .maxTokenNum=%d, .maxBlocksPerSeq=%d, .maxNumBlocks=%d, .temporaryAttentionWindow=%d, "
            ".windowSize=%d, .isSWA=%d }",
            allottedPrimaryBlocks, allottedSecondaryBlocks, absolutePoolsOffset, numPools, maxTokenNum, maxBlocksPerSeq,
            maxNumBlocks, temporaryAttentionWindow, windowSize, isSWA);
    }
};

std::vector<MmKey> generateBlockHashExtraKeys(
    tensorrt_llm::batch_manager::LlmRequest const& llmRequest, SizeType32 startTokenIdx, SizeType32 endTokenIdx);

struct BlockKey
{
    bool usesExtraIds = false;
    std::optional<LoraTaskIdType> loraTaskId = std::nullopt;
    VecUniqueTokens uniqueTokens;

    // Extra keys for multimodal data (similar to VLLM's approach)
    // Each extra key is a pair of (mm_hash, start_offset_in_block)
    std::vector<MmKey> extraKeys;
    std::optional<CacheSaltIDType> cacheSaltID = std::nullopt;

    BlockKey() = default;

    explicit BlockKey(VecTokens const& tokens, std::optional<LoraTaskIdType> loraTaskId = std::nullopt)
        : loraTaskId{loraTaskId}
    {
        uniqueTokens.reserve(tokens.size());
        for (auto const& token : tokens)
        {
            uniqueTokens.push_back(UniqueToken{token, 0});
        }
    }

    explicit BlockKey(bool usesExtraIds, std::optional<LoraTaskIdType> loraTaskId, VecUniqueTokens uniqueTokens,
        std::vector<MmKey> extraKeys = {}, std::optional<CacheSaltIDType> cacheSaltID = std::nullopt)
        : usesExtraIds{usesExtraIds}
        , loraTaskId{loraTaskId}
        , uniqueTokens{std::move(uniqueTokens)}
        , extraKeys{std::move(extraKeys)}
        , cacheSaltID{cacheSaltID}
    {
    }

    bool operator==(BlockKey const& other) const noexcept;

    int partialMatch(BlockKey const& other) const noexcept
    {
        SizeType32 numMatched{0};
        if (loraTaskId == other.loraTaskId && extraKeys == other.extraKeys && cacheSaltID == other.cacheSaltID)
        {
            auto [matchEnd, otherMatchEnd] = std::mismatch(
                uniqueTokens.begin(), uniqueTokens.end(), other.uniqueTokens.begin(), other.uniqueTokens.end());
            numMatched = std::distance(uniqueTokens.begin(), matchEnd);
        }
        return numMatched;
    }
};

std::vector<BlockKey> buildBlockKeys(std::list<VecUniqueTokens>& blockedUniqueTokens, LlmRequest const& llmRequest);

// Implement hash functor for BlockKey.
// This allows us to use unordered_map with BlockKey as key.
// Based on https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector/72073933#72073933
struct BlockKeyHasher
{
    [[nodiscard]] static size_t hash(BlockKey const& blockKey, std::size_t parentHash = 0) noexcept;

    std::size_t operator()(BlockKey const& blockKey, std::size_t parentHash = 0) const noexcept
    {
        return hash(blockKey, parentHash);
    }
};

using NextBlockMap = std::unordered_map<BlockKey, BlockPtr, BlockKeyHasher>;

struct KvCacheStats
{
    // Number of maximum available blocks in the primary memory pool. This is determined and set by available primary
    // memory. See calculateMaxNumBlocks for details.
    SizeType32 maxNumBlocks;
    // Number of free blocks in the primary memory pool.
    SizeType32 freeNumBlocks;
    // Number of used blocks in the primary memory pool. usedNumBlocks = maxNumBlocks - freeNumBlocks.
    SizeType32 usedNumBlocks;
    SizeType32 toksPerBlock;
    // Total number of blocks allocated by all requests.
    SizeType32 allocTotalBlocks;
    // Number of new blocks that were allocated.
    SizeType32 allocNewBlocks;
    // Number of blocks that were matched and reused.
    SizeType32 reusedBlocks;
    // Number of blocks that were not matched and not reused.
    SizeType32 missedBlocks;
    // Measuring the KV Cache reuse rate. cacheHitRate = reusedBlocks / (reusedBlocks + missedBlocks).
    float cacheHitRate;
    // Number of free blocks for every configured attention-window size.
    std::map<SizeType32, SizeType32> numFreeBlocksPerWindowSize;
    // GPU bytes allocated for KV-cache
    std::size_t allocatedBytes{};
};

// Basic building block of a paged KV cache - a single
// cache block. This class just holds metadata, no pointers
// since it is reused across all layers.
class KVCacheBlock
{
public:
    using IdType = std::int32_t;

    static constexpr IdType kCachedBlocksRootId = -1;

    explicit KVCacheBlock(IdType blockId, kernels::KVCacheIndex blockIdx);

    void startScheduling();

    [[nodiscard]] IdType getBlockId() const;

    [[nodiscard]] NextBlockMap getNextBlocks() const;

    [[nodiscard]] kernels::KVCacheIndex::UnderlyingType getMemoryPoolBlockIndex() const;

    [[nodiscard]] bool isPrimary() const;

    void swapMemoryPoolBlockOffset(std::shared_ptr<KVCacheBlock> otherBlock);

    void incRefCount();

    void decRefCount();

    void decSchedulingRefCount();

    [[nodiscard]] bool hasRefs() const;

    [[nodiscard]] bool hasSchedulingRefs() const;

    void setBlockKey(BlockKey const& blockKey, bool isFull);

    BlockKey getBlockKey();

    [[nodiscard]] VecUniqueTokens const& getUniqueTokens() const;

    BlockPtr const& getPrevBlock() const;

    void setPrevBlock(BlockPtr prevBlock);

    BlockPtr const& getPrevBlockInSeq() const;

    void setPrevBlockInSeq(BlockPtr prevBlock);

    void addNextBlock(BlockKey const& blockKey, BlockPtr block);

    void removeNextBlock(BlockKey const& blockKey);

    //! \brief Find block matching blockKey. If allowPartial is true, the returned block may match only a prefix of
    //! blockKey.
    //! @return tuple of [partialMatch, numMatched, block], partialMatch is true if not all the tokens of the block were
    //! matched.
    [[nodiscard]] std::tuple<bool, SizeType32, BlockPtr> findMatchingBlock(
        BlockKey const& blockKey, bool enablePartialReuse, bool copyOnPartialReuse) const;

    //! \brief Free block from previous block if present.
    void freeLeafBlock();

    [[nodiscard]] bool isFull() const;

    [[nodiscard]] bool isShared() const;

    [[nodiscard]] bool isLeaf() const;

    void setPriority(executor::RetentionPriority priority);

    [[nodiscard]] executor::RetentionPriority getPriority() const;

    void setDurationMs(std::optional<std::chrono::milliseconds> durationMs);

    [[nodiscard]] std::optional<std::chrono::milliseconds> getDurationMs() const;

    void setExpirationTime(std::optional<std::chrono::steady_clock::time_point::duration> expirationTime);

    [[nodiscard]] std::optional<std::chrono::steady_clock::time_point::duration> getExpirationTime() const;

    void setHash(size_t hash);

    // set hash automatically from block key and previous block in sequence
    void setHash();

    size_t getHash() const;

private:
    // Linear ID of block independent of pool
    IdType mBlockId;

    // Index of block in memory pool backing this block
    // Choice of pool is encoded into the type
    kernels::KVCacheIndex mMemoryPoolBlockIndex;

    // Number of references to the block
    SizeType32 mRefCount;

    // Number of references to the block
    SizeType32 mSchedulingRefCount;

    // Key of this block in mNextBlocks map in block pointed to by mPrevBlock
    BlockKey mBlockKey;

    // Previous block in reuse tree, or nullptr if not reusing
    BlockPtr mPrevBlock;

    // Previous block in sequence, == nullptr for first block, == mPrevBlock if reusing and not first
    BlockPtr mPrevBlockInSeq;

    // Next block(s) in sequence(s)
    NextBlockMap mNextBlocks;

    // Iterator pointing to this block in mFreeBlocks.
    std::optional<FreeBlocksQueue::iterator> mFreeBlockIterator;

    // Flag indicating if block is full
    bool mIsFull;

    // Priority of the block
    executor::RetentionPriority mPriority;
    // Duration that the block's priority level applies for
    std::optional<std::chrono::milliseconds> mDurationMs;
    // Expiration time of the block
    std::optional<std::chrono::steady_clock::time_point::duration> mExpirationTime;
    // Hash for the event manager
    size_t mHash;
};

class GenerationRequest
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;

    explicit GenerationRequest(LlmRequest::RequestIdType requestId, SizeType32 numTokens, SizeType32 beamWidth,
        std::map<SizeType32, WindowSizeMetadata> const& windowSizeToMetadata,
        executor::KvCacheRetentionConfig kvCacheRetentionConfig = executor::KvCacheRetentionConfig())
        : mRequestId(requestId)
        , mNumTokens(numTokens)
        , mBeamWidth(beamWidth)
        , mKvCacheRetentionConfig(std::move(kvCacheRetentionConfig))
        , mNumFrontBlocksRemoved(0)
    {
        auto const numWindowSizes = windowSizeToMetadata.size();
        mCacheBlockIds.reserve(numWindowSizes);
        mCacheBlockIndices.reserve(numWindowSizes);
        for (auto const [windowSize, metadata] : windowSizeToMetadata)
        {
            mCacheBlockIds[windowSize] = std::vector<std::vector<KVCacheBlock::IdType>>(beamWidth);
            auto const numPools = metadata.numPools;
            auto const maxBlocks = metadata.maxBlocksPerSeq;
            mCacheBlockIndices[windowSize]
                = runtime::BufferManager::cpu(runtime::ITensor::makeShape({numPools, beamWidth, 2, maxBlocks}),
                    runtime::TRTDataType<tensorrt_llm::kernels::KVCacheIndex>::value);
            auto cacheBlockIdsRange
                = runtime::BufferRange<tensorrt_llm::kernels::KVCacheIndex>(*mCacheBlockIndices.at(windowSize));
            std::fill(cacheBlockIdsRange.begin(), cacheBlockIdsRange.end(),
                tensorrt_llm::kernels::KVCacheIndex{
                    std::numeric_limits<tensorrt_llm::kernels::KVCacheIndex::UnderlyingType>::max()});
        }
    }

    void addNewTokens(SizeType32 n)
    {
        mNumTokens += n;
    }

    void removeTokens(SizeType32 n)
    {
        TLLM_CHECK(n <= mNumTokens);
        TLLM_CHECK(mNumTokens - n >= 0);
        mNumTokens -= n;
    }

    [[nodiscard]] LlmRequest::RequestIdType getRequestId() const
    {
        return mRequestId;
    }

    [[nodiscard]] SizeType32 getNumTokens() const
    {
        return mNumTokens;
    }

    [[nodiscard]] SizeType32 getNumFrontBlocksRemoved() const
    {
        return mNumFrontBlocksRemoved;
    }

    [[nodiscard]] SizeType32 getBeamWidth() const
    {
        return mBeamWidth;
    }

    [[nodiscard]] std::vector<std::vector<SizeType32>> const& getCacheBlockIds(SizeType32 windowSize) const
    {
        return mCacheBlockIds.at(windowSize);
    }

    [[nodiscard]] runtime::ITensor& getCacheBlockIndices(SizeType32 windowSize)
    {
        return *(mCacheBlockIndices.at(windowSize));
    }

    [[nodiscard]] runtime::ITensor const& getCacheBlockIndices(SizeType32 windowSize) const
    {
        return *(mCacheBlockIndices.at(windowSize));
    }

    void addCacheBlock(SizeType32 windowSize, SizeType32 beamIdx, KVCacheBlock::IdType blockId)
    {
        mCacheBlockIds.at(windowSize).at(beamIdx).push_back(blockId);
    }

    void changeCacheBlock(
        SizeType32 windowSize, SizeType32 beamIdx, SizeType32 pagedBlockIdx, KVCacheBlock::IdType blockId)
    {
        mCacheBlockIds.at(windowSize).at(beamIdx).at(pagedBlockIdx) = blockId;
    }

    void clearCacheBlocks(SizeType32 windowSize)
    {
        for (auto& beamBlockIds : mCacheBlockIds.at(windowSize))
        {
            beamBlockIds.clear();
        }
        mNumFrontBlocksRemoved = 0;
    }

    void removeFrontBlock(SizeType32 windowSize)
    {
        ++mNumFrontBlocksRemoved;
    }

    void removeLastBlock(SizeType32 windowSize)
    {
        for (auto& beamBlockIds : mCacheBlockIds.at(windowSize))
        {
            beamBlockIds.pop_back();
        }
    }

    [[nodiscard]] executor::RetentionPriority getDecodeRetentionPriority() const
    {
        return mKvCacheRetentionConfig.getDecodeRetentionPriority();
    }

    [[nodiscard]] std::optional<std::chrono::milliseconds> getDecodeDurationMs() const
    {
        return mKvCacheRetentionConfig.getDecodeDurationMs();
    }

    [[nodiscard]] executor::KvCacheTransferMode getTransferMode() const
    {
        return mKvCacheRetentionConfig.getTransferMode();
    }

    [[nodiscard]] std::string const& getDirectory() const
    {
        return mKvCacheRetentionConfig.getDirectory();
    }

private:
    // Request id of the sequence
    LlmRequest::RequestIdType mRequestId;
    // Current number of generated tokens
    SizeType32 mNumTokens;
    // Number of beams
    SizeType32 mBeamWidth;
    // List of block ids allocated per each window size, for each beam of the sequence
    std::unordered_map<SizeType32, std::vector<std::vector<KVCacheBlock::IdType>>> mCacheBlockIds;
    // Tensor of block indices allocated per each window size, for each beam of the sequence
    std::unordered_map<SizeType32, runtime::ITensor::SharedPtr> mCacheBlockIndices;
    // The retention priority to assign to decode blocks
    executor::KvCacheRetentionConfig mKvCacheRetentionConfig;
    // Number of front blocks removed from the sequence
    SizeType32 mNumFrontBlocksRemoved;
    // Set of used blocks by the sequence
    std::set<KVCacheBlock::IdType> mUsedBlocks;
};

// attach metadata to a pool pointer
class KVCacheBlockPool
{
public:
    SizeType32 numLayers;
    SizeType32 kvFactor;
    SizeType32 numKvHeads;
    SizeType32 sizePerHead;
    SizeType32 tokensPerBlock;
    SizeType32 blockSize;

    // Memory pools. Primary is fast memory, secondary is slower memory used for offloading.
    runtime::ITensor::SharedPtr primaryPtr;
    runtime::ITensor::SharedPtr secondaryPtr;

    // FP4 KV caches have extra pools that contain second level scales for dequantization.
    bool containsBlockScales;

    KVCacheBlockPool(SizeType32 numLayers, SizeType32 kvFactor, SizeType32 numKvHeads, SizeType32 sizePerHead,
        SizeType32 tokensPerBlock, runtime::ITensor::SharedPtr primaryPtr = nullptr,
        runtime::ITensor::SharedPtr secondaryPtr = nullptr, bool containsBlockScales = false)
        : numLayers(numLayers)
        , kvFactor(kvFactor)
        , numKvHeads(numKvHeads)
        , sizePerHead(sizePerHead)
        , tokensPerBlock(tokensPerBlock)
        , blockSize(numKvHeads * sizePerHead * tokensPerBlock)
        , primaryPtr(std::move(primaryPtr))
        , secondaryPtr(std::move(secondaryPtr))
        , containsBlockScales(containsBlockScales)
    {
    }
};

// The WindowBlockManager manages the metadata of KVCacheBlocks.
// It manages multiple arrays of cache blocks called pools.
// Layers with the same number of kv heads are grouped under the same pool.
// Each pool has shape [max_blocks, num_layers, 2, num_kv_heads, tokens_pre_block, head_size], where num_layers refers
// to the number of layers with the same num_kv_heads that share that pool.
// The metadata of KVCacheBlocks is shared between layers, so each block spans all of the managed pool - an allocated
// block matches some chunk of memory in each pool. The shape of the chunk in every pool is [2, num_kv_heads,
// tokens_per_block, head_size]. The size per block and number of blocks are pre-determined and set in the constructor.
// WindowBlockManager maintains a list of free blocks at any time.
//
// FP4 KV caches allocate additional pools for block scale factors. These pools have the same
// shape as the regular KV pools, except that the the last dim is head_size / N where N is determined
// by the precise FP4 format being used (16 for NVFP4). There is one block scale pool per normal pool.
//
// BlockManager maintains a list of free blocks at any time.
// Alloc pops off the block at the front, and Free pushes it back to the vector.
// WindowBlockManager maintains a vector of lists of request ids to allocated blocks
// per sequence. This can be used to Free all blocks belonging to a sequence.
class WindowBlockManager
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using CacheType = tensorrt_llm::batch_manager::kv_cache_manager::CacheType;
    using BaseEvictionPolicy = tensorrt_llm::batch_manager::eviction_policy::BaseEvictionPolicy;
    using BlockMap = std::unordered_multimap<size_t, BlockPtr>;
    using BlockMapIterRange = std::pair<BlockMap::const_iterator, BlockMap::const_iterator>;

    explicit WindowBlockManager(nvinfer1::DataType dtype, SizeType32 windowSize,
        std::vector<SizeType32> const& managedLayers, std::vector<SizeType32> const& numKvHeadsPerLayer,
        SizeType32 sizePerHead, SizeType32 tokensPerBlock, bool isSWA, SizeType32 blocksInPrimaryPool,
        SizeType32 blocksInSecondaryPool, SizeType32 maxNumSequences, std::shared_ptr<runtime::CudaStream> stream,
        bool onboardBlocks, CacheType cacheType, std::optional<executor::RetentionPriority> secondaryOffloadMinPriority,
        std::shared_ptr<KVCacheEventManager> eventManager, bool enablePartialReuse, bool copyOnPartialReuse,
        std::shared_ptr<kv_connector::KvCacheConnectorManager> kvCacheConnectorManager,
        std::shared_ptr<kvc::BaseLoopbackAgent> loopbackAgent = nullptr);

    ~WindowBlockManager();

    void allocatePools(bool useUvm);

    void releasePools();

    void startScheduling();

    //! \brief Assign blocks for new sequence. Try to reuse blocks.
    void addSequence(
        GenerationRequest& sequence, SizeType32 inputLength, SizeType32 numContextBlocks, LlmRequest& llmRequest);

    //! \brief Assign blocks for new sequence. Does not try to reuse blocks.
    void addSequence(GenerationRequest& sequence, SizeType32 numContextBlocks, bool isShareLastContextBlock);

    //! \brief Allocate new block for each beam of the sequence.
    //! \details Might free cached blocks if no free blocks are available.
    void allocateBlock(GenerationRequest& sequence, bool shareAmongBeams);

    void replaceSharedBlock(GenerationRequest& sequence, SizeType32 blockIdx);

    [[nodiscard]] std::optional<KVCacheBlock::IdType> storeBlocksForReuse(
        GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest, bool pinBlocks = false);

    void storeNewBlock(GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest);

    //! \brief Pin blocks associated with a sequence to prevent eviction.
    void pinBlocks(GenerationRequest& sequence);

    //! \brief Release blocks of the sequence.
    //! \details When llmRequest is provided and reuse is enabled, blocks will be stored.
    std::optional<KVCacheBlock::IdType> releaseBlocks(
        GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest);

    //! \brief Simulate freeing all blocks for that sequence to check impact on number of free blocks
    void schedulingReleaseBlocks(LlmRequest::RequestIdType requestId);

    //! \brief Update cache offsets for last block
    void updateLastCacheBlockOffsets(GenerationRequest& seq);

    //! \brief Release last block in the sequence
    void releaseLastBlock(GenerationRequest& sequence);

    //! \brief Detach front block from the sequence
    void detachFrontBlock(GenerationRequest& sequence);

    //! \brief Add/detach block(s) to/from the sequence if needed
    //! \details When we need a new block, we add it. For sliding window
    //! attention (SWA), when a block goes out-of-window (OOW), we detach it
    //! If this called in the first step of the generation phase, we may detach
    //! more than a single block since there may be more than one context block
    //! that goes OOW.
    void adjustBlocksIfNeeded(GenerationRequest& sequence);

    [[nodiscard]] SizeType32 getWindowSize() const noexcept
    {
        return mWindowSize;
    }

    [[nodiscard]] std::string const& getLogPrefix() const noexcept
    {
        return mLogPrefix;
    }

    [[nodiscard]] SizeType32 getNumFreeBlocks() const noexcept;

    [[nodiscard]] SizeType32 getNumAllocTotalBlocks() const
    {
        return mAllocTotalBlocks;
    }

    [[nodiscard]] SizeType32 getNumAllocNewBlocks() const
    {
        return mAllocNewBlocks;
    }

    [[nodiscard]] SizeType32 getNumReusedBlocks() const noexcept
    {
        return mReusedBlocks;
    }

    [[nodiscard]] SizeType32 getNumAllocatedBlocks() const noexcept
    {
        return getMaxNumBlocks() - getNumFreeBlocks();
    }

    [[nodiscard]] SizeType32 getNumMissedBlocks() const noexcept
    {
        return mMissedBlocks;
    }

    [[nodiscard]] bool hasFreeBlocks(SizeType32 numRequired = 1) const noexcept
    {
        return getNumFreeBlocks() >= numRequired;
    }

    [[nodiscard]] bool schedulingHasFreeBlocks(SizeType32 numRequired) const noexcept
    {
        return mSchedulingNumFreeBlocks >= numRequired;
    }

    [[nodiscard]] SizeType32 getMaxNumBlocks() const noexcept
    {
        return static_cast<SizeType32>(mAllBlocksById.size());
    }

    [[nodiscard]] BlockPtr const& getBlockById(KVCacheBlock::IdType blockId) const
    {
        return mAllBlocksById.at(blockId);
    }

    [[nodiscard]] SizeType32 getTokensPerBlock() const noexcept
    {
        return mTokensPerBlock;
    }

    //! \brief Get size of one K/V cache block in one layer for the specified pool.
    //! @details Volume of [numKvHeads, tokensPerBlock, sizePerHead] in the specified pool.
    [[nodiscard]] SizeType32 getBlockSize(SizeType32 poolIdx) const
    {
        return mPools.at(poolIdx).blockSize;
    }

    [[nodiscard]] SizeType32 getNumEltsPerContainer() const
    {
#ifdef ENABLE_FP4
        return mDataType == nvinfer1::DataType::kFP4 ? 2 : 1;
#else
        return 1;
#endif
    }

    [[nodiscard]] SizeType32 getNumPools(bool includeBlockScalePools = true) const noexcept
    {
        if (includeBlockScalePools)
        {
            return mPools.size();
        }
        return std::count_if(mPools.begin(), mPools.end(), [](auto const& pool) { return !pool.containsBlockScales; });
    }

    [[nodiscard]] KVCacheBlockPool const& getPool(SizeType32 poolIdx) const
    {
        return mPools.at(poolIdx);
    }

    [[nodiscard]] bool containsBlockScales(SizeType32 poolIdx) const
    {
        return mPools.at(poolIdx).containsBlockScales;
    }

    [[nodiscard]] SizeType32 getNumPrimaryBlocks() const
    {
        return mNumPrimaryBlocks;
    }

    [[nodiscard]] SizeType32 getNumSecondaryBlocks() const
    {
        return mNumSecondaryBlocks;
    }

    [[nodiscard]] SizeType32 getLayerPoolIdx(SizeType32 layerIdx) const
    {
        return mLayerToPoolIndex.at(layerIdx);
    }

    //! \brief Maps a global layer index to its layer index within its pool.
    //! \details If we only have one pool, then getPoolLayerIdx(i) == i. Otherwise,
    //! \details gives the layer index into the getLayerPoolIdx(i).
    [[nodiscard]] SizeType32 getPoolLayerIdx(SizeType32 layerIdx) const
    {
        return mLayerToIndexWithinPool.at(layerIdx);
    }

    void setOffsets(kernels::KVCacheIndex* offsetsPtr, nvinfer1::Dims const& offsetsShape, SizeType32 beamIdx,
        SizeType32 blockIdx, KVCacheBlock::IdType blockId) const;

    //! \brief Bring offloaded block from secondary to primary memory.
    //! \details Does nothing if block is already in primary memory.
    void onboardBlock(GenerationRequest& sequence, BlockPtr const& offloadBlock,
        executor::KvCacheTransferMode mode = executor::KvCacheTransferMode::DRAM, std::string const& directory = "");

    //! \brief Bring block from primary to secondary memory.
    //! \details Does nothing if block is already in secondary memory.
    void offloadBlock(BlockPtr const& block, executor::KvCacheTransferMode mode = executor::KvCacheTransferMode::DRAM,
        std::string const& directory = "");

    //! \brief Find first new block that must be allocated for context phase and return it's concatenated token vectors.
    //! \details Only full blocks are considered.
    [[nodiscard]] std::optional<BlockKey> findNewContextBlock(
        VecUniqueTokens const& uniqueTokens, LlmRequest const& llmRequest) const;

    [[nodiscard]] runtime::BufferManager const& getBufferManager() const
    {
        return mBufferManager;
    }

    //! \brief Perform per-request bookkeeping
    void refreshBlocks();

    [[nodiscard]] static bool blockInRadixTree(BlockPtr const& block);

    //! \brief Store blocks in cached blocks.
    //! \param blockKeys Key of each block.
    //! \param blockIds Id of each block.
    //! \param pinBlocks If true, increment ref count for blocks while storing (pin on store).
    //! \return Pair of (num blocks stored for reuse, id of the last block stored if any).
    [[nodiscard]] std::pair<SizeType32, std::optional<KVCacheBlock::IdType>> storeBlocks(
        std::vector<BlockKey> const& blockKeys, std::vector<KVCacheBlock::IdType> const& blockIds,
        bool pinBlocks = false);

    [[nodiscard]] bool verifyQueueIntegrity();

    // Only needed when sliding window attention + paged context fmha are used together.
    // In that case, a temporary kv cache buffer with maximum chunk size (maxNumTokens) is needed.
    // TODO: There are several things that can be improved later.
    //  1. a dynamic temporary kv cache allocation based on real chunk size might be needed.
    //  2. reuse the same temporary kv cache buffer among all layers in the same pool.
    [[nodiscard]] SizeType32 calculateTemporaryAttentionWindow(
        std::optional<TempAttentionWindowInputs> const& inputs) const
    {

        if (inputs && inputs->pagedContextFMHA && (inputs->maxInputLen > mWindowSize))
        {
            auto window = std::min(inputs->maxNumTokens, inputs->maxInputLen - mWindowSize);
            window = std::max(window, 0); // clamp negative values to 0
            return window;
        }
        return 0;
    }

    //! \brief Return whether this window is SWA.
    [[nodiscard]] bool isSWA() const
    {
        return mIsSWA;
    }

    [[nodiscard]] std::shared_ptr<KVCacheBlock> findBlocksInReuseTreeByBlockKey(BlockKey const& blockKey);

    //! \brief Unpin blocks by starting from a block id and walking prev pointers.
    void unpinBlocksById(KVCacheBlock::IdType blockId);

    void initializeSequenceStorageValidity(LlmRequest::RequestIdType requestId)
    {
        mIsValidStoreForReuseSequence[requestId] = true;
    }

    void releaseSequenceStorageValidity(LlmRequest::RequestIdType requestId)
    {
        mIsValidStoreForReuseSequence.erase(requestId);
    }

    //! \brief Return whether this sequence is valid for store for reuse
    [[nodiscard]] bool isSequenceValidForStoreForReuse(LlmRequest::RequestIdType requestId) const
    {
        TLLM_CHECK_WITH_INFO(mIsValidStoreForReuseSequence.count(requestId) > 0, "Sequence should be bookkeeped");
        return mIsValidStoreForReuseSequence.at(requestId);
    }

private:
    //! \brief Add single block to beam of sequence and mAllocatedBlocksPerSeq.
    void addBlockToBeam(BlockPtr& block, GenerationRequest& sequence, SizeType32 beamIdx);

    //! \brief Add single block to all beams of sequence.
    void addBlockToAllBeams(BlockPtr& block, GenerationRequest& sequence);

    //! \brief Try to load blocks from cache. Allocate new blocks if necessary.
    //! \param blockKeys Key of each block.
    //! \param sequence Sequence to which blocks are assigned.
    //! \return Number of matched tokens from loaded blocks.
    SizeType32 loadOrAllocateBlocks(std::vector<BlockKey> const& blockKeys, SizeType32 numContextBlocks,
        GenerationRequest& sequence, std::vector<executor::RetentionPriorityAndDuration> const& perBlockRetentions,
        executor::KvCacheTransferMode mode = executor::KvCacheTransferMode::DRAM, std::string const& directory = "");

    //! \brief Free block and all it's descendants. This makes block a claimed leaf block.
    void freeChildren(BlockPtr const& block);

    //! \brief Find block least likely to be reused, free it if necessary and return.
    //! \param sequence Sequence which the free block is allocated for
    [[nodiscard]] BlockPtr getFreeBlock(GenerationRequest& sequence,
        executor::RetentionPriority = executor::KvCacheRetentionConfig::kDefaultRetentionPriority,
        std::optional<std::chrono::milliseconds> durationMs = std::nullopt,
        executor::KvCacheTransferMode mode = executor::KvCacheTransferMode::DRAM, std::string const& directory = "");

    //! \brief Calls KVCacheBlock::freeLeafBlock to remove block from search tree.
    void freeLeafBlock(BlockPtr const& block);

    //! \brief For FP4 quantization. Creates pool objects for FP4 block scalars.
    void createBlockScalePools(SizeType32 blockSize);

private:
    nvinfer1::DataType mDataType;
    SizeType32 mWindowSize;

    // Number of blocks in pools
    SizeType32 mNumPrimaryBlocks;
    SizeType32 mNumSecondaryBlocks;

    // List of allocated blocks for each sequences
    std::unordered_map<LlmRequest::RequestIdType, std::vector<BlockPtr>> mAllocatedBlocksPerSeq;

    // Pool per unique numKvHeads in the model
    std::vector<KVCacheBlockPool> mPools;

    // Matching layers to their respective pools: {<layer #0>: <pool idx 2>, }, etc.
    std::unordered_map<SizeType32, SizeType32> mLayerToPoolIndex;
    // Matching layers to their index *within* their respective pools: {..., <layer 3>: <idx 2 within pool> }. See
    // getPoolLayerIdx
    std::unordered_map<SizeType32, SizeType32> mLayerToIndexWithinPool;

    // Whether offloaded blocks should be onboarded before reuse.
    bool mOnboardBlocks;
    // Buffer manager
    runtime::BufferManager mBufferManager;

    // Used to keep track of number of free blocks during scheduling
    SizeType32 mSchedulingNumFreeBlocks;
    // Number of tokens per one block
    SizeType32 mTokensPerBlock;
    // Whether this window is sliding window attention/full attention
    bool mIsSWA;
    // List of all blocks by idx
    std::vector<BlockPtr> mAllBlocksById;
    // Dummy block acting as root for BlockToken searches
    BlockPtr mCachedBlocksRoot;
    // KV cache type (self or cross)
    CacheType mCacheType;
    // Eviction Policy
    std::shared_ptr<BaseEvictionPolicy> mEvictionPolicy;
    // Event manager
    std::shared_ptr<KVCacheEventManager> mEventManager;
    // Pointer to parent loopback agent
    std::shared_ptr<kvc::BaseLoopbackAgent> mLoopbackAgent;
    // Transfer manager
    std::shared_ptr<KVCacheTransferManager> mTransferManager;

    // Statistics for block allocations/reuse
    // Total number of blocks allocated by all requests
    SizeType32 mAllocTotalBlocks;
    // Number of new blocks that were allocated
    SizeType32 mAllocNewBlocks;
    // Number of blocks that were reused
    SizeType32 mReusedBlocks;
    // Number of unique blocks that were reused
    SizeType32 mReusedUniqueBlocks;
    // Number of blocks that were not reused
    SizeType32 mMissedBlocks;
    // Only be 1 or 2. If 2: general KV stored. If 1: K == V for any token, so only K is stored to optimize the
    // max_num_tokens(For DeepSeek). Controlled by mCacheType
    SizeType32 mKVFactor;
    std::set<KVCacheBlock::IdType> reusedBlockIds;
    std::string const mLogPrefix;
    // Number of reused tokens
    double mReusedTokens;
    // Total number of input tokens
    double mTotalInputTokens;
    // Whether blocks that are partially matched should be reused.
    bool mEnablePartialReuse;
    // Whether partially matched blocks that are already in use should be copied and reused.
    bool mCopyOnPartialReuse;
    // The kv cache connector manager
    std::shared_ptr<kv_connector::KvCacheConnectorManager> mKvCacheConnectorManager;

    // Mutex for the cached blocks root
    std::mutex mCachedBlocksRootMutex;

    // Record which sequence is using the block
    std::map<KVCacheBlock::IdType, LlmRequest::RequestIdType> mBlockToSequence;
    // Record whether a sequence has all blocks held valid.
    // The boolean value is set to true upon first encounter of a new sequence.
    // It may be invalidated to false when other sequence acquires a block that
    // is used by another sequence.
    std::map<LlmRequest::RequestIdType, bool> mIsValidStoreForReuseSequence;
};

class BlockManager
{
public:
    using CudaStreamPtr = std::shared_ptr<runtime::CudaStream>;
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using BaseEvictionPolicy = tensorrt_llm::batch_manager::eviction_policy::BaseEvictionPolicy;

    explicit BlockManager(std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead,
        SizeType32 tokensPerBlock, BlocksPerWindow const& blocksPerWindow, SizeType32 maxNumSequences,
        CudaStreamPtr stream, SizeType32 maxSequenceLength, SizeType32 maxBeamWidth,
        std::vector<SizeType32> const& maxAttentionWindowVec,
        std::optional<TempAttentionWindowInputs> const& tempAttentionWindowInputs, nvinfer1::DataType dtype,
        SizeType32 sinkBubbleLength, bool onboardBlocks, CacheType cacheType = CacheType::kSELF,
        std::optional<executor::RetentionPriority> secondaryOffloadMinPriority = std::nullopt,
        std::shared_ptr<KVCacheEventManager> eventManager = nullptr, bool enablePartialReuse = true,
        bool copyOnPartialReuse = true,
        std::shared_ptr<kv_connector::KvCacheConnectorManager> kvCacheConnectorManager = nullptr,
        std::optional<kvc::BaseAgentConfig> agentConfig = std::nullopt);

    BlockManager(BlockManager const&) = delete;
    BlockManager& operator=(BlockManager const&) = delete;

    //! \brief Calculate the proportional share each window size receives of the total memory pool
    //! \details Example:       (uniqueWindowSizeToLayers={1024: [1], 4096: [0, 4, 5], 8192: [2, 3]})
    //!          Would Return:  {1024: 0.0345, 4096: 0.4138, 8192: 0.5517} [sums to 1.0].
    //!          See: TEST_F(KVCacheManagerTest, BlockManagerTestWindowSizeToShare).
    //! \return Map<windowSize, share> where share is a float between 0 and 1. Shares sum to 1.0.
    static std::map<SizeType32, float> calculateWindowSizeToShare(
        std::map<SizeType32, std::vector<SizeType32>> const& uniqueWindowSizeToLayers,
        std::map<SizeType32, SizeType32> const& cacheSizePerTokenPerWindowSize);

    void allocatePools(bool useUvm);

    void addSequence(GenerationRequest& sequence, SizeType32 inputLength, SizeType32 numContextBlocks,
        LlmRequest& llmRequest, SizeType32 windowSize);

    //! \brief Assign blocks for a new sequence.
    //! \param sequence  The GenerationRequest to process.
    //! \param numContextBlocks  Number of context blocks to allocate.
    //! \param windowSize  Attention window size
    //! \param isShareLastContextBlock  If true, the last context block is shared among beams.
    void addSequence(
        GenerationRequest& sequence, SizeType32 numContextBlocks, SizeType32 windowSize, bool isShareLastContextBlock);

    void allocateBlock(GenerationRequest& sequence, SizeType32 windowSize);

    void replaceSharedBlock(GenerationRequest& sequence, SizeType32 windowSize, SizeType32 blockIdx);

    std::optional<KVCacheBlock::IdType> releaseBlocks(
        GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest = std::nullopt, bool pinBlocks = false);

    [[nodiscard]] std::optional<KVCacheBlock::IdType> storeBlocksForReuse(
        GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest = std::nullopt, bool pinBlocks = false);

    void schedulingReleaseBlocks(LlmRequest::RequestIdType requestId);

    /// @brief Pin all blocks associated with a sequence across all window managers.
    /// @param sequence The generation request whose blocks should be pinned.
    void pinBlocks(GenerationRequest& sequence);

    void unpinBlocksById(KVCacheBlock::IdType blockId);

    void releaseLastBlock(GenerationRequest& sequence, SizeType32 windowSize);

    void setOffsets(kernels::KVCacheIndex* offsetsPtr, nvinfer1::Dims const& offsetsShape, SizeType32 beamIdx,
        SizeType32 blockIdx, KVCacheBlock::IdType blockId, SizeType32 windowSize) const;

    // WILL NOT WORK FOR VARIABLE WINDOW ATTENTION
    [[nodiscard]] std::optional<BlockKey> findNewContextBlock(
        VecUniqueTokens const& uniqueTokens, LlmRequest const& llmRequest) const;

    //! \brief Bring block from primary to secondary memory for window size.
    //! \details Does nothing if block is already in primary memory.
    void onboardBlock(GenerationRequest& sequence, BlockPtr const& offloadBlock, SizeType32 windowSize,
        executor::KvCacheTransferMode mode = executor::KvCacheTransferMode::DRAM, std::string const& directory = "");

    //! \brief Bring block from primary to secondary memory for window size.
    //! \details Does nothing if block is already in secondary memory.
    void offloadBlock(BlockPtr const& block, SizeType32 windowSize,
        executor::KvCacheTransferMode mode = executor::KvCacheTransferMode::DRAM, std::string const& directory = "");

    [[nodiscard]] std::pair<SizeType32, std::optional<KVCacheBlock::IdType>> storeBlocks(
        std::vector<BlockKey> const& blockKeys, std::vector<KVCacheBlock::IdType> const& blockIds,
        SizeType32 windowSize, bool pinBlocks = false)
    {
        return mWindowBlockManagers.at(windowSize).storeBlocks(blockKeys, blockIds, pinBlocks);
    }

    [[nodiscard]] bool verifyQueueIntegrity(SizeType32 windowSize);

    void releasePools();

    void startScheduling();

    [[nodiscard]] std::map<SizeType32, SizeType32> getNumFreeBlocksPerWindowSize() const
    {
        std::map<SizeType32, SizeType32> numFreeBlocksPerWindowSize;
        for (auto const& [windowSize, manager] : mWindowBlockManagers)
        {
            numFreeBlocksPerWindowSize[windowSize] = manager.getNumFreeBlocks();
        }
        return numFreeBlocksPerWindowSize;
    }

    [[nodiscard]] SizeType32 getNumFreeBlocks() const
    {
        return sumWindows([](auto const& manager) { return manager.getNumFreeBlocks(); });
    }

    [[nodiscard]] bool schedulingHasFreeBlocks(SizeType32 numRequired, SizeType32 windowSize) const
    {
        return mWindowBlockManagers.at(windowSize).schedulingHasFreeBlocks(numRequired);
    }

    [[nodiscard]] SizeType32 getNumAllocTotalBlocks() const
    {
        return sumWindows([](auto const& manager) { return manager.getNumAllocTotalBlocks(); });
    }

    [[nodiscard]] SizeType32 getFirstWindowSize() const
    {
        if (mWindowBlockManagers.empty())
        {
            return 0;
        }
        return mWindowBlockManagers.begin()->first;
    }

    [[nodiscard]] SizeType32 getNumAllocNewBlocks() const
    {
        return sumWindows([](auto const& manager) { return manager.getNumAllocNewBlocks(); });
    }

    [[nodiscard]] SizeType32 getNumReusedBlocks() const
    {
        return sumWindows([](auto const& manager) { return manager.getNumReusedBlocks(); });
    }

    [[nodiscard]] SizeType32 getNumMissedBlocks() const
    {
        return sumWindows([](auto const& manager) { return manager.getNumMissedBlocks(); });
    }

    [[nodiscard]] SizeType32 getNumLayers() const
    {
        return mNumLayers;
    }

    [[nodiscard]] CacheType getCacheType() const
    {
        return mCacheType;
    }

    [[nodiscard]] SizeType32 getLayerPoolIdx(SizeType32 layerIdx) const
    {
        auto const& manager = windowManagerByLayer(layerIdx);
        auto const absoluteOffset = absolutePoolsOffset(manager);
        auto const relativePoolIndex = manager.getLayerPoolIdx(layerIdx);
        return absoluteOffset + relativePoolIndex;
    }

    [[nodiscard]] SizeType32 getPoolLayerIdx(SizeType32 layerIdx) const
    {
        return windowManagerByLayer(layerIdx).getPoolLayerIdx(layerIdx);
    }

    [[nodiscard]] SizeType32 getTokensPerBlock() const noexcept
    {
        return mTokensPerBlock;
    }

    [[nodiscard]] SizeType32 getStreamDevice() const
    {
        return mStream->getDevice();
    }

    [[nodiscard]] std::deque<executor::KVCacheEvent> getLatestEvents(
        std::optional<std::chrono::milliseconds> timeout) const;

    void flushIterationEvents()
    {
        if (mEventManager)
        {
            mEventManager->flush();
        }
    }

    [[nodiscard]] SizeType32 getPoolWindowSize(SizeType32 poolIdx) const
    {
        return mAbsolutePoolToWindowSize.at(poolIdx);
    }

    [[nodiscard]] SizeType32 getBlockSize(SizeType32 poolIdx) const
    {
        return getPool(poolIdx).blockSize;
    }

    [[nodiscard]] SizeType32 getNumPools(bool includeBlockScalePools = true) const
    {
        return sumWindows(
            [includeBlockScalePools](auto const& manager) { return manager.getNumPools(includeBlockScalePools); });
    }

    [[nodiscard]] std::map<SizeType32, WindowSizeMetadata> const& getWindowSizesMetadata() const noexcept
    {
        return mWindowSizeToMetadata;
    }

    [[nodiscard]] WindowSizeMetadata getWindowSizeMetadata(SizeType32 windowSize) const noexcept
    {
        return mWindowSizeToMetadata.at(windowSize);
    }

    [[nodiscard]] bool isVariableWindow() const noexcept
    {
        return mIsVariableWindow;
    }

    [[nodiscard]] SizeType32 getMaxBlockPerSeqWhenSingleWindowSize() const
    {
        TLLM_CHECK_WITH_INFO(!isVariableWindow(),
            "This function was called assuming there is only a single window size, and therefore a single "
            "maxBlocksPerSeq");
        auto const windowSize = windowManagerByLayer(0).getWindowSize();
        auto const onlyWindowSizeMetadata = getWindowSizeMetadata(windowSize);
        return onlyWindowSizeMetadata.maxBlocksPerSeq;
    }

    [[nodiscard]] bool isVariableGQA() const noexcept
    {
        return mIsVariableGQA;
    }

    [[nodiscard]] runtime::ITensor::SharedPtr getPrimaryPool(SizeType32 poolIdx) const
    {
        return getPool(poolIdx).primaryPtr;
    }

    [[nodiscard]] runtime::ITensor::SharedPtr getSecondaryPool(SizeType32 poolIdx) const
    {
        return getPool(poolIdx).secondaryPtr;
    }

    [[nodiscard]] SizeType32 getNumAllocatedBlocks() const
    {
        return sumWindows([](auto const& manager) { return manager.getNumAllocatedBlocks(); });
    }

    [[nodiscard]] SizeType32 getMaxNumBlocks() const
    {
        return sumWindows([](auto const& manager) { return manager.getMaxNumBlocks(); });
    }

    [[nodiscard]] BlockPtr const& getBlockById(KVCacheBlock::IdType blockId, SizeType32 windowSize) const
    {
        return mWindowBlockManagers.at(windowSize).getBlockById(blockId);
    }

    [[nodiscard]] std::shared_ptr<KVCacheBlock> findBlocksInReuseTreeByBlockKey(
        BlockKey const& blockKey, SizeType32 windowSize)
    {
        return mWindowBlockManagers.at(windowSize).findBlocksInReuseTreeByBlockKey(blockKey);
    }

    [[nodiscard]] SizeType32 getNumPrimaryBlocks() const
    {
        return sumWindows([](auto const& manager) { return manager.getNumPrimaryBlocks(); });
    }

    [[nodiscard]] bool containsBlockScales(SizeType32 poolIdx) const
    {
        return getPool(poolIdx).containsBlockScales;
    }

    //! \brief Store context blocks
    void storeContextBlocks(GenerationRequest& sequence, LlmRequest const& llmRequest);

    //! \brief Store newest block for reuse
    void storeNewBlock(GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest);

    //! \brief Perform per-request bookkeeping
    void refreshBlocks();

    [[nodiscard]] runtime::BufferManager const& getBufferManager(SizeType32 windowSize) const
    {
        return mWindowBlockManagers.at(windowSize).getBufferManager();
    }

    [[nodiscard]] KVCacheBlockPool const& getPool(SizeType32 poolIdx) const
    {
        auto const windowSize = getPoolWindowSize(poolIdx);
        auto const relativePoolIndex = mAbsolutePoolToRelativePoolIndex.at(poolIdx);
        return mWindowBlockManagers.at(windowSize).getPool(relativePoolIndex);
    }

    //! \brief Update cache offsets for blocks initiated from sequence
    void updateSequenceCacheBlockOffsets(GenerationRequest& seq, SizeType32 windowSize);

    //! \brief Update cache offsets for block at index
    void updateCacheBlockOffsetsAtIdx(GenerationRequest& seq, SizeType32 windowSize, SizeType32 blockIdx);

    //! \brief Add/detach block(s) to/from the sequence if needed
    //! \details When we need a new block, we add it. For sliding window
    //! attention (SWA), when a block goes out-of-window (OOW), we detach it
    //! If this called in the first step of the generation phase, we may
    //! detach more than a single block since there may be more than one
    //! context block that goes OOW.
    void adjustBlocksIfNeeded(GenerationRequest& sequence);

    //! \brief Return whether the sequence is already managed by the block manager
    [[nodiscard]] bool isSequenceHeld(LlmRequest::RequestIdType requestId) const
    {
        return mManagedSequences.count(requestId) > 0;
    }

    //! \brief Add a sequence to the managed sequences
    //! \details Take the sequence into account for the manager. Initialize
    //! sequence storage validity under all window sizes.
    void holdSequence(LlmRequest::RequestIdType requestId)
    {
        mManagedSequences.insert(requestId);
        for (auto const& [windowSize, metadata] : mWindowSizeToMetadata)
        {
            mWindowBlockManagers.at(windowSize).initializeSequenceStorageValidity(requestId);
        }
    }

    //! \brief Remove a sequence from the managed sequences.
    //! \details Remove sequence from the managed sequences and remove sequence
    //! storage
    void releaseSequence(LlmRequest::RequestIdType requestId)
    {
        mManagedSequences.erase(requestId);
        for (auto const& [windowSize, metadata] : mWindowSizeToMetadata)
        {
            mWindowBlockManagers.at(windowSize).releaseSequenceStorageValidity(requestId);
        }
    }

    //! \brief Return whether the sequence is still valid for store-for-reuse
    //! regarding the specific window size.
    //! \details Currently this utility function is only used under
    //! kvCacheManagerTest.cpp. Checking for store-for-reuse for each window
    //! size is done in an iterating fashion under BlockManager::releaseBlocks.
    bool isSequenceValidForStoreForReuse(LlmRequest::RequestIdType requestId, SizeType32 windowSize) const
    {
        TLLM_CHECK_WITH_INFO(
            mWindowBlockManagers.count(windowSize) > 0, "Querying window size is not found under mWindowBlockManager");
        return mWindowBlockManagers.at(windowSize).isSequenceValidForStoreForReuse(requestId);
    }

private:
    [[nodiscard]] WindowBlockManager const& windowManagerByLayer(SizeType32 layerIdx) const
    {
        return mWindowBlockManagers.at(mLayerToWindowSize.at(layerIdx));
    }

    [[nodiscard]] SizeType32 sumWindows(std::function<SizeType32(WindowBlockManager const&)> produce) const
    {
        return std::accumulate(mWindowBlockManagers.cbegin(), mWindowBlockManagers.cend(), SizeType32{0},
            [&produce](SizeType32 acc, auto const& manager) { return acc + produce(manager.second); });
    }

    [[nodiscard]] SizeType32 absolutePoolsOffset(WindowBlockManager const& manager) const
    {
        auto const windowSize = manager.getWindowSize();
        return getWindowSizeMetadata(windowSize).absolutePoolsOffset;
    }

private:
    SizeType32 mNumLayers;
    SizeType32 mTokensPerBlock;
    std::shared_ptr<KVCacheEventManager> mEventManager;
    std::shared_ptr<kvc::BaseLoopbackAgent> mLoopbackAgent;
    CudaStreamPtr mStream;
    CacheType mCacheType;

    bool mIsVariableWindow;
    bool mIsVariableGQA;

    std::map<SizeType32, WindowBlockManager> mWindowBlockManagers;
    std::map<SizeType32, WindowSizeMetadata> mWindowSizeToMetadata;
    std::vector<SizeType32> mLayerToWindowSize;
    std::vector<SizeType32> mAbsolutePoolToWindowSize;
    std::vector<SizeType32> mAbsolutePoolToRelativePoolIndex;
    // Record what sequences are currently managed by the block manager
    std::set<LlmRequest::RequestIdType> mManagedSequences;
};

struct OffsetTableDimensions
{
    SizeType32 maxBlocksPerSeq;
    SizeType32 numPools;
    CacheType cacheType;
};

class BaseKVCacheManager
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using CudaStreamPtr = std::shared_ptr<runtime::CudaStream>;
    using CacheType = tensorrt_llm::batch_manager::kv_cache_manager::CacheType;

    virtual ~BaseKVCacheManager() {}

    virtual void allocatePools(bool useUvm = false) = 0;

    virtual void releasePools() = 0;

    virtual void startScheduling() = 0;

    [[nodiscard]] virtual SizeType32 getTokensPerBlock() const = 0;

    [[nodiscard]] virtual SizeType32 getMaxNumBlocks() const = 0;

    [[nodiscard]] virtual SizeType32 getUsedNumBlocks() const = 0;

    [[nodiscard]] virtual SizeType32 getNumFreeBlocks() const = 0;

    [[nodiscard]] virtual SizeType32 getNumPools() const = 0;

    // only used by test
    [[nodiscard]] virtual SizeType32 getNumReusedBlocks() const noexcept = 0;

    [[nodiscard]] virtual KvCacheStats getKvCacheStats() const = 0;

    [[nodiscard]] virtual OffsetTableDimensions getOffsetTableDimensions() const = 0;

    [[nodiscard]] virtual std::deque<executor::KVCacheEvent> getLatestEvents(
        std::optional<std::chrono::milliseconds> timeout = std::nullopt) const
        = 0;

    [[nodiscard]] virtual BlockManager const& getBlockManager() const = 0;

    /// @brief  Function that computes the number of KV cache blocks needed to advance a request by one or two
    /// iterations
    /// @param req The request for which we need to calculate the number of needed KV cache blocks
    /// @return  The number of blocks
    [[nodiscard]] virtual SizeType32 getNeededBlocksOneStep(
        LlmRequest const& req, bool twoStepsLookAhead, SizeType32 windowSize) const
        = 0;

    /// @brief  Function that computes the number of KV cache blocks needed to advance a request to completion (i.e. for
    /// maxNewTokens)
    /// @param req The request for which we need to calculate the number of needed KV cache blocks
    /// @return  The number of blocks
    [[nodiscard]] virtual SizeType32 getRemainingBlocksToCompletion(LlmRequest const& req, SizeType32 windowSize) const
        = 0;

    /// @brief Pin blocks associated with a request to prevent eviction.
    /// @param requestId The ID of the request whose blocks should be pinned.
    virtual void pinBlocks(LlmRequest::RequestIdType requestId) = 0;

    /// @brief Increase size for request at seqSlotIdx. Allocate new KV cache block(s) if needed.
    virtual void addToken(LlmRequest::RequestIdType requestId) = 0;

    /// @brief Add new request to the KV cache manager.
    /// @param inputLength Input length for which KV cache need to be allocated.
    /// @param beamWidth Beam width for which KV cache need to be allocated.
    /// @param llmRequest Optional request to use for KV cache lookup.
    /// @details If llmRequest is supplied and KV cache reuse is enabled, try to recover KV cache blocks for
    /// inputLength - 1 tokens and populate prepopulatedPromptLen.
    virtual void addSequence(LlmRequest::RequestIdType requestId, SizeType32 inputLength, SizeType32 beamWidth,
        OptionalRef<LlmRequest> llmRequest = std::nullopt)
        = 0;

    [[nodiscard]] virtual std::optional<KVCacheBlock::IdType> removeSequence(LlmRequest::RequestIdType requestId,
        OptionalRef<LlmRequest const> llmRequest = std::nullopt, bool pinOnRelease = false)
        = 0;

    virtual void schedulingRemoveSequence(LlmRequest::RequestIdType requestId) = 0;

    [[nodiscard]] virtual runtime::ITensor::SharedPtr getBlockPoolPointers() const = 0;

    [[nodiscard]] virtual runtime::ITensor::SharedPtr getBlockScalePoolPointers() const = 0;

    [[nodiscard]] virtual runtime::ITensor::SharedPtr getLayerToPoolMapping() const = 0;

    virtual void getBlockOffsetsOfBatch(
        runtime::ITensor& output, SizeType32 firstBatchSlotIdx, SizeType32 batchSize, SizeType32 beamWidth) const
        = 0;

    //! @return maxBlockCount of all beams
    virtual SizeType32 copyBlockOffsets(
        runtime::ITensor& output, SizeType32 outputSlotOffset, LlmRequest::RequestIdType requestId) const
        = 0;

    [[nodiscard]] virtual bool isEnableBlockReuse() const = 0;

    // void removeToken(SizeType32 seqSlotIdx);
    virtual void rewindKVCache(LlmRequest::RequestIdType requestId, SizeType32 rewindLengths) = 0;

    [[nodiscard]] virtual GenerationRequest const& getSequence(LlmRequest::RequestIdType requestId) const = 0;
    [[nodiscard]] virtual GenerationRequest& getSequence(LlmRequest::RequestIdType requestId) = 0;

    [[nodiscard]] virtual bool isCrossKv() const = 0;

    //! \brief Find first new block that must be allocated for context phase and return it's concatenated token vector.
    //! \details Only full blocks are considered.
    [[nodiscard]] virtual std::optional<BlockKey> findNewContextBlock(
        VecUniqueTokens const& uniqueTokens, LlmRequest const& llmRequest) const
        = 0;

    //! \brief Store full context blocks contributed by llmRequest.
    //! \details These blocks become reusable from next step.
    virtual void storeContextBlocks(LlmRequest const& llmRequest) = 0;

    //! \brief Store newest block for reuse.
    //! \details This block become reusable from next step.
    virtual void storeNewBlock(LlmRequest const& llmRequest) = 0;

    /// \brief Store blocks for reuse for a given request id
    [[nodiscard]] virtual std::optional<KVCacheBlock::IdType> storeBlocksForReuse(
        LlmRequest::RequestIdType requestId, OptionalRef<LlmRequest const> llmRequest, bool pinBlocks = false)
        = 0;

    //! \brief Get the block ids of a request [per beam] **for a given window size block manager**
    [[nodiscard]] virtual std::vector<std::vector<SizeType32>> const& getCacheBlockIds(
        LlmRequest::RequestIdType requestId, SizeType32 windowSize) const
        = 0;

    //! \brief Get the block ids of a batch of requests [per beam] **for a given window size block manager**
    [[nodiscard]] virtual std::vector<std::vector<std::vector<SizeType32>>> getBatchCacheBlockIds(
        std::vector<LlmRequest::RequestIdType> const& requestIds, SizeType32 windowSize) const
        = 0;

    /// @brief Get the last block id (beam 0) for a given sequence and window size
    [[nodiscard]] virtual std::optional<KVCacheBlock::IdType> getLastBlockId(LlmRequest::RequestIdType requestId) const
        = 0;

    [[nodiscard]] virtual runtime::ITensor::SharedPtr getUniquePrimaryPool() const = 0;
    [[nodiscard]] virtual runtime::ITensor::SharedPtr getPrimaryPool(SizeType32 layer_idx) const = 0;
    [[nodiscard]] virtual SizeType32 getPoolLayerIdx(SizeType32 layer_idx) const = 0;

    virtual void refreshBlocks() = 0;
    virtual void flushIterationEvents() = 0;

    [[nodiscard]] static SizeType32 getSinkBubbleLength(SizeType32 sinkTokenLen, SizeType32 tokensPerBlock);

    // Sum of numLayers * kvFactor * numKvHeads * sizePerHead for each pool
    [[nodiscard]] static SizeType32 calculateCacheSizePerTokenForSingleWindowSize(
        tensorrt_llm::runtime::ModelConfig const& modelConfig, std::vector<SizeType32> const& windowSizeLayers,
        bool isCrossAttention, SizeType32 kvFactor)
    {
        auto const nkvh = modelConfig.getNumKvHeadsForGivenLayers(windowSizeLayers, isCrossAttention);
        auto const sumLocalHeads = std::reduce(nkvh.cbegin(), nkvh.cend());
        // NOTE: We expect the initialization of modelConfig to have already taken the tp size into account and do not
        // address it here
        // consider only local layers for the calculation
        return sumLocalHeads * kvFactor * modelConfig.getSizePerHead();
    }

    /// @brief Groups model layers by their attention window size.
    /// @param maxAttentionWindowVec Vector of maximum attention window sizes per layer (may have fewer elements than
    /// numLayers, in which case it cycles)
    /// @param numLayers Total number of layers in the model
    /// @return Map from window size to vector of layer indices that use that window size
    [[nodiscard]] static std::map<SizeType32, std::vector<SizeType32>> groupLayersByWindowSize(
        std::vector<SizeType32> const& maxAttentionWindowVec, SizeType32 numLayers);

    /// @brief Calculate the free memory available for KV cache allocation.
    /// @param bufferManager Buffer manager for memory operations
    /// @param config KV cache configuration parameters
    /// @return Tuple containing the {.freePrimaryMemBytes, .freeSecondaryMemBytes}
    [[nodiscard]] static std::tuple<uint64_t, uint64_t> calculateFreeMemBytes(
        runtime::BufferManager const& bufferManager, executor::KvCacheConfig const& config);

    /// @brief Calculate the maximum number of KV cache blocks that can be allocated based on available GPU memory.
    /// @details This function computes how many blocks each WindowBlockManager should receive based on the weighted
    /// share
    ///          of memory requirements. The weighting considers both the window size and the number of
    ///          layers using each window size, as well as the sum of cache sizes per token for each window.
    /// @param config KV cache configuration parameters
    /// @param isCrossAttention Whether this is for cross-attention KV cache
    /// @param dtype Data type used for KV cache values
    /// @param modelConfig Model configuration containing layer and head information
    /// @param worldConfig World configuration for multi-GPU setups
    /// @param windowSizeToLayers Map from attention window size to vector of layer indices using that window size
    /// @param allottedPrimaryMemBytes Allotted primary memory
    /// @param allottedSecondaryMemBytes Allotted secondary memory
    /// @param extraCostMemory Additional memory cost to account for CacheTransBufferManager::preAllocBufferSize
    /// @param kvFactor Factor for KV cache size calculation (typically 2 for key+value)
    /// @return Map from window size to tuple of (primary blocks, secondary blocks)
    [[nodiscard]] static BlocksPerWindow calculateMaxNumBlocks(executor::KvCacheConfig const& config,
        bool isCrossAttention, nvinfer1::DataType dtype, tensorrt_llm::runtime::ModelConfig const& modelConfig,
        tensorrt_llm::runtime::WorldConfig const& worldConfig,
        std::map<SizeType32, std::vector<SizeType32>> const& windowSizeToLayers, uint64_t allottedPrimaryMemBytes,
        uint64_t allottedSecondaryMemBytes, size_t extraCostMemory, SizeType32 kvFactor);

    /// @brief Calculates the maximum batch size that can fit the kv-cache, given that all sequences in the batch have
    /// the provided input and output length.
    ///
    /// @param inputLength The number of input tokens in each sequence in the batch.
    /// @param outputLength The number of output tokens in each sequence in the batch.
    /// @return SizeType32 A number of sequences per batch.
    [[nodiscard]] virtual SizeType32 getMaxCapacityBatchSize(SizeType32 inputLength, SizeType32 outputLength) const = 0;

    [[nodiscard]] virtual CacheType getCacheType() const = 0;

    [[nodiscard]] virtual std::shared_ptr<KVCacheBlock> findBlocksInReuseTreeByBlockKey(
        BlockKey const& blockKey, SizeType32 windowSize)
        = 0;

    virtual void unpinBlocksById(KVCacheBlock::IdType blockId) = 0;
};

class KVCacheManager : public BaseKVCacheManager
{
public:
    friend class KVCacheManagerBindings;

    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using CudaStreamPtr = std::shared_ptr<runtime::CudaStream>;
    using CacheType = tensorrt_llm::batch_manager::kv_cache_manager::CacheType;

    KVCacheManager(std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        BlocksPerWindow const& blocksPerWindow, SizeType32 maxNumSequences, SizeType32 maxBeamWidth,
        std::vector<SizeType32> const& maxAttentionWindowVec,
        std::optional<TempAttentionWindowInputs> const& tempAttentionWindowInputs, nvinfer1::DataType dtype,
        SizeType32 sinkTokenLength, CudaStreamPtr stream, SizeType32 maxSequenceLength, bool enableBlockReuse = false,
        bool onboardBlocks = true, CacheType cacheType = CacheType::kSELF,
        std::optional<executor::RetentionPriority> secondaryOffloadMinPriority = std::nullopt,
        std::shared_ptr<KVCacheEventManager> eventManager = nullptr, bool enablePartialReuse = true,
        bool copyOnpartialReuse = true,
        std::shared_ptr<kv_connector::KvCacheConnectorManager> kvCacheConnectorManager = nullptr);

    KVCacheManager(std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        BlocksPerWindow const& blocksPerWindow, SizeType32 maxNumSequences, SizeType32 maxBeamWidth,
        std::vector<SizeType32> const& maxAttentionWindowVec,
        std::optional<TempAttentionWindowInputs> const& tempAttentionWindowInputs, nvinfer1::DataType dtype,
        SizeType32 sinkTokenLength, int64_t stream, SizeType32 maxSequenceLength, bool enableBlockReuse = false,
        bool onboardBlocks = true, CacheType cacheType = CacheType::kSELF,
        std::optional<executor::RetentionPriority> secondaryOffloadMinPriority = std::nullopt,
        std::shared_ptr<KVCacheEventManager> eventManager = nullptr, bool enablePartialReuse = true,
        bool copyOnpartialReuse = true,
        std::shared_ptr<kv_connector::KvCacheConnectorManager> kvCacheConnectorManager = nullptr);

    KVCacheManager(SizeType32 numLayers, SizeType32 numKvHeads, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        BlocksPerWindow const& blocksPerWindow, SizeType32 maxNumSequences, SizeType32 maxBeamWidth,
        std::vector<SizeType32> const& maxAttentionWindowVec,
        std::optional<TempAttentionWindowInputs> const& tempAttentionWindowInputs, nvinfer1::DataType dtype,
        SizeType32 sinkTokenLength, CudaStreamPtr stream, SizeType32 maxSequenceLength, bool enableBlockReuse = true,
        bool onboardBlocks = true, CacheType cacheType = CacheType::kSELF,
        std::optional<executor::RetentionPriority> secondaryOffloadMinPriority = std::nullopt,
        std::shared_ptr<KVCacheEventManager> eventManager = nullptr, bool enablePartialReuse = true,
        bool copyOnpartialReuse = true,
        std::shared_ptr<kv_connector::KvCacheConnectorManager> kvCacheConnectorManager = nullptr);

    KVCacheManager(SizeType32 numLayers, SizeType32 numKvHeads, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        BlocksPerWindow const& blocksPerWindow, SizeType32 maxNumSequences, SizeType32 maxBeamWidth,
        std::vector<SizeType32> const& maxAttentionWindowVec,
        std::optional<TempAttentionWindowInputs> const& tempAttentionWindowInputs, nvinfer1::DataType dtype,
        SizeType32 sinkTokenLength, int64_t stream, SizeType32 maxSequenceLength, bool enableBlockReuse = false,
        bool onboardBlocks = true, CacheType cacheType = CacheType::kSELF, bool enablePartialReuse = true,
        bool copyOnpartialReuse = true);

    ~KVCacheManager() override = default;

    void allocatePools(bool useUvm = false) override;

    void releasePools() override;

    void startScheduling() override;

    [[nodiscard]] SizeType32 getTokensPerBlock() const override
    {
        return mBlockManager.getTokensPerBlock();
    }

    [[nodiscard]] SizeType32 getMaxNumBlocks() const override
    {
        return mBlockManager.getMaxNumBlocks();
    }

    [[nodiscard]] SizeType32 getUsedNumBlocks() const override
    {
        return mBlockManager.getNumAllocatedBlocks();
    }

    [[nodiscard]] SizeType32 getNumFreeBlocks() const override
    {
        return mBlockManager.getNumFreeBlocks();
    }

    [[nodiscard]] SizeType32 getNumPools() const override
    {
        return mBlockManager.getNumPools();
    }

    [[nodiscard]] SizeType32 getNumAllocTotalBlocks() const
    {
        return mBlockManager.getNumAllocTotalBlocks();
    }

    [[nodiscard]] SizeType32 getNumAllocNewBlocks() const
    {
        return mBlockManager.getNumAllocNewBlocks();
    }

    [[nodiscard]] SizeType32 getNumReusedBlocks() const noexcept override
    {
        return mBlockManager.getNumReusedBlocks();
    }

    [[nodiscard]] SizeType32 getNumMissedBlocks() const noexcept
    {
        return mBlockManager.getNumMissedBlocks();
    }

    [[nodiscard]] std::map<SizeType32, SizeType32> getNumFreeBlocksPerWindowSize() const
    {
        return mBlockManager.getNumFreeBlocksPerWindowSize();
    }

    [[nodiscard]] KvCacheStats getKvCacheStats() const override
    {
        KvCacheStats kvCacheStats;
        kvCacheStats.maxNumBlocks = getMaxNumBlocks();
        kvCacheStats.freeNumBlocks = getNumFreeBlocks();
        kvCacheStats.usedNumBlocks = getUsedNumBlocks();
        kvCacheStats.toksPerBlock = getTokensPerBlock();
        kvCacheStats.allocTotalBlocks = getNumAllocTotalBlocks();
        kvCacheStats.allocNewBlocks = getNumAllocNewBlocks();
        kvCacheStats.reusedBlocks = getNumReusedBlocks();
        kvCacheStats.missedBlocks = getNumMissedBlocks();
        kvCacheStats.cacheHitRate = kvCacheStats.reusedBlocks == 0 ? 0
                                                                   : static_cast<float>(kvCacheStats.reusedBlocks)
                / static_cast<float>(kvCacheStats.reusedBlocks + kvCacheStats.missedBlocks);
        kvCacheStats.numFreeBlocksPerWindowSize = getNumFreeBlocksPerWindowSize();
        kvCacheStats.allocatedBytes = mAllocatedBytes;
        return kvCacheStats;
    }

    [[nodiscard]] OffsetTableDimensions getOffsetTableDimensions() const override
    {
        OffsetTableDimensions dims;
        // We use the mMaxAttentionWindow here, because we prefer to have a single offset table for simplicity,
        // And we don't mind that it should be as wide as the widest window, because that is negligible.
        dims.maxBlocksPerSeq = mBlockManager.getWindowSizeMetadata(mMaxAttentionWindow).maxBlocksPerSeq;
        dims.numPools = mBlockManager.getNumPools();
        dims.cacheType = mBlockManager.getCacheType();
        return dims;
    }

    [[nodiscard]] std::deque<executor::KVCacheEvent> getLatestEvents(
        std::optional<std::chrono::milliseconds> timeout = std::nullopt) const override
    {
        return mBlockManager.getLatestEvents(timeout);
    }

    [[nodiscard]] BlockManager const& getBlockManager() const override
    {
        return mBlockManager;
    }

    /// @brief  Function that computes the number of KV cache blocks needed to advance a request by one or two
    /// iterations
    /// @param req The request for which we need to calculate the number of needed KV cache blocks
    /// @return  The number of blocks
    [[nodiscard]] SizeType32 getNeededBlocksOneStep(
        LlmRequest const& req, bool twoStepsLookAhead, SizeType32 windowSize) const override;

    /// @brief  Function that computes the number of KV cache blocks remaining to advance a request to completion (i.e.
    /// for maxNewTokens); the allocated blocks are excluded
    /// @param req The request for which we need to calculate the number of needed KV cache blocks
    /// @return  The number of blocks
    [[nodiscard]] SizeType32 getRemainingBlocksToCompletion(
        LlmRequest const& req, SizeType32 windowSize) const override;

    /// @brief Increase size for request with requestId. Allocate new KV cache block(s) if needed.
    void addToken(LlmRequest::RequestIdType requestId) override;

    /// @brief Add new request to the KV cache manager.
    /// @param inputLength Input length for which KV cache need to be allocated.
    /// @param beamWidth Beam width for which KV cache need to be allocated.
    /// @param llmRequest Optional request to use for KV cache lookup.
    /// @details If llmRequest is supplied and KV cache reuse is enabled, try to recover KV cache blocks for
    /// inputLength - 1 tokens and populate prepopulatedPromptLen.
    void addSequence(LlmRequest::RequestIdType requestId, SizeType32 inputLength, SizeType32 beamWidth,
        OptionalRef<LlmRequest> llmRequest = std::nullopt) override;

    [[nodiscard]] std::optional<KVCacheBlock::IdType> removeSequence(LlmRequest::RequestIdType requestId,
        OptionalRef<LlmRequest const> llmRequest = std::nullopt, bool pinOnRelease = false) override;

    void schedulingRemoveSequence(LlmRequest::RequestIdType requestId) override;

    [[nodiscard]] runtime::ITensor::SharedPtr getBlockPoolPointers() const override
    {
        return mBlockPoolPointers;
    }

    [[nodiscard]] runtime::ITensor::SharedPtr getLayerToPoolMapping() const override
    {
        return mLayerToPoolMapping;
    }

    [[nodiscard]] runtime::ITensor::SharedPtr getBlockScalePoolPointers() const override
    {
        // TODO: add a new optional model input so the attention plugin can access these
        return mBlockScalePoolPointers;
    }

    void getBlockOffsetsOfBatch(runtime::ITensor& output, SizeType32 firstBatchSlotIdx, SizeType32 batchSize,
        SizeType32 beamWidth) const override;

    //! @return maxBlockCount of all beams
    SizeType32 copyBlockOffsets(
        runtime::ITensor& output, SizeType32 outputSlotOffset, LlmRequest::RequestIdType requestId) const override;

    [[nodiscard]] bool isEnableBlockReuse() const override
    {
        return mEnableBlockReuse;
    }

    void removeToken(LlmRequest::RequestIdType requestId);
    void rewindKVCache(LlmRequest::RequestIdType requestId, SizeType32 rewindLengths) override;

    [[nodiscard]] GenerationRequest const& getSequence(LlmRequest::RequestIdType requestId) const override;
    [[nodiscard]] GenerationRequest& getSequence(LlmRequest::RequestIdType requestId) override;

    [[nodiscard]] bool isCrossKv() const override
    {
        return mBlockManager.getCacheType() == CacheType::kCROSS;
    }

    [[nodiscard]] CacheType getCacheType() const override
    {
        return mBlockManager.getCacheType();
    }

    //! \brief Find first new block that must be allocated for context phase and return it's concatenated token vector.
    //! \details Only full blocks are considered.
    [[nodiscard]] std::optional<BlockKey> findNewContextBlock(
        VecUniqueTokens const& uniqueTokens, LlmRequest const& llmRequest) const override;

    //! \brief Store full context blocks contributed by llmRequest.
    //! \details These blocks become reusable from next step.
    void storeContextBlocks(LlmRequest const& llmRequest) override;

    //! \brief Store newest blocks for reuse
    void storeNewBlock(LlmRequest const& llmRequest) override;

    [[nodiscard]] std::optional<KVCacheBlock::IdType> storeBlocksForReuse(
        LlmRequest::RequestIdType requestId, OptionalRef<LlmRequest const> llmRequest, bool pinBlocks = false) override;

    [[nodiscard]] static SizeType32 getSinkBubbleLength(SizeType32 sinkTokenLen, SizeType32 tokensPerBlock);

    [[nodiscard]] SizeType32 getMaxCapacityBatchSize(SizeType32 inputLength, SizeType32 outputLength) const override;

    /// @brief Calculates the number of kv-cache blocks that a sequence will require.
    ///
    /// @param inputLength The number of input tokens in the sequence.
    /// @param outputLength The number of output tokens in the sequence.
    /// @param sinkTokenLength The number of sink tokens configured.
    /// @param maxAttentionWindow The attention window size allowed by the model.
    /// @param beamWidth The number of beams to consider for the request.
    /// @param tokensPerBlock The number of tokens a single kv-cache block contains.,
    /// @return SizeType32 A number of blocks.
    [[nodiscard]] static SizeType32 calculateMaxBlockRequirements(SizeType32 inputLength, SizeType32 outputLength,
        SizeType32 sinkTokenLength, SizeType32 windowSize, SizeType32 beamWidth, SizeType32 tokensPerBlock);

    void pinBlocks(LlmRequest::RequestIdType requestId) override;

    void unpinBlocksById(KVCacheBlock::IdType blockId) override;

    std::optional<KVCacheBlock::IdType> getLastBlockId(LlmRequest::RequestIdType requestId) const override;

    /// @brief Calculates the number of kv-cache blocks that a sequence will require, for a single beam.
    ///
    /// @param sequenceLength The total length of the sequence (input and output).
    /// @param sinkTokenLength The number of sink tokens configured.
    /// @param windowSize The attention window size
    /// @param tokensPerBlock The number of tokens in a single kv-cache block.
    /// @return SizeType32 A number of blocks.
    [[nodiscard]] static SizeType32 calculateMaxBlockRequirementsPerBeam(
        SizeType32 sequenceLength, SizeType32 sinkTokenLength, SizeType32 windowSize, SizeType32 tokensPerBlock);

    std::vector<std::vector<SizeType32>> const& getCacheBlockIds(
        LlmRequest::RequestIdType requestId, SizeType32 windowSize) const override;

    std::vector<std::vector<std::vector<SizeType32>>> getBatchCacheBlockIds(
        std::vector<LlmRequest::RequestIdType> const& requestIds, SizeType32 windowSize) const override;

    runtime::ITensor::SharedPtr getUniquePrimaryPool() const override;
    runtime::ITensor::SharedPtr getPrimaryPool(SizeType32 layer_idx) const override;

    SizeType32 getPoolLayerIdx(SizeType32 layer_idx) const override
    {
        return mBlockManager.getPoolLayerIdx(layer_idx);
    }

    //! \brief Perform per-iteration bookkeeping
    void refreshBlocks() override
    {
        mBlockManager.refreshBlocks();
    }

    void flushIterationEvents() override
    {
        mBlockManager.flushIterationEvents();
    }

    std::shared_ptr<KVCacheBlock> findBlocksInReuseTreeByBlockKey(
        BlockKey const& blockKey, SizeType32 windowSize) override
    {
        return mBlockManager.findBlocksInReuseTreeByBlockKey(blockKey, windowSize);
    }

    /// @brief Finds the maximum attention window that can be used on a sequence, given some kv-cache block capacity.
    ///
    /// @param inputLength The number of input tokens in the sequence.
    /// @param outputLength The number of output tokens in the sequence.
    /// @param sinkTokenLength The number of sink tokens.
    /// @param blockCapacity The number of kv-cache blocks available.
    /// @param beamWidth The number of beams to consider.
    /// @param tokensPerBlock The number of tokens per kv-cache block.
    /// @return SizeType32 A maximum attention window in number of tokens.
    [[nodiscard]] static SizeType32 calculateMaxAttentionWindow(SizeType32 inputLength, SizeType32 outputLength,
        SizeType32 sinkTokenLength, SizeType32 blockCapacity, SizeType32 beamWidth, SizeType32 tokensPerBlock);

private:
    // Maximum number of sequences
    SizeType32 mMaxNumSequences;
    // Maximum beam width
    SizeType32 mMaxBeamWidth;
    nvinfer1::DataType mDataType;
    // Maximum kv cache length per sequence
    SizeType32 mMaxAttentionWindow;
    // Number of tokens per block
    SizeType32 mTokensPerBlock;
    // Number of tokens to fill up the sink tokens to a full block size
    SizeType32 mSinkBubbleLength;
    // Number of tokens in the sink blocks
    SizeType32 mSinkBlockTokenLength;
    // Block manager
    BlockManager mBlockManager;
    // Map of all sequences
    std::unordered_map<LlmRequest::RequestIdType, GenerationRequest> mSequences;
    // Whether to cache KV pages for reuse
    bool mEnableBlockReuse;
    // Mutex to protect access to mSequences
    mutable std::mutex mSequencesMtx;
    // buffers for static tensors, will be created after allocating pools
    runtime::ITensor::SharedPtr mBlockPoolPointers;
    runtime::ITensor::SharedPtr mLayerToPoolMapping;
    runtime::ITensor::SharedPtr mBlockScalePoolPointers;
    // GPU bytes allocated for KV-cache
    std::size_t mAllocatedBytes{0};
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
