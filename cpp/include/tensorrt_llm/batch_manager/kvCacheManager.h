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

#include "tensorrt_llm/batch_manager/kvCacheConfig.h"
#include "tensorrt_llm/batch_manager/kvCacheEventManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h" // TODO forward declare
#include "tensorrt_llm/common/optionalRef.h"
#include "tensorrt_llm/kernels/kvCacheIndex.h"
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include <NvInferRuntime.h>

#include <cstdint>
#include <limits>
#include <list>
#include <memory>
#include <optional>
#include <set>
#include <unordered_map>
#include <vector>

namespace tensorrt_llm::batch_manager::eviction_policy
{
class BaseEvictionPolicy;
} // namespace tensorrt_llm::batch_manager::eviction_policy

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

static constexpr SizeType32 kPrimaryLevel = 0;

static constexpr SizeType32 kSecondaryLevel = 1;

class KVCacheBlock;
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

template <typename T>
using OptionalRef = tensorrt_llm::common::OptionalRef<T>;

struct BlockKey
{
    bool usesExtraIds = false;
    std::optional<LoraTaskIdType> loraTaskId = std::nullopt;
    VecUniqueTokens uniqueTokens;

    BlockKey() = default;

    explicit BlockKey(bool usesExtraIds, std::optional<LoraTaskIdType> loraTaskId, VecUniqueTokens uniqueTokens)
        : usesExtraIds(usesExtraIds)
        , loraTaskId{loraTaskId}
        , uniqueTokens{std::move(uniqueTokens)}
    {
    }

    bool operator==(BlockKey const& other) const noexcept
    {
        return (
            usesExtraIds == other.usesExtraIds && loraTaskId == other.loraTaskId && uniqueTokens == other.uniqueTokens);
    }

    int partialMatch(BlockKey const& other) const noexcept
    {
        SizeType32 numMatched{0};
        if (loraTaskId == other.loraTaskId)
        {
            auto [matchEnd, otherMatchEnd] = std::mismatch(
                uniqueTokens.begin(), uniqueTokens.end(), other.uniqueTokens.begin(), other.uniqueTokens.end());
            numMatched = std::distance(uniqueTokens.begin(), matchEnd);
        }
        return numMatched;
    }
};

// Implement hash functor for BlockKey.
// This allows us to use unordered_map with BlockKey as key.
// Based on https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector/72073933#72073933
struct BlockKeyHasher
{
    std::size_t operator()(BlockKey const& blockKey, std::size_t parentHash = 0) const noexcept
    {
        size_t seed = blockKey.uniqueTokens.size() ^ parentHash * UINT64_C(0xbf58476d1ce4e5b9);

        for (auto const& uniqueToken : blockKey.uniqueTokens)
        {
            uint32_t a = static_cast<uint32_t>(uniqueToken.tokenId);
            a = ((a >> 16) ^ a) * 0x45d9f3b;
            a = ((a >> 16) ^ a) * 0x45d9f3b;
            a = (a >> 16) ^ a;
            seed ^= a + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            if (blockKey.usesExtraIds)
            {
                uint64_t b = uniqueToken.tokenExtraId;
                b = (b ^ (b >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
                b = (b ^ (b >> 27)) * UINT64_C(0x94d049bb133111eb);
                b = b ^ (b >> 31);
                seed ^= b + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }
        }

        if (blockKey.loraTaskId)
        {
            uint64_t c = blockKey.loraTaskId.value();
            c = (c ^ (c >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
            c = (c ^ (c >> 27)) * UINT64_C(0x94d049bb133111eb);
            c = c ^ (c >> 31);
            seed ^= c + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }

        return seed;
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
        SizeType32 maxBlocks, SizeType32 cyclicThreshold, SizeType32 numPools = 1,
        executor::KvCacheRetentionConfig kvCacheRetentionConfig = executor::KvCacheRetentionConfig())
        : mRequestId(requestId)
        , mNumTokens(numTokens)
        , mBeamWidth(beamWidth)
        , mCacheBlockIds(beamWidth)
        , mCacheBlockIndices{runtime::BufferManager::cpu(
              runtime::ITensor::makeShape({numPools, beamWidth, 2, maxBlocks}),
              runtime::TRTDataType<tensorrt_llm::kernels::KVCacheIndex>::value)}
        , mKvCacheRetentionConfig(std::move(kvCacheRetentionConfig))
        , mCyclicThreshold(cyclicThreshold)
    {
        auto cacheBlockIdsRange = runtime::BufferRange<tensorrt_llm::kernels::KVCacheIndex>(*mCacheBlockIndices);
        std::fill(cacheBlockIdsRange.begin(), cacheBlockIdsRange.end(),
            tensorrt_llm::kernels::KVCacheIndex{
                std::numeric_limits<tensorrt_llm::kernels::KVCacheIndex::UnderlyingType>::max()});
    }

    void addNewTokens(SizeType32 n)
    {
        mNumTokens += n;
    }

    // ===== hstu modification start =====
    void evictBlocks(SizeType32 numBlock, SizeType32 tokensPerBlock)
    {
        mNumTokens = (mNumTokens < numBlock * tokensPerBlock) ? 0 : (mNumTokens - numBlock * tokensPerBlock);
        for (auto& beamBlockIds : mCacheBlockIds)
        {
            beamBlockIds.erase(beamBlockIds.begin(), beamBlockIds.begin() + numBlock);
        }
    }
    // ===== hstu modification end =====

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

    [[nodiscard]] SizeType32 getBeamWidth() const
    {
        return mBeamWidth;
    }

    [[nodiscard]] std::vector<std::vector<SizeType32>> const& getCacheBlockIds() const
    {
        return mCacheBlockIds;
    }

    [[nodiscard]] runtime::ITensor& getCacheBlockIndices()
    {
        return *mCacheBlockIndices;
    }

    [[nodiscard]] runtime::ITensor const& getCacheBlockIndices() const
    {
        return *mCacheBlockIndices;
    }

    void addCacheBlock(SizeType32 beamIdx, KVCacheBlock::IdType blockId)
    {
        mCacheBlockIds.at(beamIdx).push_back(blockId);
    }

    void changeCacheBlock(SizeType32 beamIdx, SizeType32 pagedBlockIdx, KVCacheBlock::IdType blockId)
    {
        mCacheBlockIds.at(beamIdx).at(pagedBlockIdx) = blockId;
    }

    void clearCacheBlocks()
    {
        for (auto& beamBlockIds : mCacheBlockIds)
        {
            beamBlockIds.clear();
        }
    }

    void removeLastBlock()
    {
        for (auto& beamBlockIds : mCacheBlockIds)
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

    // @brief Check whether the sequence uses cyclic KV cache.
    // @return `true` if we have begun overwriting the beginning of the sequence's KV cache.
    // @details If `true`, we cannot store the sequence's KV cache for reuse.
    [[nodiscard]] bool isCyclic() const
    {
        return mNumTokens >= mCyclicThreshold;
    }

private:
    // Request id of the sequence
    LlmRequest::RequestIdType mRequestId;
    // Current number of generated tokens
    SizeType32 mNumTokens;
    // Number of beams
    SizeType32 mBeamWidth;
    // List of block ids allocated for each beam of the sequence
    std::vector<std::vector<KVCacheBlock::IdType>> mCacheBlockIds;
    // Tensor of block indices allocated for each beam of the sequence
    runtime::ITensor::SharedPtr mCacheBlockIndices;
    // The retention priority to assign to decode blocks
    executor::KvCacheRetentionConfig mKvCacheRetentionConfig;

    // Number of tokens at which the KV Cache begins sliding
    SizeType32 mCyclicThreshold;
};

// attach metadata to a pool pointer
class KVCacheBlockPool
{
public:
    SizeType32 numLayers;
    SizeType32 numKvHeads;
    SizeType32 sizePerHead;
    SizeType32 tokensPerBlock;
    SizeType32 quantSize;
    SizeType32 blockSize;

    // Memory pools. Primary is fast memory, secondary is slower memory used for offloading.
    runtime::ITensor::SharedPtr primaryPtr;
    runtime::ITensor::SharedPtr secondaryPtr;

    // FP4 KV caches have extra pools that contain second level scales for dequantization.
    bool containsBlockScales;

    KVCacheBlockPool(SizeType32 numLayers, SizeType32 numKvHeads, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        SizeType32 quantSize, runtime::ITensor::SharedPtr primaryPtr = nullptr,
        runtime::ITensor::SharedPtr secondaryPtr = nullptr, bool containsBlockScales = false)
        : numLayers(numLayers)
        , numKvHeads(numKvHeads)
        , sizePerHead(sizePerHead)
        , tokensPerBlock(tokensPerBlock)
        , quantSize(quantSize)
        , blockSize((numKvHeads * sizePerHead * tokensPerBlock) / quantSize)
        , primaryPtr(std::move(primaryPtr))
        , secondaryPtr(std::move(secondaryPtr))
        , containsBlockScales(containsBlockScales)
    {
    }
};

// The BlockManager manages the metadata of KVCacheBlocks.
// It manages multiple arrays of cache blocks called pools.
// Layers with the same number of kv heads are grouped under the same pool.
// Each pool has shape [max_blocks, num_layers, 2, num_kv_heads, tokens_pre_block, head_size], where num_layers refers
// to the number of layers with the same num_kv_heads that share that pool.
// The metadata of KVCacheBlocks is shared between layers, so each block spans all of the managed pool - an allocated
// block matches some chunk of memory in each pool. The shape of the chunk in every pool is [2, num_kv_heads,
// tokens_per_block, head_size]. The size per block and number of blocks are pre-determined and set in the constructor.
//
// FP4 KV caches allocate additional pools for block scale factors. These pools have the same
// shape as the regular KV pools, except that the the last dim is head_size / N where N is determined
// by the precise FP4 format being used (16 for NVFP4). There is one block scale pool per normal pool.
//
// BlockManager maintains a list of free blocks at any time.
// Alloc pops off the block at the front, and Free pushes it back to the vector.
// BlockManager maintains a vector of lists of request ids to allocated blocks
// per sequence. This can be used to Free all blocks belonging to a sequence.
class BlockManager
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using CacheType = tensorrt_llm::batch_manager::kv_cache_manager::CacheType;
    using BaseEvictionPolicy = tensorrt_llm::batch_manager::eviction_policy::BaseEvictionPolicy;
    using BlockMap = std::unordered_multimap<size_t, BlockPtr>;
    using BlockMapIterRange = std::pair<BlockMap::const_iterator, BlockMap::const_iterator>;

    explicit BlockManager(std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead,
        SizeType32 tokensPerBlock, SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool,
        SizeType32 maxNumSequences, std::shared_ptr<runtime::CudaStream> stream, bool onboardBlocks,
        CacheType cacheType = CacheType::kSELF,
        std::optional<executor::RetentionPriority> secondaryOffloadMinPriority = std::nullopt,
        std::shared_ptr<KVCacheEventManager> eventManager = nullptr, bool enableHashKey = false,
        bool enablePartialReuse = true, bool copyOnPartialReuse = true);

    ~BlockManager();

    // ===== hstu modification start =====
    void allocatePools(nvinfer1::DataType dtype, bool useUvm, SizeType32 numReservedBlocks = 0);
    // ===== hstu modification end =====

    void releasePools();

    void startScheduling();

    //! \brief Assign blocks for new sequence. Try to reuse blocks.
    void addSequence(
        GenerationRequest& sequence, SizeType32 inputLength, SizeType32 numContextBlocks, LlmRequest& llmRequest);

    // ===== hstu modification start =====
    void evictSequence(GenerationRequest& sequence);
    void evictBlocks(GenerationRequest& sequence, SizeType32 numBlocksToEvict);
    void markSequenceEvicted(LlmRequest::RequestIdType requestId);
    void markSequenceRetained(LlmRequest::RequestIdType requestId);
    // ===== hstu modification end =====

    //! \brief Assign blocks for new sequence. Does not try to reuse blocks.
    void addSequence(GenerationRequest& sequence, SizeType32 numBlocks, SizeType32 unsharedBlockIdx);

    //! \brief Allocate new block for each beam of the sequence.
    //! \details Might free cached blocks if no free blocks are available.
    void allocateBlock(GenerationRequest& sequence, bool shareAmongBeams = false);

    void replaceSharedBlock(GenerationRequest& sequence, SizeType32 blockIdx);

    //! \brief Get the ids of all newly allocated (not reused) blocks for the sequence.
    std::vector<KVCacheBlock::IdType> getNewlyAllocatedBlockIds(GenerationRequest const& sequence) const;

    //! \brief Release blocks of the sequence. Store blocks for reuse if llmReqeust is provided.
    void releaseBlocks(GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest = std::nullopt);

    //! \brief Simulate freeing all blocks for that sequence to check impact on number of free blocks
    void schedulingReleaseBlocks(LlmRequest::RequestIdType requestId);

    //! \brief Release last block in the sequence
    void releaseLastBlock(GenerationRequest& sequence);

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

    [[nodiscard]] std::deque<executor::KVCacheEvent> getLatestEvents(
        std::optional<std::chrono::milliseconds> timeout) const;

    [[nodiscard]] bool hasFreeBlocks(SizeType32 numRequired = 1) const noexcept
    {
        return getNumFreeBlocks() >= numRequired;
    }

    [[nodiscard]] bool schedulingHasFreeBlocks(SizeType32 numRequired = 1) const noexcept
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

    [[nodiscard]] BlockMapIterRange getBlocksByHash(size_t hash) const
    {
        return mContextBlocksByHash.equal_range(hash);
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

    [[nodiscard]] SizeType32 getNumPools(bool includeBlockScalePools = true) const noexcept
    {
        if (includeBlockScalePools)
        {
            return mPools.size();
        }
        return std::count_if(mPools.begin(), mPools.end(), [](auto const& pool) { return !pool.containsBlockScales; });
    }

    [[nodiscard]] runtime::ITensor::SharedPtr getPrimaryPool(SizeType32 poolIdx) const
    {
        return mPools.at(poolIdx).primaryPtr;
    }

    [[nodiscard]] runtime::ITensor::SharedPtr getSecondaryPool(SizeType32 poolIdx) const
    {
        return mPools.at(poolIdx).secondaryPtr;
    }

    [[nodiscard]] bool containsBlockScales(SizeType32 poolIdx) const
    {
        return mPools.at(poolIdx).containsBlockScales;
    }

    [[nodiscard]] SizeType32 getNumLayers() const
    {
        return mNumLayers;
    }

    [[nodiscard]] SizeType32 getNumPrimaryBlocks() const
    {
        return mNumPrimaryBlocks;
    }

    [[nodiscard]] SizeType32 getNumSecondaryBlocks() const
    {
        return mNumSecondaryBlocks;
    }

    [[nodiscard]] CacheType getCacheType() const
    {
        return mCacheType;
    }

    [[nodiscard]] SizeType32 getLayerPoolIdx(SizeType32 layerIdx) const
    {
        return mLayerToPool.at(layerIdx);
    }

    //! \brief Maps a global layer index to its layer index within its pool.
    //! \details If we only have one pool, then getPoolLayerIdx(i) == i. Otherwise,
    //! \details gives the layer index into the getLayerPoolIdx(i).
    [[nodiscard]] SizeType32 getPoolLayerIdx(SizeType32 layerIdx) const
    {
        return mLayerIndexToPoolLayerIndex.at(layerIdx);
    }

    //! \brief Get index in pool to K or V block.
    //! \param blockId the blockId as returned by getBlockId()
    //! \param fieldIdx either 0 (K) or 1 (V),
    //! \param poolIdx the index of the pool for which the index is calculated (each pool has different strides)
    [[nodiscard]] kernels::KVCacheIndex getKOrVBlockIndex(
        KVCacheBlock::IdType blockId, SizeType32 fieldIdx, SizeType32 poolIdx) const;

    //! \brief Bring offloaded block from secondary to primary memory.
    //! \details Does nothing of block is already in primary memory.
    void onboardBlock(BlockPtr const& offloadBlock);

    //! \brief Bring block from primary to secondary memory.
    //! \details Does nothing of block is already in secondary memory.
    void offloadBlock(BlockPtr const& block);

    //! \brief Find first new block that must be allocated for context phase and return it's concatenated token vectors.
    //! \details Only full blocks are considered.
    [[nodiscard]] std::optional<BlockKey> findNewContextBlock(
        VecUniqueTokens const& uniqueTokens, LlmRequest const& llmRequest) const;

    [[nodiscard]] runtime::BufferManager const& getBufferManager() const
    {
        return mBufferManager;
    }

    // ===== hstu modification start =====
    [[nodiscard]] LlmRequest::RequestIdType getRequestIdToEvict() const
    {
        return mSeqLRUList.back();
    }
    // ===== hstu modification end =====

    //! \brief Perform per-request bookkeeping
    void refreshBlocks();

    void flushIterationEvents()
    {
        if (mEventManager)
        {
            mEventManager->flush();
        }
    }

    [[nodiscard]] static bool blockInRadixTree(BlockPtr const& block);

    [[nodiscard]] bool verifyQueueIntegrity();

    //! \brief Store context blocks
    void storeContextBlocks(GenerationRequest& sequence, LlmRequest const& llmRequest);

    [[nodiscard]] bool isEnableHashKey() const
    {
        return mEnableHashKey;
    }

private:
    //! \brief Add single block to beam of sequence and mAllocatedBlocksPerSeq.
    void addBlockToBeam(BlockPtr& block, GenerationRequest& sequence, SizeType32 beamIdx);

    //! \brief Add single block to all beams of sequence.
    void addBlockToAllBeams(BlockPtr& block, GenerationRequest& sequence);

    //! \brief Store blocks in cached blocks.
    //! \param blockKeys Key of each block.
    //! \param blockIds Id of each block.
    void storeBlocks(std::vector<BlockKey> blockKeys, std::vector<KVCacheBlock::IdType> const& blockIds);

    //! \brief Try to load blocks from cache. Allocate new blocks if necessary.
    //! \param blockKeys Key of each block.
    //! \param sequence Sequence to which blocks are assigned.
    //! \return Number of matched tokens from loaded blocks.
    SizeType32 loadOrAllocateBlocks(std::vector<BlockKey> const& blockKeys, SizeType32 numContextBlocks,
        GenerationRequest& sequence, std::vector<executor::RetentionPriorityAndDuration> const& perBlockRetentions);

    //! \brief Find block least likely to be reused, free it if necessary and return.
    [[nodiscard]] BlockPtr getFreeBlock(
        executor::RetentionPriority = executor::KvCacheRetentionConfig::kDefaultRetentionPriority,
        std::optional<std::chrono::milliseconds> durationMs = std::nullopt);

    //! \brief Free block from previous block and claim it from free blocks list.
    void claimLeafBlock(BlockPtr block, std::optional<executor::RetentionPriority> priority = std::nullopt,
        std::optional<std::chrono::milliseconds> durationMs = std::nullopt);

    void addBlockToHashMap(BlockPtr block);

    void removeBlockFromHashMap(BlockPtr block);

private:
    //! \brief For FP4 quantization. Creates pool objects for FP4 block scalars.
    void createBlockScalePools(SizeType32 blockSize);

    // Number of blocks in pools
    SizeType32 mNumPrimaryBlocks;
    SizeType32 mNumSecondaryBlocks;

    // List of allocated blocks for each sequences
    std::unordered_map<LlmRequest::RequestIdType, std::vector<BlockPtr>> mAllocatedBlocksPerSeq;

    // ===== hstu modification start =====
    // List of sequences according to eviction order
    std::list<LlmRequest::RequestIdType> mSeqLRUList;
    std::unordered_map<LlmRequest::RequestIdType,
                       typename std::list<LlmRequest::RequestIdType>::iterator> mSeqLRUTable;
    // ===== hstu modification end =====

    // Pool per unique numKvHeads in the model
    std::vector<KVCacheBlockPool> mPools;

    // Matching of model layers to their pools
    std::vector<SizeType32> mLayerToPool;
    // See getPoolLayerIdx
    std::vector<SizeType32> mLayerIndexToPoolLayerIndex;

    // Whether offloaded blocks should be onboarded before reuse.
    bool mOnboardBlocks;
    // Buffer manager
    runtime::BufferManager mBufferManager;

    // Size of a single KV heads
    SizeType32 mSizePerHead;
    // Number of layers
    SizeType32 mNumLayers;
    // Used to keep track of number of free blocks during scheduling
    SizeType32 mSchedulingNumFreeBlocks;
    // Number of tokens per one block
    SizeType32 mTokensPerBlock;
    // List of all blocks by idx
    std::vector<BlockPtr> mAllBlocksById;
    // List of all context blocks by hash
    BlockMap mContextBlocksByHash;
    // Dummy block acting as root for BlockToken searches
    BlockPtr mCachedBlocksRoot;
    // KV cache type (self or cross)
    CacheType mCacheType;
    // Eviction Policy
    std::shared_ptr<BaseEvictionPolicy> mEvictionPolicy;
    // Event manager
    std::shared_ptr<KVCacheEventManager> mEventManager;
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
    // Number of reused tokens
    double mReusedTokens;
    // Total number of input tokens
    double mTotalInputTokens;

    // Whether or not to maintain a hashmap of blocks.
    bool mEnableHashKey;

    // Whether blocks that are partially matched should be reused.
    bool mEnablePartialReuse;

    // Whether partially matched blocks that are already in use should be copied and reused.
    bool mCopyOnPartialReuse;

private:
    friend class KVCacheManager;
};

class BaseKVCacheManager
{
public:
    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using CudaStreamPtr = std::shared_ptr<runtime::CudaStream>;
    using CacheType = tensorrt_llm::batch_manager::kv_cache_manager::CacheType;

    virtual ~BaseKVCacheManager() {}

    virtual void allocatePools(nvinfer1::DataType dtype, bool useUvm = false) = 0;

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

    [[nodiscard]] virtual SizeType32 getMaxBlocksPerSeq() const = 0;

    [[nodiscard]] virtual std::deque<executor::KVCacheEvent> getLatestEvents(
        std::optional<std::chrono::milliseconds> timeout = std::nullopt) const
        = 0;

    [[nodiscard]] virtual BlockManager const& getBlockManager() const = 0;

    /// @brief  Function that computes the number of KV cache blocks needed to advance a request by one or two
    /// iterations
    /// @param req The request for which we need to calculate the number of needed KV cache blocks
    /// @return  The number of blocks
    [[nodiscard]] virtual SizeType32 getNeededBlocksOneStep(LlmRequest const& req, bool twoStepsLookAhead) const = 0;

    /// @brief  Function that computes the number of KV cache blocks needed to advance a request to completion (i.e. for
    /// maxNewTokens)
    /// @param req The request for which we need to calculate the number of needed KV cache blocks
    /// @return  The number of blocks
    [[nodiscard]] virtual SizeType32 getRemainingBlocksToCompletion(LlmRequest const& req) const = 0;

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

    virtual void removeSequence(
        LlmRequest::RequestIdType requestId, OptionalRef<LlmRequest const> llmRequest = std::nullopt)
        = 0;

    virtual void schedulingRemoveSequence(LlmRequest::RequestIdType requestId) = 0;

    // ===== hstu modification start =====
    virtual void addSequenceWithEviction(LlmRequest::RequestIdType requestId, SizeType32 start_pos, SizeType32 inputLength,
        SizeType32 beamWidth, OptionalRef<LlmRequest> llmRequest = std::nullopt)
        = 0;

    virtual void offloadSequence(
        LlmRequest::RequestIdType requestId, std::optional<SizeType32> numTokens = std::nullopt) = 0;

    virtual void evictAllSequences(void) = 0;

    virtual SizeType32 getNumTokensCached(LlmRequest::RequestIdType requestId) const = 0;
    virtual SizeType32 getCacheStartPos(LlmRequest::RequestIdType requestId) const = 0;
    // ===== hstu modification end =====

    [[nodiscard]] virtual runtime::ITensor::SharedPtr getBlockPoolPointers() const = 0;

    [[nodiscard]] virtual runtime::ITensor::SharedPtr getLayerToPoolMapping() const = 0;

    virtual void getBlockOffsetsOfBatch(
        runtime::ITensor& output, SizeType32 firstBatchSlotIdx, SizeType32 batchSize, SizeType32 beamWidth) const
        = 0;

    //! @return maxBlockCount of all beams
    virtual SizeType32 copyBlockOffsets(
        runtime::ITensor& output, SizeType32 outputSlotOffset, LlmRequest::RequestIdType requestId) const
        = 0;

    [[nodiscard]] virtual bool isEnableBlockReuse() const = 0;

    [[nodiscard]] virtual bool isUseOneMoreBlock() const = 0;

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

    [[nodiscard]] virtual bool schedulingHasFreeBlocks(SizeType32 numRequired = 1) const = 0;

    [[nodiscard]] virtual std::vector<std::vector<SizeType32>> const& getCacheBlockIds(
        LlmRequest::RequestIdType requestId) const
        = 0;

    [[nodiscard]] virtual std::vector<std::vector<std::vector<SizeType32>>> getBatchCacheBlockIds(
        std::vector<LlmRequest::RequestIdType> const& requestIds) const
        = 0;

    [[nodiscard]] virtual std::vector<KVCacheBlock::IdType> getNewlyAllocatedBlockIds(
        LlmRequest::RequestIdType requestId) const
        = 0;

    [[nodiscard]] virtual runtime::ITensor::SharedPtr getPrimaryPool(SizeType32 layer_idx) const = 0;
    [[nodiscard]] virtual SizeType32 getPoolLayerIdx(SizeType32 layer_idx) const = 0;

    virtual void refreshBlocks() = 0;
    virtual void flushIterationEvents() = 0;

    [[nodiscard]] static SizeType32 getSinkBubbleLength(SizeType32 sinkTokenLen, SizeType32 tokensPerBlock);

    // Sum of numLayers * kvFactor * numKvHeads * sizePerHead for each pool
    [[nodiscard]] static SizeType32 calculateCacheSizePerToken(tensorrt_llm::runtime::ModelConfig const& modelConfig,
        tensorrt_llm::runtime::WorldConfig const& worldConfig, bool isCrossAttention = false, SizeType32 kvFactor = 2)
    {
        // NOTE: We expect the initialization of modelConfig to have already taken the tp size into account and do not
        // address it here
        // consider only local layers for the calculation
        return modelConfig.getSumLocalKvHeads(
                   worldConfig.getPipelineParallelism(), worldConfig.getPipelineParallelRank(), isCrossAttention)
            * kvFactor * modelConfig.getSizePerHead();
    }

    [[nodiscard]] static std::tuple<SizeType32, SizeType32> calculateMaxNumBlocks(KvCacheConfig const& config,
        nvinfer1::DataType dtype, tensorrt_llm::runtime::ModelConfig const& modelConfig,
        tensorrt_llm::runtime::WorldConfig const& worldConfig, runtime::BufferManager const& bufferManager,
        SizeType32 kvFactor = 2);

    /// @brief Calculates the maximum batch size that can fit the kv-cache, given that all sequences in the batch have
    /// the provided input and output length.
    ///
    /// @param inputLength The number of input tokens in each sequence in the batch.
    /// @param outputLength The number of output tokens in each sequence in the batch.
    /// @return SizeType32 A number of sequences per batch.
    [[nodiscard]] virtual SizeType32 getMaxCapacityBatchSize(SizeType32 inputLength, SizeType32 outputLength) const = 0;

    [[nodiscard]] virtual CacheType getCacheType() const = 0;
};

class KVCacheManager : public BaseKVCacheManager
{
public:
    friend class KVCacheManagerBindings;

    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using CudaStreamPtr = std::shared_ptr<runtime::CudaStream>;
    using CacheType = tensorrt_llm::batch_manager::kv_cache_manager::CacheType;

    KVCacheManager(std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool, SizeType32 maxNumSequences,
        SizeType32 maxBeamWidth, std::vector<SizeType32> const& maxAttentionWindowVec,
        SizeType32 temporaryAttentionWindow, SizeType32 sinkTokenLength, CudaStreamPtr stream,
        std::optional<SizeType32> maxSequenceLength, bool enableBlockReuse = false, bool onboardBlocks = true,
        CacheType cacheType = CacheType::kSELF,
        std::optional<executor::RetentionPriority> secondaryOffloadMinPriority = std::nullopt,
        std::shared_ptr<KVCacheEventManager> eventManager = nullptr, bool enableHashKey = false,
        bool enablePartialReuse = true, bool copyOnpartialReuse = true,
        // ===== hstu modification start =====
        SizeType32 reservedBlocksInPrimaryPool = 0);
        // ===== hstu modification end =====

    KVCacheManager(std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool, SizeType32 maxNumSequences,
        SizeType32 maxBeamWidth, std::vector<SizeType32> const& maxAttentionWindowVec,
        SizeType32 temporaryAttentionWindow, SizeType32 sinkTokenLength, int64_t stream,
        std::optional<SizeType32> maxSequenceLength, bool enableBlockReuse = false, bool onboardBlocks = true,
        CacheType cacheType = CacheType::kSELF,
        std::optional<executor::RetentionPriority> secondaryOffloadMinPriority = std::nullopt,
        std::shared_ptr<KVCacheEventManager> eventManager = nullptr, bool enablePartialReuse = true,
        bool copyOnpartialReuse = true,
        // ===== hstu modification start =====
        SizeType32 reservedBlocksInPrimaryPool = 0);
        // ===== hstu modification end =====

    KVCacheManager(SizeType32 numLayers, SizeType32 numKvHeads, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool, SizeType32 maxNumSequences,
        SizeType32 maxBeamWidth, std::vector<SizeType32> const& maxAttentionWindowVec,
        SizeType32 temporaryAttentionWindow, SizeType32 sinkTokenLength, CudaStreamPtr stream,
        std::optional<SizeType32> maxSequenceLength, bool enableBlockReuse = true, bool onboardBlocks = true,
        CacheType cacheType = CacheType::kSELF,
        std::optional<executor::RetentionPriority> secondaryOffloadMinPriority = std::nullopt,
        std::shared_ptr<KVCacheEventManager> eventManager = nullptr, bool enableHashKey = false,
        bool enablePartialReuse = true, bool copyOnpartialReuse = true,
        // ===== hstu modification start =====
        SizeType32 reservedBlocksInPrimaryPool = 0);
        // ===== hstu modification end =====

    KVCacheManager(SizeType32 numLayers, SizeType32 numKvHeads, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool, SizeType32 maxNumSequences,
        SizeType32 maxBeamWidth, std::vector<SizeType32> const& maxAttentionWindowVec,
        SizeType32 temporaryAttentionWindow, SizeType32 sinkTokenLength, int64_t stream,
        std::optional<SizeType32> maxSequenceLength, bool enableBlockReuse = false, bool onboardBlocks = true,
        CacheType cacheType = CacheType::kSELF, bool enablePartialReuse = true, bool copyOnpartialReuse = true,
        // ===== hstu modification start =====
        SizeType32 reservedBlocksInPrimaryPool = 0);
        // ===== hstu modification end =====

    ~KVCacheManager() override = default;

    void allocatePools(nvinfer1::DataType dtype, bool useUvm = false) override;

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
        return kvCacheStats;
    }

    [[nodiscard]] SizeType32 getMaxBlocksPerSeq() const override
    {
        return mMaxBlocksPerSeq;
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
    [[nodiscard]] SizeType32 getNeededBlocksOneStep(LlmRequest const& req, bool twoStepsLookAhead) const override;

    /// @brief  Function that computes the number of KV cache blocks remaining to advance a request to completion (i.e.
    /// for maxNewTokens); the allocated blocks are excluded
    /// @param req The request for which we need to calculate the number of needed KV cache blocks
    /// @return  The number of blocks
    [[nodiscard]] SizeType32 getRemainingBlocksToCompletion(LlmRequest const& req) const override;

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

    void removeSequence(
        LlmRequest::RequestIdType requestId, OptionalRef<LlmRequest const> llmRequest = std::nullopt) override;
    
    // ===== hstu modification start =====
    void addSequenceWithEviction(LlmRequest::RequestIdType requestId, SizeType32 start_pos, SizeType32 length,
        SizeType32 beamWidth, OptionalRef<LlmRequest> llmRequest = std::nullopt);

    void offloadSequence(
        LlmRequest::RequestIdType requestId, std::optional<SizeType32> numTokens = std::nullopt) override;

    void evictAllSequences(void) override;

    SizeType32 getNumTokensCached(LlmRequest::RequestIdType requestId) const override;
    SizeType32 getCacheStartPos(LlmRequest::RequestIdType requestId) const override;
    // ===== hstu modification end =====

    void schedulingRemoveSequence(LlmRequest::RequestIdType requestId) override;

    [[nodiscard]] runtime::ITensor::SharedPtr getBlockPoolPointers() const override
    {
        return mBlockPoolPointers;
    }

    [[nodiscard]] runtime::ITensor::SharedPtr getLayerToPoolMapping() const override
    {
        return mLayerToPoolMapping;
    }

    [[nodiscard]] runtime::ITensor::SharedPtr getBlockScalePoolPointers() const
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

    [[nodiscard]] bool isEnableHashKey() const
    {
        return mEnableHashKey;
    }

    [[nodiscard]] bool isUseOneMoreBlock() const override
    {
        return mUseOneMoreBlock;
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

    [[nodiscard]] static SizeType32 getSinkBubbleLength(SizeType32 sinkTokenLen, SizeType32 tokensPerBlock);

    [[nodiscard]] SizeType32 getMaxCapacityBatchSize(SizeType32 inputLength, SizeType32 outputLength) const override;

    /// @brief Calculates the number of kv-cache blocks that a sequence will require.
    ///
    /// @param inputLength The number of input tokens in the sequence.
    /// @param outputLength The number of output tokens in the sequence.
    /// @param sinkTokenLength The number of sink tokens configured.
    /// @param maxAttentionWindow The maximum attention window allowed by the model.
    /// @param beamWidth The number of beams to consider for the request.
    /// @param tokensPerBlock The number of tokens a single kv-cache block contains.,
    /// @return SizeType32 A number of blocks.
    [[nodiscard]] static SizeType32 calculateMaxBlockRequirements(SizeType32 inputLength, SizeType32 outputLength,
        SizeType32 sinkTokenLength, SizeType32 maxAttentionWindow, SizeType32 beamWidth, SizeType32 tokensPerBlock);

    /// @brief Calculates the number of kv-cache blocks that a sequence will require, for a single beam.
    ///
    /// @param sequenceLength The total length of the sequence (input and output).
    /// @param sinkTokenLength The number of sink tokens configured.
    /// @param maxAttentionWindow The maximum attention window allowed by the model.
    /// @param tokensPerBlock The number of tokens in a single kv-cache block.
    /// @return SizeType32 A number of blocks.
    [[nodiscard]] static SizeType32 calculateMaxBlockRequirementsPerBeam(SizeType32 sequenceLength,
        SizeType32 sinkTokenLength, SizeType32 maxAttentionWindow, SizeType32 tokensPerBlock);

    bool schedulingHasFreeBlocks(SizeType32 numRequired = 1) const override;

    std::vector<std::vector<SizeType32>> const& getCacheBlockIds(LlmRequest::RequestIdType requestId) const override;

    std::vector<std::vector<std::vector<SizeType32>>> getBatchCacheBlockIds(
        std::vector<LlmRequest::RequestIdType> const& requestIds) const override;

    std::vector<SizeType32> getNewlyAllocatedBlockIds(LlmRequest::RequestIdType requestId) const override;

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
    void setOffsets(kernels::KVCacheIndex* offsetsPtr, nvinfer1::Dims const& offsetsShape, SizeType32 beamIdx,
        SizeType32 blockIdx, KVCacheBlock::IdType blockId) const;

    void cacheBlockOffsets(GenerationRequest& seq);
    void cacheNewBlockOffsets(GenerationRequest& seq);
    void updateNewBlockPointer(GenerationRequest& seq, SizeType32 blockIdx);
    void updateToken(GenerationRequest& sequence, bool addToken);
    // ===== hstu modification start =====
    void appendTokens(GenerationRequest& sequence, SizeType32 numTokens);
    // ===== hstu modification end =====

private:
    // Maximum number of sequences
    SizeType32 mMaxNumSequences;
    // Maximum beam width
    SizeType32 mMaxBeamWidth;
    // Maximum number of blocks per sequence
    SizeType32 mMaxBlocksPerSeq;
    // Maximum kv cache length per sequence
    SizeType32 mMaxAttentionWindow;
    // Minimum kv cache length per sequence
    SizeType32 mMinAttentionWindow;
    // Temporary kv cache length per sequence.
    // Only needed when chunked context + sliding window attention are used together.
    // And it should only be considered when allocating blocks.
    SizeType32 mTemporaryAttentionWindow;
    // Number of tokens per block
    SizeType32 mTokensPerBlock;
    // Number of tokens to fill up the sink tokens to a full block size
    SizeType32 mSinkBubbleLength;
    // Maximum token length (including bubble)
    SizeType32 mMaxTokenNum;
    // Number of tokens in the sink blocks
    SizeType32 mSinkBlockTokenLength;
    // Block manager
    BlockManager mBlockManager;
    // Map of all sequences
    std::unordered_map<LlmRequest::RequestIdType, GenerationRequest> mSequences;
    // Whether to cache KV pages for reuse
    bool mEnableBlockReuse;
    // Whether enable finding blocks by their hash, ignored when reuse enabled
    bool mEnableHashKey;
    // Whether use one more block for each sequence
    bool mUseOneMoreBlock;
    // Mutex to protect access to mSequences
    mutable std::mutex mSequencesMtx;
    // buffers for static tensors, will be created after allocating pools
    runtime::ITensor::SharedPtr mBlockPoolPointers;
    runtime::ITensor::SharedPtr mLayerToPoolMapping;
    runtime::ITensor::SharedPtr mBlockScalePoolPointers;

    // ===== hstu modification start =====
    // Number of reserved blocks mapped for host kv data
    SizeType32 mReservedBlocksInPrimaryPool;
    // Map of sequence starting position in KV cache
    std::unordered_map<LlmRequest::RequestIdType, SizeType32> mSeqCacheStartPos;
    // ===== hstu modification end =====
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
