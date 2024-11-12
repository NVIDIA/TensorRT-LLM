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
    bool hasLora;
    LoraTaskIdType loraTaskId;
    VecUniqueTokens uniqueTokens;

    BlockKey() = default;

    explicit BlockKey(bool hasLora, LoraTaskIdType loraTaskId, VecUniqueTokens uniqueTokens)
        : hasLora{hasLora}
        , loraTaskId{loraTaskId}
        , uniqueTokens{std::move(uniqueTokens)}
    {
    }

    bool operator==(BlockKey const& other) const noexcept
    {
        return (hasLora == other.hasLora && loraTaskId == other.loraTaskId && uniqueTokens == other.uniqueTokens);
    }
};

// Implement hash functor for BlockKey.
// This allows us to use unordered_map with BlockKey as key.
// Based on https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector/72073933#72073933
struct BlockKeyHasher
{
    std::size_t operator()(BlockKey const& blockKey, std::size_t parentHash = 0) const noexcept
    {
        size_t seed = blockKey.uniqueTokens.size() ^ parentHash;
        for (auto const& uniqueToken : blockKey.uniqueTokens)
        {
            uint32_t a = static_cast<uint32_t>(uniqueToken.tokenId);
            a = ((a >> 16) ^ a) * 0x45d9f3b;
            a = ((a >> 16) ^ a) * 0x45d9f3b;
            a = (a >> 16) ^ a;

            uint64_t b = uniqueToken.tokenExtraId;
            b = (b ^ (b >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
            b = (b ^ (b >> 27)) * UINT64_C(0x94d049bb133111eb);
            b = b ^ (b >> 31);

            seed ^= a + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= b + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }

        uint64_t c = blockKey.loraTaskId;
        c = (c ^ (c >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        c = (c ^ (c >> 27)) * UINT64_C(0x94d049bb133111eb);
        c = c ^ (c >> 31);
        seed ^= c + 0x9e3779b9 + (seed << 6) + (seed >> 2);

        uint32_t d = static_cast<uint32_t>(blockKey.hasLora);
        d = ((d >> 16) ^ d) * 0x45d9f3b;
        d = ((d >> 16) ^ d) * 0x45d9f3b;
        d = (d >> 16) ^ d;
        seed ^= d + 0x9e3779b9 + (seed << 6) + (seed >> 2);

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

    BlockPtr getPrevBlock() const;

    void setPrevBlock(BlockPtr prevBlock);

    void addNextBlock(BlockKey const& blockKey, BlockPtr block);

    void removeNextBlock(BlockKey const& blockKey);

    [[nodiscard]] BlockPtr findMatchingBlock(BlockKey const& blockKey) const;

    //! \brief Free block from previous block if present.
    void freeLeafBlock();

    [[nodiscard]] bool isFull() const;

    [[nodiscard]] bool isShared() const;

    void setPriority(executor::RetentionPriority priority);

    [[nodiscard]] executor::RetentionPriority getPriority() const;

    void setDurationMs(std::optional<std::chrono::milliseconds> durationMs);

    [[nodiscard]] std::optional<std::chrono::milliseconds> getDurationMs() const;

    void setExpirationTime(std::optional<std::chrono::steady_clock::time_point::duration> expirationTime);

    [[nodiscard]] std::optional<std::chrono::steady_clock::time_point::duration> getExpirationTime() const;

    void setHash(size_t hash);

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

    // Previous block in sequence
    BlockPtr mPrevBlock;

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
        SizeType32 maxBlocks, SizeType32 numPools = 1,
        executor::KvCacheRetentionConfig kvCacheRetentionConfig = executor::KvCacheRetentionConfig())
        : mRequestId(requestId)
        , mNumTokens(numTokens)
        , mBeamWidth(beamWidth)
        , mCacheBlockIds(beamWidth)
        , mCacheBlockIndices{runtime::BufferManager::cpu(
              runtime::ITensor::makeShape({numPools, beamWidth, 2, maxBlocks}),
              runtime::TRTDataType<tensorrt_llm::kernels::KVCacheIndex>::value)}
        , mKvCacheRetentionConfig(std::move(kvCacheRetentionConfig))
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
};

// attach metadata to a pool pointer
class KVCacheBlockPool
{
public:
    SizeType32 numKvHeads;
    SizeType32 numLayers;
    SizeType32 blockSize;

    // Memory pools. Primary is fast memory, secondary is slower memory used for offloading.
    runtime::ITensor::SharedPtr primaryPtr;
    runtime::ITensor::SharedPtr secondaryPtr;

    KVCacheBlockPool(SizeType32 numKvHeads, SizeType32 numLayers, SizeType32 blockSize,
        runtime::ITensor::SharedPtr primaryPtr = nullptr, runtime::ITensor::SharedPtr secondaryPtr = nullptr)
        : numKvHeads(numKvHeads)
        , numLayers(numLayers)
        , blockSize(blockSize)
        , primaryPtr(std::move(primaryPtr))
        , secondaryPtr(std::move(secondaryPtr))
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

    explicit BlockManager(std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead,
        SizeType32 tokensPerBlock, SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool,
        SizeType32 maxNumSequences, std::shared_ptr<runtime::CudaStream> stream, bool onboardBlocks,
        CacheType cacheType = CacheType::kSELF,
        std::optional<executor::RetentionPriority> secondaryOffloadMinPriority = std::nullopt,
        std::shared_ptr<KVCacheEventManager> eventManager = nullptr);

    ~BlockManager();

    void allocatePools(nvinfer1::DataType dtype, bool useUvm);

    void startScheduling();

    //! \brief Assign blocks for new sequence. Try to reuse blocks.
    void addSequence(
        GenerationRequest& sequence, SizeType32 inputLength, SizeType32 numContextBlocks, LlmRequest& llmRequest);

    //! \brief Assign blocks for new sequence. Does not try to reuse blocks.
    void addSequence(GenerationRequest& sequence, SizeType32 numBlocks, SizeType32 unsharedBlockIdx);

    //! \brief Allocate new block for each beam of the sequence.
    //! \details Might free cached blocks if no free blocks are available.
    void allocateBlock(GenerationRequest& sequence, bool shareAmongBeams = false);

    void replaceSharedBlock(GenerationRequest& sequence, SizeType32 blockIdx);

    //! \brief Release blocks of the sequence. Store blocks for reuse if llmReqeust is provided.
    void releaseBlocks(GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest = std::nullopt);

    //! \brief Simulate freeing all blocks for that sequence to check impact on number of free blocks
    void schedulingReleaseBlocks(GenerationRequest& sequence);

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

    [[nodiscard]] SizeType32 getNumPools() const noexcept
    {
        return mPools.size();
    }

    [[nodiscard]] runtime::ITensor::SharedPtr getPrimaryPool(SizeType32 poolIdx) const
    {
        return mPools.at(poolIdx).primaryPtr;
    }

    [[nodiscard]] runtime::ITensor::SharedPtr getSecondaryPool(SizeType32 poolIdx) const
    {
        return mPools.at(poolIdx).secondaryPtr;
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

    //! \brief Get index in pool to K or V block.
    //! \param blockId the blockId as returned by getBlockId()
    //! \param fieldIdx either 0 (K) or 1 (V),
    //! \param poolIdx the index of the pool for which the index is calculated (each pool has different strides)
    [[nodiscard]] kernels::KVCacheIndex getKOrVBlockIndex(
        KVCacheBlock::IdType blockId, SizeType32 fieldIdx, SizeType32 poolIdx) const;

    //! \brief Bring offloaded block from secondary to primary memory.
    //! \details Does nothing of block is already in primary memory.
    void onboardBlock(BlockPtr offloadBlock);

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

    void flushIterationEvents()
    {
        if (mEventManager)
        {
            mEventManager->flush();
        }
    }

    [[nodiscard]] static bool blockInRadixTree(BlockPtr const& block);

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

    //! \brief Compute pointer to raw KV block (K & V, all layers).
    [[nodiscard]] runtime::ITensor::SharedPtr computeBlockPointer(
        std::shared_ptr<KVCacheBlock> block, SizeType32 poolIdx) const;

    //! \brief Copy content of src block to dst.
    void copyBlock(BlockPtr src, BlockPtr dst);

private:
    // Number of blocks in pools
    SizeType32 mNumPrimaryBlocks;
    SizeType32 mNumSecondaryBlocks;

    // List of allocated blocks for each sequences
    std::unordered_map<LlmRequest::RequestIdType, std::vector<BlockPtr>> mAllocatedBlocksPerSeq;

    // Pool per unique numKvHeads in the model
    std::vector<KVCacheBlockPool> mPools;
    // Matching of model layers to their pools
    std::vector<SizeType32> mLayerToPool;

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
    // Dummy block acting as root for BlockToken searches
    BlockPtr mCachedBlocksRoot;
    // KV cache type (self or cross)
    CacheType mCacheType;
    // Eviction Policy
    std::shared_ptr<BaseEvictionPolicy> mEvictionPolicy;
    // Event manager
    std::shared_ptr<KVCacheEventManager> mEventManager;

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
    std::set<KVCacheBlock::IdType> reusedBlockIds;

private:
    friend class KVCacheManager;
};

class KVCacheManager
{
public:
    friend class KVCacheManagerBindings;

    using SizeType32 = tensorrt_llm::runtime::SizeType32;
    using CudaStreamPtr = std::shared_ptr<runtime::CudaStream>;
    using CacheType = tensorrt_llm::batch_manager::kv_cache_manager::CacheType;

    KVCacheManager(std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool, SizeType32 maxNumSequences,
        SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength, bool useOneMoreBlock,
        CudaStreamPtr stream, bool enableBlockReuse = false, bool onboardBlocks = true,
        CacheType cacheType = CacheType::kSELF,
        std::optional<executor::RetentionPriority> secondaryOffloadMinPriority = std::nullopt,
        std::shared_ptr<KVCacheEventManager> eventManager = nullptr);

    KVCacheManager(SizeType32 numLayers, SizeType32 numKvHeads, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool, SizeType32 maxNumSequences,
        SizeType32 maxBeamWidth, SizeType32 maxAttentionWindow, SizeType32 sinkTokenLength, bool useOneMoreBlock,
        CudaStreamPtr stream, bool enableBlockReuse = true, bool onboardBlocks = true,
        CacheType cacheType = CacheType::kSELF,
        std::optional<executor::RetentionPriority> secondaryOffloadMinPriority = std::nullopt,
        std::shared_ptr<KVCacheEventManager> eventManager = nullptr);

    void allocatePools(nvinfer1::DataType dtype, bool useUvm = false);

    void startScheduling();

    [[nodiscard]] SizeType32 getTokensPerBlock() const
    {
        return mBlockManager.getTokensPerBlock();
    }

    [[nodiscard]] SizeType32 getMaxNumBlocks() const
    {
        return mBlockManager.getMaxNumBlocks();
    }

    [[nodiscard]] SizeType32 getUsedNumBlocks() const
    {
        return mBlockManager.getNumAllocatedBlocks();
    }

    [[nodiscard]] SizeType32 getNumFreeBlocks() const
    {
        return mBlockManager.getNumFreeBlocks();
    }

    [[nodiscard]] SizeType32 getNumAllocTotalBlocks() const
    {
        return mBlockManager.getNumAllocTotalBlocks();
    }

    [[nodiscard]] SizeType32 getNumAllocNewBlocks() const
    {
        return mBlockManager.getNumAllocNewBlocks();
    }

    [[nodiscard]] SizeType32 getNumReusedBlocks() const noexcept
    {
        return mBlockManager.getNumReusedBlocks();
    }

    [[nodiscard]] SizeType32 getNumMissedBlocks() const noexcept
    {
        return mBlockManager.getNumMissedBlocks();
    }

    [[nodiscard]] KvCacheStats getKvCacheStats() const
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

    [[nodiscard]] SizeType32 getMaxBlocksPerSeq() const
    {
        return mMaxBlocksPerSeq;
    }

    [[nodiscard]] std::deque<executor::KVCacheEvent> getLatestEvents(
        std::optional<std::chrono::milliseconds> timeout = std::nullopt) const
    {
        return mBlockManager.getLatestEvents(timeout);
    }

    [[nodiscard]] BlockManager const& getBlockManager() const
    {
        return mBlockManager;
    }

    /// @brief  Function that computes the number of KV cache blocks needed to advance a request by one or two
    /// iterations
    /// @param req The request for which we need to calculate the number of needed KV cache blocks
    /// @return  The number of blocks
    [[nodiscard]] SizeType32 getNeededBlocksOneStep(LlmRequest const& req, bool twoStepsLookAhead) const;

    /// @brief  Function that computes the number of KV cache blocks remaining to advance a request to completion (i.e.
    /// for maxNewTokens); the allocated blocks are excluded
    /// @param req The request for which we need to calculate the number of needed KV cache blocks
    /// @return  The number of blocks
    [[nodiscard]] SizeType32 getRemainingBlocksToCompletion(LlmRequest const& req) const;

    void addContextTokens(LlmRequest::RequestIdType requestId, SizeType32 numTokens);

    /// @brief Increase size for request with requestId. Allocate new KV cache block(s) if needed.
    void addToken(LlmRequest::RequestIdType requestId);

    /// @brief Add new request to the KV cache manager.
    /// @param inputLength Input length for which KV cache need to be allocated.
    /// @param beamWidth Beam width for which KV cache need to be allocated.
    /// @param llmRequest Optional request to use for KV cache lookup.
    /// @details If llmRequest is supplied and KV cache reuse is enabled, try to recover KV cache blocks for
    /// inputLength - 1 tokens and populate prepopulatedPromptLen.
    void addSequence(LlmRequest::RequestIdType requestId, SizeType32 inputLength, SizeType32 beamWidth,
        OptionalRef<LlmRequest> llmRequest = std::nullopt);

    void removeSequence(LlmRequest::RequestIdType requestId, OptionalRef<LlmRequest const> llmRequest = std::nullopt);

    void schedulingRemoveSequence(LlmRequest::RequestIdType requestId);

    [[nodiscard]] runtime::ITensor::SharedPtr getBlockPoolPointers() const
    {
        return mBlockPoolPointers;
    }

    [[nodiscard]] runtime::ITensor::SharedPtr getLayerToPoolMapping() const
    {
        return mLayerToPoolMapping;
    }

    void getBlockOffsetsOfBatch(
        runtime::ITensor& output, SizeType32 firstBatchSlotIdx, SizeType32 batchSize, SizeType32 beamWidth) const;

    //! @return maxBlockCount of all beams
    SizeType32 copyBlockOffsets(
        runtime::ITensor& output, SizeType32 outputSlotOffset, LlmRequest::RequestIdType requestId) const;

    // Sum of numLayers * 2 * numKvHeads * sizePerHead for each pool
    [[nodiscard]] static SizeType32 calculateCacheSizePerToken(tensorrt_llm::runtime::ModelConfig const& modelConfig,
        tensorrt_llm::runtime::WorldConfig const& worldConfig, bool isCrossAttention = false)
    {
        // NOTE: We expect the initialization of modelConfig to have already taken the tp size into account and do not
        // address it here
        // consider only local layers for the calculation
        return modelConfig.getSumLocalKvHeads(
                   worldConfig.getPipelineParallelism(), worldConfig.getPipelineParallelRank(), isCrossAttention)
            * 2 * modelConfig.getSizePerHead();
    }

    [[nodiscard]] static std::tuple<SizeType32, SizeType32> const calculateMaxNumBlocks(KvCacheConfig const& config,
        nvinfer1::DataType dtype, tensorrt_llm::runtime::ModelConfig const& modelConfig,
        tensorrt_llm::runtime::WorldConfig const& worldConfig, runtime::BufferManager const& bufferManager);

    [[nodiscard]] bool isEnableBlockReuse() const
    {
        return mEnableBlockReuse;
    }

    void removeToken(LlmRequest::RequestIdType requestId);
    void rewindKVCache(LlmRequest::RequestIdType requestId, SizeType32 rewindLengths);

    [[nodiscard]] GenerationRequest const& getSequence(LlmRequest::RequestIdType requestId) const;

    [[nodiscard]] bool isCrossKv() const
    {
        return mBlockManager.getCacheType() == CacheType::kCROSS;
    }

    //! \brief Find first new block that must be allocated for context phase and return it's concatenated token vector.
    //! \details Only full blocks are considered.
    [[nodiscard]] std::optional<BlockKey> findNewContextBlock(
        VecUniqueTokens const& uniqueTokens, LlmRequest const& llmRequest) const;

    //! \brief Store full context blocks contributed by llmRequest.
    //! \details These blocks become reusable from next step.
    void storeContextBlocks(LlmRequest const& llmRequest);

    [[nodiscard]] static SizeType32 getSinkBubbleLength(SizeType32 sinkTokenLen, SizeType32 tokensPerBlock);

    [[nodiscard]] static SizeType32 getMaxAttentionWindowUpperBound(SizeType32 blocksInPrimaryPool,
        SizeType32 tokensPerBlock, SizeType32 maxBeamWidth, SizeType32 sinkTokenLen, bool useOneMoreBlock);

    //! \brief Get the batch size that can fill the kv cache to the maximum capacity given the sequence length
    [[nodiscard]] SizeType32 getMaxCapacityBatchSize(SizeType32 seqLen);

    //! \brief Perform per-iteration bookkeeping
    void refreshBlocks()
    {
        mBlockManager.refreshBlocks();
    }

    void flushIterationEvents()
    {
        mBlockManager.flushIterationEvents();
    }

private:
    void setOffsets(kernels::KVCacheIndex* offsetsPtr, nvinfer1::Dims const& offsetsShape, SizeType32 beamIdx,
        SizeType32 blockIdx, KVCacheBlock::IdType blockId) const;

    void cacheBlockOffsets(GenerationRequest& seq);
    void cacheNewBlockOffsets(GenerationRequest& seq);
    void updateNewBlockPointer(GenerationRequest& seq, SizeType32 blockIdx);
    void updateToken(GenerationRequest& sequence, bool addToken);

private:
    // Maximum number of sequences
    SizeType32 mMaxNumSequences;
    // Maximum beam width
    SizeType32 mMaxBeamWidth;
    // Maximum number of blocks per sequence
    SizeType32 mMaxBlocksPerSeq;
    // Maximum kv cache length per sequence
    // Enable cyclic kv cache when it exceeds
    SizeType32 mMaxAttentionWindow;
    // Number of tokens per block
    SizeType32 mTokensPerBlock;
    // Number of tokens to fill up the sink tokens to a full block size
    SizeType32 mSinkBubbleLength;
    // Use one more block for each sequence
    bool mUseOneMoreBlock;
    // Maximum token length (including bubble)
    SizeType32 mMaxTokenNum;
    // Number of tokens in the sink blocks
    SizeType32 mSinkBlockTokenLength;
    // Number of blocks in primary pool
    SizeType32 mBlocksInPrimaryPool;
    // Block manager
    BlockManager mBlockManager;
    // Map of all sequences
    std::unordered_map<LlmRequest::RequestIdType, GenerationRequest> mSequences;
    // Whether to cache KV pages for reuse
    bool mEnableBlockReuse;
    // buffers for static tensors, will be created after allocating pools
    runtime::ITensor::SharedPtr mBlockPoolPointers;
    runtime::ITensor::SharedPtr mLayerToPoolMapping;
};

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
