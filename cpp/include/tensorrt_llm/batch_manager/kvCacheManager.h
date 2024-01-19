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
#include "tensorrt_llm/batch_manager/llmRequest.h" // TODO forward declare
#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/cudaStream.h"
#include "tensorrt_llm/runtime/gptModelConfig.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <NvInferRuntime.h>
#include <cstdint>
#include <functional>
#include <list>
#include <memory>
#include <optional>
#include <unordered_map>
#include <vector>

namespace std
{

// Implement std::hash function object for vector<TokenIdType>.
// This allows us to use unordered_map with vector<TokenIdType> as key.
// Based on https://stackoverflow.com/questions/20511347/a-good-hash-function-for-a-vector/72073933#72073933

template <>
struct hash<vector<int32_t>>
{
    size_t operator()(vector<int32_t> const& vec) const noexcept
    {
        size_t seed = vec.size();
        for (auto x : vec)
        {
            uint32_t y = static_cast<uint32_t>(x);
            y = ((y >> 16) ^ y) * 0x45d9f3b;
            y = ((y >> 16) ^ y) * 0x45d9f3b;
            y = (y >> 16) ^ y;
            seed ^= y + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
        return seed;
    }
};

} // namespace std

namespace tensorrt_llm::batch_manager::kv_cache_manager
{

class KVCacheBlock;

using SizeType = tensorrt_llm::runtime::SizeType;
using TokenIdType = tensorrt_llm::runtime::TokenIdType;
using VecTokens = std::vector<TokenIdType>;
using BeamTokens = std::vector<VecTokens>;
using BlockPtr = std::shared_ptr<KVCacheBlock>;
using FreeBlocksQueue = std::list<BlockPtr>;
using NextBlockMap = std::unordered_map<VecTokens, BlockPtr>;

struct KvCacheStats
{
    SizeType maxNumBlocks;
    SizeType freeNumBlocks;
    SizeType usedNumBlocks;
    SizeType toksPerBlock;
};

// Basic building block of a paged KV cache - a single
// cache block. This class just holds metadata, no pointers
// since it is reused across all layers.
class KVCacheBlock
{
public:
    explicit KVCacheBlock(SizeType blockIdx);

    void startScheduling();

    [[nodiscard]] SizeType getBlockIdx() const;

    void incRefCount();

    void decRefCount();

    void decSchedulingRefCount();

    [[nodiscard]] bool hasRefs() const;

    [[nodiscard]] bool hasSchedulingRefs() const;

    void setTokens(VecTokens& tokens, bool isFull);

    [[nodiscard]] VecTokens const& getTokens() const;

    void setFreeBlockIterator(FreeBlocksQueue::iterator freeBlockIterator);

    void resetFreeBlockIterator();

    [[nodiscard]] std::optional<FreeBlocksQueue::iterator> const& getFreeBlockIterator() const;

    void setPrevBlock(BlockPtr prevBlock);

    void addNextBlock(VecTokens const& tokens, BlockPtr block);

    void removeNextBlock(VecTokens const& tokens);

    static std::shared_ptr<KVCacheBlock> findLeafBlock(std::shared_ptr<KVCacheBlock> searchStart);

    [[nodiscard]] BlockPtr findMatchingBlock(VecTokens const& tokens) const;

    //! \brief Free block from previous block if present.
    void freeLeafBlock();

    [[nodiscard]] bool isFull() const;

    [[nodiscard]] bool isShared() const;

private:
    // Linear index of block in pool
    SizeType mBlockIdx;

    // Number of references to the block
    SizeType mRefCount;

    // Number of references to the block
    SizeType mSchedulingRefCount;

    // Key of this block in mNextBlocks map in block pointed to by mPrevBlock
    VecTokens mTokens;

    // Previous block in sequence
    BlockPtr mPrevBlock;

    // Next block(s) in sequence(s)
    NextBlockMap mNextBlocks;

    // Iterator pointing to this block in mFreeBlocks.
    std::optional<FreeBlocksQueue::iterator> mFreeBlockIterator;

    // Flag indicating if block is full
    bool mIsFull;
};

class GenerationRequest
{
public:
    using SizeType = tensorrt_llm::runtime::SizeType;
    using SharedPtr = std::shared_ptr<GenerationRequest>;

    explicit GenerationRequest(SizeType seqSlotIdx, SizeType numTokens, SizeType beamWidth)
        : mSeqSlotIdx(seqSlotIdx)
        , mNumTokens(numTokens)
        , mBeamWidth(beamWidth)
        , mCacheBlockIds(beamWidth)
    {
    }

    void addNewTokens(SizeType n)
    {
        mNumTokens += n;
    }

    [[nodiscard]] SizeType getSequenceSlotIdx() const
    {
        return mSeqSlotIdx;
    }

    [[nodiscard]] SizeType getNumTokens() const
    {
        return mNumTokens;
    }

    [[nodiscard]] SizeType getBeamWidth() const
    {
        return mBeamWidth;
    }

    [[nodiscard]] std::vector<std::vector<SizeType>> const& getCacheBlockIds() const
    {
        return mCacheBlockIds;
    }

    void addCacheBlock(SizeType beamIdx, SizeType blockIdx)
    {
        mCacheBlockIds.at(beamIdx).push_back(blockIdx);
    }

    void changeCacheBlock(SizeType beamIdx, SizeType pagedBlockIdx, SizeType blockIdx)
    {
        mCacheBlockIds.at(beamIdx).at(pagedBlockIdx) = blockIdx;
    }

    void clearCacheBlocks()
    {
        for (auto& beamBlockIds : mCacheBlockIds)
        {
            beamBlockIds.clear();
        }
    }

    void setNumPrepopulatedTokens(std::vector<int> numPrepopulatedTokens)
    {
        mNumPrepopulatedTokens = std::move(numPrepopulatedTokens);
    }

    [[nodiscard]] std::vector<int> const& getNumPrepopulatedTokens() const
    {
        return mNumPrepopulatedTokens;
    }

private:
    // Slot id of the sequence
    SizeType mSeqSlotIdx;
    // Current number of generated tokens
    SizeType mNumTokens;
    // Number of beams
    SizeType mBeamWidth;
    // List of blocks allocated for each beam of the sequence
    std::vector<std::vector<SizeType>> mCacheBlockIds;
    // Number of tokens already in kv cache before context phase.
    // A value > 0 indicates cached kv cache blocks were reused.
    // One value per beam.
    std::vector<int> mNumPrepopulatedTokens;
};

// BlockManager manages overall metadata of KVCacheBlocks in a layer of the
// network. Layers are expected to be symmetric, so the metadata can be
// reused for all layers of the network.
// The array of cache blocks for a layer is called a pool.
// Each pool has shape [max_blocks, 2, num_heads, tokens_per_block, head_size].
// Size per block and number of blocks per pool are pre-determined and set in
// constructor. These should not be changed after.
// Block shape is [2, num_heads, tokens_per_block, head_size].
// BlockManager maintains a list of free blocks at any time.
// Alloc pops off the block at the front, and Free pushes it back to the vector.
// BlockManager maintains a vector of lists of seqSlotIdx to allocated blocks
// per sequence. This can be used to Free all blocks belonging to a sequence.
class BlockManager
{
public:
    using SizeType = tensorrt_llm::runtime::SizeType;

    explicit BlockManager(SizeType blocksInPool, SizeType tokensPerBlock);

    ~BlockManager();

    void startScheduling();

    //! \brief Assign blocks for new sequence. Try to reuse blocks.
    void addSequence(GenerationRequest& sequence, SizeType inputLength, std::shared_ptr<LlmRequest> const& llmRequest);

    //! \brief Assign blocks for new sequence. Does not try to reuse blocks.
    void addSequence(GenerationRequest& sequence, SizeType numBlocks, SizeType unsharedBlockIdx);

    //! \brief Allocate new block for each beam of the sequence.
    //! \details Might free cached blocks if no free blocks are available.
    void allocateBlock(GenerationRequest& sequence, bool shareAmongBeams = false);

    void replaceSharedBlock(GenerationRequest& sequence, SizeType blockIdx);

    //! \brief Release blocks of the sequence. Store blocks for reuse if llmReqeust is provided.
    void releaseBlocks(GenerationRequest& sequence, std::shared_ptr<LlmRequest> const& llmRequest = nullptr);

    //! \brief Simulate freeing all blocks for that sequence to check impact on number of free blocks
    void schedulingReleaseBlocks(GenerationRequest& sequence);

    [[nodiscard]] SizeType getNumFreeBlocks() const
    {
        return mFreeBlocks.size();
    }

    [[nodiscard]] SizeType getNumAllocatedBlocks() const
    {
        return getMaxNumBlocks() - getNumFreeBlocks();
    }

    [[nodiscard]] bool hasFreeBlocks(SizeType numRequired = 1) const
    {
        return getNumFreeBlocks() >= numRequired;
    }

    [[nodiscard]] bool schedulingHasFreeBlocks(SizeType numRequired = 1) const
    {
        return mSchedulingNumFreeBlocks >= numRequired;
    }

    [[nodiscard]] SizeType getMaxNumBlocks() const
    {
        return static_cast<SizeType>(mAllBlocksByIdx.size());
    }

    [[nodiscard]] SizeType getTokensPerBlock() const
    {
        return mTokensPerBlock;
    }

private:
    //! \brief Add single block to beam of sequence and mAllocatedBlocksPerSeq.
    void addBlockToBeam(BlockPtr& block, GenerationRequest& sequence, SizeType beamIdx, SizeType seqSlotIdx);

    //! \brief Store blocks in cached blocks.
    //! \param blockedTokens Tokens of each block.
    //! \param blockIds Id of each block.
    void storeBlocks(std::list<VecTokens> blockedTokens, std::vector<SizeType> const& blockIds);

    //! \brief Try to load blocks from cache. Allocate new blocks if necessary.
    //! \param blockedTokens Tokens of each block.
    //! \param sequence Sequence to which blocks are assigned.
    //! \param beamIdx Beam of sequence to which blocks are assigned.
    //! \param seqSlotIdx Batch slot of sequence to which blocks are assigned.
    //! \return Number of matched tokens from loaded blocks.
    SizeType loadOrAllocateBlocks(
        std::list<VecTokens> blockedTokens, GenerationRequest& sequence, SizeType beamIdx, SizeType seqSlotIdx);

    //! \brief Find block least likely to be reused, free it if necessary and return.
    [[nodiscard]] BlockPtr getFreeBlock();

    //! \brief Claim block if it is in free blocks list.
    void claimBlock(KVCacheBlock& block);

    //! \brief Free block from previous block and claim it from free blocks list.
    void claimLeafBlock(KVCacheBlock& block);

private:
    // List of free blocks
    FreeBlocksQueue mFreeBlocks;
    // List of allocated blocks for each sequences
    std::vector<std::vector<BlockPtr>> mAllocatedBlocksPerSeq;
    // Used to keep track of number of free blocks during scheduling
    SizeType mSchedulingNumFreeBlocks;
    // Number of tokens per one block
    SizeType mTokensPerBlock;
    // List of all blocks by idx
    std::vector<BlockPtr> mAllBlocksByIdx;
    // Dummy block acting as root for BlockToken searches
    BlockPtr mCachedBlocksRoot;
    // Statistics for block allocations/reuse
    std::size_t mAllocTotalBlocks, mAllocNewBlocks, mReusedBlocks;
};

class KVCacheManager
{
public:
    using SizeType = tensorrt_llm::runtime::SizeType;
    using SequencesPtr = GenerationRequest::SharedPtr;
    using CudaStreamPtr = std::shared_ptr<runtime::CudaStream>;

    KVCacheManager(SizeType numLayers, SizeType numHeads, SizeType numKvHeads, SizeType hiddenSize,
        SizeType tokensPerBlock, SizeType maxNumBlocks, SizeType maxNumSequences, SizeType maxBeamWidth,
        SizeType maxBlocksPerSeq, SizeType maxAttentionWindow, SizeType sinkTokenLength, bool useOneMoreBlock,
        nvinfer1::DataType dtype, CudaStreamPtr stream, bool enableBlockReuse = false);

    void startScheduling();

    [[nodiscard]] SizeType getTokensPerBlock() const
    {
        return mBlockManager.getTokensPerBlock();
    }

    [[nodiscard]] SizeType getMaxNumBlocks() const
    {
        return mBlockManager.getMaxNumBlocks();
    }

    [[nodiscard]] SizeType getUsedNumBlocks() const
    {
        return mBlockManager.getNumAllocatedBlocks();
    }

    [[nodiscard]] SizeType getNumFreeBlocks() const
    {
        return mBlockManager.getNumFreeBlocks();
    }

    [[nodiscard]] KvCacheStats getKvCacheStats() const
    {
        KvCacheStats kvCacheStats;
        kvCacheStats.maxNumBlocks = getMaxNumBlocks();
        kvCacheStats.freeNumBlocks = getNumFreeBlocks();
        kvCacheStats.usedNumBlocks = getUsedNumBlocks();
        kvCacheStats.toksPerBlock = getTokensPerBlock();

        return kvCacheStats;
    }

    // Volume of [2, numKvHeads, tokensPerBlock, sizePerHead]
    [[nodiscard]] SizeType getBlockSize() const
    {
        return mBlockSize;
    }

    [[nodiscard]] BlockManager const& getBlockManager() const
    {
        return mBlockManager;
    }

    /// @brief  Function that computes the number of KV cache blocks needed to advance a request by one or two
    /// iterations
    /// @param req The request for which we need to calculate the number of needed KV cache blocks
    /// @return  The number of blocks
    SizeType getNeededBlocksOneStep(LlmRequest const& req, bool twoStepsLookAhead) const;

    /// @brief  Function that computes the number of KV cache blocks needed to advance a request to completion (i.e. for
    /// maxNewTokens)
    /// @param req The request for which we need to calculate the number of needed KV cache blocks
    /// @return  The number of blocks
    SizeType getNeededBlocksToCompletion(LlmRequest const& req) const;

    [[nodiscard]] std::vector<runtime::ITensor::SharedPtr> const& getMemoryPools() const
    {
        return mPools;
    }

    void addContextTokens(SizeType seqSlotIdx, SizeType numTokens);

    void addToken(SizeType seqSlotIdx);

    void addSequence(SizeType seqSlotIdx, SizeType inputLength, SizeType beamWidth,
        std::shared_ptr<LlmRequest> const& llmRequest = nullptr);

    void removeSequence(SizeType seqSlotIdx, std::shared_ptr<LlmRequest> const& llmRequest = nullptr);

    void schedulingRemoveSequence(SizeType seqSlotIdx);

    void getBlockPointersOfBatch(
        runtime::ITensor& dstPointers, SizeType firstBatchSlotIdx, SizeType batchSize, SizeType beamWidth) const;

    void copyBlockPointers(
        runtime::ITensor& dstPointers, SizeType dstSlotOffset, SizeType seqSlotIdx, SizeType beamWidth) const;

    // Volume of [2, numKvHeads, tokensPerBlock, sizePerHead]
    [[nodiscard]] static SizeType constexpr calculatePageSize(tensorrt_llm::runtime::GptModelConfig const& modelConfig)
    {
        return 2 * modelConfig.getNbKvHeads() * modelConfig.getTokensPerBlock() * modelConfig.getSizePerHead();
    }

    // numLayers * 2 * numKvHeads * sizePerHead
    [[nodiscard]] static SizeType constexpr calculateCacheSizePerToken(
        tensorrt_llm::runtime::GptModelConfig const& modelConfig, tensorrt_llm::runtime::WorldConfig const& worldConfig)
    {
        return modelConfig.getNbLayers(worldConfig.getPipelineParallelism()) * 2 * modelConfig.getNbKvHeads()
            * modelConfig.getSizePerHead();
    }

    [[nodiscard]] static SizeType getMaxNumTokens(KvCacheConfig const& config, nvinfer1::DataType dtype,
        tensorrt_llm::runtime::GptModelConfig const& modelConfig, tensorrt_llm::runtime::WorldConfig const& worldConfig,
        runtime::BufferManager const& bufferManager);

    [[nodiscard]] SizeType getNumPrepopulatedTokens(SizeType batchSlotIdx, SizeType beamIdx) const
    {
        auto const& prepopulatedTokens = mSequences.at(batchSlotIdx)->getNumPrepopulatedTokens();
        return prepopulatedTokens.size() > 0 ? prepopulatedTokens.at(beamIdx) : 0;
    }

    [[nodiscard]] bool isEnableBlockReuse() const
    {
        return mEnableBlockReuse;
    }

private:
    void resetBlockPointers(SizeType seqSlotIdx, SizeType beamWidth);
    void cacheBlockPointers(GenerationRequest const& seq, SizeType seqSlotIdx);
    void cacheNewBlockPointers(GenerationRequest const& seq, SizeType seqSlotIdx);
    void updateNewBlockPointer(const GenerationRequest& seq, SizeType seqSlotIdx, SizeType blockIdx);

private:
    // Number of elements per one blocks
    SizeType mBlockSize;
    // Maximum number of sequences
    SizeType mMaxNumSequences;
    // Maximum beam width
    SizeType mMaxBeamWidth;
    // Maximum number of blocks per sequence
    SizeType mMaxBlocksPerSeq;
    // Maximum kv cache length per sequence
    // Enable cyclic kv cache when it exceeds
    SizeType mMaxAttentionWindow;
    // Sink token length in the kv cache per sequence
    SizeType mSinkTokenLength;
    // Bubble token length
    SizeType mBubbleLength;
    // Maximum token length (including bubble)
    SizeType mMaxTokenNum;
    // Number of tokens in the sink blocks
    SizeType mSinkBlockTokenLength;
    // Pools
    std::vector<runtime::ITensor::SharedPtr> mPools;
    // Block manager
    BlockManager mBlockManager;
    // List of all sequences
    std::vector<SequencesPtr> mSequences;
    // buffer for block pointers for all managed sequences
    runtime::ITensor::SharedPtr mSequenceBlockPointers;
    // Buffer manager
    runtime::BufferManager mBufferManager;
    // Whether to cache KV pages for reuse
    bool mEnableBlockReuse;
};
} // namespace tensorrt_llm::batch_manager::kv_cache_manager
