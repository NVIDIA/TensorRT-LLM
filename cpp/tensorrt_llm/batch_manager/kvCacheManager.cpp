/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "tensorrt_llm/batch_manager/kvCacheManager.h"

#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/evictionPolicy.h"
#include "tensorrt_llm/batch_manager/kvCacheTransferManager.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/common/memoryUtils.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/kernels/kvCacheIndex.h"
#include "tensorrt_llm/runtime/common.h"
#include "tensorrt_llm/runtime/iBuffer.h"
#include "tensorrt_llm/runtime/iTensor.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/worldConfig.h"

#include <algorithm>
#include <map>
#include <optional>
#include <utility>

namespace tc = tensorrt_llm::common;
namespace tk = tensorrt_llm::kernels;
namespace tle = tensorrt_llm::executor;
using namespace tle::kv_cache;
using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::batch_manager::kv_cache_manager;
using namespace tensorrt_llm::batch_manager::eviction_policy;

using BlocksPerWindow = std::map<SizeType32, std::tuple<SizeType32, SizeType32>>;

namespace
{

inline uint8_t getNthByte(SizeType32 hashPart, uint8_t byteIdx) noexcept
{
    return static_cast<uint8_t>((hashPart >> (24 - byteIdx * 8)) & 0xFF);
}

//! \brief Get all blocks in a sequence by traversing backwards from the last block.
//! \param lastBlock is a BlockPtr to the last block in the sequence to start traversal from
//! \return Vector of BlockPtr-s in sequence order
std::vector<BlockPtr> getAllSequenceBlocks(BlockPtr lastBlock)
{
    // First count the number of blocks to pre-allocate the vector
    auto currentBlock = lastBlock;
    size_t blockCount = 0;
    while (currentBlock != nullptr && currentBlock->getBlockId() != KVCacheBlock::kCachedBlocksRootId)
    {
        blockCount++;
        currentBlock = currentBlock->getPrevBlockInSeq();
    }

    if (blockCount == 0)
    {
        return {};
    }
    // Create and pre-allocate the vector with the correct size
    std::vector<BlockPtr> sequenceBlocks(blockCount);

    // Now traverse backwards and fill from the end
    currentBlock = lastBlock;
    size_t currentIndex = blockCount - 1;
    while (currentBlock != nullptr && currentBlock->getBlockId() != KVCacheBlock::kCachedBlocksRootId)
    {
        sequenceBlocks[currentIndex--] = currentBlock;
        currentBlock = currentBlock->getPrevBlockInSeq();
    }

    return sequenceBlocks;
}

} // namespace

namespace tensorrt_llm::batch_manager::kv_cache_manager
{
std::vector<MmKey> generateBlockHashExtraKeys(
    tensorrt_llm::batch_manager::LlmRequest const& llmRequest, SizeType32 startTokenIdx, SizeType32 endTokenIdx)
{
    auto const multimodalHashes = llmRequest.getMultimodalHashes();
    auto const multimodalPositions = llmRequest.getMultimodalPositions();
    auto const multimodalLengths = llmRequest.getMultimodalLengths();

    if (!multimodalHashes || !multimodalPositions || !multimodalLengths || !(*multimodalHashes)
        || (*multimodalHashes)->empty() || !(*multimodalPositions) || (*multimodalPositions)->empty()
        || !(*multimodalLengths) || (*multimodalLengths)->empty())
    {
        return {};
    }

    if ((*multimodalHashes)->size() != (*multimodalPositions)->size()
        || (*multimodalPositions)->size() != (*multimodalLengths)->size())
    {
        TLLM_LOG_WARNING("Multimodal data arrays have mismatched sizes");
        return {};
    }

    std::vector<MmKey> extraKeys; // MmKey = std::pair<std::array<uint8_t, 32>, SizeType32>
    extraKeys.reserve((*multimodalPositions)->size());
    std::array<uint8_t, 32> mmHashArray;

    for (size_t i = 0; i < (*multimodalPositions)->size(); ++i)
    {
        auto const& startPos = (*(*multimodalPositions))[i];
        auto const& length = (*(*multimodalLengths))[i];
        auto const& mmHashVector = (*(*multimodalHashes))[i];

        TLLM_CHECK_WITH_INFO(mmHashVector.size() == 8, "Multimodal hash vector has unexpected size: %zu (expected 8)",
            mmHashVector.size());

        // mmHashVector[j] comes from Python's int(hex_chunk, 16)
        // where hex_chunk like "00010203" means 0x00 is MSB and 0x03 is LSB (big endian)
        // Convert 8x 32-bit integers into a 32-byte array preserving Blake3 hash byte order
        // Example: hashPart = 0x00010203 → mmHashArray[0:3] = [0x00, 0x01, 0x02, 0x03]
        for (size_t j = 0; j < 8; ++j)
        {
            auto const& hashPart = mmHashVector[j];
            for (uint8_t byteIdx = 0; byteIdx < 4; ++byteIdx)
            {
                mmHashArray[j * 4 + byteIdx] = getNthByte(hashPart, byteIdx);
            }
        }

        // Check if this multimodal content overlaps with the current block
        if (endTokenIdx > startPos && startTokenIdx < startPos + length)
        {
            uint64_t mmStartInBlock = (startPos >= startTokenIdx) ? 0 : static_cast<uint64_t>(startTokenIdx - startPos);
            extraKeys.emplace_back(mmHashArray, mmStartInBlock);
        }
    }

    return extraKeys;
}

std::vector<BlockKey> buildBlockKeys(
    std::list<VecUniqueTokens>& blockedUniqueTokens, tensorrt_llm::batch_manager::LlmRequest const& llmRequest)
{
    std::vector<BlockKey> blockKeys;

    SizeType32 currentTokenIdx = 0;
    for (auto& uniqueTokens : blockedUniqueTokens)
    {
        auto extraKeys = generateBlockHashExtraKeys(llmRequest, currentTokenIdx, currentTokenIdx + uniqueTokens.size());
        currentTokenIdx += uniqueTokens.size();

        blockKeys.emplace_back(llmRequest.getInputTokensExtraIds().has_value(), llmRequest.getLoraTaskId(),
            std::move(uniqueTokens), std::move(extraKeys), llmRequest.getCacheSaltID());
    }
    return blockKeys;
}

bool BlockKey::operator==(BlockKey const& other) const noexcept
{
    return (usesExtraIds == other.usesExtraIds && loraTaskId == other.loraTaskId && uniqueTokens == other.uniqueTokens
        && extraKeys == other.extraKeys && cacheSaltID == other.cacheSaltID);
}

size_t BlockKeyHasher::hash(BlockKey const& blockKey, std::size_t parentHash) noexcept
{
    // Hashing algorithm adapted from StackOverflow:
    // https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
    // Constants provide very good distribution - each input bit affects each output bit with ~50% probability.
    size_t seed = blockKey.uniqueTokens.size() ^ parentHash * UINT64_C(0xbf58476d1ce4e5b9);

    if (parentHash == 0 && blockKey.cacheSaltID)
    {
        // Only hashing the cache salt ID for the first block in the sequence
        uint64_t c = blockKey.cacheSaltID.value();
        c = (c ^ (c >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
        c = (c ^ (c >> 27)) * UINT64_C(0x94d049bb133111eb);
        c = c ^ (c >> 31);
        seed ^= c + 0x9e3779b9 + (seed << 6) + (seed >> 2);
    }

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

    // Add extra keys for multimodal data mixing in external multimodal item hash and token offset within this sequence
    // block
    if (!blockKey.extraKeys.empty())
    {
        for (auto const& [mmHash, startOffset] : blockKey.extraKeys)
        {
            // Hash the multimodal hash array in 32-bit chunks (more efficient)
            for (size_t i = 0; i < 32; i += 4)
            {
                // Combine 4 bytes into a 32-bit word (construct as little endian order)
                uint32_t word = static_cast<uint32_t>(mmHash[i]) | (static_cast<uint32_t>(mmHash[i + 1]) << 8)
                    | (static_cast<uint32_t>(mmHash[i + 2]) << 16) | (static_cast<uint32_t>(mmHash[i + 3]) << 24);

                // Mix the word into the seed
                word = ((word >> 16) ^ word) * 0x45d9f3b;
                word = ((word >> 16) ^ word) * 0x45d9f3b;
                word = (word >> 16) ^ word;
                seed ^= word + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            }

            // Hash the start offset
            uint64_t e = static_cast<uint64_t>(startOffset);
            e = (e ^ (e >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
            e = (e ^ (e >> 27)) * UINT64_C(0x94d049bb133111eb);
            e = e ^ (e >> 31);
            seed ^= e + 0x9e3779b9 + (seed << 6) + (seed >> 2);
        }
    }

    return seed;
}

KVCacheBlock::KVCacheBlock(IdType blockId, tk::KVCacheIndex blockIdx)
    : mBlockId(blockId)
    , mMemoryPoolBlockIndex{blockIdx}
    , mRefCount(0)
    , mSchedulingRefCount(0)
    , mPrevBlock(nullptr)
    , mFreeBlockIterator(std::nullopt)
    , mIsFull{false}
    , mPriority{executor::KvCacheRetentionConfig::kDefaultRetentionPriority}
    , mDurationMs{std::nullopt}
    , mExpirationTime{std::nullopt}
    , mHash{0}
{
}

void KVCacheBlock::startScheduling()
{
    mSchedulingRefCount = mRefCount;
}

KVCacheBlock::IdType KVCacheBlock::getBlockId() const
{
    return mBlockId;
}

NextBlockMap KVCacheBlock::getNextBlocks() const
{
    return mNextBlocks;
}

tk::KVCacheIndex::UnderlyingType KVCacheBlock::getMemoryPoolBlockIndex() const
{
    return mMemoryPoolBlockIndex.get();
}

bool KVCacheBlock::isPrimary() const
{
    return mMemoryPoolBlockIndex.isPrimary();
}

void KVCacheBlock::swapMemoryPoolBlockOffset(std::shared_ptr<KVCacheBlock> otherBlock)
{
    std::swap(mMemoryPoolBlockIndex, otherBlock->mMemoryPoolBlockIndex);
}

void KVCacheBlock::incRefCount()
{
    mRefCount++;
}

void KVCacheBlock::decRefCount()
{
    TLLM_CHECK_WITH_INFO(
        hasRefs(), "Can't remove link from block (id=%d) that is not allocated", static_cast<int>(mBlockId));
    mRefCount--;
}

void KVCacheBlock::decSchedulingRefCount()
{
    TLLM_CHECK_WITH_INFO(hasSchedulingRefs(), "Can't remove link from block that is not allocated");
    mSchedulingRefCount--;
}

bool KVCacheBlock::hasRefs() const
{
    return mRefCount > 0;
}

bool KVCacheBlock::isShared() const
{
    // block is considered shared if ready for reuse
    return mRefCount > 1 || mPrevBlock != nullptr;
}

bool KVCacheBlock::hasSchedulingRefs() const
{
    return mSchedulingRefCount > 0;
}

void KVCacheBlock::setBlockKey(BlockKey const& blockKey, bool isFull)
{
    mBlockKey = blockKey;
    mIsFull = isFull;
}

BlockKey KVCacheBlock::getBlockKey()
{
    return mBlockKey;
}

void KVCacheBlock::setPriority(executor::RetentionPriority priority)
{
    mPriority = priority;
}

executor::RetentionPriority KVCacheBlock::getPriority() const
{
    return mPriority;
}

std::optional<std::chrono::milliseconds> KVCacheBlock::getDurationMs() const
{
    return mDurationMs;
}

void KVCacheBlock::setDurationMs(std::optional<std::chrono::milliseconds> durationMs)
{
    mDurationMs = durationMs;
}

void KVCacheBlock::setExpirationTime(std::optional<std::chrono::steady_clock::time_point::duration> expirationTime)
{
    mExpirationTime = expirationTime;
}

std::optional<std::chrono::steady_clock::time_point::duration> KVCacheBlock::getExpirationTime() const
{
    return mExpirationTime;
}

void KVCacheBlock::setHash(size_t hash)
{
    mHash = hash;
}

void KVCacheBlock::setHash()
{
    mHash = BlockKeyHasher()(mBlockKey, mPrevBlockInSeq ? mPrevBlockInSeq->getHash() : 0);
}

size_t KVCacheBlock::getHash() const
{
    return mHash;
}

VecUniqueTokens const& KVCacheBlock::getUniqueTokens() const
{
    return mBlockKey.uniqueTokens;
}

BlockPtr const& KVCacheBlock::getPrevBlock() const
{
    return mPrevBlock;
}

void KVCacheBlock::setPrevBlock(BlockPtr prevBlock)
{
    mPrevBlock = std::move(prevBlock);
}

BlockPtr const& KVCacheBlock::getPrevBlockInSeq() const
{
    return mPrevBlockInSeq;
}

void KVCacheBlock::setPrevBlockInSeq(BlockPtr prevBlock)
{
    mPrevBlockInSeq = std::move(prevBlock);
}

void KVCacheBlock::addNextBlock(BlockKey const& blockKey, BlockPtr block)
{
    if (mNextBlocks.find(blockKey) == mNextBlocks.end())
    {
        mNextBlocks[blockKey] = std::move(block);
    }
}

std::tuple<bool, SizeType32, BlockPtr> KVCacheBlock::findMatchingBlock(
    BlockKey const& blockKey, bool enablePartialReuse, bool copyOnPartialReuse) const
{
    if (blockKey.uniqueTokens.size() == 0 || mNextBlocks.size() == 0)
    {
        return {false, 0, nullptr};
    }
    auto itr = mNextBlocks.find(blockKey);
    if (itr == mNextBlocks.end())
    {
        if (enablePartialReuse)
        {
            SizeType32 bestNumMatched{0};
            BlockPtr bestBlock{nullptr};
            for (auto const& [key, block] : mNextBlocks)
            {
                if (copyOnPartialReuse || (!block->hasRefs() && block->isLeaf()))
                {
                    SizeType32 numMatched = key.partialMatch(blockKey);
                    if (numMatched > bestNumMatched)
                    {
                        bestNumMatched = numMatched;
                        bestBlock = block;
                    }
                }
            }
            if (bestNumMatched > 0)
            {
                return {true, bestNumMatched, bestBlock};
            }
        }
        return {false, 0, nullptr};
    }
    auto block = itr->second;
    return {!block->isFull(), static_cast<SizeType32>(blockKey.uniqueTokens.size()), block};
}

void KVCacheBlock::freeLeafBlock()
{
    // assure that this is a leaf block
    TLLM_CHECK(isLeaf());

    // free from previous block
    if (mPrevBlock != nullptr)
    {
        mPrevBlock->removeNextBlock(mBlockKey);
        mPrevBlock = nullptr;
    }
}

void KVCacheBlock::removeNextBlock(BlockKey const& blockKey)
{
    mNextBlocks.erase(blockKey);
}

bool KVCacheBlock::isFull() const
{
    return mIsFull;
}

bool KVCacheBlock::isLeaf() const
{
    return mNextBlocks.empty();
}

// This function calculates the number of block a layer should have, given
// the total free memory and the window size of each layer.
// For example, if we have 1 layer of window size 1024, and 2 layer of window
// size 2048, and 3 layers of 4096.
// Each layer of window size 1024 should have
//     1024 / (1024 + 2048 * 2 + 4096 * 3) proportion of the total blocks.
// Each layer of window size 2048 should have
//     2048 / (1024 + 2048 * 2 + 4096 * 3) proportion of the total blocks.
// Each layer of window size 4096 should have
//     4096 / (1024 + 2048 * 2 + 4096 * 3) proportion of the total blocks.
// NOTE: Currently the use of this function is not used for
// BaseKVCacheManager::calculateMaxNumBlocks because the we want to first
// achieve identical performance as assuming all layers as full attention.
std::map<SizeType32, float> BlockManager::calculateWindowSizeToShare(
    std::map<SizeType32, std::vector<SizeType32>> const& windowSizeToLayers,
    std::map<SizeType32, SizeType32> const& windowSizeToCacheSizePerToken)
{
    if (windowSizeToLayers.size() == 1)
    {
        return {{windowSizeToLayers.begin()->first, 1.0f}};
    }

    std::map<SizeType32, float> windowSizeToContribution;

    SizeType32 cacheSizePerTokenTotal
        = std::accumulate(windowSizeToCacheSizePerToken.begin(), windowSizeToCacheSizePerToken.end(), SizeType32{0},
            [](auto sum, auto const& windowSize) { return sum + windowSize.second; });
    for (auto const& [windowSize, cacheSizePerToken] : windowSizeToCacheSizePerToken)
    {
        auto const cacheSizeWeight = static_cast<float>(cacheSizePerToken) / cacheSizePerTokenTotal;
        windowSizeToContribution[windowSize] = cacheSizeWeight;
    }

    for (auto const& [windowSize, _] : windowSizeToLayers)
    {
        windowSizeToContribution.at(windowSize) *= windowSize;
    }
    auto const windowSizesTotalSum = std::accumulate(windowSizeToContribution.begin(), windowSizeToContribution.end(),
        0.0, [](auto sum, auto const& windowSize) { return sum + windowSize.second; });

    std::map<SizeType32, float> windowSizeToShare;
    for (auto const& [windowSize, windowSizeSum] : windowSizeToContribution)
    {
        float const fraction = windowSizeSum / windowSizesTotalSum;
        TLLM_CHECK(0.0f < fraction && fraction <= 1.0f);
        windowSizeToShare[windowSize] = fraction;
    }
    auto total = std::accumulate(windowSizeToShare.begin(), windowSizeToShare.end(), 0.0f,
        [](auto sum, auto const& windowSize) { return sum + windowSize.second; });
    TLLM_CHECK(total == 1.0f);
    return windowSizeToShare;
}

BlockManager::BlockManager(std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead,
    SizeType32 tokensPerBlock, BlocksPerWindow const& blocksPerWindow, SizeType32 maxNumSequences,
    std::shared_ptr<runtime::CudaStream> stream, SizeType32 maxSequenceLength, SizeType32 maxBeamWidth,
    std::vector<SizeType32> const& maxAttentionWindowVec,
    std::optional<TempAttentionWindowInputs> const& tempAttentionWindowInputs, nvinfer1::DataType dtype,
    SizeType32 sinkBubbleLength, bool onboardBlocks, CacheType cacheType,
    std::optional<executor::RetentionPriority> secondaryOffloadMinPriority,
    std::shared_ptr<KVCacheEventManager> eventManager, bool enablePartialReuse, bool copyOnPartialReuse,
    std::shared_ptr<kv_connector::KvCacheConnectorManager> kvCacheConnectorManager,
    std::optional<BaseAgentConfig> agentConfig)
    : mNumLayers{static_cast<SizeType32>(numKvHeadsPerLayer.size())}
    , mTokensPerBlock{tokensPerBlock}
    , mEventManager{std::move(eventManager)}
    , mStream{stream}
    , mCacheType{cacheType}
{
    if (agentConfig.has_value())
        mLoopbackAgent = makeLoopbackAgent("nixl", &agentConfig.value());
    else
        mLoopbackAgent = nullptr;

    auto const uniqueWindowSizeToLayers
        = BaseKVCacheManager::groupLayersByWindowSize(maxAttentionWindowVec, mNumLayers);

    TLLM_CHECK_WITH_INFO(kvCacheConnectorManager == nullptr || uniqueWindowSizeToLayers.size() == 1,
        "KV Cache Connector is not supported with multiple window sizes");

    auto const numUniqueWindowSizes = static_cast<SizeType32>(uniqueWindowSizeToLayers.size());

    mIsVariableWindow = numUniqueWindowSizes > 1;
    mIsVariableGQA = std::unordered_set(numKvHeadsPerLayer.begin(), numKvHeadsPerLayer.end()).size() > 1;

    mLayerToWindowSize.resize(mNumLayers);
    for (auto const& [windowSize, layersWithWindowSize] : uniqueWindowSizeToLayers)
    {
        if (windowSize > maxSequenceLength)
        {
            TLLM_LOG_WARNING("[kv cache manager] window size %d is greater than max sequence length %d", windowSize,
                maxSequenceLength);
        }
        for (auto& layerIdx : layersWithWindowSize)
        {
            mLayerToWindowSize.at(layerIdx) = windowSize;
        }
        auto const [allottedPrimaryBlocks, allottedSecondaryBlocks] = blocksPerWindow.at(windowSize);
        TLLM_CHECK(allottedPrimaryBlocks > 0); // You can't have a model with negative primary blocks...
        mWindowBlockManagers.try_emplace(windowSize, dtype, windowSize, layersWithWindowSize, numKvHeadsPerLayer,
            sizePerHead, tokensPerBlock, /*isSWA=*/windowSize < maxSequenceLength, allottedPrimaryBlocks,
            allottedSecondaryBlocks, maxNumSequences, stream, onboardBlocks, cacheType, secondaryOffloadMinPriority,
            mEventManager, enablePartialReuse, copyOnPartialReuse, kvCacheConnectorManager, mLoopbackAgent);
    }

    auto const numAllPools = getNumPools();
    mAbsolutePoolToWindowSize.reserve(numAllPools);
    mAbsolutePoolToRelativePoolIndex.reserve(numAllPools);
    auto absolutePoolsOffset = SizeType32{0};
    for (auto const& [windowSize, manager] : mWindowBlockManagers)
    {
        auto const numPools = manager.getNumPools();
        for (auto i = 0; i < numPools; ++i)
        {
            mAbsolutePoolToWindowSize.push_back(windowSize);
            mAbsolutePoolToRelativePoolIndex.push_back(i);
        }
        // SWA allocates blocks linearly, and we need as many blocks as full attention,
        // where full attention has windowSize = maxSequenceLength.
        auto const maxTokenNum = std::max(windowSize, maxSequenceLength) + sinkBubbleLength;
        auto const temporaryAttentionWindow = manager.calculateTemporaryAttentionWindow(tempAttentionWindowInputs);
        // Consider the temporaryAttentionWindow when allocating blocks.
        // Current tempAttentionWindow calculation does not consider the
        // concept of SWA right now at most occupying maxSequenceLength of
        // blocks. So the calculation of maxToken + tempAttention will exceed
        // maxSequenceLength. A temporary resolution here is to cap the
        // calculation to maxSequenceLength. I will proceed with a follow-up
        // MR to remove the tempAttentionWindow concept.
        auto const maxBlocksPerSeq
            = tc::ceilDiv(std::min(maxSequenceLength, maxTokenNum + temporaryAttentionWindow), tokensPerBlock);
        auto const [allottedPrimaryBlocks, allottedSecondaryBlocks] = blocksPerWindow.at(windowSize);
        mWindowSizeToMetadata[windowSize] = WindowSizeMetadata{allottedPrimaryBlocks, allottedSecondaryBlocks,
            absolutePoolsOffset, numPools, maxTokenNum, maxBlocksPerSeq, manager.getMaxNumBlocks(),
            temporaryAttentionWindow, windowSize, manager.isSWA()};
        TLLM_LOG_INFO(
            "Max KV cache blocks per sequence: %d [window size=%d], tokens per block=%d, primary blocks=%d, secondary "
            "blocks=%d, max sequence length=%d",
            maxBlocksPerSeq, windowSize, tokensPerBlock, allottedPrimaryBlocks, allottedSecondaryBlocks,
            maxSequenceLength);
        TLLM_LOG_DEBUG(
            "%s Metadata: %s", manager.getLogPrefix().c_str(), mWindowSizeToMetadata[windowSize].toString().c_str());
        absolutePoolsOffset += numPools;
    }

    TLLM_CHECK_WITH_INFO(mWindowBlockManagers.size() == mWindowSizeToMetadata.size()
            && std::equal(mWindowBlockManagers.cbegin(), mWindowBlockManagers.cend(), mWindowSizeToMetadata.cbegin(),
                mWindowSizeToMetadata.cend(),
                [](auto const& window1, auto const& window2) { return window1.first == window2.first; }),
        "Iteration order of window sizes between mWindowBlockManagers and mWindowSizeToMetadata *must* be ensured. "
        "Maybe you tried changing either of them to an std::unordered_map?");
}

WindowBlockManager::WindowBlockManager(nvinfer1::DataType dtype, SizeType32 windowSize,
    std::vector<SizeType32> const& managedLayers, std::vector<SizeType32> const& numKvHeadsPerLayer,
    SizeType32 sizePerHead, SizeType32 tokensPerBlock, bool isSWA, SizeType32 blocksInPrimaryPool,
    SizeType32 blocksInSecondaryPool, SizeType32 maxNumSequences, std::shared_ptr<runtime::CudaStream> stream,
    bool onboardBlocks, CacheType cacheType, std::optional<executor::RetentionPriority> secondaryOffloadMinPriority,
    std::shared_ptr<KVCacheEventManager> eventManager, bool enablePartialReuse, bool copyOnPartialReuse,
    std::shared_ptr<kv_connector::KvCacheConnectorManager> kvCacheConnectorManager,
    std::shared_ptr<kvc::BaseLoopbackAgent> loopbackAgent)
    : mDataType{dtype}
    , mWindowSize{windowSize}
    , mNumPrimaryBlocks{blocksInPrimaryPool}
    , mNumSecondaryBlocks{blocksInSecondaryPool}
    , mOnboardBlocks(onboardBlocks)
    , mBufferManager{std::move(stream)}
    , mSchedulingNumFreeBlocks{0}
    , mTokensPerBlock{tokensPerBlock}
    , mIsSWA{isSWA}
    , mCachedBlocksRoot{std::make_shared<KVCacheBlock>(KVCacheBlock::kCachedBlocksRootId, tk::KVCacheIndex{0})}
    , mCacheType{cacheType}
    , mEventManager(std::move(eventManager))
    , mLoopbackAgent{loopbackAgent}
    , mTransferManager{std::make_shared<KVCacheTransferManager>(mBufferManager, mLoopbackAgent)}
    , mAllocTotalBlocks{0}
    , mAllocNewBlocks{0}
    , mReusedBlocks{0}
    , mReusedUniqueBlocks{0}
    , mMissedBlocks{0}
    , mKVFactor{mCacheType == CacheType::kSELFKONLY ? 1 : 2}
    , mLogPrefix{tensorrt_llm::common::fmtstr("BlockManager[windowSize=%u]", mWindowSize)}
    , mReusedTokens{0.0}
    , mTotalInputTokens{0.0}
    , mEnablePartialReuse{enablePartialReuse}
    , mCopyOnPartialReuse{copyOnPartialReuse}
    , mKvCacheConnectorManager{std::move(kvCacheConnectorManager)}
{
    std::map<SizeType32, SizeType32> numLayersPerPool;

    for (auto const layerIdx : managedLayers)
    {
        auto const& layerIndexWithinPool = numLayersPerPool[numKvHeadsPerLayer.at(layerIdx)]++;
        mLayerToIndexWithinPool[layerIdx] = layerIndexWithinPool;
    }

    auto numEltsPerContainer = getNumEltsPerContainer();
#ifdef ENABLE_FP4
    if (numEltsPerContainer == 2)
    {
        TLLM_CHECK_WITH_INFO(sizePerHead % 2 == 0, "sizePerHead must be divisible by 2 for 4-bit KV cache.");
    }
#endif

    size_t poolIndex = 0;
    for (auto const [numKvHeads, numLayers] : numLayersPerPool)
    {
        for (auto const layerIdx : managedLayers)
        {
            if (numKvHeadsPerLayer.at(layerIdx) == numKvHeads)
            {
                mLayerToPoolIndex[layerIdx] = poolIndex;
            }
        }
        mPools.emplace_back(numLayers, mKVFactor, numKvHeads, sizePerHead / numEltsPerContainer, tokensPerBlock);
        ++poolIndex;
    }

#ifdef ENABLE_FP4
    // TODO(miovine): make the block size configurable. Should we have an additional argument
    // to specify FP4 related parameters (scale dtypes, etc)? This can also be passed
    // in the constructor.
    constexpr SizeType32 kQuantBlockSizeNVFP4 = 16;
    if (dtype == nvinfer1::DataType::kFP4)
    {
        createBlockScalePools(kQuantBlockSizeNVFP4);
    }
#endif

    // Create free blocks
    mAllBlocksById.reserve(blocksInPrimaryPool + blocksInSecondaryPool);
    for (KVCacheBlock::IdType blockId = 0; blockId < blocksInPrimaryPool; ++blockId)
    {
        mAllBlocksById.emplace_back(std::make_shared<KVCacheBlock>(blockId, tk::KVCacheIndex{blockId, false}));
    }
    for (KVCacheBlock::IdType blockId = 0; blockId < blocksInSecondaryPool; ++blockId)
    {
        mAllBlocksById.emplace_back(
            std::make_shared<KVCacheBlock>(blocksInPrimaryPool + blockId, tk::KVCacheIndex{blockId, true}));
    }
    mAllocatedBlocksPerSeq.reserve(maxNumSequences);

    mEvictionPolicy = std::make_shared<LRUEvictionPolicy>();
    mEvictionPolicy->initialize(
        mAllBlocksById, {blocksInPrimaryPool, blocksInSecondaryPool}, secondaryOffloadMinPriority);
    if (mEventManager)
    {
        mEventManager->enqueueCreatedEvent({blocksInPrimaryPool, blocksInSecondaryPool}, mWindowSize);
    }
}

WindowBlockManager::~WindowBlockManager()
{
    float reusedUniqueBlocksPercentage = mReusedUniqueBlocks == 0 || mAllocTotalBlocks == 0
        ? 0
        : static_cast<float>(mReusedUniqueBlocks) / static_cast<float>(mAllocNewBlocks) * 100;
    float cacheHitRate = mReusedBlocks == 0
        ? 0
        : static_cast<float>(mReusedBlocks) / (static_cast<float>(mReusedBlocks + mMissedBlocks));
    TLLM_LOG_DEBUG("%s - total allocated blocks:              %lu  ", mLogPrefix.c_str(), mAllocTotalBlocks);
    TLLM_LOG_DEBUG("%s - allocated new blocks:                %lu  ", mLogPrefix.c_str(), mAllocNewBlocks);
    TLLM_LOG_DEBUG("%s - missed blocks:                       %lu  ", mLogPrefix.c_str(), mMissedBlocks);
    TLLM_LOG_DEBUG("%s - reused blocks:                       %lu  ", mLogPrefix.c_str(), mReusedBlocks);
    TLLM_LOG_DEBUG("%s - reused unique blocks:                %lu  ", mLogPrefix.c_str(), mReusedUniqueBlocks);
    TLLM_LOG_DEBUG(
        "%s - reused unique blocks percentage (%%): %.2f ", mLogPrefix.c_str(), reusedUniqueBlocksPercentage);
    TLLM_LOG_DEBUG("%s - cache hit rate:                      %.2f ", mLogPrefix.c_str(), cacheHitRate);
    TLLM_LOG_DEBUG("%s - reused tokens:                       %.0f ", mLogPrefix.c_str(), mReusedTokens);
    TLLM_LOG_DEBUG("%s - reused tokens percentage (%%):        %.2f ", mLogPrefix.c_str(),
        100.0 * mReusedTokens / mTotalInputTokens);
}

bool BlockManager::verifyQueueIntegrity(SizeType32 windowSize)
{
    return mWindowBlockManagers.at(windowSize).verifyQueueIntegrity();
}

bool WindowBlockManager::verifyQueueIntegrity()
{
    return mEvictionPolicy->verifyQueueIntegrity();
}

void BlockManager::storeContextBlocks(GenerationRequest& sequence, LlmRequest const& llmRequest)
{
    constexpr int beamIdx = 0; // no need to consider more than one beam for input tokens
    for (auto const& [windowSize, _] : mWindowBlockManagers)
    {
        if (mWindowBlockManagers.at(windowSize).isSWA())
        {
            // SWA cannot store new blocks on the fly because the block stored
            // may go OOW and be reused by another sequence.
            continue;
        }
        auto cacheBlockIds = sequence.getCacheBlockIds(windowSize);
        auto const& uniqueTokens = llmRequest.getUniqueTokens(beamIdx);

        auto blockedUniqueTokens
            = chopVectorIntoBlocks<UniqueToken>(uniqueTokens, uniqueTokens.size() - 1, getTokensPerBlock(), false);
        auto blockKeys = buildBlockKeys(blockedUniqueTokens, llmRequest);
        (void) mWindowBlockManagers.at(windowSize).storeBlocks(std::move(blockKeys), cacheBlockIds[beamIdx]);
    }
}

void WindowBlockManager::createBlockScalePools(SizeType32 quantBlockSize)
{
    SizeType32 const numEltsPerContainer = getNumEltsPerContainer();
    auto num_pools = mPools.size();
    for (size_t i = 0; i < num_pools; ++i)
    {
        auto& kv_pool = mPools[i];
        TLLM_CHECK_WITH_INFO((kv_pool.sizePerHead * numEltsPerContainer) % quantBlockSize == 0,
            "Cannot use FP4 quantization since kv_pool.sizePerHead is not divisible by FP4 quantBlockSize.");
        auto blockScaleSizePerHead = kv_pool.sizePerHead * numEltsPerContainer / quantBlockSize;
        mPools.emplace_back(kv_pool.numLayers, kv_pool.kvFactor, kv_pool.numKvHeads, blockScaleSizePerHead,
            kv_pool.tokensPerBlock,
            /*primaryPool=*/nullptr,
            /*secondaryPool=*/nullptr,
            /*containsBlockScales=*/true);
    }
}

void BlockManager::allocatePools(bool useUvm)
{
    for (auto& [_, manager] : mWindowBlockManagers)
    {
        manager.allocatePools(useUvm);
    }
}

void WindowBlockManager::allocatePools(bool useUvm)
{
    constexpr nvinfer1::DataType kScaleDtypeNVFP4 = nvinfer1::DataType::kFP8;

    // Allocate a memory pool backing the blocks for each numKvHeads
    // TODO(oargov): allocate pools in a single buffer and split it, to avoid fragmentation
    for (auto& pool : mPools)
    {
        auto blockSize = pool.blockSize;
        auto poolDtype = pool.containsBlockScales ? kScaleDtypeNVFP4 : mDataType;

#ifdef ENABLE_FP4
        auto const poolIsFP4 = poolDtype == nvinfer1::DataType::kFP4;
#else
        auto const poolIsFP4 = false;
#endif

        if (poolIsFP4)
        {
            poolDtype = nvinfer1::DataType::kINT8;
        }

        nvinfer1::Dims const cacheShape = ITensor::makeShape({mNumPrimaryBlocks, pool.numLayers, mKVFactor, blockSize});

        TLLM_LOG_DEBUG("[%s] Allocating primary pool with %d blocks for %d layers with %d kv heads", mLogPrefix.c_str(),
            mNumPrimaryBlocks, pool.numLayers, pool.numKvHeads);

        if (useUvm)
            pool.primaryPtr = BufferManager::managed(cacheShape, poolDtype);
        else
            pool.primaryPtr = mBufferManager.gpuSync(cacheShape, poolDtype);

        if (mNumSecondaryBlocks > 0)
        {
            nvinfer1::Dims const cacheShapeOffload
                = ITensor::makeShape({mNumSecondaryBlocks, pool.numLayers, mKVFactor, blockSize});
            TLLM_LOG_DEBUG("[%s] Allocating secondary pool with %d blocks for %d layers with %d kv heads",
                mLogPrefix.c_str(), mNumSecondaryBlocks, pool.numLayers, pool.numKvHeads);
            pool.secondaryPtr = BufferManager::pinned(cacheShapeOffload, poolDtype);
        }
    }
}

void BlockManager::releasePools()
{
    for (auto& [_, manager] : mWindowBlockManagers)
    {
        manager.releasePools();
    }
}

void WindowBlockManager::releasePools()
{
    for (auto& pool : mPools)
    {
        if (pool.primaryPtr)
        {
            pool.primaryPtr->release();
        }
        if (pool.secondaryPtr)
        {
            pool.secondaryPtr->release();
        }
    }
    mBufferManager.getStream().synchronize();
    mBufferManager.memoryPoolTrimTo(0);
}

void BlockManager::startScheduling()
{
    for (auto& [_, manager] : mWindowBlockManagers)
    {
        manager.startScheduling();
    }
}

void WindowBlockManager::startScheduling()
{
    mSchedulingNumFreeBlocks = mEvictionPolicy->getNumFreeBlocks(kPrimaryLevel);
    for (auto& [requestId, slotAllocatedBlocks] : mAllocatedBlocksPerSeq)
    {
        for (auto& allocatedBlock : slotAllocatedBlocks)
        {
            allocatedBlock->startScheduling();
        }
    }
}

void WindowBlockManager::freeLeafBlock(BlockPtr const& block)
{
    // The eviction policy needs blocks to still be linked to their old parents when they're reclaimed.
    // This is so it can check if the parent should be queued for eviction.
    block->freeLeafBlock();
}

void WindowBlockManager::freeChildren(BlockPtr const& block)
{
    // Free all descendants of block
    for (auto const& p : block->getNextBlocks())
    {
        auto childBlock = p.second;
        freeChildren(childBlock);
    }

    // Free block
    if (mEventManager && blockInRadixTree(block))
    {
        mEventManager->enqueueRemovedEvent(block, mWindowSize);
    }
    freeLeafBlock(block);
}

BlockPtr WindowBlockManager::getFreeBlock(GenerationRequest& sequence, executor::RetentionPriority priority,
    std::optional<std::chrono::milliseconds> durationMs, executor::KvCacheTransferMode mode,
    std::string const& directory)
{
    // eviction policy get free primary block
    auto [block, canOffload] = mEvictionPolicy->getFreeBlock(kPrimaryLevel);
    if (block->getUniqueTokens().empty())
    {
        ++mAllocNewBlocks;
    }
    ++mAllocTotalBlocks;
    // Offloading is an option only when these conditions are met:
    // 1. Block contains state (evidenced by presence of tokens)
    // 2. Eviction policy indicated block can be offloaded
    // 3. At least one free block in secondary memory
    // 4. Onboarding is enabled (allowing block to be brought back into primary)
    if (!block->getUniqueTokens().empty() && canOffload && mEvictionPolicy->getNumFreeBlocks(kSecondaryLevel) > 0
        && mOnboardBlocks)
    {
        // Offload block in primary memory before repurposing
        auto offloadBlock = std::get<0>(mEvictionPolicy->getFreeBlock(kSecondaryLevel));
        mTransferManager->offload(block, offloadBlock, mPools, 0, mode, directory);
        // swap linear block offsets (i.e. make block the offload block)
        block->swapMemoryPoolBlockOffset(offloadBlock);

        if (mEventManager && blockInRadixTree(block))
        {
            mEventManager->enqueueUpdatedEvent(
                tle::KVCacheUpdatedData(block->getHash()).cacheLevelUpdated(kPrimaryLevel, kSecondaryLevel),
                mWindowSize);
        }
        // Update the block as a secondary block (maintaining its priority)
        mEvictionPolicy->claimBlock(block);
        // Release the block into secondary block queue
        mEvictionPolicy->releaseBlock(block);
        // We have the offloaded block as the block to use now.
        block = offloadBlock;
    }

    // Removes children of the block from the search tree
    freeChildren(block);
    // Claim the block in primary block queue
    mEvictionPolicy->claimBlock(block, priority, durationMs);

    // Deal with invalidating block save for reuse for the sequence
    if (mBlockToSequence.count(block->getBlockId()) > 0)
    {
        auto const& originalOwnerSequenceId = mBlockToSequence[block->getBlockId()];
        if (mIsValidStoreForReuseSequence.count(originalOwnerSequenceId) > 0
            && sequence.getRequestId() != originalOwnerSequenceId)
        {
            TLLM_LOG_DEBUG("%s::getFreeBlock - Block %d was originally held but released from sequence %d",
                mLogPrefix.c_str(), block->getBlockId(), originalOwnerSequenceId);
            if (mIsValidStoreForReuseSequence[originalOwnerSequenceId])
            {
                TLLM_LOG_DEBUG("%s::getFreeBlock - Invalidate store block for reuse for sequence %d",
                    mLogPrefix.c_str(), originalOwnerSequenceId);
            }
            else
            {
                TLLM_LOG_DEBUG("%s::getFreeBlock - Store block for reuse for sequence %d is already invalid",
                    mLogPrefix.c_str(), originalOwnerSequenceId);
            }
            mIsValidStoreForReuseSequence[originalOwnerSequenceId] = false;
        }
    }

    // Record which sequence is using the block
    mBlockToSequence[block->getBlockId()] = sequence.getRequestId();
    TLLM_LOG_DEBUG("%s::getFreeBlock - Block %d is now acquired by sequence %d", mLogPrefix.c_str(),
        block->getBlockId(), sequence.getRequestId());

    return block;
}

void WindowBlockManager::setOffsets(tk::KVCacheIndex* offsetsPtr, nvinfer1::Dims const& offsetsShape,
    SizeType32 beamIdx, SizeType32 blockIdx, KVCacheBlock::IdType blockId) const
{
    auto constexpr kIdx = 0;
    auto constexpr vIdx = 1;

    auto const& block = mAllBlocksById[blockId];
    for (SizeType32 poolIdx = 0; poolIdx < static_cast<SizeType32>(mPools.size()); poolIdx++)
    {
        auto const& pool = mPools.at(poolIdx);
        for (auto const xIdx : {kIdx, vIdx})
        {
            auto constexpr layerIdx = 0;
            auto const offsetIndex = tensorrt_llm::common::flat_index(offsetsShape.d, poolIdx, beamIdx, xIdx, blockIdx);
            auto const fieldIdx = mCacheType == CacheType::kSELFKONLY ? 0 : xIdx;
            auto const blockIndex = tk::KVCacheIndex{
                common::flat_index3(block->getMemoryPoolBlockIndex(), layerIdx, fieldIdx, pool.numLayers, mKVFactor)};
            offsetsPtr[offsetIndex] = blockIndex;
        }
    }
}

void BlockManager::setOffsets(tk::KVCacheIndex* offsetsPtr, nvinfer1::Dims const& offsetsShape, SizeType32 beamIdx,
    SizeType32 blockIdx, KVCacheBlock::IdType blockId, SizeType32 windowSize) const
{
    mWindowBlockManagers.at(windowSize).setOffsets(offsetsPtr, offsetsShape, beamIdx, blockIdx, blockId);
}

void BlockManager::onboardBlock(GenerationRequest& sequence, BlockPtr const& offloadBlock, SizeType32 windowSize,
    executor::KvCacheTransferMode mode, std::string const& directory)
{
    mWindowBlockManagers.at(windowSize).onboardBlock(sequence, offloadBlock, mode, directory);
}

void WindowBlockManager::onboardBlock(GenerationRequest& sequence, BlockPtr const& offloadBlock,
    executor::KvCacheTransferMode mode, std::string const& directory)
{
    if (mOnboardBlocks && !offloadBlock->isPrimary())
    {
        auto block = getFreeBlock(
            sequence, executor::KvCacheRetentionConfig::kDefaultRetentionPriority, std::nullopt, mode, directory);
        mTransferManager->onboard(offloadBlock, block, mPools, 0, mode, directory);
        // swap linear block offsets (i.e. make block the offload block and vice versa)
        offloadBlock->swapMemoryPoolBlockOffset(block);

        if (mEventManager)
        {
            mEventManager->enqueueUpdatedEvent(
                tle::KVCacheUpdatedData(offloadBlock->getHash()).cacheLevelUpdated(kSecondaryLevel, kPrimaryLevel),
                mWindowSize);
        }
        mEvictionPolicy->releaseBlock(block); // append block to offload queue
                                              // offloadBlock is now in primary memory pool
    }
}

void BlockManager::offloadBlock(
    BlockPtr const& block, SizeType32 windowSize, executor::KvCacheTransferMode mode, std::string const& directory)
{
    mWindowBlockManagers.at(windowSize).offloadBlock(block, mode, directory);
}

void WindowBlockManager::offloadBlock(
    BlockPtr const& block, executor::KvCacheTransferMode mode, std::string const& directory)
{
    // The current default behavior is to offload the out-of-window block
    // to secondary block pool to allow more free primary blocks for reuse.
    // However, such behavior does not take account whether the offloaded
    // block is useful or not and may just lead to more traffic instead.
    // The ideal way of this is to dedicate the offloading of the block
    // to the eviction policy.
    if (mOnboardBlocks && block->isPrimary())
    {
        // Offload block in primary memory before repurposing
        auto offloadBlock = std::get<0>(mEvictionPolicy->getFreeBlock(kSecondaryLevel));
        // If we're swapping a block to secondary memory, maintain the prior priority values.
        mEvictionPolicy->claimBlock(offloadBlock);
        mTransferManager->offload(block, offloadBlock, mPools, 0, mode, directory);
        // swap linear block offsets (i.e. make block the offload block)
        block->swapMemoryPoolBlockOffset(offloadBlock);

        if (mEventManager && blockInRadixTree(block))
        {
            mEventManager->enqueueUpdatedEvent(
                tle::KVCacheUpdatedData(block->getHash()).cacheLevelUpdated(kPrimaryLevel, kSecondaryLevel),
                mWindowSize);
        }
        mEvictionPolicy->releaseBlock(offloadBlock); // append offloadBlock to mFreePrimaryBlocks queue
                                                     // block is now in secondary memory
    }
}

[[nodiscard]] std::optional<BlockKey> BlockManager::findNewContextBlock(
    VecUniqueTokens const& uniqueTokens, LlmRequest const& llmRequest) const
{
    TLLM_CHECK_WITH_INFO(
        !isVariableWindow(), "The optimization of delaying requests won't work for variable window attention");
    auto const& onlyManager = mWindowBlockManagers.cbegin()->second;
    return onlyManager.findNewContextBlock(uniqueTokens, llmRequest);
}

std::optional<BlockKey> WindowBlockManager::findNewContextBlock(
    VecUniqueTokens const& uniqueTokens, LlmRequest const& llmRequest) const
{
    auto blockedUniqueTokens
        = chopVectorIntoBlocks<UniqueToken>(uniqueTokens, uniqueTokens.size(), mTokensPerBlock, false);
    auto blockKeys = buildBlockKeys(blockedUniqueTokens, llmRequest);
    BlockKey ret;
    ret.loraTaskId = llmRequest.getLoraTaskId();
    auto searchRoot = mCachedBlocksRoot;
    for (auto const& blockKey : blockKeys)
    {
        ret.uniqueTokens.insert(ret.uniqueTokens.end(), blockKey.uniqueTokens.begin(), blockKey.uniqueTokens.end());
        auto [partialMatch, numMatched, matchingBlock] = searchRoot != nullptr
            ? searchRoot->findMatchingBlock(blockKey, false, false)
            : std::make_tuple(false, 0, nullptr);
        if (matchingBlock == nullptr)
        {
            return ret;
        }
        searchRoot = std::move(matchingBlock);
    }
    return std::nullopt;
}

bool WindowBlockManager::blockInRadixTree(BlockPtr const& block)
{
    return !block->getUniqueTokens().empty() && block->getPrevBlock() != nullptr;
}

std::shared_ptr<KVCacheBlock> WindowBlockManager::findBlocksInReuseTreeByBlockKey(BlockKey const& blockKey)
{
    std::lock_guard<std::mutex> lock(mCachedBlocksRootMutex);
    auto blockedUniqueTokens
        = chopVectorIntoBlocks<UniqueToken>(blockKey.uniqueTokens, blockKey.uniqueTokens.size(), mTokensPerBlock, true);

    std::vector<BlockKey> blockKeys;
    for (auto const& blockedUniqueTokensList : blockedUniqueTokens)
    {
        blockKeys.push_back(blockKey);
        blockKeys.back().uniqueTokens = blockedUniqueTokensList;
    }
    auto searchRoot = mCachedBlocksRoot;
    for (auto const& blockKey : blockKeys)
    {
        auto [partialMatch, numMatched, matchingBlock] = searchRoot != nullptr
            ? searchRoot->findMatchingBlock(blockKey, true, true)
            : std::make_tuple(false, 0, nullptr);

        if (matchingBlock == nullptr)
        {
            return nullptr;
        }

        searchRoot = std::move(matchingBlock);
    }
    return searchRoot;
}

SizeType32 WindowBlockManager::loadOrAllocateBlocks(std::vector<BlockKey> const& blockKeys, SizeType32 numContextBlocks,
    GenerationRequest& sequence, std::vector<executor::RetentionPriorityAndDuration> const& perBlockRetentions,
    executor::KvCacheTransferMode mode, std::string const& directory)
{
    std::lock_guard<std::mutex> lock(mCachedBlocksRootMutex);
    SizeType32 numMatchedTokens{0};
    auto searchRoot = mCachedBlocksRoot;

    // The last block cannot be shared between beams because it will be written to.
    // Make sure a unique block is allocated per beam.
    auto const beamWidth = sequence.getBeamWidth();
    SizeType32 numSharedContextBlocks = beamWidth > 1 ? numContextBlocks - 1 : numContextBlocks;

    auto blockItr = blockKeys.begin();
    for (int bi = 0; bi < numSharedContextBlocks; ++bi)
    {
        auto [partialMatch, numMatched, matchingBlock] = searchRoot != nullptr && blockItr != blockKeys.end()
            ? searchRoot->findMatchingBlock(*blockItr, mEnablePartialReuse, mCopyOnPartialReuse)
            : std::make_tuple(false, 0, nullptr);
        if (matchingBlock != nullptr)
        {
            KVCacheBlock::IdType matchingBlockId = matchingBlock->getBlockId();

            numMatchedTokens += numMatched > 0 ? numMatched : blockItr->uniqueTokens.size();
            if (perBlockRetentions[bi].retentionPriority.has_value()
                && matchingBlock->getPriority() != perBlockRetentions[bi].retentionPriority && mEventManager)
            {
                mEventManager->enqueueUpdatedEvent(
                    tle::KVCacheUpdatedData(matchingBlock->getHash())
                        .priorityUpdated(matchingBlock->getPriority(), *perBlockRetentions[bi].retentionPriority),
                    mWindowSize);
            }
            if (partialMatch)
            {
                if (matchingBlock->hasRefs() || !matchingBlock->isLeaf())
                {
                    // Somebody else is using block or it is not a leaf, copy reusable tokens
                    auto newBlock = getFreeBlock(
                        sequence, matchingBlock->getPriority(), matchingBlock->getDurationMs(), mode, directory);
                    mTransferManager->onboard(matchingBlock, newBlock, mPools, numMatched, mode, directory);
                    // TODO: (optional) Send out event
                    matchingBlock = newBlock;
                    if (blockItr != blockKeys.end())
                    {
                        matchingBlock->setBlockKey(
                            *blockItr, blockItr->uniqueTokens.size() == static_cast<size_t>(mTokensPerBlock));
                    }
                    matchingBlock->setHash();
                    TLLM_LOG_DEBUG("%s::loadOrAllocateBlocks - Copied partially filled block %d", mLogPrefix.c_str(),
                        matchingBlockId);
                }
                else
                {
                    // Leaf block that nobody is using. Make block private and reuse
                    freeLeafBlock(matchingBlock);
                    mEvictionPolicy->claimBlock(
                        matchingBlock, perBlockRetentions[bi].retentionPriority, perBlockRetentions[bi].durationMs);
                    TLLM_LOG_DEBUG("%s::loadOrAllocateBlocks - Reused partially filled block %d", mLogPrefix.c_str(),
                        matchingBlockId);
                }
                searchRoot = nullptr; // no matching needed for following blocks
            }
            else
            {
                // Recover block and reuse
                mEvictionPolicy->claimBlock(
                    matchingBlock, perBlockRetentions[bi].retentionPriority, perBlockRetentions[bi].durationMs);
                TLLM_LOG_DEBUG("%s::loadOrAllocateBlocks - Matched full block %d", mLogPrefix.c_str(), matchingBlockId);
                searchRoot = matchingBlock;
            }
            onboardBlock(sequence, matchingBlock, mode, directory);
            addBlockToAllBeams(matchingBlock, sequence);
            // TODO: only add once for reused blocks
            ++mReusedBlocks;
            if (!reusedBlockIds.count(matchingBlockId))
            {
                reusedBlockIds.insert(matchingBlockId);
                ++mReusedUniqueBlocks;
            }
            ++blockItr;
        }
        else
        {
            // If we haven't set a priority, set it to the default priority level (low)
            auto freeBlock = getFreeBlock(sequence,
                perBlockRetentions[bi].retentionPriority.value_or(
                    executor::KvCacheRetentionConfig::kDefaultRetentionPriority),
                perBlockRetentions[bi].durationMs, mode, directory);
            addBlockToAllBeams(freeBlock, sequence);
            TLLM_LOG_DEBUG("%s::loadOrAllocateBlocks - No match, allocated new block %d for sequence %lu",
                mLogPrefix.c_str(), freeBlock->getBlockId(), sequence.getRequestId());
            searchRoot = nullptr; // no matching needed for following blocks
            if (blockItr != blockKeys.end())
            {
                freeBlock->setBlockKey(
                    *blockItr, blockItr->uniqueTokens.size() == static_cast<size_t>(mTokensPerBlock));
                ++blockItr;
            }
            freeBlock->setHash();
            ++mMissedBlocks;
        }
    }

    // Allocate new blocks that cannot be shared by multiple beams.
    for (int bi = numSharedContextBlocks; bi < numContextBlocks; ++bi)
    {
        // TODO: Still look for match. Clone matching block or allocate fresh ones.
        // This work is described in JIRA task https://jirasw.nvidia.com/browse/TRTLLM-2069.
        for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
        {
            // If we haven't set a priority, set it to the default priority level (low)
            auto freeBlock = getFreeBlock(sequence,
                perBlockRetentions[bi].retentionPriority.value_or(
                    executor::KvCacheRetentionConfig::kDefaultRetentionPriority),
                perBlockRetentions[bi].durationMs, mode, directory);
            addBlockToBeam(freeBlock, sequence, beamIdx);
            if (blockItr != blockKeys.end())
            {
                freeBlock->setBlockKey(
                    *blockItr, blockItr->uniqueTokens.size() == static_cast<size_t>(mTokensPerBlock));
                ++blockItr;
            }
            freeBlock->setHash();
            TLLM_LOG_DEBUG("%s::loadOrAllocateBlocks - Beam %d. Allocated non-shared block %d for bi %d",
                mLogPrefix.c_str(), beamIdx, freeBlock->getBlockId(), bi);
        }
        ++mMissedBlocks;
        if (blockItr != blockKeys.end())
        {
            ++blockItr;
        }
    }

    return numMatchedTokens;
}

void BlockManager::refreshBlocks()
{
    for (auto& [_, manager] : mWindowBlockManagers)
    {
        manager.refreshBlocks();
    }
}

void WindowBlockManager::refreshBlocks()
{
    mEvictionPolicy->refresh();
    mTransferManager->syncTransfers();
}

// There are two versions of BlockManager::addSequence function.
// This is called when block reuse is enabled.
void BlockManager::addSequence(GenerationRequest& sequence, SizeType32 inputLength, SizeType32 numContextBlocks,
    LlmRequest& llmRequest, SizeType32 windowSize)
{
    mWindowBlockManagers.at(windowSize).addSequence(sequence, inputLength, numContextBlocks, llmRequest);
}

// There are two versions of WindowBlockManager::addSequence function.
// This is called when block reuse is enabled.
void WindowBlockManager::addSequence(
    GenerationRequest& sequence, SizeType32 inputLength, SizeType32 numContextBlocks, LlmRequest& llmRequest)
{
    auto const requestId = sequence.getRequestId();
    auto const [seqIt, emplaceDone] = mAllocatedBlocksPerSeq.emplace(requestId, std::vector<BlockPtr>{});
    TLLM_CHECK(emplaceDone);

    auto constexpr beamIdx = 0;
    auto const& uniqueTokens = (mCacheType == CacheType::kSELF || mCacheType == CacheType::kSELFKONLY)
        ? llmRequest.getUniqueTokens(beamIdx)
        : *(llmRequest.getEncoderUniqueTokens().value());

    // Ignore last token because it can't be recovered
    auto blockedUniqueTokens = chopVectorIntoBlocks<UniqueToken>(uniqueTokens, inputLength - 1, mTokensPerBlock, true);
    // Add empty block if last token is separated
    if (inputLength % mTokensPerBlock == 1)
    {
        blockedUniqueTokens.emplace_back();
    }

    auto blockKeys = buildBlockKeys(blockedUniqueTokens, llmRequest);

    auto config = llmRequest.getKvCacheRetentionConfig();

    auto perBlockRetentions = config.value_or(executor::KvCacheRetentionConfig())
                                  .getPerBlockRetentionPriorityDuration(getTokensPerBlock(), inputLength);

    auto mode = config.value_or(executor::KvCacheRetentionConfig()).getTransferMode();
    auto directory = config.value_or(executor::KvCacheRetentionConfig()).getDirectory();

    if (mode != executor::KvCacheTransferMode::DRAM && directory.empty())
    {
        TLLM_LOG_WARNING(
            "Transfer mode %d specified without directory, falling back to DRAM mode", static_cast<int>(mode));
        mode = executor::KvCacheTransferMode::DRAM;
    }

    TLLM_CHECK(perBlockRetentions.size() == (size_t) numContextBlocks);

    auto const prepopulatedPromptLen
        = loadOrAllocateBlocks(blockKeys, numContextBlocks, sequence, perBlockRetentions, mode, directory);
    mReusedTokens += static_cast<double>(prepopulatedPromptLen);
    mTotalInputTokens += static_cast<double>(uniqueTokens.size());

    SizeType32 numConnectorMatchedTokens = 0;

    // If we're using a KV cache connector, check if any additional blocks can be loaded.
    if (mKvCacheConnectorManager && !llmRequest.isDummyRequest())
    {
        numConnectorMatchedTokens = mKvCacheConnectorManager->getNumNewMatchedTokens(llmRequest, prepopulatedPromptLen);
    }

    llmRequest.setPrepopulatedPromptLen(prepopulatedPromptLen + numConnectorMatchedTokens, getTokensPerBlock());
    TLLM_LOG_DEBUG("addSequence: Request %lu, inputLength %d, prepopulatedPromptLen %d, numConnectorMatchedTokens %d",
        llmRequest.mRequestId, inputLength, prepopulatedPromptLen, numConnectorMatchedTokens);
}

void BlockManager::adjustBlocksIfNeeded(GenerationRequest& sequence)
{
    for (auto& [windowSize, manager] : mWindowBlockManagers)
    {
        mWindowBlockManagers.at(windowSize).adjustBlocksIfNeeded(sequence);
    }
}

void WindowBlockManager::adjustBlocksIfNeeded(GenerationRequest& sequence)
{
    auto const minTokensForBlockDetach = mWindowSize + mTokensPerBlock;
    while (
        sequence.getNumTokens() - sequence.getNumFrontBlocksRemoved() * getTokensPerBlock() >= minTokensForBlockDetach)
    {
        // Detaching block for SWA is non-trivial due to the radix tree structure.
        // For now, when reuse is enabled, we do not detach blocks for SWA.
        TLLM_CHECK_WITH_INFO(mIsSWA, "A block only go out-of-window in SWA");
        detachFrontBlock(sequence);
    }

    if ((sequence.getNumTokens() - 1) % getTokensPerBlock() == 0)
    {
        // Allocating a new block when the last token is a block boundary
        allocateBlock(sequence, /*shareAmongBeams=*/sequence.getBeamWidth() == 1);
        updateLastCacheBlockOffsets(sequence);
    }
}

// There are two versions of BlockManager::addSequence function.
// This is called when block reuse is disabled.
void BlockManager::addSequence(
    GenerationRequest& sequence, SizeType32 numContextBlocks, SizeType32 windowSize, bool isShareLastContextBlock)
{
    mWindowBlockManagers.at(windowSize).addSequence(sequence, numContextBlocks, isShareLastContextBlock);
}

// There are two versions of WindowBlockManager::addSequence function.
// This is called when block reuse is disabled.
void WindowBlockManager::addSequence(
    GenerationRequest& sequence, SizeType32 numContextBlocks, bool isShareLastContextBlock)
{
    if (mKvCacheConnectorManager)
    {
        TLLM_LOG_WARNING(
            "KV Cache Connector specified when block reuse is disabled. The KV Cache Connector will be "
            "ignored.");
    }

    auto const requestId = sequence.getRequestId();
    auto const [seqIt, emplaceDone] = mAllocatedBlocksPerSeq.emplace(requestId, std::vector<BlockPtr>{});
    TLLM_CHECK(emplaceDone);

    TLLM_CHECK_WITH_INFO(numContextBlocks > 0, "numContextBlocks must be greater than 0");
    for (SizeType32 bi = 0; bi < numContextBlocks - 1; ++bi)
    {
        allocateBlock(sequence, /*shareAmongBeams=*/true);
    }
    allocateBlock(sequence, /*shareAmongBeams=*/isShareLastContextBlock);
}

void WindowBlockManager::addBlockToBeam(BlockPtr& block, GenerationRequest& sequence, SizeType32 beamIdx)
{
    auto const requestId = sequence.getRequestId();
    block->incRefCount();
    if (sequence.getCacheBlockIds(mWindowSize).at(beamIdx).size() == 0)
    {
        block->setPrevBlockInSeq(nullptr);
    }
    else
    {
        block->setPrevBlockInSeq(mAllBlocksById.at(sequence.getCacheBlockIds(mWindowSize)[beamIdx].back()));
    }
    sequence.addCacheBlock(mWindowSize, beamIdx, block->getBlockId());
    mAllocatedBlocksPerSeq.at(requestId).push_back(block);
}

void WindowBlockManager::addBlockToAllBeams(BlockPtr& block, GenerationRequest& sequence)
{
    auto const beamWidth = sequence.getBeamWidth();

    for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        addBlockToBeam(block, sequence, beamIdx);
    }
}

void BlockManager::allocateBlock(GenerationRequest& sequence, SizeType32 windowSize)
{
    mWindowBlockManagers.at(windowSize).allocateBlock(sequence, false);
}

void WindowBlockManager::allocateBlock(GenerationRequest& sequence, bool shareAmongBeams)
{
    auto const beamWidth = sequence.getBeamWidth();
    auto const requiredBlocks = shareAmongBeams ? 1 : beamWidth;

    TLLM_CHECK_WITH_INFO(hasFreeBlocks(requiredBlocks), "Can't allocate new blocks. No free blocks left.");

    if (shareAmongBeams)
    {
        // add same block to all beams
        auto block = getFreeBlock(sequence, sequence.getDecodeRetentionPriority(), sequence.getDecodeDurationMs(),
            sequence.getTransferMode(), sequence.getDirectory());
        for (auto beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
        {
            addBlockToBeam(block, sequence, beamIdx);
        }
    }
    else
    {
        // add different block to each beam
        for (auto beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
        {
            auto block = getFreeBlock(sequence, sequence.getDecodeRetentionPriority(), sequence.getDecodeDurationMs(),
                sequence.getTransferMode(), sequence.getDirectory());
            addBlockToBeam(block, sequence, beamIdx);
        }
    }
}

std::pair<SizeType32, std::optional<KVCacheBlock::IdType>> WindowBlockManager::storeBlocks(
    std::vector<BlockKey> const& blockKeys, std::vector<KVCacheBlock::IdType> const& blockIds, bool pinBlocks)
{
    SizeType32 numBlocksStoredForReuse = 0;
    std::lock_guard<std::mutex> lock(mCachedBlocksRootMutex);
    TLLM_LOG_DEBUG(
        "%s::storeBlocks - %zu blockKeys, %zu blockIds", mLogPrefix.c_str(), blockKeys.size(), blockIds.size());

    auto searchRoot = mCachedBlocksRoot;
    bool needMatch = true;

    auto numBlocks = blockKeys.size();
    std::vector<BlockPtr> storedBlocks;
    std::optional<KVCacheBlock::IdType> lastStoredId = std::nullopt;
    for (std::size_t blockCnt = 0; blockCnt < numBlocks; ++blockCnt)
    {
        auto const bid = blockIds[blockCnt];
        TLLM_LOG_DEBUG("%s::storeBlocks - Searching match for block %d", mLogPrefix.c_str(), bid);
        auto& block = mAllBlocksById[bid];
        auto const& blockKey = blockKeys[blockCnt];

        auto [partialMatch, numMatched, matchedBlock]
            = needMatch ? searchRoot->findMatchingBlock(blockKey, false, false) : std::make_tuple(false, 0, nullptr);
        if (matchedBlock != nullptr)
        {
            // Found match
            TLLM_LOG_DEBUG(
                "%s::storeBlocks - Found matching block %d, traverse", mLogPrefix.c_str(), matchedBlock->getBlockId());
            searchRoot = matchedBlock;
            // TODO possible optimization: if bid != matchedBlock->getBlockId(),
            // block can be freed and inserted at mFreePrimaryBlocks.begin()
        }
        else
        {
            // No match
            TLLM_LOG_DEBUG("%s::storeBlocks - No match, inserting block %d into search structure", mLogPrefix.c_str(),
                block->getBlockId());
            TLLM_CHECK_WITH_INFO(block->getBlockId() == bid,
                "Block id mismatch " + std::to_string(block->getBlockId()) + " != " + std::to_string(bid));
            needMatch = false; // no matching needed for following blocks
            block->setBlockKey(blockKey, static_cast<SizeType32>(blockKey.uniqueTokens.size()) == mTokensPerBlock);
            block->setPrevBlock(searchRoot);
            block->setPrevBlockInSeq(searchRoot);
            searchRoot->addNextBlock(blockKey, block);

            // Sanity check. The list of stored blocks should be connected.
            TLLM_CHECK(storedBlocks.empty() || block->getPrevBlock() == storedBlocks.back());

            storedBlocks.push_back(block);
            TLLM_CHECK(block->getPrevBlockInSeq() == nullptr
                || block->getPrevBlockInSeq()->getHash() == searchRoot->getHash());
            auto oldHash = block->getHash();
            auto newHash = BlockKeyHasher()(blockKey, searchRoot->getHash());
            if (oldHash != newHash)
            {
                TLLM_LOG_DEBUG("#%d block hash %zx -> %zx", block->getBlockId(), oldHash, newHash);
                block->setHash(newHash);
            }
            searchRoot = block;
            numBlocksStoredForReuse++;
        }
        if (pinBlocks)
        {
            searchRoot->incRefCount();
        }
        lastStoredId = searchRoot->getBlockId();
    }
    if (mEventManager)
    {
        mEventManager->enqueueStoredEvent(storedBlocks, mWindowSize);
    }
    return {numBlocksStoredForReuse, lastStoredId};
}

void BlockManager::replaceSharedBlock(GenerationRequest& sequence, SizeType32 windowSize, SizeType32 blockIdx)
{
    mWindowBlockManagers.at(windowSize).replaceSharedBlock(sequence, blockIdx);
}

void WindowBlockManager::replaceSharedBlock(GenerationRequest& sequence, SizeType32 blockIdx)
{
    auto const requestId = sequence.getRequestId();
    auto const beamWidth = sequence.getBeamWidth();
    auto& allocatedBlocks = mAllocatedBlocksPerSeq.at(requestId);

    if (!allocatedBlocks.at((blockIdx + 1) * beamWidth - 1)->isShared())
    {
        return;
    }
    BlockKey blockKey = allocatedBlocks.at(blockIdx * beamWidth)->getBlockKey();
    bool isFull = allocatedBlocks.at(blockIdx * beamWidth)->isFull();

    // Free shared block
    for (auto beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        auto block = allocatedBlocks.at(blockIdx * beamWidth + beamIdx);
        block->decRefCount();
        if (!block->hasRefs())
        {
            mEvictionPolicy->releaseBlock(block);
        }
    }

    // Allocate new blocks
    TLLM_CHECK_WITH_INFO(hasFreeBlocks(beamWidth), "Can't allocate new blocks. No free blocks left.");
    for (auto beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        auto block = getFreeBlock(sequence, executor::KvCacheRetentionConfig::kDefaultRetentionPriority, std::nullopt,
            sequence.getTransferMode(), sequence.getDirectory());
        block->incRefCount();
        if (sequence.getCacheBlockIds(mWindowSize).at(beamIdx).size() == 0)
        {
            block->setPrevBlockInSeq(nullptr);
        }
        else
        {
            block->setPrevBlockInSeq(mAllBlocksById.at(sequence.getCacheBlockIds(mWindowSize)[beamIdx].back()));
        }
        block->setBlockKey(blockKey, isFull);
        block->setHash();
        sequence.changeCacheBlock(mWindowSize, beamIdx, blockIdx, block->getBlockId());
        allocatedBlocks.at(blockIdx * beamWidth + beamIdx) = block;
    }
}

void BlockManager::releaseLastBlock(GenerationRequest& sequence, SizeType32 windowSize)
{
    mWindowBlockManagers.at(windowSize).releaseLastBlock(sequence);
}

void WindowBlockManager::releaseLastBlock(GenerationRequest& sequence)
{
    auto const requestId = sequence.getRequestId();
    auto& allocatedBlocks = mAllocatedBlocksPerSeq.at(requestId);
    auto it = allocatedBlocks.rbegin();
    auto& block = *it;
    // Decrease ref count
    block->decRefCount();
    // If ref count is zero, move block to free blocks
    if (!block->hasRefs())
    {
        mEvictionPolicy->releaseBlock(block, true);
    }
    // Remove block from allocated blocks
    allocatedBlocks.pop_back();
    // Remove stored block ids in sequence
    sequence.removeLastBlock(mWindowSize);
}

[[nodiscard]] SizeType32 WindowBlockManager::getNumFreeBlocks() const noexcept
{
    return mEvictionPolicy->getNumFreeBlocks(kPrimaryLevel);
}

std::deque<tle::KVCacheEvent> BlockManager::getLatestEvents(std::optional<std::chrono::milliseconds> timeout) const
{
    return mEventManager ? mEventManager->getEvents(timeout) : std::deque<tle::KVCacheEvent>{};
}

std::optional<KVCacheBlock::IdType> BlockManager::storeBlocksForReuse(
    GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest, bool pinBlocks)
{
    std::optional<KVCacheBlock::IdType> lastStoredId = std::nullopt;
    for (auto& [_, manager] : mWindowBlockManagers)
    {
        lastStoredId = manager.storeBlocksForReuse(sequence, llmRequest, pinBlocks);
    }
    return lastStoredId;
}

std::optional<KVCacheBlock::IdType> BlockManager::releaseBlocks(
    GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest, bool pinBlocks)
{
    // Released block will be stored when reuse is enabled.
    // Reuse is implied to be enabled if llmRequest is provided.
    std::optional<KVCacheBlock::IdType> lastStoredId = std::nullopt;
    for (auto& [_, manager] : mWindowBlockManagers)
    {
        if (!llmRequest.has_value() || llmRequest->isDummyRequest() || sequence.getBeamWidth() > 1)
        {
            lastStoredId = manager.releaseBlocks(sequence, std::nullopt);
        }
        else
        {
            lastStoredId = manager.releaseBlocks(sequence, llmRequest);
        }
    }
    return lastStoredId;
}

void BlockManager::pinBlocks(GenerationRequest& sequence)
{
    for (auto& [_, manager] : mWindowBlockManagers)
    {
        manager.pinBlocks(sequence);
    }
}

void BlockManager::unpinBlocksById(KVCacheBlock::IdType blockId)
{
    // Use the first window size
    if (mWindowBlockManagers.empty())
    {
        return;
    }
    auto& firstManager = mWindowBlockManagers.begin()->second;
    firstManager.unpinBlocksById(blockId);
}

void WindowBlockManager::pinBlocks(GenerationRequest& sequence)
{
    auto const requestId = sequence.getRequestId();
    auto& allocatedBlocks = mAllocatedBlocksPerSeq.at(requestId);
    for (auto& block : allocatedBlocks)
    {
        block->incRefCount();
    }
}

void WindowBlockManager::unpinBlocksById(KVCacheBlock::IdType blockId)
{
    if (blockId < 0 || static_cast<size_t>(blockId) >= mAllBlocksById.size())
    {
        return;
    }
    auto block = mAllBlocksById[blockId];
    while (block && block->getBlockId() != KVCacheBlock::kCachedBlocksRootId)
    {
        block->decRefCount();
        if (!block->hasRefs())
        {
            mEvictionPolicy->releaseBlock(block);
        }
        block = std::move(block->getPrevBlock());
    }
}

void BlockManager::storeNewBlock(GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest)
{
    for (auto& [_, manager] : mWindowBlockManagers)
    {
        if (manager.isSWA())
        {
            // SWA cannot store new blocks on the fly because the block stored
            // may go OOW and be reused by another sequence.
            continue;
        }
        manager.storeNewBlock(sequence, llmRequest);
    }
}

void WindowBlockManager::storeNewBlock(GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest)
{
    auto constexpr beamIdx = 0;
    auto const& uniqueTokens = llmRequest->getUniqueTokens(beamIdx);
    auto const& cacheBlockIds = sequence.getCacheBlockIds(mWindowSize);

    if (uniqueTokens.size() == 0)
    {
        return;
    }

    // TODO: get the caller to mark tokens as filled / not filled, so that the kv-cache manager doesn't
    // have to guess. Only (length - 1) tokens of the sequence have their kv-state recorded in kv-cache. We assume
    // the last token's state is not filled yet.
    auto const usableSize = static_cast<runtime::SizeType32>(uniqueTokens.size()) - 1;
    if (usableSize % mTokensPerBlock != 0)
    {
        return;
    }
    auto blockedUniqueTokens = chopVectorIntoBlocks<UniqueToken>(uniqueTokens, usableSize, mTokensPerBlock, true);
    auto blockKeys = buildBlockKeys(blockedUniqueTokens, *llmRequest);
    if (blockKeys.size() < 2 || cacheBlockIds[beamIdx].size() < blockKeys.size())
    {
        // store all blocks
        TLLM_LOG_DEBUG("%s::storeNewBlock - store all blocks", mLogPrefix.c_str());
        (void) storeBlocks(std::move(blockKeys), cacheBlockIds[beamIdx]);
        return;
    }

    auto lastBlock = mAllBlocksById.at(cacheBlockIds[beamIdx][blockKeys.size() - 1]);
    auto prevBlock = mAllBlocksById.at(cacheBlockIds[beamIdx][blockKeys.size() - 2]);

    // If the previous block is not in the radix tree, we need to store all blocks
    if (prevBlock->getPrevBlock() == nullptr)
    {
        TLLM_LOG_DEBUG("%s::storeNewBlock - store all blocks", mLogPrefix.c_str());
        (void) storeBlocks(std::move(blockKeys), cacheBlockIds[beamIdx]);
        return;
    }

    if (lastBlock->getPrevBlock() != nullptr)
    {
        // If the last block is not in the radix tree, we need to store all blocks
        TLLM_LOG_DEBUG("%s::storeNewBlock - no need to store", mLogPrefix.c_str());
        return;
    }
    TLLM_LOG_DEBUG("%s::storeNewBlock - store the last block", mLogPrefix.c_str());
    (void) storeBlocks(std::move(blockKeys), cacheBlockIds[beamIdx]);
}

std::optional<KVCacheBlock::IdType> WindowBlockManager::storeBlocksForReuse(
    GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest, bool pinBlocks)
{
    auto constexpr beamIdx = 0;
    auto const& uniqueTokens = llmRequest->getUniqueTokens(beamIdx);
    auto const& cacheBlockIds = sequence.getCacheBlockIds(mWindowSize);

    // TODO: get the caller to mark tokens as filled / not filled, so that the kv-cache manager doesn't
    // have to guess. Only (length - 1) tokens of the sequence have their kv-state recorded in kv-cache. We assume
    // the last token's state is not filled yet.
    auto const usableSize = static_cast<runtime::SizeType32>(uniqueTokens.size()) - 1;
    auto blockedUniqueTokens = chopVectorIntoBlocks<UniqueToken>(uniqueTokens, usableSize, mTokensPerBlock, true);
    auto blockKeys = buildBlockKeys(blockedUniqueTokens, *llmRequest);
    return storeBlocks(std::move(blockKeys), cacheBlockIds[beamIdx], pinBlocks).second;
}

std::optional<KVCacheBlock::IdType> WindowBlockManager::releaseBlocks(
    GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest)
{
    auto const requestId = sequence.getRequestId();
    std::optional<KVCacheBlock::IdType> lastStoredId = std::nullopt;
    auto node = mAllocatedBlocksPerSeq.extract(requestId);
    TLLM_CHECK(node);
    auto& allocatedBlocks = node.mapped();
    if (llmRequest.has_value())
    {
        // If llmRequest is provided, block store for reuse is enabled.
        if (!isSequenceValidForStoreForReuse(requestId))
        {
            TLLM_LOG_DEBUG(
                "%s::releaseBlocks - sequence %lu does not have all blocks valid, block is not saved for reuse",
                mLogPrefix.c_str(), sequence.getRequestId());
        }
        else
        {
            if (mIsSWA)
            {
                TLLM_LOG_DEBUG("%s::releaseBlocks - sequence %lu is valid for store for reuse", mLogPrefix.c_str(),
                    sequence.getRequestId());
            }
            auto const& uniqueTokens = llmRequest->getUniqueTokens(/*beamIdx=*/0);
            // Only (length - 1) tokens of the sequence have their kv-state
            // recorded in kv-cache. We assume the last token's state is not filled yet.
            auto const usableSize = static_cast<runtime::SizeType32>(uniqueTokens.size()) - 1;
            auto blockedUniqueTokens
                = chopVectorIntoBlocks<UniqueToken>(uniqueTokens, usableSize, mTokensPerBlock, /*allowPartial=*/true);
            auto blockKeys = buildBlockKeys(blockedUniqueTokens, *llmRequest);

            std::vector<KVCacheBlock::IdType> cacheBlockIds(allocatedBlocks.size());
            std::transform(allocatedBlocks.begin(), allocatedBlocks.end(), cacheBlockIds.begin(),
                [](BlockPtr const& block) { return block->getBlockId(); });

            auto [numBlocksStoredForReuse, lastStoredId] = storeBlocks(std::move(blockKeys), cacheBlockIds);
            TLLM_LOG_DEBUG("%s::releaseBlocks Request %lu, %d blocks stored for reuse", mLogPrefix.c_str(),
                sequence.getRequestId(), numBlocksStoredForReuse);
        }
    }
    for (auto it = allocatedBlocks.rbegin(); it != allocatedBlocks.rend() - sequence.getNumFrontBlocksRemoved(); ++it)
    {
        auto& block = *it;
        // Decrease ref count
        if (block->hasRefs())
        {
            // An out-of-window block may not have any ref count.
            block->decRefCount();
        }
        // If ref count is zero, move block to free blocks
        if (!block->hasRefs())
        {
            mEvictionPolicy->releaseBlock(block);
        }
    }
    // Remove stored block ids in sequence
    sequence.clearCacheBlocks(mWindowSize);
    return lastStoredId;
}

void BlockManager::schedulingReleaseBlocks(RequestIdType requestId)
{
    for (auto& [_, manager] : mWindowBlockManagers)
    {
        manager.schedulingReleaseBlocks(requestId);
    }
}

void WindowBlockManager::schedulingReleaseBlocks(RequestIdType requestId)
{
    for (auto& block : mAllocatedBlocksPerSeq.at(requestId))
    {
        // Decrease ref count
        block->decSchedulingRefCount();
        // If ref count is zero, move block to free blocks
        if (!block->hasSchedulingRefs())
        {
            mSchedulingNumFreeBlocks++;
        }
    }
}

KVCacheManager::KVCacheManager(SizeType32 numLayers, SizeType32 numKvHeads, SizeType32 sizePerHead,
    SizeType32 tokensPerBlock, BlocksPerWindow const& blocksPerWindow, SizeType32 maxNumSequences,
    SizeType32 maxBeamWidth, std::vector<SizeType32> const& maxAttentionWindowVec,
    std::optional<TempAttentionWindowInputs> const& tempAttentionWindowInputs, nvinfer1::DataType dtype,
    SizeType32 sinkTokenLength, int64_t stream, runtime::SizeType32 maxSequenceLength, bool enableBlockReuse,
    bool onboardBlocks, CacheType cacheType, bool enablePartialReuse, bool copyOnPartialReuse)
    : KVCacheManager(std::vector<SizeType32>(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, maxBeamWidth, maxAttentionWindowVec, tempAttentionWindowInputs, dtype, sinkTokenLength,
        std::make_shared<runtime::CudaStream>(reinterpret_cast<cudaStream_t>(stream)), maxSequenceLength,
        enableBlockReuse, onboardBlocks, cacheType, std::nullopt, nullptr, enablePartialReuse, copyOnPartialReuse)
{
}

KVCacheManager::KVCacheManager(std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead,
    SizeType32 tokensPerBlock, BlocksPerWindow const& blocksPerWindow, SizeType32 maxNumSequences,
    SizeType32 maxBeamWidth, std::vector<SizeType32> const& maxAttentionWindowVec,
    std::optional<TempAttentionWindowInputs> const& tempAttentionWindowInputs, nvinfer1::DataType dtype,
    SizeType32 sinkTokenLength, int64_t stream, runtime::SizeType32 maxSequenceLength, bool enableBlockReuse,
    bool onboardBlocks, CacheType cacheType, std::optional<executor::RetentionPriority> secondaryOffloadMinPriority,
    std::shared_ptr<KVCacheEventManager> eventManager, bool enablePartialReuse, bool copyOnPartialReuse,
    std::shared_ptr<kv_connector::KvCacheConnectorManager> kvCacheConnectorManager)
    : KVCacheManager(numKvHeadsPerLayer, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences, maxBeamWidth,
        maxAttentionWindowVec, tempAttentionWindowInputs, dtype, sinkTokenLength,
        std::make_shared<runtime::CudaStream>(reinterpret_cast<cudaStream_t>(stream)), maxSequenceLength,
        enableBlockReuse, onboardBlocks, cacheType, secondaryOffloadMinPriority, eventManager, enablePartialReuse,
        copyOnPartialReuse, kvCacheConnectorManager)
{
}

KVCacheManager::KVCacheManager(std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead,
    SizeType32 tokensPerBlock, BlocksPerWindow const& blocksPerWindow, SizeType32 maxNumSequences,
    SizeType32 maxBeamWidth, std::vector<SizeType32> const& maxAttentionWindowVec,
    std::optional<TempAttentionWindowInputs> const& tempAttentionWindowInputs, nvinfer1::DataType dtype,
    SizeType32 sinkTokenLength, CudaStreamPtr stream, runtime::SizeType32 maxSequenceLength, bool enableBlockReuse,
    bool onboardBlocks, CacheType cacheType, std::optional<executor::RetentionPriority> secondaryOffloadMinPriority,
    std::shared_ptr<KVCacheEventManager> eventManager, bool enablePartialReuse, bool copyOnPartialReuse,
    std::shared_ptr<kv_connector::KvCacheConnectorManager> kvCacheConnectorManager)
    : mMaxBeamWidth(maxBeamWidth)
    , mDataType(dtype)
    , mMaxAttentionWindow(*std::max_element(maxAttentionWindowVec.begin(), maxAttentionWindowVec.end()))
    , mTokensPerBlock(tokensPerBlock)
    , mSinkBubbleLength(BaseKVCacheManager::getSinkBubbleLength(sinkTokenLength, tokensPerBlock))
    , mSinkBlockTokenLength(mSinkBubbleLength + sinkTokenLength)
    , mBlockManager(numKvHeadsPerLayer, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
          std::move(stream), maxSequenceLength, maxBeamWidth, maxAttentionWindowVec, tempAttentionWindowInputs, dtype,
          mSinkBubbleLength, onboardBlocks, cacheType, secondaryOffloadMinPriority, std::move(eventManager),
          enablePartialReuse, copyOnPartialReuse, std::move(kvCacheConnectorManager))
    // disable block reuse for sink bubble since chopVectorIntoBlocks does not match KV cache blocks in this case
    , mEnableBlockReuse{mSinkBubbleLength > 0 ? false : enableBlockReuse}
{
    TLLM_CHECK_WITH_INFO(mSinkBlockTokenLength == 0 && mSinkBubbleLength == 0,
        "[kv cache manager] streamLLM is not supported at the moment");
    TLLM_CHECK_DEBUG(std::find(maxAttentionWindowVec.begin(), maxAttentionWindowVec.end(), mMaxAttentionWindow)
        != maxAttentionWindowVec.end());
    // The sink tokens are stored in blocks separate from other tokens.
    // If the last block of sink tokens is only partially filled,
    // we fill that block with a "bubble" to reach the number of tokens per block.

    TLLM_CHECK(mSinkBlockTokenLength % tokensPerBlock == 0);
    TLLM_LOG_DEBUG("KV cache block reuse is %s", mEnableBlockReuse ? "enabled" : "disabled");
    mSequences.reserve(maxNumSequences);
}

KVCacheManager::KVCacheManager(SizeType32 numLayers, SizeType32 numKvHeads, SizeType32 sizePerHead,
    SizeType32 tokensPerBlock, BlocksPerWindow const& blocksPerWindow, SizeType32 maxNumSequences,
    SizeType32 maxBeamWidth, std::vector<SizeType32> const& maxAttentionWindowVec,
    std::optional<TempAttentionWindowInputs> const& tempAttentionWindowInputs, nvinfer1::DataType dtype,
    SizeType32 sinkTokenLength, CudaStreamPtr stream, runtime::SizeType32 maxSequenceLength, bool enableBlockReuse,
    bool onboardBlocks, CacheType cacheType, std::optional<executor::RetentionPriority> secondaryOffloadMinPriority,
    std::shared_ptr<KVCacheEventManager> eventManager, bool enablePartialReuse, bool copyOnPartialReuse,
    std::shared_ptr<kv_connector::KvCacheConnectorManager> kvCacheConnectorManager)
    : KVCacheManager(std::vector<SizeType32>(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, maxBeamWidth, maxAttentionWindowVec, tempAttentionWindowInputs, dtype, sinkTokenLength,
        std::move(stream), maxSequenceLength, enableBlockReuse, onboardBlocks, cacheType, secondaryOffloadMinPriority,
        std::move(eventManager), enablePartialReuse, copyOnPartialReuse, std::move(kvCacheConnectorManager))
{
}

void KVCacheManager::allocatePools(bool useUvm)
{
    mBlockManager.allocatePools(useUvm);
    auto const numPools = mBlockManager.getNumPools();

    uint64_t cacheSizeBytes = 0;
    for (SizeType32 poolIdx = 0; poolIdx < numPools; poolIdx++)
    {
        auto const cacheShape = mBlockManager.getPrimaryPool(poolIdx)->getShape();
        auto const cacheVolume = ITensor::volume(cacheShape);
#ifdef ENABLE_FP4
        auto const isFp4 = mDataType == nvinfer1::DataType::kFP4;
#else
        auto const isFp4 = false;
#endif
        if (!isFp4)
        {
            cacheSizeBytes += cacheVolume * BufferDataType(mDataType).getSize();
        }
        else
        {
            cacheSizeBytes += (cacheVolume * 4) / 8;
        }
    }
    // Save the total number of bytes allocated for the KV-cache for KvCacheStats
    mAllocatedBytes = cacheSizeBytes;
    if (tc::Logger::getLogger()->getLevel() <= tc::Logger::INFO)
    {

        TLLM_LOG_INFO("Number of tokens per block: %d.", mBlockManager.getTokensPerBlock());
        auto const maxNumTokens = mBlockManager.getNumPrimaryBlocks() * mBlockManager.getTokensPerBlock();
        TLLM_LOG_INFO("[MemUsageChange] Allocated %0.2f GiB for max tokens in paged KV cache (%d).",
            cacheSizeBytes / static_cast<double>(1 << 30), maxNumTokens);
    }

    auto const numKVPools = mBlockManager.getNumPools(/*include_block_scalar_pools=*/false);
    auto const numBlockScalePools = numPools - numKVPools;

    // Code in the attention kernels is cleaner if we can access the KV values and block scales separately.
    mBlockPoolPointers = BufferManager::cpu(ITensor::makeShape({numKVPools, 2}), TRTDataType<void*>::value);
    mBlockScalePoolPointers
        = BufferManager::cpu(ITensor::makeShape({numBlockScalePools, 2}), TRTDataType<void*>::value);

    auto poolPtrsRange = BufferRange<void*>(*mBlockPoolPointers);
    auto blockScalePtrsRange = BufferRange<void*>(*mBlockScalePoolPointers);
    SizeType32 kvPoolIdx = 0;
    SizeType32 blockScalePoolIdx = 0;

    for (SizeType32 poolIdx = 0; poolIdx < numPools; poolIdx++)
    {
        auto const& pool = mBlockManager.getPool(poolIdx);
        auto& outIdx = pool.containsBlockScales ? blockScalePoolIdx : kvPoolIdx;
        auto& outRange = pool.containsBlockScales ? blockScalePtrsRange : poolPtrsRange;

        outRange[outIdx * 2] = pool.primaryPtr->data();
        outRange[outIdx * 2 + 1] = pool.secondaryPtr ? pool.secondaryPtr->data() : nullptr;
        ++outIdx;
    }

    auto const numLayers = mBlockManager.getNumLayers();
    mLayerToPoolMapping = BufferManager::cpu(ITensor::makeShape({numLayers, 2}), TRTDataType<SizeType32>::value);
    auto poolMappingRange = BufferRange<SizeType32>(*mLayerToPoolMapping);
    for (SizeType32 layerIdx = 0; layerIdx < numLayers; layerIdx++)
    {
        auto const indexOfPool = mBlockManager.getLayerPoolIdx(layerIdx);
        auto const layerIdxInCachePool = mBlockManager.getPoolLayerIdx(layerIdx);
        poolMappingRange[layerIdx * 2] = indexOfPool;
        poolMappingRange[layerIdx * 2 + 1] = layerIdxInCachePool;
    }
}

void KVCacheManager::releasePools()
{
    mBlockManager.releasePools();
}

void KVCacheManager::startScheduling()
{
    mBlockManager.startScheduling();
}

SizeType32 KVCacheManager::getNeededBlocksOneStep(
    LlmRequest const& req, bool twoStepsLookAhead, SizeType32 windowSize) const
{
    if ((req.isContextInitState() && req.isFirstContextChunk()) || req.isDisaggGenerationInitState())
    {
        auto const chunkSize = req.mMaxNewTokens;
        auto const maxDraftTokensToAdd = req.getNumDraftTokens();
        auto const promptCacheLen
            = std::min((isCrossKv() ? req.getEncoderOutputLen() : req.mPromptLen) + maxDraftTokensToAdd,
                  windowSize + chunkSize)
            + mSinkBubbleLength;
        auto const numSharedBlocks = promptCacheLen / getTokensPerBlock();
        auto const numUnSharedTokens = promptCacheLen % getTokensPerBlock();
        auto const numUnSharedBlocks
            = tc::ceilDiv(numUnSharedTokens, getTokensPerBlock()) * req.mSamplingConfig.beamWidth;
        auto const numRequiredBlocks = numSharedBlocks + numUnSharedBlocks;
        return numRequiredBlocks;
    }

    if (req.isGenerationInProgressState())
    {
        if (isCrossKv())
        {
            return 0;
        }

        auto const numCurrTokens = mSequences.at(req.mRequestId).getNumTokens();
        auto const generatedTokens = numCurrTokens - req.getPromptLen();
        auto const maxTokensToAddToKVCache = req.mMaxNewTokens - generatedTokens;
        auto const tokensPerStep = req.getNumDraftTokens() + 1;
        auto const maxTokensToAdd = std::min((twoStepsLookAhead ? 2 : 1) * tokensPerStep, maxTokensToAddToKVCache);
        auto const numNextTokens = numCurrTokens + maxTokensToAdd;

        if (numNextTokens > mBlockManager.getWindowSizeMetadata(windowSize).maxTokenNum)
        {
            return 0;
        }

        auto const numCurrBlocks = tc::ceilDiv(numCurrTokens, getTokensPerBlock());
        auto const numNextBlocks = tc::ceilDiv(numNextTokens, getTokensPerBlock());
        auto const numRequiredBlocks = (numNextBlocks - numCurrBlocks) * req.mSamplingConfig.beamWidth;
        return numRequiredBlocks;
    }

    return 0;
}

SizeType32 KVCacheManager::getRemainingBlocksToCompletion(LlmRequest const& req, SizeType32 windowSize) const
{

    if (isCrossKv())
    {
        if (req.isContextInitState() && req.getContextCurrentPosition() == 0)
        {
            return tc::ceilDiv(req.getEncoderOutputLen(), getTokensPerBlock());
        }

        return 0; // cross KV cache doesn't grow after the initial context phase
    }

    auto const temporaryAttentionWindow = mBlockManager.getWindowSizeMetadata(windowSize).temporaryAttentionWindow;

    SizeType32 const numContextBlocks
        = (std::min(req.mPromptLen, windowSize + temporaryAttentionWindow) + mSinkBubbleLength) / getTokensPerBlock();

    SizeType32 const numTotalBlocksPerBeam = tc::ceilDiv(
        std::min(req.mPromptLen + req.mMaxNewTokens, windowSize + temporaryAttentionWindow) + mSinkBubbleLength,
        getTokensPerBlock());

    SizeType32 const numGenBlocksPerBeam = numTotalBlocksPerBeam - numContextBlocks;

    SizeType32 numAllocBlocksPerBeam = 0;
    {
        std::scoped_lock lck(mSequencesMtx);
        auto const seqIt = mSequences.find(req.mRequestId);
        if (seqIt != mSequences.end())
        {
            auto const& seq = seqIt->second;
            numAllocBlocksPerBeam = seq.getCacheBlockIds(windowSize).at(0).size();
        }
    }

    // In case of sliding window attention, a new block is allocated when the
    // window slides (and then the out-of-window block is detached). So we
    // need an extra block for generation if the diff between the max sequence
    // length and the current sequence length crosses both a block boundary
    // and a window boundary.
    auto const isSlidingWindow = (req.mPromptLen + req.mMaxNewTokens) > windowSize;
    SizeType32 const currentSeqlenInBlocks = tc::ceilDiv(req.getNumTokens(0), getTokensPerBlock());
    SizeType32 const maxSeqlenInBlocks = tc::ceilDiv(req.mPromptLen + req.mMaxNewTokens, getTokensPerBlock());
    auto const willCrossBlockBoundary = maxSeqlenInBlocks > currentSeqlenInBlocks;
    auto const willCrossWindowBlockBoundary = maxSeqlenInBlocks > numTotalBlocksPerBeam;
    SizeType32 numExtraBlocksPerBeam
        = isSlidingWindow && willCrossBlockBoundary && willCrossWindowBlockBoundary ? 1 : 0;

    if (numAllocBlocksPerBeam < numContextBlocks) // Still haven't allocated all context blocks
    {
        return numContextBlocks - numAllocBlocksPerBeam
            + (numGenBlocksPerBeam + numExtraBlocksPerBeam) * req.mSamplingConfig.beamWidth;
    }

    return (numTotalBlocksPerBeam - numAllocBlocksPerBeam + numExtraBlocksPerBeam) * req.mSamplingConfig.beamWidth;
}

void BlockManager::updateSequenceCacheBlockOffsets(GenerationRequest& sequence, SizeType32 windowSize)
{
    auto const& cacheBlocks = sequence.getCacheBlockIds(windowSize);
    auto& cacheBlocksTensor = sequence.getCacheBlockIndices(windowSize);
    auto const beamWidth = sequence.getBeamWidth();

    auto* offsetsPtr = bufferCast<tk::KVCacheIndex>(cacheBlocksTensor);
    auto const& offsetsShape = cacheBlocksTensor.getShape();

    for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        auto const& beamCacheBlock = cacheBlocks[beamIdx];
        for (SizeType32 blockIdx = 0; blockIdx < static_cast<SizeType32>(beamCacheBlock.size()); ++blockIdx)
        {
            auto const blockId = beamCacheBlock.at(blockIdx);
            mWindowBlockManagers.at(windowSize).setOffsets(offsetsPtr, offsetsShape, beamIdx, blockIdx, blockId);
        }
    }
}

void WindowBlockManager::updateLastCacheBlockOffsets(GenerationRequest& sequence)
{
    auto const& cacheBlocks = sequence.getCacheBlockIds(mWindowSize);
    auto& cacheBlocksTensor = sequence.getCacheBlockIndices(mWindowSize);
    auto const beamWidth = sequence.getBeamWidth();

    auto* offsetsPtr = bufferCast<tk::KVCacheIndex>(cacheBlocksTensor);
    auto const& offsetsShape = cacheBlocksTensor.getShape();

    for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        auto const& beamCacheBlock = cacheBlocks[beamIdx];
        auto const blockId = beamCacheBlock.back();
        auto const blockIdx = static_cast<SizeType32>(beamCacheBlock.size() - 1);
        setOffsets(offsetsPtr, offsetsShape, beamIdx, blockIdx, blockId);
    }
}

void BlockManager::updateCacheBlockOffsetsAtIdx(GenerationRequest& sequence, SizeType32 windowSize, SizeType32 blockIdx)
{
    auto const& cacheBlocks = sequence.getCacheBlockIds(windowSize);
    auto& cacheBlocksTensor = sequence.getCacheBlockIndices(windowSize);
    auto const beamWidth = sequence.getBeamWidth();

    auto* offsetsPtr = bufferCast<tk::KVCacheIndex>(cacheBlocksTensor);
    auto const& offsetsShape = cacheBlocksTensor.getShape();

    for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        auto const& beamCacheBlock = cacheBlocks[beamIdx];
        auto const blockId = beamCacheBlock.at(blockIdx);
        mWindowBlockManagers.at(windowSize).setOffsets(offsetsPtr, offsetsShape, beamIdx, blockIdx, blockId);
    }
}

void KVCacheManager::addToken(RequestIdType requestId)
{
    // TODO: add streamLLM support
    auto& sequence = getSequence(requestId);
    sequence.addNewTokens(1);
    mBlockManager.adjustBlocksIfNeeded(sequence);
}

void WindowBlockManager::detachFrontBlock(GenerationRequest& sequence)
{
    // streamLLM is not supported at the moment. The out of window block will
    // always be the 0th block.
    TLLM_CHECK_WITH_INFO(
        sequence.getBeamWidth() == 1, "[kv cache manager] detachBlock does not support beamWidth > 1 now.");

    auto const requestId = sequence.getRequestId();
    auto const beamWidth = sequence.getBeamWidth();
    auto& allocatedBlocks = mAllocatedBlocksPerSeq.at(requestId);
    SizeType32 outOfWindowBlockIdx = sequence.getNumFrontBlocksRemoved();

    for (auto beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        auto outOfWindowBlock = allocatedBlocks.at(outOfWindowBlockIdx * beamWidth + beamIdx);
        TLLM_LOG_DEBUG("%s::detachFrontBlock - Detaching block %d from sequence %d", mLogPrefix.c_str(),
            outOfWindowBlock->getBlockId(), requestId);

        outOfWindowBlock->decRefCount();

        if (outOfWindowBlock->hasRefs())
        {

            TLLM_LOG_DEBUG("%s::detachFrontBlock - OOW Block %d still has a non-zero ref count", mLogPrefix.c_str(),
                outOfWindowBlock->getBlockId());
        }
        if (!outOfWindowBlock->hasRefs())
        {
            mEvictionPolicy->releaseBlock(outOfWindowBlock);
        }
    }

    // Disconnect first block from sequence and remove it from allocated blocks
    sequence.removeFrontBlock(mWindowSize);
}

std::optional<BlockKey> KVCacheManager::findNewContextBlock(
    VecUniqueTokens const& uniqueTokens, LlmRequest const& llmRequest) const
{
    auto newContextBlockOpt = mBlockManager.findNewContextBlock(uniqueTokens, llmRequest);
    return newContextBlockOpt;
}

void KVCacheManager::addSequence(
    RequestIdType requestId, SizeType32 inputLength, SizeType32 beamWidth, OptionalRef<LlmRequest> llmRequest)
{
    // TODO: add streamLLM support
    auto kvCacheRetentionConfig = llmRequest
        ? llmRequest->getKvCacheRetentionConfig().value_or(executor::KvCacheRetentionConfig())
        : executor::KvCacheRetentionConfig();

    auto const [seqIt, emplaceDone] = [&]
    {
        auto lck = std::scoped_lock(mSequencesMtx);
        return mSequences.try_emplace(requestId, requestId, inputLength, beamWidth,
            mBlockManager.getWindowSizesMetadata(), kvCacheRetentionConfig);
    }();
    TLLM_CHECK(emplaceDone);
    auto& sequence = seqIt->second;

    // Get statistics for block allocations/reuse pre request.
    SizeType32 const numAllocTotalBlocksPreRequest = mBlockManager.getNumAllocTotalBlocks();
    SizeType32 const numAllocNewBlocksPreRequest = mBlockManager.getNumAllocNewBlocks();
    SizeType32 const numReusedBlocksPreRequest = mBlockManager.getNumReusedBlocks();
    SizeType32 const numMissedBlocksPreRequest = mBlockManager.getNumMissedBlocks();

    if (!mBlockManager.isSequenceHeld(requestId))
    {
        mBlockManager.holdSequence(requestId);
        TLLM_LOG_DEBUG(
            "[kv cache manager] Encounter new sequence %d, initialize sequence storage validity for all window sizes",
            requestId);
    }
    else
    {
        TLLM_LOG_DEBUG(
            "[kv cache manager] Encounter existing sequence %d, skip sequence storage validity initialization",
            requestId);
    }
    for (auto const [windowSize, metadata] : mBlockManager.getWindowSizesMetadata())
    {
        // NOTE: Caller to KVCacheManager::addSequence should deal with the chunking
        auto const maxTokenNum = metadata.maxTokenNum;
        auto const temporaryAttentionWindow = metadata.temporaryAttentionWindow;

        // Consider the temporaryAttentionWindow when allocating blocks.
        auto const effectiveInputLength = std::min(inputLength, maxTokenNum + temporaryAttentionWindow);
        auto const numContextBlocks = tc::ceilDiv(effectiveInputLength, getTokensPerBlock());
        if (mEnableBlockReuse)
        {
            mBlockManager.addSequence(sequence, effectiveInputLength, numContextBlocks, *llmRequest, windowSize);
        }
        else
        {
            if (!mEnableBlockReuse && llmRequest && llmRequest->getKvCacheRetentionConfig().has_value())
            {
                TLLM_LOG_WARNING(
                    "Request %d has a retention configuration set, but block reuse is disabled. The retention "
                    "config "
                    "will "
                    "have no effect.",
                    llmRequest->mRequestId);
            }
            bool isShareLastContextBlock = isCrossKv() || effectiveInputLength % getTokensPerBlock() == 0;
            mBlockManager.addSequence(sequence, numContextBlocks, windowSize, isShareLastContextBlock);
        }
        mBlockManager.updateSequenceCacheBlockOffsets(sequence, windowSize);
    }

    if (llmRequest)
    {
        // Update statistics for block allocations/reuse per request.
        llmRequest->updateAllocTotalBlocksPerRequest(
            mBlockManager.getNumAllocTotalBlocks() - numAllocTotalBlocksPreRequest);
        llmRequest->updateAllocNewBlocksPerRequest(mBlockManager.getNumAllocNewBlocks() - numAllocNewBlocksPreRequest);
        llmRequest->updateReusedBlocksPerRequest(mBlockManager.getNumReusedBlocks() - numReusedBlocksPreRequest);
        llmRequest->updateMissedBlocksPerRequest(mBlockManager.getNumMissedBlocks() - numMissedBlocksPreRequest);
    }
}

void KVCacheManager::storeContextBlocks(LlmRequest const& llmRequest)
{
    auto const requestId = llmRequest.mRequestId;
    if (mSequences.find(requestId) != mSequences.end())
    {
        auto& sequence = getSequence(requestId);
        if (mEnableBlockReuse && !llmRequest.isDummyRequest())
        {
            mBlockManager.storeContextBlocks(sequence, llmRequest);
        }
    }
    else
    {
        TLLM_LOG_WARNING("[kv cache manager] storeContextBlocks: Can not find sequence for request %lu", requestId);
    }
}

void KVCacheManager::storeNewBlock(LlmRequest const& llmRequest)
{
    // We store newest block for potential reuse only if:
    // - Beam search is NOT enabled
    // - Block reuse is enabled.
    auto const requestId = llmRequest.mRequestId;
    auto& sequence = getSequence(requestId);
    if (sequence.getBeamWidth() > 1 || !mEnableBlockReuse)
    {
        return;
    }
    mBlockManager.storeNewBlock(sequence, llmRequest);
}

std::optional<KVCacheBlock::IdType> KVCacheManager::removeSequence(
    RequestIdType requestId, OptionalRef<LlmRequest const> llmRequest, bool pinBlocks)
{
    TLLM_LOG_TRACE("[%s]::%s start", isCrossKv() ? "CROSS" : "SELF", __PRETTY_FUNCTION__);
    auto sequenceNode = [this, requestId]
    {
        std::scoped_lock lock(mSequencesMtx);
        return mSequences.extract(requestId);
    }();
    std::optional<KVCacheBlock::IdType> lastStoredId = std::nullopt;
    if (!sequenceNode.empty())
    {
        if (mEnableBlockReuse)
        {
            lastStoredId = mBlockManager.releaseBlocks(sequenceNode.mapped(), llmRequest, pinBlocks);
        }
        else
        {
            lastStoredId = mBlockManager.releaseBlocks(sequenceNode.mapped(), std::nullopt, pinBlocks);
        }
    }
    if (mBlockManager.isSequenceHeld(requestId))
    {
        mBlockManager.releaseSequence(requestId);
        TLLM_LOG_DEBUG("Remove sequence %d, release sequence storage validity for all window sizes", requestId);
    }
    TLLM_CHECK(!mBlockManager.isSequenceHeld(requestId));
    TLLM_LOG_TRACE("[%s]::%s stop", isCrossKv() ? "CROSS" : "SELF", __PRETTY_FUNCTION__);
    return lastStoredId;
}

std::optional<KVCacheBlock::IdType> KVCacheManager::storeBlocksForReuse(
    RequestIdType requestId, OptionalRef<LlmRequest const> llmRequest, bool pinBlocks)
{
    TLLM_LOG_TRACE("[%s]::%s start", isCrossKv() ? "CROSS" : "SELF", __PRETTY_FUNCTION__);
    auto& sequence = getSequence(requestId);
    std::optional<KVCacheBlock::IdType> lastStoredId
        = mBlockManager.storeBlocksForReuse(sequence, llmRequest, pinBlocks);
    TLLM_LOG_TRACE("[%s]::%s stop", isCrossKv() ? "CROSS" : "SELF", __PRETTY_FUNCTION__);
    return lastStoredId;
}

void KVCacheManager::schedulingRemoveSequence(RequestIdType requestId)
{
    // Mimic Free all blocks for this sequence
    mBlockManager.schedulingReleaseBlocks(requestId);
}

void KVCacheManager::pinBlocks(RequestIdType requestId)
{
    auto& sequence = getSequence(requestId);
    mBlockManager.pinBlocks(sequence);
}

void KVCacheManager::unpinBlocksById(KVCacheBlock::IdType blockId)
{
    mBlockManager.unpinBlocksById(blockId);
}

SizeType32 KVCacheManager::copyBlockOffsets(ITensor& output, SizeType32 outputSlotOffset, RequestIdType requestId) const
{
    auto const& sequence = getSequence(requestId);
    auto const beamWidth = sequence.getBeamWidth();

    auto* dstPtr = bufferCast<tk::KVCacheIndex>(output);
    auto const& dstShape = output.getShape();

    SizeType32 constexpr kIdx = 0;
    SizeType32 constexpr vIdx = 1;

    SizeType32 maxBlockCount{0};
    // Get page table for each KV cache pool
    SizeType32 absolutePoolIdx = 0;
    for (auto const [windowSize, metadata] : mBlockManager.getWindowSizesMetadata())
    {
        auto const& cacheBlocksTensor = sequence.getCacheBlockIndices(windowSize);
        auto const* srcPtr = bufferCast<tk::KVCacheIndex>(cacheBlocksTensor);
        auto const& srcShape = cacheBlocksTensor.getShape();
        auto const& cacheBlockIds = sequence.getCacheBlockIds(windowSize);
        for (SizeType32 poolIdx = 0; poolIdx < metadata.numPools; poolIdx++, absolutePoolIdx++)
        {
            for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
            {
                auto const beamBlockCount = cacheBlockIds[beamIdx].size();
                auto const copyChunkSize = beamBlockCount * sizeof(tk::KVCacheIndex);
                for (auto xIdx : {kIdx, vIdx})
                {
                    auto const srcIndex = tc::flat_index(srcShape.d, poolIdx, beamIdx, xIdx, 0);
                    auto const dstIndex
                        = tc::flat_index(dstShape.d, absolutePoolIdx, outputSlotOffset + beamIdx, xIdx, 0);
                    std::memcpy(dstPtr + dstIndex, srcPtr + srcIndex, copyChunkSize);
                }
                maxBlockCount = std::max<SizeType32>(maxBlockCount, static_cast<SizeType32>(beamBlockCount));
            }
        }
    }
    return maxBlockCount;
}

void KVCacheManager::getBlockOffsetsOfBatch(
    ITensor& output, SizeType32 firstBatchSlotIdx, SizeType32 batchSize, SizeType32 beamWidth) const
{
    // Get page table for each KV cache pool
    for (auto batchSlotIdx = 0; batchSlotIdx < batchSize; ++batchSlotIdx)
    {
        copyBlockOffsets(output, batchSlotIdx * beamWidth, firstBatchSlotIdx + batchSlotIdx);
    }
}

std::map<SizeType32, std::vector<SizeType32>> BaseKVCacheManager::groupLayersByWindowSize(
    std::vector<SizeType32> const& maxAttentionWindowVec, SizeType32 numLayers)
{
    auto const numNonUniqueWindowSizes = static_cast<SizeType32>(maxAttentionWindowVec.size());
    std::map<SizeType32, std::vector<SizeType32>> uniqueWindowSizeToLayers;
    for (SizeType32 layerIdx = 0; layerIdx < numLayers; layerIdx++)
    {
        /*
        At this point (Deep in the construction of TrtGptModel), maxAttentionWindowVec isn't "stretched" to the
        length of numLayers yet. So, we need to rotate the window sizes per layer with modulo.
        */
        auto const windowSize = maxAttentionWindowVec.at(layerIdx % numNonUniqueWindowSizes);
        uniqueWindowSizeToLayers[windowSize].push_back(layerIdx);
    }
    return uniqueWindowSizeToLayers;
}

std::tuple<uint64_t, uint64_t> BaseKVCacheManager::calculateFreeMemBytes(
    runtime::BufferManager const& bufferManager, executor::KvCacheConfig const& config)
{
    auto const freeMemFraction
        = config.getFreeGpuMemoryFraction().value_or(executor::KvCacheConfig::kDefaultGpuMemFraction);
    TLLM_CHECK_WITH_INFO(freeMemFraction < 1.0F,
        "Invalid freeMemFraction, freeMemFraction (%f) must be smaller than 1.0f", freeMemFraction);
    if (config.getMaxTokens().has_value())
    {
        if (config.getFreeGpuMemoryFraction().has_value())
        {
            TLLM_LOG_WARNING(
                "Both freeGpuMemoryFraction (aka kv_cache_free_gpu_mem_fraction) "
                "and maxTokens (aka max_tokens_in_paged_kv_cache) "
                "are set (to %f and %ld, respectively). The smaller value will be used.",
                freeMemFraction, (int64_t) config.getMaxTokens().value());
        }
    }

    TLLM_CUDA_CHECK(::cudaDeviceSynchronize());
    auto const [freeMem, totalMem] = tc::getDeviceMemoryInfo(config.getUseUvm());
    auto const finalFreeMem = freeMem + bufferManager.memoryPoolFree();
    TLLM_LOG_INFO("Memory usage when calculating max tokens in paged kv cache: total: %0.2f GiB, available: %0.2f GiB",
        totalMem / static_cast<double>(1 << 30), finalFreeMem / static_cast<double>(1 << 30));
    TLLM_CHECK_WITH_INFO(finalFreeMem <= totalMem, "Free memory cannot exceed total memory");

    auto const freePrimaryMemBytes = static_cast<uint64_t>(finalFreeMem * freeMemFraction);
    auto const freeSecondaryMemBytes = config.getHostCacheSize().value_or(0);

    TLLM_LOG_DEBUG("Calculated free memory: {.freePrimaryMemBytes=%" PRIu64 ", .freeSecondaryMemBytes=%" PRIu64 "}",
        freePrimaryMemBytes, freeSecondaryMemBytes);

    return std::make_tuple(freePrimaryMemBytes, freeSecondaryMemBytes);
}

namespace
{
bool isSortedVectorIdenticalAcrossAllRanks(WorldConfig const& worldConfig, std::vector<SizeType32> const& vector)
{
    auto const numRanks = worldConfig.getSize();
    auto const numElements = static_cast<int>(vector.size());
    int maxNumElements = 0;
    int minNumElements = 0;
    COMM_SESSION.allreduce(&numElements, &maxNumElements, 1, mpi::MpiType::kINT32, mpi::MpiOp::MAX);
    COMM_SESSION.allreduce(&numElements, &minNumElements, 1, mpi::MpiType::kINT32, mpi::MpiOp::MIN);
    if (maxNumElements != minNumElements)
    {
        return false;
    }
    std::vector<SizeType32> allElements(numElements * numRanks);
    COMM_SESSION.allgather(vector.data(), allElements.data(), numElements, mpi::MpiType::kUINT32);

    for (int i = 0; i < numElements; ++i)
    {
        auto const ref = allElements.at(i);
        for (int rank = 1; rank < numRanks; ++rank)
        {
            if (allElements[rank * numElements + i] != ref)
                return false;
        }
    }
    return true;
}
} // namespace

BlocksPerWindow BaseKVCacheManager::calculateMaxNumBlocks(executor::KvCacheConfig const& config, bool isCrossAttention,
    nvinfer1::DataType dtype, ModelConfig const& modelConfig, WorldConfig const& worldConfig,
    std::map<SizeType32, std::vector<SizeType32>> const& windowSizeToLayers, uint64_t allottedPrimaryMemBytes,
    uint64_t allottedSecondaryMemBytes, size_t extraCostMemory, SizeType32 kvFactor)
{
    TLLM_LOG_DEBUG("Calculating max num blocks for %s: {.allottedPrimaryMemBytes=%" PRIu64
                   ", .allottedSecondaryMemBytes=%" PRIu64 "}",
        isCrossAttention ? "Cross KvCacheManager" : "Self KvCacheManager", allottedPrimaryMemBytes,
        allottedSecondaryMemBytes);

    if (config.getMaxTokens().has_value() && windowSizeToLayers.size() > 1)
    {
        TLLM_LOG_WARNING(
            "Setting maxTokens when using Variable Sliding Window Attention is a strange concept, as it limits "
            "the number of max tokens *per window size* [limiting the sum of all window sizes is even stranger]. "
            "Anticipating the effects of this requires quite a complex calculation, and it probably isn't the "
            "configuration you meant to use.");
    }

    std::map<SizeType32, SizeType32> cacheSizeBytesPerTokenPerWindow;
    for (auto const& [windowSize, managedLayers] : windowSizeToLayers)
    {
        auto const cacheSizePerToken = BaseKVCacheManager::calculateCacheSizePerTokenForSingleWindowSize(
            modelConfig, managedLayers, isCrossAttention, kvFactor);
        auto const cacheSizeBytesPerToken = cacheSizePerToken * BufferDataType(dtype).getSize();
        cacheSizeBytesPerTokenPerWindow[windowSize] = cacheSizeBytesPerToken;
    }
    bool const isVSWA = cacheSizeBytesPerTokenPerWindow.size() > 1;

    TLLM_LOG_DEBUG("extraCostMemory [Gib]: %0.2f", extraCostMemory / static_cast<double>(1 << 30));
    allottedPrimaryMemBytes = allottedPrimaryMemBytes - extraCostMemory;
    auto const tokensPerBlock = modelConfig.getTokensPerBlock();
    auto const calculatePrimaryBlocks
        = [&](SizeType32 windowSize, float windowSizeShare, SizeType32 cacheSizeBytesPerToken)
    {
        TLLM_LOG_DEBUG("windowSizeShare: %f, cacheSizeBytesPerToken: %d", windowSizeShare, cacheSizeBytesPerToken);
        auto maxTokens = static_cast<uint64_t>(
            allottedPrimaryMemBytes * windowSizeShare / static_cast<double>(cacheSizeBytesPerToken));
        // kv_cache_config.max_tokens is not effective in VSWA scheme
        if (config.getMaxTokens().has_value() && !isVSWA)
        {
            auto const maxTokensFromConfig = static_cast<uint64_t>(config.getMaxTokens().value());
            if (maxTokensFromConfig < maxTokens)
            {
                TLLM_LOG_DEBUG("Maximum kv-cache token overridden by configuration as '%ld'.", maxTokensFromConfig);
                maxTokens = std::min(maxTokensFromConfig, maxTokens);
            }
        }
        TLLM_LOG_DEBUG("Primary maxTokens for windowSize %d: %ld", windowSize, maxTokens);
        SizeType32 const blocksInPrimaryPool = tc::ceilDiv(maxTokens, tokensPerBlock);
        TLLM_LOG_DEBUG(
            "Number of blocks in KV cache primary pool for windowSize %d: %d", windowSize, blocksInPrimaryPool);
        return blocksInPrimaryPool;
    };

    auto const calculateSecondaryBlocks
        = [&](SizeType32 windowSize, float windowSizeShare, SizeType32 cacheSizeBytesPerToken)
    {
        auto const maxTokensSecondary
            = static_cast<SizeType32>(allottedSecondaryMemBytes * windowSizeShare / cacheSizeBytesPerToken);
        SizeType32 const blocksInSecondaryPool = std::max(0, maxTokensSecondary / tokensPerBlock);
        TLLM_LOG_DEBUG(
            "Number of blocks in KV cache secondary pool for windowSize %d: %d, onboard blocks to primary memory "
            "before reuse: %s",
            windowSize, blocksInSecondaryPool, config.getOnboardBlocks() ? "true" : "false");
        return blocksInSecondaryPool;
    };

    std::map<SizeType32, float> windowSizeToShare;
    // By default, we allocate equal proportion shares of memory for all
    // window sizes (see the else case). With TRTLLM_WINDOW_SIZE_SHARES,
    // we can override this behavior to adjust the memory share of each
    // window size. For example, if we have window size of [512, 32768],
    // then setting TRTLLM_WINDOW_SIZE_SHARES=0.4,0.6 will be allocating
    // 40% of the memory to window size 512 and 60% of the memory to window
    // size 32768.
    if (auto envStr = std::getenv("TRTLLM_WINDOW_SIZE_SHARES"))
    {
        std::stringstream ss(envStr);
        std::vector<float> shares;
        float share;
        while (ss >> share)
        {
            shares.push_back(share);
            if (ss.peek() == ',')
                ss.ignore();
        }

        TLLM_CHECK_WITH_INFO(shares.size() == windowSizeToLayers.size(),
            "Number of shares in TRTLLM_WINDOW_SIZE_SHARES (%ld) must match number of window sizes (%ld)",
            shares.size(), windowSizeToLayers.size());
        float sumShares = 0.0f;
        for (auto s : shares)
        {
            TLLM_CHECK_WITH_INFO(0.0f <= s && s <= 1.0f, "Shares must be in value range [0,1], got %f", s);
            sumShares += s;
        }
        TLLM_CHECK_WITH_INFO(sumShares > 0.0f, "Sum of shares must be > 0.");
        // Normalize shares to 1.0
        for (auto& s : shares)
        {
            s /= sumShares;
        }
        size_t i = 0;
        for (auto const& [windowSize, _] : windowSizeToLayers)
        {
            windowSizeToShare[windowSize] = shares[i++];
        }
    }
    else
    {
        // NOTE: Righteously, blocks allocated should be proportional with
        // regard to window size. Currently, we are first allocating identical
        // number of blocks for all layers to achieve identical performance.
        for (auto const& [windowSize, _] : windowSizeToLayers)
        {
            windowSizeToShare[windowSize] = 1.0f / windowSizeToLayers.size();
        }
    }

    std::vector<SizeType32> blocksPrimary;
    std::vector<SizeType32> blocksSecondary;
    for (auto const& [windowSize, managedLayers] : windowSizeToLayers)
    {
        auto const cacheSizeBytesPerToken = cacheSizeBytesPerTokenPerWindow.at(windowSize);
        auto const windowSizeShare = windowSizeToShare.at(windowSize);
        auto const blocksInPrimaryPool = calculatePrimaryBlocks(windowSize, windowSizeShare, cacheSizeBytesPerToken);
        auto const blocksInSecondaryPool
            = calculateSecondaryBlocks(windowSize, windowSizeShare, cacheSizeBytesPerToken);
        blocksPrimary.push_back(blocksInPrimaryPool);
        blocksSecondary.push_back(blocksInSecondaryPool);
    }

    std::vector<SizeType32> windowSizes;
    windowSizes.reserve(windowSizeToLayers.size());
    for (auto const& [k, _] : windowSizeToLayers)
    {
        windowSizes.push_back(k);
    }
    if (worldConfig.getSize() > 1)
    {
        TLLM_CHECK(worldConfig.validMpiConfig());
        auto const rank = worldConfig.getRank();
        using tensorrt_llm::common::vec2str;
        TLLM_CHECK_WITH_INFO(isSortedVectorIdenticalAcrossAllRanks(
                                 worldConfig, windowSizes), // sorted thanks to windowSizeToLayers being a std::map
            "[RANK %d] Asymmetrical pipeline parallelism detected: Ranks either have a different number of window "
            "sizes, or differing values. This is not supported with Variable Sliding Window Attention. Local window "
            "sizes for reference: %s",
            rank, vec2str(windowSizes).c_str());
        TLLM_LOG_DEBUG(
            "[RANK %d] Before mpi::MpiOp::MIN reduction: window sizes %s / primary blocks %s / secondary blocks %s",
            rank, vec2str(windowSizes).c_str(), vec2str(blocksPrimary).c_str(), vec2str(blocksSecondary).c_str());
        // make sure all ranks use same value for max blocks
        auto blocksWorld = blocksPrimary;
        COMM_SESSION.allreduce(
            blocksPrimary.data(), blocksWorld.data(), windowSizes.size(), mpi::MpiType::kINT32, mpi::MpiOp::MIN);
        blocksPrimary = blocksWorld;
        COMM_SESSION.allreduce(
            blocksSecondary.data(), blocksWorld.data(), windowSizes.size(), mpi::MpiType::kINT32, mpi::MpiOp::MIN);
        blocksSecondary = blocksWorld;
        TLLM_LOG_DEBUG(
            "[RANK %d] After mpi::MpiOp::MIN reduction: window sizes %s / primary blocks %s / secondary blocks %s",
            rank, vec2str(windowSizes).c_str(), vec2str(blocksPrimary).c_str(), vec2str(blocksSecondary).c_str());
    }

    BlocksPerWindow windowSizeToBlocks;
    TLLM_LOG_INFO("Blocks per window size:");
    for (size_t i = 0; i < windowSizes.size(); ++i)
    {
        auto const windowSize = windowSizes.at(i);
        auto const primaryBlocks = blocksPrimary.at(i);
        auto const secondayBlocks = blocksSecondary.at(i);
        TLLM_LOG_INFO(
            "[windowSize=%d] {.primaryBlocks=%d, .secondayBlocks=%d}", windowSize, primaryBlocks, secondayBlocks);
        windowSizeToBlocks[windowSize] = {primaryBlocks, secondayBlocks};
    }
    return windowSizeToBlocks;
}

void KVCacheManager::removeToken(RequestIdType requestId)
{
    // TODO: add streamLLM support
    auto& sequence = getSequence(requestId);
    if (sequence.getNumTokens() == 0)
    {
        return;
    }
    TLLM_CHECK_WITH_INFO(sequence.getBeamWidth() == 1, "[kv cache manager] removeToken does not support beamWidth > 1");
    sequence.removeTokens(1);
    for (auto const [windowSize, metadata] : mBlockManager.getWindowSizesMetadata())
    {
        SizeType32 const tokensInWindow = sequence.getNumTokens() % windowSize;
        if (tokensInWindow % getTokensPerBlock() == 0)
        {
            mBlockManager.releaseLastBlock(sequence, windowSize);
        }
    }
}

void KVCacheManager::rewindKVCache(RequestIdType requestId, SizeType32 rewindLengths)
{
    for (SizeType32 si = 0; si < rewindLengths; ++si)
    {
        removeToken(requestId);
    }
}

GenerationRequest const& KVCacheManager::getSequence(RequestIdType requestId) const
{
    auto lck = std::scoped_lock(mSequencesMtx);
    return mSequences.at(requestId);
}

GenerationRequest& KVCacheManager::getSequence(RequestIdType requestId)
{
    auto lck = std::scoped_lock(mSequencesMtx);
    return mSequences.at(requestId);
}

SizeType32 BaseKVCacheManager::getSinkBubbleLength(SizeType32 sinkTokenLen, SizeType32 tokensPerBlock)
{
    auto const sinkTokensInLastBlock = sinkTokenLen % tokensPerBlock;
    auto const sinkBubbleLength = sinkTokensInLastBlock == 0 ? 0 : tokensPerBlock - sinkTokensInLastBlock;
    return sinkBubbleLength;
}

std::vector<std::vector<SizeType32>> const& KVCacheManager::getCacheBlockIds(
    RequestIdType requestId, SizeType32 windowSize) const
{
    return getSequence(requestId).getCacheBlockIds(windowSize);
}

std::vector<std::vector<std::vector<SizeType32>>> KVCacheManager::getBatchCacheBlockIds(
    std::vector<LlmRequest::RequestIdType> const& requestIds, SizeType32 windowSize) const
{
    std::vector<std::vector<std::vector<SizeType32>>> result{};
    result.reserve(requestIds.size());
    for (auto const& requestId : requestIds)
    {
        auto const& sequence = getSequence(requestId);
        result.emplace_back(sequence.getCacheBlockIds(windowSize));
    }
    return result;
}

std::optional<KVCacheBlock::IdType> KVCacheManager::getLastBlockId(LlmRequest::RequestIdType requestId) const
{
    auto const& seq = getSequence(requestId);
    // Use the first window size
    auto firstWindowSize = mBlockManager.getFirstWindowSize();
    if (firstWindowSize == 0)
    {
        return std::nullopt;
    }
    auto const& perBeam = seq.getCacheBlockIds(firstWindowSize);
    if (perBeam.empty() || perBeam[0].empty())
    {
        return std::nullopt;
    }
    return perBeam[0].back();
}

runtime::ITensor::SharedPtr KVCacheManager::getUniquePrimaryPool() const
{
    TLLM_CHECK_WITH_INFO(mBlockManager.getWindowSizesMetadata().size() == 1,
        "getUniquePrimaryPool is only supported for a single window size");
    return mBlockManager.getPrimaryPool(0);
}

runtime::ITensor::SharedPtr KVCacheManager::getPrimaryPool(SizeType32 layer_idx) const
{
    return mBlockManager.getPrimaryPool(mBlockManager.getLayerPoolIdx(layer_idx));
}

SizeType32 KVCacheManager::getMaxCapacityBatchSize(SizeType32 inputLength, SizeType32 outputLength) const
{
    auto minMaxBatchSizeAllWindows = std::numeric_limits<SizeType32>::max();
    for (auto const [windowSize, metadata] : mBlockManager.getWindowSizesMetadata())
    {
        auto const blockRequirementsPerSequence = KVCacheManager::calculateMaxBlockRequirements(
            inputLength, outputLength, mSinkBlockTokenLength, windowSize, mMaxBeamWidth, mTokensPerBlock);
        auto const maxBatchSizeWindow = metadata.allottedPrimaryBlocks / blockRequirementsPerSequence;
        // The window with the *smallest* max batch size is the limiting factor
        // Hence, the std::*min* of all the max batch sizes is chosen
        minMaxBatchSizeAllWindows = std::min(minMaxBatchSizeAllWindows, maxBatchSizeWindow);
    }
    return minMaxBatchSizeAllWindows;
}

SizeType32 KVCacheManager::calculateMaxBlockRequirementsPerBeam(
    SizeType32 sequenceLength, SizeType32 sinkTokenLength, SizeType32 windowSize, SizeType32 tokensPerBlock)
{
    auto const sinkBubbleLength = BaseKVCacheManager::getSinkBubbleLength(sinkTokenLength, tokensPerBlock);
    auto const actualSeqLen = std::min(sequenceLength, windowSize);
    auto actualMaxTokenNum = actualSeqLen + sinkBubbleLength;
    auto numBlocks = tc::ceilDiv(actualMaxTokenNum, tokensPerBlock);
    if (sequenceLength > windowSize)
    {
        numBlocks += kSWAExtraBlock;
    }
    return numBlocks;
}

SizeType32 KVCacheManager::calculateMaxBlockRequirements(SizeType32 inputLength, SizeType32 outputLength,
    SizeType32 sinkTokenLength, SizeType32 windowSize, SizeType32 beamWidth, SizeType32 tokensPerBlock)
{
    // We split beam width > 1, as it introduces a lot of complexity.
    auto const wholeSequenceLength = inputLength + outputLength;
    if (beamWidth == 1)
    {
        return KVCacheManager::calculateMaxBlockRequirementsPerBeam(
            wholeSequenceLength, sinkTokenLength, windowSize, tokensPerBlock);
    }

    if (windowSize <= outputLength)
    {
        // We at most will need outputLength of distinct blocks for SWA
        return KVCacheManager::calculateMaxBlockRequirementsPerBeam(
                   outputLength, sinkTokenLength, windowSize, tokensPerBlock)
            * beamWidth;
    }

    // Otherwise, we calculate how many tokens will be in output blocks.
    auto const effectiveAttentionWindow = std::min(windowSize, wholeSequenceLength);
    auto const numContextTokensInAttentionWindow
        = effectiveAttentionWindow - outputLength; // This is positive because we handled the other case above.
    auto const sinkBubbleLength = BaseKVCacheManager::getSinkBubbleLength(sinkTokenLength, tokensPerBlock);
    auto const numContextBlocks = (numContextTokensInAttentionWindow + sinkBubbleLength) / tokensPerBlock;
    auto const leftoverContextToken = numContextTokensInAttentionWindow - numContextBlocks * tokensPerBlock;
    auto numOutputBlocks = tc::ceilDiv(outputLength + leftoverContextToken, tokensPerBlock);
    if (wholeSequenceLength > windowSize)
    {
        numOutputBlocks += kSWAExtraBlock;
    }
    return numContextBlocks + numOutputBlocks * beamWidth;
}

[[nodiscard]] SizeType32 KVCacheManager::calculateMaxAttentionWindow(SizeType32 inputLength, SizeType32 outputLength,
    SizeType32 sinkTokenLength, SizeType32 blockCapacity, SizeType32 beamWidth, SizeType32 tokensPerBlock)
{
    // The function that gives the number of blocks required given an attention window is only linear by part. It is
    // however, monotonically increasing in the attention window. Therefore, we need to find in which part of the
    // function we are. First, are we in a case where not even the entire output will fit?
    auto const outputBlockRequirements
        = calculateMaxBlockRequirements(0, outputLength, sinkTokenLength, outputLength, beamWidth, tokensPerBlock);
    if (outputBlockRequirements > blockCapacity)
    {
        return (blockCapacity / beamWidth) * tokensPerBlock;
    }

    // Otherwise, we need to determine how many context tokens we can fit on top of the output tokens. First, there
    // are a few context tokens we might be able to fit 'for free' because the output is not a multiple of the
    // number of tokens per block.
    auto const leftoverBlockCapacity = blockCapacity - outputBlockRequirements;
    return std::min(outputLength + leftoverBlockCapacity * tokensPerBlock, inputLength + outputLength);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
