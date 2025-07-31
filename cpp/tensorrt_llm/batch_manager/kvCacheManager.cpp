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
using namespace tensorrt_llm::runtime;
using namespace tensorrt_llm::batch_manager::kv_cache_manager;
using namespace tensorrt_llm::batch_manager::eviction_policy;

using BlocksPerWindow = std::map<SizeType32, std::tuple<SizeType32, SizeType32>>;

namespace
{

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

inline uint8_t getNthByte(SizeType32 hashPart, uint8_t byteIdx) noexcept
{
    return static_cast<uint8_t>((hashPart >> (24 - byteIdx * 8)) & 0xFF);
}

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
            SizeType32 mmStartInBlock = (startPos >= startTokenIdx) ? 0 : startTokenIdx - startPos;
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
            std::move(uniqueTokens), std::move(extraKeys));
    }
    return blockKeys;
}

} // namespace

namespace tensorrt_llm::batch_manager::kv_cache_manager
{
size_t BlockKeyHasher::hash(BlockKey const& blockKey, std::size_t parentHash) noexcept
{
    // Hashing algorithm adapted from StackOverflow:
    // https://stackoverflow.com/questions/664014/what-integer-hash-function-are-good-that-accepts-an-integer-hash-key
    // Constants provide very good distribution - each input bit affects each output bit with ~50% probability.
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
    TLLM_CHECK_WITH_INFO(hasRefs(), "Can't remove link from block that is not allocated");
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
    std::shared_ptr<runtime::CudaStream> stream, std::optional<SizeType32> maxSequenceLength, SizeType32 maxBeamWidth,
    std::vector<SizeType32> const& maxAttentionWindowVec,
    std::optional<TempAttentionWindowInputs> const& tempAttentionWindowInputs, nvinfer1::DataType dtype,
    SizeType32 sinkBubbleLength, bool onboardBlocks, CacheType cacheType,
    std::optional<executor::RetentionPriority> secondaryOffloadMinPriority,
    std::shared_ptr<KVCacheEventManager> eventManager, bool enableHashKey, bool enablePartialReuse,
    bool copyOnPartialReuse)
    : mNumLayers{static_cast<SizeType32>(numKvHeadsPerLayer.size())}
    , mTokensPerBlock{tokensPerBlock}
    , mEventManager{std::move(eventManager)}
    , mStream{stream}
    , mCacheType{cacheType}
{
    auto const uniqueWindowSizeToLayers
        = BaseKVCacheManager::groupLayersByWindowSize(maxAttentionWindowVec, mNumLayers);
    auto const numUniqueWindowSizes = static_cast<SizeType32>(uniqueWindowSizeToLayers.size());

    mIsVariableWindow = numUniqueWindowSizes > 1;
    mIsVariableGQA = std::unordered_set(numKvHeadsPerLayer.begin(), numKvHeadsPerLayer.end()).size() > 1;

    mLayerToWindowSize.resize(mNumLayers);
    for (auto const& [windowSize, layersWithWindowSize] : uniqueWindowSizeToLayers)
    {
        for (auto& layerIdx : layersWithWindowSize)
        {
            mLayerToWindowSize.at(layerIdx) = windowSize;
        }
        auto const [allottedPrimaryBlocks, allottedSecondaryBlocks] = blocksPerWindow.at(windowSize);
        TLLM_CHECK(allottedPrimaryBlocks > 0); // You can't have a model with negative primary blocks...
        mWindowBlockManagers.try_emplace(windowSize, dtype, windowSize, layersWithWindowSize, numKvHeadsPerLayer,
            sizePerHead, tokensPerBlock, allottedPrimaryBlocks, allottedSecondaryBlocks, maxNumSequences, stream,
            onboardBlocks, cacheType, secondaryOffloadMinPriority, mEventManager, enableHashKey, enablePartialReuse,
            copyOnPartialReuse);
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
        auto const maxTokenNum = windowSize + sinkBubbleLength
            + (isUseOneMoreBlock(windowSize, maxSequenceLength, maxBeamWidth) ? tokensPerBlock : 0);
        auto const temporaryAttentionWindow = manager.calculateTemporaryAttentionWindow(tempAttentionWindowInputs);
        // Consider the temporaryAttentionWindow when allocating blocks.
        auto const maxBlocksPerSeq = tc::ceilDiv(maxTokenNum + temporaryAttentionWindow, tokensPerBlock);
        auto const [allottedPrimaryBlocks, allottedSecondaryBlocks] = blocksPerWindow.at(windowSize);
        mWindowSizeToMetadata[windowSize]
            = WindowSizeMetadata{allottedPrimaryBlocks, allottedSecondaryBlocks, absolutePoolsOffset, numPools,
                maxTokenNum, maxBlocksPerSeq, manager.getMaxNumBlocks(), temporaryAttentionWindow};
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
    SizeType32 sizePerHead, SizeType32 tokensPerBlock, SizeType32 blocksInPrimaryPool, SizeType32 blocksInSecondaryPool,
    SizeType32 maxNumSequences, std::shared_ptr<runtime::CudaStream> stream, bool onboardBlocks, CacheType cacheType,
    std::optional<executor::RetentionPriority> secondaryOffloadMinPriority,
    std::shared_ptr<KVCacheEventManager> eventManager, bool enableHashKey, bool enablePartialReuse,
    bool copyOnPartialReuse)
    : mDataType{dtype}
    , mWindowSize{windowSize}
    , mNumPrimaryBlocks{blocksInPrimaryPool}
    , mNumSecondaryBlocks{blocksInSecondaryPool}
    , mOnboardBlocks(onboardBlocks)
    , mBufferManager{std::move(stream)}
    , mSchedulingNumFreeBlocks{0}
    , mTokensPerBlock{tokensPerBlock}
    , mCachedBlocksRoot{std::make_shared<KVCacheBlock>(KVCacheBlock::kCachedBlocksRootId, tk::KVCacheIndex{0})}
    , mCacheType{cacheType}
    , mEventManager(std::move(eventManager))
    , mTransferManager{std::make_shared<KVCacheTransferManager>(mBufferManager)}
    , mAllocTotalBlocks{0}
    , mAllocNewBlocks{0}
    , mReusedBlocks{0}
    , mReusedUniqueBlocks{0}
    , mMissedBlocks{0}
    , mKVFactor{mCacheType == CacheType::kSELFKONLY ? 1 : 2}
    , mLogPrefix{tensorrt_llm::common::fmtstr("BlockManager[windowSize=%u]", mWindowSize)}
    , mReusedTokens{0.0}
    , mTotalInputTokens{0.0}
    , mEnableHashKey{enableHashKey}
    , mEnablePartialReuse{enablePartialReuse}
    , mCopyOnPartialReuse{copyOnPartialReuse}
{
    std::map<SizeType32, SizeType32> numLayersPerPool;

    for (auto const layerIdx : managedLayers)
    {
        auto const& layerIndexWithinPool = numLayersPerPool[numKvHeadsPerLayer.at(layerIdx)]++;
        mLayerToIndexWithinPool[layerIdx] = layerIndexWithinPool;
    }

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
        mPools.emplace_back(numLayers, mKVFactor, numKvHeads, sizePerHead, tokensPerBlock, 1);
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
        auto cacheBlockIds = sequence.getCacheBlockIds(windowSize);
        auto const& uniqueTokens = llmRequest.getUniqueTokens(beamIdx);

        auto blockedUniqueTokens
            = chopVectorIntoBlocks<UniqueToken>(uniqueTokens, uniqueTokens.size() - 1, getTokensPerBlock(), false);
        auto blockKeys = buildBlockKeys(blockedUniqueTokens, llmRequest);
        storeBlocks(std::move(blockKeys), cacheBlockIds[beamIdx], windowSize);
    }
}

void WindowBlockManager::createBlockScalePools(SizeType32 quantBlockSize)
{
    auto num_pools = mPools.size();
    for (size_t i = 0; i < num_pools; ++i)
    {
        auto& kv_pool = mPools[i];
        TLLM_CHECK_WITH_INFO(kv_pool.blockSize % quantBlockSize == 0,
            "Cannot use FP4 quantization since kv_pool.blockSize is not divisible by FP4 quantBlockSize.");

        mPools.emplace_back(kv_pool.numLayers, kv_pool.kvFactor, kv_pool.numKvHeads, kv_pool.sizePerHead,
            kv_pool.tokensPerBlock, quantBlockSize,
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
            TLLM_CHECK_WITH_INFO(blockSize % 2 == 0, "Block size must be divisible by 2 for FP4 KV cache.");
            // Divide by 2. We can't create FP4 buffers directly, so we'll have to create a uint8 buffer with
            // half the expected number of elements.
            blockSize /= 2;
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

void WindowBlockManager::claimLeafBlock(BlockPtr const& block, std::optional<executor::RetentionPriority> priority,
    std::optional<std::chrono::milliseconds> durationMs)
{
    // The eviction policy needs blocks to still be linked to their old parents when they're reclaimed.
    // This is so it can check if the parent should be queued for eviction.
    mEvictionPolicy->claimBlock(block, priority, durationMs);
    block->freeLeafBlock();
}

void WindowBlockManager::freeChildren(
    BlockPtr const& block, executor::RetentionPriority priority, std::optional<std::chrono::milliseconds> durationMs)
{
    // Free all descendants of block
    for (auto const& p : block->getNextBlocks())
    {
        auto childBlock = p.second;
        freeChildren(childBlock, priority, durationMs);
    }

    // Free block
    if (mEventManager && blockInRadixTree(block))
    {
        mEventManager->enqueueRemovedEvent(block, mWindowSize);
    }

    claimLeafBlock(block, priority, durationMs);
}

BlockPtr WindowBlockManager::getFreeBlock(
    executor::RetentionPriority priority, std::optional<std::chrono::milliseconds> durationMs)
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
        // If we're swapping a block to secondary memory, maintain the prior priority values.
        mEvictionPolicy->claimBlock(block);
        // Offload block in primary memory before repurposing
        auto offloadBlock = std::get<0>(mEvictionPolicy->getFreeBlock(kSecondaryLevel));
        mTransferManager->offload(block, offloadBlock, mPools);
        // swap linear block offsets (i.e. make block the offload block)
        block->swapMemoryPoolBlockOffset(offloadBlock);

        if (mEventManager && blockInRadixTree(block))
        {
            mEventManager->enqueueUpdatedEvent(
                tle::KVCacheUpdatedData(block->getHash()).cacheLevelUpdated(kPrimaryLevel, kSecondaryLevel),
                mWindowSize);
        }
        mEvictionPolicy->releaseBlock(block); // append offload block to mFreeSecondaryBlocks queue
        block = offloadBlock;
    }

    // Ensure that returned block is a leaf block by freeing all it's children.
    // Most blocks returned by mEvictionPolicy->getFreeBlock will be leaf blocks,
    // but there are situations where they are not. One example is when a primary
    // block has children in secondary memory and offloading is not possible
    // because there are no free secondary blocks.
    freeChildren(block, priority, durationMs);

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

void WindowBlockManager::addBlockToHashMap(BlockPtr const& block)
{
    if (!mEnableHashKey)
    {
        return;
    }
    auto range = mContextBlocksByHash.equal_range(block->getHash());
    for (auto it = range.first; it != range.second; ++it)
    {
        if (it->second == block)
        {
            // TODO: change to assert when reused block is added only once
            TLLM_LOG_TRACE(
                "Block %d by %zx exists", block->getBlockId(), block->getHash(), mContextBlocksByHash.size());
            return;
        }
    }
    TLLM_LOG_TRACE(
        "Add block %d by %zx, block n = %zu", block->getBlockId(), block->getHash(), mContextBlocksByHash.size());
    mContextBlocksByHash.emplace(block->getHash(), std::move(block));
}

void WindowBlockManager::removeBlockFromHashMap(BlockPtr const& block)
{
    if (mContextBlocksByHash.empty() || block->getBlockKey().uniqueTokens.empty())
    {
        // Hash key not enabled / Empty block
        return;
    }
    auto range = mContextBlocksByHash.equal_range(block->getHash());
    TLLM_LOG_TRACE(
        "Remove block %d by %zx, block n = %zu", block->getBlockId(), block->getHash(), mContextBlocksByHash.size());
    for (auto it = range.first; it != range.second; ++it)
    {
        if (it->second == block)
        {
            mContextBlocksByHash.erase(it);
            return;
        }
    }
    // TODO: should be unreachable
    TLLM_LOG_DEBUG("Trying to remove block %d by %zx that is not in hash map", block->getBlockId(), block->getHash());
}

void BlockManager::onboardBlock(BlockPtr const& offloadBlock, SizeType32 windowSize)
{
    mWindowBlockManagers.at(windowSize).onboardBlock(offloadBlock);
}

void WindowBlockManager::onboardBlock(BlockPtr const& offloadBlock)
{
    if (mOnboardBlocks && !offloadBlock->isPrimary())
    {
        auto block = getFreeBlock();
        mTransferManager->onboard(offloadBlock, block, mPools);
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

void BlockManager::offloadBlock(BlockPtr const& block, SizeType32 windowSize)
{
    mWindowBlockManagers.at(windowSize).offloadBlock(block);
}

void WindowBlockManager::offloadBlock(BlockPtr const& block)
{
    if (mOnboardBlocks && block->isPrimary())
    {
        // Offload block in primary memory before repurposing
        auto offloadBlock = std::get<0>(mEvictionPolicy->getFreeBlock(kSecondaryLevel));
        // If we're swapping a block to secondary memory, maintain the prior priority values.
        mEvictionPolicy->claimBlock(offloadBlock);
        mTransferManager->offload(block, offloadBlock, mPools);
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

SizeType32 WindowBlockManager::loadOrAllocateBlocks(std::vector<BlockKey> const& blockKeys, SizeType32 numContextBlocks,
    GenerationRequest& sequence, std::vector<executor::RetentionPriorityAndDuration> const& perBlockRetentions)
{
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
                    auto newBlock = getFreeBlock(matchingBlock->getPriority(), matchingBlock->getDurationMs());
                    mTransferManager->onboard(matchingBlock, newBlock, mPools, numMatched);
                    // TODO: (optional) Send out event
                    matchingBlock = newBlock;
                    TLLM_LOG_DEBUG("%s::loadOrAllocateBlocks - Copied partially filled block %d", mLogPrefix.c_str(),
                        matchingBlockId);
                }
                else
                {
                    // Leaf block that nobody is using. Make block private and reuse
                    claimLeafBlock(
                        matchingBlock, perBlockRetentions[bi].retentionPriority, perBlockRetentions[bi].durationMs);
                    TLLM_LOG_DEBUG("%s::loadOrAllocateBlocks - Reused partially filled block %d", mLogPrefix.c_str(),
                        matchingBlockId);
                    addBlockToHashMap(matchingBlock);
                }
                searchRoot = nullptr; // no matching needed for following blocks
            }
            else
            {
                // Recover block and reuse
                mEvictionPolicy->claimBlock(
                    matchingBlock, perBlockRetentions[bi].retentionPriority, perBlockRetentions[bi].durationMs);
                TLLM_LOG_DEBUG("%s::loadOrAllocateBlocks - Matched full block %d", mLogPrefix.c_str(), matchingBlockId);
                addBlockToHashMap(matchingBlock);
                searchRoot = matchingBlock;
            }
            onboardBlock(matchingBlock);
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
            auto freeBlock = getFreeBlock(perBlockRetentions[bi].retentionPriority.value_or(
                                              executor::KvCacheRetentionConfig::kDefaultRetentionPriority),
                perBlockRetentions[bi].durationMs);
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
            addBlockToHashMap(freeBlock);
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
            auto freeBlock = getFreeBlock(perBlockRetentions[bi].retentionPriority.value_or(
                                              executor::KvCacheRetentionConfig::kDefaultRetentionPriority),
                perBlockRetentions[bi].durationMs);
            addBlockToBeam(freeBlock, sequence, beamIdx);
            if (blockItr != blockKeys.end())
            {
                freeBlock->setBlockKey(
                    *blockItr, blockItr->uniqueTokens.size() == static_cast<size_t>(mTokensPerBlock));
                ++blockItr;
            }
            freeBlock->setHash();
            addBlockToHashMap(freeBlock);
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

void BlockManager::addSequence(GenerationRequest& sequence, SizeType32 inputLength, SizeType32 numContextBlocks,
    LlmRequest& llmRequest, OptionalRef<kv_connector::KvCacheConnectorManager> kvCacheConnectorManager,
    SizeType32 windowSize)
{
    mWindowBlockManagers.at(windowSize)
        .addSequence(sequence, inputLength, numContextBlocks, llmRequest, kvCacheConnectorManager);
}

void WindowBlockManager::addSequence(GenerationRequest& sequence, SizeType32 inputLength, SizeType32 numContextBlocks,
    LlmRequest& llmRequest, OptionalRef<kv_connector::KvCacheConnectorManager> kvCacheConnectorManager)
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

    TLLM_CHECK(perBlockRetentions.size() == (size_t) numContextBlocks);

    auto const prepopulatedPromptLen = loadOrAllocateBlocks(blockKeys, numContextBlocks, sequence, perBlockRetentions);
    mReusedTokens += static_cast<double>(prepopulatedPromptLen);
    mTotalInputTokens += static_cast<double>(uniqueTokens.size());

    SizeType32 numConnectorMatchedTokens = 0;

    // If we're using a KV cache connector, check if any additional blocks can be loaded.
    if (kvCacheConnectorManager)
    {
        numConnectorMatchedTokens = kvCacheConnectorManager->getNumNewMatchedTokens(llmRequest, prepopulatedPromptLen);
        TLLM_CHECK_WITH_INFO(prepopulatedPromptLen + numConnectorMatchedTokens < llmRequest.getPromptLen(),
            "There must be at least one uncomputed token in the prompt!");
    }

    llmRequest.setPrepopulatedPromptLen(prepopulatedPromptLen + numConnectorMatchedTokens, getTokensPerBlock());
    TLLM_LOG_DEBUG("addSequence: Request %lu, inputLength %d, prepopulatedPromptLen %d, numConnectorMatchedTokens %d",
        llmRequest.mRequestId, inputLength, prepopulatedPromptLen, numConnectorMatchedTokens);
}

void BlockManager::addSequence(
    GenerationRequest& sequence, SizeType32 numBlocks, SizeType32 unsharedBlockIdx, SizeType32 windowSize)
{
    mWindowBlockManagers.at(windowSize).addSequence(sequence, numBlocks, unsharedBlockIdx);
}

void WindowBlockManager::addSequence(GenerationRequest& sequence, SizeType32 numBlocks, SizeType32 unsharedBlockIdx)
{
    auto const requestId = sequence.getRequestId();
    auto const [seqIt, emplaceDone] = mAllocatedBlocksPerSeq.emplace(requestId, std::vector<BlockPtr>{});
    TLLM_CHECK(emplaceDone);

    // Allocate blocks
    for (SizeType32 bi = 0; bi < numBlocks; ++bi)
    {
        bool shareAmongBeams = bi != unsharedBlockIdx;
        allocateBlock(sequence, shareAmongBeams);
    }
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
        auto block = getFreeBlock(sequence.getDecodeRetentionPriority(), sequence.getDecodeDurationMs());
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
            auto block = getFreeBlock(sequence.getDecodeRetentionPriority(), sequence.getDecodeDurationMs());
            addBlockToBeam(block, sequence, beamIdx);
        }
    }
}

void WindowBlockManager::storeBlocks(
    std::vector<BlockKey> const& blockKeys, std::vector<KVCacheBlock::IdType> const& blockIds)
{
    TLLM_LOG_DEBUG(
        "%s::storeBlocks - %zu blockKeys, %zu blockIds", mLogPrefix.c_str(), blockKeys.size(), blockIds.size());

    auto searchRoot = mCachedBlocksRoot;
    bool needMatch = true;

    auto numBlocks = blockKeys.size();
    std::vector<BlockPtr> storedBlocks;
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
                removeBlockFromHashMap(block);
                block->setHash(newHash);
                addBlockToHashMap(block);
            }
            searchRoot = block;
        }
    }
    if (mEventManager)
    {
        mEventManager->enqueueStoredEvent(storedBlocks, mWindowSize);
    }
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
            removeBlockFromHashMap(block);
        }
    }

    // Allocate new blocks
    TLLM_CHECK_WITH_INFO(hasFreeBlocks(beamWidth), "Can't allocate new blocks. No free blocks left.");
    for (auto beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        auto block = getFreeBlock();
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

std::vector<KVCacheBlock::IdType> BlockManager::getNewlyAllocatedBlockIds(
    GenerationRequest const& sequence, SizeType32 windowSize) const
{
    return mWindowBlockManagers.at(windowSize).getNewlyAllocatedBlockIds(sequence);
}

std::vector<KVCacheBlock::IdType> WindowBlockManager::getNewlyAllocatedBlockIds(GenerationRequest const& sequence) const
{
    std::vector<KVCacheBlock::IdType> allocatedBlockIds;
    for (auto const& beamBlockIds : sequence.getCacheBlockIds(mWindowSize))
    {
        for (auto const& blockId : beamBlockIds)
        {
            auto const& block = mAllBlocksById.at(blockId);
            if (!blockInRadixTree(block))
            {
                allocatedBlockIds.push_back(blockId);
            }
        }
    }
    return allocatedBlockIds;
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
        removeBlockFromHashMap(block);
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

void BlockManager::releaseBlocks(GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest)
{
    // When releasing the blocks for a sequence, we store those blocks for potential reuse only if:
    // - Block reuse is enabled.
    // - A request was provided to this function call to identify which tokens these blocks cover
    // - Beam search is NOT enabled <=> beam width == 1
    // - The sequence was not marked for use with cyclic kv-cache when it was added (when its context is too long to fit
    // the max attention window).
    // - The sequence did not switch to cyclic kv-cache during generation phase.
    //  A sequence is cyclic if its *minimum window size* is crossed, even if other window sizes were not reached.
    // - The sequence is not a dummy request.
    bool const storeBlocksForReuse = sequence.getBeamWidth() == 1 && llmRequest.has_value() && !sequence.isCyclic()
        && !llmRequest->isDummyRequest();
    for (auto& [_, manager] : mWindowBlockManagers)
    {
        if (storeBlocksForReuse)
        {
            manager.storeBlocksForReuse(sequence, llmRequest);
        }
        manager.releaseBlocks(sequence);
    }
}

void BlockManager::storeNewBlock(GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest)
{
    // we store newest block for potential reuse only if:
    // - Block reuse is enabled.
    // - A request was provided to this function call to identify which tokens these blocks cover
    // - Beam search is NOT enabled <=> beam width == 1
    // - The sequence was not marked for use with cyclic kv-cache when it was added (when its context is too long to fit
    // the max attention window).
    // - The sequence did not switch to cyclic kv-cache during generation phase.
    //  A sequence is cyclic if its *minimum window size* is crossed, even if other window sizes were not reached.
    bool const storeBlocksForReuse = sequence.getBeamWidth() == 1 && llmRequest.has_value() && !sequence.isCyclic();
    if (!storeBlocksForReuse)
    {
        return;
    }
    for (auto& [_, manager] : mWindowBlockManagers)
    {
        manager.storeNewBlock(sequence, llmRequest);
    }
}

[[nodiscard]] std::vector<kv_connector::KvCacheConnectorPoolData> BlockManager::getKvCacheConnectorPoolsData() const
{
    TLLM_CHECK_WITH_INFO(
        mWindowBlockManagers.size() == 1, "KV Cache connector is not supported with multiple window sizes");
    std::vector<kv_connector::KvCacheConnectorPoolData> poolsData;
    poolsData.reserve(1);
    for (auto const& [_, manager] : mWindowBlockManagers)
    {
        poolsData.emplace_back(manager.getKvCacheConnectorPoolData());
    }
    return poolsData;
}

[[nodiscard]] kv_connector::KvCacheConnectorPoolData WindowBlockManager::getKvCacheConnectorPoolData() const
{
    return kv_connector::KvCacheConnectorPoolData(mPools.at(0).primaryPtr, mNumPrimaryBlocks);
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
        storeBlocks(std::move(blockKeys), cacheBlockIds[beamIdx]);
        return;
    }

    auto lastBlock = mAllBlocksById.at(cacheBlockIds[beamIdx][blockKeys.size() - 1]);
    auto prevBlock = mAllBlocksById.at(cacheBlockIds[beamIdx][blockKeys.size() - 2]);

    // If the previous block is not in the radix tree, we need to store all blocks
    if (prevBlock->getPrevBlock() == nullptr)
    {
        TLLM_LOG_DEBUG("%s::storeNewBlock - store all blocks", mLogPrefix.c_str());
        storeBlocks(std::move(blockKeys), cacheBlockIds[beamIdx]);
        return;
    }

    if (lastBlock->getPrevBlock() != nullptr)
    {
        // If the last block is not in the radix tree, we need to store all blocks
        TLLM_LOG_DEBUG("%s::storeNewBlock - no need to store", mLogPrefix.c_str());
        return;
    }
    TLLM_LOG_DEBUG("%s::storeNewBlock - store the last block", mLogPrefix.c_str());
    storeBlocks(std::move(blockKeys), cacheBlockIds[beamIdx]);
}

void WindowBlockManager::storeBlocksForReuse(GenerationRequest& sequence, OptionalRef<LlmRequest const> llmRequest)
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
    storeBlocks(std::move(blockKeys), cacheBlockIds[beamIdx]);
}

void WindowBlockManager::releaseBlocks(GenerationRequest& sequence)
{
    auto const requestId = sequence.getRequestId();

    auto node = mAllocatedBlocksPerSeq.extract(requestId);
    TLLM_CHECK(node);
    auto& allocatedBlocks = node.mapped();
    for (auto it = allocatedBlocks.rbegin(); it != allocatedBlocks.rend(); ++it)
    {
        auto& block = *it;
        // Decrease ref count
        block->decRefCount();
        // If ref count is zero, move block to free blocks
        if (!block->hasRefs())
        {
            mEvictionPolicy->releaseBlock(block);
            removeBlockFromHashMap(block);
        }
    }
    // Remove stored block ids in sequence
    sequence.clearCacheBlocks(mWindowSize);
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
    SizeType32 sinkTokenLength, int64_t stream, std::optional<runtime::SizeType32> maxSequenceLength,
    bool enableBlockReuse, bool onboardBlocks, CacheType cacheType, bool enablePartialReuse, bool copyOnPartialReuse)
    : KVCacheManager(std::vector<SizeType32>(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, maxBeamWidth, maxAttentionWindowVec, tempAttentionWindowInputs, dtype, sinkTokenLength,
        std::make_shared<runtime::CudaStream>(reinterpret_cast<cudaStream_t>(stream)), maxSequenceLength,
        enableBlockReuse, onboardBlocks, cacheType, std::nullopt, nullptr, false, enablePartialReuse,
        copyOnPartialReuse)
{
}

KVCacheManager::KVCacheManager(std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead,
    SizeType32 tokensPerBlock, BlocksPerWindow const& blocksPerWindow, SizeType32 maxNumSequences,
    SizeType32 maxBeamWidth, std::vector<SizeType32> const& maxAttentionWindowVec,
    std::optional<TempAttentionWindowInputs> const& tempAttentionWindowInputs, nvinfer1::DataType dtype,
    SizeType32 sinkTokenLength, int64_t stream, std::optional<runtime::SizeType32> maxSequenceLength,
    bool enableBlockReuse, bool onboardBlocks, CacheType cacheType,
    std::optional<executor::RetentionPriority> secondaryOffloadMinPriority,
    std::shared_ptr<KVCacheEventManager> eventManager, bool enablePartialReuse, bool copyOnPartialReuse)
    : KVCacheManager(numKvHeadsPerLayer, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences, maxBeamWidth,
        maxAttentionWindowVec, tempAttentionWindowInputs, dtype, sinkTokenLength,
        std::make_shared<runtime::CudaStream>(reinterpret_cast<cudaStream_t>(stream)), maxSequenceLength,
        enableBlockReuse, onboardBlocks, cacheType, secondaryOffloadMinPriority, eventManager, enablePartialReuse,
        copyOnPartialReuse)
{
}

KVCacheManager::KVCacheManager(std::vector<SizeType32> const& numKvHeadsPerLayer, SizeType32 sizePerHead,
    SizeType32 tokensPerBlock, BlocksPerWindow const& blocksPerWindow, SizeType32 maxNumSequences,
    SizeType32 maxBeamWidth, std::vector<SizeType32> const& maxAttentionWindowVec,
    std::optional<TempAttentionWindowInputs> const& tempAttentionWindowInputs, nvinfer1::DataType dtype,
    SizeType32 sinkTokenLength, CudaStreamPtr stream, std::optional<runtime::SizeType32> maxSequenceLength,
    bool enableBlockReuse, bool onboardBlocks, CacheType cacheType,
    std::optional<executor::RetentionPriority> secondaryOffloadMinPriority,
    std::shared_ptr<KVCacheEventManager> eventManager, bool enableHashKey, bool enablePartialReuse,
    bool copyOnPartialReuse)
    : mMaxBeamWidth(maxBeamWidth)
    , mDataType(dtype)
    , mMaxAttentionWindow(*std::max_element(maxAttentionWindowVec.begin(), maxAttentionWindowVec.end()))
    , mTokensPerBlock(tokensPerBlock)
    , mSinkBubbleLength(BaseKVCacheManager::getSinkBubbleLength(sinkTokenLength, tokensPerBlock))
    , mSinkBlockTokenLength(mSinkBubbleLength + sinkTokenLength)
    , mBlockManager(numKvHeadsPerLayer, sizePerHead, tokensPerBlock, blocksPerWindow, maxNumSequences,
          std::move(stream), maxSequenceLength, maxBeamWidth, maxAttentionWindowVec, tempAttentionWindowInputs, dtype,
          mSinkBubbleLength, onboardBlocks, cacheType, secondaryOffloadMinPriority, std::move(eventManager),
          enableHashKey, enablePartialReuse, copyOnPartialReuse)
    // disable block reuse for sink bubble since chopVectorIntoBlocks does not match KV cache blocks in this case
    , mEnableBlockReuse{mSinkBubbleLength > 0 ? false : enableBlockReuse}
    , mEnableHashKey{enableHashKey}
{
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
    SizeType32 sinkTokenLength, CudaStreamPtr stream, std::optional<runtime::SizeType32> maxSequenceLength,
    bool enableBlockReuse, bool onboardBlocks, CacheType cacheType,
    std::optional<executor::RetentionPriority> secondaryOffloadMinPriority,
    std::shared_ptr<KVCacheEventManager> eventManager, bool enableHashKey, bool enablePartialReuse,
    bool copyOnPartialReuse)
    : KVCacheManager(std::vector<SizeType32>(numLayers, numKvHeads), sizePerHead, tokensPerBlock, blocksPerWindow,
        maxNumSequences, maxBeamWidth, maxAttentionWindowVec, tempAttentionWindowInputs, dtype, sinkTokenLength,
        std::move(stream), maxSequenceLength, enableBlockReuse, onboardBlocks, cacheType, secondaryOffloadMinPriority,
        std::move(eventManager), enableHashKey, enablePartialReuse, copyOnPartialReuse)
{
}

void KVCacheManager::allocatePools(bool useUvm)
{
    mBlockManager.allocatePools(useUvm);
    auto const numPools = mBlockManager.getNumPools();

    if (tc::Logger::getLogger()->getLevel() <= tc::Logger::INFO)
    {
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
        auto const maxTokensToAddToKVCache = req.mMaxNewTokens;
        auto const maxDraftTokensToAdd = std::min(req.getNumDraftTokens(), maxTokensToAddToKVCache);
        // Assumes shared among beam = True
        auto const promptCacheLen
            = std::min((isCrossKv() ? req.getEncoderOutputLen() : req.mPromptLen) + maxDraftTokensToAdd, windowSize)
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

    if (numAllocBlocksPerBeam < numContextBlocks)
    {
        return numContextBlocks - numAllocBlocksPerBeam + numGenBlocksPerBeam * req.mSamplingConfig.beamWidth;
    }

    return (numTotalBlocksPerBeam - numAllocBlocksPerBeam) * req.mSamplingConfig.beamWidth;
}

void KVCacheManager::cacheBlockOffsets(GenerationRequest& sequence, SizeType32 windowSize)
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
            mBlockManager.setOffsets(offsetsPtr, offsetsShape, beamIdx, blockIdx, blockId, windowSize);
        }
    }
}

void KVCacheManager::cacheNewBlockOffsets(GenerationRequest& sequence, SizeType32 windowSize)
{
    auto const& cacheBlocks = sequence.getCacheBlockIds(windowSize);
    auto& cacheBlocksTensor = sequence.getCacheBlockIndices(windowSize);
    auto const beamWidth = sequence.getBeamWidth();

    auto* offsetsPtr = bufferCast<tk::KVCacheIndex>(cacheBlocksTensor);
    auto const& offsetsShape = cacheBlocksTensor.getShape();

    for (SizeType32 beamIdx = 0; beamIdx < beamWidth; ++beamIdx)
    {
        auto const& beamCacheBlock = cacheBlocks[beamIdx];
        auto const blockId = beamCacheBlock.back();
        auto const blockIdx = static_cast<SizeType32>(beamCacheBlock.size() - 1);
        mBlockManager.setOffsets(offsetsPtr, offsetsShape, beamIdx, blockIdx, blockId, windowSize);
    }
}

void KVCacheManager::updateNewBlockPointer(GenerationRequest& sequence, SizeType32 windowSize, SizeType32 blockIdx)
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
        mBlockManager.setOffsets(offsetsPtr, offsetsShape, beamIdx, blockIdx, blockId, windowSize);
    }
}

std::optional<SizeType32> KVCacheManager::updateToken(GenerationRequest& sequence, bool addToken)
{
    auto currNumTokens = sequence.getNumTokens();

    if (addToken)
    {
        sequence.addNewTokens(1);
    }
    else
    {
        sequence.removeTokens(1);
    }

    auto newNumTokens = sequence.getNumTokens();

    if (!addToken)
    {
        std::swap(currNumTokens, newNumTokens);
    }

    for (auto const [windowSize, metadata] : mBlockManager.getWindowSizesMetadata())
    {
        auto const maxTokenNum = metadata.maxTokenNum;
        SizeType32 const cyclicTokenNum = maxTokenNum - mSinkBlockTokenLength;
        SizeType32 const nextTokenIdxInCycle = (currNumTokens - mSinkBlockTokenLength) % cyclicTokenNum;
        SizeType32 const nextTokenIdxInCache = mSinkBlockTokenLength + nextTokenIdxInCycle;

        // (nextTokenIdxInCache - mSinkBlockTokenLength) % cyclicTokenNum == 0)
        // <=> nextTokenIdxInCycle == 0
        // <=> nextTokenIdxInCache == mSinkBlockTokenLength
        // => nextTokenIdxInCache % getTokensPerBlock() == 0

        // Check if require a new block
        if (nextTokenIdxInCache % getTokensPerBlock() == 0)
        {
            if (newNumTokens <= maxTokenNum)
            {
                if (addToken)
                {
                    mBlockManager.allocateBlock(sequence, windowSize);
                    cacheNewBlockOffsets(sequence, windowSize);
                }
                else
                {
                    mBlockManager.releaseLastBlock(sequence, windowSize);
                }
            }
            else if (sequence.getBeamWidth() > 1)
            {
                TLLM_CHECK_WITH_INFO(addToken, "Remove token is not supported with beam search");
                // Get next block index
                SizeType32 nextBlockIdx = nextTokenIdxInCache / getTokensPerBlock();
                // Replace the shared block with the unshared ones
                mBlockManager.replaceSharedBlock(sequence, windowSize, nextBlockIdx);
                updateNewBlockPointer(sequence, windowSize, nextBlockIdx);
            }
        }
    }

    return std::nullopt;
}

std::optional<SizeType32> KVCacheManager::addToken(RequestIdType requestId)
{
    auto& sequence = getSequence(requestId);
    return updateToken(sequence, true);
}

std::optional<BlockKey> KVCacheManager::findNewContextBlock(
    VecUniqueTokens const& uniqueTokens, LlmRequest const& llmRequest) const
{
    auto newContextBlockOpt = mBlockManager.findNewContextBlock(uniqueTokens, llmRequest);
    return newContextBlockOpt;
}

void KVCacheManager::addSequence(RequestIdType requestId, SizeType32 inputLength, SizeType32 beamWidth,
    OptionalRef<LlmRequest> llmRequest, OptionalRef<kv_connector::KvCacheConnectorManager> kvCacheConnectorManager)
{
    // Need to add the bubble after the sink tokens to use even block size
    inputLength += mSinkBubbleLength;

    if (kvCacheConnectorManager)
    {
        TLLM_CHECK_WITH_INFO(beamWidth == 1, "KV Cache Connector is not supported with beam search");
    }

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

    for (auto const [windowSize, metadata] : mBlockManager.getWindowSizesMetadata())
    {
        auto const maxTokenNum = metadata.maxTokenNum;
        auto const temporaryAttentionWindow = metadata.temporaryAttentionWindow;

        // Get the final token index in kv cache
        SizeType32 const finalTokenKVIdx = mSinkBlockTokenLength
            + ((inputLength - 1 - mSinkBlockTokenLength) % (maxTokenNum - mSinkBlockTokenLength));

        // Get block index that with shareAmongBeams=False.
        // For cross kv cache in encoder-decoder models, always shareAmongBeams=True.
        SizeType32 unsharedBlockIdx = -1;
        if ((!sequence.isCyclic() || beamWidth > 1 || finalTokenKVIdx % getTokensPerBlock() > 0) && !isCrossKv())
        {
            unsharedBlockIdx = ((finalTokenKVIdx + 1) % getTokensPerBlock() == 0)
                ? finalTokenKVIdx / getTokensPerBlock() + 1
                : finalTokenKVIdx / getTokensPerBlock();
        }

        // Consider the temporaryAttentionWindow when allocating blocks.
        auto const effectiveInputLength = std::min(inputLength, maxTokenNum + temporaryAttentionWindow);
        auto const numContextBlocks = tc::ceilDiv(effectiveInputLength, getTokensPerBlock());
        if (!sequence.isCyclic() && mEnableBlockReuse)
        {
            mBlockManager.addSequence(
                sequence, effectiveInputLength, numContextBlocks, *llmRequest, kvCacheConnectorManager, windowSize);
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
                if (kvCacheConnectorManager.has_value())
                {
                    TLLM_LOG_WARNING(
                        "KV Cache Connector specified when block reuse is disabled. The KV Cache Connector will be "
                        "ignored.");
                }
            }
            mBlockManager.addSequence(sequence, numContextBlocks, unsharedBlockIdx, windowSize);
            if (mEnableHashKey && llmRequest.has_value() && beamWidth == 1)
            {
                constexpr SizeType32 beamIdx = 0;
                auto const& blockIds = sequence.getCacheBlockIds(windowSize).at(beamIdx);
                auto const& uniqueTokens = llmRequest->getUniqueTokens(beamIdx);
                auto blockedUniqueTokens = chopVectorIntoBlocks<UniqueToken>(
                    uniqueTokens, uniqueTokens.size() - 1, getTokensPerBlock(), true);
                auto blockKeys = buildBlockKeys(blockedUniqueTokens, *llmRequest);
                auto tokensPerBlock = static_cast<size_t>(getTokensPerBlock());
                for (size_t i = 0; i < blockIds.size(); i++)
                {
                    auto const& block = mBlockManager.getBlockById(blockIds[i], windowSize);
                    if (i < blockKeys.size())
                    {
                        block->setBlockKey(blockKeys[i], blockKeys[i].uniqueTokens.size() == tokensPerBlock);
                    }
                    else
                    {
                        block->setBlockKey({}, false);
                    }
                    block->setHash();
                    mBlockManager.addBlockToHashMap(block, windowSize);
                }
            }
        }
        cacheBlockOffsets(sequence, windowSize);
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
    auto& sequence = getSequence(requestId);
    if (mEnableBlockReuse && !sequence.isCyclic() && !llmRequest.isDummyRequest())
    {
        mBlockManager.storeContextBlocks(sequence, llmRequest);
    }
}

void KVCacheManager::storeNewBlock(LlmRequest const& llmRequest)
{
    auto const requestId = llmRequest.mRequestId;
    auto& sequence = getSequence(requestId);
    bool const storeBlocksForReuse = sequence.getBeamWidth() == 1 && !sequence.isCyclic();
    if (mEnableBlockReuse && storeBlocksForReuse)
    {
        mBlockManager.storeNewBlock(sequence, llmRequest);
    }
}

void KVCacheManager::removeSequence(RequestIdType requestId, OptionalRef<LlmRequest const> llmRequest)
{
    TLLM_LOG_TRACE("[%s]::%s start", isCrossKv() ? "CROSS" : "SELF", __PRETTY_FUNCTION__);
    auto sequenceNode = [this, requestId]
    {
        std::scoped_lock lock(mSequencesMtx);
        return mSequences.extract(requestId);
    }();
    if (!sequenceNode.empty())
    {
        // Free all blocks for this sequence
        if (mEnableBlockReuse)
        {
            mBlockManager.releaseBlocks(sequenceNode.mapped(), llmRequest);
        }
        else
        {
            mBlockManager.releaseBlocks(sequenceNode.mapped(), std::nullopt);
        }
    }
    TLLM_LOG_TRACE("[%s]::%s stop", isCrossKv() ? "CROSS" : "SELF", __PRETTY_FUNCTION__);
}

void KVCacheManager::schedulingRemoveSequence(RequestIdType requestId)
{
    // Mimic Free all blocks for this sequence
    mBlockManager.schedulingReleaseBlocks(requestId);
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

    TLLM_LOG_DEBUG("extraCostMemory [Gib]: %0.2f", extraCostMemory / static_cast<double>(1 << 30));
    allottedPrimaryMemBytes = allottedPrimaryMemBytes - extraCostMemory;
    auto const tokensPerBlock = modelConfig.getTokensPerBlock();
    auto const calculatePrimaryBlocks
        = [&](SizeType32 windowSize, float windowSizeShare, SizeType32 cacheSizeBytesPerToken)
    {
        TLLM_LOG_DEBUG("windowSizeShare: %f, cacheSizeBytesPerToken: %d", windowSizeShare, cacheSizeBytesPerToken);
        auto maxTokens = static_cast<uint64_t>(
            allottedPrimaryMemBytes * windowSizeShare / static_cast<double>(cacheSizeBytesPerToken));
        if (config.getMaxTokens().has_value())
        {
            auto const maxTokensFromConfig = static_cast<uint64_t>(config.getMaxTokens().value());
            TLLM_LOG_DEBUG("Maximum kv-cache token overridden by configuration as '%ld'.", maxTokensFromConfig);
            maxTokens = std::min(maxTokensFromConfig, maxTokens);
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

    auto const windowSizeToShare
        = BlockManager::calculateWindowSizeToShare(windowSizeToLayers, cacheSizeBytesPerTokenPerWindow);

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
    auto& sequence = getSequence(requestId);
    auto const beamWidth = sequence.getBeamWidth();

    TLLM_CHECK_WITH_INFO(beamWidth == 1, "removeToken does not support beamWidth > 1");
    if (sequence.getNumTokens() == 0)
    {
        return;
    }
    updateToken(sequence, false);
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

std::vector<SizeType32> KVCacheManager::getNewlyAllocatedBlockIds(
    LlmRequest::RequestIdType requestId, SizeType32 windowSize) const
{
    return mBlockManager.getNewlyAllocatedBlockIds(getSequence(requestId), windowSize);
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
    return tc::ceilDiv(actualMaxTokenNum, tokensPerBlock);
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

    // If the whole attention window can fit in the output, then we can simply multiply the cost of a sequence of
    // length max attention window by the beam width.
    if (windowSize <= outputLength)
    {
        return KVCacheManager::calculateMaxBlockRequirementsPerBeam(
                   windowSize, sinkTokenLength, windowSize, tokensPerBlock)
            * beamWidth;
    }

    // Otherwise, we calculate how many tokens will be in output blocks.
    auto const effectiveAttentionWindow = std::min(windowSize, wholeSequenceLength);
    auto const numContextTokensInAttentionWindow
        = effectiveAttentionWindow - outputLength; // This is positive because we handled the other case above.
    auto const sinkBubbleLength = BaseKVCacheManager::getSinkBubbleLength(sinkTokenLength, tokensPerBlock);
    auto const numContextBlocks = (numContextTokensInAttentionWindow + sinkBubbleLength) / tokensPerBlock;
    auto const leftoverContextToken = numContextTokensInAttentionWindow - numContextBlocks * tokensPerBlock;
    auto const numOutputBlocks = tc::ceilDiv(outputLength + leftoverContextToken, tokensPerBlock);
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

[[nodiscard]] kv_connector::KvCacheConnectorPoolsData KVCacheManager::getKvCacheConnectorPoolsData() const
{
    auto poolsData = mBlockManager.getKvCacheConnectorPoolsData();

    auto layerToPoolView = BufferRange<SizeType32>(*mLayerToPoolMapping);

    auto numLayers = mBlockManager.getNumLayers();

    auto layerToPool = std::vector<SizeType32>(numLayers);

    for (size_t layer = 0; layer < static_cast<size_t>(numLayers); layer++)
    {
        layerToPool[layer] = layerToPoolView[layer * 2];
    }

    return kv_connector::KvCacheConnectorPoolsData(poolsData, layerToPool);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager
