/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/medusaModule.h"
#include <stack>
#include <vector>

namespace tensorrt_llm::runtime
{

void MedusaModule::initMedusaTensorsFromChoices(MedusaChoices const& choices, std::vector<SizeType32>& topKs,
    TensorPtr& generationInputLengths, TensorPtr& positionOffsets, TensorPtr& treeIds, TensorPtr& paths,
    TensorPtr& packedMask, SizeType32& totalPaths) const noexcept
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto const numChoices = static_cast<SizeType32>(choices.size());

    std::vector<SizeType32> sortedIndices(numChoices);
    std::iota(sortedIndices.begin(), sortedIndices.end(), 0);
    std::vector<Prefix> prefixes(numChoices);
    // Fill prefixes for all nodes
    for (SizeType32 ci = 0; ci < numChoices; ++ci)
    {
        auto const& choice = choices[ci];
        prefixes[ci] = computePrefix(choice, choice.size());
    }

    // Sort indices based on depth and prefix
    // Prefix has property that on the same depth, nodes that are more left in the tree have smaller prefix
    std::sort(sortedIndices.begin(), sortedIndices.end(),
        [&prefixes, &choices](SizeType32 const& a, SizeType32 const& b)
        {
            auto const aSize = choices[a].size();
            auto const bSize = choices[b].size();
            return aSize < bSize || (aSize == bSize && prefixes[a] < prefixes[b]);
        });

    dumpChoices(choices, sortedIndices);

    topKs.resize(getMaxDraftPathLen(), 0);
    auto generationInputLengthsPtr = bufferCast<SizeType32>(*generationInputLengths);
    auto positionOffsetsPtr = bufferCast<SizeType32>(*positionOffsets);
    auto treeIdsPtr = bufferCast<SizeType32>(*treeIds);

    // Fixed sequence length currently.
    std::fill(generationInputLengthsPtr, generationInputLengthsPtr + generationInputLengths->getSize(),
        getMaxDecodingTokens());
    std::fill(positionOffsetsPtr, positionOffsetsPtr + positionOffsets->getSize(), -1);
    std::fill(treeIdsPtr, treeIdsPtr + treeIds->getSize(), -1);

    // +1 for root
    std::vector<MedusaTreeNode> tree(choices.size() + 1);
    auto& rootNode = tree[0];
    rootNode.depth = 0;
    // That's how we recognize root
    rootNode.nodeId = -1;

    // Start depth from 1 because we count root node implicetely
    SizeType32 depth = 1;
    // Running max TopK, reset at every new level
    SizeType32 maxTopK = 0;
    // Global node in tree idx is sum of TopKs at previous levels
    SizeType32 globalNodeInTreeIdx = 0;
    // Hash table to map prefix to linear idx at previous level
    std::unordered_map<Prefix, SizeType32> prevPrefixToLinearIdxMap;
    // Hash table to map prefix to linear idx at current level
    std::unordered_map<Prefix, SizeType32> curPrefixToLinearIdxMap;

    // Add root node
    prevPrefixToLinearIdxMap[0] = 0;
    positionOffsetsPtr[0] = 0;

    TLLM_CHECK(numChoices <= getMaxDecodingDraftTokens());

    for (SizeType32 ci = 0; ci < numChoices; ++ci)
    {
        auto const index = sortedIndices[ci];
        // Access choices based on sorting
        auto const& choice = choices[index];
        auto const curDepth = static_cast<SizeType32>(choice.size());

        // If new depth is found (choices are sorted by depth)
        if (curDepth != depth)
        {
            TLLM_CHECK(depth + 1 == curDepth);
            TLLM_CHECK_WITH_INFO(depth <= getMaxDraftPathLen(),
                "Medusa choices require more Medusa heads than the engine was built with.");
            // Save TopK
            topKs[depth - 1] = maxTopK;

            // Accumulate TopK for global indexing in tree
            globalNodeInTreeIdx += maxTopK;

            // Swap hash maps
            prevPrefixToLinearIdxMap = curPrefixToLinearIdxMap;

            // Reset counters
            maxTopK = 0;
            curPrefixToLinearIdxMap.clear();

            // Increment depth
            depth++;
        }

        MedusaTreeNode node;
        node.depth = depth;
        node.linearIdx = ci + 1;
        // nodeId is index in array of sibling nodes
        node.nodeId = choice.back();

        // Compute prefix from current node and store its linear idx to hash map
        curPrefixToLinearIdxMap[prefixes[index]] = node.linearIdx;

        // Compute prefix of the parent node
        auto const parentPrefix = computePrefix(choice, choice.size() - 1);

        // Store parent's linear idx
        node.parentLinearIdx = prevPrefixToLinearIdxMap[parentPrefix];

        // Update TopK
        maxTopK = std::max(maxTopK, node.nodeId + 1);

        // Position offset is the depth of the node
        positionOffsetsPtr[node.linearIdx] = depth;
        // Save tree ids
        treeIdsPtr[node.linearIdx - 1] = globalNodeInTreeIdx + node.nodeId;

        // Save node
        tree[node.linearIdx] = node;
    }

    // Write TopK for the last level
    topKs[depth - 1] = maxTopK;

    for (SizeType32 ci = 0; ci < numChoices + 1; ++ci)
    {
        auto& node = tree[ci];
        // For all nodes except root
        if (node.nodeId != -1)
        {
            // Add current linear index to parent's child
            tree[node.parentLinearIdx].childLinearIndices.push_back(ci);
        }
    }

    totalPaths = computePathsAndMask(tree, packedMask, paths);
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
}

SizeType32 MedusaModule::computePathsAndMask(
    std::vector<MedusaTreeNode> const& tree, TensorPtr& packedMask, TensorPtr& paths) const
{
    TLLM_LOG_TRACE("%s start", __PRETTY_FUNCTION__);
    auto pathsPtr = bufferCast<SizeType32>(*paths);

    std::fill(pathsPtr, pathsPtr + paths->getSize(), -1);
    std::fill(bufferCast<SizeType32>(*packedMask), bufferCast<SizeType32>(*packedMask) + packedMask->getSize(), 0);

    SizeType32 numPaths = 0;
    // Count leaves
    for (auto const& node : tree)
    {
        // Leaf
        if (node.childLinearIndices.size() == 0)
        {
            numPaths++;
        }
    }

    // Stack contains indices of the node in the 'tree' vector
    std::stack<SizeType32> stack;
    // Add first node
    stack.push(0);

    // Index of the current tracked path
    SizeType32 pathIdx = 0;

    // Do DFS to construct paths and mask
    while (!stack.empty())
    {
        auto const ci = stack.top();
        stack.pop();

        auto const& node = tree[ci];

        // If not root copy mask of your parent
        if (node.nodeId != -1)
        {
            copyPackedMask(packedMask, node.parentLinearIdx, ci);
        }
        // Set value on the diagonal
        setOnePackedMask(packedMask, ci, ci);

        // Leaf
        if (node.childLinearIndices.size() == 0)
        {
            // Current nodeIdx
            SizeType32 nodeIdx = ci;
            // Until we hit the root
            while (tree[nodeIdx].nodeId != -1)
            {
                auto const& curNode = tree[nodeIdx];
                pathsPtr[(numPaths - 1 - pathIdx) * getMaxPathLen() + curNode.depth] = curNode.linearIdx;
                // Go from top to the bottom
                nodeIdx = curNode.parentLinearIdx;
            }
            // Fill data for root
            // +0 is for root of the paths
            // getMaxPathLen() is because paths includes max accepted tokens and root
            // numPaths - 1 is because numPaths is the size of the paths tensor, but we need an index of the last path.
            pathsPtr[(numPaths - 1 - pathIdx) * getMaxPathLen() + 0] = 0;
            // Fill next path
            pathIdx++;
        }
        // Go over all children of the node and visit them
        for (auto const& childLinearIdx : node.childLinearIndices)
        {
            stack.push(childLinearIdx);
        }
    }
    TLLM_LOG_TRACE("%s stop", __PRETTY_FUNCTION__);
    return numPaths;
}

void MedusaModule::copyPackedMask(TensorPtr& mask, SizeType32 srcIdx, SizeType32 dstIdx) const
{
    auto srcRow = ITensor::slice(mask, srcIdx, 1);
    auto dstRow = ITensor::slice(mask, dstIdx, 1);
    std::memcpy(
        bufferCast<SizeType32>(*dstRow), bufferCast<SizeType32>(*srcRow), getNumPackedMasks() * sizeof(SizeType32));
}

void MedusaModule::setOnePackedMask(TensorPtr& mask, SizeType32 row, SizeType32 col) const
{
    auto const maskIdx = static_cast<SizeType32>(col / 32);
    auto const bitIdx = col % 32;
    auto setMask = 1 << bitIdx;
    bufferCast<SizeType32>(*mask)[row * getNumPackedMasks() + maskIdx] |= setMask;
}

MedusaModule::Prefix MedusaModule::computePrefix(std::vector<SizeType32> const& vec, SizeType32 len) const
{
    SizeType32 constexpr BITS_PER_BYTE = 8;
    // Check that prefix fits into the underlying data type
    TLLM_CHECK_WITH_INFO(static_cast<SizeType32>(sizeof(Prefix)) * BITS_PER_BYTE / PREFIX_CHUNK_SIZE_BITS >= len,
        "Provided choices have depth (%d) larger than Prefix can fit (%ld).", len,
        sizeof(Prefix) * BITS_PER_BYTE / PREFIX_CHUNK_SIZE_BITS);

    Prefix prefix = 0;
    for (SizeType32 ci = 0; ci < len; ++ci)
    {
        auto val = vec[ci];
        TLLM_CHECK_WITH_INFO(val <= PREFIX_MAX_VALUE,
            "Provided choices have too large node degree (%d). Larger than Prefix can fit (%d).", val,
            PREFIX_MAX_VALUE);
        // Prefix has property that on the same depth, nodes that are more left in the tree have smaller prefix
        prefix |= (vec[ci] << PREFIX_CHUNK_SIZE_BITS * (len - 1 - ci));
    }
    return prefix;
}

void MedusaModule::dumpChoices(MedusaChoices const& choices, std::vector<SizeType32> const& indices) const
{
    std::stringstream ss;
    ss << "Medusa choices = [";
    for (size_t ci = 0; ci < indices.size(); ++ci)
    {
        auto const idx = indices[ci];
        auto const& choice = choices[idx];
        ss << "[";
        for (size_t vi = 0; vi < choice.size(); ++vi)
        {
            ss << choice[vi];
            if (vi < choice.size() - 1)
            {
                ss << ", ";
            }
        }
        ss << "]";
        if (ci < indices.size() - 1)
        {
            ss << ", ";
        }
    }
    ss << "]" << std::endl;
    TLLM_LOG_DEBUG(ss.str().c_str());
}
} // namespace tensorrt_llm::runtime
