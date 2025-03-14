/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <deque>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <variant>

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"
#include "tensorrt_llm/plugins/api/tllmPlugin.h"
#include <cxxopts.hpp>

namespace tlc = tensorrt_llm::common;
namespace tle = tensorrt_llm::executor;
namespace fs = std::filesystem;

struct RuntimeOptions
{
    std::string trtEnginePath;
    tle::SizeType32 numSysPrompts;

    tle::SizeType32 sysPromptTokens;
    tle::SizeType32 contextTokens;

    tle::SizeType32 maxTokensMean;
    tle::SizeType32 maxTokensStddev;

    tle::SizeType32 numRequests;

    size_t hostCacheSize;
    size_t maxTokensInPagedKvCache;
};

struct KVCacheBlock
{
    KVCacheBlock(size_t hash, int cacheLevel, int priority, std::optional<size_t> loraId = std::nullopt,
        std::shared_ptr<KVCacheBlock> prevBlock = nullptr);

    size_t hash;
    int cacheLevel;
    int priority;

    std::optional<size_t> loraId;

    std::shared_ptr<KVCacheBlock> prevBlock;
    std::unordered_map<size_t, std::shared_ptr<KVCacheBlock>> nextBlocks;
};

class RadixTree
{
public:
    explicit RadixTree(tle::Executor& executor);
    // Check the executor for new events.
    void pollEvents();

private:
    std::shared_ptr<tle::KVCacheEventManager> mCacheEventManager;
    // The root block of the radix tree
    std::shared_ptr<KVCacheBlock> root;
    // A table mapping block hashes to their pointers
    std::unordered_map<size_t, std::shared_ptr<KVCacheBlock>> blockTable;
    // Event counter
    size_t eventCounter;
};

// Utility function to parse input arguments
RuntimeOptions parseArgs(int argc, char* argv[]);

// Create a tle::Request
tle::Request makeRequest(int sysPromptTokens, int contextTokens, std::uniform_int_distribution<int> sysPromptSelector,
    std::normal_distribution<double> maxNumTokensSelector);

std::default_random_engine gen;

int main(int argc, char* argv[])
{
    // Register the TRT-LLM plugins
    initTrtLlmPlugins();

    auto runtimeOpts = parseArgs(argc, argv);

    // Create the executor for this engine
    auto executorConfig = tle::ExecutorConfig(1); // Beam width 1 is required for cache block reuse
    auto kvCacheConfig = tle::KvCacheConfig(true,
        runtimeOpts.maxTokensInPagedKvCache ? std::optional(runtimeOpts.maxTokensInPagedKvCache)
                                            : std::nullopt); // Enable cache block reuse
    kvCacheConfig.setHostCacheSize(runtimeOpts.hostCacheSize);
    kvCacheConfig.setEventBufferMaxSize(32768);
    executorConfig.setKvCacheConfig(kvCacheConfig);

    auto executor = tle::Executor(runtimeOpts.trtEnginePath, tle::ModelType::kDECODER_ONLY, executorConfig);

    auto radixTree = RadixTree(executor);

    auto activeRequests = runtimeOpts.numRequests;

    std::uniform_int_distribution sysPromptSelector(
        1, runtimeOpts.numSysPrompts); // Select a system prompt between 1 and `runtimeOpts.numSysPrompts`
    std::normal_distribution<double> maxNumTokensSelector(runtimeOpts.maxTokensMean, runtimeOpts.maxTokensStddev);

    // Create and enqueue the requests
    for (int i = 0; i < runtimeOpts.numRequests; i++)
    {
        std::ignore = executor.enqueueRequest(makeRequest(
            runtimeOpts.sysPromptTokens, runtimeOpts.contextTokens, sysPromptSelector, maxNumTokensSelector));
    }

    while (activeRequests > 0)
    {
        auto responses = executor.awaitResponses(std::chrono::milliseconds(20));
        for (auto const& response : responses)
        {
            if (response.getResult().isFinal)
                activeRequests--;
        }
        // Only call pollEvents once every 20ms. Events are only added to the queue once per iteration, so no need to
        // poll faster than this.
        radixTree.pollEvents();
    }

    return 0;
}

RuntimeOptions parseArgs(int argc, char* argv[])
{
    RuntimeOptions runtimeOpts;

    cxxopts::Options options(argv[0], "Example that demonstrates how to use the ExecutorKVCacheManager API");
    options.add_options()("h,help", "Print usage");
    options.add_options()("engine_dir", "Directory that store the engines.", cxxopts::value<std::string>());
    options.add_options()("num_sys_prompts", "Amount of unique simulated system prompts to use",
        cxxopts::value<int>()->default_value("10"));
    options.add_options()(
        "sys_prompt_tokens", "Size of the simulated system prompts", cxxopts::value<int>()->default_value("256"));
    options.add_options()("context_tokens", "Amount of varying context tokens coming after the system prompts",
        cxxopts::value<int>()->default_value("128"));
    options.add_options()(
        "max_tokens_mean", "Mean number of max output tokens", cxxopts::value<int>()->default_value("128"));
    options.add_options()(
        "max_tokens_stddev", "Standard deviation of max output tokens", cxxopts::value<int>()->default_value("32"));
    options.add_options()(
        "num_requests", "Amount of requests to send to the engine", cxxopts::value<int>()->default_value("100"));
    options.add_options()("host_cache_size", "Size of the KV Cache in host memory in bytes",
        cxxopts::value<size_t>()->default_value("0"));
    options.add_options()("max_tokens_in_paged_kv_cache", "Amount of tokens in the kv cache",
        cxxopts::value<size_t>()->default_value("0"));

    auto parsedOptions = options.parse(argc, argv);

    // Argument: help
    if (parsedOptions.count("help"))
    {
        TLLM_LOG_ERROR(options.help());
        exit(0);
    }

    // Argument: Engine directory
    if (!parsedOptions.count("engine_dir"))
    {
        TLLM_LOG_ERROR(options.help());
        TLLM_LOG_ERROR("Please specify engine directory.");
        exit(1);
    }
    runtimeOpts.trtEnginePath = parsedOptions["engine_dir"].as<std::string>();
    if (!fs::exists(runtimeOpts.trtEnginePath) || !fs::is_directory(runtimeOpts.trtEnginePath))
    {
        TLLM_LOG_ERROR("Engine directory doesn't exist.");
        exit(1);
    }

    runtimeOpts.numSysPrompts = parsedOptions["num_sys_prompts"].as<int>();
    runtimeOpts.sysPromptTokens = parsedOptions["sys_prompt_tokens"].as<int>();
    runtimeOpts.contextTokens = parsedOptions["context_tokens"].as<int>();
    runtimeOpts.maxTokensMean = parsedOptions["max_tokens_mean"].as<int>();
    runtimeOpts.maxTokensStddev = parsedOptions["max_tokens_stddev"].as<int>();
    runtimeOpts.numRequests = parsedOptions["num_requests"].as<int>();
    runtimeOpts.hostCacheSize = parsedOptions["host_cache_size"].as<size_t>();
    runtimeOpts.maxTokensInPagedKvCache = parsedOptions["max_tokens_in_paged_kv_cache"].as<size_t>();

    return runtimeOpts;
}

KVCacheBlock::KVCacheBlock(
    size_t hash, int cacheLevel, int priority, std::optional<size_t> loraId, std::shared_ptr<KVCacheBlock> prevBlock)
    : hash{hash}
    , cacheLevel{cacheLevel}
    , priority{priority}
    , loraId{loraId}
    , prevBlock{prevBlock}
    , nextBlocks{}
{
}

RadixTree::RadixTree(tle::Executor& executor)
    : mCacheEventManager(*executor.getKVCacheEventManager())
    , eventCounter{1}
{
    // Use id=-1 for the root block. Doesn't matter what exact id is used, just that it is unique.
    root = std::make_shared<KVCacheBlock>(-1, -1, -1);
    blockTable[-1] = root;

    // Wait for the `CREATED` event to be emitted.
    while (true)
    {
        auto events = mCacheEventManager->getLatestEvents();
        if (events.size() == 1)
        {
            auto const& eventData = std::get<tle::KVCacheCreatedData>(events.front().data);
            TLLM_LOG_INFO("Event ID %d: KV Cache Manager initialized with blocks per level of: %s",
                events.front().eventId, tlc::vec2str(eventData.numBlocksPerCacheLevel).c_str());
            break;
        }
    }
};

void RadixTree::pollEvents()
{
    auto events = mCacheEventManager->getLatestEvents(std::chrono::milliseconds(20));
    for (tle::KVCacheEvent const& event : events)
    {
        TLLM_CHECK(event.eventId == eventCounter++);
        if (std::holds_alternative<tle::KVCacheStoredData>(event.data))
        {
            // Blocks have been stored into the radix tree
            auto const& eventData = std::get<tle::KVCacheStoredData>(event.data);
            auto prevBlock = blockTable[eventData.parentHash.value_or(-1)];

            // This block should be in the tree
            TLLM_CHECK(blockTable.find(prevBlock->hash) != blockTable.end());

            for (auto& block : eventData.blocks)
            {

                TLLM_LOG_INFO("Event ID %d: Block %04x was inserted into the radix tree with parent %04x.",
                    event.eventId, block.blockHash, prevBlock->hash);

                // This block shouldn't already exist in the tree, and should have tokens associated with it
                TLLM_CHECK(blockTable.find(block.blockHash) == blockTable.end());
                TLLM_CHECK(block.tokens.size() > 0);

                auto thisBlock = std::make_shared<KVCacheBlock>(
                    block.blockHash, block.cacheLevel, block.priority, block.loraId, prevBlock);

                blockTable[block.blockHash] = thisBlock;
                // Link the parent to the new block
                prevBlock->nextBlocks[block.blockHash] = thisBlock;

                prevBlock = thisBlock;
            }
        }
        else if (std::holds_alternative<tle::KVCacheRemovedData>(event.data))
        {
            auto const& eventData = std::get<tle::KVCacheRemovedData>(event.data);

            for (auto const& hash : eventData.blockHashes)
            {

                TLLM_LOG_INFO("Event ID %d: Block %04x was removed from the radix tree.", event.eventId, hash);

                // This block should exist in the tree
                TLLM_CHECK(blockTable.find(hash) != blockTable.end());

                auto& block = blockTable[hash];

                // Check that the block has no children, and that the parent has the block listed as a child
                TLLM_CHECK(block->nextBlocks.size() == 0);
                TLLM_CHECK(block->prevBlock->nextBlocks.find(block->hash) != block->prevBlock->nextBlocks.end());

                // Remove the block from it's parent, and remove the entry in the block table
                block->prevBlock->nextBlocks.erase(block->hash);
                blockTable.erase(hash);
            }
        }
        else if (std::holds_alternative<tle::KVCacheUpdatedData>(event.data))
        {
            auto const& eventData = std::get<tle::KVCacheUpdatedData>(event.data);

            if (eventData.priority.has_value())
            {
                // The block priority was updated
                TLLM_LOG_INFO("Event ID %d: Block %04x priority was changed from %d to %d", event.eventId,
                    eventData.blockHash, eventData.priority->oldValue, eventData.priority->newValue);

                TLLM_CHECK(blockTable[eventData.blockHash]->priority == eventData.priority->oldValue);
                blockTable[eventData.blockHash]->priority = eventData.priority->newValue;
            }

            if (eventData.cacheLevel.has_value())
            {
                // The block cache level was updated
                TLLM_LOG_INFO("Event ID %d: Block %04x cache level was changed from %d to %d", event.eventId,
                    eventData.blockHash, eventData.cacheLevel->oldValue, eventData.cacheLevel->newValue);

                TLLM_CHECK(blockTable[eventData.blockHash]->cacheLevel == eventData.cacheLevel->oldValue);
                blockTable[eventData.blockHash]->cacheLevel = eventData.cacheLevel->newValue;
            }
        }
        else
        {
            TLLM_LOG_ERROR("Unsupported event type. This shouldn't happen!");
        }
    }
}

tle::Request makeRequest(int sysPromptTokens, int contextTokens, std::uniform_int_distribution<int> sysPromptSelector,
    std::normal_distribution<double> maxNumTokensSelector)
{
    int sysPromptVersion = sysPromptSelector(gen);
    tle::VecTokens inputTokens;

    // Add `sysPromptTokens` tokens. Add the version to the token ids to create a unique system prompt
    for (int i = 0; i < sysPromptTokens; i++)
    {
        inputTokens.emplace_back(sysPromptVersion + i);
    }
    // Add random context tokens
    for (int i = 0; i < contextTokens; i++)
    {
        inputTokens.emplace_back(rand() % 1000);
    }

    return tle::Request(inputTokens, maxNumTokensSelector(gen));
}
