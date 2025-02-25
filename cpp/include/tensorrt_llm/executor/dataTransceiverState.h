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

#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/executor/types.h"
#include "tensorrt_llm/runtime/modelConfig.h"
#include "tensorrt_llm/runtime/worldConfig.h"
#include <string>
#include <variant>
#include <vector>

namespace tensorrt_llm::executor
{

class Serialization;

namespace kv_cache
{

// Describe the data structure for cache layout, which can be used to infer
// cache layouts and location associations between different processes, in order
// to determine suitable senders and receivers.
class CacheState final
{
public:
    struct ModelConfig;

    explicit CacheState(ModelConfig modelConfig, runtime::WorldConfig const& worldConfig, nvinfer1::DataType dataType)
        : mModelConfig(std::move(modelConfig))
        , mParallelConfig{worldConfig.getTensorParallelism(), worldConfig.getPipelineParallelism()}
        , mDataType{dataType}
    {
    }

    CacheState(std::vector<SizeType32> const& nbKvHeadPerLayer, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        SizeType32 tensorParallelism, SizeType32 pipelineParallelism, nvinfer1::DataType dataType)
        : mModelConfig{std::move(nbKvHeadPerLayer), sizePerHead, tokensPerBlock}
        , mParallelConfig{tensorParallelism, pipelineParallelism}
        , mDataType{dataType}

    {
    }

    CacheState(SizeType32 nbAttentionLayers, SizeType32 nbKvHeads, SizeType32 sizePerHead, SizeType32 tokensPerBlock,
        SizeType32 tensorParallelism, SizeType32 pipelineParallelism, nvinfer1::DataType dataType)
        : mModelConfig{std::vector(nbAttentionLayers, nbKvHeads), sizePerHead, tokensPerBlock}
        , mParallelConfig{tensorParallelism, pipelineParallelism}
        , mDataType{dataType}
    {
    }

    [[nodiscard]] bool operator==(kv_cache::CacheState const& other) const noexcept
    {
        return mModelConfig == other.mModelConfig && mParallelConfig == other.mParallelConfig
            && mDataType == other.mDataType;
    }

    struct ModelConfig
    {
        std::vector<SizeType32> mNbKvHeadsPerLayer;
        SizeType32 mSizePerHead;
        SizeType32 mTokensPerBlock;

        [[nodiscard]] bool operator==(ModelConfig const& other) const noexcept
        {
            return mNbKvHeadsPerLayer == other.mNbKvHeadsPerLayer && mSizePerHead == other.mSizePerHead
                && mTokensPerBlock == other.mTokensPerBlock;
        }
    };

    struct ParallelConfig
    {
        SizeType32 mTensorParallelism;
        SizeType32 mPipelineParallelism;

        [[nodiscard]] bool operator==(ParallelConfig const& other) const noexcept
        {
            return mTensorParallelism == other.mTensorParallelism && mPipelineParallelism == other.mPipelineParallelism;
        }
    };

    [[nodiscard]] ModelConfig const& getModelConfig() const
    {
        return mModelConfig;
    }

    [[nodiscard]] ParallelConfig const& getParallelConfig() const
    {
        return mParallelConfig;
    }

    [[nodiscard]] nvinfer1::DataType const& getDataType() const
    {
        return mDataType;
    }

    [[nodiscard]] std::string toString() const
    {
        std::stringstream sstring;
        sstring << "numKvHeads\n";
        for (auto kvHead : mModelConfig.mNbKvHeadsPerLayer)
        {
            sstring << kvHead << "\n";
        }
        sstring << "sizePerHead:" << mModelConfig.mSizePerHead << "\n";
        sstring << "mTokensPerBlock:" << mModelConfig.mTokensPerBlock << "\n";
        sstring << "tp:" << mParallelConfig.mTensorParallelism << "\n";
        sstring << "pp:" << mParallelConfig.mPipelineParallelism << "\n";
        sstring << "datatype:" << static_cast<int32_t>(mDataType) << "\n";
        return sstring.str();
    }

private:
    friend class tensorrt_llm::executor::Serialization;
    ModelConfig mModelConfig;
    ParallelConfig mParallelConfig;
    nvinfer1::DataType mDataType;
};

struct MpiState
{
    [[nodiscard]] bool operator==(MpiState const& other) const noexcept
    {
        return mRanks == other.mRanks;
    }

    [[nodiscard]] std::string toString() const
    {
        std::stringstream sstring;
        sstring << "[";
        for (auto rank : mRanks)
        {
            sstring << rank << ",";
        }
        sstring << "]";
        return sstring.str();
    }

    std::vector<SizeType32> mRanks;
};

struct SocketState
{
    [[nodiscard]] bool operator==(SocketState const& other) const noexcept
    {
        return mPort == other.mPort && mIp == other.mIp;
    }

    [[nodiscard]] std::string toString() const
    {
        return mIp + ":" + std::to_string(mPort);
    }

    std::uint16_t mPort;
    std::string mIp;
};

class CommState final
{
public:
    CommState() = default;

    explicit CommState(std::vector<SizeType32> ranks, int selfIdx = -1)
        : mState{MpiState{std::move(ranks)}}
        , mSelfIdx{selfIdx}
    {
    }

    explicit CommState(std::vector<SocketState> socketState, int selfIdx = -1)
        : mState{std::move(socketState)}
        , mSelfIdx{selfIdx}
    {
    }

    CommState(std::uint16_t port, std::string ip)
        : mState{std::vector<SocketState>{SocketState{port, std::move(ip)}}}
        , mSelfIdx{0}
    {
    }

    [[nodiscard]] bool isMpiState() const noexcept
    {
        return std::holds_alternative<MpiState>(mState);
    }

    [[nodiscard]] bool isSocketState() const noexcept
    {
        return std::holds_alternative<std::vector<SocketState>>(mState);
    }

    [[nodiscard]] MpiState const& getMpiState() const
    {
        TLLM_CHECK(isMpiState());
        return std::get<MpiState>(mState);
    }

    [[nodiscard]] std::vector<SocketState> const& getSocketState() const
    {
        TLLM_CHECK(isSocketState());
        return std::get<std::vector<SocketState>>(mState);
    }

    [[nodiscard]] int getSelfIdx() const noexcept
    {
        return mSelfIdx;
    }

    [[nodiscard]] bool operator==(CommState const& other) const noexcept
    {
        return mState == other.mState;
    }

    [[nodiscard]] std::string toString() const
    {
        if (isMpiState())
        {
            return "MPI:" + getMpiState().toString();
        }
        if (isSocketState())
        {
            std::stringstream sstring;
            sstring << "SOCKET:[";
            for (auto&& socket : getSocketState())
            {
                sstring << socket.toString() << ",";
            }
            sstring << "]";
            return sstring.str();
        }
        return "";
    }

private:
    friend class tensorrt_llm::executor::Serialization;
    std::variant<std::monostate, MpiState, std::vector<SocketState>> mState;
    int mSelfIdx{-1};
};

} // namespace kv_cache

class DataTransceiverState final
{
public:
    DataTransceiverState() = default;

    DataTransceiverState(kv_cache::CacheState cacheState, kv_cache::CommState commState)
        : mCacheState{std::move(cacheState)}
        , mCommState{std::move(commState)}
    {
    }

    void setCacheState(kv_cache::CacheState state)
    {
        mCacheState = std::move(state);
    }

    [[nodiscard]] std::optional<kv_cache::CacheState> const& getCacheState() const noexcept
    {
        return mCacheState;
    }

    void setCommState(kv_cache::CommState state)
    {
        mCommState = std::move(state);
    }

    [[nodiscard]] std::optional<kv_cache::CommState> const& getCommState() const noexcept
    {
        return mCommState;
    }

    [[nodiscard]] bool operator==(DataTransceiverState const& other) const noexcept
    {
        return mCacheState == other.mCacheState && mCommState == other.mCommState;
    }

    [[nodiscard]] std::string toString() const
    {
        std::stringstream sstring;
        if (mCacheState)
        {
            sstring << mCacheState.value().toString();
        }
        if (mCommState)
        {
            sstring << mCommState.value().toString();
        }
        return sstring.str();
    }

private:
    friend class Serialization;
    std::optional<kv_cache::CacheState> mCacheState;
    std::optional<kv_cache::CommState> mCommState;
};

} // namespace tensorrt_llm::executor
