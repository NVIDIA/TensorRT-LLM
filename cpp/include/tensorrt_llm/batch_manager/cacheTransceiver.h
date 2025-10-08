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

#include "tensorrt_llm/batch_manager/cacheTransBuffer.h"
#include "tensorrt_llm/batch_manager/common.h"
#include "tensorrt_llm/batch_manager/kvCacheManager.h"
#include "tensorrt_llm/batch_manager/llmRequest.h"
#include "tensorrt_llm/executor/cacheCommunicator.h"
#include "tensorrt_llm/executor/dataTransceiverState.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"
#include "tensorrt_llm/runtime/utils/pgUtils.h"
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/custom_class.h>
#include <torch/python.h>
#include <type_traits>
#include <vector>

using SizeType32 = tensorrt_llm::runtime::SizeType32;

namespace tensorrt_llm::batch_manager
{

class ContextProgress;
class BaseCacheTransceiver;

namespace kv_cache_manager
{
class BaseKVCacheManager;
} // namespace kv_cache_manager

class CacheSender;
class CacheReceiver;

class CacheTransceiverComm
{
public:
    // Construct from a non-owning raw pointer, won't take ownership of the pointer
    explicit CacheTransceiverComm(mpi::MpiComm const* mpiComm)
        : mMpiComm(std::shared_ptr<mpi::MpiComm const>(nullptr), mpiComm)
    {
    }

    // Construct from a shared_ptr with shared ownership
    explicit CacheTransceiverComm(std::shared_ptr<mpi::MpiComm const> mpiComm)
        : mMpiComm(std::move(mpiComm))
    {
    }

    // Construct from a ProcessGroup communicator
    explicit CacheTransceiverComm(c10::intrusive_ptr<c10d::ProcessGroup> pgComm)
        : mPgComm(std::move(pgComm))
    {
    }

    ~CacheTransceiverComm() = default;

    bool isMpi() const noexcept
    {
        return mMpiComm != nullptr;
    }

    int getRank() const
    {
        if (isMpi())
        {
            return mMpiComm->getRank();
        }
        return mPgComm->getRank();
    }

    int getSize() const
    {
        if (isMpi())
        {
            return mMpiComm->getSize();
        }
        return mPgComm->getSize();
    }

    void allgather(void const* sendbuf, void* recvbuf, int count, mpi::MpiType dtype) const
    {
        if (isMpi())
        {
            mMpiComm->allgather(sendbuf, recvbuf, count, dtype);
            return;
        }
        TLLM_THROW("Input arguments only supported in mpi");
    }

    template <typename Input, typename Output>
    bool allgather(Input input, Output output, c10d::AllgatherOptions options = c10d::AllgatherOptions()) const
    {
        if (isMpi())
        {
            TLLM_THROW("Input arguments only supported in pg");
        }
        tensorrt_llm::pg_utils::PgHelper pgh{mPgComm};

        PGCHECK_THROW(pgh.allgather(input, output, options));
        return true;
    }

    template <typename Input, typename Output>
    bool allgatherv(Input input, Output output, std::vector<int> const& sizes,
        c10d::AllgatherOptions options = c10d::AllgatherOptions()) const
    {
        if (isMpi())
        {
            TLLM_THROW("Input arguments only supported in pg");
        }
        tensorrt_llm::pg_utils::PgHelper pgh{mPgComm};
        PGCHECK_THROW(pgh.allgatherv(input, output, sizes, options));
        return true;
    }

    bool allgatherv(void const* sendbuf, int sendcount, mpi::MpiType sendtype, void* recvbuf,
        std::vector<int> const& recvcounts, std::vector<int> const& displs, mpi::MpiType recvtype) const
    {
        if (isMpi())
        {
            mMpiComm->allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype);
            return true;
        }
        TLLM_THROW("Input arguments only supported in mpi");
    }

    CacheTransceiverComm split(int color, int key)
    {
        if (isMpi())
        {
            auto subgroup = mMpiComm->split(color, key);
            return CacheTransceiverComm(std::make_shared<mpi::MpiComm const>(std::move(subgroup)));
        }
        bool const initialized = Py_IsInitialized();
        TLLM_CHECK_WITH_INFO(initialized, "Trying to use ProcessGroup communicator but Python is not initialized");
        try
        {
            c10::intrusive_ptr<c10d::ProcessGroup> pgSub;
            {
                pybind11::gil_scoped_acquire gil;
                auto const m = pybind11::module::import("tensorrt_llm._torch.distributed.pg_utils");
                // Properly box the existing intrusive_ptr ProcessGroup into an IValue
                // and convert to a Python object without constructing a new instance.
                auto const py_pg = torch::jit::toPyObject(c10::IValue(mPgComm));

                auto const py_sub_pg = m.attr("split")(color, key, py_pg);
                pgSub = torch::jit::toCustomClass<c10d::ProcessGroup>(py_sub_pg);
            }
            return CacheTransceiverComm(pgSub);
        }
        catch (...)
        {
            TLLM_THROW("Failed to split process group");
        }
    }

private:
    std::shared_ptr<mpi::MpiComm const> mMpiComm;
    c10::intrusive_ptr<c10d::ProcessGroup> mPgComm;
};

class CacheTransceiverFactory
{
public:
    static std::unique_ptr<BaseCacheTransceiver> createCacheTransceiver(
        kv_cache_manager::BaseKVCacheManager* cacheManager, runtime::ModelConfig const& modelConfig,
        runtime::WorldConfig const& worldConfig,
        executor::kv_cache::CacheState::AttentionType attentionType
        = executor::kv_cache::CacheState::AttentionType::kDEFAULT,
        std::optional<executor::CacheTransceiverConfig> cacheTransceiverConfig = std::nullopt);
};

class BaseCacheTransceiver
{
public:
    virtual ~BaseCacheTransceiver() = default;
    virtual void respondAndSendAsync(LlmRequest* llmRequest) = 0;
    virtual void respondAndSendLayerWise(
        RequestVector const& requests, std::shared_ptr<ContextProgress> const& progress)
        = 0;

    virtual void requestAndReceiveSync(LlmRequest* llmRequest) = 0;
    virtual void requestAndReceiveAsync(LlmRequest* llmRequest) = 0;

    virtual void checkContextTransferStatus(std::optional<int> const& atLeastRequestNum = std::nullopt) = 0;

    virtual void checkGenTransferStatus(std::optional<int> const& atLeastRequestNum = std::nullopt) = 0;

    [[nodiscard]] virtual bool checkGenTransferComplete() const = 0;

    virtual bool cancelRequest(LlmRequest* llmRequest) = 0;
};

class CacheTransceiver : public BaseCacheTransceiver
{
public:
    CacheTransceiver(kv_cache_manager::BaseKVCacheManager* cacheManager,
        executor::kv_cache::CacheState::ModelConfig const& cacheStateModelCfg, runtime::WorldConfig const& worldConfig,
        std::vector<SizeType32> const& attentionLayerNumPerPP, nvinfer1::DataType dataType,
        executor::kv_cache::CacheState::AttentionType attentionType
        = executor::kv_cache::CacheState::AttentionType::kDEFAULT,
        std::optional<executor::CacheTransceiverConfig> cacheTransceiverConfig = std::nullopt);

    CacheTransceiver(kv_cache_manager::BaseKVCacheManager* cacheManager, std::vector<SizeType32> numKvHeadsPerLayer,
        SizeType32 sizePerHead, SizeType32 tokensPerBlock, runtime::WorldConfig const& worldConfig,
        std::vector<SizeType32> const& attentionLayerNumPerPP, nvinfer1::DataType dataType,
        executor::kv_cache::CacheState::AttentionType attentionType
        = executor::kv_cache::CacheState::AttentionType::kDEFAULT,
        std::optional<executor::CacheTransceiverConfig> cacheTransceiverConfig = std::nullopt)
        : CacheTransceiver(cacheManager,
            executor::kv_cache::CacheState::ModelConfig{numKvHeadsPerLayer, sizePerHead, tokensPerBlock}, worldConfig,
            attentionLayerNumPerPP, dataType, attentionType, cacheTransceiverConfig)
    {
    }

    virtual ~CacheTransceiver();

    void respondAndSendAsync(LlmRequest* llmRequest) override;

    void respondAndSendLayerWise(
        RequestVector const& requests, std::shared_ptr<ContextProgress> const& progress) override;

    void requestAndReceiveSync(LlmRequest* llmRequest) override;
    void requestAndReceiveAsync(LlmRequest* llmRequest) override;

    void checkContextTransferStatus(std::optional<int> const& atLeastRequestNum = std::nullopt) override;

    void checkGenTransferStatus(std::optional<int> const& atLeastRequestNum = std::nullopt) override;

    [[nodiscard]] bool checkGenTransferComplete() const override;

    virtual bool cancelRequest(LlmRequest* llmRequest) override;

private:
    void initializeCommState();

    void setContextState(LlmRequest* llmRequest);

    std::unique_ptr<CacheSender> mCacheSender;
    std::unique_ptr<CacheReceiver> mCacheReceiver;
    std::vector<std::pair<LlmRequest*, std::future<void>>> mSenderFutures;
    std::vector<std::pair<LlmRequest*, std::future<void>>> mRequesterFutures;
    mpi::MpiComm const* mMpiWorldComm{nullptr};

    std::shared_ptr<CacheTransceiverComm> mGroupComm;
    std::shared_ptr<CacheTransceiverComm> mGroupTensorParaComm, mGroupPipeParaComm, mGroupDataComm, mGroupTPInDPComm;

    executor::kv_cache::CommState const* mCommState;
    std::unique_ptr<executor::kv_cache::CacheState> mCacheState;
    std::unique_ptr<executor::kv_cache::ConnectionManager> mManager;
    std::optional<executor::CacheTransceiverConfig> mCacheTransceiverConfig;
    std::unique_ptr<kv_cache_manager::CacheTransBufferManager> mCacheTransBufferManager;
    // library handle to the communicator related features,
    // this is used to defer dependency resolution until needed.
    static std::mutex mDllMutex;
    void* mWrapperLibHandle{nullptr};
};

} // namespace tensorrt_llm::batch_manager
