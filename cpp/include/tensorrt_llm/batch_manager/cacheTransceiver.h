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
#include <torch/custom_class.h>
#include <torch/python.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <pybind11/pybind11.h>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <type_traits>
#include <vector>

using SizeType32 = tensorrt_llm::runtime::SizeType32;

namespace tensorrt_llm::batch_manager
{

class ContextProgress;
class BaseCacheTransceiver;
class DataResponder;
class DataRequester;

class CacheTransceiverComm
{
public:
    // Construct from a non-owning raw pointer (the pointed object must outlive this wrapper)
    explicit CacheTransceiverComm(mpi::MpiComm const* mpiComm)
    {
        if (mpiComm)
        {
            mMpiComm = std::shared_ptr<mpi::MpiComm const>(mpiComm, [](mpi::MpiComm const*) {});
        }
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

    bool isMpi() const noexcept { return mMpiComm.has_value() && static_cast<bool>(mMpiComm.value()); }

    int getRank() const
    {
        if (isMpi())
        {
            return mMpiComm.value()->getRank();
        }
        return mPgComm->getRank();
    }

    int getSize() const
    {
        if (isMpi())
        {
            return mMpiComm.value()->getSize();
        }
        return mPgComm->getSize();
    }

    void allgather(void const* sendbuf, void* recvbuf, int count, mpi::MpiType dtype) const {
        if (isMpi()) {
            mMpiComm.value()->allgather(sendbuf, recvbuf, count, dtype);
            return;
        }
        TLLM_THROW("Input arguments only supported in mpi");
    }

    template <typename Input, typename Output>
    bool allgather(Input input, Output output, c10d::AllgatherOptions options = c10d::AllgatherOptions()) const{
        if (isMpi()) {
            TLLM_THROW("Input arguments only supported in pg");
        }
        tensorrt_llm::pg_utils::PgHelper pgh{mPgComm};

        PGCHECK_THROW(pgh.allgather(input, output, options));
        return true;
    }

    template <typename Input, typename Output>
    bool allgatherv(Input input, Output output, std::vector<int> const& sizes, c10d::AllgatherOptions options = c10d::AllgatherOptions()) const{
        if (isMpi()) {
            TLLM_THROW("Input arguments only supported in pg");
        }
        tensorrt_llm::pg_utils::PgHelper pgh{mPgComm};
        PGCHECK_THROW(pgh.allgatherv(input, output, sizes, options));
        return true;
    }

    bool allgatherv(void const* sendbuf, int sendcount, mpi::MpiType sendtype, void* recvbuf,
        std::vector<int> const& recvcounts, std::vector<int> const& displs, mpi::MpiType recvtype) const{
        if (isMpi()) {
            mMpiComm.value()->allgatherv(sendbuf, sendcount, sendtype, recvbuf, recvcounts, displs, recvtype);
            return true;
        }
        TLLM_THROW("Input arguments only supported in mpi");
    }

    CacheTransceiverComm split(int color, int key) {
        if (isMpi())
        {
            auto subgroup = mMpiComm.value()->split(color, key);
            return CacheTransceiverComm(std::make_shared<mpi::MpiComm const>(std::move(subgroup)));
        }
        try {
            pybind11::gil_scoped_acquire gil;
            if (!Py_IsInitialized()) {
                Py_Initialize();
            }
            pybind11::module m = pybind11::module::import("tensorrt_llm._torch.distributed.pg_utils");
            // Properly box the existing intrusive_ptr ProcessGroup into an IValue
            // and convert to a Python object without constructing a new instance.
            c10::IValue iv_pg(mPgComm);
            pybind11::object py_pg = torch::jit::toPyObject(iv_pg);

            pybind11::object py_sub_pg = m.attr("split")(color, key, py_pg);
            auto pgSub = torch::jit::toCustomClass<c10d::ProcessGroup>(py_sub_pg);
            return CacheTransceiverComm(pgSub);
        } catch (...) {
            TLLM_THROW("Failed to split process group");
        }
    }

private:
    // Store MPI communicator with shared ownership (or non-owning alias via custom deleter)
    std::optional<std::shared_ptr<mpi::MpiComm const>> mMpiComm;
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
};

class CacheTransceiver : public BaseCacheTransceiver
{
public:
    CacheTransceiver(kv_cache_manager::BaseKVCacheManager* cacheManager,
        executor::kv_cache::CacheState::ModelConfig const& cacheStateModelCfg, runtime::WorldConfig const& worldConfig,
        nvinfer1::DataType dataType,
        executor::kv_cache::CacheState::AttentionType attentionType
        = executor::kv_cache::CacheState::AttentionType::kDEFAULT,
        std::optional<executor::CacheTransceiverConfig> cacheTransceiverConfig = std::nullopt);

    CacheTransceiver(kv_cache_manager::BaseKVCacheManager* cacheManager, std::vector<SizeType32> numKvHeadsPerLayer,
        SizeType32 sizePerHead, SizeType32 tokensPerBlock, runtime::WorldConfig const& worldConfig,
        nvinfer1::DataType dataType,
        executor::kv_cache::CacheState::AttentionType attentionType
        = executor::kv_cache::CacheState::AttentionType::kDEFAULT,
        std::optional<executor::CacheTransceiverConfig> cacheTransceiverConfig = std::nullopt)
        : CacheTransceiver(cacheManager,
            executor::kv_cache::CacheState::ModelConfig{numKvHeadsPerLayer, sizePerHead, tokensPerBlock}, worldConfig,
            dataType, attentionType, cacheTransceiverConfig)
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

private:
    void initializeCommState();

    void setContextState(LlmRequest* llmRequest);

    std::unique_ptr<DataResponder> mDataResponder;
    std::unique_ptr<DataRequester> mDataRequester;
    std::vector<std::pair<LlmRequest*, std::future<void>>> mResponderFutures;
    std::vector<std::pair<LlmRequest*, std::future<void>>> mRequesterFutures;
    // only for mpi backend, don't need it for ucx backend
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
