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
#include "tensorrt_llm/batch_manager/rnnCacheTransBuffer.h"
#include "tensorrt_llm/batch_manager/rnnStateManager.h"
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

struct RequestStatuses
{
    /// Requests that have completed their transfer successfully.
    std::unordered_set<LlmRequest::RequestIdType> completedRequestIds;
    /// Requests that have encountered an error during their transfer.
    std::unordered_set<LlmRequest::RequestIdType> errorRequestIds;
};

/// Structured outcome of a cancellation request. Replaces the historical
/// boolean return that conflated "Python may free request resources" with
/// "C++ has nothing to do." Only @c kCancelledBeforeAdvertise and
/// @c kAlreadyComplete imply that downstream cleanup may proceed
/// immediately. The other states require Python to leave the request owned
/// by C++ until a final transfer state is observed (or the transceiver
/// reports unhealthy).
enum class TransferCancelResult : uint8_t
{
    /// Request is unknown to the transceiver (already cleaned up or never
    /// registered).
    kNotFound = 0,
    /// Request has already reached worker-final state; the future is ready
    /// and any allocated buffers have been released.
    kAlreadyComplete = 1,
    /// Request was queued but no transfer buffer/handle was advertised to
    /// a peer yet. Memory is safe to release immediately.
    kCancelledBeforeAdvertise = 2,
    /// Cancellation was accepted but the worker is still in flight. The
    /// transceiver retains ownership of the request and its future until a
    /// final state is observed; Python must not free request resources.
    kCancelRequestedInFlight = 3,
    /// The transceiver is unhealthy (quarantine budget exceeded or no
    /// backend progress past the global deadline). Orchestration should
    /// restart the worker.
    kBackendUnhealthy = 4,
    /// Cancellation cannot be performed at this point (e.g., the sender is
    /// in the middle of advertising the buffer to a peer). Retry later.
    kNotCancellable = 5,
};

/// Health snapshot exposed for observability. All counters are best-effort
/// approximations updated on the executor thread; they are not real-time
/// kernel-level metrics.
struct TransceiverHealth
{
    bool isHealthy{true};
    /// Number of in-flight transfers that have exceeded their per-request
    /// timeout but whose worker future has not yet reached a final state.
    /// These transfers are tracked but their request resources are kept
    /// pinned by C++ — Python must not free them.
    size_t quarantinedTransferCount{0};
    /// Maximum number of quarantined transfers permitted before the
    /// transceiver is flipped to unhealthy.
    size_t quarantineBudget{0};
    /// Wall-clock seconds since the last observed worker future
    /// transition. Only meaningful when the transceiver has tracked at
    /// least one transfer.
    double secondsSinceLastProgress{0.0};
    /// Wall-clock seconds beyond which "no progress" is treated as a
    /// global backend wedge.
    double globalProgressDeadlineSeconds{0.0};
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

    /// Non-blocking poll. Returns the requests that have completed or
    /// encountered an error since the last call. With
    /// @c atLeastRequestNum unset the call defaults to a pure poll: it
    /// must @b never block on an unready future on the executor thread.
    /// Callers that genuinely need to drain (shutdown only) must use
    /// @ref drainContextTransferStatus instead.
    virtual RequestStatuses checkContextTransferStatus(
        std::optional<int> const& atLeastRequestNum = std::nullopt, bool markComplete = false)
        = 0;

    /// Non-blocking poll. Same contract as @ref checkContextTransferStatus.
    virtual void checkGenTransferStatus(std::optional<int> const& atLeastRequestNum = std::nullopt) = 0;

    /// Blocking drain — @b only safe on dedicated drain/shutdown paths.
    /// The default forwards to the polling variant for backward
    /// compatibility; subclasses that own worker futures override this to
    /// actually wait for completion.
    virtual RequestStatuses drainContextTransferStatus(bool markComplete = false)
    {
        return checkContextTransferStatus(std::nullopt, markComplete);
    }

    /// Blocking drain — @b only safe on dedicated drain/shutdown paths.
    virtual void drainGenTransferStatus()
    {
        checkGenTransferStatus(std::nullopt);
    }

    [[nodiscard]] virtual bool checkGenTransferComplete() const = 0;

    /// Structured cancellation. See @ref TransferCancelResult for the full
    /// contract. Subclasses override this; the boolean wrapper below maps
    /// the structured result onto the historical "is the request safe to
    /// release now" bool.
    virtual TransferCancelResult cancelRequestStructured(LlmRequest* llmRequest)
    {
        return cancelRequest(llmRequest) ? TransferCancelResult::kCancelledBeforeAdvertise
                                         : TransferCancelResult::kNotFound;
    }

    /// Backward-compatible wrapper: returns true only when Python is safe
    /// to free the request's resources immediately (pre-advertise cancel
    /// or already-complete worker). Callers that need to distinguish
    /// in-flight cancellation from "nothing to do" should use
    /// @ref cancelRequestStructured.
    virtual bool cancelRequest(LlmRequest* llmRequest) = 0;

    /// Whether the transceiver is currently healthy. Flips to false when
    /// quarantined transfers exceed the budget or when no backend
    /// progress has been observed past the global deadline.
    [[nodiscard]] virtual bool isHealthy() const
    {
        return true;
    }

    /// Health snapshot for orchestration / metrics.
    [[nodiscard]] virtual TransceiverHealth getHealth() const
    {
        return {};
    }
};

class CacheTransceiver : public BaseCacheTransceiver
{
public:
    CacheTransceiver(kv_cache_manager::BaseKVCacheManager* cacheManager,
        executor::kv_cache::CacheState::ModelConfig const& cacheStateModelCfg, runtime::WorldConfig const& worldConfig,
        std::vector<SizeType32> const& attentionLayerNumPerPP, nvinfer1::DataType dataType,
        executor::kv_cache::CacheState::AttentionType attentionType
        = executor::kv_cache::CacheState::AttentionType::kDEFAULT,
        std::optional<executor::CacheTransceiverConfig> cacheTransceiverConfig = std::nullopt,
        rnn_state_manager::RnnStateManager* rnnStateManager = nullptr,
        std::vector<SizeType32> const& rnnLayerNumPerPP = {});

    CacheTransceiver(kv_cache_manager::BaseKVCacheManager* cacheManager, std::vector<SizeType32> numKvHeadsPerLayer,
        SizeType32 sizePerHead, SizeType32 tokensPerBlock, runtime::WorldConfig const& worldConfig,
        std::vector<SizeType32> const& attentionLayerNumPerPP, nvinfer1::DataType dataType,
        executor::kv_cache::CacheState::AttentionType attentionType
        = executor::kv_cache::CacheState::AttentionType::kDEFAULT,
        std::optional<executor::CacheTransceiverConfig> cacheTransceiverConfig = std::nullopt,
        rnn_state_manager::RnnStateManager* rnnStateManager = nullptr,
        std::vector<SizeType32> const& rnnLayerNumPerPP = {})
        : CacheTransceiver(cacheManager,
            executor::kv_cache::CacheState::ModelConfig{numKvHeadsPerLayer, sizePerHead, tokensPerBlock}, worldConfig,
            attentionLayerNumPerPP, dataType, attentionType, cacheTransceiverConfig, rnnStateManager, rnnLayerNumPerPP)
    {
    }

    virtual ~CacheTransceiver();

    void respondAndSendAsync(LlmRequest* llmRequest) override;

    void respondAndSendLayerWise(
        RequestVector const& requests, std::shared_ptr<ContextProgress> const& progress) override;

    void requestAndReceiveSync(LlmRequest* llmRequest) override;
    void requestAndReceiveAsync(LlmRequest* llmRequest) override;

    RequestStatuses checkContextTransferStatus(
        std::optional<int> const& atLeastRequestNum = std::nullopt, bool markComplete = false) override;

    void checkGenTransferStatus(std::optional<int> const& atLeastRequestNum = std::nullopt) override;

    /// Blocking drain — only invoked from shutdown/teardown.
    RequestStatuses drainContextTransferStatus(bool markComplete = false) override;

    /// Blocking drain — only invoked from shutdown/teardown.
    void drainGenTransferStatus() override;

    [[nodiscard]] bool checkGenTransferComplete() const override;

    TransferCancelResult cancelRequestStructured(LlmRequest* llmRequest) override;

    virtual bool cancelRequest(LlmRequest* llmRequest) override;

    [[nodiscard]] bool isHealthy() const override;

    [[nodiscard]] TransceiverHealth getHealth() const override;

private:
    void initializeCommState();

    void setContextState(LlmRequest* llmRequest);

    /// Per-entry tracking for tracked worker futures. We keep the future
    /// pinned until the worker reaches a final state, even after the
    /// per-request deadline elapses, so that cancelling a still-running
    /// worker never frees its KV/buffer resources prematurely.
    struct TrackedFuture
    {
        LlmRequest* request;
        std::future<void> future;
        std::chrono::steady_clock::time_point deadline;
        /// True once the per-request timeout has fired and we have flipped
        /// the request to an error state. The entry stays in the vector
        /// (and the future stays pinned) until the worker actually
        /// finishes — possibly forever if the backend is wedged, in which
        /// case the global progress deadline raises an unhealthy signal.
        bool quarantined{false};
        /// True once we have already advertised a buffer/handle to a
        /// peer. Pre-advertise cancellation can release normally; post-
        /// advertise cancellation must wait for worker quiescence.
        bool advertised{false};
    };

    /// Update the global "last progress" timestamp and re-evaluate the
    /// transceiver health flag against the quarantine budget and the
    /// global progress deadline. Caller must hold @ref mHealthMutex.
    void updateHealthLocked();

    /// Drop a TrackedFuture entry from a vector. Decrements the
    /// quarantine counter if the entry was quarantined, marks progress.
    void releaseTrackedFutureLocked(std::vector<TrackedFuture>& vec, size_t index);

    /// Apply per-entry timeout policy: if the configured
    /// kvTransferTimeoutMs has elapsed and the future is not yet ready,
    /// flip the entry to quarantined and surface an error to the caller.
    /// Returns true if the entry was newly quarantined.
    bool maybeQuarantineLocked(TrackedFuture& entry, RequestStatuses* outStatus);

    /// Internal worker for the public polling and drain entry points.
    /// @c allowBlocking is true only on shutdown drain — that is the one
    /// path where waiting on a worker future is acceptable.
    RequestStatuses checkContextTransferStatusImpl(
        std::optional<int> const& atLeastRequestNum, bool markComplete, bool allowBlocking);
    void checkGenTransferStatusImpl(std::optional<int> const& atLeastRequestNum, bool allowBlocking);

    /// Compute the per-entry deadline for a tracked transfer.
    [[nodiscard]] std::chrono::steady_clock::time_point computeTrackedFutureDeadline(
        std::chrono::steady_clock::time_point requestStart) const;

    std::unique_ptr<CacheSender> mCacheSender;
    std::unique_ptr<CacheReceiver> mCacheReceiver;
    std::vector<TrackedFuture> mSenderFutures;
    std::vector<TrackedFuture> mRequesterFutures;
    mpi::MpiComm const* mMpiWorldComm{nullptr};

    /// Health/quarantine accounting. The mutex protects only this small
    /// block; the executor thread is the sole writer/reader of the future
    /// vectors above.
    mutable std::mutex mHealthMutex;
    size_t mQuarantinedTransferCount{0};
    size_t mQuarantineBudget{16};
    std::chrono::steady_clock::time_point mLastProgressTime{std::chrono::steady_clock::now()};
    std::chrono::milliseconds mGlobalProgressDeadlineMs{60'000};
    bool mIsHealthy{true};

    std::shared_ptr<CacheTransceiverComm> mGroupComm;
    std::shared_ptr<CacheTransceiverComm> mGroupTensorParaComm, mGroupPipeParaComm, mGroupDataComm, mGroupTPInDPComm;

    executor::kv_cache::CommState const* mCommState;
    std::unique_ptr<executor::kv_cache::CacheState> mCacheState;
    std::unique_ptr<executor::kv_cache::ConnectionManager> mManager;
    std::optional<executor::CacheTransceiverConfig> mCacheTransceiverConfig;
    std::vector<std::unique_ptr<kv_cache_manager::CacheTransBufferManager>> mCacheTransBufferManagers;
    std::vector<BaseTransBufferManager*> mCacheTransBufferManagerPtrs;

    rnn_state_manager::RnnStateManager* mRnnStateManager{nullptr};
    // TODO(shreyasm): update this to use same container as kv by using base trans buffers instead
    std::unique_ptr<rnn_state_manager::RnnCacheTransBufferManager> mRnnCacheTransBufferManager{nullptr};

    // library handle to the communicator related features,
    // this is used to defer dependency resolution until needed.
    static std::mutex mDllMutex;
    void* mWrapperLibHandle{nullptr};
};

} // namespace tensorrt_llm::batch_manager
