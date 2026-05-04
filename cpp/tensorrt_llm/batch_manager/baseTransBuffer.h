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

#pragma once

#include "tensorrt_llm/runtime/bufferManager.h"
#include "tensorrt_llm/runtime/iTensor.h"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager
{
class FabricMemory;
} // namespace tensorrt_llm::batch_manager::kv_cache_manager

namespace tensorrt_llm::batch_manager
{

enum class BufferKind : uint8_t
{
    kKV = 0,
    kKV_INDEXER = 1,
    kRNN = 2
};

class BaseTransBufferManager;

/// @brief RAII scoped holder for a buffer index acquired from
///        BaseTransBufferManager::assignBufferIndexForRecv /
///        assignBufferIndexForSend. Releases the index on destruction,
///        including stack unwind from exceptions.
///
/// Motivation: CacheReceiver::Impl::requestSync has at least six exit paths
/// (normal, early-cancel, not-ready, cancel-after-ready, receiveReadySignal
/// cancelled, exception from requestSync). Pre-fix, the buffer-index
/// release lived inside receiveSync's formatter — so any exit path that
/// skipped receiveSync leaked one index. v12 saturation test reproduced
/// this: one `(not-ready)` early return permanently wedged the size-1
/// pool and every subsequent request waited forever.
///
/// This holder closes that class of bug rather than patching the one
/// observed branch. Move-only so ownership is unambiguous; `detach()`
/// hands off ownership when the formatter inside receiveSync takes the
/// buffer's release responsibility on the happy path.
class BufferIndexHolder
{
public:
    BufferIndexHolder() = default;

    BufferIndexHolder(BaseTransBufferManager& mgr, std::optional<int> index, bool isRecv,
        std::optional<uint64_t> requestIdForLog = std::nullopt) noexcept
        : mMgr(&mgr)
        , mIndex(index)
        , mHeld(index.has_value())
        , mIsRecv(isRecv)
        , mRequestIdForLog(requestIdForLog)
    {
    }

    ~BufferIndexHolder()
    {
        releaseWithLog();
    }

    BufferIndexHolder(BufferIndexHolder const&) = delete;
    BufferIndexHolder& operator=(BufferIndexHolder const&) = delete;

    BufferIndexHolder(BufferIndexHolder&& other) noexcept
        : mMgr(other.mMgr)
        , mIndex(other.mIndex)
        , mHeld(other.mHeld)
        , mIsRecv(other.mIsRecv)
        , mRequestIdForLog(other.mRequestIdForLog)
    {
        other.mHeld = false;
    }

    BufferIndexHolder& operator=(BufferIndexHolder&& other) noexcept
    {
        if (this != &other)
        {
            release();
            mMgr = other.mMgr;
            mIndex = other.mIndex;
            mHeld = other.mHeld;
            mIsRecv = other.mIsRecv;
            mRequestIdForLog = other.mRequestIdForLog;
            other.mHeld = false;
        }
        return *this;
    }

    [[nodiscard]] std::optional<int> index() const noexcept
    {
        return mIndex;
    }

    [[nodiscard]] bool held() const noexcept
    {
        return mHeld;
    }

    /// @brief Relinquish ownership without releasing. Use when a downstream
    ///        owner (e.g. the formatter inside receiveSync) takes over the
    ///        release responsibility on the happy path.
    std::optional<int> detach() noexcept
    {
        mHeld = false;
        return mIndex;
    }

    /// @brief Happy-path release. Frees the slot immediately and disarms the
    ///        destructor so AUTO_RELEASE is NOT logged. Use this on any path
    ///        where the caller has confirmed the slot is no longer needed and
    ///        the release is the expected outcome (e.g. the sender formatter
    ///        after sendAllBuffers returns). After this call, the holder owns
    ///        nothing; subsequent destructor or move-assignment is a no-op.
    ///
    ///        Contrast with the destructor's fallback path: if the holder goes
    ///        out of scope with mHeld still true (exception, early return that
    ///        forgot to call release/detach), the destructor logs AUTO_RELEASE
    ///        so the non-happy exit is visible in [buf] diagnostics.
    void release() noexcept;

private:
    /// @brief Destructor-fallback path. Emits AUTO_RELEASE log (tagged with
    ///        reqId if known) and then performs the same work as release().
    void releaseWithLog() noexcept;

    BaseTransBufferManager* mMgr{nullptr};
    std::optional<int> mIndex{};
    bool mHeld{false};
    bool mIsRecv{true};
    std::optional<uint64_t> mRequestIdForLog{};
};

/// @brief Base class for cache transfer buffer management.
/// Handles buffer pool allocation, index assignment, and slicing.
/// Derived classes provide cache-specific size calculations.
class BaseTransBufferManager
{
public:
    virtual ~BaseTransBufferManager() = default;

    [[nodiscard]] virtual BufferKind getBufferKind() const = 0;

    /// @brief Assign a buffer index for sending.
    /// @param perRequestCancel Optional per-request cancel flag. When non-null
    ///        and flipped true while this call is parked on the pool-exhausted
    ///        CV wait, the function throws so the caller (sender worker) can
    ///        unwind instead of blocking indefinitely. Checked every
    ///        `waitSliceMs` during the wait. Parity with
    ///        assignBufferIndexForRecv.
    /// @param waitSliceMs Per-iteration timeout for the internal condition
    ///        variable wait (ms). Defaults to 100 ms.
    /// @param requestIdForLog Optional request id used to tag [buf] log lines
    ///        so a pool-exhausted wedge on the send side can be attributed
    ///        to a specific reqId.
    /// @return Assigned buffer index, or nullopt if using dynamic buffers.
    std::optional<int> assignBufferIndexForSend(std::atomic<bool> const* perRequestCancel = nullptr,
        int64_t waitSliceMs = 100, std::optional<uint64_t> requestIdForLog = std::nullopt);

    /// @brief Free a buffer index used for sending.
    /// @param bufferId The buffer index to free.
    void freeBufferIndexForSend(std::optional<int> bufferId);

    /// @brief Assign a buffer index for receiving.
    /// @param perRequestCancel Optional per-request cancel flag. When non-null
    ///        and flipped true while this call is parked on the pool-exhausted
    ///        CV wait, the function throws so the caller (drain worker) can
    ///        unwind instead of blocking indefinitely. Checked every
    ///        `waitSliceMs` during the wait; also bounds the wait by polling
    ///        for recovery even without an explicit cancel.
    /// @param waitSliceMs Per-iteration timeout for the internal condition
    ///        variable wait (ms). Defaults to 100 ms.
    /// @param requestIdForLog Optional request id used to tag [buf] log lines
    ///        so a pool-exhausted wedge can be attributed to a specific reqId
    ///        when cross-referenced with the drain-worker's [reqSync] trail.
    /// @return Assigned buffer index, or nullopt if using dynamic buffers.
    std::optional<int> assignBufferIndexForRecv(std::atomic<bool> const* perRequestCancel = nullptr,
        int64_t waitSliceMs = 100, std::optional<uint64_t> requestIdForLog = std::nullopt);

    /// @brief Free a buffer index used for receiving.
    /// @param bufferId The buffer index to free.
    void freeBufferIndexForRecv(std::optional<int> bufferId);

    /// @brief Get or allocate send buffers for cache transfer.
    /// @param bufferId The assigned buffer ID.
    /// @param targetNum Number of target sequences.
    /// @param requestedNumberOfElements Sizes requested for each target.
    /// @param bufferManagerToUse Buffer manager for dynamic allocation.
    /// @return Tuple of (buffers, covered target count, is dynamic only).
    std::tuple<std::vector<runtime::ITensor::SharedPtr>, size_t, bool> getOrAllocateSendBuffers(
        std::optional<int> bufferId, int targetNum, std::vector<size_t> const& requestedNumberOfElements,
        runtime::BufferManager const& bufferManagerToUse);

    /// @brief Get or allocate receive buffers for cache transfer.
    /// @param bufferId The assigned buffer ID.
    /// @param targetNum Number of target sequences.
    /// @param requestedNumberOfElements Sizes requested for each target.
    /// @param bufferManagerToUse Buffer manager for dynamic allocation.
    /// @return Tuple of (buffers, covered target count, is dynamic only).
    std::tuple<std::vector<runtime::ITensor::SharedPtr>, size_t, bool> getOrAllocateRecvBuffers(
        std::optional<int> bufferId, int targetNum, std::vector<size_t> const& requestedNumberOfElements,
        runtime::BufferManager const& bufferManagerToUse);

    /// @brief Get the send buffer for a given buffer ID.
    runtime::ITensor::SharedPtr getSendBuffer(std::optional<int> bufferId);

    /// @brief Get the receive buffer for a given buffer ID.
    runtime::ITensor::SharedPtr getRecvBuffer(std::optional<int> bufferId);

    /// @brief Get the number of receive buffers.
    size_t getRecvBufferCount();

    /// @brief Get the number of send buffers.
    size_t getSendBufferCount();

    /// @brief Get the maximum number of tokens configured.
    std::optional<size_t> getMaxNumTokens()
    {
        return mMaxNumTokens;
    }

protected:
    /// @brief Constructor - derived classes call this after computing buffer sizes.
    /// @param transferBufferSize Size of each transfer buffer in bytes.
    /// @param dataType Data type for the buffers.
    /// @param maxNumTokens Optional max tokens for sizing.
    BaseTransBufferManager(
        size_t transferBufferSize, nvinfer1::DataType dataType, std::optional<size_t> maxNumTokens = std::nullopt);

    struct ConcurrenceResource
    {
        std::unordered_map<int, runtime::ITensor::SharedPtr> mBuffers;
        std::vector<int> mBufferIndexFlag;
        std::mutex mBuffersMutex;
        std::condition_variable mBuffersCV;
        std::atomic<int> mConcurrence{0};
    };

    std::tuple<std::vector<runtime::ITensor::SharedPtr>, size_t, bool> getOrAllocateBuffers(std::optional<int> bufferId,
        int targetNum, std::vector<size_t> const& requestedNumberOfElements,
        runtime::BufferManager const& bufferManagerToUse, ConcurrenceResource& concurrenceResource);

    void allocateBuffer();
    std::optional<int> assignBufferIndex(ConcurrenceResource& resource, size_t bufferCount, bool onlyUseDynamicBuffer,
        std::atomic<bool> const* perRequestCancel = nullptr, int64_t waitSliceMs = 100,
        std::optional<uint64_t> requestIdForLog = std::nullopt);
    void freeBufferIndex(
        ConcurrenceResource& resource, std::optional<int> bufferId, size_t bufferCount, bool onlyUseDynamicBuffer);

    size_t mPreAllocBufferSize;
    size_t mRecvBufferCount;
    size_t mSendBufferCount;
    size_t mTransferBufferSize;
    bool mOnlyUseDynamicBuffer;
    bool mUseFabricMemory;
    size_t mNumberOfElements;
    nvinfer1::DataType mDataType;
    ConcurrenceResource mConcurrenceSendResource;
    ConcurrenceResource mConcurrenceRecvResource;
    runtime::BufferManager mBufferManager;
    std::vector<std::unique_ptr<kv_cache_manager::FabricMemory>> mFabricMemory;
    std::optional<size_t> mMaxNumTokens;
};

} // namespace tensorrt_llm::batch_manager
