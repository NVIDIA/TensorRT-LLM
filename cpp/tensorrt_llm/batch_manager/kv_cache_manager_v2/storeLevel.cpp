/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "kv_cache_manager_v2/storeLevel.h"
#include "kv_cache_manager_v2/exceptions.h"

#include "tensorrt_llm/common/logger.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace
{

//! One op on its way to the IO table: owned by the host function that submits it.
struct SubmitPayload
{
    TllmKvStoreLevelIoV1 const* table = nullptr;
    std::shared_ptr<detail::StoreLevelShared> shared;
    TllmKvStoreOp op{};
    std::vector<TllmKvStoreRow> rows;
};

uint32_t loadDoneWord(uint32_t* word) noexcept
{
    return std::atomic_ref<uint32_t>(*word).load(std::memory_order_acquire);
}

//! Hands one op to the IO table. Runs on a CUDA host-function thread: no CUDA call, no wait on the store (unless the
//! blocking fallback is on, where it waits for its own op instead of the stream).
void CUDA_CB storeSubmitTrampoline(void* userData)
{
    std::unique_ptr<SubmitPayload> payload(static_cast<SubmitPayload*>(userData));
    payload->op.rows = payload->rows.data();
    payload->table->submit(payload->table->ctx, &payload->op);
    payload->shared->opsSubmitted.fetch_add(1, std::memory_order_release);
    if (payload->shared->blockingHostFunction)
    {
        while (loadDoneWord(payload->op.doneWord) < payload->op.doneValue)
        {
            std::this_thread::yield();
        }
    }
}

[[noreturn]] void refuseTable(std::string const& reason)
{
    throw std::invalid_argument("attach_store_level_io: " + reason);
}

[[noreturn]] void failStep(char const* step, long long code)
{
    throw std::runtime_error(std::string("store level: ") + step + " failed with " + std::to_string(code));
}

bool blockingHostFunctionRequested()
{
    char const* value = std::getenv("TLLM_KV_STORE_TIER_BLOCKING_HOSTFN");
    return value != nullptr && std::string(value) == "1";
}

//! The degrade fault mode (store_tier.py validates the knob and runs the IO table in the same mode).
bool degradeFaultModeRequested()
{
    char const* value = std::getenv("TLLM_KV_STORE_TIER_FAULT_MODE");
    if (value == nullptr)
    {
        return false;
    }
    std::string mode(value);
    auto const first = mode.find_first_not_of(" \t\n");
    auto const last = mode.find_last_not_of(" \t\n");
    return first != std::string::npos && mode.substr(first, last - first + 1) == "degrade";
}

double breakerSecondsRequested()
{
    char const* value = std::getenv("TLLM_KV_STORE_TIER_BREAKER_S");
    if (value == nullptr || *value == '\0')
    {
        return StoreLevel::kDefaultBreakerS;
    }
    char* end = nullptr;
    double const seconds = std::strtod(value, &end);
    return end != value && seconds >= 0.0 ? seconds : StoreLevel::kDefaultBreakerS;
}

// cuStreamWaitValue32(GEQ) passes when (int32_t)(*word - value) >= 0: a failed op's word (value | flag) must still
// pass its own wait, so the flag has to be a positive int32_t.
static_assert(static_cast<int32_t>(TLLM_KV_STORE_DONE_FAILED) > 0,
    "TLLM_KV_STORE_DONE_FAILED must keep the cyclic GEQ stream wait of a failed op satisfied");

uint32_t plainDoneValue(uint32_t word) noexcept
{
    return word & ~uint32_t{TLLM_KV_STORE_DONE_FAILED};
}

} // namespace

StoreLevel::~StoreLevel()
{
    if (mTable != nullptr)
    {
        // Not detached (the manager was destroyed without shutdown()): release what can be released safely.
        uint64_t const enqueued = mOpsEnqueued.load(std::memory_order_relaxed);
        bool const drained
            = mShared->opsSubmitted.load(std::memory_order_acquire) == enqueued && mTable->drain(mTable->ctx, 5.0) == 0;
        if (!drained)
        {
            // A stream may still wait on the ring, and the table may still read the ranges: leak both, and keep the
            // parked streams out of the pool for good.
            TLLM_LOG_ERROR("store level destroyed with IO ops in flight; leaking its completion ring");
            for (auto& parked : mParked)
            {
                static_cast<void>(parked.release());
            }
            return;
        }
        for (auto const& range : mRanges)
        {
            mTable->unregisterMemory(mTable->ctx, reinterpret_cast<void*>(range.first));
        }
        mTable = nullptr;
    }
    if (mDoneHost != nullptr)
    {
        cuMemFreeHost(mDoneHost);
        mDoneHost = nullptr;
    }
}

void StoreLevel::ensureRing()
{
    if (mDoneHost != nullptr)
    {
        return;
    }
    size_t const bytes = (size_t{kDoneWords} + 1) * sizeof(uint32_t);
    void* host = nullptr;
    cuCheck(cuMemHostAlloc(&host, bytes, CU_MEMHOSTALLOC_DEVICEMAP | CU_MEMHOSTALLOC_PORTABLE));
    std::memset(host, 0, bytes);
    CUdeviceptr device = 0;
    CUresult const result = cuMemHostGetDevicePointer(&device, host, 0);
    if (result != CUDA_SUCCESS)
    {
        cuMemFreeHost(host);
        cuCheck(result);
    }
    mDoneHost = static_cast<uint32_t*>(host);
    mDoneDevice = device;
}

void StoreLevel::selfTestRing()
{
    uint32_t* const word = mDoneHost + kDoneWords;
    CUdeviceptr const wordDevice = mDoneDevice + size_t{kDoneWords} * sizeof(uint32_t);
    uint32_t const value = ++mSelfTestValue;
    CUstream stream = nullptr;
    cuCheck(cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING));
    CUresult const waitResult = cuStreamWaitValue32(stream, wordDevice, value, CU_STREAM_WAIT_VALUE_GEQ);
    if (waitResult != CUDA_SUCCESS)
    {
        cuStreamDestroy(stream);
        throw std::runtime_error("store level: completion ring self-test failed: cuStreamWaitValue32 returned "
            + std::to_string(static_cast<int>(waitResult)) + " (TLLM_KV_STORE_TIER_BLOCKING_HOSTFN=1 avoids it)");
    }
    bool const releasedEarly = cuStreamQuery(stream) == CUDA_SUCCESS;
    std::thread flipper(
        [word, value]()
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            std::atomic_ref<uint32_t>(*word).store(value, std::memory_order_release);
        });
    auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    bool released = false;
    while (!released && std::chrono::steady_clock::now() < deadline)
    {
        released = cuStreamQuery(stream) == CUDA_SUCCESS;
        if (!released)
        {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    flipper.join();
    // The word is set, so the wait (if it is still queued) completes and the stream's resources are released then.
    cuStreamDestroy(stream);
    if (releasedEarly)
    {
        throw std::runtime_error("store level: completion ring self-test failed: the stream passed the wait early");
    }
    if (!released)
    {
        throw std::runtime_error(
            "store level: completion ring self-test failed: a stream waiting on the ring was not released within 5 s "
            "(TLLM_KV_STORE_TIER_BLOCKING_HOSTFN=1 avoids it)");
    }
}

void StoreLevel::attach(TllmKvStoreLevelIoV1 const* table, std::vector<std::pair<MemAddress, size_t>> const& ranges,
    StoreLevelLayout const& layout)
{
    if (mTable != nullptr)
    {
        refuseTable("the store level IO is already attached");
    }
    if (table == nullptr)
    {
        refuseTable("null IO table");
    }
    if (table->abiVersion != TLLM_KV_STORE_LEVEL_IO_ABI_VERSION)
    {
        refuseTable("IO table ABI version " + std::to_string(table->abiVersion) + ", expected "
            + std::to_string(TLLM_KV_STORE_LEVEL_IO_ABI_VERSION));
    }
    if (table->structBytes != sizeof(TllmKvStoreLevelIoV1))
    {
        refuseTable("IO table of " + std::to_string(table->structBytes) + " bytes, expected "
            + std::to_string(sizeof(TllmKvStoreLevelIoV1)));
    }
    if (table->maxRowsPerOp == 0)
    {
        refuseTable("IO table takes no rows per op");
    }
    if (table->registerMemory == nullptr || table->unregisterMemory == nullptr || table->configure == nullptr
        || table->submit == nullptr || table->drain == nullptr || table->stats == nullptr)
    {
        refuseTable("IO table has a null entry point");
    }
    if (layout.slotBytes.size() != layout.numSlots.size() || layout.slotBytes.empty())
    {
        refuseTable("store level layout is empty or inconsistent");
    }

    ensureRing();
    bool const blocking = blockingHostFunctionRequested();
    if (!blocking)
    {
        selfTestRing();
    }
    mShared->blockingHostFunction = blocking;

    size_t numRegistered = 0;
    auto unregisterRegistered = [&]()
    {
        while (numRegistered > 0)
        {
            --numRegistered;
            table->unregisterMemory(table->ctx, reinterpret_cast<void*>(ranges[numRegistered].first));
        }
    };
    for (auto const& [base, bytes] : ranges)
    {
        int32_t const code = table->registerMemory(table->ctx, reinterpret_cast<void*>(base), bytes);
        if (code != 0)
        {
            unregisterRegistered();
            failStep("registerMemory", code);
        }
        ++numRegistered;
    }
    int32_t const code = table->configure(
        table->ctx, static_cast<uint32_t>(layout.slotBytes.size()), layout.slotBytes.data(), layout.numSlots.data());
    if (code != 0)
    {
        unregisterRegistered();
        failStep("configure", code);
    }

    mTrackFailures = degradeFaultModeRequested();
    mBreakerS = breakerSecondsRequested();
    mTrackedOps.clear();
    mPendingPutSeq.clear();
    mSlotLost.clear();
    mBreakerUntil = {};
    if (mTrackFailures)
    {
        for (uint64_t const slots : layout.numSlots)
        {
            mPendingPutSeq.emplace_back(static_cast<size_t>(slots), kNoPendingPut);
            mSlotLost.emplace_back(static_cast<size_t>(slots), uint8_t{0});
        }
        mLostSlots.store(0, std::memory_order_relaxed);
    }

    std::lock_guard<std::mutex> lock(mStatsMutex);
    mTable = table;
    mRanges = ranges;
    if (blocking)
    {
        TLLM_LOG_WARNING(
            "store level: TLLM_KV_STORE_TIER_BLOCKING_HOSTFN=1, every IO op holds its CUDA host-function thread");
    }
    if (mTrackFailures)
    {
        TLLM_LOG_INFO(
            "store level: fault mode degrade, a failed store op loses its pages instead of ending the process "
            "(breaker %.1f s)",
            mBreakerS);
    }
}

void StoreLevel::detach(double timeoutS)
{
    if (mTable == nullptr)
    {
        return;
    }
    using Clock = std::chrono::steady_clock;
    auto const deadline = Clock::now()
        + std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(std::max(0.0, timeoutS)));
    uint64_t const enqueued = mOpsEnqueued.load(std::memory_order_relaxed);
    for (;;)
    {
        uint64_t const submitted = mShared->opsSubmitted.load(std::memory_order_acquire);
        if (submitted >= enqueued)
        {
            break;
        }
        if (Clock::now() >= deadline)
        {
            throw std::runtime_error("store level: " + std::to_string(enqueued - submitted) + " of "
                + std::to_string(enqueued) + " enqueued IO ops were never submitted");
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    double const remainingS = std::max(0.0, std::chrono::duration<double>(deadline - Clock::now()).count());
    uint64_t const pending = mTable->drain(mTable->ctx, remainingS);
    if (pending != 0)
    {
        throw std::runtime_error(
            "store level: " + std::to_string(pending) + " IO ops still pending after the drain timed out");
    }
    reapCompletedOps();
    TllmKvStoreLevelIoStats last{};
    mTable->stats(mTable->ctx, &last);
    int32_t firstError = 0;
    for (auto const& range : mRanges)
    {
        int32_t const code = mTable->unregisterMemory(mTable->ctx, reinterpret_cast<void*>(range.first));
        if (code != 0 && firstError == 0)
        {
            firstError = code;
        }
    }
    {
        std::lock_guard<std::mutex> lock(mStatsMutex);
        mLastIoStats = last;
        mTable = nullptr;
    }
    mRanges.clear();
    // Every op completed: the parked streams finish their remaining (GPU-only) work on their own.
    mParked.clear();
    mNumParked.store(0, std::memory_order_relaxed);
    if (firstError != 0)
    {
        failStep("unregisterMemory", firstError);
    }
}

void StoreLevel::enqueue(CUstream stream, uint32_t kind, TllmKvStoreRow const* rows, size_t numRows)
{
    if (mTable == nullptr)
    {
        throw LogicError("store level used before its IO was attached");
    }
    TLLM_CHECK(kind == TLLM_KV_STORE_OP_PUT || kind == TLLM_KV_STORE_OP_GET);
    size_t const maxRows = mTable->maxRowsPerOp;
    for (size_t offset = 0; offset < numRows; offset += maxRows)
    {
        size_t const count = std::min(maxRows, numRows - offset);
        uint64_t const seq = mNextSeq;
        uint32_t* const word = mDoneHost + seq % kDoneWords;
        uint32_t const value = static_cast<uint32_t>(seq / kDoneWords + 1);
        TLLM_CHECK_WITH_INFO(!mTrackFailures || value < TLLM_KV_STORE_DONE_FAILED,
            "store level: the completion ring's lap count reached the failure flag");
        // A degrade IO may have ORed TLLM_KV_STORE_DONE_FAILED into the word of the op one lap earlier.
        if (plainDoneValue(loadDoneWord(word)) != value - 1)
        {
            // 65536 ops in flight: a store stall. Wait (bounded) for the op one lap earlier rather than fail the
            // migration at once.
            auto const deadline = std::chrono::steady_clock::now()
                + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                    std::chrono::duration<double>(std::max(0.0, mLapWaitS)));
            while (plainDoneValue(loadDoneWord(word)) != value - 1)
            {
                if (std::chrono::steady_clock::now() >= deadline)
                {
                    throw std::runtime_error("store level: more than 65536 store ops in flight");
                }
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
        }
        if (loadDoneWord(word) != value - 1)
        {
            // The op one lap earlier failed: fold its outcome in, then clear the flag, which would otherwise satisfy
            // this op's stream wait (value or more) before this op completed.
            reapCompletedOps();
            std::atomic_ref<uint32_t>(*word).store(value - 1, std::memory_order_release);
        }
        auto payload = std::make_unique<SubmitPayload>();
        payload->table = mTable;
        payload->shared = mShared;
        payload->rows.assign(rows + offset, rows + offset + count);
        payload->op.seq = seq;
        payload->op.kind = kind;
        payload->op.numRows = static_cast<uint32_t>(count);
        payload->op.doneWord = word;
        payload->op.doneValue = value;
        TrackedOp tracked{seq, kind, word, value, {}};
        if (mTrackFailures && kind == TLLM_KV_STORE_OP_PUT)
        {
            tracked.putSlots.reserve(count);
            for (size_t index = offset; index < offset + count; ++index)
            {
                TLLM_CHECK_WITH_INFO(rows[index].poolGroup < mPendingPutSeq.size()
                        && rows[index].slot < mPendingPutSeq[rows[index].poolGroup].size(),
                    "store level: a put row outside the layout");
                tracked.putSlots.emplace_back(rows[index].poolGroup, rows[index].slot);
            }
        }
        cuCheck(cuLaunchHostFunc(stream, &storeSubmitTrampoline, payload.get()));
        // The host function owns the payload from here on.
        static_cast<void>(payload.release());
        if (mTrackFailures)
        {
            // Under the exclusive lock, so no reap runs before the op is tracked, however early it completes. A slot is
            // unreadable until its put is confirmed: a get queued behind a put that then fails would read nothing
            // valid.
            for (auto const& [poolGroup, slot] : tracked.putSlots)
            {
                mPendingPutSeq[poolGroup][slot] = seq;
            }
            mTrackedOps.push_back(std::move(tracked));
        }
        ++mNextSeq;
        mOpsEnqueued.fetch_add(1, std::memory_order_relaxed);
        if (!mShared->blockingHostFunction)
        {
            CUdeviceptr const wordDevice = mDoneDevice + (seq % kDoneWords) * sizeof(uint32_t);
            cuCheck(cuStreamWaitValue32(stream, wordDevice, value, CU_STREAM_WAIT_VALUE_GEQ));
        }
    }
}

void StoreLevel::parkStream(CachedCudaStream stream)
{
    mParked.push_back(std::make_unique<CachedCudaStream>(std::move(stream)));
    reapParkedStreams();
}

void StoreLevel::reapParkedStreams()
{
    // Streams finish roughly in park order; a stream stuck behind a slow op only holds back the ones parked after
    // it, which costs pool streams, never a wait.
    while (!mParked.empty() && cuStreamQuery(mParked.front()->handle()) == CUDA_SUCCESS)
    {
        mParked.pop_front();
    }
    mNumParked.store(mParked.size(), std::memory_order_relaxed);
}

void StoreLevel::setSlotLost(uint32_t poolGroup, uint64_t slot, bool lost) noexcept
{
    uint8_t& state = mSlotLost[poolGroup][slot];
    if (static_cast<bool>(state) == lost)
    {
        return;
    }
    state = lost ? 1 : 0;
    if (lost)
    {
        mLostSlots.fetch_add(1, std::memory_order_relaxed);
    }
    else
    {
        mLostSlots.fetch_sub(1, std::memory_order_relaxed);
    }
}

void StoreLevel::reapCompletedOps()
{
    if (!mTrackFailures || mTrackedOps.empty())
    {
        return;
    }
    auto const now = std::chrono::steady_clock::now();
    uint64_t failedGets = 0;
    uint64_t failedPuts = 0;
    // Ops complete out of order (several IO threads): every tracked op is looked at.
    for (auto it = mTrackedOps.begin(); it != mTrackedOps.end();)
    {
        uint32_t const observed = loadDoneWord(it->word);
        if (plainDoneValue(observed) < it->value)
        {
            ++it;
            continue;
        }
        bool const failed = (observed & TLLM_KV_STORE_DONE_FAILED) != 0;
        if (it->kind == TLLM_KV_STORE_OP_PUT)
        {
            for (auto const& [poolGroup, slot] : it->putSlots)
            {
                // A later put of the slot (enqueued after this one) decides its state.
                if (mPendingPutSeq[poolGroup][slot] == it->seq)
                {
                    mPendingPutSeq[poolGroup][slot] = kNoPendingPut;
                    setSlotLost(poolGroup, slot, failed);
                }
            }
            if (failed)
            {
                ++failedPuts;
                auto const until = now
                    + std::chrono::duration_cast<std::chrono::steady_clock::duration>(
                        std::chrono::duration<double>(mBreakerS));
                mBreakerUntil = std::max(mBreakerUntil, until);
            }
        }
        else if (failed)
        {
            ++failedGets;
        }
        it = mTrackedOps.erase(it);
    }
    if (failedPuts != 0)
    {
        mFailedPutOps.fetch_add(failedPuts, std::memory_order_relaxed);
        TLLM_LOG_WARNING(
            "store level: %llu store put op(s) failed (%llu so far): their pages are lost, and store pages are not "
            "reused for %.1f s",
            static_cast<unsigned long long>(failedPuts),
            static_cast<unsigned long long>(mFailedPutOps.load(std::memory_order_relaxed)), mBreakerS);
    }
    if (failedGets != 0)
    {
        mFailedGetOps.fetch_add(failedGets, std::memory_order_relaxed);
        TLLM_LOG_ERROR(
            "store level: %llu store get op(s) failed (%llu so far): they were already queued, so the requests reading "
            "those pages got invalid KV",
            static_cast<unsigned long long>(failedGets),
            static_cast<unsigned long long>(mFailedGetOps.load(std::memory_order_relaxed)));
    }
}

bool StoreLevel::slotReadable(uint32_t poolGroup, uint64_t slot) const noexcept
{
    if (!mTrackFailures)
    {
        return true;
    }
    if (poolGroup >= mPendingPutSeq.size() || slot >= mPendingPutSeq[poolGroup].size())
    {
        return false;
    }
    if (mBreakerUntil != std::chrono::steady_clock::time_point{} && std::chrono::steady_clock::now() < mBreakerUntil)
    {
        return false;
    }
    return mPendingPutSeq[poolGroup][slot] == kNoPendingPut && mSlotLost[poolGroup][slot] == 0;
}

std::pair<bool, TllmKvStoreLevelIoStats> StoreLevel::ioStats() const
{
    std::lock_guard<std::mutex> lock(mStatsMutex);
    if (mTable != nullptr)
    {
        TllmKvStoreLevelIoStats stats{};
        mTable->stats(mTable->ctx, &stats);
        return {true, stats};
    }
    return {false, mLastIoStats};
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
