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

#include "tensorrt_llm/common/opUtils.h"
#include "tensorrt_llm/runtime/utils/mpiTags.h"
#include "tensorrt_llm/runtime/utils/mpiUtils.h"

#include "cuda.h"
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <functional>
#include <mutex>
#include <thread>

#if ENABLE_MULTI_DEVICE

std::unordered_map<nvinfer1::DataType, ncclDataType_t>* getDtypeMap()
{
    static std::unordered_map<nvinfer1::DataType, ncclDataType_t> dtypeMap = {
        {nvinfer1::DataType::kFLOAT, ncclFloat32},
        {nvinfer1::DataType::kHALF, ncclFloat16},
        {nvinfer1::DataType::kBF16, ncclBfloat16},
        {nvinfer1::DataType::kFP8, ncclInt8},
        {nvinfer1::DataType::kBOOL, ncclInt8},
        {nvinfer1::DataType::kINT32, ncclInt32},
        {nvinfer1::DataType::kINT64, ncclInt64},
        {nvinfer1::DataType::kUINT8, ncclUint8},
        {nvinfer1::DataType::kINT8, ncclInt8},
    };
    return &dtypeMap;
}

namespace
{

// Get NCCL unique ID for a group of ranks.
ncclUniqueId getUniqueId(std::set<int> const& group)
{
    auto const rank = COMM_SESSION.getRank();
    TLLM_LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, rank);
    ncclUniqueId id;
    if (rank == *group.begin())
    {
        NCCLCHECK_THROW(ncclGetUniqueId(&id));
        for (auto it = std::next(std::begin(group), 1); it != group.end(); ++it)
        {
            COMM_SESSION.sendValue(id, *it, tensorrt_llm::mpi::MpiTag::kDefault);
        }
    }
    else
    {
        COMM_SESSION.recvValue(id, *group.begin(), tensorrt_llm::mpi::MpiTag::kDefault);
    }
    TLLM_LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, rank);
    return id;
}
} // namespace

std::shared_ptr<ncclComm_t> getComm(std::set<int> const& group)
{
    auto const rank = COMM_SESSION.getRank();
    TLLM_LOG_TRACE("%s start for rank %d", __PRETTY_FUNCTION__, rank);
    static std::map<std::set<int>, std::shared_ptr<ncclComm_t>> commMap;
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);
    std::ostringstream oss;
    int index = 0;
    for (auto const& rank : group)
    {
        if (index != 0)
        {
            oss << ",";
        }
        oss << rank;
        index++;
    }
    auto groupStr = oss.str();
    auto it = commMap.find(group);
    if (it != commMap.end())
    {
        auto ncclComm = it->second;
        TLLM_LOG_TRACE("NCCL comm for group(%s) is cached for rank %d", groupStr.c_str(), rank);
        return ncclComm;
    }

    TLLM_LOG_TRACE("Init NCCL comm for group(%s) for rank %d", groupStr.c_str(), rank);
    ncclUniqueId id = getUniqueId(group);
    int groupRank = 0;
    for (auto const& currentRank : group)
    {
        if (rank == currentRank)
            break;
        ++groupRank;
    }
    TLLM_CHECK(static_cast<size_t>(groupRank) < group.size());
    std::shared_ptr<ncclComm_t> ncclComm(new ncclComm_t,
        [](ncclComm_t* comm)
        {
            ncclCommDestroy(*comm);
            delete comm;
        });
#if defined(_WIN32)
    // Need static connection initialization for accurate KV cache size estimation
    if (getenv("NCCL_RUNTIME_CONNECT") == nullptr)
        _putenv_s("NCCL_RUNTIME_CONNECT", "0");
    // Disable graph register to avoid startup hangs
    if (getenv("NCCL_GRAPH_REGISTER") == nullptr)
        _putenv_s("NCCL_GRAPH_REGISTER", "0");
#else
    setenv("NCCL_RUNTIME_CONNECT", "0", 0);
    setenv("NCCL_GRAPH_REGISTER", "0", 0);
#endif // _WIN32
    NCCLCHECK_THROW(ncclCommInitRank(ncclComm.get(), group.size(), id, groupRank));
    commMap[group] = ncclComm;
    TLLM_LOG_TRACE("%s stop for rank %d", __PRETTY_FUNCTION__, rank);
    return ncclComm;
}
#endif // ENABLE_MULTI_DEVICE

void const* tensorrt_llm::common::op::getCommSessionHandle()
{
#if ENABLE_MULTI_DEVICE
    return &COMM_SESSION;
#else
    return nullptr;
#endif // ENABLE_MULTI_DEVICE
}

namespace
{
using tensorrt_llm::common::op::hash;

// Get current cuda context, a default context will be created if there is no context.
inline CUcontext getCurrentCudaCtx()
{
    CUcontext ctx{};
    CUresult err = cuCtxGetCurrent(&ctx);
    if (err == CUDA_ERROR_NOT_INITIALIZED || ctx == nullptr)
    {
        TLLM_CUDA_CHECK(cudaFree(nullptr));
        err = cuCtxGetCurrent(&ctx);
    }
    TLLM_CHECK(err == CUDA_SUCCESS);
    return ctx;
}

// Helper to create per-cuda-context and per-thread singleton managed by std::shared_ptr.
// Unlike conventional singletons, singleton created with this will be released
// when not needed, instead of on process exit.
// Objects of this class shall always be declared static / global, and shall never own CUDA
// resources.
template <typename T>
class PerCudaCtxPerThreadSingletonCreator
{
public:
    using CreatorFunc = std::function<std::unique_ptr<T>()>;
    using DeleterFunc = std::function<void(T*)>;

    // creator returning std::unique_ptr is by design.
    // It forces separation of memory for T and memory for control blocks.
    // So when T is released, but we still have observer weak_ptr in mObservers, the T mem block can be released.
    // creator itself must not own CUDA resources. Only the object it creates can.
    PerCudaCtxPerThreadSingletonCreator(CreatorFunc creator, DeleterFunc deleter)
        : mCreator{std::move(creator)}
        , mDeleter{std::move(deleter)}
    {
    }

    std::shared_ptr<T> operator()()
    {
        std::lock_guard<std::mutex> lk{mMutex};
        CUcontext ctx{getCurrentCudaCtx()};
        std::thread::id thread = std::this_thread::get_id();
        auto const key = std::make_tuple(ctx, thread);
        std::shared_ptr<T> result = mObservers[key].lock();
        if (result == nullptr)
        {
            TLLM_LOG_TRACE("creating singleton instance for CUDA context %lu and thread %lu", ctx, thread);
            // Create the resource and register with an observer.
            result = std::shared_ptr<T>{mCreator().release(),
                [this, key](T* obj)
                {
                    if (obj == nullptr)
                    {
                        return;
                    }
                    mDeleter(obj);

                    // Clears observer to avoid growth of mObservers, in case users creates/destroys cuda contexts
                    // frequently.
                    std::shared_ptr<T> observedObjHolder; // Delay destroy to avoid dead lock.
                    std::lock_guard<std::mutex> lk{mMutex};
                    // Must check observer again because another thread may created new instance for this ctx and this
                    // thread just before we lock mMutex. We can't infer that the observer is stale from the fact that
                    // obj is destroyed, because shared_ptr ref-count checking and observer removing are not in one
                    // atomic operation, and the observer may be changed to observe another instance.
                    if (mObservers.find(key) == mObservers.end())
                    {
                        return;
                    }
                    observedObjHolder = mObservers.at(key).lock();
                    if (observedObjHolder == nullptr)
                    {
                        mObservers.erase(key);
                    }
                }};
            mObservers.at(key) = result;
        }
        else
        {
            TLLM_LOG_TRACE("singleton instance for CUDA context %d and thread %d is cached", ctx, thread);
        }
        return result;
    }

private:
    CreatorFunc mCreator;
    DeleterFunc mDeleter;
    mutable std::mutex mMutex;
    // CUDA resources are per-context and per-thread.
    using CacheKey = std::tuple<CUcontext, std::thread::id>;
    std::unordered_map<CacheKey, std::weak_ptr<T>, hash<CacheKey>> mObservers;
};

} // namespace

std::shared_ptr<cublasHandle_t> getCublasHandle()
{
    static PerCudaCtxPerThreadSingletonCreator<cublasHandle_t> creator(
        []() -> auto
        {
            auto handle = std::unique_ptr<cublasHandle_t>(new cublasHandle_t);
            TLLM_CUDA_CHECK(cublasCreate(handle.get()));
            return handle;
        },
        [](cublasHandle_t* handle)
        {
            TLLM_CUDA_CHECK(cublasDestroy(*handle));
            delete handle;
        });
    return creator();
}

std::shared_ptr<cublasLtHandle_t> getCublasLtHandle()
{
    static PerCudaCtxPerThreadSingletonCreator<cublasLtHandle_t> creator(
        []() -> auto
        {
            auto handle = std::unique_ptr<cublasLtHandle_t>(new cublasLtHandle_t);
            TLLM_CUDA_CHECK(cublasLtCreate(handle.get()));
            return handle;
        },
        [](cublasLtHandle_t* handle)
        {
            TLLM_CUDA_CHECK(cublasLtDestroy(*handle));
            delete handle;
        });
    return creator();
}
