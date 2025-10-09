/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/runtime/utils/mpiTags.h"
#include "tensorrt_llm/runtime/utils/multiDeviceUtils.h"

#include <functional>
#include <limits>

#ifdef ENABLE_FP8
#include <cuda_fp8.h>
#endif
#ifdef ENABLE_BF16
#include <cuda_bf16.h>
#endif

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>

#if ENABLE_MULTI_DEVICE
#include <mpi.h>
#else
// Dummy defines to avoid #if in wider places.
typedef void* MPI_Datatype;
typedef void* MPI_Comm;
typedef void* MPI_Request;
typedef void* MPI_Message;
typedef void* MPI_Op;

typedef struct MPI_Status
{
    int dummy;
} MPI_Status;

#define MPI_THREAD_SINGLE 0
#define MPI_THREAD_FUNNELED 1
#define MPI_THREAD_SERIALIZED 2
#define MPI_THREAD_MULTIPLE 3
#define MPI_COMM_WORLD ((MPI_Comm) 0x44000000)
#define MPI_COMM_NULL ((MPI_Comm) 0x04000000)
#endif // ENABLE_MULTI_DEVICE

#include <type_traits>
#include <vector>

#define MPICHECK(cmd) TLLM_MPI_CHECK(cmd)

namespace tensorrt_llm::runtime
{
class IBuffer;
}

// A wrapper module of the MPI library.
namespace tensorrt_llm::mpi
{

// A wrapper of MPI data type. MpiType::{data_type}
enum class MpiType
{
    kBYTE,
    kHALF,
    kFLOAT,
    kDOUBLE,
    kBOOL,
    kINT8,
    kUINT8,
    kINT32,
    kUINT32,
    kINT64,
    kUINT64,
    kFP8,
    kBF16,
    kCHAR,
};

//! \brief For converting a C++ data type to a TensorRT data type.
template <typename T>
struct MpiTypeConverter
{
};

template <>
struct MpiTypeConverter<std::byte>
{
    static constexpr auto value = MpiType::kBYTE;
};

template <>
struct MpiTypeConverter<half>

{
    static constexpr auto value = MpiType::kHALF;
};

template <>
struct MpiTypeConverter<float>
{
    static constexpr auto value = MpiType::kFLOAT;
};

template <>
struct MpiTypeConverter<double>
{
    static constexpr auto value = MpiType::kDOUBLE;
};

template <>
struct MpiTypeConverter<bool>
{
    static constexpr auto value = MpiType::kBOOL;
};

template <>
struct MpiTypeConverter<std::int8_t>
{
    static constexpr auto value = MpiType::kINT8;
};

template <>
struct MpiTypeConverter<std::uint8_t>

{
    static constexpr auto value = MpiType::kUINT8;
};

template <>
struct MpiTypeConverter<std::int32_t>
{
    static constexpr auto value = MpiType::kINT32;
};

template <>
struct MpiTypeConverter<std::uint32_t>
{
    static constexpr auto value = MpiType::kUINT32;
};

template <>
struct MpiTypeConverter<std::int64_t>
{
    static constexpr auto value = MpiType::kINT64;
};

template <>
struct MpiTypeConverter<std::uint64_t>
{
    static constexpr auto value = MpiType::kUINT64;
};

template <>
struct MpiTypeConverter<char>
{
    static constexpr auto value = MpiType::kCHAR;
};

#ifdef ENABLE_FP8
template <>
struct MpiTypeConverter<__nv_fp8_e4m3>
{
    static constexpr auto value = MpiType::kFP8;
};
#endif

#ifdef ENABLE_BF16
template <>
struct MpiTypeConverter<__nv_bfloat16>
{
    static constexpr auto value = MpiType::kBF16;
};
#endif

// A wrapper of MPI_Op type.
enum class MpiOp
{
    NULLOP,
    MAX,
    MIN,
    SUM,
    PROD,
    LAND,
    BAND,
    LOR,
    BOR,
    LXOR,
    BXOR,
    MINLOC,
    MAXLOC,
    REPLACE,
};

// A wrapper of the level of MPI thread support
enum class MpiThreadSupport : int
{
    THREAD_SINGLE = MPI_THREAD_SINGLE,
    THREAD_FUNNELED = MPI_THREAD_FUNNELED,
    THREAD_SERIALIZED = MPI_THREAD_SERIALIZED,
    THREAD_MULTIPLE = MPI_THREAD_MULTIPLE,
};

class MpiRequest
{
public:
    MpiRequest() = default;

    ~MpiRequest() = default;

    void wait()
    {
#if ENABLE_MULTI_DEVICE
        // TODO: Don't ignore return status
        TLLM_MPI_CHECK(MPI_Wait(&mRequest, MPI_STATUS_IGNORE));
#else
        TLLM_THROW("Multi device support is disabled.");
#endif
    }

    void cancel()
    {
#if ENABLE_MULTI_DEVICE
        TLLM_MPI_CHECK(MPI_Cancel(&mRequest));
#else
        TLLM_THROW("Multi device support is disabled.");
#endif
    }

    MPI_Request mRequest{};
};

MPI_Datatype getMpiDtype(MpiType dtype);

class MpiComm
{
public:
    explicit MpiComm(MPI_Comm g, bool freeComm);
    ~MpiComm() noexcept;

    // no copy
    MpiComm(MpiComm const&) = delete;
    MpiComm& operator=(MpiComm const&) = delete;

    // move
    MpiComm(MpiComm&&) noexcept;
    MpiComm& operator=(MpiComm&&) noexcept;

    [[nodiscard]] int getRank() const;
    [[nodiscard]] int getSize() const;

    operator MPI_Comm() const // NOLINT(*-explicit-constructor)
    {
        return mComm;
    }

    //! \brief Returns the MPI world communicator.
    static MpiComm const& world();

    //! \brief Corresponds to `world()` by default, but can be overridden per process.
    static MpiComm const& session()
    {
        return mutableSession();
    }

    //! \brief Returns the MPI local communicator.
    static MpiComm const& localSession()
    {
        return mutableLocalSession();
    }

    static MpiComm const& setSession(MpiComm comm)
    {
        auto& session = mutableSession();
        session = std::move(comm);
        refreshLocalSession();
        return session;
    }

    static MpiComm const& setRawSessionByFortran(int64_t fortranHandle);

    [[nodiscard]] MpiComm split(int color, int key) const;

    std::unique_ptr<MpiRequest> bcastAsync(void* buffer, size_t size, MpiType dtype, int root) const;

    std::unique_ptr<MpiRequest> bcastAsync(runtime::IBuffer& buf, int root) const;

    void bcast(void* buffer, size_t size, MpiType dtype, int root) const;

    void bcast(runtime::IBuffer& buf, int root) const;

    template <typename T>
    void bcastValue(T& value, int root) const
    {
        if constexpr (std::is_fundamental_v<std::remove_cv_t<T>>)
        {
            bcast(&value, 1, MpiTypeConverter<std::remove_cv_t<T>>::value, root);
        }
        else
        {
            bcast(&value, sizeof(T), MpiType::kBYTE, root);
        }
    }

    template <typename T>
    void bcast(std::vector<T>& vec, int root) const
    {
        auto const rank = getRank();
        auto vecSize = (rank == root) ? static_cast<int64_t>(vec.size()) : int64_t(0);
        bcast(&vecSize, 1, MpiType::kINT64, root);
        vec.resize(vecSize);
        if (vec.empty())
        {
            return;
        }

        size_t bcastSize = vec.size() * sizeof(T);
        if constexpr (std::is_fundamental_v<std::remove_cv_t<T>>)
        {
            bcastSize = vec.size();
        }

        // To prevent overflowing int32_t limit
        size_t const maxChunkSize = std::numeric_limits<int32_t>::max();
        for (size_t pos = 0; pos < bcastSize; pos += maxChunkSize)
        {
            auto chunkSize = std::min(bcastSize - pos, maxChunkSize);
            auto intChunkSize = static_cast<int>(chunkSize);
            if constexpr (std::is_fundamental_v<std::remove_cv_t<T>>)
            {
                bcast(vec.data() + pos, intChunkSize, MpiTypeConverter<std::remove_cv_t<T>>::value, root);
            }
            else
            {
                bcast(reinterpret_cast<char*>(vec.data()) + pos, intChunkSize, MpiType::kBYTE, root);
            }
        }
    }

    std::unique_ptr<MpiRequest> sendAsync(
        void const* buffer, std::size_t size, MpiType dtype, int dest, MpiTag tag) const;
    std::unique_ptr<MpiRequest> sendAsync(runtime::IBuffer const& buf, int dest, MpiTag tag) const;
    //! \deprecated This function is discouraged. Use the one with MpiTag enum instead.
    void sendRawTag(void const* buffer, std::size_t size, MpiType dtype, int dest, int tag) const;
    void send(void const* buffer, std::size_t size, MpiType dtype, int dest, MpiTag tag) const;
    void send(runtime::IBuffer const& buf, int dest, MpiTag tag) const;

    template <typename T>
    void sendValue(T const& value, int dest, MpiTag tag) const
    {
        if constexpr (std::is_fundamental_v<std::remove_cv_t<T>>)
        {
            send(&value, 1, MpiTypeConverter<std::remove_cv_t<T>>::value, dest, tag);
        }
        else
        {
            send(&value, sizeof(T), MpiType::kBYTE, dest, tag);
        }
    }

    //! \deprecated This function is discouraged. Use the one with MpiTag enum instead.
    MPI_Status recvRawTag(void* buffer, size_t size, MpiType dtype, int source, int tag) const;
    MPI_Status recv(void* buffer, size_t size, MpiType dtype, int source, MpiTag tag) const;
    MPI_Status recv(runtime::IBuffer& buf, int source, MpiTag tag) const;

    template <typename T>
    MPI_Status recvValue(T& value, int source, MpiTag tag) const
    {
#if ENABLE_MULTI_DEVICE
        if constexpr (std::is_fundamental_v<std::remove_cv_t<T>>)
        {
            return recv(&value, 1, MpiTypeConverter<std::remove_cv_t<T>>::value, source, tag);
        }
        else
        {
            return recv(&value, sizeof(T), MpiType::kBYTE, source, tag);
        }
#else
        TLLM_THROW("Multi device support is disabled.");
#endif
    }

    void allreduce(void const* sendbuf, void* recvbuf, int count, MpiType dtype, MpiOp op) const;
    void allgather(void const* sendbuf, void* recvbuf, int count, MpiType dtype) const;

    void allgatherv(void const* sendbuf, int sendcount, MpiType sendtype, void* recvbuf,
        std::vector<int> const& recvcounts, std::vector<int> const& displs, MpiType recvtype) const;

    void barrier() const;

    //! \deprecated This function is discouraged. Use the one with MpiTag enum instead.
    void mprobeRawTag(int source, int tag, MPI_Message* msg, MPI_Status* status) const;
    void mprobe(int source, MpiTag tag, MPI_Message* msg, MPI_Status* status) const;
    bool improbe(int source, MpiTag tag, MPI_Message* msg, MPI_Status* status) const;

    //! \brief Returns if a message with the specified source and tag is available
    bool iprobe(int source, MpiTag tag, MPI_Status* status) const;

    //! \brief Poll every periodMs until a message is available
    void recvPoll(int source, MpiTag tag, int periodMs) const;

    bool operator==(MpiComm const& rhs) const
    {
        return mComm == rhs.mComm;
    }

    bool operator!=(MpiComm const& rhs) const
    {
        return !(rhs == *this);
    }

    bool couldUseMPI() const
    {
        if (!mDisableMPI.has_value())
        {
            char* val = std::getenv("TLLM_DISABLE_MPI");
            if (val != NULL && std::string(val) == "1")
            {
                mDisableMPI = true;
            }
            else
            {
                mDisableMPI = false;
            }
        }
        if (mDisableMPI.value())
        {
            throw std::runtime_error("MPI is disabled, DON\'T USE MPI");
        }
        return true;
    }

private:
    mutable std::optional<bool> mDisableMPI;
    //! \brief Corresponds to `world()` by default, but can be overridden per process.
    static MpiComm& mutableSession();

    //! \brief Returns the MPI local communicator.
    static MpiComm& mutableLocalSession();

    static void refreshLocalSession();

    MPI_Comm mComm;
    bool mFreeComm;
};

std::vector<int> getWorldRanks(MpiComm const& comm);

int getNumNodes();

void initialize(MpiThreadSupport threadMode = MpiThreadSupport::THREAD_MULTIPLE, bool forwardAbortToParent = false);

class MpiWaitThread
{
public:
    explicit MpiWaitThread(std::string name, std::function<void()> funcWait, std::function<void()> funcSetup = nullptr);
    ~MpiWaitThread();

    void waitStop();
    void notifyStart();

private:
    void sideThread();

    void waitStart();
    void notifyStop();

    std::string mName;
    std::function<void()> mFuncWait;
    std::function<void()> mFuncSetup;
    std::unique_ptr<std::thread> mThread;
    std::mutex mMutex;
    std::condition_variable mCondVar;
    bool mRunning{true};
    std::atomic<bool> mShouldExit{false};
};

} // namespace tensorrt_llm::mpi

#define COMM_SESSION tensorrt_llm::mpi::MpiComm::session()
#define LOCAL_COMM_SESSION tensorrt_llm::mpi::MpiComm::localSession()
