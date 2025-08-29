#include "tensorrt_llm/batch_manager/kvCacheManagerV2Utils.h"
#include "tensorrt_llm/common/logger.h"
#include <cassert>
#include <cstdio>
#include <cuda.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

template <typename Func>
bool loopedReadWrite(Func&& func, ssize_t size) noexcept
{
    ssize_t count = 0;
    while (count < size)
    {
        ssize_t bytes = func(count);
        if (bytes <= 0)
        {
            if (errno == EINTR)
            {
                continue; // Retry on interrupt
            }
            TLLM_LOG_ERROR("Disk read/write failed: %s\n", strerror(errno));
            return false;
        }
        count += bytes;
    }
    assert(count == size);
    return true;
}

bool writeAll(int fd, ssize_t pos, void const* data, ssize_t size) noexcept
{
    return loopedReadWrite([=](ssize_t finished)
        { return pwrite(fd, static_cast<std::byte const*>(data) + finished, size - finished, pos + finished); },
        size);
}

bool readAll(int fd, ssize_t pos, void* data, ssize_t size) noexcept
{
    return loopedReadWrite([=](ssize_t finished)
        { return pread(fd, static_cast<std::byte*>(data) + finished, size - finished, pos + finished); },
        size);
}

template <typename DstAddr, typename SrcAddr>
struct UserData
{
    std::vector<Task<DstAddr, SrcAddr>> tasks;
    ssize_t numBytes;
};

CUDA_CB void hostFnDiskToDiskCopy(void* userData) noexcept
{
    // @TODO: enable multi-threading with a thread pool
    auto const data = static_cast<UserData<DiskAddress, DiskAddress>*>(userData);
    std::vector<std::byte> buffer(data->numBytes);
    bool success = true;
    for (auto const& t : data->tasks)
    {
        success = success && readAll(t.src.fd, t.src.pos, buffer.data(), data->numBytes);
        success = success && writeAll(t.dst.fd, t.dst.pos, buffer.data(), data->numBytes);
    }
    delete data;
    if (!success)
    {
        TLLM_LOG_ERROR("[kvCacheManagerV2Utils] hostFnDiskToDiskCopy failed.\n");
    }
}

CUDA_CB void hostFnDiskToHostCopy(void* userData) noexcept
{
    // @TODO: enable multi-threading with a thread pool
    auto const data = static_cast<UserData<MemAddress, DiskAddress>*>(userData);
    bool success = true;
    for (auto const& t : data->tasks)
    {
        success = success && readAll(t.src.fd, t.src.pos, reinterpret_cast<void*>(t.dst), data->numBytes);
    }
    delete data;
    if (!success)
    {
        TLLM_LOG_ERROR("[kvCacheManagerV2Utils] hostFnDiskToHostCopy failed.\n");
    }
}

CUDA_CB void hostFnHostToDiskCopy(void* userData) noexcept
{
    // @TODO: enable multi-threading with a thread pool
    auto const data = static_cast<UserData<DiskAddress, MemAddress>*>(userData);
    bool success = true;
    for (auto const& t : data->tasks)
    {
        success = success && writeAll(t.dst.fd, t.dst.pos, reinterpret_cast<void const*>(t.src), data->numBytes);
    }
    delete data;
    if (!success)
    {
        TLLM_LOG_ERROR("[kvCacheManagerV2Utils] hostFnHostToDiskCopy failed.\n");
    }
}

CUDA_CB void hostFnHostToHostCopy(void* userData) noexcept
{
    // @TODO: enable multi-threading with a thread pool
    auto const data = static_cast<UserData<MemAddress, MemAddress>*>(userData);
    for (auto const& t : data->tasks)
    {
        memcpy(reinterpret_cast<void*>(t.dst), reinterpret_cast<void const*>(t.src), data->numBytes);
    }
    delete data;
}

CUresult copyDiskToDisk(
    std::vector<Task<DiskAddress, DiskAddress>> const& tasks, ssize_t numBytes, CUstream stream) noexcept
{
    auto const data = new UserData<DiskAddress, DiskAddress>{std::move(tasks), numBytes};
    return cuLaunchHostFunc(stream, hostFnDiskToDiskCopy, data);
}

CUresult copyDiskToHost(
    std::vector<Task<MemAddress, DiskAddress>> const& tasks, ssize_t numBytes, CUstream stream) noexcept
{
    auto const data = new UserData<MemAddress, DiskAddress>{std::move(tasks), numBytes};
    return cuLaunchHostFunc(stream, hostFnDiskToHostCopy, data);
}

CUresult copyHostToDisk(
    std::vector<Task<DiskAddress, MemAddress>> const& tasks, ssize_t numBytes, CUstream stream) noexcept
{
    auto const data = new UserData<DiskAddress, MemAddress>{std::move(tasks), numBytes};
    return cuLaunchHostFunc(stream, hostFnHostToDiskCopy, data);
}

CUresult copyHostToHost(
    std::vector<Task<MemAddress, MemAddress>> const& tasks, ssize_t numBytes, CUstream stream) noexcept
{
    auto const data = new UserData<MemAddress, MemAddress>{std::move(tasks), numBytes};
    return cuLaunchHostFunc(stream, hostFnHostToHostCopy, data);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
