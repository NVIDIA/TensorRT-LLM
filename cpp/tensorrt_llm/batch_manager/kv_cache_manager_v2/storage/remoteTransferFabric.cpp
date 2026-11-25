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

#include "kv_cache_manager_v2/storage/remoteTransferFabric.h"
#include "kv_cache_manager_v2/exceptions.h"
#include "kv_cache_manager_v2/utils/cudaContextScope.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/logger.h"

#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>

namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
{

namespace
{

std::string cuErrorName(CUresult result)
{
    char const* name = nullptr;
    if (cuGetErrorName(result, &name) != CUDA_SUCCESS || name == nullptr)
    {
        return "CUresult " + std::to_string(static_cast<int>(result));
    }
    return name;
}

//! A submission's copies, enqueued on the caller's stream: concluded at once with the enqueue's outcome.
class FabricTransferAttempt final : public RemoteTransferAttempt
{
public:
    FabricTransferAttempt(std::string fault, CUstream stream)
        : mOutcome{std::move(fault)}
        , mStream(stream)
    {
    }

    std::optional<RemoteTransferOutcome> poll() override
    {
        return mOutcome;
    }

    //! The stream has run everything enqueued on it so far, these copies included.
    bool quiesced() override
    {
        mQuiesced = mQuiesced || cuStreamQuery(mStream) == CUDA_SUCCESS;
        return mQuiesced;
    }

    std::optional<RemoteTransferOutcome> wait(int64_t /*timeoutMs*/) override
    {
        return mOutcome;
    }

private:
    RemoteTransferOutcome mOutcome;
    CUstream mStream;
    bool mQuiesced = false;
};

} // namespace

FabricTransferBackend::FabricTransferBackend()
{
    CUcontext context = nullptr;
    cuCheck(cuCtxGetCurrent(&context));
    if (context == nullptr)
    {
        throw RemoteMemError(
            "KVCM2 remote level: the fabric transport needs a current CUDA context on the calling "
            "thread");
    }
    cuCheck(cuCtxGetDevice(&mDevice));
    // Callers enqueue copies in the device's primary context (as PyTorch does); teardown synchronizes it.
    CUcontext primary = nullptr;
    cuCheck(cuDevicePrimaryCtxRetain(&primary, mDevice));
    if (primary != context)
    {
        cuDevicePrimaryCtxRelease(mDevice);
        throw RemoteMemError("KVCM2 remote level: the fabric transport needs the device's primary context current");
    }
    mContext = primary; // released by release()
}

FabricTransferBackend::~FabricTransferBackend()
{
    release();
}

void FabricTransferBackend::release() noexcept
{
    // Copies still queued on callers' streams must finish before their mappings go.
    if (CUresult const result = synchronizeContext(); result != CUDA_SUCCESS)
    {
        TLLM_LOG_WARNING(
            "KVCM2 remote level: fabric transport teardown: synchronizing failed: %s", cuErrorName(result).c_str());
    }
    for (auto const& mapping : mMappings)
    {
        cuMemUnmap(mapping.address, mapping.bytes);
        cuMemAddressFree(mapping.address, mapping.bytes);
        cuMemRelease(mapping.handle);
    }
    mMappings.clear();
    mMappingEnd.clear();
    if (mContext != nullptr)
    {
        cuDevicePrimaryCtxRelease(mDevice);
        mContext = nullptr;
    }
}

void FabricTransferBackend::drain() noexcept
{
    // Copies on callers' streams reach memory that is about to be released or reused.
    if (CUresult const result = synchronizeContext(); result != CUDA_SUCCESS)
    {
        TLLM_LOG_ERROR("KVCM2 remote level: draining the fabric copies failed: %s", cuErrorName(result).c_str());
    }
}

CUresult FabricTransferBackend::synchronizeContext() const noexcept
{
    if (mContext == nullptr)
    {
        return CUDA_SUCCESS;
    }
    CudaContextScope const scope(mContext);
    return scope.usable() ? cuCtxSynchronize() : scope.error();
}

bool FabricTransferBackend::acceptPayload(RemoteDonor const& donor) const
{
    if (donor.payload.size() != donor.chunks.size() * sizeof(CUmemFabricHandle))
    {
        TLLM_LOG_WARNING(
            "KVCM2 remote level: ignoring donor %s: %zu B of fabric handles for %zu chunk(s), "
            "expected %zu B each",
            donor.name.c_str(), donor.payload.size(), donor.chunks.size(), sizeof(CUmemFabricHandle));
        return false;
    }
    return true;
}

uint64_t FabricTransferBackend::attachChunk(RemoteDonor& donor, size_t chunk)
{
    // The chunk is its own fabric allocation, imported only as it is claimed.
    uint64_t const bytes = donor.chunks[chunk].second;
    CUmemFabricHandle fabricHandle{};
    std::memcpy(&fabricHandle, donor.payload.data() + chunk * sizeof(CUmemFabricHandle), sizeof(fabricHandle));
    CUmemGenericAllocationHandle handle = 0;
    CUresult result = cuMemImportFromShareableHandle(&handle, &fabricHandle, CU_MEM_HANDLE_TYPE_FABRIC);
    if (result != CUDA_SUCCESS)
    {
        throw RemoteMemError("KVCM2 remote level: importing chunk " + std::to_string(chunk) + " of donor "
            + donor.name + " (" + donor.host + ") failed: " + cuErrorName(result)
            + "; fabric memory needs both trays in one NVLink domain (NVL72 rack), the job's IMEX channel and a live "
              "donor");
    }
    CUdeviceptr address = 0;
    result = cuMemAddressReserve(&address, bytes, 0, 0, 0);
    if (result == CUDA_SUCCESS)
    {
        // An imported fabric allocation maps only whole: offset 0, its full size.
        result = cuMemMap(address, bytes, 0, handle, 0);
        if (result == CUDA_SUCCESS)
        {
            CUmemAccessDesc access{};
            access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            access.location.id = static_cast<int>(mDevice);
            access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            result = cuMemSetAccess(address, bytes, &access, 1);
            if (result != CUDA_SUCCESS)
            {
                cuMemUnmap(address, bytes);
            }
        }
        if (result != CUDA_SUCCESS)
        {
            cuMemAddressFree(address, bytes);
        }
    }
    if (result != CUDA_SUCCESS)
    {
        cuMemRelease(handle);
        // A local failure (e.g. out of GPU address space or memory), not the donor's: it fails the claim instead of
        // passing over a live donor.
        throw std::runtime_error("KVCM2 remote level: mapping chunk " + std::to_string(chunk) + " ("
            + std::to_string(bytes) + " B) of donor " + donor.name + " (" + donor.host
            + ") into this GPU failed: " + cuErrorName(result));
    }
    mMappings.push_back(FabricMapping{address, bytes, handle});
    mMappingEnd.emplace(address, address + bytes);
    donor.attached = true;
    return address;
}

std::unique_ptr<RemoteTransferAttempt> FabricTransferBackend::submit(bool write,
    std::vector<RemoteDonor> const& /*donors*/, std::vector<RemoteIo> const& ios, size_t bytes, CUstream stream)
{
    // Runs of pages that are contiguous on both sides go as one copy, within one mapped chunk: two chunks' separate
    // mappings may happen to be adjacent in the address space.
    size_t i = 0;
    while (i < ios.size())
    {
        auto const mapping = mMappingEnd.upper_bound(static_cast<CUdeviceptr>(ios[i].remoteAddr));
        TLLM_CHECK_WITH_INFO(mapping != mMappingEnd.begin(), "KVCM2 remote level: fabric address outside every chunk");
        CUdeviceptr const mappingEnd = std::prev(mapping)->second;
        size_t j = i + 1;
        while (j < ios.size() && ios[j].local == ios[j - 1].local + bytes
            && ios[j].remoteAddr == ios[j - 1].remoteAddr + bytes && ios[j].remoteAddr + bytes <= mappingEnd)
        {
            ++j;
        }
        auto const local = static_cast<CUdeviceptr>(ios[i].local);
        auto const remote = static_cast<CUdeviceptr>(ios[i].remoteAddr);
        size_t const n = (j - i) * bytes;
        CUresult const result
            = write ? cuMemcpyAsync(remote, local, n, stream) : cuMemcpyAsync(local, remote, n, stream);
        if (result != CUDA_SUCCESS)
        {
            // The copies enqueued before it run on; the stream orders them before anything reuses their memory.
            return std::make_unique<FabricTransferAttempt>(std::string("KVCM2 remote level: enqueuing a fabric ")
                    + (write ? "WRITE" : "READ") + " of " + std::to_string(n) + " B failed: " + cuErrorName(result),
                stream);
        }
        i = j;
    }
    return std::make_unique<FabricTransferAttempt>(std::string{}, stream);
}

} // namespace tensorrt_llm::batch_manager::kv_cache_manager_v2
