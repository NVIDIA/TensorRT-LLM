/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

// example.cu — demonstrates CudaVmmArena grow/shrink.
//
// Build:
//   nvcc -std=c++17 -lcuda example.cu cuda_vmm_arena.cpp -o vmm_example
//
// Requirements: CUDA 10.2+, GPU with compute capability >= 7.0,
//               driver support for cuMemAddressReserve.

#include "tensorrt_llm/batch_manager/cudaVmmArena.h"

#include <cstdio>

using namespace tensorrt_llm::batch_manager::vmm;
#include <cstring>

// Simple kernel: write the index into each element.
__global__ void fill_kernel(int* data, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        data[i] = i;
}

// Verify the fill on the host.
static bool verify(int const* host, int n)
{
    for (int i = 0; i < n; ++i)
    {
        if (host[i] != i)
        {
            std::printf("MISMATCH at index %d: got %d, expected %d\n", i, host[i], i);
            return false;
        }
    }
    return true;
}

int main()
{
    // Initialize the CUDA driver API.
    if (cuInit(0) != CUDA_SUCCESS)
    {
        std::printf("cuInit failed\n");
        return 1;
    }

    CUdevice dev{};
    CUcontext ctx{};
    cuDeviceGet(&dev, 0);
    cuCtxCreate(&ctx, nullptr, 0, dev);

    try
    {
        constexpr size_t MB = 1ULL << 20;
        constexpr size_t MAX = 512 * MB; // reserve 512 MiB VA
        constexpr size_t STEP = 16 * MB; // commit in 16 MiB increments

        CudaVmmArena arena(MAX, /*device=*/0);
        std::printf("Granularity : %zu bytes\n", arena.granularity());
        std::printf("Reserved VA : %zu bytes (%zu MiB)\n", arena.max_size(), arena.max_size() / MB);

        // ----------------------------------------------------------------
        // Phase 1: grow to 32 MiB in two steps.
        // ----------------------------------------------------------------
        arena.grow(STEP);
        std::printf("\n[grow] committed = %zu MiB\n", arena.committed_size() / MB);

        int n1 = static_cast<int>(arena.committed_size() / sizeof(int));
        int* d_ptr = reinterpret_cast<int*>(arena.ptr());
        fill_kernel<<<(n1 + 255) / 256, 256>>>(d_ptr, n1);
        cudaDeviceSynchronize();

        arena.grow(2 * STEP);
        std::printf("[grow] committed = %zu MiB\n", arena.committed_size() / MB);

        int n2 = static_cast<int>(arena.committed_size() / sizeof(int));
        fill_kernel<<<(n2 + 255) / 256, 256>>>(d_ptr, n2);
        cudaDeviceSynchronize();

        // Copy back and verify.
        std::vector<int> host(n2);
        cudaMemcpy(host.data(), d_ptr, n2 * sizeof(int), cudaMemcpyDeviceToHost);
        std::printf("Verify 32 MiB fill: %s\n", verify(host.data(), n2) ? "OK" : "FAIL");

        // ----------------------------------------------------------------
        // Phase 2: shrink back to 16 MiB — verify original data still intact.
        // ----------------------------------------------------------------
        arena.shrink(STEP);
        std::printf("\n[shrink] committed = %zu MiB\n", arena.committed_size() / MB);

        int n3 = static_cast<int>(arena.committed_size() / sizeof(int));
        std::vector<int> host2(n3);
        cudaMemcpy(host2.data(), d_ptr, n3 * sizeof(int), cudaMemcpyDeviceToHost);
        std::printf("Verify 16 MiB region still intact: %s\n", verify(host2.data(), n3) ? "OK" : "FAIL");

        // ----------------------------------------------------------------
        // Phase 3: use resize() to go back up.
        // ----------------------------------------------------------------
        arena.resize(4 * STEP);
        std::printf("\n[resize] committed = %zu MiB\n", arena.committed_size() / MB);

        std::printf("\nAll phases completed successfully.\n");
    }
    catch (CudaVmmError const& e)
    {
        std::printf("CudaVmmError: %s (CUresult=%d)\n", e.what(), e.result());
        return 1;
    }

    cuCtxDestroy(ctx);
    return 0;
}
