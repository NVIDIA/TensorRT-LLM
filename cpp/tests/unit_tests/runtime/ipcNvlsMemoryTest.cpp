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

#include "tensorrt_llm/runtime/ipcNvlsMemory.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

namespace tr = tensorrt_llm::runtime;

namespace
{
#if ENABLE_MULTI_DEVICE
// Recompute the static NVLS multicast capability exactly as ipcNvlsSupported()
// must: CUDA driver >= 12010 and CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED on the
// device. Deliberately does NO fabric/IMEX probe -- that is what
// ipcNvlsFabricUsable() adds on top.
bool computeStaticNvlsCapability()
{
    int driverVersion = -1;
    TLLM_CUDA_CHECK(cudaDriverGetVersion(&driverVersion));
    if (driverVersion < 12010)
    {
        return false;
    }
    int cudaDev = -1;
    TLLM_CUDA_CHECK(cudaGetDevice(&cudaDev));
    CUdevice device{};
    EXPECT_EQ(cuDeviceGet(&device, cudaDev), CUDA_SUCCESS);
    int multicastSupported = 0;
    EXPECT_EQ(cuDeviceGetAttribute(&multicastSupported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, device), CUDA_SUCCESS);
    return multicastSupported != 0;
}
#endif // ENABLE_MULTI_DEVICE

class IpcNvlsMemoryTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
#if !ENABLE_MULTI_DEVICE
        GTEST_SKIP() << "ipcNvls* helpers require ENABLE_MULTI_DEVICE.";
#else
        int deviceCount = 0;
        TLLM_CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
        if (deviceCount == 0)
        {
            GTEST_SKIP() << "No CUDA devices available.";
        }
        // Ensure the CUDA runtime (and primary context) is initialized so the
        // driver-API device attribute query below is valid.
        TLLM_CUDA_CHECK(cudaFree(nullptr));
#endif // ENABLE_MULTI_DEVICE
    }
};

// Regression guard for https://nvbugs/6396420.
//
// ipcNvlsSupported() must report only the *static* multicast capability
// (driver + CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED). It must NOT depend on
// whether the NVLink fabric/IMEX plane is provisioned: single-node NVLS is
// allocated over a POSIX-FD handle and does not need the fabric plane.
//
// PR #15302 folded a fabric probe into ipcNvlsSupported(), which on a
// multicast-capable but fabric-unprovisioned node made it return false. That
// wrongly disabled the single-node NVLS allocator (ipcNvlsAllocate) and fused
// GEMM-allreduce. The fabric probe now lives in ipcNvlsFabricUsable(); if it
// leaks back into ipcNvlsSupported() this assertion fails on such nodes.
TEST_F(IpcNvlsMemoryTest, SupportedReflectsStaticCapabilityOnly)
{
#if ENABLE_MULTI_DEVICE
    EXPECT_EQ(tr::ipcNvlsSupported(), computeStaticNvlsCapability());
#endif // ENABLE_MULTI_DEVICE
}

// Fabric usability is a strict refinement of static capability, so it must
// never be reported as usable when NVLS is not even statically supported.
TEST_F(IpcNvlsMemoryTest, FabricUsableImpliesSupported)
{
#if ENABLE_MULTI_DEVICE
    if (tr::ipcNvlsFabricUsable())
    {
        EXPECT_TRUE(tr::ipcNvlsSupported());
    }
#endif // ENABLE_MULTI_DEVICE
}
} // namespace
