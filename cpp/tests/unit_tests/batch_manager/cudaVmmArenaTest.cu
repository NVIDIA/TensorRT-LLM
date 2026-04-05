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

// test_cuda_vmm_arena.cu — Google Test suite for CudaVmmArena.
//
// Build (adjust paths to your GTest installation):
//   nvcc -std=c++17 -lcuda \
//        -I/path/to/googletest/include \
//        -L/path/to/googletest/lib -lgtest -lgtest_main \
//        test_cuda_vmm_arena.cu ../cuda_vmm_arena.cpp \
//        -o test_vmm_arena
//
// Or with CMake, see the accompanying CMakeLists.txt.

#include "tensorrt_llm/batch_manager/cudaVmmArena.h"

#include <gtest/gtest.h>

using namespace tensorrt_llm::batch_manager::vmm;
#include <cstdint>
#include <cuda_runtime.h> // cudaMemcpy, cudaDeviceSynchronize
#include <numeric>
#include <vector>

// ---------------------------------------------------------------------------
// Test fixture: initialises CUDA driver once per process, creates a CUDA
// context, and provides a fresh arena for each test case.
// ---------------------------------------------------------------------------

/// Per-process CUDA driver + context setup.
class CudaEnv : public ::testing::Environment
{
public:
    void SetUp() override
    {
        ASSERT_EQ(cuInit(0), CUDA_SUCCESS) << "cuInit failed";
        ASSERT_EQ(cuDeviceGet(&dev_, 0), CUDA_SUCCESS);
        ASSERT_EQ(cuCtxCreate(&ctx_, nullptr, 0, dev_), CUDA_SUCCESS);
    }

    void TearDown() override
    {
        if (ctx_)
            cuCtxDestroy(ctx_);
    }

    static CUdevice dev_;
    static CUcontext ctx_;
};

CUdevice CudaEnv::dev_{};
CUcontext CudaEnv::ctx_{};

/// Base fixture used by all test cases.
class ArenaTest : public ::testing::Test
{
protected:
    // Default arena parameters.  Each test that needs different values
    // creates its own arena inline.
    static constexpr size_t kMaxSize = 256ULL << 20; // 256 MiB reserved VA

    // Skip the test early if the device doesn't support VMM.
    // GTEST_SKIP() requires a void context — SetUp() qualifies.
    void SetUp() override
    {
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = 0;
        size_t g = 0;
        CUresult res = cuMemGetAllocationGranularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        if (res == CUDA_ERROR_NOT_SUPPORTED || res == CUDA_ERROR_NO_DEVICE)
            GTEST_SKIP() << "VMM not supported on this system (CUresult=" << res << ")";
        if (res != CUDA_SUCCESS || g == 0)
            GTEST_SKIP() << "Could not query VMM granularity (CUresult=" << res << ")";
    }

    // Helper: build a fresh arena. Throws CudaVmmError on hard failure.
    static CudaVmmArena* make_arena(size_t max_size = kMaxSize, int device = 0)
    {
        return new CudaVmmArena(max_size, device);
    }

    // Granularity helper (we need it before the arena exists in some tests).
    static size_t query_granularity(int device = 0)
    {
        CUmemAllocationProp prop{};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device;
        size_t g = 0;
        cuMemGetAllocationGranularity(&g, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM);
        return g;
    }
};

// ===========================================================================
// Construction
// ===========================================================================

TEST_F(ArenaTest, ConstructionSetsInitialState)
{
    std::unique_ptr<CudaVmmArena> a(make_arena(kMaxSize));

    EXPECT_GT(a->granularity(), 0u);
    EXPECT_GE(a->max_size(), kMaxSize); // rounded up >= requested
    EXPECT_EQ(a->committed_size(), 0u); // nothing committed yet
    EXPECT_NE(a->ptr(), 0u);            // VA range was reserved
    EXPECT_EQ(a->device(), 0);
}

TEST_F(ArenaTest, MaxSizeAlignedUpToGranularity)
{
    const size_t g = query_granularity();
    if (g == 0)
        GTEST_SKIP() << "Cannot query granularity";

    // Request a size that is deliberately not aligned.
    const size_t unaligned = g + 1;
    std::unique_ptr<CudaVmmArena> a(make_arena(unaligned));

    EXPECT_EQ(a->max_size() % g, 0u) << "max_size must be a multiple of granularity";
    EXPECT_GE(a->max_size(), unaligned);
}

TEST_F(ArenaTest, SmallestPossibleReservation)
{
    // A reservation of exactly one granule should succeed.
    const size_t g = query_granularity();
    if (g == 0)
        GTEST_SKIP();

    std::unique_ptr<CudaVmmArena> a(make_arena(g));
    EXPECT_EQ(a->max_size(), g);
    EXPECT_EQ(a->committed_size(), 0u);
}

// ===========================================================================
// grow()
// ===========================================================================

TEST_F(ArenaTest, GrowCommitsSingleChunk)
{
    std::unique_ptr<CudaVmmArena> a(make_arena());
    const size_t g = a->granularity();

    a->grow(g);

    EXPECT_EQ(a->committed_size(), g);
}

TEST_F(ArenaTest, GrowCommitsMultipleChunks)
{
    std::unique_ptr<CudaVmmArena> a(make_arena());
    const size_t g = a->granularity();

    a->grow(4 * g);

    EXPECT_EQ(a->committed_size(), 4 * g);
}

TEST_F(ArenaTest, GrowRoundsUpToGranularity)
{
    std::unique_ptr<CudaVmmArena> a(make_arena());
    const size_t g = a->granularity();

    // Request 1 byte — should round up to one granule.
    a->grow(1);

    EXPECT_EQ(a->committed_size(), g);
    EXPECT_EQ(a->committed_size() % g, 0u);
}

TEST_F(ArenaTest, GrowIncrementallyAccumulatesCommitted)
{
    std::unique_ptr<CudaVmmArena> a(make_arena());
    const size_t g = a->granularity();

    a->grow(g);
    EXPECT_EQ(a->committed_size(), g);

    a->grow(2 * g);
    EXPECT_EQ(a->committed_size(), 2 * g);

    a->grow(4 * g);
    EXPECT_EQ(a->committed_size(), 4 * g);
}

TEST_F(ArenaTest, GrowThrowsWhenNotLargerThanCommitted)
{
    std::unique_ptr<CudaVmmArena> a(make_arena());
    const size_t g = a->granularity();
    a->grow(2 * g);

    // Same size — must throw.
    EXPECT_THROW(a->grow(2 * g), CudaVmmError);
    // Smaller size — must throw.
    EXPECT_THROW(a->grow(g), CudaVmmError);
}

TEST_F(ArenaTest, GrowThrowsWhenExceedsMaxSize)
{
    const size_t g = query_granularity();
    if (g == 0)
        GTEST_SKIP();

    // Reserve exactly one granule.
    std::unique_ptr<CudaVmmArena> a(make_arena(g));

    // Trying to grow beyond the reserved VA must throw.
    EXPECT_THROW(a->grow(2 * g), CudaVmmError);
}

// ===========================================================================
// shrink()
// ===========================================================================

TEST_F(ArenaTest, ShrinkReleasesChunks)
{
    std::unique_ptr<CudaVmmArena> a(make_arena());
    const size_t g = a->granularity();
    a->grow(4 * g);

    a->shrink(2 * g);

    EXPECT_EQ(a->committed_size(), 2 * g);
}

TEST_F(ArenaTest, ShrinkToZeroReleasesAll)
{
    std::unique_ptr<CudaVmmArena> a(make_arena());
    const size_t g = a->granularity();
    a->grow(4 * g);

    a->shrink(0);

    EXPECT_EQ(a->committed_size(), 0u);
}

TEST_F(ArenaTest, ShrinkRoundsDownToGranularity)
{
    std::unique_ptr<CudaVmmArena> a(make_arena());
    const size_t g = a->granularity();
    a->grow(4 * g);

    // Request shrink to "2*g + 1" — rounds DOWN to 2*g.
    a->shrink(2 * g + 1);

    EXPECT_EQ(a->committed_size(), 2 * g);
}

TEST_F(ArenaTest, ShrinkThrowsWhenNotSmallerThanCommitted)
{
    std::unique_ptr<CudaVmmArena> a(make_arena());
    const size_t g = a->granularity();
    a->grow(2 * g);

    // Same size (after alignment) — must throw.
    EXPECT_THROW(a->shrink(2 * g), CudaVmmError);
    // Larger — must throw.
    EXPECT_THROW(a->shrink(4 * g), CudaVmmError);
}

TEST_F(ArenaTest, ShrinkThrowsOnUninitializedArena)
{
    std::unique_ptr<CudaVmmArena> a(make_arena());
    // committed_size == 0; any shrink target is >= committed_size.
    EXPECT_THROW(a->shrink(0), CudaVmmError);
}

// ===========================================================================
// resize()
// ===========================================================================

TEST_F(ArenaTest, ResizeGrowsWhenLarger)
{
    std::unique_ptr<CudaVmmArena> a(make_arena());
    const size_t g = a->granularity();

    a->resize(3 * g);

    EXPECT_EQ(a->committed_size(), 3 * g);
}

TEST_F(ArenaTest, ResizeShrinksWhenSmaller)
{
    std::unique_ptr<CudaVmmArena> a(make_arena());
    const size_t g = a->granularity();
    a->grow(4 * g);

    a->resize(2 * g);

    EXPECT_EQ(a->committed_size(), 2 * g);
}

TEST_F(ArenaTest, ResizeIsNoOpWhenAlreadyAtTarget)
{
    std::unique_ptr<CudaVmmArena> a(make_arena());
    const size_t g = a->granularity();
    a->grow(2 * g);

    // resize to the exact same aligned size — must not throw.
    EXPECT_NO_THROW(a->resize(2 * g));
    EXPECT_EQ(a->committed_size(), 2 * g);
}

// ===========================================================================
// grow → shrink → grow cycle
// ===========================================================================

TEST_F(ArenaTest, GrowShrinkGrowCycle)
{
    std::unique_ptr<CudaVmmArena> a(make_arena());
    const size_t g = a->granularity();

    a->grow(4 * g);
    EXPECT_EQ(a->committed_size(), 4 * g);

    a->shrink(2 * g);
    EXPECT_EQ(a->committed_size(), 2 * g);

    a->grow(6 * g);
    EXPECT_EQ(a->committed_size(), 6 * g);

    a->shrink(0);
    EXPECT_EQ(a->committed_size(), 0u);
}

// ===========================================================================
// ptr() stability
// ===========================================================================

TEST_F(ArenaTest, BasePointerDoesNotChangeAfterGrow)
{
    std::unique_ptr<CudaVmmArena> a(make_arena());
    const CUdeviceptr base = a->ptr();

    a->grow(a->granularity());
    EXPECT_EQ(a->ptr(), base) << "grow() must not change the base pointer";

    a->grow(2 * a->granularity());
    EXPECT_EQ(a->ptr(), base);
}

TEST_F(ArenaTest, BasePointerDoesNotChangeAfterShrink)
{
    std::unique_ptr<CudaVmmArena> a(make_arena());
    const size_t g = a->granularity();
    a->grow(4 * g);
    const CUdeviceptr base = a->ptr();

    a->shrink(2 * g);
    EXPECT_EQ(a->ptr(), base) << "shrink() must not change the base pointer";
}

// ===========================================================================
// Memory accessibility — write via kernel, read back to host
// ===========================================================================

/// Device kernel: write sequential uint32_t values starting at `offset_elems`.
__global__ void write_seq_kernel(uint32_t* data, uint32_t n, uint32_t offset_elems)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        data[i] = offset_elems + i;
}

/// Device kernel: verify sequential uint32_t values, store 1 on mismatch.
__global__ void verify_seq_kernel(uint32_t const* data, uint32_t n, uint32_t offset_elems, int* mismatch_flag)
{
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && data[i] != offset_elems + i)
        atomicExch(mismatch_flag, 1);
}

static void launch_write(CUdeviceptr base, size_t bytes, uint32_t offset_elems = 0)
{
    auto* p = reinterpret_cast<uint32_t*>(base);
    uint32_t n = static_cast<uint32_t>(bytes / sizeof(uint32_t));
    write_seq_kernel<<<(n + 255) / 256, 256>>>(p, n, offset_elems);
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
}

static bool device_verify(CUdeviceptr base, size_t bytes, uint32_t offset_elems = 0)
{
    auto* p = reinterpret_cast<uint32_t const*>(base);
    uint32_t n = static_cast<uint32_t>(bytes / sizeof(uint32_t));

    int* d_flag{};
    cudaMalloc(&d_flag, sizeof(int));
    cudaMemset(d_flag, 0, sizeof(int));

    verify_seq_kernel<<<(n + 255) / 256, 256>>>(p, n, offset_elems, d_flag);
    cudaDeviceSynchronize();

    int flag = 0;
    cudaMemcpy(&flag, d_flag, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_flag);
    return flag == 0;
}

TEST_F(ArenaTest, CommittedMemoryIsWriteable)
{
    std::unique_ptr<CudaVmmArena> a(make_arena());
    const size_t g = a->granularity();
    a->grow(g);

    ASSERT_NO_THROW(launch_write(a->ptr(), g));
    EXPECT_TRUE(device_verify(a->ptr(), g));
}

TEST_F(ArenaTest, DataInRetainedChunksSurvivesShrink)
{
    std::unique_ptr<CudaVmmArena> a(make_arena());
    const size_t g = a->granularity();
    a->grow(4 * g);

    // Fill the entire 4-chunk region.
    launch_write(a->ptr(), 4 * g);

    // Shrink to 2 chunks — the lower half must be intact.
    a->shrink(2 * g);

    EXPECT_TRUE(device_verify(a->ptr(), 2 * g)) << "Data in retained chunks should survive shrink()";
}

TEST_F(ArenaTest, NewChunksAfterRegrowAreWriteable)
{
    std::unique_ptr<CudaVmmArena> a(make_arena());
    const size_t g = a->granularity();

    a->grow(2 * g);
    launch_write(a->ptr(), 2 * g);

    a->shrink(g);

    // Regrow: the new chunk is fresh physical memory — just check it's writable.
    a->grow(2 * g);
    const size_t new_chunk_offset = g;
    const uint32_t off_elems = static_cast<uint32_t>(new_chunk_offset / sizeof(uint32_t));
    launch_write(a->ptr() + new_chunk_offset, g, off_elems);
    EXPECT_TRUE(device_verify(a->ptr() + new_chunk_offset, g, off_elems));
}

TEST_F(ArenaTest, MultipleSequentialGrowsAllAccessible)
{
    std::unique_ptr<CudaVmmArena> a(make_arena());
    const size_t g = a->granularity();

    // Grow in three steps.
    a->grow(g);
    a->grow(3 * g);
    a->grow(6 * g);

    // Fill and verify the whole 6-chunk region in one pass.
    launch_write(a->ptr(), 6 * g);
    EXPECT_TRUE(device_verify(a->ptr(), 6 * g));
}

// ===========================================================================
// Destructor safety
// ===========================================================================

TEST_F(ArenaTest, DestructorWithCommittedMemoryDoesNotLeak)
{
    // Simply constructing and immediately destroying a grown arena should not
    // crash, assert, or leak CUDA resources.
    {
        std::unique_ptr<CudaVmmArena> a(make_arena());
        const size_t g = a->granularity();
        a->grow(4 * g);
        // Destructor runs here.
    }
    SUCCEED(); // If we reach here, no crash occurred.
}

TEST_F(ArenaTest, DestructorWithZeroCommittedMemoryDoesNotCrash)
{
    {
        std::unique_ptr<CudaVmmArena> a(make_arena());
        // Never committed anything.
    }
    SUCCEED();
}

// ===========================================================================
// Multiple independent arenas
// ===========================================================================

TEST_F(ArenaTest, TwoArenasAreIndependent)
{
    std::unique_ptr<CudaVmmArena> a(make_arena(64ULL << 20));
    std::unique_ptr<CudaVmmArena> b(make_arena(64ULL << 20));

    const size_t g = a->granularity();
    a->grow(g);
    b->grow(g);

    // Base pointers must differ.
    EXPECT_NE(a->ptr(), b->ptr());

    // Writes to one must not alias the other.
    uint32_t n = static_cast<uint32_t>(g / sizeof(uint32_t));
    write_seq_kernel<<<(n + 255) / 256, 256>>>(reinterpret_cast<uint32_t*>(a->ptr()), n, /*offset=*/0);
    write_seq_kernel<<<(n + 255) / 256, 256>>>(reinterpret_cast<uint32_t*>(b->ptr()), n, /*offset=*/n);
    cudaDeviceSynchronize();

    EXPECT_TRUE(device_verify(a->ptr(), g, /*offset_elems=*/0));
    EXPECT_TRUE(device_verify(b->ptr(), g, /*offset_elems=*/n));
}

// ===========================================================================
// main
// ===========================================================================

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    ::testing::AddGlobalTestEnvironment(new CudaEnv());
    return RUN_ALL_TESTS();
}
