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

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_lora_pointer_expand.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

namespace
{

using ::tensorrt_llm::kernels::cutlass_kernels::launchMoeLoraPointerExpand;
using ::tensorrt_llm::kernels::cutlass_kernels::MoeLoraExpandModule;

// Host-side reference reproducing the per-permuted-row pointer arithmetic
// from CutlassMoeFCRunner::setupLoraWorkspace. Same control flow as the
// device kernel; used only as ground truth for parity checks.
struct RefModule
{
    std::vector<int32_t> ranks_src;
    std::vector<int64_t> ptrs_src;
    int64_t dim_a;
    int64_t dim_b;
    std::vector<int32_t> ranks_out;
    std::vector<int64_t> ptrs_out;
};

void cpuExpand(std::vector<int32_t> const& permuted_rows, std::vector<int64_t> const& expert_first_token_offset,
    int32_t num_experts_per_node, int32_t start_expert, int64_t num_rows, int64_t expanded_num_rows,
    int64_t lora_dtype_bytes, RefModule& fc1, RefModule& fc2, RefModule* gated)
{
    auto expand_one = [&](RefModule& mod, int64_t i, int32_t source_index, int64_t weight_index)
    {
        int32_t const rank = mod.ranks_src[source_index];
        int64_t const a_stride = weight_index * mod.dim_a * rank * lora_dtype_bytes;
        int64_t const b_stride = weight_index * mod.dim_b * rank * lora_dtype_bytes;
        mod.ptrs_out[2 * i + 0] = mod.ptrs_src[2 * source_index + 0] + a_stride;
        mod.ptrs_out[2 * i + 1] = mod.ptrs_src[2 * source_index + 1] + b_stride;
        mod.ranks_out[i] = rank;
    };

    fc1.ranks_out.assign(expanded_num_rows, 0);
    fc1.ptrs_out.assign(expanded_num_rows * 2, 0);
    fc2.ranks_out.assign(expanded_num_rows, 0);
    fc2.ptrs_out.assign(expanded_num_rows * 2, 0);
    if (gated)
    {
        gated->ranks_out.assign(expanded_num_rows, 0);
        gated->ptrs_out.assign(expanded_num_rows * 2, 0);
    }

    for (int32_t expert_idx = 0; expert_idx < num_experts_per_node; ++expert_idx)
    {
        int64_t const weight_index = static_cast<int64_t>(expert_idx) + start_expert;
        for (int64_t i = expert_first_token_offset[expert_idx]; i < expert_first_token_offset[expert_idx + 1]; ++i)
        {
            int32_t const source_index = static_cast<int32_t>(permuted_rows[i] % num_rows);
            expand_one(fc1, i, source_index, weight_index);
            expand_one(fc2, i, source_index, weight_index);
            if (gated)
            {
                expand_one(*gated, i, source_index, weight_index);
            }
        }
    }
}

template <typename T>
T* deviceUpload(std::vector<T> const& host)
{
    T* dev = nullptr;
    auto const bytes = host.size() * sizeof(T);
    if (bytes > 0)
    {
        TLLM_CUDA_CHECK(cudaMalloc(&dev, bytes));
        TLLM_CUDA_CHECK(cudaMemcpy(dev, host.data(), bytes, cudaMemcpyHostToDevice));
    }
    return dev;
}

template <typename T>
T* deviceAllocZero(size_t count)
{
    T* dev = nullptr;
    auto const bytes = count * sizeof(T);
    TLLM_CUDA_CHECK(cudaMalloc(&dev, bytes));
    TLLM_CUDA_CHECK(cudaMemset(dev, 0, bytes));
    return dev;
}

// Like deviceAllocZero but pre-fills with a non-zero byte pattern. Simulates
// reused scratch holding stale values, so tests can verify the kernel actively
// zeroes ghost rows.
template <typename T>
T* deviceAllocFilled(size_t count, int byte_pattern)
{
    T* dev = nullptr;
    auto const bytes = count * sizeof(T);
    TLLM_CUDA_CHECK(cudaMalloc(&dev, bytes));
    TLLM_CUDA_CHECK(cudaMemset(dev, byte_pattern, bytes));
    return dev;
}

template <typename T>
void deviceDownload(T* dev, std::vector<T>& host)
{
    if (host.empty())
    {
        return;
    }
    TLLM_CUDA_CHECK(cudaMemcpy(host.data(), dev, host.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

class MoeLoraPointerExpandTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        TLLM_CUDA_CHECK(cudaStreamCreate(&mStream));
    }

    void TearDown() override
    {
        for (auto* p : mAllocations)
        {
            (void) cudaFree(p);
        }
        (void) cudaStreamDestroy(mStream);
    }

    template <typename T>
    T* upload(std::vector<T> const& host)
    {
        T* p = deviceUpload(host);
        if (p != nullptr)
        {
            mAllocations.push_back(p);
        }
        return p;
    }

    template <typename T>
    T* allocZero(size_t count)
    {
        T* p = deviceAllocZero<T>(count);
        mAllocations.push_back(p);
        return p;
    }

    template <typename T>
    T* allocFilled(size_t count, int byte_pattern)
    {
        T* p = deviceAllocFilled<T>(count, byte_pattern);
        mAllocations.push_back(p);
        return p;
    }

    // Run the kernel against ref and assert the device outputs match. When
    // prefill_garbage is true the output buffers start with a non-zero pattern,
    // forcing the kernel to explicitly zero ghost rows for the comparison to
    // pass.
    void runAndCompare(std::vector<int32_t> const& permuted_rows, std::vector<int64_t> const& expert_first_token_offset,
        int32_t num_experts_per_node, int32_t start_expert, int64_t num_rows, int64_t expanded_num_rows,
        int64_t lora_dtype_bytes, RefModule& fc1_ref, RefModule& fc2_ref, RefModule* gated_ref,
        bool prefill_garbage = false)
    {
        cpuExpand(permuted_rows, expert_first_token_offset, num_experts_per_node, start_expert, num_rows,
            expanded_num_rows, lora_dtype_bytes, fc1_ref, fc2_ref, gated_ref);

        auto* permuted_rows_dev = upload(permuted_rows);
        auto* offsets_dev = upload(expert_first_token_offset);

        auto build_module = [&](RefModule const& r)
        {
            MoeLoraExpandModule m;
            m.ranks_src = upload(r.ranks_src);
            m.ptrs_src = upload(r.ptrs_src);
            m.dim_a = r.dim_a;
            m.dim_b = r.dim_b;
            m.ranks_out = prefill_garbage ? allocFilled<int32_t>(expanded_num_rows, 0x7F)
                                          : allocZero<int32_t>(expanded_num_rows);
            m.ptrs_out = prefill_garbage ? allocFilled<int64_t>(expanded_num_rows * 2, 0x7F)
                                         : allocZero<int64_t>(expanded_num_rows * 2);
            return m;
        };

        MoeLoraExpandModule fc1_dev = build_module(fc1_ref);
        MoeLoraExpandModule fc2_dev = build_module(fc2_ref);
        MoeLoraExpandModule gated_dev{};
        MoeLoraExpandModule const* gated_dev_ptr = nullptr;
        if (gated_ref != nullptr)
        {
            gated_dev = build_module(*gated_ref);
            gated_dev_ptr = &gated_dev;
        }

        launchMoeLoraPointerExpand(permuted_rows_dev, offsets_dev, num_experts_per_node, start_expert, num_rows,
            expanded_num_rows, lora_dtype_bytes, fc1_dev, fc2_dev, gated_dev_ptr, mStream);
        TLLM_CUDA_CHECK(cudaStreamSynchronize(mStream));

        // Compare per-module.
        auto compare = [&](RefModule const& ref_mod, MoeLoraExpandModule const& dev_mod, char const* name)
        {
            std::vector<int32_t> host_ranks(expanded_num_rows, 0);
            std::vector<int64_t> host_ptrs(expanded_num_rows * 2, 0);
            deviceDownload(dev_mod.ranks_out, host_ranks);
            deviceDownload(dev_mod.ptrs_out, host_ptrs);
            for (int64_t i = 0; i < expanded_num_rows; ++i)
            {
                EXPECT_EQ(host_ranks[i], ref_mod.ranks_out[i]) << name << " rank mismatch at i=" << i;
                EXPECT_EQ(host_ptrs[2 * i + 0], ref_mod.ptrs_out[2 * i + 0]) << name << " A ptr mismatch at i=" << i;
                EXPECT_EQ(host_ptrs[2 * i + 1], ref_mod.ptrs_out[2 * i + 1]) << name << " B ptr mismatch at i=" << i;
            }
        };

        compare(fc1_ref, fc1_dev, "fc1");
        compare(fc2_ref, fc2_dev, "fc2");
        if (gated_ref != nullptr)
        {
            compare(*gated_ref, gated_dev, "gated");
        }
    }

    cudaStream_t mStream{};
    std::vector<void*> mAllocations;
};

// Helper: build a "fake but distinct" pointer for source token s of module
// tag. Encoding the (tag, s, side) lets the test cheaply verify the
// kernel reads the right slot of ptrs_src. The high bits guarantee
// (ptr + per-expert-byte-offset) doesn't alias another (tag, s, side).
int64_t fakePtr(int tag, int32_t s, int side)
{
    return (static_cast<int64_t>(tag) << 56) | (static_cast<int64_t>(side) << 48) | (static_cast<int64_t>(s + 1) << 32);
}

// Smallest non-trivial case: 4 source tokens, 3 experts, top_k=2 so the
// permuted batch has 8 rows. Per-expert, no gated.
TEST_F(MoeLoraPointerExpandTest, PerExpertNoGated)
{
    int32_t const num_experts_per_node = 3;
    int32_t const start_expert = 0;
    int64_t const num_rows = 4;
    int64_t const expanded_num_rows = 8; // top_k=2

    // (expert_id assignment is irrelevant to the kernel; we only need
    // expert_first_token_offset for the lookup and permuted_rows for the
    // source-index reverse.)
    std::vector<int32_t> permuted_rows = {0, 4, 1, 5, 2, 6, 3, 7};
    std::vector<int64_t> expert_first_token_offset = {0, 3, 5, 8};

    RefModule fc1{};
    fc1.dim_a = 16; // "hidden_size"
    fc1.dim_b = 32; // "inter_size"
    fc1.ranks_src = {2, 0, 4, 1};
    fc1.ptrs_src.resize(num_rows * 2);
    for (int32_t s = 0; s < num_rows; ++s)
    {
        fc1.ptrs_src[2 * s + 0] = fakePtr(/*tag=*/1, s, /*side=*/0);
        fc1.ptrs_src[2 * s + 1] = fakePtr(/*tag=*/1, s, /*side=*/1);
    }

    RefModule fc2{};
    fc2.dim_a = 32;
    fc2.dim_b = 16;
    fc2.ranks_src = {1, 2, 0, 3};
    fc2.ptrs_src.resize(num_rows * 2);
    for (int32_t s = 0; s < num_rows; ++s)
    {
        fc2.ptrs_src[2 * s + 0] = fakePtr(/*tag=*/2, s, /*side=*/0);
        fc2.ptrs_src[2 * s + 1] = fakePtr(/*tag=*/2, s, /*side=*/1);
    }

    runAndCompare(permuted_rows, expert_first_token_offset, num_experts_per_node, start_expert, num_rows,
        expanded_num_rows, /*lora_dtype_bytes=*/2, fc1, fc2, /*gated=*/nullptr);
}

// Gated activation: three modules, exercises the gated arg path.
TEST_F(MoeLoraPointerExpandTest, GatedActivation)
{
    int32_t const num_experts_per_node = 4;
    int32_t const start_expert = 2; // exercises start_expert != 0
    int64_t const num_rows = 5;
    int64_t const expanded_num_rows = 10;

    std::vector<int32_t> permuted_rows = {0, 5, 1, 6, 2, 7, 3, 8, 4, 9};
    std::vector<int64_t> expert_first_token_offset = {0, 2, 5, 7, 10};

    auto build_basic = [&](int tag, int64_t dim_a, int64_t dim_b)
    {
        RefModule m{};
        m.dim_a = dim_a;
        m.dim_b = dim_b;
        m.ranks_src = {3, 0, 1, 4, 2};
        m.ptrs_src.resize(num_rows * 2);
        for (int32_t s = 0; s < num_rows; ++s)
        {
            m.ptrs_src[2 * s + 0] = fakePtr(tag, s, 0);
            m.ptrs_src[2 * s + 1] = fakePtr(tag, s, 1);
        }
        return m;
    };

    RefModule fc1 = build_basic(/*tag=*/1, /*hidden=*/8, /*inter=*/24);
    RefModule fc2 = build_basic(/*tag=*/2, /*inter=*/24, /*hidden=*/8);
    RefModule gated = build_basic(/*tag=*/3, /*hidden=*/8, /*inter=*/24);
    runAndCompare(permuted_rows, expert_first_token_offset, num_experts_per_node, start_expert, num_rows,
        expanded_num_rows, /*lora_dtype_bytes=*/2, fc1, fc2, &gated);
}

// Non-trivial lora_dtype_bytes (e.g. fp32 = 4) to verify the stride scaling
// flows through the offset arithmetic.
TEST_F(MoeLoraPointerExpandTest, Fp32StrideBytes)
{
    int32_t const num_experts_per_node = 2;
    int32_t const start_expert = 0;
    int64_t const num_rows = 2;
    int64_t const expanded_num_rows = 4;

    std::vector<int32_t> permuted_rows = {0, 1, 0, 1};
    std::vector<int64_t> expert_first_token_offset = {0, 2, 4};

    RefModule fc1{};
    fc1.dim_a = 4;
    fc1.dim_b = 8;
    fc1.ranks_src = {2, 3};
    fc1.ptrs_src = {fakePtr(1, 0, 0), fakePtr(1, 0, 1), fakePtr(1, 1, 0), fakePtr(1, 1, 1)};

    RefModule fc2 = fc1;
    fc2.dim_a = 8;
    fc2.dim_b = 4;

    runAndCompare(permuted_rows, expert_first_token_offset, num_experts_per_node, start_expert, num_rows,
        expanded_num_rows, /*lora_dtype_bytes=*/4, fc1, fc2, /*gated=*/nullptr);
}

// Ghost rows: expanded_num_rows exceeds the last valid expert offset, so the
// trailing rows have expert_idx == num_experts_per_node and must be zeroed by
// the kernel. Buffers are pre-filled with garbage to verify the kernel actively
// resets them.
TEST_F(MoeLoraPointerExpandTest, GhostRowsRemainZero)
{
    int32_t const num_experts_per_node = 3;
    int32_t const start_expert = 0;
    int64_t const num_rows = 4;
    // expert_first_token_offset.back() == 6, but we run two extra ghost rows.
    int64_t const expanded_num_rows = 8;

    std::vector<int32_t> permuted_rows = {0, 4, 1, 5, 2, 6, 0, 0};
    std::vector<int64_t> expert_first_token_offset = {0, 2, 4, 6};

    RefModule fc1{};
    fc1.dim_a = 16;
    fc1.dim_b = 32;
    fc1.ranks_src = {2, 0, 4, 1};
    fc1.ptrs_src.resize(num_rows * 2);
    for (int32_t s = 0; s < num_rows; ++s)
    {
        fc1.ptrs_src[2 * s + 0] = fakePtr(/*tag=*/1, s, /*side=*/0);
        fc1.ptrs_src[2 * s + 1] = fakePtr(/*tag=*/1, s, /*side=*/1);
    }

    RefModule fc2 = fc1;
    fc2.dim_a = 32;
    fc2.dim_b = 16;

    runAndCompare(permuted_rows, expert_first_token_offset, num_experts_per_node, start_expert, num_rows,
        expanded_num_rows, /*lora_dtype_bytes=*/2, fc1, fc2, /*gated=*/nullptr, /*prefill_garbage=*/true);
}

// num_experts_per_node above kMaxExpertsInSmem (1024) forces the kernel to take
// the global-memory expert-offset scan instead of the shared-memory path.
TEST_F(MoeLoraPointerExpandTest, ForceGlobalScanNumExperts1025)
{
    int32_t const num_experts_per_node = 1025;
    int32_t const start_expert = 0;
    int64_t const num_rows = 4;
    int64_t const expanded_num_rows = 4; // top_k=1

    std::vector<int32_t> permuted_rows = {0, 1, 2, 3};
    // All tokens land in expert 0; every other expert is empty. Offset array has
    // num_experts_per_node + 1 == 1026 entries.
    std::vector<int64_t> expert_first_token_offset(num_experts_per_node + 1, expanded_num_rows);
    expert_first_token_offset[0] = 0;

    RefModule fc1{};
    fc1.dim_a = 8;
    fc1.dim_b = 16;
    fc1.ranks_src = {1, 2, 3, 4};
    fc1.ptrs_src.resize(num_rows * 2);
    for (int32_t s = 0; s < num_rows; ++s)
    {
        fc1.ptrs_src[2 * s + 0] = fakePtr(/*tag=*/1, s, /*side=*/0);
        fc1.ptrs_src[2 * s + 1] = fakePtr(/*tag=*/1, s, /*side=*/1);
    }

    RefModule fc2 = fc1;
    fc2.dim_a = 16;
    fc2.dim_b = 8;

    runAndCompare(permuted_rows, expert_first_token_offset, num_experts_per_node, start_expert, num_rows,
        expanded_num_rows, /*lora_dtype_bytes=*/2, fc1, fc2, /*gated=*/nullptr);
}

} // namespace
