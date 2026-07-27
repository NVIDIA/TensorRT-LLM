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
#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_lora_slot_expand.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

namespace
{

using ::tensorrt_llm::kernels::cutlass_kernels::launchMoeLoraSlotExpand;
using ::tensorrt_llm::kernels::cutlass_kernels::MoeLoraSlotExpandModule;

// Host-side reference reproducing the slot -> per-source-token gather the
// kernel performs. Entries whose slot is outside [0, num_slots) are emitted as
// inactive (rank 0, null pointers), matching the kernel's valid_slot branch.
struct RefModule
{
    std::vector<int32_t> slot_ranks; // [num_slots]
    std::vector<int64_t> slot_ptrs;  // [num_slots * 3] (A, B, dora)
    std::vector<int32_t> ranks_out;  // [num_tokens]
    std::vector<int64_t> ptrs_out;   // [num_tokens * 2]
};

void cpuSlotExpand(std::vector<int32_t> const& token_to_slot, int64_t num_tokens, int64_t num_slots, RefModule& fc1,
    RefModule& fc2, RefModule* gated)
{
    auto expand_one = [&](RefModule& mod)
    {
        mod.ranks_out.assign(num_tokens, 0);
        mod.ptrs_out.assign(num_tokens * 2, 0);
        for (int64_t t = 0; t < num_tokens; ++t)
        {
            int32_t const slot = token_to_slot[t];
            bool const valid = (slot >= 0) && (static_cast<int64_t>(slot) < num_slots);
            if (valid)
            {
                mod.ranks_out[t] = mod.slot_ranks[slot];
                mod.ptrs_out[2 * t + 0] = mod.slot_ptrs[3 * slot + 0];
                mod.ptrs_out[2 * t + 1] = mod.slot_ptrs[3 * slot + 1];
            }
        }
    };
    expand_one(fc1);
    expand_one(fc2);
    if (gated)
    {
        expand_one(*gated);
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

template <typename T>
void deviceDownload(T* dev, std::vector<T>& host)
{
    if (host.empty())
    {
        return;
    }
    TLLM_CUDA_CHECK(cudaMemcpy(host.data(), dev, host.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

class MoeLoraSlotExpandTest : public ::testing::Test
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

    // Run the kernel against the CPU reference and assert the device outputs match.
    void runAndCompare(std::vector<int32_t> const& token_to_slot, int64_t num_tokens, int64_t num_slots,
        RefModule& fc1_ref, RefModule& fc2_ref, RefModule* gated_ref)
    {
        cpuSlotExpand(token_to_slot, num_tokens, num_slots, fc1_ref, fc2_ref, gated_ref);

        auto* token_to_slot_dev = upload(token_to_slot);

        auto build_module = [&](RefModule const& r)
        {
            MoeLoraSlotExpandModule m;
            m.slot_ranks = upload(r.slot_ranks);
            m.slot_ptrs = upload(r.slot_ptrs);
            m.ranks_out = allocZero<int32_t>(num_tokens);
            m.ptrs_out = allocZero<int64_t>(num_tokens * 2);
            return m;
        };

        MoeLoraSlotExpandModule fc1_dev = build_module(fc1_ref);
        MoeLoraSlotExpandModule fc2_dev = build_module(fc2_ref);
        MoeLoraSlotExpandModule gated_dev{};
        MoeLoraSlotExpandModule const* gated_dev_ptr = nullptr;
        if (gated_ref != nullptr)
        {
            gated_dev = build_module(*gated_ref);
            gated_dev_ptr = &gated_dev;
        }

        launchMoeLoraSlotExpand(token_to_slot_dev, num_tokens, num_slots, fc1_dev, fc2_dev, gated_dev_ptr, mStream);
        TLLM_CUDA_CHECK(cudaStreamSynchronize(mStream));

        auto compare = [&](RefModule const& ref_mod, MoeLoraSlotExpandModule const& dev_mod, char const* name)
        {
            std::vector<int32_t> host_ranks(num_tokens, 0);
            std::vector<int64_t> host_ptrs(num_tokens * 2, 0);
            deviceDownload(dev_mod.ranks_out, host_ranks);
            deviceDownload(dev_mod.ptrs_out, host_ptrs);
            for (int64_t t = 0; t < num_tokens; ++t)
            {
                EXPECT_EQ(host_ranks[t], ref_mod.ranks_out[t]) << name << " rank mismatch at t=" << t;
                EXPECT_EQ(host_ptrs[2 * t + 0], ref_mod.ptrs_out[2 * t + 0]) << name << " A ptr mismatch at t=" << t;
                EXPECT_EQ(host_ptrs[2 * t + 1], ref_mod.ptrs_out[2 * t + 1]) << name << " B ptr mismatch at t=" << t;
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

// Build a "fake but distinct" pointer for slot s of module tag. Encoding
// (tag, s, side) lets the test cheaply verify the kernel copies the right slot
// row of slot_ptrs into the per-token output.
int64_t fakePtr(int tag, int32_t s, int side)
{
    return (static_cast<int64_t>(tag) << 56) | (static_cast<int64_t>(side) << 48) | (static_cast<int64_t>(s + 1) << 32);
}

// Build a RefModule slot table: per-slot ranks plus (A, B, dora) pointer rows.
// dora is set to a sentinel that must NOT appear in the output (the kernel
// copies only A and B).
RefModule buildSlotTable(int tag, std::vector<int32_t> const& ranks)
{
    RefModule m{};
    m.slot_ranks = ranks;
    m.slot_ptrs.resize(ranks.size() * 3);
    for (int32_t s = 0; s < static_cast<int32_t>(ranks.size()); ++s)
    {
        m.slot_ptrs[3 * s + 0] = fakePtr(tag, s, /*side=*/0); // A
        m.slot_ptrs[3 * s + 1] = fakePtr(tag, s, /*side=*/1); // B
        m.slot_ptrs[3 * s + 2] = fakePtr(tag, s, /*side=*/2); // dora (ignored)
    }
    return m;
}

// Smallest non-trivial case: a handful of tokens mapping into 3 slots, no gated.
TEST_F(MoeLoraSlotExpandTest, BasicMultiSlotNoGated)
{
    int64_t const num_slots = 3;
    std::vector<int32_t> token_to_slot = {0, 1, 2, 0, 2, 1};
    int64_t const num_tokens = static_cast<int64_t>(token_to_slot.size());

    RefModule fc1 = buildSlotTable(/*tag=*/1, /*ranks=*/{2, 0, 4});
    RefModule fc2 = buildSlotTable(/*tag=*/2, /*ranks=*/{1, 3, 0});

    runAndCompare(token_to_slot, num_tokens, num_slots, fc1, fc2, /*gated=*/nullptr);
}

// Gated activation: three modules exercised in one pass.
TEST_F(MoeLoraSlotExpandTest, GatedActivation)
{
    int64_t const num_slots = 4;
    std::vector<int32_t> token_to_slot = {3, 0, 1, 2, 0, 3, 1};
    int64_t const num_tokens = static_cast<int64_t>(token_to_slot.size());

    RefModule fc1 = buildSlotTable(/*tag=*/1, {3, 0, 1, 4});
    RefModule fc2 = buildSlotTable(/*tag=*/2, {1, 2, 0, 3});
    RefModule gated = buildSlotTable(/*tag=*/3, {2, 4, 1, 0});

    runAndCompare(token_to_slot, num_tokens, num_slots, fc1, fc2, &gated);
}

// Out-of-range and negative slots must produce inactive (rank 0, null ptr)
// entries instead of indexing past the slot tables. This exercises the
// valid_slot == false branch that the higher-level Python tests never hit.
TEST_F(MoeLoraSlotExpandTest, OutOfRangeAndNegativeSlots)
{
    int64_t const num_slots = 2;
    // -1 (negative), 2 and 5 (>= num_slots) are all inactive; 0 and 1 are valid.
    std::vector<int32_t> token_to_slot = {0, -1, 1, 2, 5, 0};
    int64_t const num_tokens = static_cast<int64_t>(token_to_slot.size());

    RefModule fc1 = buildSlotTable(/*tag=*/1, {7, 9});
    RefModule fc2 = buildSlotTable(/*tag=*/2, {3, 5});
    RefModule gated = buildSlotTable(/*tag=*/3, {2, 6});

    runAndCompare(token_to_slot, num_tokens, num_slots, fc1, fc2, &gated);
}

// Empty call (num_tokens == 0) must be a no-op and must not launch a kernel
// with a zero grid.
TEST_F(MoeLoraSlotExpandTest, EmptyIsNoOp)
{
    int64_t const num_slots = 2;
    std::vector<int32_t> empty_token_to_slot;
    RefModule fc1 = buildSlotTable(/*tag=*/1, {1, 2});
    RefModule fc2 = buildSlotTable(/*tag=*/2, {3, 4});

    auto* slot_ranks_fc1 = upload(fc1.slot_ranks);
    auto* slot_ptrs_fc1 = upload(fc1.slot_ptrs);
    auto* slot_ranks_fc2 = upload(fc2.slot_ranks);
    auto* slot_ptrs_fc2 = upload(fc2.slot_ptrs);

    MoeLoraSlotExpandModule fc1_dev{slot_ranks_fc1, slot_ptrs_fc1, nullptr, nullptr};
    MoeLoraSlotExpandModule fc2_dev{slot_ranks_fc2, slot_ptrs_fc2, nullptr, nullptr};

    launchMoeLoraSlotExpand(/*token_to_slot=*/nullptr, /*num_tokens=*/0, num_slots, fc1_dev, fc2_dev,
        /*gated=*/nullptr, mStream);
    TLLM_CUDA_CHECK(cudaStreamSynchronize(mStream));
}

} // namespace
