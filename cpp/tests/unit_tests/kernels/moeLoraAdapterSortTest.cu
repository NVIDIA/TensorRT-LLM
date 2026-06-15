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
#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_lora_adapter_sort.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <set>
#include <vector>

namespace
{

using ::tensorrt_llm::kernels::cutlass_kernels::launchMoeLoraAdapterRegroup;

// Emulates the base expert sort: a stable group-by-expert over the
// (token, k) pairs. unpermuted_row = k * num_tokens + token. Produces the same
// forward/inverse maps and expert_first_token_offset the fused prologue would.
struct BaseSort
{
    std::vector<int> p2u;                       // [total]
    std::vector<int> u2p;                       // [total]
    std::vector<int64_t> expert_first_token_off; // [num_experts + 1]
};

BaseSort buildBaseSort(
    std::vector<int> const& token_selected_experts, int num_tokens, int experts_per_token, int num_experts)
{
    int const total = num_tokens * experts_per_token;
    BaseSort b;
    b.p2u.reserve(total);
    b.u2p.assign(total, -1);
    b.expert_first_token_off.assign(num_experts + 1, 0);

    for (int e = 0; e < num_experts; ++e)
    {
        b.expert_first_token_off[e] = static_cast<int64_t>(b.p2u.size());
        for (int token = 0; token < num_tokens; ++token)
        {
            for (int k = 0; k < experts_per_token; ++k)
            {
                if (token_selected_experts[token * experts_per_token + k] == e)
                {
                    int const unpermuted = k * num_tokens + token;
                    int const permuted = static_cast<int>(b.p2u.size());
                    b.p2u.push_back(unpermuted);
                    b.u2p[unpermuted] = permuted;
                }
            }
        }
    }
    b.expert_first_token_off[num_experts] = static_cast<int64_t>(b.p2u.size());
    return b;
}

template <typename T>
T* upload(std::vector<T> const& h, std::vector<void*>& allocs)
{
    if (h.empty())
    {
        return nullptr;
    }
    T* dev = nullptr;
    TLLM_CUDA_CHECK(cudaMalloc(&dev, h.size() * sizeof(T)));
    TLLM_CUDA_CHECK(cudaMemcpy(dev, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    allocs.push_back(dev);
    return dev;
}

class MoeLoraAdapterSortTest : public ::testing::Test
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

    // Runs the regroup and validates: (1) expert_first_token_offset unchanged,
    // (2) within each expert block rows are contiguous by adapter slot, (3) the
    // maps form a consistent permutation preserving each block's row set.
    void runAndValidate(std::vector<int> const& token_selected_experts, std::vector<int> const& token_to_slot,
        int num_tokens, int experts_per_token, int num_experts, int num_slots)
    {
        int const total = num_tokens * experts_per_token;
        BaseSort base = buildBaseSort(token_selected_experts, num_tokens, experts_per_token, num_experts);

        int* p2u_dev = upload(base.p2u, mAllocations);
        int* u2p_dev = upload(base.u2p, mAllocations);
        int64_t* efto_dev = upload(base.expert_first_token_off, mAllocations);
        int32_t* slot_dev = upload(token_to_slot, mAllocations);

        int* scratch_dev = nullptr;
        TLLM_CUDA_CHECK(cudaMalloc(&scratch_dev, total * sizeof(int)));
        mAllocations.push_back(scratch_dev);

        bool const ran = launchMoeLoraAdapterRegroup(p2u_dev, u2p_dev, scratch_dev, efto_dev, slot_dev, num_experts,
            num_tokens, total, num_slots, mStream);
        TLLM_CUDA_CHECK(cudaStreamSynchronize(mStream));
        ASSERT_TRUE(ran) << "regroup unexpectedly skipped";

        std::vector<int> p2u(total), u2p(total);
        std::vector<int64_t> efto(num_experts + 1);
        TLLM_CUDA_CHECK(cudaMemcpy(p2u.data(), p2u_dev, total * sizeof(int), cudaMemcpyDeviceToHost));
        TLLM_CUDA_CHECK(cudaMemcpy(u2p.data(), u2p_dev, total * sizeof(int), cudaMemcpyDeviceToHost));
        TLLM_CUDA_CHECK(
            cudaMemcpy(efto.data(), efto_dev, (num_experts + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost));

        // (1) expert_first_token_offset is read-only: must be byte-identical.
        for (int e = 0; e <= num_experts; ++e)
        {
            EXPECT_EQ(efto[e], base.expert_first_token_off[e]) << "expert_first_token_offset changed at e=" << e;
        }

        auto slot_of = [&](int unpermuted)
        {
            int const tok = unpermuted % num_tokens;
            int const s = token_to_slot[tok];
            return (s >= 0 && s < num_slots) ? s : num_slots; // fold invalid -> catch-all bin
        };

        for (int e = 0; e < num_experts; ++e)
        {
            int const s = static_cast<int>(base.expert_first_token_off[e]);
            int const en = static_cast<int>(base.expert_first_token_off[e + 1]);

            // (2) Contiguity by slot: once we leave a slot's run we must never
            // see it again within the block.
            std::set<int> seen_slots;
            int prev_slot = -1;
            for (int p = s; p < en; ++p)
            {
                int const cur = slot_of(p2u[p]);
                if (cur != prev_slot)
                {
                    EXPECT_EQ(seen_slots.count(cur), 0u)
                        << "slot " << cur << " is not contiguous within expert " << e;
                    seen_slots.insert(cur);
                    prev_slot = cur;
                }
            }

            // (3a) Row set per block preserved (same unpermuted rows, reordered).
            std::set<int> before(base.p2u.begin() + s, base.p2u.begin() + en);
            std::set<int> after(p2u.begin() + s, p2u.begin() + en);
            EXPECT_EQ(before, after) << "expert " << e << " row set changed";
        }

        // (3b) Maps are consistent inverses over the live region.
        for (int p = 0; p < total; ++p)
        {
            int const u = p2u[p];
            ASSERT_GE(u, 0);
            ASSERT_LT(u, total);
            EXPECT_EQ(u2p[u], p) << "inverse map inconsistent at permuted row " << p;
        }
    }

    cudaStream_t mStream{};
    std::vector<void*> mAllocations;
};

// Two experts, three adapter slots, adapters interleaved within each expert.
// After regrouping, each expert block must be contiguous by slot.
TEST_F(MoeLoraAdapterSortTest, InterleavedAdaptersGrouped)
{
    int const num_tokens = 8;
    int const experts_per_token = 1;
    int const num_experts = 2;
    int const num_slots = 3;

    // token -> expert (top_k=1), chosen so both experts get an interleaved mix.
    std::vector<int> experts = {0, 1, 0, 1, 0, 1, 0, 1};
    // token -> adapter slot, deliberately interleaved (0,1,2,0,1,2,...).
    std::vector<int> slots = {0, 1, 2, 0, 1, 2, 0, 1};

    runAndValidate(experts, slots, num_tokens, experts_per_token, num_experts, num_slots);
}

// top_k = 2 so each token contributes two unpermuted rows; mixed slots including
// a no-adapter token (slot = -1) which must fold into the catch-all bin.
TEST_F(MoeLoraAdapterSortTest, TopK2WithNoAdapterToken)
{
    int const num_tokens = 6;
    int const experts_per_token = 2;
    int const num_experts = 3;
    int const num_slots = 2;

    // token*top_k + k -> expert.
    std::vector<int> experts = {
        0, 1, // token 0
        1, 2, // token 1
        0, 2, // token 2
        0, 1, // token 3
        1, 2, // token 4
        0, 2, // token 5
    };
    std::vector<int> slots = {0, 1, -1, 0, 1, -1}; // tokens 2 and 5 have no adapter

    runAndValidate(experts, slots, num_tokens, experts_per_token, num_experts, num_slots);
}

// All tokens share one adapter: regroup is a no-op grouping (single run per
// expert), maps must stay a valid permutation.
TEST_F(MoeLoraAdapterSortTest, SingleAdapterAllTokens)
{
    int const num_tokens = 8;
    int const experts_per_token = 1;
    int const num_experts = 2;
    int const num_slots = 1;

    std::vector<int> experts = {0, 0, 0, 0, 1, 1, 1, 1};
    std::vector<int> slots(num_tokens, 0);

    runAndValidate(experts, slots, num_tokens, experts_per_token, num_experts, num_slots);
}

// num_slots too large to histogram: the launch must report it skipped (returns
// false) rather than corrupting the maps.
TEST_F(MoeLoraAdapterSortTest, TooManySlotsSkips)
{
    int const num_tokens = 4;
    int const experts_per_token = 1;
    int const num_experts = 1;
    int const total = num_tokens * experts_per_token;
    int const num_slots = (48 * 1024 / static_cast<int>(sizeof(int))); // exceeds the 3*(slots+1) budget

    std::vector<int> experts(num_tokens, 0);
    std::vector<int> slots(num_tokens, 0);
    BaseSort base = buildBaseSort(experts, num_tokens, experts_per_token, num_experts);

    int* p2u_dev = upload(base.p2u, mAllocations);
    int* u2p_dev = upload(base.u2p, mAllocations);
    int64_t* efto_dev = upload(base.expert_first_token_off, mAllocations);
    int32_t* slot_dev = upload(slots, mAllocations);
    int* scratch_dev = nullptr;
    TLLM_CUDA_CHECK(cudaMalloc(&scratch_dev, total * sizeof(int)));
    mAllocations.push_back(scratch_dev);

    bool const ran = launchMoeLoraAdapterRegroup(
        p2u_dev, u2p_dev, scratch_dev, efto_dev, slot_dev, num_experts, num_tokens, total, num_slots, mStream);
    TLLM_CUDA_CHECK(cudaStreamSynchronize(mStream));
    EXPECT_FALSE(ran) << "regroup should skip when num_slots exceeds the shared-memory budget";
}

} // namespace
