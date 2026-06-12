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
#include "tensorrt_llm/kernels/cutlass_kernels/include/moe_lora_problem_builder.h"

#include "cutlass/gemm_coord.h"

#include <cstdint>
#include <gtest/gtest.h>
#include <vector>

namespace
{

using ::tensorrt_llm::kernels::cutlass_kernels::launchMoeLoraProblemBuilder;
using ::tensorrt_llm::kernels::cutlass_kernels::MoeLoraGemmGroupArrays;

// Host-side reference reproducing the builder's run-length-aggregating logic.
// Consecutive rows sharing the same (rank, A_ptr, B_ptr) are merged into one
// M=run_length problem in slots [0, num_groups); the tail [num_groups, P) is
// filled with N=0 no-ops. Same formulas as the kernel; parity ground truth.
struct RefOutputs
{
    std::vector<cutlass::gemm::GemmCoord> problem_sizes_in;
    std::vector<cutlass::gemm::GemmCoord> problem_sizes_out;
    std::vector<int64_t> a_ptrs_in; // store as int64 bits for simple compare
    std::vector<int64_t> b_ptrs_in;
    std::vector<int64_t> d_ptrs_in;
    std::vector<int64_t> b_ptrs_out;
    std::vector<int64_t> d_ptrs_out;
    std::vector<int64_t> lda_in;
    std::vector<int64_t> ldb_in;
    std::vector<int64_t> ldd_in;
    std::vector<int64_t> ldb_out;
    std::vector<int64_t> ldd_out;
    std::vector<int64_t> splitk_offsets;
};

RefOutputs cpuReference(std::vector<int32_t> const& ranks, std::vector<int64_t> const& ptrs, int64_t input_base,
    int64_t lowrank_workspace, int64_t output_base, int64_t in_hidden_size, int64_t out_hidden_size,
    int64_t max_lora_rank, int64_t dtype_bytes, int64_t splitk_slices)
{
    int64_t const P = static_cast<int64_t>(ranks.size());
    int64_t const in_row_stride = in_hidden_size * dtype_bytes;
    int64_t const work_row_stride = max_lora_rank * dtype_bytes;
    int64_t const out_row_stride = out_hidden_size * dtype_bytes;

    RefOutputs r;
    r.problem_sizes_in.resize(P);
    r.problem_sizes_out.resize(P);
    r.a_ptrs_in.resize(P);
    r.b_ptrs_in.resize(P);
    r.d_ptrs_in.resize(P);
    r.b_ptrs_out.resize(P);
    r.d_ptrs_out.resize(P);
    r.lda_in.resize(P);
    r.ldb_in.resize(P);
    r.ldd_in.resize(P);
    r.ldb_out.resize(P);
    r.ldd_out.resize(P);
    r.splitk_offsets.resize(P + 1);

    // Phase 1 equivalent: initialize every slot as an N=0 no-op.
    for (int64_t p = 0; p < P; ++p)
    {
        r.problem_sizes_in[p] = cutlass::gemm::GemmCoord(1, 0, static_cast<int>(in_hidden_size));
        r.problem_sizes_out[p] = cutlass::gemm::GemmCoord(1, 0, 0);
        r.a_ptrs_in[p] = input_base;
        r.b_ptrs_in[p] = 0;
        r.d_ptrs_in[p] = lowrank_workspace;
        r.b_ptrs_out[p] = 0;
        r.d_ptrs_out[p] = output_base;
        r.lda_in[p] = in_hidden_size;
        r.ldb_in[p] = in_hidden_size;
        r.ldd_in[p] = max_lora_rank;
        r.ldb_out[p] = 0;
        r.ldd_out[p] = out_hidden_size;
    }

    // Phase 2 equivalent: merge maximal same-(rank, A_ptr, B_ptr) runs.
    int64_t group = 0;
    int64_t splitk_acc = 0;
    int64_t i = 0;
    while (i < P)
    {
        int32_t const rank = ranks[i];
        int64_t const a_ptr_bits = ptrs[2 * i + 0];
        int64_t const b_ptr_bits = ptrs[2 * i + 1];
        int64_t j = i + 1;
        while (j < P && ranks[j] == rank && ptrs[2 * j + 0] == a_ptr_bits && ptrs[2 * j + 1] == b_ptr_bits)
        {
            ++j;
        }
        int const M = static_cast<int>(j - i);
        int const out_n = (rank > 0) ? static_cast<int>(out_hidden_size) : 0;

        r.problem_sizes_in[group] = cutlass::gemm::GemmCoord(M, rank, static_cast<int>(in_hidden_size));
        r.problem_sizes_out[group] = cutlass::gemm::GemmCoord(M, out_n, rank);
        r.a_ptrs_in[group] = input_base + i * in_row_stride;
        r.b_ptrs_in[group] = a_ptr_bits;
        r.d_ptrs_in[group] = lowrank_workspace + i * work_row_stride;
        r.b_ptrs_out[group] = b_ptr_bits;
        r.d_ptrs_out[group] = output_base + i * out_row_stride;
        r.lda_in[group] = in_hidden_size;
        r.ldb_in[group] = in_hidden_size;
        r.ldd_in[group] = max_lora_rank;
        r.ldb_out[group] = rank;
        r.ldd_out[group] = out_hidden_size;
        r.splitk_offsets[group] = splitk_acc;
        splitk_acc += static_cast<int64_t>(M) * max_lora_rank * splitk_slices;

        ++group;
        i = j;
    }
    for (int64_t p = group; p < P; ++p)
    {
        r.splitk_offsets[p] = splitk_acc;
    }
    r.splitk_offsets[P] = splitk_acc;
    return r;
}

template <typename T>
T* deviceUpload(std::vector<T> const& host)
{
    if (host.empty())
    {
        return nullptr;
    }
    T* dev = nullptr;
    TLLM_CUDA_CHECK(cudaMalloc(&dev, host.size() * sizeof(T)));
    TLLM_CUDA_CHECK(cudaMemcpy(dev, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    return dev;
}

template <typename T>
T* deviceAllocZero(size_t count)
{
    T* dev = nullptr;
    TLLM_CUDA_CHECK(cudaMalloc(&dev, count * sizeof(T)));
    TLLM_CUDA_CHECK(cudaMemset(dev, 0, count * sizeof(T)));
    return dev;
}

template <typename T>
void deviceDownload(T const* dev, std::vector<T>& host)
{
    if (host.empty())
    {
        return;
    }
    TLLM_CUDA_CHECK(cudaMemcpy(host.data(), dev, host.size() * sizeof(T), cudaMemcpyDeviceToHost));
}

class MoeLoraProblemBuilderTest : public ::testing::Test
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
    T* upload(std::vector<T> const& h)
    {
        T* p = deviceUpload(h);
        if (p)
        {
            mAllocations.push_back(p);
        }
        return p;
    }

    template <typename T>
    T* allocZero(size_t n)
    {
        T* p = deviceAllocZero<T>(n);
        mAllocations.push_back(p);
        return p;
    }

    // When with_splitk is false, out.splitk_offsets is left null to exercise the
    // kernel's null-offset branch (and the launch_count path that drops the +1
    // sentinel thread); the splitk_offsets comparison is then skipped.
    void runAndCompare(std::vector<int32_t> const& ranks, std::vector<int64_t> const& ptrs, int64_t input_base,
        int64_t lowrank_workspace, int64_t output_base, int64_t in_hidden_size, int64_t out_hidden_size,
        int64_t max_lora_rank, int64_t dtype_bytes, int64_t splitk_slices, bool with_splitk = true)
    {
        auto const P = static_cast<int64_t>(ranks.size());
        RefOutputs ref = cpuReference(ranks, ptrs, input_base, lowrank_workspace, output_base, in_hidden_size,
            out_hidden_size, max_lora_rank, dtype_bytes, splitk_slices);

        int32_t* ranks_dev = upload(ranks);
        int64_t* ptrs_dev = upload(ptrs);

        MoeLoraGemmGroupArrays out;
        out.problem_sizes_in
            = reinterpret_cast<cutlass::gemm::GemmCoord*>(allocZero<int8_t>(P * sizeof(cutlass::gemm::GemmCoord)));
        out.problem_sizes_out
            = reinterpret_cast<cutlass::gemm::GemmCoord*>(allocZero<int8_t>(P * sizeof(cutlass::gemm::GemmCoord)));
        out.a_ptrs_in = reinterpret_cast<void**>(allocZero<int64_t>(P));
        out.b_ptrs_in = reinterpret_cast<void**>(allocZero<int64_t>(P));
        out.d_ptrs_in = reinterpret_cast<void**>(allocZero<int64_t>(P));
        out.b_ptrs_out = reinterpret_cast<void**>(allocZero<int64_t>(P));
        out.d_ptrs_out = reinterpret_cast<void**>(allocZero<int64_t>(P));
        out.lda_in = allocZero<int64_t>(P);
        out.ldb_in = allocZero<int64_t>(P);
        out.ldd_in = allocZero<int64_t>(P);
        out.ldb_out = allocZero<int64_t>(P);
        out.ldd_out = allocZero<int64_t>(P);
        out.splitk_offsets = with_splitk ? allocZero<int64_t>(P + 1) : nullptr;

        launchMoeLoraProblemBuilder(ranks_dev, ptrs_dev, reinterpret_cast<void const*>(input_base),
            reinterpret_cast<void*>(lowrank_workspace), reinterpret_cast<void*>(output_base), P, in_hidden_size,
            out_hidden_size, max_lora_rank, dtype_bytes, splitk_slices, out, mStream);
        TLLM_CUDA_CHECK(cudaStreamSynchronize(mStream));

        // Compare device outputs to host reference.
        auto check_int64 = [&](char const* name, int64_t* dev, std::vector<int64_t> const& ref_vec)
        {
            std::vector<int64_t> host(ref_vec.size(), 0);
            deviceDownload(dev, host);
            for (size_t i = 0; i < ref_vec.size(); ++i)
            {
                EXPECT_EQ(host[i], ref_vec[i]) << name << " mismatch at i=" << i;
            }
        };
        auto check_ptr_array = [&](char const* name, void** dev, std::vector<int64_t> const& ref_vec)
        { check_int64(name, reinterpret_cast<int64_t*>(dev), ref_vec); };
        auto check_problem_sizes
            = [&](char const* name, cutlass::gemm::GemmCoord* dev, std::vector<cutlass::gemm::GemmCoord> const& ref_vec)
        {
            std::vector<cutlass::gemm::GemmCoord> host(ref_vec.size());
            TLLM_CUDA_CHECK(cudaMemcpy(
                host.data(), dev, ref_vec.size() * sizeof(cutlass::gemm::GemmCoord), cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < ref_vec.size(); ++i)
            {
                EXPECT_EQ(host[i].m(), ref_vec[i].m()) << name << " M mismatch at i=" << i;
                EXPECT_EQ(host[i].n(), ref_vec[i].n()) << name << " N mismatch at i=" << i;
                EXPECT_EQ(host[i].k(), ref_vec[i].k()) << name << " K mismatch at i=" << i;
            }
        };

        check_problem_sizes("problem_sizes_in", out.problem_sizes_in, ref.problem_sizes_in);
        check_problem_sizes("problem_sizes_out", out.problem_sizes_out, ref.problem_sizes_out);
        check_ptr_array("a_ptrs_in", out.a_ptrs_in, ref.a_ptrs_in);
        check_ptr_array("b_ptrs_in", out.b_ptrs_in, ref.b_ptrs_in);
        check_ptr_array("d_ptrs_in", out.d_ptrs_in, ref.d_ptrs_in);
        check_ptr_array("b_ptrs_out", out.b_ptrs_out, ref.b_ptrs_out);
        check_ptr_array("d_ptrs_out", out.d_ptrs_out, ref.d_ptrs_out);
        check_int64("lda_in", out.lda_in, ref.lda_in);
        check_int64("ldb_in", out.ldb_in, ref.ldb_in);
        check_int64("ldd_in", out.ldd_in, ref.ldd_in);
        check_int64("ldb_out", out.ldb_out, ref.ldb_out);
        check_int64("ldd_out", out.ldd_out, ref.ldd_out);
        if (with_splitk)
        {
            check_int64("splitk_offsets", out.splitk_offsets, ref.splitk_offsets);
        }
    }

    cudaStream_t mStream{};
    std::vector<void*> mAllocations;
};

// "Pretend" adapter pointers. The kernel treats these as opaque bits, so
// we use easily-distinguishable patterns to catch indexing mistakes.
int64_t fakeAdapter(int tag, int32_t i, int side)
{
    return (static_cast<int64_t>(tag) << 56) | (static_cast<int64_t>(side) << 48) | (static_cast<int64_t>(i + 1) << 32);
}

TEST_F(MoeLoraProblemBuilderTest, Bf16Smoke)
{
    int64_t const in_hidden_size = 16;
    int64_t const out_hidden_size = 32;
    int64_t const max_lora_rank = 8;
    int64_t const dtype_bytes = 2;
    int64_t const splitk_slices = 4;

    int64_t const input_base = static_cast<int64_t>(0x1'0000'0000ull);
    int64_t const lowrank_workspace = static_cast<int64_t>(0x2'0000'0000ull);
    int64_t const output_base = static_cast<int64_t>(0x3'0000'0000ull);

    std::vector<int32_t> ranks = {2, 0, 4, 1, 8, 3};
    std::vector<int64_t> ptrs;
    for (int32_t i = 0; i < static_cast<int32_t>(ranks.size()); ++i)
    {
        ptrs.push_back(fakeAdapter(/*tag=*/1, i, /*side=*/0));
        ptrs.push_back(fakeAdapter(/*tag=*/1, i, /*side=*/1));
    }

    runAndCompare(ranks, ptrs, input_base, lowrank_workspace, output_base, in_hidden_size, out_hidden_size,
        max_lora_rank, dtype_bytes, splitk_slices);
}

TEST_F(MoeLoraProblemBuilderTest, Fp32StrideBytes)
{
    int64_t const in_hidden_size = 12;
    int64_t const out_hidden_size = 24;
    int64_t const max_lora_rank = 16;
    int64_t const dtype_bytes = 4;
    int64_t const splitk_slices = 8;

    int64_t const input_base = static_cast<int64_t>(0x4'0000'0000ull);
    int64_t const lowrank_workspace = static_cast<int64_t>(0x5'0000'0000ull);
    int64_t const output_base = static_cast<int64_t>(0x6'0000'0000ull);

    std::vector<int32_t> ranks = {1, 16, 8};
    std::vector<int64_t> ptrs;
    for (int32_t i = 0; i < static_cast<int32_t>(ranks.size()); ++i)
    {
        ptrs.push_back(fakeAdapter(/*tag=*/2, i, /*side=*/0));
        ptrs.push_back(fakeAdapter(/*tag=*/2, i, /*side=*/1));
    }

    runAndCompare(ranks, ptrs, input_base, lowrank_workspace, output_base, in_hidden_size, out_hidden_size,
        max_lora_rank, dtype_bytes, splitk_slices);
}

// Cover an empty call (no-op) and a single-token call (smallest live case)
// to lock down the corner cases the larger tests don't exercise.
TEST_F(MoeLoraProblemBuilderTest, BoundaryCases)
{
    int64_t const in_hidden_size = 8;
    int64_t const out_hidden_size = 8;
    int64_t const max_lora_rank = 4;
    int64_t const dtype_bytes = 2;
    int64_t const splitk_slices = 2;
    int64_t const input_base = static_cast<int64_t>(0x7'0000'0000ull);
    int64_t const lowrank_workspace = static_cast<int64_t>(0x8'0000'0000ull);
    int64_t const output_base = static_cast<int64_t>(0x9'0000'0000ull);

    // Empty call: P = 0, no allocations needed; launch should be a no-op.
    {
        MoeLoraGemmGroupArrays empty{};
        launchMoeLoraProblemBuilder(nullptr, nullptr, reinterpret_cast<void const*>(input_base),
            reinterpret_cast<void*>(lowrank_workspace), reinterpret_cast<void*>(output_base),
            /*num_permuted_tokens=*/0, in_hidden_size, out_hidden_size, max_lora_rank, dtype_bytes, splitk_slices,
            empty, mStream);
        TLLM_CUDA_CHECK(cudaStreamSynchronize(mStream));
    }

    // Single-token call: P = 1, exercises the +1 sentinel write at index 1.
    {
        std::vector<int32_t> ranks = {3};
        std::vector<int64_t> ptrs = {fakeAdapter(3, 0, 0), fakeAdapter(3, 0, 1)};
        runAndCompare(ranks, ptrs, input_base, lowrank_workspace, output_base, in_hidden_size, out_hidden_size,
            max_lora_rank, dtype_bytes, splitk_slices);
    }
}

// splitk_offsets == nullptr: the caller does not need the split-K scratch
// offsets, so the kernel must skip the sentinel write and the per-row offset
// store while still producing all other arrays correctly.
TEST_F(MoeLoraProblemBuilderTest, NullSplitkOffsets)
{
    int64_t const in_hidden_size = 16;
    int64_t const out_hidden_size = 32;
    int64_t const max_lora_rank = 8;
    int64_t const dtype_bytes = 2;
    int64_t const splitk_slices = 4;

    int64_t const input_base = static_cast<int64_t>(0xA'0000'0000ull);
    int64_t const lowrank_workspace = static_cast<int64_t>(0xB'0000'0000ull);
    int64_t const output_base = static_cast<int64_t>(0xC'0000'0000ull);

    std::vector<int32_t> ranks = {2, 0, 4, 1, 8};
    std::vector<int64_t> ptrs;
    for (int32_t i = 0; i < static_cast<int32_t>(ranks.size()); ++i)
    {
        ptrs.push_back(fakeAdapter(/*tag=*/4, i, /*side=*/0));
        ptrs.push_back(fakeAdapter(/*tag=*/4, i, /*side=*/1));
    }

    runAndCompare(ranks, ptrs, input_base, lowrank_workspace, output_base, in_hidden_size, out_hidden_size,
        max_lora_rank, dtype_bytes, splitk_slices, /*with_splitk=*/false);
}

// Returns a stable pointer for a given adapter id + side, independent of the
// row. Rows that share (adapter_id, rank) therefore form an aggregation run.
int64_t adapterById(int adapter_id, int side)
{
    return (static_cast<int64_t>(side + 1) << 56) | (static_cast<int64_t>(adapter_id + 1) << 24);
}

// Builds the per-row (A_ptr, B_ptr) table from a per-row adapter id. Rank-0
// rows get null pointers, matching the pointer-expand kernel's no-adapter path.
std::vector<int64_t> ptrsFromAdapterIds(std::vector<int32_t> const& ranks, std::vector<int> const& adapter_ids)
{
    std::vector<int64_t> ptrs;
    ptrs.reserve(ranks.size() * 2);
    for (size_t i = 0; i < ranks.size(); ++i)
    {
        if (ranks[i] == 0)
        {
            ptrs.push_back(0);
            ptrs.push_back(0);
        }
        else
        {
            ptrs.push_back(adapterById(adapter_ids[i], /*side=*/0));
            ptrs.push_back(adapterById(adapter_ids[i], /*side=*/1));
        }
    }
    return ptrs;
}

// All rows share one adapter and rank -> a single merged M=P problem, with the
// remaining P-1 slots padded as no-ops.
TEST_F(MoeLoraProblemBuilderTest, AllSameAdapterRun)
{
    int64_t const in_hidden_size = 16;
    int64_t const out_hidden_size = 32;
    int64_t const max_lora_rank = 8;
    int64_t const dtype_bytes = 2;
    int64_t const splitk_slices = 4;
    int64_t const input_base = static_cast<int64_t>(0xD'0000'0000ull);
    int64_t const lowrank_workspace = static_cast<int64_t>(0xE'0000'0000ull);
    int64_t const output_base = static_cast<int64_t>(0xF'0000'0000ull);

    std::vector<int32_t> ranks(6, 8);
    std::vector<int> adapter_ids(6, 7);
    auto ptrs = ptrsFromAdapterIds(ranks, adapter_ids);

    runAndCompare(ranks, ptrs, input_base, lowrank_workspace, output_base, in_hidden_size, out_hidden_size,
        max_lora_rank, dtype_bytes, splitk_slices);
}

// Adapters alternate A, B, A, B, ... so no two neighbors merge: every problem
// stays M=1, matching the un-aggregated behavior.
TEST_F(MoeLoraProblemBuilderTest, AlternatingAdaptersNoMerge)
{
    int64_t const in_hidden_size = 16;
    int64_t const out_hidden_size = 32;
    int64_t const max_lora_rank = 8;
    int64_t const dtype_bytes = 2;
    int64_t const splitk_slices = 4;
    int64_t const input_base = static_cast<int64_t>(0x11'0000'0000ull);
    int64_t const lowrank_workspace = static_cast<int64_t>(0x12'0000'0000ull);
    int64_t const output_base = static_cast<int64_t>(0x13'0000'0000ull);

    std::vector<int32_t> ranks = {8, 8, 8, 8, 8, 8};
    std::vector<int> adapter_ids = {0, 1, 0, 1, 0, 1};
    auto ptrs = ptrsFromAdapterIds(ranks, adapter_ids);

    runAndCompare(ranks, ptrs, input_base, lowrank_workspace, output_base, in_hidden_size, out_hidden_size,
        max_lora_rank, dtype_bytes, splitk_slices);
}

// Blocked runs of varying length and rank, including a contiguous rank-0 run
// (null pointers) that merges into a single no-op problem. Exercises the
// split-K prefix sum over uneven run lengths and the no-op padding tail.
TEST_F(MoeLoraProblemBuilderTest, BlockedRunsVaryingRanksAndRank0)
{
    int64_t const in_hidden_size = 16;
    int64_t const out_hidden_size = 32;
    int64_t const max_lora_rank = 16;
    int64_t const dtype_bytes = 2;
    int64_t const splitk_slices = 4;
    int64_t const input_base = static_cast<int64_t>(0x14'0000'0000ull);
    int64_t const lowrank_workspace = static_cast<int64_t>(0x15'0000'0000ull);
    int64_t const output_base = static_cast<int64_t>(0x16'0000'0000ull);

    //                       |-- run A (rank 8) --|  |-- rank-0 --|  |-- run B (rank 16) --|  | C |
    std::vector<int32_t> ranks = {8, 8, 8, 0, 0, 16, 16, 4};
    std::vector<int> adapter_ids = {1, 1, 1, -1, -1, 2, 2, 3};
    auto ptrs = ptrsFromAdapterIds(ranks, adapter_ids);

    runAndCompare(ranks, ptrs, input_base, lowrank_workspace, output_base, in_hidden_size, out_hidden_size,
        max_lora_rank, dtype_bytes, splitk_slices);
}

// Same adapter id but a rank change must still break the run (rank is part of
// the aggregation key).
TEST_F(MoeLoraProblemBuilderTest, RankChangeBreaksRun)
{
    int64_t const in_hidden_size = 16;
    int64_t const out_hidden_size = 32;
    int64_t const max_lora_rank = 16;
    int64_t const dtype_bytes = 2;
    int64_t const splitk_slices = 2;
    int64_t const input_base = static_cast<int64_t>(0x17'0000'0000ull);
    int64_t const lowrank_workspace = static_cast<int64_t>(0x18'0000'0000ull);
    int64_t const output_base = static_cast<int64_t>(0x19'0000'0000ull);

    std::vector<int32_t> ranks = {8, 8, 16, 16};
    std::vector<int> adapter_ids = {5, 5, 5, 5}; // same adapter, differing rank
    auto ptrs = ptrsFromAdapterIds(ranks, adapter_ids);

    runAndCompare(ranks, ptrs, input_base, lowrank_workspace, output_base, in_hidden_size, out_hidden_size,
        max_lora_rank, dtype_bytes, splitk_slices);
}

} // namespace
