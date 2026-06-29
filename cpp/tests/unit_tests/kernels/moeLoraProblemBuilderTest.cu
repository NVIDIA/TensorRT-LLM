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

// Host-side reference reproducing the builder's per-row logic. Same
// formulas as the kernel; used only as parity ground truth.
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
    for (int64_t i = 0; i < P; ++i)
    {
        int32_t const rank = ranks[i];
        int const out_n = (rank > 0) ? static_cast<int>(out_hidden_size) : 0;
        r.problem_sizes_in[i] = cutlass::gemm::GemmCoord(1, rank, static_cast<int>(in_hidden_size));
        r.problem_sizes_out[i] = cutlass::gemm::GemmCoord(1, out_n, rank);

        int64_t const in_row_stride = in_hidden_size * dtype_bytes;
        int64_t const work_row_stride = max_lora_rank * dtype_bytes;
        int64_t const out_row_stride = out_hidden_size * dtype_bytes;

        r.a_ptrs_in[i] = input_base + i * in_row_stride;
        r.b_ptrs_in[i] = ptrs[2 * i + 0];
        r.d_ptrs_in[i] = lowrank_workspace + i * work_row_stride;
        r.b_ptrs_out[i] = ptrs[2 * i + 1];
        r.d_ptrs_out[i] = output_base + i * out_row_stride;

        r.lda_in[i] = in_hidden_size;
        r.ldb_in[i] = in_hidden_size;
        r.ldd_in[i] = max_lora_rank;
        r.ldb_out[i] = rank;
        r.ldd_out[i] = out_hidden_size;

        r.splitk_offsets[i] = i * max_lora_rank * splitk_slices;
    }
    r.splitk_offsets[P] = P * max_lora_rank * splitk_slices;
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

} // namespace
