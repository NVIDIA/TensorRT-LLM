/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 */

// Kernel-level numerical accuracy test for cascade attention.
//
// Validates that `launch_cascade_attention` produces bit-accurate (within FP16
// tolerance) results compared to the baseline `masked_multihead_attention`
// kernel under beam-search decode conditions.
//
// Strategy:
//   1. Construct a realistic beam-search MMHA scenario with random Q/KV data.
//   2. Call baseline MMHA (via the external API, env-var off → cascade disabled).
//   3. Call cascade attention directly (bypasses env-var check).
//   4. Compare outputs element-by-element with FP16 tolerance.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention.h"
#include "tensorrt_llm/kernels/decoderMaskedMultiheadAttention/cascadeAttentionKernel.h"
#include "tensorrt_llm/kernels/kvCacheUtils.h"

namespace tk = tensorrt_llm::kernels;
namespace tc = tensorrt_llm::common;

namespace
{

// Test configuration constants.
constexpr int kDh = 128;
constexpr int kNumHeads = 8;
constexpr int kNumKvHeads = 8;
constexpr int kBeamWidth = 4;
constexpr int kNumRequests = 2;
constexpr int kBatchSize = kNumRequests * kBeamWidth; // = 8
constexpr int kPrefixLen = 64;
constexpr int kSuffixLen = 16;
constexpr int kTotalLen = kPrefixLen + kSuffixLen; // = 80
constexpr int kMaxSeqLen = 256;

// FP16 tolerance: allow |a - b| <= atol + rtol * |b|.
inline bool almostEqualFp16(float a, float b, float atol = 5e-3f, float rtol = 1e-2f)
{
    if (std::isnan(a) && std::isnan(b))
    {
        return true;
    }
    return std::fabs(a - b) <= (atol + rtol * std::fabs(b));
}

// RAII helper for CUDA memory allocation.
struct CudaBuf
{
    void* ptr = nullptr;
    size_t bytes = 0;

    CudaBuf() = default;

    explicit CudaBuf(size_t n)
        : bytes(n)
    {
        TLLM_CUDA_CHECK(cudaMalloc(&ptr, n));
        TLLM_CUDA_CHECK(cudaMemset(ptr, 0, n));
    }

    ~CudaBuf()
    {
        if (ptr)
        {
            cudaFree(ptr);
        }
    }

    CudaBuf(CudaBuf const&) = delete;
    CudaBuf& operator=(CudaBuf const&) = delete;

    CudaBuf(CudaBuf&& o) noexcept
        : ptr(o.ptr)
        , bytes(o.bytes)
    {
        o.ptr = nullptr;
    }

    template <typename T>
    T* as()
    {
        return static_cast<T*>(ptr);
    }
};

// Fill a host buffer with random FP16 values (normal distribution).
void fillRandomHalf(half* dst, size_t count, std::mt19937& rng, float mean = 0.0f, float stddev = 0.3f)
{
    std::normal_distribution<float> dist(mean, stddev);
    for (size_t i = 0; i < count; ++i)
    {
        dst[i] = __float2half(dist(rng));
    }
}

} // namespace

class CascadeAttentionNumericsTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        int smVersion = tc::getSMVersion();
        if (smVersion < 80)
        {
            GTEST_SKIP() << "Cascade attention requires SM >= 80, current SM = " << smVersion;
        }
    }
};

// Core test: cascade output must match baseline MMHA output for beam-search decode.
TEST_F(CascadeAttentionNumericsTest, CascadeMatchesBaselineBeamSearch)
{
    std::mt19937 rng(42);
    cudaStream_t stream;
    TLLM_CUDA_CHECK(cudaStreamCreate(&stream));

    // ----- Allocate host data -----
    size_t const q_elems = static_cast<size_t>(kBatchSize) * kNumHeads * kDh;
    size_t const kv_cache_size_per_seq = 2 * kNumKvHeads * kDh; // K+V per token
    size_t const kv_cache_total = static_cast<size_t>(kBatchSize) * kMaxSeqLen * kv_cache_size_per_seq;
    size_t const cache_indir_elems = static_cast<size_t>(kBatchSize) * kMaxSeqLen;

    std::vector<half> h_q(q_elems);
    std::vector<half> h_kv_cache(kv_cache_total);
    std::vector<int> h_cache_indir(cache_indir_elems, 0);
    std::vector<int> h_input_lengths(kBatchSize, kPrefixLen);
    std::vector<int> h_length_per_sample(kBatchSize, kTotalLen);

    fillRandomHalf(h_q.data(), q_elems, rng);
    fillRandomHalf(h_kv_cache.data(), kv_cache_total, rng);

    // Set up cache_indir: for beam search, beam 0 of each request is the
    // "parent". Simple identity mapping for test (no beam divergence at suffix).
    for (int seq = 0; seq < kBatchSize; ++seq)
    {
        int req = seq / kBeamWidth;
        int beam0_seq = req * kBeamWidth; // first beam of the request
        for (int t = 0; t < kTotalLen; ++t)
        {
            // Before prefix end: all beams share parent beam 0.
            // After prefix: each beam is its own parent (identity).
            h_cache_indir[seq * kMaxSeqLen + t] = (t < kPrefixLen) ? beam0_seq : seq;
        }
    }

    // ----- Allocate device memory -----
    CudaBuf d_q(q_elems * sizeof(half));
    CudaBuf d_kv_cache(kv_cache_total * sizeof(half));
    CudaBuf d_cache_indir(cache_indir_elems * sizeof(int));
    CudaBuf d_input_lengths(kBatchSize * sizeof(int));
    CudaBuf d_length_per_sample(kBatchSize * sizeof(int));
    CudaBuf d_out_baseline(q_elems * sizeof(half));
    CudaBuf d_out_cascade(q_elems * sizeof(half));

    // Baseline multi-block workspace (generous allocation).
    int const max_seq_len_tile = 64;
    CudaBuf d_partial_out(static_cast<size_t>(max_seq_len_tile) * kBatchSize * kNumHeads * kDh * sizeof(float));
    CudaBuf d_partial_sum(static_cast<size_t>(max_seq_len_tile) * kBatchSize * kNumHeads * sizeof(float));
    CudaBuf d_partial_max(static_cast<size_t>(max_seq_len_tile) * kBatchSize * kNumHeads * sizeof(float));
    CudaBuf d_block_counter(static_cast<size_t>(kBatchSize) * kNumHeads * sizeof(int));

    // Cascade workspace.
    auto ws = tk::mmha::cascade::getCascadeWorkspaceSizes(kBatchSize, kNumHeads, kDh);
    CudaBuf d_cascade_out(ws.out);
    CudaBuf d_cascade_max(ws.mMax);
    CudaBuf d_cascade_sum(ws.lSum);

    // ----- H2D copies -----
    TLLM_CUDA_CHECK(cudaMemcpy(d_q.ptr, h_q.data(), q_elems * sizeof(half), cudaMemcpyHostToDevice));
    TLLM_CUDA_CHECK(
        cudaMemcpy(d_kv_cache.ptr, h_kv_cache.data(), kv_cache_total * sizeof(half), cudaMemcpyHostToDevice));
    TLLM_CUDA_CHECK(
        cudaMemcpy(d_cache_indir.ptr, h_cache_indir.data(), cache_indir_elems * sizeof(int), cudaMemcpyHostToDevice));
    TLLM_CUDA_CHECK(
        cudaMemcpy(d_input_lengths.ptr, h_input_lengths.data(), kBatchSize * sizeof(int), cudaMemcpyHostToDevice));
    TLLM_CUDA_CHECK(cudaMemcpy(
        d_length_per_sample.ptr, h_length_per_sample.data(), kBatchSize * sizeof(int), cudaMemcpyHostToDevice));

    // ----- Build KVLinearBuffer -----
    // KVLinearBuffer layout: [batch_size, max_seq_len, 2 * kv_heads * Dh]
    // Data pointer type is int8_t (raw bytes).
    int const sizePerToken = kv_cache_size_per_seq * sizeof(half);
    tk::KVLinearBuffer kv_buffer(
        kBatchSize, kMaxSeqLen, sizePerToken, kMaxSeqLen, /*sinkTokenLen=*/0, false, d_kv_cache.as<int8_t>());
    // shift_k_cache is unused when position_shift is disabled; pass a dummy.
    tk::KVLinearBuffer shift_k_cache(0, 0, 0, 0, 0, false, nullptr);

    // ----- Fill Masked_multihead_attention_params -----
    auto buildParams = [&](void* out_ptr) -> tk::Masked_multihead_attention_params<half>
    {
        tk::Masked_multihead_attention_params<half> p{};
        p.out = out_ptr;
        p.q = d_q.as<half>();
        p.k = nullptr; // K/V come from cache for decode
        p.v = nullptr;
        p.cache_indir = d_cache_indir.as<int>();
        p.batch_size = kBatchSize;
        p.beam_width = kBeamWidth;
        p.num_heads = kNumHeads;
        p.num_kv_heads = kNumKvHeads;
        p.hidden_size_per_head = kDh;
        p.inv_sqrt_dh = 1.0f / sqrtf(static_cast<float>(kDh));
        p.timestep = kTotalLen - 1; // 0-indexed last token position
        p.max_decoder_seq_len = kMaxSeqLen;
        p.input_lengths = d_input_lengths.as<int>();
        p.length_per_sample = d_length_per_sample.as<int>();
        p.position_embedding_type = tk::PositionEmbeddingType::kLEARNED_ABSOLUTE;
        p.position_shift_enabled = false;
        p.block_sparse_attention = false;
        p.attn_logit_softcapping_scale = 0.0f;
        p.relative_attention_bias = nullptr;
        p.linear_bias_slopes = nullptr;
        p.attention_mask = nullptr;
        p.attention_sinks = nullptr;
        p.int8_kv_cache = false;
        p.fp8_kv_cache = false;
        p.cyclic_attention_window_size = kMaxSeqLen;
        p.max_attention_window_size = kMaxSeqLen;
        p.chunked_attention_size = INT_MAX;
        p.sink_token_length = 0;
        // Multi-block mode workspace (for baseline).
        p.multi_block_mode = true;
        p.multi_processor_count = tc::getMultiProcessorCount();
        p.min_seq_len_tile = 1;
        p.max_seq_len_tile = max_seq_len_tile;
        p.partial_out = d_partial_out.as<half>();
        p.partial_sum = d_partial_sum.as<float>();
        p.partial_max = d_partial_max.as<float>();
        p.block_counter = d_block_counter.as<int>();
        // Cascade workspace (only used by cascade path).
        p.cascade_partial_out = d_cascade_out.as<float>();
        p.cascade_partial_max = d_cascade_max.as<float>();
        p.cascade_partial_sum = d_cascade_sum.as<float>();
        return p;
    };

    // ----- Run baseline MMHA -----
    {
        auto params = buildParams(d_out_baseline.ptr);
        // Use reinterpret_cast because the external API uses uint16_t for half.
        auto const& params_u16 = reinterpret_cast<tk::Masked_multihead_attention_params<uint16_t> const&>(params);
        tk::masked_multihead_attention(params_u16, kv_buffer, shift_k_cache, stream);
        TLLM_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // ----- Run cascade attention -----
    {
        auto params = buildParams(d_out_cascade.ptr);
        bool launched = tk::mmha::cascade::launch_cascade_attention<half, half, tk::KVLinearBuffer, kDh>(
            params, kv_buffer, stream);
        ASSERT_TRUE(launched) << "launch_cascade_attention returned false (workspace issue?)";
        TLLM_CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // ----- D2H copy and compare -----
    std::vector<half> h_out_baseline(q_elems);
    std::vector<half> h_out_cascade(q_elems);
    TLLM_CUDA_CHECK(
        cudaMemcpy(h_out_baseline.data(), d_out_baseline.ptr, q_elems * sizeof(half), cudaMemcpyDeviceToHost));
    TLLM_CUDA_CHECK(
        cudaMemcpy(h_out_cascade.data(), d_out_cascade.ptr, q_elems * sizeof(half), cudaMemcpyDeviceToHost));

    int mismatches = 0;
    constexpr int kMaxMismatchPrint = 10;
    for (size_t i = 0; i < q_elems; ++i)
    {
        float const baseline_val = __half2float(h_out_baseline[i]);
        float const cascade_val = __half2float(h_out_cascade[i]);
        if (!almostEqualFp16(cascade_val, baseline_val))
        {
            if (mismatches < kMaxMismatchPrint)
            {
                int seq = static_cast<int>(i / (kNumHeads * kDh));
                int head = static_cast<int>((i % (kNumHeads * kDh)) / kDh);
                int ch = static_cast<int>(i % kDh);
                TLLM_LOG_ERROR("Mismatch at [seq=%d, head=%d, ch=%d]: baseline=%e, cascade=%e, diff=%e", seq, head, ch,
                    baseline_val, cascade_val, cascade_val - baseline_val);
            }
            ++mismatches;
        }
    }

    float const mismatch_pct = 100.0f * mismatches / static_cast<float>(q_elems);
    EXPECT_EQ(mismatches, 0) << mismatches << " / " << q_elems << " elements mismatched (" << mismatch_pct << "%)";

    TLLM_CUDA_CHECK(cudaStreamDestroy(stream));
}
