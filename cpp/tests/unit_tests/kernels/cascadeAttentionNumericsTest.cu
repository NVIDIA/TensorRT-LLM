/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

// Kernel-level numerical accuracy test for cascade attention.
//
// Validates that `launch_cascade_attention` produces bit-accurate (within FP16
// tolerance) results compared to the baseline `masked_multihead_attention`
// kernel under beam-search decode conditions.
//
// Strategy:
//   1. Construct a realistic beam-search MMHA scenario with random Q/KV data.
//   2. Call baseline MMHA (via the external API, env-var off -> cascade disabled).
//   3. Call cascade attention directly (bypasses env-var check).
//   4. Compare outputs element-by-element with FP16 tolerance.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
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

// Fixed head dimension — cascade kernel only supports Dh == 128.
constexpr int kDh = 128;

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

// Parameters that vary across test cases.
struct CascadeTestParams
{
    int numHeads;
    int numKvHeads;
    int beamWidth;
    int numRequests;
    int prefixLen;
    int suffixLen;
    int maxSeqLen;
    bool useBias; // when true, populate q/k/v_bias (qkv_bias enabled, e.g. Qwen2)
    bool useRope; // when true, use GPT-NeoX RoPE instead of learned-absolute
};

// Pretty-print for gtest output.
std::string PrintCascadeTestParams(testing::TestParamInfo<CascadeTestParams> const& info)
{
    auto const& p = info.param;
    return "H" + std::to_string(p.numHeads) + "_KVH" + std::to_string(p.numKvHeads) + "_B" + std::to_string(p.beamWidth)
        + "_R" + std::to_string(p.numRequests) + "_P" + std::to_string(p.prefixLen) + "_S" + std::to_string(p.suffixLen)
        + (p.useBias ? "_bias" : "") + (p.useRope ? "_rope" : "");
}

} // namespace

class CascadeAttentionNumericsTest : public ::testing::TestWithParam<CascadeTestParams>
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
TEST_P(CascadeAttentionNumericsTest, CascadeMatchesBaseline)
{
    auto const& tp = GetParam();
    int const batchSize = tp.numRequests * tp.beamWidth;
    int const totalLen = tp.prefixLen + tp.suffixLen;

    std::mt19937 rng(42);
    cudaStream_t stream;
    TLLM_CUDA_CHECK(cudaStreamCreate(&stream));

    // ----- Allocate host data -----
    size_t const q_elems = static_cast<size_t>(batchSize) * tp.numHeads * kDh;
    // KVLinearBuffer layout: [B, 2(K/V), H, S, D].  sizePerToken is per-K (or
    // per-V) only; the buffer internally doubles via mValidRowsPerSeq=2.
    size_t const kv_elems_per_token = static_cast<size_t>(tp.numKvHeads) * kDh;
    size_t const sizePerToken = kv_elems_per_token * sizeof(half);
    size_t const kv_cache_total = static_cast<size_t>(batchSize) * 2 * tp.maxSeqLen * kv_elems_per_token;
    size_t const cache_indir_elems = static_cast<size_t>(batchSize) * tp.maxSeqLen;

    std::vector<half> h_q(q_elems);
    std::vector<half> h_kv_cache(kv_cache_total);
    // Current-step K and V inputs: the MMHA kernel reads these and writes them
    // into the KV cache at position `timestep`.
    size_t const kv_step_elems = static_cast<size_t>(batchSize) * tp.numKvHeads * kDh;
    std::vector<half> h_k(kv_step_elems);
    std::vector<half> h_v(kv_step_elems);
    std::vector<int> h_cache_indir(cache_indir_elems, 0);
    std::vector<int> h_input_lengths(batchSize, tp.prefixLen);
    std::vector<int> h_length_per_sample(batchSize, totalLen);

    fillRandomHalf(h_q.data(), q_elems, rng);
    fillRandomHalf(h_kv_cache.data(), kv_cache_total, rng);
    fillRandomHalf(h_k.data(), kv_step_elems, rng);
    fillRandomHalf(h_v.data(), kv_step_elems, rng);

    // qkv_bias buffers (only populated when tp.useBias).  Baseline MMHA applies
    // q/k/v bias before RoPE; the cascade kernels must reproduce the same result.
    // q_bias is sized per Q head, k/v_bias per KV head (matches kernel indexing).
    size_t const q_bias_elems = static_cast<size_t>(tp.numHeads) * kDh;
    size_t const kv_bias_elems = static_cast<size_t>(tp.numKvHeads) * kDh;
    std::vector<half> h_q_bias(q_bias_elems, __float2half(0.f));
    std::vector<half> h_k_bias(kv_bias_elems, __float2half(0.f));
    std::vector<half> h_v_bias(kv_bias_elems, __float2half(0.f));
    if (tp.useBias)
    {
        fillRandomHalf(h_q_bias.data(), q_bias_elems, rng);
        fillRandomHalf(h_k_bias.data(), kv_bias_elems, rng);
        fillRandomHalf(h_v_bias.data(), kv_bias_elems, rng);
    }

    // Set up cache_indir: the kernel reads beam_offset = cache_indir[batch_beam * max_attn_win + t]
    // and computes seqIdx = batch_idx * beam_width + beam_offset.  So the stored
    // value must be the *beam-local* index (0 .. beam_width-1), not the global
    // batch-beam index.
    for (int seq = 0; seq < batchSize; ++seq)
    {
        int beam = seq % tp.beamWidth;
        for (int t = 0; t < totalLen; ++t)
        {
            h_cache_indir[seq * tp.maxSeqLen + t] = (t < tp.prefixLen) ? 0 : beam;
        }
    }

    // ----- Allocate device memory -----
    CudaBuf d_q(q_elems * sizeof(half));
    CudaBuf d_k(kv_step_elems * sizeof(half));
    CudaBuf d_v(kv_step_elems * sizeof(half));
    CudaBuf d_kv_cache(kv_cache_total * sizeof(half));
    CudaBuf d_cache_indir(cache_indir_elems * sizeof(int));
    CudaBuf d_input_lengths(batchSize * sizeof(int));
    CudaBuf d_length_per_sample(batchSize * sizeof(int));
    CudaBuf d_out_baseline(q_elems * sizeof(half));
    CudaBuf d_out_cascade(q_elems * sizeof(half));
    CudaBuf d_q_bias(q_bias_elems * sizeof(half));
    CudaBuf d_k_bias(kv_bias_elems * sizeof(half));
    CudaBuf d_v_bias(kv_bias_elems * sizeof(half));

    // Baseline multi-block workspace (generous allocation).
    int const max_seq_len_tile = 64;
    CudaBuf d_partial_out(static_cast<size_t>(max_seq_len_tile) * batchSize * tp.numHeads * kDh * sizeof(float));
    CudaBuf d_partial_sum(static_cast<size_t>(max_seq_len_tile) * batchSize * tp.numHeads * sizeof(float));
    CudaBuf d_partial_max(static_cast<size_t>(max_seq_len_tile) * batchSize * tp.numHeads * sizeof(float));
    CudaBuf d_block_counter(static_cast<size_t>(batchSize) * tp.numHeads * sizeof(int));

    // Cascade workspace.
    auto ws = tk::mmha::cascade::getCascadeWorkspaceSizes(batchSize, tp.numHeads, kDh);
    CudaBuf d_cascade_out(ws.out);
    CudaBuf d_cascade_max(ws.mMax);
    CudaBuf d_cascade_sum(ws.lSum);

    // ----- H2D copies -----
    TLLM_CUDA_CHECK(cudaMemcpy(d_q.ptr, h_q.data(), q_elems * sizeof(half), cudaMemcpyHostToDevice));
    TLLM_CUDA_CHECK(cudaMemcpy(d_k.ptr, h_k.data(), kv_step_elems * sizeof(half), cudaMemcpyHostToDevice));
    TLLM_CUDA_CHECK(cudaMemcpy(d_v.ptr, h_v.data(), kv_step_elems * sizeof(half), cudaMemcpyHostToDevice));
    TLLM_CUDA_CHECK(
        cudaMemcpy(d_kv_cache.ptr, h_kv_cache.data(), kv_cache_total * sizeof(half), cudaMemcpyHostToDevice));
    TLLM_CUDA_CHECK(
        cudaMemcpy(d_cache_indir.ptr, h_cache_indir.data(), cache_indir_elems * sizeof(int), cudaMemcpyHostToDevice));
    TLLM_CUDA_CHECK(
        cudaMemcpy(d_input_lengths.ptr, h_input_lengths.data(), batchSize * sizeof(int), cudaMemcpyHostToDevice));
    TLLM_CUDA_CHECK(cudaMemcpy(
        d_length_per_sample.ptr, h_length_per_sample.data(), batchSize * sizeof(int), cudaMemcpyHostToDevice));
    if (tp.useBias)
    {
        TLLM_CUDA_CHECK(cudaMemcpy(d_q_bias.ptr, h_q_bias.data(), q_bias_elems * sizeof(half), cudaMemcpyHostToDevice));
        TLLM_CUDA_CHECK(
            cudaMemcpy(d_k_bias.ptr, h_k_bias.data(), kv_bias_elems * sizeof(half), cudaMemcpyHostToDevice));
        TLLM_CUDA_CHECK(
            cudaMemcpy(d_v_bias.ptr, h_v_bias.data(), kv_bias_elems * sizeof(half), cudaMemcpyHostToDevice));
    }

    // ----- Build KVLinearBuffer -----
    // KVLinearBuffer layout: [B, 2(K/V), H, S, D].
    // sizePerToken is per-K (or per-V); mValidRowsPerSeq=2 handles the x2.
    tk::KVLinearBuffer kv_buffer(batchSize, tp.maxSeqLen, static_cast<int32_t>(sizePerToken), tp.maxSeqLen,
        /*sinkTokenLen=*/0, /*onlyKorV=*/false, d_kv_cache.as<int8_t>());
    // shift_k_cache is unused when position_shift is disabled; pass a dummy.
    tk::KVLinearBuffer shift_k_cache(0, 0, 0, 0, 0, false, nullptr);

    // ----- Fill Masked_multihead_attention_params -----
    auto buildParams = [&](void* out_ptr) -> tk::Masked_multihead_attention_params<half>
    {
        tk::Masked_multihead_attention_params<half> p{};
        p.out = out_ptr;
        p.q = d_q.as<half>();
        p.k = d_k.as<half>();
        p.v = d_v.as<half>();
        p.cache_indir = d_cache_indir.as<int>();
        p.batch_size = batchSize;
        p.beam_width = tp.beamWidth;
        p.num_heads = tp.numHeads;
        p.num_kv_heads = tp.numKvHeads;
        p.hidden_size_per_head = kDh;
        p.inv_sqrt_dh = 1.0f / sqrtf(static_cast<float>(kDh));
        p.timestep = totalLen - 1;
        p.max_decoder_seq_len = tp.maxSeqLen;
        p.input_lengths = d_input_lengths.as<int>();
        p.length_per_sample = d_length_per_sample.as<int>();
        if (tp.useRope)
        {
            // GPT-NeoX full-head RoPE, no scaling.  m_scale (=1.0) and vision_start/
            // length (=-1) come from the param struct defaults, so cascade (which
            // hardcodes mscale=1.0 and no vision shift) and baseline compute the
            // exact same rotary coefficients via rotary_embedding_coefficient().
            p.position_embedding_type = tk::PositionEmbeddingType::kROPE_GPT_NEOX;
            p.rotary_embedding_dim = kDh;
            p.rotary_embedding_base = 1000000.0f;
            p.rotary_embedding_scale = 1.0f;
            p.rotary_embedding_scale_type = tk::RotaryScalingType::kNONE;
        }
        else
        {
            p.position_embedding_type = tk::PositionEmbeddingType::kLEARNED_ABSOLUTE;
        }
        p.position_shift_enabled = false;
        p.block_sparse_attention = false;
        p.attn_logit_softcapping_scale = 0.0f;
        p.relative_attention_bias = nullptr;
        p.linear_bias_slopes = nullptr;
        p.attention_mask = nullptr;
        p.attention_sinks = nullptr;
        p.int8_kv_cache = false;
        p.fp8_kv_cache = false;
        p.cyclic_attention_window_size = tp.maxSeqLen;
        p.max_attention_window_size = tp.maxSeqLen;
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
        if (tp.useBias)
        {
            p.q_bias = d_q_bias.as<half>();
            p.k_bias = d_k_bias.as<half>();
            p.v_bias = d_v_bias.as<half>();
        }
        return p;
    };

    // ----- Run baseline MMHA -----
    {
        auto params = buildParams(d_out_baseline.ptr);
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
                int seq = static_cast<int>(i / (tp.numHeads * kDh));
                int head = static_cast<int>((i % (tp.numHeads * kDh)) / kDh);
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

// clang-format off
INSTANTIATE_TEST_SUITE_P(CascadeNumerics, CascadeAttentionNumericsTest,
    ::testing::Values(
        //                       numHeads  numKvHeads  beamWidth  numRequests  prefixLen  suffixLen  maxSeqLen  useBias  useRope
        CascadeTestParams{               8,         8,         4,           2,        64,        16,       256,    false,   false},
        CascadeTestParams{               8,         8,         2,           4,        64,        16,       256,    false,   false},
        CascadeTestParams{              32,         8,         4,           1,       128,        32,       512,    false,   false},
        CascadeTestParams{              16,        16,         4,           2,       256,         8,       512,    false,   false},
        CascadeTestParams{               8,         8,         4,           1,        32,         4,       128,    false,   false},
        CascadeTestParams{               8,         1,         4,           2,        64,        16,       256,    false,   false},
        // qkv_bias enabled (learned-absolute): exercises q_bias/k_bias/v_bias add.
        // MHA / GQA / MQA head groupings.
        CascadeTestParams{               8,         8,         4,           2,        64,        16,       256,    true,    false},
        CascadeTestParams{              32,         8,         4,           1,       128,        32,       512,    true,    false},
        CascadeTestParams{               8,         1,         4,           2,        64,        16,       256,    true,    false},
        // GPT-NeoX RoPE, no bias: covers the cascade RoPE path numerically.
        CascadeTestParams{               8,         8,         4,           2,        64,        16,       256,    false,   true},
        CascadeTestParams{              32,         8,         4,           1,       128,        32,       512,    false,   true},
        // GPT-NeoX RoPE + qkv_bias: the real Qwen2-style config (bias before RoPE).
        CascadeTestParams{               8,         8,         4,           2,        64,        16,       256,    true,    true},
        CascadeTestParams{              32,         8,         4,           1,       128,        32,       512,    true,    true},
        CascadeTestParams{               8,         1,         4,           2,        64,        16,       256,    true,    true}
    ),
    PrintCascadeTestParams);
// clang-format on
