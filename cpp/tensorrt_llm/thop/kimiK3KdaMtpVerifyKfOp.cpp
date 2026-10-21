/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/kernels/kimiK3KdaMtpVerifyKf.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>
#include <limits>
#include <optional>

namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

constexpr int64_t kHeadDim = 128;   // K = V = 128
constexpr int64_t kConvHistory = 3; // conv width 4

void checkDevice(at::Tensor const& t, char const* name, at::Device const& device)
{
    TORCH_CHECK(t.is_cuda() && t.device() == device, "kimi_k3_kda_mtp_verify_kf: ", name,
        " must be a CUDA tensor on the device of x_q");
}

// [1, T, H, 128] BF16 with contiguous head blocks; returns the row stride.
int64_t headRowsStride(at::Tensor const& t, char const* name, int64_t T, int64_t H)
{
    TORCH_CHECK(t.scalar_type() == at::kBFloat16, "kimi_k3_kda_mtp_verify_kf: ", name, " must be bfloat16");
    TORCH_CHECK(t.dim() == 4 && t.size(0) == 1 && t.size(1) == T && t.size(2) == H && t.size(3) == kHeadDim,
        "kimi_k3_kda_mtp_verify_kf: ", name, " must have shape [1, T, H, 128]");
    TORCH_CHECK(t.stride(3) == 1 && t.stride(2) == kHeadDim, "kimi_k3_kda_mtp_verify_kf: ", name,
        " must have contiguous head blocks");
    return t.stride(1);
}

void checkContiguousF32(at::Tensor const& t, char const* name, std::initializer_list<int64_t> shape)
{
    TORCH_CHECK(t.scalar_type() == at::kFloat && t.is_contiguous() && t.sizes().vec() == std::vector<int64_t>(shape),
        "kimi_k3_kda_mtp_verify_kf: ", name, " must be a contiguous float32 tensor of the expected shape");
}

// Replay cache [>= pool, num_spec, width] with a contiguous last dimension.
void checkReplayCache(
    at::Tensor const& t, char const* name, at::ScalarType dtype, int64_t pool, int64_t numSpec, int64_t width)
{
    TORCH_CHECK(t.scalar_type() == dtype, "kimi_k3_kda_mtp_verify_kf: ", name, " has the wrong dtype");
    TORCH_CHECK(t.dim() == 3 && t.size(0) >= pool && t.size(1) == numSpec && t.size(2) == width,
        "kimi_k3_kda_mtp_verify_kf: ", name, " must have shape [>= pool, num_spec, ", width, "]");
    TORCH_CHECK(t.stride(2) == 1, "kimi_k3_kda_mtp_verify_kf: ", name, " must be contiguous in its last dimension");
}

// Conv window [>= pool, H * 128, >= 3 + num_spec] BF16, dim-contiguous.
void checkConvWindow(at::Tensor const& t, char const* name, int64_t pool, int64_t dim, int64_t numSpec)
{
    TORCH_CHECK(t.scalar_type() == at::kBFloat16, "kimi_k3_kda_mtp_verify_kf: ", name, " must be bfloat16");
    TORCH_CHECK(t.dim() == 3 && t.size(0) >= pool && t.size(1) == dim && t.size(2) >= kConvHistory + numSpec,
        "kimi_k3_kda_mtp_verify_kf: ", name, " must have shape [>= pool, H * 128, >= 3 + num_spec]");
    TORCH_CHECK(t.stride(1) == 1, "kimi_k3_kda_mtp_verify_kf: ", name,
        " must be dim-contiguous (allocate [pool, S, dim] and transpose(1, 2))");
}

} // namespace

// Kimi K3 KDA MTP verify on the packed token layout with the fused gated output RMSNorm: the contract of
// trtllm::kda_mtp_decode (out BF16 [1, T, H, 128], out_scale None) and trtllm::kda_mtp_decode_fp8 (out E4M3
// [T, H * 128] plus out_scale, the UE8M0 R128c4 scale bytes) with packed_token_layout = True and
// fuse_output_norm = True, through the Kernel Factory kernel of invokeKimiK3KdaMtpVerifyKf (which currently needs a
// BF16 state and the MXFP8 output, see isKimiK3KdaMtpVerifyKfSupported). Outputs are caller-allocated; conv windows,
// state and replay caches are updated in place. Slot indices are not range-checked (device values), as in the
// reference ops.
void kimi_k3_kda_mtp_verify_kf(at::Tensor const& x_q, at::Tensor const& x_k, at::Tensor const& x_v, at::Tensor const& g,
    at::Tensor const& beta, at::Tensor const& onorm_g, at::Tensor const& w_q, at::Tensor const& w_k,
    at::Tensor const& w_v, at::Tensor const& A_log, at::Tensor const& dt_bias, at::Tensor const& onorm_weight,
    at::Tensor const& ssm_state_indices, at::Tensor const& num_accepted_tokens, at::Tensor const& cs_q,
    at::Tensor const& cs_k, at::Tensor const& cs_v, at::Tensor const& recurrent_state, at::Tensor const& k_cache,
    at::Tensor const& g_cache, at::Tensor const& v_cache, at::Tensor const& beta_cache, at::Tensor const& out,
    std::optional<at::Tensor> const& out_scale, int64_t num_spec, double lower_bound, double scale, double onorm_eps,
    bool enable_pdl)
{
    TORCH_CHECK(x_q.is_cuda(), "kimi_k3_kda_mtp_verify_kf: x_q must be a CUDA tensor");
    at::Device const device = x_q.device();
    c10::cuda::CUDAGuard const guard(device);
    for (auto const& [t, name] : {std::pair<at::Tensor const&, char const*>{x_k, "x_k"}, {x_v, "x_v"}, {g, "g"},
             {beta, "beta"}, {onorm_g, "onorm_g"}, {w_q, "w_q"}, {w_k, "w_k"}, {w_v, "w_v"}, {A_log, "A_log"},
             {dt_bias, "dt_bias"}, {onorm_weight, "onorm_weight"}, {ssm_state_indices, "ssm_state_indices"},
             {num_accepted_tokens, "num_accepted_tokens"}, {cs_q, "cs_q"}, {cs_k, "cs_k"}, {cs_v, "cs_v"},
             {recurrent_state, "recurrent_state"}, {k_cache, "k_cache"}, {g_cache, "g_cache"}, {v_cache, "v_cache"},
             {beta_cache, "beta_cache"}, {out, "out"}})
    {
        checkDevice(t, name, device);
    }

    TORCH_CHECK(num_spec == 3 || num_spec == 5, "kimi_k3_kda_mtp_verify_kf: num_spec must be 3 or 5");
    TORCH_CHECK(A_log.dim() == 1, "kimi_k3_kda_mtp_verify_kf: A_log must have shape [H]");
    int64_t const H = A_log.size(0);
    int64_t const D = H * kHeadDim;
    TORCH_CHECK(ssm_state_indices.dim() == 1 && ssm_state_indices.scalar_type() == at::kInt
            && ssm_state_indices.is_contiguous(),
        "kimi_k3_kda_mtp_verify_kf: ssm_state_indices must be a contiguous int32 [B] tensor");
    int64_t const B = ssm_state_indices.size(0);
    TORCH_CHECK(x_q.dim() == 4, "kimi_k3_kda_mtp_verify_kf: x_q must have shape [1, T, H, 128]");
    int64_t const T = x_q.size(1);
    TORCH_CHECK(B >= 1 && T == B * (num_spec + 1),
        "kimi_k3_kda_mtp_verify_kf: packed layout expects T == B * (num_spec + 1) new-token rows");

    tk::KimiK3KdaMtpVerifyKfParams p;
    p.xQRowStride = headRowsStride(x_q, "x_q", T, H);
    p.xKRowStride = headRowsStride(x_k, "x_k", T, H);
    p.xVRowStride = headRowsStride(x_v, "x_v", T, H);
    p.gRowStride = headRowsStride(g, "g", T, H);
    p.onormGRowStride = headRowsStride(onorm_g, "onorm_g", T, H);
    TORCH_CHECK(beta.scalar_type() == at::kBFloat16 && beta.dim() == 3 && beta.size(0) == 1 && beta.size(1) == T
            && beta.size(2) == H && beta.stride(2) == 1,
        "kimi_k3_kda_mtp_verify_kf: beta must be bfloat16 [1, T, H] with contiguous heads");
    p.betaRowStride = beta.stride(1);

    checkContiguousF32(w_q, "w_q", {D, 4});
    checkContiguousF32(w_k, "w_k", {D, 4});
    checkContiguousF32(w_v, "w_v", {D, 4});
    checkContiguousF32(A_log, "A_log", {H});
    checkContiguousF32(dt_bias, "dt_bias", {D});
    checkContiguousF32(onorm_weight, "onorm_weight", {kHeadDim});

    TORCH_CHECK(recurrent_state.dim() == 4 && recurrent_state.size(1) == H && recurrent_state.size(2) == kHeadDim
            && recurrent_state.size(3) == kHeadDim,
        "kimi_k3_kda_mtp_verify_kf: recurrent_state must have shape [pool, H, 128, 128] (V-first)");
    TORCH_CHECK(recurrent_state.scalar_type() == at::kFloat || recurrent_state.scalar_type() == at::kBFloat16,
        "kimi_k3_kda_mtp_verify_kf: recurrent_state must be float32 or bfloat16");
    TORCH_CHECK(recurrent_state.stride(3) == 1 && recurrent_state.stride(2) == kHeadDim
            && recurrent_state.stride(1) == kHeadDim * kHeadDim,
        "kimi_k3_kda_mtp_verify_kf: recurrent_state must be contiguous within each slot");
    int64_t const pool = recurrent_state.size(0);
    TORCH_CHECK(pool <= std::numeric_limits<int32_t>::max(), "kimi_k3_kda_mtp_verify_kf: state pool too large");
    TORCH_CHECK(num_accepted_tokens.dim() == 1 && num_accepted_tokens.scalar_type() == at::kInt
            && num_accepted_tokens.size(0) >= pool,
        "kimi_k3_kda_mtp_verify_kf: num_accepted_tokens must be a slot-indexed int32 tensor covering the state pool");

    checkConvWindow(cs_q, "cs_q", pool, D, num_spec);
    checkConvWindow(cs_k, "cs_k", pool, D, num_spec);
    checkConvWindow(cs_v, "cs_v", pool, D, num_spec);
    checkReplayCache(k_cache, "k_cache", at::kBFloat16, pool, num_spec, D);
    checkReplayCache(g_cache, "g_cache", at::kFloat, pool, num_spec, D);
    checkReplayCache(v_cache, "v_cache", at::kBFloat16, pool, num_spec, D);
    checkReplayCache(beta_cache, "beta_cache", at::kBFloat16, pool, num_spec, H);

    bool const quantize = out_scale.has_value();
    if (quantize)
    {
        at::Tensor const& sc = *out_scale;
        checkDevice(sc, "out_scale", device);
        TORCH_CHECK(out.scalar_type() == at::kFloat8_e4m3fn && out.is_contiguous() && out.dim() == 2 && out.size(0) == T
                && out.size(1) == D,
            "kimi_k3_kda_mtp_verify_kf: MXFP8 out must be a contiguous float8_e4m3fn [T, H * 128] tensor");
        TORCH_CHECK(sc.scalar_type() == at::kByte && sc.is_contiguous() && sc.numel() >= (T + 127) / 128 * 128 * H * 4,
            "kimi_k3_kda_mtp_verify_kf: out_scale must be a contiguous uint8 tensor of ceil(T / 128) * 128 * H * 4 "
            "bytes");
    }
    else
    {
        TORCH_CHECK(out.scalar_type() == at::kBFloat16 && out.is_contiguous() && out.numel() == T * D,
            "kimi_k3_kda_mtp_verify_kf: BF16 out must be a contiguous tensor of T * H * 128 elements");
    }

    p.xQ = x_q.data_ptr();
    p.xK = x_k.data_ptr();
    p.xV = x_v.data_ptr();
    p.g = g.data_ptr();
    p.beta = beta.data_ptr();
    p.onormG = onorm_g.data_ptr();
    p.wQ = w_q.data_ptr<float>();
    p.wK = w_k.data_ptr<float>();
    p.wV = w_v.data_ptr<float>();
    p.logA = A_log.data_ptr<float>();
    p.dtBias = dt_bias.data_ptr<float>();
    p.onormWeight = onorm_weight.data_ptr<float>();
    p.ssmStateIndices = ssm_state_indices.data_ptr<int32_t>();
    p.numAcceptedTokens = num_accepted_tokens.data_ptr<int32_t>();
    p.numAcceptedTokensStride = num_accepted_tokens.stride(0); // the hybrid cache keeps a count block per slot
    p.convQ = cs_q.data_ptr();
    p.convK = cs_k.data_ptr();
    p.convV = cs_v.data_ptr();
    p.convQSlotStride = cs_q.stride(0);
    p.convQPosStride = cs_q.stride(2);
    p.convKSlotStride = cs_k.stride(0);
    p.convKPosStride = cs_k.stride(2);
    p.convVSlotStride = cs_v.stride(0);
    p.convVPosStride = cs_v.stride(2);
    p.state = recurrent_state.data_ptr();
    p.stateIsBf16 = recurrent_state.scalar_type() == at::kBFloat16;
    p.stateSlotStride = recurrent_state.stride(0);
    p.statePoolSize = static_cast<int>(pool);
    p.kCache = k_cache.data_ptr();
    p.gCache = g_cache.data_ptr<float>();
    p.vCache = v_cache.data_ptr();
    p.betaCache = beta_cache.data_ptr();
    p.kCacheSlotStride = k_cache.stride(0);
    p.kCacheRowStride = k_cache.stride(1);
    p.gCacheSlotStride = g_cache.stride(0);
    p.gCacheRowStride = g_cache.stride(1);
    p.vCacheSlotStride = v_cache.stride(0);
    p.vCacheRowStride = v_cache.stride(1);
    p.betaCacheSlotStride = beta_cache.stride(0);
    p.betaCacheRowStride = beta_cache.stride(1);
    p.out = out.data_ptr();
    p.outScale = quantize ? out_scale->data_ptr<uint8_t>() : nullptr;
    p.numRequests = static_cast<int>(B);
    p.numHeads = static_cast<int>(H);
    p.numSpec = static_cast<int>(num_spec);
    p.lowerBound = static_cast<float>(lower_bound);
    p.qScale = static_cast<float>(scale);
    p.onormEps = static_cast<float>(onorm_eps);

    char const* reason = "";
    TORCH_CHECK(
        tk::isKimiK3KdaMtpVerifyKfSupported(p, &reason), "kimi_k3_kda_mtp_verify_kf: unsupported input (", reason, ")");
    tk::invokeKimiK3KdaMtpVerifyKf(p, enable_pdl, at::cuda::getCurrentCUDAStream(device.index()));
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "kimi_k3_kda_mtp_verify_kf(Tensor x_q, Tensor x_k, Tensor x_v, Tensor g, Tensor beta, Tensor onorm_g, "
        "Tensor w_q, Tensor w_k, Tensor w_v, Tensor A_log, Tensor dt_bias, Tensor onorm_weight, "
        "Tensor ssm_state_indices, Tensor num_accepted_tokens, Tensor(a!) cs_q, Tensor(b!) cs_k, Tensor(c!) cs_v, "
        "Tensor(d!) recurrent_state, Tensor(e!) k_cache, Tensor(f!) g_cache, Tensor(g!) v_cache, "
        "Tensor(h!) beta_cache, Tensor(i!) out, Tensor(j!)? out_scale, int num_spec, float lower_bound, "
        "float scale, float onorm_eps, bool enable_pdl) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("kimi_k3_kda_mtp_verify_kf", &tensorrt_llm::torch_ext::kimi_k3_kda_mtp_verify_kf);
}
