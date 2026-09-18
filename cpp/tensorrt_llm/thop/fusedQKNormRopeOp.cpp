/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/fusedQKNormRopeKernel.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include <limits>
#include <torch/extension.h>

#include <tuple>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

// Shared input validation for the in-place and out-of-place operators. Returns num_tokens.
int64_t validateFusedQKNormRopeInputs(torch::Tensor const& qkv, torch::Tensor const& position_ids,
    torch::Tensor const& q_weight, torch::Tensor const& k_weight, int64_t num_heads_q, int64_t num_heads_k,
    int64_t num_heads_v, int64_t head_dim, bool use_mrope)
{
    TORCH_CHECK(qkv.dim() == 2, "QKV tensor must be 2D: [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]");
    // Plain RoPE: position_ids is 1D [num_tokens]. Interleaved mRoPE: 2D [3, num_tokens].
    TORCH_CHECK(position_ids.dim() == 1 || (position_ids.dim() == 2 && position_ids.size(0) == 3),
        "Position IDs must be 1D [num_tokens] (plain RoPE) or 2D [3, num_tokens] (mRoPE)");
    TORCH_CHECK(!use_mrope || position_ids.dim() == 2, "use_mrope requires 2D [3, num_tokens] position_ids");
    TORCH_CHECK(q_weight.dim() == 1, "Query weights must be 1D: [head_dim]");
    TORCH_CHECK(k_weight.dim() == 1, "Key weights must be 1D: [head_dim]");
    TORCH_CHECK(q_weight.size(0) == head_dim, "Query weights size must match head dimension");
    TORCH_CHECK(k_weight.size(0) == head_dim, "Key weights size must match head dimension");

    CHECK_INPUT(qkv, torch::kBFloat16);
    CHECK_INPUT(position_ids, torch::kInt32);
    CHECK_INPUT(q_weight, torch::kBFloat16);
    CHECK_INPUT(k_weight, torch::kBFloat16);

    int64_t num_tokens = qkv.size(0);
    TORCH_CHECK(position_ids.size(-1) == num_tokens, "Number of tokens in position_ids must match QKV");

    // The kernel narrows these to int, so reject anything that would not survive it.
    TORCH_CHECK(num_heads_q >= 0 && num_heads_k >= 0 && num_heads_v >= 0 && head_dim > 0,
        "Head counts must be non-negative and head_dim must be positive");
    int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
    TORCH_CHECK(
        num_tokens <= std::numeric_limits<int>::max() && total_heads * head_dim <= std::numeric_limits<int>::max(),
        "QKV dimensions exceed the kernel's supported range");
    TORCH_CHECK(
        qkv.size(1) == total_heads * head_dim, "QKV tensor size must match total number of heads and head dimension");

    return num_tokens;
}

void checkMinimaxM3HndKVPool(torch::Tensor const& kvCache, int64_t numHeads, int64_t headDim)
{
    TORCH_CHECK(kvCache.is_cuda(), "kv_cache must be a CUDA tensor");
    TORCH_CHECK(kvCache.scalar_type() == at::ScalarType::Float8_e4m3fn, "kv_cache must use torch.float8_e4m3fn");
    TORCH_CHECK(kvCache.dim() == 5, "kv_cache must be HND [num_pages, 2, num_heads, page_size, head_dim]");
    TORCH_CHECK(kvCache.size(0) > 0 && kvCache.size(3) > 0, "kv_cache must have positive num_pages and page_size");
    TORCH_CHECK(kvCache.size(1) == 2, "kv_cache plane dimension must contain K and V");
    TORCH_CHECK(kvCache.size(2) == numHeads, "kv_cache num_heads mismatch");
    TORCH_CHECK(kvCache.size(4) == headDim, "kv_cache head_dim mismatch");
    TORCH_CHECK(kvCache.stride(4) == 1 && kvCache.stride(3) == headDim,
        "kv_cache must have contiguous head_dim rows in HND layout");
    TORCH_CHECK(kvCache.stride(2) == kvCache.size(3) * kvCache.stride(3),
        "kv_cache must have contiguous [page_size, head_dim] blocks in HND layout");
    TORCH_CHECK(kvCache.stride(1) >= kvCache.size(2) * kvCache.stride(2), "kv_cache K and V planes must not overlap");
    TORCH_CHECK(kvCache.stride(0) >= kvCache.size(1) * kvCache.stride(1),
        "kv_cache page stride must not overlap adjacent HND pages");
    TORCH_CHECK(kvCache.stride(0) % 4 == 0 && kvCache.stride(1) % 4 == 0,
        "kv_cache page and plane strides must preserve 32-bit FP8 store alignment");
}

void checkMinimaxM3Int32LaunchGeometry(int64_t numTokens, int64_t slotsPerToken)
{
    TORCH_CHECK(numTokens <= std::numeric_limits<int>::max(), "MiniMax-M3 producer num_tokens exceeds int32");
    TORCH_CHECK(slotsPerToken > 0 && slotsPerToken <= std::numeric_limits<int>::max(),
        "MiniMax-M3 producer head geometry exceeds int32");
    TORCH_CHECK(numTokens == 0 || slotsPerToken <= std::numeric_limits<int>::max() / numTokens,
        "MiniMax-M3 producer launch geometry exceeds int32");
}

} // namespace

// Function for fused QK Norm and RoPE
// This operator applies RMS normalization and RoPE to Q and K tensors in a single CUDA kernel.
// The OP performs operations in-place on the input qkv tensor.
void fused_qk_norm_rope(
    torch::Tensor& qkv,          // Combined QKV tensor [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]
    int64_t num_heads_q,         // Number of query heads
    int64_t num_heads_k,         // Number of key heads
    int64_t num_heads_v,         // Number of value heads
    int64_t head_dim,            // Dimension per head
    int64_t rotary_dim,          // Dimension for RoPE
    double eps,                  // Epsilon for RMS normalization
    torch::Tensor& q_weight,     // RMSNorm weights for query [head_dim]
    torch::Tensor& k_weight,     // RMSNorm weights for key [head_dim]
    double base,                 // Base for RoPE computation
    bool is_neox,                // Whether RoPE is applied in Neox style
    torch::Tensor& position_ids, // Position IDs for RoPE [num_tokens]
    // parameters for yarn
    double factor, // factor in rope_scaling in config.json. When it is not 1.0, it means the model is using yarn.
    double low,    // threshold for high frequency
    double high,   // threshold for low frequency
    double attention_factor, // attention_factor applied on cos and sin
    bool is_qk_norm,         // Whether to apply QK norm
    bool use_gemma,          // Whether QK norm uses Gemma-style RMSNorm (scale by (1 + weight))
    bool use_mrope,          // Whether to use interleaved mRoPE position selection
    int64_t mrope_section1,  // mrope_section[1] (height); ignored when use_mrope is false
    int64_t mrope_section2   // mrope_section[2] (width)
)
{
    int64_t num_tokens = validateFusedQKNormRopeInputs(
        qkv, position_ids, q_weight, k_weight, num_heads_q, num_heads_k, num_heads_v, head_dim, use_mrope);

    auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

    tensorrt_llm::kernels::launchFusedQKNormRope(reinterpret_cast<__nv_bfloat16*>(qkv.data_ptr()),
        static_cast<int>(num_tokens), static_cast<int>(num_heads_q), static_cast<int>(num_heads_k),
        static_cast<int>(num_heads_v), static_cast<int>(head_dim), static_cast<int>(rotary_dim),
        static_cast<float>(eps), reinterpret_cast<__nv_bfloat16*>(q_weight.data_ptr()),
        reinterpret_cast<__nv_bfloat16*>(k_weight.data_ptr()), static_cast<float>(base),
        !is_neox, // interleave
        reinterpret_cast<int const*>(position_ids.data_ptr()), static_cast<float>(factor), static_cast<float>(low),
        static_cast<float>(high), static_cast<float>(attention_factor), stream, is_qk_norm, use_gemma, use_mrope,
        static_cast<int>(mrope_section1), static_cast<int>(mrope_section2));
}

// Out-of-place FP8 variant of fused_qk_norm_rope: applies RMSNorm + RoPE to Q/K,
// copy-casts V, and returns a new FP8 (E4M3) tensor of the same shape.
torch::Tensor fused_qk_norm_rope_to_fp8(torch::Tensor const& qkv, // [num_tokens, (num_q+num_k+num_v)*head_dim] BF16
    int64_t num_heads_q, int64_t num_heads_k, int64_t num_heads_v, int64_t head_dim, int64_t rotary_dim, double eps,
    torch::Tensor const& q_weight, torch::Tensor const& k_weight, double base, bool is_neox,
    torch::Tensor const& position_ids, double factor, double low, double high, double attention_factor, bool is_qk_norm,
    bool use_gemma, bool use_mrope, int64_t mrope_section1, int64_t mrope_section2)
{
    int64_t num_tokens = validateFusedQKNormRopeInputs(
        qkv, position_ids, q_weight, k_weight, num_heads_q, num_heads_k, num_heads_v, head_dim, use_mrope);

    int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
    auto out = torch::empty({num_tokens, total_heads * head_dim}, qkv.options().dtype(torch::kFloat8_e4m3fn));

    auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());

    tensorrt_llm::kernels::launchFusedQKNormRopeToFp8(qkv.data_ptr(), out.data_ptr(), static_cast<int>(num_tokens),
        static_cast<int>(num_heads_q), static_cast<int>(num_heads_k), static_cast<int>(num_heads_v),
        static_cast<int>(head_dim), static_cast<int>(rotary_dim), static_cast<float>(eps), q_weight.data_ptr(),
        k_weight.data_ptr(), static_cast<float>(base), !is_neox, reinterpret_cast<int const*>(position_ids.data_ptr()),
        static_cast<float>(factor), static_cast<float>(low), static_cast<float>(high),
        static_cast<float>(attention_factor), stream, is_qk_norm, use_gemma, use_mrope,
        static_cast<int>(mrope_section1), static_cast<int>(mrope_section2));

    return out;
}

// Meta (fake) implementation for torch.compile / tracing: only shape+dtype.
torch::Tensor fused_qk_norm_rope_to_fp8_meta(torch::Tensor const& qkv, int64_t num_heads_q, int64_t num_heads_k,
    int64_t num_heads_v, int64_t head_dim, int64_t /*rotary_dim*/, double /*eps*/, torch::Tensor const& /*q_weight*/,
    torch::Tensor const& /*k_weight*/, double /*base*/, bool /*is_neox*/, torch::Tensor const& /*position_ids*/,
    double /*factor*/, double /*low*/, double /*high*/, double /*attention_factor*/, bool /*is_qk_norm*/,
    bool /*use_gemma*/, bool /*use_mrope*/, int64_t /*mrope_section1*/, int64_t /*mrope_section2*/)
{
    int64_t num_tokens = qkv.size(0);
    int64_t total_heads = num_heads_q + num_heads_k + num_heads_v;
    return torch::empty({num_tokens, total_heads * head_dim}, qkv.options().dtype(torch::kFloat8_e4m3fn));
}

torch::Tensor minimaxM3Fp8QKNormRopeKVInsert(torch::Tensor const& qkv, torch::Tensor& kvCache,
    torch::Tensor const& outCacheLoc, int64_t numHeadsQ, int64_t numHeadsK, int64_t numHeadsV, int64_t headDim,
    int64_t rotaryDim, double eps, torch::Tensor const& qWeight, torch::Tensor const& kWeight, double base, bool isNeox,
    torch::Tensor const& positionIds)
{
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kRotaryDim = 64;
    constexpr int64_t kPageSize = 128;
    TORCH_CHECK(numHeadsQ > 0, "MiniMax-M3 FP8 main Q/K/V producer requires num_heads_q > 0");
    TORCH_CHECK(numHeadsK > 0 && numHeadsV > 0, "MiniMax-M3 FP8 main Q/K/V producer requires K and V heads");
    TORCH_CHECK(numHeadsK == numHeadsV, "MiniMax-M3 FP8 main Q/K/V producer requires equal K and V head counts");
    TORCH_CHECK(headDim == kHeadDim, "MiniMax-M3 FP8 main Q/K/V producer requires head_dim=128");
    TORCH_CHECK(rotaryDim == kRotaryDim, "MiniMax-M3 FP8 main Q/K/V producer requires rotary_dim=64");
    TORCH_CHECK(isNeox, "MiniMax-M3 FP8 main Q/K/V producer requires NeoX RoPE");
    TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "MiniMax-M3 FP8 main Q/K/V producer requires finite eps > 0");
    TORCH_CHECK(std::isfinite(base) && base > 0.0, "MiniMax-M3 FP8 main Q/K/V producer requires finite RoPE base > 0");
    auto const epsFloat = static_cast<float>(eps);
    auto const baseFloat = static_cast<float>(base);
    TORCH_CHECK(std::isfinite(epsFloat) && epsFloat > 0.0F,
        "MiniMax-M3 FP8 main Q/K/V producer eps must remain finite and positive in float32");
    TORCH_CHECK(std::isfinite(baseFloat) && baseFloat > 0.0F,
        "MiniMax-M3 FP8 main Q/K/V producer RoPE base must remain finite and positive in float32");

    TORCH_CHECK(qkv.dim() == 2, "QKV tensor must be 2D: [num_tokens, (num_heads_q+num_heads_k+num_heads_v)*head_dim]");
    TORCH_CHECK(outCacheLoc.dim() == 1, "out_cache_loc must be one-dimensional");
    TORCH_CHECK(positionIds.dim() == 1, "position_ids must be one-dimensional");
    TORCH_CHECK(qWeight.dim() == 1 && kWeight.dim() == 1, "Q/K norm weights must be one-dimensional");

    CHECK_INPUT(qkv, torch::kBFloat16);
    CHECK_INPUT(outCacheLoc, torch::kInt32);
    CHECK_INPUT(positionIds, torch::kInt32);
    CHECK_INPUT(qWeight, torch::kBFloat16);
    CHECK_INPUT(kWeight, torch::kBFloat16);
    checkMinimaxM3HndKVPool(kvCache, numHeadsK, headDim);
    TORCH_CHECK(kvCache.size(3) == kPageSize, "MiniMax-M3 FP8 main Q/K/V producer requires page_size=128");

    int64_t const numTokens = qkv.size(0);
    int64_t const totalHeads = numHeadsQ + numHeadsK + numHeadsV;
    checkMinimaxM3Int32LaunchGeometry(numTokens, totalHeads);
    TORCH_CHECK(qkv.size(1) == totalHeads * headDim,
        "QKV tensor width must equal (num_heads_q + num_heads_k + num_heads_v) * head_dim");
    TORCH_CHECK(outCacheLoc.numel() >= numTokens, "out_cache_loc is shorter than num_tokens");
    TORCH_CHECK(positionIds.numel() == numTokens, "position_ids length must equal num_tokens");
    TORCH_CHECK(qWeight.numel() == headDim && kWeight.numel() == headDim, "Q/K norm weight width must equal head_dim");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(qkv.data_ptr()) % 8 == 0,
        "QKV input must start at an 8-byte-aligned address for vectorized BF16 loads");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(kvCache.data_ptr()) % 4 == 0,
        "K/V cache must start at a 4-byte-aligned address for packed E4M3 stores");
    TORCH_CHECK(qkv.get_device() == kvCache.get_device() && qkv.get_device() == outCacheLoc.get_device()
            && qkv.get_device() == positionIds.get_device() && qkv.get_device() == qWeight.get_device()
            && qkv.get_device() == kWeight.get_device(),
        "All MiniMax-M3 FP8 main Q/K/V producer tensors must be on the same CUDA device");

    auto qOut = torch::empty({numTokens, numHeadsQ, headDim}, qkv.options().dtype(at::ScalarType::Float8_e4m3fn));
    if (numTokens == 0)
    {
        return qOut;
    }

    auto stream = at::cuda::getCurrentCUDAStream(qkv.get_device());
    tensorrt_llm::kernels::launchMinimaxM3Fp8QKNormRopeKVInsert(qkv.data_ptr(), qOut.data_ptr(), kvCache.data_ptr(),
        outCacheLoc.data_ptr<int>(), kvCache.stride(0), kvCache.stride(1), kvCache.stride(2), kvCache.stride(3),
        kvCache.size(0), static_cast<int>(kvCache.size(3)), static_cast<int>(numTokens), static_cast<int>(numHeadsQ),
        static_cast<int>(numHeadsK), static_cast<int>(numHeadsV), static_cast<int>(headDim),
        static_cast<int>(rotaryDim), epsFloat, qWeight.data_ptr(), kWeight.data_ptr(), baseFloat,
        positionIds.data_ptr<int>(), stream);
    return qOut;
}

torch::Tensor minimaxM3Fp8QKNormRopeKVInsertMeta(torch::Tensor const& qkv, torch::Tensor& /*kvCache*/,
    torch::Tensor const& /*outCacheLoc*/, int64_t numHeadsQ, int64_t /*numHeadsK*/, int64_t /*numHeadsV*/,
    int64_t headDim, int64_t /*rotaryDim*/, double /*eps*/, torch::Tensor const& /*qWeight*/,
    torch::Tensor const& /*kWeight*/, double /*base*/, bool /*isNeox*/, torch::Tensor const& /*positionIds*/)
{
    return torch::empty({qkv.size(0), numHeadsQ, headDim}, qkv.options().dtype(at::ScalarType::Float8_e4m3fn));
}

std::tuple<torch::Tensor, torch::Tensor> minimaxM3Fp8QKVIndexerNormRopeKVInsert(torch::Tensor const& packed,
    torch::Tensor& kvCache, torch::Tensor& indexKCache, torch::Tensor const& outCacheLoc, int64_t numHeadsQ,
    int64_t numHeadsKV, int64_t numHeadsIndex, int64_t headDim, int64_t rotaryDim, double eps,
    torch::Tensor const& qWeight, torch::Tensor const& kWeight, torch::Tensor const& indexQWeight,
    torch::Tensor const& indexKWeight, torch::Tensor const& rotaryCosSin, torch::Tensor const& positionIds)
{
    constexpr int64_t kHeadDim = 128;
    constexpr int64_t kRotaryDim = 64;
    constexpr int64_t kPageSize = 128;
    TORCH_CHECK(numHeadsQ > 0 && numHeadsKV > 0 && numHeadsIndex > 0,
        "MiniMax-M3 horizontal producer requires Q, KV, and index heads");
    TORCH_CHECK(numHeadsKV == numHeadsIndex, "MiniMax-M3 horizontal producer requires index heads to equal KV heads");
    TORCH_CHECK(headDim == kHeadDim, "MiniMax-M3 horizontal producer requires head_dim=128");
    TORCH_CHECK(rotaryDim == kRotaryDim, "MiniMax-M3 horizontal producer requires rotary_dim=64");
    TORCH_CHECK(std::isfinite(eps) && eps > 0.0, "MiniMax-M3 horizontal producer requires finite eps > 0");
    auto const epsFloat = static_cast<float>(eps);
    TORCH_CHECK(std::isfinite(epsFloat) && epsFloat > 0.0F,
        "MiniMax-M3 horizontal producer eps must remain finite and positive in float32");

    TORCH_CHECK(packed.dim() == 2, "Packed QKV+index tensor must be two-dimensional");
    TORCH_CHECK(outCacheLoc.dim() == 1, "out_cache_loc must be one-dimensional");
    TORCH_CHECK(positionIds.dim() == 1, "position_ids must be one-dimensional");
    CHECK_INPUT(packed, torch::kBFloat16);
    CHECK_INPUT(outCacheLoc, torch::kInt32);
    CHECK_INPUT(positionIds, torch::kInt32);
    CHECK_INPUT(qWeight, torch::kBFloat16);
    CHECK_INPUT(kWeight, torch::kBFloat16);
    CHECK_INPUT(indexQWeight, torch::kBFloat16);
    CHECK_INPUT(indexKWeight, torch::kBFloat16);
    CHECK_INPUT(rotaryCosSin, torch::kFloat32);
    checkMinimaxM3HndKVPool(kvCache, numHeadsKV, headDim);
    TORCH_CHECK(kvCache.size(3) == kPageSize, "MiniMax-M3 horizontal producer requires page_size=128");
    TORCH_CHECK(indexKCache.is_cuda() && indexKCache.scalar_type() == at::ScalarType::Float8_e4m3fn,
        "Index-K cache must be CUDA torch.float8_e4m3fn");
    TORCH_CHECK(indexKCache.dim() == 4 && indexKCache.size(1) == 1 && indexKCache.size(2) == kPageSize
            && indexKCache.size(3) == kHeadDim,
        "Index-K cache must be HND [num_pages, 1, 128, 128]");
    TORCH_CHECK(indexKCache.stride(3) == 1 && indexKCache.stride(2) == kHeadDim,
        "Index-K cache must have contiguous token rows");
    TORCH_CHECK(indexKCache.stride(1) >= indexKCache.size(2) * indexKCache.stride(2)
            && indexKCache.stride(0) >= indexKCache.size(1) * indexKCache.stride(1),
        "Index-K cache pages must not overlap");
    TORCH_CHECK(indexKCache.stride(0) % 4 == 0 && indexKCache.stride(1) % 4 == 0,
        "Index-K cache page/head strides must preserve 32-bit FP8 store alignment");
    TORCH_CHECK(
        indexKCache.size(0) == kvCache.size(0), "Main K/V and index-K caches must contain the same number of pages");
    TORCH_CHECK(rotaryCosSin.dim() == 3 && rotaryCosSin.size(1) == 2 && rotaryCosSin.size(2) == kRotaryDim / 2,
        "rotary_cos_sin must be [max_positions, 2, rotary_dim/2]");

    int64_t const numTokens = packed.size(0);
    int64_t const totalHeads = numHeadsQ + 2 * numHeadsKV + numHeadsIndex + 1;
    checkMinimaxM3Int32LaunchGeometry(numTokens, totalHeads);
    TORCH_CHECK(
        packed.size(1) == totalHeads * headDim, "Packed tensor width must equal (Q + 2*KV + index-Q + 1) * head_dim");
    TORCH_CHECK(outCacheLoc.numel() >= numTokens, "out_cache_loc is shorter than num_tokens");
    TORCH_CHECK(positionIds.numel() == numTokens, "position_ids length must equal num_tokens");
    TORCH_CHECK(qWeight.dim() == 1 && kWeight.dim() == 1 && indexQWeight.dim() == 1 && indexKWeight.dim() == 1,
        "All norm weights must be one-dimensional");
    TORCH_CHECK(qWeight.numel() == headDim && kWeight.numel() == headDim && indexQWeight.numel() == headDim
            && indexKWeight.numel() == headDim,
        "All norm weights must contain head_dim elements");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(packed.data_ptr()) % 8 == 0,
        "Packed input must start at an 8-byte-aligned address for vectorized BF16 loads");
    TORCH_CHECK(reinterpret_cast<uintptr_t>(kvCache.data_ptr()) % 4 == 0
            && reinterpret_cast<uintptr_t>(indexKCache.data_ptr()) % 4 == 0,
        "Paged caches must start at 4-byte-aligned addresses for packed E4M3 stores");
    TORCH_CHECK(packed.get_device() == kvCache.get_device() && packed.get_device() == indexKCache.get_device()
            && packed.get_device() == outCacheLoc.get_device() && packed.get_device() == positionIds.get_device()
            && packed.get_device() == qWeight.get_device() && packed.get_device() == kWeight.get_device()
            && packed.get_device() == indexQWeight.get_device() && packed.get_device() == indexKWeight.get_device()
            && packed.get_device() == rotaryCosSin.get_device(),
        "All MiniMax-M3 horizontal producer tensors must be on the same CUDA device");

    auto qOut = torch::empty({numTokens, numHeadsQ, headDim}, packed.options().dtype(at::ScalarType::Float8_e4m3fn));
    auto indexQOut
        = torch::empty({numTokens, numHeadsIndex, headDim}, packed.options().dtype(at::ScalarType::Float8_e4m3fn));
    if (numTokens == 0)
    {
        return {qOut, indexQOut};
    }

    auto stream = at::cuda::getCurrentCUDAStream(packed.get_device());
    tensorrt_llm::kernels::launchMinimaxM3Fp8QKVIndexerNormRopeKVInsert(packed.data_ptr(), qOut.data_ptr(),
        indexQOut.data_ptr(), kvCache.data_ptr(), indexKCache.data_ptr(), outCacheLoc.data_ptr<int>(),
        kvCache.stride(0), kvCache.stride(1), kvCache.stride(2), kvCache.stride(3), indexKCache.stride(0),
        indexKCache.stride(2), kvCache.size(0), static_cast<int>(kvCache.size(3)), static_cast<int>(numTokens),
        static_cast<int>(numHeadsQ), static_cast<int>(numHeadsKV), static_cast<int>(numHeadsIndex),
        static_cast<int>(headDim), static_cast<int>(rotaryDim), epsFloat, qWeight.data_ptr(), kWeight.data_ptr(),
        indexQWeight.data_ptr(), indexKWeight.data_ptr(), rotaryCosSin.data_ptr<float>(), positionIds.data_ptr<int>(),
        stream);
    return {qOut, indexQOut};
}

std::tuple<torch::Tensor, torch::Tensor> minimaxM3Fp8QKVIndexerNormRopeKVInsertMeta(torch::Tensor const& packed,
    torch::Tensor& /*kvCache*/, torch::Tensor& /*indexKCache*/, torch::Tensor const& /*outCacheLoc*/, int64_t numHeadsQ,
    int64_t /*numHeadsKV*/, int64_t numHeadsIndex, int64_t headDim, int64_t /*rotaryDim*/, double /*eps*/,
    torch::Tensor const& /*qWeight*/, torch::Tensor const& /*kWeight*/, torch::Tensor const& /*indexQWeight*/,
    torch::Tensor const& /*indexKWeight*/, torch::Tensor const& /*rotaryCosSin*/, torch::Tensor const& /*positionIds*/)
{
    auto options = packed.options().dtype(at::ScalarType::Float8_e4m3fn);
    return {
        torch::empty({packed.size(0), numHeadsQ, headDim}, options),
        torch::empty({packed.size(0), numHeadsIndex, headDim}, options),
    };
}

// Register the PyTorch operators
TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "fused_qk_norm_rope(Tensor(a!) qkv, int num_heads_q, int num_heads_k, int num_heads_v, int head_dim, int "
        "rotary_dim, float "
        "eps, Tensor q_weight, Tensor k_weight, float base, bool is_neox, Tensor position_ids, float factor, float "
        "low, float high, float attention_factor, bool is_qk_norm, bool use_gemma, bool use_mrope, int "
        "mrope_section1, int mrope_section2) -> ()");
    m.def(
        "fused_qk_norm_rope_to_fp8(Tensor qkv, int num_heads_q, int num_heads_k, int num_heads_v, int head_dim, int "
        "rotary_dim, float eps, Tensor q_weight, Tensor k_weight, float base, bool is_neox, Tensor position_ids, float "
        "factor, float low, float high, float attention_factor, bool is_qk_norm, bool use_gemma, bool use_mrope, int "
        "mrope_section1, int mrope_section2) -> Tensor");
    m.def(
        "minimax_m3_fp8_qk_norm_rope_kv_insert(Tensor qkv, Tensor(a!) kv_cache, Tensor out_cache_loc, int "
        "num_heads_q, int num_heads_k, int num_heads_v, int head_dim, int rotary_dim, float eps, Tensor q_weight, "
        "Tensor k_weight, float base, bool is_neox, Tensor position_ids) -> Tensor");
    m.def(
        "minimax_m3_fp8_qkv_indexer_norm_rope_kv_insert(Tensor packed, Tensor(a!) kv_cache, Tensor(b!) "
        "index_k_cache, Tensor out_cache_loc, int num_heads_q, int num_heads_kv, int num_heads_index, int head_dim, "
        "int rotary_dim, float eps, Tensor q_weight, Tensor k_weight, Tensor index_q_weight, Tensor index_k_weight, "
        "Tensor rotary_cos_sin, Tensor position_ids) -> (Tensor, Tensor)");
}

// Register the CUDA implementation
TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_qk_norm_rope", &fused_qk_norm_rope);
    m.impl("fused_qk_norm_rope_to_fp8", &fused_qk_norm_rope_to_fp8);
    m.impl("minimax_m3_fp8_qk_norm_rope_kv_insert", &minimaxM3Fp8QKNormRopeKVInsert);
    m.impl("minimax_m3_fp8_qkv_indexer_norm_rope_kv_insert", &minimaxM3Fp8QKVIndexerNormRopeKVInsert);
}

// Register the Meta implementation (shape/dtype inference for torch.compile).
TORCH_LIBRARY_IMPL(trtllm, Meta, m)
{
    m.impl("fused_qk_norm_rope_to_fp8", &fused_qk_norm_rope_to_fp8_meta);
    m.impl("minimax_m3_fp8_qk_norm_rope_kv_insert", &minimaxM3Fp8QKNormRopeKVInsertMeta);
    m.impl("minimax_m3_fp8_qkv_indexer_norm_rope_kv_insert", &minimaxM3Fp8QKVIndexerNormRopeKVInsertMeta);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
