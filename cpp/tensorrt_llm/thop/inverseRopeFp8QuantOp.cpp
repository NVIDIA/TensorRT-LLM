/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include "tensorrt_llm/kernels/inverseRopeFp8QuantKernel.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{
constexpr int kRopeDim = 64;
constexpr int kQuantGroupSize = 128;

inline int64_t pad_up(int64_t x, int64_t a)
{
    return (x + a - 1) / a * a;
}
} // namespace

// Fused inverse-RoPE + 1x128 block-scaled FP8 quant for DeepSeek-V4
// absorption-mode attention output. Replaces the Triton implementation
// previously exposed via `trtllm::fused_inv_rope_fp8_quant_vllm_port`.
//
// Hardcoded constraints (DSv4 layout):
//   * quant_group_size == 128
//   * rope_dim == 64, half_rope == 32, NEOX layout (is_neox == true)
//   * head_dim == nope_dim + rope_dim, head_dim % 128 == 0
//   * head_dim in {128, 256, 384, 512}; DSv4-Flash/Pro production uses 512
//   * num_heads == n_groups * heads_per_group
//   * positions int32 or int64; cos_sin_cache fp32 with stride-per-position
//     matching kCsStride = 2 * kHalfRope = 64 fp32.
//
// The op allocates fp8_buf and scale_buf to match the layout consumed by
// `cute_dsl_fp8_bmm_blackwell`:
//   * fp8_buf:   [n_groups, num_tokens, heads_per_group * head_dim] e4m3
//   * scale_buf: [n_groups, heads_per_group * head_dim/quant_group_size,
//                 pad_up(num_tokens, 4)] fp32
std::tuple<torch::Tensor, torch::Tensor> fusedInvRopeFp8Quant( //
    torch::Tensor o,                                           //
    torch::Tensor positions,                                   //
    torch::Tensor cos_sin_cache,                               //
    int64_t n_groups,                                          //
    int64_t heads_per_group,                                   //
    int64_t nope_dim,                                          //
    int64_t rope_dim,                                          //
    int64_t quant_group_size,                                  //
    bool is_neox)
{
    TORCH_CHECK(o.is_cuda() && o.scalar_type() == torch::kBFloat16, "o must be CUDA bf16");
    TORCH_CHECK(o.is_contiguous(), "o must be contiguous");
    TORCH_CHECK(o.dim() == 3, "o must be 3D [num_tokens, num_heads, head_dim]");
    TORCH_CHECK(positions.is_cuda(), "positions must be on CUDA");
    TORCH_CHECK(positions.scalar_type() == torch::kInt64 || positions.scalar_type() == torch::kInt32,
        "positions must be int32 or int64");
    TORCH_CHECK(
        cos_sin_cache.is_cuda() && cos_sin_cache.scalar_type() == torch::kFloat32, "cos_sin_cache must be CUDA fp32");

    int64_t const num_tokens = o.size(0);
    int64_t const num_heads = o.size(1);
    int64_t const head_dim = o.size(2);

    TORCH_CHECK(quant_group_size == kQuantGroupSize, "Only quant_group_size=", kQuantGroupSize, " supported, got ",
        quant_group_size);
    TORCH_CHECK(rope_dim == kRopeDim, "Only rope_dim=", kRopeDim, " supported, got ", rope_dim);
    TORCH_CHECK(head_dim == nope_dim + rope_dim, "head_dim(", head_dim, ") != nope_dim(", nope_dim, ") + rope_dim(",
        rope_dim, ")");
    TORCH_CHECK(
        head_dim % kQuantGroupSize == 0, "head_dim must be a multiple of ", kQuantGroupSize, ", got ", head_dim);
    int64_t const chunks_per_head = head_dim / kQuantGroupSize;
    TORCH_CHECK(chunks_per_head >= 1 && chunks_per_head <= 4, "head_dim/", kQuantGroupSize, " must be in [1, 4], got ",
        chunks_per_head);
    // Layout requirement: rope lives in the second half of the last quant
    // chunk, so nope_dim % quant_group_size == quant_group_size - rope_dim.
    int64_t const expected_nope_mod = kQuantGroupSize - rope_dim;
    TORCH_CHECK(nope_dim % kQuantGroupSize == expected_nope_mod, "Layout requires nope_dim % ", kQuantGroupSize,
        " == ", expected_nope_mod, ", got ", (nope_dim % kQuantGroupSize));
    TORCH_CHECK(num_heads == n_groups * heads_per_group, "num_heads(", num_heads, ") != n_groups(", n_groups,
        ") * heads_per_group(", heads_per_group, ")");
    // Both NEOX (DSv4 default) and interleaved / GPT-J (used by the
    // sparse-MLA absorption-mode test path) are supported via the
    // IS_NEOX template parameter inside the kernel.

    int64_t const d = heads_per_group * head_dim;
    int64_t const num_scale_blocks = d / quant_group_size;
    int64_t const tma_aligned_T = pad_up(num_tokens, 4);

    // FP8 output: [n_groups, num_tokens, d] contiguous — matches the BMM
    // consumer's expected (G, T, K) stride (T*K, K, 1).
    auto fp8_buf = torch::empty(
        {n_groups, num_tokens, d}, torch::TensorOptions().dtype(torch::kFloat8_e4m3fn).device(o.device()));
    // Scale buffer: [n_groups, num_scale_blocks, pad_up(num_tokens, 4)] fp32.
    auto scale_buf = torch::empty(
        {n_groups, num_scale_blocks, tma_aligned_T}, torch::TensorOptions().dtype(torch::kFloat32).device(o.device()));

    TORCH_CHECK(static_cast<int64_t>(static_cast<int>(num_tokens)) == num_tokens, "num_tokens exceeds int range");
    TORCH_CHECK(
        static_cast<int64_t>(static_cast<int>(tma_aligned_T)) == tma_aligned_T, "scale_buf_m exceeds int range");

    // The kernel reads `positions` as int64. If the caller passed int32 we
    // need an upcast — cheap, single dim.
    torch::Tensor positions_i64 = positions;
    if (positions.scalar_type() == torch::kInt32)
    {
        positions_i64 = positions.to(torch::kInt64);
    }

    auto stream = at::cuda::getCurrentCUDAStream(o.get_device());

    // Strides in element units. Input layout is [T, num_heads, head_dim]
    // flat; output fp8/scale buffers are [n_groups, ...] with G outermost.
    int const o_stride_token = static_cast<int>(o.stride(0));
    int const o_stride_head = static_cast<int>(o.stride(1));
    int const fp8_stride_group = static_cast<int>(fp8_buf.stride(0));
    int const fp8_stride_token = static_cast<int>(fp8_buf.stride(1));
    int const scale_stride_group = static_cast<int>(scale_buf.stride(0));
    int const scale_stride_k = static_cast<int>(scale_buf.stride(1));

    tensorrt_llm::kernels::invokeInverseRopeFp8Quant(o.data_ptr(),                                    //
        positions_i64.data_ptr(), cos_sin_cache.data_ptr(), fp8_buf.data_ptr(), scale_buf.data_ptr(), //
        static_cast<int>(num_tokens), static_cast<int>(num_heads), static_cast<int>(heads_per_group), //
        static_cast<int>(chunks_per_head), is_neox, static_cast<int>(tma_aligned_T),                  //
        o_stride_token, o_stride_head, fp8_stride_group, fp8_stride_token, scale_stride_group,        //
        scale_stride_k, stream);

    return {fp8_buf, scale_buf};
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    // Public name kept as `fused_inv_rope_fp8_quant_vllm_port` so existing
    // call sites in attention.py and prior tests do not need to change.
    // The original Triton implementation has been replaced; the schema is
    // unchanged.
    m.def(
        "fused_inv_rope_fp8_quant_vllm_port("
        "Tensor o, Tensor positions, Tensor cos_sin_cache, "
        "int n_groups, int heads_per_group, int nope_dim, int rope_dim, "
        "int quant_group_size, bool is_neox) -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("fused_inv_rope_fp8_quant_vllm_port", &tensorrt_llm::torch_ext::fusedInvRopeFp8Quant);
}
