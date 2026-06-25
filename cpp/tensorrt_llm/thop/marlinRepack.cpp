/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/kernels/marlin/marlin_nvfp4.h"
#include "tensorrt_llm/thop/thUtils.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

using torch::Tensor;

// Marlin tile constants (must match marlin.cuh)
static constexpr int kMarlinTileSize = 16;
static constexpr int kMarlinTileKSize = kMarlinTileSize;
static constexpr int kMarlinTileNSize = kMarlinTileKSize * 4;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// Repack quantized weights from row-major to Marlin tiled format.
//
// b_q_weight: [K/pack_factor, N] int32 row-major packed weights
// perm:       [K] int32 permutation (empty for no-perm)
// size_k:     reduction dimension K
// size_n:     output dimension N
// num_bits:   quantization bits (4 for FP4)
Tensor gptq_marlin_repack(
    Tensor& b_q_weight, Tensor& perm, int64_t size_k, int64_t size_n, int64_t num_bits, bool is_a_8bit = false)
{
    TORCH_CHECK(
        size_k % kMarlinTileKSize == 0, "size_k = ", size_k, " not divisible by tile_k_size = ", kMarlinTileKSize);
    TORCH_CHECK(
        size_n % kMarlinTileNSize == 0, "size_n = ", size_n, " not divisible by tile_n_size = ", kMarlinTileNSize);
    TORCH_CHECK(num_bits == 4 || num_bits == 8, "num_bits must be 4 or 8. Got = ", num_bits);

    int const pack_factor = 32 / static_cast<int>(num_bits);

    TORCH_CHECK((size_k / pack_factor) == b_q_weight.size(0),
        "Shape mismatch: b_q_weight.size(0) = ", b_q_weight.size(0), ", size_k = ", size_k,
        ", pack_factor = ", pack_factor);
    TORCH_CHECK(b_q_weight.size(1) == size_n, "b_q_weight.size(1) = ", b_q_weight.size(1), " != size_n = ", size_n);

    TORCH_CHECK(b_q_weight.device().is_cuda(), "b_q_weight is not on GPU");
    TORCH_CHECK(b_q_weight.is_contiguous(), "b_q_weight is not contiguous");
    TORCH_CHECK(b_q_weight.dtype() == at::kInt, "b_q_weight type is not kInt");

    TORCH_CHECK(perm.device().is_cuda(), "perm is not on GPU");
    TORCH_CHECK(perm.is_contiguous(), "perm is not contiguous");
    TORCH_CHECK(perm.dtype() == at::kInt, "perm type is not at::kInt");

    const at::cuda::OptionalCUDAGuard device_guard(device_of(b_q_weight));
    auto options = torch::TensorOptions().dtype(b_q_weight.dtype()).device(b_q_weight.device());
    torch::Tensor out = torch::empty({size_k / kMarlinTileSize, size_n * kMarlinTileSize / pack_factor}, options);

    bool has_perm = perm.size(0) != 0;

    uint32_t const* b_q_weight_ptr = reinterpret_cast<uint32_t const*>(b_q_weight.data_ptr());
    uint32_t const* perm_ptr = reinterpret_cast<uint32_t const*>(perm.data_ptr());
    uint32_t* out_ptr = reinterpret_cast<uint32_t*>(out.data_ptr());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(b_q_weight.get_device());

    ::marlin_nvfp4::gptq_marlin_repack_dispatch(b_q_weight_ptr, perm_ptr, out_ptr, static_cast<int>(size_k),
        static_cast<int>(size_n), static_cast<int>(num_bits), has_perm, is_a_8bit, stream);

    return out;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "gptq_marlin_repack(Tensor b_q_weight, Tensor perm, int size_k, int size_n,"
        " int num_bits, bool is_a_8bit=False) -> (Tensor out)");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("gptq_marlin_repack", &tensorrt_llm::torch_ext::gptq_marlin_repack);
}
