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

#include "minimax_m3/csrc/m31/dflashMergeLaterKeys.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

bool aligned8(torch::Tensor const& t, int outerDims)
{
    if (reinterpret_cast<uintptr_t>(t.data_ptr()) % 8 != 0)
    {
        return false;
    }
    for (int d = 0; d < outerDims; ++d)
    {
        if (t.stride(d) % 4 != 0)
        {
            return false;
        }
    }
    return true;
}

} // namespace

//! The DFlash / DSpark non-causal block merge (modeling_dflash.dflash_merge_later_block_keys_), in place on out (see
//! dflashMergeLaterKeys.h). out: bf16 [B * blk, Hq, 128]; lse_log2: float32 [B * blk, Hq]; query: bf16 [B, blk, Hq,
//! 128]; block_key / block_value: bf16 [B, blk, Hkv, 128]; every last dim unit-stride, every bf16 base 8-byte aligned
//! with outer strides a multiple of 4; blk <= 16.
void minimaxM31DflashMergeLaterKeys(torch::Tensor& out, torch::Tensor const& lseLog2, torch::Tensor const& query,
    torch::Tensor const& blockKey, torch::Tensor const& blockValue, double scale, int64_t windowRight, bool pdl)
{
    using TRTLLM_MV2_NS::kernels::minimaxM31::kDflashMergeHeadDim;
    using TRTLLM_MV2_NS::kernels::minimaxM31::kDflashMergeMaxBlock;
    TORCH_CHECK(query.dim() == 4 && blockKey.dim() == 4 && blockValue.dim() == 4 && out.dim() == 3,
        "minimax_m31_dflash_merge_later_keys: query / key / value [B, blk, H, D], out [B * blk, Hq, D]");
    TORCH_CHECK(query.scalar_type() == at::kBFloat16 && blockKey.scalar_type() == at::kBFloat16
            && blockValue.scalar_type() == at::kBFloat16 && out.scalar_type() == at::kBFloat16,
        "minimax_m31_dflash_merge_later_keys: bf16 query / key / value / out");
    TORCH_CHECK(lseLog2.scalar_type() == at::kFloat && lseLog2.dim() == 2,
        "minimax_m31_dflash_merge_later_keys: lse_log2 must be float32 [B * blk, Hq]");
    TORCH_CHECK(out.is_cuda() && lseLog2.device() == out.device() && query.device() == out.device()
            && blockKey.device() == out.device() && blockValue.device() == out.device(),
        "minimax_m31_dflash_merge_later_keys: every operand on out's CUDA device");
    int64_t const B = query.size(0), blk = query.size(1), hq = query.size(2), d = query.size(3);
    int64_t const hkv = blockKey.size(2);
    TORCH_CHECK(d == kDflashMergeHeadDim && blk <= kDflashMergeMaxBlock && hkv > 0 && hq % hkv == 0,
        "minimax_m31_dflash_merge_later_keys: head_dim 128, blk <= 16, Hq a multiple of Hkv");
    TORCH_CHECK(blockKey.size(0) == B && blockKey.size(1) == blk && blockKey.size(3) == d
            && blockValue.sizes() == blockKey.sizes() && out.size(0) == B * blk && out.size(1) == hq && out.size(2) == d
            && lseLog2.size(0) == B * blk && lseLog2.size(1) == hq,
        "minimax_m31_dflash_merge_later_keys: operand shapes disagree");
    TORCH_CHECK(query.stride(3) == 1 && blockKey.stride(3) == 1 && blockValue.stride(3) == 1 && out.stride(2) == 1,
        "minimax_m31_dflash_merge_later_keys: unit inner strides");
    TORCH_CHECK(aligned8(out, 2) && aligned8(query, 3) && aligned8(blockKey, 3) && aligned8(blockValue, 3),
        "minimax_m31_dflash_merge_later_keys: 8-byte aligned bf16 bases, outer strides a multiple of 4");
    TRTLLM_MV2_NS::kernels::minimaxM31::DflashMergeParams p{};
    p.out = reinterpret_cast<__nv_bfloat16*>(out.data_ptr());
    p.lseLog2 = lseLog2.data_ptr<float>();
    p.query = reinterpret_cast<__nv_bfloat16 const*>(query.data_ptr());
    p.key = reinterpret_cast<__nv_bfloat16 const*>(blockKey.data_ptr());
    p.value = reinterpret_cast<__nv_bfloat16 const*>(blockValue.data_ptr());
    p.outRow = out.stride(0);
    p.outHead = out.stride(1);
    p.lseRow = lseLog2.stride(0);
    p.lseHead = lseLog2.stride(1);
    p.qB = query.stride(0);
    p.qT = query.stride(1);
    p.qH = query.stride(2);
    p.kB = blockKey.stride(0);
    p.kT = blockKey.stride(1);
    p.kH = blockKey.stride(2);
    p.vB = blockValue.stride(0);
    p.vT = blockValue.stride(1);
    p.vH = blockValue.stride(2);
    p.B = static_cast<int>(B);
    p.blk = static_cast<int>(blk);
    p.numHeads = static_cast<int>(hq);
    p.group = static_cast<int>(hq / hkv);
    p.windowRight = static_cast<int>(windowRight);
    p.scale = static_cast<float>(scale);
    at::cuda::CUDAGuard const guard(out.device());
    cudaError_t const e = TRTLLM_MV2_NS::kernels::minimaxM31::invokeDflashMergeLaterKeys(
        p, pdl, at::cuda::getCurrentCUDAStream(out.get_device()));
    TORCH_CHECK(e == cudaSuccess, "minimax_m31_dflash_merge_later_keys launch: ", cudaGetErrorString(e));
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "minimax_m31_dflash_merge_later_keys(Tensor(a!) out, Tensor lse_log2, Tensor query, Tensor block_key, Tensor "
        "block_value, float scale, int window_right, bool pdl=True) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("minimax_m31_dflash_merge_later_keys", &TRTLLM_MV2_NS::torch_ext::minimaxM31DflashMergeLaterKeys);
}
