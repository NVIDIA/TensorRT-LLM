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

#include "tensorrt_llm/common/mcastDevMemUtils.h"
#include "tensorrt_llm/kernels/glmMoeDsaRubin/glmMnnvlHop.h"
#include "tensorrt_llm/runtime/mcastDeviceMemory.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>

#include <cstdint>
#include <limits>
#include <vector>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace
{

namespace glm = tensorrt_llm::kernels::glm_moe_dsa;

bool isAlignedBf16(at::Tensor const& t)
{
    return t.scalar_type() == at::ScalarType::BFloat16 && t.is_contiguous()
        && reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0;
}

bool hopShapeSupported(int64_t numTokens, int64_t hiddenSize, int64_t nRanks)
{
    constexpr int64_t kIntMax = std::numeric_limits<int>::max();
    if (numTokens > kIntMax || hiddenSize > kIntMax || nRanks > kIntMax)
    {
        return false;
    }
    return glm::mnnvlHopSupported(static_cast<int>(numTokens), static_cast<int>(hiddenSize), static_cast<int>(nRanks));
}

void checkRowsAndGamma(at::Tensor const& residual, at::Tensor const& gamma, char const* op)
{
    TORCH_CHECK(residual.is_cuda() && residual.dim() == 2 && isAlignedBf16(residual), op,
        ": residual must be a contiguous, 16-byte aligned BF16 [num_tokens, hidden] CUDA tensor");
    TORCH_CHECK(gamma.dim() == 1 && gamma.size(0) == residual.size(1) && isAlignedBf16(gamma), op,
        ": gamma must be a contiguous, 16-byte aligned BF16 [hidden] tensor");
    TORCH_CHECK(gamma.device() == residual.device(), op, ": gamma must be on the device of the rows");
}

// The MNNVL all-reduce workspace that comm_buffer views ([3, N]: this rank's three Lamport buffers) for a hop of the
// rows of residual: the shape must be one the hops serve and its Lamport data must fit one buffer.
tensorrt_llm::runtime::McastDeviceMemory* hopWorkspace(
    at::Tensor const& comm_buffer, at::Tensor const& buffer_flags, at::Tensor const& residual, char const* op)
{
    auto* mcastMem = tensorrt_llm::common::findMcastDevMemBuffer(comm_buffer.data_ptr());
    TORCH_CHECK(mcastMem != nullptr, op, ": comm_buffer must be obtained from a McastGPUBuffer instance");
    TORCH_CHECK(mcastMem->isMapped(), op, ": the MNNVL workspace handles are not attached");
    TORCH_CHECK(comm_buffer.dim() == 2 && comm_buffer.size(0) == 3 && comm_buffer.is_contiguous(), op,
        ": comm_buffer must be the [3, N] view of the workspace's three Lamport buffers");
    TORCH_CHECK(buffer_flags.element_size() == 4 && buffer_flags.numel() >= 9 && buffer_flags.is_contiguous(), op,
        ": buffer_flags must hold the workspace's nine 32-bit Lamport flags");
    TORCH_CHECK(comm_buffer.device() == residual.device() && buffer_flags.device() == residual.device(), op,
        ": the workspace must be on the device of the rows");
    int64_t const numTokens = residual.size(0);
    int64_t const hiddenSize = residual.size(1);
    int64_t const nRanks = static_cast<int64_t>(mcastMem->getWorldSize());
    TORCH_CHECK(hopShapeSupported(numTokens, hiddenSize, nRanks), op, ": unsupported shape: ", numTokens,
        " tokens, hidden ", hiddenSize, ", ", nRanks, " ranks");
    int64_t const bufferBytes = comm_buffer.size(1) * comm_buffer.element_size();
    int64_t const hopBytes
        = glm::mnnvlHopBufferBytes(static_cast<int>(numTokens), static_cast<int>(hiddenSize), static_cast<int>(nRanks));
    TORCH_CHECK(hopBytes <= bufferBytes, op, ": ", hopBytes, " bytes of Lamport data exceed the Lamport buffer of ",
        bufferBytes, " bytes");
    return mcastMem;
}

glm::MnnvlHopParams makeHopParams(tensorrt_llm::runtime::McastDeviceMemory* mcastMem, at::Tensor& buffer_flags,
    at::Tensor const& residual, at::Tensor const& gamma, double epsilon, at::Tensor& normOut, at::Tensor& residualOut)
{
    glm::MnnvlHopParams params{};
    params.nRanks = static_cast<int>(mcastMem->getWorldSize());
    params.rank = static_cast<int>(mcastMem->getRank());
    params.numTokens = static_cast<int>(residual.size(0));
    params.tokenDim = static_cast<int>(residual.size(1));
    params.bufferPtrsDev = mcastMem->getBufferPtrsDev();
    params.multicastPtr = mcastMem->getMulticastPtr();
    params.bufferFlags = reinterpret_cast<uint32_t*>(buffer_flags.mutable_data_ptr());
    params.input = nullptr;
    params.residualIn = residual.const_data_ptr();
    params.gamma = gamma.const_data_ptr();
    params.epsilon = static_cast<float>(epsilon);
    params.residualOut = residualOut.mutable_data_ptr();
    params.normOut = normOut.mutable_data_ptr();
    params.shardPublished = false;
    params.stream = at::cuda::getCurrentCUDAStream(residual.get_device());
    return params;
}

} // namespace

// The one-shot MNNVL all-reduce of BF16 rows with the residual add and RMSNorm (kernels/glmMoeDsaRubin/glmMnnvlHop.h)
// on the MNNVL all-reduce workspace comm_buffer / buffer_flags. Returns [norm_out, residual_out]. With shard_published
// the producer of input already stored this rank's rows into the workspace (the o_proj GEMM's Lamport publish), and
// input only gives the shape.
std::vector<at::Tensor> glmMnnvlHop(at::Tensor const& input, at::Tensor const& residual, at::Tensor const& gamma,
    double epsilon, at::Tensor& comm_buffer, at::Tensor& buffer_flags, bool shard_published)
{
    char const* op = "glm_mnnvl_hop";
    checkRowsAndGamma(residual, gamma, op);
    TORCH_CHECK(input.sizes() == residual.sizes(), op, ": input must have the shape of residual");
    if (!shard_published)
    {
        TORCH_CHECK(isAlignedBf16(input) && input.device() == residual.device(), op,
            ": input must be a contiguous, 16-byte aligned BF16 tensor on the device of residual");
    }
    at::Tensor normOut = at::empty_like(residual);
    at::Tensor residualOut = at::empty_like(residual);
    if (residual.size(0) == 0)
    {
        return {normOut, residualOut};
    }
    auto* mcastMem = hopWorkspace(comm_buffer, buffer_flags, residual, op);
    auto params = makeHopParams(mcastMem, buffer_flags, residual, gamma, epsilon, normOut, residualOut);
    params.input = shard_published ? nullptr : input.const_data_ptr();
    params.shardPublished = shard_published;
    glm::mnnvlHop(params);
    return {normOut, residualOut};
}

// The TRTLLM-Gen MoE finalize (top-8, BF16) fused into the hop: the inputs are the MoE's do_finalize=false outputs (FC2
// rows, routing weights, expanded-to-permuted index) and the addend (the shared experts' output). Returns [norm_out,
// residual_out], those of glm_mnnvl_hop on the finalized rows. The routing weights and index are read before the
// kernel's grid dependency wait: the kernel that wrote them must have completed when this one launches.
std::vector<at::Tensor> glmMnnvlMoeFinalizeHop(at::Tensor const& gemm2_output, at::Tensor const& expert_weights,
    at::Tensor const& expanded_idx_to_permuted_idx, at::Tensor const& addend, at::Tensor const& residual,
    at::Tensor const& gamma, double epsilon, at::Tensor& comm_buffer, at::Tensor& buffer_flags)
{
    char const* op = "glm_mnnvl_moe_finalize_hop";
    checkRowsAndGamma(residual, gamma, op);
    int64_t const numTokens = residual.size(0);
    TORCH_CHECK(gemm2_output.dim() == 2 && isAlignedBf16(gemm2_output), op,
        ": gemm2_output must be a contiguous, 16-byte aligned BF16 2D tensor");
    TORCH_CHECK(gemm2_output.size(1) >= residual.size(1) && gemm2_output.size(1) % 8 == 0, op,
        ": gemm2_output rows must hold the hidden dimension in a multiple of 8 elements");
    TORCH_CHECK(expanded_idx_to_permuted_idx.scalar_type() == at::ScalarType::Int
            && expanded_idx_to_permuted_idx.dim() == 2 && expanded_idx_to_permuted_idx.is_contiguous()
            && expanded_idx_to_permuted_idx.size(0) == numTokens
            && expanded_idx_to_permuted_idx.size(1) == glm::kMnnvlHopMoeTopK,
        op, ": expanded_idx_to_permuted_idx must be contiguous int32 [num_tokens, 8]");
    TORCH_CHECK(expert_weights.scalar_type() == at::ScalarType::BFloat16 && expert_weights.is_contiguous()
            && expert_weights.sizes() == expanded_idx_to_permuted_idx.sizes(),
        op, ": expert_weights must be contiguous BF16 of the index's shape");
    TORCH_CHECK(addend.sizes() == residual.sizes() && isAlignedBf16(addend), op,
        ": addend must be a contiguous, 16-byte aligned BF16 tensor of the shape of residual");
    for (auto const* t : {&gemm2_output, &expert_weights, &expanded_idx_to_permuted_idx, &addend})
    {
        TORCH_CHECK(t->device() == residual.device(), op, ": all inputs must be on one device");
    }
    at::Tensor normOut = at::empty_like(residual);
    at::Tensor residualOut = at::empty_like(residual);
    if (numTokens == 0)
    {
        return {normOut, residualOut};
    }
    auto* mcastMem = hopWorkspace(comm_buffer, buffer_flags, residual, op);
    auto const params = makeHopParams(mcastMem, buffer_flags, residual, gamma, epsilon, normOut, residualOut);
    glm::MnnvlMoeFinalizeParams moe{};
    moe.fc2Output = gemm2_output.const_data_ptr();
    moe.hiddenDimPadded = static_cast<int>(gemm2_output.size(1));
    moe.expertWeights = expert_weights.const_data_ptr();
    moe.expandedIdxToPermutedIdx = expanded_idx_to_permuted_idx.const_data_ptr<int32_t>();
    moe.addend = addend.const_data_ptr();
    glm::mnnvlMoeFinalizeHop(params, moe);
    return {normOut, residualOut};
}

// Whether glm_mnnvl_hop and glm_mnnvl_moe_finalize_hop serve [num_tokens, hidden_size] BF16 rows over num_ranks ranks
// on the current device.
bool glmMnnvlHopSupported(int64_t num_tokens, int64_t hidden_size, int64_t num_ranks)
{
    return hopShapeSupported(num_tokens, hidden_size, num_ranks);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "glm_mnnvl_hop(Tensor input, Tensor residual, Tensor gamma, float epsilon, Tensor(a!) comm_buffer, "
        "Tensor(b!) buffer_flags, bool shard_published) -> Tensor[]");
    m.def(
        "glm_mnnvl_moe_finalize_hop(Tensor gemm2_output, Tensor expert_weights, Tensor expanded_idx_to_permuted_idx, "
        "Tensor addend, Tensor residual, Tensor gamma, float epsilon, Tensor(a!) comm_buffer, Tensor(b!) buffer_flags) "
        "-> Tensor[]");
    m.def("glm_mnnvl_hop_supported", &tensorrt_llm::torch_ext::glmMnnvlHopSupported);
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("glm_mnnvl_hop", &tensorrt_llm::torch_ext::glmMnnvlHop);
    m.impl("glm_mnnvl_moe_finalize_hop", &tensorrt_llm::torch_ext::glmMnnvlMoeFinalizeHop);
}
