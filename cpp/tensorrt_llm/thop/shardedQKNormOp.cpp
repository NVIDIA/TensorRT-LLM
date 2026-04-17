/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/kernels/shardedQKNormKernel.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

namespace tensorrt_llm::torch_ext
{

/**
 * Fused sharded QK RMSNorm using NVLink-based cross-GPU barrier.
 *
 * @param q         Q tensor [..., local_q_dim] (fp16 or bf16)
 * @param k         K tensor [..., local_k_dim] (fp16 or bf16)
 * @param weight_q  RMSNorm weight for Q [local_q_dim] (must be float32)
 * @param weight_k  RMSNorm weight for K [local_k_dim] (must be float32)
 * @param q_out     Pre-allocated output for Q (same shape/dtype as q)
 * @param k_out     Pre-allocated output for K (same shape/dtype as k)
 * @param workspace LongTensor encoding void** workspace (from get_allreduce_workspace)
 * @param eps       RMSNorm epsilon
 * @param world_size Tensor parallel size
 * @param rank      This rank's index
 */
void lamport_sharded_qk_rmsnorm(at::Tensor const& q, at::Tensor const& k, at::Tensor const& weight_q,
    at::Tensor const& weight_k, at::Tensor& q_out, at::Tensor& k_out, at::Tensor const& workspace, double eps,
    int64_t world_size, int64_t rank)
{
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(q.scalar_type() == k.scalar_type(), "q and k must have the same dtype");
    TORCH_CHECK(q.scalar_type() == at::ScalarType::Half || q.scalar_type() == at::ScalarType::BFloat16,
        "q must be fp16 or bf16");
    TORCH_CHECK(weight_q.scalar_type() == at::ScalarType::Float, "weight_q must be float32");
    TORCH_CHECK(weight_k.scalar_type() == at::ScalarType::Float, "weight_k must be float32");
    TORCH_CHECK(workspace.scalar_type() == at::ScalarType::Long, "workspace must be a LongTensor");

    // Inputs may be strided views (e.g. slices of a fused Q+K+V GEMM output).
    // Require the last dim to be contiguous (element stride 1); use row stride for the kernel.
    TORCH_CHECK(q.dim() >= 2 && k.dim() >= 2, "q and k must be at least 2D");
    TORCH_CHECK(q.stride(-1) == 1, "q must be contiguous along its last dim");
    TORCH_CHECK(k.stride(-1) == 1, "k must be contiguous along its last dim");
    TORCH_CHECK(q_out.is_contiguous(), "q_out must be contiguous");
    TORCH_CHECK(k_out.is_contiguous(), "k_out must be contiguous");

    // Flatten to [n_tokens, local_dim]
    int64_t n_tokens = q.numel() / q.size(-1);
    int64_t local_q_dim = q.size(-1);
    int64_t local_k_dim = k.size(-1);
    // Row stride in elements (handles both packed Q/K and fused-GEMM-slice inputs).
    // For a 2D tensor with stride(-1) == 1, stride(-2) is the row stride in elements.
    int64_t q_stride = (q.dim() >= 2) ? q.stride(-2) : local_q_dim;
    int64_t k_stride = (k.dim() >= 2) ? k.stride(-2) : local_k_dim;

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    bool is_bf16 = (q.scalar_type() == at::ScalarType::BFloat16);

    // The workspace tensor is a LongTensor whose data pointer IS the void** array
    void** ws_ptr = reinterpret_cast<void**>(workspace.data_ptr());

    tensorrt_llm::kernels::launchShardedQKNormKernel(q.data_ptr(), k.data_ptr(), q_out.data_ptr(), k_out.data_ptr(),
        weight_q.data_ptr(), weight_k.data_ptr(), ws_ptr, static_cast<int>(n_tokens), static_cast<int>(local_q_dim),
        static_cast<int>(local_k_dim), static_cast<int>(q_stride), static_cast<int>(k_stride),
        static_cast<int>(world_size), static_cast<int>(rank), static_cast<float>(eps), is_bf16, stream);
}

} // namespace tensorrt_llm::torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "lamport_sharded_qk_rmsnorm("
        "Tensor q,"
        "Tensor k,"
        "Tensor weight_q,"
        "Tensor weight_k,"
        "Tensor(a!) q_out,"
        "Tensor(a!) k_out,"
        "Tensor workspace,"
        "float eps,"
        "int world_size,"
        "int rank) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("lamport_sharded_qk_rmsnorm", &tensorrt_llm::torch_ext::lamport_sharded_qk_rmsnorm);
}
