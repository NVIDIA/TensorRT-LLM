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

#include "tensorrt_llm/kernels/minimaxM31/f3ArPrepushedSm107.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdlib>
#include <optional>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

namespace
{

//! The shipped libm31_f3_ar.so's launch rule: the programmatic launch only with TRTLLM_ENABLE_PDL=1 (read once).
bool f3PdlEnabled()
{
    static char const* e = std::getenv("TRTLLM_ENABLE_PDL");
    return e != nullptr && e[0] == '1';
}

} // namespace

//! F3 attention boundary (TP2 decode): the one-shot Lamport all-reduce + residual + RMSNorm over the rows the o_proj's
//! F3 GEMM epilogue already pushed into the AllReduce workspace's data slots (no push loop), + optionally the MXFP8 of
//! norm_out (mx_out e4m3 [rows, hidden], mx_sf swizzled UE8M0 of pad128(rows) rows). allreduce_in is shape-only (its
//! bytes are not read). Bitwise to the stock one-shot allreduce_fusion RESIDUAL_RMS_NORM (+ mxfp8_quantize).
void minimaxM31F3ArPrepushed(int64_t rank, torch::Tensor workspace, torch::Tensor allreduceIn, torch::Tensor residualIn,
    torch::Tensor residualOut, torch::Tensor normOut, torch::Tensor gamma, double eps, bool triggerCompletionAtEnd,
    std::optional<torch::Tensor> mxOut, std::optional<torch::Tensor> mxSf)
{
    TORCH_CHECK(allreduceIn.scalar_type() == at::kBFloat16 && allreduceIn.is_contiguous());
    tensorrt_llm::kernels::minimaxM31::F3ArParams p;
    p.rank = static_cast<int>(rank);
    p.size = static_cast<int>(allreduceIn.numel());
    p.hidden_dim = static_cast<int>(allreduceIn.size(-1));
    p.workspace = reinterpret_cast<void**>(workspace.data_ptr());
    p.allreduce_in = allreduceIn.data_ptr();
    p.residual_in = residualIn.data_ptr();
    p.residual_out = residualOut.data_ptr();
    p.norm_out = normOut.data_ptr();
    p.rms_gamma = gamma.data_ptr();
    p.rms_eps = static_cast<float>(eps);
    p.mx_out = nullptr;
    p.mx_sf = nullptr;
    if (mxOut.has_value())
    {
        TORCH_CHECK(mxSf.has_value() && mxOut->is_contiguous() && mxSf->is_contiguous());
        int64_t const rows = p.size / p.hidden_dim;
        TORCH_CHECK(mxOut->numel() * mxOut->element_size() == rows * p.hidden_dim, "mx_out must be [rows, H] e4m3");
        TORCH_CHECK(mxSf->numel() * mxSf->element_size() == (rows + 127) / 128 * 128 * (p.hidden_dim / 32),
            "mx_sf must be the swizzled [pad128(rows), H / 32] UE8M0 scales");
        p.mx_out = mxOut->data_ptr();
        p.mx_sf = mxSf->data_ptr();
    }
    TORCH_CHECK(p.size % p.hidden_dim == 0 && p.hidden_dim % 8 == 0);
    at::cuda::CUDAGuard const guard(allreduceIn.device());
    C10_CUDA_CHECK(tensorrt_llm::kernels::minimaxM31::invokeF3ArPrepushed(
        p, triggerCompletionAtEnd, f3PdlEnabled(), at::cuda::getCurrentCUDAStream(allreduceIn.get_device())));
}

//! 8 bytes at a device address (the Lamport workspace's comm size lives in device memory); setup only, never in a
//! CUDA graph.
int64_t minimaxM31F3ReadI64(int64_t addr)
{
    int64_t v = 0;
    C10_CUDA_CHECK(cudaMemcpy(&v, reinterpret_cast<void*>(addr), 8, cudaMemcpyDeviceToHost));
    return v;
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("minimax_m31_f3_read_i64(int addr) -> int", &tensorrt_llm::torch_ext::minimaxM31F3ReadI64);
    m.def(
        "minimax_m31_f3_ar_prepushed(int rank, Tensor workspace, Tensor allreduce_in, Tensor residual_in, "
        "Tensor(a!) residual_out, Tensor(b!) norm_out, Tensor gamma, float eps, bool trigger_completion_at_end, "
        "Tensor(c!)? mx_out, Tensor(d!)? mx_sf) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m31_f3_ar_prepushed", &tensorrt_llm::torch_ext::minimaxM31F3ArPrepushed);
}
