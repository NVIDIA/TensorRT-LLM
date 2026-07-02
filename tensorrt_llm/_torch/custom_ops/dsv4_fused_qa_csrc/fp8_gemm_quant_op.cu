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

// Torch op wrapper for the fused fp8-out GEMM (fp8_gemm_quant.cu): reuses the 2-CTA
// cluster kernel + launcher verbatim, exposes fp8_gemm_quant_out(A_fp8, A_sf, B_fp8, B_sf)
// -> (D_fp8, D_sf) where D = A @ B^T quantized per-(token,128) into the packed UE8M0 layout
// deep_gemm's fp8_gemm_nt reads. Inputs A_sf/B_sf are the SAME packed UE8M0 format
// (fp8_quantize_1x128_packed_ue8m0). NOTE: launcher hardcodes sfa/sfb leading dim = M/N,
// so this op currently requires M%4==0 and N%4==0 (m_aligned==M, n_aligned==N).
#define FP8_GEMM_QUANT_NO_MAIN
#include "fp8_gemm_quant.cu"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> fp8_gemm_quant_out(torch::Tensor A_fp8, // [M, K] e4m3 row-major
    torch::Tensor A_sf,                                            // packed UE8M0 int32, leading dim M (m_aligned)
    torch::Tensor B_fp8,                                           // [N, K] e4m3 row-major
    torch::Tensor B_sf)                                            // packed UE8M0 int32, leading dim N (n_aligned)
{
    TORCH_CHECK(A_fp8.is_cuda() && B_fp8.is_cuda(), "inputs must be CUDA");
    TORCH_CHECK(A_fp8.scalar_type() == torch::kFloat8_e4m3fn, "A must be e4m3");
    TORCH_CHECK(B_fp8.scalar_type() == torch::kFloat8_e4m3fn, "B must be e4m3");
    A_fp8 = A_fp8.contiguous();
    B_fp8 = B_fp8.contiguous();
    int M = A_fp8.size(0), K = A_fp8.size(1), N = B_fp8.size(0);
    TORCH_CHECK(B_fp8.size(1) == K, "K mismatch");
    // M is arbitrary (sfa stride uses m_aligned; epilogue guards token<M). N must be a multiple of
    // STORE_BLOCK_N=128 for the fp8 TMA store, and the sfb leading dim equals N.
    TORCH_CHECK(N % 128 == 0, "this op requires N % 128 == 0");

    int num_n_blocks = (N + 127) / 128;
    int num_packed_sf_k = (num_n_blocks + 3) / 4;
    int m_aligned = (M + 3) / 4 * 4;                // activation/output scale leading dim padding

    auto D = torch::empty({M, N}, A_fp8.options()); // e4m3
    auto Dsf = torch::zeros({num_packed_sf_k, m_aligned}, torch::dtype(torch::kInt32).device(A_fp8.device()));

    GemmProblem prob;
    prob.M = (uint32_t) M;
    prob.N = (uint32_t) N;
    prob.K = (uint32_t) K;
    prob.A = (__nv_fp8_e4m3*) A_fp8.data_ptr();
    prob.B = (__nv_fp8_e4m3*) B_fp8.data_ptr();
    prob.D = (__nv_fp8_e4m3*) D.data_ptr();
    prob.sf = (uint8_t*) Dsf.data_ptr();
    prob.sfa = (uint32_t*) A_sf.data_ptr();
    prob.sfb = (uint32_t*) B_sf.data_ptr();

    cudaStream_t stream = at::cuda::getCurrentCUDAStream(A_fp8.get_device()).stream();
    launch_fp8_gemm_2cta(prob, stream);
    return {D, Dsf};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("fp8_gemm_quant_out", &fp8_gemm_quant_out, "fused fp8-out GEMM (A@B^T -> fp8 + packed UE8M0)");
}
