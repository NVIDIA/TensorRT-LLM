/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/ulyssesPermuteScatterKernel.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{

// Standalone Python entry point for ulyssesPermuteScatterKernel — used by
// unit tests. The production caller is the Ulysses async A2A path in
// alltoallOp.cpp, which combines this kernel with IPC writes + an LSA
// barrier; that whole sequence is multi-rank and not amenable to a
// single-GPU pytest. By exposing just the kernel, we can validate the
// permute+scatter layout transform independently.
//
// Layout:
//   input    : bf16 [B, S_local, H,        D]   contiguous
//   send_buf : bf16 [P, B, S_local, H/P,   D]   contiguous
//   recv_buf : bf16 [P, B, S_local, H/P,   D]   contiguous
// For each (b, s, h, d):
//   peer    = h // (H/P)
//   h_local = h %  (H/P)
//   dst     = (recv_buf if peer == my_rank else send_buf)
//   slot    = peer  // applies to both branches
//   dst[slot, b, s, h_local, d] = input[b, s, h, d]
void ulysses_permute_scatter(torch::Tensor& input, // [B, S_local, H, D]
    torch::Tensor& send_buf,                       // [P, B, S_local, H/P, D]
    torch::Tensor& recv_buf,                       // [P, B, S_local, H/P, D]
    int64_t my_rank, int64_t P)
{
    TORCH_CHECK(input.dim() == 4, "input must be 4D [B, S_local, H, D]");
    TORCH_CHECK(send_buf.dim() == 5 && recv_buf.dim() == 5, "send_buf / recv_buf must be 5D [P, B, S_local, H/P, D]");
    CHECK_INPUT(input, torch::kBFloat16);
    CHECK_INPUT(send_buf, torch::kBFloat16);
    CHECK_INPUT(recv_buf, torch::kBFloat16);

    int64_t const B = input.size(0);
    int64_t const S_local = input.size(1);
    int64_t const H = input.size(2);
    int64_t const D = input.size(3);
    TORCH_CHECK(H % P == 0, "H must be divisible by P");
    TORCH_CHECK(D % 8 == 0, "D must be divisible by 8 (uint4 vec)");
    int64_t const H_local = H / P;
    TORCH_CHECK(send_buf.size(0) == P && send_buf.size(1) == B && send_buf.size(2) == S_local
            && send_buf.size(3) == H_local && send_buf.size(4) == D,
        "send_buf shape mismatch");
    TORCH_CHECK(recv_buf.size(0) == P && recv_buf.size(1) == B && recv_buf.size(2) == S_local
            && recv_buf.size(3) == H_local && recv_buf.size(4) == D,
        "recv_buf shape mismatch");
    TORCH_CHECK(0 <= my_rank && my_rank < P, "my_rank out of range");

    auto stream = at::cuda::getCurrentCUDAStream();
    tensorrt_llm::kernels::launchUlyssesPermuteScatter(input.data_ptr(), send_buf.data_ptr(), recv_buf.data_ptr(),
        static_cast<int>(my_rank), static_cast<int>(B), static_cast<int>(S_local), static_cast<int>(H),
        static_cast<int>(D), static_cast<int>(P), stream);
}

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "ulysses_permute_scatter(Tensor(a!) input, Tensor(b!) send_buf, Tensor(c!) recv_buf, "
        "int my_rank, int P) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("ulysses_permute_scatter", &ulysses_permute_scatter);
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END
