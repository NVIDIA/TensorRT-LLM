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

#include "tensorrt_llm/kernels/minimaxM31/tp2ctxPeer.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace
{
int smCount(int device)
{
    static int count[64] = {0};
    if (count[device] == 0)
    {
        cudaDeviceGetAttribute(&count[device], cudaDevAttrMultiProcessorCount, device);
    }
    return count[device];
}
} // namespace

//! TP2CTX_AR1 / TP2CTX_SP: out[:numel] = bf16(float(a) + float(b)) where a / b are raw device pointers (b typically the
//! peer rank's symmetric buffer). The caller orders it after a cross-rank barrier (torch symmetric memory). out is
//! written in place; numel % 8 == 0 and the pointers 16-byte aligned.
void minimaxM31Tp2ctxPeerAdd(int64_t a, int64_t b, torch::Tensor& out, int64_t numel)
{
    TORCH_CHECK(out.is_cuda() && out.is_contiguous() && out.scalar_type() == at::kBFloat16 && out.numel() >= numel,
        "minimax_m31_tp2ctx_peer_add: out must be a contiguous bf16 CUDA tensor of >= numel elements");
    TORCH_CHECK(numel % 8 == 0 && a % 16 == 0 && b % 16 == 0 && reinterpret_cast<uintptr_t>(out.data_ptr()) % 16 == 0,
        "minimax_m31_tp2ctx_peer_add: numel % 8 == 0 and 16-byte aligned pointers");
    at::cuda::CUDAGuard const guard(out.device());
    tensorrt_llm::kernels::minimaxM31::launchTp2ctxPeerAdd(reinterpret_cast<void const*>(a),
        reinterpret_cast<void const*>(b), out.data_ptr(), numel, smCount(out.get_device()),
        at::cuda::getCurrentCUDAStream(out.get_device()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

//! TP2CTX_SP all-gather half / two-shot push: dst (raw pointer, typically the peer's symmetric buffer) <- src bytes.
void minimaxM31Tp2ctxPeerPush(torch::Tensor const& src, int64_t dst)
{
    TORCH_CHECK(src.is_cuda() && src.is_contiguous(), "minimax_m31_tp2ctx_peer_push: a contiguous CUDA tensor");
    int64_t const nBytes = src.numel() * src.element_size();
    TORCH_CHECK(nBytes % 16 == 0 && dst % 16 == 0 && reinterpret_cast<uintptr_t>(src.data_ptr()) % 16 == 0,
        "minimax_m31_tp2ctx_peer_push: 16-byte multiple size and aligned pointers");
    at::cuda::CUDAGuard const guard(src.device());
    tensorrt_llm::kernels::minimaxM31::launchTp2ctxPeerPush(src.data_ptr(), reinterpret_cast<void*>(dst), nBytes,
        smCount(src.get_device()), at::cuda::getCurrentCUDAStream(src.get_device()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

//! TP2CTX_SP / AR1 two-rank barrier over MPI-IPC flag buffers: local_flags / peer_flags are raw device pointers (this
//! rank's and the peer's int[2] flag slots), state an int32 CUDA tensor holding the local epoch (zero-initialised, kept
//! for the process).
void minimaxM31Tp2ctxIpcBarrier(int64_t localFlags, int64_t peerFlags, torch::Tensor& state, int64_t rank)
{
    TORCH_CHECK(state.is_cuda() && state.scalar_type() == at::kInt && state.numel() >= 1,
        "minimax_m31_tp2ctx_ipc_barrier: state must be an int32 CUDA tensor");
    TORCH_CHECK(rank == 0 || rank == 1, "minimax_m31_tp2ctx_ipc_barrier: two ranks only");
    at::cuda::CUDAGuard const guard(state.device());
    tensorrt_llm::kernels::minimaxM31::launchTp2ctxIpcBarrier(reinterpret_cast<int*>(localFlags),
        reinterpret_cast<int*>(peerFlags), state.data_ptr<int>(), static_cast<int>(rank),
        at::cuda::getCurrentCUDAStream(state.get_device()));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def("minimax_m31_tp2ctx_peer_add(int a, int b, Tensor(a!) out, int numel) -> ()");
    m.def("minimax_m31_tp2ctx_peer_push(Tensor src, int dst) -> ()");
    m.def("minimax_m31_tp2ctx_ipc_barrier(int local_flags, int peer_flags, Tensor(a!) state, int rank) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("minimax_m31_tp2ctx_peer_add", &tensorrt_llm::torch_ext::minimaxM31Tp2ctxPeerAdd);
    m.impl("minimax_m31_tp2ctx_peer_push", &tensorrt_llm::torch_ext::minimaxM31Tp2ctxPeerPush);
    m.impl("minimax_m31_tp2ctx_ipc_barrier", &tensorrt_llm::torch_ext::minimaxM31Tp2ctxIpcBarrier);
}
