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
//
// DWDP fabric gather: pull remote expert weights with the SMs instead of a copy engine.
//
// The plan (which peer owns which chunk, and the group order) is built on the host by DWDPWeightManager;
// this op only takes the resulting device-resident descriptor arrays and launches.  Deliberately a plain
// kernel launch, not cudaMemcpyBatchAsync: it is CUDA-graph capturable and has no per-descriptor
// submission cost.

#include "dwdpFabricGather.h"

#include <ATen/cuda/CUDAContext.h>
#include <torch/library.h>
#include <torch/torch.h>

namespace tensorrt_llm::torch_ext
{

namespace
{

int64_t const* checkedInt64(torch::Tensor const& t, char const* name, int64_t expectedNumel)
{
    TORCH_CHECK(t.is_cuda(), "dwdp_fabric_gather: ", name, " must be a CUDA tensor");
    TORCH_CHECK(t.scalar_type() == torch::kInt64, "dwdp_fabric_gather: ", name, " must be int64");
    TORCH_CHECK(t.is_contiguous(), "dwdp_fabric_gather: ", name, " must be contiguous");
    TORCH_CHECK(t.numel() == expectedNumel, "dwdp_fabric_gather: ", name, " has ", t.numel(), " entries, expected ",
        expectedNumel);
    return t.data_ptr<int64_t>();
}

} // namespace

//! Launch the gather described by `dstPtrs` / `srcPtrs` / `lines`, grouped by `rowStarts`.
//!
//! Runs on the current stream.  Pointers are raw device addresses (destinations in the local weight
//! buffer, sources in a peer's MNNVL mapping); every pointer and byte count must be 16-byte aligned,
//! which the host-side plan builder checks (falling back to the copy engine otherwise).
void dwdpFabricGather(torch::Tensor const& dstPtrs, torch::Tensor const& srcPtrs, torch::Tensor const& lines,
    torch::Tensor const& rowStarts, int64_t blocksPerRow, int64_t threads)
{
    int64_t const numDescriptors = dstPtrs.numel();
    TORCH_CHECK(numDescriptors > 0, "dwdp_fabric_gather: empty descriptor list");
    TORCH_CHECK(rowStarts.numel() >= 2, "dwdp_fabric_gather: row_starts must hold rows + 1 entries");
    int64_t const rows = rowStarts.numel() - 1;

    auto const* dsts = checkedInt64(dstPtrs, "dst_ptrs", numDescriptors);
    auto const* srcs = checkedInt64(srcPtrs, "src_ptrs", numDescriptors);
    auto const* lineCounts = checkedInt64(lines, "lines", numDescriptors);
    auto const* starts = checkedInt64(rowStarts, "row_starts", rows + 1);

    TORCH_CHECK(blocksPerRow > 0, "dwdp_fabric_gather: blocks_per_row must be positive");
    TORCH_CHECK(threads > 0 && threads % 32 == 0 && threads <= 1024,
        "dwdp_fabric_gather: threads must be a positive multiple of 32 up to 1024, got ", threads);

    launchDwdpFabricGather(reinterpret_cast<void* const*>(dsts), reinterpret_cast<void const* const*>(srcs), lineCounts,
        starts, rows, blocksPerRow, threads, at::cuda::getCurrentCUDAStream());
}

} // namespace tensorrt_llm::torch_ext

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "dwdp_fabric_gather(Tensor dst_ptrs, Tensor src_ptrs, Tensor lines, Tensor row_starts, "
        "int blocks_per_row, int threads) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("dwdp_fabric_gather", &tensorrt_llm::torch_ext::dwdpFabricGather);
}
