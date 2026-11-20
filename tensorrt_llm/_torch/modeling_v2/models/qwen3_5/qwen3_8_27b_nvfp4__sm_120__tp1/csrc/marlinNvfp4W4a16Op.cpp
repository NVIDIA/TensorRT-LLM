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

#include "qwen3_5/qwen3_8_27b_nvfp4__sm_120__tp1/csrc/marlin/marlinNvfp4.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"
#include "tensorrt_llm/thop/thUtils.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <algorithm>
#include <mutex>
#include <unordered_map>

using torch::Tensor;

TRTLLM_MV2_NAMESPACE_BEGIN

namespace
{

// Persistent zeroed lock workspace for the dense Marlin GEMM.
//
// The kernel leaves every lock it uses at 0 when it completes (use_atomic_add
// is false: the last slice of each split-K reduction resets its lock in
// barrier_release, and blocks_per_sm == 1 keeps locks_off < sms), so a buffer
// zeroed once can be reused by every call that
// cannot overlap with another call on the same buffer. Calls on one stream are
// ordered (the kernel issues griddepcontrol.wait before any lock access, so
// this also holds under PDL), so each stream gets its own slot.
//
// The pool is created on the first non-capturing call on a device. A stream
// without a slot binds a free one with no allocation, which is what a CUDA
// graph capture stream gets (torch.cuda.graph captures on its own side stream
// that never runs eager work). With no pool yet or no free slot, the call falls
// back to a per-call at::zeros workspace.
constexpr int kMarlinWorkspaceSlots = 32;

struct MarlinWorkspacePool
{
    at::Tensor buffer; // [kMarlinWorkspaceSlots * sms] int32, zero-initialized
    int sms = 0;
    int nextSlot = 0;
    std::unordered_map<cudaStream_t, int> slotOfStream;
};

std::mutex& marlinWorkspaceMutex()
{
    static std::mutex m;
    return m;
}

// Intentionally leaked so no tensor is released after CUDA teardown at exit.
std::unordered_map<int, MarlinWorkspacePool*>& marlinWorkspacePools()
{
    static auto* pools = new std::unordered_map<int, MarlinWorkspacePool*>();
    return *pools;
}

// Returns this stream's zeroed slot (at least `sms` ints), or nullptr to fall back.
int* getPersistentMarlinWorkspace(Tensor const& like, int device, int sms, cudaStream_t stream)
{
    cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
    TLLM_CUDA_CHECK(cudaStreamIsCapturing(stream, &status));
    bool const capturing = status != cudaStreamCaptureStatusNone;

    std::lock_guard<std::mutex> guard(marlinWorkspaceMutex());
    auto& pools = marlinWorkspacePools();
    auto it = pools.find(device);
    MarlinWorkspacePool* pool = it == pools.end() ? nullptr : it->second;
    if (pool == nullptr)
    {
        if (capturing)
        {
            return nullptr;
        }
        pool = new MarlinWorkspacePool();
        pool->sms = sms;
        pool->buffer = at::zeros({kMarlinWorkspaceSlots * sms}, like.options().dtype(at::kInt));
        // Make the zero fill visible to every stream that later binds a slot.
        TLLM_CUDA_CHECK(cudaStreamSynchronize(stream));
        pools[device] = pool;
    }
    if (pool->sms < sms)
    {
        return nullptr;
    }
    int slot = -1;
    auto sit = pool->slotOfStream.find(stream);
    if (sit != pool->slotOfStream.end())
    {
        slot = sit->second;
    }
    else
    {
        if (pool->nextSlot >= kMarlinWorkspaceSlots)
        {
            return nullptr;
        }
        slot = pool->nextSlot++;
        pool->slotOfStream.emplace(stream, slot);
    }
    return pool->buffer.data_ptr<int>() + static_cast<int64_t>(slot) * pool->sms;
}

// W4A16 NVFP4 GEMM on the target's Marlin copy (csrc/marlin).
//
// mat_a:               [M, K] BF16 activations
// mat_b:               Marlin-packed FP4 weights (int32), from gptq_marlin_repack
// scale_b:             Marlin-processed weight block scales
// weight_global_scale: BF16 Marlin-processed weight global scale
// size_n, size_k:      N and K
// workspace_mode:      0 = split-K partial sums chained through the BF16 output; 2 = split-K partial
//                      sums reduced in FP32 (per-call FP32 C_tmp), so each output is rounded to BF16 once
// launch_hint:         0 = the stock launch configuration; 1 = the target's decode launch tables
// Returns [M, N] BF16. The lock workspace is persistent per (device, stream); see above.
Tensor marlin_nvfp4_w4a16_gemm(Tensor const& mat_a, Tensor const& mat_b, Tensor const& scale_b,
    Tensor const& weight_global_scale, int64_t size_n, int64_t size_k, int64_t workspace_mode, int64_t launch_hint)
{
    TORCH_CHECK(workspace_mode == 0 || workspace_mode == 2, "workspace_mode must be 0 or 2, got ", workspace_mode);
    TORCH_CHECK(launch_hint == marlin_nvfp4::kMarlinNvfp4LaunchHintDefault
            || launch_hint == marlin_nvfp4::kMarlinNvfp4LaunchHintDecodeTables,
        "launch_hint must be 0 or 1, got ", launch_hint);
    CHECK_TH_CUDA(mat_a);
    CHECK_TH_CUDA(mat_b);
    CHECK_TH_CUDA(scale_b);
    CHECK_TH_CUDA(weight_global_scale);
    TORCH_CHECK(mat_a.dim() == 2 && mat_a.scalar_type() == at::ScalarType::BFloat16 && mat_a.is_contiguous(),
        "mat_a must be a contiguous 2D BF16 tensor");
    TORCH_CHECK(mat_a.size(1) == size_k, "mat_a has ", mat_a.size(1), " columns, size_k is ", size_k);

    int32_t const m = static_cast<int32_t>(mat_a.size(0));
    int32_t const n = static_cast<int32_t>(size_n);
    int32_t const k = static_cast<int32_t>(size_k);

    at::cuda::OptionalCUDAGuard const deviceGuard(device_of(mat_a));
    Tensor out = at::empty({m, n}, mat_a.options());
    if (m == 0)
    {
        return out;
    }
    auto stream = at::cuda::getCurrentCUDAStream(mat_a.get_device());

    int sms = 0;
    TLLM_CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, mat_a.get_device()));

    bool const fp32Reduce = workspace_mode == 2;
    Tensor workspace;
    int* workspacePtr = getPersistentMarlinWorkspace(mat_a, mat_a.get_device(), sms, stream);
    if (workspacePtr == nullptr)
    {
        workspace = at::zeros({sms}, mat_a.options().dtype(at::kInt));
        workspacePtr = workspace.data_ptr<int>();
    }

    // workspace_mode 2: one [tb_m x tb_n] FP32 tile per lock slot (locks_off < sms, blocks_per_sm == 1),
    // tb_m <= 64 rows, tb_n <= 256 columns. Slice 0 writes its tile before any slice reads it, so the
    // buffer needs no zero fill.
    Tensor cTmp;
    void* cTmpPtr = nullptr;
    if (fp32Reduce)
    {
        int64_t const tileRows = std::min<int64_t>((static_cast<int64_t>(m) + 15) / 16 * 16, 64);
        cTmp = at::empty({static_cast<int64_t>(sms) * tileRows * 256}, mat_a.options().dtype(at::kFloat));
        cTmpPtr = cTmp.data_ptr();
    }

    marlin_nvfp4::marlinNvfp4Gemm(mat_a.data_ptr(), mat_b.data_ptr(), out.data_ptr(), cTmpPtr, scale_b.data_ptr(),
        weight_global_scale.data_ptr(), m, n, k, workspacePtr, /*num_groups=*/k / 16, /*group_size=*/16, fp32Reduce,
        stream, tensorrt_llm::common::getEnvEnablePDL(), static_cast<int>(launch_hint));
    return out;
}

} // namespace

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "marlin_nvfp4_w4a16_gemm(Tensor mat_a, Tensor mat_b, Tensor scale_b, Tensor weight_global_scale,"
        " int size_n, int size_k, int workspace_mode, int launch_hint) -> Tensor");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("marlin_nvfp4_w4a16_gemm", &TRTLLM_MV2_NS::marlin_nvfp4_w4a16_gemm);
}
