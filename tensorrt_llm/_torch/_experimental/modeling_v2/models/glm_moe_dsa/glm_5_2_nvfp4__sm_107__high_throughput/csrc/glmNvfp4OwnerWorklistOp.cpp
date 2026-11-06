/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "glm_moe_dsa/glm_5_2_nvfp4__sm_107__high_throughput/csrc/glmNvfp4OwnerWorklist.h"
#include "tensorrt_llm/modeling_v2/common/namespace.h"
#include "tensorrt_llm/modeling_v2/common/torchLibrary.h"
#include "tensorrt_llm/runtime/torchUtils.h"

#include <ATen/MemoryOverlap.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace th = torch;
namespace tk = TRTLLM_MV2_NS::kernels;

TRTLLM_MV2_NAMESPACE_BEGIN

namespace torch_ext
{
namespace
{

struct NvFp4WorklistPools
{
    uint8_t const* data;
    __nv_fp8_e4m3 const* scales;
};

void checkWorklistDisjoint(std::initializer_list<th::Tensor const*> tensors)
{
    for (auto first = tensors.begin(); first != tensors.end(); ++first)
    {
        for (auto second = first + 1; second != tensors.end(); ++second)
        {
            at::assert_no_overlap(**first, **second);
        }
    }
}

NvFp4WorklistPools resolveNvFp4WorklistPools(
    th::Tensor const& hostPoolPointers, th::Tensor const& hostPoolMapping, int64_t layerIdx, int64_t numPoolTokens)
{
    TORCH_CHECK(hostPoolPointers.device().is_cpu() && hostPoolPointers.scalar_type() == th::kInt64
            && hostPoolPointers.is_contiguous() && hostPoolPointers.dim() == 3 && hostPoolPointers.size(1) == 2
            && hostPoolPointers.size(2) == 2,
        "host_pool_pointers must be contiguous CPU int64 [num_pools, 2, 2]");
    TORCH_CHECK(hostPoolMapping.device().is_cpu() && hostPoolMapping.scalar_type() == th::kInt32
            && hostPoolMapping.is_contiguous() && hostPoolMapping.dim() == 2 && hostPoolMapping.size(1) == 2,
        "host_pool_mapping must be contiguous CPU int32 [num_layers, 2]");
    TORCH_CHECK(layerIdx >= 0 && layerIdx < hostPoolMapping.size(0), "layer_idx is out of range");
    TORCH_CHECK(numPoolTokens > 0 && numPoolTokens <= std::numeric_limits<int32_t>::max(),
        "num_pool_tokens must be a positive int32 value");
    int32_t const poolIdx = hostPoolMapping.data_ptr<int32_t>()[layerIdx * 2];
    TORCH_CHECK(poolIdx >= 0 && poolIdx < hostPoolPointers.size(0), "mapped pool index is out of range");
    auto const* pointers = hostPoolPointers.data_ptr<int64_t>() + static_cast<int64_t>(poolIdx) * 4;
    int64_t const data = pointers[0];
    int64_t const scales = pointers[1];
    TORCH_CHECK(data != 0 && scales != 0 && data % 16 == 0 && scales % 8 == 0,
        "NVFP4 worklist requires aligned non-null data and scale pools");
    TORCH_CHECK(data != scales && scales != data + 320, "NVFP4 worklist requires separate data and scale pools");
    return {reinterpret_cast<uint8_t const*>(data), reinterpret_cast<__nv_fp8_e4m3 const*>(scales)};
}

void checkWorklistOutputs(th::Tensor const& output, th::Tensor const& compactIndices, th::Tensor const& ownerCount,
    th::Tensor const& globalDequantScale, c10::Device device)
{
    TORCH_CHECK(output.is_cuda() && output.device() == device && output.scalar_type() == th::kFloat8_e4m3fn
            && output.is_contiguous() && output.dim() == 3 && output.size(0) > 0 && output.size(1) > 0
            && output.size(2) == 576 && output.size(0) * output.size(1) <= std::numeric_limits<int32_t>::max(),
        "worklist output must be contiguous float8_e4m3fn [rows, top_k, 576] on the worklist device");
    TORCH_CHECK(compactIndices.device() == device && compactIndices.scalar_type() == th::kInt32
            && compactIndices.is_contiguous() && compactIndices.dim() == 2 && compactIndices.size(0) == output.size(0)
            && compactIndices.size(1) == output.size(1),
        "compact_indices must be contiguous int32 [rows, top_k]");
    TORCH_CHECK(ownerCount.device() == device && ownerCount.scalar_type() == th::kInt32 && ownerCount.is_contiguous()
            && ownerCount.numel() == 1,
        "owner_count must contain exactly one contiguous int32 value");
    TORCH_CHECK(globalDequantScale.device() == device && globalDequantScale.scalar_type() == th::kFloat32
            && globalDequantScale.is_contiguous() && globalDequantScale.numel() == 1,
        "global_dequant_scale must contain exactly one contiguous float32 value");
}

} // namespace

void glmRubinNvfp4WorklistPipelined(th::Tensor const& hostPoolPointers, th::Tensor const& hostPoolMapping,
    th::Tensor const& memberIndices, th::Tensor const& queryReqIndices, th::Tensor& output, th::Tensor& compactIndices,
    th::Tensor& ownerTable, th::Tensor& memberOwners, th::Tensor& ownerCount, th::Tensor& ownerState,
    th::Tensor const& globalDequantScale, int64_t layerIdx, int64_t numRequests, int64_t numPoolTokens)
{
    auto const pools = resolveNvFp4WorklistPools(hostPoolPointers, hostPoolMapping, layerIdx, numPoolTokens);
    TORCH_CHECK(memberIndices.is_cuda() && memberIndices.scalar_type() == th::kInt32 && memberIndices.is_contiguous()
            && memberIndices.dim() == 3 && memberIndices.size(0) > 0
            && memberIndices.size(0) <= std::numeric_limits<int32_t>::max(),
        "member_indices must be contiguous CUDA int32 [members, rows, top_k]");
    auto const device = memberIndices.device();
    checkWorklistOutputs(output, compactIndices, ownerCount, globalDequantScale, device);
    TORCH_CHECK(memberIndices.size(1) == output.size(0) && memberIndices.size(2) == output.size(1),
        "member_indices rows and top_k must match the output");
    int64_t const numPairs = output.size(0) * output.size(1);
    int64_t const numMembers = memberIndices.size(0);
    TORCH_CHECK(numRequests > 0 && numRequests <= std::numeric_limits<int32_t>::max(),
        "worklist num_requests must be a positive int32 value");
    TORCH_CHECK(queryReqIndices.device() == device && queryReqIndices.scalar_type() == th::kInt32
            && queryReqIndices.is_contiguous() && queryReqIndices.dim() == 1
            && queryReqIndices.size(0) == output.size(0),
        "query_req_indices must be contiguous int32 with one request ID per active row");
    TORCH_CHECK(ownerTable.device() == device && ownerTable.scalar_type() == th::kInt64 && ownerTable.is_contiguous()
            && ownerTable.dim() == 2 && ownerTable.size(0) >= numRequests && ownerTable.size(1) > 1
            && ownerTable.size(1) <= std::numeric_limits<int32_t>::max()
            && (ownerTable.size(1) & (ownerTable.size(1) - 1)) == 0
            && reinterpret_cast<uintptr_t>(ownerTable.data_ptr()) % 16 == 0,
        "owner_table must be 16-byte aligned contiguous int64 [capacity >= num_requests, power-of-two int32 "
        "slots > 1]");
    TORCH_CHECK(ownerState.device() == device && ownerState.scalar_type() == th::kInt32 && ownerState.is_contiguous()
            && ownerState.numel() == 2,
        "owner_state must contain exactly two contiguous int32 values");
    TORCH_CHECK(memberOwners.device() == device && memberOwners.scalar_type() == th::kInt32
            && memberOwners.is_contiguous() && memberOwners.dim() == 3 && memberOwners.size(0) >= numMembers
            && memberOwners.size(1) >= numPairs && memberOwners.size(2) == 2,
        "member_owners must be contiguous int32 [>= members, >= rows * top_k, 2]");
    checkWorklistDisjoint({&memberIndices, &queryReqIndices, &output, &compactIndices, &ownerTable, &memberOwners,
        &ownerCount, &ownerState, &globalDequantScale});
    c10::cuda::CUDAGuard const deviceGuard(device);
    auto stream = at::cuda::getCurrentCUDAStream(device.index()).stream();
    tk::invokeGlmRubinNvfp4WorklistPipelined(pools.data, pools.scales, memberIndices.data_ptr<int32_t>(),
        static_cast<int32_t>(numMembers), queryReqIndices.data_ptr<int32_t>(),
        reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr()), compactIndices.data_ptr<int32_t>(),
        ownerTable.data_ptr<int64_t>(), memberOwners.data_ptr<int32_t>(), memberOwners.size(1),
        ownerCount.data_ptr<int32_t>(), ownerState.data_ptr<int32_t>(), globalDequantScale.data_ptr<float>(),
        static_cast<int32_t>(output.size(0)), static_cast<int32_t>(output.size(1)), static_cast<int32_t>(numRequests),
        static_cast<int32_t>(ownerTable.size(1)), numPoolTokens, stream);
}

void glmRubinNvfp4WorklistMember(th::Tensor const& hostPoolPointers, th::Tensor const& hostPoolMapping,
    th::Tensor const& memberOwners, th::Tensor& output, th::Tensor& compactIndices, th::Tensor const& ownerCount,
    th::Tensor const& globalDequantScale, int64_t layerIdx, int64_t numPoolTokens)
{
    auto const pools = resolveNvFp4WorklistPools(hostPoolPointers, hostPoolMapping, layerIdx, numPoolTokens);
    TORCH_CHECK(memberOwners.is_cuda() && memberOwners.scalar_type() == th::kInt32 && memberOwners.is_contiguous()
            && memberOwners.dim() == 2 && memberOwners.size(1) == 2
            && reinterpret_cast<uintptr_t>(memberOwners.data_ptr()) % 8 == 0,
        "member_owners must be 8-byte aligned contiguous CUDA int32 [capacity, 2]");
    auto const device = memberOwners.device();
    checkWorklistOutputs(output, compactIndices, ownerCount, globalDequantScale, device);
    TORCH_CHECK(memberOwners.size(0) >= output.size(0) * output.size(1),
        "member_owners must hold an entry for every active pair");
    checkWorklistDisjoint({&memberOwners, &output, &compactIndices, &ownerCount, &globalDequantScale});
    c10::cuda::CUDAGuard const deviceGuard(device);
    auto stream = at::cuda::getCurrentCUDAStream(device.index()).stream();
    tk::invokeGlmRubinNvfp4WorklistMember(pools.data, pools.scales, memberOwners.data_ptr<int32_t>(),
        ownerCount.data_ptr<int32_t>(), reinterpret_cast<__nv_fp8_e4m3*>(output.data_ptr()),
        compactIndices.data_ptr<int32_t>(), globalDequantScale.data_ptr<float>(), static_cast<int32_t>(output.size(0)),
        static_cast<int32_t>(output.size(1)), stream);
}

} // namespace torch_ext

TRTLLM_MV2_NAMESPACE_END

TRTLLM_MV2_LIBRARY_FRAGMENT(m)
{
    m.def(
        "glm_rubin_nvfp4_mla_gather_worklist_pipelined(Tensor host_pool_pointers, Tensor host_pool_mapping, "
        "Tensor member_indices, Tensor query_req_indices, Tensor(a!) output, Tensor(b!) compact_indices, "
        "Tensor(c!) owner_table, Tensor(d!) member_owners, Tensor(e!) owner_count, Tensor(f!) owner_state, "
        "Tensor global_dequant_scale, int layer_idx, int num_requests, int num_pool_tokens) -> ()");
    m.def(
        "glm_rubin_nvfp4_mla_gather_worklist_member(Tensor host_pool_pointers, Tensor host_pool_mapping, "
        "Tensor member_owners, Tensor(a!) output, Tensor(b!) compact_indices, Tensor owner_count, "
        "Tensor global_dequant_scale, int layer_idx, int num_pool_tokens) -> ()");
}

TRTLLM_MV2_LIBRARY_IMPL(CUDA, m)
{
    m.impl("glm_rubin_nvfp4_mla_gather_worklist_pipelined", &TRTLLM_MV2_NS::torch_ext::glmRubinNvfp4WorklistPipelined);
    m.impl("glm_rubin_nvfp4_mla_gather_worklist_member", &TRTLLM_MV2_NS::torch_ext::glmRubinNvfp4WorklistMember);
}
