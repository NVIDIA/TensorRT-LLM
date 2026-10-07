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

#include "torchFamilyRegistration.h"

#include <torch/custom_class_detail.h>

namespace
{

constexpr TorchRegistration kRegistrations[] = {{"trtllm::default_moe_routing_op", c10::DispatchKey::CUDA, true},
    {"trtllm::dsv3_router_gemm_op", c10::DispatchKey::CUDA, true},
    {"trtllm::fp8_per_tensor_scale_moe_runner", c10::DispatchKey::CUDA, true},
    {"trtllm::fused_topk_softmax", c10::DispatchKey::CUDA, true},
    {"trtllm::gate_forward", c10::DispatchKey::CUDA, true},
    {"trtllm::get_moe_commworkspace_size_per_rank", c10::DispatchKey::CompositeExplicitAutograd, true},
    {"trtllm::get_moe_prepare_workspace_size_per_rank", c10::DispatchKey::CompositeExplicitAutograd, true},
    {"trtllm::kimi_k3_noaux_tc_mxfp8_quant", c10::DispatchKey::CUDA, true},
    {"trtllm::llama4_bf16_bf16_gemm", c10::DispatchKey::CUDA, true},
    {"trtllm::llama4_fp8_bf16_gemm", c10::DispatchKey::CUDA, true},
    {"trtllm::llama4_fp8_fp8_gemm_swiglu", c10::DispatchKey::CUDA, true},
    {"trtllm::llama4_moe_tp8ep1_min_latency", c10::DispatchKey::CUDA, true},
    {"trtllm::marlin_nvfp4_moe_gemm", c10::DispatchKey::CUDA, true},
    {"trtllm::megamoe_prepare", c10::DispatchKey::CUDA, true},
    {"trtllm::memset_expert_ids", c10::DispatchKey::CUDA, true},
    {"trtllm::migrate_to_host_accessible", c10::DispatchKey::CUDA, true},
    {"trtllm::mnnvl_moe_alltoallv_prepare_without_allgather", c10::DispatchKey::CUDA, true},
    {"trtllm::moe_a2a_cft_initialize", c10::DispatchKey::CUDA, true},
    {"trtllm::moe_a2a_combine", c10::DispatchKey::CUDA, true},
    {"trtllm::moe_a2a_dispatch", c10::DispatchKey::CUDA, true},
    {"trtllm::moe_a2a_get_aux_data_size", c10::DispatchKey::CompositeImplicitAutograd, true},
    {"trtllm::moe_a2a_get_combine_payload_tensor", c10::DispatchKey::CUDA, true},
    {"trtllm::moe_a2a_initialize", c10::DispatchKey::CUDA, true},
    {"trtllm::moe_a2a_sanitize_expert_ids", c10::DispatchKey::CUDA, true},
    {"trtllm::moe_a2a_set_warmup", c10::DispatchKey::CompositeImplicitAutograd, true},
    {"trtllm::moe_align_block_size", c10::DispatchKey::CUDA, true}, {"trtllm::moe_comm", c10::DispatchKey::CUDA, true},
    {"trtllm::moe_finalize_scale_op", c10::DispatchKey::CUDA, true}, {"trtllm::moe_gelu", c10::DispatchKey::CUDA, true},
    {"trtllm::moe_hierarchical_statistic_local_device", c10::DispatchKey::CUDA, true},
    {"trtllm::moe_hierarchical_statistic_update", c10::DispatchKey::CUDA, true},
    {"trtllm::moe_initialize_workspace", c10::DispatchKey::CUDA, true},
    {"trtllm::moe_load_balance_routing", c10::DispatchKey::CUDA, true},
    {"trtllm::moe_load_balance_set_cpu_stage", c10::DispatchKey::CompositeExplicitAutograd, true},
    {"trtllm::moe_load_balance_statistic", c10::DispatchKey::CUDA, true},
    {"trtllm::moe_load_balance_wait_gpu_stage", c10::DispatchKey::CompositeExplicitAutograd, true},
    {"trtllm::moe_output_memset_from_expert_counts_inplace", c10::DispatchKey::CUDA, true},
    {"trtllm::moe_output_memset_inplace", c10::DispatchKey::CUDA, true},
    {"trtllm::moe_permute", c10::DispatchKey::CUDA, true}, {"trtllm::moe_permute_op", c10::DispatchKey::CUDA, true},
    {"trtllm::moe_sort", c10::DispatchKey::CUDA, true}, {"trtllm::moe_swiglu", c10::DispatchKey::CUDA, true},
    {"trtllm::moe_swiglu_nvfp4_quantize", c10::DispatchKey::CUDA, true},
    {"trtllm::moe_topk_sort", c10::DispatchKey::CUDA, true}, {"trtllm::moe_unpermute", c10::DispatchKey::CUDA, true},
    {"trtllm::moe_unpermute_inplace", c10::DispatchKey::CUDA, true},
    {"trtllm::noaux_tc_op", c10::DispatchKey::CUDA, true},
    {"trtllm::renorm_moe_routing_op", c10::DispatchKey::CUDA, true},
    {"trtllm::set_fine_grained_sync_disabled_override", c10::DispatchKey::CompositeImplicitAutograd, true},
    {"trtllm::set_moe_max_usable_sm_count", c10::DispatchKey::CompositeExplicitAutograd, true},
    {"trtllm::shuffle_matrix", c10::DispatchKey::CompositeImplicitAutograd, true}};

} // namespace

TEST(TorchMoeRegistration, SchemasAndDispatchKernels)
{
    expectTorchRegistrations(kRegistrations);
}

TEST(TorchMoeRegistration, ExcludesUnrelatedOperators)
{
    expectOnlyTorchFamily(kRegistrations);
}

TEST(TorchMoeRegistration, CustomClasses)
{
    for (auto const* name : {"FusedMoeRunner", "FP8BlockScaleMoERunner", "FP4BlockScaleMoERunner",
             "FP8FP4BlockScaleMoERunner", "Bf16MxE2m1BlockScaleMoERunner", "MxE4m3MxE2m1BlockScaleMoERunner"})
    {
        EXPECT_NE(torch::getCustomClass(std::string("__torch__.torch.classes.trtllm.") + name), nullptr) << name;
    }
}
