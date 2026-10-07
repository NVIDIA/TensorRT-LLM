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

constexpr TorchRegistration kRegistrations[] = {{"trtllm::allgather", c10::DispatchKey::CUDA, true},
    {"trtllm::allgather_list", c10::DispatchKey::CUDA, true},
    {"trtllm::allgather_list_pg", c10::DispatchKey::CUDA, true}, {"trtllm::allgather_pg", c10::DispatchKey::CUDA, true},
    {"trtllm::allocate_output", c10::DispatchKey::CUDA, true},
    {"trtllm::allocate_output_with_nccl_window", c10::DispatchKey::CUDA, true},
    {"trtllm::allreduce", c10::DispatchKey::CUDA, true}, {"trtllm::allreduce_pg", c10::DispatchKey::CUDA, true},
    {"trtllm::alltoall_helix", c10::DispatchKey::CUDA, true},
    {"trtllm::alltoall_helix_native", c10::DispatchKey::CUDA, true},
    {"trtllm::autotuned_allreduce", c10::DispatchKey::CUDA, true},
    {"trtllm::clear_allreduce_tactic_cache", c10::DispatchKey::CompositeExplicitAutograd, true},
    {"trtllm::create_userbuffers_tensor", c10::DispatchKey::CompositeImplicitAutograd, true},
    {"trtllm::helix_post_process", c10::DispatchKey::CUDA, true},
    {"trtllm::helix_post_process_native", c10::DispatchKey::CUDA, true},
    {"trtllm::initialize_helix_workspace", c10::DispatchKey::CUDA, true},
    {"trtllm::initialize_static_lowprecision_buffers", c10::DispatchKey::CPU, true},
    {"trtllm::is_nccl_window_buffer", c10::DispatchKey::CUDA, true},
    {"trtllm::minimax_allreduce_rms", c10::DispatchKey::CUDA, true},
    {"trtllm::minimax_allreduce_rms_qk", c10::DispatchKey::CUDA, true},
    {"trtllm::mnnvl_fusion_allreduce", c10::DispatchKey::CUDA, true},
    {"trtllm::moe_allreduce", c10::DispatchKey::CUDA, true},
    {"trtllm::moe_finalize_allreduce", c10::DispatchKey::CUDA, true},
    {"trtllm::preallocate_nccl_window_buffer", c10::DispatchKey::CUDA, true},
    {"trtllm::reducescatter", c10::DispatchKey::CUDA, true},
    {"trtllm::reducescatter_list", c10::DispatchKey::CUDA, true},
    {"trtllm::reducescatter_list_pg", c10::DispatchKey::CUDA, true},
    {"trtllm::reducescatter_pg", c10::DispatchKey::CUDA, true},
    {"trtllm::register_allreduce_tactic", c10::DispatchKey::CUDA, true},
    {"trtllm::ulysses_a2a_async_barrier", c10::DispatchKey::CompositeExplicitAutograd, true},
    {"trtllm::ulysses_a2a_async_prepare", c10::DispatchKey::CompositeExplicitAutograd, true},
    {"trtllm::ulysses_a2a_async_push", c10::DispatchKey::CompositeExplicitAutograd, true},
    {"trtllm::ulysses_permute_scatter", c10::DispatchKey::CUDA, true},
    {"trtllm::ulysses_post_unscatter_qkv", c10::DispatchKey::CUDA, true},
    {"trtllm::userbuffers_allreduce_finalize", c10::DispatchKey::CUDA, true},
    {"trtllm::validate_allreduce_tuning_buckets", c10::DispatchKey::CompositeExplicitAutograd, true}};

} // namespace

TEST(TorchDistributedRegistration, SchemasAndDispatchKernels)
{
    expectTorchRegistrations(kRegistrations);
}

TEST(TorchDistributedRegistration, ExcludesUnrelatedOperators)
{
    expectOnlyTorchFamily(kRegistrations);
}

TEST(TorchDistributedRegistration, CustomClasses)
{
    for (auto const* name : {"NcclCommunicatorOp", "SendHandle", "Fp4GemmAllreduceRunner"})
    {
        EXPECT_NE(torch::getCustomClass(std::string("__torch__.torch.classes.trtllm.") + name), nullptr) << name;
    }
}
