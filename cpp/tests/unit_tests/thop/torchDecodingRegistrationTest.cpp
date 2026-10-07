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

namespace
{

constexpr TorchRegistration kRegistrations[] = {{"trtllm::build_dynamic_tree_op", c10::DispatchKey::CUDA, true},
    {"trtllm::compressor_paged_kv_compress", c10::DispatchKey::CUDA, true},
    {"trtllm::compressor_postprocess_scatter", c10::DispatchKey::CUDA, true},
    {"trtllm::compressor_prefill_reduction", c10::DispatchKey::CUDA, true},
    {"trtllm::fused_compute_probs_from_logits", c10::DispatchKey::CUDA, true},
    {"trtllm::fused_sample_from_logits", c10::DispatchKey::CUDA, true},
    {"trtllm::fused_sample_from_logits_with_probs", c10::DispatchKey::CUDA, true},
    {"trtllm::inplace_slice_copy", c10::DispatchKey::CUDA, true},
    {"trtllm::logits_bitmask", c10::DispatchKey::CUDA, true},
    {"trtllm::lora_group_gemm_param_fill_row_reorder_fusion", c10::DispatchKey::CUDA, true},
    {"trtllm::lora_grouped_gemm", c10::DispatchKey::CUDA, true},
    {"trtllm::lora_grouped_gemm_cuda_graph", c10::DispatchKey::CUDA, true},
    {"trtllm::lora_grouped_gemm_supports_fp8", c10::DispatchKey::CompositeImplicitAutograd, true},
    {"trtllm::mhc_big_fuse", c10::DispatchKey::CUDA, true}, {"trtllm::mhc_fused_hc", c10::DispatchKey::CUDA, true},
    {"trtllm::mhc_fused_hc_mma_enabled", c10::DispatchKey::CompositeExplicitAutograd, true},
    {"trtllm::mhc_gemm_sqrsum_fma", c10::DispatchKey::CUDA, true},
    {"trtllm::mhc_hc_head_apply", c10::DispatchKey::CUDA, true},
    {"trtllm::mhc_post_mapping", c10::DispatchKey::CUDA, true},
    {"trtllm::mtp_prepare_drafter_inputs_op", c10::DispatchKey::CUDA, true},
    {"trtllm::mtp_relaxed_acceptance_op", c10::DispatchKey::CUDA, true},
    {"trtllm::mtp_sampling_and_accepted_draft_tokens_op", c10::DispatchKey::CUDA, true},
    {"trtllm::mtp_update_hidden_states_op", c10::DispatchKey::CUDA, true},
    {"trtllm::verify_dynamic_tree_greedy_out_packed_op", c10::DispatchKey::CUDA, true},
    {"trtllm::verify_dynamic_tree_rejection_out_op", c10::DispatchKey::CUDA, true}};

} // namespace

TEST(TorchDecodingRegistration, SchemasAndDispatchKernels)
{
    expectTorchRegistrations(kRegistrations);
}

TEST(TorchDecodingRegistration, ExcludesUnrelatedOperators)
{
    expectOnlyTorchFamily(kRegistrations);
}
