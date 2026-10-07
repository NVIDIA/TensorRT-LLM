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

constexpr TorchRegistration kRegistrations[]
    = {{"tensorrt_llm::pack_fmha_mask_by_input", c10::DispatchKey::CompositeImplicitAutograd, true},
        {"tensorrt_llm::pack_fmha_mask_by_type", c10::DispatchKey::CompositeImplicitAutograd, true},
        {"tensorrt_llm::relative_attention_bias", c10::DispatchKey::CompositeImplicitAutograd, true},
        {"trtllm::attention_supports_nvfp4_output", c10::DispatchKey::CompositeImplicitAutograd, true},
        {"trtllm::attn_res_add_rmsnorm_fwd", c10::DispatchKey::CUDA, true},
        {"trtllm::attn_res_add_rmsnorm_persistent_fwd", c10::DispatchKey::CUDA, true},
        {"trtllm::attn_res_fwd", c10::DispatchKey::CUDA, true},
        {"trtllm::attn_res_rmsnorm_fwd", c10::DispatchKey::CUDA, true},
        {"trtllm::convert_req_index_to_global", c10::DispatchKey::CUDA, true},
        {"trtllm::convert_req_index_to_global_grouped", c10::DispatchKey::CUDA, true},
        {"trtllm::deepseek_v4_compute_sliding_block_tables", c10::DispatchKey::CUDA, true},
        {"trtllm::deepseek_v4_compute_sliding_block_tables_with_scratch", c10::DispatchKey::CUDA, true},
        {"trtllm::fused_inv_rope_fp8_quant_vllm_port", c10::DispatchKey::CUDA, true},
        {"trtllm::fused_qk_norm_rope", c10::DispatchKey::CUDA, true},
        {"trtllm::fused_qk_norm_rope_to_fp8", c10::DispatchKey::CUDA, true},
        {"trtllm::indexer_k_cache_gather_op", c10::DispatchKey::CUDA, true},
        {"trtllm::indexer_k_cache_scatter_op", c10::DispatchKey::CUDA, true},
        {"trtllm::indexer_topk_decode", c10::DispatchKey::CUDA, true},
        {"trtllm::indexer_topk_prefill", c10::DispatchKey::CUDA, true},
        {"trtllm::kda_decode", c10::DispatchKey::CUDA, true},
        {"trtllm::load_chunked_kv_cache_for_mla", c10::DispatchKey::CUDA, true},
        {"trtllm::load_paged_kv_cache_for_mla", c10::DispatchKey::CUDA, true},
        {"trtllm::merge_chunked_attention_for_mla", c10::DispatchKey::CUDA, true},
        {"trtllm::minimax_m3_fp8_indexer_qk_norm_rope", c10::DispatchKey::CUDA, true},
        {"trtllm::minimax_m3_fp8_qk_norm_rope_kv_insert", c10::DispatchKey::CUDA, true},
        {"trtllm::minimax_m3_fp8_qkv_indexer_norm_rope_kv_insert", c10::DispatchKey::CUDA, true},
        {"trtllm::minimax_m3_nvfp4_qkv_indexer_norm_rope_kv_insert", c10::DispatchKey::CUDA, true},
        {"trtllm::minimax_m3_select_blocks", c10::DispatchKey::CUDA, true},
        {"trtllm::mla_rope_append_paged_kv_assign_q", c10::DispatchKey::CUDA, true},
        {"trtllm::mla_rope_generation", c10::DispatchKey::CUDA, true},
        {"trtllm::mla_rope_inplace", c10::DispatchKey::CUDA, true},
        {"trtllm::nvfp4_mla_context_kv_cache_gather", c10::DispatchKey::CUDA, true},
        {"trtllm::nvfp4_mla_context_kv_cache_gather_direct", c10::DispatchKey::CUDA, true},
        {"trtllm::nvfp4_mla_kv_cache_gather", c10::DispatchKey::CUDA, true},
        {"trtllm::nvfp4_mla_kv_cache_gather_direct", c10::DispatchKey::CUDA, true},
        {"trtllm::sparse_kv_cache_compact_layers", c10::DispatchKey::CUDA, true}};

} // namespace

TEST(TorchAttentionRegistration, SchemasAndDispatchKernels)
{
    expectTorchRegistrations(kRegistrations);
}

TEST(TorchAttentionRegistration, ExcludesUnrelatedOperators)
{
    expectOnlyTorchFamily(kRegistrations);
}
