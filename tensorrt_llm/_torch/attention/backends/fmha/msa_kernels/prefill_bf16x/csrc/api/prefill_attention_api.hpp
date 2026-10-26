// SPDX-FileCopyrightText: Copyright (c) 2026 MiniMax
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
// SPDX-License-Identifier: MIT

#pragma once

#include <torch/extension.h>

namespace fmha_sm100::prefill_bf16x {

void prefill_run(torch::Tensor q,
                 torch::Tensor packed_k,
                 torch::Tensor packed_v,
                 torch::Tensor k_scale,
                 torch::Tensor v_scale,
                 torch::Tensor kv_indices,
                 c10::optional<torch::Tensor> kv_indptr,
                 torch::Tensor cu_seqlens_q,
                 torch::Tensor cu_seqlens_k,
                 c10::optional<torch::Tensor> seqused_k,
                 torch::Tensor k2q_row_ptr,
                 torch::Tensor qsplit_indices,
                 torch::Tensor scheduler_metadata,
                 torch::Tensor work_count,
                 torch::Tensor o_partial,
                 torch::Tensor lse_partial,
                 c10::optional<torch::Tensor> k_global_scale,
                 c10::optional<torch::Tensor> v_global_scale,
                 double softmax_scale,
                 double output_scale,
                 c10::optional<torch::Tensor> combine_tile_counters = c10::nullopt);

// The block-scale shift this extension was built with.
int64_t block_scale_shift();
// The data path this extension was built with: 0 = E4M3, 1 = BF16.
int64_t data_path();

// Memory locality domains of a device (Rubin uGPU; one on other GPUs).
int64_t locality_domain_count(int64_t device);
// Allocation granularity of domain-localized memory, the page at which a striped buffer
// alternates domains.
int64_t locality_page_bytes(int64_t device);
// The locality domain of every SM id (%smid) as an int32 CPU tensor, -1 where unknown.
torch::Tensor sm_locality_domains(int64_t device);
// A uint8 device buffer whose pages alternate between the locality domains: page p (of
// locality_page_bytes) lives in domain p % locality_domain_count. bytes must be a multiple of
// the page size times the domain count.
torch::Tensor alloc_locality_striped(int64_t bytes, int64_t device);

}  // namespace fmha_sm100::prefill_bf16x
