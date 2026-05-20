/*
 * Copyright (c) 2025 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "tvm_ffi_utils.h"

using tvm::ffi::Optional;

namespace flashinfer::mamba::checkpointing {

void checkpointing_ssu(
    TensorView state,  // (cache, nheads, dim, dstate)
    TensorView x,  // 4D (batch, T, nheads, dim) or 4D (1, total_tokens, nheads, dim) under varlen
    TensorView
        dt,        // (batch, T, nheads, dim) tie_hdim / (1, total_tokens, nheads, dim) under varlen
    TensorView A,  // (nheads, dim, dstate) tie_hdim
    TensorView B,  // (batch, T, ngroups, dstate) / (1, total_tokens, ngroups, dstate) under varlen
    TensorView C,  // same as B
    TensorView output,  // same layout as x
    // Cache tensors
    TensorView old_x,              // (cache, T, nheads, dim)
    TensorView old_B,              // (cache, 2, T, ngroups, dstate)
    TensorView old_dt,             // (cache, 2, nheads, T) f32
    TensorView old_cumAdt,         // (cache, 2, nheads, T) f32
    TensorView cache_buf_idx,      // (cache,) int32
    TensorView prev_num_accepted,  // (cache,) int32
    // Optional tensors
    Optional<TensorView> D,        // (nheads, dim)
    Optional<TensorView> z,        // same layout as x
    Optional<TensorView> dt_bias,  // (nheads, dim) tie_hdim
    bool dt_softplus,
    Optional<TensorView> state_batch_indices,  // (batch,) int32
    int64_t pad_slot_id,
    Optional<TensorView> state_scale,  // (cache, nheads, dim) f32
    Optional<TensorView> rand_seed,    // single int64
    int64_t d_split,                   // v12 §59: per-head DIM split factor (1, 2, or 4)
    Optional<TensorView> cu_seqlens);  // (batch+1,) int32 — varlen mode

}  // namespace flashinfer::mamba::checkpointing

TVM_FFI_DLL_EXPORT_TYPED_FUNC(checkpointing_ssu,
                              flashinfer::mamba::checkpointing::checkpointing_ssu);
