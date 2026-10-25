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
// Per-forward batch metadata for this target's kernels (step_meta.cu).
#pragma once

#include "../common.h"
#include "sampling.h"

namespace glm {
namespace k {

// cu [B + 1], token_seq [T], has_state [B], replay [B] (int32 outputs) and tiles [B] (optional:
// {first token, tokens} of every sequence) from the runtime's seq_lens [B] (int32), has_initial
// [>= num_contexts] (bool, may be null), slots [B] (int32) and prev_accepted (int32, slot s at
// s * prev_stride; may be null). Batch rows: contexts first.
void step_meta(int* cu, int* token_seq, int* has_state, int* replay, int2* tiles,
               const int* seq_lens, const bool* has_initial, const int* slots,
               const int* prev_accepted, int64_t prev_stride, int B, int num_contexts, int T,
               cudaStream_t st);

// The temperature-only head's k::SampleParams rows [0, S) from the runtime's per-row temperatures
// (fp32; below greedy_below: 0, greedy), seeds and offsets (int64, the Philox stream of the row and
// step); the other fields keep what the buffer holds.
void spec_sample_params(SampleParams* out, const float* temps, const int64_t* seeds,
                        const int64_t* offsets, int S, float greedy_below, cudaStream_t st);

// What the one-model MTP worker hands the runtime after a step, from the head's tokens [S] (int32),
// the drafts [B, D] and the accepted drafts [B] of every batch row b:
// new_tokens [B, D + 1] = the row's head tokens (rows row0[b] + j, or (D + 1) b + j without row0;
// clamped below S), lens [B] = accepted[b] + 1, next [B, D + 1] = the last accepted token of batch
// row batch_idx[b] (identity when null), then the row's drafts; and the KDA replay record of the
// generation rows (b >= num_contexts) that are not dummy[b] (bool, may be null):
// prev_accepted[slots[b] * prev_stride] = accepted[b], the drafts the slot's next verify replays.
void spec_accept(int* new_tokens, int* lens, int* next, const int* tokens, const int* drafts,
                 const int* accepted, const int* row0, const int* batch_idx, int S, const int* slots,
                 const bool* dummy, int* prev_accepted, int64_t prev_stride, int B,
                 int num_contexts, int D, cudaStream_t st);

}  // namespace k
}  // namespace glm
