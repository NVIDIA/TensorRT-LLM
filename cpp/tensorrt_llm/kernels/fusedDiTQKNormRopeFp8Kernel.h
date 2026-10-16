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

#pragma once

#include "tensorrt_llm/common/config.h"

#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{

// Full-dim (cross-head) QK RMSNorm + RoPE + per-tensor FP8 (e4m3) quantization of a packed QKV
// tensor, for diffusion transformers (Wan). Q and K are normalized over num_heads * head_dim,
// rotated, rounded to bf16 and multiplied by quant_mul[0] / quant_mul[1]; V is rounded and
// multiplied by quant_mul[2]. Stores saturate to the finite e4m3 range.
//
// qkv:      bf16 [num_tokens, 3 * num_heads * head_dim]
// q8/k8/v8: e4m3 [num_tokens, num_heads * head_dim]
// cos/sin:  fp32 [cos_rows, head_dim] (shared across heads) or [cos_rows, num_heads * head_dim]
// cos_seq_per_batch: 0 when cos_rows == num_tokens, else cos_rows (broadcast over the batch).
// Requires SM90+, head_dim 128, num_heads <= 64 (see fusedDiTQKNormRopeFp8Supported) and
// 16-byte aligned tensors.
void launchFusedDiTQKNormRopeFullDimFp8(void const* qkv, void* q8, void* k8, void* v8, int num_tokens, int num_heads,
    int head_dim, float eps, void const* q_weight, void const* k_weight, float const* cos_emb, float const* sin_emb,
    bool interleave, bool per_head_cos, int cos_seq_per_batch, float const* quant_mul, cudaStream_t stream);

// Whether launchFusedDiTQKNormRopeFullDimFp8 supports this layout on an SM `sm` GPU.
bool fusedDiTQKNormRopeFp8Supported(int num_heads, int head_dim, int sm);

// Number of fp32 partial maxima launchDiTVAmax needs in its workspace on `device`.
int getDiTVAmaxWorkspaceSize(int device);

// |V| maximum of the packed QKV tensor into amax[0]. When mul_v / scale_v are not null, also
// writes scale_v[0] = max(amax, 1e-12) / 448 and mul_v[0] = 1 / scale_v[0].
// workspace: fp32 [getDiTVAmaxWorkspaceSize(device)].
void launchDiTVAmax(void const* qkv, int num_tokens, int num_heads, int head_dim, float* workspace, int workspace_size,
    float* amax, float* mul_v, float* scale_v, cudaStream_t stream);

// scale_v[0] = max(amax[0], 1e-12) / 448 and mul_v[0] = 1 / scale_v[0], from an amax already
// reduced elsewhere (for example across sequence-parallel ranks).
void launchDiTVScaleFromAmax(float const* amax, float* mul_v, float* scale_v, cudaStream_t stream);

} // namespace kernels

TRTLLM_NAMESPACE_END
