/*
 * Copyright (c) 2022-2026, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <cuda_runtime.h>

namespace tensorrt_llm::kernels
{

/**
 * Fused sharded QK RMSNorm using NVLink-based cross-GPU barrier.
 *
 * Normalizes Q and K (both column-sharded across tensor-parallel ranks) using a
 * single NVLink barrier on packed [N_tokens, 2] variance scalars rather than two
 * NCCL all-reduces. The barrier uses the SyncComm/Barrier infrastructure from
 * allReduceFusionKernels.cu (st.global.release.sys / ld.global.acquire.sys).
 *
 * Workspace layout (from get_allreduce_workspace):
 *   workspace[0..tp-1]      : comm buffers — we reuse the first tp entries to
 *                             store packed float2 [q_sumsq, k_sumsq] per token
 *   workspace[tp..2*tp-1]   : barrier_flags (one int per rank per block)
 *   workspace[3*tp]         : flag buffer [counter, sync_flag, lamport_flag]
 *
 * @param q_in         Input Q tensor pointer (fp16 or bf16); row stride = q_stride elements
 * @param k_in         Input K tensor pointer; row stride = k_stride elements
 * @param q_out        Output Q tensor (packed, row stride = local_q_dim)
 * @param k_out        Output K tensor (packed, row stride = local_k_dim)
 * @param weight_q     RMSNorm weight for Q [local_q_dim] (float32)
 * @param weight_k     RMSNorm weight for K [local_k_dim] (float32)
 * @param workspace    Void** workspace pointer array from get_allreduce_workspace
 * @param n_tokens     Number of tokens (rows) to process
 * @param local_q_dim  Local head dimension for Q (= full_q_dim / world_size)
 * @param local_k_dim  Local head dimension for K (= full_k_dim / world_size)
 * @param q_stride     Row stride of q_in in elements (>= local_q_dim; e.g. = local_q_dim
 *                     for packed Q, = fused_qkv_dim for a view into fused Q+K+V output).
 *                     Must have last-dim-contiguous (element stride of 1).
 * @param k_stride     Row stride of k_in in elements.
 * @param world_size   Tensor parallel size (number of ranks)
 * @param rank         This rank's index [0, world_size)
 * @param eps          RMSNorm epsilon for numerical stability
 * @param is_bf16      true for bfloat16, false for float16
 * @param stream       CUDA stream
 */
void launchShardedQKNormKernel(void* q_in, void* k_in, void* q_out, void* k_out, void* weight_q, void* weight_k,
    void** workspace, int n_tokens, int local_q_dim, int local_k_dim, int q_stride, int k_stride, int world_size,
    int rank, float eps, bool is_bf16, cudaStream_t stream);

} // namespace tensorrt_llm::kernels
