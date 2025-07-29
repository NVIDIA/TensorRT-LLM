/*
 * Copyright (c) 2019-2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/kvCacheUtils.h"

namespace tensorrt_llm
{
namespace kernels
{
// merged_attn [q_total_len, H=128, D=128] (T)
// merged_softmax_sum [q_total_len, H, 2] (float), the first part is the max value for each
// row of P = QK^T, the second part is the softmax sum
// if merge_op[b] == 0, we just skip this batch, if merge_op[b] == 1, we merge the pre-attn and curr-attn, if
// merge_op[b]
// == 2, we only copy curr_attn and curr_softmax_sum to merged_attn and merged_softmax_sum
template <typename T>
void invokeMergeAttnWithSoftmax(T* merged_attn, float* merged_softmax_stats, T const* pre_attn,
    float const* pre_softmax_stats, T const* curr_attn, float const* curr_softmax_stats, float bmm1_scale,
    int const batch_size, int64_t const* cu_q_seq_len, int max_q_seq_len, int64_t const* merge_op, int const num_heads,
    int const head_size, cudaStream_t stream);

// load single chunk kv from kv_cache for each request
template <typename T, typename TCache>
void invokeMLALoadChunkedKV(T* output_kv_ptr, T* output_k_pe_ptr, KVBlockArray const& kv_cache, int const num_contexts,
    int64_t const* cu_ctx_chunked_len, int lora_size, int rope_size, int chunked_size, int chunked_idx,
    float const* kv_scale_quant_orig_ptr, cudaStream_t stream);

// output_kv {B, 2, ceil(chunked_size / kv_cache_tokens_per_block), h, kv_cache_tokens_per_block, d}, padding with
// zero
// kv {total_token, 2, H, uncompressed_h=128} 0 for k and 1 for v, k_pe {total_token, h=1, rope_h}
// input kv and k_pe can be cached tokens or uncached tokens
template <typename T>
void invokeMLASetChunkedKV(T* output_kv, T const* kv, T const* k_pe, int const batch_size, int const max_seq_len,
    int const num_heads, int uncompressed_head_size, int rope_size, int64_t const* cu_seq_lens,
    int const kv_cache_tokens_per_block, cudaStream_t stream);
} // namespace kernels
} // namespace tensorrt_llm
