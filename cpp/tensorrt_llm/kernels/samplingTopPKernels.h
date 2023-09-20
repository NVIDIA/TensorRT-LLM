/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <curand_kernel.h>

namespace tensorrt_llm
{
namespace kernels
{

void invokeTopPInitialize(int* topp_id_val_buf, int* topp_offset_buf, int* begin_topp_offset_buf_,
    const size_t batch_size, const int n, cudaStream_t stream);

template <typename T>
void invokeTopPSampling(void* workspace, size_t& workspace_size, size_t& cub_temp_storage_size, int** output_ids,
    int* sequence_length, bool* finished_buf, float* cum_log_probs, float* output_log_probs, const T* log_probs,
    const int* id_vals, int* offset_buf, int* begin_offset_buf, curandState_t* curandstate, const int batch_size,
    const size_t vocab_size_padded, const int* end_ids, const float top_p, cudaStream_t stream,
    cudaDeviceProp* cuda_device_prop, const bool* skip_decode);

template <typename T>
void invokeBatchTopPSampling(void* workspace, size_t& workspace_size, size_t& cub_temp_storage_size, int** output_ids,
    int* sequence_length, bool* finished_buf, float* cum_log_probs, float* output_log_probs, const T* log_probs,
    const int* id_vals, int* offset_buf, int* begin_offset_buf, curandState_t* curandstate, const int batch_size,
    const size_t vocab_size_padded, const int* end_ids, const float max_top_p, const float* top_ps, cudaStream_t stream,
    cudaDeviceProp* cuda_device_prop, const bool* skip_decode);

template <typename T>
void invokeAddBiasSoftMax(T* logits, const T* bias, const int* end_ids, const bool* finished, const int m,
    const int n_padded, const int n, cudaStream_t stream);

namespace segmented_topp_impl
{
enum DType_t
{
    kFLOAT,
    kHALF,
    kINT8
};

template <typename Key_Data_Type_ = float, typename Value_Data_Type_ = int32_t, int BLOCK_THREADS_ = 256,
    int KEYS_PER_LDG_ = 1>
struct Segmented_topk_kernel_params
{
    typedef Key_Data_Type_ Key_Data_Type;
    typedef Value_Data_Type_ Value_Data_Type;

    enum
    {
        BLOCK_THREADS = BLOCK_THREADS_
    };

    enum
    {
        ITEMS_INCREMENT = 32
    };

    // enum { KEYS_PER_LDG = 2 * 4 / sizeof(Key_Data_Type_) };
    enum
    {
        KEYS_PER_LDG = KEYS_PER_LDG_
    };
};

struct TopKPerSegmentContext
{
    TopKPerSegmentContext()
        : sm_count(0)
        , sm_shared_size(0)
        , sm_version(0){};
    int sm_count;
    int sm_shared_size;
    int sm_version;
};

struct TopKPerSegmentParams
{
    // input/output keys and values
    void *gmem_src_keys, *gmem_dst_keys, *gmem_dst_vals;
    // not used in the custom implementation
    void* gmem_src_vals;
    // int array of size num_segments
    int* gmem_active_count_per_segment;
    int* gmem_active_count_total;
    int* gmem_begin_offsets;
    // gmem_end_offsets will be populated
    int* gmem_end_offsets;
    void* workspace;
    // total number of items for all segments
    int num_items;
    int num_segments;
    // top_k per segment
    int num_top_k;
    float top_p;
    float confidence_threshold;
};

int topPPerSegment(const TopKPerSegmentContext& context, TopKPerSegmentParams& params, const DType_t DT_SCORE,
    void* temp_storage, size_t& temp_storage_bytes, cudaStream_t stream);
} // namespace segmented_topp_impl

void invokeComputeToppDecay(float* runtime_top_p, const float* runtime_initial_top_p, const int** output_ids,
    const float* top_p_decay, const float* top_p_min, const int32_t* top_p_reset_ids, const int* sequence_lengths,
    const int local_batch_size, cudaStream_t stream);

} // namespace kernels
} // namespace tensorrt_llm
