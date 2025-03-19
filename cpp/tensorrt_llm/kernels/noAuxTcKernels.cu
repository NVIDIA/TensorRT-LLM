/*
 * Copyright (c) 2019-2024, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
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

#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/kernels/noAuxTcKernels.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;
using namespace tensorrt_llm::common;

namespace tensorrt_llm::kernels
{
constexpr unsigned FULL_WARP_MASK = 0xffffffff;
constexpr int32_t WARP_SIZE = 32;
constexpr int32_t BLOCK_SIZE = 512;
constexpr int32_t NUM_WARPS_PER_BLOCK = BLOCK_SIZE / WARP_SIZE;

namespace warp_topk
{

template <int size, typename T>
__host__ __device__ constexpr T round_up_to_multiple_of(T len)
{
    if (len == 0)
    {
        return 0;
    }
    return ((len - 1) / size + 1) * size;
}

template <typename T>
constexpr __host__ __device__ bool isPowerOf2(T v)
{
    return (v && !(v & (v - 1)));
}

template <bool greater, typename T>
__device__ bool is_better_than(T val, T baseline)
{
    return (val > baseline && greater) || (val < baseline && !greater);
}

template <typename T, typename idxT>
int calc_smem_size_for_block_wide(int num_of_warp, int64_t k)
{
    int64_t cache_topk = (sizeof(T) + sizeof(idxT)) * num_of_warp * k;
    int64_t n = std::max<int>(num_of_warp / 2 * k, num_of_warp * WARP_SIZE);
    return max(cache_topk, round_up_to_multiple_of<256>(n * sizeof(T)) + n * sizeof(idxT));
}

template <int size, bool ascending, typename T, typename idxT>
struct BitonicMerge
{
    // input should be a bitonic sequence, and sort it to be a monotonic sequence
    __device__ static void merge(T* __restrict__ val_arr, idxT* __restrict__ idx_arr)
    {
        static_assert(isPowerOf2(size));
        static_assert(size >= 2 * WARP_SIZE);
        constexpr int arr_len = size / WARP_SIZE;

        constexpr int stride = arr_len / 2;
        for (int i = 0; i < stride; ++i)
        {
            int const other_i = i + stride;
            T& val = val_arr[i];
            T& other_val = val_arr[other_i];
            if ((val > other_val && ascending) || (val < other_val && !ascending))
            {
                T tmp = val;
                val = other_val;
                other_val = tmp;

                idxT tmp2 = idx_arr[i];
                idx_arr[i] = idx_arr[other_i];
                idx_arr[other_i] = tmp2;
            }
        }

        BitonicMerge<size / 2, ascending, T, idxT>::merge(val_arr, idx_arr);
        BitonicMerge<size / 2, ascending, T, idxT>::merge(val_arr + arr_len / 2, idx_arr + arr_len / 2);
    }
};

template <int size, bool ascending, typename T, typename idxT>
struct BitonicSort
{
    __device__ static void sort(T* __restrict__ val_arr, idxT* __restrict__ idx_arr)
    {
        static_assert(isPowerOf2(size));
        static_assert(size >= 2 * WARP_SIZE);
        constexpr int arr_len = size / WARP_SIZE;

        BitonicSort<size / 2, true, T, idxT>::sort(val_arr, idx_arr);
        BitonicSort<size / 2, false, T, idxT>::sort(val_arr + arr_len / 2, idx_arr + arr_len / 2);
        BitonicMerge<size, ascending, T, idxT>::merge(val_arr, idx_arr);
    }
};

template <bool ascending, typename T, typename idxT>
struct BitonicSort<32, ascending, T, idxT>
{
    __device__ static void sort(T* __restrict__ val_arr, idxT* __restrict__ idx_arr)
    {
        int const lane = threadIdx.x % WARP_SIZE;

        // ascending doesn't matter before merging since all we need is a bitonic sequence
        for (int stage = 0; stage < 4; ++stage)
        {
            for (int stride = (1 << stage); stride > 0; stride /= 2)
            {
                bool reverse = (lane >> stage) & 2;
                bool is_second = lane & stride;

                T other = __shfl_xor_sync(FULL_WARP_MASK, *val_arr, stride);
                idxT other_idx = __shfl_xor_sync(FULL_WARP_MASK, *idx_arr, stride);
                if (*val_arr != other && (*val_arr > other) != (reverse != is_second))
                {
                    *val_arr = other;
                    *idx_arr = other_idx;
                }
            }
        }

        BitonicMerge<32, ascending, T, idxT>::merge(val_arr, idx_arr);
    }
};

template <bool ascending, typename T, typename idxT>
struct BitonicMerge<32, ascending, T, idxT>
{
    __device__ static void merge(T* __restrict__ val_arr, idxT* __restrict__ idx_arr)
    {
        int const lane = threadIdx.x % WARP_SIZE;
        for (int stride = WARP_SIZE / 2; stride > 0; stride /= 2)
        {
            bool is_second = lane & stride;
            T& val = *val_arr;
            T other = __shfl_xor_sync(FULL_WARP_MASK, val, stride);
            idxT& idx = *idx_arr;
            idxT other_idx = __shfl_xor_sync(FULL_WARP_MASK, idx, stride);
            if (val != other && ((val > other) == (ascending != is_second)))
            {
                val = other;
                idx = other_idx;
            }
        }
    }
};

template <int capacity, bool greater, typename T, typename idxT>
class WarpSort
{
public:
    __device__ WarpSort(idxT k, T dummy)
        : lane_(threadIdx.x % WARP_SIZE)
        , k_(k)
        , dummy_(dummy)
    {
        static_assert(capacity >= WARP_SIZE && isPowerOf2(capacity));

        for (int i = 0; i < max_arr_len_; ++i)
        {
            val_arr_[i] = dummy_;
            idx_arr_[i] = 0;
        }
    }

    // load and merge k sorted values
    __device__ void load_sorted(T const* __restrict__ in, idxT const* __restrict__ in_idx, idxT start)
    {
        idxT idx = start + WARP_SIZE - 1 - lane_;
        for (int i = max_arr_len_ - 1; i >= 0; --i, idx += WARP_SIZE)
        {
            if (idx < start + k_)
            {
                T t = in[idx];
                if (is_better_than<greater>(t, val_arr_[i]))
                {
                    val_arr_[i] = t;
                    idx_arr_[i] = in_idx[idx];
                }
            }
        }

        BitonicMerge<capacity, !greater, T, idxT>::merge(val_arr_, idx_arr_);
    }

    __device__ void dump(T* __restrict__ out, idxT* __restrict__ out_idx) const
    {
        for (int i = 0; i < max_arr_len_; ++i)
        {
            idxT out_i = i * WARP_SIZE + lane_;
            if (out_i < k_)
            {
                out[out_i] = val_arr_[i];
                out_idx[out_i] = idx_arr_[i];
            }
        }
    }

    __device__ void dumpIdx(idxT* __restrict__ out_idx) const
    {
        for (int i = 0; i < max_arr_len_; ++i)
        {
            idxT out_i = i * WARP_SIZE + lane_;
            if (out_i < k_)
            {
                out_idx[out_i] = idx_arr_[i];
            }
        }
    }

protected:
    static constexpr int max_arr_len_ = capacity / WARP_SIZE;

    T val_arr_[max_arr_len_];
    idxT idx_arr_[max_arr_len_];

    int const lane_;
    idxT const k_;
    T const dummy_;

}; // end class WarpSort

template <int capacity, bool greater, typename T, typename idxT>
class WarpSelect : public WarpSort<capacity, greater, T, idxT>
{
public:
    __device__ WarpSelect(idxT k, T dummy)
        : WarpSort<capacity, greater, T, idxT>(k, dummy)
        , k_th_(dummy)
        , k_th_lane_((k - 1) % WARP_SIZE)
    {

        extern __shared__ char smem_buf[]; // extern __shared__ T smem_buf[];

        int const num_of_warp = blockDim.x / WARP_SIZE;
        int const warp_id = threadIdx.x / WARP_SIZE;
        val_smem_ = reinterpret_cast<T*>(smem_buf);
        val_smem_ += warp_id * WARP_SIZE;
        idx_smem_
            = reinterpret_cast<idxT*>(smem_buf + round_up_to_multiple_of<256>(num_of_warp * sizeof(T) * WARP_SIZE));
        idx_smem_ += warp_id * WARP_SIZE;
    }

    __device__ void add(T const* in, idxT start, idxT end)
    {
        idxT const end_for_fullwarp = round_up_to_multiple_of<WARP_SIZE>(end - start) + start;
        for (idxT i = start + lane_; i < end_for_fullwarp; i += WARP_SIZE)
        {
            T val = (i < end) ? in[i] : dummy_;
            add(val, i);
        }
    }

    __device__ void add(T val, idxT idx)
    {
        bool do_add = is_better_than<greater>(val, k_th_);
        uint32_t mask = __ballot_sync(FULL_WARP_MASK, do_add);
        if (mask == 0)
        {
            return;
        }

        int pos = smem_buf_len_ + __popc(mask & ((0x1u << lane_) - 1));
        if (do_add && pos < WARP_SIZE)
        {
            val_smem_[pos] = val;
            idx_smem_[pos] = idx;
            do_add = false;
        }
        smem_buf_len_ += __popc(mask);
        if (smem_buf_len_ >= WARP_SIZE)
        {
            __syncwarp();
            merge_buf_(val_smem_[lane_], idx_smem_[lane_]);
            smem_buf_len_ -= WARP_SIZE;
        }
        if (do_add)
        {
            pos -= WARP_SIZE;
            val_smem_[pos] = val;
            idx_smem_[pos] = idx;
        }
        __syncwarp();
    }

    __device__ void done()
    {
        if (smem_buf_len_)
        {
            T val = (lane_ < smem_buf_len_) ? val_smem_[lane_] : dummy_;
            idxT idx = (lane_ < smem_buf_len_) ? idx_smem_[lane_] : 0;
            merge_buf_(val, idx);
        }

        // after done(), smem is used for merging results among warps
        __syncthreads();
    }

private:
    __device__ void set_k_th_()
    {
        k_th_ = __shfl_sync(FULL_WARP_MASK, val_arr_[max_arr_len_ - 1], k_th_lane_);
    }

    __device__ void merge_buf_(T val, idxT idx)
    {
        BitonicSort<WARP_SIZE, greater, T, idxT>::sort(&val, &idx);

        T& old = val_arr_[max_arr_len_ - 1];
        if (is_better_than<greater>(val, old))
        {
            old = val;
            idx_arr_[max_arr_len_ - 1] = idx;
        }

        BitonicMerge<capacity, !greater, T, idxT>::merge(val_arr_, idx_arr_);

        set_k_th_();
    }

    using WarpSort<capacity, greater, T, idxT>::max_arr_len_;
    using WarpSort<capacity, greater, T, idxT>::val_arr_;
    using WarpSort<capacity, greater, T, idxT>::idx_arr_;
    using WarpSort<capacity, greater, T, idxT>::lane_;
    using WarpSort<capacity, greater, T, idxT>::k_;
    using WarpSort<capacity, greater, T, idxT>::dummy_;

    T* val_smem_;
    idxT* idx_smem_;
    int smem_buf_len_ = 0;

    T k_th_;
    int const k_th_lane_;
}; // end class WarpSelect
} // namespace warp_topk

template <typename T>
__device__ void topk_with_k2(T* output, T const* input, cg::thread_block_tile<32> const& tile, int32_t const lane_id,
    int const num_experts_per_group)
{
    // Get the top2 per thread
    T largest = -INFINITY;
    T second_largest = -INFINITY;

    if (num_experts_per_group > WARP_SIZE)
    {
        for (int i = lane_id; i < num_experts_per_group; i += WARP_SIZE)
        {
            T value = input[i];
            if (value > largest)
            {
                second_largest = largest;
                largest = value;
            }
            else if (value > second_largest)
            {
                second_largest = value;
            }
        }
    }
    else
    {
        for (int i = lane_id; i < num_experts_per_group; i += WARP_SIZE)
        {
            largest = input[i];
        }
    }

    __syncwarp(); // Ensure all threads have valid data before reduction
    // Get the top2 warpwise
    T max1 = cg::reduce(tile, largest, cg::greater<T>());

    T max2 = max1;
    bool equal_to_max1 = (max1 == largest);

    int count_max1 = __popc(__ballot_sync(FULL_WARP_MASK, equal_to_max1));

    if (count_max1 == 1)
    {
        largest = (largest == max1) ? second_largest : largest;
        max2 = cg::reduce(tile, largest, cg::greater<T>());
    }

    if (lane_id == 0)
    {
        *output = max1 + max2;
    }
}

template <typename T>
__global__ void topk_with_k2_kernel(T* output, T* input, int64_t const num_tokens, int64_t const num_cases,
    int64_t const n_group, int64_t const num_experts_per_group)
{

    int32_t warp_id = threadIdx.x / WARP_SIZE;
    int32_t lane_id = threadIdx.x % WARP_SIZE;

    int32_t case_id = blockIdx.x * NUM_WARPS_PER_BLOCK + warp_id;
    if (case_id < num_cases)
    {
        input += case_id * num_experts_per_group;
        output += case_id;

        cg::thread_block block = cg::this_thread_block();
        cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        asm volatile("griddepcontrol.wait;");
#endif
        topk_with_k2(output, input, tile, lane_id, num_experts_per_group);
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename T, typename IdxT>
__global__ void group_idx_and_topk_idx_kernel(T* scores, T const* group_scores, T* topk_values, IdxT* topk_indices,
    T* scores_with_bias, int64_t const num_tokens, int64_t const n_group, int64_t const topk_group, int64_t const topk,
    int64_t const num_experts, int64_t const num_experts_per_group, double routed_scaling_factor)
{
    int32_t warp_id = threadIdx.x / WARP_SIZE;
    int32_t lane_id = threadIdx.x % WARP_SIZE;
    int32_t case_id = blockIdx.x * NUM_WARPS_PER_BLOCK + warp_id; // one per token
    scores_with_bias += case_id * num_experts;
    scores += case_id * num_experts;
    group_scores += case_id * n_group;
    topk_values += case_id * topk;
    topk_indices += case_id * topk;

    int32_t align_num_experts_per_group = warp_topk::round_up_to_multiple_of<WARP_SIZE>(num_experts_per_group);

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> tile = cg::tiled_partition<32>(block);

    extern __shared__ char smem_buf[]; // NOTE: reuse the shared memory here to store the target topk idx
    int32_t* s_topk_idx = reinterpret_cast<int32_t*>(smem_buf);
    T* s_topk_value = reinterpret_cast<T*>(s_topk_idx + NUM_WARPS_PER_BLOCK * topk) + warp_id * topk;
    s_topk_idx += warp_id * topk;

    T value = cuda::std::numeric_limits<T>::min();
    T topk_group_value = cuda::std::numeric_limits<T>::min();
    int32_t num_equalto_topkth_group;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.wait;"); // I think all prolog can be put before acqbulk because it's ptr arithmetic
#endif

    if (case_id < num_tokens)
    {
        // calculate group_idx
        int32_t target_num_min = WARP_SIZE - n_group + topk_group;
        if (lane_id < n_group
            && (isfinite(cuda_cast<float, T>(group_scores[lane_id])))) // The check is necessary to avoid abnormal input
        {
            value = group_scores[lane_id];
        }

        int count_equal_to_top_value = WARP_SIZE - n_group;
        int pre_count_equal_to_top_value = 0;
        // Use loop to find the largset top_group
        while (count_equal_to_top_value < target_num_min)
        {
            __syncwarp(); // Ensure all threads have valid data before reduction
            topk_group_value = cg::reduce(tile, value, cg::greater<T>());
            if (value == topk_group_value)
            {
                value = cuda::std::numeric_limits<T>::min();
            }
            pre_count_equal_to_top_value = count_equal_to_top_value;
            count_equal_to_top_value
                = __popc(__ballot_sync(FULL_WARP_MASK, (value == cuda::std::numeric_limits<T>::min())));
        }
        num_equalto_topkth_group = target_num_min - pre_count_equal_to_top_value;
    }
    __syncthreads();

    warp_topk::WarpSelect</*capability*/ WARP_SIZE, /*greater*/ true, T, int32_t> queue((int32_t) topk, -INFINITY);

    int count_equalto_topkth_group = 0;
    bool if_proceed_next_topk = (topk_group_value != cuda::std::numeric_limits<T>::min());
    if (case_id < num_tokens && if_proceed_next_topk)
    {
        for (int i_group = 0; i_group < n_group; i_group++)
        {
            if ((group_scores[i_group] > topk_group_value)
                || ((group_scores[i_group] == topk_group_value)
                    && (count_equalto_topkth_group < num_equalto_topkth_group)))
            {
                int32_t offset = i_group * num_experts_per_group;
                for (int32_t i = lane_id; i < align_num_experts_per_group; i += WARP_SIZE)
                {
                    T candidates
                        = (i < num_experts_per_group) && isfinite(cuda_cast<float, T>(scores_with_bias[offset + i]))
                        ? scores_with_bias[offset + i]
                        : cuda::std::numeric_limits<T>::min();
                    queue.add(candidates, offset + i);
                }
                if (group_scores[i_group] == topk_group_value)
                {
                    count_equalto_topkth_group++;
                }
            }
        }
        queue.done();
        __syncwarp();
        // Get the topk_idx
        queue.dumpIdx(s_topk_idx);
        __syncwarp();
    }

    // Load the valid score value
    // Calculate the summation
    float topk_sum = 1e-20;
    if (case_id < num_tokens && if_proceed_next_topk)
    {
        for (int i = lane_id; i < warp_topk::round_up_to_multiple_of<WARP_SIZE>(topk); i += WARP_SIZE)
        {
            T value = i < topk ? scores[s_topk_idx[i]] : cuda_cast<T, float>(0.0f); // Load the valid value of expert
            if (i < topk)
            {
                s_topk_value[i] = value;
            }
            topk_sum += reduce(tile, cuda_cast<float, T>(value), cg::plus<float>());
        }
    }

    __syncthreads();

    if (case_id < num_tokens)
    {
        if (if_proceed_next_topk)
        {
            for (int i = lane_id; i < topk; i += WARP_SIZE)
            {
                float value = cuda_cast<float, T>(s_topk_value[i]) / topk_sum * routed_scaling_factor;
                topk_indices[i] = s_topk_idx[i];
                topk_values[i] = cuda_cast<T, float>(value);
            }
        }
        else
        {
            for (int i = lane_id; i < topk; i += WARP_SIZE)
            {
                topk_indices[i] = i;
                topk_values[i] = cuda_cast<T, float>(1.0f / topk);
            }
        }
        // Note: when if_proceed_next_topk==false, choose the first 8 experts as the default result.
        //@TODO: check if this default strategy is acceptable. Might need to leave it as nan array.
    }
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    asm volatile("griddepcontrol.launch_dependents;");
#endif
}

template <typename T, typename IdxT>
void invokeNoAuxTc(T* scores, T* group_scores, T* topk_values, IdxT* topk_indices, T* scores_with_bias,
    int64_t const num_tokens, int64_t const num_experts, int64_t const n_group, int64_t const topk_group,
    int64_t const topk, double const routed_scaling_factor, cudaStream_t const stream)
{
    int64_t num_cases = num_tokens * n_group;
    int64_t topk_with_k2_num_blocks = (num_cases - 1) / NUM_WARPS_PER_BLOCK + 1;
    auto* kernel_instance1 = &topk_with_k2_kernel<T>;
    cudaLaunchConfig_t config;
    config.gridDim = topk_with_k2_num_blocks;
    config.blockDim = BLOCK_SIZE;
    config.dynamicSmemBytes = 0;
    config.stream = stream;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, kernel_instance1, group_scores, scores_with_bias, num_tokens, num_cases, n_group,
        num_experts / n_group);
    sync_check_cuda_error(stream);

    int64_t topk_with_k_group_num_blocks = (num_tokens - 1) / NUM_WARPS_PER_BLOCK + 1;
    size_t dynamic_smem_in_bytes = warp_topk::calc_smem_size_for_block_wide<T, int32_t>(NUM_WARPS_PER_BLOCK, topk);
    auto* kernel_instance2 = &group_idx_and_topk_idx_kernel<T, IdxT>;
    config.gridDim = topk_with_k_group_num_blocks;
    config.blockDim = BLOCK_SIZE;
    config.dynamicSmemBytes = dynamic_smem_in_bytes;
    config.stream = stream;
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL();
    config.numAttrs = 1;
    config.attrs = attrs;
    cudaLaunchKernelEx(&config, kernel_instance2, scores, group_scores, topk_values, topk_indices, scores_with_bias,
        num_tokens, n_group, topk_group, topk, num_experts, num_experts / n_group, routed_scaling_factor);
    sync_check_cuda_error(stream);
}

#define INSTANTIATE_NOAUX_TC(T, IdxT)                                                                                  \
    template void invokeNoAuxTc<T, IdxT>(T * scores, T * group_scores, T * topk_values, IdxT * topk_indices,           \
        T * scores_with_bias, int64_t const num_tokens, int64_t const num_experts, int64_t const n_group,              \
        int64_t const topk_group, int64_t const topk, double const routed_scaling_factor, cudaStream_t const stream);

INSTANTIATE_NOAUX_TC(float, int32_t);
INSTANTIATE_NOAUX_TC(half, int32_t);
#ifdef ENABLE_BF16
INSTANTIATE_NOAUX_TC(__nv_bfloat16, int32_t);
#endif
} // namespace tensorrt_llm::kernels
