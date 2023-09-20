/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#else
#include "3rdparty/cub/cub.cuh"
#endif

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/samplingTopPKernels.h"

constexpr int ENABLE_SINGLE_PASS_TOP_P = 0;
constexpr float SINGLE_PASS_THRESHOLD = 0.9;

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{

namespace segmented_topp_impl
{

template <int HALF_ELEMENTS_PER_WARP_LOAD>
using Copy_half_t = typename std::conditional<HALF_ELEMENTS_PER_WARP_LOAD == 32, half,
    typename std::conditional<HALF_ELEMENTS_PER_WARP_LOAD == 64, int,
        typename std::conditional<HALF_ELEMENTS_PER_WARP_LOAD == 128, int2, int4>::type>::type>::type;

template <typename T, int ELEMENTS_PER_WARP_LOAD>
using Copy_t = Copy_half_t<sizeof(T) / sizeof(half) * ELEMENTS_PER_WARP_LOAD>;

template <typename T>
struct Float_as_int_
{
};

template <>
struct Float_as_int_<float>
{
    using Type = uint32_t;
};

template <>
struct Float_as_int_<__half>
{
    using Type = uint16_t;
};

using kernel_params_float = Segmented_topk_kernel_params<float, int32_t, 256, 2>;
using kernel_params_float_1 = Segmented_topk_kernel_params<float, int32_t, 256, 1>;
using kernel_params_half = Segmented_topk_kernel_params<__half, int32_t, 256, 4>;
using kernel_params_half_1 = Segmented_topk_kernel_params<__half, int32_t, 256, 1>;

///////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float to_float(uint32_t src)
{
    return __int_as_float(src);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float to_float(uint16_t src)
{
    __half dst = __ushort_as_half(src);
    return __half2float(dst);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// sort one segment per cta
template <typename T_SCORE, int BLOCK_THREADS, int ELEMENTS_PER_THREAD>
__global__ void blockSortKernel(const T_SCORE* d_keys_in, T_SCORE* d_keys_out, const int32_t* d_values_in,
    int32_t* d_values_out, const int32_t* active_counts, int num_items_, int stride_items, int num_segments)
{
    // Specialize BlockRadixSort for a 1D block
    typedef cub::BlockRadixSort<T_SCORE, BLOCK_THREADS, ELEMENTS_PER_THREAD, int32_t> BlockRadixSort;

    // Allocate shared memory for BlockRadixSort
    __shared__ typename BlockRadixSort::TempStorage temp_storage;

    if (blockIdx.x >= num_segments)
    {
        return;
    }

    int num_items = active_counts[blockIdx.x]; // > num_items_ ? num_items_ :
                                               // active_counts[blockIdx.x];

    if (num_items == 0)
    {
        return;
    }

    // Obtain a segment of consecutive items that are blocked across threads
    T_SCORE thread_keys[ELEMENTS_PER_THREAD];
    int32_t thread_values[ELEMENTS_PER_THREAD];

    int32_t block_offset = blockIdx.x * stride_items;
    cub::LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, d_keys_out + block_offset, thread_keys, num_items, 0);
    cub::LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, d_values_out + block_offset, thread_values, num_items, -1);
    __syncthreads();

    // Collectively sort the keys and values among block threads
    BlockRadixSort(temp_storage).SortDescendingBlockedToStriped(thread_keys, thread_values);

    // Store output in striped fashion
    cub::StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_keys_out + block_offset, thread_keys, num_items);
    cub::StoreDirectStriped<BLOCK_THREADS>(threadIdx.x, d_values_out + block_offset, thread_values, num_items);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// block sort kernel
template <typename T_SCORE>
void blockSort(const T_SCORE* d_keys_in, T_SCORE* d_keys_out, const int32_t* d_values_in, int32_t* d_values_out,
    const int32_t* active_counts, int num_items, int stride_items, int num_segments, cudaStream_t stream)
{
    if (num_items == 0)
    {
        return;
    }

    int kernel_index = divUp(num_items, 128) - 1;
    int warps_per_cta = (kernel_index + 1) * 128 / 32;
    if (kernel_index > 7)
    {
        kernel_index = 7 + divUp(num_items, 1024) - 1;
        warps_per_cta = 1024 / 32;
    }
    assert(warps_per_cta <= 32);

    dim3 block(warps_per_cta * 32);
    dim3 grid(num_segments);

    using kernel_func = void (*)(const T_SCORE* d_keys_in, T_SCORE* d_keys_out, const int32_t* d_values_in,
        int32_t* d_values_out, const int32_t* active_counts, int num_items, int stride_items, int num_segments);

    static const kernel_func kernel_funcs[] = {
        &blockSortKernel<T_SCORE, 128, 1>, &blockSortKernel<T_SCORE, 256, 1>, &blockSortKernel<T_SCORE, 384, 1>,
        &blockSortKernel<T_SCORE, 512, 1>, &blockSortKernel<T_SCORE, 640, 1>, &blockSortKernel<T_SCORE, 768, 1>,
        &blockSortKernel<T_SCORE, 896, 1>, &blockSortKernel<T_SCORE, 1024, 1>, &blockSortKernel<T_SCORE, 1024, 2>,
        &blockSortKernel<T_SCORE, 1024, 4>,
        //&blockSortKernel<T_SCORE, 1024, 6>,
    };
    kernel_funcs[kernel_index]<<<grid, block, 0, stream>>>(
        d_keys_in, d_keys_out, d_values_in, d_values_out, active_counts, num_items, stride_items, num_segments);
}

///////////////////////////////////////////////////////////////////////////////////////////////////

struct BlockPrefixCallbackOp
{
    // Running prefix
    int running_total;

    // Constructor
    __device__ BlockPrefixCallbackOp(uint32_t running_total)
        : running_total(running_total)
    {
    }

    // Callback operator to be entered by the first warp of threads in the block.
    // Thread-0 is responsible for returning a value for seeding the block-wide
    // scan.
    __device__ int operator()(uint32_t block_aggregate)
    {
        uint32_t old_prefix = running_total;
        running_total += block_aggregate;
        return old_prefix;
    }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

#define DO_DEBUG_PRINT 0

// governs the split between regs and smem
constexpr float SMEM_FRACTION = 0.5F;
constexpr float P_EPSILON = 0.01F;

constexpr int MAX_TOP_K = 3072;
constexpr int WARP_SZ = 32;

template <typename Kernel_params, int ITEMS_PER_THREAD>
__global__ __launch_bounds__(Kernel_params::BLOCK_THREADS, 1) void segmented_top_p_single_pass(
    TopKPerSegmentParams params)
{
#if DO_DEBUG_PRINT
    constexpr int debug_block_id = 26;
#endif

    using Key_Data_Type = typename Kernel_params::Key_Data_Type;
    using Int_Key_Data_Type = typename Float_as_int_<Key_Data_Type>::Type;

    // 4 fp16 keys or 2 fp32 keys
    constexpr int KEYS_PER_LDG = Kernel_params::KEYS_PER_LDG;
    typedef Copy_t<Key_Data_Type, WARP_SZ * KEYS_PER_LDG> copy_t;

    union access_t
    {
        copy_t v;
        Int_Key_Data_Type x[KEYS_PER_LDG]; // supported size 1,2,4
    };

    constexpr int BLOCK_THREADS = Kernel_params::BLOCK_THREADS;

    constexpr int ITEMS_PER_THREAD_IN_REGS = ITEMS_PER_THREAD * (1.0F - SMEM_FRACTION);
    constexpr int ITEMS_PER_THREAD_IN_SMEM = ITEMS_PER_THREAD - ITEMS_PER_THREAD_IN_REGS;

#if DO_DEBUG_PRINT == 1
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        printf(
            "ITEMS_PER_THREAD, ITEMS_PER_THREAD_IN_REGS, "
            "ITEMS_PER_THREAD_IN_SMEM = %d, %d, %d\n",
            ITEMS_PER_THREAD, ITEMS_PER_THREAD_IN_REGS, ITEMS_PER_THREAD_IN_SMEM);
    }
#endif

    constexpr int MIN_KEY = 0;
    constexpr int ENABLED_PER_THREAD = (ITEMS_PER_THREAD + 32 - 1) / 32;
    extern __shared__ int2 dynamic_smem[];
    int2* smem_selected_elements = dynamic_smem;
    Int_Key_Data_Type* smem_thread_items = reinterpret_cast<Int_Key_Data_Type*>(smem_selected_elements + MAX_TOP_K);

    __shared__ unsigned int smem_selected_count;

    // Specialize BlockScan type for our thread block
    typedef cub::BlockScan<uint32_t, BLOCK_THREADS> BlockScan;

    // Specialize BlockScan type for our thread block
    typedef cub::BlockReduce<float, BLOCK_THREADS> BlockReduce;
    __shared__ float smem_p_sum_total;

    __shared__ union
    {
        typename BlockScan::TempStorage scan;

        typename BlockReduce::TempStorage reduce;
    } temp_storage;

    // Initialize running total
    BlockPrefixCallbackOp prefix_op(0);

    unsigned int old_selected_count;

    uint32_t segment = blockIdx.y * gridDim.x + blockIdx.x;

    // Preceding TopK has shortcutted this segment
    if (params.gmem_begin_offsets[segment] == params.gmem_end_offsets[segment])
    {
        if (threadIdx.x == 0)
        {
            params.gmem_active_count_per_segment[segment] = 1;
            atomicMax(params.gmem_active_count_total, 1);
        }
        return;
    }

    Int_Key_Data_Type* gmem_src_keys = reinterpret_cast<Int_Key_Data_Type*>(params.gmem_src_keys);
    Int_Key_Data_Type* gmem_dst_keys = reinterpret_cast<Int_Key_Data_Type*>(params.gmem_dst_keys);
    int32_t* gmem_dst_vals = reinterpret_cast<int32_t*>(params.gmem_dst_vals);

    constexpr int BITS_IN_KEY = sizeof(Key_Data_Type) * 8;

    int items = params.num_items / params.num_segments;
    int first_index = segment * items;
    gmem_src_keys += first_index;
    gmem_dst_keys += first_index;
    gmem_dst_vals += first_index;

    int index_limit = items;
    Int_Key_Data_Type thread_items[ITEMS_PER_THREAD_IN_REGS] = {0};

    // Load all keys into registers and smem
    const int lane_id = threadIdx.x % WARP_SZ;
    const int warp_id = threadIdx.x / WARP_SZ;
    constexpr int NUM_WARPS = BLOCK_THREADS / WARP_SZ;

    access_t ZERO;
    for (int i = 0; i < KEYS_PER_LDG; i++)
    {
        ZERO.x[i] = MIN_KEY;
    }

    // registers
    for (int iter = 0; iter < ITEMS_PER_THREAD_IN_REGS; iter++)
    {
        int offset = (iter + threadIdx.x * ITEMS_PER_THREAD);
        thread_items[iter] = (offset < index_limit) ? gmem_src_keys[offset] : MIN_KEY;
    }

    // shared memory
    for (int c = warp_id; c < BLOCK_THREADS; c += NUM_WARPS)
    {
        for (int iter = lane_id * KEYS_PER_LDG; iter < ITEMS_PER_THREAD_IN_SMEM; iter += WARP_SZ * KEYS_PER_LDG)
        {
            int offset = iter + c * ITEMS_PER_THREAD + ITEMS_PER_THREAD_IN_REGS;
            access_t val;
            val.v = (offset < index_limit) ? *reinterpret_cast<copy_t*>(&gmem_src_keys[offset]) : ZERO.v;
            for (int i = 0; i < KEYS_PER_LDG; i++)
            {
                smem_thread_items[c + (iter + i) * BLOCK_THREADS] = val.x[i];
            }
            // smem_thread_items[c  + iter * BLOCK_THREADS] = (offset < index_limit)?
            // gmem_src_keys[offset] : MIN_KEY;
        }
    }

    Int_Key_Data_Type select_mask = 0;
    Int_Key_Data_Type save_mask = 0;

    // Int_Key_Data_Type save_bit = 0;
    // set to true when we finish with too few keys, so we go back to
    // last_save_mask one more time
    bool is_last_iter = false;

    if (threadIdx.x == 0)
    {
        smem_selected_count = 0;
        old_selected_count = 0;
    }

    // iterate over bits.
    // skip the first two bits,
    // * bit 31 is the sign bit. all values are positive
    // * bit 30 is only set for values >= 2, but the input consists only of values
    // in the range of [0,1]
    constexpr int START_BIT = BITS_IN_KEY - 1;
    constexpr int SKIP_BITS = 2;
    constexpr Int_Key_Data_Type ONE = (Int_Key_Data_Type) 1;
    uint32_t selected;
    uint32_t sc;
    float p_sum_total = 0.0F;
    float old_p_sum_total = 0.0F;
    uint32_t offset = 0;
    for (Int_Key_Data_Type bit = START_BIT - SKIP_BITS; true; --bit)
    {
        __syncthreads();
        Int_Key_Data_Type bit_mask = select_mask | (ONE << bit);

        uint32_t enabled[ENABLED_PER_THREAD] = {0};
        float thread_sum = 0.0F;

        for (int item = 0; item < ITEMS_PER_THREAD_IN_REGS; ++item)
        {
            // check if all the bits from bit mask are contained in the thread_item.
            // If yes, set respective bit of enabled
            auto val = thread_items[item];
            uint32_t is_enabled = uint32_t(((val ^ bit_mask) & bit_mask) == 0);
            // thread_sum += (is_enabled)? to_float(val) : 0.0F;
            thread_sum += is_enabled * to_float(val);
            enabled[item / 32] |= is_enabled << (item % 32);
        }

        for (int item = 0; item < ITEMS_PER_THREAD_IN_SMEM; ++item)
        {
            int idx = threadIdx.x + item * BLOCK_THREADS;
            // int idx = item + ITEMS_PER_THREAD_IN_SMEM * threadIdx.x;
            auto val = smem_thread_items[idx];
            uint32_t is_enabled = uint32_t(((val ^ bit_mask) & bit_mask) == 0);
            // thread_sum += (is_enabled)? to_float(val) : 0.0F;
            thread_sum += is_enabled * to_float(val);
            enabled[(ITEMS_PER_THREAD_IN_REGS + item) / 32] |= is_enabled << ((ITEMS_PER_THREAD_IN_REGS + item) % 32);
        }

        selected = 0;
#pragma unroll
        for (int i = 0; i < ENABLED_PER_THREAD; i++)
        {
            selected += __popc(enabled[i]);
        }

        float p_sum = BlockReduce(temp_storage.reduce).Sum(thread_sum);

        if (threadIdx.x == 0)
        {
            p_sum_total += p_sum;
            smem_p_sum_total = p_sum_total;
        }

        __syncthreads();
        p_sum_total = smem_p_sum_total;
        __syncthreads();

        BlockScan(temp_storage.scan).ExclusiveSum(selected, offset, prefix_op);

        if (threadIdx.x == 0)
        {
            smem_selected_count = prefix_op.running_total;
        }

        __syncthreads();
        sc = smem_selected_count;
        __syncthreads();

        // float p_diff = params.top_p - p_sum_total;
        float p_diff = p_sum_total - params.top_p;

        if ((p_sum_total <= params.top_p + P_EPSILON && p_sum_total > 0)
            || (p_sum_total > params.top_p && sc <= MAX_TOP_K) || (bit == 0 && p_sum_total > 0) || is_last_iter)
        {

#if DO_DEBUG_PRINT == 1
            __syncthreads();
            if (threadIdx.x == 0 && blockIdx.x == debug_block_id)
            {
                sc = smem_selected_count;
                printf(
                    "bit %d bit_mask %d offset %d (%d, %d), sc = %d, p_sum = %f, "
                    "p_sum_total = %f\n",
                    bit, bit_mask, offset, blockIdx.x, threadIdx.x, sc, p_sum, p_sum_total);
            }
            __syncthreads();
#endif

            for (int item = 0; item < ITEMS_PER_THREAD_IN_REGS; ++item)
            {
                // last condition should not trigger with well trained weights, but we
                // will get illegal mewmory access if we do not have one in those rare
                // cases
                if (enabled[item / 32] & (ONE << (item % 32)) && offset < MAX_TOP_K)
                {
                    smem_selected_elements[offset]
                        = make_int2(thread_items[item], item + threadIdx.x * ITEMS_PER_THREAD);
                    ++offset;
                    thread_items[item] = MIN_KEY;
                }
            }

            for (int item = 0; item < ITEMS_PER_THREAD_IN_SMEM; ++item)
            {
                if (enabled[(item + ITEMS_PER_THREAD_IN_REGS) / 32] & (ONE << ((item + ITEMS_PER_THREAD_IN_REGS) % 32))
                    && offset < MAX_TOP_K)
                {
                    int idx = threadIdx.x + item * BLOCK_THREADS;
                    // int idx = item + ITEMS_PER_THREAD_IN_SMEM * threadIdx.x;
                    // if (idx <  params.num_items_per_segment_in_smem)
                    {
                        smem_selected_elements[offset] = make_int2(
                            smem_thread_items[idx], item + threadIdx.x * ITEMS_PER_THREAD + ITEMS_PER_THREAD_IN_REGS);
                        ++offset;
                        smem_thread_items[idx] = MIN_KEY;
                    }
                }
            }
        }

#if DO_DEBUG_PRINT == 1
        if (threadIdx.x == 0 && blockIdx.x == debug_block_id)
        {
            printf(
                "!!!! bit %d bit_mask %d offset %d (%d, %d), sc = %d, p_sum = %f, "
                "p_sum_total = %f\n",
                bit, bit_mask, offset, blockIdx.x, threadIdx.x, sc, p_sum, p_sum_total);
        }
#endif

        if (p_diff <= P_EPSILON && p_diff >= 0 || (p_sum_total > params.top_p && sc <= MAX_TOP_K) || bit == 0)
        {

            break;
        }
        // p > top_p
        else if (p_diff > P_EPSILON)
        {
            // There are too many bits in the current selection
            // Save the current state and go to the next bit
            // If there are not enough items left using the next bit
            // it's necessary to restart here with the current bit not set
            save_mask = bit_mask;
            select_mask |= bit_mask;

            if (threadIdx.x == 0)
            {
                smem_selected_count = old_selected_count;
                p_sum_total = old_p_sum_total;

                prefix_op.running_total = old_selected_count;
            }
        }
        else
        {
            // sc < num_top_k branch
            if (save_mask)
            {
                select_mask = save_mask;

                save_mask = 0;
            }
            if (threadIdx.x == 0)
            {
                old_selected_count = smem_selected_count;
                old_p_sum_total = p_sum_total;
            }
        }
    }

    __syncthreads();

    // store data to global memory
    sc = (p_sum_total < params.top_p) ? params.num_items / params.num_segments : smem_selected_count;
    if (threadIdx.x == 0)
    {
        params.gmem_active_count_per_segment[segment] = sc;
        atomicMax(params.gmem_active_count_total, sc);
    }
    if (sc >= MAX_TOP_K)
    {
        return;
    }
    for (int i = threadIdx.x; i < sc; i += blockDim.x)
    {
        int2 selected_element = smem_selected_elements[i];
        gmem_dst_keys[i] = selected_element.x;
        gmem_dst_vals[i] = selected_element.y;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_params>
int getSmemSizeAndCheck(const TopKPerSegmentContext& context, const TopKPerSegmentParams& params)
{
    constexpr int BLOCK_THREADS = Kernel_params::BLOCK_THREADS;
    using Key_Data_Type = typename Kernel_params::Key_Data_Type;
    int num_items_per_segment = params.num_items / params.num_segments;
    constexpr int ITEMS_INCREMENT = Kernel_params::ITEMS_INCREMENT;
    int kernel_index = divUp(num_items_per_segment, BLOCK_THREADS * ITEMS_INCREMENT) - 1;

    int smem_size = MAX_TOP_K * sizeof(int2);
    const int items_per_thread = (kernel_index + 1) * ITEMS_INCREMENT;
    const int items_per_thread_in_regs = items_per_thread * (1.0F - SMEM_FRACTION);
    const int items_per_thread_in_smem = items_per_thread - items_per_thread_in_regs;

    smem_size += items_per_thread_in_smem * BLOCK_THREADS * sizeof(typename Float_as_int_<Key_Data_Type>::Type);

    int keys_per_ldg = 2 * sizeof(Key_Data_Type) / 2;
    if (smem_size + BLOCK_THREADS * sizeof(float) > (size_t) context.sm_shared_size || // dynamic + static memory
        items_per_thread_in_regs + items_per_thread_in_smem != items_per_thread || params.top_p + P_EPSILON > 1.0F
        || items_per_thread_in_regs % keys_per_ldg != 0 || items_per_thread_in_smem % keys_per_ldg != 0
        || num_items_per_segment % keys_per_ldg != 0)
    {
        return -1;
    }

    return smem_size;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int getSmemSizeAndCheck(
    const TopKPerSegmentContext& context, const TopKPerSegmentParams& params, const DType_t DT_SCORE)
{
    int num_items_per_segment = params.num_items / params.num_segments;
    if (DT_SCORE == kFLOAT)
    {
        if (num_items_per_segment % 2 == 0)
        {
            return getSmemSizeAndCheck<kernel_params_float>(context, params);
        }
        else
        {
            return getSmemSizeAndCheck<kernel_params_float_1>(context, params);
        }
    }
    else
    {
        if (num_items_per_segment % 4 == 0)
        {
            return getSmemSizeAndCheck<kernel_params_half>(context, params);
        }
        else
        {
            return getSmemSizeAndCheck<kernel_params_half_1>(context, params);
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_params>
void segmentedTopPSinglePass_dispatch(
    const TopKPerSegmentParams& params, const TopKPerSegmentContext& context, cudaStream_t stream)
{

    constexpr int BLOCK_THREADS = Kernel_params::BLOCK_THREADS;
    using Key_Data_Type = typename Kernel_params::Key_Data_Type;
    using Value_Data_Type = typename Kernel_params::Value_Data_Type;

    int num_items_per_segment = params.num_items / params.num_segments;

    constexpr int ITEMS_INCREMENT = Kernel_params::ITEMS_INCREMENT;
    int kernel_index = divUp(num_items_per_segment, BLOCK_THREADS * ITEMS_INCREMENT) - 1;

#define KERNEL_RUN(INDEX)                                                                                              \
    {                                                                                                                  \
        if (smem_size > 0)                                                                                             \
            check_cuda_error(                                                                                          \
                cudaFuncSetAttribute(segmented_top_p_single_pass<Kernel_params, ITEMS_INCREMENT*(INDEX + 1)>,          \
                    cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));                                          \
        segmented_top_p_single_pass<Kernel_params, ITEMS_INCREMENT*(INDEX + 1)>                                        \
            <<<grid_dim, Kernel_params::BLOCK_THREADS, smem_size, stream>>>(params);                                   \
    }

    int smem_size = getSmemSizeAndCheck<Kernel_params>(context, params);

    dim3 grid_dim(params.num_segments, 1);

    switch (kernel_index)
    {
    case 0: KERNEL_RUN(0) break;
    case 1: KERNEL_RUN(1) break;
    case 2: KERNEL_RUN(2) break;
    case 3: KERNEL_RUN(3) break;
    case 4: KERNEL_RUN(4) break;
    case 5: KERNEL_RUN(5) break;
    case 6: KERNEL_RUN(6) break;
    case 7: KERNEL_RUN(7) break;
    default: exit(1);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Kernel_params>
void topPPerSegment_dispatch(const TopKPerSegmentContext& context, TopKPerSegmentParams& params, void* temp_storage,
    size_t& temp_storage_bytes, cudaStream_t stream)
{

    using Key_Data_Type = typename Kernel_params::Key_Data_Type;
    using Value_Data_Type = typename Kernel_params::Value_Data_Type;

    if (temp_storage == nullptr)
    {
        if (params.num_segments > 1)
        {
            cub::DeviceSegmentedRadixSort::SortPairsDescending(temp_storage, temp_storage_bytes,
                reinterpret_cast<Key_Data_Type*>(params.gmem_src_keys),
                reinterpret_cast<Key_Data_Type*>(params.gmem_dst_keys),
                reinterpret_cast<Value_Data_Type*>(params.gmem_src_vals),
                reinterpret_cast<Value_Data_Type*>(params.gmem_dst_vals), params.num_items, params.num_segments,
                params.gmem_begin_offsets, params.gmem_end_offsets, 0, sizeof(Key_Data_Type) * 8, stream);
        }
        else
        {
            cub::DeviceRadixSort::SortPairsDescending(temp_storage, temp_storage_bytes,
                reinterpret_cast<Key_Data_Type*>(params.gmem_src_keys),
                reinterpret_cast<Key_Data_Type*>(params.gmem_dst_keys),
                reinterpret_cast<Value_Data_Type*>(params.gmem_src_vals),
                reinterpret_cast<Value_Data_Type*>(params.gmem_dst_vals), params.num_items, 0,
                sizeof(Key_Data_Type) * 8, stream);
        }
        temp_storage_bytes = divUp(temp_storage_bytes, 256) * 256;
        // total active counts
        temp_storage_bytes += divUp(sizeof(int), 256) * 256;
        // storage for gmem_end_offsets
        temp_storage_bytes += divUp(sizeof(int) * params.num_segments, 256) * 256;
        return;
    }

    size_t cub_temp_storage_bytes
        = temp_storage_bytes - divUp(sizeof(int), 256) * 256 - divUp(sizeof(int) * params.num_segments, 256) * 256;
    void* cub_temp_storage = temp_storage;
    params.gmem_active_count_total = reinterpret_cast<int*>((char*) temp_storage + cub_temp_storage_bytes);
    params.gmem_active_count_per_segment
        = reinterpret_cast<int*>((char*) params.gmem_active_count_total + divUp(sizeof(int), 256) * 256);

    int num_items_per_segment = params.num_items / params.num_segments;

    cudaMemsetAsync(params.gmem_active_count_total, 0, sizeof(int), stream);
    cudaMemsetAsync(params.gmem_dst_keys, 0, params.num_items * sizeof(Key_Data_Type), stream);
    segmentedTopPSinglePass_dispatch<Kernel_params>(params, context, stream);

    int max_num_items = 0;
    cudaMemcpyAsync(&max_num_items, params.gmem_active_count_total, sizeof(int), cudaMemcpyDeviceToHost, stream);

    cudaStreamSynchronize(stream);

    if (max_num_items >= MAX_TOP_K || max_num_items == 0)
    {
        if (params.num_segments > 1)
        {
            cub::DeviceSegmentedRadixSort::SortPairsDescending(cub_temp_storage, cub_temp_storage_bytes,
                reinterpret_cast<Key_Data_Type*>(params.gmem_src_keys),
                reinterpret_cast<Key_Data_Type*>(params.gmem_dst_keys),
                reinterpret_cast<Value_Data_Type*>(params.gmem_src_vals),
                reinterpret_cast<Value_Data_Type*>(params.gmem_dst_vals), params.num_items, params.num_segments,
                params.gmem_begin_offsets, params.gmem_end_offsets, 0, sizeof(Key_Data_Type) * 8, stream);
        }
        else
        {
            cub::DeviceRadixSort::SortPairsDescending(cub_temp_storage, cub_temp_storage_bytes,
                reinterpret_cast<Key_Data_Type*>(params.gmem_src_keys),
                reinterpret_cast<Key_Data_Type*>(params.gmem_dst_keys),
                reinterpret_cast<Value_Data_Type*>(params.gmem_src_vals),
                reinterpret_cast<Value_Data_Type*>(params.gmem_dst_vals), params.num_items, 0,
                sizeof(Key_Data_Type) * 8, stream);
        }
    }
    else
    {
        // run at max supported value
        blockSort<Key_Data_Type>((const Key_Data_Type*) (params.gmem_dst_keys), (Key_Data_Type*) (params.gmem_dst_keys),
            (const Value_Data_Type*) (params.gmem_dst_vals), (Value_Data_Type*) (params.gmem_dst_vals),
            params.gmem_active_count_per_segment, max_num_items, num_items_per_segment, params.num_segments, stream);
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

int topPPerSegment(const TopKPerSegmentContext& context, TopKPerSegmentParams& params, const DType_t DT_SCORE,
    void* temp_storage, size_t& temp_storage_bytes, cudaStream_t stream)
{
    int num_items_per_segment = params.num_items / params.num_segments;
    if (DT_SCORE == kFLOAT)
    {
        if (num_items_per_segment % 2 == 0)
        {
            topPPerSegment_dispatch<kernel_params_float>(context, params, temp_storage, temp_storage_bytes, stream);
        }
        else
        {
            topPPerSegment_dispatch<kernel_params_float_1>(context, params, temp_storage, temp_storage_bytes, stream);
        }
    }
    else
    {
        if (num_items_per_segment % 4 == 0)
        {
            topPPerSegment_dispatch<kernel_params_half>(context, params, temp_storage, temp_storage_bytes, stream);
        }
        else
        {
            topPPerSegment_dispatch<kernel_params_half_1>(context, params, temp_storage, temp_storage_bytes, stream);
        }
    }

    return 0;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace segmented_topp_impl

__global__ void topPInitialize(
    int* topp_id_val_buf, int* topp_offset_buf, int* begin_topp_offset_buf_, const int batch_size, const int n)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    if (bid == 0)
    {
        for (int i = tid; i < batch_size + 1; i += blockDim.x)
        {
            topp_offset_buf[i] = i * n;
            begin_topp_offset_buf_[i] = topp_offset_buf[i];
        }
    }

    int index = tid + bid * blockDim.x;

    while (index < batch_size * n)
    {
        topp_id_val_buf[index] = index % n;
        index += blockDim.x * gridDim.x;
    }
}

void invokeTopPInitialize(int* topp_id_val_buf, int* topp_offset_buf, int* begin_topp_offset_buf_,
    const size_t batch_size, const int n, cudaStream_t stream)
{
    // n: the column number of logits_buffer for top_p sampling
    topPInitialize<<<32, 512, 0, stream>>>(topp_id_val_buf, topp_offset_buf, begin_topp_offset_buf_, batch_size, n);
}

template <typename T, int MAX_K, int THREADBLOCK_SIZE>
__launch_bounds__(THREADBLOCK_SIZE) __global__ void topp_beam_topk_kernel(const T* log_probs, // prob.
    int* topk_tmp_id_buf, T* topk_tmp_val_buf, const int vocab_size, int* offset_buf, int* begin_offset_buf,
    const float top_p, const float* top_ps, const bool* skip_decode)
{
    int thread_id = threadIdx.x;
    int batch_id = blockIdx.x;
    if (skip_decode != nullptr && skip_decode[batch_id])
    {
        return;
    }
    float p_threshold = (top_ps != nullptr) ? top_ps[batch_id] : top_p;

    typedef cub::BlockReduce<TopK<T, MAX_K>, THREADBLOCK_SIZE> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    TopK<T, MAX_K> partial;

    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;

#pragma unroll
    for (int i = 0; i < MAX_K; ++i)
    {
        partial.p[i] = -1;
        partial.u[i] = -MAX_T_VAL;
    }

#pragma unroll
    for (int elem_id = thread_id; elem_id < vocab_size; elem_id += THREADBLOCK_SIZE)
    {
        int index = elem_id + batch_id * vocab_size;
        partial.insert(log_probs[index], index);
    }

    TopK<T, MAX_K> total = BlockReduce(temp_storage).Reduce(partial, reduce_topk_op<T, MAX_K>);

    if (thread_id == 0)
    {
        begin_offset_buf[batch_id] = offset_buf[batch_id];
        T sum_prob = (T) (0.0f);

#pragma unroll
        for (int i = 0; i < MAX_K; i++)
        {
            sum_prob += total.u[i];
        }

        if ((float) sum_prob >= p_threshold)
        {
            begin_offset_buf[batch_id] += vocab_size;
            int index = batch_id * vocab_size;

#pragma unroll
            for (int i = 0; i < MAX_K; ++i)
            {
                topk_tmp_id_buf[index + i] = total.p[i] % vocab_size;
                topk_tmp_val_buf[index + i] = total.u[i];
            }
        }
    }
}

struct BlockPrefixCallbackOp
{
    // Running prefix
    float running_total;

    // Constructor
    __device__ BlockPrefixCallbackOp(float running_total)
        : running_total(running_total)
    {
    }

    // Callback operator to be entered by the first warp of threads in the block.
    // Thread-0 is responsible for returning a value for seeding the block-wide
    // scan.
    __device__ float operator()(float block_aggregate)
    {
        float old_prefix = running_total;
        running_total += block_aggregate;
        return old_prefix;
    }
};

template <typename T, int BLOCK_SIZE>
__global__ void topp_sampling(T* sorted_log_probs, int* sorted_id_vals, int** ids, int* sequence_length,
    bool* finished_buf, float* cum_log_probs, float* output_log_probs, const int* begin_offset_buf,
    const int* offset_buf, const int vocab_size, curandState_t* curandstate, const float top_p, const float* top_ps,
    const int* end_ids, const int batch_size, const bool* skip_decode)
{
    __shared__ int stop_shared;
    __shared__ float rand_num_s;

    const int tid = threadIdx.x;
    const int batch_id = blockIdx.x;
    if (skip_decode != nullptr && skip_decode[batch_id])
    {
        return;
    }

    constexpr int WARP_SIZE = 32;
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const float prob_threshold = (top_ps != nullptr) ? top_ps[batch_id] : top_p;
    const int current_step = sequence_length[batch_id];

    if (threadIdx.x == 0)
    {
        stop_shared = 0;
        rand_num_s = curand_uniform(curandstate + blockIdx.x) * prob_threshold;
    }

    // if begin_offset_buf and offset_buf of sorting have same value,
    // this means that we have find best one in beam_topK_kernel_for_topP
    // and skip the sorting. So, we can skip then during sampling.
    if (begin_offset_buf[batch_id] == offset_buf[batch_id])
    {
        if (tid == 0)
        {
            int offset = batch_id * vocab_size;
            ids[batch_id][current_step] = sorted_id_vals[offset];

            if (cum_log_probs != nullptr || output_log_probs != nullptr)
            {
                float lprob = logf(sorted_log_probs[offset]);
                if (cum_log_probs != nullptr)
                {
                    cum_log_probs[batch_id] += lprob;
                }
                if (output_log_probs != nullptr)
                {
                    output_log_probs[batch_id] = lprob;
                }
            }
            if (sequence_length != nullptr && finished_buf != nullptr)
            {
                sequence_length[batch_id]
                    = finished_buf[batch_id] ? sequence_length[batch_id] : sequence_length[batch_id] + 1;
                finished_buf[batch_id] = ids[batch_id][current_step] == end_ids[batch_id] ? 1 : 0;
            }
        }
        return;
    }

    typedef cub::BlockScan<float, BLOCK_SIZE> BlockScan;
    __shared__ typename BlockScan::TempStorage temp_storage;
    __shared__ uint32_t selected_shared[NUM_WARPS];
    // Initialize running total
    BlockPrefixCallbackOp prefix_op(0);

    if (lane_id == 0)
    {
        selected_shared[warp_id] = 0;
    }

    __syncthreads();

    int offset = batch_id * vocab_size;
    ids[batch_id][current_step] = sorted_id_vals[offset];
    int end = ((vocab_size + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    int i_active = 0;
    float thread_offset = 0;
    for (int i = tid; i < end; i += BLOCK_SIZE)
    {
        float thread_count = (i < vocab_size) ? (float) sorted_log_probs[offset + i] : 0.f;
        BlockScan(temp_storage).InclusiveSum(thread_count, thread_offset, prefix_op);

        uint32_t active_mask = __ballot_sync(0xFFFFFFFF, rand_num_s <= thread_offset);

        i_active = i;
        if (active_mask != 0)
        {
            if (lane_id == 0)
            {
                atomicAdd(&stop_shared, 1);
                selected_shared[warp_id] = active_mask;
            }
        }
        __syncthreads();
        if (stop_shared > 0)
        {
            break;
        }
    };

    // select first active warp
    bool skip = (selected_shared[warp_id] > 0) ? false : true;
    for (int i = 0; i < warp_id; i++)
    {
        if (selected_shared[i] != 0)
        {
            skip = true;
        }
    }
    if (!skip)
    {
        int active_lane_id = WARP_SIZE - __popc(selected_shared[warp_id]);
        if (lane_id == active_lane_id)
        {
            ids[batch_id][current_step] = sorted_id_vals[offset + i_active];
            if (cum_log_probs != nullptr || output_log_probs != nullptr)
            {
                float lprob = logf(sorted_log_probs[offset + i_active]);
                if (cum_log_probs != nullptr)
                {
                    cum_log_probs[batch_id] += lprob;
                }
                if (output_log_probs != nullptr)
                {
                    output_log_probs[batch_id] = lprob;
                }
            }
            if (sequence_length != nullptr && finished_buf != nullptr)
            {
                sequence_length[batch_id]
                    = finished_buf[batch_id] ? sequence_length[batch_id] : sequence_length[batch_id] + 1;
                finished_buf[batch_id] = ids[batch_id][current_step] == end_ids[batch_id] ? 1 : 0;
            }
        }
    }
}

template <typename T>
void invokeBatchTopPSampling(void* workspace, size_t& workspace_size, size_t& cub_temp_storage_size, int** output_ids,
    int* sequence_length, bool* finished_buf, float* cum_log_probs, float* output_log_probs, const T* log_probs,
    const int* id_vals, int* offset_buf, int* begin_offset_buf, curandState_t* curandstate, const int batch_size,
    const size_t vocab_size_padded, const int* end_ids, const float max_top_p, const float* top_ps, cudaStream_t stream,
    cudaDeviceProp* cuda_device_prop, const bool* skip_decode)
{
    // Here, we put batch size as an argument because the batch size of
    // initialization and inference may be different due to pipeline parallelism.
    const int vocab_size = vocab_size_padded;
    const int block_size = 256;

    size_t sorted_log_prob_buf_size = batch_size * vocab_size * sizeof(T);  // type T
    size_t sorted_id_vals_buf_size = batch_size * vocab_size * sizeof(int); // type int
    sorted_log_prob_buf_size = divUp(sorted_log_prob_buf_size, 256) * 256;
    sorted_id_vals_buf_size = divUp(sorted_id_vals_buf_size, 256) * 256;

    void* cub_temp_storage = workspace;
    T* sorted_log_probs = (T*) ((char*) cub_temp_storage + cub_temp_storage_size);
    int* sorted_id_vals = (int*) ((char*) sorted_log_probs + sorted_log_prob_buf_size);

    bool do_radix_sort = (ENABLE_SINGLE_PASS_TOP_P == 0 || max_top_p >= SINGLE_PASS_THRESHOLD);
    int smem_size = -1;

    segmented_topp_impl::TopKPerSegmentContext context;
    segmented_topp_impl::TopKPerSegmentParams params;
    segmented_topp_impl::DType_t dataTypeKind
        = (std::is_same<T, float>::value) ? segmented_topp_impl::kFLOAT : segmented_topp_impl::kHALF;

    if (!do_radix_sort)
    {
        TLLM_CHECK(cuda_device_prop != nullptr);
        memset(&context, 0, sizeof(context));
        context.sm_count = cuda_device_prop->multiProcessorCount;
        context.sm_shared_size = cuda_device_prop->sharedMemPerMultiprocessor;
        context.sm_version = cuda_device_prop->major * 100 + cuda_device_prop->minor * 10;

        memset(&params, 0, sizeof(params));
        params.gmem_src_keys = reinterpret_cast<void*>(const_cast<T*>(log_probs));
        params.gmem_dst_keys = sorted_log_probs;
        params.gmem_src_vals = reinterpret_cast<void*>(const_cast<int*>(id_vals));
        params.gmem_dst_vals = reinterpret_cast<void*>(sorted_id_vals);
        params.gmem_begin_offsets = begin_offset_buf;
        params.gmem_end_offsets = offset_buf + 1;
        params.workspace = nullptr;
        params.num_items = vocab_size * batch_size;
        params.num_segments = batch_size;
        params.top_p = max_top_p;
        params.confidence_threshold = 0.0F;

        smem_size = getSmemSizeAndCheck(context, params, dataTypeKind);
        do_radix_sort = smem_size < 0;
    }

    if (do_radix_sort)
    {
        if (workspace == nullptr)
        {
            check_cuda_error(cub::DeviceSegmentedRadixSort::SortPairsDescending(nullptr, cub_temp_storage_size,
                log_probs, (T*) nullptr, id_vals, (int*) nullptr, vocab_size * batch_size, batch_size, begin_offset_buf,
                offset_buf + 1,
                0,             // begin_bit
                sizeof(T) * 8, // end_bit = sizeof(KeyT) * 8
                stream));      // cudaStream_t
            cub_temp_storage_size = divUp(cub_temp_storage_size, 256) * 256;
            workspace_size = sorted_log_prob_buf_size + sorted_id_vals_buf_size + cub_temp_storage_size;
            return;
        }

        topp_beam_topk_kernel<T, 1, block_size><<<batch_size, block_size, 0, stream>>>(log_probs, sorted_id_vals,
            sorted_log_probs, vocab_size, offset_buf, begin_offset_buf, max_top_p, top_ps, skip_decode);

        check_cuda_error(cub::DeviceSegmentedRadixSort::SortPairsDescending(cub_temp_storage, cub_temp_storage_size,
            log_probs, sorted_log_probs, id_vals, sorted_id_vals, vocab_size * batch_size, batch_size, begin_offset_buf,
            offset_buf + 1,
            0,             // begin_bit
            sizeof(T) * 8, // end_bit = sizeof(KeyT) * 8
            stream));      // cudaStream_t
    }
    else
    {
        if (workspace == nullptr)
        {
            segmented_topp_impl::topPPerSegment(
                context, params, dataTypeKind, cub_temp_storage, cub_temp_storage_size, stream);
            workspace_size = sorted_log_prob_buf_size + sorted_id_vals_buf_size + cub_temp_storage_size;
            return;
        }
        else
        {
            topp_beam_topk_kernel<T, 1, block_size><<<batch_size, block_size, 0, stream>>>(log_probs, sorted_id_vals,
                sorted_log_probs, vocab_size, offset_buf, begin_offset_buf, max_top_p, top_ps, skip_decode);
            segmented_topp_impl::topPPerSegment(
                context, params, dataTypeKind, cub_temp_storage, cub_temp_storage_size, stream);
        }
    }

    constexpr int SAMPLING_BLOCK_SIZE = 256;
    dim3 grid(batch_size);
    topp_sampling<T, SAMPLING_BLOCK_SIZE><<<grid, SAMPLING_BLOCK_SIZE, 0, stream>>>(sorted_log_probs, sorted_id_vals,
        output_ids, sequence_length, finished_buf, cum_log_probs, output_log_probs, begin_offset_buf, offset_buf + 1,
        vocab_size, curandstate, max_top_p, top_ps, end_ids, batch_size, skip_decode);
}

template void invokeBatchTopPSampling(void* workspace, size_t& workspace_size, size_t& cub_temp_storage_size,
    int** output_ids, int* sequence_length, bool* finished_buf, float* cum_log_probs, float* output_log_probs,
    const float* log_probs, const int* id_vals, int* offset_buf, int* begin_offset_buf, curandState_t* curandstate,
    const int batch_size, const size_t vocab_size_padded, const int* end_ids, const float max_top_p,
    const float* top_ps, cudaStream_t stream, cudaDeviceProp* cuda_device_prop, const bool* skip_decode);

template void invokeBatchTopPSampling(void* workspace, size_t& workspace_size, size_t& cub_temp_storage_size,
    int** output_ids, int* sequence_length, bool* finished_buf, float* cum_log_probs, float* output_log_probs,
    const half* log_probs, const int* id_vals, int* offset_buf, int* begin_offset_buf, curandState_t* curandstate,
    const int batch_size, const size_t vocab_size_padded, const int* end_ids, const float max_top_p,
    const float* top_ps, cudaStream_t stream, cudaDeviceProp* cuda_device_prop, const bool* skip_decode);

template <typename T>
void invokeTopPSampling(void* workspace, size_t& workspace_size, size_t& cub_temp_storage_size, int** output_ids,
    int* sequence_length, bool* finished_buf, float* cum_log_probs, float* output_log_probs, const T* log_probs,
    const int* id_vals, int* offset_buf, int* begin_offset_buf, curandState_t* curandstate, const int batch_size,
    const size_t vocab_size_padded, const int* end_ids, const float top_p, cudaStream_t stream,
    cudaDeviceProp* cuda_device_prop, const bool* skip_decode)
{
    invokeBatchTopPSampling(workspace, workspace_size, cub_temp_storage_size, output_ids, sequence_length, finished_buf,
        cum_log_probs, output_log_probs, log_probs, id_vals, offset_buf, begin_offset_buf, curandstate, batch_size,
        vocab_size_padded, end_ids, top_p, nullptr, stream, cuda_device_prop, skip_decode);
}

template void invokeTopPSampling(void* workspace, size_t& workspace_size, size_t& cub_temp_storage_size,
    int** output_ids, int* sequence_length, bool* finished_buf, float* cum_log_probs, float* output_log_probs,
    const float* log_probs, const int* id_vals, int* offset_buf, int* begin_offset_buf, curandState_t* curandstate,
    const int batch_size, const size_t vocab_size_padded, const int* end_ids, const float top_p, cudaStream_t stream,
    cudaDeviceProp* cuda_device_prop, const bool* skip_decode);

template void invokeTopPSampling(void* workspace, size_t& workspace_size, size_t& cub_temp_storage_size,
    int** output_ids, int* sequence_length, bool* finished_buf, float* cum_log_probs, float* output_log_probs,
    const half* log_probs, const int* id_vals, int* offset_buf, int* begin_offset_buf, curandState_t* curandstate,
    const int batch_size, const size_t vocab_size_padded, const int* end_ids, const float top_p, cudaStream_t stream,
    cudaDeviceProp* cuda_device_prop, const bool* skip_decode);

template <typename T>
__global__ void addBiasSoftMax(
    T* logits, const T* bias, const int* end_ids, const bool* finished, const int n_padded, const int n)
{
    int bid = blockIdx.x;
    bool finish = (finished != nullptr) ? finished[bid] : false;
    int offset = bid * n_padded;

    float max_val = -1 * FLT_MAX;
    const bool IS_FP16 = std::is_same<T, half>::value;
    const T MAX_T_VAL = (IS_FP16) ? HALF_FLT_MAX : FLT_MAX;
    __shared__ float s_max_val;
    __shared__ float s_sum_val;

    for (int tid = threadIdx.x; tid < n_padded; tid += blockDim.x)
    {
        if (tid < n)
        {
            if (finish)
            {
                logits[offset + tid] = (tid == end_ids[bid]) ? MAX_T_VAL : -MAX_T_VAL;
            }
            else
            {
                T bias_val = (bias != nullptr) ? bias[tid] : (T) 0.0f;
                logits[offset + tid] += bias_val;
            }
        }
        else
        {
            logits[offset + tid] = -MAX_T_VAL;
        }
        max_val = max(max_val, (float) logits[offset + tid]);
    }

    max_val = blockReduceMax<float>((float) max_val);
    if (threadIdx.x == 0)
    {
        s_max_val = max_val;
    }
    __syncthreads();

    float sum_val = 0.0f;
    for (int tid = threadIdx.x; tid < n_padded; tid += blockDim.x)
    {
        logits[offset + tid] = __expf((float) logits[offset + tid] - s_max_val);
        sum_val += (float) logits[offset + tid];
    }

    sum_val = blockReduceSum<float>(sum_val);
    if (threadIdx.x == 0)
    {
        s_sum_val = sum_val;
    }
    __syncthreads();

    for (int tid = threadIdx.x; tid < n_padded; tid += blockDim.x)
    {
        logits[offset + tid] = ((float) logits[offset + tid] / (s_sum_val + 1e-6f));
    }
}

template <typename T>
void invokeAddBiasSoftMax(T* logits, const T* bias, const int* end_ids, const bool* finished, const int m,
    const int n_padded, const int n, cudaStream_t stream)
{
    dim3 grid(m);
    dim3 block(min(n, 1024));
    /*n is the vocab_size, e.g., 30000, 7000.... vocab_size is usually very big.
     */
    addBiasSoftMax<<<grid, block, 0, stream>>>(logits, bias, end_ids, finished, n_padded, n);
}

template void invokeAddBiasSoftMax(float* logits, const float* bias, const int* end_ids, const bool* finished,
    const int m, const int n_padded, const int n, cudaStream_t stream);

template void invokeAddBiasSoftMax(half* logits, const half* bias, const int* end_ids, const bool* finished,
    const int m, const int n_padded, const int n, cudaStream_t stream);

__global__ void computeToppDecay(float* runtime_top_p, const float* runtime_initial_top_p, const int** output_ids,
    const float* top_p_decay, const float* top_p_min, const int32_t* top_p_reset_ids, const int* sequence_lengths)
{
    /**
     * @brief Compute the topp decay by https://arxiv.org/pdf/2206.04624.pdf
     *        In short, the formula is
     *          runtime_top_p = max(runtime_top_p * top_p_decay, top_p_min)
     *        If generating the top_p_reset_ids, then reset the runtime_top_p.
     *
     * \param runtime_top_p          [local_batch_size]
     * \param runtime_initial_top_p  [local_batch_size]
     * \param output_ids             [local_batch_size]
     * \param top_p_decay            [local_batch_size]
     * \param top_p_min              [local_batch_size]
     * \param top_p_reset_ids         [local_batch_size]
     * \param local_batch_size
     *
     */

    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const auto current_step{sequence_lengths[idx]};
    if (output_ids[idx][current_step] == top_p_reset_ids[idx])
    {
        runtime_top_p[idx] = runtime_initial_top_p[idx];
    }
    else
    {
        runtime_top_p[idx] = max(runtime_top_p[idx] * top_p_decay[idx], top_p_min[idx]);
    }
}

void invokeComputeToppDecay(float* runtime_top_p, const float* runtime_initial_top_p, const int** output_ids,
    const float* top_p_decay, const float* top_p_min, const int32_t* top_p_reset_ids, const int* sequence_lengths,
    const int local_batch_size, cudaStream_t stream)
{
    dim3 block(min(local_batch_size, 512));
    dim3 grid((local_batch_size + block.x - 1) / block.x);
    computeToppDecay<<<grid, block, 0, stream>>>(
        runtime_top_p, runtime_initial_top_p, output_ids, top_p_decay, top_p_min, top_p_reset_ids, sequence_lengths);
}

} // namespace kernels
} // namespace tensorrt_llm
