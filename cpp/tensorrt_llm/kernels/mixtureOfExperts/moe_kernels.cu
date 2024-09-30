/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/common/workspace.h"
#include <algorithm>
#include <cuda.h>
#include <cuda_fp16.h>
#include <float.h>
#include <math.h>
#include <numeric>
#include <random>
#include <sstream>

// Ignore CUTLASS warnings about type punning
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#include "cute/tensor.hpp"
#include "cutlass/conv/convolution.h"
// Order matters here, packed_stride.hpp is missing cute and convolution includes
#include "cutlass/util/packed_stride.hpp"

#include "cutlass/array.h"
#include "cutlass/epilogue/thread/activation.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/numeric_types.h"

#include "cutlass_extensions/epilogue/thread/fused_activations.h"

#pragma GCC diagnostic pop

#include "tensorrt_llm/common/cudaUtils.h"
#include "tensorrt_llm/common/dataType.h"
#include "tensorrt_llm/kernels/mixtureOfExperts/moe_kernels.h"

#ifndef CUDART_VERSION
#error CUDART_VERSION Undefined!
#elif (CUDART_VERSION >= 11050)
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/util_type.cuh>
#include <curand_kernel.h>
#include <curand_philox4x32_x.h>
#else
#include "3rdparty/cub/cub.cuh"
#include "3rdparty/cub/device/device_radix_sort.cuh"
#include "3rdparty/cub/util_type.cuh"
#endif

using namespace tensorrt_llm::kernels;
using namespace tensorrt_llm::common;

namespace tensorrt_llm::kernels
{

static constexpr int WARP_SIZE = 32;

// ====================== Softmax things ===============================
// We have our own implementation of softmax here so we can support transposing the output
// in the softmax kernel when we extend this module to support expert-choice routing.
template <int TPB>
__launch_bounds__(TPB) __global__
    void moeSoftmax(float const* input, bool const* finished, float* output, int64_t const num_cols)
{
    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    __shared__ float normalizing_factor;
    __shared__ float float_max;

    int64_t const thread_row_offset = blockIdx.x * num_cols;

    cub::Sum sum;
    float threadData(-FLT_MAX);

    // Don't touch finished rows.
    if ((finished != nullptr) && finished[blockIdx.x])
    {
        return;
    }

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        int64_t const idx = thread_row_offset + ii;
        threadData = max(input[idx], threadData);
    }

    float const maxElem = BlockReduce(tmpStorage).Reduce(threadData, cub::Max());
    if (threadIdx.x == 0)
    {
        float_max = maxElem;
    }
    __syncthreads();

    threadData = 0;

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        int64_t const idx = thread_row_offset + ii;
        threadData += exp((static_cast<float>(input[idx]) - float_max));
    }

    auto const Z = BlockReduce(tmpStorage).Reduce(threadData, sum);

    if (threadIdx.x == 0)
    {
        normalizing_factor = 1.f / Z;
    }
    __syncthreads();

    for (int ii = threadIdx.x; ii < num_cols; ii += TPB)
    {
        int64_t const idx = thread_row_offset + ii;
        float const val = exp((static_cast<float>(input[idx]) - float_max)) * normalizing_factor;
        output[idx] = val;
    }
}

template <int TPB>
__launch_bounds__(TPB) __global__ void moeTopK(float const* inputs_after_softmax, bool const* finished, float* output,
    int* indices, int* source_rows, int const num_experts, int const k, int const startk, int const endk,
    int const start_expert, int const end_expert, MOEExpertScaleNormalizationMode norm_mode)
{

    using cub_kvp = cub::KeyValuePair<int, float>;
    using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmpStorage;

    cub_kvp thread_kvp;
    cub::ArgMax arg_max;

    int64_t const num_rows = gridDim.x;
    int64_t const block_row = blockIdx.x;

    float renorm_value = 0.0f;
    bool const row_is_active = finished ? !finished[block_row] : true;
    int64_t const thread_read_offset = blockIdx.x * num_experts;
    for (int k_idx = startk; k_idx < endk; ++k_idx)
    {
        thread_kvp.key = 0;
        thread_kvp.value = -1.f; // This is OK because inputs are probabilities

        cub_kvp inp_kvp;
        for (int expert = threadIdx.x; expert < num_experts; expert += TPB)
        {
            int64_t const idx = thread_read_offset + expert;
            inp_kvp.key = expert;
            inp_kvp.value = inputs_after_softmax[idx];

            for (int prior_k = startk; prior_k < k_idx; ++prior_k)
            {
                int const prior_winning_expert = indices[k * block_row + prior_k];

                if (prior_winning_expert == expert)
                {
                    inp_kvp = thread_kvp;
                }
            }

            thread_kvp = arg_max(inp_kvp, thread_kvp);
        }

        cub_kvp const result_kvp = BlockReduce(tmpStorage).Reduce(thread_kvp, arg_max);
        if (threadIdx.x == 0)
        {
            // Ignore experts the node isn't responsible for with expert parallelism
            int const expert = result_kvp.key;
            bool const node_uses_expert = expert >= start_expert && expert < end_expert;
            bool const should_process_row = row_is_active && node_uses_expert;

            int64_t const idx = k * block_row + k_idx;
            output[idx] = result_kvp.value;
            indices[idx] = should_process_row ? (expert - start_expert) : (num_experts + expert);
            assert(indices[idx] >= 0);
            source_rows[idx] = k_idx * num_rows + block_row;

            if (norm_mode == MOEExpertScaleNormalizationMode::RENORMALIZE)
            {
                renorm_value += result_kvp.value;
            }
        }
        __syncthreads();
    }

    if (norm_mode == MOEExpertScaleNormalizationMode::RENORMALIZE && threadIdx.x == 0 && renorm_value != 0.f)
    {
        assert(startk == 0 && endk == k);
        renorm_value = 1 / renorm_value;
        for (int k_idx = 0; k_idx < k; k_idx++)
        {
            int64_t const idx = k * block_row + k_idx;
            output[idx] *= renorm_value;
        }
    }
}

// ====================== TopK softmax things ===============================

/*
  A Top-K gating softmax written to exploit when the number of experts in the MoE layers
  are a small power of 2. This allows us to cleanly share the rows among the threads in
  a single warp and eliminate communication between warps (so no need to use shared mem).

  It fuses the softmax, max and argmax into a single kernel.

  Limitations:
  1) This implementation is intended for when the number of experts is a small power of 2.
  2) This implementation assumes k is small, but will work for any k.
*/

template <int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA* WARP_SIZE) __global__ void topkGatingSoftmax(float const* input, bool const* finished,
    float* output, int64_t const num_rows, int* indices, int* source_rows, int const k, int const startk,
    int const endk, int const start_expert, int const end_expert, MOEExpertScaleNormalizationMode norm_mode)
{
    // We begin by enforcing compile time assertions and setting up compile time constants.
    static_assert(VPT == (VPT & -VPT), "VPT must be power of 2");
    static_assert(NUM_EXPERTS == (NUM_EXPERTS & -NUM_EXPERTS), "NUM_EXPERTS must be power of 2");
    static_assert(BYTES_PER_LDG == (BYTES_PER_LDG & -BYTES_PER_LDG), "BYTES_PER_LDG must be power of 2");
    static_assert(BYTES_PER_LDG <= 16, "BYTES_PER_LDG must be leq 16");

    // Number of bytes each thread pulls in per load
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(float);
    static constexpr int ELTS_PER_ROW = NUM_EXPERTS;
    static constexpr int THREADS_PER_ROW = ELTS_PER_ROW / VPT;
    static constexpr int LDG_PER_THREAD = VPT / ELTS_PER_LDG;

    // Restrictions based on previous section.
    static_assert(VPT % ELTS_PER_LDG == 0, "The elements per thread must be a multiple of the elements per ldg");
    static_assert(WARP_SIZE % THREADS_PER_ROW == 0, "The threads per row must cleanly divide the threads per warp");
    static_assert(THREADS_PER_ROW == (THREADS_PER_ROW & -THREADS_PER_ROW), "THREADS_PER_ROW must be power of 2");
    static_assert(THREADS_PER_ROW <= WARP_SIZE, "THREADS_PER_ROW can be at most warp size");

    // We have NUM_EXPERTS elements per row. We specialize for small #experts
    static constexpr int ELTS_PER_WARP = WARP_SIZE * VPT;
    static constexpr int ROWS_PER_WARP = ELTS_PER_WARP / ELTS_PER_ROW;
    static constexpr int ROWS_PER_CTA = WARPS_PER_CTA * ROWS_PER_WARP;

    // Restrictions for previous section.
    static_assert(ELTS_PER_WARP % ELTS_PER_ROW == 0, "The elts per row must cleanly divide the total elt per warp");

    // ===================== From this point, we finally start computing run-time variables. ========================

    // Compute CTA and warp rows. We pack multiple rows into a single warp, and a block contains WARPS_PER_CTA warps.
    // This, each block processes a chunk of rows. We start by computing the start row for each block.
    int64_t const cta_base_row = blockIdx.x * ROWS_PER_CTA;

    // Now, using the base row per thread block, we compute the base row per warp.
    int64_t const warp_base_row = cta_base_row + threadIdx.y * ROWS_PER_WARP;

    // The threads in a warp are split into sub-groups that will work on a row.
    // We compute row offset for each thread sub-group
    int const thread_row_in_warp = threadIdx.x / THREADS_PER_ROW;
    int64_t const thread_row = warp_base_row + thread_row_in_warp;

    // Threads with indices out of bounds should early exit here.
    if (thread_row >= num_rows)
    {
        return;
    }
    bool const row_is_active = finished ? !finished[thread_row] : true;

    // We finally start setting up the read pointers for each thread. First, each thread jumps to the start of the
    // row it will read.
    float const* thread_row_ptr = input + thread_row * ELTS_PER_ROW;

    // Now, we compute the group each thread belong to in order to determine the first column to start loads.
    int const thread_group_idx = threadIdx.x % THREADS_PER_ROW;
    int const first_elt_read_by_thread = thread_group_idx * ELTS_PER_LDG;
    float const* thread_read_ptr = thread_row_ptr + first_elt_read_by_thread;

    // Determine the pointer type to use to read in the data depending on the BYTES_PER_LDG template param. In theory,
    // this can support all powers of 2 up to 16.
    using AccessType = cutlass::AlignedArray<float, ELTS_PER_LDG>;

    // Finally, we pull in the data from global mem
    cutlass::Array<float, VPT> row_chunk;
    AccessType* row_chunk_vec_ptr = reinterpret_cast<AccessType*>(&row_chunk);
    AccessType const* vec_thread_read_ptr = reinterpret_cast<AccessType const*>(thread_read_ptr);
#pragma unroll
    for (int ii = 0; ii < LDG_PER_THREAD; ++ii)
    {
        row_chunk_vec_ptr[ii] = vec_thread_read_ptr[ii * THREADS_PER_ROW];
    }

    // First, we perform a max reduce within the thread. We can do the max in fp16 safely (I think) and just
    // convert to float afterwards for the exp + sum reduction.
    float thread_max = row_chunk[0];
#pragma unroll
    for (int ii = 1; ii < VPT; ++ii)
    {
        thread_max = max(thread_max, row_chunk[ii]);
    }

// Now, we find the max within the thread group and distribute among the threads. We use a butterfly reduce.
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        thread_max = max(thread_max, __shfl_xor_sync(0xFFFFFFFF, thread_max, mask, THREADS_PER_ROW));
    }

    // From this point, thread max in all the threads have the max within the row.
    // Now, we subtract the max from each element in the thread and take the exp. We also compute the thread local sum.
    float row_sum = 0;
#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = expf(row_chunk[ii] - thread_max);
        row_sum += row_chunk[ii];
    }

// Now, we perform the sum reduce within each thread group. Similar to the max reduce, we use a bufferfly pattern.
#pragma unroll
    for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
    {
        row_sum += __shfl_xor_sync(0xFFFFFFFF, row_sum, mask, THREADS_PER_ROW);
    }

    // From this point, all threads have the max and the sum for their rows in the thread_max and thread_sum variables
    // respectively. Finally, we can scale the rows for the softmax. Technically, for top-k gating we don't need to
    // compute the entire softmax row. We can likely look at the maxes and only compute for the top-k values in the row.
    // However, this kernel will likely not be a bottle neck and it seems better to closer match torch and find the
    // argmax after computing the softmax.
    float const reciprocal_row_sum = 1.f / row_sum;

#pragma unroll
    for (int ii = 0; ii < VPT; ++ii)
    {
        row_chunk[ii] = row_chunk[ii] * reciprocal_row_sum;
    }

    // Now, softmax_res contains the softmax of the row chunk. Now, I want to find the topk elements in each row, along
    // with the max index.
    int start_col = first_elt_read_by_thread;
    static constexpr int COLS_PER_GROUP_LDG = ELTS_PER_LDG * THREADS_PER_ROW;

    float renorm_value = 0.0f;

    for (int k_idx = startk; k_idx < endk; ++k_idx)
    {
        // First, each thread does the local argmax
        float max_val = row_chunk[0];
        int expert = start_col;
#pragma unroll
        for (int ldg = 0, col = start_col; ldg < LDG_PER_THREAD; ++ldg, col += COLS_PER_GROUP_LDG)
        {
#pragma unroll
            for (int ii = 0; ii < ELTS_PER_LDG; ++ii)
            {
                float val = row_chunk[ldg * ELTS_PER_LDG + ii];

                // No check on the experts here since columns with the smallest index are processed first and only
                // updated if > (not >=)
                if (val > max_val)
                {
                    max_val = val;
                    expert = col + ii;
                }
            }
        }

// Now, we perform the argmax reduce. We use the butterfly pattern so threads reach consensus about the max.
// This will be useful for K > 1 so that the threads can agree on "who" had the max value. That thread can
// then blank out their max with -inf and the warp can run more iterations...
#pragma unroll
        for (int mask = THREADS_PER_ROW / 2; mask > 0; mask /= 2)
        {
            float other_max = __shfl_xor_sync(0xFFFFFFFF, max_val, mask, THREADS_PER_ROW);
            int other_expert = __shfl_xor_sync(0xFFFFFFFF, expert, mask, THREADS_PER_ROW);

            // We want lower indices to "win" in every thread so we break ties this way
            if (other_max > max_val || (other_max == max_val && other_expert < expert))
            {
                max_val = other_max;
                expert = other_expert;
            }
        }

        // Write the max for this k iteration to global memory.
        if (thread_group_idx == 0)
        {
            // Add a guard to ignore experts not included by this node
            bool const node_uses_expert = expert >= start_expert && expert < end_expert;
            bool const should_process_row = row_is_active && node_uses_expert;

            // The lead thread from each sub-group will write out the final results to global memory. (This will be a
            // single) thread per row of the input/output matrices.
            int64_t const idx = k * thread_row + k_idx;
            output[idx] = max_val;
            indices[idx] = should_process_row ? (expert - start_expert) : (NUM_EXPERTS + expert);
            source_rows[idx] = k_idx * num_rows + thread_row;

            // Accumulate renorm scalar
            if (norm_mode == MOEExpertScaleNormalizationMode::RENORMALIZE)
            {
                renorm_value += max_val;
            }
        }

        // Finally, we clear the value in the thread with the current max if there is another iteration to run.
        if (k_idx + 1 < endk)
        {
            int const ldg_group_for_expert = expert / COLS_PER_GROUP_LDG;
            int const thread_to_clear_in_group = (expert / ELTS_PER_LDG) % THREADS_PER_ROW;

            // Only the thread in the group which produced the max will reset the "winning" value to -inf.
            if (thread_group_idx == thread_to_clear_in_group)
            {
                int const offset_for_expert = expert % ELTS_PER_LDG;
                // Safe to set to any negative value since row_chunk values must be between 0 and 1.
                row_chunk[ldg_group_for_expert * ELTS_PER_LDG + offset_for_expert] = -10000.f;
            }
        }
    }

    if (norm_mode == MOEExpertScaleNormalizationMode::RENORMALIZE && thread_group_idx == 0 && renorm_value != 0.f)
    {
        assert(startk == 0 && endk == k);
        renorm_value = 1 / renorm_value;
        for (int k_idx = 0; k_idx < k; k_idx++)
        {
            int64_t const idx = k * thread_row + k_idx;
            output[idx] *= renorm_value;
        }
    }
}

namespace detail
{
// Constructs some constants needed to partition the work across threads at compile time.
template <int EXPERTS, int BYTES_PER_LDG>
struct TopkConstants
{
    static constexpr int ELTS_PER_LDG = BYTES_PER_LDG / sizeof(float);
    static_assert(EXPERTS / (ELTS_PER_LDG * WARP_SIZE) == 0 || EXPERTS % (ELTS_PER_LDG * WARP_SIZE) == 0);
    static constexpr int VECs_PER_THREAD = std::max(1, EXPERTS / (ELTS_PER_LDG * WARP_SIZE));
    static constexpr int VPT = VECs_PER_THREAD * ELTS_PER_LDG;
    static constexpr int THREADS_PER_ROW = EXPERTS / VPT;
    static constexpr int ROWS_PER_WARP = WARP_SIZE / THREADS_PER_ROW;
};
} // namespace detail

template <int EXPERTS, int WARPS_PER_TB>
void topkGatingSoftmaxLauncherHelper(float const* input, bool const* finished, float* output, int* indices,
    int* source_row, int64_t const num_rows, int const k, int const startk, int const endk, int const start_expert,
    int const end_expert, MOEExpertScaleNormalizationMode norm_mode, cudaStream_t stream)
{
    static constexpr std::size_t MAX_BYTES_PER_LDG = 16;

    static constexpr int BYTES_PER_LDG = std::min(MAX_BYTES_PER_LDG, sizeof(float) * EXPERTS);
    using Constants = detail::TopkConstants<EXPERTS, BYTES_PER_LDG>;
    static constexpr int VPT = Constants::VPT;
    static constexpr int ROWS_PER_WARP = Constants::ROWS_PER_WARP;
    int64_t const num_warps = (num_rows + ROWS_PER_WARP - 1) / ROWS_PER_WARP;
    int64_t const num_blocks = (num_warps + WARPS_PER_TB - 1) / WARPS_PER_TB;

    dim3 block_dim(WARP_SIZE, WARPS_PER_TB);
    topkGatingSoftmax<VPT, EXPERTS, WARPS_PER_TB, BYTES_PER_LDG><<<num_blocks, block_dim, 0, stream>>>(
        input, finished, output, num_rows, indices, source_row, k, startk, endk, start_expert, end_expert, norm_mode);
}

void topkGatingSoftmaxKernelLauncher(float const* input, float* output, float* softmax_temp_output, int* indices,
    int* source_row, int64_t const num_rows, int const num_experts, int const k, int const startk, int const endk,
    int const start_expert, int const end_expert, MOEExpertScaleNormalizationMode norm_mode, cudaStream_t stream)
{
    static constexpr int WARPS_PER_TB = 4;

    switch (num_experts)
    {
    case 1:
    {
        topkGatingSoftmaxLauncherHelper<1, WARPS_PER_TB>(input, nullptr, output, indices, source_row, num_rows, k,
            startk, endk, start_expert, end_expert, norm_mode, stream);
        break;
    }
    case 2:
    {
        topkGatingSoftmaxLauncherHelper<2, WARPS_PER_TB>(input, nullptr, output, indices, source_row, num_rows, k,
            startk, endk, start_expert, end_expert, norm_mode, stream);
        break;
    }
    case 4:
    {
        topkGatingSoftmaxLauncherHelper<4, WARPS_PER_TB>(input, nullptr, output, indices, source_row, num_rows, k,
            startk, endk, start_expert, end_expert, norm_mode, stream);
        break;
    }
    case 8:
    {
        topkGatingSoftmaxLauncherHelper<8, WARPS_PER_TB>(input, nullptr, output, indices, source_row, num_rows, k,
            startk, endk, start_expert, end_expert, norm_mode, stream);
        break;
    }
    case 16:
    {
        topkGatingSoftmaxLauncherHelper<16, WARPS_PER_TB>(input, nullptr, output, indices, source_row, num_rows, k,
            startk, endk, start_expert, end_expert, norm_mode, stream);
        break;
    }
    case 32:
    {
        topkGatingSoftmaxLauncherHelper<32, WARPS_PER_TB>(input, nullptr, output, indices, source_row, num_rows, k,
            startk, endk, start_expert, end_expert, norm_mode, stream);
        break;
    }
    case 64:
    {
        topkGatingSoftmaxLauncherHelper<64, WARPS_PER_TB>(input, nullptr, output, indices, source_row, num_rows, k,
            startk, endk, start_expert, end_expert, norm_mode, stream);
        break;
    }
    case 128:
    {
        topkGatingSoftmaxLauncherHelper<128, WARPS_PER_TB>(input, nullptr, output, indices, source_row, num_rows, k,
            startk, endk, start_expert, end_expert, norm_mode, stream);
        break;
    }
    case 256:
    {
        topkGatingSoftmaxLauncherHelper<256, WARPS_PER_TB>(input, nullptr, output, indices, source_row, num_rows, k,
            startk, endk, start_expert, end_expert, norm_mode, stream);
        break;
    }
    default:
    {
        static constexpr int TPB = 256;
        TLLM_CHECK(softmax_temp_output != nullptr);
        moeSoftmax<TPB><<<num_rows, TPB, 0, stream>>>(input, nullptr, softmax_temp_output, num_experts);
        moeTopK<TPB><<<num_rows, TPB, 0, stream>>>(softmax_temp_output, nullptr, output, indices, source_row,
            num_experts, k, startk, endk, start_expert, end_expert, norm_mode);
    }
    }
}

__global__ void sparseMixerMask(float const* input, float* output, int const* indices, int k_idx, int k, int num_tokens,
    int num_experts, int start_expert, float epsilon)
{
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (token_idx >= num_tokens)
    {
        return;
    }

    // Mask out the largest value selected in the previous iteration
    int last_selected = (k_idx > 0) ? indices[k * token_idx + (k_idx - 1)] : INT_MIN;
    // Adjust the selected index to correct for the expert parallel transformation
    last_selected = last_selected >= num_experts ? last_selected - num_experts : last_selected + start_expert;

    // Find the max value in the current row
    float max_val = -INFINITY;
    for (int i = 0; i < num_experts; ++i)
    {
        if (i != last_selected)
        {
            float const val = input[token_idx * num_experts + i];
            max_val = max(val, max_val);
        }
    }

    // Mask out any values that fail the condition '(max - value) / std::max(abs(value), max) > 2 * epsilon'
    for (int i = 0; i < num_experts; ++i)
    {
        float val = input[token_idx * num_experts + i];
        float mask = (max_val - val) / max(abs(val), max_val);
        bool mask_value = (mask > 2 * epsilon) || i == last_selected;
        output[token_idx * num_experts + i] = mask_value ? -INFINITY : val;
    }
}

void sparseMixerTopkSoftmax(float const* input, float* output, float* mixer_temp_output, float* softmax_temp_output,
    int* indices, int* source_row, int64_t const num_rows, int const num_experts, int const k, int const start_expert,
    int const end_expert, float epsilon, cudaStream_t stream)
{
    // TODO we need to update the sparseMixerMask() function to mask all previous experts instead of just the most
    //  recent one.
    TLLM_CHECK_WITH_INFO(k <= 2, "Current sparse mixer only supports k <= 2");

    // Each thread handles one token
    constexpr int threads_per_block = 256;
    int num_blocks = ceilDiv(num_rows, threads_per_block);
    for (int k_idx = 0; k_idx < k; ++k_idx)
    {
        // Run softmax and topk in serial for each selection, recalculating the mask for each step
        sparseMixerMask<<<num_blocks, threads_per_block, 0, stream>>>(
            input, mixer_temp_output, indices, k_idx, k, num_rows, num_experts, start_expert, epsilon);

        topkGatingSoftmaxKernelLauncher(mixer_temp_output, output, softmax_temp_output, indices, source_row, num_rows,
            num_experts, k, k_idx, k_idx + 1, start_expert, end_expert, MOEExpertScaleNormalizationMode::NONE, stream);
    }
}

void selectExpertsForTokens(float const* input, float* output, float* mixer_temp_output, float* softmax_temp_output,
    int* indices, int* source_row, int64_t const num_rows, int const num_experts, int const k, int const start_expert,
    int const end_expert, float mixer_epsilon, MOEExpertScaleNormalizationMode norm_mode, cudaStream_t stream)
{
    if (norm_mode == MOEExpertScaleNormalizationMode::SPARSE_MIXER)
    {
        TLLM_CHECK_WITH_INFO(mixer_temp_output, "Sparse mixer output is null when running sparse mixer");
        sparseMixerTopkSoftmax(input, output, mixer_temp_output, softmax_temp_output, indices, source_row, num_rows,
            num_experts, k, start_expert, end_expert, mixer_epsilon, stream);
    }
    else
    {
        topkGatingSoftmaxKernelLauncher(input, output, softmax_temp_output, indices, source_row, num_rows, num_experts,
            k, 0, k, start_expert, end_expert, norm_mode, stream);
    }
}

// ========================== CUB Sorting things ====================================
CubKeyValueSorter::CubKeyValueSorter()
    : num_experts_(0)
    , num_bits_(sizeof(int) * 8)
{
}

int CubKeyValueSorter::expertsToBits(int num_experts)
{
    // Max value we represent is V = num_experts + (num_experts - 1) = 2 * num_experts - 1
    // The maximum number of bits is therefore floor(log2(V)) + 1
    return static_cast<int>(log2(2 * num_experts - 1)) + 1;
}

CubKeyValueSorter::CubKeyValueSorter(int const num_experts)
    : num_experts_(num_experts)
    , num_bits_(expertsToBits(num_experts))
{
}

void CubKeyValueSorter::updateNumExperts(int const num_experts)
{
    num_experts_ = num_experts;
    num_bits_ = expertsToBits(num_experts);
}

size_t CubKeyValueSorter::getWorkspaceSize(size_t const num_key_value_pairs, int const num_experts)
{
    int num_bits = expertsToBits(num_experts);
    size_t required_storage = 0;
    int* null_int = nullptr;
    cub::DeviceRadixSort::SortPairs(
        nullptr, required_storage, null_int, null_int, null_int, null_int, num_key_value_pairs, 0, num_bits);

    // TODO: fix DeviceRadixSort
    //   when num_key_value_pairs, num_experts, num_bits, required_storage = 64, 4, 3, 0
    //   The required_storage seems to vary between 0 and 1 for the same inputs
    if (required_storage == 0)
    {
        required_storage = 1;
    }
    return required_storage;
}

void CubKeyValueSorter::run(void* workspace, size_t const workspace_size, int const* keys_in, int* keys_out,
    int const* values_in, int* values_out, size_t const num_key_value_pairs, cudaStream_t stream)
{
    size_t expected_ws_size = getWorkspaceSize(num_key_value_pairs, num_experts_);
    size_t actual_ws_size = workspace_size;

    TLLM_CHECK_WITH_INFO(expected_ws_size <= workspace_size,
        "[CubKeyValueSorter::run] The allocated workspace is too small to run this problem.");
    cub::DeviceRadixSort::SortPairs(
        workspace, actual_ws_size, keys_in, keys_out, values_in, values_out, num_key_value_pairs, 0, num_bits_, stream);
}

// ============================== Infer GEMM sizes =================================
// TODO Could linear search be better for small # experts
template <class T>
__device__ inline int64_t findTotalEltsLessThanTarget(T const* sorted_indices, int64_t const arr_length, T const target)
{
    int64_t low = 0, high = arr_length - 1, target_location = -1;
    while (low <= high)
    {
        int64_t mid = (low + high) / 2;

        if (sorted_indices[mid] >= target)
        {
            high = mid - 1;
        }
        else
        {
            low = mid + 1;
            target_location = mid;
        }
    }
    return target_location + 1;
}

// Calculates the start offset of the tokens for a given expert. The last element is the total number of valid tokens
__global__ void computeExpertFirstTokenOffsetKernel(int const* sorted_experts, int64_t const sorted_experts_len,
    int64_t const num_experts, int64_t* expert_first_token_offset)
{
    // First, compute the global tid. We only need 1 thread per expert.
    int const expert = blockIdx.x * blockDim.x + threadIdx.x;

    // Note that expert goes [0, num_experts] (inclusive) because we want a count for the total number of active tokens
    // at the end of the scan.
    if (expert >= num_experts + 1)
    {
        return;
    }
    expert_first_token_offset[expert] = findTotalEltsLessThanTarget(sorted_experts, sorted_experts_len, expert);
}

void computeExpertFirstTokenOffset(int const* sorted_indices, int const total_indices, int const num_experts,
    int64_t* expert_first_token_offset, cudaStream_t stream)
{
    int const num_entries = num_experts + 1;
    int const threads = std::min(1024, num_entries);
    int const blocks = (num_entries + threads - 1) / threads;

    computeExpertFirstTokenOffsetKernel<<<blocks, threads, 0, stream>>>(
        sorted_indices, total_indices, num_experts, expert_first_token_offset);
}

// ====================== Compute FP8 dequant scale only ===============================
__global__ void computeFP8DequantScaleKernel(
    float const** alpha_scale_ptr_array, int64_t const num_experts, float const* fp8_dequant)
{
    // First, compute the global tid. We only need 1 thread per expert.
    int const expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts)
    {
        return;
    }

    assert(fp8_dequant != nullptr);
    alpha_scale_ptr_array[expert] = fp8_dequant + expert;
}

float const** computeFP8DequantScale(
    float const** alpha_scale_ptr_array, int const num_experts, float const* fp8_dequant, cudaStream_t stream)
{
    if (!fp8_dequant)
    {
        return nullptr;
    }

    int const threads = std::min(1024, num_experts);
    int const blocks = (num_experts + threads - 1) / threads;

    computeFP8DequantScaleKernel<<<blocks, threads, 0, stream>>>(alpha_scale_ptr_array, num_experts, fp8_dequant);

    return alpha_scale_ptr_array;
}

__device__ void computeHopperInputStrides(
    HopperGroupedGemmInput layout_info, int gemm_m, int gemm_n, int gemm_k, int64_t out_idx)
{
    layout_info.stride_a[out_idx]
        = cutlass::make_cute_packed_stride(HopperGroupedGemmInput::StrideA{}, cute::make_shape(gemm_m, gemm_k, 1));
    layout_info.stride_b[out_idx]
        = cutlass::make_cute_packed_stride(HopperGroupedGemmInput::StrideB{}, cute::make_shape(gemm_n, gemm_k, 1));
    if (layout_info.stride_c)
    {
        assert(false && "CUTLASS does not support a 1xN bias");
        //        layout_info.stride_c[out_idx] = cute::make_stride(0, cute::Int<1>{}, 0);
        layout_info.stride_c[out_idx]
            = cutlass::make_cute_packed_stride(HopperGroupedGemmInput::StrideC{}, cute::make_shape(1, gemm_n, 1));
    }
    if (layout_info.fusion == HopperGroupedGemmInput::EpilogueFusion::NONE)
    {
        layout_info.default_epilogue.stride_d[out_idx] = cutlass::make_cute_packed_stride(
            HopperGroupedGemmInput::DefaultEpilogue::StrideD{}, cute::make_shape(gemm_n, gemm_m, 1));
    }
}

template <class T, class WeightType, class OutputType>
__device__ void computeHopperInputPointers(HopperGroupedGemmInput layout_info, int64_t gemm_m, int64_t gemm_n,
    int64_t gemm_k, int num_tokens_before_expert, int64_t expert, T const* in, WeightType const* weights, T const* bias,
    OutputType* output, int64_t const out_idx)
{
    // The input prior to this contains K elements per token, with `num_tokens_before_expert` tokens
    layout_info.ptr_a[out_idx] = in + num_tokens_before_expert * gemm_k;

    // Each expert's weight matrix is a constant size NxK, with `expert` experts
    layout_info.ptr_b[out_idx] = weights + expert * (gemm_n * gemm_k);

    //    if (bias)
    //    {
    //        // Each expert's bias is a constant size N, with `expert` experts
    //        layout_info.ptr_c[out_idx] = bias + expert * gemm_n;
    //    }

    if (layout_info.fusion == HopperGroupedGemmInput::EpilogueFusion::NONE)
    {
        // The output prior to this contains N elements per token, with `num_tokens_before_expert` tokens
        layout_info.default_epilogue.ptr_d[out_idx] = output + num_tokens_before_expert * gemm_n;
    }
}

// TODO Some of this setup could be cached
template <class T, class WeightType, class OutputType>
__global__ void computeStridesHopperKernel(int64_t const* expert_first_token_offset, HopperGroupedGemmInput layout_info,
    int64_t gemm_n, int64_t gemm_k, int64_t const num_experts, T const* in, WeightType const* weights,
    float const* fp8_dequant, T const* bias, OutputType* output)
{
    // First, compute the global tid. We only need 1 thread per expert.
    int const expert = blockIdx.x * blockDim.x + threadIdx.x;
    if (expert >= num_experts)
    {
        return;
    }

    auto const num_tokens_before_expert = expert_first_token_offset[expert];
    auto const num_tokens_including_expert = expert_first_token_offset[expert + 1];
    auto const num_tokens_to_expert = num_tokens_including_expert - num_tokens_before_expert;
    auto const gemm_m = num_tokens_to_expert;

    // M and N transposed since we are using the #tokens as the N dimension
    layout_info.shape_info.problem_shapes[expert]
        = HopperGroupedGemmInput::ProblemShape::UnderlyingProblemShape(gemm_n, gemm_m, gemm_k);

    if (fp8_dequant)
    {
        layout_info.alpha_scale_ptr_array[expert] = fp8_dequant + expert;
    }

    assert(gemm_m <= INT32_MAX);
    assert(gemm_n <= INT32_MAX);
    assert(gemm_k <= INT32_MAX);
    computeHopperInputStrides(layout_info, gemm_m, gemm_n, gemm_k, expert);

    computeHopperInputPointers(
        layout_info, gemm_m, gemm_n, gemm_k, num_tokens_before_expert, expert, in, weights, bias, output, expert);
}

// ========================== Permutation things =======================================

template <class T, class U>
__host__ __device__ constexpr static U arrayConvert(T const& input)
{
    using Type = typename U::Element;
    static_assert(T::kElements == U::kElements);
    U u;
#pragma unroll
    for (int i = 0; i < U::kElements; i++)
    {
        u[i] = static_cast<Type>(input[i]);
    }
    return u;
}

// Duplicated and permutes rows for MoE. In addition, reverse the permutation map to help with finalizing routing.

// "expanded_x_row" simply means that the number of values is num_rows x k. It is "expanded" since we will have to
// duplicate some rows in the input matrix to match the dimensions. Duplicates will always get routed to separate
// experts in the end.

// Note that the expanded_dest_row_to_expanded_source_row map referred to here has indices in the range (0,
// k*rows_in_input - 1). However, it is set up so that index 0, rows_in_input, 2*rows_in_input ... (k-1)*rows_in_input
// all map to row 0 in the original matrix. Thus, to know where to read in the source matrix, we simply take the modulus
// of the expanded index.

constexpr static int EXPAND_THREADS_PER_BLOCK = 256;

template <typename T, bool CHECK_SKIPPED>
__global__ void expandInputRowsKernel(T const* unpermuted_input, T* permuted_output, float* unpermuted_scales,
    float* permuted_scales, int const* expanded_dest_row_to_expanded_source_row,
    int* expanded_source_row_to_expanded_dest_row, int64_t const num_rows, int64_t const* num_dest_rows,
    int64_t const cols, int64_t k)
{

    // Reverse permutation map.
    // I do this so that later, we can use the source -> dest map to do the k-way reduction and unpermuting. I need the
    // reverse map for that reduction to allow each threadblock to do 1 k-way reduce without atomics later in MoE. 1
    // thread block will be responsible for all k summations.
    int64_t const expanded_dest_row = blockIdx.x;
    int64_t const expanded_source_row = expanded_dest_row_to_expanded_source_row[expanded_dest_row];
    if (threadIdx.x == 0)
    {
        assert(expanded_dest_row <= INT32_MAX);
        expanded_source_row_to_expanded_dest_row[expanded_source_row] = static_cast<int>(expanded_dest_row);
    }

    if (!CHECK_SKIPPED || blockIdx.x < *num_dest_rows)
    {
        // Load 128-bits per thread
        constexpr int64_t ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<T>::value;
        using DataElem = cutlass::Array<T, ELEM_PER_THREAD>;

        // Duplicate and permute rows
        int64_t const source_k_rank = expanded_source_row / num_rows;
        int64_t const source_row = expanded_source_row % num_rows;

        auto const* source_row_ptr = reinterpret_cast<DataElem const*>(unpermuted_input + source_row * cols);
        auto* dest_row_ptr = reinterpret_cast<DataElem*>(permuted_output + expanded_dest_row * cols);

        int64_t const start_offset = threadIdx.x;
        int64_t const stride = EXPAND_THREADS_PER_BLOCK;
        int64_t const num_elems_in_col = cols / ELEM_PER_THREAD;

        for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
        {
            dest_row_ptr[elem_index] = source_row_ptr[elem_index];
        }

        if (permuted_scales && threadIdx.x == 0)
        {
            int64_t const source_k_idx = source_row * k + source_k_rank;
            permuted_scales[expanded_dest_row] = unpermuted_scales[source_k_idx];
        }
    }
}

template <typename T>
void expandInputRowsKernelLauncher(T const* unpermuted_input, T* permuted_output, float* unpermuted_scales,
    float* permuted_scales, int const* expanded_dest_row_to_expanded_source_row,
    int* expanded_source_row_to_expanded_dest_row, int64_t const num_rows, int64_t const* num_valid_tokens_ptr,
    int64_t const cols, int const k, cudaStream_t stream)
{
    int64_t const blocks = num_rows * k;
    int64_t const threads = EXPAND_THREADS_PER_BLOCK;
    auto func = (num_valid_tokens_ptr != nullptr) ? expandInputRowsKernel<T, true> : expandInputRowsKernel<T, false>;
    func<<<blocks, threads, 0, stream>>>(unpermuted_input, permuted_output, unpermuted_scales, permuted_scales,
        expanded_dest_row_to_expanded_source_row, expanded_source_row_to_expanded_dest_row, num_rows,
        num_valid_tokens_ptr, cols, k);
}

enum class ScaleMode : int
{
    NO_SCALE = 0,
    DEFAULT = 1,
    RENORM_SCALE = 2,
};

constexpr static int FINALIZE_THREADS_PER_BLOCK = 256;

// Final kernel to unpermute and scale
// This kernel unpermutes the original data, does the k-way reduction and performs the final skip connection.
template <typename T, typename OutputType, class GemmOutputType, class ScaleBiasType, ScaleMode SCALE_MODE,
    bool CHECK_SKIPPED>
__global__ void finalizeMoeRoutingKernel(GemmOutputType const* expanded_permuted_rows,
    OutputType* reduced_unpermuted_output, ScaleBiasType const* bias, float const* scales,
    int const* expanded_source_row_to_expanded_dest_row, int const* expert_for_source_row, int64_t const orig_cols,
    int64_t const k, int64_t const* num_valid_ptr)
{
    assert(orig_cols % 4 == 0);
    int64_t const original_row = blockIdx.x;
    int64_t const num_rows = gridDim.x;
    auto const offset = original_row * orig_cols;
    OutputType* reduced_row_ptr = reduced_unpermuted_output + offset;
    int64_t const num_valid = *num_valid_ptr;

    // Load 128-bits per thread, according to the smallest data type we read/write
    constexpr int64_t FINALIZE_ELEM_PER_THREAD
        = 128 / std::min(cutlass::sizeof_bits<OutputType>::value, cutlass::sizeof_bits<GemmOutputType>::value);

    int64_t const start_offset = threadIdx.x;
    int64_t const stride = FINALIZE_THREADS_PER_BLOCK;
    int64_t const num_elems_in_col = orig_cols / FINALIZE_ELEM_PER_THREAD;

    using BiasElem = cutlass::Array<ScaleBiasType, FINALIZE_ELEM_PER_THREAD>;
    using InputElem = cutlass::Array<GemmOutputType, FINALIZE_ELEM_PER_THREAD>;
    using OutputElem = cutlass::Array<OutputType, FINALIZE_ELEM_PER_THREAD>;
    using ComputeElem = cutlass::Array<float, FINALIZE_ELEM_PER_THREAD>;
    auto const* bias_v = reinterpret_cast<BiasElem const*>(bias);
    auto const* expanded_permuted_rows_v = reinterpret_cast<InputElem const*>(expanded_permuted_rows);
    auto* reduced_row_ptr_v = reinterpret_cast<OutputElem*>(reduced_row_ptr);

#pragma unroll
    for (int elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        bool has_valid = false;
        ComputeElem thread_output;
        thread_output.fill(0);
        float row_rescale{0.f};
        for (int k_idx = 0; k_idx < k; ++k_idx)
        {
            int64_t const expanded_original_row = original_row + k_idx * num_rows;
            int64_t const expanded_permuted_row = expanded_source_row_to_expanded_dest_row[expanded_original_row];

            int64_t const k_offset = original_row * k + k_idx;
            float const row_scale = (SCALE_MODE == ScaleMode::NO_SCALE) ? 1.f : scales[k_offset];
            if constexpr (SCALE_MODE == ScaleMode::RENORM_SCALE)
            {
                row_rescale = row_rescale + row_scale;
            }

            // Check after row_rescale has accumulated
            if (CHECK_SKIPPED && expanded_permuted_row >= num_valid)
            {
                continue;
            }

            auto const* expanded_permuted_rows_row_ptr
                = expanded_permuted_rows_v + expanded_permuted_row * num_elems_in_col;

            int64_t const expert_idx = expert_for_source_row[k_offset];

            auto const* bias_ptr = bias_v + expert_idx * num_elems_in_col;
            ComputeElem bias_value;
            if (bias)
            {
                bias_value = arrayConvert<BiasElem, ComputeElem>(bias_ptr[elem_index]);
            }
            else
            {
                bias_value.fill(0);
            }

            ComputeElem expert_result
                = arrayConvert<InputElem, ComputeElem>(expanded_permuted_rows_row_ptr[elem_index]);
            thread_output = thread_output + row_scale * (expert_result + bias_value);
            has_valid = true;
        }

        if (SCALE_MODE == ScaleMode::RENORM_SCALE && (!CHECK_SKIPPED || has_valid))
        {
            assert(row_rescale != 0.f);
            for (auto& elem : thread_output)
            {
                elem /= row_rescale;
            }
        }

        OutputElem output_elem = arrayConvert<ComputeElem, OutputElem>(thread_output);
        reduced_row_ptr_v[elem_index] = output_elem;
    }
}

template <class T, class OutputType, class GemmOutputType, class ScaleBiasType>
void finalizeMoeRoutingKernelLauncher(GemmOutputType const* expanded_permuted_rows,
    OutputType* reduced_unpermuted_output, ScaleBiasType const* bias, float const* scales,
    int const* expanded_source_row_to_expanded_dest_row, int const* expert_for_source_row, int64_t const num_rows,
    int64_t const cols, int64_t const k, int64_t const* num_valid_ptr, MOEParallelismConfig parallelism_config,
    MOEExpertScaleNormalizationMode normalization_mode, cudaStream_t stream)
{
    int64_t const blocks = num_rows;
    int64_t const threads = FINALIZE_THREADS_PER_BLOCK;

    // Only add bias on rank 0 for tensor parallelism
    bool const is_rank_0 = parallelism_config.tp_rank == 0;
    ScaleBiasType const* bias_ptr = is_rank_0 ? bias : nullptr;

    bool const check_finished = num_valid_ptr != nullptr;

    ScaleMode renorm_scales = ScaleMode::DEFAULT;
    if (normalization_mode == MOEExpertScaleNormalizationMode::RENORMALIZE)
    {
        renorm_scales = k == 1 ? ScaleMode::NO_SCALE : ScaleMode::RENORM_SCALE;
    }

    using FuncPtr
        = decltype(&finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleBiasType, ScaleMode::DEFAULT, false>);
    FuncPtr func_map[2][3] = {
        {
            &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleBiasType, ScaleMode::NO_SCALE, false>,
            &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleBiasType, ScaleMode::DEFAULT, false>,
            &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleBiasType, ScaleMode::RENORM_SCALE, false>,
        },
        {
            &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleBiasType, ScaleMode::NO_SCALE, true>,
            &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleBiasType, ScaleMode::DEFAULT, true>,
            &finalizeMoeRoutingKernel<T, OutputType, GemmOutputType, ScaleBiasType, ScaleMode::RENORM_SCALE, true>,
        },
    };
    auto* const func = func_map[check_finished][int(renorm_scales)];
    func<<<blocks, threads, 0, stream>>>(expanded_permuted_rows, reduced_unpermuted_output, bias_ptr, scales,
        expanded_source_row_to_expanded_dest_row, expert_for_source_row, cols, k, num_valid_ptr);
}

// ============================== Gated Activation =================================
constexpr static int ACTIVATION_THREADS_PER_BLOCK = 256;

template <class T, class OutputType, template <class> class ActFn>
__global__ void doGatedActivationKernel(
    T* output, OutputType const* gemm_result, int64_t const* num_valid_tokens_ptr, int64_t inter_size)
{
    int64_t const tid = threadIdx.x;
    int64_t const token = blockIdx.x;
    if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr)
    {
        return;
    }

    output = output + token * inter_size;
    gemm_result = gemm_result + token * inter_size * 2;

    constexpr int64_t ACTIVATION_ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<T>::value;

    using DataElem = cutlass::Array<T, ACTIVATION_ELEM_PER_THREAD>;
    using ComputeElem = cutlass::Array<float, ACTIVATION_ELEM_PER_THREAD>;
    auto gemm_result_vec = reinterpret_cast<DataElem const*>(gemm_result);
    auto output_vec = reinterpret_cast<DataElem*>(output);
    int64_t const start_offset = tid;
    int64_t const stride = ACTIVATION_THREADS_PER_BLOCK;
    assert(inter_size % ACTIVATION_ELEM_PER_THREAD == 0);
    int64_t const num_elems_in_col = inter_size / ACTIVATION_ELEM_PER_THREAD;
    int64_t const inter_size_vec = inter_size / ACTIVATION_ELEM_PER_THREAD;

    ActFn<ComputeElem> fn{};
    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        auto fc1_value = arrayConvert<DataElem, ComputeElem>(gemm_result_vec[elem_index]);
        // BF16 isn't supported, use FP32 for activation function
        auto gate_value = arrayConvert<DataElem, ComputeElem>(gemm_result_vec[elem_index + inter_size_vec]);
        auto gate_act = fn(gate_value);
        output_vec[elem_index] = arrayConvert<ComputeElem, DataElem>(fc1_value * gate_act);
    }
}

template <typename T, typename OutputType>
void doGatedActivation(T* output, OutputType const* gemm_result, int64_t const* num_valid_tokens_ptr,
    int64_t inter_size, int64_t num_tokens, ActivationType activation_type, cudaStream_t stream)
{
    int64_t const blocks = num_tokens;
    int64_t const threads = ACTIVATION_THREADS_PER_BLOCK;

    // TODO For some reason Volta fails on GELU_taylor here with Warp Illegal Instruction.

    auto* fn = activation_type == ActivationType::Swiglu
        ? &doGatedActivationKernel<T, OutputType, cutlass::epilogue::thread::SiLu>
        : &doGatedActivationKernel<T, OutputType, cutlass::epilogue::thread::GELU>;
    fn<<<blocks, threads, 0, stream>>>(output, gemm_result, num_valid_tokens_ptr, inter_size);
}

// ============================== Activation =================================

template <class T, class GemmOutputType, class ScaleBiasType, template <class> class ActFn>
__global__ void doActivationKernel(T* output, GemmOutputType const* gemm_result, float const* fp8_quant,
    ScaleBiasType const* bias_ptr, bool bias_is_broadcast, int64_t const* expert_first_token_offset, int num_experts,
    int64_t inter_size, bool gated)
{
    int64_t const tid = threadIdx.x;
    int64_t const token = blockIdx.x;
    if (token >= expert_first_token_offset[num_experts])
    {
        return;
    }

    size_t gated_size_mul = gated ? 2 : 1;
    size_t gated_off = gated ? inter_size : 0;

    gemm_result = gemm_result + token * inter_size * gated_size_mul;
    output = output + token * inter_size; // Aliases gemm_result for non-gated, non-fp8 cases

    int64_t expert = 0;
    if (bias_ptr)
    {
        // TODO this is almost certainly faster as a linear scan
        expert = findTotalEltsLessThanTarget(expert_first_token_offset, num_experts, (int64_t) token + 1) - 1;
    }

    float const quant_scale = fp8_quant ? *fp8_quant : 1.f;

    if (bias_ptr)
    {
        size_t bias_offset
            = (bias_is_broadcast ? expert * inter_size * gated_size_mul : token * inter_size * gated_size_mul);
        bias_ptr = bias_ptr + bias_offset;
    }

    // Load 128-bits per thread, according to the smallest data type we read/write
    constexpr int64_t ACTIVATION_ELEM_PER_THREAD
        = 128 / std::min(cutlass::sizeof_bits<T>::value, cutlass::sizeof_bits<GemmOutputType>::value);

    using BiasElem = cutlass::Array<ScaleBiasType, ACTIVATION_ELEM_PER_THREAD>;
    using GemmResultElem = cutlass::Array<GemmOutputType, ACTIVATION_ELEM_PER_THREAD>;
    using OutputElem = cutlass::Array<T, ACTIVATION_ELEM_PER_THREAD>;
    using ComputeElem = cutlass::Array<float, ACTIVATION_ELEM_PER_THREAD>;
    auto gemm_result_vec = reinterpret_cast<GemmResultElem const*>(gemm_result);
    auto output_vec = reinterpret_cast<OutputElem*>(output);
    auto bias_ptr_vec = reinterpret_cast<BiasElem const*>(bias_ptr);
    int64_t const start_offset = tid;
    int64_t const stride = ACTIVATION_THREADS_PER_BLOCK;
    assert(inter_size % ACTIVATION_ELEM_PER_THREAD == 0);
    int64_t const num_elems_in_col = inter_size / ACTIVATION_ELEM_PER_THREAD;
    assert(gated_off % ACTIVATION_ELEM_PER_THREAD == 0);
    int64_t const gated_off_vec = gated_off / ACTIVATION_ELEM_PER_THREAD;

    ActFn<ComputeElem> fn{};
    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        auto fc1_value = arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index + gated_off_vec]);
        if (bias_ptr)
        {
            fc1_value = fc1_value + arrayConvert<BiasElem, ComputeElem>(bias_ptr_vec[elem_index + gated_off_vec]);
        }

        auto gate_act = fn(fc1_value);

        if (gated)
        {
            auto gate_mul = arrayConvert<GemmResultElem, ComputeElem>(gemm_result_vec[elem_index]);
            if (bias_ptr_vec)
            {
                gate_mul = gate_mul + arrayConvert<BiasElem, ComputeElem>(bias_ptr_vec[elem_index]);
            }
            gate_act = gate_act * gate_mul;
        }

        output_vec[elem_index] = arrayConvert<ComputeElem, OutputElem>(gate_act * quant_scale);
    }
}

template <class T, class GemmOutputType, class ScaleBiasType>
void doActivation(T* output, GemmOutputType const* gemm_result, float const* fp8_quant, ScaleBiasType const* bias,
    bool bias_is_broadcast, int64_t const* expert_first_token_offset, int num_experts, int64_t inter_size,
    int64_t num_tokens, ActivationType activation_type, cudaStream_t stream)
{
    int64_t const blocks = num_tokens;
    int64_t const threads = ACTIVATION_THREADS_PER_BLOCK;

    auto fn_list = std::array{
        &doActivationKernel<T, GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::GELU>,    // Gelu
        &doActivationKernel<T, GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::ReLu>,    // Relu
        &doActivationKernel<T, GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::SiLu>,    // Silu
        &doActivationKernel<T, GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::SiLu>,    // Swiglu
        &doActivationKernel<T, GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::GELU>,    // Geglu
        &doActivationKernel<T, GemmOutputType, ScaleBiasType, cutlass::epilogue::thread::Identity> // Identity
    };
    auto fn = fn_list[static_cast<int>(activation_type)];
    fn<<<blocks, threads, 0, stream>>>(output, gemm_result, fp8_quant, bias, bias_is_broadcast,
        expert_first_token_offset, num_experts, inter_size, isGatedActivation(activation_type));
}

// ============================== Lora Add Bias =================================
constexpr static int LORA_KERNELS_THREADS_PER_BLOCK = 256;

template <class ScaleBiasType, class LoraType, bool IsGated>
__global__ void loraAddBiasKernel(ScaleBiasType* output, LoraType const* lora_result, ScaleBiasType const* bias,
    int64_t const* num_valid_tokens_ptr, int* permuted_experts, int64_t inter_size)
{
    int64_t const tid = threadIdx.x;
    int64_t const token = blockIdx.x;
    int64_t const num_tokens = gridDim.x;
    if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr)
    {
        return;
    }

    LoraType const* lora_result_1 = lora_result + token * inter_size;
    int expert_id = permuted_experts[token];
    if constexpr (IsGated)
    {
        output = output + token * inter_size * 2;
        bias = bias + expert_id * inter_size * 2;
    }
    else
    {
        output = output + token * inter_size;
        bias = bias + expert_id * inter_size;
    }

    constexpr int64_t LORA_ADD_BIAS_ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<LoraType>::value;

    using DataElem = cutlass::Array<LoraType, LORA_ADD_BIAS_ELEM_PER_THREAD>;
    using BiasElem = cutlass::Array<ScaleBiasType, LORA_ADD_BIAS_ELEM_PER_THREAD>;
    auto lora_result_1_vec = reinterpret_cast<DataElem const*>(lora_result_1);
    auto bias_vec = reinterpret_cast<BiasElem const*>(bias);
    auto output_vec = reinterpret_cast<BiasElem*>(output);

    int64_t const start_offset = tid;
    int64_t const stride = LORA_KERNELS_THREADS_PER_BLOCK;
    assert(inter_size % LORA_ADD_BIAS_ELEM_PER_THREAD == 0);
    int64_t const num_elems_in_col = inter_size / LORA_ADD_BIAS_ELEM_PER_THREAD;

    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        auto lora_value = lora_result_1_vec[elem_index];
        auto bias_value = bias_vec[elem_index];
        output_vec[elem_index] = bias_value + arrayConvert<DataElem, BiasElem>(lora_value);
    }

    if constexpr (IsGated)
    {
        auto lora_result_2_vec = reinterpret_cast<DataElem const*>(lora_result_1 + num_tokens * inter_size);
        int64_t const inter_size_vec = inter_size / LORA_ADD_BIAS_ELEM_PER_THREAD;
        for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
        {
            auto lora_value = lora_result_2_vec[elem_index];
            auto bias_value = bias_vec[elem_index + inter_size_vec];
            output_vec[elem_index + inter_size_vec] = bias_value + arrayConvert<DataElem, BiasElem>(lora_value);
        }
    }
}

template <class ScaleBiasType, class LoraType>
void loraAddBias(ScaleBiasType* output, LoraType const* lora_result, ScaleBiasType const* bias,
    int64_t const* num_valid_tokens_ptr, int64_t inter_size, int* permuted_experts, int64_t num_tokens,
    bool is_gated_activation, cudaStream_t stream)
{
    int64_t const blocks = num_tokens;
    int64_t const threads = LORA_KERNELS_THREADS_PER_BLOCK;

    auto selected_fn = is_gated_activation ? loraAddBiasKernel<ScaleBiasType, LoraType, true>
                                           : loraAddBiasKernel<ScaleBiasType, LoraType, false>;
    selected_fn<<<blocks, threads, 0, stream>>>(
        output, lora_result, bias, num_valid_tokens_ptr, permuted_experts, inter_size);
}

template <class T>
__global__ void loraReorderKernel(
    T* output, T const* lora_result, int64_t const* num_valid_tokens_ptr, int64_t inter_size)
{
    int64_t const tid = threadIdx.x;
    int64_t const token = blockIdx.x;
    int64_t const num_tokens = gridDim.x;
    if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr)
    {
        return;
    }

    T const* lora_result_1 = lora_result + token * inter_size;
    output = output + token * inter_size * 2;

    constexpr int64_t LORA_REORDER_ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<T>::value;

    using DataElem = cutlass::Array<T, LORA_REORDER_ELEM_PER_THREAD>;
    auto lora_result_1_vec = reinterpret_cast<DataElem const*>(lora_result_1);
    auto output_vec = reinterpret_cast<DataElem*>(output);

    int64_t const start_offset = tid;
    int64_t const stride = LORA_KERNELS_THREADS_PER_BLOCK;
    assert(inter_size % LORA_REORDER_ELEM_PER_THREAD == 0);
    int64_t const num_elems_in_col = inter_size / LORA_REORDER_ELEM_PER_THREAD;

    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        auto lora_value = lora_result_1_vec[elem_index];
        output_vec[elem_index] = lora_value;
    }

    auto lora_result_2_vec = reinterpret_cast<DataElem const*>(lora_result_1 + num_tokens * inter_size);
    int64_t const inter_size_vec = inter_size / LORA_REORDER_ELEM_PER_THREAD;
    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        auto lora_value = lora_result_2_vec[elem_index];
        output_vec[elem_index + inter_size_vec] = lora_value;
    }
}

template <class T>
void loraReorder(T* output, T const* lora_result, int64_t const* num_valid_tokens_ptr, int64_t inter_size,
    int64_t num_tokens, cudaStream_t stream)
{
    int64_t const blocks = num_tokens;
    int64_t const threads = LORA_KERNELS_THREADS_PER_BLOCK;

    loraReorderKernel<T><<<blocks, threads, 0, stream>>>(output, lora_result, num_valid_tokens_ptr, inter_size);
}

// ============================== DEQUANT_FP8 =================================
constexpr static int DEQUANT_KERNELS_THREADS_PER_BLOCK = 256;

template <class OutputType, class InputType>
__global__ void dequantFP8Kernel(OutputType* output, InputType const* input, int64_t const* num_valid_tokens_ptr,
    int64_t inter_size, float const* scale, bool scale_is_dequant)
{
    int64_t const tid = threadIdx.x;
    int64_t const token = blockIdx.x;
    if (num_valid_tokens_ptr && token >= *num_valid_tokens_ptr)
    {
        return;
    }

    output = output + token * inter_size;
    input = input + token * inter_size;

    constexpr int64_t DEQUANT_ELEM_PER_THREAD = 128 / cutlass::sizeof_bits<InputType>::value;

    using DataElem = cutlass::Array<InputType, DEQUANT_ELEM_PER_THREAD>;
    using OutputElem = cutlass::Array<OutputType, DEQUANT_ELEM_PER_THREAD>;
    using ComputeElem = cutlass::Array<float, DEQUANT_ELEM_PER_THREAD>;
    auto input_vec = reinterpret_cast<DataElem const*>(input);
    auto output_vec = reinterpret_cast<OutputElem*>(output);

    int64_t const start_offset = tid;
    int64_t const stride = DEQUANT_KERNELS_THREADS_PER_BLOCK;
    assert(inter_size % DEQUANT_ELEM_PER_THREAD == 0);
    int64_t const num_elems_in_col = inter_size / DEQUANT_ELEM_PER_THREAD;

    ComputeElem deqaunt_scale_value;
    float dequant_scale = scale[0];
    if (!scale_is_dequant)
    {
        dequant_scale = 1.f / dequant_scale;
    }
    deqaunt_scale_value.fill(dequant_scale);

    for (int64_t elem_index = start_offset; elem_index < num_elems_in_col; elem_index += stride)
    {
        auto input_value = arrayConvert<DataElem, ComputeElem>(input_vec[elem_index]);
        output_vec[elem_index] = arrayConvert<ComputeElem, OutputElem>(input_value * deqaunt_scale_value);
    }
}

template <class OutputType, class InputType>
void dequantFP8(OutputType* output, InputType const* input, int64_t const* num_valid_tokens_ptr, int64_t inter_size,
    int64_t num_tokens, float const* scale, bool scale_is_dequant, cudaStream_t stream)
{
    int64_t const blocks = num_tokens;
    int64_t const threads = DEQUANT_KERNELS_THREADS_PER_BLOCK;

    dequantFP8Kernel<OutputType, InputType>
        <<<blocks, threads, 0, stream>>>(output, input, num_valid_tokens_ptr, inter_size, scale, scale_is_dequant);
}

template <class T, class WeightType, class OutputType, class ScaleBiasType, class Enable>
std::vector<size_t> CutlassMoeFCRunner<T, WeightType, OutputType, ScaleBiasType, Enable>::getWorkspaceDeviceBufferSizes(
    int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size, int const num_experts,
    int const num_experts_per_node, int const k, ActivationType activation_type,
    MOEExpertScaleNormalizationMode norm_mode, bool use_lora) const
{
    size_t const num_moe_inputs = k * num_rows;
    size_t const permuted_elems = num_moe_inputs * hidden_size;
    size_t const interbuf_elems = num_moe_inputs * inter_size;
    size_t glu_inter_elems = 0;
    bool is_gated_activation = isGatedActivation(activation_type);
    if (is_gated_activation)
    {
        glu_inter_elems = interbuf_elems * 2;
    }
    else if (mayHaveDifferentGEMMOutputType())
    {
        // In this case we are using activation quantization, and some intermediate buffers will be unquantized
        // We need to have separate memory for these as we can no longer alias the output buffer for reuse
        glu_inter_elems = interbuf_elems;
    }

    bool using_hopper = moe_gemm_runner_.supportsHopperSpecialisation();

    size_t const gemm_output_dtype = sizeof(UnfusedGemmOutputType);

    size_t sparse_mixer_outs = 0;
    if (norm_mode == MOEExpertScaleNormalizationMode::SPARSE_MIXER)
    {
        sparse_mixer_outs = num_rows * num_experts;
    }

    size_t num_softmax_outs = 0;
    bool const is_pow_2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    if (!is_pow_2 || num_experts > 256)
    {
        num_softmax_outs = num_rows * num_experts;
    }

    size_t const source_rows_size = num_moe_inputs * sizeof(int);
    size_t const permuted_rows_size = num_moe_inputs * sizeof(int);
    size_t const permuted_experts_size = num_moe_inputs * sizeof(int);
    size_t const permuted_data_size = permuted_elems * sizeof(T);
    size_t const expert_first_token_offset_size = (num_experts_per_node + 1) * sizeof(int64_t);
    size_t const sparse_mixer_out_size = sparse_mixer_outs * sizeof(float);
    size_t const softmax_out_size = num_softmax_outs * sizeof(float);
    size_t const permuted_scales_size = mayHaveFinalizeFused() ? num_moe_inputs * sizeof(float) : 0;
    size_t const glu_inter_size = glu_inter_elems * gemm_output_dtype; // May be an intermediate type for quantization
    size_t const fc1_result_size = interbuf_elems * sizeof(T);         // Acitvation quantizes so back to sizeof(T)
    size_t const sorter_size = CubKeyValueSorter::getWorkspaceSize(num_rows, num_experts);
    size_t const fc2_result_size = permuted_elems * gemm_output_dtype; // May be an intermediate type for quantization

    size_t const hopper_size = using_hopper ? HopperGroupedGemmInput::workspaceSize(num_experts_per_node) : 0;

    size_t const gemm_workspace_size = moe_gemm_runner_.getMaxWorkspaceSize(num_experts_per_node);

    // lora related
    size_t const lora_input_size
        = (use_lora && use_fp8) ? std::max(permuted_elems, interbuf_elems) * sizeof(ScaleBiasType) : 0;
    size_t const lora_fc1_result_size = use_lora
        ? (is_gated_activation ? 2 * interbuf_elems * sizeof(ScaleBiasType) : interbuf_elems * sizeof(ScaleBiasType))
        : 0;
    size_t const lora_add_bias_size = use_lora ? lora_fc1_result_size : 0;
    size_t const lora_fc2_result_size = use_lora ? permuted_elems * sizeof(ScaleBiasType) : 0;

    // We do some overlapping of the large workspace buffers. Although we could overlap some of the other buffers, they
    // are small enough (i.e no factor of hidden size) they will only be a couple MiB at most, so we don't bother
    // in the case of fused activation we overlap permuted_data and fc2_result
    // in the case of unfused activation we overlap permuted_data and fc1_result
    // we need to calculate the max possible size, so use the max of all three
    size_t overlapped_gemm1_gemm2_inputs = std::max(permuted_data_size, fc2_result_size);
    // When glu_inter_elems is 0 we are always fused, otherwise we may need the un-fused case
    if (glu_inter_elems > 0)
    {
        overlapped_gemm1_gemm2_inputs = std::max(overlapped_gemm1_gemm2_inputs, fc1_result_size);
    }

    size_t const alpha_scale_ptr_array_size = num_experts_per_node * sizeof(float**);

    // if we have glu_inter we overlap it with fc2_result, otherwise we use fc1_result by itself
    size_t overlapped_gemm1_gemm2_outputs = fc1_result_size;
    if (glu_inter_elems > 0)
    {
        overlapped_gemm1_gemm2_outputs
            = std::max(std::max(glu_inter_size, fc2_result_size), overlapped_gemm1_gemm2_outputs);
    }

    std::vector<size_t> workspace{source_rows_size, //
        permuted_rows_size,                         //
        permuted_experts_size,                      //
        expert_first_token_offset_size,             //
        sparse_mixer_out_size,                      //
        softmax_out_size,                           //
        permuted_scales_size,                       //
        sorter_size,                                //
        // These pointers reuse the same memory
        overlapped_gemm1_gemm2_inputs,  //
        overlapped_gemm1_gemm2_outputs, //
        alpha_scale_ptr_array_size,     //
        hopper_size,                    //
        gemm_workspace_size,            //
        lora_input_size,                //
        lora_fc1_result_size,           //
        lora_add_bias_size,             //
        lora_fc2_result_size};
    return workspace;
}

template <class T, class WeightType, class OutputType, class ScaleBiasType, class Enable>
size_t CutlassMoeFCRunner<T, WeightType, OutputType, ScaleBiasType, Enable>::getWorkspaceSize(int64_t const num_rows,
    int64_t const hidden_size, int64_t const inter_size, int const num_experts, int const k,
    ActivationType activation_type, MOEExpertScaleNormalizationMode norm_mode, MOEParallelismConfig parallelism_config,
    bool use_lora) const
{
    int const ep_size = parallelism_config.ep_size;
    TLLM_CHECK_WITH_INFO(num_experts % ep_size == 0, "Number of experts must be a multiple of ep size");
    auto workspace = getWorkspaceDeviceBufferSizes(
        num_rows, hidden_size, inter_size, num_experts, num_experts / ep_size, k, activation_type, norm_mode, use_lora);
    auto ws_size = tensorrt_llm::common::calculateTotalWorkspaceSize(workspace.data(), workspace.size());
    TLLM_LOG_DEBUG("Mixture Of Experts Plugin requires workspace of %2f MiB", ws_size / 1024.f / 1024.f);
    return ws_size;
}

template <class T, class WeightType, class OutputType, class ScaleBiasType, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, ScaleBiasType, Enable>::configureWsPtrs(char* ws_ptr,
    int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size, int const num_experts,
    int const num_experts_per_node, int const k, ActivationType activation_type,
    MOEExpertScaleNormalizationMode norm_mode, bool use_lora)

{
    auto ws_sizes = getWorkspaceDeviceBufferSizes(
        num_rows, hidden_size, inter_size, num_experts, num_experts_per_node, k, activation_type, norm_mode, use_lora);

    std::vector<int8_t*> ws_sliced{(int8_t*) ws_ptr};
    for (auto size : ws_sizes)
    {
        ws_sliced.push_back(nextWorkspacePtr(ws_sliced.back(), size));
    }
    ws_sliced.pop_back();

    source_rows_ = (int*) ws_sliced[0];
    permuted_rows_ = (int*) ws_sliced[1];
    permuted_experts_ = (int*) ws_sliced[2];

    expert_first_token_offset_ = (int64_t*) ws_sliced[3];

    sparse_mixer_out_ = nullptr;
    if (norm_mode == MOEExpertScaleNormalizationMode::SPARSE_MIXER)
    {
        sparse_mixer_out_ = (float*) ws_sliced[4];
    }

    softmax_out_ = nullptr;
    bool const is_pow_2 = (num_experts != 0) && ((num_experts & (num_experts - 1)) == 0);
    if (!is_pow_2 || num_experts > 256)
    {
        softmax_out_ = (float*) ws_sliced[5];
    }

    bool const gemm2_using_hopper = moe_gemm_runner_.isHopperSpecialised(*gemm2_config_);
    permuted_scales_ = (gemm2_using_hopper && mayHaveFinalizeFused()) ? (float*) ws_sliced[6] : nullptr;

    sorter_ws_ = (char*) ws_sliced[7];

    // Always same index, but overlapped with either fc1_result_ or fc2_result_
    permuted_data_ = (T*) ws_sliced[8];

    bool const is_gated_activation = isGatedActivation(activation_type);
    bool const gemm1_using_fused_moe
        = moe_gemm_runner_.isFusedGatedActivation(*gemm1_config_, is_gated_activation, inter_size, hidden_size);
    bool const gemm1_using_hopper = moe_gemm_runner_.isHopperSpecialised(*gemm1_config_);
    bool const hopper_has_glu = gemm1_using_hopper && (mayHaveDifferentGEMMOutputType() || is_gated_activation);
    // We always use fused path if we can
    bool const non_hopper_has_glu = !gemm1_using_fused_moe && is_gated_activation;
    bool const has_glu_inter_result = hopper_has_glu || non_hopper_has_glu || use_fp8;
    // Always same index, ignored if not needed
    glu_inter_result_ = has_glu_inter_result ? (T*) ws_sliced[9] : nullptr;

    // fc1 and fc2 alias one of the above pointers, but it depends on if actfn is fused/unfused which is overlapped
    // NOTE: It is important to get the order of these correct as the wrong order will cause the buffer to be used as an
    // input and output for the same gemm, which will cause corruption
    fc1_result_ = has_glu_inter_result ? (T*) ws_sliced[8] : (T*) ws_sliced[9];
    fc2_result_ = has_glu_inter_result ? (T*) ws_sliced[9] : (T*) ws_sliced[8];

    alpha_scale_ptr_array_ = reinterpret_cast<float const**>(ws_sliced[10]);

    hopper_grouped_gemm_input_ = {};
    if (moe_gemm_runner_.supportsHopperSpecialisation())
    {
        hopper_grouped_gemm_input_.configureWorkspace(ws_sliced[11], num_experts_per_node, ws_sliced[12], ws_sizes[12]);
    }

    lora_fc1_result_ = {};
    lora_add_bias_ = {};
    lora_fc2_result_ = {};

    if (use_lora)
    {
        lora_input_ = (ScaleBiasType*) ws_sliced[13];
        lora_fc1_result_ = (ScaleBiasType*) ws_sliced[14];
        lora_add_bias_ = (ScaleBiasType*) ws_sliced[15];
        lora_fc2_result_ = (ScaleBiasType*) ws_sliced[16];
    }
}

void sortAndScanSoftmaxOutput(int* expert_for_source_row, int* source_rows, int* permuted_experts, int* permuted_rows,
    int64_t* expert_first_token_offset, int64_t num_rows, int64_t num_experts, int64_t num_experts_per_node, int64_t k,
    CubKeyValueSorter& sorter, void* sorter_ws, cudaStream_t stream)
{
    int64_t const expanded_num_rows = k * num_rows;
    // We need to use the full num_experts because that is the sentinel value used by topk for disabled experts
    sorter.updateNumExperts(num_experts);
    size_t const sorter_ws_size_bytes = pad_to_multiple_of_16(sorter.getWorkspaceSize(expanded_num_rows, num_experts));
    sorter.run((void*) sorter_ws, sorter_ws_size_bytes, expert_for_source_row, permuted_experts, source_rows,
        permuted_rows, expanded_num_rows, stream);

    sync_check_cuda_error();

    // Upper bound on number of expanded rows
    computeExpertFirstTokenOffset(
        permuted_experts, expanded_num_rows, num_experts_per_node, expert_first_token_offset, stream);
}

template <class T, class WeightType, class OutputType, class ScaleBiasType, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, ScaleBiasType, Enable>::gemm1(
    MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>& gemm_runner, T const* const input, T* const output,
    void* const intermediate_result, int64_t const* const expert_first_token_offset,
    HopperGroupedGemmInput const hopper_input_template, WeightType const* const fc1_expert_weights,
    ScaleBiasType const* const fc1_expert_biases, int64_t const* const num_valid_tokens_ptr,
    ScaleBiasType const* const fc1_int_scales, float const* const fc1_fp8_dequant, float const* const fc2_fp8_quant,
    int64_t const expanded_num_rows, int64_t const hidden_size, int64_t const inter_size,
    int const num_experts_per_node, ActivationType fc1_activation_type, float const** alpha_scale_ptr_array,
    bool bias_is_broadcast, cudaStream_t stream, cutlass_extensions::CutlassGemmConfig config)
{
    bool const using_hopper_gemm1 = gemm_runner.isHopperSpecialised(config);
    bool const is_gated_activation = isGatedActivation(fc1_activation_type);
    bool const use_ampere_activation_fusion
        = gemm_runner.isFusedGatedActivation(config, is_gated_activation, inter_size, hidden_size);
    size_t const fc1_out_size = ((!use_ampere_activation_fusion) && is_gated_activation) ? inter_size * 2 : inter_size;

    int64_t const* total_tokens_including_expert = expert_first_token_offset + 1;

    if (using_hopper_gemm1)
    {
        TLLM_CHECK(config.is_sm90);
        TLLM_CHECK(!use_ampere_activation_fusion);
        bool has_different_gemm_output_type = using_hopper_gemm1 && !std::is_same_v<T, OutputType>;
        bool const has_intermediate = has_different_gemm_output_type || is_gated_activation;
        TLLM_CHECK_WITH_INFO(has_intermediate || input != output, "Input and output buffers are overlapping");
        auto* gemm_output = has_intermediate ? intermediate_result : static_cast<void*>(output);

        auto hopper_input = hopper_input_template;
        hopper_input.fusion = HopperGroupedGemmInput::EpilogueFusion::NONE;
        hopper_input = computeStridesHopper(expert_first_token_offset, hopper_input, fc1_out_size, hidden_size,
            num_experts_per_node, input, fc1_expert_weights, fc1_fp8_dequant, nullptr,
            static_cast<UnfusedGemmOutputType*>(gemm_output), stream);
        sync_check_cuda_error();

        gemm_runner.moeGemm(input, nullptr, nullptr, nullptr, total_tokens_including_expert, hopper_input,
            expanded_num_rows, fc1_out_size, hidden_size, num_experts_per_node, false, alpha_scale_ptr_array, stream,
            config);

        sync_check_cuda_error();
        // TODO: when bias_is_broadcast is false, fuse bias to gemm
        doActivation<T, UnfusedGemmOutputType>(output, static_cast<UnfusedGemmOutputType const*>(gemm_output),
            fc2_fp8_quant, fc1_expert_biases, bias_is_broadcast, expert_first_token_offset, num_experts_per_node,
            inter_size, expanded_num_rows, fc1_activation_type, stream);

        sync_check_cuda_error();
    }
    else if (use_fp8)
    {
        TLLM_CHECK(!use_ampere_activation_fusion);
        TLLM_CHECK(!config.is_sm90);

        alpha_scale_ptr_array
            = computeFP8DequantScale(alpha_scale_ptr_array, num_experts_per_node, fc1_fp8_dequant, stream);

        gemm_runner.moeGemm(input, fc1_expert_weights, nullptr,
            reinterpret_cast<UnfusedGemmOutputType*>(intermediate_result), total_tokens_including_expert,
            HopperGroupedGemmInput{}, expanded_num_rows, fc1_out_size, hidden_size, num_experts_per_node, false,
            alpha_scale_ptr_array, stream, config);

        doActivation<T, UnfusedGemmOutputType>(output, static_cast<UnfusedGemmOutputType const*>(intermediate_result),
            fc2_fp8_quant, fc1_expert_biases, bias_is_broadcast, expert_first_token_offset, num_experts_per_node,
            inter_size, expanded_num_rows, fc1_activation_type, stream);

        sync_check_cuda_error();
    }
    else if (!is_gated_activation)
    {
        TLLM_CHECK(!use_ampere_activation_fusion);
        TLLM_CHECK(!config.is_sm90);
        gemm_runner.moeGemmBiasAct(input, fc1_expert_weights, fc1_int_scales, fc1_expert_biases, bias_is_broadcast,
            output, total_tokens_including_expert, HopperGroupedGemmInput{}, expanded_num_rows, fc1_out_size,
            hidden_size, num_experts_per_node, fc1_activation_type, false, alpha_scale_ptr_array, stream, config);

        sync_check_cuda_error();
    }
    else
    {
        TLLM_CHECK(!config.is_sm90);
        TLLM_CHECK(is_gated_activation);
        TLLM_CHECK_WITH_INFO(
            !use_ampere_activation_fusion || input != output, "Input and output buffers are overlapping");

        // Run the GEMM with activation function overridden with `Identity`, we do the activation separately
        ActivationType activation_type = use_ampere_activation_fusion ? fc1_activation_type : ActivationType::Identity;
        void* gemm_result = use_ampere_activation_fusion ? static_cast<void*>(output) : intermediate_result;
        gemm_runner.moeGemmBiasAct(input, fc1_expert_weights, fc1_int_scales, fc1_expert_biases, bias_is_broadcast,
            gemm_result, total_tokens_including_expert, HopperGroupedGemmInput{}, expanded_num_rows, fc1_out_size,
            hidden_size, num_experts_per_node, activation_type, use_ampere_activation_fusion, alpha_scale_ptr_array,
            stream, config);

        sync_check_cuda_error();

        if (!use_ampere_activation_fusion)
        {
            doGatedActivation<T, UnfusedGemmOutputType>(output,
                static_cast<UnfusedGemmOutputType const*>(intermediate_result), num_valid_tokens_ptr, inter_size,
                expanded_num_rows, fc1_activation_type, stream);

            sync_check_cuda_error();
        }
    }
}

template <class T, class WeightType, class OutputType, class ScaleBiasType, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, ScaleBiasType, Enable>::gemm2(
    MoeGemmRunner<T, WeightType, OutputType, ScaleBiasType>& gemm_runner, T const* const input, void* const gemm_output,
    OutputType* const final_output, int64_t const* const expert_first_token_offset,
    HopperGroupedGemmInput const hopper_input_template, WeightType const* const fc2_expert_weights,
    ScaleBiasType const* const fc2_expert_biases, ScaleBiasType const* const fc2_int_scales,
    float const* const fc2_fp8_dequant, float const* const token_topk_unpermuted_scales,
    float const* const token_topk_permuted_scales, int const* const expanded_source_row_to_expanded_dest_row,
    int const* expanded_dest_row_to_expanded_source_row, int const* const expert_for_source_row,
    int64_t const* const num_valid_tokens_ptr, int64_t const num_rows, int64_t const expanded_num_rows,
    int64_t const hidden_size, int64_t const inter_size, int const num_experts_per_node, int64_t const k,
    bool using_hopper_fused_finalize, float const** alpha_scale_ptr_array, bool use_lora, void* fc2_lora,
    cudaStream_t stream, MOEParallelismConfig parallelism_config, cutlass_extensions::CutlassGemmConfig config)
{
    int64_t const* total_tokens_including_expert = expert_first_token_offset + 1;

    bool const using_hopper_gemm2 = gemm_runner.isHopperSpecialised(config);
    HopperGroupedGemmInput hopper_input{};
    if (using_hopper_gemm2)
    {
        bool apply_bias = parallelism_config.tp_rank == 0;
        hopper_input = hopper_input_template;
        hopper_input.fusion = HopperGroupedGemmInput::EpilogueFusion::NONE;
        if (using_hopper_fused_finalize)
        {
            hopper_input.fusion = HopperGroupedGemmInput::EpilogueFusion::FINALIZE;
            hopper_input.setFinalizeFusionParams(final_output, token_topk_permuted_scales, expert_first_token_offset,
                expanded_dest_row_to_expanded_source_row, apply_bias ? fc2_expert_biases : nullptr, hidden_size,
                num_rows);
            check_cuda_error(cudaMemsetAsync(final_output, 0x0, sizeof(OutputType) * num_rows * hidden_size, stream));
        }

        hopper_input = computeStridesHopper(expert_first_token_offset, hopper_input, hidden_size, inter_size,
            num_experts_per_node, input, fc2_expert_weights, fc2_fp8_dequant, nullptr,
            static_cast<UnfusedGemmOutputType*>(gemm_output), stream);

        sync_check_cuda_error();
    }
    else if (use_fp8)
    {
        alpha_scale_ptr_array
            = computeFP8DequantScale(alpha_scale_ptr_array, num_experts_per_node, fc2_fp8_dequant, stream);
    }

    bool fuse_lora_bias = use_lora && !(use_fp8 || using_hopper_gemm2);

    gemm_runner.moeGemmBiasAct(input, fc2_expert_weights, fc2_int_scales,
        fuse_lora_bias ? static_cast<ScaleBiasType const*>(fc2_lora) : nullptr, false, gemm_output,
        total_tokens_including_expert, hopper_input, expanded_num_rows, hidden_size, inter_size, num_experts_per_node,
        ActivationType::Identity, false, alpha_scale_ptr_array, stream, config);
    sync_check_cuda_error();

    if (use_lora && !fuse_lora_bias)
    {
        auto loraBiasApplyFunc = doActivation<UnfusedGemmOutputType, UnfusedGemmOutputType, ScaleBiasType>;
        loraBiasApplyFunc(static_cast<UnfusedGemmOutputType*>(gemm_output),
            static_cast<UnfusedGemmOutputType const*>(gemm_output), nullptr,
            static_cast<ScaleBiasType const*>(fc2_lora), false, expert_first_token_offset, num_experts_per_node,
            hidden_size, expanded_num_rows, ActivationType::Identity, stream);
        sync_check_cuda_error();
    }

    bool has_different_output_type_ampere = use_fp8 && !using_hopper_gemm2;
    bool has_different_output_type_hopper = !using_hopper_fused_finalize && using_hopper_gemm2;

    if (has_different_output_type_ampere || has_different_output_type_hopper)
    {
        finalizeMoeRoutingKernelLauncher<T, OutputType, UnfusedGemmOutputType>(
            static_cast<UnfusedGemmOutputType const*>(gemm_output), final_output, fc2_expert_biases,
            token_topk_unpermuted_scales, expanded_source_row_to_expanded_dest_row, expert_for_source_row, num_rows,
            hidden_size, k, num_valid_tokens_ptr, parallelism_config, MOEExpertScaleNormalizationMode::NONE, stream);
    }
    else if (!using_hopper_gemm2)
    {
        finalizeMoeRoutingKernelLauncher<T, OutputType, T>(static_cast<T const*>(gemm_output), final_output,
            fc2_expert_biases, token_topk_unpermuted_scales, expanded_source_row_to_expanded_dest_row,
            expert_for_source_row, num_rows, hidden_size, k, num_valid_tokens_ptr, parallelism_config,
            MOEExpertScaleNormalizationMode::NONE, stream);
    }

    sync_check_cuda_error();
}

template <class T, class WeightType, class OutputType, class ScaleBiasType, class Enable>
bool CutlassMoeFCRunner<T, WeightType, OutputType, ScaleBiasType, Enable>::setupLoraWorkspace(int64_t expanded_num_rows,
    int64_t num_rows, int64_t inter_size, int64_t hidden_size, int start_expert, bool is_gated_activation,
    int num_experts_per_node, bool needs_num_valid, LoraParams& lora_params, cudaStream_t stream)
{
    std::vector<int>& host_permuted_rows = host_lora_workspace_.host_permuted_rows;
    std::vector<void const*>& host_permuted_fc1_weight_ptrs = host_lora_workspace_.host_permuted_fc1_weight_ptrs;
    std::vector<void const*>& host_permuted_fc2_weight_ptrs = host_lora_workspace_.host_permuted_fc2_weight_ptrs;
    std::vector<void const*>& host_permuted_gated_weight_ptrs = host_lora_workspace_.host_permuted_gated_weight_ptrs;

    std::vector<int32_t>& host_permuted_fc1_lora_ranks = host_lora_workspace_.host_permuted_fc1_lora_ranks;
    std::vector<int32_t>& host_permuted_fc2_lora_ranks = host_lora_workspace_.host_permuted_fc2_lora_ranks;
    std::vector<int32_t>& host_permuted_gated_lora_ranks = host_lora_workspace_.host_permuted_gated_lora_ranks;
    std::vector<int64_t>& host_expert_first_token_offset = host_lora_workspace_.host_expert_first_token_offset;

    bool all_token_without_lora = true;

    host_permuted_fc1_weight_ptrs.resize(expanded_num_rows * 2);
    host_permuted_fc1_lora_ranks.resize(expanded_num_rows);
    host_permuted_fc2_weight_ptrs.resize(expanded_num_rows * 2);
    host_permuted_fc2_lora_ranks.resize(expanded_num_rows);

    if (is_gated_activation)
    {
        host_permuted_gated_weight_ptrs.resize(expanded_num_rows * 2);
        host_permuted_gated_lora_ranks.resize(expanded_num_rows);
    }

    TLLM_CUDA_CHECK(cudaEventSynchronize(*(lora_params.memcpy_event_ptr)));

    size_t num_valid_tokens
        = needs_num_valid ? host_expert_first_token_offset[num_experts_per_node] : expanded_num_rows;

    for (int expert_idx = 0; expert_idx < num_experts_per_node; ++expert_idx)
    {
        int weight_index = expert_idx + start_expert;
        for (size_t i = host_expert_first_token_offset[expert_idx]; i < host_expert_first_token_offset[expert_idx + 1];
             ++i)
        {
            int source_index = host_permuted_rows[i] % num_rows;
            int32_t lora_rank = lora_params.fc1_lora_ranks[source_index];
            host_permuted_fc1_weight_ptrs[i * 2]
                = reinterpret_cast<ScaleBiasType const*>(lora_params.fc1_lora_weight_ptrs[source_index * 2])
                + weight_index * hidden_size * lora_rank;
            host_permuted_fc1_weight_ptrs[i * 2 + 1]
                = reinterpret_cast<ScaleBiasType const*>(lora_params.fc1_lora_weight_ptrs[source_index * 2 + 1])
                + weight_index * lora_rank * inter_size;
            host_permuted_fc1_lora_ranks[i] = lora_rank;

            lora_rank = lora_params.fc2_lora_ranks[source_index];
            host_permuted_fc2_weight_ptrs[i * 2]
                = reinterpret_cast<ScaleBiasType const*>(lora_params.fc2_lora_weight_ptrs[source_index * 2])
                + weight_index * inter_size * lora_rank;
            host_permuted_fc2_weight_ptrs[i * 2 + 1]
                = reinterpret_cast<ScaleBiasType const*>(lora_params.fc2_lora_weight_ptrs[source_index * 2 + 1])
                + weight_index * lora_rank * hidden_size;
            host_permuted_fc2_lora_ranks[i] = lora_rank;

            if (host_permuted_fc1_lora_ranks[i] || host_permuted_fc2_lora_ranks[i])
            {
                all_token_without_lora = false;
            }

            if (is_gated_activation)
            {
                lora_rank = lora_params.gated_lora_ranks[source_index];
                host_permuted_gated_weight_ptrs[i * 2]
                    = reinterpret_cast<ScaleBiasType const*>(lora_params.gated_lora_weight_ptrs[source_index * 2])
                    + weight_index * hidden_size * lora_rank;
                host_permuted_gated_weight_ptrs[i * 2 + 1]
                    = reinterpret_cast<ScaleBiasType const*>(lora_params.gated_lora_weight_ptrs[source_index * 2 + 1])
                    + weight_index * lora_rank * inter_size;
                host_permuted_gated_lora_ranks[i] = lora_rank;

                if (host_permuted_gated_lora_ranks[i])
                {
                    all_token_without_lora = false;
                }
            }
        }
    }
    return all_token_without_lora;
}

template <class T, class WeightType, class OutputType, class ScaleBiasType, class Enable>
ScaleBiasType const* CutlassMoeFCRunner<T, WeightType, OutputType, ScaleBiasType, Enable>::loraFC1(
    int64_t expanded_num_rows, int64_t inter_size, int64_t hidden_size, int num_experts_per_node, int start_expert,
    int64_t const* num_valid_tokens_ptr, bool is_gated_activation, ScaleBiasType const* fc1_expert_biases,
    LoraParams& lora_params, float const* input_fp8_dequant, cudaStream_t stream)
{
    std::vector<void const*>& host_permuted_fc1_weight_ptrs = host_lora_workspace_.host_permuted_fc1_weight_ptrs;
    std::vector<void const*>& host_permuted_gated_weight_ptrs = host_lora_workspace_.host_permuted_gated_weight_ptrs;

    std::vector<int32_t>& host_permuted_fc1_lora_ranks = host_lora_workspace_.host_permuted_fc1_lora_ranks;
    std::vector<int32_t>& host_permuted_gated_lora_ranks = host_lora_workspace_.host_permuted_gated_lora_ranks;
    std::vector<int64_t>& host_expert_first_token_offset = host_lora_workspace_.host_expert_first_token_offset;

    auto fc1_lora_impl = lora_params.fc1_lora_impl;
    int num_reqs = lora_params.num_reqs;

    ScaleBiasType *lora_gated_out = nullptr, *lora_fc1_result = nullptr;

    if (is_gated_activation)
    {
        lora_gated_out = lora_fc1_result_;
        lora_fc1_result = lora_fc1_result_ + expanded_num_rows * inter_size;
    }
    else
    {
        lora_fc1_result = lora_fc1_result_;
    }

    ScaleBiasType* input;
    if constexpr (use_fp8)
    {
        bool const scale_is_dequant = true;
        dequantFP8<ScaleBiasType, T>(lora_input_, permuted_data_, num_valid_tokens_ptr, hidden_size, expanded_num_rows,
            input_fp8_dequant, scale_is_dequant, stream);
        sync_check_cuda_error();
        input = lora_input_;
    }
    else
    {
        input = permuted_data_;
    }

    void* lora_workspace = lora_params.workspace;
    void* tmp_lora_fc_result = static_cast<void*>(lora_fc1_result);
    int64_t num_valid_tokens = host_expert_first_token_offset[num_experts_per_node];
    int64_t num_reqs_lora = std::min(num_valid_tokens, static_cast<int64_t>(num_reqs * num_experts_per_node));

    fc1_lora_impl->run(num_valid_tokens, num_reqs_lora, input, host_permuted_fc1_lora_ranks.data(),
        host_permuted_fc1_weight_ptrs.data(), 0, &tmp_lora_fc_result, lora_workspace, stream);

    if (is_gated_activation)
    {
        void* tmp_lora_gated_result = static_cast<void*>(lora_gated_out);
        fc1_lora_impl->run(num_valid_tokens, num_reqs_lora, input, host_permuted_gated_lora_ranks.data(),
            host_permuted_gated_weight_ptrs.data(), 0, &tmp_lora_gated_result, lora_workspace, stream);
    }

    // add bias and reorder
    if (fc1_expert_biases != nullptr)
    {
        loraAddBias(lora_add_bias_, lora_fc1_result_, fc1_expert_biases, num_valid_tokens_ptr, inter_size,
            permuted_experts_, expanded_num_rows, is_gated_activation, stream);
        return lora_add_bias_;
    }
    else if (is_gated_activation)
    {
        loraReorder(lora_add_bias_, lora_fc1_result_, num_valid_tokens_ptr, inter_size, expanded_num_rows, stream);
        return lora_add_bias_;
    }
    else
    {
        return lora_fc1_result_;
    }
}

template <class T, class WeightType, class OutputType, class ScaleBiasType, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, ScaleBiasType, Enable>::loraFC2(int64_t inter_size,
    int64_t hidden_size, int num_experts_per_node, int start_expert, int64_t const* num_valid_tokens_ptr,
    int64_t num_tokens, LoraParams& lora_params, float const* fc2_fp8_quant, cudaStream_t stream)
{
    std::vector<void const*>& host_permuted_fc2_weight_ptrs = host_lora_workspace_.host_permuted_fc2_weight_ptrs;
    std::vector<int32_t>& host_permuted_fc2_lora_ranks = host_lora_workspace_.host_permuted_fc2_lora_ranks;
    std::vector<int64_t>& host_expert_first_token_offset = host_lora_workspace_.host_expert_first_token_offset;
    auto fc2_lora_impl = lora_params.fc2_lora_impl;
    int num_reqs = lora_params.num_reqs;

    ScaleBiasType* input;
    if constexpr (use_fp8)
    {
        bool const scale_is_dequant = false;
        dequantFP8(lora_input_, fc1_result_, num_valid_tokens_ptr, inter_size, num_tokens, fc2_fp8_quant,
            scale_is_dequant, stream);
        sync_check_cuda_error();
        input = lora_input_;
    }
    else
    {
        input = fc1_result_;
    }

    void* lora_workspace = lora_params.workspace;
    int64_t num_valid_tokens = host_expert_first_token_offset[num_experts_per_node];
    void* tmp_lora_fc_result = static_cast<void*>(lora_fc2_result_);
    int64_t num_reqs_lora = std::min(num_valid_tokens, static_cast<int64_t>(num_reqs * num_experts_per_node));

    fc2_lora_impl->run(num_valid_tokens, num_reqs_lora, input, host_permuted_fc2_lora_ranks.data(),
        host_permuted_fc2_weight_ptrs.data(), 0, &tmp_lora_fc_result, lora_workspace, stream);
    sync_check_cuda_error();
}

template <class T, class WeightType, class OutputType, class ScaleBiasType, class Enable>
void CutlassMoeFCRunner<T, WeightType, OutputType, ScaleBiasType, Enable>::runMoe(void const* input_activations_void,
    float const* gating_output, void const* fc1_expert_weights_void, void const* fc1_expert_biases_void,
    ActivationType fc1_activation_type, void const* fc2_expert_weights_void, void const* fc2_expert_biases_void,
    QuantParams quant_params, int64_t const num_rows, int64_t const hidden_size, int64_t const inter_size,
    int const num_experts, int const k, char* workspace_ptr, void* final_output_void, bool const* finished,
    int64_t const active_rows, void* token_topk_final_scales_void, int* expanded_source_row_to_expanded_dest_row,
    int* expert_for_source_row, float sparse_mixer_epsilon, MOEParallelismConfig parallelism_config,
    MOEExpertScaleNormalizationMode normalization_mode, bool use_lora, LoraParams& lora_params, cudaStream_t stream)
{
    static constexpr bool int_scales_required
        = std::is_same<WeightType, uint8_t>::value || std::is_same<WeightType, cutlass::uint4b_t>::value;
    static constexpr bool fp8_scales_required
        = std::is_same<WeightType, __nv_fp8_e4m3>::value || std::is_same<WeightType, __nv_fp8_e5m2>::value;

    auto const* input_activations = static_cast<T const*>(input_activations_void);
    auto const* fc1_expert_weights = static_cast<WeightType const*>(fc1_expert_weights_void);
    auto const* fc1_expert_biases = reinterpret_cast<ScaleBiasType const*>(fc1_expert_biases_void);
    auto const* fc2_expert_weights = static_cast<WeightType const*>(fc2_expert_weights_void);
    auto const* fc1_int_scales = reinterpret_cast<ScaleBiasType const*>(quant_params.fc1_weight_scales);
    auto const* fc2_int_scales = reinterpret_cast<ScaleBiasType const*>(quant_params.fc2_weight_scales);

    auto const* fc1_fp8_dequant = quant_params.dequant_fc1;
    auto const* fc2_fp8_quant = quant_params.quant_fc2;
    auto const* fc2_fp8_dequant = quant_params.dequant_fc2;
    auto const* input_fp8_dequant = quant_params.dequant_input;
    auto const* fc2_expert_biases = reinterpret_cast<ScaleBiasType const*>(fc2_expert_biases_void);
    auto* final_output = static_cast<OutputType*>(final_output_void);
    auto* token_topk_unpermuted_scales = static_cast<float*>(token_topk_final_scales_void);

    TLLM_CHECK_WITH_INFO(finished == nullptr, "Using 'finished' is deprecated and will be removed in future versions");
    TLLM_CHECK_WITH_INFO(
        num_rows == active_rows, "Using 'finished' is deprecated and will be removed in future versions");
    TLLM_CHECK(input_activations);
    TLLM_CHECK(gating_output);
    TLLM_CHECK(fc1_expert_weights);
    TLLM_CHECK(fc2_expert_weights);
    TLLM_CHECK(workspace_ptr);
    TLLM_CHECK(token_topk_unpermuted_scales);
    TLLM_CHECK(expanded_source_row_to_expanded_dest_row);
    TLLM_CHECK(expert_for_source_row);
    TLLM_CHECK(num_experts % parallelism_config.ep_size == 0);
    TLLM_CHECK_WITH_INFO(hidden_size >= 128 / cutlass::sizeof_bits<WeightType>::value,
        "Hidden size is too small to meet alignment requirements for MOE GEMM");
    TLLM_CHECK_WITH_INFO(hidden_size % (128 / cutlass::sizeof_bits<WeightType>::value) == 0,
        "Hidden size does not meet minimum alignment requirements for MOE GEMM");
    TLLM_CHECK_WITH_INFO(inter_size % (128 / cutlass::sizeof_bits<WeightType>::value) == 0,
        "Inter size does not meet minimum alignment requirements for MOE GEMM");

    // These values must fit into an int for building the source maps
    TLLM_CHECK_WITH_INFO(num_rows <= std::numeric_limits<int>::max(), "Number of rows is too large");
    TLLM_CHECK_WITH_INFO(
        num_rows * num_experts <= std::numeric_limits<int>::max(), "Number of rows * num_experts is too large");
    TLLM_CHECK_WITH_INFO(k * num_experts <= std::numeric_limits<int>::max(), "k * num_experts is too large");

    TLLM_CHECK_WITH_INFO(gemm1_config_, "MOE GEMM1 Config is not set");
    TLLM_CHECK_WITH_INFO(gemm2_config_, "MOE GEMM2 Config is not set");

    if (int_scales_required)
    {
        TLLM_CHECK_WITH_INFO(
            fc1_int_scales != nullptr, "Weight scales expected but scale for first matmul is a null pointer");
        TLLM_CHECK_WITH_INFO(
            fc2_int_scales != nullptr, "Weight scales expected but scale for second matmul is a null pointer");

        TLLM_CHECK_WITH_INFO(fc1_fp8_dequant == nullptr && fc2_fp8_quant == nullptr && fc2_fp8_dequant == nullptr,
            "FP8 scales are provided for integer quantization");
    }
    else if (fp8_scales_required)
    {
        TLLM_CHECK_WITH_INFO(fc1_expert_biases == nullptr, "Bias is not supported with FP8");
        TLLM_CHECK_WITH_INFO(fc2_expert_biases == nullptr, "Bias is not supported with FP8");

        TLLM_CHECK_WITH_INFO(
            fc1_fp8_dequant != nullptr, "FP8 scales expected but dequant scale for FC1 is a null pointer");
        TLLM_CHECK_WITH_INFO(fc2_fp8_quant != nullptr, "FP8 scales expected but quant scale for FC2 is a null pointer");
        TLLM_CHECK_WITH_INFO(
            fc2_fp8_dequant != nullptr, "FP8 scales expected but quant scale for FC2 is a null pointer");

        TLLM_CHECK_WITH_INFO(
            fc1_int_scales == nullptr && fc2_int_scales == nullptr, "Integer scales are provided for FP8 quantization");
    }
    else if (use_lora && use_fp8)
    {
        TLLM_CHECK_WITH_INFO(
            input_fp8_dequant != nullptr, "FP8 scales expected but quant scale for input is a null pointer");
    }
    else
    {
        TLLM_CHECK_WITH_INFO(
            fc1_int_scales == nullptr, "Scales are ignored for fp32/fp16/bf16 but received weight scale for FC1");
        TLLM_CHECK_WITH_INFO(
            fc2_int_scales == nullptr, "Scales are ignored for fp32/fp16/bf16 but received weight scale for FC2");
        TLLM_CHECK_WITH_INFO(
            fc1_fp8_dequant == nullptr, "Scales are ignored for fp32/fp16/bf16 but received dequant scale for FC1");
        TLLM_CHECK_WITH_INFO(
            fc2_fp8_quant == nullptr, "Scales are ignored for fp32/fp16/bf16 but received quant scale for FC2");
        TLLM_CHECK_WITH_INFO(
            fc2_fp8_dequant == nullptr, "Scales are ignored for fp32/fp16/bf16 but received quant scale for FC2");
    }

    int const num_experts_per_node = num_experts / parallelism_config.ep_size;

    configureWsPtrs(workspace_ptr, num_rows, hidden_size, inter_size, num_experts, num_experts_per_node, k,
        fc1_activation_type, normalization_mode, use_lora);

    int const start_expert = num_experts_per_node * parallelism_config.ep_rank;
    int const end_expert = start_expert + num_experts_per_node;

    selectExpertsForTokens(gating_output, token_topk_unpermuted_scales, sparse_mixer_out_, softmax_out_,
        expert_for_source_row, source_rows_, num_rows, num_experts, k, start_expert, end_expert, sparse_mixer_epsilon,
        normalization_mode, stream);

    sync_check_cuda_error();

    sortAndScanSoftmaxOutput(expert_for_source_row, source_rows_, permuted_experts_, permuted_rows_,
        expert_first_token_offset_, num_rows, num_experts, num_experts_per_node, k, sorter_,
        static_cast<void*>(sorter_ws_), stream);

    sync_check_cuda_error();

    int64_t const expanded_num_rows = k * num_rows;
    bool is_gated_activation = isGatedActivation(fc1_activation_type);

    if (use_lora)
    {
        std::vector<int>& host_permuted_rows = host_lora_workspace_.host_permuted_rows;
        std::vector<int64_t>& host_expert_first_token_offset = host_lora_workspace_.host_expert_first_token_offset;
        host_permuted_rows.resize(expanded_num_rows);
        TLLM_CUDA_CHECK(cudaMemcpyAsync(host_permuted_rows.data(), permuted_rows_, expanded_num_rows * sizeof(int),
            cudaMemcpyDeviceToHost, stream));
        host_expert_first_token_offset.resize(num_experts_per_node + 1);
        TLLM_CUDA_CHECK(cudaMemcpyAsync(host_expert_first_token_offset.data(), expert_first_token_offset_,
            (num_experts_per_node + 1) * sizeof(int64_t), cudaMemcpyDeviceToHost, stream));
        TLLM_CUDA_CHECK(cudaEventRecord(*(lora_params.memcpy_event_ptr), stream));
    }

    // Actually permute the data
    bool const needs_num_valid = finished || parallelism_config.ep_size > 1;
    int64_t const* num_valid_tokens_ptr = needs_num_valid ? expert_first_token_offset_ + num_experts_per_node : nullptr;
    expandInputRowsKernelLauncher(input_activations, permuted_data_, token_topk_unpermuted_scales, permuted_scales_,
        permuted_rows_, expanded_source_row_to_expanded_dest_row, num_rows, num_valid_tokens_ptr, hidden_size, k,
        stream);

    sync_check_cuda_error();

    if (use_lora)
    {
        bool all_token_without_lora = setupLoraWorkspace(expanded_num_rows, num_rows, inter_size, hidden_size,
            start_expert, is_gated_activation, num_experts_per_node, needs_num_valid, lora_params, stream);

        if (!all_token_without_lora)
        {
            fc1_expert_biases = loraFC1(expanded_num_rows, inter_size, hidden_size, num_experts_per_node, start_expert,
                num_valid_tokens_ptr, is_gated_activation, fc1_expert_biases, lora_params, input_fp8_dequant, stream);
            sync_check_cuda_error();
        }
        else
        {
            use_lora = false;
        }
    }

    Self::gemm1(moe_gemm_runner_, permuted_data_, fc1_result_, glu_inter_result_, expert_first_token_offset_,
        hopper_grouped_gemm_input_, fc1_expert_weights, fc1_expert_biases, num_valid_tokens_ptr, fc1_int_scales,
        fc1_fp8_dequant, fc2_fp8_quant, expanded_num_rows, hidden_size, inter_size, num_experts_per_node,
        fc1_activation_type, alpha_scale_ptr_array_, !use_lora, stream, *gemm1_config_);

    sync_check_cuda_error();

    if (use_lora)
    {
        loraFC2(inter_size, hidden_size, num_experts_per_node, start_expert, num_valid_tokens_ptr, expanded_num_rows,
            lora_params, fc2_fp8_quant, stream);
        sync_check_cuda_error();
    }

    Self::gemm2(moe_gemm_runner_, fc1_result_, fc2_result_, final_output, expert_first_token_offset_,
        hopper_grouped_gemm_input_, fc2_expert_weights, fc2_expert_biases, fc2_int_scales, fc2_fp8_dequant,
        token_topk_unpermuted_scales, permuted_scales_, expanded_source_row_to_expanded_dest_row, permuted_rows_,
        expert_for_source_row, num_valid_tokens_ptr, num_rows, expanded_num_rows, hidden_size, inter_size,
        num_experts_per_node, k, !use_deterministic_hopper_reduce_, alpha_scale_ptr_array_, use_lora, lora_fc2_result_,
        stream, parallelism_config, *gemm2_config_);

    sync_check_cuda_error();
}

template <class T, class WeightType, class OutputType, class ScaleBiasType, class Enable>
HopperGroupedGemmInput CutlassMoeFCRunner<T, WeightType, OutputType, ScaleBiasType, Enable>::computeStridesHopper(
    int64_t const* expert_first_token_offset, HopperGroupedGemmInput layout_info, int64_t gemm_n, int64_t gemm_k,
    int const num_experts, T const* in, WeightType const* weights, float const* fp8_dequant, T const* bias,
    UnfusedGemmOutputType* output, cudaStream_t stream)
{
    // Always nullptr
    layout_info.ptr_c = nullptr;
    layout_info.stride_c = nullptr;

    if (!fp8_dequant)
    {
        layout_info.alpha_scale_ptr_array = nullptr;
    }

    int const threads = std::min(1024, num_experts);
    int const blocks = (num_experts + threads - 1) / threads;

    computeStridesHopperKernel<<<blocks, threads, 0, stream>>>(
        expert_first_token_offset, layout_info, gemm_n, gemm_k, num_experts, in, weights, fp8_dequant, bias, output);

    return layout_info;
}

// ==================== Helper for getting load balanced routing for profiling ==================================

template <class T>
__global__ void initRoutingKernelDiagonal(void* data_void, int num_experts, int num_tokens, int k, int stride)
{
    assert(k == 1 || (stride % num_experts) != 0);
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= num_tokens)
    {
        return;
    }
    T* data = reinterpret_cast<T*>(data_void) + token * num_experts;
    int start = token % num_experts;
    for (int i = 0; i < k; i++)
    {
        data[start] = T{1.f};
        start += stride;
        if (start >= num_experts) // Wrap
            start -= num_experts;
    }
}

void makeLoadBalancedRoutingConfiguration(
    void* data_void, int num_experts, int num_tokens, int k, nvinfer1::DataType type, cudaStream_t stream)
{
    TLLM_CHECK_WITH_INFO(type == nvinfer1::DataType::kFLOAT, "Routing configuration must be float");
    check_cuda_error(
        cudaMemsetAsync(data_void, 0x0, int64_t{num_experts} * int64_t{num_tokens} * sizeof(float), stream));

    int stride = tensorrt_llm::common::ceilDiv(num_experts, k);

    int blockDim = 256;
    int gridDim = tensorrt_llm::common::ceilDiv(num_tokens, blockDim);
    initRoutingKernelDiagonal<float><<<gridDim, blockDim, 0, stream>>>(data_void, num_experts, num_tokens, k, stride);

    sync_check_cuda_error();
}

__global__ void prepareFakeRouterBuffers(int* unpermuted_source_rows, int* unpermuted_expert_selection,
    int64_t num_tokens, int64_t k, int64_t num_experts, int64_t num_experts_per_node)
{
    int64_t tid = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    int64_t sample = blockIdx.y;
    if (tid >= num_tokens)
    {
        return;
    }

    // Offset the buffers to the start of the sample
    unpermuted_source_rows += sample * num_tokens * k;
    unpermuted_expert_selection += sample * num_tokens * k;

    // This is not perf sensitive we just init the state here every time prepare is called
    // This means the first N tokens will always have the same distribution, regardless of num_tokens
    curandStatePhilox4_32_10_t state;
    curand_init(sample, tid, 0, &state);
    for (int k_idx = 0; k_idx < k; k_idx++)
    {
        while (true)
        {
            // curand_uniform includes 1 but not 0, so round up and subtract 1
            int expert = std::ceil(static_cast<float>(num_experts) * curand_uniform(&state)) - 1;

            bool valid = true;
            for (int prev_k = 0; prev_k < k_idx; prev_k++)
            {
                int prev_expert = unpermuted_expert_selection[k * tid + prev_k];
                if (expert == prev_expert)
                {
                    valid = false;
                    break;
                }
            }

            if (valid)
            {
                int64_t const idx = k * tid + k_idx;
                unpermuted_expert_selection[idx] = expert < num_experts_per_node ? expert : num_experts;
                unpermuted_source_rows[idx] = k_idx * num_tokens + tid;
                break;
            }
        }
    }
}

__global__ void buildReverseMap(int* expanded_source_row_to_expanded_dest_row,
    int const* expanded_dest_row_to_expanded_source_row, int64_t expanded_num_tokens)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < expanded_num_tokens)
    {
        assert(expanded_dest_row_to_expanded_source_row[tid] >= 0);
        assert(expanded_dest_row_to_expanded_source_row[tid] < expanded_num_tokens);
        expanded_source_row_to_expanded_dest_row[expanded_dest_row_to_expanded_source_row[tid]] = tid;
    }
}

void GemmProfilerBackend::prepare(int num_tokens, char* workspace, cudaStream_t stream)
{
    mAllTacticsSaved = mInterface->getTactics();
    mSampleIndex = 0;

    int64_t num_expanded_tokens = num_tokens * mK;

    mSorter.updateNumExperts(mNumExperts);

    auto getNext = getWorkspacePointerGenerator(workspace, num_tokens, mSM >= 90);
    int64_t* expert_first_token_offset_base = reinterpret_cast<int64_t*>(getNext());
    int* source_to_dest_base = reinterpret_cast<int*>(getNext());
    int* dest_to_source_base = reinterpret_cast<int*>(getNext());
    int* unpermuted_expert_selection_base = reinterpret_cast<int*>(getNext());
    int* unpermuted_source_rows_base = reinterpret_cast<int*>(getNext());

    int* permuted_experts = reinterpret_cast<int*>(getNext());
    int* sorter_ws = reinterpret_cast<int*>(getNext());

    uint32_t num_threads = 256;
    dim3 grid_dim{(num_tokens + num_threads - 1) / num_threads, NUM_ROUTING_SAMPLES, 1};
    prepareFakeRouterBuffers<<<grid_dim, num_threads, 0, stream>>>(
        unpermuted_source_rows_base, unpermuted_expert_selection_base, num_tokens, mK, mNumExperts, mNumExpertsPerNode);
    sync_check_cuda_error();

    for (int64_t i = 0; i < NUM_ROUTING_SAMPLES; i++)
    {
        int64_t* expert_first_token_offset = expert_first_token_offset_base + i * (mNumExpertsPerNode + 1);
        int* source_to_dest = source_to_dest_base + i * num_expanded_tokens;
        int* dest_to_source = dest_to_source_base + i * num_expanded_tokens;
        int* unpermuted_expert_selection = unpermuted_expert_selection_base + i * num_expanded_tokens;
        int* unpermuted_source_rows = unpermuted_source_rows_base + i * num_expanded_tokens;

        sortAndScanSoftmaxOutput(unpermuted_expert_selection, unpermuted_source_rows, permuted_experts, dest_to_source,
            expert_first_token_offset, num_tokens, mNumExperts, mNumExpertsPerNode, mK, mSorter, sorter_ws, stream);

        sync_check_cuda_error();

        int grid_dim = (num_expanded_tokens + num_threads - 1) / num_threads;
        buildReverseMap<<<grid_dim, num_threads, 0, stream>>>(source_to_dest, dest_to_source, num_expanded_tokens);
    }
}

std::vector<size_t> GemmProfilerBackend::getProfilerWorkspaces(int maxM, bool is_hopper)
{
    size_t k = mK;
    size_t num_expanded_tokens = maxM * k;

    size_t dtype_bytes = tensorrt_llm::common::getDTypeSize(mDType);
    float weight_bytes
        = mWType == nvinfer1::DataType::kINT4 ? 0.5f : static_cast<float>(tensorrt_llm::common::getDTypeSize(mWType));
    size_t output_bytes = tensorrt_llm::common::getDTypeSize(mOType);
    size_t gemm_output_bytes = (mOType == nvinfer1::DataType::kFP8)
        ? sizeof(HopperGroupedGemmInput::OutputTypeAdaptor_t<__nv_fp8_e4m3>)
        : output_bytes;

    size_t hidden_size = mExpertHiddenSize;
    size_t inter_size = mExpertInterSize; // Already divided by TP
    size_t num_experts_per_node = mNumExpertsPerNode;

    size_t fc1_out_size = inter_size;
    if (isGatedActivation(mActivationType))
    {
        fc1_out_size = inter_size * 2;
    }

    // TODO Needs updated when gather/finalize fusion is integrated
    size_t input_size1 = hidden_size * num_expanded_tokens * dtype_bytes;
    size_t output_size1 = inter_size * num_expanded_tokens * dtype_bytes;

    size_t input_size2 = inter_size * num_expanded_tokens * dtype_bytes;
    size_t output_size2 = hidden_size * output_bytes;

    size_t input_size = mGemmToProfile == GemmToProfile::GEMM_1 ? input_size1 : input_size2;
    size_t output_size = mGemmToProfile == GemmToProfile::GEMM_1 ? output_size1 : output_size2;

    // This may allocate a pointer when not required. That's fine it will be ignored at the cost of some memory
    size_t intermediate_size1 = fc1_out_size * num_expanded_tokens * gemm_output_bytes; // Note gemm_output_bytes
    size_t intermediate_size2 = hidden_size * num_expanded_tokens * gemm_output_bytes;  // Note gemm_output_bytes

    size_t intermediate_size = mGemmToProfile == GemmToProfile::GEMM_1 ? intermediate_size1 : intermediate_size2;

    size_t weights_1 = hidden_size * fc1_out_size * num_experts_per_node * weight_bytes;
    size_t bias_1 = mBias ? fc1_out_size * num_experts_per_node * dtype_bytes : 0;
    if (mUseLora && !is_hopper)
        bias_1 = output_size1;
    size_t weights_2 = hidden_size * inter_size * num_experts_per_node * weight_bytes;
    size_t bias_2 = mBias ? hidden_size * num_experts_per_node * dtype_bytes : 0;

    size_t weights = mGemmToProfile == GemmToProfile::GEMM_1 ? weights_1 : weights_2;
    size_t bias = mGemmToProfile == GemmToProfile::GEMM_1 ? bias_1 : bias_2;

    // TODO Make quant 2 & 4 bigger for FP8 if we ever change to scaling per expert
    bool is_int_w_quant = mWType == nvinfer1::DataType::kINT8 || mWType == nvinfer1::DataType::kINT4;
    bool is_fp8_w_quant = mWType == nvinfer1::DataType::kFP8;

    // Int sizes
    size_t quant_1 = is_int_w_quant ? fc1_out_size * num_experts_per_node * dtype_bytes : 0;
    size_t quant_2 = is_int_w_quant ? hidden_size * num_experts_per_node * dtype_bytes : 0;

    // FP8 sizes
    quant_1 = is_fp8_w_quant ? num_experts_per_node * sizeof(float) : quant_1;
    quant_2 = is_fp8_w_quant ? sizeof(float) : quant_2;
    size_t quant_3 = is_fp8_w_quant ? num_experts_per_node * sizeof(float) : 0;
    size_t quant_4 = 0; // Currently ignored by the GEMM

    size_t hopper_workspace_size = 0;
    if (is_hopper)
    {
        hopper_workspace_size = HopperGroupedGemmInput::workspaceSize(num_experts_per_node);
    }

    size_t alpha_scale_ptr_array_size = num_experts_per_node * sizeof(float**);
    size_t gemm_workspace_size = mInterface->getGemmWorkspaceSize(num_experts_per_node);

    // Routing info
    size_t expert_first_token_offset_size = (num_experts_per_node + 1) * sizeof(int64_t) * NUM_ROUTING_SAMPLES;
    size_t map_size = NUM_ROUTING_SAMPLES * num_expanded_tokens * sizeof(int);
    size_t unpermuted_size = NUM_ROUTING_SAMPLES * num_expanded_tokens * sizeof(int);
    size_t permuted_size = num_expanded_tokens * sizeof(int);
    size_t sorter_ws_size = mSorter.getWorkspaceSize(num_expanded_tokens, mNumExperts);
    size_t token_topk_final_scale_size = num_expanded_tokens * sizeof(float);

    // Warning: order is sensitive. Routing info must be first and gemm_workspace_size must be last
    return {// These go first because they are needed for prepare
        expert_first_token_offset_size, map_size, map_size, unpermuted_size, unpermuted_size, permuted_size,
        permuted_size, sorter_ws_size, token_topk_final_scale_size,
        // The below are for the actual run
        input_size, output_size, intermediate_size, weights, bias, quant_1, quant_2, quant_3, quant_4,
        hopper_workspace_size, alpha_scale_ptr_array_size, gemm_workspace_size};
}

size_t GemmProfilerBackend::getWorkspaceSize(int maxM)
{
    auto sizes = getProfilerWorkspaces(maxM, mSM >= 90);
    return calculateTotalWorkspaceSize(sizes.data(), sizes.size());
}

std::function<void*()> GemmProfilerBackend::getWorkspacePointerGenerator(char* ws, int maxM, bool is_hopper)
{
    int8_t* workspace_ptr = reinterpret_cast<int8_t*>(ws);
    auto workspaces = getProfilerWorkspaces(maxM, is_hopper);
    auto index = 0;
    auto getNext = [=]() mutable -> void*
    {
        TLLM_CHECK_WITH_INFO(index < workspaces.size(), "Mismatching scratch space allocation");
        auto res = workspace_ptr;
        size_t element_size_bytes = workspaces[index];
        workspace_ptr = nextWorkspacePtr(workspace_ptr, element_size_bytes);
        index++;
        // Return nullptr if size is 0
        return element_size_bytes != 0 ? res : nullptr;
    };
    return getNext;
}

void GemmProfilerBackend::runProfiler(
    int original_num_tokens, Config const& tactic, char* workspace_ptr_char, cudaStream_t const& stream)
{
    int64_t expanded_num_tokens = original_num_tokens * mK;
    int64_t num_experts_per_node = mNumExpertsPerNode;

    mSampleIndex = (mSampleIndex + 1) % NUM_ROUTING_SAMPLES;

    auto workspaces = getProfilerWorkspaces(original_num_tokens, tactic.is_sm90);
    auto getNext = getWorkspacePointerGenerator(workspace_ptr_char, original_num_tokens, tactic.is_sm90);
    // Routing goes first as we need to manually initialise it in prepare(), everything else can be uninit
    // If we didn't init routing all the values could go to one expert, causing the profile to be unreliable
    auto const* expert_first_token_offset
        = static_cast<int64_t const*>(getNext()) + mSampleIndex * (mNumExpertsPerNode + 1);
    auto const* source_to_dest = static_cast<int const*>(getNext()) + mSampleIndex * expanded_num_tokens;
    auto const* dest_to_source = static_cast<int const*>(getNext()) + mSampleIndex * expanded_num_tokens;
    auto const* expert_for_source_row = static_cast<int const*>(getNext()) + mSampleIndex * expanded_num_tokens;

    std::ignore = getNext(); // Only used in prepare()
    std::ignore = getNext(); // Only used in prepare()
    std::ignore = getNext(); // Only used in prepare()
    std::ignore = getNext(); // Only used in prepare()

    // These are uninitialised so just alias them as they don't matter for performance
    auto const* token_topk_unpermuted_scales = static_cast<float const*>(getNext());
    auto const* token_topk_permuted_scales = token_topk_unpermuted_scales;

    void const* inputs = getNext();
    void* outputs = getNext();
    void* intermediate = getNext();
    void const* weights = getNext();
    void const* bias = getNext();
    void const* scale_1 = getNext();
    void const* scale_2 = getNext();
    void const* scale_3 = getNext();
    void const* scale_4 = getNext();
    void* hopper_workspace = getNext();
    float const** alpha_scale_ptr_array = reinterpret_cast<float const**>(getNext());
    void* gemm_workspace = getNext(); // NOTE we rely on this being last below (i.e. workspaces.back())

    HopperGroupedGemmInput hopper_input_template;
    if (tactic.is_sm90)
    {
        hopper_input_template.configureWorkspace(
            static_cast<int8_t*>(hopper_workspace), num_experts_per_node, gemm_workspace, workspaces.back());
    }

    QuantParams quant_params;
    if (mWType == nvinfer1::DataType::kINT8 || mWType == nvinfer1::DataType::kINT4)
    {
        TLLM_CHECK(scale_1 && scale_2);
        quant_params = QuantParams::Int(scale_1, scale_2);
    }
    else if (mWType == nvinfer1::DataType::kFP8)
    {
        TLLM_CHECK(scale_1 && scale_2 && scale_3);
        quant_params = QuantParams::FP8(static_cast<float const*>(scale_1), static_cast<float const*>(scale_2),
            static_cast<float const*>(scale_3), static_cast<float const*>(scale_4));
    }

    mInterface->is_profiler = true;
    if (mGemmToProfile == GemmToProfile::GEMM_1)
    {
        mInterface->gemm1(inputs,                             //
            outputs,                                          //
            intermediate,                                     //
            expert_first_token_offset,                        //
            hopper_input_template,                            //
            weights,                                          //
            bias,                                             //
            expert_first_token_offset + num_experts_per_node, //
            quant_params.fc1_weight_scales,                   //
            quant_params.dequant_fc1,                         //
            quant_params.quant_fc2,                           //
            expanded_num_tokens,                              //
            mExpertHiddenSize,                                //
            mExpertInterSize,                                 //
            num_experts_per_node,                             //
            mActivationType,                                  //
            alpha_scale_ptr_array,                            //
            !mUseLora,                                        //
            stream,                                           //
            tactic);
    }
    else
    {
        TLLM_CHECK(mGemmToProfile == GemmToProfile::GEMM_2);
        mInterface->gemm2(inputs,                           //
            intermediate,                                   //
            outputs,                                        //
            expert_first_token_offset,                      //
            hopper_input_template,                          //
            weights,                                        //
            bias,                                           //
            quant_params.fc2_weight_scales,                 //
            quant_params.dequant_fc2,                       //
            token_topk_unpermuted_scales,                   //
            token_topk_permuted_scales,                     //
            source_to_dest,                                 //
            dest_to_source,                                 //
            expert_for_source_row,                          //
            expert_first_token_offset + mNumExpertsPerNode, //
            original_num_tokens,                            //
            expanded_num_tokens,                            //
            mExpertHiddenSize,                              //
            mExpertInterSize,                               //
            num_experts_per_node,                           //
            mK,                                             //
            !mInterface->use_deterministic_hopper_reduce_,  //
            alpha_scale_ptr_array,                          //
            false,                                          //
            nullptr,                                        //
            stream,                                         //
            mParallelismConfig,                             //
            tactic);
    }
    mInterface->is_profiler = false;

    sync_check_cuda_error();
}

// ==================== Variable batched GEMM specializations ==================================
template class CutlassMoeFCRunner<float, float>;

#ifdef ENABLE_BF16
template class CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>;
template class CutlassMoeFCRunner<__nv_bfloat16, uint8_t>;
template class CutlassMoeFCRunner<__nv_bfloat16, cutlass::uint4b_t>;
#endif

template class CutlassMoeFCRunner<half, half>;
template class CutlassMoeFCRunner<half, uint8_t>;
template class CutlassMoeFCRunner<half, cutlass::uint4b_t>;
#ifdef ENABLE_FP8
// template class CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp8_e4m3>;
template class CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp8_e4m3, half>;
#ifdef ENABLE_BF16
template class CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp8_e4m3, __nv_bfloat16>;
#endif
#endif

} // namespace tensorrt_llm::kernels
