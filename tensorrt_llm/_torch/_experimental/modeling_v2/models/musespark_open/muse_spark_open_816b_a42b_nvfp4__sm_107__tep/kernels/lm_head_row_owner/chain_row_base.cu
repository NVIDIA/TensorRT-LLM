/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Portions copyright (c) 2024 by FlashInfer team.
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

// This target's copy of flashinfer's chain speculative sampling kernel
// (include/flashinfer/sampling.cuh ``ChainSpeculativeSampling``) with one change: the
// Philox subsequence of block b is ``row_base + b`` instead of ``b``. The stock kernel
// seeds every block with ``curand_init(seed_arr[0], blockIdx.x, offset_arr[0])``, so the
// random numbers of a request depend on its row in the batch; an owner that verifies
// requests [row_base, row_base + n) of a batch launches n blocks over its own rows and
// draws the numbers the stock kernel draws for those rows. ``seed_arr`` / ``offset_arr``
// are the batch's tensors (element 0 is read, as in the stock kernel). Everything else
// is the stock body, built by flashinfer's JIT with flashinfer's flags (``owner.py``
// ``chain_module``), so with ``row_base`` 0 and the whole batch it is the stock kernel.

#include <flashinfer/sampling.cuh>

#include "tvm_ffi_utils.h"

using namespace flashinfer;

using tvm::ffi::Optional;

namespace flashinfer
{
namespace sampling
{

template <uint32_t BLOCK_THREADS, BlockScanAlgorithm SCAN_ALGORITHM, BlockReduceAlgorithm REDUCE_ALGORITHM,
    uint32_t VEC_SIZE, bool DETERMINISTIC, typename DType, typename IdType>
__global__ FLASHINFER_SAMPLING_LAUNCH_BOUNDS(BLOCK_THREADS) void ChainSpeculativeSamplingRowBase(DType* draft_probs,
    IdType* draft_token_ids, DType* target_probs, IdType* output_token_ids, IdType* output_accepted_token_num,
    IdType* output_emitted_draft_token_num, uint32_t num_speculative_tokens, uint32_t d, uint64_t* seed_arr,
    uint64_t seed_val, uint64_t* offset_arr, uint64_t offset_val, uint32_t row_base)
{
    const uint32_t bx = blockIdx.x, tx = threadIdx.x;
    const uint32_t row_idx = bx;

    // Resolve seed/offset from tensor or scalar
    uint64_t philox_seed = seed_arr ? seed_arr[0] : seed_val;
    uint64_t philox_offset = offset_arr ? offset_arr[0] : offset_val;

    curandStatePhilox4_32_10_t curand_state;
    curand_init(philox_seed, row_base + bx, philox_offset, &curand_state);

    extern __shared__ __align__(alignof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>))
        uint8_t smem_sampling[];
    auto& temp_storage
        = reinterpret_cast<SamplingTempStorage<BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM>&>(smem_sampling);

    uint32_t pos = num_speculative_tokens;
    for (uint32_t i = 0; i < num_speculative_tokens; ++i)
    {
        IdType draft_id = draft_token_ids[row_idx * num_speculative_tokens + i];
        float q = target_probs[(row_idx * (num_speculative_tokens + 1) + i) * d + draft_id],
              p = draft_probs[(row_idx * num_speculative_tokens + i) * d + draft_id];
        float u = curand_uniform(&curand_state);
        if (u * p < q)
        {
            // accept the draft models output
            output_token_ids[row_idx * (num_speculative_tokens + 1) + i] = draft_id;
        }
        else
        {
            pos = i;
            break;
        }
    }

    uint32_t emitted_token_num = pos;
    uint32_t accepted_token_num = pos;
    for (uint32_t i = pos; i < num_speculative_tokens; ++i)
    {
        int draft_id = draft_token_ids[row_idx * num_speculative_tokens + i];
        float q = target_probs[(row_idx * (num_speculative_tokens + 1) + i) * d + draft_id],
              p = draft_probs[(row_idx * num_speculative_tokens + i) * d + draft_id];
        float u = curand_uniform(&curand_state);
        if (u * p < q)
        {
            ++accepted_token_num;
        }
    }

    if (tx == 0)
    {
        output_accepted_token_num[row_idx] += accepted_token_num;
        output_emitted_draft_token_num[row_idx] += emitted_token_num;
    }

    // sample from relu(target_probs - draft_probs)
    float sum_relu_q_minus_p = 0;
    vec_t<float, VEC_SIZE> q_vec, p_vec;
    float relu_q_minus_p[VEC_SIZE];
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i)
    {
        q_vec.fill(0);
        p_vec.fill(0);
        if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d)
        {
            q_vec.cast_load(target_probs + (row_idx * (num_speculative_tokens + 1) + pos) * d
                + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
            if (pos != num_speculative_tokens)
            {
                // there is no draft_probs for the bonus token
                p_vec.cast_load(draft_probs + (row_idx * num_speculative_tokens + pos) * d
                    + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
            }
        }
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j)
        {
            relu_q_minus_p[j] = max(q_vec[j] - p_vec[j], 0.0f);
        }
        sum_relu_q_minus_p += BlockReduce<float, BLOCK_THREADS, REDUCE_ALGORITHM>(temp_storage.block_prim.reduce)
                                  .Sum<VEC_SIZE>(relu_q_minus_p);
        __syncthreads();
    }
    if (tx == 0)
    {
        temp_storage.block_aggregate.value = sum_relu_q_minus_p;
    }
    // init the first rejected token to d
    temp_storage.sampled_id = d;
    __syncthreads();
    sum_relu_q_minus_p = temp_storage.block_aggregate.value;
    float u = curand_uniform(&curand_state) * sum_relu_q_minus_p;

    float aggregate_relu_q_minus_p(0);
#pragma unroll 2
    for (uint32_t i = 0; i < ceil_div(d, BLOCK_THREADS * VEC_SIZE); ++i)
    {
        q_vec.fill(0);
        p_vec.fill(0);
        if ((i * BLOCK_THREADS + tx) * VEC_SIZE < d)
        {
            q_vec.cast_load(target_probs + (row_idx * (num_speculative_tokens + 1) + pos) * d
                + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
            if (pos != num_speculative_tokens)
            {
                // there is no draft_probs for the bonus token
                p_vec.cast_load(draft_probs + (row_idx * num_speculative_tokens + pos) * d
                    + i * BLOCK_THREADS * VEC_SIZE + tx * VEC_SIZE);
            }
        }

        vec_t<float, VEC_SIZE> relu_q_minus_p_vec;
#pragma unroll
        for (uint32_t j = 0; j < VEC_SIZE; ++j)
        {
            relu_q_minus_p_vec[j] = max(q_vec[j] - p_vec[j], 0.0f);
        }

        DeviceSamplingFromProb<VEC_SIZE, BLOCK_THREADS, SCAN_ALGORITHM, REDUCE_ALGORITHM, DETERMINISTIC>(
            i, d, [&](float x) { return x > 0; }, u, relu_q_minus_p_vec, aggregate_relu_q_minus_p, &temp_storage);
        if (aggregate_relu_q_minus_p > u)
        {
            break;
        }
    }
    __syncthreads();
    int sampled_id = temp_storage.sampled_id;
    if (sampled_id == d)
    {
        // NOTE(Zihao): this would happen when u is very close to 1
        // and the sum of probabilities is smaller than u
        // In this case, we use the last valid index as the sampled id
        sampled_id = temp_storage.last_valid_id;
    }
    // set the first rejected token
    output_token_ids[row_idx * (num_speculative_tokens + 1) + pos] = sampled_id;
    // move to the next token
    pos++;

    // pad remaining tokens with -1
    for (; pos < num_speculative_tokens + 1; ++pos)
    {
        output_token_ids[row_idx * (num_speculative_tokens + 1) + pos] = -1;
    }
}

template <typename DType, typename IdType>
cudaError_t ChainSpeculativeSamplingRowBaseLaunch(DType* draft_probs, IdType* draft_token_ids, DType* target_probs,
    IdType* output_token_ids, IdType* output_accepted_token_num, IdType* output_emitted_draft_token_num,
    uint32_t batch_size, uint32_t num_speculative_tokens, uint32_t d, bool deterministic, uint64_t* seed_arr,
    uint64_t seed_val, uint64_t* offset_arr, uint64_t offset_val, uint32_t row_base, cudaStream_t stream = 0)
{
    const uint32_t vec_size = std::gcd(16 / sizeof(DType), d);

    auto compute_capacity = GetCudaComputeCapability();
    DISPATCH_COMPUTE_CAP_NUM_THREADS(compute_capacity, BLOCK_THREADS, {
        const uint32_t smem_size = sizeof(SamplingTempStorage<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO>);
        dim3 nblks(batch_size);
        dim3 nthrs(BLOCK_THREADS);
        void* args[] = {&draft_probs, &draft_token_ids, &target_probs, &output_token_ids, &output_accepted_token_num,
            &output_emitted_draft_token_num, &num_speculative_tokens, &d, &seed_arr, &seed_val, &offset_arr,
            &offset_val, &row_base};
        DISPATCH_ALIGNED_VEC_SIZE(vec_size, VEC_SIZE, {DISPATCH_DETERMINISTIC(deterministic, DETERMINISTIC, {
            auto kernel = ChainSpeculativeSamplingRowBase<BLOCK_THREADS, SCAN_ALGO, REDUCE_ALGO, VEC_SIZE,
                DETERMINISTIC, DType, IdType>;
            FLASHINFER_CUDA_CALL(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
            FLASHINFER_CUDA_CALL(cudaLaunchKernel((void*) kernel, nblks, nthrs, args, smem_size, stream));
        })});
        return cudaSuccess;
    });
}

} // namespace sampling
} // namespace flashinfer

//! flashinfer's ``chain_speculative_sampling`` binding with a row base; seed / offset are the
//! batch's int64 tensors (element 0 read).
void chain_speculative_sampling_row_base(TensorView draft_probs, TensorView draft_token_ids, TensorView target_probs,
    TensorView output_token_ids, TensorView output_accepted_token_num, TensorView output_emitted_draft_token_num,
    bool deterministic, TensorView seed_arr, TensorView offset_arr, int64_t row_base)
{
    CHECK_INPUT(draft_probs);
    CHECK_INPUT(draft_token_ids);
    CHECK_INPUT(target_probs);
    CHECK_INPUT(output_token_ids);
    CHECK_INPUT(seed_arr);
    CHECK_INPUT(offset_arr);
    CHECK_DEVICE(draft_token_ids, draft_probs);
    CHECK_DEVICE(target_probs, draft_probs);
    CHECK_DEVICE(seed_arr, draft_probs);
    CHECK_DEVICE(offset_arr, draft_probs);
    CHECK_DIM(3, draft_probs);
    CHECK_DIM(2, draft_token_ids);
    CHECK_DIM(3, target_probs);
    CHECK_DIM(1, seed_arr);
    CHECK_DIM(1, offset_arr);
    TVM_FFI_ICHECK(seed_arr.dtype() == dl_int64 || seed_arr.dtype() == dl_uint64) << "seed must be int64";
    TVM_FFI_ICHECK(offset_arr.dtype() == dl_int64 || offset_arr.dtype() == dl_uint64) << "offset must be int64";
    TVM_FFI_ICHECK(seed_arr.size(0) >= 1 && offset_arr.size(0) >= 1) << "seed / offset hold the batch's entries";
    TVM_FFI_ICHECK(row_base >= 0 && row_base < (int64_t(1) << 31)) << "row_base out of range";
    unsigned int batch_size = draft_probs.size(0);
    unsigned int num_speculate_tokens = draft_probs.size(1);
    unsigned int vocab_size = draft_probs.size(2);
    TVM_FFI_ICHECK_EQ(batch_size, draft_token_ids.size(0));
    TVM_FFI_ICHECK_EQ(batch_size, target_probs.size(0));
    TVM_FFI_ICHECK_EQ(num_speculate_tokens + 1, target_probs.size(1));
    TVM_FFI_ICHECK_EQ(vocab_size, target_probs.size(2));
    TVM_FFI_ICHECK_EQ(batch_size, output_token_ids.size(0));
    TVM_FFI_ICHECK_EQ(num_speculate_tokens + 1, output_token_ids.size(1));
    TVM_FFI_ICHECK_EQ(batch_size, output_accepted_token_num.size(0));
    TVM_FFI_ICHECK_EQ(batch_size, output_emitted_draft_token_num.size(0));
    if (batch_size == 0)
    {
        return;
    }

    ffi::CUDADeviceGuard device_guard(draft_probs.device().device_id);
    auto stream = get_stream(draft_probs.device());
    cudaError_t status
        = sampling::ChainSpeculativeSamplingRowBaseLaunch<float, int>(static_cast<float*>(draft_probs.data_ptr()),
            static_cast<int*>(draft_token_ids.data_ptr()), static_cast<float*>(target_probs.data_ptr()),
            static_cast<int*>(output_token_ids.data_ptr()), static_cast<int*>(output_accepted_token_num.data_ptr()),
            static_cast<int*>(output_emitted_draft_token_num.data_ptr()), batch_size, num_speculate_tokens, vocab_size,
            deterministic, static_cast<uint64_t*>(seed_arr.data_ptr()), 0,
            static_cast<uint64_t*>(offset_arr.data_ptr()), 0, static_cast<uint32_t>(row_base), stream);
    TVM_FFI_ICHECK(status == cudaSuccess)
        << "ChainSpeculativeSamplingRowBase failed with error code " << cudaGetErrorString(status);
}

TVM_FFI_DLL_EXPORT_TYPED_FUNC(chain_speculative_sampling_row_base, chain_speculative_sampling_row_base);
