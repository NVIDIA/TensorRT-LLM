/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include "loraGroupGEMMParamFillRowReorderFusion.h"

#include "tensorrt_llm/common/assert.h"
#include "tensorrt_llm/common/cudaUtils.h"

#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>

#include <algorithm>

TRTLLM_NAMESPACE_BEGIN

namespace kernels
{
namespace
{

template <class T>
__forceinline__ constexpr T (&as_singleton_array(T& obj))[1]
{
    return reinterpret_cast<T(&)[1]>(obj);
}

enum ParamIndex
{
    IN_SIZES_INDEX,
    OUT_SIZES_INDEX,
    LDA_INDEX,
    LDD_INDEX,
    LDB_PRIME_INDEX,
    LDD_PRIME_INDEX,
    A_PTRS_INDEX,
    D_PTRS_INDEX,
    D_PRIME_PTRS_INDEX,
    SPLITK_OFFSETS_INDEX,
    PARAM_COUNT,
};

int constexpr VECTOR_LOAD_WIDTH = 16;
} // namespace

/**
 * Fused kernel for LoRA group GEMM parameter filling, row gather and zero fillings.
 * Needs to be called with at least PARAM_COUNT blocks. And total number of threads need to be enough to reorder `input`
 * and fill zeros for intermediate and output buffers. Specifically, (n_threads >= max(input_bytes_to_reorder,
 * intermediate_bytes_to_fill, output_bytes_to_fill) / VECTOR_LOAD_WIDTH)
 *
 * Template parameters:
 * - BlockDim: Number of threads per block (1D, >= max_lora_count * module_count, >= 256, divisible by 32)
 * - MODULE_COUNT: Number of modules per layer
 */
template <int BlockDim, int MODULE_COUNT>
__global__ void loraGroupGEMMParamFillRowReorderFusionKernel(
    // Output parameters
    int32_t* in_sizes, int32_t* out_sizes, int64_t* a_ptrs, int64_t* d_ptrs, int64_t* d_prime_ptrs, int64_t* lda,
    int64_t* ldd, int64_t* ldb_prime, int64_t* ldd_prime, int64_t* splitk_offsets, uint8_t* reordered_input,
    // Input parameters
    int32_t max_lora_count, int32_t max_lora_rank, int32_t sum_output_hidden_size, int32_t input_hidden_size,
    int64_t dtype_element_size, int64_t batch_size, int64_t a_base, int64_t d_base, int64_t d_prime_base,
    int32_t const* slot_counts, int32_t const* slot_ranks, int64_t const* slot_offsets, int32_t const* module_out_sizes,
    int64_t const* module_out_prefix, int64_t const* b_ptrs, int64_t const* b_prime_ptrs, uint8_t const* input,
    int64_t const* sorted_ids)
{
    int const linearIdx = threadIdx.x;
    int const blockLinearIdx = blockIdx.x + blockIdx.y * gridDim.x;
    int constexpr THREADS_PER_BLOCK = BlockDim;

    // Calculate lora_id and module_id from linearIdx
    int const lora_id = linearIdx % max_lora_count;
    int const module_id = linearIdx / max_lora_count;

    using BlockLoad = cub::BlockLoad<int32_t, BlockDim, 1, cub::BLOCK_LOAD_DIRECT>;
    using BlockStore = cub::BlockStore<int32_t, BlockDim, 1, cub::BLOCK_STORE_DIRECT>;
    using BlockLoad64 = cub::BlockLoad<int64_t, BlockDim, 1, cub::BLOCK_LOAD_DIRECT>;
    using BlockStore64 = cub::BlockStore<int64_t, BlockDim, 1, cub::BLOCK_STORE_DIRECT>;
    using BlockScan = cub::BlockScan<int64_t, BlockDim, cub::BLOCK_SCAN_WARP_SCANS>;
    using BlockStore3 = cub::BlockStore<int32_t, BlockDim, 3, cub::BLOCK_STORE_WARP_TRANSPOSE>;

    __shared__ union
    {
        typename BlockStore3::TempStorage storage3;
        typename BlockScan::TempStorage scan;
    } large_shared;

    __shared__ int32_t
        row_count_to_gather; // rows > row_count_to_gather belong to non-LoRA requests, write zero directly

    switch (blockLinearIdx)
    {
    case IN_SIZES_INDEX:
    {
        int32_t slot_count = slot_counts[lora_id];
        int32_t rank = slot_ranks[lora_id];
        int64_t b_ptr = b_ptrs[linearIdx % (max_lora_count * MODULE_COUNT)];
        int32_t row[3] = {0};
        if (b_ptr != 0)
        {
            row[0] = slot_count;
            row[1] = rank;
            row[2] = input_hidden_size;
        }
        BlockStore3(large_shared.storage3).Store(in_sizes, row, max_lora_count * MODULE_COUNT * 3);
    }
    break;
    case OUT_SIZES_INDEX:
    {
        int32_t slot_count = slot_counts[lora_id];
        int32_t output_hidden_size = module_out_sizes[module_id];
        int32_t rank = slot_ranks[lora_id];
        int64_t b_ptr = b_ptrs[linearIdx % (max_lora_count * MODULE_COUNT)];
        int32_t row[3] = {0};
        if (b_ptr != 0)
        {
            row[0] = slot_count;
            row[1] = output_hidden_size;
            row[2] = rank;
        }
        BlockStore3(large_shared.storage3).Store(out_sizes, row, max_lora_count * MODULE_COUNT * 3);
    }
    break;
    case LDA_INDEX:
    {
        int64_t input_hidden_size_64[1] = {static_cast<int64_t>(input_hidden_size)};
        BlockStore64().Store(lda, input_hidden_size_64, max_lora_count * MODULE_COUNT);
    }
    break;
    case LDD_INDEX:
    {
        int64_t max_lora_rank_64[1] = {static_cast<int64_t>(max_lora_rank)};
        BlockStore64().Store(ldd, max_lora_rank_64, max_lora_count * MODULE_COUNT);
    }
    break;
    case LDB_PRIME_INDEX:
    {
        int64_t rank = slot_ranks[lora_id];
        BlockStore64().Store(ldb_prime, as_singleton_array(rank), max_lora_count * MODULE_COUNT);
    }
    break;
    case LDD_PRIME_INDEX:
    {
        int64_t sum_output_hidden_size_64[1] = {static_cast<int64_t>(sum_output_hidden_size)};
        BlockStore64().Store(ldd_prime, sum_output_hidden_size_64, max_lora_count * MODULE_COUNT);
    }
    break;
    case A_PTRS_INDEX:
    {
        int64_t slot_offset = 0;
        BlockLoad64().Load(slot_offsets, as_singleton_array(slot_offset), max_lora_count + 1);
        if (linearIdx == max_lora_count)
        {
            row_count_to_gather = static_cast<int32_t>(slot_offset);
        }
        slot_offset *= input_hidden_size;
        slot_offset *= dtype_element_size;
        slot_offset += a_base;
        for (int i = 0; i < MODULE_COUNT; i++)
        {
            BlockStore64().Store(a_ptrs + i * max_lora_count, as_singleton_array(slot_offset), max_lora_count);
        }
    }
    break;
    case D_PTRS_INDEX:
    {
        int64_t slot_offset = 0;
        BlockLoad64().Load(slot_offsets, as_singleton_array(slot_offset), max_lora_count + 1);
        for (int i = 0; i < MODULE_COUNT; i++)
        {
            int64_t offset = slot_offset;
            offset += i * batch_size;
            offset *= max_lora_rank;
            offset *= dtype_element_size;
            offset += d_base;
            BlockStore64().Store(d_ptrs + i * max_lora_count, as_singleton_array(offset), max_lora_count);
        }
        if (linearIdx == max_lora_count)
        {
            row_count_to_gather = static_cast<int32_t>(slot_offset);
        }
    }
    break;
    case D_PRIME_PTRS_INDEX:
    {
        int64_t slot_offset = 0;
        BlockLoad64().Load(slot_offsets, as_singleton_array(slot_offset), max_lora_count + 1);
        if (linearIdx == max_lora_count)
        {
            row_count_to_gather = static_cast<int32_t>(slot_offset);
        }
        slot_offset *= sum_output_hidden_size;
        for (int i = 0; i < MODULE_COUNT; i++)
        {
            int64_t offset = slot_offset;
            offset += module_out_prefix[i];
            offset *= dtype_element_size;
            offset += d_prime_base;
            BlockStore64().Store(d_prime_ptrs + i * max_lora_count, as_singleton_array(offset), max_lora_count);
        }
    }
    break;
    case SPLITK_OFFSETS_INDEX:
    {
        int64_t slot_count = slot_counts[lora_id];
        int64_t rank = slot_ranks[lora_id];
        int64_t b_ptr = b_ptrs[linearIdx % (max_lora_count * MODULE_COUNT)];
        int64_t splitk_offset = (b_ptr == 0) ? 0 : (slot_count * rank);
        BlockScan(large_shared.scan).ExclusiveSum(splitk_offset, splitk_offset);
        BlockStore64().Store(splitk_offsets, as_singleton_array(splitk_offset), max_lora_count * MODULE_COUNT);
    }
    break;
    }

    // Set row_count_to_gather for non-pointer blocks
    switch (blockLinearIdx)
    {
    case A_PTRS_INDEX:
    case D_PTRS_INDEX:
    case D_PRIME_PTRS_INDEX: break;
    default:
        if (linearIdx == 0)
        {
            row_count_to_gather = static_cast<int32_t>(slot_offsets[max_lora_count]);
        }
    }

    int constexpr ITEM_PER_THREAD = VECTOR_LOAD_WIDTH;
    using BlockStoreRow = cub::BlockStore<uint8_t, BlockDim, ITEM_PER_THREAD, cub::BLOCK_STORE_VECTORIZE>;

    {
        // Write zero to intermediate buffer and output buffer
        auto intermediate_cast = reinterpret_cast<uint8_t*>(d_base);
        auto model_output_cast = reinterpret_cast<uint8_t*>(d_prime_base);

        int intermediate_size = MODULE_COUNT * batch_size * max_lora_rank * dtype_element_size;
        int output_size = batch_size * sum_output_hidden_size * dtype_element_size;

        uint8_t all_zeroes[ITEM_PER_THREAD] = {0};

        int const blockOffset = THREADS_PER_BLOCK * ITEM_PER_THREAD * blockLinearIdx;
        BlockStoreRow().Store(intermediate_cast + blockOffset, all_zeroes, intermediate_size - blockOffset);
        BlockStoreRow().Store(model_output_cast + blockOffset, all_zeroes, output_size - blockOffset);
    }

    __syncthreads();

    // Row gather
    if (blockIdx.y < batch_size)
    {
        using BlockLoadRow = cub::BlockLoad<uint8_t, BlockDim, ITEM_PER_THREAD, cub::BLOCK_LOAD_VECTORIZE>;

        auto const row_size = input_hidden_size * dtype_element_size;

        auto output_cast = reinterpret_cast<uint8_t*>(reordered_input);

        uint8_t tile[ITEM_PER_THREAD] = {0};
        int constexpr x_stride = THREADS_PER_BLOCK * ITEM_PER_THREAD;
        int const y_stride = row_size;
        int tail = row_size - blockIdx.x * x_stride;
        if (blockIdx.y < row_count_to_gather)
        {
            auto const input_cast = reinterpret_cast<uint8_t const*>(input);
            auto const src_row = sorted_ids[blockIdx.y];
            BlockLoadRow().Load(input_cast + blockIdx.x * x_stride + src_row * y_stride, tile, tail);
        }
        BlockStoreRow().Store(output_cast + blockIdx.x * x_stride + blockIdx.y * y_stride, tile, tail);
    }
}

/**
 * Launch function that instantiates the appropriate kernel based on module count.
 */
template <int BlockDim>
void launchKernelWithModuleCount(int32_t* in_sizes, int32_t* out_sizes, int64_t* a_ptrs, int64_t* d_ptrs,
    int64_t* d_prime_ptrs, int64_t* lda, int64_t* ldd, int64_t* ldb_prime, int64_t* ldd_prime, int64_t* splitk_offsets,
    void* reordered_input, int32_t max_lora_count, int32_t max_lora_rank, int32_t sum_output_hidden_size,
    int32_t input_hidden_size, int64_t dtype_element_size, int64_t batch_size, int64_t a_base, int64_t d_base,
    int64_t d_prime_base, int32_t const* slot_counts, int32_t const* slot_ranks, int64_t const* slot_offsets,
    int32_t const* module_out_sizes, int64_t const* module_out_prefix, int64_t const* b_ptrs,
    int64_t const* b_prime_ptrs, void const* input, int64_t const* sorted_ids, int32_t module_count,
    cudaStream_t stream)
{
    int constexpr THREADS_PER_BLOCK = BlockDim;

    // Grid dimensions for row gather
    int constexpr ITEMS_PER_BLOCK = THREADS_PER_BLOCK * VECTOR_LOAD_WIDTH;
    int const gridDimX = common::ceilDiv(input_hidden_size * dtype_element_size, ITEMS_PER_BLOCK);
    int gridDimY = std::max(
        static_cast<int>(common::ceilDiv(static_cast<int>(PARAM_COUNT), gridDimX)), static_cast<int>(batch_size));

    // calculate threads needed for writing zeros to intermediate buffer and output buffer
    int const itemsPerRow = ITEMS_PER_BLOCK * gridDimX;
    gridDimY = std::max(gridDimY,
        common::ceilDiv(static_cast<int>(module_count * batch_size * max_lora_rank * dtype_element_size), itemsPerRow));
    gridDimY = std::max(gridDimY,
        common::ceilDiv(static_cast<int>(batch_size * sum_output_hidden_size * dtype_element_size), itemsPerRow));

    dim3 grid(gridDimX, gridDimY);
    dim3 block(BlockDim);

    auto* reordered_input_cast = reinterpret_cast<uint8_t*>(reordered_input);
    auto const* input_cast = reinterpret_cast<uint8_t const*>(input);

    // Dispatch based on module count
    switch (module_count)
    {
    case 1:
        loraGroupGEMMParamFillRowReorderFusionKernel<BlockDim, 1><<<grid, block, 0, stream>>>(in_sizes, out_sizes,
            a_ptrs, d_ptrs, d_prime_ptrs, lda, ldd, ldb_prime, ldd_prime, splitk_offsets, reordered_input_cast,
            max_lora_count, max_lora_rank, sum_output_hidden_size, input_hidden_size, dtype_element_size, batch_size,
            a_base, d_base, d_prime_base, slot_counts, slot_ranks, slot_offsets, module_out_sizes, module_out_prefix,
            b_ptrs, b_prime_ptrs, input_cast, sorted_ids);
        break;
    case 2:
        loraGroupGEMMParamFillRowReorderFusionKernel<BlockDim, 2><<<grid, block, 0, stream>>>(in_sizes, out_sizes,
            a_ptrs, d_ptrs, d_prime_ptrs, lda, ldd, ldb_prime, ldd_prime, splitk_offsets, reordered_input_cast,
            max_lora_count, max_lora_rank, sum_output_hidden_size, input_hidden_size, dtype_element_size, batch_size,
            a_base, d_base, d_prime_base, slot_counts, slot_ranks, slot_offsets, module_out_sizes, module_out_prefix,
            b_ptrs, b_prime_ptrs, input_cast, sorted_ids);
        break;
    case 3:
        loraGroupGEMMParamFillRowReorderFusionKernel<BlockDim, 3><<<grid, block, 0, stream>>>(in_sizes, out_sizes,
            a_ptrs, d_ptrs, d_prime_ptrs, lda, ldd, ldb_prime, ldd_prime, splitk_offsets, reordered_input_cast,
            max_lora_count, max_lora_rank, sum_output_hidden_size, input_hidden_size, dtype_element_size, batch_size,
            a_base, d_base, d_prime_base, slot_counts, slot_ranks, slot_offsets, module_out_sizes, module_out_prefix,
            b_ptrs, b_prime_ptrs, input_cast, sorted_ids);
        break;
    case 4:
        loraGroupGEMMParamFillRowReorderFusionKernel<BlockDim, 4><<<grid, block, 0, stream>>>(in_sizes, out_sizes,
            a_ptrs, d_ptrs, d_prime_ptrs, lda, ldd, ldb_prime, ldd_prime, splitk_offsets, reordered_input_cast,
            max_lora_count, max_lora_rank, sum_output_hidden_size, input_hidden_size, dtype_element_size, batch_size,
            a_base, d_base, d_prime_base, slot_counts, slot_ranks, slot_offsets, module_out_sizes, module_out_prefix,
            b_ptrs, b_prime_ptrs, input_cast, sorted_ids);
        break;
    default: TLLM_CHECK_WITH_INFO(false, "Unsupported module_count: %d (max 4)", module_count);
    }
}

void launchLoraGroupGEMMParamFillRowReorderFusion(int32_t* in_sizes, int32_t* out_sizes, int64_t* a_ptrs,
    int64_t* d_ptrs, int64_t* d_prime_ptrs, int64_t* lda, int64_t* ldd, int64_t* ldb_prime, int64_t* ldd_prime,
    int64_t* splitk_offsets, void* reordered_input, int32_t max_lora_count, int32_t max_lora_rank,
    int32_t sum_output_hidden_size, int32_t input_hidden_size, int64_t dtype_element_size, int64_t batch_size,
    int64_t a_base, int64_t d_base, int64_t d_prime_base, int32_t const* slot_counts, int32_t const* slot_ranks,
    int64_t const* slot_offsets, int32_t const* module_out_sizes, int64_t const* module_out_prefix,
    int64_t const* b_ptrs, int64_t const* b_prime_ptrs, void const* input, int64_t const* sorted_ids,
    int32_t module_count, nvinfer1::DataType dtype, cudaStream_t stream)
{
    // Determine block dimensions (1D)
    // Requirements: 1) >= max_lora_count * module_count 2) >= 256 3) divisible by 32

    int constexpr MIN_THREADS = 256;
    int constexpr WARP_SIZE = 32;

    int const min_threads_needed = max_lora_count * module_count;
    int const threads_per_block = std::max(MIN_THREADS, common::ceilDiv(min_threads_needed, WARP_SIZE) * WARP_SIZE);

    if (threads_per_block == 256)
    {
        launchKernelWithModuleCount<256>(in_sizes, out_sizes, a_ptrs, d_ptrs, d_prime_ptrs, lda, ldd, ldb_prime,
            ldd_prime, splitk_offsets, reordered_input, max_lora_count, max_lora_rank, sum_output_hidden_size,
            input_hidden_size, dtype_element_size, batch_size, a_base, d_base, d_prime_base, slot_counts, slot_ranks,
            slot_offsets, module_out_sizes, module_out_prefix, b_ptrs, b_prime_ptrs, input, sorted_ids, module_count,
            stream);
    }
    else if (threads_per_block == 288)
    {
        launchKernelWithModuleCount<288>(in_sizes, out_sizes, a_ptrs, d_ptrs, d_prime_ptrs, lda, ldd, ldb_prime,
            ldd_prime, splitk_offsets, reordered_input, max_lora_count, max_lora_rank, sum_output_hidden_size,
            input_hidden_size, dtype_element_size, batch_size, a_base, d_base, d_prime_base, slot_counts, slot_ranks,
            slot_offsets, module_out_sizes, module_out_prefix, b_ptrs, b_prime_ptrs, input, sorted_ids, module_count,
            stream);
    }
    else if (threads_per_block == 320)
    {
        launchKernelWithModuleCount<320>(in_sizes, out_sizes, a_ptrs, d_ptrs, d_prime_ptrs, lda, ldd, ldb_prime,
            ldd_prime, splitk_offsets, reordered_input, max_lora_count, max_lora_rank, sum_output_hidden_size,
            input_hidden_size, dtype_element_size, batch_size, a_base, d_base, d_prime_base, slot_counts, slot_ranks,
            slot_offsets, module_out_sizes, module_out_prefix, b_ptrs, b_prime_ptrs, input, sorted_ids, module_count,
            stream);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false,
            "Unsupported threads_per_block: %d (calculated from max_lora_count=%d * module_count=%d)",
            threads_per_block, max_lora_count, module_count);
    }

    sync_check_cuda_error(stream);
}

} // namespace kernels

TRTLLM_NAMESPACE_END
