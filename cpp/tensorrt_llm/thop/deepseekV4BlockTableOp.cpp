/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "tensorrt_llm/kernels/deepseekV4BlockTable.h"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>
#include <torch/extension.h>

namespace th = torch;
namespace tk = tensorrt_llm::kernels;

TRTLLM_NAMESPACE_BEGIN

namespace torch_ext
{
namespace
{

void checkInt32Tensor(th::Tensor const& tensor, char const* name)
{
    TORCH_CHECK(tensor.scalar_type() == th::kInt32, name, " must be int32");
}

void checkCudaContiguousTensor(th::Tensor const& tensor, char const* name, int device)
{
    TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
    TORCH_CHECK(tensor.get_device() == device, name, " must be on the same CUDA device as output");
    TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void checkLayerAttnShape(th::Tensor const& tensor, char const* name, int64_t numLayers, int64_t numAttnTypes)
{
    TORCH_CHECK(tensor.dim() == 2, name, " must be 2D [num_layers, num_attn_types]");
    TORCH_CHECK(tensor.size(0) == numLayers && tensor.size(1) == numAttnTypes, name,
        " must match pool_ids shape [num_layers, num_attn_types]");
}

int32_t checkedInt32Size(int64_t value, char const* name)
{
    TORCH_CHECK(value <= std::numeric_limits<int32_t>::max(), name, " exceeds int32 range");
    return static_cast<int32_t>(value);
}

void checkCommonInputs(th::Tensor const& blockOffsets, th::Tensor const& copyIdx, th::Tensor const& poolIds,
    th::Tensor const& validPool, th::Tensor const& scales, th::Tensor const& layerOffsets, th::Tensor const& output)
{
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    int const device = output.get_device();
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    checkCudaContiguousTensor(blockOffsets, "block_offsets", device);
    checkCudaContiguousTensor(copyIdx, "copy_idx", device);
    checkCudaContiguousTensor(poolIds, "pool_ids", device);
    checkCudaContiguousTensor(validPool, "valid_pool", device);
    checkCudaContiguousTensor(scales, "scales", device);
    checkCudaContiguousTensor(layerOffsets, "layer_offsets", device);

    checkInt32Tensor(blockOffsets, "block_offsets");
    checkInt32Tensor(copyIdx, "copy_idx");
    TORCH_CHECK(poolIds.scalar_type() == th::kInt64, "pool_ids must be int64");
    TORCH_CHECK(validPool.scalar_type() == th::kBool, "valid_pool must be bool");
    checkInt32Tensor(scales, "scales");
    checkInt32Tensor(layerOffsets, "layer_offsets");
    checkInt32Tensor(output, "output");

    TORCH_CHECK(blockOffsets.dim() == 4, "block_offsets must be 4D [num_pools, table_capacity, 2, max_blocks]");
    TORCH_CHECK(blockOffsets.size(2) == 2, "block_offsets dim 2 must be 2");
    TORCH_CHECK(copyIdx.dim() == 1, "copy_idx must be 1D");
    TORCH_CHECK(poolIds.dim() == 2, "pool_ids must be 2D [num_layers, num_attn_types]");
    TORCH_CHECK(output.dim() == 4, "output must be 4D [num_layers, num_attn_types, num_tables, max_blocks]");

    int64_t const numLayers = poolIds.size(0);
    int64_t const numAttnTypes = poolIds.size(1);
    int64_t const numTables = copyIdx.size(0);
    int64_t const maxBlocksPerSeq = blockOffsets.size(3);

    checkLayerAttnShape(validPool, "valid_pool", numLayers, numAttnTypes);
    checkLayerAttnShape(scales, "scales", numLayers, numAttnTypes);
    checkLayerAttnShape(layerOffsets, "layer_offsets", numLayers, numAttnTypes);

    TORCH_CHECK(output.size(0) == numLayers && output.size(1) == numAttnTypes && output.size(2) == numTables
            && output.size(3) == maxBlocksPerSeq,
        "output shape must be [pool_ids.size(0), pool_ids.size(1), copy_idx.size(0), block_offsets.size(3)]");
}

} // namespace

void deepseekV4ComputeSlidingBlockTables(th::Tensor const& blockOffsets, th::Tensor const& copyIdx,
    th::Tensor const& poolIds, th::Tensor const& validPool, th::Tensor const& scales, th::Tensor const& layerOffsets,
    th::Tensor const& output)
{
    checkCommonInputs(blockOffsets, copyIdx, poolIds, validPool, scales, layerOffsets, output);
    c10::cuda::CUDAGuard const deviceGuard(output.device());

    int32_t const numPools = checkedInt32Size(blockOffsets.size(0), "num_pools");
    int32_t const copyIdxCapacity = checkedInt32Size(blockOffsets.size(1), "copy_idx_capacity");
    int32_t const numLayers = checkedInt32Size(poolIds.size(0), "num_layers");
    int32_t const numAttnTypes = checkedInt32Size(poolIds.size(1), "num_attn_types");
    int32_t const numTables = checkedInt32Size(copyIdx.size(0), "num_tables");
    int32_t const maxBlocksPerSeq = checkedInt32Size(blockOffsets.size(3), "max_blocks_per_seq");

    auto stream = at::cuda::getCurrentCUDAStream(output.get_device());
    tk::invokeDeepseekV4ComputeSlidingBlockTables(blockOffsets.data_ptr<int32_t>(), copyIdx.data_ptr<int32_t>(),
        poolIds.data_ptr<int64_t>(), validPool.data_ptr<bool>(), scales.data_ptr<int32_t>(),
        layerOffsets.data_ptr<int32_t>(), output.data_ptr<int32_t>(), numPools, copyIdxCapacity, numLayers,
        numAttnTypes, numTables, maxBlocksPerSeq, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void deepseekV4ComputeSlidingBlockTablesWithScratch(th::Tensor const& blockOffsets, th::Tensor const& copyIdx,
    th::Tensor const& poolIds, th::Tensor const& validPool, th::Tensor const& scales, th::Tensor const& layerOffsets,
    th::Tensor const& scratchPages, th::Tensor const& scratchBegs, th::Tensor const& scratchEnds,
    th::Tensor const& scratchSlots, th::Tensor const& numContexts, th::Tensor const& output)
{
    checkCommonInputs(blockOffsets, copyIdx, poolIds, validPool, scales, layerOffsets, output);
    int const device = output.get_device();
    checkCudaContiguousTensor(scratchPages, "scratch_pages", device);
    checkCudaContiguousTensor(scratchBegs, "scratch_begs", device);
    checkCudaContiguousTensor(scratchEnds, "scratch_ends", device);
    checkCudaContiguousTensor(scratchSlots, "scratch_slots", device);
    checkCudaContiguousTensor(numContexts, "num_contexts", device);

    checkInt32Tensor(scratchPages, "scratch_pages");
    checkInt32Tensor(scratchBegs, "scratch_begs");
    checkInt32Tensor(scratchEnds, "scratch_ends");
    checkInt32Tensor(scratchSlots, "scratch_slots");
    checkInt32Tensor(numContexts, "num_contexts");

    int64_t const numLayers = poolIds.size(0);
    int64_t const numAttnTypes = poolIds.size(1);
    checkLayerAttnShape(scratchPages, "scratch_pages", numLayers, numAttnTypes);
    TORCH_CHECK(scratchBegs.dim() == 2, "scratch_begs must be 2D [num_pools, scratch_capacity]");
    TORCH_CHECK(scratchEnds.dim() == 2, "scratch_ends must be 2D [num_pools, scratch_capacity]");
    TORCH_CHECK(scratchSlots.dim() == 3, "scratch_slots must be 3D [num_pools, scratch_capacity, max_scratch_slots]");
    TORCH_CHECK(scratchBegs.size(0) == blockOffsets.size(0), "scratch_begs.size(0) must match num_pools");
    TORCH_CHECK(scratchEnds.size(0) == scratchBegs.size(0) && scratchEnds.size(1) == scratchBegs.size(1),
        "scratch_ends shape must match scratch_begs");
    TORCH_CHECK(scratchSlots.size(0) == scratchBegs.size(0) && scratchSlots.size(1) == scratchBegs.size(1),
        "scratch_slots first two dimensions must match scratch_begs");
    TORCH_CHECK(scratchBegs.size(1) <= output.size(2), "scratch_capacity must not exceed num_tables");
    TORCH_CHECK(numContexts.dim() == 0 && numContexts.numel() == 1, "num_contexts must be a scalar tensor");
    TORCH_CHECK(scratchSlots.size(2) > 0 || scratchBegs.size(1) == 0,
        "max_scratch_slots must be positive when scratch_capacity is nonzero");

    c10::cuda::CUDAGuard const deviceGuard(output.device());
    int32_t const numPools = checkedInt32Size(blockOffsets.size(0), "num_pools");
    int32_t const copyIdxCapacity = checkedInt32Size(blockOffsets.size(1), "copy_idx_capacity");
    int32_t const numLayers32 = checkedInt32Size(numLayers, "num_layers");
    int32_t const numAttnTypes32 = checkedInt32Size(numAttnTypes, "num_attn_types");
    int32_t const numTables = checkedInt32Size(copyIdx.size(0), "num_tables");
    int32_t const maxBlocksPerSeq = checkedInt32Size(blockOffsets.size(3), "max_blocks_per_seq");
    int32_t const scratchCapacity = checkedInt32Size(scratchBegs.size(1), "scratch_capacity");
    int32_t const maxScratchSlots = checkedInt32Size(scratchSlots.size(2), "max_scratch_slots");

    auto stream = at::cuda::getCurrentCUDAStream(output.get_device());
    tk::invokeDeepseekV4ComputeSlidingBlockTablesWithScratch(blockOffsets.data_ptr<int32_t>(),
        copyIdx.data_ptr<int32_t>(), poolIds.data_ptr<int64_t>(), validPool.data_ptr<bool>(),
        scales.data_ptr<int32_t>(), layerOffsets.data_ptr<int32_t>(), scratchPages.data_ptr<int32_t>(),
        scratchBegs.data_ptr<int32_t>(), scratchEnds.data_ptr<int32_t>(), scratchSlots.data_ptr<int32_t>(),
        numContexts.data_ptr<int32_t>(), output.data_ptr<int32_t>(), numPools, copyIdxCapacity, numLayers32,
        numAttnTypes32, numTables, maxBlocksPerSeq, scratchCapacity, maxScratchSlots, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

} // namespace torch_ext

TRTLLM_NAMESPACE_END

TORCH_LIBRARY_FRAGMENT(trtllm, m)
{
    m.def(
        "deepseek_v4_compute_sliding_block_tables(Tensor block_offsets, Tensor copy_idx, Tensor pool_ids, "
        "Tensor valid_pool, Tensor scales, Tensor layer_offsets, Tensor(a!) output) -> ()");
    m.def(
        "deepseek_v4_compute_sliding_block_tables_with_scratch(Tensor block_offsets, Tensor copy_idx, "
        "Tensor pool_ids, Tensor valid_pool, Tensor scales, Tensor layer_offsets, Tensor scratch_pages, "
        "Tensor scratch_begs, Tensor scratch_ends, Tensor scratch_slots, Tensor num_contexts, "
        "Tensor(a!) output) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("deepseek_v4_compute_sliding_block_tables", &tensorrt_llm::torch_ext::deepseekV4ComputeSlidingBlockTables);
    m.impl("deepseek_v4_compute_sliding_block_tables_with_scratch",
        &tensorrt_llm::torch_ext::deepseekV4ComputeSlidingBlockTablesWithScratch);
}
