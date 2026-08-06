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

#include "tensorrt_llm/kernels/attentionMetadataKernels.h"
#include "tensorrt_llm/kernels/deepseekV4BlockTable.h"
#include "tensorrt_llm/kernels/deepseekV4CompressedMeta.h"
#include "tensorrt_llm/kernels/deepseekV4Indices.h"

#include <ATen/cuda/CUDAContext.h>
#include <algorithm>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <limits>
#include <optional>
#include <torch/extension.h>
#include <vector>

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

void computeSharedBlockTable(
    th::Tensor const& blockOffsets, th::Tensor const& copyIdx, int64_t poolId, int64_t scale, th::Tensor const& output)
{
    int const device = output.get_device();
    checkCudaContiguousTensor(blockOffsets, "block_offsets", device);
    checkCudaContiguousTensor(copyIdx, "copy_idx", device);
    checkCudaContiguousTensor(output, "output", device);
    checkInt32Tensor(blockOffsets, "block_offsets");
    checkInt32Tensor(copyIdx, "copy_idx");
    checkInt32Tensor(output, "output");
    TORCH_CHECK(blockOffsets.dim() == 4, "block_offsets must be 4D [num_pools, copy_idx_capacity, 2, max_blocks]");
    TORCH_CHECK(copyIdx.dim() == 1, "copy_idx must be 1D [num_tables]");
    TORCH_CHECK(output.dim() == 2, "output must be 2D [num_tables, max_blocks_per_seq]");
    TORCH_CHECK(output.size(1) == blockOffsets.size(3), "output max_blocks must match block_offsets");
    TORCH_CHECK(poolId >= 0 && poolId < blockOffsets.size(0), "pool_id out of range");

    c10::cuda::CUDAGuard const deviceGuard(output.device());
    int32_t const copyIdxCapacity = checkedInt32Size(blockOffsets.size(1), "copy_idx_capacity");
    int32_t const maxBlocksPerSeq = checkedInt32Size(blockOffsets.size(3), "max_blocks_per_seq");
    // The caller may pass a fixed-capacity copy_idx buffer; honour the smaller of
    // the two so a padded staging tensor cannot write past the output rows.
    int32_t const numTables
        = std::min(checkedInt32Size(copyIdx.size(0), "num_tables"), checkedInt32Size(output.size(0), "output_rows"));

    auto stream = at::cuda::getCurrentCUDAStream(device);
    tk::invokeComputeSharedBlockTable(blockOffsets.data_ptr<int32_t>(), copyIdx.data_ptr<int32_t>(),
        output.data_ptr<int32_t>(), static_cast<int32_t>(poolId), static_cast<int32_t>(scale), copyIdxCapacity,
        numTables, maxBlocksPerSeq, stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void deepseekV4ComputeIndices(th::Tensor const& tokenPositions, int64_t windowSize, int64_t maxCompressedIndices,
    int64_t sparseMlaTopk, th::Tensor const& swaLocalIndices, th::Tensor const& compressedLocalIndices,
    std::optional<th::Tensor> const& topkLensRatio1, std::optional<th::Tensor> const& topkLensRatio4,
    std::optional<th::Tensor> const& topkLensRatio128)
{
    int const device = swaLocalIndices.get_device();
    checkCudaContiguousTensor(tokenPositions, "token_positions", device);
    checkCudaContiguousTensor(swaLocalIndices, "swa_local_indices", device);
    checkCudaContiguousTensor(compressedLocalIndices, "compressed_local_indices", device);
    checkInt32Tensor(tokenPositions, "token_positions");
    checkInt32Tensor(swaLocalIndices, "swa_local_indices");
    checkInt32Tensor(compressedLocalIndices, "compressed_local_indices");
    TORCH_CHECK(tokenPositions.dim() == 1, "token_positions must be 1D [num_tokens]");
    TORCH_CHECK(swaLocalIndices.dim() == 2, "swa_local_indices must be 2D [rows, window_size]");
    TORCH_CHECK(compressedLocalIndices.dim() == 2, "compressed_local_indices must be 2D");
    TORCH_CHECK(
        windowSize > 0 && windowSize <= swaLocalIndices.size(1), "window_size must fit swa_local_indices columns");
    TORCH_CHECK(maxCompressedIndices > 0 && maxCompressedIndices <= compressedLocalIndices.size(1),
        "max_compressed_indices must fit compressed_local_indices columns");

    int32_t const numTokens = checkedInt32Size(tokenPositions.size(0), "num_tokens");
    TORCH_CHECK(numTokens <= swaLocalIndices.size(0) && numTokens <= compressedLocalIndices.size(0),
        "output buffers must have at least num_tokens rows");

    auto ptr = [&](std::optional<th::Tensor> const& t) -> int32_t*
    {
        if (!t.has_value())
        {
            return nullptr;
        }
        checkCudaContiguousTensor(*t, "sparse_mla_topk_lens", device);
        checkInt32Tensor(*t, "sparse_mla_topk_lens");
        TORCH_CHECK(t->numel() >= numTokens, "sparse_mla_topk_lens must hold num_tokens entries");
        return t->data_ptr<int32_t>();
    };

    c10::cuda::CUDAGuard const deviceGuard(swaLocalIndices.device());
    auto stream = at::cuda::getCurrentCUDAStream(device);
    tk::invokeDeepseekV4ComputeIndices(tokenPositions.data_ptr<int32_t>(), swaLocalIndices.data_ptr<int32_t>(),
        compressedLocalIndices.data_ptr<int32_t>(), ptr(topkLensRatio1), ptr(topkLensRatio4), ptr(topkLensRatio128),
        numTokens, static_cast<int32_t>(windowSize), static_cast<int32_t>(maxCompressedIndices),
        static_cast<int32_t>(sparseMlaTopk), checkedInt32Size(swaLocalIndices.stride(0), "swa_stride"),
        checkedInt32Size(compressedLocalIndices.stride(0), "compressed_stride"), stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

namespace
{
// Fills the per-ratio pointer/scalar slots of a launch-packet struct from the
// python-side lists, validating each tensor. Returns the ratio count.
template <typename FillFn>
int32_t forEachRatio(int64_t expectedCount, FillFn&& fill)
{
    TORCH_CHECK(expectedCount > 0 && expectedCount <= tk::kMaxCompressRatios,
        "number of compression ratios must be in [1, ", tk::kMaxCompressRatios, "], got ", expectedCount);
    for (int64_t i = 0; i < expectedCount; ++i)
    {
        fill(static_cast<int32_t>(i));
    }
    return static_cast<int32_t>(expectedCount);
}

void checkPerRatioList(
    std::vector<th::Tensor> const& list, int64_t numRatios, char const* name, int64_t minElems, int device)
{
    TORCH_CHECK(static_cast<int64_t>(list.size()) == numRatios, name,
        " must have one entry per compression ratio (expected ", numRatios, ", got ", list.size(), ")");
    for (auto const& t : list)
    {
        checkCudaContiguousTensor(t, name, device);
        TORCH_CHECK(t.numel() >= minElems, name, " must hold at least ", minElems, " elements, got ", t.numel());
    }
}
} // namespace

void deepseekV4ComputePerRatioKvLens(th::Tensor const& kvLens, th::Tensor const& cachedTokens,
    std::vector<int64_t> const& ratios, std::vector<th::Tensor> const& compressedKvLens,
    std::vector<th::Tensor> const& pastKvLens, std::vector<th::Tensor> const& newCompKvLens,
    std::vector<th::Tensor> const& cuNewCompKv)
{
    int const device = kvLens.get_device();
    checkCudaContiguousTensor(kvLens, "kv_lens", device);
    checkCudaContiguousTensor(cachedTokens, "cached_tokens", device);
    checkInt32Tensor(kvLens, "kv_lens");
    checkInt32Tensor(cachedTokens, "cached_tokens");
    int32_t const batchSize = checkedInt32Size(kvLens.size(0), "batch_size");
    TORCH_CHECK(cachedTokens.size(0) >= batchSize, "cached_tokens shorter than kv_lens");
    TORCH_CHECK(batchSize <= tk::kMaxScanBatch, "batch_size ", batchSize, " exceeds the single-block scan bound ",
        tk::kMaxScanBatch);

    int64_t const numRatios = static_cast<int64_t>(ratios.size());
    checkPerRatioList(compressedKvLens, numRatios, "compressed_kv_lens", batchSize, device);
    checkPerRatioList(pastKvLens, numRatios, "past_kv_lens", batchSize, device);
    checkPerRatioList(newCompKvLens, numRatios, "new_comp_kv_lens", batchSize, device);
    checkPerRatioList(cuNewCompKv, numRatios, "cu_new_comp_kv", batchSize + 1, device);

    tk::PerRatioKvLensParams params{};
    int32_t const numRatios32 = forEachRatio(numRatios,
        [&](int32_t i)
        {
            TORCH_CHECK(ratios[i] > 0, "compression ratio must be positive");
            params.ratios[i] = static_cast<int32_t>(ratios[i]);
            params.compressedKvLens[i] = compressedKvLens[i].data_ptr<int32_t>();
            params.pastKvLens[i] = pastKvLens[i].data_ptr<int32_t>();
            params.newCompKvLens[i] = newCompKvLens[i].data_ptr<int32_t>();
            params.cuNewCompKv[i] = cuNewCompKv[i].data_ptr<int32_t>();
        });

    c10::cuda::CUDAGuard const deviceGuard(kvLens.device());
    tk::invokeDeepseekV4ComputePerRatioKvLens(kvLens.data_ptr<int32_t>(), cachedTokens.data_ptr<int32_t>(), params,
        numRatios32, batchSize, at::cuda::getCurrentCUDAStream(device));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

void deepseekV4ComputeCompressedMask(std::vector<th::Tensor> const& newCompKvLens,
    std::vector<th::Tensor> const& cuNewCompKv, std::vector<th::Tensor> const& mask,
    std::vector<int64_t> const& totalTokens, int64_t batchSizeIn)
{
    TORCH_CHECK(!mask.empty(), "compressed_mask list must not be empty");
    int const device = mask[0].get_device();
    int32_t const batchSize = checkedInt32Size(batchSizeIn, "batch_size");
    TORCH_CHECK(batchSize > 0, "batch_size must be positive");

    int64_t const numRatios = static_cast<int64_t>(totalTokens.size());
    checkPerRatioList(newCompKvLens, numRatios, "new_comp_kv_lens", batchSize, device);
    checkPerRatioList(cuNewCompKv, numRatios, "cu_new_comp_kv", batchSize + 1, device);
    TORCH_CHECK(
        static_cast<int64_t>(mask.size()) == numRatios, "compressed_mask must have one entry per compression ratio");

    tk::CompressedMaskParams params{};
    int32_t maxTotal = 0;
    int32_t const numRatios32 = forEachRatio(numRatios,
        [&](int32_t i)
        {
            int32_t const total = checkedInt32Size(totalTokens[i], "total_compressed_tokens");
            TORCH_CHECK(total >= 0, "total_compressed_tokens must be non-negative");
            checkCudaContiguousTensor(mask[i], "compressed_mask", device);
            TORCH_CHECK(mask[i].scalar_type() == th::kBool, "compressed_mask must be bool");
            TORCH_CHECK(
                mask[i].numel() >= total, "compressed_mask buffer too small: need ", total, ", have ", mask[i].numel());
            params.newCompKvLens[i] = newCompKvLens[i].data_ptr<int32_t>();
            params.cuNewCompKv[i] = cuNewCompKv[i].data_ptr<int32_t>();
            params.mask[i] = mask[i].data_ptr<bool>();
            params.totalTokens[i] = total;
            maxTotal = std::max(maxTotal, total);
        });

    c10::cuda::CUDAGuard const deviceGuard(mask[0].device());
    tk::invokeDeepseekV4ComputeCompressedMask(
        params, maxTotal, numRatios32, batchSize, at::cuda::getCurrentCUDAStream(device));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// Shared by the ctx and gen position-id entry points; `isGen` selects the
// kernel and whether `offsets` participates.
namespace
{
void computeCompressedPositionIdsImpl(std::vector<th::Tensor> const& pastKvLens,
    std::vector<th::Tensor> const& cuNewCompKv, std::vector<th::Tensor> const& positionIds,
    std::vector<int64_t> const& ratios, std::vector<int64_t> const& counts, std::vector<int64_t> const& offsets,
    int64_t numContextsIn, int64_t batchSizeIn, bool isGen)
{
    TORCH_CHECK(!positionIds.empty(), "compressed_position_ids list must not be empty");
    int const device = positionIds[0].get_device();
    int32_t const numContexts = checkedInt32Size(numContextsIn, "num_contexts");
    int32_t const batchSize = checkedInt32Size(batchSizeIn, "batch_size");
    TORCH_CHECK(
        batchSize > 0 && numContexts >= 0 && numContexts <= batchSize, "num_contexts must be within [0, batch_size]");

    int64_t const numRatios = static_cast<int64_t>(ratios.size());
    TORCH_CHECK(static_cast<int64_t>(counts.size()) == numRatios && static_cast<int64_t>(offsets.size()) == numRatios,
        "counts and offsets must have one entry per compression ratio");
    checkPerRatioList(pastKvLens, numRatios, "past_kv_lens", batchSize, device);
    checkPerRatioList(cuNewCompKv, numRatios, "cu_new_comp_kv", batchSize + 1, device);
    TORCH_CHECK(static_cast<int64_t>(positionIds.size()) == numRatios,
        "compressed_position_ids must have one entry per compression ratio");

    tk::CompressedPositionIdsParams params{};
    int32_t maxCount = 0;
    int32_t const numRatios32 = forEachRatio(numRatios,
        [&](int32_t i)
        {
            int32_t const count = checkedInt32Size(counts[i], "count");
            int32_t const offset = checkedInt32Size(offsets[i], "offset");
            TORCH_CHECK(count >= 0 && offset >= 0, "count and offset must be non-negative");
            checkCudaContiguousTensor(positionIds[i], "compressed_position_ids", device);
            checkInt32Tensor(positionIds[i], "compressed_position_ids");
            TORCH_CHECK(positionIds[i].numel() >= static_cast<int64_t>(offset) + count,
                "compressed_position_ids buffer too small: need ", offset + count, ", have ", positionIds[i].numel());
            TORCH_CHECK(ratios[i] > 0, "compression ratio must be positive");
            params.pastKvLens[i] = pastKvLens[i].data_ptr<int32_t>();
            params.cuNewCompKv[i] = cuNewCompKv[i].data_ptr<int32_t>();
            params.positionIds[i] = positionIds[i].data_ptr<int32_t>();
            params.ratios[i] = static_cast<int32_t>(ratios[i]);
            params.counts[i] = count;
            params.offsets[i] = offset;
            maxCount = std::max(maxCount, count);
        });

    c10::cuda::CUDAGuard const deviceGuard(positionIds[0].device());
    auto stream = at::cuda::getCurrentCUDAStream(device);
    if (isGen)
    {
        tk::invokeDeepseekV4ComputeGenCompressedPositionIds(
            params, maxCount, numRatios32, numContexts, batchSize, stream);
    }
    else
    {
        tk::invokeDeepseekV4ComputeCtxCompressedPositionIds(params, maxCount, numRatios32, numContexts, stream);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace

void deepseekV4ComputeCtxCompressedPositionIds(std::vector<th::Tensor> const& pastKvLens,
    std::vector<th::Tensor> const& cuNewCompKv, std::vector<th::Tensor> const& positionIds,
    std::vector<int64_t> const& ratios, std::vector<int64_t> const& counts, int64_t numContexts)
{
    std::vector<int64_t> const zeroOffsets(ratios.size(), 0);
    computeCompressedPositionIdsImpl(pastKvLens, cuNewCompKv, positionIds, ratios, counts, zeroOffsets, numContexts,
        /*batchSize=*/numContexts, /*isGen=*/false);
}

void deepseekV4ComputeGenCompressedPositionIds(std::vector<th::Tensor> const& pastKvLens,
    std::vector<th::Tensor> const& cuNewCompKv, std::vector<th::Tensor> const& positionIds,
    std::vector<int64_t> const& ratios, std::vector<int64_t> const& counts, std::vector<int64_t> const& offsets,
    int64_t numContexts, int64_t batchSize)
{
    computeCompressedPositionIdsImpl(
        pastKvLens, cuNewCompKv, positionIds, ratios, counts, offsets, numContexts, batchSize, /*isGen=*/true);
}

void computeTokenPositions(th::Tensor const& seqLens, std::optional<th::Tensor> const& cachedTokens,
    th::Tensor const& cuSeqLens, th::Tensor const& reqIdxPerToken, std::optional<th::Tensor> const& tokenPositions,
    int64_t numTokensIn, bool computeCuSeqLens)
{
    int const device = cuSeqLens.get_device();
    checkCudaContiguousTensor(seqLens, "seq_lens", device);
    checkCudaContiguousTensor(cuSeqLens, "cu_seq_lens", device);
    checkCudaContiguousTensor(reqIdxPerToken, "req_idx_per_token", device);
    checkInt32Tensor(seqLens, "seq_lens");
    checkInt32Tensor(cuSeqLens, "cu_seq_lens");
    checkInt32Tensor(reqIdxPerToken, "req_idx_per_token");

    int32_t const batchSize = checkedInt32Size(seqLens.size(0), "batch_size");
    int32_t const numTokens = checkedInt32Size(numTokensIn, "num_tokens");
    TORCH_CHECK(batchSize > 0, "batch_size must be positive");
    TORCH_CHECK(numTokens >= 0, "num_tokens must be non-negative");
    TORCH_CHECK(!computeCuSeqLens || batchSize <= tk::kMaxTokenPositionScanBatch, "batch_size ", batchSize,
        " exceeds the single-block scan bound ", tk::kMaxTokenPositionScanBatch);
    TORCH_CHECK(
        cuSeqLens.numel() >= static_cast<int64_t>(batchSize) + 1, "cu_seq_lens must hold batch_size + 1 entries");
    TORCH_CHECK(reqIdxPerToken.numel() >= numTokens, "req_idx_per_token buffer too small");

    int32_t* positionsPtr = nullptr;
    int32_t const* cachedPtr = nullptr;
    if (tokenPositions.has_value())
    {
        TORCH_CHECK(cachedTokens.has_value(), "cached_tokens is required when token_positions is requested");
        checkCudaContiguousTensor(*tokenPositions, "token_positions", device);
        checkCudaContiguousTensor(*cachedTokens, "cached_tokens", device);
        checkInt32Tensor(*tokenPositions, "token_positions");
        checkInt32Tensor(*cachedTokens, "cached_tokens");
        TORCH_CHECK(tokenPositions->numel() >= numTokens, "token_positions buffer too small");
        TORCH_CHECK(cachedTokens->size(0) >= batchSize, "cached_tokens shorter than seq_lens");
        positionsPtr = tokenPositions->data_ptr<int32_t>();
        cachedPtr = cachedTokens->data_ptr<int32_t>();
    }

    c10::cuda::CUDAGuard const deviceGuard(cuSeqLens.device());
    tk::invokeComputeTokenPositions(seqLens.data_ptr<int32_t>(), cachedPtr, cuSeqLens.data_ptr<int32_t>(),
        reqIdxPerToken.data_ptr<int32_t>(), positionsPtr, batchSize, numTokens, computeCuSeqLens,
        at::cuda::getCurrentCUDAStream(device));
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
        "compute_shared_block_table(Tensor block_offsets, Tensor copy_idx, int pool_id, int scale, "
        "Tensor! output) -> ()");
    m.def(
        "deepseek_v4_compute_per_ratio_kv_lens(Tensor kv_lens, Tensor cached_tokens, int[] ratios, "
        "Tensor(a!)[] compressed_kv_lens, Tensor(b!)[] past_kv_lens, Tensor(c!)[] new_comp_kv_lens, "
        "Tensor(d!)[] cu_new_comp_kv) -> ()");
    m.def(
        "deepseek_v4_compute_compressed_mask(Tensor[] new_comp_kv_lens, Tensor[] cu_new_comp_kv, "
        "Tensor(a!)[] compressed_mask, int[] total_tokens, int batch_size) -> ()");
    m.def(
        "deepseek_v4_compute_ctx_compressed_position_ids(Tensor[] past_kv_lens, "
        "Tensor[] cu_new_comp_kv, Tensor(a!)[] compressed_position_ids, int[] ratios, "
        "int[] counts, int num_contexts) -> ()");
    m.def(
        "deepseek_v4_compute_gen_compressed_position_ids(Tensor[] past_kv_lens, "
        "Tensor[] cu_new_comp_kv, Tensor(a!)[] compressed_position_ids, int[] ratios, "
        "int[] counts, int[] offsets, int num_contexts, int batch_size) -> ()");
    m.def(
        "compute_token_positions(Tensor seq_lens, Tensor? cached_tokens, "
        "Tensor(a!) cu_seq_lens, Tensor(b!) req_idx_per_token, Tensor(c!)? token_positions, "
        "int num_tokens, bool compute_cu_seq_lens) -> ()");
    m.def(
        "deepseek_v4_compute_indices(Tensor token_positions, int window_size, int max_compressed_indices, "
        "int sparse_mla_topk, Tensor(a!) swa_local_indices, Tensor(b!) compressed_local_indices, "
        "Tensor(c!)? topk_lens_ratio1, Tensor(d!)? topk_lens_ratio4, Tensor(e!)? topk_lens_ratio128) -> ()");
    m.def(
        "deepseek_v4_compute_sliding_block_tables_with_scratch(Tensor block_offsets, Tensor copy_idx, "
        "Tensor pool_ids, Tensor valid_pool, Tensor scales, Tensor layer_offsets, Tensor scratch_pages, "
        "Tensor scratch_begs, Tensor scratch_ends, Tensor scratch_slots, Tensor num_contexts, "
        "Tensor(a!) output) -> ()");
}

TORCH_LIBRARY_IMPL(trtllm, CUDA, m)
{
    m.impl("deepseek_v4_compute_sliding_block_tables", &tensorrt_llm::torch_ext::deepseekV4ComputeSlidingBlockTables);
    m.impl("compute_shared_block_table", &tensorrt_llm::torch_ext::computeSharedBlockTable);
    m.impl("deepseek_v4_compute_per_ratio_kv_lens", &tensorrt_llm::torch_ext::deepseekV4ComputePerRatioKvLens);
    m.impl("deepseek_v4_compute_compressed_mask", &tensorrt_llm::torch_ext::deepseekV4ComputeCompressedMask);
    m.impl("deepseek_v4_compute_ctx_compressed_position_ids",
        &tensorrt_llm::torch_ext::deepseekV4ComputeCtxCompressedPositionIds);
    m.impl("deepseek_v4_compute_gen_compressed_position_ids",
        &tensorrt_llm::torch_ext::deepseekV4ComputeGenCompressedPositionIds);
    m.impl("compute_token_positions", &tensorrt_llm::torch_ext::computeTokenPositions);
    m.impl("deepseek_v4_compute_indices", &tensorrt_llm::torch_ext::deepseekV4ComputeIndices);
    m.impl("deepseek_v4_compute_sliding_block_tables_with_scratch",
        &tensorrt_llm::torch_ext::deepseekV4ComputeSlidingBlockTablesWithScratch);
}
