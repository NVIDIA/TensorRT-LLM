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

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "tinygemm2_kernel.cuh"

void launch_tinygemm2(__nv_bfloat16* gA, __nv_bfloat16* gB, __nv_bfloat16* gC, __nv_bfloat16* bias, int batch_size,
    int output_features, int input_features, cudaStream_t stream)
{

    static int const WARP_TILE_M = 16;
    static int const TILE_M = WARP_TILE_M;
    static int const TILE_N = 8;
    static int const TILE_K = 64;
    static int const STAGES = 16;
    static int const STAGE_UNROLL = 4;
    static bool const PROFILE = false;

    CUtensorMap weight_map{};
    CUtensorMap activation_map{};

    constexpr uint32_t rank = 2;
    uint64_t size[rank] = {(uint64_t) input_features, (uint64_t) output_features};
    uint64_t stride[rank - 1] = {input_features * sizeof(__nv_bfloat16)};
    uint32_t box_size[rank] = {TILE_K, TILE_M};
    uint32_t elem_stride[rank] = {1, 1};

    CUresult res = cuTensorMapEncodeTiled(&weight_map, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, gB,
        size, stride, box_size, elem_stride, CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B, CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    assert(res == 0);

    size[1] = batch_size;
    box_size[1] = TILE_N;

    res = cuTensorMapEncodeTiled(&activation_map, CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, rank, gA, size,
        stride, box_size, elem_stride, CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B, CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    assert(res == 0);

    int smem_size
        = STAGES * STAGE_UNROLL * (TILE_M * TILE_K * sizeof(__nv_bfloat16) + TILE_N * TILE_K * sizeof(__nv_bfloat16));

    gpuErrChk(cudaFuncSetAttribute(kernel<WARP_TILE_M, TILE_M, TILE_N, TILE_K, STAGES, STAGE_UNROLL, PROFILE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    int tiles_m = (output_features + TILE_M - 1) / TILE_M;
    int tiles_n = (batch_size + TILE_N - 1) / TILE_N;

    dim3 grid(tiles_m, tiles_n);

    dim3 block(384);

    cudaLaunchConfig_t config;
    cudaLaunchAttribute attrs[1];
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = smem_size;
    config.stream = stream;
    config.attrs = attrs;
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;
    config.numAttrs = 1;

    cudaLaunchKernelEx(&config, &kernel<WARP_TILE_M, TILE_M, TILE_N, TILE_K, STAGES, STAGE_UNROLL, PROFILE>, gC, gA, gB,
        bias, output_features, batch_size, input_features, weight_map, activation_map, nullptr);
}

torch::Tensor tinygemm2_cuda_forward(torch::Tensor input, torch::Tensor weight, torch::Tensor bias)
{

    auto const batch_size = input.size(0);
    auto const input_dim = input.size(1);
    auto const output_dim = weight.size(0);

    auto output = torch::empty({batch_size, output_dim}, input.options());

    auto stream = at::cuda::getCurrentCUDAStream();

    if (input.scalar_type() == at::ScalarType::BFloat16)
    {

        launch_tinygemm2((__nv_bfloat16*) input.data_ptr(), (__nv_bfloat16*) weight.data_ptr(),
            (__nv_bfloat16*) output.data_ptr(), (__nv_bfloat16*) bias.data_ptr(), batch_size, output_dim, input_dim,
            stream);
    }
    else
    {
        assert(false);
    }
    return output;
}
