/*
 * Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

#include "prefetch.h"

#include "tensorrt_llm/common/cudaUtils.h"
#include <cute/tensor.hpp>

namespace tensorrt_llm
{
namespace kernels
{

// assume load a [M, K] row major weight matrix
// how to sync with DMA warp is tricky
template <typename T, int CTA_M, int CTA_K, class TmaLoad, class GmemTensor>
__global__ void cute_tma_prefetch_kernel(__grid_constant__ TmaLoad const tma_load, GmemTensor gmem_tensor, int K,
    int delay_start, int throttle_time, int throttle_mode)
{
    using namespace cute;

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    // if the kernel allocates tmem (min 32 columns), the compiler will mark the binary
    // as using tmem, then the kernel launch will acquire all vrc so that we have 1 cta/SM
    // then we relinquish which clears the vrc, and we deallocate the tmem
    // so that the next kernel can launch
    __shared__ uint32_t tmem_base_ptr;
    // Tmem allocator
    cute::TMEM::Allocator1Sm tmem_allocator{};
    tmem_allocator.allocate(32, &tmem_base_ptr);
    tmem_allocator.release_allocation_lock(); // relinquish
    tmem_allocator.free(tmem_base_ptr, 32);

    if (threadIdx.x == 0)
    {
        auto gmem_tensor_coord = tma_load.get_tma_tensor(shape(gmem_tensor));
        auto tma_load_per_cta = tma_load.get_slice(0);

        auto target_clock = clock64() + delay_start;
        while (clock64() < target_clock)
            ;

        for (int k = 0; k < K / CTA_K; k++)
        {
            auto gmem_tensor_coord_cta
                = local_tile(gmem_tensor_coord, Tile<Int<CTA_M>, Int<CTA_K>>{}, make_coord(blockIdx.x, k));

            prefetch(tma_load, tma_load_per_cta.partition_S(gmem_tensor_coord_cta));

            if (throttle_mode == 1)
            {
                target_clock = clock64() + throttle_time;
            }
            else if (throttle_mode == 2)
            {
                target_clock += throttle_time;
            }
            else
            {
                __trap();
            }
            while (clock64() < target_clock)
                ;
        }
    }
#else
    if (blockIdx.x == 0 && threadIdx.x == 0)
    {
        printf("Error: TMA prefetch is only supported on compute_90 and above\n");
    }
    __trap();
#endif
}

template <typename T, int CTA_M, int CTA_K>
void cute_host_prefetch(T const* data, int M, int K, int strideM, int delay_start, int throttle_time, int throttle_mode,
    bool pdl, cudaStream_t stream)
{
    using namespace cute;
    using tensorrt_llm::common::check;

    // create the GMEM tensor, row major
    auto gmem_layout = make_layout(make_shape(M, K), make_stride(strideM, 1));
    auto gmem_tensor = make_tensor(make_gmem_ptr(data), gmem_layout);

    // create the SMEM layout, row major
    // smem_layout need to use static integer
    // use dynamic integer will cause compilation error
    auto smem_layout = make_layout(make_shape(Int<CTA_M>{}, Int<CTA_K>{}), make_stride(Int<CTA_K>{}, _1{}));

    // create the TMA object
    auto tma_load = make_tma_copy(SM90_TMA_LOAD{}, gmem_tensor, smem_layout);

    // invoke the kernel
    // each CTA responsible for a range of M, and all kblock associated with it
    cudaLaunchConfig_t config{};
    cudaLaunchAttribute attrs[1];
    config.gridDim = dim3{(uint32_t) (M / CTA_M), 1, 1};
    config.blockDim = 32;
    config.stream = stream;
    if (pdl)
    {
        attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attrs[0].val.programmaticStreamSerializationAllowed = 1;
        config.attrs = attrs;
        config.numAttrs = 1;
    }

    auto* kernel_instance = &cute_tma_prefetch_kernel<T, CTA_M, CTA_K, decltype(tma_load), decltype(gmem_tensor)>;
    // default carveout is 32KB, which mismatches with gemm, so will trigger smem reconfiguration, which requires
    // draining SM, i.e. no overlapping
    check_cuda_error(cudaFuncSetAttribute(kernel_instance, cudaFuncAttributePreferredSharedMemoryCarveout, 100));

    check_cuda_error(cudaLaunchKernelEx(
        &config, kernel_instance, tma_load, gmem_tensor, K, delay_start, throttle_time, throttle_mode));
}

#define INSTANTIATE_CUTE_HOST_PREFETCH(T, CTA_M, CTA_K)                                                                \
    template void cute_host_prefetch<T, CTA_M, CTA_K>(T const*, int, int, int, int, int, int, bool, cudaStream_t);

// In the same order as in 3rdparty/cutlass/include/cute/arch/copy_sm90_desc.hpp
INSTANTIATE_CUTE_HOST_PREFETCH(uint8_t, 64, 128);
INSTANTIATE_CUTE_HOST_PREFETCH(cute::float_e4m3_t, 64, 128);
INSTANTIATE_CUTE_HOST_PREFETCH(cute::float_e5m2_t, 64, 128);
INSTANTIATE_CUTE_HOST_PREFETCH(cute::half_t, 64, 128);
INSTANTIATE_CUTE_HOST_PREFETCH(float, 64, 128);
INSTANTIATE_CUTE_HOST_PREFETCH(cute::bfloat16_t, 64, 128);

} // namespace kernels
} // namespace tensorrt_llm
