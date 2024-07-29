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

#include "tensorrt_llm/common/cudaTypeUtils.cuh"
#include "tensorrt_llm/kernels/lookupKernels.h"

using namespace tensorrt_llm::common;

namespace tensorrt_llm
{
namespace kernels
{
/* When running with multiple GPUs, we split the embedding lookup table across multiple GPUs to save the memory
requirements of embedding lookup table ([vocab_size, hidden]). This operation is equivalent to the single GPU version of
embedding() (i.e.add_gather() operation in TensorRT). As only a portion of embedding lookup table
([ceil(vocab_size/world_size), hidden]) is stored in each GPU and the value range of input IDs is [0, vocab_size]. The
add_gather() operation in TensorRT cannot get the correct results. So, we need to write a plugin to add an offset to
input IDs and get the correct results.

 * Input: Input IDs (input[token_num])
   Input: Embedding Lookup Table (weight[ceil(vocab_size/world_size), hidden])
   Output: weight[input[idx]-offset,hidden]

 * The total thread number equals to token_num*hidden
 *
 * If the input ids is out of range it writes zero, otherwise it writes the correct embedding result.
 */
template <typename Tout, typename Tin, typename Idx>
__global__ void lookup_kernel(Tout* output, Idx const* input, Tin const* weight, int64_t const token_num,
    Idx const offset, Idx const size, Idx const n_embed, Tout const* perTokenScales)
{
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < token_num * n_embed;
         index += blockDim.x * gridDim.x)
    {
        int64_t const word_index = input[index / n_embed] - offset;
        Idx const col_index = index % n_embed;
        Tout embedding;
        if (word_index < 0 || word_index >= size)
        {
            embedding = Tout(0.f);
        }
        else
        {
            embedding = (Tout) weight[word_index * n_embed + col_index];
            if (perTokenScales != nullptr)
            {
                embedding *= perTokenScales[word_index];
            }
        }
        output[index] = embedding;
    } // end for index
}

template <typename Tout, typename Tin, typename Idx>
void invokeLookUp(Tout* out, Idx const* input, Tin const* weight, int64_t const token_num, Idx const offset,
    Idx const size, Idx const n_embed, Tout const* perTokenScales, cudaStream_t stream)
{
    int64_t constexpr max_block_num = 65536;
    Idx constexpr max_block_size = 512;
    dim3 grid(min(token_num, max_block_num));
    dim3 block(min(n_embed, max_block_size));
    lookup_kernel<Tout, Tin, Idx>
        <<<grid, block, 0, stream>>>(out, input, weight, token_num, offset, size, n_embed, perTokenScales);
}

#define INSTANTIATE_LOOK_UP(Tout, Tin, Idx)                                                                            \
    template void invokeLookUp<Tout, Tin, Idx>(Tout * out, Idx const* input, Tin const* weight,                        \
        int64_t const token_num, Idx const offset, Idx const size, Idx const n_embed, Tout const* perTokenScales,      \
        cudaStream_t stream)

INSTANTIATE_LOOK_UP(float, float, int);
INSTANTIATE_LOOK_UP(float, int8_t, int);
INSTANTIATE_LOOK_UP(half, half, int);
INSTANTIATE_LOOK_UP(half, int8_t, int);

#ifdef ENABLE_BF16
INSTANTIATE_LOOK_UP(__nv_bfloat16, __nv_bfloat16, int);
INSTANTIATE_LOOK_UP(__nv_bfloat16, int8_t, int);
#endif

} // namespace kernels
} // namespace tensorrt_llm
