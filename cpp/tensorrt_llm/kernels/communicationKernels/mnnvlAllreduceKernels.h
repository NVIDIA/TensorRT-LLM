/*
 * Copyright (c) 2022-2024, NVIDIA CORPORATION.  All rights reserved.
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
#ifndef TRTLLM_MNNVL_ALLREDUCE_KERNELS_H
#define TRTLLM_MNNVL_ALLREDUCE_KERNELS_H

#include "tensorrt_llm/common/config.h"
#include <NvInferRuntime.h>
#include <cstdint>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::mnnvl
{

/**
 * \brief Parameters for MNNVL (Multi-Node NVLink) AllReduce fusion operations.
 *
 * \note This struct is used by both oneshotAllreduceFusionOp() and twoshotAllreduceFusionOp()
 *
 * \see oneshotAllreduceFusionOp
 * \see twoshotAllreduceFusionOp
 */
struct AllReduceFusionParams
{
    //! \name Environmental and Auxiliary Data
    //! @{

    int nRanks;               //!< Total number of participating ranks in the AllReduce operation
    int rank;                 //!< Current rank ID
    nvinfer1::DataType dType; //!< Data type of the tensors (e.g., FP16, BF16, FP32)
    int numTokens;            //!< Number of tokens in the input tensor
    int tokenDim;             //!< Hidden Dimension
    void** bufferPtrsDev;     //!< Unicast Device pointers to communication buffers for each rank
    void* bufferPtrLocal;     //!< Local buffer pointer for temporary storage (i.e., bufferPtrsDev[rank])
    void* multicastPtr;       //!< Multicast buffer pointer.
    uint32_t* bufferFlags;    //!< Synchronization flags for coordinating communication phases
    bool rmsNormFusion;       //!< Whether to fuse RMS normalization with the AllReduce operation

    //! @}

    //! \name Input and Output Data
    //! @{

    void const* input;      //!< Input tensor to be reduced across all ranks
    void const* residualIn; //!< Residual input tensor for skip connections (used when rmsnormFusion=true)
    void const* gamma;      //!< Gamma parameters for RMS normalization (used when rmsnormFusion=true)
    double epsilon;         //!< Epsilon value for RMS normalization numerical stability (used when rmsnormFusion=true)

    void* residualOut;      //!< Output tensor for residual connection result (used when rmsnormFusion=true)
    void* output;           //!< Final output tensor containing the AllReduce result
    cudaStream_t stream;    //!< CUDA stream for asynchronous kernel execution

    //! @}
};

void oneshotAllreduceFusionOp(AllReduceFusionParams const& params);
void twoshotAllreduceFusionOp(AllReduceFusionParams const& params);
} // namespace kernels::mnnvl

TRTLLM_NAMESPACE_END
#endif // TRTLLM_MNNVL_ALLREDUCE_KERNELS_H
