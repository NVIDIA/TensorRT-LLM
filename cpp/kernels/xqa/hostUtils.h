#pragma once
#include <cuda_runtime.h>

inline cudaLaunchConfig_t makeLaunchConfig(
    dim3 const& gridDim, dim3 const& ctaDim, size_t dynShmBytes, cudaStream_t stream, bool useFDL)
{
    static cudaLaunchAttribute fdlAttr;
    fdlAttr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    fdlAttr.val.programmaticStreamSerializationAllowed = (useFDL ? 1 : 0);

    cudaLaunchConfig_t cfg{gridDim, ctaDim, dynShmBytes, stream, &fdlAttr, 1};
    return cfg;
}
