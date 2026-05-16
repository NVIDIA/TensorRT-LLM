#pragma once
#include <cstdint>
#include <cuda_runtime.h>

inline cudaLaunchConfig_t makeLaunchConfig(dim3 const& gridDim, dim3 const& ctaDim, size_t dynShmBytes,
    cudaStream_t stream, bool usePDL, dim3 const& clusterDim = dim3{1, 1, 1})
{
    static thread_local cudaLaunchAttribute attrs[2];
    uint32_t numAttrs = 0;

    attrs[numAttrs].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[numAttrs].val.programmaticStreamSerializationAllowed = (usePDL ? 1 : 0);
    numAttrs++;

    if (clusterDim.x * clusterDim.y * clusterDim.z > 1)
    {
        attrs[numAttrs].id = cudaLaunchAttributeClusterDimension;
        attrs[numAttrs].val.clusterDim.x = clusterDim.x;
        attrs[numAttrs].val.clusterDim.y = clusterDim.y;
        attrs[numAttrs].val.clusterDim.z = clusterDim.z;
        numAttrs++;
    }

    cudaLaunchConfig_t cfg{gridDim, ctaDim, dynShmBytes, stream, attrs, numAttrs};
    return cfg;
}
