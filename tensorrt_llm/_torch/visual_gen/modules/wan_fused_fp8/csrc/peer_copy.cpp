// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Copy-engine 2D copies and stream-ordered flags over peer-mapped memory.

#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace
{

cudaStream_t asStream(int64_t stream)
{
    return reinterpret_cast<cudaStream_t>(stream);
}

} // namespace

// Device-to-device 2D copy on the copy engines; addresses may be peer-mapped.
void copy2d(int64_t dst, int64_t dpitch, int64_t src, int64_t spitch, int64_t width, int64_t height, int64_t stream)
{
    auto err = cudaMemcpy2DAsync(reinterpret_cast<void*>(dst), dpitch, reinterpret_cast<void const*>(src), spitch,
        width, height, cudaMemcpyDeviceToDevice, asStream(stream));
    TORCH_CHECK(err == cudaSuccess, "cudaMemcpy2DAsync failed: ", cudaGetErrorString(err));
}

// Writes value to a 32-bit flag after prior work on the stream.
void signal(int64_t addr, int64_t value, int64_t stream)
{
    auto err = cuStreamWriteValue32(reinterpret_cast<CUstream>(stream), static_cast<CUdeviceptr>(addr),
        static_cast<cuuint32_t>(value), CU_STREAM_WRITE_VALUE_DEFAULT);
    TORCH_CHECK(err == CUDA_SUCCESS, "cuStreamWriteValue32 failed: ", static_cast<int>(err));
}

// Blocks the stream until the 32-bit flag is >= value.
void wait_geq(int64_t addr, int64_t value, int64_t stream)
{
    auto err = cuStreamWaitValue32(reinterpret_cast<CUstream>(stream), static_cast<CUdeviceptr>(addr),
        static_cast<cuuint32_t>(value), CU_STREAM_WAIT_VALUE_GEQ);
    TORCH_CHECK(err == CUDA_SUCCESS, "cuStreamWaitValue32 failed: ", static_cast<int>(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("copy2d", &copy2d, "Copy-engine 2D device copy");
    m.def("signal", &signal, "Stream-ordered 32-bit flag write");
    m.def("wait_geq", &wait_geq, "Stream wait until 32-bit flag >= value");
}
