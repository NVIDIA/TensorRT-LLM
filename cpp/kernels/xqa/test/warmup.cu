#include "../utils.h"
#include <cstdint>
#include <cuda_runtime.h>

__global__ void kernel_warmup(uint64_t cycles)
{
    uint64_t const tic = clock64();
    while (tic + cycles < clock64())
    {
    }
}

void warmup(cudaDeviceProp const& prop, float ms, cudaStream_t stream = nullptr)
{
#if CUDA_VERSION >= 13000
    int device;
    checkCuda(cudaGetDevice(&device));
    int clockRateKHz;
    checkCuda(cudaDeviceGetAttribute(&clockRateKHz, cudaDevAttrClockRate, device));
    uint64_t const nbCycles = std::round(clockRateKHz * ms); // clockRate is in kHz
#else
    uint64_t const nbCycles = std::round(prop.clockRate * ms); // clockRate is in kHz
#endif
    kernel_warmup<<<16, 128, 0, stream>>>(nbCycles);
    checkCuda(cudaGetLastError());
}
