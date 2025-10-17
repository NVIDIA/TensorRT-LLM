// tests/kernels/testInvokeConversion.cpp
// This file tests the invokeConversion kernel function which converts between different floating-point formats

// Include TensorRT-LLM kernel headers that contain the invokeConversion function
#include "tensorrt_llm/kernels/unfusedAttentionKernels.h"
// Include BufferManager for managing GPU/CPU memory allocations and transfers
#include "tensorrt_llm/runtime/bufferManager.h"
// Include CudaStream for managing CUDA stream operations
#include "tensorrt_llm/runtime/cudaStream.h"
// Standard library for smart pointers
#include <memory>
// Google Test framework for unit testing
#include <gtest/gtest.h>
// CUDA header for FP16 (half-precision) type and operations
#include <cuda_fp16.h>          // half
// CUDA header for BF16 (bfloat16) type and operations
#include <cuda_bf16.h>          // __nv_bfloat16
// CUDA header for FP8 (8-bit floating point) type and operations
#include <cuda_fp8.h>           // __nv_fp8_e4m3

// Use tensorrt_llm namespace to avoid prefixing all TensorRT-LLM types
using namespace tensorrt_llm;
// Create shorter alias 'tr' for the runtime namespace
namespace tr = tensorrt_llm::runtime;

/* ---------- helpers ----------------------------------------------------- */
// Helper function to convert GPU buffer of type T to a CPU vector of floats
// This is useful for validating GPU computation results on the CPU
template <typename T>
std::vector<float> toFloatCPU(tr::IBuffer const& gpu, size_t n, tr::BufferManager& mgr)
{
    // Copy the GPU buffer to CPU memory (host memory)
    auto host = mgr.copyFrom(gpu, tr::MemoryType::kCPU);
    // Wait for the copy operation to complete before accessing the data
    mgr.getStream().synchronize();
    // Allocate a vector to store the converted float values
    std::vector<float> out(n);
    // Loop through each element, cast from type T to float, and store in output vector
    for (size_t i = 0; i < n; ++i) { out[i] = static_cast<float>(reinterpret_cast<T const*>(host->data())[i]); }
    // Return the converted vector
    return out;
}

/* ---------- TEST 1 : FP16 -> FP8 --------------------------------------- */
// Test case: Convert FP16 (half precision) data to FP8 (8-bit floating point)
TEST(InvokeConversion, FP16toFP8)
{
    // Define the number of elements to test
    const int N = 1024;
    // Create a shared pointer to a CUDA stream for asynchronous GPU operations
    auto streamPtr = std::make_shared<tr::CudaStream>();
    // Initialize BufferManager with the stream to manage memory operations
    tr::BufferManager mgr{streamPtr};

    /* prepare input (FP16) ------------------------------------------------ */
    // Allocate a CPU vector to hold FP16 input values
    std::vector<half> h_in(N);
    // Fill the vector with test values in a small range [-8, 8] for better FP8 precision
    // Using modulo to create a repeating pattern: i % 17 gives 0-16, then subtract 8 to get -8 to 8
    for (int i = 0; i < N; ++i) h_in[i] = __float2half_rn(float(i % 17 - 8));

    // Copy the CPU input data to GPU memory, returns a buffer of half* type
    auto d_in  = mgr.copyFrom(h_in, tr::MemoryType::kGPU);
    // Allocate a GPU output buffer with FP8 data type to store conversion results
    auto d_out = mgr.gpu(tr::ITensor::makeShape({N}), nvinfer1::DataType::kFP8);

    /* conversion ---------------------------------------------------------- */
    // Call the GPU kernel to convert from FP16 (half) to FP8 (__nv_fp8_e4m3)
    // Template parameters: <OutputType, InputType>
    kernels::invokeConversion<__nv_fp8_e4m3, half>(
        reinterpret_cast<__nv_fp8_e4m3*>(d_out->data()),    // Output: FP8 buffer pointer
        reinterpret_cast<half const*>(d_in->data()),         // Input: FP16 buffer pointer
        N,                                                    // Number of elements to convert
        nullptr,                                              // Optional scale parameter (not used here)
        mgr.getStream().get());                               // CUDA stream for async execution

    // Wait for the GPU kernel to complete execution
    mgr.getStream().synchronize();

    /* round-trip back to float for comparison ----------------------------- */
    // Convert the FP8 output back to float on CPU for verification
    auto h_out = toFloatCPU<__nv_fp8_e4m3>(*d_out, N, mgr);

    // Verify that each converted value is close to the original
    for (int i = 0; i < N; ++i)
    {
        // Convert the original FP16 input to FP32 for comparison baseline
        float ref = __half2float(h_in[i]);
        // Check that the FP8 converted value is within 0.5 of the original
        // This tolerance accounts for FP8 quantization error
        EXPECT_NEAR(h_out[i], ref, 0.5f);
    }
}

/* ---------- TEST 2 : BF16 -> FP8 --------------------------------------- */
// Test case: Convert BF16 (bfloat16) data to FP8 (8-bit floating point)
TEST(InvokeConversion, BF16toFP8)
{
    // Define the number of elements to test
    const int N = 1024;
    // Create a shared pointer to a CUDA stream for asynchronous GPU operations
    auto streamPtr = std::make_shared<tr::CudaStream>();
    // Initialize BufferManager with the stream to manage memory operations
    tr::BufferManager mgr{streamPtr};

    // Allocate a CPU vector to hold BF16 input values
    std::vector<__nv_bfloat16> h_in(N);
    // Fill the vector with test values in a small range [-11, 11] for better FP8 precision
    // Using modulo to create a repeating pattern: i % 23 gives 0-22, then subtract 11 to get -11 to 11
    for (int i = 0; i < N; ++i) h_in[i] = __float2bfloat16(float(i % 23 - 11));

    // Copy the CPU input data to GPU memory, returns a buffer of __nv_bfloat16* type
    auto d_in  = mgr.copyFrom(h_in, tr::MemoryType::kGPU);
    // Allocate a GPU output buffer with FP8 data type to store conversion results
    auto d_out = mgr.gpu(tr::ITensor::makeShape({N}), nvinfer1::DataType::kFP8);

    // Call the GPU kernel to convert from BF16 (__nv_bfloat16) to FP8 (__nv_fp8_e4m3)
    // Template parameters: <OutputType, InputType>
    kernels::invokeConversion<__nv_fp8_e4m3, __nv_bfloat16>(
        reinterpret_cast<__nv_fp8_e4m3*>(d_out->data()),       // Output: FP8 buffer pointer
        reinterpret_cast<__nv_bfloat16 const*>(d_in->data()),  // Input: BF16 buffer pointer
        N,                                                      // Number of elements to convert
        nullptr,                                                // Optional scale parameter (not used here)
        mgr.getStream().get());                                 // CUDA stream for async execution

    // Wait for the GPU kernel to complete execution
    mgr.getStream().synchronize();
    // Convert the FP8 output back to float on CPU for verification
    auto h_out = toFloatCPU<__nv_fp8_e4m3>(*d_out, N, mgr);

    // Verify that each converted value is close to the original
    for (int i = 0; i < N; ++i)
    {
        // Convert the original BF16 input to FP32 for comparison baseline
        float ref = __bfloat162float(h_in[i]);
        // Check that the FP8 converted value is within 0.5 of the original
        // This tolerance accounts for FP8 quantization error
        EXPECT_NEAR(h_out[i], ref, 0.5f);
    }
}