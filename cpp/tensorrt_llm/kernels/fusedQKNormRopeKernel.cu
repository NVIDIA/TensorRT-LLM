#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to compute RoPE rotation
__host__ __device__ void applyRoPE(
    float& x, float& y,
    float cosTheta, float sinTheta
) {
    float xNew = x * cosTheta - y * sinTheta;
    float yNew = x * sinTheta + y * cosTheta;
    x = xNew;
    y = yNew;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void qwen3RopeCacheUpdateKernel(
    const __nv_bfloat16* q,  // Input matrix Q
    const __nv_bfloat16* k,  // Input matrix K
    __nv_bfloat16* outputQ,  // Output normalized Q
    __nv_fp8_e4m3** outputK, // Output normalized K in e4m3 format
    int numRowsQ,
    int numRowsK,
    int numTokensPerPage,    // Number of tokens (rows) per page
    float base = 10000.0f    // Base for RoPE computation
) {
    // The number of columns in this matrix.
    const int numCols = 128;  

    // The position of the thread in the block.
    const int warpId = threadIdx.x / 32;
    const int laneId = threadIdx.x % 32;

    // Each block handles 8 rows.
    const int rowIdx = blockIdx.x * 8 + warpId;

    // Check if this warp's row is within bounds.
    if (rowIdx >= numRowsQ + numRowsK) {
      return;
    }

    // Determine if this warp is processing Q or K
    const bool isQ = rowIdx < numRowsQ;

    // The matrix to process.
    const __nv_bfloat16* matrix = isQ ? q : k;
    // Remap the rowIdx to the correct row in the matrix.
    const int matrixRowIdx = rowIdx - (isQ ? 0 : numRowsQ);
    
    // Each thread processes 4 consecutive columns.
    const int numColsPerThread = 4;
    
    // The sum of squares of the elements loaded by this thread.
    float sumOfSquares = 0.0f;
    // The 4 elements loaded by this thread.
    float elements[numColsPerThread];
    
    // Load the 4 elements.
    int colIdx = laneId * numColsPerThread;
    if (colIdx < numCols) {
        // Load 4 bfloat16 elements using LDG.64
        uint2 data;
        data = *reinterpret_cast<const uint2*>(&matrix[matrixRowIdx * numCols + colIdx]);
        
        // Convert bfloat16 to float.
        auto vals0 = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162&>(data.x));
        auto vals1 = __bfloat1622float2(reinterpret_cast<const __nv_bfloat162&>(data.y));

        // Store in the array of elements.
        elements[0] = vals0.x;
        elements[1] = vals0.y;
        elements[2] = vals1.x;
        elements[3] = vals1.y;

        // Update the sum of squares.
        sumOfSquares += elements[0] * elements[0];
        sumOfSquares += elements[1] * elements[1];
        sumOfSquares += elements[2] * elements[2];
        sumOfSquares += elements[3] * elements[3];
    }
    
    // Reduce the sum across warp using __shfl_xor_sync.
    for (int offset = 16; offset > 0; offset /= 2) {
        sumOfSquares += __shfl_xor_sync(0xffffffff, sumOfSquares, offset);
    }
    
    // Compute RMS scale factor.
    float rms = rsqrtf(sumOfSquares / static_cast<float>(numCols));
    
    // Normalize the FP32 elements. 
    for (int ii = 0; ii < numColsPerThread; ii++) {
      elements[ii] *= rms;
    }

    // Apply RoPE to the normalized elements
    // Process elements in pairs (x,y) for rotation
    for (int ii = 0; ii < numColsPerThread; ii += 2) {
        int pos = colIdx + ii;
        if (pos + 1 < numCols) {
            // Compute rotation angle for this position
            float theta = pos * 2.0f * M_PI / base;

            // The cos/sin values are computed once per thread.
            float cosTheta = cosf(theta);
            float sinTheta = sinf(theta);
            
            // Apply rotation to the pair of elements
            applyRoPE(elements[ii], elements[ii + 1], cosTheta, sinTheta);
        }
    }
    
    // Store the rotated elements back to memory
    if (colIdx < numCols) {
        if (isQ) {
            // For Q matrix, write directly to outputQ as bfloat16
            __nv_bfloat162 vals0 = __float22bfloat162_rn(make_float2(elements[0], elements[1]));
            __nv_bfloat162 vals1 = __float22bfloat162_rn(make_float2(elements[2], elements[3]));

            uint2 data;
            data.x = *reinterpret_cast<uint32_t*>(&vals0);
            data.y = *reinterpret_cast<uint32_t*>(&vals1);
            
            uint2* outputPtr = reinterpret_cast<uint2*>(&outputQ[matrixRowIdx * numCols + colIdx]);
            outputPtr[0] = data;
        } else {
            // For K matrix, convert to e4m3 and write to the appropriate page
            int pageIdx = matrixRowIdx / numTokensPerPage;
            int pageRow = matrixRowIdx % numTokensPerPage;

            // Get the pointer to the page.
            __nv_fp8_e4m3* pageK = outputK[pageIdx];

            // Convert and store 4 elements at once using STG.32
            float4 vals = make_float4(elements[0], elements[1], elements[2], elements[3]);
            __nv_fp8x4_e4m3 fp8x4(vals);
            *reinterpret_cast<uint32_t*>(&pageK[pageRow * numCols + colIdx]) = *reinterpret_cast<uint32_t*>(&fp8x4);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to initialize a matrix with random values
void initializeMatrix(std::vector<__nv_bfloat16>& matrix, int numRows, int numCols) {
    std::mt19937 gen(1234);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    
    for (int i = 0; i < numRows * numCols; ++i) {
        float val = dist(gen);
        matrix[i] = __float2bfloat16(val);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Helper function to compute ULP difference for FP8
__host__ __device__ int computeULPDiff(float a, float b) {
    // Convert to FP8 and back to get the ULP representation
    __nv_fp8_e4m3 fp8_a(a);
    __nv_fp8_e4m3 fp8_b(b);

    float a_fp8(fp8_a);
    float b_fp8(fp8_b);
    
    // Get the integer representation of the FP8 values
    int a_int = *reinterpret_cast<int*>(&a_fp8);
    int b_int = *reinterpret_cast<int*>(&b_fp8);
    
    // Handle sign bit
    if ((a_int ^ b_int) < 0) {
        // If signs are different, handle as special case
        if (a_int < 0) {
            a_int = 0x80000000 - a_int;
        }
        if (b_int < 0) {
            b_int = 0x80000000 - b_int;
        }
    }
    
    return std::abs(a_int - b_int);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Compute golden reference on host
template<typename T>
void computeGoldenReference(
    const std::vector<__nv_bfloat16>& input,
    std::vector<T>& output,
    int numRows,
    int numCols,
    float base = 10000.0f
) {
    for (int row = 0; row < numRows; ++row) {
        // Compute sum of squares for RMS normalization
        float sumOfSquares = 0.0f;
        std::vector<float> elements(numCols);
        
        for (int col = 0; col < numCols; ++col) {
            float val = __bfloat162float(input[row * numCols + col]);
            elements[col] = val;
            sumOfSquares += val * val;
        }
        
        // Compute RMS scale factor
        float rms = std::sqrt(sumOfSquares / numCols);
        
        // Normalize elements
        for (int col = 0; col < numCols; ++col) {
            elements[col] /= rms;
        }
        
        // Apply RoPE
        for (int col = 0; col < numCols; col += 2) {
            if (col + 1 < numCols) {
                float theta = col * 2.0f * M_PI / base;
                float cosTheta = std::cos(theta);
                float sinTheta = std::sin(theta);
                applyRoPE(elements[col], elements[col + 1], cosTheta, sinTheta);
            }
        }
        
        // Store results
        for (int col = 0; col < numCols; ++col) {
            if constexpr (std::is_same_v<T, __nv_bfloat16>) {
                output[row * numCols + col] = __float2bfloat16(elements[col]);
            } else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
                output[row * numCols + col] = __nv_fp8_e4m3(elements[col]);
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Compare results with tolerance
template<typename T>
bool compareResults(
    const std::vector<T>& golden,
    const std::vector<T>& computed,
    int numRows,
    int numCols,
    float tolerance = 1e-2f
) {
    bool passed = true;
    float maxDiff = 0.0f;
    int maxULP = 0;
    int numErrors = 0;
    
    for (int i = 0; i < numRows * numCols; ++i) {
        float goldenVal;
        float computedVal;
        if constexpr (std::is_same_v<T, __nv_bfloat16>) {
            goldenVal = __bfloat162float(golden[i]);
            computedVal = __bfloat162float(computed[i]);
            float diff = std::abs(goldenVal - computedVal);
            maxDiff = std::max(maxDiff, diff);
            
            if (diff > tolerance) {
                numErrors++;
                if (numErrors <= 10) {  // Print first 10 errors
                    std::cout << "Mismatch at index " << i 
                             << ": golden=" << goldenVal 
                             << ", computed=" << computedVal 
                             << ", diff=" << diff << std::endl;
                }
                passed = false;
            }
        } else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
            goldenVal = (float) golden[i];
            computedVal = (float) computed[i];
            int ulpDiff = computeULPDiff(goldenVal, computedVal);
            maxULP = std::max(maxULP, ulpDiff);
            
            if (ulpDiff > 1) {  // Allow 1 ULP difference
                numErrors++;
                if (numErrors <= 10) {  // Print first 10 errors
                    std::cout << "Mismatch at index " << i 
                             << ": golden=" << goldenVal 
                             << ", computed=" << computedVal 
                             << ", ULP diff=" << ulpDiff << std::endl;
                }
                passed = false;
            }
        }
    }
    
    if constexpr (std::is_same_v<T, __nv_bfloat16>) {
        std::cout << "Max difference: " << maxDiff << std::endl;
    } else if constexpr (std::is_same_v<T, __nv_fp8_e4m3>) {
        std::cout << "Max ULP difference: " << maxULP << std::endl;
    }
    std::cout << "Number of errors: " << numErrors << std::endl;
    return passed;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int main() {
    // Matrix dimensions
    const int numCols = 128;
    const int numTokens = 32;
    const int numHeadsQ = 64;
    const int numRowsQ = numTokens * numHeadsQ;
    const int numHeadsK = 4;
    const int numRowsK = numTokens * numHeadsK;
    const int numTokensPerPage = 16;
    const int numPagesK = (numRowsK + numTokensPerPage - 1) / numTokensPerPage;

    // Allocate host memory
    std::vector<__nv_bfloat16> hQ(numRowsQ * numCols);
    std::vector<__nv_bfloat16> hK(numRowsK * numCols);
    std::vector<__nv_bfloat16> hOutputQ(numRowsQ * numCols);
    std::vector<__nv_fp8_e4m3> hOutputK(numPagesK * numTokensPerPage * numCols);
    
    // Golden reference outputs
    std::vector<__nv_bfloat16> hGoldenQ(numRowsQ * numCols);
    std::vector<__nv_fp8_e4m3> hGoldenK(numRowsK * numCols);

    // Initialize input matrices
    initializeMatrix(hQ, numRowsQ, numCols);
    initializeMatrix(hK, numRowsK, numCols);

    // Compute golden reference
    std::cout << "Computing golden reference..." << std::endl;
    computeGoldenReference<__nv_bfloat16>(hQ, hGoldenQ, numRowsQ, numCols);
    computeGoldenReference<__nv_fp8_e4m3>(hK, hGoldenK, numRowsK, numCols);

    // Allocate device memory
    __nv_bfloat16 *dQ;
    cudaMalloc(&dQ, numRowsQ * numCols * sizeof(__nv_bfloat16));
    __nv_bfloat16 *dK;
    cudaMalloc(&dK, numRowsK * numCols * sizeof(__nv_bfloat16));
    __nv_bfloat16 *dOutputQ;
    cudaMalloc(&dOutputQ, numRowsQ * numCols * sizeof(__nv_bfloat16));
    __nv_fp8_e4m3 *dOutputK;
    cudaMalloc(&dOutputK, numPagesK * numTokensPerPage * numCols * sizeof(__nv_fp8_e4m3));
    __nv_fp8_e4m3 **dPagesK;
    cudaMalloc(&dPagesK, numPagesK * sizeof(__nv_fp8_e4m3*));

    // Copy input data to device
    cudaMemcpy(dQ, hQ.data(), numRowsQ * numCols * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK.data(), numRowsK * numCols * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    // Initialize page indices
    std::vector<__nv_fp8_e4m3*> hPagesK(numPagesK);
    for (int ii = 0; ii < numPagesK; ++ii) {
        hPagesK[ii] = dOutputK + ii * numTokensPerPage * numCols;  
    }
    cudaMemcpy(dPagesK, hPagesK.data(), numPagesK * sizeof(__nv_fp8_e4m3*), cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 blockDim(256);  // 8 warps * 32 threads
    dim3 gridDim((numRowsQ + numRowsK + 7) / 8);  // Ceiling division to handle all rows
    qwen3RopeCacheUpdateKernel<<<gridDim, blockDim>>>(
        dQ, dK, dOutputQ, dPagesK,
        numRowsQ, numRowsK, numTokensPerPage
    );

    // Check for kernel launch errors
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    // Wait for kernel to complete
    cudaDeviceSynchronize();

    // Copy results back to host
    cudaMemcpy(hOutputQ.data(), dOutputQ, numRowsQ * numCols * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    
    // Copy K results from pages
    for (int page = 0; page < numPagesK; ++page) {
        int rowsInPage = std::min(numTokensPerPage, numRowsK - page * numTokensPerPage);
        cudaMemcpy(hOutputK.data() + page * numTokensPerPage * numCols,
                  dOutputK + page * numTokensPerPage * numCols,
                  rowsInPage * numCols * sizeof(__nv_fp8_e4m3),
                  cudaMemcpyDeviceToHost);
    }

    // Compare results
    std::cout << "\nComparing Q results:" << std::endl;
    bool qPassed = compareResults<__nv_bfloat16>(hGoldenQ, hOutputQ, numRowsQ, numCols);
    
    std::cout << "\nComparing K results:" << std::endl;
    bool kPassed = compareResults<__nv_fp8_e4m3>(hGoldenK, hOutputK, numRowsK, numCols);

    // Free device memory
    cudaFree(dQ);
    cudaFree(dK);
    cudaFree(dOutputQ);
    cudaFree(dOutputK);
    cudaFree(dPagesK);

    // Print final result
    std::cout << "\nTest " << (qPassed && kPassed ? "PASSED" : "FAILED") << std::endl;

    return (qPassed && kPassed) ? 0 : 1;
} 

////////////////////////////////////////////////////////////////////////////////////////////////////