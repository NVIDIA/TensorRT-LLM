#include <algorithm>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

using bf16 = __nv_bfloat16;

__global__ void build_expert_maps_kernel(int const* __restrict__ selected_experts,
    float const* __restrict__ routing_weights, int const* __restrict__ expert_offsets,
    int* __restrict__ expert_write_counters, int* __restrict__ token_indices, float* __restrict__ routing_gathered,
    int batch_size, int num_selected, int num_experts)
{
    int token = blockIdx.x * blockDim.x + threadIdx.x;

    if (token < batch_size)
    {
#pragma unroll
        for (int s = 0; s < num_selected; ++s)
        {
            int expert = selected_experts[token * num_selected + s];
            if (expert >= 0 && expert < num_experts)
            {
                int base = expert_offsets[expert];
                int pos = atomicAdd(&expert_write_counters[expert], 1);
                int write_pos = base + pos;

                token_indices[write_pos] = token;
                routing_gathered[write_pos] = routing_weights[token * num_selected + s];
            }
        }
    }
}

__global__ void gather_features_kernel(bf16 const* __restrict__ x, int const* __restrict__ token_indices,
    bf16* __restrict__ x_gathered, int start_idx, int num_tokens, int hidden_dim, int batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * hidden_dim;

    for (int i = idx; i < total; i += blockDim.x * gridDim.x)
    {
        int local_token = i / hidden_dim;
        int feat = i % hidden_dim;
        int orig_token = token_indices[start_idx + local_token];

        if (orig_token >= 0 && orig_token < batch_size && feat < hidden_dim)
        {
            x_gathered[i] = x[orig_token * hidden_dim + feat];
        }
        else
        {
            x_gathered[i] = __float2bfloat16(0.0f);
        }
    }
}

__global__ void relu_squared_kernel(bf16* data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = idx; i < size; i += blockDim.x * gridDim.x)
    {
        float val = __bfloat162float(data[i]);
        val = fmaxf(val, 0.0f);
        data[i] = __float2bfloat16(val * val);
    }
}

__global__ void scatter_output_kernel(bf16 const* __restrict__ expert_out, int const* __restrict__ token_indices,
    float const* __restrict__ routing_weights, bf16* __restrict__ final_output, int start_idx, int num_tokens,
    int hidden_dim, int batch_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = num_tokens * hidden_dim;

    for (int i = idx; i < total; i += blockDim.x * gridDim.x)
    {
        int local_token = i / hidden_dim;
        int feat = i % hidden_dim;
        int orig_token = token_indices[start_idx + local_token];

        if (orig_token >= 0 && orig_token < batch_size && feat < hidden_dim)
        {
            float weight = routing_weights[start_idx + local_token];
            float val = __bfloat162float(expert_out[i]) * weight;
            bf16 val_bf16 = __float2bfloat16(val);
            atomicAdd(&final_output[orig_token * hidden_dim + feat], val_bf16);
        }
    }
}

__global__ void count_expert_tokens_kernel(int const* __restrict__ selected_experts, int* __restrict__ expert_counts,
    int batch_size, int num_selected, int num_experts)
{
    extern __shared__ int smem_counts[];

    for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
    {
        smem_counts[e] = 0;
    }
    __syncthreads();

    for (int token = blockIdx.x * blockDim.x + threadIdx.x; token < batch_size; token += gridDim.x * blockDim.x)
    {
#pragma unroll
        for (int s = 0; s < num_selected; ++s)
        {
            int expert = selected_experts[token * num_selected + s];
            if (expert >= 0 && expert < num_experts)
            {
                atomicAdd(&smem_counts[expert], 1);
            }
        }
    }
    __syncthreads();

    for (int e = threadIdx.x; e < num_experts; e += blockDim.x)
    {
        if (smem_counts[e] > 0)
        {
            atomicAdd(&expert_counts[e], smem_counts[e]);
        }
    }
}

__global__ void prefix_sum_kernel(int const* __restrict__ counts, int* __restrict__ offsets, int num_experts)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        offsets[0] = 0;
        for (int i = 0; i < num_experts; ++i)
        {
            offsets[i + 1] = offsets[i] + counts[i];
        }
    }
}

__global__ void zero_counters_kernel(int* __restrict__ counters, int num_experts)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_experts)
    {
        counters[idx] = 0;
    }
}

// Note: Removed the duplicate cublasLtGetStatusString definition to fix compilation.
// Use cublasLtGetStatusString from cublasLt.h directly.

void launch_gpu_implementation(void* x, void* selected_experts, void* routing_weights, void** w1_weights,
    void** w2_weights, void* output, int batch_size, int hidden_dim, int intermediate_dim, int num_experts,
    int num_selected, cudaStream_t stream)
{
    int threads = 256;
    cudaMemsetAsync(output, 0, batch_size * hidden_dim * sizeof(bf16), stream);

    int* expert_counts;
    int* expert_offsets;
    int* expert_write_counters;
    cudaMallocAsync(&expert_counts, num_experts * sizeof(int), stream);
    cudaMallocAsync(&expert_offsets, (num_experts + 1) * sizeof(int), stream);
    cudaMallocAsync(&expert_write_counters, num_experts * sizeof(int), stream);

    cudaMemsetAsync(expert_counts, 0, num_experts * sizeof(int), stream);

    int blocks = (batch_size + threads - 1) / threads;
    size_t smem_size = num_experts * sizeof(int);
    count_expert_tokens_kernel<<<blocks, threads, smem_size, stream>>>(
        (int const*) selected_experts, expert_counts, batch_size, num_selected, num_experts);

    prefix_sum_kernel<<<1, 1, 0, stream>>>(expert_counts, expert_offsets, num_experts);

    blocks = (num_experts + threads - 1) / threads;
    zero_counters_kernel<<<blocks, threads, 0, stream>>>(expert_write_counters, num_experts);

    int total_assignments = batch_size * num_selected;
    int* token_indices_all;
    float* routing_gathered_all;
    cudaMallocAsync(&token_indices_all, total_assignments * sizeof(int), stream);
    cudaMallocAsync(&routing_gathered_all, total_assignments * sizeof(float), stream);

    blocks = (batch_size + threads - 1) / threads;
    build_expert_maps_kernel<<<blocks, threads, 0, stream>>>((int const*) selected_experts,
        (float const*) routing_weights, expert_offsets, expert_write_counters, token_indices_all, routing_gathered_all,
        batch_size, num_selected, num_experts);

    std::vector<int> h_counts(num_experts);
    std::vector<int> h_offsets(num_experts + 1);
    cudaMemcpyAsync(h_counts.data(), expert_counts, num_experts * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaMemcpyAsync(h_offsets.data(), expert_offsets, (num_experts + 1) * sizeof(int), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cublasLtHandle_t handle;
    cublasLtCreate(&handle);

    int max_tokens = 0;
    for (int i = 0; i < num_experts; ++i)
    {
        if (h_counts[i] > max_tokens)
            max_tokens = h_counts[i];
    }
    max_tokens += 1;

    bf16* x_gathered;
    bf16* intermediate;
    bf16* expert_out;
    cudaMallocAsync(&x_gathered, max_tokens * hidden_dim * sizeof(bf16), stream);
    cudaMallocAsync(&intermediate, max_tokens * intermediate_dim * sizeof(bf16), stream);
    cudaMallocAsync(&expert_out, max_tokens * hidden_dim * sizeof(bf16), stream);

    for (int expert_idx = 0; expert_idx < num_experts; ++expert_idx)
    {
        int num_tokens = h_counts[expert_idx];
        int offset = h_offsets[expert_idx];

        if (num_tokens == 0)
            continue;

        blocks = (num_tokens * hidden_dim + threads - 1) / threads;
        gather_features_kernel<<<blocks, threads, 0, stream>>>(
            (bf16 const*) x, token_indices_all, x_gathered, offset, num_tokens, hidden_dim, batch_size);

        cublasLtMatmulDesc_t matmulDesc;
        cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

        {
            cublasOperation_t opA = CUBLAS_OP_T;
            cublasOperation_t opB = CUBLAS_OP_N;
            cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
            cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

            cublasLtMatrixLayout_t Aop_desc, Bop_desc, Cop_desc;
            cublasLtMatrixLayoutCreate(&Aop_desc, CUDA_R_16BF, hidden_dim, intermediate_dim, hidden_dim);
            cublasLtMatrixLayoutCreate(&Bop_desc, CUDA_R_16BF, hidden_dim, num_tokens, hidden_dim);
            cublasLtMatrixLayoutCreate(&Cop_desc, CUDA_R_16BF, intermediate_dim, num_tokens, intermediate_dim);

            float alpha = 1.0f, beta = 0.0f;

            cublasStatus_t stat1
                = cublasLtMatmul(handle, matmulDesc, &alpha, w1_weights[expert_idx], Aop_desc, x_gathered, Bop_desc,
                    &beta, intermediate, Cop_desc, intermediate, Cop_desc, nullptr, nullptr, 0, stream);
            if (stat1 != CUBLAS_STATUS_SUCCESS)
            {
                std::cerr << "cublasLtMatmul (first GEMM) failed: " << cublasLtGetStatusString(stat1) << std::endl;
            }

            cublasLtMatrixLayoutDestroy(Aop_desc);
            cublasLtMatrixLayoutDestroy(Bop_desc);
            cublasLtMatrixLayoutDestroy(Cop_desc);
        }

        {
            int elems = num_tokens * intermediate_dim;
            int act_blocks = (elems + threads - 1) / threads;
            relu_squared_kernel<<<act_blocks, threads, 0, stream>>>(intermediate, elems);
        }

        {
            cublasOperation_t opA = CUBLAS_OP_T;
            cublasOperation_t opB = CUBLAS_OP_N;
            cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opA, sizeof(opA));
            cublasLtMatmulDescSetAttribute(matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opB, sizeof(opB));

            cublasLtMatrixLayout_t Aop_desc2, Bop_desc2, Cop_desc2;
            cublasLtMatrixLayoutCreate(&Aop_desc2, CUDA_R_16BF, intermediate_dim, hidden_dim, intermediate_dim);
            cublasLtMatrixLayoutCreate(&Bop_desc2, CUDA_R_16BF, intermediate_dim, num_tokens, intermediate_dim);
            cublasLtMatrixLayoutCreate(&Cop_desc2, CUDA_R_16BF, hidden_dim, num_tokens, hidden_dim);

            float alpha = 1.0f, beta = 0.0f;

            cublasStatus_t stat2
                = cublasLtMatmul(handle, matmulDesc, &alpha, w2_weights[expert_idx], Aop_desc2, intermediate, Bop_desc2,
                    &beta, expert_out, Cop_desc2, expert_out, Cop_desc2, nullptr, nullptr, 0, stream);
            if (stat2 != CUBLAS_STATUS_SUCCESS)
            {
                std::cerr << "cublasLtMatmul (second GEMM) failed: " << cublasLtGetStatusString(stat2) << std::endl;
            }

            cublasLtMatrixLayoutDestroy(Aop_desc2);
            cublasLtMatrixLayoutDestroy(Bop_desc2);
            cublasLtMatrixLayoutDestroy(Cop_desc2);
        }

        blocks = (num_tokens * hidden_dim + threads - 1) / threads;
        scatter_output_kernel<<<blocks, threads, 0, stream>>>(expert_out, token_indices_all, routing_gathered_all,
            (bf16*) output, offset, num_tokens, hidden_dim, batch_size);

        cublasLtMatmulDescDestroy(matmulDesc);
    }

    cudaFreeAsync(expert_counts, stream);
    cudaFreeAsync(expert_offsets, stream);
    cudaFreeAsync(expert_write_counters, stream);
    cudaFreeAsync(token_indices_all, stream);
    cudaFreeAsync(routing_gathered_all, stream);
    cudaFreeAsync(x_gathered, stream);
    cudaFreeAsync(intermediate, stream);
    cudaFreeAsync(expert_out, stream);

    cublasLtDestroy(handle);
}
