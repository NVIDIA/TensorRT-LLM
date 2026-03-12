#include "tensorrt_llm/common/config.h"
#include "tensorrt_llm/common/envUtils.h"
#include "tensorrt_llm/common/reduceKernelUtils.cuh"
#include "tensorrt_llm/kernels/communicationKernels/MiniMaxReduceRMSKernel.h"
#include "tensorrt_llm/kernels/quantization.cuh"
#include <cooperative_groups.h>

TRTLLM_NAMESPACE_BEGIN

namespace kernels::minimax_ar
{
namespace
{ // anonymous namespace

template <int NRanks>
struct LamportComm
{
    __device__ __forceinline__ LamportComm(void** workspace, int rank)
    {
        counter_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[0];
        flag_ptr = &reinterpret_cast<int*>(workspace[NRanks * 3])[2];
        clear_ptr = &reinterpret_cast<int64_t*>(workspace[NRanks * 3 + 1])[0];
        flag_value = *flag_ptr;
        auto comm_size = reinterpret_cast<int64_t*>(workspace[NRanks * 3 + 1])[1];
        clear_size = *clear_ptr;
        int data_offset = flag_value % 3;
        int clear_offset = (flag_value + 2) % 3;
        for (int r = 0; r < NRanks; ++r)
        {
            data_bufs[r] = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + r]) + data_offset * comm_size;
        }
        clear_buf = reinterpret_cast<uint8_t*>(workspace[2 * NRanks + rank]) + clear_offset * comm_size;
        __syncthreads();
        if (threadIdx.x == 0)
        {
            atomicAdd(counter_ptr, 1);
        }
    }

    __device__ __forceinline__ void update(int64_t new_clear_size)
    {
        if (blockIdx.x == 0 && threadIdx.x == 0)
        {
            while (*reinterpret_cast<int volatile*>(counter_ptr) != gridDim.x)
            {
            }
            *flag_ptr = (flag_value + 1) % 3;
            *clear_ptr = new_clear_size;
            *counter_ptr = 0;
        }
    }

    int* counter_ptr;
    int* flag_ptr;
    int64_t* clear_ptr;
    uint8_t* data_bufs[NRanks];
    uint8_t* clear_buf;
    int64_t clear_size;
    int flag_value;
};

__device__ __forceinline__ bool is_neg_zero(float v)
{
    return *reinterpret_cast<uint32_t*>(&v) == 0x80000000;
}

__device__ __forceinline__ bool is_neg_zero(float4 v)
{
    return is_neg_zero(v.x) || is_neg_zero(v.y) || is_neg_zero(v.z) || is_neg_zero(v.w);
}

__device__ __forceinline__ float4 get_neg_zero()
{
    float4 vec;
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        reinterpret_cast<uint32_t*>(&vec)[i] = 0x80000000;
    }
    return vec;
}

__device__ __forceinline__ float4 ld_global_volatile(float4* addr)
{
    float4 val;
    asm volatile("ld.volatile.global.v4.f32 {%0, %1, %2, %3}, [%4];"
                 : "=f"(val.x), "=f"(val.y), "=f"(val.z), "=f"(val.w)
                 : "l"(addr));
    return val;
}

__device__ __forceinline__ float ld_global_volatile(float* addr)
{
    float val;
    asm volatile("ld.volatile.global.f32 %0, [%1];" : "=f"(val) : "l"(addr));
    return val;
}

template <typename DType>
class IndexHelper
{
public:
    __device__ __forceinline__ IndexHelper(MiniMaxReduceRMSParams const& params)
    {
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
        namespace cg = cooperative_groups;
        cg::cluster_group cluster = cg::this_cluster();
        cg::grid_group grid = cg::this_grid();
        token_id = grid.cluster_rank();
        access_id_in_token = cluster.thread_rank();
        token_stride = grid.num_clusters();
#else
        token_id = blockIdx.x;
        access_id_in_token = threadIdx.x;
        token_stride = gridDim.x;
#endif
        access_id = token_id * params.hidden_dim / kElemsPerAccess<DType> + access_id_in_token;
        access_stride = token_stride * params.hidden_dim / kElemsPerAccess<DType>;
        tot_access = params.size_q / kElemsPerAccess<DType>;
    }

    int token_id;
    int access_id_in_token;
    int token_stride;
    int access_id;
    int access_stride;
    int tot_access;
};

/**
* this kernel is used to for minimax attention module
* input tensor [total_tokens, hidden_dim / tp_size], fp32
* rms weight [hidden_dim / tp_size], bf16
step 1: reduce from single rank to get the variance sum (reduce(input^2, dim=-1))
step 2: reduce from all ranks to get the variance sum (all_reduce(variance_sum))
step 3: calculate the rms norm (input * rsqrt(variance + eps))
in this case, max hidden_dim is 6144 (float data), for each token, we only need 6144 / 4 / tp_size = (1536 / tp_size)
threads so we can assume cluster size is 1 (tp_size >= 2)
 */
template <typename DType, int NRanks, bool TriggerCompletionAtEnd = true>
__global__ void __launch_bounds__(1024) minimax_reduce_rms_kernel_lamport(MiniMaxReduceRMSParams params)
{
    IndexHelper<DType> index_helper(params);
    int token_id = index_helper.token_id;
    int access_id_in_token = index_helper.access_id_in_token;
    int token_stride = index_helper.token_stride;
    int access_id = index_helper.access_id;
    int access_stride = index_helper.access_stride;
    int tot_access = index_helper.tot_access;
    int tot_tokens = params.size_q / params.hidden_dim;
    float4 clear_vec = get_neg_zero();
    // FusedOp<Pattern, DType> fused_op(params, access_id, access_id_in_token);

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
    if constexpr (!TriggerCompletionAtEnd)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
    LamportComm<NRanks> comm(params.workspace, params.rank);
    int clear_access = comm.clear_size / kElemsPerAccess<DType>;
    for (int idx = access_id; idx < tot_access; idx += access_stride, token_id += token_stride)
    {
        alignas(16) DType vals[kElemsPerAccess<DType>];
        // we use float to load and store variance sum
        float sum_variance = 0.F;
        *reinterpret_cast<float4*>(vals) = reinterpret_cast<float4*>(params.allreduce_in)[idx];
#pragma unroll
        for (int i = 0; i < kElemsPerAccess<DType>; ++i)
        {
            sum_variance += static_cast<float>(vals[i]) * static_cast<float>(vals[i]);
        }
        // step 1: reduce from single rank to get the variance sum
        tensorrt_llm::common::blockReduceSumV2<float, 1>(&sum_variance);
        if (is_neg_zero(sum_variance))
        {
            sum_variance = 0.F;
        }
        // step 2: reduce from all ranks to get the variance sum
        // be careful, we only use float to load and store variance sum
        // but we use float4 to load input tensor
        // Push data to other ranks
        // we only need the first thread to push data to other ranks
        if (threadIdx.x == 0)
        {
            for (int r = 0; r < NRanks; ++r)
            {
                // temp data buffer [nranks, total_tokens, 1]
                reinterpret_cast<float*>(comm.data_bufs[r])[(params.rank * tot_tokens) + token_id] = (sum_variance);
            }
        }

        // Load data from other ranks
        bool done = false;
        float vars_all_ranks[NRanks];
        while (!done)
        {
            done = true;
#pragma unroll
            for (int r = 0; r < NRanks; ++r)
            {
                vars_all_ranks[r] = ld_global_volatile(
                    &reinterpret_cast<float*>(comm.data_bufs[params.rank])[(r * tot_tokens) + token_id]);
                done &= !is_neg_zero(vars_all_ranks[r]);
            }
        }
        sum_variance = 0.F;
#pragma unroll
        for (int r = 0; r < NRanks; ++r)
        {
            sum_variance += vars_all_ranks[r];
        }

        // step 3: calculate the rms norm (input * rsqrt(variance + eps))

        // load norm weight
        // TODO: correct the access_id_in_token
        __nv_bfloat16 norm_weight[kElemsPerAccess<DType>];
        *reinterpret_cast<typename ElemsPerAccess<DType>::norm_weight_type*>(norm_weight)
            = reinterpret_cast<typename ElemsPerAccess<DType>::norm_weight_type*>(params.rms_gamma)[access_id_in_token];

#pragma unroll
        for (int i = 0; i < kElemsPerAccess<DType>; ++i)
        {
            vals[i] = static_cast<DType>(static_cast<float>(vals[i])
                * rsqrtf((sum_variance / static_cast<float>(params.hidden_dim) / NRanks) + params.rms_eps)
                * static_cast<float>(norm_weight[i]));
        }

        // step 4: store the rms norm
        reinterpret_cast<float4*>(params.rms_norm_out)[idx] = *reinterpret_cast<float4*>(vals);
    }
    for (int idx = access_id; idx < clear_access; idx += access_stride)
    {
        // Clear comm buffer that previous kernel used
        reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
    }
    comm.update(params.size_q * NRanks);
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if constexpr (TriggerCompletionAtEnd)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
}

/**
 * Float4 variant: process 4 rows at once, allreduce variance sums as float4 for better memory coalescing.
 * sum_variance is always float; applies to all DTypes (half, bf16, float).
 * When tot_tokens % 4 != 0, the last group pads rows with zeros; padded rows are not written to rms_norm_out.
 * IsQK: when true, process Q+K in one loop with doubled comm buffer; when false, single-matrix (Q only).
 */
template <typename DType, int NRanks, bool IsQK, bool TriggerCompletionAtEnd = true>
__global__ void __launch_bounds__(1024) minimax_reduce_rms_kernel_lamport_float4(MiniMaxReduceRMSParams params)
{
    int tot_tokens = params.size_q / params.hidden_dim;
    int tot_groups = (tot_tokens + 3) / 4; // ceiling: last group may have 1-3 valid rows
    int access_per_row_q = params.hidden_dim / kElemsPerAccess<DType>;
    int access_per_row_k = IsQK ? (params.hidden_dim_k / kElemsPerAccess<DType>) : 0;
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    namespace cg = cooperative_groups;
    cg::cluster_group cluster = cg::this_cluster();
    cg::grid_group grid = cg::this_grid();
    int group_id = grid.cluster_rank();
    int access_id_in_token = cluster.thread_rank();
    int group_stride = grid.num_clusters();
#else
    int group_id = blockIdx.x;
    int access_id_in_token = threadIdx.x;
    int group_stride = gridDim.x;
#endif
    float4 clear_vec = get_neg_zero();

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
    if constexpr (!TriggerCompletionAtEnd)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
    LamportComm<NRanks> comm(params.workspace, params.rank);

    for (int g = group_id; g < tot_groups; g += group_stride)
    {
        alignas(16) DType vals[4][kElemsPerAccess<DType>];
        float sum_variance[4] = {0.F, 0.F, 0.F, 0.F};
        float sum_variance_k[4] = {0.F, 0.F, 0.F, 0.F};

        // Load 4 rows of Q and compute partial sum of squares per row
#pragma unroll
        for (int r = 0; r < 4; ++r)
        {
            int token_r = g * 4 + r;
            if (token_r < tot_tokens)
            {
                int idx_r = token_r * access_per_row_q + access_id_in_token;
                *reinterpret_cast<float4*>(&vals[r][0]) = reinterpret_cast<float4 const*>(params.allreduce_in)[idx_r];
#pragma unroll
                for (int i = 0; i < kElemsPerAccess<DType>; ++i)
                {
                    sum_variance[r] += static_cast<float>(vals[r][i]) * static_cast<float>(vals[r][i]);
                }
            }
            else
            {
                *reinterpret_cast<float4*>(&vals[r][0]) = make_float4(0.F, 0.F, 0.F, 0.F);
                sum_variance[r] = 0.F;
            }
        }

        // Load 4 rows of K only when this thread is in K column range (access_id_in_token < access_per_row_k)
        if constexpr (IsQK)
        {
#pragma unroll
            for (int r = 0; r < 4; ++r)
            {
                int token_r = g * 4 + r;
                if (token_r < tot_tokens && access_id_in_token < access_per_row_k)
                {
                    int idx_r = token_r * access_per_row_k + access_id_in_token;
                    alignas(16) DType vals_k[kElemsPerAccess<DType>];
                    *reinterpret_cast<float4*>(vals_k) = reinterpret_cast<float4 const*>(params.allreduce_in_k)[idx_r];
#pragma unroll
                    for (int i = 0; i < kElemsPerAccess<DType>; ++i)
                    {
                        sum_variance_k[r] += static_cast<float>(vals_k[i]) * static_cast<float>(vals_k[i]);
                    }
                }
            }
        }

        tensorrt_llm::common::blockReduceSumV2<float, 4>(sum_variance);
        if constexpr (IsQK)
        {
            tensorrt_llm::common::blockReduceSumV2<float, 4>(sum_variance_k);
        }
#pragma unroll
        for (int r = 0; r < 4; ++r)
        {
            if (is_neg_zero(sum_variance[r]))
            {
                sum_variance[r] = 0.F;
            }
            if constexpr (IsQK)
            {
                if (is_neg_zero(sum_variance_k[r]))
                {
                    sum_variance_k[r] = 0.F;
                }
            }
        }

        // Allreduce: write float4(s) to comm
        if (threadIdx.x == 0)
        {
            float4 sum4;
            sum4.x = sum_variance[0];
            sum4.y = sum_variance[1];
            sum4.z = sum_variance[2];
            sum4.w = sum_variance[3];
            for (int r = 0; r < NRanks; ++r)
            {
                if constexpr (IsQK)
                {
                    reinterpret_cast<float4*>(comm.data_bufs[r])[(params.rank * 2 * tot_groups) + 2 * g] = sum4;
                }
                else
                {
                    reinterpret_cast<float4*>(comm.data_bufs[r])[(params.rank * tot_groups) + g] = sum4;
                }
            }
            if constexpr (IsQK)
            {
                sum4.x = sum_variance_k[0];
                sum4.y = sum_variance_k[1];
                sum4.z = sum_variance_k[2];
                sum4.w = sum_variance_k[3];
                for (int r = 0; r < NRanks; ++r)
                {
                    reinterpret_cast<float4*>(comm.data_bufs[r])[(params.rank * 2 * tot_groups) + 2 * g + 1] = sum4;
                }
            }
        }

        // Read Q from buffer first, sum, then RMS and store Q; then read K, sum, RMS and store K
        bool done = false;
        float4 vars_all_ranks[NRanks];
        while (!done)
        {
            done = true;
#pragma unroll
            for (int r = 0; r < NRanks; ++r)
            {
                if constexpr (IsQK)
                {
                    vars_all_ranks[r] = ld_global_volatile(
                        &reinterpret_cast<float4*>(comm.data_bufs[params.rank])[(r * 2 * tot_groups) + 2 * g]);
                }
                else
                {
                    vars_all_ranks[r] = ld_global_volatile(
                        &reinterpret_cast<float4*>(comm.data_bufs[params.rank])[(r * tot_groups) + g]);
                }
                done &= !is_neg_zero(vars_all_ranks[r]);
            }
        }
        sum_variance[0] = 0.F;
        sum_variance[1] = 0.F;
        sum_variance[2] = 0.F;
        sum_variance[3] = 0.F;
#pragma unroll
        for (int r = 0; r < NRanks; ++r)
        {
            sum_variance[0] += vars_all_ranks[r].x;
            sum_variance[1] += vars_all_ranks[r].y;
            sum_variance[2] += vars_all_ranks[r].z;
            sum_variance[3] += vars_all_ranks[r].w;
        }

        // Load norm weight for Q (same column for all 4 rows)
        __nv_bfloat16 norm_weight[kElemsPerAccess<DType>];
        *reinterpret_cast<typename ElemsPerAccess<DType>::norm_weight_type*>(norm_weight)
            = reinterpret_cast<typename ElemsPerAccess<DType>::norm_weight_type const*>(
                params.rms_gamma)[access_id_in_token];

        // RMS norm and store 4 rows of Q (skip write for padded rows)
        for (int r = 0; r < 4; ++r)
        {
            int token_r = g * 4 + r;
            if (token_r >= tot_tokens)
            {
                continue;
            }
            float scale = rsqrtf((sum_variance[r] / static_cast<float>(params.hidden_dim) / NRanks) + params.rms_eps);
#pragma unroll
            for (int i = 0; i < kElemsPerAccess<DType>; ++i)
            {
                vals[r][i]
                    = static_cast<DType>(static_cast<float>(vals[r][i]) * scale * static_cast<float>(norm_weight[i]));
            }
            int idx_out = token_r * access_per_row_q + access_id_in_token;
            reinterpret_cast<float4*>(params.rms_norm_out)[idx_out] = *reinterpret_cast<float4*>(&vals[r][0]);
        }

        // Then read K from buffer, sum, RMS and store K (only when IsQK and access_id_in_token < access_per_row_k)
        if constexpr (IsQK)
        {
            float4 vars_k_all_ranks[NRanks];
            done = false;
            while (!done)
            {
                done = true;
#pragma unroll
                for (int r = 0; r < NRanks; ++r)
                {
                    vars_k_all_ranks[r] = ld_global_volatile(
                        &reinterpret_cast<float4*>(comm.data_bufs[params.rank])[(r * 2 * tot_groups) + 2 * g + 1]);
                    done &= !is_neg_zero(vars_k_all_ranks[r]);
                }
            }
            sum_variance_k[0] = 0.F;
            sum_variance_k[1] = 0.F;
            sum_variance_k[2] = 0.F;
            sum_variance_k[3] = 0.F;
#pragma unroll
            for (int r = 0; r < NRanks; ++r)
            {
                sum_variance_k[0] += vars_k_all_ranks[r].x;
                sum_variance_k[1] += vars_k_all_ranks[r].y;
                sum_variance_k[2] += vars_k_all_ranks[r].z;
                sum_variance_k[3] += vars_k_all_ranks[r].w;
            }

            if (access_id_in_token < access_per_row_k)
            {
                __nv_bfloat16 norm_weight_k[kElemsPerAccess<DType>];
                *reinterpret_cast<typename ElemsPerAccess<DType>::norm_weight_type*>(norm_weight_k)
                    = reinterpret_cast<typename ElemsPerAccess<DType>::norm_weight_type const*>(
                        params.rms_gamma_k)[access_id_in_token];
                for (int r = 0; r < 4; ++r)
                {
                    int token_r = g * 4 + r;
                    if (token_r >= tot_tokens)
                    {
                        continue;
                    }
                    alignas(16) DType vals_k[kElemsPerAccess<DType>];
                    int idx_r = token_r * access_per_row_k + access_id_in_token;
                    *reinterpret_cast<float4*>(vals_k) = reinterpret_cast<float4 const*>(params.allreduce_in_k)[idx_r];
                    float scale_k = rsqrtf(
                        (sum_variance_k[r] / static_cast<float>(params.hidden_dim_k) / NRanks) + params.rms_eps);
#pragma unroll
                    for (int i = 0; i < kElemsPerAccess<DType>; ++i)
                    {
                        vals_k[i] = static_cast<DType>(
                            static_cast<float>(vals_k[i]) * scale_k * static_cast<float>(norm_weight_k[i]));
                    }
                    reinterpret_cast<float4*>(params.rms_norm_out_k)[idx_r] = *reinterpret_cast<float4*>(vals_k);
                }
            }
        }
    }

    // Clear comm buffer: clear full size set by previous kernel (same as non-float4 kernel).
    int clear_access = static_cast<int>(comm.clear_size / kElemsPerAccess<DType>);
    int clear_stride = group_stride * access_per_row_q;
    for (int idx = group_id * access_per_row_q + threadIdx.x; idx < clear_access; idx += clear_stride)
    {
        reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
    }
    comm.update(IsQK ? (2 * tot_groups * 8 * NRanks) : (tot_groups * 8 * NRanks));
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    if constexpr (TriggerCompletionAtEnd)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
}

int get_sm_count()
{
    static int sm_count = 0;
    if (sm_count == 0)
    {
        int device_id;
        TLLM_CUDA_CHECK(cudaGetDevice(&device_id));
        cudaDeviceProp device_prop;
        cudaGetDeviceProperties(&device_prop, device_id);
        sm_count = device_prop.multiProcessorCount;
    }
    return sm_count;
}

template <typename DType, int NRanks>
void minimax_reduce_rms_kernel_launcher(MiniMaxReduceRMSParams const& params)
{
    TLLM_CHECK(params.size_q % params.hidden_dim == 0);
    TLLM_CHECK(params.hidden_dim % kElemsPerAccess<DType> == 0);
    static int SM = tensorrt_llm::common::getSMVersion();
    int token_num = params.size_q / params.hidden_dim;
    // for current problem size, we only need one cluster
    int sm_count = get_sm_count();
    int cluster_size = 1;
    int cluster_num = token_num;
    int threads_per_token = params.hidden_dim / kElemsPerAccess<DType>;
    int block_size = threads_per_token;
    int grid_size = (std::min(sm_count, cluster_num * cluster_size) / cluster_size) * cluster_size;

    cudaLaunchConfig_t cfg;
    cfg.gridDim = grid_size;
    cfg.blockDim = block_size;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = params.stream;

    cudaLaunchAttribute attribute[2];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    attribute[1].id = cudaLaunchAttributeClusterDimension;
    attribute[1].val.clusterDim.x = cluster_size;
    attribute[1].val.clusterDim.y = 1;
    attribute[1].val.clusterDim.z = 1;
    cfg.attrs = attribute;
    cfg.numAttrs = SM >= 90 ? 2 : 0;

    bool trigger_completion_at_end = params.trigger_completion_at_end;
    if (trigger_completion_at_end)
    {
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg, minimax_reduce_rms_kernel_lamport<DType, NRanks, true>, params));
    }
    else
    {
        TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg, minimax_reduce_rms_kernel_lamport<DType, NRanks, false>, params));
    }
}

template <typename DType, int NRanks>
void minimax_reduce_rms_kernel_launcher_float4(MiniMaxReduceRMSParams const& params)
{
    TLLM_CHECK(params.size_q % params.hidden_dim == 0);
    TLLM_CHECK(params.hidden_dim % kElemsPerAccess<DType> == 0);
    if (params.allreduce_in_k != nullptr)
    {
        TLLM_CHECK(params.hidden_dim >= params.hidden_dim_k);
        TLLM_CHECK(params.size_k % params.hidden_dim_k == 0);
        TLLM_CHECK(params.hidden_dim_k % kElemsPerAccess<DType> == 0);
        TLLM_CHECK(params.size_q / params.hidden_dim == params.size_k / params.hidden_dim_k);
    }
    int token_num = params.size_q / params.hidden_dim;
    int tot_groups = (token_num + 3) / 4; // ceiling
    if (tot_groups == 0)
    {
        return;
    }
    static int SM = tensorrt_llm::common::getSMVersion();
    int sm_count = get_sm_count();
    int cluster_size = 1;
    int cluster_num = tot_groups;
    int threads_per_token = params.hidden_dim / kElemsPerAccess<DType>;
    int block_size = threads_per_token;
    int grid_size = (std::min(sm_count, cluster_num * cluster_size) / cluster_size) * cluster_size;

    cudaLaunchConfig_t cfg;
    cfg.gridDim = grid_size;
    cfg.blockDim = block_size;
    cfg.dynamicSmemBytes = 0;
    cfg.stream = params.stream;

    cudaLaunchAttribute attribute[2];
    attribute[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attribute[0].val.programmaticStreamSerializationAllowed = tensorrt_llm::common::getEnvEnablePDL() ? 1 : 0;
    attribute[1].id = cudaLaunchAttributeClusterDimension;
    attribute[1].val.clusterDim.x = cluster_size;
    attribute[1].val.clusterDim.y = 1;
    attribute[1].val.clusterDim.z = 1;
    cfg.attrs = attribute;
    cfg.numAttrs = SM >= 90 ? 2 : 0;

    bool trigger_completion_at_end = params.trigger_completion_at_end;
    bool is_qk = (params.allreduce_in_k != nullptr);
    if (trigger_completion_at_end)
    {
        if (is_qk)
        {
            TLLM_CUDA_CHECK(
                cudaLaunchKernelEx(&cfg, minimax_reduce_rms_kernel_lamport_float4<DType, NRanks, true, true>, params));
        }
        else
        {
            TLLM_CUDA_CHECK(
                cudaLaunchKernelEx(&cfg, minimax_reduce_rms_kernel_lamport_float4<DType, NRanks, false, true>, params));
        }
    }
    else
    {
        if (is_qk)
        {
            TLLM_CUDA_CHECK(
                cudaLaunchKernelEx(&cfg, minimax_reduce_rms_kernel_lamport_float4<DType, NRanks, true, false>, params));
        }
        else
        {
            TLLM_CUDA_CHECK(cudaLaunchKernelEx(
                &cfg, minimax_reduce_rms_kernel_lamport_float4<DType, NRanks, false, false>, params));
        }
    }
}

template <int NRanks>
void dispatch_dtype(MiniMaxReduceRMSParams const& params)
{
    int token_num = params.size_q / params.hidden_dim;
    constexpr int kFloat4MinTokens = 32;
    bool use_float4 = (token_num >= kFloat4MinTokens);

    if (params.dtype == nvinfer1::DataType::kHALF)
    {
        if (use_float4)
        {
            minimax_reduce_rms_kernel_launcher_float4<half, NRanks>(params);
        }
        else
        {
            minimax_reduce_rms_kernel_launcher<half, NRanks>(params);
        }
    }
    else if (params.dtype == nvinfer1::DataType::kBF16)
    {
        if (use_float4)
        {
            minimax_reduce_rms_kernel_launcher_float4<__nv_bfloat16, NRanks>(params);
        }
        else
        {
            minimax_reduce_rms_kernel_launcher<__nv_bfloat16, NRanks>(params);
        }
    }
    else if (params.dtype == nvinfer1::DataType::kFLOAT)
    {
        if (use_float4)
        {
            minimax_reduce_rms_kernel_launcher_float4<float, NRanks>(params);
        }
        else
        {
            minimax_reduce_rms_kernel_launcher<float, NRanks>(params);
        }
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false, "Unsupported data type for minimax_reduce_rms_op");
    }
}
} // namespace

void minimax_reduce_rms_op(MiniMaxReduceRMSParams const& params)
{
    if (params.nranks == 2)
    {
        dispatch_dtype<2>(params);
    }
    else if (params.nranks == 4)
    {
        dispatch_dtype<4>(params);
    }
    else if (params.nranks == 8)
    {
        dispatch_dtype<8>(params);
    }
    else if (params.nranks == 16)
    {
        dispatch_dtype<16>(params);
    }
    else
    {
        TLLM_CHECK_WITH_INFO(false, "minimax_reduce_rms_op: unsupported ranks number!");
    }
}

} // namespace kernels::minimax_ar

TRTLLM_NAMESPACE_END
