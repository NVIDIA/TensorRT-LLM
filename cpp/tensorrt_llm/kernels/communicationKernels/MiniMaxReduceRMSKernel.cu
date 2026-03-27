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

#define MINIMAX_REDUCE_RMS_WARP_SIZE 32
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

template <int Dim>
__device__ __forceinline__ float rms_rsqrt(float& v, float eps)
{
    constexpr float kInvDim = 1.0F / static_cast<float>(Dim);
    v = rsqrtf((v * kInvDim) + eps);
    return v;
}

template <int Dim>
__device__ __forceinline__ float4 rms_rsqrt(float4& v, float eps)
{
    constexpr float kInvDim = 1.0F / static_cast<float>(Dim);
    v.x = rsqrtf((v.x * kInvDim) + eps);
    v.y = rsqrtf((v.y * kInvDim) + eps);
    v.z = rsqrtf((v.z * kInvDim) + eps);
    v.w = rsqrtf((v.w * kInvDim) + eps);
    return v;
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

template <typename T, int NUM>
__device__ __forceinline__ void blockReduceSumRange(T* val, int rangeStart, int rangeEnd)
{
    constexpr int kWarpSize = 32;
    constexpr unsigned kFullMask = 0xffffffffu;
    static __shared__ T shared[NUM][33];

    int const activeThreadCount = max(rangeEnd - rangeStart, 0);
    bool const isActive = threadIdx.x >= rangeStart && threadIdx.x < rangeEnd;
    int const lane = threadIdx.x & (kWarpSize - 1);
    unsigned const activeMask = __ballot_sync(kFullMask, isActive);

    if (isActive)
    {
#pragma unroll
        for (int i = 0; i < NUM; ++i)
        {
            T sum = val[i];
#pragma unroll
            for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
            {
                sum += __shfl_down_sync(activeMask, sum, offset, kWarpSize);
            }
            val[i] = sum;
        }
    }

    if (isActive && lane == 0)
    {
        int const localWarpId = (threadIdx.x - rangeStart) >> 5;
#pragma unroll
        for (int i = 0; i < NUM; ++i)
        {
            shared[i][localWarpId] = val[i];
        }
    }

    __syncthreads();

    int const shiftedTid = threadIdx.x - rangeStart;
    int const warpCount = (activeThreadCount + kWarpSize - 1) / kWarpSize;
    bool const inLeaderWarp = shiftedTid >= 0 && shiftedTid < kWarpSize;
    bool const leaderLaneIsValid = inLeaderWarp && shiftedTid < warpCount;
    unsigned const leaderMask = __ballot_sync(kFullMask, leaderLaneIsValid);

    if (inLeaderWarp)
    {
#pragma unroll
        for (int i = 0; i < NUM; ++i)
        {
            T sum = leaderLaneIsValid ? shared[i][shiftedTid] : static_cast<T>(0);
#pragma unroll
            for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
            {
                sum += __shfl_down_sync(leaderMask, sum, offset, kWarpSize);
            }
            if (threadIdx.x == rangeStart)
            {
                val[i] = sum;
            }
        }
    }
}

// from sglang python/sglang/jit_kernel/include/sgl_kernel/warp.cuh
template <uint32_t kNumThreads, typename T>
__device__ __forceinline__ void local_warp_reduce_sum(T& value, uint32_t active_mask = 0xffffffffu)
{
    static_assert(kNumThreads >= 1 && kNumThreads <= MINIMAX_REDUCE_RMS_WARP_SIZE);
#pragma unroll
    for (int mask = kNumThreads / 2; mask > 0; mask >>= 1)
    {
        value += __shfl_xor_sync(active_mask, value, mask, MINIMAX_REDUCE_RMS_WARP_SIZE);
    }
}

// for float4 version
template <uint32_t kNumThreads, typename T, int ArraySize = 4>
__device__ __forceinline__ void local_warp_reduce_sum_array(T* value_ptr, uint32_t active_mask = 0xffffffffu)
{
    static_assert(kNumThreads >= 1 && kNumThreads <= MINIMAX_REDUCE_RMS_WARP_SIZE);
#pragma unroll
    for (int i = 0; i < ArraySize; ++i)
    {
#pragma unroll
        for (int mask = kNumThreads / 2; mask > 0; mask >>= 1)
        {
            value_ptr[i] += __shfl_xor_sync(active_mask, value_ptr[i], mask, MINIMAX_REDUCE_RMS_WARP_SIZE);
        }
    }
}

constexpr int next_pow2(int val)
{
    int result = 1;
    while (result < val)
    {
        result <<= 1;
    }
    return result;
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
    __shared__ float shared_vars_all_ranks;
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
#pragma unroll
            for (int r = 0; r < NRanks; ++r)
            {
                // temp data buffer [nranks, total_tokens, 1]
                reinterpret_cast<float*>(comm.data_bufs[r])[(params.rank * tot_tokens) + token_id] = (sum_variance);
            }
            // we only use the first thread to pull data from other ranks
            bool done = false;
            float vals_all_ranks[NRanks];
            while (!done)
            {
                done = true;
#pragma unroll
                for (int r = 0; r < NRanks; ++r)
                {
                    vals_all_ranks[r] = ld_global_volatile(
                        &reinterpret_cast<float*>(comm.data_bufs[params.rank])[(r * tot_tokens) + token_id]);
                    done &= !is_neg_zero(vals_all_ranks[r]);
                }
            }

            sum_variance = 0.F;
#pragma unroll
            for (int r = 0; r < NRanks; ++r)
            {
                sum_variance += vals_all_ranks[r];
            }
            sum_variance = sqrtf(sum_variance / NRanks / static_cast<float>(params.hidden_dim) + params.rms_eps);
            shared_vars_all_ranks = sum_variance;
        }

        __syncthreads();
        sum_variance = shared_vars_all_ranks;

        // step 3: calculate the rms norm (input * rsqrt(variance + eps))

        // load norm weight
        // TODO: correct the access_id_in_token
        __nv_bfloat16 norm_weight[kElemsPerAccess<DType>];
        *reinterpret_cast<typename ElemsPerAccess<DType>::norm_weight_type*>(norm_weight)
            = reinterpret_cast<typename ElemsPerAccess<DType>::norm_weight_type*>(params.rms_gamma)[access_id_in_token];

#pragma unroll
        for (int i = 0; i < kElemsPerAccess<DType>; ++i)
        {
            vals[i]
                = static_cast<DType>(static_cast<float>(vals[i]) * sum_variance * static_cast<float>(norm_weight[i]));
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
template <typename DType, int NRanks, int OriginQDim, int OriginKDim, int TokenPerBlock = 4,
    bool TriggerCompletionAtEnd = true>
__global__ void __launch_bounds__(1024) minimax_reduce_qk_rms_kernel_lamport_float4(MiniMaxReduceRMSParams params)
{
    static_assert(TokenPerBlock == 1 || TokenPerBlock == 4, "TokenPerBlock must be 1 or 4");
    constexpr int RankQDim = OriginQDim / NRanks;
    constexpr int RankKDim = OriginKDim / NRanks;
    constexpr int ThreadsPerRowQ = RankQDim / kElemsPerAccess<DType>;
    constexpr int ThreadsPerRowK = RankKDim / kElemsPerAccess<DType>;
    constexpr int NumWarpQ = (ThreadsPerRowQ + MINIMAX_REDUCE_RMS_WARP_SIZE - 1) / MINIMAX_REDUCE_RMS_WARP_SIZE;
    constexpr int NumWarpK = (ThreadsPerRowK + MINIMAX_REDUCE_RMS_WARP_SIZE - 1) / MINIMAX_REDUCE_RMS_WARP_SIZE;
    int tot_tokens = params.size_q / RankQDim;
    int tot_groups = (tot_tokens + TokenPerBlock - 1) / TokenPerBlock; // ceiling: last group may have 1-3 valid rows

    using AccumType = std::conditional_t<TokenPerBlock == 1, float, float4>;

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
    bool is_q = (access_id_in_token < NumWarpQ * MINIMAX_REDUCE_RMS_WARP_SIZE);
    int q_thread_idx = access_id_in_token;
    int k_thread_idx = (access_id_in_token - (NumWarpQ * MINIMAX_REDUCE_RMS_WARP_SIZE));
    bool is_valid_token = is_q ? (access_id_in_token < ThreadsPerRowQ) : (k_thread_idx < ThreadsPerRowK);
    float4 clear_vec = get_neg_zero();

    __shared__ float block_reduce_sum[TokenPerBlock][MINIMAX_REDUCE_RMS_WARP_SIZE + 1]; // 33 > warpQ + warpK
    __shared__ float global_scale_q[TokenPerBlock];
    __shared__ float global_scale_k[TokenPerBlock];

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
    cudaGridDependencySynchronize();
    if constexpr (!TriggerCompletionAtEnd)
    {
        cudaTriggerProgrammaticLaunchCompletion();
    }
#endif
    LamportComm<NRanks> comm(params.workspace, params.rank);

    // first step load rms params scale
    __nv_bfloat16 norm_weight[kElemsPerAccess<DType>]{};
    if (access_id_in_token < NumWarpQ * MINIMAX_REDUCE_RMS_WARP_SIZE) // Q branch
    {
        // load rms params scale
        if (is_valid_token)
        {
            *reinterpret_cast<typename ElemsPerAccess<DType>::norm_weight_type*>(norm_weight)
                = reinterpret_cast<typename ElemsPerAccess<DType>::norm_weight_type const*>(
                    params.rms_gamma)[access_id_in_token];
        }
    }
    else // K branch
    {
        // load rms params scale
        if (is_valid_token)
        {
            *reinterpret_cast<typename ElemsPerAccess<DType>::norm_weight_type*>(norm_weight)
                = reinterpret_cast<typename ElemsPerAccess<DType>::norm_weight_type const*>(
                    params.rms_gamma_k)[k_thread_idx];
        }
    }

    for (int g = group_id; g < tot_groups; g += group_stride)
    {
        alignas(16) DType vals[TokenPerBlock][kElemsPerAccess<DType>]{};
        float warp_sum_variance[TokenPerBlock]{0.F};

        if (is_q)
        {
            // Q branch: each thread only covers 128bit * TokenPerBlock
#pragma unroll
            for (int row = 0; row < TokenPerBlock; ++row)
            {
                int token_r = (g * TokenPerBlock) + row;
                if (token_r >= tot_tokens || (!is_valid_token))
                {
                    continue;
                }
                int idx_r = (token_r * ThreadsPerRowQ) + access_id_in_token;
                *reinterpret_cast<float4*>(&vals[row][0]) = reinterpret_cast<float4 const*>(params.allreduce_in)[idx_r];
#pragma unroll
                for (int i = 0; i < kElemsPerAccess<DType>; ++i)
                {
                    auto x = static_cast<float>(vals[row][i]);
                    warp_sum_variance[row] += x * x;
                }
            }
        }
        else // k branch
        {
// K branch: k_thread_idx = threadIdx.x - q_warps, each thread covers 32 K columns
#pragma unroll
            for (int row = 0; row < TokenPerBlock; ++row)
            {
                int token_r = (g * TokenPerBlock) + row;
                if (token_r >= tot_tokens || (!is_valid_token))
                {
                    continue;
                }

                int idx_r = (token_r * ThreadsPerRowK) + k_thread_idx;
                *reinterpret_cast<float4*>(&vals[row][0])
                    = reinterpret_cast<float4 const*>(params.allreduce_in_k)[idx_r];
#pragma unroll
                for (int i = 0; i < kElemsPerAccess<DType>; ++i)
                {
                    auto x = static_cast<float>(vals[row][i]);
                    warp_sum_variance[row] += x * x;
                }
            }
        }

        // Local warp reduce:
        // here we use all threads to reduce warp_sum_variance
        local_warp_reduce_sum_array<MINIMAX_REDUCE_RMS_WARP_SIZE, float, TokenPerBlock>(warp_sum_variance);
        // each warp write the warp reduce result to the shared memory
        int line = threadIdx.x & (MINIMAX_REDUCE_RMS_WARP_SIZE - 1);
        if (line == 0)
        {
#pragma unroll
            for (int _ = 0; _ < TokenPerBlock; ++_)
            {
                block_reduce_sum[_][threadIdx.x / MINIMAX_REDUCE_RMS_WARP_SIZE] = warp_sum_variance[_];
            }
        }
        __syncthreads();
        int tid = threadIdx.x;
        // then two warps process q block reduce and k block reduce respectively

        if (tid < MINIMAX_REDUCE_RMS_WARP_SIZE)
        {
            constexpr int kNumWarpQPow2 = next_pow2(NumWarpQ) > NRanks ? next_pow2(NumWarpQ) : NRanks;
            float local_sum[TokenPerBlock];
#pragma unroll
            for (int _ = 0; _ < TokenPerBlock; ++_)
            {
                local_sum[_] = tid < NumWarpQ ? block_reduce_sum[_][tid] : 0.F;
            }
            local_warp_reduce_sum_array<kNumWarpQPow2, float, TokenPerBlock>(local_sum);
            // for thread [0, NRanks), we need to push data to comm buffer
            if (tid < NRanks)
            {
#pragma unroll
                for (int _ = 0; _ < TokenPerBlock; ++_)
                {
                    if (is_neg_zero(local_sum[_]))
                    {
                        local_sum[_] = 0.F;
                    }
                }
                // push data to comm buffer, for each thread, we only need to push data to one rank

                reinterpret_cast<AccumType*>(comm.data_bufs[tid])[(params.rank * tot_groups * 2) + (2 * g)]
                    = *reinterpret_cast<AccumType*>(local_sum);
                // pull data from other ranks
                bool done = false;
                AccumType var_all_ranks;
                while (!done)
                {
                    done = true;
                    var_all_ranks = ld_global_volatile(
                        &reinterpret_cast<AccumType*>(comm.data_bufs[params.rank])[(tid * tot_groups * 2) + (2 * g)]);
                    done &= !is_neg_zero(var_all_ranks);
                }
                // local reduce
                constexpr uint32_t kActiveMask = (1 << NRanks) - 1;
                local_warp_reduce_sum_array<NRanks, float, TokenPerBlock>(
                    reinterpret_cast<float*>(&var_all_ranks), kActiveMask);
                if (tid == 0)
                {
                    *reinterpret_cast<AccumType*>(global_scale_q)
                        = rms_rsqrt<OriginQDim>(var_all_ranks, params.rms_eps);
                }
            }
        }
        // k branch
        else if (threadIdx.x >= MINIMAX_REDUCE_RMS_WARP_SIZE * NumWarpQ
            && threadIdx.x < MINIMAX_REDUCE_RMS_WARP_SIZE * (NumWarpQ + 1))
        {
            constexpr int kNumWarpKPow2 = next_pow2(NumWarpK) > NRanks ? next_pow2(NumWarpK) : NRanks;
            float local_sum[TokenPerBlock];
#pragma unroll
            for (int _ = 0; _ < TokenPerBlock; ++_)
            {
                local_sum[_] = k_thread_idx < NumWarpK ? block_reduce_sum[_][NumWarpQ + k_thread_idx] : 0.F;
            }
            local_warp_reduce_sum_array<kNumWarpKPow2, float, TokenPerBlock>(local_sum);
            // for thread [0, NRanks), we need to push data to comm buffer
            if (k_thread_idx < NRanks)
            {
#pragma unroll
                for (int _ = 0; _ < TokenPerBlock; ++_)
                {
                    if (is_neg_zero(local_sum[_]))
                    {
                        local_sum[_] = 0.F;
                    }
                }
                // push data to comm buffer, for each thread, we only need to push data to one rank
                reinterpret_cast<AccumType*>(comm.data_bufs[k_thread_idx])[(params.rank * tot_groups * 2) + (2 * g + 1)]
                    = *reinterpret_cast<AccumType*>(local_sum);
                // pull data from other ranks
                bool done = false;
                AccumType var_all_ranks;
                while (!done)
                {
                    done = true;
                    var_all_ranks = ld_global_volatile(&reinterpret_cast<AccumType*>(
                        comm.data_bufs[params.rank])[(k_thread_idx * tot_groups * 2) + (2 * g + 1)]);
                    done &= !is_neg_zero(var_all_ranks);
                }
                // local reduce
                constexpr uint32_t kActiveMask = (1 << NRanks) - 1;
                local_warp_reduce_sum_array<NRanks, float, TokenPerBlock>(
                    reinterpret_cast<float*>(&var_all_ranks), kActiveMask);
                if (k_thread_idx == 0)
                {
                    *reinterpret_cast<AccumType*>(global_scale_k)
                        = rms_rsqrt<OriginKDim>(var_all_ranks, params.rms_eps);
                }
            }
        }
        __syncthreads();
        // final part
        if (is_q)
        {
#pragma unroll
            for (int _ = 0; _ < TokenPerBlock; ++_)
            {
                warp_sum_variance[_] = global_scale_q[_];
            }
#pragma unroll
            for (int r = 0; r < TokenPerBlock; ++r)
            {
#pragma unroll
                for (int i = 0; i < kElemsPerAccess<DType>; ++i)
                {
                    vals[r][i] = static_cast<DType>(
                        static_cast<float>(vals[r][i]) * warp_sum_variance[r] * static_cast<float>(norm_weight[i]));
                }
                // store to rms_norm_out
                int token_r = (g * TokenPerBlock) + r;
                if (token_r >= tot_tokens || (!is_valid_token))
                {
                    continue;
                }
                int idx_r = (token_r * ThreadsPerRowQ) + access_id_in_token;
                reinterpret_cast<float4*>(params.rms_norm_out)[idx_r] = *reinterpret_cast<float4*>(&vals[r][0]);
            }
        }
        else
        {
#pragma unroll
            for (int _ = 0; _ < TokenPerBlock; ++_)
            {
                warp_sum_variance[_] = global_scale_k[_];
            }
#pragma unroll
            for (int r = 0; r < TokenPerBlock; ++r)
            {
#pragma unroll
                for (int i = 0; i < kElemsPerAccess<DType>; ++i)
                {
                    vals[r][i] = static_cast<DType>(
                        static_cast<float>(vals[r][i]) * warp_sum_variance[r] * static_cast<float>(norm_weight[i]));
                }
                // store to rms_norm_out
                int token_r = (g * TokenPerBlock) + r;
                if (token_r >= tot_tokens || (!is_valid_token))
                {
                    continue;
                }
                int idx_r = (token_r * ThreadsPerRowK) + k_thread_idx;
                reinterpret_cast<float4*>(params.rms_norm_out_k)[idx_r] = *reinterpret_cast<float4*>(&vals[r][0]);
            }
        }
    }

    // Clear comm buffer
    int clear_access = static_cast<int>(comm.clear_size / (sizeof(float4) / sizeof(DType)));
    int clear_stride = group_stride * blockDim.x;

    for (int idx = group_id * blockDim.x + threadIdx.x; idx < clear_access; idx += clear_stride)
    {
        reinterpret_cast<float4*>(comm.clear_buf)[idx] = clear_vec;
    }

    comm.update((2 * tot_groups * TokenPerBlock * sizeof(float) / sizeof(DType) * NRanks));
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

template <typename DType, int NRanks, int OriginQDim, int OriginKDim>
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
    int access_per_row_q = params.hidden_dim / kElemsPerAccess<DType>;
    int access_per_row_k = (params.allreduce_in_k != nullptr) ? (params.hidden_dim_k / kElemsPerAccess<DType>) : 0;
    auto divUp = [](int a, int b) { return (a + b - 1) / b * b; }; // round up to the nearest multiple of b
    int block_size = divUp(access_per_row_q, MINIMAX_REDUCE_RMS_WARP_SIZE)
        + ((params.allreduce_in_k != nullptr) ? divUp(access_per_row_k, MINIMAX_REDUCE_RMS_WARP_SIZE) : 0);
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
            TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg,
                minimax_reduce_qk_rms_kernel_lamport_float4<DType, NRanks, OriginQDim, OriginKDim, 4, true>, params));
        }
        else
        {
            TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg,
                minimax_reduce_qk_rms_kernel_lamport_float4<DType, NRanks, OriginQDim, OriginKDim, 4, true>, params));
        }
    }
    else
    {
        if (is_qk)
        {
            TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg,
                minimax_reduce_qk_rms_kernel_lamport_float4<DType, NRanks, OriginQDim, OriginKDim, 4, false>, params));
        }
        else
        {
            TLLM_CUDA_CHECK(cudaLaunchKernelEx(&cfg,
                minimax_reduce_qk_rms_kernel_lamport_float4<DType, NRanks, OriginQDim, OriginKDim, 4, false>, params));
        }
    }
}

template <int NRanks>
void dispatch_dtype(MiniMaxReduceRMSParams const& params)
{
    bool use_float4 = (params.allreduce_in_k != nullptr) && (params.hidden_dim * params.nranks == 6144)
        && (params.hidden_dim_k * params.nranks == 1024);

    if (params.dtype == nvinfer1::DataType::kHALF)
    {
        if (use_float4)
        {
            minimax_reduce_rms_kernel_launcher_float4<half, NRanks, 6144, 1024>(params);
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
            minimax_reduce_rms_kernel_launcher_float4<__nv_bfloat16, NRanks, 6144, 1024>(params);
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
            minimax_reduce_rms_kernel_launcher_float4<float, NRanks, 6144, 1024>(params);
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
