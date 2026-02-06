#include "cutlass/numeric_conversion.h"
#include "tensorrt_llm/common/config.h"
#include <cmath>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

TRTLLM_NAMESPACE_BEGIN

namespace cg = cooperative_groups;

namespace
{

#define FP4_MAX 6
#define FP8_MAX 448
#define SCALE_EPS 0.001953125
#define MAX_HIDDEN_DIM 12288
#define GROUP_NUM(x) ((x) / 16)

__forceinline__ __device__ __host__ float clamp(float x, float a, float b)
{
    return max((float) a, min((float) b, (float) x));
}

template <typename T, typename U, int Size = sizeof(U) / sizeof(T)>
__forceinline__ __device__ float local_abs_max(U* vec, float maxv)
{
    T* view = reinterpret_cast<T*>(vec);
#pragma unroll 4
    for (int i = 0; i < Size; ++i)
    {
        maxv = max((float) maxv, (float) abs((float) view[i]));
    }
    return maxv;
}

__forceinline__ __device__ int get_sf_offset(int row_id, int pos, int bdx)
{
    return (row_id % 32) * 16 + ((row_id / 32) % 4) * 4 + (row_id / 128) * (32 * 16 * bdx / 4) + (pos % 4) * 1
        + (pos / 4) * 512;
}

struct PackFp4
{
    int8_t low : 4;
    int8_t high : 4;
};

} // namespace

namespace kernels
{

// From https://github.com/actypedef/ARCQuant/blob/main/kernels/src/reorder.cu
template <typename T, int GROUP_SIZE>
__global__ void reorder_activationn_nvfp4_kernel(
    T* hidden_states, int16_t* reorder_index, uint8_t* q_out, uint8_t* q_scale, int KQ, int KE)
{
    int const hidden_dim = KQ;
    int const bdx = hidden_dim / GROUP_SIZE;
    constexpr int elements_per_thread = GROUP_SIZE;
    cg::thread_block cta = cg::this_thread_block();

    cutlass::bfloat16_t* input = reinterpret_cast<cutlass::bfloat16_t*>(hidden_states);
    cutlass::float_ue4m3_t* q_scale_tensor = reinterpret_cast<cutlass::float_ue4m3_t*>(q_scale);
    // One block solves one row of hidden states.
    __shared__ uint8_t smem[MAX_HIDDEN_DIM * sizeof(cutlass::bfloat16_t)];
    cutlass::bfloat16_t* input_smem = reinterpret_cast<cutlass::bfloat16_t*>(smem);
    // Local memory stores the reordered hidden states.
    cutlass::bfloat16_t input_frag[elements_per_thread];
    cutlass::bfloat16_t output_frag[elements_per_thread / 2];
    // Row are independent
    int row_id = blockIdx.x;
    input = input + row_id * hidden_dim;
    q_out = q_out + row_id * (GROUP_SIZE * GROUP_NUM(KQ + KE)) / 2;
    // Coalesced access global memory
    int tx = threadIdx.x;
    int tid = tx;
    int const bytes_per_iter = bdx * 16;
    int const iters = hidden_dim * sizeof(cutlass::bfloat16_t) / bytes_per_iter;
    cutlass::NumericConverter<cutlass::float_e2m1_t, float, cutlass::FloatRoundStyle::round_to_nearest> Float2E2m1;
    cutlass::NumericConverter<float, cutlass::float_e2m1_t, cutlass::FloatRoundStyle::round_to_nearest> E2m12Float;
    cutlass::NumericConverter<cutlass::float_ue4m3_t, float, cutlass::FloatRoundStyle::round_to_nearest> Float2Ue4m3;
    cutlass::NumericConverter<float, cutlass::float_ue4m3_t, cutlass::FloatRoundStyle::round_to_nearest> Ue4m32Float;
    cutlass::NumericConverter<cutlass::bfloat16_t, float, cutlass::FloatRoundStyle::round_to_nearest> Float2Bfloat16;
#pragma unroll
    for (int i = 0; i < iters; ++i)
    {
        // Each thread loads 16 bytes
        int offset = i * bytes_per_iter + tid * 16;
        *(float4*) (reinterpret_cast<uint8_t*>(input_smem) + offset)
            = *(float4*) (reinterpret_cast<uint8_t*>(input) + offset);
    }
    cta.sync();
// Reorder
#pragma unroll 4
    for (int i = 0; i < elements_per_thread; ++i)
    {
        int offset = tid * GROUP_SIZE + i;
        input_frag[i] = input_smem[reorder_index[offset]];
    }
    // Reduce to get max
    // Each ty should get its max value
    float4* input_frag_float4 = reinterpret_cast<float4*>(input_frag);
    constexpr int float4_per_thread = elements_per_thread * sizeof(cutlass::bfloat16_t) / sizeof(float4);
    float maxv = 0, scale = 1.0, r_scale = 1.0;
#pragma unroll
    for (int i = 0; i < float4_per_thread; ++i)
    {
        maxv = local_abs_max<cutlass::bfloat16_t, float4>(input_frag_float4 + i, maxv);
    }
    cta.sync();
    // Calculate scales
    // Specific layout
    float lower_bound, upper_bound;
    // Q quantize
    lower_bound = -FP4_MAX;
    upper_bound = FP4_MAX;
    scale = clamp(maxv / FP4_MAX, SCALE_EPS, FP8_MAX);
    int pos = tid + max(0, tid - GROUP_NUM(KQ - KE));
    // SF (((_32,_4),M/128),((_16,_4),K/4),(_1,1)):(((_16,_4),32*16*bdx/4),((_0,_1),_512),(_0,total_padded))
    // auto logical_coord0 = make_coord(make_coord(row_id % 32, (row_id / 32) % 4), row_id / 128);
    // auto logical_coord1 = make_coord(make_coord(0, pos % 4), pos / 4);
    // auto logical_coord2 = make_coord(0, 0);
    // Tensor q_scale_tensor = cute::make_tensor(q_scale_tensor, filter_zeros(make_layout(seq_len, KQ + KE)));
    // q_scale_tensor(make_coord(logical_coord0, logical_coord1, logical_coord2)) = Float2Ue4m3(scale);
    int sf_offset = (row_id % 32) * 16 + ((row_id / 32) % 4) * 4 + (row_id / 128) * (32 * 16 * hidden_dim / 64);
    sf_offset += (pos % 4) * 1 + (pos / 4) * 512;
    q_scale_tensor[sf_offset] = Float2Ue4m3(scale);
    // Use reverse scale to replace division by multiplication
    r_scale = 1.0 / Ue4m32Float(Float2Ue4m3(scale));
    // Quantize each thread's value
    // Each iteration quantize two things, convenient for packing int4
    PackFp4* output_frag_fp4 = reinterpret_cast<PackFp4*>(output_frag);
    for (int i = 0; i < elements_per_thread; i += 4)
    {
        float result_0, result_1, result_2, result_3;
        result_0 = clamp(((float) input_frag[i + 0] * r_scale), lower_bound, upper_bound);
        result_1 = clamp(((float) input_frag[i + 1] * r_scale), lower_bound, upper_bound);
        result_2 = clamp(((float) input_frag[i + 2] * r_scale), lower_bound, upper_bound);
        result_3 = clamp(((float) input_frag[i + 3] * r_scale), lower_bound, upper_bound);
        input_frag[i + 0] = Float2Bfloat16(
            (float) input_frag[i + 0] - E2m12Float(Float2E2m1(result_0)) * Ue4m32Float(Float2Ue4m3(scale)));
        input_frag[i + 1] = Float2Bfloat16(
            (float) input_frag[i + 1] - E2m12Float(Float2E2m1(result_1)) * Ue4m32Float(Float2Ue4m3(scale)));
        input_frag[i + 2] = Float2Bfloat16(
            (float) input_frag[i + 2] - E2m12Float(Float2E2m1(result_2)) * Ue4m32Float(Float2Ue4m3(scale)));
        input_frag[i + 3] = Float2Bfloat16(
            (float) input_frag[i + 3] - E2m12Float(Float2E2m1(result_3)) * Ue4m32Float(Float2Ue4m3(scale)));
        output_frag_fp4[i / 2 + 0].low = Float2E2m1(result_0).storage;
        output_frag_fp4[i / 2 + 0].high = Float2E2m1(result_1).storage;
        output_frag_fp4[i / 2 + 1].low = Float2E2m1(result_2).storage;
        output_frag_fp4[i / 2 + 1].high = Float2E2m1(result_3).storage;
    }
    // Residual part.
    int const ke_thread_count = GROUP_NUM(KE);
    int const kq_thread_count = bdx - ke_thread_count;
    if (tid >= bdx - GROUP_NUM(KE))
    {
        maxv = 0;
#pragma unroll
        for (int i = 0; i < float4_per_thread; ++i)
        {
            maxv = local_abs_max<cutlass::bfloat16_t, float4>(input_frag_float4 + i, maxv);
        }
        scale = clamp(maxv / FP4_MAX, SCALE_EPS, FP8_MAX);
        // logical_coord1 = make_coord(make_coord(0, (pos + 1) % 4), (pos + 1) / 4);
        // q_scale_tensor(make_coord(logical_coord0, logical_coord1, logical_coord2)) = Float2Ue4m3(scale);
        sf_offset = (row_id % 32) * 16 + ((row_id / 32) % 4) * 4 + (row_id / 128) * (32 * 16 * hidden_dim / 64);
        sf_offset += ((pos + 1) % 4) * 1 + ((pos + 1) / 4) * 512;
        q_scale_tensor[sf_offset] = Float2Ue4m3(scale);

        r_scale = 1.0 / Ue4m32Float(Float2Ue4m3(scale));
        int q_offset = elements_per_thread / 2;
        for (int i = 0; i < elements_per_thread; i += 4)
        {
            float result_0, result_1, result_2, result_3;
            result_0 = clamp(((float) input_frag[i + 0] * r_scale), lower_bound, upper_bound);
            result_1 = clamp(((float) input_frag[i + 1] * r_scale), lower_bound, upper_bound);
            result_2 = clamp(((float) input_frag[i + 2] * r_scale), lower_bound, upper_bound);
            result_3 = clamp(((float) input_frag[i + 3] * r_scale), lower_bound, upper_bound);
            output_frag_fp4[i / 2 + q_offset + 0].low = Float2E2m1(result_0).storage;
            output_frag_fp4[i / 2 + q_offset + 0].high = Float2E2m1(result_1).storage;
            output_frag_fp4[i / 2 + q_offset + 1].low = Float2E2m1(result_2).storage;
            output_frag_fp4[i / 2 + q_offset + 1].high = Float2E2m1(result_3).storage;
        }

        int const kq_region_bytes = kq_thread_count * 8;
        int const ke_thread_idx = tid - kq_thread_count;
        int const ke_thread_offset = kq_region_bytes + ke_thread_idx * 16;

        float4* q_out_ptr = reinterpret_cast<float4*>(q_out + ke_thread_offset);
        *q_out_ptr = *(reinterpret_cast<float4*>(output_frag));
    }
    else
    {
        float2* q_out_ptr = reinterpret_cast<float2*>(q_out + tid * 8);
        *q_out_ptr = *(reinterpret_cast<float2*>(output_frag));
    }
}

template <typename T, int group_size>
void run_reorder_activation_nvfp4(
    int16_t* hidden_states, int16_t* reorder_index, uint8_t* q_out, uint8_t* q_scale, int seq_len, int KQ, int KE)
{
    dim3 grids(seq_len);
    int hidden_dim = KQ;
    dim3 blocks(hidden_dim / group_size);
    if (std::is_same_v<T, __nv_bfloat16>)
    {
        reorder_activationn_nvfp4_kernel<__nv_bfloat16, group_size>
            <<<grids, blocks>>>((__nv_bfloat16*) hidden_states, reorder_index, q_out, q_scale, KQ, KE);
    }
}

// Explicit template instantiation for the specific types used
template void run_reorder_activation_nvfp4<__nv_bfloat16, 16>(
    int16_t* hidden_states, int16_t* reorder_index, uint8_t* q_out, uint8_t* q_scale, int seq_len, int KQ, int KE);

} // namespace kernels

TRTLLM_NAMESPACE_END
