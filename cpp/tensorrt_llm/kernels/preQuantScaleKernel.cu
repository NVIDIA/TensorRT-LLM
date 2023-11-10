#include "tensorrt_llm/kernels/preQuantScaleKernel.h"

namespace tensorrt_llm
{
namespace kernels
{
namespace
{
template <typename T>
struct Vec2Type;

template <>
struct Vec2Type<half>
{
    using type = half2;
};
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && defined(ENABLE_BF16))
template <>
struct Vec2Type<__nv_bfloat16>
{
    using type = __nv_bfloat162;
};
#endif
}; // namespace

template <typename T, int kProcessRows, typename AccessType>
__global__ void apply_per_channel_scale(T* smoothed_act, const T* act, const T* per_channel_scale, int rows, int cols)
{
    static constexpr int kElems = sizeof(AccessType) / sizeof(T);
    T scale[kElems], act_vec[kElems];
    int col_offset = blockIdx.x * blockDim.x + threadIdx.x;
    int row_offset = blockIdx.y;
    if (col_offset * kElems >= cols || row_offset * kProcessRows >= rows)
        return;
    act += row_offset * kProcessRows * cols;
    smoothed_act += row_offset * kProcessRows * cols;
    *reinterpret_cast<AccessType*>(scale) = reinterpret_cast<const AccessType*>(per_channel_scale)[col_offset];
#pragma unroll
    for (int i = 0; i < kProcessRows; ++i)
    {
        *reinterpret_cast<AccessType*>(act_vec) = reinterpret_cast<const AccessType*>(act + i * cols)[col_offset];
        if constexpr ((std::is_same_v<T, half>
#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && defined(ENABLE_BF16))
                          || std::is_same_v<T, __nv_bfloat16>
#endif
                          ) &&(kElems % 2 == 0))
        {
            using Vec2 = typename Vec2Type<T>::type;
#pragma unroll
            for (int j = 0; j < kElems; j += 2)
            {
                *reinterpret_cast<Vec2*>(act_vec + j)
                    = __hmul2(*reinterpret_cast<Vec2*>(act_vec + j), *reinterpret_cast<Vec2*>(scale + j));
            }
        }
        else
        {
#pragma unroll
            for (int j = 0; j < kElems; ++j)
            {
                act_vec[j] = static_cast<T>(static_cast<float>(act_vec[j]) * static_cast<float>(scale[j]));
            }
        }
        reinterpret_cast<AccessType*>(smoothed_act + i * cols)[col_offset] = *reinterpret_cast<AccessType*>(act_vec);
    }
}

template <typename T, int kProcessRows, typename AccessType = float4>
void apply_per_channel_scale_kernel_launcher_(
    T* smoothed_act, const T* act, const T* per_channel_scale, int rows, int cols, cudaStream_t stream = 0)
{
    static constexpr int kElems = sizeof(AccessType) / sizeof(T);
    dim3 block(128);
    dim3 grid((cols / kElems + block.x - 1) / block.x, (rows + kProcessRows - 1) / kProcessRows);
    apply_per_channel_scale<T, kProcessRows, AccessType>
        <<<grid, block, 0, stream>>>(smoothed_act, act, per_channel_scale, rows, cols);
}

template <typename T>
void apply_per_channel_scale_kernel_launcher(
    T* smoothed_act, const T* act, const T* per_channel_scale, int rows, int cols, cudaStream_t stream)
{
    int elems = rows * cols;
    if (elems < 2048 * 2048)
    {
        apply_per_channel_scale_kernel_launcher_<T, 1, float4>(
            smoothed_act, act, per_channel_scale, rows, cols, stream);
    }
    else if (elems < 4096 * 4096)
    {
        apply_per_channel_scale_kernel_launcher_<T, 4, float4>(
            smoothed_act, act, per_channel_scale, rows, cols, stream);
    }
    else if (elems < 8192 * 8192)
    {
        apply_per_channel_scale_kernel_launcher_<T, 8, float4>(
            smoothed_act, act, per_channel_scale, rows, cols, stream);
    }
    else
    {
        apply_per_channel_scale_kernel_launcher_<T, 16, float4>(
            smoothed_act, act, per_channel_scale, rows, cols, stream);
    }
}

#define INSTANTIATE_PREQUANT_SCALE(T)                                                                                  \
    template void apply_per_channel_scale_kernel_launcher<T>(                                                          \
        T * smoothed_act, const T* act, const T* per_channel_scale, int rows, int cols, cudaStream_t stream)

INSTANTIATE_PREQUANT_SCALE(half);
#if defined(ENABLE_BF16)
INSTANTIATE_PREQUANT_SCALE(__nv_bfloat16);
#endif

} // namespace kernels
} // namespace tensorrt_llm
