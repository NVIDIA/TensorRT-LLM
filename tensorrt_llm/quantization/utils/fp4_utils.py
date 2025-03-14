import torch

# The declarations must be aligned with thUtils.h
SF_DTYPE = torch.uint8
FLOAT4_E2M1X2 = torch.uint8


def pad_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


# For GEMM autotuning.
# Taken from https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/include/tensorrt_llm/runtime//modelConfig.h#L38
# TODO: move to model config, tune for blackwell hardware
FP4_BUCKETS = [64, 128, 256, 512, 1024]

# Export
float4_e2m1x2 = FLOAT4_E2M1X2
float4_sf_dtype = SF_DTYPE
fp4_buckets = FP4_BUCKETS

__all__ = ['float4_e2m1x2', 'float4_sf_dtype', 'pad_up', 'fp4_buckets']


def get_fp4_shape(input_shape, sf_vec_size):
    m = 1
    for i in range(len(input_shape) - 1):
        m *= input_shape[i]

    output_shape = [i for i in input_shape]
    output_shape[-1] //= 2

    scale_shape = pad_up(m, 128) * pad_up(input_shape[-1] // sf_vec_size, 4)
    return output_shape, scale_shape
