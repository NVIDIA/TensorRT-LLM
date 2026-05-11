from enum import IntEnum

import torch

# The declarations must be aligned with thUtils.h
SF_DTYPE = torch.uint8
FLOAT4_E2M1X2 = torch.uint8

# For GEMM autotuning.
# Taken from https://github.com/NVIDIA/TensorRT-LLM/blob/main/cpp/include/tensorrt_llm/runtime//modelConfig.h#L38
# TODO: move to model config, tune for blackwell hardware
FP4_BUCKETS = [64, 128, 256, 512, 1024]

# Export
float4_e2m1x2 = FLOAT4_E2M1X2
float4_sf_dtype = SF_DTYPE
fp4_buckets = FP4_BUCKETS

E2M1_VALUES = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6])

__all__ = [
    'float4_e2m1x2', 'float4_sf_dtype', 'pad_up', 'fp4_buckets', 'E2M1_VALUES',
    'dequantize_nvfp4'
]


def pad_up(x: int, y: int) -> int:
    return ((x + y - 1) // y) * y


class FP4GemmType(IntEnum):
    W4A4_NVFP4_NVFP4 = 0
    W4A8_MXFP4_MXFP8 = 1


def get_fp4_shape(input_shape, sf_vec_size, is_swizzled_layout=True):
    m = 1
    for i in range(len(input_shape) - 1):
        m *= input_shape[i]

    output_shape = [i for i in input_shape]
    output_shape[-1] //= 2

    scale_shape = pad_up(m, 128) * pad_up(
        input_shape[-1] // sf_vec_size,
        4) if is_swizzled_layout else m * (input_shape[-1] // sf_vec_size)
    return output_shape, scale_shape


def get_reorder_rows_for_gated_act_gemm_row_indices(x) -> torch.Tensor:
    """
    Reorders rows in the gemm/MOE_gemm weight matrix for min-latency
    [r0, r1, r2, r3, ..., rN/2, r(N/2+1), .. r(N-1)]
    to
    [r0, rN/2, r1, rN/2+1, ..., r(N/2-1), r(N-1)]
    """
    M = x.shape[0]
    assert M % 2 == 0, f"x.shape[0] must be even, not {M}"

    row_indices = torch.arange(M, dtype=torch.long)

    # We split into top half and bottom half, but if M is odd,
    # the bottom half is one row larger.
    top = row_indices[:(M + 1) // 2]  # round up
    bot = row_indices[(M + 1) // 2:]  # remainder

    # Create the output
    permuted_row_indices = torch.empty_like(row_indices)

    # We'll place rows of `top` and `bot` in alternation
    permuted_row_indices[0::2] = top
    permuted_row_indices[1::2] = bot

    return permuted_row_indices


def reorder_rows_for_gated_act_gemm(x):
    """
    PyTorch implementation of trt-llm gen `reorderRowsForGatedActGemm`
    """
    row_indices = get_reorder_rows_for_gated_act_gemm_row_indices(x)

    permute = lambda x: x[row_indices]

    return permute(x)


# yapf: disable
srcToDstBlk16RowMap = [
    0,  8,
    1,  9,
    2, 10,
    3, 11,
    4, 12,
    5, 13,
    6, 14,
    7, 15
]

srcToDstBlk32RowMap = [
    0,  8, 16, 24,
    1,  9, 17, 25,
    2, 10, 18, 26,
    3, 11, 19, 27,
    4, 12, 20, 28,
    5, 13, 21, 29,
    6, 14, 22, 30,
    7, 15, 23, 31
]
# yapf: enable


def get_shuffle_block_size(epilogue_tile_m: int) -> int:
    shuffle_block_size = 16
    if epilogue_tile_m % 128 == 0:
        shuffle_block_size = 32
    return shuffle_block_size


def get_shuffle_matrix_a_row_indices(input_tensor: torch.Tensor,
                                     epilogue_tile_m: int) -> torch.Tensor:
    """
    Higher-level PyTorch approach to reorder the rows in blocks of size 16 or 32.
    - We do NOT try to handle custom e2m1 memory usage (i.e. no 'K/2' bytes).
    - Instead, we purely reorder rows in a standard PyTorch shape [M, K].
    """
    # M from the input
    M = input_tensor.shape[0]

    # Choose block size 16 or 32
    shuffle_block_size = get_shuffle_block_size(epilogue_tile_m)
    row_map = (srcToDstBlk16RowMap
               if shuffle_block_size == 16 else srcToDstBlk32RowMap)

    assert M % shuffle_block_size == 0, f"input_tensor.shape[0] must be multiples of {shuffle_block_size}"

    # row_indices[new_row] = old_row
    # so row_indices is an array of size M telling us from which old_row
    # the new_row should be taken.
    row_indices = torch.empty(M, dtype=torch.long)

    for old_row in range(M):
        block_idx = old_row // shuffle_block_size
        row_in_block = old_row % shuffle_block_size
        mapped_row_in_block = row_map[row_in_block]

        new_row = block_idx * shuffle_block_size + mapped_row_in_block

        row_indices[new_row] = old_row

    return row_indices


def shuffle_matrix_a(input_tensor: torch.Tensor,
                     epilogue_tile_m: int) -> torch.Tensor:
    """
    PyTorch equivalent of trtllm-gen `shuffleMatrixA`
    """
    row_indices = get_shuffle_matrix_a_row_indices(input_tensor,
                                                   epilogue_tile_m)

    return torch.ops.trtllm.shuffle_matrix(input_tensor,
                                           row_indices.to(input_tensor.device))


def get_shuffle_matrix_sf_a_row_indices(
        input_tensor: torch.Tensor,
        epilogue_tile_m: int,
        num_elts_per_sf: int = 16) -> torch.Tensor:

    assert input_tensor.dtype == float4_sf_dtype
    assert num_elts_per_sf == 16 or num_elts_per_sf == 32

    assert input_tensor.dim(
    ) == 2, f"input_tensor should be a 2D tensor, not {input_tensor.dim()}"

    # M, K from the input
    M, K = input_tensor.shape
    assert M % 128 == 0
    assert K % 4 == 0

    row_indices = get_shuffle_matrix_a_row_indices(input_tensor,
                                                   epilogue_tile_m)

    return row_indices


def shuffle_matrix_sf_a(
    input_tensor: torch.Tensor,
    epilogue_tile_m: int,
    num_elts_per_sf: int = 16,
):
    """
    Cuda implementation of trtllm-gen `shuffleMatrixSfA` but with a caveat.
    `shuffleMatrixSfA` expects the input to be in 128x4 layout and then
    apply the same shuffling in `shuffleMatrixA` and writes out in 128x4
    layout.
    This function expects the input to be in linear layout. It's done this
    way because the scaling factors in the NVFP4 checkpoints are quantized
    and are in linear layout.
    This function doesn't add padding.
    """

    row_indices = get_shuffle_matrix_sf_a_row_indices(input_tensor,
                                                      epilogue_tile_m)

    w_shuffled = torch.ops.trtllm.shuffle_matrix(
        input_tensor, row_indices.to(input_tensor.device))

    # 128x4
    return torch.ops.trtllm.block_scale_interleave(w_shuffled)


def dequantize_nvfp4(
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_scale_2: torch.Tensor,
    out_features: int,
    in_features: int,
    target_dtype: torch.dtype = torch.bfloat16,
    block_size: int = 16,
) -> torch.Tensor:
    """Dequantize NVFP4 packed weights to a floating-point dtype.

    Args:
        packed_weight: [N, K/2] uint8 with two E2M1 nibbles per byte.
        weight_scale: Per-block scales (FP8 or FP32), flat or shaped.
        weight_scale_2: Per-tensor global scale (FP32 scalar).
        out_features: Unpadded number of output rows (N).
        in_features: Unpadded number of input columns (K).
        target_dtype: Output dtype (default bfloat16).
        block_size: Number of elements per scale block (default 16).

    Returns:
        Dequantized weight tensor of shape [out_features, in_features].
    """
    packed_uint8 = (packed_weight.view(torch.uint8)
                    if packed_weight.dtype != torch.uint8 else packed_weight)
    N_stored = packed_uint8.shape[0]
    K = packed_uint8.shape[1] * 2
    device = packed_uint8.device

    low = (packed_uint8 & 0x0F).long()
    high = ((packed_uint8 >> 4) & 0x0F).long()
    idx = torch.empty(N_stored, K, dtype=torch.long, device=device)
    idx[:, 0::2] = low
    idx[:, 1::2] = high
    vals = E2M1_VALUES.to(device)[idx]

    num_blocks = N_stored * (K // block_size)
    ws = weight_scale.to(torch.float32).reshape(-1)[:num_blocks]
    s2 = weight_scale_2.to(torch.float32)
    block_scales = (ws * s2).view(N_stored, K // block_size, 1)
    vals = vals.view(N_stored, K // block_size, block_size) * block_scales
    vals = vals.view(N_stored, K)

    return vals[:out_features, :in_features].to(target_dtype)
