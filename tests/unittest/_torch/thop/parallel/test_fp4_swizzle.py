import pytest
import torch
from utils.util import skip_pre_blackwell

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.utils import (compute_swizzled_sf_shape, reswizzle_sf,
                                       swizzle_sf, unswizzle_sf)
from tensorrt_llm.math_utils import ceil_div


# Reference PyTorch implementations (original)
def swizzle_sf_ref(sf: torch.Tensor,
                   row: int,
                   col: int,
                   scaling_vector_size: int = 16):
    """Reference PyTorch implementation of swizzle_sf"""
    col_sf = ceil_div(col, scaling_vector_size)
    num_m_tiles = ceil_div(row, 128)
    num_k_tiles = ceil_div(col_sf, 4)
    # SF layout [num_m_tiles, num_k_tiles, 32 (m_tile column major), 4 (m_tile column major), 4(k_tile)]
    sf_full = torch.zeros(num_m_tiles * 32 * 4,
                          num_k_tiles * 4,
                          dtype=sf.dtype,
                          device=sf.device)
    sf_full[:row, :col_sf] = sf[:row, :col_sf]
    sf_full_reshaped = sf_full.view(num_m_tiles, 4, 32, num_k_tiles, 4)
    sf_full_swizzle = sf_full_reshaped.transpose(1, 3)
    sf_swizzle = sf_full_swizzle.reshape(-1)
    return sf_swizzle


def unswizzle_sf_ref(sf: torch.Tensor,
                     row: int,
                     col: int,
                     scaling_vector_size: int = 16):
    """Reference PyTorch implementation of unswizzle_sf"""
    cols_sf = ceil_div(col, scaling_vector_size)
    num_m_tiles = ceil_div(row, 128)
    num_k_tiles = ceil_div(cols_sf, 4)
    # SF layout [num_m_tiles, num_k_tiles, 32 (m_tile column major), 4 (m_tile column major), 4(k_tile)]
    sf_reshaped = sf.view(num_m_tiles, num_k_tiles, 32, 4, 4)
    sf_unswizzle = sf_reshaped.transpose(1, 3)
    sf_unswizzle = sf_unswizzle.reshape(num_m_tiles * 32 * 4, num_k_tiles * 4)
    return sf_unswizzle.contiguous()


def reswizzle_sf_ref(sf: torch.Tensor,
                     row: int,
                     col: int,
                     scaling_vector_size: int = 16):
    """Reference PyTorch implementation of reswizzle_sf"""
    cols_sf = ceil_div(col, scaling_vector_size)
    num_m_tiles = ceil_div(row, 128)
    num_k_tiles = ceil_div(cols_sf, 4)
    partition_size = num_m_tiles * num_k_tiles * 32 * 4 * 4
    num_partitions = sf.numel() // partition_size
    sf_reshaped = sf.view(num_partitions, num_m_tiles, num_k_tiles, 32, 4, 4)
    sf_unswizzle = sf_reshaped.transpose(2, 4)
    sf_unswizzle = sf_unswizzle.reshape(num_partitions, num_m_tiles * 32 * 4,
                                        num_k_tiles * 4)
    total_rows = num_partitions * row
    num_m_tiles_out = ceil_div(total_rows, 128)
    sf_out = torch.zeros(
        num_m_tiles_out,
        4,
        32,
        num_k_tiles,
        4,
        dtype=sf.dtype,
        device=sf.device,
    )
    sf_out_reshaped = sf_out.view(num_m_tiles_out * 32 * 4, num_k_tiles * 4)
    sf_out_reshaped[:total_rows] = sf_unswizzle[:, :row].reshape(total_rows, -1)
    sf_out_swizzle = sf_out.transpose(1, 3).reshape(-1)
    return sf_out_swizzle


@skip_pre_blackwell
@pytest.mark.parametrize(
    "rows,cols",
    [
        (1, 1),  # test padding
        (1, 16),
        (1, 63),
        (127, 1),
        (127, 16),
        (127, 63),
        (128, 64),  # 1x1 tiles
        (128, 128),  # 1×2 tiles
        (256, 64),  # 2×1 tiles
        (512, 256),  # 4×4 tiles
    ])
def test_swizzle_sf(rows, cols):
    """Test C++ swizzle_sf against PyTorch reference implementation"""
    scaling_vector_size = 16
    sf_cols = ceil_div(cols, scaling_vector_size)

    # Create scaling factor data using fp4_sf_dtype
    sf_data = torch.randint(0,
                            256, (rows * sf_cols, ),
                            dtype=fp4_utils.float4_sf_dtype,
                            device="cuda").view(rows, sf_cols)

    # Apply reference implementation
    ref_result = swizzle_sf_ref(sf_data, rows, cols, scaling_vector_size)

    # Apply C++ implementation
    result = swizzle_sf(sf_data, rows, cols, scaling_vector_size)

    # Verify results are equivalent
    torch.testing.assert_close(result, ref_result)


@skip_pre_blackwell
@pytest.mark.parametrize(
    "rows,cols",
    [
        (128, 64),  # 1x1 tiles
        (128, 128),  # 1×2 tiles
        (256, 64),  # 2×1 tiles
        (512, 256),  # 4×4 tiles
    ])
def test_unswizzle_sf(rows, cols):
    """Test C++ unswizzle_sf against PyTorch reference implementation"""
    scaling_vector_size = 16
    sf_cols = ceil_div(cols, scaling_vector_size)

    # Create scaling factor data by first swizzling with reference implementation
    original_sf_data = torch.randint(0,
                                     256, (rows * sf_cols, ),
                                     dtype=fp4_utils.float4_sf_dtype,
                                     device="cuda").view(rows, sf_cols)
    swizzled_sf_data = swizzle_sf_ref(original_sf_data, rows, cols,
                                      scaling_vector_size)
    # Apply reference unswizzle
    ref_result = unswizzle_sf_ref(swizzled_sf_data, rows, cols,
                                  scaling_vector_size)

    # Note that unlike swizzle_sf, unswizzle_sf does not return a 1D tensor
    result = unswizzle_sf(swizzled_sf_data, rows, cols, scaling_vector_size)

    # Verify C++ result matches reference result
    torch.testing.assert_close(result, ref_result)


@skip_pre_blackwell
@pytest.mark.parametrize(
    "rows,cols",
    [
        (1, 1),  # test padding
        (1, 16),
        (1, 63),
        (127, 1),
        (127, 16),
        (127, 63),
        (128, 64),  # 1x1 tiles
        (128, 128),  # 1×2 tiles
        (256, 64),  # 2×1 tiles
        (512, 256),  # 4×4 tiles
    ])
def test_swizzle_round_trip(rows, cols):
    """Test that swizzle/unswizzle operations are inverse of each other"""
    scaling_vector_size = 16
    sf_cols = ceil_div(cols, scaling_vector_size)

    # Create scaling factor data
    original_sf_data = torch.randint(0,
                                     256, (rows * sf_cols, ),
                                     dtype=fp4_utils.float4_sf_dtype,
                                     device="cuda")

    # Apply swizzle then unswizzle using the utils functions
    swizzled_sf = swizzle_sf(original_sf_data, rows, cols, scaling_vector_size)

    padded_rows, padded_sf_cols = compute_swizzled_sf_shape(rows, sf_cols)

    # Check that the swizzled scaling factor data is padded correctly
    assert padded_rows * padded_sf_cols == swizzled_sf.numel()

    padded_cols = padded_sf_cols * scaling_vector_size
    unswizzled_sf = unswizzle_sf(swizzled_sf, padded_rows, padded_cols,
                                 scaling_vector_size)
    unswizzled_sf = unswizzled_sf[:rows, :sf_cols]

    # Verify round-trip preserves original scaling factor data
    torch.testing.assert_close(original_sf_data.view(rows, sf_cols),
                               unswizzled_sf[:rows, :sf_cols])


@skip_pre_blackwell
@pytest.mark.parametrize("num_partitions", [1, 2, 3])
@pytest.mark.parametrize(
    "rows,cols",
    [
        (1, 1),  # test padding
        (1, 16),
        (1, 63),
        (127, 1),
        (127, 16),
        (127, 63),
        (128, 64),  # 1x1 tiles
        (128, 128),  # 1×2 tiles
        (256, 64),  # 2×1 tiles
        (512, 256),  # 4×4 tiles
    ])
def test_reswizzle_sf(num_partitions, rows, cols):
    """Test C++ reswizzle_sf against PyTorch reference implementation"""
    scaling_vector_size = 16
    sf_cols = ceil_div(cols, scaling_vector_size)

    original_sf_data = torch.randint(0,
                                     256, (num_partitions, rows, sf_cols),
                                     dtype=fp4_utils.float4_sf_dtype,
                                     device="cuda")
    swizzled_sf_data = swizzle_sf(original_sf_data, rows, cols,
                                  scaling_vector_size)

    # Apply reference reswizzle
    ref_result = reswizzle_sf_ref(swizzled_sf_data, rows, cols,
                                  scaling_vector_size)

    # Apply C++ reswizzle
    result = reswizzle_sf(swizzled_sf_data, rows, cols, scaling_vector_size)

    # Verify results are equivalent
    torch.testing.assert_close(result, ref_result)
