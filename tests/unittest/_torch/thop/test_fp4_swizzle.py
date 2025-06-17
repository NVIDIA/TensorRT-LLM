import pytest
import torch
from utils.util import skip_pre_blackwell

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.utils import reswizzle_sf, swizzle_sf, unswizzle_sf


# Reference PyTorch implementations (original)
def swizzle_sf_ref(sf: torch.Tensor,
                   row: int,
                   col: int,
                   scaling_vector_size: int = 16):
    """Reference PyTorch implementation of swizzle_sf"""
    factor = scaling_vector_size * 4
    num_m_tiles = (row + 128 - 1) // 128
    num_k_tiles = (col + factor - 1) // factor
    # SF layout [num_m_tiles, num_k_tiles, 32 (m_tile column major), 4 (m_tile column major), 4(k_tile)]
    sf_full = torch.zeros(num_m_tiles * 32 * 4,
                          num_k_tiles * 4,
                          dtype=sf.dtype,
                          device=sf.device)
    sf_full[:row, :(col //
                    scaling_vector_size)] = sf[:row, :(col //
                                                       scaling_vector_size)]
    sf_full_reshaped = sf_full.view(num_m_tiles, 4, 32, num_k_tiles, 4)
    sf_full_swizzle = sf_full_reshaped.transpose(1, 3)
    sf_swizzle = sf_full_swizzle.reshape(-1)
    return sf_swizzle


def unswizzle_sf_ref(sf: torch.Tensor,
                     row: int,
                     col: int,
                     scaling_vector_size: int = 16):
    """Reference PyTorch implementation of unswizzle_sf"""
    factor = scaling_vector_size * 4
    num_m_tiles = (row + 128 - 1) // 128
    num_k_tiles = (col + factor - 1) // factor
    # SF layout [num_m_tiles, num_k_tiles, 32 (m_tile column major), 4 (m_tile column major), 4(k_tile)]
    sf_reshaped = sf.view(num_m_tiles, num_k_tiles, 32, 4, 4)
    sf_unswizzle = sf_reshaped.transpose(1, 3)
    sf_unswizzle = sf_unswizzle.reshape(num_m_tiles * 32 * 4, num_k_tiles * 4)
    sf_unswizzle_sliced = sf_unswizzle[:row, :(col // scaling_vector_size)]
    return sf_unswizzle_sliced.contiguous()


def reswizzle_sf_ref(sf: torch.Tensor,
                     row: int,
                     col: int,
                     scaling_vector_size: int = 16):
    """Reference PyTorch implementation of reswizzle_sf"""
    factor = scaling_vector_size * 4
    num_m_tiles = (row + 128 - 1) // 128
    num_k_tiles = (col + factor - 1) // factor
    partition_size = num_m_tiles * num_k_tiles * 32 * 4 * 4
    num_partitions = sf.numel() // partition_size
    sf_reshaped = sf.view(num_partitions, num_m_tiles, num_k_tiles, 32, 4, 4)
    sf_unswizzle = sf_reshaped.transpose(2, 4)
    sf_unswizzle = sf_unswizzle.reshape(num_partitions, num_m_tiles * 32 * 4,
                                        num_k_tiles * 4)
    total_rows = num_partitions * row
    num_m_tiles_out = (total_rows + 128 - 1) // 128
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
        (128, 64),  # Minimal: 1 tile
        (256, 128),  # 2×2 tiles
        (384, 192),  # 3×3 tiles
        (512, 256),  # 4×4 tiles
        (1024, 512),  # Large matrix
    ])
def test_swizzle_sf(rows, cols):
    """Test C++ swizzle_sf against PyTorch reference implementation"""
    scaling_vector_size = 16
    sf_cols = cols // scaling_vector_size

    # Create scaling factor data using uint8 dtype
    sf_data = torch.arange(rows * sf_cols,
                           dtype=fp4_utils.float4_sf_dtype).view(rows, sf_cols)

    # Apply reference implementation (convert to float32 for ref)
    ref_result = swizzle_sf_ref(sf_data.float(), rows, cols,
                                scaling_vector_size)

    # Apply C++ implementation
    result = swizzle_sf(sf_data, rows, cols, scaling_vector_size)

    # Verify results are equivalent
    torch.testing.assert_close(result.float(), ref_result.float())


@skip_pre_blackwell
@pytest.mark.parametrize("rows,cols", [
    (256, 128),
    (512, 256),
])
def test_unswizzle_sf(rows, cols):
    """Test C++ unswizzle_sf against PyTorch reference implementation"""
    scaling_vector_size = 16
    sf_cols = cols // scaling_vector_size

    # Create scaling factor data by first swizzling with reference implementation
    original_sf_data = torch.arange(rows * sf_cols,
                                    dtype=torch.uint8).view(rows, sf_cols)
    swizzled_sf_data = swizzle_sf_ref(original_sf_data.float(), rows, cols,
                                      scaling_vector_size)

    # Apply reference unswizzle
    ref_result = unswizzle_sf_ref(swizzled_sf_data, rows, cols,
                                  scaling_vector_size)

    # Apply C++ unswizzle (convert to uint8 for C++ op)
    swizzled_uint8 = (swizzled_sf_data % 256).to(torch.uint8)
    cpp_result = unswizzle_sf(swizzled_uint8, rows, cols, scaling_vector_size)

    # Verify C++ result matches reference result
    torch.testing.assert_close(ref_result,
                               cpp_result.float(),
                               atol=1.0,
                               rtol=0.1)


@skip_pre_blackwell
@pytest.mark.parametrize(
    "rows,cols",
    [
        (128, 64),  # Minimal: 1 tile
        (256, 128),  # 2×2 tiles
        (384, 192),  # 3×3 tiles
        (512, 256),  # 4×4 tiles
        (640, 320),  # 5×5 tiles
        (1024, 512),  # Large matrix
    ])
def test_swizzle_round_trip(rows, cols):
    """Test that swizzle/unswizzle operations are inverse of each other"""
    scaling_vector_size = 16
    sf_cols = cols // scaling_vector_size

    # Create scaling factor data
    original_sf_data = torch.randint(0,
                                     256, (rows, sf_cols),
                                     dtype=fp4_utils.float4_sf_dtype)

    # Apply swizzle then unswizzle using the utils functions
    swizzled_sf = swizzle_sf(original_sf_data, rows, cols, scaling_vector_size)
    unswizzled_sf = unswizzle_sf(swizzled_sf, rows, cols, scaling_vector_size)

    # Verify round-trip preserves original scaling factor data
    torch.testing.assert_close(original_sf_data.float(), unswizzled_sf.float())


@skip_pre_blackwell
@pytest.mark.parametrize("rows,cols,num_partitions", [
    (128, 64, 2),
    (256, 128, 4),
])
def test_reswizzle_sf(rows, cols, num_partitions):
    """Test C++ reswizzle_sf against PyTorch reference implementation"""
    scaling_vector_size = 16
    sf_cols = cols // scaling_vector_size

    # Create scaling factor data: multiple partitions of swizzled data
    partition_sf_data = []
    for i in range(num_partitions):
        sf_data = torch.arange(i * 100,
                               i * 100 + rows * sf_cols,
                               dtype=torch.uint8).view(rows, sf_cols)
        swizzled_sf = swizzle_sf_ref(sf_data.float(), rows, cols,
                                     scaling_vector_size)
        partition_sf_data.append(swizzled_sf)

    # Concatenate partitions
    multi_partition_sf_data = torch.cat(partition_sf_data)

    # Apply reference reswizzle
    ref_result = reswizzle_sf_ref(multi_partition_sf_data, rows, cols,
                                  scaling_vector_size)

    # Apply C++ reswizzle
    result = reswizzle_sf(multi_partition_sf_data, rows, cols,
                          scaling_vector_size)

    # Verify results are equivalent
    torch.testing.assert_close(result.float(), ref_result.float())
