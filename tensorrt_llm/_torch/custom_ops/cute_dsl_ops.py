# from typing import Tuple, Type
import sys
# from pathlib import Path
import math

from cuda import cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

import torch

# sys.path.append('/home/lmin/scratch/trt-dkg/cutlass_ir/compiler/python/examples/hopper')
# from blockwise_gemm import HopperBlockwiseGemmKernel
# from contiguous_grouped_gemm import HopperContiguousGroupedBlockwiseGemmKernel
from .cute_dsl_kernel.blockwise_gemm import HopperBlockwiseGemmKernel
from .cute_dsl_kernel.contiguous_grouped_gemm import HopperContiguousGroupedBlockwiseGemmKernel

kernel_dict = {}
contiguous_group_gemm_kernel_dict = {}

# input is fp8, weight is fp8, input_scale is fp32, weight_scale is fp32
# output is bf16
def cute_dsl_fp8_linear(a: torch.Tensor, b: torch.Tensor, a_sf: torch.Tensor, b_sf: torch.Tensor, tile_shape_mnk = (128, 128, 128), cluster_shape_mnk = (1, 1, 1)) -> torch.Tensor:
    """Performs linear operation using cute-dsl with autotuning.

    :param a: Input tensor of shape (M, K)
    :type a: torch.Tensor, type: fp8
    :param b: Weight tensor of shape (N, K)
    :type b: torch.Tensor, type: fp8
    :param a_sf: Input scale tensor of shape (P). P is computed by the following formula:
        P = (div_up(shape_m_4_align * div_up(shape_k, 128) * sizeof(float), 128) * 128)/sizeof(float)
    :type a_sf: torch.Tensor, type: fp32
    :param b_sf: Weight scale tensor of shape (w_n, w_k)
    :type b_sf: torch.Tensor, type: fp32

    :return: Output tensor of shape (M, N)
    :rtype: torch.Tensor, type: bf16
    """
    m, n, k = a.shape[0], b.shape[0], a.shape[1]
    w_n, w_k = b_sf.shape[0], b_sf.shape[1]
    c = torch.empty(*(m, n), dtype=torch.bfloat16, device="cuda")
    # print(f"limin: m = {m}, n = {n}, k = {k}, w_n = {w_n}, w_k = {w_k}")
    # print("limin: a.dtype = ", a.dtype)
    # print("limin: b.dtype = ", b.dtype)
    # print("limin: a_sf.dtype = ", a_sf.dtype)
    # print("limin: b_sf.dtype = ", b_sf.dtype)

    # torch_tensor -> cute.tensor
    a_tmp = a.as_strided((m, k, 1), (k, 1, m * k)).view(torch.uint8)
    b_tmp = b.as_strided((n, k, 1), (k, 1, n * k)).view(torch.uint8)
    c_tmp = c.as_strided((m, n, 1), (n, 1, m * n))

    weight_scale_tmp = b_sf.as_strided((w_n, w_k, 1), (w_k, 1, w_n * w_k))

    m_padded = (m + 3) // 4 * 4
    input_scale_tmp = a_sf[0:m_padded * w_k]
    # print(f"limin: 0, input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")
    input_scale_tmp = input_scale_tmp.reshape(-1, m_padded)
    # print(f"limin: 1, input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")
    input_scale_tmp = input_scale_tmp[:w_k, :m_padded].contiguous().permute(1, 0)
    # print(f"limin: 2, input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")
    input_scale_tmp = input_scale_tmp.as_strided((m_padded, w_k, 1), (1, m_padded, m_padded * w_k))
    # print(f"limin: 3, input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")

    mA = from_dlpack(a_tmp, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mB = from_dlpack(b_tmp, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mC = from_dlpack(c_tmp, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mA.element_type = cutlass.Float8E4M3FN
    mB.element_type = cutlass.Float8E4M3FN

    # TODO: mSFA is column major
    mSFA = from_dlpack(input_scale_tmp, assumed_align=16).mark_layout_dynamic(leading_dim=0)
    mSFB = from_dlpack(weight_scale_tmp, assumed_align=16).mark_layout_dynamic(leading_dim=1)

    # print(f"limin: mA.shape = {mA.shape}, mA.stride = {mA.stride}")
    # print(f"limin: mB.shape = {mB.shape}, mB.stride = {mB.stride}")
    # print(f"limin: mC.shape = {mC.shape}, mC.stride = {mC.stride}")
    # print(f"limin: mSFA.shape = {mSFA.shape}, mSFA.stride = {mSFA.stride}")
    # print(f"limin: mSFB.shape = {mSFB.shape}, mSFB.stride = {mSFB.stride}")

    gemm = HopperBlockwiseGemmKernel(
        cutlass.Float32, # acc_dtype,
        # (64, 128, 128), (128, 128, 128)
        tile_shape_mnk,
        #  [(1, 1), (2, 1), (2, 2), (1, 2), (1, 4), (4, 1)],
        # (1, 1, 1),
        cluster_shape_mnk,
    )

    # get stream
    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)

    # cache_key = ((128, 128, 128), (1, 1, 1))
    cache_key = (tile_shape_mnk, cluster_shape_mnk)
    if cache_key not in kernel_dict:
        compiled_gemm = cute.compile(
            gemm,
            mA,
            mB,
            mC,
            mSFA,
            mSFB,
            stream,
        )
        kernel_dict[cache_key] = compiled_gemm
    else:
        compiled_gemm = kernel_dict[cache_key]

    # launch gemm kernel
    compiled_gemm(mA, mB, mC, mSFA, mSFB, stream)

    return c


def cute_dsl_fp8_linear_ref(a: torch.Tensor, b: torch.Tensor, a_sf: torch.Tensor, b_sf: torch.Tensor) -> torch.Tensor:
    m, n, k = a.shape[0], b.shape[0], a.shape[1]
    w_n, w_k = b_sf.shape[0], b_sf.shape[1]
    # c = torch.empty(*(m, n), dtype=torch.bfloat16, device="cuda")

    a_tmp = a.as_strided((m, k, 1), (k, 1, m * k)).view(torch.uint8)
    b_tmp = b.as_strided((n, k, 1), (k, 1, n * k)).view(torch.uint8)
    # c_tmp = c.as_strided((m, n, 1), (n, 1, m * n))

    weight_scale_tmp = b_sf.as_strided((w_n, w_k, 1), (w_k, 1, w_n * w_k))

    m_padded = (m + 3) // 4 * 4
    # input_scale_tmp = a_sf.slice(0, 0, m_padded * w_k)
    input_scale_tmp = a_sf[0:m_padded * w_k]
    # print(f"limin: input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")
    input_scale_tmp = input_scale_tmp.reshape(-1, m_padded)
    # print(f"limin: 1, input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")
    input_scale_tmp = input_scale_tmp[:w_k, :m].contiguous().permute(1, 0)
    # print(f"limin: 2, input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")
    input_scale_tmp = input_scale_tmp.as_strided((m, w_k, 1), (1, m, m * w_k))
    # print(f"limin: 3, input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")

    import math
    # update
    def pad_and_multiply(scale, tensor):
        cm, ck, _ = scale.shape
        m, k, _ = tensor.shape
        IsGroupWise = False
        IsBlockWise = False
        if ck == math.ceil(k / 128):
            IsGroupWise = True
        if cm == math.ceil(m / 128):
            IsBlockWise = True
        if not IsBlockWise and not IsGroupWise:
            raise ValueError("Only support granularity = 128")

        k_idx = torch.arange(k, device=scale.device)
        if IsGroupWise:
            k_idx = k_idx // 128
        m_idx = torch.arange(m, device=scale.device)
        if IsBlockWise:
            m_idx = m_idx // 128
        expanded_scale = scale[m_idx[:, None], k_idx, :]

        result = expanded_scale * tensor

        return result

    updated_a = pad_and_multiply(input_scale_tmp, a_tmp.to(torch.float32))
    updated_b = pad_and_multiply(weight_scale_tmp, b_tmp.to(torch.float32))

    ref = torch.einsum("mkl,nkl->mnl", updated_a, updated_b).to(torch.bfloat16)
    ref = ref.reshape(m, n)
    return ref


def get_gridx_mapping(group_m_lists: torch.Tensor, num_group: int, tile_x = 128) -> torch.Tensor:
    """
    :param group_m_lists:  tensor of shape (num_group)
    :type group_m_lists: torch.Tensor, type: int32
    :param num_group:  int
    :type num_group: int
    :return: Output tensor of shape (xxx)
    """
    gidx_mapping = []
    for i in range(num_group):
        group_m = group_m_lists[i]
        assert group_m % tile_x == 0
        gidx_mapping.extend([i] * (group_m // tile_x))

    return torch.tensor(gidx_mapping, device="cuda", dtype=torch.int32)


# TODO: group blockwise gemm
def cute_dsl_fp8_group_blockwise_gemm_impl(a: torch.Tensor, b: torch.Tensor, a_sf: torch.Tensor, b_sf: torch.Tensor, group_m: torch.Tensor) -> torch.Tensor:
    """Performs group blockwise gemm operation using cute-dsl with autotuning.

    :param a: Input tensor of shape (M, K)
    :type a: torch.Tensor, type: fp8
    :param b: Weight tensor of shape (L, N, K)
    :type b: torch.Tensor, type: fp8
    :param a_sf: Input scale tensor of shape (P). P is computed by the following formula:
        P = (div_up(shape_m_4_align * div_up(shape_k, 128) * sizeof(float), 128) * 128)/sizeof(float)
    :type a_sf: torch.Tensor, type: fp32
    :param b_sf: Weight scale tensor of shape (L, w_n, w_k)
    :type b_sf: torch.Tensor, type: fp32
    :param group_m:  tensor of shape (M)
    :type group_m: torch.Tensor, type: int32

    :return: Output tensor of shape (M, N)
    :rtype: torch.Tensor, type: bf16
    """
    m, k = a.shape[0], a.shape[1]
    l, n, k = b.shape[0], b.shape[1], b.shape[2]
    num_group, w_n, w_k = b_sf.shape[0], b_sf.shape[1], b_sf.shape[2]
    print(f"limin: m = {m}, k = {k}, l = {l}, n = {n}, num_group = {num_group}, w_n = {w_n}, w_k = {w_k}")
    c = torch.empty(*(m, n), dtype=torch.bfloat16, device="cuda")

    a_tmp = a.as_strided((m, k, 1), (k, 1, m * k)).view(torch.uint8)
    b_tmp = b.permute(1, 2, 0).view(torch.uint8)
    c_tmp = c.as_strided((m, n, 1), (n, 1, m * n))
    # print(f"limin: a_tmp.shape = {a_tmp.shape}, a_tmp.stride = {a_tmp.stride()}")
    # print(f"limin: b_tmp.shape = {b_tmp.shape}, b_tmp.stride = {b_tmp.stride()}")
    # print(f"limin: c_tmp.shape = {c_tmp.shape}, c_tmp.stride = {c_tmp.stride()}")

    mA = from_dlpack(a_tmp, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mB = from_dlpack(b_tmp, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mC = from_dlpack(c_tmp, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mA.element_type = cutlass.Float8E4M3FN
    mB.element_type = cutlass.Float8E4M3FN

    b_sf_tmp = b_sf.permute(1, 2, 0)
    mSFB = from_dlpack(b_sf_tmp, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    # print(f"limin: b_sf_tmp.shape = {b_sf_tmp.shape}, b_sf_tmp.stride = {b_sf_tmp.stride()}")

    # TODO:
    m_padded = (m + 3) // 4 * 4
    input_scale_tmp = a_sf[0:m_padded * w_k]
    # print(f"limin: 0, input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")
    input_scale_tmp = input_scale_tmp.reshape(-1, m_padded)
    # print(f"limin: 1, input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")
    input_scale_tmp = input_scale_tmp[:w_k, :m].contiguous().permute(1, 0)
    # print(f"limin: 2, input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")
    input_scale_tmp = input_scale_tmp.as_strided((m, w_k, 1), (1, m, m * w_k))
    # print(f"limin: input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")

    mSFA = from_dlpack(input_scale_tmp, assumed_align=16).mark_layout_dynamic(leading_dim=0)

    print("group_m = ", group_m)
    # TODO:
    tile_x = 128
    grid_mapping_tmp = get_gridx_mapping(group_m, num_group, tile_x)
    print(f"limin: grid_mapping_tmp.shape = {grid_mapping_tmp.shape}, grid_mapping_tmp.stride = {grid_mapping_tmp.stride()}, grid_mapping_tmp = {grid_mapping_tmp}")
    gidx_mapping = from_dlpack(grid_mapping_tmp).mark_layout_dynamic()

    # print("limin: input = ", a_tmp)
    # print("limin: weight = ", b_tmp)
    # print("limin: input_scale = ", input_scale_tmp)
    # print("limin: weight_scale = ", b_sf_tmp)    

    gemm = HopperContiguousGroupedBlockwiseGemmKernel(
        cutlass.Float32, # acc_dtype,
        # (64, 128, 128)
        (tile_x, 128, 128), # tile_shape_mnk,
        #  [(1, 1), (1, 2), (1, 4)
        (1, 1, 1), # cluster_shape_mnk,
    )

    torch_stream = torch.cuda.current_stream()
    stream = cuda.CUstream(torch_stream.cuda_stream)
    # compile gemm kernel
    cache_key = ((tile_x, 128, 128), (1, 1, 1))
    if cache_key not in contiguous_group_gemm_kernel_dict:
        compiled_gemm = cute.compile(gemm, mA, mB, mC, mSFA, mSFB, gidx_mapping, stream)
        contiguous_group_gemm_kernel_dict[cache_key] = compiled_gemm
    else:
        compiled_gemm = contiguous_group_gemm_kernel_dict[cache_key]
    # execution
    compiled_gemm(mA, mB, mC, mSFA, mSFB, gidx_mapping, stream)

    # print("limin: c = ", c)
    return c


def cute_dsl_fp8_group_blockwise_gemm_ref(a: torch.Tensor, b: torch.Tensor, a_sf: torch.Tensor, b_sf: torch.Tensor, group_m_list: torch.Tensor, use_offset_array: bool = False) -> torch.Tensor:
    m, k = a.shape[0], a.shape[1]
    l, n, k = b.shape[0], b.shape[1], b.shape[2]
    num_group, w_n, w_k = b_sf.shape[0], b_sf.shape[1], b_sf.shape[2]
    # print(f"limin: m = {m}, k = {k}, l = {l}, n = {n}, num_group = {num_group}, w_n = {w_n}, w_k = {w_k}")
    # c = torch.empty(*(m, n), dtype=torch.bfloat16, device="cuda")

    # TODO: view(int8) will cause error.
    a_tmp = a.as_strided((m, k, 1), (k, 1, m * k))
    b_tmp = b.permute(1, 2, 0)
    # c_tmp = c.as_strided((m, n, 1), (n, 1, m * n))
    # print(f"limin: a_tmp.shape = {a_tmp.shape}, a_tmp.stride = {a_tmp.stride()}")
    # print(f"limin: b_tmp.shape = {b_tmp.shape}, b_tmp.stride = {b_tmp.stride()}")
    # print(f"limin: c_tmp.shape = {c_tmp.shape}, c_tmp.stride = {c_tmp.stride()}")

    m_padded = (m + 3) // 4 * 4
    input_scale_tmp = a_sf[0:m_padded * w_k]
    # print(f"limin: 0, input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")
    input_scale_tmp = input_scale_tmp.reshape(-1, m_padded)
    # print(f"limin: 1, input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")
    input_scale_tmp = input_scale_tmp[:w_k, :m].contiguous().permute(1, 0)
    # print(f"limin: 2, input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")
    input_scale_tmp = input_scale_tmp.as_strided((m, w_k, 1), (1, m, m * w_k))
    # print(f"limin: input_scale_tmp.shape = {input_scale_tmp.shape}, input_scale_tmp.stride = {input_scale_tmp.stride()}")

    # TOOD: contiguous
    weight_scale_tmp = b_sf.permute(1, 2, 0)
    # print(f"limin: weight_scale_tmp.shape = {weight_scale_tmp.shape}, weight_scale_tmp.stride = {weight_scale_tmp.stride()}")

    # print("limin: input = ", a_tmp)
    # print("limin: weight = ", b_tmp)
    # print("limin: input_scale = ", input_scale_tmp)
    # print("limin: weight_scale = ", weight_scale_tmp)

    # print("limin: input negative_numbers = ",  torch.sum(a_tmp < 0).item())
    # print("limin: weight negative_numbers = ",  torch.sum(b_tmp < 0).item())
    # print("limin: input_scale negative_numbers = ",  torch.sum(input_scale_tmp < 0).item())
    # print("limin: weight_scale negative_numbers = ",  torch.sum(weight_scale_tmp < 0).item())
    # update
    def pad_and_multiply(scale, tensor):
        cm, ck, _ = scale.shape
        m, k, _ = tensor.shape
        IsGroupWise = False
        IsBlockWise = False
        if ck == math.ceil(k / 128):
            IsGroupWise = True
        if cm == math.ceil(m / 128):
            IsBlockWise = True
        if not IsBlockWise and not IsGroupWise:
            raise ValueError("Only support granularity = 128")

        k_idx = torch.arange(k, device=scale.device)
        if IsGroupWise:
            k_idx = k_idx // 128
        m_idx = torch.arange(m, device=scale.device)
        if IsBlockWise:
            m_idx = m_idx // 128
        expanded_scale = scale[m_idx[:, None], k_idx, :]

        result = expanded_scale * tensor

        return result

    updated_a = pad_and_multiply(input_scale_tmp, a_tmp.to(torch.float32))
    updated_b = pad_and_multiply(weight_scale_tmp, b_tmp.to(torch.float32))

    # print(f"limin: updated_a = {updated_a}")
    # print(f"limin: updated_b = {updated_b}")

    ref = torch.zeros((m, n), device="cuda", dtype=torch.float32)
    if not use_offset_array:
        start = 0
        for i, group_m in enumerate(group_m_list):
            end = start + group_m
            ref[start:end, :] = torch.einsum(
                "mk,nk->mn", updated_a[start:end, :, 0], updated_b[:, :, i]
            )
            start = end
    else:
        # print(f"limin: group_m_list = {group_m_list}")
        # [0, 2, 5, 6]
        len_group_m_list = group_m_list.shape[0]
        for i in range(len_group_m_list - 1):
            start = group_m_list[i]
            end = group_m_list[i+1]
            # assert start <= end, f"Invalid group boundaries: start={start} > end={end}"
            ref[start:end, :] = torch.einsum(
                "mk,nk->mn", updated_a[start:end, :, 0], updated_b[:, :, i]
            )

    ref = ref.to(torch.bfloat16)
    # print(f"limin: ref.shape = {ref.shape}, ref.stride = {ref.stride()}, ref = {ref}")
    return ref
