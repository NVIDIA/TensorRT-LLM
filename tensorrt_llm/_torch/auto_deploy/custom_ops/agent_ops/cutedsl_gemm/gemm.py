from functools import partial
from typing import Optional

import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass import Float32
from cutlass.cute.runtime import from_dlpack, make_ptr
from torch import Tensor

from .cute_dsl_utils import get_device_capacity, get_max_active_clusters
from .gemm_default_epi import GemmDefaultSm90
from .gemm_wrapper_utils import GemmWrapperBase


def gemm(
    # (l, m, k) or (total_m, k) if varlen_m or (m, total_k) if varlen_k or (whatever, k)
    # if gather_A_varlen_m or (m, whatever) if gather_A_varlen_k
    A: Tensor,
    B: Tensor,  # (l, n, k) or (n, total_k) if varlen_k
    D: Tensor,  # (l, m, n) or (total_m, n) if varlen_m
    C: Optional[Tensor],  # (l, m, n) or (total_m, n) if varlen_m
    tile_count_semaphore: Optional[Tensor],  # (1,)
    tile_M: int,
    tile_N: int,
    cluster_M: int,
    cluster_N: int,
    pingpong: bool = False,
    persistent: bool = True,
    max_swizzle_size: int = 8,
    rowvec_bias: Optional[Tensor] = None,  # (l, n)
    colvec_bias: Optional[Tensor] = None,  # (l, m), or (total_m,) if varlen_m
    alpha: float | Tensor = 1.0,
    beta: float | Tensor = 1.0,
    cu_seqlens_m: Optional[Tensor] = None,  # (l+1,) cumulative sum of m values for variable length
    cu_seqlens_k: Optional[Tensor] = None,  # (l+1,) cumulative sum of k values for variable length
    A_idx: Optional[Tensor] = None,  # (total_m,) or (total_k,) indices for gather_A when varlen
    batch_idx_permute: Optional[Tensor] = None,  # (l,) permutation of batch indices for scheduler
    add_to_output: bool = False,
) -> None:
    varlen = cu_seqlens_m is not None or cu_seqlens_k is not None
    assert not (cu_seqlens_m is not None and cu_seqlens_k is not None), (
        "Only one of cu_seqlens_m and cu_seqlens_k can be specified"
    )
    gather_A = A_idx is not None
    if gather_A:
        assert varlen, "gather_A requires varlen (cu_seqlens_m or cu_seqlens_k must be specified)"
        assert cluster_N == 1, "gather_A requires cluster_N=1"
    if varlen:
        assert persistent, "varlen requires persistent=True"
    if add_to_output:
        assert cu_seqlens_m is None, "Add to output not supported with varlen_m"
    if cu_seqlens_m is not None:
        assert A.stride(-1) == 1, "varlen_m requires A to be k-major"
        assert D.stride(-1) == 1, "varlen_m requires D to be n-major"
    if cu_seqlens_k is not None:
        assert A.stride(-2) == 1, "varlen_k requires A to be m-major"
        assert B.stride(-2) == 1, "varlen_k requires B to be n-major"

    L, M, K, N, tensor_infos = GemmWrapperBase.validate_and_prepare_tensors(
        A, B, D, C, cu_seqlens_m=cu_seqlens_m, cu_seqlens_k=cu_seqlens_k, A_idx=A_idx
    )
    GemmWrapperBase.permute_tensors(
        tensor_infos,
        varlen_m=cu_seqlens_m is not None,
        varlen_k=cu_seqlens_k is not None,
    )
    GemmWrapperBase.extract_dtypes(tensor_infos)
    major_configs = {
        "A": ("m", "k", "l"),
        "B": ("n", "k", "l"),
        "D": ("m", "n", "l"),
        "C": ("m", "n", "l"),
    }
    GemmWrapperBase.determine_major_orders(tensor_infos, major_configs)

    device_capacity = get_device_capacity(A.device)
    assert device_capacity[0] == 9, "Only SM90 (H100) is supported"
    GemmCls = GemmDefaultSm90

    acc_dtype = Float32
    tile_shape_mn = (tile_M, tile_N)
    cluster_shape_mnk = (cluster_M, cluster_N, 1)
    if not GemmCls.is_valid_dtypes(
        tensor_infos["A"].dtype,
        tensor_infos["B"].dtype,
        acc_dtype,
        tensor_infos["D"].dtype,
        tensor_infos["A"].major,
        tensor_infos["B"].major,
    ):
        raise TypeError("Skipping due to unsupported combination of types and majors")

    max_active_clusters = get_max_active_clusters(cluster_M * cluster_N) if persistent else 0
    GemmWrapperBase.create_cute_tensors(tensor_infos, major_configs)

    def scalar_arg(scalar: float | Tensor):
        if isinstance(scalar, float):
            return Float32(scalar) if scalar != 1.0 else None
        else:
            assert isinstance(scalar, Tensor)
            return make_ptr(Float32, scalar.data_ptr(), cute.AddressSpace.gmem, assumed_align=4)

    epi_args = GemmCls.EpilogueArguments(
        scalar_arg(alpha),
        scalar_arg(beta),
        mRowVecBroadcast=(
            from_dlpack(rowvec_bias.detach(), assumed_align=4).mark_layout_dynamic(leading_dim=1)
            if rowvec_bias is not None
            else None
        ),
        mColVecBroadcast=(
            from_dlpack(colvec_bias.detach(), assumed_align=4).mark_layout_dynamic(
                leading_dim=1 if cu_seqlens_m is None else 0
            )
            if colvec_bias is not None
            else None
        ),
        add_to_output=add_to_output,
    )
    scheduler_args = GemmWrapperBase.create_scheduler_args(
        max_active_clusters,
        tile_count_semaphore,
        batch_idx_permute,
        max_swizzle_size,
    )

    # Create varlen arguments if needed (assumes persistent=True when varlen)
    varlen_args = GemmWrapperBase.create_varlen_args(
        cu_seqlens_m,
        cu_seqlens_k,
        A_idx,
        max_active_clusters,
        cluster_shape_mnk,
        tensor_infos,
        GemmCls.num_epi_tensormaps,
        pingpong,
    )

    current_stream = cutlass_torch.current_stream()
    compile_key = GemmWrapperBase.get_compile_key(
        tensor_infos,
        None,  # activation
        tile_shape_mn,
        cluster_shape_mnk,
        pingpong,
        persistent,
        tile_count_semaphore is not None,
        device_capacity,
        # Technically we don't need to recompile for different max_swizzle_size, but currently
        # not recompiling will skew the autotuning results due to power throttling.
        # Effectively we're recompiling as a way to pause between benchmarks during autotuning.
        max_swizzle_size,
        rowvec_bias.dtype if rowvec_bias is not None else None,
        colvec_bias.dtype if colvec_bias is not None else None,
        2 if isinstance(alpha, Tensor) else (1 if alpha == 1.0 else 0),
        2 if isinstance(beta, Tensor) else (1 if beta == 1.0 else 0),
        add_to_output,
        cu_seqlens_m is not None,
        cu_seqlens_k is not None,
        gather_A,
        batch_idx_permute is not None,
        key_tensor_names=("A", "B", "D", "C"),
    )
    cache = gemm.compile_cache
    if compile_key not in cache:
        GemmCls = partial(GemmCls, pingpong=pingpong, is_persistent=persistent)
        gemm_obj = GemmCls(
            acc_dtype,
            tensor_infos["A"].dtype,
            tile_shape_mn,
            cluster_shape_mnk,
            gather_A=gather_A,
        )
        cache[compile_key] = cute.compile(
            gemm_obj,
            tensor_infos["A"].cute_tensor,
            tensor_infos["B"].cute_tensor,
            tensor_infos["D"].cute_tensor,
            tensor_infos["C"].cute_tensor,
            epi_args,
            scheduler_args,
            varlen_args,
            current_stream,
        )
    cache[compile_key](
        tensor_infos["A"].cute_tensor,
        tensor_infos["B"].cute_tensor,
        tensor_infos["D"].cute_tensor,
        tensor_infos["C"].cute_tensor,
        epi_args,
        scheduler_args,
        varlen_args,
        current_stream,
    )


gemm.compile_cache = {}
