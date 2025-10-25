from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass.cute.runtime import from_dlpack, make_ptr
from torch import Tensor

from .cute_dsl_utils import torch2cute_dtype_map
from .tile_scheduler import TileSchedulerOptions
from .varlen_utils import VarlenArguments


@dataclass
class GemmTensorInfo:
    tensor: Optional[Tensor]
    dtype: Optional[Any] = None
    major: Optional[str] = None
    cute_tensor: Optional[cute.Tensor] = None


class GemmWrapperBase:
    @staticmethod
    def validate_tensor(tensor: Tensor, name: str, ndim: int) -> None:
        assert tensor.dim() == ndim and tensor.is_cuda, f"{name} must be a {ndim}D CUDA tensor"
        assert tensor.dtype in torch2cute_dtype_map, f"Unsupported dtype for {name}"

    @staticmethod
    def validate_shape(tensor: Tensor, expected_shape: Tuple[int, ...], name: str) -> None:
        assert tensor.shape == expected_shape, (
            f"{name} must have shape {expected_shape}, got {tensor.shape}"
        )

    @staticmethod
    def get_major_order(tensor: Tensor, dims: Tuple[str, str, str]) -> str:
        # Tensor is already permuted to (dims[0], dims[1], dims[2])
        # stride(1) == 1 means dims[1] is contiguous (innermost)
        return dims[1] if tensor.stride(1) == 1 else dims[0]

    @staticmethod
    def create_cute_tensor(
        tensor: Optional[Tensor],
        major: Optional[str],
        dims: Tuple[str, str, str],
        assumed_align: int = 16,
    ) -> Optional[cute.Tensor]:
        if tensor is None:
            return None
        # Tensor is already permuted to (dims[0], dims[1], dims[2]) or (dim[0], dim[1])
        # If major is dims[1], leading_dim is 1; if major is dims[0], leading_dim is 0
        leading_dim = 1 if major == dims[1] else 0
        return from_dlpack(tensor.detach(), assumed_align=assumed_align).mark_layout_dynamic(
            leading_dim=leading_dim
        )

    @staticmethod
    def validate_and_prepare_tensors(
        A: Tensor,
        B: Tensor,
        D: Optional[Tensor] = None,
        C: Optional[Tensor] = None,
        additional_tensors: Optional[Dict[str, Tensor]] = None,
        cu_seqlens_m: Optional[Tensor] = None,
        cu_seqlens_k: Optional[Tensor] = None,
        A_idx: Optional[Tensor] = None,
    ) -> Tuple[int, int, int, int, Dict[str, GemmTensorInfo]]:
        assert not (cu_seqlens_m is not None and cu_seqlens_k is not None), (
            "Only one of cu_seqlens_m and cu_seqlens_k can be specified"
        )
        assert B.dtype == A.dtype, "A and B must have the same dtype"

        # Validate A_idx if provided (for gather_A case)
        gather_A = A_idx is not None
        if gather_A:
            assert cu_seqlens_m is not None or cu_seqlens_k is not None, (
                "gather_A requires either varlen_m or varlen_k"
            )
            assert A_idx.dtype == torch.int32, f"A_idx must be int32, got {A_idx.dtype}"
            assert A_idx.dim() == 1, f"A_idx must be 1D, got {A_idx.dim()}D"

        # Determine mode and extract dimensions
        if cu_seqlens_m is not None:
            # varlen_m: A is (total_m, k) or (whatever, k) if gather_A, B is (l, n, k), D/C are (total_m, n)
            assert A.dim() == 2, f"A must be 2D when using varlen_m, got {A.dim()}D"
            assert B.dim() == 3, f"B must be 3D with varlen_m, got {B.dim()}D"

            if gather_A:
                # When gather_A, A can have any number of rows, we use A_idx.shape[0] as total_M
                total_M = A_idx.shape[0]
                _, K = A.shape
            else:
                total_M, K = A.shape

            L, N, K_B = B.shape
            assert K == K_B, f"K dimension mismatch: A has {K}, B has {K_B}"
            assert cu_seqlens_m.shape == (L + 1,), (
                f"cu_seqlens_m must have shape ({L + 1},), got {cu_seqlens_m.shape}"
            )
            M = total_M
            dc_shape = (total_M, N)
            dc_ndim = 2
        elif cu_seqlens_k is not None:
            # varlen_k: A is (m, total_k) or (m, whatever) if gather_A, B is (n, total_k), D/C are (l, m, n)
            assert A.dim() == 2, f"A must be 2D when using varlen_k, got {A.dim()}D"
            assert B.dim() == 2, f"B must be 2D with varlen_k, got {B.dim()}D"

            if gather_A:
                # When gather_A with varlen_k, A can have any number of columns, we use A_idx.shape[0] as total_K
                M, _ = A.shape
                total_K = A_idx.shape[0]
            else:
                M, total_K = A.shape

            N, K_B = B.shape
            assert total_K == K_B, f"K dimension mismatch: expected {total_K}, B has {K_B}"
            L = cu_seqlens_k.shape[0] - 1
            assert cu_seqlens_k.shape == (L + 1,), (
                f"cu_seqlens_k must have shape ({L + 1},), got {cu_seqlens_k.shape}"
            )
            K = total_K
            dc_shape = (L, M, N)
            dc_ndim = 3
        else:
            # Normal case - all tensors must be 3D
            GemmWrapperBase.validate_tensor(A, "A", 3)
            GemmWrapperBase.validate_tensor(B, "B", 3)
            L, M, K = A.shape
            _, N, K_B = B.shape
            assert K == K_B, f"K dimension mismatch: A has {K}, B has {K_B}"
            GemmWrapperBase.validate_shape(B, (L, N, K), "B")
            dc_shape = (L, M, N)
            dc_ndim = 3

        # Validate D and C shapes uniformly
        for tensor, name in [(D, "D"), (C, "C")]:
            if tensor is not None:
                assert tensor.dim() == dc_ndim, (
                    f"{name} must be {dc_ndim}D for this mode, got {tensor.dim()}D"
                )
                assert tensor.shape == dc_shape, (
                    f"{name} shape {tensor.shape} doesn't match expected {dc_shape}"
                )

        tensors = {
            "A": GemmTensorInfo(A),
            "B": GemmTensorInfo(B),
            "D": GemmTensorInfo(D),
            "C": GemmTensorInfo(C),
        }

        if additional_tensors:
            for name, tensor in additional_tensors.items():
                if tensor is not None:
                    assert tensor.dim() == dc_ndim, (
                        f"{name} must be {dc_ndim}D for this mode, got {tensor.dim()}D"
                    )
                    assert tensor.shape == dc_shape, (
                        f"{name} shape {tensor.shape} doesn't match expected {dc_shape}"
                    )
                tensors[name] = GemmTensorInfo(tensor)

        return L, M, K, N, tensors

    @staticmethod
    def permute_tensors(
        tensors: Dict[str, GemmTensorInfo],
        varlen_m: bool = False,
        varlen_k: bool = False,
    ) -> None:
        # Determine which tensors need permutation
        if varlen_m:
            # Only B needs permutation (3D tensor)
            tensors_to_permute = ["B"]
        elif varlen_k:
            # Only D and C need permutation (3D tensors)
            tensors_to_permute = ["D", "C"]
        else:
            # All tensors need permutation
            tensors_to_permute = None

        # Apply permutation from (L, *, *) -> (*, *, L) for selected tensors
        for name, info in tensors.items():
            if info.tensor is not None and info.tensor.ndim == 3:
                if tensors_to_permute is None or name in tensors_to_permute:
                    info.tensor = info.tensor.permute(1, 2, 0)

    @staticmethod
    def extract_dtypes(tensors: Dict[str, GemmTensorInfo]) -> None:
        for name, info in tensors.items():
            if info.tensor is not None:
                info.dtype = torch2cute_dtype_map[info.tensor.dtype]

    @staticmethod
    def determine_major_orders(
        tensors: Dict[str, GemmTensorInfo],
        major_configs: Dict[str, Tuple[str, str, str]],
    ) -> None:
        for name, dims in major_configs.items():
            if name in tensors and tensors[name].tensor is not None:
                tensors[name].major = GemmWrapperBase.get_major_order(tensors[name].tensor, dims)

    @staticmethod
    def create_cute_tensors(
        tensors: Dict[str, GemmTensorInfo],
        major_configs: Dict[str, Tuple[str, str, str]],
    ) -> None:
        for name, info in tensors.items():
            if info.tensor is not None and name in major_configs:
                info.cute_tensor = GemmWrapperBase.create_cute_tensor(
                    info.tensor, info.major, major_configs[name]
                )

    @staticmethod
    def create_scheduler_args(
        max_active_clusters: int,
        tile_count_semaphore: Optional[Tensor] = None,
        batch_idx_permute: Optional[Tensor] = None,
        max_swizzle_size: int = 8,
    ) -> TileSchedulerOptions:
        return TileSchedulerOptions(
            Int32(max_active_clusters),
            tile_count_semaphore=(
                make_ptr(
                    Int32,
                    tile_count_semaphore.data_ptr(),
                    cute.AddressSpace.gmem,
                    assumed_align=4,
                )
                if tile_count_semaphore is not None
                else None
            ),
            batch_idx_permute=(
                (from_dlpack(batch_idx_permute, assumed_align=4).mark_layout_dynamic(leading_dim=0))
                if batch_idx_permute is not None
                else None
            ),
            max_swizzle_size=Int32(max_swizzle_size),
        )

    @staticmethod
    def create_varlen_args(
        cu_seqlens_m: Optional[Tensor],
        cu_seqlens_k: Optional[Tensor],
        A_idx: Optional[Tensor],
        max_active_clusters: int,
        cluster_shape_mnk: Tuple[int, int, int],
        tensors: Dict[str, GemmTensorInfo],
        num_epi_tensormaps: int = 0,
        pingpong: bool = False,
    ) -> Optional[Any]:
        if cu_seqlens_m is None and cu_seqlens_k is None:
            return None
        # When varlen_m, we assume persistent=True
        # Grid size depends on num_active_clusters and cluster size
        cluster_size = cluster_shape_mnk[0] * cluster_shape_mnk[1]
        num_blocks = max_active_clusters * cluster_size
        # Calculate number of tensormaps needed
        if cu_seqlens_m is not None:
            # For varlen_m: need tensormaps for D and epilogue tensors
            num_tensormaps = num_epi_tensormaps * (1 if not pingpong else 2)
            if tensors["D"].tensor is not None:
                num_tensormaps += 1 if not pingpong else 2  # D tensormap
        else:
            # For varlen_k: need tensormaps for A & B
            num_tensormaps = 2 if A_idx is None else 1
        # Create tensormap buffer (each tensormap is 128 bytes = 16 int64s)
        tensormap_size = 128 // 8  # 16 int64s
        if num_tensormaps > 0:
            device = cu_seqlens_m.device if cu_seqlens_m is not None else cu_seqlens_k.device
            tensormaps = torch.empty(
                (num_blocks, num_tensormaps, tensormap_size),
                dtype=torch.int64,
                device=device,
            )
            tensormaps_cute = from_dlpack(tensormaps, assumed_align=128).mark_compact_shape_dynamic(
                mode=0, stride_order=(0, 1, 2)
            )
        else:
            tensormaps_cute = None

        return VarlenArguments(
            mCuSeqlensM=(
                from_dlpack(cu_seqlens_m, assumed_align=4).mark_layout_dynamic(leading_dim=0)
                if cu_seqlens_m is not None
                else None
            ),
            mCuSeqlensK=(
                from_dlpack(cu_seqlens_k, assumed_align=4).mark_layout_dynamic(leading_dim=0)
                if cu_seqlens_k is not None
                else None
            ),
            mTensormaps=tensormaps_cute,
            mAIdx=(
                from_dlpack(A_idx, assumed_align=4).mark_layout_dynamic(leading_dim=0)
                if A_idx is not None
                else None
            ),
        )

    @staticmethod
    def get_compile_key(
        tensors: Dict[str, GemmTensorInfo],
        activation: Optional[str],
        tile_shape_mn: Tuple[int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        pingpong: bool,
        persistent: bool,
        has_semaphore: bool,
        *args,
        key_tensor_names: Tuple[str, ...] = ("A", "B", "D", "C"),
    ) -> Tuple:
        key_parts = []
        for name in key_tensor_names:
            if name in tensors:
                key_parts.append(tensors[name].dtype)
        key_parts.append(activation)
        key_parts.extend([tile_shape_mn, cluster_shape_mnk])
        for name in key_tensor_names:
            if name in tensors:
                key_parts.append(tensors[name].major)
        key_parts.extend([pingpong, persistent, has_semaphore])
        key_parts.extend(args)
        return tuple(key_parts)
