from dataclasses import dataclass
from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Int32, const_expr
from cutlass.utils import LayoutEnum

from .cute_dsl_utils import ArgumentsBase, ParamsBase
from .tensormap_manager import TensorMapManagerSm90


# Grouping arguments together that should be passed to __call__
@dataclass
class VarlenArguments(ArgumentsBase):
    mCuSeqlensM: Optional[cute.Tensor] = None
    mCuSeqlensK: Optional[cute.Tensor] = None
    mTensormaps: Optional[cute.Tensor] = None
    mAIdx: Optional[cute.Tensor] = None


class VarlenManager:
    bytes_per_tensormap = 128

    @dataclass
    class Params(ParamsBase):
        cu_seqlens_m: Optional[cute.Tensor] = None
        cu_seqlens_k: Optional[cute.Tensor] = None
        tensormaps: Optional[cute.Tensor] = None
        mAIdx: Optional[cute.Tensor] = None

        @staticmethod
        @cute.jit
        def create(args: VarlenArguments, *, loc=None, ip=None) -> "VarlenManager.Params":
            return VarlenManager.Params(
                cu_seqlens_m=args.mCuSeqlensM,
                cu_seqlens_k=args.mCuSeqlensK,
                tensormaps=args.mTensormaps,
                mAIdx=args.mAIdx,
            )

    def __init__(
        self,
        params: Params,
        tensormap_manager: Optional[cutlass.utils.TensorMapManager],
        tensormap_a_ptr: Optional[cute.Pointer],
        tensormap_b_ptr: Optional[cute.Pointer],
        tensormap_d_ptr: Optional[cute.Pointer],
        tensormap_epi_ptrs: list[Optional[cute.Pointer]],
        len_m_static: Int32,
        len_k_static: Int32,
        last_batch_idx: Int32 = Int32(-1),
        is_group_changed: Boolean = Boolean(True),
        *,
        loc=None,
        ip=None,
    ):
        self.params = params
        self.tensormap_manager = tensormap_manager
        self._tensormap_a_ptr = tensormap_a_ptr
        self._tensormap_b_ptr = tensormap_b_ptr
        self._tensormap_d_ptr = tensormap_d_ptr
        self._tensormap_epi_ptrs = tensormap_epi_ptrs
        self._len_m_static = len_m_static
        self._len_k_static = len_k_static
        self._last_batch_idx = last_batch_idx
        self._is_group_changed = is_group_changed
        self.varlen_m = const_expr(params.cu_seqlens_m is not None)
        self.varlen_k = const_expr(params.cu_seqlens_k is not None)
        self.gather_A = const_expr(params.mAIdx is not None)
        self._loc = loc
        self._ip = ip

    @staticmethod
    def to_underlying_arguments(args: VarlenArguments, *, loc=None, ip=None) -> Params:
        assert not (args.mCuSeqlensM is not None and args.mCuSeqlensK is not None), (
            "Only support either varlen_m or varlen_k"
        )
        return VarlenManager.Params.create(args, loc=loc, ip=ip)

    @staticmethod
    @cute.jit
    def create(
        params: Params,
        has_D: bool,
        num_epi_tensormaps: int,
        len_m_static: Int32,
        len_k_static: Int32,
        pingpong: bool = False,
        warp_idx: int | Int32 = 0,
        *,
        loc=None,
        ip=None,
    ) -> "VarlenManager":
        tensormap_manager = None
        tensormap_a_ptr, tensormap_b_ptr, tensormap_d_ptr = None, None, None
        tensormap_epi_ptrs = [None] * num_epi_tensormaps
        varlen_m = const_expr(params.cu_seqlens_m is not None)
        varlen_k = const_expr(params.cu_seqlens_k is not None)
        if const_expr(varlen_m or varlen_k):
            tensormap_manager = TensorMapManagerSm90(
                cutlass.utils.TensorMapUpdateMode.GMEM,
                VarlenManager.bytes_per_tensormap,
            )
            # equivalent to bidx + bidy * gridDim.x + bidxz * gridDim.x * gridDim.y
            tensormap_workspace_idx = cute.make_layout(cute.arch.grid_dim())(cute.arch.block_idx())
            if const_expr(varlen_m):
                tensormap_d_idx = warp_idx // 4 if const_expr(pingpong) else 0
                tensormap_epi_offset = tensormap_d_idx
                if const_expr(has_D):
                    tensormap_d_ptr = tensormap_manager.get_tensormap_ptr(
                        params.tensormaps[tensormap_workspace_idx, tensormap_d_idx, None].iterator
                    )
                    tensormap_epi_offset += 1 if not pingpong else 2
                tensormap_epi_ptrs = [
                    tensormap_manager.get_tensormap_ptr(
                        params.tensormaps[
                            tensormap_workspace_idx,
                            tensormap_epi_offset + i * (1 if not pingpong else 2),
                            None,
                        ].iterator
                    )
                    for i in range(num_epi_tensormaps)
                ]
            else:
                assert varlen_k
                gather_A = const_expr(params.mAIdx is not None)
                if const_expr(not gather_A):
                    tensormap_a_ptr = tensormap_manager.get_tensormap_ptr(
                        params.tensormaps[tensormap_workspace_idx, 0, None].iterator
                    )
                tensormap_b_ptr = tensormap_manager.get_tensormap_ptr(
                    params.tensormaps[
                        tensormap_workspace_idx, 1 if not gather_A else 0, None
                    ].iterator
                )
        return VarlenManager(
            params,
            tensormap_manager,
            tensormap_a_ptr,
            tensormap_b_ptr,
            tensormap_d_ptr,
            tensormap_epi_ptrs,
            len_m_static=len_m_static,
            len_k_static=len_k_static,
        )

    def len_m(self, batch_idx: Int32) -> Int32:
        if const_expr(self.varlen_m):
            return self.params.cu_seqlens_m[batch_idx + 1] - self.params.cu_seqlens_m[batch_idx]
        else:
            return self._len_m_static

    def len_k(self, batch_idx: Int32) -> Int32:
        if const_expr(self.varlen_k):
            return self.params.cu_seqlens_k[batch_idx + 1] - self.params.cu_seqlens_k[batch_idx]
        else:
            return self._len_k_static

    def offset_batch_A(self, mA_mkl: cute.Tensor, batch_idx: Int32) -> cute.Tensor:
        params = self.params
        if const_expr(self.varlen_m):
            mA_mk = cute.domain_offset((params.cu_seqlens_m[batch_idx], 0), mA_mkl)
        elif const_expr(self.varlen_k):
            mA_mk = cute.domain_offset((0, params.cu_seqlens_k[batch_idx]), mA_mkl)
        else:
            mA_mk = mA_mkl[None, None, batch_idx]
        return mA_mk

    def offset_batch_AIdx(self, batch_idx: Int32) -> cute.Tensor:
        params = self.params
        if const_expr(self.varlen_m):
            mAIdx_mk = cute.domain_offset((params.cu_seqlens_m[batch_idx],), params.mAIdx)
        elif const_expr(self.varlen_k):
            mAIdx_mk = cute.domain_offset((params.cu_seqlens_k[batch_idx],), params.mAIdx)
        else:
            mAIdx_mk = params.mAIdx[None, batch_idx]
        return mAIdx_mk

    def offset_batch_B(self, mB_nkl: cute.Tensor, batch_idx: Int32) -> cute.Tensor:
        params = self.params
        if const_expr(self.varlen_k):
            mB_nk = cute.domain_offset((0, params.cu_seqlens_k[batch_idx]), mB_nkl)
        else:
            mB_nk = mB_nkl[None, None, batch_idx]
        return mB_nk

    def offset_batch_epi(self, mD_mnl: cute.Tensor, batch_idx: Int32) -> cute.Tensor:
        params = self.params
        if const_expr(self.varlen_m):
            mD_mn = cute.domain_offset((params.cu_seqlens_m[batch_idx], 0), mD_mnl)
        else:
            mD_mn = mD_mnl[None, None, batch_idx]
        return mD_mn

    def init_tensormap_AB(
        self,
        tma_atom_a: Optional[cute.CopyAtom],
        tma_atom_b: cute.CopyAtom,
        is_manager_warp: bool | Boolean = True,
    ) -> None:
        if const_expr(self.varlen_k):
            if const_expr(not self.gather_A):
                self.tensormap_manager.init_tensormap_from_atom(
                    tma_atom_a, self._tensormap_a_ptr, is_manager_warp
                )
            self.tensormap_manager.init_tensormap_from_atom(
                tma_atom_b, self._tensormap_b_ptr, is_manager_warp
            )

    def init_tensormap_epi(
        self,
        tma_atom_d: Optional[cute.CopyAtom],
        tma_atoms_epi: list[cute.CopyAtom],
        is_manager_warp: bool | Boolean = True,
    ) -> None:
        if const_expr(self.varlen_m):
            if const_expr(self._tensormap_d_ptr is not None):
                self.tensormap_manager.init_tensormap_from_atom(
                    tma_atom_d, self._tensormap_d_ptr, is_manager_warp
                )
            for tma_atom, tensormap_epi_ptr in zip(tma_atoms_epi, self._tensormap_epi_ptrs):
                self.tensormap_manager.init_tensormap_from_atom(
                    tma_atom, tensormap_epi_ptr, is_manager_warp
                )

    def fence_tensormap_init(self) -> None:
        self.tensormap_manager.fence_tensormap_initialization()

    @cute.jit
    def update_tensormap_AB(
        self,
        batch_idx: Int32,
        a_layout: LayoutEnum,
        b_layout: LayoutEnum,
        is_manager_warp: bool | Boolean = True,
    ) -> None:
        if const_expr(self.varlen_k):
            self._is_group_changed = Boolean(batch_idx != self._last_batch_idx)
            self._last_batch_idx = batch_idx
            if self._is_group_changed:
                # construct tensor A/B based on real address, shape and stride information
                cu_seqlens_k = self.params.cu_seqlens_k
                tensormap_ptrs = [self._tensormap_b_ptr]
                shapes = [cu_seqlens_k[batch_idx + 1]]
                orders = [0 if const_expr(b_layout == LayoutEnum.ROW_MAJOR) else 1]
                if const_expr(not self.gather_A):
                    tensormap_ptrs.insert(0, self._tensormap_a_ptr)
                    shapes.insert(0, cu_seqlens_k[batch_idx + 1])
                    orders.insert(0, 0 if const_expr(a_layout == LayoutEnum.ROW_MAJOR) else 1)
                self.tensormap_manager.update_tensormap_shape(
                    tensormap_ptrs,
                    is_manager_warp=is_manager_warp,
                    shapes=shapes,
                    orders=orders,
                    tensormap_smem_ptr=None,
                )

    @cute.jit
    def update_tensormap_epi(
        self,
        batch_idx: Int32,
        d_layout: LayoutEnum,
        epi_shapes: list[Int32],
        epi_orders: list[int],
        is_manager_warp: bool | Boolean = True,
    ) -> None:
        if const_expr(self.varlen_m):
            self._is_group_changed = Boolean(batch_idx != self._last_batch_idx)
            self._last_batch_idx = batch_idx
            # Cute-DSL doesn't like this under if statement
            order_d = (
                (0 if const_expr(d_layout.is_m_major_c()) else 1) if d_layout is not None else None
            )
            if self._is_group_changed:
                # construct tensor A/B based on real address, shape and stride information
                cu_seqlens_m = self.params.cu_seqlens_m
                # construct tensor D based on real address, shape and stride information
                tensormap_ptrs, shapes, orders = [], [], []
                if const_expr(self._tensormap_d_ptr is not None):
                    tensormap_ptrs.append(self._tensormap_d_ptr)
                    shapes.append(cu_seqlens_m[batch_idx + 1])
                    orders.append(order_d)
                tensormap_ptrs.extend(self._tensormap_epi_ptrs)
                shapes.extend(epi_shapes)
                orders.extend(epi_orders)
                self.tensormap_manager.update_tensormap_shape(
                    tensormap_ptrs,
                    is_manager_warp=is_manager_warp,
                    shapes=shapes,
                    orders=orders,
                    tensormap_smem_ptr=None,
                )

    @cute.jit
    def fence_tensormap_update_AB(self, is_manager_warp: bool | Boolean = True) -> None:
        if const_expr(self.varlen_k):
            if self._is_group_changed and is_manager_warp:
                if const_expr(not self.gather_A):
                    self.tensormap_manager.fence_tensormap_update(self._tensormap_a_ptr)
                self.tensormap_manager.fence_tensormap_update(self._tensormap_b_ptr)

    @cute.jit
    def fence_tensormap_update_epi(self, is_manager_warp: bool | Boolean = True) -> None:
        if const_expr(self.varlen_m):
            if self._is_group_changed and is_manager_warp:
                if const_expr(self._tensormap_d_ptr is not None):
                    self.tensormap_manager.fence_tensormap_update(self._tensormap_d_ptr)
                for tensormap_epi_ptr in self._tensormap_epi_ptrs:
                    if const_expr(tensormap_epi_ptr is not None):
                        self.tensormap_manager.fence_tensormap_update(tensormap_epi_ptr)

    def get_tma_desc_a_ptr(self) -> Optional[cute.Pointer]:
        tma_desc_a_ptr = None
        if const_expr(self.varlen_k and self._tensormap_a_ptr is not None):
            tma_desc_a_ptr = self.tensormap_manager.get_tensormap_ptr(
                self._tensormap_a_ptr, cute.AddressSpace.generic
            )
        return tma_desc_a_ptr

    def get_tma_desc_b_ptr(self) -> Optional[cute.Pointer]:
        tma_desc_b_ptr = None
        if const_expr(self.varlen_k):
            tma_desc_b_ptr = self.tensormap_manager.get_tensormap_ptr(
                self._tensormap_b_ptr, cute.AddressSpace.generic
            )
        return tma_desc_b_ptr

    def get_tma_desc_d_ptr(self) -> Optional[cute.Pointer]:
        tma_desc_d_ptr = None
        if const_expr(self.varlen_m and self._tensormap_d_ptr is not None):
            tma_desc_d_ptr = self.tensormap_manager.get_tensormap_ptr(
                self._tensormap_d_ptr, cute.AddressSpace.generic
            )
        return tma_desc_d_ptr

    def get_tma_desc_epi_ptrs(self) -> list[Optional[cute.Pointer]]:
        tma_desc_epi_ptrs = [None] * len(self._tensormap_epi_ptrs)
        if const_expr(self.varlen_m):
            for i, tensormap_epi_ptr in enumerate(self._tensormap_epi_ptrs):
                if const_expr(tensormap_epi_ptr is not None):
                    tma_desc_epi_ptrs[i] = self.tensormap_manager.get_tensormap_ptr(
                        tensormap_epi_ptr, cute.AddressSpace.generic
                    )
        return tma_desc_epi_ptrs

    def __extract_mlir_values__(self):
        values, self._values_pos = [], []
        for obj in [
            self.params,
            self.tensormap_manager,
            self._tensormap_a_ptr,
            self._tensormap_b_ptr,
            self._tensormap_d_ptr,
            self._tensormap_epi_ptrs,
            self._len_m_static,
            self._len_k_static,
            self._last_batch_idx,
            self._is_group_changed,
        ]:
            obj_values = cutlass.extract_mlir_values(obj)
            values += obj_values
            self._values_pos.append(len(obj_values))
        return values

    def __new_from_mlir_values__(self, values):
        obj_list = []
        for obj, n_items in zip(
            [
                self.params,
                self.tensormap_manager,
                self._tensormap_a_ptr,
                self._tensormap_b_ptr,
                self._tensormap_d_ptr,
                self._tensormap_epi_ptrs,
                self._len_m_static,
                self._len_k_static,
                self._last_batch_idx,
                self._is_group_changed,
            ],
            self._values_pos,
        ):
            obj_list.append(cutlass.new_from_mlir_values(obj, values[:n_items]))
            values = values[n_items:]
        return self.__class__(*(tuple(obj_list)), loc=self._loc)
