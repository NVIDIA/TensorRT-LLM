from dataclasses import dataclass
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Boolean, Int32, const_expr
from cutlass.utils import TensorMapManager, TensorMapUpdateMode


@dataclass(frozen=True)
class TensorMapManagerSm90(TensorMapManager):
    """
    We have to subclass cutlass.utils.TensorMapManager bc it takes in warp_id and only
    perform the operation if warp_id matches the current warp.
    But for Hopper pingpong gemm we want to call it with warp_id 0 and 4.
    So we take in a boolean `is_manager_warp` to determine whether to perform the operation or not.
    """

    @cute.jit
    def init_tensormap_from_atom(
        self, copy_atom: cute.CopyAtom, dst_ptr: cute.Pointer, is_manager_warp: Boolean
    ) -> None:
        if is_manager_warp:
            with cute.arch.elect_one():
                cute.nvgpu.cpasync.copy_tensormap(copy_atom, dst_ptr)
        cute.arch.sync_warp()
        return

    @cute.jit
    def update_tensormap(
        self,
        tensor_gmem: Tuple[cute.Tensor, ...],
        tma_copy_atom: Tuple[cute.CopyAtom, ...],
        tensormap_gmem_ptr: Tuple[cute.Pointer, ...],
        is_manager_warp: Boolean,
        tensormap_smem_ptr: Tuple[cute.Pointer, ...],
    ) -> None:
        # updates before touching tensormap in global memory
        if is_manager_warp:
            if const_expr(self.tensormap_update_mode == TensorMapUpdateMode.SMEM):
                for copy_atom, tensor, smem_ptr in zip(
                    tma_copy_atom, tensor_gmem, tensormap_smem_ptr
                ):
                    cute.nvgpu.cpasync.update_tma_descriptor(copy_atom, tensor, smem_ptr)
            # wait until it's safe to update tensormap in global memory
            with cute.arch.elect_one():
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.sync_warp()
            # updates to tensormap in global memory
            if const_expr(self.tensormap_update_mode == TensorMapUpdateMode.SMEM):
                for gmem_ptr, smem_ptr in zip(tensormap_gmem_ptr, tensormap_smem_ptr):
                    cute.nvgpu.cpasync.cp_fence_tma_desc_release(gmem_ptr, smem_ptr)
            else:
                for copy_atom, tensor, gmem_ptr in zip(
                    tma_copy_atom, tensor_gmem, tensormap_gmem_ptr
                ):
                    cute.nvgpu.cpasync.update_tma_descriptor(copy_atom, tensor, gmem_ptr)
                cute.arch.sync_warp()
                cute.nvgpu.cpasync.fence_tma_desc_release()

    @cute.jit
    def update_tensormap_shape(
        self,
        tensormap_gmem_ptr: Tuple[cute.Pointer, ...],
        is_manager_warp: Boolean,
        tensormap_smem_ptr: Tuple[cute.Pointer, ...],
        shapes: Tuple[Int32, ...],
        orders: cutlass.Constexpr[Tuple[int, ...]],
    ) -> None:
        # updates before touching tensormap in global memory
        if is_manager_warp:
            if const_expr(self.tensormap_update_mode == TensorMapUpdateMode.SMEM):
                for smem_ptr, shape, order in zip(tensormap_smem_ptr, shapes, orders):
                    smem_ptr_i32 = smem_ptr.toint().ir_value()
                    llvm.inline_asm(
                        None,
                        [
                            smem_ptr_i32,
                            Int32(shape).ir_value(),
                            Int32(order).ir_value(),
                        ],
                        "{\n\t"
                        ".reg .b64 smem_ptr_i64;\n\t"
                        "cvt.u64.u32 smem_ptr_i64, $0;\n\t"
                        f"tensormap.replace.tile.global_dim.shared::cta.b1024.b32 [smem_ptr_i64], {order}, $1;\n\t"
                        "}\n",
                        "r,r",
                        has_side_effects=True,
                        is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT,
                    )
            # wait until it's safe to update tensormap in global memory
            with cute.arch.elect_one():
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
            cute.arch.sync_warp()
            # updates to tensormap in global memory
            if const_expr(self.tensormap_update_mode == TensorMapUpdateMode.SMEM):
                for gmem_ptr, smem_ptr in zip(tensormap_gmem_ptr, tensormap_smem_ptr):
                    cute.nvgpu.cpasync.cp_fence_tma_desc_release(gmem_ptr, smem_ptr)
            else:
                assert len(shapes) == len(orders) == len(tensormap_gmem_ptr)
                for gmem_ptr, shape, order in zip(tensormap_gmem_ptr, shapes, orders):
                    gmem_ptr_i64 = gmem_ptr.toint().ir_value()
                    llvm.inline_asm(
                        None,
                        [
                            gmem_ptr_i64,
                            Int32(shape).ir_value(),
                            Int32(order).ir_value(),
                        ],
                        f"tensormap.replace.tile.global_dim.global.b1024.b32 [$0], {order}, $1;",
                        "l,r",
                        has_side_effects=True,
                        is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT,
                    )
                cute.arch.sync_warp()
                cute.nvgpu.cpasync.fence_tma_desc_release()
