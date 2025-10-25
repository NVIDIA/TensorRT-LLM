from dataclasses import dataclass
from functools import partial
from typing import Optional, Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, const_expr

from tensorrt_llm._torch.auto_deploy.custom_ops.agent_ops.cutedsl_gemm import copy_utils, utils

from .cute_dsl_utils import ArgumentsBase, ParamsBase
from .gemm_sm90 import GemmSm90
from .sm90_utils import partition_for_epilogue
from .varlen_utils import VarlenManager


class GemmDefaultEpiMixin:
    num_epi_tensormaps: int = 0

    @dataclass
    class EpilogueArguments(ArgumentsBase):
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None
        add_to_output: bool = False

    @dataclass
    class EpilogueParams(ParamsBase):
        alpha: Optional[Float32 | cute.Tensor] = None
        beta: Optional[Float32 | cute.Tensor] = None
        mRowVecBroadcast: Optional[cute.Tensor] = None
        mColVecBroadcast: Optional[cute.Tensor] = None

    def epi_to_underlying_arguments(
        self, args: EpilogueArguments, *, loc=None, ip=None
    ) -> EpilogueParams:
        # Assume all strides are divisible by 32 bits except the last stride
        def new_stride(t):
            return tuple(
                (cute.assume(s, divby=32 // t.element_type.width) if not cute.is_static(s) else s)
                for s in t.stride
            )

        mRowVecBroadcast, mColVecBroadcast = [
            (
                cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t)))
                if t is not None
                else None
            )
            for t in (args.mRowVecBroadcast, args.mColVecBroadcast)
        ]
        return self.EpilogueParams(
            alpha=args.alpha,
            beta=args.beta,
            mRowVecBroadcast=mRowVecBroadcast,
            mColVecBroadcast=mColVecBroadcast,
        )

    @cute.jit
    def epi_begin(
        self,
        params: EpilogueParams,
        epi_smem_tensors: Tuple[cute.Tensor, ...],
        epi_tile: cute.Tile,
        tiled_copy_t2r: Optional[cute.TiledCopy],
        tiled_copy_r2s: cute.TiledCopy,
        tile_coord_mnkl: cute.Coord,
        varlen_manager: VarlenManager,
        epilogue_barrier: cutlass.pipeline.NamedBarrier,
        tidx: Int32,
    ):
        alpha, beta = None, None
        if const_expr(hasattr(params, "alpha") and params.alpha is not None):
            alpha = utils.load_scalar_or_pointer(params.alpha)
        if const_expr(hasattr(params, "beta") and params.beta is not None):
            beta = utils.load_scalar_or_pointer(params.beta)
        sRowVec, sColVec, *rest = epi_smem_tensors
        tile_M, tile_N = self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]
        batch_idx = tile_coord_mnkl[3]
        num_epi_threads = self.num_epi_warps * cute.arch.WARP_SIZE
        # Don't need sync as we assume the previous epilogue has finished

        partition_for_epilogue_fn = partial(
            partition_for_epilogue,
            epi_tile=epi_tile,
            tiled_copy=tiled_copy_t2r if tiled_copy_t2r is not None else tiled_copy_r2s,
            tidx=tidx,
            reference_src=tiled_copy_t2r is None,
        )

        tDsRowVec = None
        if const_expr(params.mRowVecBroadcast is not None):
            rowvec_dtype = params.mRowVecBroadcast.element_type
            num_copy_elems = const_expr(max(32, rowvec_dtype.width)) // rowvec_dtype.width
            thr_copy_RV = copy_utils.tiled_copy_1d(
                params.mRowVecBroadcast.element_type,
                num_epi_threads,
                num_copy_elems,
                is_async=True,
            ).get_slice(tidx)
            mRowVec = params.mRowVecBroadcast[batch_idx, None]
            gRowVec = cute.local_tile(mRowVec, (tile_N,), (tile_coord_mnkl[1],))
            tRVgRV = thr_copy_RV.partition_S(gRowVec)
            tRVsRV = thr_copy_RV.partition_D(sRowVec)
            tRVcRV = thr_copy_RV.partition_S(cute.make_identity_tensor(tile_N))
            limit_n = min(mRowVec.shape[0] - tile_coord_mnkl[1] * tile_N, tile_N)
            tRVpRV = cute.make_fragment((1, cute.size(tRVsRV.shape[1])), Boolean)
            for m in cutlass.range(cute.size(tRVsRV.shape[1]), unroll_full=True):
                tRVpRV[0, m] = tRVcRV[0, m] < limit_n
            cute.copy(thr_copy_RV, tRVgRV, tRVsRV, pred=tRVpRV)
            # (CPY, CPY_M, CPY_N, EPI_M, EPI_N)
            tDsRowVec = partition_for_epilogue_fn(
                cute.make_tensor(
                    sRowVec.iterator, cute.make_layout((tile_M, tile_N), stride=(0, 1))
                )
            )
            if const_expr(tiled_copy_t2r is not None):
                tDsRowVec = tiled_copy_r2s.retile(tDsRowVec)

        tDsColVec = None
        if const_expr(params.mColVecBroadcast is not None):
            colvec_dtype = params.mColVecBroadcast.element_type
            num_copy_elems = const_expr(max(32, colvec_dtype.width)) // colvec_dtype.width
            thr_copy_CV = copy_utils.tiled_copy_1d(
                params.mColVecBroadcast.element_type,
                num_epi_threads,
                num_copy_elems,
                is_async=True,
            ).get_slice(tidx)
            if const_expr(not varlen_manager.varlen_m):
                mColVec = params.mColVecBroadcast[batch_idx, None]
            else:
                mColVec = cute.domain_offset(
                    (varlen_manager.params.cu_seqlens_m[batch_idx],),
                    params.mColVecBroadcast,
                )
            gColVec = cute.local_tile(mColVec, (tile_M,), (tile_coord_mnkl[0],))
            tCVgCV = thr_copy_CV.partition_S(gColVec)
            tCVsCV = thr_copy_CV.partition_D(sColVec)
            tCVcCV = thr_copy_CV.partition_S(cute.make_identity_tensor(tile_M))
            limit_m = min(varlen_manager.len_m(batch_idx) - tile_coord_mnkl[0] * tile_M, tile_M)
            tCVpCV = cute.make_fragment((1, cute.size(tCVsCV.shape[1])), Boolean)
            for m in cutlass.range(cute.size(tCVsCV.shape[1]), unroll_full=True):
                tCVpCV[0, m] = tCVcCV[0, m] < limit_m
            cute.copy(thr_copy_CV, tCVgCV, tCVsCV, pred=tCVpCV)
            tDsColVec = partition_for_epilogue_fn(
                cute.make_tensor(
                    sColVec.iterator, cute.make_layout((tile_M, tile_N), stride=(1, 0))
                )
            )
            if const_expr(tiled_copy_t2r is not None):
                tDsColVec = tiled_copy_r2s.retile(tDsColVec)

        if const_expr(params.mRowVecBroadcast is not None or params.mColVecBroadcast is not None):
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            epilogue_barrier.arrive_and_wait()
        return alpha, beta, tDsRowVec, tDsColVec

    def epi_begin_loop(self, params: EpilogueParams, epi_tensors, epi_coord: cute.Coord):
        alpha, beta, tDsRowVec, tDsColVec = epi_tensors
        tDrRowVec_cvt = None
        if const_expr(tDsRowVec is not None):
            tDsRowVec_cur = cute.group_modes(tDsRowVec, 3, cute.rank(tDsRowVec))[
                None, None, None, epi_coord
            ]
            # tDrRowVec = cute.make_fragment_like(tDsRowVec_cur)
            tDrRowVec = cute.make_fragment(tDsRowVec_cur.layout, tDsRowVec_cur.element_type)
            cute.autovec_copy(cute.filter_zeros(tDsRowVec_cur), cute.filter_zeros(tDrRowVec))
            tDrRowVec_cvt = cute.make_fragment_like(tDrRowVec, self.acc_dtype)
            tDrRowVec_cvt.store(tDrRowVec.load().to(self.acc_dtype))
        tDrColVec_cvt = None
        if const_expr(tDsColVec is not None):
            tDsColVec_cur = cute.group_modes(tDsColVec, 3, cute.rank(tDsColVec))[
                None, None, None, epi_coord
            ]
            # This somehow doesn't work, some dim with stride 0 turns to non-zero stride
            # tDrRowVec = cute.make_fragment_like(tDsRowVec_cur)
            tDrColVec = cute.make_fragment(tDsColVec_cur.layout, tDsColVec_cur.element_type)
            cute.autovec_copy(cute.filter_zeros(tDsColVec_cur), cute.filter_zeros(tDrColVec))
            tDrColVec_cvt = cute.make_fragment_like(tDrColVec, self.acc_dtype)
            tDrColVec_cvt.store(tDrColVec.load().to(self.acc_dtype))
        return alpha, beta, tDrRowVec_cvt, tDrColVec_cvt

    @cute.jit
    def epi_visit_subtile(
        self,
        params: EpilogueParams,
        epi_loop_tensors: Tuple[cute.Tensor, ...],
        tRS_rD: cute.Tensor,
        tRS_rC: Optional[cute.Tensor] = None,
    ) -> Optional[cute.Tensor]:
        alpha, beta, tDrRowVec, tDrColVec = epi_loop_tensors
        rD = tRS_rD.load()
        # Apply alpha scaling to accumulator if alpha is provided (not None)
        if const_expr(hasattr(params, "alpha") and params.alpha is not None):
            alpha = utils.load_scalar_or_pointer(params.alpha)
            rD *= alpha
        # Apply C with beta scaling
        if const_expr(tRS_rC is not None):
            if const_expr(not hasattr(params, "beta") or params.beta is None):
                # beta is None, default behavior: add C (beta=1.0)
                rD += tRS_rC.load().to(tRS_rD.element_type)
            else:
                beta = utils.load_scalar_or_pointer(params.beta)
                rD += beta * tRS_rC.load().to(tRS_rD.element_type)
        tRS_rD.store(rD)
        if const_expr(tDrRowVec is not None):
            for i in cutlass.range(cute.size(tDrRowVec), unroll_full=True):
                tRS_rD[i] += tDrRowVec[i]
        if const_expr(tDrColVec is not None):
            for i in cutlass.range(cute.size(tDrColVec), unroll_full=True):
                tRS_rD[i] += tDrColVec[i]
        return None

    @staticmethod
    def epi_smem_bytes_per_stage(
        args: Optional[EpilogueArguments],
        cta_tile_shape_mnk: Tuple[int, int, int],
        epi_tile: cute.Tile,
    ) -> int:
        row_vec_smem_size = 0 if args.mRowVecBroadcast is None else cta_tile_shape_mnk[1]
        col_vec_smem_size = 0 if args.mColVecBroadcast is None else cta_tile_shape_mnk[0]
        row_vec_dtype = (
            args.mRowVecBroadcast.element_type if args.mRowVecBroadcast is not None else Float32
        )
        col_vec_dtype = (
            args.mColVecBroadcast.element_type if args.mColVecBroadcast is not None else Float32
        )
        return (
            row_vec_smem_size * row_vec_dtype.width + col_vec_smem_size * col_vec_dtype.width
        ) // 8

    def epi_get_smem_struct(self, params: EpilogueParams):
        row_vec_smem_size = 0 if params.mRowVecBroadcast is None else self.cta_tile_shape_mnk[1]
        col_vec_smem_size = 0 if params.mColVecBroadcast is None else self.cta_tile_shape_mnk[0]
        row_vec_dtype = (
            params.mRowVecBroadcast.element_type if params.mRowVecBroadcast is not None else Float32
        )
        col_vec_dtype = (
            params.mColVecBroadcast.element_type if params.mColVecBroadcast is not None else Float32
        )

        @cute.struct
        class EpiSharedStorage:
            sRowVec: cute.struct.Align[cute.struct.MemRange[row_vec_dtype, row_vec_smem_size], 16]
            sColVec: cute.struct.Align[cute.struct.MemRange[col_vec_dtype, col_vec_smem_size], 16]

        return EpiSharedStorage

    def epi_get_smem_tensors(self, params: EpilogueParams, storage) -> Tuple[cute.Tensor, ...]:
        sRowVec = None
        if const_expr(params.mRowVecBroadcast is not None):
            sRowVec = storage.epi.sRowVec.get_tensor(cute.make_layout(self.cta_tile_shape_mnk[1]))
        sColVec = None
        if const_expr(params.mColVecBroadcast is not None):
            sColVec = storage.epi.sColVec.get_tensor(cute.make_layout(self.cta_tile_shape_mnk[0]))
        return (sRowVec, sColVec)


class GemmDefaultSm90(GemmDefaultEpiMixin, GemmSm90):
    pass
