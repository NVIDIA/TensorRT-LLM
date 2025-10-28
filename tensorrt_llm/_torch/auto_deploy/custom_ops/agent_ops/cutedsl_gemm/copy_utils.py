import math
from typing import Callable, Optional, Tuple, Type

import cutlass
import cutlass.cute as cute
import cutlass.pipeline
from cutlass import Boolean, Int32, const_expr
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import dsl_user_op


@dsl_user_op
def cvt_copy(
    atom: cute.CopyAtom,
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: Optional[cute.Tensor] = None,
    loc=None,
    ip=None,
    **kwargs,
) -> None:
    assert isinstance(src.iterator, cute.Pointer) and src.memspace == cute.AddressSpace.rmem
    if const_expr(src.element_type != dst.element_type):
        src_cvt = cute.make_fragment_like(src, dst.element_type)
        src_cvt.store(src.load().to(dst.element_type))
        src = src_cvt
    cute.copy(atom, src, dst, pred=pred, loc=loc, ip=ip, **kwargs)


@dsl_user_op
def get_copy_atom(
    dtype: Type[cutlass.Numeric],
    num_copy_elems: int,
    is_async: bool = False,
    *,
    loc=None,
    ip=None,
) -> cute.CopyAtom:
    num_copy_bits = const_expr(min(128, num_copy_elems * dtype.width))
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    return cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)


@dsl_user_op
def copy(
    src: cute.Tensor,
    dst: cute.Tensor,
    *,
    pred: Optional[cute.Tensor] = None,
    num_copy_elems: int = 1,
    is_async: bool = False,
    loc=None,
    ip=None,
    **kwargs,
) -> None:
    copy_atom = get_copy_atom(src.element_type, num_copy_elems, is_async)
    cute.copy(copy_atom, src, dst, pred=pred, loc=loc, ip=ip, **kwargs)


def tiled_copy_1d(
    dtype: Type[cutlass.Numeric],
    num_threads: int,
    num_copy_elems: int = 1,
    is_async: bool = False,
) -> cute.TiledCopy:
    num_copy_bits = num_copy_elems * dtype.width
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
    thr_layout = cute.make_layout(num_threads)
    val_layout = cute.make_layout(num_copy_elems)
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


def tiled_copy_2d(
    dtype: Type[cutlass.Numeric],
    major_mode_size: int,
    num_threads: int,
    is_async: bool = True,
) -> cute.TiledCopy:
    num_copy_bits = math.gcd(major_mode_size, 128 // dtype.width) * dtype.width
    copy_elems = num_copy_bits // dtype.width
    copy_op = cpasync.CopyG2SOp() if is_async else cute.nvgpu.CopyUniversalOp()
    copy_atom = cute.make_copy_atom(copy_op, dtype, num_bits_per_copy=num_copy_bits)
    gmem_threads_per_row = major_mode_size // copy_elems
    assert num_threads % gmem_threads_per_row == 0
    thr_layout = cute.make_ordered_layout(
        (num_threads // gmem_threads_per_row, gmem_threads_per_row),
        order=(1, 0),
    )
    val_layout = cute.make_layout((1, copy_elems))
    return cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)


def tma_get_copy_fn(
    atom: cute.CopyAtom,
    cta_coord: cute.Coord,
    cta_layout: cute.Layout,
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    filter_zeros: bool = False,
    **kwargs,
) -> Callable:
    src_is_smem = const_expr(
        isinstance(src_tensor.iterator, cute.Pointer)
        and src_tensor.memspace == cute.AddressSpace.smem
    )
    smem_tensor, gmem_tensor = (src_tensor, dst_tensor) if src_is_smem else (dst_tensor, src_tensor)
    # ((atom_v, rest_v), STAGE), ((atom_v, rest_v), RestK)
    s, g = cpasync.tma_partition(
        atom,
        cta_coord,
        cta_layout,
        cute.group_modes(smem_tensor, 0, cute.rank(smem_tensor) - 1),
        cute.group_modes(gmem_tensor, 0, cute.rank(gmem_tensor) - 1),
    )
    if const_expr(filter_zeros):
        s = cute.filter_zeros(s)
        g = cute.filter_zeros(g)
    src, dst = (s, g) if src_is_smem else (g, s)

    def copy_tma(src_idx, dst_idx, **new_kwargs):
        cute.copy(atom, src[None, src_idx], dst[None, dst_idx], **new_kwargs, **kwargs)

    return copy_tma, s, g


def tma_producer_copy_fn(copy: Callable, pipeline: cutlass.pipeline.PipelineAsync):
    def copy_fn(src_idx, producer_state: cutlass.pipeline.PipelineState, **new_kwargs):
        copy(
            src_idx=src_idx,
            dst_idx=producer_state.index,
            tma_bar_ptr=pipeline.producer_get_barrier(producer_state),
            **new_kwargs,
        )

    return copy_fn


@cute.jit
def gather_m_get_copy_fn(
    thr_copy_A: cute.core.ThrCopy,
    mA: cute.Tensor,  # (whatever, K)
    sA: cute.Tensor,  # (tile_M, tile_N, STAGE)
    gsAIdx: cute.Tensor,  # (tile_M), either gmem or smem
    limit_m: Int32,
    limit_k: Int32,
) -> Callable:
    tile_shape_mk = (cute.size(sA, mode=[0]), cute.size(sA, mode=[1]))
    tAsA = thr_copy_A.partition_D(sA)
    # k-major
    assert tAsA.shape[2] == 1
    tAsA = cute.group_modes(cute.slice_(tAsA, (None, None, 0, None)), 0, 2)

    is_even_m_smem = tile_shape_mk[0] % thr_copy_A.tiler_mn[0].shape == 0
    if const_expr(not is_even_m_smem):
        limit_m = min(limit_m, tile_shape_mk[0])
    elems_per_load = cute.size(tAsA.shape[0][0])
    cA = cute.make_identity_tensor(tile_shape_mk)
    tAcA = thr_copy_A.partition_S(cA)
    t0AcA = thr_copy_A.get_slice(0).partition_S(cA)
    # Instead of comparing tAcA to limit_m, we instead compare t0AcA to limit_m - tAcA[0][0]
    # since we know that tAcA[m][0] = t0AcA[m][0] + tAcA[0][0].
    # This is so that when we do the comparison, t0AcA is known at compile time.
    limit_m = limit_m - tAcA[0][0]
    limit_k = limit_k - tAcA[0][1]
    # Read and cache indices for A
    rows_per_thread = const_expr(cute.size(tAcA.shape, mode=[1]))
    cols_per_thread = const_expr(cute.size(tAcA.shape, mode=[2]))
    tApA_m = cute.make_fragment(rows_per_thread, Boolean)
    for m in cutlass.range(rows_per_thread, unroll_full=True):
        tApA_m[m] = t0AcA[0, m, 0][0] < limit_m
    m_idx = cute.make_fragment(rows_per_thread, Int32)
    for m in cutlass.range(rows_per_thread, unroll_full=True):
        row_idx = tAcA[0, m, 0][0]
        if tApA_m[m]:
            m_idx[m] = gsAIdx[row_idx]
        else:
            m_idx[m] = 0  # It's ok to load row 0 in the case of OOB

    mA_k = cute.logical_divide(mA, (None, tile_shape_mk[1]))

    def copy_fn(src_idx, dst_idx, pred: bool = False):
        tApA_k = None
        if const_expr(pred):
            tApA_k = cute.make_fragment(cols_per_thread, Boolean)
            limit_k_cur = limit_k - src_idx * tile_shape_mk[1]
            for k in cutlass.range(cols_per_thread, unroll_full=True):
                tApA_k[k] = t0AcA[0, 0, k][1] < limit_k_cur
        mA_cur = mA_k[None, (None, src_idx)]
        for m in cutlass.range_constexpr(tAcA.shape[1]):
            # cute.tiled_divide(mA_cur[m_idx[m], None], (elems_per_load,)) would give shape
            # ((elems_per_load), thread_per_row)
            # But we actually want shape ((elems_per_load, 1), thread_per_row) to match tAsA
            # So we append 1s to the last dimension and then do tiled_divide, then slice.
            mA_row = cute.tiled_divide(
                cute.append_ones(mA_cur[m_idx[m], None], up_to_rank=2),
                (elems_per_load, 1),
            )[None, None, 0]
            if const_expr(is_even_m_smem) or tApA_m[m]:
                # There's only 1 load per row
                assert cute.size(tAcA.shape, mode=[2]) == 1
                ki = tAcA[0, 0, 0][1] // elems_per_load
                cute.copy(thr_copy_A, mA_row[None, ki], tAsA[(None, m), dst_idx], pred=tApA_k)

    return copy_fn


@cute.jit
def gather_k_get_copy_fn(
    thr_copy_A: cute.core.ThrCopy,
    mA: cute.Tensor,  # (tile_M, whatever)
    sA: cute.Tensor,  # (tile_M, tile_N, STAGE)
    gsAIdx: cute.Tensor,  # (tile_K, RestK), either gmem or smem
    limit_m: Int32,
    limit_k: Int32,
) -> Callable:
    gAIdx, sAIdx = None, None
    if const_expr(gsAIdx.memspace == cute.AddressSpace.gmem):
        gAIdx = gsAIdx
    else:
        assert gsAIdx.memspace == cute.AddressSpace.smem
        sAIdx = gsAIdx
    tile_shape_mk = (cute.size(sA, mode=[0]), cute.size(sA, mode=[1]))
    # (atom_v, CPY_M, 1, STAGE)
    tAsA = thr_copy_A.partition_D(sA)
    # m-major
    tAsA = cute.group_modes(tAsA, 0, 3)

    is_even_m_smem = tile_shape_mk[0] % thr_copy_A.tiler_mn[0].shape == 0
    if const_expr(not is_even_m_smem):
        limit_m = min(limit_m, tile_shape_mk[0])
    elems_per_load = cute.size(tAsA.shape[0][0])
    cA = cute.make_identity_tensor(tile_shape_mk)
    tAcA = thr_copy_A.partition_S(cA)
    t0AcA = thr_copy_A.get_slice(0).partition_S(cA)
    # Instead of comparing tAcA to limit_m, we instead compare t0AcA to limit_m - tAcA[0][0]
    # since we know that tAcA[m][0] = t0AcA[m][0] + tAcA[0][0].
    # This is so that when we do the comparison, t0AcA is known at compile time.
    limit_m = limit_m - tAcA[0][0]
    limit_k = limit_k - tAcA[0][1]
    # Read and cache indices for A
    rows_per_thread = const_expr(cute.size(tAcA.shape, mode=[1]))
    cols_per_thread = const_expr(cute.size(tAcA.shape, mode=[2]))
    tApA_m = cute.make_fragment(rows_per_thread, Boolean)
    for m in cutlass.range(rows_per_thread, unroll_full=True):
        tApA_m[m] = t0AcA[0, m, 0][0] < limit_m
    threads_per_col = const_expr(thr_copy_A.tiler_mn[0].shape // elems_per_load)
    # This is very convoluted but idk a better way
    # for tile_M=128, flat_divide gives (8, 16, K),
    # then logical_divide gives ((8, 1), (8, 2), K).
    tidx = thr_copy_A.thr_idx
    tAmA = cute.logical_divide(
        cute.flat_divide(mA, (elems_per_load,)), (elems_per_load, threads_per_col)
    )[None, (tidx % threads_per_col, None), None]  # ((8, 1), 2, K)

    def prefetch_from_gmem_fn(src_idx, pred: bool = False) -> Tuple[cute.Tensor, cute.Tensor]:
        # Prefetch mAIdx early, even before smem is free
        tApA_k = None
        if const_expr(pred):
            tApA_k = cute.make_fragment(cols_per_thread, Boolean)
            limit_k_cur = limit_k - src_idx * tile_shape_mk[1]
            for k in cutlass.range(cols_per_thread, unroll_full=True):
                tApA_k[k] = t0AcA[0, 0, k][1] < limit_k_cur
        gAIdx_cur = gAIdx[None, src_idx]
        k_idx = cute.make_fragment(cols_per_thread, Int32)
        for k in cutlass.range(cols_per_thread):
            col_idx = tAcA[0, 0, k][1]
            if const_expr(not pred):
                k_idx[k] = gAIdx_cur[col_idx]
            else:
                if tApA_k[k]:
                    k_idx[k] = gAIdx_cur[col_idx]
                else:
                    k_idx[k] = -1
        return k_idx, tApA_k

    def prefetch_from_smem_fn(
        a_prefetch_pipeline,
        src_idx,
        dst_idx,
        a_prefetch_consumer_state,
        pred: bool = False,
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        tApA_k = None
        if const_expr(pred):
            tApA_k = cute.make_fragment(cols_per_thread, Boolean)
            limit_k_cur = limit_k - src_idx * tile_shape_mk[1]
            for k in cutlass.range(cols_per_thread, unroll_full=True):
                tApA_k[k] = t0AcA[0, 0, k][1] < limit_k_cur
        a_prefetch_pipeline.consumer_wait(a_prefetch_consumer_state)
        sAIdx_cur = sAIdx[None, dst_idx]
        k_idx = cute.make_fragment(cols_per_thread, Int32)
        for k in cutlass.range(cols_per_thread):
            col_idx = tAcA[0, 0, k][1]
            k_idx[k] = sAIdx_cur[col_idx]
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            a_prefetch_pipeline.consumer_release(a_prefetch_consumer_state)
        return k_idx, tApA_k

    def copy_fn(
        src_idx,
        dst_idx,
        k_idx_tApA_k: Tuple[cute.Tensor, cute.Tensor],
        pred: bool = False,
    ):
        k_idx, tApA_k = k_idx_tApA_k
        tApA_k_pred = None
        if const_expr(pred):
            tApA_k_pred = cute.prepend_ones(tApA_k, up_to_rank=2)  # (1, cols_per_thread)
        for k in cutlass.range_constexpr(tAcA.shape[2]):
            # copy_A(tAmA[None, None, k_idx[k]], tAsA[(None, None, k), smem_idx],
            #   pred=cute.prepend_ones(tApA_m, up_to_rank=2))
            for m in cutlass.range_constexpr(tAcA.shape[1]):
                if tApA_m[m]:
                    cute.copy(
                        thr_copy_A,
                        tAmA[None, m, k_idx[k]],
                        tAsA[(None, m, k), dst_idx],
                        pred=(None if const_expr(tApA_k_pred is None) else tApA_k_pred[None, k]),
                    )

    return copy_fn, (
        prefetch_from_gmem_fn if const_expr(gAIdx is not None) else prefetch_from_smem_fn
    )
