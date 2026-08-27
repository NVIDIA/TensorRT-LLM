"""CuTe copy helpers used by the Hopper mainloop."""

from typing import Callable

import cutlass
import cutlass.cute as cute
from cutlass import const_expr
from cutlass import pipeline
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import dsl_user_op


_RAGGED_BASE = 2**30
_RAGGED_LIMIT = 2**31 - 1
_RAGGED_WRAP_STRIDE = 2**64 // _RAGGED_BASE


@dsl_user_op
def create_ragged_tensor_for_tma(
    tensor: cute.Tensor,
    ragged_dim: int = 0,
    ptr_shift: bool = False,
    *,
    loc=None,
    ip=None,
) -> cute.Tensor:
    rank = cute.rank(tensor)
    if ragged_dim < 0:
        ragged_dim += rank
    if ptr_shift:
        shape = (
            tensor.shape[:ragged_dim]
            + (_RAGGED_BASE,)
            + tensor.shape[ragged_dim + 1 :]
            + (_RAGGED_LIMIT,)
        )
        stride = tensor.stride + (tensor.stride[ragged_dim],)
        offset = (
            (None,) * ragged_dim
            + (-_RAGGED_BASE,)
            + (None,) * (rank - ragged_dim - 1)
        )
        pointer = cute.domain_offset(offset, tensor).iterator
        return cute.make_tensor(
            pointer,
            cute.make_layout(shape, stride=stride),
        )

    ragged_stride = tensor.stride[ragged_dim]
    shape = (
        tensor.shape[:ragged_dim]
        + (_RAGGED_BASE,)
        + tensor.shape[ragged_dim + 1 :]
        + (_RAGGED_LIMIT, _RAGGED_LIMIT)
    )
    stride = (
        tensor.stride[:ragged_dim]
        + (ragged_stride,)
        + tensor.stride[ragged_dim + 1 :]
        + (_RAGGED_WRAP_STRIDE - ragged_stride, ragged_stride)
    )
    return cute.make_tensor(
        tensor.iterator,
        cute.make_layout(shape, stride=stride),
    )


def tma_get_copy_fn(
    atom: cute.CopyAtom,
    cta_coord: cute.Coord,
    cta_layout: cute.Layout,
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    filter_zeros: bool = False,
    single_stage: bool = False,
    *,
    loc=None,
    ip=None,
    **kwargs,
) -> Callable:
    source_is_smem = const_expr(
        isinstance(src_tensor.iterator, cute.Pointer)
        and src_tensor.memspace == cute.AddressSpace.smem
    )
    smem, gmem = (
        (src_tensor, dst_tensor)
        if source_is_smem
        else (dst_tensor, src_tensor)
    )
    smem_rank = const_expr(cute.rank(smem) - (0 if single_stage else 1))
    gmem_rank = const_expr(cute.rank(gmem) - (0 if single_stage else 1))
    smem, gmem = cpasync.tma_partition(
        atom,
        cta_coord,
        cta_layout,
        cute.group_modes(smem, 0, smem_rank),
        cute.group_modes(gmem, 0, gmem_rank),
        loc=loc,
        ip=ip,
    )
    if const_expr(filter_zeros):
        smem = cute.filter_zeros(smem)
        gmem = cute.filter_zeros(gmem)
    source, destination = (
        (smem, gmem) if source_is_smem else (gmem, smem)
    )

    @dsl_user_op
    def copy_tma(
        src_idx,
        dst_idx,
        *,
        loc=None,
        ip=None,
        **call_kwargs,
    ):
        cute.copy(
            atom,
            source[None, src_idx],
            destination[None, dst_idx],
            **call_kwargs,
            **kwargs,
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    def copy_single_stage(*, loc=None, ip=None, **call_kwargs):
        cute.copy(
            atom,
            source,
            destination,
            **call_kwargs,
            **kwargs,
            loc=loc,
            ip=ip,
        )

    return (
        copy_tma if const_expr(not single_stage) else copy_single_stage,
        smem,
        gmem,
    )


def tma_producer_copy_fn(
    copy: Callable,
    copy_pipeline: pipeline.PipelineAsync,
):
    def copy_fn(
        src_idx,
        producer_state: pipeline.PipelineState,
        **kwargs,
    ):
        copy(
            src_idx=src_idx,
            dst_idx=producer_state.index,
            tma_bar_ptr=copy_pipeline.producer_get_barrier(producer_state),
            **kwargs,
        )

    return copy_fn


__all__ = [
    "create_ragged_tensor_for_tma",
    "tma_get_copy_fn",
    "tma_producer_copy_fn",
]
