from typing import Type, Union

import cutlass.cute as cute
import cutlass.utils.hopper_helpers as sm90_utils_og
from cutlass import Int32, const_expr
from cutlass.cute.nvgpu import warpgroup
from cutlass.cutlass_dsl import Numeric, dsl_user_op
from cutlass.utils import LayoutEnum


@dsl_user_op
def make_smem_layout_epi(
    epi_dtype: Type[Numeric],
    epi_layout: LayoutEnum,
    epi_tile: cute.Tile,
    epi_stage: int,
    *,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    epilog_shape = cute.product_each(cute.shape(epi_tile, loc=loc, ip=ip), loc=loc, ip=ip)
    epi_major_mode_size = epilog_shape[1] if epi_layout.is_n_major_c() else epilog_shape[0]
    epi_smem_layout_atom = warpgroup.make_smem_layout_atom(
        sm90_utils_og.get_smem_layout_atom(epi_layout, epi_dtype, epi_major_mode_size),
        epi_dtype,
    )
    epi_smem_layout_staged = cute.tile_to_shape(
        epi_smem_layout_atom,
        cute.append(epilog_shape, epi_stage),
        order=(1, 0, 2) if epi_layout.is_m_major_c() else (0, 1, 2),
    )
    return epi_smem_layout_staged


@dsl_user_op
def partition_for_epilogue(
    cT: cute.Tensor,
    epi_tile: cute.Tile,
    tiled_copy: cute.TiledCopy,
    tidx: Int32,
    reference_src: bool,  # do register tensors reference the src or dst layout of the tiled copy
    *,
    loc=None,
    ip=None,
) -> cute.Tensor:
    thr_copy = tiled_copy.get_slice(tidx)
    cT_epi = cute.flat_divide(cT, epi_tile)
    # (CPY, CPY_M, CPY_N, EPI_M, EPI_N)
    if const_expr(reference_src):
        return thr_copy.partition_S(cT_epi, loc=loc, ip=ip)
    else:
        return thr_copy.partition_D(cT_epi, loc=loc, ip=ip)
