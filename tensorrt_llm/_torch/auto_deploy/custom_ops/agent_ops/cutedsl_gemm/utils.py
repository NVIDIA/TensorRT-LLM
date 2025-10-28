import math
from functools import partial
from typing import Optional, Tuple, Type, Union

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass._mlir.dialects import llvm, nvvm, vector
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import T, dsl_user_op

# cute.arch.{fma,mul,add}_packed_f32x2 uses RZ rounding mode by default
fma_packed_f32x2 = partial(cute.arch.fma_packed_f32x2, rnd=nvvm.RoundingModeKind.RN)
mul_packed_f32x2 = partial(cute.arch.mul_packed_f32x2, rnd=nvvm.RoundingModeKind.RN)
add_packed_f32x2 = partial(cute.arch.add_packed_f32x2, rnd=nvvm.RoundingModeKind.RN)
sub_packed_f32x2 = partial(
    cute.arch.calc_packed_f32x2_op,
    src_c=None,
    calc_func=nvvm.sub_packed_f32x2,
    rnd=nvvm.RoundingModeKind.RN,
)


def convert_from_dlpack(x, leading_dim, alignment=16, divisibility=1) -> cute.Tensor:
    return (
        from_dlpack(x, assumed_align=alignment)
        .mark_layout_dynamic(leading_dim=leading_dim)
        .mark_compact_shape_dynamic(
            mode=leading_dim, stride_order=x.dim_order(), divisibility=divisibility
        )
    )


def transpose_view(a: cute.Tensor) -> cute.Tensor:
    """Transpose the first two dimensions of a tensor on smem."""
    shape = (a.shape[1], a.shape[0], *a.shape[2:])
    order = (1, 0, *range(2, cute.rank(a)))
    return cute.composition(a, cute.make_ordered_layout(shape, order=order))


def select(a: cute.Tensor, mode: list[int]) -> cute.Tensor:
    return cute.make_tensor(a.iterator, cute.select(a.layout, mode))


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord: cute.Coord, *, loc=None, ip=None) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


@cute.jit
def load_scalar_or_pointer(x: Float32 | cute.Pointer) -> Float32:
    if const_expr(isinstance(x, cute.Pointer)):
        return Float32(cute.make_tensor(x, cute.make_layout(1))[0])
    else:
        assert isinstance(x, Float32)
        return x


@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: cute.Int32, *, loc=None, ip=None
) -> cutlass.Int32:
    """Map the given smem pointer to the address at another CTA rank in the cluster."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def store_shared_remote(
    val: float | Float32 | Int32 | cutlass.Int64,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: cute.typing.Int,
    *,
    loc=None,
    ip=None,
) -> None:
    remote_smem_ptr_i32 = set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    if const_expr(isinstance(val, float)):
        val = Float32(val)
    assert isinstance(val, (Float32, Int32, cutlass.Int64)), "val must be Float32, Int32, or Int64"
    suffix = {Float32: "f32", Int32: "s32", cutlass.Int64: "s64"}[type(val)]
    constraint = {Float32: "f", Int32: "r", cutlass.Int64: "l"}[type(val)]
    llvm.inline_asm(
        None,
        [remote_smem_ptr_i32, val.ir_value(loc=loc, ip=ip), remote_mbar_ptr_i32],
        f"st.async.shared::cluster.mbarrier::complete_tx::bytes.{suffix} [$0], $1, [$2];",
        f"r,{constraint},r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def fmin(a: Union[float, Float32], b: Union[float, Float32], *, loc=None, ip=None) -> Float32:
    return Float32(
        nvvm.fmin(
            T.f32(),
            Float32(a).ir_value(loc=loc, ip=ip),
            Float32(b).ir_value(loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def sqrt(a: float | Float32, *, loc=None, ip=None) -> Float32:
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "sqrt.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def ceil(a: float | Float32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "cvt.rpi.ftz.s32.f32 $0, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def prmt(a: int | Int32, b: int | Int32, c: int | Int32, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int32(a).ir_value(loc=loc, ip=ip),
                Int32(b).ir_value(loc=loc, ip=ip),
                Int32(c).ir_value(loc=loc, ip=ip),
            ],
            "prmt.b32 $0, $1, $2, $3;",
            "=r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@cute.jit
def permute_gated_Cregs_b16(t: cute.Tensor) -> None:
    assert t.element_type.width == 16
    assert cute.size(t.shape) % 4 == 0, "Tensor size must be a multiple of 4 for b16 permutation"
    t_u32 = cute.recast_tensor(t, Int32)

    quad_idx = cute.arch.lane_idx() % 4
    lane_03 = quad_idx == 0 or quad_idx == 3
    selector_upper = Int32(0x5410) if lane_03 else Int32(0x1054)
    selector_lower = Int32(0x7632) if lane_03 else Int32(0x3276)
    # upper_map = [0, 3, 1, 2]
    # lower_map = [1, 2, 0, 3]
    # upper_idx = upper_map[quad_idx]
    # indexing isn't supported so we have to do arithmetic
    upper_idx = quad_idx // 2 if quad_idx % 2 == 0 else 3 - quad_idx // 2
    lower_idx = upper_idx ^ 1

    # 1 -> 0b11111, 2 -> 0b11110, 4 -> 0b11100, 8 -> 0b11000, 16 -> 0b10000, 32 -> 0b00000
    width = 4
    mask = cute.arch.WARP_SIZE - width
    clamp = cute.arch.WARP_SIZE - 1
    mask_and_clamp = mask << 8 | clamp

    for i in cutlass.range(cute.size(t_u32.shape) // 2, unroll_full=True):
        upper, lower = t_u32[i * 2 + 0], t_u32[i * 2 + 1]
        upper0 = upper if lane_03 else lower
        lower0 = lower if lane_03 else upper
        upper0 = cute.arch.shuffle_sync(upper0, offset=upper_idx, mask_and_clamp=mask_and_clamp)
        lower0 = cute.arch.shuffle_sync(lower0, offset=lower_idx, mask_and_clamp=mask_and_clamp)
        t_u32[i * 2 + 0] = prmt(upper0, lower0, selector_upper)
        t_u32[i * 2 + 1] = prmt(upper0, lower0, selector_lower)


@cute.jit
def predicate_k(tAcA: cute.Tensor, limit: cutlass.Int32) -> cute.Tensor:
    # Only compute predicates for the "k" dimension. For the mn dimension, we will use "if"
    tApA = cute.make_fragment(
        cute.make_layout(
            (
                cute.size(tAcA, mode=[0, 1]),
                cute.size(tAcA, mode=[1]),
                cute.size(tAcA, mode=[2]),
            ),
            stride=(cute.size(tAcA, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tApA.shape[0]):
        for rest_k in cutlass.range_constexpr(tApA.shape[2]):
            tApA[rest_v, 0, rest_k] = cute.elem_less(tAcA[(0, rest_v), 0, rest_k][1], limit)
    return tApA


@cute.jit
def fill_oob(tXsX: cute.Tensor, tXpX: Optional[cute.Tensor], fill_value: cute.Numeric) -> None:
    """Fill out-of-bounds values in shared memory tensor.

    Args:
        tXsX: Shared memory tensor to fill
        tXpX: Predicate tensor indicating valid elements
        fill_value: Value to fill OOB locations with
    """
    tXrX_fill = cute.make_fragment_like(tXsX[(None, 0), None, 0])
    tXrX_fill.fill(fill_value)
    for rest_v in cutlass.range_constexpr(tXsX.shape[0][1]):
        for rest_k in cutlass.range_constexpr(tXsX.shape[2]):
            if const_expr(tXpX is not None):
                if not tXpX[rest_v, 0, rest_k]:
                    cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])
            else:
                cute.autovec_copy(tXrX_fill, tXsX[(None, rest_v), None, rest_k])


@dsl_user_op
def f32x2_to_i64(a: Float32, b: Float32, *, loc=None, ip=None) -> cutlass.Int64:
    vec_f32x2 = vector.from_elements(
        T.vector(2, T.f32()), (a.ir_value(), b.ir_value()), loc=loc, ip=ip
    )
    vec_i64x1 = vector.bitcast(T.vector(1, T.i64()), vec_f32x2)
    res = cutlass.Int64(
        vector.extract(vec_i64x1, dynamic_position=[], static_position=[0], loc=loc, ip=ip)
    )
    return res


@dsl_user_op
def i64_to_f32x2(c: cutlass.Int64, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    vec_i64x1 = vector.from_elements(T.vector(1, T.i64()), (c.ir_value(),), loc=loc, ip=ip)
    vec_f32x2 = vector.bitcast(T.vector(2, T.f32()), vec_i64x1)
    res0 = Float32(
        vector.extract(vec_f32x2, dynamic_position=[], static_position=[0], loc=loc, ip=ip)
    )
    res1 = Float32(
        vector.extract(vec_f32x2, dynamic_position=[], static_position=[1], loc=loc, ip=ip)
    )
    return res0, res1


@dsl_user_op
def domain_offset_i64(coord: cute.Coord, tensor: cute.Tensor, *, loc=None, ip=None) -> cute.Tensor:
    flat_coord_i64 = tuple(cutlass.Int64(c) for c in cute.flatten(coord))
    flat_stride = cute.flatten_to_tuple(tensor.stride)
    assert len(flat_coord_i64) == len(flat_stride), (
        "Coordinate and stride must have the same length"
    )
    offset = sum(c * s for c, s in zip(flat_coord_i64, flat_stride))
    assert isinstance(tensor.iterator, cute.Pointer)
    # HACK: we assume that applying the offset does not change the pointer alignment
    new_ptr = cute.make_ptr(
        tensor.element_type,
        tensor.iterator.toint() + offset * tensor.element_type.width // 8,
        tensor.memspace,
        assumed_align=tensor.iterator.max_alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)


@dsl_user_op
def coord_offset_i64(
    idx: cute.typing.Int, tensor: cute.Tensor, dim: int, *, loc=None, ip=None
) -> cute.Tensor:
    offset = cutlass.Int64(idx) * cute.size(tensor.stride[dim])
    assert isinstance(tensor.iterator, cute.Pointer)
    # HACK: we assume that applying the offset does not change the pointer alignment
    new_ptr = cute.make_ptr(
        tensor.element_type,
        tensor.iterator.toint() + offset * tensor.element_type.width // 8,
        tensor.memspace,
        assumed_align=tensor.iterator.max_alignment,
    )
    return cute.make_tensor(new_ptr, tensor.layout)


@cute.jit
def warp_prefix_sum(val: cutlass.Int32, lane: Optional[cutlass.Int32] = None) -> cutlass.Int32:
    if const_expr(lane is None):
        lane = cute.arch.lane_idx()
    for i in cutlass.range_constexpr(int(math.log2(cute.arch.WARP_SIZE))):
        offset = 1 << i
        # Very important that we set mask_and_clamp to 0
        partial_sum = cute.arch.shuffle_sync_up(val, offset=offset, mask_and_clamp=0)
        if lane >= offset:
            val += partial_sum
    return val


def convert_layout_acc_mn(acc_layout: cute.Layout) -> cute.Layout:
    """
    For Sm80, convert ((2, 2), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, MMA_N), ...).
    For Sm90, convert ((2, 2, V), MMA_M, MMA_N, ...) to ((2, MMA_M), (2, V, MMA_N), ...).
    """
    acc_layout_col_major = cute.make_layout(acc_layout.shape)
    acc_layout_mn = cute.make_layout(
        (
            (acc_layout_col_major.shape[0][1], acc_layout_col_major.shape[1]),  # MMA_M
            (
                acc_layout_col_major.shape[0][0],
                *acc_layout_col_major.shape[0][2:],
                acc_layout_col_major.shape[2],
            ),  # MMA_N
            *acc_layout_col_major.shape[3:],
        ),
        stride=(
            (
                acc_layout_col_major.stride[0][1],
                acc_layout_col_major.stride[1],
            ),  # MMA_M
            (
                acc_layout_col_major.stride[0][0],
                *acc_layout_col_major.stride[0][2:],
                acc_layout_col_major.stride[2],
            ),  # MMA_N
            *acc_layout_col_major.stride[3:],
        ),
    )
    return cute.composition(acc_layout, acc_layout_mn)


def make_acc_tensor_mn_view(acc: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(acc.iterator, convert_layout_acc_mn(acc.layout))


def convert_layout_zero_stride(
    input: cute.Tensor | cute.Layout, ref_layout: cute.Layout
) -> cute.Layout:
    layout = input.layout if const_expr(isinstance(input, cute.Tensor)) else input
    # Group the modes with non-zero stride in the ref_layout together,
    # and the modes with zero stride together
    layout_flat = cute.flatten(layout)
    ref_layout_flat = cute.flatten(ref_layout)
    nonzero_modes = [i for i in range(cute.rank(layout_flat)) if ref_layout_flat[i].stride != 0]
    zero_modes = [i for i in range(cute.rank(layout_flat)) if ref_layout_flat[i].stride == 0]
    # There's an edge case when all modes are zero stride
    new_shape = (
        (tuple(layout_flat[i].shape for i in nonzero_modes) if len(nonzero_modes) > 0 else (1,)),
        tuple(layout_flat[i].shape for i in zero_modes),
    )
    new_stride = (
        (tuple(layout_flat[i].stride for i in nonzero_modes) if len(nonzero_modes) > 0 else (0,)),
        tuple(layout_flat[i].stride for i in zero_modes),
    )
    out_layout = cute.make_layout(new_shape, stride=new_stride)
    if const_expr(isinstance(input, cute.Tensor)):
        return cute.make_tensor(input.iterator, out_layout)
    else:
        return out_layout


@dsl_user_op
def sm90_get_smem_load_op(
    layout_c: cutlass.utils.LayoutEnum,
    elem_ty_c: Type[cutlass.Numeric],
    *,
    loc=None,
    ip=None,
) -> cute.CopyAtom:
    """
    Selects the largest vectorized smem load atom available subject to constraint of gmem layout.

    Parameters:
    -----------
    layout_c : LayoutEnum
        The layout enum of the output tensor D.

    elem_ty_c : Type[Numeric]
        The element type for output tensor D.

    Returns:
    --------
    Either SmemLoadMatrix or SimtSyncCopy, based on the input parameters.
    """

    if not isinstance(elem_ty_c, cutlass.cutlass_dsl.NumericMeta):
        raise TypeError(f"elem_ty_c must be a Numeric, but got {elem_ty_c}")
    is_m_major = layout_c.is_m_major_c()
    if elem_ty_c.width == 16:
        return cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(is_m_major, 4), elem_ty_c, loc=loc, ip=ip
        )
    else:
        return cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), elem_ty_c, loc=loc, ip=ip)


@dsl_user_op
def atomic_add_i32(a: int | Int32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    return nvvm.atomicrmw(
        res=T.i32(),
        op=nvvm.AtomicOpKind.ADD,
        ptr=gmem_ptr.llvm_ptr,
        a=Int32(a).ir_value(),
    )


@dsl_user_op
def atomic_inc_i32(a: int | Int32, gmem_ptr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    return nvvm.atomicrmw(
        res=T.i32(),
        op=nvvm.AtomicOpKind.INC,
        ptr=gmem_ptr.llvm_ptr,
        a=Int32(a).ir_value(),
    )
