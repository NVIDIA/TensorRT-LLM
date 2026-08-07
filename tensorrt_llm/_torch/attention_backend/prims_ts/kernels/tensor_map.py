# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Ragged tensor-map helpers for FlashInfer Attention-TS kernels.

The helpers do not interact with the CUDA driver or runtime directly. They
orchestrate DSL/IR-level work and delegate to the tensor-map primitives in
``cutlass.experimental.cuda`` (such as :func:`create_tensor_map_tiled`).
"""

from typing import Sequence, Tuple, Type

import cutlass.cute as cute
from cutlass.cute import depth, leading_dim
from cutlass.cutlass_dsl import (
    Int8,
    Int32,
    Int64,
    Numeric,
    dsl_user_op,
    is_dynamic_expression,
)

from cutlass.experimental.cuda.tensor_map import (
    TensorMap,
    TensorMapDataFormat,
    TensorMapDataType,
    TensorMapFloatOOBFill,
    TensorMapInterleave,
    TensorMapL2Promotion,
    TensorMapSwizzle,
    create_tensor_map_tiled,
    create_tensor_map_tiled_from_view,  # noqa: F401 - public re-export
    get_dsl_type_to_tensormap_type,
    get_tensormap_type_to_dsl_type,
)


# Module-private constants used by the ragged-TMA helpers below. The
# descriptor splits the ragged axis into three TMA dimensions of size
# (box, TmaDimMax, TmaDimMax) whose contributions to the global address
# sum to a multiple of 2^64 (so they wrap to zero) for any element type
# whose width is at least 4 bits:
#   LargeN * XLargeN * elem_bytes = 2^30 * 2^35 * elem_bytes
#                                 = 2^65 * elem_bytes
#                                 ≡ 0 (mod 2^64)  for elem_bytes ≥ 1/2.
# Match the large-dimension sentinels used by the reference ragged tensor-map
# implementation. Kept private — callers never need to reference these
# directly; they are an implementation detail of
# the splice the helpers build.
_RAGGED_LARGE_N = 1 << 30
_RAGGED_XLARGE_N = 1 << 35
_TMA_DIM_MAX = 1 << 31

_LEGACY_TMA_TYPE_TO_FORMAT: dict[TensorMapDataType, TensorMapDataFormat] = {
    TensorMapDataType.uint8: TensorMapDataFormat.BYTE,
    TensorMapDataType.uint16: TensorMapDataFormat.DEFAULT,
    TensorMapDataType.uint32: TensorMapDataFormat.DEFAULT,
    TensorMapDataType.int32: TensorMapDataFormat.DEFAULT,
    TensorMapDataType.uint64: TensorMapDataFormat.DEFAULT,
    TensorMapDataType.int64: TensorMapDataFormat.DEFAULT,
    TensorMapDataType.float16: TensorMapDataFormat.DEFAULT,
    TensorMapDataType.float32: TensorMapDataFormat.DEFAULT,
    TensorMapDataType.float64: TensorMapDataFormat.DEFAULT,
    TensorMapDataType.bfloat16: TensorMapDataFormat.DEFAULT,
    TensorMapDataType.float32_ftz: TensorMapDataFormat.F32_FTZ,
    TensorMapDataType.tfloat32: TensorMapDataFormat.DEFAULT,
    TensorMapDataType.tfloat32_ftz: TensorMapDataFormat.TF32_FTZ,
    TensorMapDataType.f416u4_align8b: TensorMapDataFormat.B4X16,
    TensorMapDataType.f416u4_align16b: TensorMapDataFormat.B4X16_P64,
    TensorMapDataType.f416u6_align16b: TensorMapDataFormat.B6X16_P32,
}


@dsl_user_op
def create_tensor_map_ragged(
    global_address: Int64 | int,
    tma_format: TensorMapDataType | Type[Numeric],
    global_dims: Sequence[Int32 | int],
    global_strides: Sequence[Int64 | int],
    box_dims: Sequence[Int8 | int],
    *,
    ragged_dim_idx: int,
    interleave: TensorMapInterleave | None = None,
    swizzle: TensorMapSwizzle | None = None,
    l2_promotion: TensorMapL2Promotion | None = None,
    oob_fill: TensorMapFloatOOBFill | None = None,
    loc=None,
    ip=None,
) -> TensorMap:
    """Low-level: build a ragged TMA descriptor from explicit parameters.

    Same role for ragged TMA that
    :func:`cutlass.experimental.cuda.create_tensor_map_tiled` plays
    for dense TMA: takes raw column-major-ordered dims / strides / box,
    inserts the synthetic out-of-bounds dimension splice
    around ``ragged_dim_idx``, and delegates to ``create_tensor_map_tiled``.
    Prefer :func:`create_tensor_map_ragged_from_tensor` when a
    ``cute.Tensor`` is available.

    The splice replaces the ragged axis at TMA position
    ``ragged_dim_idx`` with three TMA dimensions of size
    ``(box_dims[ragged_dim_idx], TmaDimMax, TmaDimMax)``.  The
    synthetic strides ``(S, XLargeN - S, S)`` (element units) are
    chosen so that ``LargeN * XLargeN * elem_bytes ≡ 0 (mod 2^64)``
    for any element type whose width is at least 4 bits.  At kernel
    time, the matching :func:`transform_ragged_coords` helper folds a
    runtime ``ragged_extent`` into the coords so out-of-range elements
    fall past the TMA box boundary and are filled per ``oob_fill``.

    :param global_address: Device pointer to the first element of the
        global tensor.
    :type global_address: Int64 or int
    :param tma_format: Element data type (e.g.
        ``TensorMapDataType.float16`` or ``cutlass.Float16``).
    :type tma_format: TensorMapDataType or CUTLASS dtype
    :param global_dims: Shape of the **unspliced** global tensor in
        TMA (column-major) order, length ``R ∈ {2, 3}``.
    :type global_dims: Sequence[Int32 or int]
    :param global_strides: Inter-dimension strides of the unspliced
        tensor in **16-byte units**, length ``R - 1`` (same convention
        as ``create_tensor_map_tiled``: innermost stride is implicit).
    :type global_strides: Sequence[Int64 or int]
    :param box_dims: Tile (box) dimensions in column-major order,
        length ``R``.  ``box_dims[ragged_dim_idx]`` doubles as the
        wraparound period of the synthetic splice.
    :type box_dims: Sequence[Int8 or int]
    :param ragged_dim_idx: TMA-order index of the ragged axis.  Must
        satisfy ``1 ≤ ragged_dim_idx ≤ R - 1`` — the innermost
        (stride-1) axis is rejected because the wraparound stride
        would truncate to 0 under the 16-byte-unit convention for
        sub-128-bit element types.
    :type ragged_dim_idx: int
    :param interleave: Interleave mode, defaults to ``none``.
    :param swizzle: Swizzle mode, defaults to ``none``.
    :param l2_promotion: L2 promotion hint, defaults to ``none``.
    :param oob_fill: OOB fill mode.  Default is format-aware:
        floating TMA formats (``float16``/``float32``/``float64``/
        ``bfloat16``/``float32_ftz``/``tfloat32``/``tfloat32_ftz``)
        get ``nan_request_zero_fma``; everything else (integer types
        and packed sub-byte float types) gets ``none``, because
        ``nan_request_zero_fma`` is only legal for full-precision
        IEEE float TMA formats and the hardware would otherwise fault.
    :raises ValueError: For rank not in ``{2, 3}``, ``ragged_dim_idx``
        out of range or pointing at the innermost axis, or
        ``box_dims`` / ``global_strides`` of wrong length.
    :return: A :class:`TensorMap` of TMA rank ``R + 2``.
    :rtype: TensorMap
    """
    rank = len(global_dims)
    if not 2 <= rank <= 3:
        raise ValueError(
            f"create_tensor_map_ragged supports rank 2 or 3 "
            f"(resulting TMA rank ≤ 5); got input rank {rank}.  Rank 1 "
            f"is rejected because the unit element-stride of the ragged "
            f"axis truncates to 0 in the 16-byte-unit stride convention "
            f"for sub-128-bit element types."
        )
    if len(box_dims) != rank:
        raise ValueError(
            f"box_dims length {len(box_dims)} does not match global_dims rank {rank}"
        )
    if len(global_strides) != rank - 1:
        raise ValueError(
            f"global_strides length {len(global_strides)} != rank-1 = {rank - 1}"
        )
    if not 1 <= ragged_dim_idx <= rank - 1:
        raise ValueError(
            "ragged_dim_idx must satisfy 1 ≤ idx ≤ rank-1 "
            f"(innermost axis at TMA position 0 is not allowed); "
            f"got ragged_dim_idx={ragged_dim_idx} for rank-{rank} input"
        )

    # Coerce CUTLASS dtypes (e.g. ``cutlass.Float16``) to their TensorMapDataType
    # representative so the format-aware OOB-fill default below works
    # uniformly with whichever spelling the caller passed.
    fmt_resolved = (
        tma_format
        if isinstance(tma_format, TensorMapDataType)
        else get_dsl_type_to_tensormap_type(tma_format)
    )
    elem_bits = fmt_resolved.bit_width

    # Splice global_dims: replace the ragged entry with (box, MaxDim, MaxDim).
    box_at_ragged = box_dims[ragged_dim_idx]
    spliced_global_dims = (
        list(global_dims[:ragged_dim_idx])
        + [box_at_ragged, _TMA_DIM_MAX, _TMA_DIM_MAX]
        + list(global_dims[ragged_dim_idx + 1 :])
    )

    # Splice box_dims: replace the ragged entry with (box, 1, 1).
    spliced_box_dims = (
        list(box_dims[:ragged_dim_idx])
        + [box_at_ragged, 1, 1]
        + list(box_dims[ragged_dim_idx + 1 :])
    )

    # Splice strides (16-byte units).  `global_strides[i]` is associated
    # with `global_dims[i+1]` (the innermost stride is implicit), so the
    # ragged-axis stride lives at `global_strides[ragged_dim_idx - 1]`.
    # The synthetic wraparound stride ``XLargeN`` is converted from
    # element units to 16-byte units here so the formula below stays
    # uniform with the pass-through entries.
    s_ragged_16b = global_strides[ragged_dim_idx - 1]
    xlarge_n_16b = _RAGGED_XLARGE_N * elem_bits // 128
    spliced_global_strides = (
        list(global_strides[: ragged_dim_idx - 1])
        + [s_ragged_16b, xlarge_n_16b - s_ragged_16b, s_ragged_16b]
        + list(global_strides[ragged_dim_idx:])
    )

    if oob_fill is None:
        # nan_request_zero_fma is only legal for full-precision IEEE
        # float TMA formats; integer types and packed sub-byte float
        # formats (uint8 for FP8, f416u4, ...) must use `none` or the
        # descriptor faults at issue.
        _FLOAT_TMA_FORMATS = (
            TensorMapDataType.float16,
            TensorMapDataType.float32,
            TensorMapDataType.float64,
            TensorMapDataType.bfloat16,
            TensorMapDataType.float32_ftz,
            TensorMapDataType.tfloat32,
            TensorMapDataType.tfloat32_ftz,
        )
        oob_fill = (
            TensorMapFloatOOBFill.nan_request_zero_fma
            if fmt_resolved in _FLOAT_TMA_FORMATS
            else TensorMapFloatOOBFill.none
        )

    descriptor_dtype = (
        tma_format
        if isinstance(tma_format, type)
        else get_tensormap_type_to_dsl_type(fmt_resolved)
    )
    descriptor_format = (
        None
        if isinstance(tma_format, type)
        else _LEGACY_TMA_TYPE_TO_FORMAT[fmt_resolved]
    )

    return create_tensor_map_tiled(
        global_address=global_address,
        dtype=descriptor_dtype,
        tma_format=descriptor_format,
        global_dims=spliced_global_dims,
        global_strides=spliced_global_strides,
        box_dims=spliced_box_dims,
        interleave=interleave,
        swizzle=swizzle,
        l2_promotion=l2_promotion,
        oob_fill=oob_fill,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def create_tensor_map_ragged_from_tensor(
    tensor: cute.Tensor,
    box_dims: Tuple[Int8 | int, ...],
    *,
    ragged_dim: int,
    stride_order: Tuple[int, ...] | None = None,
    interleave: TensorMapInterleave | None = None,
    swizzle: TensorMapSwizzle | None = None,
    l2_promotion: TensorMapL2Promotion | None = None,
    oob_fill: TensorMapFloatOOBFill | None = None,
    tma_format: TensorMapDataType | None = None,
    loc=None,
    ip=None,
) -> TensorMap:
    """Build a ragged TMA descriptor from a :class:`cute.Tensor`.

    Thin wrapper around :func:`create_tensor_map_ragged`: derives
    ``global_address`` / ``global_dims`` / ``global_strides`` /
    ``tma_format`` from ``tensor`` (auto-inferring stride order from
    the tensor's strides unless ``stride_order`` is given), then
    delegates the splice + descriptor encoding.  Pairs with
    :func:`transform_ragged_coords` on the kernel side.  This is the
    synthetic out-of-bounds pattern exposed as a convenience helper.

    :param tensor: Input ``cute.Tensor`` of rank 2 or 3 with a
        flattened layout and a stride-1 leading dimension.
    :type tensor: cute.Tensor
    :param box_dims: Tile dimensions in **tensor mode order** (same
        convention as
        ``create_tensor_map_tiled_from_view``).
        ``box_dims[ragged_dim]`` doubles as the wraparound period of
        the synthetic splice.
    :type box_dims: tuple[Int8 or int, ...]
    :param ragged_dim: Tensor mode index of the ragged axis (the axis
        whose runtime length varies per CTA).  Must not be the
        innermost (stride-1) axis when rank > 1.
    :type ragged_dim: int
    :param stride_order: Explicit dimension order from innermost to
        outermost (same semantics as
        ``create_tensor_map_tiled_from_view``).
    :type stride_order: tuple[int, ...], optional
    :param interleave: Interleave mode, defaults to ``none``.
    :param swizzle: Swizzle mode, defaults to ``none``.
    :param l2_promotion: L2 promotion hint, defaults to ``none``.
    :param oob_fill: OOB fill mode; see
        :func:`create_tensor_map_ragged` for the format-aware default.
    :param tma_format: Override the element data type, defaults to
        auto-detect from ``tensor.element_type``.
    :raises ValueError: For unsupported rank (``R ∉ {2, 3}``),
        invalid ``ragged_dim``, ragged axis being the innermost, or
        ambiguous stride ordering.
    :return: A :class:`TensorMap` of TMA rank ``R + 2``.
    :rtype: TensorMap

    Example — fp16 ``[outer_padded, inner]`` row-major, ragged on
    the outer axis::

        desc = create_tensor_map_ragged_from_tensor(
            t,
            box_dims=(tile_outer, inner),  # tensor mode order
            ragged_dim=0,                  # outer (ragged) axis
            swizzle=TensorMapSwizzle.s128b,
        )
    """
    if depth(tensor) > 1:
        raise ValueError(
            f"Expected tensor to have flattened layout, got {tensor.layout}"
        )

    rank = len(tensor.shape)
    if not 0 <= ragged_dim < rank:
        raise ValueError(f"ragged_dim={ragged_dim} out of range for rank-{rank} tensor")

    leading_mode = leading_dim(tensor.shape, tensor.stride)
    if leading_mode is None or not isinstance(leading_mode, int):
        raise ValueError(
            "Expected tensor to have a leading (stride-1) dimension, but got "
            f"tensor layout {tensor.layout}"
        )

    tensor_shapes = list(tensor.shape)
    tensor_strides = list(tensor.stride)
    box_dims_list = list(box_dims)
    if len(box_dims_list) != rank:
        raise ValueError(
            f"box_dims rank {len(box_dims_list)} does not match tensor rank {rank}"
        )

    if stride_order is not None:
        order = list(stride_order)
    else:
        if any(is_dynamic_expression(s) for s in tensor_strides):
            raise ValueError(
                f"Cannot infer a unique stride order from tensor strides "
                f"{tensor_strides} due to dynamic strides. Please provide "
                "`stride_order` explicitly."
            )
        if len(set(tensor_strides)) < rank:
            raise ValueError(
                f"Cannot infer a unique stride order from tensor strides "
                f"{tensor_strides} due to duplicate strides. Please provide "
                "`stride_order` explicitly."
            )
        order = sorted(range(rank), key=lambda i: tensor_strides[i])

    try:
        ragged_pos = order.index(ragged_dim)
    except ValueError as e:
        raise ValueError(
            f"ragged_dim={ragged_dim} not present in stride_order {order}"
        ) from e

    # Reorder shapes / strides / box into TMA column-major order.
    tma_dims = [tensor_shapes[order[j]] for j in range(rank)]
    tma_strides_orig = [tensor_strides[order[j]] for j in range(rank)]
    tma_box = [box_dims_list[order[j]] for j in range(rank)]

    # Convert to the 16-byte-unit stride convention the low-level helper
    # (and create_tensor_map_tiled) expects: length rank - 1, drops the
    # innermost stride.
    elem_bits = tensor.element_type.width
    global_strides = [
        tma_strides_orig[i + 1] * elem_bits // 128 for i in range(rank - 1)
    ]

    fmt = (
        tma_format
        if tma_format is not None
        else get_dsl_type_to_tensormap_type(tensor.element_type)
    )

    return create_tensor_map_ragged(
        global_address=tensor.iterator.toint(),
        tma_format=fmt,
        global_dims=tma_dims,
        global_strides=global_strides,
        box_dims=tma_box,
        ragged_dim_idx=ragged_pos,
        interleave=interleave,
        swizzle=swizzle,
        l2_promotion=l2_promotion,
        oob_fill=oob_fill,
        loc=loc,
        ip=ip,
    )


def transform_ragged_coords(
    coords: Sequence[Int32 | int],
    *,
    ragged_dim_idx: int,
    ragged_box_size: Int32 | int,
    ragged_extent: Int32 | int,
) -> Tuple[Int32, ...]:
    """Expand a logical coordinate tuple into the rank-``R + 2`` form
    expected by a descriptor built with
    :func:`create_tensor_map_ragged_from_tensor`.

    Call from inside ``@cute.kernel``.  Given the kernel's original
    coordinate tuple (rank ``R = len(coords)`` in TMA order, the same
    rank you would pass to ``cp_async_bulk_tensor_*`` for the dense
    path) plus a runtime ``ragged_extent`` (number of valid elements
    along the ragged axis), this inserts the two synthetic LargeN
    coordinates and rewrites the ragged-axis coordinate so
    out-of-range elements fall past the TMA box boundary.  The math
    follows the reference coordinate transformation, with an explicit
    ``ragged_extent == 0`` carve-out so a
    fully-empty tile is treated as fully OOB rather than fully
    in-bounds)::

        ext = clamp(ragged_extent, 0, ragged_box_size)
        ext_mod = ext % ragged_box_size
        dist    = ragged_box_size - ext_mod
        d_mod   = dist % ragged_box_size       # (ext == box ⇒ 0)
        if ext == 0:
            d_mod = ragged_box_size            # all OOB carve-out
        bal     = -d_mod
        out[ragged_dim_idx + 0] = d_mod
        out[ragged_dim_idx + 1] = LargeN
        out[ragged_dim_idx + 2] = coords[ragged_dim_idx] + LargeN + bal

    All other axes pass through unchanged.

    Behavior across the ``ragged_extent`` value range, with
    ``box = ragged_box_size``:

    +-------------------+------------------------------------------+-----------+
    | ``ragged_extent`` | code path                                | outcome   |
    +===================+==========================================+===========+
    | ``0``             | empty-tile carve-out → ``d_mod = box``   | all OOB   |
    +-------------------+------------------------------------------+-----------+
    | ``0 < ext < box`` | normal formula → ``d_mod = box - ext``   | partial   |
    +-------------------+------------------------------------------+-----------+
    | ``ext == box``    | normal formula → ``d_mod = 0``           | all valid |
    +-------------------+------------------------------------------+-----------+
    | ``ext > box``     | upper-clamp → ``ext = box`` → ``d_mod=0``| all valid |
    +-------------------+------------------------------------------+-----------+
    | ``ext < 0``       | lower-clamp → ``ext = 0`` → carve-out    | all OOB   |
    +-------------------+------------------------------------------+-----------+

    :param coords: Original coordinates in TMA order.  Length is the
        original tensor rank ``R``; must be 2 or 3 (matching the
        descriptor helper's rank constraint).
    :type coords: Sequence[Int32 or int]
    :param ragged_dim_idx: TMA-order index of the ragged axis (i.e.,
        ``stride_order.index(ragged_dim)`` for the descriptor that
        this coord tuple drives).  Must be ≥ 1.
    :type ragged_dim_idx: int
    :param ragged_box_size: Box size along the ragged axis (same
        as ``box_dims[ragged_dim]`` passed to the descriptor helper).
        Used as the wraparound modulus.
    :type ragged_box_size: Int32 or int
    :param ragged_extent: Runtime number of valid elements along the
        ragged axis starting at this TMA coordinate's ragged-axis origin.
        This is a tile-local remaining length, typically computed as
        ``ragged_limit - coords[ragged_dim_idx]``.  It is not an absolute
        limit coordinate or the total logical length of the ragged axis.
        Any value is accepted; the helper clamps to ``[0, ragged_box_size]``
        internally so callers can pass a raw per-TMA-tile difference
        (e.g., ``mn_limit - tile_origin``) without preconditioning.
        ``ragged_extent == 0`` produces all-OOB coordinates (TMA
        load fills the whole tile per ``oob_fill``; TMA store is a
        full no-op).  ``ragged_extent >= ragged_box_size`` produces
        all-in-bounds coordinates.
    :type ragged_extent: Int32 or int
    :return: Expanded coordinate tuple of length ``len(coords) + 2``.
    :rtype: tuple[Int32, ...]
    """
    rank = len(coords)
    if not 2 <= rank <= 3:
        raise ValueError(f"transform_ragged_coords supports rank 2 or 3; got {rank}")
    if not 0 <= ragged_dim_idx < rank:
        raise ValueError(
            f"ragged_dim_idx={ragged_dim_idx} out of range for rank-{rank}"
        )
    if ragged_dim_idx == 0:
        raise ValueError("ragged axis cannot be the innermost (stride-1) TMA dimension")

    box = Int32(ragged_box_size)
    ext = Int32(ragged_extent)
    # Clamp ext to [0, box].  The OOB-trick formula is only
    # well-defined in that range — `ext > box` would treat the
    # excess like a partial tile and falsely mark some rows OOB; a
    # negative `ext` (e.g. when the caller passes
    # `mn_limit - tile_origin` for a tile fully past the limit)
    # likewise produces nonsense.  Doing the clamp here lets every
    # call site pass a raw per-CTA difference and stay readable.
    is_neg = Int32(ext < Int32(0))
    ext = ext - ext * is_neg  # ext < 0  ⇒  0
    is_over = Int32(ext > box)
    ext = ext + (box - ext) * is_over  # ext > box ⇒  box

    ext_mod = ext % box
    dist = box - ext_mod
    d_mod = dist % box
    # When `ragged_extent == 0`, the formula above yields d_mod = 0
    # (same as a fully-utilized tile).  Override so a fully-empty
    # tile gets d_mod = ragged_box_size, pushing the entire box past
    # the TMA bound so all lanes are OOB-handled.
    is_empty = Int32(ext == Int32(0))
    d_mod = d_mod + box * is_empty
    bal = Int32(0) - d_mod
    large_n = Int32(_RAGGED_LARGE_N)

    orig = Int32(coords[ragged_dim_idx])
    expanded = (
        tuple(Int32(c) for c in coords[:ragged_dim_idx])
        + (d_mod, large_n, orig + large_n + bal)
        + tuple(Int32(c) for c in coords[ragged_dim_idx + 1 :])
    )
    return expanded


__all__ = [
    "create_tensor_map_ragged",
    "create_tensor_map_ragged_from_tensor",
    "transform_ragged_coords",
]
