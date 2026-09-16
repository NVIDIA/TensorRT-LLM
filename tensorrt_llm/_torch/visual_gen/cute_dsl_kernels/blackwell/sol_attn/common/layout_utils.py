"""Tensor-layout helpers shared by the two CuTe kernels."""

import cutlass.cute as cute
from cutlass import const_expr


def transpose_view(tensor: cute.Tensor) -> cute.Tensor:
    shape = (tensor.shape[1], tensor.shape[0], *tensor.shape[2:])
    order = (1, 0, *range(2, cute.rank(tensor)))
    return cute.composition(
        tensor,
        cute.make_ordered_layout(shape, order=order),
    )


def select(tensor: cute.Tensor, modes: list[int]) -> cute.Tensor:
    return cute.make_tensor(
        tensor.iterator,
        cute.select(tensor.layout, modes),
    )


def _accumulator_mn_layout(
    layout: cute.Layout,
    transpose: bool = False,
) -> cute.Layout:
    column_major = cute.make_layout(layout.shape)
    shape = (
        (column_major.shape[0][1], column_major.shape[1]),
        (
            column_major.shape[0][0],
            *column_major.shape[0][2:],
            column_major.shape[2],
        ),
        *column_major.shape[3:],
    )
    stride = (
        (column_major.stride[0][1], column_major.stride[1]),
        (
            column_major.stride[0][0],
            *column_major.stride[0][2:],
            column_major.stride[2],
        ),
        *column_major.stride[3:],
    )
    if const_expr(transpose):
        shape = (shape[1], shape[0], *shape[2:])
        stride = (stride[1], stride[0], *stride[2:])
    return cute.composition(
        layout,
        cute.make_layout(shape, stride=stride),
    )


def reshape_acc_to_mn(
    accumulator: cute.Tensor,
    transpose: bool = False,
) -> cute.Tensor:
    return cute.make_tensor(
        accumulator.iterator,
        _accumulator_mn_layout(accumulator.layout, transpose),
    )


@cute.jit
def _accumulator_frga_layout(layout: cute.Layout) -> cute.Layout:
    if const_expr(cute.rank(layout.shape[0]) == 3):
        divisor = 2 if const_expr(layout.shape[0][2] % 2 == 0) else 1
        divided = cute.logical_divide(
            layout,
            ((None, None, divisor), None, None),
        )
        return cute.make_layout(
            (
                (
                    divided.shape[0][0],
                    divided.shape[0][1],
                    divided.shape[0][2][0],
                ),
                divided.shape[1],
                (divided.shape[0][2][1], divided.shape[2]),
            ),
            stride=(
                (
                    divided.stride[0][0],
                    divided.stride[0][1],
                    divided.stride[0][2][0],
                ),
                divided.stride[1],
                (divided.stride[0][2][1], divided.stride[2]),
            ),
        )

    assert layout.shape[2] % 2 == 0
    divided = cute.logical_divide(layout, (None, None, 2))
    return cute.make_layout(
        (
            (
                divided.shape[0][0],
                divided.shape[0][1],
                divided.shape[2][0],
            ),
            divided.shape[1],
            divided.shape[2][1],
        ),
        stride=(
            (
                divided.stride[0][0],
                divided.stride[0][1],
                divided.stride[2][0],
            ),
            divided.stride[1],
            divided.stride[2][1],
        ),
    )


def reshape_acc_to_frgA(accumulator: cute.Tensor) -> cute.Tensor:
    return cute.make_tensor(
        accumulator.iterator,
        _accumulator_frga_layout(accumulator.layout),
    )


__all__ = [
    "reshape_acc_to_frgA",
    "reshape_acc_to_mn",
    "select",
    "transpose_view",
]
