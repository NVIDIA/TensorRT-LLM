"""Small integer and layout helpers shared by workspace implementations."""

from typing import Iterable, List, Tuple, Union

import cutlass


IntegerType = Union[int, cutlass.Int32, cutlass.Int64, cutlass.Uint32, cutlass.Uint64]


def round_up(value: IntegerType, alignment: IntegerType) -> IntegerType:
    return ((value + alignment - 1) // alignment) * alignment


def ceil_div(value: IntegerType, divisor: IntegerType) -> IntegerType:
    return (value + divisor - 1) // divisor


def padded_expert_rows(token_count: IntegerType, padding_block: IntegerType) -> IntegerType:
    """Row span one expert occupies in a block-padded pool.

    The single definition of a pool's per-expert stride. Communication components
    write bases derived from it while the FC12 scheduler rebuilds the same bases
    from ``expert_sizes`` at runtime; if the two ever disagree the metadata a
    kernel reads no longer describes the rows it loads.
    """
    return round_up(token_count, padding_block)


def worst_case_padded_rows(source_tokens: int, local_experts: int, topk: int, padding_block: int) -> int:
    """Worst-case rows of a block-padded pool: ``source_tokens`` tokens, each routed to at most ``topk`` of the
    ``local_experts`` experts of this rank, every expert padded to whole ``padding_block`` rows.

    The single definition of that bound: the routers size their pools with it, and a kernel asks it before it has
    a router (ring_token_blocks_in_effect).
    """
    if padding_block <= 0:
        raise ValueError(f"The padding block of a pool must be positive, got {padding_block}.")
    route_capacity = source_tokens * min(topk, local_experts)
    active_expert_capacity = min(local_experts, route_capacity)
    route_budget_blocks = active_expert_capacity + (route_capacity - active_expert_capacity) // padding_block
    expert_bound_blocks = active_expert_capacity * int(ceil_div(source_tokens, padding_block))
    return min(route_budget_blocks, expert_bound_blocks) * padding_block


def ring_token_blocks_in_effect(
    requested,
    *,
    tokens_per_block: int,
    source_tokens: int,
    local_experts: int,
    topk: int,
    token_padding_block: int,
    sf_padding_block: int,
) -> int:
    """The ring workspace a kernel runs with, in token blocks, for the ``ring_token_blocks`` its caller asked for.

    None and 0 ask for no ring. The pools hold at most ``requested`` token blocks; where that many blocks are not
    fewer than the worst case of the problem, the kernel is the kernel without the key, and this returns 0. A
    caller cannot compute that limit, so reaching it is not an error. The scale rows count as well: a ring slot
    owns whole 128-row scale atoms, so a ring of small token tiles can have fewer data rows than the worst case
    and still more scale rows.
    """
    if requested is None:
        return 0
    if type(requested) is not int or requested < 0:
        raise ValueError(f"ring_token_blocks must be a non-negative int or None, got {requested!r}.")
    if requested == 0:
        return 0
    worst_case_rows = worst_case_padded_rows(source_tokens, local_experts, topk, token_padding_block)
    worst_case_sf_rows = worst_case_padded_rows(source_tokens, local_experts, topk, sf_padding_block)
    saves_rows = (
        requested * tokens_per_block < worst_case_rows
        and requested * round_up(tokens_per_block, 128) <= worst_case_sf_rows
    )
    return requested if saves_rows else 0


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def is_nested_shape(shape: Tuple) -> bool:
    return any(isinstance(dimension, tuple) for dimension in shape)


def validate_static_integer_tuple(value: Tuple, *, field_name: str) -> None:
    for element in value:
        if isinstance(element, tuple):
            validate_static_integer_tuple(element, field_name=field_name)
        elif not isinstance(element, int):
            raise TypeError(f"{field_name} must contain Python ints, got {type(element)}.")


def flatten_shape_stride(shape: Tuple, stride: Tuple) -> List[Tuple[IntegerType, IntegerType]]:
    pairs: List[Tuple[IntegerType, IntegerType]] = []
    for size, step in zip(shape, stride):
        if isinstance(size, tuple):
            pairs.extend(flatten_shape_stride(size, step))
        else:
            pairs.append((size, step))
    return pairs


def strides_equal_ignoring_singletons(shape, lhs_stride, rhs_stride) -> bool:
    """Compare strides while ignoring leaves whose logical extent is one."""
    if isinstance(shape, tuple):
        if not isinstance(lhs_stride, tuple) or not isinstance(rhs_stride, tuple):
            return False
        if len(shape) != len(lhs_stride) or len(shape) != len(rhs_stride):
            return False
        return all(
            strides_equal_ignoring_singletons(child_shape, lhs_step, rhs_step)
            for child_shape, lhs_step, rhs_step in zip(shape, lhs_stride, rhs_stride)
        )
    return shape == 1 or lhs_stride == rhs_stride


def ordered_stride(shape: Tuple[int, ...], mem_order: Tuple[int, ...]) -> Tuple[Tuple[int, ...], int]:
    stride = [0] * len(shape)
    cosize = 1
    for mode in sorted(range(len(shape)), key=lambda index: mem_order[index]):
        stride[mode] = cosize
        cosize *= shape[mode]
    return tuple(stride), cosize


def row_major_stride(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    if is_nested_shape(shape):
        raise ValueError("A nested shape needs an explicit stride.")
    stride, _ = ordered_stride(shape, tuple(reversed(range(len(shape)))))
    return stride


def cosize_from_shape_stride_tuples(shape: Tuple, stride: Tuple) -> IntegerType:
    leaf_pairs = flatten_shape_stride(shape, stride) if shape else []
    return 1 + sum((size - 1) * step for size, step in leaf_pairs)


def product(values: Iterable[IntegerType]) -> IntegerType:
    result: IntegerType = 1
    for value in values:
        result = result * value
    return result


__all__ = [
    "IntegerType",
    "ceil_div",
    "cosize_from_shape_stride_tuples",
    "flatten_shape_stride",
    "is_nested_shape",
    "is_power_of_two",
    "ordered_stride",
    "padded_expert_rows",
    "product",
    "ring_token_blocks_in_effect",
    "row_major_stride",
    "round_up",
    "strides_equal_ignoring_singletons",
    "validate_static_integer_tuple",
    "worst_case_padded_rows",
]
