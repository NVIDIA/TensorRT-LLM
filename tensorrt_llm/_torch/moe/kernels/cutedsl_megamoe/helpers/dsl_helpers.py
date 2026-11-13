"""General-purpose CuTe DSL helpers."""

from typing import Callable, Literal, Optional

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Pointer
from cutlass.cutlass_dsl import Boolean, Int16, Int32

from .ptx_helpers import nanosleep
from .utils import ceil_div


@cute.jit
def smem_exclusive_prefix(
    input_tensor: cute.Tensor,
    output_tensor: cute.Tensor,
    warp_totals: cute.Tensor,
    block_thread_count: int,
    thread_idx: Int32,
    lane_idx: Int32,
    warp_idx: Int32,
) -> Int32:
    """Compute a CTA-wide exclusive prefix over an Int32 SMEM tensor."""
    num_elements = cute.size(input_tensor)
    scan_rows = num_elements // 4
    num_warps = block_thread_count // 32
    input_vectors = cute.make_tensor(input_tensor.iterator, cute.make_layout((scan_rows, 4), stride=(4, 1)))
    load_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=128)

    values = cute.make_rmem_tensor((4,), cutlass.Int32)
    carry = Int32(0)
    for segment in cutlass.range_constexpr(ceil_div(scan_rows, block_thread_count)):
        row = Int32(segment * block_thread_count) + thread_idx
        if row < Int32(scan_rows):
            row_slice = input_vectors[row, None]
            pointer = row_slice.iterator
            aligned_row_slice = cute.make_tensor(
                cute.make_ptr(pointer.dtype, pointer.toint(), pointer.memspace, assumed_align=16),
                row_slice.layout,
            )
            cute.copy(load_atom, aligned_row_slice, values)
        else:
            for element in cutlass.range_constexpr(4):
                values[element] = Int32(0)

        local_prefix = (Int32(0), values[0], values[0] + values[1], values[0] + values[1] + values[2])
        lane_total = local_prefix[3] + values[3]
        inclusive = lane_total
        for step_log in cutlass.range_constexpr(5):
            step = Int32(1 << step_log)
            previous = Int32(cute.arch.shuffle_sync(inclusive, lane_idx - step))
            if lane_idx >= step:
                inclusive = inclusive + previous
        lane_base = inclusive - lane_total
        warp_total = Int32(cute.arch.shuffle_sync(inclusive, Int32(31)))
        if lane_idx == Int32(0):
            warp_totals[warp_idx] = warp_total
        cute.arch.sync_threads()

        region_total = Int32(0)
        if lane_idx < Int32(num_warps):
            region_total = warp_totals[lane_idx]
        inclusive_region = region_total
        for step_log in cutlass.range_constexpr(5):
            step = Int32(1 << step_log)
            previous = Int32(cute.arch.shuffle_sync(inclusive_region, lane_idx - step))
            if lane_idx >= step:
                inclusive_region = inclusive_region + previous
        warp_base = Int32(cute.arch.shuffle_sync(inclusive_region - region_total, warp_idx))
        segment_total = Int32(cute.arch.shuffle_sync(inclusive_region, Int32(31)))
        base = carry + warp_base + lane_base
        if row < Int32(scan_rows):
            first_element = row * Int32(4)
            for element in cutlass.range_constexpr(4):
                output_tensor[first_element + Int32(element)] = base + local_prefix[element]
        carry = carry + segment_total
        cute.arch.sync_threads()
    return carry


@cute.jit
def mark_alignment(tensor: cute.Tensor, byte_alignment: int) -> cute.Tensor:
    pointer = tensor.iterator
    return cute.make_tensor(
        cute.make_ptr(pointer.dtype, pointer.toint(), pointer.memspace, assumed_align=byte_alignment), tensor.layout
    )


@cute.jit
def spin_peek(pointer: Pointer, condition: Callable[[Int32], Boolean], scope: str = "gpu") -> Boolean:
    """Perform one acquire load and test its value."""
    value = cute.arch.load(pointer, pointer.dtype, sem="acquire", scope=scope)
    return Boolean(condition(value))


@cute.jit
def spin_wait(
    pointer: Pointer,
    condition: Callable[[Int32], Boolean],
    scope: str = "gpu",
    sleep_cycles: int = 150,
    peek_status: Optional[Boolean] = None,
) -> None:
    """Wait until an acquire-loaded value satisfies the condition."""
    wait_required = Boolean(True)
    if cutlass.const_expr(peek_status is not None):
        wait_required = not peek_status

    if wait_required:
        value = cute.arch.load(pointer, pointer.dtype, sem="acquire", scope=scope)
        while not condition(value):
            nanosleep(sleep_cycles)
            value = cute.arch.load(pointer, pointer.dtype, sem="acquire", scope=scope)


def _tma_multicast_pattern(
    cluster_mn: tuple[int, int], mma_cta_count: int, tensor_role: Literal["a", "b", "sfa", "sfb"]
) -> int:
    cluster_m, cluster_n = cluster_mn
    if tensor_role in ("a", "sfa"):
        return sum(1 << (cluster_n_index * cluster_m) for cluster_n_index in range(cluster_n))
    if tensor_role == "b":
        return sum(1 << (cluster_m_index * mma_cta_count) for cluster_m_index in range(cluster_m // mma_cta_count))
    if tensor_role == "sfb":
        return (1 << cluster_m) - 1
    raise ValueError(f"Unsupported TMA tensor role {tensor_role!r}.")


@cute.jit
def tma_multicast_mask(
    preferred_cluster_mn: tuple[int, int],
    fallback_cluster_mn: Optional[tuple[int, int]],
    cta_coord_in_cluster: cute.Coord,
    is_preferred: Optional[Boolean],
    is_2cta: bool,
    tensor_role: Literal["a", "b", "sfa", "sfb"],
) -> Int16:
    """Build a preferred/fallback TMA multicast mask."""
    preferred_m, preferred_n = preferred_cluster_mn
    if cutlass.const_expr(preferred_m <= 0 or preferred_n <= 0 or preferred_m * preferred_n > 16):
        raise ValueError(f"Invalid preferred cluster shape {preferred_cluster_mn}.")

    mma_cta_count = 2 if cutlass.const_expr(is_2cta) else 1
    if cutlass.const_expr(preferred_m % mma_cta_count != 0):
        raise ValueError("Preferred cluster M must be divisible by the MMA CTA count.")

    preferred_pattern = _tma_multicast_pattern(preferred_cluster_mn, mma_cta_count, tensor_role)
    cta_m = Int32(cta_coord_in_cluster[0])
    if cutlass.const_expr(fallback_cluster_mn is None):
        if cutlass.const_expr(tensor_role in ("a", "sfa")):
            result = cute.arch.inline_ptx(
                f"shl.b16 {{$w0}}, 0x{preferred_pattern:04x}, {{$r0}};",
                write_only_types=[Int16],
                read_only_args=[cta_m],
            )
        else:
            cta_n = Int32(cta_coord_in_cluster[1])
            mma_cta_index = Int32(0) if cutlass.const_expr(tensor_role == "sfb" or mma_cta_count == 1) else cta_m % 2
            result = cute.arch.inline_ptx(
                "{\n\t"
                ".reg .u32 offset;\n\t"
                f"mad.lo.u32 offset, {{$r0}}, {preferred_m}, {{$r1}};\n\t"
                f"shl.b16 {{$w0}}, 0x{preferred_pattern:04x}, offset;\n\t"
                "}",
                write_only_types=[Int16],
                read_only_args=[cta_n, mma_cta_index],
            )
        return Int16(result)

    fallback_m, fallback_n = fallback_cluster_mn
    if cutlass.const_expr(fallback_m <= 0 or fallback_n <= 0 or fallback_m * fallback_n > 16):
        raise ValueError(f"Invalid fallback cluster shape {fallback_cluster_mn}.")
    if cutlass.const_expr(preferred_m % fallback_m != 0 or preferred_n % fallback_n != 0):
        raise ValueError("Preferred cluster dimensions must be divisible by fallback dimensions.")
    if cutlass.const_expr(fallback_m % mma_cta_count != 0):
        raise ValueError("Fallback cluster M must be divisible by the MMA CTA count.")

    fallback_pattern = _tma_multicast_pattern(fallback_cluster_mn, mma_cta_count, tensor_role)
    if cutlass.const_expr(preferred_pattern == fallback_pattern):
        if cutlass.const_expr(tensor_role in ("a", "sfa")):
            result = cute.arch.inline_ptx(
                f"shl.b16 {{$w0}}, 0x{preferred_pattern:04x}, {{$r0}};",
                write_only_types=[Int16],
                read_only_args=[cta_m],
            )
        else:
            cta_n = Int32(cta_coord_in_cluster[1])
            mma_cta_index = Int32(0) if cutlass.const_expr(tensor_role == "sfb" or mma_cta_count == 1) else cta_m % 2
            result = cute.arch.inline_ptx(
                "{\n\t"
                ".reg .u32 offset;\n\t"
                f"mad.lo.u32 offset, {{$r0}}, {preferred_m}, {{$r1}};\n\t"
                f"shl.b16 {{$w0}}, 0x{preferred_pattern:04x}, offset;\n\t"
                "}",
                write_only_types=[Int16],
                read_only_args=[cta_n, mma_cta_index],
            )
        return Int16(result)

    if cutlass.const_expr(is_preferred is None):
        raise ValueError("is_preferred is required when the preferred and fallback multicast patterns differ.")

    if cutlass.const_expr(tensor_role in ("a", "sfa")):
        result = cute.arch.inline_ptx(
            "{\n\t"
            f"mov.b16 {{$w0}}, 0x{preferred_pattern:04x};\n\t"
            f"@!{{$r0}} mov.b16 {{$w0}}, 0x{fallback_pattern:04x};\n\t"
            "shl.b16 {$w0}, {$w0}, {$r1};\n\t"
            "}",
            write_only_types=[Int16],
            read_only_args=[is_preferred, cta_m],
        )
    else:
        if cutlass.const_expr(fallback_pattern & preferred_pattern != fallback_pattern):
            raise ValueError("Fallback B/SFB multicast pattern must be a subset of the preferred pattern.")
        cta_n = Int32(cta_coord_in_cluster[1])
        mma_cta_index = Int32(0) if cutlass.const_expr(tensor_role == "sfb" or mma_cta_count == 1) else cta_m % 2
        result = cute.arch.inline_ptx(
            "{\n\t"
            ".reg .u32 cluster_m, offset;\n\t"
            f"mov.u32 cluster_m, {preferred_m};\n\t"
            f"@!{{$r0}} mov.u32 cluster_m, {fallback_m};\n\t"
            "mad.lo.u32 offset, {$r1}, cluster_m, {$r2};\n\t"
            f"mov.b16 {{$w0}}, 0x{preferred_pattern:04x};\n\t"
            f"@!{{$r0}} and.b16 {{$w0}}, {{$w0}}, 0x{fallback_pattern:04x};\n\t"
            "shl.b16 {$w0}, {$w0}, offset;\n\t"
            "}",
            write_only_types=[Int16],
            read_only_args=[is_preferred, cta_n, mma_cta_index],
        )
    return Int16(result)


__all__ = ["mark_alignment", "smem_exclusive_prefix", "spin_peek", "spin_wait", "tma_multicast_mask"]
