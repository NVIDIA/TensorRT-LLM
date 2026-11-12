from dataclasses import dataclass
from math import gcd
from typing import Optional, Tuple, Union

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import OperandMajorMode

from .smem_workspace import SmemRegion


# Every block-scaled tcgen05 MMA kind accumulates in F32 and no other accumulator is legal, so
# this belongs to the instruction family rather than being a caller choice.
tcgen05_block_scaled_acc_dtype = cutlass.Float32


@dataclass(frozen=True)
class Tcgen05MmaInstruction:
    a_type: type[cutlass.Numeric]
    b_type: type[cutlass.Numeric]
    instruction_mnk: Tuple[int, int, int]
    participates: int
    acc_type: type[cutlass.Numeric] = cutlass.Float32
    sfa_type: Optional[type[cutlass.Numeric]] = None
    sfb_type: Optional[type[cutlass.Numeric]] = None
    sf_vec_size: Optional[int] = None


@dataclass(frozen=True)
class Tcgen05TmemPlan:
    """Column-level TMEM placement and accumulator staging contract."""

    allocation_columns: int
    accumulator_columns: int
    accumulator_stage_columns: int
    accumulator_stage_count: int
    accumulator_stage_stride_columns: int
    accumulator_pipeline_stages: int
    sfa_columns: int
    sfb_columns: int


def make_tcgen05_tmem_plan(
    mma_instruction: Tcgen05MmaInstruction, arch: str, mma_tiler_mnk: Tuple[int, int, int]
) -> Tcgen05TmemPlan:
    """Plan TMEM for canonical dense or block-scaled TCGen05 MMA.

    This planner requires A and B to reside in SMEM, per-CTA M to equal 128,
    and tile M to equal instruction M. A-from-TMEM, sparse MMA, B-reuse, and
    custom atom layouts or permutations are outside its contract.
    SM100/SM103 use the canonical two-stage overlap for 256-column block-scaled
    accumulators; other supported cases maximize disjoint accumulator stages.
    """
    if mma_instruction.participates not in (1, 2):
        raise ValueError(f"TCGen05 MMA participates must be one or two, got {mma_instruction.participates}.")
    if len(mma_instruction.instruction_mnk) != 3 or len(mma_tiler_mnk) != 3:
        raise ValueError("instruction_mnk and mma_tiler_mnk must each contain three dimensions.")
    if any(dimension <= 0 for dimension in (*mma_instruction.instruction_mnk, *mma_tiler_mnk)):
        raise ValueError("MMA instruction and tiler dimensions must be positive.")

    instruction_m, instruction_n, instruction_k = mma_instruction.instruction_mnk
    tile_m, tile_n, tile_k = mma_tiler_mnk
    if instruction_m % mma_instruction.participates != 0 or instruction_n % mma_instruction.participates != 0:
        raise ValueError("MMA instruction M and N must be divisible by participates.")
    if tile_m % instruction_m != 0 or tile_n % instruction_n != 0 or tile_k % instruction_k != 0:
        raise ValueError("MMA instruction dimensions must divide mma_tiler_mnk.")
    if tile_m != instruction_m:
        raise ValueError("TCGen05 TMEM planning does not support M repetition or B-reuse.")
    if tile_m // mma_instruction.participates != 128:
        raise ValueError("TCGen05 TMEM planning requires per-CTA M to equal 128.")

    sf_fields = (mma_instruction.sfa_type, mma_instruction.sfb_type, mma_instruction.sf_vec_size)
    has_scale_factors = any(value is not None for value in sf_fields)
    if has_scale_factors and any(value is None for value in sf_fields):
        raise ValueError("sfa_type, sfb_type, and sf_vec_size must be provided together.")

    sfa_columns = 0
    sfb_columns = 0
    if has_scale_factors:
        sf_vec_size = mma_instruction.sf_vec_size
        if not isinstance(sf_vec_size, int) or isinstance(sf_vec_size, bool) or sf_vec_size <= 0:
            raise ValueError("sf_vec_size must be a positive Python int.")
        if instruction_k % sf_vec_size != 0:
            raise ValueError("Instruction K must be divisible by sf_vec_size.")
        if tile_k % (sf_vec_size * 4) != 0:
            raise ValueError("Tile K must contain complete block-scaled basic chunks.")
        sfa_columns = tile_k // sf_vec_size
        sfb_columns = max(tile_n // 128, 1) * tile_k // sf_vec_size

    tmem_column_capacity = cute.arch.get_max_tmem_alloc_cols(arch)
    accumulator_stage_columns = tile_n
    scale_factor_columns = sfa_columns + sfb_columns
    accumulator_stage_count = (tmem_column_capacity - scale_factor_columns) // accumulator_stage_columns
    if accumulator_stage_count < 1:
        raise ValueError("Scale-factor TMEM leaves no accumulator stage.")

    accumulator_stage_stride_columns = accumulator_stage_columns
    accumulator_pipeline_stages = accumulator_stage_count
    accumulator_columns = accumulator_stage_columns * accumulator_stage_count

    arch_number = _parse_arch_number(arch)
    use_sm100_overlap = (
        arch_number in (100, 103)
        and has_scale_factors
        and accumulator_stage_columns == 256
        and accumulator_stage_count == 1
        and scale_factor_columns <= 64
    )
    if use_sm100_overlap:
        accumulator_stage_count = 2
        accumulator_stage_stride_columns = accumulator_stage_columns - scale_factor_columns
        accumulator_pipeline_stages = 1
        accumulator_columns = accumulator_stage_columns + accumulator_stage_stride_columns

    used_columns = accumulator_columns + scale_factor_columns
    allocation_columns = _round_tmem_allocation_columns(used_columns, tmem_column_capacity)
    return Tcgen05TmemPlan(
        allocation_columns=allocation_columns,
        accumulator_columns=accumulator_columns,
        accumulator_stage_columns=accumulator_stage_columns,
        accumulator_stage_count=accumulator_stage_count,
        accumulator_stage_stride_columns=accumulator_stage_stride_columns,
        accumulator_pipeline_stages=accumulator_pipeline_stages,
        sfa_columns=sfa_columns,
        sfb_columns=sfb_columns,
    )


def _parse_arch_number(arch: str) -> int:
    if not isinstance(arch, str):
        raise TypeError(f"arch must be a string, got {type(arch)}.")
    normalized = arch.lower()
    if normalized.startswith("sm_"):
        normalized = normalized[3:]
    elif normalized.startswith("sm"):
        normalized = normalized[2:]
    digits = []
    for character in normalized:
        if not character.isdigit():
            break
        digits.append(character)
    if not digits:
        raise ValueError(f"Cannot parse architecture {arch!r}.")
    return int("".join(digits))


def _round_tmem_allocation_columns(used_columns: int, capacity_columns: int) -> int:
    if used_columns <= 0:
        raise ValueError("TMEM usage must be positive.")
    if used_columns <= 512:
        allocation_columns = max(32, 1 << (used_columns - 1).bit_length())
    else:
        allocation_columns = _round_up(used_columns, 32)
    if allocation_columns > capacity_columns:
        raise ValueError(f"TMEM plan needs {allocation_columns} columns, exceeding {capacity_columns}.")
    return allocation_columns


def tcgen05_smem_alloc_type(
    dtype: type[cutlass.Numeric], peer_dtype: type[cutlass.Numeric], arch: str
) -> type[cutlass.Numeric]:
    """SMEM container type for one block-scaled TCGen05 operand.

    Blackwell mixed-width MMA consumes a uniform byte-per-element SMEM image, so its narrow operand
    arrives through U4_UNPACK_U8. Rubin consumes mixed FP4 directly from packed SMEM. A 6-bit
    operand always uses U6_UNPACK_U8 because no packed U6 TMA format exists.
    """
    arch_number = _parse_arch_number(arch)
    if arch_number not in (100, 103, 107):
        raise ValueError(f"Unsupported TCGen05 architecture {arch!r}.")
    needs_mixed_width_unpack = arch_number in (100, 103) and dtype.width != peer_dtype.width
    needs_unpack = needs_mixed_width_unpack or 6 in (dtype.width, peer_dtype.width)
    return cutlass.Int8 if (needs_unpack and dtype.width < 8) else dtype


def make_smem_layouts(
    mma_inst: Tcgen05MmaInstruction,
    mma_tiler_mnk: Tuple[int, int, int],
    stages: Union[int, Tuple[int, ...]],
    ab_gmem_major_modes: Tuple[OperandMajorMode, OperandMajorMode],
    arch: str,
) -> Union[Tuple[SmemRegion, SmemRegion], Tuple[SmemRegion, SmemRegion, SmemRegion, SmemRegion]]:
    """Derive workspace-ready TCGen05 operand regions without an MLIR context."""
    has_scale_factors = any(value is not None for value in (mma_inst.sfa_type, mma_inst.sfb_type, mma_inst.sf_vec_size))
    if has_scale_factors and any(
        value is None for value in (mma_inst.sfa_type, mma_inst.sfb_type, mma_inst.sf_vec_size)
    ):
        raise ValueError("sfa_type, sfb_type, and sf_vec_size must be provided together.")

    operand_count = 4 if has_scale_factors else 2
    if isinstance(stages, int):
        operand_stages = (stages,) * operand_count
    else:
        operand_stages = stages
        if len(operand_stages) != operand_count:
            raise ValueError(f"Expected {operand_count} stage counts, got {len(operand_stages)}.")
    if any(stage <= 0 for stage in operand_stages):
        raise ValueError(f"All stage counts must be positive, got {operand_stages}.")

    if mma_inst.participates not in (1, 2):
        raise ValueError(f"TCGen05 MMA participates must be one or two, got {mma_inst.participates}.")
    if len(mma_inst.instruction_mnk) != 3 or len(mma_tiler_mnk) != 3:
        raise ValueError("instruction_mnk and mma_tiler_mnk must each contain three dimensions.")
    if any(dimension <= 0 for dimension in (*mma_inst.instruction_mnk, *mma_tiler_mnk)):
        raise ValueError("MMA instruction and tiler dimensions must be positive.")

    inst_m, inst_n, inst_k = mma_inst.instruction_mnk
    tile_m, tile_n, tile_k = mma_tiler_mnk
    if inst_m % mma_inst.participates != 0 or inst_n % mma_inst.participates != 0:
        raise ValueError("MMA instruction M and N must be divisible by participates.")
    if tile_m % inst_m != 0 or tile_n % inst_n != 0 or tile_k % inst_k != 0:
        raise ValueError("MMA instruction dimensions must divide mma_tiler_mnk.")

    a_region = _make_ab_region(
        mma_inst.a_type,
        tcgen05_smem_alloc_type(mma_inst.a_type, mma_inst.b_type, arch),
        tile_m // mma_inst.participates,
        tile_k,
        (inst_m // mma_inst.participates, inst_k),
        operand_stages[0],
        ab_gmem_major_modes[0],
    )
    b_region = _make_ab_region(
        mma_inst.b_type,
        tcgen05_smem_alloc_type(mma_inst.b_type, mma_inst.a_type, arch),
        tile_n // mma_inst.participates,
        tile_k,
        (inst_n // mma_inst.participates, inst_k),
        operand_stages[1],
        ab_gmem_major_modes[1],
    )
    if not has_scale_factors:
        return a_region, b_region

    sfa_region = _make_sf_region(
        mma_inst.sfa_type,
        inst_m // mma_inst.participates // 128,
        tile_m // inst_m,
        tile_k // inst_k,
        inst_k,
        mma_inst.sf_vec_size,
        operand_stages[2],
    )
    sfb_region = _make_sf_region(
        mma_inst.sfb_type,
        _round_up(inst_n, 128) // 128,
        tile_n // inst_n,
        tile_k // inst_k,
        inst_k,
        mma_inst.sf_vec_size,
        operand_stages[3],
    )
    return a_region, b_region, sfa_region, sfb_region


def _round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def _canonical_mode(shape, stride):
    if not isinstance(shape, tuple):
        return (1, 0) if shape == 1 else (shape, stride)

    result_shape = []
    result_stride = []
    for current_shape, current_stride in zip(shape, stride):
        if current_shape == 1:
            continue
        if result_shape and result_shape[-1] * result_stride[-1] == current_stride:
            result_shape[-1] *= current_shape
        else:
            result_shape.append(current_shape)
            result_stride.append(current_stride)
    if not result_shape:
        return 1, 0
    if len(result_shape) == 1:
        return result_shape[0], result_stride[0]
    return tuple(result_shape), tuple(result_stride)


def _shape_size(shape) -> int:
    if isinstance(shape, tuple):
        result = 1
        for child_shape in shape:
            result *= _shape_size(child_shape)
        return result
    return shape


def _prefix_tile_profile(shape, tile_extent: int):
    """Represent a scalar prefix tile using the source mode boundaries."""
    if tile_extent <= 0 or _shape_size(shape) % tile_extent != 0:
        raise ValueError(f"Tile extent {tile_extent} must divide mode {shape}.")
    if not isinstance(shape, tuple):
        return tile_extent

    tile_modes = []
    remaining_extent = tile_extent
    for child_shape in shape:
        if remaining_extent == 1:
            break

        child_size = _shape_size(child_shape)
        if remaining_extent >= child_size and remaining_extent % child_size == 0:
            tile_modes.append(child_shape)
            remaining_extent //= child_size
        elif child_size % remaining_extent == 0:
            tile_modes.append(_prefix_tile_profile(child_shape, remaining_extent))
            remaining_extent = 1
        else:
            raise ValueError(f"Tile extent {tile_extent} does not divide a prefix of mode {shape}.")

    if remaining_extent != 1:
        raise ValueError(f"Tile extent {tile_extent} exceeds the prefix of mode {shape}.")
    if not tile_modes:
        return 1
    if len(tile_modes) == 1:
        return tile_modes[0]
    return tuple(tile_modes)


def _flatten_mode(shape, stride):
    if isinstance(shape, tuple):
        result = []
        for child_shape, child_stride in zip(shape, stride):
            result.extend(_flatten_mode(child_shape, child_stride))
        return result
    return [(shape, stride)]


def _flatten_shape(shape):
    if isinstance(shape, tuple):
        result = []
        for child_shape in shape:
            result.extend(_flatten_shape(child_shape))
        return result
    return [shape]


def _rebuild_stride_like(shape, flat_strides):
    stride_iterator = iter(flat_strides)

    def rebuild(current_shape):
        if isinstance(current_shape, tuple):
            return tuple(rebuild(child_shape) for child_shape in current_shape)
        return next(stride_iterator)

    return rebuild(shape)


def _divide_mode_by_tile(shape, stride, tile_shape):
    """Apply a static tile profile and return its stride and canonical rest."""
    remaining_modes = _flatten_mode(shape, stride)
    tile_strides = []
    for tile_extent in _flatten_shape(tile_shape):
        if not remaining_modes:
            raise ValueError(f"Tile {tile_shape} exceeds mode {shape}.")

        mode_extent, mode_stride = remaining_modes[0]
        if tile_extent <= 0 or mode_extent % tile_extent != 0:
            raise ValueError(f"Tile mode {tile_extent} must divide source mode {mode_extent}.")

        tile_strides.append(mode_stride if tile_extent > 1 else 0)
        rest_extent = mode_extent // tile_extent
        if rest_extent > 1:
            remaining_modes[0] = (rest_extent, mode_stride * tile_extent)
        else:
            remaining_modes.pop(0)

    rest_shape, rest_stride = _canonical_mode(
        tuple(mode_extent for mode_extent, _ in remaining_modes),
        tuple(mode_stride for _, mode_stride in remaining_modes),
    )
    return (_rebuild_stride_like(tile_shape, tile_strides), rest_shape, rest_stride)


def _tiled_divide_2d(shape, stride, tile_shape):
    """Tile a static rank-2 layout and group value modes before rest modes."""
    if (
        not isinstance(shape, tuple)
        or not isinstance(stride, tuple)
        or not isinstance(tile_shape, tuple)
        or len(shape) != 2
        or len(stride) != 2
        or len(tile_shape) != 2
    ):
        raise ValueError("A 2-D tiled divide requires rank-2 shape, stride, and tile.")

    mn_tile_stride, mn_rest_shape, mn_rest_stride = _divide_mode_by_tile(shape[0], stride[0], tile_shape[0])
    k_tile_stride, k_rest_shape, k_rest_stride = _divide_mode_by_tile(shape[1], stride[1], tile_shape[1])
    return (
        ((tile_shape[0], tile_shape[1]), mn_rest_shape, k_rest_shape),
        ((mn_tile_stride, k_tile_stride), mn_rest_stride, k_rest_stride),
    )


def _make_ab_region(
    dtype: type[cutlass.Numeric],
    alloc_dtype: type[cutlass.Numeric],
    mn_extent: int,
    k_extent: int,
    value_shape: Tuple[int, int],
    stages: int,
    major_mode: OperandMajorMode,
) -> SmemRegion:
    """Plan one operand's SMEM region.

    ``dtype`` is the logical element type the MMA sees; ``alloc_dtype`` is the container it
    occupies in SMEM. The two differ only for a sub-byte operand loaded through the unpacking TMA
    (see ``tcgen05_smem_alloc_type``), where the layout must be sized and swizzled for 1-byte
    containers while the major-mode rule still follows the logical type.
    """
    if dtype.width in (4, 6) and major_mode != OperandMajorMode.K:
        raise ValueError(f"{dtype} TCGen05 operands require K-major SMEM.")

    leading_extent = k_extent if major_mode == OperandMajorMode.K else mn_extent
    leading_bits = leading_extent * alloc_dtype.width
    if leading_bits % 8 != 0:
        raise ValueError("The leading dimension must occupy a whole number of bytes.")
    swizzle_bytes = gcd(leading_bits // 8, 128)
    swizzle_by_bytes = {16: (0, 4, 3), 32: (1, 4, 3), 64: (2, 4, 3), 128: (3, 4, 3)}
    if swizzle_bytes not in swizzle_by_bytes:
        raise ValueError(f"Unsupported leading dimension size {leading_bits // 8} bytes.")
    swizzle = swizzle_by_bytes[swizzle_bytes]
    if major_mode == OperandMajorMode.MN and alloc_dtype.width == 32 and swizzle_bytes == 128:
        swizzle = (2, 5, 2)

    chunk_elements = swizzle_bytes * 8 // alloc_dtype.width
    if leading_extent % chunk_elements != 0:
        raise ValueError(f"Leading extent {leading_extent} must be divisible by {chunk_elements}.")
    repeats = leading_extent // chunk_elements
    stage_stride = mn_extent * k_extent

    if major_mode == OperandMajorMode.K:
        mn_shape = mn_extent
        mn_stride = chunk_elements if repeats > 1 else k_extent
        if repeats > 1:
            k_shape = (chunk_elements, repeats)
            k_stride = (1, mn_extent * chunk_elements)
        else:
            k_shape = k_extent
            k_stride = 1
    else:
        k_shape = k_extent
        k_stride = chunk_elements if repeats > 1 else mn_extent
        if repeats > 1:
            mn_shape = (chunk_elements, repeats)
            mn_stride = (1, k_extent * chunk_elements)
        else:
            mn_shape = mn_extent
            mn_stride = 1

    mma_mn_shape = _prefix_tile_profile(mn_shape, value_shape[0])
    mma_k_shape = _prefix_tile_profile(k_shape, value_shape[1])
    mma_shape, mma_stride = _tiled_divide_2d((mn_shape, k_shape), (mn_stride, k_stride), (mma_mn_shape, mma_k_shape))
    return SmemRegion(
        name="",
        kind="tensor",
        dtype=alloc_dtype,
        shape=(*mma_shape, stages),
        stride=(*mma_stride, stage_stride if stages > 1 else 0),
        swizzle=swizzle,
        byte_alignment=128,
    )


def _make_sf_region(
    dtype: type[cutlass.Numeric],
    instruction_mn_blocks: int,
    mn_iterations: int,
    k_iterations: int,
    instruction_k: int,
    sf_vec_size: int,
    stages: int,
) -> SmemRegion:
    if instruction_mn_blocks <= 0:
        raise ValueError("A scale-factor instruction must cover at least one 128-element MN block.")
    if instruction_k % sf_vec_size != 0:
        raise ValueError("instruction K must be divisible by sf_vec_size.")

    cta_mn_blocks = instruction_mn_blocks * mn_iterations
    cta_k = instruction_k * k_iterations
    basic_chunk_k = sf_vec_size * 4
    if cta_k % basic_chunk_k != 0:
        raise ValueError("The CTA K extent must contain complete block-scaled basic chunks.")
    basic_chunk_repetitions = cta_k // basic_chunk_k

    full_mn_shape = ((32, 4), cta_mn_blocks)
    full_mn_stride = ((16, 4), basic_chunk_repetitions * 512 if cta_mn_blocks > 1 else 0)
    full_k_shape = ((sf_vec_size, 4), basic_chunk_repetitions)
    full_k_stride = ((0, 1), 512 if basic_chunk_repetitions > 1 else 0)

    mma_mn_shape = ((32, 4), instruction_mn_blocks)
    mma_k_shape = (sf_vec_size, _prefix_tile_profile((4, basic_chunk_repetitions), instruction_k // sf_vec_size))
    mma_shape, mma_stride = _tiled_divide_2d(
        (full_mn_shape, full_k_shape), (full_mn_stride, full_k_stride), (mma_mn_shape, mma_k_shape)
    )
    stage_stride = cta_mn_blocks * basic_chunk_repetitions * 512
    return SmemRegion(
        name="",
        kind="tensor",
        dtype=dtype,
        shape=(*mma_shape, stages),
        stride=(*mma_stride, stage_stride if stages > 1 else 0),
        swizzle=(0, 4, 3),
        byte_alignment=128,
    )


__all__ = [
    "Tcgen05MmaInstruction",
    "Tcgen05TmemPlan",
    "make_smem_layouts",
    "make_tcgen05_tmem_plan",
    "tcgen05_block_scaled_acc_dtype",
    "tcgen05_smem_alloc_type",
]
