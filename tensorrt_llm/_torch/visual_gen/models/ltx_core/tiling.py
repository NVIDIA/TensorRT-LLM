from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import Callable, NamedTuple, Sequence

import torch

from tensorrt_llm._torch.visual_gen.models.ltx_core.types import VIDEO_SCALE_FACTORS, SpatioTemporalScaleFactors, VideoPixelShape


def compute_trapezoidal_mask_1d(
    length: int,
    ramp_left: int,
    ramp_right: int,
    left_starts_from_0: bool = False,
) -> torch.Tensor:
    """
    Generate a 1D trapezoidal blending mask with linear ramps.
    Args:
        length: Output length of the mask.
        ramp_left: Fade-in length on the left.
        ramp_right: Fade-out length on the right.
        left_starts_from_0: Whether the ramp starts from 0 or first non-zero value.
            Useful for temporal tiles where the first tile is causal.
    Returns:
        A 1D tensor of shape `(length,)` with values in [0, 1].
    """
    if length <= 0:
        raise ValueError("Mask length must be positive.")

    ramp_left = max(0, min(ramp_left, length))
    ramp_right = max(0, min(ramp_right, length))

    mask = torch.ones(length)

    if ramp_left > 0:
        interval_length = ramp_left + 1 if left_starts_from_0 else ramp_left + 2
        fade_in = torch.linspace(0.0, 1.0, interval_length)[:-1]
        if not left_starts_from_0:
            fade_in = fade_in[1:]
        mask[:ramp_left] *= fade_in

    if ramp_right > 0:
        fade_out = torch.linspace(1.0, 0.0, steps=ramp_right + 2)[1:-1]
        mask[-ramp_right:] *= fade_out

    return mask.clamp_(0, 1)


def compute_rectangular_mask_1d(
    length: int,
    left_ramp: int,
    right_ramp: int,
) -> torch.Tensor:
    """
    Generate a 1D rectangular (pulse) mask.
    Args:
        length: Output length of the mask.
        left_ramp: Number of elements at the start of the mask to set to 0.
        right_ramp: Number of elements at the end of the mask to set to 0.
    Returns:
        A 1D tensor of shape `(length,)` with values 0 or 1.
    """
    if length <= 0:
        raise ValueError("Mask length must be positive.")

    mask = torch.ones(length)
    if left_ramp > 0:
        mask[:left_ramp] = 0
    if right_ramp > 0:
        mask[-right_ramp:] = 0
    return mask


@dataclass(frozen=True)
class DimensionInterval:
    start: int
    end: int
    left_ramp: int
    right_ramp: int


@dataclass(frozen=True)
class DimensionIntervals:
    """Intervals which a single dimension of the latent space is split into.
    Each interval is defined by its start, end, left ramp, and right ramp.
    The start and end are the indices of the first and last element (exclusive) in the interval.
    Ramps are regions of the interval where the value of the mask tensor is
    interpolated between 0 and 1 for blending with neighboring intervals.
    The left ramp and right ramp values are the lengths of the left and right ramps.
    """

    intervals: list[DimensionInterval]


@dataclass(frozen=True)
class LatentIntervals:
    """Intervals which the latent tensor of given shape is split into.
    Each dimension of the latent space is split into intervals based on the length along said dimension.
    """

    original_shape: torch.Size
    dimension_intervals: tuple[DimensionIntervals, ...]


# Operation to split a single dimension of the tensor into intervals based on the length along the dimension.
SplitOperation = Callable[[int], DimensionIntervals]
# Operation to map the intervals in input dimension to slices and masks along a corresponding output dimension.
MappingOperation = Callable[[DimensionIntervals], tuple[list[slice], list[torch.Tensor]]]


def default_split_operation(length: int) -> DimensionIntervals:
    return DimensionIntervals(intervals=[DimensionInterval(start=0, end=length, left_ramp=0, right_ramp=0)])


DEFAULT_SPLIT_OPERATION: SplitOperation = default_split_operation


def untiled_mask_1d() -> torch.Tensor:
    """Length-1 ones that broadcast over an untiled axis (historical ``None`` mask)."""
    return torch.ones(1)


def default_mapping_operation(
    _intervals: DimensionIntervals,
) -> tuple[list[slice], list[torch.Tensor]]:
    return [slice(0, None)], [untiled_mask_1d()]


DEFAULT_MAPPING_OPERATION: MappingOperation = default_mapping_operation


# ---------------------------------------------------------------------------
# Split functions
# ---------------------------------------------------------------------------


def _grow_last_tile_to_min(intervals: list[DimensionInterval], min_tile_size: int) -> list[DimensionInterval]:
    """Grow a short last tile left to ``min_tile_size``; widen penultimate ``right_ramp``."""
    if len(intervals) <= 1:
        return list(intervals)
    last = intervals[-1]
    if last.end - last.start >= min_tile_size:
        return list(intervals)
    new_start = last.end - min_tile_size
    prev = intervals[-2]
    new_overlap = prev.end - new_start
    return [
        *intervals[:-2],
        replace(prev, right_ramp=new_overlap),
        replace(last, start=new_start, left_ramp=new_overlap),
    ]


def _validate_tile_intervals(intervals: list[DimensionInterval], *, dim_size: int, min_tile_size: int) -> None:
    """Validate coverage, ramp/overlap consistency, and ``min_tile_size``."""
    if not intervals or intervals[0].start != 0 or intervals[-1].end != dim_size:
        raise ValueError(f"tiles must cover [0, {dim_size})")
    for i, iv in enumerate(intervals):
        length = iv.end - iv.start
        if length < min_tile_size:
            raise ValueError(f"tile {i} length {length} is below min_tile_size={min_tile_size}")
        if iv.left_ramp < 0 or iv.right_ramp < 0 or iv.left_ramp > length or iv.right_ramp > length:
            raise ValueError(f"tile {i} has invalid ramps: left={iv.left_ramp}, right={iv.right_ramp}, length={length}")
        if i == 0:
            continue
        overlap = intervals[i - 1].end - iv.start
        if overlap < 0 or intervals[i - 1].right_ramp != overlap or iv.left_ramp != overlap:
            raise ValueError(f"tiles {i - 1}/{i}: ramp/overlap mismatch (overlap={overlap})")


def split_by_size(size: int, overlap: int, min_tile_size: int | None = None) -> SplitOperation:
    """Split a dimension into overlapping tiles of a given size.
    Tiles are sized ``size`` with ``overlap`` shared elements between
    consecutive tiles.  The last tile may be shorter if the dimension
    doesn't divide evenly.  If ``min_tile_size`` is set and the last tile is
    shorter, it is grown leftward (penultimate ``right_ramp`` widens); the
    result is validated and invalid layouts raise ``ValueError``.
    Args:
        size: Target tile size (in axis units).
        overlap: Overlap between consecutive tiles.
        min_tile_size: Optional minimum tile length. ``None`` keeps legacy
            short-last-tile behavior.
    Returns:
        A split operation that divides a dimension into tiles.
    """
    if size <= 0:
        raise ValueError(f"size must be > 0, got {size}")
    if overlap < 0 or overlap >= size:
        raise ValueError(f"overlap must satisfy 0 <= overlap < size, got overlap={overlap}, size={size}")
    if min_tile_size is not None and min_tile_size < 1:
        raise ValueError(f"min_tile_size must be >= 1, got {min_tile_size}")

    def split(dimension_size: int) -> DimensionIntervals:
        if min_tile_size is not None and dimension_size < min_tile_size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        if dimension_size <= size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        amount = (dimension_size + size - 2 * overlap - 1) // (size - overlap)
        intervals = [
            DimensionInterval(start=0, end=size, left_ramp=0, right_ramp=overlap),
            *(
                DimensionInterval(
                    start=i * (size - overlap),
                    end=i * (size - overlap) + size,
                    left_ramp=overlap,
                    right_ramp=overlap,
                )
                for i in range(1, amount - 1)
            ),
            DimensionInterval(
                start=(amount - 1) * (size - overlap), end=dimension_size, left_ramp=overlap, right_ramp=0
            ),
        ]
        if min_tile_size is not None:
            intervals = _grow_last_tile_to_min(intervals, min_tile_size)
            _validate_tile_intervals(intervals, dim_size=dimension_size, min_tile_size=min_tile_size)
        return DimensionIntervals(intervals=intervals)

    return split


def split_by_size_pinned(size: int, min_overlap: int) -> SplitOperation:
    """Split into equal-size tiles with first and last origins pinned to the extent.
    Unlike :func:`split_by_size`, this operation never emits a short trailing
    tile. ``min_overlap`` is a target of at most half ``size``. The tile count
    grows until the target is met, capped so three tiles never cover one cell.
    The realized overlap therefore follows from the extent and can differ from
    the target in either direction: below it when the cap is hit, and up to
    ``size - 1`` when two tiles span an extent barely longer than one tile
    (33 cells, ``size=32``: overlap 31).
    This layout is intended for models trained on a fixed spatial window, where
    changing the final tile's shape changes the model's effective content scale.
    """
    if size <= 0:
        raise ValueError(f"size must be > 0, got {size}")
    if min_overlap < 0 or min_overlap * 2 > size:
        raise ValueError(f"min_overlap must be at most half the tile size, got min_overlap={min_overlap}, size={size}")

    def split(dimension_size: int) -> DimensionIntervals:
        if dimension_size <= size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)

        # Cap so a tile and the one two steps away cannot overlap.
        limit = max(2, 1 + (dimension_size - size) // ((size + 1) // 2))
        tiles_count = 2
        while tiles_count < limit and (tiles_count * size - dimension_size) / (tiles_count - 1) < min_overlap:
            tiles_count += 1
        base, remainder = divmod(dimension_size - size, tiles_count - 1)
        origins = [0]
        for index in range(tiles_count - 1):
            origins.append(origins[-1] + base + (1 if index < remainder else 0))

        intervals: list[DimensionInterval] = []
        for index, start in enumerate(origins):
            left_overlap = 0 if index == 0 else origins[index - 1] + size - start
            right_overlap = 0 if index == tiles_count - 1 else start + size - origins[index + 1]
            intervals.append(
                DimensionInterval(
                    start=start,
                    end=start + size,
                    left_ramp=left_overlap,
                    right_ramp=right_overlap,
                )
            )
        _validate_tile_intervals(intervals, dim_size=dimension_size, min_tile_size=size)
        return DimensionIntervals(intervals=intervals)

    return split


def split_temporal_causal(size: int, overlap: int, min_tile_size: int | None = None) -> SplitOperation:
    """Split a temporal axis into overlapping tiles with causal handling.
    Each tile after the first is shifted back by 1 and its left ramp is
    increased by 1, ensuring causal continuity through the blend ramps.
    Args:
        size: Tile size in axis units.
        overlap: Overlap between tiles in the same units.
        min_tile_size: Optional floor forwarded to :func:`split_by_size`.
    Returns:
        Split operation that divides temporal dimension with causal handling.
    """
    non_causal_split = split_by_size(size, overlap, min_tile_size=min_tile_size)

    def split(dimension_size: int) -> DimensionIntervals:
        if dimension_size <= size:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        dim_intervals = non_causal_split(dimension_size)
        if len(dim_intervals.intervals) <= 1:
            return dim_intervals
        modified_intervals = [dim_intervals.intervals[0]] + [
            replace(interval, start=interval.start - 1, left_ramp=interval.left_ramp + 1)
            for interval in dim_intervals.intervals[1:]
        ]
        return DimensionIntervals(intervals=modified_intervals)

    return split


def split_temporal(tile_size_frames: int, overlap_frames: int) -> SplitOperation:
    """Split a temporal axis in video frame space into overlapping tiles.
    Args:
        tile_size_frames: Tile length in frames.
        overlap_frames: Overlap between consecutive tiles in frames.
    Returns:
        Split operation that takes frame count and returns DimensionIntervals in frame indices.
    """
    non_causal_split = split_by_size(tile_size_frames, overlap_frames)

    def split(dimension_size: int) -> DimensionIntervals:
        if dimension_size <= tile_size_frames:
            return DEFAULT_SPLIT_OPERATION(dimension_size)
        dim_intervals = non_causal_split(dimension_size)
        modified_intervals = [
            replace(interval, end=interval.end + 1, right_ramp=0) for interval in dim_intervals.intervals[:-1]
        ] + [replace(dim_intervals.intervals[-1], right_ramp=0)]
        return DimensionIntervals(intervals=modified_intervals)

    return split


def split_by_count_temporal_causal(
    num_tiles: int, overlap: int = 0, min_tile_size: int | None = None
) -> SplitOperation:
    """Split a temporal dimension by count with causal handling.
    Wraps :func:`split_by_count` with the same causal adjustment as
    :func:`split_temporal_causal`: each tile after the first is shifted
    back by 1 and its left ramp is increased by 1.
    Args:
        num_tiles: Number of tiles. Must be >= 1.
        overlap: Overlap between adjacent tiles (default 0).
        min_tile_size: Optional floor forwarded to :func:`split_by_count`.
    Returns:
        A split operation that divides a temporal dimension into tiles.
    """
    non_causal_split = split_by_count(num_tiles, overlap, min_tile_size=min_tile_size)

    def split(dimension_size: int) -> DimensionIntervals:
        dim_intervals = non_causal_split(dimension_size)
        if len(dim_intervals.intervals) <= 1:
            return dim_intervals
        modified_intervals = [dim_intervals.intervals[0]] + [
            replace(interval, start=interval.start - 1, left_ramp=interval.left_ramp + 1)
            for interval in dim_intervals.intervals[1:]
        ]
        return DimensionIntervals(intervals=modified_intervals)

    return split


def split_at_seams(boundaries: Sequence[int], num_tiles: int, overlap: int = 0) -> SplitOperation:
    """Split a dimension on boundary cells whose content is already known, dropping the overlap.
    ``boundaries`` are the ``K + 1`` segment edges in grid cells, starting at 0 and ending at the
    last cell of the dimension. The ``K`` segments are dealt largest-first so leftover segments go to the leading tiles;
    ``num_tiles`` larger than ``K`` is clamped. Each tile but the first starts ``overlap`` cells
    before the boundary it resumes after. That run-up is context only: it lands in the interval's
    ``left_ramp``, which :func:`identity_mapping_operation` with ``rectangular=True`` masks to zero,
    so the earlier tile keeps the boundary cell and this one contributes strictly after it.
    The point of cutting here is that nothing needs blending. A ramp is what a pair of tiles needs
    when neither of them knows the truth at the seam; on a boundary cell both reproduce the same
    known frame, so averaging them only smears it.
    Args:
        boundaries: Segment edges in grid cells, strictly increasing, starting at 0.
        num_tiles: Number of tiles. Must be >= 1. Extra tiles beyond the segment count are dropped.
        overlap: Context cells each non-first tile denoises before the cell it resumes at, in grid
            units. Clamped at the start of the dimension.
    Returns:
        A split operation that divides a dimension on ``boundaries``.
    """
    boundaries = tuple(boundaries)
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be >= 1, got {num_tiles}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if len(boundaries) < 2 or boundaries[0] != 0:
        raise ValueError(f"boundaries must start at 0 and hold at least one segment, got {list(boundaries)}")
    if any(b <= a for a, b in itertools.pairwise(boundaries)):
        raise ValueError(f"boundaries must be strictly increasing, got {list(boundaries)}")
    n_segments = len(boundaries) - 1
    n_tiles = min(num_tiles, n_segments)
    base, leftover = divmod(n_segments, n_tiles)
    counts = [base + (1 if index < leftover else 0) for index in range(n_tiles)]

    def split(dim_size: int) -> DimensionIntervals:
        if boundaries[-1] != dim_size - 1:
            raise ValueError(f"boundaries must end at the last cell ({dim_size - 1}), got {boundaries[-1]}")
        intervals: list[DimensionInterval] = []
        cursor = 0
        for tile_index, count in enumerate(counts):
            resume = boundaries[cursor] + 1
            start = 0 if tile_index == 0 else max(0, resume - overlap)
            cursor += count
            intervals.append(
                DimensionInterval(
                    start=start,
                    end=boundaries[cursor] + 1,
                    left_ramp=0 if tile_index == 0 else resume - start,
                    right_ramp=0,
                )
            )
        return DimensionIntervals(intervals=intervals)

    return split


def split_by_count(num_tiles: int, overlap: int = 0, min_tile_size: int | None = None) -> SplitOperation:
    """Split a dimension into a given number of tiles with overlap.
    Computes the tile size as
    ``(dim_size + overlap * (num_tiles - 1)) // num_tiles`` so that
    ``num_tiles`` tiles of that size with ``overlap`` shared elements
    cover the dimension evenly.  Delegates to :func:`split_by_size` for
    the actual interval construction.
    When the total ``dim_size + overlap * (num_tiles - 1)`` is not evenly
    divisible by ``num_tiles``, the first ``remainder`` tiles each absorb
    one extra unit.
    Args:
        num_tiles: Number of tiles. Must be >= 1.
        overlap: Overlap between adjacent tiles (default 0). Must be >= 0
            and less than the computed tile size.
        min_tile_size: Optional floor forwarded to last-tile growth / validation.
    Returns:
        A split operation that divides a dimension into tiles.
    """
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be >= 1, got {num_tiles}")
    if overlap < 0:
        raise ValueError(f"overlap must be >= 0, got {overlap}")
    if min_tile_size is not None and min_tile_size < 1:
        raise ValueError(f"min_tile_size must be >= 1, got {min_tile_size}")

    def split(dim_size: int) -> DimensionIntervals:
        if num_tiles > dim_size:
            raise ValueError(
                f"num_tiles ({num_tiles}) exceeds dim_size ({dim_size}). Cannot assign at least 1 unit per tile."
            )
        if num_tiles == 1:
            return DEFAULT_SPLIT_OPERATION(dim_size)

        total = dim_size + overlap * (num_tiles - 1)
        tile_size = total // num_tiles
        if tile_size <= overlap:
            raise ValueError(
                f"split_by_count produced size={tile_size} <= overlap={overlap} "
                f"for dim_size={dim_size}, num_tiles={num_tiles}"
            )
        remainder = total % num_tiles

        base_intervals = split_by_size(tile_size, overlap)(dim_size - remainder).intervals

        # First `remainder` tiles each absorb 1 extra unit; shift subsequent boundaries.
        intervals: list[DimensionInterval] = []
        for i, iv in enumerate(base_intervals):
            shift = min(i, remainder)
            grow = 1 if i < remainder else 0
            intervals.append(replace(iv, start=iv.start + shift, end=iv.end + shift + grow))

        if min_tile_size is not None:
            intervals = _grow_last_tile_to_min(intervals, min_tile_size)
            _validate_tile_intervals(intervals, dim_size=dim_size, min_tile_size=min_tile_size)

        return DimensionIntervals(intervals=intervals)

    return split


# ---------------------------------------------------------------------------
# Mapping operations
# ---------------------------------------------------------------------------


def identity_mapping_operation(
    intervals: DimensionIntervals,
    *,
    rectangular: bool = False,
) -> tuple[list[slice], list[torch.Tensor]]:
    """Map each DimensionInterval to an output region at the same position.
    For every interval the output start/end matches the input start/end and a 1-D mask is built
    from the interval's left_ramp and right_ramp. The default mask is trapezoidal (blend on the
    ramps). ``rectangular=True`` drops the ramps outright: the overlap is context the tile denoised
    but does not contribute. Pair that with a split whose ramps are one-sided, such as
    :func:`split_at_seams` -- ramps on both sides of an interval would leave a hole between tiles.
    """
    mask_1d = compute_rectangular_mask_1d if rectangular else compute_trapezoidal_mask_1d
    out_slices: list[slice] = []
    masks: list[torch.Tensor] = []
    for iv in intervals.intervals:
        out_slices.append(slice(iv.start, iv.end))
        masks.append(mask_1d(iv.end - iv.start, iv.left_ramp, iv.right_ramp))
    return out_slices, masks


class Tile(NamedTuple):
    """
    Represents a single tile.
    Attributes:
        in_coords:
            Tuple of slices specifying where to cut the tile from the INPUT tensor.
        out_coords:
            Tuple of slices specifying where this tile's OUTPUT should be placed in the reconstructed OUTPUT tensor.
        masks_1d:
            Per-dimension masks in OUTPUT units.
            Untiled axes use a length-1 ones tensor (broadcasts). These are used
            for separable blending (and for the dense ``blend_mask`` property).
    Methods:
        blend_mask:
            Create a single N-D mask from the per-dimension masks.
    """

    in_coords: tuple[slice, ...]
    out_coords: tuple[slice, ...]
    masks_1d: tuple[torch.Tensor, ...]

    @property
    def blend_mask(self) -> torch.Tensor:
        num_dims = len(self.out_coords)
        per_dimension_masks: list[torch.Tensor] = []

        for dim_idx in range(num_dims):
            mask_1d = self.masks_1d[dim_idx]
            view_shape = [1] * num_dims
            # Reshape (L,) -> (1, ..., L, ..., 1) so masks across dimensions broadcast-multiply.
            view_shape[dim_idx] = mask_1d.shape[0]
            per_dimension_masks.append(mask_1d.view(*view_shape))

        # Multiply per-dimension masks to form the full N-D mask (separable blending window).
        combined_mask = per_dimension_masks[0]
        for mask in per_dimension_masks[1:]:
            combined_mask = combined_mask * mask

        return combined_mask


def scale_by_masks_1d(x: torch.Tensor, masks_1d: Sequence[torch.Tensor]) -> torch.Tensor:
    """Multiply ``x`` by separable 1d masks with broadcasting.
    ``len(masks_1d)`` must equal ``x.ndim``. Prefer float32 masks so bf16/fp16 ``x`` promotes.
    Length-1 masks (untiled axes) broadcast over that dimension.
    """
    if len(masks_1d) != x.ndim:
        raise ValueError(f"masks_1d length {len(masks_1d)} != x.ndim {x.ndim}")
    out = x
    for axis, mask in enumerate(masks_1d):
        view_shape = [1] * x.ndim
        view_shape[axis] = -1
        out = out * mask.reshape(*view_shape)
    return out


def masks_are_complementary(
    tiles: Sequence[Tile],
    full_shape: Sequence[int],
    *,
    atol: float = 1e-5,
) -> bool:
    """Return whether per-axis 1d blend masks partition unity (sum to 1).
    Checks each axis independently over the unique out-slices on that axis
    (cartesian tile products would otherwise multi-count the same 1d interval).
    When True, weighted accumulation needs no denominator.
    """
    if not tiles:
        return True
    ndim = len(full_shape)
    for tile in tiles:
        if len(tile.out_coords) != ndim or len(tile.masks_1d) != ndim:
            raise ValueError(
                f"Tile out_coords/masks_1d rank {len(tile.out_coords)}/{len(tile.masks_1d)} != full_shape rank {ndim}"
            )
    for axis, length in enumerate(full_shape):
        # Explicit CPU float32: masks may live on CUDA; a non-CPU default device
        # must not place ``acc`` on GPU (device-mismatch on ``acc[sl] +=``).
        acc = torch.zeros(length, dtype=torch.float32, device="cpu")
        seen: set[tuple[int | None, int | None]] = set()
        for tile in tiles:
            sl = tile.out_coords[axis]
            key = (sl.start, sl.stop)
            if key in seen:
                continue
            seen.add(key)
            # Length-1 untiled masks broadcast over ``acc[sl]``.
            acc[sl] += tile.masks_1d[axis].detach().float().cpu()
        if not torch.allclose(acc, torch.ones(length, dtype=torch.float32), atol=atol, rtol=0.0):
            return False
    return True


def compute_summed_weights(
    tiles: Sequence[Tile],
    full_shape: Sequence[int],
) -> torch.Tensor:
    """Build the dense denominator for weighted blending over ``full_shape``.
    Uses separable per-axis mask broadcasts — never ``Tile.blend_mask``.
    Requires concrete ``out_coords`` (``stop`` not ``None``) on every axis.
    Always builds on CPU float32 so CUDA masks / a non-CPU default device cannot
    place a multi-GB ``[F,H,W]`` tensor on GPU.
    """
    weights = torch.zeros(*full_shape, dtype=torch.float32, device="cpu")
    for tile in tiles:
        masks = tuple(m.detach().float().cpu() for m in tile.masks_1d)
        region_shape = tuple(s.stop - s.start for s in tile.out_coords)
        region = torch.ones(region_shape, dtype=torch.float32, device="cpu")
        weights[tile.out_coords] += scale_by_masks_1d(region, masks)
    return weights.clamp(min=1e-8)


def create_tiles_from_intervals_and_mappers(
    intervals: LatentIntervals,
    mappers: list[MappingOperation],
) -> list[Tile]:
    full_dim_input_slices: list[list[slice]] = []
    full_dim_output_slices: list[list[slice]] = []
    full_dim_masks_1d: list[list[torch.Tensor]] = []
    for axis_index in range(len(intervals.original_shape)):
        dimension_intervals = intervals.dimension_intervals[axis_index]
        input_slices = [slice(interval.start, interval.end) for interval in dimension_intervals.intervals]
        output_slices, masks_1d = mappers[axis_index](dimension_intervals)
        n_intervals = len(input_slices)
        if len(output_slices) != n_intervals or len(masks_1d) != n_intervals:
            raise ValueError(
                f"Axis {axis_index}: mapper produced {len(output_slices)} output slices and "
                f"{len(masks_1d)} masks for {n_intervals} input intervals"
            )
        full_dim_input_slices.append(input_slices)
        full_dim_output_slices.append(output_slices)
        full_dim_masks_1d.append(masks_1d)

    return [
        Tile(in_coords=in_coord, out_coords=out_coord, masks_1d=mask_1d)
        for in_coord, out_coord, mask_1d in zip(
            itertools.product(*full_dim_input_slices),
            itertools.product(*full_dim_output_slices),
            itertools.product(*full_dim_masks_1d),
            strict=True,
        )
    ]


def create_tiles(
    latent_shape: torch.Size,
    splitters: list[SplitOperation],
    mappers: list[MappingOperation],
) -> list[Tile]:
    if len(splitters) != len(latent_shape):
        raise ValueError(
            f"Number of splitters must be equal to number of dimensions in latent shape, "
            f"got {len(splitters)} and {len(latent_shape)}"
        )
    if len(mappers) != len(latent_shape):
        raise ValueError(
            f"Number of mappers must be equal to number of dimensions in latent shape, "
            f"got {len(mappers)} and {len(latent_shape)}"
        )
    intervals = [splitter(length) for splitter, length in zip(splitters, latent_shape, strict=True)]
    latent_intervals = LatentIntervals(original_shape=latent_shape, dimension_intervals=tuple(intervals))
    return create_tiles_from_intervals_and_mappers(latent_intervals, mappers)


def group_tiles_by_temporal_slice(tiles: list[Tile]) -> list[list[Tile]]:
    """Group consecutive tiles that share the same temporal ``out_coords`` slice.
    Assumes ``tiles`` is ordered with the temporal axis varying slowest (true
    for every tile list this codebase builds via ``itertools.product`` with
    the temporal axis first), so equal temporal slices are always contiguous.
    """
    if not tiles:
        return []

    groups = []
    current_slice = tiles[0].out_coords[2]
    current_group = []

    for tile in tiles:
        tile_slice = tile.out_coords[2]
        if tile_slice == current_slice:
            current_group.append(tile)
        else:
            groups.append(current_group)
            current_slice = tile_slice
            current_group = [tile]

    if current_group:
        groups.append(current_group)

    return groups


# ---------------------------------------------------------------------------
# Video-grid tiling configs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DimensionTilingConfig:
    """Tiling parameters for a single dimension of the patchified grid.
    Attributes:
        num_tiles: Number of tiles along this dimension. ``1`` with ``overlap=0``
            means the axis is not tiled.
        overlap: Overlap between adjacent tiles, in latent grid units.
            Adjacent tiles share ``overlap`` grid cells at their
            boundary, producing an overlap zone blended with
            trapezoidal masks.
    """

    num_tiles: int = 1
    overlap: int = 0

    def __post_init__(self) -> None:
        if self.num_tiles < 1:
            raise ValueError(f"num_tiles must be >= 1, got {self.num_tiles}")
        if self.overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {self.overlap}")

    def is_tiled(self) -> bool:
        """True when this axis is split into more than one tile (or has overlap)."""
        return self.num_tiles > 1 or self.overlap > 0

    @classmethod
    def from_tile_size(cls, dim_size: int, tile_size: int, overlap: int = 0) -> DimensionTilingConfig:
        """Create config by computing ``num_tiles`` from dimension size and tile size.
        Args:
            dim_size: Total length of the dimension.
            tile_size: Desired tile size.
            overlap: Overlap between consecutive tiles.
        Returns:
            A ``DimensionTilingConfig`` with the computed ``num_tiles``.
        """
        split_op = split_by_size(tile_size, overlap)
        intervals = split_op(dim_size)
        return cls(num_tiles=len(intervals.intervals), overlap=overlap)


@dataclass(frozen=True)
class DimensionSizeConfig:
    """Tile size and overlap for a single video axis (frames / height / width).
    Mirrors :class:`DimensionTilingConfig`, but specifies tile *size* rather than
    tile *count*. ``tile_size=0`` means the axis is not tiled (covers the whole
    length).     Axis-specific VAE pixel constraints (divisibility / minimums) are
    enforced by :meth:`TileSizeConfig.validate` for tiled axes only.
    """

    tile_size: int = 0
    overlap: int = 0

    def __post_init__(self) -> None:
        if self.tile_size < 0:
            raise ValueError(f"tile_size must be >= 0, got {self.tile_size}")
        if self.overlap < 0:
            raise ValueError(f"overlap must be >= 0, got {self.overlap}")
        if self.tile_size == 0:
            if self.overlap != 0:
                raise ValueError("untiled axis (tile_size=0) must have overlap=0")
            return
        if self.overlap >= self.tile_size:
            raise ValueError(f"Overlap must be less than tile size, got {self.overlap} and {self.tile_size}")

    def is_tiled(self) -> bool:
        """True when this axis has a positive tile size (caller intends to split it)."""
        return self.tile_size > 0


@dataclass(frozen=True)
class TileCountConfig:
    """Tiling layout for a ``(F, H, W)`` grid by tile *counts*.
    Overlaps are in latent-grid units. Mirror of :class:`TileSizeConfig`.
    Attributes:
        frames: Tiling along the temporal (frames) dimension.
        height: Tiling along the latent height dimension.
        width: Tiling along the latent width dimension.
    """

    frames: DimensionTilingConfig = DimensionTilingConfig()
    height: DimensionTilingConfig = DimensionTilingConfig()
    width: DimensionTilingConfig = DimensionTilingConfig()

    def validate(self, scale_factors: SpatioTemporalScaleFactors, video_shape: VideoPixelShape) -> None:
        """Raise if this count layout cannot tile ``video_shape`` under ``scale_factors``.
        Counts/overlaps are in latent-grid units. ``video_shape.frames <= 0`` skips the
        temporal axis (duration not yet known). Spatial axes always checked.
        """
        check_temporal = _assert_video_on_vae_grid(scale_factors, video_shape)
        latent_h = video_shape.height // scale_factors.height
        latent_w = video_shape.width // scale_factors.width
        _validate_count_axis(self.height, latent_h, "height")
        _validate_count_axis(self.width, latent_w, "width")
        if check_temporal:
            latent_f = (video_shape.frames - 1) // scale_factors.time + 1
            _validate_count_axis(self.frames, latent_f, "frames")

    def to_splitters(
        self,
        scale_factors: SpatioTemporalScaleFactors,
        min_tile_size: tuple[int, int, int] | None = None,
        *,
        causal_temporal: bool = True,
    ) -> tuple[SplitOperation, SplitOperation, SplitOperation]:
        """Build ``(T, H, W)`` latent-grid split operations for this count layout.
        ``scale_factors`` is accepted for signature parity with
        :meth:`TileSizeConfig.to_splitters` and ignored — counts are already in
        grid units. When ``causal_temporal`` is True (VAE encode/decode), the
        frames axis uses :func:`split_by_count_temporal_causal`; otherwise plain
        :func:`split_by_count`. ``min_tile_size`` is a per-axis floor in the same
        units as the split.
        """
        del scale_factors
        min_t = min_h = min_w = None
        if min_tile_size is not None:
            min_t, min_h, min_w = min_tile_size

        def axis_split(cfg: DimensionTilingConfig, axis_min: int | None, *, temporal: bool) -> SplitOperation:
            if not cfg.is_tiled():
                return DEFAULT_SPLIT_OPERATION
            if temporal and causal_temporal:
                return split_by_count_temporal_causal(cfg.num_tiles, cfg.overlap, min_tile_size=axis_min)
            return split_by_count(cfg.num_tiles, cfg.overlap, min_tile_size=axis_min)

        return (
            axis_split(self.frames, min_t, temporal=True),
            axis_split(self.height, min_h, temporal=False),
            axis_split(self.width, min_w, temporal=False),
        )

    def video_chunks_number(self, num_frames: int) -> int:
        """Number of temporal decode chunks for ``num_frames`` under this layout."""
        del num_frames
        return max(1, self.frames.num_tiles)


@dataclass(frozen=True)
class TileSizeConfig:
    """Size-based tiling layout for a ``(F, H, W)`` video — mirror of ``TileCountConfig``.
    Each axis is a non-optional :class:`DimensionSizeConfig`; ``tile_size=0`` means
    untiled on that axis (:meth:`DimensionSizeConfig.is_tiled`). Sizes and overlaps
    are in pixel / frame units. Conversion to a split grid is an explicit
    ``scale_factors`` argument to :meth:`to_splitters` (not stored on the config).
    Legality vs a VAE grid is checked by :meth:`validate` (same factors decode will
    pass to :meth:`to_splitters`), not at construction.
    Attributes:
        frames: Temporal tile size/overlap in frames.
        height: Spatial height tile size/overlap in pixels.
        width: Spatial width tile size/overlap in pixels.
    """

    frames: DimensionSizeConfig = DimensionSizeConfig()
    height: DimensionSizeConfig = DimensionSizeConfig()
    width: DimensionSizeConfig = DimensionSizeConfig()

    def validate(self, scale_factors: SpatioTemporalScaleFactors, video_shape: VideoPixelShape) -> None:
        """Raise if this size layout is illegal for ``video_shape`` under ``scale_factors``.
        Checks tile/overlap divisibility and minimums against the VAE grid, and that
        the video extents are compatible with that grid. ``video_shape.frames <= 0``
        skips the temporal axis (duration not yet known); height/width always checked.
        """
        check_temporal = _assert_video_on_vae_grid(scale_factors, video_shape)
        _validate_size_axis(self.height, scale_factors.height, "height")
        _validate_size_axis(self.width, scale_factors.width, "width")
        if check_temporal:
            _validate_size_axis(self.frames, scale_factors.time, "frames")

    @classmethod
    def default(cls) -> TileSizeConfig:
        return cls(
            frames=DimensionSizeConfig(tile_size=80, overlap=24),
            height=DimensionSizeConfig(tile_size=768, overlap=64),
            width=DimensionSizeConfig(tile_size=768, overlap=64),
        )

    @classmethod
    def from_long_side(
        cls,
        *,
        long_side: DimensionSizeConfig,
        height: int,
        width: int,
        scale_factors: SpatioTemporalScaleFactors,
        frames: DimensionSizeConfig | None = None,
    ) -> TileSizeConfig:
        """Aspect-coupled construction — old single-spatial long-side behavior, explicit.
        Matches main-era ``latent_tile_splitters``: scale the long-side tile in
        *latent* units with ``round(size_lat * axis_lat / long_lat)``, then
        multiply back by the VAE factor. Pixel-space ``round`` + ceil-snap would
        bias the short axis up by almost one latent (e.g. 680 → 704 vs 672).
        Both axes share ``long_side.overlap``.
        """
        if height < 1 or width < 1:
            raise ValueError(f"height/width must be >= 1, got {height}x{width}")
        if not long_side.is_tiled():
            raise ValueError("long_side must be tiled (tile_size > 0)")
        if scale_factors.height < 1 or scale_factors.width < 1:
            raise ValueError(f"scale_factors height/width must be >= 1, got {scale_factors}")
        span = max(height, width)

        def axis_size(axis_len: int, factor: int) -> int:
            # Latent-grid round (same as main decode enable_on_axis), not pixel ceil.
            axis_lat = axis_len // factor
            long_lat = span // factor
            size_lat = long_side.tile_size // factor
            overlap_lat = long_side.overlap // factor
            lower_threshold = max(2, overlap_lat + 1)
            tile_lat = max(lower_threshold, round(size_lat * axis_lat / long_lat))
            tile_px = tile_lat * factor
            min_legal = max(2 * factor, long_side.overlap + factor)
            return max(tile_px, min_legal)

        return cls(
            frames=DimensionSizeConfig() if frames is None else frames,
            height=DimensionSizeConfig(tile_size=axis_size(height, scale_factors.height), overlap=long_side.overlap),
            width=DimensionSizeConfig(tile_size=axis_size(width, scale_factors.width), overlap=long_side.overlap),
        )

    def to_splitters(
        self,
        scale_factors: SpatioTemporalScaleFactors,
        min_tile_size: tuple[int, int, int] | None = None,
        *,
        causal_temporal: bool = True,
    ) -> tuple[SplitOperation, SplitOperation, SplitOperation]:
        """Build ``(T, H, W)`` grid split ops from pixel/frame sizes via ``scale_factors``.
        When ``causal_temporal`` is True, frames use :func:`split_temporal_causal`.
        """
        min_t = min_h = min_w = None
        if min_tile_size is not None:
            min_t, min_h, min_w = min_tile_size

        def enable_size_axis(
            factor: int,
            axis_min: int | None,
            cfg: DimensionSizeConfig,
            axis_name: str,
            *,
            temporal: bool,
        ) -> SplitOperation:
            if not cfg.is_tiled():
                return DEFAULT_SPLIT_OPERATION
            _validate_size_axis(cfg, factor, axis_name)
            size = cfg.tile_size // factor
            overlap = cfg.overlap // factor
            lower_threshold = max(2, overlap + 1)
            tile = max(lower_threshold, size)
            if temporal and causal_temporal:
                return split_temporal_causal(tile, overlap, min_tile_size=axis_min)
            return split_by_size(tile, overlap, min_tile_size=axis_min)

        return (
            enable_size_axis(scale_factors.time, min_t, self.frames, "frames", temporal=True),
            enable_size_axis(scale_factors.height, min_h, self.height, "height", temporal=False),
            enable_size_axis(scale_factors.width, min_w, self.width, "width", temporal=False),
        )

    def video_chunks_number(self, num_frames: int, *, time_scale: int = VIDEO_SCALE_FACTORS.time) -> int:
        """Number of temporal decode chunks for ``num_frames`` under this layout.
        Mirrors what decode actually does: :meth:`to_splitters` converts this axis to the
        latent grid and hands it to :func:`split_by_size`, so the count must be taken there
        too. Doing the arithmetic in pixel units instead over-reports by one whenever the
        trailing tile is absorbed -- including the common case of a tile larger than the
        clip, which is a single tile but used to report two.
        """
        if not self.frames.is_tiled():
            return 1
        # Same derivation as ``to_splitters.enable_size_axis``.
        overlap = self.frames.overlap // time_scale
        size = max(2, overlap + 1, self.frames.tile_size // time_scale)
        latent_frames = (num_frames - 1) // time_scale + 1
        if latent_frames <= size:
            return 1
        # Same tile count as ``split_by_size``.
        return (latent_frames + size - 2 * overlap - 1) // (size - overlap)


# Either size-based (single-GPU VAE) or count-based (MGPU / explicit counts).
TilingConfig = TileSizeConfig | TileCountConfig


class AutoTiling:
    """Sentinel: pipeline should recommend decode tiling (DiffVAE-aware / Conv default).
    Distinct from ``None``, which means untiled decode.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "AUTO_TILING"


AUTO_TILING = AutoTiling()

# Pipeline ``tiling_config`` argument: explicit layout, auto-recommend, or untiled.
PipelineTiling = TilingConfig | AutoTiling | None


def _assert_video_on_vae_grid(
    scale_factors: SpatioTemporalScaleFactors,
    video_shape: VideoPixelShape,
) -> bool:
    """Raise if ``video_shape`` is incompatible with the VAE ``scale_factors`` grid.
    Returns whether the temporal axis is known (``video_shape.frames > 0``). When
    False, callers skip frames-axis checks (duration not yet resolved).
    """
    if scale_factors.time < 1 or scale_factors.height < 1 or scale_factors.width < 1:
        raise ValueError(f"scale_factors must be >= 1 on each axis, got {scale_factors}")
    if video_shape.height < 1 or video_shape.width < 1:
        raise ValueError(f"video_shape height/width must be >= 1, got {video_shape.height}x{video_shape.width}")
    if video_shape.height % scale_factors.height != 0:
        raise ValueError(f"video height {video_shape.height} must be divisible by scale {scale_factors.height}")
    if video_shape.width % scale_factors.width != 0:
        raise ValueError(f"video width {video_shape.width} must be divisible by scale {scale_factors.width}")
    if video_shape.frames <= 0:
        return False
    if (video_shape.frames - 1) % scale_factors.time != 0:
        raise ValueError(f"video frames {video_shape.frames} must satisfy (frames - 1) % {scale_factors.time} == 0")
    return True


def _validate_size_axis(cfg: DimensionSizeConfig, factor: int, axis_name: str) -> None:
    """Pixel/frame size-axis legality vs VAE ``factor``."""
    if not cfg.is_tiled():
        return
    min_size = 2 * factor
    if cfg.tile_size < min_size:
        raise ValueError(f"{axis_name}.tile_size must be at least {min_size}, got {cfg.tile_size}")
    if cfg.tile_size % factor != 0:
        raise ValueError(f"{axis_name}.tile_size must be divisible by {factor}, got {cfg.tile_size}")
    if cfg.overlap % factor != 0:
        raise ValueError(f"{axis_name}.overlap must be divisible by {factor}, got {cfg.overlap}")


def _validate_count_axis(cfg: DimensionTilingConfig, latent_extent: int, axis_name: str) -> None:
    """Latent count-axis legality vs latent ``extent``."""
    if not cfg.is_tiled():
        return
    if cfg.num_tiles > latent_extent:
        raise ValueError(f"{axis_name}.num_tiles {cfg.num_tiles} exceeds latent {axis_name} extent {latent_extent}")
    # split_by_count requires overlap < tile_size; tile_size grows with extent/n.
    max_overlap = latent_extent - cfg.num_tiles
    if cfg.overlap > max_overlap:
        raise ValueError(
            f"{axis_name}.overlap {cfg.overlap} exceeds latent bound {max_overlap} "
            f"for extent {latent_extent} with {cfg.num_tiles} tiles"
        )


def _validate_overlap(
    tiling_config: TilingConfig,
    *,
    min_overlap_frames: int,
    min_overlap_pixels: int,
) -> None:
    """Raise if any tiled ``TileSizeConfig`` axis overlap is below the given floors."""
    if not isinstance(tiling_config, TileSizeConfig):
        return

    for axis_name, cfg, recommended, unit in (
        ("frames", tiling_config.frames, min_overlap_frames, "frames"),
        ("height", tiling_config.height, min_overlap_pixels, "px"),
        ("width", tiling_config.width, min_overlap_pixels, "px"),
    ):
        if cfg.is_tiled() and cfg.overlap < recommended:
            raise ValueError(f"{axis_name} overlap {cfg.overlap} {unit} is below the required {recommended} {unit}.")


def balanced_tile_split(num_tiles: int) -> tuple[int, int]:
    """Factor ``num_tiles`` into ``(small, large)`` as square as possible.
    ``small`` is the largest divisor not exceeding the square root, so
    ``small * large == num_tiles`` and ``small <= large``. E.g. 2 -> (1, 2),
    4 -> (2, 2), 8 -> (2, 4), 16 -> (4, 4). The caller decides which tiled
    dimension gets which factor.
    """
    if num_tiles < 1:
        raise ValueError(f"num_tiles must be >= 1, got {num_tiles}")
    small = next(d for d in range(math.isqrt(num_tiles), 0, -1) if num_tiles % d == 0)
    return small, num_tiles // small
