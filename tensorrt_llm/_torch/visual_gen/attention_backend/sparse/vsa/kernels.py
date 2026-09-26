# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Memory-bound kernels of the Video Sparse Attention predictor and post-process.

Every helper streams ``[batch, tokens, heads, head_dim]`` activations row by row, where a row
is one token's ``heads * head_dim`` values. CUDA tensors run Triton kernels; other tensors use
PyTorch implementations with the same numerics. Launch shapes are derived from the row width
and the grid size, so no autotuning happens at call time and every launch is CUDA Graph safe.
"""

import torch
import triton
import triton.language as tl

_MAX_BLOCK = 1024
_MIN_ELEMENTS_PER_THREAD = 4
_SMALL_GRID_PROGRAMS = 2048
_MAX_SLOTS_PER_ITERATION = 4
_MAX_TRITON_SORT_LENGTH = 2048


def _row_launch_config(row_width: int, num_row_programs: int) -> tuple[int, int, int]:
    """Choose ``(block, num_chunks, num_warps)`` for a kernel that streams rows in chunks.

    A block covers up to 1024 columns; wider rows are split into chunks. Small grids get
    more warps per program to expose parallelism, large grids fewer warps so that every
    thread keeps several elements in flight.
    """
    block = min(_MAX_BLOCK, triton.next_power_of_2(row_width))
    num_chunks = triton.cdiv(row_width, block)
    num_warps = 4 if num_row_programs * num_chunks < _SMALL_GRID_PROGRAMS else 2
    num_warps = min(num_warps, max(1, block // (32 * _MIN_ELEMENTS_PER_THREAD)))
    return block, num_chunks, num_warps


def _slots_per_iteration(cube_size: int) -> int:
    """Largest power of two up to four that divides the cube, so slot tiles stay rectangular."""
    slots = _MAX_SLOTS_PER_ITERATION
    while cube_size % slots:
        slots //= 2
    return slots


@triton.jit
def _tile_and_pool_cubes_kernel(
    x_ptr,
    source_ptr,
    count_ptr,
    tiled_ptr,
    pooled_ptr,
    num_cubes,
    stride_batch,
    stride_token,
    ROW: tl.constexpr,
    CUBE: tl.constexpr,
    SLOTS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One program per (batch, cube, column chunk).

    The cube's CUBE slots are streamed SLOTS at a time: each slot is copied from its compact
    source token (zero for a pad slot) into the tiled layout while an fp32 accumulator builds
    the cube mean.
    """
    batch_cube = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    batch = batch_cube // num_cubes
    cube = batch_cube % num_cubes
    columns = chunk * BLOCK + tl.arange(0, BLOCK)
    in_row = columns < ROW
    slot_offsets = tl.arange(0, SLOTS)

    x_batch_ptr = x_ptr + batch * stride_batch
    tiled_cube_ptr = tiled_ptr + batch_cube * CUBE * ROW
    total = tl.zeros([BLOCK], dtype=tl.float32)
    for first_slot in range(0, CUBE, SLOTS):
        slots = first_slot + slot_offsets
        sources = tl.load(source_ptr + cube * CUBE + slots)
        values = tl.load(
            x_batch_ptr + sources[:, None] * stride_token + columns[None, :],
            mask=(sources >= 0)[:, None] & in_row[None, :],
            other=0.0,
        )
        tl.store(
            tiled_cube_ptr + slots[:, None] * ROW + columns[None, :],
            values,
            mask=(slots >= 0)[:, None] & in_row[None, :],
        )
        total += tl.sum(values.to(tl.float32), axis=0)

    count = tl.load(count_ptr + cube).to(tl.float32)
    mean = (total / count).to(pooled_ptr.dtype.element_ty)
    tl.store(pooled_ptr + batch_cube * ROW + columns, mean, mask=in_row)


def _tile_and_pool_cubes_torch(
    x: torch.Tensor,
    tile_source_index: torch.Tensor,
    cube_valid_counts: torch.Tensor,
    *,
    cube_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, _, num_heads, head_dim = x.shape
    num_cubes = cube_valid_counts.shape[0]
    tiled = x.index_select(1, tile_source_index.clamp(min=0))
    is_valid_slot = (tile_source_index >= 0).view(1, -1, 1, 1)
    tiled = torch.where(is_valid_slot, tiled, torch.zeros((), dtype=x.dtype, device=x.device))
    total = tiled.view(batch_size, num_cubes, cube_size, num_heads, head_dim).sum(
        dim=2, dtype=torch.float32
    )
    mean = total / cube_valid_counts.view(1, -1, 1, 1).to(torch.float32)
    return tiled, mean.to(x.dtype)


def tile_and_pool_cubes(
    x: torch.Tensor,
    tile_source_index: torch.Tensor,
    cube_valid_counts: torch.Tensor,
    *,
    cube_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather tokens into the tile-major padded layout and mean-pool every cube.

    Args:
        x: ``[batch, seq_len, heads, head_dim]`` activations; the heads and head_dim
            dimensions must be contiguous, batch and sequence strides are arbitrary.
        tile_source_index: ``[num_cubes * cube_size]`` long tensor with the compact token of
            every padded slot, or ``-1`` for a pad slot.
        cube_valid_counts: ``[num_cubes]`` number of valid tokens per cube (at least one).
        cube_size: Tokens per cube.

    Returns:
        ``tiled`` ``[batch, num_cubes * cube_size, heads, head_dim]`` with zeroed pad slots and
        ``pooled`` ``[batch, num_cubes, heads, head_dim]`` cube means in ``x.dtype`` computed
        with an fp32 accumulator.
    """
    batch_size, _, num_heads, head_dim = x.shape
    if x.stride(3) != 1 or (num_heads > 1 and x.stride(2) != head_dim):
        raise ValueError("heads and head_dim must be contiguous in the VSA input")
    num_cubes = cube_valid_counts.shape[0]
    if tile_source_index.numel() != num_cubes * cube_size:
        raise ValueError(
            f"tile_source_index must have {num_cubes * cube_size} slots, "
            f"got {tile_source_index.numel()}"
        )
    if not x.is_cuda:
        return _tile_and_pool_cubes_torch(
            x, tile_source_index, cube_valid_counts, cube_size=cube_size
        )

    row_width = num_heads * head_dim
    tiled = torch.empty(
        (batch_size, num_cubes * cube_size, num_heads, head_dim), dtype=x.dtype, device=x.device
    )
    pooled = torch.empty(
        (batch_size, num_cubes, num_heads, head_dim), dtype=x.dtype, device=x.device
    )
    block, num_chunks, num_warps = _row_launch_config(row_width, batch_size * num_cubes)
    _tile_and_pool_cubes_kernel[(batch_size * num_cubes, num_chunks)](
        x,
        tile_source_index,
        cube_valid_counts,
        tiled,
        pooled,
        num_cubes,
        x.stride(0),
        x.stride(1),
        ROW=row_width,
        CUBE=cube_size,
        SLOTS=_slots_per_iteration(cube_size),
        BLOCK=block,
        num_warps=num_warps,
    )
    return tiled, pooled


@triton.jit
def _sort_rows_kernel(values_ptr, sorted_ptr, ROW: tl.constexpr, BLOCK: tl.constexpr):
    """One program per row; pad slots sort to the end and are never stored."""
    row = tl.program_id(0).to(tl.int64)
    columns = tl.arange(0, BLOCK)
    in_row = columns < ROW
    values = tl.load(values_ptr + row * ROW + columns, mask=in_row, other=2147483647)
    tl.store(sorted_ptr + row * ROW + columns, tl.sort(values), mask=in_row)


def _sort_last_dim_torch(values: torch.Tensor) -> torch.Tensor:
    return torch.sort(values, dim=-1).values


def sort_last_dim(values: torch.Tensor) -> torch.Tensor:
    """Sort an int32 tensor ascending along its last dimension.

    Rows up to 2048 entries are sorted by one Triton program each; longer rows and non-CUDA
    tensors use ``torch.sort``.
    """
    if values.dtype != torch.int32:
        raise TypeError(f"sort_last_dim expects int32 values, got {values.dtype}")
    row_length = values.shape[-1]
    if not values.is_cuda or row_length > _MAX_TRITON_SORT_LENGTH or values.numel() == 0:
        return _sort_last_dim_torch(values)

    rows = values.reshape(-1, row_length).contiguous()
    sorted_rows = torch.empty_like(rows)
    _sort_rows_kernel[(rows.shape[0],)](
        rows,
        sorted_rows,
        ROW=row_length,
        BLOCK=triton.next_power_of_2(row_length),
        num_warps=1,
    )
    return sorted_rows.view(values.shape)


@triton.jit
def _blend_coarse_fine_kernel(
    fine_ptr,
    coarse_ptr,
    gate_compress_ptr,
    gate_fine_ptr,
    untile_ptr,
    out_ptr,
    seq_len,
    num_cubes,
    stride_fine_batch,
    stride_fine_token,
    stride_fine_head,
    ROW: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    CUBE: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_GATE_FINE: tl.constexpr,
    FINE_IS_TILED: tl.constexpr,
):
    """One program per (batch, compact token, column chunk).

    The fine output is addressed through explicit strides so that head-major storage
    (``[batch, heads, tokens, head_dim]``) is consumed without a copy. Products and the final
    sum are rounded to the output dtype after each operation, which matches the PyTorch
    expression ``gate_compress * coarse + gate_fine * fine``.
    """
    batch_token = tl.program_id(0).to(tl.int64)
    chunk = tl.program_id(1)
    batch = batch_token // seq_len
    token = batch_token % seq_len
    slot = tl.load(untile_ptr + token)
    cube = slot // CUBE
    if FINE_IS_TILED:
        fine_row = slot
    else:
        fine_row = token
    columns = chunk * BLOCK + tl.arange(0, BLOCK)
    in_row = columns < ROW
    head = (columns // HEAD_DIM).to(tl.int64)
    dim = columns % HEAD_DIM
    out_dtype = out_ptr.dtype.element_ty

    fine_ptrs = (
        fine_ptr
        + batch * stride_fine_batch
        + fine_row * stride_fine_token
        + head * stride_fine_head
        + dim
    )
    fine = tl.load(fine_ptrs, mask=in_row, other=0.0)
    coarse = tl.load(
        coarse_ptr + (batch * num_cubes + cube) * ROW + columns, mask=in_row, other=0.0
    )
    gate_compress = tl.load(gate_compress_ptr + batch_token * ROW + columns, mask=in_row, other=0.0)
    coarse_term = (gate_compress.to(tl.float32) * coarse.to(tl.float32)).to(out_dtype)
    if HAS_GATE_FINE:
        gate_fine = tl.load(gate_fine_ptr + batch_token * ROW + columns, mask=in_row, other=0.0)
        fine_term = (gate_fine.to(tl.float32) * fine.to(tl.float32)).to(out_dtype)
    else:
        fine_term = fine.to(out_dtype)
    result = (coarse_term.to(tl.float32) + fine_term.to(tl.float32)).to(out_dtype)
    tl.store(out_ptr + batch_token * ROW + columns, result, mask=in_row)


def _blend_coarse_fine_torch(
    fine: torch.Tensor,
    coarse: torch.Tensor,
    gate_compress: torch.Tensor,
    gate_fine: torch.Tensor | None,
    untile_index: torch.Tensor,
    *,
    cube_size: int,
    fine_is_tiled: bool,
) -> torch.Tensor:
    coarse_per_token = coarse.index_select(1, untile_index // cube_size)
    fine_compact = fine.index_select(1, untile_index) if fine_is_tiled else fine
    if gate_fine is not None:
        fine_compact = gate_fine * fine_compact
    return gate_compress * coarse_per_token + fine_compact


def blend_coarse_fine(
    fine: torch.Tensor,
    coarse: torch.Tensor,
    gate_compress: torch.Tensor,
    gate_fine: torch.Tensor | None,
    untile_index: torch.Tensor,
    *,
    cube_size: int,
    fine_is_tiled: bool,
) -> torch.Tensor:
    """Restore compact token order and blend the coarse and fine VSA outputs.

    Computes ``gate_compress * coarse[cube(t)] + gate_fine * fine[src(t)]`` for every compact
    token ``t``, where ``cube(t)`` is the cube holding the token and ``src(t)`` is its padded
    slot when the fine output is tiled, or ``t`` itself otherwise.

    Args:
        fine: ``[batch, padded_len or seq_len, heads, head_dim]`` fine-stage output; only
            head_dim has to be contiguous, so head-major kernel outputs are accepted as views.
        coarse: ``[batch, num_cubes, heads, head_dim]`` coarse-stage output per cube.
        gate_compress: ``[batch, seq_len, heads, head_dim]`` gate for the coarse term.
        gate_fine: Optional gate for the fine term with the same shape as ``gate_compress``.
        untile_index: ``[seq_len]`` long tensor with the padded slot of every compact token.
        cube_size: Tokens per cube.
        fine_is_tiled: Whether ``fine`` is in the padded tile-major layout.

    Returns:
        ``[batch, seq_len, heads, head_dim]`` blended output in the gate dtype.
    """
    if not gate_compress.is_cuda:
        return _blend_coarse_fine_torch(
            fine,
            coarse,
            gate_compress,
            gate_fine,
            untile_index,
            cube_size=cube_size,
            fine_is_tiled=fine_is_tiled,
        )
    if fine.stride(3) != 1:
        raise ValueError("head_dim must be contiguous in the VSA fine output")

    coarse = coarse.contiguous()
    gate_compress = gate_compress.contiguous()
    batch_size, seq_len, num_heads, head_dim = gate_compress.shape
    row_width = num_heads * head_dim
    out = torch.empty_like(gate_compress)
    block, num_chunks, num_warps = _row_launch_config(row_width, batch_size * seq_len)
    _blend_coarse_fine_kernel[(batch_size * seq_len, num_chunks)](
        fine,
        coarse,
        gate_compress,
        gate_compress if gate_fine is None else gate_fine.contiguous(),
        untile_index,
        out,
        seq_len,
        coarse.shape[1],
        fine.stride(0),
        fine.stride(1),
        fine.stride(2),
        ROW=row_width,
        HEAD_DIM=head_dim,
        CUBE=cube_size,
        BLOCK=block,
        HAS_GATE_FINE=gate_fine is not None,
        FINE_IS_TILED=fine_is_tiled,
        num_warps=num_warps,
    )
    return out


__all__ = [
    "blend_coarse_fine",
    "sort_last_dim",
    "tile_and_pool_cubes",
]
