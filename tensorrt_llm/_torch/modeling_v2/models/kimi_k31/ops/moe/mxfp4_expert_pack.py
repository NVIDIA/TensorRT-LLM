# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3.1 routed-expert weights packed for the TRTLLM-Gen MXFP4 x MXFP8 runner.

The checkpoint stores every routed expert as group-32 packed MXFP4 (two e2m1 values per uint8, one
UE8M0 uint8 scale per 32 values along K): ``w1`` / ``w3`` ``[inter, hidden / 2]`` with scales
``[inter, hidden / 32]``, ``w2`` ``[hidden, inter / 2]`` with scales ``[hidden, inter / 32]``.

The runner reads four tensors per rank, one slot per local expert (:func:`expert_weight_shapes`):

* ``w3_w1_weight`` ``[E, 2 * inter_pad, hidden_pad / 2]``: this rank's ``w3`` rows above its ``w1``
  rows, reordered for the gated-activation GEMM and shuffled into 128-row epilogue tiles;
* ``w2_weight`` ``[E, hidden, inter_pad / 2]``: this rank's ``w2`` columns, rows shuffled;
* ``w3_w1_weight_scale`` / ``w2_weight_scale``: the scales permuted like their weights, then
  interleaved into the runner's 128x4 block-scale order.

``inter_pad`` is the per-rank intermediate size rounded up to the runner's 128-value tile. The
runner is told how many of those values are real (:func:`intermediate_size_lean`).

:func:`load_packed_mxfp4_expert` fills one slot from one checkpoint expert: each rank's range of the
intermediate dimension (``inter / moe_tp`` values, whole scale groups; routing refuses other splits) is sliced out
and zero-padded locally to the slot's shape (K3.1's ``inter`` 3072 pads nothing up to MoE TP 8 and pads each rank's
192 / 96 values to the 256 / 128-value tile at MoE TP 16 / 32).

The slots hold the same bytes as the TRTLLM-Gen MXFP4 quant method's packing, through the same two kernels
(:data:`k31_shuffle_matrix`, :data:`k31_block_scale_interleave`) and the same permute indices; the entry's test checks
every destination tensor bit for bit against it at MoE TP 1 to 32.

Thread safety: targets load experts from a thread pool. A slot is claimed under one module-level
lock in the caller's ``loaded_slots`` set before its tensors are written, so a slot loaded twice
fails instead of being transformed twice; the caller compares the set with every slot afterwards.
The permute-index cache is shared by all threads; a race on its first fill computes the same
indices twice.
"""

from __future__ import annotations

import threading
from typing import Dict, NamedTuple, Optional, Set, Tuple

import torch
import torch.nn.functional as F

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.modeling_v2.catalog._op import Arch, Cell, OpWrapper
from tensorrt_llm.quantization.utils.fp4_utils import (
    float4_sf_dtype,
    get_reorder_rows_for_gated_act_gemm_row_indices,
    get_shuffle_matrix_a_row_indices,
    get_shuffle_matrix_sf_a_row_indices,
)

#: Packed weights and block scales are both uint8 (two e2m1 values / one UE8M0 exponent per byte).
EXPERT_WEIGHT_DTYPE = torch.uint8
#: Values one uint8 of packed weight holds.
_VALUES_PER_BYTE = 2
#: Values one block scale covers (MXFP4 group size).
SCALING_VECTOR_SIZE = 32
#: The runner's tile along the intermediate dimension, and the alignment of the FC2 output rows.
WEIGHT_ALIGNMENT = 128
#: Alignment of the FC1 K (hidden) dimension: a multiple of 512 admits every runner config.
INPUT_HIDDEN_ALIGNMENT = 512
#: Row tile of the runner's epilogue, which the weight and scale row shuffles follow.
EPILOGUE_TILE_M = 128

#: K3.1 routed-expert geometry: latent hidden 3584, intermediate 3072 (per rank: 3072 / MoE TP).
_K31_HIDDEN = 3584
_K31_INTERMEDIATE = 3072


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _shuffle_cells() -> Tuple[Cell, ...]:
    """One cell per tensor the loader shuffles, per K3.1 MoE-TP layout."""
    cells = []
    for moe_tp, why in (
        (1, "MoE TP 1 (whole experts): inter 3072"),
        (2, "MoE TP 2: inter 1536 per rank"),
        (4, "MoE TP 4: inter 768 per rank"),
        (16, "MoE TP 16: 192 logical values padded to a 256-value tile"),
    ):
        inter_pad = _round_up(_K31_INTERMEDIATE // moe_tp, WEIGHT_ALIGNMENT)
        cells += [
            Cell(
                why=f"{why}, w3_w1 weight rows",
                spec=dict(rows=2 * inter_pad, cols=_K31_HIDDEN // 2, kind="w31", sf=None),
            ),
            Cell(
                why=f"{why}, w3_w1 scale rows",
                spec=dict(
                    rows=2 * inter_pad,
                    cols=_K31_HIDDEN // SCALING_VECTOR_SIZE,
                    kind="w31",
                    sf=SCALING_VECTOR_SIZE,
                ),
            ),
            Cell(
                why=f"{why}, w2 weight rows",
                spec=dict(rows=_K31_HIDDEN, cols=inter_pad // 2, kind="w2", sf=None),
            ),
            Cell(
                why=f"{why}, w2 scale rows",
                spec=dict(
                    rows=_K31_HIDDEN,
                    cols=inter_pad // SCALING_VECTOR_SIZE,
                    kind="w2",
                    sf=SCALING_VECTOR_SIZE,
                ),
            ),
        ]
    return tuple(cells)


class K31ShuffleMatrix(OpWrapper):
    """``trtllm::shuffle_matrix``: the rows of a 2-D uint8 tensor in ``permute_indices`` order.

    The loader passes a packed weight or block-scale slot and the indices of
    :func:`_w3_w1_permute_indices` / :func:`_w2_permute_indices`; the result is a new tensor of the
    same shape.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = _shuffle_cells()

    note = """
    Index-select along dim 0 on the device: no arithmetic, so bit-exact by construction. The
    indices must be on the input's device.
    """

    def __call__(self, input: torch.Tensor, permute_indices: torch.Tensor) -> torch.Tensor:
        return torch.ops.trtllm.shuffle_matrix(input, permute_indices)

    def reference(self, input: torch.Tensor, permute_indices: torch.Tensor) -> torch.Tensor:
        """Advanced indexing on dim 0."""
        return input[permute_indices.long()]


def _interleave_cells() -> Tuple[Cell, ...]:
    """One cell per scale tensor the loader interleaves, per K3.1 MoE-TP layout."""
    cells = []
    for moe_tp, why in (
        (1, "MoE TP 1: inter 3072"),
        (2, "MoE TP 2: inter 1536 per rank"),
        (4, "MoE TP 4: inter 768 per rank"),
        (16, "MoE TP 16: a 256-value tile, 8 scale columns for w2"),
        (32, "MoE TP 32: a 128-value tile, w2's 4 scale columns are one 4-column group"),
    ):
        inter_pad = _round_up(_K31_INTERMEDIATE // moe_tp, WEIGHT_ALIGNMENT)
        cells += [
            Cell(
                why=f"{why}, w3_w1 scales",
                spec=dict(rows=2 * inter_pad, cols=_K31_HIDDEN // SCALING_VECTOR_SIZE),
            ),
            Cell(
                why=f"{why}, w2 scales",
                spec=dict(rows=_K31_HIDDEN, cols=inter_pad // SCALING_VECTOR_SIZE),
            ),
        ]
    return tuple(cells)


class K31BlockScaleInterleave(OpWrapper):
    """``trtllm::block_scale_interleave``: a 2-D uint8 block-scale matrix ``[rows, cols]`` into the
    runner's 128x4 interleaved order, flattened, rows padded to 128 and columns to 4 with zeros.

    Element ``(r, c)`` lands at ``(c % 4) + (c // 4) * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4
    + (r // 128) * 128 * pad4(cols)``.
    """

    ARCHS = frozenset({Arch.SM_103, Arch.SM_107})
    CELLS: tuple[Cell, ...] = _interleave_cells()

    note = """
    A byte permutation with zero fill: bit-exact by construction. The loader only passes 2-D
    matrices whose rows are a multiple of 128 and columns a multiple of 4, so nothing is padded.
    """

    def __call__(self, block_scale: torch.Tensor) -> torch.Tensor:
        return torch.ops.trtllm.block_scale_interleave(block_scale)

    def reference(self, block_scale: torch.Tensor) -> torch.Tensor:
        """Scatter of every element to its 128x4 offset in a zeroed, padded flat buffer (2-D input,
        the only rank the loader passes)."""
        rows, cols = block_scale.shape
        padded_cols = _round_up(cols, 4)
        r = torch.arange(rows, device=block_scale.device).view(-1, 1)
        c = torch.arange(cols, device=block_scale.device).view(1, -1)
        offset = (
            (c % 4)
            + (c // 4) * (4 * 128)
            + (r % 32) * 16
            + ((r % 128) // 32) * 4
            + (r // 128) * (128 * padded_cols)
        )
        out = torch.zeros(
            _round_up(rows, 128) * padded_cols, dtype=block_scale.dtype, device=block_scale.device
        )
        out[offset.reshape(-1)] = block_scale.reshape(-1)
        return out


k31_shuffle_matrix = K31ShuffleMatrix()
k31_block_scale_interleave = K31BlockScaleInterleave()


class ExpertWeightShapes(NamedTuple):
    """Shapes of the four per-rank expert tensors; every one is :data:`EXPERT_WEIGHT_DTYPE`."""

    w3_w1_weight: Tuple[int, int, int]
    w3_w1_weight_scale: Tuple[int, int, int]
    w2_weight: Tuple[int, int, int]
    w2_weight_scale: Tuple[int, int, int]


def expert_weight_shapes(
    num_local_experts: int, intermediate_size: int, hidden_size: int, moe_tp_size: int
) -> ExpertWeightShapes:
    """The runner's per-rank expert tensors for ``num_local_experts`` slots.

    The experts carry no bias: K3.1's routed experts have none, and the runner is called without one.
    """
    inter_pad = _round_up(intermediate_size // moe_tp_size, WEIGHT_ALIGNMENT)
    w3_w1_hidden = _round_up(hidden_size, INPUT_HIDDEN_ALIGNMENT)
    w2_hidden = _round_up(hidden_size, WEIGHT_ALIGNMENT)
    return ExpertWeightShapes(
        w3_w1_weight=(num_local_experts, 2 * inter_pad, w3_w1_hidden // _VALUES_PER_BYTE),
        w3_w1_weight_scale=(num_local_experts, 2 * inter_pad, w3_w1_hidden // SCALING_VECTOR_SIZE),
        w2_weight=(num_local_experts, w2_hidden, inter_pad // _VALUES_PER_BYTE),
        w2_weight_scale=(num_local_experts, w2_hidden, inter_pad // SCALING_VECTOR_SIZE),
    )


def intermediate_size_lean(intermediate_size: int, moe_tp_size: int, moe_tp_rank: int) -> int:
    """How many of rank ``moe_tp_rank``'s padded intermediate values are real (the runner's
    ``valid_intermediate_size``): its ``inter / moe_tp`` values."""
    return intermediate_size // moe_tp_size


def _load_logical_tp_shard(
    tensor: torch.Tensor,
    dst_shape: torch.Size,
    shard_dim: int,
    values_per_storage_element: int,
    *,
    intermediate_size: int,
    moe_tp_size: int,
    moe_tp_rank: int,
    device: torch.device,
) -> torch.Tensor:
    """Slice rank ``moe_tp_rank``'s logical range along ``shard_dim`` and zero-pad it to ``dst_shape``.

    ``values_per_storage_element`` is 2 for packed weights along K, 32 for block scales along K and
    1 along N.
    """
    logical_size = intermediate_size // moe_tp_size
    if logical_size % values_per_storage_element != 0:
        raise ValueError(
            f"FP4 logical TP shard ({logical_size}) must be divisible by "
            f"values_per_storage_element ({values_per_storage_element})."
        )
    storage_size = logical_size // values_per_storage_element
    start = moe_tp_rank * storage_size
    stop = start + storage_size
    if not 0 <= moe_tp_rank < moe_tp_size:
        raise ValueError(f"tp_rank ({moe_tp_rank}) must be in [0, {moe_tp_size}).")
    if stop > tensor.shape[shard_dim]:
        raise ValueError(
            f"FP4 tensor shape {tuple(tensor.shape)} does not contain the "
            f"logical TP shard [{start}:{stop}] on dim {shard_dim}."
        )
    shard = tensor.narrow(shard_dim, start, storage_size).contiguous()
    if len(shard.shape) != len(dst_shape) or any(
        src > dst for src, dst in zip(shard.shape, dst_shape)
    ):
        raise ValueError(
            f"FP4 logical TP shard shape {tuple(shard.shape)} cannot be "
            f"padded to destination shape {tuple(dst_shape)}."
        )
    if shard.shape != tuple(dst_shape):
        shard = F.pad(
            shard, (0, dst_shape[1] - shard.shape[1], 0, dst_shape[0] - shard.shape[0])
        ).contiguous()
    return shard.to(device)


#: Permute indices by ``(destination shape, "w31" | "w2", values per scale or -1)``: the same
#: shape always gives the same indices, and computing them costs far more than the shuffle.
_PERMUTE_INDICES: Dict[Tuple[torch.Size, str, int], torch.Tensor] = {}


def _w3_w1_permute_indices(
    dst_w3_w1: torch.Tensor, num_elts_per_sf: Optional[int] = None
) -> torch.Tensor:
    """Row order of a ``[w3; w1]`` slot: the gated-activation row reorder, then the epilogue-tile
    shuffle (of weight rows, or of scale rows when ``num_elts_per_sf`` is given)."""
    key = (dst_w3_w1.shape, "w31", int(num_elts_per_sf or -1))
    if key not in _PERMUTE_INDICES:
        permute0 = get_reorder_rows_for_gated_act_gemm_row_indices(dst_w3_w1)
        if num_elts_per_sf is None:
            permute1 = get_shuffle_matrix_a_row_indices(dst_w3_w1, epilogue_tile_m=EPILOGUE_TILE_M)
        else:
            permute1 = get_shuffle_matrix_sf_a_row_indices(
                dst_w3_w1, epilogue_tile_m=EPILOGUE_TILE_M, num_elts_per_sf=num_elts_per_sf
            )
        _PERMUTE_INDICES[key] = permute0[permute1].to(dst_w3_w1.device)
    return _PERMUTE_INDICES[key]


def _w2_permute_indices(
    dst_w2: torch.Tensor, num_elts_per_sf: Optional[int] = None
) -> torch.Tensor:
    """Row order of a ``w2`` slot: the epilogue-tile shuffle of weight rows, or of scale rows."""
    key = (dst_w2.shape, "w2", int(num_elts_per_sf or -1))
    if key not in _PERMUTE_INDICES:
        if num_elts_per_sf is None:
            permute_indices = get_shuffle_matrix_a_row_indices(dst_w2, EPILOGUE_TILE_M).to(
                dst_w2.device
            )
        else:
            permute_indices = get_shuffle_matrix_sf_a_row_indices(
                dst_w2, epilogue_tile_m=EPILOGUE_TILE_M, num_elts_per_sf=num_elts_per_sf
            ).to(dst_w2.device)
        _PERMUTE_INDICES[key] = permute_indices
    return _PERMUTE_INDICES[key]


def _load_w3_w1_weight(
    w1_weight: torch.Tensor,
    w3_weight: torch.Tensor,
    dst_w3_w1_weight: torch.Tensor,
    *,
    intermediate_size: int,
    moe_tp_size: int,
    moe_tp_rank: int,
    device: torch.device,
) -> None:
    dst_w3_weight, dst_w1_weight = dst_w3_w1_weight.chunk(2, dim=0)
    shard_args = dict(
        intermediate_size=intermediate_size,
        moe_tp_size=moe_tp_size,
        moe_tp_rank=moe_tp_rank,
        device=device,
    )
    w1_weight_shard = _load_logical_tp_shard(w1_weight, dst_w1_weight.shape, 0, 1, **shard_args)
    w3_weight_shard = _load_logical_tp_shard(w3_weight, dst_w3_weight.shape, 0, 1, **shard_args)

    dst_w3_weight.copy_(w3_weight_shard.view(dst_w3_weight.dtype))
    dst_w1_weight.copy_(w1_weight_shard.view(dst_w1_weight.dtype))

    permute_indices = _w3_w1_permute_indices(dst_w3_w1_weight)
    processed = k31_shuffle_matrix(dst_w3_w1_weight, permute_indices.to(dst_w3_w1_weight.device))
    dst_w3_w1_weight.copy_(processed.view(dst_w3_w1_weight.dtype), non_blocking=True)


def _load_w2_weight(
    w2_weight: torch.Tensor,
    dst_w2_weight: torch.Tensor,
    *,
    intermediate_size: int,
    moe_tp_size: int,
    moe_tp_rank: int,
    device: torch.device,
) -> None:
    w2_weight_shard = _load_logical_tp_shard(
        w2_weight,
        dst_w2_weight.shape,
        1,
        _VALUES_PER_BYTE,
        intermediate_size=intermediate_size,
        moe_tp_size=moe_tp_size,
        moe_tp_rank=moe_tp_rank,
        device=device,
    )

    dst_w2_weight.copy_(w2_weight_shard.view(dst_w2_weight.dtype), non_blocking=True)

    permute_indices = _w2_permute_indices(dst_w2_weight)
    processed = k31_shuffle_matrix(dst_w2_weight, permute_indices.to(dst_w2_weight.device))
    dst_w2_weight.copy_(processed.view(dst_w2_weight.dtype), non_blocking=True)


def _load_w3_w1_weight_scale(
    w1_weight_scale: torch.Tensor,
    w3_weight_scale: torch.Tensor,
    dst_w3_w1_weight_scale: torch.Tensor,
    *,
    intermediate_size: int,
    moe_tp_size: int,
    moe_tp_rank: int,
    device: torch.device,
) -> None:
    dst_w3_weight_scale, dst_w1_weight_scale = dst_w3_w1_weight_scale.chunk(2, dim=0)
    shard_args = dict(
        intermediate_size=intermediate_size,
        moe_tp_size=moe_tp_size,
        moe_tp_rank=moe_tp_rank,
        device=device,
    )
    w1_weight_scale = _load_logical_tp_shard(
        w1_weight_scale, dst_w1_weight_scale.shape, 0, 1, **shard_args
    )
    w3_weight_scale = _load_logical_tp_shard(
        w3_weight_scale, dst_w3_weight_scale.shape, 0, 1, **shard_args
    )

    dst_w3_weight_scale.copy_(w3_weight_scale.view(dst_w3_weight_scale.dtype))
    dst_w1_weight_scale.copy_(w1_weight_scale.view(dst_w1_weight_scale.dtype))

    orig_shape = dst_w3_w1_weight_scale.shape
    permute_indices = _w3_w1_permute_indices(
        dst_w3_w1_weight_scale.view(float4_sf_dtype), num_elts_per_sf=SCALING_VECTOR_SIZE
    )
    shuffled = k31_shuffle_matrix(dst_w3_w1_weight_scale.view(float4_sf_dtype), permute_indices)
    interleaved = k31_block_scale_interleave(shuffled.view(float4_sf_dtype).reshape(orig_shape))
    dst_w3_w1_weight_scale.copy_(interleaved.view(EXPERT_WEIGHT_DTYPE).reshape(orig_shape))


def _load_w2_weight_scale(
    w2_weight_scale: torch.Tensor,
    dst_w2_weight_scale: torch.Tensor,
    *,
    intermediate_size: int,
    moe_tp_size: int,
    moe_tp_rank: int,
    device: torch.device,
) -> None:
    w2_weight_scale = _load_logical_tp_shard(
        w2_weight_scale,
        dst_w2_weight_scale.shape,
        1,
        SCALING_VECTOR_SIZE,
        intermediate_size=intermediate_size,
        moe_tp_size=moe_tp_size,
        moe_tp_rank=moe_tp_rank,
        device=device,
    )

    dst_w2_weight_scale.copy_(w2_weight_scale.view(dst_w2_weight_scale.dtype))

    orig_shape = dst_w2_weight_scale.shape
    permute_indices = _w2_permute_indices(
        dst_w2_weight_scale.view(float4_sf_dtype), num_elts_per_sf=SCALING_VECTOR_SIZE
    )
    shuffled = k31_shuffle_matrix(dst_w2_weight_scale.view(dtype=float4_sf_dtype), permute_indices)
    interleaved = k31_block_scale_interleave(shuffled)
    dst_w2_weight_scale.copy_(interleaved.view(EXPERT_WEIGHT_DTYPE).reshape(orig_shape))


#: Serializes slot claims across the loader threads (one claim is a check plus an insert).
_SLOT_CLAIM_LOCK = threading.Lock()


def load_packed_mxfp4_expert(
    dst_w3_w1_weight: torch.Tensor,
    dst_w3_w1_weight_scale: torch.Tensor,
    dst_w2_weight: torch.Tensor,
    dst_w2_weight_scale: torch.Tensor,
    local_slot_id: int,
    loaded_slots: Set[int],
    *,
    intermediate_size: int,
    moe_tp_size: int,
    moe_tp_rank: int,
    w1_weight: torch.Tensor,
    w1_weight_scale: torch.Tensor,
    w2_weight: torch.Tensor,
    w2_weight_scale: torch.Tensor,
    w3_weight: torch.Tensor,
    w3_weight_scale: torch.Tensor,
) -> None:
    """Pack one checkpoint expert into slot ``local_slot_id`` of the four runner tensors.

    ``dst_*`` are the rank's ``[E, ...]`` tensors of :func:`expert_weight_shapes`, on the current
    CUDA device. ``w*`` are the expert's full (unsharded) group-32 packed checkpoint tensors, on any
    device; this rank's MoE-TP share of them is cut here. ``intermediate_size`` is the checkpoint's
    (unsharded) expert intermediate size. The slot is added to ``loaded_slots`` (under a lock) before
    anything is written; a slot already in it raises ``ValueError``.
    """
    num_slots = dst_w3_w1_weight.shape[0]
    if not 0 <= local_slot_id < num_slots:
        raise IndexError(f"local_slot_id={local_slot_id} is outside [0, {num_slots}).")
    destinations = {
        "dst_w3_w1_weight": dst_w3_w1_weight,
        "dst_w3_w1_weight_scale": dst_w3_w1_weight_scale,
        "dst_w2_weight": dst_w2_weight,
        "dst_w2_weight_scale": dst_w2_weight_scale,
    }
    for name, value in destinations.items():
        if not value.is_cuda:
            raise ValueError(f"{name} must be a CUDA tensor, got one on {value.device}.")
    tensors = {
        "w1_weight": w1_weight,
        "w1_weight_scale": w1_weight_scale,
        "w2_weight": w2_weight,
        "w2_weight_scale": w2_weight_scale,
        "w3_weight": w3_weight,
        "w3_weight_scale": w3_weight_scale,
    }
    for name, value in tensors.items():
        if value.dtype != torch.uint8:
            raise TypeError(f"{name} must contain packed MXFP4 uint8 data, got {value.dtype}.")
        if value.dim() != 2:
            raise ValueError(f"{name} must be 2-D, got shape {tuple(value.shape)}.")

    # The slot is claimed before the loaders run: a failed load leaves the slot partially
    # transformed, and a retry must not transform it again.
    with _SLOT_CLAIM_LOCK:
        if local_slot_id in loaded_slots:
            raise ValueError(f"Packed MXFP4 local slot {local_slot_id} was loaded twice.")
        loaded_slots.add(local_slot_id)

    shard_args = dict(
        intermediate_size=intermediate_size,
        moe_tp_size=moe_tp_size,
        moe_tp_rank=moe_tp_rank,
        device=torch.device(f"cuda:{torch.cuda.current_device()}"),
    )
    _load_w3_w1_weight(w1_weight, w3_weight, dst_w3_w1_weight.data[local_slot_id], **shard_args)
    _load_w2_weight(w2_weight, dst_w2_weight.data[local_slot_id], **shard_args)
    _load_w3_w1_weight_scale(
        w1_weight_scale, w3_weight_scale, dst_w3_w1_weight_scale.data[local_slot_id], **shard_args
    )
    _load_w2_weight_scale(w2_weight_scale, dst_w2_weight_scale.data[local_slot_id], **shard_args)
