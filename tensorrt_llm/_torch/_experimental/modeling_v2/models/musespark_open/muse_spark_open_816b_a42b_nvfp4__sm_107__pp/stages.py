# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which decoder layers each pipeline stage holds, and which splits this target serves. Pure.

The decoder has ``num_layers`` layers; layer ``l`` is a full-attention (NoPE, causal) layer
when ``(num_layers - l - 1) % every_n == 0`` and a sliding-window layer otherwise (the
built-in model's rule). A stage's KV cache is sized by its full-attention layers: they hold
every token of a request, the sliding-window layers only a window. The KV cache manager
syncs the resumable capacity of every stage to the smallest one in tokens, so the stages
must carry the same number of full-attention layers, or the stage with more of them sets
every stage's capacity and the others idle part of their pool (and a stage with none has a
pool sized for windows only).

The split is ``Mapping.pp_layers``: ``pp_partition`` when it is set, else the first
``num_layers % pp_size`` stages take one layer more. ``stage_ranges`` reproduces it without
torch, so the composition can be decided, and tested, on any host.
"""

from __future__ import annotations

import itertools
from typing import List, Optional, Sequence, Tuple


def full_attention_layers(num_layers: int, every_n: int) -> Tuple[int, ...]:
    """The full-attention layers of a decoder of ``num_layers`` layers, ascending."""
    return tuple(layer for layer in range(num_layers) if (num_layers - layer - 1) % every_n == 0)


def stage_ranges(
    num_layers: int, pp_size: int, partition: Optional[Sequence[int]] = None
) -> Tuple[Tuple[int, int], ...]:
    """``(first, end)`` of every stage's layers, as ``Mapping.pp_layers`` splits them.

    Raises:
        ValueError: for a partition of another length or sum (as ``Mapping.pp_layers``).
    """
    if partition is not None:
        sizes = [int(size) for size in partition]
        if len(sizes) != pp_size:
            raise ValueError(f"pp_partition {list(sizes)} has {len(sizes)} stages, not {pp_size}")
        if sum(sizes) != num_layers:
            raise ValueError(f"pp_partition {list(sizes)} sums to {sum(sizes)}, not {num_layers}")
    else:
        base, extra = divmod(num_layers, pp_size)
        sizes = [base + (1 if stage < extra else 0) for stage in range(pp_size)]
    ranges, first = [], 0
    for size in sizes:
        ranges.append((first, first + size))
        first += size
    return tuple(ranges)


def full_layers_per_stage(
    num_layers: int, every_n: int, ranges: Sequence[Tuple[int, int]]
) -> Tuple[int, ...]:
    """How many full-attention layers each stage of ``ranges`` holds."""
    full = full_attention_layers(num_layers, every_n)
    return tuple(sum(1 for layer in full if first <= layer < end) for first, end in ranges)


def partition_refusal(
    num_layers: int, every_n: int, pp_size: int, partition: Optional[Sequence[int]] = None
) -> Optional[str]:
    """Why this target does not serve the split, or None.

    It serves a split whose every stage is non-empty and holds the same number of
    full-attention layers (``num_full / pp_size``).
    """
    try:
        ranges = stage_ranges(num_layers, pp_size, partition)
    except ValueError as error:
        return str(error)
    sizes = [end - first for first, end in ranges]
    if min(sizes) < 1:
        return f"the split {sizes} leaves a stage without a layer"
    full = full_layers_per_stage(num_layers, every_n, ranges)
    total = len(full_attention_layers(num_layers, every_n))
    if total % pp_size or any(count != total // pp_size for count in full):
        return (
            f"the split {sizes} puts {list(full)} full-attention layers on the stages; every "
            f"stage needs {total / pp_size:g} of the {total} so that no stage sets the others' "
            "KV capacity"
        )
    return None


def served_partitions(num_layers: int, every_n: int, pp_size: int) -> List[Tuple[int, ...]]:
    """Every split ``partition_refusal`` accepts, in lexicographic order."""
    served = []
    for cuts in itertools.combinations(range(1, num_layers), pp_size - 1):
        bounds = (0, *cuts, num_layers)
        sizes = tuple(bounds[i + 1] - bounds[i] for i in range(pp_size))
        if partition_refusal(num_layers, every_n, pp_size, sizes) is None:
            served.append(sizes)
    return served
