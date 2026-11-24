# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A static routed-expert placement of the K3.1 Rubin targets: its file and its tables.

Expert parallelism over ``W`` ranks gives rank ``r`` the routed-expert ids ("slots")
``[r * E / W, (r + 1) * E / W)``. A placement chooses, per MoE layer, the checkpoint expert each
slot holds: slot ``s`` holds checkpoint expert ``expert_of_slot[s]``. A target loads that expert's
weights into slot ``s`` and the router's row (and score-correction bias) of that expert into row
``s``; routing then returns slots, and nothing past the load changes. The relabeled router scores
every expert with the same bits; only a tie of two scores breaks by slot instead of by checkpoint
expert.

The file is the generic Kimi K3 work's (``TRTLLM_K3_EXPERT_PLACEMENT``)::

    {"format": "kimi_k3_expert_placement/v1",
     "num_experts": 896,
     "ep_size": 16,
     "expert_of_slot": {"<layer>": [num_experts checkpoint expert ids], ...}}

``<layer>`` is the global decoder layer. ``ep_size`` is the EP size the placement was balanced for;
any EP size that divides the experts loads it (a placement is a relabeling). An EP size that
divides ``ep_size`` keeps its balance: each of its ranks holds whole ranks of the balanced one.
A ``kimi_k3_expert_placement/v2`` file loads as its v1 part (its redundant copies are not
served). Layers the file leaves out keep the identity placement.

Not an entry: the targets' weight tables import from it.
"""

from __future__ import annotations

import json
from types import MappingProxyType
from typing import Iterable, Mapping, Optional

#: The file formats this loads; v2 adds redundant copies to v1 and loads as its v1 part.
FORMATS = ("kimi_k3_expert_placement/v1", "kimi_k3_expert_placement/v2")


def read_expert_placement(
    path: str, num_experts: int, moe_layers: Iterable[int], ep_size: int
) -> tuple[Mapping[int, tuple[int, ...]], Optional[int]]:
    """``(expert_of_slot by layer, the file's ep_size)`` of the placement file at ``path``.

    The tables leave out identity layers. Raises ``ValueError`` when the file is not a placement
    of ``num_experts`` experts, a table is not a permutation of them, a listed layer is not one of
    ``moe_layers``, or ``ep_size`` does not divide the experts.
    """
    with open(path) as f:
        data = json.load(f)
    return parse_expert_placement(data, num_experts, moe_layers, ep_size, where=path)


def parse_expert_placement(
    data: dict, num_experts: int, moe_layers: Iterable[int], ep_size: int, where: str = "placement"
) -> tuple[Mapping[int, tuple[int, ...]], Optional[int]]:
    """:func:`read_expert_placement` of the file's parsed JSON ``data``."""
    if not isinstance(data, dict) or data.get("format") not in FORMATS:
        got = data.get("format") if isinstance(data, dict) else type(data).__name__
        raise ValueError(f"{where}: format {got!r}, not one of {FORMATS}")
    if data.get("num_experts") != num_experts:
        raise ValueError(
            f"{where}: a placement of {data.get('num_experts')} experts; the model has {num_experts}"
        )
    if ep_size < 1 or num_experts % ep_size:
        raise ValueError(f"{where}: EP {ep_size} does not split the {num_experts} experts")
    file_ep = data.get("ep_size")
    if file_ep is not None:
        file_ep = int(file_ep)
        if file_ep < 1 or num_experts % file_ep:
            raise ValueError(f"{where}: ep_size {file_ep} does not split {num_experts} experts")
    layers = set(int(i) for i in moe_layers)
    identity = tuple(range(num_experts))
    tables = {}
    for key, expert_of_slot in dict(data.get("expert_of_slot") or {}).items():
        layer = int(key)
        if layer not in layers:
            raise ValueError(f"{where}: layer {layer} is not a MoE layer of the model")
        table = tuple(int(e) for e in expert_of_slot)
        if sorted(table) != list(identity):
            raise ValueError(
                f"{where}: layer {layer}: expert_of_slot is not a permutation of "
                f"range({num_experts})"
            )
        if table != identity:
            tables[layer] = table
    return MappingProxyType(tables), file_ep


def balanced_for(file_ep: Optional[int], ep_size: int) -> bool:
    """Whether a placement balanced for ``file_ep`` ranks stays balanced over ``ep_size`` ranks:
    each of the ``ep_size`` ranks then holds whole ranks of the balanced placement."""
    return file_ep is not None and file_ep % ep_size == 0


def checkpoint_expert(
    expert_of_slot: Optional[Mapping[int, tuple[int, ...]]], layer: int, slot: int
) -> int:
    """The checkpoint expert slot ``slot`` of layer ``layer`` holds (``slot`` where the layer keeps
    the identity placement)."""
    table = None if expert_of_slot is None else expert_of_slot.get(layer)
    return slot if table is None else table[slot]


def rank_loads(counts: Iterable[float], expert_of_slot: tuple[int, ...], ep_size: int) -> list:
    """The routed rows each of ``ep_size`` ranks receives when slot ``s`` holds expert
    ``expert_of_slot[s]`` and expert ``e`` is picked ``counts[e]`` times."""
    counts = list(counts)
    per_rank = len(expert_of_slot) // ep_size
    return [
        sum(counts[e] for e in expert_of_slot[r * per_rank : (r + 1) * per_rank])
        for r in range(ep_size)
    ]


__all__ = [
    "FORMATS",
    "balanced_for",
    "checkpoint_expert",
    "parse_expert_placement",
    "rank_loads",
    "read_expert_placement",
]
