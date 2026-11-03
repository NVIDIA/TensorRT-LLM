# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static routed-expert placement for the Kimi K3 MoE.

Placement (``TRTLLM_K3_EXPERT_PLACEMENT=<file.json>``)
    Expert parallelism gives rank ``r`` the contiguous routed-expert ids ("slots")
    ``[r * n, (r + 1) * n)``. A placement file chooses, per MoE layer, which checkpoint expert
    each slot holds: the model loads checkpoint expert ``expert_of_slot[s]`` into slot ``s`` and,
    once after loading, permutes the router weight rows and the score-correction bias into slot
    order. Routing then returns slot ids directly, so the forward path is unchanged (nothing
    runs at decode time, CUDA graphs are unaffected). K3 routes with one expert group, so the
    relabelled router computes bitwise the same scores and selects the same experts with the same
    weights, except where two scores are equal in fp32: the routing kernels break such ties by
    routed-expert id, i.e. by slot instead of by checkpoint expert. With the real K3 routers that
    reorders the top-k of about 0.1% of the token-layers and changes the selected expert (between
    two equal-score experts) of about 0.01%. Layers the file does not list keep the identity
    placement.

    File format::

        {"format": "kimi_k3_expert_placement/v1",
         "num_experts": 896,
         "ep_size": 16,
         "expert_of_slot": {"<layer_idx>": [num_experts checkpoint expert ids], ...}}

    ``ep_size`` is the expert-parallel size the placement was balanced for; another size still
    loads correctly (a placement is a relabelling) but is not balanced. A
    ``kimi_k3_expert_placement/v2`` file loads as its v1 fields.
"""

import functools
import json
import os
from typing import Optional

import torch

from tensorrt_llm.logger import logger

EXPERT_PLACEMENT_ENV = "TRTLLM_K3_EXPERT_PLACEMENT"

_PLACEMENT_FORMAT = "kimi_k3_expert_placement/v1"
_PLACEMENT_FORMAT_V2 = "kimi_k3_expert_placement/v2"


@functools.lru_cache(maxsize=4)
def load_expert_placement(path: str, num_experts: int) -> dict[int, tuple[int, ...]]:
    """Per-layer ``expert_of_slot`` tables of a placement file, identity layers dropped.

    Raises ``ValueError`` if the file is not a placement for ``num_experts`` experts or a
    table is not a permutation of ``range(num_experts)``.
    """
    with open(path) as f:
        data = json.load(f)
    if data.get("format") not in (_PLACEMENT_FORMAT, _PLACEMENT_FORMAT_V2):
        raise ValueError(
            f"{path}: expected format {_PLACEMENT_FORMAT!r} or {_PLACEMENT_FORMAT_V2!r}, got "
            f"{data.get('format')!r}"
        )
    if data.get("num_experts") != num_experts:
        raise ValueError(
            f"{path}: placement is for {data.get('num_experts')} experts, the model has {num_experts}"
        )
    identity = tuple(range(num_experts))
    tables = {}
    for layer, expert_of_slot in data["expert_of_slot"].items():
        table = tuple(int(e) for e in expert_of_slot)
        if sorted(table) != list(identity):
            raise ValueError(
                f"{path}: layer {layer} expert_of_slot is not a permutation of range({num_experts})"
            )
        if table != identity:
            tables[int(layer)] = table
    return tables


def expert_placement_for_layer(
    layer_idx: int, num_experts: int, ep_size: int
) -> Optional[tuple[int, ...]]:
    """``expert_of_slot`` of ``layer_idx`` from ``TRTLLM_K3_EXPERT_PLACEMENT``, or ``None``."""
    path = os.environ.get(EXPERT_PLACEMENT_ENV)
    if not path:
        return None
    tables = load_expert_placement(path, num_experts)
    with open(path) as f:
        placement_ep = json.load(f).get("ep_size")
    if placement_ep is not None and placement_ep != ep_size:
        logger.warning_once(
            f"{EXPERT_PLACEMENT_ENV}={path} was balanced for ep_size={placement_ep}; the routed "
            f"MoE runs with ep_size={ep_size}, so the placement is applied but not balanced.",
            key="kimi_k3_expert_placement_ep_mismatch",
        )
    logger.info_once(
        f"Kimi K3 static expert placement from {path}: {len(tables)} permuted MoE layers",
        key="kimi_k3_expert_placement_loaded",
    )
    return tables.get(layer_idx)


def permute_router_to_slots(
    weight: torch.Tensor, bias: torch.Tensor, expert_of_slot: tuple[int, ...]
) -> None:
    """Reorder router rows in place so that row ``s`` scores checkpoint expert ``expert_of_slot[s]``.

    ``weight``: ``[num_experts, hidden]``; ``bias``: ``[num_experts]`` (score-correction bias).
    """
    index = torch.tensor(expert_of_slot, dtype=torch.long, device=weight.device)
    with torch.no_grad():
        weight.copy_(weight.index_select(0, index))
        bias.copy_(bias.index_select(0, index.to(bias.device)))
