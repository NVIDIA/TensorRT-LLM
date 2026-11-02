# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Beam MoE dispatch and weighted combination of decoder block 3 / sm_103 / tp1, called directly.

A component, not a routed target. There is no ``routing.py``, no router-index entry and
no ``ForCausalLM``: nothing in the engine calls this function, and it serves no request.
A caller passes the hidden states that enter block 3's feed-forward sublayer, the router's
result for them -- eight expert ids and eight mixing weights per token -- and an expert
provider. The function sends each token's rows to its experts through the provider and
returns the weighted, combined and scaled expert outputs. It holds no state: no router
parameter, no expert matrix and no configuration object, so there is nothing to build or
load before calling it. It does not call the built-in Beam modeling.

Ownership. This component owns dispatch and combination; the provider owns the expert
computation. ``expert_provider(expert_id, rows)`` is called once for every selected
expert, in ascending id order, with that expert's ``rows`` -- ``[n, 5120]``
``torch.bfloat16``, ``n >= 1``, the routed tokens in ascending token order -- and returns
the expert's unweighted output for them: ``[n, 5120]`` ``torch.bfloat16`` on the same
device, row ``i`` for input row ``i``. It applies no mixing weight and no routed scaling
factor; both are applied here. A routed expert component per id adapts to it as
``lambda expert_id, rows: experts[expert_id](rows)``. The router owns selection: no expert
is re-selected, no bias is added and no weight is re-normalized here. No residual,
normalization or collective is applied.

The combination, as the source model computes it (``ops/moe/beam_dispatch_combine.py``)::

    for each selected expert e, ascending:
        out   = expert_provider(e, hidden[tokens routed to e])     [n, 5120] bf16
        route = (out.float() * weights[:, None]).to(bfloat16)      [n, 5120] each route rounded once
        acc[tokens] += route.float()                               [T, 5120] fp32
    output = (acc * 2.5).to(bfloat16)                              [T, 5120] bf16
"""

from __future__ import annotations

import functools
from typing import Callable

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.moe.beam_dispatch_combine import (
    beam_dispatch_combine_op,
)

HIDDEN_SIZE = 5120
NUM_EXPERTS = 256
EXPERTS_PER_TOKEN = 8
ROUTED_SCALING_FACTOR = 2.5

#: ``expert_provider(expert_id, rows) -> output``; see the module docstring.
ExpertProvider = Callable[[int, torch.Tensor], torch.Tensor]

_SM = (10, 3)


def _check_tensor(name: str, tensor, dtype: torch.dtype) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must be {dtype}, got {tensor.dtype}")
    if tensor.layout != torch.strided:
        raise ValueError(f"{name} must be a dense tensor, got {tensor.layout}")


def _check_inputs(
    hidden_states: torch.Tensor,
    expert_ids: torch.Tensor,
    mixing_weights: torch.Tensor,
    expert_provider: ExpertProvider,
) -> None:
    """Raise on any input this component would otherwise compute wrongly.

    Reads tensor metadata only. The ids' values are not checked: each in ``[0, 256)`` and
    the eight ids of a row distinct are the caller's preconditions -- the router's output
    satisfies both -- because checking them would copy the routing table to the host on
    every call.
    """
    _check_tensor("hidden_states", hidden_states, torch.bfloat16)
    _check_tensor("expert_ids", expert_ids, torch.int64)
    _check_tensor("mixing_weights", mixing_weights, torch.float32)
    if hidden_states.ndim != 2 or hidden_states.shape[1] != HIDDEN_SIZE:
        raise ValueError(
            f"hidden_states must be [tokens, {HIDDEN_SIZE}], got {list(hidden_states.shape)}"
        )
    tokens = hidden_states.shape[0]
    if tokens < 1:
        raise ValueError("hidden_states must hold at least one token")
    for name, tensor in (("expert_ids", expert_ids), ("mixing_weights", mixing_weights)):
        if tuple(tensor.shape) != (tokens, EXPERTS_PER_TOKEN):
            raise ValueError(
                f"{name} must be [tokens, {EXPERTS_PER_TOKEN}] = [{tokens}, "
                f"{EXPERTS_PER_TOKEN}] for {tokens} tokens, got {list(tensor.shape)}"
            )
    device = hidden_states.device
    if device.type != "cuda" or torch.cuda.get_device_capability(device) != _SM:
        raise ValueError(
            f"hidden_states must be on an sm_{_SM[0]}{_SM[1]} CUDA device, got {device}"
        )
    for name, tensor in (("expert_ids", expert_ids), ("mixing_weights", mixing_weights)):
        if tensor.device != device:
            raise ValueError(
                f"{name} must be on hidden_states' device {device}, got {tensor.device}"
            )
    if not callable(expert_provider):
        raise TypeError(f"expert_provider must be callable, got {type(expert_provider).__name__}")


def _checked_expert_output(
    expert_provider: ExpertProvider, expert_id: int, rows: torch.Tensor
) -> torch.Tensor:
    """Call the provider once and return its output unchanged, or raise if it breaks the contract."""
    out = expert_provider(expert_id, rows)
    if not isinstance(out, torch.Tensor):
        raise TypeError(
            f"expert_provider({expert_id}) must return a torch.Tensor, got {type(out).__name__}"
        )
    if out.dtype != torch.bfloat16:
        raise TypeError(f"expert_provider({expert_id}) must return torch.bfloat16, got {out.dtype}")
    if out.shape != rows.shape:
        raise ValueError(
            f"expert_provider({expert_id}) must return {list(rows.shape)} for "
            f"{rows.shape[0]} rows, got {list(out.shape)}"
        )
    if out.device != rows.device or out.layout != torch.strided:
        raise ValueError(
            f"expert_provider({expert_id}) must return a dense tensor on {rows.device}, "
            f"got {out.layout} on {out.device}"
        )
    return out


@torch.no_grad()
def beam_dispatch_combine(
    hidden_states: torch.Tensor,
    expert_ids: torch.Tensor,
    mixing_weights: torch.Tensor,
    expert_provider: ExpertProvider,
) -> torch.Tensor:
    """Dispatch tokens to their experts and return the combined output.

    ``hidden_states`` ``[T, 5120]`` ``torch.bfloat16`` on an sm_103 CUDA device, T >= 1;
    ``expert_ids`` ``[T, 8]`` ``torch.int64`` and ``mixing_weights`` ``[T, 8]``
    ``torch.float32`` on the same device, as the router returns them; ``expert_provider``
    as in the module docstring. Returns ``[T, 5120]`` ``torch.bfloat16`` on that device,
    row ``t`` for token ``t``. Inputs outside that, and a provider output outside its
    contract, are rejected with an error naming the argument; the three tensors are never
    modified.

    Preconditions, not checked: every id in ``[0, 256)`` and the eight ids of each row
    distinct. Mixing weights are used as given; they need not sum to one, and a zero
    weight still calls its expert.
    """
    _check_inputs(hidden_states, expert_ids, mixing_weights, expert_provider)
    return beam_dispatch_combine_op(
        hidden_states,
        expert_ids,
        mixing_weights,
        functools.partial(_checked_expert_output, expert_provider),
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
    )
