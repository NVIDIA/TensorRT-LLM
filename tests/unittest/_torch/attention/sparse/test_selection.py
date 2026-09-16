# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Logical selection and entry-layout contracts."""

from dataclasses import replace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.dsa.selection import DSASelectionPolicy
from tensorrt_llm._torch.attention.backends.sparse.kv_layout import EntryComponent, EntryLayout
from tensorrt_llm._torch.attention.backends.sparse.selection import (
    SelectionContext,
    SelectionPolicy,
)


def _layout(layer_id: int = 7, life_cycle_id: int = 3) -> EntryLayout:
    # Four native entries per page. KV and scales share one pool; V uses another.
    return EntryLayout(
        layer_id,
        life_cycle_id,
        2,
        4,
        (64, 32),
        (
            EntryComponent("kv", 0, 0, 8, 4),
            EntryComponent("scale", 0, 32, 1, 1),
            EntryComponent("v", 1, 0, 4, 2),
        ),
    )


def test_dsa_borrows_positions_and_keeps_identity_order_and_mask() -> None:
    positions = torch.tensor([[4, 1, 4, -1]], dtype=torch.int32)
    context = SelectionContext(
        torch.tensor([123]),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([5], dtype=torch.int32),
        8,
        2,
    )
    mask = torch.tensor([[True, False, True, True]])
    policy: SelectionPolicy = DSASelectionPolicy()
    result = policy.select(positions, context, mask)
    assert result.positions is positions
    assert result.context is context
    assert result.valid_mask is mask
    # IndexShare shares positions, but never substitutes the KV layer's identity.
    next_layer = policy.select(positions, replace(context, layer_id=9, life_cycle_id=4))
    assert next_layer.positions is positions
    assert (next_layer.context.layer_id, next_layer.context.life_cycle_id) == (9, 4)
    assert positions.tolist() == [[4, 1, 4, -1]]


@pytest.mark.parametrize(
    "changes",
    [
        {"entries_per_page": 0},
        {"layer_id": -1},
        {"pool_group_index": -1},
        {"pool_slot_bytes": (16, 32)},
        {"components": ()},
        {"components": (EntryComponent("bad", 2, 0, 4, 4),)},
        {"components": (EntryComponent("same", 0, 0, 4, 4), EntryComponent("same", 1, 0, 4, 4))},
    ],
)
def test_rejects_invalid_layout(changes: dict) -> None:
    with pytest.raises(ValueError):
        replace(_layout(), **changes)


@pytest.mark.parametrize(
    "values", [("", 0, 0, 4, 4), ("kv", -1, 0, 4, 4), ("kv", 0, -1, 4, 4), ("kv", 0, 0, 2, 4)]
)
def test_rejects_invalid_component(values: tuple) -> None:
    with pytest.raises(ValueError):
        EntryComponent(*values)


@pytest.mark.parametrize(
    "invalid", ["positions_dtype", "query_count", "mask_shape", "request_dtype", "length_shape"]
)
def test_rejects_invalid_selection_metadata(invalid: str) -> None:
    ctx = SelectionContext(
        torch.tensor([100]),
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([4], dtype=torch.int32),
        7,
        3,
    )
    positions = torch.tensor([[1, 2]], dtype=torch.int32)
    with pytest.raises(ValueError):
        if invalid == "positions_dtype":
            DSASelectionPolicy().select(positions.long(), ctx)
        elif invalid == "query_count":
            DSASelectionPolicy().select(positions.expand(2, -1), ctx)
        elif invalid == "mask_shape":
            DSASelectionPolicy().select(positions, ctx, torch.ones((1, 3), dtype=torch.bool))
        elif invalid == "request_dtype":
            replace(ctx, request_ids=ctx.request_ids.int())
        else:
            replace(ctx, valid_lengths=torch.ones(2, dtype=torch.int32))
