# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Borrowed selection metadata and model-only entry formats."""

from dataclasses import replace

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.kv_layout import EntryComponent, EntryFormat
from tensorrt_llm._torch.attention.backends.sparse.selection import (
    SelectedEntries,
    SelectionContext,
)


def _context() -> SelectionContext:
    return SelectionContext(
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([5], dtype=torch.int32),
        torch.tensor([3], dtype=torch.int32),
        torch.tensor([12], dtype=torch.uint64),
    )


def _format() -> EntryFormat:
    # BufferId's tuple contract keeps format tests independent of the native binding.
    return EntryFormat(
        (7, "key"),
        (
            EntryComponent("kv", torch.float8_e4m3fn, (4, 512), 0),
            EntryComponent("scale", torch.float32, (4, 4), 0),
        ),
        tokens_per_entry=4,
    )


def test_selection_borrows_metadata_positions_and_mask() -> None:
    positions = torch.tensor([[4, 1, 4, -1]], dtype=torch.int32)
    mask = torch.tensor([[True, False, True, True]])
    context = _context()
    result = SelectedEntries(context, positions, mask)
    assert result.positions is positions
    assert result.context is context
    assert result.valid_mask is mask
    assert positions.tolist() == [[4, 1, 4, -1]]
    # Requests use direct table rows plus a generation, never a request-ID search.
    assert context.host_source_rows.tolist() == [3]
    assert context.host_source_generations.tolist() == [12]


def test_model_format_is_independent_of_storage_placement() -> None:
    fmt = _format()
    assert fmt.buffer_id == (7, "key")
    assert fmt.tokens_per_entry == 4
    assert fmt.components[0].shape[0] == 4
    assert set(vars(fmt)) == {"buffer_id", "components", "tokens_per_entry"}
    head_major = EntryFormat((7, "value"), (EntryComponent("v", torch.bfloat16, (8, 16, 64), 1),))
    assert head_major.components[0].shape[head_major.components[0].entry_axis] == 16


@pytest.mark.parametrize(
    "changes",
    [
        {"tokens_per_entry": 0},
        {"buffer_id": (-1, "key")},
        {"buffer_id": (7, "")},
        {"components": ()},
        {"components": (EntryComponent("same", torch.float16, (4, 32), 0),) * 2},
        {
            "components": (
                EntryComponent("k", torch.float16, (4, 32), 0),
                EntryComponent("scale", torch.float32, (3, 1), 0),
            )
        },
    ],
)
def test_rejects_invalid_format(changes: dict) -> None:
    with pytest.raises(ValueError):
        replace(_format(), **changes)


@pytest.mark.parametrize(
    "values",
    [
        ("", torch.float16, (4, 32), 0),
        ("k", torch.float16, (0, 32), 0),
        ("k", torch.float16, (), 0),
        ("k", torch.float16, (4, 32), 2),
        ("k", torch.float16, (4, 32), -1),
        ("k", "fp16", (4, 32), 0),
    ],
)
def test_rejects_invalid_component(values: tuple) -> None:
    with pytest.raises(ValueError):
        EntryComponent(*values)


@pytest.mark.parametrize(
    "invalid",
    [
        "positions_dtype",
        "query_count",
        "mask_shape",
        "row_dtype",
        "length_shape",
        "generation_dtype",
    ],
)
def test_rejects_invalid_selection_metadata(invalid: str) -> None:
    context = _context()
    positions = torch.tensor([[1, 2]], dtype=torch.int32)
    with pytest.raises(ValueError):
        if invalid == "positions_dtype":
            SelectedEntries(context, positions.long())
        elif invalid == "query_count":
            SelectedEntries(context, positions.expand(2, -1))
        elif invalid == "mask_shape":
            SelectedEntries(context, positions, torch.ones((1, 3), dtype=torch.bool))
        elif invalid == "row_dtype":
            replace(context, host_source_rows=context.host_source_rows.long())
        elif invalid == "generation_dtype":
            replace(context, host_source_generations=context.host_source_generations.long())
        else:
            replace(context, kv_lens_cuda=torch.ones(2, dtype=torch.int32))
