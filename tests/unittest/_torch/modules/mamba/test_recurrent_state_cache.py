# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for shared recurrent-state cache operations."""

import pytest
import torch

from tensorrt_llm._torch.modules.mamba.recurrent_state_cache import reset_recurrent_state_rows

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


@torch.no_grad()
@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
def test_reset_recurrent_state_rows(index_dtype: torch.dtype) -> None:
    slots = 5
    recurrent_states = torch.arange(
        slots * 2 * 3 * 4,
        dtype=torch.float32,
        device="cuda",
    ).reshape(slots, 2, 3, 4)
    conv_states = torch.arange(
        slots * 3 * 2,
        dtype=torch.bfloat16,
        device="cuda",
    ).reshape(slots, 3, 2)
    recurrent_before = recurrent_states.clone()
    conv_before = conv_states.clone()
    conv_expected = conv_states.clone()
    conv_expected[4].zero_()

    state_indices = torch.tensor([4, -1, 1, slots], dtype=index_dtype, device="cuda")
    has_initial_states = torch.tensor([False, False, True, False], device="cuda")
    reset_recurrent_state_rows(
        recurrent_states,
        state_indices,
        has_initial_states,
        conv_states,
    )

    assert torch.count_nonzero(recurrent_states[4]).item() == 0
    assert torch.count_nonzero(conv_states[4]).item() == 0
    torch.testing.assert_close(recurrent_states[:4], recurrent_before[:4])
    torch.testing.assert_close(conv_states[:4], conv_before[:4])

    reset_recurrent_state_rows(
        recurrent_states,
        torch.tensor([2], dtype=index_dtype, device="cuda"),
        torch.tensor([False], device="cuda"),
    )
    assert torch.count_nonzero(recurrent_states[2]).item() == 0
    torch.testing.assert_close(conv_states, conv_expected)


@torch.no_grad()
@pytest.mark.parametrize("state_kind", ["recurrent", "convolution"])
@pytest.mark.parametrize("overlapping", [False, True], ids=["noncontiguous", "overlapping"])
def test_reset_recurrent_state_rows_rejects_invalid_rows(
    state_kind: str,
    overlapping: bool,
) -> None:
    state_indices = torch.tensor([0], dtype=torch.int32, device="cuda")
    has_initial_states = torch.tensor([False], device="cuda")
    recurrent_states = torch.ones(2, 2, 3, 4, dtype=torch.float32, device="cuda")
    conv_states = torch.ones(2, 3, 2, dtype=torch.bfloat16, device="cuda")

    if state_kind == "recurrent":
        if overlapping:
            recurrent_states = torch.as_strided(
                torch.ones(36, dtype=torch.float32, device="cuda"),
                size=(2, 2, 3, 4),
                stride=(12, 12, 4, 1),
            )
        else:
            recurrent_states = torch.ones(
                2,
                2,
                3,
                8,
                dtype=torch.float32,
                device="cuda",
            )[..., ::2]
    elif overlapping:
        conv_states = torch.as_strided(
            torch.ones(9, dtype=torch.bfloat16, device="cuda"),
            size=(2, 3, 2),
            stride=(3, 2, 1),
        )
    else:
        conv_states = torch.ones(2, 3, 4, dtype=torch.bfloat16, device="cuda")[..., ::2]

    state_name = "recurrent" if state_kind == "recurrent" else "convolution"
    constraint = "must not overlap" if overlapping else "must be contiguous"
    with pytest.raises(ValueError, match=f"{state_name} state rows {constraint}"):
        reset_recurrent_state_rows(
            recurrent_states,
            state_indices,
            has_initial_states,
            conv_states,
        )
