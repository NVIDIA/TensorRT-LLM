# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Spec-tail contract rejection checks use real CPU tensors and never call the kernels."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.spec.m3_slot_scatter import (
    m3_slot_scatter,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.spec.m3_step_gather import m3_step_gather

pytestmark = pytest.mark.cpu_only


def _gather_args() -> dict:
    return dict(
        new_tok=torch.zeros(4, 8, dtype=torch.int32, device="cpu"),
        next_draft=torch.zeros(8, 3, dtype=torch.int32, device="cpu"),
        lens=torch.full((8,), 128, dtype=torch.int32, device="cpu"),
        slots=torch.arange(2, dtype=torch.int32, device="cpu"),
        ppi=torch.arange(2, dtype=torch.int32, device="cpu").repeat_interleave(4),
        P=2,
        T=4,
        D=3,
        ids=torch.zeros(8, dtype=torch.int32, device="cpu"),
        ids0=0,
        draft=torch.zeros(6, dtype=torch.int32, device="cpu"),
        d0=0,
        pos_off=torch.zeros(8, dtype=torch.int32, device="cpu"),
        kv_off=torch.zeros(2, dtype=torch.int32, device="cpu"),
        b0=0,
    )


@pytest.mark.parametrize("previous_requests", [0, 2])
def test_gather_contract_admits_valid_and_empty_batches(previous_requests: int) -> None:
    args = _gather_args()
    args["P"] = previous_requests
    m3_step_gather.is_valid(**args)


@pytest.mark.parametrize("field", ["slots", "ppi"])
def test_gather_contract_rejects_short_index_buffers(field: str) -> None:
    args = _gather_args()
    args[field] = args[field][:1]
    with pytest.raises(AssertionError):
        m3_step_gather.is_valid(**args)


@pytest.mark.parametrize("field", ["P", "T", "D", "ids0", "d0", "b0"])
def test_gather_contract_rejects_negative_counts_and_offsets(field: str) -> None:
    args = _gather_args()
    args[field] = -1
    with pytest.raises(AssertionError):
        m3_step_gather.is_valid(**args)


def _scatter_args() -> dict:
    return dict(
        slots=torch.tensor([1, 3], dtype=torch.long, device="cpu"),
        sa=torch.zeros(2, 4, dtype=torch.int32, device="cpu"),
        da=torch.zeros(8, 4, dtype=torch.int32, device="cpu"),
        sb=torch.zeros(2, 4, dtype=torch.int32, device="cpu"),
        db=torch.zeros(8, 4, dtype=torch.int32, device="cpu"),
        sl=torch.ones(2, dtype=torch.int32, device="cpu"),
        dl=torch.zeros(8, dtype=torch.int32, device="cpu"),
        sd=torch.zeros(2, 3, dtype=torch.int32, device="cpu"),
        dd=torch.zeros(8, 3, dtype=torch.int32, device="cpu"),
    )


@pytest.mark.parametrize("empty", [False, True])
def test_scatter_contract_admits_valid_and_empty_batches(empty: bool) -> None:
    args = _scatter_args()
    if empty:
        args["slots"] = args["slots"][:0]
    m3_slot_scatter.is_valid(**args)


@pytest.mark.parametrize("field", ["sa", "sb", "sd", "sl"])
def test_scatter_contract_rejects_each_short_source(field: str) -> None:
    args = _scatter_args()
    args[field] = args[field][:1]
    with pytest.raises(AssertionError):
        m3_slot_scatter.is_valid(**args)


@pytest.mark.parametrize("field", ["da", "db", "dd", "dl"])
def test_scatter_contract_rejects_each_short_destination(field: str) -> None:
    args = _scatter_args()
    args[field] = args[field][:3]
    with pytest.raises(AssertionError):
        m3_slot_scatter.is_valid(**args)
