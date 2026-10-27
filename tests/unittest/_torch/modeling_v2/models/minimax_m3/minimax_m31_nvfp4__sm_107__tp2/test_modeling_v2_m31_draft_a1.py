# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""QA-only draft-head routing checks; multi-rank numerics require the separate GPU gate."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (
    dspark as d,
)


class Inner(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, *args):
        self.calls.append(args)
        return "stock"


def make_wrapper(monkeypatch):
    inner = Inner()
    calls = []
    head = SimpleNamespace(apply_linear=lambda x, bias: calls.append((x, bias)) or x)
    monkeypatch.setattr(d._allgather_cast, "allgather_cast", lambda x, **ws: (x, ws))
    monkeypatch.setattr(d.m3_report, "work", lambda *args: None)
    return d._DraftA1Logits(inner, head, {"world": 2}, (66, 84, 154, 196)), head, calls


@pytest.mark.parametrize("rows", [66, 84, 154, 196])
def test_compiled_draft_shape_uses_head_and_existing_workspace(monkeypatch, rows):
    wrapper, head, calls = make_wrapper(monkeypatch)
    hidden = torch.empty((rows, 8), dtype=torch.bfloat16)
    result = wrapper(hidden, head, SimpleNamespace(num_contexts=0), True)
    assert result[0] is hidden and result[1] == wrapper._workspace
    assert calls == [(hidden, None)]
    assert not wrapper.inner.calls


@pytest.mark.parametrize(
    "reason", ["small", "uncompiled", "context", "last_token", "other_head", "dtype"]
)
def test_declines_before_local_head(monkeypatch, reason):
    wrapper, head, calls = make_wrapper(monkeypatch)
    rows = 64 if reason == "small" else (67 if reason == "uncompiled" else 66)
    hidden = torch.empty((rows, 8), dtype=torch.float32 if reason == "dtype" else torch.bfloat16)
    metadata = SimpleNamespace(num_contexts=int(reason == "context"))
    supplied_head = object() if reason == "other_head" else head
    result = wrapper(hidden, supplied_head, metadata, reason != "last_token")
    assert result == "stock" and len(wrapper.inner.calls) == 1
    assert not calls


def test_default_on_and_kill_policy():
    assert d.DRAFT_A1 == (not d.m31_disabled("DRAFT_A1"))
