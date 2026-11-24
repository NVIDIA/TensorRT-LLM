# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Verify the target head reuses valid generation rows and gathers other steps."""

import importlib
import types

import pytest
import torch

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
_PKG = f"{_FAMILY}.kimi_k31_pro_nvfp4__sm_107__helix16_moetp16ep1"
HIDDEN, SHARD = 7168, 10240


def _worker():
    modeling = importlib.import_module(f"{_PKG}.modeling")
    worker = object.__new__(modeling.DFlashWorker)
    worker.mapping = types.SimpleNamespace(rank=0)
    return modeling, worker


def _inputs(rows: int, contexts: int = 0, ids=None):
    md = types.SimpleNamespace(num_contexts=contexts)
    if ids is None:
        ids = torch.arange(rows, dtype=torch.int32, device="cuda")
    return md, types.SimpleNamespace(gather_ids=ids)


@pytest.mark.parametrize("rows", (1, 8, 16, 32))
def test_identity_rows_are_read_in_place_and_give_the_same_logits(rows: int) -> None:
    modeling, worker = _worker()
    buffer = torch.randn(64, HIDDEN, device="cuda", dtype=torch.bfloat16)
    hidden = buffer[:rows]
    weight = (torch.randn(SHARD, HIDDEN, device="cuda") * HIDDEN**-0.5).bfloat16()
    md, spec = _inputs(rows)
    gathered = hidden[spec.gather_ids]
    direct = worker.target_head_rows(hidden, md, spec)
    assert gathered.data_ptr() != hidden.data_ptr()
    assert direct is hidden
    a = modeling.head_linear(gathered, weight)
    b = modeling.head_linear(direct, weight)
    torch.cuda.synchronize()
    assert torch.equal(a.view(torch.int16), b.view(torch.int16))


def test_other_steps_still_gather() -> None:
    _, worker = _worker()
    hidden = torch.randn(16, HIDDEN, device="cuda", dtype=torch.bfloat16)
    for md, spec in (
        _inputs(16, contexts=1),
        _inputs(16, ids=torch.arange(8, dtype=torch.int32, device="cuda")),
    ):
        rows = worker.target_head_rows(hidden, md, spec)
        assert rows.data_ptr() != hidden.data_ptr()
        assert torch.equal(rows, hidden[spec.gather_ids])
    # Not contiguous: gathered.
    strided = torch.randn(16, 2 * HIDDEN, device="cuda", dtype=torch.bfloat16)[:, :HIDDEN]
    rows = worker.target_head_rows(strided, *_inputs(16))
    assert rows.is_contiguous() and torch.equal(rows, strided)


def test_ids_that_are_not_the_identity_raise() -> None:
    _, worker = _worker()
    hidden = torch.randn(8, HIDDEN, device="cuda", dtype=torch.bfloat16)
    ids = torch.arange(8, dtype=torch.int32, device="cuda").flip(0)
    with pytest.raises(RuntimeError, match="not arange"):
        worker.target_head_rows(hidden, *_inputs(8, ids=ids))


def test_graph_replay_equals_eager_gather() -> None:
    modeling, worker = _worker()
    rows = 8
    hidden = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(SHARD, HIDDEN, device="cuda") * HIDDEN**-0.5).bfloat16()
    md, spec = _inputs(rows)
    worker.target_head_rows(hidden, md, spec)  # the eager check
    modeling.head_linear(hidden, weight)  # prepares the head's kernel
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        logits = modeling.head_linear(worker.target_head_rows(hidden, md, spec), weight)
    for _ in range(3):
        hidden.normal_()
        graph.replay()
        eager = modeling.head_linear(hidden[spec.gather_ids], weight)
        torch.cuda.synchronize()
        assert torch.equal(logits.view(torch.int16), eager.view(torch.int16))
