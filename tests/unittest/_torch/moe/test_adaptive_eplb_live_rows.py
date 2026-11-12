# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The adaptive EPLB statistics ops count only the ids of the live rows of a live-row word
(R real of P requests in bits 0-15 / 16-31; tensorrt_llm/_torch/moe/live_rows.py)."""

import ast
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from _torch.moe import adaptive_eplb_reference as reference
from _torch.moe.test_adaptive_eplb_kernels import TOP_K, _dsv4_case, _stats, _tables, needs_gpu

from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb import balancer, policy
from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.policy import build_route_tables


@pytest.mark.cpu_only
def test_the_balancer_hands_both_ops_the_live_row_word(monkeypatch):
    trees = [ast.parse(inspect.getsource(module)) for module in (balancer, policy)]
    calls = [n for tree in trees for n in ast.walk(tree) if isinstance(n, ast.Call)]
    last = {ast.unparse(c.func): ast.unparse(c.args[-1]) for c in calls if c.args}
    assert last["torch.ops.trtllm.adaptive_eplb_record"] == "live_row_word"
    assert last["self._route_into"] == "self._live_rows(ids)"
    records = [c for c in calls if ast.unparse(c.func).endswith("statistics.record")]
    assert records and all(
        ast.unparse(c.args[-1]) == "self._live_rows(pending[0])" for c in records
    )
    # The statistics take the word only from a kernel that reads it.
    word = torch.zeros(1, dtype=torch.int32)
    monkeypatch.setattr(balancer, "current_word", lambda device: word)
    layer = object.__new__(balancer.AdaptiveMoeLayerLoadBalancer)
    ids = SimpleNamespace(device=torch.device("cuda", 0), is_cuda=True)
    assert layer._live_rows(ids) is None
    layer.counts_live_rows = True
    assert layer._live_rows(ids) is word


@pytest.mark.cpu_only
def test_megamoe_marks_its_balancer_by_what_its_kernels_read():
    pytest.importorskip("cutlass")
    from tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op import megamoe_kernel_classes

    reads = {
        sm: [getattr(kernel, "reads_live_row_word", False) for kernel in megamoe_kernel_classes(sm)]
        for sm in (100, 103, 107)
    }
    assert reads == {100: [True], 103: [True], 107: [False, False]}
    source = (Path(balancer.__file__).parents[1] / "mega_moe" / "mega_moe_cute_dsl.py").read_text()
    marks = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Assign)
        and [ast.unparse(t) for t in node.targets] == ["self.layer_load_balancer.counts_live_rows"]
    ]
    assert len(marks) == 1
    assert "megamoe_kernel_classes(get_sm_version())" in ast.unparse(marks[0].value)


@needs_gpu
@pytest.mark.parametrize(
    "real,requests,live",
    # 32 rows: R of 8 requests keep 4 R rows; no request or 7 requests (not dividing 32) keep 0 or 32.
    [(8, 8, 32), (5, 8, 20), (0, 8, 0), (0, 0, 0), (3, 7, 32)],
)
def test_statistics_count_the_live_rows(real, requests, live):
    layout, shares, ids = _dsv4_case(2)
    tables = _tables(*build_route_tables(layout, shares, 384, 512))
    ids = torch.tensor(ids[: 32 * TOP_K], dtype=torch.int32, device="cuda").view(32, TOP_K)
    word = torch.tensor([(requests << 16) | real], dtype=torch.int32, device="cuda")
    routed, recorded, expected = _stats(384), _stats(384), _stats(384)
    out = torch.empty_like(ids)
    torch.ops.trtllm.adaptive_eplb_route_into(ids, *tables, 3, out, None, None, *routed, True, word)
    torch.ops.trtllm.adaptive_eplb_record(ids, *recorded, True, word)
    reference.record(ids[:live], *expected)
    for stats in (routed, recorded):
        assert torch.equal(stats.rows.view(torch.int32), expected.rows.view(torch.int32))
        assert torch.equal(stats.forwards, expected.forwards)
