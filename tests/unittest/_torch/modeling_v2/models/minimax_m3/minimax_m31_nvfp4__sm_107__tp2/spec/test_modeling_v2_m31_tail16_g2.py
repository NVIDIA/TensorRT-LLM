# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TAIL16_G2 on the tp2 target: the worker's _zero_padding_rows in place and in one launch
(trtllm::minimax_m31_zero_padding_rows). CPU: the stock fallbacks and the kill switch. GPU (SM107, op built):
bitwise the stock SpecWorkerBase._zero_padding_rows result with 0 / 1 / all padding requests at 3 and 4 rows
per request."""

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2 import (  # noqa: E501
    dspark_s4 as w1,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.ops.spec import (  # noqa: E501
    m31_markov_epilogue as epi,
)
from tensorrt_llm._torch.speculative import interface
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase

DUMMY = 7


@pytest.fixture(autouse=True)
def _zeroed_padding_rows(monkeypatch):
    # The sm_107 targets keep padding rows zeroed (routing arms interface.arm_sm107_spec_paths).
    monkeypatch.setattr(interface, "_CHEAP_PADDING_ROWS", False)


def _meta(slots, ragged=False):
    return SimpleNamespace(batch_slot_ids=slots, dummy_slot_row=DUMMY, is_ragged_verify=ragged)


def test_g2_is_on_by_default():
    assert w1.TAIL16_G2


def test_kill_switch():
    code = f"from {w1.__name__} import TAIL16_G2\nprint(TAIL16_G2)"
    env = {**os.environ, "TRTLLM_M31_DISABLE_TAIL16_G2": "1"}
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    assert out.stdout.split()[-1] == "False"


@pytest.mark.parametrize(
    "why, kwargs", [("ragged verify", dict(packed_verify=True)), ("row mismatch", dict(rpr=5))]
)
def test_other_layouts_take_the_stock_path(monkeypatch, why, kwargs):
    seen = []
    monkeypatch.setattr(SpecWorkerBase, "_zero_padding_rows",
                        staticmethod(lambda *a, **k: seen.append(why) or "stock"))  # fmt: skip
    monkeypatch.setattr(w1, "markov_epilogue_op", lambda: object())
    slots = torch.tensor([1, DUMMY, 2, 3])
    logits = torch.zeros(16, 8)
    out = w1.S4WorkerMixin._zero_padding_rows(logits, _meta(slots, ragged=True), 0, 4, kwargs.get("rpr", 4),
                                              packed_verify=kwargs.get("packed_verify", False))  # fmt: skip
    assert out == "stock" and seen == [why]


_GPU = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 7)
    or epi.markov_epilogue_op() is None,
    reason="the TAIL16_G2 op runs on SM107 with the op built",
)


@_GPU
@pytest.mark.parametrize("rpr", [3, 4])
@pytest.mark.parametrize("npad", [0, 1, 4])
def test_bitwise_the_stock_zeroing(rpr, npad):
    nc, ng, v = 2, 4, 200064
    slots = torch.arange(10, 10 + nc + ng, dtype=torch.int64, device="cuda")
    slots[nc + ng - npad : nc + ng] = DUMMY
    logits = torch.randn(ng * rpr, v, device="cuda")
    logits[0, 5] = float("nan")
    ref = SpecWorkerBase._zero_padding_rows(logits.clone(), _meta(slots), nc, nc + ng, rpr)
    got = w1.S4WorkerMixin._zero_padding_rows(logits.clone(), _meta(slots), nc, nc + ng, rpr)
    assert torch.equal(ref.view(torch.int32), got.view(torch.int32))


@_GPU
def test_graph_replay():
    nc, ng, rpr, v = 0, 4, 4, 200064
    slots = torch.tensor([3, 4, DUMMY, 5], dtype=torch.int64, device="cuda")
    base = torch.randn(ng * rpr, v, device="cuda")
    ref = SpecWorkerBase._zero_padding_rows(base.clone(), _meta(slots), nc, ng, rpr)
    x = base.clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    g = torch.cuda.CUDAGraph()
    with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
        w1.S4WorkerMixin._zero_padding_rows(x, _meta(slots), nc, ng, rpr)
    torch.cuda.current_stream().wait_stream(s)
    x.copy_(base)
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(ref.view(torch.int32), x.view(torch.int32))
