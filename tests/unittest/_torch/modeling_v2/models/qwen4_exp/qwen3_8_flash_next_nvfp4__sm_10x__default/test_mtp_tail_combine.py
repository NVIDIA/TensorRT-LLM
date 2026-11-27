# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The draft layer's MoE tail past one rank (`_mtp_tail`): the combination `routed +
sigmoid(gate logit) * shared` and its injection run as one launch (`mtp_moe_combine_inject`)
where it takes the operands, else as `fused_sigmoid_gate_mul_add` + `qwen4_exp_post_inject`.
The op is held bitwise to that composition by its own cells; here the ops are recorders and no
kernel runs."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)

pytestmark = pytest.mark.cpu_only

ROWS, HIDDEN, HC = 4, 8, 4


@pytest.mark.parametrize(
    "routed_dtype, expected", [(torch.bfloat16, "one"), (torch.float32, "two")]
)
def test_tail_combines_and_injects_in_one_launch(monkeypatch, routed_dtype, expected):
    """Distinct nonzero operands: each branch is handed the post-attention stream, the mixer's
    injection, the routed output, the shared expert's gate logit and output, each in its own
    place (identity checks), and returns what its last launch returned."""
    calls = []
    x = torch.full((ROWS, HC * HIDDEN), 1.0, dtype=torch.bfloat16)
    x_after = torch.full((ROWS, HC * HIDDEN), 2.0, dtype=torch.bfloat16)  # after the injection
    inject = torch.full((ROWS, HC), 3.0, dtype=torch.bfloat16)
    inject_moe = torch.full((ROWS, HC), 4.0, dtype=torch.bfloat16)  # the MoE site's injection
    routed = torch.full((ROWS, HIDDEN), 5.0, dtype=routed_dtype)
    shared = torch.full((ROWS, HIDDEN), 6.0, dtype=torch.bfloat16)
    gate = torch.full((ROWS, 3), 7.0, dtype=torch.bfloat16)[:, 1:2]
    combined = torch.full((ROWS, HIDDEN), 8.0, dtype=torch.bfloat16)
    one, two = torch.zeros(1), torch.zeros(2)
    monkeypatch.setattr(M, "cublas_mm", lambda a, b: a)
    monkeypatch.setattr(M, "qwen4_exp_hc_inject_norm", lambda *a: (x_after, x_after))
    monkeypatch.setattr(M, "_hc_mixer", lambda *a: (x_after[:, :HIDDEN], inject_moe))

    def block(core, M_, mixed, pad_rows, routed_only=False):
        assert routed_only and pad_rows == 3
        return routed

    def fused(xx, rr, gg, ss, ii):
        want = (x_after, routed, gate, shared, inject_moe)
        assert all(a is b for a, b in zip((xx, rr, gg, ss, ii), want))
        calls.append("one")
        return one

    def combine(rr, gg, ss):
        assert rr is routed and gg is gate and ss is shared
        calls.append("combine")
        return combined

    def post_inject(xx, cc, ii):
        assert xx is x_after and cc is combined and ii is inject_moe
        calls.append("inject")
        return two

    monkeypatch.setattr(M, "_mtp_moe_block", block)
    monkeypatch.setattr(M, "_mtp_shared_expert", lambda core, M_, mixed: (gate, shared))
    monkeypatch.setattr(M, "mtp_moe_combine_inject", fused)
    monkeypatch.setattr(M, "fused_sigmoid_gate_mul_add", combine)
    monkeypatch.setattr(M, "qwen4_exp_post_inject", post_inject)
    core = SimpleNamespace(tp_sum=lambda t: t, eps=1e-6, hc_count=HC)
    weights = SimpleNamespace(o=None, mhc_norm=None, mhc_first=None, mhc_up=None)
    got = M._mtp_tail(core, weights, x, None, inject, 3)
    assert got is (one if expected == "one" else two)
    assert calls == (["one"] if expected == "one" else ["combine", "inject"])
