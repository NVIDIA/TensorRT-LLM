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
    calls = []
    x = torch.zeros(ROWS, HC * HIDDEN, dtype=torch.bfloat16)
    inject = torch.zeros(ROWS, HC, dtype=torch.bfloat16)
    routed = torch.zeros(ROWS, HIDDEN, dtype=routed_dtype)
    shared = torch.zeros(ROWS, HIDDEN, dtype=torch.bfloat16)
    gate = torch.zeros(ROWS, 3, dtype=torch.bfloat16)[:, 1:2]
    monkeypatch.setattr(M, "cublas_mm", lambda a, b: a)
    monkeypatch.setattr(M, "qwen4_exp_hc_inject_norm", lambda *a: (x, x))
    monkeypatch.setattr(M, "_hc_mixer", lambda *a: (x[:, :HIDDEN], inject))

    def block(core, M_, mixed, pad_rows, routed_only=False):
        assert routed_only and pad_rows == 3
        return routed

    monkeypatch.setattr(M, "_mtp_moe_block", block)
    monkeypatch.setattr(M, "_mtp_shared_expert", lambda core, M_, mixed: (gate, shared))
    monkeypatch.setattr(M, "mtp_moe_combine_inject", lambda *a: calls.append("one") or "one")
    monkeypatch.setattr(M, "fused_sigmoid_gate_mul_add", lambda *a: calls.append("combine"))
    monkeypatch.setattr(M, "qwen4_exp_post_inject", lambda *a: calls.append("inject") or "two")
    core = SimpleNamespace(tp_sum=lambda t: t, eps=1e-6, hc_count=HC)
    weights = SimpleNamespace(o=None, mhc_norm=None, mhc_first=None, mhc_up=None)
    assert M._mtp_tail(core, weights, x, None, inject, 3) == expected
    assert calls == (["one"] if expected == "one" else ["combine", "inject"])
