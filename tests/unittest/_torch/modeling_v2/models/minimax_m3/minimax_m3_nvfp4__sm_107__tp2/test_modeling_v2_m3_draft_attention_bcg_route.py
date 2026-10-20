# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Eagle3 drafter layer's attention under a breakable prefill CUDA graph (``M3Eagle3Draft.attention``): it runs
in the eager draft attention boundary on the step's metadata, read at the call (as the in-tree drafter's attention
boundary reads it), and writes the live rows of the padded output; eagerly it calls the op as before. A metadata that
is not the step's is refused. CPU only: the attention op, the linears and the reduction are stand-ins."""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_107__tp2 import (  # noqa: E501
    modeling,
)

pytestmark = pytest.mark.cpu_only

Q_SIZE = 8


@pytest.fixture
def draft(monkeypatch):
    """(drafter, calls of the attention op stand-in, the step's metadata)."""
    calls = []
    step_md = SimpleNamespace(num_tokens=3)

    def op(attn, qkv, md):
        calls.append((attn, qkv, md))
        return torch.full((int(md.num_tokens), Q_SIZE), 7.0, dtype=torch.bfloat16)

    monkeypatch.setattr(modeling, "m3_draft_attention", op)
    monkeypatch.setattr(modeling, "_record", lambda *a: None)
    monkeypatch.setattr(
        modeling, "get_model_extra_attrs", lambda: {"attention_metadata": lambda: step_md}
    )
    d = object.__new__(modeling.M3Eagle3Draft)
    d.__dict__.update(attn="attn", q_size=Q_SIZE, _bcg_key="draft-under-test")
    d.linear = lambda x, name: x
    monkeypatch.setitem(modeling._BCG_DRAFTS, d._bcg_key, d)
    return d, calls, step_md


def test_breakable_graph_runs_the_eager_boundary_on_the_steps_metadata(draft, monkeypatch) -> None:
    d, calls, step_md = draft
    monkeypatch.setattr(modeling, "is_in_breakable_cuda_graph", lambda: True)
    x = torch.zeros(5, 12, dtype=torch.bfloat16)  # 3 live rows + 2 pad rows of the captured bucket
    out = modeling.M3Eagle3Draft.attention(d, x, step_md, False)
    assert len(calls) == 1 and calls[0][0] == "attn" and calls[0][2] is step_md
    assert out.shape == (5, Q_SIZE)
    assert torch.equal(out[:3], torch.full((3, Q_SIZE), 7.0, dtype=torch.bfloat16))


def test_breakable_graph_refuses_a_metadata_that_is_not_the_steps(draft, monkeypatch) -> None:
    d, calls, _ = draft
    monkeypatch.setattr(modeling, "is_in_breakable_cuda_graph", lambda: True)
    with pytest.raises(RuntimeError, match="metadata is not the step's"):
        modeling.M3Eagle3Draft.attention(
            d, torch.zeros(3, 12), SimpleNamespace(num_tokens=3), False
        )
    assert not calls


def test_eager_calls_the_op_directly(draft, monkeypatch) -> None:
    d, calls, _ = draft
    monkeypatch.setattr(modeling, "is_in_breakable_cuda_graph", lambda: False)
    md = SimpleNamespace(num_tokens=2)
    out = modeling.M3Eagle3Draft.attention(d, torch.zeros(2, 12), md, False)
    assert len(calls) == 1 and calls[0][2] is md and out.shape == (2, Q_SIZE)
