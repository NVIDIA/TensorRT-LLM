# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""fast_custom_op ops have no autograd kernel, so ops whose bodies make out= calls are registered
with disable_grad=True and run below autograd when grad mode is on, as custom_op ops do."""

import types

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)
from tensorrt_llm._torch.attention import mla
from tensorrt_llm._torch.attention.backends.sparse.dsa import custom_ops as dsa_ops


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
@pytest.mark.parametrize("grad_enabled", [True, False])
def test_bmm_out_with_an_input_that_requires_grad(grad_enabled: bool) -> None:
    a = torch.randn(2, 4, 8, device="cuda", requires_grad=True)
    b = torch.randn(2, 8, 6, device="cuda")
    out = torch.empty(2, 6, 4, device="cuda").transpose(1, 2)
    with torch.set_grad_enabled(grad_enabled):
        if grad_enabled:
            with pytest.raises(RuntimeError, match="out="):
                torch.bmm(a, b, out=out)
        assert torch.ops.trtllm.bmm_out(a, b, out) is None
    with torch.no_grad():
        expected = torch.bmm(a, b)
    torch.testing.assert_close(out, expected)
    assert not out.requires_grad


def _record_grad_mode(seen: list[bool], result=None):
    def body(*args, **kwargs):
        seen.append(torch.is_grad_enabled())
        return result

    return body


def test_mla_custom_op_inplace_runs_its_body_without_grad(monkeypatch) -> None:
    seen = []
    layer = types.SimpleNamespace(forward_impl=_record_grad_mode(seen))
    monkeypatch.setattr(mla, "_extract_mla_extra_attrs", lambda layer_idx: (None, layer))
    with torch.enable_grad():
        torch.ops.trtllm.mla_custom_op_inplace(
            torch.zeros(4, 8), None, "0", torch.empty(4, 8), None
        )
    assert seen == [False]


def test_mla_dsa_proj_runs_its_body_without_grad(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(dsa_ops, "_extract_extra_attrs", lambda layer_idx: (None, None))
    monkeypatch.setattr(dsa_ops, "forward_dsa_proj", _record_grad_mode(seen, [torch.zeros(4, 8)]))
    with torch.enable_grad():
        (q,) = torch.ops.trtllm.mla_dsa_proj(torch.zeros(4, 8), None, "0")
    assert seen == [False]
    assert q.shape == (4, 8)


def test_mla_dsa_attn_inplace_runs_its_body_without_grad(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(dsa_ops, "_extract_extra_attrs", lambda layer_idx: (None, None))
    monkeypatch.setattr(dsa_ops, "_forward_dsa_attn", _record_grad_mode(seen))
    t = torch.zeros(4, 8)
    with torch.enable_grad():
        torch.ops.trtllm.mla_dsa_attn_inplace(t, t, t, t, [], None, "0", torch.empty_like(t))
    assert seen == [False]
