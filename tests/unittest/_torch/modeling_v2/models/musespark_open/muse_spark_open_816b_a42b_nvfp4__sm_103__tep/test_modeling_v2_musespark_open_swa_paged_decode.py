# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tep8 target's sliding-window paged decode (``swa_paged_decode``): its wiring.

Host-only, on a stand-in text model whose layers own built-in
``MuseSparkOpenAttention`` objects created without their constructor (which
builds the projections, the attention backend and CUDA streams). What the
decode attention computes is the GPU test's business; here: which plan the
fast path makes, which modules are re-classed, which calls carry the plan, how
``_attn_impl`` dispatches per layer kind, and that nothing changes under
``NO_PATHS``.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    attention as target_attention,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
    modeling,
)
from tensorrt_llm._torch.models.modeling_musespark_open import (
    MuseSparkOpenAttention,
    MuseSparkOpenModel,
)

pytestmark = pytest.mark.cpu_only

_CORE = modeling.MuseSparkOpenTepModelingV2
_ATTN = target_attention.MuseSparkOpenTepDecodeAttention
_PLAN = target_attention.DecodeAttentionPlan
_PAGED = dispatch.plan_of("swa_paged_decode")


# --------------------------------------------------------------------------
# The plan
# --------------------------------------------------------------------------
def test_the_fast_path_makes_its_plan(monkeypatch):
    """The paged decode quantizes Q to E4M3 as the built-in path does."""
    monkeypatch.setattr(target_attention, "_builtin_prep_kernel_mismatch", lambda: None)
    assert _CORE.adopt(_builtin_text_model(), _PAGED)._decode_attention == _PLAN(
        sliding_window=True, q_fp8=True
    )
    assert _CORE.adopt(_builtin_text_model())._decode_attention is None


# --------------------------------------------------------------------------
# Stand-ins
# --------------------------------------------------------------------------
def _builtin_attention(layer_idx: int, sliding: bool, **overrides) -> MuseSparkOpenAttention:
    """A built-in attention as far as the attributes ``check`` reads go."""
    attention = MuseSparkOpenAttention.__new__(MuseSparkOpenAttention)
    nn.Module.__init__(attention)
    attention.layer_idx = layer_idx
    attention.attn_backend = "TRTLLM"
    attention.support_fused_qkv = True
    attention.head_dim = 64
    attention.num_heads = 16
    attention.num_key_value_heads = 2
    attention.is_sliding = sliding
    attention.attention_window_size = 2048 if sliding else None
    for name, value in overrides.items():
        setattr(attention, name, value)
    return attention


class _Layer(nn.Module):
    """Records each call's extra keywords; owns a built-in attention."""

    def __init__(self, index: int, calls: list) -> None:
        super().__init__()
        self.index = index
        self.pre_qkv_proj_norm = nn.Identity()
        self.self_attn = _builtin_attention(index, sliding=index % 4 != 1)
        self.calls = calls

    def forward(
        self,
        position_ids,
        hidden_states,
        attn_metadata,
        precomputed_attn_input=None,
        next_layer_norm=None,
        sp_num_tokens=None,
        **kwargs,
    ):
        self.calls.append((self.index, dict(kwargs)))
        out = hidden_states + 1.0
        if next_layer_norm is None:
            return out
        return out, out.to(torch.bfloat16)


class _Embedding(nn.Module):
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return input_ids.to(torch.bfloat16).unsqueeze(-1).expand(-1, 4).contiguous()


class _Metadata:
    def __init__(self, num_contexts: int) -> None:
        self.num_contexts = num_contexts


def _builtin_text_model(num_layers: int = 4) -> MuseSparkOpenModel:
    model = MuseSparkOpenModel.__new__(MuseSparkOpenModel)
    nn.Module.__init__(model)
    calls: list = []
    model.embed_tokens = _Embedding()
    model.embed_norm = nn.Identity()
    model.norm = nn.Identity()
    model.layers = nn.ModuleList([_Layer(i, calls) for i in range(num_layers)])
    model.dtype = torch.bfloat16
    model.m_emb = 1.0
    model._prefill_sp_enabled = False
    model._sp_mapping = None
    model.calls = calls
    return model


# --------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------
def test_no_feature_leaves_the_attention_and_the_layer_calls_alone():
    core = _CORE.adopt(_builtin_text_model())
    assert core._decode_attention is None
    assert all(type(layer.self_attn) is MuseSparkOpenAttention for layer in core.layers)
    core(_Metadata(0), torch.tensor([3, 5], dtype=torch.int32), torch.tensor([7, 8]))
    assert [kwargs for _, kwargs in core.calls] == [{}] * 4


def test_a_paged_decode_feature_reclasses_every_attention_and_hands_decode_calls_the_plan():
    model = _builtin_text_model()
    attentions = [layer.self_attn for layer in model.layers]
    modules_before = [name for name, _ in model.named_modules()]
    core = _CORE.adopt(model, _PAGED)
    plan = _PLAN(sliding_window=True, q_fp8=True)
    assert core._decode_attention == plan
    for layer, attention in zip(core.layers, attentions):
        assert layer.self_attn is attention and type(attention) is _ATTN
        assert attention._decode_attention is None
    assert [name for name, _ in core.named_modules()] == modules_before

    ids_, positions = torch.tensor([3, 5], dtype=torch.int32), torch.tensor([7, 8])
    core(_Metadata(0), ids_, positions, spec_metadata="spec")
    assert core.calls == [
        (i, {"spec_metadata": "spec", "decode_attention": plan}) for i in range(4)
    ]
    core.calls.clear()
    core(_Metadata(1), ids_, positions, spec_metadata="spec")
    assert core.calls == [(i, {"spec_metadata": "spec"}) for i in range(4)]


@pytest.mark.parametrize(
    "override, reason",
    [
        (dict(attn_backend="FLASHINFER"), "not TRTLLM"),
        (dict(support_fused_qkv=False), "split Q/K/V"),
        (dict(head_dim=128), "head_dim is not 64"),
        (dict(num_heads=8), "8 query heads over 2 KV heads"),
        (dict(attention_window_size=None), "without a window"),
    ],
)
def test_adoption_refuses_an_attention_the_plan_is_not_written_for(override, reason):
    model = _builtin_text_model()
    for name, value in override.items():
        setattr(model.layers[2].self_attn, name, value)
    with pytest.raises(ValueError, match=f"layer 2: .*{reason}"):
        _CORE.adopt(model, _PAGED)
    assert all(type(layer.self_attn) is MuseSparkOpenAttention for layer in model.layers)


def test_attention_adopt_refuses_anything_but_the_builtin_attention():
    adopted = _ATTN.adopt(_builtin_attention(0, sliding=True))
    with pytest.raises(TypeError, match="MuseSparkOpenTepDecodeAttention"):
        _ATTN.adopt(adopted)
    with pytest.raises(TypeError, match="Module"):
        _ATTN.adopt(nn.Module())


def test_the_plan_is_scoped_to_one_forward_call(monkeypatch):
    """``forward`` holds the plan for exactly its own call and does not pass it on,
    even when the built-in forward raises."""
    seen = []

    def builtin_forward(self, position_ids, hidden_states, attn_metadata, *args, **kwargs):
        seen.append((self._decode_attention, kwargs))
        if kwargs.get("fail"):
            raise RuntimeError("boom")
        return hidden_states

    monkeypatch.setattr(MuseSparkOpenAttention, "forward", builtin_forward)
    attention = _ATTN.adopt(_builtin_attention(0, sliding=True))
    plan = _PLAN(sliding_window=True, q_fp8=False)
    attention(None, torch.zeros(1), None, decode_attention=plan, extra=1)
    assert seen == [(plan, {"extra": 1})] and attention._decode_attention is None
    with pytest.raises(RuntimeError, match="boom"):
        attention(None, torch.zeros(1), None, decode_attention=plan, fail=True)
    assert attention._decode_attention is None


def _dispatch(monkeypatch, sliding: bool, plan, decline=None):
    """Which path ``_attn_impl`` takes, with the built-in and the op path replaced by
    recorders."""
    taken = []
    monkeypatch.setattr(
        MuseSparkOpenAttention,
        "_attn_impl",
        lambda self, *a, **k: taken.append("builtin") or ("b", None),
    )
    monkeypatch.setattr(_ATTN, "_paged_decode_decline", lambda self, *a: decline)
    monkeypatch.setattr(_ATTN, "_paged_decode", lambda self, *a: taken.append(("op", a[-1])) or "o")
    attention = _ATTN.adopt(_builtin_attention(0, sliding=sliding))
    attention._decode_attention = plan
    result = attention._attn_impl(None, None, None, None, None, None, None, 2048, None)
    return result, taken


@pytest.mark.parametrize(
    "sliding, plan, decline, expected",
    [
        (True, None, None, ["builtin"]),
        (False, None, None, ["builtin"]),
        (True, _PLAN(True, False), None, [("op", False)]),
        (True, _PLAN(True, True), None, [("op", True)]),
        (True, _PLAN(True, False), "9 rows", ["builtin"]),
        (False, _PLAN(True, False), None, ["builtin"]),
        (False, _PLAN(True, True), None, ["builtin"]),
    ],
    ids=[
        "swa-no-plan",
        "full-no-plan",
        "swa-paged_decode",
        "swa-paged_decode_q8",
        "swa-declined",
        "full-paged_decode-only",
        "full-paged_decode_q8",
    ],
)
def test_attn_impl_dispatches_per_layer_kind(sliding, plan, decline, expected, monkeypatch):
    result, taken = _dispatch(monkeypatch, sliding, plan, decline)
    assert taken == expected
    assert result == (("o", None) if expected[0][0] == "op" else ("b", None))
