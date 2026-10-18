# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pre-attention prep with the paged decode attention on one attention: the class, its
keywords and the call chain.

Host-only, on a stand-in text model whose layers own built-in
``MuseSparkOpenAttention`` objects created without their constructor, with
what both fast paths' checks read. Layer ``i`` of ``n`` is a full-attention NoPE
layer when ``(n - 1 - i) % 4 == 0`` and a sliding-window RoPE layer otherwise,
as in the checkpoint. The built-in forward, prep and attention op are replaced
by recorders that keep the built-in order (prep, then attention op), so each
test reads which prep and which attention op a call reached and what the
attention op was handed. What the kernels compute is the GPU tests' business
(``test_modeling_v2_musespark_open_decode_forward.py`` runs this class on the
real weights).
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
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention.preattn_prep import (  # noqa: E501
    preattn_prep,
)
from tensorrt_llm._torch.attention.rotary_embedding import RotaryEmbedding
from tensorrt_llm._torch.models.modeling_musespark_open import (
    MuseSparkOpenAttention,
    MuseSparkOpenModel,
)
from tensorrt_llm._torch.modules.rms_norm import RMSNorm

pytestmark = pytest.mark.cpu_only

_CORE = modeling.MuseSparkOpenTepModelingV2
_PREP = target_attention.MuseSparkOpenTepAttention
_DECODE = target_attention.MuseSparkOpenTepDecodeAttention
_BOTH = target_attention.MuseSparkOpenTepPrepDecodeAttention
_PLAN = target_attention.DecodeAttentionPlan


def _full(index: int, num_layers: int) -> bool:
    return (num_layers - 1 - index) % 4 == 0


def _builtin_attention(layer_idx: int, full: bool, **overrides) -> MuseSparkOpenAttention:
    """A built-in attention as far as both features' checks and calls read."""
    attention = MuseSparkOpenAttention.__new__(MuseSparkOpenAttention)
    nn.Module.__init__(attention)
    attention.layer_idx = layer_idx
    attention._fuse_qk_norm_rope_gate = False
    attention.fuse_qk_norm_rope = False
    attention.is_qk_norm = False
    attention.q_norm = RMSNorm(hidden_size=64, eps=1e-5, has_weights=False)
    attention.k_norm = RMSNorm(hidden_size=64, eps=1e-5, has_weights=False)
    attention.rope_fusion = False
    attention.attn_backend = "TRTLLM"
    attention.support_fused_qkv = True
    attention.head_dim = 64
    attention.num_heads = 16
    attention.num_key_value_heads = 2
    attention.scale_query_by = 1.0
    attention._qkv_layout_inverse = None
    attention.is_sliding = not full
    attention.attention_window_size = None if full else 2048
    attention.skip_rope = full
    rotary = None
    if not full:
        rotary = RotaryEmbedding.__new__(RotaryEmbedding)
        nn.Module.__init__(rotary)
        rotary.is_neox = False
        rotary.inverse = False
        rotary.rotary_cos_sin = torch.zeros(16, 2, 32)
    attention.rotary_emb = rotary
    for name, value in overrides.items():
        setattr(attention, name, value)
    return attention


class _Layer(nn.Module):
    """Hands its call to its attention, as the built-in decoder layer does."""

    def __init__(self, index: int, num_layers: int, calls: list) -> None:
        super().__init__()
        self.index = index
        self.pre_qkv_proj_norm = nn.Identity()
        self.self_attn = _builtin_attention(index, full=_full(index, num_layers))
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
        self.calls.append((self.index, sorted(kwargs)))
        self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            **kwargs,
        )
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
    model.layers = nn.ModuleList([_Layer(i, num_layers, calls) for i in range(num_layers)])
    model.dtype = torch.bfloat16
    model.m_emb = 1.0
    model._prefill_sp_enabled = False
    model._sp_mapping = None
    model.calls = calls
    return model


@pytest.fixture
def adoptable(monkeypatch):
    """Both features' environment checks pass on the host."""
    monkeypatch.setattr(target_attention, "_builtin_prep_kernel_mismatch", lambda: None)


@pytest.fixture
def recorded(monkeypatch):
    """The built-in forward, prep and attention op and both features' kernels as recorders.

    The built-in forward keeps the built-in order: ``preprocess_qkv`` on the
    projection, then ``_attn_impl`` on what the prep returned.
    """
    taken: list = []

    def builtin_forward(self, position_ids, hidden_states, attn_metadata, **kwargs):
        taken.append(("keywords", self.layer_idx, self._preattn_prep, self._decode_attention))
        q, k, v, _ = self.preprocess_qkv(hidden_states, position_ids)
        return self._attn_impl(
            q, k, v, attn_metadata, None, None, None, self.attention_window_size, None
        )[0]

    def builtin_prep(self, qkv, position_ids):
        taken.append(("builtin_prep", self.layer_idx))
        return "builtin-qkv", None, None, None

    def builtin_attn(self, q, *args, **kwargs):
        taken.append(("builtin_attn", self.layer_idx, q))
        return "b", None

    def paged_decode(self, q, md, window, q_fp8):
        taken.append(("paged_decode", self.layer_idx, q, q_fp8))
        return "o"

    monkeypatch.setattr(MuseSparkOpenAttention, "forward", builtin_forward)
    monkeypatch.setattr(MuseSparkOpenAttention, "preprocess_qkv", builtin_prep)
    monkeypatch.setattr(MuseSparkOpenAttention, "_attn_impl", builtin_attn)
    monkeypatch.setattr(_PREP, "_fits_preattn_prep", lambda self, qkv, position_ids: True)
    monkeypatch.setattr(_DECODE, "_paged_decode_decline", lambda self, *a: None)
    monkeypatch.setattr(_DECODE, "_paged_decode", paged_decode)
    return taken


def _prep_entry(taken: list):
    """A stand-in ``preattn_prep``: records whether it was handed the RoPE table."""

    def entry(qkv, cos_sin, position_ids, *args):
        taken.append(("preattn_prep", cos_sin is not None))
        return "prepped-qkv", "gate"

    return entry


# --------------------------------------------------------------------------
# The class
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "preattn_prep_on, plan, expected",
    [
        (False, None, None),
        (True, None, _PREP),
        (False, _PLAN(True, False), _DECODE),
        (True, _PLAN(True, False), _BOTH),
    ],
    ids=["none", "preattn_prep", "paged_decode", "prep+paged_decode"],
)
def test_one_class_carries_what_is_selected(preattn_prep_on, plan, expected):
    assert target_attention._attention_class(preattn_prep_on, plan) is expected


def test_the_combined_class_overrides_no_method_of_either_feature_twice():
    """Each feature owns its methods; the combined class adds only its check and adopt."""

    def methods(cls):
        return {
            name
            for name, value in vars(cls).items()
            if callable(value) or isinstance(value, classmethod)
        } - {"forward", "check", "adopt"}

    assert methods(_PREP) & methods(_DECODE) == set()
    assert "preprocess_qkv" in methods(_PREP) and "_attn_impl" in methods(_DECODE)
    assert methods(_BOTH) == set()
    assert _BOTH.__mro__[:4] == (_BOTH, _PREP, _DECODE, MuseSparkOpenAttention)
    assert _BOTH.preprocess_qkv is _PREP.preprocess_qkv
    assert _BOTH._attn_impl is _DECODE._attn_impl


@pytest.mark.parametrize(
    "adopted_by",
    [_PREP.adopt, _DECODE.adopt, _BOTH.adopt],
    ids=["preattn_prep", "paged_decode", "prep+paged_decode"],
)
def test_the_combined_adopt_takes_only_the_builtin_attention(adopted_by):
    attention = adopted_by(_builtin_attention(0, full=False))
    for cls in (_PREP, _DECODE, _BOTH):
        with pytest.raises(TypeError, match="adopts the built-in MuseSparkOpenAttention"):
            cls.adopt(attention)


# --------------------------------------------------------------------------
# Adoption through the core
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "paths, cls",
    [
        (("preattn_prep",), _PREP),
        (("swa_paged_decode",), _DECODE),
        (("preattn_prep", "swa_paged_decode"), _BOTH),
    ],
    ids=["preattn_prep", "paged_decode", "prep-paged_decode"],
)
def test_the_core_reclasses_every_attention_once_with_the_class_for_the_plan(paths, cls, adoptable):
    model = _builtin_text_model()
    attentions = [layer.self_attn for layer in model.layers]
    modules_before = [name for name, _ in model.named_modules()]
    core = _CORE.adopt(model, dispatch.plan_of(*paths))
    for layer, attention in zip(core.layers, attentions):
        assert layer.self_attn is attention and type(attention) is cls
        assert getattr(attention, "_preattn_prep", None) is None
        assert getattr(attention, "_decode_attention", None) is None
    assert [name for name, _ in core.named_modules()] == modules_before
    # The paged decode always quantizes Q to E4M3 (paged_decode_q8).
    paged = _PLAN(True, True) if "swa_paged_decode" in paths else None
    assert core._decode_attention == paged
    assert core._decode.preattn_prep is (preattn_prep if "preattn_prep" in paths else None)


@pytest.mark.parametrize(
    "override, reason",
    [
        (dict(rope_fusion=True), "preattn_prep cannot replace this prep: .*inside the attention"),
        (
            dict(attn_backend="FLASHINFER"),
            "the swa_paged_decode decode attention cannot run here: .*not TRTLLM",
        ),
        (
            dict(num_heads=8),
            "the swa_paged_decode decode attention cannot run here: .*8 query heads",
        ),
        (dict(is_qk_norm=True), "preattn_prep cannot replace this prep: .*carry weights"),
    ],
    ids=["prep-check", "paged_decode-backend", "paged_decode-heads", "prep-norm"],
)
def test_a_refusal_by_either_check_leaves_every_attention_alone(override, reason, adoptable):
    model = _builtin_text_model()
    for name, value in override.items():
        setattr(model.layers[2].self_attn, name, value)
    with pytest.raises(ValueError, match=f"layer 2: {reason}"):
        _CORE.adopt(model, dispatch.plan_of("preattn_prep", "swa_paged_decode"))
    assert all(type(layer.self_attn) is MuseSparkOpenAttention for layer in model.layers)
    assert type(model) is MuseSparkOpenModel


# --------------------------------------------------------------------------
# The call chain
# --------------------------------------------------------------------------
def test_both_keywords_are_scoped_to_one_forward_call(recorded):
    """Each feature's ``forward`` takes its own keyword, hands the rest on and clears
    what it set, even when the built-in forward raises."""
    attention = _BOTH.adopt(_builtin_attention(0, full=False))
    plan = _PLAN(True, False)
    prep = _prep_entry(recorded)
    attention(None, torch.zeros(1), None, preattn_prep=prep, decode_attention=plan)
    assert recorded[0] == ("keywords", 0, prep, plan)
    assert attention._preattn_prep is None and attention._decode_attention is None

    def failing(self, *args, **kwargs):
        recorded.append(("kwargs", sorted(kwargs)))
        raise RuntimeError("boom")

    recorded.clear()
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(MuseSparkOpenAttention, "forward", failing)
        with pytest.raises(RuntimeError, match="boom"):
            attention(None, torch.zeros(1), None, preattn_prep=prep, decode_attention=plan, extra=1)
    assert recorded == [("kwargs", ["extra"])]
    assert attention._preattn_prep is None and attention._decode_attention is None


@pytest.mark.parametrize(
    "full, plan, with_prep, expected",
    [
        (
            False,
            _PLAN(True, False),
            True,
            [("preattn_prep", True), ("paged_decode", 0, "prepped-qkv", False)],
        ),
        (
            False,
            _PLAN(True, True),
            True,
            [("preattn_prep", True), ("paged_decode", 0, "prepped-qkv", True)],
        ),
        (
            True,
            _PLAN(True, False),
            True,
            [("preattn_prep", False), ("builtin_attn", 0, "prepped-qkv")],
        ),
        (
            False,
            _PLAN(True, False),
            False,
            [("builtin_prep", 0), ("paged_decode", 0, "builtin-qkv", False)],
        ),
        (False, None, True, [("preattn_prep", True), ("builtin_attn", 0, "prepped-qkv")]),
        (False, None, False, [("builtin_prep", 0), ("builtin_attn", 0, "builtin-qkv")]),
        (True, None, False, [("builtin_prep", 0), ("builtin_attn", 0, "builtin-qkv")]),
    ],
    ids=[
        "swa-prep-paged_decode",
        "swa-prep-paged_decode_q8",
        "full-prep-paged_decode",
        "swa-paged_decode-only-call",
        "swa-prep-only-call",
        "swa-prefill",
        "full-prefill",
    ],
)
def test_the_prep_output_is_what_the_decode_attention_is_handed(
    full, plan, with_prep, expected, recorded
):
    """A decode call runs the pre-attention prep and hands its fused QKV to the paged decode path for the
    layer kind; a call without a keyword runs that half as built in."""
    attention = _BOTH.adopt(_builtin_attention(0, full=full))
    keywords = {}
    if with_prep:
        keywords["preattn_prep"] = _prep_entry(recorded)
    if plan is not None:
        keywords["decode_attention"] = plan
    out = attention(torch.zeros(1, dtype=torch.int32), torch.zeros(1, 4), None, **keywords)
    assert recorded[1:] == expected
    assert out == ("o" if expected[-1][0] == "paged_decode" else "b")


def test_a_decode_step_through_the_core_reaches_each_layer_kind_and_prefill_runs_as_built_in(
    adoptable, recorded, monkeypatch
):
    entry = _prep_entry(recorded)
    monkeypatch.setattr(modeling, "preattn_prep", entry)
    num_layers = 8
    core = _CORE.adopt(
        _builtin_text_model(num_layers), dispatch.plan_of("preattn_prep", "swa_paged_decode")
    )
    plan = core._decode_attention
    ids_, positions = torch.tensor([3, 5], dtype=torch.int32), torch.tensor([7, 8])

    core(_Metadata(0), ids_, positions, spec_metadata="spec")
    expected = []
    for index in range(num_layers):
        full = _full(index, num_layers)
        expected += [("keywords", index, entry, plan), ("preattn_prep", not full)]
        if full:
            expected.append(("builtin_attn", index, "prepped-qkv"))
        else:
            expected.append(("paged_decode", index, "prepped-qkv", plan.q_fp8))
    assert recorded == expected
    assert core.calls == [
        (i, ["decode_attention", "preattn_prep", "spec_metadata"]) for i in range(num_layers)
    ]

    recorded.clear()
    core.calls.clear()
    core(_Metadata(1), ids_, positions, spec_metadata="spec")
    assert core.calls == [(i, ["spec_metadata"]) for i in range(num_layers)]
    assert recorded == [
        step
        for index in range(num_layers)
        for step in (
            ("keywords", index, None, None),
            ("builtin_prep", index),
            ("builtin_attn", index, "builtin-qkv"),
        )
    ]
