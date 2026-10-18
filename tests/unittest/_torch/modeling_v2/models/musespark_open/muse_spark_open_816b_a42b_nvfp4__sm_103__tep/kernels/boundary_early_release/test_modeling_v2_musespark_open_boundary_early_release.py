# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The early boundary release (``boundary_early_release``): the attention class, the wiring of both edges.

Host-only, on stand-in modules: a text model whose layers own built-in
``MuseSparkOpenAttention`` objects made without their constructor, recorders
for the built-in forward and output gate, and plain objects in place of
the projection gathers' joins and the ``dense_gemms`` dispatchers. What the kernels compute and when
they launch is the GPU tests' business (``attention/test_modeling_v2_norm_gate.py``,
``comm/test_modeling_v2_unpool_gather_boundary_trigger.py``).
"""

from __future__ import annotations

import types

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
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    layers as target_layers,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.boundary_early_release import (  # noqa: E501
    triggers as pdl_triggers,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.projection_gathers import (  # noqa: E501
    gathers as projection_gathers,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.attention import (  # noqa: E501
    norm_gate,
)
from tensorrt_llm._torch.models.modeling_musespark_open import (
    MuseSparkOpenAttention,
    MuseSparkOpenModel,
)

pytestmark = pytest.mark.cpu_only

_CORE = modeling.MuseSparkOpenTepModelingV2
_PLAN = target_attention.DecodeAttentionPlan


def _builtin_attention(layer_idx: int, full: bool) -> MuseSparkOpenAttention:
    """A built-in attention as far as the prep, decode-attention and norm-gate checks and calls read."""
    from tensorrt_llm._torch.attention.rotary_embedding import RotaryEmbedding
    from tensorrt_llm._torch.modules.rms_norm import RMSNorm

    attention = MuseSparkOpenAttention.__new__(MuseSparkOpenAttention)
    nn.Module.__init__(attention)
    attention.layer_idx = layer_idx
    attention._fuse_qk_norm_rope_gate = False
    attention.fuse_qk_norm_rope = False
    attention.is_qk_norm = False
    attention.q_norm = RMSNorm(hidden_size=64, eps=1e-5, has_weights=False)
    attention.k_norm = RMSNorm(hidden_size=64, eps=1e-5, has_weights=False)
    attention.o_head_norm = types.SimpleNamespace(variance_epsilon=1e-6)
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
    return attention


class _Layer(nn.Module):
    def __init__(self, index: int, num_layers: int, calls: list) -> None:
        super().__init__()
        self.index = index
        self.pre_qkv_proj_norm = nn.Identity()
        self.self_attn = _builtin_attention(index, full=(num_layers - 1 - index) % 4 == 0)
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
    """Every environment check of the prep, the decode attention and the norm gate passes on the host.

    The stand-in layers have no projection-gather joins to adopt, so that install (which
    the plan's boundary release builds on) installs nothing here; the install
    checks see its unpool site fused and its edges converted (those refusals are
    tested on their own, in ``test_modeling_v2_musespark_open_runtime_install.py``),
    and the norm-gate wiring is what runs.
    """
    monkeypatch.setattr(target_attention, "_builtin_prep_kernel_mismatch", lambda: None)
    monkeypatch.setattr(projection_gathers, "install", lambda model, unpool, pool: {})
    refuse_sites = target_layers._refuse_unfused_sites
    monkeypatch.setattr(
        target_layers,
        "_refuse_unfused_sites",
        lambda kwargs, plan: refuse_sites({**kwargs, "fused_unpool_gather": True}, plan),
    )
    monkeypatch.setattr(
        target_layers, "_refuse_unconverted_edges", lambda converted, kept, layers: None
    )


@pytest.fixture
def recorded(monkeypatch):
    """The built-in forward calls the output gate as the built-in ``forward_impl`` does."""
    taken: list = []

    def builtin_forward(self, position_ids, hidden_states, attn_metadata, **kwargs):
        taken.append(("forward", self.layer_idx, getattr(self, "_norm_gate_trigger", "absent")))
        return self.apply_output_gate("attention-output", "gate")

    def builtin_gate(self, attention_output, gate):
        taken.append(("builtin_gate", self.layer_idx))
        return "builtin"

    monkeypatch.setattr(MuseSparkOpenAttention, "forward", builtin_forward)
    monkeypatch.setattr(MuseSparkOpenAttention, "apply_output_gate", builtin_gate)
    return taken


# --------------------------------------------------------------------------
# The attention class
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "preattn_prep_on, plan, expected",
    [
        (False, None, pdl_triggers.MuseSparkOpenTepNormGateAttention),
        (True, None, pdl_triggers.MuseSparkOpenTepPrepNormGateAttention),
        (False, _PLAN(True, True), pdl_triggers.MuseSparkOpenTepDecodeNormGateAttention),
        (True, _PLAN(True, True), pdl_triggers.MuseSparkOpenTepPrepDecodeNormGateAttention),
    ],
    ids=["early_release", "prep", "decode", "prep+decode"],
)
def test_one_class_carries_the_norm_gate_on_top_of_what_is_selected(
    preattn_prep_on, plan, expected
):
    assert target_attention._attention_class(preattn_prep_on, plan, True) is expected
    base = target_attention._attention_class(preattn_prep_on, plan)
    assert expected.__mro__[1] is pdl_triggers._NormGatePdl
    assert expected.__mro__[2] is (base or MuseSparkOpenAttention)


def test_without_early_release_the_classes_are_tier1s():
    for preattn_prep_on in (False, True):
        for plan in (None, _PLAN(True, False)):
            assert target_attention._attention_class(
                preattn_prep_on, plan, False
            ) is target_attention._attention_class(preattn_prep_on, plan)


def test_the_norm_gate_overrides_only_the_output_gate():
    owned = {
        name
        for name, value in vars(pdl_triggers._NormGatePdl).items()
        if callable(value) or isinstance(value, classmethod)
    }
    assert owned == {"forward", "apply_output_gate", "_fits_norm_gate", "_check_norm_gate"}
    for cls in pdl_triggers.NORM_GATE_CLASSES.values():
        assert cls.preprocess_qkv is cls.__mro__[2].preprocess_qkv
        assert cls._attn_impl is cls.__mro__[2]._attn_impl


@pytest.mark.parametrize("cls", list(pdl_triggers.NORM_GATE_CLASSES.values()))
def test_every_norm_gate_class_adopts_only_the_builtin_attention(cls):
    attention = cls.adopt(_builtin_attention(0, full=False))
    assert type(attention) is cls and attention._norm_gate_trigger is None
    with pytest.raises(TypeError, match="adopts the built-in MuseSparkOpenAttention"):
        cls.adopt(attention)
    other = _builtin_attention(1, full=False)
    other.head_dim = 128
    with pytest.raises(ValueError, match="head_dim"):
        cls.check(other)


# --------------------------------------------------------------------------
# The call chain
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "paths",
    [
        ("boundary_early_release",),
        ("preattn_prep", "boundary_early_release"),
        ("swa_paged_decode", "boundary_early_release"),
        ("preattn_prep", "swa_paged_decode", "boundary_early_release"),
    ],
)
def test_decode_steps_carry_the_trigger_and_prefill_steps_do_not(
    paths, adoptable, recorded, monkeypatch
):
    calls_gate = []
    monkeypatch.setattr(pdl_triggers._NormGatePdl, "_fits_norm_gate", lambda self, a, g: True)
    monkeypatch.setattr(
        norm_gate,
        "attention_norm_gate",
        lambda a, g, eps, trigger: calls_gate.append((a, g, eps, trigger)) or "pdl",
    )
    model = _builtin_text_model()
    core = _CORE.adopt(model, dispatch.plan_of(*paths))
    assert core._norm_gate_trigger == pdl_triggers.NORM_GATE_TRIGGER == "after_wait"
    assert core._decode.norm_gate_trigger == "after_wait"

    core.forward(_Metadata(0), input_ids=torch.tensor([3, 4]))
    assert all("norm_gate_pdl" in kwargs for _, kwargs in model.calls)
    assert [c[3] for c in calls_gate] == ["after_wait"] * 4
    assert all(c[2] == 1e-6 for c in calls_gate)
    assert all(layer.self_attn._norm_gate_trigger is None for layer in core.layers)

    model.calls.clear()
    calls_gate.clear()
    recorded.clear()
    monkeypatch.setattr(
        MuseSparkOpenModel,
        "forward",
        lambda self, md, input_ids=None, position_ids=None, inputs_embeds=None, **kw: [
            layer(None, torch.zeros(2), md) for layer in self.layers
        ],
    )
    core.forward(_Metadata(1), input_ids=torch.tensor([3, 4]))
    assert all("norm_gate_pdl" not in kwargs for _, kwargs in model.calls)
    assert not calls_gate
    assert [r[0] for r in recorded].count("builtin_gate") == 4


class _T:
    """What ``_fits_norm_gate`` reads of a tensor."""

    def __init__(self, shape, dtype=torch.bfloat16, cuda=True, last_stride=1):
        self.shape = shape
        self.ndim = len(shape)
        self.dtype = dtype
        self.is_cuda = cuda
        self._last = last_stride

    def stride(self, dim):
        return self._last

    def numel(self):
        n = 1
        for d in self.shape:
            n *= d
        return n


@pytest.mark.parametrize(
    "attention, gate, fits",
    [
        (_T((8, 1024)), _T((8, 1024)), True),
        (_T((8, 1024)), _T((8, 16, 64)), True),
        (_T((1, 1024)), _T((1, 1024)), True),
        (_T((9, 1024)), _T((9, 1024)), False),  # above the decode width: the built-in runs
        (_T((8, 1024), dtype=torch.float16), _T((8, 1024)), False),
        (_T((8, 1024), cuda=False), _T((8, 1024)), False),
        (_T((8, 1024), last_stride=2), _T((8, 1024)), False),
        (_T((8, 1024)), _T((8, 16, 32, 2)), False),
        (_T((8, 1024)), _T((8, 32, 32)), False),
        (_T((8, 1000)), _T((8, 1000)), False),
    ],
)
def test_the_pdl_gate_runs_exactly_where_the_builtin_kernel_would_at_decode_widths(
    attention, gate, fits
):
    module = pdl_triggers.MuseSparkOpenTepNormGateAttention.adopt(_builtin_attention(0, False))
    assert module._fits_norm_gate(attention, gate) is fits


# --------------------------------------------------------------------------
# The boundary edge and the dense GEMM prefetches
# --------------------------------------------------------------------------
def _gemm_site(next_weight="w"):
    return types.SimpleNamespace(
        _low_m_gemm_dispatcher=types.SimpleNamespace(
            site=types.SimpleNamespace(next_weight=next_weight, skip_prefetch_rows=frozenset())
        )
    )


def _plan(prefetch=True, enable_pdl=True, kernels=None):
    """The ``dense_gemms`` plan as far as ``boundary_early_release`` reads it.

    The kernel per site and rows, prefetch, PDL.
    """
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.dense_gemms import (  # noqa: E501
        dispatcher as dense_gemms,
    )

    return types.SimpleNamespace(
        kernels=dense_gemms.default_kernels() if kernels is None else kernels,
        prefetch=prefetch,
        enable_pdl=enable_pdl,
    )


def _skipped(model, where):
    sites = {
        "unpool": lambda layer: layer.feed_forward.unpooler,
        "qkv": lambda layer: layer.self_attn.qkv_proj,
    }[where]
    return [sites(layer)._low_m_gemm_dispatcher.site.skip_prefetch_rows for layer in model.layers]


#: Row counts whose QKV kernel in the default dense GEMM plan loads its weight before its wait.
_SPLIT_K_ROWS = frozenset(range(3, 9))
_NONE = frozenset()


def _wired_model(num_layers: int = 3, fused=(True, True, True)):
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.projection_gathers.gathers import (  # noqa: E501
        MuseSparkOpenTepGatedResidual,
    )
    from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenGatedResidual

    layers = []
    for index in range(num_layers):
        if fused[index]:
            join = MuseSparkOpenTepGatedResidual.__new__(MuseSparkOpenTepGatedResidual)
            join.__dict__["_gather_inbox"] = object()
        else:
            join = MuseSparkOpenGatedResidual.__new__(MuseSparkOpenGatedResidual)
        layers.append(
            types.SimpleNamespace(
                add_ffn=join,
                feed_forward=types.SimpleNamespace(unpooler=_gemm_site()),
                self_attn=types.SimpleNamespace(qkv_proj=_gemm_site()),
            )
        )
    return types.SimpleNamespace(layers=layers)


def test_every_fused_boundary_but_the_last_triggers_at_entry():
    model = _wired_model(4, fused=(True, False, True, True))
    assert pdl_triggers.install_boundary_triggers(model) == ([0, 2], [])
    triggers = [getattr(layer.add_ffn, "_boundary_trigger", None) for layer in model.layers]
    assert pdl_triggers.BOUNDARY_TRIGGER == "entry"
    assert triggers == ["entry", None, "entry", "exit"]


def test_a_captured_layer_keeps_the_exit_trigger_and_its_prefetch():
    """DFlash copies a captured layer's output with a plain launch before the next QKV."""
    model = _wired_model(5, fused=(True, True, False, True, True))
    converted, kept = pdl_triggers.install_boundary_triggers(model, frozenset({1, 2, 4}))
    # Layer 2 is not fused (nothing to keep); layer 4 is the last (no next QKV).
    assert (converted, kept) == ([0, 3], [1])
    triggers = [getattr(layer.add_ffn, "_boundary_trigger", None) for layer in model.layers]
    assert triggers == ["entry", "exit", None, "entry", "exit"]
    dropped = pdl_triggers.drop_converted_prefetches(model, _plan(), qkv=True)
    assert dropped == {"qkv": (2, _SPLIT_K_ROWS)}
    assert _skipped(model, "unpool") == [_SPLIT_K_ROWS, _NONE, _NONE, _SPLIT_K_ROWS, _NONE]


def test_the_capture_layers_are_the_dflash_target_layers():
    def model(spec_config):
        return types.SimpleNamespace(model_config=types.SimpleNamespace(spec_config=spec_config))

    dflash = types.SimpleNamespace(target_layer_ids=[59, 1, 16, 30, 44])
    assert pdl_triggers.capture_layers(model(dflash)) == frozenset({1, 16, 30, 44, 59})
    assert pdl_triggers.capture_layers(model(types.SimpleNamespace(target_layer_ids=None))) == (
        frozenset()
    )
    assert pdl_triggers.capture_layers(model(types.SimpleNamespace())) == frozenset()
    assert pdl_triggers.capture_layers(model(None)) == frozenset()
    assert pdl_triggers.capture_layers(types.SimpleNamespace()) == frozenset()


def test_only_split_k_loads_its_weight_before_its_wait():
    plan = _plan()
    assert pdl_triggers.weights_before_wait_rows(plan.kernels["qkv"]) == _SPLIT_K_ROWS
    assert pdl_triggers.weights_before_wait_rows(plan.kernels["o_proj"]) == _NONE
    assert pdl_triggers.weights_before_wait_rows(plan.kernels["unpool"]) == frozenset({6, 7, 8})
    qkv = ("splitk:64,8,4,12",) + plan.kernels["qkv"][1:7] + ("cublas",)
    overridden = _plan(kernels=dict(plan.kernels, qkv=qkv))
    assert pdl_triggers.weights_before_wait_rows(overridden.kernels["qkv"]) == frozenset(
        {1, 3, 4, 5, 6, 7}
    )


def test_the_prefetch_of_a_converted_edge_is_dropped_at_its_split_k_rows():
    qkv_dropped = [True, False, False]
    model = _wired_model(3, fused=(True, False, True))
    pdl_triggers.install_boundary_triggers(model)
    plan = _plan()
    dropped = pdl_triggers.drop_converted_prefetches(model, plan, qkv=True)
    assert _skipped(model, "unpool") == [_SPLIT_K_ROWS if d else _NONE for d in qkv_dropped]
    assert _skipped(model, "qkv") == [_NONE] * 3
    # The next weight is never cleared: rows 1-2 (the direct kernel) still prefetch.
    for layer in model.layers:
        assert layer.feed_forward.unpooler._low_m_gemm_dispatcher.site.next_weight == "w"
        assert layer.self_attn.qkv_proj._low_m_gemm_dispatcher.site.next_weight == "w"
    assert dropped == {"qkv": (sum(qkv_dropped), _SPLIT_K_ROWS)}


def test_no_prefetch_no_pdl_no_change():
    model = _wired_model(2)
    pdl_triggers.install_boundary_triggers(model)
    assert pdl_triggers.drop_converted_prefetches(model, None, qkv=True) == {}
    for plan in (_plan(prefetch=False), _plan(enable_pdl=False)):
        assert pdl_triggers.drop_converted_prefetches(model, plan, qkv=True) == {}
    assert _skipped(model, "unpool") == [_NONE, _NONE]
    assert _skipped(model, "qkv") == [_NONE, _NONE]


@pytest.mark.parametrize("rows", range(1, 9))
def test_the_dispatcher_forks_no_prefetch_at_a_skipped_row_count(rows, monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.dense_gemms import (  # noqa: E501
        dispatcher as dense_gemms,
    )

    plan = dense_gemms.DenseGemmPlan(dense_gemms.default_kernels(), True, True)
    forks = []
    monkeypatch.setattr(plan, "fork_prefetch", lambda weight: forks.append(weight))
    monkeypatch.setattr(dense_gemms, "dense_gemm", lambda x, w, kernel, pdl: x @ w.t())
    site = dense_gemms._Site(plan, "unpool", 0, (4, 6), lambda: "next qkv weight")
    dispatcher = dense_gemms.DenseGemmDispatcher(site)
    monkeypatch.setattr(dispatcher, "fits", lambda x, w, b: True)
    x, weight = torch.ones(rows, 4), torch.ones(6, 4)
    assert site.skip_prefetch_rows == frozenset()
    dispatcher.apply(None, x, weight, None)
    site.skip_prefetch_rows = _SPLIT_K_ROWS
    dispatcher.apply(None, x, weight, None)
    assert forks == ["next qkv weight"] * (1 if rows in _SPLIT_K_ROWS else 2)


def test_the_log_line_counts_the_pdl_edges(monkeypatch):
    lines = []
    monkeypatch.setattr(pdl_triggers.logger, "info", lines.append)
    captured = [1, 16, 30, 44, 59]
    converted = [i for i in range(61) if i not in captured]
    pdl_triggers.log_installed(62, converted, captured, True, {"qkv": (56, frozenset(range(3, 9)))})
    assert lines == [
        "[musespark_open] boundary_early_release: early dependent launches (fused boundary -> QKV on 56 of 61 layer "
        "edges "
        "(trigger entry); DFlash capture layers [1, 16, 30, 44, 59] keep the exit trigger and "
        "the QKV weight prefetch (a plain capture copy precedes their QKV); PDL norm-gate -> o_proj "
        "on 62 layers (trigger after_wait); QKV weight prefetch dropped on converted edges: qkv x56 "
        "at rows 3-8)"
    ]
    assert pdl_triggers._spans(frozenset({1, 3, 4, 5, 8})) == "1,3-5,8"
