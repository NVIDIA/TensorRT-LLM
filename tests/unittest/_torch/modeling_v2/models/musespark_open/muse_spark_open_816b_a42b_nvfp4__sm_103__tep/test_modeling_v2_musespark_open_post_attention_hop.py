# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How the fused post-attention hop (``post_attention_hop``) is wired into the tep8 target.

Host-only. The op itself is checked bit for bit against the built-in pair by its
multi-GPU test (``allreduce/test_modeling_v2_post_attn_hop.py``); here: the
selector, which module adoption re-classes (each layer's ``add_attn`` and
nothing else) and what it refuses, that it composes with a feature that
re-classes the decoder layer, which decode steps arm the joins (and that the
arming never outlives the forward), what the armed join hands the op or the
built-in fallback, that its choice never depends on a rank-local layout, and
that the group's workspace is built once. Also how the stale-only polling reaches the
hop's workspace (its stale-only polling schedule, from 8 rows).
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    layers as target_layers,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.allreduce import (  # noqa: E501
    post_attn_hop as hop,
)
from tensorrt_llm._torch.models.modeling_musespark_open import (
    MuseSparkOpenDecoderLayer,
    MuseSparkOpenGatedResidual,
    MuseSparkOpenModel,
)
from tensorrt_llm.functional import AllReduceStrategy

pytestmark = pytest.mark.cpu_only

_CORE = modeling.MuseSparkOpenTepModelingV2
_JOIN = target_layers.MuseSparkOpenTepPostAttnJoin


# --- stand-ins ----------------------------------------------------------------------------


class _AllReduce(nn.Module):
    """A module, like the built-in ``AllReduce``: the join must not register it as its child."""

    def __init__(self, strategy=AllReduceStrategy.MNNVL, mnnvl=True, symm=False):
        super().__init__()
        self.strategy = strategy
        self.mnnvl_allreduce = object() if mnnvl else None
        self.symm_mem_allreduce = object() if symm else None
        self.calls = []

    def forward(self, tensor):
        self.calls.append(tensor)
        return tensor * 8


class _Norm:
    def __init__(self, eps, gemma=True, width=hop.HIDDEN, dtype=torch.bfloat16):
        self.variance_epsilon = eps
        self.use_gemma = gemma
        self.weight = torch.zeros(width, dtype=dtype)


def _builtin_layer(index: int = 0, cls=MuseSparkOpenDecoderLayer, **overrides):
    """A built-in decoder layer as far as class and the hop's modules go."""
    layer = cls.__new__(cls)
    nn.Module.__init__(layer)
    layer.layer_idx = index
    layer.mapping = "mapping"
    o_proj = nn.Module()
    o_proj.all_reduce = overrides.get("all_reduce", _AllReduce())
    o_proj.use_fused_gemm_allreduce = overrides.get("fused_gemm_allreduce", False)
    attn = nn.Module()
    attn.o_proj = o_proj
    layer.self_attn = attn
    layer.add_attn = MuseSparkOpenGatedResidual(hop.HIDDEN, torch.bfloat16)
    layer.add_attn._fused_enabled = overrides.get("fused_residual", True)
    layer.add_ffn = MuseSparkOpenGatedResidual(hop.HIDDEN, torch.bfloat16)
    layer.pre_ffn_norm = overrides.get("pre_ffn_norm", _Norm(1e-5))
    layer.post_attn_norm = _Norm(1e-8)
    return layer


def _model(layers) -> MuseSparkOpenModel:
    model = MuseSparkOpenModel.__new__(MuseSparkOpenModel)
    nn.Module.__init__(model)
    model.layers = nn.ModuleList(layers)
    return model


@pytest.fixture
def no_fabric(monkeypatch):
    """``shared_workspace`` without a GPU: records what it was asked to build."""
    built = []

    class _Workspace:
        multicast = True

    def shared_workspace(mapping, *, variants=(), early_trigger=False, stale_poll=False):
        built.append((mapping, list(variants), early_trigger, stale_poll))
        return _Workspace()

    monkeypatch.setattr(hop, "HAS_CUTE_DSL", True)
    monkeypatch.setattr(hop, "shared_workspace", shared_workspace)
    return built


# --- adoption -----------------------------------------------------------------------------


def test_only_the_attention_side_join_is_reclassed(no_fabric):
    layers = [_builtin_layer(i) for i in range(3)]
    # Every path, duplicates included: a module registered twice shows up twice.
    trees = [[n for n, _ in layer.named_modules(remove_duplicate=False)] for layer in layers]
    call = target_layers._build_post_attn_hop(_model(layers))
    assert isinstance(call, target_layers.PostAttnHopCall) and call.op is hop.post_attn_hop
    assert call.joins == tuple(layer.add_attn for layer in layers)
    for layer, tree in zip(layers, trees):
        assert type(layer) is MuseSparkOpenDecoderLayer
        assert type(layer.add_attn) is _JOIN and layer.add_attn._armed is None
        assert layer.add_attn._o_proj_all_reduce is layer.self_attn.o_proj.all_reduce
        assert type(layer.add_ffn) is MuseSparkOpenGatedResidual
        assert [n for n, _ in layer.named_modules(remove_duplicate=False)] == tree
    # One workspace for the group, with every layer's norm variant.
    assert no_fabric == [("mapping", [(1e-8, 1e-5, False)], False, False)]
    target_layers._build_post_attn_hop(_model([_builtin_layer()]), early_trigger=True)
    assert no_fabric[-1] == ("mapping", [(1e-8, 1e-5, False)], True, False)
    target_layers._build_post_attn_hop(_model([_builtin_layer()]), stale_poll=True)
    assert no_fabric[-1] == ("mapping", [(1e-8, 1e-5, False)], False, True)


@pytest.mark.parametrize(
    "overrides, reason",
    [
        (dict(all_reduce=None), "does not all-reduce"),
        (dict(all_reduce=_AllReduce(AllReduceStrategy.NCCL)), "not the MNNVL one-shot"),
        (dict(all_reduce=_AllReduce(mnnvl=False)), "not the MNNVL one-shot"),
        (dict(all_reduce=_AllReduce(symm=True)), "not the MNNVL one-shot"),
        (dict(fused_gemm_allreduce=True), "fuses its all-reduce into the GEMM"),
        (dict(fused_residual=False), "MUSESPARK_OPEN_FUSED_RESIDUAL"),
        (dict(pre_ffn_norm=_Norm(1e-5, gemma=False)), "not Gemma-style"),
        (dict(pre_ffn_norm=_Norm(1e-5, width=4096)), "width 8192"),
        (dict(pre_ffn_norm=_Norm(1e-5, dtype=torch.float32)), "width 8192"),
    ],
    ids=[
        "no-allreduce",
        "nccl",
        "mnnvl-off",
        "symm-mem",
        "gemm-allreduce",
        "unfused-join",
        "non-gemma",
        "narrow",
        "fp32-weight",
    ],
)
def test_adoption_refuses_a_hop_it_does_not_reproduce(no_fabric, overrides, reason):
    layers = [_builtin_layer(0), _builtin_layer(1, **overrides)]
    with pytest.raises(ValueError, match=f"layer 1: .*{reason}"):
        target_layers._build_post_attn_hop(_model(layers))
    # All or nothing: nothing was built and no join was re-classed.
    assert no_fabric == []
    assert all(type(layer.add_attn) is MuseSparkOpenGatedResidual for layer in layers)


def test_adoption_refuses_a_join_another_feature_already_reclassed(no_fabric):
    class _OtherJoin(MuseSparkOpenGatedResidual):
        pass

    layer = _builtin_layer()
    layer.add_attn.__class__ = _OtherJoin
    with pytest.raises(ValueError, match="add_attn is not the built-in"):
        target_layers._build_post_attn_hop(_model([layer]))
    with pytest.raises(TypeError, match="MuseSparkOpenGatedResidual"):
        _JOIN.adopt(layer.add_attn, layer.self_attn.o_proj.all_reduce)
    with pytest.raises(TypeError, match="MuseSparkOpenDecoderLayer"):
        target_layers.post_attn_hop_refusals(nn.Module())


# --- composition with a feature that re-classes the decoder layer ----------------------


class _OtherLaneLayer(MuseSparkOpenDecoderLayer):
    """What the projection gathers do to every layer: an exact-type guard, then a re-class."""

    @classmethod
    def install(cls, layer):
        if type(layer) is not MuseSparkOpenDecoderLayer:
            raise ValueError(f"re-classes the built-in layer, got {type(layer).__name__}")
        if type(layer.add_ffn) is not MuseSparkOpenGatedResidual:
            raise ValueError("re-classes the built-in add_ffn")
        layer.__class__ = cls


@pytest.mark.parametrize("hop_first", [True, False], ids=["hop-then-other", "other-then-hop"])
def test_hop_composes_with_a_feature_that_reclasses_the_layer(no_fabric, hop_first):
    layers = [_builtin_layer(i) for i in range(2)]
    if hop_first:
        target_layers._build_post_attn_hop(_model(layers))
        for layer in layers:
            _OtherLaneLayer.install(layer)
    else:
        for layer in layers:
            _OtherLaneLayer.install(layer)
        target_layers._build_post_attn_hop(_model(layers))
    for layer in layers:
        assert type(layer) is _OtherLaneLayer
        assert type(layer.add_attn) is _JOIN


# --- the armed join -----------------------------------------------------------------------


class _Call:
    def __init__(self):
        self.workspace = object()
        self.calls = []

        def op(*args):
            self.calls.append(args)
            return "fused"

        self.op = op


@pytest.fixture
def recorded_builtin_join(monkeypatch):
    calls = []

    def join(self, residual, branch, eps, next_norm, gain=None):
        calls.append((residual, branch, eps, next_norm, gain))
        return "builtin"

    monkeypatch.setattr(MuseSparkOpenGatedResidual, "_add_after_rmsnorm_and_norm", join)
    return calls


def _armed_join(takes: bool, monkeypatch):
    layer = _builtin_layer()
    _JOIN.adopt(layer.add_attn, layer.self_attn.o_proj.all_reduce)
    monkeypatch.setattr(hop, "hop_takes", lambda *args: takes)
    call = _Call()
    layer.add_attn._armed = call
    return layer, call


def test_the_armed_join_hands_the_op_everything_it_needs(recorded_builtin_join, monkeypatch):
    layer, call = _armed_join(True, monkeypatch)
    join = layer.add_attn
    norm = layer.pre_ffn_norm
    # The join's own coefficients, by identity: recomputing them on the CPU and
    # comparing values would test torch's elementwise kernels, not this wiring.
    alpha, beta = torch.full((hop.HIDDEN,), 0.25), torch.full((hop.HIDDEN,), 0.75)
    monkeypatch.setattr(join, "coefficients", lambda: (alpha, beta))
    residual, partial = torch.zeros(8, hop.HIDDEN), torch.ones(8, hop.HIDDEN)
    assert join._add_after_rmsnorm_and_norm(residual, partial, 1e-8, norm) == "fused"
    ((args),) = call.calls
    assert args[0] is partial and args[1] is residual
    assert args[2] is alpha and args[3] is beta
    assert args[4] is norm.weight
    assert args[5:] == (1e-8, norm.variance_epsilon, join._preqkv_norm_fp32, call.workspace)
    assert recorded_builtin_join == []
    assert layer.self_attn.o_proj.all_reduce.calls == []


def test_a_layout_the_kernel_cannot_read_is_copied_not_declined(recorded_builtin_join, monkeypatch):
    """The fused-or-built-in choice ignores strides and addresses (rank-local facts)."""
    layer, call = _armed_join(True, monkeypatch)
    join = layer.add_attn
    wide = torch.arange(8 * 2 * hop.HIDDEN, dtype=torch.float32).reshape(8, 2 * hop.HIDDEN)
    partial = wide[:, ::2]  # not contiguous
    flat = torch.arange(hop.HIDDEN + 1, dtype=torch.float32)
    alpha = flat[1:]  # 4 bytes past an aligned start
    beta = torch.ones(hop.HIDDEN)
    monkeypatch.setattr(join, "coefficients", lambda: (alpha, beta))
    residual = torch.zeros(8, hop.HIDDEN)
    assert join._add_after_rmsnorm_and_norm(residual, partial, 1e-8, layer.pre_ffn_norm) == "fused"
    ((args),) = call.calls
    assert args[0] is not partial and args[0].is_contiguous() and torch.equal(args[0], partial)
    assert args[2] is not alpha and args[2].data_ptr() % 16 == 0 and torch.equal(args[2], alpha)
    assert args[1] is residual and args[3] is beta
    assert recorded_builtin_join == [] and layer.self_attn.o_proj.all_reduce.calls == []


def test_the_armed_join_falls_back_to_the_builtin_pair(recorded_builtin_join, monkeypatch):
    layer, call = _armed_join(False, monkeypatch)
    residual, partial = torch.zeros(9, 4), torch.ones(9, 4)
    norm = layer.pre_ffn_norm
    assert layer.add_attn._add_after_rmsnorm_and_norm(residual, partial, 1e-8, norm) == "builtin"
    assert call.calls == []
    (reduced,) = layer.self_attn.o_proj.all_reduce.calls
    assert reduced is partial
    ((res, branch, eps, next_norm, gain),) = recorded_builtin_join
    assert res is residual and torch.equal(branch, partial * 8)
    assert (eps, next_norm, gain) == (1e-8, norm, None)


def test_an_unarmed_join_is_the_builtin_join(recorded_builtin_join):
    layer = _builtin_layer()
    _JOIN.adopt(layer.add_attn, layer.self_attn.o_proj.all_reduce)
    residual, branch = torch.zeros(8, 4), torch.ones(8, 4)
    norm = layer.pre_ffn_norm
    assert layer.add_attn._add_after_rmsnorm_and_norm(residual, branch, 1e-8, norm) == "builtin"
    assert recorded_builtin_join == [(residual, branch, 1e-8, norm, None)]


# --- the core and its decode target --------------------------------------------------------


class _Layer(nn.Module):
    """A decoder layer stand-in that records its call and whether its join was armed."""

    def __init__(self, index, calls):
        super().__init__()
        self.index = index
        self.pre_qkv_proj_norm = nn.Identity()
        self.add_attn = nn.Module()
        self.add_attn._armed = None
        self.calls = calls

    def forward(self, position_ids, hidden_states, attn_metadata, precomputed_attn_input=None,
                next_layer_norm=None, sp_num_tokens=None, **kwargs):  # fmt: skip
        self.calls.append(dict(kwargs, armed=self.add_attn._armed))
        if kwargs.get("boom"):
            raise RuntimeError("boom")
        if next_layer_norm is None:
            return hidden_states
        return hidden_states, hidden_states.to(torch.bfloat16)


class _Metadata:
    def __init__(self, num_contexts):
        self.num_contexts = num_contexts


def _text_model(calls) -> MuseSparkOpenModel:
    model = MuseSparkOpenModel.__new__(MuseSparkOpenModel)
    nn.Module.__init__(model)
    model.embed_tokens = nn.Embedding(16, 4)
    model.embed_norm = nn.Identity()
    model.norm = nn.Identity()
    model.layers = nn.ModuleList([_Layer(i, calls) for i in range(3)])
    model.dtype = torch.bfloat16
    model.m_emb = 1.0
    model._prefill_sp_enabled = False
    model._sp_mapping = None
    return model


def _hop_core(monkeypatch, calls):
    """The core adopted with the post-attention hop, arming the stand-in layers' joins."""
    built = []

    def build(model, early_trigger=False, stale_poll=False):
        built.append(model)
        build.early_trigger = early_trigger
        build.stale_poll = stale_poll
        return target_layers.PostAttnHopCall(object(), [layer.add_attn for layer in model.layers])

    monkeypatch.setattr(target_layers, "_build_post_attn_hop", build)

    def builtin_forward(self, md, input_ids=None, position_ids=None, inputs_embeds=None, **kw):
        return self.layers[0](position_ids, torch.zeros(1, 4), md, **kw)

    monkeypatch.setattr(MuseSparkOpenModel, "forward", builtin_forward)
    model = _text_model(calls)
    core = _CORE.adopt(model, dispatch.plan_of("post_attention_hop"))
    assert built == [model]
    # The hop triggers its dependents early and polls stale-only.
    assert build.early_trigger is True
    assert build.stale_poll is True
    return core


def test_without_hop_nothing_is_adopted_and_no_call_is_armed():
    calls = []
    core = _CORE.adopt(_text_model(calls))
    assert core._plan == dispatch.NO_PATHS and core._post_attn_hop is None
    assert core._decode.post_attn_hop is None
    core(_Metadata(0), torch.tensor([1, 2]), torch.tensor([0, 1]))
    assert len(calls) == 3
    assert all(c == dict(armed=None) for c in calls)


@pytest.mark.parametrize("rows", [1, 4, 8])
def test_a_decode_step_of_at_most_8_rows_arms_every_join_for_the_forward(monkeypatch, rows):
    calls = []
    core = _hop_core(monkeypatch, calls)
    hop_call = core._decode.post_attn_hop
    core(_Metadata(0), torch.arange(rows) % 16, torch.arange(rows), spec_metadata="s")
    assert len(calls) == 3
    for seen in calls:
        assert seen == dict(
            all_reduce_params=target_layers._NO_ALLREDUCE, spec_metadata="s", armed=hop_call
        )
    assert not target_layers._NO_ALLREDUCE.enable_allreduce
    assert all(layer.add_attn._armed is None for layer in core.layers)


def test_a_decode_step_of_more_rows_is_the_builtin_path(monkeypatch):
    calls = []
    core = _hop_core(monkeypatch, calls)
    core(_Metadata(0), torch.arange(9) % 16, torch.arange(9), spec_metadata="s")
    assert calls == [dict(spec_metadata="s", armed=None)] * 3


def test_a_prefill_step_is_never_armed(monkeypatch):
    calls = []
    core = _hop_core(monkeypatch, calls)
    core(_Metadata(1), torch.tensor([1, 2]), torch.tensor([0, 1]))
    assert calls == [dict(armed=None)]


def test_the_arming_is_dropped_when_a_layer_raises(monkeypatch):
    calls = []
    core = _hop_core(monkeypatch, calls)
    with pytest.raises(RuntimeError, match="boom"):
        core(_Metadata(0), torch.tensor([1, 2]), torch.tensor([0, 1]), boom=True)
    assert calls[0]["armed"] is core._decode.post_attn_hop
    assert all(layer.add_attn._armed is None for layer in core.layers)


# --- the group's workspace ------------------------------------------------------------------


def test_a_second_request_for_the_workspace_reuses_the_first(monkeypatch):
    """A constructor run twice (the loader's meta-init retry) must not allocate twice."""
    created, extended = [], []

    class _Workspace:
        def __init__(self, mapping, *, multicast, variants, early_trigger, stale_poll):
            created.append((mapping, multicast, list(variants), early_trigger, stale_poll))

        def build_variants(self, variants):
            extended.append(list(variants))

    monkeypatch.setattr(hop, "PostAttnHopWorkspace", _Workspace)
    monkeypatch.setattr(hop, "_SHARED", {})
    first = hop.shared_workspace("mapping", variants=[(1e-8, 1e-5, False)])
    again = hop.shared_workspace("mapping", variants=[(1e-8, 1e-5, True)])
    other = hop.shared_workspace("mapping", multicast=False)
    stale = hop.shared_workspace("mapping", stale_poll=True)
    assert again is first and other is not first and stale not in (first, other)
    assert created == [
        ("mapping", True, [(1e-8, 1e-5, False)], False, False),
        ("mapping", False, [], False, False),
        ("mapping", True, [], False, True),
    ]
    assert extended == [[(1e-8, 1e-5, True)]]


# --- the stale-only polling schedule ---------------------------------------------------------


def test_stale_poll_polls_stale_only_on_steps_of_8_rows():
    assert hop.STALE_POLL_MIN_ROWS == hop.MAX_ROWS == 8
    assert [hop.uses_stale_poll(True, rows) for rows in range(1, 9)] == [False] * 7 + [True]
    assert not any(hop.uses_stale_poll(False, rows) for rows in range(1, 9))
    assert all(hop.uses_stale_poll(True, rows, min_rows=1) for rows in range(1, 9))
