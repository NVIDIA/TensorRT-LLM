# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How the tep8 target's target-side fast paths compose on one model, host-only.

Each fast path re-classes built-in modules of every layer in place and refuses
a module it does not expect, so planning several of them is only valid if each
sees the modules it expects, whatever the others did first. On a stand-in text
model whose layers carry built-in decoder-layer, MoE and gated-residual objects
created without their constructors (no experts, workspaces or streams; the
fabric inboxes of the projection gathers and the workspace of the post-attention hop
replaced by recorders, the gathers' TP agreement on the fused sites by a group of one
rank), this checks which class each module ends up with for each plan, that the hop and
the gathers adopt in either order, and that ``DecodeTarget`` hands every layer the
keywords of the pre-attention prep, the hop and the gathers together. What the kernels
compute, alone and together, is the GPU tests' business (the combined decoder-layer test
runs them on real weights).
"""

from __future__ import annotations

from types import SimpleNamespace

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
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.projection_gathers import (  # noqa: E501
    gathers as projection_gathers,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.allreduce import (  # noqa: E501
    post_attn_hop as _hop,
)
from tensorrt_llm._torch.models.modeling_musespark_open import (
    MuseSparkOpenDecoderLayer,
    MuseSparkOpenGatedResidual,
    MuseSparkOpenModel,
    MuseSparkOpenMoE,
)
from tensorrt_llm.functional import AllReduceStrategy

pytestmark = pytest.mark.cpu_only

_CORE = modeling.MuseSparkOpenTepModelingV2
_SHELL = modeling.ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep
_GATHER_MOE = projection_gathers.MuseSparkOpenTepGatherMoE


class _Inbox:
    """Stands in for ``ProjectionInbox`` (fabric memory, collective over the TP group)."""

    built = []

    def __init__(self, mapping, row_words):
        self.mapping = mapping
        self.row_words = row_words
        _Inbox.built.append(row_words)


class _OneRankGroup:
    """Stands in for ``Distributed`` in ``fusable_sites``: a TP group of this rank alone."""

    @staticmethod
    def get(mapping):
        return _OneRankGroup()

    def tp_allgather(self, obj, small_payload=False):
        return [obj]


def _builtin(cls):
    module = cls.__new__(cls)
    nn.Module.__init__(module)
    return module


def _builtin_layer(index: int) -> MuseSparkOpenDecoderLayer:
    layer = _builtin(MuseSparkOpenDecoderLayer)
    layer.layer_idx = index
    layer.mapping = SimpleNamespace(tp_size=8)
    moe = _builtin(MuseSparkOpenMoE)
    moe.layer_idx = index
    moe.num_experts = 256
    moe.hidden_size = 8192
    moe._pool_gather_workspace = object()
    moe._unpool_gather_workspace = object()
    layer.feed_forward = moe
    layer.add_attn = _builtin(MuseSparkOpenGatedResidual)
    layer.add_ffn = _builtin(MuseSparkOpenGatedResidual)
    return layer


def _builtin_text_model(num_layers: int = 3) -> MuseSparkOpenModel:
    model = _builtin(MuseSparkOpenModel)
    model.layers = nn.ModuleList([_builtin_layer(i) for i in range(num_layers)])
    return model


@pytest.fixture
def adopt(monkeypatch):
    """``adopt(paths)``: the core's adoption of a fresh stand-in under the plan of ``paths``."""
    monkeypatch.setattr(projection_gathers, "ProjectionInbox", _Inbox)
    monkeypatch.setattr(projection_gathers._unpool, "available", lambda: True)
    monkeypatch.setattr(projection_gathers._pool, "available", lambda: True)
    # The site agreement runs over one rank.
    monkeypatch.setattr(projection_gathers, "Distributed", _OneRankGroup)
    _Inbox.built.clear()

    def _adopt(paths, model=None):
        model = _builtin_text_model() if model is None else model
        return _CORE.adopt(model, dispatch.plan_of(*paths))

    return _adopt


# --------------------------------------------------------------------------
# The projection gathers on the MoE
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "paths, moe_class, inboxes",
    [
        (("projection_gathers",), _GATHER_MOE, [512, 256]),
        ((), MuseSparkOpenMoE, []),
    ],
)
def test_each_plan_gives_every_moe_one_class(adopt, paths, moe_class, inboxes):
    model = _builtin_text_model()
    before = [layer.feed_forward for layer in model.layers]
    adopt(paths, model)
    gathers_on = "projection_gathers" in paths
    for layer, moe in zip(model.layers, before):
        assert layer.feed_forward is moe
        assert type(moe) is moe_class
        if gathers_on:
            assert type(layer) is projection_gathers.MuseSparkOpenTepDecoderLayer
            assert type(layer.add_ffn) is projection_gathers.MuseSparkOpenTepGatedResidual
            assert (moe._pool_inbox is not None) == gathers_on
        else:
            assert type(layer) is MuseSparkOpenDecoderLayer
            assert type(layer.add_ffn) is MuseSparkOpenGatedResidual
        assert type(layer.add_attn) is MuseSparkOpenGatedResidual
    # One inbox per fused site for the whole model (unpool rows 512 words, pool 256).
    assert _Inbox.built == inboxes
    expected_kwargs = {"fused_unpool_gather": True, "fused_pool_gather": True} if gathers_on else {}
    assert model._decode.layer_kwargs == expected_kwargs


def test_gathers_refuses_a_moe_another_feature_re_classed():
    class OtherMoE(MuseSparkOpenMoE):
        pass

    model = _builtin_text_model()
    model.layers[1].feed_forward.__class__ = OtherMoE
    with pytest.raises(ValueError, match="got OtherMoE in layer 1"):
        projection_gathers.install(model, unpool=True, pool=True)


def test_the_boundary_release_installs_on_the_fused_sites(monkeypatch, adopt):
    """The stand-in layers have no attention for the early release's norm-gate class: its trigger is
    taken away, so only the boundary triggers move."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.boundary_early_release import (  # noqa: E501
        triggers as pdl_triggers,
    )

    monkeypatch.setattr(pdl_triggers, "NORM_GATE_TRIGGER", None)
    model = adopt(("boundary_early_release",))
    # The fused sites re-class every post-FFN join (default trigger: exit); the early release moves
    # the trigger of every one but the last.
    assert all(
        type(layer.add_ffn) is projection_gathers.MuseSparkOpenTepGatedResidual
        for layer in model.layers
    )
    triggers = [layer.add_ffn._boundary_trigger for layer in model.layers]
    assert triggers == ["entry", "entry", "exit"]
    assert model.layers[0].add_ffn._gather_inbox is not None


# --------------------------------------------------------------------------
# The post-attention hop with the projection gathers on the decoder layer
# --------------------------------------------------------------------------
class _AllReduce(nn.Module):
    """The o_proj's built-in MNNVL all-reduce, as far as ``post_attn_hop_refusals`` reads it."""

    def __init__(self):
        super().__init__()
        self.strategy = AllReduceStrategy.MNNVL
        self.mnnvl_allreduce = object()
        self.symm_mem_allreduce = None


class _Norm:
    def __init__(self, eps):
        self.variance_epsilon = eps
        self.use_gemma = True
        self.weight = torch.zeros(_hop.HIDDEN, dtype=torch.bfloat16)


def _hop_ready_layer(index: int) -> MuseSparkOpenDecoderLayer:
    """A stand-in layer the post-attention hop and the projection gathers both accept."""
    layer = _builtin_layer(index)
    o_proj = nn.Module()
    o_proj.all_reduce = _AllReduce()
    o_proj.use_fused_gemm_allreduce = False
    layer.self_attn = nn.Module()
    layer.self_attn.o_proj = o_proj
    layer.add_attn = MuseSparkOpenGatedResidual(_hop.HIDDEN, torch.bfloat16)
    layer.add_attn._fused_enabled = True
    layer.add_ffn = MuseSparkOpenGatedResidual(_hop.HIDDEN, torch.bfloat16)
    layer.pre_ffn_norm = _Norm(1e-5)
    layer.post_attn_norm = _Norm(1e-8)
    return layer


def _hop_ready_model(num_layers: int = 3) -> MuseSparkOpenModel:
    model = _builtin(MuseSparkOpenModel)
    model.layers = nn.ModuleList([_hop_ready_layer(i) for i in range(num_layers)])
    return model


@pytest.fixture
def no_fabric(monkeypatch, adopt):
    """``adopt`` with the hop's workspace replaced by a recorder too (one entry per build)."""
    built = []

    class _Workspace:
        multicast = True

    def shared_workspace(mapping, *, variants=(), early_trigger=False, stale_poll=False):
        built.append((early_trigger, stale_poll))
        return _Workspace()

    monkeypatch.setattr(_hop, "HAS_CUTE_DSL", True)
    monkeypatch.setattr(_hop, "shared_workspace", shared_workspace)
    return built


@pytest.mark.parametrize(
    "paths",
    [("post_attention_hop",), ("post_attention_hop", "projection_gathers")],
    ids=["hop", "hop-gathers"],
)
def test_hop_adopts_whatever_gathers_and_decode_target_did_to_the_layer(adopt, no_fabric, paths):
    """The core runs the decode target, the gathers, then the hop: the hop reads the
    (re-classed) layer and re-classes only ``add_attn``; the gathers re-classed the layer and
    ``add_ffn``, which the hop does not touch."""
    model = _hop_ready_model()
    joins = [layer.add_attn for layer in model.layers]
    ffn_joins = [layer.add_ffn for layer in model.layers]
    core = adopt(paths, model)
    gathers_on = "projection_gathers" in paths
    assert no_fabric == [(True, True)]
    assert core._decode.post_attn_hop.joins == tuple(joins)
    for layer, join, ffn_join in zip(model.layers, joins, ffn_joins):
        assert layer.add_attn is join and type(join) is target_layers.MuseSparkOpenTepPostAttnJoin
        assert layer.add_ffn is ffn_join
        assert type(ffn_join) is (
            projection_gathers.MuseSparkOpenTepGatedResidual
            if gathers_on
            else MuseSparkOpenGatedResidual
        )
        assert type(layer) is (
            projection_gathers.MuseSparkOpenTepDecoderLayer
            if gathers_on
            else MuseSparkOpenDecoderLayer
        )
        # The o_proj all-reduce stays a child of o_proj only.
        assert dict(join.named_children()) == {}
        assert join._o_proj_all_reduce is layer.self_attn.o_proj.all_reduce


def test_gathers_installs_on_layers_hop_already_adopted(adopt, no_fabric):
    """The other order (the hop first, then the gathers) composes too: they re-class different joins."""
    model = _hop_ready_model()
    target_layers._build_post_attn_hop(model)
    assert projection_gathers.install(model, unpool=True, pool=True) == {
        "fused_unpool_gather": True,
        "fused_pool_gather": True,
    }
    for layer in model.layers:
        assert type(layer) is projection_gathers.MuseSparkOpenTepDecoderLayer
        assert type(layer.add_attn) is target_layers.MuseSparkOpenTepPostAttnJoin
        assert type(layer.add_ffn) is projection_gathers.MuseSparkOpenTepGatedResidual
        assert type(layer.feed_forward) is _GATHER_MOE


def test_hop_still_refuses_a_hop_it_does_not_reproduce_on_a_gathers_layer(adopt, no_fabric):
    model = _hop_ready_model()
    model.layers[2].add_attn._fused_enabled = False
    with pytest.raises(ValueError, match="layer 2: MUSESPARK_OPEN_FUSED_RESIDUAL is off"):
        adopt(("post_attention_hop", "projection_gathers"), model)


# --------------------------------------------------------------------------
# DecodeTarget: every feature's layer keywords reach every layer
# --------------------------------------------------------------------------
class _RecordingLayer(nn.Module):
    """A decoder layer that records its keywords and whether the hop's joins are armed."""

    def __init__(self, record, joins, last):
        super().__init__()
        self.record = record
        self.joins = joins
        self.pre_qkv_proj_norm = None if last else "next norm"

    def forward(self, **kwargs):
        self.record.append((dict(kwargs), {join._armed is not None for join in self.joins}))
        if kwargs["next_layer_norm"] is None:
            return kwargs["hidden_states"]
        return kwargs["hidden_states"], None


def _recording_target(paths):
    record = []
    join = target_layers.MuseSparkOpenTepPostAttnJoin.__new__(
        target_layers.MuseSparkOpenTepPostAttnJoin
    )
    nn.Module.__init__(join)
    join._armed = None
    core = _builtin(MuseSparkOpenModel)
    core.layers = nn.ModuleList(
        [_RecordingLayer(record, [join], last) for last in (False, False, True)]
    )
    core.embed_tokens = lambda ids: torch.zeros(ids.shape[0], 4)
    core.embed_norm = lambda x: x
    core.m_emb = 1.0
    core.dtype = torch.float32
    core.norm = lambda x: x
    core._plan = dispatch.plan_of(*paths)
    gathers_on = "projection_gathers" in paths
    core._layer_kwargs = (
        {"fused_unpool_gather": True, "fused_pool_gather": True} if gathers_on else {}
    )
    core._post_attn_hop = (
        target_layers.PostAttnHopCall(object(), [join]) if "post_attention_hop" in paths else None
    )
    return modeling.DecodeTarget(core), record, join


@pytest.mark.parametrize(
    "paths",
    [
        (),
        ("preattn_prep",),
        ("post_attention_hop",),
        ("projection_gathers",),
        ("preattn_prep", "post_attention_hop", "projection_gathers"),
    ],
    ids=["none", "preattn_prep", "hop", "gathers", "prep-hop-gathers"],
)
@pytest.mark.parametrize("rows", [1, 8, 9])
def test_decode_layer_keywords_compose(paths, rows):
    """``preattn_prep`` (the pre-attention prep), ``fused_unpool_gather`` / ``fused_pool_gather``
    (the gathers) and, at 1..8 rows, ``all_reduce_params`` with the joins armed (the hop) reach
    every layer together."""
    target, record, join = _recording_target(paths)
    target.forward(None, input_ids=torch.zeros(rows, dtype=torch.int64), spec_metadata="spec")
    assert len(record) == 3
    hop = "post_attention_hop" in paths and rows <= 8
    for kwargs, armed in record:
        assert kwargs["spec_metadata"] == "spec" and kwargs["sp_num_tokens"] is None
        assert ("preattn_prep" in kwargs) == ("preattn_prep" in paths)
        assert kwargs.get("fused_unpool_gather", False) == ("projection_gathers" in paths)
        assert kwargs.get("fused_pool_gather", False) == ("projection_gathers" in paths)
        if hop:
            assert kwargs["all_reduce_params"] is target_layers._NO_ALLREDUCE
            assert armed == {True}
        else:
            assert "all_reduce_params" not in kwargs
            assert armed == {False}
    assert join._armed is None
