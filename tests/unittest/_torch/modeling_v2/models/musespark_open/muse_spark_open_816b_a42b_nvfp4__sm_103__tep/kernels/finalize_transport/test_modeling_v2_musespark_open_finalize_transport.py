# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""How ``finalize_transport`` (the finalize all-reduce transport) is wired into the tep8 target.

Host-only. The op itself is checked bit for bit against the built-in kernel by
its multi-GPU test (``moe/test_modeling_v2_finalize_allreduce_transport.py``);
here: what ``install`` refuses (agreed over the TP group, so that a reason one
rank sees makes every rank refuse before the workspace's collective construction)
and what it replaces (every layer's finalize attribute, nothing else), that
adoption calls it only when planned, and which calls the workspace routes to the
built-in kernel.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.finalize_transport import (  # noqa: E501
    transport as finalize_transport,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe import (  # noqa: E501
    finalize_allreduce_transport as fin,
)
from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenModel

pytestmark = pytest.mark.cpu_only

_CORE = modeling.MuseSparkOpenTepModelingV2


@pytest.mark.parametrize(
    "tokens, takes",
    [(1, True), (8, True), (16, True), (17, False), (128, False), (0, False), (True, False),
     (8.0, False)],
)  # fmt: skip
def test_the_workspace_takes_1_to_16_tokens(tokens, takes):
    assert fin.FinalizeTransportWorkspace.takes(tokens) is takes


def test_the_feature_transports():
    assert fin.DEFAULT == fin.Transport("multicast", "red", "warp", "late")
    assert fin.ENTRY_TRIGGER == fin.Transport("multicast", "red", "warp", "entry")
    assert fin.ENTRY_TRIGGER.label() == "multicast/red/warp/entry"
    with pytest.raises(ValueError, match="publish"):
        fin.Transport(publish="broadcast")
    with pytest.raises(ValueError, match="reduction"):
        fin.Transport(reduction="tree")
    with pytest.raises(ValueError, match="trigger"):
        fin.Transport(trigger="exit")


def test_the_parity_step_covers_the_largest_grid():
    assert fin.GRID_STEP == 1 << fin.GRID_SHIFT
    assert fin.GRID_STEP >= fin.MAX_TOKENS * fin.CLUSTER
    assert (1 << 32) % (2 * fin.GRID_STEP) == 0


# --- stand-ins ----------------------------------------------------------------------------


class _Layer(nn.Module):
    def __init__(self, index, workspace):
        super().__init__()
        self.layer_idx = index
        self.mapping = "tp8-mapping"
        self.feed_forward = nn.Module()
        self.feed_forward._finalize_ar_workspace = workspace

    def forward(self, *args, **kwargs):  # pragma: no cover - never called here
        raise AssertionError


class _Legacy:
    _mapping = "tp8-mapping"


def _model(legacy=None, layers=3) -> MuseSparkOpenModel:
    model = MuseSparkOpenModel.__new__(MuseSparkOpenModel)
    nn.Module.__init__(model)
    legacy = _Legacy() if legacy is None else legacy
    model._finalize_ar_workspace = legacy
    model.layers = nn.ModuleList([_Layer(i, legacy) for i in range(layers)])
    model.embed_tokens = nn.Embedding(16, 4)
    model.embed_norm = nn.Identity()
    model.norm = nn.Identity()
    model.dtype = torch.bfloat16
    model.m_emb = 1.0
    model._prefill_sp_enabled = False
    model._sp_mapping = None
    return model


class _Group:
    """Stands in for ``Distributed``: this rank plus ``peers`` (what the other ranks send)."""

    def __init__(self, peers=()):
        self.peers = list(peers)
        self.gathered = []

    def get(self, mapping):
        assert mapping == "tp8-mapping"
        return self

    def tp_allgather(self, obj, small_payload=False):
        self.gathered.append(obj)
        return [obj] + self.peers


class _Built(list):
    """The workspaces asked for, and the stand-in TP group."""

    group: _Group


@pytest.fixture
def no_fabric(monkeypatch):
    """``FinalizeTransportWorkspace`` without a GPU: records what it was asked to build.

    The TP group is this rank alone (``built.group``; tests add peers)."""
    built = _Built()

    class _Workspace:
        def __init__(self, legacy, mapping, transport):
            built.append((legacy, mapping, transport))

    monkeypatch.setattr(fin, "HAS_CUTE_DSL", True)
    monkeypatch.setattr(fin, "FinalizeTransportWorkspace", _Workspace)
    group = _Group()
    monkeypatch.setattr(finalize_transport, "Distributed", group)
    built.group = group
    return built


# --- install ------------------------------------------------------------------------------


def test_install_replaces_every_layers_finalize_and_nothing_else(no_fabric):
    model = _model()
    legacy = model._finalize_ar_workspace
    classes = [type(layer.feed_forward) for layer in model.layers]
    workspace = finalize_transport.install(model)
    assert no_fabric == [(legacy, "tp8-mapping", fin.DEFAULT)]
    assert all(layer.feed_forward._finalize_ar_workspace is workspace for layer in model.layers)
    assert [type(layer.feed_forward) for layer in model.layers] == classes
    # The model keeps the built-in workspace, whose release it owns.
    assert model._finalize_ar_workspace is legacy


def test_install_refuses_a_model_without_the_fused_finalize(no_fabric):
    model = _model()
    model._finalize_ar_workspace = None
    with pytest.raises(ValueError, match="MUSESPARK_OPEN_FUSED_FINALIZE_AR_MNNVL"):
        finalize_transport.install(model)
    assert no_fabric == []


def test_install_refuses_a_layer_on_another_workspace(no_fabric):
    model = _model()
    model.layers[1].feed_forward._finalize_ar_workspace = object()
    with pytest.raises(ValueError, match=r"layers \[1\]"):
        finalize_transport.install(model)
    assert no_fabric == []
    assert model.layers[0].feed_forward._finalize_ar_workspace is model._finalize_ar_workspace


def test_install_refuses_on_every_rank_what_one_peer_refuses(no_fabric):
    # This rank could install; one peer cannot build the kernel: no rank builds the
    # workspace (whose construction is collective), every rank raises the peer's reason.
    peer = "[musespark_open] finalize_transport cannot run: the CuTe DSL or the built-in finalize module is unavailable"
    no_fabric.group.peers = [None] * 6 + [peer]
    model = _model()
    legacy = model._finalize_ar_workspace
    with pytest.raises(ValueError, match=r"is unavailable \(on TP ranks \[7\] of 8\)"):
        finalize_transport.install(model)
    assert no_fabric == []
    assert no_fabric.group.gathered == [None]
    assert all(layer.feed_forward._finalize_ar_workspace is legacy for layer in model.layers)


def test_install_names_this_ranks_reason_and_the_ranks_that_agree(no_fabric, monkeypatch):
    monkeypatch.setattr(fin, "HAS_CUTE_DSL", False)
    reason = (
        "[musespark_open] finalize_transport cannot run: the CuTe DSL or the built-in finalize "
        "module is unavailable"
    )
    # Every rank refuses the same way: the reason alone.
    no_fabric.group.peers = [reason] * 7
    with pytest.raises(ValueError) as refused:
        finalize_transport.install(_model())
    assert str(refused.value) == reason
    # Only this rank refuses: its peers raise too, told which rank refused.
    no_fabric.group.peers = [None] * 7
    with pytest.raises(ValueError, match=r"is unavailable \(on TP ranks \[0\] of 8\)"):
        finalize_transport.install(_model())
    assert no_fabric == []


# --- adoption -----------------------------------------------------------------------------


def test_adoption_installs_when_planned(monkeypatch):
    calls = []

    def install(model):
        calls.append(model)
        return "workspace"

    monkeypatch.setattr(finalize_transport, "install", install)
    model = _model()
    core = _CORE.adopt(model, dispatch.plan_of("finalize_transport"))
    assert calls == [model]
    assert core._finalize_transport == "workspace"


def test_without_finalize_transport_nothing_is_installed(monkeypatch):
    monkeypatch.setattr(
        finalize_transport,
        "install",
        lambda *a, **k: pytest.fail("installed without finalize_transport"),
    )
    model = _model()
    legacy = model._finalize_ar_workspace
    core = _CORE.adopt(model)
    assert core._finalize_transport is None
    assert all(layer.feed_forward._finalize_ar_workspace is legacy for layer in core.layers)
