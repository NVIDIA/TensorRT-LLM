# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tep8 target's one-chain MoE prologue (``moe_prologue``): the installer's guards.

Host-only. The kernels and the installed MoE are the 8-rank GPU tests'
(``test_modeling_v2_moe_prologue.py`` and the op tests); what the prologue
builds on is ``test_modeling_v2_musespark_open_router_fork.py``'s.
"""

from __future__ import annotations

import types

import pytest

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.moe_prologue import (  # noqa: E501
    prologue as moe_prologue,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.projection_gathers import (  # noqa: E501
    gathers as projection_gathers,
)

pytestmark = pytest.mark.cpu_only


def test_install_refuses_a_moe_that_is_not_the_gather_moe():
    layer = types.SimpleNamespace(feed_forward=object(), layer_idx=7)
    with pytest.raises(ValueError, match="layer 7 has object"):
        moe_prologue.install(types.SimpleNamespace(layers=[layer]))


def test_install_keeps_every_layer_without_a_pool_inbox(monkeypatch):
    moe = projection_gathers.MuseSparkOpenTepGatherMoE.__new__(
        projection_gathers.MuseSparkOpenTepGatherMoE
    )
    moe.__dict__["_pool_inbox"] = None
    layer = types.SimpleNamespace(feed_forward=moe, layer_idx=0)
    warnings = []
    monkeypatch.setattr(moe_prologue.logger, "warning", warnings.append)
    assert moe_prologue.install(types.SimpleNamespace(layers=[layer])) == 0
    assert type(moe) is projection_gathers.MuseSparkOpenTepGatherMoE
    assert "not fused" in warnings[0]


class _Group:
    """Stands in for ``Distributed``: this rank plus ``peers`` (what the other ranks send)."""

    def __init__(self, peers=()):
        self.peers = list(peers)
        self.gathered = []

    def get(self, mapping):
        return self

    def tp_allgather(self, obj, small_payload=False):
        self.gathered.append(obj)
        return [obj] + self.peers


def _fused_layer():
    moe = projection_gathers.MuseSparkOpenTepGatherMoE.__new__(
        projection_gathers.MuseSparkOpenTepGatherMoE
    )
    moe.__dict__["_pool_inbox"] = object()
    return moe, types.SimpleNamespace(feed_forward=moe, layer_idx=0, mapping=object())


@pytest.fixture
def no_inbox(monkeypatch):
    """No inbox may be built: every test below refuses or keeps the path before it."""
    monkeypatch.setattr(
        moe_prologue, "ProjectionInbox", lambda *a, **k: pytest.fail("built the inbox")
    )
    monkeypatch.setattr(moe_prologue._route, "available", lambda: True)


def test_install_raises_on_every_rank_when_a_peer_cannot_build_the_kernel(monkeypatch, no_inbox):
    moe, layer = _fused_layer()
    group = _Group([True] * 6 + [False])
    monkeypatch.setattr(moe_prologue, "Distributed", group)
    with pytest.raises(ValueError, match="RMSNormFP4QuantKernel on every TP rank"):
        moe_prologue.install(types.SimpleNamespace(layers=[layer]))
    assert type(moe) is projection_gathers.MuseSparkOpenTepGatherMoE
    assert group.gathered == [True]
