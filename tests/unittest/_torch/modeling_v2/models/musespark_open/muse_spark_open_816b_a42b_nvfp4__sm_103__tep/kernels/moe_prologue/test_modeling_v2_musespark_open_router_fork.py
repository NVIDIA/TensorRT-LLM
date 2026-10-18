# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tep8 target's router fork (``router_fork``, part of ``moe_prologue``): its plan and the installer.

Host-only, with a stand-in for ``Distributed``. The forked prologue itself is the
8-rank GPU tests' (``test_modeling_v2_moe_prologue.py``, the composition tests) and
the side kernel the op test's (``moe/test_modeling_v2_router_shard.py``).
"""

from __future__ import annotations

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch as tp_agreement,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.context_kv_fork import (  # noqa: E501
    fork as context_kv_fork,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.moe_prologue import (  # noqa: E501
    prologue as moe_prologue,
)

pytestmark = pytest.mark.cpu_only


def test_the_prologue_builds_on_the_gathers_and_the_early_hop():
    """The fork rides the one-chain prologue, which reads the gathers' pool and the
    hop's early trigger."""
    assert dispatch.closure({"moe_prologue"}) == {
        "moe_prologue",
        "projection_gathers",
        "post_attention_hop",
    }
    with pytest.raises(ValueError, match="builds on"):
        dispatch.Plan(frozenset({"moe_prologue", "projection_gathers"}))


class _Group:
    """Stands in for ``Distributed``: this rank (TP rank 0) first, then what ``peers`` send."""

    def __init__(self, peers=()):
        self.peers = list(peers)
        self.gathered = []

    def get(self, mapping):
        return self

    def tp_allgather(self, obj, small_payload=False):
        assert small_payload
        self.gathered.append(obj)
        return [obj] + self.peers


_TP8 = types.SimpleNamespace(tp_size=8)


def _prologue_layers(count=3):
    layers = []
    for index in range(count):
        cls = moe_prologue.MuseSparkOpenTepPrologueMoE
        moe = cls.__new__(cls)
        torch.nn.Module.__init__(moe)
        layers.append(types.SimpleNamespace(feed_forward=moe, layer_idx=index, mapping=_TP8))
    return types.SimpleNamespace(layers=layers), layers


@pytest.fixture
def side(monkeypatch):
    """A stand-in side stream; counts the streams created."""
    made = []

    def make():
        made.append(object())
        return made[-1]

    monkeypatch.setattr(moe_prologue, "dedicated_stream", make)
    return made


def test_a_layer_without_router_fork_keeps_the_one_chain_order():
    _, layers = _prologue_layers(1)
    assert layers[0].feed_forward._router_stream is None


def test_install_gives_every_layer_the_one_side_stream(monkeypatch, side):
    group = _Group([None] * 7)
    monkeypatch.setattr(tp_agreement, "Distributed", group)
    model, layers = _prologue_layers()
    infos = []
    monkeypatch.setattr(moe_prologue.logger, "info", infos.append)
    assert moe_prologue.install_router_fork(model) == 3
    assert len(side) == 1
    assert all(layer.feed_forward._router_stream is side[0] for layer in layers)
    assert group.gathered == [None]
    assert "router_fork" in infos[0] and "plain launch" in infos[0]
    # The stream is set on every layer; the fork itself runs only where the routed
    # prologue runs (not on the layers whose expert input the pool kernel does not quantize).
    assert "side stream set on 3 prologue MoE layers" in infos[0]
    assert "all 3 layers" not in infos[0]


def test_the_side_stream_is_the_targets_one_dedicated_stream_helper():
    assert moe_prologue.dedicated_stream is context_kv_fork.dedicated_stream


def test_install_refuses_here_when_a_peer_refuses(monkeypatch, side):
    peer = "[musespark_open] router_fork could not create its side stream: CUDA ERROR"
    group = _Group([None] * 6 + [peer])
    monkeypatch.setattr(tp_agreement, "Distributed", group)
    model, layers = _prologue_layers()
    with pytest.raises(ValueError, match=r"side stream: CUDA ERROR \(on TP ranks \[7\] of 8\)"):
        moe_prologue.install_router_fork(model)
    assert group.gathered == [None]
    assert all(layer.feed_forward._router_stream is None for layer in layers)


def test_install_agrees_a_failed_stream_creation(monkeypatch):
    def fail():
        raise RuntimeError("CUDA ERROR: cudaErrorUnknown")

    monkeypatch.setattr(moe_prologue, "dedicated_stream", fail)
    group = _Group(
        [
            "[musespark_open] router_fork could not create its side stream: CUDA ERROR: cudaErrorUnknown"
        ]
        * 7
    )
    monkeypatch.setattr(tp_agreement, "Distributed", group)
    model, _ = _prologue_layers()
    with pytest.raises(
        ValueError, match=r"^\[musespark_open\] router_fork could not create its side stream"
    ):
        moe_prologue.install_router_fork(model)


def test_install_refuses_a_layer_without_the_prologue(monkeypatch, side):
    group = _Group([None] * 7)
    monkeypatch.setattr(tp_agreement, "Distributed", group)
    model, layers = _prologue_layers()
    layers[1].feed_forward = types.SimpleNamespace()
    with pytest.raises(ValueError, match=r"layers \[1\] do not run the prologue"):
        moe_prologue.install_router_fork(model)
    assert side == []
