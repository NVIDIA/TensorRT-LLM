# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tep8 target's pool into route (``pool_into_route``): its plan, the installer, arming.

Host-only, with stand-ins for ``Distributed``, the transport workspace and the layers.
The transport and the routed kernel are the 8-rank op test's
(``comm/test_modeling_v2_pool_direct_route.py``), the model path the composition tests'.
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
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.dense_gemms import (  # noqa: E501
    dispatcher as dense_gemms,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.moe_prologue import (  # noqa: E501
    prologue as moe_prologue,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.pool_into_route import (  # noqa: E501
    route as pool_direct,
)

pytestmark = pytest.mark.cpu_only


def test_the_pool_into_route_builds_on_the_prologue_and_the_dense_gemms():
    """It publishes from the forked router shard and replaces the dense pool GEMM."""
    assert dispatch.closure({"pool_into_route"}) >= {"moe_prologue", "dense_gemms"}
    for missing in ("moe_prologue", "dense_gemms"):
        paths = dispatch.closure({"pool_into_route"}) - {missing}
        with pytest.raises(ValueError, match="builds on"):
            dispatch.Plan(paths)


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


class _Workspace:
    """Stands in for ``PoolDirectWorkspace``: records the variants built and the guard watch."""

    rank = 0

    def __init__(self, fail=None):
        self.built = []
        self.fail = fail
        self.watching = False

    def build_variants(self, variants):
        if self.fail is not None:
            raise self.fail
        self.built.extend(variants)

    def start_guard_watch(self):
        self.watching = True


def _dispatcher(role="pool"):
    site = types.SimpleNamespace(layer_idx=0, role=role)
    return dense_gemms.DenseGemmDispatcher(site)


def _layers(count=4, stream=True):
    layers = []
    for index in range(count):
        cls = moe_prologue.MuseSparkOpenTepPrologueMoE
        moe = cls.__new__(cls)
        torch.nn.Module.__init__(moe)
        moe.layer_idx = index
        moe._router_stream = object() if stream else None
        moe._prologue_enable_pdl = True
        pooler = dense_gemms.MuseSparkOpenTepDenseLinear.__new__(
            dense_gemms.MuseSparkOpenTepDenseLinear
        )
        torch.nn.Module.__init__(pooler)
        pooler.__dict__["_low_m_dispatcher"] = _dispatcher()
        moe.pooler = pooler
        layers.append(types.SimpleNamespace(feed_forward=moe, layer_idx=index, mapping=_TP8))
    return types.SimpleNamespace(layers=layers, _layer_kwargs={"fused_pool_gather": True}), layers


@pytest.fixture
def workspace(monkeypatch):
    made = []

    def shared(mapping):
        made.append(_Workspace())
        return made[-1]

    monkeypatch.setattr(pool_direct._pdr, "shared_workspace", shared)
    return made


def test_a_layer_without_pool_direct_is_never_armed():
    _, layers = _layers(1)
    assert layers[0].feed_forward._pool_direct_armed is None


def test_install_builds_the_workspace_after_the_agreement(monkeypatch, workspace):
    group = _Group([None] * 7)
    monkeypatch.setattr(tp_agreement, "Distributed", group)
    model, _ = _layers()
    direct = pool_direct.install(model, object(), mapping=_TP8)
    assert direct.workspace is workspace[0] and direct.sites == ()
    assert group.gathered == [None]


@pytest.mark.parametrize(
    "case, match",
    [
        ("plan", "select dense_gemms"),
        ("stream", "without the router fork's side stream"),
        ("pooler", "not the dense_gemms MuseSparkOpenTepDenseLinear"),
    ],
)
def test_install_refuses_on_every_rank(monkeypatch, workspace, case, match):
    group = _Group([None] * 7)
    monkeypatch.setattr(tp_agreement, "Distributed", group)
    model, layers = _layers(stream=case != "stream")
    if case == "pooler":
        layers[2].feed_forward.pooler = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match=match):
        pool_direct.install(model, None if case == "plan" else object(), mapping=_TP8)
    assert workspace == []
    assert len(group.gathered) == 1 and group.gathered[0] is not None


def test_install_refuses_here_when_a_peer_refuses(monkeypatch, workspace):
    peer = "[musespark_open] pool_into_route publishes from the forked router shard; layer 3 has X"
    group = _Group([None] * 6 + [peer])
    monkeypatch.setattr(tp_agreement, "Distributed", group)
    model, _ = _layers()
    with pytest.raises(ValueError, match=r"layer 3 has X \(on TP ranks \[7\] of 8\)"):
        pool_direct.install(model, object(), mapping=_TP8)
    assert workspace == []


def _place(kind):
    def placement(moe):
        return kind, 32, None

    return placement


def test_install_sites_claims_the_routed_layers_and_alternates_buffers(monkeypatch):
    group = _Group([None] * 7)
    monkeypatch.setattr(tp_agreement, "Distributed", group)
    model, layers = _layers(6)
    routed = {1, 2, 3, 4}
    monkeypatch.setattr(pool_direct, "_routes", lambda moe: moe.layer_idx in routed)
    monkeypatch.setattr(moe_prologue, "_placement", _place(pool_direct._pdr.PLACE_BALANCE))
    work = _Workspace()
    direct = pool_direct.DirectPool(work)
    assert direct.install_sites(model, _TP8) == 4
    assert [moe.layer_idx for moe, _ in direct.sites] == [1, 2, 3, 4]
    assert [site.buffer for _, site in direct.sites] == [0, 1, 0, 1]
    for moe, site in direct.sites:
        assert site.dispatcher is moe.pooler._low_m_dispatcher
        assert site.dispatcher.publish_policy is pool_direct.PUBLISH_POLICY
        assert site.dispatcher.publish is None
    for index in (0, 5):
        assert layers[index].feed_forward.pooler._low_m_dispatcher.publish_policy is None
    assert work.built == [pool_direct._pdr.variant(pool_direct._pdr.PLACE_BALANCE, 32, True, True)]
    assert work.watching
    assert group.gathered == [None, None]


def test_install_sites_builds_the_publishing_variant_too_with_early_ids(monkeypatch):
    monkeypatch.setattr(tp_agreement, "Distributed", _Group([None] * 7))
    model, layers = _layers(3)
    early = types.SimpleNamespace(words_address=4096, epoch_address=8192, trigger=1)
    early.stamps_address = 0
    for layer in layers:
        layer.feed_forward._early_ids_state = early
    monkeypatch.setattr(pool_direct, "_routes", lambda moe: True)
    kind = pool_direct._pdr.PLACE_BALANCE3
    monkeypatch.setattr(moe_prologue, "_placement", _place(kind))
    work = _Workspace()
    pool_direct.DirectPool(work).install_sites(model, _TP8)
    assert sorted(work.built) == sorted(
        [
            pool_direct._pdr.variant(kind, 32, True, True),
            pool_direct._pdr.variant(kind, 32, True, True, early),
        ]
    )
    assert pool_direct._pdr.variant(kind, 32, True, True, early)[4:7] == (4096, 8192, 1)


def test_install_sites_buffers_of_an_odd_count_end_on_the_third(monkeypatch):
    monkeypatch.setattr(tp_agreement, "Distributed", _Group([None] * 7))
    model, _ = _layers(5)
    monkeypatch.setattr(pool_direct, "_routes", lambda moe: True)
    monkeypatch.setattr(moe_prologue, "_placement", _place(pool_direct._pdr.PLACE_NONE))
    direct = pool_direct.DirectPool(_Workspace())
    direct.install_sites(model, _TP8)
    assert [site.buffer for _, site in direct.sites] == [0, 1, 0, 1, 2]


def test_install_sites_refuses_with_fewer_than_two_routed_layers(monkeypatch):
    group = _Group([None] * 7)
    monkeypatch.setattr(tp_agreement, "Distributed", group)
    model, _ = _layers(3)
    monkeypatch.setattr(pool_direct, "_routes", lambda moe: moe.layer_idx == 0)
    work = _Workspace()
    with pytest.raises(ValueError, match="found 1 layer"):
        pool_direct.DirectPool(work).install_sites(model, _TP8)
    assert work.built == []


def test_install_sites_refuses_without_the_fused_pool_site(monkeypatch):
    monkeypatch.setattr(tp_agreement, "Distributed", _Group([None] * 7))
    model, _ = _layers(3)
    model._layer_kwargs = {"fused_unpool_gather": True}
    monkeypatch.setattr(pool_direct, "_routes", lambda moe: True)
    work = _Workspace()
    with pytest.raises(ValueError, match="needs the fused pool site"):
        pool_direct.DirectPool(work).install_sites(model, _TP8)
    assert work.built == []


def test_install_sites_refuses_a_pool_gemm_another_feature_publishes(monkeypatch):
    monkeypatch.setattr(tp_agreement, "Distributed", _Group([None] * 7))
    model, layers = _layers(3)
    other = dense_gemms.PublishPolicy("unknown_path", frozenset(), lambda slot, product: None)
    dense_gemms.claim_publish(layers[1].feed_forward.pooler._low_m_dispatcher, other)
    monkeypatch.setattr(pool_direct, "_routes", lambda moe: True)
    with pytest.raises(ValueError, match="already publishes for unknown_path"):
        pool_direct.DirectPool(_Workspace()).install_sites(model, _TP8)


def test_install_sites_agrees_a_failed_compile(monkeypatch):
    peer = "[musespark_open] pool_into_route could not build its routed kernel variants: RuntimeError('x')"
    group = _Group([None] * 7)
    monkeypatch.setattr(tp_agreement, "Distributed", group)
    model, _ = _layers(2)
    monkeypatch.setattr(pool_direct, "_routes", lambda moe: True)
    monkeypatch.setattr(moe_prologue, "_placement", _place(pool_direct._pdr.PLACE_NONE))
    work = _Workspace(fail=RuntimeError("x"))
    with pytest.raises(ValueError, match=r"could not build its routed kernel variants"):
        pool_direct.DirectPool(work).install_sites(model, _TP8)
    assert group.gathered[-1] == peer


def test_takes_only_eight_rows_inside_a_capture(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    assert pool_direct.DirectPool.takes(8)
    assert not pool_direct.DirectPool.takes(7)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert not pool_direct.DirectPool.takes(8)


def test_armed_gives_each_site_to_its_layer_and_disarms_on_error():
    _, layers = _layers(2)
    direct = pool_direct.DirectPool(_Workspace())
    sites = [pool_direct.DirectPoolSite(direct.workspace, b, _dispatcher()) for b in (0, 1)]
    direct.sites = tuple((layer.feed_forward, site) for layer, site in zip(layers, sites))
    with pytest.raises(KeyError):
        with direct.armed(8):
            assert [layer.feed_forward._pool_direct_armed for layer in layers] == sites
            raise KeyError("boom")
    assert all(layer.feed_forward._pool_direct_armed is None for layer in layers)


def test_an_armed_layer_whose_call_declines_raises(monkeypatch):
    _, layers = _layers(1)
    moe = layers[0].feed_forward
    monkeypatch.setattr(type(moe), "_prologue_router_warps", lambda self, hidden: None)
    moe._pool_direct_armed = object()
    with pytest.raises(RuntimeError, match="armed for the direct pool transport"):
        moe.prepare_moe_inputs(torch.zeros(8, 16, dtype=torch.bfloat16))
