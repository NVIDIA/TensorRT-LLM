# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tep8 target's fused drafter hops (``drafter_hops``): its plan and what the shell installs.

Host-only. The hops are a drafter path: the shell installs them from its
drafter load, after ``draft.install`` (``drafter_block_decode``, whose block decode carries the
hop sites) and ``drafter_worker.install``.
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
    runtime as _runtime,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    spec_worker as drafter_worker,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.drafter_block_decode import (  # noqa: E501
    block_decode as draft,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.drafter_hops import (  # noqa: E501
    hops as drafter_hops,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.allreduce import (  # noqa: E501
    drafter_hop as _hop,
)
from tensorrt_llm._torch.models.modeling_musespark_open_omni import (
    MuseSparkOpenOmniForConditionalGeneration,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _no_runtime_hooks(monkeypatch):
    """The executor hooks pin host staging blocks; these shells are built without them."""
    monkeypatch.setattr(_runtime, "make_runtime_hooks", lambda model: None)


_SHELL = modeling.ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep


def test_the_hops_build_on_the_block_decode():
    assert dispatch.closure({"drafter_hops"}) == {"drafter_hops", "drafter_block_decode"}
    with pytest.raises(ValueError, match="builds on"):
        dispatch.Plan(frozenset({"drafter_hops"}))


@pytest.fixture
def build(monkeypatch):
    """Shells built through their own constructor with the built-in one stubbed."""
    events = []
    drafter, worker = nn.Module(), object()

    def builtin_init(self, model_config, *args, **kwargs):
        if hasattr(self, "llm"):
            return
        nn.Module.__init__(self)
        self.llm = SimpleNamespace(model=nn.Module(), draft_model=drafter, spec_worker=worker)

    def builtin_load_draft_weights(self, *args, **kwargs):
        events.append("built-in draft load")
        return "loaded"

    monkeypatch.setattr(MuseSparkOpenOmniForConditionalGeneration, "__init__", builtin_init)
    monkeypatch.setattr(
        MuseSparkOpenOmniForConditionalGeneration, "load_draft_weights", builtin_load_draft_weights
    )
    monkeypatch.setattr(
        modeling.MuseSparkOpenTepModelingV2,
        "adopt",
        classmethod(lambda cls, model, plan=dispatch.NO_PATHS: None),
    )
    monkeypatch.setattr(draft, "install", lambda *args: events.append("block decode install"))
    monkeypatch.setattr(
        drafter_worker, "install", lambda *args: events.append("drafter worker install")
    )
    monkeypatch.setattr(
        drafter_hops, "install", lambda *args: events.append(("drafter hops install", args))
    )

    def _build(*paths):
        plan = dispatch.plan_of(*paths)
        monkeypatch.setattr(modeling._dispatch, "compose", lambda *args, **kwargs: plan)
        events.clear()
        return _SHELL(model_config=SimpleNamespace(mapping="the mapping"))

    _build.events = events
    _build.drafter = drafter
    return _build


@pytest.mark.parametrize(
    "paths",
    [("drafter_hops",), ("drafter_fixed_gemms", "drafter_hops")],
    ids=["hops", "hops-fixed-gemms"],
)
def test_the_hops_install_after_the_other_drafter_features(build, paths):
    shell = build(*paths)
    assert shell.load_draft_weights("weights") == "loaded"
    names = [e if isinstance(e, str) else e[0] for e in build.events]
    assert names[0] == "built-in draft load"
    assert names.count("drafter hops install") == 1
    # After the installs it builds on; another drafter install may follow it.
    hop = names.index("drafter hops install")
    assert names.index("block decode install") < hop
    if "drafter worker install" in names:
        assert names.index("drafter worker install") < hop
    # With the shell's mapping: the refusal is agreed over its TP group.
    assert build.events[hop][1] == (build.drafter, "the mapping")


def test_without_the_hops_none_install(build):
    shell = build("drafter_block_decode")
    shell.load_draft_weights("weights")
    assert not any(isinstance(e, tuple) and e[0] == "drafter hops install" for e in build.events)


def test_install_refuses_a_missing_or_foreign_drafter():
    with pytest.raises(ValueError, match="no drafter"):
        drafter_hops.install(None)
    with pytest.raises(ValueError, match="cannot run here"):
        drafter_hops.install(nn.Linear(2, 2))


class _TpGroup:
    """Stands in for ``Distributed`` over a TP group of 8: this rank, then ``peers``."""

    def __init__(self, peers):
        self.peers = list(peers)
        self.mappings = []
        self.gathered = []

    def get(self, mapping):
        self.mappings.append(mapping)
        return self

    def tp_allgather(self, obj, small_payload=False):
        self.gathered.append(obj)
        return [obj] + self.peers


def _drafter_with_mapping(mapping):
    o_proj = SimpleNamespace(mapping=mapping)
    layer = SimpleNamespace(self_attn=SimpleNamespace(o_proj=o_proj))
    return SimpleNamespace(model=SimpleNamespace(layers=[layer]))


@pytest.mark.parametrize("shell_mapping", [True, False], ids=["shell", "drafter"])
def test_install_agrees_the_refusal_over_the_tp_group_before_the_workspace(
    monkeypatch, shell_mapping
):
    # This rank could install; one peer refuses: every rank raises the peer's reason before
    # the workspace's collective construction. The group is the shell's mapping's, or the
    # drafter's o_proj mapping when none is passed.
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
        dispatch as tp_agreement,
    )

    peer = "[musespark_open] drafter_hops cannot run here: layer 0 o_proj has a bias"
    group = _TpGroup([None] * 6 + [peer])
    monkeypatch.setattr(tp_agreement, "Distributed", group)
    monkeypatch.setattr(drafter_hops, "unsupported_reason", lambda model: None)
    built = []
    monkeypatch.setattr(_hop, "DrafterHopWorkspace", lambda *a, **k: built.append((a, k)))
    shell, own = SimpleNamespace(tp_size=8), SimpleNamespace(tp_size=8)
    drafter = _drafter_with_mapping(own)
    with pytest.raises(ValueError, match=r"has a bias \(on TP ranks \[7\] of 8\)"):
        drafter_hops.install(drafter, shell if shell_mapping else None)
    assert group.mappings == [shell if shell_mapping else own]
    assert group.gathered == [None]
    assert built == []
    assert not drafter_hops.installed(drafter)


def test_a_missing_drafter_is_agreed_over_the_shells_group(monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
        dispatch as tp_agreement,
    )

    group = _TpGroup([None] * 7)
    monkeypatch.setattr(tp_agreement, "Distributed", group)
    with pytest.raises(ValueError, match=r"no drafter \(on TP ranks \[0\] of 8\)"):
        drafter_hops.install(None, SimpleNamespace(tp_size=8))
    assert group.gathered == ["[musespark_open] drafter_hops cannot run here: no drafter"]


def test_takes_is_a_function_of_rank_uniform_facts():
    mc = drafter_hops.DrafterHopCall(workspace=None)
    assert all(mc.takes(rows, 8, torch.bfloat16) for rows in range(1, 9))
    assert not mc.takes(9, 8, torch.bfloat16)
    assert not mc.takes(16, 8, torch.bfloat16)
    assert not mc.takes(8, 8, torch.float16)


def _bare_workspace(next_site=0):
    """A workspace's host-side site bookkeeping without its fabric memory."""
    ws = object.__new__(_hop.DrafterHopWorkspace)
    ws.sites, ws.next_site, ws._forward_captured = 10, next_site, False
    return ws


def test_a_forward_whose_capture_failed_gives_the_site_order_back(monkeypatch):
    ws = _bare_workspace()
    capturing = {"now": True}
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing["now"])
    ws.begin_forward()
    ws.next_site = 4  # the captured forward raised after four sites
    capturing["now"] = False
    ws.begin_forward()
    assert ws.next_site == 0 and not ws._forward_captured
    # A complete forward leaves nothing to give back.
    ws.begin_forward()
    assert ws.next_site == 0


def test_an_eager_forward_stopped_part_way_is_refused_at_the_next_start(monkeypatch):
    ws = _bare_workspace()
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    ws.begin_forward()
    ws.next_site = 3  # the eager forward raised after three sites
    with pytest.raises(RuntimeError, match="outside graph capture"):
        ws.begin_forward()
    assert ws.next_site == 3
    # Captured or not, the next forward is refused while that order is incomplete.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="outside graph capture"):
        ws.begin_forward()


def test_the_hop_call_marks_the_forward_start_on_its_workspace(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    ws = _bare_workspace()
    drafter_hops.DrafterHopCall(workspace=ws).begin_forward()
    assert ws.next_site == 0 and ws._forward_captured


def _fake_drafter(layers=3):
    def norm():
        return SimpleNamespace(weight=torch.randn(16))

    def conv():
        return SimpleNamespace(base_kernel=torch.randn(2, 2, 16))

    return SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(post_attention_layernorm=norm(), input_layernorm=norm())
                for _ in range(layers)
            ],
            norm=norm(),
        ),
        attention_convs=[conv() for _ in range(layers)],
        mlp_convs=[conv() for _ in range(layers)],
    )


def test_the_pre_wait_operands_are_the_parameters_themselves():
    model = _fake_drafter()
    tensors = [t for _, t in drafter_hops.pre_wait_operands(model)]
    # Per layer two finish base kernels and post_attention_layernorm; input_layernorm
    # from layer 1 on (layer 0's runs as its own kernel); then model.norm.
    assert len(tensors) == 3 * 3 + 2 + 1
    for index in range(3):
        for conv in (model.attention_convs[index], model.mlp_convs[index]):
            assert any(t.data_ptr() == conv.base_kernel[1].data_ptr() for t in tensors)
        assert any(t is model.model.layers[index].post_attention_layernorm.weight for t in tensors)
    assert not any(t is model.model.layers[0].input_layernorm.weight for t in tensors)
    assert tensors[-1] is model.model.norm.weight
