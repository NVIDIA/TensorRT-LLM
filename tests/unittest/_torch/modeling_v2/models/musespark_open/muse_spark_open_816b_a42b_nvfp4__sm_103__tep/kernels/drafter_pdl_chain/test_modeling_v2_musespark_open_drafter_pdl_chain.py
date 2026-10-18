# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tep8 target's drafter PDL chain (``drafter_pdl_chain``): its plan, refusals, install order, wiring.

Host-only. The five launches (``pdl_conv`` / ``pdl_merge`` / ``pdl_silu`` /
``pdl_head`` / ``selector_fork``) are one drafter path: the shell installs them
from its drafter load after ``drafter_hops.install``, whose hops and the
``drafter_fixed_gemms`` GEMM methods their entry triggers are checked against.
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
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.draft_head import (  # noqa: E501
    head as draft_head,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.drafter_block_decode import (  # noqa: E501
    block_decode as draft,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.drafter_hops import (  # noqa: E501
    hops as drafter_hops,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.drafter_pdl_chain import (  # noqa: E501
    chain as drafter_pdl,
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

#: The drafter paths the chain sits on.
_BASE = (
    "drafter_block_conv",
    "drafter_block_decode",
    "draft_head",
    "drafter_fc_shard",
    "fused_accept",
    "drafter_fixed_gemms",
    "drafter_hops",
)


def test_the_chain_builds_on_the_drafter_paths_its_triggers_release():
    direct = {
        "drafter_block_conv",
        "drafter_block_decode",
        "drafter_fixed_gemms",
        "drafter_hops",
        "draft_head",
    }
    assert dispatch.closure({"drafter_pdl_chain"}) == direct | {"drafter_pdl_chain"}
    for missing in sorted(direct):
        with pytest.raises(ValueError, match="builds on"):
            dispatch.Plan(dispatch.closure({"drafter_pdl_chain"}) - {missing})


@pytest.fixture
def build(monkeypatch):
    """Shells built through their own constructor with the built-in one stubbed."""
    events = []
    llm_holder = {}

    def builtin_init(self, model_config, *args, **kwargs):
        if hasattr(self, "llm"):
            return
        nn.Module.__init__(self)
        self.llm = SimpleNamespace(model=nn.Module(), draft_model=nn.Module(), spec_worker=object())
        llm_holder["llm"] = self.llm

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
    monkeypatch.setattr(draft_head, "install", lambda *args: events.append("draft head install"))
    monkeypatch.setattr(
        drafter_worker, "install", lambda *args: events.append("drafter worker install")
    )
    monkeypatch.setattr(
        drafter_hops, "install", lambda *args: events.append("drafter hops install")
    )
    monkeypatch.setattr(
        drafter_pdl, "install", lambda *args: events.append(("pdl chain install", args))
    )

    def _build(*paths):
        plan = dispatch.plan_of(*paths)
        monkeypatch.setattr(modeling._dispatch, "compose", lambda *args, **kwargs: plan)
        events.clear()
        return _SHELL(model_config=SimpleNamespace(mapping="the mapping"))

    _build.events = events
    _build.llm = llm_holder
    return _build


def test_the_chain_installs_after_the_hops(build):
    shell = build(*_BASE, "drafter_pdl_chain")
    assert shell.load_draft_weights("weights") == "loaded"
    names = [e if isinstance(e, str) else e[0] for e in build.events]
    assert names.count("pdl chain install") == 1
    chain = names.index("pdl chain install")
    for before in (
        "block decode install",
        "draft head install",
        "drafter worker install",
        "drafter hops install",
    ):
        assert names.index(before) < chain
    # With the shell's mapping: the refusal is agreed over its TP group.
    assert build.events[chain][1] == (build.llm["llm"], "the mapping")


def test_without_the_chain_none_installs(build):
    shell = build(*_BASE)
    shell.load_draft_weights("weights")
    assert not any(isinstance(e, tuple) and e[0] == "pdl chain install" for e in build.events)


def test_install_refuses_a_missing_drafter():
    with pytest.raises(ValueError, match="no drafter"):
        drafter_pdl.install(SimpleNamespace(draft_model=None))


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


def test_install_agrees_the_refusal_over_the_tp_group(monkeypatch):
    # This rank could install; one peer refuses: every rank raises the peer's reason and
    # nothing is installed here.
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
        dispatch as tp_agreement,
    )

    peer = "[musespark_open] pdl_conv: the conv's successor qkv_proj is not on the fixed tactic"
    group = _TpGroup([None] * 6 + [peer])
    monkeypatch.setattr(tp_agreement, "Distributed", group)
    monkeypatch.setattr(drafter_pdl, "unsupported_reason", lambda llm: None)
    conv = nn.Module()
    draft_model = nn.Module()
    draft_model.attention_convs = nn.ModuleList([conv])
    draft_model.mlp_convs = nn.ModuleList()
    llm = SimpleNamespace(draft_model=draft_model, spec_worker=None)
    mapping = SimpleNamespace(tp_size=8)
    with pytest.raises(ValueError, match=r"fixed tactic \(on TP ranks \[7\] of 8\)"):
        drafter_pdl.install(llm, mapping)
    assert group.mappings == [mapping] and group.gathered == [None]
    assert "_pdl_conv" not in conv.__dict__


def test_a_missing_drafter_is_agreed_over_the_shells_group(monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
        dispatch as tp_agreement,
    )

    group = _TpGroup([None] * 7)
    monkeypatch.setattr(tp_agreement, "Distributed", group)
    with pytest.raises(ValueError, match=r"no drafter \(on TP ranks \[0\] of 8\)"):
        drafter_pdl.install(SimpleNamespace(draft_model=None), SimpleNamespace(tp_size=8))
    assert group.gathered == ["[musespark_open] no drafter"]


# --------------------------------------------------------------------------
# selector_fork: the selector fork's rows
# --------------------------------------------------------------------------


def test_the_selector_fork_takes_only_the_one_request_draft_rows():
    """``selector_fork`` forks for the K = 7 contiguous rows of one request; any other call forks nothing."""
    assert drafter_pdl.HEAD_ROWS == drafter_pdl.ROWS - 1 == 7
    fork = object.__new__(drafter_pdl.SelectorFork)
    fork.projection = torch.nn.Linear(16, 4, bias=False)
    fork.stream = None
    fork.launches = fork.taken = 0
    fork._pending = None
    for rows in (1, 6, 8, 14, 21):
        fork.launch(torch.randn(rows, 16))
        assert fork.launches == 0 and fork._pending is None, f"{rows} rows"
    fork.launch(torch.randn(16, 7).t())  # 7 rows, not contiguous
    fork.launch(torch.randn(1, 7, 16))  # not 2-D
    fork.launch(torch.randn(7, 16, dtype=torch.float64))  # not the weight's dtype
    assert fork.launches == 0 and fork._pending is None


# --------------------------------------------------------------------------
# pdl_head: the slot range
# --------------------------------------------------------------------------
def test_a_covering_clamp_of_the_row_range_is_its_slice():
    ids = torch.arange(1, 8)
    rows = drafter_pdl.DraftRowRange(ids, 1, 8)
    assert rows.clamp(max=7) == slice(1, 8)
    assert rows.clamp(max=100) == slice(1, 8)
    assert rows.clamp(min=0, max=7) == slice(1, 8)
    hidden = torch.randn(8, 16)
    assert torch.equal(hidden[rows.clamp(max=7)], hidden[ids.clamp(max=7)])
    # A bound that cuts a slot is the tensor's own clamp.
    cut = rows.clamp(max=5)
    assert isinstance(cut, torch.Tensor) and torch.equal(cut, ids.clamp(max=5))
    with pytest.raises(AttributeError):
        rows.long()


def test_the_row_range_is_the_builtin_slot_ids_of_one_request():
    from tensorrt_llm._torch.speculative.dflash import dflash_draft_slot_ids

    ids = dflash_draft_slot_ids(1, 8, 7, False, device="cpu")
    assert drafter_pdl.draft_row_range(1, 8, 7, ids) == (1, 8)
    assert drafter_pdl.draft_row_range(1, 4, 3, dflash_draft_slot_ids(1, 4, 3, False, "cpu")) == (
        1,
        4,
    )
    # Two requests: not one range.
    assert (
        drafter_pdl.draft_row_range(2, 8, 7, dflash_draft_slot_ids(2, 8, 7, False, "cpu")) is None
    )
    # Ids that are not the built-in ones are not trusted.
    assert drafter_pdl.draft_row_range(1, 8, 7, ids + 1) is None
