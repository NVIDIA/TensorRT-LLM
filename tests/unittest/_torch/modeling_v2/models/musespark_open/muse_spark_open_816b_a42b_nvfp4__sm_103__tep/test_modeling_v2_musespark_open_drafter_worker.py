# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tep8 target's drafter worker paths: what the shell installs.

Host-only. The shell decides its plan once, at construction. Under
``NO_PATHS`` its drafter load is the built-in one and leaves the drafter and
the DFlash worker as built; with a drafter worker path planned it hands the drafter, the
worker, the mapping and the switches to ``drafter_worker.install`` after the
built-in load returns.
"""

from __future__ import annotations

import inspect
import types
from types import SimpleNamespace

import pytest
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
from tensorrt_llm._torch.models.modeling_musespark_open_omni import (
    MuseSparkOpenOmniForConditionalGeneration,
)
from tensorrt_llm._torch.speculative.dflash import DFlashWorker

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _no_runtime_hooks(monkeypatch):
    """The executor hooks pin host staging blocks; these shells are built without them."""
    monkeypatch.setattr(_runtime, "make_runtime_hooks", lambda model: None)


_SHELL = modeling.ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep


@pytest.fixture
def build(monkeypatch):
    """Construct shells through their own constructor with the built-in one stubbed."""
    events = []
    drafter, worker = nn.Module(), object()

    def builtin_init(self, model_config, *args, **kwargs):
        events.append("built-in model")
        if hasattr(self, "llm"):
            return
        nn.Module.__init__(self)
        self.llm = SimpleNamespace(model=nn.Module(), draft_model=drafter, spec_worker=worker)

    def builtin_load_draft_weights(self, *args, **kwargs):
        events.append(("built-in draft load", args, kwargs))
        return "loaded"

    monkeypatch.setattr(MuseSparkOpenOmniForConditionalGeneration, "__init__", builtin_init)
    monkeypatch.setattr(
        MuseSparkOpenOmniForConditionalGeneration, "load_draft_weights", builtin_load_draft_weights
    )
    monkeypatch.setattr(
        modeling.MuseSparkOpenTepModelingV2,
        "adopt",
        classmethod(
            lambda cls, model, plan=dispatch.NO_PATHS, compute_capability=None: events.append(
                "adopt"
            )
        ),
    )
    monkeypatch.setattr(
        drafter_worker, "install", lambda *args: events.append(("drafter worker install", args))
    )

    def _build(*paths):
        plan = dispatch.plan_of(*paths)
        monkeypatch.setattr(modeling._dispatch, "compose", lambda *args, **kwargs: plan)
        events.clear()
        return _SHELL(model_config=SimpleNamespace(mapping="the mapping"))

    _build.events = events
    _build.drafter, _build.worker = drafter, worker
    return _build


def test_the_drafter_load_keeps_the_builtin_signature():
    assert (
        inspect.getfullargspec(_SHELL.load_draft_weights).args
        == inspect.getfullargspec(MuseSparkOpenOmniForConditionalGeneration.load_draft_weights).args
    )


def test_no_feature_is_the_builtin_drafter_load(build):
    shell = build()
    assert shell._plan == dispatch.NO_PATHS
    assert shell.load_draft_weights("weights", "mapper") == "loaded"
    assert build.events == [
        "built-in model",
        "adopt",
        ("built-in draft load", ("weights", "mapper"), {}),
    ]


@pytest.mark.parametrize(
    "paths",
    [
        ("drafter_fc_shard",),
        ("fused_accept",),
        ("drafter_fixed_gemms",),
        ("drafter_fc_shard", "fused_accept", "drafter_fixed_gemms"),
    ],
)
def test_a_drafter_worker_feature_installs_after_the_builtin_load(build, paths):
    shell = build(*paths)
    assert shell.load_draft_weights("weights") == "loaded"
    assert build.events[-2] == ("built-in draft load", ("weights",), {})
    switches = tuple(
        name in paths for name in ("drafter_fc_shard", "fused_accept", "drafter_fixed_gemms")
    )
    assert build.events[-1] == (
        "drafter worker install",
        (build.drafter, build.worker, "the mapping", *switches),
    )


def test_the_plan_is_decided_once_at_construction(monkeypatch, build):
    shell = build()
    monkeypatch.setattr(
        modeling._dispatch, "compose", lambda *a, **k: dispatch.plan_of("fused_accept")
    )
    shell.load_draft_weights("weights")
    assert build.events[-1][0] == "built-in draft load"


def test_fused_accept_reclasses_the_builtin_worker():
    worker = DFlashWorker.__new__(DFlashWorker)
    # The DFlash context arena's windowing state, as DFlashWorker.__init__ sets it.
    worker._ctx_base = None
    worker._ctx_base_host = None
    worker._ctx_keep = 0
    worker.windowed_ctx_arena = False
    worker._saved_ctx_base = None
    worker._saved_ctx_base_host = None
    assert drafter_worker.install_fused_accept(worker) is None
    assert type(worker) is drafter_worker.FusedAcceptDFlashWorker
    assert drafter_worker.install_fused_accept(worker) is None
    assert type(worker) is drafter_worker.FusedAcceptDFlashWorker


def test_fused_accept_composes_with_another_features_worker_class():
    class OtherFeatureWorker(DFlashWorker):
        def _draft_slot_ids(self):
            return "other feature"

    worker = OtherFeatureWorker.__new__(OtherFeatureWorker)
    assert drafter_worker.install_fused_accept(worker) is None
    cls = type(worker)
    assert issubclass(cls, drafter_worker.FusedAcceptDFlashWorker)
    assert issubclass(cls, OtherFeatureWorker)
    assert worker._draft_slot_ids() == "other feature"
    assert (
        cls._sample_and_accept_draft_tokens_base
        is drafter_worker.FusedAcceptDFlashWorker._sample_and_accept_draft_tokens_base
    )
    # super() from the fused method reaches the built-in through the other class.
    assert cls.__mro__.index(OtherFeatureWorker) < cls.__mro__.index(DFlashWorker)
    assert drafter_worker.install_fused_accept(worker) is None and type(worker) is cls
    second = OtherFeatureWorker.__new__(OtherFeatureWorker)
    assert drafter_worker.install_fused_accept(second) is None and type(second) is cls


@pytest.mark.parametrize("method", drafter_worker.ACCEPT_METHODS)
def test_fused_accept_leaves_a_worker_class_that_overrides_acceptance(method):
    overriding = type("Overriding", (DFlashWorker,), {method: lambda self, *a, **k: None})
    worker = overriding.__new__(overriding)
    reason = drafter_worker.install_fused_accept(worker)
    assert f"overrides {method}" in reason
    assert type(worker) is overriding
    assert "not a DFlashWorker" in drafter_worker.install_fused_accept(object())


@pytest.mark.parametrize("method", drafter_worker.ACCEPT_METHODS)
def test_fused_accept_leaves_a_worker_with_an_instance_binding(method):
    worker = DFlashWorker.__new__(DFlashWorker)
    # The DFlash context arena's windowing state, as DFlashWorker.__init__ sets it.
    worker._ctx_base = None
    worker._ctx_base_host = None
    worker._ctx_keep = 0
    worker.windowed_ctx_arena = False
    worker._saved_ctx_base = None
    worker._saved_ctx_base_host = None
    bound = types.MethodType(lambda self, *a, **k: None, worker)
    worker.__dict__[method] = bound
    reason = drafter_worker.install_fused_accept(worker)
    assert reason.startswith(f"{method} is already replaced on this worker by ")
    assert type(worker) is DFlashWorker and worker.__dict__[method] is bound


def test_fused_accept_declines_other_rng_strides(monkeypatch):
    monkeypatch.setattr(drafter_worker._spec_interface, "_FORCE_ACCEPT_RNG_COUNTER_STRIDE", 7919)
    worker = DFlashWorker.__new__(DFlashWorker)
    # The DFlash context arena's windowing state, as DFlashWorker.__init__ sets it.
    worker._ctx_base = None
    worker._ctx_base_host = None
    worker._ctx_keep = 0
    worker.windowed_ctx_arena = False
    worker._saved_ctx_base = None
    worker._saved_ctx_base_host = None
    assert "strides" in drafter_worker.install_fused_accept(worker)
    assert type(worker) is DFlashWorker


def test_force_plan_follows_the_builtin_branches(monkeypatch):
    monkeypatch.setattr(drafter_worker.torch.cuda, "is_current_stream_capturing", lambda: False)
    worker = DFlashWorker.__new__(DFlashWorker)
    # The DFlash context arena's windowing state, as DFlashWorker.__init__ sets it.
    worker._ctx_base = None
    worker._ctx_base_host = None
    worker._ctx_keep = 0
    worker.windowed_ctx_arena = False
    worker._saved_ctx_base = None
    worker._saved_ctx_base_host = None
    drafter_worker.install_fused_accept(worker)
    md = SimpleNamespace(is_cuda_graph=False)
    plans = {}
    for rate in (0.0, 3.0, 3.05, 6.5, 7.0, 7.5, 9.0):
        worker.force_num_accepted_tokens = rate
        plans[rate] = worker._force_plan(7, md)
    assert plans[0.0] == (drafter_worker.FORCE_NONE, 0, 0, 0.0)
    assert plans[3.0] == (drafter_worker.FORCE_FIXED, 4, 8, 0.0)
    assert plans[3.05][:3] == (drafter_worker.FORCE_FRACTIONAL, 4, 8)
    assert plans[3.05][3] == 3.05 - 3
    assert plans[6.5][:3] == (drafter_worker.FORCE_FRACTIONAL, 7, 8)
    assert plans[7.0] == plans[7.5] == plans[9.0] == (drafter_worker.FORCE_FIXED, 8, 8, 0.0)
    worker.force_num_accepted_tokens = 3.05
    assert (
        worker._force_plan(7, SimpleNamespace(is_cuda_graph=True))[0] == drafter_worker.FORCE_NONE
    )


def _record_logs(monkeypatch):
    logged = []
    monkeypatch.setattr(
        drafter_worker,
        "logger",
        SimpleNamespace(
            info=lambda msg: logged.append(("info", msg)),
            warning=lambda msg: logged.append(("warning", msg)),
        ),
    )
    return logged


@pytest.mark.parametrize(
    "feature, drafter, worker, match",
    [
        ("fc_shard", None, None, "drafter_fc_shard cannot run here: no drafter"),
        ("fused_accept", None, None, "fused_accept cannot run here: no spec worker"),
        ("fixed_gemms", None, None, "drafter_fixed_gemms cannot run here: no drafter"),
        (
            "fc_shard",
            nn.Module(),
            None,
            r"drafter_fc_shard cannot run here: not a DFlashForCausalLM \(Module\)",
        ),
        (
            "fused_accept",
            None,
            object(),
            "fused_accept cannot run here: the spec worker is object, not a DFlash",
        ),
        (
            "fixed_gemms",
            nn.Module(),
            None,
            r"drafter_fixed_gemms cannot run here: 1 of 1 drafter GEMM call sites take no fixed tactic "
            r"\(not a DFlashForCausalLM \(Module\)\)",
        ),
    ],
    ids=[
        "fc_shard-none",
        "fused_accept-none",
        "fixed_gemms-none",
        "fc_shard-module",
        "fused_accept-object",
        "fixed_gemms-module",
    ],
)
def test_install_refuses_a_selected_feature_it_cannot_install(
    monkeypatch, feature, drafter, worker, match
):
    """Running the built-in path under the feature's name would mislabel the run."""
    logged = _record_logs(monkeypatch)
    switches = [name == feature for name in ("fc_shard", "fused_accept", "fixed_gemms")]
    with pytest.raises(ValueError, match=match):
        drafter_worker.install(drafter, worker, None, *switches)
    assert logged == []


def _context_kv_drafter(convs=()):
    """A DFlashForCausalLM without its constructor: no decoder Linear, ``convs`` as its
    attention convolutions, and the built-in context K/V GEMM (one fixed-tactic site)."""
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

    drafter = DFlashForCausalLM.__new__(DFlashForCausalLM)
    nn.Module.__init__(drafter)
    drafter.attention_convs = nn.ModuleList(convs) if convs else None
    drafter.mlp_convs = None
    drafter.model = nn.Module()
    drafter.model.layers = nn.ModuleList()
    return drafter


def test_install_logs_a_feature_only_on_the_call_that_changed_something(monkeypatch):
    """The shell installs from every load hook: the second call finds everything in
    place and logs nothing."""
    logged = _record_logs(monkeypatch)
    worker = DFlashWorker.__new__(DFlashWorker)
    # The DFlash context arena's windowing state, as DFlashWorker.__init__ sets it.
    worker._ctx_base = None
    worker._ctx_base_host = None
    worker._ctx_keep = 0
    worker.windowed_ctx_arena = False
    worker._saved_ctx_base = None
    worker._saved_ctx_base_host = None
    drafter = _context_kv_drafter()
    drafter_worker.install(drafter, worker, None, False, True, True)
    assert [level for level, _ in logged] == ["info", "info"]
    assert "fused_accept: DFlash acceptance" in logged[0][1]
    assert "worker class FusedAcceptDFlashWorker" in logged[0][1]
    assert (
        "drafter_fixed_gemms: 1 of 1 drafter GEMM call sites on fixed low-M tactics" in logged[1][1]
    )
    drafter_worker.install(drafter, worker, None, False, True, True)
    assert len(logged) == 2
    sites = drafter_worker.install_fixed_tactic_gemms(drafter, enable_pdl=False)
    assert (sites.installed, sites.changed, sites.declined) == (1, 0, ())


def test_a_drafter_gemm_site_without_a_fixed_tactic_fails_the_install(monkeypatch):
    """The sites follow from the drafter's structure: a site left built-in is an error."""
    logged = _record_logs(monkeypatch)
    drafter = _context_kv_drafter(convs=[nn.Identity()])
    with pytest.raises(
        ValueError,
        match=r"drafter_fixed_gemms cannot run here: 1 of 2 drafter GEMM call sites take no fixed tactic "
        r"\(attention_convs.0 is Identity, not a DFlash2BlockConv\)",
    ):
        drafter_worker.install(drafter, None, None, False, False, True)
    assert logged == []


def test_fc_shard_logs_once_and_installs_through_install_sharded_fc(monkeypatch):
    logged = _record_logs(monkeypatch)
    calls = []
    drafter = _context_kv_drafter()

    def bind(model, mapping, enable_pdl):
        calls.append(mapping)
        if not drafter_worker.sharded_fc_installed(model):
            model._fc_shard = drafter_worker.ShardedFc(0, 1024, object(), enable_pdl)
            model.project_target_hidden = types.MethodType(
                drafter_worker.sharded_project_target_hidden, model
            )
        return None

    monkeypatch.setattr(drafter_worker, "install_sharded_fc", bind)
    assert not drafter_worker.sharded_fc_installed(drafter)
    drafter_worker.install(drafter, None, "the mapping", True, False, False)
    drafter_worker.install(drafter, None, "the mapping", True, False, False)
    assert calls == ["the mapping"] * 2
    assert drafter_worker.sharded_fc_installed(drafter)
    assert [level for level, _ in logged] == ["info"]
    assert "drafter_fc_shard: DFlash fc column-sharded" in logged[0][1]
