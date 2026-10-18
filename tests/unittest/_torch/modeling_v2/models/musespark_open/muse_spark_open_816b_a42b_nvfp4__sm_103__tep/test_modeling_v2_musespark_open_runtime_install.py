# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The construction-time refusals of the plan's fast paths, and the executor's step staging.

Host-only. A fast path whose projection-gather site the install kept on the built-in path
is refused, the early release is refused when it converted none of the edges it owns, and
the one-chain prologue is refused when it installed on no layer. The executor
hooks (``runtime.py``) are imported only under a composition, and the shell
builds them (both parts) after the composition is decided and before the text
model is adopted; under ``NO_PATHS`` it declares none.
"""

from __future__ import annotations

import sys
import types

import pytest
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    layers as target_layers,
)
from tensorrt_llm._torch.models.modeling_musespark_open_omni import (
    MuseSparkOpenOmniForConditionalGeneration,
)
from tensorrt_llm._torch.pyexecutor.runtime_hooks import (
    DEFAULT_RUNTIME_HOOKS,
    RuntimeHooks,
    runtime_hooks_of,
)

pytestmark = pytest.mark.cpu_only

_SHELL = modeling.ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep
_PKG = modeling.__package__

_PLAN = dispatch.plan_of
#: The plan of every fast path.
_SERVED = dispatch.Plan(dispatch.FAST_PATH_NAMES)


# --------------------------------------------------------------------------
# A selected feature whose projection-gather site install kept on the built-in path
# --------------------------------------------------------------------------
_FUSED = {"fused_unpool_gather": True, "fused_pool_gather": True}


# --------------------------------------------------------------------------
# A fast path whose projection-gather site install kept on the built-in path
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "plan, message",
    [
        (_PLAN("boundary_early_release"), "early_release moves the trigger"),
        (_PLAN("capture_fold"), "capture_fold writes the DFlash capture"),
        (_PLAN("unpool_into_boundary"), "unpool_direct publishes into"),
        (_SERVED, "early_release moves the trigger"),
    ],
)
def test_a_feature_on_a_site_kept_builtin_is_refused(plan, message):
    with pytest.raises(ValueError, match=message):
        target_layers._refuse_unfused_sites({"fused_pool_gather": True}, plan)


@pytest.mark.parametrize(
    "plan, kwargs",
    [
        (dispatch.NO_PATHS, {}),
        (_PLAN("projection_gathers", "moe_prologue"), {"fused_pool_gather": True}),
        (_SERVED, _FUSED),
    ],
)
def test_a_feature_on_a_fused_site_is_accepted(plan, kwargs):
    target_layers._refuse_unfused_sites(kwargs, plan)


def test_a_prologue_installed_on_no_layer_is_refused():
    # moe_prologue.install keeps every layer on its path (0) when the gathers' pool site runs
    # the built-in path.
    with pytest.raises(ValueError, match=r"installed the one-chain MoE prologue on no layer"):
        target_layers._refuse_uninstalled_prologue(0)


@pytest.mark.parametrize("installed", [1, 60])
def test_a_prologue_installed_on_some_layer_is_accepted(installed):
    target_layers._refuse_uninstalled_prologue(installed)


@pytest.mark.parametrize(
    "converted, kept, layers, refused",
    [
        ([], [], 62, "none of the 61"),
        ([], [1, 16, 30, 44, 59], 62, "none of the 56"),
        ([0], [], 2, None),
        (list(range(56)), [1, 16, 30, 44, 59], 62, None),
        # Every edge is a DFlash capture layer's, or there is no edge.
        ([], [0, 1], 3, None),
        ([], [], 1, None),
    ],
)
def test_early_release_converting_no_edge_it_owns_is_refused(converted, kept, layers, refused):
    if refused is None:
        target_layers._refuse_unconverted_edges(converted, kept, layers)
    else:
        with pytest.raises(ValueError, match=refused):
            target_layers._refuse_unconverted_edges(converted, kept, layers)


# --------------------------------------------------------------------------
# The executor's step staging
# --------------------------------------------------------------------------


class _Llm(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.draft_model = nn.Module()


@pytest.fixture
def build(monkeypatch):
    """Build shells through their own constructor, the built-in one stubbed; ``events``
    records the built-in constructor, adoption, and the hook build (a stand-in
    ``runtime``)."""
    events = []
    hooks = RuntimeHooks(step_input_stager=object(), sampler_store_writer=object())

    def builtin_init(self, model_config, *args, **kwargs):
        events.append("built-in model")
        nn.Module.__init__(self)
        self.llm = _Llm()

    def make_runtime_hooks(model):
        events.append(("hooks", model._mapping.pp_size))
        return hooks

    monkeypatch.setattr(MuseSparkOpenOmniForConditionalGeneration, "__init__", builtin_init)
    monkeypatch.setattr(
        modeling.MuseSparkOpenTepModelingV2,
        "adopt",
        classmethod(lambda cls, model, plan=dispatch.NO_PATHS: events.append("adopt")),
    )
    runtime = types.SimpleNamespace(make_runtime_hooks=make_runtime_hooks)
    monkeypatch.setitem(sys.modules, f"{_PKG}.runtime", runtime)
    monkeypatch.setattr(sys.modules[_PKG], "runtime", runtime, raising=False)

    def _build(plan, pp_size=1):
        monkeypatch.setattr(modeling._dispatch, "compose", lambda *args, **kwargs: plan)
        events.clear()
        config = types.SimpleNamespace(mapping=types.SimpleNamespace(pp_size=pp_size))
        return _SHELL(model_config=config)

    _build.events = events
    _build.hooks = hooks
    return _build


@pytest.mark.parametrize("tp", sorted(modeling.TARGET_PATHS))
def test_every_composition_serves_both_executor_paths(tp):
    served = modeling.TARGET_PATHS[tp].served
    assert {"deferred_input_prep", "sampler_store"} <= served


@pytest.mark.parametrize("tp", sorted(modeling.TARGET_PATHS))
def test_the_shell_builds_its_hooks_after_compose_and_before_adopt(build, tp):
    shell = build(dispatch.Plan(modeling.TARGET_PATHS[tp].served))
    assert build.events == ["built-in model", ("hooks", 1), "adopt"]
    assert shell.runtime_hooks is build.hooks
    assert runtime_hooks_of(shell) is build.hooks


def test_the_shell_hands_its_pipeline_size_to_the_hooks(build):
    build(_SERVED, pp_size=2)
    assert ("hooks", 2) in build.events


def test_the_shell_builds_its_hooks_once(build):
    shell = build(_SERVED)
    config = types.SimpleNamespace(mapping=types.SimpleNamespace(pp_size=1))
    _SHELL.__init__(shell, model_config=config)
    assert [e for e in build.events if e[0] == "hooks"] == [("hooks", 1)]
    assert shell.runtime_hooks is build.hooks


def test_without_a_composition_the_shell_declares_no_hooks(build, monkeypatch):
    # Even with the module unimportable: nothing tries.
    monkeypatch.setitem(sys.modules, f"{_PKG}.runtime", None)
    shell = build(dispatch.NO_PATHS)
    assert build.events == ["built-in model", "adopt"]
    assert shell.runtime_hooks is None
    assert runtime_hooks_of(shell) is DEFAULT_RUNTIME_HOOKS
