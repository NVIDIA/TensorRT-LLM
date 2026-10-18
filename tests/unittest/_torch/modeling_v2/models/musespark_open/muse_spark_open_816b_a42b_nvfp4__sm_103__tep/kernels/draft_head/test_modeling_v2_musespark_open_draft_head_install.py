# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shell's drafter hooks with the draft head (``draft_head``), host-only.

That the shell's ``load_draft_weights`` is the built-in load and nothing else
unless its plan, decided when the shell was built, has a drafter path; and that
the loader's post-load walks, which every load path runs (``LoadFormat.DUMMY``
included, which never calls ``load_draft_weights``), install a planned path
through the shell's ``post_load_weights`` / ``cache_derived_state``. The GPU
behavior of the installed draft head is in ``spec/test_modeling_v2_draft_head.py``.
"""

from __future__ import annotations

import inspect
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
    modeling,
)
from tensorrt_llm._torch.models.modeling_musespark_open_omni import (
    MuseSparkOpenOmniForConditionalGeneration,
)

pytestmark = pytest.mark.cpu_only

_SHELL = modeling.ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep


def test_load_draft_weights_keeps_the_builtin_signature():
    builtin = inspect.getfullargspec(MuseSparkOpenOmniForConditionalGeneration.load_draft_weights)
    assert inspect.getfullargspec(_SHELL.load_draft_weights) == builtin


def _shell_with(monkeypatch, calls, paths=()):
    """A shell without its constructor, holding the plan the constructor would have made."""

    def builtin_load(self, *args, **kwargs):
        calls.append(("builtin", args, kwargs))
        return "loaded"

    monkeypatch.setattr(
        MuseSparkOpenOmniForConditionalGeneration, "load_draft_weights", builtin_load
    )
    shell = _SHELL.__new__(_SHELL)
    torch.nn.Module.__init__(shell)
    shell.llm = types.SimpleNamespace(spec_worker=None, draft_model=None)
    shell._plan = dispatch.plan_of(*paths)
    return shell


def test_no_feature_is_the_builtin_load_only(monkeypatch):
    calls = []
    shell = _shell_with(monkeypatch, calls)
    assert shell.load_draft_weights({"w": 1}, weight_mapper="m") == "loaded"
    assert calls == [("builtin", ({"w": 1},), {"weight_mapper": "m"})]


def test_draft_head_installs_after_the_builtin_load(monkeypatch):
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.draft_head import (  # noqa: E501
        head as draft_head,
    )

    calls = []
    shell = _shell_with(monkeypatch, calls, ("draft_head",))
    monkeypatch.setattr(draft_head, "install", lambda llm: calls.append(("install", llm)))
    assert shell.load_draft_weights({}) == "loaded"
    assert [c[0] for c in calls] == ["builtin", "install"]
    assert calls[1][1] is shell.llm


def test_the_load_uses_the_plan_made_at_construction(monkeypatch):
    """The plan is made once, when the shell is built."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.draft_head import (  # noqa: E501
        head as draft_head,
    )

    calls = []
    shell = _shell_with(monkeypatch, calls)
    monkeypatch.setattr(draft_head, "install", lambda *args: calls.append(("install",)))
    monkeypatch.setattr(
        modeling._dispatch, "compose", lambda *a, **k: dispatch.plan_of("draft_head")
    )
    assert shell.load_draft_weights({}) == "loaded"
    assert [c[0] for c in calls] == ["builtin"]


def test_draft_head_without_a_dflash2_drafter_is_refused(monkeypatch):
    """No DFlash worker, or a drafter without a candidate selector (a plain DFlash
    drafter, or one whose checkpoint never loaded): the install raises instead
    of running the built-in head."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.draft_head import (  # noqa: E501
        head as draft_head,
    )

    for llm in (
        types.SimpleNamespace(spec_worker=None, draft_model=None),
        types.SimpleNamespace(
            spec_worker=object(), draft_model=types.SimpleNamespace(candidate_selector=None)
        ),
    ):
        with pytest.raises(
            ValueError, match="draft_head cannot run here: there is no DFlash 2 drafter"
        ):
            draft_head.install(llm)


def _record_install(monkeypatch, calls):
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.draft_head import (  # noqa: E501
        head as draft_head,
    )

    monkeypatch.setattr(draft_head, "install", lambda llm: calls.append(("install", llm)))


def test_builtin_shell_has_no_post_load_hooks_of_its_own():
    """The shell's two post-load hooks replace nothing: they add the drafter installs only."""
    for name in ("post_load_weights", "cache_derived_state"):
        assert not hasattr(MuseSparkOpenOmniForConditionalGeneration, name), name
        assert name in vars(_SHELL), name


@pytest.mark.parametrize("walk", ["_walk_full_post_load", "_walk_cache_state"])
def test_loader_post_load_walks_install_a_planned_path(walk, monkeypatch):
    """What a load that never calls ``load_draft_weights`` (``LoadFormat.DUMMY``) still runs."""
    from tensorrt_llm._torch.pyexecutor.model_loader import ModelLoader

    calls = []
    shell = _shell_with(monkeypatch, calls, ("draft_head",))
    _record_install(monkeypatch, calls)
    getattr(ModelLoader, walk)(shell)
    assert calls == [("install", shell.llm)]


@pytest.mark.parametrize("hook", ["post_load_weights", "cache_derived_state"])
def test_post_load_hooks_do_nothing_without_a_drafter_path(hook, monkeypatch):
    calls = []
    shell = _shell_with(monkeypatch, calls)
    _record_install(monkeypatch, calls)
    assert getattr(shell, hook)() is None
    assert calls == []


def test_install_after_an_install_changes_nothing():
    """The drafter load installs first; the post-load walk's second call is a no-op."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.draft_head import (  # noqa: E501
        head as draft_head,
    )

    worker = draft_head.ShardedDraftHeadDFlashWorker.__new__(
        draft_head.ShardedDraftHeadDFlashWorker
    )
    llm = types.SimpleNamespace(spec_worker=worker, draft_model=None)
    assert draft_head.installed(llm)

    def rebuild(mapping):
        raise AssertionError("the key exchange was built twice")

    assert draft_head.install(llm, make_allreduce=rebuild) is True
