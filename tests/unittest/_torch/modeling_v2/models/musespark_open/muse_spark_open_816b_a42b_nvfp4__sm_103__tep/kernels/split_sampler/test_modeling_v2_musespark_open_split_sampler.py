# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host tests of the head paths ``split_sampler`` / ``lm_head_inbox`` (``target_head.py``): wiring, refusals."""

from __future__ import annotations

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
    dispatch,
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.split_sampler import (  # noqa: E501
    target_head,
)


def _shell(paths):
    shell = types.SimpleNamespace()
    shell._plan = dispatch.plan_of(*paths)
    shell._mapping = object()
    shell.llm = types.SimpleNamespace(draft_model=None, spec_worker=None)
    return shell


@pytest.mark.parametrize("gather", [False, True], ids=["split-sampler", "lm-head-inbox"])
def test_shell_installs_the_head_after_the_drafter_features(monkeypatch, gather):
    calls = []
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
        spec_worker as drafter_worker,
    )

    monkeypatch.setattr(drafter_worker, "install", lambda *a, **k: calls.append("drafter_worker"))
    monkeypatch.setattr(
        target_head, "install", lambda llm, mapping, gather: calls.append(("head", gather))
    )
    head = ("split_sampler", "lm_head_inbox") if gather else ("split_sampler",)
    shell = _shell(("fused_accept",) + head)
    modeling.ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep._install_draft_features(shell)
    assert calls == ["drafter_worker", ("head", gather)]


def test_shell_installs_nothing_without_the_split_sampler(monkeypatch):
    calls = []
    monkeypatch.setattr(target_head, "install", lambda *a: calls.append(a))
    modeling.ModelingV2MuseSparkOpen816bA42bNvfp4Sm103Tep._install_draft_features(_shell(()))
    assert calls == []


def test_install_refuses_a_model_without_the_musespark_processor():
    llm = types.SimpleNamespace(
        logits_processor=torch.nn.Identity(), spec_worker=None, lm_head=None
    )
    with pytest.raises(
        ValueError, match="split_sampler cannot run here: the target logits processor"
    ):
        target_head.install(llm, object(), False)


class _TpGroup:
    """Stands in for ``Distributed`` over a TP group of 8: this rank, then ``peers``."""

    def __init__(self, peers):
        self.peers = list(peers)
        self.gathered = []

    def get(self, mapping):
        return self

    def tp_allgather(self, obj, small_payload=False):
        self.gathered.append(obj)
        return [obj] + self.peers


@pytest.mark.parametrize("name, gather", [("lm_head_inbox", True), ("split_sampler", False)])
def test_install_agrees_the_refusal_over_the_tp_group_before_any_collective(
    monkeypatch, name, gather
):
    # This rank could install; one peer refuses: every rank raises the peer's reason before
    # the head's collective construction (library agreement, inbox allocation).
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
        dispatch as tp_agreement,
    )

    peer = (
        f"[musespark_open] {name} cannot run here: the LM head is not an unbiased BF16 projection"
    )
    group = _TpGroup([None] * 6 + [peer])
    monkeypatch.setattr(tp_agreement, "Distributed", group)
    asked = []
    monkeypatch.setattr(
        target_head, "unsupported_reason", lambda llm, mapping, gather: asked.append(gather)
    )
    built = []
    monkeypatch.setattr(target_head, "TargetHead", lambda *args: built.append(args))
    llm = types.SimpleNamespace(logits_processor=None, spec_worker=None, lm_head=None)
    with pytest.raises(ValueError, match=r"BF16 projection \(on TP ranks \[7\] of 8\)"):
        target_head.install(llm, types.SimpleNamespace(tp_size=8), gather)
    assert asked == [gather]
    assert group.gathered == [None]
    assert built == []
    assert llm.logits_processor is None
