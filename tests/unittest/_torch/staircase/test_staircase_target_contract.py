# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every op a target names has to exist in the build it is running against.

A target reads private engine surface: custom ops registered under
``torch.ops.trtllm`` and one pybind entry point. With no version pin, nothing
declares which build it was written for -- so the only thing that can tell you
the surface moved is running against it.

This used to be an ``assert`` loop that each target ran at import. That made
every import of a modeling module pay for the check and put a test in the
product tree; the missing symbol is a fact of the build, so the place to find
it out is a test on a machine that has the extension built. Each target now
declares ``REQUIRED_TRTLLM_OPS`` and this asserts it.

Distinct from ``test_staircase_claims.py``, which is deliberately import-free
and runs anywhere: this one imports the targets, so it needs a real build.
"""

from __future__ import annotations

import importlib

import pytest
import torch

import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
from tensorrt_llm._torch.staircase._router_index import STAIRCASE_ROUTERS, routing_module

_PACKAGE = "tensorrt_llm._torch.staircase"
_ARCHS = sorted(STAIRCASE_ROUTERS)


def _target_modules():
    """(target name, imported modeling module) for every routed target."""
    for arch in _ARCHS:
        routing = routing_module(arch)
        for name, dotted in routing.TARGET_MODULES.items():
            yield name, importlib.import_module(f"{_PACKAGE}.{dotted}")


def _target_ids():
    return [name for name, _ in _target_modules()]


@pytest.mark.parametrize("name,module", list(_target_modules()), ids=_target_ids())
def test_every_declared_op_exists(name, module):
    """A named op that is not registered is a build the target cannot run on."""
    declared = getattr(module, "REQUIRED_TRTLLM_OPS", None)
    assert declared, f"{name}: modeling.py declares no REQUIRED_TRTLLM_OPS"

    missing = [op for op in declared if not hasattr(torch.ops.trtllm, op)]
    assert not missing, (
        f"{name} names {len(missing)} op(s) this build does not register: "
        f"{', '.join(missing)}. Either the op was renamed upstream and the "
        f"target has not followed, or this build predates it."
    )


@pytest.mark.parametrize("name,module", list(_target_modules()), ids=_target_ids())
def test_the_pybind_attention_entry_point_exists(name, module):
    """``thop.attention`` is reached through the bindings, not torch.ops.

    Separate from the loop above because a missing pybind symbol fails in a
    different way -- an ImportError or an AttributeError on the module object
    rather than a missing torch.ops entry -- and both targets go through it.
    """
    from tensorrt_llm.bindings.internal import thop

    assert hasattr(thop, "attention"), (
        f"{name} calls the attention op through tensorrt_llm.bindings.internal"
        f".thop.attention, which this build does not expose"
    )
