# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The pybind surface a target reaches for has to exist in the build.

A target reads private engine surface with no version pin, so the only thing
that can tell you the surface moved is running against it.

Custom ops are not checked here. Naming them took a hand-kept list per target
that nothing could prove complete: an op added to a forward and left out of the
list cost nothing to forget, and a stale entry still passed because the op was
still registered upstream. The forward reaches every op it needs anyway, so a
renamed op fails there -- at the call, which says where. A list that has to be
maintained to restate that is not worth its own upkeep.

Distinct from ``test_modeling_v2_claims.py``, which is deliberately import-free
and runs anywhere: this one imports the targets, so it needs a real build.
"""

from __future__ import annotations

import importlib

import pytest

from tensorrt_llm._torch._experimental.modeling_v2._router_index import (
    MODELING_V2_ROUTERS,
    routing_module,
)

_PACKAGE = "tensorrt_llm._torch._experimental.modeling_v2"
_ARCHS = sorted(MODELING_V2_ROUTERS)


def _target_modules():
    """(target name, imported modeling module) for every routed target."""
    for arch in _ARCHS:
        routing = routing_module(arch)
        for name, dotted in routing.TARGET_MODULES.items():
            yield name, importlib.import_module(f"{_PACKAGE}.{dotted}")


def _target_ids():
    return [name for name, _ in _target_modules()]


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
