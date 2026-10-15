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
from pathlib import Path

import pytest

import tensorrt_llm._torch._experimental.modeling_v2 as _modeling_v2
import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*

_ROOT = Path(_modeling_v2.__file__).resolve().parent


def _target_modules():
    """(target directory name, imported modeling module) for every target on disk."""
    for modeling in sorted((_ROOT / "models").glob("*/*/modeling.py")):
        target_dir = modeling.parent
        dotted = (
            f"{_modeling_v2.__name__}.models.{target_dir.parent.name}.{target_dir.name}.modeling"
        )
        yield target_dir.name, importlib.import_module(dotted)


def _target_ids():
    return [name for name, _ in _target_modules()]


@pytest.mark.parametrize("name,module", list(_target_modules()), ids=_target_ids())
def test_the_pybind_attention_entry_point_exists(name, module):
    """FallbackFmha dispatches both targets through the native AttentionOp methods."""
    from tensorrt_llm.bindings.internal import thop

    for symbol in ("AttentionOp", "StaticAttentionConfig", "FmhaParams"):
        assert hasattr(thop, symbol), f"{name} requires thop.{symbol}"
    for method in (
        "get_attention_workspace_size",
        "run_context",
        "run_generation",
        "run_mla_generation",
    ):
        assert callable(getattr(thop.AttentionOp, method, None)), (
            f"{name} requires thop.AttentionOp.{method}"
        )
