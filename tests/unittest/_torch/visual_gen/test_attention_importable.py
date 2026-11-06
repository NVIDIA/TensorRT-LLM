# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Guards the VisualGen attention import graph.

``visual_gen.attention.attention`` imports ``visual_gen.modules.rms_norm``, while
``visual_gen.modules`` re-exports ``Attention`` and ``QKVMode`` from it. The
re-export is resolved lazily so that either package can be imported first; the
order checks run in a fresh interpreter because the order only matters before
anything is cached in ``sys.modules``.

The same file guards the two forwarding modules left behind at the retired
``visual_gen.attention_backend`` and ``visual_gen.modules.attention`` paths: they
must hand back the canonical objects themselves, their re-export list must not
drift away from the canonical one, and importing them must warn.
"""

import importlib
import subprocess
import sys

import pytest

pytestmark = pytest.mark.cpu_only

ATTENTION = "tensorrt_llm._torch.visual_gen.attention.attention"
BACKENDS = "tensorrt_llm._torch.visual_gen.attention.backends"
MODULES = "tensorrt_llm._torch.visual_gen.modules"

# Import paths retired by the VisualGen attention consolidation and kept alive
# by a definition-free forwarding module for the deprecation window.
SHIM_MODULES = [
    "tensorrt_llm._torch.visual_gen.modules.attention",
    "tensorrt_llm._torch.visual_gen.attention_backend",
]


@pytest.mark.parametrize(
    "first, second",
    [(ATTENTION, MODULES), (MODULES, ATTENTION)],
    ids=["attention-first", "modules-first"],
)
def test_import_order_is_free(first, second):
    """Both packages import cleanly whichever one a caller imports first."""
    code = (
        f"import importlib; importlib.import_module({first!r}); importlib.import_module({second!r}); "
        f"import {MODULES} as m, {ATTENTION} as a; "
        "assert m.Attention is a.Attention and m.QKVMode is a.QKVMode"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


def test_modules_reexport_is_canonical():
    modules = importlib.import_module(MODULES)
    attention = importlib.import_module(ATTENTION)
    assert modules.Attention is attention.Attention
    assert modules.QKVMode is attention.QKVMode
    with pytest.raises(AttributeError):
        modules.NotAnAttribute  # noqa: B018


def test_modules_attention_shim_forwards_canonical_objects():
    """The retired module path must hand back the canonical objects, not copies of them."""
    shim = importlib.import_module("tensorrt_llm._torch.visual_gen.modules.attention")
    canonical = importlib.import_module(ATTENTION)
    for name in shim.__all__:
        assert getattr(shim, name) is getattr(canonical, name), name


def test_attention_backend_shim_forwards_canonical_objects():
    """Every name the retired package root re-exports must be the canonical object.

    Identity rather than equality: callers keep isinstance checks against these
    classes, and a re-exported copy would silently fail them.
    """
    shim = importlib.import_module("tensorrt_llm._torch.visual_gen.attention_backend")
    canonical = importlib.import_module(BACKENDS)
    for name in shim.__all__:
        assert getattr(shim, name) is getattr(canonical, name), name


def test_attention_backend_shim_exports_match_canonical():
    """The shim's ``__all__`` must track the canonical package's, as a set."""
    shim = importlib.import_module("tensorrt_llm._torch.visual_gen.attention_backend")
    canonical = importlib.import_module(BACKENDS)
    assert canonical.__all__, "an empty canonical __all__ would make this guard vacuous"
    assert set(shim.__all__) == set(canonical.__all__)


@pytest.mark.parametrize("module_name", SHIM_MODULES)
def test_shim_warns_on_import(module_name):
    """Importing a retired path must warn, and specifically with FutureWarning.

    DeprecationWarning is on Python's stock ignore list outside ``__main__``, so
    it would never reach the out-of-tree callers these modules exist for. The
    entry is dropped from ``sys.modules`` to force one fresh execution of the
    module body and put back afterwards.
    """
    parent_name, _, attribute = module_name.rpartition(".")
    parent = importlib.import_module(parent_name)
    previous = sys.modules.pop(module_name, None)
    try:
        with pytest.warns(FutureWarning, match="has moved to"):
            importlib.import_module(module_name)
    finally:
        if previous is not None:
            sys.modules[module_name] = previous
            setattr(parent, attribute, previous)
