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
"""Standalone loader for the bounce_v2 pure-logic package.

The package (tensorrt_llm/_torch/disaggregation/bounce_v2) is stdlib+numpy
only and uses relative imports, so it is importable WITHOUT the compiled
tensorrt_llm bindings. Importing it through the ``tensorrt_llm`` namespace
would pull in the bindings (absent in a source tree), so we load it under the
top-level name ``bounce_v2`` directly from its files instead.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

# conftest.py lives at tests/unittest/disaggregated/bounce_v2/; counting up:
# [0]=bounce_v2, [1]=disaggregated, [2]=unittest, [3]=tests, [4]=repo root.
_PKG = (
    Path(__file__).resolve().parents[4] / "tensorrt_llm" / "_torch" / "disaggregation" / "bounce_v2"
)


def load_bounce_v2() -> types.ModuleType:
    """Load (once) and return the bounce_v2 package as a standalone module."""
    if "bounce_v2" in sys.modules:
        return sys.modules["bounce_v2"]
    spec = importlib.util.spec_from_file_location(
        "bounce_v2",
        _PKG / "__init__.py",
        submodule_search_locations=[str(_PKG)],
    )
    assert spec is not None and spec.loader is not None, f"cannot load {_PKG}"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["bounce_v2"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="session")
def bounce_v2() -> types.ModuleType:
    """The bounce_v2 package, loaded standalone (no tensorrt_llm import)."""
    return load_bounce_v2()
