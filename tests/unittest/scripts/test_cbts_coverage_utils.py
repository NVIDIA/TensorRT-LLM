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

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
COVERAGE_UTILS = REPO_ROOT / "jenkins" / "scripts" / "cbts" / "coverage_utils"
CBTS_PYSTART_PATH = COVERAGE_UTILS / "cbts_pystart.py"


def _load_cbts_pystart() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_cbts_pystart_test", CBTS_PYSTART_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CBTS_PYSTART = _load_cbts_pystart()
PyStartTracker = CBTS_PYSTART.PyStartTracker
_product_imports_settled = CBTS_PYSTART._product_imports_settled


def _product_module(name: str, initializing: bool) -> ModuleType:
    module = ModuleType(name)
    module.__spec__ = SimpleNamespace(_initializing=initializing)
    return module


def test_product_imports_settled_waits_for_full_module_chain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    product = "_cbts_test_product"
    child = f"{product}.child"

    assert not _product_imports_settled({product})

    monkeypatch.setitem(sys.modules, product, _product_module(product, initializing=False))
    monkeypatch.setitem(sys.modules, child, _product_module(child, initializing=True))
    assert not _product_imports_settled({product})

    sys.modules[child].__spec__._initializing = False
    assert _product_imports_settled({product})


def test_tracker_skips_unchanged_save_and_resaves_after_fork(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tracker = PyStartTracker([], str(tmp_path), stage="test")
    tracker.record_outcome("test_node", "passed")

    first_path = tracker.save()
    assert first_path is not None
    assert Path(first_path).is_file()
    assert tracker.save() is None

    # Re-recording the same metadata does not dirty the snapshot.
    tracker.record_outcome("test_node", "passed")
    assert tracker.save() is None

    # A forked child inherits the revision state but must still write its PID-specific artifact.
    child_pid = 987654
    monkeypatch.setattr(CBTS_PYSTART.os, "getpid", lambda: child_pid)
    child_path = tracker.save()
    assert child_path is not None
    assert f".pid{child_pid}.sqlite" in child_path
    assert Path(child_path).is_file()
