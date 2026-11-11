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
"""``MUSESPARK_OPEN_SM103_TEP4_EXTRA_PATHS`` on the sm103 TEP4 composition, on the host only.

``levers.py`` and ``dispatch.py`` are loaded from their files with the three leaf modules
``dispatch`` imports stubbed; the compositions are ``modeling.py``'s ``_SM10*`` / ``_TEP*`` /
``TARGET_PATHS`` statements run over that ``dispatch``. The tests pin:

* unset, empty and blank values leave both compositions exactly as without the variable;
* the four allowed names are added to the TEP4 composition only, closed under ``depends_on``;
* the lever table resolves them on under ``auto`` (and ``off`` removes them), and
  ``MUSESPARK_OPEN_LEVER_<NAME>=on`` raises without the variable;
* a name outside the allowed set raises.
"""

from __future__ import annotations

import ast
import enum
import importlib
import re
import sys
import types
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[7]
_TARGET = (
    _ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/musespark_open"
    "/muse_spark_open_816b_a42b_nvfp4__sm_103__tep"
)
_PACKAGE = "_musespark_open_tep4_extra_paths_under_test"
ENV = "MUSESPARK_OPEN_SM103_TEP4_EXTRA_PATHS"
EXTRA = frozenset(
    {"early_expert_ids", "pool_into_route", "unpool_into_boundary", "attention_output_into_hop"}
)


class _Mode(str, enum.Enum):
    OFF = "off"
    AUTO = "auto"
    REQUIRE = "require"


class _Log:
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    return module


def _load():
    stubs = {
        "tensorrt_llm._torch._experimental.modeling_v2._router_index": _module(
            "_router_index", ModelingV2Mode=_Mode
        ),
        "tensorrt_llm._torch.distributed.communicator": _module("communicator", Distributed=object),
        "tensorrt_llm.logger": _module("logger", logger=_Log()),
    }
    saved = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        package = types.ModuleType(_PACKAGE)
        package.__path__ = [str(_TARGET)]
        sys.modules[_PACKAGE] = package
        levers = importlib.import_module(f"{_PACKAGE}.levers")
        dispatch = importlib.import_module(f"{_PACKAGE}.dispatch")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    return levers, dispatch


def _assigned(node) -> list:
    if isinstance(node, ast.Assign):
        return [target.id for target in node.targets if isinstance(target, ast.Name)]
    if isinstance(node, (ast.AnnAssign, ast.AugAssign)) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


LEVERS, DISPATCH = _load()


def _namespace() -> dict:
    """``modeling.py``'s composition statements, run now over ``dispatch`` (the environment is
    read as they run)."""
    path = _TARGET / "modeling.py"
    wanted = re.compile(r"^(_(TEP|SM10|PREFETCH)\w*|TARGET_PATHS)$")
    body = [
        node
        for node in ast.parse(path.read_text()).body
        if _assigned(node) and all(wanted.match(name) for name in _assigned(node))
    ]
    namespace = {"_dispatch": DISPATCH}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def _compose(monkeypatch, value=None) -> dict:
    if value is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, value)
    return _namespace()


def _table(namespace, knobs=None, arena="windowed"):
    return DISPATCH.table_for(namespace["TARGET_PATHS"], 103, 4, 1, "agg", arena, dict(knobs or {}))


@pytest.mark.parametrize("value", [None, "", " ", ",", " , ,"])
def test_no_names_leave_the_compositions_unchanged(monkeypatch, value):
    namespace = _compose(monkeypatch, value)
    paths = namespace["TARGET_PATHS"]
    assert paths[(103, 4)].served == namespace["_SM103_TEP4_PATHS"]
    assert paths[(103, 8)].served == namespace["_TEP8_PATHS"] | namespace["_PREFETCH_LEVERS"]
    assert not EXTRA & paths[(103, 4)].served


@pytest.mark.parametrize(
    "value", [",".join(sorted(EXTRA)), " attention_output_into_hop , pool_into_route,"]
)
def test_the_names_are_added_to_the_tep4_composition_only(monkeypatch, value):
    base = _compose(monkeypatch)
    namespace = _compose(monkeypatch, value)
    names = {item.strip() for item in value.split(",") if item.strip()}
    paths = namespace["TARGET_PATHS"]
    assert paths[(103, 4)].served == namespace["_SM103_TEP4_PATHS"] | names
    assert paths[(103, 8)].served == base["TARGET_PATHS"][(103, 8)].served
    # TargetPaths closes the served set under depends_on when it is built, so reaching here
    # means every name's dependencies are served.
    assert DISPATCH.closure(paths[(103, 4)].served) == paths[(103, 4)].served


@pytest.mark.parametrize("arena", ["windowed", "dense", "paged"])
def test_the_lever_table_serves_the_names_on_under_auto(monkeypatch, arena):
    base = _table(_compose(monkeypatch), arena=arena)
    table = _table(_compose(monkeypatch, ",".join(EXTRA)), arena=arena)
    for name in EXTRA:
        assert name not in base.paths()
        assert table.on(name), name
    assert table.paths() - EXTRA == base.paths()


def test_the_lever_knob_turns_a_served_name_off_and_raises_where_it_is_not_served(monkeypatch):
    knob = LEVERS.knob_name("early_expert_ids")
    base = _table(_compose(monkeypatch))
    with pytest.raises(ValueError, match="early_expert_ids"):
        _table(_compose(monkeypatch), {knob: "on"})
    namespace = _compose(monkeypatch, "early_expert_ids")
    assert _table(namespace, {knob: "on"}).on("early_expert_ids")
    off = _table(namespace, {knob: "off"})
    assert not off.on("early_expert_ids")
    assert off.paths() == base.paths()


@pytest.mark.parametrize("name", ["dense_gemm_prefetch", "next_qkv_prefetch", "no_such_path"])
def test_a_name_outside_the_allowed_set_raises(monkeypatch, name):
    with pytest.raises(ValueError, match=name):
        _compose(monkeypatch, f"early_expert_ids,{name}")
