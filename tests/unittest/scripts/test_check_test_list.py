#!/usr/bin/env python3
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
"""Unit tests for scripts/check_test_list.py AST-based param-ID validation.

These are pure-Python (no GPU, no built wheel) and are marked ``cpu_only`` so
they run on the CPU CI stage.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "check_test_list.py"


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("check_test_list", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _index(mod, tmp_path, src: str, name: str = "mod_under_test.py"):
    """Write ``src`` to a temp file and return build_ast_index's result."""
    p = tmp_path / name
    p.write_text(src, encoding="utf-8")
    return mod.build_ast_index(str(p))


def _valid_ids(mod, tmp_path, src: str, func: str = "test_f"):
    classes, _top, top_nodes, _cmn, consts = _index(mod, tmp_path, src)
    return mod._compute_valid_param_ids(top_nodes[func], None, consts)


# --------------------------------------------------------------------------
# _compute_valid_param_ids / _get_parametrize_ids
# --------------------------------------------------------------------------


def test_literal_list_ids(mod, tmp_path):
    src = 'import pytest\n@pytest.mark.parametrize("x", ["a", "b"])\ndef test_f(x): pass\n'
    assert _valid_ids(mod, tmp_path, src) == {"a", "b"}


def test_explicit_ids_kwarg(mod, tmp_path):
    # Non-literal argvalues, but explicit string ids= make it resolvable.
    src = (
        "import pytest\n"
        '@pytest.mark.parametrize("x", [object(), object()], '
        'ids=["a", "b"])\n'
        "def test_f(x): pass\n"
    )
    assert _valid_ids(mod, tmp_path, src) == {"a", "b"}


def test_callable_ids_kwarg_punts(mod, tmp_path):
    src = (
        "import pytest\n"
        '@pytest.mark.parametrize("x", [1, 2], ids=lambda v: str(v))\n'
        "def test_f(x): pass\n"
    )
    assert _valid_ids(mod, tmp_path, src) is None


def test_pytest_param_id(mod, tmp_path):
    src = (
        "import pytest\n"
        '@pytest.mark.parametrize("a,b", '
        '[pytest.param(1, 2, id="p"), pytest.param(3, 4, id="q")])\n'
        "def test_f(a, b): pass\n"
    )
    assert _valid_ids(mod, tmp_path, src) == {"p", "q"}


def test_multi_arg_tuple_rows(mod, tmp_path):
    # pytest joins per-arg scalar IDs with "-".
    src = (
        "import pytest\n"
        '@pytest.mark.parametrize("a,b", [(1, "x"), (2, "y")])\n'
        "def test_f(a, b): pass\n"
    )
    assert _valid_ids(mod, tmp_path, src) == {"1-x", "2-y"}


def test_single_argname_tuple_value_punts(mod, tmp_path):
    # A tuple bound to ONE argname gets a positional "x0" ID we cannot
    # reproduce statically — must punt, not guess.
    src = 'import pytest\n@pytest.mark.parametrize("x", [(1, 2), (3, 4)])\ndef test_f(x): pass\n'
    assert _valid_ids(mod, tmp_path, src) is None


def test_non_scalar_tuple_element_punts(mod, tmp_path):
    src = (
        'import pytest\n@pytest.mark.parametrize("a,b", [(1, object())])\ndef test_f(a, b): pass\n'
    )
    assert _valid_ids(mod, tmp_path, src) is None


def test_id_collision_punts(mod, tmp_path):
    # pytest would suffix the duplicate as "a0"/"a1"; we cannot, so punt.
    src = 'import pytest\n@pytest.mark.parametrize("x", ["a", "a"])\ndef test_f(x): pass\n'
    assert _valid_ids(mod, tmp_path, src) is None


def test_stacked_decorators_product(mod, tmp_path):
    src = (
        "import pytest\n"
        '@pytest.mark.parametrize("x", ["a", "b"])\n'
        '@pytest.mark.parametrize("y", ["c", "d"])\n'
        "def test_f(x, y): pass\n"
    )
    assert _valid_ids(mod, tmp_path, src) == {"c-a", "c-b", "d-a", "d-b"}


def test_no_parametrize_returns_empty(mod, tmp_path):
    src = "def test_f(): pass\n"
    assert _valid_ids(mod, tmp_path, src) == set()


# --------------------------------------------------------------------------
# module-level constant resolution (build_ast_index + Name following)
# --------------------------------------------------------------------------


def test_name_argvalues_resolved(mod, tmp_path):
    src = (
        "import pytest\n"
        'PARAMS = ["a", "b"]\n'
        '@pytest.mark.parametrize("x", PARAMS)\n'
        "def test_f(x): pass\n"
    )
    assert _valid_ids(mod, tmp_path, src) == {"a", "b"}


def test_name_transitive_resolved(mod, tmp_path):
    src = (
        "import pytest\n"
        'BASE = ["a", "b"]\n'
        "PARAMS = BASE\n"
        '@pytest.mark.parametrize("x", PARAMS)\n'
        "def test_f(x): pass\n"
    )
    assert _valid_ids(mod, tmp_path, src) == {"a", "b"}


def test_reassigned_name_dropped(mod, tmp_path):
    # Assigned twice -> unsound to resolve -> punt.
    src = (
        "import pytest\n"
        'PARAMS = ["a"]\n'
        'PARAMS = ["b"]\n'
        '@pytest.mark.parametrize("x", PARAMS)\n'
        "def test_f(x): pass\n"
    )
    assert _valid_ids(mod, tmp_path, src) is None


def test_augassigned_name_dropped(mod, tmp_path):
    src = (
        "import pytest\n"
        'PARAMS = ["a"]\n'
        'PARAMS += ["b"]\n'
        '@pytest.mark.parametrize("x", PARAMS)\n'
        "def test_f(x): pass\n"
    )
    assert _valid_ids(mod, tmp_path, src) is None


def test_call_argvalues_punts(mod, tmp_path):
    src = (
        "import pytest\n"
        'def gen(): return ["a"]\n'
        '@pytest.mark.parametrize("x", gen())\n'
        "def test_f(x): pass\n"
    )
    assert _valid_ids(mod, tmp_path, src) is None


def test_build_ast_index_module_consts(mod, tmp_path):
    src = "SINGLE = [1]\nDOUBLE = [1]\nDOUBLE = [2]\nAUG = [1]\nAUG += [2]\n"
    _c, _t, _tn, _cmn, consts = _index(mod, tmp_path, src)
    assert "SINGLE" in consts
    assert "DOUBLE" not in consts
    assert "AUG" not in consts


# --------------------------------------------------------------------------
# validate_test_lists end-to-end (errors vs unverifiable buckets)
# --------------------------------------------------------------------------


def _make_layout(tmp_path, source: str, list_lines: list[str]):
    """Create a defs source file + a test-list dir; return (lists_dir, base)."""
    base = tmp_path / "tests" / "integration" / "defs"
    base.mkdir(parents=True)
    (base / "test_sample.py").write_text(source, encoding="utf-8")
    lists_dir = tmp_path / "lists"
    lists_dir.mkdir()
    (lists_dir / "l0_sample.txt").write_text("\n".join(list_lines) + "\n", encoding="utf-8")
    return str(lists_dir), str(base)


def test_validate_flags_invalid_id(mod, tmp_path):
    source = 'import pytest\n@pytest.mark.parametrize("x", ["a", "b"])\ndef test_f(x): pass\n'
    lists_dir, base = _make_layout(tmp_path, source, ["test_sample.py::test_f[zzz]"])
    errors, unverifiable, accepted, rejected = mod.validate_test_lists(lists_dir, base)
    assert any("INVALID PARAMETRIZE ID" in e for e in errors)
    assert unverifiable == []
    assert accepted == []
    assert rejected == [("test_sample.py", None, "test_f", "zzz")]


def test_validate_accepts_valid_id(mod, tmp_path):
    source = (
        "import pytest\n"
        'PARAMS = ["a", "b"]\n'
        '@pytest.mark.parametrize("x", PARAMS)\n'
        "def test_f(x): pass\n"
    )
    lists_dir, base = _make_layout(tmp_path, source, ["test_sample.py::test_f[a]"])
    errors, unverifiable, accepted, rejected = mod.validate_test_lists(lists_dir, base)
    assert errors == []
    assert unverifiable == []
    assert accepted == [("test_sample.py", None, "test_f", "a")]
    assert rejected == []


def test_validate_reports_unverifiable(mod, tmp_path):
    source = (
        "import pytest\n"
        '@pytest.mark.parametrize("x", [1, 2], ids=lambda v: str(v))\n'
        "def test_f(x): pass\n"
    )
    lists_dir, base = _make_layout(tmp_path, source, ["test_sample.py::test_f[1]"])
    errors, unverifiable, accepted, rejected = mod.validate_test_lists(lists_dir, base)
    assert errors == []
    assert len(unverifiable) == 1
    rel, cls, func, pid, reason, _refs = unverifiable[0]
    assert func == "test_f" and pid == "1"
    assert reason == "ids=callable-or-nonliteral"
    # An unverifiable entry is neither accepted nor rejected -- it has no static
    # verdict, so it must not enter the parity buckets.
    assert accepted == []
    assert rejected == []


def test_write_unverifiable_report(mod, tmp_path):
    unverifiable = [
        ("dir/test_a.py", None, "test_f", "p1", "argvalues=call-result", ["lists/l0.txt:3"]),
    ]
    out = tmp_path / "report.txt"
    mod.write_unverifiable_report(unverifiable, str(out))
    text = out.read_text(encoding="utf-8")
    assert "argvalues=call-result" in text
    assert "dir/test_a.py::test_f[p1]" in text


# --------------------------------------------------------------------------
# validate <-> collection parity (compute_parity / load_collectable_entries)
# --------------------------------------------------------------------------


def test_compute_parity_all_collectable(mod):
    accepted = [("test_a.py", None, "test_f", "a"), ("test_a.py", "TestC", "test_g", "b")]
    collectable = set(accepted)
    false_confidence, false_alarm = mod.compute_parity(accepted, [], collectable)
    assert false_confidence == []
    assert false_alarm == []


def test_compute_parity_false_confidence(mod):
    # Accepted by --validate but pytest cannot collect it -> gate-worthy.
    accepted = [("test_a.py", None, "test_f", "a"), ("test_a.py", None, "test_f", "ghost")]
    collectable = {("test_a.py", None, "test_f", "a")}
    false_confidence, false_alarm = mod.compute_parity(accepted, [], collectable)
    assert false_confidence == [("test_a.py", None, "test_f", "ghost")]
    assert false_alarm == []


def test_compute_parity_false_alarm(mod):
    # Rejected by --validate but pytest does collect it -> resolver too strict.
    rejected = [("test_a.py", None, "test_f", "real")]
    collectable = {("test_a.py", None, "test_f", "real")}
    false_confidence, false_alarm = mod.compute_parity([], rejected, collectable)
    assert false_confidence == []
    assert false_alarm == [("test_a.py", None, "test_f", "real")]


def test_compute_parity_rejected_not_collectable_is_silent(mod):
    # Rejected and genuinely not collectable -> validator was right; no parity
    # finding in either bucket.
    rejected = [("test_a.py", None, "test_f", "zzz")]
    false_confidence, false_alarm = mod.compute_parity([], rejected, set())
    assert false_confidence == []
    assert false_alarm == []


def test_load_collectable_entries_reads_both_lists(mod, tmp_path):
    (tmp_path / "l0_test.txt").write_text(
        "accuracy/test_x.py::TestA::test_f[a]\n"
        "# a comment line\n"
        "full:GH200/accuracy/test_x.py::test_g[b] TIMEOUT 90\n",
        encoding="utf-8",
    )
    (tmp_path / "qa_test.txt").write_text("accuracy/test_y.py::test_h[c]\n", encoding="utf-8")
    collectable = mod.load_collectable_entries(str(tmp_path))
    assert collectable == {
        ("accuracy/test_x.py", "TestA", "test_f", "a"),
        # full:GH200/ hardware prefix and the trailing TIMEOUT marker are
        # normalized away by parse_test_entry, matching the validator's tuples.
        ("accuracy/test_x.py", None, "test_g", "b"),
        ("accuracy/test_y.py", None, "test_h", "c"),
    }


def test_load_collectable_entries_missing_returns_none(mod, tmp_path):
    assert mod.load_collectable_entries(str(tmp_path)) is None
