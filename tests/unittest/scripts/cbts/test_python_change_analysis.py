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
"""Tests for CBTS's shared static Python change analysis (changed_bindings/callers/consumers)."""

from __future__ import annotations

import pytest

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
from cbts.coverage.selection.python_change_analysis import analyze_python_changes
from cbts.rules._helpers import iter_diff_deleted_post_lines, iter_diff_post_line_numbers

pytestmark = pytest.mark.cpu_only


def _analyze(source: str, diff: str):
    return analyze_python_changes(
        source,
        iter_diff_post_line_numbers(diff),
        iter_diff_deleted_post_lines(diff),
    )


def test_literal_assignment_resolves_consumers_and_callers_without_name_policy() -> None:
    source = (
        "VALUE = 521\n\ndef helper():\n    return VALUE\n\ndef caller():\n    return helper()\n"
    )
    analysis = _analyze(source, "@@ -0,0 +1 @@\n+VALUE = 521\n")

    assert not analysis.limitation
    assert analysis.changed_bindings == {"VALUE"}
    assert analysis.binding_consumers == {"helper"}
    assert analysis.callers == {"helper": {"caller"}}


@pytest.mark.parametrize(
    ("source", "reason"),
    (
        ("_VALUE = create_value()\n", "effectful module statement"),
        ("class Config:\n    value = 521\n", "class/signature import change"),
    ),
)
def test_unsupported_import_time_changes_remain_fail_closed(source: str, reason: str) -> None:
    changed_line = 2 if source.startswith("class") else 1
    diff = f"@@ -0,0 +{changed_line} @@\n+changed\n"

    analysis = _analyze(source, diff)

    assert analysis.limitation == reason


def test_replaced_literal_is_resolved_when_both_images_are_literals() -> None:
    source = "_VALUE = 521\n"
    diff = "@@ -1 +1 @@\n-_VALUE = 520\n+_VALUE = 521\n"

    analysis = _analyze(source, diff)

    assert not analysis.limitation


def test_replaced_effectful_assignment_remains_fail_closed() -> None:
    source = "_VALUE = 521\n"
    diff = "@@ -1 +1 @@\n-_VALUE = create_value()\n+_VALUE = 521\n"

    analysis = _analyze(source, diff)

    assert analysis.limitation == "unresolved import replacement"


@pytest.mark.parametrize("body", ("", "# explanatory comment"))
def test_added_module_noop_line_is_ignored(body: str) -> None:
    source = f"VALUE = 521\n{body}\n"
    diff = f"@@ -1 +1,2 @@\n VALUE = 521\n+{body}\n"

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert not analysis.changed_bindings


@pytest.mark.parametrize("body", ("", "# obsolete comment"))
def test_deleted_module_noop_line_is_ignored(body: str) -> None:
    source = "VALUE = 521\n"
    diff = f"@@ -1,2 +1 @@\n VALUE = 521\n-{body}\n"

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert not analysis.changed_bindings


def test_added_decorator_line_remains_fail_closed() -> None:
    source = "@decorate\ndef consumer():\n    return 1\n"
    diff = "@@ -1,0 +1 @@\n+@decorate\n"

    analysis = _analyze(source, diff)

    assert analysis.limitation == "effectful module statement"


def test_postponed_annotated_literal_is_resolved() -> None:
    source = "from __future__ import annotations\n\n_CACHE: dict[str, object] = {}\n"
    diff = "@@ -2,0 +3 @@\n+_CACHE: dict[str, object] = {}\n"

    analysis = _analyze(source, diff)

    assert not analysis.limitation


def test_builtin_import_is_safe_when_it_is_only_added() -> None:
    source = "import sys\n\ndef shutdown():\n    return sys.modules\n"
    diff = "@@ -0,0 +1 @@\n+import sys\n"

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.binding_consumers == {"shutdown"}


def test_builtin_import_replacement_remains_fail_closed() -> None:
    source = "import _thread as runtime\n"
    diff = "@@ -1 +1 @@\n-import sys as runtime\n+import _thread as runtime\n"

    analysis = _analyze(source, diff)

    assert analysis.limitation == "unresolved import replacement"


def test_literal_assignment_resolves_class_method_consumer() -> None:
    source = "_VALUE = 521\n\nclass Config:\n    def check(self):\n        return _VALUE\n"

    analysis = _analyze(source, "@@ -0,0 +1 @@\n+_VALUE = 521\n")

    assert not analysis.limitation
    assert analysis.binding_consumers == {"Config.check"}


def test_literal_assignment_attributes_closure_consumer_to_outer() -> None:
    source = (
        "_VALUE = 521\n\ndef outer():\n    def inner():\n        return _VALUE\n    return inner\n"
    )

    analysis = _analyze(source, "@@ -0,0 +1 @@\n+_VALUE = 521\n")

    assert not analysis.limitation
    assert analysis.binding_consumers == {"outer"}


def test_import_time_consumer_remains_fail_closed() -> None:
    source = "_VALUE = 521\n\n_ALIAS = _VALUE\n"

    analysis = _analyze(source, "@@ -0,0 +1 @@\n+_VALUE = 521\n")

    assert analysis.limitation == "import-time binding consumer"


def test_plain_function_declaration_with_postponed_annotations_is_safe() -> None:
    source = (
        "from __future__ import annotations\n\ndef _helper(value: object) -> int:\n    return 1\n"
    )
    diff = "@@ -2,0 +3,2 @@\n+def _helper(value: object) -> int:\n+    return 1\n"

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.binding_consumers == {"_helper"}


def test_local_shadow_does_not_create_direct_caller_edge() -> None:
    source = (
        "_VALUE = 521\n"
        "\n"
        "def _helper():\n"
        "    return _VALUE\n"
        "\n"
        "def caller(_helper):\n"
        "    return _helper()\n"
    )

    analysis = _analyze(source, "@@ -0,0 +1 @@\n+_VALUE = 521\n")

    assert not analysis.limitation
    assert analysis.callers == {}


def test_local_shadow_does_not_create_binding_consumer() -> None:
    source = "VALUE = 521\n\ndef helper(VALUE):\n    return VALUE\n"

    analysis = _analyze(source, "@@ -0,0 +1 @@\n+VALUE = 521\n")

    assert not analysis.limitation
    assert analysis.binding_consumers == set()


def test_comprehension_target_does_not_shadow_outer_iterable_load() -> None:
    source = "VALUE = (1, 2, 3)\n\ndef consumer():\n    return [VALUE for VALUE in VALUE]\n"

    analysis = _analyze(source, "@@ -0,0 +1 @@\n+VALUE = (1, 2, 3)\n")

    assert not analysis.limitation
    assert analysis.binding_consumers == {"consumer"}


def test_function_alias_marks_caller_graph_incomplete() -> None:
    source = (
        "VALUE = 521\n\n"
        "def helper():\n"
        "    return VALUE\n\n"
        "def caller():\n"
        "    alias = helper\n"
        "    return alias()\n"
    )

    analysis = _analyze(source, "@@ -0,0 +1 @@\n+VALUE = 521\n")

    assert analysis.binding_consumers == {"helper"}
    assert analysis.callers == {}
    assert analysis.callable_escapes == {"helper"}
