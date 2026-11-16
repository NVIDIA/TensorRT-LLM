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

import ast
import difflib
from pathlib import Path

import pytest

from ._dataclass_helpers import ADDED_DATACLASS, added_dataclass_change

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
from cbts.coverage.selection.python_change_analysis import (
    ImportTarget,
    analyze_python_changes,
    qualnames_for_lines,
)
from cbts.rules._helpers import (
    iter_diff_deleted_post_lines,
    iter_diff_post_line_numbers,
    reconstruct_diff_pre_image,
)

REPO_ROOT = Path(__file__).resolve().parents[4]

pytestmark = pytest.mark.cpu_only


def _analyze(source: str, diff: str):
    return analyze_python_changes(
        source,
        iter_diff_post_line_numbers(diff),
        iter_diff_deleted_post_lines(diff),
        pre_source=reconstruct_diff_pre_image(source, diff),
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


@pytest.mark.parametrize("delete_binding", (False, True))
@pytest.mark.parametrize(
    "consumer",
    (
        "def helper():\n    return C.VALUE\n",
        "class Kernel:\n    def kern(self):\n        return C.VALUE\n",
        "OBSERVED = C.VALUE\n",
        "def helper():\n    alias = C\n    return alias.VALUE\n",
        "def helper():\n    def nested():\n        return C.VALUE\n    return nested\n",
    ),
)
def test_module_alias_binding_consumer_remains_fail_closed(
    consumer: str, delete_binding: bool
) -> None:
    prefix = "import sys\nC = sys.modules[__name__]\n"
    suffix = "UNCHANGED = 0\n" + consumer
    before = prefix + "VALUE = 1\n" + suffix
    source = prefix + ("" if delete_binding else "VALUE = 2\n") + suffix
    diff = "".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True)))

    assert _analyze(source, diff).limitation == "unresolved module attribute consumer"


def test_pre_image_module_alias_consumer_remains_fail_closed() -> None:
    prefix = "import sys\nC = sys.modules[__name__]\n"
    before = prefix + "VALUE = 1\ndef helper():\n    return C.VALUE\n"
    source = prefix + "VALUE = 2\ndef helper():\n    return 0\n"
    diff = "".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True)))

    assert _analyze(source, diff).limitation == "unresolved module attribute consumer"


def test_repository_gvr_module_alias_constant_change_remains_fail_closed() -> None:
    path = "tensorrt_llm/_torch/kernels/blackwell/top_k/gvr_topk_decode_self_sampling.py"
    before = (REPO_ROOT / path).read_text()
    source = before.replace("SENT_LO = -3.0e38", "SENT_LO = -2.0e38", 1)
    assert source != before
    diff = "".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True)))

    assert _analyze(source, diff).limitation == "unresolved module attribute consumer"


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


def test_added_literal_class_attribute_resolves_class_and_method() -> None:
    source = (
        "from typing import ClassVar, Optional\n\n"
        "class Example:\n"
        "    VALUE: ClassVar[Optional[str]] = None\n\n"
        "    def value(self):\n"
        "        return self.VALUE\n"
    )
    diff = (
        "@@ -1,6 +1,7 @@\n"
        " from typing import ClassVar, Optional\n"
        " \n"
        " class Example:\n"
        "+    VALUE: ClassVar[Optional[str]] = None\n"
        " \n"
        "     def value(self):\n"
        "         return self.VALUE\n"
    )

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.binding_consumers == {"Example", "Example.value"}


def test_conditional_class_resolves_added_attribute_method_and_docstring() -> None:
    source = (
        'if AVAILABLE:\n    class Example:\n        "new docs"\n'
        "        CACHE: dict = {}\n\n"
        "        def resolve(self, key: int) -> int:\n"
        "            return self.CACHE[key]\n"
    )
    diff = (
        "@@ -1,3 +1,7 @@\n if AVAILABLE:\n     class Example:\n"
        '-        "old docs"\n+        "new docs"\n'
        "+        CACHE: dict = {}\n+\n"
        "+        def resolve(self, key: int) -> int:\n"
        "+            return self.CACHE[key]\n"
    )

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.binding_consumers == {"Example", "Example.resolve"}


def test_conditional_function_is_included_as_binding_consumer() -> None:
    source = "VALUE = 2\nif AVAILABLE:\n    def helper():\n        return VALUE\n"
    diff = "@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n"

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.binding_consumers == {"helper"}


@pytest.mark.parametrize("declaration", ("class Example:", "def helper():"))
def test_ambiguous_conditional_definition_remains_fail_closed(declaration: str) -> None:
    source = (
        f"if AVAILABLE:\n    {declaration}\n        VALUE = 1\n"
        f"else:\n    {declaration}\n        VALUE = 2\n"
    )
    diff = f"@@ -2,2 +2,2 @@\n     {declaration}\n-        VALUE = 0\n+        VALUE = 1\n"
    if declaration.startswith("def"):
        diff = f"@@ -2 +2 @@\n-    def helper(old):\n+    {declaration}\n"

    analysis = _analyze(source, diff)

    assert analysis.limitation == "ambiguous class/function declaration"


def test_changed_condition_is_not_resolved_by_conditional_class_support() -> None:
    source = "if check_available():\n    class Example:\n        VALUE = 1\n"
    diff = "@@ -1 +1 @@\n-if AVAILABLE:\n+if check_available():\n"

    analysis = _analyze(source, diff)

    assert analysis.limitation == "effectful module statement"


def test_changed_binding_with_ambiguous_consumer_resolves_to_the_shared_qualname() -> None:
    """Only one branch of an ambiguously-named def reads the changed binding.

    Both branches share one qualname in the collector too (same `co_qualname`),
    so a DB lookup on it already returns the sound union of whichever branch a
    test actually touched; no need to decline just because two source
    definitions happen to share the name.
    """
    source = (
        "VALUE = 2\nif AVAILABLE:\n    def helper():\n        return VALUE\n"
        "else:\n    def helper():\n        return 0\n"
    )

    analysis = _analyze(source, "@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n")

    assert not analysis.limitation
    assert analysis.binding_consumers == {"helper"}


def test_unrelated_property_accessors_do_not_block_literal_binding_change() -> None:
    source = (
        "VALUE = 2\n"
        "class Example:\n"
        "    @property\n"
        "    def state(self):\n"
        "        return self._state\n"
        "    @state.setter\n"
        "    def state(self, value):\n"
        "        self._state = value\n"
        "def helper():\n"
        "    return VALUE\n"
    )

    analysis = _analyze(source, "@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n")

    assert not analysis.limitation
    assert analysis.binding_consumers == {"helper"}


@pytest.mark.parametrize("consumer", ("getter", "setter"))
def test_changed_binding_consumed_by_property_accessor_resolves_to_the_shared_qualname(
    consumer: str,
) -> None:
    """A property's getter and setter share one qualname (`Class.state`) by construction.

    Only one of the two reads the changed binding here, same shape as the
    conditional-definition case above; the shared qualname's DB rows are
    still a sound union over both.
    """
    getter = "VALUE" if consumer == "getter" else "self._state"
    setter = "VALUE" if consumer == "setter" else "value"
    source = (
        "VALUE = 2\n"
        "class Example:\n"
        "    @property\n"
        "    def state(self):\n"
        f"        return {getter}\n"
        "    @state.setter\n"
        "    def state(self, value):\n"
        f"        self._state = {setter}\n"
    )

    analysis = _analyze(source, "@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n")

    assert not analysis.limitation
    assert analysis.binding_consumers == {"Example.state"}


def test_removed_ambiguous_definition_is_not_treated_as_a_new_declaration() -> None:
    source = "def helper(value: int):\n    return value\n"
    diff = (
        "@@ -1,6 +1,2 @@\n"
        "-if AVAILABLE:\n"
        "-    def helper():\n"
        "-        return 0\n"
        "-else:\n"
        "-    def helper(value: int):\n"
        "-        return value\n"
        "+def helper(value: int):\n"
        "+    return value\n"
    )

    analysis = _analyze(source, diff)

    assert analysis.limitation == "ambiguous class/function declaration"


def test_conditional_callers_include_every_branch_with_an_unrelated_binding_change() -> None:
    source = (
        "VALUE = 2\n"
        "def helper():\n    return 1\n"
        "if AVAILABLE:\n    def caller():\n        return helper()\n"
        "else:\n    def caller():\n        return 0\n"
    )
    diff = "@@ -1 +1 @@\n-VALUE = 1\n+VALUE = 2\n"

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert not analysis.binding_consumers
    assert analysis.callers == {"helper": {"caller"}}


@pytest.mark.parametrize(
    "declaration",
    (
        "VALUE = factory()",
        "VALUE: CustomType = None",
    ),
)
def test_unsafe_added_class_attribute_remains_fail_closed(declaration: str) -> None:
    source = f"class Example:\n    pass\n    {declaration}\n"
    diff = f"@@ -1,2 +1,3 @@\n class Example:\n     pass\n+    {declaration}\n"

    analysis = _analyze(source, diff)

    assert analysis.limitation == "class/signature import change"


def test_replaced_literal_class_attribute_remains_fail_closed() -> None:
    source = "class Example:\n    VALUE = 2\n"
    diff = "@@ -1,2 +1,2 @@\n class Example:\n-    VALUE = 1\n+    VALUE = 2\n"

    analysis = _analyze(source, diff)

    assert analysis.limitation == "class/signature import change"


def test_module_docstring_replacement_keeps_import_and_runtime_consumers() -> None:
    source = '"new docs"\n\ndef helper():\n    return __doc__\n'
    diff = '@@ -1,4 +1,4 @@\n-"old docs"\n+"new docs"\n \n def helper():\n     return __doc__\n'

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.changed_bindings == {"__doc__"}
    assert analysis.binding_consumers == {"<module>", "helper"}


def test_class_docstring_replacement_keeps_class_and_reflective_consumers() -> None:
    source = (
        'class Example:\n    "new docs"\n\n'
        '    def docs(self):\n        return getattr(self, "__doc__")\n'
    )
    diff = '@@ -1,2 +1,2 @@\n class Example:\n-    "old docs"\n+    "new docs"\n'

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.binding_consumers == {"Example", "Example.docs"}


@pytest.mark.parametrize(
    "source,diff",
    (
        ('"new docs"\n', '@@ -1 +1 @@\n-f"{factory()}"\n+"new docs"\n'),
        ('"new docs"\n', '@@ -0,0 +1 @@\n+"new docs"\n'),
        (
            'VALUE = 1\n"new docs"\n',
            '@@ -1,2 +1,2 @@\n VALUE = 1\n-"old docs"\n+"new docs"\n',
        ),
    ),
)
def test_unsupported_string_expression_changes_remain_fail_closed(source: str, diff: str) -> None:
    analysis = _analyze(source, diff)

    assert analysis.limitation


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


def test_deleted_literal_resolves_pre_image_consumers() -> None:
    source = "\ndef helper():\n    return _VALUE\n"
    diff = "@@ -1,4 +1,3 @@\n-_VALUE = 521\n \n def helper():\n     return _VALUE\n"

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.changed_bindings == {"_VALUE"}
    assert analysis.binding_consumers == {"helper"}


def test_deleted_literal_anchored_to_unchanged_assignment_retains_old_consumer() -> None:
    source = "STABLE = 0.5\ndef helper():\n    return 0\n"
    diff = (
        "@@ -1,4 +1,3 @@\n-VALUE = 1000\n STABLE = 0.5\n def helper():\n"
        "-    return VALUE\n+    return 0\n"
    )

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.changed_bindings == {"VALUE"}
    assert analysis.binding_consumers == {"helper"}


@pytest.mark.parametrize("declaration", ("def helper():", "async def helper():"))
@pytest.mark.parametrize("decorator", ("", "@decorate\n"))
def test_deleted_literal_adjacent_to_signature_retains_consumer(
    declaration: str, decorator: str
) -> None:
    source = f"{decorator}{declaration}\n    return VALUE\n"
    diff = "".join(
        difflib.unified_diff(
            ("VALUE = 521\n" + source).splitlines(keepends=True),
            source.splitlines(keepends=True),
        )
    )

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.changed_bindings == {"VALUE"}
    assert analysis.binding_consumers == {"helper"}


@pytest.mark.parametrize(
    "deleted",
    (
        "initialize()\n",
        "initialize(\n    1,\n)\n",
        "VALUE = factory()\n",
        "import sys as VALUE\n",
        "from .helpers import VALUE\n",
        "def removed():\n    return 1\n",
        "class Removed:\n    VALUE = 1\n",
    ),
)
def test_unsupported_deletion_adjacent_to_signature_remains_fail_closed(deleted: str) -> None:
    source = "def helper():\n    return 0\n"
    diff = "".join(
        difflib.unified_diff(
            (deleted + source).splitlines(keepends=True), source.splitlines(keepends=True)
        )
    )

    analysis = _analyze(source, diff)

    assert analysis.limitation == "unresolved import replacement"


def test_deleted_class_attribute_adjacent_to_method_remains_fail_closed() -> None:
    source = "class Example:\n    def helper(self):\n        return self.VALUE\n"
    diff = (
        "@@ -1,4 +1,3 @@\n class Example:\n-    VALUE = 521\n"
        "     def helper(self):\n         return self.VALUE\n"
    )

    analysis = _analyze(source, diff)

    assert analysis.limitation == "unresolved import replacement"


def test_deleted_literal_adjacent_to_signature_retains_pre_image_consumer() -> None:
    source = "def helper():\n    return 0\n"
    diff = "@@ -1,3 +1,2 @@\n-VALUE = 521\n def helper():\n-    return VALUE\n+    return 0\n"

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.changed_bindings == {"VALUE"}
    assert analysis.binding_consumers == {"helper"}


@pytest.mark.parametrize("old_binding", ("VALUE = factory()", "import sys as VALUE"))
def test_effectful_deletion_at_unchanged_assignment_remains_fail_closed(old_binding: str) -> None:
    source = "STABLE = 0.5\n"
    diff = f"@@ -1,2 +1 @@\n-{old_binding}\n STABLE = 0.5\n"

    analysis = _analyze(source, diff)

    assert analysis.limitation == "unresolved import replacement"


def test_deletion_anchor_cannot_hide_a_changed_following_assignment() -> None:
    source = "STABLE = 0.75\n"
    diff = "@@ -1,2 +1 @@\n-VALUE = 1000\n-STABLE = 0.5\n+STABLE = 0.75\n"

    analysis = _analyze(source, diff)

    assert analysis.limitation == "unresolved import replacement"


def test_deleted_literal_with_import_time_consumer_remains_fail_closed() -> None:
    source = "\nOTHER = _VALUE\n"
    diff = "@@ -1,3 +1,2 @@\n-_VALUE = 521\n \n OTHER = _VALUE\n"

    analysis = _analyze(source, diff)

    assert analysis.limitation == "import-time binding consumer"


def test_deleted_effectful_assignment_remains_fail_closed() -> None:
    source = "\ndef helper():\n    return _VALUE\n"
    diff = "@@ -1,4 +1,3 @@\n-_VALUE = create_value()\n \n def helper():\n     return _VALUE\n"

    analysis = _analyze(source, diff)

    assert analysis.limitation == "unresolved import replacement"


def test_same_module_import_from_replacement_resolves_consumers_and_targets() -> None:
    source = "from .helpers import new_helper, stable\n\ndef consumer():\n    return new_helper()\n"
    diff = (
        "@@ -1,4 +1,4 @@\n"
        "-from .helpers import old_helper, stable\n"
        "+from .helpers import new_helper, stable\n"
        " \n"
        " def consumer():\n"
        "-    return old_helper()\n"
        "+    return new_helper()\n"
    )

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.changed_bindings == {"old_helper", "new_helper"}
    assert analysis.binding_consumers == {"consumer"}
    assert analysis.old_import_targets == {ImportTarget("helpers", 1, "old_helper")}
    assert analysis.new_import_targets == {ImportTarget("helpers", 1, "new_helper")}
    assert analysis.new_import_bindings == {"new_helper"}


def test_multiline_import_from_addition_resolves_consumer_and_new_target() -> None:
    source = (
        "from .helpers import (new_helper,\n"
        "                      stable)\n\n"
        "def consumer():\n"
        "    return new_helper()\n"
    )
    diff = (
        "@@ -1,4 +1,5 @@\n"
        "-from .helpers import (stable)\n"
        "+from .helpers import (new_helper,\n"
        "+                      stable)\n"
        " \n"
        " def consumer():\n"
        "     return new_helper()\n"
    )

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.changed_bindings == {"new_helper"}
    assert analysis.binding_consumers == {"consumer"}
    assert not analysis.old_import_targets
    assert analysis.new_import_targets == {ImportTarget("helpers", 1, "new_helper")}
    assert analysis.new_import_bindings == {"new_helper"}


def test_parenthesized_import_from_replacement_resolves_consumers_and_targets() -> None:
    source = (
        "from .helpers import (\n    new_helper,\n    stable,\n)\n\n"
        "def consumer():\n    return new_helper()\n"
    )
    old_source = source.replace("new_helper", "old_helper")
    diff = "".join(
        difflib.unified_diff(old_source.splitlines(keepends=True), source.splitlines(keepends=True))
    )

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.changed_bindings == {"old_helper", "new_helper"}
    assert analysis.binding_consumers == {"consumer"}
    assert analysis.old_import_targets == {ImportTarget("helpers", 1, "old_helper")}
    assert analysis.new_import_targets == {ImportTarget("helpers", 1, "new_helper")}


def test_new_import_from_statement_resolves_consumer_and_target() -> None:
    source = (
        "from .helpers import stable\n"
        "from .new_helpers import new_helper\n\n"
        "def consumer():\n"
        "    return new_helper()\n"
    )
    diff = (
        "@@ -1,4 +1,5 @@\n"
        " from .helpers import stable\n"
        "+from .new_helpers import new_helper\n"
        " \n"
        " def consumer():\n"
        "     return new_helper()\n"
    )

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.changed_bindings == {"new_helper"}
    assert analysis.binding_consumers == {"consumer"}
    assert not analysis.old_import_targets
    assert analysis.new_import_targets == {ImportTarget("new_helpers", 1, "new_helper")}
    assert analysis.new_import_bindings == {"new_helper"}


def test_import_addition_is_not_new_when_pre_image_already_bound_the_name() -> None:
    source = "new_helper = None\nfrom .helpers import new_helper, stable\n"
    diff = (
        "@@ -1,2 +1,2 @@\n"
        " new_helper = None\n"
        "-from .helpers import stable\n"
        "+from .helpers import new_helper, stable\n"
    )

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.changed_bindings == {"new_helper"}
    assert not analysis.new_import_bindings


@pytest.mark.parametrize(
    ("before", "after"),
    (
        (
            "value: int, enabled: bool = True",
            "value: int, callback: int = None, enabled: bool = True",
        ),
        (
            "value: int, *, enabled: bool = True",
            "value: int, *, callback: int = None, enabled: bool = True",
        ),
    ),
)
def test_inserted_safe_parameter_preserves_existing_arguments(before: str, after: str) -> None:
    source = f"def helper({after}):\n    return value\n"
    diff = f"@@ -1,2 +1,2 @@\n-def helper({before}):\n+def helper({after}):\n     return value\n"
    analysis = _analyze(source, diff)
    assert not analysis.limitation
    assert analysis.binding_consumers == {"helper"}


@pytest.mark.parametrize("declaration", ("def", "async def"))
@pytest.mark.parametrize("parameter", ("option: int = 0", "*, option: int = 0"))
@pytest.mark.parametrize(
    "decorator", ("register", 'fast_custom_op("trtllm::helper")', "property", "staticmethod")
)
def test_decorated_parameter_addition_remains_fail_closed(
    decorator: str, parameter: str, declaration: str
) -> None:
    before = f"@{decorator}\n{declaration} helper(value: int):\n    return value\n"
    source = f"@{decorator}\n{declaration} helper(value: int, {parameter}):\n    return value\n"
    diff = "".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True)))

    assert _analyze(source, diff).limitation == "class/signature import change"


def test_custom_decorated_method_parameter_addition_remains_fail_closed() -> None:
    prefix = "class Example:\n    @register\n"
    before = prefix + "    def helper(self, value: int):\n        return value\n"
    source = (
        prefix + "    def helper(self, value: int, *, option: int = 0):\n        return value\n"
    )
    diff = "".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True)))

    assert _analyze(source, diff).limitation == "class/signature import change"


def test_repository_nvfp4_decorated_parameter_addition_remains_fail_closed() -> None:
    before = (REPO_ROOT / "tensorrt_llm/_torch/custom_ops/torch_custom_ops.py").read_text()
    function = next(
        node
        for node in ast.parse(before).body
        if isinstance(node, ast.FunctionDef) and node.name == "nvfp4_gemm"
    )
    lines = before.splitlines(True)
    lines.insert(function.args.args[-1].end_lineno, "    option: int = 0,\n")
    source = "".join(lines)
    diff = "".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True)))

    assert _analyze(source, diff).limitation == "class/signature import change"


@pytest.mark.parametrize(
    "after",
    (
        "enabled: bool = True, value: int = None, callback: int = None",
        "value: int, callback: int = None, enabled: bool = False",
        "value: str, callback: int = None, enabled: bool = True",
        "value: int, callback: int = None",
        "value: int, callback: int = factory(), enabled: bool = True",
    ),
)
def test_inserted_parameter_rejects_changed_existing_contracts(after: str) -> None:
    before = "value: int, enabled: bool = True"
    source = f"def helper({after}):\n    return value\n"
    diff = f"@@ -1,2 +1,2 @@\n-def helper({before}):\n+def helper({after}):\n     return value\n"
    assert _analyze(source, diff).limitation == "class/signature import change"


@pytest.mark.parametrize("alias", ("Callable", "Callback"))
def test_added_cached_typing_name_used_in_safe_annotation(alias: str) -> None:
    imported = "Callable" if alias == "Callable" else "Callable as Callback"
    before = "from typing import Optional\ndef helper(value: int):\n    return value\n"
    source = (
        f"from typing import Optional, {imported}\n"
        f"def helper(value: int, callback: Optional[{alias}[[int], None]] = None):\n"
        "    return value\n"
    )
    diff = "".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True)))
    analysis = _analyze(source, diff)
    assert not analysis.limitation
    assert analysis.binding_consumers == {"helper"}
    assert not analysis.new_import_targets


def test_first_typing_import_still_requires_import_target_validation() -> None:
    before = "def helper(value: int):\n    return value\n"
    source = (
        "from typing import Callable\n"
        "def helper(value: int, callback: Callable[[int], None] = None):\n"
        "    return value\n"
    )
    diff = "".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True)))
    analysis = _analyze(source, diff)
    assert analysis.limitation == "import-time binding consumer"


@pytest.mark.parametrize("mutation", ("typing.Callable = Custom", "del typing.Callable"))
def test_mutated_typing_module_does_not_exempt_new_annotation_import(mutation: str) -> None:
    before = (
        f"import typing\n{mutation}\nfrom typing import Optional\n"
        "def helper(value: int):\n    return value\n"
    )
    source = (
        f"import typing\n{mutation}\nfrom typing import Optional, Callable\n"
        "def helper(value: int, callback: Callable[[int], None] = None):\n    return value\n"
    )
    diff = "".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True)))
    assert _analyze(source, diff).limitation == "import-time binding consumer"


def test_cached_typing_name_runtime_module_consumer_remains_fail_closed() -> None:
    before = "from typing import Optional\ndef helper(value: int):\n    return value\n"
    source = (
        "from typing import Optional, Callable\nALIAS = Callable\n"
        "def helper(value: int, callback: Callable[[int], None] = None):\n"
        "    return value\n"
    )
    diff = "".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True)))
    assert _analyze(source, diff).limitation == "effectful module statement"


def test_added_method_accepts_cached_typing_import_and_dtype_load() -> None:
    before = "import torch\nfrom typing import Optional\nclass Example:\n    pass\n"
    source = (
        "import torch\nfrom typing import Optional, Callable\nclass Example:\n    pass\n"
        "    def helper(self, callback: Optional[Callable[..., torch.Tensor]] = None,\n"
        "               output_dtype: Optional[torch.dtype] = None) -> torch.Tensor:\n"
        "        return callback()\n"
    )
    diff = "".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True)))
    analysis = _analyze(source, diff)
    assert not analysis.limitation
    assert analysis.binding_consumers == {"Example.helper"}
    assert not analysis.new_import_targets


@pytest.mark.parametrize(
    "prefix", ("from .typing import Callable", "from typing import Callable\nCallable = Custom")
)
def test_relative_or_rebound_typing_name_does_not_trust_subscript(prefix: str) -> None:
    before = f"{prefix}\ndef helper(value: int):\n    return value\n"
    source = f"{prefix}\ndef helper(value: int, callback: Callable[[int], None] = None):\n    return value\n"
    diff = "".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True)))
    assert _analyze(source, diff).limitation == "class/signature import change"


@pytest.mark.parametrize("existing_method", (False, True))
def test_typing_generic_annotation_rejects_class_binding_shadow(existing_method: bool) -> None:
    prefix = "from typing import Callable\nclass Example:\n    Callable = Custom\n"
    before = prefix + ("    def helper(self):\n        pass\n" if existing_method else "")
    source = (
        prefix + "    def helper(self, callback: Callable[[int], None] = None):\n        pass\n"
    )
    diff = "".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True)))
    assert _analyze(source, diff).limitation == "class/signature import change"


def test_optional_parameter_addition_resolves_function_and_method() -> None:
    source = (
        "from typing import Optional\n\n"
        "def helper(value: int, option: Optional[int] = None):\n"
        "    return value\n\n"
        "class Example:\n"
        "    def method(self, value: int, option: Optional[int] = None):\n"
        "        return value\n"
    )
    diff = (
        "@@ -1,8 +1,8 @@\n"
        " from typing import Optional\n"
        " \n"
        "-def helper(value: int):\n"
        "+def helper(value: int, option: Optional[int] = None):\n"
        "     return value\n"
        " \n"
        " class Example:\n"
        "-    def method(self, value: int):\n"
        "+    def method(self, value: int, option: Optional[int] = None):\n"
        "         return value\n"
    )

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.binding_consumers == {"helper", "Example.method"}


def test_keyword_only_optional_parameter_addition_resolves_method() -> None:
    source = (
        "from typing import Optional\n\n"
        "class Example:\n"
        "    def method(self, value: int, *, option: Optional[int] = None, **kwargs):\n"
        "        return value\n"
    )
    diff = (
        "@@ -1,5 +1,5 @@\n"
        " from typing import Optional\n"
        " \n"
        " class Example:\n"
        "-    def method(self, value: int, **kwargs):\n"
        "+    def method(self, value: int, *, option: Optional[int] = None, **kwargs):\n"
        "         return value\n"
    )

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.binding_consumers == {"Example.method"}


def test_required_keyword_only_parameter_addition_resolves_method() -> None:
    source = (
        "class Example:\n"
        "    def method(self, value: int, *, option: int, **kwargs):\n"
        "        return value\n"
    )
    diff = (
        "@@ -1,3 +1,3 @@\n"
        " class Example:\n"
        "-    def method(self, value: int, **kwargs):\n"
        "+    def method(self, value: int, *, option: int, **kwargs):\n"
        "         return value\n"
    )

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.binding_consumers == {"Example.method"}


def test_required_positional_parameter_addition_resolves_method() -> None:
    source = (
        "class Example:\n    def method(self, value: int, required: bool):\n        return value\n"
    )
    diff = (
        "@@ -1,3 +1,3 @@\n"
        " class Example:\n"
        "-    def method(self, value: int):\n"
        "+    def method(self, value: int, required: bool):\n"
        "         return value\n"
    )

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.binding_consumers == {"Example.method"}


@pytest.mark.parametrize("annotation", ("None", "bool", "list[int]", "Optional[int]"))
def test_added_safe_return_annotation_resolves_function_and_method(annotation: str) -> None:
    source = (
        "from typing import Optional\n"
        f"def helper(value: int) -> {annotation}:\n"
        "    return value\n"
        "class Example:\n"
        f"    def method(self, value: int) -> {annotation}:\n"
        "        return value\n"
    )
    diff = (
        "@@ -1,6 +1,6 @@\n"
        " from typing import Optional\n"
        "-def helper(value: int):\n"
        f"+def helper(value: int) -> {annotation}:\n"
        "     return value\n"
        " class Example:\n"
        "-    def method(self, value: int):\n"
        f"+    def method(self, value: int) -> {annotation}:\n"
        "         return value\n"
    )

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.binding_consumers == {"helper", "Example.method"}


def test_parameter_and_return_annotation_additions_resolve_method() -> None:
    source = (
        "class Example:\n"
        "    def method(self, value: int, metadata: list[dict[str, list[int]]] | None = None) -> None:\n"
        "        return value\n"
    )
    diff = (
        "@@ -1,3 +1,3 @@\n"
        " class Example:\n"
        "-    def method(self, value: int):\n"
        "+    def method(self, value: int, metadata: list[dict[str, list[int]]] | None = None) -> None:\n"
        "         return value\n"
    )

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.binding_consumers == {"Example.method"}


@pytest.mark.parametrize(
    "annotation", ("factory()", "CustomType", "CustomType | None", "types.Result")
)
def test_unsafe_added_return_annotation_remains_fail_closed(annotation: str) -> None:
    source = f"def helper(value: int) -> {annotation}:\n    return value\n"
    diff = (
        "@@ -1,2 +1,2 @@\n-def helper(value: int):\n"
        f"+def helper(value: int) -> {annotation}:\n     return value\n"
    )

    analysis = _analyze(source, diff)

    assert analysis.limitation == "class/signature import change"


@pytest.mark.parametrize(("old_return", "new_return"), ((" -> int", " -> None"), (" -> int", "")))
def test_return_annotation_replacement_or_deletion_remains_fail_closed(
    old_return: str, new_return: str
) -> None:
    source = f"def helper(value: int){new_return}:\n    return value\n"
    diff = (
        f"@@ -1,2 +1,2 @@\n-def helper(value: int){old_return}:\n"
        f"+def helper(value: int){new_return}:\n     return value\n"
    )

    analysis = _analyze(source, diff)

    assert analysis.limitation == "class/signature import change"


@pytest.mark.parametrize("decorator", ("register", "property", "staticmethod"))
def test_decorated_return_annotation_addition_remains_fail_closed(decorator: str) -> None:
    source = f"@{decorator}\ndef helper(value: int) -> None:\n    return value\n"
    diff = (
        f"@@ -1,3 +1,3 @@\n @{decorator}\n-def helper(value: int):\n"
        "+def helper(value: int) -> None:\n     return value\n"
    )

    analysis = _analyze(source, diff)

    assert analysis.limitation == "class/signature import change"


@pytest.mark.parametrize(
    "new_parameter",
    (
        "option: Optional[int] = factory()",
        "option: CustomType = None",
    ),
)
def test_unsafe_parameter_addition_remains_fail_closed(new_parameter: str) -> None:
    source = (
        "from typing import Optional\n\n"
        f"def helper(value: int, {new_parameter}):\n"
        "    return value\n"
    )
    diff = (
        "@@ -1,4 +1,4 @@\n"
        " from typing import Optional\n"
        " \n"
        "-def helper(value: int):\n"
        f"+def helper(value: int, {new_parameter}):\n"
        "     return value\n"
    )

    analysis = _analyze(source, diff)

    assert analysis.limitation == "class/signature import change"


@pytest.mark.parametrize("alias", ("torch", "torch as th"))
def test_added_function_accepts_canonical_tensor_annotation_load(alias: str) -> None:
    local = "th" if " as " in alias else "torch"
    source = (
        f"import {alias}\ndef helper(value: {local}.Tensor) -> {local}.Tensor:\n    return value\n"
    )
    diff = (
        f"@@ -1 +1,3 @@\n import {alias}\n"
        f"+def helper(value: {local}.Tensor) -> {local}.Tensor:\n+    return value\n"
    )

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.binding_consumers == {"helper"}


@pytest.mark.parametrize(
    "annotation", ("torch.Tensor | None", "torch.Tensor[int]", "torch.Unknown")
)
def test_tensor_annotation_load_does_not_trust_type_operators_or_other_attributes(
    annotation: str,
) -> None:
    source = f"import torch\ndef helper(value: {annotation}):\n    return value\n"
    diff = f"@@ -1 +1,3 @@\n import torch\n+def helper(value: {annotation}):\n+    return value\n"

    analysis = _analyze(source, diff)

    assert analysis.limitation == "class/signature import change"


@pytest.mark.parametrize(
    "mutation",
    (
        "torch = other",
        "del torch.Tensor",
        "torch.Tensor = other",
        "from custom import *",
        "def stable(value=(torch := other)): pass",
        "class Setup: torch.Tensor = other",
    ),
)
def test_tensor_annotation_load_rejects_a_changed_module_binding(mutation: str) -> None:
    source = f"import torch\n{mutation}\ndef helper(value: torch.Tensor):\n    return value\n"
    diff = (
        f"@@ -1,2 +1,4 @@\n import torch\n {mutation}\n"
        "+def helper(value: torch.Tensor):\n+    return value\n"
    )

    analysis = _analyze(source, diff)

    assert analysis.limitation == "class/signature import change"


def test_tensor_annotation_load_rejects_a_class_alias_shadow() -> None:
    source = (
        "import torch\nclass Example:\n    torch = other\n"
        "    def helper(self, value: torch.Tensor):\n        return value\n"
    )
    diff = (
        "@@ -1,3 +1,5 @@\n import torch\n class Example:\n     torch = other\n"
        "+    def helper(self, value: torch.Tensor):\n+        return value\n"
    )

    analysis = _analyze(source, diff)

    assert analysis.limitation == "class/signature import change"


def test_tensor_annotation_load_rejects_a_noncanonical_module_alias() -> None:
    source = "import custom as torch\ndef helper(value: torch.Tensor):\n    return value\n"
    diff = (
        "@@ -1 +1,3 @@\n import custom as torch\n"
        "+def helper(value: torch.Tensor):\n+    return value\n"
    )

    analysis = _analyze(source, diff)

    assert analysis.limitation == "class/signature import change"


@pytest.mark.parametrize(
    ("imports", "annotation"),
    (
        ("from typing import Callable, Optional", "Optional[Callable[[], None]]"),
        ("from typing import Callable", "Callable[[int, str], bool]"),
        ("from collections.abc import Callable as Callback", "Callback[[int], None]"),
        ("import typing", "typing.Callable[[int], None]"),
    ),
)
def test_added_method_with_callable_parameter_list_resolves_consumer(
    imports: str, annotation: str
) -> None:
    source = (
        f"{imports}\n\nclass Example:\n    pass\n"
        f"    @staticmethod\n    def close(callback: {annotation} = None) -> None:\n"
        "        if callback is not None:\n            callback()\n"
    )
    diff = (
        f"@@ -1,4 +1,8 @@\n {imports}\n \n class Example:\n     pass\n"
        f"+    @staticmethod\n+    def close(callback: {annotation} = None) -> None:\n"
        "+        if callback is not None:\n+            callback()\n"
    )

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.binding_consumers == {"Example.close"}


@pytest.mark.parametrize("parameter_types", ("factory()", "*TYPES", "CustomType"))
def test_effectful_or_unknown_callable_parameter_list_remains_fail_closed(
    parameter_types: str,
) -> None:
    source = (
        "from typing import Callable\n\n"
        f"def helper(value: int, callback: Callable[[{parameter_types}], None] = None):\n"
        "    return value\n"
    )
    diff = (
        "@@ -1,4 +1,4 @@\n from typing import Callable\n \n"
        "-def helper(value: int):\n"
        f"+def helper(value: int, callback: Callable[[{parameter_types}], None] = None):\n"
        "     return value\n"
    )

    analysis = _analyze(source, diff)

    assert analysis.limitation == "class/signature import change"


def test_reconstruct_diff_pre_image_rejects_mismatched_post_source() -> None:
    diff = "@@ -1 +1 @@\n-old\n+new\n"

    assert reconstruct_diff_pre_image("different\n", diff) is None


def test_import_from_replacement_from_different_module_resolves_targets() -> None:
    source = "from .new_helpers import helper\n"
    diff = "@@ -1 +1 @@\n-from .old_helpers import helper\n+from .new_helpers import helper\n"

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.changed_bindings == {"helper"}
    assert analysis.old_import_targets == {ImportTarget("old_helpers", 1, "helper")}
    assert analysis.new_import_targets == {ImportTarget("new_helpers", 1, "helper")}
    assert not analysis.new_import_bindings


def test_import_only_type_checking_addition_is_ignored() -> None:
    source = (
        "from typing import TYPE_CHECKING\n\n"
        "if TYPE_CHECKING:\n"
        "    from .model import Model\n"
        "    from .scheduler import ScheduledRequests\n"
    )
    diff = (
        "@@ -1,4 +1,5 @@\n"
        " from typing import TYPE_CHECKING\n"
        " \n"
        " if TYPE_CHECKING:\n"
        "+    from .model import Model\n"
        "     from .scheduler import ScheduledRequests\n"
    )

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert not analysis.changed_bindings


@pytest.mark.parametrize(
    "source",
    (
        (
            "from typing import TYPE_CHECKING\n"
            "TYPE_CHECKING = enabled()\n\n"
            "if TYPE_CHECKING:\n"
            "    from .model import Model\n"
            "    from .scheduler import ScheduledRequests\n"
        ),
        (
            "from typing import TYPE_CHECKING\n\n"
            "if TYPE_CHECKING:\n"
            "    initialize()\n"
            "    from .model import Model\n"
            "    from .scheduler import ScheduledRequests\n"
        ),
    ),
)
def test_untrusted_type_checking_import_addition_remains_fail_closed(source: str) -> None:
    added_line = 5
    diff = f"@@ -{added_line - 1},0 +{added_line} @@\n+    from .model import Model\n"

    analysis = _analyze(source, diff)

    assert analysis.limitation == "effectful module statement"


@pytest.mark.parametrize("alias", ("typing", "t"))
@pytest.mark.parametrize("rebound", (False, True))
@pytest.mark.parametrize("has_else", (False, True))
def test_typing_attribute_guard_requires_trusted_alias_and_no_else(
    alias: str, rebound: bool, has_else: bool
) -> None:
    source = "import typing\n" if alias == "typing" else f"import typing as {alias}\n"
    if rebound:
        source += f"{alias} = fake()\n"
    source += (
        f"if {alias}.TYPE_CHECKING:\n"
        "    from .model import Model\n"
        "    from .scheduler import ScheduledRequests\n"
    )
    if has_else:
        source += "else:\n    initialize()\n"
    before = source.replace("    from .model import Model\n", "")
    diff = "".join(
        difflib.unified_diff(before.splitlines(keepends=True), source.splitlines(keepends=True))
    )

    analysis = _analyze(source, diff)

    if rebound or has_else:
        assert analysis.limitation == "effectful module statement"
    else:
        assert not analysis.limitation
        assert not analysis.changed_bindings
        assert not analysis.binding_consumers


def test_type_checking_name_guard_with_else_remains_fail_closed() -> None:
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from .model import Model\n"
        "    from .scheduler import ScheduledRequests\n"
        "else:\n    initialize()\n"
    )
    diff = "@@ -2,0 +3 @@\n+    from .model import Model\n"

    analysis = _analyze(source, diff)

    assert analysis.limitation == "effectful module statement"


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


def test_literal_assignment_resolves_closure_consumer_to_the_closure_itself() -> None:
    """Binding-consumer resolution attributes a closure's own load to the closure.

    ``_recorded_definition_groups`` records a nested def under its own qualname
    (matching the collector -- see ``pystart.py``'s ``_SKIP_QUALNAMES``), so
    ``_FunctionNames`` resolves its body as its own fact rather than rolling it
    into the outer def. Only the outer def's genuinely own loads -- its direct
    statements, plus the nested def's *signature* (decorators, defaults,
    annotations), which runs in the outer scope at definition time -- count as
    the outer def's own consumption.
    """
    source = (
        "_VALUE = 521\n\ndef outer():\n    def inner():\n        return _VALUE\n    return inner\n"
    )

    analysis = _analyze(source, "@@ -0,0 +1 @@\n+_VALUE = 521\n")

    assert not analysis.limitation
    assert analysis.binding_consumers == {"outer.<locals>.inner"}


def test_literal_assignment_attributes_closure_signature_consumer_to_outer() -> None:
    """A nested def's default-argument expression attributes to the outer def.

    The default value is evaluated in the outer scope when the nested def is
    defined, not when it is later called, so a changed binding it reads is a
    consumer of the outer def, not of the nested def whose body never runs yet.
    """
    source = "_VALUE = 521\n\ndef outer():\n    def inner(x=_VALUE):\n        return x\n    return inner\n"

    analysis = _analyze(source, "@@ -0,0 +1 @@\n+_VALUE = 521\n")

    assert not analysis.limitation
    assert analysis.binding_consumers == {"outer"}


def test_qualnames_for_lines_resolves_closure_body_line_to_itself() -> None:
    """A changed line inside a named closure attributes to the closure, not its outer def.

    The closure is recorded under its own qualname (``pystart.py``), so a line
    inside it resolves directly rather than falling back to an ancestor.
    """
    source = "def decorator(func):\n    def wrapper(*a, **kw):\n        return func(*a, **kw)\n    return wrapper\n"

    qualnames, ok = qualnames_for_lines(source, {3})

    assert ok
    assert qualnames == {"decorator.<locals>.wrapper"}


def test_qualnames_for_lines_still_attributes_lambda_body_to_enclosing_def() -> None:
    """A lambda/genexpr never gets its own qualname.

    A change inside one still resolves to the nearest enclosing named def, same as
    before. See ``pystart.py``'s ``_SKIP_QUALNAMES`` for why: distinct lambdas in
    one scope share an identical ``co_qualname`` and cannot be told apart.
    """
    source = "def outer():\n    callback = lambda x: x + 1\n    return callback\n"

    qualnames, ok = qualnames_for_lines(source, {2})

    assert ok
    assert qualnames == {"outer"}


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


def test_added_module_function_with_safe_definition_expressions_is_resolved() -> None:
    source = (
        "from .request import Request\n\n"
        "def helper(request: Request, retries: int = 1, *, timeout: int | None = None):\n"
        "    return request\n"
    )
    diff = (
        "@@ -1 +1,4 @@\n"
        " from .request import Request\n"
        "+\n"
        "+def helper(request: Request, retries: int = 1, *, timeout: int | None = None):\n"
        "+    return request\n"
    )

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.changed_bindings == {"helper"}
    assert analysis.binding_consumers == {"helper"}
    assert analysis.new_declaration_bindings == {"helper"}


@pytest.mark.parametrize("alias", ("dataclass", "record"))
@pytest.mark.parametrize("postponed", (False, True))
def test_added_dataclass_resolves_binding_consumers(alias: str, postponed: bool) -> None:
    prefix = "from __future__ import annotations\n" if postponed else ""
    prefix += f"from dataclasses import dataclass as {alias}\n\n"
    source, diff = added_dataclass_change(
        ADDED_DATACLASS.replace("@dataclass", f"@{alias}"), prefix=prefix
    )

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.changed_bindings == {"Budget"}
    assert analysis.new_declaration_bindings == analysis.new_class_bindings == {"Budget"}
    assert analysis.binding_consumers == {"helper"}
    assert analysis.callers == {"helper": {"caller"}}


@pytest.mark.parametrize("decorator", ("@dataclass", "@dataclass()", "@dataclass(frozen=False)"))
def test_added_dataclass_accepts_immutable_defaults(decorator: str) -> None:
    source, diff = added_dataclass_change(
        f"{decorator}\nclass Budget:\n    amount: int = 1\n    offsets: tuple = (1, -2)\n"
    )

    assert not _analyze(source, diff).limitation


def test_added_dataclass_annotation_contributes_its_function_consumer() -> None:
    prefix = "from dataclasses import dataclass\n\n"
    before = prefix + "def caller():\n    return None\n"
    source = (
        prefix + ADDED_DATACLASS + "\n"
        "def helper() -> Budget:\n    return Budget(2)\n\n"
        "def caller():\n    return helper()\n"
    )
    diff = "".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True)))

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.binding_consumers == {"helper", "caller"}


@pytest.mark.parametrize("consumer", ("INSTANCE = Budget(2)\n", "register(Budget)\n"))
def test_added_dataclass_import_time_consumers_still_decline(consumer: str) -> None:
    source, diff = added_dataclass_change(ADDED_DATACLASS + "\n" + consumer)

    assert _analyze(source, diff).limitation


@pytest.mark.parametrize(
    "declaration",
    (
        "@custom\nclass Budget:\n    amount: int\n",
        "@dataclass(slots=True)\nclass Budget:\n    amount: int\n",
        "@dataclass(frozen=setting)\nclass Budget:\n    amount: int\n",
        "@dataclass(factory())\nclass Budget:\n    amount: int\n",
        "@dataclass\nclass Budget(Base):\n    amount: int\n",
        "@dataclass\nclass Budget(metaclass=Meta):\n    amount: int\n",
        "@dataclass\nclass Budget:\n    amount: Custom\n",
        "@dataclass\nclass Budget:\n    amount: int = factory()\n",
        "@dataclass\nclass Budget:\n    amount: list = []\n",
        "@dataclass\nclass Budget:\n    amount: int = 1\n    required: int\n",
        "@dataclass\nclass Budget:\n    amount: int\n    amount: int\n",
        "@dataclass\nclass Budget:\n    amount: int\n    register()\n",
        "@dataclass\nclass Budget:\n    amount: int\n    class Nested:\n        pass\n",
        "@dataclass\nclass Budget:\n    amount: int\n    def method(self, value=factory()):\n        pass\n",
        "@dataclass\nclass Budget:\n    amount: int\n    @custom\n    def method(self):\n        pass\n",
        "@dataclass\nclass Budget:\n    amount: int\n    def __init_subclass__(cls):\n        register(cls)\n",
    ),
)
def test_added_dataclass_preserves_unsupported_declaration_effects(declaration: str) -> None:
    source, diff = added_dataclass_change(declaration)

    assert _analyze(source, diff).limitation == "class/signature import change"


@pytest.mark.parametrize(
    "prefix",
    (
        "from project import dataclass\n\n",
        "from dataclasses import dataclass\ndataclass = custom\n\n",
        "from dataclasses import dataclass, field as dataclass\n\n",
        "from dataclasses import dataclass\nfrom project import *\n\n",
        "from dataclasses import dataclass\nint = Custom\n\n",
        "from dataclasses import dataclass\nproperty = custom\n\n",
        "from dataclasses import dataclass\nBudget = None\n\n",
        "from dataclasses import dataclass\ndef discover(name):\n    return globals()[name]\n\n",
        "from dataclasses import dataclass\ndef discover(module):\n    return vars(module)\n\n",
        "from dataclasses import dataclass\nnamespace = globals\n\n",
        "from dataclasses import dataclass\nfrom builtins import globals as namespace\n\n",
        "from dataclasses import dataclass\ndef discover(fn):\n    return fn.__globals__\n\n",
        "from dataclasses import dataclass\ndef discover(module):\n    return dir(module)\n\n",
        "from dataclasses import dataclass\nnamespace = dir\n\n",
        "from dataclasses import dataclass\nimport builtins\n"
        "def discover(module):\n    return builtins.dir(module)\n\n",
        "from dataclasses import dataclass\nfrom builtins import dir as namespace\n\n",
        "from dataclasses import dataclass\n"
        "def discover(name):\n    return getattr(__import__(__name__), name)\n\n",
        "from dataclasses import dataclass\nimport builtins\n"
        "def discover(name):\n    return getattr(builtins.__import__(__name__), name)\n\n",
        "from dataclasses import dataclass\nfrom builtins import __import__ as load\n"
        "def discover(name):\n    return getattr(load(__name__), name)\n\n",
        "from dataclasses import dataclass\nimport importlib\n"
        "def discover(name):\n    return getattr(importlib.import_module(__name__), name)\n\n",
        "from dataclasses import dataclass\nimport importlib as loader\n"
        "def discover(name):\n    return getattr(loader.import_module(__name__), name)\n\n",
        "from dataclasses import dataclass\nfrom importlib import import_module\n"
        "def discover(name):\n    return getattr(import_module(__name__), name)\n\n",
        "from dataclasses import dataclass\nfrom importlib import import_module as load\n"
        "def discover(name):\n    return getattr(load(__name__), name)\n\n",
    ),
)
def test_added_dataclass_preserves_shadowing_and_namespace_discovery(prefix: str) -> None:
    source, diff = added_dataclass_change(ADDED_DATACLASS, prefix=prefix)

    assert _analyze(source, diff).limitation


def test_added_dataclass_cannot_absorb_an_effectful_deletion() -> None:
    prefix = "from dataclasses import dataclass\n\n"
    before = prefix + "register()\n"
    source = prefix + ADDED_DATACLASS
    diff = "".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True)))

    assert _analyze(source, diff).limitation


def test_added_dataclass_requires_an_existing_stdlib_import() -> None:
    source = "from dataclasses import dataclass\n\n" + ADDED_DATACLASS
    diff = "".join(difflib.unified_diff([], source.splitlines(True)))

    assert _analyze(source, diff).limitation


def test_added_dataclass_does_not_hide_decorated_method_signature_changes() -> None:
    prefix = "from dataclasses import dataclass\n\n"
    kernel = "class Kernel:\n    @jit\n    def run(self, value):\n        return value\n"
    before = prefix + kernel
    source = prefix + ADDED_DATACLASS + kernel.replace("value):", "value, option=0):")
    diff = "".join(difflib.unified_diff(before.splitlines(True), source.splitlines(True)))

    assert _analyze(source, diff).limitation == "class/signature import change"


def test_added_method_can_use_earlier_module_class_annotation() -> None:
    source = (
        "class Budget:\n"
        "    pass\n\n"
        "class Scheduler:\n"
        "    def existing(self):\n"
        "        return 0\n\n"
        "    def has_budget(self, budget: Budget) -> bool:\n"
        "        return True\n"
    )
    diff = (
        "@@ -1,6 +1,9 @@\n"
        " class Budget:\n"
        "     pass\n"
        " \n"
        " class Scheduler:\n"
        "     def existing(self):\n"
        "         return 0\n"
        "+\n"
        "+    def has_budget(self, budget: Budget) -> bool:\n"
        "+        return True\n"
    )

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.binding_consumers == {"Scheduler.has_budget"}


def test_added_function_cannot_use_later_module_binding_annotation() -> None:
    source = "VALUE = 1\n\ndef helper(value: Later):\n    return value\n\nclass Later:\n    pass\n"
    diff = (
        "@@ -1,4 +1,7 @@\n"
        " VALUE = 1\n"
        "+\n"
        "+def helper(value: Later):\n"
        "+    return value\n"
        " \n"
        " class Later:\n"
        "     pass\n"
    )

    analysis = _analyze(source, diff)

    assert analysis.limitation == "class/signature import change"


def test_added_plain_and_builtin_decorated_methods_are_resolved() -> None:
    source = (
        "class Example:\n"
        "    def existing(self):\n"
        "        return 0\n\n"
        "    def plain(self, required, option=None):\n"
        "        return required\n\n"
        "    @property\n"
        "    def value(self):\n"
        "        return 1\n\n"
        "    @staticmethod\n"
        "    def static(value):\n"
        "        return value\n\n"
        "    @classmethod\n"
        "    def create(cls):\n"
        "        return cls()\n"
    )
    diff = (
        "@@ -1,3 +1,18 @@\n"
        " class Example:\n"
        "     def existing(self):\n"
        "         return 0\n"
        "+\n"
        "+    def plain(self, required, option=None):\n"
        "+        return required\n"
        "+\n"
        "+    @property\n"
        "+    def value(self):\n"
        "+        return 1\n"
        "+\n"
        "+    @staticmethod\n"
        "+    def static(value):\n"
        "+        return value\n"
        "+\n"
        "+    @classmethod\n"
        "+    def create(cls):\n"
        "+        return cls()\n"
    )

    analysis = _analyze(source, diff)

    assert not analysis.limitation
    assert analysis.binding_consumers == {
        "Example.create",
        "Example.plain",
        "Example.static",
        "Example.value",
    }
    assert not analysis.new_declaration_bindings


@pytest.mark.parametrize(
    "declaration",
    (
        "@decorate\ndef helper():\n    return 1",
        "def helper(value=factory()):\n    return value",
    ),
)
def test_added_function_with_effectful_definition_expression_falls_back(
    declaration: str,
) -> None:
    source = f"VALUE = 1\n\n{declaration}\n"
    added = "\n".join(f"+{line}" for line in declaration.splitlines())
    diff = f"@@ -1 +1,{len(declaration.splitlines()) + 2} @@\n VALUE = 1\n+\n{added}\n"

    analysis = _analyze(source, diff)

    assert analysis.limitation == "class/signature import change"


def test_added_function_with_custom_subscript_annotation_falls_back() -> None:
    source = "from .types import Custom\n\ndef helper(value: Custom[int]):\n    return value\n"
    diff = (
        "@@ -1 +1,4 @@\n"
        " from .types import Custom\n"
        "+\n"
        "+def helper(value: Custom[int]):\n"
        "+    return value\n"
    )

    analysis = _analyze(source, diff)

    assert analysis.limitation == "class/signature import change"


def test_shadowed_builtin_method_decorator_falls_back() -> None:
    source = (
        "property = decorate\n\n"
        "class Example:\n"
        "    def existing(self):\n"
        "        return 0\n\n"
        "    @property\n"
        "    def value(self):\n"
        "        return 1\n"
    )
    diff = (
        "@@ -1,5 +1,9 @@\n"
        " property = decorate\n"
        " \n"
        " class Example:\n"
        "     def existing(self):\n"
        "         return 0\n"
        "+\n"
        "+    @property\n"
        "+    def value(self):\n"
        "+        return 1\n"
    )

    analysis = _analyze(source, diff)

    assert analysis.limitation == "class/signature import change"


def test_replaced_module_binding_is_not_treated_as_new_function() -> None:
    source = "def helper():\n    return 1\n"
    diff = "@@ -1 +1,2 @@\n-helper = None\n+def helper():\n+    return 1\n"

    analysis = _analyze(source, diff)

    assert analysis.limitation == "class/signature import change"


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


@pytest.mark.parametrize("deleted_binding", (False, True))
@pytest.mark.parametrize(
    "consumer",
    (
        "def consumer(VALUE):\n    return VALUE()\n",
        "def consumer():\n    VALUE = lambda: 0\n    return VALUE()\n",
        "def consumer():\n    import sys as VALUE\n    return VALUE()\n",
    ),
)
def test_shadowed_direct_call_is_not_a_binding_consumer(
    deleted_binding: bool, consumer: str
) -> None:
    before = "VALUE = 521\n\n" + consumer
    after = "\n" + consumer if deleted_binding else before.replace("521", "522")
    diff = "".join(
        difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True))
    )

    analysis = _analyze(after, diff)

    assert not analysis.limitation
    assert analysis.changed_bindings == {"VALUE"}
    assert not analysis.binding_consumers


@pytest.mark.parametrize("deleted_binding", (False, True))
def test_explicit_global_direct_call_remains_a_binding_consumer(deleted_binding: bool) -> None:
    before = "VALUE = 521\n\ndef consumer():\n    global VALUE\n    return VALUE()\n"
    after = (
        before.removeprefix("VALUE = 521\n") if deleted_binding else before.replace("521", "522")
    )
    diff = "".join(
        difflib.unified_diff(before.splitlines(keepends=True), after.splitlines(keepends=True))
    )

    analysis = _analyze(after, diff)

    assert not analysis.limitation
    assert analysis.binding_consumers == {"consumer"}


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
