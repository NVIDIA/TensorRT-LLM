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
"""Exercise the static checker against source files that it never imports."""

import json
import subprocess
import sys
from pathlib import Path
from textwrap import dedent, indent

import pytest

pytestmark = pytest.mark.cpu_only
CHECKER = Path(__file__).resolve().parents[3] / "scripts/check_mock_interfaces.py"


def _check(tmp_path: Path, production: str, test: str) -> list[dict]:
    checkout = tmp_path / "checkout"
    tests = checkout / "tests"
    tests.mkdir(parents=True)
    (checkout / "production.py").write_text(dedent(production))
    (tests / "test_fixture.py").write_text(dedent(test))
    report_path = tmp_path / "report.json"
    result = subprocess.run(
        [
            sys.executable,
            str(CHECKER),
            "--root",
            checkout.name,
            "--report",
            str(report_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert report_path.is_file(), result.stdout + result.stderr
    report = json.loads(report_path.read_text())
    assert not report["analysis_errors"], report["analysis_errors"]
    assert result.returncode == report["check_exit_code"], result.stdout + result.stderr
    return report["results"]


def test_namespace_respects_constructor_scope(tmp_path: Path):
    results = _check(
        tmp_path,
        "class Real:\n    x = 1\n",
        """
        from types import SimpleNamespace
        from test_common.interface_checks import verify_instance

        def local_fixture():
            verify_instance(SimpleNamespace(x=1), "production:Real", members=("x",))

        def shadowed_constructor(SimpleNamespace):
            verify_instance(SimpleNamespace(x=1), "production:Real", members=("x",))
        """,
    )
    assert [result["status"] for result in results] == ["matched", "unverified"]


def test_selected_members_skip_only_unknown_compiled_base_members(tmp_path: Path):
    results = _check(
        tmp_path,
        """
        from tensorrt_llm.bindings import NativeBase

        class Real(NativeBase):
            x = 1
        """,
        """
        from types import SimpleNamespace
        from test_common.interface_checks import verify_instance

        verify_instance(SimpleNamespace(x=1), "production:Real", members=("x", "native_field"))
        verify_instance(SimpleNamespace(), "production:Real", members=("x", "native_field"))
        """,
    )
    assert [result["status"] for result in results] == ["skipped", "mismatched"]
    for result in results:
        assert result["skipped_checks"][0]["member"] == "tensorrt_llm.bindings.NativeBase"
    assert [error["code"] for error in results[1]["diagnostics"]] == ["missing-initialization"]


def test_selected_members_account_for_custom_lookup(tmp_path: Path):
    results = _check(
        tmp_path,
        """
        class Real:
            x = 1

        class CustomLookup(Real):
            def __getattribute__(self, name):
                raise AttributeError(name)
        """,
        """
        from production import CustomLookup, Real
        from test_common.interface_checks import verify_instance

        verify_instance(object.__new__(Real), "production:Real", members=("x",))
        verify_instance(object.__new__(CustomLookup), "production:CustomLookup", members=("x",))
        """,
    )
    assert [result["status"] for result in results] == ["matched", "unverified"]


def test_deferred_and_rebound_fixtures_are_unverified(tmp_path: Path):
    results = _check(
        tmp_path,
        "class Real:\n    x = 1\n",
        """
        from types import SimpleNamespace
        from test_common.interface_checks import verify_instance

        global_fixture = SimpleNamespace(x=1)

        def captured_fixture():
            verify_instance(global_fixture, "production:Real", members=("x",))

        del global_fixture.x

        def rebound_fixture():
            fixture = SimpleNamespace(x=1)
            if (fixture := None) is None:
                pass
            verify_instance(fixture, "production:Real", members=("x",))
        """,
    )
    assert [result["status"] for result in results] == ["unverified", "unverified"]


def test_nested_mutations_update_tracked_instance(tmp_path: Path):
    results = _check(
        tmp_path,
        "class Real:\n    x = 1\n",
        """
        from types import SimpleNamespace
        from test_common.interface_checks import verify_instance

        inner = SimpleNamespace(x=1)
        outer = SimpleNamespace(inner=inner)
        verify_instance(inner, "production:Real", members=("x",))
        del outer.inner.x
        verify_instance(inner, "production:Real", members=("x",))
        outer.inner.x = 1
        verify_instance(inner, "production:Real", members=("x",))
        """,
    )
    assert [result["status"] for result in results] == ["matched", "mismatched", "matched"]


def test_checkpoint_uses_reachable_branches(tmp_path: Path):
    results = _check(
        tmp_path,
        "class Real:\n    x = 1\n",
        """
        from types import SimpleNamespace
        from test_common.interface_checks import verify_instance

        def one_path(skip):
            fixture = SimpleNamespace()
            if skip:
                return
            else:
                fixture.x = 1
            verify_instance(fixture, "production:Real", members=("x",))

        def two_paths(skip):
            fixture = SimpleNamespace()
            if not skip:
                fixture.x = 1
            verify_instance(fixture, "production:Real", members=("x",))
        """,
    )
    assert [result["status"] for result in results] == ["matched", "mismatched"]
    assert results[1]["diagnostics"][0]["code"] == "possibly-uninitialized"


def test_method_replacement_requires_compatible_binding(tmp_path: Path):
    results = _check(
        tmp_path,
        """
        class Real:
            def send(self, *, key):
                pass

            def send_many(*args, key):
                pass

            @staticmethod
            def encode(key):
                pass
        """,
        """
        from test_common.interface_checks import mock_target

        @mock_target("production:Real")
        class MissingReceiver:
            def send(*, key):
                pass

            def send_many(*args, key):
                pass

            @staticmethod
            def encode(key):
                pass

        mock_target("production:Real.send")(staticmethod(lambda self, *, key: None))
        mock_target("production:Real.send")(lambda self, *, key: None)
        mock_target("production:Real.encode")(staticmethod(lambda key: None))
        """,
    )
    assert [result["status"] for result in results] == [
        "mismatched",
        "mismatched",
        "matched",
        "matched",
    ]
    for result in results[:2]:
        assert [error["code"] for error in result["diagnostics"]] == ["signature"]


@pytest.mark.parametrize(("binding", "receiver"), [("staticmethod", ""), ("classmethod", "cls, ")])
def test_replacements_preserve_descriptor_binding(
    tmp_path: Path, binding: str, receiver: str
) -> None:
    results = _check(
        tmp_path,
        f"""
        class Factory:
            @{binding}
            def make({receiver}key):
                pass
        """,
        f"""
        from test_common.interface_checks import mock_target

        @mock_target("production:Factory.make")
        def plain(key):
            pass

        @mock_target("production:Factory.make")
        @{binding}
        def wrapped({receiver}key):
            pass

        mock_target("production:Factory.make")(lambda key: None)
        mock_target("production:Factory.make")({binding}(lambda {receiver}key: None))
        """,
    )
    assert [result["status"] for result in results] == [
        "mismatched",
        "matched",
        "mismatched",
        "matched",
    ]


@pytest.mark.parametrize(
    "mutation",
    [
        pytest.param("def clear():\n    del fixture.x\nclear()", id="closure"),
        pytest.param('clear = lambda: delattr(fixture, "x")\nclear()', id="lambda"),
        pytest.param(
            'def invoke(callback):\n    callback()\ninvoke(lambda: delattr(fixture, "x"))',
            id="inline-callback",
        ),
        pytest.param(
            "alias = fixture\ndef clear(value=alias):\n    del value.x\nalias = None\nclear()",
            id="function-default",
        ),
        pytest.param(
            'alias = fixture\nclear = lambda value=alias: delattr(value, "x")\n'
            "alias = None\nclear()",
            id="lambda-default",
        ),
        pytest.param(
            "alias = fixture\ndef clear():\n    nonlocal alias\n    del alias.x\n"
            "    alias = None\nclear()",
            id="nonlocal-alias",
        ),
        pytest.param(
            "def clear():\n    def inner():\n        del fixture.x\n    inner()\nclear()",
            id="nested-closure",
        ),
        pytest.param(
            "alias = None\ndef clear():\n    del alias.x\n"
            "callbacks = [clear]\nalias = fixture\ncallbacks[0]()",
            id="container-callback",
        ),
        pytest.param("holder = [fixture]\ndel holder[0].x", id="container"),
        pytest.param(
            "alias = fixture if flag else other\ndel alias.x", id="conditional-expression"
        ),
        pytest.param(
            "if flag:\n    alias = fixture\nelse:\n    alias = other\ndel alias.x", id="branch"
        ),
        pytest.param("class Clear:\n    del fixture.x", id="class-body"),
        pytest.param(
            "def clear(value):\n    del value.x\n"
            "holder = SimpleNamespace(inner=fixture)\nclear(holder.inner)",
            id="nested-argument",
        ),
        pytest.param(
            "def clear_inner(value):\n    del value.inner.x\n"
            "holder = SimpleNamespace(inner=fixture)\nclear_inner(holder)",
            id="reachable-field",
        ),
        pytest.param(
            "holder = SimpleNamespace()\nalias = [holder]\nholder.inner = fixture\n"
            "def clear_inner(value):\n    del value.inner.x\nclear_inner(alias[0])",
            id="reachable-through-container",
        ),
        pytest.param(
            'holder = SimpleNamespace(inner=fixture)\nalias = getattr(holder, "inner")\ndel alias.x',
            id="getattr-alias",
        ),
        pytest.param(
            'holder = SimpleNamespace(inner=fixture)\nalias = vars(holder)["inner"]\ndel alias.x',
            id="vars-alias",
        ),
        pytest.param(
            "holder = SimpleNamespace(items=[fixture])\nalias = holder.items[0]\ndel alias.x",
            id="nested-container",
        ),
        pytest.param("if flag and (alias := fixture):\n    del alias.x", id="short-circuit-walrus"),
        pytest.param(
            "items = [(alias := item) for item in (fixture,)]\ndel alias.x",
            id="comprehension-walrus",
        ),
        pytest.param(
            "def drop():\n    del fixture.x\n"
            "def decorate(fn):\n    drop()\n    return fn\n"
            "@decorate\ndef decorated():\n    pass",
            id="decorator-calls-closure",
        ),
        pytest.param(
            "def invoke(fn):\n    fn()\n    return fn\n@invoke\ndef drop():\n    del fixture.x",
            id="decorator-calls-decorated-function",
        ),
        pytest.param(
            "def drop():\n    del fixture.x\nhooks = SimpleNamespace(drop=drop)\nhooks.drop()",
            id="namespace-calls-closure",
        ),
    ],
)
def test_indirect_mutation_invalidates_instance_evidence(tmp_path: Path, mutation: str) -> None:
    results = _check(
        tmp_path,
        "class Real:\n    x = 1\n",
        "from types import SimpleNamespace\n"
        "from test_common.interface_checks import verify_instance\n\n"
        "def test_fixture(flag):\n"
        "    fixture = SimpleNamespace(x=1)\n"
        "    other = SimpleNamespace(x=1)\n"
        '    verify_instance(fixture, "production:Real", members=("x",))\n'
        + indent(mutation, "    ")
        + '\n    verify_instance(fixture, "production:Real", members=("x",))\n',
    )
    before, after = results
    assert before["status"] == "matched"
    assert after["status"] == "unverified"


def test_callback_default_preserves_the_original_binding(tmp_path: Path) -> None:
    results = _check(
        tmp_path,
        "class Real:\n    x = 1\n",
        """
        from types import SimpleNamespace
        from test_common.interface_checks import verify_instance

        fixture = SimpleNamespace(x=1)
        callbacks = [lambda value=fixture: delattr(value, "x")]
        verify_instance(fixture, "production:Real", members=("x",))
        fixture = SimpleNamespace(x=1)
        callbacks[0]()
        verify_instance(fixture, "production:Real", members=("x",))
        """,
    )
    assert [result["status"] for result in results] == ["matched", "matched"]


def test_descriptor_preserves_captured_instance_evidence(tmp_path: Path) -> None:
    results = _check(
        tmp_path,
        "class Real:\n    x = 1\n",
        """
        from types import SimpleNamespace
        from test_common.interface_checks import verify_instance

        def test_fixture():
            fixture = SimpleNamespace(x=1)

            @staticmethod
            def drop():
                del fixture.x

            verify_instance(fixture, "production:Real", members=("x",))
            drop()
            verify_instance(fixture, "production:Real", members=("x",))
        """,
    )
    assert [result["status"] for result in results] == ["matched", "unverified"]


@pytest.mark.parametrize("flag", [False, True])
def test_short_circuit_setattr_marks_conditional_member_unverified(
    tmp_path: Path, flag: bool
) -> None:
    results = _check(
        tmp_path,
        "class Real:\n    limit = 1\n",
        f"""
        from types import SimpleNamespace
        from test_common.interface_checks import verify_instance

        flag = {flag!r}
        always = SimpleNamespace()
        if setattr(always, "limit", 1) or flag:
            pass
        verify_instance(always, "production:Real", members=("limit",))

        state = SimpleNamespace()
        if flag and setattr(state, "limit", 1):
            pass
        verify_instance(state, "production:Real", members=("limit",))
        """,
    )
    assert [result["status"] for result in results] == ["matched", "unverified"]


def test_namespace_reads_preserve_instance_evidence(tmp_path: Path) -> None:
    results = _check(
        tmp_path,
        "class Real:\n    x = 1\n",
        """
        from types import SimpleNamespace
        from test_common.interface_checks import verify_instance

        def test_fixture():
            fixture = SimpleNamespace(x=1)
            holder = SimpleNamespace(inner=fixture, items=[fixture])
            alias = getattr(holder, "inner")
            alias = vars(holder)["inner"]
            alias = holder.items[0]
            verify_instance(fixture, "production:Real", members=("x",))
        """,
    )
    assert [result["status"] for result in results] == ["matched"]


@pytest.mark.skipif(sys.version_info < (3, 11), reason="except* requires Python 3.11")
def test_exception_group_checkpoints_are_unverified(tmp_path: Path) -> None:
    results = _check(
        tmp_path,
        "class Real:\n    x = 1\n",
        """
        from types import SimpleNamespace
        from test_common.interface_checks import verify_instance

        def test_fixture():
            fixture = SimpleNamespace(x=1)
            try:
                del fixture.x
                verify_instance(fixture, "production:Real", members=("x",))
            except* ValueError:
                pass
            verify_instance(fixture, "production:Real", members=("x",))
        """,
    )
    assert [result["status"] for result in results] == ["unverified", "unverified"]


def test_class_allocator_must_be_resolved(tmp_path: Path) -> None:
    results = _check(
        tmp_path,
        """
        class Plain:
            x = 1

            def __init__(self):
                initialize(self)

        class Custom:
            x = 1

            def __new__(cls):
                return object()

        class Conditional:
            x = 1

            if True:
                def __new__(cls):
                    return object()

        class Inherited(Conditional):
            pass

        def custom_allocator(cls):
            cls.__new__ = staticmethod(lambda cls: object())
            return cls

        @custom_allocator
        class Decorated:
            x = 1
        """,
        """
        from production import Conditional, Custom, Decorated, Inherited, Plain
        from test_common.interface_checks import verify_instance

        verify_instance(Plain.__new__(Plain), "production:Plain", members=("x",))
        verify_instance(Custom.__new__(Custom), "production:Custom", members=("x",))
        verify_instance(Conditional.__new__(Conditional), "production:Conditional", members=("x",))
        verify_instance(Inherited.__new__(Inherited), "production:Inherited", members=("x",))
        verify_instance(Decorated.__new__(Decorated), "production:Decorated", members=("x",))
        verify_instance(object.__new__(Conditional), "production:Conditional", members=("x",))
        """,
    )
    assert [result["status"] for result in results] == [
        "matched",
        "unverified",
        "unverified",
        "unverified",
        "unverified",
        "matched",
    ]


@pytest.mark.parametrize("in_test", [False, True], ids=["production", "test"])
@pytest.mark.parametrize(
    "assignment",
    [
        pytest.param("Base.__new__ = staticmethod(lambda cls: object())", id="attribute"),
        pytest.param('setattr(Base, "__new__", staticmethod(lambda cls: object()))', id="setattr"),
        pytest.param(
            "alias = Base\nalias.__new__ = staticmethod(lambda cls: object())",
            id="alias",
        ),
    ],
)
def test_allocator_rebound_after_class_definition(
    tmp_path: Path, in_test: bool, assignment: str
) -> None:
    production = "class Base:\n    x = 1\n\nclass Real(Base):\n    pass\n"
    test = "from production import Base, Real\nfrom test_common.interface_checks import verify_instance\n"
    if in_test:
        test += assignment + "\n"
    else:
        production += assignment + "\n"
    test += (
        'verify_instance(Real.__new__(Real), "production:Real", members=("x",))\n'
        'verify_instance(object.__new__(Real), "production:Real", members=("x",))\n'
    )
    results = _check(tmp_path, production, test)
    assert [result["status"] for result in results] == ["unverified", "matched"]


def test_inline_checkpoint_follows_call_argument_order(tmp_path: Path) -> None:
    results = _check(
        tmp_path,
        "class Real:\n    x = 1\n",
        """
        from types import SimpleNamespace
        from test_common.interface_checks import verify_instance

        def clear(value):
            del value.x

        def consume(*args):
            pass

        def check_after_mutation():
            fixture = SimpleNamespace(x=1)
            consume(clear(fixture), verify_instance(fixture, "production:Real", members=("x",)))

        def check_before_mutation():
            fixture = SimpleNamespace(x=1)
            consume(verify_instance(fixture, "production:Real", members=("x",)), clear(fixture))
        """,
    )
    assert [result["status"] for result in results] == ["unverified", "matched"]


@pytest.mark.parametrize("assignment", ["send = None", "send, other = None, None"])
def test_class_assignment_respects_binding_order(tmp_path: Path, assignment: str) -> None:
    results = _check(
        tmp_path,
        f"""
        class Real:
            other = None
            {assignment}

            def send(self, key):
                pass
        """,
        f"""
        from test_common.interface_checks import mock_target

        @mock_target("production:Real")
        class Broken:
            other = None

            def send(self, key):
                pass

            {assignment}

        @mock_target("production:Real")
        class Working:
            other = None
            {assignment}

            def send(self, key):
                pass

        @mock_target("production:Real")
        class MissingMethod:
            other = None
            send = None
        """,
    )
    assert [result["status"] for result in results] == ["mismatched", "matched", "mismatched"]
    assert [issue["code"] for issue in results[0]["diagnostics"]] == ["member-kind"]
    assert [issue["code"] for issue in results[2]["diagnostics"]] == ["member-kind"]


@pytest.mark.parametrize(
    ("body", "status", "diagnostics"),
    [
        ("pass", "matched", []),
        ("def __init__(self, endpoint=1, **kwargs):\n    pass", "mismatched", ["missing-member"]),
        (
            "def __init__(self, endpoint=1, **kwargs):\n    super().__init__(endpoint, **kwargs)",
            "matched",
            [],
        ),
        (
            "def __init__(self, endpoint=1, **kwargs):\n    super().__init__(endpoint, endpoint)",
            "unverified",
            ["unverified-interface"],
        ),
        (
            "def __init__(self, endpoint=1, **kwargs):\n    initialize(self, endpoint)",
            "unverified",
            ["unverified-interface"],
        ),
    ],
)
def test_initializer_members_follow_active_initializer(
    tmp_path: Path, body: str, status: str, diagnostics: list[str]
):
    results = _check(
        tmp_path,
        """
        class Base:
            def __init__(self, endpoint=1, **kwargs):
                self.endpoint = endpoint

        class Real(Base):
            pass
        """,
        "from production import Base\n"
        "from test_common.interface_checks import mock_target\n\n"
        '@mock_target("production:Real")\n'
        "class Fake(Base):\n" + indent(body, "    "),
    )
    (result,) = results
    assert result["status"] == status
    assert [error["code"] for error in result["diagnostics"]] == diagnostics
