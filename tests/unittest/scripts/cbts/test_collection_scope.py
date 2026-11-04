# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage scope and file identity across checkout and installed-package paths."""

import configparser
import json
import os
import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
from cbts.command.coverage.collection import pystart_report
from cbts.coverage.collection.compact_db import canonicalize_path
from cbts.coverage.collection.pystart import PyStartTracker, read_source_config

pytestmark = pytest.mark.cpu_only
_REQUIRES_MONITORING = pytest.mark.skipif(
    not hasattr(sys, "monitoring"), reason="sys.monitoring requires Python 3.12+"
)


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("checkout/tensorrt_llm/x.py", "tensorrt_llm/x.py"),
        ("checkout/examples/tensorrt_llm/x.py", "examples/tensorrt_llm/x.py"),
        ("env/lib/site-packages/tensorrt_llm/x.py", "tensorrt_llm/x.py"),
        ("checkout/.venv/lib/site-packages/tensorrt_llm/x.py", "tensorrt_llm/x.py"),
        ("checkout/.venv/lib/dist-packages/tensorrt_llm/x.py", "tensorrt_llm/x.py"),
        ("outside/x.py", None),
    ],
)
def test_canonicalize_checkout_and_wheel_paths(
    tmp_path: Path, path: str, expected: str | None
) -> None:
    """Installed package identity takes priority over an enclosing checkout."""
    absolute = str(tmp_path / path)
    assert canonicalize_path(absolute, str(tmp_path / "checkout")) == (expected or absolute)
    assert canonicalize_path("examples/x.py", str(tmp_path / "checkout")) == "examples/x.py"


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("checkout/tensorrt_llm/x.py", "tensorrt_llm/x.py"),
        ("env/lib/site-packages/tensorrt_llm/x.py", "tensorrt_llm/x.py"),
        ("env/lib/dist-packages/tensorrt_llm/x.py", "tensorrt_llm/x.py"),
        ("outside/x.py", None),
    ],
)
def test_canonicalize_without_repository_root(
    tmp_path: Path, path: str, expected: str | None
) -> None:
    """Legacy callers retain installed/source package identities without a checkout root."""
    absolute = str(tmp_path / path)
    assert canonicalize_path(absolute) == (expected or absolute)
    assert canonicalize_path("examples/x.py") == "examples/x.py"


@pytest.mark.parametrize(
    ("path", "included"),
    [
        ("tensorrt_llm/a/b.py", True),
        ("examples/a/b.py", True),
        ("scripts/a/b.py", True),
        ("tests/test_common/a/b.py", True),
        ("conftest.py", True),
        ("tests/conftest.py", True),
        ("tests/unittest/a/conftest.py", True),
        ("tests/unittest/a/test_x.py", False),
        ("jenkins/scripts/x.py", False),
        ("examples/data.json", False),
    ],
)
def test_tracker_uses_configured_include_patterns(
    tmp_path: Path, path: str, included: bool
) -> None:
    """Include globs cross directory separators without including ordinary tests."""
    tracker = PyStartTracker(
        [str(tmp_path)],
        str(tmp_path),
        repository_root=str(tmp_path),
        include=read_source_config().get("run", "include").split(),
    )
    assert tracker._in_source(str(tmp_path / path)) is included


@pytest.mark.parametrize("linked_root", [False, True])
@_REQUIRES_MONITORING
def test_symlinked_checkout_records_one_file_identity(tmp_path: Path, linked_root: bool) -> None:
    """Real and symlinked filenames produce the same persisted coverage symbol."""
    tool_id = next(
        (slot for slot in (3, 5, 4, 2, 1, 0) if sys.monitoring.get_tool(slot) is None), None
    )
    if tool_id is None:
        pytest.skip("No sys.monitoring tool ID is available")
    checkout = tmp_path / "checkout"
    source = checkout / "examples/x.py"
    source.parent.mkdir(parents=True)
    source.write_text("def run():\n    return 1\n")
    link = tmp_path / "linked-checkout"
    link.symlink_to(checkout, target_is_directory=True)
    root = link if linked_root else checkout
    tracker = PyStartTracker(
        [str(root)],
        str(tmp_path),
        tool_id=tool_id,
        repository_root=str(root),
        include=["examples/*.py"],
    )
    tracker.switch_test_context("test_scope")
    assert tracker.start()
    try:
        for filename in (source, link / "examples/x.py"):
            assert tracker._in_source(str(filename))
            namespace = {}
            exec(compile(source.read_text(), str(filename), "exec"), namespace)
            namespace["run"]()
    finally:
        tracker.stop()
    database = tracker.save()
    assert database is not None
    with sqlite3.connect(database) as connection:
        rows = connection.execute(
            "SELECT file, qualname FROM touch_rows WHERE qualname = 'run'"
        ).fetchall()
    assert rows == [("examples/x.py", "run")]


@_REQUIRES_MONITORING
def test_tracker_records_named_closures_but_skips_lambdas_and_genexprs(tmp_path: Path) -> None:
    """A decorator's wrapper earns its own row; a lambda or genexpr earns none.

    The wrapper is a real, individually addressable function -- just one nested
    inside another -- so it is recorded under its own dotted qualname like any
    other function. A lambda or genexpr has no such identity: two of them directly
    inside the same enclosing scope share the identical ``co_qualname``, so neither
    is ever recorded (see ``PyStartTracker._SKIP_QUALNAMES``).
    """
    tool_id = next(
        (slot for slot in (3, 5, 4, 2, 1, 0) if sys.monitoring.get_tool(slot) is None), None
    )
    if tool_id is None:
        pytest.skip("No sys.monitoring tool ID is available")
    checkout = tmp_path / "checkout"
    source = checkout / "examples/x.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def decorator(func):\n"
        "    def wrapper(*args, **kwargs):\n"
        "        return func(*args, **kwargs)\n"
        "    return wrapper\n"
        "\n"
        "@decorator\n"
        "def foo():\n"
        "    return 1\n"
        "\n"
        "def make_values():\n"
        "    doubled = [x * 2 for x in range(3)]\n"
        "    squared = list(x * x for x in range(3))\n"
        "    adder = lambda x: x + 1\n"
        "    return doubled, squared, adder(1)\n"
    )
    tracker = PyStartTracker(
        [str(checkout)],
        str(tmp_path),
        tool_id=tool_id,
        repository_root=str(checkout),
        include=["examples/*.py"],
    )
    tracker.switch_test_context("test_scope")
    assert tracker.start()
    try:
        namespace = {}
        exec(compile(source.read_text(), str(source), "exec"), namespace)
        namespace["foo"]()
        namespace["make_values"]()
    finally:
        tracker.stop()
    database = tracker.save()
    assert database is not None
    with sqlite3.connect(database) as connection:
        qualnames = {
            row[0]
            for row in connection.execute(
                "SELECT qualname FROM touch_rows WHERE file = 'examples/x.py'"
            ).fetchall()
        }
    assert "decorator.<locals>.wrapper" in qualnames
    assert "foo" in qualnames
    assert "make_values" in qualnames
    assert not any("<lambda>" in q or "<genexpr>" in q for q in qualnames)


@pytest.mark.parametrize("source_kind", ["checkout", "symlink", "legacy"])
def test_report_scope_and_directory_pruning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_kind: str
) -> None:
    """The denominator retains configured files while unrelated trees stay unvisited."""
    checkout = tmp_path / "checkout"
    included = {
        "tensorrt_llm/x.py",
        "examples/a/b.py",
        "scripts/x.py",
        "tests/test_common/x.py",
        "conftest.py",
        "tests/conftest.py",
        "tests/unittest/a/conftest.py",
    }
    excluded = {".git/x.py", "cpp/build/x.py", "3rdparty/x.py", "tests/unittest/test_x.py"}
    for relative in included | excluded:
        path = checkout / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("def run():\n    return 1\n")
    source_root = checkout
    if source_kind == "symlink":
        source_root = tmp_path / "linked-checkout"
        source_root.symlink_to(checkout, target_is_directory=True)
    elif source_kind == "legacy":
        source_root = checkout / "tensorrt_llm"
        included = {"tensorrt_llm/x.py"}
    visited = set()
    original_walk = os.walk

    def observe_walk(root: str) -> Iterator[tuple[str, list[str], list[str]]]:
        for current, directories, files in original_walk(root):
            visited.add(os.path.relpath(current, checkout))
            yield current, directories, files

    monkeypatch.setattr(pystart_report.os, "walk", observe_walk)
    files, definitions = pystart_report.enumerate_defs(str(source_root))
    assert files == included
    assert definitions == {(path, "run") for path in included}
    assert not visited.intersection({".git", "cpp", "cpp/build", "3rdparty"})


@pytest.mark.parametrize("pattern", ["tools/*.py", "*.py"])
def test_report_pruning_respects_custom_patterns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pattern: str
) -> None:
    """A wildcard in the first component may match files in any subtree."""
    config = configparser.ConfigParser()
    config.read_dict({"run": {"include": pattern}})
    monkeypatch.setattr(pystart_report, "read_source_config", lambda: config)
    source = tmp_path / "tools/deep/x.py"
    source.parent.mkdir(parents=True)
    source.write_text("def run():\n    return 1\n")
    files, definitions = pystart_report.enumerate_defs(str(tmp_path))
    assert files == {"tools/deep/x.py"}
    assert definitions == {("tools/deep/x.py", "run")}


def test_report_denominator_includes_closures_but_not_lambdas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The coverage-rate denominator counts a decorator's wrapper like any function.

    It has to match what the collector actually records (`PyStartTracker`), or
    the rate the report prints would be computed over the wrong population.
    """
    config = configparser.ConfigParser()
    config.read_dict({"run": {"include": "tools/*.py"}})
    monkeypatch.setattr(pystart_report, "read_source_config", lambda: config)
    source = tmp_path / "tools/x.py"
    source.parent.mkdir(parents=True)
    source.write_text(
        "def decorator(func):\n"
        "    def wrapper(*args, **kwargs):\n"
        "        return func(*args, **kwargs)\n"
        "    return wrapper\n"
        "\n"
        "def make_values():\n"
        "    doubled = [x * 2 for x in range(3)]\n"
        "    adder = lambda x: x + 1\n"
        "    return doubled, adder(1)\n"
    )
    _files, definitions = pystart_report.enumerate_defs(str(tmp_path))
    assert definitions == {
        ("tools/x.py", "decorator"),
        ("tools/x.py", "decorator.<locals>.wrapper"),
        ("tools/x.py", "make_values"),
    }


@pytest.mark.parametrize("config_kind", ["valid", "missing", "malformed"])
def test_bootstrap_passes_scope_and_disables_on_config_errors(
    tmp_path: Path, config_kind: str
) -> None:
    """Bootstrap forwards repository/include settings and tolerates unreadable configs."""
    rcfile = tmp_path / ".coveragerc"
    if config_kind == "valid":
        rcfile.write_text(
            f"[run]\nrepository_root = {tmp_path}\n"
            "source = %(repository_root)s/\ninclude = examples/*.py\n"
            f"data_file = {tmp_path}/.coverage.unit\n"
        )
    elif config_kind == "malformed":
        rcfile.write_text("not a configuration file\n")
    cbts_root = Path(__file__).resolve().parents[4] / "jenkins/scripts/cbts"
    environment = dict(os.environ)
    environment.update(
        PYTHONPATH=os.pathsep.join([str(cbts_root / "cbts_injectors"), str(cbts_root)]),
        CBTS_COVERAGE_CONFIG=str(rcfile),
        CBTS_PROCESS_ROLE="outer_pytest",
    )
    probe = (
        "import json, sitecustomize; t = sitecustomize._tracker; "
        "print(json.dumps([t.available, t.repository_root, t.include]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    if config_kind == "valid":
        assert json.loads(result.stdout) == [
            hasattr(sys, "monitoring"),
            os.path.realpath(tmp_path),
            ["examples/*.py"],
        ]
    else:
        assert json.loads(result.stdout) == [False, None, []]
        assert "coverage disabled" in result.stderr
