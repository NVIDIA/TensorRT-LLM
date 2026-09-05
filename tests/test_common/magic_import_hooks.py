# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pytest plugin that enables ``__extra_import_path__`` for a test run.

Nothing here is called directly. The plugin is loaded via
``-p test_common.magic_import_hooks`` from each test tree's pytest.ini, which
is already configured, so declaring ``__extra_import_path__`` in a test file or
``conftest.py`` is all that is required. See :mod:`test_common.magic_import`
for the declaration syntax and scoping rules.

Implementation notes follow.

The finder is installed at plugin import time, during preparse and therefore
before any conftest.py is imported, so that a conftest may itself rely on a
declaration.

Purging is what confines a generic helper name such as ``utils`` to the file
that imported it, and must therefore happen at both points at which test code
performs imports:

* ``pytest_make_collect_report`` -- ``Module.collect()`` is what imports a
  collected test file, so this covers every module-level import.
* ``pytest_runtest_protocol`` -- covers imports performed inside a test
  function or a fixture.

Both are wrappers rather than plain hooks so that the purge also runs when the
import or the test raises.

The plugin also guards the invariant that ``__extra_import_path__`` exists to
protect: test code must not put project directories on ``sys.path``, because
such an entry is global and permanent and therefore changes how *other* test
files resolve their imports. See ``_check_sys_path`` below.
"""

import os
import sys

import pytest

from test_common.magic_import import MagicFinder

# Project directories that may be put into sys.path by third-party packages.
#
# TensorRT LLM code should prevent doing this, unless in standalone scripts.
# Each entry must have comments to explain the reason it have to be added:
_NON_TEST_TREES = (
    # CuTeDSL appends its own package directory when imported:
    # 3rdparty/cutlass/python/CuTeDSL/base_dsl/compiler.py runs
    # ``sys.path.append(_SCRIPT_PATH)`` at module level. Vendored third-party
    # code, so not something we can avoid.
    "3rdparty",
    # The same CuTe DSL append, reached instead through the copy of cutlass
    # that CMake fetches into the build tree, e.g.
    # cpp/<build-dir>/_deps/cutlass-src/python. The build directory is named by
    # the developer, so the whole tree is accepted.
    "cpp",
)

# Debug variable for local runs to skip this check.
_SKIP_CHECK_ENV = "TRTLLM_SKIP_SYS_PATH_CHECK"


def _relative_pythonpath_entries() -> list[str]:
    """Entries of the PYTHONPATH environment variable that are not absolute.

    Such an entry names no fixed location: it resolves against whatever the
    working directory happens to be, which is not the same in the pytest
    process and in a worker it spawns. Nothing needs one, so its presence means
    the variable was built wrongly -- most often by a value that was meant to
    be one entry but contained os.pathsep and so became several.

    Empty entries are excluded. They also mean the working directory, but they
    are the ordinary artifact of appending to an unset PYTHONPATH rather than a
    sign that something went wrong.
    """
    entries = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    return [entry for entry in entries if entry and not os.path.isabs(entry)]


def _write_banner(config, title: str, message: str) -> None:
    """Writes a highlighted block to the terminal, falling back to stderr.

    The terminal reporter is not registered yet during early hooks, so a
    caller that runs at configure time may legitimately get nothing back.
    """
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        print(f"{title}\n{message}", file=sys.stderr)
        return
    reporter.write_sep("=", title, red=True)
    reporter.write_line(message)


def pytest_configure(config):
    MagicFinder.install()  # idempotent; covers plugin loading via other paths
    # Under --assert=plain this attribute holds a DummyRewriteHook that has no
    # find_spec, so test for the capability rather than for the attribute.
    hook = getattr(config.pluginmanager, "rewrite_hook", None)
    MagicFinder.rewrite_hook = hook if hasattr(hook, "find_spec") else None

    # Reported here rather than from the sys.path check at session finish: the
    # environment is already wrong before a single test runs, and the symptom
    # it produces later -- an unexplained directory on sys.path -- is far
    # harder to read back to its cause than this message is.
    if not os.environ.get(_SKIP_CHECK_ENV):
        relative = _relative_pythonpath_entries()
        if relative:
            message = (
                "PYTHONPATH contains entries that are not absolute paths: "
                + ", ".join(repr(entry) for entry in relative)
                + ". Such an entry resolves against the working directory rather "
                "than naming a fixed location, so it means something different in "
                "every process that inherits it. Check whichever script or CI step "
                "sets PYTHONPATH: a value meant to be a single entry becomes several "
                f"if it contains `{os.pathsep!r}`. Set {_SKIP_CHECK_ENV}=1 to bypass "
                "this check."
            )
            _write_banner(config, "PYTHONPATH check failed", message)
            # TODO: enable separately after magic import is merged.
            # raise pytest.UsageError(message)


def pytest_plugin_registered(plugin, manager):
    """Supplies conftest modules to the finder as pytest imports them.

    This is the only reliable handle on them. Pytest registers every conftest
    as a plugin, but imports them all under the bare module name ``conftest``,
    so ``sys.modules`` retains only the most recently imported one.
    """
    MagicFinder.register_conftest(plugin)


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    try:
        return (yield)
    finally:
        MagicFinder.purge_magic_sys_modules()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_protocol(item, nextitem):
    try:
        return (yield)
    finally:
        MagicFinder.purge_magic_sys_modules()


def _project_relative(entry: str) -> str | None:
    """Path of ``entry`` relative to the project root, or None if outside it.

    An empty entry means the current directory, which is how pytest spells the
    invocation directory.
    """
    root = os.path.realpath(MagicFinder.project_root)
    resolved = os.path.realpath(entry or os.getcwd())
    if resolved != root and not resolved.startswith(root + os.sep):
        return None
    return os.path.relpath(resolved, root)


def _is_pytest_basedir(directory: str) -> bool:
    """True for a directory that pytest itself prepends.

    Under the default ``prepend`` import mode, importing a test module or a
    conftest inserts the first ancestor directory that is not a package. Such a
    directory therefore has no ``__init__.py`` and holds either the imported
    file itself or the package that contains it.
    """
    if os.path.exists(os.path.join(directory, "__init__.py")):
        return False
    try:
        names = os.listdir(directory)
    except OSError:
        return False
    for name in names:
        if name == "conftest.py" or (name.startswith("test_") and name.endswith(".py")):
            return True
        if os.path.exists(os.path.join(directory, name, "__init__.py")):
            return True
    return False


def _under_exempt_tree(relative: str) -> bool:
    """Whether a project-relative entry sits in one of _NON_TEST_TREES.

    An entry covers the directory it names and everything below it, so a
    nested path such as a build directory matches its whole tree.
    """
    for tree in _NON_TEST_TREES:
        prefix = tree.replace("/", os.sep)
        if relative == prefix or relative.startswith(prefix + os.sep):
            return True
    return False


def _expected_entries(config) -> set[str]:
    """Project-relative sys.path entries that a run is allowed to contain."""
    expected = {"."}  # the project root itself
    expected.add(_project_relative(str(config.rootpath)) or ".")
    expected.add(_project_relative(str(config.invocation_params.dir)) or ".")
    for entry in config.getini("pythonpath"):
        relative = _project_relative(str(entry))
        if relative is not None:
            expected.add(relative)
    # An absolute PYTHONPATH entry is an explicit declaration by whoever
    # launched the run -- the same standing as pytest.ini's pythonpath above,
    # and not something a test file did to itself at import time. A CI harness
    # may legitimately point it at a directory inside the project.
    #
    # Only absolute entries are accepted. A relative one already failed the run
    # at pytest_configure, so reaching here means the check was bypassed, and
    # in that case it stays reportable rather than being quietly excused.
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        if entry and os.path.isabs(entry):
            relative = _project_relative(entry)
            if relative is not None:
                expected.add(relative)
    # The two conftest bootstraps that make test_common and the unittest tree
    # importable in the first place; both are the test roots themselves.
    expected.add(_project_relative(MagicFinder.test_root) or ".")
    return expected


def _check_sys_path(config) -> list[str]:
    """Report project directories that test code put on ``sys.path``.

    ``__extra_import_path__`` exists so that a file can satisfy its own imports
    without a global, permanent ``sys.path`` entry that also changes how every
    other test file resolves imports. Anything under the project root that is
    neither pytest's own doing nor a side effect of importing the product is a
    regression back to that pattern.
    """
    expected = _expected_entries(config)
    root = os.path.realpath(MagicFinder.project_root)
    unexpected = []
    for entry in sys.path:
        relative = _project_relative(entry)
        if relative is None or relative in expected:
            continue
        if _under_exempt_tree(relative):
            continue
        if _is_pytest_basedir(os.path.join(root, relative)):
            continue
        if relative not in unexpected:
            unexpected.append(relative)
    return unexpected


def pytest_sessionfinish(session, exitstatus):
    """Fail the run if test code leaked a project directory onto ``sys.path``.

    Checked once at the end rather than per test: an entry that a fixture adds
    and removes again is not a leak, and one added at import time is still
    present here.
    """
    if os.environ.get(_SKIP_CHECK_ENV):
        return
    unexpected = _check_sys_path(session.config)
    if not unexpected:
        return
    lines = "\n".join(f"  {entry}" for entry in unexpected)
    message = (
        f"Unexpected sys.path entries under the project root:\n{lines}\n"
        "A test file must not add a project directory to sys.path: the entry is "
        "global and permanent, so it also changes how unrelated test files "
        "resolve their imports. Declare __extra_import_path__ in the file that "
        "needs the import instead -- see test_common/magic_import.py."
    )
    _write_banner(session.config, "sys.path check failed", message)
    # TODO: enable separately after magic import is merged.
    # if exitstatus == pytest.ExitCode.OK:
    #     session.exitstatus = pytest.ExitCode.USAGE_ERROR
