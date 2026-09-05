# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the CBTS coverage bootstrap.

Covers jenkins/scripts/cbts/cbts_injectors and jenkins/scripts/cbts/cbts/coverage/collection.
These pin the properties that fail silently when broken: a hookless plugin loses every
per-test context, and a leaked role makes a child answer to its parent's.
"""

import os
import shutil
import socket
import subprocess
import sys
import threading
from pathlib import Path

import pytest

__extra_import_path__ = ["~/jenkins/scripts/cbts"]
from cbts.coverage.collection import process_roles
from cbts.coverage.collection.pystart import PyStartTracker

pytestmark = pytest.mark.cpu_only

_CBTS_OUTER = Path(__file__).resolve().parents[4] / "jenkins" / "scripts" / "cbts"
_CBTS_INJECTORS = _CBTS_OUTER / "cbts_injectors"
# Both entries are needed: cbts_injectors/ for bare `sitecustomize`/`cbts_plugin`
# imports (what Python's site machinery and pytest -p actually load by name), the
# outer dir for the `cbts.*` package those two files import normally.
_BOOTSTRAP_PYTHONPATH = os.pathsep.join([str(_CBTS_INJECTORS), str(_CBTS_OUTER)])

_PROBE_PLUGIN = """
import os

import sitecustomize
from cbts.coverage.collection.channel import announced_address


def pytest_runtest_teardown(item):
    print(f"[probe] ctx={sitecustomize._tracker._ctx}")
    # pytest_configure bound the channel and published where it is.
    address = announced_address()
    print(f"[probe] channel={'yes' if address and os.path.exists(address) else 'no'}")
"""


@pytest.mark.parametrize(
    ("orig_argv", "expected"),
    [
        ([sys.executable, "-m", "mpi4py.futures.server"], True),
        # FlashInfer per-process workspace isolation overrides python_args to "-c <bootstrap>",
        # but mpi4py's client_spawn still appends the trailing "-m mpi4py.futures.server"
        # tokens (inert under -c, but still present in orig_argv).
        (
            [
                sys.executable,
                "-c",
                "<bootstrap script>",
                "/some/path",
                "-m",
                "mpi4py.futures.server",
            ],
            True,
        ),
        ([sys.executable, "-m", "pytest"], False),
        ([sys.executable, "trtllm-serve"], False),
        ([sys.executable, "-c", "print(1)"], False),
        ([], False),
    ],
)
def test_is_mpi_pool_worker_matches_the_spawn_signature(monkeypatch, orig_argv, expected):
    monkeypatch.setattr(sys, "orig_argv", orig_argv, raising=False)
    assert process_roles.is_mpi_pool_worker() is expected


def test_note_taint_none_means_self_but_empty_string_is_preserved(tmp_path):
    """Regression: `process_uid or self.process_uid` conflated an unidentified subscriber.

    An empty-string uid used to be indistinguishable from "this tracker" -- only `None`
    may substitute `self.process_uid`.
    """
    tracker = PyStartTracker([], str(tmp_path))

    tracker.note_taint(None, "test_a", "incomplete", "worker_activation_timeout")
    tracker.note_taint("", "test_b", "incomplete", "context_channel_unreachable")
    tracker.note_taint("some-other-pid", "test_c", "attribution", "context_not_acknowledged")

    by_test = {nodeid: process_uid for process_uid, nodeid, _kind, _reason in tracker._taints}
    assert by_test["test_a"] == tracker.process_uid
    assert by_test["test_b"] == ""
    assert by_test["test_c"] == "some-other-pid"


_ROLE_REPORTER = """
def test_role(capsys):
    import os, subprocess, sys
    child = subprocess.run(
        [sys.executable, "-c", "import os; print(os.environ.get('CBTS_PROCESS_ROLE', '<unset>'))"],
        capture_output=True, text=True, check=True,
    )
    with capsys.disabled():
        print(f"[probe] self={os.environ.get('CBTS_PROCESS_ROLE', '<unset>')} "
              f"child={child.stdout.strip()}")
"""


@pytest.fixture
def cbts_env(tmp_path):
    """A minimal instrumented workspace: rcfile, dummy test, probe plugin, env."""
    source_root = tmp_path / "product"
    source_root.mkdir()
    rcfile = tmp_path / ".coveragerc"
    rcfile.write_text(
        f"[run]\nsource =\n    {source_root}/\ndata_file = {tmp_path}/.coverage.unit\n"
    )
    (tmp_path / "test_dummy.py").write_text("def test_a():\n    pass\n")
    (tmp_path / "probe_plugin.py").write_text(_PROBE_PLUGIN)

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([_BOOTSTRAP_PYTHONPATH, str(tmp_path)])
    env["CBTS_COVERAGE_CONFIG"] = str(rcfile)
    env["CBTS_STAGE"] = "unit"
    env.pop("CBTS_TEST_ID", None)
    env.pop("CBTS_CONTEXT_SOCKET", None)
    return env, tmp_path


def _invocations():
    """Interpreter spellings that must all reach the same plugin registration."""
    yield pytest.param([sys.executable, "-m", "pytest"], id="python-m-pytest")
    # An interpreter option before -m pushes "pytest" further along argv; positional
    # argv scanning used to miss it and silently disable per-test attribution.
    yield pytest.param([sys.executable, "-X", "importtime", "-m", "pytest"], id="python-X-m-pytest")
    console_script = shutil.which("pytest")
    if console_script:
        yield pytest.param([console_script], id="pytest-console-script")


@pytest.mark.parametrize("launcher", list(_invocations()))
def test_plugin_registers_regardless_of_invocation(cbts_env, launcher):
    env, workdir = cbts_env
    env["CBTS_PROCESS_ROLE"] = "outer_pytest"
    result = subprocess.run(
        launcher
        + [
            "-q",
            "-s",
            "-p",
            "no:cacheprovider",
            "-p",
            "cbts_plugin",
            "-p",
            "probe_plugin",
            "test_dummy.py",
        ],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # The context only reaches the tracker through the plugin's runtest hook.
    assert "[probe] ctx=test_dummy.py::test_a" in result.stdout, (
        f"plugin hooks never ran\n{result.stdout}{result.stderr}"
    )
    # pytest_configure bound the channel and published its address for subprocesses.
    assert "[probe] channel=yes" in result.stdout, result.stdout


def test_process_role_is_consumed_not_inherited(cbts_env):
    """A child must not answer to its parent's role."""
    env, workdir = cbts_env
    env["CBTS_PROCESS_ROLE"] = "outer_pytest"
    (workdir / "test_role.py").write_text(_ROLE_REPORTER)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider", "test_role.py"],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[probe] self=<unset> child=<unset>" in result.stdout, result.stdout


def test_build_and_ray_processes_opt_out(cbts_env):
    """Launch targets that must disable instrumentation for themselves and their subtree."""
    env, workdir = cbts_env
    probe = "import os; print(os.environ.get('CBTS_COVERAGE_CONFIG', '<unset>'))"
    (workdir / "setup.py").write_text(probe)
    (workdir / "default_worker.py").write_text(probe)
    (workdir / "regular.py").write_text(probe)

    for script, expected in (
        ("setup.py", "<unset>"),
        ("default_worker.py", "<unset>"),
        ("regular.py", env["CBTS_COVERAGE_CONFIG"]),
    ):
        result = subprocess.run(
            [sys.executable, script],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout.strip() == expected, f"{script}: {result.stdout}"


def _write_fakeprod(workdir):
    """A tiny product package: a class (import-time body) plus a function (call-time body)."""
    package = workdir / "product" / "fakeprod"
    package.mkdir()
    (package / "__init__.py").write_text(
        "class AtImportTime:\n    pass\n\n\ndef work():\n    return 1\n"
    )
    # The source root is the package itself, so "fakeprod" is the top-level name watched for.
    (workdir / ".coveragerc").write_text(
        f"[run]\nsource =\n    {package}/\ndata_file = {workdir}/.coverage.unit\n"
    )


_RECORD_PROBE = (
    "print('before', sitecustomize._tracker._active)\n"
    "import fakeprod\n"
    "print('after', sitecustomize._tracker._active)\n"
    "fakeprod.work()\n"
    "print('recorded', sorted(q for _f, q in "
    "sitecustomize._tracker._data.get('suite.py::test_x', ())))\n"
)


def test_a_generic_product_process_activates_immediately(cbts_env):
    """Only an actual MPI pool worker defers activation.

    Any other product process (trtllm-serve, disagg helpers, ...) has no wait_shutdown
    barrier to protect and activates right away.
    """
    env, workdir = cbts_env
    _write_fakeprod(workdir)
    env["PYTHONPATH"] = os.pathsep.join([_BOOTSTRAP_PYTHONPATH, str(workdir / "product")])
    env["CBTS_TEST_ID"] = "suite.py::test_x"
    env.pop("CBTS_PROCESS_ROLE", None)

    result = subprocess.run(
        [sys.executable, "-c", "import sitecustomize\n" + _RECORD_PROBE],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "before True" in result.stdout, result.stdout
    assert "after True" in result.stdout, result.stdout
    # Nothing is deferred, so the class body's import-time execution is recorded too.
    assert "recorded ['<module>', 'AtImportTime', 'work']" in result.stdout, result.stdout


def test_an_mpi_pool_worker_activates_when_the_framework_import_finishes(cbts_env):
    """The one process type with a wait_shutdown barrier to protect.

    Defer PY_START until the framework import completes, then record on that thread.
    """
    env, workdir = cbts_env
    _write_fakeprod(workdir)
    env["PYTHONPATH"] = os.pathsep.join([_BOOTSTRAP_PYTHONPATH, str(workdir / "product")])
    env["CBTS_TEST_ID"] = "suite.py::test_x"
    env.pop("CBTS_PROCESS_ROLE", None)

    # `-S` skips automatic sitecustomize import at startup (PYTHONPATH entries still land on
    # sys.path regardless), so this script can fake sys.orig_argv to the mpi4py.futures pool
    # worker spawn signature *before* triggering site processing itself via site.main() --
    # otherwise sitecustomize would already have inspected the real argv by the time any -c
    # code got control.
    script = (
        "import sys\n"
        "sys.orig_argv = [sys.executable, '-m', 'mpi4py.futures.server']\n"
        "import site\n"
        "site.main()\n"
        "import sitecustomize\n" + _RECORD_PROBE
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "before False" in result.stdout, result.stdout
    # Activation happens inline on the importing thread, so nothing runs unrecorded after it.
    assert "after True" in result.stdout, result.stdout
    # work() is entered after activation; AtImportTime's body ran during the deferred import.
    assert "recorded ['work']" in result.stdout, result.stdout


def test_an_mpi_pool_worker_that_hits_the_activation_timeout_is_tainted(cbts_env):
    """A backstop-triggered activation is an anomaly, not the routine unrecorded-import gap.

    The default budget is already past the documented worst-case cold import, so hitting it
    at all signals an abnormally slow run and gets an INCOMPLETE taint.
    """
    env, workdir = cbts_env
    # Nothing ever imports the watched "fakeprod" name, so only the backstop can activate.
    # CBTS_WORKER_ACTIVATE_MAX_SECONDS is floored at 1.0s in sitecustomize.py, so this is the
    # fastest the backstop can fire.
    env["CBTS_WORKER_ACTIVATE_MAX_SECONDS"] = "0.05"
    env["CBTS_TEST_ID"] = "suite.py::test_x"
    env.pop("CBTS_PROCESS_ROLE", None)

    script = (
        "import sys\n"
        "sys.orig_argv = [sys.executable, '-m', 'mpi4py.futures.server']\n"
        "import site\n"
        "site.main()\n"
        "import sitecustomize\n"
        "import time; time.sleep(1.3)\n"  # let the (floored 1.0s) backstop thread fire before we look
        "print('active', sitecustomize._tracker._active)\n"
        "print('taints', sorted((k, r) for _p, _n, k, r in sitecustomize._tracker._taints))\n"
    )
    result = subprocess.run(
        [sys.executable, "-S", "-c", script],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "active True" in result.stdout, result.stdout
    assert "taints [('incomplete', 'worker_activation_timeout')]" in result.stdout, result.stdout


def test_a_worker_that_joins_after_stop_does_not_leave_py_start_enabled(cbts_env):
    """Regression: ContextSubscriber.subscribe() can run on_stop synchronously.

    It reads the first frame inline, and here the (fake) producer answers a new
    connection with the STOP frame it already sent to everyone else (real ContextServer
    behavior once self._stopped is set: cbts_channel.py's _accept()). _activate_tracker
    must not then re-enable PY_START for a session that already saved and will never save
    again.
    """
    env, workdir = cbts_env
    _write_fakeprod(workdir)

    address = str(workdir / "fake-producer.sock")
    listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    listener.bind(address)
    listener.listen(1)

    def _fake_already_stopped_producer():
        conn, _ = listener.accept()
        conn.recv(4096)  # drain the subscriber's IDENTITY frame; unread data + close resets
        conn.sendall(b"X 1\n")  # a STOP frame, as if this join arrived after close()'s STOP
        conn.close()

    threading.Thread(target=_fake_already_stopped_producer, daemon=True).start()

    env["PYTHONPATH"] = os.pathsep.join([_BOOTSTRAP_PYTHONPATH, str(workdir / "product")])
    env["CBTS_CONTEXT_SOCKET"] = address
    env["CBTS_TEST_ID"] = "suite.py::test_x"
    env.pop("CBTS_PROCESS_ROLE", None)

    script = (
        "import sys\n"
        "sys.orig_argv = [sys.executable, '-m', 'mpi4py.futures.server']\n"
        "import site\n"
        "site.main()\n"
        "import sitecustomize\n"
        "import fakeprod\n"
        "print('active', sitecustomize._tracker._active)\n"
    )
    try:
        result = subprocess.run(
            [sys.executable, "-S", "-c", script],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
        )
    finally:
        listener.close()
    assert result.returncode == 0, result.stdout + result.stderr
    assert "active False" in result.stdout, result.stdout


def test_pool_patch_is_installed_as_mpi4py_futures_is_imported(cbts_env):
    """No window between the executor class existing and its constructor being counted."""
    pytest.importorskip("mpi4py.futures")
    env, workdir = cbts_env
    env.pop("CBTS_PROCESS_ROLE", None)
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sitecustomize\n"
            "from mpi4py.futures import MPIPoolExecutor\n"
            "print('patched', getattr(MPIPoolExecutor.__init__, '_cbts_patched_pool_init', False))\n",
        ],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "patched True" in result.stdout, result.stdout


def test_program_arguments_are_not_read_as_interpreter_options(cbts_env):
    """A script's own ``-m`` must not be mistaken for the interpreter's."""
    env, workdir = cbts_env
    (workdir / "regular.py").write_text(
        "import os; print(os.environ.get('CBTS_COVERAGE_CONFIG', '<unset>'))"
    )
    result = subprocess.run(
        [sys.executable, "regular.py", "-m", "pip"],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == env["CBTS_COVERAGE_CONFIG"], result.stdout


def test_a_session_without_a_channel_taints_every_test(cbts_env):
    """No channel means no subprocess ever left its inherited context."""
    import sqlite3

    env, workdir = cbts_env
    env["CBTS_PROCESS_ROLE"] = "outer_pytest"
    # An address the producer cannot bind, so pytest_configure gets no channel.
    env["CBTS_CONTEXT_SOCKET"] = "/proc/cbts-cannot-bind-here/ctx.sock"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "cbts_plugin",
            "test_dummy.py",
        ],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # The reason reaches the operator, not just the fact.
    assert "could not bind the context channel" in result.stderr, result.stderr
    assert "cbts-cannot-bind-here" in result.stderr, result.stderr
    assert "Error" in result.stderr, "the underlying OSError was swallowed:\n" + result.stderr

    # The taint rides in the producer's own leaf database, beside its touches.
    rows = []
    for leaf in workdir.glob(".cbtscov.*.sqlite"):
        with sqlite3.connect(f"file://{leaf}?mode=ro", uri=True) as db:
            rows += db.execute("SELECT test, kind, reason FROM taint_rows").fetchall()
    # Stage-scoped (empty test) and both kinds: without a channel every test may hold
    # another's rows and be missing its own, and nothing narrower can be named.
    assert sorted(rows) == [
        ("", "attribution", "no_context_channel"),
        ("", "incomplete", "no_context_channel"),
    ], f"{rows}\n{result.stdout}{result.stderr}"


_HARD_EXIT_PLUGIN = """
import os


def pytest_unconfigure(config):
    # Skips atexit, as a killed or hard-exiting process would.
    os._exit(0)
"""


def test_session_record_is_on_disk_without_the_atexit_save(cbts_env):
    """Sessionfinish writes touches, outcomes and taints; atexit is only a backstop."""
    import sqlite3

    env, workdir = cbts_env
    env["CBTS_PROCESS_ROLE"] = "outer_pytest"
    # No channel, so the session is tainted and the taint must survive alongside the rest.
    env["CBTS_CONTEXT_SOCKET"] = "/proc/cbts-cannot-bind-here/ctx.sock"
    (workdir / "hard_exit.py").write_text(_HARD_EXIT_PLUGIN)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-p",
            "cbts_plugin",
            "-p",
            "hard_exit",
            "test_dummy.py",
        ],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    outcomes, taints = [], []
    for leaf in workdir.glob(".cbtscov.*.sqlite"):
        with sqlite3.connect(f"file://{leaf}?mode=ro", uri=True) as db:
            outcomes += db.execute(
                "SELECT test, outcome FROM test_case_meta WHERE test != ''"
            ).fetchall()
            taints += db.execute("SELECT test, kind, reason FROM taint_rows").fetchall()
    assert outcomes == [("unit/test_dummy.py::test_a", "passed")], outcomes
    assert sorted(taints) == [
        ("", "attribution", "no_context_channel"),
        ("", "incomplete", "no_context_channel"),
    ], taints
