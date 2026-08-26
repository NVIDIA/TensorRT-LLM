# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the CBTS coverage bootstrap (jenkins/scripts/cbts/coverage_utils).

These pin the properties that fail silently when broken: a hookless plugin loses
every per-test context, and a leaked role makes a child answer to its parent's.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

COVERAGE_UTILS = (Path(__file__).resolve().parents[3] / "jenkins" / "scripts" / "cbts" /
                  "coverage_utils")

_PROBE_PLUGIN = """
import os

import sitecustomize
from cbts_channel import announced_address


def pytest_runtest_teardown(item):
    print(f"[probe] ctx={sitecustomize._tracker._ctx}")
    # pytest_configure bound the channel and published where it is.
    address = announced_address()
    print(f"[probe] channel={'yes' if address and os.path.exists(address) else 'no'}")
"""

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
    rcfile.write_text(f"[run]\nsource =\n    {source_root}/\n"
                      f"data_file = {tmp_path}/.coverage.unit\n")
    (tmp_path / "test_dummy.py").write_text("def test_a():\n    pass\n")
    (tmp_path / "probe_plugin.py").write_text(_PROBE_PLUGIN)

    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(COVERAGE_UTILS), str(tmp_path)])
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
        launcher + [
            "-q", "-s", "-p", "no:cacheprovider", "-p", "cbts_plugin", "-p", "probe_plugin",
            "test_dummy.py"
        ],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # The context only reaches the tracker through the plugin's runtest hook.
    assert "[probe] ctx=test_dummy.py::test_a" in result.stdout, (
        f"plugin hooks never ran\n{result.stdout}{result.stderr}")
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

    for script, expected in (("setup.py", "<unset>"), ("default_worker.py", "<unset>"),
                             ("regular.py", env["CBTS_COVERAGE_CONFIG"])):
        result = subprocess.run([sys.executable, script],
                                cwd=workdir,
                                env=env,
                                capture_output=True,
                                text=True,
                                check=True)
        assert result.stdout.strip() == expected, f"{script}: {result.stdout}"


def test_unlabelled_process_activates_when_the_framework_import_finishes(cbts_env):
    """A product process defers until the framework is imported, then records on that thread."""
    env, workdir = cbts_env
    package = workdir / "product" / "fakeprod"
    package.mkdir()
    (package / "__init__.py").write_text("class AtImportTime:\n    pass\n\n\ndef work():\n    return 1\n")
    # The source root is the package itself, so "fakeprod" is the top-level name watched for.
    (workdir / ".coveragerc").write_text(f"[run]\nsource =\n    {package}/\n"
                                         f"data_file = {workdir}/.coverage.unit\n")
    env["PYTHONPATH"] = os.pathsep.join([str(COVERAGE_UTILS), str(workdir / "product")])
    env["CBTS_TEST_ID"] = "suite.py::test_x"
    env.pop("CBTS_PROCESS_ROLE", None)

    result = subprocess.run(
        [
            sys.executable, "-c", "import sitecustomize\n"
            "print('before', sitecustomize._tracker._active)\n"
            "import fakeprod\n"
            "print('after', sitecustomize._tracker._active)\n"
            "fakeprod.work()\n"
            "print('recorded', sorted(q for _f, q in "
            "sitecustomize._tracker._data.get('suite.py::test_x', ())))\n"
        ],
        cwd=workdir,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "before False" in result.stdout, result.stdout
    # Activation happens inline on the importing thread, so nothing runs unrecorded after it.
    assert "after True" in result.stdout, result.stdout
    # work() is entered after activation; AtImportTime's body ran during the import.
    assert "recorded ['work']" in result.stdout, result.stdout


def test_pool_patch_is_installed_as_mpi4py_futures_is_imported(cbts_env):
    """No window between the executor class existing and its constructor being counted."""
    pytest.importorskip("mpi4py.futures")
    env, workdir = cbts_env
    env.pop("CBTS_PROCESS_ROLE", None)
    result = subprocess.run(
        [
            sys.executable, "-c", "import sitecustomize\n"
            "from mpi4py.futures import MPIPoolExecutor\n"
            "print('patched', getattr(MPIPoolExecutor.__init__, '_cbts_patched_pool_init', False))\n"
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
        "import os; print(os.environ.get('CBTS_COVERAGE_CONFIG', '<unset>'))")
    result = subprocess.run([sys.executable, "regular.py", "-m", "pip"],
                            cwd=workdir,
                            env=env,
                            capture_output=True,
                            text=True,
                            check=True)
    assert result.stdout.strip() == env["CBTS_COVERAGE_CONFIG"], result.stdout
