# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``build_wheel.py --install`` must use the venv interpreter.

A fresh checkout has to start ``build_wheel.py`` with the system interpreter,
so installing with ``sys.executable`` puts the package into the system
site-packages and leaves the venv that ``setup_venv()`` just created without
it. The install has to run with ``venv_python``.
"""

import importlib.util
import sys
from pathlib import Path

import pytest

_BUILD_WHEEL = Path(__file__).resolve().parents[3] / "scripts" / "build_wheel.py"


@pytest.fixture(scope="module")
def build_wheel():
    """``scripts/build_wheel.py`` loaded by path.

    It is a script rather than a package member. Importing it runs only its
    imports and constants; ``main()`` is behind the usual ``__main__`` guard.
    """
    assert _BUILD_WHEEL.is_file(), _BUILD_WHEEL
    spec = importlib.util.spec_from_file_location("_build_wheel_under_test", _BUILD_WHEEL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


def test_install_runs_with_the_venv_interpreter(build_wheel, monkeypatch, tmp_path):
    commands = []
    monkeypatch.setattr(build_wheel, "build_run", lambda cmd, **kwargs: commands.append(cmd))

    venv_python = tmp_path / "venv" / "bin" / "python"
    build_wheel.install_editable_package(venv_python)

    assert len(commands) == 1
    command = commands[0]
    assert command.startswith(f'"{venv_python}" -m pip install -e ')
    assert str(venv_python) in command
    assert sys.executable not in command
