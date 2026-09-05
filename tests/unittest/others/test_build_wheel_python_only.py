# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``build_wheel.py --python_only``.

``--python_only`` sets up a checkout whose compiled artifacts come from
somewhere else -- a worktree sharing another build's ``.so`` files, or a
dev/CI container that already ships them. It builds the venv, installs the
requirements, puts the checkout on the venv's path with a ``.pth``, writes the
console scripts, and compiles nothing.

No filesystem beyond ``tmp_path`` and no subprocesses: the interpreter calls
are mocked.
"""

import importlib.util
import sys
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace

import pytest

_BUILD_WHEEL = Path(__file__).resolve().parents[3] / "scripts" / "build_wheel.py"


@pytest.fixture(scope="module")
def build_wheel():
    """``scripts/build_wheel.py`` imported as a module.

    It is a script, not a package member, so it is loaded by path. Importing
    it runs only its imports and constants; ``main()`` is behind the usual
    ``__main__`` guard.
    """
    assert _BUILD_WHEEL.is_file(), _BUILD_WHEEL
    spec = importlib.util.spec_from_file_location("_build_wheel_under_test", _BUILD_WHEEL)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
def _parse(build_wheel, argv):
    parser = ArgumentParser()
    build_wheel.add_arguments(parser)
    return parser.parse_args(argv)


def test_python_only_is_accepted_and_defaults_off(build_wheel):
    assert _parse(build_wheel, []).python_only is False
    assert _parse(build_wheel, ["--python_only"]).python_only is True


def test_python_only_is_not_cpp_only(build_wheel):
    """They are opposites, and neither implies the other."""
    args = _parse(build_wheel, ["--python_only"])
    assert args.cpp_only is False
    assert _parse(build_wheel, ["--cpp_only"]).python_only is False


def test_python_only_reaches_main_as_a_keyword(build_wheel):
    """``main(**vars(args))`` is how the script hands the flag over."""
    import inspect

    signature = inspect.signature(build_wheel.main)
    assert "python_only" in signature.parameters
    assert signature.parameters["python_only"].default is False


# --------------------------------------------------------------------------
# setup_python_only
# --------------------------------------------------------------------------
@pytest.fixture
def python_only_env(build_wheel, monkeypatch, tmp_path):
    """A fake venv and checkout, with the subprocess call stubbed."""
    project_dir = tmp_path / "checkout"
    (project_dir / "tensorrt_llm" / "llmapi").mkdir(parents=True)
    (project_dir / "tensorrt_llm" / "llmapi" / "trtllm-llmapi-launch").write_text(
        "#!/bin/sh\n", encoding="utf-8"
    )

    site_packages = tmp_path / "venv" / "lib" / "site-packages"
    site_packages.mkdir(parents=True)
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(build_wheel, "check_output", lambda *a, **k: f"{site_packages}\n")

    return SimpleNamespace(
        project_dir=project_dir,
        venv_python=venv_python,
        site_packages=site_packages,
    )


def test_the_pth_points_at_this_checkout(build_wheel, python_only_env):
    """Nothing is copied from another checkout; the venv points back here."""
    env = python_only_env
    build_wheel.setup_python_only(env.project_dir, env.venv_python)

    pth = env.site_packages / "tensorrt_llm_checkout.pth"
    assert pth.read_text(encoding="utf-8").strip() == str(env.project_dir)


def test_the_console_scripts_and_launcher_are_written(build_wheel, python_only_env):
    env = python_only_env
    build_wheel.setup_python_only(env.project_dir, env.venv_python)

    bin_dir = env.venv_python.parent
    for name in build_wheel._CONSOLE_SCRIPTS:
        script = bin_dir / name
        assert script.is_file(), name
        assert script.read_text(encoding="utf-8").startswith(f"#!{env.venv_python}")
    launcher = bin_dir / "trtllm-llmapi-launch"
    assert launcher.is_symlink()
    assert Path(launcher.resolve()).is_file()


def test_the_launcher_symlink_is_replaced_not_stacked(build_wheel, python_only_env):
    """Re-running the setup on an existing venv has to be idempotent."""
    env = python_only_env
    build_wheel.setup_python_only(env.project_dir, env.venv_python)
    build_wheel.setup_python_only(env.project_dir, env.venv_python)

    launcher = env.venv_python.parent / "trtllm-llmapi-launch"
    assert launcher.is_symlink()


def test_a_checkout_without_the_launcher_still_sets_up(build_wheel, python_only_env):
    env = python_only_env
    (env.project_dir / "tensorrt_llm" / "llmapi" / "trtllm-llmapi-launch").unlink()

    build_wheel.setup_python_only(env.project_dir, env.venv_python)

    assert (env.site_packages / "tensorrt_llm_checkout.pth").is_file()
    assert not (env.venv_python.parent / "trtllm-llmapi-launch").exists()
