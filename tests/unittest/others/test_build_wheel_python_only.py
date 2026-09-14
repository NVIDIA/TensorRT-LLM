# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``build_wheel.py --python_only`` and the FlashInfer patch step.

``--python_only`` sets up a checkout whose compiled artifacts come from
somewhere else -- a worktree sharing another build's ``.so`` files, or a
reproducibility check whose ``git diff -- cpp/`` against the artifact commit is
empty. It builds the venv, installs the requirements, puts the checkout on the
venv's path with a ``.pth``, writes the console scripts and applies the
FlashInfer header patches, and compiles nothing.

The patch step is the part with teeth. FlashInfer compiles its attention
kernels from headers that ship inside the installed package, so the in-tree
fixes live in that package and any install that replaces FlashInfer reverts
them. ``patch`` applies a hunk at a time, so a rejected patch leaves some
headers stock and others changed -- a build that reported success over that
would ship kernels nobody can account for. Every non-zero exit except "no
FlashInfer installed" therefore fails the build.

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
    assert _parse(build_wheel, ["--python_only"]).python_only is True


# --------------------------------------------------------------------------
# apply_flashinfer_patches: what fails the build and what does not
# --------------------------------------------------------------------------
@pytest.fixture
def patch_script(build_wheel, monkeypatch, tmp_path):
    """Pretend the in-tree patch script exists, and record how it is run."""
    calls = []

    def fake_run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=fake_run.returncode)

    fake_run.returncode = 0
    monkeypatch.setattr(build_wheel, "run", fake_run)
    monkeypatch.setattr(Path, "is_file", lambda self: True)
    return fake_run, calls


def test_a_clean_patch_run_is_a_pass(build_wheel, patch_script):
    fake_run, calls = patch_script
    fake_run.returncode = 0

    build_wheel.apply_flashinfer_patches("/venv/bin/python")

    assert len(calls) == 1
    assert "/venv/bin/python" in calls[0][0]
    assert "apply-flashinfer-patches.py" in calls[0][0]


def test_an_absent_flashinfer_is_not_a_build_failure(build_wheel, patch_script):
    """Plenty of builds never install FlashInfer. That one exit code is benign."""
    fake_run, _ = patch_script
    fake_run.returncode = build_wheel.FLASHINFER_ABSENT

    build_wheel.apply_flashinfer_patches("/venv/bin/python")


def test_the_benign_exit_code_is_reserved_and_not_zero(build_wheel):
    """A rejected patch must not be able to masquerade as 'not installed'."""
    assert build_wheel.FLASHINFER_ABSENT == 4
    assert build_wheel.FLASHINFER_ABSENT != 0


@pytest.mark.parametrize("returncode", [1, 2, 3, 5, 127])
def test_a_patch_that_does_not_apply_fails_the_build(build_wheel, patch_script, returncode):
    """The regression this exists for: the step used to warn and return success.

    A half-patched install is the dangerous state -- some headers fixed, some
    stock, and a green build over the top of it.
    """
    fake_run, _ = patch_script
    fake_run.returncode = returncode

    with pytest.raises(RuntimeError, match="FlashInfer header patches failed"):
        build_wheel.apply_flashinfer_patches("/venv/bin/python")


def test_the_failure_says_how_to_inspect_and_undo(build_wheel, patch_script):
    fake_run, _ = patch_script
    fake_run.returncode = 1

    with pytest.raises(RuntimeError) as excinfo:
        build_wheel.apply_flashinfer_patches("/venv/bin/python")

    message = str(excinfo.value)
    assert "--check" in message and "--revert" in message
    assert str(fake_run.returncode) in message


def test_a_missing_patch_script_fails_the_build(build_wheel, monkeypatch):
    """A checkout that lost the patches must not build as if it had them."""
    monkeypatch.setattr(Path, "is_file", lambda self: False)

    with pytest.raises(RuntimeError, match="is missing"):
        build_wheel.apply_flashinfer_patches("/venv/bin/python")


# --------------------------------------------------------------------------
# setup_python_only
# --------------------------------------------------------------------------
@pytest.fixture
def python_only_env(build_wheel, monkeypatch, tmp_path):
    """A fake venv and checkout, with the two subprocess calls stubbed."""
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
    patched_with = []
    monkeypatch.setattr(build_wheel, "apply_flashinfer_patches", patched_with.append)

    return SimpleNamespace(
        project_dir=project_dir,
        venv_python=venv_python,
        site_packages=site_packages,
        patched_with=patched_with,
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


def test_the_patches_are_applied_to_the_venv_interpreter(build_wheel, python_only_env):
    """Not ``sys.executable``.

    On a fresh checkout ``build_wheel.py`` has to be launched with the system
    interpreter, so using ``sys.executable`` here patched the *system*
    FlashInfer and left the venv's copy stock -- silently running unpatched
    headers. This path has no compile to hide behind, so it applies them here
    or not at all.
    """
    env = python_only_env
    build_wheel.setup_python_only(env.project_dir, env.venv_python)

    assert env.patched_with == [str(env.venv_python)]
    assert str(env.venv_python) != sys.executable
