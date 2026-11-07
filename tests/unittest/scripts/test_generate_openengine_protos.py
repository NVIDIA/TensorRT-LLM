# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Safety tests for OpenEngine binding publication."""

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = PROJECT_ROOT / "scripts/generate_openengine_protos.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_openengine_protos", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_publish_refuses_to_replace_an_unowned_directory(tmp_path):
    generator = _load_generator()
    staged_output = tmp_path / "staged"
    staged_output.mkdir()
    (staged_output / "generated.py").write_text("generated\n", encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    sentinel = output_dir / "sentinel"
    sentinel.write_text("keep\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="not owned by the OpenEngine generator"):
        generator._publish_generated_output(staged_output, output_dir)

    assert sentinel.read_text(encoding="utf-8") == "keep\n"
    assert staged_output.is_dir()


def test_main_replaces_an_output_symlink_without_touching_its_target(tmp_path, monkeypatch):
    generator = _load_generator()
    symlink_target = tmp_path / "target"
    symlink_target.mkdir()
    sentinel = symlink_target / "sentinel"
    sentinel.write_text("keep\n", encoding="utf-8")
    output_dir = tmp_path / "output"
    try:
        output_dir.symlink_to(symlink_target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")
    staged_output = tmp_path / "staged"
    staged_output.mkdir()
    (staged_output / "generated.py").write_text("generated\n", encoding="utf-8")

    monkeypatch.setattr(
        generator,
        "_generate",
        lambda project_root, generated_output: generator._publish_generated_output(
            staged_output, generated_output
        ),
    )
    generator.main(["--project-root", str(PROJECT_ROOT), "--output", str(output_dir)])

    assert not output_dir.is_symlink()
    assert (output_dir / "generated.py").is_file()
    assert sentinel.read_text(encoding="utf-8") == "keep\n"


def test_check_detects_stale_tracked_bindings(tmp_path, monkeypatch):
    generator = _load_generator()
    output_dir = tmp_path / "tracked"
    output_dir.mkdir()
    (output_dir / "generated.py").write_text("stale\n", encoding="utf-8")

    def generate(_project_root, expected_dir):
        expected_dir.mkdir()
        (expected_dir / "generated.py").write_text("current\n", encoding="utf-8")

    monkeypatch.setattr(generator, "_generate", generate)

    with pytest.raises(
        RuntimeError, match=r"(?s)Tracked OpenEngine bindings.*changed generated\.py"
    ):
        generator._check(PROJECT_ROOT, output_dir)

    assert (output_dir / "generated.py").read_text(encoding="utf-8") == "stale\n"


def test_check_ignores_interpreter_bytecode(tmp_path, monkeypatch):
    generator = _load_generator()
    output_dir = tmp_path / "tracked"
    output_dir.mkdir()
    (output_dir / "generated.py").write_text("current\n", encoding="utf-8")
    bytecode_dir = output_dir / "__pycache__"
    bytecode_dir.mkdir()
    (bytecode_dir / "generated.cpython-312.pyc").write_bytes(b"bytecode")

    def generate(_project_root, expected_dir):
        expected_dir.mkdir()
        (expected_dir / "generated.py").write_text("current\n", encoding="utf-8")

    monkeypatch.setattr(generator, "_generate", generate)

    generator._check(PROJECT_ROOT, output_dir)


def test_isolated_check_forwards_check_mode(tmp_path, monkeypatch):
    generator = _load_generator()
    requirements_path = PROJECT_ROOT / "requirements-build-openengine.txt"
    requirements_digest = generator._sha256(requirements_path)
    environment_key = (
        f"py{generator.sys.version_info.major}{generator.sys.version_info.minor}-"
        f"{requirements_digest[:16]}"
    )
    venv_dir = tmp_path / "tools" / environment_key
    python = generator._venv_python(venv_dir)
    python.parent.mkdir(parents=True)
    python.touch()
    (venv_dir / ".requirements.sha256").write_text(requirements_digest + "\n", encoding="utf-8")
    commands = []
    monkeypatch.setattr(
        generator.subprocess,
        "run",
        lambda command, **_kwargs: commands.append(command),
    )

    generator.main(
        [
            "--project-root",
            str(PROJECT_ROOT),
            "--output",
            str(tmp_path / "output"),
            "--tool-env-root",
            str(tmp_path / "tools"),
            "--check",
        ]
    )

    assert len(commands) == 1
    assert "--check" in commands[0]
