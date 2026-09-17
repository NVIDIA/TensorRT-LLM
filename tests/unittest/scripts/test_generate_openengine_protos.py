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


def test_publish_replaces_legacy_output_after_init_metadata_changes(tmp_path):
    generator = _load_generator()
    staged_output = tmp_path / "staged"
    staged_output.mkdir()
    (staged_output / "generated.py").write_text("new\n", encoding="utf-8")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    legacy_init = (
        "# An older generated copyright header.\n\n"
        '"""Private OpenEngine bindings generated during the build."""\n'
    )
    assert legacy_init != generator._GENERATED_INIT_CONTENT
    (output_dir / "__init__.py").write_text(legacy_init, encoding="utf-8")
    stale_file = output_dir / "stale.py"
    stale_file.write_text("stale\n", encoding="utf-8")

    generator._publish_generated_output(staged_output, output_dir)

    assert (output_dir / "generated.py").read_text(encoding="utf-8") == "new\n"
    assert (output_dir / generator._OWNERSHIP_MARKER_NAME).is_file()
    assert not stale_file.exists()


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
