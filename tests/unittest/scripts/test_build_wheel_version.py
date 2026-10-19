# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only checks for source wheel version selection."""

import importlib.util
from argparse import ArgumentParser
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only


@pytest.fixture(scope="module")
def build_wheel():
    path = Path(__file__).resolve().parents[3] / "scripts" / "build_wheel.py"
    spec = importlib.util.spec_from_file_location("build_wheel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "environment,cli,expected",
    [
        (None, [], "1.4.0a0"),
        ("", [], "1.4.0a0"),
        ("1.4.0a0.dev202610190100", [], "1.4.0a0.dev202610190100"),
        ("1.4.0alpha0.dev202610190100", [], "1.4.0a0.dev202610190100"),
        (".dev202610190100", [], "1.4.0a0.dev202610190100"),
        ("+local", [], "1.4.0a0+local"),
        (None, ["--version-override", "1.5.0"], "1.5.0"),
        ("1.4.0a0", ["--version-override", "1.5.0"], "1.5.0"),
        ("1.5.0", ["--version-override", ""], "1.4.0a0"),
    ],
)
def test_version_selection(build_wheel, monkeypatch, tmp_path, environment, cli, expected):
    if environment is None:
        monkeypatch.delenv("TRTLLM_BUILD_VERSION", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_BUILD_VERSION", environment)
    parser = ArgumentParser()
    build_wheel.add_arguments(parser)
    version_file = tmp_path / "tensorrt_llm" / "version.py"
    version_file.parent.mkdir()
    version_file.write_text('__version__ = "1.4.0a0"\n')

    build_wheel.apply_version_override(tmp_path, parser.parse_args(cli).version_override)

    assert version_file.read_text() == f'__version__ = "{expected}"\n'
