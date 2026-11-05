# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The isolated FA4 setup preserves package layout and verifies its source pins."""

import difflib
import hashlib
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.cpu_only
@pytest.mark.parametrize("namespace", [True, False], ids=["namespace", "regular"])
def test_fa4_setup_supports_package_layouts(tmp_path: Path, namespace: bool) -> None:
    """Run the real setup flow without importing CUDA-dependent FA4 implementation code."""
    package = tmp_path / "site" / "flash_attn"
    cute = package / "cute"
    cute.mkdir(parents=True)
    if not namespace:
        (package / "__init__.py").write_text("")
    (cute / "__init__.py").write_text('raise RuntimeError("setup must not import FA4")\n')
    base = "rescale_threshold = 4.0\n"
    patched = "rescale_threshold = 0.0\n"
    forward = cute / "flash_fwd_sm100.py"
    forward.write_text(base)
    relative = "flash_attn/cute/flash_fwd_sm100.py"
    patch_text = "".join(
        difflib.unified_diff(
            base.splitlines(keepends=True),
            patched.splitlines(keepends=True),
            fromfile=f"a/{relative}",
            tofile=f"b/{relative}",
        )
    )
    repository = tmp_path / "repository"
    patch_file = repository / "3rdparty/patches/fa4_fp8_rescale0.patch"
    patch_file.parent.mkdir(parents=True)
    patch_file.write_text(patch_text)
    source = (
        Path(__file__).resolve().parents[6] / "examples/minimax_m3/rubin/setup_fa4.sh"
    ).read_text()
    # Tiny fixture files exercise the real hash, copy and patch flow without a CUDA package installation.
    for name, contents in (
        ("BASE_SHA256", base),
        ("PATCHED_SHA256", patched),
        ("PATCH_SHA256", patch_text),
    ):
        digest = hashlib.sha256(contents.encode()).hexdigest()
        source, count = re.subn(
            rf'^readonly {name}="[^"]+"$', f'readonly {name}="{digest}"', source, flags=re.MULTILINE
        )
        assert count == 1
    setup = repository / "examples/minimax_m3/rubin/setup_fa4.sh"
    setup.parent.mkdir(parents=True)
    setup.write_text(source)
    binaries = tmp_path / "bin"
    binaries.mkdir()
    interpreter = binaries / "python3"
    # An installed regular flash_attn package otherwise overrides the namespace fixture.
    interpreter.write_text(f'#!/usr/bin/env bash\nexec {shlex.quote(sys.executable)} -S "$@"\n')
    interpreter.chmod(0o755)
    env = {
        **os.environ,
        "PATH": str(binaries) + os.pathsep + os.environ["PATH"],
        "PYTHONPATH": str(package.parent),
        "PYTHONPYCACHEPREFIX": str(tmp_path / "pycache"),
    }
    destination = tmp_path / "patched"
    completed = subprocess.run(
        ["bash", str(setup), str(destination)], env=env, capture_output=True, text=True, timeout=30
    )
    assert completed.returncode == 0, completed.stderr
    assert (destination / relative).read_text() == patched
    assert (destination / "flash_attn/__init__.py").exists() is (not namespace)
    assert forward.read_text() == base
    forward.write_text("unexpected source\n")
    rejected = tmp_path / "rejected"
    completed = subprocess.run(
        ["bash", str(setup), str(rejected)], env=env, capture_output=True, text=True, timeout=30
    )
    assert completed.returncode != 0
    assert "Unexpected flash_attn/cute/flash_fwd_sm100.py SHA256" in completed.stderr
    assert not rejected.exists()
