# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU coverage of FA4 patch validation and installed wheel contents."""

import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only
ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def package_inputs(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("prepare_fa4", ROOT / "3rdparty/prepare_fa4.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    source = tmp_path / "source"
    cute = source / "flash_attn/cute"
    cute.mkdir(parents=True)
    (cute / "__init__.py").write_text(
        'from importlib.metadata import version\n__version__ = version("fa4")\n'
        "from .interface import _flash_attn_fwd\n"
    )
    interface = cute / "interface.py"
    original = (
        "from flash_attn.cute.utils import identity\ndef _flash_attn_fwd(): return identity(1)\n"
    )
    interface.write_text(original)
    (cute / "utils.py").write_text("def identity(value): return value\n")
    (cute / "pyproject.toml").write_text('[project]\nname = "flash-attn-4"\n')
    for name in ("LICENSE", "AUTHORS"):
        (source / name).write_text(f"Upstream {name}\n")
    monkeypatch.setattr(module, "SOURCE_SHA256", module.source_digest(source))
    interface.write_text(original + "_flash_attn_fwd.visual_gen_tuning_api = 1\n")
    monkeypatch.setattr(module, "PATCHED_SHA256", module.source_digest(source))
    interface.write_text(original)
    thirdparty = tmp_path / "3rdparty"
    thirdparty.mkdir()
    patch = thirdparty / "test.patch"
    patch.write_text(
        "--- a/flash_attn/cute/interface.py\n+++ b/flash_attn/cute/interface.py\n"
        "@@ -1,2 +1,3 @@\n from flash_attn.cute.utils import identity\n"
        " def _flash_attn_fwd(): return identity(1)\n"
        "+_flash_attn_fwd.visual_gen_tuning_api = 1\n"
    )
    monkeypatch.setattr(module, "PATCH_SHA256", hashlib.sha256(patch.read_bytes()).hexdigest())
    (thirdparty / "fetch_content.json").write_text(
        json.dumps(
            {
                "dependencies": [
                    {
                        "name": "flash_attn_4",
                        "git_tag": module.VALIDATED_REVISION,
                        "patch_file": "test.patch",
                    }
                ]
            }
        )
    )
    return module, source, thirdparty


@pytest.mark.parametrize("mismatch", ["revision", "patch", "source", "unpatched"])
def test_rejects_unvalidated_inputs(package_inputs, tmp_path, mismatch):
    module, source, thirdparty = package_inputs
    destination = tmp_path / "output"
    if mismatch == "revision":
        manifest = thirdparty / "fetch_content.json"
        manifest.write_text(manifest.read_text().replace(module.VALIDATED_REVISION, "new-revision"))
    elif mismatch == "patch":
        with (thirdparty / "test.patch").open("a") as patch:
            patch.write("# changed patch\n")
    elif mismatch == "source":
        (source / "flash_attn/cute/utils.py").write_text("# different upstream source\n")
    with pytest.raises(RuntimeError, match="FA4"):
        module.prepare(
            source, thirdparty, apply_patch=mismatch != "unpatched", destination=destination
        )
    assert not destination.exists()


def test_incremental_staging_revalidates_and_removes_stale_files(package_inputs, tmp_path):
    module, source, thirdparty = package_inputs
    destination = tmp_path / "output"
    module.prepare(source, thirdparty, apply_patch=True, destination=destination)
    first = {p.name: p.read_bytes() for p in destination.iterdir()}
    (destination / "stale.py").touch()
    module.prepare(source, thirdparty, apply_patch=True, destination=destination)
    assert {p.name: p.read_bytes() for p in destination.iterdir()} == first
    (source / "flash_attn/cute/utils.py").write_text("# stale populated source tree\n")
    with pytest.raises(RuntimeError, match="source/patch mismatch"):
        module.prepare(source, thirdparty, destination=destination)


def test_real_setup_builds_and_installs_owned_fa4_package(package_inputs, tmp_path):
    """Use the real setup.py with tiny package fixtures; no native/GPU build."""
    module, source, thirdparty = package_inputs
    project = tmp_path / "project"
    project.mkdir()
    for name in ("setup.py", "constraints.txt", "LICENSE", "README.md"):
        shutil.copy2(ROOT / name, project / name)
    for path in [*ROOT.glob("requirements*.txt"), *ROOT.glob("ATTRIBUTIONS-CPP-*.md")]:
        shutil.copy2(path, project / path.name)
    for package in ("tensorrt_llm", "tensorrt_llm/bindings", "3rdparty/fmha_sm100"):
        path = project / package
        path.mkdir(parents=True, exist_ok=True)
        (path / "__init__.py").touch()
    (project / "tensorrt_llm/version.py").write_text('__version__ = "0.0.0"\n')
    launch = project / "tensorrt_llm/llmapi/trtllm-llmapi-launch"
    launch.parent.mkdir()
    launch.write_text("#!/bin/sh\n")
    module.prepare(
        source, thirdparty, apply_patch=True, destination=project / "3rdparty/trtllm_flash_attn"
    )
    result = subprocess.run(
        [sys.executable, "setup.py", "bdist_wheel"], cwd=project, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
    (wheel,) = (project / "dist").glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        for name in (
            "interface.py",
            "utils.py",
            "__init__.py",
            "_build_info.py",
            "LICENSE",
            "AUTHORS",
        ):
            assert f"trtllm_flash_attn/{name}" in archive.namelist()
        assert not any(name.startswith("flash_attn/") for name in archive.namelist())
        metadata = next(name for name in archive.namelist() if name.endswith(".dist-info/METADATA"))
        assert "Requires-Dist: flash-attn-4" not in archive.read(metadata).decode()
    installed = tmp_path / "installed"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--no-deps",
            "--target",
            str(installed),
            str(wheel),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    # A stock distribution alongside the wheel must not supply any of our imports.
    stock = installed / "flash_attn/cute"
    stock.mkdir(parents=True)
    (stock / "__init__.py").write_text('raise RuntimeError("stock FA4 must not be imported")\n')
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            f"""
import sys
sys.path.insert(0, {str(installed)!r})
from trtllm_flash_attn import __version__, _flash_attn_fwd
from trtllm_flash_attn._build_info import BUILD_ID
assert __version__ == {module.VERSION!r}
assert BUILD_ID[1] == {module.VALIDATED_REVISION!r}
assert _flash_attn_fwd.visual_gen_tuning_api == 1
assert _flash_attn_fwd() == 1
assert not any(name.startswith('flash_attn.') for name in sys.modules)
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
