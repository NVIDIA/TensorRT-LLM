# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU coverage of FA4 patch validation and installed wheel contents."""

import hashlib
import importlib.util
import json
import os
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
        'from importlib.metadata import version\n__version__ = version("flash-attn-4")\n'
        "from .interface import _flash_attn_fwd\n"
    )
    interface = cute / "interface.py"
    original = (
        "from flash_attn.cute.utils import identity\ndef _flash_attn_fwd(): return identity(1)\n"
    )
    interface.write_text(original)
    (cute / "utils.py").write_text("def identity(value): return value\n")
    (cute / "pyproject.toml").write_text(
        '[build-system]\nrequires = ["setuptools>=75", "setuptools-scm>=8"]\n'
        'build-backend = "setuptools.build_meta"\n'
        '[project]\nname = "flash-attn-4"\ndynamic = ["version"]\n'
        '[tool.setuptools]\npackages = ["flash_attn.cute"]\n'
        'package-dir = {"flash_attn.cute" = "."}\n'
        '[tool.setuptools_scm]\nroot = "../.."\nfallback_version = "0.0.0"\n'
    )
    (cute / "README.md").write_text("FA4 fixture\n")
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
    (tmp_path / "requirements-fa4.txt").write_text(f"flash-attn-4=={module.WHEEL_VERSION}\n")
    (tmp_path / "requirements.txt").write_text(f"flash-attn-4=={module.VERSION}\n")
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
    first = {
        p.relative_to(destination): p.read_bytes() for p in destination.rglob("*") if p.is_file()
    }
    (destination / "stale.py").touch()
    module.prepare(source, thirdparty, apply_patch=True, destination=destination)
    assert {
        p.relative_to(destination): p.read_bytes() for p in destination.rglob("*") if p.is_file()
    } == first
    (source / "flash_attn/cute/utils.py").write_text("# stale populated source tree\n")
    with pytest.raises(RuntimeError, match="source/patch mismatch"):
        module.prepare(source, thirdparty, destination=destination)


def test_real_setup_resolves_separate_patched_fa4_wheel(package_inputs, tmp_path):
    """Exercise both real wheel builders and pip's transitive dependency resolution."""
    module, source, thirdparty = package_inputs
    fa4_project = tmp_path / "fa4-project"
    module.prepare(source, thirdparty, apply_patch=True, destination=fa4_project)
    wheelhouse = tmp_path / "wheels"
    fa4_wheel = module.build_wheel(fa4_project, thirdparty, wheelhouse)
    with zipfile.ZipFile(fa4_wheel) as archive:
        assert "flash_attn/cute/interface.py" in archive.namelist()
        assert "flash_attn/cute/_trtllm_build_info.py" in archive.namelist()
        assert "flash_attn/__init__.py" not in archive.namelist()
        for name in ("LICENSE", "AUTHORS"):
            assert any(p.endswith(f"/{name}") for p in archive.namelist())

    project = tmp_path / "project"
    project.mkdir()
    for name in ("setup.py", "LICENSE", "README.md", "requirements-fa4.txt"):
        shutil.copy2(ROOT / name, project / name)
    # Isolate native code and unrelated dependencies while keeping the real
    # setup.py dependency override, package discovery and wheel metadata.
    for path in [*ROOT.glob("requirements*.txt"), ROOT / "constraints.txt"]:
        if path.name != "requirements-fa4.txt":
            (project / path.name).write_text("")
    (project / "requirements.txt").write_text(f"flash-attn-4=={module.VERSION}\n")
    for path in ROOT.glob("ATTRIBUTIONS-CPP-*.md"):
        shutil.copy2(path, project / path.name)
    for package in ("tensorrt_llm", "tensorrt_llm/bindings", "3rdparty/fmha_sm100"):
        path = project / package
        path.mkdir(parents=True, exist_ok=True)
        (path / "__init__.py").touch()
    (project / "tensorrt_llm/version.py").write_text('__version__ = "0.0.0"\n')
    launch = project / "tensorrt_llm/llmapi/trtllm-llmapi-launch"
    launch.parent.mkdir()
    launch.write_text("#!/bin/sh\n")
    result = subprocess.run(
        [sys.executable, "setup.py", "bdist_wheel", "--dist-dir", str(wheelhouse)],
        cwd=project,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    (trtllm_wheel,) = wheelhouse.glob("tensorrt_llm-*.whl")
    with zipfile.ZipFile(trtllm_wheel) as archive:
        assert not any(
            n.startswith(("flash_attn/", "trtllm_flash_attn/")) for n in archive.namelist()
        )
        metadata = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        assert (
            f"Requires-Dist: flash-attn-4=={module.WHEEL_VERSION}"
            in archive.read(metadata).decode()
        )

    installed = tmp_path / "installed"
    # Deliberately allow dependency resolution: installing just TRT-LLM must
    # locate the companion wheel in the same wheelhouse without any index.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--ignore-installed",
            "--find-links",
            str(wheelhouse),
            "--target",
            str(installed),
            str(trtllm_wheel),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            f"""
import sys
sys.path.insert(0, {str(installed)!r})
from importlib.metadata import distribution, version
from flash_attn.cute import _flash_attn_fwd
from flash_attn.cute._trtllm_build_info import BUILD_ID
assert version('flash-attn-4') == {module.WHEEL_VERSION!r}
assert BUILD_ID[1] == {module.VALIDATED_REVISION!r}
assert _flash_attn_fwd.visual_gen_tuning_api == 1
assert _flash_attn_fwd() == 1
assert not any(name.startswith('trtllm_flash_attn') for name in sys.modules)
assert 'flash_attn/cute/interface.py' in {{str(p) for p in distribution('flash-attn-4').files}}
assert not any(str(p).startswith('flash_attn/') for p in distribution('tensorrt_llm').files)
""",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    # The TRT-LLM dependency must not resolve against a stock-version b19 wheel.
    from packaging.requirements import Requirement

    assert module.VERSION not in Requirement(f"flash-attn-4=={module.WHEEL_VERSION}").specifier
    assert module.WHEEL_VERSION in Requirement(f"flash-attn-4=={module.VERSION}").specifier
    fa4_wheel.unlink()
    stock_env = os.environ.copy()
    stock_env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_FLASH_ATTN_4"] = module.VERSION
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheelhouse),
            str(fa4_project / "flash_attn/cute"),
        ],
        env=stock_env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (wheelhouse / f"flash_attn_4-{module.VERSION}-py3-none-any.whl").is_file()
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--ignore-installed",
            "--find-links",
            str(wheelhouse),
            "--target",
            str(tmp_path / "missing"),
            str(trtllm_wheel),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert f"flash-attn-4=={module.WHEEL_VERSION}" in result.stderr


@pytest.mark.parametrize("requirements", ["requirements.txt", "requirements-fa4.txt"])
def test_version_bump_requires_revalidation(package_inputs, tmp_path, requirements):
    module, source, thirdparty = package_inputs
    project = tmp_path / "project"
    module.prepare(source, thirdparty, apply_patch=True, destination=project)
    (thirdparty.parent / requirements).write_text("flash-attn-4==4.0.0b20\n")
    with pytest.raises(RuntimeError, match="pin changed"):
        module.build_wheel(project, thirdparty, tmp_path / "wheels")
