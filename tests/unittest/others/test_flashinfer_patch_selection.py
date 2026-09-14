# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""How the FlashInfer patch step picks a patch directory.

The hunks are written against one FlashInfer release, so
``flashinfer-patches/`` keeps a directory per version and the apply script
chooses it from the ``flashinfer-python`` pin in ``requirements.txt``. These
tests cover that choice and the fail-closed path: a pin with no directory must
stop the build rather than apply another release's hunks or none at all.

CPU only, no GPU and no FlashInfer install; the header trees are fakes.
"""

import importlib.util
import pathlib
import shutil
import subprocess

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "flashinfer-patches" / "apply-flashinfer-patches.py"


def load_script():
    """Import the apply script, whose file name is not a module name."""
    spec = importlib.util.spec_from_file_location("apply_flashinfer_patches", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def apply_script():
    assert SCRIPT.is_file(), f"{SCRIPT} is missing"
    return load_script()


def write_requirements(tmp_path, line):
    path = tmp_path / "requirements.txt"
    path.write_text(f"numpy\n{line}\ntorch==2.9.0\n")
    return path


def test_pinned_version_reads_the_flashinfer_line(apply_script, tmp_path):
    path = write_requirements(tmp_path, "flashinfer-python==1.2.3")
    assert apply_script.pinned_version(path) == "1.2.3"


def test_pinned_version_rejects_a_range(apply_script, tmp_path):
    path = write_requirements(tmp_path, "flashinfer-python>=1.2.3")
    with pytest.raises(SystemExit, match="not a single '=='"):
        apply_script.pinned_version(path)


def test_pinned_version_needs_the_package(apply_script, tmp_path):
    path = tmp_path / "requirements.txt"
    path.write_text("torch==2.9.0\n")
    with pytest.raises(SystemExit, match="no flashinfer-python requirement"):
        apply_script.pinned_version(path)


def test_pinned_version_needs_the_file(apply_script, tmp_path):
    with pytest.raises(SystemExit, match="does not exist"):
        apply_script.pinned_version(tmp_path / "absent.txt")


def test_patch_dir_picks_the_pinned_version(apply_script, tmp_path):
    (tmp_path / "1.2.3").mkdir()
    (tmp_path / "9.9.9").mkdir()
    assert apply_script.patch_dir("1.2.3", tmp_path) == tmp_path / "1.2.3"


def test_patch_dir_fails_closed_on_an_unknown_pin(apply_script, tmp_path):
    (tmp_path / "1.2.3").mkdir()
    with pytest.raises(SystemExit) as exc:
        apply_script.patch_dir("4.5.6", tmp_path)
    message = str(exc.value)
    assert "4.5.6" in message and "1.2.3" in message


def test_patches_are_ordered_and_required(apply_script, tmp_path):
    version = tmp_path / "1.2.3"
    version.mkdir()
    with pytest.raises(SystemExit, match="no patch files"):
        apply_script.patches(version)
    for name in ("0002-second.patch", "0001-first.patch", "notes.md"):
        (version / name).write_text("")
    assert [p.name for p in apply_script.patches(version)] == [
        "0001-first.patch",
        "0002-second.patch",
    ]


def test_every_shipped_version_directory_has_patches(apply_script):
    """The checked-in tree must carry patches for the version it pins."""
    version = apply_script.pinned_version()
    directory = apply_script.patch_dir(version)
    assert [p.name for p in apply_script.patches(directory)]


@pytest.mark.skipif(shutil.which("patch") is None, reason="the patch command is not installed")
def test_selection_applies_the_right_version(apply_script, tmp_path):
    """End to end over fake header trees: the pin decides what lands.

    Two versions of a header differ, and each version directory holds the patch
    derived against its own copy. Applying the directory the pin names works;
    the other version's patch does not even dry-run.
    """
    patches_root = tmp_path / "flashinfer-patches"
    headers = {}
    for version, stock in (("1.2.3", "int old_shape;\n"), ("4.5.6", "int new_shape;\n")):
        tree = tmp_path / "headers" / version
        (tree / "flashinfer").mkdir(parents=True)
        (tree / "flashinfer" / "h.cuh").write_text(f"// header\n{stock}// end\n")
        headers[version] = tree

        patched = tmp_path / "patched" / version
        (patched / "flashinfer").mkdir(parents=True)
        (patched / "flashinfer" / "h.cuh").write_text(f"// header\n{stock}// MARKER\n// end\n")

        directory = patches_root / version
        directory.mkdir(parents=True)
        diff = subprocess.run(
            [
                "diff",
                "-u",
                "--label",
                "a/flashinfer/h.cuh",
                "--label",
                "b/flashinfer/h.cuh",
                str(tree / "flashinfer" / "h.cuh"),
                str(patched / "flashinfer" / "h.cuh"),
            ],
            capture_output=True,
            text=True,
        )
        (directory / "0001-marker.patch").write_text(diff.stdout)

    requirements = write_requirements(tmp_path, "flashinfer-python==4.5.6")
    version = apply_script.pinned_version(requirements)
    directory = apply_script.patch_dir(version, patches_root)
    patch = apply_script.patches(directory)[0]
    root = headers[version]

    assert not apply_script.applied(patch, root)
    assert apply_script.run_patch(patch, root, revert=False) == 0
    assert apply_script.applied(patch, root)
    assert "MARKER" in (root / "flashinfer" / "h.cuh").read_text()

    # The other version's patch is derived from different text and does not apply.
    other = apply_script.patches(apply_script.patch_dir("1.2.3", patches_root))[0]
    assert (
        subprocess.run(
            ["patch", "-p1", "--dry-run", "--batch", "--fuzz=0", "-i", str(other)],
            cwd=headers["4.5.6"],
            capture_output=True,
        ).returncode
        != 0
    )
