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
"""Hermetic lifecycle tests for the generic source-vendoring tool."""

from __future__ import annotations

import datetime
import hashlib
import importlib.util
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

pytestmark = pytest.mark.cpu_only


_REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_VENDOR_SOURCES = _REPO_ROOT / "scripts" / "vendor_sources.py"
_VENDOR_NAME = "example"
_SOURCE = "python/example"
_DESTINATION = "src/example"
_INCLUDE = "**/*.py"
_FUTURE_EXPIRY = (datetime.date.today() + datetime.timedelta(days=14)).isoformat()


def _command_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C.UTF-8",
        }
    )
    return env


def _run(
    command: list[str | Path],
    *,
    cwd: Path,
    check: bool = True,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = _command_env()
    if env_overrides is not None:
        env.update(env_overrides)
    result = subprocess.run(
        [str(argument) for argument in command],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if check and result.returncode != 0:
        pytest.fail(
            f"Command failed ({result.returncode}): {' '.join(map(str, command))}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    return result


def _git(repo: Path, *arguments: str) -> str:
    result = _run(["git", "-C", repo, *arguments], cwd=repo)
    return result.stdout.strip()


def _write_files(root: Path, files: dict[str, str]) -> None:
    for relative_path, content in files.items():
        path = root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")


def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


def _make_upstream(tmp_path: Path, files: dict[str, str]) -> tuple[Path, str]:
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    _run(["git", "init", "--initial-branch=main", upstream], cwd=tmp_path)
    _git(upstream, "config", "user.name", "Vendor Sources Test")
    _git(upstream, "config", "user.email", "vendor-sources@example.invalid")
    _write_files(upstream / _SOURCE, files)
    return upstream, _commit(upstream, "initial upstream source")


def _make_consumer(tmp_path: Path) -> tuple[Path, Path]:
    consumer = tmp_path / "consumer"
    lock = consumer / "3rdparty" / "vendor-sources.yml"
    lock.parent.mkdir(parents=True)
    return consumer, lock


def _vendor(
    consumer: Path,
    lock: Path,
    *arguments: str | Path,
    check: bool = True,
    env_overrides: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return _run(
        [sys.executable, _VENDOR_SOURCES, "--lock", lock, *arguments],
        cwd=consumer,
        check=check,
        env_overrides=env_overrides,
    )


def _copy_python_sources(upstream: Path, consumer: Path) -> None:
    source = upstream / _SOURCE
    destination = consumer / _DESTINATION
    for source_path in source.rglob("*.py"):
        relative_path = source_path.relative_to(source)
        target = destination / relative_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target)


def _create_vendor(
    consumer: Path,
    lock: Path,
    upstream: Path,
    commit: str,
    *,
    mode: str,
    url: str | None = None,
) -> None:
    _vendor(
        consumer,
        lock,
        "create",
        _VENDOR_NAME,
        "--url",
        url or upstream.as_uri(),
        "--commit",
        commit,
        "--source",
        _SOURCE,
        "--destination",
        _DESTINATION,
        "--include",
        _INCLUDE,
        "--repo",
        upstream,
        "--adopt",
        mode,
    )


def _lock_data(lock: Path) -> dict[str, object]:
    data = yaml.safe_load(lock.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    return data


def _vendor_data(lock: Path) -> dict[str, object]:
    vendors = _lock_data(lock)["vendors"]
    assert isinstance(vendors, dict)
    vendor = vendors[_VENDOR_NAME]
    assert isinstance(vendor, dict)
    return vendor


def _assert_failure(result: subprocess.CompletedProcess[str], text: str) -> None:
    assert result.returncode != 0
    combined_output = f"{result.stdout}\n{result.stderr}".lower()
    assert text.lower() in combined_output


def _tree_snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    if not root.exists():
        return {}
    return {
        path.relative_to(root).as_posix(): (
            path.read_bytes(),
            stat.S_IMODE(path.stat().st_mode),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _set_patch_digest(lock: Path, content: bytes) -> None:
    data = _lock_data(lock)
    vendors = data["vendors"]
    assert isinstance(vendors, dict)
    vendor = vendors[_VENDOR_NAME]
    assert isinstance(vendor, dict)
    vendor["patch_digest"] = f"sha256:{hashlib.sha256(content).hexdigest()}"
    lock.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def _load_vendor_sources_module() -> ModuleType:
    module_name = "_vendor_sources_under_test"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(module_name, _VENDOR_SOURCES)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_create_check_digest_and_sync(tmp_path: Path) -> None:
    upstream, commit = _make_upstream(
        tmp_path,
        {
            "__init__.py": "VALUE = 1\n",
            "nested/kernel.py": "def kernel() -> int:\n    return 1\n",
            "README.md": "not selected\n",
        },
    )
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)

    _create_vendor(consumer, lock, upstream, commit, mode="exact")

    vendor = _vendor_data(lock)
    assert vendor["commit"] == commit
    assert str(vendor["digest"]).startswith("sha256-tree-v1:")
    assert not (consumer / _DESTINATION / "README.md").exists()
    _vendor(consumer, lock, "check", _VENDOR_NAME)

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    sentinel = tmp_path / "git-was-invoked"
    fake_git = fake_bin / "git"
    fake_git.write_text(
        '#!/bin/sh\nprintf invoked > "$VENDOR_GIT_SENTINEL"\nexit 97\n',
        encoding="utf-8",
    )
    fake_git.chmod(0o755)
    fake_git_environment = {
        "PATH": str(fake_bin),
        "VENDOR_GIT_SENTINEL": str(sentinel),
    }
    for mode in ([], ["--offline"]):
        _vendor(
            consumer,
            lock,
            "check",
            _VENDOR_NAME,
            *mode,
            env_overrides=fake_git_environment,
        )
        assert not sentinel.exists(), "offline checks must not invoke Git"
    invalid_mixed_mode = _vendor(
        consumer,
        lock,
        "check",
        _VENDOR_NAME,
        "--offline",
        "--repo",
        upstream,
        check=False,
        env_overrides=fake_git_environment,
    )
    _assert_failure(invalid_mixed_mode, "either")
    assert not sentinel.exists(), "rejected offline checks must not invoke Git"

    kernel = consumer / _DESTINATION / "nested" / "kernel.py"
    kernel.write_text("def kernel() -> int:\n    return 99\n", encoding="utf-8")
    _assert_failure(_vendor(consumer, lock, "check", _VENDOR_NAME, check=False), "digest")

    _vendor(consumer, lock, "sync", _VENDOR_NAME, "--repo", upstream)
    assert kernel.read_text(encoding="utf-8").endswith("return 1\n")
    _vendor(consumer, lock, "check", _VENDOR_NAME)


def test_create_materializes_exact_tree_and_remote_mismatch_is_fatal(tmp_path: Path) -> None:
    upstream, first_commit = _make_upstream(
        tmp_path,
        {
            "kernel.py": "VALUE = 1\n",
            "nested/helper.py": "HELPER = True\n",
            "README.md": "not selected\n",
        },
    )
    consumer, lock = _make_consumer(tmp_path)

    _vendor(
        consumer,
        lock,
        "create",
        _VENDOR_NAME,
        "--url",
        upstream.as_uri(),
        "--branch",
        "main",
        "--commit",
        first_commit,
        "--source",
        _SOURCE,
        "--destination",
        _DESTINATION,
        "--include",
        _INCLUDE,
        "--repo",
        upstream,
    )

    destination = consumer / _DESTINATION
    assert (destination / "kernel.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert (destination / "nested" / "helper.py").is_file()
    assert not (destination / "README.md").exists()
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--upstream")

    (upstream / _SOURCE / "kernel.py").write_text("VALUE = 2\n", encoding="utf-8")
    second_commit = _commit(upstream, "change accessible upstream source")
    original_lock = lock.read_text(encoding="utf-8")
    assert first_commit in original_lock
    lock.write_text(original_lock.replace(first_commit, second_commit), encoding="utf-8")

    # Offline integrity still holds because only source provenance changed.
    _vendor(consumer, lock, "check", _VENDOR_NAME)
    remote_check = _vendor(
        consumer,
        lock,
        "check",
        _VENDOR_NAME,
        "--upstream",
        check=False,
    )
    _assert_failure(remote_check, "materialization digest mismatch")
    assert "skip-remote-unavailable" not in f"{remote_check.stdout}\n{remote_check.stderr}".lower()


def test_git_subprocesses_ignore_ambient_repository_environment(tmp_path: Path) -> None:
    upstream, commit = _make_upstream(tmp_path, {"kernel.py": "VALUE = 'upstream'\n"})
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    _create_vendor(consumer, lock, upstream, commit, mode="exact")
    destination = consumer / _DESTINATION
    (destination / "kernel.py").write_text("VALUE = 'downstream'\n", encoding="utf-8")

    upstream_head = _git(upstream, "rev-parse", "HEAD")
    upstream_branch = _git(upstream, "symbolic-ref", "HEAD")
    upstream_status = _git(upstream, "status", "--porcelain=v1", "--untracked-files=all")
    upstream_index = (upstream / ".git" / "index").read_bytes()
    upstream_source = _tree_snapshot(upstream / _SOURCE)

    _vendor(
        consumer,
        lock,
        "patch",
        _VENDOR_NAME,
        "create",
        "--repo",
        upstream,
        env_overrides={
            "GIT_DIR": str(upstream / ".git"),
            "GIT_WORK_TREE": str(upstream),
            "GIT_INDEX_FILE": str(upstream / ".git" / "index"),
            "GIT_OBJECT_DIRECTORY": str(upstream / ".git" / "objects"),
        },
    )

    assert _git(upstream, "rev-parse", "HEAD") == upstream_head
    assert _git(upstream, "symbolic-ref", "HEAD") == upstream_branch
    assert (upstream / ".git" / "index").read_bytes() == upstream_index
    assert _git(upstream, "status", "--porcelain=v1", "--untracked-files=all") == upstream_status
    assert _tree_snapshot(upstream / _SOURCE) == upstream_source
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)


def test_exact_materialization_ignores_git_export_attributes(tmp_path: Path) -> None:
    marker = 'REVISION = "$Format:%H$"\n'
    upstream, _ = _make_upstream(
        tmp_path,
        {
            ".gitattributes": "ignored.py export-ignore\nsubstituted.py export-subst\n",
            "ignored.py": "IGNORED_BY_ARCHIVE = True\n",
            "substituted.py": marker,
        },
    )
    (upstream / _SOURCE / "ignored.py").chmod(0o755)
    commit = _commit(upstream, "record executable mode")
    consumer, lock = _make_consumer(tmp_path)

    _vendor(
        consumer,
        lock,
        "create",
        _VENDOR_NAME,
        "--url",
        upstream.as_uri(),
        "--commit",
        commit,
        "--source",
        _SOURCE,
        "--destination",
        _DESTINATION,
        "--include",
        _INCLUDE,
        "--repo",
        upstream,
    )

    destination = consumer / _DESTINATION
    assert (destination / "ignored.py").read_text(encoding="utf-8") == (
        "IGNORED_BY_ARCHIVE = True\n"
    )
    assert (destination / "ignored.py").stat().st_mode & stat.S_IXUSR
    assert (destination / "substituted.py").read_text(encoding="utf-8") == marker
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--upstream", "--require-access")


def test_patched_adoption_preserves_raw_crlf_with_selected_text_attribute(
    tmp_path: Path,
) -> None:
    upstream, _ = _make_upstream(tmp_path, {"value.txt": "seed\n"})
    _git(upstream, "config", "core.autocrlf", "false")
    upstream_value = upstream / _SOURCE / "value.txt"
    upstream_value.write_bytes(b"upstream\r\n")
    _commit(upstream, "record raw CRLF blob")

    attributes = upstream / _SOURCE / ".gitattributes"
    attributes.write_text("*.txt text\n", encoding="utf-8")
    _git(upstream, "add", "--", f"{_SOURCE}/.gitattributes")
    _git(upstream, "commit", "-m", "select text attribute")
    commit = _git(upstream, "rev-parse", "HEAD")
    committed_blob = _git(upstream, "rev-parse", f"{commit}:{_SOURCE}/value.txt")
    raw_worktree_blob = _git(upstream, "hash-object", "--no-filters", f"{_SOURCE}/value.txt")
    assert committed_blob == raw_worktree_blob

    consumer, lock = _make_consumer(tmp_path)
    destination = consumer / _DESTINATION
    shutil.copytree(upstream / _SOURCE, destination)
    destination_value = destination / "value.txt"
    destination_value.write_bytes(b"downstream\r\n")
    destination_value.chmod(0o755)
    accepted_tree = _tree_snapshot(destination)
    assert accepted_tree["value.txt"] == (b"downstream\r\n", 0o755)

    _vendor(
        consumer,
        lock,
        "create",
        _VENDOR_NAME,
        "--url",
        upstream.as_uri(),
        "--commit",
        commit,
        "--source",
        _SOURCE,
        "--destination",
        _DESTINATION,
        "--repo",
        upstream,
        "--adopt",
        "patched",
    )
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)

    destination_value.write_bytes(b"broken\n")
    destination_value.chmod(0o644)
    _vendor(consumer, lock, "sync", _VENDOR_NAME, "--repo", upstream)
    assert _tree_snapshot(destination) == accepted_tree
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)


def test_local_replacement_refs_do_not_change_locked_content(tmp_path: Path) -> None:
    upstream, first_commit = _make_upstream(tmp_path, {"kernel.py": "VALUE = 1\n"})
    (upstream / _SOURCE / "kernel.py").write_text("VALUE = 2\n", encoding="utf-8")
    second_commit = _commit(upstream, "replacement content")
    _git(upstream, "replace", first_commit, second_commit)
    consumer, lock = _make_consumer(tmp_path)

    _vendor(
        consumer,
        lock,
        "create",
        _VENDOR_NAME,
        "--url",
        upstream.as_uri(),
        "--commit",
        first_commit,
        "--source",
        _SOURCE,
        "--destination",
        _DESTINATION,
        "--include",
        _INCLUDE,
        "--repo",
        upstream,
    )

    assert (consumer / _DESTINATION / "kernel.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert _vendor_data(lock)["commit"] == first_commit
    _git(upstream, "replace", "-d", first_commit)
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)


def test_lock_rejects_duplicate_keys_and_unsafe_paths(tmp_path: Path) -> None:
    upstream, commit = _make_upstream(tmp_path, {"kernel.py": "VALUE = 1\n"})
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    _create_vendor(consumer, lock, upstream, commit, mode="exact")

    valid_lock = lock.read_text(encoding="utf-8")
    first_data_line = next(
        line
        for line in valid_lock.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    lock.write_text(f"{first_data_line}\n{valid_lock}", encoding="utf-8")
    _assert_failure(_vendor(consumer, lock, "check", check=False), "duplicate")

    lock.write_text(valid_lock, encoding="utf-8")
    vendor_marker = f"  {_VENDOR_NAME}:\n"
    assert vendor_marker in valid_lock
    lock.write_text(
        valid_lock.replace(vendor_marker, f"{vendor_marker}    unexpected: true\n", 1),
        encoding="utf-8",
    )
    _assert_failure(_vendor(consumer, lock, "check", check=False), "unexpected")

    lock.write_text(valid_lock, encoding="utf-8")
    unsafe_create = _vendor(
        consumer,
        lock,
        "create",
        "unsafe",
        "--url",
        upstream.as_uri(),
        "--commit",
        commit,
        "--source",
        _SOURCE,
        "--destination",
        "../outside",
        "--repo",
        upstream,
        check=False,
    )
    _assert_failure(unsafe_create, "destination")
    assert not (tmp_path / "outside").exists()

    unsafe_source = _vendor(
        consumer,
        lock,
        "create",
        "unsafe-source",
        "--url",
        upstream.as_uri(),
        "--commit",
        commit,
        "--source",
        "../outside",
        "--destination",
        "src/unsafe-source",
        "--repo",
        upstream,
        check=False,
    )
    _assert_failure(unsafe_source, "source")

    existing_destination = consumer / "src" / "existing"
    _write_files(existing_destination, {"kernel.py": "DO_NOT_REPLACE = True\n"})
    accidental_overwrite = _vendor(
        consumer,
        lock,
        "create",
        "existing",
        "--url",
        upstream.as_uri(),
        "--commit",
        commit,
        "--source",
        _SOURCE,
        "--destination",
        "src/existing",
        "--include",
        _INCLUDE,
        "--repo",
        upstream,
        check=False,
    )
    _assert_failure(accidental_overwrite, "already exists")
    assert (existing_destination / "kernel.py").read_text(encoding="utf-8") == (
        "DO_NOT_REPLACE = True\n"
    )

    git_destination = _vendor(
        consumer,
        lock,
        "create",
        "git-internals",
        "--url",
        upstream.as_uri(),
        "--commit",
        commit,
        "--source",
        _SOURCE,
        "--destination",
        ".git/vendor",
        "--repo",
        upstream,
        check=False,
    )
    _assert_failure(git_destination, "destination")

    unsafe_tree = consumer / "src" / "unsafe-tree"
    _write_files(
        unsafe_tree,
        {
            "kernel.py": "VALUE = 1\n",
            ".git/config": "[core]\n\thooksPath = /tmp/unsafe\n",
        },
    )
    unsafe_tree_create = _vendor(
        consumer,
        lock,
        "create",
        "unsafe-tree",
        "--url",
        upstream.as_uri(),
        "--commit",
        commit,
        "--source",
        _SOURCE,
        "--destination",
        "src/unsafe-tree",
        "--repo",
        upstream,
        "--adopt",
        "patched",
        check=False,
    )
    _assert_failure(unsafe_tree_create, "unsafe path")

    outside_destination = tmp_path / "outside-destination"
    _write_files(outside_destination, {"kernel.py": "VALUE = 1\n"})
    linked_destination = consumer / "src" / "linked-destination"
    linked_destination.symlink_to(outside_destination, target_is_directory=True)
    symlink_escape = _vendor(
        consumer,
        lock,
        "create",
        "unsafe-symlink",
        "--url",
        upstream.as_uri(),
        "--commit",
        commit,
        "--source",
        _SOURCE,
        "--destination",
        "src/linked-destination",
        "--include",
        _INCLUDE,
        "--repo",
        upstream,
        "--adopt",
        "exact",
        check=False,
    )
    _assert_failure(symlink_escape, "destination")
    assert (outside_destination / "kernel.py").read_text(encoding="utf-8") == "VALUE = 1\n"


def test_branch_tag_validation_and_overlapping_destinations(tmp_path: Path) -> None:
    upstream, commit = _make_upstream(tmp_path, {"kernel.py": "VALUE = 1\n"})
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    _create_vendor(consumer, lock, upstream, commit, mode="exact")

    valid_lock = lock.read_text(encoding="utf-8")
    url_line = next(
        line for line in valid_lock.splitlines(keepends=True) if line.startswith("    url:")
    )
    lock.write_text(
        valid_lock.replace(url_line, f"{url_line}    branch: main\n    tag: v1.0\n", 1),
        encoding="utf-8",
    )
    _assert_failure(_vendor(consumer, lock, "check", check=False), "both branch and tag")
    lock.write_text(valid_lock, encoding="utf-8")

    common_arguments: list[str | Path] = [
        "create",
        "second",
        "--url",
        upstream.as_uri(),
        "--commit",
        commit,
        "--source",
        _SOURCE,
        "--destination",
        "src/second",
        "--repo",
        upstream,
    ]
    conflicting_reference = _vendor(
        consumer,
        lock,
        *common_arguments,
        "--branch",
        "main",
        "--tag",
        "v1.0",
        check=False,
    )
    _assert_failure(conflicting_reference, "not allowed with argument")

    long_reference = _vendor(
        consumer,
        lock,
        *common_arguments,
        "--branch",
        "refs/heads/main",
        check=False,
    )
    _assert_failure(long_reference, "short Git name")

    overlapping = _vendor(
        consumer,
        lock,
        "create",
        "overlap",
        "--url",
        upstream.as_uri(),
        "--commit",
        commit,
        "--source",
        _SOURCE,
        "--destination",
        f"{_DESTINATION}/nested",
        "--repo",
        upstream,
        check=False,
    )
    _assert_failure(overlapping, "overlap")
    assert set(_lock_data(lock)["vendors"]) == {_VENDOR_NAME}


def test_patched_vendor_reproduces_added_modified_and_deleted_files(tmp_path: Path) -> None:
    upstream, commit = _make_upstream(
        tmp_path,
        {
            "keep.py": "KEEP = True\n",
            "modify.py": "VALUE = 'upstream'\n",
            "delete.py": "DELETE_ME = True\n",
        },
    )
    consumer, lock = _make_consumer(tmp_path)
    destination = consumer / _DESTINATION
    _write_files(
        destination,
        {
            "keep.py": "KEEP = True\n",
            "modify.py": "VALUE = 'downstream'\n",
            "add.py": "ADDED = True\n",
        },
    )
    (destination / "modify.py").chmod(0o755)

    _create_vendor(consumer, lock, upstream, commit, mode="patched")

    vendor = _vendor_data(lock)
    patch_path = consumer / str(vendor["patch"])
    assert patch_path.is_file()
    original_patch = patch_path.read_bytes()
    original_patch_mode = patch_path.stat().st_mode & 0o777
    patch_text = original_patch.decode("utf-8")
    assert all(filename in patch_text for filename in ("add.py", "modify.py", "delete.py"))
    assert "new mode 100755" in patch_text

    valid_lock = lock.read_text(encoding="utf-8")
    lock.write_text(
        valid_lock.replace(str(vendor["patch"]), "../outside.patch"),
        encoding="utf-8",
    )
    _assert_failure(_vendor(consumer, lock, "check", _VENDOR_NAME, check=False), "patch")
    lock.write_text(valid_lock, encoding="utf-8")

    external_patch = tmp_path / "external.patch"
    external_patch.write_bytes(original_patch)
    patch_path.unlink()
    patch_path.symlink_to(external_patch)
    _assert_failure(_vendor(consumer, lock, "check", _VENDOR_NAME, check=False), "patch")
    patch_path.unlink()
    patch_path.write_bytes(original_patch)
    patch_path.chmod(original_patch_mode)

    assert b"a/modify.py" in original_patch
    malicious_patch = original_patch.replace(b"a/modify.py", b"a/../outside.py").replace(
        b"b/modify.py", b"b/../outside.py"
    )
    patch_path.write_bytes(malicious_patch)
    _set_patch_digest(lock, malicious_patch)
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--offline")
    malicious_check = _vendor(
        consumer,
        lock,
        "check",
        _VENDOR_NAME,
        "--repo",
        upstream,
        check=False,
        env_overrides={"TMPDIR": str(tmp_path)},
    )
    _assert_failure(malicious_check, "failed to apply vendor patch")
    assert not (tmp_path / "outside.py").exists()

    outside_include_patch = b"""diff --git a/outside.txt b/outside.txt
new file mode 100644
--- /dev/null
+++ b/outside.txt
@@ -0,0 +1 @@
+outside include
"""
    patch_path.write_bytes(outside_include_patch)
    _set_patch_digest(lock, outside_include_patch)
    outside_include_check = _vendor(
        consumer,
        lock,
        "check",
        _VENDOR_NAME,
        "--repo",
        upstream,
        check=False,
    )
    _assert_failure(outside_include_check, "outside the vendor include set")

    patch_path.write_bytes(original_patch)
    patch_path.chmod(original_patch_mode)
    lock.write_text(valid_lock, encoding="utf-8")

    (destination / "add.py").unlink()
    (destination / "modify.py").write_text("BROKEN = True\n", encoding="utf-8")
    (destination / "modify.py").chmod(0o644)
    (destination / "delete.py").write_text("STALE = True\n", encoding="utf-8")
    _vendor(consumer, lock, "sync", _VENDOR_NAME, "--repo", upstream)

    assert (destination / "keep.py").read_text(encoding="utf-8") == "KEEP = True\n"
    assert (destination / "modify.py").read_text(encoding="utf-8") == "VALUE = 'downstream'\n"
    assert destination.joinpath("modify.py").stat().st_mode & 0o100
    assert (destination / "add.py").read_text(encoding="utf-8") == "ADDED = True\n"
    assert not (destination / "delete.py").exists()
    _vendor(consumer, lock, "check", _VENDOR_NAME)

    # A refresh is also the recovery path if a prior update was interrupted
    # after replacing the patch but before updating its recorded digest.
    patch_path.write_bytes(original_patch + b"\n")
    (destination / "add.py").write_text("ADDED = 'refreshed'\n", encoding="utf-8")
    _vendor(consumer, lock, "patch", _VENDOR_NAME, "refresh", "--repo", upstream)
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)
    accepted_tree = _tree_snapshot(destination)

    (destination / "add.py").unlink()
    (destination / "modify.py").write_text("BROKEN_AGAIN = True\n", encoding="utf-8")
    (destination / "modify.py").chmod(0o644)
    (destination / "delete.py").write_text("STALE_AGAIN = True\n", encoding="utf-8")
    _vendor(consumer, lock, "sync", _VENDOR_NAME, "--repo", upstream)
    assert _tree_snapshot(destination) == accepted_tree
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)


def test_divergence_capture_refresh_expiry_and_clear(tmp_path: Path) -> None:
    upstream, commit = _make_upstream(tmp_path, {"kernel.py": "VALUE = 'upstream'\n"})
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    _create_vendor(consumer, lock, upstream, commit, mode="exact")
    kernel = consumer / _DESTINATION / "kernel.py"

    kernel.write_text("VALUE = 'urgent-fix'\n", encoding="utf-8")
    excessive_expiry = (datetime.date.today() + datetime.timedelta(days=31)).isoformat()
    excessive_divergence = _vendor(
        consumer,
        lock,
        "divergence",
        _VENDOR_NAME,
        "capture",
        "--reason",
        "unbounded exception",
        "--expires",
        excessive_expiry,
        "--repo",
        upstream,
        check=False,
    )
    _assert_failure(excessive_divergence, "30 days")
    _vendor(
        consumer,
        lock,
        "divergence",
        _VENDOR_NAME,
        "capture",
        "--reason",
        "urgent correctness fix",
        "--expires",
        _FUTURE_EXPIRY,
        "--repo",
        upstream,
    )
    divergence = _vendor_data(lock)["divergence"]
    assert isinstance(divergence, dict)
    assert divergence["reason"] == "urgent correctness fix"
    assert divergence["files"] == ["kernel.py"]
    assert str(divergence["digest"]).startswith("sha256-tree-v1:")
    _vendor(consumer, lock, "check", _VENDOR_NAME)

    valid_divergence_lock = lock.read_text(encoding="utf-8")
    destination_snapshot = _tree_snapshot(consumer / _DESTINATION)
    future_created = datetime.date.today() + datetime.timedelta(days=365)
    future_data = _lock_data(lock)
    future_vendor = future_data["vendors"][_VENDOR_NAME]
    assert isinstance(future_vendor, dict)
    future_divergence = future_vendor["divergence"]
    assert isinstance(future_divergence, dict)
    future_divergence["created"] = future_created.isoformat()
    future_divergence["expires"] = (future_created + datetime.timedelta(days=30)).isoformat()
    lock.write_text(yaml.safe_dump(future_data, sort_keys=False), encoding="utf-8")
    future_check = _vendor(
        consumer,
        lock,
        "check",
        _VENDOR_NAME,
        "--offline",
        check=False,
    )
    _assert_failure(future_check, "creation date cannot be in the future")
    assert _tree_snapshot(consumer / _DESTINATION) == destination_snapshot
    lock.write_text(valid_divergence_lock, encoding="utf-8")

    kernel.write_text("VALUE = 'urgent-fix-v2'\n", encoding="utf-8")
    _assert_failure(_vendor(consumer, lock, "check", _VENDOR_NAME, check=False), "digest")
    _vendor(consumer, lock, "divergence", _VENDOR_NAME, "refresh", "--repo", upstream)
    _vendor(consumer, lock, "check", _VENDOR_NAME)

    current_lock = lock.read_text(encoding="utf-8")
    assert _FUTURE_EXPIRY in current_lock
    lock.write_text(current_lock.replace(_FUTURE_EXPIRY, "2000-01-01"), encoding="utf-8")
    _assert_failure(_vendor(consumer, lock, "check", _VENDOR_NAME, check=False), "expired")
    lock.write_text(current_lock, encoding="utf-8")

    _vendor(consumer, lock, "divergence", _VENDOR_NAME, "clear")
    assert "divergence" not in _vendor_data(lock)
    _assert_failure(_vendor(consumer, lock, "check", _VENDOR_NAME, check=False), "digest")


def test_exact_adoption_rejects_unrepresented_differences(tmp_path: Path) -> None:
    upstream, commit = _make_upstream(tmp_path, {"kernel.py": "VALUE = 'upstream'\n"})
    consumer, lock = _make_consumer(tmp_path)
    destination = consumer / _DESTINATION
    _write_files(destination, {"kernel.py": "VALUE = 'different'\n"})
    destination_snapshot = _tree_snapshot(destination)

    adoption = _vendor(
        consumer,
        lock,
        "create",
        _VENDOR_NAME,
        "--url",
        upstream.as_uri(),
        "--commit",
        commit,
        "--source",
        _SOURCE,
        "--destination",
        _DESTINATION,
        "--include",
        _INCLUDE,
        "--repo",
        upstream,
        "--adopt",
        "exact",
        check=False,
    )

    _assert_failure(adoption, "is not exact")
    assert "kernel.py" in f"{adoption.stdout}\n{adoption.stderr}"
    assert _tree_snapshot(destination) == destination_snapshot
    assert not lock.exists()
    assert not (consumer / "3rdparty" / "vendor_patches" / "example.patch").exists()


def test_create_rolls_back_destination_and_patch_when_lock_save_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    upstream, commit = _make_upstream(tmp_path, {"kernel.py": "VALUE = 'upstream'\n"})
    exact_root = tmp_path / "exact-case"
    exact_root.mkdir()
    exact_consumer, exact_lock = _make_consumer(exact_root)
    patched_root = tmp_path / "patched-case"
    patched_root.mkdir()
    patched_consumer, patched_lock = _make_consumer(patched_root)
    patched_destination = patched_consumer / _DESTINATION
    _write_files(patched_destination, {"kernel.py": "VALUE = 'downstream'\n"})
    patched_snapshot = _tree_snapshot(patched_destination)
    module = _load_vendor_sources_module()

    def fail_save_lock(_: object) -> None:
        raise OSError("injected lock-save failure")

    with monkeypatch.context() as failure:
        failure.setattr(module, "_save_lock", fail_save_lock)
        exact_result = module.main(
            [
                "--lock",
                str(exact_lock),
                "create",
                _VENDOR_NAME,
                "--url",
                upstream.as_uri(),
                "--commit",
                commit,
                "--source",
                _SOURCE,
                "--destination",
                _DESTINATION,
                "--include",
                _INCLUDE,
                "--repo",
                str(upstream),
            ]
        )
        patched_result = module.main(
            [
                "--lock",
                str(patched_lock),
                "create",
                _VENDOR_NAME,
                "--url",
                upstream.as_uri(),
                "--commit",
                commit,
                "--source",
                _SOURCE,
                "--destination",
                _DESTINATION,
                "--include",
                _INCLUDE,
                "--repo",
                str(upstream),
                "--adopt",
                "patched",
            ]
        )

    assert exact_result == 1
    assert patched_result == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err.lower()
    assert "injected lock-save failure" in captured.err
    assert not exact_lock.exists()
    assert not (exact_consumer / _DESTINATION).exists()
    assert not list((exact_consumer / "src").glob(".example.vendor-*"))
    assert not patched_lock.exists()
    assert _tree_snapshot(patched_destination) == patched_snapshot
    assert not (patched_consumer / "3rdparty/vendor_patches/example.patch").exists()
    assert not list(patched_destination.parent.glob(".example.vendor-*"))

    _vendor(
        exact_consumer,
        exact_lock,
        "create",
        _VENDOR_NAME,
        "--url",
        upstream.as_uri(),
        "--commit",
        commit,
        "--source",
        _SOURCE,
        "--destination",
        _DESTINATION,
        "--include",
        _INCLUDE,
        "--repo",
        upstream,
    )
    _create_vendor(patched_consumer, patched_lock, upstream, commit, mode="patched")


def test_sync_rolls_back_divergence_when_lock_save_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    upstream, commit = _make_upstream(tmp_path, {"kernel.py": "VALUE = 'upstream'\n"})
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    _create_vendor(consumer, lock, upstream, commit, mode="exact")
    destination = consumer / _DESTINATION
    (destination / "kernel.py").write_text("VALUE = 'divergent'\n", encoding="utf-8")
    (destination / "kernel.py").chmod(0o755)
    _vendor(
        consumer,
        lock,
        "divergence",
        _VENDOR_NAME,
        "capture",
        "--reason",
        "test atomic synchronization",
        "--expires",
        _FUTURE_EXPIRY,
        "--repo",
        upstream,
    )
    lock_snapshot = lock.read_bytes()
    destination_snapshot = _tree_snapshot(destination)
    parent_entries = sorted(path.name for path in destination.parent.iterdir())
    module = _load_vendor_sources_module()

    def fail_save_lock(_: object) -> None:
        raise OSError("injected lock-save failure")

    with monkeypatch.context() as failure:
        failure.setattr(module, "_save_lock", fail_save_lock)
        result = module.main(
            [
                "--lock",
                str(lock),
                "sync",
                _VENDOR_NAME,
                "--repo",
                str(upstream),
            ]
        )

    assert result == 1
    captured = capsys.readouterr()
    assert "error:" in captured.err.lower()
    assert "injected lock-save failure" in captured.err
    assert lock.read_bytes() == lock_snapshot
    assert _tree_snapshot(destination) == destination_snapshot
    assert sorted(path.name for path in destination.parent.iterdir()) == parent_entries
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--offline")

    _vendor(consumer, lock, "sync", _VENDOR_NAME, "--repo", upstream)
    assert (destination / "kernel.py").read_text(encoding="utf-8") == "VALUE = 'upstream'\n"
    assert "divergence" not in _vendor_data(lock)


def test_stale_backup_does_not_leak_staging_or_modify_trees(tmp_path: Path) -> None:
    upstream, commit = _make_upstream(tmp_path, {"kernel.py": "VALUE = 'upstream'\n"})
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    _create_vendor(consumer, lock, upstream, commit, mode="exact")
    destination = consumer / _DESTINATION
    backup = destination.parent / ".example.vendor-backup"
    _write_files(backup, {"sentinel.py": "DO_NOT_TOUCH = True\n"})
    destination_snapshot = _tree_snapshot(destination)
    backup_snapshot = _tree_snapshot(backup)
    lock_snapshot = lock.read_bytes()
    parent_entries = sorted(path.name for path in destination.parent.iterdir())

    sync = _vendor(
        consumer,
        lock,
        "sync",
        _VENDOR_NAME,
        "--repo",
        upstream,
        check=False,
    )

    _assert_failure(sync, "stale vendor backup")
    assert sorted(path.name for path in destination.parent.iterdir()) == parent_entries
    assert _tree_snapshot(destination) == destination_snapshot
    assert _tree_snapshot(backup) == backup_snapshot
    assert lock.read_bytes() == lock_snapshot


def test_patch_drop_and_remove_preserve_destination_and_unrelated_patch(tmp_path: Path) -> None:
    upstream, commit = _make_upstream(tmp_path, {"kernel.py": "VALUE = 'upstream'\n"})
    consumer, lock = _make_consumer(tmp_path)
    destination = consumer / _DESTINATION
    _write_files(destination, {"kernel.py": "VALUE = 'downstream'\n"})
    _create_vendor(consumer, lock, upstream, commit, mode="patched")
    patch_path = consumer / str(_vendor_data(lock)["patch"])
    rejected_lock = lock.read_bytes()
    rejected_patch = patch_path.read_bytes()
    rejected_destination = _tree_snapshot(destination)

    rejected_drop = _vendor(
        consumer,
        lock,
        "patch",
        _VENDOR_NAME,
        "drop",
        "--repo",
        upstream,
        check=False,
    )
    _assert_failure(rejected_drop, "not exact upstream")
    assert lock.read_bytes() == rejected_lock
    assert patch_path.read_bytes() == rejected_patch
    assert _tree_snapshot(destination) == rejected_destination

    (destination / "kernel.py").write_text("VALUE = 'upstream'\n", encoding="utf-8")
    exact_snapshot = _tree_snapshot(destination)
    unrelated_patch = patch_path.parent / "unrelated.patch"
    unrelated_patch.write_bytes(b"unrelated sentinel\n")
    _vendor(consumer, lock, "patch", _VENDOR_NAME, "drop", "--repo", upstream)
    vendor = _vendor_data(lock)
    assert "patch" not in vendor
    assert "patch_digest" not in vendor
    assert not patch_path.exists()
    assert unrelated_patch.read_bytes() == b"unrelated sentinel\n"
    assert _tree_snapshot(destination) == exact_snapshot
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)

    (destination / "kernel.py").write_text("VALUE = 'downstream-again'\n", encoding="utf-8")
    _vendor(consumer, lock, "patch", _VENDOR_NAME, "create", "--repo", upstream)
    recreated_patch = consumer / str(_vendor_data(lock)["patch"])
    removal_snapshot = _tree_snapshot(destination)
    _vendor(consumer, lock, "remove", _VENDOR_NAME)
    assert _lock_data(lock)["vendors"] == {}
    assert not recreated_patch.exists()
    assert unrelated_patch.read_bytes() == b"unrelated sentinel\n"
    assert _tree_snapshot(destination) == removal_snapshot


def test_export_rejects_dirty_source_and_mismatched_head_without_changes(tmp_path: Path) -> None:
    upstream, commit = _make_upstream(tmp_path, {"kernel.py": "VALUE = 'upstream'\n"})
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    _create_vendor(consumer, lock, upstream, commit, mode="exact")
    destination = consumer / _DESTINATION
    (destination / "kernel.py").write_text("VALUE = 'downstream'\n", encoding="utf-8")
    _vendor(
        consumer,
        lock,
        "divergence",
        _VENDOR_NAME,
        "capture",
        "--reason",
        "test export guards",
        "--expires",
        _FUTURE_EXPIRY,
        "--repo",
        upstream,
    )
    lock_snapshot = lock.read_bytes()
    destination_snapshot = _tree_snapshot(destination)

    upstream_kernel = upstream / _SOURCE / "kernel.py"
    upstream_kernel.write_text("VALUE = 'dirty'\n", encoding="utf-8")
    dirty_source_snapshot = _tree_snapshot(upstream / _SOURCE)
    dirty_export = _vendor(
        consumer,
        lock,
        "export",
        _VENDOR_NAME,
        "--repo",
        upstream,
        check=False,
    )
    _assert_failure(dirty_export, "uncommitted changes")
    assert _tree_snapshot(upstream / _SOURCE) == dirty_source_snapshot
    assert lock.read_bytes() == lock_snapshot
    assert _tree_snapshot(destination) == destination_snapshot

    _git(upstream, "restore", "--", _SOURCE)
    _write_files(upstream, {"unrelated.txt": "new head\n"})
    _commit(upstream, "unrelated upstream change")
    mismatched_source_snapshot = _tree_snapshot(upstream / _SOURCE)
    mismatched_export = _vendor(
        consumer,
        lock,
        "export",
        _VENDOR_NAME,
        "--repo",
        upstream,
        check=False,
    )
    _assert_failure(mismatched_export, "head must equal locked commit")
    assert _tree_snapshot(upstream / _SOURCE) == mismatched_source_snapshot
    assert lock.read_bytes() == lock_snapshot
    assert _tree_snapshot(destination) == destination_snapshot


def test_upstream_access_policy_export_and_pin(tmp_path: Path) -> None:
    upstream, first_commit = _make_upstream(tmp_path, {"kernel.py": "VALUE = 'upstream'\n"})
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    inaccessible_url = (tmp_path / "inaccessible-upstream.git").as_uri()
    _create_vendor(
        consumer,
        lock,
        upstream,
        first_commit,
        mode="exact",
        url=inaccessible_url,
    )

    best_effort = _vendor(consumer, lock, "check", _VENDOR_NAME, "--upstream")
    assert "unavailable" in f"{best_effort.stdout}\n{best_effort.stderr}".lower()
    strict = _vendor(
        consumer,
        lock,
        "check",
        _VENDOR_NAME,
        "--upstream",
        "--require-access",
        check=False,
    )
    _assert_failure(strict, "unavailable")
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)

    kernel = consumer / _DESTINATION / "kernel.py"
    kernel.write_text("VALUE = 'exported-fix'\n", encoding="utf-8")
    _vendor(
        consumer,
        lock,
        "divergence",
        _VENDOR_NAME,
        "capture",
        "--reason",
        "validate downstream fix upstream",
        "--expires",
        _FUTURE_EXPIRY,
        "--repo",
        upstream,
    )

    _git(upstream, "switch", "-c", "vendor-fix")
    _vendor(consumer, lock, "export", _VENDOR_NAME, "--repo", upstream)
    assert (upstream / _SOURCE / "kernel.py").read_text(
        encoding="utf-8"
    ) == "VALUE = 'exported-fix'\n"
    second_commit = _commit(upstream, "apply exported downstream fix")

    _vendor(
        consumer,
        lock,
        "pin",
        _VENDOR_NAME,
        "--url",
        upstream.as_uri(),
        "--branch",
        "vendor-fix",
        "--commit",
        second_commit,
        "--repo",
        upstream,
    )
    vendor = _vendor_data(lock)
    assert vendor["url"] == upstream.as_uri()
    assert vendor["branch"] == "vendor-fix"
    assert vendor["commit"] == second_commit
    assert "divergence" not in vendor
    assert "patch" not in vendor
    _vendor(consumer, lock, "check", _VENDOR_NAME)
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)
