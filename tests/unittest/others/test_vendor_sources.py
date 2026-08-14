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

import hashlib
import importlib.util
import os
import shutil
import stat
import subprocess
import sys
import threading
import time
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


def _prepare_absorbed_patch_migration(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path, str]:
    upstream, base_commit = _make_upstream(tmp_path, {"compat.py": "MODE = 'upstream'\n"})
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    destination = consumer / _DESTINATION
    (destination / "compat.py").write_text("MODE = 'trtllm'\n", encoding="utf-8")
    _create_vendor(consumer, lock, upstream, base_commit, mode="patched")
    patch_path = consumer / str(_vendor_data(lock)["patch"])
    (upstream / _SOURCE / "compat.py").write_text("MODE = 'trtllm'\n", encoding="utf-8")
    absorbed_commit = _commit(upstream, "absorb compatibility patch")
    return upstream, consumer, lock, destination, patch_path, absorbed_commit


def test_atomic_write_syncs_file_before_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_vendor_sources_module()
    target = tmp_path / "lock" / "vendor-sources.yml"
    target.parent.mkdir()
    events: list[str] = []
    real_fsync = os.fsync
    real_replace = os.replace

    def record_fsync(file_descriptor: int) -> None:
        mode = os.fstat(file_descriptor).st_mode
        events.append("directory-fsync" if stat.S_ISDIR(mode) else "file-fsync")
        real_fsync(file_descriptor)

    def record_replace(source: str | bytes | Path, destination: str | bytes | Path) -> None:
        events.append("replace")
        real_replace(source, destination)

    with monkeypatch.context() as recording:
        recording.setattr(module.os, "fsync", record_fsync)
        recording.setattr(module.os, "replace", record_replace)
        module._atomic_write(target, b"updated\n")

    assert target.read_bytes() == b"updated\n"
    assert events == ["file-fsync", "replace"]


def test_create_and_patch_updates_do_not_sync_replacement_directories(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream, commit = _make_upstream(tmp_path, {"kernel.py": "VALUE = 'upstream'\n"})
    exact_root = tmp_path / "exact-case"
    exact_root.mkdir()
    exact_consumer, exact_lock = _make_consumer(exact_root)
    patched_root = tmp_path / "patched-case"
    patched_root.mkdir()
    patched_consumer, patched_lock = _make_consumer(patched_root)
    patched_destination = patched_consumer / _DESTINATION
    _write_files(patched_destination, {"kernel.py": "VALUE = 'adopted'\n"})
    module = _load_vendor_sources_module()
    real_fsync = os.fsync
    directory_syncs = 0

    def reject_directory_sync(file_descriptor: int) -> None:
        nonlocal directory_syncs
        if stat.S_ISDIR(os.fstat(file_descriptor).st_mode):
            directory_syncs += 1
            raise OSError("unexpected replacement-directory sync")
        real_fsync(file_descriptor)

    def create_arguments(lock: Path, *, adopt: str | None = None) -> list[str]:
        arguments = [
            "--lock",
            str(lock),
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
        if adopt is not None:
            arguments.extend(["--adopt", adopt])
        return arguments

    with monkeypatch.context() as failure:
        failure.setattr(module.os, "fsync", reject_directory_sync)
        assert module.main(create_arguments(exact_lock)) == 0
        assert module.main(create_arguments(patched_lock, adopt="patched")) == 0

        exact_destination = exact_consumer / _DESTINATION
        (exact_destination / "kernel.py").write_text("VALUE = 'patched'\n", encoding="utf-8")
        assert (
            module.main(
                [
                    "--lock",
                    str(exact_lock),
                    "patch",
                    _VENDOR_NAME,
                    "create",
                    "--repo",
                    str(upstream),
                ]
            )
            == 0
        )
        (exact_destination / "kernel.py").write_text("VALUE = 'refreshed'\n", encoding="utf-8")
        assert (
            module.main(
                [
                    "--lock",
                    str(exact_lock),
                    "patch",
                    _VENDOR_NAME,
                    "refresh",
                    "--repo",
                    str(upstream),
                ]
            )
            == 0
        )

    assert directory_syncs == 0
    _vendor(exact_consumer, exact_lock, "check", _VENDOR_NAME, "--repo", upstream)
    _vendor(patched_consumer, patched_lock, "check", _VENDOR_NAME, "--repo", upstream)


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


@pytest.mark.parametrize("use_local_checkout", [True, False])
def test_create_rejects_annotated_tag_object_id(
    tmp_path: Path,
    use_local_checkout: bool,
) -> None:
    upstream, commit = _make_upstream(tmp_path, {"kernel.py": "VALUE = 1\n"})
    _git(upstream, "tag", "-a", "v1", "-m", "annotated release", commit)
    tag_object = _git(upstream, "rev-parse", "refs/tags/v1")
    assert tag_object != commit
    assert _git(upstream, "cat-file", "-t", tag_object) == "tag"
    consumer, lock = _make_consumer(tmp_path)
    arguments: list[str | Path] = [
        "create",
        _VENDOR_NAME,
        "--url",
        upstream.as_uri(),
        "--tag",
        "v1",
        "--commit",
        tag_object,
        "--source",
        _SOURCE,
        "--destination",
        _DESTINATION,
        "--include",
        _INCLUDE,
    ]
    if use_local_checkout:
        arguments.extend(["--repo", upstream])

    result = _vendor(consumer, lock, *arguments, check=False)

    _assert_failure(result, "commit")
    assert not lock.exists()
    assert not (consumer / _DESTINATION).exists()


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

    lock.write_text(
        valid_lock.replace(vendor_marker, f"{vendor_marker}    divergence: legacy\n", 1),
        encoding="utf-8",
    )
    _assert_failure(_vendor(consumer, lock, "check", check=False), "divergence")

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
    _assert_failure(
        _vendor(consumer, lock, "check", _VENDOR_NAME, "--offline", check=False),
        "digest mismatch",
    )
    status = _vendor(consumer, lock, "status", _VENDOR_NAME, check=False)
    _assert_failure(status, "fail-local-integrity")
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


def test_export_rejects_ignored_selected_source_files_without_changes(tmp_path: Path) -> None:
    upstream, commit = _make_upstream(tmp_path, {"kernel.py": "VALUE = 'upstream'\n"})
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    _create_vendor(consumer, lock, upstream, commit, mode="exact")
    destination = consumer / _DESTINATION
    (destination / "kernel.py").write_text("VALUE = 'downstream'\n", encoding="utf-8")
    ignored_path = upstream / _SOURCE / "generated.py"
    ignored_path.write_text("GENERATED = True\n", encoding="utf-8")
    (upstream / ".git" / "info" / "exclude").write_text(
        f"/{_SOURCE}/generated.py\n", encoding="utf-8"
    )
    assert _git(upstream, "status", "--porcelain", "--", _SOURCE) == ""
    upstream_snapshot = _tree_snapshot(upstream / _SOURCE)
    lock_snapshot = lock.read_bytes()
    destination_snapshot = _tree_snapshot(destination)

    result = _vendor(
        consumer,
        lock,
        "export",
        _VENDOR_NAME,
        "--repo",
        upstream,
        check=False,
    )

    _assert_failure(result, "ignored untracked files")
    assert _tree_snapshot(upstream / _SOURCE) == upstream_snapshot
    assert ignored_path.read_text(encoding="utf-8") == "GENERATED = True\n"
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
    no_change = _vendor(
        consumer,
        lock,
        "export",
        _VENDOR_NAME,
        "--repo",
        upstream,
        check=False,
    )
    _assert_failure(no_change, "no downstream change")

    kernel = consumer / _DESTINATION / "kernel.py"
    locked_state = lock.read_bytes()
    kernel.write_text("VALUE = 'exported-fix'\n", encoding="utf-8")
    pending_tree = _tree_snapshot(consumer / _DESTINATION)
    _assert_failure(
        _vendor(consumer, lock, "check", _VENDOR_NAME, "--offline", check=False),
        "digest mismatch",
    )

    _git(upstream, "switch", "-c", "vendor-fix")
    _vendor(consumer, lock, "export", _VENDOR_NAME, "--repo", upstream)
    assert lock.read_bytes() == locked_state
    assert _tree_snapshot(consumer / _DESTINATION) == pending_tree
    _assert_failure(
        _vendor(consumer, lock, "check", _VENDOR_NAME, "--offline", check=False),
        "digest mismatch",
    )
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
        "--repo",
        upstream,
    )
    vendor = _vendor_data(lock)
    assert vendor["url"] == upstream.as_uri()
    assert vendor["branch"] == "vendor-fix"
    assert vendor["commit"] == second_commit
    assert "patch" not in vendor
    assert _tree_snapshot(consumer / _DESTINATION) == pending_tree
    _vendor(consumer, lock, "check", _VENDOR_NAME)
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)


def test_partial_upstream_acceptance_uses_temporary_branches(tmp_path: Path) -> None:
    upstream, base_commit = _make_upstream(
        tmp_path,
        {
            "a.py": "A = 0\n",
            "b.py": "B = 0\n",
        },
    )
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    _create_vendor(consumer, lock, upstream, base_commit, mode="exact")
    destination = consumer / _DESTINATION
    _write_files(destination, {"a.py": "A = 1\n", "b.py": "B = 1\n"})
    accepted_destination = _tree_snapshot(destination)

    _git(upstream, "switch", "-c", "vendor-a-and-b")
    _vendor(consumer, lock, "export", _VENDOR_NAME, "--repo", upstream)
    temporary_ab_commit = _commit(upstream, "export A and B")
    _vendor(
        consumer,
        lock,
        "pin",
        _VENDOR_NAME,
        "--branch",
        "vendor-a-and-b",
        "--commit",
        temporary_ab_commit,
        "--repo",
        upstream,
    )
    assert _vendor_data(lock)["commit"] == temporary_ab_commit
    assert _tree_snapshot(destination) == accepted_destination

    _git(upstream, "switch", "main")
    _write_files(upstream / _SOURCE, {"a.py": "A = 1\n"})
    canonical_a_commit = _commit(upstream, "accept A upstream")
    lock_snapshot = lock.read_bytes()
    rejected = _vendor(
        consumer,
        lock,
        "pin",
        _VENDOR_NAME,
        "--branch",
        "main",
        "--commit",
        canonical_a_commit,
        "--repo",
        upstream,
        check=False,
    )
    _assert_failure(rejected, "b.py")
    assert lock.read_bytes() == lock_snapshot
    assert _tree_snapshot(destination) == accepted_destination

    _git(upstream, "switch", "-c", "vendor-b")
    _write_files(upstream / _SOURCE, {"b.py": "B = 1\n"})
    temporary_b_commit = _commit(upstream, "carry B after A")
    _vendor(
        consumer,
        lock,
        "pin",
        _VENDOR_NAME,
        "--branch",
        "vendor-b",
        "--commit",
        temporary_b_commit,
        "--repo",
        upstream,
    )
    assert _vendor_data(lock)["commit"] == temporary_b_commit
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)

    _git(upstream, "switch", "main")
    _write_files(upstream / _SOURCE, {"b.py": "B = 1\n"})
    canonical_ab_commit = _commit(upstream, "accept B upstream")
    _vendor(
        consumer,
        lock,
        "pin",
        _VENDOR_NAME,
        "--branch",
        "main",
        "--commit",
        canonical_ab_commit,
        "--repo",
        upstream,
    )
    vendor = _vendor_data(lock)
    assert vendor["branch"] == "main"
    assert vendor["commit"] == canonical_ab_commit
    assert _tree_snapshot(destination) == accepted_destination
    _vendor(consumer, lock, "check", _VENDOR_NAME)
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)


def test_export_and_pin_retain_then_drop_compatibility_patch(tmp_path: Path) -> None:
    upstream, base_commit = _make_upstream(
        tmp_path,
        {
            "compat.py": "MODE = 'upstream'\n",
            "feature.py": "FEATURE = 'base'\n",
        },
    )
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    destination = consumer / _DESTINATION
    (destination / "compat.py").write_text("MODE = 'trtllm'\n", encoding="utf-8")
    _create_vendor(consumer, lock, upstream, base_commit, mode="patched")
    original_vendor = _vendor_data(lock)
    patch_path = consumer / str(original_vendor["patch"])
    patch_content = patch_path.read_bytes()
    patch_mode = stat.S_IMODE(patch_path.stat().st_mode)
    patch_digest = original_vendor["patch_digest"]

    no_change = _vendor(
        consumer,
        lock,
        "export",
        _VENDOR_NAME,
        "--repo",
        upstream,
        check=False,
    )
    _assert_failure(no_change, "no downstream change")

    (destination / "feature.py").write_text("FEATURE = 'exported'\n", encoding="utf-8")
    pending_destination = _tree_snapshot(destination)
    lock_snapshot = lock.read_bytes()
    _git(upstream, "switch", "-c", "feature-fix")
    _vendor(consumer, lock, "export", _VENDOR_NAME, "--repo", upstream)
    assert (upstream / _SOURCE / "compat.py").read_text(encoding="utf-8") == "MODE = 'upstream'\n"
    assert (upstream / _SOURCE / "feature.py").read_text(
        encoding="utf-8"
    ) == "FEATURE = 'exported'\n"
    assert lock.read_bytes() == lock_snapshot
    assert _tree_snapshot(destination) == pending_destination
    assert patch_path.read_bytes() == patch_content
    feature_commit = _commit(upstream, "export feature change")

    _vendor(
        consumer,
        lock,
        "pin",
        _VENDOR_NAME,
        "--branch",
        "feature-fix",
        "--commit",
        feature_commit,
        "--repo",
        upstream,
    )
    retained_vendor = _vendor_data(lock)
    assert retained_vendor["patch"] == original_vendor["patch"]
    assert retained_vendor["patch_digest"] == patch_digest
    assert patch_path.read_bytes() == patch_content
    assert stat.S_IMODE(patch_path.stat().st_mode) == patch_mode
    assert _tree_snapshot(destination) == pending_destination
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)

    _git(upstream, "switch", "-c", "incomplete-pin")
    (upstream / _SOURCE / "feature.py").write_text("FEATURE = 'wrong'\n", encoding="utf-8")
    incomplete_commit = _commit(upstream, "candidate does not reproduce destination")
    retained_lock = lock.read_bytes()
    rejected = _vendor(
        consumer,
        lock,
        "pin",
        _VENDOR_NAME,
        "--branch",
        "incomplete-pin",
        "--commit",
        incomplete_commit,
        "--repo",
        upstream,
        check=False,
    )
    _assert_failure(rejected, "feature.py")
    assert lock.read_bytes() == retained_lock
    assert patch_path.read_bytes() == patch_content
    assert stat.S_IMODE(patch_path.stat().st_mode) == patch_mode
    assert _tree_snapshot(destination) == pending_destination

    _git(upstream, "switch", "feature-fix")
    (upstream / _SOURCE / "compat.py").write_text("MODE = 'trtllm'\n", encoding="utf-8")
    absorbed_commit = _commit(upstream, "absorb TensorRT-LLM compatibility change")
    _vendor(
        consumer,
        lock,
        "pin",
        _VENDOR_NAME,
        "--branch",
        "feature-fix",
        "--commit",
        absorbed_commit,
        "--repo",
        upstream,
    )
    exact_vendor = _vendor_data(lock)
    assert exact_vendor["commit"] == absorbed_commit
    assert "patch" not in exact_vendor
    assert "patch_digest" not in exact_vendor
    assert not patch_path.exists()
    assert _tree_snapshot(destination) == pending_destination
    _vendor(consumer, lock, "check", _VENDOR_NAME)
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)


def test_pin_restores_absorbed_patch_when_lock_save_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    upstream, base_commit = _make_upstream(tmp_path, {"compat.py": "MODE = 'upstream'\n"})
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    destination = consumer / _DESTINATION
    (destination / "compat.py").write_text("MODE = 'trtllm'\n", encoding="utf-8")
    _create_vendor(consumer, lock, upstream, base_commit, mode="patched")
    patch_path = consumer / str(_vendor_data(lock)["patch"])
    patch_content = patch_path.read_bytes()
    patch_mode = stat.S_IMODE(patch_path.stat().st_mode)
    lock_content = lock.read_bytes()
    destination_snapshot = _tree_snapshot(destination)

    _git(upstream, "switch", "-c", "absorb-compat")
    (upstream / _SOURCE / "compat.py").write_text("MODE = 'trtllm'\n", encoding="utf-8")
    absorbed_commit = _commit(upstream, "absorb compatibility change")
    module = _load_vendor_sources_module()

    def fail_save_lock(_: object) -> None:
        raise OSError("injected pin lock-save failure")

    with monkeypatch.context() as failure:
        failure.setattr(module, "_save_lock", fail_save_lock)
        result = module.main(
            [
                "--lock",
                str(lock),
                "pin",
                _VENDOR_NAME,
                "--branch",
                "absorb-compat",
                "--commit",
                absorbed_commit,
                "--repo",
                str(upstream),
            ]
        )

    assert result == 1
    assert "injected pin lock-save failure" in capsys.readouterr().err
    assert lock.read_bytes() == lock_content
    assert patch_path.read_bytes() == patch_content
    assert stat.S_IMODE(patch_path.stat().st_mode) == patch_mode
    assert _tree_snapshot(destination) == destination_snapshot

    _vendor(
        consumer,
        lock,
        "pin",
        _VENDOR_NAME,
        "--branch",
        "absorb-compat",
        "--commit",
        absorbed_commit,
        "--repo",
        upstream,
    )
    assert not patch_path.exists()
    assert "patch" not in _vendor_data(lock)
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)


def test_pin_directory_sync_failure_retains_absorbed_patch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    upstream, base_commit = _make_upstream(tmp_path, {"compat.py": "MODE = 'upstream'\n"})
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    destination = consumer / _DESTINATION
    (destination / "compat.py").write_text("MODE = 'trtllm'\n", encoding="utf-8")
    _create_vendor(consumer, lock, upstream, base_commit, mode="patched")
    patch_path = consumer / str(_vendor_data(lock)["patch"])
    patch_content = patch_path.read_bytes()

    _git(upstream, "switch", "-c", "absorb-compat")
    (upstream / _SOURCE / "compat.py").write_text("MODE = 'trtllm'\n", encoding="utf-8")
    absorbed_commit = _commit(upstream, "absorb compatibility change")
    module = _load_vendor_sources_module()
    real_fsync = os.fsync

    def fail_directory_sync(file_descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(file_descriptor).st_mode):
            raise OSError("injected lock-directory sync failure")
        real_fsync(file_descriptor)

    with monkeypatch.context() as failure:
        failure.setattr(module.os, "fsync", fail_directory_sync)
        result = module.main(
            [
                "--lock",
                str(lock),
                "pin",
                _VENDOR_NAME,
                "--branch",
                "absorb-compat",
                "--commit",
                absorbed_commit,
                "--repo",
                str(upstream),
            ]
        )

    assert result == 1
    assert "injected lock-directory sync failure" in capsys.readouterr().err
    assert patch_path.read_bytes() == patch_content
    updated_vendor = _vendor_data(lock)
    assert updated_vendor["commit"] == absorbed_commit
    assert "patch" not in updated_vendor
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--offline")


def test_pin_patch_cleanup_failure_succeeds_with_orphan_warning(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    upstream, base_commit = _make_upstream(tmp_path, {"compat.py": "MODE = 'upstream'\n"})
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    destination = consumer / _DESTINATION
    (destination / "compat.py").write_text("MODE = 'trtllm'\n", encoding="utf-8")
    _create_vendor(consumer, lock, upstream, base_commit, mode="patched")
    patch_path = consumer / str(_vendor_data(lock)["patch"])
    patch_content = patch_path.read_bytes()

    _git(upstream, "switch", "-c", "absorb-compat")
    (upstream / _SOURCE / "compat.py").write_text("MODE = 'trtllm'\n", encoding="utf-8")
    absorbed_commit = _commit(upstream, "absorb compatibility change")
    module = _load_vendor_sources_module()
    real_save_lock_checked = module._save_lock_checked
    real_sync_directory = module._sync_directory
    real_unlink = Path.unlink
    events: list[str] = []

    def record_lock_save(lock_file: object, description: str) -> None:
        real_save_lock_checked(lock_file, description)
        events.append("lock-save")

    def record_directory_sync(path: Path) -> None:
        real_sync_directory(path)
        events.append("directory-sync")

    def fail_patch_cleanup(path: Path, missing_ok: bool = False) -> None:
        if path == patch_path:
            events.append("patch-unlink")
            raise OSError("injected absorbed-patch cleanup failure")
        real_unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as failure:
        failure.setattr(module, "_save_lock_checked", record_lock_save)
        failure.setattr(module, "_sync_directory", record_directory_sync)
        failure.setattr(Path, "unlink", fail_patch_cleanup)
        result = module.main(
            [
                "--lock",
                str(lock),
                "pin",
                _VENDOR_NAME,
                "--branch",
                "absorb-compat",
                "--commit",
                absorbed_commit,
                "--repo",
                str(upstream),
            ]
        )

    captured = capsys.readouterr()
    assert result == 0
    assert "offline enforcement restored" in captured.out
    assert "warning:" in captured.err.lower()
    assert "lock is committed and valid" in captured.err
    assert "unreferenced orphan" in captured.err
    assert "delete that file manually" in captured.err.lower()
    assert str(patch_path) in captured.err
    assert events == ["lock-save", "directory-sync", "patch-unlink"]
    assert patch_path.read_bytes() == patch_content
    updated_vendor = _vendor_data(lock)
    assert updated_vendor["commit"] == absorbed_commit
    assert "patch" not in updated_vendor
    assert "patch_digest" not in updated_vendor
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--offline")


def test_pin_interruption_after_lock_save_leaves_valid_lock_and_orphan_patch(
    tmp_path: Path,
) -> None:
    upstream, base_commit = _make_upstream(tmp_path, {"compat.py": "MODE = 'upstream'\n"})
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    destination = consumer / _DESTINATION
    (destination / "compat.py").write_text("MODE = 'trtllm'\n", encoding="utf-8")
    _create_vendor(consumer, lock, upstream, base_commit, mode="patched")
    patch_path = consumer / str(_vendor_data(lock)["patch"])
    patch_content = patch_path.read_bytes()
    patch_mode = stat.S_IMODE(patch_path.stat().st_mode)

    _git(upstream, "switch", "-c", "absorb-compat")
    (upstream / _SOURCE / "compat.py").write_text("MODE = 'trtllm'\n", encoding="utf-8")
    absorbed_commit = _commit(upstream, "absorb compatibility change")
    lock_saved = tmp_path / "pin-lock-saved"
    interrupt_script = "\n".join(
        [
            "import importlib.util",
            "import pathlib",
            "import sys",
            "import time",
            "module_path, lock, upstream, commit, marker = sys.argv[1:]",
            "spec = importlib.util.spec_from_file_location('_vendor_sources_interrupted', module_path)",
            "assert spec is not None and spec.loader is not None",
            "module = importlib.util.module_from_spec(spec)",
            "sys.modules[spec.name] = module",
            "spec.loader.exec_module(module)",
            "save_lock_checked = module._save_lock_checked",
            "def save_then_pause(lock_file, description):",
            "    save_lock_checked(lock_file, description)",
            "    pathlib.Path(marker).write_text('saved', encoding='utf-8')",
            "    time.sleep(30)",
            "module._save_lock_checked = save_then_pause",
            "raise SystemExit(module.main([",
            "    '--lock', lock, 'pin', 'example', '--branch', 'absorb-compat',",
            "    '--commit', commit, '--repo', upstream,",
            "]))",
        ]
    )
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            interrupt_script,
            str(_VENDOR_SOURCES),
            str(lock),
            str(upstream),
            absorbed_commit,
            str(lock_saved),
        ],
        cwd=consumer,
        env=_command_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not lock_saved.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if not lock_saved.exists():
            stdout, stderr = process.communicate(timeout=5)
            pytest.fail(
                f"Pin did not reach lock-save barrier.\nstdout:\n{stdout}\nstderr:\n{stderr}"
            )
        process.kill()
        process.communicate(timeout=5)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=5)

    assert process.returncode is not None and process.returncode < 0
    interrupted_vendor = _vendor_data(lock)
    assert interrupted_vendor["commit"] == absorbed_commit
    assert "patch" not in interrupted_vendor
    assert "patch_digest" not in interrupted_vendor
    assert patch_path.read_bytes() == patch_content
    assert stat.S_IMODE(patch_path.stat().st_mode) == patch_mode
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--offline")
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)


def test_sync_migrates_exact_vendor_and_validates_revision_arguments(tmp_path: Path) -> None:
    upstream, base_commit = _make_upstream(
        tmp_path,
        {
            "kernel.py": "VALUE = 1\n",
            "obsolete.py": "OBSOLETE = True\n",
        },
    )
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
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
        base_commit,
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
    )
    destination = consumer / _DESTINATION
    assert _vendor_data(lock)["branch"] == "main"
    original_digest = _vendor_data(lock)["digest"]
    original_lock = lock.read_bytes()
    original_destination = _tree_snapshot(destination)
    replacement_url = (tmp_path / "replacement-upstream.git").as_uri()

    invalid_arguments = [
        (["--url", replacement_url], "--commit"),
        (["--branch", "main"], "--commit"),
        (["--tag", "v2"], "--commit"),
        (["--commit", "deadbeef"], "full lowercase 40-hex sha"),
        (
            ["--commit", base_commit, "--branch", "main", "--tag", "v2"],
            "not allowed with argument",
        ),
    ]
    for arguments, expected_error in invalid_arguments:
        result = _vendor(
            consumer,
            lock,
            "sync",
            _VENDOR_NAME,
            *arguments,
            "--repo",
            upstream,
            check=False,
        )
        _assert_failure(result, expected_error)
        assert lock.read_bytes() == original_lock
        assert _tree_snapshot(destination) == original_destination

    (upstream / _SOURCE / "obsolete.py").unlink()
    _write_files(
        upstream / _SOURCE,
        {
            "kernel.py": "VALUE = 2\n",
            "new_kernel.py": "NEW_VALUE = 2\n",
        },
    )
    updated_commit = _commit(upstream, "update exact vendor source")
    _git(upstream, "tag", "v2", updated_commit)

    _vendor(
        consumer,
        lock,
        "sync",
        _VENDOR_NAME,
        "--url",
        replacement_url,
        "--tag",
        "v2",
        "--commit",
        updated_commit,
        "--repo",
        upstream,
    )

    vendor = _vendor_data(lock)
    assert vendor["url"] == replacement_url
    assert vendor["tag"] == "v2"
    assert "branch" not in vendor
    assert vendor["commit"] == updated_commit
    assert vendor["digest"] != original_digest
    assert vendor["source"] == _SOURCE
    assert vendor["destination"] == _DESTINATION
    assert vendor["include"] == [_INCLUDE]
    assert "patch" not in vendor
    assert "patch_digest" not in vendor
    assert (destination / "kernel.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert (destination / "new_kernel.py").read_text(encoding="utf-8") == "NEW_VALUE = 2\n"
    assert not (destination / "obsolete.py").exists()
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--offline")
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)

    _vendor(
        consumer,
        lock,
        "sync",
        _VENDOR_NAME,
        "--branch",
        "main",
        "--commit",
        updated_commit,
        "--repo",
        upstream,
    )
    branch_vendor = _vendor_data(lock)
    assert branch_vendor["branch"] == "main"
    assert "tag" not in branch_vendor


def test_sync_migration_rejects_dirty_destination_without_changes(tmp_path: Path) -> None:
    upstream, base_commit = _make_upstream(tmp_path, {"kernel.py": "VALUE = 1\n"})
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    _create_vendor(consumer, lock, upstream, base_commit, mode="exact")
    destination = consumer / _DESTINATION

    (upstream / _SOURCE / "kernel.py").write_text("VALUE = 2\n", encoding="utf-8")
    updated_commit = _commit(upstream, "update upstream source")
    (destination / "kernel.py").write_text("VALUE = 'pending'\n", encoding="utf-8")
    dirty_destination = _tree_snapshot(destination)
    original_lock = lock.read_bytes()
    original_destination_entries = sorted(path.name for path in destination.parent.iterdir())

    result = _vendor(
        consumer,
        lock,
        "sync",
        _VENDOR_NAME,
        "--commit",
        updated_commit,
        "--repo",
        upstream,
        check=False,
    )

    _assert_failure(result, "destination")
    assert lock.read_bytes() == original_lock
    assert _tree_snapshot(destination) == dirty_destination
    assert not (consumer / "3rdparty/vendor_patches/example.patch").exists()
    assert (
        sorted(path.name for path in destination.parent.iterdir()) == original_destination_entries
    )


def test_sync_migration_rejects_destination_edit_made_during_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    upstream, base_commit = _make_upstream(tmp_path, {"kernel.py": "VALUE = 1\n"})
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    _create_vendor(consumer, lock, upstream, base_commit, mode="exact")
    destination = consumer / _DESTINATION
    (upstream / _SOURCE / "kernel.py").write_text("VALUE = 2\n", encoding="utf-8")
    updated_commit = _commit(upstream, "update upstream during migration")
    original_lock = lock.read_bytes()
    module = _load_vendor_sources_module()
    real_source_tree = module._source_tree

    def fetch_then_edit(vendor: object, repository: Path | None) -> object:
        tree = real_source_tree(vendor, repository)
        if vendor.commit == updated_commit:
            (destination / "kernel.py").write_text("VALUE = 'late edit'\n", encoding="utf-8")
        return tree

    with monkeypatch.context() as edit_during_fetch:
        edit_during_fetch.setattr(module, "_source_tree", fetch_then_edit)
        result = module.main(
            [
                "--lock",
                str(lock),
                "sync",
                _VENDOR_NAME,
                "--commit",
                updated_commit,
                "--repo",
                str(upstream),
            ]
        )

    assert result == 1
    assert "changed while synchronization was in progress" in capsys.readouterr().err
    assert lock.read_bytes() == original_lock
    assert (destination / "kernel.py").read_text(encoding="utf-8") == "VALUE = 'late edit'\n"
    assert not list(destination.parent.glob(".example.vendor-*"))


def test_sync_migration_preserves_late_unselected_edit_at_destination_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    upstream, base_commit = _make_upstream(tmp_path, {"kernel.py": "VALUE = 1\n"})
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    destination = consumer / _DESTINATION
    notes = destination / "notes.txt"
    notes.write_text("before migration\n", encoding="utf-8")
    _create_vendor(consumer, lock, upstream, base_commit, mode="exact")
    (upstream / _SOURCE / "kernel.py").write_text("VALUE = 2\n", encoding="utf-8")
    updated_commit = _commit(upstream, "update upstream before unselected edit")
    original_lock = lock.read_bytes()
    backup = destination.parent / ".example.vendor-backup"
    module = _load_vendor_sources_module()
    real_replace = os.replace

    def edit_immediately_before_rename(
        source: str | bytes | Path, target: str | bytes | Path
    ) -> None:
        if Path(source) == destination and Path(target) == backup:
            notes.write_text("late unselected edit\n", encoding="utf-8")
        real_replace(source, target)

    with monkeypatch.context() as late_edit:
        late_edit.setattr(module.os, "replace", edit_immediately_before_rename)
        result = module.main(
            [
                "--lock",
                str(lock),
                "sync",
                _VENDOR_NAME,
                "--commit",
                updated_commit,
                "--repo",
                str(upstream),
            ]
        )

    assert result == 1
    assert "changed while synchronization was in progress" in capsys.readouterr().err
    assert lock.read_bytes() == original_lock
    assert (destination / "kernel.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    assert notes.read_text(encoding="utf-8") == "late unselected edit\n"
    assert not backup.exists()
    assert not list(destination.parent.glob(".example.vendor-*"))


@pytest.mark.parametrize("interrupt_point", ("backup", "install"))
def test_sync_migration_restores_destination_after_setup_rename_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt_point: str,
) -> None:
    upstream, base_commit = _make_upstream(tmp_path, {"kernel.py": "VALUE = 1\n"})
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    _create_vendor(consumer, lock, upstream, base_commit, mode="exact")
    destination = consumer / _DESTINATION
    backup = destination.parent / ".example.vendor-backup"
    original_lock = lock.read_bytes()
    original_destination = _tree_snapshot(destination)
    (upstream / _SOURCE / "kernel.py").write_text("VALUE = 2\n", encoding="utf-8")
    updated_commit = _commit(upstream, f"update before {interrupt_point} interrupt")
    module = _load_vendor_sources_module()
    real_replace = os.replace
    interrupted = False

    def replace_then_interrupt(source: str | bytes | Path, target: str | bytes | Path) -> None:
        nonlocal interrupted
        source_path = Path(source)
        target_path = Path(target)
        real_replace(source, target)
        is_target = (
            interrupt_point == "backup" and source_path == destination and target_path == backup
        ) or (
            interrupt_point == "install"
            and source_path.parent == destination.parent
            and source_path.name.startswith(".example.vendor-")
            and target_path == destination
        )
        if is_target and not interrupted:
            interrupted = True
            raise KeyboardInterrupt(f"injected {interrupt_point} rename interruption")

    with monkeypatch.context() as interruption:
        interruption.setattr(module.os, "replace", replace_then_interrupt)
        with pytest.raises(KeyboardInterrupt, match=f"{interrupt_point} rename interruption"):
            module.main(
                [
                    "--lock",
                    str(lock),
                    "sync",
                    _VENDOR_NAME,
                    "--commit",
                    updated_commit,
                    "--repo",
                    str(upstream),
                ]
            )

    assert interrupted
    assert lock.read_bytes() == original_lock
    assert _tree_snapshot(destination) == original_destination
    assert not list(destination.parent.glob(".example.vendor-*"))


def test_atomic_file_replacement_restores_old_generation_after_setup_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_vendor_sources_module()
    patch_path = tmp_path / "compat.patch"
    patch_path.write_bytes(b"old patch\n")
    patch_path.chmod(0o600)
    real_atomic_write = module._atomic_write
    interrupted = False

    def write_then_interrupt(path: Path, content: bytes) -> None:
        nonlocal interrupted
        real_atomic_write(path, content)
        if path == patch_path and content == b"new patch\n" and not interrupted:
            interrupted = True
            raise KeyboardInterrupt("injected patch replacement interruption")

    with monkeypatch.context() as interruption:
        interruption.setattr(module, "_atomic_write", write_then_interrupt)
        with pytest.raises(KeyboardInterrupt, match="patch replacement interruption"):
            with module._atomic_file_replacement(patch_path, b"new patch\n"):
                pass

    assert interrupted
    assert patch_path.read_bytes() == b"old patch\n"
    assert stat.S_IMODE(patch_path.stat().st_mode) == 0o600


def test_reverse_patch_reconstruction_rejects_a_nonempty_noop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_vendor_sources_module()
    patch_path = tmp_path / "compat.patch"
    patch_path.write_bytes(b"nonempty patch sentinel\n")
    baseline = {"compat.py": module.FileEntry(b"MODE = 'trtllm'\n")}

    def successful_noop(*_: object, capture_bytes: bool = False, **__: object) -> bytes | str:
        return b"" if capture_bytes else ""

    with monkeypatch.context() as no_op_git:
        no_op_git.setattr(module, "_run_git", successful_noop)
        with pytest.raises(module.VendorError, match="made no changes"):
            module._apply_patch(baseline, patch_path, (_INCLUDE,), reverse=True)


def test_sync_migration_isolates_patch_application_from_parent_repository(
    tmp_path: Path,
) -> None:
    upstream, base_commit = _make_upstream(
        tmp_path,
        {
            "compat.py": "MODE = 'upstream'\n",
            "feature.py": "FEATURE = 0\n",
        },
    )
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    destination = consumer / _DESTINATION
    (destination / "compat.py").write_text("MODE = 'trtllm'\n", encoding="utf-8")
    _create_vendor(consumer, lock, upstream, base_commit, mode="patched")
    patch_path = consumer / str(_vendor_data(lock)["patch"])
    original_patch = patch_path.read_bytes()
    _run(["git", "init", "--initial-branch=main", consumer], cwd=tmp_path)
    nested_tmp = consumer / ".tmp"
    nested_tmp.mkdir()

    (upstream / _SOURCE / "feature.py").write_text("FEATURE = 1\n", encoding="utf-8")
    updated_commit = _commit(upstream, "update source outside compatibility patch")
    _vendor(
        consumer,
        lock,
        "sync",
        _VENDOR_NAME,
        "--commit",
        updated_commit,
        "--repo",
        upstream,
        env_overrides={"TMPDIR": str(nested_tmp)},
    )

    vendor = _vendor_data(lock)
    assert vendor["commit"] == updated_commit
    assert vendor["patch"] == "3rdparty/vendor_patches/example.patch"
    assert patch_path.read_bytes() == original_patch
    assert (destination / "compat.py").read_text(encoding="utf-8") == "MODE = 'trtllm'\n"
    assert (destination / "feature.py").read_text(encoding="utf-8") == "FEATURE = 1\n"
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--offline")
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)


def test_sync_migration_drops_fully_absorbed_patch_with_adjacent_upstream_edit(
    tmp_path: Path,
) -> None:
    upstream, base_commit = _make_upstream(tmp_path, {"compat.py": "MODE = 'upstream'\n"})
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    destination = consumer / _DESTINATION
    (destination / "compat.py").write_text("MODE = 'trtllm'\n", encoding="utf-8")
    _create_vendor(consumer, lock, upstream, base_commit, mode="patched")
    patch_path = consumer / str(_vendor_data(lock)["patch"])

    (upstream / _SOURCE / "compat.py").write_text(
        "MODE = 'trtllm'\nFEATURE = 1\n", encoding="utf-8"
    )
    absorbed_commit = _commit(upstream, "absorb patch next to new upstream source")
    _vendor(
        consumer,
        lock,
        "sync",
        _VENDOR_NAME,
        "--commit",
        absorbed_commit,
        "--repo",
        upstream,
    )

    vendor = _vendor_data(lock)
    assert vendor["commit"] == absorbed_commit
    assert "patch" not in vendor
    assert "patch_digest" not in vendor
    assert not patch_path.exists()
    assert (destination / "compat.py").read_text(encoding="utf-8") == (
        "MODE = 'trtllm'\nFEATURE = 1\n"
    )
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--offline")
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)


def test_sync_migration_does_not_treat_a_conflicting_deletion_as_absorbed(
    tmp_path: Path,
) -> None:
    upstream, base_commit = _make_upstream(
        tmp_path, {"compat.py": "REMOVE_DOWNSTREAM = True\nKEEP = True\n"}
    )
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    destination = consumer / _DESTINATION
    (destination / "compat.py").write_text("KEEP = True\n", encoding="utf-8")
    _create_vendor(consumer, lock, upstream, base_commit, mode="patched")
    patch_path = consumer / str(_vendor_data(lock)["patch"])
    original_lock = lock.read_bytes()
    original_destination = _tree_snapshot(destination)
    original_patch = patch_path.read_bytes()

    (upstream / _SOURCE / "compat.py").write_text(
        "UPSTREAM_REPLACEMENT = True\nKEEP = True\n", encoding="utf-8"
    )
    conflicting_commit = _commit(upstream, "replace the line deleted downstream")
    result = _vendor(
        consumer,
        lock,
        "sync",
        _VENDOR_NAME,
        "--commit",
        conflicting_commit,
        "--repo",
        upstream,
        check=False,
    )

    _assert_failure(result, "conflict")
    assert lock.read_bytes() == original_lock
    assert _tree_snapshot(destination) == original_destination
    assert patch_path.read_bytes() == original_patch


@pytest.mark.parametrize(
    "upstream_source",
    (
        "MODE = 'new-upstream'\nFEATURE = 1\nMODE = 'trtllm'\n",
        "MODE = 'trtllm'\nMODE = 'upstream'\n",
    ),
)
def test_sync_migration_does_not_absorb_relocated_duplicate_content(
    tmp_path: Path,
    upstream_source: str,
) -> None:
    upstream, base_commit = _make_upstream(tmp_path, {"compat.py": "MODE = 'upstream'\n"})
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    destination = consumer / _DESTINATION
    (destination / "compat.py").write_text("MODE = 'trtllm'\n", encoding="utf-8")
    _create_vendor(consumer, lock, upstream, base_commit, mode="patched")
    patch_path = consumer / str(_vendor_data(lock)["patch"])
    original_lock = lock.read_bytes()
    original_destination = _tree_snapshot(destination)
    original_patch = patch_path.read_bytes()

    (upstream / _SOURCE / "compat.py").write_text(upstream_source, encoding="utf-8")
    conflicting_commit = _commit(upstream, "relocate duplicated downstream content")
    result = _vendor(
        consumer,
        lock,
        "sync",
        _VENDOR_NAME,
        "--commit",
        conflicting_commit,
        "--repo",
        upstream,
        check=False,
    )

    _assert_failure(result, "conflict")
    assert lock.read_bytes() == original_lock
    assert _tree_snapshot(destination) == original_destination
    assert patch_path.read_bytes() == original_patch


def test_sync_migration_rebases_and_drops_absorbed_patch(tmp_path: Path) -> None:
    upstream, base_commit = _make_upstream(
        tmp_path,
        {
            "a.py": "A = 0\n",
            "b.py": "B = 0\n",
            "feature.py": "FEATURE = 0\n",
        },
    )
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    destination = consumer / _DESTINATION
    _write_files(destination, {"a.py": "A = 1\n", "b.py": "B = 1\n"})
    _create_vendor(consumer, lock, upstream, base_commit, mode="patched")
    patch_path = consumer / str(_vendor_data(lock)["patch"])
    original_patch = patch_path.read_bytes()

    (upstream / _SOURCE / "feature.py").write_text("FEATURE = 1\n", encoding="utf-8")
    unrelated_commit = _commit(upstream, "update unrelated upstream source")
    _vendor(
        consumer,
        lock,
        "sync",
        _VENDOR_NAME,
        "--branch",
        "main",
        "--commit",
        unrelated_commit,
        "--repo",
        upstream,
    )
    unrelated_vendor = _vendor_data(lock)
    assert unrelated_vendor["branch"] == "main"
    assert unrelated_vendor["commit"] == unrelated_commit
    assert unrelated_vendor["patch"] == "3rdparty/vendor_patches/example.patch"
    assert patch_path.read_bytes() == original_patch
    assert (destination / "a.py").read_text(encoding="utf-8") == "A = 1\n"
    assert (destination / "b.py").read_text(encoding="utf-8") == "B = 1\n"
    assert (destination / "feature.py").read_text(encoding="utf-8") == "FEATURE = 1\n"
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)

    (upstream / _SOURCE / "a.py").write_text("A = 1\n", encoding="utf-8")
    partial_commit = _commit(upstream, "absorb compatibility change A")
    _vendor(
        consumer,
        lock,
        "sync",
        _VENDOR_NAME,
        "--commit",
        partial_commit,
        "--repo",
        upstream,
    )
    partial_vendor = _vendor_data(lock)
    partial_patch = patch_path.read_bytes()
    assert partial_vendor["commit"] == partial_commit
    assert partial_vendor["patch_digest"] == f"sha256:{hashlib.sha256(partial_patch).hexdigest()}"
    assert b"diff --git a/a.py b/a.py" not in partial_patch
    assert b"diff --git a/b.py b/b.py" in partial_patch
    assert (destination / "a.py").read_text(encoding="utf-8") == "A = 1\n"
    assert (destination / "b.py").read_text(encoding="utf-8") == "B = 1\n"
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)

    (upstream / _SOURCE / "b.py").write_text("B = 1\n", encoding="utf-8")
    absorbed_commit = _commit(upstream, "absorb compatibility change B")
    _vendor(
        consumer,
        lock,
        "sync",
        _VENDOR_NAME,
        "--commit",
        absorbed_commit,
        "--repo",
        upstream,
    )
    exact_vendor = _vendor_data(lock)
    assert exact_vendor["commit"] == absorbed_commit
    assert exact_vendor["branch"] == "main"
    assert "patch" not in exact_vendor
    assert "patch_digest" not in exact_vendor
    assert not patch_path.exists()
    assert (destination / "a.py").read_text(encoding="utf-8") == "A = 1\n"
    assert (destination / "b.py").read_text(encoding="utf-8") == "B = 1\n"
    assert (destination / "feature.py").read_text(encoding="utf-8") == "FEATURE = 1\n"
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--offline")
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)


def test_sync_migration_uses_new_repo_without_locked_commit(tmp_path: Path) -> None:
    old_upstream, old_commit = _make_upstream(
        tmp_path,
        {
            "compat.py": "MODE = 'upstream'\n",
            "feature.py": "FEATURE = 'old'\n",
        },
    )
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(old_upstream, consumer)
    destination = consumer / _DESTINATION
    (destination / "compat.py").write_text("MODE = 'trtllm'\n", encoding="utf-8")
    _create_vendor(consumer, lock, old_upstream, old_commit, mode="patched")
    patch_path = consumer / str(_vendor_data(lock)["patch"])
    original_patch = patch_path.read_bytes()

    replacement_root = tmp_path / "replacement"
    replacement_root.mkdir()
    new_upstream, new_commit = _make_upstream(
        replacement_root,
        {
            "compat.py": "MODE = 'upstream'\n",
            "feature.py": "FEATURE = 'new'\n",
        },
    )
    missing_old_commit = _run(
        ["git", "-C", new_upstream, "cat-file", "-e", f"{old_commit}^{{commit}}"],
        cwd=new_upstream,
        check=False,
    )
    assert missing_old_commit.returncode != 0
    unreachable_old_upstream = tmp_path / "unreachable-old-upstream"
    old_upstream.rename(unreachable_old_upstream)
    assert not old_upstream.exists()

    _vendor(
        consumer,
        lock,
        "sync",
        _VENDOR_NAME,
        "--url",
        new_upstream.as_uri(),
        "--branch",
        "main",
        "--commit",
        new_commit,
        "--repo",
        new_upstream,
    )

    vendor = _vendor_data(lock)
    assert vendor["url"] == new_upstream.as_uri()
    assert vendor["branch"] == "main"
    assert vendor["commit"] == new_commit
    assert vendor["patch"] == "3rdparty/vendor_patches/example.patch"
    assert patch_path.read_bytes() == original_patch
    assert (destination / "compat.py").read_text(encoding="utf-8") == "MODE = 'trtllm'\n"
    assert (destination / "feature.py").read_text(encoding="utf-8") == "FEATURE = 'new'\n"
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--offline")
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", new_upstream)


def test_merge_trees_preserves_independent_content_modes_and_binary_changes() -> None:
    module = _load_vendor_sources_module()
    file_entry = module.FileEntry
    base_text = "\n".join(
        [
            "DOWNSTREAM = 0",
            *(f"KEEP_{index} = {index}" for index in range(10)),
            "UPSTREAM = 0",
            "",
        ]
    ).encode()
    downstream_text = base_text.replace(b"DOWNSTREAM = 0", b"DOWNSTREAM = 1")
    upstream_text = base_text.replace(b"UPSTREAM = 0", b"UPSTREAM = 1")
    base = {
        "script.py": file_entry(base_text, executable=False),
        "downstream.bin": file_entry(b"\x00base-downstream\xff", executable=False),
        "upstream.bin": file_entry(b"\x00base-upstream\xff", executable=False),
    }
    downstream = {
        "script.py": file_entry(downstream_text, executable=True),
        "downstream.bin": file_entry(b"\x00trtllm\xfe", executable=True),
        "upstream.bin": base["upstream.bin"],
    }
    upstream = {
        "script.py": file_entry(upstream_text, executable=False),
        "downstream.bin": base["downstream.bin"],
        "upstream.bin": file_entry(b"\x00new-upstream\xfd", executable=False),
    }

    merged = module._merge_trees(base, downstream, upstream)

    expected_text = upstream_text.replace(b"DOWNSTREAM = 0", b"DOWNSTREAM = 1")
    assert merged["script.py"] == file_entry(expected_text, executable=True)
    assert merged["downstream.bin"] == downstream["downstream.bin"]
    assert merged["upstream.bin"] == upstream["upstream.bin"]


def test_merge_trees_reports_same_file_binary_changes_as_a_content_conflict() -> None:
    module = _load_vendor_sources_module()
    file_entry = module.FileEntry
    with pytest.raises(module.VendorError, match="binary.*shared.bin"):
        module._merge_trees(
            {"shared.bin": file_entry(b"\0base\xff")},
            {"shared.bin": file_entry(b"\0downstream\xfe")},
            {"shared.bin": file_entry(b"\0upstream\xfd")},
        )


def test_merge_file_translates_preparation_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_vendor_sources_module()
    real_write_bytes = Path.write_bytes

    def fail_merge_input_write(path: Path, data: bytes) -> int:
        if path.name == "base" and path.parent.name.startswith("trtllm-vendor-merge-"):
            raise OSError("injected merge-input write failure")
        return real_write_bytes(path, data)

    with monkeypatch.context() as failure:
        failure.setattr(Path, "write_bytes", fail_merge_input_write)
        with pytest.raises(module.VendorError, match="prepare source merge.*compat.py"):
            module._merge_file_contents(
                "compat.py",
                b"MODE = 'base'\n",
                b"MODE = 'downstream'\n",
                b"MODE = 'upstream'\n",
            )


def test_merge_file_does_not_mislabel_operational_git_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_vendor_sources_module()

    def git_not_found(*_: object, **__: object) -> None:
        raise FileNotFoundError("injected missing Git")

    with monkeypatch.context() as failure:
        failure.setattr(module.subprocess, "run", git_not_found)
        with pytest.raises(module.VendorError) as missing_git:
            module._merge_file_contents(
                "compat.py",
                b"MODE = 'base'\n",
                b"MODE = 'downstream'\n",
                b"MODE = 'upstream'\n",
            )
    assert "not found" in str(missing_git.value).lower()
    assert "conflict" not in str(missing_git.value).lower()

    def fail_operationally(*_: object, **__: object) -> bytes:
        raise module._GitCommandError("injected Git option failure", 129)

    with monkeypatch.context() as failure:
        failure.setattr(module, "_run_git", fail_operationally)
        with pytest.raises(module._GitCommandError) as operational_failure:
            module._merge_file_contents(
                "compat.py",
                b"MODE = 'base'\n",
                b"MODE = 'downstream'\n",
                b"MODE = 'upstream'\n",
            )
    assert "conflict" not in str(operational_failure.value).lower()


def test_sync_migration_conflict_does_not_mutate_state(tmp_path: Path) -> None:
    upstream, base_commit = _make_upstream(tmp_path, {"compat.py": "MODE = 'upstream'\n"})
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    destination = consumer / _DESTINATION
    (destination / "compat.py").write_text("MODE = 'trtllm'\n", encoding="utf-8")
    _create_vendor(consumer, lock, upstream, base_commit, mode="patched")
    patch_path = consumer / str(_vendor_data(lock)["patch"])

    (upstream / _SOURCE / "compat.py").write_text("MODE = 'new-upstream'\n", encoding="utf-8")
    conflicting_commit = _commit(upstream, "conflicting upstream change")
    original_lock = lock.read_bytes()
    original_destination = _tree_snapshot(destination)
    original_patch = patch_path.read_bytes()
    original_patch_mode = stat.S_IMODE(patch_path.stat().st_mode)
    original_patch_directory = _tree_snapshot(patch_path.parent)
    original_destination_entries = sorted(path.name for path in destination.parent.iterdir())

    result = _vendor(
        consumer,
        lock,
        "sync",
        _VENDOR_NAME,
        "--branch",
        "main",
        "--commit",
        conflicting_commit,
        "--repo",
        upstream,
        check=False,
    )

    _assert_failure(result, "conflict")
    _assert_failure(result, "compat.py")
    assert lock.read_bytes() == original_lock
    assert _tree_snapshot(destination) == original_destination
    assert patch_path.read_bytes() == original_patch
    assert stat.S_IMODE(patch_path.stat().st_mode) == original_patch_mode
    assert _tree_snapshot(patch_path.parent) == original_patch_directory
    assert (
        sorted(path.name for path in destination.parent.iterdir()) == original_destination_entries
    )
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)


def test_sync_migration_rolls_back_when_lock_save_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    original_source = "\n".join(
        [
            "UPSTREAM = 0",
            *(f"KEEP_{index} = {index}" for index in range(10)),
            "DOWNSTREAM = 0",
            "",
        ]
    )
    downstream_source = original_source.replace("DOWNSTREAM = 0", "DOWNSTREAM = 1")
    updated_source = original_source.replace("UPSTREAM = 0", "UPSTREAM = 1")
    upstream, base_commit = _make_upstream(tmp_path, {"compat.py": original_source})
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    destination = consumer / _DESTINATION
    (destination / "compat.py").write_text(downstream_source, encoding="utf-8")
    _create_vendor(consumer, lock, upstream, base_commit, mode="patched")
    patch_path = consumer / str(_vendor_data(lock)["patch"])

    (upstream / _SOURCE / "compat.py").write_text(updated_source, encoding="utf-8")
    updated_commit = _commit(upstream, "update upstream half of compatibility source")
    original_lock = lock.read_bytes()
    original_destination = _tree_snapshot(destination)
    original_patch = patch_path.read_bytes()
    original_patch_mode = stat.S_IMODE(patch_path.stat().st_mode)
    original_patch_directory = _tree_snapshot(patch_path.parent)
    original_destination_entries = sorted(path.name for path in destination.parent.iterdir())
    module = _load_vendor_sources_module()

    def fail_save_lock(_: object) -> None:
        raise OSError("injected sync lock-save failure")

    with monkeypatch.context() as failure:
        failure.setattr(module, "_save_lock", fail_save_lock)
        result = module.main(
            [
                "--lock",
                str(lock),
                "sync",
                _VENDOR_NAME,
                "--commit",
                updated_commit,
                "--repo",
                str(upstream),
            ]
        )

    assert result == 1
    assert "injected sync lock-save failure" in capsys.readouterr().err
    assert lock.read_bytes() == original_lock
    assert _tree_snapshot(destination) == original_destination
    assert patch_path.read_bytes() == original_patch
    assert stat.S_IMODE(patch_path.stat().st_mode) == original_patch_mode
    assert _tree_snapshot(patch_path.parent) == original_patch_directory
    assert (
        sorted(path.name for path in destination.parent.iterdir()) == original_destination_entries
    )

    _vendor(
        consumer,
        lock,
        "sync",
        _VENDOR_NAME,
        "--commit",
        updated_commit,
        "--repo",
        upstream,
    )
    merged_source = updated_source.replace("DOWNSTREAM = 0", "DOWNSTREAM = 1")
    assert (destination / "compat.py").read_text(encoding="utf-8") == merged_source
    assert patch_path.read_bytes() != original_patch
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)


def test_sync_migration_keeps_committed_state_after_visible_lock_interrupt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_source = "\n".join(
        [
            "UPSTREAM = 0",
            *(f"KEEP_{index} = {index}" for index in range(10)),
            "DOWNSTREAM = 0",
            "",
        ]
    )
    downstream_source = original_source.replace("DOWNSTREAM = 0", "DOWNSTREAM = 1")
    updated_source = original_source.replace("UPSTREAM = 0", "UPSTREAM = 1")
    upstream, base_commit = _make_upstream(tmp_path, {"compat.py": original_source})
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    destination = consumer / _DESTINATION
    (destination / "compat.py").write_text(downstream_source, encoding="utf-8")
    _create_vendor(consumer, lock, upstream, base_commit, mode="patched")
    patch_path = consumer / str(_vendor_data(lock)["patch"])
    original_patch = patch_path.read_bytes()
    (upstream / _SOURCE / "compat.py").write_text(updated_source, encoding="utf-8")
    updated_commit = _commit(upstream, "update upstream before interrupted migration")
    module = _load_vendor_sources_module()
    real_contextlib = module.contextlib
    real_exit_stack = real_contextlib.ExitStack

    class InterruptAfterBodyExitStack(real_exit_stack):
        def __exit__(
            self,
            exception_type: type[BaseException] | None,
            exception: BaseException | None,
            traceback: object,
        ) -> bool:
            if exception_type is not None:
                return real_exit_stack.__exit__(self, exception_type, exception, traceback)
            interrupt = KeyboardInterrupt("injected interruption after lock replacement")
            handled = real_exit_stack.__exit__(
                self, KeyboardInterrupt, interrupt, interrupt.__traceback__
            )
            assert not handled
            raise interrupt

    class ContextlibProxy:
        ExitStack = InterruptAfterBodyExitStack

        def __getattr__(self, name: str) -> object:
            return getattr(real_contextlib, name)

    with monkeypatch.context() as interruption:
        interruption.setattr(module, "contextlib", ContextlibProxy())
        with pytest.raises(KeyboardInterrupt, match="after lock replacement"):
            module.main(
                [
                    "--lock",
                    str(lock),
                    "sync",
                    _VENDOR_NAME,
                    "--commit",
                    updated_commit,
                    "--repo",
                    str(upstream),
                ]
            )

    migrated_vendor = _vendor_data(lock)
    migrated_patch = patch_path.read_bytes()
    assert migrated_vendor["commit"] == updated_commit
    assert migrated_vendor["patch_digest"] == (
        f"sha256:{hashlib.sha256(migrated_patch).hexdigest()}"
    )
    assert migrated_patch != original_patch
    assert (destination / "compat.py").read_text(encoding="utf-8") == updated_source.replace(
        "DOWNSTREAM = 0", "DOWNSTREAM = 1"
    )
    assert not list(destination.parent.glob(".example.vendor-*"))
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--offline")
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)


def test_sync_migration_backup_cleanup_failure_is_warning_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    upstream, base_commit = _make_upstream(tmp_path, {"kernel.py": "VALUE = 1\n"})
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    _create_vendor(consumer, lock, upstream, base_commit, mode="exact")
    destination = consumer / _DESTINATION
    backup = destination.parent / ".example.vendor-backup"
    (upstream / _SOURCE / "kernel.py").write_text("VALUE = 2\n", encoding="utf-8")
    updated_commit = _commit(upstream, "update exact source before cleanup failure")
    module = _load_vendor_sources_module()
    real_remove_path = module._remove_path

    def fail_backup_cleanup(path: Path) -> None:
        if path == backup:
            raise OSError("injected destination-backup cleanup failure")
        real_remove_path(path)

    with monkeypatch.context() as cleanup_failure:
        cleanup_failure.setattr(module, "_remove_path", fail_backup_cleanup)
        result = module.main(
            [
                "--lock",
                str(lock),
                "sync",
                _VENDOR_NAME,
                "--commit",
                updated_commit,
                "--repo",
                str(upstream),
            ]
        )

    captured = capsys.readouterr()
    assert result == 0
    assert "warning:" in captured.err.lower()
    assert "replacement is committed" in captured.err
    assert "delete that backup manually" in captured.err.lower()
    assert str(backup) in captured.err
    assert backup.is_dir()
    assert _vendor_data(lock)["commit"] == updated_commit
    assert (destination / "kernel.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--offline")
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)


def test_sync_absorbed_patch_directory_sync_failure_retains_valid_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    upstream, consumer, lock, destination, patch_path, absorbed_commit = (
        _prepare_absorbed_patch_migration(tmp_path)
    )
    patch_content = patch_path.read_bytes()
    module = _load_vendor_sources_module()

    def fail_directory_sync(_: Path) -> None:
        raise OSError("injected sync lock-directory failure")

    with monkeypatch.context() as sync_failure:
        sync_failure.setattr(module, "_sync_directory", fail_directory_sync)
        result = module.main(
            [
                "--lock",
                str(lock),
                "sync",
                _VENDOR_NAME,
                "--commit",
                absorbed_commit,
                "--repo",
                str(upstream),
            ]
        )

    assert result == 1
    assert "injected sync lock-directory failure" in capsys.readouterr().err
    vendor = _vendor_data(lock)
    assert vendor["commit"] == absorbed_commit
    assert "patch" not in vendor
    assert "patch_digest" not in vendor
    assert patch_path.read_bytes() == patch_content
    assert (destination / "compat.py").read_text(encoding="utf-8") == "MODE = 'trtllm'\n"
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--offline")
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)


def test_sync_absorbed_patch_interrupt_after_visible_lock_leaves_valid_orphan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    upstream, consumer, lock, destination, patch_path, absorbed_commit = (
        _prepare_absorbed_patch_migration(tmp_path)
    )
    patch_content = patch_path.read_bytes()
    patch_mode = stat.S_IMODE(patch_path.stat().st_mode)
    module = _load_vendor_sources_module()
    real_save_lock = module._save_lock

    def save_then_interrupt(lock_file: object) -> None:
        real_save_lock(lock_file)
        raise KeyboardInterrupt("injected absorbed-patch interruption")

    with monkeypatch.context() as interruption:
        interruption.setattr(module, "_save_lock", save_then_interrupt)
        with pytest.raises(KeyboardInterrupt, match="absorbed-patch interruption"):
            module.main(
                [
                    "--lock",
                    str(lock),
                    "sync",
                    _VENDOR_NAME,
                    "--commit",
                    absorbed_commit,
                    "--repo",
                    str(upstream),
                ]
            )

    vendor = _vendor_data(lock)
    assert vendor["commit"] == absorbed_commit
    assert "patch" not in vendor
    assert "patch_digest" not in vendor
    assert patch_path.read_bytes() == patch_content
    assert stat.S_IMODE(patch_path.stat().st_mode) == patch_mode
    assert (destination / "compat.py").read_text(encoding="utf-8") == "MODE = 'trtllm'\n"
    assert not list(destination.parent.glob(".example.vendor-*"))
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--offline")
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)


def test_sync_absorbed_patch_cleanup_failure_warns_in_commit_order(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    upstream, consumer, lock, destination, patch_path, absorbed_commit = (
        _prepare_absorbed_patch_migration(tmp_path)
    )
    patch_content = patch_path.read_bytes()
    module = _load_vendor_sources_module()
    real_save_lock_checked = module._save_lock_checked
    real_sync_directory = module._sync_directory
    real_unlink = Path.unlink
    events: list[str] = []

    def record_lock_save(lock_file: object, description: str) -> None:
        real_save_lock_checked(lock_file, description)
        events.append("lock-save")

    def record_directory_sync(path: Path) -> None:
        real_sync_directory(path)
        events.append("directory-sync")

    def fail_patch_cleanup(path: Path, missing_ok: bool = False) -> None:
        if path == patch_path:
            events.append("patch-unlink")
            raise OSError("injected sync absorbed-patch cleanup failure")
        real_unlink(path, missing_ok=missing_ok)

    with monkeypatch.context() as cleanup_failure:
        cleanup_failure.setattr(module, "_save_lock_checked", record_lock_save)
        cleanup_failure.setattr(module, "_sync_directory", record_directory_sync)
        cleanup_failure.setattr(Path, "unlink", fail_patch_cleanup)
        result = module.main(
            [
                "--lock",
                str(lock),
                "sync",
                _VENDOR_NAME,
                "--commit",
                absorbed_commit,
                "--repo",
                str(upstream),
            ]
        )

    captured = capsys.readouterr()
    assert result == 0
    assert events == ["lock-save", "directory-sync", "patch-unlink"]
    assert "warning:" in captured.err.lower()
    assert "lock is committed and valid" in captured.err
    assert "unreferenced orphan" in captured.err
    vendor = _vendor_data(lock)
    assert vendor["commit"] == absorbed_commit
    assert "patch" not in vendor
    assert "patch_digest" not in vendor
    assert patch_path.read_bytes() == patch_content
    assert (destination / "compat.py").read_text(encoding="utf-8") == "MODE = 'trtllm'\n"
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--offline")
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)


def test_plain_sync_and_migration_serialize_state_mutation(tmp_path: Path) -> None:
    upstream, base_commit = _make_upstream(tmp_path, {"kernel.py": "VALUE = 1\n"})
    consumer, lock = _make_consumer(tmp_path)
    _copy_python_sources(upstream, consumer)
    _create_vendor(consumer, lock, upstream, base_commit, mode="exact")
    destination = consumer / _DESTINATION
    (upstream / _SOURCE / "kernel.py").write_text("VALUE = 2\n", encoding="utf-8")
    updated_commit = _commit(upstream, "update source for concurrent migration")
    plain_cached = tmp_path / "plain-sync-cached"
    release_plain = tmp_path / "release-plain-sync"
    migration_attempted = tmp_path / "migration-lock-attempted"
    migration_entered = tmp_path / "migration-handler-entered"

    plain_script = "\n".join(
        [
            "import importlib.util",
            "import pathlib",
            "import sys",
            "import time",
            "module_path, lock, upstream, cached, release = sys.argv[1:]",
            "spec = importlib.util.spec_from_file_location('_vendor_sources_plain_sync', module_path)",
            "assert spec is not None and spec.loader is not None",
            "module = importlib.util.module_from_spec(spec)",
            "sys.modules[spec.name] = module",
            "spec.loader.exec_module(module)",
            "replace_destination = module._replace_selected_destination",
            "def pause_after_materialization(lock_file, vendor, tree):",
            "    pathlib.Path(cached).write_text('cached', encoding='utf-8')",
            "    while not pathlib.Path(release).exists():",
            "        time.sleep(0.01)",
            "    replace_destination(lock_file, vendor, tree)",
            "module._replace_selected_destination = pause_after_materialization",
            "raise SystemExit(module.main([",
            "    '--lock', lock, 'sync', 'example', '--repo', upstream,",
            "]))",
        ]
    )
    migration_script = "\n".join(
        [
            "import contextlib",
            "import importlib.util",
            "import pathlib",
            "import sys",
            "module_path, lock, upstream, commit, attempted, entered = sys.argv[1:]",
            "spec = importlib.util.spec_from_file_location('_vendor_sources_migration', module_path)",
            "assert spec is not None and spec.loader is not None",
            "module = importlib.util.module_from_spec(spec)",
            "sys.modules[spec.name] = module",
            "spec.loader.exec_module(module)",
            "state_lock = module._vendor_state_lock",
            "@contextlib.contextmanager",
            "def marked_state_lock(path, *, exclusive):",
            "    pathlib.Path(attempted).write_text('attempted', encoding='utf-8')",
            "    with state_lock(path, exclusive=exclusive):",
            "        pathlib.Path(entered).write_text('entered', encoding='utf-8')",
            "        yield",
            "module._vendor_state_lock = marked_state_lock",
            "raise SystemExit(module.main([",
            "    '--lock', lock, 'sync', 'example', '--commit', commit, '--repo', upstream,",
            "]))",
        ]
    )

    def wait_for_marker(marker: Path, process: subprocess.Popen[str], description: str) -> None:
        deadline = time.monotonic() + 10
        while not marker.exists() and process.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        if not marker.exists():
            stdout, stderr = process.communicate(timeout=5)
            pytest.fail(f"{description} was not reached.\nstdout:\n{stdout}\nstderr:\n{stderr}")

    plain = subprocess.Popen(
        [
            sys.executable,
            "-c",
            plain_script,
            str(_VENDOR_SOURCES),
            str(lock),
            str(upstream),
            str(plain_cached),
            str(release_plain),
        ],
        cwd=consumer,
        env=_command_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    migration: subprocess.Popen[str] | None = None
    try:
        wait_for_marker(plain_cached, plain, "plain-sync materialization barrier")
        migration = subprocess.Popen(
            [
                sys.executable,
                "-c",
                migration_script,
                str(_VENDOR_SOURCES),
                str(lock),
                str(upstream),
                updated_commit,
                str(migration_attempted),
                str(migration_entered),
            ],
            cwd=consumer,
            env=_command_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        wait_for_marker(migration_attempted, migration, "migration lock-attempt barrier")
        time.sleep(0.2)
        assert migration.poll() is None
        assert not migration_entered.exists()

        release_plain.write_text("release", encoding="utf-8")
        plain_stdout, plain_stderr = plain.communicate(timeout=10)
        migration_stdout, migration_stderr = migration.communicate(timeout=10)
    finally:
        if plain.poll() is None:
            plain.kill()
            plain.communicate(timeout=5)
        if migration is not None and migration.poll() is None:
            migration.kill()
            migration.communicate(timeout=5)

    assert plain.returncode == 0, f"stdout:\n{plain_stdout}\nstderr:\n{plain_stderr}"
    assert migration.returncode == 0, f"stdout:\n{migration_stdout}\nstderr:\n{migration_stderr}"
    assert migration_entered.exists()
    assert _vendor_data(lock)["commit"] == updated_commit
    assert (destination / "kernel.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--offline")
    _vendor(consumer, lock, "check", _VENDOR_NAME, "--repo", upstream)


@pytest.mark.parametrize("waiter_exclusive", (True, False))
def test_state_lock_remains_held_across_lock_file_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    waiter_exclusive: bool,
) -> None:
    module = _load_vendor_sources_module()
    lock = tmp_path / "vendor-sources.lock.yaml"
    lock.write_text("old lock\n", encoding="utf-8")
    replacement_visible = threading.Event()
    release_holder = threading.Event()
    waiter_attempted = threading.Event()
    waiter_entered = threading.Event()
    observed_locks: list[bytes] = []
    failures: list[BaseException] = []
    waiter: threading.Thread
    real_flock = module.fcntl.flock

    def marked_flock(file_descriptor: int, operation: int) -> None:
        waiter_operation = module.fcntl.LOCK_EX if waiter_exclusive else module.fcntl.LOCK_SH
        if threading.current_thread() is waiter and operation == waiter_operation:
            waiter_attempted.set()
        real_flock(file_descriptor, operation)

    def hold_lock_through_replacement() -> None:
        try:
            with module._vendor_state_lock(lock, exclusive=True):
                module._atomic_write(lock, b"new lock\n")
                replacement_visible.set()
                if not release_holder.wait(timeout=5):
                    raise AssertionError("timed out waiting to release the state lock")
        except BaseException as error:
            failures.append(error)

    def wait_for_replaced_lock() -> None:
        try:
            with module._vendor_state_lock(lock, exclusive=waiter_exclusive):
                observed_locks.append(lock.read_bytes())
                waiter_entered.set()
        except BaseException as error:
            failures.append(error)

    monkeypatch.setattr(module.fcntl, "flock", marked_flock)
    holder = threading.Thread(target=hold_lock_through_replacement)
    waiter = threading.Thread(target=wait_for_replaced_lock)
    holder.start()
    assert replacement_visible.wait(timeout=5)
    waiter.start()
    assert waiter_attempted.wait(timeout=5)
    assert not waiter_entered.wait(timeout=0.1)
    release_holder.set()
    holder.join(timeout=5)
    waiter.join(timeout=5)

    assert not holder.is_alive()
    assert not waiter.is_alive()
    assert failures == []
    assert waiter_entered.is_set()
    assert observed_locks == [b"new lock\n"]
