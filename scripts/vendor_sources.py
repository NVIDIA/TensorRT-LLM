#!/usr/bin/env python3
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

"""Manage source trees vendored from immutable Git commits."""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import fnmatch
import hashlib
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.parse
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_LOCK = _REPO_ROOT / "3rdparty" / "vendor_sources.lock.yaml"
_SCHEMA_VERSION = 1
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]*\Z")
_DIGEST_PREFIX = "sha256-tree-v1:"
_PATCH_DIGEST_PREFIX = "sha256:"
_GIT_TIMEOUT_SECONDS = 60
_GIT_LOCAL_ENVIRONMENT_VARIABLES = {
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_ATTR_GLOBAL",
    "GIT_ATTR_SOURCE",
    "GIT_ATTR_SYSTEM",
    "GIT_CEILING_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_CONFIG",
    "GIT_CONFIG_COUNT",
    "GIT_CONFIG_PARAMETERS",
    "GIT_DIR",
    "GIT_GRAFT_FILE",
    "GIT_IMPLICIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_NO_REPLACE_OBJECTS",
    "GIT_OBJECT_DIRECTORY",
    "GIT_PREFIX",
    "GIT_REPLACE_REF_BASE",
    "GIT_SHALLOW_FILE",
    "GIT_WORK_TREE",
}
_VENDOR_FIELDS = {
    "url",
    "branch",
    "tag",
    "commit",
    "source",
    "destination",
    "include",
    "patch",
    "patch_digest",
    "digest",
}


class VendorError(RuntimeError):
    """A user-actionable vendoring failure."""


class UpstreamUnavailable(VendorError):
    """The recorded upstream could not be accessed."""


class _UniqueKeyLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(
    loader: _UniqueKeyLoader,
    node: yaml.MappingNode,
    deep: bool = False,
) -> dict[object, object]:
    if any(
        isinstance(key_node, yaml.ScalarNode) and key_node.value == "<<"
        for key_node, _ in node.value
    ):
        raise yaml.constructor.ConstructorError(
            "while constructing a mapping",
            node.start_mark,
            "YAML merge keys are not allowed in the vendor lock",
            node.start_mark,
        )
    loader.flatten_mapping(node)
    result: dict[object, object] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in result
        except TypeError as error:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "found an unhashable mapping key",
                key_node.start_mark,
            ) from error
        if duplicate:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"found duplicate key {key!r}",
                key_node.start_mark,
            )
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


@dataclasses.dataclass(frozen=True)
class FileEntry:
    """One regular file in a normalized selected tree."""

    data: bytes
    executable: bool = False


Tree = dict[str, FileEntry]


@dataclasses.dataclass
class Vendor:
    """Validated lock entry for one managed source tree."""

    name: str
    url: str
    commit: str
    source: str
    destination: str
    include: tuple[str, ...]
    digest: str
    branch: str | None = None
    tag: str | None = None
    patch: str | None = None
    patch_digest: str | None = None

    def to_mapping(self) -> dict[str, object]:
        """Serialize this entry in the canonical field order."""
        result: dict[str, object] = {"url": self.url}
        if self.branch is not None:
            result["branch"] = self.branch
        if self.tag is not None:
            result["tag"] = self.tag
        result.update(
            {
                "commit": self.commit,
                "source": self.source,
                "destination": self.destination,
                "include": list(self.include),
            }
        )
        if self.patch is not None:
            result["patch"] = self.patch
            result["patch_digest"] = self.patch_digest
        result["digest"] = self.digest
        return result


@dataclasses.dataclass
class LockFile:
    """Validated collection of vendor entries and its consumer root."""

    path: Path
    root: Path
    vendors: dict[str, Vendor]


def _require_mapping(value: object, description: str) -> Mapping[object, object]:
    if not isinstance(value, Mapping):
        raise VendorError(f"{description} must be a mapping.")
    return value


def _validate_mapping_keys(
    mapping: Mapping[object, object], allowed: set[str], description: str
) -> set[str]:
    keys: set[str] = set()
    for key in mapping:
        if not isinstance(key, str):
            raise VendorError(f"{description} keys must be strings; got {key!r}.")
        keys.add(key)
    unknown = keys - allowed
    if unknown:
        raise VendorError(f"{description} has unexpected fields: {sorted(unknown)}.")
    return keys


def _require_string(value: object, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise VendorError(f"{description} must be a non-empty string.")
    return value


def _validate_relative_path(value: object, description: str) -> str:
    path_text = _require_string(value, description)
    if "\\" in path_text:
        raise VendorError(f"{description} must use POSIX separators: {path_text!r}.")
    path = PurePosixPath(path_text)
    if (
        path.is_absolute()
        or path_text == "."
        or path_text.startswith(":")
        or any(part in ("", ".", "..", ".git") for part in path.parts)
    ):
        raise VendorError(f"{description} must be a safe relative path: {path_text!r}.")
    return path.as_posix()


def _validate_url(value: object, vendor_name: str) -> str:
    url = _require_string(value, f"Vendor {vendor_name!r} url")
    parsed = urllib.parse.urlsplit(url)
    if parsed.password is not None or (
        parsed.scheme in ("http", "https") and parsed.username is not None
    ):
        raise VendorError(
            f"Vendor {vendor_name!r} URL must not contain credentials; use Git credential helpers."
        )
    return url


def _validate_include(value: object, vendor_name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise VendorError(f"Vendor {vendor_name!r} include must be a non-empty list.")
    patterns: list[str] = []
    for index, item in enumerate(value):
        pattern = _require_string(item, f"Vendor {vendor_name!r} include[{index}]")
        if "\\" in pattern or pattern.startswith("/"):
            raise VendorError(f"Vendor {vendor_name!r} has an unsafe include pattern: {pattern!r}.")
        if any(part == ".." for part in PurePosixPath(pattern).parts):
            raise VendorError(f"Vendor {vendor_name!r} has an unsafe include pattern: {pattern!r}.")
        patterns.append(pattern)
    if len(patterns) != len(set(patterns)):
        raise VendorError(f"Vendor {vendor_name!r} include patterns must be unique.")
    return tuple(patterns)


def _validate_digest(value: object, description: str) -> str:
    digest = _require_string(value, description)
    suffix = digest.removeprefix(_DIGEST_PREFIX)
    if not digest.startswith(_DIGEST_PREFIX) or not re.fullmatch(r"[0-9a-f]{64}", suffix):
        raise VendorError(f"{description} must use {_DIGEST_PREFIX}<64 lowercase hex digits>.")
    return digest


def _validate_patch_digest(value: object, description: str) -> str:
    digest = _require_string(value, description)
    suffix = digest.removeprefix(_PATCH_DIGEST_PREFIX)
    if not digest.startswith(_PATCH_DIGEST_PREFIX) or not re.fullmatch(r"[0-9a-f]{64}", suffix):
        raise VendorError(
            f"{description} must use {_PATCH_DIGEST_PREFIX}<64 lowercase hex digits>."
        )
    return digest


def _patch_digest(content: bytes) -> str:
    return f"{_PATCH_DIGEST_PREFIX}{hashlib.sha256(content).hexdigest()}"


def _validate_vendor(name: object, value: object) -> Vendor:
    vendor_name = _require_string(name, "Vendor name")
    if _NAME_PATTERN.fullmatch(vendor_name) is None:
        raise VendorError(
            f"Vendor name {vendor_name!r} must contain only lowercase letters, digits, '.', '_', or '-'."
        )
    mapping = _require_mapping(value, f"Vendor {vendor_name!r}")
    fields = _validate_mapping_keys(mapping, _VENDOR_FIELDS, f"Vendor {vendor_name!r}")
    required = {"url", "commit", "source", "destination", "include", "digest"}
    missing = required - fields
    if missing:
        raise VendorError(f"Vendor {vendor_name!r} is missing fields: {sorted(missing)}.")
    branch = mapping.get("branch")
    tag = mapping.get("tag")
    if branch is not None and tag is not None:
        raise VendorError(f"Vendor {vendor_name!r} cannot record both branch and tag.")
    branch_text = (
        None if branch is None else _require_string(branch, f"Vendor {vendor_name!r} branch")
    )
    tag_text = None if tag is None else _require_string(tag, f"Vendor {vendor_name!r} tag")
    for kind, ref in (("branch", branch_text), ("tag", tag_text)):
        if ref is not None:
            invalid_ref = (
                ref.startswith(("refs/", ".", "/"))
                or ref.endswith((".", "/", ".lock"))
                or ".." in ref
                or "@{" in ref
                or "//" in ref
                or any(char.isspace() or char in "~^:?*[\\" for char in ref)
            )
            if invalid_ref:
                raise VendorError(f"Vendor {vendor_name!r} {kind} must be a short Git name.")
    commit = _require_string(mapping["commit"], f"Vendor {vendor_name!r} commit")
    if _COMMIT_PATTERN.fullmatch(commit) is None:
        raise VendorError(f"Vendor {vendor_name!r} commit must be a full lowercase 40-hex SHA.")
    patch_value = mapping.get("patch")
    patch = (
        None
        if patch_value is None
        else _validate_relative_path(patch_value, f"Vendor {vendor_name!r} patch")
    )
    expected_patch = f"3rdparty/vendor_patches/{vendor_name}.patch"
    if patch is not None and patch != expected_patch:
        raise VendorError(f"Vendor {vendor_name!r} patch must be {expected_patch!r}.")
    patch_digest_value = mapping.get("patch_digest")
    if (patch is None) != (patch_digest_value is None):
        raise VendorError(
            f"Vendor {vendor_name!r} patch and patch_digest must either both be present or both be absent."
        )
    patch_digest = (
        None
        if patch_digest_value is None
        else _validate_patch_digest(patch_digest_value, f"Vendor {vendor_name!r} patch_digest")
    )
    include = _validate_include(mapping["include"], vendor_name)
    return Vendor(
        name=vendor_name,
        url=_validate_url(mapping["url"], vendor_name),
        branch=branch_text,
        tag=tag_text,
        commit=commit,
        source=_validate_relative_path(mapping["source"], f"Vendor {vendor_name!r} source"),
        destination=_validate_relative_path(
            mapping["destination"], f"Vendor {vendor_name!r} destination"
        ),
        include=include,
        patch=patch,
        patch_digest=patch_digest,
        digest=_validate_digest(mapping["digest"], f"Vendor {vendor_name!r} digest"),
    )


def _consumer_root(lock_path: Path) -> Path:
    if lock_path.parent.name == "3rdparty":
        return lock_path.parent.parent.resolve()
    return lock_path.parent.resolve()


def _validate_destination_layout(lock: LockFile) -> None:
    destinations: list[tuple[str, PurePosixPath]] = []
    try:
        lock_relative = PurePosixPath(lock.path.relative_to(lock.root).as_posix())
    except ValueError as error:
        raise VendorError(f"Vendor lock must be inside the consumer root: {lock.path}.") from error
    patch_directory = PurePosixPath("3rdparty/vendor_patches")
    for name, vendor in lock.vendors.items():
        path = PurePosixPath(vendor.destination)
        for other_name, other_path in destinations:
            if (
                path == other_path
                or path.is_relative_to(other_path)
                or other_path.is_relative_to(path)
            ):
                raise VendorError(
                    f"Vendor destinations overlap: {name!r} ({path}) and {other_name!r} ({other_path})."
                )
        destinations.append((name, path))
        if (
            path == patch_directory
            or path.is_relative_to(patch_directory)
            or patch_directory.is_relative_to(path)
        ):
            raise VendorError(
                f"Vendor {name!r} destination overlaps vendoring control files: {path}."
            )
        if lock_relative == path or lock_relative.is_relative_to(path):
            raise VendorError(f"Vendor {name!r} destination contains the vendor lock: {path}.")
        if vendor.patch is not None:
            patch_path = PurePosixPath(vendor.patch)
            if (
                patch_path == path
                or patch_path.is_relative_to(path)
                or path.is_relative_to(patch_path)
            ):
                raise VendorError(f"Vendor {name!r} destination overlaps its patch path: {path}.")


def _load_lock(
    path: Path,
    *,
    allow_missing: bool = False,
    allow_patch_recovery: bool = False,
) -> LockFile:
    resolved = path.resolve()
    if not resolved.exists():
        if allow_missing:
            return LockFile(path=resolved, root=_consumer_root(resolved), vendors={})
        raise VendorError(f"Vendor lock does not exist: {resolved}.")
    try:
        content = resolved.read_text(encoding="utf-8")
        if any(
            isinstance(token, (yaml.tokens.AliasToken, yaml.tokens.AnchorToken))
            for token in yaml.scan(content)
        ):
            raise VendorError("Vendor lock YAML aliases and anchors are not allowed.")
        loaded = yaml.load(content, Loader=_UniqueKeyLoader)
    except VendorError:
        raise
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise VendorError(f"Failed to parse vendor lock {resolved}: {error}") from error
    mapping = _require_mapping(loaded, "Vendor lock")
    _validate_mapping_keys(mapping, {"schema_version", "vendors"}, "Vendor lock")
    if mapping.get("schema_version") != _SCHEMA_VERSION:
        raise VendorError(f"Vendor lock schema_version must be {_SCHEMA_VERSION}.")
    raw_vendors = _require_mapping(mapping.get("vendors"), "Vendor lock vendors")
    vendors = {
        _require_string(name, "Vendor name"): _validate_vendor(name, value)
        for name, value in raw_vendors.items()
    }
    lock = LockFile(path=resolved, root=_consumer_root(resolved), vendors=vendors)
    _validate_destination_layout(lock)
    for vendor in vendors.values():
        _checked_root_path(lock.root, vendor.destination, f"Vendor {vendor.name!r} destination")
        if vendor.patch is not None:
            patch_path = _checked_root_path(
                lock.root, vendor.patch, f"Vendor {vendor.name!r} patch"
            )
            if patch_path.is_symlink() or (patch_path.exists() and not patch_path.is_file()):
                raise VendorError(
                    f"Vendor {vendor.name!r} patch does not name a regular file: {vendor.patch}."
                )
            if not patch_path.exists() and allow_patch_recovery:
                continue
            if not patch_path.is_file():
                raise VendorError(
                    f"Vendor {vendor.name!r} patch does not name a regular file: {vendor.patch}."
                )
            actual_patch_digest = _patch_digest(patch_path.read_bytes())
            if actual_patch_digest != vendor.patch_digest and not allow_patch_recovery:
                raise VendorError(
                    f"Vendor {vendor.name!r} patch digest mismatch: expected "
                    f"{vendor.patch_digest}, got {actual_patch_digest}."
                )
    return lock


def _save_lock(lock: LockFile) -> None:
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "vendors": {name: vendor.to_mapping() for name, vendor in sorted(lock.vendors.items())},
    }
    content = yaml.safe_dump(
        payload,
        sort_keys=False,
        default_flow_style=False,
        allow_unicode=True,
    ).encode("utf-8")
    _atomic_write(lock.path, content)


def _save_lock_checked(lock: LockFile, description: str) -> None:
    try:
        _save_lock(lock)
    except (OSError, yaml.YAMLError) as error:
        raise VendorError(f"Failed to save {description}: {error}") from error


def _checked_root_path(root: Path, relative_path: str, description: str) -> Path:
    target = root.joinpath(*PurePosixPath(relative_path).parts)
    try:
        resolved = target.resolve(strict=False)
    except OSError as error:
        raise VendorError(f"Could not resolve {description}: {error}") from error
    if not resolved.is_relative_to(root.resolve()):
        raise VendorError(f"{description} escapes the consumer root: {relative_path!r}.")
    current = root.resolve()
    for part in PurePosixPath(relative_path).parts:
        current = current / part
        if current.is_symlink():
            raise VendorError(f"{description} traverses a symlink: {relative_path!r}.")
        if not current.exists():
            break
    return target


def _atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        temporary.chmod(0o644)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sync_directory(path: Path) -> None:
    directory_descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(directory_descriptor)
    finally:
        os.close(directory_descriptor)


def _matches(path: str, patterns: Sequence[str]) -> bool:
    pure_path = PurePosixPath(path)
    for pattern in patterns:
        if pure_path.match(pattern) or fnmatch.fnmatchcase(path, pattern):
            return True
        if pattern.startswith("**/") and fnmatch.fnmatchcase(path, pattern[3:]):
            return True
    return False


def _validate_tree_relative_path(path: str) -> str:
    pure_path = PurePosixPath(path)
    if (
        pure_path.is_absolute()
        or any(part in ("", ".", "..", ".git") for part in pure_path.parts)
        or pure_path.as_posix() != path
    ):
        raise VendorError(f"Managed tree contains an unsafe path: {path!r}.")
    return path


def _tree_digest(tree: Mapping[str, FileEntry]) -> str:
    digest = hashlib.sha256()
    digest.update(b"sha256-tree-v1\0")
    for path, entry in sorted(tree.items()):
        encoded_path = path.encode("utf-8")
        digest.update(len(encoded_path).to_bytes(8, "big"))
        digest.update(encoded_path)
        digest.update(b"x" if entry.executable else b"-")
        digest.update(len(entry.data).to_bytes(8, "big"))
        digest.update(entry.data)
    return f"{_DIGEST_PREFIX}{digest.hexdigest()}"


def _read_directory_tree(directory: Path, patterns: Sequence[str]) -> Tree:
    if not directory.exists():
        return {}
    if directory.is_symlink() or not directory.is_dir():
        raise VendorError(f"Managed destination must be a regular directory: {directory}.")
    tree: Tree = {}
    for path in sorted(directory.rglob("*")):
        relative = _validate_tree_relative_path(path.relative_to(directory).as_posix())
        if path.is_symlink():
            raise VendorError(f"Managed trees cannot contain symlinks: {relative}.")
        if path.is_dir():
            continue
        if not path.is_file():
            raise VendorError(f"Managed trees cannot contain special files: {relative}.")
        if not _matches(relative, patterns):
            continue
        mode = path.stat().st_mode
        tree[relative] = FileEntry(path.read_bytes(), bool(mode & stat.S_IXUSR))
    return tree


def _read_all_directory_files(directory: Path, patterns: Sequence[str]) -> Tree:
    tree = _read_directory_tree(directory, patterns)
    for path in sorted(directory.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(directory).as_posix()
        if not _matches(relative, patterns):
            raise VendorError(f"Patch produced a file outside the vendor include set: {relative}.")
    return tree


def _write_tree(directory: Path, tree: Mapping[str, FileEntry]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for relative, entry in tree.items():
        _validate_tree_relative_path(relative)
        target = directory.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(entry.data)
        target.chmod(0o755 if entry.executable else 0o644)


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


@contextlib.contextmanager
def _replace_selected_destination_transaction(
    lock: LockFile, vendor: Vendor, tree: Mapping[str, FileEntry]
) -> Iterator[None]:
    destination = _checked_root_path(
        lock.root, vendor.destination, f"Vendor {vendor.name!r} destination"
    )
    backup = destination.parent / f".{destination.name}.vendor-backup"
    staging: Path | None = None
    destination_existed = False
    replacement_installed = False
    preserve_staging = False
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            _read_directory_tree(destination, vendor.include)
        if backup.exists() or backup.is_symlink():
            raise VendorError(f"Stale vendor backup blocks synchronization: {backup}.")
        staging = Path(
            tempfile.mkdtemp(prefix=f".{destination.name}.vendor-", dir=destination.parent)
        )
        destination_existed = destination.exists()
        if destination_existed:
            shutil.copytree(destination, staging, dirs_exist_ok=True)
        for path in sorted(staging.rglob("*"), reverse=True):
            if path.is_dir():
                with contextlib.suppress(OSError):
                    path.rmdir()
                continue
            relative = path.relative_to(staging).as_posix()
            if _matches(relative, vendor.include):
                path.unlink()
        _write_tree(staging, tree)
        if destination_existed:
            os.replace(destination, backup)
        try:
            os.replace(staging, destination)
        except OSError:
            if backup.exists() and not destination.exists():
                os.replace(backup, destination)
            raise
        replacement_installed = True
        try:
            yield
        except BaseException as triggering_error:
            try:
                if replacement_installed and (destination.exists() or destination.is_symlink()):
                    os.replace(destination, staging)
                if destination_existed and (backup.exists() or backup.is_symlink()):
                    os.replace(backup, destination)
            except OSError as rollback_error:
                preserve_staging = True
                raise VendorError(
                    f"Failed to roll back destination for vendor {vendor.name!r} after "
                    f"{triggering_error}; recovery data remains at {backup} and {staging}: "
                    f"{rollback_error}"
                ) from rollback_error
            raise
        else:
            if backup.exists() or backup.is_symlink():
                _remove_path(backup)
    except OSError as error:
        raise VendorError(
            f"Failed to replace destination for vendor {vendor.name!r}: {error}"
        ) from error
    finally:
        if (
            staging is not None
            and not preserve_staging
            and (staging.exists() or staging.is_symlink())
        ):
            try:
                _remove_path(staging)
            except OSError as error:
                raise VendorError(
                    f"Failed to clean staging directory for vendor {vendor.name!r}: "
                    f"{staging}: {error}"
                ) from error


def _replace_selected_destination(
    lock: LockFile, vendor: Vendor, tree: Mapping[str, FileEntry]
) -> None:
    with _replace_selected_destination_transaction(lock, vendor, tree):
        pass


@contextlib.contextmanager
def _atomic_file_replacement(path: Path, content: bytes) -> Iterator[None]:
    try:
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise VendorError(f"Generated patch path must be a regular file: {path}.")
        existed = path.exists()
        old_content = path.read_bytes() if existed else None
        old_mode = stat.S_IMODE(path.stat().st_mode) if existed else None
        _atomic_write(path, content)
    except VendorError:
        raise
    except OSError as error:
        raise VendorError(f"Failed to write generated patch {path}: {error}") from error
    try:
        yield
    except BaseException:
        try:
            if old_content is None:
                path.unlink(missing_ok=True)
            else:
                _atomic_write(path, old_content)
                assert old_mode is not None
                path.chmod(old_mode)
        except OSError as error:
            raise VendorError(f"Failed to roll back generated patch {path}: {error}") from error
        raise


def _git_environment(*, isolated: bool, repository_ceiling: Path | None = None) -> dict[str, str]:
    environment = os.environ.copy()
    for variable in _GIT_LOCAL_ENVIRONMENT_VARIABLES:
        environment.pop(variable, None)
    environment.update(
        {
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_LITERAL_PATHSPECS": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_HTTP_LOW_SPEED_LIMIT": "1",
            "GIT_HTTP_LOW_SPEED_TIME": "30",
            "GIT_AUTHOR_NAME": "TensorRT-LLM Vendor Tool",
            "GIT_AUTHOR_EMAIL": "vendor-tool@nvidia.com",
            "GIT_COMMITTER_NAME": "TensorRT-LLM Vendor Tool",
            "GIT_COMMITTER_EMAIL": "vendor-tool@nvidia.com",
        }
    )
    if isolated:
        environment["GIT_CONFIG_NOSYSTEM"] = "1"
        environment["GIT_CONFIG_GLOBAL"] = os.devnull
    if repository_ceiling is not None:
        environment["GIT_CEILING_DIRECTORIES"] = str(repository_ceiling.resolve())
    environment.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes -o ConnectTimeout=20")
    return environment


def _run_git(
    arguments: Sequence[str | Path],
    *,
    cwd: Path,
    capture_bytes: bool = False,
    isolated: bool = False,
    input_bytes: bytes | None = None,
    repository_ceiling: Path | None = None,
) -> bytes | str:
    if input_bytes is not None and not capture_bytes:
        raise ValueError("Git commands with byte input must capture byte output.")
    command = ["git", *[str(argument) for argument in arguments]]
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=_git_environment(
                isolated=isolated,
                repository_ceiling=repository_ceiling,
            ),
            check=True,
            capture_output=True,
            input=input_bytes,
            text=not capture_bytes,
            timeout=_GIT_TIMEOUT_SECONDS,
        )
    except FileNotFoundError as error:
        raise VendorError("Git is required for this command but was not found.") from error
    except subprocess.TimeoutExpired as error:
        raise VendorError(
            f"Git command timed out after {_GIT_TIMEOUT_SECONDS} seconds: {' '.join(command)}"
        ) from error
    except subprocess.CalledProcessError as error:
        stderr_value = error.stderr
        if isinstance(stderr_value, bytes):
            stderr_text = stderr_value.decode("utf-8", errors="replace")
        else:
            stderr_text = stderr_value or ""
        raise VendorError(
            f"Git command failed: {' '.join(command)}\n{stderr_text.strip()}"
        ) from error
    return result.stdout


def _read_git_blobs(repo: Path, object_ids: Sequence[bytes], source: str) -> list[bytes]:
    batch_input = b"".join(object_id + b"\n" for object_id in object_ids)
    output = _run_git(
        ["-C", repo, "cat-file", "--batch"],
        cwd=repo,
        capture_bytes=True,
        input_bytes=batch_input,
    )
    assert isinstance(output, bytes)
    blobs: list[bytes] = []
    position = 0
    for expected_object_id in object_ids:
        header_end = output.find(b"\n", position)
        if header_end < 0:
            raise VendorError(f"Git produced truncated blob metadata for source {source!r}.")
        header = output[position:header_end]
        fields = header.split(b" ")
        if len(fields) != 3:
            raise VendorError(f"Git produced invalid blob metadata for source {source!r}.")
        object_id, object_type, raw_size = fields
        try:
            size = int(raw_size)
        except ValueError as error:
            raise VendorError(
                f"Git produced an invalid blob size for source {source!r}."
            ) from error
        if object_id != expected_object_id or object_type != b"blob" or size < 0:
            raise VendorError(f"Git returned an unexpected object for source {source!r}.")
        data_start = header_end + 1
        data_end = data_start + size
        if data_end >= len(output) or output[data_end : data_end + 1] != b"\n":
            raise VendorError(f"Git produced truncated blob data for source {source!r}.")
        blobs.append(output[data_start:data_end])
        position = data_end + 1
    if position != len(output):
        raise VendorError(f"Git produced trailing blob data for source {source!r}.")
    return blobs


def _archive_tree(repo: Path, commit: str, source: str, patterns: Sequence[str]) -> Tree:
    try:
        listing = _run_git(
            ["-C", repo, "ls-tree", "-r", "-z", "--full-tree", commit, "--", source],
            cwd=repo,
            capture_bytes=True,
        )
    except VendorError as error:
        raise VendorError(
            f"Could not read source {source!r} at commit {commit}: {error}"
        ) from error
    assert isinstance(listing, bytes)
    if listing and not listing.endswith(b"\0"):
        raise VendorError(f"Git produced an invalid tree listing for source {source!r}.")

    source_path = PurePosixPath(source)
    selected: list[tuple[str, bytes, bool]] = []
    seen_paths: set[str] = set()
    for record in listing.split(b"\0")[:-1]:
        metadata, separator, raw_path = record.partition(b"\t")
        fields = metadata.split(b" ")
        if not separator or len(fields) != 3:
            raise VendorError(f"Git produced an invalid tree entry for source {source!r}.")
        raw_mode, object_type, object_id = fields
        try:
            path = raw_path.decode("utf-8")
            mode = raw_mode.decode("ascii")
        except UnicodeError as error:
            raise VendorError(
                f"Git tree paths and modes must be valid UTF-8 for {source!r}."
            ) from error
        _validate_tree_relative_path(path)
        try:
            relative = PurePosixPath(path).relative_to(source_path).as_posix()
        except ValueError as error:
            raise VendorError(f"Git tree entry escaped source {source!r}: {path!r}.") from error
        if relative in ("", "."):
            continue
        _validate_tree_relative_path(relative)
        if object_type != b"blob" or mode not in ("100644", "100755"):
            raise VendorError(
                f"Selected upstream trees cannot contain links or special files: {relative}."
            )
        if not _matches(relative, patterns):
            continue
        if relative in seen_paths:
            raise VendorError(
                f"Git produced a duplicate tree entry for source {source!r}: {relative}."
            )
        if re.fullmatch(rb"[0-9a-f]{40}|[0-9a-f]{64}", object_id) is None:
            raise VendorError(f"Git produced an invalid object ID for source {source!r}.")
        seen_paths.add(relative)
        selected.append((relative, object_id, mode == "100755"))

    blobs = _read_git_blobs(repo, [object_id for _, object_id, _ in selected], source)
    tree = {
        relative: FileEntry(data=data, executable=executable)
        for (relative, _, executable), data in zip(selected, blobs, strict=True)
    }
    if not tree:
        raise VendorError(f"Source {source!r} at {commit} selected no files.")
    return tree


def _local_source_tree(vendor: Vendor, repo: Path) -> Tree:
    repository = repo.resolve()
    if not repository.is_dir():
        raise VendorError(f"Local upstream repository does not exist: {repository}.")
    try:
        _run_git(
            ["-C", repository, "cat-file", "-e", f"{vendor.commit}^{{commit}}"], cwd=repository
        )
    except VendorError as error:
        raise VendorError(
            f"Local repository {repository} does not contain locked commit {vendor.commit}."
        ) from error
    return _archive_tree(repository, vendor.commit, vendor.source, vendor.include)


@contextlib.contextmanager
def _remote_repository(vendor: Vendor) -> Iterator[Path]:
    with tempfile.TemporaryDirectory(prefix="trtllm-vendor-fetch-") as temporary_directory:
        repository = Path(temporary_directory)
        try:
            _run_git(["init", "--quiet"], cwd=repository)
            _run_git(["remote", "add", "origin", vendor.url], cwd=repository)
            _run_git(["ls-remote", "--quiet", "origin"], cwd=repository)
        except VendorError as error:
            raise UpstreamUnavailable(
                f"Upstream unavailable for {vendor.name!r} at {vendor.url}: {error}"
            ) from error
        try:
            try:
                _run_git(
                    ["fetch", "--quiet", "--depth", "1", "origin", vendor.commit], cwd=repository
                )
            except VendorError:
                ref = None
                if vendor.branch is not None:
                    ref = f"refs/heads/{vendor.branch}"
                elif vendor.tag is not None:
                    ref = f"refs/tags/{vendor.tag}"
                if ref is None:
                    raise
                _run_git(["fetch", "--quiet", "origin", ref], cwd=repository)
            _run_git(["cat-file", "-e", f"{vendor.commit}^{{commit}}"], cwd=repository)
        except VendorError as error:
            raise VendorError(
                f"Upstream {vendor.url} is reachable but locked commit {vendor.commit} "
                f"could not be fetched: {error}"
            ) from error
        yield repository


def _source_tree(vendor: Vendor, repo: Path | None) -> Tree:
    if repo is not None:
        return _local_source_tree(vendor, repo)
    with _remote_repository(vendor) as remote:
        return _archive_tree(remote, vendor.commit, vendor.source, vendor.include)


def _store_git_tree(repository: Path, tree: Mapping[str, FileEntry]) -> str:
    _run_git(["read-tree", "--empty"], cwd=repository, isolated=True)
    index_entries = bytearray()
    for relative, entry in sorted(tree.items()):
        _validate_tree_relative_path(relative)
        object_id = _run_git(
            ["hash-object", "-w", "--stdin", "--no-filters"],
            cwd=repository,
            capture_bytes=True,
            isolated=True,
            input_bytes=entry.data,
        ).strip()
        assert isinstance(object_id, bytes)
        if re.fullmatch(rb"[0-9a-f]{40}", object_id) is None:
            raise VendorError(f"Git produced an invalid blob object ID for {relative!r}.")
        mode = b"100755" if entry.executable else b"100644"
        index_entries.extend(mode + b" " + object_id + b"\t" + relative.encode("utf-8") + b"\0")
    if index_entries:
        _run_git(
            ["update-index", "-z", "--index-info"],
            cwd=repository,
            capture_bytes=True,
            isolated=True,
            input_bytes=bytes(index_entries),
        )
    object_id = _run_git(["write-tree"], cwd=repository, isolated=True)
    assert isinstance(object_id, str)
    object_id = object_id.strip()
    if re.fullmatch(r"[0-9a-f]{40}", object_id) is None:
        raise VendorError("Git produced an invalid tree object ID.")
    return object_id


def _generate_patch(
    baseline: Mapping[str, FileEntry], result_tree: Mapping[str, FileEntry]
) -> bytes:
    with tempfile.TemporaryDirectory(prefix="trtllm-vendor-patch-") as temporary_directory:
        repository = Path(temporary_directory)
        _run_git(["init", "--quiet", "--object-format=sha1"], cwd=repository, isolated=True)
        baseline_object = _store_git_tree(repository, baseline)
        result_object = _store_git_tree(repository, result_tree)
        # Keep selected attributes out of Git's index fallback during the tree diff.
        _run_git(["read-tree", "--empty"], cwd=repository, isolated=True)
        patch = _run_git(
            [
                "-c",
                f"core.attributesFile={os.devnull}",
                "diff-tree",
                "-p",
                "-r",
                "--binary",
                "--full-index",
                "--no-renames",
                "--no-ext-diff",
                "--no-textconv",
                "--unified=3",
                "--diff-algorithm=myers",
                "--no-indent-heuristic",
                baseline_object,
                result_object,
            ],
            cwd=repository,
            capture_bytes=True,
            isolated=True,
        )
        assert isinstance(patch, bytes)
    if patch:
        reproduced = _apply_patch_bytes(baseline, patch, ("**/*",))
    else:
        reproduced = dict(baseline)
    expected = dict(result_tree)
    if reproduced != expected:
        differing = sorted(
            path
            for path in set(reproduced) | set(expected)
            if reproduced.get(path) != expected.get(path)
        )
        raise VendorError(
            f"Generated vendor patch does not exactly reproduce its result tree: {differing}."
        )
    return patch


def _apply_patch(
    baseline: Mapping[str, FileEntry],
    patch_path: Path,
    patterns: Sequence[str],
) -> Tree:
    with tempfile.TemporaryDirectory(prefix="trtllm-vendor-apply-") as temporary_directory:
        directory = Path(temporary_directory)
        _write_tree(directory, baseline)
        try:
            # Stop before Git can inspect a parent that belongs to an enclosing worktree.
            _run_git(
                ["apply", "--check", "--binary", patch_path],
                cwd=directory,
                isolated=True,
                repository_ceiling=directory.parent,
            )
            _run_git(
                ["apply", "--binary", patch_path],
                cwd=directory,
                isolated=True,
                repository_ceiling=directory.parent,
            )
        except VendorError as error:
            raise VendorError(f"Failed to apply vendor patch {patch_path}: {error}") from error
        return _read_all_directory_files(directory, patterns)


def _normal_tree(lock: LockFile, vendor: Vendor, repo: Path | None) -> Tree:
    tree = _source_tree(vendor, repo)
    if vendor.patch is None:
        return tree
    patch_path = _checked_root_path(lock.root, vendor.patch, f"Vendor {vendor.name!r} patch")
    return _apply_patch(tree, patch_path, vendor.include)


def _current_tree(lock: LockFile, vendor: Vendor) -> Tree:
    destination = _checked_root_path(
        lock.root, vendor.destination, f"Vendor {vendor.name!r} destination"
    )
    return _read_directory_tree(destination, vendor.include)


def _changed_paths(
    baseline: Mapping[str, FileEntry], result_tree: Mapping[str, FileEntry]
) -> list[str]:
    return sorted(
        path
        for path in set(baseline) | set(result_tree)
        if baseline.get(path) != result_tree.get(path)
    )


def _select_vendors(lock: LockFile, name: str | None) -> list[Vendor]:
    if name is None:
        return [lock.vendors[key] for key in sorted(lock.vendors)]
    try:
        return [lock.vendors[name]]
    except KeyError as error:
        raise VendorError(f"Unknown vendor: {name!r}.") from error


def _require_vendor(lock: LockFile, name: str) -> Vendor:
    return _select_vendors(lock, name)[0]


def _verify_offline(lock: LockFile, vendor: Vendor) -> None:
    actual = _tree_digest(_current_tree(lock, vendor))
    if actual != vendor.digest:
        raise VendorError(
            f"Vendor {vendor.name!r} destination digest mismatch: "
            f"expected {vendor.digest}, got {actual}."
        )


def _verify_source(lock: LockFile, vendor: Vendor, repo: Path | None) -> None:
    normal = _normal_tree(lock, vendor, repo)
    normal_digest = _tree_digest(normal)
    if normal_digest != vendor.digest:
        raise VendorError(
            f"Vendor {vendor.name!r} locked materialization digest mismatch: "
            f"expected {vendor.digest}, got {normal_digest}."
        )
    current = _current_tree(lock, vendor)
    if current != normal:
        raise VendorError(
            f"Vendor {vendor.name!r} destination differs from its upstream materialization."
        )


def _command_list(args: argparse.Namespace) -> None:
    lock = _load_lock(args.lock)
    for vendor in _select_vendors(lock, None):
        state = "patched" if vendor.patch else "exact"
        print(f"{vendor.name}\t{state}\t{vendor.commit}\t{vendor.destination}")


def _command_status(args: argparse.Namespace) -> None:
    lock = _load_lock(args.lock)
    failed = False
    for vendor in _select_vendors(lock, args.name):
        try:
            _verify_offline(lock, vendor)
        except VendorError as error:
            failed = True
            print(f"FAIL-LOCAL-INTEGRITY {vendor.name}: {error}")
        else:
            state = "patched" if vendor.patch is not None else "exact"
            print(f"PASS-OFFLINE {vendor.name}: {state} at {vendor.commit}")
    if failed:
        raise VendorError("One or more vendor snapshots failed offline integrity checks.")


def _new_vendor_from_args(args: argparse.Namespace) -> Vendor:
    include = args.include if args.include is not None else ["**/*"]
    value: dict[str, object] = {
        "url": args.url,
        "commit": args.commit,
        "source": args.source,
        "destination": args.destination,
        "include": include,
        "digest": f"{_DIGEST_PREFIX}{'0' * 64}",
    }
    if args.branch is not None:
        value["branch"] = args.branch
    if args.tag is not None:
        value["tag"] = args.tag
    return _validate_vendor(args.name, value)


def _command_create(args: argparse.Namespace) -> None:
    lock = _load_lock(args.lock, allow_missing=True)
    if args.name in lock.vendors:
        raise VendorError(f"Vendor {args.name!r} already exists.")
    vendor = _new_vendor_from_args(args)
    lock.vendors[vendor.name] = vendor
    _validate_destination_layout(lock)
    destination = _checked_root_path(
        lock.root, vendor.destination, f"Vendor {vendor.name!r} destination"
    )
    if args.adopt is None and destination.exists():
        raise VendorError(
            f"Vendor {vendor.name!r} destination already exists; use --adopt exact or patched."
        )
    source_tree = _source_tree(vendor, args.repo)

    destination_transaction: contextlib.AbstractContextManager[None] = contextlib.nullcontext()
    patch_transaction: contextlib.AbstractContextManager[None] = contextlib.nullcontext()
    if args.adopt is None:
        result_tree = source_tree
        destination_transaction = _replace_selected_destination_transaction(
            lock, vendor, result_tree
        )
    else:
        if not destination.exists():
            raise VendorError(f"Cannot adopt missing destination for vendor {vendor.name!r}.")
        result_tree = _current_tree(lock, vendor)
        if args.adopt == "exact":
            if result_tree != source_tree:
                changed = _changed_paths(source_tree, result_tree)
                raise VendorError(
                    f"Vendor {vendor.name!r} is not exact; differing files: {changed}. "
                    "Use --adopt patched to record the difference."
                )
        else:
            patch = _generate_patch(source_tree, result_tree)
            if patch:
                patch_relative = f"3rdparty/vendor_patches/{vendor.name}.patch"
                patch_path = _checked_root_path(
                    lock.root, patch_relative, f"Vendor {vendor.name!r} patch"
                )
                vendor.patch = patch_relative
                vendor.patch_digest = _patch_digest(patch)
                patch_transaction = _atomic_file_replacement(patch_path, patch)
    vendor.digest = _tree_digest(result_tree)
    with contextlib.ExitStack() as transaction:
        transaction.enter_context(destination_transaction)
        transaction.enter_context(patch_transaction)
        _save_lock_checked(lock, f"lock state for new vendor {vendor.name!r}")
    print(f"Created {vendor.name} at {vendor.commit} ({'patched' if vendor.patch else 'exact'}).")


def _command_sync(args: argparse.Namespace) -> None:
    lock = _load_lock(args.lock)
    vendor = _require_vendor(lock, args.name)
    normal = _normal_tree(lock, vendor, args.repo)
    digest = _tree_digest(normal)
    if digest != vendor.digest:
        raise VendorError(
            f"Vendor {vendor.name!r} materializes to {digest}, but the lock records {vendor.digest}. "
            "Refresh the patch or pin instead of silently changing the accepted digest."
        )
    _replace_selected_destination(lock, vendor, normal)
    print(f"Synchronized {vendor.name} to {vendor.commit}.")


def _command_check(args: argparse.Namespace) -> None:
    if args.require_access and not args.upstream:
        raise VendorError("--require-access requires --upstream.")
    if args.repo is not None and args.name is None:
        raise VendorError("--repo requires one vendor name.")
    if args.repo is not None and args.upstream:
        raise VendorError("Use either --repo or --upstream, not both.")
    if args.repo is not None and args.offline:
        raise VendorError("Use either --repo or --offline, not both.")
    lock = _load_lock(args.lock)
    vendors = _select_vendors(lock, args.name)
    for vendor in vendors:
        _verify_offline(lock, vendor)
        print(f"PASS-OFFLINE {vendor.name}: local snapshot verified; upstream not verified.")

    if args.repo is not None:
        vendor = vendors[0]
        _verify_source(lock, vendor, args.repo)
        print(f"PASS-LOCAL {vendor.name}: verified from {args.repo}.")
        return
    if not args.upstream:
        return
    for vendor in vendors:
        try:
            _verify_source(lock, vendor, None)
        except UpstreamUnavailable as error:
            if args.require_access:
                raise VendorError(f"REMOTE-UNAVAILABLE {vendor.name}: {error}") from error
            print(f"SKIP-REMOTE-UNAVAILABLE {vendor.name}: {error}", file=sys.stderr)
            continue
        print(f"PASS-REMOTE {vendor.name}: verified from {vendor.url}.")


def _update_patch(lock: LockFile, vendor: Vendor, repo: Path | None, *, create: bool) -> None:
    if create and vendor.patch is not None:
        raise VendorError(f"Vendor {vendor.name!r} already has a patch; use refresh.")
    if not create and vendor.patch is None:
        raise VendorError(f"Vendor {vendor.name!r} has no patch; use create.")
    raw = _source_tree(vendor, repo)
    current = _current_tree(lock, vendor)
    patch = _generate_patch(raw, current)
    old_patch = vendor.patch
    old_patch_path = (
        None
        if old_patch is None
        else _checked_root_path(lock.root, old_patch, f"Vendor {vendor.name!r} patch")
    )
    old_patch_content = (
        old_patch_path.read_bytes()
        if old_patch_path is not None and old_patch_path.is_file()
        else None
    )
    written_patch_path: Path | None = None
    if patch:
        patch_relative = old_patch or f"3rdparty/vendor_patches/{vendor.name}.patch"
        patch_path = _checked_root_path(lock.root, patch_relative, f"Vendor {vendor.name!r} patch")
        _atomic_write(patch_path, patch)
        written_patch_path = patch_path
        vendor.patch = patch_relative
        vendor.patch_digest = _patch_digest(patch)
    else:
        vendor.patch = None
        vendor.patch_digest = None
    vendor.digest = _tree_digest(current)
    try:
        _save_lock(lock)
    except (OSError, yaml.YAMLError) as error:
        if old_patch_path is not None and old_patch_content is not None:
            _atomic_write(old_patch_path, old_patch_content)
        elif written_patch_path is not None:
            written_patch_path.unlink(missing_ok=True)
        raise VendorError(f"Failed to save refreshed patch state: {error}") from error
    if not patch and old_patch_path is not None:
        old_patch_path.unlink(missing_ok=True)
    print(f"{'Created' if create else 'Refreshed'} patch for {vendor.name}.")


def _command_patch(args: argparse.Namespace) -> None:
    lock = _load_lock(args.lock, allow_patch_recovery=True)
    vendor = _require_vendor(lock, args.name)
    if args.action == "create":
        _update_patch(lock, vendor, args.repo, create=True)
        return
    if args.action == "refresh":
        _update_patch(lock, vendor, args.repo, create=False)
        return
    if vendor.patch is None:
        raise VendorError(f"Vendor {vendor.name!r} has no patch to drop.")
    raw = _source_tree(vendor, args.repo)
    current = _current_tree(lock, vendor)
    if current != raw:
        raise VendorError(
            f"Vendor {vendor.name!r} destination is not exact upstream; refusing to drop its patch."
        )
    patch_path = _checked_root_path(lock.root, vendor.patch, f"Vendor {vendor.name!r} patch")
    vendor.patch = None
    vendor.patch_digest = None
    vendor.digest = _tree_digest(raw)
    _save_lock(lock)
    patch_path.unlink(missing_ok=True)
    print(f"Dropped patch for {vendor.name}.")


def _worktree_head(repo: Path) -> str:
    result = _run_git(["-C", repo, "rev-parse", "HEAD"], cwd=repo)
    assert isinstance(result, str)
    return result.strip()


def _require_clean_source_worktree(repo: Path, source: str, patterns: Sequence[str]) -> None:
    result = _run_git(
        ["-C", repo, "status", "--porcelain", "--untracked-files=all", "--", source],
        cwd=repo,
    )
    assert isinstance(result, str)
    if result.strip():
        raise VendorError(f"Local upstream source {source!r} has uncommitted changes.")
    ignored = _run_git(
        [
            "-C",
            repo,
            "ls-files",
            "-z",
            "--others",
            "--ignored",
            "--exclude-standard",
            "--",
            source,
        ],
        cwd=repo,
        capture_bytes=True,
    )
    assert isinstance(ignored, bytes)
    if ignored and not ignored.endswith(b"\0"):
        raise VendorError(f"Git produced an invalid ignored-file listing for source {source!r}.")
    source_path = PurePosixPath(source)
    selected: list[str] = []
    for raw_path in ignored.split(b"\0")[:-1]:
        try:
            path = raw_path.decode("utf-8")
        except UnicodeError as error:
            raise VendorError(
                f"Ignored paths under local upstream source {source!r} must be valid UTF-8."
            ) from error
        _validate_tree_relative_path(path)
        try:
            relative = PurePosixPath(path).relative_to(source_path).as_posix()
        except ValueError as error:
            raise VendorError(
                f"Git ignored-file entry escaped source {source!r}: {path!r}."
            ) from error
        if relative not in ("", ".") and _matches(relative, patterns):
            selected.append(relative)
    if selected:
        raise VendorError(
            f"Local upstream source {source!r} has ignored untracked files selected "
            f"for export: {selected}."
        )


def _apply_patch_bytes(
    baseline: Mapping[str, FileEntry],
    patch: bytes,
    patterns: Sequence[str],
) -> Tree:
    with tempfile.TemporaryDirectory(prefix="trtllm-vendor-delta-") as temporary_directory:
        patch_path = Path(temporary_directory) / "delta.patch"
        patch_path.write_bytes(patch)
        return _apply_patch(baseline, patch_path, patterns)


def _command_export(args: argparse.Namespace) -> None:
    lock = _load_lock(args.lock)
    vendor = _require_vendor(lock, args.name)
    repository = args.repo.resolve()
    if _worktree_head(repository) != vendor.commit:
        raise VendorError(
            f"Local upstream HEAD must equal locked commit {vendor.commit} before export."
        )
    _require_clean_source_worktree(repository, vendor.source, vendor.include)
    raw = _local_source_tree(vendor, repository)
    normal = raw
    if vendor.patch is not None:
        patch_path = _checked_root_path(lock.root, vendor.patch, f"Vendor {vendor.name!r} patch")
        normal = _apply_patch(raw, patch_path, vendor.include)
    if _tree_digest(normal) != vendor.digest:
        raise VendorError(
            f"Vendor {vendor.name!r} normal materialization does not match its digest."
        )
    current = _current_tree(lock, vendor)
    delta = _generate_patch(normal, current)
    if not delta:
        raise VendorError(f"Vendor {vendor.name!r} has no downstream change to export.")
    try:
        exported = _apply_patch_bytes(raw, delta, vendor.include)
    except VendorError as error:
        raise VendorError(
            "The local change overlaps the vendor compatibility patch and cannot be exported "
            f"automatically: {error}"
        ) from error
    export_vendor = dataclasses.replace(vendor, destination=vendor.source)
    export_lock = LockFile(path=lock.path, root=repository, vendors={vendor.name: export_vendor})
    _replace_selected_destination(export_lock, export_vendor, exported)
    print(
        f"Exported {vendor.name} changes to {repository / vendor.source}; review and commit them upstream."
    )


def _command_pin(args: argparse.Namespace) -> None:
    lock = _load_lock(args.lock)
    vendor = _require_vendor(lock, args.name)
    repository = args.repo.resolve()
    commit = args.commit or _worktree_head(repository)
    value = vendor.to_mapping()
    value["url"] = args.url or vendor.url
    value["commit"] = commit
    if args.branch is not None:
        value["branch"] = args.branch
        value.pop("tag", None)
    elif args.tag is not None:
        value["tag"] = args.tag
        value.pop("branch", None)
    value["digest"] = vendor.digest
    proposed = _validate_vendor(vendor.name, value)
    raw = _local_source_tree(proposed, repository)
    current = _current_tree(lock, vendor)
    drop_patch = False
    if proposed.patch is None:
        materialized = raw
    else:
        patch_path = _checked_root_path(lock.root, proposed.patch, f"Vendor {vendor.name!r} patch")
        try:
            materialized = _apply_patch(raw, patch_path, proposed.include)
        except VendorError:
            if raw != current:
                raise
            materialized = raw
            drop_patch = True
    if materialized != current:
        if proposed.patch is not None and raw == current:
            materialized = raw
            drop_patch = True
        else:
            changed = _changed_paths(materialized, current)
            raise VendorError(
                f"New pin does not reproduce the destination for {vendor.name!r}; "
                f"differing files: {changed}."
            )
    old_patch = proposed.patch
    if drop_patch:
        proposed.patch = None
        proposed.patch_digest = None
    proposed.digest = _tree_digest(materialized)
    lock.vendors[vendor.name] = proposed
    patch_path: Path | None = None
    if drop_patch and old_patch is not None:
        patch_path = _checked_root_path(lock.root, old_patch, f"Vendor {vendor.name!r} patch")
        if patch_path.is_symlink() or not patch_path.is_file():
            raise VendorError(f"Generated patch path must be a regular file: {patch_path}.")
    _save_lock_checked(lock, f"new pin for vendor {vendor.name!r}")
    if patch_path is not None:
        try:
            _sync_directory(lock.path.parent)
        except OSError as error:
            raise VendorError(
                f"Failed to make the new pin durable for vendor {vendor.name!r}: {error}"
            ) from error
        try:
            patch_path.unlink()
        except OSError as error:
            print(
                f"warning: Pinned vendor {vendor.name!r}, but could not remove its absorbed "
                f"patch {patch_path}: {error}. The lock is committed and valid; this patch is "
                "now an unreferenced orphan. Delete that file manually.",
                file=sys.stderr,
            )
    print(f"Pinned {vendor.name} to {commit}; offline enforcement restored.")


def _command_remove(args: argparse.Namespace) -> None:
    lock = _load_lock(args.lock)
    vendor = _require_vendor(lock, args.name)
    del lock.vendors[vendor.name]
    _save_lock(lock)
    if vendor.patch is not None and all(
        other.patch != vendor.patch for other in lock.vendors.values()
    ):
        _checked_root_path(lock.root, vendor.patch, f"Vendor {vendor.name!r} patch").unlink(
            missing_ok=True
        )
    print(f"Removed {vendor.name} from the lock; destination bytes were preserved.")


def _add_repo_argument(parser: argparse.ArgumentParser, *, required: bool = False) -> None:
    parser.add_argument(
        "--repo",
        type=Path,
        required=required,
        help="Local Git checkout containing the locked commit; never contacts the lock URL.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--lock",
        type=Path,
        default=_DEFAULT_LOCK,
        help=f"Vendor lock path (default: {_DEFAULT_LOCK.relative_to(_REPO_ROOT)}).",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser("list", help="List configured vendors.")
    list_parser.set_defaults(handler=_command_list)

    status = commands.add_parser("status", help="Show offline vendor status.")
    status.add_argument("name", nargs="?")
    status.set_defaults(handler=_command_status)

    create = commands.add_parser("create", help="Create or adopt a vendor entry.")
    create.add_argument("name")
    create.add_argument("--url", required=True)
    reference = create.add_mutually_exclusive_group()
    reference.add_argument("--branch")
    reference.add_argument("--tag")
    create.add_argument("--commit", required=True)
    create.add_argument("--source", required=True)
    create.add_argument("--destination", required=True)
    create.add_argument("--include", action="append")
    create.add_argument("--adopt", choices=("exact", "patched"))
    _add_repo_argument(create)
    create.set_defaults(handler=_command_create)

    sync = commands.add_parser("sync", help="Restore a locked vendor materialization.")
    sync.add_argument("name")
    _add_repo_argument(sync)
    sync.set_defaults(handler=_command_sync)

    check = commands.add_parser("check", help="Verify vendor integrity.")
    check.add_argument("name", nargs="?")
    check_mode = check.add_mutually_exclusive_group()
    check_mode.add_argument(
        "--offline",
        action="store_true",
        help="Explicitly select the default no-network integrity check.",
    )
    check_mode.add_argument(
        "--upstream", action="store_true", help="Attempt upstream verification."
    )
    check.add_argument(
        "--require-access",
        action="store_true",
        help="Fail when an upstream is inaccessible; for trusted maintainer environments.",
    )
    _add_repo_argument(check)
    check.set_defaults(handler=_command_check)

    patch = commands.add_parser("patch", help="Create, refresh, or drop a downstream patch.")
    patch.add_argument("name")
    patch.add_argument("action", choices=("create", "refresh", "drop"))
    _add_repo_argument(patch)
    patch.set_defaults(handler=_command_patch)

    export = commands.add_parser("export", help="Export pending destination changes upstream.")
    export.add_argument("name")
    _add_repo_argument(export, required=True)
    export.set_defaults(handler=_command_export)

    pin = commands.add_parser("pin", help="Pin a vendor to a new committed source revision.")
    pin.add_argument("name")
    pin.add_argument("--url")
    pin_reference = pin.add_mutually_exclusive_group()
    pin_reference.add_argument("--branch")
    pin_reference.add_argument("--tag")
    pin.add_argument("--commit")
    _add_repo_argument(pin, required=True)
    pin.set_defaults(handler=_command_pin)

    remove = commands.add_parser(
        "remove", help="Remove a vendor lock entry, preserving its destination."
    )
    remove.add_argument("name")
    remove.set_defaults(handler=_command_remove)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the vendoring command-line interface."""
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.lock != _DEFAULT_LOCK:
        args.lock = args.lock.resolve()
    try:
        args.handler(args)
    except VendorError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
