from __future__ import annotations

import hashlib
import re
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests
import yaml

from .models import DependencyMetadata, DependencyMetadataEntry, FilesToDependency

HASH_FS_DIR = "cas"
FILES_TO_DEPENDENCY_YML = "files_to_dependency.yml"
DEPENDENCY_METADATA_YML = "dependency_metadata.yml"
CHECKSUM_TO_PATHS_YML = "checksum_to_paths.yml"
COPYRIGHT_TEXT_RE = re.compile(r"copyright\s*(\(c\)|©)?\s*\d{4}", re.IGNORECASE)
SPDX_COPYRIGHT_RE = re.compile(
    r"^\s*(?:[#/*;\-]{1,2}\s*)?SPDX-FileCopyrightText:\s*(.+)$", re.MULTILINE
)
# Traditional copyright notices like "Copyright (c) 2016-2017 ZeroMQ community"
# Captures full notice including "Copyright (c)", excluding trailing comment markers
TRADITIONAL_COPYRIGHT_RE = re.compile(
    r"^\s*[/*#;\-]*\s*(Copyright\s*(?:\(c\)|©)?\s*\d{4}(?:\s*[-–]\s*\d{4})?\s+[^/*#\n]+)",
    re.IGNORECASE | re.MULTILINE,
)
COMMIT_HASH_RE = re.compile(r"[a-f0-9]{40}")
# Dependencies exempt from copyright notice requirement (use NVIDIA EULA)
COPYRIGHT_EXEMPT_DEPS = {"cuda", "tensorrt"}


def compute_cas_address(content: bytes) -> str:
    # Use blake2b for reproducible CAS address; prefix like example
    digest = hashlib.blake2b(content, digest_size=16).hexdigest()
    return digest


def _read_and_hash_file(
    file_path_str: str,
) -> tuple[str, str, int, list[str]] | tuple[str, None, None, None]:
    """Read a file, compute its checksum, and extract copyright notices.

    Used by ProcessPoolExecutor. Takes string path instead of Path object
    to reduce pickle overhead.

    Returns:
        Tuple of (path_str, checksum, size, copyrights) on success,
        or (path_str, None, None, None) on error.
    """
    try:
        content = Path(file_path_str).read_bytes()
        checksum = compute_cas_address(content)
        copyrights = _extract_file_copyrights(content.decode("utf-8", errors="ignore"))
        return file_path_str, checksum, len(content), copyrights
    except (OSError, IOError):
        return file_path_str, None, None, None


def _ensure_dirs(data_dir: Path) -> None:
    (data_dir / HASH_FS_DIR).mkdir(parents=True, exist_ok=True)


def _cas_put(data_dir: Path, content: str) -> str:
    _ensure_dirs(data_dir)
    # Preserve leading whitespace, only strip trailing whitespace
    content = content.rstrip()
    content_bytes = content.encode("utf-8")
    addr = compute_cas_address(content_bytes)
    content_bytes += b"\n"
    shard = addr[0:2]
    dest_dir = data_dir / HASH_FS_DIR / shard
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_file = dest_dir / addr
    if not dest_file.exists():
        dest_file.write_bytes(content_bytes)
    return addr


def _cas_get(data_dir: Path, address: str) -> str | None:
    shard = address[0:2]
    path = data_dir / HASH_FS_DIR / shard / address
    if path.exists():
        # Preserve leading whitespace, only strip trailing (we add a newline on write)
        return path.read_text(encoding="utf-8").rstrip()
    return None


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def _dump_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=True)


def _extract_file_copyrights(text: str) -> list[str]:
    """Extract copyright notices from file text.

    Supports both SPDX format (SPDX-FileCopyrightText: ...) and traditional
    format (Copyright (c) YEAR Name).
    """
    cleaned: list[str] = []
    # Extract SPDX-formatted copyright notices
    for m in SPDX_COPYRIGHT_RE.findall(text):
        line = m.strip()
        if line:
            cleaned.append(line)
    # Extract traditional copyright notices (e.g., "Copyright (c) 2016-2017 ZeroMQ community")
    for m in TRADITIONAL_COPYRIGHT_RE.findall(text):
        line = m.strip()
        if line:
            cleaned.append(line)
    return cleaned


@dataclass
class ValidationResult:
    ok: bool
    problems: list[str]


class DependencyDatabase:
    def __init__(self, data_dir: Path) -> None:
        self.data_dir = data_dir
        self.meta_path = data_dir / DEPENDENCY_METADATA_YML
        self.f2d_path = data_dir / FILES_TO_DEPENDENCY_YML
        self.checksum_to_paths_path = data_dir / CHECKSUM_TO_PATHS_YML
        self.meta_raw = _load_yaml(self.meta_path)
        self.f2d_raw = _load_yaml(self.f2d_path)
        self.checksum_to_paths: dict[str, list[str]] = _load_yaml(self.checksum_to_paths_path)
        self.file_address_cache: dict[Path, str] = {}

    def cas_put(self, content: str) -> str:
        return _cas_put(self.data_dir, content)

    def cas_get(self, address: str) -> str | None:
        return _cas_get(self.data_dir, address)

    def add_dependency_metadata(
        self, key: str, metadata: DependencyMetadataEntry, overwrite: bool = False
    ) -> None:
        if key in self.meta_raw and not overwrite:
            raise ValueError(f"Dependency metadata exists: {key}. Use --overwrite to update.")
        problems = self._validate_metadata_entry(key, metadata)
        if problems:
            raise ValueError("; ".join(problems))
        self.meta_raw[key] = metadata.model_dump(exclude_none=True, mode="json")
        _dump_yaml(self.meta_path, self.meta_raw)

    def register_dependency_files(self, key: str, files: Iterable[Path]) -> None:
        files_list = list(files)  # Consume iterator once
        current = set(self.f2d_raw.get(key, []) or [])
        dep_name = key.split("/")[0]
        # Build set of checksums already mapped to same dependency (any version)
        already_mapped: set[str] = set()
        for existing_key, existing_files in self.f2d_raw.items():
            if existing_key.split("/")[0] == dep_name:
                already_mapped.update(existing_files or [])

        new_files: list[str] = []
        checksums_with_copyrights: set[str] = set()
        for p in files_list:
            content_bytes = Path(p).read_bytes()
            checksum = compute_cas_address(content_bytes)
            self.file_address_cache[p] = checksum
            # Skip if already mapped to same dependency (possibly different version)
            if checksum in already_mapped:
                continue
            new_files.append(checksum)
            # Store checksum -> path mapping for better error messages
            if checksum not in self.checksum_to_paths:
                self.checksum_to_paths[checksum] = []
            path_str = str(p)
            if path_str not in self.checksum_to_paths[checksum]:
                self.checksum_to_paths[checksum].append(path_str)
            # Track which files have copyright notices (for validation)
            notices = _extract_file_copyrights(content_bytes.decode("utf-8", errors="ignore"))
            if notices:
                checksums_with_copyrights.add(checksum)

        problems = self._validate_file_mappings(
            key, new_files, checksums_with_copyrights=checksums_with_copyrights
        )
        if problems:
            raise ValueError("; ".join(problems))

        for fp in new_files:
            current.add(fp)
        self.f2d_raw[key] = sorted(current)

        _dump_yaml(self.f2d_path, self.f2d_raw)
        if self.checksum_to_paths:
            _dump_yaml(self.checksum_to_paths_path, self.checksum_to_paths)

    def validate_all(self) -> ValidationResult:
        problems: list[str] = []
        meta = DependencyMetadata(self.meta_raw).root
        f2d = FilesToDependency(self.f2d_raw).root

        for dep, entry in meta.items():
            problems.extend(self._validate_metadata_entry(dep, entry))

        seen_file_to_dep: dict[str, str] = {}
        for dep, files in f2d.items():
            problems.extend(self._validate_file_mappings(dep, files, seen_file_to_dep))

        return ValidationResult(ok=not problems, problems=problems)

    def map_files_to_dependencies(
        self, files: list[Path], num_workers: int = 1
    ) -> tuple[list[Path], DependencyMetadata, dict[str, list[str]]]:
        """Map input files to dependencies and return the unmapped input files, dependency metadata entries, and per-file copyright notices.

        This function assumes validate_all has been called and passed.
        Per-file copyright notices are extracted on-the-fly from file content
        during the same read used for checksumming.

        Args:
            files: List of input files to map to dependencies
            num_workers: Number of parallel workers for file reading/hashing (default: 1)

        Returns:
            unmapped_files: List of unmapped input files
            dep_to_meta: Dictionary of dependency metadata entries
            dep_to_file_notices: Dictionary of per-file copyright notices for each dependency
        """  # noqa: E501
        cs_to_dep: dict[str, str] = {}
        for dep, file_list in self.f2d_raw.items():
            for cs in file_list or []:
                cs_to_dep[cs] = dep

        unmapped_files: list[Path] = []
        dep_to_meta: dict[str, dict] = {}
        dep_to_file_notices: dict[str, list[str]] = {}

        # Read files, compute checksums, and extract copyrights in one pass
        # (parallel if num_workers > 1, else sequential)
        file_strs = [str(f) for f in files]
        file_checksums: dict[Path, str] = {}
        file_copyrights: dict[Path, list[str]] = {}

        def process_results(results):
            for path_str, checksum, size, copyrights in results:
                if checksum is not None:
                    file_checksums[Path(path_str)] = checksum
                    if copyrights:
                        file_copyrights[Path(path_str)] = copyrights

        if num_workers > 1 and len(files) > 10:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                chunksize = max(1, len(files) // num_workers)
                process_results(executor.map(_read_and_hash_file, file_strs, chunksize=chunksize))
        else:
            process_results(map(_read_and_hash_file, file_strs))

        # Process checksums and copyrights to build results
        for f, checksum in file_checksums.items():
            dep = cs_to_dep.get(checksum)
            copyrights = file_copyrights.get(f)
            if copyrights:
                unique_lines = sorted(set(copyrights))
                notice_text = "\n".join(unique_lines)
                dep_to_file_notices.setdefault(dep, []).append(notice_text)
            if dep is None:
                unmapped_files.append(f)
                continue
            if dep not in dep_to_meta:
                dep_to_meta[dep] = self.meta_raw[dep]

        return unmapped_files, DependencyMetadata(dep_to_meta), dep_to_file_notices

    def _validate_metadata_entry(self, dep: str, entry: DependencyMetadataEntry) -> list[str]:
        problems: list[str] = []
        for addr in (entry.license, entry.copyright, entry.attribution):
            if not addr:
                continue
            text = self.cas_get(addr)
            if text is None:
                problems.append(f"Missing CAS object {addr} referenced by {dep}")
            else:
                recomputed = compute_cas_address(text.encode("utf-8"))
                if recomputed != addr:
                    problems.append(f"CAS mismatch for {dep} expected {addr} got {recomputed}")

        if not entry.license:
            problems.append(f"Dependency {dep} missing license text")

        if entry.source:
            src = str(entry.source)
            # TODO - enable these requirements when they can be met automatically
            # Check that the source URL contains a commit hash
            # url_match = COMMIT_HASH_RE.search(src)
            # if not url_match:
            #     problems.append(
            #         f"Source URL for {dep} does not appear to contain a commit hash: {src}"
            #     )
            # elif url_match.group(0) not in dep:
            #     problems.append(
            #         f"The commit hash for {src} is missing from the dependency version: {dep}"
            #     )

            # Check that the source URL is reachable
            try:
                r = requests.head(src, timeout=5)
                if r.status_code >= 400:
                    problems.append(f"Source URL for {dep} not reachable: {src} ({r.status_code})")
            except Exception as e:
                problems.append(f"Source URL for {dep} not reachable: {src} ({e})")
        return problems

    def _validate_file_mappings(
        self,
        dep: str,
        new_files: Iterable[str],
        seen_file_to_dep: dict[str, str] | None = None,
        checksums_with_copyrights: set[str] | None = None,
    ) -> list[str]:
        problems: list[str] = []
        # Check that dependency metadata exists for this dependency
        if dep not in self.meta_raw:
            problems.append(
                f"Input files map to dependency '{dep}' which has no metadata entry. Please use the 'import' command to add metadata for this dependency first."  # noqa: E501
            )
            # Cannot validate copyright notices without metadata
            return problems
        # Ensure each used file is associated with a copyright notice.
        # This check is only possible when file content is available (i.e.,
        # during registration). During validate_all, checksums_with_copyrights
        # is None and this check is skipped.
        if checksums_with_copyrights is not None:
            dep_name = dep.split("/")[0]
            dep_meta = DependencyMetadataEntry.model_validate(self.meta_raw[dep])
            lic_text = self.cas_get(dep_meta.license)
            if (
                not dep_meta.copyright
                and not COPYRIGHT_TEXT_RE.search(lic_text)
                and dep_name not in COPYRIGHT_EXEMPT_DEPS
            ):
                all_files_have = all(cs in checksums_with_copyrights for cs in new_files)
                if not all_files_have:
                    problems.append(
                        f"The license for '{dep}' does not include a copyright notice. Please run the 'import' command to add a copyright notice for '{dep}', or ensure every input file has a copyright header."  # noqa: E501
                    )
        # Check for duplicate mappings
        if seen_file_to_dep is None:
            return problems
        for fp in new_files:
            mapped = seen_file_to_dep.get(fp)
            if mapped and mapped != dep:
                # Include file paths for better error messages
                paths = self.checksum_to_paths.get(fp, [])
                paths_str = ", ".join(paths[:3])  # Show up to 3 paths
                if len(paths) > 3:
                    paths_str += f", ... ({len(paths)} total)"
                problems.append(
                    f"File {fp} mapped to multiple deps: {mapped} and {dep}\n"
                    f"  File path(s): {paths_str or 'unknown'}"
                )
            else:
                seen_file_to_dep[fp] = dep
        return problems
