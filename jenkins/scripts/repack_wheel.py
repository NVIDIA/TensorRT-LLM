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
"""Add a PEP 440 local version to an existing TensorRT-LLM wheel.

The archive is rewritten without extracting it. Large members, notably shared
libraries, are copied and hashed in bounded chunks while the wheel's RECORD is
regenerated in full. Compiled artifacts are copied byte-for-byte, so version
strings embedded in shared libraries keep the input wheel's public version.
"""

from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import io
import os
import re
import stat
import struct
import sys
import tempfile
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_COPY_BUFFER_SIZE = 8 * 1024 * 1024
_LOCAL_VERSION_RE = re.compile(r"^[A-Za-z0-9]+(?:[-_.][A-Za-z0-9]+)*$")
_DISTRIBUTION_RE = re.compile(r"^[A-Za-z0-9_]+$")
_BUILD_TAG_RE = re.compile(r"^[0-9][A-Za-z0-9_.]*$")
_METADATA_VERSION_RE = re.compile(
    rb"(?m)^(?P<prefix>Version:[ \t]*)(?P<version>[^ \t\r\n]+)(?P<suffix>[ \t]*)(?P<cr>\r?)$"
)
_PYTHON_VERSION_RE = re.compile(
    rb"(?m)^(?P<prefix>__version__[ \t]*=[ \t]*)(?P<quote>[\"'])(?P<version>[^\"'\r\n]+)"
    rb"(?P=quote)(?P<suffix>[ \t]*(?:#[^\r\n]*)?)(?P<cr>\r?)$"
)
_ZIP64_EXTRA_FIELD_ID = 0x0001


@dataclass(frozen=True)
class _WheelFilename:
    distribution: str
    version: str
    tail: str


def _parse_wheel_filename(path: Path) -> _WheelFilename:
    """Parse the portions of a wheel filename needed for version rewriting."""
    if path.suffix != ".whl":
        raise ValueError(f"Expected a .whl input, got: {path.name}")

    components = path.stem.split("-")
    if len(components) not in (5, 6):
        raise ValueError(f"Invalid wheel filename: {path.name}")

    distribution, version = components[:2]
    if not _DISTRIBUTION_RE.fullmatch(distribution):
        raise ValueError(f"Invalid distribution in wheel filename: {distribution!r}")
    if not version:
        raise ValueError(f"Missing version in wheel filename: {path.name}")
    if any(not tag for tag in components[-3:]):
        raise ValueError(f"Invalid compatibility tags in wheel filename: {path.name}")

    build_tags = components[2:-3]
    if build_tags and not _BUILD_TAG_RE.fullmatch(build_tags[0]):
        raise ValueError(f"Invalid build tag in wheel filename: {build_tags[0]!r}")

    return _WheelFilename(
        distribution=distribution,
        version=version,
        tail="-" + "-".join(components[2:]) + ".whl",
    )


def _normalize_local_version(local_version: str) -> str:
    """Validate and normalize a PEP 440 local-version label."""
    if not _LOCAL_VERSION_RE.fullmatch(local_version):
        raise ValueError(
            "Local version must contain alphanumeric segments separated by '.', '-' or '_', "
            "without a leading '+'"
        )
    return re.sub(r"[-_.]+", ".", local_version).lower()


def _is_dist_info_member(filename: str, basename: str) -> bool:
    parts = filename.split("/")
    return len(parts) == 2 and parts[0].endswith(".dist-info") and parts[1] == basename


def _find_single_dist_info_member(infos: list[zipfile.ZipInfo], basename: str) -> zipfile.ZipInfo:
    matches = [info for info in infos if _is_dist_info_member(info.filename, basename)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one top-level .dist-info/{basename} entry, found {len(matches)}"
        )
    return matches[0]


def _read_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> bytes:
    data = bytearray()
    with archive.open(info, "r") as source:
        while chunk := source.read(_COPY_BUFFER_SIZE):
            data.extend(chunk)
    return bytes(data)


def _replace_metadata_version(data: bytes, expected: str, replacement: str) -> bytes:
    matches = list(_METADATA_VERSION_RE.finditer(data))
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one Version header in METADATA, found {len(matches)}")

    match = matches[0]
    actual = match.group("version").decode("ascii")
    if actual != expected:
        raise ValueError(
            f"Wheel filename version {expected!r} does not match METADATA version {actual!r}"
        )
    start, end = match.span("version")
    return data[:start] + replacement.encode("ascii") + data[end:]


def _replace_python_version(data: bytes, expected: str, replacement: str) -> bytes:
    matches = list(_PYTHON_VERSION_RE.finditer(data))
    if len(matches) != 1:
        raise ValueError(
            "Expected exactly one __version__ assignment in tensorrt_llm/version.py, "
            f"found {len(matches)}"
        )

    match = matches[0]
    actual = match.group("version").decode("ascii")
    if actual != expected:
        raise ValueError(
            f"Wheel filename version {expected!r} does not match version.py version {actual!r}"
        )
    start, end = match.span("version")
    return data[:start] + replacement.encode("ascii") + data[end:]


def _rename_prefix(filename: str, old_prefix: str, new_prefix: str) -> str:
    if filename == old_prefix:
        return new_prefix
    if filename.startswith(old_prefix + "/"):
        return new_prefix + filename[len(old_prefix) :]
    return filename


def _rename_member(
    filename: str,
    old_dist_info: str,
    new_dist_info: str,
    old_data: str,
    new_data: str,
) -> str:
    renamed = _rename_prefix(filename, old_dist_info, new_dist_info)
    return _rename_prefix(renamed, old_data, new_data)


def _without_zip64_extra(extra: bytes) -> bytes:
    """Drop structural ZIP64 data, which zipfile regenerates with new sizes."""
    offset = 0
    fields = bytearray()
    while offset < len(extra):
        if len(extra) - offset < 4:
            return extra
        field_id, field_size = struct.unpack_from("<HH", extra, offset)
        field_end = offset + 4 + field_size
        if field_end > len(extra):
            return extra
        if field_id != _ZIP64_EXTRA_FIELD_ID:
            fields.extend(extra[offset:field_end])
        offset = field_end
    return bytes(fields)


def _clone_zip_info(info: zipfile.ZipInfo, filename: str, file_size: int) -> zipfile.ZipInfo:
    cloned = copy.copy(info)
    cloned.filename = filename
    cloned.orig_filename = filename
    cloned.file_size = file_size
    cloned.extra = _without_zip64_extra(info.extra)
    return cloned


def _encoded_record_hash(digest: bytes) -> str:
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}"


def _write_bytes(archive: zipfile.ZipFile, info: zipfile.ZipInfo, data: bytes) -> tuple[str, int]:
    output_info = _clone_zip_info(info, info.filename, len(data))
    with archive.open(output_info, "w", force_zip64=len(data) > zipfile.ZIP64_LIMIT) as destination:
        destination.write(data)
    return _encoded_record_hash(hashlib.sha256(data).digest()), len(data)


def _copy_member(
    source_archive: zipfile.ZipFile,
    destination_archive: zipfile.ZipFile,
    source_info: zipfile.ZipInfo,
    destination_info: zipfile.ZipInfo,
) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with (
        source_archive.open(source_info, "r") as source,
        destination_archive.open(
            destination_info,
            "w",
            force_zip64=source_info.file_size > zipfile.ZIP64_LIMIT,
        ) as destination,
    ):
        while chunk := source.read(_COPY_BUFFER_SIZE):
            destination.write(chunk)
            digest.update(chunk)
            size += len(chunk)
    return _encoded_record_hash(digest.digest()), size


def _serialize_record(rows: list[tuple[str, str, str]], record_filename: str) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerows([*rows, (record_filename, "", "")])
    return output.getvalue().encode("utf-8")


def _rewrite_archive(
    source: zipfile.ZipFile,
    destination: zipfile.ZipFile,
    infos: list[zipfile.ZipInfo],
    renamed_members: dict[str, str],
    replacements: dict[str, bytes],
    source_record: zipfile.ZipInfo,
) -> None:
    rows: list[tuple[str, str, str]] = []
    destination.comment = source.comment

    for source_info in infos:
        if source_info.filename == source_record.filename:
            continue

        output_name = renamed_members[source_info.filename]
        if source_info.is_dir():
            directory_info = _clone_zip_info(source_info, output_name, 0)
            destination.writestr(directory_info, b"")
            continue

        output_info = _clone_zip_info(source_info, output_name, source_info.file_size)
        if source_info.filename in replacements:
            record_hash, size = _write_bytes(
                destination, output_info, replacements[source_info.filename]
            )
        else:
            record_hash, size = _copy_member(source, destination, source_info, output_info)
        rows.append((output_name, record_hash, str(size)))

    output_record_name = renamed_members[source_record.filename]
    record_data = _serialize_record(rows, output_record_name)
    output_record_info = _clone_zip_info(source_record, output_record_name, len(record_data))
    _write_bytes(destination, output_record_info, record_data)


def repack_wheel(input_wheel: Path, local_version: str, output_dir: Path | None = None) -> Path:
    """Create a local-version variant of a TensorRT-LLM wheel.

    Args:
        input_wheel: Wheel whose version has no PEP 440 local component.
        local_version: Local label without the leading ``+``.
        output_dir: Destination directory. Defaults to the input directory.

    Returns:
        Absolute path to the repacked wheel.

    Raises:
        OSError: If an input or output file operation fails.
        ValueError: If the wheel layout or versions are inconsistent.
        zipfile.BadZipFile: If the input is not a valid ZIP archive.
    """
    input_path = input_wheel.expanduser().resolve(strict=True)
    wheel_filename = _parse_wheel_filename(input_path)
    if "+" in wheel_filename.version:
        raise ValueError(f"Input wheel already has a local version: {wheel_filename.version!r}")

    normalized_local_version = _normalize_local_version(local_version)
    output_version = f"{wheel_filename.version}+{normalized_local_version}"
    output_directory = (
        output_dir.expanduser().resolve() if output_dir is not None else input_path.parent
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / (
        f"{wheel_filename.distribution}-{output_version}{wheel_filename.tail}"
    )

    expected_dist_info = f"{wheel_filename.distribution}-{wheel_filename.version}.dist-info"
    output_dist_info = f"{wheel_filename.distribution}-{output_version}.dist-info"
    expected_data = f"{wheel_filename.distribution}-{wheel_filename.version}.data"
    output_data = f"{wheel_filename.distribution}-{output_version}.data"

    temporary_path: Path | None = None
    try:
        with zipfile.ZipFile(input_path, "r") as source:
            infos = source.infolist()
            filenames = [info.filename for info in infos]
            if len(filenames) != len(set(filenames)):
                raise ValueError("Input wheel contains duplicate archive member names")

            signature_entries = [
                info.filename
                for info in infos
                if _is_dist_info_member(info.filename, "RECORD.jws")
                or _is_dist_info_member(info.filename, "RECORD.p7s")
            ]
            if signature_entries:
                raise ValueError("Cannot repack a signed wheel containing RECORD.jws or RECORD.p7s")

            metadata_info = _find_single_dist_info_member(infos, "METADATA")
            actual_dist_info = metadata_info.filename.rsplit("/", 1)[0]
            if actual_dist_info != expected_dist_info:
                raise ValueError(
                    f"Wheel filename expects dist-info directory {expected_dist_info!r}, "
                    f"found {actual_dist_info!r}"
                )

            record_info = _find_single_dist_info_member(infos, "RECORD")
            if record_info.filename != f"{expected_dist_info}/RECORD":
                raise ValueError(
                    f"RECORD is not in the expected dist-info directory {expected_dist_info!r}"
                )

            info_by_name = {info.filename: info for info in infos}
            version_info = info_by_name.get("tensorrt_llm/version.py")
            if version_info is None or version_info.is_dir():
                raise ValueError("Input wheel is missing tensorrt_llm/version.py")

            metadata_data = _replace_metadata_version(
                _read_member(source, metadata_info),
                wheel_filename.version,
                output_version,
            )
            version_data = _replace_python_version(
                _read_member(source, version_info),
                wheel_filename.version,
                output_version,
            )

            renamed_members = {
                info.filename: _rename_member(
                    info.filename,
                    expected_dist_info,
                    output_dist_info,
                    expected_data,
                    output_data,
                )
                for info in infos
            }
            if len(set(renamed_members.values())) != len(renamed_members):
                raise ValueError("Version rewriting would create duplicate archive member names")

            replacements = {
                metadata_info.filename: metadata_data,
                version_info.filename: version_data,
            }

            with tempfile.NamedTemporaryFile(
                dir=output_directory,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)

            with zipfile.ZipFile(temporary_path, "w", allowZip64=True) as destination:
                _rewrite_archive(
                    source,
                    destination,
                    infos,
                    renamed_members,
                    replacements,
                    record_info,
                )

        temporary_path.chmod(stat.S_IMODE(input_path.stat().st_mode))
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    return output_path.resolve()


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add a local version to a TensorRT-LLM wheel without rebuilding it."
    )
    parser.add_argument("input_wheel", type=Path, help="wheel without a local version")
    parser.add_argument("local_version", help="PEP 440 local label without the leading '+'")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="destination directory (defaults to the input wheel's directory)",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        output_path = repack_wheel(args.input_wheel, args.local_version, args.output_dir)
    except (OSError, RuntimeError, ValueError, zipfile.BadZipFile) as error:
        print(f"repack_wheel.py: error: {error}", file=sys.stderr)
        return 1

    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
