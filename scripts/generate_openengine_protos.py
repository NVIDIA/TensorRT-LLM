#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate or verify private OpenEngine Python bindings from the vendored schema."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Sequence

_BINDINGS_PACKAGE = "tensorrt_llm.grpc.openengine._generated"
_PROTO_PACKAGE = Path("openengine/v1")
_PROTO_NAMES = (
    "error",
    "generation",
    "kv",
    "lifecycle",
    "lora",
    "model",
    "openengine",
    "server",
)
_GRPC_PROTO_NAME = "openengine"
_COPYRIGHT_HEADER = (
    "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. "
    "All rights reserved.\n"
    "# SPDX-License-Identifier: Apache-2.0\n\n"
)
_GENERATED_INIT_CONTENT = (
    _COPYRIGHT_HEADER + '"""Private OpenEngine bindings generated from the vendored schema."""\n'
)
# Stable ownership contract for generated directories; do not rename.
_OWNERSHIP_MARKER_NAME = ".openengine-generated"
_OWNERSHIP_MARKER_CONTENT = "Owned by scripts/generate_openengine_protos.py; safe to replace.\n"
_GENERATOR_DISTRIBUTIONS = {
    "grpcio": "grpcio",
    "grpcio_tools": "grpcio-tools",
    "protobuf": "protobuf",
    "setuptools": "setuptools",
}
_TOP_LEVEL_OPENENGINE_IMPORT = re.compile(
    r"^(?:from\s+openengine(?:\.|\s)|import\s+openengine(?:\.|\s))", re.MULTILINE
)
_PROTOBUF_VERSION = re.compile(r"^# Protobuf Python Version: ([0-9.]+)$", re.MULTILINE)
_GRPC_VERSION = re.compile(r"^GRPC_GENERATED_VERSION = ['\"]([0-9.]+)['\"]$", re.MULTILINE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _version_tuple(version: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError as error:
        raise RuntimeError(f"Expected a numeric version, got {version!r}") from error


def _read_runtime_floor(path: Path, package: str) -> str:
    requirement = re.compile(rf"^{re.escape(package)}>=(?P<floor>[0-9.]+)(?:,|$)")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = requirement.match(raw_line.strip())
        if match:
            return match.group("floor")
    raise RuntimeError(f"Could not find a lower bound for {package!r} in {path}")


def _read_runtime_ceiling(path: Path, package: str) -> str:
    requirement = re.compile(rf"^{re.escape(package)}(?P<specifiers>[^;#]+)")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        match = requirement.match(raw_line.strip())
        if match is None:
            continue
        for specifier in match.group("specifiers").split(","):
            ceiling = re.fullmatch(r"<(?P<version>[0-9.]+)", specifier.strip())
            if ceiling is not None:
                return ceiling.group("version")
    raise RuntimeError(f"Could not find an exclusive upper bound for {package!r} in {path}")


def _read_generator_requirements(path: Path) -> dict[str, str]:
    requirement = re.compile(r"^(?P<name>[A-Za-z0-9_.-]+)==(?P<version>[0-9]+(?:\.[0-9]+)*)$")
    requirements: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = requirement.fullmatch(line)
        if match is None:
            raise RuntimeError(
                f"Generator requirement must be an exact numeric pin in {path}: {line!r}"
            )
        requirements[match.group("name").lower()] = match.group("version")
    return requirements


def _load_manifest(project_root: Path) -> tuple[Path, dict[str, object]]:
    schema_root = project_root / "tensorrt_llm/grpc/openengine/proto"
    manifest_path = schema_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise RuntimeError(f"{manifest_path} must contain a 'files' checksum map")
    expected_files = {
        *(f"{_PROTO_PACKAGE.as_posix()}/{name}.proto" for name in _PROTO_NAMES),
    }
    if set(files) != expected_files:
        missing = sorted(expected_files - set(files))
        unexpected = sorted(set(files) - expected_files)
        raise RuntimeError(
            f"Unexpected vendored source manifest; missing={missing}, unexpected={unexpected}"
        )
    for relative_path, expected_checksum in files.items():
        if not isinstance(relative_path, str) or not isinstance(expected_checksum, str):
            raise RuntimeError(f"{manifest_path} contains an invalid checksum entry")
        source_path = schema_root / relative_path
        actual_checksum = _sha256(source_path)
        if actual_checksum != expected_checksum:
            raise RuntimeError(
                f"Vendored OpenEngine source checksum mismatch for {source_path}: "
                f"expected {expected_checksum}, got {actual_checksum}"
            )
    return schema_root, manifest


def _rewrite_generated_file(path: Path) -> None:
    content = path.read_text(encoding="utf-8")
    content = re.sub(
        r"^from openengine\.v1 import ([A-Za-z0-9_]+) as ",
        r"from . import \1 as ",
        content,
        flags=re.MULTILINE,
    )
    for proto_name in _PROTO_NAMES:
        content = content.replace(
            f"'{_PROTO_PACKAGE.as_posix().replace('/', '.')}.{proto_name}_pb2'",
            f"'{_BINDINGS_PACKAGE}.{proto_name}_pb2'",
        )
    if _TOP_LEVEL_OPENENGINE_IMPORT.search(content):
        raise RuntimeError(f"Generated binding retains a top-level OpenEngine import: {path}")
    if content.startswith("# -*- coding: utf-8 -*-\n"):
        content = content.replace(
            "# -*- coding: utf-8 -*-\n",
            f"# -*- coding: utf-8 -*-\n{_COPYRIGHT_HEADER}",
            1,
        )
    else:
        content = _COPYRIGHT_HEADER + content
    path.write_text(content, encoding="utf-8", newline="\n")


def _validate_gencode_versions(
    generated_dir: Path, manifest: dict[str, object], project_root: Path
) -> None:
    generator = manifest.get("generator")
    runtime_floors = manifest.get("runtime_floors")
    if not isinstance(generator, dict) or not isinstance(runtime_floors, dict):
        raise RuntimeError("Schema manifest must declare generator and runtime_floors mappings")

    declared_requirements = _read_generator_requirements(
        project_root / "requirements-build-openengine.txt"
    )
    expected_requirements = {
        distribution: str(generator[key]) for key, distribution in _GENERATOR_DISTRIBUTIONS.items()
    }
    if declared_requirements != expected_requirements:
        raise RuntimeError(
            "Generator requirements do not match the versions recorded in the schema manifest: "
            f"expected={expected_requirements}, declared={declared_requirements}"
        )
    for distribution, expected_version in expected_requirements.items():
        installed_version = importlib.metadata.version(distribution)
        if installed_version != expected_version:
            raise RuntimeError(
                f"Expected {distribution}=={expected_version}, but the generator environment has "
                f"{installed_version}"
            )

    protobuf_versions: set[str] = set()
    for path in generated_dir.glob("*_pb2.py"):
        match = _PROTOBUF_VERSION.search(path.read_text(encoding="utf-8"))
        if match is None:
            raise RuntimeError(f"Could not find the protobuf gencode version in {path}")
        protobuf_versions.add(match.group(1))
    if protobuf_versions != {str(generator["protobuf_gencode"])}:
        raise RuntimeError(
            "Unexpected protobuf gencode version(s): " + ", ".join(sorted(protobuf_versions))
        )

    grpc_path = generated_dir / "openengine_pb2_grpc.py"
    grpc_match = _GRPC_VERSION.search(grpc_path.read_text(encoding="utf-8"))
    if grpc_match is None:
        raise RuntimeError(f"Could not find the gRPC gencode version in {grpc_path}")
    grpc_version = grpc_match.group(1)
    if grpc_version != str(generator["grpc_gencode"]):
        raise RuntimeError(f"Expected gRPC gencode {generator['grpc_gencode']}, got {grpc_version}")

    requirements_path = project_root / "requirements.txt"
    declared_protobuf_floor = _read_runtime_floor(requirements_path, "protobuf")
    declared_protobuf_ceiling = _read_runtime_ceiling(requirements_path, "protobuf")
    declared_grpc_floor = _read_runtime_floor(requirements_path, "grpcio")
    expected_protobuf_floor = str(runtime_floors["protobuf"])
    expected_grpc_floor = str(runtime_floors["grpcio"])
    if declared_protobuf_floor != expected_protobuf_floor:
        raise RuntimeError(
            f"Manifest records protobuf>={expected_protobuf_floor}, but requirements.txt declares "
            f"protobuf>={declared_protobuf_floor}"
        )
    if declared_grpc_floor != expected_grpc_floor:
        raise RuntimeError(
            f"Manifest records grpcio>={expected_grpc_floor}, but requirements.txt "
            f"declares grpcio>={declared_grpc_floor}"
        )
    if _version_tuple(next(iter(protobuf_versions))) > _version_tuple(declared_protobuf_floor):
        raise RuntimeError(
            f"Protobuf gencode {next(iter(protobuf_versions))} is newer than the declared runtime "
            f"floor {declared_protobuf_floor}"
        )
    protobuf_gencode_major = _version_tuple(next(iter(protobuf_versions)))[0]
    protobuf_runtime_major = _version_tuple(declared_protobuf_floor)[0]
    if protobuf_runtime_major not in {
        protobuf_gencode_major,
        protobuf_gencode_major + 1,
    }:
        raise RuntimeError(
            f"Protobuf gencode major {protobuf_gencode_major} is incompatible with the declared "
            f"runtime major {protobuf_runtime_major}; they must match or differ by one"
        )
    first_unsupported_protobuf_major = protobuf_gencode_major + 2
    protobuf_ceiling = _version_tuple(declared_protobuf_ceiling)
    # An exclusive ceiling at the first unsupported major is safe (for
    # example, <7). A later ceiling, including <7.1, admits that major.
    if protobuf_ceiling[0] > first_unsupported_protobuf_major or (
        protobuf_ceiling[0] == first_unsupported_protobuf_major
        and any(part != 0 for part in protobuf_ceiling[1:])
    ):
        raise RuntimeError(
            f"Protobuf runtime constraint <{declared_protobuf_ceiling} admits unsupported major "
            f"{first_unsupported_protobuf_major} for gencode major {protobuf_gencode_major}"
        )
    if _version_tuple(grpc_version) > _version_tuple(declared_grpc_floor):
        raise RuntimeError(
            f"gRPC gencode {grpc_version} is newer than the declared runtime floor "
            f"{declared_grpc_floor}"
        )


def _publish_generated_output(staged_output: Path, output_dir: Path) -> None:
    """Replace only output that this generator can prove it owns."""
    (staged_output / _OWNERSHIP_MARKER_NAME).write_text(
        _OWNERSHIP_MARKER_CONTENT, encoding="utf-8", newline="\n"
    )
    if output_dir.is_symlink():
        output_dir.unlink()
        staged_output.rename(output_dir)
        return
    if output_dir.exists():
        if not output_dir.is_dir():
            raise RuntimeError(f"Refusing to replace non-directory output path: {output_dir}")
        ownership_marker = output_dir / _OWNERSHIP_MARKER_NAME
        owned_output = ownership_marker.is_file() and not ownership_marker.is_symlink()
        if not owned_output:
            raise RuntimeError(
                f"Refusing to replace {output_dir}: it is not owned by the OpenEngine generator"
            )

        previous_output = staged_output.parent / "previous-package"
        output_dir.rename(previous_output)
        try:
            staged_output.rename(output_dir)
        except OSError:
            previous_output.rename(output_dir)
            raise
        return
    staged_output.rename(output_dir)


def _check_generated_output(expected_dir: Path, output_dir: Path) -> None:
    """Raise when generated bindings differ from the tracked package."""
    if not output_dir.is_dir():
        raise RuntimeError(_stale_bindings_message(output_dir, ["missing output directory"]))

    expected_files = _generated_files(expected_dir)
    actual_files = _generated_files(output_dir)
    differences = [
        *(f"missing {path}" for path in sorted(expected_files - actual_files)),
        *(f"unexpected {path}" for path in sorted(actual_files - expected_files)),
        *(
            f"changed {path}"
            for path in sorted(expected_files & actual_files)
            if expected_dir.joinpath(path).read_bytes() != output_dir.joinpath(path).read_bytes()
        ),
    ]
    if differences:
        raise RuntimeError(_stale_bindings_message(output_dir, differences))


def _generated_files(directory: Path) -> set[Path]:
    """Return source artifacts while ignoring interpreter bytecode caches."""
    return {
        relative_path
        for path in directory.rglob("*")
        if path.is_file()
        if "__pycache__" not in (relative_path := path.relative_to(directory)).parts
        and path.suffix != ".pyc"
    }


def _stale_bindings_message(output_dir: Path, differences: list[str]) -> str:
    details = "\n  ".join(differences)
    return (
        f"Tracked OpenEngine bindings at {output_dir} are stale:\n  {details}\n"
        "Regenerate them with:\n"
        "  python scripts/generate_openengine_protos.py "
        "--tool-env-root build/openengine-proto-tools"
    )


def _generate(project_root: Path, output_dir: Path) -> None:
    from grpc_tools import _proto as grpc_tools_proto
    from grpc_tools import protoc

    schema_root, manifest = _load_manifest(project_root)
    proto_files = [schema_root / _PROTO_PACKAGE / f"{name}.proto" for name in _PROTO_NAMES]
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="openengine-protos-", dir=output_dir.parent) as temp:
        temporary_root = Path(temp)
        common_args = [
            "grpc_tools.protoc",
            f"-I{schema_root}",
            f"-I{next(iter(grpc_tools_proto.__path__))}",
        ]
        result = protoc.main(
            [
                *common_args,
                f"--python_out={temporary_root}",
                f"--pyi_out={temporary_root}",
                *(str(path) for path in proto_files),
            ]
        )
        if result != 0:
            raise RuntimeError(f"protoc failed while generating protobuf bindings (exit {result})")
        result = protoc.main(
            [
                *common_args,
                f"--grpc_python_out={temporary_root}",
                str(schema_root / _PROTO_PACKAGE / f"{_GRPC_PROTO_NAME}.proto"),
            ]
        )
        if result != 0:
            raise RuntimeError(f"protoc failed while generating gRPC bindings (exit {result})")

        nested_output = temporary_root / _PROTO_PACKAGE
        staged_output = temporary_root / "package"
        staged_output.mkdir()
        expected_files = {
            *(f"{name}_pb2.py" for name in _PROTO_NAMES),
            *(f"{name}_pb2.pyi" for name in _PROTO_NAMES),
            "openengine_pb2_grpc.py",
        }
        actual_files = {path.name for path in nested_output.iterdir() if path.is_file()}
        if actual_files != expected_files:
            missing = sorted(expected_files - actual_files)
            unexpected = sorted(actual_files - expected_files)
            raise RuntimeError(
                f"Unexpected generated binding set; missing={missing}, unexpected={unexpected}"
            )
        for name in sorted(expected_files):
            destination = staged_output / name
            shutil.copyfile(nested_output / name, destination)
            _rewrite_generated_file(destination)
        (staged_output / "__init__.py").write_text(
            _GENERATED_INIT_CONTENT, encoding="utf-8", newline="\n"
        )

        _validate_gencode_versions(staged_output, manifest, project_root)
        _publish_generated_output(staged_output, output_dir)

    print(f"Generated OpenEngine bindings in {output_dir}")


def _check(project_root: Path, output_dir: Path) -> None:
    """Generate to temporary storage and compare it with tracked bindings."""
    with tempfile.TemporaryDirectory(
        prefix="openengine-protos-check-", dir=output_dir.parent
    ) as temp:
        expected_dir = Path(temp) / "generated"
        _generate(project_root, expected_dir)
        _check_generated_output(expected_dir, output_dir)
    print(f"OpenEngine bindings are up to date in {output_dir}")


def _venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts/python.exe"
    return venv_dir / "bin/python"


def _run_in_isolated_environment(
    project_root: Path, output_dir: Path, tool_env_root: Path, check: bool
) -> None:
    requirements_path = project_root / "requirements-build-openengine.txt"
    requirements_digest = _sha256(requirements_path)
    environment_key = (
        f"py{sys.version_info.major}{sys.version_info.minor}-{requirements_digest[:16]}"
    )
    venv_dir = tool_env_root / environment_key
    python = _venv_python(venv_dir)
    stamp = venv_dir / ".requirements.sha256"
    if not python.is_file():
        subprocess.run([sys.executable, "-m", "venv", str(venv_dir)], check=True)
    if not stamp.is_file() or stamp.read_text(encoding="utf-8").strip() != requirements_digest:
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--requirement",
                str(requirements_path),
            ],
            check=True,
        )
        stamp.write_text(requirements_digest + "\n", encoding="utf-8")
    command = [
        str(python),
        str(Path(__file__).resolve()),
        "--project-root",
        str(project_root),
        "--output",
        str(output_dir),
    ]
    if check:
        command.append("--check")
    subprocess.run(command, check=True)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    default_project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_project_root)
    parser.add_argument(
        "--output",
        type=Path,
        help="Generated package directory (default: the source package's _generated directory)",
    )
    parser.add_argument(
        "--tool-env-root",
        type=Path,
        help="Create or reuse an isolated, requirements-keyed grpcio-tools environment here",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Fail if generated bindings differ from the requested output directory",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    project_root = args.project_root.resolve()
    output_dir = (
        Path(os.path.abspath(args.output))
        if args.output is not None
        else project_root / "tensorrt_llm/grpc/openengine/_generated"
    )
    if args.tool_env_root is not None:
        _run_in_isolated_environment(
            project_root, output_dir, args.tool_env_root.resolve(), args.check
        )
    elif args.check:
        _check(project_root, output_dir)
    else:
        _generate(project_root, output_dir)


if __name__ == "__main__":
    main()
