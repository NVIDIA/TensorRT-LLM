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
"""Regenerate constraints.txt from the currently-installed environment.

constraints.txt pins the pure-Python dependencies of TensorRT-LLM to the
versions verified in our NGC base container. It is a pip *constraint* file
(applied via `-c`), not an install list.

Intended workflow:
    1. Enter a fresh NGC container that matches the desired CI baseline.
    2. `pip install -r requirements-dev.txt` (or otherwise populate the env).
    3. `python scripts/generate_constraints.py`  -> rewrites constraints.txt.
    4. Review the diff and commit.

Packages provided by the container (torch, triton, tensorrt, nixl, cuda-*,
nvidia-*, flashinfer, modelopt, cutlass-dsl, ...) are deliberately omitted --
they are inherited via `virtualenv --system-site-packages`.

Packages already carrying an explicit version specifier in requirements.txt /
requirements-dev.txt are skipped too (no point duplicating an existing pin).
"""

from __future__ import annotations

import argparse
import re
import sys
from importlib import metadata
from pathlib import Path

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS = PROJECT_ROOT / "requirements.txt"
REQUIREMENTS_DEV = PROJECT_ROOT / "requirements-dev.txt"
CONSTRAINTS = PROJECT_ROOT / "constraints.txt"

# ---------------------------------------------------------------------------
# Exclusions -- container-provided packages. Normalized names (PEP 503:
# lowercase, '-' and '_' and '.' collapsed to '-').
# ---------------------------------------------------------------------------
CONTAINER_PROVIDED = {
    # PyTorch stack
    "torch",
    "torchvision",
    "torchao",
    # Triton
    "triton",
    "triton-kernels",
    # TensorRT
    "tensorrt",
    "tensorrt-llm",
    # NIXL / NCCL
    "nixl",
    "nccl-cu13",
    # CUDA Python stack
    "cuda-python",
    "cuda-core",
    "cuda-tile",
    "cupti-python",
    # FlashInfer / ModelOpt / CUTLASS DSL
    "flashinfer-python",
}
# Packages whose normalized name starts with any of these prefixes are also
# treated as container-provided.
CONTAINER_PREFIXES = ("nvidia-",)

# ---------------------------------------------------------------------------
# File preamble and section headers (kept in sync with constraints.txt layout)
# ---------------------------------------------------------------------------
FILE_HEADER = """\
# Pinned versions of pure-Python dependencies for reproducible CI/dev builds.
#
# This file is a pip *constraints* file -- only pins already-requested packages
# to a specific version, never triggers installs on its own. It is applied via
# `-c constraints.txt` from requirements.txt and requirements-dev.txt.
#
# Scope:
#   - Pure-Python deps we install ourselves (pandas, click, fastapi, ...).
#   - Container-provided packages (torch, triton, tensorrt, nixl, nccl,
#     cuda-*, nvidia-*, flashinfer, modelopt, cutlass-dsl, ...) are
#     DELIBERATELY OMITTED -- they are inherited via
#     `virtualenv --system-site-packages` from the NGC base image.
#   - Packages already pinned in requirements.txt / requirements-dev.txt are
#     omitted to avoid duplication.
#
# Versions below correspond to the NGC container used by current CI. Regenerate
# inside the target container via:  python scripts/generate_constraints.py
#
# ---------------------------------------------------------------------------
# Security WARs inherited from the base image (pytorch:<base_tag>). Remove
# these entries when the base image is updated past the advisory. These three
# entries are also force-installed by docker/Dockerfile.multi to upgrade the
# vulnerable versions shipped in the base image.
# ---------------------------------------------------------------------------
# WAR against https://github.com/advisories/GHSA-8rrh-rw8j-w5fx
wheel>=0.46.2
# WAR against https://github.com/advisories/GHSA-qjxf-f2mg-c6mc
tornado>=6.5.5
# WAR against https://github.com/advisories/GHSA-3936-cmfr-pm3m
black>=26.3.1
"""

SECTION_HEADER = """\

# ---------------------------------------------------------------------------
# Pins from {filename}
# ---------------------------------------------------------------------------
"""


def normalize(name: str) -> str:
    """Normalize a distribution name per PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


def is_container_provided(name: str) -> bool:
    n = normalize(name)
    if n in CONTAINER_PROVIDED:
        return True
    return any(n.startswith(p) for p in CONTAINER_PREFIXES)


# A requirement line is "unpinned" when it carries no version specifier at all
# (no `==`, `~=`, `>=`, `<=`, `>`, `<`, `!=`, `===`). Environment markers and
# extras are allowed but do not count as a version specifier.
_SPECIFIER_CHARS = ("==", "~=", "===", "!=", ">=", "<=", ">", "<")


def parse_unpinned(requirements_file: Path) -> list[str]:
    """Return ordered list of unpinned package names from a requirements file.

    Skips comments, blank lines, `-r`/`-c`/`--` directives, URL requirements,
    and any requirement that already carries a version specifier.
    """
    names: list[str] = []
    seen: set[str] = set()
    for raw in requirements_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith(("-r", "-c", "--", "-i ")):
            continue
        if line.startswith(("git+", "http://", "https://")):
            continue

        # Strip environment marker, e.g. `pkg ; python_version>='3.10'`
        spec = line.split(";", 1)[0].strip()
        # Strip inline comment
        spec = spec.split("#", 1)[0].strip()
        if not spec:
            continue
        if any(s in spec for s in _SPECIFIER_CHARS):
            continue

        # Strip extras: `pkg[extra1,extra2]` -> `pkg`
        name = re.split(r"[\[\s]", spec, maxsplit=1)[0].strip()
        if not name:
            continue
        key = normalize(name)
        if key in seen:
            continue
        seen.add(key)
        names.append(name)
    return names


def compatible_release(version: str) -> str:
    """Return a PEP 440 compatible-release specifier for `version`.

    Policy:
      - `0.0.X`  -> `~=0.0.X` (lock to patch, allow nothing beyond 0.1.0)
      - otherwise -> `~=MAJOR.MINOR` (allow patch bumps within the same
        MAJOR.MINOR series)
    """
    parts = version.split(".")
    if len(parts) < 2:
        return f"=={version}"
    major, minor = parts[0], parts[1]
    if major == "0" and minor == "0":
        # e.g. python-multipart 0.0.26 -> ~=0.0.26
        patch = parts[2] if len(parts) >= 3 else "0"
        # Strip any pre/post/dev tag from patch (e.g. `0.17.2.4` -> `0`).
        patch = re.match(r"\d+", patch).group(0) if re.match(r"\d+", patch) else "0"
        return f"~=0.0.{patch}"
    return f"~={major}.{minor}"


def installed_version(name: str) -> str | None:
    """Return the installed version of `name` or None if not installed."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def build_section(requirements_file: Path, already_emitted: set[str]) -> tuple[str, list[str]]:
    """Build a section of constraint lines for one requirements file.

    Returns (section_text, warnings_list). `already_emitted` is mutated with
    newly emitted package keys so later sections don't duplicate them.
    """
    warnings: list[str] = []
    lines: list[str] = []
    for name in parse_unpinned(requirements_file):
        key = normalize(name)
        if key in already_emitted:
            continue
        if is_container_provided(name):
            continue
        version = installed_version(name)
        if version is None:
            warnings.append(f"  {name}: not installed in the current environment -- skipped")
            continue
        spec = compatible_release(version)
        lines.append(f"{name}{spec}")
        already_emitted.add(key)

    if not lines:
        return "", warnings
    section = SECTION_HEADER.format(filename=requirements_file.name) + "\n".join(lines) + "\n"
    return section, warnings


def render_constraints() -> tuple[str, list[str]]:
    """Render the full constraints.txt content from the current environment."""
    already_emitted: set[str] = {"wheel", "tornado", "black"}
    body = FILE_HEADER
    all_warnings: list[str] = []
    for req in (REQUIREMENTS, REQUIREMENTS_DEV):
        section, warnings = build_section(req, already_emitted)
        body += section
        all_warnings.extend(warnings)
    return body, all_warnings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the generated file to stdout instead of writing it",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if constraints.txt would change. Useful in CI.",
    )
    args = parser.parse_args()

    rendered, warnings = render_constraints()

    for warning in warnings:
        print(f"[warning] {warning}", file=sys.stderr)

    if args.stdout:
        sys.stdout.write(rendered)
        return 0

    if args.check:
        current = CONSTRAINTS.read_text() if CONSTRAINTS.exists() else ""
        if current != rendered:
            print(
                f"{CONSTRAINTS} is stale; run `python scripts/generate_constraints.py`"
                " inside the target container to refresh it.",
                file=sys.stderr,
            )
            return 1
        return 0

    CONSTRAINTS.write_text(rendered)
    print(f"Wrote {CONSTRAINTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
