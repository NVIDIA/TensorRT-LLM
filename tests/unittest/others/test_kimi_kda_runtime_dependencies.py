# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi KDA load-time imports must be available from runtime requirements."""

import ast
from importlib import metadata
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parents[3]
KDA_PACKAGE = REPO_ROOT / "tensorrt_llm" / "_torch" / "modules" / "kimi_kda"


def test_kda_load_time_imports_are_runtime_requirements() -> None:
    requirements = {
        canonicalize_name(Requirement(line.split("#", 1)[0].strip()).name)
        for line in (REPO_ROOT / "requirements.txt").read_text().splitlines()
        if line.strip() and not line.lstrip().startswith(("#", "-"))
    }
    providers = metadata.packages_distributions()
    assert providers.get("fla"), "No installed distribution provides the `fla` package"

    imports = set()
    for path in KDA_PACKAGE.glob("*.py"):
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports.add(node.module.split(".")[0])

    missing = {
        name
        for name in imports - {"tensorrt_llm"}
        if (distributions := {canonicalize_name(dist) for dist in providers.get(name, ())})
        and not distributions & requirements
    }
    assert not missing, (
        f"Kimi KDA load-time imports are not runtime requirements: {sorted(missing)}"
    )
