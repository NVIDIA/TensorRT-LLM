# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The packages the Kimi KDA layers import when they load must be runtime requirements.

CI installs requirements-dev.txt, so a package listed only there passes every test, yet it is
missing from a plain `pip install tensorrt_llm` and from the release container, where loading
Kimi Linear or Kimi K3 then fails.
"""

import ast
from importlib import metadata
from pathlib import Path

import pytest
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

pytestmark = pytest.mark.cpu_only

REPO_ROOT = Path(__file__).resolve().parents[3]
KDA_PACKAGE = REPO_ROOT / "tensorrt_llm" / "_torch" / "modules" / "kimi_kda"


def _runtime_requirements() -> set:
    names = set()
    for line in (REPO_ROOT / "requirements.txt").read_text().splitlines():
        # Trailing comments are not PEP 508; option lines (-c, --extra-index-url) name no package.
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        try:
            names.add(canonicalize_name(Requirement(line).name))
        except InvalidRequirement:
            continue
    return names


def _load_time_imports(path: Path) -> set:
    """Top-level packages of a module's module-level absolute imports."""
    names = set()
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_kda_load_time_imports_are_runtime_requirements() -> None:
    requirements = _runtime_requirements()
    providers = metadata.packages_distributions()
    # The KDA layers import `fla` at load; with no distribution to map it to, the check is vacuous.
    assert providers.get("fla"), (
        "flash-linear-attention is not installed: `fla` cannot be mapped to a distribution"
    )
    missing = []
    for path in sorted(KDA_PACKAGE.glob("*.py")):
        for name in sorted(_load_time_imports(path) - {"tensorrt_llm"}):
            distributions = {canonicalize_name(d) for d in providers.get(name, ())}
            if distributions and not distributions & requirements:
                missing.append(
                    f"{path.name} imports `{name}`, provided by {', '.join(sorted(distributions))}"
                )
    assert not missing, "not runtime requirements (requirements.txt):\n" + "\n".join(missing)
