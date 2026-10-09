# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prevent retired transceiver options from returning to serving recipes."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[3]
_CONFIG_DIRS = (
    "examples",
    "tests/integration/defs/disaggregated/test_configs",
    "tests/integration/defs/stress_test/disagg_cancel/configs",
    "tests/scripts/perf",
    "tests/scripts/perf-sanity",
)


def _config_paths() -> list[Path]:
    paths = set()
    for directory in _CONFIG_DIRS:
        for path in (_ROOT / directory).rglob("*"):
            if path.suffix not in (".yaml", ".yml") or "auto_deploy" in path.parts:
                continue
            if "cache_transceiver_config:" in path.read_text():
                paths.add(path)
    return sorted(paths)


def _transceiver_configs(value: object) -> Iterator[dict]:
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "cache_transceiver_config" and isinstance(child, dict):
                yield child
            yield from _transceiver_configs(child)
    elif isinstance(value, list):
        for child in value:
            yield from _transceiver_configs(child)


@pytest.mark.parametrize("path", _config_paths(), ids=lambda p: str(p.relative_to(_ROOT)))
def test_e2e_recipe_avoids_legacy_transceiver_options(path: Path) -> None:
    """Preserve DEFAULT/auto so ordinary E2Es follow future runtime defaults.

    Runtime/schema tests own resolution semantics. This check only prevents
    explicit legacy options from returning to the migrated recipes.
    """
    configs = list(_transceiver_configs(yaml.safe_load(path.read_text())))
    assert configs, f"No transceiver configuration found in {path}"
    for config in configs:
        assert config.get("backend") not in {"UCX", "MPI", "MOONCAKE"}, (path, config)
        # Omitted runtime means auto; an explicit null still selects C++ today.
        assert config.get("transceiver_runtime", "auto") not in {"CPP", None}, (path, config)
        assert "max_tokens_in_buffer" not in config, (path, config)
