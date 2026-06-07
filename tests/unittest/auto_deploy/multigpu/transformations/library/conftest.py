# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Local conftest for multigpu/transformations/library tests.

``test_sharding_ir_equivalence`` is parametrized intrinsically (one invocation
per modeling file × dist config). The two CLI options pin a single value
to that axis for per-file debugging; when omitted, the test auto-discovers
every ``modeling_*.py`` under ``tensorrt_llm/_torch/auto_deploy/models/custom/``
and runs the cheapest dist config across all of them. Legacy (non-IR-marked)
modeling files pass as a no-op identity (``apply_sharding_hints`` finds no
markers and leaves the graph unchanged, so sharded == unsharded by
construction); IR-marked files exercise the full equivalence path.
"""

from pathlib import Path

import pytest

_DIST_CONFIG_CHOICES = ("tp-only", "ep-only", "tep", "attn-dp")

# When --sharding-ir-dist-config is omitted:
#   * if --sharding-ir-modeling-file is supplied: keep the legacy single-file
#     debugging default (heavier 4-rank config, broader coverage).
#   * otherwise: auto-discovery mode for CI -- iterate every modeling file at
#     the cheapest dist config (2 ranks, TP only) to keep matrix cost low.
_DIST_CONFIG_DEFAULT_CLI = "tep"
_DIST_CONFIG_DEFAULT_AUTO = "tp-only"

_REPO_ROOT = Path(__file__).resolve().parents[6]
_MODELING_DIR = _REPO_ROOT / "tensorrt_llm" / "_torch" / "auto_deploy" / "models" / "custom"


def pytest_addoption(parser):
    parser.addoption(
        "--sharding-ir-modeling-file",
        action="store",
        default=None,
        help=(
            "Path to a single modeling file to verify with "
            "test_sharding_ir_equivalence. Accepts an absolute path, a path "
            "relative to cwd or repo root, or a bare module short name "
            "(resolved under tensorrt_llm._torch.auto_deploy.models.custom). "
            "When omitted, the test auto-discovers and parametrizes over "
            "EVERY modeling_*.py in custom/."
        ),
    )
    parser.addoption(
        "--sharding-ir-dist-config",
        action="store",
        default=None,
        choices=_DIST_CONFIG_CHOICES,
        help=(
            "Parallelism config to exercise in test_sharding_ir_equivalence. "
            "See test_sharding_ir_equivalence._DIST_CONFIGS for grids: "
            "'tp-only' (2 ranks), 'ep-only' (2 ranks), 'tep' (4 ranks), "
            "'attn-dp' (4 ranks, attention-DP + MoEAllToAll). "
            f"Defaults: {_DIST_CONFIG_DEFAULT_CLI!r} when "
            "--sharding-ir-modeling-file is supplied, "
            f"{_DIST_CONFIG_DEFAULT_AUTO!r} (cheapest, broad-coverage mode) "
            "when it's not."
        ),
    )
    parser.addoption(
        "--sharding-ir-eagle-draft",
        action="store",
        default=None,
        help=(
            "Base model_type of an Eagle draft to verify with "
            "test_sharding_ir_eagle_draft_equivalence (e.g. 'llama', "
            "'nemotron_h'). Builds a tiny EagleDrafterForCausalLM for that "
            "model_type and checks sharded == unsharded draft prefill. The test "
            "is skipped (not failed) when this option is absent."
        ),
    )


def _modeling_id(path: str) -> str:
    """Compact pytest id: strip ``modeling_`` prefix and ``.py`` suffix.

    e.g. ``.../modeling_qwen3_next.py`` -> ``qwen3_next``. Keeps parametrize
    output readable (``test_sharding_ir_equivalence[qwen3_next-tp-only]``).
    """
    name = Path(path).name
    if name.startswith("modeling_"):
        name = name[len("modeling_") :]
    if name.endswith(".py"):
        name = name[: -len(".py")]
    return name


def _discover_modeling_files() -> list:
    """Sorted absolute paths of every modeling_*.py in the AD custom models dir.

    Deterministic ordering is important: a flaky test mapped to a different
    parametrize id from run to run would mislead bisection.
    """
    return sorted(str(p) for p in _MODELING_DIR.glob("modeling_*.py"))


def pytest_generate_tests(metafunc):
    """Materialize the modeling-file × dist-config matrix from CLI options.

    The CLI options pin a single value; when both are omitted, we iterate
    every modeling file at the cheapest dist config (auto-discovery mode for
    CI). When only ``--sharding-ir-modeling-file`` is supplied, the dist
    config falls back to the legacy single-file default (``tep``, 4 ranks).
    """
    cli_file = metafunc.config.getoption("--sharding-ir-modeling-file")
    cli_dist = metafunc.config.getoption("--sharding-ir-dist-config")

    if "sharding_ir_modeling_file" in metafunc.fixturenames:
        files = [cli_file] if cli_file is not None else _discover_modeling_files()
        metafunc.parametrize("sharding_ir_modeling_file", files, ids=_modeling_id)

    if "sharding_ir_dist_config" in metafunc.fixturenames:
        if cli_dist is not None:
            configs = [cli_dist]
        elif cli_file is not None:
            configs = [_DIST_CONFIG_DEFAULT_CLI]
        else:
            configs = [_DIST_CONFIG_DEFAULT_AUTO]
        metafunc.parametrize("sharding_ir_dist_config", configs)


@pytest.fixture
def sharding_ir_eagle_draft(request) -> str:
    """Pin a single Eagle draft model_type for ``test_sharding_ir_eagle_draft_equivalence``.

    Kept as a fixture (not parametrized via ``pytest_generate_tests``) because
    the Eagle draft test is single-target by design -- one model_type per
    invocation, on-demand.
    """
    model_type = request.config.getoption("--sharding-ir-eagle-draft")
    if model_type is None:
        pytest.skip(
            "--sharding-ir-eagle-draft not supplied; Eagle draft sharding IR "
            "equivalence test is only run on-demand per draft model_type."
        )
    return model_type
