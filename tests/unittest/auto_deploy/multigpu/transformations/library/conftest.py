# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Local conftest for multigpu/transformations/library tests.

``test_sharding_num_correctness`` is parametrized intrinsically (one invocation
per modeling file × dist config). The two CLI options pin a single value
to that axis for per-file debugging; when omitted, the test auto-discovers
the supported ``modeling_*.py`` files from the active AutoDeploy package and
runs the cheapest dist config across all of them.

IR-marked files (graph carries ``torch.ops.auto_deploy.all_reduce`` per skill
rule A3) run ``apply_sharding_hints`` + ``strip_sharding_hints``. Legacy
(non-IR-marked) files run ``detect_sharding`` + ``sharding_transform_executor``
with ``HEURISTIC`` source only -- the standalone-export harness has no
``ModelFactory`` or per-model YAML to feed FACTORY / MANUAL. Models for
which the legacy heuristic inserts zero collectives ``pytest.skip`` cleanly
(rather than rubber-stamping a trivial identity pass).
"""

from importlib.util import find_spec
from pathlib import Path

import pytest

_DIST_CONFIG_CHOICES = ("tp-only", "ep-only", "tep", "attn-dp")
_QUANT_CHOICES = ("none", "nvfp4")

# When --sharding-ir-dist-config is omitted:
#   * if --sharding-ir-modeling-file is supplied: keep the legacy single-file
#     debugging default (heavier 4-rank config, broader coverage).
#   * otherwise: auto-discovery mode for CI -- iterate every modeling file at
#     the cheapest dist config (2 ranks, TP only) to keep matrix cost low.
_DIST_CONFIG_DEFAULT_CLI = "tep"
_DIST_CONFIG_DEFAULT_AUTO = "tp-only"

# This value is rewritten to ``llmc`` with the copied tests. Resolve the package
# without importing it so test collection does not eagerly load every custom model.
_AUTO_DEPLOY_PACKAGE = "tensorrt_llm._torch.auto_deploy"

# These modeling files cannot participate in standalone LLMC auto-discovery.
# They either depend on TRT-LLM-only router ops, need trust-remote-code config
# classes that LLMC does not package, or are conditional-generation models not
# registered as CausalLM classes for this harness. Keep canonical discovery
# unchanged so the native TensorRT-LLM test matrix still covers them.
_STANDALONE_DISCOVERY_EXCLUDED = {
    "modeling_decilm.py",
    "modeling_deepseek.py",
    "modeling_eagle.py",
    "modeling_glm4_moe.py",
    "modeling_gpt_oss.py",
    "modeling_hunyuan_moe.py",
    "modeling_internlm3.py",
    "modeling_nemotron_flash.py",
    "modeling_nemotron_h.py",
    "modeling_skywork_r1v2.py",
}


def _package_dir(package_name: str) -> Path:
    """Resolve a package directory without importing the package."""
    top_level, *subpackages = package_name.split(".")
    spec = find_spec(top_level)
    if spec is None or spec.submodule_search_locations is None:
        raise ImportError(f"Could not locate package {package_name!r}")
    for location in spec.submodule_search_locations:
        package_dir = Path(location).joinpath(*subpackages)
        if package_dir.is_dir():
            return package_dir
    raise ImportError(f"Package {package_name!r} has no filesystem location")


_MODELING_DIR = _package_dir(_AUTO_DEPLOY_PACKAGE) / "models" / "custom"


def pytest_addoption(parser):
    parser.addoption(
        "--sharding-ir-modeling-file",
        action="store",
        default=None,
        help=(
            "Path to a single modeling file to verify with "
            "test_sharding_num_correctness. Accepts an absolute path, a path "
            "relative to cwd, or a bare module short name (resolved under "
            "the active AutoDeploy package's models.custom package). "
            "When omitted, the test auto-discovers and parametrizes over "
            "every supported modeling_*.py in custom/."
        ),
    )
    parser.addoption(
        "--sharding-ir-dist-config",
        action="store",
        default=None,
        choices=_DIST_CONFIG_CHOICES,
        help=(
            "Parallelism config to exercise in test_sharding_num_correctness. "
            "See test_sharding_num_correctness._DIST_CONFIGS for grids: "
            "'tp-only' (2 ranks), 'ep-only' (2 ranks), 'tep' (4 ranks), "
            "'attn-dp' (4 ranks, attention-DP + MoEAllToAll). "
            f"Defaults: {_DIST_CONFIG_DEFAULT_CLI!r} when "
            "--sharding-ir-modeling-file is supplied, "
            f"{_DIST_CONFIG_DEFAULT_AUTO!r} (cheapest, broad-coverage mode) "
            "when it's not."
        ),
    )
    parser.addoption(
        "--sharding-ir-quant",
        action="store",
        default=None,
        choices=_QUANT_CHOICES,
        help=(
            "Quantization to apply before sharding in "
            "test_sharding_num_correctness. 'none' (default) exercises the "
            "bf16 path; 'nvfp4' runs the NVFP4 quant pre-pass "
            "(quantize_nvfp4_linear_from_config + match_nvfp4_swiglu_pattern + "
            "quantize_nvfp4_moe) on both the sharded and unsharded graphs so "
            "the FP4 weight-scale sharding paths are verified. The NVFP4 path "
            "is skipped (not failed) on non-Blackwell hardware."
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
    output readable (``test_sharding_num_correctness[qwen3_next-tp-only]``).
    """
    name = Path(path).name
    if name.startswith("modeling_"):
        name = name[len("modeling_") :]
    if name.endswith(".py"):
        name = name[: -len(".py")]
    return name


def _discover_modeling_files() -> list:
    """Sorted absolute paths of supported modeling files in the AD custom models dir.

    Deterministic ordering is important: a flaky test mapped to a different
    parametrize id from run to run would mislead bisection.
    """
    excluded = _STANDALONE_DISCOVERY_EXCLUDED if _AUTO_DEPLOY_PACKAGE == "llmc" else set()
    return sorted(
        str(path) for path in _MODELING_DIR.glob("modeling_*.py") if path.name not in excluded
    )


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

    if "sharding_ir_quant" in metafunc.fixturenames:
        # Default to bf16-only ('none') so existing CI behavior is unchanged;
        # opt into NVFP4 explicitly via --sharding-ir-quant nvfp4.
        cli_quant = metafunc.config.getoption("--sharding-ir-quant")
        quants = [cli_quant] if cli_quant is not None else ["none"]
        metafunc.parametrize("sharding_ir_quant", quants, ids=lambda q: f"quant-{q}")


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
