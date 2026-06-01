# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Local conftest for multigpu/transformations/library tests."""

import pytest

_DIST_CONFIG_CHOICES = ("tp-only", "ep-only", "tep", "attn-dp")
_DIST_CONFIG_DEFAULT = "tep"


def pytest_addoption(parser):
    parser.addoption(
        "--sharding-ir-modeling-file",
        action="store",
        default=None,
        help=(
            "Path to a sharding-IR-aware modeling file to verify with "
            "test_sharding_ir_equivalence. Accepts an absolute path, a path "
            "relative to cwd or repo root, or a bare module short name "
            "(resolved under tensorrt_llm._torch.auto_deploy.models.custom). "
            "No filename pattern is required. The test is skipped (not "
            "failed) when this option is absent."
        ),
    )
    parser.addoption(
        "--sharding-ir-dist-config",
        action="store",
        choices=_DIST_CONFIG_CHOICES,
        default=_DIST_CONFIG_DEFAULT,
        help=(
            "Parallelism config to exercise in test_sharding_ir_equivalence. "
            "See test_sharding_ir_equivalence._DIST_CONFIGS for grids: "
            "'tp-only' (2 ranks), 'ep-only' (2 ranks), 'tep' (4 ranks, default), "
            "'attn-dp' (4 ranks, attention-DP + MoEAllToAll)."
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


@pytest.fixture
def sharding_ir_modeling_file(request) -> str:
    path = request.config.getoption("--sharding-ir-modeling-file")
    if path is None:
        pytest.skip(
            "--sharding-ir-modeling-file not supplied; sharding IR equivalence "
            "test is only run on-demand per modeling file."
        )
    return path


@pytest.fixture
def sharding_ir_dist_config(request) -> str:
    return request.config.getoption("--sharding-ir-dist-config")


@pytest.fixture
def sharding_ir_eagle_draft(request) -> str:
    model_type = request.config.getoption("--sharding-ir-eagle-draft")
    if model_type is None:
        pytest.skip(
            "--sharding-ir-eagle-draft not supplied; Eagle draft sharding IR "
            "equivalence test is only run on-demand per draft model_type."
        )
    return model_type
