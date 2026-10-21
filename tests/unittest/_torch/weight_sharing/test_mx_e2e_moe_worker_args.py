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
"""Unit tests for how the MX qualification harness passes MoE settings to its workers.

Every role of one donor/receiver row must load the model with the same MoE
partition and finalize setting, and rows without MoE settings pass no MoE
keyword arguments to `LLM`.
"""

from __future__ import annotations

import pytest

from tensorrt_llm.llmapi import MoeConfig

__extra_import_path__ = ["~/tests/integration"]
from defs.model_express.mx_e2e_worker import _llm_kwargs, _parse_args  # noqa: E402
from defs.model_express.mx_harness import MxE2ECase, moe_worker_args  # noqa: E402

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "case_overrides, expected_args, expected_moe_kwargs",
    [
        pytest.param({}, (), {}, id="dense-row"),
        pytest.param(
            {"moe_disable_finalize_fusion": True},
            ("--moe-disable-finalize-fusion",),
            {"moe_config": MoeConfig(disable_finalize_fusion=True)},
            id="moe-tp-row",
        ),
        pytest.param(
            {"moe_ep_size": 2, "moe_disable_finalize_fusion": True},
            ("--moe-ep-size", "2", "--moe-disable-finalize-fusion"),
            {
                "moe_expert_parallel_size": 2,
                "moe_config": MoeConfig(disable_finalize_fusion=True),
            },
            id="moe-ep-row",
        ),
    ],
)
@pytest.mark.parametrize("role", ["baseline", "donor", "receiver"])
def test_moe_row_settings_reach_every_role(
    case_overrides: dict[str, object],
    expected_args: tuple[str, ...],
    expected_moe_kwargs: dict[str, object],
    role: str,
) -> None:
    case = MxE2ECase(
        model_env="TRTLLM_MX_TEST_MODEL",
        default_model_subdir="model",
        repository_cache_prefix="models--trtllm-mx-e2e--test-tp2",
        tp_size=2,
        **case_overrides,
    )
    worker_args = moe_worker_args(case)
    argv = ["--role", role, "--model", "/model", "--tp-size", "2", "--output", "/out.json"]
    if role != "baseline":
        argv.extend(("--mx-url", "http://mx:8001"))

    kwargs = _llm_kwargs(_parse_args([*argv, *worker_args]))

    assert worker_args == expected_args
    moe_kwargs = {
        key: kwargs[key] for key in ("moe_expert_parallel_size", "moe_config") if key in kwargs
    }
    assert moe_kwargs == expected_moe_kwargs
    assert kwargs == {**_llm_kwargs(_parse_args(argv)), **expected_moe_kwargs}
