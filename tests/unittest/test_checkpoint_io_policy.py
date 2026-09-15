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

import json
from pathlib import Path
from typing import get_args

import pytest
import yaml
from pydantic import BaseModel

from tensorrt_llm._checkpoint_io_policy import (
    AUTO_IO_POLICY,
    CHECKPOINT_IO_POLICIES,
    CHECKPOINT_IO_POLICY_INFO,
    DEMAND_ORDERED_IO_POLICY,
    EXECUTABLE_CHECKPOINT_IO_POLICIES,
    NATIVE_IO_POLICY,
    RANK_STRIPED_IO_POLICIES,
    RANK_STRIPED_IO_POLICY,
    CheckpointIoPolicy,
)
from tensorrt_llm.usage.llmapi_config import manifest_rows

pytestmark = pytest.mark.cpu_only


def test_policy_catalog_is_complete() -> None:
    assert CHECKPOINT_IO_POLICIES == get_args(CheckpointIoPolicy)
    assert tuple(CHECKPOINT_IO_POLICY_INFO) == CHECKPOINT_IO_POLICIES
    assert RANK_STRIPED_IO_POLICIES == (RANK_STRIPED_IO_POLICY, DEMAND_ORDERED_IO_POLICY)
    assert EXECUTABLE_CHECKPOINT_IO_POLICIES == (NATIVE_IO_POLICY, *RANK_STRIPED_IO_POLICIES)
    assert CHECKPOINT_IO_POLICIES == (AUTO_IO_POLICY, *EXECUTABLE_CHECKPOINT_IO_POLICIES)
    for policy, info in CHECKPOINT_IO_POLICY_INFO.items():
        assert info.display_name.strip() and info.description.strip()
        assert info.kind == ("selector" if policy == AUTO_IO_POLICY else "implementation")


def test_policy_field_preserves_string_api() -> None:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

    field = TorchLlmArgs.model_fields["checkpoint_io_policy"]
    assert field.annotation == CheckpointIoPolicy
    assert field.default == AUTO_IO_POLICY
    assert field.json_schema_extra["type"] == str(CheckpointIoPolicy).replace("typing.", "")


def test_policy_catalog_matches_committed_artifacts() -> None:
    class PolicyArgs(BaseModel):
        checkpoint_io_policy: CheckpointIoPolicy = AUTO_IO_POLICY

    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root / "tensorrt_llm/usage/llm_args_golden_manifest.json").read_text())
    entry = next(
        item for item in manifest["TorchLlmArgs"] if item["path"] == "checkpoint_io_policy"
    )
    assert entry == manifest_rows(PolicyArgs)[0]
    reference = yaml.safe_load(
        (root / "tests/unittest/api_stability/references/llm.yaml").read_text()
    )["methods"]["__init__"]["parameters"]["checkpoint_io_policy"]
    assert reference["annotation"] == str(CheckpointIoPolicy).replace("typing.", "")
    assert reference["default"] == AUTO_IO_POLICY
