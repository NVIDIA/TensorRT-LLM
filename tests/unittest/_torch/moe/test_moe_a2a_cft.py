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

from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.communication.moe_alltoall import (
    get_force_cft as get_force_cft_standalone,
)
from tensorrt_llm._torch.moe.fused_moe.communication.moe_alltoall import (
    should_use_cft as should_use_cft_standalone,
)
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import (
    FORCE_CFT_ENV,
    NVLinkOneSided,
    get_force_cft,
    should_use_cft,
)

# Every check in this file is pure mock/env logic with no GPU work, so it can
# ride the CPU-only CI stage. tests/unittest/conftest.py drops any file without
# this marker from CPU stages.
pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("2", None),
        ("true", None),
        (" 1 ", None),
        ("0", False),
        ("1", True),
    ],
)
def test_get_force_cft(monkeypatch: pytest.MonkeyPatch, value: str | None, expected: bool | None):
    if value is None:
        monkeypatch.delenv(FORCE_CFT_ENV, raising=False)
    else:
        monkeypatch.setenv(FORCE_CFT_ENV, value)

    assert get_force_cft() is expected
    assert get_force_cft_standalone() is expected


@pytest.mark.parametrize(
    ("can_use_cft", "force_cft", "runtime_max_tokens_per_rank", "expected"),
    [
        (True, None, 128, True),
        (True, None, 129, False),
        (True, False, 1, False),
        (True, True, 129, True),
        (False, True, 1, False),
        (False, None, 1, False),
    ],
)
def test_should_use_cft(
    can_use_cft: bool,
    force_cft: bool | None,
    runtime_max_tokens_per_rank: int,
    expected: bool,
):
    assert should_use_cft(can_use_cft, force_cft, 128, runtime_max_tokens_per_rank) is expected
    assert (
        should_use_cft_standalone(can_use_cft, force_cft, 128, runtime_max_tokens_per_rank)
        is expected
    )


def test_destroy_releases_cft_manager_before_workspace_allocation(
    monkeypatch: pytest.MonkeyPatch,
):
    workspace_key = ("cft-workspace",)
    workspace = object()
    mnnvl_mem = object()
    workspace_state = {
        "cft_initialized": True,
        "workspace": workspace,
        "ep_rank": 3,
        "mnnvl_mem": mnnvl_mem,
    }
    monkeypatch.setattr(NVLinkOneSided, "_WORKSPACES", {workspace_key: workspace_state})
    monkeypatch.setattr(NVLinkOneSided, "_WORKSPACE_REFCOUNTS", {workspace_key: 1})
    monkeypatch.setattr(NVLinkOneSided, "_WORKSPACE", workspace_state)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    release_cft_manager = MagicMock()

    def verify_workspace_is_alive(workspace_arg: object, ep_rank: int) -> None:
        assert workspace_arg is workspace
        assert ep_rank == 3
        assert NVLinkOneSided._WORKSPACES[workspace_key]["mnnvl_mem"] is mnnvl_mem

    release_cft_manager.side_effect = verify_workspace_is_alive
    monkeypatch.setattr(torch.ops.trtllm, "moe_a2a_cft_release", release_cft_manager, raising=False)

    comm = NVLinkOneSided.__new__(NVLinkOneSided)
    comm._destroyed = False
    comm._workspace_key = workspace_key
    comm._workspace_state = workspace_state
    comm._workspace_lifecycle = None
    comm._workspace_registered = True
    comm.mnnvl_mem = mnnvl_mem
    comm.workspace = workspace
    comm._dispatch_state = {"phase": "idle"}

    comm.destroy()

    release_cft_manager.assert_called_once_with(workspace, 3)
    assert workspace_key not in NVLinkOneSided._WORKSPACES
    assert workspace_key not in NVLinkOneSided._WORKSPACE_REFCOUNTS
    assert NVLinkOneSided._WORKSPACE is None
    assert workspace_state == {}
    assert comm.mnnvl_mem is None
    assert comm.workspace is None
