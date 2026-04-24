# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from tensorrt_llm._mnnvl_utils import MnnvlMoe


class _FakeWorkspace:
    allocated_map = {}
    closed_ptrs = []

    def __init__(self, mapping, ptr, segment_size):
        self.mapping = mapping
        self.ptr = ptr
        self.segment_size = segment_size

    @classmethod
    def get_comm(cls, mapping):
        return mapping.comm

    @classmethod
    def close_mnnvl_memory(cls, ptr):
        cls.closed_ptrs.append(ptr)
        cls.allocated_map.pop(ptr)


@pytest.fixture(autouse=True)
def reset_mnnvl_moe_state():
    previous = {
        "moe_workspace": MnnvlMoe.moe_workspace,
        "moe_prepare_workspace": MnnvlMoe.moe_prepare_workspace,
        "moe_workspace_tensor": MnnvlMoe.moe_workspace_tensor,
        "moe_prepare_workspace_tensor": MnnvlMoe.moe_prepare_workspace_tensor,
        "moe_mapping": MnnvlMoe.moe_mapping,
    }
    _FakeWorkspace.allocated_map = {}
    _FakeWorkspace.closed_ptrs = []
    MnnvlMoe.moe_workspace = None
    MnnvlMoe.moe_prepare_workspace = None
    MnnvlMoe.moe_workspace_tensor = None
    MnnvlMoe.moe_prepare_workspace_tensor = None
    MnnvlMoe.moe_mapping = None
    yield
    for attr, value in previous.items():
        setattr(MnnvlMoe, attr, value)


def test_mnnvl_moe_release_workspaces_closes_mappings_and_drops_tensors():
    mapping = Mock()
    mapping.comm = Mock()
    workspace = _FakeWorkspace(mapping, ptr=11, segment_size=256)
    prepare_workspace = _FakeWorkspace(mapping, ptr=22, segment_size=512)
    _FakeWorkspace.allocated_map = {
        11: (mapping, 1024, [], 0, 0, 0),
        22: (mapping, 2048, [], 0, 0, 0),
    }

    MnnvlMoe.moe_workspace = workspace
    MnnvlMoe.moe_prepare_workspace = prepare_workspace
    MnnvlMoe.moe_workspace_tensor = object()
    MnnvlMoe.moe_prepare_workspace_tensor = object()
    MnnvlMoe.moe_mapping = mapping

    with patch("torch.cuda.synchronize"):
        released_bytes = MnnvlMoe.release_workspaces()

    assert released_bytes == 3072
    assert _FakeWorkspace.closed_ptrs == [11, 22]
    assert not hasattr(workspace, "ptr")
    assert not hasattr(prepare_workspace, "ptr")
    assert MnnvlMoe.moe_workspace is None
    assert MnnvlMoe.moe_prepare_workspace is None
    assert MnnvlMoe.moe_workspace_tensor is None
    assert MnnvlMoe.moe_prepare_workspace_tensor is None
    assert MnnvlMoe.moe_mapping is mapping
    assert mapping.comm.barrier.call_count == 2


def test_mnnvl_moe_restore_workspaces_recreates_missing_workspaces():
    mapping = Mock()
    mapping.comm = Mock()

    def restore_workspace(restored_mapping):
        assert restored_mapping is mapping
        MnnvlMoe.moe_workspace = _FakeWorkspace(mapping, ptr=33, segment_size=128)
        MnnvlMoe.moe_workspace_tensor = object()
        _FakeWorkspace.allocated_map[33] = (mapping, 1024, [], 0, 0, 0)

    def restore_prepare_workspace(restored_mapping):
        assert restored_mapping is mapping
        MnnvlMoe.moe_prepare_workspace = _FakeWorkspace(
            mapping, ptr=44, segment_size=256
        )
        MnnvlMoe.moe_prepare_workspace_tensor = object()
        _FakeWorkspace.allocated_map[44] = (mapping, 2048, [], 0, 0, 0)

    with (
        patch.object(MnnvlMoe, "get_moe_workspaces", side_effect=restore_workspace),
        patch.object(
            MnnvlMoe,
            "get_moe_prepare_workspace",
            side_effect=restore_prepare_workspace,
        ),
        patch("torch.cuda.synchronize") as synchronize,
    ):
        restored_bytes = MnnvlMoe.restore_workspaces(mapping)

    assert restored_bytes == 3072
    assert MnnvlMoe.moe_workspace.ptr == 33
    assert MnnvlMoe.moe_prepare_workspace.ptr == 44
    synchronize.assert_called_once()


def test_mnnvl_moe_workspace_bytes_falls_back_to_segment_size():
    mapping = Mock()
    mapping.comm = Mock()
    MnnvlMoe.moe_workspace = _FakeWorkspace(mapping, ptr=55, segment_size=128)
    MnnvlMoe.moe_prepare_workspace = _FakeWorkspace(
        mapping, ptr=66, segment_size=256
    )

    assert MnnvlMoe.workspace_bytes() == 384
