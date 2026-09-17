# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capacity checks for stable one-sided dispatch/combine workspace regions."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.communication import nvlink_one_sided as a2a
from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided


@pytest.fixture(params=[False, True], ids=["fence", "cft"])
def comm(request):
    instance = NVLinkOneSided.__new__(NVLinkOneSided)
    instance._destroyed = True
    instance._workspace_state = {}
    instance._dispatch_state = {"phase": "idle"}
    instance.ep_size = 4
    instance.max_num_tokens_per_rank = 32
    instance.can_use_cft_counted_writes = request.param
    instance.workspace_size_per_rank = (
        128 + (2 + int(request.param)) * 4 * 32 * 128 * 2 + 4 * 32 * 8 * 8
    )
    instance.PAYLOAD_DATA_OFFSET_INDEX = 0
    instance._workspace_lifecycle = SimpleNamespace(metainfo=torch.tensor([128]))
    return instance


def test_fixed_offset_is_shared_and_independent_of_dispatch_layout(comm):
    offset = comm._reserve_combine_region(128, torch.bfloat16)
    for count in (1, 9, 32, 2):
        for quantized in (False, True):
            payloads = [
                torch.empty(
                    (0, 64 if quantized else 128),
                    dtype=torch.uint8 if quantized else torch.bfloat16,
                )
            ]
            if quantized:
                payloads.append(torch.empty((0, 8), dtype=torch.uint8))
            payloads += [torch.empty((0, 8), dtype=torch.int32), torch.empty((0, 8))]
            assert comm._check_dispatch_region(payloads, count) <= offset
            assert comm._reserve_combine_region(128, torch.bfloat16) == offset
    assert offset % 128 == 0
    regions = 2 if comm.can_use_cft_counted_writes else 1
    assert offset + regions * 4 * 32 * 128 * 2 <= comm.workspace_size_per_rank


@pytest.mark.parametrize("count", [0, -1, 33])
def test_rejects_runtime_count_outside_capacity(comm, count):
    with pytest.raises(ValueError, match="capacity"):
        comm._check_dispatch_region([torch.empty((0, 128))], count)


def test_rejects_dispatch_overlap_before_native_write(comm):
    comm._reserve_combine_region(128, torch.bfloat16)
    with pytest.raises(ValueError, match="overlaps"):
        comm._check_dispatch_region([torch.empty((0, 1024))], 32)


@pytest.mark.parametrize("hidden,dtype", [(64, torch.bfloat16), (128, torch.float32)])
def test_rejects_shared_combine_layout_change(comm, hidden, dtype):
    comm._reserve_combine_region(128, torch.bfloat16)
    with pytest.raises(ValueError, match="stable combine"):
        comm._reserve_combine_region(hidden, dtype)


def test_rejects_insufficient_combine_capacity(comm):
    comm.workspace_size_per_rank = 256
    with pytest.raises(ValueError, match="too small"):
        comm._reserve_combine_region(128, torch.bfloat16)
    assert not comm._workspace_state


def test_lazy_reservation_checks_first_dispatch(comm):
    comm._dispatch_state["dispatch_payload_end"] = comm.workspace_size_per_rank
    with pytest.raises(ValueError, match="overlaps"):
        comm._reserve_combine_region(128, torch.bfloat16)
    assert not comm._workspace_state


def test_lazy_reservation_guards_following_dispatch(comm):
    payload = torch.empty((0, 64), dtype=torch.uint8)
    comm._dispatch_state["dispatch_payload_end"] = comm._check_dispatch_region([payload], 9)
    offset = comm._reserve_combine_region(128, torch.bfloat16)
    assert comm._check_dispatch_region([payload], 32) <= offset
    with pytest.raises(ValueError, match="overlaps"):
        comm._check_dispatch_region([torch.empty((0, 1024))], 32)


@pytest.mark.parametrize("use_cft", [False, True])
def test_failed_reservation_does_not_publish_workspace(monkeypatch, use_cft):
    def init_base(self, mapping):
        self.mapping = mapping
        self.ep_size = mapping.world_size
        self.ep_rank = mapping.rank

    monkeypatch.setattr(a2a.Communication, "__init__", init_base)
    monkeypatch.setattr(NVLinkOneSided, "_init_constants", classmethod(lambda cls: None))
    monkeypatch.setattr(NVLinkOneSided, "get_aux_data_size", staticmethod(lambda *args: 128))
    for name in (
        "PAYLOAD_DATA_OFFSET_INDEX",
        "FLAG_VAL_OFFSET_INDEX",
        "DISPATCH_COMPLETION_FLAGS_OFFSET_INDEX",
        "COMBINE_COMPLETION_FLAGS_OFFSET_INDEX",
    ):
        monkeypatch.setattr(NVLinkOneSided, name, 0)
    monkeypatch.setattr(NVLinkOneSided, "_WORKSPACES", {})
    monkeypatch.setattr(NVLinkOneSided, "_WORKSPACE_REFCOUNTS", {})
    monkeypatch.setattr(NVLinkOneSided, "_WORKSPACE", None)
    for name in ("MnnvlMemory", "CftMnnvlMemory"):
        monkeypatch.setattr(a2a, name, MagicMock())
    monkeypatch.setattr(
        torch.ops.trtllm,
        "moe_a2a_initialize",
        MagicMock(return_value=torch.tensor([128])),
        raising=False,
    )
    lifecycle = SimpleNamespace(metainfo=torch.tensor([128]), register=MagicMock())
    monkeypatch.setattr(
        a2a,
        "_MnnvlAlltoAllWorkspaceLifecycle",
        SimpleNamespace(get_or_create=MagicMock(return_value=lifecycle)),
    )
    monkeypatch.setenv("TRTLLM_MOE_A2A_WORKSPACE_MB", "1")
    monkeypatch.delenv("TRTLLM_MOE_A2A_FORCE_CFT", raising=False)
    with pytest.raises(ValueError, match="too small"):
        NVLinkOneSided(
            SimpleNamespace(world_size=4, rank=0, has_cp_helix=lambda: False),
            256,
            8,
            1024,
            hidden_size=1024,
            dtype=torch.bfloat16,
            can_use_cft_counted_writes=use_cft,
        )
    lifecycle.register.assert_not_called()
    assert not NVLinkOneSided._WORKSPACES
    assert not NVLinkOneSided._WORKSPACE_REFCOUNTS
    assert NVLinkOneSided._WORKSPACE is None
