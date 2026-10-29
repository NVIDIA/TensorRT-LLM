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
"""CPU tests for the host side of the DWDP fabric gather (plan building, knobs, SM reserve)."""

import os

import pytest
import torch

from tensorrt_llm._torch.modules.dwdp import fabric_gather

pytestmark = pytest.mark.cpu_only

_CPU = torch.device("cpu")


def _unset(monkeypatch, name: str) -> None:
    """Unset ``name`` so that monkeypatch also restores it if the code under test sets it."""
    monkeypatch.setenv(name, "")
    monkeypatch.delenv(name)


def test_disabled_by_default(monkeypatch):
    _unset(monkeypatch, "TRTLLM_DWDP_PREFETCH_GATHER")
    assert not fabric_gather.is_enabled()
    monkeypatch.setenv("TRTLLM_DWDP_PREFETCH_GATHER", "0")
    assert not fabric_gather.is_enabled()
    monkeypatch.setenv("TRTLLM_DWDP_PREFETCH_GATHER", "1")
    assert fabric_gather.is_enabled()


def test_build_plan_groups_rows_and_counts_lines(monkeypatch):
    monkeypatch.setenv("TRTLLM_DWDP_GATHER_BLOCKS", "30")
    groups = [
        [(0x1000, 0x9000, 64), (0x2000, 0xA000, 32)],
        [(0x3000, 0xB000, 16)],
    ]
    plan = fabric_gather.build_plan(groups, _CPU)
    assert plan is not None
    dst_ptrs, src_ptrs, lines, row_starts, blocks_per_row = plan
    assert dst_ptrs.tolist() == [0x1000, 0x2000, 0x3000]
    assert src_ptrs.tolist() == [0x9000, 0xA000, 0xB000]
    assert lines.tolist() == [4, 2, 1]
    assert row_starts.tolist() == [0, 2, 3]
    assert all(t.dtype == torch.int64 for t in (dst_ptrs, src_ptrs, lines, row_starts))
    assert blocks_per_row == 15


@pytest.mark.parametrize(
    "descriptor",
    [(0x1008, 0x9000, 16), (0x1000, 0x9004, 16), (0x1000, 0x9000, 24)],
    ids=["dst", "src", "size"],
)
def test_build_plan_rejects_misaligned_descriptors(descriptor):
    assert fabric_gather.build_plan([[(0x2000, 0xA000, 16)], [descriptor]], _CPU) is None


def test_build_plan_rejects_empty_plan():
    assert fabric_gather.build_plan([[]], _CPU) is None


def test_blocks_per_row_never_drops_below_one(monkeypatch):
    monkeypatch.setenv("TRTLLM_DWDP_GATHER_BLOCKS", "4")
    assert fabric_gather.blocks_per_row(16) == 1
    assert fabric_gather.blocks_per_row(2) == 2


def test_compute_side_defaults_respect_explicit_values(monkeypatch):
    monkeypatch.setenv("TRTLLM_CUTEDSL_DENSE_GEMM_SCHEDULER", "static")
    monkeypatch.setenv("TLLM_ROUTING_DISABLE_COOP_LAUNCH", "0")
    fabric_gather.apply_compute_side_defaults()
    assert os.environ["TRTLLM_CUTEDSL_DENSE_GEMM_SCHEDULER"] == "static"
    assert os.environ["TLLM_ROUTING_DISABLE_COOP_LAUNCH"] == "0"
    _unset(monkeypatch, "TRTLLM_CUTEDSL_DENSE_GEMM_SCHEDULER")
    _unset(monkeypatch, "TLLM_ROUTING_DISABLE_COOP_LAUNCH")
    fabric_gather.apply_compute_side_defaults()
    assert os.environ["TRTLLM_CUTEDSL_DENSE_GEMM_SCHEDULER"] == "clc_dynamic"
    assert os.environ["TLLM_ROUTING_DISABLE_COOP_LAUNCH"] == "1"


def test_resident_sms_is_the_block_count_of_a_resident_gather(monkeypatch):
    _unset(monkeypatch, "TRTLLM_CUTEDSL_SM_RESERVE")
    monkeypatch.setenv("TRTLLM_DWDP_GATHER_BLOCKS", "30")
    monkeypatch.setattr(fabric_gather, "available", lambda: True)
    monkeypatch.setattr(fabric_gather, "THREADS", 1024)
    assert fabric_gather.resident_sms() == 30
    # Small blocks share SMs with the compute kernels.
    monkeypatch.setattr(fabric_gather, "THREADS", 128)
    assert fabric_gather.resident_sms() == 0
    # No gather, no reserve.
    monkeypatch.setattr(fabric_gather, "THREADS", 1024)
    monkeypatch.setattr(fabric_gather, "available", lambda: False)
    assert fabric_gather.resident_sms() == 0
    # An explicit count wins, with any gather configuration.
    monkeypatch.setenv("TRTLLM_CUTEDSL_SM_RESERVE", "24")
    assert fabric_gather.resident_sms() == 24


def test_sm_reserve_is_set_explicitly():
    from tensorrt_llm._torch import cute_dsl_utils

    assert cute_dsl_utils.sm_reserve() == 0
    cute_dsl_utils.set_sm_reserve(30)
    try:
        assert cute_dsl_utils.sm_reserve() == 30
    finally:
        cute_dsl_utils.set_sm_reserve(0)
    with pytest.raises(ValueError):
        cute_dsl_utils.set_sm_reserve(-1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_sm_reserve_must_leave_an_sm():
    from tensorrt_llm._torch import cute_dsl_utils

    sm_count = torch.cuda.get_device_properties(torch.cuda.current_device()).multi_processor_count
    with pytest.raises(ValueError, match="must leave at least one"):
        cute_dsl_utils.set_sm_reserve(sm_count)
    assert cute_dsl_utils.sm_reserve() == 0


@pytest.mark.parametrize(
    "full, cluster_size, reserve, expected",
    [(148, 1, 30, 118), (74, 2, 30, 59), (2, 2, 30, 1), (0, 2, 30, 0), (148, 1, 0, 148)],
    ids=["one_sm", "pairs", "floor_one", "unsupported_stays_zero", "no_reserve"],
)
def test_sm_reserve_shrinks_supported_grids_only(full, cluster_size, reserve, expected):
    from tensorrt_llm._torch.cute_dsl_utils import reserved_max_active_clusters

    assert reserved_max_active_clusters(full, cluster_size, reserve) == expected
