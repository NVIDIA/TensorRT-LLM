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

from types import SimpleNamespace

import torch

from tensorrt_llm._torch.autotuner import OptimizationProfile
from tensorrt_llm._torch.custom_ops.torch_custom_ops import AllReduceRunner
from tensorrt_llm._torch.distributed import ops as distributed_ops
from tensorrt_llm.functional import AllReduceStrategy


def test_sm103_excludes_nccl_symmetric_from_auto(monkeypatch):
    monkeypatch.setattr(distributed_ops, "get_sm_version", lambda: 103)

    assert not distributed_ops._nccl_symmetric_auto_tactic_supported()


def test_other_capability_preserves_nccl_symmetric_auto(monkeypatch):
    monkeypatch.setattr(distributed_ops, "get_sm_version", lambda: 100)

    assert distributed_ops._nccl_symmetric_auto_tactic_supported()


def test_unavailable_capability_preserves_existing_policy(monkeypatch):
    def fail_device_discovery():
        raise RuntimeError("CUDA device discovery unavailable")

    monkeypatch.setattr(distributed_ops, "get_sm_version", fail_device_discovery)

    assert distributed_ops._nccl_symmetric_auto_tactic_supported()


def test_sm103_auto_tactics_and_cache_miss_use_safe_collectives(monkeypatch):
    monkeypatch.setattr(distributed_ops, "_nccl_symmetric_auto_tactic_supported", lambda: False)
    runner = AllReduceRunner(
        tp_size=4,
        group=[0, 1, 2, 3],
        input_dtype=torch.bfloat16,
        op=0,
        eps=1e-6,
        trigger_completion_at_end=False,
    )
    tactics = runner.get_valid_tactics([torch.empty(4, 128)], OptimizationProfile())

    assert AllReduceStrategy.NCCL_SYMMETRIC.value not in tactics
    assert AllReduceStrategy.NCCL.value in tactics
    assert AllReduceStrategy.ONESHOT.value in tactics
    assert runner._cache_miss_fallback_tactic() == AllReduceStrategy.NCCL.value


def test_non_sm103_auto_preserves_symmetric_tactic_and_fallback(monkeypatch):
    monkeypatch.setattr(distributed_ops, "_nccl_symmetric_auto_tactic_supported", lambda: True)
    runner = AllReduceRunner(
        tp_size=4,
        group=[0, 1, 2, 3],
        input_dtype=torch.bfloat16,
        op=0,
        eps=1e-6,
        trigger_completion_at_end=False,
    )
    tactics = runner.get_valid_tactics([torch.empty(4, 128)], OptimizationProfile())

    assert AllReduceStrategy.NCCL_SYMMETRIC.value in tactics
    assert runner._cache_miss_fallback_tactic() == AllReduceStrategy.NCCL_SYMMETRIC.value


def test_sm103_auto_does_not_request_nccl_window_output(monkeypatch):
    monkeypatch.setattr(distributed_ops, "_nccl_symmetric_auto_tactic_supported", lambda: False)
    monkeypatch.setattr(distributed_ops, "_NCCL_SYMMETRIC_ZERO_COPY", True)
    allreduce = object.__new__(distributed_ops.AllReduce)
    allreduce.mapping = SimpleNamespace(tp_size=4)
    allreduce._disable_mpi = False
    allreduce.strategy = AllReduceStrategy.AUTO

    assert not allreduce.uses_nccl_symmetric_memory_window()

    allreduce.strategy = AllReduceStrategy.NCCL
    assert allreduce.uses_nccl_symmetric_memory_window()

    allreduce.strategy = AllReduceStrategy.NCCL_SYMMETRIC
    assert allreduce.uses_nccl_symmetric_memory_window()


def test_non_sm103_auto_preserves_nccl_window_output(monkeypatch):
    monkeypatch.setattr(distributed_ops, "_nccl_symmetric_auto_tactic_supported", lambda: True)
    monkeypatch.setattr(distributed_ops, "_NCCL_SYMMETRIC_ZERO_COPY", True)
    allreduce = object.__new__(distributed_ops.AllReduce)
    allreduce.mapping = SimpleNamespace(tp_size=4)
    allreduce._disable_mpi = False
    allreduce.strategy = AllReduceStrategy.AUTO

    assert allreduce.uses_nccl_symmetric_memory_window()
