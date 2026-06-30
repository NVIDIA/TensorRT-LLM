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
"""Unit tests for run_concurrently's CPU-thread / pool bounding.

run_concurrently must not oversubscribe the host CPU when multiple ranks are
co-located on one node: it bounds both the worker-pool size and the per-op
intra-op thread count to this rank's fair share of cores, and restores the
process-wide thread count afterwards.
"""

import os
from unittest import mock

import pytest
import torch

from tensorrt_llm._torch.models import modeling_utils


def _available_cpus():
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


@pytest.fixture(autouse=True)
def _restore_thread_state():
    original = torch.get_num_threads()
    yield
    torch.set_num_threads(original)


def test_bounds_intra_op_threads_to_per_rank_share(monkeypatch):
    """With N co-located ranks, intra-op threads during tasks must not exceed
    this rank's core share (cores // N)."""
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    before = torch.get_num_threads()
    observed = []

    def task(x):
        observed.append(torch.get_num_threads())
        return x

    with mock.patch.object(modeling_utils, "local_mpi_size", return_value=8):
        modeling_utils.run_concurrently(task, [(i,) for i in range(16)])

    per_rank = max(1, _available_cpus() // 8)
    assert observed, "task was never executed"
    assert max(observed) <= per_rank
    # Thread count is restored after the call.
    assert torch.get_num_threads() == before


def test_respects_user_omp_num_threads(monkeypatch):
    """An explicit user OMP_NUM_THREADS must not be overridden."""
    monkeypatch.setenv("OMP_NUM_THREADS", "5")
    torch.set_num_threads(5)
    observed = []

    with mock.patch.object(modeling_utils, "local_mpi_size", return_value=8):
        modeling_utils.run_concurrently(
            lambda x: observed.append(torch.get_num_threads()), [(i,) for i in range(4)]
        )

    assert observed and all(t == 5 for t in observed)
    assert torch.get_num_threads() == 5


def test_results_are_reduced_and_complete():
    """Functional check: every task runs and reduce_func sees every result."""
    seen = []
    with mock.patch.object(modeling_utils, "local_mpi_size", return_value=1):
        modeling_utils.run_concurrently(
            lambda x: x * 2, [(i,) for i in range(32)], reduce_func=lambda r: seen.append(r)
        )
    assert sorted(seen) == [i * 2 for i in range(32)]


def test_restores_thread_count_when_task_raises(monkeypatch):
    """The process-wide thread count is restored even if a task raises."""
    monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
    before = torch.get_num_threads()

    def boom(x):
        raise RuntimeError("boom")

    with mock.patch.object(modeling_utils, "local_mpi_size", return_value=8):
        with pytest.raises(RuntimeError, match="boom"):
            modeling_utils.run_concurrently(boom, [(1,)])

    assert torch.get_num_threads() == before


def test_rejects_non_positive_num_workers():
    """An explicit non-positive num_workers is rejected with a clear error."""
    with mock.patch.object(modeling_utils, "local_mpi_size", return_value=1):
        with pytest.raises(ValueError, match="num_workers"):
            modeling_utils.run_concurrently(lambda x: x, [(1,)], num_workers=0)
