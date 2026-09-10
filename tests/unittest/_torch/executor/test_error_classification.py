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

import importlib.util
import sys
from pathlib import Path

import pytest

_MODULE_PATH = (
    Path(__file__).resolve().parents[4] / "tensorrt_llm/_torch/pyexecutor/error_classification.py"
)
_SPEC = importlib.util.spec_from_file_location("error_classification_under_test", _MODULE_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_ERROR_CLASSIFICATION = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _ERROR_CLASSIFICATION
_SPEC.loader.exec_module(_ERROR_CLASSIFICATION)

ErrorBudget = _ERROR_CLASSIFICATION.ErrorBudget
classify_error = _ERROR_CLASSIFICATION.classify_error


@pytest.mark.parametrize(
    "message",
    [
        "CUDA context destroyed",
        "cudaErrorContextIsDestroyed during collective setup",
        "CUDA error: cudaErrorNoDevice on rank 37",
        "NVRM: Xid 79, GPU has fallen off the bus",
        "CUDA context is destroyed after device reset",
        "CUDA context was destroyed after device reset",
        "CUDA context is unrecoverable after device reset",
        "Unrecoverable error in the engine",
        "ncclCommAbort returned during communicator teardown",
        "NCCL communicator was aborted during PP send",
        "NVSHMEM peer unreachable for EP rank 12",
        "MPI rank terminated with signal 9",
        "MPI worker exited unexpectedly",
    ],
)
def test_wide_ep_immediate_fatal_patterns(message):
    assert classify_error(message) == "immediate_fatal"


@pytest.mark.parametrize(
    "message",
    [
        "NCCL timeout",
        "AlltoAll timeout waiting for completion_flags[3][7]",
        "AlltoAll watchdog timed out waiting for peer rank 7",
        "completion_flags timeout",
        "NCCL operation timed out during all_reduce",
        "deep_ep buffer barrier hang",
        "deepep buffer barrier hang",
        "DeepEP Buffer.__del__ hung in intranode::barrier",
        "Symmetric memory access violation reading dead peer",
        "RDMA timeout on cross-node transfer",
        "NIXL transfer failed: remote peer closed",
        "NIXL transfer entered error state",
        "NIXL transfer wait timed out after 5000 ms",
    ],
)
def test_wide_ep_severe_patterns(message):
    assert classify_error(message) == "severe"


@pytest.mark.parametrize(
    "message",
    [
        "AlltoAll slow path observed for rank 3",
        "NCCL retry scheduled after transient transport hiccup",
        "ECC correctable error threshold warning",
        "Application marked the request unrecoverable but retryable",
    ],
)
def test_wide_ep_nonfatal_signals_remain_transient(message):
    assert classify_error(message) == "transient"


def test_error_budget_charges_wide_ep_severe_patterns():
    budget = ErrorBudget(recovery_rate=0.0)

    assert not budget.consume("NIXL transfer failed: remote peer closed")
    assert budget.consume("AlltoAll timeout waiting for completion_flags[0][2]")


def test_error_budget_bypasses_budget_for_wide_ep_fatal_patterns():
    budget = ErrorBudget(recovery_rate=0.0)

    assert budget.consume("MPI rank terminated with signal 9")
    assert budget.budget == 1.0
