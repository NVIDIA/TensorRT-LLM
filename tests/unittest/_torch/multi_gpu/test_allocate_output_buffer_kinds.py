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
"""Multi-GPU tests for allocate_output across all three BufferKind values.

Each test allocates an output tensor with DEFAULT / USERBUFFERS / NCCL_WINDOW,
fills it with known values, runs an AllReduce, and verifies correctness.
"""

import os
import pickle
import sys
import traceback

import cloudpickle
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
import tensorrt_llm.bindings.internal.userbuffers as ub
from tensorrt_llm._torch.distributed import (
    AllReduce,
    AllReduceFusionOp,
    AllReduceParams,
    AllReduceStrategy,
    userbuffers_allreduce_finalize,
)
from tensorrt_llm.mapping import Mapping

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)

# needed since we reuse the mpi executor pool, first test running will leak a thread
pytestmark = pytest.mark.threadleak(enabled=False)

_SKIP = "skip"
_FAILED_PREFIX = "FAILED:"


def run_single_rank(tp_size, single_rank_func, *args):
    """Run single_rank_func in one MPI worker and return its result.

    Returns a string starting with _FAILED_PREFIX on any exception so that mpi4py
    never needs to pickle an exception with __traceback__ frames (which can
    contain unpicklable torch.ops references via AllReduce locals).

    torch.inference_mode() is applied here rather than as a decorator on the
    individual _run_* helpers to avoid pickling failures.  With cloudpickle
    register_pickle_by_value, by-value serialization of a function whose
    globals dict contains itself causes cloudpickle 3.1.2 to encounter
    torch.ops C++ objects (which are not picklable), yielding
    "TypeError: cannot pickle '_Ops' object".  Keeping the helpers as plain
    functions and restricting direct torch.ops calls to nested inner
    functions (per the pattern in test_allreduce.py) avoids this cycle.
    """
    rank = -1
    try:
        rank = tensorrt_llm.mpi_rank()
        torch.cuda.set_device(rank)
        with torch.inference_mode():
            return single_rank_func(tp_size, rank, *args)
    except BaseException:
        err = traceback.format_exc()
        sys.stderr.write(f"Worker error on rank {rank}:\n{err}\n")
        sys.stderr.flush()
        return f"{_FAILED_PREFIX}\n{err}"


def _run_allreduce_default(tp_size, tp_rank, seq_len, hidden_size):
    """AllReduce on a DEFAULT-allocated buffer."""
    from tensorrt_llm.bindings.internal.thop import BufferKind

    def _allocate(ref, kind, tp_group):
        return torch.ops.trtllm.allocate_output(ref, kind, tp_group)

    dtype = torch.bfloat16
    mapping = Mapping(world_size=tp_size, tp_size=tp_size, rank=tp_rank)

    ref = torch.zeros(seq_len, hidden_size, dtype=dtype, device="cuda")
    out, actual_kind_int = _allocate(ref, int(BufferKind.DEFAULT), mapping.tp_group)
    assert actual_kind_int == int(BufferKind.DEFAULT)

    out.fill_(1.0)

    allreduce = AllReduce(mapping=mapping, strategy=AllReduceStrategy.NCCL)
    result = allreduce(out, all_reduce_params=AllReduceParams(fusion_op=AllReduceFusionOp.NONE))

    expected = torch.full((seq_len, hidden_size), tp_size, dtype=dtype, device="cuda")
    torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)
    return True


def _run_allreduce_userbuffers(tp_size, tp_rank, seq_len, hidden_size):
    """AllReduce+RMSNorm fusion on a USERBUFFERS-allocated input buffer.

    UB is designed for fused allreduce ops (RESIDUAL_RMS_NORM and variants).
    Returns True vacuously when UB is not supported on the hardware.
    """
    from tensorrt_llm.bindings.internal.thop import BufferKind

    if not ub.ub_supported():
        return True

    dtype = torch.bfloat16
    mapping = Mapping(world_size=tp_size, tp_size=tp_size, rank=tp_rank)
    eps = 1e-5

    # Build the input in a UB-backed tensor via allocate_output.
    # Each rank contributes 1.0; after allreduce the sum is tp_size.
    residual = torch.zeros(seq_len, hidden_size, dtype=dtype, device="cuda")
    norm_weight = torch.ones(hidden_size, dtype=dtype, device="cuda")

    def _allocate(ref, kind, tp_group):
        return torch.ops.trtllm.allocate_output(ref, kind, tp_group)

    if not ub.ub_is_initialized():
        # Initialize once with the maximum size needed across all parametrized
        # cases.  UserBuffersManager reuses pool buffers by size; initializing
        # with a larger buffer_size_ later can cause OOB reuse of smaller
        # previously-allocated buffers.
        max_bytes = 256 * 4096 * torch.tensor([], dtype=dtype).element_size()
        ub.initialize_userbuffers_manager(
            tp_size, 1, 1, tp_rank, torch.cuda.device_count(), max_bytes
        )

    ref = torch.zeros(seq_len, hidden_size, dtype=dtype, device="cuda")
    out, actual_kind_int = _allocate(ref, int(BufferKind.USERBUFFERS), mapping.tp_group)
    assert actual_kind_int == int(BufferKind.USERBUFFERS)

    out.fill_(1.0)

    allreduce = AllReduce(mapping=mapping, strategy=AllReduceStrategy.UB)
    ar_params = AllReduceParams(
        strategy=AllReduceStrategy.UB,
        fusion_op=AllReduceFusionOp.RESIDUAL_RMS_NORM,
        residual=residual,
        norm_weight=norm_weight,
        eps=eps,
    )
    norm_out, residual_out = allreduce(out, all_reduce_params=ar_params)
    norm_out = norm_out.clone()
    residual_out = userbuffers_allreduce_finalize(residual_out, False)

    # Each rank contributed 1.0 → allreduce sum = tp_size.
    # residual_out = allreduced + zeros = tp_size everywhere.
    expected_residual = torch.full((seq_len, hidden_size), tp_size, dtype=dtype, device="cuda")
    # norm_out = rms_norm(tp_size * ones) = 1.0 everywhere (constant tensors
    # normalize to ±1; with all positive values this is exactly 1.0).
    expected_norm = torch.ones(seq_len, hidden_size, dtype=dtype, device="cuda")

    torch.testing.assert_close(residual_out, expected_residual, atol=5e-1, rtol=1e-2)
    torch.testing.assert_close(norm_out, expected_norm, atol=5e-1, rtol=1e-2)
    return True


def _run_allreduce_nccl_window(tp_size, tp_rank, seq_len, hidden_size):
    """AllReduce on a NCCL_WINDOW-allocated buffer; skip if allocation falls back."""
    from tensorrt_llm.bindings.internal.thop import BufferKind

    def _allocate(ref, kind, tp_group):
        return torch.ops.trtllm.allocate_output(ref, kind, tp_group)

    dtype = torch.bfloat16
    mapping = Mapping(world_size=tp_size, tp_size=tp_size, rank=tp_rank)

    ref = torch.zeros(seq_len, hidden_size, dtype=dtype, device="cuda")
    out, actual_kind_int = _allocate(ref, int(BufferKind.NCCL_WINDOW), mapping.tp_group)

    if actual_kind_int != int(BufferKind.NCCL_WINDOW):
        # Window allocation not available in this environment; skip gracefully.
        return _SKIP

    out.fill_(1.0)

    allreduce = AllReduce(mapping=mapping, strategy=AllReduceStrategy.NCCL_SYMMETRIC)
    result = allreduce(out, all_reduce_params=AllReduceParams(fusion_op=AllReduceFusionOp.NONE))

    expected = torch.full((seq_len, hidden_size), tp_size, dtype=dtype, device="cuda")
    torch.testing.assert_close(result, expected, atol=1e-2, rtol=1e-2)
    return True


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
@pytest.mark.parametrize(
    "seq_len,hidden_size",
    [
        pytest.param(16, 512, id="seqlen16_hidden512"),
        pytest.param(256, 4096, id="seqlen256_hidden4096"),
    ],
)
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_allreduce_default_buffer(seq_len, hidden_size, mpi_pool_executor):
    tp_size = mpi_pool_executor.num_workers
    results = list(
        mpi_pool_executor.map(
            run_single_rank,
            *zip(*[(tp_size, _run_allreduce_default, seq_len, hidden_size)] * tp_size),
        )
    )
    for r in results:
        assert r is True, r


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
@pytest.mark.parametrize(
    "seq_len,hidden_size",
    [
        pytest.param(16, 512, id="seqlen16_hidden512"),
        pytest.param(256, 4096, id="seqlen256_hidden4096"),
    ],
)
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_allreduce_userbuffers_buffer(seq_len, hidden_size, mpi_pool_executor):
    tp_size = mpi_pool_executor.num_workers
    results = list(
        mpi_pool_executor.map(
            run_single_rank,
            *zip(*[(tp_size, _run_allreduce_userbuffers, seq_len, hidden_size)] * tp_size),
        )
    )
    for r in results:
        assert r is True, r


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Requires at least 2 GPUs")
@pytest.mark.parametrize(
    "seq_len,hidden_size",
    [
        pytest.param(16, 512, id="seqlen16_hidden512"),
        pytest.param(256, 4096, id="seqlen256_hidden4096"),
    ],
)
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_allreduce_nccl_window_buffer(seq_len, hidden_size, mpi_pool_executor):
    tp_size = mpi_pool_executor.num_workers
    results = list(
        mpi_pool_executor.map(
            run_single_rank,
            *zip(*[(tp_size, _run_allreduce_nccl_window, seq_len, hidden_size)] * tp_size),
        )
    )
    for r in results:
        assert r is True or r == _SKIP, f"Unexpected result: {r}"
