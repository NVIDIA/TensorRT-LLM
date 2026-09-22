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
"""Multi-rank regression tests for zero-local-KV Helix post-processing.

NCCL neutralizes empty rows in the compiled input reformat, while fifo v2 does
it in the native sender before protocol packing. These tests poison empty rows
with NaN, drive the complete exchange and combine, and compare with a float64
reference. A negative control proves that the poison is observable when the
mask is withheld.
"""

import pickle
import sys
import time

import _torch.attention.multi_gpu.helix_test_utils as helix_utils
import cloudpickle
import pytest
import torch
from _torch.attention.multi_gpu.helix_test_utils import parse_comms_medium, run_single_rank
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
from utils.util import skip_pre_blackwell

import tensorrt_llm
from tensorrt_llm._torch.attention.attention import _helix_post_process
from tensorrt_llm.mapping import CpType, Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
cloudpickle.register_pickle_by_value(helix_utils)
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

WORLD_SIZE = 2
NUM_HEADS = 6
VALUE_DIM = 512
BARRIER_TIMEOUT_S = 300.0
TOLERANCE = 2e-2


def _bounded_barrier(comm, label: str, timeout_s: float = BARRIER_TIMEOUT_S) -> None:
    """Fail instead of hanging when a peer never reaches a graph-capture phase."""
    request = comm.Ibarrier()
    deadline = time.monotonic() + timeout_s
    while not request.Test():
        if time.monotonic() > deadline:
            request.Cancel()
            raise TimeoutError(
                f"rank {comm.Get_rank()} waited {timeout_s:.0f}s at the '{label}' barrier; "
                "a peer never arrived"
            )
        time.sleep(0.01)


def _zero_kv_mask(
    rank: int, world_size: int, num_tokens: int, device: torch.device
) -> torch.Tensor:
    """Build a different non-degenerate zero-KV mask on every rank."""
    token_idx = torch.arange(num_tokens, device=device)
    mask = token_idx == rank
    mask |= (token_idx >= world_size) & (token_idx % (2 * world_size + 1) == rank)
    return mask


def _reference(
    all_o: list[torch.Tensor],
    all_stats: list[torch.Tensor],
    all_mask: list[torch.Tensor],
    destination_rank: int,
    cp_size: int,
) -> torch.Tensor:
    """Combine every rank's pre-sanitize tensors in float64."""
    num_tokens = all_o[0].shape[0]
    partial_o = torch.stack(
        [
            tensor.view(num_tokens, cp_size, NUM_HEADS, VALUE_DIM)[:, destination_rank]
            for tensor in all_o
        ]
    )
    stats = torch.stack(
        [
            tensor.view(num_tokens, cp_size, NUM_HEADS, 2)[:, destination_rank]
            for tensor in all_stats
        ]
    )
    mask = torch.stack(all_mask)[:, :, None]

    partial_o = torch.where(
        mask[..., None], torch.zeros((), dtype=torch.float64), partial_o.double()
    )
    softmax_max = torch.where(
        mask, torch.full((), float("-inf"), dtype=torch.float64), stats[..., 0].double()
    )
    softmax_sum = torch.where(mask, torch.zeros((), dtype=torch.float64), stats[..., 1].double())
    weight = softmax_sum * torch.exp(softmax_max - softmax_max.max(dim=0).values)
    weight = weight / weight.sum(dim=0)
    return (partial_o * weight[..., None]).sum(dim=0).reshape(num_tokens, NUM_HEADS * VALUE_DIM)


def _zero_kv_rank(
    rank: int,
    world_size: int,
    num_tokens: int,
    comms_medium: str,
    use_cuda_graph: bool,
    withhold_mask: bool,
) -> tuple[int, float]:
    """Run one rank and return its output NaN count and maximum reference error."""
    comm = tensorrt_llm.mpi_comm()
    device = torch.device("cuda", torch.cuda.current_device())

    torch.manual_seed(1234 + rank)
    partial_o = torch.randn(
        num_tokens, world_size * NUM_HEADS * VALUE_DIM, device=device, dtype=torch.bfloat16
    )
    stats = torch.empty(num_tokens, world_size * NUM_HEADS, 2, device=device, dtype=torch.float32)
    stats[..., 0].normal_(0.0, 2.0)
    stats[..., 1].uniform_(0.5, 2.0)

    mask = _zero_kv_mask(rank, world_size, num_tokens, device)
    partial_o[mask] = float("nan")
    stats[mask, :, 0] = 1e4
    stats[mask, :, 1] = 7.0

    all_o = [torch.from_numpy(value) for value in comm.allgather(partial_o.float().cpu().numpy())]
    all_stats = [torch.from_numpy(value) for value in comm.allgather(stats.cpu().numpy())]
    all_mask = [torch.from_numpy(value) for value in comm.allgather(mask.cpu().numpy())]
    reference = _reference(all_o, all_stats, all_mask, rank, world_size).to(device)

    use_nccl_for_alltoall, fifo_version = parse_comms_medium(comms_medium)
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        cp_size=world_size,
        cp_config={
            "cp_type": CpType.HELIX,
            "use_nccl_for_alltoall": use_nccl_for_alltoall,
            "fifo_version": fifo_version,
        },
    )
    zero_kv_mask = None if withhold_mask else mask

    if use_cuda_graph:
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            for _ in range(3):
                _helix_post_process(
                    partial_o,
                    stats,
                    mapping,
                    NUM_HEADS,
                    VALUE_DIM,
                    zero_kv_mask=zero_kv_mask,
                )
        torch.cuda.current_stream().wait_stream(side_stream)
        torch.cuda.synchronize()
        _bounded_barrier(comm, "before capture")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = _helix_post_process(
                partial_o,
                stats,
                mapping,
                NUM_HEADS,
                VALUE_DIM,
                zero_kv_mask=zero_kv_mask,
            )
        torch.cuda.synchronize()
        _bounded_barrier(comm, "after capture")
        graph.replay()
        torch.cuda.synchronize()
    else:
        output = _helix_post_process(
            partial_o.clone(),
            stats.clone(),
            mapping,
            NUM_HEADS,
            VALUE_DIM,
            zero_kv_mask=zero_kv_mask,
        )

    output = output.double()
    nan_count = int(torch.isnan(output).sum())
    max_error = float("nan") if nan_count else float((output - reference).abs().max())
    return nan_count, max_error


def _launch(
    num_tokens: int,
    comms_medium: str,
    *,
    use_cuda_graph: bool = False,
    withhold_mask: bool = False,
) -> list[tuple[int, float]]:
    """Run the regression on both ranks and collect their results."""
    args = (_zero_kv_rank, WORLD_SIZE, num_tokens, comms_medium, use_cuda_graph, withhold_mask)
    with MPIPoolExecutor(max_workers=WORLD_SIZE) as executor:
        return list(executor.map(run_single_rank, *zip(*[args] * WORLD_SIZE)))


def _assert_neutral(results: list[tuple[int, float]]) -> None:
    """Require finite output matching the reference on every rank."""
    for rank, (nan_count, max_error) in enumerate(results):
        assert nan_count == 0, (
            f"rank {rank}: {nan_count} NaN in the output, so a zero-local-KV row "
            "reached the combine unsanitized"
        )
        assert max_error < TOLERANCE, f"rank {rank}: max|out - ref| = {max_error:.3e}"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs to run this test")
@skip_pre_blackwell
@pytest.mark.parametrize(("comms_medium", "num_tokens"), [("nccl", 17), ("fifo_v2", 17)])
def test_zero_kv_rows_are_neutral(comms_medium: str, num_tokens: int) -> None:
    """Masked rows must contribute nothing on both optimized exchange paths."""
    _assert_neutral(_launch(num_tokens, comms_medium))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs to run this test")
@skip_pre_blackwell
def test_zero_kv_rows_are_neutral_under_cuda_graph() -> None:
    """The fifo-v2 sender must preserve the contract under graph replay."""
    _assert_neutral(_launch(96, "fifo_v2", use_cuda_graph=True))


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs 2 GPUs to run this test")
@skip_pre_blackwell
def test_zero_kv_negative_control() -> None:
    """Withholding the mask must expose the poisoned rows as NaN."""
    results = _launch(17, "fifo_v2", withhold_mask=True)
    assert any(nan_count > 0 for nan_count, _ in results), (
        "no NaN with the mask withheld, so the positive tests cannot detect a missing sanitize"
    )
